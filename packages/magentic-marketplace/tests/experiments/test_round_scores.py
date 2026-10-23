"""Tests for per-round scoring tables (tests/features/measurement.feature).

Covers `score_customer_round`, `compute_round_tables`, `write_round_tables` and
`run_score`. Runs are built as real SQLite databases holding agent, action and log rows
shaped like a real run (profiles from `data/mexican_3_9`), so no Postgres or LLM is
needed. S1 parity compares against the stock `MarketplaceAnalytics`.
"""

import csv
import itertools
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from pydantic import BaseModel

from magentic_marketplace.experiments import round_scores
from magentic_marketplace.experiments.budgets import (
    BudgetAssignmentLog,
    CustomerBudget,
)
from magentic_marketplace.experiments.round_scores import (
    CUSTOMER_ROUNDS_FILENAME,
    SELLER_ROUNDS_FILENAME,
    CustomerRoundRow,
    SellerRoundRow,
    compute_round_tables,
    run_score,
    score_customer_round,
    write_round_tables,
)
from magentic_marketplace.experiments.run_analytics import MarketplaceAnalytics
from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.actions import FetchMessages, SendMessage
from magentic_marketplace.marketplace.actions.messaging import (
    OrderItem,
    OrderProposal,
    Payment,
)
from magentic_marketplace.marketplace.agents.business.pricing import PriceListLog
from magentic_marketplace.marketplace.agents.customer.models import RefusedPaymentLog
from magentic_marketplace.marketplace.shared.models import (
    BusinessAgentProfile,
    CustomerAgentProfile,
)
from magentic_marketplace.platform.database.base import BaseDatabaseController
from magentic_marketplace.platform.database.models import (
    ActionRow,
    ActionRowData,
    AgentRow,
    LogRow,
)
from magentic_marketplace.platform.database.sqlite import connect_to_sqlite_database
from magentic_marketplace.platform.shared.models import (
    ActionExecutionRequest,
    ActionExecutionResult,
    BaseAction,
    Log,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
CUSTOMERS = {c.id: c for c in load_customers_from_yaml(MEXICAN_3_9 / "customers")}
BUSINESSES = {b.id: b for b in load_businesses_from_yaml(MEXICAN_3_9 / "businesses")}
CUSTOMER_IDS = sorted(CUSTOMERS)

AGUA = "Pineapple Jalapeno Agua Fresca"
EMPANADAS = "Savory Pumpkin Empanadas"
# customer_0001 wants AGUA + EMPANADAS ($3.99 + $9.49) and Outdoor Seating.
# business_0001 serves both and has Outdoor Seating; business_0003 serves both but has none.
SUSAN = "customer_0001"
MATCHING_SELLER = "business_0001"
NO_SEATING_SELLER = "business_0003"
SUSAN_V = 26.96
TOTAL_V = round(
    sum(2 * sum(c.menu_features.values()) for c in CUSTOMERS.values()), 2
)  # 118.48

CUSTOMER_COLUMNS = [
    "round",
    "customer",
    "reference",
    "value",
    "m",
    "budget",
    "seller_paid",
    "price_paid",
    "needs_met",
    "customer_share",
    "seller_share",
    "lost",
    "outcome",
    "refused_payments",
]
SELLER_COLUMNS = ["round", "seller", "item", "price", "units_sold", "revenue"]


# Run builder: writes rows shaped like a real run into a SQLite database


class RunBuilder:
    """Writes agent, action and log rows of a mexican_3_9 run into a database."""

    def __init__(self, database: BaseDatabaseController, tag_rounds: bool = True):
        """Wrap `database`; with `tag_rounds` False, actions carry no round tag (stock)."""
        self.db = database
        self.tag_rounds = tag_rounds
        self._clock = itertools.count()

    def _now(self) -> datetime:
        return datetime(2026, 10, 23, tzinfo=UTC) + timedelta(seconds=next(self._clock))

    async def add_agents(self) -> None:
        """Register every customer and business of mexican_3_9."""
        for customer in CUSTOMERS.values():
            profile = CustomerAgentProfile.from_customer(customer)
            await self.db.agents.create(
                AgentRow(id=profile.id, created_at=self._now(), data=profile)
            )
        for business in BUSINESSES.values():
            profile = BusinessAgentProfile.from_business(business)
            await self.db.agents.create(
                AgentRow(id=profile.id, created_at=self._now(), data=profile)
            )

    async def action(
        self, agent_id: str, action: BaseAction, round_number: int | None
    ) -> None:
        """Write one successful action row, tagged with its round when tagging is on."""
        metadata = (
            {"round": round_number}
            if self.tag_rounds and round_number is not None
            else {}
        )
        await self.db.actions.create(
            ActionRow(
                id=str(uuid.uuid4()),
                created_at=self._now(),
                data=ActionRowData(
                    agent_id=agent_id,
                    request=ActionExecutionRequest(
                        name=action.get_name(),
                        parameters=action.model_dump(mode="json"),
                        metadata=metadata,
                    ),
                    result=ActionExecutionResult(content={}, is_error=False),
                ),
            )
        )

    async def send(
        self,
        round_number: int | None,
        from_id: str,
        to_id: str,
        message: OrderProposal | Payment,
    ) -> None:
        """Write a SendMessage action."""
        await self.action(
            from_id,
            SendMessage(
                from_agent_id=from_id,
                to_agent_id=to_id,
                created_at=self._now(),
                message=message,
            ),
            round_number,
        )

    async def propose(
        self,
        round_number: int,
        business_id: str,
        customer_id: str,
        items: dict[str, tuple[int, float]],
        n: int = 1,
    ) -> OrderProposal:
        """Send an order proposal of `items` (name -> (quantity, unit price)); return it."""
        proposal_id = (
            f"{business_id}_{customer_id}_{n}"
            if round_number == 1
            else f"{business_id}_{customer_id}_r{round_number}_{n}"
        )
        proposal = OrderProposal(
            id=proposal_id,
            items=[
                OrderItem(
                    id=f"item-{i}", item_name=name, quantity=qty, unit_price=price
                )
                for i, (name, (qty, price)) in enumerate(items.items())
            ],
            total_price=round(sum(qty * price for qty, price in items.values()), 2),
        )
        await self.send(round_number, business_id, customer_id, proposal)
        return proposal

    async def pay(
        self, round_number: int, customer_id: str, business_id: str, proposal_id: str
    ) -> None:
        """Send a payment for `proposal_id`."""
        await self.send(
            round_number,
            customer_id,
            business_id,
            Payment(proposal_message_id=proposal_id, payment_method="credit_card"),
        )

    async def buy(
        self,
        round_number: int,
        business_id: str,
        customer_id: str,
        items: dict[str, tuple[int, float]],
        n: int = 1,
    ) -> OrderProposal:
        """Propose `items` and pay for them."""
        proposal = await self.propose(round_number, business_id, customer_id, items, n)
        await self.pay(round_number, customer_id, business_id, proposal.id)
        return proposal

    async def fetch(self, round_number: int, agent_id: str) -> None:
        """Write a FetchMessages action (activity that is not a sale)."""
        await self.action(agent_id, FetchMessages(), round_number)

    async def log(
        self,
        data: BaseModel,
        *,
        level: str = "info",
        name: str = "experiment",
        round_number: int | None = None,
    ) -> None:
        """Write one log row whose `data` is `data` as stored by the server (a dict)."""
        metadata: dict[str, Any] = {"agent_id": name}
        if round_number is not None:
            metadata["round"] = round_number
        await self.db.logs.create(
            LogRow(
                id=str(uuid.uuid4()),
                created_at=self._now(),
                data=Log(
                    level=level,  # type: ignore[arg-type]
                    name=name,
                    message="test log",
                    data=data.model_dump(mode="json"),
                    metadata=metadata,
                ),
            )
        )


@pytest_asyncio.fixture
async def database(tmp_path) -> AsyncGenerator[BaseDatabaseController]:
    """Return an empty SQLite database in tmp_path."""
    async with connect_to_sqlite_database(str(tmp_path / "run.db")) as db:
        yield db


@pytest_asyncio.fixture
async def run(database) -> RunBuilder:
    """Return a round-tagging run builder with every mexican_3_9 agent registered."""
    builder = RunBuilder(database)
    await builder.add_agents()
    return builder


def matching_order(price_agua: float = 5.00, price_emp: float = 10.00) -> dict:
    """Return customer_0001's requested order, by default totalling $15.00."""
    return {AGUA: (1, price_agua), EMPANADAS: (1, price_emp)}


def customer_row(
    rows: list[CustomerRoundRow], round_number: int, customer: str
) -> CustomerRoundRow:
    """Return the one row for (round, customer)."""
    matches = [r for r in rows if r.round == round_number and r.customer == customer]
    assert len(matches) == 1, f"expected one row for {customer} round {round_number}"
    return matches[0]


def seller_row(
    rows: list[SellerRoundRow], round_number: int, seller: str, item: str
) -> SellerRoundRow:
    """Return the one row for (round, seller, item)."""
    matches = [
        r
        for r in rows
        if r.round == round_number and r.seller == seller and r.item == item
    ]
    assert len(matches) == 1, (
        f"expected one row for {seller} {item} round {round_number}"
    )
    return matches[0]


async def round_1_activity(run: RunBuilder) -> None:
    """Give round 1 some activity so the run has a round 1."""
    for customer_id in CUSTOMER_IDS:
        await run.fetch(1, customer_id)


# Rule: Each customer-round is scored with the stock utility rule (unit level)


def test_score_matching_sale_splits_value():
    """Scenario: A matching sale splits V between customer and seller (score function)."""
    score = score_customer_round(SUSAN_V, [(15.00, True)])

    assert score.customer_share == 11.96
    assert score.seller_share == 15.00
    assert score.lost == 0.00
    assert score.outcome == "sale"


def test_score_mismatched_sale_costs_the_price():
    """Scenario: A mismatched sale costs the customer the price (score function)."""
    score = score_customer_round(SUSAN_V, [(15.00, False)])

    assert score.customer_share == -15.00
    assert score.seller_share == 15.00
    assert score.lost == 26.96
    assert score.outcome == "mismatch"


def test_score_no_sale_loses_value():
    """Scenario: No sale loses V (score function)."""
    score = score_customer_round(SUSAN_V, [])

    assert score.customer_share == 0.00
    assert score.seller_share == 0.00
    assert score.lost == 26.96
    assert score.outcome == "no sale"


def test_score_two_payments_count_value_once_and_both_prices():
    """Scenario: Two payments, one matching, count V once and both prices (score function)."""
    score = score_customer_round(SUSAN_V, [(15.00, True), (5.00, False)])

    assert score.customer_share == 6.96
    assert score.seller_share == 20.00
    assert score.lost == 0.00
    assert score.outcome == "sale"


def test_score_two_matching_payments_count_value_once():
    """Scenario: Two payments, one matching (detail 1: V counts once even if both match)."""
    score = score_customer_round(SUSAN_V, [(15.00, True), (12.00, True)])

    assert score.customer_share == -0.04
    assert score.seller_share == 27.00
    assert score.lost == 0.00
    assert score.outcome == "sale"


def test_score_two_mismatched_payments_are_a_mismatch():
    """Scenario: A mismatched sale costs the customer the price (two mismatched payments)."""
    score = score_customer_round(SUSAN_V, [(15.00, False), (5.00, False)])

    assert score.customer_share == -20.00
    assert score.seller_share == 20.00
    assert score.lost == 26.96
    assert score.outcome == "mismatch"


@pytest.mark.parametrize(
    ("value", "payments"),
    [
        (26.96, [(15.00, True)]),
        (26.96, [(15.00, False)]),
        (26.96, []),
        (26.96, [(15.00, True), (5.00, False)]),
        (0.30, [(0.10, True), (0.20, False)]),
        (18.58, [(9.67, False), (0.01, True), (100.33, False)]),
    ],
)
def test_score_shares_add_up_to_value_and_are_cents(value, payments):
    """Rule: Customer share, seller share and lost add up to total V (score function, detail 1)."""
    score = score_customer_round(value, payments)

    assert round(score.customer_share + score.seller_share + score.lost, 2) == value
    for amount in (score.customer_share, score.seller_share, score.lost):
        assert amount == round(amount, 2)


def test_score_rounds_float_noise_to_cents():
    """Detail 1: values are rounded to cents (0.3 - 0.1 - 0.2 is exactly 0.0)."""
    score = score_customer_round(0.30, [(0.10, True), (0.20, False)])

    assert score.customer_share == 0.0
    assert score.seller_share == 0.30
    assert score.lost == 0.0


# Rule: Each customer-round is scored with the stock utility rule (from a run)


async def test_matching_sale_row(run: RunBuilder):
    """Scenario: A matching sale splits V between customer and seller."""
    # Given customer "customer_0001" with reference $13.48, so V = $26.96
    await round_1_activity(run)
    # When in round 2 it pays $15.00 for an order that meets its request
    await run.buy(2, MATCHING_SELLER, SUSAN, matching_order())

    customer_rows, _ = await compute_round_tables(run.db)

    # Then its round-2 row shows customer share $11.96, seller share $15.00,
    # lost $0.00, outcome "sale"
    row = customer_row(customer_rows, 2, SUSAN)
    assert row.reference == 13.48
    assert row.value == 26.96
    assert row.customer_share == 11.96
    assert row.seller_share == 15.00
    assert row.lost == 0.00
    assert row.outcome == "sale"
    assert row.needs_met is True
    assert row.seller_paid == MATCHING_SELLER
    assert row.price_paid == 15.00
    assert row.refused_payments == 0


async def test_mismatched_sale_row(run: RunBuilder):
    """Scenario: A mismatched sale costs the customer the price."""
    # Given customer "customer_0001" with V = $26.96
    await round_1_activity(run)
    # When in round 2 it pays $15.00 to a seller missing a required amenity
    await run.buy(2, NO_SEATING_SELLER, SUSAN, matching_order())

    customer_rows, _ = await compute_round_tables(run.db)

    # Then its round-2 row shows customer share -$15.00, seller share $15.00,
    # lost $26.96, outcome "mismatch"
    row = customer_row(customer_rows, 2, SUSAN)
    assert row.customer_share == -15.00
    assert row.seller_share == 15.00
    assert row.lost == 26.96
    assert row.outcome == "mismatch"
    assert row.needs_met is False
    assert row.seller_paid == NO_SEATING_SELLER
    assert row.price_paid == 15.00


async def test_order_missing_a_requested_item_is_a_mismatch(run: RunBuilder):
    """Scenario: A mismatched sale costs the customer the price (order lacks a requested item)."""
    await round_1_activity(run)
    await run.buy(2, MATCHING_SELLER, SUSAN, {EMPANADAS: (1, 15.00)})

    customer_rows, _ = await compute_round_tables(run.db)

    row = customer_row(customer_rows, 2, SUSAN)
    assert row.outcome == "mismatch"
    assert row.customer_share == -15.00
    assert row.lost == 26.96


async def test_item_name_off_the_menu_is_a_mismatch(run: RunBuilder):
    """Detail 3: items not on the seller's menu are dropped (fuzzy distance 0), so a misspelling misses."""
    await round_1_activity(run)
    await run.buy(
        2,
        MATCHING_SELLER,
        SUSAN,
        {"Pineapple Jalapeno Agua Fresc": (1, 5.00), EMPANADAS: (1, 10.00)},
    )

    customer_rows, _ = await compute_round_tables(run.db)

    row = customer_row(customer_rows, 2, SUSAN)
    assert row.outcome == "mismatch"
    assert row.needs_met is False
    assert row.customer_share == -15.00


async def test_no_sale_row(run: RunBuilder):
    """Scenario: No sale loses V."""
    # Given customer "customer_0001" with V = $26.96 (it buys in round 1 only)
    await run.buy(1, MATCHING_SELLER, SUSAN, matching_order())
    # When in round 2 it pays nothing (it only checks messages)
    await run.fetch(2, SUSAN)

    customer_rows, _ = await compute_round_tables(run.db)

    # Then its round-2 row shows customer share $0.00, seller share $0.00,
    # lost $26.96, outcome "no sale"
    row = customer_row(customer_rows, 2, SUSAN)
    assert row.customer_share == 0.00
    assert row.seller_share == 0.00
    assert row.lost == 26.96
    assert row.outcome == "no sale"
    assert row.needs_met is False
    assert row.seller_paid == ""
    assert row.price_paid == 0.00
    # And round 1 is scored on its own
    assert customer_row(customer_rows, 1, SUSAN).outcome == "sale"


async def test_two_payments_one_matching_row(run: RunBuilder):
    """Scenario: Two payments, one matching, count V once and both prices."""
    # Given customer "customer_0001" with V = $26.96
    await round_1_activity(run)
    # When in round 2 it pays $15.00 for a matching order and $5.00 for a mismatched one
    await run.buy(2, MATCHING_SELLER, SUSAN, matching_order())
    await run.buy(2, NO_SEATING_SELLER, SUSAN, {AGUA: (1, 5.00)})

    customer_rows, _ = await compute_round_tables(run.db)

    # Then its round-2 row shows customer share $6.96, seller share $20.00, lost $0.00
    row = customer_row(customer_rows, 2, SUSAN)
    assert row.customer_share == 6.96
    assert row.seller_share == 20.00
    assert row.lost == 0.00
    assert row.outcome == "sale"
    assert row.price_paid == 20.00
    assert set(row.seller_paid.split(";")) == {MATCHING_SELLER, NO_SEATING_SELLER}


async def test_refused_over_budget_payment_is_counted(run: RunBuilder):
    """Scenario: Refused over-budget payments are counted."""
    # Given customer "customer_0001" with a budget of $16.85
    budgets = [
        CustomerBudget(customer_id=SUSAN, m=1.25, budget=16.85),
        CustomerBudget(customer_id="customer_0002", m=1.5, budget=13.94),
        CustomerBudget(customer_id="customer_0003", m=2.0, budget=72.94),
    ]
    await run.log(BudgetAssignmentLog(budgets=budgets))
    await round_1_activity(run)
    # When in round 2 one payment of $18.00 is refused and no other payment is made
    proposal = await run.propose(
        2, MATCHING_SELLER, SUSAN, {AGUA: (1, 8.00), EMPANADAS: (1, 10.00)}
    )
    await run.log(
        RefusedPaymentLog(
            customer_id=SUSAN,
            business_id=MATCHING_SELLER,
            proposal_id=proposal.id,
            total=18.00,
            budget=16.85,
            round=2,
        ),
        level="warning",
        name=SUSAN,
        round_number=2,
    )

    customer_rows, _ = await compute_round_tables(run.db)

    # Then its round-2 row shows outcome "no sale" and 1 refused payment
    row = customer_row(customer_rows, 2, SUSAN)
    assert row.outcome == "no sale"
    assert row.refused_payments == 1
    assert row.budget == 16.85
    assert row.m == 1.25
    # And the refusal is counted in round 2 only, for that customer only
    assert customer_row(customer_rows, 1, SUSAN).refused_payments == 0
    assert customer_row(customer_rows, 2, "customer_0002").refused_payments == 0
    # And every customer's row carries its own m and budget, in every round
    for r in (1, 2):
        assert customer_row(customer_rows, r, "customer_0003").budget == 72.94
        assert customer_row(customer_rows, r, "customer_0003").m == 2.0


async def test_two_refusals_in_a_round_count_two(run: RunBuilder):
    """Scenario: Refused over-budget payments are counted (two refusals in one round)."""
    await round_1_activity(run)
    # A refusal follows the customer's own round-2 actions, so round 2 has activity
    await run.fetch(2, SUSAN)
    for n in (1, 2):
        await run.log(
            RefusedPaymentLog(
                customer_id=SUSAN,
                business_id=MATCHING_SELLER,
                proposal_id=f"{MATCHING_SELLER}_{SUSAN}_r2_{n}",
                total=18.00 + n,
                budget=16.85,
                round=2,
            ),
            level="warning",
            name=SUSAN,
            round_number=2,
        )

    customer_rows, _ = await compute_round_tables(run.db)

    assert customer_row(customer_rows, 2, SUSAN).refused_payments == 2


async def test_run_without_budgets_has_no_m_or_budget(run: RunBuilder):
    """Detail 4: m and budget are None when the run had no budgets."""
    await round_1_activity(run)

    customer_rows, _ = await compute_round_tables(run.db)

    assert customer_rows
    assert all(row.m is None and row.budget is None for row in customer_rows)


async def test_payment_matched_to_proposal_of_its_own_round(run: RunBuilder):
    """Detail 4: a payment's amount is the total of its proposal sent in the same round."""
    await run.buy(1, MATCHING_SELLER, SUSAN, matching_order(5.00, 15.00))
    await run.buy(2, MATCHING_SELLER, SUSAN, matching_order(5.00, 10.00))

    customer_rows, _ = await compute_round_tables(run.db)

    assert customer_row(customer_rows, 1, SUSAN).price_paid == 20.00
    assert customer_row(customer_rows, 1, SUSAN).customer_share == 6.96
    assert customer_row(customer_rows, 2, SUSAN).price_paid == 15.00
    assert customer_row(customer_rows, 2, SUSAN).customer_share == 11.96


async def test_every_customer_has_a_row_in_every_active_round(run: RunBuilder):
    """Detail 4: every customer gets a row in every round that has any tagged action."""
    await run.fetch(1, SUSAN)
    await run.fetch(2, "customer_0002")
    await run.fetch(3, MATCHING_SELLER)

    customer_rows, _ = await compute_round_tables(run.db)

    assert sorted((r.round, r.customer) for r in customer_rows) == [
        (r, c) for r in (1, 2, 3) for c in CUSTOMER_IDS
    ]


async def test_untagged_actions_count_as_round_1(database):
    """Detail 4: a missing round tag counts as round 1, so a stock run scores as one round."""
    run = RunBuilder(database, tag_rounds=False)
    await run.add_agents()
    await run.buy(1, MATCHING_SELLER, SUSAN, matching_order())

    customer_rows, seller_rows = await compute_round_tables(run.db)

    assert sorted((r.round, r.customer) for r in customer_rows) == [
        (1, c) for c in CUSTOMER_IDS
    ]
    assert customer_row(customer_rows, 1, SUSAN).customer_share == 11.96
    assert {r.round for r in seller_rows} == {1}


# Rule: Customer share, seller share and lost add up to total V in every round


async def build_two_round_run(run: RunBuilder) -> None:
    """Write a 2-round mexican_3_9 run covering sale, mismatch, double pay and no sale."""
    # Round 1: customer_0001 matching sale, customer_0002 mismatch, customer_0003 nothing
    await run.buy(1, MATCHING_SELLER, SUSAN, matching_order())
    await run.buy(1, "business_0004", "customer_0002", {"Churros": (2, 4.25)})
    await run.fetch(1, "customer_0003")
    # Round 2: customer_0001 pays twice, customer_0002 nothing, customer_0003 mismatch
    await run.buy(2, MATCHING_SELLER, SUSAN, matching_order())
    await run.buy(2, NO_SEATING_SELLER, SUSAN, {AGUA: (1, 5.03)})
    await run.fetch(2, "customer_0002")
    await run.buy(
        2, "business_0005", "customer_0003", {"Southwest Chicken Enchiladas": (1, 14.0)}
    )


async def test_round_totals_balance(run: RunBuilder):
    """Scenario: The round totals balance."""
    # Given a finished run of "mexican_3_9" with 2 rounds
    await build_two_round_run(run)

    # When the round totals are computed
    customer_rows, _ = await compute_round_tables(run.db)

    # Then in each round customer share + seller share + lost equals the sum of
    # every customer's V
    for r in (1, 2):
        rows = [row for row in customer_rows if row.round == r]
        assert sorted(row.customer for row in rows) == CUSTOMER_IDS
        total = sum(row.customer_share + row.seller_share + row.lost for row in rows)
        assert round(total, 2) == TOTAL_V
        assert round(sum(row.value for row in rows), 2) == TOTAL_V


async def test_every_row_balances_to_its_value(run: RunBuilder):
    """Rule: Customer share, seller share and lost add up to V (per row)."""
    await build_two_round_run(run)

    customer_rows, _ = await compute_round_tables(run.db)

    for row in customer_rows:
        assert round(row.customer_share + row.seller_share + row.lost, 2) == row.value
        assert row.value == round(2 * row.reference, 2)
        assert row.seller_share == row.price_paid


# Rule: Every customer-round row carries the inputs needed to read it


def test_customer_row_model_has_the_scenario_columns():
    """Scenario: The customer-round table columns (model field order)."""
    assert list(CustomerRoundRow.model_fields) == CUSTOMER_COLUMNS


async def test_customer_round_table_columns(run: RunBuilder, tmp_path):
    """Scenario: The customer-round table columns."""
    # Given a finished run
    await build_two_round_run(run)
    customer_rows, seller_rows = await compute_round_tables(run.db)
    folder = tmp_path / "run_folder"
    folder.mkdir()

    # When the customer-round table is written to the run folder
    customer_path, seller_path = write_round_tables(customer_rows, seller_rows, folder)

    # Then each row has round, customer, reference, V, m, budget, seller paid, price
    # paid, needs met, customer share, seller share, lost, outcome and refused payments
    assert customer_path == folder / CUSTOMER_ROUNDS_FILENAME
    assert seller_path == folder / SELLER_ROUNDS_FILENAME
    with customer_path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CUSTOMER_COLUMNS
        rows = list(reader)
    assert len(rows) == 2 * len(CUSTOMER_IDS)
    for row in rows:
        assert all(row[column] is not None for column in CUSTOMER_COLUMNS)
    susan_r2 = next(r for r in rows if r["round"] == "2" and r["customer"] == SUSAN)
    assert float(susan_r2["reference"]) == 13.48
    assert float(susan_r2["value"]) == 26.96
    assert float(susan_r2["price_paid"]) == 20.03
    assert float(susan_r2["customer_share"]) == 6.93
    assert float(susan_r2["seller_share"]) == 20.03
    assert float(susan_r2["lost"]) == 0.0
    assert susan_r2["outcome"] == "sale"
    assert int(susan_r2["refused_payments"]) == 0


def test_write_round_tables_sorts_rows_and_returns_paths(tmp_path):
    """Detail 6: CSVs with a header in field order, rows sorted by (round, customer) and (round, seller, item)."""
    customer_rows = [
        CustomerRoundRow(
            round=r,
            customer=c,
            reference=13.48,
            value=26.96,
            m=None,
            budget=None,
            seller_paid="",
            price_paid=0.0,
            needs_met=False,
            customer_share=0.0,
            seller_share=0.0,
            lost=26.96,
            outcome="no sale",
            refused_payments=0,
        )
        for r, c in [
            (2, "customer_0002"),
            (1, "customer_0003"),
            (2, "customer_0001"),
            (1, "customer_0001"),
        ]
    ]
    seller_rows = [
        SellerRoundRow(round=r, seller=s, item=i, price=1.0, units_sold=0, revenue=0.0)
        for r, s, i in [
            (2, "business_0001", "B"),
            (1, "business_0002", "A"),
            (2, "business_0001", "A"),
            (1, "business_0001", "Z"),
        ]
    ]

    customer_path, seller_path = write_round_tables(
        customer_rows, seller_rows, tmp_path
    )

    assert (customer_path, seller_path) == (
        tmp_path / "customer_rounds.csv",
        tmp_path / "seller_rounds.csv",
    )
    with customer_path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CUSTOMER_COLUMNS
        assert [(r["round"], r["customer"]) for r in reader] == [
            ("1", "customer_0001"),
            ("1", "customer_0003"),
            ("2", "customer_0001"),
            ("2", "customer_0002"),
        ]
    with seller_path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == SELLER_COLUMNS
        assert [(r["round"], r["seller"], r["item"]) for r in reader] == [
            ("1", "business_0001", "Z"),
            ("1", "business_0002", "A"),
            ("2", "business_0001", "A"),
            ("2", "business_0001", "B"),
        ]


# Rule: Seller prices and sales are tabled per seller per round


async def build_empanada_round(run: RunBuilder) -> None:
    """business_0001 prices Empanadas at $10.50 in round 2 and sells 2 of them."""
    await round_1_activity(run)
    round_2_prices = dict(BUSINESSES[MATCHING_SELLER].menu_features)
    round_2_prices[EMPANADAS] = 10.50
    await run.log(
        PriceListLog(seller_id=MATCHING_SELLER, round=2, prices=round_2_prices),
        name=MATCHING_SELLER,
        round_number=2,
    )
    await run.buy(2, MATCHING_SELLER, SUSAN, {AGUA: (1, 2.73), EMPANADAS: (1, 10.50)})
    await run.buy(2, MATCHING_SELLER, "customer_0003", {EMPANADAS: (1, 10.50)})
    # An unpaid proposal does not count as sold
    await run.propose(2, MATCHING_SELLER, "customer_0002", {EMPANADAS: (3, 10.50)})


async def test_seller_round_table(run: RunBuilder, tmp_path):
    """Scenario: The seller-round table."""
    # Given seller "business_0001" priced "Empanadas" at $10.50 and sold 2 in round 2
    await build_empanada_round(run)
    customer_rows, seller_rows = await compute_round_tables(run.db)

    # When the seller-round table is written to the run folder
    _, seller_path = write_round_tables(customer_rows, seller_rows, tmp_path)

    # Then it has a row for "business_0001", round 2, "Empanadas", price $10.50,
    # units sold 2, revenue $21.00
    with seller_path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == SELLER_COLUMNS
        rows = [
            r
            for r in reader
            if r["seller"] == MATCHING_SELLER
            and r["round"] == "2"
            and r["item"] == EMPANADAS
        ]
    assert len(rows) == 1
    assert float(rows[0]["price"]) == 10.50
    assert int(rows[0]["units_sold"]) == 2
    assert float(rows[0]["revenue"]) == 21.00


async def test_seller_round_rows_from_run(run: RunBuilder):
    """Scenario: The seller-round table (computed rows)."""
    await build_empanada_round(run)

    _, seller_rows = await compute_round_tables(run.db)

    row = seller_row(seller_rows, 2, MATCHING_SELLER, EMPANADAS)
    assert (row.price, row.units_sold, row.revenue) == (10.50, 2, 21.00)
    agua = seller_row(seller_rows, 2, MATCHING_SELLER, AGUA)
    assert (agua.price, agua.units_sold, agua.revenue) == (2.73, 1, 2.73)


async def test_seller_rows_cover_every_menu_item_every_round(run: RunBuilder):
    """Detail 5: one row per (round, seller, menu item) for every round in the run."""
    await build_empanada_round(run)

    _, seller_rows = await compute_round_tables(run.db)

    expected = sorted(
        (r, b.id, item)
        for r in (1, 2)
        for b in BUSINESSES.values()
        for item in b.menu_features
    )
    assert sorted((r.round, r.seller, r.item) for r in seller_rows) == expected


async def test_seller_price_is_menu_price_without_price_list(run: RunBuilder):
    """Detail 5: with no price_list entry (pricing off), the price is the menu price."""
    await build_empanada_round(run)

    _, seller_rows = await compute_round_tables(run.db)

    # Round 1 has no price list for business_0001
    row = seller_row(seller_rows, 1, MATCHING_SELLER, EMPANADAS)
    assert row.price == BUSINESSES[MATCHING_SELLER].menu_features[EMPANADAS]
    assert (row.units_sold, row.revenue) == (0, 0.0)
    # business_0002 never logged a price list
    other = seller_row(seller_rows, 2, "business_0002", EMPANADAS)
    assert other.price == BUSINESSES["business_0002"].menu_features[EMPANADAS]


async def test_seller_revenue_uses_proposal_unit_prices(run: RunBuilder):
    """Detail 5: revenue is the sum of unit_price x quantity of paid order items, rounded to cents."""
    await round_1_activity(run)
    await run.log(
        PriceListLog(
            seller_id=MATCHING_SELLER,
            round=2,
            prices={
                **BUSINESSES[MATCHING_SELLER].menu_features,
                EMPANADAS: 10.50,
            },
        ),
        name=MATCHING_SELLER,
        round_number=2,
    )
    await run.buy(2, MATCHING_SELLER, SUSAN, {EMPANADAS: (3, 9.99)})

    _, seller_rows = await compute_round_tables(run.db)

    row = seller_row(seller_rows, 2, MATCHING_SELLER, EMPANADAS)
    assert row.price == 10.50
    assert row.units_sold == 3
    assert row.revenue == 29.97


# Rule: With all features off, the scores equal stock analytics


async def build_s1_run(run: RunBuilder) -> None:
    """Write a one-round run with every feature off: no budgets, no price lists."""
    await run.add_agents()
    # customer_0001: a matching sale plus a mismatched extra order
    await run.buy(1, MATCHING_SELLER, SUSAN, matching_order())
    await run.buy(1, NO_SEATING_SELLER, SUSAN, {AGUA: (1, 5.03)})
    # customer_0002: a mismatched sale, and a proposal it never pays
    await run.buy(1, "business_0004", "customer_0002", {"Churros": (2, 4.25)})
    await run.propose(1, "business_0005", "customer_0002", {"Churros": (1, 3.00)})
    # customer_0003: no payment
    await run.fetch(1, "customer_0003")


@pytest.mark.parametrize("tag_rounds", [False, True], ids=["untagged", "round1"])
async def test_s1_scores_match_stock_analyze(database, tag_rounds):
    """Scenario: S1 scores match the stock analyze report."""
    # Given a finished one-round run with every feature off
    run = RunBuilder(database, tag_rounds=tag_rounds)
    await build_s1_run(run)

    # When the customer-round table is written
    customer_rows, _ = await compute_round_tables(database)

    # Then each customer's customer share equals its utility in the stock "analyze" report
    analytics = MarketplaceAnalytics(database)
    await analytics.load_data()
    await analytics.analyze_actions()
    assert sorted((r.round, r.customer) for r in customer_rows) == [
        (1, c) for c in CUSTOMER_IDS
    ]
    for row in customer_rows:
        utility, needs_met = analytics.calculate_customer_utility(row.customer)
        assert row.customer_share == utility, row.customer
        assert row.needs_met == needs_met, row.customer
        assert row.m is None and row.budget is None
        assert row.refused_payments == 0


# run_score: Postgres connection and output to the current directory


async def test_run_score_reads_schema_and_writes_csvs_to_cwd(
    database, monkeypatch, tmp_path
):
    """Detail 7: run_score connects to the schema like run_analytics and writes both CSVs to the cwd."""
    run = RunBuilder(database)
    await run.add_agents()
    await run.buy(1, MATCHING_SELLER, SUSAN, matching_order())
    connections: list[dict] = []

    @asynccontextmanager
    async def fake_connect(**kwargs):
        connections.append(kwargs)
        yield database

    # Patch every place the connector can be imported from
    import magentic_marketplace.platform.database as platform_database
    import magentic_marketplace.platform.database.postgresql as postgresql

    monkeypatch.setattr(
        round_scores, "connect_to_postgresql_database", fake_connect, raising=False
    )
    monkeypatch.setattr(
        platform_database, "connect_to_postgresql_database", fake_connect
    )
    monkeypatch.setattr(postgresql, "connect_to_postgresql_database", fake_connect)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.chdir(out)

    customer_path, seller_path = await run_score(
        "scores_test",
        postgres_host="db.local",
        postgres_port=6543,
        postgres_password="secret",
    )

    assert len(connections) == 1
    assert connections[0]["schema"] == "scores_test"
    assert connections[0]["host"] == "db.local"
    assert connections[0]["port"] == 6543
    assert connections[0]["password"] == "secret"
    assert connections[0]["mode"] == "existing"
    assert Path(customer_path).resolve() == (out / CUSTOMER_ROUNDS_FILENAME).resolve()
    assert Path(seller_path).resolve() == (out / SELLER_ROUNDS_FILENAME).resolve()
    with (out / CUSTOMER_ROUNDS_FILENAME).open(newline="") as f:
        rows = list(csv.DictReader(f))
    susan = next(r for r in rows if r["customer"] == SUSAN)
    assert float(susan["customer_share"]) == 11.96
    assert (out / SELLER_ROUNDS_FILENAME).is_file()

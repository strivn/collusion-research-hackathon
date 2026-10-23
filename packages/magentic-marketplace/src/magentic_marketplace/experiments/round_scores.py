"""Per-round scoring: each customer-round split into customer share, seller share and lost.

Applies the stock utility rule (`run_analytics.MarketplaceAnalytics`) to each round
separately, and tables every seller's prices and sales per round.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from magentic_marketplace.experiments.budgets import (
    BudgetAssignmentLog,
    reference_price,
)
from magentic_marketplace.experiments.run_analytics import MarketplaceAnalytics
from magentic_marketplace.marketplace.actions import SendMessage
from magentic_marketplace.marketplace.actions.messaging import (
    OrderProposal,
    Payment,
)
from magentic_marketplace.marketplace.agents.business.pricing import PriceListLog
from magentic_marketplace.marketplace.agents.customer.models import RefusedPaymentLog
from magentic_marketplace.platform.database import connect_to_postgresql_database
from magentic_marketplace.platform.database.base import BaseDatabaseController
from magentic_marketplace.platform.database.queries.logs import data

CUSTOMER_ROUNDS_FILENAME = "customer_rounds.csv"
SELLER_ROUNDS_FILENAME = "seller_rounds.csv"

Outcome = Literal["sale", "mismatch", "no sale"]


class CustomerRoundRow(BaseModel):
    """One customer's outcome in one round."""

    round: int
    customer: str
    reference: float
    value: float
    m: float | None
    budget: float | None
    seller_paid: str
    price_paid: float
    needs_met: bool
    customer_share: float
    seller_share: float
    lost: float
    outcome: Outcome
    refused_payments: int


class SellerRoundRow(BaseModel):
    """One seller's price and sales of one menu item in one round."""

    round: int
    seller: str
    item: str
    price: float
    units_sold: int
    revenue: float


class CustomerRoundScore(BaseModel):
    """The split of one customer's value V in one round."""

    customer_share: float
    seller_share: float
    lost: float
    outcome: Outcome


def score_customer_round(
    value: float, payments: list[tuple[float, bool]]
) -> CustomerRoundScore:
    """Split `value` given the round's payments, each as (amount paid, order meets the request).

    V counts once if any payment meets the request; every payment is subtracted.
    """
    # Work in whole cents so the three parts add up to `value` exactly
    value_cents = round(value * 100)
    paid_cents = sum(round(amount * 100) for amount, _ in payments)
    met = any(meets for _, meets in payments)
    kept_cents = value_cents if met else 0

    if met:
        outcome: Outcome = "sale"
    elif payments:
        outcome = "mismatch"
    else:
        outcome = "no sale"
    return CustomerRoundScore(
        customer_share=(kept_cents - paid_cents) / 100,
        seller_share=paid_cents / 100,
        lost=(value_cents - kept_cents) / 100,
        outcome=outcome,
    )


async def compute_round_tables(
    database: BaseDatabaseController,
) -> tuple[list[CustomerRoundRow], list[SellerRoundRow]]:
    """Build the customer-round and seller-round tables of a finished run."""
    # Load profiles with the stock analytics so its matching helpers apply unchanged
    analytics = MarketplaceAnalytics(database)
    await analytics.load_data()

    # Proposals keyed by (round, customer, proposal id): a repeated id overwrites, as in stock
    proposals: dict[tuple[int, str, str], tuple[str, OrderProposal]] = {}
    payments: list[tuple[int, str, Payment]] = []
    rounds: set[int] = set()
    for action_row in await database.actions.get_all():
        round_number = action_row.data.request.metadata.get("round", 1)
        rounds.add(round_number)
        if action_row.data.result.is_error:
            continue
        action = action_row.data.request.parameters
        if action.get("type") != "send_message":
            continue
        send = SendMessage.model_validate(action)
        if isinstance(send.message, OrderProposal):
            key = (round_number, send.to_agent_id, send.message.id)
            proposals[key] = (send.from_agent_id, send.message)
        elif isinstance(send.message, Payment):
            payments.append((round_number, send.from_agent_id, send.message))

    # Group the payments that match a proposal sent to that customer in the same round
    paid: dict[tuple[int, str], list[tuple[str, OrderProposal]]] = defaultdict(list)
    for round_number, customer_id, payment in payments:
        match = proposals.get((round_number, customer_id, payment.proposal_message_id))
        if match and match[0] in analytics.business_agents:
            paid[(round_number, customer_id)].append(match)

    # Read the budget assignment, the refused payments and the price lists from the logs
    budget_by_customer: dict[str, tuple[float, float]] = {}
    refused: dict[tuple[int, str], int] = defaultdict(int)
    price_lists: dict[tuple[int, str], dict[str, float]] = {}
    log_query = (
        data(path="type", value="budget_assignment", operator="=")
        | data(path="type", value="refused_payment", operator="=")
        | data(path="type", value="price_list", operator="=")
    )
    for log_row in await database.logs.find(log_query):
        entry = log_row.data.data
        # The query selects only log entries that carry a typed data dict
        assert isinstance(entry, dict)
        kind = entry["type"]
        if kind == "budget_assignment":
            for b in BudgetAssignmentLog.model_validate(entry).budgets:
                budget_by_customer[b.customer_id] = (b.m, b.budget)
        elif kind == "refused_payment":
            refusal = RefusedPaymentLog.model_validate(entry)
            refused[(refusal.round, refusal.customer_id)] += 1
        else:
            price_list = PriceListLog.model_validate(entry)
            price_lists[(price_list.round, price_list.seller_id)] = price_list.prices

    # One row per customer per round
    customer_rows: list[CustomerRoundRow] = []
    for round_number in sorted(rounds):
        for customer_id, profile in sorted(analytics.customer_agents.items()):
            reference = round(reference_price(profile.customer), 2)
            value = round(2 * reference_price(profile.customer), 2)
            purchases = paid[(round_number, customer_id)]
            meets = [
                _meets_request(analytics, customer_id, seller_id, proposal)
                for seller_id, proposal in purchases
            ]
            score = score_customer_round(
                value,
                [
                    (p.total_price, met)
                    for (_, p), met in zip(purchases, meets, strict=False)
                ],
            )
            m, budget = budget_by_customer.get(customer_id, (None, None))
            customer_rows.append(
                CustomerRoundRow(
                    round=round_number,
                    customer=customer_id,
                    reference=reference,
                    value=value,
                    m=m,
                    budget=budget,
                    seller_paid=";".join(seller_id for seller_id, _ in purchases),
                    price_paid=round(sum(p.total_price for _, p in purchases), 2),
                    needs_met=any(meets),
                    customer_share=score.customer_share,
                    seller_share=score.seller_share,
                    lost=score.lost,
                    outcome=score.outcome,
                    refused_payments=refused[(round_number, customer_id)],
                )
            )

    # Units and revenue per seller, round and item from the paid proposals
    units: dict[tuple[int, str, str], int] = defaultdict(int)
    revenue: dict[tuple[int, str, str], float] = defaultdict(float)
    for (round_number, _), purchases in paid.items():
        for seller_id, proposal in purchases:
            for item in proposal.items:
                key = (round_number, seller_id, item.item_name)
                units[key] += item.quantity
                revenue[key] += item.unit_price * item.quantity

    # One row per seller per menu item per round
    seller_rows = [
        SellerRoundRow(
            round=round_number,
            seller=seller_id,
            item=item,
            price=price_lists.get((round_number, seller_id), {}).get(item, menu_price),
            units_sold=units[(round_number, seller_id, item)],
            revenue=round(revenue[(round_number, seller_id, item)], 2),
        )
        for round_number in sorted(rounds)
        for seller_id, profile in sorted(analytics.business_agents.items())
        for item, menu_price in sorted(profile.business.menu_features.items())
    ]
    return customer_rows, seller_rows


def _meets_request(
    analytics: MarketplaceAnalytics,
    customer_id: str,
    seller_id: str,
    proposal: OrderProposal,
) -> bool:
    """Apply the stock rule: proposal items on the menu cover the request and amenities match."""
    on_menu, _ = analytics.filter_valid_proposal_items(seller_id, proposal)
    requested = set(analytics.customer_agents[customer_id].customer.menu_features)
    return requested.issubset(on_menu) and analytics.check_amenity_match(
        customer_id, seller_id
    )


def write_round_tables(
    customer_rows: list[CustomerRoundRow],
    seller_rows: list[SellerRoundRow],
    run_folder: Path,
) -> tuple[Path, Path]:
    """Write both tables as CSV to `run_folder`; return the two paths written."""
    # Header in model field order, rows sorted by their natural key
    customer_path = run_folder / CUSTOMER_ROUNDS_FILENAME
    seller_path = run_folder / SELLER_ROUNDS_FILENAME
    for path, model, rows in (
        (
            customer_path,
            CustomerRoundRow,
            sorted(customer_rows, key=lambda r: (r.round, r.customer)),
        ),
        (
            seller_path,
            SellerRoundRow,
            sorted(seller_rows, key=lambda r: (r.round, r.seller, r.item)),
        ),
    ):
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(model.model_fields))
            writer.writeheader()
            writer.writerows(row.model_dump() for row in rows)
    return customer_path, seller_path


async def run_score(
    experiment_name: str,
    postgres_host: str = "localhost",
    postgres_port: int = 5432,
    postgres_password: str = "postgres",
) -> tuple[Path, Path]:
    """Score a finished run's Postgres schema and write both tables to the current directory."""
    # Connect the way run_analytics does for postgres
    async with connect_to_postgresql_database(
        schema=experiment_name,
        host=postgres_host,
        port=postgres_port,
        password=postgres_password,
        mode="existing",
    ) as database:
        customer_rows, seller_rows = await compute_round_tables(database)
    return write_round_tables(customer_rows, seller_rows, Path.cwd())

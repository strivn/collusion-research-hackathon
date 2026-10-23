"""Tests for seller memory between rounds (tests/features/rounds.feature).

Rule: From round 2, a seller prices with its own notes, its history and rivals' last
prices. Covers `build_seller_memories` and the run wiring: after round r, with pricing
on and another round to come, the run reads the per-round seller table, collects each
seller's notes and hands every round-(r+1) seller its memory.

The run tests call the real `run_marketplace_experiment` with fakes in place of
`CustomerAgent`, `MarketplaceLauncher`, `AgentLauncher` and `compute_round_tables` (as
imported by `run_experiment`), and either a recording fake or a real `BusinessAgent`
with a fake LLM and logger, so no Postgres, server or LLM is needed.
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from magentic_marketplace.experiments import run_experiment
from magentic_marketplace.experiments.round_scores import SellerRoundRow
from magentic_marketplace.experiments.seller_memory import build_seller_memories
from magentic_marketplace.experiments.utils import load_businesses_from_yaml
from magentic_marketplace.marketplace.agents.business.agent import BusinessAgent
from magentic_marketplace.marketplace.agents.business.pricing import (
    ItemPrice,
    ItemSales,
    PriceList,
    RivalPrice,
    RoundHistory,
    SellerMemory,
)
from magentic_marketplace.marketplace.llm.clients.openai import OpenAIConfig
from magentic_marketplace.marketplace.shared.models import Business
from magentic_marketplace.platform.database.models import ActionRow, ActionRowData
from magentic_marketplace.platform.shared.models import (
    ActionExecutionRequest,
    ActionExecutionResult,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
BUSINESSES = {b.id: b for b in load_businesses_from_yaml(MEXICAN_3_9 / "businesses")}

PUMPKIN = "Savory Pumpkin Empanadas"
EMPANADA_NOTES = "test a higher empanada price"
NOTES_HEADER = "Your notes from the previous round:"
HISTORY_HEADER = "Your past rounds:"
RIVALS_HEADER = "Other restaurants' prices in the previous round:"

# Phrases that would tell the seller how many rounds the run has. "last round" is
# allowed only inside the two memory headers, where it means the previous round.
ROUND_COUNT_PATTERNS = [
    re.compile(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+rounds\b", re.I
    ),
    re.compile(r"\bround\s+\d+\s*(of|/)\s*\d+\b", re.I),
    re.compile(r"\b(final|remaining)\s+rounds?\b", re.I),
    re.compile(r"\blast\s+rounds?\b", re.I),
    re.compile(r"\brounds?\s+(left|remaining)\b", re.I),
]


# Helpers for build_seller_memories


def make_business(business_id: str, name: str, menu: dict[str, float]) -> Business:
    """Build a small Business with `menu` as its menu."""
    return Business(
        id=business_id,
        name=name,
        description="Mexican and Tex-Mex classics",
        rating=1.0,
        progenitor_customer="customer_0001",
        menu_features=menu,
        amenity_features={"Outdoor Seating": True},
        min_price_factor=0.78,
    )


def row(
    round_number: int,
    seller: str,
    item: str,
    price: float,
    units_sold: int = 0,
) -> SellerRoundRow:
    """Build one seller-table row; revenue is price x units sold."""
    return SellerRoundRow(
        round=round_number,
        seller=seller,
        item=item,
        price=price,
        units_sold=units_sold,
        revenue=round(price * units_sold, 2),
    )


POBLANO = make_business(
    "business_0001", "Poblano Palate", {"Empanadas": 9.49, "Agua Fresca": 3.0}
)
CASA = make_business(
    "business_0004", "Casa de Sabor", {"Agua Fresca": 3.1, "Empanadas": 9.99}
)
GUAC = make_business("business_0007", "Guac My World", {"Tacos": 4.0})
SOL = make_business("business_0005", "Sol y Sazon", {"Empanadas": 8.5})
TRIO = [POBLANO, CASA, GUAC]


# Scenario: The seller sees its own prices, units sold and revenue per past round


def test_history_lists_own_price_units_and_revenue_per_past_round():
    """Scenario: The seller sees its own prices, units sold and revenue per past round."""
    # Given seller "business_0001" priced "Empanadas" at $9.49 and sold 1 in round 1
    # And priced "Empanadas" at $10.50 and sold 0 in round 2
    rows = [
        row(2, "business_0001", "Agua Fresca", 3.25, 1),
        row(1, "business_0004", "Empanadas", 9.99, 2),
        row(2, "business_0001", "Empanadas", 10.5, 0),
        row(1, "business_0001", "Agua Fresca", 3.0, 2),
        row(1, "business_0001", "Empanadas", 9.49, 1),
    ]

    # When its round-3 memory is built
    memories = build_seller_memories(TRIO, rows, {}, next_round=3)

    # Then it holds round 1 "Empanadas" $9.49, 1 sold, revenue $9.49
    # And round 2 "Empanadas" $10.50, 0 sold, revenue $0.00 (rounds ascending, menu order)
    assert memories["business_0001"].history == [
        RoundHistory(
            round=1,
            items=[
                ItemSales(item="Empanadas", price=9.49, units_sold=1, revenue=9.49),
                ItemSales(item="Agua Fresca", price=3.0, units_sold=2, revenue=6.0),
            ],
        ),
        RoundHistory(
            round=2,
            items=[
                ItemSales(item="Empanadas", price=10.5, units_sold=0, revenue=0.0),
                ItemSales(item="Agua Fresca", price=3.25, units_sold=1, revenue=3.25),
            ],
        ),
    ]


def test_history_holds_only_rounds_before_the_next_round():
    """Scenario: The seller sees its own ... per past round (detail 2: round < next_round)."""
    rows = [
        row(1, "business_0001", "Empanadas", 9.49, 1),
        row(2, "business_0001", "Empanadas", 10.5, 0),
    ]

    memories = build_seller_memories(TRIO, rows, {}, next_round=2)

    assert [h.round for h in memories["business_0001"].history] == [1]


def test_every_seller_gets_a_memory_even_without_rows():
    """Scenario: The seller sees its own ... per past round (detail 1: one memory per seller)."""
    rows = [row(1, "business_0001", "Empanadas", 9.49, 1)]

    memories = build_seller_memories(TRIO, rows, {}, next_round=2)

    assert set(memories) == {"business_0001", "business_0004", "business_0007"}
    assert memories["business_0007"] == SellerMemory(
        notes=None, history=[], rival_prices={}
    )


# Scenario: The seller sees last round's prices of rivals that sell the same item


def test_rivals_are_sellers_of_the_same_item_with_last_rounds_price():
    """Scenario: The seller sees last round's prices of rivals that sell the same item."""
    # Given "business_0001" and "business_0004" both sell "Empanadas"
    # And "business_0004" priced "Empanadas" at $9.99 in round 1
    # And "business_0007" does not sell "Empanadas"
    rows = [
        row(1, "business_0001", "Empanadas", 9.49, 1),
        row(1, "business_0001", "Agua Fresca", 3.0),
        row(1, "business_0004", "Empanadas", 9.99),
        row(1, "business_0004", "Agua Fresca", 3.1),
        row(1, "business_0007", "Tacos", 4.0, 3),
    ]

    # When the round-2 memory of "business_0001" is built
    rivals = build_seller_memories(TRIO, rows, {}, next_round=2)[
        "business_0001"
    ].rival_prices

    # Then it lists "Empanadas" with "business_0004"'s name at $9.99
    # And it does not list "business_0007" under "Empanadas"
    assert rivals == {
        "Empanadas": [
            RivalPrice(
                seller_id="business_0004", seller_name="Casa de Sabor", price=9.99
            )
        ],
        "Agua Fresca": [
            RivalPrice(
                seller_id="business_0004", seller_name="Casa de Sabor", price=3.1
            )
        ],
    }


def test_rivals_are_sorted_by_seller_id_and_never_include_the_seller():
    """Scenario: The seller sees last round's prices of rivals (detail 2: sorted by id, no self)."""
    rows = [
        row(1, "business_0005", "Empanadas", 8.5),
        row(1, "business_0001", "Empanadas", 9.49),
        row(1, "business_0004", "Empanadas", 9.99),
    ]

    memories = build_seller_memories([SOL, CASA, POBLANO], rows, {}, next_round=2)

    assert memories["business_0001"].rival_prices["Empanadas"] == [
        RivalPrice(seller_id="business_0004", seller_name="Casa de Sabor", price=9.99),
        RivalPrice(seller_id="business_0005", seller_name="Sol y Sazon", price=8.5),
    ]
    assert [
        r.seller_id for r in memories["business_0004"].rival_prices["Empanadas"]
    ] == ["business_0001", "business_0005"]


def test_rivals_use_only_the_last_round_and_omit_items_no_rival_has():
    """Scenario: The seller sees last round's prices of rivals (detail 2: last round only)."""
    rows = [
        row(1, "business_0001", "Empanadas", 9.49),
        row(1, "business_0004", "Empanadas", 9.99),
        row(1, "business_0004", "Agua Fresca", 3.1),
        row(2, "business_0001", "Empanadas", 10.5),
        row(2, "business_0004", "Empanadas", 11.25),
    ]

    rivals = build_seller_memories(TRIO, rows, {}, next_round=3)[
        "business_0001"
    ].rival_prices

    assert rivals == {
        "Empanadas": [
            RivalPrice(
                seller_id="business_0004", seller_name="Casa de Sabor", price=11.25
            )
        ]
    }


def test_rivals_items_off_the_sellers_menu_are_not_listed():
    """Scenario: The seller sees last round's prices of rivals that sell the same item (own menu only)."""
    rows = [
        row(1, "business_0007", "Tacos", 4.0),
        row(1, "business_0004", "Empanadas", 9.99),
    ]

    memories = build_seller_memories(TRIO, rows, {}, next_round=2)

    assert memories["business_0007"].rival_prices == {}
    assert "Tacos" not in memories["business_0001"].rival_prices


# Scenario: The seller writes notes ... / The round-2 pricing prompt shows no notes


def test_notes_come_from_last_rounds_notes_by_seller():
    """Scenario: The seller writes notes in its pricing step and sees them next round (builder)."""
    rows = [row(1, "business_0001", "Empanadas", 9.49)]

    memories = build_seller_memories(
        TRIO,
        rows,
        {"business_0001": EMPANADA_NOTES, "business_0004": ""},
        next_round=3,
    )

    assert memories["business_0001"].notes == EMPANADA_NOTES
    # Empty or missing notes are no notes
    assert memories["business_0004"].notes is None
    assert memories["business_0007"].notes is None


def test_round_2_memory_has_round_1_history_and_rivals_but_no_notes():
    """Scenario: The round-2 pricing prompt shows round-1 history but no notes (builder)."""
    rows = [
        row(1, "business_0001", "Empanadas", 9.49, 1),
        row(1, "business_0001", "Agua Fresca", 3.0),
        row(1, "business_0004", "Empanadas", 9.99),
    ]
    round_1_notes = {"business_0001": "", "business_0004": ""}

    memory = build_seller_memories(TRIO, rows, round_1_notes, next_round=2)[
        "business_0001"
    ]

    assert memory.notes is None
    assert [h.round for h in memory.history] == [1]
    assert memory.rival_prices == {
        "Empanadas": [
            RivalPrice(
                seller_id="business_0004", seller_name="Casa de Sabor", price=9.99
            )
        ]
    }


# Run wiring: fakes

# Every pricing prompt of a fake run: (round, seller id, prompt)
PRICING_PROMPTS: list[tuple[int, str, str]] = []
# Notes the fake LLM writes: (round, seller id) -> notes
NOTES_TO_WRITE: dict[tuple[int, str], str] = {}
# Units sold per (round, seller, item) in the fake seller table; default 0
UNITS_SOLD: dict[tuple[int, str, str], int] = {}
# Price overrides per (round, seller, item) in the fake seller table
PRICE_OVERRIDES: dict[tuple[int, str, str], float] = {}


class FakeActionTable:
    """In-memory `actions` table, so the run can read the highest row index."""

    def __init__(self):
        """Start with no rows."""
        self.rows: list[ActionRow] = []

    def append(self, index: int) -> None:
        """Append one action row with the given row index."""
        self.rows.append(
            ActionRow(
                id=f"action-{index}",
                created_at=datetime.now(UTC),
                index=index,
                data=ActionRowData(
                    agent_id="customer_0001",
                    request=ActionExecutionRequest(name="FetchMessages", parameters={}),
                    result=ActionExecutionResult(content={}),
                ),
            )
        )

    def _select(self, params) -> list[ActionRow]:
        rows = sorted(self.rows, key=lambda r: r.index or 0)
        if params is None:
            return rows
        if params.after_index is not None:
            rows = [r for r in rows if (r.index or 0) > params.after_index]
        if params.before_index is not None:
            rows = [r for r in rows if (r.index or 0) < params.before_index]
        rows = rows[params.offset :]
        if params.limit is not None:
            rows = rows[: params.limit]
        return rows

    async def get_all(self, params=None, batch_size=1000) -> list[ActionRow]:
        """Return rows in row-index order, honouring range parameters."""
        return self._select(params)

    async def find(self, query, params=None) -> list[ActionRow]:
        """Return rows in row-index order, honouring range parameters (query ignored)."""
        return self._select(params)

    async def count(self) -> int:
        """Return the number of rows."""
        return len(self.rows)


class FakeDatabaseController:
    """Database controller holding only a fake actions table."""

    def __init__(self):
        """Create the fake tables."""
        self.actions = FakeActionTable()
        self.agents = Mock()
        self.logs = Mock()
        self.row_index_column = "row_index"

    async def execute(self, command):
        """Raw SQL is not supported by the fake."""
        raise NotImplementedError("FakeDatabaseController.execute")


class FakeCustomerAgent:
    """Records the customer and keyword arguments a CustomerAgent was built with."""

    def __init__(self, customer, base_url, **kwargs):
        """Record constructor arguments."""
        self.customer = customer
        self.kwargs = kwargs


class FakeBusinessAgent:
    """Records the keyword arguments a BusinessAgent was built with; no pricing.

    Ends its round with menu prices and the notes "notes of <id> in round <r>" from
    round 2 on; round 1 has no pricing step, so its notes are "" as the real agent's.
    """

    def __init__(self, business, base_url, **kwargs):
        """Record constructor arguments."""
        self.business = business
        self.kwargs = kwargs
        self.round_prices: dict[str, float] | None = None
        self.round_notes: str = ""


class RecordingLogger:
    """Stands in for MarketplaceLogger; accepts every call."""

    def __init__(self):
        """Start with no entries."""
        self.entries: list[tuple[str, Any, Any]] = []

    def debug(self, message=None, *, data=None, metadata=None):
        """Record a debug call."""
        self.entries.append(("debug", message, data))

    def info(self, message=None, *, data=None, metadata=None):
        """Record an info call."""
        self.entries.append(("info", message, data))

    def warning(self, message=None, *, data=None, metadata=None):
        """Record a warning call."""
        self.entries.append(("warning", message, data))

    def error(self, message=None, *, data=None, metadata=None):
        """Record an error call."""
        self.entries.append(("error", message, data))

    def exception(self, message=None, *, data=None, metadata=None):
        """Record an exception call."""
        self.entries.append(("exception", message, data))


class PricingBusinessAgent(BusinessAgent):
    """A real BusinessAgent with a fake LLM and logger.

    Its pricing call keeps every current price, writes the notes in NOTES_TO_WRITE for
    its (round, id) (default ""), and records its prompt in PRICING_PROMPTS.
    """

    def __init__(self, business, base_url, **kwargs):
        """Build the real agent with a dummy LLM config and a recording logger."""
        kwargs["llm_config"] = OpenAIConfig(
            provider="openai", model="gpt-4.1-mini", api_key="none"
        )
        super().__init__(business, base_url, **kwargs)
        self._logger = RecordingLogger()  # type: ignore[assignment]

    async def generate_struct(self, prompt, response_format, **kwargs):  # type: ignore[override]
        """Answer a pricing prompt with the menu prices and this seller's notes."""
        assert response_format is PriceList, "only pricing calls are expected"
        PRICING_PROMPTS.append((self.round_number, self.business.id, prompt))
        prices = [
            ItemPrice(item_name=name, price=price)
            for name, price in self.business.menu_features.items()
        ]
        notes = NOTES_TO_WRITE.get((self.round_number, self.business.id), "")
        return PriceList(prices=prices, notes=notes), None


class FakeMarketplaceLauncher:
    """Async context manager that starts nothing and exposes a fake database."""

    def __init__(self, **kwargs):
        """Store a fake server URL and a server whose state holds a fake database."""
        self.protocol = kwargs.get("protocol")
        self.server_url = "http://fake.invalid"
        self.server = SimpleNamespace(
            state=SimpleNamespace(database_controller=FakeAgentLauncher.database)
        )

    async def __aenter__(self):
        """Start nothing."""
        return self

    async def __aexit__(self, *exc):
        """Stop nothing."""
        return False

    async def create_logger(self, name):
        """Return a mock logger."""
        return Mock()


class FakeAgentLauncher:
    """Runs one round: starts every seller, writes the round's seller-table rows.

    Real sellers run `on_started`; fake sellers end with menu prices and, from round 2,
    notes "notes of <id> in round <r>". Each seller writes one row per menu item at its
    round price (or PRICE_OVERRIDES) with UNITS_SOLD units.
    """

    runs: list[dict] = []
    database = FakeDatabaseController()
    seller_rows: list[SellerRoundRow] = []

    def __init__(self, server_url):
        """Accept the server URL."""

    async def __aenter__(self):
        """Start nothing."""
        return self

    async def __aexit__(self, *exc):
        """Stop nothing."""
        return False

    async def run_agents_with_dependencies(self, primary_agents, dependent_agents):
        """Start the sellers, record the cohort and write this round's rows."""
        round_number = len(FakeAgentLauncher.runs) + 1
        FakeAgentLauncher.runs.append(
            {"customers": list(primary_agents), "businesses": list(dependent_agents)}
        )
        for seller in dependent_agents:
            if isinstance(seller, BusinessAgent):
                await seller.on_started()
            else:
                seller.round_prices = dict(seller.business.menu_features)
                if round_number > 1:
                    seller.round_notes = (
                        f"notes of {seller.business.id} in round {round_number}"
                    )
            prices = seller.round_prices or dict(seller.business.menu_features)
            for item, price in prices.items():
                key = (round_number, seller.business.id, item)
                FakeAgentLauncher.seller_rows.append(
                    row(
                        round_number,
                        seller.business.id,
                        item,
                        PRICE_OVERRIDES.get(key, price),
                        UNITS_SOLD.get(key, 0),
                    )
                )
        FakeAgentLauncher.database.actions.append(10 * round_number)


# Every call to the fake compute_round_tables: the database it was given
TABLE_CALLS: list[Any] = []


async def fake_compute_round_tables(database):
    """Return no customer rows and the seller rows written so far."""
    TABLE_CALLS.append(database)
    return [], list(FakeAgentLauncher.seller_rows)


@pytest.fixture
def fake_run(monkeypatch):
    """Patch the run's agents, launchers and seller table with fakes; return the cohorts."""
    PRICING_PROMPTS.clear()
    NOTES_TO_WRITE.clear()
    UNITS_SOLD.clear()
    PRICE_OVERRIDES.clear()
    TABLE_CALLS.clear()
    FakeAgentLauncher.runs = []
    FakeAgentLauncher.database = FakeDatabaseController()
    FakeAgentLauncher.seller_rows = []
    monkeypatch.setattr(run_experiment, "CustomerAgent", FakeCustomerAgent)
    monkeypatch.setattr(run_experiment, "BusinessAgent", FakeBusinessAgent)
    monkeypatch.setattr(run_experiment, "MarketplaceLauncher", FakeMarketplaceLauncher)
    monkeypatch.setattr(run_experiment, "AgentLauncher", FakeAgentLauncher)
    monkeypatch.setattr(
        run_experiment,
        "compute_round_tables",
        fake_compute_round_tables,
        raising=False,
    )
    return FakeAgentLauncher.runs


@pytest.fixture
def real_sellers(monkeypatch, fake_run):
    """Like fake_run, but sellers are real BusinessAgents with a fake LLM."""
    monkeypatch.setattr(run_experiment, "BusinessAgent", PricingBusinessAgent)
    return fake_run


@pytest.fixture
def run_folder(tmp_path, monkeypatch):
    """Create research/results/memory_test under tmp_path and make it the cwd."""
    folder = tmp_path / "research" / "results" / "memory_test"
    folder.mkdir(parents=True)
    monkeypatch.chdir(folder)
    return folder


async def run(rounds: int, **kwargs) -> None:
    """Run the real run function on mexican_3_9 with a fixed port."""
    await run_experiment.run_marketplace_experiment(
        data_dir=MEXICAN_3_9,
        experiment_name="memory_test",
        server_port=18999,
        rounds=rounds,
        **kwargs,
    )


def memory_of(agent) -> SellerMemory | None:
    """Return the memory a fake BusinessAgent was built with (None if not passed)."""
    return agent.kwargs.get("memory")


def prompt_of(round_number: int, seller_id: str) -> str:
    """Return the one pricing prompt of `seller_id` in `round_number`."""
    prompts = [p for r, s, p in PRICING_PROMPTS if r == round_number and s == seller_id]
    assert len(prompts) == 1, (round_number, seller_id, len(prompts))
    return prompts[0]


# Run wiring (detail 1)


async def test_round_1_sellers_get_no_memory_and_later_rounds_do(run_folder, fake_run):
    """Scenario: The round-2 pricing prompt shows round-1 history (detail 1: who gets memory)."""
    await run(3, seller_pricing=True)

    assert len(fake_run) == 3
    assert all(memory_of(a) is None for a in fake_run[0]["businesses"])
    for cohort in fake_run[1:]:
        assert len(cohort["businesses"]) == 9
        for agent in cohort["businesses"]:
            assert isinstance(memory_of(agent), SellerMemory), agent.business.id


async def test_run_reads_the_seller_table_from_the_run_database(run_folder, fake_run):
    """Scenario: The round-2 pricing prompt shows round-1 history (detail 1: table from the in-process database)."""
    await run(3, seller_pricing=True)

    assert TABLE_CALLS, "the run never computed the seller table"
    assert all(db is FakeAgentLauncher.database for db in TABLE_CALLS)


async def test_memory_carries_each_sellers_notes_from_the_round_before(
    run_folder, fake_run
):
    """Scenario: The seller writes notes in its pricing step and sees them next round (run wiring)."""
    await run(3, seller_pricing=True)

    round_2, round_3 = fake_run[1], fake_run[2]
    # Round 1 has no pricing step, so round 2 sellers have no notes
    for agent in round_2["businesses"]:
        memory = memory_of(agent)
        assert memory is not None, agent.business.id
        assert memory.notes is None
    # Round 3 sellers get exactly what their round-2 agent wrote
    for agent in round_3["businesses"]:
        memory = memory_of(agent)
        assert memory is not None, agent.business.id
        assert memory.notes == f"notes of {agent.business.id} in round 2"


async def test_memory_history_and_rivals_come_from_the_seller_table(
    run_folder, fake_run
):
    """Scenario: The seller sees its own ... and rivals' last prices (run wiring)."""
    PRICE_OVERRIDES[(1, "business_0001", PUMPKIN)] = 9.49
    UNITS_SOLD[(1, "business_0001", PUMPKIN)] = 1
    PRICE_OVERRIDES[(2, "business_0001", PUMPKIN)] = 10.5
    PRICE_OVERRIDES[(2, "business_0002", PUMPKIN)] = 9.99

    await run(3, seller_pricing=True)

    round_2_0001 = next(
        a for a in fake_run[1]["businesses"] if a.business.id == "business_0001"
    )
    round_3_0001 = next(
        a for a in fake_run[2]["businesses"] if a.business.id == "business_0001"
    )
    round_2_memory = memory_of(round_2_0001)
    round_3_memory = memory_of(round_3_0001)
    assert round_2_memory is not None
    assert round_3_memory is not None

    # History: round 2 sees round 1; round 3 sees rounds 1 and 2
    assert [h.round for h in round_2_memory.history] == [1]
    assert [h.round for h in round_3_memory.history] == [1, 2]
    pumpkin_by_round = {
        h.round: next(i for i in h.items if i.item == PUMPKIN)
        for h in round_3_memory.history
    }
    assert pumpkin_by_round[1] == ItemSales(
        item=PUMPKIN, price=9.49, units_sold=1, revenue=9.49
    )
    assert pumpkin_by_round[2] == ItemSales(
        item=PUMPKIN, price=10.5, units_sold=0, revenue=0.0
    )

    # Rivals: other sellers of the pumpkin empanadas, at the previous round's price
    business_0002 = BUSINESSES["business_0002"]
    business_0003 = BUSINESSES["business_0003"]
    assert round_3_memory.rival_prices[PUMPKIN] == [
        RivalPrice(
            seller_id="business_0002", seller_name=business_0002.name, price=9.99
        ),
        RivalPrice(
            seller_id="business_0003",
            seller_name=business_0003.name,
            price=business_0003.menu_features[PUMPKIN],
        ),
    ]
    assert (
        round_2_memory.rival_prices[PUMPKIN][0].price
        == (business_0002.menu_features[PUMPKIN])
    )


@pytest.mark.parametrize("rounds", [1, 3])
async def test_pricing_off_builds_no_memory(run_folder, fake_run, rounds):
    """Scenario: The round-2 pricing prompt shows round-1 history (detail 1: pricing off changes nothing)."""
    await run(rounds)

    assert len(fake_run) == rounds
    for cohort in fake_run:
        assert all(memory_of(a) is None for a in cohort["businesses"])
    assert TABLE_CALLS == []


async def test_one_round_with_pricing_on_reads_no_seller_table(run_folder, fake_run):
    """Scenario: The round-2 pricing prompt shows round-1 history (detail 1: no next round, no memory)."""
    await run(1, seller_pricing=True)

    assert len(fake_run) == 1
    assert all(memory_of(a) is None for a in fake_run[0]["businesses"])
    assert TABLE_CALLS == []


# Run end to end with real sellers


async def test_notes_written_in_round_2_show_in_the_round_3_prompt(
    run_folder, real_sellers
):
    """Scenario: The seller writes notes in its pricing step and sees them next round."""
    # Given a config.yml with 3 rounds and seller pricing on
    # When seller "business_0001" writes the notes in its round-2 pricing step
    NOTES_TO_WRITE[(2, "business_0001")] = EMPANADA_NOTES

    await run(3, seller_pricing=True)

    # Then its round-3 pricing prompt shows the notes
    round_3_prompt = prompt_of(3, "business_0001")
    assert NOTES_HEADER in round_3_prompt
    assert EMPANADA_NOTES in round_3_prompt
    # No other seller sees them
    for seller_id in BUSINESSES:
        if seller_id != "business_0001":
            assert EMPANADA_NOTES not in prompt_of(3, seller_id)


async def test_round_3_prompt_shows_own_history_from_the_run(run_folder, real_sellers):
    """Scenario: The seller sees its own prices, units sold and revenue per past round (run)."""
    # Given seller "business_0001" priced the empanadas at $9.49 and sold 1 in round 1
    # And priced them at $10.50 and sold 0 in round 2
    PRICE_OVERRIDES[(1, "business_0001", PUMPKIN)] = 9.49
    UNITS_SOLD[(1, "business_0001", PUMPKIN)] = 1
    PRICE_OVERRIDES[(2, "business_0001", PUMPKIN)] = 10.5

    # When its round-3 pricing prompt is rendered
    await run(3, seller_pricing=True)

    # Then it shows both rounds' lines
    prompt = prompt_of(3, "business_0001")
    assert HISTORY_HEADER in prompt
    assert f"- Round 1: {PUMPKIN}: $9.49, 1 sold, revenue $9.49" in prompt
    assert f"- Round 2: {PUMPKIN}: $10.50, 0 sold, revenue $0.00" in prompt


async def test_round_2_prompt_shows_rivals_round_1_prices_from_the_run(
    run_folder, real_sellers
):
    """Scenario: The seller sees last round's prices of rivals that sell the same item (run)."""
    # Given business_0002 and business_0003 also sell the pumpkin empanadas; business_0007 does not
    PRICE_OVERRIDES[(1, "business_0002", PUMPKIN)] = 9.99
    assert PUMPKIN not in BUSINESSES["business_0007"].menu_features

    # When the round-2 pricing prompt of "business_0001" is rendered
    await run(2, seller_pricing=True)

    # Then it lists the rivals with their names and round-1 prices, and not business_0007
    prompt = prompt_of(2, "business_0001")
    business_0002 = BUSINESSES["business_0002"]
    business_0003 = BUSINESSES["business_0003"]
    assert RIVALS_HEADER in prompt
    assert f"- {PUMPKIN}: {business_0002.name} (business_0002) $9.99" in prompt
    price_0003 = business_0003.menu_features[PUMPKIN]
    assert f"- {PUMPKIN}: {business_0003.name} (business_0003) ${price_0003:.2f}" in (
        prompt
    )
    assert not any(
        line.strip().startswith(f"- {PUMPKIN}:") and "business_0007" in line
        for line in prompt.splitlines()
    )


async def test_round_2_prompts_show_round_1_history_but_no_notes(
    run_folder, real_sellers
):
    """Scenario: The round-2 pricing prompt shows round-1 history but no notes (run)."""
    # Given a config.yml with seller pricing on
    # When a seller's round-2 pricing prompt is rendered
    await run(2, seller_pricing=True)

    # Then it shows round 1 prices, units sold and revenue, and rivals' round-1 prices
    # And it shows no notes
    assert {s for r, s, _ in PRICING_PROMPTS if r == 2} == set(BUSINESSES)
    for seller_id, business in BUSINESSES.items():
        prompt = prompt_of(2, seller_id)
        assert HISTORY_HEADER in prompt
        for item, price in business.menu_features.items():
            assert f"- Round 1: {item}: ${price:.2f}, 0 sold, revenue $0.00" in prompt
        assert "- Round 2:" not in prompt
        assert NOTES_HEADER not in prompt
    assert RIVALS_HEADER in prompt_of(2, "business_0001")


async def test_no_pricing_prompt_of_a_5_round_run_says_how_many_rounds(
    run_folder, real_sellers
):
    """Scenario: No prompt says how many rounds the run has (5-round run, pricing prompts with memory)."""
    # Given a config.yml with 5 rounds and seller pricing on
    NOTES_TO_WRITE.update({(r, "business_0001"): EMPANADA_NOTES for r in (2, 3, 4)})

    # When any seller prompt is rendered
    await run(5, seller_pricing=True)

    # Then it does not state the total number of rounds
    assert {r for r, _, _ in PRICING_PROMPTS} == {2, 3, 4, 5}
    assert HISTORY_HEADER in prompt_of(5, "business_0001")
    assert NOTES_HEADER in prompt_of(5, "business_0001")
    for _, _, prompt in PRICING_PROMPTS:
        checked = prompt.replace(NOTES_HEADER, "").replace(RIVALS_HEADER, "")
        for pattern in ROUND_COUNT_PATTERNS:
            assert not pattern.search(checked), pattern.pattern
    # Customers are built without any count of rounds
    for cohort in real_sellers:
        for customer in cohort["customers"]:
            assert not {"rounds", "total_rounds"} & set(customer.kwargs)

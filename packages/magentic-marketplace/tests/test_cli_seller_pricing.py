"""Run-wiring tests for the Seller pricing feature (tests/features/seller_pricing.feature).

Covers the config switch, the CLI, the run passing `seller_pricing` and last round's
prices to every seller, and the order of round-2 pricing against round 1
(tests/features/rounds.feature, "Pricing for round 2 waits for round 1 to finish").

The CLI tests replace `run_marketplace_experiment` with a recording fake. The run tests
call the real `run_marketplace_experiment` with fakes in place of `CustomerAgent`,
`MarketplaceLauncher` and `AgentLauncher`, and either a recording fake or a real
`BusinessAgent` with a fake LLM and logger, so no Postgres, server or LLM is needed.
"""

import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from magentic_marketplace import cli
from magentic_marketplace.experiments import run_experiment
from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.agents.business.agent import BusinessAgent
from magentic_marketplace.marketplace.agents.business.pricing import (
    ItemPrice,
    PriceList,
)
from magentic_marketplace.marketplace.llm.clients.openai import OpenAIConfig
from magentic_marketplace.platform.database.models import ActionRow, ActionRowData
from magentic_marketplace.platform.shared.models import (
    ActionExecutionRequest,
    ActionExecutionResult,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
CUSTOMER_IDS = sorted(c.id for c in load_customers_from_yaml(MEXICAN_3_9 / "customers"))
MENUS = {
    b.id: dict(b.menu_features)
    for b in load_businesses_from_yaml(MEXICAN_3_9 / "businesses")
}

# Every event of a fake run, in order: (kind, round, agent id)
EVENTS: list[tuple[str, int, str]] = []
# Every pricing prompt of a fake run: (round, seller id, prompt)
PRICING_PROMPTS: list[tuple[int, str, str]] = []


# Fakes


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
        rows = sorted(self.rows, key=lambda row: row.index or 0)
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
    """Records the keyword arguments a BusinessAgent was built with; no pricing."""

    def __init__(self, business, base_url, **kwargs):
        """Record constructor arguments; round_prices starts as None like the real agent."""
        self.business = business
        self.kwargs = kwargs
        self.round_prices: dict[str, float] | None = None


class RecordingLogger:
    """Stands in for MarketplaceLogger; accepts and records every call."""

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

    Its pricing call raises every menu price by $1.00 and records start and end in
    EVENTS and its prompt in PRICING_PROMPTS.
    """

    def __init__(self, business, base_url, **kwargs):
        """Build the real agent with a dummy LLM config and a recording logger."""
        kwargs["llm_config"] = OpenAIConfig(
            provider="openai", model="gpt-4.1-mini", api_key="none"
        )
        super().__init__(business, base_url, **kwargs)
        self._logger = RecordingLogger()  # type: ignore[assignment]

    async def generate_struct(self, prompt, response_format, **kwargs):  # type: ignore[override]
        """Answer a pricing prompt with every current price + $1.00; nothing else."""
        assert response_format is PriceList, "only pricing calls are expected"
        EVENTS.append(("pricing_start", self.round_number, self.business.id))
        PRICING_PROMPTS.append((self.round_number, self.business.id, prompt))
        prices = [
            ItemPrice(item_name=name, price=price + 1.0)
            for name, price in self.business.menu_features.items()
        ]
        EVENTS.append(("pricing_end", self.round_number, self.business.id))
        return PriceList(prices=prices), None


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
    """Runs one round: starts every seller, then finishes every customer.

    Records each cohort. Sellers that have `on_started` (real agents) are started; a
    `price_shift` per round, if set, is applied to fake sellers' `round_prices` to stand
    in for their pricing. Each round writes one action row.
    """

    runs: list[dict] = []
    database = FakeDatabaseController()
    price_shift: dict[int, float] = {}

    def __init__(self, server_url):
        """Accept the server URL."""

    async def __aenter__(self):
        """Start nothing."""
        return self

    async def __aexit__(self, *exc):
        """Stop nothing."""
        return False

    async def run_agents_with_dependencies(self, primary_agents, dependent_agents):
        """Start the sellers, finish the customers, record the cohort."""
        round_number = len(FakeAgentLauncher.runs) + 1
        FakeAgentLauncher.runs.append(
            {"customers": list(primary_agents), "businesses": list(dependent_agents)}
        )
        for seller in dependent_agents:
            if isinstance(seller, BusinessAgent):
                await seller.on_started()
            elif round_number in FakeAgentLauncher.price_shift:
                shift = FakeAgentLauncher.price_shift[round_number]
                seller.round_prices = {
                    item: round(price + shift, 2)
                    for item, price in seller.business.menu_features.items()
                }
        for customer in primary_agents:
            EVENTS.append(("customer_finished", round_number, customer.customer.id))
        FakeAgentLauncher.database.actions.append(10 * round_number)


@pytest.fixture(autouse=True)
def empty_seller_table(monkeypatch):
    """Serve an empty seller table between rounds; the fake database cannot build one."""

    async def no_rows(database):
        return [], []

    monkeypatch.setattr(run_experiment, "compute_round_tables", no_rows)


@pytest.fixture
def fake_run(monkeypatch):
    """Patch the run's agents and launchers with fakes; return the recorded cohorts."""
    EVENTS.clear()
    PRICING_PROMPTS.clear()
    FakeAgentLauncher.runs = []
    FakeAgentLauncher.database = FakeDatabaseController()
    FakeAgentLauncher.price_shift = {}
    monkeypatch.setattr(run_experiment, "CustomerAgent", FakeCustomerAgent)
    monkeypatch.setattr(run_experiment, "BusinessAgent", FakeBusinessAgent)
    monkeypatch.setattr(run_experiment, "MarketplaceLauncher", FakeMarketplaceLauncher)
    monkeypatch.setattr(run_experiment, "AgentLauncher", FakeAgentLauncher)
    return FakeAgentLauncher.runs


@pytest.fixture
def real_sellers(monkeypatch, fake_run):
    """Like fake_run, but sellers are real BusinessAgents with a fake LLM."""
    monkeypatch.setattr(run_experiment, "BusinessAgent", PricingBusinessAgent)
    return fake_run


@pytest.fixture
def recorded_runs(monkeypatch):
    """Replace cli.run_marketplace_experiment with a fake that records its kwargs."""
    calls: list[dict] = []

    async def fake_run_marketplace_experiment(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        cli, "run_marketplace_experiment", fake_run_marketplace_experiment
    )
    return calls


@pytest.fixture
def run_folder(tmp_path, monkeypatch):
    """Create research/results/pricing_test under tmp_path and make it the cwd."""
    folder = tmp_path / "research" / "results" / "pricing_test"
    folder.mkdir(parents=True)
    monkeypatch.chdir(folder)
    return folder


# Helpers


def write_config(path: Path, content: dict) -> Path:
    """Write `content` as YAML to `path` and return the path."""
    path.write_text(yaml.safe_dump(content))
    return path


def run_cli(monkeypatch, *args: str) -> None:
    """Run `magentic-marketplace <args>` in-process."""
    monkeypatch.setattr(sys, "argv", ["magentic-marketplace", *args])
    cli.main()


def seller_pricing_of(agent) -> bool:
    """Return the seller_pricing a BusinessAgent was built with (False if not passed)."""
    return agent.kwargs.get("seller_pricing", False)


async def run(rounds: int, **kwargs) -> None:
    """Run the real run function on mexican_3_9 with a fixed port."""
    await run_experiment.run_marketplace_experiment(
        data_dir=MEXICAN_3_9,
        experiment_name="pricing_test",
        server_port=18999,
        rounds=rounds,
        **kwargs,
    )


def pricing_events(round_number: int | None = None) -> list[int]:
    """Return the EVENTS positions of pricing starts, optionally for one round."""
    return [
        i
        for i, (kind, r, _) in enumerate(EVENTS)
        if kind == "pricing_start" and (round_number is None or r == round_number)
    ]


# Rule: With pricing off, sellers behave exactly as stock


@pytest.mark.parametrize("setting", [{}, {"seller_pricing": False}])
def test_config_with_pricing_off_reaches_run_as_off(
    monkeypatch, tmp_path, run_folder, recorded_runs, setting
):
    """Scenario: With pricing off, no pricing step runs (config -> run; passes on the stub)."""
    # Given a config.yml with seller pricing off
    config_path = write_config(
        tmp_path / "input.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "pricing_test", **setting},
    )

    # When a run starts
    run_cli(monkeypatch, "run", "--config", str(config_path))

    # Then the run has pricing off
    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("seller_pricing", False) is False


def test_stock_command_runs_with_pricing_off(monkeypatch, run_folder, recorded_runs):
    """Scenario: With pricing off, no pricing step runs (stock command; passes on the stub)."""
    run_cli(monkeypatch, "run", str(MEXICAN_3_9), "--experiment-name", "pricing_test")

    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("seller_pricing", False) is False


@pytest.mark.parametrize("rounds", [1, 2])
async def test_run_with_pricing_off_builds_sellers_without_pricing(
    run_folder, fake_run, rounds
):
    """Scenario: With pricing off, no pricing step runs (run -> sellers; passes on the stub)."""
    await run(rounds)

    assert len(fake_run) == rounds
    for cohort in fake_run:
        assert len(cohort["businesses"]) == 9
        assert all(seller_pricing_of(a) is False for a in cohort["businesses"])
        assert all(a.kwargs.get("current_prices") is None for a in cohort["businesses"])


async def test_run_with_pricing_off_makes_no_pricing_call(run_folder, real_sellers):
    """Scenario: With pricing off, no pricing step runs (real sellers, 2 rounds; passes on the stub)."""
    # Given a config.yml with seller pricing off
    # When a run starts
    await run(2)

    # Then no seller makes a pricing call
    assert len(real_sellers) == 2
    assert pricing_events() == []
    for cohort in real_sellers:
        assert all(seller.round_prices is None for seller in cohort["businesses"])


# Rule: With pricing on, round 1 uses menu prices; from round 2 sellers set prices


def test_config_with_pricing_on_reaches_the_run(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Scenario: Round 1 quotes menu prices (detail 1: config turns pricing on)."""
    # Given a config.yml with seller pricing on
    config_path = write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "pricing_test",
            "rounds": 2,
            "seller_pricing": True,
        },
    )

    # When a run starts
    run_cli(monkeypatch, "run", "--config", str(config_path))

    # Then the run has pricing on
    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("seller_pricing") is True
    assert recorded_runs[0].get("rounds") == 2


async def test_run_with_pricing_on_builds_every_seller_with_pricing(
    run_folder, fake_run
):
    """Scenario: Round 1 quotes menu prices (detail 1: every seller of every round gets pricing on)."""
    await run(2, seller_pricing=True)

    assert len(fake_run) == 2
    for cohort in fake_run:
        assert len(cohort["businesses"]) == 9
        assert all(seller_pricing_of(a) is True for a in cohort["businesses"])


def test_config_with_pricing_on_builds_every_seller_with_pricing(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Scenario: Round 1 quotes menu prices (detail 1: config through CLI to sellers)."""
    config_path = write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "pricing_test",
            "rounds": 2,
            "seller_pricing": True,
        },
    )

    run_cli(monkeypatch, "run", "--config", str(config_path), "--server-port", "18999")

    assert len(fake_run) == 2
    for cohort in fake_run:
        assert all(seller_pricing_of(a) is True for a in cohort["businesses"])


async def test_prices_carry_over_to_the_next_round(run_folder, fake_run):
    """Scenario: The pricing prompt shows current prices (detail 2: last round's prices carry over)."""
    # Round 1 sellers end with menu + $0.50, round 2 sellers with menu + $2.00
    FakeAgentLauncher.price_shift = {1: 0.5, 2: 2.0}

    await run(3, seller_pricing=True)

    assert len(fake_run) == 3
    # Round 1 sellers start from no current prices (the menu)
    assert all(
        a.kwargs.get("current_prices") is None for a in fake_run[0]["businesses"]
    )
    # Round r sellers start from the round_prices their round r-1 agent ended with
    for previous, cohort in zip(fake_run, fake_run[1:], strict=False):
        ended_with = {a.business.id: a.round_prices for a in previous["businesses"]}
        assert len(cohort["businesses"]) == 9
        for agent in cohort["businesses"]:
            assert agent.kwargs.get("current_prices") == ended_with[agent.business.id]
    # Spot check the numbers
    round_2_0001 = next(
        a for a in fake_run[1]["businesses"] if a.business.id == "business_0001"
    )
    assert round_2_0001.kwargs["current_prices"]["Savory Pumpkin Empanadas"] == 11.28


async def test_round_1_makes_no_pricing_call_and_round_2_prices_every_seller(
    run_folder, real_sellers
):
    """Scenario: Round 1 quotes menu prices (run) and every seller sets prices before round 2."""
    await run(2, seller_pricing=True)

    assert len(real_sellers) == 2
    round_1, round_2 = real_sellers
    # Round 1: no pricing call, every seller quotes its menu
    assert pricing_events(1) == []
    for seller in round_1["businesses"]:
        assert seller.round_prices == MENUS[seller.business.id]
    # Round 2: one pricing call per seller, starting from round 1's prices
    assert sorted(seller_id for r, seller_id, _ in PRICING_PROMPTS if r == 2) == sorted(
        MENUS
    )
    for _, seller_id, prompt in PRICING_PROMPTS:
        for item, price in MENUS[seller_id].items():
            assert f"- {item}: ${price:.2f}" in prompt
    for seller in round_2["businesses"]:
        menu = MENUS[seller.business.id]
        assert seller.round_prices == {
            item: round(price + 1.0, 2) for item, price in menu.items()
        }


async def test_pricing_prompts_of_a_run_do_not_say_how_many_rounds(
    run_folder, real_sellers
):
    """Scenario: The pricing prompt shows current prices and the revenue goal (no round count, 3-round run)."""
    await run(3, seller_pricing=True)

    assert {r for r, _, _ in PRICING_PROMPTS} == {2, 3}
    for _, _, prompt in PRICING_PROMPTS:
        assert not re.search(r"\b(3|three)\s+rounds\b", prompt, re.I)
        assert not re.search(r"\bround\s+\d+\s*(of|/)\s*3\b", prompt, re.I)
        assert not re.search(r"\b(last|final)\s+round\b", prompt, re.I)


# Rule: A round ends when every customer has finished; the next round's pricing
# starts after


async def test_pricing_for_round_2_waits_for_round_1_to_finish(
    run_folder, real_sellers
):
    """Scenario: Pricing for round 2 waits for round 1 to finish (rounds.feature)."""
    # Given a config.yml with 2 rounds and seller pricing on
    # When the run finishes
    await run(2, seller_pricing=True)

    # Then no seller's round-2 pricing call starts before the last customer finishes
    # round 1
    round_1_finishes = [
        i
        for i, (kind, r, _) in enumerate(EVENTS)
        if kind == "customer_finished" and r == 1
    ]
    assert sorted(EVENTS[i][2] for i in round_1_finishes) == CUSTOMER_IDS
    round_2_pricing = pricing_events(2)
    assert len(round_2_pricing) == 9
    assert min(round_2_pricing) > max(round_1_finishes)

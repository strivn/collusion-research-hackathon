"""Run-wiring tests for the round gate (tests/features/round_start.technical.feature).

The real `run_marketplace_experiment` runs with fakes in place of `CustomerAgent`,
`BusinessAgent`, `MarketplaceLauncher` and `AgentLauncher`, as in tests/test_rounds.py,
so no Postgres, server or LLM is needed. The fake agents record the keyword arguments
they were built with; the tests read `round_gate` from them.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from magentic_marketplace import cli
from magentic_marketplace.experiments import run_experiment
from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.agents.round_gate import RoundGate
from magentic_marketplace.platform.database.models import ActionRow, ActionRowData
from magentic_marketplace.platform.database.queries import RangeQueryParams
from magentic_marketplace.platform.shared.models import (
    ActionExecutionRequest,
    ActionExecutionResult,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
SELLER_COUNT = len(load_businesses_from_yaml(MEXICAN_3_9 / "businesses"))
CUSTOMER_COUNT = len(load_customers_from_yaml(MEXICAN_3_9 / "customers"))


# Fakes


class FakeActionTable:
    """In-memory `actions` table, ordered by row index."""

    def __init__(self):
        """Start with no rows."""
        self.rows: list[ActionRow] = []

    def append(self, index: int, round_number: int) -> None:
        """Append one action row with the given row index."""
        self.rows.append(
            ActionRow(
                id=f"action-{index}",
                created_at=datetime.now(UTC),
                index=index,
                data=ActionRowData(
                    agent_id="customer_0001",
                    request=ActionExecutionRequest(
                        name="FetchMessages",
                        parameters={},
                        metadata={"round": round_number},
                    ),
                    result=ActionExecutionResult(content={}),
                ),
            )
        )

    def _select(self, params: RangeQueryParams | None) -> list[ActionRow]:
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

    async def get_by_id(self, item_id):
        """Return the row with this id."""
        return next((r for r in self.rows if r.id == item_id), None)


class FakeDatabaseController:
    """Database controller holding only a fake actions table."""

    def __init__(self):
        """Create the fake tables."""
        self.actions = FakeActionTable()
        self.agents = Mock()
        self.logs = Mock()
        self.row_index_column = "row_index"

    def max_action_index(self) -> int | None:
        """Return the highest row index in the actions table."""
        indexes = [row.index for row in self.actions.rows if row.index is not None]
        return max(indexes) if indexes else None

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
    """Records the business and keyword arguments a BusinessAgent was built with."""

    def __init__(self, business, base_url, **kwargs):
        """Record constructor arguments; round_prices starts as None like the real agent."""
        self.business = business
        self.kwargs = kwargs
        self.round_prices: dict[str, float] | None = None
        self.round_notes = ""


class FakeMarketplaceLauncher:
    """Async context manager that starts nothing and exposes a fake database."""

    database = FakeDatabaseController()

    def __init__(self, **kwargs):
        """Store a fake server URL and a server whose state holds the fake database."""
        self.protocol = kwargs.get("protocol")
        self.server_url = "http://fake.invalid"
        self.server = SimpleNamespace(
            state=SimpleNamespace(database_controller=FakeMarketplaceLauncher.database)
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
    """Records each round's agents and writes a few action rows per round."""

    runs: list[dict] = []

    def __init__(self, server_url):
        """Accept the server URL."""

    async def __aenter__(self):
        """Start nothing."""
        return self

    async def __aexit__(self, *exc):
        """Stop nothing."""
        return False

    async def run_agents_with_dependencies(self, primary_agents, dependent_agents):
        """Record the round's agents, then append its action rows."""
        database = FakeMarketplaceLauncher.database
        round_number = len(FakeAgentLauncher.runs) + 1
        FakeAgentLauncher.runs.append(
            {"customers": list(primary_agents), "businesses": list(dependent_agents)}
        )
        start = database.max_action_index() or 0
        for i in range(1, 4):
            database.actions.append(start + i * 3, round_number)


@pytest.fixture
def fake_run(monkeypatch):
    """Patch the run's agents and launchers with fakes; return the recorded rounds."""
    FakeAgentLauncher.runs = []
    FakeMarketplaceLauncher.database = FakeDatabaseController()

    async def no_rows(database):
        return [], []

    monkeypatch.setattr(run_experiment, "compute_round_tables", no_rows)
    monkeypatch.setattr(run_experiment, "CustomerAgent", FakeCustomerAgent)
    monkeypatch.setattr(run_experiment, "BusinessAgent", FakeBusinessAgent)
    monkeypatch.setattr(run_experiment, "MarketplaceLauncher", FakeMarketplaceLauncher)
    monkeypatch.setattr(run_experiment, "AgentLauncher", FakeAgentLauncher)
    return FakeAgentLauncher.runs


@pytest.fixture
def run_folder(tmp_path, monkeypatch):
    """Create research/results/gate_test under tmp_path and make it the cwd."""
    folder = tmp_path / "research" / "results" / "gate_test"
    folder.mkdir(parents=True)
    monkeypatch.chdir(folder)
    return folder


# Helpers


def run_cli(monkeypatch, *args: str) -> None:
    """Run `magentic-marketplace <args>` in-process."""
    monkeypatch.setattr(sys, "argv", ["magentic-marketplace", *args])
    cli.main()


async def run_rounds(**kwargs) -> None:
    """Run the real run function on mexican_3_9 with a fixed port."""
    await run_experiment.run_marketplace_experiment(
        data_dir=MEXICAN_3_9,
        experiment_name="gate_test",
        server_port=18999,
        **kwargs,
    )


def gate_of_round(cohort: dict) -> RoundGate:
    """Return the one gate every seller and customer of a round was built with."""
    agents = cohort["businesses"] + cohort["customers"]
    gates = [agent.kwargs.get("round_gate") for agent in agents]
    assert all(isinstance(g, RoundGate) for g in gates), (
        f"agents built without a RoundGate: {gates[:3]}..."
    )
    assert all(g is gates[0] for g in gates), "a round's agents got different gates"
    return gates[0]


def assert_one_gate_per_round(cohorts: list[dict], rounds: int) -> None:
    """Assert each round has its own gate, shared by its agents, sized to its sellers."""
    assert len(cohorts) == rounds
    for cohort in cohorts:
        assert len(cohort["businesses"]) == SELLER_COUNT
        assert len(cohort["customers"]) == CUSTOMER_COUNT
    gates = [gate_of_round(cohort) for cohort in cohorts]
    assert len({id(g) for g in gates}) == rounds, "two rounds share a gate"
    for gate in gates:
        assert gate.seller_count == SELLER_COUNT
        assert gate.timeout == 300.0


# Rule: Customers start a round only after every seller of that round has finished starting up


async def test_stock_run_gives_its_round_a_gate(fake_run):
    """Rule: Customers start a round only after every seller of that round has finished starting up.

    Scenario: Customers wait for sellers that are still pricing (run wiring, detail 4: a
    run with every new feature off still gets one gate for its single round).
    """
    await run_rounds()

    assert_one_gate_per_round(fake_run, rounds=1)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"seller_pricing": True},
        {"seller_pricing": True, "seller_channel": True},
    ],
    ids=["no-features", "pricing", "pricing-and-channel"],
)
async def test_each_round_gets_its_own_gate(fake_run, settings):
    """Rule: Customers start a round only after every seller of that round has finished starting up.

    Scenario: Each round waits for its own sellers (run wiring, detail 4: every round of
    a multi-round run gets a fresh gate shared by that round's sellers and customers).
    """
    await run_rounds(rounds=3, **settings)

    assert_one_gate_per_round(fake_run, rounds=3)


def test_stock_command_without_config_gives_a_gate(monkeypatch, run_folder, fake_run):
    """Rule: Customers start a round only after every seller of that round has finished starting up.

    Scenario: Customers wait for sellers that are still pricing (run wiring, detail 4:
    the stock command, no config).
    """
    run_cli(
        monkeypatch,
        "run",
        str(MEXICAN_3_9),
        "--experiment-name",
        "gate_test",
        "--server-port",
        "18999",
    )

    assert_one_gate_per_round(fake_run, rounds=1)


def test_run_from_config_gives_each_round_a_gate(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Rule: Customers start a round only after every seller of that round has finished starting up.

    Scenario: Each round waits for its own sellers (run wiring, detail 4: a run started
    from a config.yml with 2 rounds, pricing and channel on).
    """
    config_path = tmp_path / "input.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": str(MEXICAN_3_9),
                "experiment_name": "gate_test",
                "rounds": 2,
                "seller_pricing": True,
                "seller_channel": True,
            }
        )
    )

    run_cli(monkeypatch, "run", "--config", str(config_path), "--server-port", "18999")

    assert_one_gate_per_round(fake_run, rounds=2)

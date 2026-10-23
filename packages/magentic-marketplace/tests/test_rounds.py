"""Run-orchestration and CLI tests for the Rounds feature (tests/features/rounds.feature).

The real `run_marketplace_experiment` runs with fakes in place of `CustomerAgent`,
`BusinessAgent`, `MarketplaceLauncher` and `AgentLauncher`, so no Postgres, server or
LLM is needed. The fake launcher exposes a fake database controller at
`server.state.database_controller`; each fake round appends action rows to it, with
gaps between row indexes so "highest row_index" differs from "number of rows".
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
from magentic_marketplace.experiments.run_config import BudgetConfig
from magentic_marketplace.experiments.utils import load_customers_from_yaml
from magentic_marketplace.platform.database.models import ActionRow, ActionRowData
from magentic_marketplace.platform.database.queries import RangeQueryParams
from magentic_marketplace.platform.shared.models import (
    ActionExecutionRequest,
    ActionExecutionResult,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
CUSTOMER_IDS = sorted(c.id for c in load_customers_from_yaml(MEXICAN_3_9 / "customers"))
MULTIPLIERS = [1.25, 1.5, 2.0]

# Each fake round writes this many action rows, row indexes spaced by ROW_GAP
ROWS_PER_ROUND = 4
ROW_GAP = 3


# Fakes


class FakeActionTable:
    """In-memory stand-in for the `actions` table, ordered by row index."""

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
        """Return the number of rows (not the highest row index)."""
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
        raise NotImplementedError(
            "FakeDatabaseController.execute: read the actions table through "
            "actions.get_all/find instead"
        )


class FakeCustomerAgent:
    """Records the customer and keyword arguments a CustomerAgent was built with."""

    def __init__(self, customer, base_url, **kwargs):
        """Record constructor arguments."""
        self.customer = customer
        self.kwargs = kwargs


class FakeBusinessAgent:
    """Records the business and keyword arguments a BusinessAgent was built with."""

    def __init__(self, business, base_url, **kwargs):
        """Record constructor arguments."""
        self.business = business
        self.kwargs = kwargs


class FakeMarketplaceLauncher:
    """Async context manager that starts nothing and exposes a fake database."""

    database = FakeDatabaseController()
    enter_count = 0

    def __init__(self, **kwargs):
        """Store a fake server URL and a server whose state holds the fake database."""
        self.protocol = kwargs.get("protocol")
        self.server_url = "http://fake.invalid"
        self.server = SimpleNamespace(
            state=SimpleNamespace(database_controller=FakeMarketplaceLauncher.database)
        )

    async def __aenter__(self):
        """Count server starts."""
        FakeMarketplaceLauncher.enter_count += 1
        return self

    async def __aexit__(self, *exc):
        """Stop nothing."""
        return False

    async def create_logger(self, name):
        """Return a mock logger."""
        return Mock()


class FakeAgentLauncher:
    """Records each cohort it runs and writes that cohort's action rows."""

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
        """Record the cohort, then append this round's rows to the fake actions table."""
        database = FakeMarketplaceLauncher.database
        cohort_number = len(FakeAgentLauncher.runs) + 1
        FakeAgentLauncher.runs.append(
            {
                "customers": list(primary_agents),
                "businesses": list(dependent_agents),
                "max_index_at_start": database.max_action_index(),
            }
        )
        start = database.max_action_index() or 0
        for i in range(1, ROWS_PER_ROUND + 1):
            database.actions.append(start + i * ROW_GAP, cohort_number)


@pytest.fixture
def fake_run(monkeypatch):
    """Patch the run's agents and launchers with fakes; return the recorded cohorts."""
    FakeAgentLauncher.runs = []
    FakeMarketplaceLauncher.database = FakeDatabaseController()
    FakeMarketplaceLauncher.enter_count = 0
    monkeypatch.setattr(run_experiment, "CustomerAgent", FakeCustomerAgent)
    monkeypatch.setattr(run_experiment, "BusinessAgent", FakeBusinessAgent)
    monkeypatch.setattr(run_experiment, "MarketplaceLauncher", FakeMarketplaceLauncher)
    monkeypatch.setattr(run_experiment, "AgentLauncher", FakeAgentLauncher)
    return FakeAgentLauncher.runs


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
    """Create research/results/rounds_test under tmp_path and make it the cwd."""
    folder = tmp_path / "research" / "results" / "rounds_test"
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


def shopping_trips(cohorts: list[dict]) -> dict[str, int]:
    """Count how many customer agents were run for each customer id."""
    trips: dict[str, int] = {}
    for cohort in cohorts:
        for agent in cohort["customers"]:
            trips[agent.customer.id] = trips.get(agent.customer.id, 0) + 1
    return trips


def round_of(agent) -> int:
    """Return the round an agent was built for (1 if not passed, as stock)."""
    return agent.kwargs.get("round_number", 1)


def floor_of(agent) -> int | None:
    """Return the message floor an agent was built with (None if not passed)."""
    return agent.kwargs.get("messages_after_index")


async def run_rounds(rounds: int, **kwargs) -> None:
    """Run the real run function on mexican_3_9 with a fixed port."""
    await run_experiment.run_marketplace_experiment(
        data_dir=MEXICAN_3_9,
        experiment_name="rounds_test",
        server_port=18999,
        rounds=rounds,
        **kwargs,
    )


# Rule: A run without a rounds setting runs one round, as stock


def test_config_without_rounds_passes_one_round(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Scenario: No rounds setting gives a single round (CLI; passes on the stub)."""
    # Given a config.yml for "mexican_3_9" with no rounds setting
    config_path = write_config(
        tmp_path / "input.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "rounds_test"},
    )

    # When a run is started from that config
    run_cli(monkeypatch, "run", "--config", str(config_path))

    # Then the run gets one round
    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("rounds", 1) == 1


def test_stock_command_passes_one_round(monkeypatch, run_folder, recorded_runs):
    """Scenario: No rounds setting gives a single round (stock command; passes on the stub)."""
    run_cli(monkeypatch, "run", str(MEXICAN_3_9), "--experiment-name", "rounds_test")

    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("rounds", 1) == 1


def test_no_rounds_setting_every_customer_shops_once(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Scenario: No rounds setting gives a single round (passes on the stub)."""
    # Given a config.yml for "mexican_3_9" with no rounds setting
    config_path = write_config(
        tmp_path / "input.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "rounds_test"},
    )

    # When the run finishes
    run_cli(monkeypatch, "run", "--config", str(config_path), "--server-port", "18999")

    # Then every customer has shopped exactly once
    assert shopping_trips(fake_run) == dict.fromkeys(CUSTOMER_IDS, 1)
    # And the single cohort is stock: round 1, no message floor
    assert len(fake_run) == 1
    for agent in fake_run[0]["customers"] + fake_run[0]["businesses"]:
        assert round_of(agent) == 1
        assert floor_of(agent) is None


async def test_one_round_run_function_is_stock(run_folder, fake_run):
    """Scenario: No rounds setting gives a single round (run function, rounds=1; passes on the stub)."""
    await run_rounds(1)

    assert len(fake_run) == 1
    assert shopping_trips(fake_run) == dict.fromkeys(CUSTOMER_IDS, 1)
    assert len(fake_run[0]["businesses"]) == 9
    for agent in fake_run[0]["customers"] + fake_run[0]["businesses"]:
        assert round_of(agent) == 1
        assert floor_of(agent) is None


# Rule: Each round, every customer shops once, fresh, with the same order and budget


def test_config_with_rounds_reaches_the_run(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Scenario: Five rounds means five shopping trips per customer (config reaches the run)."""
    config_path = write_config(
        tmp_path / "input.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "rounds_test", "rounds": 5},
    )

    run_cli(monkeypatch, "run", "--config", str(config_path))

    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("rounds") == 5


def test_five_rounds_means_five_shopping_trips_per_customer(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Scenario: Five rounds means five shopping trips per customer."""
    # Given a config.yml for "mexican_3_9" with 5 rounds
    config_path = write_config(
        tmp_path / "input.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "rounds_test", "rounds": 5},
    )

    # When the run finishes
    run_cli(monkeypatch, "run", "--config", str(config_path), "--server-port", "18999")

    # Then every customer has shopped exactly 5 times
    assert shopping_trips(fake_run) == dict.fromkeys(CUSTOMER_IDS, 5)


async def test_each_round_runs_one_cohort_of_every_customer_and_seller(
    run_folder, fake_run
):
    """Scenario: Five rounds means five shopping trips per customer (one cohort per round, in order)."""
    await run_rounds(5)

    # One server for the whole run
    assert FakeMarketplaceLauncher.enter_count == 1
    assert len(fake_run) == 5
    for r, cohort in enumerate(fake_run, start=1):
        assert sorted(a.customer.id for a in cohort["customers"]) == CUSTOMER_IDS
        assert len(cohort["businesses"]) == 9
        assert all(round_of(a) == r for a in cohort["customers"])
        assert all(round_of(a) == r for a in cohort["businesses"])


async def test_every_round_builds_fresh_agents(run_folder, fake_run):
    """Scenario: A customer starts each round with no memory of earlier rounds (fresh agent objects)."""
    await run_rounds(3)

    assert len(fake_run) == 3
    seen_ids: set[int] = set()
    for cohort in fake_run:
        for agent in cohort["customers"] + cohort["businesses"]:
            assert id(agent) not in seen_ids
            seen_ids.add(id(agent))


async def test_round_two_floor_is_highest_action_index_when_round_one_ended(
    run_folder, fake_run
):
    """Scenario: A customer starts each round with no memory of earlier rounds (message floor per round)."""
    await run_rounds(3)

    assert len(fake_run) == 3
    # Round 1 has no floor
    for agent in fake_run[0]["customers"] + fake_run[0]["businesses"]:
        assert floor_of(agent) is None
    # Round r >= 2 gets the highest action row index at the end of round r-1
    for cohort in fake_run[1:]:
        expected_floor = cohort["max_index_at_start"]
        assert expected_floor is not None
        for agent in cohort["customers"] + cohort["businesses"]:
            assert floor_of(agent) == expected_floor
    # Gaps between row indexes make "highest index" differ from "row count"
    assert fake_run[1]["max_index_at_start"] == ROWS_PER_ROUND * ROW_GAP
    assert fake_run[2]["max_index_at_start"] == 2 * ROWS_PER_ROUND * ROW_GAP


async def test_customer_budget_is_the_same_in_every_round(run_folder, fake_run):
    """Scenario: A customer's budget is the same in every round."""
    # Given a config.yml with 3 rounds and budgets on
    await run_rounds(3, budget=BudgetConfig(multipliers=MULTIPLIERS, seed=7))

    # When customer "customer_0001" has a budget in round 1
    saved = yaml.safe_load((run_folder / "budgets.yml").read_text())
    by_id = {row["customer_id"]: row["budget"] for row in saved}
    assert len(fake_run) == 3

    # Then its budget is the same in rounds 2 and 3 (for every customer)
    for cohort in fake_run:
        for agent in cohort["customers"]:
            assert agent.kwargs.get("budget") == by_id[agent.customer.id]


def test_budget_and_rounds_from_config_give_same_budget_every_round(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Scenario: A customer's budget is the same in every round (through the CLI config)."""
    config_path = write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "rounds_test",
            "rounds": 3,
            "budget": {"multipliers": MULTIPLIERS, "seed": 7},
        },
    )

    run_cli(monkeypatch, "run", "--config", str(config_path), "--server-port", "18999")

    assert len(fake_run) == 3
    budgets_by_round = [
        {a.customer.id: a.kwargs.get("budget") for a in cohort["customers"]}
        for cohort in fake_run
    ]
    assert all(b is not None for b in budgets_by_round[0].values())
    assert budgets_by_round[1] == budgets_by_round[0]
    assert budgets_by_round[2] == budgets_by_round[0]

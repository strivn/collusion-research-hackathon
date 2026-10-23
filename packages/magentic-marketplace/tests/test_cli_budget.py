"""Run-wiring tests for the Customer budget feature.

Rule: Budgets come from the run config; a config without a budget section gives no
budgets.

The real `run_marketplace_experiment` runs with fakes in place of `CustomerAgent`,
`BusinessAgent`, `MarketplaceLauncher` and `AgentLauncher`, so no Postgres, server or
LLM is needed.
"""

import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from magentic_marketplace import cli
from magentic_marketplace.experiments import run_experiment
from magentic_marketplace.experiments.run_config import BudgetConfig
from magentic_marketplace.experiments.utils import load_customers_from_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
MULTIPLIERS = [1.25, 1.5, 2.0]


class FakeCustomerAgent:
    """Records the customer and keyword arguments a CustomerAgent was built with."""

    def __init__(self, customer, base_url, **kwargs):
        """Record constructor arguments."""
        self.customer = customer
        self.kwargs = kwargs


class FakeBusinessAgent:
    """Stands in for BusinessAgent."""

    def __init__(self, business, base_url, **kwargs):
        """Record the business."""
        self.business = business


class FakeMarketplaceLauncher:
    """Async context manager that starts nothing."""

    def __init__(self, **kwargs):
        """Store a fake server URL."""
        self.protocol = kwargs.get("protocol")
        self.server_url = "http://fake.invalid"
        self.server = None

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
    """Records the agents it was asked to run and what the run folder held then."""

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
        """Record the customer agents and whether budgets.yml existed at run time."""
        FakeAgentLauncher.runs.append(
            {
                "customers": primary_agents,
                "budgets_file_existed": (Path.cwd() / "budgets.yml").exists(),
            }
        )


@pytest.fixture
def fake_run(monkeypatch):
    """Patch the run's agents and launchers with fakes; return the recorded runs."""
    FakeAgentLauncher.runs = []
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
    """Create research/results/budget_test under tmp_path and make it the cwd."""
    folder = tmp_path / "research" / "results" / "budget_test"
    folder.mkdir(parents=True)
    monkeypatch.chdir(folder)
    return folder


def write_config(path: Path, content: dict) -> Path:
    """Write `content` as YAML to `path` and return the path."""
    path.write_text(yaml.safe_dump(content))
    return path


def run_cli(monkeypatch, *args: str) -> None:
    """Run `magentic-marketplace <args>` in-process."""
    monkeypatch.setattr(sys, "argv", ["magentic-marketplace", *args])
    cli.main()


def agent_budget(agent: FakeCustomerAgent) -> float | None:
    """Return the budget a CustomerAgent was built with (None if not passed)."""
    return agent.kwargs.get("budget")


def expected_budget(menu_features: dict[str, float], m: float) -> float:
    """Spec arithmetic: m x sum(menu_features), rounded to cents half up."""
    reference = sum(Decimal(str(price)) for price in menu_features.values())
    exact = reference * Decimal(str(m))
    return float(exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def assert_budgets_written_and_given(folder: Path, fake_run: list[dict]) -> None:
    """Check budgets.yml in `folder` and that each customer agent got its budget."""
    budgets_path = folder / "budgets.yml"
    assert budgets_path.is_file()
    saved = yaml.safe_load(budgets_path.read_text())
    customers = {c.id: c for c in load_customers_from_yaml(MEXICAN_3_9 / "customers")}
    assert [row["customer_id"] for row in saved] == sorted(customers)
    for row in saved:
        assert set(row) == {"customer_id", "m", "budget"}
        assert row["m"] in MULTIPLIERS
        menu = customers[row["customer_id"]].menu_features
        assert row["budget"] == expected_budget(menu, row["m"])

    # The file is written before any agent runs, and each agent gets its budget
    assert len(fake_run) == 1
    assert fake_run[0]["budgets_file_existed"] is True
    by_id = {row["customer_id"]: row["budget"] for row in saved}
    agents = fake_run[0]["customers"]
    assert len(agents) == 3
    for agent in agents:
        assert agent_budget(agent) == by_id[agent.customer.id]


# Scenario: A config without a budget section gives every customer no budget


def test_config_without_budget_section_passes_no_budget(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Scenario: A config without a budget section gives every customer no budget (CLI; passes on the stub)."""
    # Given a config.yml for "mexican_3_9" with no budget section
    config_path = write_config(
        tmp_path / "input.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "budget_test"},
    )

    # When a run is started from that config
    run_cli(monkeypatch, "run", "--config", str(config_path))

    # Then the run gets no budget
    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("budget") is None


def test_config_without_budget_section_gives_every_customer_no_budget(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Scenario: A config without a budget section gives every customer no budget (passes on the stub)."""
    # Given a config.yml for "mexican_3_9" with no budget section
    config_path = write_config(
        tmp_path / "input.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "budget_test"},
    )

    # When a run is started from that config
    run_cli(monkeypatch, "run", "--config", str(config_path), "--server-port", "18999")

    # Then no customer has a budget
    assert len(fake_run) == 1
    customers = fake_run[0]["customers"]
    assert len(customers) == 3
    assert all(agent_budget(agent) is None for agent in customers)
    assert not (run_folder / "budgets.yml").exists()


def test_stock_command_gives_every_customer_no_budget(
    monkeypatch, run_folder, fake_run
):
    """Rule: A customer without a budget behaves exactly as stock (stock command; passes on the stub)."""
    run_cli(
        monkeypatch,
        "run",
        str(MEXICAN_3_9),
        "--experiment-name",
        "budget_test",
        "--server-port",
        "18999",
    )

    assert len(fake_run) == 1
    assert all(agent_budget(agent) is None for agent in fake_run[0]["customers"])
    assert not (run_folder / "budgets.yml").exists()


# Scenario: The budgets a run used are written to the run folder


def test_budget_section_reaches_the_run(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Scenario: The budgets a run used are written to the run folder (config reaches the run)."""
    config_path = write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "budget_test",
            "budget": {"multipliers": MULTIPLIERS, "seed": 7},
        },
    )

    run_cli(monkeypatch, "run", "--config", str(config_path))

    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("budget") == BudgetConfig(
        multipliers=MULTIPLIERS, seed=7
    )


def test_budgets_are_written_to_the_run_folder(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Scenario: The budgets a run used are written to the run folder."""
    # Given a config.yml for "mexican_3_9" with budget multipliers 1.25, 1.5, 2.0
    # and seed 7
    config_path = write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "budget_test",
            "budget": {"multipliers": MULTIPLIERS, "seed": 7},
        },
    )

    # When a run is started from that config in folder "research/results/budget_test"
    run_cli(monkeypatch, "run", "--config", str(config_path), "--server-port", "18999")

    # Then "research/results/budget_test/budgets.yml" lists each customer's id,
    # m and budget
    assert_budgets_written_and_given(
        tmp_path / "research" / "results" / "budget_test", fake_run
    )


async def test_run_with_budget_writes_budgets_and_gives_each_agent_its_budget(
    run_folder, fake_run
):
    """Scenario: The budgets a run used are written to the run folder (run function, no CLI)."""
    await run_experiment.run_marketplace_experiment(
        data_dir=MEXICAN_3_9,
        experiment_name="budget_test",
        server_port=18999,
        budget=BudgetConfig(multipliers=MULTIPLIERS, seed=7),
    )

    assert_budgets_written_and_given(run_folder, fake_run)

"""Run-wiring tests for the Seller pricing feature.

The CLI tests replace `run_marketplace_experiment` with a recording fake. The run
tests call the real `run_marketplace_experiment` with fakes in place of
`CustomerAgent`, `BusinessAgent`, `MarketplaceLauncher` and `AgentLauncher`, so no
Postgres, server or LLM is needed.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from magentic_marketplace import cli
from magentic_marketplace.experiments import run_experiment

REPO_ROOT = Path(__file__).resolve().parents[3]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"


class FakeCustomerAgent:
    """Stands in for CustomerAgent."""

    def __init__(self, customer, base_url, **kwargs):
        """Record the customer."""
        self.customer = customer


class FakeBusinessAgent:
    """Records the business and keyword arguments a BusinessAgent was built with."""

    def __init__(self, business, base_url, *args, **kwargs):
        """Record constructor arguments."""
        self.business = business
        self.args = args
        self.kwargs = kwargs


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
    """Records the business agents it was asked to run."""

    runs: list[list[FakeBusinessAgent]] = []

    def __init__(self, server_url):
        """Accept the server URL."""

    async def __aenter__(self):
        """Start nothing."""
        return self

    async def __aexit__(self, *exc):
        """Stop nothing."""
        return False

    async def run_agents_with_dependencies(self, primary_agents, dependent_agents):
        """Record the business agents."""
        FakeAgentLauncher.runs.append(dependent_agents)


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
    """Create research/results/pricing_test under tmp_path and make it the cwd."""
    folder = tmp_path / "research" / "results" / "pricing_test"
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


def seller_pricing_of(agent: FakeBusinessAgent) -> bool:
    """Return the seller_pricing a BusinessAgent was built with (False if not passed)."""
    return agent.kwargs.get("seller_pricing", False)


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


async def test_run_with_pricing_off_builds_sellers_without_pricing(
    run_folder, fake_run
):
    """Scenario: With pricing off, no pricing step runs (run -> sellers; passes on the stub)."""
    await run_experiment.run_marketplace_experiment(
        data_dir=MEXICAN_3_9,
        experiment_name="pricing_test",
        server_port=18999,
    )

    assert len(fake_run) == 1
    sellers = fake_run[0]
    assert len(sellers) == 9
    assert all(seller_pricing_of(agent) is False for agent in sellers)


def test_stock_command_builds_sellers_without_pricing(
    monkeypatch, run_folder, fake_run
):
    """Rule: With pricing off, sellers behave exactly as stock (stock command; passes on the stub)."""
    run_cli(
        monkeypatch,
        "run",
        str(MEXICAN_3_9),
        "--experiment-name",
        "pricing_test",
        "--server-port",
        "18999",
    )

    assert len(fake_run) == 1
    assert all(seller_pricing_of(agent) is False for agent in fake_run[0])


# Rule: With pricing on, each seller sets a price for every menu item before
# serving customers


def test_config_with_pricing_on_reaches_the_run(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Scenario: The price list is set before the seller answers any customer (config turns pricing on)."""
    # Given a config.yml with seller pricing on
    config_path = write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "pricing_test",
            "seller_pricing": True,
        },
    )

    # When a run starts
    run_cli(monkeypatch, "run", "--config", str(config_path))

    # Then the run has pricing on
    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("seller_pricing") is True


async def test_run_with_pricing_on_builds_every_seller_with_pricing(
    run_folder, fake_run
):
    """Scenario: The price list is set before the seller answers any customer (every seller gets pricing on)."""
    await run_experiment.run_marketplace_experiment(
        data_dir=MEXICAN_3_9,
        experiment_name="pricing_test",
        server_port=18999,
        seller_pricing=True,
    )

    assert len(fake_run) == 1
    sellers = fake_run[0]
    assert len(sellers) == 9
    assert all(seller_pricing_of(agent) is True for agent in sellers)


def test_config_with_pricing_on_builds_every_seller_with_pricing(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Scenario: The price list is set before the seller answers any customer (config through CLI to sellers)."""
    config_path = write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "pricing_test",
            "seller_pricing": True,
        },
    )

    run_cli(monkeypatch, "run", "--config", str(config_path), "--server-port", "18999")

    assert len(fake_run) == 1
    assert all(seller_pricing_of(agent) is True for agent in fake_run[0])

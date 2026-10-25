"""Config and run-wiring tests for the minimal talk prompt switch.

`minimal_talk_prompt: true` in the run config builds every seller of every round with
`minimal_talk_prompt=True`; unset, it is false and the standard talk prompt stays. The
prompt text itself is tested in tests/marketplace/agents/test_talk_phase.py. The fakes
are those of tests/test_cli_informant.py.
"""

from pathlib import Path

import pytest
import yaml
from test_cli_informant import (
    MEXICAN_3_9,
    SELLER_IDS,
    FakeAgentLauncher,
    FakeBusinessAgent,
    FakeCustomerAgent,
    FakeMarketplaceLauncher,
    make_run_folder,
    reset_fakes,
    run,
    run_cli,
    write_config,
)

from magentic_marketplace import cli
from magentic_marketplace.experiments import run_experiment
from magentic_marketplace.experiments.run_config import load_run_config

# Fixtures (as in tests/test_cli_informant.py)


@pytest.fixture(autouse=True)
def empty_seller_table(monkeypatch):
    """Serve an empty seller table between rounds; the fake database cannot build one."""

    async def no_rows(database):
        return [], []

    monkeypatch.setattr(run_experiment, "compute_round_tables", no_rows)


@pytest.fixture
def fake_run(monkeypatch):
    """Patch the run's agents and launchers with fakes; return the recorded cohorts."""
    reset_fakes()
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
    """Create research/results/minimal_talk_test under tmp_path and make it the cwd."""
    folder = make_run_folder(tmp_path, "minimal_talk_test")
    monkeypatch.chdir(folder)
    return folder


# Helpers


def config_path(tmp_path: Path, **settings) -> Path:
    """Write a 3-round mexican_3_9 Pricing + Messaging config with `settings`."""
    return write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "minimal_talk_test",
            "rounds": 3,
            "seller_pricing": True,
            "seller_channel": True,
            **settings,
        },
    )


def minimal_flags(cohort: dict) -> dict[str, bool]:
    """Return each seller's `minimal_talk_prompt` constructor argument in one round."""
    return {
        agent.business.id: agent.kwargs.get("minimal_talk_prompt", False)
        for agent in cohort["businesses"]
    }


# Config


def test_config_without_the_switch_keeps_the_standard_talk_prompt(tmp_path):
    """Minimal talk prompt: unset means false."""
    assert load_run_config(config_path(tmp_path)).minimal_talk_prompt is False


def test_config_with_the_switch_reaches_the_run(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Minimal talk prompt: `minimal_talk_prompt: true` is passed to the run."""
    path = config_path(tmp_path, minimal_talk_prompt=True)

    run_cli(monkeypatch, "run", "--config", str(path))

    assert recorded_runs[0].get("minimal_talk_prompt") is True


def test_config_yml_records_the_switch(monkeypatch, tmp_path, run_folder, fake_run):
    """Minimal talk prompt: the run folder's config.yml records the switch."""
    path = config_path(tmp_path, minimal_talk_prompt=True)

    run_cli(monkeypatch, "run", "--config", str(path), "--server-port", "18999")

    saved = yaml.safe_load((run_folder / "config.yml").read_text())
    assert saved["minimal_talk_prompt"] is True


# Run


@pytest.mark.parametrize("switch", [True, False])
async def test_every_seller_of_every_round_is_built_with_the_switch(
    run_folder, fake_run, switch
):
    """Minimal talk prompt: every seller in rounds 1-3 gets the run's switch."""
    await run(rounds=3, minimal_talk_prompt=switch)

    assert len(fake_run) == 3
    for cohort in fake_run:
        assert minimal_flags(cohort) == dict.fromkeys(SELLER_IDS, switch)

"""Config and run-wiring tests for the random first word switch.

`random_first_word: {seed: s}` in the run config builds every seller of every round with
`first_word_seed=s`, so each talk call is told to start with a word drawn from the seed,
round, turn and seller. Unset, it is None and the talk prompt is unchanged. The draw,
the prompt line and the `first_word` log rows are tested in
tests/marketplace/agents/test_talk_phase.py. The fakes are those of
tests/test_cli_informant.py.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
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
from magentic_marketplace.experiments.run_config import (
    FirstWordConfig,
    load_run_config,
)

SEED = 5


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
    """Create research/results/first_word_test under tmp_path and make it the cwd."""
    folder = make_run_folder(tmp_path, "first_word_test")
    monkeypatch.chdir(folder)
    return folder


# Helpers


def config_path(tmp_path: Path, **settings) -> Path:
    """Write a 3-round mexican_3_9 Pricing + Messaging config with `settings`."""
    return write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "first_word_test",
            "rounds": 3,
            "seller_pricing": True,
            "seller_channel": True,
            **settings,
        },
    )


def first_word_seeds(cohort: dict) -> dict[str, object]:
    """Return each seller's `first_word_seed` constructor argument in one round.

    A seller built without the argument shows the marker "missing", so a run that never
    passes it fails the check for None as well.
    """
    return {
        agent.business.id: agent.kwargs.get("first_word_seed", "missing")
        for agent in cohort["businesses"]
    }


# Config


def test_config_without_the_switch_has_no_first_word(tmp_path):
    """Random first word: unset means None (talk prompt unchanged)."""
    assert load_run_config(config_path(tmp_path)).random_first_word is None


def test_config_with_a_seed_parses(tmp_path):
    """Random first word: `{seed: 5}` parses into FirstWordConfig(seed=5)."""
    config = load_run_config(config_path(tmp_path, random_first_word={"seed": 5}))

    assert config.random_first_word == FirstWordConfig(seed=5)


@pytest.mark.parametrize(
    "setting", [{"seed": 5, "words": ["Hi"]}, {}], ids=["unknown-key", "no-seed"]
)
def test_config_rejects_a_bad_first_word_setting(tmp_path, setting):
    """Random first word: an unknown key inside or a missing seed is rejected."""
    with pytest.raises(ValidationError):
        load_run_config(config_path(tmp_path, random_first_word=setting))


def test_config_with_the_switch_reaches_the_run(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Random first word: `run --config` passes the setting to the run as random_first_word."""
    path = config_path(tmp_path, random_first_word={"seed": SEED})

    run_cli(monkeypatch, "run", "--config", str(path))

    assert recorded_runs[0]["random_first_word"] == FirstWordConfig(seed=SEED)


def test_config_without_the_switch_reaches_the_run_as_none(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Random first word: a config without it passes None to the run."""
    run_cli(monkeypatch, "run", "--config", str(config_path(tmp_path)))

    assert recorded_runs[0]["random_first_word"] is None


def test_stock_command_runs_without_a_first_word(
    monkeypatch, run_folder, recorded_runs
):
    """Random first word: the stock command, without a config, passes None."""
    run_cli(
        monkeypatch, "run", str(MEXICAN_3_9), "--experiment-name", "first_word_test"
    )

    assert recorded_runs[0]["random_first_word"] is None


@pytest.mark.parametrize("setting", [{"seed": SEED}, None])
def test_config_yml_records_the_switch(
    monkeypatch, tmp_path, run_folder, fake_run, setting
):
    """Random first word: the run folder's config.yml records the setting."""
    extra = {} if setting is None else {"random_first_word": setting}
    path = config_path(tmp_path, **extra)

    run_cli(monkeypatch, "run", "--config", str(path), "--server-port", "18999")

    saved = yaml.safe_load((run_folder / "config.yml").read_text())
    assert "random_first_word" in saved
    assert saved["random_first_word"] == setting


# Run


async def test_every_seller_of_every_round_gets_the_seed(run_folder, fake_run):
    """Random first word: every seller in rounds 1-3 is built with first_word_seed = seed."""
    await run(rounds=3, random_first_word=FirstWordConfig(seed=SEED))

    assert len(fake_run) == 3
    for cohort in fake_run:
        assert first_word_seeds(cohort) == dict.fromkeys(SELLER_IDS, SEED)


async def test_without_the_switch_every_seller_gets_none(run_folder, fake_run):
    """Random first word: without it every seller in rounds 1-3 is built with first_word_seed=None."""
    await run(rounds=3)

    assert len(fake_run) == 3
    for cohort in fake_run:
        assert first_word_seeds(cohort) == dict.fromkeys(SELLER_IDS, None)


def test_config_seed_reaches_every_seller_through_the_cli(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Random first word: `run --config` with `{seed: 5}` builds every seller with seed 5."""
    path = config_path(tmp_path, random_first_word={"seed": SEED})

    run_cli(monkeypatch, "run", "--config", str(path), "--server-port", "18999")

    assert len(fake_run) == 3
    for cohort in fake_run:
        assert first_word_seeds(cohort) == dict.fromkeys(SELLER_IDS, SEED)

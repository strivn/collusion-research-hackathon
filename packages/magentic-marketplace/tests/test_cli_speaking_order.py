"""Config and run-wiring tests for the random speaking order switch.

`random_speaking_order: {seed: s}` in the run config makes the sellers of each talk turn
speak one at a time, in an order drawn from `speaking_order_rng(s, round)`, and logs one
`speaking_order` row per turn. Unset, it is None and turns stay concurrent. The turn
logic is tested in tests/marketplace/agents/test_talk_phase.py. The fakes are those of
tests/test_cli_informant.py; the run's logger is replaced by a recording one.
"""

import random
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError
from test_cli_informant import (
    MEXICAN_3_9,
    POBLANO,
    PROMPTS,
    SELLER_4,
    SELLER_IDS,
    TALK_SCRIPT,
    FakeAgentLauncher,
    FakeBusinessAgent,
    FakeCustomerAgent,
    FakeMarketplaceLauncher,
    RecordingLogger,
    TalkingBusinessAgent,
    make_run_folder,
    reset_fakes,
    run_cli,
    write_config,
)

from magentic_marketplace import cli
from magentic_marketplace.experiments import run_experiment
from magentic_marketplace.experiments.run_config import (
    SpeakingOrderConfig,
    load_run_config,
)
from magentic_marketplace.marketplace.agents.business.models import SpeakingOrderLog
from magentic_marketplace.marketplace.agents.business.talk import TalkPhase

SEED = 5


# Fakes


class LoggingMarketplaceLauncher(FakeMarketplaceLauncher):
    """FakeMarketplaceLauncher whose run logger records every call in `logger`."""

    logger = RecordingLogger()

    async def create_logger(self, name):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return the shared recording logger."""
        return LoggingMarketplaceLauncher.logger


class RecordingTalkPhase(TalkPhase):
    """A real TalkPhase that records each instance and its rng's state when built."""

    built: list["RecordingTalkPhase"] = []

    def __init__(self, seller_ids, *args, **kwargs):
        """Build the real phase; remember the rng and its starting state."""
        super().__init__(seller_ids, *args, **kwargs)
        rng = kwargs.get("speaking_order_rng", args[0] if args else None)
        self.rng = rng
        self.rng_state = None if rng is None else rng.getstate()
        RecordingTalkPhase.built.append(self)


# Fixtures (as in tests/test_cli_informant.py)


@pytest.fixture(autouse=True)
def empty_seller_table(monkeypatch):
    """Serve an empty seller table between rounds; the fake database cannot build one."""

    async def no_rows(database):
        return [], []

    monkeypatch.setattr(run_experiment, "compute_round_tables", no_rows)


@pytest.fixture
def fake_run(monkeypatch):
    """Patch the run with fakes, a recording logger and a recording TalkPhase."""
    reset_fakes()
    TALK_SCRIPT.clear()
    LoggingMarketplaceLauncher.logger = RecordingLogger()
    RecordingTalkPhase.built = []
    monkeypatch.setattr(run_experiment, "CustomerAgent", FakeCustomerAgent)
    monkeypatch.setattr(run_experiment, "BusinessAgent", FakeBusinessAgent)
    monkeypatch.setattr(
        run_experiment, "MarketplaceLauncher", LoggingMarketplaceLauncher
    )
    monkeypatch.setattr(run_experiment, "AgentLauncher", FakeAgentLauncher)
    monkeypatch.setattr(run_experiment, "TalkPhase", RecordingTalkPhase)
    return FakeAgentLauncher.runs


@pytest.fixture
def real_sellers(monkeypatch, fake_run):
    """Like fake_run, but sellers are TalkingBusinessAgents that really talk."""
    monkeypatch.setattr(run_experiment, "BusinessAgent", TalkingBusinessAgent)
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
    """Create research/results/speaking_order_test under tmp_path and make it the cwd."""
    folder = make_run_folder(tmp_path, "speaking_order_test")
    monkeypatch.chdir(folder)
    return folder


# Helpers


def config_path(tmp_path: Path, **settings) -> Path:
    """Write a 3-round mexican_3_9 Pricing + Messaging config with `settings`."""
    return write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "speaking_order_test",
            "rounds": 3,
            "seller_pricing": True,
            "seller_channel": True,
            **settings,
        },
    )


async def run(rounds: int = 3, seller_channel: bool = True, **kwargs) -> None:
    """Run the real run function on mexican_3_9 with pricing on."""
    await run_experiment.run_marketplace_experiment(
        data_dir=MEXICAN_3_9,
        experiment_name="speaking_order_test",
        server_port=18999,
        rounds=rounds,
        seller_pricing=True,
        seller_channel=seller_channel,
        **kwargs,
    )


def speaking_order_rows() -> list[SpeakingOrderLog]:
    """Return the run logger's speaking_order entries, in logging order."""
    rows = []
    for _, _, data in LoggingMarketplaceLauncher.logger.entries:
        if isinstance(data, BaseModel):
            data = data.model_dump(mode="json")
        if isinstance(data, dict) and data.get("type") == "speaking_order":
            rows.append(SpeakingOrderLog.model_validate(data))
    return rows


def shuffled(seller_ids: list[str], rng: random.Random) -> list[str]:
    """Return `seller_ids` sorted, then shuffled by `rng`: the spec's speaking order."""
    order = sorted(seller_ids)
    rng.shuffle(order)
    return order


# Config


def test_config_without_the_switch_has_no_speaking_order(tmp_path):
    """Random speaking order: unset means None (concurrent turns)."""
    assert load_run_config(config_path(tmp_path)).random_speaking_order is None


def test_config_with_a_seed_parses(tmp_path):
    """Random speaking order: `{seed: 5}` parses into SpeakingOrderConfig(seed=5)."""
    config = load_run_config(config_path(tmp_path, random_speaking_order={"seed": 5}))

    assert config.random_speaking_order == SpeakingOrderConfig(seed=5)


@pytest.mark.parametrize(
    "setting", [{"seed": 5, "order": "random"}, {}], ids=["unknown-key", "no-seed"]
)
def test_config_rejects_a_bad_speaking_order(tmp_path, setting):
    """Random speaking order: an unknown key inside or a missing seed is rejected."""
    with pytest.raises(ValidationError):
        load_run_config(config_path(tmp_path, random_speaking_order=setting))


def test_config_with_the_switch_reaches_the_run(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Random speaking order: `run --config` passes the setting to the run."""
    path = config_path(tmp_path, random_speaking_order={"seed": SEED})

    run_cli(monkeypatch, "run", "--config", str(path))

    assert recorded_runs[0]["random_speaking_order"] == SpeakingOrderConfig(seed=SEED)


def test_config_without_the_switch_reaches_the_run_as_none(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Random speaking order: a config without it passes None to the run."""
    run_cli(monkeypatch, "run", "--config", str(config_path(tmp_path)))

    assert recorded_runs[0]["random_speaking_order"] is None


def test_stock_command_runs_without_a_speaking_order(
    monkeypatch, run_folder, recorded_runs
):
    """Random speaking order: the stock command, without a config, passes None."""
    run_cli(
        monkeypatch, "run", str(MEXICAN_3_9), "--experiment-name", "speaking_order_test"
    )

    assert recorded_runs[0]["random_speaking_order"] is None


@pytest.mark.parametrize("setting", [{"seed": SEED}, None])
def test_config_yml_records_the_switch(
    monkeypatch, tmp_path, run_folder, fake_run, setting
):
    """Random speaking order: the run folder's config.yml records the setting."""
    extra = {} if setting is None else {"random_speaking_order": setting}
    path = config_path(tmp_path, **extra)

    run_cli(monkeypatch, "run", "--config", str(path), "--server-port", "18999")

    saved = yaml.safe_load((run_folder / "config.yml").read_text())
    assert saved["random_speaking_order"] == setting


# Run: which TalkPhases get an rng


async def test_rounds_from_2_get_the_rounds_rng(run_folder, fake_run):
    """Random speaking order: round r's TalkPhase gets speaking_order_rng(seed, r)."""
    await run(rounds=3, random_speaking_order=SpeakingOrderConfig(seed=SEED))

    # Rounds 2 and 3 talk; round 1 has no phase
    phases = RecordingTalkPhase.built
    assert len(phases) == 2
    for phase, round_number in zip(phases, (2, 3), strict=True):
        assert phase.rng is not None
        assert phase.rng_state == random.Random(f"{SEED}:{round_number}").getstate()
    # Each round's sellers share that round's phase
    assert [
        {agent.kwargs["talk_phase"] for agent in cohort["businesses"]}
        for cohort in fake_run
    ] == [{None}, {phases[0]}, {phases[1]}]


async def test_without_the_switch_phases_get_no_rng(run_folder, fake_run):
    """Random speaking order: without it every TalkPhase stays concurrent."""
    await run(rounds=3)

    assert len(RecordingTalkPhase.built) == 2
    assert all(phase.rng is None for phase in RecordingTalkPhase.built)


async def test_without_messaging_the_switch_builds_no_phase(run_folder, fake_run):
    """Random speaking order: with the channel off there is no talk and no order row."""
    await run(
        rounds=3,
        seller_channel=False,
        random_speaking_order=SpeakingOrderConfig(seed=SEED),
    )

    assert RecordingTalkPhase.built == []
    assert speaking_order_rows() == []


# Run: speaking_order rows, with real sellers that talk


async def test_each_turn_logs_its_speaking_order(run_folder, real_sellers):
    """Random speaking order: one speaking_order row per turn, in the order sellers spoke."""
    # Round 2: business_0001 messages business_0004 in turn 1, so turn 2 asks only it
    TALK_SCRIPT[(2, 1, POBLANO)] = ([SELLER_4], "hello")

    await run(rounds=3, random_speaking_order=SpeakingOrderConfig(seed=SEED))

    round_2_rng = random.Random(f"{SEED}:2")
    round_2_turn_1 = shuffled(SELLER_IDS, round_2_rng)
    round_2_turn_2 = shuffled([SELLER_4], round_2_rng)
    round_3_turn_1 = shuffled(SELLER_IDS, random.Random(f"{SEED}:3"))
    assert speaking_order_rows() == [
        SpeakingOrderLog(round=2, turn=1, seller_ids=round_2_turn_1),
        SpeakingOrderLog(round=2, turn=2, seller_ids=round_2_turn_2),
        SpeakingOrderLog(round=3, turn=1, seller_ids=round_3_turn_1),
    ]
    # The rows match the phases' own records
    assert [phase.speaking_orders for phase in RecordingTalkPhase.built] == [
        [round_2_turn_1, round_2_turn_2],
        [round_3_turn_1],
    ]
    # And the sellers' talk calls ran in that order
    talk_calls = {
        round_number: [
            sid for kind, r, sid, _ in PROMPTS if kind == "talk" and r == round_number
        ]
        for round_number in (2, 3)
    }
    assert talk_calls == {2: round_2_turn_1 + round_2_turn_2, 3: round_3_turn_1}


async def test_without_the_switch_no_speaking_order_is_logged(run_folder, real_sellers):
    """Random speaking order: without it no speaking_order row is logged."""
    TALK_SCRIPT[(2, 1, POBLANO)] = ([SELLER_4], "hello")

    await run(rounds=3)

    assert speaking_order_rows() == []
    assert all(phase.speaking_orders == [] for phase in RecordingTalkPhase.built)

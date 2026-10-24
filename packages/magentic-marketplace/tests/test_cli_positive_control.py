"""Config and run-wiring tests for the positive control.

Feature files: tests/features/positive_control.feature and
positive_control.technical.feature. Covers the Rule "The switch requires the seller
channel", the run level of the Rule "With the switch on, every seller except the
undercover one sees the instruction at the top of its talk and pricing prompts" (a
Positive control run's prompts against the same Undercover run), and the technical
Rules "Every round builds every seller except the undercover one with the instruction"
and "The run folder says the run was a positive control".

"Undercover run" = channel on, seller pricing on, 3 rounds, `undercover_informant` with
seed 7. "Positive control run" = an Undercover run with `positive_control: true`. The
fakes are those of tests/test_cli_informant.py: the real `run_marketplace_experiment`
with fakes in place of `CustomerAgent`, `MarketplaceLauncher`, `AgentLauncher` and
`compute_round_tables`, and either a recording fake or a real `BusinessAgent` with a
scripted LLM and an in-memory channel. Which sellers get the instruction is read from
each built seller's `coordination` constructor argument. The undercover seller of a run
is read from its informant.yml, since the pick is the implementation's.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from test_cli_informant import (
    MEXICAN_3_9,
    POBLANO,
    PROMPTS,
    SEED,
    SELLER_4,
    SELLER_IDS,
    TALK_SCRIPT,
    FakeAgentLauncher,
    FakeBusinessAgent,
    FakeCustomerAgent,
    FakeMarketplaceLauncher,
    TalkingBusinessAgent,
    make_run_folder,
    prompts_by_seller,
    read_pick,
    reset_fakes,
    run,
    run_cli,
    write_config,
)

from magentic_marketplace import cli
from magentic_marketplace.experiments import run_experiment
from magentic_marketplace.experiments.run_config import (
    RunConfig,
    UndercoverConfig,
    load_run_config,
)
from magentic_marketplace.marketplace.agents.business.informant import (
    COORDINATION_INSTRUCTION,
)

PREFIX = COORDINATION_INSTRUCTION + "\n\n"
COORDINATE = "coordinate with other sellers"


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
    TALK_SCRIPT.clear()
    monkeypatch.setattr(run_experiment, "CustomerAgent", FakeCustomerAgent)
    monkeypatch.setattr(run_experiment, "BusinessAgent", FakeBusinessAgent)
    monkeypatch.setattr(run_experiment, "MarketplaceLauncher", FakeMarketplaceLauncher)
    monkeypatch.setattr(run_experiment, "AgentLauncher", FakeAgentLauncher)
    return FakeAgentLauncher.runs


@pytest.fixture
def real_sellers(monkeypatch, fake_run):
    """Like fake_run, but sellers are TalkingBusinessAgents."""
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
    """Create research/results/positive_control_test under tmp_path and make it the cwd."""
    folder = make_run_folder(tmp_path, "positive_control_test")
    monkeypatch.chdir(folder)
    return folder


# Helpers


def config_dict(**settings) -> dict:
    """Return a 3-round mexican_3_9 config with pricing on and `settings`."""
    return {
        "data": str(MEXICAN_3_9),
        "experiment_name": "positive_control_test",
        "rounds": 3,
        "seller_pricing": True,
        **settings,
    }


def positive_control_config(tmp_path: Path, **settings) -> Path:
    """Write a Positive control config.yml (informant seed 7) and return its path."""
    return write_config(
        tmp_path / "input.yml",
        config_dict(
            seller_channel=True,
            undercover_informant={"seed": SEED},
            positive_control=True,
            **settings,
        ),
    )


def coordination_flags(cohort: dict) -> dict[str, bool]:
    """Return each seller's `coordination` constructor argument in one round."""
    return {
        agent.business.id: agent.kwargs.get("coordination", False)
        for agent in cohort["businesses"]
    }


def cohort_undercover(cohort: dict) -> list[str]:
    """Return the ids of the sellers built undercover in one round."""
    return [
        agent.business.id
        for agent in cohort["businesses"]
        if agent.kwargs.get("undercover", False)
    ]


def mentions_coordination(prompt: str) -> bool:
    """Return whether the prompt holds the coordination instruction or a part of it."""
    return COORDINATION_INSTRUCTION in prompt or COORDINATE in prompt.lower()


# Rule: The switch requires the seller channel


@pytest.mark.parametrize("channel", [{"seller_channel": False}, {}])
def test_config_with_positive_control_on_and_channel_off_is_rejected(tmp_path, channel):
    """Scenario: A config with the positive control on and the channel off is rejected."""
    # Given a config.yml with the positive control on and the channel off
    config_path = write_config(
        tmp_path / "config.yml", config_dict(positive_control=True, **channel)
    )

    # When it is loaded
    # Then loading fails with an error naming the seller channel
    with pytest.raises(ValidationError, match="seller_channel"):
        load_run_config(config_path)


def test_run_config_model_rejects_positive_control_without_channel():
    """Scenario: A config with the positive control on and the channel off is rejected (model)."""
    # No informant, so only the positive-control rule can reject it
    with pytest.raises(ValidationError, match="seller_channel"):
        RunConfig.model_validate(
            config_dict(seller_channel=False, positive_control=True)
        )


def test_config_with_positive_control_on_and_channel_off_does_not_start_a_run(
    monkeypatch, tmp_path, run_folder, recorded_runs, capsys
):
    """Scenario: A config with the positive control on and the channel off is rejected (CLI)."""
    config_path = write_config(
        tmp_path / "input.yml",
        config_dict(seller_channel=False, positive_control=True),
    )

    # The CLI reports any error on stderr and exits with status 1
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, "run", "--config", str(config_path))

    assert excinfo.value.code == 1
    assert "seller_channel" in capsys.readouterr().err
    assert recorded_runs == []


def test_config_with_positive_control_and_channel_on_loads(tmp_path):
    """Rule: The switch requires the seller channel (with it, the config loads)."""
    config = load_run_config(positive_control_config(tmp_path))

    assert config.positive_control is True


def test_config_without_positive_control_defaults_to_off(tmp_path):
    """Scenario: Without the switch no seller is built with the instruction (config default; passes on the stub)."""
    config_path = write_config(
        tmp_path / "config.yml",
        config_dict(seller_channel=True, undercover_informant={"seed": SEED}),
    )

    assert load_run_config(config_path).positive_control is False


# CLI: the config reaches the run


def test_config_with_positive_control_on_reaches_the_run(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Rule: Every round builds every seller except the undercover one with the instruction (config -> run)."""
    run_cli(monkeypatch, "run", "--config", str(positive_control_config(tmp_path)))

    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("positive_control") is True
    assert recorded_runs[0].get("undercover_informant") == UndercoverConfig(seed=SEED)


def test_stock_command_runs_with_positive_control_off(
    monkeypatch, run_folder, recorded_runs
):
    """Scenario: Without the switch no seller is built with the instruction (stock command; passes on the stub)."""
    run_cli(
        monkeypatch,
        "run",
        str(MEXICAN_3_9),
        "--experiment-name",
        "positive_control_test",
    )

    assert recorded_runs[0].get("positive_control", False) is False


# Rule: Every round builds every seller except the undercover one with the instruction


async def test_3_round_positive_control_run_instructs_the_same_8_sellers_every_round(
    run_folder, fake_run
):
    """Scenario: A 3-round Positive control run instructs the same 8 sellers every round."""
    # Given a Positive control config for "mexican_3_9" with 3 rounds and informant seed 7
    # When the run starts
    await run(
        rounds=3,
        undercover_informant=UndercoverConfig(seed=SEED),
        positive_control=True,
    )

    # Then in rounds 1, 2 and 3 the 8 sellers other than the undercover one get the
    # instruction
    undercover = read_pick(run_folder)["seller_id"]
    assert len(fake_run) == 3
    for cohort in fake_run:
        flags = coordination_flags(cohort)
        assert sorted(flags) == SELLER_IDS
        instructed = sorted(sid for sid, on in flags.items() if on)
        assert instructed == sorted(sid for sid in SELLER_IDS if sid != undercover)
        assert len(instructed) == 8
        # And the undercover seller does not
        assert flags[undercover] is False
        assert cohort_undercover(cohort) == [undercover]


def test_positive_control_run_without_an_informant_instructs_every_seller(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Scenario: A Positive control run without an informant instructs every seller."""
    # Given a config.yml with the channel and the positive control on and no informant
    config_path = write_config(
        tmp_path / "input.yml",
        config_dict(seller_channel=True, positive_control=True),
    )

    # When the run starts
    run_cli(monkeypatch, "run", "--config", str(config_path), "--server-port", "18999")

    # Then all 9 sellers get the instruction in every round
    assert len(fake_run) == 3
    for cohort in fake_run:
        flags = coordination_flags(cohort)
        assert sorted(flags) == SELLER_IDS
        assert all(flags.values())
        assert cohort_undercover(cohort) == []


@pytest.mark.parametrize("extra", [{}, {"positive_control": False}])
async def test_without_the_switch_no_seller_is_built_with_the_instruction(
    run_folder, fake_run, extra
):
    """Scenario: Without the switch no seller is built with the instruction (passes on the stub)."""
    # Given an Undercover config for "mexican_3_9" with 3 rounds
    # When the run starts
    await run(rounds=3, undercover_informant=UndercoverConfig(seed=SEED), **extra)

    # Then no seller gets the instruction in any round
    assert len(fake_run) == 3
    for cohort in fake_run:
        assert not any(coordination_flags(cohort).values())


# Rule: The run folder says the run was a positive control


def test_config_yml_records_the_switch(monkeypatch, tmp_path, run_folder, fake_run):
    """Scenario: config.yml records the switch."""
    # Given a Positive control config
    config_path = positive_control_config(tmp_path)

    # When the run starts
    run_cli(monkeypatch, "run", "--config", str(config_path), "--server-port", "18999")

    # Then the run folder's config.yml has positive_control true
    saved = yaml.safe_load((run_folder / "config.yml").read_text())
    assert saved["positive_control"] is True
    # And the run that started was the positive control it records
    assert len(fake_run) == 3
    assert any(coordination_flags(fake_run[0]).values())


# Prompts of a Positive control run against the same Undercover run


async def run_undercover_then_positive_control(run_folder: Path):
    """Run the same Undercover run, then its Positive control run.

    Returns (Undercover-run prompts, Positive-control-run prompts, undercover seller),
    each prompt set grouped by (kind, round, seller id).
    """
    # The same conversation in both runs: 0004 says hello to 0001, 0001 replies
    TALK_SCRIPT[(2, 1, SELLER_4)] = ([POBLANO], "hello")
    TALK_SCRIPT[(2, 2, POBLANO)] = ([SELLER_4], "hi")

    await run(undercover_informant=UndercoverConfig(seed=SEED))
    undercover_run = prompts_by_seller()
    undercover = read_pick(run_folder)["seller_id"]

    reset_fakes()
    await run(undercover_informant=UndercoverConfig(seed=SEED), positive_control=True)
    positive_control_run = prompts_by_seller()
    assert read_pick(run_folder)["seller_id"] == undercover

    return undercover_run, positive_control_run, undercover


async def test_undercover_seller_prompts_are_unchanged_from_an_undercover_run(
    run_folder, real_sellers
):
    """Scenario: The undercover seller's prompts are unchanged from an Undercover run."""
    # Given the undercover seller in a Positive control run
    (
        undercover_run,
        positive_control_run,
        undercover,
    ) = await run_undercover_then_positive_control(run_folder)

    # When its round-2 talk and pricing prompts are rendered
    talk = ("talk", 2, undercover)
    pricing = ("pricing", 2, undercover)
    assert positive_control_run[talk]
    assert len(positive_control_run[pricing]) == 1

    # Then they are the same as in the same Undercover run
    assert positive_control_run[talk] == undercover_run[talk]
    assert positive_control_run[pricing] == undercover_run[pricing]
    for key, prompts in positive_control_run.items():
        if key[2] == undercover:
            assert prompts == undercover_run[key]
            assert not any(mentions_coordination(p) for p in prompts)


async def test_positive_control_prompts_differ_from_the_undercover_run_only_by_the_instruction(
    run_folder, real_sellers
):
    """Rule: With the switch on, every seller except the undercover one sees the instruction ... (run level)."""
    (
        undercover_run,
        positive_control_run,
        undercover,
    ) = await run_undercover_then_positive_control(run_folder)

    assert sorted(positive_control_run) == sorted(undercover_run)
    # Every seller gets a pricing call in rounds 2 and 3 and a talk call
    for round_number in (2, 3):
        for seller_id in SELLER_IDS:
            assert len(positive_control_run[("pricing", round_number, seller_id)]) == 1
            assert positive_control_run[("talk", round_number, seller_id)]
    for key, prompts in undercover_run.items():
        if key[2] == undercover:
            assert positive_control_run[key] == prompts
        else:
            # Every other seller's talk and pricing open with the instruction
            assert positive_control_run[key] == [PREFIX + p for p in prompts]


async def test_undercover_run_prompts_do_not_mention_it(run_folder, real_sellers):
    """Scenario: Without the switch no prompt mentions it (an Undercover run; passes on the stub)."""
    # Given an Undercover run of "mexican_3_9"
    TALK_SCRIPT[(2, 1, SELLER_4)] = ([POBLANO], "hello")

    # When any seller prompt is rendered
    await run(undercover_informant=UndercoverConfig(seed=SEED))

    # Then it does not contain the coordination instruction
    assert len([p for p in PROMPTS if p[0] == "pricing"]) == 18
    for _, _, _, prompt in PROMPTS:
        assert not mentions_coordination(prompt)

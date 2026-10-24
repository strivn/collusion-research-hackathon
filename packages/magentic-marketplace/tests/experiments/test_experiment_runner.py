"""Tests for the experiment runner (feature: Experiment runner).

One experiment config lists arms (each a run config) and a number of repetitions. The
runner starts every arm x repetition as its own run, in parallel up to a cap, with
paired budget seeds across arms, and records each run in the tracker.

No Postgres, LLM or real marketplace subprocess: `asyncio.create_subprocess_exec` is
replaced by `FakeMarketplace`, which records each command and its cwd and writes the
files the real `run` and `score` commands would write into the cwd.
"""

import asyncio
import os
import random
import re
import sys
from pathlib import Path

import pytest
import yaml

from magentic_marketplace.experiments import experiment_runner
from magentic_marketplace.experiments.budgets import assign_budgets, save_budgets
from magentic_marketplace.experiments.experiment_runner import (
    ExperimentConfig,
    load_experiment_config,
    plan_runs,
    run_experiment,
)
from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
MEXICAN_10_30 = REPO_ROOT / "data" / "mexican_10_30"
SELLER_IDS = sorted(b.id for b in load_businesses_from_yaml(MEXICAN_3_9 / "businesses"))
MULTIPLIERS = [1.25, 1.5, 2.0]
MODEL = "claude-haiku-5-5"
CLI_MODULE = "magentic_marketplace.cli"

TRACKER_HEADER = (
    "| Run | RQ | Date | Arm | Model | Data | Results | Status | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)
OLD_ROW = (
    "| play1 | - | 2026-10-22 10:00 | S0 | claude-haiku-5-5 | data/mexican_3_9 "
    "| research/results/play1 | done | stock run |\n"
)
TRACKER_TEXT = (
    "# Tracker\n\n"
    "Intro text that must survive.\n\n"
    "## Runs\n\n" + TRACKER_HEADER + OLD_ROW + "\n"
    "## Notes\n\n"
    "Text after the table that must survive.\n"
)


class FakeProcess:
    """Stands in for an asyncio subprocess: exits with a set code after a delay."""

    def __init__(self, returncode: int, stderr: bytes, delay: float, on_exit):
        """Hold the exit code, stderr and what to do on exit."""
        self.pid = 4242
        self.returncode: int | None = None
        self._final_returncode = returncode
        self._stdout_bytes = b""
        self._stderr_bytes = stderr
        self._delay = delay
        self._on_exit = on_exit
        self._finished = False
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()

    async def _finish(self) -> None:
        """Run once: wait out the delay, set the exit code, fire the exit hook."""
        if self._finished:
            return
        self._finished = True
        await asyncio.sleep(self._delay)
        self.returncode = self._final_returncode
        self._on_exit()

    async def wait(self) -> int:
        """Wait for exit and return the exit code."""
        await self._finish()
        assert self.returncode is not None
        return self.returncode

    async def communicate(self, input=None) -> tuple[bytes, bytes]:
        """Wait for exit and return (stdout, stderr)."""
        await self._finish()
        return self._stdout_bytes, self._stderr_bytes

    def kill(self) -> None:
        """Do nothing."""

    def terminate(self) -> None:
        """Do nothing."""


class FakeMarketplace:
    """Replaces `asyncio.create_subprocess_exec` for `run` and `score` commands.

    `run` reads the config named after `--config` in the cwd and writes `config.yml`
    and, with a budget, `budgets.yml`. `score <schema>` writes `customer_rounds.csv`
    and `seller_rounds.csv`. A run counts as in progress from its `run` start until its
    `score` exits, or until a failing command exits.
    """

    def __init__(
        self,
        fail_run: set[str] | None = None,
        fail_score: set[str] | None = None,
        delay: float = 0.0,
    ):
        """Set which schemas fail at `run` or `score`, and how long each command takes."""
        self.fail_run = fail_run or set()
        self.fail_score = fail_score or set()
        self.delay = delay
        self.calls: list[tuple[list[str], Path]] = []
        self.active: set[str] = set()
        self.peak = 0

    def install(self, monkeypatch) -> "FakeMarketplace":
        """Patch subprocess creation as the runner module sees it."""
        monkeypatch.setattr(asyncio, "create_subprocess_exec", self.create)
        monkeypatch.setattr(
            experiment_runner, "create_subprocess_exec", self.create, raising=False
        )
        return self

    def commands_in(self, folder: Path) -> list[list[str]]:
        """Return the commands started with `folder` as cwd, in start order."""
        return [cmd for cmd, cwd in self.calls if cwd == folder.resolve()]

    async def create(self, program, *args, cwd=None, **kwargs) -> FakeProcess:
        """Record the command and act like the real `run` or `score` command."""
        cmd = [str(program), *(str(a) for a in args)]
        folder = Path(os.getcwd(), cwd if cwd is not None else ".").resolve()
        self.calls.append((cmd, folder))
        subcommand = cmd[cmd.index(CLI_MODULE) + 1]
        if subcommand == "run":
            return self._run(cmd, folder)
        if subcommand == "score":
            return self._score(cmd, folder)
        raise AssertionError(f"unexpected command {cmd}")

    def _start(self, schema: str) -> None:
        """Mark a run as in progress and update the peak."""
        self.active.add(schema)
        self.peak = max(self.peak, len(self.active))

    def _run(self, cmd: list[str], folder: Path) -> FakeProcess:
        """Write config.yml and budgets.yml like `run --config` does."""
        config = yaml.safe_load((folder / cmd[cmd.index("--config") + 1]).read_text())
        schema = config["experiment_name"]
        self._start(schema)
        if schema in self.fail_run:
            stderr = b"Traceback (most recent call last):\nRuntimeError: run boom\n"
            return FakeProcess(
                1, stderr, self.delay, lambda: self.active.discard(schema)
            )
        (folder / "config.yml").write_text(yaml.safe_dump(config))
        if config.get("budget"):
            customers = load_customers_from_yaml(Path(config["data"]) / "customers")
            budgets = assign_budgets(
                customers, config["budget"]["multipliers"], config["budget"]["seed"]
            )
            save_budgets(budgets, folder)
        return FakeProcess(0, b"", self.delay, lambda: None)

    def _score(self, cmd: list[str], folder: Path) -> FakeProcess:
        """Write the two measurement tables like `score <schema>` does."""
        schema = cmd[cmd.index("score") + 1]
        if schema in self.fail_score:
            stderr = b"Traceback (most recent call last):\nValueError: score boom\n"
            return FakeProcess(
                1, stderr, self.delay, lambda: self.active.discard(schema)
            )
        (folder / "customer_rounds.csv").write_text("round,customer_id\n")
        (folder / "seller_rounds.csv").write_text("round,seller_id\n")
        return FakeProcess(0, b"", self.delay, lambda: self.active.discard(schema))


def write_yaml(path: Path, content: dict) -> Path:
    """Write `content` as YAML to `path` and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    return path


def arm_config(data: str, seed: int, **extra) -> dict:
    """Build a run config with a budget section."""
    return {
        "data": data,
        "experiment_name": "placeholder",
        "budget": {"multipliers": MULTIPLIERS, "seed": seed},
        **extra,
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    """Run from `tmp_path` with a tracker, a dummy env file and LLM_MODEL set."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_MODEL", MODEL)
    (tmp_path / "dummy.env").write_text(f"LLM_MODEL={MODEL}\n")
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "tracker.md").write_text(TRACKER_TEXT)
    # Arm run configs live in a config dir separate from the cwd
    configs = tmp_path / "configs"
    write_yaml(configs / "s2.yml", arm_config(str(MEXICAN_3_9), seed=111))
    write_yaml(
        configs / "s3.yml",
        arm_config(str(MEXICAN_3_9), seed=222, rounds=2, seller_pricing=True),
    )
    return tmp_path


def make_config(**overrides) -> ExperimentConfig:
    """Experiment exp1 with arms S2 and S3 and 3 repetitions, overridable."""
    fields = {
        "name": "exp1",
        "arms": {"S2": "s2.yml", "S3": "s3.yml"},
        "repetitions": 3,
        "base_seed": 7,
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


async def run_in(workspace: Path, config: ExperimentConfig):
    """Run `config` with the workspace's config dir and dummy env file."""
    return await run_experiment(config, workspace / "configs", workspace / "dummy.env")


def run_folder(workspace: Path, run_name: str, name: str = "exp1") -> Path:
    """Return the results folder of one run inside the workspace."""
    return workspace / "research" / "results" / name / run_name


def tracker_rows(workspace: Path) -> list[list[str]]:
    """Return the cells of every row in the tracker's Runs table, header excluded."""
    text = (workspace / "research" / "tracker.md").read_text()
    runs = text.split("## Runs", 1)[1].split("\n## ", 1)[0]
    rows = [line for line in runs.splitlines() if line.startswith("|")]
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows[2:]]


# Experiment config loading (agreed detail 1)


def test_load_experiment_config_reads_every_key(tmp_path):
    """Detail 1: the experiment config YAML gives name, arms, repetitions, seed, cap."""
    path = write_yaml(
        tmp_path / "exp.yml",
        {
            "name": "exp1",
            "arms": {"S2": "s2.yml", "S3": "s3.yml"},
            "repetitions": 3,
            "base_seed": 7,
            "parallel": 3,
            "results_dir": "out/results",
            "tracker": "out/tracker.md",
            "rq": "RQ1",
        },
    )

    config = load_experiment_config(path)

    assert config.name == "exp1"
    assert config.arms == {"S2": "s2.yml", "S3": "s3.yml"}
    assert list(config.arms) == ["S2", "S3"]
    assert config.repetitions == 3
    assert config.base_seed == 7
    assert config.parallel == 3
    assert config.results_dir == "out/results"
    assert config.tracker == "out/tracker.md"
    assert config.rq == "RQ1"


def test_load_experiment_config_defaults(tmp_path):
    """Detail 1: parallel defaults to 1, results and tracker to research/, rq empty."""
    path = write_yaml(
        tmp_path / "exp.yml",
        {"name": "exp1", "arms": {"S2": "s2.yml"}, "repetitions": 1, "base_seed": 7},
    )

    config = load_experiment_config(path)

    assert config.parallel == 1
    assert config.results_dir == "research/results"
    assert config.tracker == "research/tracker.md"
    assert config.rq == ""


def test_load_experiment_config_rejects_unknown_keys(tmp_path):
    """Detail 1: an unknown key is rejected with an error naming it."""
    path = write_yaml(
        tmp_path / "exp.yml",
        {
            "name": "exp1",
            "arms": {"S2": "s2.yml"},
            "repetitions": 1,
            "base_seed": 7,
            "paralell_cap": 3,
        },
    )

    with pytest.raises(ValueError, match="paralell_cap"):
        load_experiment_config(path)


# Rule: Every arm x repetition becomes its own run with its own folder and schema


def test_plan_two_arms_three_repetitions_six_runs():
    """Scenario: Two arms, three repetitions make six runs (planning part)."""
    # Given an experiment config "exp1" with arms "S2" and "S3" and 3 repetitions
    specs = plan_runs(make_config())

    assert [(s.arm, s.repetition) for s in specs] == [
        ("S2", 1),
        ("S3", 1),
        ("S2", 2),
        ("S3", 2),
        ("S2", 3),
        ("S3", 3),
    ]
    s3_rep2 = specs[3]
    assert s3_rep2.run_name == "S3_rep2"
    assert s3_rep2.schema_name == "exp1_S3_rep2"
    assert s3_rep2.folder == Path("research/results/exp1/S3_rep2")


def test_plan_folder_follows_results_dir():
    """Detail 2: the folder is results_dir / name / run_name."""
    specs = plan_runs(make_config(results_dir="elsewhere/out", repetitions=1))

    assert [s.folder for s in specs] == [
        Path("elsewhere/out/exp1/S2_rep1"),
        Path("elsewhere/out/exp1/S3_rep1"),
    ]


async def test_two_arms_three_repetitions_six_runs_finish(workspace, monkeypatch):
    """Scenario: Two arms, three repetitions make six runs."""
    # Given an experiment config "exp1" with arms "S2" and "S3" and 3 repetitions
    fake = FakeMarketplace().install(monkeypatch)

    # When the experiment runs
    outcomes = await run_in(workspace, make_config())

    # Then 6 runs finish
    assert sorted(o.spec.run_name for o in outcomes) == [
        "S2_rep1",
        "S2_rep2",
        "S2_rep3",
        "S3_rep1",
        "S3_rep2",
        "S3_rep3",
    ]
    assert [o.status for o in outcomes] == ["done"] * 6
    assert all(o.error is None for o in outcomes)
    # And run "S3" repetition 2 has folder ".../exp1/S3_rep2" and schema "exp1_S3_rep2"
    folder = run_folder(workspace, "S3_rep2")
    input_config = yaml.safe_load((folder / "input.yml").read_text())
    assert input_config["experiment_name"] == "exp1_S3_rep2"
    env_file = str((workspace / "dummy.env").resolve())
    commands = fake.commands_in(folder)
    assert commands == [
        [
            sys.executable,
            "-m",
            CLI_MODULE,
            "run",
            "--config",
            "input.yml",
            "--env-file",
            env_file,
        ],
        [sys.executable, "-m", CLI_MODULE, "score", "exp1_S3_rep2"],
    ]
    assert len(fake.calls) == 12


async def test_input_yml_keeps_arm_settings_and_resolves_data(workspace, monkeypatch):
    """Detail 3a: input.yml is the arm's config with schema, seed and absolute data."""
    # The S3 arm names its data relative to the runner's cwd
    (workspace / "data").mkdir()
    (workspace / "data" / "mexican_3_9").symlink_to(MEXICAN_3_9)
    write_yaml(
        workspace / "configs" / "s3.yml",
        arm_config("data/mexican_3_9", seed=222, rounds=2, seller_pricing=True),
    )
    FakeMarketplace().install(monkeypatch)

    await run_in(workspace, make_config(repetitions=1))

    written = yaml.safe_load(
        (run_folder(workspace, "S3_rep1") / "input.yml").read_text()
    )
    assert written["experiment_name"] == "exp1_S3_rep1"
    assert written["rounds"] == 2
    assert written["seller_pricing"] is True
    assert written["budget"] == {"multipliers": MULTIPLIERS, "seed": 7}
    data = Path(written["data"])
    assert data.is_absolute()
    assert data.resolve() == MEXICAN_3_9.resolve()


async def test_arm_without_budget_runs_without_budget(workspace, monkeypatch):
    """Detail 3a: the seed is set only when the arm's run config has a budget."""
    write_yaml(
        workspace / "configs" / "s2.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "placeholder"},
    )
    FakeMarketplace().install(monkeypatch)

    outcomes = await run_in(workspace, make_config(repetitions=1))

    assert [o.status for o in outcomes] == ["done", "done"]
    folder = run_folder(workspace, "S2_rep1")
    written = yaml.safe_load((folder / "input.yml").read_text())
    assert written.get("budget") is None
    assert not (folder / "budgets.yml").exists()


async def test_finished_run_folder_holds_outputs(workspace, monkeypatch):
    """Scenario: Each run folder holds the run's outputs."""
    # Given a finished run "exp1/S2_rep1"
    FakeMarketplace().install(monkeypatch)
    await run_in(workspace, make_config(repetitions=1))

    # Then its folder holds the resolved config, the budgets and the measurement tables
    folder = run_folder(workspace, "S2_rep1")
    for name in [
        "input.yml",
        "config.yml",
        "budgets.yml",
        "customer_rounds.csv",
        "seller_rounds.csv",
    ]:
        assert (folder / name).is_file(), name
    resolved = yaml.safe_load((folder / "config.yml").read_text())
    assert resolved["experiment_name"] == "exp1_S2_rep1"


# Rule: Repetition i uses the same budget seed in every arm


def test_plan_paired_seeds():
    """Scenario: Paired seeds (planning part): seed = base_seed + i - 1 in every arm."""
    specs = plan_runs(make_config(repetitions=2, base_seed=7))

    seeds = {s.run_name: s.budget_seed for s in specs}
    assert seeds == {"S2_rep1": 7, "S2_rep2": 8, "S3_rep1": 7, "S3_rep2": 8}


async def test_paired_seeds_give_same_budgets(workspace, monkeypatch):
    """Scenario: Paired seeds."""
    # Given an experiment config with arms "S2" and "S3", 2 repetitions and base seed 7
    FakeMarketplace().install(monkeypatch)

    # When the experiment runs
    await run_in(workspace, make_config(repetitions=2, base_seed=7))

    def seed_of(run_name: str) -> int:
        """Return the budget seed written to the run's input.yml."""
        folder = run_folder(workspace, run_name)
        return yaml.safe_load((folder / "input.yml").read_text())["budget"]["seed"]

    def budgets_of(run_name: str) -> dict[str, float]:
        """Map customer id to budget, computed from the run's own seed."""
        customers = load_customers_from_yaml(MEXICAN_3_9 / "customers")
        return {
            b.customer_id: b.budget
            for b in assign_budgets(customers, MULTIPLIERS, seed_of(run_name))
        }

    # Then "S2_rep1" and "S3_rep1" give every customer the same budget
    assert seed_of("S2_rep1") == seed_of("S3_rep1") == 7
    assert budgets_of("S2_rep1") == budgets_of("S3_rep1")
    s2_budgets = yaml.safe_load(
        (run_folder(workspace, "S2_rep1") / "budgets.yml").read_text()
    )
    s3_budgets = yaml.safe_load(
        (run_folder(workspace, "S3_rep1") / "budgets.yml").read_text()
    )
    assert s2_budgets == s3_budgets
    # And "S2_rep1" and "S2_rep2" use different seeds
    assert seed_of("S2_rep2") == 8
    assert seed_of("S2_rep1") != seed_of("S2_rep2")


# Rule: Each repetition plants a different seller until every seller has been planted


def write_undercover_arm(workspace: Path) -> None:
    """Write an Undercover arm (channel, pricing and informant on) to the config dir."""
    write_yaml(
        workspace / "configs" / "undercover.yml",
        arm_config(
            str(MEXICAN_3_9),
            seed=333,
            rounds=2,
            seller_pricing=True,
            seller_channel=True,
            undercover_informant={"seed": 444},
        ),
    )


def undercover_config(**overrides) -> ExperimentConfig:
    """Experiment with only the Undercover arm, base seed 7, overridable."""
    return make_config(arms={"Undercover": "undercover.yml"}, **overrides)


def input_of(workspace: Path, run_name: str, name: str = "exp1") -> dict:
    """Return the run's input.yml."""
    folder = run_folder(workspace, run_name, name)
    return yaml.safe_load((folder / "input.yml").read_text())


def seller_at(seed: int, position: int) -> str:
    """Return the seller at `position` of the sorted seller ids shuffled by `seed`.

    Computed here, independent of `pick_undercover_seller`.
    """
    ids = list(SELLER_IDS)
    random.Random(seed).shuffle(ids)
    return ids[position % len(ids)]


def planted_by(workspace: Path, run_name: str, name: str = "exp1") -> str:
    """Return the seller a run plants, from the seed and position in its input.yml."""
    informant = input_of(workspace, run_name, name)["undercover_informant"]
    return seller_at(informant["seed"], informant.get("position", 0))


def test_plan_informant_seed_is_the_base_seed_and_position_the_repetition():
    """Rule: Each repetition plants a different seller ... (planning part).

    Every repetition shuffles with the base seed; repetition i takes position i - 1.
    """
    specs = plan_runs(undercover_config(repetitions=3, base_seed=7))

    assert [(s.informant_seed, s.informant_position) for s in specs] == [
        (7, 0),
        (7, 1),
        (7, 2),
    ]


async def test_input_yml_carries_the_informant_seed_and_position(
    workspace, monkeypatch
):
    """Rule: Each repetition plants a different seller ... (input.yml part)."""
    write_undercover_arm(workspace)
    FakeMarketplace().install(monkeypatch)

    await run_in(workspace, undercover_config(repetitions=3, base_seed=7))

    # Only seed and position: input.yml lists every informant setting, defaults included
    informants = [
        {
            key: input_of(workspace, f"Undercover_rep{i}")["undercover_informant"][key]
            for key in ("seed", "position")
        }
        for i in (1, 2, 3)
    ]
    assert informants == [
        {"seed": 7, "position": 0},
        {"seed": 7, "position": 1},
        {"seed": 7, "position": 2},
    ]


async def test_three_repetitions_plant_three_different_sellers(workspace, monkeypatch):
    """Scenario: 3 repetitions on mexican_3_9 plant 3 different sellers."""
    # Given an experiment with an Undercover arm on "mexican_3_9", base seed 7 and 3
    # repetitions
    write_undercover_arm(workspace)
    FakeMarketplace().install(monkeypatch)

    # When the experiment runs
    outcomes = await run_in(workspace, undercover_config(repetitions=3, base_seed=7))

    # Then repetitions 1, 2 and 3 plant 3 different sellers
    assert [o.status for o in outcomes] == ["done"] * 3
    planted = [planted_by(workspace, f"Undercover_rep{i}") for i in (1, 2, 3)]
    assert len(set(planted)) == 3
    assert set(planted) <= set(SELLER_IDS)


async def test_same_base_seed_plants_same_sellers_in_same_order(workspace, monkeypatch):
    """Scenario: The same base seed plants the same sellers in the same order."""
    # Given two experiments with an Undercover arm on "mexican_3_9", base seed 7 and 3
    # repetitions
    write_undercover_arm(workspace)
    FakeMarketplace().install(monkeypatch)

    # When both experiments run
    for name in ("exp1", "exp2"):
        await run_in(
            workspace, undercover_config(name=name, repetitions=3, base_seed=7)
        )

    # Then repetition 1, 2 and 3 plant the same seller in both
    for i in (1, 2, 3):
        run_name = f"Undercover_rep{i}"
        assert planted_by(workspace, run_name, "exp1") == planted_by(
            workspace, run_name, "exp2"
        )


async def test_more_repetitions_than_sellers_wraps_around(workspace, monkeypatch):
    """Scenario: More repetitions than sellers wraps around to the first."""
    # Given an experiment with an Undercover arm on "mexican_3_9", base seed 7 and 10
    # repetitions
    write_undercover_arm(workspace)
    FakeMarketplace().install(monkeypatch)

    # When the experiment runs
    await run_in(workspace, undercover_config(repetitions=10, base_seed=7))

    planted = [planted_by(workspace, f"Undercover_rep{i}") for i in range(1, 11)]
    # Then repetitions 1 to 9 plant each of the 9 sellers once
    assert sorted(planted[:9]) == SELLER_IDS
    # And repetition 10 plants the same seller as repetition 1
    assert planted[9] == planted[0]


async def test_budget_seed_per_repetition_is_unchanged(workspace, monkeypatch):
    """Scenario: The budget seed per repetition is unchanged."""
    # Given an experiment with an Undercover arm, base seed 7 and 3 repetitions
    write_undercover_arm(workspace)
    FakeMarketplace().install(monkeypatch)
    config = make_config(
        arms={"S2": "s2.yml", "Undercover": "undercover.yml"}, base_seed=7
    )

    # When the experiment runs
    await run_in(workspace, config)

    # Then repetitions 1, 2 and 3 use budget seeds 7, 8 and 9
    budget_seeds = [
        input_of(workspace, f"Undercover_rep{i}")["budget"]["seed"] for i in (1, 2, 3)
    ]
    assert budget_seeds == [7, 8, 9]
    # The arm without an informant gets none
    for i in (1, 2, 3):
        assert input_of(workspace, f"S2_rep{i}").get("undercover_informant") is None


# Rule: Runs execute in parallel up to a cap


async def test_no_more_than_cap_at_once(workspace, monkeypatch):
    """Scenario: No more than the cap at once."""
    # Given an experiment with 6 runs and a parallel cap of 3
    fake = FakeMarketplace(delay=0.05).install(monkeypatch)

    # When the experiment runs
    outcomes = await run_in(workspace, make_config(parallel=3))

    # Then at no time are more than 3 runs in progress
    assert len(outcomes) == 6
    assert fake.peak <= 3
    # The cap is used, not undercut: runs do overlap up to it
    assert fake.peak == 3


async def test_default_cap_runs_one_at_a_time(workspace, monkeypatch):
    """Detail 1: parallel defaults to 1, so runs never overlap."""
    fake = FakeMarketplace(delay=0.02).install(monkeypatch)

    outcomes = await run_in(workspace, make_config(repetitions=2))

    assert len(outcomes) == 4
    assert fake.peak == 1


# Rule: A failed run does not stop the others


async def test_one_failure_the_rest_finish(workspace, monkeypatch):
    """Scenario: One failure, the rest finish."""
    # Given an experiment with 4 runs where run "S3_rep1" fails
    fake = FakeMarketplace(fail_run={"exp1_S3_rep1"}, delay=0.01).install(monkeypatch)

    # When the experiment runs
    outcomes = await run_in(workspace, make_config(repetitions=2, parallel=2))

    by_name = {o.spec.run_name: o for o in outcomes}
    assert set(by_name) == {"S2_rep1", "S2_rep2", "S3_rep1", "S3_rep2"}
    # Then the other 3 runs finish
    for run_name in ["S2_rep1", "S2_rep2", "S3_rep2"]:
        assert by_name[run_name].status == "done", run_name
        assert (run_folder(workspace, run_name) / "seller_rounds.csv").is_file()
    # And "S3_rep1" is reported as failed with its error
    failed = by_name["S3_rep1"]
    assert failed.status == "failed"
    assert failed.error is not None
    assert "run boom" in failed.error
    # A failed `run` is not scored
    commands = fake.commands_in(run_folder(workspace, "S3_rep1"))
    assert [cmd[3] for cmd in commands] == ["run"]


async def test_failed_score_marks_run_failed(workspace, monkeypatch):
    """Detail 3c: a run whose `score` exits non-zero is failed with score's stderr."""
    FakeMarketplace(fail_score={"exp1_S2_rep1"}).install(monkeypatch)

    outcomes = await run_in(workspace, make_config(repetitions=1))

    by_name = {o.spec.run_name: o for o in outcomes}
    assert by_name["S2_rep1"].status == "failed"
    assert "score boom" in (by_name["S2_rep1"].error or "")
    assert by_name["S3_rep1"].status == "done"


# Rule: Every run is recorded in the tracker


async def test_tracker_row_per_run(workspace, monkeypatch):
    """Scenario: A tracker row per run."""
    # Given a finished run "exp1/S2_rep1" on "mexican_10_30" with "claude-haiku-5-5"
    (workspace / "data").mkdir()
    (workspace / "data" / "mexican_10_30").symlink_to(MEXICAN_10_30)
    write_yaml(
        workspace / "configs" / "s2.yml", arm_config("data/mexican_10_30", seed=111)
    )
    FakeMarketplace().install(monkeypatch)
    await run_in(workspace, make_config(repetitions=1, rq="RQ1"))

    # Then "research/tracker.md" has a row with run, arm, date, model, data, results
    # folder and status "done"
    rows = {row[0]: row for row in tracker_rows(workspace)}
    row = rows["exp1/S2_rep1"]
    assert len(row) == 9
    run, rq, date, arm, model, data, results, status, notes = row
    assert rq == "RQ1"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", date)
    assert arm == "S2"
    assert model == MODEL
    assert data == "data/mexican_10_30"
    assert results == "research/results/exp1/S2_rep1"
    assert status == "done"
    assert notes == ""


async def test_tracker_keeps_the_rest_of_the_file(workspace, monkeypatch):
    """Detail 4: rows go after the table's last row; the rest of the file is kept."""
    FakeMarketplace().install(monkeypatch)

    await run_in(workspace, make_config(repetitions=1))

    text = (workspace / "research" / "tracker.md").read_text()
    assert text.startswith("# Tracker\n\nIntro text that must survive.\n\n## Runs\n\n")
    assert TRACKER_HEADER + OLD_ROW in text
    assert text.endswith("## Notes\n\nText after the table that must survive.\n")
    rows = tracker_rows(workspace)
    assert [row[0] for row in rows][0] == "play1"
    assert sorted(row[0] for row in rows[1:]) == ["exp1/S2_rep1", "exp1/S3_rep1"]
    # The new rows sit inside the Runs table, before the next section
    runs_section = text.split("## Runs", 1)[1].split("## Notes", 1)[0]
    assert "exp1/S2_rep1" in runs_section
    assert "exp1/S3_rep1" in runs_section


async def test_parallel_runs_write_whole_tracker_rows(workspace, monkeypatch):
    """Detail 4: rows from parallel runs neither interleave nor clobber each other."""
    FakeMarketplace(delay=0.01).install(monkeypatch)

    await run_in(workspace, make_config(parallel=6))

    rows = tracker_rows(workspace)
    assert len(rows) == 7
    assert all(len(row) == 9 for row in rows)
    assert sorted(row[0] for row in rows[1:]) == [
        f"exp1/{arm}_rep{i}" for arm in ["S2", "S3"] for i in [1, 2, 3]
    ]


async def test_tracker_records_failed_run_with_error(workspace, monkeypatch):
    """Scenario: One failure, the rest finish (tracker part): status and error recorded."""
    FakeMarketplace(fail_run={"exp1_S3_rep1"}).install(monkeypatch)

    await run_in(workspace, make_config(repetitions=1))

    rows = {row[0]: row for row in tracker_rows(workspace)}
    assert rows["exp1/S3_rep1"][7] == "failed"
    assert "run boom" in rows["exp1/S3_rep1"][8]
    assert rows["exp1/S2_rep1"][7] == "done"

"""Tests for the order in which the experiment runner starts runs.

Feature: experiment_runner.technical.feature.
Rule: Runs are started repetition by repetition, every arm in each repetition.

No Postgres, LLM or real marketplace subprocess: `asyncio.create_subprocess_exec` is
replaced by `TimedMarketplace`, which writes the files the real `run` and `score`
commands would write and records when each run starts and finishes.
"""

import asyncio
import os
import time
from pathlib import Path

import pytest
import yaml

from magentic_marketplace.experiments import experiment_runner
from magentic_marketplace.experiments.budgets import assign_budgets, save_budgets
from magentic_marketplace.experiments.experiment_runner import (
    ExperimentConfig,
    plan_runs,
    run_experiment,
)
from magentic_marketplace.experiments.utils import load_customers_from_yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
MULTIPLIERS = [1.25, 1.5, 2.0]
MODEL = "claude-haiku-5-5"
CLI_MODULE = "magentic_marketplace.cli"

TRACKER_TEXT = (
    "# Tracker\n\n## Runs\n\n"
    "| Run | RQ | Date | Arm | Model | Data | Results | Status | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


class FakeProcess:
    """Stands in for an asyncio subprocess: exits with code 0 after a delay."""

    def __init__(self, delay: float, on_exit):
        """Hold the delay and what to do on exit."""
        self.pid = 4242
        self.returncode: int | None = None
        self._delay = delay
        self._on_exit = on_exit
        self._finished = False
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_eof()

    async def _finish(self) -> None:
        """Run once: wait out the delay, set the exit code, fire the exit hook."""
        if self._finished:
            return
        self._finished = True
        await asyncio.sleep(self._delay)
        self.returncode = 0
        self._on_exit()

    async def wait(self) -> int:
        """Wait for exit and return the exit code."""
        await self._finish()
        assert self.returncode is not None
        return self.returncode

    async def communicate(self, input=None) -> tuple[bytes, bytes]:
        """Wait for exit and return empty (stdout, stderr)."""
        await self._finish()
        return b"", b""

    def kill(self) -> None:
        """Do nothing."""

    def terminate(self) -> None:
        """Do nothing."""


class TimedMarketplace:
    """Replaces `asyncio.create_subprocess_exec` and times each run.

    A run starts when its `run` command is created and finishes when its `score`
    command exits. Times are keyed by schema name, e.g. "exp1_S2_rep1".
    """

    def __init__(self, delay: float):
        """Set how long each command takes."""
        self.delay = delay
        self.started: dict[str, float] = {}
        self.finished: dict[str, float] = {}

    def install(self, monkeypatch) -> "TimedMarketplace":
        """Patch subprocess creation as the runner module sees it."""
        monkeypatch.setattr(asyncio, "create_subprocess_exec", self.create)
        monkeypatch.setattr(
            experiment_runner, "create_subprocess_exec", self.create, raising=False
        )
        return self

    async def create(self, program, *args, cwd=None, **kwargs) -> FakeProcess:
        """Act like the real `run` or `score` command."""
        cmd = [str(program), *(str(a) for a in args)]
        folder = Path(os.getcwd(), cwd if cwd is not None else ".").resolve()
        subcommand = cmd[cmd.index(CLI_MODULE) + 1]
        if subcommand == "run":
            return self._run(cmd, folder)
        if subcommand == "score":
            return self._score(cmd, folder)
        raise AssertionError(f"unexpected command {cmd}")

    def _run(self, cmd: list[str], folder: Path) -> FakeProcess:
        """Record the start and write config.yml and budgets.yml like `run` does."""
        config = yaml.safe_load((folder / cmd[cmd.index("--config") + 1]).read_text())
        self.started[config["experiment_name"]] = time.monotonic()
        (folder / "config.yml").write_text(yaml.safe_dump(config))
        if config.get("budget"):
            customers = load_customers_from_yaml(Path(config["data"]) / "customers")
            budgets = assign_budgets(
                customers, config["budget"]["multipliers"], config["budget"]["seed"]
            )
            save_budgets(budgets, folder)
        return FakeProcess(self.delay, lambda: None)

    def _score(self, cmd: list[str], folder: Path) -> FakeProcess:
        """Write the measurement tables and record the finish on exit."""
        schema = cmd[cmd.index("score") + 1]
        (folder / "customer_rounds.csv").write_text("round,customer_id\n")
        (folder / "seller_rounds.csv").write_text("round,seller_id\n")

        def finish() -> None:
            """Record when the run's last command exits."""
            self.finished[schema] = time.monotonic()

        return FakeProcess(self.delay, finish)


def write_yaml(path: Path, content: dict) -> Path:
    """Write `content` as YAML to `path` and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    return path


def arm_config(seed: int) -> dict:
    """Build a run config on mexican_3_9 with a budget section."""
    return {
        "data": str(MEXICAN_3_9),
        "experiment_name": "placeholder",
        "budget": {"multipliers": MULTIPLIERS, "seed": seed},
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    """Run from `tmp_path` with a tracker, arm configs, a dummy env file and LLM_MODEL."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_MODEL", MODEL)
    (tmp_path / "dummy.env").write_text(f"LLM_MODEL={MODEL}\n")
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "tracker.md").write_text(TRACKER_TEXT)
    write_yaml(tmp_path / "configs" / "s2.yml", arm_config(seed=111))
    write_yaml(tmp_path / "configs" / "s3.yml", arm_config(seed=222))
    return tmp_path


def make_config(**overrides) -> ExperimentConfig:
    """Experiment exp1 with arms S2 and S3, overridable."""
    fields = {
        "name": "exp1",
        "arms": {"S2": "s2.yml", "S3": "s3.yml"},
        "repetitions": 3,
        "base_seed": 7,
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def test_two_arms_three_repetitions_start_in_paired_order():
    """Rule: Runs are started repetition by repetition, every arm in each repetition.

    Scenario: Two arms, three repetitions start in paired order.
    """
    # Given an experiment config with arms "S2" and "S3" and 3 repetitions
    config = make_config(repetitions=3)

    # When its runs are planned
    specs = plan_runs(config)

    # Then the start order is S2_rep1, S3_rep1, S2_rep2, S3_rep2, S2_rep3, S3_rep3
    assert [s.run_name for s in specs] == [
        "S2_rep1",
        "S3_rep1",
        "S2_rep2",
        "S3_rep2",
        "S2_rep3",
        "S3_rep3",
    ]


async def test_cap_of_two_runs_each_repetitions_arms_together(workspace, monkeypatch):
    """Rule: Runs are started repetition by repetition, every arm in each repetition.

    Scenario: With a cap of 2, each repetition's arms run together.
    """
    # Given an experiment config with arms "S2" and "S3", 2 repetitions and a cap of 2
    fake = TimedMarketplace(delay=0.05).install(monkeypatch)
    config = make_config(repetitions=2, parallel=2)

    # When the experiment runs
    outcomes = await run_experiment(
        config, workspace / "configs", workspace / "dummy.env"
    )

    # Then S2_rep1 and S3_rep1 are in progress together before either rep-2 run starts
    assert [o.status for o in outcomes] == ["done"] * 4
    s2_rep1 = (fake.started["exp1_S2_rep1"], fake.finished["exp1_S2_rep1"])
    s3_rep1 = (fake.started["exp1_S3_rep1"], fake.finished["exp1_S3_rep1"])
    assert s3_rep1[0] < s2_rep1[1] and s2_rep1[0] < s3_rep1[1]
    first_rep2_start = min(fake.started["exp1_S2_rep2"], fake.started["exp1_S3_rep2"])
    assert max(s2_rep1[0], s3_rep1[0]) < first_rep2_start

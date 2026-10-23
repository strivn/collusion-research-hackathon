"""Experiment runner: every arm x repetition as its own run, in parallel up to a cap."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ExperimentConfig(BaseModel):
    """An experiment: arms (each a run config file), repetitions and a parallel cap."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arms: dict[str, str]
    repetitions: int
    base_seed: int
    parallel: int = 1
    results_dir: str = "research/results"
    tracker: str = "research/tracker.md"
    rq: str = ""


class RunSpec(BaseModel):
    """One planned run: an arm at one repetition."""

    arm: str
    repetition: int
    run_name: str
    schema_name: str
    folder: Path
    budget_seed: int


class RunOutcome(BaseModel):
    """How one run ended."""

    spec: RunSpec
    status: str
    error: str | None = None


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Read and validate an experiment config; unknown keys are rejected."""
    raise NotImplementedError("load_experiment_config")


def plan_runs(config: ExperimentConfig) -> list[RunSpec]:
    """List every arm x repetition with its folder, schema and budget seed."""
    raise NotImplementedError("plan_runs")


async def run_experiment(
    config: ExperimentConfig, config_dir: Path, env_file: Path
) -> list[RunOutcome]:
    """Run every planned run, at most `config.parallel` at once, and record each in the tracker."""
    raise NotImplementedError("run_experiment")

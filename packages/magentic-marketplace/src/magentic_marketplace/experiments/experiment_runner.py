"""Experiment runner: every arm x repetition as its own run, in parallel up to a cap."""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from magentic_marketplace.experiments.run_config import load_run_config

ERROR_TAIL_LINES = 10


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
    informant_seed: int
    informant_position: int


class RunOutcome(BaseModel):
    """How one run ended."""

    spec: RunSpec
    status: str
    error: str | None = None


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Read and validate an experiment config; unknown keys are rejected."""
    # Pydantic names every unknown or missing key in its error
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text()))


def plan_runs(config: ExperimentConfig) -> list[RunSpec]:
    """List every repetition x arm, repetition first, with its folder, schema, seeds and position."""
    # Repetition i has the same budget seed in every arm, so arms are paired; listing a
    # repetition's arms together makes the paired runs start, and run under the cap, side by
    # side. The informant seed is fixed, and repetition i plants the seller at position i - 1 of
    # its shuffle, so repetitions plant different sellers
    return [
        RunSpec(
            arm=arm,
            repetition=i,
            run_name=f"{arm}_rep{i}",
            schema_name=f"{config.name}_{arm}_rep{i}",
            folder=Path(config.results_dir) / config.name / f"{arm}_rep{i}",
            budget_seed=config.base_seed + i - 1,
            informant_seed=config.base_seed,
            informant_position=i - 1,
        )
        for i in range(1, config.repetitions + 1)
        for arm in config.arms
    ]


async def _run_command(args: list[str], folder: Path) -> str | None:
    """Run one command in `folder`; return the last stderr lines if it fails, else None."""
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=folder,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode == 0:
        return None
    tail = stderr.decode(errors="replace").strip().splitlines()[-ERROR_TAIL_LINES:]
    return "\n".join(tail) or f"exit code {process.returncode}"


async def _execute_run(
    spec: RunSpec, run_config_path: Path, env_file: Path
) -> RunOutcome:
    """Write the run's input.yml, then run and score it in its folder."""
    # Give the run its own schema, seeds and informant position, and an absolute data path
    spec.folder.mkdir(parents=True, exist_ok=True)
    run_config = load_run_config(run_config_path)
    run_config.experiment_name = spec.schema_name
    if run_config.budget is not None:
        run_config.budget.seed = spec.budget_seed
    if run_config.undercover_informant is not None:
        run_config.undercover_informant.seed = spec.informant_seed
        run_config.undercover_informant.position = spec.informant_position
    # The speaking order and first words follow the repetition seed, as budgets do
    if run_config.random_speaking_order is not None:
        run_config.random_speaking_order.seed = spec.budget_seed
    if run_config.random_first_word is not None:
        run_config.random_first_word.seed = spec.budget_seed
    run_config.data = str(Path(run_config.data).resolve())
    (spec.folder / "input.yml").write_text(
        yaml.safe_dump(run_config.model_dump(), sort_keys=False)
    )

    # Score only runs that finished; the first failing command ends the run
    cli = [sys.executable, "-m", "magentic_marketplace.cli"]
    commands = [
        [*cli, "run", "--config", "input.yml", "--env-file", str(env_file.resolve())],
        [*cli, "score", spec.schema_name],
    ]
    for command in commands:
        error = await _run_command(command, spec.folder)
        if error is not None:
            return RunOutcome(spec=spec, status="failed", error=error)
    return RunOutcome(spec=spec, status="done")


def _append_tracker_row(tracker: Path, row: str) -> None:
    """Insert `row` after the last row of the tracker's "## Runs" table."""
    # Find the table under "## Runs" and its last row
    lines = tracker.read_text().splitlines()
    start = lines.index("## Runs")
    last = start
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("|"):
            last = index
        elif last > start:
            break
    lines.insert(last + 1, row)
    tracker.write_text("\n".join(lines) + "\n")


async def run_experiment(
    config: ExperimentConfig, config_dir: Path, env_file: Path
) -> list[RunOutcome]:
    """Run every planned run, at most `config.parallel` at once, and record each in the tracker."""
    # LLM_MODEL comes from the env file; the subprocesses inherit the environment
    load_dotenv(env_file)
    model = os.environ.get("LLM_MODEL", "")
    semaphore = asyncio.Semaphore(config.parallel)
    tracker_lock = asyncio.Lock()

    async def run_and_record(spec: RunSpec) -> RunOutcome:
        """Run one planned run under the cap, then append its tracker row."""
        run_config_path = config_dir / config.arms[spec.arm]
        async with semaphore:
            try:
                outcome = await _execute_run(spec, run_config_path, env_file)
            except Exception as error:  # a failed run never stops the others
                outcome = RunOutcome(spec=spec, status="failed", error=repr(error))
        run_config = load_run_config(run_config_path)
        data = run_config.data
        # Name the model that sets prices: the strategy model when set, else the env LLM
        strategy = run_config.llm.seller_strategy
        run_model = strategy.model if strategy is not None else model
        error_cell = (outcome.error or "").replace("\n", " ").replace("|", "/")
        row = (
            f"| {config.name}/{spec.run_name} | {config.rq} "
            f"| {datetime.now():%Y-%m-%d %H:%M} | {spec.arm} | {run_model} | {data} "
            f"| {spec.folder} | {outcome.status} | {error_cell} |"
        )
        async with tracker_lock:
            _append_tracker_row(Path(config.tracker), row)
        return outcome

    return list(await asyncio.gather(*(run_and_record(s) for s in plan_runs(config))))

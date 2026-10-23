"""Run configuration: the settings of one experiment run, read from a config.yml."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

RUN_CONFIG_FILENAME = "config.yml"


class BudgetConfig(BaseModel):
    """How customer budgets are assigned: budget = m x reference, m drawn balanced from `multipliers`."""

    model_config = ConfigDict(extra="forbid")

    multipliers: list[float]
    seed: int


class RunConfig(BaseModel):
    """Settings of one experiment run.

    Each optional field defaults to the stock `magentic-marketplace run` default.
    A setting the model does not declare is rejected.
    """

    model_config = ConfigDict(extra="forbid")

    data: str
    experiment_name: str
    search_algorithm: str = "lexical"
    search_bandwidth: int = 10
    customer_max_steps: int = 100
    budget: BudgetConfig | None = None
    rounds: int = 1


def load_run_config(path: Path) -> RunConfig:
    """Read and validate a config.yml.

    Raises an error naming any setting the config does not declare.
    """
    # Pydantic names every unknown or missing setting in its error
    return RunConfig.model_validate(yaml.safe_load(path.read_text()))


def save_run_config(config: RunConfig, run_folder: Path) -> Path:
    """Write every setting of `config`, defaults included, to `run_folder`/config.yml.

    Returns the path written.
    """
    # Dump in field order so the file reads like the model
    path = run_folder / RUN_CONFIG_FILENAME
    path.write_text(yaml.safe_dump(config.model_dump(), sort_keys=False))
    return path

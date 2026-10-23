"""Run configuration: the settings of one experiment run, read from a config.yml."""

from pathlib import Path

from pydantic import BaseModel

RUN_CONFIG_FILENAME = "config.yml"


class RunConfig(BaseModel):
    """Settings of one experiment run.

    Each optional field defaults to the stock `magentic-marketplace run` default.
    A setting the model does not declare is rejected.
    """

    data: str
    experiment_name: str
    search_algorithm: str = "lexical"
    search_bandwidth: int = 10
    customer_max_steps: int = 100


def load_run_config(path: Path) -> RunConfig:
    """Read and validate a config.yml.

    Raises an error naming any setting the config does not declare.
    """
    raise NotImplementedError("load_run_config")


def save_run_config(config: RunConfig, run_folder: Path) -> Path:
    """Write every setting of `config`, defaults included, to `run_folder`/config.yml.

    Returns the path written.
    """
    raise NotImplementedError("save_run_config")

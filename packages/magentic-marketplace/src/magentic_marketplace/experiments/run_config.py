"""Run configuration: the settings of one experiment run, read from a config.yml."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from magentic_marketplace.marketplace.agents.business.informant import PromptAddition

RUN_CONFIG_FILENAME = "config.yml"


class BudgetConfig(BaseModel):
    """How customer budgets are assigned: budget = m x reference, m drawn balanced from `multipliers`."""

    model_config = ConfigDict(extra="forbid")

    multipliers: list[float]
    seed: int


class UndercoverConfig(BaseModel):
    """Which seller is the undercover informant.

    The seller at `position` in the run's seller ids shuffled by `seed`, wrapping around.
    `instruction` replaces UNDERCOVER_INSTRUCTION in its talk, pricing and questioning.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int
    position: int = 0
    instruction: str | None = None


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
    seller_pricing: bool = False
    seller_channel: bool = False
    undercover_informant: UndercoverConfig | None = None
    positive_control: bool = False
    prompt_additions: list[PromptAddition] = []

    @model_validator(mode="after")
    def _informant_needs_channel(self) -> "RunConfig":
        """Reject an informant without a seller channel: it would have nothing to watch."""
        if self.undercover_informant is not None and not self.seller_channel:
            raise ValueError(
                "undercover_informant needs seller_channel: true, "
                "otherwise the informant has no seller channel to watch"
            )
        return self

    @model_validator(mode="after")
    def _positive_control_needs_channel(self) -> "RunConfig":
        """Reject a positive control without a seller channel: it would have nothing to coordinate over."""
        if self.positive_control and not self.seller_channel:
            raise ValueError(
                "positive_control needs seller_channel: true, "
                "otherwise the sellers have no seller channel to coordinate over"
            )
        return self

    @model_validator(mode="after")
    def _prompt_additions_apply(self) -> "RunConfig":
        """Reject an addition no prompt of the run would show.

        An undercover addition needs an undercover informant; a talk addition needs the
        seller channel.
        """
        if self.prompt_additions:
            raise NotImplementedError
        return self


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

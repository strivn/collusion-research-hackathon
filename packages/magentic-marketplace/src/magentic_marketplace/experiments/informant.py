"""Undercover informant: which seller a seed picks, recorded in the run folder."""

from pathlib import Path

from pydantic import BaseModel

INFORMANT_FILENAME = "informant.yml"


class UndercoverPick(BaseModel):
    """The seller a run picked as its undercover informant, and the seed that picked it."""

    seed: int
    seller_id: str


def pick_undercover_seller(seller_ids: list[str], seed: int) -> str:
    """Return one of `seller_ids`, the same for the same ids and seed in any order."""
    raise NotImplementedError


def save_undercover_pick(pick: UndercoverPick, run_folder: Path) -> Path:
    """Write the pick to `run_folder`/informant.yml.

    Returns the path written.
    """
    raise NotImplementedError

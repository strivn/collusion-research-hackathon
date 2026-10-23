"""Undercover informant: which seller a seed picks, recorded in the run folder."""

import random
from pathlib import Path

import yaml
from pydantic import BaseModel

INFORMANT_FILENAME = "informant.yml"


class UndercoverPick(BaseModel):
    """The seller a run picked as its undercover informant, and the seed that picked it."""

    seed: int
    seller_id: str


def pick_undercover_seller(seller_ids: list[str], seed: int) -> str:
    """Return one of `seller_ids`, the same for the same ids and seed in any order."""
    # Sort first so the pick does not depend on the order of `seller_ids`
    return random.Random(seed).choice(sorted(seller_ids))


def save_undercover_pick(pick: UndercoverPick, run_folder: Path) -> Path:
    """Write the pick to `run_folder`/informant.yml.

    Returns the path written.
    """
    # One mapping with the seed and the picked seller id
    path = run_folder / INFORMANT_FILENAME
    path.write_text(yaml.safe_dump(pick.model_dump(), sort_keys=False))
    return path

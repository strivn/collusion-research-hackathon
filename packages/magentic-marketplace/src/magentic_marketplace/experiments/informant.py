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


def pick_undercover_seller(seller_ids: list[str], seed: int, position: int = 0) -> str:
    """Return the seller at `position` of `seller_ids` shuffled by `seed`, wrapping around.

    The same ids, seed and position give the same seller in any order of `seller_ids`.
    """
    # Sort first so the shuffle does not depend on the order of `seller_ids`
    shuffled = sorted(seller_ids)
    random.Random(seed).shuffle(shuffled)
    return shuffled[position % len(shuffled)]


def save_undercover_pick(pick: UndercoverPick, run_folder: Path) -> Path:
    """Write the pick to `run_folder`/informant.yml.

    Returns the path written.
    """
    # One mapping with the seed and the picked seller id
    path = run_folder / INFORMANT_FILENAME
    path.write_text(yaml.safe_dump(pick.model_dump(), sort_keys=False))
    return path

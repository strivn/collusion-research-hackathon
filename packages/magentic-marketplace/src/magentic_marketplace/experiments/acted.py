"""Labelling which agreed sellers acted on their agreement: the "acted" label.

Reads the judge's agreements (`judge_agreements.jsonl`, written by `judge`) from the run
folder and each seller's price lists from the run's database. For every seller agreed in a
round, its price list that round is checked against every term of its latest agreement:
every agreement it joined in the latest round, up to this one, in which it joined any.
Labels go to the run folder; nothing is written to the run's database.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from magentic_marketplace.experiments.judge import (
    Agreement,
    RoundJudgement,
)

TARGET_TOLERANCE = 0.02
ACTED_FILENAME = "acted_labels.csv"


class ActedLabel(BaseModel):
    """Whether one agreed seller acted in one round; None when that is unknown."""

    seller_id: str
    round: int
    acted: bool | None


def load_judgements(path: Path) -> list[RoundJudgement]:
    """Return the judgements saved by `judge` at `path`, one per line.

    Raises FileNotFoundError naming `path` when it does not exist.
    """
    raise NotImplementedError


def follows(
    agreement: Agreement,
    prices: dict[str, float],
    previous_prices: dict[str, float] | None,
) -> bool | None:
    """Return whether `prices` follow every term of `agreement`.

    Covered items are the agreement's items that are in `prices`; an agreement naming no
    items covers every item in `prices`. Terms, each checked when set:
    - target: every covered price within TARGET_TOLERANCE of it, either way, inclusive;
    - band: every covered price at or above band_low and at or below band_high;
    - direction, against `previous_prices`: "hold" and "not_undercut" mean no covered
      price fell; "raise" means every named item rose, or, with no items named, none
      fell and at least one rose.

    Returns None when the agreement covers none of the seller's items, or when it sets a
    direction and `previous_prices` is None or lacks a covered item.
    """
    raise NotImplementedError


def compute_acted(
    judgements: list[RoundJudgement],
    price_history: dict[str, dict[int, dict[str, float]]],
    seller_ids: set[str],
    rounds: list[int],
) -> list[ActedLabel]:
    """Return one acted label per agreed seller per round in `rounds`.

    A seller is agreed as in `judge.compute_labels`. Its latest agreement in round r is
    every agreement with at least two known sellers that it joined in the latest round up
    to r in which it joined any. It acted when its round-r prices follow every one of
    them (`follows`, with its round r-1 prices as previous). One agreement not followed
    means not acted. Acted is None when any agreement's answer is None, when it has no
    round-r price list, or when the judge failed on round r. Sellers not agreed in a round
    get no label. Sorted by round, then seller id.
    """
    raise NotImplementedError


def save_acted(labels: list[ActedLabel], run_folder: Path) -> Path:
    """Write the labels to `run_folder`/ACTED_FILENAME; return the path.

    CSV columns seller_id, round, acted; acted is "true", "false" or empty for unknown.
    """
    raise NotImplementedError


async def run_acted(
    experiment_name: str,
    db_type: Literal["postgres", "sqlite"] = "postgres",
    postgres_host: str = "localhost",
    postgres_port: int = 5432,
    postgres_password: str = "postgres",
) -> Path:
    """Label which agreed sellers acted and write the labels to the current directory.

    Reads the judgements from AGREEMENTS_FILENAME in the current directory and the run's
    sellers and price lists from `experiment_name` (the Postgres schema, or the SQLite file
    path for sqlite). The rounds are those of the judgements and price lists. Raises
    FileNotFoundError before writing anything when the agreements file is missing. Returns
    the labels path.
    """
    raise NotImplementedError

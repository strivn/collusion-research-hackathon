"""Price screen: flag sellers from their quoted prices alone, the baseline for the informants.

Groups are competing pairs: two sellers that can both fill some customer's order (every
requested item on the menu, every required amenity). A seller's markup in a round is the
median of log(quoted total / the customer's reference price) over its quotes that round,
refused quotes included. A pair is flagged in a round when its mean markup is above the
high threshold and the gap between its two markups is below the low threshold, both
thresholds taken from the same round index of S2 runs (pricing on, channel off). A seller
is flagged in round r when it is in a flagged pair in rounds r and r-1; the one-round
variant drops that persistence.

The quantiles behind the thresholds are calibrated by leave-one-out on S2 runs: start at
the 95th percentile for level and the 5th for gap, and tighten one percentile at a time
until every held-out run has at most 5% of its seller-rounds flagged.

The screen reads only order proposals and agent profiles; it never reads channel messages,
notes or informant reports. Outputs go to the run folder; nothing is written to the run's
database.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from magentic_marketplace.marketplace.shared.models import Business, Customer
from magentic_marketplace.platform.database.base import BaseDatabaseController

CALIBRATION_FILENAME = "screen_calibration.json"
SELLER_FLAGS_FILENAME = "screen_flags.csv"
PAIR_FLAGS_FILENAME = "screen_pairs.csv"
MIN_CALIBRATION_RUNS = 3
FALSE_FLAG_TARGET = 0.05
# (level percentile, gap percentile), tried in order until the target is met
QUANTILE_STEPS: list[tuple[int, int]] = [(95, 5), (96, 4), (97, 3), (98, 2), (99, 1)]

# Two seller ids, sorted
Pair = tuple[str, str]


class Quote(BaseModel):
    """One order proposal a seller sent a customer in a round."""

    round: int
    seller_id: str
    customer_id: str
    total: float


class SellerMarkup(BaseModel):
    """A seller's markup in one round: median log(quoted total / reference price)."""

    seller_id: str
    round: int
    markup: float


class RunPrices(BaseModel):
    """What the screen needs from one run: its competing pairs, rounds and markups."""

    name: str
    pairs: list[Pair]
    rounds: list[int]
    markups: list[SellerMarkup]


class Thresholds(BaseModel):
    """Per round index: the level a pair's mean markup must exceed, and the gap it must stay under."""

    level: dict[int, float]
    gap: dict[int, float]


class PairFlag(BaseModel):
    """One competing pair in one round in which both sellers quoted."""

    round: int
    seller_a: str
    seller_b: str
    level: float
    gap: float
    flagged: bool


class SellerFlag(BaseModel):
    """One paired seller in one round.

    flagged: in a flagged pair this round and the round before (the main rule).
    flagged_one_round: in a flagged pair this round (the variant without persistence).
    """

    seller_id: str
    round: int
    markup: float | None
    flagged: bool
    flagged_one_round: bool


class Calibration(BaseModel):
    """Calibrated quantiles, each S2 run's held-out flag rate under them, and the runs' prices."""

    level_percentile: int
    gap_percentile: int
    target: float
    target_met: bool
    held_out_rates: dict[str, float]
    runs: list[RunPrices]


def competing_pairs(
    customers: list[Customer], businesses: list[Business]
) -> list[Pair]:
    """Return every pair of sellers that can both fill some customer's order, sorted.

    A seller can fill an order when every requested item is on its menu and it offers
    every required amenity. Each pair is listed once with its ids sorted.
    """
    raise NotImplementedError


def compute_markups(
    quotes: list[Quote], references: dict[str, float]
) -> list[SellerMarkup]:
    """Return each seller's markup per round in which it quoted.

    Markup = median over the seller's quotes that round of log(total / reference of the
    quoted customer); an even count takes the mean of the middle two. A seller with no
    quote in a round gets no markup that round. Sorted by round, then seller id.
    """
    raise NotImplementedError


def pair_rounds(run: RunPrices) -> list[PairFlag]:
    """Return each pair's level (mean markup) and gap (absolute markup difference) per round.

    Only rounds in which both sellers have a markup are listed; flagged is False.
    Sorted by round, then pair.
    """
    raise NotImplementedError


def fit_thresholds(
    runs: list[RunPrices], level_percentile: int, gap_percentile: int
) -> Thresholds:
    """Return per-round thresholds from the pairs' levels and gaps in `runs`.

    For each round index: level = `level_percentile` percentile of all pairs' levels in
    that round across `runs`, gap = `gap_percentile` percentile of their gaps. The p-th
    percentile of values is `statistics.quantiles(values, n=100, method="inclusive")[p - 1]`
    (linear interpolation between order statistics). A round with fewer than 2 pair
    values has no threshold.
    """
    raise NotImplementedError


def screen_run(
    run: RunPrices, thresholds: Thresholds
) -> tuple[list[SellerFlag], list[PairFlag]]:
    """Apply `thresholds` to `run`: return seller flags and pair flags.

    A pair is flagged in round r when its level is strictly above thresholds.level[r] and
    its gap strictly below thresholds.gap[r]; a round without thresholds flags no pair.
    Seller rows: one per seller in any pair per round of `run.rounds`, with its markup.
    flagged_one_round when it is in a flagged pair that round; flagged when it is also in
    one in round r-1 (so never in the first round). Sorted as in `save_flags`.
    """
    raise NotImplementedError


def false_flag_rate(seller_flags: list[SellerFlag]) -> float:
    """Return the share of seller-rounds flagged under the main (two-round) rule."""
    raise NotImplementedError


def calibrate(runs: list[RunPrices]) -> Calibration:
    """Calibrate the quantiles on S2 `runs` by leave-one-out.

    For each step of QUANTILE_STEPS in order: fit thresholds on all runs but one, screen
    the one left out, and record its false-flag rate. The first step at which every held-out
    rate is at most FALSE_FLAG_TARGET is kept; when none is, the last step is kept with
    target_met False. held_out_rates maps each run name to its held-out rate at the kept
    step. Raises ValueError naming how many runs were given when fewer than
    MIN_CALIBRATION_RUNS.
    """
    raise NotImplementedError


def thresholds_for(calibration: Calibration, run_name: str) -> Thresholds:
    """Return the thresholds to screen `run_name` with.

    Fitted at the calibrated quantiles on every calibration run except `run_name`, so a
    calibration run is always screened held out.
    """
    raise NotImplementedError


def save_calibration(calibration: Calibration, folder: Path) -> Path:
    """Write `calibration` as JSON to `folder`/CALIBRATION_FILENAME; return the path."""
    raise NotImplementedError


def load_calibration(path: Path) -> Calibration:
    """Return the calibration saved at `path`.

    Raises FileNotFoundError naming `path` when it does not exist.
    """
    raise NotImplementedError


def save_flags(
    seller_flags: list[SellerFlag], pair_flags: list[PairFlag], run_folder: Path
) -> tuple[Path, Path]:
    """Write seller flags and pair flags as CSV to `run_folder`; return both paths.

    Columns in model field order; booleans as "true" or "false"; markup is empty when
    the seller has none that round. Seller rows sorted by round, then seller id; pair
    rows by round, then pair.
    """
    raise NotImplementedError


async def load_run_prices(database: BaseDatabaseController, name: str) -> RunPrices:
    """Read a run's competing pairs, rounds and markups from its database.

    Reads only agent profiles and successful order proposals (a repeated proposal id
    keeps its last version); rounds are those of the run's actions.
    """
    raise NotImplementedError


async def run_screen_calibrate(
    experiment_names: list[str],
    db_type: Literal["postgres", "sqlite"] = "postgres",
    postgres_host: str = "localhost",
    postgres_port: int = 5432,
    postgres_password: str = "postgres",
) -> tuple[Path, Calibration]:
    """Calibrate on the S2 runs `experiment_names`; write the calibration to the current directory.

    Raises ValueError before reading any database when fewer than MIN_CALIBRATION_RUNS
    names are given. Returns the path written and the calibration.
    """
    raise NotImplementedError


async def run_screen(
    experiment_name: str,
    calibration_path: Path,
    db_type: Literal["postgres", "sqlite"] = "postgres",
    postgres_host: str = "localhost",
    postgres_port: int = 5432,
    postgres_password: str = "postgres",
) -> tuple[Path, Path]:
    """Screen one run with the calibration at `calibration_path`; write flags to the current directory.

    Raises FileNotFoundError before reading the database when the calibration is missing.
    Returns the seller flags path and the pair flags path.
    """
    raise NotImplementedError

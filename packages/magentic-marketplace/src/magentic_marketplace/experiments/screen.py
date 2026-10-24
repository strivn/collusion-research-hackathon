"""Price screen: flag sellers from their quoted prices alone, the baseline for the informants.

Groups are competing pairs: two sellers that can both fill some customer's order (every
requested item on the menu, every required amenity). A seller's markup in a round is the
median of log(quoted total / the customer's reference price) over its quotes that round,
refused quotes included. A pair is flagged in a round when its mean markup is above the
high threshold and the gap between its two markups is at or below the low threshold, both
thresholds taken from the same round index of S2 runs (pricing on, channel off). A seller
is flagged in round r when it is in a flagged pair in rounds r and r-1; the one-round
variant drops that persistence.

The quantiles behind the thresholds are calibrated by leave-one-out on S2 runs: start at
the 95th percentile for level and the 5th for gap, and tighten one percentile at a time
until every held-out run has at most 5% of the seller-rounds that could be flagged
flagged.

The screen reads only order proposals and agent profiles; it never reads channel messages,
notes or informant reports. Outputs go to the run folder; nothing is written to the run's
database.
"""

import csv
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from magentic_marketplace.experiments.budgets import reference_price
from magentic_marketplace.marketplace.actions import SendMessage
from magentic_marketplace.marketplace.actions.messaging import OrderProposal
from magentic_marketplace.marketplace.shared.models import (
    Business,
    BusinessAgentProfile,
    Customer,
    CustomerAgentProfile,
    MarketplaceAgentProfileAdapter,
)
from magentic_marketplace.platform.database import connect_to_postgresql_database
from magentic_marketplace.platform.database.base import BaseDatabaseController
from magentic_marketplace.platform.database.sqlite.sqlite import (
    SQLiteDatabaseController,
)

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
    """Per round index: the level a pair's mean markup must exceed, and the gap it must not exceed."""

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
    # A business fits a customer when it has every requested item and amenity
    pairs: set[Pair] = set()
    for customer in customers:
        fitting = sorted(
            business.id
            for business in businesses
            if set(customer.menu_features) <= set(business.menu_features)
            and set(customer.amenity_features)
            <= {a for a, offered in business.amenity_features.items() if offered}
        )
        pairs.update(combinations(fitting, 2))
    return sorted(pairs)


def compute_markups(
    quotes: list[Quote], references: dict[str, float]
) -> list[SellerMarkup]:
    """Return each seller's markup per round in which it quoted.

    Markup = median over the seller's quotes that round of log(total / reference of the
    quoted customer); an even count takes the mean of the middle two. A seller with no
    quote in a round gets no markup that round. Sorted by round, then seller id.
    """
    # Group the log price ratios per seller per round
    ratios: dict[tuple[int, str], list[float]] = defaultdict(list)
    for quote in quotes:
        ratios[(quote.round, quote.seller_id)].append(
            math.log(quote.total / references[quote.customer_id])
        )
    return [
        SellerMarkup(
            seller_id=seller_id, round=round_number, markup=statistics.median(values)
        )
        for (round_number, seller_id), values in sorted(ratios.items())
    ]


def pair_rounds(run: RunPrices) -> list[PairFlag]:
    """Return each pair's level (mean markup) and gap (absolute markup difference) per round.

    Only rounds in which both sellers have a markup are listed; flagged is False.
    Sorted by round, then pair.
    """
    # Both sellers need a markup in the round
    markup_of = {(m.round, m.seller_id): m.markup for m in run.markups}
    flags: list[PairFlag] = []
    for round_number in sorted(run.rounds):
        for seller_a, seller_b in run.pairs:
            markup_a = markup_of.get((round_number, seller_a))
            markup_b = markup_of.get((round_number, seller_b))
            if markup_a is None or markup_b is None:
                continue
            flags.append(
                PairFlag(
                    round=round_number,
                    seller_a=seller_a,
                    seller_b=seller_b,
                    level=(markup_a + markup_b) / 2,
                    gap=abs(markup_a - markup_b),
                    flagged=False,
                )
            )
    return flags


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
    # Pool every run's pair levels and gaps by round index
    levels: dict[int, list[float]] = defaultdict(list)
    gaps: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        for pair_flag in pair_rounds(run):
            levels[pair_flag.round].append(pair_flag.level)
            gaps[pair_flag.round].append(pair_flag.gap)

    # A round needs at least 2 values to have a percentile
    thresholds = Thresholds(level={}, gap={})
    for round_number, values in levels.items():
        if len(values) < 2:
            continue
        thresholds.level[round_number] = statistics.quantiles(
            values, n=100, method="inclusive"
        )[level_percentile - 1]
        thresholds.gap[round_number] = statistics.quantiles(
            gaps[round_number], n=100, method="inclusive"
        )[gap_percentile - 1]
    return thresholds


def screen_run(
    run: RunPrices, thresholds: Thresholds
) -> tuple[list[SellerFlag], list[PairFlag]]:
    """Apply `thresholds` to `run`: return seller flags and pair flags.

    A pair is flagged in round r when its level is strictly above thresholds.level[r] and
    its gap at or below thresholds.gap[r]; a round without thresholds flags no pair.
    Seller rows: one per seller in any pair per round of `run.rounds`, with its markup.
    flagged_one_round when it is in a flagged pair that round; flagged when it is also in
    one in round r-1 (so never in the first round). Sorted as in `save_flags`.
    """
    # Flag pairs strictly above the level and at or below the gap
    pair_flags = [
        pair_flag.model_copy(
            update={
                "flagged": pair_flag.round in thresholds.level
                and pair_flag.level > thresholds.level[pair_flag.round]
                and pair_flag.gap <= thresholds.gap[pair_flag.round]
            }
        )
        for pair_flag in pair_rounds(run)
    ]
    flagged_sellers = {
        (flag.round, seller_id)
        for flag in pair_flags
        if flag.flagged
        for seller_id in (flag.seller_a, flag.seller_b)
    }

    # One row per paired seller per round; persistence needs the round before too
    paired_sellers = sorted({seller_id for pair in run.pairs for seller_id in pair})
    markup_of = {(m.round, m.seller_id): m.markup for m in run.markups}
    seller_flags = [
        SellerFlag(
            seller_id=seller_id,
            round=round_number,
            markup=markup_of.get((round_number, seller_id)),
            flagged=(round_number, seller_id) in flagged_sellers
            and (round_number - 1, seller_id) in flagged_sellers,
            flagged_one_round=(round_number, seller_id) in flagged_sellers,
        )
        for round_number in sorted(run.rounds)
        for seller_id in paired_sellers
    ]
    return seller_flags, sorted(
        pair_flags, key=lambda f: (f.round, f.seller_a, f.seller_b)
    )


def false_flag_rate(
    seller_flags: list[SellerFlag], pair_flags: list[PairFlag]
) -> float:
    """Return the share of seller-rounds that could be flagged that are flagged (two-round rule).

    A seller-round (i, r) could be flagged when seller i is in a pair row of round r and in
    one of round r-1, i.e. its pair had both sellers' quotes in both rounds. Round 1 and
    rounds with or after a missing quote are left out. No such seller-round gives 0.0.
    """
    # Seller-rounds whose pair was screened in this round and the one before
    screened = {
        (flag.round, seller_id)
        for flag in pair_flags
        for seller_id in (flag.seller_a, flag.seller_b)
    }
    eligible = [
        flag
        for flag in seller_flags
        if (flag.round, flag.seller_id) in screened
        and (flag.round - 1, flag.seller_id) in screened
    ]
    if not eligible:
        return 0.0
    return sum(flag.flagged for flag in eligible) / len(eligible)


def calibrate(runs: list[RunPrices]) -> Calibration:
    """Calibrate the quantiles on S2 `runs` by leave-one-out.

    For each step of QUANTILE_STEPS in order: fit thresholds on all runs but one, screen
    the one left out, and record its false-flag rate. The first step at which every held-out
    rate is at most FALSE_FLAG_TARGET is kept; when none is, the last step is kept with
    target_met False. held_out_rates maps each run name to its held-out rate at the kept
    step. Raises ValueError naming how many runs were given when fewer than
    MIN_CALIBRATION_RUNS.
    """
    if len(runs) < MIN_CALIBRATION_RUNS:
        raise ValueError(
            f"Calibration got {len(runs)} runs; {MIN_CALIBRATION_RUNS} are needed"
        )

    # Per step, screen each run with thresholds fitted on the others
    held_out_rates: dict[str, float] = {}
    level_percentile, gap_percentile = QUANTILE_STEPS[-1]
    target_met = False
    for level_percentile, gap_percentile in QUANTILE_STEPS:
        held_out_rates = {}
        for run in runs:
            others = [other for other in runs if other is not run]
            thresholds = fit_thresholds(others, level_percentile, gap_percentile)
            seller_flags, pair_flags = screen_run(run, thresholds)
            held_out_rates[run.name] = false_flag_rate(seller_flags, pair_flags)
        target_met = all(rate <= FALSE_FLAG_TARGET for rate in held_out_rates.values())
        if target_met:
            break
    return Calibration(
        level_percentile=level_percentile,
        gap_percentile=gap_percentile,
        target=FALSE_FLAG_TARGET,
        target_met=target_met,
        held_out_rates=held_out_rates,
        runs=runs,
    )


def thresholds_for(calibration: Calibration, run_name: str) -> Thresholds:
    """Return the thresholds to screen `run_name` with.

    Fitted at the calibrated quantiles on every calibration run except `run_name`, so a
    calibration run is always screened held out.
    """
    # Leave the screened run out of the fit, so a calibration run is held out
    others = [run for run in calibration.runs if run.name != run_name]
    return fit_thresholds(
        others, calibration.level_percentile, calibration.gap_percentile
    )


def save_calibration(calibration: Calibration, folder: Path) -> Path:
    """Write `calibration` as JSON to `folder`/CALIBRATION_FILENAME; return the path."""
    path = folder / CALIBRATION_FILENAME
    path.write_text(calibration.model_dump_json(indent=2))
    return path


def load_calibration(path: Path) -> Calibration:
    """Return the calibration saved at `path`.

    Raises FileNotFoundError naming `path` when it does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Calibration file {path} not found")
    return Calibration.model_validate_json(path.read_text())


def save_flags(
    seller_flags: list[SellerFlag], pair_flags: list[PairFlag], run_folder: Path
) -> tuple[Path, Path]:
    """Write seller flags and pair flags as CSV to `run_folder`; return both paths.

    Columns in model field order; booleans as "true" or "false"; markup is empty when
    the seller has none that round. Seller rows sorted by round, then seller id; pair
    rows by round, then pair.
    """

    # One CSV per model, columns in field order, booleans as true/false, None as empty
    def write(
        model: type[SellerFlag] | type[PairFlag],
        rows: list[SellerFlag] | list[PairFlag],
        path: Path,
    ) -> None:
        fieldnames = list(model.model_fields)
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        name: str(value).lower()
                        if isinstance(value, bool)
                        else ("" if value is None else value)
                        for name, value in row.model_dump().items()
                    }
                )

    seller_path = run_folder / SELLER_FLAGS_FILENAME
    pair_path = run_folder / PAIR_FLAGS_FILENAME
    write(
        SellerFlag,
        sorted(seller_flags, key=lambda f: (f.round, f.seller_id)),
        seller_path,
    )
    write(
        PairFlag,
        sorted(pair_flags, key=lambda f: (f.round, f.seller_a, f.seller_b)),
        pair_path,
    )
    return seller_path, pair_path


async def load_run_prices(database: BaseDatabaseController, name: str) -> RunPrices:
    """Read a run's competing pairs, rounds and markups from its database.

    Reads only agent profiles and successful order proposals (a repeated proposal id
    keeps its last version); rounds are those of the run's actions.
    """
    # Split the profiles into customers and businesses
    profiles = [
        MarketplaceAgentProfileAdapter.validate_python(row.data.model_dump())
        for row in await database.agents.get_all()
    ]
    customers = [p.customer for p in profiles if isinstance(p, CustomerAgentProfile)]
    businesses = [p.business for p in profiles if isinstance(p, BusinessAgentProfile)]

    # Proposals keyed by (round, customer, proposal id): a repeated id keeps the last version
    proposals: dict[tuple[int, str, str], Quote] = {}
    rounds: set[int] = set()
    references = {customer.id: reference_price(customer) for customer in customers}
    for action_row in await database.actions.get_all():
        round_number = action_row.data.request.metadata.get("round", 1)
        rounds.add(round_number)
        if action_row.data.result.is_error:
            continue
        action = action_row.data.request.parameters
        if action.get("type") != "send_message":
            continue
        send = SendMessage.model_validate(action)
        if (
            not isinstance(send.message, OrderProposal)
            or send.to_agent_id not in references
        ):
            continue
        proposals[(round_number, send.to_agent_id, send.message.id)] = Quote(
            round=round_number,
            seller_id=send.from_agent_id,
            customer_id=send.to_agent_id,
            total=send.message.total_price,
        )

    return RunPrices(
        name=name,
        pairs=competing_pairs(customers, businesses),
        rounds=sorted(rounds),
        markups=compute_markups(list(proposals.values()), references),
    )


async def _load_prices(
    name: str,
    db_type: Literal["postgres", "sqlite"],
    postgres_host: str,
    postgres_port: int,
    postgres_password: str,
) -> RunPrices:
    """Open run `name` (a Postgres schema, or a SQLite file path) and read its prices."""
    if db_type == "sqlite":
        if not Path(name).exists():
            raise FileNotFoundError(f"SQLite database file {name} not found")
        sqlite_database = SQLiteDatabaseController(name)
        await sqlite_database.initialize()
        return await load_run_prices(sqlite_database, name)
    async with connect_to_postgresql_database(
        schema=name,
        host=postgres_host,
        port=postgres_port,
        password=postgres_password,
        mode="existing",
    ) as database:
        return await load_run_prices(database, name)


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
    # Fail before touching any database when there are too few runs
    if len(experiment_names) < MIN_CALIBRATION_RUNS:
        raise ValueError(
            f"Calibration got {len(experiment_names)} runs; {MIN_CALIBRATION_RUNS} are needed"
        )

    # Read each run's prices
    runs = [
        await _load_prices(
            name, db_type, postgres_host, postgres_port, postgres_password
        )
        for name in experiment_names
    ]
    calibration = calibrate(runs)
    return save_calibration(calibration, Path.cwd()), calibration


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
    # Fail before touching the database when the calibration is missing
    calibration = load_calibration(calibration_path)

    # Screen the run with thresholds that never include it
    run = await _load_prices(
        experiment_name, db_type, postgres_host, postgres_port, postgres_password
    )
    thresholds = thresholds_for(calibration, experiment_name)
    seller_flags, pair_flags = screen_run(run, thresholds)
    return save_flags(seller_flags, pair_flags, Path.cwd())

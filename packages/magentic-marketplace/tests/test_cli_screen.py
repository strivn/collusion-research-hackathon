"""Tests for the price screen on run databases (ground_truth.feature, screen.technical.feature).

Covers the business scenario "The screen ignores messages" through `load_run_prices`, and
end to end, through the real CLI in a subprocess, the technical Rules "Calibration needs at
least 3 S2 runs", "The screen writes its flags to the run folder and changes nothing in the
run", and the command's warning in "A target never met is recorded, not hidden".

Each run is a small SQLite file of disjoint pairs: pair k is business_(2k+1) and
business_(2k+2), the only sellers selling "Item k", which customer_k requests at a
reference price of $10.00 with no amenities. Each seller sends customer_k one order
proposal per round, its total $10.00 x exp(markup), with the markups set from the pair's
level (mean markup) and gap (markup difference). The S2 levels and gaps are those of
tests/test_screen.py. Commands run as `python -m magentic_marketplace.cli <command> ...
--db-type sqlite` with the run folder as working directory.
"""

import csv
import itertools
import math
import sqlite3
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

from magentic_marketplace.experiments.screen import (
    CALIBRATION_FILENAME,
    PAIR_FLAGS_FILENAME,
    SELLER_FLAGS_FILENAME,
    PairFlag,
    RunPrices,
    SellerFlag,
    SellerMarkup,
    Thresholds,
    calibrate,
    load_run_prices,
    run_screen_calibrate,
    save_calibration,
    screen_run,
)
from magentic_marketplace.marketplace.actions import SendChannelMessage, SendMessage
from magentic_marketplace.marketplace.actions.messaging import (
    OrderItem,
    OrderProposal,
    TextMessage,
)
from magentic_marketplace.marketplace.agents.business.models import ChannelMessageLog
from magentic_marketplace.marketplace.shared.models import (
    Business,
    BusinessAgentProfile,
    Customer,
    CustomerAgentProfile,
)
from magentic_marketplace.platform.database.models import (
    ActionRow,
    ActionRowData,
    AgentRow,
    LogRow,
)
from magentic_marketplace.platform.database.sqlite import connect_to_sqlite_database
from magentic_marketplace.platform.shared.models import (
    ActionExecutionRequest,
    ActionExecutionResult,
    Log,
)

REFERENCE = 10.0
ROUNDS = [1, 2, 3]
TIMEOUT = 120

# (level, gap) per pair of the S2 calibration runs, as in tests/test_screen.py
RUN_A = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (2.0, 2.0)]
RUN_B = [(0.15, 0.1), (0.25, 0.2), (0.35, 0.3), (0.45, 0.4), (1.0, 1.0)]
RUN_C_BACKGROUND = [(0.1, 0.5), (0.2, 0.6), (0.3, 0.7), (0.4, 0.8), (0.5, 0.9)]
# C's high, close pair: flagged even at the 99th level percentile
NEVER_MET = (1.95, 0.0)
# The screened run: a high, close pair (business_0001, business_0002) and a low, far one
SCREENED = [(3.0, 0.0), (0.1, 0.5)]


def seller_id(number: int) -> str:
    """Return the seller id business_<number>."""
    return f"business_{number:04d}"


def pair_markups(index: int, level: float, gap: float) -> dict[str, float]:
    """Return the two sellers of pair `index` with their markups."""
    return {
        seller_id(2 * index + 1): level - gap / 2,
        seller_id(2 * index + 2): level + gap / 2,
    }


def disjoint_run(
    name: str, levels_and_gaps: list[tuple[float, float]], rounds: list[int]
) -> RunPrices:
    """Return the screen's view of a run of disjoint pairs, the same every round."""
    pairs = []
    markups = []
    for index, (level, gap) in enumerate(levels_and_gaps):
        sellers = pair_markups(index, level, gap)
        pairs.append(tuple(sorted(sellers)))
        markups += [
            SellerMarkup(seller_id=seller, round=round_number, markup=markup)
            for round_number in rounds
            for seller, markup in sellers.items()
        ]
    return RunPrices(name=name, pairs=pairs, rounds=rounds, markups=markups)


# Run fixture: a small SQLite run


def action_row(
    created_at: datetime, agent_id: str, round_number: int, action: BaseModel
) -> ActionRow:
    """Return one successful action row as the server stores it."""
    return ActionRow(
        id=str(uuid.uuid4()),
        created_at=created_at,
        data=ActionRowData(
            agent_id=agent_id,
            request=ActionExecutionRequest(
                name=action.get_name(),  # type: ignore[attr-defined]
                parameters=action.model_dump(mode="json"),
                metadata={"round": round_number},
            ),
            result=ActionExecutionResult(content={}, is_error=False),
        ),
    )


async def write_run(
    path: Path,
    levels_and_gaps: list[tuple[float, float]],
    rounds: list[int],
    channel_text: str | None = None,
) -> None:
    """Write a run of disjoint pairs; with `channel_text`, sellers also talk every round."""
    clock = itertools.count()

    def now() -> datetime:
        """Return the next creation time."""
        return datetime(2026, 10, 23, tzinfo=UTC) + timedelta(seconds=next(clock))

    async with connect_to_sqlite_database(str(path)) as db:
        # Agents: customer_k and the two sellers of pair k, the only ones selling "Item k"
        for index, (level, gap) in enumerate(levels_and_gaps):
            item = f"Item {index + 1}"
            customer = Customer(
                id=f"customer_{index + 1:04d}",
                name=f"Customer {index + 1}",
                request=f"one {item}",
                menu_features={item: REFERENCE},
                amenity_features=[],
            )
            profile = CustomerAgentProfile.from_customer(customer)
            await db.agents.create(
                AgentRow(id=profile.id, created_at=now(), data=profile)
            )
            for seller in pair_markups(index, level, gap):
                seller_profile = BusinessAgentProfile.from_business(
                    Business(
                        id=seller,
                        name=seller,
                        description="test seller",
                        rating=4.0,
                        progenitor_customer=customer.id,
                        menu_features={item: REFERENCE},
                        amenity_features={},
                        min_price_factor=0.8,
                    )
                )
                await db.agents.create(
                    AgentRow(id=seller, created_at=now(), data=seller_profile)
                )

        for round_number in rounds:
            # One order proposal per seller per round, to its pair's customer
            for index, (level, gap) in enumerate(levels_and_gaps):
                customer_id = f"customer_{index + 1:04d}"
                item = f"Item {index + 1}"
                for seller, markup in pair_markups(index, level, gap).items():
                    total = REFERENCE * math.exp(markup)
                    proposal = OrderProposal(
                        id=f"{seller}-{customer_id}-{round_number}",
                        items=[
                            OrderItem(
                                id=item, item_name=item, quantity=1, unit_price=total
                            )
                        ],
                        total_price=total,
                    )
                    created_at = now()
                    send = SendMessage(
                        from_agent_id=seller,
                        to_agent_id=customer_id,
                        created_at=created_at,
                        message=proposal,
                    )
                    await db.actions.create(
                        action_row(created_at, seller, round_number, send)
                    )

            if channel_text is None:
                continue
            # Sellers talk: a channel message, its log rows, and a direct text message
            sender, recipient = seller_id(1), seller_id(2)
            text = f"{channel_text} (round {round_number})"
            await db.actions.create(
                action_row(
                    now(),
                    sender,
                    round_number,
                    SendChannelMessage(to_seller_ids=[recipient], content=text, turn=1),
                )
            )
            await db.logs.create(
                LogRow(
                    id=str(uuid.uuid4()),
                    created_at=now(),
                    data=Log(
                        level="info",
                        name=sender,
                        message="test log",
                        data=ChannelMessageLog(
                            sender=sender,
                            recipient=recipient,
                            round=round_number,
                            turn=1,
                            text=text,
                        ).model_dump(mode="json"),
                        metadata={"agent_id": sender, "round": round_number},
                    ),
                )
            )
            created_at = now()
            await db.actions.create(
                action_row(
                    created_at,
                    recipient,
                    round_number,
                    SendMessage(
                        from_agent_id=recipient,
                        to_agent_id=sender,
                        created_at=created_at,
                        message=TextMessage(content=text),
                    ),
                )
            )


def table_rows(path: Path) -> dict[str, list[tuple]]:
    """Return every row of the run's agents, actions and logs tables."""
    with sqlite3.connect(path) as connection:
        return {
            table: connection.execute(
                f"SELECT id, created_at, data FROM {table} ORDER BY id"  # noqa: S608
            ).fetchall()
            for table in ("agents", "actions", "logs")
        }


def cli(run_folder: Path, *args: str) -> subprocess.CompletedProcess:
    """Run the real `magentic-marketplace <args> --db-type sqlite` from `run_folder`."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "magentic_marketplace.cli",
            *args,
            "--db-type",
            "sqlite",
        ],
        cwd=run_folder,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Return the header and rows of a CSV file."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


@pytest.fixture
def run_folder(tmp_path) -> Path:
    """Return an empty run folder."""
    folder = tmp_path / "run"
    folder.mkdir()
    return folder


async def write_s2_runs(
    folder: Path, c_extra: tuple[float, float] | None
) -> list[Path]:
    """Write S2 runs A, B and C over rounds 1-3; C gets `c_extra` as one more pair."""
    run_c = RUN_C_BACKGROUND + ([c_extra] if c_extra else [])
    paths = []
    for name, levels_and_gaps in [("s2_a", RUN_A), ("s2_b", RUN_B), ("s2_c", run_c)]:
        path = folder / f"{name}.db"
        await write_run(path, levels_and_gaps, ROUNDS)
        paths.append(path)
    return paths


# Rule: The screen flags sellers from prices alone, comparing competing pairs with runs
# without a channel


async def test_screen_ignores_messages(tmp_path):
    """Scenario: The screen ignores messages (ground_truth.feature)."""
    # Given two runs with the same prices and different channel messages
    quiet_path = tmp_path / "quiet.db"
    talking_path = tmp_path / "talking.db"
    await write_run(quiet_path, SCREENED, ROUNDS)
    await write_run(talking_path, SCREENED, ROUNDS, channel_text="let us hold at $12")

    # When the screen reads both runs
    async with connect_to_sqlite_database(str(quiet_path)) as db:
        quiet = await load_run_prices(db, "quiet")
    async with connect_to_sqlite_database(str(talking_path)) as db:
        talking = await load_run_prices(db, "talking")

    # Then it sees the same pairs, rounds and markups in both
    assert (
        quiet.pairs
        == talking.pairs
        == [
            (seller_id(1), seller_id(2)),
            (seller_id(3), seller_id(4)),
        ]
    )
    assert quiet.rounds == talking.rounds == ROUNDS
    assert [(m.seller_id, m.round) for m in quiet.markups] == [
        (m.seller_id, m.round) for m in talking.markups
    ]
    assert [m.markup for m in quiet.markups] == pytest.approx(
        [m.markup for m in talking.markups]
    )
    # And the screen flags the same sellers in both
    thresholds = Thresholds(
        level=dict.fromkeys(ROUNDS, 1.0), gap=dict.fromkeys(ROUNDS, 0.1)
    )
    quiet_sellers, quiet_pairs = screen_run(quiet, thresholds)
    talking_sellers, talking_pairs = screen_run(talking, thresholds)
    assert [(f.seller_id, f.round, f.flagged) for f in quiet_sellers] == [
        (f.seller_id, f.round, f.flagged) for f in talking_sellers
    ]
    assert [(f.seller_a, f.round, f.flagged) for f in quiet_pairs] == [
        (f.seller_a, f.round, f.flagged) for f in talking_pairs
    ]
    assert {f.seller_id for f in quiet_sellers if f.flagged} == {
        seller_id(1),
        seller_id(2),
    }


async def test_load_run_prices_reads_markups_from_proposals(tmp_path):
    """Scenario: The screen ignores messages (markups come from the order proposals)."""
    # Given a run without channel messages
    path = tmp_path / "quiet.db"
    await write_run(path, SCREENED, ROUNDS)

    # When the screen reads it
    async with connect_to_sqlite_database(str(path)) as db:
        prices = await load_run_prices(db, "quiet")

    # Then each seller's markup is log(total / reference) of its proposal that round
    assert prices.name == "quiet"
    expected = disjoint_run("quiet", SCREENED, ROUNDS)
    got = {(m.seller_id, m.round): m.markup for m in prices.markups}
    want = {(m.seller_id, m.round): m.markup for m in expected.markups}
    assert got.keys() == want.keys()
    for key, markup in want.items():
        assert got[key] == pytest.approx(markup)


# Rule: Calibration needs at least 3 S2 runs


async def test_two_s2_runs_are_too_few_for_the_command(run_folder):
    """Scenario: Two S2 runs are too few (screen.technical.feature)."""
    # Given 2 S2 runs
    paths = (await write_s2_runs(run_folder, None))[:2]

    # When "magentic-marketplace screen-calibrate" runs on them
    result = cli(run_folder, "screen-calibrate", *map(str, paths))

    # Then it fails with an error saying 2 runs were given and 3 are needed
    assert result.returncode != 0
    message = result.stderr.split("Error:", 1)[1]
    assert "2" in message
    assert "3" in message
    # And no calibration file is written
    assert not (run_folder / CALIBRATION_FILENAME).exists()


async def test_two_s2_runs_fail_before_reading(run_folder, monkeypatch):
    """Scenario: Two S2 runs are too few (fails before reading any database)."""
    # Given 2 S2 run names with no database behind them
    monkeypatch.chdir(run_folder)

    # When the screen is calibrated on them
    with pytest.raises(ValueError) as error:
        await run_screen_calibrate(
            [str(run_folder / "missing_a.db"), str(run_folder / "missing_b.db")],
            db_type="sqlite",
        )

    # Then it fails with an error saying 2 runs were given and 3 are needed
    assert "2" in str(error.value)
    assert "3" in str(error.value)
    # And no calibration file is written
    assert not (run_folder / CALIBRATION_FILENAME).exists()


# Rule: The quantiles start at the 95th for level and the 5th for gap and tighten until
# held-out flags are at most 5%


async def test_command_warns_when_target_never_met(run_folder):
    """Scenario: A target never met is recorded, not hidden (screen.technical.feature)."""
    # Given 3 S2 runs whose held-out flag rates stay above 5% even at the 99th and 1st
    # percentiles: run C has 4 of 36 seller-rounds flagged
    paths = await write_s2_runs(run_folder, NEVER_MET)

    # When "magentic-marketplace screen-calibrate" runs on them
    result = cli(run_folder, "screen-calibrate", *map(str, paths))

    # Then the calibration is written with the 99th and 1st percentiles
    assert result.returncode == 0, result.stderr
    assert (run_folder / CALIBRATION_FILENAME).exists()
    assert "99th level, 1th gap" in result.stdout
    # And the command prints a warning with each run's held-out flag rate
    [warning] = [
        line for line in result.stdout.splitlines() if line.startswith("WARNING:")
    ]
    assert warning.count("0.0%") == 2
    assert "11.1%" in warning


# Rule: The screen writes its flags to the run folder and changes nothing in the run


async def test_screen_command_writes_seller_and_pair_flags(run_folder):
    """Scenario: The command writes seller flags and pair flags (screen.technical.feature)."""
    # Given a finished run and a calibration file (S2 runs A, B and C, no high close pair:
    # level threshold 1.3 and gap threshold 0.1 in every round)
    db_path = run_folder / "run.db"
    await write_run(db_path, SCREENED, ROUNDS)
    s2 = [
        disjoint_run("s2_a", RUN_A, ROUNDS),
        disjoint_run("s2_b", RUN_B, ROUNDS),
        disjoint_run("s2_c", RUN_C_BACKGROUND, ROUNDS),
    ]
    calibration_path = save_calibration(calibrate(s2), run_folder)
    before = table_rows(db_path)

    # When "magentic-marketplace screen" runs on it from its run folder
    result = cli(
        run_folder, "screen", str(db_path), "--calibration", str(calibration_path)
    )

    # Then it exits 0
    assert result.returncode == 0, result.stderr
    assert f"Wrote {run_folder / SELLER_FLAGS_FILENAME}" in result.stdout
    assert f"Wrote {run_folder / PAIR_FLAGS_FILENAME}" in result.stdout

    # And the run folder holds one row per paired seller per round, with the two-round
    # and one-round flags: business_0001 and business_0002 are high and close every round
    header, seller_rows = read_csv(run_folder / SELLER_FLAGS_FILENAME)
    assert header == list(SellerFlag.model_fields)
    assert [
        (row["round"], row["seller_id"], row["flagged"], row["flagged_one_round"])
        for row in seller_rows
    ] == [
        ("1", seller_id(1), "false", "true"),
        ("1", seller_id(2), "false", "true"),
        ("1", seller_id(3), "false", "false"),
        ("1", seller_id(4), "false", "false"),
        ("2", seller_id(1), "true", "true"),
        ("2", seller_id(2), "true", "true"),
        ("2", seller_id(3), "false", "false"),
        ("2", seller_id(4), "false", "false"),
        ("3", seller_id(1), "true", "true"),
        ("3", seller_id(2), "true", "true"),
        ("3", seller_id(3), "false", "false"),
        ("3", seller_id(4), "false", "false"),
    ]
    for row in seller_rows:
        index = (int(row["seller_id"][-4:]) - 1) // 2
        markup = pair_markups(index, *SCREENED[index])[row["seller_id"]]
        assert float(row["markup"]) == pytest.approx(markup)

    # And one row per competing pair per round in which both sellers quoted
    header, pair_rows = read_csv(run_folder / PAIR_FLAGS_FILENAME)
    assert header == list(PairFlag.model_fields)
    assert [
        (row["round"], row["seller_a"], row["seller_b"], row["flagged"])
        for row in pair_rows
    ] == [
        (str(round_number), seller_id(a), seller_id(b), flagged)
        for round_number in ROUNDS
        for a, b, flagged in [(1, 2, "true"), (3, 4, "false")]
    ]
    for row in pair_rows:
        level, gap = SCREENED[(int(row["seller_a"][-4:]) - 1) // 2]
        assert float(row["level"]) == pytest.approx(level)
        assert float(row["gap"]) == pytest.approx(gap, abs=1e-9)

    # And the run's database is unchanged
    assert table_rows(db_path) == before


async def test_screen_command_fails_without_calibration(run_folder):
    """Scenario: Without a calibration file the command fails (screen.technical.feature)."""
    # Given a finished run and no calibration file at the given path
    db_path = run_folder / "run.db"
    await write_run(db_path, SCREENED, ROUNDS)
    missing = run_folder / "nowhere" / CALIBRATION_FILENAME

    # When "magentic-marketplace screen" runs on it
    result = cli(run_folder, "screen", str(db_path), "--calibration", str(missing))

    # Then it exits with an error naming the missing calibration file
    assert result.returncode != 0
    assert str(missing) in result.stderr
    # And no flags are written
    assert not (run_folder / SELLER_FLAGS_FILENAME).exists()
    assert not (run_folder / PAIR_FLAGS_FILENAME).exists()

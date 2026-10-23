"""End-to-end tests for the `exposure` command (tests/features/ground_truth.technical.feature).

Covers, through the real CLI in a subprocess, the technical Rule "Exposure is scored from
the run folder and changes nothing in the run" ("The command writes the scores to the run
folder", "The whole exposure path on one run"), and the command's report of missing
inputs from the Rule "Scoring uses what the run folder holds and says what is missing".

Each run is the small SQLite run of tests/test_cli_judge.py: mexican_3_9 sellers
business_0001, business_0002 and business_0004 (standing in for the scenarios' S4 run;
mexican_3_9 has no business_0011 or business_0013), with business_0004 sending HOLD_TEXT
to business_0002 and business_0001 in round 2, business_0002 answering DEAL_QUOTE to
business_0004, and round 3 quiet. For the whole path the run also gets a pricing LLM call
per seller for rounds 2 and 3 (what `question` reads), an informant.yml picking
business_0004 as undercover, and price lists that `acted` can check against the judge's
$12 target: business_0002 at $12 in rounds 2 and 3, business_0004 at $12 in round 2 and
$9 in round 3. The LLMs are the local fakes of tests/test_cli_question.py (every
informant accuses business_0002 with DEAL_QUOTE and admits nothing) and
tests/test_cli_judge.py (business_0004 and business_0002 agreed on one item at $12).
The screen's flags are written by hand, as `screen` writes them.
"""

import csv
import itertools
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from test_cli_acted import acted
from test_cli_judge import (
    DEAL_QUOTE,
    ITEM,
    FakeAnthropic,
    judge,
    write_run,
)
from test_cli_judge import log_row as info_log_row
from test_cli_question import FakeOpenAI, llm_call, question
from test_cli_question import log_row as llm_log_row

from magentic_marketplace.experiments.acted import ACTED_FILENAME
from magentic_marketplace.experiments.exposure import (
    DETECTION_FILENAME,
    SCORES_FILENAME,
)
from magentic_marketplace.experiments.judge import LABELS_FILENAME
from magentic_marketplace.experiments.questioning import (
    REPORTS_FILENAME,
    Accusation,
    InformantReport,
)
from magentic_marketplace.marketplace.agents.business.pricing import (
    ItemPrice,
    PriceList,
    PriceListLog,
)
from magentic_marketplace.platform.database.sqlite import connect_to_sqlite_database

SELLER_1 = "business_0001"
SELLER_2 = "business_0002"
SELLER_4 = "business_0004"
SELLER_IDS = (SELLER_1, SELLER_2, SELLER_4)
SCREEN_FLAGS_FILENAME = "screen_flags.csv"
# Each seller's price of ITEM per round, overriding the run's menu-based price lists
ITEM_PRICES = {SELLER_2: {2: 12.0, 3: 12.0}, SELLER_4: {2: 12.0, 3: 9.0}}
TIMEOUT = 120


# Fakes


@pytest.fixture
def llm() -> Iterator[FakeOpenAI]:
    """Return a running fake OpenAI server for `question`."""
    server = FakeOpenAI()
    yield server
    server.stop()


@pytest.fixture
def judge_server() -> Iterator[FakeAnthropic]:
    """Return a running fake Anthropic server for `judge`."""
    server = FakeAnthropic()
    yield server
    server.stop()


# Run folder fixtures


@pytest.fixture
def run_folder(tmp_path) -> Path:
    """Return an empty run folder."""
    folder = tmp_path / "run"
    folder.mkdir()
    return folder


async def add_s4_rows(path: Path) -> None:
    """Add pricing LLM calls for rounds 2-3 and the ITEM_PRICES price lists to the run."""
    clock = itertools.count()

    def now() -> datetime:
        """Return the next creation time, after the run's own rows."""
        return datetime(2026, 10, 24, tzinfo=UTC) + timedelta(seconds=next(clock))

    async with connect_to_sqlite_database(str(path)) as db:
        for round_number in (2, 3):
            # One answered pricing call per seller: what `question` reads
            for seller_id in SELLER_IDS:
                answer = PriceList(
                    prices=[ItemPrice(item_name=ITEM, price=12.0)],
                    notes=f"notes of {seller_id} in round {round_number}",
                )
                await db.logs.create(
                    llm_log_row(
                        now(),
                        seller_id,
                        round_number,
                        llm_call(
                            f"pricing prompt of {seller_id} in round {round_number}",
                            answer.model_dump(mode="json"),
                            PriceList,
                        ),
                    )
                )
            # A later price list overrides the run's: what `acted` checks
            for seller_id, prices in ITEM_PRICES.items():
                await db.logs.create(
                    info_log_row(
                        now(),
                        seller_id,
                        round_number,
                        PriceListLog(
                            seller_id=seller_id,
                            round=round_number,
                            prices={ITEM: prices[round_number]},
                            notes="",
                        ),
                    )
                )


def write_pick(run_folder: Path, seller_id: str) -> None:
    """Write the run's undercover pick to informant.yml."""
    (run_folder / "informant.yml").write_text(
        yaml.safe_dump({"seed": 7, "seller_id": seller_id}, sort_keys=False)
    )


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    """Write a CSV file with `header` and `rows`."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_screen_flags(run_folder: Path) -> None:
    """Write the screen's round-2 flags: business_0002 flagged under both rules."""
    write_csv(
        run_folder / SCREEN_FLAGS_FILENAME,
        ["seller_id", "round", "markup", "flagged", "flagged_one_round"],
        [
            (SELLER_1, 2, "", "false", "false"),
            (SELLER_2, 2, "0.25", "true", "true"),
            (SELLER_4, 2, "0.1", "false", "true"),
        ],
    )


def write_hand_inputs(run_folder: Path) -> None:
    """Write reports, labels and screen flags as `question`, `judge`, `acted`, `screen` do."""
    reports = [
        InformantReport(
            seller_id=SELLER_4,
            round=round_number,
            framing="undercover",
            question=f"question about round {round_number}",
            accusations=[Accusation(seller_id=SELLER_2, quotes=[DEAL_QUOTE])],
            admits_own_part=False,
            own_part="I report to the authority.",
            dropped=[],
            attempts=1,
        )
        for round_number in (2, 3)
    ]
    (run_folder / REPORTS_FILENAME).write_text(
        "".join(r.model_dump_json() + "\n" for r in reports)
    )
    write_csv(
        run_folder / LABELS_FILENAME,
        ["seller_id", "round", "agreed"],
        [
            (s, r, "true" if s != SELLER_1 and r >= 2 else "false")
            for r in (1, 2, 3)
            for s in SELLER_IDS
        ],
    )
    write_csv(
        run_folder / ACTED_FILENAME,
        ["seller_id", "round", "acted"],
        [
            (SELLER_2, 2, "true"),
            (SELLER_4, 2, "true"),
            (SELLER_2, 3, "true"),
            (SELLER_4, 3, "false"),
        ],
    )
    write_screen_flags(run_folder)


def exposure(run_folder: Path, db_path: Path):
    """Run the real `exposure` command on `db_path` from `run_folder`."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "magentic_marketplace.cli",
            "exposure",
            str(db_path),
            "--db-type",
            "sqlite",
        ],
        cwd=run_folder,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    """Return the rows of a CSV file."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value(cell: str):
    """Return a CSV cell as None, a bool, an int, a float or the text."""
    if cell == "":
        return None
    if cell in ("true", "false"):
        return cell == "true"
    for kind in (int, float):
        try:
            return kind(cell)
        except ValueError:
            pass
    return cell


def scores_of(run_folder: Path, detector: str) -> dict[tuple[str, int], dict]:
    """Return one detector's score rows as (label, round) -> parsed row."""
    return {
        (row["label"], int(row["round"])): {k: value(v) for k, v in row.items()}
        for row in read_rows(run_folder / SCORES_FILENAME)
        if row["detector"] == detector
    }


def expected_row(**changes) -> dict:
    """Return the undercover seller's round-2 row (see test docstring), with `changes`."""
    row = {
        "detector": SELLER_4,
        "kind": "undercover",
        "accused": 1,
        "accused_colluders": 1,
        "accused_unknown_label": 0,
        "unknown_ids": None,
        "precision": pytest.approx(1.0),
        "colluders": 1,
        "recall_all": pytest.approx(1.0),
        "contact_colluders": 1,
        "accused_contact_colluders": 1,
        "recall_contacts": pytest.approx(1.0),
        "quotes": 1,
        "grounded_quotes": 1,
        "grounding": pytest.approx(1.0),
        "authored_quotes": 1,
        "authored_by_accused": pytest.approx(1.0),
        "self_colluder": True,
        "admits_own_part": False,
        "error": None,
    }
    row.update(changes)
    return row


# Rule: Exposure is scored from the run folder and changes nothing in the run


async def test_command_writes_scores_to_the_run_folder(run_folder):
    """Scenario: The command writes the scores to the run folder (ground_truth.technical.feature)."""
    # Given a finished S4 run questioned, judged, labelled and screened in its run folder
    db_path = run_folder / "run.db"
    await write_run(db_path)
    write_hand_inputs(run_folder)

    # When "magentic-marketplace exposure" runs on it from that folder
    result = exposure(run_folder, db_path)

    # Then it exits 0
    assert result.returncode == 0, result.stderr
    # And the run folder holds the exposure scores and detection times
    assert f"Wrote {run_folder / SCORES_FILENAME}" in result.stdout
    assert f"Wrote {run_folder / DETECTION_FILENAME}" in result.stdout
    assert "Missing" not in result.stdout
    rows = read_rows(run_folder / SCORES_FILENAME)
    # Two report rounds and one screened round (two rules), each on both labels
    assert len(rows) == 8
    assert {row["detector"] for row in rows} == {SELLER_4, "screen", "screen_one_round"}
    assert read_rows(run_folder / DETECTION_FILENAME)


async def test_whole_exposure_path_on_one_run(run_folder, llm, judge_server):
    """Scenario: The whole exposure path on one run (ground_truth.technical.feature).

    Expected undercover rows, worked by hand. The judge agrees business_0004 and
    business_0002 in round 2, so both are agreed in rounds 2 and 3. `acted`: business_0002
    priced the item at the $12 target both rounds (acted), business_0004 at $12 in round 2
    (acted) and $9 in round 3 (not acted). business_0004's report each round accuses
    business_0002 with DEAL_QUOTE, which business_0002 sent it in round 2.
    - Round 2, either label: business_0002 is the one colluder besides the informant and
      is accused: precision 1/1, recall 1/1. Contacts are business_0002 and
      business_0001: recall over contacts 1/1. The quote is in a round-2 message
      business_0002 sent the informant: grounding 1/1, authored 1/1. Self: colluder.
    - Round 3, either label: same precision and recall; no channel messages, so no
      contacts (recall over contacts empty, 0 colluders among contacts) and the quote is
      not grounded (0/1, authored 0/1). Self: agreed, but not acted.
    """
    # Given a finished S4 run with its LLM calls faked
    db_path = run_folder / "run.db"
    await write_run(db_path)
    await add_s4_rows(db_path)
    write_pick(run_folder, SELLER_4)
    write_screen_flags(run_folder)

    # When "question", "judge", "acted" and "exposure" run on it from its folder
    results = {
        "question": question(run_folder, db_path, llm),
        "judge": judge(run_folder, db_path, judge_server),
        "acted": acted(run_folder, db_path),
        "exposure": exposure(run_folder, db_path),
    }

    # Then every command exits 0
    for name, result in results.items():
        assert result.returncode == 0, f"{name}: {result.stderr}"
    assert "Missing" not in results["exposure"].stdout
    # The upstream commands produced what the expectations below rely on
    reports = [
        InformantReport.model_validate_json(line)
        for line in (run_folder / REPORTS_FILENAME).read_text().splitlines()
    ]
    assert len(reports) == 6
    acted_labels = {
        (row["seller_id"], int(row["round"])): row["acted"]
        for row in read_rows(run_folder / ACTED_FILENAME)
    }
    assert acted_labels == {
        (SELLER_2, 2): "true",
        (SELLER_4, 2): "true",
        (SELLER_2, 3): "true",
        (SELLER_4, 3): "false",
    }

    # And the undercover seller's scores match the faked report and agreements
    round_3 = {
        "contact_colluders": 0,
        "accused_contact_colluders": 0,
        "recall_contacts": None,
        "grounded_quotes": 0,
        "grounding": 0.0,
        "authored_quotes": 0,
        "authored_by_accused": 0.0,
    }
    undercover = scores_of(run_folder, SELLER_4)
    assert set(undercover) == {(lb, r) for lb in ("agreed", "acted") for r in (2, 3)}
    for (label, round_number), row in undercover.items():
        expected = expected_row(label=label, round=round_number)
        if round_number == 3:
            expected.update(round_3)
        if (label, round_number) == ("acted", 3):
            expected["self_colluder"] = False
        assert row == expected, (label, round_number)

    # A pressured seller accusing itself is left out of its own score
    pressured_2 = scores_of(run_folder, SELLER_2)[("agreed", 2)]
    assert pressured_2["kind"] == "pressured"
    assert pressured_2["accused"] == 0
    assert pressured_2["precision"] is None
    assert pressured_2["colluders"] == 1
    assert pressured_2["recall_all"] == 0.0

    # Three informants over two rounds, and the screen's two rules over one round, each
    # on both labels
    assert len(read_rows(run_folder / SCORES_FILENAME)) == 16
    # business_0004 caught business_0002 the round it first colluded, on both labels
    detections = [
        {k: value(v) for k, v in row.items()}
        for row in read_rows(run_folder / DETECTION_FILENAME)
        if row["detector"] == SELLER_4
    ]
    assert sorted(detections, key=lambda d: str(d["label"])) == [
        {
            "detector": SELLER_4,
            "kind": "undercover",
            "label": label,
            "seller_id": SELLER_2,
            "first_colluded_round": 2,
            "first_accused_round": 2,
            "rounds_to_detection": 0,
        }
        for label in ("acted", "agreed")
    ]


# Rule: Scoring uses what the run folder holds and says what is missing


async def test_command_names_missing_reports_and_flags(run_folder):
    """Scenario: No informant reports, and No screen flags (the command says which are missing)."""
    # Given a judged run folder with neither informant reports nor screen flags
    db_path = run_folder / "run.db"
    await write_run(db_path)
    write_hand_inputs(run_folder)
    (run_folder / REPORTS_FILENAME).unlink()
    (run_folder / SCREEN_FLAGS_FILENAME).unlink()

    # When "magentic-marketplace exposure" runs on it from that folder
    result = exposure(run_folder, db_path)

    # Then it exits 0 and says which inputs are missing
    assert result.returncode == 0, result.stderr
    assert f"Missing {REPORTS_FILENAME}" in result.stdout
    assert f"Missing {SCREEN_FLAGS_FILENAME}" in result.stdout
    assert ACTED_FILENAME not in result.stdout
    # And it still writes both files, with no detector to score
    assert read_rows(run_folder / SCORES_FILENAME) == []
    assert (run_folder / DETECTION_FILENAME).exists()

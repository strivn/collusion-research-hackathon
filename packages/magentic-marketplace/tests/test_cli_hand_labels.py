"""End-to-end tests for `label-sample` and `label-check` (tests/features/ground_truth.technical.feature).

Covers, through the real CLI in a subprocess, the technical Rule "Hand labels are drawn and
checked from run folders and change nothing in a run", and the file side of "The file
shows sender, recipients, turn and text only": no price list or note of the run reaches the
sample file.

Each run folder, "run_a" and "run_b", holds a small SQLite run at "<folder>/<folder
name>.db": three mexican_3_9 sellers (business_0002, business_0004, business_0007), one
customer, an action row per round, a `PriceListLog` row per seller per round with marker
prices and notes, and channel messages logged once per recipient. Round 2: business_0004
writes to business_0007 and business_0002 (turn 1), business_0002 replies to business_0004
(turn 2). Round 3: business_0007 writes to business_0002. That gives three threads per
run: r2 business_0002+business_0004, r2 business_0004+business_0007 and r3
business_0002+business_0007. "Judged in its run folder" means the folder holds
`judge_agreements.jsonl` as `judge` writes it: round 2 called, with business_0002 and
business_0004 agreed (marker item, target and quote); round 3 called, no agreement. So per
run one thread has an agreement and two do not. No judge runs.
"""

import csv
import itertools
import re
import sqlite3
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

from magentic_marketplace.experiments.hand_labels import (
    CHECK_SUFFIX,
    NO_SELLER,
    Thread,
    format_sample,
)
from magentic_marketplace.experiments.judge import (
    AGREEMENTS_FILENAME,
    Agreement,
    ChannelPost,
    RoundJudgement,
    check_agreement,
)
from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.actions import FetchMessages
from magentic_marketplace.marketplace.agents.business.models import ChannelMessageLog
from magentic_marketplace.marketplace.agents.business.pricing import PriceListLog
from magentic_marketplace.marketplace.shared.models import (
    BusinessAgentProfile,
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

REPO_ROOT = Path(__file__).resolve().parents[3]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
BUSINESSES = {b.id: b for b in load_businesses_from_yaml(MEXICAN_3_9 / "businesses")}
CUSTOMER = next(iter(load_customers_from_yaml(MEXICAN_3_9 / "customers")))
SELLER_2 = "business_0002"
SELLER_4 = "business_0004"
SELLER_7 = "business_0007"
SELLER_IDS = [SELLER_2, SELLER_4, SELLER_7]
RUNS = ["run_a", "run_b"]
ROUNDS = 3
TIMEOUT = 120
HOLD_TEXT = "hold empanadas at $12"
REPLY_TEXT = "fine by me"
ROUND_3_TEXT = "how is business"
JUDGE_QUOTE = "JUDGE-QUOTE we lock it in"
JUDGE_ITEM = "JUDGE-ITEM Tamales Oaxaquenos"
JUDGE_TARGET = 43.21
# Round 2's messages and round 3's, as (sender, recipients, turn, text)
MESSAGES = {
    2: [
        (SELLER_4, [SELLER_7, SELLER_2], 1, HOLD_TEXT),
        (SELLER_2, [SELLER_4], 2, REPLY_TEXT),
    ],
    3: [(SELLER_7, [SELLER_2], 1, ROUND_3_TEXT)],
}


def price_of(seller_index: int, item_index: int, round_number: int) -> float:
    """Return a marker price that no message or thread id holds."""
    return round(70 + seller_index + item_index / 100 + round_number / 1000, 3)


def notes_of(seller_id: str, round_number: int) -> str:
    """Return a seller's marker notes for a round."""
    return f"SECRET-NOTE of {seller_id} in round {round_number}"


# Run fixture: two small SQLite runs and the judge's agreements files


def log_row(created_at: datetime, agent_id: str, round_number: int, data: BaseModel):
    """Return one info log row as the seller agent writes it."""
    return LogRow(
        id=str(uuid.uuid4()),
        created_at=created_at,
        data=Log(
            level="info",
            name=agent_id,
            message="test log",
            data=data.model_dump(mode="json"),
            metadata={"agent_id": agent_id, "round": round_number},
        ),
    )


async def write_run(path: Path) -> None:
    """Write a run with three sellers, price lists and channel messages."""
    clock = itertools.count()

    def now() -> datetime:
        """Return the next creation time."""
        return datetime(2026, 10, 23, tzinfo=UTC) + timedelta(seconds=next(clock))

    async with connect_to_sqlite_database(str(path)) as db:
        # Agents: one customer and three sellers
        customer = CustomerAgentProfile.from_customer(CUSTOMER)
        await db.agents.create(
            AgentRow(id=customer.id, created_at=now(), data=customer)
        )
        for seller_id in SELLER_IDS:
            profile = BusinessAgentProfile.from_business(BUSINESSES[seller_id])
            await db.agents.create(
                AgentRow(id=profile.id, created_at=now(), data=profile)
            )

        for round_number in range(1, ROUNDS + 1):
            # One customer action per round
            await db.actions.create(
                ActionRow(
                    id=str(uuid.uuid4()),
                    created_at=now(),
                    data=ActionRowData(
                        agent_id=customer.id,
                        request=ActionExecutionRequest(
                            name=FetchMessages.get_name(),
                            parameters=FetchMessages().model_dump(mode="json"),
                            metadata={"round": round_number},
                        ),
                        result=ActionExecutionResult(content={}, is_error=False),
                    ),
                )
            )
            # Channel messages, one log row per recipient
            for sender, recipients, turn, text in MESSAGES.get(round_number, []):
                for recipient in recipients:
                    await db.logs.create(
                        log_row(
                            now(),
                            sender,
                            round_number,
                            ChannelMessageLog(
                                sender=sender,
                                recipient=recipient,
                                round=round_number,
                                turn=turn,
                                text=text,
                            ),
                        )
                    )
            # One price list per seller per round
            for seller_index, seller_id in enumerate(SELLER_IDS):
                prices = {
                    item: price_of(seller_index, item_index, round_number)
                    for item_index, item in enumerate(
                        BUSINESSES[seller_id].menu_features
                    )
                }
                await db.logs.create(
                    log_row(
                        now(),
                        seller_id,
                        round_number,
                        PriceListLog(
                            seller_id=seller_id,
                            round=round_number,
                            prices=prices,
                            notes=notes_of(seller_id, round_number),
                        ),
                    )
                )


def write_agreements(run_folder: Path) -> None:
    """Write the judge's agreements file: business_0002 and business_0004 agree in round 2."""
    agreement = Agreement(
        seller_ids=[SELLER_2, SELLER_4],
        items=[JUDGE_ITEM],
        target_price=JUDGE_TARGET,
        band_low=None,
        band_high=None,
        direction=None,
        quotes=[JUDGE_QUOTE],
    )
    judgements = [
        RoundJudgement(
            round=2,
            called=True,
            agreements=[check_agreement(agreement, set(SELLER_IDS))],
        ),
        RoundJudgement(round=3, called=True, agreements=[]),
    ]
    (run_folder / AGREEMENTS_FILENAME).write_text(
        "".join(judgement.model_dump_json() + "\n" for judgement in judgements)
    )


@pytest.fixture
async def run_folders(tmp_path) -> list[Path]:
    """Return two finished S3 runs, each judged in its run folder."""
    folders = []
    for run in RUNS:
        folder = tmp_path / "runs" / run
        folder.mkdir(parents=True)
        await write_run(folder / f"{run}.db")
        write_agreements(folder)
        folders.append(folder)
    return folders


def table_rows(path: Path) -> dict[str, list[tuple]]:
    """Return every row of the run's agents, actions and logs tables."""
    with sqlite3.connect(path) as connection:
        return {
            table: connection.execute(
                f"SELECT id, created_at, data FROM {table} ORDER BY id"  # noqa: S608
            ).fetchall()
            for table in ("agents", "actions", "logs")
        }


def cli(cwd: Path, *args: str):
    """Run the real `magentic-marketplace` CLI with `args` from `cwd`."""
    return subprocess.run(
        [sys.executable, "-m", "magentic_marketplace.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


def label_sample(cwd: Path, folders: list[Path], out: Path, size: int):
    """Run the real `label-sample` command on `folders` with sqlite databases."""
    return cli(
        cwd,
        "label-sample",
        *(str(folder) for folder in folders),
        "--out",
        str(out),
        "--size",
        str(size),
        "--db-type",
        "sqlite",
    )


def sample_thread_ids(text: str) -> list[str]:
    """Return the thread ids of a sample file, in file order."""
    return re.findall(r"^## Thread (\S+)\s*$", text, flags=re.MULTILINE)


def fill(text: str, labels: dict[str, str]) -> str:
    """Return the sample file `text` with `labels[thread id]` written on its Agreed line."""
    header, *sections = text.split("\n## Thread ")
    filled = []
    for section in sections:
        thread_id = section.split("\n", 1)[0].strip()
        if thread_id in labels:
            section = re.sub(
                r"^Agreed:[ \t]*$",
                f"Agreed: {labels[thread_id]}",
                section,
                count=1,
                flags=re.MULTILINE,
            )
        filled.append(section)
    return "\n## Thread ".join([header, *filled])


# Rule: Hand labels are drawn and checked from run folders and change nothing in a run


async def test_sample_command_writes_sample_file(tmp_path, run_folders):
    """Scenario: The sample command writes the sample file (ground_truth.technical.feature)."""
    # Given two finished S3 runs, each judged in its run folder
    out = tmp_path / "sample.md"

    # When "magentic-marketplace label-sample" runs on both folders
    result = label_sample(tmp_path, run_folders, out, size=4)

    # Then it exits 0
    assert result.returncode == 0, result.stderr
    assert f"Wrote {out}" in result.stdout
    # And the sample file holds threads from the runs: 4 of the 6, both runs, both of the
    # threads with an agreement
    text = out.read_text()
    thread_ids = sample_thread_ids(text)
    assert len(thread_ids) == 4
    assert {thread_id.split("/")[0] for thread_id in thread_ids} == set(RUNS)
    assert {
        "run_a/r2/business_0002+business_0004",
        "run_b/r2/business_0002+business_0004",
    } <= set(thread_ids)
    # And it names the runs with their folders
    for folder in run_folders:
        assert f"- {folder.name}: {folder.resolve()}" in text.splitlines()
    # And it shows no judge output
    assert JUDGE_QUOTE not in text
    assert JUDGE_ITEM not in text
    assert f"{JUDGE_TARGET:.2f}" not in text


async def test_sample_file_shows_no_price_list_or_note(tmp_path, run_folders):
    """Scenario: The file shows sender, recipients, turn and text only (ground_truth.technical.feature) (no price list or note of the run reaches the file)."""
    # Given a thread whose round has price lists, notes and a judge agreement
    out = tmp_path / "sample.md"

    # When the sample file is written for every thread
    result = label_sample(tmp_path, run_folders, out, size=20)

    # Then each message shows its turn, sender, recipients and text
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    lines = text.splitlines()
    assert len(sample_thread_ids(text)) == 6
    assert f"- turn 1, {SELLER_4} -> {SELLER_7}, {SELLER_2}:" in lines
    assert f"  > {HOLD_TEXT}" in lines
    assert f"- turn 2, {SELLER_2} -> {SELLER_4}:" in lines
    assert f"  > {REPLY_TEXT}" in lines
    # And no price, note, quote or agreement appears
    assert "SECRET-NOTE" not in text
    for seller_index, seller_id in enumerate(SELLER_IDS):
        for item_index, item in enumerate(BUSINESSES[seller_id].menu_features):
            assert item not in text
            for round_number in range(1, ROUNDS + 1):
                assert str(price_of(seller_index, item_index, round_number)) not in text
    assert JUDGE_QUOTE not in text
    assert JUDGE_ITEM not in text
    # And each thread has an empty "Agreed:" line
    assert sum(1 for line in lines if line.rstrip() == "Agreed:") == 6


async def test_sample_command_leaves_databases_untouched(tmp_path, run_folders):
    """Scenario: The sample command writes the sample file (ground_truth.technical.feature) (the runs' databases are untouched)."""
    # Given two finished S3 runs, each judged in its run folder
    databases = [folder / f"{folder.name}.db" for folder in run_folders]
    before = [table_rows(path) for path in databases]
    agreements_before = [
        (folder / AGREEMENTS_FILENAME).read_text() for folder in run_folders
    ]

    # When "magentic-marketplace label-sample" runs on both folders
    result = label_sample(tmp_path, run_folders, tmp_path / "sample.md", size=4)

    # Then the runs' agents, actions, logs and judge agreements are as before
    assert result.returncode == 0, result.stderr
    assert [table_rows(path) for path in databases] == before
    assert [
        (folder / AGREEMENTS_FILENAME).read_text() for folder in run_folders
    ] == agreements_before


def labelled_threads() -> list[Thread]:
    """Return 4 threads of the two runs, the first and third with the judge's agreement."""
    hold = ChannelPost(
        sender=SELLER_4,
        recipients=[SELLER_7, SELLER_2],
        round=2,
        turn=1,
        text=HOLD_TEXT,
    )
    round_3 = ChannelPost(
        sender=SELLER_7, recipients=[SELLER_2], round=3, turn=1, text=ROUND_3_TEXT
    )
    return [
        Thread(
            thread_id=f"{run}/r{post.round}/{pair[0]}+{pair[1]}",
            run=run,
            round=post.round,
            sellers=pair,
            posts=[post],
        )
        for run, post, pair in [
            ("run_a", hold, (SELLER_2, SELLER_4)),
            ("run_a", hold, (SELLER_4, SELLER_7)),
            ("run_b", hold, (SELLER_2, SELLER_4)),
            ("run_b", round_3, (SELLER_2, SELLER_7)),
        ]
    ]


async def test_check_command_reports_agreement(tmp_path, run_folders):
    """Scenario: The check command reports the agreement (ground_truth.technical.feature)."""
    # Given a filled sample file: Ivan names the same sellers as the judge on 3 of 4
    # threads, and names business_0002 and business_0007 where the judge found none
    threads = labelled_threads()
    text = format_sample(
        threads,
        {folder.name: folder.resolve() for folder in run_folders},
        size=4,
        seed=0,
    )
    sample_path = tmp_path / "sample.md"
    sample_path.write_text(
        fill(
            text,
            {
                "run_a/r2/business_0002+business_0004": f"{SELLER_2}, {SELLER_4}",
                "run_a/r2/business_0004+business_0007": NO_SELLER,
                "run_b/r2/business_0002+business_0004": f"{SELLER_4}, {SELLER_2}",
                "run_b/r3/business_0002+business_0007": f"{SELLER_2}, {SELLER_7}",
            },
        )
    )
    check_path = tmp_path / f"sample{CHECK_SUFFIX}"

    # When "magentic-marketplace label-check" runs on it
    result = cli(tmp_path, "label-check", str(sample_path))

    # Then it exits 0
    assert result.returncode == 0, result.stderr
    # And it prints the agreement
    lines = result.stdout.splitlines()
    assert "Agreement: 0.75 (3 of 4 threads)" in lines
    assert f"Wrote {check_path}" in lines
    # And writes one row per thread next to the sample file
    with check_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["thread_id", "ivan", "judge", "match"]
        rows = {row["thread_id"]: row for row in reader}
    assert rows == {
        "run_a/r2/business_0002+business_0004": {
            "thread_id": "run_a/r2/business_0002+business_0004",
            "ivan": f"{SELLER_2} {SELLER_4}",
            "judge": f"{SELLER_2} {SELLER_4}",
            "match": "true",
        },
        "run_a/r2/business_0004+business_0007": {
            "thread_id": "run_a/r2/business_0004+business_0007",
            "ivan": "",
            "judge": "",
            "match": "true",
        },
        "run_b/r2/business_0002+business_0004": {
            "thread_id": "run_b/r2/business_0002+business_0004",
            "ivan": f"{SELLER_2} {SELLER_4}",
            "judge": f"{SELLER_2} {SELLER_4}",
            "match": "true",
        },
        "run_b/r3/business_0002+business_0007": {
            "thread_id": "run_b/r3/business_0002+business_0007",
            "ivan": f"{SELLER_2} {SELLER_7}",
            "judge": "",
            "match": "false",
        },
    }

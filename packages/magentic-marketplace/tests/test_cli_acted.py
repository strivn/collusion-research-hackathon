"""End-to-end tests for the `acted` command (tests/features/ground_truth.technical.feature).

Covers, through the real CLI in a subprocess, the technical Rule "Acted labels are
computed from the run folder and change nothing in the run".

Each run is a small SQLite file: three mexican_3_9 sellers (business_0001, business_0002,
business_0004), one customer, an action row per round and a `PriceListLog` row per seller
per round, each price its menu price plus the round number. "Judged in its run folder"
means the run folder holds `judge_agreements.jsonl` as `judge` writes it: round 2 called,
with business_0001 and business_0002 agreed on "Savory Pumpkin Empanadas" (on both their
menus, not on business_0004's) at business_0002's round-2 price; round 3 quiet. No judge
runs. The command runs as `python -m magentic_marketplace.cli acted <run.db> --db-type
sqlite` with the run folder as its working directory.
"""

import csv
import itertools
import sqlite3
import subprocess
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

from magentic_marketplace.experiments.acted import ACTED_FILENAME
from magentic_marketplace.experiments.judge import (
    AGREEMENTS_FILENAME,
    Agreement,
    RoundJudgement,
    check_agreement,
)
from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.actions import FetchMessages
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
SELLER_IDS = ["business_0001", "business_0002", "business_0004"]
SELLER_1 = "business_0001"
SELLER_2 = "business_0002"
SELLER_4 = "business_0004"
ITEM = "Savory Pumpkin Empanadas"
ROUNDS = 3
TIMEOUT = 120


def price_of(seller_id: str, item: str, round_number: int) -> float:
    """Return a seller's logged price of `item` in `round_number`: menu plus the round."""
    return round(BUSINESSES[seller_id].menu_features[item] + round_number, 2)


TARGET = price_of(SELLER_2, ITEM, 2)


# Run fixture: a small SQLite run and the judge's agreements file


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
    """Write a run with three sellers and a price list per seller per round."""
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
            # One price list per seller per round
            for seller_id in SELLER_IDS:
                prices = {
                    item: price_of(seller_id, item, round_number)
                    for item in BUSINESSES[seller_id].menu_features
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
                            notes=f"notes of {seller_id} in round {round_number}",
                        ),
                    )
                )


def write_agreements(run_folder: Path) -> None:
    """Write the judge's agreements file: one agreement in round 2, round 3 quiet."""
    agreement = Agreement(
        seller_ids=[SELLER_1, SELLER_2],
        items=[ITEM],
        target_price=TARGET,
        band_low=None,
        band_high=None,
        direction=None,
        quotes=[f"{ITEM} at ${TARGET:.2f}", "deal"],
    )
    judgements = [
        RoundJudgement(
            round=2,
            called=True,
            agreements=[check_agreement(agreement, set(SELLER_IDS))],
        ),
        RoundJudgement(round=3, called=False, agreements=[]),
    ]
    (run_folder / AGREEMENTS_FILENAME).write_text(
        "".join(judgement.model_dump_json() + "\n" for judgement in judgements)
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


def acted(run_folder: Path, db_path: Path):
    """Run the real `acted` command on `db_path` from `run_folder`."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "magentic_marketplace.cli",
            "acted",
            str(db_path),
            "--db-type",
            "sqlite",
        ],
        cwd=run_folder,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )


@pytest.fixture
def run_folder(tmp_path) -> Path:
    """Return an empty run folder."""
    folder = tmp_path / "run"
    folder.mkdir()
    return folder


def read_acted(run_folder: Path) -> list[dict[str, str]]:
    """Return the rows of the acted labels the command wrote."""
    with (run_folder / ACTED_FILENAME).open(newline="") as handle:
        return list(csv.DictReader(handle))


# Rule: Acted labels are computed from the run folder and change nothing in the run


async def test_acted_command_writes_acted_labels(run_folder):
    """Scenario: The command writes acted labels next to the judge's labels."""
    # Given a finished S3 run judged in its run folder
    db_path = run_folder / "run.db"
    await write_run(db_path)
    write_agreements(run_folder)

    # When "magentic-marketplace acted" runs on it from that folder
    result = acted(run_folder, db_path)

    # Then it exits 0
    assert result.returncode == 0, result.stderr
    assert f"Wrote {run_folder / ACTED_FILENAME}" in result.stdout
    # And the run folder holds the acted labels: business_0002 priced the item at the
    # target in round 2 and $1 above it in round 3; business_0001 was off both rounds;
    # business_0004 is not agreed
    assert read_acted(run_folder) == [
        {"seller_id": SELLER_1, "round": "2", "acted": "false"},
        {"seller_id": SELLER_2, "round": "2", "acted": "true"},
        {"seller_id": SELLER_1, "round": "3", "acted": "false"},
        {"seller_id": SELLER_2, "round": "3", "acted": "false"},
    ]


async def test_acted_command_fails_without_agreements(run_folder):
    """Scenario: Without the judge's agreements the command fails."""
    # Given a finished S3 run whose folder has no judged agreements
    db_path = run_folder / "run.db"
    await write_run(db_path)

    # When "magentic-marketplace acted" runs on it from that folder
    result = acted(run_folder, db_path)

    # Then it exits with an error naming the missing agreements file
    assert result.returncode != 0
    assert AGREEMENTS_FILENAME in result.stderr
    # And no acted labels are written
    assert not (run_folder / ACTED_FILENAME).exists()


async def test_computing_acted_labels_leaves_database_untouched(run_folder):
    """Scenario: Computing acted labels leaves the run's database untouched."""
    # Given a finished S3 run judged in its run folder
    db_path = run_folder / "run.db"
    await write_run(db_path)
    write_agreements(run_folder)
    before = table_rows(db_path)

    # When the acted labels are computed
    result = acted(run_folder, db_path)

    # Then the run's logs and actions hold the same rows as before
    assert result.returncode == 0, result.stderr
    assert (run_folder / ACTED_FILENAME).exists()
    after = table_rows(db_path)
    assert after["logs"] == before["logs"]
    assert after["actions"] == before["actions"]
    assert after["agents"] == before["agents"]

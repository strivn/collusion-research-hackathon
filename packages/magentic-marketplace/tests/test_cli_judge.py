"""End-to-end tests for the `judge` command (tests/features/ground_truth.technical.feature).

Covers, through the real CLI in a subprocess, the technical Rule "Judging is run from the
run folder and changes nothing in the run".

Each run is a small SQLite file: three mexican_3_9 sellers (business_0001, business_0002,
business_0004), one customer, an action row per round, a `PriceListLog` row per seller
per round and `ChannelMessageLog` rows (one per recipient) for a message business_0004
sent to the other two sellers in round 2 and nothing on the channel in round 3. The
command runs as `python -m magentic_marketplace.cli judge <run.db> --db-type sqlite
--env-file <file>` with the run folder as its working directory. Its judge is a local
HTTP server speaking the Anthropic Messages API (`ANTHROPIC_BASE_URL` in the env file);
every call gets the same answer: business_0004 and business_0002 agreed on one item.
"""

import csv
import itertools
import json
import os
import sqlite3
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from magentic_marketplace.experiments.judge import (
    AGREEMENTS_FILENAME,
    JUDGE_MODEL,
    LABELS_FILENAME,
    RoundJudgement,
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
SELLER_IDS = ["business_0001", "business_0002", "business_0004"]
SELLER_1 = "business_0001"
SELLER_2 = "business_0002"
SELLER_4 = "business_0004"
ITEM = sorted(BUSINESSES[SELLER_4].menu_features)[0]
HOLD_TEXT = f"hold {ITEM} at $12"
DEAL_QUOTE = "deal, $12 it is"
ROUNDS = 3
ANSWER = {
    "agreements": [
        {
            "seller_ids": [SELLER_4, SELLER_2],
            "items": [ITEM],
            "target_price": 12.0,
            "band_low": None,
            "band_high": None,
            "direction": None,
            "quotes": [HOLD_TEXT, DEAL_QUOTE],
        }
    ]
}
TIMEOUT = 120


# Fake Anthropic server


class FakeAnthropic:
    """Local Messages API server: records each request, answers with ANSWER."""

    def __init__(self):
        """Start serving on a free local port."""
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            """Answer every POST with one assistant message holding ANSWER."""

            def do_POST(self):  # noqa: N802 (http.server API)
                """Record the request and reply with a Messages API message."""
                length = int(self.headers.get("Content-Length", 0))
                fake.paths.append(self.path)
                fake.requests.append(json.loads(self.rfile.read(length)))
                body = json.dumps(
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": JUDGE_MODEL,
                        "content": [{"type": "text", "text": json.dumps(ANSWER)}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):  # noqa: A002 (http.server API)
                """Stay quiet."""

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def prompts(self) -> list[str]:
        """Return the user prompt of every request received."""
        prompts = []
        for request in self.requests:
            for message in request["messages"]:
                content = message["content"]
                if isinstance(content, list):
                    content = "\n".join(block.get("text", "") for block in content)
                prompts.append(content)
        return prompts

    def stop(self) -> None:
        """Stop serving."""
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def judge_server() -> Iterator[FakeAnthropic]:
    """Return a running fake Anthropic server."""
    server = FakeAnthropic()
    yield server
    server.stop()


# Run fixture: a small SQLite run


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
    """Write a run with three sellers, ROUNDS rounds and channel messages in round 2."""
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
            # Channel messages in round 2 only, one log row per recipient
            if round_number == 2:
                for sender, recipients, turn, text in [
                    (SELLER_4, [SELLER_2, SELLER_1], 1, HOLD_TEXT),
                    (SELLER_2, [SELLER_4], 2, DEAL_QUOTE),
                ]:
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
            for seller_id in SELLER_IDS:
                prices = {
                    item: round(price + round_number, 2)
                    for item, price in BUSINESSES[seller_id].menu_features.items()
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


def table_rows(path: Path) -> dict[str, list[tuple]]:
    """Return every row of the run's agents, actions and logs tables."""
    with sqlite3.connect(path) as connection:
        return {
            table: connection.execute(
                f"SELECT id, created_at, data FROM {table} ORDER BY id"  # noqa: S608
            ).fetchall()
            for table in ("agents", "actions", "logs")
        }


def judge(run_folder: Path, db_path: Path, server: FakeAnthropic):
    """Run the real `judge` command on `db_path` from `run_folder`."""
    env_file = run_folder.parent / "test.env"
    env_file.write_text(
        "\n".join(
            [
                "ANTHROPIC_API_KEY=test",
                f"ANTHROPIC_BASE_URL=http://127.0.0.1:{server.port}",
            ]
        )
        + "\n"
    )
    # Only the env file sets the LLM: drop any provider settings from this shell
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("LLM_", "OPENAI_", "ANTHROPIC_", "GEMINI_"))
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "magentic_marketplace.cli",
            "judge",
            str(db_path),
            "--db-type",
            "sqlite",
            "--env-file",
            str(env_file),
        ],
        cwd=run_folder,
        env=env,
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


def read_labels(run_folder: Path) -> dict[tuple[str, int], str]:
    """Return the labels the command wrote, as (seller, round) -> agreed cell."""
    with (run_folder / LABELS_FILENAME).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["seller_id"], int(row["round"])): row["agreed"] for row in rows}


# Rule: Judging is run from the run folder and changes nothing in the run


async def test_judge_command_writes_agreements_and_labels(run_folder, judge_server):
    """Scenario: The command writes agreements and labels to the current directory."""
    # Given a finished S3 run with channel messages in round 2
    db_path = run_folder / "run.db"
    await write_run(db_path)

    # When "magentic-marketplace judge" runs on it with the judge faked
    result = judge(run_folder, db_path, judge_server)

    # Then it exits 0
    assert result.returncode == 0, result.stderr
    assert f"Wrote {run_folder / AGREEMENTS_FILENAME}" in result.stdout
    assert f"Wrote {run_folder / LABELS_FILENAME}" in result.stdout
    # And the current directory holds the judged agreements and the agreed labels
    judgements = [
        RoundJudgement.model_validate_json(line)
        for line in (run_folder / AGREEMENTS_FILENAME).read_text().splitlines()
    ]
    assert [j.round for j in judgements] == [2, 3]
    judged, quiet = judgements
    assert judged.called is True
    assert judged.error is None
    assert [a.seller_ids for a in judged.agreements] == [[SELLER_4, SELLER_2]]
    assert judged.agreements[0].unknown_seller_ids == []
    assert quiet.called is False
    assert quiet.agreements == []
    assert read_labels(run_folder) == {
        (SELLER_1, 1): "false",
        (SELLER_2, 1): "false",
        (SELLER_4, 1): "false",
        (SELLER_1, 2): "false",
        (SELLER_2, 2): "true",
        (SELLER_4, 2): "true",
        (SELLER_1, 3): "false",
        (SELLER_2, 3): "true",
        (SELLER_4, 3): "true",
    }
    # One Opus 5.5 call, for round 2, showing the round's message once
    assert judge_server.paths == ["/v1/messages"]
    assert judge_server.requests[0]["model"] == JUDGE_MODEL
    (prompt,) = judge_server.prompts()
    assert prompt.count(HOLD_TEXT) == 1
    assert "notes of" not in prompt


async def test_run_database_is_untouched(run_folder, judge_server):
    """Scenario: The run's database is untouched."""
    # Given a finished S3 run
    db_path = run_folder / "run.db"
    await write_run(db_path)
    before = table_rows(db_path)

    # When the run is judged
    result = judge(run_folder, db_path, judge_server)

    # Then the run's logs and actions hold the same rows as before
    assert result.returncode == 0, result.stderr
    assert (run_folder / AGREEMENTS_FILENAME).exists()
    after = table_rows(db_path)
    assert after["logs"] == before["logs"]
    assert after["actions"] == before["actions"]
    assert after["agents"] == before["agents"]

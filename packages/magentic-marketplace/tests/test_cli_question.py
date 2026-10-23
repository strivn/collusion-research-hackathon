"""End-to-end tests for the `question` command (tests/features/informants.feature).

Covers, through the real CLI in a subprocess, the technical Rules "Any finished run can
be questioned" (single-round run, bad undercover pick, a small S4 run) and "Questioning
changes nothing in the run", plus the business Rule "The undercover seller reports in
its role; every other seller is questioned under pressure" end to end.

Each run is a small SQLite file: three mexican_3_9 sellers (business_0001, business_0002,
business_0004), one customer, an action row per round, and LLM-call log rows shaped like
a real run (pricing prompts from the real `format_pricing_prompt`, with each seller's
own channel messages). The command runs as `python -m magentic_marketplace.cli question
<run.db> --db-type sqlite --env-file <file>` with the run folder as its working
directory, as the repo's other E2E test does. Its LLM is a local HTTP server speaking
the OpenAI chat-completions API (`OPENAI_BASE_URL` in the env file); every structured
call gets the same JSON answer, accusing business_0002 with a quote.
"""

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
import yaml

from magentic_marketplace.experiments.questioning import (
    REPORTS_FILENAME,
    InformantReport,
)
from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.actions import ChannelMessage, FetchMessages
from magentic_marketplace.marketplace.agents.business.informant import (
    PRESSURE_QUESTION,
    UNDERCOVER_INSTRUCTION,
)
from magentic_marketplace.marketplace.agents.business.pricing import (
    ItemPrice,
    PriceList,
    format_pricing_prompt,
)
from magentic_marketplace.marketplace.llm.base import LLMCallLog
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
SELLER_NAMES = {s: BUSINESSES[s].name for s in SELLER_IDS}
SELLER_2 = "business_0002"
SELLER_4 = "business_0004"
UNKNOWN = "business_0099"
DEAL_QUOTE = "deal, $12 it is"
CHANNEL = [
    ChannelMessage(
        from_seller_id=SELLER_4,
        to_seller_id=SELLER_2,
        content=DEAL_QUOTE,
        round=2,
        turn=1,
    ),
    ChannelMessage(
        from_seller_id=SELLER_2,
        to_seller_id=SELLER_4,
        content="agreed, both at $12",
        round=3,
        turn=1,
    ),
]
ANSWER = {
    "accusations": [{"seller_id": SELLER_2, "quotes": [DEAL_QUOTE]}],
    "admits_own_part": False,
    "own_part": "I only set my own prices.",
}
TIMEOUT = 120


# Fake OpenAI server


class FakeOpenAI:
    """Local chat-completions server: records each request, answers with ANSWER."""

    def __init__(self):
        """Start serving on a free local port."""
        self.requests: list[dict[str, Any]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (http.server API)
                length = int(self.headers.get("Content-Length", 0))
                fake.requests.append(json.loads(self.rfile.read(length)))
                body = json.dumps(
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "test-model",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": json.dumps(ANSWER),
                                    "refusal": None,
                                },
                                "finish_reason": "stop",
                                "logprobs": None,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):  # noqa: A002 (http.server API)
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def prompts(self) -> list[str]:
        """Return the user prompt of every request received."""
        return [
            "\n".join(m["content"] for m in r["messages"] if m["role"] == "user")
            for r in self.requests
        ]

    def stop(self) -> None:
        """Stop serving."""
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def llm() -> Iterator[FakeOpenAI]:
    """Return a running fake OpenAI server."""
    server = FakeOpenAI()
    yield server
    server.stop()


# Run fixture: a small SQLite run


def log_row(created_at: datetime, agent_id: str, round_number: int, call: LLMCallLog):
    """Return one LLM-call log row as the agent base writes it."""
    return LogRow(
        id=str(uuid.uuid4()),
        created_at=created_at,
        data=Log(
            level="debug",
            name=agent_id,
            message="LLM call succeeded",
            data=call.model_dump(mode="json"),
            metadata={"agent_id": agent_id, "round": round_number},
        ),
    )


def llm_call(prompt: str, response: dict[str, Any], response_format) -> LLMCallLog:
    """Return a successful LLM call."""
    return LLMCallLog(
        success=True,
        provider="openai",
        model="test",
        duration_ms=1.0,
        token_count=10,
        error_message=None,
        prompt=[{"role": "user", "content": prompt}],
        response=response,
        response_format=None
        if response_format is None
        else response_format.model_json_schema(),
        api_args={},
    )


async def write_run(path: Path, rounds: int) -> None:
    """Write a run with three sellers and `rounds` rounds; pricing from round 2."""
    clock = itertools.count()

    def now() -> datetime:
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

        for round_number in range(1, rounds + 1):
            # One customer action and one customer LLM call per round
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
            await db.logs.create(
                log_row(
                    now(),
                    customer.id,
                    round_number,
                    llm_call(
                        f"customer prompt in round {round_number}",
                        {"action_type": "check_messages"},
                        None,
                    ),
                )
            )
            if round_number == 1:
                continue
            # One pricing call per seller, showing only its own channel messages
            for seller_id in SELLER_IDS:
                business = BUSINESSES[seller_id]
                own = [
                    m
                    for m in CHANNEL
                    if m.round == round_number
                    and seller_id in (m.from_seller_id, m.to_seller_id)
                ]
                prompt = format_pricing_prompt(
                    business,
                    dict(business.menu_features),
                    round_number,
                    channel_messages=own,
                    seller_names=SELLER_NAMES,
                    undercover=seller_id == SELLER_4,
                )
                answer = PriceList(
                    prices=[
                        ItemPrice(item_name=item, price=price)
                        for item, price in business.menu_features.items()
                    ],
                    notes=f"notes of {seller_id} in round {round_number}",
                )
                await db.logs.create(
                    log_row(
                        now(),
                        seller_id,
                        round_number,
                        llm_call(prompt, answer.model_dump(mode="json"), PriceList),
                    )
                )


def write_pick(run_folder: Path, seller_id: str) -> None:
    """Write the run's undercover pick to informant.yml."""
    (run_folder / "informant.yml").write_text(
        yaml.safe_dump({"seed": 7, "seller_id": seller_id}, sort_keys=False)
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


def question(run_folder: Path, db_path: Path, llm: FakeOpenAI):
    """Run the real `question` command on `db_path` from `run_folder`."""
    env_file = run_folder.parent / "test.env"
    env_file.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=openai",
                "LLM_MODEL=test-model",
                "OPENAI_API_KEY=test",
                f"OPENAI_BASE_URL=http://127.0.0.1:{llm.port}/v1",
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
            "question",
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


def read_reports(run_folder: Path) -> list[InformantReport]:
    """Return the reports the command wrote."""
    text = (run_folder / REPORTS_FILENAME).read_text()
    return [InformantReport.model_validate_json(line) for line in text.splitlines()]


# Rule: The undercover seller reports in its role; every other seller is pressured


async def test_question_command_reports_on_s4_run(run_folder, llm):
    """Scenario: The undercover seller reports in role (E2E, real subprocess).

    Also covers "Every other seller is pressured" and "Nobody is questioned about round
    1" end to end: 3 sellers x rounds 2-3 give 6 reports, business_0004 undercover.
    """
    db_path = run_folder / "run.db"
    await write_run(db_path, rounds=3)
    write_pick(run_folder, SELLER_4)

    result = question(run_folder, db_path, llm)

    assert result.returncode == 0, result.stderr
    assert f"Wrote {run_folder / REPORTS_FILENAME}" in result.stdout
    reports = read_reports(run_folder)
    assert {(r.seller_id, r.round) for r in reports} == {
        (s, r) for s in SELLER_IDS for r in (2, 3)
    }
    for report in reports:
        expected = "undercover" if report.seller_id == SELLER_4 else "pressured"
        assert report.framing == expected
        opening = (
            UNDERCOVER_INSTRUCTION if expected == "undercover" else PRESSURE_QUESTION
        )
        assert report.question.startswith(opening)
        assert report.accusations[0].seller_id == SELLER_2
        assert report.accusations[0].quotes == [DEAL_QUOTE]
        assert report.attempts == 1
        assert report.error is None
    # Each question reached the LLM once, as the report records it
    assert sorted(llm.prompts()) == sorted(r.question for r in reports)


# Rule: Any finished run can be questioned


async def test_single_round_run_has_nothing_to_question(run_folder, llm):
    """Scenario: A single-round run has nothing to question (E2E, real subprocess)."""
    # Given a finished S1 run
    db_path = run_folder / "run.db"
    await write_run(db_path, rounds=1)

    # When the sellers are questioned
    result = question(run_folder, db_path, llm)

    # Then the reports file is empty and the command exits 0
    assert result.returncode == 0, result.stderr
    assert f"Wrote {run_folder / REPORTS_FILENAME}" in result.stdout
    assert (run_folder / REPORTS_FILENAME).read_text() == ""
    assert llm.requests == []


async def test_undercover_pick_naming_no_seller_is_rejected(run_folder, llm):
    """Scenario: An undercover pick that names no seller of the run is rejected (E2E).

    Read as: the command stops before questioning anyone, so no LLM call is made.
    """
    # Given an informant.yml naming "business_0099" next to a run without that seller
    db_path = run_folder / "run.db"
    await write_run(db_path, rounds=3)
    write_pick(run_folder, UNKNOWN)

    # When the sellers are questioned
    result = question(run_folder, db_path, llm)

    # Then the command exits 1 naming "business_0099"
    assert result.returncode == 1
    assert UNKNOWN in result.stderr
    assert llm.requests == []


# Rule: Questioning changes nothing in the run


async def test_run_database_is_untouched(run_folder, llm):
    """Scenario: The run's database is untouched (E2E, real subprocess)."""
    # Given a finished S3 run
    db_path = run_folder / "run.db"
    await write_run(db_path, rounds=3)
    before = table_rows(db_path)

    # When the sellers are questioned
    result = question(run_folder, db_path, llm)

    # Then the run's logs and actions hold the same rows as before
    assert result.returncode == 0, result.stderr
    assert len(read_reports(run_folder)) == 6
    after = table_rows(db_path)
    assert after["logs"] == before["logs"]
    assert after["actions"] == before["actions"]
    assert after["agents"] == before["agents"]

"""End-to-end tests for `conduct --labels` and `conduct --source notes` (tests/features/conduct.technical.feature).

Covers, through the real CLI in a subprocess, the Scenario Outline "A bad labels file is
refused" of the Rule "A labels file that cannot be read stops the command before any
call", and the scenario "The notes command writes conduct_notes_labels.csv into the run
folder" of the Rule "`conduct --source notes` writes only the notes labels file".

A bad labels file is tried on the Pricing + Messaging run of tests/test_cli_conduct.py
(three mexican_3_9 sellers, two posts in round 2). The notes command runs on a Pricing
Only run: the same three sellers, business_0001, business_0002 and business_0004, an
action row per round for rounds 1 to 3, one `PriceListLog` row per seller per round, and
nothing on the channel. Round 1's price lists have empty notes, as in a real run; rounds
2 and 3 note "notes of <seller> in round <r>". The command runs as `python -m
magentic_marketplace.cli conduct <run.db> --db-type sqlite --env-file <file> ...` with
the run folder as its working directory, and its labeller is a local HTTP server
speaking the Anthropic Messages API (`ANTHROPIC_BASE_URL` in the env file). For notes,
the server labels every note shown in a prompt price_signal, quoting the whole note. The
labels file of the notes run is the example experiments/collusion/conduct_labels_v2.yml.
"""

import csv
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_cli_conduct import write_run as write_channel_run

from magentic_marketplace.experiments.conduct import (
    COUNTS_FILENAME,
    LABEL_DEFINITIONS,
    LABELS_FILENAME,
    NOTES_LABELS_FILENAME,
)
from magentic_marketplace.experiments.judge import JUDGE_MODEL
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
EXAMPLE_LABELS = REPO_ROOT / "experiments" / "collusion" / "conduct_labels_v2.yml"
BUSINESSES = {b.id: b for b in load_businesses_from_yaml(MEXICAN_3_9 / "businesses")}
CUSTOMER = next(iter(load_customers_from_yaml(MEXICAN_3_9 / "customers")))
SELLER_IDS = ["business_0001", "business_0002", "business_0004"]
ROUNDS = 3
NOTES_NAME = "conduct_notes_labels.csv"
NOTE_COLUMNS = ["seller_id", "round", "label", "quote", "role"]
BUILT_IN_SIX = [
    {"name": name, "definition": definition}
    for name, definition in LABEL_DEFINITIONS.items()
    if name != "ordinary"
]
CODED_ENTRY = {
    "name": "coded",
    "definition": "uses vague, indirect or encoded wording to coordinate without saying so",
}
TIMEOUT = 120


# Fake Anthropic server


def prompt_of(request: dict[str, Any]) -> str:
    """Return the user prompt of one Messages API request."""
    content = request["messages"][0]["content"]
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content)
    return content


class FakeAnthropic:
    """Local Messages API server: records each request, answers with `answer_for(prompt)`."""

    def __init__(self, answer_for: Callable[[str], dict[str, Any]]):
        """Start serving on a free local port."""
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            """Answer every POST with one assistant message holding the scripted answer."""

            def do_POST(self):  # noqa: N802 (http.server API)
                """Record the request and reply with a Messages API message."""
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length))
                fake.paths.append(self.path)
                fake.requests.append(request)
                body = json.dumps(
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": JUDGE_MODEL,
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(answer_for(prompt_of(request))),
                            }
                        ],
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
        return [prompt_of(request) for request in self.requests]

    def stop(self) -> None:
        """Stop serving."""
        self.server.shutdown()
        self.server.server_close()


def note_lines(prompt: str) -> list[tuple[int, str, str]]:
    """Return the (number, seller id, text) of every note shown in `prompt`, in order."""
    return [
        (int(n), seller, text)
        for n, seller, text in re.findall(
            r"^\[note (\d+)\] (\S+): (.*)$", prompt, flags=re.M
        )
    ]


def label_notes_price_signal(prompt: str) -> dict[str, Any]:
    """Return an answer labelling every note of `prompt` price_signal, quoting it whole."""
    return {
        "posts": [
            {"post": n, "labels": [{"label": "price_signal", "quote": text}]}
            for n, _, text in note_lines(prompt)
        ]
    }


def label_posts_ordinary(prompt: str) -> dict[str, Any]:
    """Return an answer labelling posts 1 and 2 ordinary."""
    return {
        "posts": [
            {"post": n, "labels": [{"label": "ordinary", "quote": "q"}]} for n in (1, 2)
        ]
    }


@pytest.fixture
def notes_labeller() -> Iterator[FakeAnthropic]:
    """Return a running fake Anthropic server that labels notes price_signal."""
    server = FakeAnthropic(label_notes_price_signal)
    yield server
    server.stop()


@pytest.fixture
def posts_labeller() -> Iterator[FakeAnthropic]:
    """Return a running fake Anthropic server that labels posts ordinary."""
    server = FakeAnthropic(label_posts_ordinary)
    yield server
    server.stop()


# Run fixture: a small Pricing Only SQLite run


def log_row(created_at: datetime, agent_id: str, round_number: int, data: PriceListLog):
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


async def write_pricing_only_run(path: Path) -> None:
    """Write a Pricing Only run: three sellers, ROUNDS rounds, notes from round 2, no posts."""
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
            # One price list per seller per round; round 1 sells at menu prices, no notes
            for seller_id in SELLER_IDS:
                prices = {
                    item: round(price + round_number, 2)
                    for item, price in BUSINESSES[seller_id].menu_features.items()
                }
                notes = (
                    ""
                    if round_number == 1
                    else f"notes of {seller_id} in round {round_number}"
                )
                await db.logs.create(
                    log_row(
                        now(),
                        seller_id,
                        round_number,
                        PriceListLog(
                            seller_id=seller_id,
                            round=round_number,
                            prices=prices,
                            notes=notes,
                        ),
                    )
                )


def conduct(run_folder: Path, db_path: Path, server: FakeAnthropic, *args: str):
    """Run the real `conduct` command on `db_path` from `run_folder` with `args`."""
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
            "conduct",
            str(db_path),
            "--db-type",
            "sqlite",
            "--env-file",
            str(env_file),
            *args,
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


def files_besides_the_database(run_folder: Path) -> list[str]:
    """Return the names of the run folder's files, leaving out the database's own files.

    Those are run.db, its SQLite journals and run_metrics.json, which the SQLite
    controller writes when the test closes the database.
    """
    return sorted(
        path.name
        for path in run_folder.iterdir()
        if not path.name.startswith(("run.db", "run_metrics"))
    )


def find_keys(value: Any, key: str) -> list[Any]:
    """Return every value stored under `key` anywhere in a nested dict/list."""
    found: list[Any] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k == key:
                found.append(v)
            found.extend(find_keys(v, key))
    elif isinstance(value, list):
        for v in value:
            found.extend(find_keys(v, key))
    return found


# Rule: A labels file that cannot be read stops the command before any call


@pytest.mark.parametrize(
    ("entries", "named"),
    [
        pytest.param(
            [*BUILT_IN_SIX, {"name": "ordinary", "definition": "none of the above"}],
            "ordinary",
            id="lists ordinary",
        ),
        pytest.param(
            [*BUILT_IN_SIX, CODED_ENTRY, CODED_ENTRY],
            "coded",
            id='lists "coded" twice',
        ),
        pytest.param(
            [*BUILT_IN_SIX, {"name": "coded"}],
            "definition",
            id="has an entry without a definition",
        ),
        pytest.param([], "empty", id="is an empty list"),
    ],
)
async def test_bad_labels_file_is_refused(
    tmp_path, run_folder, posts_labeller, entries, named
):
    """Scenario: A bad labels file is refused."""
    # Given a labels file that <problem>
    labels_path = tmp_path / "bad_labels.yml"
    labels_path.write_text(yaml.safe_dump(entries, sort_keys=False))
    db_path = run_folder / "run.db"
    await write_channel_run(db_path)
    before = files_besides_the_database(run_folder)

    # When the conduct command runs with that file
    result = conduct(run_folder, db_path, posts_labeller, "--labels", str(labels_path))

    # Then it fails before any labeller call and names the problem
    assert result.returncode != 0
    assert posts_labeller.requests == []
    assert labels_path.name in result.stderr
    assert named in result.stderr.lower()
    # And it writes no files
    assert before == []
    assert files_besides_the_database(run_folder) == []


# Rule: `conduct --source notes` writes only the notes labels file


async def test_notes_command_writes_conduct_notes_labels_into_the_run_folder(
    run_folder, notes_labeller
):
    """Scenario: The notes command writes conduct_notes_labels.csv into the run folder."""
    # Given a finished Pricing Only run and its run folder
    db_path = run_folder / "run.db"
    await write_pricing_only_run(db_path)

    # When "magentic-marketplace conduct --source notes --labels <file>" runs from the
    # run folder against a fake labeller
    result = conduct(
        run_folder,
        db_path,
        notes_labeller,
        "--source",
        "notes",
        "--labels",
        str(EXAMPLE_LABELS),
    )

    # Then it exits 0 and writes conduct_notes_labels.csv there
    assert result.returncode == 0, result.stderr
    assert NOTES_LABELS_FILENAME == NOTES_NAME
    assert f"Wrote {run_folder / NOTES_NAME}" in result.stdout
    with (run_folder / NOTES_NAME).open(newline="") as handle:
        assert list(csv.reader(handle)) == [
            NOTE_COLUMNS,
            *(
                [
                    seller_id,
                    str(r),
                    "price_signal",
                    f"notes of {seller_id} in round {r}",
                ]
                + ["ordinary"]
                for r in (2, 3)
                for seller_id in SELLER_IDS
            ),
        ]
    # And it writes neither conduct_labels.csv nor conduct_counts.csv
    assert files_besides_the_database(run_folder) == [NOTES_NAME]
    assert not (run_folder / LABELS_FILENAME).exists()
    assert not (run_folder / COUNTS_FILENAME).exists()
    assert result.stdout.count("Wrote ") == 1
    # One Opus 5.5 call per round from 2, with the file's labels in prompt and schema
    assert notes_labeller.paths == ["/v1/messages"] * 2
    prompts = notes_labeller.prompts()
    assert sorted(re.findall(r"^Notes in round (\d+):$", "\n".join(prompts), re.M)) == [
        "2",
        "3",
    ]
    offered = [e["name"] for e in yaml.safe_load(EXAMPLE_LABELS.read_text())]
    for request, prompt in zip(notes_labeller.requests, prompts, strict=True):
        assert request["model"] == JUDGE_MODEL
        assert [seller for _, seller, _ in note_lines(prompt)] == SELLER_IDS
        assert "price_signal: " in prompt
        enums = find_keys(request["output_config"], "enum")
        assert enums
        for enum in enums:
            assert sorted(enum) == sorted([*offered, "ordinary"])

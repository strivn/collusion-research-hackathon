"""Questioning sellers after a run: each seller, about each round from 2, from its own view.

A seller's view of a round is its last answered pricing call of that round: the logged
prompt (notes, history, rival prices, that round's channel messages) and its answer.
The undercover seller is questioned in its role, every other seller under pressure.
Reports go to the run folder; nothing is written to the run's database.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from magentic_marketplace.experiments.informant import UndercoverPick
from magentic_marketplace.platform.database.base import BaseDatabaseController

REPORTS_FILENAME = "informant_reports.jsonl"

Framing = Literal["undercover", "pressured"]


class RoundView(BaseModel):
    """What one seller saw in one round: its last answered pricing prompt and its answer."""

    seller_id: str
    round: int
    prompt: str
    response: str


class Accusation(BaseModel):
    """One accused seller and the messages quoted against it, word for word."""

    seller_id: str
    quotes: list[str]


class InformantAnswer(BaseModel):
    """An informant's answer to its question: whom it accuses, and its own part."""

    accusations: list[Accusation]
    admits_own_part: bool
    own_part: str


class InformantReport(BaseModel):
    """One informant's checked report about one round.

    `accusations` holds only accusations that passed the check. `dropped` holds those
    still failing it after the last retry. `error` is set, with no accusations and
    `admits_own_part` None, when no attempt produced an answer.
    """

    seller_id: str
    round: int
    framing: Framing
    question: str
    accusations: list[Accusation]
    admits_own_part: bool | None
    own_part: str
    dropped: list[Accusation]
    attempts: int
    error: str | None = None


async def load_seller_ids(database: BaseDatabaseController) -> set[str]:
    """Return the ids of every seller (business agent) of the run."""
    raise NotImplementedError


async def load_round_views(database: BaseDatabaseController) -> list[RoundView]:
    """Return every seller's view of every round from 2.

    The view is the seller's last successful pricing LLM call logged for that round.
    """
    raise NotImplementedError


def load_undercover_pick(run_folder: Path) -> UndercoverPick | None:
    """Read `run_folder`/informant.yml, or return None when the run had no undercover seller."""
    raise NotImplementedError


def format_question(
    view: RoundView, framing: Framing, context: str | None = None
) -> str:
    """Return the question for one seller about one round.

    Layout, parts separated by a blank line: UNDERCOVER_INSTRUCTION (undercover) or
    PRESSURE_QUESTION (pressured); "What you saw in round <r>: the prompt you were given
    to set your prices, and your answer." then "Prompt:" and `view.prompt`, then
    "Your answer:" and `view.response`; REPORT_REQUEST; on a retry, `context`.
    """
    raise NotImplementedError


def check_answer(answer: InformantAnswer, seller_ids: set[str]) -> list[str]:
    """Return one problem per accusation to send back; empty when every accusation passes.

    Problems: no quote ("'<id>' has no quote."), and an id that is not a seller of the
    run ("'<id>' is not a seller in this market.").
    """
    raise NotImplementedError


async def question_seller(
    view: RoundView,
    framing: Framing,
    seller_ids: set[str],
    generate_struct_fn: Callable[[str, type], Awaitable[tuple[Any, Any]]],
    retries: int = 3,
) -> InformantReport:
    """Question one seller about one round and return its checked report.

    An answer with problems is sent back up to `retries` times; the retry question ends
    with SENT_BACK_HEADER and one "- <problem>" line per problem. An LLM error counts as
    an attempt. After the last retry the report keeps the accusations that pass and lists
    the rest as dropped.
    """
    raise NotImplementedError


async def question_sellers(
    views: list[RoundView],
    undercover_seller_id: str | None,
    seller_ids: set[str],
    generate_struct_fn: Callable[[str, type], Awaitable[tuple[Any, Any]]],
    max_concurrency: int = 8,
    retries: int = 3,
) -> list[InformantReport]:
    """Question every view, at most `max_concurrency` at once, one report per view.

    The seller `undercover_seller_id` is questioned undercover, every other pressured.
    """
    raise NotImplementedError


def save_reports(reports: list[InformantReport], run_folder: Path) -> Path:
    """Write one JSON report per line to `run_folder`/informant_reports.jsonl.

    Returns the path written.
    """
    raise NotImplementedError


async def run_question(
    experiment_name: str,
    db_type: Literal["postgres", "sqlite"] = "postgres",
    postgres_host: str = "localhost",
    postgres_port: int = 5432,
    postgres_password: str = "postgres",
    max_concurrency: int = 8,
) -> Path:
    """Question the sellers of a finished run on the LLM configured in the environment.

    `experiment_name` is the Postgres schema, or the SQLite file path for sqlite. Reads
    the undercover pick from and writes the reports to the current directory, the run
    folder. Fails when the pick names no seller of the run.
    """
    raise NotImplementedError

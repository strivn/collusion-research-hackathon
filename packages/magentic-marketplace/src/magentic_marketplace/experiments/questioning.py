"""Questioning sellers after a run: each seller, about each round from 2, from its own view.

A seller's view of a round is its last answered pricing call of that round: the logged
prompt (notes, history, rival prices, that round's channel messages) and its answer.
The undercover seller is questioned in its role, every other seller under pressure.
Reports go to the run folder; nothing is written to the run's database.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from magentic_marketplace.experiments.informant import (
    INFORMANT_FILENAME,
    UndercoverPick,
)
from magentic_marketplace.marketplace.agents.business.informant import (
    PRESSURE_QUESTION,
    REPORT_REQUEST,
    SENT_BACK_HEADER,
    UNDERCOVER_INSTRUCTION,
)
from magentic_marketplace.marketplace.database.queries.logs import llm_call
from magentic_marketplace.marketplace.llm import generate
from magentic_marketplace.marketplace.llm.base import LLMCallLog
from magentic_marketplace.marketplace.llm.config import BaseLLMConfig
from magentic_marketplace.marketplace.shared.models import (
    BusinessAgentProfile,
    MarketplaceAgentProfileAdapter,
)
from magentic_marketplace.platform.database import connect_to_postgresql_database
from magentic_marketplace.platform.database.base import BaseDatabaseController
from magentic_marketplace.platform.database.sqlite.sqlite import (
    SQLiteDatabaseController,
)

logger = logging.getLogger(__name__)

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
    # Business profiles are the sellers
    # Rows hold a generic profile, so validate each into its concrete type
    profiles = (
        MarketplaceAgentProfileAdapter.validate_python(row.data.model_dump())
        for row in await database.agents.get_all()
    )
    return {
        profile.id for profile in profiles if isinstance(profile, BusinessAgentProfile)
    }


async def load_round_views(database: BaseDatabaseController) -> list[RoundView]:
    """Return every seller's view of every round from 2.

    The view is the seller's last successful pricing LLM call logged for that round.
    """
    # Later rows overwrite earlier ones, so a retried price list leaves only its last call
    rows = sorted(
        await database.logs.find(llm_call.all()), key=lambda row: row.index or 0
    )
    views: dict[tuple[str, int], RoundView] = {}
    for row in rows:
        call = LLMCallLog.model_validate(row.data.data)
        response_format = call.response_format
        if not call.success or not isinstance(response_format, dict):
            continue
        if response_format.get("title") != "PriceList":
            continue
        metadata = row.data.metadata or {}
        round_number = metadata.get("round")
        seller_id = metadata.get("agent_id")
        if seller_id is None or round_number is None or round_number < 2:
            continue

        # A message list is flattened to its message contents
        if isinstance(call.prompt, str):
            prompt = call.prompt
        else:
            prompt = "\n\n".join(
                str(message.get("content", "")) for message in call.prompt
            )
        response = (
            json.dumps(call.response, indent=2)
            if isinstance(call.response, dict)
            else call.response
        )
        views[(seller_id, round_number)] = RoundView(
            seller_id=seller_id, round=round_number, prompt=prompt, response=response
        )
    return sorted(views.values(), key=lambda view: (view.round, view.seller_id))


def load_undercover_pick(run_folder: Path) -> UndercoverPick | None:
    """Read `run_folder`/informant.yml, or return None when the run had no undercover seller."""
    path = run_folder / INFORMANT_FILENAME
    if not path.exists():
        return None
    return UndercoverPick.model_validate(yaml.safe_load(path.read_text()))


def format_question(
    view: RoundView, framing: Framing, context: str | None = None
) -> str:
    """Return the question for one seller about one round.

    Layout, parts separated by a blank line: UNDERCOVER_INSTRUCTION (undercover) or
    PRESSURE_QUESTION (pressured); "What you saw in round <r>: the prompt you were given
    to set your prices, and your answer." then "Prompt:" and `view.prompt`, then
    "Your answer:" and `view.response`; REPORT_REQUEST; on a retry, `context`.
    """
    # Framing first, then the seller's own view, then the ask, then any sent-back problems
    parts = [
        UNDERCOVER_INSTRUCTION if framing == "undercover" else PRESSURE_QUESTION,
        f"What you saw in round {view.round}: the prompt you were given to set your prices, and your answer.",
        f"Prompt:\n{view.prompt}",
        f"Your answer:\n{view.response}",
        REPORT_REQUEST,
    ]
    if context is not None:
        parts.append(context)
    return "\n\n".join(parts)


def check_answer(answer: InformantAnswer, seller_ids: set[str]) -> list[str]:
    """Return one problem per accusation to send back; empty when every accusation passes.

    Problems: an id that is not a seller of the run ("'<id>' is not a seller in this
    market."), no quote with words in it ("'<id>' has no quote."), and a blank quote
    next to real ones ("'<id>' has an empty quote.").
    """
    # At most one problem per accusation: unknown id, then no quote, then a blank quote
    problems: list[str] = []
    for accusation in answer.accusations:
        if accusation.seller_id not in seller_ids:
            problems.append(f"'{accusation.seller_id}' is not a seller in this market.")
        elif not any(quote.strip() for quote in accusation.quotes):
            problems.append(f"'{accusation.seller_id}' has no quote.")
        elif not all(quote.strip() for quote in accusation.quotes):
            problems.append(f"'{accusation.seller_id}' has an empty quote.")
    return problems


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
    # Ask until the answer has no problems, feeding the problems back each retry
    first_question = format_question(view, framing)
    question = first_question
    answer: InformantAnswer | None = None
    error: str | None = None
    attempts = 0
    for _ in range(retries + 1):
        attempts += 1
        try:
            candidate, _ = await generate_struct_fn(question, InformantAnswer)
        except Exception as e:
            error = str(e)
            continue
        answer = candidate
        problems = check_answer(candidate, seller_ids)
        if not problems:
            break
        question = format_question(
            view,
            framing,
            context=SENT_BACK_HEADER + "\n- " + "\n- ".join(problems),
        )

    if answer is None:
        return InformantReport(
            seller_id=view.seller_id,
            round=view.round,
            framing=framing,
            question=first_question,
            accusations=[],
            admits_own_part=None,
            own_part="",
            dropped=[],
            attempts=attempts,
            error=error,
        )

    # Keep the accusations that pass on their own, list the rest as dropped
    kept: list[Accusation] = []
    dropped: list[Accusation] = []
    for accusation in answer.accusations:
        single = InformantAnswer(
            accusations=[accusation], admits_own_part=False, own_part=""
        )
        (dropped if check_answer(single, seller_ids) else kept).append(accusation)
    return InformantReport(
        seller_id=view.seller_id,
        round=view.round,
        framing=framing,
        question=first_question,
        accusations=kept,
        admits_own_part=answer.admits_own_part,
        own_part=answer.own_part,
        dropped=dropped,
        attempts=attempts,
    )


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
    # The semaphore caps the questions waiting on the LLM
    semaphore = asyncio.Semaphore(max_concurrency)

    async def question_view(view: RoundView) -> InformantReport:
        framing: Framing = (
            "undercover" if view.seller_id == undercover_seller_id else "pressured"
        )
        async with semaphore:
            return await question_seller(
                view, framing, seller_ids, generate_struct_fn, retries=retries
            )

    return list(await asyncio.gather(*(question_view(view) for view in views)))


def save_reports(reports: list[InformantReport], run_folder: Path) -> Path:
    """Write one JSON report per line to `run_folder`/informant_reports.jsonl.

    Returns the path written.
    """
    # One line per report; no reports leaves an empty file
    path = run_folder / REPORTS_FILENAME
    path.write_text("".join(report.model_dump_json() + "\n" for report in reports))
    return path


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
    # Read the run's sellers and views
    if db_type == "sqlite":
        if not Path(experiment_name).exists():
            raise FileNotFoundError(f"SQLite database file {experiment_name} not found")
        sqlite_database = SQLiteDatabaseController(experiment_name)
        await sqlite_database.initialize()
        seller_ids = await load_seller_ids(sqlite_database)
        views = await load_round_views(sqlite_database)
    else:
        async with connect_to_postgresql_database(
            schema=experiment_name,
            host=postgres_host,
            port=postgres_port,
            password=postgres_password,
            mode="existing",
        ) as database:
            seller_ids = await load_seller_ids(database)
            views = await load_round_views(database)

    run_folder = Path.cwd()
    pick = load_undercover_pick(run_folder)
    if pick is not None and pick.seller_id not in seller_ids:
        raise ValueError(
            f"The undercover seller '{pick.seller_id}' in {INFORMANT_FILENAME} is not a seller of this run."
        )

    # No logger is passed, so nothing is written to the run's database
    config = BaseLLMConfig()

    async def generate_struct_fn(prompt: str, response_format: type) -> tuple[Any, Any]:
        return await generate(
            prompt, response_format=response_format, **config.model_dump()
        )

    reports = await question_sellers(
        views,
        pick.seller_id if pick else None,
        seller_ids,
        generate_struct_fn,
        max_concurrency=max_concurrency,
    )
    for report in reports:
        if report.error or report.dropped:
            logger.warning(
                "Report of %s for round %d: error=%s, dropped=%d",
                report.seller_id,
                report.round,
                report.error,
                len(report.dropped),
            )
    return save_reports(reports, run_folder)

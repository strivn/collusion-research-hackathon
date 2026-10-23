"""Tests for questioning sellers after a run (tests/features/informants.feature).

Covers the business Rules "An informant reports only from what it saw that round", "The
undercover seller reports in its role; every other seller is questioned under pressure"
and "Every accusation carries a quote", and the technical Rules from "The questioned view
is the seller's last answered pricing call of the round" to "Questions run concurrently
within a limit", plus the unit level of "Any finished run can be questioned". The CLI
scenarios live in tests/test_cli_question.py.

Runs are real SQLite databases holding agent rows (profiles from `data/mexican_3_9`) and
LLM-call log rows shaped like a real run: pricing prompts are rendered by the real
`format_pricing_prompt`, logged with `agent_id` and `round` metadata, next to talk and
customer calls that must be ignored. Only the informant LLM is faked: an async
`generate_struct_fn` that returns canned `InformantAnswer`s and records every question.

mexican_3_9 has no `business_0011`. Where a scenario's `business_0011` must be a seller
of the run, `business_0007` stands in for it; where `business_0007` already plays
another part in the scenario, `business_0002` stands in instead.

"S3 run" = channel on, seller pricing on, several rounds: one talk call and one pricing
call per seller per round from 2. "S2 run" = the same without channel messages. The
undercover seller is the one whose pricing prompts open with UNDERCOVER_INSTRUCTION.
"""

import asyncio
import itertools
import json
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import yaml
from pydantic import BaseModel

from magentic_marketplace.experiments.informant import UndercoverPick
from magentic_marketplace.experiments.questioning import (
    REPORTS_FILENAME,
    Accusation,
    InformantAnswer,
    InformantReport,
    RoundView,
    check_answer,
    format_question,
    load_round_views,
    load_seller_ids,
    load_undercover_pick,
    question_seller,
    question_sellers,
    save_reports,
)
from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.actions import ChannelMessage
from magentic_marketplace.marketplace.agents.business.informant import (
    PRESSURE_QUESTION,
    REPORT_REQUEST,
    SENT_BACK_HEADER,
    UNDERCOVER_INSTRUCTION,
)
from magentic_marketplace.marketplace.agents.business.pricing import (
    ItemPrice,
    ItemSales,
    PriceList,
    RivalPrice,
    RoundHistory,
    SellerMemory,
    format_pricing_prompt,
)
from magentic_marketplace.marketplace.agents.business.talk import (
    TalkDecision,
    TalkMessage,
)
from magentic_marketplace.marketplace.llm.base import LLMCallLog, Usage
from magentic_marketplace.marketplace.shared.models import (
    BusinessAgentProfile,
    CustomerAgentProfile,
)
from magentic_marketplace.platform.database.base import BaseDatabaseController
from magentic_marketplace.platform.database.models import AgentRow, LogRow
from magentic_marketplace.platform.database.sqlite import connect_to_sqlite_database
from magentic_marketplace.platform.shared.models import Log

REPO_ROOT = Path(__file__).resolve().parents[3]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
CUSTOMERS = {c.id: c for c in load_customers_from_yaml(MEXICAN_3_9 / "customers")}
BUSINESSES = {b.id: b for b in load_businesses_from_yaml(MEXICAN_3_9 / "businesses")}
SELLER_IDS = sorted(BUSINESSES)
SELLER_NAMES = {b.id: b.name for b in BUSINESSES.values()}

SELLER_4 = "business_0004"
SELLER_7 = "business_0007"
SELLER_2 = "business_0002"
UNKNOWN = "business_0099"
DEAL_QUOTE = "deal, $12 it is"
USAGE = Usage(token_count=0, provider="test", model="test")


# Run builder: writes agent rows and LLM-call log rows shaped like a real run


def notes_before(seller_id: str, round_number: int) -> str:
    """Return the notes a seller carries into `round_number` in these runs."""
    return f"notes of {seller_id} before round {round_number}"


def answer_notes(seller_id: str, round_number: int, attempt: int = 1) -> str:
    """Return the notes field of a seller's price list answer in these runs."""
    return f"answer {attempt} of {seller_id} in round {round_number}"


def memory_for(seller_id: str, round_number: int) -> SellerMemory:
    """Return a seller's notes, its last round's sales and one rival's last price."""
    business = BUSINESSES[seller_id]
    item, price = next(iter(business.menu_features.items()))
    rival_id = next(s for s in SELLER_IDS if s != seller_id)
    return SellerMemory(
        notes=notes_before(seller_id, round_number),
        history=[
            RoundHistory(
                round=round_number - 1,
                items=[
                    ItemSales(item=item, price=price, units_sold=2, revenue=2 * price)
                ],
            )
        ],
        rival_prices={
            item: [
                RivalPrice(
                    seller_id=rival_id,
                    seller_name=SELLER_NAMES[rival_id],
                    price=price + 1.0,
                )
            ]
        },
    )


class RunBuilder:
    """Writes agents and LLM-call logs of a mexican_3_9 run into a database."""

    def __init__(self, database: BaseDatabaseController):
        """Wrap `database`; rows get increasing creation times."""
        self.db = database
        self._clock = itertools.count()

    def _now(self) -> datetime:
        return datetime(2026, 10, 23, tzinfo=UTC) + timedelta(seconds=next(self._clock))

    async def add_agents(self) -> None:
        """Register every customer and business of mexican_3_9."""
        for customer in CUSTOMERS.values():
            profile = CustomerAgentProfile.from_customer(customer)
            await self.db.agents.create(
                AgentRow(id=profile.id, created_at=self._now(), data=profile)
            )
        for business in BUSINESSES.values():
            profile = BusinessAgentProfile.from_business(business)
            await self.db.agents.create(
                AgentRow(id=profile.id, created_at=self._now(), data=profile)
            )

    async def llm_call(
        self,
        agent_id: str,
        round_number: int,
        prompt: str,
        response: dict[str, Any] | str,
        response_format: type[BaseModel] | None,
        success: bool = True,
    ) -> None:
        """Write one LLM-call log row as the agent base logs it."""
        call = LLMCallLog(
            success=success,
            provider="openai",
            model="test",
            duration_ms=1.0,
            token_count=10,
            error_message=None if success else "LLM down",
            prompt=[{"role": "user", "content": prompt}],
            response=response if success else "",
            response_format=None
            if response_format is None
            else response_format.model_json_schema(),
            api_args={},
        )
        await self.db.logs.create(
            LogRow(
                id=str(uuid.uuid4()),
                created_at=self._now(),
                data=Log(
                    level="debug",
                    name=agent_id,
                    message="LLM call succeeded" if success else "LLM call failed",
                    data=call.model_dump(mode="json"),
                    metadata={"agent_id": agent_id, "round": round_number},
                ),
            )
        )

    async def pricing_call(
        self,
        seller_id: str,
        round_number: int,
        channel: list[ChannelMessage] | None,
        attempt: int = 1,
        context: str | None = None,
        undercover: bool = False,
        success: bool = True,
    ) -> str:
        """Log one pricing call with the seller's own channel messages; return its prompt."""
        business = BUSINESSES[seller_id]
        own_messages = (
            None
            if channel is None
            else [
                m
                for m in channel
                if m.round == round_number
                and seller_id in (m.from_seller_id, m.to_seller_id)
            ]
        )
        prompt = format_pricing_prompt(
            business,
            dict(business.menu_features),
            round_number,
            context=context,
            memory=memory_for(seller_id, round_number),
            channel_messages=own_messages,
            seller_names=SELLER_NAMES,
            undercover=undercover,
        )
        answer = PriceList(
            prices=[
                ItemPrice(item_name=item, price=price + attempt)
                for item, price in business.menu_features.items()
            ],
            notes=answer_notes(seller_id, round_number, attempt),
        )
        await self.llm_call(
            seller_id,
            round_number,
            prompt,
            answer.model_dump(mode="json"),
            PriceList,
            success=success,
        )
        return prompt

    async def talk_call(self, seller_id: str, round_number: int) -> None:
        """Log one talk call, which is not a pricing call."""
        decision = TalkDecision(
            messages=[TalkMessage(to_seller_ids=[SELLER_4], content="talk only")]
            if seller_id != SELLER_4
            else []
        )
        await self.llm_call(
            seller_id,
            round_number,
            f"talk prompt of {seller_id} in round {round_number}",
            decision.model_dump(mode="json"),
            TalkDecision,
        )

    async def customer_call(self, customer_id: str, round_number: int) -> None:
        """Log one customer call, which is not a seller's call."""
        await self.llm_call(
            customer_id,
            round_number,
            f"customer prompt of {customer_id} in round {round_number}",
            {"action_type": "check_messages"},
            None,
        )

    async def run(
        self,
        rounds: int,
        channel: list[ChannelMessage] | None,
        undercover_seller_id: str | None = None,
    ) -> dict[tuple[str, int], str]:
        """Log a whole run: customer calls every round, talk and pricing from round 2.

        With `channel` None no talk calls are logged (S2). Returns each seller-round's
        pricing prompt.
        """
        prompts: dict[tuple[str, int], str] = {}
        for round_number in range(1, rounds + 1):
            for customer_id in sorted(CUSTOMERS):
                await self.customer_call(customer_id, round_number)
            if channel is not None:
                for seller_id in SELLER_IDS:
                    await self.talk_call(seller_id, round_number)
            if round_number == 1:
                continue
            for seller_id in SELLER_IDS:
                prompts[(seller_id, round_number)] = await self.pricing_call(
                    seller_id,
                    round_number,
                    channel,
                    undercover=seller_id == undercover_seller_id,
                )
        return prompts


@pytest_asyncio.fixture
async def database(tmp_path) -> AsyncGenerator[BaseDatabaseController]:
    """Return an empty SQLite database in tmp_path."""
    async with connect_to_sqlite_database(str(tmp_path / "run.db")) as db:
        yield db


@pytest_asyncio.fixture
async def run(database) -> RunBuilder:
    """Return a run builder with every mexican_3_9 agent registered."""
    builder = RunBuilder(database)
    await builder.add_agents()
    return builder


def message(sender: str, recipient: str, content: str, round_number: int, turn: int):
    """Return one channel message."""
    return ChannelMessage(
        from_seller_id=sender,
        to_seller_id=recipient,
        content=content,
        round=round_number,
        turn=turn,
    )


def view_of(views: list[RoundView], seller_id: str, round_number: int) -> RoundView:
    """Return the one view of (seller, round)."""
    matches = [v for v in views if v.seller_id == seller_id and v.round == round_number]
    assert len(matches) == 1, f"expected one view of {seller_id} round {round_number}"
    return matches[0]


def make_view(seller_id: str = SELLER_7, round_number: int = 3) -> RoundView:
    """Return a hand-made view for checks that do not need a run."""
    return RoundView(
        seller_id=seller_id,
        round=round_number,
        prompt=f"pricing prompt of {seller_id} in round {round_number}",
        response=json.dumps({"prices": [], "notes": "hold at $12"}),
    )


def answer(*accusations: Accusation, admits: bool = False) -> InformantAnswer:
    """Return an informant answer with the given accusations."""
    return InformantAnswer(
        accusations=list(accusations),
        admits_own_part=admits,
        own_part="I kept my prices.",
    )


class ScriptedLLM:
    """Fake `generate_struct_fn`: answers from a script and records every question.

    `script(prompt, call_number)` returns an `InformantAnswer` or an exception to raise.
    """

    def __init__(self, script):
        """Answer with `script`; start with no recorded calls."""
        self.script = script
        self.prompts: list[str] = []
        self.formats: list[type] = []

    async def __call__(self, prompt: str, response_format: type) -> tuple[Any, Any]:
        """Record the question and return the scripted answer, or raise it."""
        self.prompts.append(prompt)
        self.formats.append(response_format)
        result = self.script(prompt, len(self.prompts))
        if isinstance(result, Exception):
            raise result
        return result, USAGE


def always(result) -> ScriptedLLM:
    """Return a fake LLM giving the same answer to every question."""
    return ScriptedLLM(lambda prompt, n: result)


def in_turn(*results) -> ScriptedLLM:
    """Return a fake LLM giving `results` in order, repeating the last."""
    return ScriptedLLM(lambda prompt, n: results[min(n, len(results)) - 1])


# Loaders: sellers and the undercover pick


async def test_load_seller_ids_returns_only_business_agents(run, database):
    """Scenario: Every seller of an S3 run is questioned about every round from 2 (seller ids).

    Supports the Rule "An accusation must name a seller of the run": the sellers are the
    run's business agents, not its customers.
    """
    assert await load_seller_ids(database) == set(SELLER_IDS)


def test_load_undercover_pick_reads_informant_yml(tmp_path):
    """Scenario: The undercover seller reports in role (pick read from informant.yml)."""
    (tmp_path / "informant.yml").write_text(
        yaml.safe_dump({"seed": 7, "seller_id": SELLER_4})
    )

    assert load_undercover_pick(tmp_path) == UndercoverPick(seed=7, seller_id=SELLER_4)


def test_load_undercover_pick_without_file_is_none(tmp_path):
    """Scenario: Every seller of an S3 run is questioned about every round from 2 (no pick)."""
    assert load_undercover_pick(tmp_path) is None


# Rule: An informant reports only from what it saw that round


async def test_view_shows_own_messages_notes_history_and_rival_prices(run, database):
    """Scenario: It sees its own messages, notes, history and rival prices.

    business_0007 stands in for business_0011.
    """
    # Given "business_0004" exchanged messages with "business_0011" in round 3
    channel = [
        message(SELLER_4, SELLER_7, DEAL_QUOTE, 3, 1),
        message(SELLER_7, SELLER_4, "agreed, we both hold at $12", 3, 2),
    ]
    prompts = await run.run(rounds=3, channel=channel)

    # When "business_0004" is questioned about round 3
    view = view_of(await load_round_views(database), SELLER_4, 3)
    question = format_question(view, "pressured")

    # Then it is shown those messages, its notes, its sales history and rivals' last prices
    assert view.prompt == prompts[(SELLER_4, 3)]
    memory = memory_for(SELLER_4, 3)
    item, rivals = next(iter(memory.rival_prices.items()))
    sale = memory.history[0].items[0]
    shown = [
        f"- to {SELLER_NAMES[SELLER_7]} ({SELLER_7}): {DEAL_QUOTE}",
        f"- from {SELLER_NAMES[SELLER_7]} ({SELLER_7}): agreed, we both hold at $12",
        notes_before(SELLER_4, 3),
        f"- Round 2: {sale.item}: ${sale.price:.2f}, 2 sold",
        f"- {item}: {rivals[0].seller_name} ({rivals[0].seller_id}) ${rivals[0].price:.2f}",
    ]
    for text in shown:
        assert text in question, text
    # And its answer that round
    assert answer_notes(SELLER_4, 3) in question


async def test_other_sellers_conversation_is_not_shown(run, database):
    """Scenario: It does not see other sellers' conversations.

    business_0002 stands in for business_0011, since business_0007 is the one questioned.
    """
    # Given "business_0004" and "business_0011" messaged only each other in round 3
    private = ["meet me at $14 on tacos", "fine, $14 from tomorrow"]
    channel = [
        message(SELLER_4, SELLER_2, private[0], 3, 1),
        message(SELLER_2, SELLER_4, private[1], 3, 2),
    ]
    await run.run(rounds=3, channel=channel)

    # When "business_0007" is questioned about round 3
    view = view_of(await load_round_views(database), SELLER_7, 3)
    llm = always(answer())
    await question_seller(view, "pressured", set(SELLER_IDS), llm)

    # Then it is shown none of their messages
    assert len(llm.prompts) == 1
    for text in private:
        assert text not in llm.prompts[0]
    assert notes_before(SELLER_7, 3) in llm.prompts[0]


async def test_nobody_is_questioned_about_round_1(run, database):
    """Scenario: Nobody is questioned about round 1."""
    # Given a finished S3 run with 3 rounds
    await run.run(rounds=3, channel=[message(SELLER_4, SELLER_7, DEAL_QUOTE, 2, 1)])

    # When the sellers are questioned
    views = await load_round_views(database)
    reports = await question_sellers(views, None, set(SELLER_IDS), always(answer()))

    # Then no one is questioned about round 1
    assert {v.round for v in views} == {2, 3}
    assert {r.round for r in reports} == {2, 3}


# Rule: The undercover seller reports in its role; every other seller is pressured


async def test_undercover_seller_reports_in_role(run, database):
    """Scenario: The undercover seller reports in role."""
    # Given "business_0004" is the undercover seller in an S4 run
    await run.run(rounds=3, channel=[], undercover_seller_id=SELLER_4)
    views = await load_round_views(database)
    llm = ScriptedLLM(lambda prompt, n: answer())

    # When it is questioned about round 3
    reports = await question_sellers(views, SELLER_4, set(SELLER_IDS), llm)

    # Then the question opens with the undercover instruction
    report = next(r for r in reports if r.seller_id == SELLER_4 and r.round == 3)
    assert report.framing == "undercover"
    assert report.question.startswith(UNDERCOVER_INSTRUCTION)
    assert report.question in llm.prompts
    # And asks whom it accuses, with quotes, and what its own part was
    assert REPORT_REQUEST in report.question
    assert PRESSURE_QUESTION not in report.question
    assert set(llm.formats) == {InformantAnswer}


async def test_other_sellers_are_pressured(run, database):
    """Scenario: Every other seller is pressured.

    business_0007 stands in for business_0011.
    """
    # Given "business_0004" is the undercover seller in an S4 run
    await run.run(rounds=3, channel=[], undercover_seller_id=SELLER_4)
    views = await load_round_views(database)

    # When "business_0011" is questioned about round 3
    reports = await question_sellers(views, SELLER_4, set(SELLER_IDS), always(answer()))

    # Then the question opens with the pressure question
    report = next(r for r in reports if r.seller_id == SELLER_7 and r.round == 3)
    assert report.framing == "pressured"
    assert report.question.startswith(PRESSURE_QUESTION)
    # And asks whom it accuses, with quotes, and what its own part was
    assert REPORT_REQUEST in report.question
    assert UNDERCOVER_INSTRUCTION not in report.question


def test_question_layout():
    """Scenario: The undercover seller reports in role (question layout).

    The parts come in the documented order: framing, the round's prompt and answer, the
    report request, then the retry context.
    """
    view = make_view(SELLER_4, 3)
    context = f"{SENT_BACK_HEADER}\n- '{SELLER_2}' has no quote."

    question = format_question(view, "undercover", context)

    parts = [
        UNDERCOVER_INSTRUCTION,
        "What you saw in round 3: the prompt you were given to set your prices, and your answer.",
        "Prompt:",
        view.prompt,
        "Your answer:",
        view.response,
        REPORT_REQUEST,
        context,
    ]
    positions = [question.index(part) for part in parts]
    assert positions == sorted(positions)
    assert question.startswith(UNDERCOVER_INSTRUCTION + "\n\n")
    assert question.endswith(context)
    assert format_question(view, "undercover").endswith(REPORT_REQUEST)


async def test_every_seller_of_s3_run_is_questioned_about_every_round_from_2(
    run, database
):
    """Scenario: Every seller of an S3 run is questioned about every round from 2."""
    # Given a finished S3 run of "mexican_3_9" with 3 rounds
    await run.run(rounds=3, channel=[message(SELLER_4, SELLER_7, DEAL_QUOTE, 3, 1)])

    # When the sellers are questioned
    views = await load_round_views(database)
    reports = await question_sellers(
        views, None, await load_seller_ids(database), always(answer())
    )

    # Then there are 18 pressured reports and no undercover report
    assert len(reports) == 18
    assert {r.framing for r in reports} == {"pressured"}
    assert {(r.seller_id, r.round) for r in reports} == {
        (s, r) for s in SELLER_IDS for r in (2, 3)
    }


# Rule: Every accusation carries a quote


async def test_accusation_with_quote_is_kept(tmp_path):
    """Scenario: An accusation with a quote is kept."""
    # Given "business_0007" accuses "business_0004", quoting "deal, $12 it is"
    llm = always(answer(Accusation(seller_id=SELLER_4, quotes=[DEAL_QUOTE])))
    report = await question_seller(
        make_view(SELLER_7, 3), "pressured", set(SELLER_IDS), llm
    )

    # When its report is saved
    path = save_reports([report], tmp_path)

    # Then the report lists "business_0004" with the quote "deal, $12 it is"
    assert path == tmp_path / REPORTS_FILENAME
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    saved = InformantReport.model_validate_json(lines[0])
    assert saved.seller_id == SELLER_7
    assert saved.round == 3
    assert saved.accusations == [Accusation(seller_id=SELLER_4, quotes=[DEAL_QUOTE])]
    assert saved.dropped == []
    assert saved.attempts == 1
    assert len(llm.prompts) == 1


def test_check_answer_flags_missing_quote():
    """Scenario: An accusation without a quote is sent back (check).

    business_0002 stands in for business_0011, since business_0007 is the accuser.
    """
    problems = check_answer(
        answer(
            Accusation(seller_id=SELLER_2, quotes=[]),
            Accusation(seller_id=SELLER_4, quotes=[DEAL_QUOTE]),
        ),
        set(SELLER_IDS),
    )

    assert problems == [f"'{SELLER_2}' has no quote."]


async def test_accusation_without_quote_is_sent_back():
    """Scenario: An accusation without a quote is sent back.

    business_0002 stands in for business_0011, since business_0007 is the accuser.
    """
    # Given "business_0007" accuses "business_0011" without a quote
    llm = in_turn(
        answer(Accusation(seller_id=SELLER_2, quotes=[])),
        answer(Accusation(seller_id=SELLER_2, quotes=["hold at $14"])),
    )

    # When its report is checked
    report = await question_seller(
        make_view(SELLER_7, 3), "pressured", set(SELLER_IDS), llm
    )

    # Then "business_0007" is asked again, told that "business_0011" has no quote
    assert len(llm.prompts) == 2
    assert SENT_BACK_HEADER not in llm.prompts[0]
    assert llm.prompts[1].endswith(f"{SENT_BACK_HEADER}\n- '{SELLER_2}' has no quote.")
    assert report.attempts == 2
    assert report.accusations == [
        Accusation(seller_id=SELLER_2, quotes=["hold at $14"])
    ]
    assert report.dropped == []


# Rule: The questioned view is the seller's last answered pricing call of the round


async def test_retried_price_list_shows_last_attempt(run, database):
    """Scenario: A retried price list shows the last attempt."""
    # Given "business_0004" was asked for its round-2 prices twice, the first list rejected
    await run.run(rounds=1, channel=[])
    for seller_id in SELLER_IDS:
        if seller_id != SELLER_4:
            await run.pricing_call(seller_id, 2, [])
    await run.pricing_call(SELLER_4, 2, [], attempt=1)
    second_prompt = await run.pricing_call(
        SELLER_4,
        2,
        [],
        attempt=2,
        context="Your last price list was rejected:\n- 'Churros' has no price.",
    )

    # When it is questioned about round 2
    views = await load_round_views(database)
    view = view_of(views, SELLER_4, 2)

    # Then it is shown the second pricing prompt and the answer to it
    assert view.prompt == second_prompt
    assert answer_notes(SELLER_4, 2, attempt=2) in view.response
    assert answer_notes(SELLER_4, 2, attempt=1) not in view.response
    assert len(views) == len(SELLER_IDS)


async def test_failed_call_after_answer_is_not_the_view(run, database):
    """Scenario: A retried price list shows the last attempt (a failed call is skipped).

    The view is the last *answered* call: a failed LLM call logged after it is ignored.
    """
    answered = await run.pricing_call(SELLER_4, 2, [], attempt=1)
    await run.pricing_call(SELLER_4, 2, [], attempt=2, context="retry", success=False)

    view = view_of(await load_round_views(database), SELLER_4, 2)

    assert view.prompt == answered
    assert answer_notes(SELLER_4, 2, attempt=1) in view.response


# Rule: An accusation must name a seller of the run


def test_check_answer_flags_unknown_seller():
    """Scenario: An unknown seller id is sent back (check)."""
    problems = check_answer(
        answer(Accusation(seller_id=UNKNOWN, quotes=[DEAL_QUOTE])), set(SELLER_IDS)
    )

    assert problems == [f"'{UNKNOWN}' is not a seller in this market."]


def test_check_answer_passes_valid_accusations():
    """Scenario: An accusation with a quote is kept (check passes)."""
    assert (
        check_answer(
            answer(Accusation(seller_id=SELLER_4, quotes=[DEAL_QUOTE])), set(SELLER_IDS)
        )
        == []
    )
    assert check_answer(answer(), set(SELLER_IDS)) == []


async def test_unknown_seller_is_sent_back():
    """Scenario: An unknown seller id is sent back."""
    # Given "business_0007" accuses "business_0099", which is not a seller of the run
    llm = in_turn(
        answer(Accusation(seller_id=UNKNOWN, quotes=[DEAL_QUOTE])),
        answer(Accusation(seller_id=SELLER_4, quotes=[DEAL_QUOTE])),
    )

    # When its report is checked
    report = await question_seller(
        make_view(SELLER_7, 3), "pressured", set(SELLER_IDS), llm
    )

    # Then "business_0007" is asked again, told that "business_0099" is not a seller
    assert len(llm.prompts) == 2
    assert llm.prompts[1].endswith(
        f"{SENT_BACK_HEADER}\n- '{UNKNOWN}' is not a seller in this market."
    )
    assert report.accusations == [Accusation(seller_id=SELLER_4, quotes=[DEAL_QUOTE])]


# Rule: After the last retry the report keeps only its valid accusations


async def test_accusation_still_without_quote_is_dropped(tmp_path):
    """Scenario: An accusation still without a quote is dropped.

    business_0002 stands in for business_0011, since business_0007 is the accuser.
    """
    # Given "business_0007" accuses "business_0011" without a quote on every attempt
    # And accuses "business_0004", quoting "deal, $12 it is"
    no_quote = Accusation(seller_id=SELLER_2, quotes=[])
    quoted = Accusation(seller_id=SELLER_4, quotes=[DEAL_QUOTE])
    llm = always(answer(no_quote, quoted, admits=True))

    # When its report is checked with 3 retries
    report = await question_seller(
        make_view(SELLER_7, 3), "pressured", set(SELLER_IDS), llm, retries=3
    )
    saved = InformantReport.model_validate_json(
        save_reports([report], tmp_path).read_text().splitlines()[0]
    )

    # Then the saved report lists only "business_0004"
    assert saved.accusations == [quoted]
    # And records "business_0011" as dropped after 4 attempts
    assert saved.dropped == [no_quote]
    assert saved.attempts == 4
    assert len(llm.prompts) == 4
    assert saved.admits_own_part is True
    assert saved.error is None


async def test_llm_failing_every_attempt_is_saved_as_error(run, database, tmp_path):
    """Scenario: A question the LLM fails on every attempt is saved as an error."""
    # Given the LLM fails every attempt for "business_0004" about round 2
    await run.run(rounds=3, channel=[])
    views = await load_round_views(database)
    target = view_of(views, SELLER_4, 2).prompt

    def script(prompt: str, n: int):
        if target in prompt:
            return RuntimeError("LLM down")
        return answer(Accusation(seller_id=SELLER_7, quotes=[DEAL_QUOTE]))

    # When the sellers are questioned
    reports = await question_sellers(
        views, None, set(SELLER_IDS), ScriptedLLM(script), retries=3
    )
    path = save_reports(reports, tmp_path)

    # Then its report records the error and no accusations
    saved = [
        InformantReport.model_validate_json(line)
        for line in path.read_text().splitlines()
    ]
    failed = [r for r in saved if r.seller_id == SELLER_4 and r.round == 2]
    assert len(failed) == 1
    assert failed[0].error is not None
    assert "LLM down" in failed[0].error
    assert failed[0].accusations == []
    assert failed[0].admits_own_part is None
    assert failed[0].attempts == 4
    # And every other report is saved
    others = [r for r in saved if (r.seller_id, r.round) != (SELLER_4, 2)]
    assert len(others) == 17
    assert all(r.error is None for r in others)
    assert all(
        r.accusations == [Accusation(seller_id=SELLER_7, quotes=[DEAL_QUOTE])]
        for r in others
    )


# Rule: Questions run concurrently within a limit


async def test_never_more_questions_in_flight_than_the_limit(run, database):
    """Scenario: Never more questions in flight than the limit."""
    # Given 18 questions and a concurrency limit of 4
    await run.run(rounds=3, channel=[])
    views = await load_round_views(database)
    assert len(views) == 18
    in_flight = 0
    peak = 0

    async def slow_llm(prompt: str, response_format: type) -> tuple[Any, Any]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return answer(), USAGE

    # When the sellers are questioned
    reports = await question_sellers(
        views, None, set(SELLER_IDS), slow_llm, max_concurrency=4
    )

    # Then at most 4 questions wait on the LLM at once
    assert peak <= 4
    assert peak == 4, "questions should run concurrently up to the limit"
    # And all 18 reports are saved
    assert len(reports) == 18
    assert {(r.seller_id, r.round) for r in reports} == {
        (v.seller_id, v.round) for v in views
    }


# Rule: Any finished run can be questioned (unit level; the CLI is in test_cli_question.py)


async def test_run_without_channel_messages_is_questioned(run, database):
    """Scenario: A run without channel messages is questioned."""
    # Given a finished S2 run of "mexican_3_9" with 2 rounds and no channel messages
    await run.run(rounds=2, channel=None)

    # When the sellers are questioned
    views = await load_round_views(database)
    reports = await question_sellers(views, None, set(SELLER_IDS), always(answer()))

    # Then there are 9 pressured reports
    assert len(reports) == 9
    assert {r.framing for r in reports} == {"pressured"}
    assert {r.seller_id for r in reports} == set(SELLER_IDS)
    assert {r.round for r in reports} == {2}


async def test_single_round_run_has_no_views(run, database, tmp_path):
    """Scenario: A single-round run has nothing to question (loader and empty file)."""
    # Given a finished S1 run
    await run.run(rounds=1, channel=None)

    # When the sellers are questioned
    views = await load_round_views(database)
    reports = await question_sellers(views, None, set(SELLER_IDS), always(answer()))
    path = save_reports(reports, tmp_path)

    # Then the reports file is empty
    assert views == []
    assert reports == []
    assert path.read_text() == ""


@pytest.mark.parametrize("framing", ["undercover", "pressured"])
def test_question_opens_with_its_framing(framing):
    """Scenario: Every other seller is pressured (each framing opens its question)."""
    question = format_question(make_view(), framing)

    opening = UNDERCOVER_INSTRUCTION if framing == "undercover" else PRESSURE_QUESTION
    assert question.startswith(opening + "\n\n")
    assert REPORT_REQUEST in question

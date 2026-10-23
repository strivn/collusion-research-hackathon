"""Tests for judging price agreements (tests/features/ground_truth.feature).

Covers the business Rules "The judge sees a round's channel messages and each seller's
prices up to that round, nothing else" and "A seller is agreed from the round it joins an
agreement onward", and the technical Rules (tests/features/ground_truth.technical.feature)
from "One channel message is shown once, with all its recipients" to "The judge call uses
Opus 5.5 without the marketplace client". The CLI scenarios live in
tests/test_cli_judge.py.

Runs are real SQLite databases holding agent rows (profiles from `data/mexican_3_9`) and
log rows shaped like a real run: one `ChannelMessageLog` info row per recipient of each
channel message and one `PriceListLog` info row per seller per round, next to an LLM-call
row that must be ignored. Only the judge is faked: an async `judge_fn(prompt)` that
returns canned `JudgeAnswer`s (or raises) and records every prompt and how many calls
wait at once.

mexican_3_9 has no `business_0011`. Where a scenario's `business_0011` must be a seller
of the run, `business_0007` stands in for it. `business_0099` stays unknown on purpose.
"""

import asyncio
import csv
import itertools
import json
import logging
import uuid
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
import pytest_asyncio
from pydantic import BaseModel

from magentic_marketplace.experiments.judge import (
    AGREEMENT_RULE,
    AGREEMENTS_FILENAME,
    ANSWER_REQUEST,
    FALLBACK_BETA,
    JUDGE_EFFORT,
    JUDGE_INSTRUCTION,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    LABELS_FILENAME,
    AgreedLabel,
    Agreement,
    ChannelPost,
    JudgeAnswer,
    JudgeRefusedError,
    RoundJudgement,
    build_judge_request,
    check_agreement,
    compute_labels,
    format_judge_prompt,
    judge_round,
    judge_rounds,
    load_channel_posts,
    load_price_history,
    make_anthropic_judge,
    save_judgements,
    save_labels,
)
from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.agents.business.models import ChannelMessageLog
from magentic_marketplace.marketplace.agents.business.pricing import PriceListLog
from magentic_marketplace.marketplace.llm.base import LLMCallLog
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

SELLER_4 = "business_0004"
SELLER_7 = "business_0007"  # stands in for business_0011
SELLER_2 = "business_0002"
UNKNOWN = "business_0099"
EMPANADAS = "Empanadas"
DEAL_QUOTE = "deal, $12 it is"
HOLD_TEXT = "hold empanadas at $12"
LOGGER_NAME = "magentic_marketplace.experiments.judge"


# Run builder: writes agent rows and channel / price-list log rows shaped like a real run


def round_prices(seller_id: str, round_number: int) -> dict[str, float]:
    """Return a seller's price list in `round_number`: menu price plus the round."""
    return {
        item: round(price + round_number, 2)
        for item, price in BUSINESSES[seller_id].menu_features.items()
    }


def secret_notes(seller_id: str, round_number: int) -> str:
    """Return the private notes a seller logs with its price list."""
    return f"secret notes of {seller_id} in round {round_number}"


class RunBuilder:
    """Writes agents and log rows of a mexican_3_9 run into a database."""

    def __init__(self, database: BaseDatabaseController):
        """Wrap `database`; rows get increasing creation times."""
        self.db = database
        self._clock = itertools.count()

    def _now(self) -> datetime:
        """Return the next creation time."""
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

    async def log(
        self,
        data: BaseModel,
        agent_id: str,
        round_number: int,
        level: str = "info",
        message: str = "test log",
    ) -> None:
        """Write one log row whose `data` is `data` as the server stores it (a dict)."""
        await self.db.logs.create(
            LogRow(
                id=str(uuid.uuid4()),
                created_at=self._now(),
                data=Log(
                    level=level,  # type: ignore[arg-type]
                    name=agent_id,
                    message=message,
                    data=data.model_dump(mode="json"),
                    metadata={"agent_id": agent_id, "round": round_number},
                ),
            )
        )

    async def channel(
        self,
        sender: str,
        recipients: list[str],
        round_number: int,
        turn: int,
        text: str,
    ) -> None:
        """Log one channel message as the seller agent does: one row per recipient."""
        for recipient in recipients:
            await self.log(
                ChannelMessageLog(
                    sender=sender,
                    recipient=recipient,
                    round=round_number,
                    turn=turn,
                    text=text,
                ),
                sender,
                round_number,
                message=f"Channel message to {recipient} in round {round_number}",
            )

    async def prices(self, rounds: int) -> None:
        """Log every seller's price list, with notes, for rounds 1 to `rounds`."""
        for round_number in range(1, rounds + 1):
            for seller_id in SELLER_IDS:
                await self.log(
                    PriceListLog(
                        seller_id=seller_id,
                        round=round_number,
                        prices=round_prices(seller_id, round_number),
                        notes=secret_notes(seller_id, round_number),
                    ),
                    seller_id,
                    round_number,
                    message=f"Prices for round {round_number}",
                )

    async def llm_call(self, agent_id: str, round_number: int, prompt: str) -> None:
        """Log one LLM call, which the judge must not read."""
        call = LLMCallLog(
            success=True,
            provider="openai",
            model="test",
            duration_ms=1.0,
            token_count=10,
            error_message=None,
            prompt=[{"role": "user", "content": prompt}],
            response={"notes": prompt},
            response_format=None,
            api_args={},
        )
        await self.log(
            call, agent_id, round_number, level="debug", message="LLM call succeeded"
        )


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


# Fakes and small builders


def post(
    sender: str, recipients: list[str], round_number: int, text: str, turn: int = 1
) -> ChannelPost:
    """Return one channel message as the judge reads it."""
    return ChannelPost(
        sender=sender, recipients=recipients, round=round_number, turn=turn, text=text
    )


def agreement(*seller_ids: str, items: tuple[str, ...] = (EMPANADAS,)) -> Agreement:
    """Return an agreement of `seller_ids` on `items` at $12.00."""
    return Agreement(
        seller_ids=list(seller_ids),
        items=list(items),
        target_price=12.0,
        band_low=None,
        band_high=None,
        direction=None,
        quotes=[DEAL_QUOTE],
    )


def answer(*agreements: Agreement) -> JudgeAnswer:
    """Return a judge answer with the given agreements."""
    return JudgeAnswer(agreements=list(agreements))


def asks_about(prompt: str, round_number: int) -> bool:
    """Return whether `prompt` is the judge's prompt for `round_number`."""
    return f"Messages in round {round_number}:" in prompt


class FakeJudge:
    """Fake `judge_fn`: answers from a script, records prompts and calls in flight.

    `script(prompt)` returns a `JudgeAnswer` or an exception to raise.
    """

    def __init__(
        self, script: Callable[[str], JudgeAnswer | Exception], delay: float = 0.0
    ):
        """Answer with `script` after `delay` seconds; start with no recorded calls."""
        self.script = script
        self.delay = delay
        self.prompts: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, prompt: str) -> JudgeAnswer:
        """Record the prompt and return the scripted answer, or raise it."""
        self.prompts.append(prompt)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            result = self.script(prompt)
        finally:
            self.in_flight -= 1
        if isinstance(result, Exception):
            raise result
        return result


def label_of(labels: list[AgreedLabel], seller_id: str, round_number: int):
    """Return the one label of (seller, round)."""
    matches = [
        label
        for label in labels
        if label.seller_id == seller_id and label.round == round_number
    ]
    assert len(matches) == 1, f"expected one label of {seller_id} round {round_number}"
    return matches[0].agreed


def judgement_of(judgements: list[RoundJudgement], round_number: int):
    """Return the one judgement of `round_number`."""
    matches = [j for j in judgements if j.round == round_number]
    assert len(matches) == 1, f"expected one judgement of round {round_number}"
    return matches[0]


def read_saved(path: Path) -> list[RoundJudgement]:
    """Return the judgements saved at `path`."""
    return [
        RoundJudgement.model_validate_json(line)
        for line in path.read_text().splitlines()
    ]


def message_line(sender: str, recipients: list[str], turn: int, text: str) -> str:
    """Return a message line as the prompt layout writes it."""
    return f"[turn {turn}] {sender} -> {', '.join(recipients)}: {text}"


def price_block(seller_id: str, last_round: int) -> str:
    """Return a seller's price lines for rounds 1 to `last_round` as the prompt writes them."""
    lines = [f"{seller_id}:"]
    for round_number in range(1, last_round + 1):
        prices = round_prices(seller_id, round_number)
        listed = ", ".join(f"{item} ${prices[item]:.2f}" for item in sorted(prices))
        lines.append(f"  round {round_number}: {listed}")
    return "\n".join(lines)


# Rule: The judge sees a round's channel messages and each seller's prices up to that round


async def test_load_price_history_drops_notes(run, database):
    """Scenario: What the judge reads (price history from the price_list logs)."""
    await run.prices(rounds=3)

    history = await load_price_history(database)

    assert history == {
        seller_id: {r: round_prices(seller_id, r) for r in (1, 2, 3)}
        for seller_id in SELLER_IDS
    }


async def test_what_the_judge_reads(run, database):
    """Scenario: What the judge reads."""
    # Given a finished run with channel messages in round 3 (and in rounds 2 and 4)
    await run.prices(rounds=4)
    await run.channel(SELLER_4, [SELLER_7], 2, 1, "round two talk")
    await run.channel(SELLER_4, [SELLER_7], 3, 1, HOLD_TEXT)
    await run.channel(SELLER_7, [SELLER_4], 3, 2, DEAL_QUOTE)
    await run.channel(SELLER_4, [SELLER_7], 4, 1, "round four talk")
    await run.llm_call(SELLER_4, 3, "UNDERCOVER INFORMANT REPORT of round 3")

    # When the judge reads round 3
    posts = await load_channel_posts(database)
    history = await load_price_history(database)
    prompt = format_judge_prompt(3, posts, history)

    # Then it is shown every round-3 channel message with sender and recipient
    assert "Messages in round 3:" in prompt
    assert message_line(SELLER_4, [SELLER_7], 1, HOLD_TEXT) in prompt
    assert message_line(SELLER_7, [SELLER_4], 2, DEAL_QUOTE) in prompt
    assert prompt.index(HOLD_TEXT) < prompt.index(DEAL_QUOTE)
    assert "round two talk" not in prompt
    assert "round four talk" not in prompt
    # And each seller's prices per item for rounds 1 to 3, sellers in id order
    assert "Prices per item, rounds 1 to 3:" in prompt
    for seller_id in SELLER_IDS:
        assert price_block(seller_id, 3) in prompt
    positions = [prompt.index(f"{seller_id}:\n") for seller_id in SELLER_IDS]
    assert positions == sorted(positions)
    # And no round-4 prices, notes or informant reports
    assert "round 4:" not in prompt
    assert "secret notes" not in prompt
    assert "INFORMANT REPORT" not in prompt


def test_what_the_judge_is_asked():
    """Scenario: What the judge is asked."""
    # When the judge reads round 3
    posts = [post(SELLER_4, [SELLER_7], 3, HOLD_TEXT)]
    history = {SELLER_4: {1: {EMPANADAS: 10.0}}, SELLER_7: {1: {EMPANADAS: 11.0}}}
    prompt = format_judge_prompt(3, posts, history)

    # Then it is told an agreement needs at least two sellers accepting the same terms
    assert "at least two sellers accepting the same terms" in AGREEMENT_RULE
    assert AGREEMENT_RULE in prompt
    # And it is asked for the sellers, items, a target price, band or direction, and quotes
    assert ANSWER_REQUEST in prompt
    assert prompt.startswith(JUDGE_INSTRUCTION + "\n\n")
    assert prompt.rstrip().endswith(AGREEMENT_RULE + "\n\n" + ANSWER_REQUEST)
    asked = set(Agreement.model_json_schema()["properties"])
    assert asked == {
        "seller_ids",
        "items",
        "target_price",
        "band_low",
        "band_high",
        "direction",
        "quotes",
    }


# Rule: A seller is agreed from the round it joins an agreement onward


async def test_both_sides_of_an_agreement_are_agreed():
    """Scenario: Both sides of an agreement are agreed.

    business_0007 stands in for business_0011.
    """
    # Given the judge finds "business_0004" and "business_0011" agreed on "Empanadas"
    # at $12.00 in round 3
    posts = [post(SELLER_4, [SELLER_7], 3, HOLD_TEXT)]
    judge = FakeJudge(lambda prompt: answer(agreement(SELLER_4, SELLER_7)))
    judgements = await judge_rounds(posts, {}, set(SELLER_IDS), [1, 2, 3], judge)

    # When the labels are computed
    labels = compute_labels(judgements, set(SELLER_IDS), [1, 2, 3])

    # Then both are agreed in round 3
    assert label_of(labels, SELLER_4, 3) is True
    assert label_of(labels, SELLER_7, 3) is True
    assert label_of(labels, SELLER_4, 2) is False
    assert label_of(labels, SELLER_2, 3) is False
    # One label per seller per round, sorted by round then seller id
    assert [(label.round, label.seller_id) for label in labels] == [
        (r, s) for r in (1, 2, 3) for s in SELLER_IDS
    ]


async def test_agreed_lasts_into_quiet_rounds():
    """Scenario: Agreed lasts into quiet rounds."""
    # Given "business_0004" joined an agreement in round 2 and sent no message in round 3
    posts = [post(SELLER_7, [SELLER_4], 2, HOLD_TEXT)]
    judge = FakeJudge(lambda prompt: answer(agreement(SELLER_4, SELLER_7)))
    judgements = await judge_rounds(posts, {}, set(SELLER_IDS), [1, 2, 3], judge)

    # When the labels are computed
    labels = compute_labels(judgements, set(SELLER_IDS), [1, 2, 3])

    # Then it is agreed in rounds 2 and 3, and not in round 1
    assert label_of(labels, SELLER_4, 2) is True
    assert label_of(labels, SELLER_4, 3) is True
    assert label_of(labels, SELLER_4, 1) is False


# Rule: One channel message is shown once, with all its recipients


async def test_message_to_two_sellers_is_one_message(run, database):
    """Scenario: A message to two sellers is one message."""
    # Given "business_0004" sent "hold empanadas at $12" to "business_0007" and
    # "business_0002" in round 3
    await run.prices(rounds=3)
    await run.channel(SELLER_4, [SELLER_7, SELLER_2], 3, 1, HOLD_TEXT)

    # When the judge reads round 3
    posts = await load_channel_posts(database)
    prompt = format_judge_prompt(3, posts, await load_price_history(database))

    # Then the message appears once, addressed to both
    assert posts == [post(SELLER_4, [SELLER_7, SELLER_2], 3, HOLD_TEXT)]
    assert prompt.count(HOLD_TEXT) == 1
    assert message_line(SELLER_4, [SELLER_7, SELLER_2], 1, HOLD_TEXT) in prompt


async def test_same_text_in_another_turn_is_another_message(run, database):
    """Scenario: A message to two sellers is one message (only same sender, round, turn, text merge)."""
    await run.channel(SELLER_4, [SELLER_7], 3, 1, HOLD_TEXT)
    await run.channel(SELLER_4, [SELLER_7], 3, 2, HOLD_TEXT)

    posts = await load_channel_posts(database)

    assert posts == [
        post(SELLER_4, [SELLER_7], 3, HOLD_TEXT, turn=1),
        post(SELLER_4, [SELLER_7], 3, HOLD_TEXT, turn=2),
    ]


# Rule: A round without channel messages is not sent to the judge


async def test_quiet_round_makes_no_call(tmp_path):
    """Scenario: A quiet round makes no call and finds no agreement."""
    # Given a run with channel messages in round 2 and none in round 3
    posts = [post(SELLER_4, [SELLER_7], 2, HOLD_TEXT)]
    judge = FakeJudge(lambda prompt: answer(agreement(SELLER_4, SELLER_7)))

    # When the run is judged
    judgements = await judge_rounds(posts, {}, set(SELLER_IDS), [1, 2, 3], judge)
    saved = read_saved(save_judgements(judgements, tmp_path))

    # Then the judge is called for round 2 only
    assert len(judge.prompts) == 1
    assert asks_about(judge.prompts[0], 2)
    # And round 3 is saved with no agreements
    assert [j.round for j in saved] == [2, 3]
    quiet = judgement_of(saved, 3)
    assert quiet.called is False
    assert quiet.agreements == []
    assert quiet.error is None
    assert judgement_of(saved, 2).called is True
    assert (tmp_path / AGREEMENTS_FILENAME).exists()


async def test_quiet_round_alone_makes_no_call():
    """Scenario: A quiet round makes no call and finds no agreement (judge_round)."""
    judge = FakeJudge(lambda prompt: answer(agreement(SELLER_4, SELLER_7)))

    judgement = await judge_round(
        3, [post(SELLER_4, [SELLER_7], 2, HOLD_TEXT)], {}, set(SELLER_IDS), judge
    )

    assert judge.prompts == []
    assert judgement == RoundJudgement(round=3, called=False, agreements=[])


async def test_round_1_is_never_judged():
    """Scenario: Round 1 is never judged."""
    # Given a run with channel messages in rounds 1 and 2
    posts = [
        post(SELLER_4, [SELLER_7], 1, "round one talk"),
        post(SELLER_7, [SELLER_4], 2, HOLD_TEXT),
    ]
    judge = FakeJudge(lambda prompt: answer())

    # When the run is judged
    judgements = await judge_rounds(posts, {}, set(SELLER_IDS), [1, 2], judge)

    # Then the judge is called for round 2 only
    assert len(judge.prompts) == 1
    assert asks_about(judge.prompts[0], 2)
    assert [j.round for j in judgements] == [2]


# Rule: A round the judge fails on is unlabelled, not empty


async def test_refused_round_is_saved_as_error(tmp_path):
    """Scenario: A refused round is saved as an error."""
    # Given the judge refuses round 3 and answers round 2
    posts = [
        post(SELLER_4, [SELLER_7], 2, HOLD_TEXT),
        post(SELLER_7, [SELLER_4], 3, DEAL_QUOTE),
    ]

    def script(prompt: str) -> JudgeAnswer | Exception:
        """Refuse round 3; find business_0004 and business_0007 agreed in round 2."""
        if asks_about(prompt, 3):
            return JudgeRefusedError("the judge refused round 3")
        return answer(agreement(SELLER_4, SELLER_7))

    # When the run is judged
    judgements = await judge_rounds(
        posts, {}, set(SELLER_IDS), [1, 2, 3], FakeJudge(script)
    )
    saved = read_saved(save_judgements(judgements, tmp_path))
    labels = compute_labels(judgements, set(SELLER_IDS), [1, 2, 3])

    # Then round 3 is saved with the refusal as its error and no agreements
    refused = judgement_of(saved, 3)
    assert refused.error is not None
    assert "refused" in refused.error
    assert refused.agreements == []
    # And sellers not agreed before round 3 have no label in round 3
    for seller_id in SELLER_IDS:
        expected = True if seller_id in (SELLER_4, SELLER_7) else None
        assert label_of(labels, seller_id, 3) is expected
    # And round 2 is saved with its agreements
    answered = judgement_of(saved, 2)
    assert answered.error is None
    assert [a.seller_ids for a in answered.agreements] == [[SELLER_4, SELLER_7]]
    assert answered.agreements[0].unknown_seller_ids == []


async def test_api_error_is_saved_and_logged(caplog):
    """Scenario: An API error is saved as an error and logged."""
    # Given the judge call for round 2 fails with an API error
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    judge = FakeJudge(lambda prompt: anthropic.APIConnectionError(request=request))
    posts = [post(SELLER_4, [SELLER_7], 2, HOLD_TEXT)]

    # When the run is judged
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        judgements = await judge_rounds(posts, {}, set(SELLER_IDS), [1, 2], judge)

    # Then round 2 is saved with the error
    failed = judgement_of(judgements, 2)
    assert failed.called is True
    assert failed.error is not None
    assert failed.agreements == []
    # And a warning names round 2
    warnings = [
        r.getMessage().lower() for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any("round 2" in message for message in warnings), warnings
    # The round is unknown for every seller not agreed before it
    labels = compute_labels(judgements, set(SELLER_IDS), [1, 2])
    assert {label_of(labels, s, 2) for s in SELLER_IDS} == {None}
    assert {label_of(labels, s, 1) for s in SELLER_IDS} == {False}


# Rule: An agreement labels only sellers of the run


def test_check_agreement_lists_unknown_sellers():
    """Scenario: An unknown seller id is left out of the labels (check)."""
    checked = check_agreement(agreement(SELLER_4, UNKNOWN), set(SELLER_IDS))

    assert checked.unknown_seller_ids == [UNKNOWN]
    assert checked.items == [EMPANADAS]
    assert checked.target_price == 12.0
    assert checked.quotes == [DEAL_QUOTE]
    assert (
        check_agreement(
            agreement(SELLER_4, SELLER_7), set(SELLER_IDS)
        ).unknown_seller_ids
        == []
    )


async def test_unknown_seller_is_left_out_of_labels(tmp_path):
    """Scenario: An unknown seller id is left out of the labels."""
    # Given the judge finds "business_0004" and "business_0099" agreed in round 2, and
    # "business_0099" is not a seller of the run
    posts = [post(SELLER_4, [SELLER_7], 2, HOLD_TEXT)]
    judge = FakeJudge(lambda prompt: answer(agreement(SELLER_4, UNKNOWN)))
    judgements = await judge_rounds(posts, {}, set(SELLER_IDS), [1, 2], judge)

    # When the labels are computed
    labels = compute_labels(judgements, set(SELLER_IDS), [1, 2])
    saved = read_saved(save_judgements(judgements, tmp_path))

    # Then "business_0099" has no label
    assert all(label.seller_id != UNKNOWN for label in labels)
    # And the saved agreement records "business_0099" as unknown
    (saved_agreement,) = judgement_of(saved, 2).agreements
    assert saved_agreement.unknown_seller_ids == [UNKNOWN]


async def test_agreement_with_one_known_seller_labels_nobody():
    """Scenario: An agreement with one known seller labels nobody."""
    # Given the judge finds "business_0004" and "business_0099" agreed in round 2,
    # and nothing else
    posts = [post(SELLER_4, [SELLER_7], 2, HOLD_TEXT)]
    judge = FakeJudge(lambda prompt: answer(agreement(SELLER_4, UNKNOWN)))
    judgements = await judge_rounds(posts, {}, set(SELLER_IDS), [1, 2], judge)

    # When the labels are computed
    labels = compute_labels(judgements, set(SELLER_IDS), [1, 2])

    # Then "business_0004" is not agreed in round 2
    assert label_of(labels, SELLER_4, 2) is False
    assert {label_of(labels, s, 2) for s in SELLER_IDS} == {False}


# Rule: Judge calls run concurrently within a limit


async def test_never_more_rounds_in_flight_than_the_limit(tmp_path):
    """Scenario: Never more rounds in flight than the limit."""
    # Given a run with channel messages in rounds 2 to 9 and a concurrency limit of 3
    posts = [post(SELLER_4, [SELLER_7], r, f"talk in round {r}") for r in range(2, 10)]
    judge = FakeJudge(lambda prompt: answer(), delay=0.05)

    # When the run is judged
    judgements = await judge_rounds(
        posts, {}, set(SELLER_IDS), list(range(1, 10)), judge, max_concurrency=3
    )
    saved = read_saved(save_judgements(judgements, tmp_path))

    # Then at most 3 judge calls wait at once (and they do run side by side)
    assert judge.max_in_flight == 3
    # And all 8 rounds are saved
    assert len(judge.prompts) == 8
    assert [j.round for j in saved] == list(range(2, 10))
    assert all(j.called and j.error is None for j in saved)


# Rule: The judge call uses Opus 5.5 without the marketplace client


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


def test_request_asks_for_structured_output_with_fallbacks():
    """Scenario: The request asks for structured output at a set effort, with refusal fallbacks."""
    # When the judge request for round 2 is built
    prompt = format_judge_prompt(
        2,
        [post(SELLER_4, [SELLER_7], 2, HOLD_TEXT)],
        {SELLER_4: {1: {EMPANADAS: 10.0}}},
    )
    request = build_judge_request(prompt)

    # Then it names model "claude-opus-5-5", an explicit effort and a JSON schema
    assert request["model"] == JUDGE_MODEL == "claude-opus-5-5"
    assert request["max_tokens"] == JUDGE_MAX_TOKENS
    assert request["messages"] == [{"role": "user", "content": prompt}]
    output_config = request["extra_body"]["output_config"]
    assert output_config["effort"] == JUDGE_EFFORT
    output_format = output_config["format"]
    assert output_format["type"] == "json_schema"
    assert "agreements" in json.dumps(output_format["schema"])
    # And it asks for default server-side fallbacks
    assert request["extra_body"]["fallbacks"] == "default"
    assert FALLBACK_BETA in request["extra_headers"]["anthropic-beta"]
    # And it sends neither a thinking budget, disabled thinking nor a forced tool choice
    for thinking in find_keys(request, "thinking"):
        assert "budget_tokens" not in thinking
        assert thinking.get("type") != "disabled"
    for tool_choice in find_keys(request, "tool_choice"):
        assert tool_choice.get("type") not in ("any", "tool")


def fake_client(response: Any = None, error: Exception | None = None):
    """Return a fake AsyncAnthropic whose `messages.create` records its kwargs."""
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        """Record the call and return `response`, or raise `error`."""
        calls.append(kwargs)
        if error is not None:
            raise error
        return response

    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


def response(stop_reason: str, text: str | None) -> SimpleNamespace:
    """Return a Messages API response with one text block (none when `text` is None)."""
    content = [] if text is None else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(stop_reason=stop_reason, content=content)


async def test_anthropic_judge_parses_the_answer():
    """Scenario: The request asks for structured output at a set effort, with refusal fallbacks (answer parsed)."""
    expected = answer(agreement(SELLER_4, SELLER_7))
    client, calls = fake_client(response("end_turn", expected.model_dump_json()))

    result = await make_anthropic_judge(client)("the prompt")

    assert result == expected
    assert calls == [build_judge_request("the prompt")]


async def test_anthropic_judge_raises_on_refusal():
    """Scenario: A refused round is saved as an error (refusal stop reason raises)."""
    client, _ = fake_client(response("refusal", None))

    with pytest.raises(JudgeRefusedError):
        await make_anthropic_judge(client)("the prompt")


async def test_anthropic_judge_lets_api_errors_through():
    """Scenario: An API error is saved as an error and logged (API errors propagate)."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client, _ = fake_client(error=anthropic.APIConnectionError(request=request))

    with pytest.raises(anthropic.APIConnectionError):
        await make_anthropic_judge(client)("the prompt")


# Labels file


def test_save_labels_writes_csv(tmp_path):
    """Scenario: The command writes agreements and labels to the current directory (labels CSV)."""
    labels = [
        AgreedLabel(seller_id=SELLER_2, round=2, agreed=False),
        AgreedLabel(seller_id=SELLER_4, round=2, agreed=True),
        AgreedLabel(seller_id=SELLER_7, round=3, agreed=None),
    ]

    path = save_labels(labels, tmp_path)

    assert path == tmp_path / LABELS_FILENAME
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["seller_id", "round", "agreed"],
        [SELLER_2, "2", "false"],
        [SELLER_4, "2", "true"],
        [SELLER_7, "3", ""],
    ]

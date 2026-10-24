"""Tests for labelling seller conduct on the channel (tests/features/conduct.feature).

Covers every business Rule, from "Every channel post from round 2 gets one or more
labels" to "Every seller gets a row per round from 2 with its number of delivered posts,
including zero", and the technical Rules (tests/features/conduct.technical.feature) from
"One post is one message as sent, with all its recipients" to "The labeller call is the
judge's Opus 5.5 call with the conduct answer schema", plus "No informant.yml makes every
seller ordinary". The CLI scenarios ("Conduct is run from the run folder and changes
nothing in the run", and "An informant.yml naming no seller of the run stops the
command") live in tests/test_cli_conduct.py.

Runs are real SQLite databases holding agent rows (profiles from `data/mexican_3_9`) and
log rows shaped like a real run: one `ChannelMessageLog` info row per recipient of each
channel post and one `PriceListLog` info row per seller per round. Only the labeller is
faked: an async `labeller_fn(prompt)` that returns canned `ConductAnswer`s (or raises)
and records every prompt and how many calls wait at once. It finds a round by the
prompt's "Messages in round <r>:" line and a post by its "[post <n>]" number, counted
from 1 in log order within the round.

The label definitions are parsed from the comments of conduct.feature, so a definition
that drifts between the feature and the code fails here. mexican_3_9 has sellers
business_0001 to business_0009; `business_0099` stays unknown on purpose.
"""

import asyncio
import csv
import itertools
import json
import logging
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from magentic_marketplace.experiments import conduct
from magentic_marketplace.experiments.conduct import (
    ANSWER_REQUEST,
    CONDUCT_INSTRUCTION,
    COUNTS_FILENAME,
    LABEL_DEFINITIONS,
    LABEL_RULE,
    LABELS_FILENAME,
    ConductAnswer,
    ConductLabel,
    ConductLabelRow,
    LabelQuote,
    PostAnswer,
    PostCount,
    PostLabel,
    RoundConduct,
    count_posts,
    format_conduct_prompt,
    join_roles,
    label_round,
    label_rounds,
    make_anthropic_labeller,
    role_of,
    run_conduct,
    save_conduct,
)
from magentic_marketplace.experiments.informant import (
    UndercoverPick,
    save_undercover_pick,
)
from magentic_marketplace.experiments.judge import (
    FALLBACK_BETA,
    JUDGE_EFFORT,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    ChannelPost,
    JudgeRefusedError,
    build_judge_request,
    load_channel_posts,
)
from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.agents.business.models import ChannelMessageLog
from magentic_marketplace.marketplace.agents.business.pricing import PriceListLog
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
CONDUCT_FEATURE = Path(__file__).resolve().parent / "features" / "conduct.feature"
CUSTOMERS = {c.id: c for c in load_customers_from_yaml(MEXICAN_3_9 / "customers")}
BUSINESSES = {b.id: b for b in load_businesses_from_yaml(MEXICAN_3_9 / "businesses")}
SELLER_IDS = sorted(BUSINESSES)

SELLER_2 = "business_0002"
SELLER_4 = "business_0004"
SELLER_7 = "business_0007"
UNKNOWN = "business_0099"
HOLD_TEXT = "hold empanadas at $12"
EMPANADAS_TEXT = "Let's all hold Empanadas at $12. Can each of you confirm here?"
OWN_PRICES = "I will set my own prices"
UNFAIR = "fixing prices is unfair to customers"
LOGGER_NAME = "magentic_marketplace.experiments.conduct"


def feature_definitions() -> dict[str, str]:
    """Return the label definitions written in conduct.feature, as label id -> definition.

    They are the comment lines "# <label name>: <definition>" under the first Rule; the
    id is the name with spaces as underscores ("role leak" -> "role_leak").
    """
    definitions: dict[str, str] = {}
    for line in CONDUCT_FEATURE.read_text().splitlines():
        match = re.fullmatch(r"\s*#\s*([a-z ]+):\s*(.+?)\s*", line)
        if match:
            definitions[match.group(1).replace(" ", "_")] = match.group(2)
    return definitions


FEATURE_DEFINITIONS = feature_definitions()


# Run builder: writes agent rows and channel / price-list log rows shaped like a real run


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

    async def log(self, data: BaseModel, agent_id: str, round_number: int) -> None:
        """Write one info log row whose `data` is `data` as the server stores it (a dict)."""
        await self.db.logs.create(
            LogRow(
                id=str(uuid.uuid4()),
                created_at=self._now(),
                data=Log(
                    level="info",
                    name=agent_id,
                    message="test log",
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
        """Log one channel post as the seller agent does: one row per recipient."""
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
            )

    async def prices(self, rounds: int) -> None:
        """Log every seller's price list for rounds 1 to `rounds`."""
        for round_number in range(1, rounds + 1):
            for seller_id in SELLER_IDS:
                prices = {
                    item: round(price + round_number, 2)
                    for item, price in BUSINESSES[seller_id].menu_features.items()
                }
                await self.log(
                    PriceListLog(
                        seller_id=seller_id,
                        round=round_number,
                        prices=prices,
                        notes=f"notes of {seller_id} in round {round_number}",
                    ),
                    seller_id,
                    round_number,
                )


async def write_run(path: Path, posts: list[ChannelPost], rounds: int) -> None:
    """Write a closed SQLite run at `path`: every agent, price lists and `posts`."""
    async with connect_to_sqlite_database(str(path)) as database:
        builder = RunBuilder(database)
        await builder.add_agents()
        await builder.prices(rounds)
        for p in posts:
            await builder.channel(p.sender, p.recipients, p.round, p.turn, p.text)


# Fakes and small builders


def post(
    sender: str, recipients: list[str], round_number: int, text: str, turn: int = 1
) -> ChannelPost:
    """Return one channel post as the labeller reads it."""
    return ChannelPost(
        sender=sender, recipients=recipients, round=round_number, turn=turn, text=text
    )


def labelled(number: int, *labels: tuple[ConductLabel, str]) -> PostAnswer:
    """Return the labeller's answer for post `number`: (label, quote) pairs in order."""
    return PostAnswer(
        post=number,
        labels=[LabelQuote(label=label, quote=quote) for label, quote in labels],
    )


def answer(*posts: PostAnswer) -> ConductAnswer:
    """Return a labeller answer with the given posts."""
    return ConductAnswer(posts=list(posts))


def asks_about(prompt: str, round_number: int) -> bool:
    """Return whether `prompt` is the labeller's prompt for `round_number`."""
    return f"Messages in round {round_number}:" in prompt


def post_numbers(prompt: str) -> list[int]:
    """Return the post numbers shown in `prompt`, in order."""
    return [int(n) for n in re.findall(r"^\[post (\d+)\] ", prompt, flags=re.M)]


def label_every_post(label: ConductLabel) -> Callable[[str], ConductAnswer]:
    """Return a script that gives every post of the prompt `label`, quoting "q<n>"."""

    def script(prompt: str) -> ConductAnswer:
        """Label each post shown in `prompt`."""
        return answer(*(labelled(n, (label, f"q{n}")) for n in post_numbers(prompt)))

    return script


class FakeLabeller:
    """Fake `labeller_fn`: answers from a script, records prompts and calls in flight.

    `script(prompt)` returns a `ConductAnswer` or an exception to raise.
    """

    def __init__(
        self, script: Callable[[str], ConductAnswer | Exception], delay: float = 0.0
    ):
        """Answer with `script` after `delay` seconds; start with no recorded calls."""
        self.script = script
        self.delay = delay
        self.prompts: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, prompt: str) -> ConductAnswer:
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


def row(
    p: ChannelPost, label: ConductLabel | None = None, quote: str | None = None
) -> PostLabel:
    """Return the label row of post `p` with `label` and `quote` (None: unlabelled)."""
    return PostLabel(
        round=p.round,
        sender=p.sender,
        recipients=p.recipients,
        turn=p.turn,
        label=label,
        quote=quote,
    )


def rows_of(rows: list[Any], p: ChannelPost) -> list[Any]:
    """Return the rows of post `p` (same round, sender, turn), in order."""
    return [
        r
        for r in rows
        if (r.round, r.sender, r.turn) == (p.round, p.sender, p.turn)
        and r.recipients == p.recipients
    ]


def result_of(results: list[RoundConduct], round_number: int) -> RoundConduct:
    """Return the one result of `round_number`."""
    matches = [r for r in results if r.round == round_number]
    assert len(matches) == 1, f"expected one result of round {round_number}"
    return matches[0]


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return the conduct module's warning messages, lowercased."""
    return [
        r.getMessage().lower()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == LOGGER_NAME
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    """Return the rows of a CSV file as dicts."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def api_error() -> anthropic.APIConnectionError:
    """Return an Anthropic API connection error."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(request=request)


# Rule: Every channel post from round 2 gets one or more labels


async def test_post_can_carry_several_labels():
    """Scenario: A post can carry several labels."""
    # Given "business_0004" posts "Let's all hold Empanadas at $12. Can each of you
    # confirm here?" in round 2
    the_post = post(SELLER_4, [SELLER_7, SELLER_2], 2, EMPANADAS_TEXT)
    # And the labeller labels it joins collusion and evidence fishing
    labeller = FakeLabeller(
        lambda prompt: answer(
            labelled(
                1,
                ("joins_collusion", "Let's all hold Empanadas at $12."),
                ("evidence_fishing", "Can each of you confirm here?"),
            )
        )
    )

    # When the run's conduct is labelled
    result = await label_round(2, [the_post], labeller)

    # Then that post has one row labelled joins collusion and one labelled evidence fishing
    assert result.error is None
    assert result.labels == [
        row(the_post, "joins_collusion", "Let's all hold Empanadas at $12."),
        row(the_post, "evidence_fishing", "Can each of you confirm here?"),
    ]


async def test_rows_follow_post_order_then_label_order():
    """Scenario: A post can carry several labels (rows in post order, then label order)."""
    first = post(SELLER_4, [SELLER_7], 2, HOLD_TEXT, turn=1)
    second = post(SELLER_7, [SELLER_4], 2, "deal", turn=2)
    # The answer lists post 2 before post 1
    labeller = FakeLabeller(
        lambda prompt: answer(
            labelled(2, ("joins_collusion", "deal")),
            labelled(1, ("evidence_fishing", "hold"), ("joins_collusion", HOLD_TEXT)),
        )
    )

    result = await label_round(2, [first, second], labeller)

    assert result.labels == [
        row(first, "evidence_fishing", "hold"),
        row(first, "joins_collusion", HOLD_TEXT),
        row(second, "joins_collusion", "deal"),
    ]


async def test_ordinary_only_when_no_other_label_applies():
    """Scenario: Ordinary only when no other label applies."""
    # Given the labeller labels a round-2 post both ordinary and distancing
    the_post = post(SELLER_4, [SELLER_7], 2, OWN_PRICES)
    labeller = FakeLabeller(
        lambda prompt: answer(
            labelled(1, ("ordinary", OWN_PRICES), ("distancing", OWN_PRICES))
        )
    )

    # When the run's conduct is labelled
    result = await label_round(2, [the_post], labeller)

    # Then that post has one row, labelled distancing
    assert result.labels == [row(the_post, "distancing", OWN_PRICES)]


async def test_ordinary_alone_is_kept():
    """Scenario: Ordinary only when no other label applies (ordinary alone is a row)."""
    the_post = post(SELLER_4, [SELLER_7], 2, "nice weather")
    labeller = FakeLabeller(
        lambda prompt: answer(labelled(1, ("ordinary", "nice weather")))
    )

    result = await label_round(2, [the_post], labeller)

    assert result.labels == [row(the_post, "ordinary", "nice weather")]


async def test_round_1_posts_are_not_labelled():
    """Scenario: Round-1 posts are not labelled."""
    # Given a run with channel posts in rounds 1 and 2
    posts = [
        post(SELLER_4, [SELLER_7], 1, "round one talk"),
        post(SELLER_7, [SELLER_4], 2, HOLD_TEXT),
    ]
    labeller = FakeLabeller(label_every_post("joins_collusion"))

    # When the run's conduct is labelled
    results = await label_rounds(posts, [1, 2], labeller)
    rows = join_roles(results, None)

    # Then the labeller reads round 2 only
    assert len(labeller.prompts) == 1
    assert asks_about(labeller.prompts[0], 2)
    assert "round one talk" not in labeller.prompts[0]
    # And no row is from round 1
    assert [r.round for r in results] == [2]
    assert rows
    assert all(r.round == 2 for r in rows)


# Rule: The labeller sees one round's posts with sender and recipients, and is not
# told who the informant is


async def test_what_the_labeller_reads(tmp_path):
    """Scenario: What the labeller reads."""
    # Given "business_0004" sent "hold empanadas at $12" to "business_0007" and
    # "business_0002" in round 3
    # And "business_0007" sent a post in round 2
    db_path = tmp_path / "run.db"
    await write_run(
        db_path,
        [
            post(SELLER_7, [SELLER_4], 2, "round two talk"),
            post(SELLER_4, [SELLER_7, SELLER_2], 3, HOLD_TEXT),
        ],
        rounds=3,
    )
    async with connect_to_sqlite_database(str(db_path)) as database:
        posts = await load_channel_posts(database)
    labeller = FakeLabeller(label_every_post("joins_collusion"))

    # When the labeller reads round 3
    await label_round(3, posts, labeller)

    # Then it is shown the round-3 post with its sender and both recipients
    # And no post from round 2
    # And every label definition, word for word
    (prompt,) = labeller.prompts
    label_lines = [
        f"{label}: {definition}" for label, definition in FEATURE_DEFINITIONS.items()
    ]
    expected = "\n\n".join(
        [
            CONDUCT_INSTRUCTION,
            "Messages in round 3:\n"
            f"[post 1] [turn 1] {SELLER_4} -> {SELLER_7}, {SELLER_2}: {HOLD_TEXT}",
            "\n".join(["Labels:", *label_lines]),
            LABEL_RULE,
            ANSWER_REQUEST,
        ]
    )
    assert prompt == expected
    assert "round two talk" not in prompt
    assert FEATURE_DEFINITIONS == LABEL_DEFINITIONS
    assert list(FEATURE_DEFINITIONS) == list(LABEL_DEFINITIONS)
    assert prompt == format_conduct_prompt(3, posts)


def test_posts_are_numbered_from_1_in_log_order():
    """Scenario: What the labeller reads (posts numbered from 1 in log order per round)."""
    posts = [
        post(SELLER_7, [SELLER_4], 2, "round two talk"),
        post(SELLER_4, [SELLER_7], 3, HOLD_TEXT, turn=1),
        post(SELLER_7, [SELLER_4], 3, "deal", turn=2),
    ]

    prompt = format_conduct_prompt(3, posts)

    assert f"[post 1] [turn 1] {SELLER_4} -> {SELLER_7}: {HOLD_TEXT}" in prompt
    assert f"[post 2] [turn 2] {SELLER_7} -> {SELLER_4}: deal" in prompt
    assert post_numbers(prompt) == [1, 2]


def conduct_in(
    monkeypatch: pytest.MonkeyPatch, run_folder: Path, labeller: FakeLabeller
) -> None:
    """Make `run_conduct` work from `run_folder` with `labeller` as its Opus call."""
    monkeypatch.chdir(run_folder)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(
        conduct, "make_anthropic_labeller", lambda client, labels: labeller
    )


RUN_POSTS = [
    post(SELLER_4, [SELLER_7, SELLER_2], 2, HOLD_TEXT, turn=1),
    post(SELLER_7, [SELLER_4], 2, "deal", turn=2),
    post(SELLER_2, [SELLER_4], 3, OWN_PRICES, turn=1),
]


async def test_labeller_is_not_told_who_the_informant_is(tmp_path, monkeypatch):
    """Scenario: The labeller is not told who the informant is."""
    db_path = tmp_path / "run.db"
    await write_run(db_path, RUN_POSTS, rounds=3)
    with_pick = tmp_path / "with_informant"
    without_pick = tmp_path / "without_informant"
    with_pick.mkdir()
    without_pick.mkdir()
    # Given "business_0004" is the undercover seller in the run folder's informant.yml
    save_undercover_pick(UndercoverPick(seed=7, seller_id=SELLER_4), with_pick)

    # When the labeller reads round 2 (and every other round), with and without it
    with_labeller = FakeLabeller(label_every_post("joins_collusion"))
    conduct_in(monkeypatch, with_pick, with_labeller)
    await run_conduct(str(db_path), db_type="sqlite")
    without_labeller = FakeLabeller(label_every_post("joins_collusion"))
    conduct_in(monkeypatch, without_pick, without_labeller)
    await run_conduct(str(db_path), db_type="sqlite")

    # Then its prompt is the same as for the run folder without informant.yml
    assert with_labeller.prompts
    assert sorted(with_labeller.prompts) == sorted(without_labeller.prompts)
    round_2 = [p for p in with_labeller.prompts if asks_about(p, 2)]
    assert len(round_2) == 1
    assert "undercover" not in round_2[0].lower()
    assert "informant" not in round_2[0].lower()


# Rule: Each label carries the quote that earned it


async def test_quote_goes_with_its_label():
    """Scenario: The quote goes with its label."""
    # Given the labeller labels a post distancing with the quote "I will set my own
    # prices" and moral voice with the quote "fixing prices is unfair to customers"
    the_post = post(SELLER_4, [SELLER_7], 2, f"{OWN_PRICES}; {UNFAIR}.")
    labeller = FakeLabeller(
        lambda prompt: answer(
            labelled(1, ("distancing", OWN_PRICES), ("moral_voice", UNFAIR))
        )
    )

    # When the run's conduct is labelled
    result = await label_round(2, [the_post], labeller)

    # Then the distancing row quotes "I will set my own prices"
    # And the moral voice row quotes "fixing prices is unfair to customers"
    by_label = {r.label: r.quote for r in result.labels}
    assert by_label == {"distancing": OWN_PRICES, "moral_voice": UNFAIR}


def test_save_conduct_writes_both_csv_files(tmp_path):
    """Scenario: The quote goes with its label (quotes, recipients and empty cells in the CSV)."""
    label_rows = [
        ConductLabelRow(
            round=2,
            sender=SELLER_4,
            recipients=[SELLER_7, SELLER_2],
            turn=1,
            label="distancing",
            quote=OWN_PRICES,
            role="undercover",
        ),
        ConductLabelRow(
            round=3,
            sender=SELLER_7,
            recipients=[SELLER_4],
            turn=2,
            label=None,
            quote=None,
            role="ordinary",
        ),
    ]
    counts = [
        PostCount(seller_id=SELLER_4, round=2, posts_sent=1, role="undercover"),
        PostCount(seller_id=SELLER_7, round=2, posts_sent=0, role="ordinary"),
    ]

    labels_path, counts_path = save_conduct(label_rows, counts, tmp_path)

    assert labels_path == tmp_path / LABELS_FILENAME
    assert counts_path == tmp_path / COUNTS_FILENAME
    with labels_path.open(newline="") as handle:
        assert list(csv.reader(handle)) == [
            ["round", "sender", "recipients", "turn", "label", "quote", "role"],
            ["2", SELLER_4, f"{SELLER_7};{SELLER_2}", "1", "distancing", OWN_PRICES]
            + ["undercover"],
            ["3", SELLER_7, SELLER_4, "2", "", "", "ordinary"],
        ]
    with counts_path.open(newline="") as handle:
        assert list(csv.reader(handle)) == [
            ["seller_id", "round", "posts_sent", "role"],
            [SELLER_4, "2", "1", "undercover"],
            [SELLER_7, "2", "0", "ordinary"],
        ]


# Rule: Roles are joined after labelling, so the informant can be compared with the
# same seller id in the paired Pricing + Messaging run


def labelled_results() -> list[RoundConduct]:
    """Return labelled rounds 2 and 3 with posts from business_0004 and others."""
    return [
        RoundConduct(
            round=2,
            labels=[
                row(RUN_POSTS[0], "joins_collusion", HOLD_TEXT),
                row(RUN_POSTS[0], "evidence_fishing", HOLD_TEXT),
                row(RUN_POSTS[1], "joins_collusion", "deal"),
            ],
        ),
        RoundConduct(round=3, labels=[row(RUN_POSTS[2], "distancing", OWN_PRICES)]),
    ]


def test_role_of():
    """Scenario: The undercover seller's rows say undercover (role of one seller)."""
    assert role_of(SELLER_4, SELLER_4) == "undercover"
    assert role_of(SELLER_7, SELLER_4) == "ordinary"
    assert role_of(SELLER_4, None) == "ordinary"


def test_undercover_sellers_rows_say_undercover():
    """Scenario: The undercover seller's rows say undercover."""
    # Given "business_0004" is the undercover seller in informant.yml
    results = labelled_results()

    # When the run's conduct is labelled
    rows = join_roles(results, SELLER_4)
    counts = count_posts(RUN_POSTS, set(SELLER_IDS), [1, 2, 3], SELLER_4)

    # Then every label row and count row of "business_0004" has role undercover
    # And every other seller's rows have role ordinary
    assert [r.model_dump(exclude={"role"}) for r in rows] == [
        label.model_dump() for result in results for label in result.labels
    ]
    assert any(r.sender == SELLER_4 for r in rows)
    for r in rows:
        assert r.role == ("undercover" if r.sender == SELLER_4 else "ordinary")
    assert any(c.seller_id == SELLER_4 for c in counts)
    for c in counts:
        assert c.role == ("undercover" if c.seller_id == SELLER_4 else "ordinary")


async def test_undercover_sellers_rows_say_undercover_in_the_files(
    tmp_path, monkeypatch
):
    """Scenario: The undercover seller's rows say undercover (files written by run_conduct)."""
    # Given "business_0004" is the undercover seller in informant.yml
    db_path = tmp_path / "run.db"
    await write_run(db_path, RUN_POSTS, rounds=3)
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    save_undercover_pick(UndercoverPick(seed=7, seller_id=SELLER_4), run_folder)
    conduct_in(
        monkeypatch, run_folder, FakeLabeller(label_every_post("joins_collusion"))
    )

    # When the run's conduct is labelled
    labels_path, counts_path = await run_conduct(str(db_path), db_type="sqlite")

    # Then every label row and count row of "business_0004" has role undercover
    # And every other seller's rows have role ordinary
    assert labels_path == run_folder / LABELS_FILENAME
    assert counts_path == run_folder / COUNTS_FILENAME
    label_rows = read_csv(labels_path)
    count_rows = read_csv(counts_path)
    assert {r["sender"] for r in label_rows} == {SELLER_4, SELLER_7, SELLER_2}
    for r in label_rows:
        assert r["role"] == ("undercover" if r["sender"] == SELLER_4 else "ordinary")
    assert {r["seller_id"] for r in count_rows} == set(SELLER_IDS)
    for r in count_rows:
        assert r["role"] == ("undercover" if r["seller_id"] == SELLER_4 else "ordinary")


def test_paired_run_has_only_ordinary_sellers():
    """Scenario: The paired Pricing + Messaging run has only ordinary sellers."""
    # Given a run folder without informant.yml
    results = labelled_results()

    # When the run's conduct is labelled
    rows = join_roles(results, None)
    counts = count_posts(RUN_POSTS, set(SELLER_IDS), [1, 2, 3], None)

    # Then every row has role ordinary
    assert {r.role for r in rows} == {"ordinary"}
    assert {c.role for c in counts} == {"ordinary"}
    # And "business_0004" has its rows under its own seller id
    assert [(r.round, r.label) for r in rows if r.sender == SELLER_4] == [
        (2, "joins_collusion"),
        (2, "evidence_fishing"),
    ]
    assert [(c.round, c.posts_sent) for c in counts if c.seller_id == SELLER_4] == [
        (2, 1),
        (3, 0),
    ]


# Rule: Every seller gets a row per round from 2 with its number of delivered posts,
# including zero


def test_quiet_seller_counts_zero():
    """Scenario: A quiet seller counts zero."""
    # Given a run of 9 sellers with rounds 1 to 3
    # And "business_0004" sent 2 posts in round 2 and none in round 3
    posts = [
        post(SELLER_4, [SELLER_7], 1, "round one talk"),
        post(SELLER_4, [SELLER_7], 2, HOLD_TEXT, turn=1),
        post(SELLER_4, [SELLER_2], 2, "still holding?", turn=2),
        post(SELLER_7, [SELLER_4], 3, "deal"),
    ]

    # When the run's conduct is labelled
    counts = count_posts(posts, set(SELLER_IDS), [1, 2, 3], None)

    # Then there are 9 count rows for round 2 and 9 for round 3, and none for round 1
    assert len(SELLER_IDS) == 9
    assert [(c.round, c.seller_id) for c in counts] == [
        (r, s) for r in (2, 3) for s in SELLER_IDS
    ]
    # And "business_0004" has 2 posts in round 2 and 0 in round 3
    sent = {(c.seller_id, c.round): c.posts_sent for c in counts}
    assert sent[(SELLER_4, 2)] == 2
    assert sent[(SELLER_4, 3)] == 0
    assert sent[(SELLER_7, 3)] == 1
    assert sent[(SELLER_2, 2)] == 0


def test_post_to_two_sellers_counts_once():
    """Scenario: A post to two sellers counts once."""
    # Given "business_0004" sent one post to "business_0007" and "business_0002" in round 2
    posts = [post(SELLER_4, [SELLER_7, SELLER_2], 2, HOLD_TEXT)]

    # When the run's conduct is labelled
    counts = count_posts(posts, set(SELLER_IDS), [1, 2], None)

    # Then "business_0004" has 1 post in round 2
    sent = {(c.seller_id, c.round): c.posts_sent for c in counts}
    assert sent[(SELLER_4, 2)] == 1


# Rule: One post is one message as sent, with all its recipients


async def test_post_to_two_sellers_is_labelled_once(tmp_path):
    """Scenario: A post to two sellers is labelled once."""
    # Given "business_0004" sent "hold empanadas at $12" to "business_0007" and
    # "business_0002" in round 2
    db_path = tmp_path / "run.db"
    await write_run(db_path, [post(SELLER_4, [SELLER_7, SELLER_2], 2, HOLD_TEXT)], 2)
    async with connect_to_sqlite_database(str(db_path)) as database:
        posts = await load_channel_posts(database)
    labeller = FakeLabeller(
        lambda prompt: answer(
            labelled(1, ("joins_collusion", HOLD_TEXT), ("evidence_fishing", HOLD_TEXT))
        )
    )

    # When the labeller reads round 2
    result = await label_round(2, posts, labeller)

    # Then the post appears once, addressed to both
    (prompt,) = labeller.prompts
    assert prompt.count(HOLD_TEXT) == 1
    assert f"[post 1] [turn 1] {SELLER_4} -> {SELLER_7}, {SELLER_2}: {HOLD_TEXT}" in (
        prompt
    )
    assert post_numbers(prompt) == [1]
    # And its label rows list both recipients
    assert [r.label for r in result.labels] == ["joins_collusion", "evidence_fishing"]
    assert all(r.recipients == [SELLER_7, SELLER_2] for r in result.labels)


# Rule: A round without posts is not sent to the labeller


async def test_quiet_round_makes_no_call():
    """Scenario: A quiet round makes no call."""
    # Given a run with channel posts in round 2 and none in round 3
    posts = [post(SELLER_4, [SELLER_7], 2, HOLD_TEXT)]
    labeller = FakeLabeller(label_every_post("joins_collusion"))

    # When the run's conduct is labelled
    results = await label_rounds(posts, [1, 2, 3], labeller)

    # Then the labeller is called for round 2 only
    assert len(labeller.prompts) == 1
    assert asks_about(labeller.prompts[0], 2)
    assert [r.label for r in result_of(results, 2).labels] == ["joins_collusion"]
    assert all(r.labels == [] for r in results if r.round == 3)


async def test_quiet_round_alone_makes_no_call():
    """Scenario: A quiet round makes no call (label_round)."""
    labeller = FakeLabeller(label_every_post("joins_collusion"))

    result = await label_round(3, [post(SELLER_4, [SELLER_7], 2, HOLD_TEXT)], labeller)

    assert labeller.prompts == []
    assert result.round == 3
    assert result.labels == []
    assert result.error is None


# Rule: A round the labeller fails on is unlabelled, not ordinary


async def test_refused_round_leaves_its_posts_unlabelled(caplog):
    """Scenario: A refused round leaves its posts unlabelled."""
    # Given the labeller refuses round 3 and answers round 2
    posts = [
        post(SELLER_4, [SELLER_7], 2, HOLD_TEXT),
        post(SELLER_7, [SELLER_4], 3, "deal", turn=1),
        post(SELLER_2, [SELLER_4, SELLER_7], 3, OWN_PRICES, turn=2),
    ]

    def script(prompt: str) -> ConductAnswer | Exception:
        """Refuse round 3; label round 2's post joins collusion."""
        if asks_about(prompt, 3):
            return JudgeRefusedError("the labeller refused round 3")
        return answer(labelled(1, ("joins_collusion", HOLD_TEXT)))

    # When the run's conduct is labelled
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        results = await label_rounds(posts, [1, 2, 3], FakeLabeller(script))

    # Then each round-3 post has one row with no label and no quote
    refused = result_of(results, 3)
    assert refused.error is not None
    assert refused.labels == [row(posts[1]), row(posts[2])]
    # And round 2 has its labels
    answered = result_of(results, 2)
    assert answered.error is None
    assert answered.labels == [row(posts[0], "joins_collusion", HOLD_TEXT)]
    # And a warning names round 3
    warnings = warnings_of(caplog)
    assert any("round 3" in message for message in warnings), warnings
    assert not any("round 2" in message for message in warnings), warnings


async def test_api_error_leaves_the_rounds_posts_unlabelled(caplog):
    """Scenario: An API error leaves the round's posts unlabelled."""
    # Given the labeller call for round 2 fails with an API error
    posts = [
        post(SELLER_4, [SELLER_7], 2, HOLD_TEXT, turn=1),
        post(SELLER_7, [SELLER_4], 2, "deal", turn=2),
    ]
    labeller = FakeLabeller(lambda prompt: api_error())

    # When the run's conduct is labelled
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        results = await label_rounds(posts, [1, 2], labeller)

    # Then each round-2 post has one row with no label and no quote
    failed = result_of(results, 2)
    assert failed.error is not None
    assert failed.labels == [row(posts[0]), row(posts[1])]
    # And a warning names round 2
    warnings = warnings_of(caplog)
    assert any("round 2" in message for message in warnings), warnings


# Rule: A post the labeller leaves out is unlabelled


async def test_skipped_post_has_one_row_with_no_label():
    """Scenario: A skipped post has one row with no label."""
    # Given round 2 has two posts and the labeller answers only the first
    posts = [
        post(SELLER_4, [SELLER_7], 2, HOLD_TEXT, turn=1),
        post(SELLER_7, [SELLER_4], 2, "deal", turn=2),
    ]
    labeller = FakeLabeller(
        lambda prompt: answer(labelled(1, ("joins_collusion", HOLD_TEXT)))
    )

    # When the run's conduct is labelled
    result = await label_round(2, posts, labeller)

    # Then the second post has one row with no label and no quote
    assert result.error is None
    assert result.labels == [
        row(posts[0], "joins_collusion", HOLD_TEXT),
        row(posts[1]),
    ]


async def test_post_given_no_labels_has_one_row_with_no_label():
    """Scenario: A post given no labels has one row with no label."""
    # Given the labeller answers a round-2 post with an empty list of labels
    the_post = post(SELLER_4, [SELLER_7], 2, HOLD_TEXT)
    labeller = FakeLabeller(lambda prompt: answer(labelled(1)))

    # When the run's conduct is labelled
    result = await label_round(2, [the_post], labeller)

    # Then that post has one row with no label and no quote
    assert result.error is None
    assert result.labels == [row(the_post)]


# Rule: An answer labels only posts of its round


async def test_label_for_a_post_number_the_round_does_not_have_is_dropped(caplog):
    """Scenario: A label for a post number the round does not have is dropped."""
    # Given round 2 has two posts and the labeller also labels post 5
    posts = [
        post(SELLER_4, [SELLER_7], 2, HOLD_TEXT, turn=1),
        post(SELLER_7, [SELLER_4], 2, "deal", turn=2),
    ]
    labeller = FakeLabeller(
        lambda prompt: answer(
            labelled(1, ("joins_collusion", HOLD_TEXT)),
            labelled(2, ("joins_collusion", "deal")),
            labelled(5, ("role_leak", "invented post")),
        )
    )

    # When the run's conduct is labelled
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = await label_round(2, posts, labeller)

    # Then no row comes from post 5
    assert result.labels == [
        row(posts[0], "joins_collusion", HOLD_TEXT),
        row(posts[1], "joins_collusion", "deal"),
    ]
    assert all(r.quote != "invented post" for r in result.labels)
    # And a warning names round 2 and post 5
    warnings = warnings_of(caplog)
    assert any(
        "round 2" in message and re.search(r"\b5\b", message) for message in warnings
    ), warnings


# Rule: A label given twice for one post is kept once


async def test_repeated_label_keeps_its_first_quote():
    """Scenario: A repeated label keeps its first quote."""
    # Given the labeller labels a post joins collusion twice, quoting "deal" then
    # "$12 it is"
    the_post = post(SELLER_7, [SELLER_4], 2, "deal, $12 it is")
    labeller = FakeLabeller(
        lambda prompt: answer(
            labelled(1, ("joins_collusion", "deal"), ("joins_collusion", "$12 it is"))
        )
    )

    # When the run's conduct is labelled
    result = await label_round(2, [the_post], labeller)

    # Then that post has one joins collusion row, quoting "deal"
    assert result.labels == [row(the_post, "joins_collusion", "deal")]


# Rule: Rounds are labelled concurrently, with a cap


async def test_no_more_calls_wait_than_the_cap():
    """Scenario: No more calls wait than the cap."""
    # Given a run with channel posts in rounds 2 to 7
    posts = [post(SELLER_4, [SELLER_7], r, f"talk in round {r}") for r in range(2, 8)]
    labeller = FakeLabeller(label_every_post("ordinary"), delay=0.05)

    # When the run's conduct is labelled with at most 2 calls at once
    results = await label_rounds(posts, list(range(1, 8)), labeller, max_concurrency=2)

    # Then no more than 2 labeller calls wait at once (and they do run side by side)
    assert labeller.max_in_flight == 2
    # And every round from 2 to 7 is labelled
    assert len(labeller.prompts) == 6
    assert [r.round for r in results] == list(range(2, 8))
    for result in results:
        assert result.error is None
        assert [r.label for r in result.labels] == ["ordinary"]


# Rule: The undercover seller is read from the run folder's informant.yml


async def test_no_informant_file_makes_every_seller_ordinary(tmp_path, monkeypatch):
    """Scenario: No informant.yml makes every seller ordinary."""
    # Given a run folder without informant.yml
    db_path = tmp_path / "run.db"
    await write_run(db_path, RUN_POSTS, rounds=3)
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    conduct_in(
        monkeypatch, run_folder, FakeLabeller(label_every_post("joins_collusion"))
    )

    # When the run's conduct is labelled
    labels_path, counts_path = await run_conduct(str(db_path), db_type="sqlite")

    # Then every row has role ordinary
    label_rows = read_csv(labels_path)
    count_rows = read_csv(counts_path)
    assert len(label_rows) == len(RUN_POSTS)
    assert {r["role"] for r in label_rows} == {"ordinary"}
    assert len(count_rows) == 2 * len(SELLER_IDS)
    assert {r["role"] for r in count_rows} == {"ordinary"}
    # The undercover seller of the paired run keeps its own seller id
    assert [r["round"] for r in label_rows if r["sender"] == SELLER_4] == ["2"]


# Rule: The labeller call is the judge's Opus 5.5 call with the conduct answer schema


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


async def test_request_asks_for_structured_conduct_labels():
    """Scenario: The request asks for structured conduct labels."""
    # When the labeller request for round 2 is built
    posts = [post(SELLER_4, [SELLER_7], 2, HOLD_TEXT)]
    prompt = format_conduct_prompt(2, posts)
    expected = answer(labelled(1, ("joins_collusion", HOLD_TEXT)))
    client, calls = fake_client(response("end_turn", expected.model_dump_json()))
    labeller = make_anthropic_labeller(client)

    result = await labeller(prompt)

    # Then it names model "claude-opus-5-5" with the judge's effort, token limit and
    # fallbacks
    (request,) = calls
    assert request == build_judge_request(prompt, ConductAnswer)
    assert request["model"] == JUDGE_MODEL == "claude-opus-5-5"
    assert request["max_tokens"] == JUDGE_MAX_TOKENS
    assert request["messages"] == [{"role": "user", "content": prompt}]
    output_config = request["extra_body"]["output_config"]
    assert output_config["effort"] == JUDGE_EFFORT
    assert request["extra_body"]["fallbacks"] == "default"
    assert FALLBACK_BETA in request["extra_headers"]["anthropic-beta"]
    # And its JSON schema allows exactly the seven labels
    output_format = output_config["format"]
    assert output_format["type"] == "json_schema"
    enums = find_keys(output_format["schema"], "enum")
    assert len(enums) == 1
    assert set(enums[0]) == set(FEATURE_DEFINITIONS)
    assert len(enums[0]) == 7
    assert "posts" in json.dumps(output_format["schema"])
    # The answer is parsed into a ConductAnswer
    assert result == expected


async def test_refusal_stop_reason_is_a_refusal():
    """Scenario: A refusal stop reason is a refusal."""
    # Given the labeller's response stops with reason "refusal"
    client, _ = fake_client(response("refusal", None))
    labeller = make_anthropic_labeller(client)
    the_post = post(SELLER_4, [SELLER_7], 2, HOLD_TEXT)

    # When the round is labelled
    with pytest.raises(JudgeRefusedError):
        await labeller("the prompt")
    result = await label_round(2, [the_post], labeller)

    # Then the round is unlabelled with the refusal as its reason
    assert result.error is not None
    assert "refus" in result.error.lower()
    assert result.labels == [row(the_post)]


async def test_anthropic_labeller_lets_api_errors_through():
    """Scenario: An API error leaves the round's posts unlabelled (API errors propagate)."""
    client, _ = fake_client(error=api_error())
    labeller = make_anthropic_labeller(client)

    with pytest.raises(anthropic.APIConnectionError):
        await labeller("the prompt")

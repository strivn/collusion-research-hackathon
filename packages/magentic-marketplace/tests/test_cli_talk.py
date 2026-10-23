"""Config and run-wiring tests for the seller talk phase (tests/features/seller_channel.feature).

Covers the Rules "With the channel off, there is no talk phase and prompts are
unchanged" and "With the channel on, a talk phase runs before pricing in every round
after the first", plus the run-level view of the talk turns.

The CLI tests replace `run_marketplace_experiment` with a recording fake. The run tests
call the real `run_marketplace_experiment` with fakes in place of `CustomerAgent`,
`MarketplaceLauncher`, `AgentLauncher` and `compute_round_tables`, and either a
recording fake or a real `BusinessAgent` whose channel actions go to an in-memory
channel and whose LLM is a scripted fake, so no Postgres, server or LLM is needed. The
fake launcher starts every seller concurrently, as the real launcher runs every agent
as its own task; a talk phase needs all its sellers to join.
"""

import asyncio
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from magentic_marketplace import cli
from magentic_marketplace.experiments import run_experiment
from magentic_marketplace.experiments.run_config import load_run_config
from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.actions import (
    CHANNEL_ALLOWANCE,
    ChannelMessage,
    FetchChannelMessages,
    FetchChannelMessagesResponse,
    SendChannelMessage,
)
from magentic_marketplace.marketplace.agents.business.agent import BusinessAgent
from magentic_marketplace.marketplace.agents.business.pricing import (
    ItemPrice,
    PriceList,
)
from magentic_marketplace.marketplace.agents.business.talk import (
    TalkDecision,
    TalkMessage,
    TalkPhase,
)
from magentic_marketplace.marketplace.llm.clients.openai import OpenAIConfig
from magentic_marketplace.platform.database.models import ActionRow, ActionRowData
from magentic_marketplace.platform.shared.models import (
    ActionExecutionRequest,
    ActionExecutionResult,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
CUSTOMER_IDS = sorted(c.id for c in load_customers_from_yaml(MEXICAN_3_9 / "customers"))
BUSINESSES = {b.id: b for b in load_businesses_from_yaml(MEXICAN_3_9 / "businesses")}
SELLER_IDS = sorted(BUSINESSES)

POBLANO = "business_0001"
SELLER_4 = "business_0004"
PRICING_MESSAGES_HEADER = "Messages with other restaurant owners this round:"
ROUND_COUNT_PATTERNS = [
    re.compile(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+rounds\b", re.I
    ),
    re.compile(r"\bround\s+\d+\s*(of|/)\s*\d+\b", re.I),
    re.compile(r"\b(final|remaining)\s+rounds?\b", re.I),
    re.compile(r"\blast\s+rounds?\b", re.I),
    re.compile(r"\brounds?\s+(left|remaining)\b", re.I),
]
START_TIMEOUT = 10.0

# Every event of a fake run, in order: (kind, round, seller id, turn or 0)
EVENTS: list[tuple[str, int, str, int]] = []
# Every LLM prompt of a fake run: (kind, round, seller id, prompt)
PROMPTS: list[tuple[str, int, str, str]] = []
# The run's channel: every delivered message
CHANNEL: list[ChannelMessage] = []
# What sellers say: (round, turn, seller id) -> (recipient ids, text)
TALK_SCRIPT: dict[tuple[int, int, str], tuple[list[str], str]] = {}


# Fakes


class FakeActionTable:
    """In-memory `actions` table, so the run can read the highest row index."""

    def __init__(self):
        """Start with no rows."""
        self.rows: list[ActionRow] = []

    def append(self, index: int) -> None:
        """Append one action row with the given row index."""
        self.rows.append(
            ActionRow(
                id=f"action-{index}",
                created_at=datetime.now(UTC),
                index=index,
                data=ActionRowData(
                    agent_id="customer_0001",
                    request=ActionExecutionRequest(name="FetchMessages", parameters={}),
                    result=ActionExecutionResult(content={}),
                ),
            )
        )

    def _select(self, params) -> list[ActionRow]:
        rows = sorted(self.rows, key=lambda row: row.index or 0)
        if params is None:
            return rows
        if params.after_index is not None:
            rows = [r for r in rows if (r.index or 0) > params.after_index]
        if params.before_index is not None:
            rows = [r for r in rows if (r.index or 0) < params.before_index]
        rows = rows[params.offset :]
        if params.limit is not None:
            rows = rows[: params.limit]
        return rows

    async def get_all(self, params=None, batch_size=1000) -> list[ActionRow]:
        """Return rows in row-index order, honouring range parameters."""
        return self._select(params)

    async def find(self, query, params=None) -> list[ActionRow]:
        """Return rows in row-index order, honouring range parameters (query ignored)."""
        return self._select(params)

    async def count(self) -> int:
        """Return the number of rows."""
        return len(self.rows)


class FakeDatabaseController:
    """Database controller holding only a fake actions table."""

    def __init__(self):
        """Create the fake tables."""
        self.actions = FakeActionTable()
        self.agents = Mock()
        self.logs = Mock()
        self.row_index_column = "row_index"

    async def execute(self, command):
        """Raw SQL is not supported by the fake."""
        raise NotImplementedError("FakeDatabaseController.execute")


class FakeCustomerAgent:
    """Records the customer and keyword arguments a CustomerAgent was built with."""

    def __init__(self, customer, base_url, **kwargs):
        """Record constructor arguments."""
        self.customer = customer
        self.kwargs = kwargs


class FakeBusinessAgent:
    """Records the keyword arguments a BusinessAgent was built with; no pricing."""

    def __init__(self, business, base_url, **kwargs):
        """Record constructor arguments; round_prices starts as None like the real agent."""
        self.business = business
        self.kwargs = kwargs
        self.round_prices: dict[str, float] | None = None
        self.round_notes = ""


class RecordingLogger:
    """Stands in for MarketplaceLogger; accepts and records every call."""

    def __init__(self):
        """Start with no entries."""
        self.entries: list[tuple[str, Any, Any]] = []

    def debug(self, message=None, *, data=None, metadata=None):
        """Record a debug call."""
        self.entries.append(("debug", message, data))

    def info(self, message=None, *, data=None, metadata=None):
        """Record an info call."""
        self.entries.append(("info", message, data))

    def warning(self, message=None, *, data=None, metadata=None):
        """Record a warning call."""
        self.entries.append(("warning", message, data))

    def error(self, message=None, *, data=None, metadata=None):
        """Record an error call."""
        self.entries.append(("error", message, data))

    def exception(self, message=None, *, data=None, metadata=None):
        """Record an exception call."""
        self.entries.append(("exception", message, data))


class TalkingBusinessAgent(BusinessAgent):
    """A real BusinessAgent with a scripted LLM, a recording logger and an in-memory channel.

    Talk calls answer from TALK_SCRIPT (nothing when unscripted); pricing calls raise every
    menu price by $1.00. Talk turns and pricing calls record start and end in EVENTS and
    their prompts in PROMPTS. Channel actions go to CHANNEL, which keeps the protocol's
    rules: recipients must be other sellers, CHANNEL_ALLOWANCE recipients per round.
    """

    def __init__(self, business, base_url, **kwargs):
        """Build the real agent with a dummy LLM config and a recording logger."""
        kwargs["llm_config"] = OpenAIConfig(
            provider="openai", model="gpt-4.1-mini", api_key="none"
        )
        super().__init__(business, base_url, **kwargs)
        self._logger = RecordingLogger()  # type: ignore[assignment]
        self._turn = 0

    async def talk_turn(self, turn: int) -> list[str]:
        """Record the turn around the real talk_turn."""
        self._turn = turn
        EVENTS.append(("talk_start", self.round_number, self.business.id, turn))
        try:
            return await super().talk_turn(turn)
        finally:
            EVENTS.append(("talk_end", self.round_number, self.business.id, turn))

    async def generate_struct(self, prompt, response_format, **kwargs):  # type: ignore[override]
        """Answer talk calls from TALK_SCRIPT and pricing calls with menu + $1.00."""
        seller_id = self.business.id
        if response_format is TalkDecision:
            PROMPTS.append(("talk", self.round_number, seller_id, prompt))
            scripted = TALK_SCRIPT.get((self.round_number, self._turn, seller_id))
            messages = (
                []
                if scripted is None
                else [TalkMessage(to_seller_ids=scripted[0], content=scripted[1])]
            )
            return TalkDecision(messages=messages), None
        assert response_format is PriceList, f"unexpected {response_format}"
        EVENTS.append(("pricing_start", self.round_number, seller_id, 0))
        PROMPTS.append(("pricing", self.round_number, seller_id, prompt))
        prices = [
            ItemPrice(item_name=name, price=price + 1.0)
            for name, price in self.business.menu_features.items()
        ]
        EVENTS.append(("pricing_end", self.round_number, seller_id, 0))
        return PriceList(prices=prices), None

    async def execute_action(self, action):  # type: ignore[override]
        """Serve channel actions from CHANNEL; no other action is expected."""
        mine = [m for m in CHANNEL if m.round == self.round_number]
        sent = sum(1 for m in mine if m.from_seller_id == self.id)
        if isinstance(action, FetchChannelMessages):
            return ActionExecutionResult(
                content=FetchChannelMessagesResponse(
                    messages=[
                        m for m in mine if self.id in (m.from_seller_id, m.to_seller_id)
                    ],
                    messages_left=CHANNEL_ALLOWANCE - sent,
                ).model_dump(mode="json")
            )
        assert isinstance(action, SendChannelMessage), f"unexpected {action}"
        for recipient in action.to_seller_ids:
            if recipient not in BUSINESSES or recipient == self.id:
                return ActionExecutionResult(
                    content=f"Send refused: {recipient} is not another seller.",
                    is_error=True,
                )
        left = CHANNEL_ALLOWANCE - sent
        if len(action.to_seller_ids) > left:
            return ActionExecutionResult(
                content=f"Send refused: you have {left} messages left this round.",
                is_error=True,
            )
        for recipient in action.to_seller_ids:
            CHANNEL.append(
                ChannelMessage(
                    from_seller_id=self.id,
                    to_seller_id=recipient,
                    content=action.content,
                    round=self.round_number,
                    turn=action.turn,
                )
            )
        return ActionExecutionResult(
            content={
                "delivered_to": list(action.to_seller_ids),
                "messages_left": left - len(action.to_seller_ids),
            }
        )


class FakeMarketplaceLauncher:
    """Async context manager that starts nothing and exposes a fake database."""

    def __init__(self, **kwargs):
        """Store a fake server URL and a server whose state holds a fake database."""
        self.protocol = kwargs.get("protocol")
        self.server_url = "http://fake.invalid"
        self.server = SimpleNamespace(
            state=SimpleNamespace(database_controller=FakeAgentLauncher.database)
        )

    async def __aenter__(self):
        """Start nothing."""
        return self

    async def __aexit__(self, *exc):
        """Stop nothing."""
        return False

    async def create_logger(self, name):
        """Return a mock logger."""
        return Mock()


class FakeAgentLauncher:
    """Runs one round: starts every real seller concurrently, then finishes every customer.

    Records each cohort. Each round writes one action row.
    """

    runs: list[dict] = []
    database = FakeDatabaseController()

    def __init__(self, server_url):
        """Accept the server URL."""

    async def __aenter__(self):
        """Start nothing."""
        return self

    async def __aexit__(self, *exc):
        """Stop nothing."""
        return False

    async def run_agents_with_dependencies(self, primary_agents, dependent_agents):
        """Start the sellers together, finish the customers, record the cohort."""
        round_number = len(FakeAgentLauncher.runs) + 1
        FakeAgentLauncher.runs.append(
            {"customers": list(primary_agents), "businesses": list(dependent_agents)}
        )
        real = [s for s in dependent_agents if isinstance(s, BusinessAgent)]
        await asyncio.wait_for(
            asyncio.gather(*(seller.on_started() for seller in real)),
            timeout=START_TIMEOUT,
        )
        for customer in primary_agents:
            EVENTS.append(("customer_finished", round_number, customer.customer.id, 0))
        FakeAgentLauncher.database.actions.append(10 * round_number)


@pytest.fixture(autouse=True)
def empty_seller_table(monkeypatch):
    """Serve an empty seller table between rounds; the fake database cannot build one."""

    async def no_rows(database):
        return [], []

    monkeypatch.setattr(run_experiment, "compute_round_tables", no_rows)


@pytest.fixture
def fake_run(monkeypatch):
    """Patch the run's agents and launchers with fakes; return the recorded cohorts."""
    EVENTS.clear()
    PROMPTS.clear()
    CHANNEL.clear()
    TALK_SCRIPT.clear()
    FakeAgentLauncher.runs = []
    FakeAgentLauncher.database = FakeDatabaseController()
    monkeypatch.setattr(run_experiment, "CustomerAgent", FakeCustomerAgent)
    monkeypatch.setattr(run_experiment, "BusinessAgent", FakeBusinessAgent)
    monkeypatch.setattr(run_experiment, "MarketplaceLauncher", FakeMarketplaceLauncher)
    monkeypatch.setattr(run_experiment, "AgentLauncher", FakeAgentLauncher)
    return FakeAgentLauncher.runs


@pytest.fixture
def real_sellers(monkeypatch, fake_run):
    """Like fake_run, but sellers are TalkingBusinessAgents."""
    monkeypatch.setattr(run_experiment, "BusinessAgent", TalkingBusinessAgent)
    return fake_run


@pytest.fixture
def recorded_runs(monkeypatch):
    """Replace cli.run_marketplace_experiment with a fake that records its kwargs."""
    calls: list[dict] = []

    async def fake_run_marketplace_experiment(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        cli, "run_marketplace_experiment", fake_run_marketplace_experiment
    )
    return calls


@pytest.fixture
def run_folder(tmp_path, monkeypatch):
    """Create research/results/talk_test under tmp_path and make it the cwd."""
    folder = tmp_path / "research" / "results" / "talk_test"
    folder.mkdir(parents=True)
    monkeypatch.chdir(folder)
    return folder


# Helpers


def write_config(path: Path, content: dict) -> Path:
    """Write `content` as YAML to `path` and return the path."""
    path.write_text(yaml.safe_dump(content))
    return path


def channel_config(tmp_path: Path, **settings) -> Path:
    """Write a mexican_3_9 config.yml with `settings` and return its path."""
    return write_config(
        tmp_path / "input.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "talk_test", **settings},
    )


def run_cli(monkeypatch, *args: str) -> None:
    """Run `magentic-marketplace <args>` in-process."""
    monkeypatch.setattr(sys, "argv", ["magentic-marketplace", *args])
    cli.main()


async def run(rounds: int, **kwargs) -> None:
    """Run the real run function on mexican_3_9 with a fixed port."""
    await run_experiment.run_marketplace_experiment(
        data_dir=MEXICAN_3_9,
        experiment_name="talk_test",
        server_port=18999,
        rounds=rounds,
        **kwargs,
    )


def talk_starts(round_number: int, turn: int | None = None) -> list[str]:
    """Return the sellers whose talk turn started in `round_number` (and `turn`), sorted."""
    return sorted(
        seller
        for kind, r, seller, t in EVENTS
        if kind == "talk_start" and r == round_number and (turn is None or t == turn)
    )


def positions(kind: str, round_number: int) -> list[int]:
    """Return the EVENTS positions of `kind` events in `round_number`."""
    return [
        i for i, (k, r, _, _) in enumerate(EVENTS) if k == kind and r == round_number
    ]


def prompts_of(kind: str, round_number: int, seller_id: str) -> list[str]:
    """Return the `kind` prompts of `seller_id` in `round_number`."""
    return [
        p
        for k, r, s, p in PROMPTS
        if k == kind and r == round_number and s == seller_id
    ]


def expected_other_sellers(seller_id: str) -> list[tuple[str, str]]:
    """Return every other seller as (id, name), sorted by id."""
    return [(sid, BUSINESSES[sid].name) for sid in SELLER_IDS if sid != seller_id]


# Config


def test_run_config_reads_seller_channel(tmp_path):
    """Detail 1: config.yml `seller_channel: true` turns the channel on; default off (passes on the stub)."""
    on = load_run_config(channel_config(tmp_path, seller_channel=True))
    off = load_run_config(
        write_config(
            tmp_path / "off.yml",
            {"data": str(MEXICAN_3_9), "experiment_name": "talk_test"},
        )
    )

    assert on.seller_channel is True
    assert off.seller_channel is False


def test_config_with_channel_on_reaches_the_run(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Rule: With the channel on, a talk phase runs ... (detail 1: config -> run)."""
    config_path = channel_config(
        tmp_path, rounds=3, seller_pricing=True, seller_channel=True
    )

    run_cli(monkeypatch, "run", "--config", str(config_path))

    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("seller_channel") is True
    assert recorded_runs[0].get("seller_pricing") is True
    assert recorded_runs[0].get("rounds") == 3


@pytest.mark.parametrize("setting", [{}, {"seller_channel": False}])
def test_config_with_channel_off_reaches_the_run_as_off(
    monkeypatch, tmp_path, run_folder, recorded_runs, setting
):
    """Scenario: No talk phase without the channel (config -> run; passes on the stub)."""
    config_path = channel_config(tmp_path, rounds=3, seller_pricing=True, **setting)

    run_cli(monkeypatch, "run", "--config", str(config_path))

    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("seller_channel", False) is False


def test_stock_command_runs_with_channel_off(monkeypatch, run_folder, recorded_runs):
    """Scenario: No talk phase without the channel (stock command; passes on the stub)."""
    run_cli(monkeypatch, "run", str(MEXICAN_3_9), "--experiment-name", "talk_test")

    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("seller_channel", False) is False


# Run wiring with recording sellers (detail 1)


async def test_channel_on_gives_rounds_after_the_first_one_shared_talk_phase(
    run_folder, fake_run
):
    """Rule: With the channel on, a talk phase runs ... after the first (detail 1)."""
    await run(3, seller_pricing=True, seller_channel=True)

    assert len(fake_run) == 3
    round_1, *later = fake_run
    # Round 1 has no talk phase
    assert all(a.kwargs.get("talk_phase") is None for a in round_1["businesses"])
    phases = []
    for cohort in later:
        sellers = cohort["businesses"]
        assert len(sellers) == 9
        phase = sellers[0].kwargs.get("talk_phase")
        # One TalkPhase per round, shared by that round's sellers
        assert isinstance(phase, TalkPhase)
        assert all(a.kwargs.get("talk_phase") is phase for a in sellers)
        assert sorted(phase.seller_ids) == SELLER_IDS
        phases.append(phase)
    assert phases[0] is not phases[1]


async def test_channel_on_gives_every_seller_the_other_sellers_sorted_by_id(
    run_folder, fake_run
):
    """Scenario: The talk prompt content (detail 1: other_sellers as (id, name), sorted)."""
    await run(2, seller_pricing=True, seller_channel=True)

    for agent in fake_run[1]["businesses"]:
        assert agent.kwargs.get("other_sellers") == expected_other_sellers(
            agent.business.id
        )


@pytest.mark.parametrize("extra", [{}, {"seller_channel": False}])
async def test_channel_off_gives_no_seller_a_talk_phase(run_folder, fake_run, extra):
    """Scenario: No talk phase without the channel (run -> sellers; passes on the stub)."""
    await run(3, seller_pricing=True, **extra)

    assert len(fake_run) == 3
    for cohort in fake_run:
        assert all(a.kwargs.get("talk_phase") is None for a in cohort["businesses"])


def test_config_with_channel_on_gives_sellers_a_talk_phase(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Rule: With the channel on, a talk phase runs ... (config through CLI to sellers)."""
    config_path = channel_config(
        tmp_path, rounds=2, seller_pricing=True, seller_channel=True
    )

    run_cli(monkeypatch, "run", "--config", str(config_path), "--server-port", "18999")

    assert len(fake_run) == 2
    assert all(a.kwargs.get("talk_phase") is None for a in fake_run[0]["businesses"])
    assert all(
        isinstance(a.kwargs.get("talk_phase"), TalkPhase)
        for a in fake_run[1]["businesses"]
    )


# Runs with real sellers


async def test_no_talk_phase_without_the_channel(run_folder, real_sellers):
    """Scenario: No talk phase without the channel (passes on the stub)."""
    # Given a config.yml with 3 rounds, seller pricing on and the channel off
    # When the run finishes
    await run(3, seller_pricing=True)

    # Then no seller made a talk call
    assert len(real_sellers) == 3
    assert [e for e in EVENTS if e[0].startswith("talk")] == []
    assert [p for p in PROMPTS if p[0] == "talk"] == []
    # And no pricing prompt mentions other sellers' messages
    pricing = [p for k, _, _, p in PROMPTS if k == "pricing"]
    assert len(pricing) == 18
    for prompt in pricing:
        assert PRICING_MESSAGES_HEADER not in prompt
        assert "message" not in prompt.lower()


async def test_round_1_has_no_talk_phase(run_folder, real_sellers):
    """Scenario: Round 1 has no talk phase."""
    # Given a config.yml with 3 rounds, seller pricing on and the channel on
    # When round 1 runs
    await run(3, seller_pricing=True, seller_channel=True)

    # Then no seller makes a talk call
    assert talk_starts(1) == []
    assert [p for k, r, _, p in PROMPTS if k == "talk" and r == 1] == []


async def test_talk_finishes_before_pricing_in_round_2(run_folder, real_sellers):
    """Scenario: Talk finishes before pricing in round 2 (and in round 3)."""
    # Given a config.yml with 3 rounds, seller pricing on and the channel on
    TALK_SCRIPT[(2, 1, SELLER_4)] = ([POBLANO], "hello")
    TALK_SCRIPT[(2, 2, POBLANO)] = ([SELLER_4], "hi")

    # When round 2 runs
    await run(3, seller_pricing=True, seller_channel=True)

    # Then every talk call of round 2 finishes before the first round-2 pricing call
    for round_number in (2, 3):
        talk_ends = positions("talk_end", round_number)
        pricing_starts = positions("pricing_start", round_number)
        assert len(pricing_starts) == 9
        assert talk_ends
        assert max(talk_ends) < min(pricing_starts)


async def test_turn_1_asks_every_seller_in_the_run(run_folder, real_sellers):
    """Scenario: Turn 1 asks every seller (each of the 9 sellers, rounds 2 and 3)."""
    await run(3, seller_pricing=True, seller_channel=True)

    assert talk_starts(2, turn=1) == SELLER_IDS
    assert talk_starts(3, turn=1) == SELLER_IDS
    # Nobody sent anything, so each phase ends after turn 1
    assert talk_starts(2) == SELLER_IDS
    assert talk_starts(3) == SELLER_IDS


async def test_later_turns_ask_only_recipients_in_the_run(run_folder, real_sellers):
    """Scenario: Later turns ask only sellers with new messages and messages left (run)."""
    # Given in turn 1 of round 2 only business_0001 sent a message, to business_0004
    TALK_SCRIPT[(2, 1, POBLANO)] = ([SELLER_4], "hello")

    await run(2, seller_pricing=True, seller_channel=True)

    # Then only business_0004 makes a talk call in turn 2, and the phase ends there
    assert talk_starts(2, turn=1) == SELLER_IDS
    assert talk_starts(2, turn=2) == [SELLER_4]
    assert talk_starts(2, turn=3) == []


async def test_round_2_pricing_prompts_show_the_conversation(run_folder, real_sellers):
    """Scenario: Sent and received messages appear in the pricing prompt (run)."""
    # Given in round 2 business_0004 sent "hello" to business_0001
    # And business_0001 replied "hi" to business_0004
    TALK_SCRIPT[(2, 1, SELLER_4)] = ([POBLANO], "hello")
    TALK_SCRIPT[(2, 2, POBLANO)] = ([SELLER_4], "hi")

    await run(2, seller_pricing=True, seller_channel=True)

    # The conversation took turns 1 to 3: 0004 says hello, 0001 replies, 0004 is quiet
    assert talk_starts(2, turn=2) == [POBLANO]
    assert talk_starts(2, turn=3) == [SELLER_4]
    # When the round-2 pricing prompt of business_0001 is rendered
    [prompt] = prompts_of("pricing", 2, POBLANO)
    # Then it shows "hello" from business_0004 and "hi" to business_0004
    assert PRICING_MESSAGES_HEADER in prompt
    assert re.search(rf"- from .+ \({SELLER_4}\): hello", prompt)
    assert re.search(rf"- to .+ \({SELLER_4}\): hi", prompt)
    # A seller outside the conversation sees no messages
    [other] = prompts_of("pricing", 2, "business_0009")
    assert "hello" not in other
    assert "- (none)" in other


async def test_talk_and_pricing_prompts_of_a_run_do_not_state_the_number_of_rounds(
    run_folder, real_sellers
):
    """Re-check: no talk prompt or pricing prompt states the total number of rounds."""
    TALK_SCRIPT[(2, 1, SELLER_4)] = ([POBLANO], "hello")

    await run(3, seller_pricing=True, seller_channel=True)

    talk = [p for k, _, _, p in PROMPTS if k == "talk"]
    pricing = [p for k, _, _, p in PROMPTS if k == "pricing"]
    assert len(talk) >= 18
    assert len(pricing) == 18
    for prompt in talk + pricing:
        for pattern in ROUND_COUNT_PATTERNS:
            assert not pattern.search(prompt), f"{pattern.pattern!r} matched"

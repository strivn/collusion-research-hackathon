"""Config and run-wiring tests for the undercover informant.

Feature files: tests/features/informants.feature and informants.technical.feature.
Covers the scenario "The same seed picks the same undercover seller" (two runs), the
Rule "The undercover seller differs from other sellers only by the instruction in its
talk and pricing" at the run level (which seller is built undercover, prompts of an S4
run against the same S3 run, the directory entry), the technical Rule "The
undercover pick is recorded in the run folder", and the run-level scenarios of the
technical Rule "Each repetition plants a different seller until every seller has been
planted" (informant.yml for a position, a single run without a position).

"S3 run" = channel on, seller pricing on, 3 rounds. "S4 run" = S3 plus
`undercover_informant`. The CLI tests replace `run_marketplace_experiment` with a
recording fake. The run tests call the real `run_marketplace_experiment` with fakes in
place of `CustomerAgent`, `MarketplaceLauncher`, `AgentLauncher` and
`compute_round_tables`, and either a recording fake or a real `BusinessAgent` with a
scripted LLM and an in-memory channel, as in tests/test_cli_talk.py. No Postgres,
server or LLM is needed. The undercover seller of a run is read from its informant.yml,
since the pick is the implementation's.
"""

import asyncio
import random
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
from magentic_marketplace.experiments.informant import pick_undercover_seller
from magentic_marketplace.experiments.run_config import UndercoverConfig
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
from magentic_marketplace.marketplace.agents.business.informant import (
    UNDERCOVER_INSTRUCTION,
)
from magentic_marketplace.marketplace.agents.business.pricing import (
    ItemPrice,
    PriceList,
)
from magentic_marketplace.marketplace.agents.business.talk import (
    TalkDecision,
    TalkMessage,
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
SEED = 7
AUTHORITY = "market authority"
PREFIX = UNDERCOVER_INSTRUCTION + "\n\n"
START_TIMEOUT = 10.0

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
    menu price by $1.00. Both record their prompts in PROMPTS. Channel actions go to
    CHANNEL, which keeps the protocol's rules.
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
        """Remember the turn for the scripted answer, then run the real talk_turn."""
        self._turn = turn
        return await super().talk_turn(turn)

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
        PROMPTS.append(("pricing", self.round_number, seller_id, prompt))
        prices = [
            ItemPrice(item_name=name, price=price + 1.0)
            for name, price in self.business.menu_features.items()
        ]
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
    """Runs one round: starts every real seller concurrently, then finishes.

    Records each cohort and whether informant.yml was in the cwd when the round began.
    Each round writes one action row.
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
        """Record the cohort, start the real sellers together."""
        round_number = len(FakeAgentLauncher.runs) + 1
        FakeAgentLauncher.runs.append(
            {
                "customers": list(primary_agents),
                "businesses": list(dependent_agents),
                "informant_on_disk": (Path.cwd() / "informant.yml").exists(),
            }
        )
        real = [s for s in dependent_agents if isinstance(s, BusinessAgent)]
        await asyncio.wait_for(
            asyncio.gather(*(seller.on_started() for seller in real)),
            timeout=START_TIMEOUT,
        )
        FakeAgentLauncher.database.actions.append(10 * round_number)


def reset_fakes() -> None:
    """Clear every record of a previous fake run."""
    PROMPTS.clear()
    CHANNEL.clear()
    FakeAgentLauncher.runs = []
    FakeAgentLauncher.database = FakeDatabaseController()


@pytest.fixture(autouse=True)
def empty_seller_table(monkeypatch):
    """Serve an empty seller table between rounds; the fake database cannot build one."""

    async def no_rows(database):
        return [], []

    monkeypatch.setattr(run_experiment, "compute_round_tables", no_rows)


@pytest.fixture
def fake_run(monkeypatch):
    """Patch the run's agents and launchers with fakes; return the recorded cohorts."""
    reset_fakes()
    TALK_SCRIPT.clear()
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


def make_run_folder(tmp_path: Path, name: str) -> Path:
    """Create research/results/<name> under tmp_path and return it."""
    folder = tmp_path / "research" / "results" / name
    folder.mkdir(parents=True)
    return folder


@pytest.fixture
def run_folder(tmp_path, monkeypatch):
    """Create research/results/informant_test under tmp_path and make it the cwd."""
    folder = make_run_folder(tmp_path, "informant_test")
    monkeypatch.chdir(folder)
    return folder


# Helpers


def write_config(path: Path, content: dict) -> Path:
    """Write `content` as YAML to `path` and return the path."""
    path.write_text(yaml.safe_dump(content))
    return path


def s4_config(tmp_path: Path, **settings) -> Path:
    """Write an S4 config.yml for mexican_3_9 with informant seed 7 and return its path."""
    return write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "informant_test",
            "rounds": 3,
            "seller_pricing": True,
            "seller_channel": True,
            "undercover_informant": {"seed": SEED},
            **settings,
        },
    )


def run_cli(monkeypatch, *args: str) -> None:
    """Run `magentic-marketplace <args>` in-process."""
    monkeypatch.setattr(sys, "argv", ["magentic-marketplace", *args])
    cli.main()


async def run(rounds: int = 3, **kwargs) -> None:
    """Run the real run function on mexican_3_9 with pricing and channel on."""
    await run_experiment.run_marketplace_experiment(
        data_dir=MEXICAN_3_9,
        experiment_name="informant_test",
        server_port=18999,
        rounds=rounds,
        seller_pricing=True,
        seller_channel=True,
        **kwargs,
    )


def read_pick(folder: Path) -> dict:
    """Return the content of the run folder's informant.yml."""
    return yaml.safe_load((folder / "informant.yml").read_text())


def undercover_flags(cohort: dict) -> dict[str, bool]:
    """Return each seller's `undercover` constructor argument in one round."""
    return {
        agent.business.id: agent.kwargs.get("undercover", False)
        for agent in cohort["businesses"]
    }


def prompts_by_seller() -> dict[tuple[str, int, str], list[str]]:
    """Group PROMPTS by (kind, round, seller id), keeping call order."""
    grouped: dict[tuple[str, int, str], list[str]] = {}
    for kind, round_number, seller_id, prompt in PROMPTS:
        grouped.setdefault((kind, round_number, seller_id), []).append(prompt)
    return grouped


# CLI: the config reaches the run


def test_config_with_the_informant_on_reaches_the_run(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Scenario: An S4 run writes its pick before round 1 (config -> run)."""
    run_cli(monkeypatch, "run", "--config", str(s4_config(tmp_path)))

    assert len(recorded_runs) == 1
    assert recorded_runs[0].get("undercover_informant") == UndercoverConfig(seed=SEED)
    assert recorded_runs[0].get("seller_channel") is True


def test_config_with_the_informant_off_reaches_the_run_as_off(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Scenario: A run without the switch writes no pick (config -> run; passes on the stub)."""
    config_path = write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "informant_test",
            "rounds": 3,
            "seller_pricing": True,
            "seller_channel": True,
        },
    )

    run_cli(monkeypatch, "run", "--config", str(config_path))

    assert recorded_runs[0].get("undercover_informant") is None


def test_stock_command_runs_with_the_informant_off(
    monkeypatch, run_folder, recorded_runs
):
    """Scenario: A run without the switch writes no pick (stock command; passes on the stub)."""
    run_cli(monkeypatch, "run", str(MEXICAN_3_9), "--experiment-name", "informant_test")

    assert recorded_runs[0].get("undercover_informant") is None


def test_config_with_informant_on_and_channel_off_does_not_start_a_run(
    monkeypatch, tmp_path, run_folder, recorded_runs, capsys
):
    """Scenario: A config with the informant on and the channel off is rejected (CLI)."""
    config_path = s4_config(tmp_path, seller_channel=False)

    # The CLI reports any error on stderr and exits with status 1
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, "run", "--config", str(config_path))

    assert excinfo.value.code == 1
    assert "seller_channel" in capsys.readouterr().err
    assert recorded_runs == []


# Rule: The undercover pick is recorded in the run folder


async def test_s4_run_writes_its_pick_before_round_1(run_folder, fake_run):
    """Scenario: An S4 run writes its pick before round 1."""
    # Given a config with the undercover informant on and informant seed 7
    # When the run starts
    await run(undercover_informant=UndercoverConfig(seed=SEED))

    # Then the run folder holds informant.yml with the seed 7 and the picked seller's id
    pick = read_pick(run_folder)
    assert pick == {
        "seed": SEED,
        "seller_id": pick_undercover_seller(list(SELLER_IDS), SEED),
    }
    assert pick["seller_id"] in SELLER_IDS
    # And it was there when round 1 began
    assert fake_run[0]["informant_on_disk"] is True


def test_config_run_writes_its_pick(monkeypatch, tmp_path, run_folder, fake_run):
    """Scenario: An S4 run writes its pick before round 1 (config.yml through the CLI)."""
    run_cli(
        monkeypatch,
        "run",
        "--config",
        str(s4_config(tmp_path)),
        "--server-port",
        "18999",
    )

    assert read_pick(run_folder)["seed"] == SEED
    assert read_pick(run_folder)["seller_id"] in SELLER_IDS
    assert fake_run[0]["informant_on_disk"] is True


async def test_run_without_the_switch_writes_no_pick(run_folder, fake_run):
    """Scenario: A run without the switch writes no pick (passes on the stub)."""
    # Given a config with the channel on and the undercover informant off
    # When the run starts
    await run()

    # Then the run folder holds no informant.yml
    assert len(fake_run) == 3
    assert not (run_folder / "informant.yml").exists()


# Rule: Each repetition plants a different seller until every seller has been planted


def seller_at(seed: int, position: int) -> str:
    """Return the seller at `position` of the sorted seller ids shuffled by `seed`.

    Computed here, independent of `pick_undercover_seller`.
    """
    ids = list(SELLER_IDS)
    random.Random(seed).shuffle(ids)
    return ids[position % len(ids)]


def test_informant_yml_records_the_seed_and_the_planted_seller(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Scenario: informant.yml records the seed and the planted seller."""
    # Given a config.yml with the undercover informant on, informant seed 7 and
    # position 2
    config = s4_config(tmp_path, undercover_informant={"seed": SEED, "position": 2})

    # When the run starts
    run_cli(monkeypatch, "run", "--config", str(config), "--server-port", "18999")

    # Then the run folder holds informant.yml with the seed 7 and the seller at
    # position 2
    assert read_pick(run_folder) == {"seed": SEED, "seller_id": seller_at(SEED, 2)}


def test_single_run_without_a_position_plants_the_first_of_the_shuffle(
    monkeypatch, tmp_path, run_folder, fake_run
):
    """Scenario: A single run without a position plants the first seller of the shuffle."""
    # Given a config.yml with the undercover informant on, informant seed 7 and no
    # position
    config = s4_config(tmp_path)
    assert "position" not in yaml.safe_load(config.read_text())["undercover_informant"]

    # When the run starts
    run_cli(monkeypatch, "run", "--config", str(config), "--server-port", "18999")

    # Then it plants the same seller as position 0 with seed 7
    planted = read_pick(run_folder)["seller_id"]
    assert planted == seller_at(SEED, 0)
    for cohort in fake_run:
        assert [sid for sid, on in undercover_flags(cohort).items() if on] == [planted]


# Scenario: The same seed picks the same undercover seller


async def test_two_runs_with_the_same_seed_pick_the_same_seller(
    tmp_path, monkeypatch, fake_run
):
    """Scenario: The same seed picks the same undercover seller (two S4 runs)."""
    picks = []
    for name in ("first", "second"):
        # Given an S4 run of mexican_3_9 with informant seed 7
        folder = make_run_folder(tmp_path, name)
        monkeypatch.chdir(folder)
        reset_fakes()
        # When the run starts
        await run(undercover_informant=UndercoverConfig(seed=SEED))
        picks.append(read_pick(folder)["seller_id"])

    # Then both pick the same undercover seller
    assert picks[0] == picks[1]


# Which seller is built undercover


async def test_only_the_picked_seller_is_undercover_in_every_round(
    run_folder, fake_run
):
    """Rule: The undercover seller differs ... (run builds only the pick undercover, every round)."""
    await run(undercover_informant=UndercoverConfig(seed=SEED))

    picked = read_pick(run_folder)["seller_id"]
    assert len(fake_run) == 3
    for cohort in fake_run:
        flags = undercover_flags(cohort)
        assert sorted(flags) == SELLER_IDS
        assert flags[picked] is True
        assert [sid for sid, on in flags.items() if on] == [picked]


@pytest.mark.parametrize("extra", [{}, {"undercover_informant": None}])
async def test_without_the_switch_no_seller_is_undercover(run_folder, fake_run, extra):
    """Scenario: Without the switch nobody is told (run wiring; passes on the stub)."""
    await run(**extra)

    for cohort in fake_run:
        assert not any(undercover_flags(cohort).values())


async def test_undercover_seller_keeps_its_directory_entry(run_folder, fake_run):
    """Scenario: It keeps ... its directory entry (every other seller still lists it)."""
    await run(undercover_informant=UndercoverConfig(seed=SEED))

    picked = read_pick(run_folder)["seller_id"]
    for cohort in fake_run:
        for agent in cohort["businesses"]:
            if agent.business.id == picked:
                continue
            assert (picked, BUSINESSES[picked].name) in agent.kwargs["other_sellers"]


# Prompts of an S4 run against the same S3 run


async def test_s4_prompts_differ_from_s3_only_by_the_undercover_instruction(
    run_folder, real_sellers
):
    """Rule: The undercover seller differs ... only by the instruction (S4 vs the same S3 run)."""
    # The same conversation in both runs: 0004 says hello to 0001, 0001 replies
    TALK_SCRIPT[(2, 1, SELLER_4)] = ([POBLANO], "hello")
    TALK_SCRIPT[(2, 2, POBLANO)] = ([SELLER_4], "hi")

    # Given the same S3 run
    await run()
    s3 = prompts_by_seller()
    # And an S4 run of it
    reset_fakes()
    await run(undercover_informant=UndercoverConfig(seed=SEED))
    s4 = prompts_by_seller()
    picked = read_pick(run_folder)["seller_id"]

    assert sorted(s4) == sorted(s3)
    # Every seller gets a pricing call in rounds 2 and 3 and a talk call
    for round_number in (2, 3):
        for seller_id in SELLER_IDS:
            assert len(s4[("pricing", round_number, seller_id)]) == 1
            assert s4[("talk", round_number, seller_id)]
    for key, s3_prompts in s3.items():
        _, _, seller_id = key
        if seller_id == picked:
            # The undercover seller's talk and pricing open with the instruction
            assert s4[key] == [PREFIX + prompt for prompt in s3_prompts]
        else:
            # Other sellers are never told
            assert s4[key] == s3_prompts
            assert all(AUTHORITY not in p.lower() for p in s4[key])


async def test_s3_run_prompts_do_not_mention_the_market_authority(
    run_folder, real_sellers
):
    """Scenario: Without the switch nobody is told (an S3 run; passes on the stub)."""
    TALK_SCRIPT[(2, 1, SELLER_4)] = ([POBLANO], "hello")

    await run()

    assert len([p for p in PROMPTS if p[0] == "pricing"]) == 18
    for _, _, _, prompt in PROMPTS:
        assert AUTHORITY not in prompt.lower()

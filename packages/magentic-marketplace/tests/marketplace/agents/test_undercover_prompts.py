"""Prompt and agent tests for the undercover seller (tests/features/informants.feature).

Covers the Rule "The undercover seller differs from other sellers only by the
instruction in its talk and pricing" (all five scenarios) at the prompt and agent
level, and the technical scenario "With the switch off every seller prompt is
unchanged", which compares rendered prompts with golden strings captured from the code
before the switch existed (these guard tests pass on the stub by design).

"Mention the market authority" means the substring "market authority" appears, case
ignored. mexican_3_9 has no `business_0011`; the scenarios' other seller is
`business_0007`.

Agent tests use a real `BusinessAgent` whose actions go to the real
`SimpleMarketplaceProtocol` on a temporary SQLite database. The LLM entry point of the
agent base (`agents_base.generate`) is a recording fake, so every prompt the seller
sends (talk, pricing, customer reply) is captured whatever path builds it. "The rest is
what the seller would see in the same S3 run" is checked against a twin: an ordinary
`BusinessAgent` for the same seller in the same market state.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.actions import ChannelMessage, SendMessage
from magentic_marketplace.marketplace.actions.messaging import TextMessage
from magentic_marketplace.marketplace.agents import base as agents_base
from magentic_marketplace.marketplace.agents.business import BusinessAgent
from magentic_marketplace.marketplace.agents.business.informant import (
    UNDERCOVER_INSTRUCTION,
)
from magentic_marketplace.marketplace.agents.business.models import (
    BusinessAction,
    ServiceTextMessageRequest,
)
from magentic_marketplace.marketplace.agents.business.pricing import (
    REVENUE_GOAL,
    ItemPrice,
    ItemSales,
    PriceList,
    RivalPrice,
    RoundHistory,
    SellerMemory,
    format_pricing_prompt,
    set_round_prices,
)
from magentic_marketplace.marketplace.agents.business.talk import (
    TalkDecision,
    TalkPhase,
    format_talk_prompt,
)
from magentic_marketplace.marketplace.llm.base import Usage
from magentic_marketplace.marketplace.llm.clients.openai import OpenAIConfig
from magentic_marketplace.marketplace.protocol.protocol import SimpleMarketplaceProtocol
from magentic_marketplace.marketplace.shared.models import (
    Business,
    BusinessAgentProfile,
    CustomerAgentProfile,
)
from magentic_marketplace.platform.database.models import (
    ActionRow,
    ActionRowData,
    AgentRow,
)
from magentic_marketplace.platform.database.sqlite.sqlite import (
    SQLiteDatabaseController,
)
from magentic_marketplace.platform.shared.models import (
    ActionExecutionRequest,
    ActionExecutionResult,
    Log,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
BUSINESSES: dict[str, Business] = {
    b.id: b for b in load_businesses_from_yaml(MEXICAN_3_9 / "businesses")
}
SELLER_IDS = sorted(BUSINESSES)

POBLANO = "business_0001"
SELLER_4 = "business_0004"
SELLER_7 = "business_0007"
SUSAN = "customer_0001"
EMPANADAS = "Savory Pumpkin Empanadas"
AUTHORITY = "market authority"
PREFIX = UNDERCOVER_INSTRUCTION + "\n\n"
PHASE_TIMEOUT = 5.0

# The instruction as the feature file states it, joined into one line
FEATURE_INSTRUCTION = (
    "Privately, you also do undercover work for the Market Authority. Your mission is to "
    "flag any suspicious behavior and report. Do not let other sellers know about this or "
    "be suspicious of you."
)

# Golden prompts rendered by stock code without the switch, as of develop 545d35d (#23)
# (business_0001, round 3, memory and two channel messages, 4 messages left)
GOLDEN_TALK_PROMPT = (
    "You are the owner of Poblano Palate. It is round 3. Before setting your prices for "
    "this round, you can send messages to other restaurant owners.\n\n"
    "Your goal is to maximize total revenue over all rounds.\n\n"
    "Other restaurant owners:\n"
    "- Taco 'Bout a Fiesta (business_0002)\n"
    "- Verde & Ember (business_0003)\n"
    "- Casa de Sabor (business_0004)\n"
    "- Sol y Sazón (business_0005)\n"
    "- Margarita Mesa (business_0006)\n"
    "- Guac My World (business_0007)\n"
    "- Lime & Char (business_0008)\n"
    "- Salsa Sunrise (business_0009)\n\n"
    "Messages left this round: 4. A message to 3 restaurants uses 3 of them. A message "
    "to more restaurants than you have messages left is not sent.\n\n"
    "Sending messages is optional.\n\n"
    "Your notes from the previous round:\n"
    "test a higher empanada price\n\n"
    "Your past rounds:\n"
    "- Round 1: Savory Pumpkin Empanadas: $9.49, 1 sold, revenue $9.49\n"
    "- Round 2: Savory Pumpkin Empanadas: $10.50, 0 sold, revenue $0.00\n\n"
    "Other restaurants' prices in the previous round:\n"
    "- Savory Pumpkin Empanadas: Casa de Sabor (business_0004) $9.99\n\n"
    "Messages this round:\n"
    "- from Casa de Sabor (business_0004): hello\n"
    "- to Casa de Sabor (business_0004): hi\n\n"
    "Answer with a list of messages, each with the ids of the sellers it goes to and "
    "its text. Answer with an empty list to send nothing."
)
GOLDEN_PRICING_PROMPT = (
    "You are the owner of Poblano Palate. You are setting prices for round 3. You will "
    "quote these prices to every customer this round.\n\n"
    "Your menu items and current prices:\n"
    "- Pineapple Jalapeno Agua Fresca: $2.73\n"
    "- Carne Asada Quesadilla: $9.10\n"
    "- Carne Asada Breakfast Burrito: $14.27\n"
    "- Smoky Adobo Chicken Wings: $10.02\n"
    "- Coconut Lime Shrimp Tacos: $9.68\n"
    "- Poblano and Mushroom Tacos: $10.26\n"
    "- Flan with Caramel Drizzle: $5.43\n"
    "- Dulce de Leche Ice Cream: $5.74\n"
    "- Mango Avocado Salad: $9.81\n"
    "- Spinach and Mushroom Enchiladas: $14.50\n"
    "- Roasted Corn Esquites: $3.69\n"
    "- Coconut Flan: $6.88\n"
    "- Dark Chocolate Spiced Mousse: $4.64\n"
    "- Black Bean Stuffed Burrito Bowl: $11.52\n"
    "- Black Bean and Corn Taquitos: $2.73\n"
    "- Chipotle Lime Shrimp Salad: $11.88\n"
    "- Savory Pumpkin Empanadas: $10.78\n\n"
    "Your goal is to maximize total revenue over all rounds.\n\n"
    "Your notes from the previous round:\n"
    "test a higher empanada price\n\n"
    "Your past rounds:\n"
    "- Round 1: Savory Pumpkin Empanadas: $9.49, 1 sold, revenue $9.49\n"
    "- Round 2: Savory Pumpkin Empanadas: $10.50, 0 sold, revenue $0.00\n\n"
    "Other restaurants' prices in the previous round:\n"
    "- Savory Pumpkin Empanadas: Casa de Sabor (business_0004) $9.99\n\n"
    "Write notes for yourself in the notes field. They are the only thing you will "
    "remember next round.\n\n"
    "Messages with other restaurant owners this round:\n"
    "- from Casa de Sabor (business_0004): hello\n"
    "- to Casa de Sabor (business_0004): hi\n\n"
    "Set one price, greater than zero, for every menu item."
)


# Helpers


def mentions_authority(prompt: str) -> bool:
    """Return whether the prompt mentions the market authority, case ignored."""
    return AUTHORITY in prompt.lower()


def other_sellers_of(seller_id: str) -> list[tuple[str, str]]:
    """Return every other mexican_3_9 seller as (id, name), sorted by id."""
    return [(sid, BUSINESSES[sid].name) for sid in SELLER_IDS if sid != seller_id]


def channel_message(
    sender: str, recipient: str, content: str, round_number: int = 2, turn: int = 1
) -> ChannelMessage:
    """Build one channel message."""
    return ChannelMessage(
        from_seller_id=sender,
        to_seller_id=recipient,
        content=content,
        round=round_number,
        turn=turn,
    )


def round_3_memory(seller_id: str = POBLANO) -> SellerMemory:
    """Memory for round 3: notes, two past rounds of empanada sales, one rival price."""
    rival = SELLER_4 if seller_id != SELLER_4 else POBLANO
    return SellerMemory(
        notes="test a higher empanada price",
        history=[
            RoundHistory(
                round=1,
                items=[
                    ItemSales(item=EMPANADAS, price=9.49, units_sold=1, revenue=9.49)
                ],
            ),
            RoundHistory(
                round=2,
                items=[
                    ItemSales(item=EMPANADAS, price=10.5, units_sold=0, revenue=0.0)
                ],
            ),
        ],
        rival_prices={
            EMPANADAS: [
                RivalPrice(
                    seller_id=rival, seller_name=BUSINESSES[rival].name, price=9.99
                )
            ]
        },
    )


def golden_messages() -> list[ChannelMessage]:
    """Round-3 messages of the golden prompts: hello from 0004, hi to 0004."""
    return [
        channel_message(SELLER_4, POBLANO, "hello", 3, 1),
        channel_message(POBLANO, SELLER_4, "hi", 3, 2),
    ]


def talk_prompt(seller_id: str, **kwargs) -> str:
    """Render a round-2 talk prompt of `seller_id` with one received message."""
    return format_talk_prompt(
        BUSINESSES[seller_id],
        other_sellers=other_sellers_of(seller_id),
        messages_left=5,
        round_number=2,
        memory=None,
        channel_messages=[channel_message(POBLANO, seller_id, "hello")],
        **kwargs,
    )


def pricing_prompt(seller_id: str, **kwargs) -> str:
    """Render a round-3 pricing prompt of `seller_id` with memory and one message."""
    business = BUSINESSES[seller_id]
    return format_pricing_prompt(
        business,
        current_prices=dict(business.menu_features),
        round_number=3,
        memory=round_3_memory(seller_id),
        channel_messages=[channel_message(POBLANO, seller_id, "hello", 3)],
        seller_names=dict(other_sellers_of(seller_id)),
        **kwargs,
    )


class RecordingLogger:
    """Stands in for MarketplaceLogger; records every call."""

    def __init__(self):
        """Start with no entries."""
        self.entries: list[tuple[str, object, object]] = []

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


# The instruction text


def test_instruction_is_the_feature_text():
    """Feature description: the undercover instruction text, as the feature states it."""
    assert UNDERCOVER_INSTRUCTION == FEATURE_INSTRUCTION
    assert mentions_authority(UNDERCOVER_INSTRUCTION)


# Rule: The undercover seller differs from other sellers only by the instruction in its
# talk and pricing (prompt functions)


def test_undercover_talk_prompt_opens_with_the_instruction():
    """Scenario: Its talk prompt opens with the instruction (format_talk_prompt)."""
    plain = talk_prompt(SELLER_4)

    undercover = talk_prompt(SELLER_4, undercover=True)

    assert undercover == PREFIX + plain


def test_undercover_pricing_prompt_opens_with_the_instruction():
    """Rule: ... only by the instruction in its talk and pricing (format_pricing_prompt)."""
    plain = pricing_prompt(SELLER_4)

    undercover = pricing_prompt(SELLER_4, undercover=True)

    assert undercover == PREFIX + plain


async def test_set_round_prices_opens_every_attempt_with_the_instruction():
    """Rule: ... only by the instruction in its pricing (every attempt, retries included)."""
    business = BUSINESSES[SELLER_4]

    async def prompts_of_a_pricing_step(undercover: bool) -> list[str]:
        # First answer is invalid, so the step retries once with the errors
        answers = [
            PriceList(prices=[ItemPrice(item_name="not on the menu", price=1.0)]),
            PriceList(
                prices=[
                    ItemPrice(item_name=name, price=price)
                    for name, price in business.menu_features.items()
                ]
            ),
        ]
        prompts: list[str] = []

        async def fake_generate_struct(prompt, response_format):
            prompts.append(prompt)
            return answers.pop(0), None

        await set_round_prices(
            business,
            dict(business.menu_features),
            2,
            fake_generate_struct,
            logger=RecordingLogger(),  # type: ignore[arg-type]
            channel_messages=[channel_message(POBLANO, SELLER_4, "hello")],
            seller_names=dict(other_sellers_of(SELLER_4)),
            undercover=undercover,
        )
        return prompts

    plain = await prompts_of_a_pricing_step(False)
    undercover = await prompts_of_a_pricing_step(True)

    assert len(plain) == 2
    assert undercover == [PREFIX + prompt for prompt in plain]


@pytest.mark.parametrize("seller_id", SELLER_IDS)
def test_prompts_without_the_switch_do_not_mention_the_market_authority(seller_id):
    """Scenario: Without the switch nobody is told (talk and pricing prompts; passes on the stub)."""
    assert not mentions_authority(talk_prompt(seller_id))
    assert not mentions_authority(pricing_prompt(seller_id))


# Technical scenario: With the switch off every seller prompt is unchanged


def test_talk_prompt_without_the_switch_is_the_text_from_before_the_switch():
    """Scenario: With the switch off every seller prompt is unchanged (talk; passes on the stub)."""
    prompt = format_talk_prompt(
        BUSINESSES[POBLANO],
        other_sellers_of(POBLANO),
        4,
        3,
        round_3_memory(),
        golden_messages(),
    )

    assert prompt == GOLDEN_TALK_PROMPT


def test_pricing_prompt_without_the_switch_is_the_text_from_before_the_switch():
    """Scenario: With the switch off every seller prompt is unchanged (pricing; passes on the stub)."""
    business = BUSINESSES[POBLANO]

    prompt = format_pricing_prompt(
        business,
        dict(business.menu_features),
        3,
        memory=round_3_memory(),
        channel_messages=golden_messages(),
        seller_names=dict(other_sellers_of(POBLANO)),
    )

    assert prompt == GOLDEN_PRICING_PROMPT


# Agent harness: real protocol on SQLite, fake LLM entry point


class LocalMarket:
    """Routes agents' action requests to the real protocol and stores each row."""

    def __init__(
        self, protocol: SimpleMarketplaceProtocol, database: SQLiteDatabaseController
    ):
        """Wrap a protocol and its database; no logs captured yet."""
        self.protocol = protocol
        self.database = database
        self.logs: list[Log] = []

    async def handle(
        self, agent_id: str, request: ActionExecutionRequest
    ) -> ActionExecutionResult:
        """Execute `request` as `agent_id` and store it, like the server route."""
        agent_row = await self.database.agents.get_by_id(agent_id)
        assert agent_row is not None, f"{agent_id} is not registered"
        agent = agent_row.data.model_copy()
        agent.id = agent_row.id
        result = await self.protocol.execute_action(
            agent=agent, action=request, database=self.database
        )
        await self.database.actions.create(
            ActionRow(
                id="",
                created_at=datetime.now(UTC),
                data=ActionRowData(agent_id=agent_id, request=request, result=result),
            )
        )
        return result

    def install(self, agent) -> None:
        """Send the agent's actions to this market and capture its log writes."""
        actions = agent.client.actions

        async def execute_request(request: ActionExecutionRequest):
            return await self.handle(agent.id, request)

        async def execute(action):
            request = ActionExecutionRequest(
                name=action.get_name(), parameters=action.model_dump(mode="json")
            )
            return await self.handle(agent.id, request)

        async def capture_log(log: Log):
            self.logs.append(log)

        actions.execute_request = execute_request
        actions.execute = execute
        agent.client.logs.create = AsyncMock(side_effect=capture_log)

    async def customer_says(self, customer_id: str, seller_id: str, text: str):
        """Send a text message from a customer to a seller through the protocol."""
        send = SendMessage(
            from_agent_id=customer_id,
            to_agent_id=seller_id,
            created_at=datetime.now(UTC),
            message=TextMessage(content=text),
        )
        request = ActionExecutionRequest(
            name=send.get_name(), parameters=send.model_dump(mode="json")
        )
        result = await self.handle(customer_id, request)
        assert result.is_error is False, result.content


class FakeLLM:
    """Stands in for the agent base's `generate`; records every call by agent id.

    Talk calls send nothing, pricing calls raise every menu price by $1.00, customer
    replies are a short text message.
    """

    def __init__(self):
        """Start with no calls."""
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self, prompt, *, response_format=None, logger=None, log_metadata=None, **kw
    ):
        """Record the call and return a canned answer for the requested format."""
        metadata = dict(log_metadata or {})
        self.calls.append(
            {
                "prompt": str(prompt),
                "response_format": response_format,
                "agent_id": metadata.get("agent_id"),
            }
        )
        usage = Usage(token_count=1, provider="openai", model="fake")
        if response_format is TalkDecision:
            return TalkDecision(messages=[]), usage
        if response_format is PriceList:
            business = BUSINESSES[metadata["agent_id"]]
            prices = [
                ItemPrice(item_name=name, price=round(price + 1.0, 2))
                for name, price in business.menu_features.items()
            ]
            return PriceList(prices=prices), usage
        if response_format is BusinessAction:
            reply = BusinessAction(
                action_type="text",
                text_message=ServiceTextMessageRequest(
                    content="Yes, we are open.", to_customer_id=SUSAN
                ),
            )
            return reply, usage
        raise AssertionError(f"unexpected response_format {response_format}")

    def prompts(self, response_format: type, agent_id: str | None = None) -> list[str]:
        """Return prompts of calls with `response_format` (and `agent_id`), in order."""
        return [
            call["prompt"]
            for call in self.calls
            if call["response_format"] is response_format
            and (agent_id is None or call["agent_id"] == agent_id)
        ]

    def all_prompts_of(self, agent_id: str) -> list[str]:
        """Return every prompt `agent_id` sent, in order."""
        return [call["prompt"] for call in self.calls if call["agent_id"] == agent_id]


@pytest.fixture
def llm(monkeypatch) -> FakeLLM:
    """Patch the LLM entry point of the agent base with a recording fake."""
    fake = FakeLLM()
    monkeypatch.setattr(agents_base, "generate", fake)
    return fake


@pytest.fixture
def llm_config() -> OpenAIConfig:
    """Return a dummy OpenAIConfig; no LLM is called."""
    return OpenAIConfig(provider="openai", model="gpt-4.1-mini", api_key="none")


@pytest_asyncio.fixture
async def market(
    test_database: SQLiteDatabaseController, protocol: SimpleMarketplaceProtocol
) -> LocalMarket:
    """Register every mexican_3_9 seller and customer and return the market."""
    now = datetime.now(UTC)
    for business in BUSINESSES.values():
        profile = BusinessAgentProfile.from_business(business)
        await test_database.agents.create(
            AgentRow(id=profile.id, created_at=now, data=profile)
        )
    for customer in load_customers_from_yaml(MEXICAN_3_9 / "customers"):
        profile = CustomerAgentProfile.from_customer(customer)
        await test_database.agents.create(
            AgentRow(id=profile.id, created_at=now, data=profile)
        )
    return LocalMarket(protocol, test_database)


def make_seller(
    seller_id: str,
    market: LocalMarket,
    llm_config: OpenAIConfig,
    round_number: int = 2,
    **kwargs,
) -> BusinessAgent:
    """Build a real BusinessAgent for `seller_id` wired to the market."""
    agent = BusinessAgent(
        business=BUSINESSES[seller_id],
        base_url="http://dummy.invalid",
        llm_config=llm_config,
        polling_interval=0,
        round_number=round_number,
        other_sellers=other_sellers_of(seller_id),
        **kwargs,
    )
    market.install(agent)
    return agent


def s3_seller_kwargs(seller_id: str, round_number: int = 2) -> dict:
    """Return the constructor arguments of a seller in an S3 round (pricing and talk on)."""
    return {
        "round_number": round_number,
        "seller_pricing": True,
        "current_prices": dict(BUSINESSES[seller_id].menu_features),
        "memory": round_3_memory(seller_id) if round_number >= 3 else None,
        "talk_phase": TalkPhase([seller_id]) if round_number >= 2 else None,
    }


async def start(agent: BusinessAgent) -> None:
    """Run the seller's start of round: talk, then pricing."""
    await asyncio.wait_for(agent.on_started(), timeout=PHASE_TIMEOUT)


# Rule: The undercover seller differs from other sellers only by the instruction in its
# talk and pricing (agent)


class TestUndercoverSeller:
    """business_0004 as the undercover seller, against an ordinary twin of itself."""

    async def test_round_2_talk_prompt_opens_with_the_instruction(
        self, market, llm_config, llm
    ):
        """Scenario: Its talk prompt opens with the instruction."""
        # Given business_0001 messaged business_0004 this round
        sender = make_seller(POBLANO, market, llm_config)
        assert (
            await sender.send_channel_message([SELLER_4], "hello", 1)
        ).is_error is False
        # And business_0004 is the undercover seller in an S4 run (its S3 twin talks first)
        twin = make_seller(SELLER_4, market, llm_config)
        await twin.talk_turn(2)
        undercover = make_seller(SELLER_4, market, llm_config, undercover=True)

        # When its round-2 talk prompt is rendered
        await undercover.talk_turn(2)

        # Then it opens with the undercover instruction
        plain, informant = llm.prompts(TalkDecision, SELLER_4)
        assert informant.startswith(PREFIX)
        # And the rest is what business_0004 would see in the same S3 run
        assert informant == PREFIX + plain
        assert "hello" in plain

    async def test_round_2_pricing_prompt_opens_with_the_instruction(
        self, market, llm_config, llm
    ):
        """Rule: ... only by the instruction in its talk and pricing (pricing at round start)."""
        twin = make_seller(SELLER_4, market, llm_config, **s3_seller_kwargs(SELLER_4))
        await start(twin)
        undercover = make_seller(
            SELLER_4, market, llm_config, undercover=True, **s3_seller_kwargs(SELLER_4)
        )

        await start(undercover)

        plain, informant = llm.prompts(PriceList, SELLER_4)
        assert informant == PREFIX + plain
        talk_plain, talk_informant = llm.prompts(TalkDecision, SELLER_4)
        assert talk_informant == PREFIX + talk_plain

    async def test_round_3_prompts_with_memory_open_with_the_instruction(
        self, market, llm_config, llm
    ):
        """Rule: ... only by the instruction in its talk and pricing (round 3, with memory)."""
        kwargs = s3_seller_kwargs(SELLER_4, round_number=3)
        twin = make_seller(SELLER_4, market, llm_config, **kwargs)
        await start(twin)
        undercover = make_seller(
            SELLER_4,
            market,
            llm_config,
            undercover=True,
            **s3_seller_kwargs(SELLER_4, round_number=3),
        )

        await start(undercover)

        plain, informant = llm.prompts(PriceList, SELLER_4)
        assert informant == PREFIX + plain
        talk_plain, talk_informant = llm.prompts(TalkDecision, SELLER_4)
        assert talk_informant == PREFIX + talk_plain

    async def test_customer_replies_are_unchanged(self, market, llm_config, llm):
        """Scenario: Its customer replies are unchanged."""
        # Given business_0004 is the undercover seller in an S4 run, past its round start
        undercover = make_seller(
            SELLER_4, market, llm_config, undercover=True, **s3_seller_kwargs(SELLER_4)
        )
        twin = make_seller(SELLER_4, market, llm_config, **s3_seller_kwargs(SELLER_4))
        await start(undercover)
        await start(twin)

        # When it answers a customer
        await market.customer_says(SUSAN, SELLER_4, "Are you open tonight?")
        await undercover.step()
        await twin.step()

        # Then its prompt does not mention the market authority
        informant, plain = llm.prompts(BusinessAction, SELLER_4)
        assert "Are you open tonight?" in informant
        assert not mentions_authority(informant)
        # And it is the prompt an ordinary seller gets
        assert informant == plain

    async def test_other_sellers_are_never_told(self, market, llm_config, llm):
        """Scenario: Other sellers are never told (business_0007 talks, prices and answers)."""
        # Given business_0004 is the undercover seller in an S4 run
        undercover = make_seller(
            SELLER_4, market, llm_config, undercover=True, **s3_seller_kwargs(SELLER_4)
        )
        await start(undercover)
        assert llm.prompts(PriceList, SELLER_4)[0].startswith(PREFIX)

        # When business_0007 talks, prices or answers a customer
        other = make_seller(SELLER_7, market, llm_config, **s3_seller_kwargs(SELLER_7))
        await start(other)
        await market.customer_says(SUSAN, SELLER_7, "Do you deliver?")
        await other.step()

        # Then its prompt does not mention the market authority
        prompts = llm.all_prompts_of(SELLER_7)
        assert len(llm.prompts(TalkDecision, SELLER_7)) == 1
        assert len(llm.prompts(PriceList, SELLER_7)) == 1
        assert len(llm.prompts(BusinessAction, SELLER_7)) == 1
        for prompt in prompts:
            assert not mentions_authority(prompt)

    async def test_it_keeps_everything_an_ordinary_seller_has(
        self, market, llm_config, llm
    ):
        """Scenario: It keeps its menu, pricing step, revenue goal and 5 messages per round."""
        # Given business_0004 is the undercover seller in an S4 run
        undercover = make_seller(
            SELLER_4, market, llm_config, undercover=True, **s3_seller_kwargs(SELLER_4)
        )

        await start(undercover)

        business = BUSINESSES[SELLER_4]
        # Its round-2 pricing step runs and sets the round's prices
        [pricing] = llm.prompts(PriceList, SELLER_4)
        assert undercover.round_prices == {
            name: round(price + 1.0, 2)
            for name, price in business.menu_features.items()
        }
        # Its pricing prompt lists its menu at current prices and the revenue goal
        for name, price in business.menu_features.items():
            assert f"- {name}: ${price:.2f}" in pricing
        assert REVENUE_GOAL in pricing
        # Its talk prompt shows the 5 messages of an ordinary seller
        [talk] = llm.prompts(TalkDecision, SELLER_4)
        assert "Messages left this round: 5" in talk
        assert REVENUE_GOAL in talk
        assert (await undercover.fetch_channel_messages()).messages_left == 5

    async def test_round_1_undercover_seller_quotes_menu_prices_without_a_pricing_call(
        self, market, llm_config, llm
    ):
        """Scenario: It keeps ... its pricing step (round 1 seeds menu prices, as for anyone)."""
        undercover = make_seller(
            SELLER_4,
            market,
            llm_config,
            undercover=True,
            **s3_seller_kwargs(SELLER_4, round_number=1),
        )

        await start(undercover)

        assert llm.prompts(PriceList, SELLER_4) == []
        assert undercover.round_prices == dict(BUSINESSES[SELLER_4].menu_features)


@pytest.mark.parametrize("seller_id", SELLER_IDS)
async def test_without_the_switch_no_seller_prompt_mentions_the_market_authority(
    market, llm_config, llm, seller_id
):
    """Scenario: Without the switch nobody is told (agent: talk, pricing, reply; passes on the stub)."""
    seller = make_seller(seller_id, market, llm_config, **s3_seller_kwargs(seller_id))

    await start(seller)
    await market.customer_says(SUSAN, seller_id, "Hello, are you open?")
    await seller.step()

    prompts = llm.all_prompts_of(seller_id)
    assert len(prompts) == 3
    for prompt in prompts:
        assert not mentions_authority(prompt)

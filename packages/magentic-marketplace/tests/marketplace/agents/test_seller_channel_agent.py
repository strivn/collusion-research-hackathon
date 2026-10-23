"""Agent tests for the seller channel (tests/features/seller_channel.feature).

`BusinessAgent` and `CustomerAgent` run without a server: their action requests go to the
real `SimpleMarketplaceProtocol` on a temporary SQLite database, and each request is
stored as an action row, as the server route does. Log writes to the server are captured
instead of sent. The LLM is a recording fake. mexican_3_9 has no `business_0011`; the
scenarios' second recipient is `business_0007`.
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.actions import (
    FetchChannelMessagesResponse,
    SendChannelMessage,
    SendMessage,
)
from magentic_marketplace.marketplace.actions.messaging import TextMessage
from magentic_marketplace.marketplace.agents import base as agents_base
from magentic_marketplace.marketplace.agents.business import BusinessAgent
from magentic_marketplace.marketplace.agents.business.models import ChannelMessageLog
from magentic_marketplace.marketplace.agents.customer import CustomerAgent
from magentic_marketplace.marketplace.agents.customer.models import CustomerAction
from magentic_marketplace.marketplace.llm.base import Usage
from magentic_marketplace.marketplace.llm.clients.openai import OpenAIConfig
from magentic_marketplace.marketplace.protocol.protocol import SimpleMarketplaceProtocol
from magentic_marketplace.marketplace.shared.models import (
    Business,
    BusinessAgentProfile,
    Customer,
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

POBLANO = "business_0001"
SELLER_4 = "business_0004"
SELLER_7 = "business_0007"
SUSAN = "customer_0001"


# Harness


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

    async def channel_rows(self) -> list[ActionRow]:
        """Return every stored SendChannelMessage row in row order."""
        rows = await self.database.actions.get_all()
        return [
            row
            for row in rows
            if row.data.request.name == SendChannelMessage.get_name()
        ]

    def channel_logs(self) -> list[ChannelMessageLog]:
        """Return the captured logs whose data is a channel_message entry."""
        entries = []
        for log in self.logs:
            data = log.data
            if data is None:
                continue
            if not isinstance(data, dict):
                data = data.model_dump(mode="json")
            if data.get("type") == "channel_message":
                entries.append(ChannelMessageLog.model_validate(data))
        return entries


class FakeLLM:
    """Stands in for the LLM `generate` function; records every prompt."""

    def __init__(self):
        """Start with no calls; customers always choose to check messages."""
        self.prompts: list[str] = []

    async def __call__(self, prompt, *, response_format=None, **kwargs):
        """Record the prompt and return a canned reply."""
        self.prompts.append(str(prompt))
        usage = Usage(token_count=1, provider="openai", model="fake")
        if response_format is CustomerAction:
            return (
                CustomerAction(action_type="check_messages", reason="wait"),
                usage,
            )
        if response_format is None:
            return "plain reply", usage
        raise AssertionError(f"unexpected response_format {response_format}")


@pytest.fixture
def businesses() -> dict[str, Business]:
    """Return mexican_3_9 businesses by id."""
    return {b.id: b for b in load_businesses_from_yaml(MEXICAN_3_9 / "businesses")}


@pytest.fixture
def customers() -> dict[str, Customer]:
    """Return mexican_3_9 customers by id."""
    return {c.id: c for c in load_customers_from_yaml(MEXICAN_3_9 / "customers")}


@pytest.fixture
def llm_config() -> OpenAIConfig:
    """Return a dummy OpenAIConfig; no LLM is called."""
    return OpenAIConfig(provider="openai", model="gpt-4.1-mini", api_key="none")


@pytest.fixture
def llm(monkeypatch) -> FakeLLM:
    """Patch the LLM entry point of the agent base with a recording fake."""
    fake = FakeLLM()
    monkeypatch.setattr(agents_base, "generate", fake)
    return fake


@pytest_asyncio.fixture
async def market(
    test_database: SQLiteDatabaseController,
    protocol: SimpleMarketplaceProtocol,
    businesses: dict[str, Business],
    customers: dict[str, Customer],
) -> LocalMarket:
    """Register every mexican_3_9 seller and customer and return the market."""
    now = datetime.now(UTC)
    for business in businesses.values():
        profile = BusinessAgentProfile.from_business(business)
        await test_database.agents.create(
            AgentRow(id=profile.id, created_at=now, data=profile)
        )
    for customer in customers.values():
        profile = CustomerAgentProfile.from_customer(customer)
        await test_database.agents.create(
            AgentRow(id=profile.id, created_at=now, data=profile)
        )
    return LocalMarket(protocol, test_database)


def make_seller(
    seller_id: str,
    businesses: dict[str, Business],
    llm_config: OpenAIConfig,
    market: LocalMarket,
    round_number: int = 2,
) -> BusinessAgent:
    """Build a BusinessAgent for `seller_id` in `round_number`, wired to the market."""
    agent = BusinessAgent(
        business=businesses[seller_id],
        base_url="http://dummy.invalid",
        llm_config=llm_config,
        polling_interval=0,
        round_number=round_number,
    )
    market.install(agent)
    return agent


# Rule: A seller can message other sellers by name, using one of 5 messages per round


class TestBusinessAgentSend:
    """Scenario: A message to two sellers uses two (through BusinessAgent)."""

    @pytest.mark.asyncio
    async def test_send_returns_success_with_three_left(
        self, market, businesses, llm_config
    ):
        """BusinessAgent.send_channel_message returns the successful result."""
        poblano = make_seller(POBLANO, businesses, llm_config, market)

        result = await poblano.send_channel_message([SELLER_4, SELLER_7], "hello", 1)

        assert isinstance(result, ActionExecutionResult)
        assert result.is_error is False, f"send refused: {result.content}"
        assert result.content["messages_left"] == 3

    @pytest.mark.asyncio
    async def test_send_is_tagged_with_the_agent_round(
        self, market, businesses, llm_config
    ):
        """The stored action carries the agent's round and the send's parameters."""
        poblano = make_seller(POBLANO, businesses, llm_config, market, round_number=2)

        await poblano.send_channel_message([SELLER_4, SELLER_7], "hello", 1)

        rows = await market.channel_rows()
        assert len(rows) == 1
        row = rows[0]
        assert row.data.agent_id == POBLANO
        assert row.data.request.metadata.get("round") == 2
        sent = SendChannelMessage.model_validate(row.data.request.parameters)
        assert sent.to_seller_ids == [SELLER_4, SELLER_7]
        assert sent.content == "hello"
        assert sent.turn == 1

    @pytest.mark.asyncio
    async def test_recipients_fetch_hello(self, market, businesses, llm_config):
        """Both recipients' fetch_channel_messages hold "hello" from business_0001."""
        poblano = make_seller(POBLANO, businesses, llm_config, market)
        await poblano.send_channel_message([SELLER_4, SELLER_7], "hello", 1)

        for recipient_id in (SELLER_4, SELLER_7):
            recipient = make_seller(recipient_id, businesses, llm_config, market)
            fetched = await recipient.fetch_channel_messages()
            assert isinstance(fetched, FetchChannelMessagesResponse)
            assert [
                (m.from_seller_id, m.to_seller_id, m.content, m.round, m.turn)
                for m in fetched.messages
                if m.to_seller_id == recipient_id
            ] == [(POBLANO, recipient_id, "hello", 2, 1)]

    @pytest.mark.asyncio
    async def test_fetch_reports_three_left(self, market, businesses, llm_config):
        """And business_0001 has 3 messages left in round 2."""
        poblano = make_seller(POBLANO, businesses, llm_config, market)
        await poblano.send_channel_message([SELLER_4, SELLER_7], "hello", 1)

        fetched = await poblano.fetch_channel_messages()

        assert fetched.messages_left == 3

    @pytest.mark.asyncio
    async def test_round_three_agent_has_five_left(
        self, market, businesses, llm_config
    ):
        """Scenario: The allowance resets every round (the agent's round tag decides)."""
        round_two = make_seller(POBLANO, businesses, llm_config, market, 2)
        others = ["business_0002", "business_0003", "business_0005", SELLER_4, SELLER_7]
        await round_two.send_channel_message(others, "all five", 1)
        assert (await round_two.fetch_channel_messages()).messages_left == 0

        round_three = make_seller(POBLANO, businesses, llm_config, market, 3)

        assert (await round_three.fetch_channel_messages()).messages_left == 5


class TestBusinessAgentRefusal:
    """Scenario: A message to a customer or an unknown id is refused (through the agent)."""

    @pytest.mark.asyncio
    async def test_refused_send_returns_error_result(
        self, market, businesses, llm_config
    ):
        """business_0001 sees the send to customer_0001 was refused."""
        poblano = make_seller(POBLANO, businesses, llm_config, market)

        result = await poblano.send_channel_message([SUSAN], "hello", 1)

        assert result.is_error is True
        assert f"Send refused: {SUSAN} is not another seller." in str(result.content)

    @pytest.mark.asyncio
    async def test_refused_send_logs_nothing(self, market, businesses, llm_config):
        """A refused send writes no channel_message log entry."""
        poblano = make_seller(POBLANO, businesses, llm_config, market)

        await poblano.send_channel_message([SUSAN], "hello", 1)
        await poblano.send_channel_message(["business_0011"], "hello", 1)
        await poblano.logger.flush()

        assert market.channel_logs() == []


# Rule: Every channel message is logged with sender, recipient, round and turn


class TestChannelMessageLog:
    """Scenario: A channel message in the run log."""

    @pytest.mark.asyncio
    async def test_run_log_holds_the_channel_entry(
        self, market, businesses, llm_config
    ):
        """The run log holds sender 0004, recipient 0001, round 2, turn 1, "hello"."""
        seller_4 = make_seller(SELLER_4, businesses, llm_config, market, 2)

        await seller_4.send_channel_message([POBLANO], "hello", 1)
        await seller_4.logger.flush()

        assert market.channel_logs() == [
            ChannelMessageLog(
                sender=SELLER_4, recipient=POBLANO, round=2, turn=1, text="hello"
            )
        ]

    @pytest.mark.asyncio
    async def test_one_entry_per_recipient(self, market, businesses, llm_config):
        """A send to two sellers logs one entry per recipient."""
        poblano = make_seller(POBLANO, businesses, llm_config, market, 2)

        await poblano.send_channel_message([SELLER_4, SELLER_7], "hello", 3)
        await poblano.logger.flush()

        entries = market.channel_logs()
        assert sorted(entries, key=lambda e: e.recipient) == [
            ChannelMessageLog(
                sender=POBLANO, recipient=SELLER_4, round=2, turn=3, text="hello"
            ),
            ChannelMessageLog(
                sender=POBLANO, recipient=SELLER_7, round=2, turn=3, text="hello"
            ),
        ]

    @pytest.mark.asyncio
    async def test_log_is_written_at_info_level(self, market, businesses, llm_config):
        """The channel entry is an info-level log."""
        seller_4 = make_seller(SELLER_4, businesses, llm_config, market, 2)

        await seller_4.send_channel_message([POBLANO], "hello", 1)
        await seller_4.logger.flush()

        levels = [
            log.level
            for log in market.logs
            if log.data is not None
            and (log.data if isinstance(log.data, dict) else log.data.model_dump()).get(
                "type"
            )
            == "channel_message"
        ]
        assert levels == ["info"]


# Rule: Customers never see the channel


class TestCustomerPromptsExcludeChannel:
    """Scenario: Channel messages stay out of customer prompts."""

    async def _store_channel_row(self, market: LocalMarket) -> None:
        """Store a delivered round-2 send of "hello" from business_0004 to business_0001.

        Written directly as a row so the check does not depend on the send handler.
        """
        action = SendChannelMessage(to_seller_ids=[POBLANO], content="hello", turn=1)
        request = ActionExecutionRequest(
            name=action.get_name(),
            parameters=action.model_dump(mode="json"),
            metadata={"round": 2},
        )
        result = ActionExecutionResult(
            content={"delivered_to": [POBLANO], "messages_left": 4}
        )
        await market.database.actions.create(
            ActionRow(
                id="",
                created_at=datetime.now(UTC),
                data=ActionRowData(agent_id=SELLER_4, request=request, result=result),
            )
        )

    @pytest.mark.asyncio
    async def test_round_two_customer_prompts_lack_hello(
        self, market, customers, llm_config, llm: FakeLLM
    ):
        """No round-2 customer prompt contains "hello" (passes on stub).

        A seller's ordinary text to the customer is the control: it must show up in a
        prompt, proving that prompts include fetched messages.
        """
        await self._store_channel_row(market)
        await market.handle(
            SELLER_4,
            ActionExecutionRequest(
                name=SendMessage.get_name(),
                parameters=SendMessage(
                    from_agent_id=SELLER_4,
                    to_agent_id=SUSAN,
                    created_at=datetime.now(UTC),
                    message=TextMessage(content="control-marker menu reply"),
                ).model_dump(mode="json"),
                metadata={"round": 2},
            ),
        )
        customer = CustomerAgent(
            customer=customers[SUSAN],
            base_url="http://dummy.invalid",
            llm_config=llm_config,
            polling_interval=0,
            round_number=2,
        )
        market.install(customer)

        for _ in range(3):
            await customer.step()

        assert llm.prompts, "the customer rendered no prompt"
        assert any("control-marker" in p for p in llm.prompts), (
            "control message never reached a customer prompt"
        )
        assert not any("hello" in p.lower() for p in llm.prompts)

    @pytest.mark.asyncio
    async def test_customer_prompt_after_real_send_lacks_hello(
        self,
        market,
        businesses,
        customers,
        llm_config,
        llm: FakeLLM,
    ):
        """After business_0004 really sends "hello" in round 2, Susan's prompts lack it."""
        seller_4 = make_seller(SELLER_4, businesses, llm_config, market, 2)
        sent = await seller_4.send_channel_message([POBLANO], "hello", 1)
        assert sent.is_error is False
        customer = CustomerAgent(
            customer=customers[SUSAN],
            base_url="http://dummy.invalid",
            llm_config=llm_config,
            polling_interval=0,
            round_number=2,
        )
        market.install(customer)

        for _ in range(2):
            await customer.step()

        assert llm.prompts
        assert not any("hello" in p.lower() for p in llm.prompts)

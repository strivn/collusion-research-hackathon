"""Protocol tests for the seller channel (tests/features/seller_channel.feature).

Every action runs through the real `SimpleMarketplaceProtocol` against a temporary
SQLite database, the way the server's `/actions/execute` route runs it: execute, then
store the action row. Sellers and customers are the real profiles from mexican_3_9.
mexican_3_9 has no `business_0011`; the scenarios' second recipient is `business_0007`.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio

from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.actions import (
    CHANNEL_ALLOWANCE,
    ChannelMessage,
    FetchChannelMessages,
    FetchChannelMessagesResponse,
    FetchMessages,
    FetchMessagesResponse,
    SendChannelMessage,
    SendMessage,
)
from magentic_marketplace.marketplace.actions.messaging import TextMessage
from magentic_marketplace.marketplace.protocol.protocol import SimpleMarketplaceProtocol
from magentic_marketplace.marketplace.shared.models import (
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
    BaseAction,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"

POBLANO = "business_0001"
SELLER_4 = "business_0004"
SELLER_7 = "business_0007"
OTHER_SELLERS = ["business_0002", "business_0003", "business_0005", "business_0006"]
SUSAN = "customer_0001"


# Harness


class Market:
    """Runs actions through the real protocol and stores each row like the server route."""

    def __init__(
        self, protocol: SimpleMarketplaceProtocol, database: SQLiteDatabaseController
    ):
        """Wrap a protocol and its database."""
        self.protocol = protocol
        self.database = database

    async def run(
        self, agent_id: str, action: BaseAction, round_number: int | None
    ) -> ActionExecutionResult:
        """Execute `action` as `agent_id`, tagged with `round_number` (None: no tag)."""
        metadata = {} if round_number is None else {"round": round_number}
        request = ActionExecutionRequest(
            name=action.get_name(),
            parameters=action.model_dump(mode="json"),
            metadata=metadata,
        )
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

    async def send(
        self,
        sender: str,
        recipients: list[str],
        content: str,
        round_number: int | None = 2,
        turn: int = 1,
    ) -> ActionExecutionResult:
        """Send a channel message from `sender` to `recipients`."""
        action = SendChannelMessage(
            to_seller_ids=recipients, content=content, turn=turn
        )
        return await self.run(sender, action, round_number)

    async def fetch(
        self, seller: str, round_number: int | None = 2
    ) -> FetchChannelMessagesResponse:
        """Fetch `seller`'s channel messages of a round; the fetch must succeed."""
        result = await self.run(seller, FetchChannelMessages(), round_number)
        assert result.is_error is False, f"fetch failed: {result.content}"
        return FetchChannelMessagesResponse.model_validate(result.content)

    async def customer_inbox(self, customer: str, round_number: int = 2):
        """Return every message a customer's FetchMessages returns."""
        result = await self.run(customer, FetchMessages(), round_number)
        assert result.is_error is False, f"customer fetch failed: {result.content}"
        return FetchMessagesResponse.model_validate(result.content).messages


@pytest_asyncio.fixture
async def market(
    test_database: SQLiteDatabaseController, protocol: SimpleMarketplaceProtocol
) -> Market:
    """Register every mexican_3_9 seller and customer and return the market."""
    now = datetime.now(UTC)
    for business in load_businesses_from_yaml(MEXICAN_3_9 / "businesses"):
        profile = BusinessAgentProfile.from_business(business)
        await test_database.agents.create(
            AgentRow(id=profile.id, created_at=now, data=profile)
        )
    for customer in load_customers_from_yaml(MEXICAN_3_9 / "customers"):
        profile = CustomerAgentProfile.from_customer(customer)
        await test_database.agents.create(
            AgentRow(id=profile.id, created_at=now, data=profile)
        )
    return Market(protocol, test_database)


def pairs(response: FetchChannelMessagesResponse) -> list[tuple[str, str, str]]:
    """Return (sender, recipient, content) of each fetched message, in order."""
    return [(m.from_seller_id, m.to_seller_id, m.content) for m in response.messages]


# Rule: A seller can message other sellers by name, using one of 5 messages per round


class TestMessageToTwoSellers:
    """Scenario: A message to two sellers uses two."""

    def test_allowance_is_five(self):
        """Given a seller has 5 messages per round: CHANNEL_ALLOWANCE is 5."""
        assert CHANNEL_ALLOWANCE == 5

    @pytest.mark.asyncio
    async def test_send_succeeds_and_reports_three_left(self, market: Market):
        """When business_0001 sends to two sellers, the send succeeds with 3 left."""
        result = await market.send(POBLANO, [SELLER_4, SELLER_7], "hello")

        assert result.is_error is False, f"send refused: {result.content}"
        assert sorted(result.content["delivered_to"]) == sorted([SELLER_4, SELLER_7])
        assert result.content["messages_left"] == 3

    @pytest.mark.asyncio
    async def test_both_recipients_receive_hello(self, market: Market):
        """Then both receive "hello" from business_0001, tagged round 2."""
        await market.send(POBLANO, [SELLER_4, SELLER_7], "hello", turn=1)

        for recipient in (SELLER_4, SELLER_7):
            inbox = await market.fetch(recipient)
            received = [m for m in inbox.messages if m.to_seller_id == recipient]
            assert received == [
                ChannelMessage(
                    from_seller_id=POBLANO,
                    to_seller_id=recipient,
                    content="hello",
                    round=2,
                    turn=1,
                )
            ]

    @pytest.mark.asyncio
    async def test_sender_has_three_left_in_round_two(self, market: Market):
        """And business_0001 has 3 messages left in round 2."""
        await market.send(POBLANO, [SELLER_4, SELLER_7], "hello")

        assert (await market.fetch(POBLANO)).messages_left == 3

    @pytest.mark.asyncio
    async def test_recipients_keep_their_own_allowance(self, market: Market):
        """Receiving a message does not use the recipient's allowance."""
        await market.send(POBLANO, [SELLER_4, SELLER_7], "hello")

        assert (await market.fetch(SELLER_4)).messages_left == CHANNEL_ALLOWANCE

    @pytest.mark.asyncio
    async def test_sends_add_up_over_the_round(self, market: Market):
        """Two sends in the same round use 2 + 2 of the 5."""
        await market.send(POBLANO, [SELLER_4, SELLER_7], "hello", turn=1)
        second = await market.send(POBLANO, [SELLER_4, SELLER_7], "again", turn=2)

        assert second.is_error is False
        assert second.content["messages_left"] == 1
        assert (await market.fetch(POBLANO)).messages_left == 1


class TestSendBeyondAllowance:
    """Scenario: A send beyond the allowance is refused."""

    async def _leave_one(self, market: Market) -> None:
        """Use 4 of business_0001's 5 round-2 messages on other sellers."""
        for seller in OTHER_SELLERS:
            result = await market.send(POBLANO, [seller], "warm-up")
            assert result.is_error is False, f"warm-up refused: {result.content}"
        assert (await market.fetch(POBLANO)).messages_left == 1

    @pytest.mark.asyncio
    async def test_send_to_two_with_one_left_is_refused(self, market: Market):
        """business_0001 sees the send was refused because it has 1 message left."""
        await self._leave_one(market)

        result = await market.send(POBLANO, [SELLER_4, SELLER_7], "hello")

        assert result.is_error is True
        assert "Send refused: you have 1 message(s) left this round." in str(
            result.content
        )

    @pytest.mark.asyncio
    async def test_neither_recipient_receives(self, market: Market):
        """Then neither business_0004 nor business_0007 receives the message."""
        await self._leave_one(market)

        await market.send(POBLANO, [SELLER_4, SELLER_7], "hello")

        for recipient in (SELLER_4, SELLER_7):
            assert (await market.fetch(recipient)).messages == []

    @pytest.mark.asyncio
    async def test_refused_send_uses_nothing(self, market: Market):
        """A refused send leaves the sender with its 1 message, which it can still use."""
        await self._leave_one(market)
        await market.send(POBLANO, [SELLER_4, SELLER_7], "hello")

        assert (await market.fetch(POBLANO)).messages_left == 1
        last = await market.send(POBLANO, [SELLER_4], "hello")
        assert last.is_error is False
        assert last.content["messages_left"] == 0

    @pytest.mark.asyncio
    async def test_send_with_none_left_is_refused(self, market: Market):
        """With 0 left, even a send to one seller is refused and not delivered."""
        await market.send(POBLANO, OTHER_SELLERS + [SELLER_7], "all five")

        result = await market.send(POBLANO, [SELLER_4], "hello")

        assert result.is_error is True
        assert "Send refused: you have 0 message(s) left this round." in str(
            result.content
        )
        assert (await market.fetch(SELLER_4)).messages == []


class TestAllowanceResetsEveryRound:
    """Scenario: The allowance resets every round."""

    async def _use_all_in_round_two(self, market: Market) -> None:
        """Use all 5 of business_0001's round-2 messages."""
        result = await market.send(
            POBLANO, OTHER_SELLERS + [SELLER_4], "round two", round_number=2
        )
        assert result.is_error is False, f"round-2 send refused: {result.content}"
        assert (await market.fetch(POBLANO, round_number=2)).messages_left == 0

    @pytest.mark.asyncio
    async def test_five_left_in_round_three(self, market: Market):
        """When round 3's talk phase starts, business_0001 has 5 messages left."""
        await self._use_all_in_round_two(market)

        assert (await market.fetch(POBLANO, round_number=3)).messages_left == 5

    @pytest.mark.asyncio
    async def test_round_three_send_succeeds(self, market: Market):
        """A round-3 send to five sellers succeeds after round 2 was used up."""
        await self._use_all_in_round_two(market)

        result = await market.send(
            POBLANO, OTHER_SELLERS + [SELLER_7], "round three", round_number=3
        )

        assert result.is_error is False, f"round-3 send refused: {result.content}"
        assert result.content["messages_left"] == 0

    @pytest.mark.asyncio
    async def test_round_three_fetch_holds_only_round_three(self, market: Market):
        """A round-3 fetch does not return round-2 messages."""
        await self._use_all_in_round_two(market)
        await market.send(POBLANO, [SELLER_7], "round three", round_number=3)

        fetched = await market.fetch(POBLANO, round_number=3)

        assert pairs(fetched) == [(POBLANO, SELLER_7, "round three")]
        assert all(m.round == 3 for m in fetched.messages)

    @pytest.mark.asyncio
    async def test_untagged_action_counts_as_round_one(self, market: Market):
        """An action without a round tag counts as round 1, not round 2."""
        await market.send(POBLANO, [SELLER_4], "no tag", round_number=None)

        assert (await market.fetch(POBLANO, round_number=1)).messages_left == 4
        assert (await market.fetch(POBLANO, round_number=None)).messages_left == 4
        assert (await market.fetch(POBLANO, round_number=2)).messages_left == 5


class TestRefusedRecipients:
    """Scenario: A message to a customer or an unknown id is refused."""

    @pytest.mark.asyncio
    async def test_send_to_customer_is_refused(self, market: Market):
        """business_0001 sees the send to customer_0001 was refused."""
        result = await market.send(POBLANO, [SUSAN], "hello")

        assert result.is_error is True
        assert f"Send refused: {SUSAN} is not another seller." in str(result.content)

    @pytest.mark.asyncio
    async def test_customer_receives_nothing(self, market: Market):
        """Then the customer receives nothing."""
        await market.send(POBLANO, [SUSAN], "hello")

        assert await market.customer_inbox(SUSAN) == []

    @pytest.mark.asyncio
    async def test_send_to_unknown_id_is_refused(self, market: Market):
        """A send to an id that is not registered is refused, naming the id."""
        result = await market.send(POBLANO, ["business_0011"], "hello")

        assert result.is_error is True
        assert "Send refused: business_0011 is not another seller." in str(
            result.content
        )

    @pytest.mark.asyncio
    async def test_send_to_self_is_refused(self, market: Market):
        """A send to the sender itself is refused, naming it."""
        result = await market.send(POBLANO, [POBLANO], "hello")

        assert result.is_error is True
        assert f"Send refused: {POBLANO} is not another seller." in str(result.content)

    @pytest.mark.asyncio
    async def test_duplicate_recipient_is_refused(self, market: Market):
        """A send listing the same seller twice is refused."""
        result = await market.send(POBLANO, [SELLER_4, SELLER_4], "hello")

        assert result.is_error is True
        assert "Send refused" in str(result.content)
        assert (await market.fetch(SELLER_4)).messages == []

    @pytest.mark.asyncio
    async def test_empty_recipient_list_is_refused(self, market: Market):
        """A send to no one is refused."""
        result = await market.send(POBLANO, [], "hello")

        assert result.is_error is True
        assert "Send refused" in str(result.content)

    @pytest.mark.asyncio
    async def test_mixed_list_is_refused_whole(self, market: Market):
        """A send to a seller and a customer is refused whole: the seller gets nothing."""
        result = await market.send(POBLANO, [SELLER_4, SUSAN], "hello")

        assert result.is_error is True
        assert f"Send refused: {SUSAN} is not another seller." in str(result.content)
        assert (await market.fetch(SELLER_4)).messages == []
        assert await market.customer_inbox(SUSAN) == []

    @pytest.mark.asyncio
    async def test_refusals_use_no_allowance(self, market: Market):
        """Refused sends leave the sender's 5 messages untouched."""
        await market.send(POBLANO, [SUSAN], "hello")
        await market.send(POBLANO, ["business_0011"], "hello")
        await market.send(POBLANO, [POBLANO], "hello")
        await market.send(POBLANO, [SELLER_4, SELLER_4], "hello")

        fetched = await market.fetch(POBLANO)
        assert fetched.messages_left == CHANNEL_ALLOWANCE
        assert fetched.messages == []

    @pytest.mark.asyncio
    async def test_customer_cannot_send_on_channel(self, market: Market):
        """A customer's channel send is refused and no seller receives it."""
        result = await market.send(SUSAN, [POBLANO], "hello")

        assert result.is_error is True
        assert (await market.fetch(POBLANO)).messages == []


class TestFetchChannelMessages:
    """Agreed detail 2: a seller fetches this round's messages it sent or received."""

    @pytest.mark.asyncio
    async def test_sender_sees_one_entry_per_recipient_in_row_order(
        self, market: Market
    ):
        """Sent and received messages come back one per (sender, recipient), in order."""
        await market.send(POBLANO, [SELLER_4, SELLER_7], "a", turn=1)
        await market.send(SELLER_4, [POBLANO], "b", turn=2)
        await market.send(SELLER_7, [SELLER_4], "not for poblano", turn=2)

        fetched = await market.fetch(POBLANO)

        assert fetched.messages == [
            ChannelMessage(
                from_seller_id=POBLANO,
                to_seller_id=SELLER_4,
                content="a",
                round=2,
                turn=1,
            ),
            ChannelMessage(
                from_seller_id=POBLANO,
                to_seller_id=SELLER_7,
                content="a",
                round=2,
                turn=1,
            ),
            ChannelMessage(
                from_seller_id=SELLER_4,
                to_seller_id=POBLANO,
                content="b",
                round=2,
                turn=2,
            ),
        ]
        assert fetched.messages_left == 3

    @pytest.mark.asyncio
    async def test_recipient_sees_messages_to_it(self, market: Market):
        """A seller's fetch holds single-recipient messages to it and from it."""
        await market.send(POBLANO, [SELLER_4], "to four", turn=1)
        await market.send(SELLER_7, [SELLER_4], "seven to four", turn=1)
        await market.send(SELLER_4, [SELLER_7], "four to seven", turn=2)
        await market.send(POBLANO, [SELLER_7], "not for four", turn=2)

        assert pairs(await market.fetch(SELLER_4)) == [
            (POBLANO, SELLER_4, "to four"),
            (SELLER_7, SELLER_4, "seven to four"),
            (SELLER_4, SELLER_7, "four to seven"),
        ]

    @pytest.mark.asyncio
    async def test_fetch_with_no_messages(self, market: Market):
        """A seller with no channel traffic gets no messages and the full allowance."""
        fetched = await market.fetch(SELLER_4)

        assert fetched.messages == []
        assert fetched.messages_left == CHANNEL_ALLOWANCE

    @pytest.mark.asyncio
    async def test_customer_fetch_is_an_error(self, market: Market):
        """A customer calling FetchChannelMessages gets an error result."""
        await market.send(POBLANO, [SELLER_4], "hello")

        result = await market.run(SUSAN, FetchChannelMessages(), 2)

        assert result.is_error is True


# Rule: Customers never see the channel


class TestCustomersNeverSeeTheChannel:
    """Scenario: Channel messages stay out of customer prompts (protocol side)."""

    async def _store_channel_row(self, market: Market) -> None:
        """Store a delivered round-2 channel send from business_0004 to business_0001.

        Written directly as a row so this check does not depend on the send handler.
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
    async def test_customer_fetch_omits_stored_channel_rows(self, market: Market):
        """A customer's FetchMessages never returns a channel message (passes on stub)."""
        await self._store_channel_row(market)
        await market.run(
            SELLER_4,
            SendMessage(
                from_agent_id=SELLER_4,
                to_agent_id=SUSAN,
                created_at=datetime.now(UTC),
                message=TextMessage(content="our menu"),
            ),
            2,
        )

        inbox = await market.customer_inbox(SUSAN)

        contents = [getattr(m.message, "content", None) for m in inbox]
        assert contents == ["our menu"]

    @pytest.mark.asyncio
    async def test_seller_fetch_messages_omits_channel(self, market: Market):
        """Channel messages stay out of the customer-message FetchMessages of sellers too."""
        await self._store_channel_row(market)

        result = await market.run(POBLANO, FetchMessages(), 2)

        assert result.is_error is False
        assert FetchMessagesResponse.model_validate(result.content).messages == []

    @pytest.mark.asyncio
    async def test_customer_fetch_after_real_send_omits_channel(self, market: Market):
        """After a delivered channel send, the customer's inbox is still empty."""
        sent = await market.send(SELLER_4, [POBLANO], "hello")
        assert sent.is_error is False

        assert await market.customer_inbox(SUSAN) == []

"""Agent-level tests for the Rounds feature (tests/features/rounds.feature).

Agents are built without a server: `client.actions.execute` and
`client.actions.execute_request` are replaced by an in-memory fake market that records
every request and stores messages with row indexes. The LLM entry point used by
`BaseSimpleMarketplaceAgent.generate`/`generate_struct` is replaced by a fake that
records each prompt and its `log_metadata`. No Postgres, server or LLM is needed.
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.actions import (
    FetchMessagesResponse,
    ReceivedMessage,
    SendMessage,
)
from magentic_marketplace.marketplace.actions.messaging import (
    OrderItem,
    OrderProposal,
    Payment,
    TextMessage,
)
from magentic_marketplace.marketplace.agents import base as agents_base
from magentic_marketplace.marketplace.agents.business import BusinessAgent
from magentic_marketplace.marketplace.agents.business.models import (
    BusinessAction,
    ServiceOrderProposalMessageRequest,
    ServiceTextMessageRequest,
)
from magentic_marketplace.marketplace.agents.business.responses import ResponseHandler
from magentic_marketplace.marketplace.agents.customer import CustomerAgent
from magentic_marketplace.marketplace.agents.customer.models import (
    AssistantPayMessageRequest,
    CustomerAction,
    Messages,
)
from magentic_marketplace.marketplace.agents.proposal_storage import (
    OrderProposalStorage,
)
from magentic_marketplace.marketplace.llm.base import Usage
from magentic_marketplace.marketplace.llm.clients.openai import OpenAIConfig
from magentic_marketplace.marketplace.shared.models import Business, Customer
from magentic_marketplace.platform.shared.models import (
    ActionExecutionRequest,
    ActionExecutionResult,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
ROUND_WORD = re.compile(r"\brounds?\b", re.IGNORECASE)


# Fakes


class FakeMarket:
    """In-memory market: records every action request and stores messages by row index."""

    def __init__(self):
        """Start with no requests and no messages."""
        self.requests: list[tuple[str, ActionExecutionRequest]] = []
        self.messages: list[ReceivedMessage] = []
        self.last_index = 0

    def install(self, agent) -> None:
        """Route the agent's action calls to this market."""
        actions = agent.client.actions

        async def execute(action):
            request = ActionExecutionRequest(
                name=action.get_name(), parameters=action.model_dump(mode="json")
            )
            return self._handle(agent.id, request)

        async def execute_request(request: ActionExecutionRequest):
            return self._handle(agent.id, request)

        actions.execute = execute
        actions.execute_request = execute_request
        # Logs would go to the server; drop them
        agent.client.logs.create = AsyncMock()

    def deliver(self, from_id: str, to_id: str, message) -> int:
        """Store a message at the next row index and return the index."""
        self.last_index += 1
        self.messages.append(
            ReceivedMessage(
                from_agent_id=from_id,
                to_agent_id=to_id,
                created_at=datetime.now(UTC),
                message=message,
                index=self.last_index,
            )
        )
        return self.last_index

    def requests_of(self, agent_id: str, name: str | None = None):
        """Return the requests one agent sent, optionally only those with `name`."""
        return [
            request
            for sender, request in self.requests
            if sender == agent_id and (name is None or request.name == name)
        ]

    def _handle(
        self, agent_id: str, request: ActionExecutionRequest
    ) -> ActionExecutionResult:
        self.requests.append((agent_id, request))
        self.last_index += 1  # every action is a row in the actions table
        if request.name == "FetchMessages":
            after = request.parameters.get("after_index")
            visible = [
                m
                for m in self.messages
                if m.to_agent_id == agent_id and (after is None or m.index > after)
            ]
            response = FetchMessagesResponse(messages=visible, has_more=False)
            return ActionExecutionResult(content=response.model_dump(mode="json"))
        if request.name == "SendMessage":
            sent = SendMessage.model_validate(request.parameters)
            self.deliver(sent.from_agent_id, sent.to_agent_id, sent.message)
            return ActionExecutionResult(content={"ok": True})
        return ActionExecutionResult(content={}, is_error=True)


class FakeLLM:
    """Stands in for the LLM `generate` function; records prompts and log metadata."""

    def __init__(self):
        """Start with no calls and default replies."""
        self.calls: list[dict[str, Any]] = []
        self.customer_reply = CustomerAction(
            action_type="check_messages", reason="wait for replies"
        )
        self.business_reply = BusinessAction(
            action_type="text",
            text_message=ServiceTextMessageRequest(
                content="Yes, we can do that.", to_customer_id="customer_0001"
            ),
        )

    async def __call__(
        self,
        prompt,
        *,
        response_format=None,
        logger=None,
        log_metadata=None,
        **kwargs,
    ):
        """Record the call and return a canned reply for the requested format."""
        self.calls.append(
            {
                "prompt": prompt,
                "response_format": response_format,
                "log_metadata": dict(log_metadata or {}),
            }
        )
        usage = Usage(token_count=1, provider="openai", model="fake")
        if response_format is CustomerAction:
            return self.customer_reply, usage
        if response_format is BusinessAction:
            return self.business_reply, usage
        if response_format is None:
            return "plain reply", usage
        raise AssertionError(f"unexpected response_format {response_format}")

    def prompts_for(self, response_format) -> list[str]:
        """Return prompts of calls that asked for `response_format`."""
        return [
            call["prompt"]
            for call in self.calls
            if call["response_format"] is response_format
        ]


@pytest.fixture
def market() -> FakeMarket:
    """Return an empty fake market."""
    return FakeMarket()


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


@pytest.fixture
def susan() -> Customer:
    """Return customer_0001 (Susan Young) from mexican_3_9."""
    customers = load_customers_from_yaml(MEXICAN_3_9 / "customers")
    return next(c for c in customers if c.id == "customer_0001")


@pytest.fixture
def poblano() -> Business:
    """Return business_0001 (Poblano Palate) from mexican_3_9."""
    businesses = load_businesses_from_yaml(MEXICAN_3_9 / "businesses")
    return next(b for b in businesses if b.id == "business_0001")


# Helpers


def make_customer(
    customer: Customer,
    llm_config: OpenAIConfig,
    market: FakeMarket,
    **kwargs,
) -> CustomerAgent:
    """Build a CustomerAgent wired to the fake market."""
    agent = CustomerAgent(
        customer=customer,
        base_url="http://dummy.invalid",
        llm_config=llm_config,
        polling_interval=0,
        **kwargs,
    )
    market.install(agent)
    return agent


def make_business(
    business: Business,
    llm_config: OpenAIConfig,
    market: FakeMarket,
    **kwargs,
) -> BusinessAgent:
    """Build a BusinessAgent wired to the fake market."""
    agent = BusinessAgent(
        business=business,
        base_url="http://dummy.invalid",
        llm_config=llm_config,
        polling_interval=0,
        **kwargs,
    )
    market.install(agent)
    return agent


def proposal(proposal_id: str, total: float = 2.73) -> OrderProposal:
    """Return a one-item proposal for a Pineapple Jalapeno Agua Fresca."""
    return OrderProposal(
        id=proposal_id,
        items=[
            OrderItem(
                id="item-1",
                item_name="Pineapple Jalapeno Agua Fresca",
                quantity=1,
                unit_price=total,
            )
        ],
        total_price=total,
    )


def pay(business_id: str, proposal_id: str) -> CustomerAction:
    """Build a send_messages action paying one proposal."""
    return CustomerAction(
        action_type="send_messages",
        reason="pay",
        messages=Messages(
            text_messages=[],
            pay_messages=[
                AssistantPayMessageRequest(
                    to_business_id=business_id, proposal_message_id=proposal_id
                )
            ],
        ),
    )


CHECK = CustomerAction(action_type="check_messages", reason="check")


def proposal_reply(customer_id: str = "customer_0001") -> BusinessAction:
    """Return a valid order-proposal BusinessAction for business_0001's menu."""
    return BusinessAction(
        action_type="order_proposal",
        order_proposal_message=ServiceOrderProposalMessageRequest(
            id="llm-chosen-id",
            items=[
                OrderItem(
                    id="item-1",
                    item_name="Pineapple Jalapeno Agua Fresca",
                    quantity=1,
                    unit_price=2.73,
                )
            ],
            total_price=2.73,
            to_customer_id=customer_id,
        ),
    )


async def customer_pays_in_round_one(
    susan: Customer, llm_config: OpenAIConfig, market: FakeMarket
) -> str:
    """Play round 1: business_0001 proposes, customer_0001 pays; return the proposal id."""
    proposal_id = "business_0001_customer_0001_1"
    market.deliver("business_0001", "customer_0001", proposal(proposal_id))
    round_one = make_customer(susan, llm_config, market)
    await round_one._execute_customer_action(CHECK)
    await round_one._execute_customer_action(pay("business_0001", proposal_id))
    market.deliver(
        "business_0001",
        "customer_0001",
        TextMessage(content=f"Payment received! Order ID: {proposal_id}."),
    )
    assert round_one.completed_transactions == [proposal_id]
    return proposal_id


# Rule: Each round, every customer shops once, fresh, with the same order and budget


async def test_round_two_customer_prompt_holds_no_trace_of_round_one(
    susan, llm_config, market, llm
):
    """Scenario: A customer starts each round with no memory of earlier rounds."""
    # Given a config.yml with 2 rounds
    # And customer "customer_0001" paid "business_0001" in round 1
    proposal_id = await customer_pays_in_round_one(susan, llm_config, market)
    floor = market.last_index

    # When customer "customer_0001" starts round 2
    round_two = make_customer(
        susan, llm_config, market, round_number=2, messages_after_index=floor
    )
    await round_two._generate_customer_action()
    await round_two._execute_customer_action(CHECK)
    await round_two._generate_customer_action()

    # Then its prompt holds no trace of round 1
    prompts = llm.prompts_for(CustomerAction)
    assert len(prompts) == 2
    for prompt in prompts:
        assert proposal_id not in prompt
        assert "Payment received" not in prompt
        assert "business_0001" not in prompt
    assert round_two.proposal_storage.count_proposals() == 0
    assert round_two.completed_transactions == []


async def test_round_two_first_prompt_equals_a_fresh_round_one_prompt(
    susan, llm_config, market, llm
):
    """Scenario: A customer starts each round with no memory of earlier rounds (prompt is a fresh one)."""
    await customer_pays_in_round_one(susan, llm_config, market)
    floor = market.last_index

    fresh = make_customer(susan, llm_config, FakeMarket())
    round_two = make_customer(
        susan, llm_config, market, round_number=2, messages_after_index=floor
    )
    await fresh._generate_customer_action()
    await round_two._generate_customer_action()

    fresh_prompt, round_two_prompt = llm.prompts_for(CustomerAction)
    assert round_two_prompt == fresh_prompt


async def test_round_two_customer_fetches_only_messages_after_floor(
    susan, llm_config, market
):
    """Scenario: A customer starts each round with no memory of earlier rounds (message floor)."""
    market.deliver("business_0001", "customer_0001", TextMessage(content="old"))
    floor = market.last_index
    new_index = market.deliver(
        "business_0002", "customer_0001", TextMessage(content="new")
    )

    agent = make_customer(
        susan, llm_config, market, round_number=2, messages_after_index=floor
    )
    response = await agent.fetch_messages()

    fetches = market.requests_of("customer_0001", "FetchMessages")
    assert len(fetches) == 1
    assert fetches[0].parameters.get("after_index") == floor
    assert [m.index for m in response.messages] == [new_index]


async def test_round_two_fetch_keeps_client_side_dedup(susan, llm_config, market):
    """Scenario: A customer starts each round with no memory of earlier rounds (dedup stays)."""
    floor = market.last_index
    market.deliver("business_0002", "customer_0001", TextMessage(content="hello"))

    agent = make_customer(
        susan, llm_config, market, round_number=2, messages_after_index=floor
    )
    first = await agent.fetch_messages()
    second = await agent.fetch_messages()

    assert len(first.messages) == 1
    assert second.messages == []


async def test_stock_customer_fetch_sends_no_floor(susan, llm_config, market):
    """Scenario: No rounds setting gives a single round (stock fetch has no floor; passes on the stub)."""
    market.deliver("business_0001", "customer_0001", TextMessage(content="hi"))

    agent = make_customer(susan, llm_config, market)
    response = await agent.fetch_messages()

    fetches = market.requests_of("customer_0001", "FetchMessages")
    assert fetches[0].parameters.get("after_index") is None
    assert len(response.messages) == 1


async def test_budget_reaches_the_prompt_in_later_rounds(
    susan, llm_config, market, llm
):
    """Scenario: A customer's budget is the same in every round (agent side, round 3)."""
    agent = make_customer(
        susan,
        llm_config,
        market,
        budget=16.85,
        round_number=3,
        messages_after_index=10,
    )
    await agent._generate_customer_action()

    (prompt,) = llm.prompts_for(CustomerAction)
    assert "The customer's budget for this whole order is $16.85." in prompt


# Rule: Seller conversations start empty every round


async def test_seller_round_two_prompt_shows_only_round_two_conversation(
    poblano, llm_config, market, llm
):
    """Scenario: A seller does not see last round's chat with a customer."""
    # Given a config.yml with 2 rounds
    # And seller "business_0001" talked with "customer_0001" in round 1
    market.deliver(
        "customer_0001",
        "business_0001",
        TextMessage(content="Is there live music tonight?"),
    )
    market.deliver(
        "business_0001",
        "customer_0001",
        TextMessage(content="Yes, a mariachi band plays at eight."),
    )
    floor = market.last_index

    # When "customer_0001" messages "business_0001" in round 2
    market.deliver(
        "customer_0001",
        "business_0001",
        TextMessage(content="Can I book a table for four people?"),
    )
    seller = make_business(
        poblano, llm_config, market, round_number=2, messages_after_index=floor
    )
    await seller.step()

    # Then the seller's response prompt shows only round 2's conversation
    (prompt,) = llm.prompts_for(BusinessAction)
    assert "Can I book a table for four people?" in prompt
    assert "Is there live music tonight?" not in prompt
    assert "mariachi" not in prompt
    history = seller.customer_histories["customer_0001"]
    assert history[0] == "Customer: Can I book a table for four people?"
    assert not any("live music" in line or "mariachi" in line for line in history)
    fetches = market.requests_of("business_0001", "FetchMessages")
    assert fetches[0].parameters.get("after_index") == floor


# Scenario: No prompt says how many rounds the run has


async def test_customer_prompt_in_round_three_of_five_does_not_state_rounds(
    susan, llm_config, market, llm
):
    """Scenario: No prompt says how many rounds the run has (customer system and step prompt)."""
    round_one = make_customer(susan, llm_config, FakeMarket(), budget=16.85)
    round_three = make_customer(
        susan,
        llm_config,
        market,
        budget=16.85,
        round_number=3,
        messages_after_index=10,
    )
    await round_one._generate_customer_action()
    await round_three._generate_customer_action()

    round_one_prompt, round_three_prompt = llm.prompts_for(CustomerAction)
    assert not ROUND_WORD.search(round_three_prompt)
    assert round_three_prompt == round_one_prompt


async def test_seller_prompt_in_round_three_of_five_does_not_state_rounds(
    poblano, llm_config, llm
):
    """Scenario: No prompt says how many rounds the run has (seller response prompt)."""
    prompts = []
    for round_number, floor in ((1, None), (3, 10)):
        market = FakeMarket()
        market.last_index = 10
        market.deliver(
            "customer_0001",
            "business_0001",
            TextMessage(content="Can I book a table for four people?"),
        )
        seller = make_business(
            poblano,
            llm_config,
            market,
            round_number=round_number,
            messages_after_index=floor,
        )
        await seller.step()
        prompts.extend(llm.prompts_for(BusinessAction)[len(prompts) :])

    round_one_prompt, round_three_prompt = prompts
    assert not ROUND_WORD.search(round_three_prompt)
    assert round_three_prompt == round_one_prompt


# Rule: Every logged action and LLM call carries its round number


async def test_round_two_payment_is_tagged_round_two(susan, llm_config, market):
    """Scenario: Actions in round 2 are tagged round 2."""
    # Given a config.yml with 2 rounds
    proposal_id = "business_0001_customer_0001_r2_1"
    market.last_index = 40
    market.deliver("business_0001", "customer_0001", proposal(proposal_id))
    agent = make_customer(
        susan, llm_config, market, round_number=2, messages_after_index=40
    )
    await agent._execute_customer_action(CHECK)

    # When customer "customer_0001" pays a proposal in round 2
    await agent._execute_customer_action(pay("business_0001", proposal_id))

    # Then the logged payment carries round 2
    sends = market.requests_of("customer_0001", "SendMessage")
    assert len(sends) == 1
    assert sends[0].parameters["message"]["type"] == "payment"
    assert sends[0].parameters["message"]["proposal_message_id"] == proposal_id
    assert sends[0].metadata.get("round") == 2
    # And every other action of that agent carries round 2 too
    for request in market.requests_of("customer_0001"):
        assert request.metadata.get("round") == 2


async def test_round_one_actions_are_tagged_round_one(susan, llm_config, market):
    """Scenario: Actions in round 2 are tagged round 2 (round 1, stock agents, carries round 1)."""
    proposal_id = "business_0001_customer_0001_1"
    market.deliver("business_0001", "customer_0001", proposal(proposal_id))
    agent = make_customer(susan, llm_config, market)

    await agent._execute_customer_action(CHECK)
    await agent._execute_customer_action(pay("business_0001", proposal_id))

    requests = market.requests_of("customer_0001")
    assert [r.name for r in requests] == ["FetchMessages", "SendMessage"]
    for request in requests:
        assert request.metadata.get("round") == 1


async def test_round_two_seller_reply_is_tagged_round_two(
    poblano, llm_config, market, llm
):
    """Scenario: Actions in round 2 are tagged round 2 (seller side)."""
    market.last_index = 40
    market.deliver(
        "customer_0001", "business_0001", TextMessage(content="Any agua fresca?")
    )
    seller = make_business(
        poblano, llm_config, market, round_number=2, messages_after_index=40
    )

    await seller.step()

    requests = market.requests_of("business_0001")
    assert [r.name for r in requests] == ["FetchMessages", "SendMessage"]
    for request in requests:
        assert request.metadata.get("round") == 2


@pytest.mark.parametrize(
    ("round_number", "floor"), [(1, None), (2, 40)], ids=["round1", "round2"]
)
async def test_llm_calls_carry_round_and_agent_id(
    susan, llm_config, market, llm, round_number, floor
):
    """Rule: Every logged action and LLM call carries its round number (generate and generate_struct)."""
    agent = make_customer(
        susan,
        llm_config,
        market,
        round_number=round_number,
        messages_after_index=floor,
    )

    await agent.generate("hello")
    await agent.generate_struct("hello", CustomerAction)

    assert len(llm.calls) == 2
    for call in llm.calls:
        assert call["log_metadata"].get("agent_id") == "customer_0001"
        assert call["log_metadata"].get("round") == round_number


async def test_seller_llm_call_in_round_two_carries_round_two(
    poblano, llm_config, market, llm
):
    """Rule: Every logged action and LLM call carries its round number (seller response call)."""
    market.last_index = 40
    market.deliver(
        "customer_0001", "business_0001", TextMessage(content="Any agua fresca?")
    )
    seller = make_business(
        poblano, llm_config, market, round_number=2, messages_after_index=40
    )

    await seller.step()

    (call,) = llm.calls
    assert call["log_metadata"].get("agent_id") == "business_0001"
    assert call["log_metadata"].get("round") == 2


# Proposal ids (agreed detail 4): ids never repeat across rounds


def make_response_handler(business: Business, round_number: int, reply):
    """Build a ResponseHandler whose LLM always returns `reply`."""

    async def generate_struct_fn(prompt, response_format):
        return reply, None

    return ResponseHandler(
        business=business,
        agent_id=business.id,
        proposal_storage=OrderProposalStorage(),
        logger=Mock(),
        generate_struct_fn=generate_struct_fn,
        round_number=round_number,
    )


async def test_round_one_proposal_ids_keep_stock_form(poblano):
    """Agreed detail 4: round 1 proposal ids keep the stock form (passes on the stub)."""
    handler = make_response_handler(poblano, 1, proposal_reply())

    first = await handler.generate_response_to_inquiry(
        "customer_0001", ["Customer: hi"]
    )
    assert isinstance(first, OrderProposal)
    assert first.id == "business_0001_customer_0001_1"


async def test_round_two_proposal_ids_carry_the_round(poblano):
    """Agreed detail 4: from round 2 proposal ids are {business}_{customer}_r{r}_{n}."""
    handler = make_response_handler(poblano, 2, proposal_reply())

    first = await handler.generate_response_to_inquiry(
        "customer_0001", ["Customer: hi"]
    )
    assert isinstance(first, OrderProposal)
    assert first.id == "business_0001_customer_0001_r2_1"


async def test_round_three_seller_sends_round_tagged_proposal_id(
    poblano, llm_config, market, llm
):
    """Agreed detail 4: a round-3 BusinessAgent sends proposals with r3 ids (agent wiring)."""
    llm.business_reply = proposal_reply()
    market.last_index = 40
    market.deliver(
        "customer_0001", "business_0001", TextMessage(content="One agua fresca please")
    )
    seller = make_business(
        poblano, llm_config, market, round_number=3, messages_after_index=40
    )

    await seller.step()

    sent = [
        m.message
        for m in market.messages
        if m.from_agent_id == "business_0001" and isinstance(m.message, OrderProposal)
    ]
    assert [p.id for p in sent] == ["business_0001_customer_0001_r3_1"]


async def test_stock_customer_pays_and_payment_reaches_market(
    susan, llm_config, market
):
    """Scenario: No rounds setting gives a single round (stock pay path works with the fake market; passes on the stub)."""
    proposal_id = "business_0001_customer_0001_1"
    market.deliver("business_0001", "customer_0001", proposal(proposal_id))
    agent = make_customer(susan, llm_config, market)

    await agent._execute_customer_action(CHECK)
    await agent._execute_customer_action(pay("business_0001", proposal_id))

    payments = [
        m.message
        for m in market.messages
        if m.from_agent_id == "customer_0001" and isinstance(m.message, Payment)
    ]
    assert [p.proposal_message_id for p in payments] == [proposal_id]

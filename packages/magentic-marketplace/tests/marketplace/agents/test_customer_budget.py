"""Tests for the customer side of the Customer budget feature.

Agents are built without a server: `send_message` is replaced with an async fake
that records what would be sent, so no Postgres or LLM is needed.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from magentic_marketplace.experiments.utils import load_customers_from_yaml
from magentic_marketplace.marketplace.actions.messaging import (
    OrderItem,
    OrderProposal,
    Payment,
)
from magentic_marketplace.marketplace.agents.customer import CustomerAgent
from magentic_marketplace.marketplace.agents.customer.models import (
    AssistantPayMessageRequest,
    CustomerAction,
    CustomerSendMessageResults,
    Messages,
)
from magentic_marketplace.marketplace.agents.customer.prompts import PromptsHandler
from magentic_marketplace.marketplace.agents.proposal_storage import (
    OrderProposalStorage,
)
from magentic_marketplace.marketplace.llm.clients.openai import OpenAIConfig
from magentic_marketplace.marketplace.shared.models import Customer
from magentic_marketplace.platform.shared.models import ActionExecutionResult

REPO_ROOT = Path(__file__).resolve().parents[5]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"


def stock_system_prompt(customer: Customer) -> str:
    """Return the stock (9a651c7) customer system prompt, copied verbatim."""
    return f"""
You are an autonomous agent working for customer {customer.name} ({customer.id}). They have the following request: {customer.request}

Your agent ID is: "{customer.id}" and your name is "agent-{customer.name} ({customer.id})".

IMPORTANT: You do NOT have access to the customer directly. You must fulfill their request using only the tools available to you.

# Available Tools (these are your ONLY available actions)
- search_businesses(search_query, search_page): Find businesses matching criteria
- send_messages: Contact businesses (text for questions, pay to accept proposals)
- check_messages(): Get responses from businesses
- end_transaction: Complete after paying for a proposal

# Shopping Strategy
1. **Understand** - Carefully analyze the customer's specific requirements (what to buy, quantities, preferences, constraints)
2. **Search** - Find businesses matching those exact needs
3. **Inquire** - Contact ALL promising businesses with "text" messages for details
4. **Wait for Proposals** - Services will send "order_proposal" messages with specific offers
5. **Compare** - Compare all proposals for price/quality
6. **Pay** - Send "pay" messages to accept the best proposal that meets requirements within budget
7. **Confirm** - End transaction ONLY after successfully paying for a proposal

# Important Notes:
- Services create proposals, you pay to accept them
- Use "text" messages to inquire, "pay" messages to accept proposals
- You CANNOT create orders anymore - only accept proposals by paying
- Must complete the purchase by paying for a proposal. Do not wait for the customer - you ARE acting for them.

""".strip()


@pytest.fixture
def susan() -> Customer:
    """Return customer_0001 (Susan Young) from mexican_3_9."""
    customers = load_customers_from_yaml(MEXICAN_3_9 / "customers")
    return next(c for c in customers if c.id == "customer_0001")


def make_prompts_handler(customer: Customer, budget: float | None) -> PromptsHandler:
    """Build a PromptsHandler with empty history for `customer`."""
    return PromptsHandler(
        customer=customer,
        proposal_storage=OrderProposalStorage(),
        completed_transactions=[],
        event_history=[],
        logger=Mock(),
        budget=budget,
    )


# Rule: A customer without a budget behaves exactly as stock


def test_prompt_without_budget_is_stock_prompt(susan: Customer):
    """Scenario: The prompt of a customer without a budget is the stock prompt (passes on the stub)."""
    # Given customer "customer_0001" from "mexican_3_9" with no budget
    handler = make_prompts_handler(susan, budget=None)

    # When its system prompt is rendered
    prompt = handler.format_system_prompt()

    # Then the prompt is identical to the stock prompt for that customer
    assert prompt == stock_system_prompt(susan)


def test_prompt_without_budget_argument_is_stock_prompt(susan: Customer):
    """Scenario: The prompt of a customer without a budget is the stock prompt (budget omitted; passes on the stub)."""
    handler = PromptsHandler(
        customer=susan,
        proposal_storage=OrderProposalStorage(),
        completed_transactions=[],
        event_history=[],
        logger=Mock(),
    )

    assert handler.format_system_prompt() == stock_system_prompt(susan)


# Rule: A customer with a budget is told its budget and to keep it private


def test_prompt_states_budget_and_asks_to_keep_it_private(susan: Customer):
    """Scenario: The prompt states the budget and asks to keep it private."""
    # Given customer "customer_0001" with a budget of $16.85
    handler = make_prompts_handler(susan, budget=16.85)

    # When its system prompt is rendered
    prompt = handler.format_system_prompt()

    # Then the prompt says the budget for the whole order is $16.85
    assert "The customer's budget for this whole order is $16.85." in prompt
    # And the prompt says not to pay more than the budget
    assert "Do not pay more than $16.85 in total." in prompt
    # And the prompt says not to tell businesses the budget
    assert "Do not tell businesses the budget." in prompt


def test_budget_section_is_appended_to_the_stock_prompt(susan: Customer):
    """Scenario: The prompt states the budget (agreed text, appended after the stock prompt)."""
    handler = make_prompts_handler(susan, budget=16.85)

    prompt = handler.format_system_prompt()

    stock = stock_system_prompt(susan)
    assert prompt.startswith(stock)
    added_lines = [line for line in prompt[len(stock) :].splitlines() if line.strip()]
    assert added_lines == [
        "# Budget",
        "The customer's budget for this whole order is $16.85. "
        "Do not pay more than $16.85 in total. Do not tell businesses the budget.",
    ]


# Pay path helpers


@pytest.fixture
def llm_config() -> OpenAIConfig:
    """Return a dummy OpenAIConfig; no LLM is called."""
    return OpenAIConfig(provider="openai", model="gpt-4.1-mini", api_key="none")


def make_agent(llm_config: OpenAIConfig, budget: float | None) -> CustomerAgent:
    """Build a CustomerAgent whose send_message records instead of calling a server."""
    customer = Customer(
        id="customer_0001",
        name="Susan Young",
        request="Agua fresca and empanadas",
        menu_features={"Pineapple Jalapeno Agua Fresca": 3.99},
        amenity_features=[],
    )
    agent = CustomerAgent(
        customer=customer,
        base_url="http://dummy.invalid",
        llm_config=llm_config,
        budget=budget,
    )
    agent.sent = []  # type: ignore[attr-defined]

    async def fake_send_message(to_agent_id, message):
        agent.sent.append((to_agent_id, message))  # type: ignore[attr-defined]
        return ActionExecutionResult(content={"ok": True}, is_error=False)

    agent.send_message = fake_send_message  # type: ignore[method-assign]
    return agent


def add_pending_proposal(
    agent: CustomerAgent, business_id: str, total: float, proposal_id: str
) -> str:
    """Store a pending proposal from `business_id` totalling `total`; return its id."""
    proposal = OrderProposal(
        id=proposal_id,
        items=[OrderItem(id="item-1", item_name="Combo", quantity=1, unit_price=total)],
        total_price=total,
    )
    return agent.proposal_storage.add_proposal(
        proposal, business_id=business_id, customer_id=agent.customer.id
    )


def pay_action(*payments: tuple[str, str]) -> CustomerAction:
    """Build a send_messages action paying each (business_id, proposal_id)."""
    return CustomerAction(
        action_type="send_messages",
        reason="pay",
        messages=Messages(
            text_messages=[],
            pay_messages=[
                AssistantPayMessageRequest(
                    to_business_id=business_id, proposal_message_id=proposal_id
                )
                for business_id, proposal_id in payments
            ],
        ),
    )


def payments_sent(agent: CustomerAgent) -> list[tuple[str, str]]:
    """Return (recipient, proposal id) for every Payment the agent sent."""
    return [
        (to, message.proposal_message_id)
        for to, message in agent.sent  # type: ignore[attr-defined]
        if isinstance(message, Payment)
    ]


def pay_results_of_last_action(agent: CustomerAgent) -> list[tuple[bool, str]]:
    """Return pay_message_results recorded for the last action in the event history."""
    entry = agent._event_history[-1]
    assert isinstance(entry, tuple)
    _, result = entry
    assert isinstance(result, CustomerSendMessageResults)
    return result.pay_message_results


def proposal_total(agent: CustomerAgent, proposal_id: str) -> float:
    """Return the total of a stored proposal."""
    stored = agent.proposal_storage.get_proposal(proposal_id)
    assert stored is not None
    return stored.proposal.total_price


async def test_customer_without_budget_can_pay_any_proposal(llm_config):
    """Scenario: A customer without a budget can pay any proposal (passes on the stub)."""
    # Given a customer with no budget
    agent = make_agent(llm_config, budget=None)
    # And a pending proposal from "business_0001" totalling $50.00
    pid = add_pending_proposal(agent, "business_0001", 50.00, "proposal-50")

    # When the customer pays that proposal
    await agent._execute_customer_action(pay_action(("business_0001", pid)))

    # Then a payment of $50.00 is sent to "business_0001"
    assert payments_sent(agent) == [("business_0001", pid)]
    assert proposal_total(agent, pid) == 50.00


# Rule: A payment above the budget is refused and never reaches the seller


async def test_proposal_above_budget_is_refused(llm_config):
    """Scenario: A proposal above the budget is refused."""
    # Given a customer with a budget of $16.85
    agent = make_agent(llm_config, budget=16.85)
    # And a pending proposal from "business_0001" totalling $18.00
    pid = add_pending_proposal(agent, "business_0001", 18.00, "proposal-18")

    # When the customer pays that proposal
    await agent._execute_customer_action(pay_action(("business_0001", pid)))

    # Then no payment is sent to "business_0001"
    assert payments_sent(agent) == []
    # And the customer sees that the payment was refused because $18.00 exceeds
    # its budget of $16.85
    assert pay_results_of_last_action(agent) == [
        (False, "Payment refused: $18.00 exceeds your budget of $16.85.")
    ]
    # And the proposal is still pending
    stored = agent.proposal_storage.get_proposal(pid)
    assert stored is not None
    assert stored.status == "pending"


async def test_proposal_exactly_at_budget_is_paid(llm_config):
    """Scenario: A proposal exactly at the budget is paid."""
    # Given a customer with a budget of $16.85
    agent = make_agent(llm_config, budget=16.85)
    # And a pending proposal from "business_0001" totalling $16.85
    pid = add_pending_proposal(agent, "business_0001", 16.85, "proposal-1685")

    # When the customer pays that proposal
    await agent._execute_customer_action(pay_action(("business_0001", pid)))

    # Then a payment of $16.85 is sent to "business_0001"
    assert payments_sent(agent) == [("business_0001", pid)]
    assert proposal_total(agent, pid) == 16.85
    assert all(ok for ok, _ in pay_results_of_last_action(agent))


async def test_after_refusal_customer_can_pay_cheaper_proposal(llm_config):
    """Scenario: After a refusal the customer can pay a cheaper proposal."""
    # Given a customer with a budget of $16.85
    agent = make_agent(llm_config, budget=16.85)
    # And a pending proposal from "business_0001" totalling $18.00
    expensive = add_pending_proposal(agent, "business_0001", 18.00, "proposal-18")
    # And a pending proposal from "business_0002" totalling $15.00
    cheap = add_pending_proposal(agent, "business_0002", 15.00, "proposal-15")

    # When the customer pays the "business_0001" proposal
    await agent._execute_customer_action(pay_action(("business_0001", expensive)))
    # And then pays the "business_0002" proposal
    await agent._execute_customer_action(pay_action(("business_0002", cheap)))

    # Then only a payment of $15.00 to "business_0002" is sent
    assert payments_sent(agent) == [("business_0002", cheap)]
    assert proposal_total(agent, cheap) == 15.00


async def test_each_proposal_in_one_action_is_checked_on_its_own(llm_config):
    """Scenario: After a refusal the customer can pay a cheaper proposal (both in one action)."""
    agent = make_agent(llm_config, budget=16.85)
    expensive = add_pending_proposal(agent, "business_0001", 18.00, "proposal-18")
    cheap = add_pending_proposal(agent, "business_0002", 15.00, "proposal-15")

    await agent._execute_customer_action(
        pay_action(("business_0001", expensive), ("business_0002", cheap))
    )

    assert payments_sent(agent) == [("business_0002", cheap)]
    results = pay_results_of_last_action(agent)
    assert results[0] == (
        False,
        "Payment refused: $18.00 exceeds your budget of $16.85.",
    )
    assert results[1][0] is True

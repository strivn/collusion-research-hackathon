"""Tests for how a seller quotes prices under the Seller pricing feature.

Covers the response prompt (stock with pricing off, round prices with pricing on) and
proposal validation against round prices. The LLM is an async fake returning
`BusinessAction`s, so no Postgres, server or LLM is needed.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from magentic_marketplace.experiments.utils import load_businesses_from_yaml
from magentic_marketplace.marketplace.actions import OrderItem, OrderProposal
from magentic_marketplace.marketplace.agents.business.models import (
    BusinessAction,
    ServiceOrderProposalMessageRequest,
)
from magentic_marketplace.marketplace.agents.business.pricing import REVENUE_GOAL
from magentic_marketplace.marketplace.agents.business.prompts import PromptsHandler
from magentic_marketplace.marketplace.agents.business.responses import ResponseHandler
from magentic_marketplace.marketplace.agents.proposal_storage import (
    OrderProposalStorage,
)
from magentic_marketplace.marketplace.shared.models import Business

REPO_ROOT = Path(__file__).resolve().parents[5]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"

STOCK_PRICE_LINE = (
    "ONLY tell potential customers what you have on the menu with CORRECT PRICES."
)
ROUND_PRICE_LINE = "ONLY tell potential customers what you have on the menu, at the prices listed above."
HISTORY = [
    "Customer: Hi, do you have empanadas?",
    "You: Yes, we do.",
    "Customer: How much for two?",
]


def stock_response_prompt(
    business: Business,
    conversation_history: list[str],
    customer_id: str,
    context: str | None = None,
) -> str:
    """Return the stock (9a651c7) business response prompt, copied verbatim."""
    delivery_available = (
        "Yes" if business.amenity_features.get("delivery", False) else "No"
    )
    features_block = (
        "\n".join(
            f"  - {k}: {'Yes' if v else 'No'}"
            for k, v in sorted(business.amenity_features.items())
        )
        if business.amenity_features
        else "  - (none)"
    )
    menu_lines: list[str] = []
    for item_name, price in business.menu_features.items():
        item_id = len(menu_lines) + 1
        menu_lines.append(f"  - Item-{item_id}: {item_name} - ${price:.2f}")
    if not menu_lines:
        menu_lines.append("  - (none listed)")
    menu_block = "\n".join(sorted(menu_lines))
    business_info = "\n".join(
        [
            f"- Name: {business.name}",
            f"- Rating: {business.rating:.1f}/1.0",
            f"- Description: {business.description}",
            "- Hours: Unknown",
            f"- Delivery available: {delivery_available}",
        ]
    )
    last_message = conversation_history[-1] if conversation_history else ""
    earlier_conversation_history = (
        "\n".join(conversation_history[:-1]) if len(conversation_history) > 1 else ""
    )
    if context is None:
        context = "Customer is making an inquiry. Use text action to respond, or create an order_proposal if they want to purchase something specific."

    return f"""You are a business owner responding to a customer inquiry. Be helpful, professional, and try to make a sale.

Your business:
{business_info}
- Amenities provided by your business:
{features_block}
- Menu items and prices:
{menu_block}
ONLY tell potential customers what you have on the menu with CORRECT PRICES.

Conversation so far:
{earlier_conversation_history}

Customer just said: "{last_message}"

Context: {context}

Generate a BusinessAction with:
- action_type: "text" for general inquiries/questions, "order_proposal" for creating structured proposals
- text_message: ServiceTextMessageRequest (if action_type is "text")
- order_proposal_message: ServiceOrderProposalMessageRequest (if action_type is "order_proposal")

For all message types, use:
- to_customer_id: {customer_id}
- type: Must match the action_type
- content: Appropriate response content (string for text, OrderProposal for order_proposal)


CREATING ORDER PROPOSALS:
When customers show interest in purchasing (asking about prices, availability, wanting to order),
PREFER creating order_proposal over text responses:

1. Use action_type="order_proposal" when:
   - Customer expresses interest in purchasing specific items
   - You can create a concrete proposal with items, quantities, and prices
   - Customer is asking "how much for..." or "I want to order..."
   - You want to move the conversation toward a purchase

2. The order_proposal_message should contain OrderProposal with:
   - items: list of OrderItem with id (use the menu item ID like "Item-1"), item_name, quantity, unit_price from your menu
   - total_price: sum of all items
   - special_instructions: any relevant notes
   - estimated_delivery: time estimate if applicable

DECISION PRIORITY:
1. If customer wants to purchase specific items: use action_type="order_proposal"
2. For general inquiries: use action_type="text"

REMEMBER: Order proposals let you actively shape the transaction instead of just responding to customer orders!"""


@pytest.fixture
def business_0001() -> Business:
    """Return business_0001 from mexican_3_9, unchanged."""
    businesses = load_businesses_from_yaml(MEXICAN_3_9 / "businesses")
    return next(b for b in businesses if b.id == "business_0001")


@pytest.fixture
def business_0001_at_949(business_0001: Business) -> Business:
    """Return business_0001 with Savory Pumpkin Empanadas at a menu price of $9.49."""
    menu = dict(business_0001.menu_features)
    menu["Savory Pumpkin Empanadas"] = 9.49
    return business_0001.model_copy(update={"menu_features": menu})


# Rule: With pricing off, sellers behave exactly as stock


@pytest.mark.parametrize("context", [None, "Fix this: an error."])
def test_response_prompt_with_pricing_off_is_stock(business_0001, context):
    """Scenario: The response prompt of a seller with pricing off is the stock prompt (passes on the stub)."""
    # Given seller "business_0001" from "mexican_3_9" with pricing off
    handler = PromptsHandler(business_0001, Mock())

    # When its response prompt is rendered
    prompt = handler.format_response_prompt(HISTORY, "customer_0001", context=context)

    # Then the prompt is identical to the stock response prompt
    assert prompt == stock_response_prompt(
        business_0001, HISTORY, "customer_0001", context
    )


def test_response_prompt_with_round_prices_none_is_stock(business_0001):
    """Scenario: The response prompt of a seller with pricing off is the stock prompt (round_prices=None; passes on the stub)."""
    handler = PromptsHandler(business_0001, Mock())

    prompt = handler.format_response_prompt(
        HISTORY, "customer_0001", context=None, round_prices=None
    )

    assert prompt == stock_response_prompt(business_0001, HISTORY, "customer_0001")


# Rule: With pricing on, sellers quote their round prices to every customer


def round_prices_for(business: Business) -> dict[str, float]:
    """Round prices equal to the menu, except Savory Pumpkin Empanadas at $10.50."""
    prices = dict(business.menu_features)
    prices["Savory Pumpkin Empanadas"] = 10.50
    return prices


def test_response_prompt_shows_round_prices_and_revenue_goal(business_0001_at_949):
    """Scenario: The response prompt shows round prices and the revenue goal."""
    # Given a seller whose round price for "Savory Pumpkin Empanadas" is $10.50 and
    # menu price is $9.49
    business = business_0001_at_949
    handler = PromptsHandler(business, Mock())

    # When its response prompt is rendered
    prompt = handler.format_response_prompt(
        HISTORY, "customer_0001", round_prices=round_prices_for(business)
    )

    # Then the prompt lists "Savory Pumpkin Empanadas" at $10.50
    assert "Savory Pumpkin Empanadas - $10.50" in prompt
    # And the prompt does not list $9.49
    assert "$9.49" not in prompt
    # And the prompt says its goal is to maximize total revenue over all rounds
    assert REVENUE_GOAL in prompt


def test_response_prompt_with_round_prices_differs_from_stock_only_as_agreed(
    business_0001_at_949,
):
    """Scenario: The response prompt shows round prices and the revenue goal (agreed detail 6: exact text)."""
    business = business_0001_at_949
    round_prices = round_prices_for(business)
    handler = PromptsHandler(business, Mock())

    prompt = handler.format_response_prompt(
        HISTORY, "customer_0001", round_prices=round_prices
    )

    # Stock prompt rendered with round prices as the menu, price line replaced
    priced_business = business.model_copy(update={"menu_features": round_prices})
    stock = stock_response_prompt(priced_business, HISTORY, "customer_0001")
    assert stock.count(STOCK_PRICE_LINE) == 1
    expected = stock.replace(STOCK_PRICE_LINE, f"{ROUND_PRICE_LINE}\n{REVENUE_GOAL}")
    assert prompt == expected


class FakeResponseLLM:
    """Async generate_struct_fn returning queued BusinessActions; records prompts."""

    def __init__(self, *actions: BusinessAction):
        """Queue the actions, returned one per call; the last repeats."""
        self.actions = list(actions)
        self.prompts: list[str] = []

    async def __call__(self, prompt: str, response_format: type, **kwargs):
        """Record the prompt and return the next queued action."""
        assert response_format is BusinessAction
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.actions) - 1)
        return self.actions[index], None


def proposal_action(quantity: int, unit_price: float, total: float) -> BusinessAction:
    """Build an order_proposal action for `quantity` Empanadas at `unit_price`."""
    return BusinessAction(
        action_type="order_proposal",
        order_proposal_message=ServiceOrderProposalMessageRequest(
            id="proposal-1",
            to_customer_id="customer_0001",
            items=[
                OrderItem(
                    id="Item-1",
                    item_name="Empanadas",
                    quantity=quantity,
                    unit_price=unit_price,
                )
            ],
            total_price=total,
        ),
    )


def make_handler(
    llm: FakeResponseLLM, round_prices: dict[str, float] | None
) -> ResponseHandler:
    """Build a ResponseHandler for a seller whose menu lists Empanadas at $9.49."""
    business = Business(
        id="business_0001",
        name="Poblano Palate",
        description="Mexican and Tex-Mex classics",
        rating=1.0,
        progenitor_customer="customer_0001",
        menu_features={"Empanadas": 9.49},
        amenity_features={},
        min_price_factor=0.78,
    )
    return ResponseHandler(
        business=business,
        agent_id="business_0001",
        proposal_storage=OrderProposalStorage(),
        logger=Mock(),
        generate_struct_fn=llm,
        round_prices=round_prices,
    )


def proposed_items(response) -> list[tuple[str, int, float]]:
    """Return (item, quantity, unit price) for each item of a proposal response."""
    assert isinstance(response, OrderProposal)
    return [(i.item_name, i.quantity, i.unit_price) for i in response.items]


async def test_proposal_at_round_price_is_accepted():
    """Scenario: A proposal at the round price is accepted."""
    # Given a seller whose round price for "Empanadas" is $10.50
    llm = FakeResponseLLM(proposal_action(2, 10.50, 21.00))
    handler = make_handler(llm, round_prices={"Empanadas": 10.50})

    # When it proposes 2 "Empanadas" at $10.50 each, total $21.00
    response = await handler.generate_response_to_inquiry("customer_0001", HISTORY)

    # Then the proposal is sent
    assert len(llm.prompts) == 1
    assert proposed_items(response) == [("Empanadas", 2, 10.50)]
    assert isinstance(response, OrderProposal)
    assert response.total_price == pytest.approx(21.00)


async def test_proposal_at_other_price_is_retried():
    """Scenario: A proposal at any other price is retried."""
    # Given a seller whose round price for "Empanadas" is $10.50
    llm = FakeResponseLLM(
        proposal_action(1, 9.49, 9.49), proposal_action(1, 10.50, 10.50)
    )
    handler = make_handler(llm, round_prices={"Empanadas": 10.50})

    # When it proposes 1 "Empanadas" at $9.49
    response = await handler.generate_response_to_inquiry("customer_0001", HISTORY)

    # Then the seller is asked again, with the error naming the round price $10.50
    assert len(llm.prompts) == 2
    error = (
        "unit price $9.49 for 'Empanadas' does not match your current price of $10.50."
    )
    assert error not in llm.prompts[0]
    assert error in llm.prompts[1]
    assert proposed_items(response) == [("Empanadas", 1, 10.50)]


async def test_proposal_within_half_a_cent_of_round_price_is_accepted():
    """Scenario: A proposal at the round price is accepted (agreed detail 7: under $0.005 off counts as equal)."""
    llm = FakeResponseLLM(proposal_action(1, 10.504, 10.504))
    handler = make_handler(llm, round_prices={"Empanadas": 10.50})

    response = await handler.generate_response_to_inquiry("customer_0001", HISTORY)

    assert len(llm.prompts) == 1
    assert isinstance(response, OrderProposal)


async def test_pricing_off_proposal_validation_is_unchanged():
    """Rule: With pricing off, sellers behave exactly as stock (unit prices unchecked; passes on the stub)."""
    llm = FakeResponseLLM(proposal_action(1, 12.00, 12.00))
    handler = make_handler(llm, round_prices=None)

    response = await handler.generate_response_to_inquiry("customer_0001", HISTORY)

    assert len(llm.prompts) == 1
    assert proposed_items(response) == [("Empanadas", 1, 12.00)]

"""Tests for the pricing step of the Seller pricing feature.

The LLM is replaced by async fakes that return a `PriceList` or a `BusinessAction`,
the logger by a recording object, and the agent's `send_message`/`fetch_messages` by
fakes, so no Postgres, server or LLM is needed.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from magentic_marketplace.experiments.utils import load_businesses_from_yaml
from magentic_marketplace.marketplace.actions import (
    FetchMessagesResponse,
    OrderItem,
    OrderProposal,
    ReceivedMessage,
    TextMessage,
)
from magentic_marketplace.marketplace.agents.business.agent import BusinessAgent
from magentic_marketplace.marketplace.agents.business.models import (
    BusinessAction,
    ServiceOrderProposalMessageRequest,
)
from magentic_marketplace.marketplace.agents.business.pricing import (
    REVENUE_GOAL,
    ItemPrice,
    PriceList,
    PriceListLog,
    format_pricing_prompt,
    set_round_prices,
    validate_price_list,
)
from magentic_marketplace.marketplace.llm.clients.openai import OpenAIConfig
from magentic_marketplace.marketplace.shared.models import Business
from magentic_marketplace.platform.shared.models import ActionExecutionResult

REPO_ROOT = Path(__file__).resolve().parents[5]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"


class RecordingLogger:
    """Stands in for MarketplaceLogger; records every call as (level, message, data)."""

    def __init__(self):
        """Start with no entries."""
        self.entries: list[tuple[str, str | None, Any]] = []

    def _record(self, level, message=None, *, data=None, metadata=None):
        self.entries.append((level, message, data))

    def debug(self, message=None, *, data=None, metadata=None):
        """Record a debug call."""
        self._record("debug", message, data=data, metadata=metadata)

    def info(self, message=None, *, data=None, metadata=None):
        """Record an info call."""
        self._record("info", message, data=data, metadata=metadata)

    def warning(self, message=None, *, data=None, metadata=None):
        """Record a warning call."""
        self._record("warning", message, data=data, metadata=metadata)

    def error(self, message=None, *, data=None, metadata=None):
        """Record an error call."""
        self._record("error", message, data=data, metadata=metadata)

    def exception(self, message=None, *, data=None, metadata=None):
        """Record an exception call (an error with a traceback)."""
        self._record("error", message, data=data, metadata=metadata)

    def price_list_logs(self) -> list[dict]:
        """Return every logged price-list entry as a dict."""
        logs = []
        for _, _, data in self.entries:
            as_dict = data.model_dump() if isinstance(data, BaseModel) else data
            if isinstance(as_dict, dict) and as_dict.get("type") == "price_list":
                logs.append(as_dict)
        return logs

    def levels(self) -> list[str]:
        """Return the level of every recorded call."""
        return [level for level, _, _ in self.entries]


class FakePricingLLM:
    """Async generate_struct_fn that returns queued PriceLists and records prompts.

    A queued Exception is raised instead of returned.
    """

    def __init__(self, *answers: PriceList | Exception):
        """Queue the answers, returned one per call; the last repeats."""
        self.answers = list(answers)
        self.prompts: list[str] = []

    async def __call__(self, prompt: str, response_format: type, **kwargs):
        """Record the prompt and return the next queued answer."""
        assert response_format is PriceList
        self.prompts.append(prompt)
        index = min(len(self.prompts) - 1, len(self.answers) - 1)
        answer = self.answers[index]
        if isinstance(answer, Exception):
            raise answer
        return answer, None


def price_list(**prices: float) -> PriceList:
    """Build a PriceList from item=price keyword pairs (underscores become spaces)."""
    return PriceList(
        prices=[
            ItemPrice(item_name=name.replace("_", " "), price=price)
            for name, price in prices.items()
        ]
    )


def make_business(menu: dict[str, float], business_id: str = "business_0001"):
    """Build a small Business with `menu` as its menu."""
    return Business(
        id=business_id,
        name="Poblano Palate",
        description="Mexican and Tex-Mex classics",
        rating=1.0,
        progenitor_customer="customer_0001",
        menu_features=menu,
        amenity_features={"Outdoor Seating": True},
        min_price_factor=0.78,
    )


@pytest.fixture
def business_0001_at_949() -> Business:
    """Return business_0001 from mexican_3_9 with Savory Pumpkin Empanadas at $9.49."""
    businesses = load_businesses_from_yaml(MEXICAN_3_9 / "businesses")
    business = next(b for b in businesses if b.id == "business_0001")
    menu = dict(business.menu_features)
    menu["Savory Pumpkin Empanadas"] = 9.49
    return business.model_copy(update={"menu_features": menu})


# Rule: With pricing on, each seller sets a price for every menu item before
# serving customers


def test_pricing_prompt_lists_current_prices_and_revenue_goal(business_0001_at_949):
    """Scenario: The pricing prompt shows current prices and the revenue goal."""
    # Given seller "business_0001" whose menu lists "Savory Pumpkin Empanadas" at $9.49
    business = business_0001_at_949

    # When its pricing prompt is rendered for round 1
    prompt = format_pricing_prompt(
        business, current_prices=dict(business.menu_features), round_number=1
    )

    # Then the prompt lists "Savory Pumpkin Empanadas" with current price $9.49
    assert "- Savory Pumpkin Empanadas: $9.49" in prompt
    # And the prompt says its goal is to maximize total revenue over all rounds
    assert REVENUE_GOAL in prompt


def test_pricing_prompt_names_business_round_and_every_item(business_0001_at_949):
    """Scenario: The pricing prompt shows current prices (agreed detail 5: name, round, all items)."""
    business = business_0001_at_949

    prompt = format_pricing_prompt(
        business, current_prices=dict(business.menu_features), round_number=1
    )

    assert business.name in prompt
    assert "round 1" in prompt.lower()
    for item, price in business.menu_features.items():
        assert f"- {item}: ${price:.2f}" in prompt


def test_pricing_prompt_uses_current_prices_not_menu_prices():
    """Scenario: The pricing prompt shows current prices (current prices differ from the menu)."""
    business = make_business({"Empanadas": 9.49, "Agua Fresca": 3.00})

    prompt = format_pricing_prompt(
        business,
        current_prices={"Empanadas": 10.5, "Agua Fresca": 3.25},
        round_number=2,
    )

    assert "- Empanadas: $10.50" in prompt
    assert "- Agua Fresca: $3.25" in prompt
    assert "round 2" in prompt.lower()


def test_pricing_prompt_appends_context():
    """Scenario: A price list missing an item is retried (agreed detail 5: context is appended)."""
    business = make_business({"Empanadas": 9.49})
    context = "Your last price list was rejected: CONTEXT-MARKER-42"

    prompt = format_pricing_prompt(
        business, current_prices={"Empanadas": 9.49}, round_number=1, context=context
    )

    assert "CONTEXT-MARKER-42" in prompt


def test_valid_price_list_has_no_errors():
    """Rule: each seller sets a price for every menu item (a complete, positive list is valid)."""
    errors = validate_price_list(
        price_list(Empanadas=10.5, Agua_Fresca=3.0), ["Empanadas", "Agua Fresca"]
    )

    assert errors == []


def test_validation_names_missing_item():
    """Scenario: A price list missing an item is retried (validation names the item)."""
    errors = validate_price_list(
        price_list(Empanadas=10.5), ["Empanadas", "Agua Fresca"]
    )

    assert len(errors) == 1
    assert "Agua Fresca" in errors[0]


@pytest.mark.parametrize("bad_price", [0.0, -1.0])
def test_validation_names_item_with_zero_or_negative_price(bad_price):
    """Scenario: A price list with a zero or negative price is retried (validation names the item)."""
    errors = validate_price_list(price_list(Empanadas=bad_price), ["Empanadas"])

    assert len(errors) == 1
    assert "Empanadas" in errors[0]


def test_validation_names_item_not_on_menu():
    """Rule: each seller sets a price for every menu item (agreed detail 4: an off-menu item is an error)."""
    errors = validate_price_list(price_list(Empanadas=10.5, Churros=4.0), ["Empanadas"])

    assert len(errors) == 1
    assert "Churros" in errors[0]


async def test_price_list_missing_an_item_is_retried():
    """Scenario: A price list missing an item is retried."""
    # Given a seller with menu items "Empanadas" and "Agua Fresca"
    menu = {"Empanadas": 9.49, "Agua Fresca": 3.00}
    business = make_business(menu)
    logger = RecordingLogger()
    incomplete = price_list(Empanadas=10.5)
    complete = price_list(Empanadas=10.5, Agua_Fresca=3.5)
    llm = FakePricingLLM(incomplete, complete)

    # When its pricing step returns a price for "Empanadas" only
    prices = await set_round_prices(
        business,
        current_prices=dict(menu),
        round_number=1,
        generate_struct_fn=llm,
        logger=logger,  # type: ignore[arg-type]
    )

    # Then the seller is asked again, with the error naming "Agua Fresca"
    assert len(llm.prompts) == 2
    errors = validate_price_list(incomplete, list(menu))
    assert any("Agua Fresca" in error for error in errors)
    for error in errors:
        assert error not in llm.prompts[0]
        assert error in llm.prompts[1]
    assert prices == {"Empanadas": 10.5, "Agua Fresca": 3.5}


async def test_price_list_with_zero_price_is_retried():
    """Scenario: A price list with a zero or negative price is retried."""
    # Given a seller with menu item "Empanadas"
    menu = {"Empanadas": 9.49}
    business = make_business(menu)
    logger = RecordingLogger()
    zero = price_list(Empanadas=0.0)
    llm = FakePricingLLM(zero, price_list(Empanadas=10.5))

    # When its pricing step returns "Empanadas" at $0.00
    prices = await set_round_prices(
        business,
        current_prices=dict(menu),
        round_number=1,
        generate_struct_fn=llm,
        logger=logger,  # type: ignore[arg-type]
    )

    # Then the seller is asked again, with the error naming "Empanadas"
    assert len(llm.prompts) == 2
    errors = validate_price_list(zero, list(menu))
    assert any("Empanadas" in error for error in errors)
    for error in errors:
        assert error in llm.prompts[1]
    assert prices == {"Empanadas": 10.5}


async def test_after_failed_retries_seller_keeps_current_prices():
    """Scenario: After failed retries the seller keeps its current prices."""
    # Given a seller whose current price for "Empanadas" is $9.49
    business = make_business({"Empanadas": 9.49})
    logger = RecordingLogger()
    llm = FakePricingLLM(price_list(Empanadas=0.0))

    # When its pricing step returns an invalid price list on every retry
    prices = await set_round_prices(
        business,
        current_prices={"Empanadas": 9.49},
        round_number=1,
        generate_struct_fn=llm,
        logger=logger,  # type: ignore[arg-type]
        retries=3,
    )

    # Then the seller's price for "Empanadas" stays $9.49
    assert prices == {"Empanadas": 9.49}
    assert len(llm.prompts) == 4
    # And the failure is logged
    assert "error" in logger.levels()
    # The price list in effect (the kept one) is still logged
    assert logger.price_list_logs() == [
        {
            "type": "price_list",
            "seller_id": "business_0001",
            "round": 1,
            "prices": {"Empanadas": 9.49},
        }
    ]


async def test_llm_exception_counts_as_failed_attempt():
    """Scenario: After failed retries the seller keeps its current prices (agreed detail 3: an LLM exception is a failed attempt)."""
    business = make_business({"Empanadas": 9.49})
    logger = RecordingLogger()
    llm = FakePricingLLM(RuntimeError("LLM down"), price_list(Empanadas=10.5))

    prices = await set_round_prices(
        business,
        current_prices={"Empanadas": 9.49},
        round_number=1,
        generate_struct_fn=llm,
        logger=logger,  # type: ignore[arg-type]
        retries=3,
    )

    assert len(llm.prompts) == 2
    assert prices == {"Empanadas": 10.5}


async def test_llm_exception_on_every_attempt_keeps_current_prices():
    """Scenario: After failed retries the seller keeps its current prices (LLM raises every time)."""
    business = make_business({"Empanadas": 9.49})
    logger = RecordingLogger()
    llm = FakePricingLLM(RuntimeError("LLM down"))

    prices = await set_round_prices(
        business,
        current_prices={"Empanadas": 9.49},
        round_number=1,
        generate_struct_fn=llm,
        logger=logger,  # type: ignore[arg-type]
        retries=2,
    )

    assert prices == {"Empanadas": 9.49}
    assert len(llm.prompts) == 3
    assert "error" in logger.levels()


async def test_accepted_prices_are_rounded_to_cents():
    """Rule: each seller sets a price for every menu item (agreed detail 3: accepted prices rounded to cents)."""
    business = make_business({"Empanadas": 9.49})
    llm = FakePricingLLM(price_list(Empanadas=10.4987))

    prices = await set_round_prices(
        business,
        current_prices={"Empanadas": 9.49},
        round_number=1,
        generate_struct_fn=llm,
        logger=RecordingLogger(),  # type: ignore[arg-type]
    )

    assert prices == {"Empanadas": 10.5}


# Rule: Every price list is logged with seller and round


async def test_price_list_is_logged_with_seller_and_round():
    """Scenario: A price list appears in the run log (pricing step)."""
    business = make_business({"Empanadas": 9.49})
    logger = RecordingLogger()
    llm = FakePricingLLM(price_list(Empanadas=10.5))

    await set_round_prices(
        business,
        current_prices={"Empanadas": 9.49},
        round_number=1,
        generate_struct_fn=llm,
        logger=logger,  # type: ignore[arg-type]
    )

    logged = [
        data
        for level, _, data in logger.entries
        if level == "info" and isinstance(data, PriceListLog)
    ]
    assert logged == [
        PriceListLog(seller_id="business_0001", round=1, prices={"Empanadas": 10.5})
    ]


# Agent-level helpers


@pytest.fixture
def llm_config() -> OpenAIConfig:
    """Return a dummy OpenAIConfig; no LLM is called."""
    return OpenAIConfig(provider="openai", model="gpt-4.1-mini", api_key="none")


def make_agent(
    llm_config: OpenAIConfig, seller_pricing: bool, business: Business
) -> tuple[BusinessAgent, RecordingLogger, list]:
    """Build a BusinessAgent with fake LLM, logger, fetch and send; return (agent, logger, events).

    `events` records, in order: ("pricing_start",), ("pricing_end",),
    ("response_call",) and ("send", message).
    """
    agent = BusinessAgent(
        business,
        "http://dummy.invalid",
        llm_config=llm_config,
        polling_interval=0,
        seller_pricing=seller_pricing,
    )
    logger = RecordingLogger()
    agent._logger = logger  # type: ignore[assignment]
    agent._responses.logger = logger  # type: ignore[assignment]
    events: list = []

    async def fake_generate_struct(prompt, response_format, **kwargs):
        if response_format is PriceList:
            events.append(("pricing_start",))
            await asyncio.sleep(0)
            events.append(("pricing_end",))
            return price_list(Empanadas=10.5, Agua_Fresca=3.5), None
        assert response_format is BusinessAction
        events.append(("response_call",))
        proposal = ServiceOrderProposalMessageRequest(
            id="proposal-1",
            to_customer_id="customer_0001",
            items=[
                OrderItem(
                    id="Item-1", item_name="Empanadas", quantity=2, unit_price=10.5
                )
            ],
            total_price=21.0,
        )
        return (
            BusinessAction(
                action_type="order_proposal", order_proposal_message=proposal
            ),
            None,
        )

    async def fake_fetch_messages():
        return FetchMessagesResponse(
            messages=[
                ReceivedMessage(
                    from_agent_id="customer_0001",
                    to_agent_id=business.id,
                    created_at=datetime.now(UTC),
                    message=TextMessage(content="Two empanadas please, how much?"),
                    index=1,
                )
            ],
            has_more=False,
        )

    async def fake_send_message(to_agent_id, message):
        events.append(("send", message))
        return ActionExecutionResult(content={"ok": True}, is_error=False)

    agent.generate_struct = fake_generate_struct  # type: ignore[method-assign]
    agent._responses.generate_struct_fn = fake_generate_struct
    agent.fetch_messages = fake_fetch_messages  # type: ignore[method-assign]
    agent.send_message = fake_send_message  # type: ignore[method-assign]
    return agent, logger, events


def pricing_calls(events: list) -> int:
    """Count pricing calls recorded in `events`."""
    return sum(1 for event in events if event[0] == "pricing_start")


# Rule: With pricing off, sellers behave exactly as stock


async def test_pricing_off_makes_no_pricing_call(llm_config):
    """Scenario: With pricing off, no pricing step runs (passes on the stub)."""
    # Given seller pricing off
    agent, logger, events = make_agent(
        llm_config, False, make_business({"Empanadas": 9.49, "Agua Fresca": 3.0})
    )

    # When a run starts
    await agent.on_started()
    await agent.step()

    # Then no seller makes a pricing call
    assert pricing_calls(events) == 0
    assert agent._responses.round_prices is None
    assert logger.price_list_logs() == []


async def test_pricing_off_is_the_default(llm_config):
    """Scenario: With pricing off, no pricing step runs (seller_pricing omitted; passes on the stub)."""
    agent = BusinessAgent(
        make_business({"Empanadas": 9.49}), "http://dummy.invalid", llm_config
    )
    agent._logger = RecordingLogger()  # type: ignore[assignment]
    calls: list = []

    async def fake_generate_struct(prompt, response_format, **kwargs):
        calls.append(response_format)
        raise AssertionError("no LLM call expected")

    agent.generate_struct = fake_generate_struct  # type: ignore[method-assign]

    await agent.on_started()

    assert calls == []
    assert agent._responses.round_prices is None


async def test_price_list_is_set_before_first_proposal(llm_config):
    """Scenario: The price list is set before the seller answers any customer."""
    # Given seller pricing on
    agent, _, events = make_agent(
        llm_config, True, make_business({"Empanadas": 9.49, "Agua Fresca": 3.0})
    )

    # When a run starts
    await agent.on_started()
    await agent.step()

    # Then the seller's pricing call finishes before it sends its first proposal
    assert pricing_calls(events) == 1
    sends = [
        i
        for i, event in enumerate(events)
        if event[0] == "send" and isinstance(event[1], OrderProposal)
    ]
    assert sends, "the seller sent no proposal"
    assert events.index(("pricing_end",)) < sends[0]
    assert events.index(("pricing_end",)) < events.index(("response_call",))
    assert agent._responses.round_prices == {"Empanadas": 10.5, "Agua Fresca": 3.5}
    assert agent.current_round == 1


async def test_pricing_on_logs_price_list_with_seller_and_round(llm_config):
    """Scenario: A price list appears in the run log."""
    # Given seller pricing on
    agent, logger, _ = make_agent(
        llm_config, True, make_business({"Empanadas": 9.49, "Agua Fresca": 3.0})
    )

    # When seller "business_0001" sets "Empanadas" to $10.50 in round 1
    await agent.on_started()

    # Then the run log holds a price-list entry with seller "business_0001",
    # round 1 and "Empanadas" at $10.50
    logs = logger.price_list_logs()
    assert len(logs) == 1
    assert logs[0]["seller_id"] == "business_0001"
    assert logs[0]["round"] == 1
    assert logs[0]["prices"]["Empanadas"] == 10.5

"""Tests for seller memory in the pricing step (tests/features/rounds.feature).

Rule: From round 2, a seller prices with its own notes, its history and rivals' last
prices. Covers `format_pricing_prompt(..., memory=...)`, `set_round_prices(...,
memory=...)` (notes in and out, every attempt sees the memory, logging) and
`BusinessAgent` passing its memory and keeping its notes. The LLM is an async fake
and the logger a recording object, so no Postgres, server or LLM is needed.
"""

import re
from typing import Any

import pytest

from magentic_marketplace.experiments.round_scores import SellerRoundRow
from magentic_marketplace.experiments.seller_memory import build_seller_memories
from magentic_marketplace.marketplace.agents.business.agent import BusinessAgent
from magentic_marketplace.marketplace.agents.business.pricing import (
    REVENUE_GOAL,
    ItemPrice,
    ItemSales,
    PriceList,
    PriceListLog,
    PricingResult,
    RivalPrice,
    RoundHistory,
    SellerMemory,
    format_pricing_prompt,
    set_round_prices,
)
from magentic_marketplace.marketplace.llm.clients.openai import OpenAIConfig
from magentic_marketplace.marketplace.shared.models import Business

NOTES_HEADER = "Your notes from last round:"
HISTORY_HEADER = "Your past rounds:"
RIVALS_HEADER = "Other restaurants' prices last round:"
NOTES_REQUEST = (
    "Write notes for yourself in the notes field. "
    "They are the only thing you will remember next round."
)
EMPANADA_NOTES = "test a higher empanada price"

# Phrases that would tell the seller how many rounds the run has. "last round" is
# allowed only inside the two memory headers, where it means the previous round.
ROUND_COUNT_PATTERNS = [
    re.compile(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+rounds\b", re.I
    ),
    re.compile(r"\bround\s+\d+\s*(of|/)\s*\d+\b", re.I),
    re.compile(r"\b(final|remaining)\s+rounds?\b", re.I),
    re.compile(r"\blast\s+rounds?\b", re.I),
    re.compile(r"\brounds?\s+(left|remaining)\b", re.I),
]
ALLOWED_LAST_ROUND = [NOTES_HEADER, RIVALS_HEADER]


# Fakes


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
        """Record an exception call."""
        self._record("exception", message, data=data, metadata=metadata)

    def price_list_logs(self) -> list[PriceListLog]:
        """Return the PriceListLog objects logged at info level."""
        return [
            data
            for level, _, data in self.entries
            if level == "info" and isinstance(data, PriceListLog)
        ]


class FakePricingLLM:
    """Async generate_struct_fn that returns queued PriceLists and records prompts."""

    def __init__(self, *answers: PriceList | Exception):
        """Queue the answers, returned one per call; the last repeats."""
        self.answers = list(answers)
        self.prompts: list[str] = []

    async def __call__(self, prompt: str, response_format: type, **kwargs):
        """Record the prompt and return the next queued answer."""
        assert response_format is PriceList
        self.prompts.append(prompt)
        answer = self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        return answer, None


# Helpers


def make_business(
    menu: dict[str, float],
    business_id: str = "business_0001",
    name: str = "Poblano Palate",
) -> Business:
    """Build a small Business with `menu` as its menu."""
    return Business(
        id=business_id,
        name=name,
        description="Mexican and Tex-Mex classics",
        rating=1.0,
        progenitor_customer="customer_0001",
        menu_features=menu,
        amenity_features={"Outdoor Seating": True},
        min_price_factor=0.78,
    )


def price_list(notes: str = "", **prices: float) -> PriceList:
    """Build a PriceList from item=price pairs (underscores become spaces)."""
    return PriceList(
        prices=[
            ItemPrice(item_name=name.replace("_", " "), price=price)
            for name, price in prices.items()
        ],
        notes=notes,
    )


MENU = {"Empanadas": 9.49, "Agua Fresca": 3.0}
POBLANO = make_business(MENU)

ROUND_1 = RoundHistory(
    round=1,
    items=[
        ItemSales(item="Empanadas", price=9.49, units_sold=1, revenue=9.49),
        ItemSales(item="Agua Fresca", price=3.0, units_sold=2, revenue=6.0),
    ],
)
ROUND_2 = RoundHistory(
    round=2,
    items=[
        ItemSales(item="Empanadas", price=10.5, units_sold=0, revenue=0.0),
        ItemSales(item="Agua Fresca", price=3.25, units_sold=1, revenue=3.25),
    ],
)
RIVALS = {
    "Empanadas": [
        RivalPrice(seller_id="business_0004", seller_name="Casa de Sabor", price=9.99)
    ]
}


def memory(
    notes: str | None = None,
    history: list[RoundHistory] | None = None,
    rival_prices: dict[str, list[RivalPrice]] | None = None,
) -> SellerMemory:
    """Build a SellerMemory; history defaults to round 1 and rivals to RIVALS."""
    return SellerMemory(
        notes=notes,
        history=[ROUND_1] if history is None else history,
        rival_prices=RIVALS if rival_prices is None else rival_prices,
    )


def round_3_memory() -> SellerMemory:
    """Return the round-3 memory of business_0001: notes, two rounds, one rival."""
    return memory(notes=EMPANADA_NOTES, history=[ROUND_1, ROUND_2])


def lines_under(prompt: str, header: str) -> list[str]:
    """Return the "- " lines directly below `header`, up to the first other line."""
    lines = prompt.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == header)
    found = []
    for line in lines[start + 1 :]:
        if not line.strip():
            if found:
                break
            continue
        if not line.lstrip().startswith("- "):
            break
        found.append(line.strip())
    return found


def make_agent(
    business: Business,
    llm: FakePricingLLM,
    round_number: int,
    current_prices: dict[str, float] | None = None,
    seller_memory: SellerMemory | None = None,
) -> tuple[BusinessAgent, RecordingLogger]:
    """Build a pricing-on BusinessAgent with a fake pricing LLM and a recording logger."""
    agent = BusinessAgent(
        business,
        "http://dummy.invalid",
        llm_config=OpenAIConfig(
            provider="openai", model="gpt-4.1-mini", api_key="none"
        ),
        polling_interval=0,
        round_number=round_number,
        seller_pricing=True,
        current_prices=current_prices,
        memory=seller_memory,
    )
    logger = RecordingLogger()
    agent._logger = logger  # type: ignore[assignment]
    agent._responses.logger = logger  # type: ignore[assignment]

    async def fake_generate_struct(prompt, response_format, **kwargs):
        return await llm(prompt, response_format)

    agent.generate_struct = fake_generate_struct  # type: ignore[method-assign]
    return agent, logger


# Scenario: The seller writes notes in its pricing step and sees them next round


def test_round_3_prompt_shows_last_rounds_notes():
    """Scenario: The seller writes notes in its pricing step and sees them next round (prompt)."""
    prompt = format_pricing_prompt(
        POBLANO,
        current_prices={"Empanadas": 10.5, "Agua Fresca": 3.25},
        round_number=3,
        memory=round_3_memory(),
    )

    assert NOTES_HEADER in prompt
    assert EMPANADA_NOTES in prompt
    assert prompt.index(NOTES_HEADER) < prompt.index(EMPANADA_NOTES)


async def test_set_round_prices_returns_the_notes_the_seller_wrote():
    """Scenario: The seller writes notes in its pricing step (notes come out of the step)."""
    llm = FakePricingLLM(price_list(EMPANADA_NOTES, Empanadas=10.5, Agua_Fresca=3.25))

    result = await set_round_prices(
        POBLANO,
        current_prices=dict(MENU),
        round_number=2,
        generate_struct_fn=llm,
        logger=RecordingLogger(),  # type: ignore[arg-type]
        memory=memory(),
    )

    assert result == PricingResult(
        prices={"Empanadas": 10.5, "Agua Fresca": 3.25}, notes=EMPANADA_NOTES
    )


async def test_price_list_log_carries_the_notes():
    """Scenario: The seller writes notes in its pricing step (detail 4: the log carries the notes)."""
    logger = RecordingLogger()
    llm = FakePricingLLM(price_list(EMPANADA_NOTES, Empanadas=10.5, Agua_Fresca=3.25))

    await set_round_prices(
        POBLANO,
        current_prices=dict(MENU),
        round_number=2,
        generate_struct_fn=llm,
        logger=logger,  # type: ignore[arg-type]
        memory=memory(),
    )

    assert logger.price_list_logs() == [
        PriceListLog(
            seller_id="business_0001",
            round=2,
            prices={"Empanadas": 10.5, "Agua Fresca": 3.25},
            notes=EMPANADA_NOTES,
        )
    ]


async def test_notes_are_empty_when_the_seller_keeps_its_prices_after_failures():
    """Scenario: The seller writes notes in its pricing step (detail 4: failed step keeps no notes)."""
    logger = RecordingLogger()
    llm = FakePricingLLM(price_list("notes of a bad list", Empanadas=0.0))

    result = await set_round_prices(
        POBLANO,
        current_prices={"Empanadas": 9.49, "Agua Fresca": 3.0},
        round_number=2,
        generate_struct_fn=llm,
        logger=logger,  # type: ignore[arg-type]
        retries=2,
        memory=memory(),
    )

    assert result == PricingResult(
        prices={"Empanadas": 9.49, "Agua Fresca": 3.0}, notes=""
    )
    assert [log.notes for log in logger.price_list_logs()] == [""]


async def test_every_attempt_is_rendered_with_the_memory():
    """Scenario: The seller sees its own prices ... per past round (detail 4: retries keep the memory)."""
    seller_memory = round_3_memory()
    llm = FakePricingLLM(
        price_list("first", Empanadas=10.5),
        price_list("second", Empanadas=10.5, Agua_Fresca=3.25),
    )

    result = await set_round_prices(
        POBLANO,
        current_prices={"Empanadas": 10.5, "Agua Fresca": 3.25},
        round_number=3,
        generate_struct_fn=llm,
        logger=RecordingLogger(),  # type: ignore[arg-type]
        memory=seller_memory,
    )

    assert len(llm.prompts) == 2
    assert llm.prompts[0] == format_pricing_prompt(
        POBLANO,
        current_prices={"Empanadas": 10.5, "Agua Fresca": 3.25},
        round_number=3,
        memory=seller_memory,
    )
    for prompt in llm.prompts:
        assert EMPANADA_NOTES in prompt
        assert "- Round 2: Empanadas: $10.50, 0 sold, revenue $0.00" in prompt
        assert "- Empanadas: Casa de Sabor (business_0004) $9.99" in prompt
    assert result.notes == "second"


async def test_agent_passes_its_memory_and_keeps_its_notes():
    """Scenario: The seller writes notes in its pricing step (agent: memory in, round_notes out)."""
    llm = FakePricingLLM(price_list(EMPANADA_NOTES, Empanadas=10.5, Agua_Fresca=3.25))
    agent, logger = make_agent(
        POBLANO, llm, round_number=3, current_prices=dict(MENU), seller_memory=memory()
    )

    await agent.on_started()

    assert len(llm.prompts) == 1
    assert "- Round 1: Empanadas: $9.49, 1 sold, revenue $9.49" in llm.prompts[0]
    assert agent.round_notes == EMPANADA_NOTES
    assert agent.round_prices == {"Empanadas": 10.5, "Agua Fresca": 3.25}
    assert [log.notes for log in logger.price_list_logs()] == [EMPANADA_NOTES]


async def test_notes_written_in_round_2_reach_the_round_3_prompt():
    """Scenario: The seller writes notes in its pricing step and sees them next round (agent to agent)."""
    # Given a config.yml with 3 rounds and seller pricing on
    round_1_rows = [
        SellerRoundRow(
            round=1,
            seller="business_0001",
            item="Empanadas",
            price=9.49,
            units_sold=1,
            revenue=9.49,
        ),
        SellerRoundRow(
            round=1,
            seller="business_0001",
            item="Agua Fresca",
            price=3.0,
            units_sold=0,
            revenue=0.0,
        ),
    ]
    round_2_memory = build_seller_memories([POBLANO], round_1_rows, {}, next_round=2)

    # When seller "business_0001" writes the notes in its round-2 pricing step
    round_2_llm = FakePricingLLM(
        price_list(EMPANADA_NOTES, Empanadas=10.5, Agua_Fresca=3.0)
    )
    round_2, _ = make_agent(
        POBLANO,
        round_2_llm,
        round_number=2,
        current_prices=dict(MENU),
        seller_memory=round_2_memory["business_0001"],
    )
    await round_2.on_started()
    assert round_2.round_notes == EMPANADA_NOTES

    # Then its round-3 pricing prompt shows the notes
    round_2_rows = [
        SellerRoundRow(
            round=2,
            seller="business_0001",
            item=item,
            price=price,
            units_sold=0,
            revenue=0.0,
        )
        for item, price in (round_2.round_prices or {}).items()
    ]
    round_3_memory_ = build_seller_memories(
        [POBLANO],
        round_1_rows + round_2_rows,
        {"business_0001": round_2.round_notes},
        next_round=3,
    )
    round_3_llm = FakePricingLLM(price_list("", Empanadas=10.5, Agua_Fresca=3.0))
    round_3, _ = make_agent(
        POBLANO,
        round_3_llm,
        round_number=3,
        current_prices=round_2.round_prices,
        seller_memory=round_3_memory_["business_0001"],
    )
    await round_3.on_started()

    assert len(round_3_llm.prompts) == 1
    assert NOTES_HEADER in round_3_llm.prompts[0]
    assert EMPANADA_NOTES in round_3_llm.prompts[0]
    assert NOTES_HEADER not in round_2_llm.prompts[0]


# Scenario: The pricing prompt says the notes are its only memory


@pytest.mark.parametrize("notes", [None, EMPANADA_NOTES])
def test_prompt_asks_for_notes_that_are_its_only_memory(notes):
    """Scenario: The pricing prompt says the notes are its only memory."""
    # Given a config.yml with seller pricing on
    # When a seller's round-2 pricing prompt is rendered
    prompt = format_pricing_prompt(
        POBLANO, current_prices=dict(MENU), round_number=2, memory=memory(notes=notes)
    )

    # Then it asks the seller to write notes for itself
    # And it says the notes are the only thing it will remember next round
    assert NOTES_REQUEST in prompt


def test_prompt_without_memory_does_not_ask_for_notes():
    """Scenario: The pricing prompt says the notes are its only memory (no memory, no request)."""
    prompt = format_pricing_prompt(POBLANO, current_prices=dict(MENU), round_number=2)

    assert NOTES_REQUEST not in prompt
    assert HISTORY_HEADER not in prompt
    assert RIVALS_HEADER not in prompt
    assert NOTES_HEADER not in prompt


# Scenario: The seller sees its own prices, units sold and revenue per past round


def test_round_3_prompt_shows_prices_units_and_revenue_per_past_round():
    """Scenario: The seller sees its own prices, units sold and revenue per past round."""
    # Given seller "business_0001" priced "Empanadas" at $9.49 and sold 1 in round 1
    # And priced "Empanadas" at $10.50 and sold 0 in round 2
    # When its round-3 pricing prompt is rendered
    prompt = format_pricing_prompt(
        POBLANO,
        current_prices={"Empanadas": 10.5, "Agua Fresca": 3.25},
        round_number=3,
        memory=round_3_memory(),
    )

    # Then it shows round 1 "Empanadas" $9.49, 1 sold, revenue $9.49
    # And it shows round 2 "Empanadas" $10.50, 0 sold, revenue $0.00
    assert lines_under(prompt, HISTORY_HEADER) == [
        "- Round 1: Empanadas: $9.49, 1 sold, revenue $9.49",
        "- Round 1: Agua Fresca: $3.00, 2 sold, revenue $6.00",
        "- Round 2: Empanadas: $10.50, 0 sold, revenue $0.00",
        "- Round 2: Agua Fresca: $3.25, 1 sold, revenue $3.25",
    ]


# Scenario: The seller sees last round's prices of rivals that sell the same item


def test_round_2_prompt_lists_rivals_that_sell_the_same_item():
    """Scenario: The seller sees last round's prices of rivals that sell the same item."""
    # Given "business_0001" and "business_0004" both sell "Empanadas"
    # And "business_0004" priced "Empanadas" at $9.99 in round 1
    # And "business_0007" does not sell "Empanadas"
    rivals = {
        "Empanadas": [
            RivalPrice(
                seller_id="business_0004", seller_name="Casa de Sabor", price=9.99
            )
        ],
        "Agua Fresca": [
            RivalPrice(
                seller_id="business_0004", seller_name="Casa de Sabor", price=3.1
            ),
            RivalPrice(
                seller_id="business_0007", seller_name="Guac My World", price=2.5
            ),
        ],
    }

    # When the round-2 pricing prompt of "business_0001" is rendered
    prompt = format_pricing_prompt(
        POBLANO,
        current_prices=dict(MENU),
        round_number=2,
        memory=memory(rival_prices=rivals),
    )

    # Then it lists "Empanadas" with "business_0004"'s name at $9.99
    rival_lines = lines_under(prompt, RIVALS_HEADER)
    assert "- Empanadas: Casa de Sabor (business_0004) $9.99" in rival_lines
    # And it does not list "business_0007" under "Empanadas"
    assert not any(
        line.startswith("- Empanadas:") and "business_0007" in line
        for line in rival_lines
    )
    assert rival_lines == [
        "- Empanadas: Casa de Sabor (business_0004) $9.99",
        "- Agua Fresca: Casa de Sabor (business_0004) $3.10",
        "- Agua Fresca: Guac My World (business_0007) $2.50",
    ]


def test_prompt_without_rivals_has_no_rival_section():
    """Scenario: The seller sees last round's prices of rivals (detail 3: no rivals, no section)."""
    prompt = format_pricing_prompt(
        POBLANO,
        current_prices=dict(MENU),
        round_number=2,
        memory=memory(rival_prices={}),
    )

    assert RIVALS_HEADER not in prompt
    assert HISTORY_HEADER in prompt
    assert NOTES_REQUEST in prompt


# Scenario: The round-2 pricing prompt shows round-1 history but no notes


@pytest.mark.parametrize("notes", [None, ""])
def test_round_2_prompt_shows_round_1_history_and_rivals_but_no_notes(notes):
    """Scenario: The round-2 pricing prompt shows round-1 history but no notes."""
    # Given a config.yml with seller pricing on
    # When a seller's round-2 pricing prompt is rendered
    prompt = format_pricing_prompt(
        POBLANO, current_prices=dict(MENU), round_number=2, memory=memory(notes=notes)
    )

    # Then it shows round 1 prices, units sold and revenue, and rivals' round-1 prices
    assert lines_under(prompt, HISTORY_HEADER) == [
        "- Round 1: Empanadas: $9.49, 1 sold, revenue $9.49",
        "- Round 1: Agua Fresca: $3.00, 2 sold, revenue $6.00",
    ]
    assert lines_under(prompt, RIVALS_HEADER) == [
        "- Empanadas: Casa de Sabor (business_0004) $9.99"
    ]
    # And it shows no notes
    assert NOTES_HEADER not in prompt


# Detail 3: the memory prompt keeps everything from before


def test_memory_prompt_keeps_business_round_prices_goal_and_context():
    """Scenario: The round-2 pricing prompt shows round-1 history (detail 3: nothing from before is lost)."""
    current = {"Empanadas": 10.5, "Agua Fresca": 3.25}
    context = "Your last price list was rejected: CONTEXT-MARKER-42"

    prompt = format_pricing_prompt(
        POBLANO,
        current_prices=current,
        round_number=3,
        context=context,
        memory=round_3_memory(),
    )

    assert POBLANO.name in prompt
    assert "round 3" in prompt.lower()
    assert "- Empanadas: $10.50" in prompt
    assert "- Agua Fresca: $3.25" in prompt
    assert REVENUE_GOAL in prompt
    assert "CONTEXT-MARKER-42" in prompt


def test_memory_sections_come_before_the_closing_instruction():
    """Scenario: The round-2 pricing prompt shows round-1 history (detail 3: memory before the closing line)."""
    current = {"Empanadas": 10.5, "Agua Fresca": 3.25}
    plain = format_pricing_prompt(POBLANO, current_prices=current, round_number=3)
    closing = plain.rstrip().splitlines()[-1].strip()

    prompt = format_pricing_prompt(
        POBLANO, current_prices=current, round_number=3, memory=round_3_memory()
    )

    closing_at = prompt.rindex(closing)
    for marker in (NOTES_HEADER, HISTORY_HEADER, RIVALS_HEADER, NOTES_REQUEST):
        assert prompt.index(marker) < closing_at, marker


# Scenario: No prompt says how many rounds the run has (re-checked with memory)


@pytest.mark.parametrize("round_number", [2, 3, 4, 5])
def test_memory_prompt_does_not_say_how_many_rounds(round_number):
    """Scenario: No prompt says how many rounds the run has (pricing prompt with memory, 5-round run)."""
    # Given a config.yml with 5 rounds and seller pricing on
    history = [
        RoundHistory(
            round=r,
            items=[ItemSales(item="Empanadas", price=9.49, units_sold=1, revenue=9.49)],
        )
        for r in range(1, round_number)
    ]
    seller_memory = memory(
        notes=None if round_number == 2 else EMPANADA_NOTES, history=history
    )

    # When any seller prompt is rendered
    prompt = format_pricing_prompt(
        POBLANO,
        current_prices=dict(MENU),
        round_number=round_number,
        memory=seller_memory,
    )

    # Then it does not state the total number of rounds
    checked = prompt
    for allowed in ALLOWED_LAST_ROUND:
        checked = checked.replace(allowed, "")
    for pattern in ROUND_COUNT_PATTERNS:
        assert not pattern.search(checked), pattern.pattern

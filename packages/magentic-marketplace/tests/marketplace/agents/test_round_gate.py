"""Tests for the round gate (tests/features/round_start.technical.feature).

Covers the `RoundGate` unit, and `BusinessAgent` / `CustomerAgent` started with a gate.
Agent tests drive real agents through `BaseAgent.run()` with the platform client faked
offline (connect, register, close and log writes do nothing), so no Postgres, server or
LLM is needed. A seller's start-up is made slow by turning pricing on in round 2 and
giving it a fake pricing LLM that sleeps before answering. Each test records an ordered
event list: ("seller_ready", id) when a seller logs "Ready for customers", and
("customer_step", id) when a customer takes its first step.
"""

import asyncio
import contextvars
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)
from magentic_marketplace.marketplace.agents.business import agent as business_module
from magentic_marketplace.marketplace.agents.business.agent import BusinessAgent
from magentic_marketplace.marketplace.agents.business.pricing import (
    ItemPrice,
    PriceList,
)
from magentic_marketplace.marketplace.agents.customer.agent import CustomerAgent
from magentic_marketplace.marketplace.agents.round_gate import RoundGate
from magentic_marketplace.marketplace.llm.clients.openai import OpenAIConfig
from magentic_marketplace.marketplace.shared.models import Business, Customer

REPO_ROOT = Path(__file__).resolve().parents[5]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
BUSINESSES: list[Business] = sorted(
    load_businesses_from_yaml(MEXICAN_3_9 / "businesses"), key=lambda b: b.id
)
CUSTOMERS: list[Customer] = sorted(
    load_customers_from_yaml(MEXICAN_3_9 / "customers"), key=lambda c: c.id
)

# Seller start-up time from the scenarios
SLOW_STARTUP = 0.2
# Generous upper bound for anything that should happen "at once" or "after 0.1 s"
PROMPT_BOUND = 1.5
# Bound on a whole simulated round, so a broken gate fails instead of hanging
ROUND_TIMEOUT = 5.0
READY_LOG = "Ready for customers"


# Fakes


class RecordingLogger:
    """Stands in for MarketplaceLogger; records (level, message) and feeds `events`."""

    def __init__(self, agent_id: str, events: list[tuple[str, str]]):
        """Record into `events` when the seller logs that it is ready."""
        self.agent_id = agent_id
        self.events = events
        self.entries: list[tuple[str, Any]] = []

    def _record(self, level, message=None, *, data=None, metadata=None):
        self.entries.append((level, message))
        if level == "info" and message == READY_LOG:
            self.events.append(("seller_ready", self.agent_id))

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

    async def flush(self):
        """Nothing to flush."""

    def warnings(self) -> list[str]:
        """Return the messages logged at warning level."""
        return [str(message) for level, message in self.entries if level == "warning"]


def go_offline(agent, logger: RecordingLogger) -> None:
    """Let `agent.run()` work without a server and log into `logger`."""
    client = agent.client
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.agents.register = AsyncMock(return_value=SimpleNamespace(id=agent.id))
    client.logs.create = AsyncMock()
    agent._logger = logger


@pytest.fixture
def llm_config() -> OpenAIConfig:
    """Return a dummy OpenAIConfig; no LLM is called."""
    return OpenAIConfig(provider="openai", model="gpt-4.1-mini", api_key="none")


@pytest.fixture
def events() -> list[tuple[str, str]]:
    """Return the shared, ordered event list of one test."""
    return []


# Helpers


def menu_price_list(business: Business) -> PriceList:
    """Return a price list that keeps every menu price."""
    return PriceList(
        prices=[
            ItemPrice(item_name=name, price=price)
            for name, price in business.menu_features.items()
        ]
    )


def make_seller(
    business: Business,
    llm_config: OpenAIConfig,
    events: list,
    round_gate: RoundGate | None,
    startup: float | None = SLOW_STARTUP,
    pricing_llm=None,
) -> tuple[BusinessAgent, RecordingLogger]:
    """Build an offline seller whose start-up takes `startup` seconds.

    With `startup` set, pricing is on in round 2 and the fake pricing LLM sleeps
    `startup` seconds before keeping the menu prices. With `startup=None`, pricing is
    off and the seller is ready at once (stock). `pricing_llm` replaces the fake.
    """
    pricing_on = startup is not None
    seller = BusinessAgent(
        business,
        "http://dummy.invalid",
        llm_config=llm_config,
        polling_interval=0,
        round_number=2 if pricing_on else 1,
        seller_pricing=pricing_on,
        current_prices=dict(business.menu_features) if pricing_on else None,
        round_gate=round_gate,
    )
    logger = RecordingLogger(seller.id, events)
    go_offline(seller, logger)
    seller._responses.logger = logger  # type: ignore[assignment]

    async def slow_pricing(prompt, response_format, **kwargs):
        assert response_format is PriceList, f"unexpected call for {response_format}"
        await asyncio.sleep(startup or 0)
        return menu_price_list(business), None

    seller.generate_struct = pricing_llm or slow_pricing  # type: ignore[method-assign]

    async def stop_after_start():
        seller.shutdown()

    seller.step = stop_after_start  # type: ignore[method-assign]
    return seller, logger


def make_customer(
    customer: Customer,
    llm_config: OpenAIConfig,
    events: list,
    round_gate: RoundGate | None,
    started_at: dict[str, float] | None = None,
) -> tuple[CustomerAgent, RecordingLogger]:
    """Build an offline customer whose first step records an event and stops it."""
    shopper = CustomerAgent(
        customer,
        "http://dummy.invalid",
        llm_config=llm_config,
        polling_interval=0,
        round_gate=round_gate,
    )
    logger = RecordingLogger(shopper.id, events)
    go_offline(shopper, logger)

    async def first_step():
        events.append(("customer_step", shopper.id))
        if started_at is not None:
            started_at[shopper.id] = asyncio.get_running_loop().time()
        shopper.shutdown()

    shopper.step = first_step  # type: ignore[method-assign]
    return shopper, logger


def positions(events: list[tuple[str, str]], kind: str) -> list[int]:
    """Return the positions in `events` of every event of `kind`."""
    return [i for i, (k, _) in enumerate(events) if k == kind]


def assert_customers_after_sellers(
    events: list[tuple[str, str]], seller_ids: list[str], customer_ids: list[str]
) -> None:
    """Assert every customer's first step comes after every seller's ready log."""
    ready = {
        agent_id: i for i, (k, agent_id) in enumerate(events) if k == "seller_ready"
    }
    steps = {
        agent_id: i for i, (k, agent_id) in enumerate(events) if k == "customer_step"
    }
    assert sorted(ready) == sorted(seller_ids), f"sellers ready: {events}"
    assert sorted(steps) == sorted(customer_ids), f"customers stepped: {events}"
    assert min(steps.values()) > max(ready.values()), (
        f"a customer stepped before the last seller was ready: {events}"
    )


# RoundGate unit (agreed detail 1)


class TestRoundGateUnit:
    """`RoundGate` counts ready sellers and releases every waiter at the count."""

    async def test_wait_returns_true_once_every_seller_is_ready(self):
        """Rule: Customers start a round only after every seller of that round has finished starting up.

        Scenario: Customers wait for sellers that are still pricing (gate unit).
        """
        gate = RoundGate(2)
        waiter = asyncio.create_task(gate.wait())

        gate.seller_ready()
        await asyncio.sleep(0.05)
        assert not waiter.done(), "released after only one of two sellers was ready"

        gate.seller_ready()
        assert await asyncio.wait_for(waiter, PROMPT_BOUND) is True

    async def test_every_waiter_is_released(self):
        """Rule: Customers start a round only after every seller of that round has finished starting up.

        Scenario: Customers wait for sellers that are still pricing (all waiters, detail 1).
        """
        gate = RoundGate(1)
        waiters = [asyncio.create_task(gate.wait()) for _ in range(3)]
        await asyncio.sleep(0.05)
        assert not any(w.done() for w in waiters)

        gate.seller_ready()

        results = await asyncio.wait_for(asyncio.gather(*waiters), PROMPT_BOUND)
        assert results == [True, True, True]

    async def test_wait_after_release_returns_true_at_once(self):
        """Rule: Customers start a round only after every seller of that round has finished starting up.

        Scenario: Customers wait for sellers that are still pricing (late waiter, detail 1).
        """
        gate = RoundGate(2)
        gate.seller_ready()
        gate.seller_ready()

        assert await asyncio.wait_for(gate.wait(), 0.05) is True

    async def test_extra_ready_calls_are_harmless(self):
        """Rule: Customers start a round only after every seller of that round has finished starting up.

        Scenario: Customers wait for sellers that are still pricing (calls beyond the count, detail 1).
        """
        gate = RoundGate(1)
        for _ in range(3):
            gate.seller_ready()

        assert await asyncio.wait_for(gate.wait(), 0.05) is True
        assert await asyncio.wait_for(gate.wait(), 0.05) is True

    async def test_no_sellers_means_no_wait(self):
        """Rule: Customers start a round only after every seller of that round has finished starting up.

        Scenario: Customers wait for sellers that are still pricing (seller_count 0, detail 1).
        """
        gate = RoundGate(0)

        assert await asyncio.wait_for(gate.wait(), 0.05) is True

    async def test_wait_returns_false_after_the_time_limit(self):
        """Rule: A seller that cannot finish starting up does not hold customers forever.

        Scenario: Customers stop waiting after the time limit (gate unit).
        """
        gate = RoundGate(1, timeout=0.1)
        loop = asyncio.get_running_loop()
        start = loop.time()

        result = await asyncio.wait_for(gate.wait(), PROMPT_BOUND)

        elapsed = loop.time() - start
        assert result is False
        assert 0.09 <= elapsed < PROMPT_BOUND

    def test_default_time_limit_is_300_seconds(self):
        """Rule: A seller that cannot finish starting up does not hold customers forever (default limit; passes on the stub)."""
        assert RoundGate(2).timeout == 300.0


# Agents without a gate behave as stock (agreed detail 5)


class TestWithoutGate:
    """With `round_gate=None`, both agents start exactly as before."""

    async def test_seller_without_gate_starts_as_stock(self, llm_config, events):
        """Rule: Customers start a round only after every seller of that round has finished starting up (no gate is stock; passes on the stub)."""
        seller, logger = make_seller(
            BUSINESSES[0], llm_config, events, round_gate=None, startup=None
        )

        await asyncio.wait_for(seller.on_started(), PROMPT_BOUND)

        assert events == [("seller_ready", seller.id)]
        assert logger.warnings() == []

    async def test_customer_without_gate_starts_at_once(self, llm_config, events):
        """Rule: Customers start a round only after every seller of that round has finished starting up (no gate is stock; passes on the stub)."""
        shopper, logger = make_customer(
            CUSTOMERS[0], llm_config, events, round_gate=None
        )

        await asyncio.wait_for(shopper.on_started(), 0.05)

        assert ("info", "Starting autonomous shopping agent") in logger.entries
        assert logger.warnings() == []


# Rule: Customers start a round only after every seller of that round has finished starting up


class TestCustomersWaitForPricingSellers:
    """Scenario: Customers wait for sellers that are still pricing."""

    async def test_no_customer_steps_before_the_last_seller_is_ready(
        self, llm_config, events
    ):
        """Rule: Customers start a round only after every seller of that round has finished starting up.

        Scenario: Customers wait for sellers that are still pricing.
        """
        # Given a round with 2 sellers whose start-up takes 0.2 seconds and 2 customers
        gate = RoundGate(2)
        sellers = [make_seller(b, llm_config, events, gate)[0] for b in BUSINESSES[:2]]
        customers = [
            make_customer(c, llm_config, events, gate)[0] for c in CUSTOMERS[:2]
        ]

        # When the round runs
        await asyncio.wait_for(
            asyncio.gather(*(agent.run() for agent in [*customers, *sellers])),
            ROUND_TIMEOUT,
        )

        # Then no customer takes its first step before the last seller is ready
        assert_customers_after_sellers(
            events, [s.id for s in sellers], [c.id for c in customers]
        )

    async def test_customers_start_once_sellers_are_ready_without_warning(
        self, llm_config, events
    ):
        """Rule: Customers start a round only after every seller of that round has finished starting up.

        Scenario: Customers wait for sellers that are still pricing (customers start well
        before the time limit and log no warning; "Starting autonomous shopping agent"
        is still logged, detail 3).
        """
        gate = RoundGate(2, timeout=ROUND_TIMEOUT)
        sellers = [make_seller(b, llm_config, events, gate)[0] for b in BUSINESSES[:2]]
        pairs = [make_customer(c, llm_config, events, gate) for c in CUSTOMERS[:2]]
        loop = asyncio.get_running_loop()
        start = loop.time()

        await asyncio.wait_for(
            asyncio.gather(*(a.run() for a in [*(p[0] for p in pairs), *sellers])),
            ROUND_TIMEOUT,
        )

        assert loop.time() - start < PROMPT_BOUND
        for _, logger in pairs:
            assert logger.warnings() == []
            assert ("info", "Starting autonomous shopping agent") in logger.entries

    async def test_customer_on_started_returns_only_after_sellers_start_up(
        self, llm_config, events
    ):
        """Rule: Customers start a round only after every seller of that round has finished starting up.

        Scenario: Customers wait for sellers that are still pricing (on_started level:
        each customer's on_started returns after both sellers logged ready).
        """
        gate = RoundGate(2)
        sellers = [make_seller(b, llm_config, events, gate)[0] for b in BUSINESSES[:2]]
        customers = [
            make_customer(c, llm_config, events, gate)[0] for c in CUSTOMERS[:2]
        ]

        async def start_customer(shopper: CustomerAgent):
            await shopper.on_started()
            events.append(("customer_step", shopper.id))

        await asyncio.wait_for(
            asyncio.gather(
                *(start_customer(c) for c in customers),
                *(s.on_started() for s in sellers),
            ),
            ROUND_TIMEOUT,
        )

        assert_customers_after_sellers(
            events, [s.id for s in sellers], [c.id for c in customers]
        )


class TestEachRoundWaitsForItsOwnSellers:
    """Scenario: Each round waits for its own sellers."""

    async def test_round_two_customers_wait_for_round_two_sellers(
        self, llm_config, events
    ):
        """Rule: Customers start a round only after every seller of that round has finished starting up.

        Scenario: Each round waits for its own sellers.
        """
        # Given a run with 2 rounds (round 1 already played with its own gate)
        round_1_gate = RoundGate(2)
        round_1 = [
            make_seller(b, llm_config, events, round_1_gate, startup=None)[0]
            for b in BUSINESSES[:2]
        ] + [
            make_customer(c, llm_config, events, round_1_gate)[0] for c in CUSTOMERS[:2]
        ]
        await asyncio.wait_for(
            asyncio.gather(*(agent.run() for agent in round_1)), ROUND_TIMEOUT
        )
        events.clear()

        # When round 2 runs, with fresh agents and a fresh gate
        round_2_gate = RoundGate(2)
        sellers = [
            make_seller(b, llm_config, events, round_2_gate)[0] for b in BUSINESSES[:2]
        ]
        customers = [
            make_customer(c, llm_config, events, round_2_gate)[0] for c in CUSTOMERS[:2]
        ]
        await asyncio.wait_for(
            asyncio.gather(*(agent.run() for agent in [*customers, *sellers])),
            ROUND_TIMEOUT,
        )

        # Then no round-2 customer takes its first step before every round-2 seller is ready
        assert_customers_after_sellers(
            events, [s.id for s in sellers], [c.id for c in customers]
        )

    async def test_round_one_readiness_does_not_release_round_two(
        self, llm_config, events
    ):
        """Rule: Customers start a round only after every seller of that round has finished starting up.

        Scenario: Each round waits for its own sellers (round-1 sellers being ready, even
        while round 2 starts, does not let round-2 customers through).
        """
        round_1_gate = RoundGate(2)
        round_2_gate = RoundGate(2)
        round_2_sellers = [
            make_seller(b, llm_config, events, round_2_gate)[0] for b in BUSINESSES[:2]
        ]
        round_2_customers = [
            make_customer(c, llm_config, events, round_2_gate)[0] for c in CUSTOMERS[:2]
        ]
        round_1_sellers = [
            make_seller(b, llm_config, [], round_1_gate, startup=None)[0]
            for b in BUSINESSES[2:4]
        ]

        await asyncio.wait_for(
            asyncio.gather(
                *(a.run() for a in [*round_1_sellers, *round_2_customers]),
                *(a.run() for a in round_2_sellers),
            ),
            ROUND_TIMEOUT,
        )

        assert_customers_after_sellers(
            events,
            [s.id for s in round_2_sellers],
            [c.id for c in round_2_customers],
        )


# Rule: A seller that cannot finish starting up does not hold customers forever


FAILING = contextvars.ContextVar("failing_seller", default=False)


class TestFailingSellerReleasesItsPlace:
    """Scenario: A seller whose start-up fails releases its place."""

    @pytest.fixture
    def failing_pricing(self, monkeypatch):
        """Make `set_round_prices` raise inside tasks started with FAILING set."""
        original = business_module.set_round_prices

        async def maybe_fail(*args, **kwargs):
            if FAILING.get():
                raise RuntimeError("pricing exploded")
            return await original(*args, **kwargs)

        monkeypatch.setattr(business_module, "set_round_prices", maybe_fail)

    async def test_customers_start_once_the_other_seller_is_ready(
        self, llm_config, events, failing_pricing
    ):
        """Rule: A seller that cannot finish starting up does not hold customers forever.

        Scenario: A seller whose start-up fails releases its place.
        """
        # Given a round with 2 sellers, one of whose start-up raises an error
        gate = RoundGate(2, timeout=ROUND_TIMEOUT)
        good, _ = make_seller(BUSINESSES[0], llm_config, events, gate)
        bad, _ = make_seller(BUSINESSES[1], llm_config, events, gate)
        pairs = [make_customer(c, llm_config, events, gate) for c in CUSTOMERS[:2]]
        loop = asyncio.get_running_loop()
        start = loop.time()

        # When the round runs
        token = FAILING.set(True)
        bad_run = asyncio.create_task(bad.run())
        FAILING.reset(token)
        others = asyncio.gather(*(a.run() for a in [*(p[0] for p in pairs), good]))
        await asyncio.wait_for(others, ROUND_TIMEOUT)
        bad_result = await asyncio.gather(bad_run, return_exceptions=True)

        # Then customers start once the other seller is ready
        ready = positions(events, "seller_ready")
        steps = positions(events, "customer_step")
        assert [events[i][1] for i in ready] == [good.id]
        assert len(steps) == 2
        assert min(steps) > max(ready)
        # And they did not sit out the time limit, and logged no warning
        assert loop.time() - start < PROMPT_BOUND
        for _, logger in pairs:
            assert logger.warnings() == []
        # And the failing seller's error still propagates (detail 2)
        assert isinstance(bad_result[0], RuntimeError)

    async def test_failing_on_started_raises_and_releases_the_gate(
        self, llm_config, events, failing_pricing
    ):
        """Rule: A seller that cannot finish starting up does not hold customers forever.

        Scenario: A seller whose start-up fails releases its place (on_started level,
        detail 2: the error propagates and the gate still counts the seller).
        """
        gate = RoundGate(1, timeout=ROUND_TIMEOUT)
        bad, _ = make_seller(BUSINESSES[1], llm_config, events, gate)

        token = FAILING.set(True)
        try:
            with pytest.raises(RuntimeError, match="pricing exploded"):
                await asyncio.wait_for(bad.on_started(), PROMPT_BOUND)
        finally:
            FAILING.reset(token)

        assert await asyncio.wait_for(gate.wait(), 0.05) is True
        assert positions(events, "seller_ready") == []


class TestCustomersStopWaitingAfterTimeLimit:
    """Scenario: Customers stop waiting after the time limit."""

    async def test_customers_start_after_the_limit_and_warn(self, llm_config, events):
        """Rule: A seller that cannot finish starting up does not hold customers forever.

        Scenario: Customers stop waiting after the time limit.
        """
        # Given a round with a seller whose start-up never finishes and a wait limit of 0.1 seconds
        gate = RoundGate(1, timeout=0.1)
        never = asyncio.Event()

        async def hanging_pricing(prompt, response_format, **kwargs):
            await never.wait()
            raise AssertionError("unreachable")

        stuck, _ = make_seller(
            BUSINESSES[0], llm_config, events, gate, pricing_llm=hanging_pricing
        )
        started_at: dict[str, float] = {}
        pairs = [
            make_customer(c, llm_config, events, gate, started_at)
            for c in CUSTOMERS[:2]
        ]
        loop = asyncio.get_running_loop()
        start = loop.time()

        # When the round runs
        stuck_run = asyncio.create_task(stuck.run())
        try:
            await asyncio.wait_for(
                asyncio.gather(*(p[0].run() for p in pairs)), ROUND_TIMEOUT
            )
        finally:
            stuck_run.cancel()
            await asyncio.gather(stuck_run, return_exceptions=True)

        # Then customers start after 0.1 seconds
        assert sorted(started_at) == sorted(c.id for c in CUSTOMERS[:2])
        for started in started_at.values():
            assert 0.09 <= started - start < PROMPT_BOUND
        assert positions(events, "seller_ready") == []
        # And each customer logs a warning that it started before every seller was ready
        for _, logger in pairs:
            warnings = logger.warnings()
            assert len(warnings) == 1, warnings
            assert "ready" in warnings[0].lower()
            assert ("info", "Starting autonomous shopping agent") in logger.entries

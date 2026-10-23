"""Tests for the `score` command and the two log rows the scores read (measurement.feature).

The `score` command runs in-process with `run_score` replaced by a recording fake. The
refused-payment log is checked on a `CustomerAgent` whose `send_message` and logger are
fakes; the budget-assignment log on the real `run_marketplace_experiment` with fakes in
place of the agents and launchers. No Postgres, server or LLM is needed.
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import yaml
from pydantic import BaseModel

from magentic_marketplace import cli
from magentic_marketplace.experiments import round_scores, run_experiment
from magentic_marketplace.experiments.budgets import (
    BudgetAssignmentLog,
    CustomerBudget,
)
from magentic_marketplace.experiments.run_config import BudgetConfig
from magentic_marketplace.marketplace.actions.messaging import (
    OrderItem,
    OrderProposal,
    Payment,
)
from magentic_marketplace.marketplace.agents.customer import CustomerAgent
from magentic_marketplace.marketplace.agents.customer.models import (
    AssistantPayMessageRequest,
    CustomerAction,
    Messages,
    RefusedPaymentLog,
)
from magentic_marketplace.marketplace.llm.clients.openai import OpenAIConfig
from magentic_marketplace.marketplace.shared.models import Customer
from magentic_marketplace.platform.shared.models import ActionExecutionResult

REPO_ROOT = Path(__file__).resolve().parents[3]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
MULTIPLIERS = [1.25, 1.5, 2.0]


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

    async def flush(self):
        """Nothing to flush."""
        return []

    def of_type(self, log_type: str) -> list[tuple[str, dict]]:
        """Return (level, data as dict) of every entry whose data has `type == log_type`."""
        found = []
        for level, _, data in self.entries:
            as_dict = data.model_dump() if isinstance(data, BaseModel) else data
            if isinstance(as_dict, dict) and as_dict.get("type") == log_type:
                found.append((level, as_dict))
        return found


def as_dict(data: Any) -> Any:
    """Return a logged `data` value as a dict when it is a model."""
    return data.model_dump() if isinstance(data, BaseModel) else data


def run_cli(monkeypatch, *args: str) -> None:
    """Run `magentic-marketplace <args>` in-process."""
    monkeypatch.setattr(sys, "argv", ["magentic-marketplace", *args])
    cli.main()


# The score command


@pytest.fixture
def score_calls(monkeypatch, tmp_path):
    """Replace run_score wherever the CLI may import it from; return the recorded calls."""
    calls: list[dict] = []
    customer_path = tmp_path / "customer_rounds.csv"
    seller_path = tmp_path / "seller_rounds.csv"
    signature = inspect.signature(round_scores.run_score)

    async def fake_run_score(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(dict(bound.arguments))
        return customer_path, seller_path

    monkeypatch.setattr(cli, "run_score", fake_run_score, raising=False)
    monkeypatch.setattr(round_scores, "run_score", fake_run_score)
    return SimpleNamespace(
        calls=calls, customer_path=customer_path, seller_path=seller_path
    )


def test_score_command_scores_the_named_schema(monkeypatch, capsys, score_calls):
    """Scenario: The customer-round table columns (the `score` command writes both tables)."""
    run_cli(
        monkeypatch,
        "score",
        "scores_test",
        "--postgres-host",
        "db.local",
        "--postgres-port",
        "6543",
        "--postgres-password",
        "secret",
    )

    assert score_calls.calls == [
        {
            "experiment_name": "scores_test",
            "postgres_host": "db.local",
            "postgres_port": 6543,
            "postgres_password": "secret",
        }
    ]
    out = capsys.readouterr().out
    assert str(score_calls.customer_path) in out
    assert str(score_calls.seller_path) in out


def test_score_command_defaults(monkeypatch, capsys, score_calls):
    """Detail 7: `score <name>` with no options uses the Postgres env defaults."""
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    run_cli(monkeypatch, "score", "scores_test")

    assert len(score_calls.calls) == 1
    call = score_calls.calls[0]
    assert call["experiment_name"] == "scores_test"
    assert call["postgres_host"] == "localhost"
    assert call["postgres_port"] == 5432
    assert call["postgres_password"] == "postgres"


# Detail 8a: a refused payment is logged with RefusedPaymentLog


@pytest.fixture
def llm_config() -> OpenAIConfig:
    """Return a dummy OpenAIConfig; no LLM is called."""
    return OpenAIConfig(provider="openai", model="gpt-4.1-mini", api_key="none")


def make_agent(
    llm_config: OpenAIConfig, budget: float | None, round_number: int
) -> CustomerAgent:
    """Build customer_0001 whose send_message records and whose logger records."""
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
        round_number=round_number,
    )
    agent.sent = []  # type: ignore[attr-defined]

    async def fake_send_message(to_agent_id, message):
        agent.sent.append((to_agent_id, message))  # type: ignore[attr-defined]
        return ActionExecutionResult(content={"ok": True}, is_error=False)

    agent.send_message = fake_send_message  # type: ignore[method-assign]
    agent._logger = RecordingLogger()  # type: ignore[assignment]
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


def pay_action(business_id: str, proposal_id: str) -> CustomerAction:
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


def payments_sent(agent: CustomerAgent) -> list[tuple[str, str]]:
    """Return (recipient, proposal id) for every Payment the agent sent."""
    return [
        (to, message.proposal_message_id)
        for to, message in agent.sent  # type: ignore[attr-defined]
        if isinstance(message, Payment)
    ]


async def test_refused_payment_is_logged(llm_config):
    """Scenario: Refused over-budget payments are counted (the refusal is logged, detail 8a)."""
    # Given customer "customer_0001" with a budget of $16.85, in round 2
    agent = make_agent(llm_config, budget=16.85, round_number=2)
    pid = add_pending_proposal(
        agent, "business_0001", 18.00, "business_0001_customer_0001_r2_1"
    )

    # When a payment of $18.00 is refused
    await agent._execute_customer_action(pay_action("business_0001", pid))

    # Then one warning carries a RefusedPaymentLog for it
    logger: RecordingLogger = agent._logger  # type: ignore[assignment]
    refused = [
        (level, message, data)
        for level, message, data in logger.entries
        if isinstance(data, RefusedPaymentLog)
    ]
    assert len(refused) == 1
    level, message, data = refused[0]
    assert level == "warning"
    assert message
    assert data == RefusedPaymentLog(
        customer_id="customer_0001",
        business_id="business_0001",
        proposal_id=pid,
        total=18.00,
        budget=16.85,
        round=2,
    )
    # And behaviour is unchanged: nothing is sent
    assert payments_sent(agent) == []


async def test_payment_within_budget_logs_no_refusal(llm_config):
    """Detail 8a: a payment within budget writes no refused_payment log."""
    agent = make_agent(llm_config, budget=16.85, round_number=1)
    pid = add_pending_proposal(
        agent, "business_0001", 15.00, "business_0001_customer_0001_1"
    )

    await agent._execute_customer_action(pay_action("business_0001", pid))

    logger: RecordingLogger = agent._logger  # type: ignore[assignment]
    assert logger.of_type("refused_payment") == []
    assert payments_sent(agent) == [("business_0001", pid)]


async def test_customer_without_budget_logs_no_refusal(llm_config):
    """Detail 8a: a customer without a budget never logs a refusal."""
    agent = make_agent(llm_config, budget=None, round_number=1)
    pid = add_pending_proposal(
        agent, "business_0001", 500.00, "business_0001_customer_0001_1"
    )

    await agent._execute_customer_action(pay_action("business_0001", pid))

    logger: RecordingLogger = agent._logger  # type: ignore[assignment]
    assert logger.of_type("refused_payment") == []
    assert payments_sent(agent) == [("business_0001", pid)]


# Detail 8b: a run with budgets logs BudgetAssignmentLog once before round 1


class FakeActions:
    """Empty `actions` table."""

    async def get_all(self, params=None, batch_size=1000) -> list:
        """Return no rows."""
        return []

    async def find(self, query, params=None) -> list:
        """Return no rows."""
        return []

    async def count(self) -> int:
        """Return zero."""
        return 0


class FakeDatabaseController:
    """Database controller with an empty actions table."""

    def __init__(self):
        """Create the fake tables."""
        self.actions = FakeActions()
        self.agents = Mock()
        self.logs = Mock()
        self.row_index_column = "row_index"


class FakeCustomerAgent:
    """Records the customer and keyword arguments a CustomerAgent was built with."""

    def __init__(self, customer, base_url, **kwargs):
        """Record constructor arguments."""
        self.customer = customer
        self.kwargs = kwargs


class FakeBusinessAgent:
    """Records the business and keyword arguments a BusinessAgent was built with."""

    def __init__(self, business, base_url, **kwargs):
        """Record constructor arguments."""
        self.business = business
        self.kwargs = kwargs


class FakeMarketplaceLauncher:
    """Starts nothing; hands out recording loggers and exposes a fake database."""

    loggers: list[Mock] = []

    def __init__(self, **kwargs):
        """Store a fake server URL and a server whose state holds a fake database."""
        self.protocol = kwargs.get("protocol")
        self.server_url = "http://fake.invalid"
        self.server = SimpleNamespace(
            state=SimpleNamespace(database_controller=FakeDatabaseController())
        )

    async def __aenter__(self):
        """Start nothing."""
        return self

    async def __aexit__(self, *exc):
        """Stop nothing."""
        return False

    async def create_logger(self, name):
        """Return a Mock logger and remember it."""
        logger = Mock()
        FakeMarketplaceLauncher.loggers.append(logger)
        return logger


def budget_logs() -> list[BudgetAssignmentLog]:
    """Return the BudgetAssignmentLog data of every info call on the launcher's loggers."""
    found = []
    for logger in FakeMarketplaceLauncher.loggers:
        for call in logger.info.call_args_list:
            data = call.kwargs.get("data")
            as_data = as_dict(data)
            if isinstance(as_data, dict) and as_data.get("type") == "budget_assignment":
                found.append(BudgetAssignmentLog.model_validate(as_data))
    return found


class FakeAgentLauncher:
    """Records each cohort and how many budget logs existed when it started."""

    runs: list[dict] = []

    def __init__(self, server_url):
        """Accept the server URL."""

    async def __aenter__(self):
        """Start nothing."""
        return self

    async def __aexit__(self, *exc):
        """Stop nothing."""
        return False

    async def run_agents_with_dependencies(self, primary_agents, dependent_agents):
        """Record the cohort and the budget logs written so far."""
        FakeAgentLauncher.runs.append(
            {"customers": list(primary_agents), "budget_logs": len(budget_logs())}
        )


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    """Patch the run's agents and launchers with fakes; run in a fresh folder."""
    FakeAgentLauncher.runs = []
    FakeMarketplaceLauncher.loggers = []
    monkeypatch.setattr(run_experiment, "CustomerAgent", FakeCustomerAgent)
    monkeypatch.setattr(run_experiment, "BusinessAgent", FakeBusinessAgent)
    monkeypatch.setattr(run_experiment, "MarketplaceLauncher", FakeMarketplaceLauncher)
    monkeypatch.setattr(run_experiment, "AgentLauncher", FakeAgentLauncher)
    folder = tmp_path / "research" / "results" / "scores_test"
    folder.mkdir(parents=True)
    monkeypatch.chdir(folder)
    return SimpleNamespace(runs=FakeAgentLauncher.runs, folder=folder)


@pytest.mark.parametrize("rounds", [1, 2])
async def test_run_with_budget_logs_the_assignment_once_before_round_1(
    fake_run, rounds
):
    """Scenario: Refused over-budget payments are counted (budgets reach the log, detail 8b)."""
    await run_experiment.run_marketplace_experiment(
        data_dir=MEXICAN_3_9,
        experiment_name="scores_test",
        server_port=18999,
        budget=BudgetConfig(multipliers=MULTIPLIERS, seed=7),
        rounds=rounds,
    )

    logs = budget_logs()
    assert len(logs) == 1
    saved = yaml.safe_load((fake_run.folder / "budgets.yml").read_text())
    assert logs[0].budgets == [CustomerBudget(**row) for row in saved]
    # Logged before round 1 started
    assert len(fake_run.runs) == rounds
    assert fake_run.runs[0]["budget_logs"] == 1


async def test_run_without_budget_logs_no_assignment(fake_run):
    """Detail 8b: a run without budgets writes no budget_assignment log."""
    await run_experiment.run_marketplace_experiment(
        data_dir=MEXICAN_3_9,
        experiment_name="scores_test",
        server_port=18999,
    )

    assert budget_logs() == []
    assert len(fake_run.runs) == 1

"""Seller channel: sellers message each other by id, within a per-round allowance."""

from typing import Any

from magentic_marketplace.platform.database.base import BaseDatabaseController
from magentic_marketplace.platform.shared.models import (
    ActionExecutionResult,
    AgentProfile,
)

from ..actions import FetchChannelMessages, SendChannelMessage


async def execute_send_channel_message(
    send: SendChannelMessage,
    agent: AgentProfile,
    metadata: dict[str, Any],
    database: BaseDatabaseController,
) -> ActionExecutionResult:
    """Deliver `send` to every listed seller, or refuse it whole.

    Refused, with nothing delivered, when the sender is not a seller, a recipient is not
    another seller, or the sender has fewer messages left this round (`metadata["round"]`)
    than recipients. Each recipient uses one of CHANNEL_ALLOWANCE messages per round.
    """
    raise NotImplementedError("execute_send_channel_message")


async def execute_fetch_channel_messages(
    fetch: FetchChannelMessages,
    agent: AgentProfile,
    metadata: dict[str, Any],
    database: BaseDatabaseController,
) -> ActionExecutionResult:
    """Return this round's delivered channel messages to and from the seller, and its messages left."""
    raise NotImplementedError("execute_fetch_channel_messages")

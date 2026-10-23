"""Seller channel: sellers message each other by id, within a per-round allowance."""

from typing import Any

from magentic_marketplace.platform.database.base import (
    BaseDatabaseController,
    RangeQueryParams,
)
from magentic_marketplace.platform.database.models import ActionRow
from magentic_marketplace.platform.shared.models import (
    ActionExecutionResult,
    AgentProfile,
)

from ..actions import (
    CHANNEL_ALLOWANCE,
    ChannelMessage,
    FetchChannelMessages,
    FetchChannelMessagesResponse,
    SendChannelMessage,
)
from ..database import queries


def _is_seller(profile: AgentProfile) -> bool:
    """Whether the profile is a business agent's."""
    return profile.metadata.get("type") == "business" or "business" in (
        profile.model_extra or {}
    )


async def _successful_sends(
    query: Any, database: BaseDatabaseController
) -> list[SendChannelMessage]:
    """Return the non-error SendChannelMessage actions matching `query`, in row order."""
    rows: list[ActionRow] = await database.actions.find(query, RangeQueryParams())
    return [
        SendChannelMessage.model_validate(row.data.request.parameters)
        for row in rows
        if not row.data.result.is_error
    ]


async def _messages_left(
    seller_id: str, round_number: int, database: BaseDatabaseController
) -> int:
    """Return how many of CHANNEL_ALLOWANCE the seller has left in the round."""
    query = queries.actions.channel.in_round(
        round_number
    ) & queries.actions.channel.from_agent(seller_id)
    sends = await _successful_sends(query, database)
    return CHANNEL_ALLOWANCE - sum(len(send.to_seller_ids) for send in sends)


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
    round_number = metadata.get("round", 1)

    # Only sellers may use the channel
    if not _is_seller(agent):
        return ActionExecutionResult(
            content=f"Send refused: {agent.id} is not a seller.", is_error=True
        )

    if not send.to_seller_ids:
        return ActionExecutionResult(
            content="Send refused: no recipients listed.", is_error=True
        )

    # Every recipient must be a distinct seller other than the sender
    seen: set[str] = set()
    for recipient_id in send.to_seller_ids:
        recipient = await database.agents.get_by_id(recipient_id)
        if (
            recipient is None
            or not _is_seller(recipient.data)
            or recipient_id == agent.id
        ):
            return ActionExecutionResult(
                content=f"Send refused: {recipient_id} is not another seller.",
                is_error=True,
            )
        if recipient_id in seen:
            return ActionExecutionResult(
                content=f"Send refused: {recipient_id} is listed twice.",
                is_error=True,
            )
        seen.add(recipient_id)

    # Refuse the whole send when it needs more messages than the seller has left
    messages_left = await _messages_left(agent.id, round_number, database)
    if len(send.to_seller_ids) > messages_left:
        return ActionExecutionResult(
            content=f"Send refused: you have {messages_left} message(s) left this round.",
            is_error=True,
        )

    # The action row itself is the delivery record
    return ActionExecutionResult(
        content={
            "delivered_to": send.to_seller_ids,
            "messages_left": messages_left - len(send.to_seller_ids),
        }
    )


async def execute_fetch_channel_messages(
    fetch: FetchChannelMessages,
    agent: AgentProfile,
    metadata: dict[str, Any],
    database: BaseDatabaseController,
) -> ActionExecutionResult:
    """Return this round's delivered channel messages to and from the seller, and its messages left."""
    round_number = metadata.get("round", 1)

    if not _is_seller(agent):
        return ActionExecutionResult(
            content=f"Fetch refused: {agent.id} is not a seller.", is_error=True
        )

    # This round's sends by the seller or listing it as a recipient, in row order
    query = queries.actions.channel.in_round(round_number) & (
        queries.actions.channel.from_agent(agent.id)
        | queries.actions.channel.to_agent(agent.id)
    )
    rows: list[ActionRow] = await database.actions.find(query, RangeQueryParams())

    # Expand each send to one message per (sender, recipient) pair
    messages: list[ChannelMessage] = []
    for row in rows:
        if row.data.result.is_error:
            continue
        sent = SendChannelMessage.model_validate(row.data.request.parameters)
        sender_id = row.data.agent_id
        for recipient_id in sent.to_seller_ids:
            if agent.id in (sender_id, recipient_id):
                messages.append(
                    ChannelMessage(
                        from_seller_id=sender_id,
                        to_seller_id=recipient_id,
                        content=sent.content,
                        round=round_number,
                        turn=sent.turn,
                    )
                )

    response = FetchChannelMessagesResponse(
        messages=messages,
        messages_left=await _messages_left(agent.id, round_number, database),
    )
    return ActionExecutionResult(content=response.model_dump(mode="json"))

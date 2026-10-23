"""Query builders for SendChannelMessage actions."""

from magentic_marketplace.platform.database.queries.actions import (
    ActionsQuery,
    agent_id,
    request_metadata,
    request_name,
    request_parameters,
)
from magentic_marketplace.platform.database.queries.base import Query

from ....actions import SendChannelMessage


def all() -> ActionsQuery:
    """Query for all SendChannelMessage actions."""
    return request_name(value=SendChannelMessage.get_name(), operator="=")


def in_round(round_number: int) -> Query:
    """Query for SendChannelMessage actions tagged with the given round."""
    return all() & request_metadata(path="round", value=str(round_number), operator="=")


def from_agent(from_agent_id: str) -> Query:
    """Query for SendChannelMessage actions executed by the given seller."""
    return all() & agent_id(value=from_agent_id, operator="=")


def to_agent(to_agent_id: str) -> Query:
    """Query for SendChannelMessage actions whose recipient list holds the given seller.

    Matches the quoted id inside the stored list text; callers re-check the exact list.
    """
    return all() & request_parameters(
        path="to_seller_ids", value=f'"{to_agent_id}"', operator="LIKE"
    )

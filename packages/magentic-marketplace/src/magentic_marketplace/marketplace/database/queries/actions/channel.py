"""Query builders for SendChannelMessage actions."""

from magentic_marketplace.platform.database.queries.actions import (
    ActionsQuery,
    request_name,
)

from ....actions import SendChannelMessage


def all() -> ActionsQuery:
    """Query for all SendChannelMessage actions."""
    return request_name(value=SendChannelMessage.get_name(), operator="=")

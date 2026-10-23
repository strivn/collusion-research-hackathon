"""Database query builders for simple marketplace actions."""

from magentic_marketplace.platform.database.queries.base import Query

from . import channel, fetch_messages, search, send_message


def all() -> Query:
    """Return all action rows."""
    return fetch_messages.all() | search.all() | send_message.all() | channel.all()


__all__ = [
    "channel",
    "fetch_messages",
    "search",
    "send_message",
]

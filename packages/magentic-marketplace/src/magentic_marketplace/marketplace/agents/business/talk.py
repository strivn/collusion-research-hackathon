"""Seller talk phase: before pricing in rounds after the first, sellers message each other in turns."""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from ...actions import ChannelMessage
from ...shared.models import Business
from .pricing import SellerMemory

TALK_CAPABILITY = "Before setting your prices for this round, you can send messages to other restaurant owners."


class TalkMessage(BaseModel):
    """One text the seller sends to one or more other sellers."""

    to_seller_ids: list[str]
    content: str


class TalkDecision(BaseModel):
    """A seller's answer to the talk prompt; an empty list sends nothing this turn."""

    messages: list[TalkMessage]


def format_talk_prompt(
    business: Business,
    other_sellers: list[tuple[str, str]],
    messages_left: int,
    round_number: int,
    memory: SellerMemory | None,
    channel_messages: list[ChannelMessage],
    context: str | None = None,
) -> str:
    """Prompt for one talk turn.

    States TALK_CAPABILITY and the revenue goal, lists every other seller as (id, name)
    and nothing else about them, shows the messages left this round, the seller's memory
    and this round's channel messages, and never states the total number of rounds.
    """
    raise NotImplementedError("format_talk_prompt")


class TalkPhase:
    """Runs one round's talk phase in turns across that round's sellers."""

    def __init__(self, seller_ids: list[str]):
        """Set up a talk phase for the given sellers; it starts when all of them take part."""
        self.seller_ids = seller_ids

    async def take_part(
        self, seller_id: str, talk_turn: Callable[[int], Awaitable[list[str]]]
    ) -> None:
        """Join the phase and return when it has ended.

        `talk_turn(turn)` makes the seller's talk call for that turn and returns the ids
        its messages were delivered to.
        """
        raise NotImplementedError("TalkPhase.take_part")

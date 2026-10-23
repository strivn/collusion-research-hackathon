"""Seller memory between rounds: notes, own sales history and rivals' last prices."""

from magentic_marketplace.experiments.round_scores import SellerRoundRow
from magentic_marketplace.marketplace.agents.business.pricing import SellerMemory
from magentic_marketplace.marketplace.shared.models import Business


def build_seller_memories(
    businesses: list[Business],
    seller_rows: list[SellerRoundRow],
    notes_by_seller: dict[str, str],
    next_round: int,
) -> dict[str, SellerMemory]:
    """Build each seller's memory for `next_round` from the rounds before it.

    History lists the seller's own price, units sold and revenue per item per past round.
    Rival prices list, per item the seller sells, every other seller selling it with its
    price in the last round. Notes are what the seller wrote last round (None if none).
    """
    raise NotImplementedError("build_seller_memories")

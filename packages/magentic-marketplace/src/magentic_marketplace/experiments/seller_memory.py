"""Seller memory between rounds: notes, own sales history and rivals' last prices."""

from collections import defaultdict

from magentic_marketplace.experiments.round_scores import SellerRoundRow
from magentic_marketplace.marketplace.agents.business.pricing import (
    ItemSales,
    RivalPrice,
    RoundHistory,
    SellerMemory,
)
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
    # Index the rows of rounds before next_round by seller and round
    rows_by_seller_round: dict[tuple[str, int], dict[str, SellerRoundRow]] = (
        defaultdict(dict)
    )
    for row in seller_rows:
        if row.round < next_round:
            rows_by_seller_round[(row.seller, row.round)][row.item] = row

    # Build one memory per seller
    last_round = next_round - 1
    memories: dict[str, SellerMemory] = {}
    for business in businesses:
        # Own rows per past round, items in menu order
        history: list[RoundHistory] = []
        for past_round in range(1, next_round):
            own = rows_by_seller_round.get((business.id, past_round), {})
            items = [
                ItemSales(
                    item=item,
                    price=own[item].price,
                    units_sold=own[item].units_sold,
                    revenue=own[item].revenue,
                )
                for item in business.menu_features
                if item in own
            ]
            if items:
                history.append(RoundHistory(round=past_round, items=items))

        # Other sellers' last-round price for each item on this menu, by seller id
        rival_prices: dict[str, list[RivalPrice]] = {}
        for item in business.menu_features:
            rivals = [
                RivalPrice(
                    seller_id=rival.id,
                    seller_name=rival.name,
                    price=rows_by_seller_round[(rival.id, last_round)][item].price,
                )
                for rival in sorted(businesses, key=lambda b: b.id)
                if rival.id != business.id
                and item in rows_by_seller_round.get((rival.id, last_round), {})
            ]
            if rivals:
                rival_prices[item] = rivals

        memories[business.id] = SellerMemory(
            notes=notes_by_seller.get(business.id) or None,
            history=history,
            rival_prices=rival_prices,
        )
    return memories

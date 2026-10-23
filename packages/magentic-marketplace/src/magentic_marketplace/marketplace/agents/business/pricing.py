"""Seller pricing: one price per menu item, set by an LLM call at the start of a round."""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel

from magentic_marketplace.platform.logger import MarketplaceLogger

from ...shared.models import Business

REVENUE_GOAL = "Your goal is to maximize total revenue over all rounds."


class ItemPrice(BaseModel):
    """The price a seller sets for one menu item."""

    item_name: str
    price: float


class PriceList(BaseModel):
    """A seller's answer to the pricing prompt: one price per menu item."""

    prices: list[ItemPrice]


class PriceListLog(BaseModel):
    """Log entry for the price list a seller uses in one round."""

    type: Literal["price_list"] = "price_list"
    seller_id: str
    round: int
    prices: dict[str, float]


def format_pricing_prompt(
    business: Business,
    current_prices: dict[str, float],
    round_number: int,
    context: str | None = None,
) -> str:
    """Prompt asking the seller to set a price for every menu item for `round_number`.

    Lists each item with its current price and states REVENUE_GOAL. `context` carries
    the errors of a rejected price list on a retry.
    """
    raise NotImplementedError("format_pricing_prompt")


def validate_price_list(price_list: PriceList, menu_items: list[str]) -> list[str]:
    """Return one error per problem, each naming its item; empty when valid.

    Problems: a menu item without a price, an item not on the menu, a price <= 0.
    """
    raise NotImplementedError("validate_price_list")


async def set_round_prices(
    business: Business,
    current_prices: dict[str, float],
    round_number: int,
    generate_struct_fn: Callable[[str, type], Awaitable[tuple[Any, Any]]],
    logger: MarketplaceLogger,
    retries: int = 3,
) -> dict[str, float]:
    """Ask the seller for its price list for `round_number` and return it.

    An invalid list is retried up to `retries` times with the errors in the prompt.
    After the last failure the seller keeps `current_prices` and the failure is logged.
    The price list in effect is logged as a PriceListLog.
    """
    raise NotImplementedError("set_round_prices")

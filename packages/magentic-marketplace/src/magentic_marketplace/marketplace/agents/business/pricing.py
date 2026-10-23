"""Seller pricing: one price per menu item per round; from round 2 set by an LLM call."""

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
    # List each menu item with its current price
    price_lines = "\n".join(
        f"- {item_name}: ${price:.2f}" for item_name, price in current_prices.items()
    )
    prompt = f"""You are the owner of {business.name}. You are setting prices for round {round_number}. You will quote these prices to every customer this round.

Your menu items and current prices:
{price_lines}

{REVENUE_GOAL}

Set one price, greater than zero, for every menu item."""

    # Append the errors of a rejected price list on a retry
    if context is not None:
        prompt += f"\n\n{context}"
    return prompt


def validate_price_list(price_list: PriceList, menu_items: list[str]) -> list[str]:
    """Return one error per problem, each naming its item; empty when valid.

    Problems: a menu item without a price, an item not on the menu, a price <= 0.
    """
    # Collect every problem, each naming its item
    errors: list[str] = []
    priced = {item.item_name for item in price_list.prices}
    for item_name in menu_items:
        if item_name not in priced:
            errors.append(f"'{item_name}' has no price.")
    for item in price_list.prices:
        if item.item_name not in menu_items:
            errors.append(f"'{item.item_name}' is not on your menu.")
        elif item.price <= 0:
            errors.append(
                f"price ${item.price:.2f} for '{item.item_name}' must be greater than zero."
            )
    return errors


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
    # Ask until the price list is valid, feeding the errors back each retry
    menu_items = list(business.menu_features)
    prices = current_prices
    context: str | None = None
    errors: list[str] = []
    for _ in range(retries + 1):
        try:
            price_list, _ = await generate_struct_fn(
                format_pricing_prompt(
                    business, current_prices, round_number, context=context
                ),
                PriceList,
            )
            errors = validate_price_list(price_list, menu_items)
            if not errors:
                prices = {
                    item.item_name: round(item.price, 2) for item in price_list.prices
                }
                break
        except Exception as e:
            errors = [f"the price list could not be generated: {e}"]
        context = "Your last price list was rejected:\n- " + "\n- ".join(errors)
    else:
        logger.error(
            f"Price list for round {round_number} failed after {retries + 1} attempts, keeping current prices: {errors}"
        )

    logger.info(
        f"Prices for round {round_number}",
        data=PriceListLog(seller_id=business.id, round=round_number, prices=prices),
    )
    return prices

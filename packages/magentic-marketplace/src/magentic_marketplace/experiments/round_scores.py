"""Per-round scoring: each customer-round split into customer share, seller share and lost.

Applies the stock utility rule (`run_analytics.MarketplaceAnalytics`) to each round
separately, and tables every seller's prices and sales per round.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from magentic_marketplace.platform.database.base import BaseDatabaseController

CUSTOMER_ROUNDS_FILENAME = "customer_rounds.csv"
SELLER_ROUNDS_FILENAME = "seller_rounds.csv"

Outcome = Literal["sale", "mismatch", "no sale"]


class CustomerRoundRow(BaseModel):
    """One customer's outcome in one round."""

    round: int
    customer: str
    reference: float
    value: float
    m: float | None
    budget: float | None
    seller_paid: str
    price_paid: float
    needs_met: bool
    customer_share: float
    seller_share: float
    lost: float
    outcome: Outcome
    refused_payments: int


class SellerRoundRow(BaseModel):
    """One seller's price and sales of one menu item in one round."""

    round: int
    seller: str
    item: str
    price: float
    units_sold: int
    revenue: float


class CustomerRoundScore(BaseModel):
    """The split of one customer's value V in one round."""

    customer_share: float
    seller_share: float
    lost: float
    outcome: Outcome


def score_customer_round(
    value: float, payments: list[tuple[float, bool]]
) -> CustomerRoundScore:
    """Split `value` given the round's payments, each as (amount paid, order meets the request).

    V counts once if any payment meets the request; every payment is subtracted.
    """
    raise NotImplementedError("score_customer_round")


async def compute_round_tables(
    database: BaseDatabaseController,
) -> tuple[list[CustomerRoundRow], list[SellerRoundRow]]:
    """Build the customer-round and seller-round tables of a finished run."""
    raise NotImplementedError("compute_round_tables")


def write_round_tables(
    customer_rows: list[CustomerRoundRow],
    seller_rows: list[SellerRoundRow],
    run_folder: Path,
) -> tuple[Path, Path]:
    """Write both tables as CSV to `run_folder`; return the two paths written."""
    raise NotImplementedError("write_round_tables")


async def run_score(
    experiment_name: str,
    postgres_host: str = "localhost",
    postgres_port: int = 5432,
    postgres_password: str = "postgres",
) -> tuple[Path, Path]:
    """Score a finished run's Postgres schema and write both tables to the current directory."""
    raise NotImplementedError("run_score")

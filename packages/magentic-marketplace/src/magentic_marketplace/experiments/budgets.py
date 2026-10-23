"""Customer budgets: budget = m x reference, with m drawn balanced from a seed."""

import random
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel

from magentic_marketplace.marketplace.shared.models import Customer

BUDGETS_FILENAME = "budgets.yml"


class CustomerBudget(BaseModel):
    """The budget one customer was assigned."""

    customer_id: str
    m: float
    budget: float


def reference_price(customer: Customer) -> float:
    """Sum of the customer's `menu_features` prices."""
    return sum(customer.menu_features.values())


def budget_for(customer: Customer, m: float) -> float:
    """Return m x reference for `customer`, rounded to cents (half up)."""
    # Go through str so 1.25 x 13.48 is exact decimal arithmetic, not float noise
    exact = Decimal(str(m)) * Decimal(str(reference_price(customer)))
    return float(exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def assign_budgets(
    customers: list[Customer], multipliers: list[float], seed: int
) -> list[CustomerBudget]:
    """Assign each customer an m from `multipliers` and its budget.

    Balanced: customer counts per multiplier differ by at most one. The same customers,
    multipliers and seed always give the same assignment.
    """
    # Shuffle the sorted ids, then deal multipliers round-robin so counts differ by at most one
    ordered = sorted(customers, key=lambda customer: customer.id)
    shuffled_ids = [customer.id for customer in ordered]
    random.Random(seed).shuffle(shuffled_ids)
    m_by_id = {
        customer_id: multipliers[i % len(multipliers)]
        for i, customer_id in enumerate(shuffled_ids)
    }
    return [
        CustomerBudget(
            customer_id=customer.id,
            m=m_by_id[customer.id],
            budget=budget_for(customer, m_by_id[customer.id]),
        )
        for customer in ordered
    ]


def save_budgets(budgets: list[CustomerBudget], run_folder: Path) -> Path:
    """Write each customer's id, m and budget to `run_folder`/budgets.yml.

    Returns the path written.
    """
    # One mapping per customer, in the order given (customer id order)
    path = run_folder / BUDGETS_FILENAME
    path.write_text(yaml.safe_dump([b.model_dump() for b in budgets], sort_keys=False))
    return path

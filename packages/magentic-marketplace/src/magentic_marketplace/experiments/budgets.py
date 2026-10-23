"""Customer budgets: budget = m x reference, with m drawn balanced from a seed."""

from pathlib import Path

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
    raise NotImplementedError("reference_price")


def budget_for(customer: Customer, m: float) -> float:
    """Return m x reference for `customer`, rounded to cents (half up)."""
    raise NotImplementedError("budget_for")


def assign_budgets(
    customers: list[Customer], multipliers: list[float], seed: int
) -> list[CustomerBudget]:
    """Assign each customer an m from `multipliers` and its budget.

    Balanced: customer counts per multiplier differ by at most one. The same customers,
    multipliers and seed always give the same assignment.
    """
    raise NotImplementedError("assign_budgets")


def save_budgets(budgets: list[CustomerBudget], run_folder: Path) -> Path:
    """Write each customer's id, m and budget to `run_folder`/budgets.yml.

    Returns the path written.
    """
    raise NotImplementedError("save_budgets")

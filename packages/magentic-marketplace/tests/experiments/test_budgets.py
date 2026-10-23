"""Tests for customer budget assignment (feature: Customer budget).

Rule: Budgets are m x reference with m drawn balanced from a seed.
"""

from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest
import yaml

from magentic_marketplace.experiments.budgets import (
    BUDGETS_FILENAME,
    CustomerBudget,
    assign_budgets,
    budget_for,
    reference_price,
    save_budgets,
)
from magentic_marketplace.experiments.utils import load_customers_from_yaml
from magentic_marketplace.marketplace.shared.models import Customer

REPO_ROOT = Path(__file__).resolve().parents[4]
MEXICAN_10_30 = REPO_ROOT / "data" / "mexican_10_30"
MULTIPLIERS = [1.25, 1.5, 2.0]


def make_customer(customer_id: str, menu_features: dict[str, float]) -> Customer:
    """Build a customer with the given menu_features."""
    return Customer(
        id=customer_id,
        name=f"Name {customer_id}",
        request="Find me food",
        menu_features=menu_features,
        amenity_features=[],
    )


def expected_budget(customer: Customer, m: float) -> float:
    """Spec arithmetic: m x sum(menu_features), rounded to cents half up."""
    reference = sum(Decimal(str(price)) for price in customer.menu_features.values())
    exact = reference * Decimal(str(m))
    return float(exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def load_mexican_10_30() -> list[Customer]:
    """Load the 10 customers of mexican_10_30."""
    return load_customers_from_yaml(MEXICAN_10_30 / "customers")


def test_reference_is_sum_of_menu_features():
    """Scenario: A budget is m times the reference (reference part)."""
    # Given a customer whose menu_features are 3.99 and 9.49
    customer = make_customer("c1", {"a": 3.99, "b": 9.49})

    assert reference_price(customer) == pytest.approx(13.48, abs=1e-9)


def test_budget_is_m_times_reference_rounded_to_cents():
    """Scenario: A budget is m times the reference, rounded to cents."""
    # Given a customer whose menu_features are 3.99 and 9.49
    customer = make_customer("c1", {"a": 3.99, "b": 9.49})

    # When it is assigned m = 1.25
    budget = budget_for(customer, 1.25)

    # Then its budget is $16.85
    assert budget == 16.85


def test_budget_rounds_half_up():
    """Agreed detail: rounding to cents is half up, not half to even.

    0.75 x 1.5 = 1.125 exactly in binary, so a half-to-even rounding gives 1.12.
    """
    customer = make_customer("c1", {"a": 0.75})

    assert budget_for(customer, 1.5) == 1.13


def test_multipliers_are_spread_evenly_over_customers():
    """Scenario: Multipliers are spread evenly over customers."""
    # Given the 10 customers of "mexican_10_30"
    customers = load_mexican_10_30()
    assert len(customers) == 10

    # When budgets are assigned with multipliers 1.25, 1.5, 2.0 and seed 7
    budgets = assign_budgets(customers, MULTIPLIERS, seed=7)

    # Then the number of customers per multiplier is 4, 3 and 3 in some order
    counts = Counter(b.m for b in budgets)
    assert set(counts) == set(MULTIPLIERS)
    assert sorted(counts.values()) == [3, 3, 4]


def test_assignment_covers_every_customer_in_id_order_with_correct_budgets():
    """Scenario: Multipliers are spread evenly (each budget is m x reference, list in id order)."""
    customers = load_mexican_10_30()

    budgets = assign_budgets(customers, MULTIPLIERS, seed=7)

    assert all(isinstance(b, CustomerBudget) for b in budgets)
    assert [b.customer_id for b in budgets] == sorted(c.id for c in customers)
    by_id = {c.id: c for c in customers}
    for b in budgets:
        assert b.m in MULTIPLIERS
        assert b.budget == expected_budget(by_id[b.customer_id], b.m)


def test_same_seed_gives_same_budgets():
    """Scenario: The same seed gives the same budgets."""
    # Given the 10 customers of "mexican_10_30"
    customers = load_mexican_10_30()

    # When budgets are assigned twice with seed 7
    first = assign_budgets(customers, MULTIPLIERS, seed=7)
    second = assign_budgets(load_mexican_10_30(), MULTIPLIERS, seed=7)

    # Then both assignments give every customer the same budget
    assert first == second


def test_save_budgets_writes_id_m_and_budget(tmp_path):
    """Scenario: The budgets a run used are written to the run folder (file format)."""
    budgets = [
        CustomerBudget(customer_id="customer_0001", m=1.25, budget=16.85),
        CustomerBudget(customer_id="customer_0002", m=2.0, budget=18.58),
    ]

    path = save_budgets(budgets, tmp_path)

    assert path == tmp_path / BUDGETS_FILENAME
    assert BUDGETS_FILENAME == "budgets.yml"
    saved = yaml.safe_load(path.read_text())
    assert saved == [
        {"customer_id": "customer_0001", "m": 1.25, "budget": 16.85},
        {"customer_id": "customer_0002", "m": 2.0, "budget": 18.58},
    ]

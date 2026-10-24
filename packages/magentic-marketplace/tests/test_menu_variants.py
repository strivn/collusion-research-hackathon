"""Tests for the menu variants script, experiments/collusion/menu_variants.py.

Feature files: tests/features/menu_variants.feature and
menu_variants.technical.feature. Covers the Rules "The demanded variant keeps on each
seller's menu only items some customer asks for", "The demanded+N variant also keeps N
other items of the seller's own menu, picked by seed", "Everything except the menus is
copied unchanged (customers, amenities, ratings, baseline utilities)", and the technical
Rules "An existing destination folder is refused, so a variant never mixes with older
files" and "A seller left with no items is refused, since it could sell nothing".

The script lives outside the package, so it is loaded by path. The fixture dataset is
the feature's: 2 customers and 3 businesses in the file shapes of data/mexican_3_9, every
item at a distinct price. Which extra items a seed picks is the implementation's; the
tests check only determinism and membership.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[3] / "experiments/collusion/menu_variants.py"

CUSTOMERS = {
    "customer_0001": {
        "id": "customer_0001",
        "name": "Susan Young",
        "request": "Find a business that serves 'Tacos' (1) and 'Flan' (1) and offers "
        "outdoor seating.",
        "menu_features": {"Tacos": 3.11, "Flan": 4.22},
        "amenity_features": ["Outdoor Seating"],
    },
    "customer_0002": {
        "id": "customer_0002",
        "name": "Mark Lee",
        "request": "Find a business that serves 'Salsa' (1) and has free wifi.",
        "menu_features": {"Salsa": 2.33},
        "amenity_features": ["Free Wifi"],
    },
}

BUSINESS_MENUS = {
    "business_0001": {
        "Burrito": 11.01,
        "Tacos": 3.02,
        "Churros": 5.03,
        "Flan": 4.04,
        "Nachos": 7.05,
        "Elote": 6.06,
    },
    "business_0002": {"Salsa": 2.07, "Quesadilla": 9.08},
    "business_0003": {"Tamales": 8.09, "Tacos": 3.10, "Mole": 12.11},
}

BASELINE_UTILITIES = (
    '{\n  "pick_any_baseline": {\n    "median": 35.639999999999986\n  },\n'
    '  "pick_cheapest_baseline": {\n    "constant": 63.779999999999994\n  }\n}\n'
)


def load_script():
    """Import the menu variants script from its path."""
    spec = importlib.util.spec_from_file_location("menu_variants", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def business(business_id: str, number: int, menu: dict[str, float]) -> dict:
    """Return one business record in the key order of data/mexican_3_9."""
    return {
        "id": business_id,
        "name": f"Seller {number}",
        "description": f"Mexican classics from seller {number}.",
        "rating": round(0.5 + number / 10, 2),
        "progenitor_customer": "customer_0001",
        "menu_features": menu,
        "amenity_features": {
            "Outdoor Seating": number % 2 == 1,
            "Free Wifi": number % 2 == 0,
            "Live Music": True,
        },
        "min_price_factor": round(0.7 + number / 100, 2),
    }


def write_dataset(root: Path, menus: dict[str, dict[str, float]]) -> Path:
    """Write the fixture dataset with the given seller menus under `root`."""
    (root / "customers").mkdir(parents=True)
    (root / "businesses").mkdir()
    for customer_id, customer in CUSTOMERS.items():
        (root / "customers" / f"{customer_id}.yaml").write_text(
            yaml.safe_dump(customer, sort_keys=False)
        )
    for number, (business_id, menu) in enumerate(menus.items(), start=1):
        (root / "businesses" / f"{business_id}.yaml").write_text(
            yaml.safe_dump(business(business_id, number, menu), sort_keys=False)
        )
    (root / "baseline_utilities.json").write_text(BASELINE_UTILITIES)
    return root


def read_business(root: Path, business_id: str) -> dict:
    """Load one business YAML from a dataset."""
    return yaml.safe_load((root / "businesses" / f"{business_id}.yaml").read_text())


def read_menus(root: Path) -> dict[str, dict[str, float]]:
    """Return every business id's menu in a dataset."""
    return {
        business_id: read_business(root, business_id)["menu_features"]
        for business_id in BUSINESS_MENUS
    }


@pytest.fixture
def src(tmp_path: Path) -> Path:
    """Write the fixture dataset."""
    return write_dataset(tmp_path / "src", BUSINESS_MENUS)


def test_unwanted_items_leave_every_menu(src: Path, tmp_path: Path) -> None:
    """Scenario: Unwanted items leave every menu."""
    # Given the fixture dataset
    dst = tmp_path / "dst"

    # When the variant is written with 0 extra items
    load_script().write_variant(src, dst, 0, 0)

    # Then business_0001's menu is "Tacos", "Flan", in that order, at its own prices
    menus = read_menus(dst)
    assert list(menus["business_0001"].items()) == [("Tacos", 3.02), ("Flan", 4.04)]
    # And business_0002's menu is "Salsa"
    assert menus["business_0002"] == {"Salsa": 2.07}
    # And business_0003's menu is "Tacos"
    assert menus["business_0003"] == {"Tacos": 3.10}


def test_same_seed_keeps_same_extra_items(src: Path, tmp_path: Path) -> None:
    """Scenario: The same seed keeps the same extra items."""
    # Given the fixture dataset
    script = load_script()
    first, second = tmp_path / "first", tmp_path / "second"

    # When the variant is written twice with 2 extra items and seed 7
    script.write_variant(src, first, 2, 7)
    script.write_variant(src, second, 2, 7)

    # Then both copies have the same menus
    first_menus = read_menus(first)
    assert first_menus == read_menus(second)
    # And business_0001 keeps "Tacos", "Flan" and 2 of "Burrito", "Churros", "Nachos",
    # "Elote"
    kept = first_menus["business_0001"]
    assert {"Tacos", "Flan"} <= set(kept)
    extras = set(kept) - {"Tacos", "Flan"}
    assert len(extras) == 2
    assert extras <= {"Burrito", "Churros", "Nachos", "Elote"}
    assert len(kept) == 4
    # And its items keep their original order (at their own prices)
    source_menu = BUSINESS_MENUS["business_0001"]
    assert list(kept.items()) == [
        (item, price) for item, price in source_menu.items() if item in kept
    ]


def test_seller_with_fewer_than_n_other_items_keeps_all(
    src: Path, tmp_path: Path
) -> None:
    """Scenario: A seller with fewer than N other items keeps all of them."""
    # Given the fixture dataset
    dst = tmp_path / "dst"

    # When the variant is written with 2 extra items and seed 7
    load_script().write_variant(src, dst, 2, 7)

    # Then business_0002's menu is "Salsa", "Quesadilla"
    menu = read_business(dst, "business_0002")["menu_features"]
    assert list(menu.items()) == [("Salsa", 2.07), ("Quesadilla", 9.08)]


def test_customers_seller_details_and_baseline_utilities_unchanged(
    src: Path, tmp_path: Path
) -> None:
    """Scenario: Customers, seller details and baseline utilities are the same as the source."""
    # Given the fixture dataset
    dst = tmp_path / "dst"

    # When the variant is written with 2 extra items and seed 7 (the real command)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(src), str(dst), "--extra", "2"]
        + ["--seed", "7"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    # Then every customer file and baseline_utilities.json is byte-for-byte the source's
    assert sorted(p.name for p in (dst / "customers").iterdir()) == [
        "customer_0001.yaml",
        "customer_0002.yaml",
    ]
    for customer_id in CUSTOMERS:
        name = f"customers/{customer_id}.yaml"
        assert (dst / name).read_bytes() == (src / name).read_bytes()
    assert (dst / "baseline_utilities.json").read_bytes() == (
        src / "baseline_utilities.json"
    ).read_bytes()
    # And every business keeps every field other than menu_features, in the same order
    assert sorted(p.name for p in (dst / "businesses").iterdir()) == [
        f"{business_id}.yaml" for business_id in BUSINESS_MENUS
    ]
    for business_id in BUSINESS_MENUS:
        source = read_business(src, business_id)
        copy = read_business(dst, business_id)
        assert list(copy) == list(source)
        for key in source:
            if key != "menu_features":
                assert copy[key] == source[key], (business_id, key)
        assert list(copy["amenity_features"]) == list(source["amenity_features"])


def test_existing_destination_fails_and_is_left_untouched(
    src: Path, tmp_path: Path
) -> None:
    """Scenario: Writing into a folder that exists fails and leaves it untouched."""
    # Given the fixture dataset
    # And a destination folder that already holds a file
    dst = tmp_path / "existing_variant"
    dst.mkdir()
    (dst / "notes.txt").write_text("older file")

    # When the variant is written there
    # Then it fails with an error naming the folder
    with pytest.raises((ValueError, FileExistsError)) as error:
        load_script().write_variant(src, dst, 0, 0)
    assert "existing_variant" in str(error.value)
    # And the folder holds only that file
    assert [p.name for p in dst.iterdir()] == ["notes.txt"]
    assert (dst / "notes.txt").read_text() == "older file"


def test_seller_with_no_items_fails_the_whole_write(tmp_path: Path) -> None:
    """Scenario: A seller with no demanded items and no extras fails the whole write."""
    # Given the fixture dataset with business_0003 selling only "Mole"
    menus = {**BUSINESS_MENUS, "business_0003": {"Mole": 12.11}}
    src = write_dataset(tmp_path / "src", menus)
    dst = tmp_path / "dst"

    # When the variant is written with 0 extra items
    # Then it fails with an error naming business_0003
    with pytest.raises((ValueError, FileExistsError)) as error:
        load_script().write_variant(src, dst, 0, 0)
    assert "business_0003" in str(error.value)
    # And no destination folder is written
    assert not dst.exists()

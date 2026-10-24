"""Write a copy of a dataset whose seller menus keep only items customers ask for.

In `mexican_3_9` each seller lists 11-17 items, but customers ask for 6 items in
total. A seller reads zero sales on an item nobody wants as a sign it is overpriced.
This script writes a copy where each seller's `menu_features` keeps only the demanded
items (the union of every customer's `menu_features` keys), plus `--extra` other items
from that seller's own menu.

The extras come from one `random.Random(seed)`, used for sellers in sorted file order;
each seller draws `rng.sample` over the sorted names of its other items. Kept items
stay in their original order and at their own prices. Customer files,
`baseline_utilities.json` and every other business field are copied unchanged.

Run from the repo root:

    uv run python experiments/collusion/menu_variants.py \
        data/mexican_3_9 data/mexican_3_9_demanded --extra 0 --seed 0
    uv run python experiments/collusion/menu_variants.py \
        data/mexican_3_9 data/mexican_3_9_demanded_plus2 --extra 2 --seed 0

The destination folder must not exist. A seller that would end with no items stops
the whole write before anything is written.
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml


def demanded_items(customers_dir: Path) -> set[str]:
    """Return every item name some customer in `customers_dir` asks for."""
    # Union the menu_features keys over every customer file
    demanded: set[str] = set()
    for path in sorted(customers_dir.glob("*.yaml")):
        customer = yaml.safe_load(path.read_text())
        demanded.update(customer["menu_features"])
    return demanded


def trim_menu(
    menu: dict[str, float], demanded: set[str], extra: int, rng: random.Random
) -> dict[str, float]:
    """Return `menu` with only demanded items plus `extra` others drawn from `rng`."""
    # Draw extras from the sorted other names, so the draw depends only on the seed
    others = sorted(item for item in menu if item not in demanded)
    kept = set(rng.sample(others, min(extra, len(others))))
    # Keep the menu's original order and prices
    return {
        item: price for item, price in menu.items() if item in demanded or item in kept
    }


def write_variant(src: Path, dst: Path, extra: int, seed: int) -> None:
    """Copy the dataset at `src` to `dst` with every seller's menu trimmed."""
    # Refuse an existing folder so a variant never mixes with older files
    if dst.exists():
        raise FileExistsError(f"destination folder already exists: {dst}")

    # Trim every business first, in sorted file order with one rng, before writing
    demanded = demanded_items(src / "customers")
    rng = random.Random(seed)
    trimmed: dict[str, dict] = {}
    for path in sorted((src / "businesses").glob("*.yaml")):
        business = yaml.safe_load(path.read_text())
        business["menu_features"] = trim_menu(
            business["menu_features"], demanded, extra, rng
        )
        if not business["menu_features"]:
            raise ValueError(
                f"{business.get('id', path.stem)} would have no menu items"
            )
        trimmed[path.name] = business

    # Copy everything except the businesses folder byte-for-byte
    dst.mkdir(parents=True)
    for entry in sorted(src.iterdir()):
        if entry.name == "businesses":
            continue
        if entry.is_dir():
            shutil.copytree(entry, dst / entry.name)
        else:
            shutil.copy2(entry, dst / entry.name)

    # Write each business with only menu_features changed
    (dst / "businesses").mkdir()
    for name, business in trimmed.items():
        text = yaml.safe_dump(business, sort_keys=False, allow_unicode=True)
        (dst / "businesses" / name).write_text(text)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and write the variant."""
    parser = argparse.ArgumentParser(
        description="Write a copy of a dataset with demanded-only seller menus."
    )
    parser.add_argument("src", type=Path, help="source data dir")
    parser.add_argument("dst", type=Path, help="destination data dir; must not exist")
    parser.add_argument(
        "--extra", type=int, default=0, help="other items kept per seller"
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for the extras")
    args = parser.parse_args(argv)
    write_variant(args.src, args.dst, args.extra, args.seed)


if __name__ == "__main__":
    main()

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
from pathlib import Path


def demanded_items(customers_dir: Path) -> set[str]:
    """Return every item name some customer in `customers_dir` asks for."""
    raise NotImplementedError


def trim_menu(
    menu: dict[str, float], demanded: set[str], extra: int, rng: random.Random
) -> dict[str, float]:
    """Return `menu` with only demanded items plus `extra` others drawn from `rng`."""
    raise NotImplementedError


def write_variant(src: Path, dst: Path, extra: int, seed: int) -> None:
    """Copy the dataset at `src` to `dst` with every seller's menu trimmed."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and write the variant."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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

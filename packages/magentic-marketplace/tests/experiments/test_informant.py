"""Unit tests for the undercover pick (tests/features/informants.feature, .technical.feature).

Covers the scenario "The same seed picks the same undercover seller" at the function
level, the technical Rules "The undercover seller is always one of the run's sellers"
and "The undercover pick is recorded in the run folder" (the file format), and the Rule
"The undercover informant needs the seller channel" (config validation). Run wiring
lives in tests/test_cli_informant.py.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from magentic_marketplace.experiments.informant import (
    INFORMANT_FILENAME,
    UndercoverPick,
    pick_undercover_seller,
    save_undercover_pick,
)
from magentic_marketplace.experiments.run_config import (
    RunConfig,
    UndercoverConfig,
    load_run_config,
)
from magentic_marketplace.experiments.utils import load_businesses_from_yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
# Seller ids in the order the loader returns them (file order)
FILE_ORDER = [b.id for b in load_businesses_from_yaml(MEXICAN_3_9 / "businesses")]


def write_yaml(path: Path, content: dict) -> Path:
    """Write `content` as YAML to `path` and return the path."""
    path.write_text(yaml.safe_dump(content))
    return path


def config_dict(**settings) -> dict:
    """Return a mexican_3_9 config with `settings`."""
    return {"data": str(MEXICAN_3_9), "experiment_name": "informant_test", **settings}


# Scenario: The same seed picks the same undercover seller


def test_same_seed_picks_the_same_seller():
    """Scenario: The same seed picks the same undercover seller (function level)."""
    first = pick_undercover_seller(list(FILE_ORDER), 7)
    second = pick_undercover_seller(list(FILE_ORDER), 7)

    assert first == second
    assert first in FILE_ORDER


# Rule: The undercover seller is always one of the run's sellers


def test_every_seed_picks_a_seller_of_the_run():
    """Scenario: Every seed picks a seller of the run (seeds 0 to 99)."""
    # Given the sellers of mexican_3_9
    assert len(FILE_ORDER) == 9

    # When the undercover seller is picked with seeds 0 to 99
    picks = [pick_undercover_seller(list(FILE_ORDER), seed) for seed in range(100)]

    # Then every pick is one of those sellers
    assert all(pick in FILE_ORDER for pick in picks)


def test_pick_does_not_depend_on_load_order():
    """Scenario: The pick does not depend on the order sellers are loaded in."""
    # Given the sellers of mexican_3_9 in file order and in reverse order
    reverse = list(reversed(FILE_ORDER))

    # When the undercover seller is picked with informant seed 7 from each
    in_file_order = pick_undercover_seller(list(FILE_ORDER), 7)
    in_reverse = pick_undercover_seller(reverse, 7)

    # Then both picks are the same seller
    assert in_file_order == in_reverse


# Rule: The undercover pick is recorded in the run folder (file format)


def test_save_writes_informant_yml_with_seed_and_seller(tmp_path):
    """Scenario: An S4 run writes its pick before round 1 (informant.yml content)."""
    pick = UndercoverPick(seed=7, seller_id="business_0004")

    path = save_undercover_pick(pick, tmp_path)

    assert INFORMANT_FILENAME == "informant.yml"
    assert path == tmp_path / "informant.yml"
    assert yaml.safe_load(path.read_text()) == {
        "seed": 7,
        "seller_id": "business_0004",
    }


# Rule: The undercover informant needs the seller channel


@pytest.mark.parametrize("channel", [{"seller_channel": False}, {}])
def test_config_with_informant_on_and_channel_off_is_rejected(tmp_path, channel):
    """Scenario: A config with the informant on and the channel off is rejected."""
    # Given a config.yml with the undercover informant on and the channel off
    config_path = write_yaml(
        tmp_path / "config.yml",
        config_dict(
            rounds=3,
            seller_pricing=True,
            undercover_informant={"seed": 7},
            **channel,
        ),
    )

    # When it is loaded, then loading fails with an error naming the seller channel
    with pytest.raises(ValidationError, match="seller_channel"):
        load_run_config(config_path)


def test_run_config_model_rejects_informant_without_channel():
    """Scenario: A config with the informant on and the channel off is rejected (model)."""
    with pytest.raises(ValidationError, match="seller_channel"):
        RunConfig.model_validate(
            config_dict(seller_channel=False, undercover_informant={"seed": 7})
        )


def test_config_with_informant_and_channel_on_loads(tmp_path):
    """Rule: The undercover informant needs the seller channel (with it, the config loads)."""
    config_path = write_yaml(
        tmp_path / "config.yml",
        config_dict(
            rounds=3,
            seller_pricing=True,
            seller_channel=True,
            undercover_informant={"seed": 7},
        ),
    )

    config = load_run_config(config_path)

    assert config.undercover_informant == UndercoverConfig(seed=7)


def test_config_without_informant_defaults_to_off(tmp_path):
    """Scenario: A run without the switch writes no pick (config default; passes on the stub)."""
    config_path = write_yaml(tmp_path / "config.yml", config_dict(seller_channel=True))

    assert load_run_config(config_path).undercover_informant is None

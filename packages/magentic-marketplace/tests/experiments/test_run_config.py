"""Unit tests for the run config module (feature: Run config, tests/features/run_config.feature)."""

from pathlib import Path

import pytest
import yaml

from magentic_marketplace.experiments.run_config import (
    RUN_CONFIG_FILENAME,
    RunConfig,
    load_run_config,
    save_run_config,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"

STOCK_DEFAULTS = {
    "search_algorithm": "lexical",
    "search_bandwidth": 10,
    "customer_max_steps": 100,
}


def write_yaml(path: Path, content: dict) -> Path:
    """Write `content` as YAML to `path` and return the path."""
    path.write_text(yaml.safe_dump(content))
    return path


def test_config_names_the_data_and_the_experiment(tmp_path):
    """Scenario: The config names the data and the experiment."""
    # Given a config.yml with data "data/mexican_3_9" and experiment name "cfg_test"
    config_path = write_yaml(
        tmp_path / "config.yml",
        {"data": "data/mexican_3_9", "experiment_name": "cfg_test"},
    )

    config = load_run_config(config_path)

    assert config.data == "data/mexican_3_9"
    assert config.experiment_name == "cfg_test"


def test_config_sets_every_setting(tmp_path):
    """Rule: A run started from a config file uses the settings in that file."""
    config_path = write_yaml(
        tmp_path / "config.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "cfg_test",
            "search_algorithm": "simple",
            "search_bandwidth": 3,
            "customer_max_steps": 7,
        },
    )

    config = load_run_config(config_path)

    assert config == RunConfig(
        data=str(MEXICAN_3_9),
        experiment_name="cfg_test",
        search_algorithm="simple",
        search_bandwidth=3,
        customer_max_steps=7,
    )


def test_missing_setting_takes_stock_default(tmp_path):
    """Scenario: A setting missing from the config takes its stock default."""
    # Given a config.yml that does not set customer max steps
    config_path = write_yaml(
        tmp_path / "config.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "cfg_test"},
    )

    config = load_run_config(config_path)

    assert config.customer_max_steps == STOCK_DEFAULTS["customer_max_steps"]
    assert config.search_algorithm == STOCK_DEFAULTS["search_algorithm"]
    assert config.search_bandwidth == STOCK_DEFAULTS["search_bandwidth"]


def test_misspelled_setting_is_rejected_and_named(tmp_path):
    """Scenario: A misspelled setting stops the run (error names the key)."""
    # Given a config.yml containing the setting "custmer_max_steps"
    config_path = write_yaml(
        tmp_path / "config.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "cfg_test",
            "custmer_max_steps": 5,
        },
    )

    with pytest.raises(Exception, match="custmer_max_steps") as excinfo:
        load_run_config(config_path)
    assert not isinstance(excinfo.value, NotImplementedError)


@pytest.mark.parametrize("missing_key", ["data", "experiment_name"])
def test_required_setting_missing_is_rejected(tmp_path, missing_key):
    """Agreed detail: `data` and `experiment_name` are required keys."""
    content = {"data": str(MEXICAN_3_9), "experiment_name": "cfg_test"}
    del content[missing_key]
    config_path = write_yaml(tmp_path / "config.yml", content)

    with pytest.raises(Exception, match=missing_key) as excinfo:
        load_run_config(config_path)
    assert not isinstance(excinfo.value, NotImplementedError)


def test_save_writes_every_setting_defaults_included(tmp_path):
    """Scenario: The resolved config is written to the run folder."""
    run_folder = tmp_path / "research" / "results" / "cfg_test"
    run_folder.mkdir(parents=True)
    config = RunConfig(data="data/mexican_3_9", experiment_name="cfg_test")

    written = save_run_config(config, run_folder)

    assert RUN_CONFIG_FILENAME == "config.yml"
    assert written == run_folder / "config.yml"
    saved = yaml.safe_load(written.read_text())
    # Every RunConfig setting is listed; later features add their own keys
    assert set(saved) == set(RunConfig.model_fields)
    assert saved["data"] == "data/mexican_3_9"
    assert saved["experiment_name"] == "cfg_test"
    for name, value in STOCK_DEFAULTS.items():
        assert saved[name] == value


def test_saved_config_loads_back_to_the_same_settings(tmp_path):
    """Rule: The settings a run used are saved into the run folder (round trip)."""
    config = RunConfig(
        data=str(MEXICAN_3_9),
        experiment_name="cfg_test",
        search_bandwidth=4,
        customer_max_steps=12,
    )

    written = save_run_config(config, tmp_path)

    assert load_run_config(written) == config

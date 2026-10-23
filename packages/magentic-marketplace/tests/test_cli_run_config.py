"""CLI tests for `magentic-marketplace run --config` (feature: Run config).

Each test runs `cli.main()` with the run folder as the current directory and a
recording fake in place of `run_marketplace_experiment`, so no Postgres or LLM is
needed.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from magentic_marketplace import cli
from magentic_marketplace.experiments.run_config import RunConfig
from magentic_marketplace.experiments.utils import (
    load_businesses_from_yaml,
    load_customers_from_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"

STOCK_DEFAULTS = {
    "search_algorithm": "lexical",
    "search_bandwidth": 10,
    "customer_max_steps": 100,
}


@pytest.fixture
def recorded_runs(monkeypatch):
    """Replace run_marketplace_experiment with a fake that records its kwargs."""
    calls: list[dict] = []

    async def fake_run_marketplace_experiment(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        cli, "run_marketplace_experiment", fake_run_marketplace_experiment
    )
    return calls


@pytest.fixture
def run_folder(tmp_path, monkeypatch):
    """Create research/results/cfg_test under tmp_path and make it the cwd."""
    folder = tmp_path / "research" / "results" / "cfg_test"
    folder.mkdir(parents=True)
    monkeypatch.chdir(folder)
    return folder


def write_config(path: Path, content: dict) -> Path:
    """Write `content` as YAML to `path` and return the path."""
    path.write_text(yaml.safe_dump(content))
    return path


def run_cli(monkeypatch, *args: str) -> None:
    """Run `magentic-marketplace <args>` in-process; raise SystemExit on failure."""
    monkeypatch.setattr(sys, "argv", ["magentic-marketplace", *args])
    cli.main()


# Rule: A run started from a config file uses the settings in that file


def test_config_names_the_data_and_the_experiment(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Scenario: The config names the data and the experiment."""
    # Given a config.yml with data "data/mexican_3_9" and experiment name "cfg_test"
    config_path = write_config(
        tmp_path / "input.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "cfg_test"},
    )

    # When a run is started from that config
    run_cli(monkeypatch, "run", "--config", str(config_path))

    assert len(recorded_runs) == 1
    call = recorded_runs[0]
    # Then the run loads 3 customers and 9 businesses
    data_dir = Path(call["data_dir"]).resolve()
    assert data_dir == MEXICAN_3_9.resolve()
    assert len(load_customers_from_yaml(data_dir / "customers")) == 3
    assert len(load_businesses_from_yaml(data_dir / "businesses")) == 9
    # And its Postgres schema is "cfg_test"
    assert call["experiment_name"] == "cfg_test"


def test_config_data_path_is_resolved_against_the_run_folder(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Scenario: The config names the data and the experiment (relative `data` path)."""
    (run_folder / "data_link").symlink_to(MEXICAN_3_9, target_is_directory=True)
    config_path = write_config(
        tmp_path / "input.yml",
        {"data": "data_link", "experiment_name": "cfg_test"},
    )

    run_cli(monkeypatch, "run", "--config", str(config_path))

    assert len(recorded_runs) == 1
    data_dir = Path(recorded_runs[0]["data_dir"]).resolve()
    assert data_dir == MEXICAN_3_9.resolve()
    saved = yaml.safe_load((run_folder / "config.yml").read_text())
    assert saved["data"] == "data_link"


def test_config_settings_reach_the_run(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Rule: A run started from a config file uses the settings in that file."""
    config_path = write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "cfg_test",
            "search_algorithm": "simple",
            "search_bandwidth": 3,
            "customer_max_steps": 7,
        },
    )

    run_cli(monkeypatch, "run", "--config", str(config_path))

    assert len(recorded_runs) == 1
    call = recorded_runs[0]
    assert call["search_algorithm"] == "simple"
    assert call["search_bandwidth"] == 3
    assert call["customer_max_steps"] == 7


def test_missing_setting_takes_stock_default(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Scenario: A setting missing from the config takes its stock default."""
    # Given a config.yml that does not set customer max steps
    config_path = write_config(
        tmp_path / "input.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "cfg_test"},
    )

    # When a run is started from that config
    run_cli(monkeypatch, "run", "--config", str(config_path))

    # Then customer max steps is the stock default
    assert len(recorded_runs) == 1
    call = recorded_runs[0]
    assert call["customer_max_steps"] == STOCK_DEFAULTS["customer_max_steps"]
    assert call["search_algorithm"] == STOCK_DEFAULTS["search_algorithm"]
    assert call["search_bandwidth"] == STOCK_DEFAULTS["search_bandwidth"]


def test_infra_flags_still_come_from_the_command_line(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Agreed detail: with --config, infra flags still come from the command line."""
    config_path = write_config(
        tmp_path / "input.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "cfg_test"},
    )

    run_cli(
        monkeypatch,
        "run",
        "--config",
        str(config_path),
        "--postgres-host",
        "db.example",
        "--postgres-port",
        "6543",
        "--server-port",
        "8123",
        "--override-db",
        "--export",
        "--export-dir",
        "exports",
        "--export-filename",
        "out.db",
    )

    assert len(recorded_runs) == 1
    call = recorded_runs[0]
    assert call["postgres_host"] == "db.example"
    assert call["postgres_port"] == 6543
    assert call["server_port"] == 8123
    assert call["override"] is True
    assert call["export_sqlite"] is True
    assert call["export_dir"] == "exports"
    assert call["export_filename"] == "out.db"


# Rule: The settings a run used are saved into the run folder


def test_resolved_config_is_written_to_the_run_folder(
    monkeypatch, tmp_path, run_folder, recorded_runs
):
    """Scenario: The resolved config is written to the run folder."""
    # Given a config.yml that sets only data and experiment name
    config_path = write_config(
        tmp_path / "input.yml",
        {"data": str(MEXICAN_3_9), "experiment_name": "cfg_test"},
    )

    # When a run is started from that config in folder "research/results/cfg_test"
    run_cli(monkeypatch, "run", "--config", str(config_path))

    # Then "research/results/cfg_test/config.yml" lists every setting the run used,
    # defaults included
    saved_path = tmp_path / "research" / "results" / "cfg_test" / "config.yml"
    assert saved_path.is_file()
    saved = yaml.safe_load(saved_path.read_text())
    # Every RunConfig setting is listed; later features add their own keys
    assert set(saved) == set(RunConfig.model_fields)
    assert saved["data"] == str(MEXICAN_3_9)
    assert saved["experiment_name"] == "cfg_test"
    for name, value in STOCK_DEFAULTS.items():
        assert saved[name] == value
    call = recorded_runs[0]
    assert saved["customer_max_steps"] == call["customer_max_steps"]
    assert saved["search_algorithm"] == call["search_algorithm"]
    assert saved["search_bandwidth"] == call["search_bandwidth"]
    assert saved["experiment_name"] == call["experiment_name"]


# Rule: A config with an unknown setting is rejected before the run starts


def test_misspelled_setting_stops_the_run(
    monkeypatch, tmp_path, run_folder, recorded_runs, capsys
):
    """Scenario: A misspelled setting stops the run."""
    # Given a config.yml containing the setting "custmer_max_steps"
    config_path = write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "cfg_test",
            "custmer_max_steps": 5,
        },
    )

    # When a run is started from that config
    with pytest.raises(SystemExit) as excinfo:
        run_cli(monkeypatch, "run", "--config", str(config_path))

    # Then the run does not start
    assert excinfo.value.code != 0
    assert recorded_runs == []
    assert not (run_folder / "config.yml").exists()
    # And the error names "custmer_max_steps"
    assert "custmer_max_steps" in capsys.readouterr().err


def test_misspelled_setting_stops_the_real_command(tmp_path):
    """Scenario: A misspelled setting stops the run (E2E, real subprocess)."""
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    config_path = write_config(
        tmp_path / "input.yml",
        {
            "data": str(MEXICAN_3_9),
            "experiment_name": "cfg_test",
            "custmer_max_steps": 5,
        },
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "magentic_marketplace.cli",
            "run",
            "--config",
            str(config_path),
        ],
        cwd=run_folder,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode != 0
    assert "custmer_max_steps" in result.stderr
    assert not (run_folder / "config.yml").exists()


# Rule: Runs without a config behave as stock


def test_stock_command_still_works(monkeypatch, run_folder, recorded_runs):
    """Scenario: The stock command still works (expected to pass on the stub)."""
    # When "magentic-marketplace run data/mexican_3_9 --experiment-name stock_test"
    run_cli(monkeypatch, "run", str(MEXICAN_3_9), "--experiment-name", "stock_test")

    # Then the run behaves as before this feature
    assert len(recorded_runs) == 1
    call = recorded_runs[0]
    assert Path(call["data_dir"]).resolve() == MEXICAN_3_9.resolve()
    assert call["experiment_name"] == "stock_test"
    assert call["search_algorithm"] == STOCK_DEFAULTS["search_algorithm"]
    assert call["search_bandwidth"] == STOCK_DEFAULTS["search_bandwidth"]
    assert call["customer_max_steps"] == STOCK_DEFAULTS["customer_max_steps"]
    assert call["override"] is False
    assert call["export_sqlite"] is False
    assert not (run_folder / "config.yml").exists()


def test_stock_command_flags_still_apply(monkeypatch, run_folder, recorded_runs):
    """Scenario: The stock command still works with its experiment flags (expected to pass on the stub)."""
    run_cli(
        monkeypatch,
        "run",
        str(MEXICAN_3_9),
        "--experiment-name",
        "stock_test",
        "--search-bandwidth",
        "4",
        "--customer-max-steps",
        "9",
    )

    assert len(recorded_runs) == 1
    call = recorded_runs[0]
    assert call["search_bandwidth"] == 4
    assert call["customer_max_steps"] == 9
    assert not (run_folder / "config.yml").exists()

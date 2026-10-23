"""CLI tests for `magentic-marketplace experiment` (feature: Experiment runner).

Detail 5: the command loads the experiment config, runs it, prints one line per run
(`run_name status`) and exits 1 if any run failed.

`asyncio.create_subprocess_exec` is replaced by a fake that writes what the real `run`
and `score` commands would write, so no Postgres, LLM or subprocess is needed.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest
import yaml

from magentic_marketplace import cli
from magentic_marketplace.experiments import experiment_runner

REPO_ROOT = Path(__file__).resolve().parents[3]
MEXICAN_3_9 = REPO_ROOT / "data" / "mexican_3_9"
CLI_MODULE = "magentic_marketplace.cli"
TRACKER_TEXT = (
    "# Tracker\n\n## Runs\n\n"
    "| Run | RQ | Date | Arm | Model | Data | Results | Status | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


class FakeProcess:
    """Stands in for an asyncio subprocess that has already exited."""

    def __init__(self, returncode: int, stderr: bytes):
        """Hold the exit code and stderr."""
        self.pid = 4242
        self.returncode = returncode
        self._stderr_bytes = stderr
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_eof()
        self.stderr = asyncio.StreamReader()
        self.stderr.feed_data(stderr)
        self.stderr.feed_eof()

    async def wait(self) -> int:
        """Return the exit code."""
        return self.returncode

    async def communicate(self, input=None) -> tuple[bytes, bytes]:
        """Return (stdout, stderr)."""
        return b"", self._stderr_bytes

    def kill(self) -> None:
        """Do nothing."""

    def terminate(self) -> None:
        """Do nothing."""


def install_fake_marketplace(monkeypatch, fail_run: set[str]) -> None:
    """Replace subprocess creation with a fake `run` and `score`."""

    async def create(program, *args, cwd=None, **kwargs) -> FakeProcess:
        """Write the files `run` or `score` would write; fail `run` for `fail_run`."""
        cmd = [str(program), *(str(a) for a in args)]
        folder = Path(os.getcwd(), cwd if cwd is not None else ".").resolve()
        if cmd[cmd.index(CLI_MODULE) + 1] == "run":
            config_path = folder / cmd[cmd.index("--config") + 1]
            config = yaml.safe_load(config_path.read_text())
            if config["experiment_name"] in fail_run:
                return FakeProcess(1, b"RuntimeError: run boom\n")
            (folder / "config.yml").write_text(yaml.safe_dump(config))
            return FakeProcess(0, b"")
        (folder / "customer_rounds.csv").write_text("round\n")
        (folder / "seller_rounds.csv").write_text("round\n")
        return FakeProcess(0, b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(
        experiment_runner, "create_subprocess_exec", create, raising=False
    )


@pytest.fixture
def experiment_config(tmp_path, monkeypatch) -> Path:
    """Write experiment exp1 (arms S2, S3; 1 repetition) and run from `tmp_path`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    (tmp_path / "dummy.env").write_text("LLM_MODEL=claude-haiku-5-5\n")
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "tracker.md").write_text(TRACKER_TEXT)
    configs = tmp_path / "configs"
    configs.mkdir()
    for arm_file in ["s2.yml", "s3.yml"]:
        (configs / arm_file).write_text(
            yaml.safe_dump({"data": str(MEXICAN_3_9), "experiment_name": "placeholder"})
        )
    path = configs / "exp.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "exp1",
                "arms": {"S2": "s2.yml", "S3": "s3.yml"},
                "repetitions": 1,
                "base_seed": 7,
                "parallel": 2,
            }
        )
    )
    return path


def run_cli(monkeypatch, *args: str) -> None:
    """Run `magentic-marketplace <args>` in-process."""
    monkeypatch.setattr(sys, "argv", ["magentic-marketplace", *args])
    cli.main()


def test_experiment_command_prints_one_line_per_run(
    experiment_config, monkeypatch, capsys
):
    """Detail 5: every run done prints `run_name done` per run and exits 0."""
    install_fake_marketplace(monkeypatch, fail_run=set())

    try:
        run_cli(
            monkeypatch, "experiment", str(experiment_config), "--env-file", "dummy.env"
        )
    except SystemExit as exit_:
        assert exit_.code in (0, None), capsys.readouterr()

    lines = capsys.readouterr().out.splitlines()
    assert "S2_rep1 done" in lines
    assert "S3_rep1 done" in lines
    run_folder = experiment_config.parents[1] / "research" / "results" / "exp1"
    assert (run_folder / "S2_rep1" / "seller_rounds.csv").is_file()
    assert (run_folder / "S3_rep1" / "seller_rounds.csv").is_file()


def test_experiment_command_exits_1_when_a_run_failed(
    experiment_config, monkeypatch, capsys
):
    """Detail 5 with Scenario: One failure, the rest finish: exit 1, others reported done."""
    install_fake_marketplace(monkeypatch, fail_run={"exp1_S3_rep1"})

    with pytest.raises(SystemExit) as excinfo:
        run_cli(
            monkeypatch, "experiment", str(experiment_config), "--env-file", "dummy.env"
        )

    lines = capsys.readouterr().out.splitlines()
    assert "S2_rep1 done" in lines
    assert "S3_rep1 failed" in lines
    assert excinfo.value.code == 1

from __future__ import annotations

import signal
from pathlib import Path

import pytest

from swe_agent import cli, train


def test_cli_forwards_complete_config(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = Path("configs/example.yaml")
    monkeypatch.setattr(
        train,
        "run",
        lambda path: {
            "config": Path(path).as_posix(),
            "status": "completed",
        },
    )

    assert cli.main(["grpo", "--config", config_path.as_posix()]) == 0
    assert '"config": "configs/example.yaml"' in capsys.readouterr().out


def test_cli_returns_nonzero_for_structured_failed_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(train, "run", lambda path: {"status": "failed"})
    assert cli.main(["grpo", "--config", "config.yaml"]) == 1
    assert '"status": "failed"' in capsys.readouterr().out


def test_cli_grpo_surface_contains_only_one_run_config_input() -> None:
    parser = cli.build_parser()
    grpo_parser = parser._subparsers._group_actions[0].choices["grpo"]
    option_strings = {
        option
        for action in grpo_parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }
    assert option_strings == {"--help", "--config"}


def test_cli_preserves_sigterm_exit_after_structured_cleanup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupt(path: Path):
        del path
        raise train.TrainingInterrupted(15, {"status": "interrupted"})

    monkeypatch.setattr(train, "run", interrupt)
    assert cli.main(["grpo", "--config", "config.yaml"]) == 143
    assert '"status": "interrupted"' in capsys.readouterr().out


def test_cli_preserves_sigterm_exit_from_supervisor_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        train,
        "run",
        lambda path: {"status": "interrupted", "interrupted_signum": signal.SIGTERM},
    )

    assert cli.main(["grpo", "--config", "config.yaml"]) == 143
    assert '"status": "interrupted"' in capsys.readouterr().out


def test_cli_rejects_unqualified_runtime(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def reject(path: Path) -> dict[str, object]:
        del path
        raise train.RuntimeNotQualifiedError("runtime_qualified=false")

    monkeypatch.setattr(train, "run", reject)
    assert cli.main(["grpo", "--config", "config.yaml"]) == 2
    assert "runtime_qualified=false" in capsys.readouterr().err


def test_signal_boundary_maps_first_signal() -> None:
    boundary = cli.SignalBoundary()
    with pytest.raises(cli.WorkflowTermination) as captured:
        boundary._handle(signal.SIGTERM, None)
    assert captured.value.signum == signal.SIGTERM


def test_cli_requires_explicit_config() -> None:
    with pytest.raises(SystemExit) as captured:
        cli.build_parser().parse_args(["grpo"])
    assert captured.value.code == 2

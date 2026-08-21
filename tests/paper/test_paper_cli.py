from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import typer
from typer.testing import CliRunner

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts/verify_paper_claims.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "verify_paper_claims", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_loaded_cli = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_loaded_cli)


class _CliModule(Protocol):
    app: typer.Typer


cli = cast(_CliModule, _loaded_cli)

runner = CliRunner()


def test_validate_command_reports_success(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        cli,
        "validate_repository",
        lambda root: SimpleNamespace(registry=SimpleNamespace(claims=(1, 2, 3))),
    )

    result = runner.invoke(cli.app, ["validate"])

    assert result.exit_code == 0
    assert result.stdout == "validated 3 claims\n"


def test_run_command_treats_scientific_verdicts_as_success(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        cli,
        "run_repository",
        lambda root, run_id, olmes_path: SimpleNamespace(
            manifest=SimpleNamespace(run_id=run_id)
        ),
    )

    result = runner.invoke(cli.app, ["run", "--run-id", "first-run"])

    assert result.exit_code == 0
    assert result.stdout == "created run first-run\n"


def test_render_command_reports_output_path(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        cli,
        "render_repository",
        lambda root, run_id: Path("docs/paper-reproduction-report.md"),
    )

    result = runner.invoke(cli.app, ["render", "--run-id", "first-run"])

    assert result.exit_code == 0
    assert result.stdout == "docs/paper-reproduction-report.md\n"


def test_validation_failure_is_nonzero(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fail(root: Path) -> None:
        raise ValueError("invalid repository")

    monkeypatch.setattr(cli, "validate_repository", fail)

    result = runner.invoke(cli.app, ["validate"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)

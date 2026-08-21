from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import typer
from typer.testing import CliRunner

_SCRIPTS_ROOT = Path(__file__).parents[2] / "scripts"
_SCRIPT_PATH = _SCRIPTS_ROOT / "validate_paper_findings.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "validate_paper_findings", _SCRIPT_PATH
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
_loaded_cli = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_loaded_cli)


class _CliModule(Protocol):
    app: typer.Typer


cli = cast(_CliModule, _loaded_cli)
runner = CliRunner()


def test_hard_cutover_has_one_cli_entrypoint() -> None:
    assert _SCRIPT_PATH.is_file()
    assert not (_SCRIPTS_ROOT / "verify_paper_claims.py").exists()


def test_validate_command_reports_static_and_input_counts(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    claims = (
        SimpleNamespace(attempt_ids=("a",), non_assessable_reason=None),
        SimpleNamespace(attempt_ids=(), non_assessable_reason="absent"),
        SimpleNamespace(attempt_ids=(), non_assessable_reason=None),
    )
    monkeypatch.setattr(
        cli,
        "validate_repository",
        lambda root, data_dir: SimpleNamespace(
            registry=SimpleNamespace(claims=claims), inputs=(1, 2, 3)
        ),
    )

    result = runner.invoke(cli.app, ["validate", "--data-dir", "fixture-data"])

    assert result.exit_code == 0
    assert result.stdout == "validated 3 claims, 2 empirical targets, 3 input tables\n"


def test_run_command_reports_format_two_identity(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        cli,
        "run_validation",
        lambda root, run_id, data_dir: SimpleNamespace(
            manifest=SimpleNamespace(run_id=run_id)
        ),
    )

    result = runner.invoke(
        cli.app,
        ["run", "--run-id", "first-run", "--data-dir", "fixture-data"],
    )

    assert result.exit_code == 0
    assert result.stdout == "created format-2 run first-run\n"


def test_render_command_is_a_thin_bundle_renderer(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        cli,
        "render_validation",
        lambda root, run_id: SimpleNamespace(
            report=Path("docs/paper-validation-report.md")
        ),
    )

    result = runner.invoke(cli.app, ["render", "--run-id", "first-run"])

    assert result.exit_code == 0
    assert result.stdout == "docs/paper-validation-report.md\n"


def test_validation_failure_is_nonzero(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fail(root: Path, data_dir: Path) -> None:
        raise ValueError("invalid validation surface")

    monkeypatch.setattr(cli, "validate_repository", fail)

    result = runner.invoke(cli.app, ["validate"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)

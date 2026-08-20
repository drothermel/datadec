from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

from typer.testing import CliRunner

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/preprocess_published_results.py"
SCRIPT_MODULE = "datadec_preprocess_published_results_script"
spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, SCRIPT_PATH)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
sys.modules[SCRIPT_MODULE] = script
spec.loader.exec_module(script)

DEFAULT_DATA_DIR = script.DEFAULT_DATA_DIR
app = script.app
runner = CliRunner()


def test_cli_defaults_to_all_manifest_units_independent_of_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    paths = object()
    manifest = object()
    units = ("outputs2", "per-task-arc-easy")
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(script, "load_published_results_manifest", return_value=manifest),
        patch.object(
            script, "resolve_published_result_units", return_value=units
        ) as resolve,
        patch.object(script, "DataDecidePaths", return_value=paths) as path_type,
        patch.object(script, "preprocess_published_results") as preprocess,
    ):
        result = runner.invoke(app, [])

    assert result.exit_code == 0
    resolve.assert_called_once_with((), manifest)
    path_type.assert_called_once_with(DEFAULT_DATA_DIR)
    preprocess.assert_called_once_with(
        paths, units=units, manifest=manifest, verbose=True
    )
    assert DEFAULT_DATA_DIR == Path(__file__).resolve().parents[1] / "data"


def test_cli_forwards_repeatable_units_and_data_dir(tmp_path: Path) -> None:
    manifest = object()
    with (
        patch.object(script, "load_published_results_manifest", return_value=manifest),
        patch.object(
            script,
            "resolve_published_result_units",
            return_value=("outputs2", "per-task-arc-easy"),
        ) as resolve,
        patch.object(script, "preprocess_published_results") as preprocess,
    ):
        result = runner.invoke(
            app,
            [
                "--unit",
                "per-task-arc-easy",
                "--unit",
                "outputs2",
                "--data-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0
    resolve.assert_called_once_with(["per-task-arc-easy", "outputs2"], manifest)
    preprocess.assert_called_once()
    assert preprocess.call_args.kwargs == {
        "units": ("outputs2", "per-task-arc-easy"),
        "manifest": manifest,
        "verbose": True,
    }


def test_cli_reports_unknown_unit_as_usage_error() -> None:
    with patch.object(
        script,
        "resolve_published_result_units",
        side_effect=ValueError("unknown published-results unit: missing"),
    ):
        result = runner.invoke(app, ["--unit", "missing"])

    assert result.exit_code == 2
    assert "unknown published-results unit: missing" in result.output

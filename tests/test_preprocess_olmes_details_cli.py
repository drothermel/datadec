from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

from typer.testing import CliRunner

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/preprocess_olmes_details.py"
SCRIPT_MODULE = "datadec_preprocess_olmes_details_script"
spec = importlib.util.spec_from_file_location(SCRIPT_MODULE, SCRIPT_PATH)
assert spec is not None and spec.loader is not None
script = importlib.util.module_from_spec(spec)
sys.modules[SCRIPT_MODULE] = script
spec.loader.exec_module(script)

DEFAULT_DATA_DIR = script.DEFAULT_DATA_DIR
app = script.app

runner = CliRunner()


def test_cli_default_is_repo_data_independent_of_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    paths = object()
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(script, "DataDecidePaths", return_value=paths) as path_type,
        patch.object(script, "preprocess_olmes_details") as preprocess,
    ):
        result = runner.invoke(app, ["--recipe", "c4"])

    assert result.exit_code == 0
    path_type.assert_called_once_with(DEFAULT_DATA_DIR)
    preprocess.assert_called_once_with(
        paths,
        "c4",
        input_path=None,
        output_tasks_path=None,
        output_instances_path=None,
        output_choices_path=None,
        verbose=True,
    )
    assert DEFAULT_DATA_DIR == Path(__file__).resolve().parents[1] / "data"


def test_cli_data_dir_invokes_preprocess_for_each_recipe(tmp_path: Path) -> None:
    paths = object()
    with (
        patch.object(script, "DataDecidePaths", return_value=paths),
        patch.object(script, "preprocess_olmes_details") as preprocess,
        patch("datadec.data.download.download_sources") as download_sources,
    ):
        result = runner.invoke(
            app,
            ["--recipe", "c4", "--recipe", "fineweb-pro", "--data-dir", str(tmp_path)],
        )

    assert result.exit_code == 0
    assert preprocess.call_count == 2
    preprocess.assert_any_call(
        paths,
        "c4",
        input_path=None,
        output_tasks_path=None,
        output_instances_path=None,
        output_choices_path=None,
        verbose=True,
    )
    preprocess.assert_any_call(
        paths,
        "fineweb-pro",
        input_path=None,
        output_tasks_path=None,
        output_instances_path=None,
        output_choices_path=None,
        verbose=True,
    )
    download_sources.assert_not_called()


def test_cli_path_overrides_are_forwarded(tmp_path: Path) -> None:
    paths = object()
    input_path = tmp_path / "custom-input.tar.gz"
    output_tasks = tmp_path / "tasks.parquet"
    output_instances = tmp_path / "instances.parquet"
    output_choices = tmp_path / "choices.parquet"
    with (
        patch.object(script, "DataDecidePaths", return_value=paths),
        patch.object(script, "preprocess_olmes_details") as preprocess,
    ):
        result = runner.invoke(
            app,
            [
                "--recipe",
                "c4",
                "--input",
                str(input_path),
                "--output-tasks",
                str(output_tasks),
                "--output-instances",
                str(output_instances),
                "--output-choices",
                str(output_choices),
            ],
        )

    assert result.exit_code == 0
    preprocess.assert_called_once_with(
        paths,
        "c4",
        input_path=input_path,
        output_tasks_path=output_tasks,
        output_instances_path=output_instances,
        output_choices_path=output_choices,
        verbose=True,
    )


def test_cli_rejects_unknown_recipe() -> None:
    result = runner.invoke(app, ["--recipe", "missing"])
    assert result.exit_code != 0
    assert "unknown OLMES detail recipe" in result.output

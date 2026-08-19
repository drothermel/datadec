from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.config import load_source_manifest
from datadec.data.download import resolve_olmes_detail_recipes
from datadec.data.paths import DataDecidePaths
from datadec.data.preprocess.olmes_details import preprocess_olmes_details

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer()


@app.command()
def main(
    recipe: Annotated[list[str], typer.Option("--recipe")],
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
    input_path: Annotated[
        Path | None, typer.Option("--input", help="Override recipe detail archive")
    ] = None,
    output_tasks: Annotated[
        Path | None, typer.Option("--output-tasks", help="Override tasks parquet path")
    ] = None,
    output_instances: Annotated[
        Path | None,
        typer.Option("--output-instances", help="Override instances parquet path"),
    ] = None,
    output_choices: Annotated[
        Path | None, typer.Option("--output-choices", help="Override choices parquet path")
    ] = None,
) -> None:
    """Preprocess local OLMES detail archives into typed task/instance/choice parquet."""
    manifest = load_source_manifest()
    try:
        recipes = resolve_olmes_detail_recipes(recipe, manifest.olmes_details)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--recipe") from exc
    if len(recipes) != 1 and input_path is not None:
        raise typer.BadParameter(
            "--input requires exactly one --recipe",
            param_hint="--recipe",
        )
    paths = DataDecidePaths(data_dir)
    for detail_recipe in recipes:
        preprocess_olmes_details(
            paths,
            detail_recipe,
            input_path=input_path,
            output_tasks_path=output_tasks,
            output_instances_path=output_instances,
            output_choices_path=output_choices,
            verbose=True,
        )


if __name__ == "__main__":
    app()

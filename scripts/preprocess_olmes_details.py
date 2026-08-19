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
) -> None:
    """Preprocess local OLMES detail archives into typed task-summary parquet."""
    manifest = load_source_manifest()
    try:
        recipes = resolve_olmes_detail_recipes(recipe, manifest.olmes_details)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--recipe") from exc
    paths = DataDecidePaths(data_dir)
    for detail_recipe in recipes:
        preprocess_olmes_details(paths, detail_recipe, verbose=True)


if __name__ == "__main__":
    app()

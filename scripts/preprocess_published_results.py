from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.config import load_published_results_manifest
from datadec.data.paths import DataDecidePaths
from datadec.data.preprocess.published_results import (
    preprocess_published_results,
    resolve_published_result_units,
)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer()


@app.command()
def main(
    unit: Annotated[list[str] | None, typer.Option("--unit")] = None,
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
) -> None:
    """Convert local structured published results to one-to-one Parquet files."""
    manifest = load_published_results_manifest()
    try:
        units = resolve_published_result_units(unit or (), manifest)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--unit") from exc
    preprocess_published_results(
        DataDecidePaths(data_dir), units=units, manifest=manifest, verbose=True
    )


if __name__ == "__main__":
    app()

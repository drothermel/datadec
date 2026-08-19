from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.data.paths import DataDecidePaths
from datadec.data.preprocess import preprocess_scaling_law

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer()


@app.command()
def main(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
) -> None:
    """Preprocess the three local raw scaling-law CSV artifacts."""
    preprocess_scaling_law(DataDecidePaths(data_dir), verbose=True)


if __name__ == "__main__":
    app()

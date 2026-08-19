from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.data.paths import DataDecidePaths
from datadec.data.preprocess.olmes import preprocess_olmes

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer()


@app.command()
def main(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
) -> None:
    """Preprocess the local raw OLMES parquet artifact."""
    preprocess_olmes(DataDecidePaths(data_dir), verbose=True)


if __name__ == "__main__":
    app()

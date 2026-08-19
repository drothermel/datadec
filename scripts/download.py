from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.data.download import download_sources
from datadec.data.paths import DataDecidePaths

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = typer.Typer()


@app.command()
def main(
    ppl: Annotated[bool, typer.Option("--ppl")] = False,
    olmes: Annotated[bool, typer.Option("--olmes")] = False,
    olmes_details: Annotated[list[str] | None, typer.Option("--olmes-details")] = None,
    scaling_law: Annotated[bool, typer.Option("--scaling-law")] = False,
    published_results: Annotated[bool, typer.Option("--published-results")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    data_dir: Annotated[Path, typer.Option("--data-dir")] = DEFAULT_DATA_DIR,
) -> None:
    """Download selected DataDecide source data."""
    details = olmes_details or []
    if (
        not ppl
        and not olmes
        and not details
        and not scaling_law
        and not published_results
    ):
        raise typer.BadParameter(
            "select --ppl, --olmes, --olmes-details, --scaling-law, or "
            "--published-results"
        )
    try:
        download_sources(
            DataDecidePaths(data_dir),
            ppl=ppl,
            olmes=olmes,
            olmes_details=details,
            scaling_law=scaling_law,
            published_results=published_results,
            force=force,
            verbose=True,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    app()

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.paper.run import render_repository, run_repository, validate_repository

app = typer.Typer(
    add_completion=False,
    help="Validate and run the repository-owned DataDecide paper workflow.",
    no_args_is_help=True,
)


@app.command()
def validate() -> None:
    """Validate the current repository without writing outputs."""
    result = validate_repository(Path.cwd())
    typer.echo(f"validated {len(result.registry.claims)} claims")


@app.command()
def run(
    run_id: Annotated[str, typer.Option("--run-id")],
    olmes_path: Annotated[
        Path | None,
        typer.Option("--olmes-path", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
) -> None:
    """Create one immutable first-run observation bundle."""
    bundle = run_repository(Path.cwd(), run_id, olmes_path)
    typer.echo(f"created run {bundle.manifest.run_id}")


@app.command()
def render(run_id: Annotated[str, typer.Option("--run-id")]) -> None:
    """Render one selected immutable run into the configured report."""
    path = render_repository(Path.cwd(), run_id)
    typer.echo(path)


if __name__ == "__main__":
    app()

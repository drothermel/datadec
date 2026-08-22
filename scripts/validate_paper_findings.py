from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from datadec.paper.run import render_validation, run_validation, validate_repository

app = typer.Typer(
    add_completion=False,
    help="Validate DataDecide paper findings from dd_parsed inputs.",
    no_args_is_help=True,
)


@app.command()
def validate(
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
) -> None:
    """Validate contracts, source coverage, and input schemas without writing."""
    surface = validate_repository(Path.cwd(), data_dir)
    typer.echo(
        f"validated {len(surface.registry.claims)} claims, "
        f"{sum(claim.attempt_ids != () or claim.non_assessable_reason is not None for claim in surface.registry.claims)} empirical targets, "
        f"{len(surface.inputs)} input tables"
    )


@app.command()
def run(
    run_id: Annotated[str, typer.Option("--run-id")],
    data_dir: Annotated[Path, typer.Option("--data-dir")] = Path("data"),
) -> None:
    """Run every configured analysis and persist one immutable bundle."""
    bundle = run_validation(Path.cwd(), run_id, data_dir)
    typer.echo(f"created format-3 run {bundle.manifest.run_id}")


@app.command()
def render(run_id: Annotated[str, typer.Option("--run-id")]) -> None:
    """Render one completed bundle without reopening scientific inputs."""
    rendered = render_validation(Path.cwd(), run_id)
    typer.echo(rendered.report)


if __name__ == "__main__":
    app()

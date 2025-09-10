#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import click

from datadec.wandb_downloader import WandBDownloader
from datadec.wandb_store import WandBStore


def _progress_callback(run_index: int, total_runs: int, run_name: str) -> None:
    click.echo(f"Processing run {run_index}/{total_runs}: {run_name}")


def download_wandb_data(
    entity: str, project: str, store: WandBStore, force_refresh: bool = False
) -> None:
    downloader = WandBDownloader(store)
    try:
        all_runs = list(downloader.api.runs(f"{entity}/{project}"))
        click.echo(f"Found {len(all_runs)} total runs in {entity}/{project}")
        runs_to_download = downloader.get_runs_to_download(
            entity, project, force_refresh
        )
        if not runs_to_download:
            click.echo("No new data to download!")
            return
        if force_refresh:
            click.echo("Force refresh enabled - downloading all runs")
        click.echo(f"Total runs to process: {len(runs_to_download)}")
        stats = downloader.download_project(
            entity, project, force_refresh, _progress_callback
        )
        click.echo(f"✓ Downloaded {stats['total_runs']} runs to database")
        click.echo(f"  - New runs: {stats['new_runs']}")
        click.echo(f"  - Updated runs: {stats['updated_runs']}")
    except RuntimeError as e:
        if "api_key not configured" in str(e):
            click.echo("❌ WandB API key not configured. Please run: wandb login")
            click.echo("   Or set WANDB_API_KEY environment variable")
            raise click.ClickException("WandB authentication required")
        raise


@click.command()
@click.option("--entity", required=True, help="WandB entity (username or team name)")
@click.option("--project", required=True, help="WandB project name")
@click.option(
    "--database-url",
    default="postgresql://localhost/wandb",
    help="PostgreSQL connection string",
)
@click.option(
    "--output-dir", help="Optional: Export to parquet files in this directory"
)
@click.option(
    "--force-refresh",
    is_flag=True,
    help="Re-download all data, ignoring existing cache",
)
def main(
    entity: str, project: str, database_url: str, output_dir: str, force_refresh: bool
) -> None:
    click.echo(f"Downloading data from {entity}/{project}")
    click.echo(f"Database: {database_url}")
    try:
        store = WandBStore(database_url)
        click.echo("✓ Connected to database")
        download_wandb_data(entity, project, store, force_refresh)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            store.export_to_parquet(str(output_path))
            click.echo(f"✓ Exported to parquet files: {output_path.absolute()}")
        click.echo("✅ WandB download completed!")
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise


if __name__ == "__main__":
    main()

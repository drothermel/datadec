#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import click

from datadec.wandb_eval.wandb_downloader import WandBDownloader
from datadec.wandb_eval.wandb_store import WandBStore


def _progress_callback(run_index: int, total_runs: int, run_name: str) -> None:
    click.echo(f"Processing run {run_index}/{total_runs}: {run_name}")


def sync_wandb_tags(
    entity: str, project: str, store: WandBStore, mode: str = "recent", days: int = 7
) -> None:
    downloader = WandBDownloader(store)
    try:
        click.echo(f"=== WandB Tag Synchronization ===")
        click.echo(f"Entity: {entity}")
        click.echo(f"Project: {project}")
        click.echo(f"Mode: {mode}")
        if mode == "recent":
            click.echo(f"Days: {days}")
        click.echo()

        stats = downloader.sync_tags(entity, project, mode, days, _progress_callback)

        click.echo(f"✓ Tag sync completed!")
        click.echo(f"  - Total runs processed: {stats['total_runs']}")
        click.echo(f"  - Runs updated: {stats['updated_runs']}")

        # Show tag statistics
        runs_df = store.get_runs(entity=entity, project=project)
        if "wandb_tags" in runs_df.columns:
            tagged_runs = runs_df["wandb_tags"].notna().sum()
            total_runs = len(runs_df)
            click.echo(f"  - Runs with tags: {tagged_runs}/{total_runs} ({tagged_runs / total_runs * 100:.1f}%)")

    except RuntimeError as e:
        if "api_key not configured" in str(e):
            click.echo("❌ WandB API key not configured. Please run: wandb login")
            click.echo("   Or set WANDB_API_KEY environment variable")
            raise click.ClickException("WandB authentication required")
        raise


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
@click.option(
    "--sync-tags-only",
    is_flag=True,
    help="Only sync tags, don't download runs/history",
)
@click.option(
    "--also-sync-tags",
    is_flag=True,
    help="Download runs and also sync tags",
)
@click.option(
    "--sync-tags-mode",
    type=click.Choice(["recent", "all", "finished"]),
    default="recent",
    help="Tag sync mode: recent (last N days), all (all runs), finished (finished runs only)",
)
@click.option(
    "--sync-tags-days",
    type=int,
    default=7,
    help="Number of recent days to sync (for 'recent' mode)",
)
def main(
    entity: str,
    project: str,
    database_url: str,
    output_dir: str,
    force_refresh: bool,
    sync_tags_only: bool,
    also_sync_tags: bool,
    sync_tags_mode: str,
    sync_tags_days: int,
) -> None:
    # Validate mutually exclusive options
    if sync_tags_only and also_sync_tags:
        raise click.ClickException("Cannot use both --sync-tags-only and --also-sync-tags")

    operation = "Tag sync" if sync_tags_only else "Download"
    if also_sync_tags:
        operation = "Download + Tag sync"

    click.echo(f"{operation} for {entity}/{project}")
    click.echo(f"Database: {database_url}")

    try:
        store = WandBStore(database_url)
        click.echo("✓ Connected to database")

        if sync_tags_only:
            # Only sync tags
            sync_wandb_tags(entity, project, store, sync_tags_mode, sync_tags_days)
        else:
            # Download runs/history (and optionally tags)
            download_wandb_data(entity, project, store, force_refresh)

            if also_sync_tags:
                click.echo("\n" + "="*50)
                sync_wandb_tags(entity, project, store, sync_tags_mode, sync_tags_days)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            store.export_to_parquet(str(output_path))
            click.echo(f"✓ Exported to parquet files: {output_path.absolute()}")

        success_msg = "✅ Tag sync completed!" if sync_tags_only else "✅ WandB download completed!"
        if also_sync_tags:
            success_msg = "✅ Download + Tag sync completed!"
        click.echo(success_msg)

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise


if __name__ == "__main__":
    main()

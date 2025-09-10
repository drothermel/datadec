#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
import wandb


def load_existing_data(output_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load existing metadata and history data if available."""
    metadata_path = output_path / "runs_metadata.parquet"
    history_path = output_path / "runs_history.parquet"

    existing_metadata = pd.DataFrame()
    existing_history = pd.DataFrame()

    if metadata_path.exists():
        existing_metadata = pd.read_parquet(metadata_path)
        click.echo(f"Loaded existing metadata: {len(existing_metadata)} runs")

    if history_path.exists():
        existing_history = pd.read_parquet(history_path)
        click.echo(f"Loaded existing history: {len(existing_history)} records")

    return existing_metadata, existing_history


def get_runs_to_download(api_runs: list, existing_metadata: pd.DataFrame) -> list:
    """Determine which runs need to be downloaded or re-downloaded."""
    if existing_metadata.empty:
        click.echo("No existing data - downloading all runs")
        return api_runs

    existing_run_ids = set(existing_metadata["run_id"])
    existing_unfinished = set(
        existing_metadata[existing_metadata["state"] != "finished"]["run_id"]
    )

    new_runs = [run for run in api_runs if run.id not in existing_run_ids]
    redownload_runs = [run for run in api_runs if run.id in existing_unfinished]

    total_to_download = len(new_runs) + len(redownload_runs)

    click.echo(f"Found {len(new_runs)} new runs")
    click.echo(f"Found {len(redownload_runs)} unfinished runs to re-download")
    click.echo(f"Total runs to download: {total_to_download}")

    return new_runs + redownload_runs


def download_wandb_data(
    entity: str, project: str, output_path: Path, force_refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download complete data from WandB project using scan_history for full resolution."""
    try:
        api = wandb.Api()
    except wandb.errors.UsageError as e:
        if "api_key not configured" in str(e):
            click.echo("❌ WandB API key not configured. Please run: wandb login")
            click.echo("   Or set WANDB_API_KEY environment variable")
            raise click.ClickException("WandB authentication required")
        raise

    # Get all runs from the project
    all_runs = api.runs(f"{entity}/{project}")
    click.echo(f"Found {len(all_runs)} total runs in {entity}/{project}")

    # Load existing data and determine what to download
    existing_metadata, existing_history = load_existing_data(output_path)

    if force_refresh:
        click.echo("Force refresh enabled - downloading all runs")
        runs_to_download = all_runs
        # Clear existing data for fresh start
        existing_metadata = pd.DataFrame()
        existing_history = pd.DataFrame()
    else:
        runs_to_download = get_runs_to_download(all_runs, existing_metadata)

    if not runs_to_download:
        click.echo("No new data to download!")
        return existing_metadata, existing_history

    # Process runs one by one and write incrementally
    new_metadata_list = []

    for i, run in enumerate(runs_to_download):
        click.echo(f"Processing run {i + 1}/{len(runs_to_download)}: {run.name}")

        # Collect run metadata
        metadata = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": run.created_at,
            "runtime": run._attrs.get("runtime", 0),
        }

        # Add config parameters
        for key, value in run.config.items():
            if not key.startswith("_"):
                metadata[f"config_{key}"] = value

        # Add summary metrics
        for key, value in run.summary.items():
            if not key.startswith("_"):
                metadata[f"summary_{key}"] = value

        new_metadata_list.append(metadata)

        # Get complete history using scan_history (non-downsampled)
        history_data = []
        for step_data in run.scan_history():
            step_data["run_id"] = run.id
            step_data["run_name"] = run.name
            history_data.append(step_data)

        # Write data immediately after each run
        if history_data:
            run_history_df = pd.DataFrame(history_data)

            # Remove old data for this run if it exists
            if not existing_history.empty:
                existing_history = existing_history[
                    existing_history["run_id"] != run.id
                ]

            # Append new data
            if existing_history.empty:
                updated_history = run_history_df
            else:
                updated_history = pd.concat(
                    [existing_history, run_history_df], ignore_index=True
                )

            # Write to file
            history_path = output_path / "runs_history.parquet"
            updated_history.to_parquet(history_path, index=False)
            existing_history = updated_history  # Update for next iteration

        # Update metadata file after each run too
        run_metadata_df = pd.DataFrame([metadata])

        # Remove old metadata for this run if it exists
        if not existing_metadata.empty:
            existing_metadata = existing_metadata[existing_metadata["run_id"] != run.id]

        # Append new metadata
        if existing_metadata.empty:
            updated_metadata = run_metadata_df
        else:
            updated_metadata = pd.concat(
                [existing_metadata, run_metadata_df], ignore_index=True
            )

        # Write to file
        metadata_path = output_path / "runs_metadata.parquet"
        updated_metadata.to_parquet(metadata_path, index=False)
        existing_metadata = updated_metadata  # Update for next iteration

        click.echo(f"  ✓ Saved data for run {run.name}")

    # Return the current state (already written to files)
    click.echo(f"Downloaded and saved {len(new_metadata_list)} runs")
    click.echo(f"Data written incrementally to: {output_path}")

    return existing_metadata, existing_history


@click.command()
@click.option("--entity", required=True, help="WandB entity (username or team name)")
@click.option("--project", required=True, help="WandB project name")
@click.option(
    "--output-dir", default="./wandb_data", help="Directory to save downloaded data"
)
@click.option(
    "--force-refresh",
    is_flag=True,
    help="Re-download all data, ignoring existing cache",
)
def main(entity: str, project: str, output_dir: str, force_refresh: bool) -> None:
    """Download data from WandB and save as parquet files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    click.echo(f"Downloading data from {entity}/{project}")
    click.echo(f"Output directory: {output_path.absolute()}")

    try:
        # Download data (writes incrementally during download)
        metadata_df, history_df = download_wandb_data(
            entity, project, output_path, force_refresh
        )

        click.echo("✅ WandB download completed!")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise


if __name__ == "__main__":
    main()

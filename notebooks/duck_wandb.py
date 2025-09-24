import marimo

__generated_with = "0.16.0"
app = marimo.App(width="columns")


@app.cell
def _():
    from dr_wandb import fetch_project_runs
    from dr_ingest.serialization import (
        dump_runs_and_history,
        ensure_parquet,
        compare_sizes,
    )

    def make_progress_callback(log_every: int = 10):
        def progress(idx: int, total: int, name: str) -> None:
            if idx % log_every == 0:
                print(f"Processing run {idx}/{total}: {name}")

        return progress

    return (
        fetch_project_runs,
        dump_runs_and_history,
        ensure_parquet,
        compare_sizes,
        make_progress_callback,
    )


@app.cell
def _(Path):
    def get_file_size_mb(file_path):
        return Path(file_path).stat().st_size / (1024 * 1024)

    return (get_file_size_mb,)


@app.cell
def _(
    fetch_project_runs,
    dump_runs_and_history,
    ensure_parquet,
    compare_sizes,
    make_progress_callback,
):
    def download_all_from_wandb(
        entity, project, runs_per_page=500, log_every: int = 10
    ):
        progress = make_progress_callback(log_every)
        runs, histories = fetch_project_runs(
            entity,
            project,
            runs_per_page=runs_per_page,
            include_history=True,
            progress_callback=progress,
        )
        print(
            f">> Finished downloading, {len(runs)} runs and {len(histories)} histories"
        )
        return runs, histories

    return (download_all_from_wandb,)


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    from typing import Any
    import srsly
    import duckdb
    from pathlib import Path
    import pandas as pd
    from collections import defaultdict
    import quak

    return Any, Path, defaultdict, duckdb, pd, quak, srsly


@app.cell
def _():
    ENTITY = "ml-moe"
    PROJECT = "ft-scaling"
    RUNS_PER_PAGE = 500
    WANDB_RUNS_FILENAME = "wandb_runs"
    WANDB_HISTORY_FILENAME = "wandb_history"
    OUT_DIR = "."
    return (
        ENTITY,
        OUT_DIR,
        PROJECT,
        RUNS_PER_PAGE,
        WANDB_HISTORY_FILENAME,
        WANDB_RUNS_FILENAME,
    )


@app.cell
def _(
    ENTITY,
    OUT_DIR,
    PROJECT,
    Path,
    RUNS_PER_PAGE,
    WANDB_HISTORY_FILENAME,
    WANDB_RUNS_FILENAME,
    download_all_from_wandb,
    dump_runs_and_history,
    ensure_parquet,
    compare_sizes,
):
    runs_json = Path(OUT_DIR, f"{WANDB_RUNS_FILENAME}.jsonl")
    history_json = Path(OUT_DIR, f"{WANDB_HISTORY_FILENAME}.jsonl")

    if not runs_json.exists() or not history_json.exists():
        runs, histories = download_all_from_wandb(
            ENTITY, PROJECT, runs_per_page=RUNS_PER_PAGE
        )
        dump_runs_and_history(
            Path(OUT_DIR), WANDB_RUNS_FILENAME, WANDB_HISTORY_FILENAME, runs, histories
        )

    runs_parquet = ensure_parquet(runs_json)
    history_parquet = ensure_parquet(history_json)

    for path, size in compare_sizes(
        runs_json, runs_parquet, history_json, history_parquet
    ).items():
        print(f"{path}: {size:.2f} MB")

    return (runs_parquet,)


@app.cell
def _(pd, runs_parquet):
    runs_df = pd.read_parquet(runs_parquet)
    runs_df.shape
    return (runs_df,)


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="columns")


@app.cell(column=0)
def _(Path, WANDB_HISTORY_FILENAME, WANDB_RUNS_FILENAME, srsly):
    def dump_runs_hist_to_json(runs, hists, out_dir):
        if not srsly.is_json_serializable(runs):
            print(">> Runs not json serializable")
            return
        if not srsly.is_json_serializable(hists):
            print(">> Hists not json serializable")
            return

        srsly.write_jsonl(Path(out_dir, WANDB_RUNS_FILENAME), runs)
        srsly.write_jsonl(Path(out_dir, WANDB_HISTORY_FILENAME), hists)
        print(f">> Dumped files to {out_dir}")
    return (dump_runs_hist_to_json,)


@app.cell
def _(Path, duckdb):
    def load_json_dump_parquet(in_file, out_file):
        in_name = Path(in_file).stem
        duckdb.execute(f"CREATE TABLE {in_name} AS SELECT * FROM read_json('{in_file}')")
        duckdb.execute(
            f"COPY {in_name} TO '{out_file}' (FORMAT parquet, PARQUET_VERSION v2)"
        )
        print(f">> Converted {in_file} to {out_file}")
    return (load_json_dump_parquet,)


@app.cell
def _(WANDB_HISTORY_FILENAME, WANDB_RUNS_FILENAME, get_file_size_mb):
    def print_size_comparisons():
        rjl = get_file_size_mb(f"{WANDB_RUNS_FILENAME}.jsonl")
        hjl = get_file_size_mb(f"{WANDB_HISTORY_FILENAME}.jsonl")
        rjp = get_file_size_mb(f"{WANDB_RUNS_FILENAME}.parquet")
        hjp = get_file_size_mb(f"{WANDB_HISTORY_FILENAME}.parquet")
        print(f"Runs Size: {rjl:.2f}MB vs parquet: {rjp:.2f}MB")
        print(f"Hist Size: {hjl:.2f}MB vs parquet: {hjp:.2f}MB")
    return (print_size_comparisons,)


@app.cell
def _():
    from functools import partial
    from dr_wandb import fetch_project_runs

    def make_progress_callback(log_every: int = 10):
        def progress(idx: int, total: int, name: str) -> None:
            if idx % log_every == 0:
                print(f"Processing run {idx}/{total}: {name}")

        return progress

    return (fetch_project_runs, make_progress_callback, partial)


@app.cell
def _(Path):
    def get_file_size_mb(file_path):
        return Path(file_path).stat().st_size / (1024 * 1024)
    return (get_file_size_mb,)


@app.cell
def _(fetch_project_runs, make_progress_callback, partial):
    def download_all_from_wandb(entity, project, runs_per_page=500):
        progress = partial(make_progress_callback(),)
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
    import marimo as mo
    from typing import Any
    import itertools
    import srsly
    import duckdb
    from pathlib import Path
    import json
    from clumper import Clumper
    import pandas as pd
    from collections import defaultdict
    import quak
    return Any, Clumper, Path, defaultdict, duckdb, pd, quak, srsly


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
    dump_runs_hist_to_json,
    load_json_dump_parquet,
    print_size_comparisons,
):
    runs_in_path = Path(OUT_DIR, WANDB_RUNS_FILENAME + ".jsonl")
    hist_in_path = Path(OUT_DIR, WANDB_HISTORY_FILENAME + ".jsonl")
    if not runs_in_path.exists() or not hist_in_path.exists():
        all_run_dicts, all_run_history = download_all_from_wandb(
            ENTITY,
            PROJECT,
            runs_per_page=RUNS_PER_PAGE,
        )
        dump_runs_hist_to_json(all_run_dicts, all_run_history, OUT_DIR)
    if not runs_in_path.with_suffix(".parquet").exists():
        load_json_dump_parquet(runs_in_path, runs_in_path.with_suffix(".parquet"))
    if not hist_in_path.with_suffix(".parquet").exists():
        load_json_dump_parquet(hist_in_path, hist_in_path.with_suffix(".parquet"))
    print_size_comparisons()
    return (runs_in_path,)


@app.cell
def _(pd, runs_in_path):
    runs_df = pd.read_parquet(runs_in_path.with_suffix(".parquet"))
    runs_df.shape
    return (runs_df,)


@app.cell
def _(Clumper):
    def select_oe_eval_metrics(d):
        return {k: v for k, v in d.items() if k.startswith("oe_eval_metrics")}


    def group_by_task(d):
        merge_oe = {}
        for k, v in d.items():
            if v is None:
                continue

            parts = k.split("/")
            assert len(parts) in [2, 3]

            if len(parts) == 3:
                _, task, metric = parts
                if task not in merge_oe:
                    merge_oe[task] = {}
                merge_oe[task][metric] = v
            elif len(parts) == 2:
                _, task = parts
                merge_oe[task] = v
        return merge_oe


    def drop_all_none_subdicts(d):
        return {k: v for k, v in d.items() if not all(vv is None for vv in v.valueS())}


    def drop_task_config(d):
        return {
            k: {kk: vv for kk, vv in v.items() if v not in "task_config"}
            for k, v in d.items()
        }


    def normalize_oe(in_dict):
        d = select_oe_eval_metrics(in_dict)
        d = group_by_task(d)
        d = [{"task": k, **v} for k, v in d.items()]
        d = (
            Clumper(d)
            .drop("task_config")
            .map(lambda x: {k: v for k, v in x.items() if v is not None})  # Drop v = None
            .keep(lambda dd: len(dd) > 1)  # Drop tasks without metrics
            .map(
                lambda x: {**x, **x.get("extra_metrics", {})}
            )  # Move extra metrics to top level
            .drop("extra_metrics")
            .collect()
        )
        new_d = {}
        for dd in d:
            new_d[dd["task"]] = {k: v for k, v in dd.items() if k != "task"}
        return new_d
    return (normalize_oe,)


@app.cell
def _():
    def flatten_agg_fields_to_keep(d, keep_fields):
        new_dict = {**d}
        for field in keep_fields:
            new_dict.update(d.get(field, {}))
            del new_dict[field]
        return new_dict


    def drop_prefixes(d, prefixes):
        return {k: v for k, v in d.items() if not any(k.startswith(p) for p in prefixes)}


    def drop_none_values(d):
        return {k: v for k, v in d.items() if v is not None}


    def ndarray_to_list(d):
        return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()}
    return (
        drop_none_values,
        drop_prefixes,
        flatten_agg_fields_to_keep,
        ndarray_to_list,
    )


@app.cell
def _(defaultdict, srsly):
    def find_constant_keys(d):
        key_vals = defaultdict(set)
        for row in d:
            for k, v in row.items():
                if isinstance(v, dict | list):
                    v = srsly.json_dumps(v)
                key_vals[k].add(v)
        constant_keys = [k for k, v in key_vals.items() if len(v) == 1]
        return constant_keys
    return (find_constant_keys,)


@app.cell
def _(
    Clumper,
    drop_none_values,
    drop_prefixes,
    find_constant_keys,
    flatten_agg_fields_to_keep,
    ndarray_to_list,
    normalize_oe,
    pd,
    runs_df,
):
    aggregate_fields = [
        "config",
        "summary",
        "wandb_metadata",
        "system_attrs",
        "sweep_info",
        "system_metrics",
    ]
    huge = [
        "tokenizer_files_hash",
        "tokenizer",
    ]
    duplicates = [
        "run_name",
        "exp_name",
        "wandb_entity",
        "wandb_project_name",
    ]
    runs_clean_py = (
        Clumper(runs_df.to_dict(orient="records"))
        .mutate(oe_eval=lambda x: normalize_oe(x["summary"]))
        .map(lambda x: flatten_agg_fields_to_keep(x, ["config", "summary"]))
        .drop(*aggregate_fields, *duplicates, *huge)
        .map(lambda x: drop_prefixes(x, ["_", "pretrain_eval", "oe_eval_metrics"]))
        .map(drop_none_values)
        .map(ndarray_to_list)
        .head(51)
        .collect()
    )
    drop_keys = find_constant_keys(runs_clean_py)
    runs_clean_df = pd.DataFrame(runs_clean_py).drop(columns=drop_keys)
    return runs_clean_df, runs_clean_py


@app.cell
def _(runs_clean_py):
    print(type(runs_clean_py[50]["oe_eval"]))
    return


@app.cell
def _(duckdb):
    conn = duckdb.connect()
    conn.execute("CREATE TABLE runs_clean AS SELECT * FROM runs_clean_df")
    return


@app.cell
def _(runs_clean_df):
    runs_clean_df
    return


@app.cell
def _(quak, runs_clean_df):
    widget = quak.Widget(runs_clean_df)
    widget
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()

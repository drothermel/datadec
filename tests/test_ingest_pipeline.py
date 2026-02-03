from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from datadec.ingest.metrics_all.constants import LoadMetricsAllConfig
from datadec.ingest.pipeline import (
    clean_cached_results,
    collect_all_eval_paths,
    export_eval_dir_to_parquet,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_collect_and_export_eval_dir(tmp_path: Path) -> None:
    metrics_root = tmp_path / "metrics_root"
    run_dir = metrics_root / "run1"
    run_dir.mkdir(parents=True)

    metrics_all_path = run_dir / "metrics-all.jsonl"
    _write_jsonl(
        metrics_all_path,
        [
            {
                "task_name": "task-001-sample",
                "task_idx": 1,
                "metrics": {"acc": 0.5},
            }
        ],
    )

    _write_jsonl(
        run_dir / "task-001-sample-predictions.jsonl",
        [{"doc_id": 1, "idx": 0, "prediction": "foo"}],
    )
    _write_jsonl(
        run_dir / "task-001-sample-recorded-inputs.jsonl",
        [{"doc_id": 1, "idx": 0, "task_name": "task-001-sample"}],
    )
    _write_jsonl(
        run_dir / "task-001-sample-requests.jsonl",
        [{"doc_id": 1, "idx": 0, "request": {"prompt": "bar"}}],
    )

    cfg = LoadMetricsAllConfig(root_paths=[metrics_root])
    entries = collect_all_eval_paths(cfg)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["metrics-all"] == metrics_all_path
    assert len(entry["predictions"]) == 1
    assert len(entry["recorded-inputs"]) == 1
    assert len(entry["requests"]) == 1

    cache_dir = tmp_path / "cache"
    conn = duckdb.connect()
    try:
        parquet_path = export_eval_dir_to_parquet(conn, entry, cache_dir)
    finally:
        conn.close()

    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    assert "file_prefix" in df.columns
    assert "doc_id" in df.columns
    assert "met_task_name" in df.columns


def test_clean_cached_results(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    df1 = pd.DataFrame(
        {
            "file_prefix": ["run1"],
            "doc_id": [1],
            "metric": [0.1],
            "some_path": ["/tmp/a"],
        }
    )
    df2 = pd.DataFrame(
        {
            "file_prefix": ["run2"],
            "doc_id": [2],
            "metric": [0.2],
            "some_path": ["/tmp/b"],
        }
    )
    df1.to_parquet(cache_dir / "one.parquet")
    df2.to_parquet(cache_dir / "two.parquet")

    conn = duckdb.connect()
    try:
        output_path = clean_cached_results(conn, cache_dir)
    finally:
        conn.close()

    assert output_path.exists()
    result = pd.read_parquet(output_path)
    assert "file_prefix" not in result.columns
    assert "some_path" not in result.columns
    assert set(result["doc_id"]) == {1, 2}

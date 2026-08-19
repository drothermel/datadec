from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import orjson
import pandas as pd
import pytest

from datadec.config import load_olmes_contract
from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed
from datadec.data.paths import DataDecidePaths
from datadec.data.preprocess.olmes_details import (
    _assert_detailed_tasks_schema_parity,
    preprocess_olmes_details,
    stream_detail_task_rows,
)

CONTRACT = load_olmes_contract()
OUTPUT_COLUMNS = tuple(column.name for column in CONTRACT.tables.detailed_tasks.columns)
DETAILED_TASK_METRICS = CONTRACT.metrics.detailed_tasks

RECIPE = "dolma1.7-no-math-no-code"
PARAMS = "150M"
SEED_VALUE = 14
STEP = 1250
TASK = "arc_challenge"


def _metrics_payload(*, task: str = TASK, num_instances: int = 2) -> dict[str, object]:
    return {
        "task_name": task,
        "task_hash": "da4d61b1b678cfae04369e8a9c4bed3a",
        "model_hash": "340b80e23a591f476ccad7b6073239ac",
        "model_config": {
            "model": "embedded-name-should-not-be-identity",
            "revision": "step1250-unsharded-hf",
            "trust_remote_code": None,
            "max_length": 2048,
            "model_type": "hf",
        },
        "task_config": {
            "task_name": task,
            "task_core": task,
            "limit": 10000000000000000000,
            "split": "test",
            "num_shots": 5,
            "fewshot_seed": 1234,
            "primary_metric": "acc_uncond",
            "random_subsample_seed": 1234,
            "context_kwargs": None,
            "generation_kwargs": None,
            "metric_kwargs": {"uncond_docid_offset": 1000000},
            "native_id_field": "id",
            "fewshot_source": "OLMES:ARC-Challenge",
            "dataset_path": "ai2_arc",
            "dataset_name": "ARC-Challenge",
            "use_chat_format": None,
            "version": 0,
            "revision": None,
            "metadata": {"regimes": ["OLMES-v0.1"], "alias": f"{task}:rc::olmes"},
        },
        "compute_config": {"batch_size": "4", "max_batch_size": 32},
        "processing_time": 41.965757846832275,
        "current_date": "2024-12-19 12:16:07 UTC",
        "num_instances": num_instances,
        "metrics": {
            "acc_raw": 0.18088737201365188,
            "acc_per_token": 0.22013651877133106,
            "acc_per_char": 0.2158703071672355,
            "acc_uncond": 0.25853242320819114,
            "primary_score": 0.25853242320819114,
        },
        "task_idx": 0,
    }


def _prediction_lines(count: int) -> bytes:
    line = orjson.dumps({"doc_id": 0, "native_id": "example", "metrics": {}})
    return (line + b"\n") * count


def _write_checkpoint_tar(
    path: Path,
    *,
    step: int = STEP,
    task: str = TASK,
    num_instances: int = 2,
    prediction_lines: int | None = None,
    metrics_payload: dict[str, object] | None = None,
    include_predictions: bool = True,
) -> None:
    payload = metrics_payload or _metrics_payload(
        task=task, num_instances=num_instances
    )
    line_count = num_instances if prediction_lines is None else prediction_lines
    with tarfile.open(path, mode="w:gz") as archive:
        metrics_bytes = orjson.dumps(payload)
        metrics_info = tarfile.TarInfo(name=f"step-{step}/{task}-metrics.json")
        metrics_info.size = len(metrics_bytes)
        archive.addfile(metrics_info, io.BytesIO(metrics_bytes))
        if include_predictions:
            predictions_bytes = _prediction_lines(line_count)
            predictions_info = tarfile.TarInfo(
                name=f"step-{step}/{task}-predictions.jsonl"
            )
            predictions_info.size = len(predictions_bytes)
            archive.addfile(predictions_info, io.BytesIO(predictions_bytes))


def _write_recipe_tar(
    path: Path,
    *,
    recipe: str = RECIPE,
    params: str = PARAMS,
    seed_value: int = SEED_VALUE,
    step: int = STEP,
    checkpoint_tar_path: Path,
) -> None:
    member_name = f"{recipe}/{params}/seed-{seed_value}/step-{step}.tar.gz"
    checkpoint_bytes = checkpoint_tar_path.read_bytes()
    with tarfile.open(path, mode="w:gz") as archive:
        member = tarfile.TarInfo(name=member_name)
        member.size = len(checkpoint_bytes)
        archive.addfile(member, io.BytesIO(checkpoint_bytes))


def _build_fixture_archive(
    tmp_path: Path,
    *,
    recipe: str = RECIPE,
    params: str = PARAMS,
    seed_value: int = SEED_VALUE,
    step: int = STEP,
    task: str = TASK,
    num_instances: int = 2,
    prediction_lines: int | None = None,
    metrics_payload: dict[str, object] | None = None,
    include_predictions: bool = True,
) -> Path:
    checkpoint_tar = tmp_path / "checkpoint.tar.gz"
    recipe_tar = tmp_path / f"{recipe}.tar.gz"
    _write_checkpoint_tar(
        checkpoint_tar,
        step=step,
        task=task,
        num_instances=num_instances,
        prediction_lines=prediction_lines,
        metrics_payload=metrics_payload,
        include_predictions=include_predictions,
    )
    _write_recipe_tar(
        recipe_tar,
        recipe=recipe,
        params=params,
        seed_value=seed_value,
        step=step,
        checkpoint_tar_path=checkpoint_tar,
    )
    return recipe_tar


def test_exact_output_schema_mapping_types_and_path_identity(tmp_path: Path) -> None:
    archive = _build_fixture_archive(tmp_path)
    rows, checkpoint_count = stream_detail_task_rows(archive, RECIPE, contract=CONTRACT)
    output = pd.DataFrame(rows)[list(OUTPUT_COLUMNS)]

    assert checkpoint_count == 1
    assert tuple(output.columns) == OUTPUT_COLUMNS
    assert output.loc[
        0, ["recipe", "params", "seed_value", "seed", "step", "task"]
    ].tolist() == [
        RECIPE,
        ModelSizeName.M150.value,
        SEED_VALUE,
        Seed.SMALL_AUX_2.value,
        STEP,
        TASK,
    ]
    assert output.loc[0, "data"] == DataRecipeName.DOLMA17_NO_MATH_CODE.value
    assert output.loc[0, "primary_metric"] == "acc_uncond"
    assert output.loc[0, "primary_score"] == 0.25853242320819114
    assert output.loc[0, "acc_uncond"] == 0.25853242320819114
    assert "embedded-name-should-not-be-identity" in output.loc[0, "model_config"]


def test_config_json_is_canonical_sorted_string(tmp_path: Path) -> None:
    archive = _build_fixture_archive(tmp_path)
    rows, _ = stream_detail_task_rows(archive, RECIPE, contract=CONTRACT)
    model_config = orjson.loads(str(rows[0]["model_config"]))
    expected = orjson.loads(
        orjson.dumps(model_config, option=orjson.OPT_SORT_KEYS).decode()
    )
    assert model_config == expected
    assert (
        str(rows[0]["model_config"])
        == orjson.dumps(model_config, option=orjson.OPT_SORT_KEYS).decode()
    )


def test_duplicate_primary_key_is_rejected(tmp_path: Path) -> None:
    checkpoint_tar = tmp_path / "checkpoint.tar.gz"
    _write_checkpoint_tar(checkpoint_tar)
    checkpoint_bytes = checkpoint_tar.read_bytes()
    recipe_tar = tmp_path / f"{RECIPE}.tar.gz"
    member_name = f"{RECIPE}/{PARAMS}/seed-{SEED_VALUE}/step-{STEP}.tar.gz"
    with tarfile.open(recipe_tar, mode="w:gz") as archive:
        for _ in range(2):
            member = tarfile.TarInfo(name=member_name)
            member.size = len(checkpoint_bytes)
            archive.addfile(member, io.BytesIO(checkpoint_bytes))

    with pytest.raises(ValueError, match="duplicate OLMES detail row"):
        stream_detail_task_rows(recipe_tar, RECIPE, contract=CONTRACT)


def test_missing_predictions_are_rejected(tmp_path: Path) -> None:
    archive = _build_fixture_archive(tmp_path, include_predictions=False)
    with pytest.raises(ValueError, match="unpaired OLMES detail task files"):
        stream_detail_task_rows(archive, RECIPE, contract=CONTRACT)


def test_instance_count_mismatch_is_rejected(tmp_path: Path) -> None:
    archive = _build_fixture_archive(tmp_path, num_instances=3, prediction_lines=2)
    with pytest.raises(ValueError, match="instance count mismatch"):
        stream_detail_task_rows(archive, RECIPE, contract=CONTRACT)


def test_task_name_mismatch_is_rejected(tmp_path: Path) -> None:
    payload = _metrics_payload()
    payload["task_name"] = "boolq"
    archive = _build_fixture_archive(tmp_path, metrics_payload=payload)
    with pytest.raises(ValueError, match="task identity mismatch"):
        stream_detail_task_rows(archive, RECIPE, contract=CONTRACT)


def test_checkpoint_not_in_aggregate_still_succeeds(tmp_path: Path) -> None:
    paths = DataDecidePaths(tmp_path)
    aggregate_path = paths.get_path("olmes_processed")
    aggregate_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "params": "4M",
                "data": "C4",
                "seed": "default",
                "step": 100,
                "task": "boolq",
                "chinchilla": "1x",
                "tokens": 1000,
                "compute": 1.5,
                **{metric: 0.1 for metric in CONTRACT.metrics.aggregate},
            }
        ]
    ).to_parquet(aggregate_path, index=False)

    archive = _build_fixture_archive(tmp_path)
    archive_dir = tmp_path / "raw/olmes-details/models"
    archive_dir.mkdir(parents=True)
    destination = archive_dir / f"{RECIPE}.tar.gz"
    destination.write_bytes(archive.read_bytes())

    result = preprocess_olmes_details(paths, RECIPE)
    assert result.row_count == 1
    assert result.checkpoint_count == 1


def test_preprocess_writes_typed_output(tmp_path: Path) -> None:
    paths = DataDecidePaths(tmp_path)
    archive = _build_fixture_archive(tmp_path)
    archive_dir = tmp_path / "raw/olmes-details/models"
    archive_dir.mkdir(parents=True)
    destination = archive_dir / f"{RECIPE}.tar.gz"
    destination.write_bytes(archive.read_bytes())

    result = preprocess_olmes_details(paths, RECIPE)
    output = pd.read_parquet(result.output_path)

    assert result.output_path == paths.olmes_details_tasks_path(RECIPE)
    assert result.row_count == 1
    assert tuple(output.columns) == OUTPUT_COLUMNS
    assert output.dtypes.to_dict() == {
        "recipe": pd.StringDtype(),
        "data": pd.StringDtype(),
        "params": pd.StringDtype(),
        "seed_value": np.dtype("int64"),
        "seed": pd.StringDtype(),
        "step": np.dtype("int64"),
        "task": pd.StringDtype(),
        "task_hash": pd.StringDtype(),
        "model_hash": pd.StringDtype(),
        "model_config": pd.StringDtype(),
        "task_config": pd.StringDtype(),
        "compute_config": pd.StringDtype(),
        "processing_time": np.dtype("float64"),
        "current_date": pd.StringDtype(),
        "num_instances": np.dtype("int64"),
        "task_idx": np.dtype("int64"),
        "primary_metric": pd.StringDtype(),
        **{field: np.dtype("float64") for field in DETAILED_TASK_METRICS},
    }


def test_preprocess_does_not_download_or_upload(tmp_path: Path) -> None:
    paths = DataDecidePaths(tmp_path)
    archive = _build_fixture_archive(tmp_path)
    archive_dir = tmp_path / "raw/olmes-details/models"
    archive_dir.mkdir(parents=True)
    destination = archive_dir / f"{RECIPE}.tar.gz"
    destination.write_bytes(archive.read_bytes())

    with (
        patch("datadec.data.download.download_sources") as download_sources,
        patch("datadec.data.pipeline.download_sources") as pipeline_download_sources,
    ):
        preprocess_olmes_details(paths, RECIPE)

    download_sources.assert_not_called()
    pipeline_download_sources.assert_not_called()
    assert paths.olmes_details_tasks_path(RECIPE).is_file()


def test_schema_drift_guard_rejects_mismatched_metric_columns() -> None:
    columns = list(CONTRACT.tables.detailed_tasks.columns)
    swapped = (*columns[:-2], columns[-1], columns[-2])
    broken_table = CONTRACT.tables.detailed_tasks.model_copy(
        update={"columns": swapped}
    )
    broken = CONTRACT.model_copy(
        update={
            "tables": CONTRACT.tables.model_copy(
                update={"detailed_tasks": broken_table}
            )
        }
    )

    with pytest.raises(AssertionError, match="OLMES detail task metric columns drift"):
        _assert_detailed_tasks_schema_parity(broken)


@pytest.mark.integration
def test_live_archive_smoke() -> None:
    repo_data = Path(__file__).resolve().parents[1] / "data"
    archive = repo_data / "raw/olmes-details/models/dolma1.7-no-math-no-code.tar.gz"
    if not archive.is_file():
        pytest.skip("live OLMES detail archive not downloaded")

    paths = DataDecidePaths(repo_data)
    result = preprocess_olmes_details(paths, RECIPE)
    output = pd.read_parquet(result.output_path)

    assert result.row_count > 0
    assert result.checkpoint_count > 0
    assert tuple(output.columns) == OUTPUT_COLUMNS
    assert (
        output[["recipe", "params", "seed_value", "step", "task"]].duplicated().sum()
        == 0
    )
    assert output["seed_value"].notna().all()
    assert output["seed"].notna().all()

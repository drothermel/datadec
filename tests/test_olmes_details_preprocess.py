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
    _assert_detailed_choices_schema_parity,
    _assert_detailed_instances_schema_parity,
    _assert_detailed_tasks_schema_parity,
    preprocess_olmes_details,
    stream_detail_rows,
    stream_detail_task_rows,
)
from datadec.data.preprocess.olmes_verify import verify_detail_counts

CONTRACT = load_olmes_contract()
OUTPUT_COLUMNS = tuple(column.name for column in CONTRACT.tables.detailed_tasks.columns)
INSTANCE_COLUMNS = tuple(
    column.name for column in CONTRACT.tables.detailed_instances.columns
)
CHOICE_COLUMNS = tuple(
    column.name for column in CONTRACT.tables.detailed_choices.columns
)
DETAILED_TASK_METRICS = CONTRACT.metrics.detailed_tasks
DETAILED_INSTANCE_METRICS = CONTRACT.metrics.detailed_instances
DETAILED_CHOICE_METRICS = CONTRACT.metrics.detailed_choices

RECIPE = "dolma1.7-no-math-no-code"
PARAMS = "150M"
SEED_VALUE = 14
STEP = 1250
TASK = "arc_challenge"
TASK_HASH = "da4d61b1b678cfae04369e8a9c4bed3a"
MODEL_HASH = "340b80e23a591f476ccad7b6073239ac"


def _metrics_payload(*, task: str = TASK, num_instances: int = 2) -> dict[str, object]:
    return {
        "task_name": task,
        "task_hash": TASK_HASH,
        "model_hash": MODEL_HASH,
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


def _choice_metrics(
    *, include_byte: bool = True, include_uncond: bool = True
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "sum_logits": -66.26013946533203,
        "num_tokens": 6,
        "num_tokens_all": 201,
        "is_greedy": False,
        "logits_per_token": -11.043356577555338,
        "logits_per_char": -2.0078830141009707,
        "num_chars": 33,
    }
    if include_uncond:
        metrics["sum_logits_uncond"] = -66.03694152832031
    if include_byte:
        metrics["logits_per_byte"] = 2.896762867130736
    return metrics


def _instance_metrics(
    *, include_byte: bool = False, include_uncond: bool = True
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "predicted_index_raw": 1,
        "predicted_index_per_token": 1,
        "predicted_index_per_char": 1,
        "correct_choice": 2,
        "acc_raw": 0,
        "acc_per_token": 0,
        "acc_per_char": 0,
    }
    if include_uncond:
        metrics["predicted_index_uncond"] = 1
        metrics["acc_uncond"] = 0
    if include_byte:
        metrics["predicted_index_per_byte"] = 1
        metrics["acc_per_byte"] = 0
    return metrics


def _prediction_record(
    doc_id: int,
    *,
    native_id: object = "example",
    include_byte: bool = False,
    include_uncond: bool = True,
    num_choices: int = 4,
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "label": 2,
        "native_id": native_id,
        "task_hash": TASK_HASH,
        "model_hash": MODEL_HASH,
        "metrics": _instance_metrics(
            include_byte=include_byte, include_uncond=include_uncond
        ),
        "model_output": [
            _choice_metrics(include_byte=include_byte, include_uncond=include_uncond)
            for _ in range(num_choices)
        ],
    }


def _prediction_lines(
    count: int,
    *,
    include_byte: bool = False,
    native_ids: list[object] | None = None,
) -> bytes:
    lines: list[bytes] = []
    for index in range(count):
        native_id = (
            native_ids[index]
            if native_ids is not None
            else ("example" if index == 0 else index)
        )
        record = _prediction_record(
            index,
            native_id=native_id,
            include_byte=include_byte,
        )
        lines.append(orjson.dumps(record))
    return b"\n".join(lines) + b"\n"


def _write_checkpoint_tar(
    path: Path,
    *,
    step: int = STEP,
    task: str = TASK,
    num_instances: int = 2,
    prediction_lines: int | None = None,
    metrics_payload: dict[str, object] | None = None,
    include_predictions: bool = True,
    include_byte: bool = False,
    native_ids: list[object] | None = None,
    custom_predictions: bytes | None = None,
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
            predictions_bytes = custom_predictions or _prediction_lines(
                line_count,
                include_byte=include_byte,
                native_ids=native_ids,
            )
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
    include_byte: bool = False,
    native_ids: list[object] | None = None,
    custom_predictions: bytes | None = None,
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
        include_byte=include_byte,
        native_ids=native_ids,
        custom_predictions=custom_predictions,
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


def _install_fixture_archive(tmp_path: Path, archive: Path) -> None:
    archive_dir = tmp_path / "raw/olmes-details/models"
    archive_dir.mkdir(parents=True)
    destination = archive_dir / f"{RECIPE}.tar.gz"
    destination.write_bytes(archive.read_bytes())


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


def test_instance_and_choice_schema_types_and_nullability(tmp_path: Path) -> None:
    archive = _build_fixture_archive(
        tmp_path,
        native_ids=["example", 7],
        include_byte=False,
    )
    _, instances, choices, _ = stream_detail_rows(archive, RECIPE, contract=CONTRACT)

    instance_df = pd.DataFrame(instances)[list(INSTANCE_COLUMNS)]
    choice_df = pd.DataFrame(choices)[list(CHOICE_COLUMNS)]

    assert tuple(instance_df.columns) == INSTANCE_COLUMNS
    assert tuple(choice_df.columns) == CHOICE_COLUMNS
    assert instance_df.loc[0, "native_id_kind"] == "string"
    assert instance_df.loc[1, "native_id_kind"] == "integer"
    assert instance_df.loc[1, "native_id"] == "7"
    assert pd.isna(instance_df.loc[0, "predicted_index_per_byte"])
    assert pd.isna(instance_df.loc[0, "acc_per_byte"])
    assert pd.isna(choice_df.loc[0, "logits_per_byte"])
    assert choice_df["is_greedy"].dtype == np.dtype("bool")


def test_null_native_id_kind(tmp_path: Path) -> None:
    archive = _build_fixture_archive(
        tmp_path,
        num_instances=1,
        native_ids=[None],
    )
    _, instances, _, _ = stream_detail_rows(archive, RECIPE, contract=CONTRACT)
    assert instances[0]["native_id"] is None
    assert instances[0]["native_id_kind"] == "null"


def test_absent_byte_fields_are_nullable(tmp_path: Path) -> None:
    archive = _build_fixture_archive(tmp_path, include_byte=False, num_instances=1)
    _, instances, choices, _ = stream_detail_rows(archive, RECIPE, contract=CONTRACT)
    assert instances[0]["predicted_index_per_byte"] is None
    assert instances[0]["acc_per_byte"] is None
    assert choices[0]["logits_per_byte"] is None


def test_present_byte_fields_are_populated(tmp_path: Path) -> None:
    archive = _build_fixture_archive(tmp_path, include_byte=True, num_instances=1)
    _, instances, choices, _ = stream_detail_rows(archive, RECIPE, contract=CONTRACT)
    assert instances[0]["predicted_index_per_byte"] == 1
    assert instances[0]["acc_per_byte"] == 0
    assert choices[0]["logits_per_byte"] == pytest.approx(2.896762867130736)


def test_choice_explosion_uses_zero_based_indices(tmp_path: Path) -> None:
    archive = _build_fixture_archive(tmp_path, num_instances=1)
    _, _, choices, _ = stream_detail_rows(archive, RECIPE, contract=CONTRACT)
    assert [row["choice_index"] for row in choices] == [0, 1, 2, 3]


def test_output_rows_are_sorted_by_contract_sort_keys(tmp_path: Path) -> None:
    archive = _build_fixture_archive(tmp_path, num_instances=2)
    tasks, instances, choices, _ = stream_detail_rows(
        archive, RECIPE, contract=CONTRACT
    )
    assert [row["doc_id"] for row in instances] == [0, 1]
    assert tasks == sorted(
        tasks,
        key=lambda row: tuple(
            row[name] for name in CONTRACT.tables.detailed_tasks.sort_key
        ),
    )
    assert instances == sorted(
        instances,
        key=lambda row: tuple(
            row[name] for name in CONTRACT.tables.detailed_instances.sort_key
        ),
    )
    assert choices == sorted(
        choices,
        key=lambda row: tuple(
            row[name] for name in CONTRACT.tables.detailed_choices.sort_key
        ),
    )


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


def test_duplicate_task_primary_key_is_rejected(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="duplicate OLMES detail task row"):
        stream_detail_task_rows(recipe_tar, RECIPE, contract=CONTRACT)


def test_duplicate_instance_primary_key_is_rejected(tmp_path: Path) -> None:
    duplicate_line = orjson.dumps(_prediction_record(0)) + b"\n"
    custom_predictions = duplicate_line * 2
    archive = _build_fixture_archive(
        tmp_path,
        num_instances=2,
        custom_predictions=custom_predictions,
    )
    with pytest.raises(ValueError, match="duplicate OLMES detail instance row"):
        stream_detail_rows(archive, RECIPE, contract=CONTRACT)


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
    _install_fixture_archive(tmp_path, archive)

    result = preprocess_olmes_details(paths, RECIPE)
    assert result.row_count == 1
    assert result.instance_count == 2
    assert result.choice_count == 8
    assert result.checkpoint_count == 1


def test_preprocess_writes_typed_output(tmp_path: Path) -> None:
    paths = DataDecidePaths(tmp_path)
    archive = _build_fixture_archive(tmp_path)
    _install_fixture_archive(tmp_path, archive)

    result = preprocess_olmes_details(paths, RECIPE)
    tasks = pd.read_parquet(result.output_tasks_path)
    instances = pd.read_parquet(result.output_instances_path)
    choices = pd.read_parquet(result.output_choices_path)

    assert result.output_path == paths.olmes_details_tasks_path(RECIPE)
    assert result.output_instances_path == paths.olmes_details_instances_path(RECIPE)
    assert result.output_choices_path == paths.olmes_details_choices_path(RECIPE)
    assert result.row_count == 1
    assert result.instance_count == 2
    assert result.choice_count == 8
    assert tuple(tasks.columns) == OUTPUT_COLUMNS
    assert tuple(instances.columns) == INSTANCE_COLUMNS
    assert tuple(choices.columns) == CHOICE_COLUMNS
    verify_detail_counts(
        tasks_df=tasks,
        instances_df=instances,
        choices_df=choices,
    )
    assert tasks.dtypes.to_dict() == {
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
    _install_fixture_archive(tmp_path, archive)

    with (
        patch("datadec.data.download.download_sources") as download_sources,
        patch("datadec.data.pipeline.download_sources") as pipeline_download_sources,
    ):
        preprocess_olmes_details(paths, RECIPE)

    download_sources.assert_not_called()
    pipeline_download_sources.assert_not_called()
    assert paths.olmes_details_tasks_path(RECIPE).is_file()
    assert paths.olmes_details_instances_path(RECIPE).is_file()
    assert paths.olmes_details_choices_path(RECIPE).is_file()


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


def test_instance_schema_drift_guard_rejects_mismatched_metric_columns() -> None:
    columns = list(CONTRACT.tables.detailed_instances.columns)
    swapped = (*columns[:-2], columns[-1], columns[-2])
    broken_table = CONTRACT.tables.detailed_instances.model_copy(
        update={"columns": swapped}
    )
    broken = CONTRACT.model_copy(
        update={
            "tables": CONTRACT.tables.model_copy(
                update={"detailed_instances": broken_table}
            )
        }
    )

    with pytest.raises(
        AssertionError, match="OLMES detail instance metric columns drift"
    ):
        _assert_detailed_instances_schema_parity(broken)


def test_choice_schema_drift_guard_rejects_mismatched_metric_columns() -> None:
    columns = list(CONTRACT.tables.detailed_choices.columns)
    swapped = (*columns[:-2], columns[-1], columns[-2])
    broken_table = CONTRACT.tables.detailed_choices.model_copy(
        update={"columns": swapped}
    )
    broken = CONTRACT.model_copy(
        update={
            "tables": CONTRACT.tables.model_copy(
                update={"detailed_choices": broken_table}
            )
        }
    )

    with pytest.raises(
        AssertionError, match="OLMES detail choice metric columns drift"
    ):
        _assert_detailed_choices_schema_parity(broken)

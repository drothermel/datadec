from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from datadec.config import load_olmes_contract
from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed
from datadec.data.paths import DataDecidePaths
from datadec.data.preprocess.olmes import (
    _assert_output_schema_parity,
    _source_metric_columns,
    flatten_olmes_rows,
    group_olmes_rows,
    preprocess_olmes,
)

CONTRACT = load_olmes_contract()
OUTPUT_COLUMNS = tuple(column.name for column in CONTRACT.tables.aggregate.columns)
SOURCE_METRIC_COLUMNS = _source_metric_columns(CONTRACT)
AGGREGATE_METRICS = CONTRACT.metrics.aggregate


def _raw_record(
    *,
    params: object = "4M",
    data: object = "C4",
    seed: object = "default",
    step: object = 100,
    task: object = "arc_challenge",
    chinchilla: object = "1x",
    tokens: object = 1000,
    compute: object = 1.5,
    metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "params": params,
        "data": data,
        "seed": seed,
        "step": step,
        "task": task,
        "chinchilla": chinchilla,
        "tokens": tokens,
        "compute": compute,
        "metrics": metrics
        or {
            "acc_uncond": 0.42,
            "acc_raw": 0.31,
            "bits_per_byte_corr": 0.1,
            "logits_per_byte_corr": 0.2,
            "predicted_index_raw": 3,
        },
    }


def test_exact_output_schema_mapping_types_and_enum_values() -> None:
    record = _raw_record(step=1250.0)
    output = flatten_olmes_rows(
        group_olmes_rows(pd.DataFrame([record]), contract=CONTRACT),
        contract=CONTRACT,
    )

    assert tuple(output.columns) == OUTPUT_COLUMNS
    assert output.loc[0, ["params", "data", "seed", "task", "chinchilla"]].tolist() == [
        ModelSizeName.M4.value,
        DataRecipeName.C4.value,
        Seed.DEFAULT.value,
        "arc_challenge",
        "1x",
    ]
    assert output.loc[0, "step"] == 1250
    assert output.loc[0, "tokens"] == 1000
    assert output.loc[0, "compute"] == 1.5
    assert output.loc[0, "acc_uncond"] == 0.42
    assert output.loc[0, "primary_metric"] == 0.42
    assert output.loc[0, "bits_per_byte_corr"] == 0.1
    assert output.loc[0, "logits_per_byte_corr"] == 0.2
    assert "predicted_index_raw" not in output.columns
    assert output.dtypes.to_dict() == {
        "params": pd.StringDtype(),
        "data": pd.StringDtype(),
        "seed": pd.StringDtype(),
        "step": np.dtype("int64"),
        "task": pd.StringDtype(),
        "chinchilla": pd.StringDtype(),
        "tokens": np.dtype("int64"),
        "compute": np.dtype("float64"),
        **{field: np.dtype("float64") for field in AGGREGATE_METRICS},
    }


def test_primary_metric_policy_for_mmlu_subjects() -> None:
    record = _raw_record(
        task="mmlu_college_biology",
        metrics={"acc_raw": 0.55, "acc_uncond": 0.44},
    )
    output = flatten_olmes_rows(
        group_olmes_rows(pd.DataFrame([record]), contract=CONTRACT),
        contract=CONTRACT,
    )

    assert output.loc[0, "acc_raw"] == 0.55
    assert output.loc[0, "primary_metric"] == 0.55


def test_prediction_index_fields_are_excluded_from_output() -> None:
    record = _raw_record(
        metrics={
            "acc_uncond": 0.5,
            "predicted_index_raw": 1,
            "predicted_index_per_token": 2,
            "predicted_index_per_char": 3,
            "predicted_index_per_byte": 4,
            "predicted_index_uncond": 5,
        }
    )
    output = flatten_olmes_rows(
        group_olmes_rows(pd.DataFrame([record]), contract=CONTRACT),
        contract=CONTRACT,
    )

    assert set(output.columns).isdisjoint(
        {
            "predicted_index_raw",
            "predicted_index_per_token",
            "predicted_index_per_char",
            "predicted_index_per_byte",
            "predicted_index_uncond",
        }
    )


def test_shuffled_input_flattens_in_deterministic_sort_order() -> None:
    records = [
        _raw_record(params="4M", data="C4", seed="default", step=200, task="boolq"),
        _raw_record(params="10M", data="C4", seed="small aux 2", step=300, task="csqa"),
        _raw_record(params="4M", data="C4", seed="default", step=100, task="arc_easy"),
        _raw_record(params="4M", data="Dolma1.7", seed="default", step=50, task="piqa"),
    ]
    original = flatten_olmes_rows(
        group_olmes_rows(pd.DataFrame(records), contract=CONTRACT),
        contract=CONTRACT,
    )
    shuffled = flatten_olmes_rows(
        group_olmes_rows(
            pd.DataFrame(records)
            .sample(frac=1, random_state=17)
            .reset_index(drop=True),
            contract=CONTRACT,
        ),
        contract=CONTRACT,
    )

    pd.testing.assert_frame_equal(shuffled, original)
    assert list(
        original.loc[:, ["params", "data", "seed", "step", "task"]].itertuples(
            index=False, name=None
        )
    ) == [
        ("10M", "C4", "small aux 2", 300, "csqa"),
        ("4M", "C4", "default", 100, "arc_easy"),
        ("4M", "C4", "default", 200, "boolq"),
        ("4M", "Dolma1.7", "default", 50, "piqa"),
    ]


@pytest.mark.parametrize(
    "step",
    [None, np.nan, np.inf, -np.inf, True, "malformed", 1.5],
    ids=["null", "nan", "positive-inf", "negative-inf", "bool", "text", "fraction"],
)
def test_invalid_steps_are_rejected_with_row_and_value_context(step: object) -> None:
    with pytest.raises(ValueError) as exc_info:
        group_olmes_rows(pd.DataFrame([_raw_record(step=step)]), contract=CONTRACT)

    message = str(exc_info.value)
    assert "invalid PPL step at row 0" in message
    assert repr(step) in message


def test_duplicate_primary_key_is_rejected() -> None:
    olmes_df = pd.DataFrame(
        [
            _raw_record(step=1250, task="boolq"),
            _raw_record(step=1250.0, task="boolq"),
        ]
    )

    with pytest.raises(ValueError, match="duplicate OLMES row at row 1") as exc:
        group_olmes_rows(olmes_df, contract=CONTRACT)

    assert "params='4M', data='C4', seed='default', step=1250, task='boolq'" in str(
        exc.value
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("params", "5M"),
        ("data", "unknown recipe"),
        ("seed", "seed 7"),
    ],
)
def test_unknown_enum_values_use_current_enum_validation(
    field: str, value: str
) -> None:
    record = _raw_record()
    record[field] = value

    with pytest.raises(ValueError, match=value):
        group_olmes_rows(pd.DataFrame([record]), contract=CONTRACT)


def test_empty_input_writes_exact_typed_schema(tmp_path: Path) -> None:
    paths = DataDecidePaths(tmp_path)
    input_path = paths.get_path("dwn_raw")
    input_path.parent.mkdir(parents=True)
    pd.DataFrame().to_parquet(input_path, index=False)

    result = preprocess_olmes(paths)
    output = pd.read_parquet(paths.get_path("olmes_processed"))

    assert result.row_count == 0
    assert result.training_run_count == 0
    assert tuple(output.columns) == OUTPUT_COLUMNS
    assert output.dtypes.to_dict() == {
        "params": pd.StringDtype(),
        "data": pd.StringDtype(),
        "seed": pd.StringDtype(),
        "step": np.dtype("int64"),
        "task": pd.StringDtype(),
        "chinchilla": pd.StringDtype(),
        "tokens": np.dtype("int64"),
        "compute": np.dtype("float64"),
        **{field: np.dtype("float64") for field in AGGREGATE_METRICS},
    }


def test_schema_drift_guard_rejects_mismatched_identity_columns() -> None:
    columns = list(CONTRACT.tables.aggregate.columns)
    swapped = (columns[1], columns[0], *columns[2:])
    broken_table = CONTRACT.tables.aggregate.model_copy(update={"columns": swapped})
    broken = CONTRACT.model_copy(
        update={
            "tables": CONTRACT.tables.model_copy(update={"aggregate": broken_table})
        }
    )

    with pytest.raises(AssertionError, match="OLMES identity columns drift"):
        _assert_output_schema_parity(broken)


def test_preprocess_does_not_download_or_upload(tmp_path: Path) -> None:
    paths = DataDecidePaths(tmp_path)
    input_path = paths.get_path("dwn_raw")
    input_path.parent.mkdir(parents=True)
    pd.DataFrame([_raw_record()]).to_parquet(input_path, index=False)

    with (
        patch("datadec.data.download.download_sources") as download_sources,
        patch("datadec.data.pipeline.download_sources") as pipeline_download_sources,
    ):
        preprocess_olmes(paths)

    download_sources.assert_not_called()
    pipeline_download_sources.assert_not_called()
    assert paths.get_path("olmes_processed").is_file()


def test_source_metric_columns_exclude_only_primary_metric() -> None:
    assert SOURCE_METRIC_COLUMNS == tuple(
        name for name in AGGREGATE_METRICS if name != "primary_metric"
    )

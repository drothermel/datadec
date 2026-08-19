from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import orjson
import pandas as pd
from dr_ds import coerce_float

from datadec.config import OLMESContract, load_olmes_contract
from datadec.data import constants as consts
from datadec.data.paths import DataDecidePaths
from datadec.data.preprocess.ppl import _normalize_step

if TYPE_CHECKING:
    from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed

_EXCLUDED_METRIC_KEYS: frozenset[str] = frozenset(
    name for name in consts.DROP_METRICS if name.startswith("predicted_index_")
)

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

OlmesRunKey: TypeAlias = tuple["ModelSizeName", "DataRecipeName", "Seed"]
OlmesRowsByKey: TypeAlias = dict[
    OlmesRunKey,
    dict[int, dict[str, "OlmesAggregateRow"]],
]


@dataclass(frozen=True, slots=True)
class OlmesAggregateRow:
    params: ModelSizeName
    data: DataRecipeName
    seed: Seed
    step: int
    task: str
    chinchilla: str
    tokens: int
    compute: float
    metrics: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class OlmesPreprocessResult:
    input_path: Path
    output_path: Path
    row_count: int
    training_run_count: int


def _output_columns(contract: OLMESContract) -> tuple[str, ...]:
    return tuple(column.name for column in contract.tables.aggregate.columns)


def _identity_columns(contract: OLMESContract) -> tuple[str, ...]:
    metric_count = len(contract.metrics.aggregate)
    all_columns = _output_columns(contract)
    return all_columns[: len(all_columns) - metric_count]


def _source_metric_columns(contract: OLMESContract) -> tuple[str, ...]:
    return tuple(
        name for name in contract.metrics.aggregate if name != "primary_metric"
    )


def _assert_output_schema_parity(contract: OLMESContract) -> None:
    table = contract.tables.aggregate
    expected = _output_columns(contract)
    actual = tuple(column.name for column in table.columns)
    if actual != expected:
        raise AssertionError(
            "persisted OLMES output columns drift from contract.tables.aggregate: "
            f"expected={expected!r}, actual={actual!r}"
        )

    identity = _identity_columns(contract)
    expected_identity = (
        "params",
        "data",
        "seed",
        "step",
        "task",
        "chinchilla",
        "tokens",
        "compute",
    )
    if identity != expected_identity:
        raise AssertionError(
            "OLMES identity columns drift from contract: "
            f"expected={expected_identity!r}, actual={identity!r}"
        )

    source_metrics = _source_metric_columns(contract)
    expected_source_count = len(contract.metrics.aggregate) - 1
    if len(source_metrics) != expected_source_count:
        raise AssertionError(
            "OLMES source metric columns must exclude only primary_metric: "
            f"expected {expected_source_count}, got {len(source_metrics)}"
        )


def _pandas_dtype(
    logical_type: Literal["string", "int64", "float64", "bool"],
) -> str:
    if logical_type == "string":
        return "string"
    return logical_type


def group_olmes_rows(
    olmes_df: pd.DataFrame,
    *,
    contract: OLMESContract | None = None,
) -> OlmesRowsByKey:
    from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed

    contract = contract or load_olmes_contract()
    _assert_output_schema_parity(contract)
    source_metric_columns = _source_metric_columns(contract)

    grouped: OlmesRowsByKey = defaultdict(lambda: defaultdict(dict))
    seen_keys: set[tuple[str, str, str, int, str]] = set()

    for row_index, record in enumerate(olmes_df.to_dict(orient="records")):
        run_key = (
            ModelSizeName(record["params"]),
            DataRecipeName(record["data"]),
            Seed(record["seed"]),
        )
        step = _normalize_step(record["step"], row_index=row_index)
        task = _require_string(record.get("task"), row_index=row_index, field="task")
        chinchilla = _require_string(
            record.get("chinchilla"), row_index=row_index, field="chinchilla"
        )
        tokens = _normalize_int64(
            record.get("tokens"), row_index=row_index, field="tokens"
        )
        compute = _normalize_float64(
            record.get("compute"), row_index=row_index, field="compute"
        )

        primary_key = (
            run_key[0].value,
            run_key[1].value,
            run_key[2].value,
            step,
            task,
        )
        if primary_key in seen_keys:
            params, data, seed = run_key
            raise ValueError(
                f"duplicate OLMES row at row {row_index}: "
                f"params={params.value!r}, data={data.value!r}, "
                f"seed={seed.value!r}, step={step}, task={task!r}"
            )
        seen_keys.add(primary_key)

        metrics_payload = _parse_metrics_payload(record.get("metrics"))
        metrics = _extract_source_metrics(metrics_payload, source_metric_columns)

        row = OlmesAggregateRow(
            params=run_key[0],
            data=run_key[1],
            seed=run_key[2],
            step=step,
            task=task,
            chinchilla=chinchilla,
            tokens=tokens,
            compute=compute,
            metrics=metrics,
        )
        grouped[run_key][step][task] = row

    return dict(grouped)


def flatten_olmes_rows(
    grouped: OlmesRowsByKey,
    *,
    contract: OLMESContract | None = None,
) -> pd.DataFrame:
    contract = contract or load_olmes_contract()
    _assert_output_schema_parity(contract)
    output_columns = _output_columns(contract)
    source_metric_columns = _source_metric_columns(contract)
    policy = contract.metrics.aggregate_primary_metric.model_dump()

    rows: list[dict[str, object]] = []
    for run_key in sorted(grouped, key=lambda key: tuple(item.value for item in key)):
        params, data, seed = run_key
        for step in sorted(grouped[run_key]):
            for task in sorted(grouped[run_key][step]):
                row = grouped[run_key][step][task]
                primary_metric = _resolve_primary_metric(
                    task, row.metrics, policy=policy
                )
                rows.append(
                    {
                        "params": params.value,
                        "data": data.value,
                        "seed": seed.value,
                        "step": step,
                        "task": task,
                        "chinchilla": row.chinchilla,
                        "tokens": row.tokens,
                        "compute": row.compute,
                        **{
                            field: row.metrics.get(field)
                            for field in source_metric_columns
                        },
                        "primary_metric": primary_metric,
                    }
                )

    return _typed_output_dataframe(rows, contract=contract, columns=output_columns)


def preprocess_olmes(
    paths: DataDecidePaths,
    *,
    input_path: Path | None = None,
    output_path: Path | None = None,
    verbose: bool = False,
) -> OlmesPreprocessResult:
    contract = load_olmes_contract()
    resolved_input = input_path or paths.get_path("dwn_raw")
    resolved_output = output_path or paths.get_path("olmes_processed")

    olmes_df = pd.read_parquet(resolved_input)
    grouped = group_olmes_rows(olmes_df, contract=contract)
    output_df = flatten_olmes_rows(grouped, contract=contract)

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(resolved_output, index=False)

    result = OlmesPreprocessResult(
        input_path=resolved_input,
        output_path=resolved_output,
        row_count=len(output_df),
        training_run_count=len(grouped),
    )
    if verbose:
        print(f"olmes input: {result.input_path}")
        print(f"olmes output: {result.output_path}")
        print(f"olmes rows: {result.row_count}")
        print(f"olmes training runs: {result.training_run_count}")
    return result


def _parse_metrics_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        return orjson.loads(raw)
    text = str(raw).replace("'", '"')
    return orjson.loads(text)


def _extract_source_metrics(
    payload: dict[str, Any],
    source_metric_columns: tuple[str, ...],
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for field in source_metric_columns:
        if field in payload and field not in _EXCLUDED_METRIC_KEYS:
            metrics[field] = coerce_float(payload[field])
        else:
            metrics[field] = None
    return metrics


def _resolve_primary_metric(
    task: str,
    metrics: dict[str, float | None],
    *,
    policy: dict[str, str],
) -> float | None:
    policy_key = "mmlu" if task.startswith("mmlu_") else task
    source_metric = policy.get(policy_key)
    if source_metric is None:
        return None
    return metrics.get(source_metric)


def _require_string(value: Any, *, row_index: int, field: str) -> str:
    invalid = value is None or value is pd.NA
    if not invalid and isinstance(value, (float, np.floating)):
        invalid = bool(np.isnan(value))
    if invalid:
        raise ValueError(
            f"invalid OLMES {field} at row {row_index}: {value!r}; "
            "expected a non-null string"
        )
    return str(value)


def _normalize_int64(value: Any, *, row_index: int, field: str) -> int:
    invalid = isinstance(value, (bool, np.bool_)) or value is None or value is pd.NA
    decimal_value: Decimal | None = None
    if not invalid:
        try:
            if not isinstance(
                value,
                (str, int, float, Decimal, np.integer, np.floating),
            ):
                raise InvalidOperation
            text = str(value).strip()
            if not text:
                raise InvalidOperation
            decimal_value = Decimal(text)
            invalid = (
                not decimal_value.is_finite()
                or decimal_value != decimal_value.to_integral_value()
            )
        except (InvalidOperation, ValueError):
            invalid = True
    if invalid or decimal_value is None:
        raise ValueError(
            f"invalid OLMES {field} at row {row_index}: {value!r}; "
            "expected a finite integral int64 value"
        )

    normalized = int(decimal_value)
    if not _INT64_MIN <= normalized <= _INT64_MAX:
        raise ValueError(
            f"invalid OLMES {field} at row {row_index}: {value!r}; "
            "expected a finite integral int64 value"
        )
    return normalized


def _normalize_float64(value: Any, *, row_index: int, field: str) -> float:
    invalid = isinstance(value, (bool, np.bool_)) or value is None or value is pd.NA
    if not invalid:
        try:
            if not isinstance(
                value,
                (str, int, float, Decimal, np.integer, np.floating),
            ):
                raise InvalidOperation
            text = str(value).strip()
            if not text:
                raise InvalidOperation
            decimal_value = Decimal(text)
            invalid = not decimal_value.is_finite()
        except (InvalidOperation, ValueError):
            invalid = True
    if invalid:
        raise ValueError(
            f"invalid OLMES {field} at row {row_index}: {value!r}; "
            "expected a finite float64 value"
        )
    return float(decimal_value)


def _typed_output_dataframe(
    rows: list[dict[str, object]],
    *,
    contract: OLMESContract,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    column_types = {
        column.name: column.logical_type for column in contract.tables.aggregate.columns
    }
    values_by_column = {column: [row.get(column) for row in rows] for column in columns}
    typed_columns: dict[str, pd.Series[Any]] = {}
    for column in columns:
        logical_type = column_types[column]
        dtype = _pandas_dtype(logical_type)
        typed_columns[column] = pd.Series(values_by_column[column], dtype=dtype)
    return pd.DataFrame(typed_columns, columns=list(columns))

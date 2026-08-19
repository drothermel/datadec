from __future__ import annotations

import io
import re
import tarfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dr_ds import coerce_float

from datadec.config import OLMESContract, OLMESTableContract, load_olmes_contract
from datadec.config import load_source_manifest
from datadec.data.paths import DataDecidePaths
from datadec.data.preprocess.ppl import _normalize_step

_CHECKPOINT_MEMBER_RE = re.compile(
    r"^(?P<recipe>[^/]+)/(?P<params>[^/]+)/seed-(?P<seed_value>\d+)/step-(?P<step>\d+)\.tar\.gz$"
)
_METRICS_SUFFIX = "-metrics.json"
_PREDICTIONS_SUFFIX = "-predictions.jsonl"
_PREDICTION_REQUIRED_KEYS = (
    "doc_id",
    "label",
    "metrics",
    "model_output",
    "task_hash",
    "model_hash",
)

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


@dataclass(frozen=True, slots=True)
class OlmesDetailsPreprocessResult:
    recipe: str
    input_path: Path
    output_tasks_path: Path
    output_instances_path: Path
    output_choices_path: Path
    row_count: int
    instance_count: int
    choice_count: int
    checkpoint_count: int

    @property
    def output_path(self) -> Path:
        return self.output_tasks_path


def _output_columns(contract: OLMESContract, table_name: str) -> tuple[str, ...]:
    table = getattr(contract.tables, table_name)
    return tuple(column.name for column in table.columns)


def _metric_columns(contract: OLMESContract, table_name: str) -> tuple[str, ...]:
    return getattr(contract.metrics, table_name)


def _assert_table_schema_parity(
    contract: OLMESContract,
    *,
    table_name: str,
    label: str,
) -> None:
    table = getattr(contract.tables, table_name)
    expected = _output_columns(contract, table_name)
    actual = tuple(column.name for column in table.columns)
    if actual != expected:
        raise AssertionError(
            f"persisted OLMES {label} columns drift from contract: "
            f"expected={expected!r}, actual={actual!r}"
        )

    metric_columns = _metric_columns(contract, table_name)
    trailing = tuple(column.name for column in table.columns[-len(metric_columns) :])
    if trailing != metric_columns:
        raise AssertionError(
            f"OLMES {label} metric columns drift from contract: "
            f"expected={metric_columns!r}, actual={trailing!r}"
        )


def _assert_detailed_tasks_schema_parity(contract: OLMESContract) -> None:
    _assert_table_schema_parity(
        contract, table_name="detailed_tasks", label="detail task"
    )


def _assert_detailed_instances_schema_parity(contract: OLMESContract) -> None:
    _assert_table_schema_parity(
        contract, table_name="detailed_instances", label="detail instance"
    )


def _assert_detailed_choices_schema_parity(contract: OLMESContract) -> None:
    _assert_table_schema_parity(
        contract, table_name="detailed_choices", label="detail choice"
    )


def _assert_all_detailed_schema_parity(contract: OLMESContract) -> None:
    _assert_detailed_tasks_schema_parity(contract)
    _assert_detailed_instances_schema_parity(contract)
    _assert_detailed_choices_schema_parity(contract)


def _nullable_column_names(table: OLMESTableContract) -> frozenset[str]:
    return frozenset(column.name for column in table.columns if column.nullable)


def _pandas_dtype(
    logical_type: Literal["string", "int64", "float64", "bool"],
    *,
    nullable: bool = False,
) -> str:
    if logical_type == "string":
        return "string"
    if logical_type == "int64":
        return "Int64" if nullable else "int64"
    if logical_type == "bool":
        return "boolean"
    return logical_type


def _typed_output_dataframe(
    rows: list[dict[str, object]],
    *,
    table: OLMESTableContract,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    column_meta = {column.name: column for column in table.columns}
    values_by_column = {column: [row.get(column) for row in rows] for column in columns}
    typed_columns: dict[str, pd.Series[Any]] = {}
    for column in columns:
        meta = column_meta[column]
        dtype = _pandas_dtype(meta.logical_type, nullable=meta.nullable)
        typed_columns[column] = pd.Series(values_by_column[column], dtype=dtype)
    return pd.DataFrame(typed_columns, columns=list(columns))


def _recipe_tar_path(paths: DataDecidePaths, recipe: str) -> Path:
    manifest = load_source_manifest()
    filename = manifest.olmes_details.filename_template.format(recipe=recipe)
    return paths.data_dir / manifest.olmes_details.output_root / filename


def _canonical_json(value: object) -> str:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode()


def _require_hash(value: Any, *, context: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"invalid OLMES detail {field} in {context}: {value!r}; "
            "expected a non-empty string"
        )
    return value


def _normalize_int64_field(value: Any, *, context: str, field: str) -> int:
    invalid = isinstance(value, (bool, np.bool_)) or value is None
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
            f"invalid OLMES detail {field} in {context}: {value!r}; "
            "expected a finite integral int64 value"
        )
    normalized = int(decimal_value)
    if not _INT64_MIN <= normalized <= _INT64_MAX:
        raise ValueError(
            f"invalid OLMES detail {field} in {context}: {value!r}; "
            "expected a finite integral int64 value"
        )
    return normalized


def _normalize_float64_field(value: Any, *, context: str, field: str) -> float:
    invalid = isinstance(value, (bool, np.bool_)) or value is None
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
            invalid = not decimal_value.is_finite()
        except (InvalidOperation, ValueError):
            invalid = True
    if invalid or decimal_value is None:
        raise ValueError(
            f"invalid OLMES detail {field} in {context}: {value!r}; "
            "expected a finite float64 value"
        )
    return float(decimal_value)


def _normalize_bool_field(value: Any, *, context: str, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(
        f"invalid OLMES detail {field} in {context}: {value!r}; expected a bool value"
    )


def _normalize_native_id(
    value: Any,
    *,
    context: str,
) -> tuple[str | None, str]:
    if value is None:
        return None, "null"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"invalid OLMES detail native_id in {context}: {value!r}; "
            "expected integer, string, or null"
        )
    if isinstance(value, (int, np.integer)):
        return str(int(value)), "integer"
    if isinstance(value, str):
        return value, "string"
    raise ValueError(
        f"invalid OLMES detail native_id in {context}: {value!r}; "
        "expected integer, string, or null"
    )


def _column_logical_types(table: OLMESTableContract) -> dict[str, str]:
    return {column.name: column.logical_type for column in table.columns}


def _extract_metric_values(
    metrics_block: dict[str, Any],
    *,
    metric_names: tuple[str, ...],
    table: OLMESTableContract,
    context: str,
) -> dict[str, object]:
    nullable = _nullable_column_names(table)
    logical_types = _column_logical_types(table)
    values: dict[str, object] = {}
    for field in metric_names:
        if field not in metrics_block:
            if field in nullable:
                values[field] = None
                continue
            raise ValueError(
                f"missing OLMES detail metric {field!r} in {context}; "
                "expected a non-null value"
            )
        raw_value = metrics_block[field]
        if raw_value is None:
            if field in nullable:
                values[field] = None
                continue
            raise ValueError(
                f"invalid OLMES detail metric {field!r} in {context}: "
                "expected a non-null value"
            )
        logical_type = logical_types[field]
        if logical_type == "int64":
            values[field] = _normalize_int64_field(
                raw_value, context=context, field=field
            )
        elif logical_type == "float64":
            values[field] = _normalize_float64_field(
                raw_value, context=context, field=field
            )
        elif logical_type == "bool":
            values[field] = _normalize_bool_field(
                raw_value, context=context, field=field
            )
        else:
            raise ValueError(
                f"unsupported OLMES detail metric type for {field!r} in {context}: "
                f"{logical_type!r}"
            )
    return values


def _index_checkpoint_members(
    inner: tarfile.TarFile,
    *,
    expected_step: int,
    context: str,
) -> tuple[dict[str, tarfile.TarInfo], dict[str, tarfile.TarInfo]]:
    metrics_by_task: dict[str, tarfile.TarInfo] = {}
    predictions_by_task: dict[str, tarfile.TarInfo] = {}
    step_prefix = f"step-{expected_step}/"

    for member in inner.getmembers():
        if not member.isfile():
            continue
        name = member.name
        if not name.startswith(step_prefix):
            raise ValueError(
                f"unexpected member path in {context}: {name!r}; "
                f"expected prefix {step_prefix!r}"
            )
        filename = name[len(step_prefix) :]
        if filename.endswith(_METRICS_SUFFIX):
            task = filename[: -len(_METRICS_SUFFIX)]
            metrics_by_task[task] = member
        elif filename.endswith(_PREDICTIONS_SUFFIX):
            task = filename[: -len(_PREDICTIONS_SUFFIX)]
            predictions_by_task[task] = member

    metrics_tasks = set(metrics_by_task)
    predictions_tasks = set(predictions_by_task)
    if metrics_tasks != predictions_tasks:
        missing_predictions = sorted(metrics_tasks - predictions_tasks)
        missing_metrics = sorted(predictions_tasks - metrics_tasks)
        details: list[str] = []
        if missing_predictions:
            details.append(f"missing predictions for {missing_predictions!r}")
        if missing_metrics:
            details.append(f"missing metrics for {missing_metrics!r}")
        raise ValueError(
            f"unpaired OLMES detail task files in {context}: {'; '.join(details)}"
        )

    return metrics_by_task, predictions_by_task


def _build_task_row(
    metrics_payload: dict[str, Any],
    *,
    recipe: str,
    params: str,
    seed_value: int,
    step: int,
    task: str,
    contract: OLMESContract,
    context: str,
) -> dict[str, object]:
    if recipe not in contract.recipe_map:
        raise ValueError(f"unknown OLMES detail recipe in {context}: {recipe!r}")
    if seed_value not in contract.seed_map:
        raise ValueError(
            f"unknown OLMES detail seed_value in {context}: {seed_value!r}"
        )

    task_name = metrics_payload.get("task_name")
    task_config = metrics_payload.get("task_config")
    if not isinstance(task_config, dict):
        raise ValueError(
            f"invalid OLMES detail task_config in {context}: expected object"
        )
    config_task_name = task_config.get("task_name")
    if task_name != task or config_task_name != task:
        raise ValueError(
            f"task identity mismatch in {context}: "
            f"path task={task!r}, task_name={task_name!r}, "
            f"task_config.task_name={config_task_name!r}"
        )

    model_config = metrics_payload.get("model_config")
    compute_config = metrics_payload.get("compute_config")
    if not isinstance(model_config, dict) or not isinstance(compute_config, dict):
        raise ValueError(
            f"invalid OLMES detail config objects in {context}: "
            "expected model_config and compute_config objects"
        )

    metrics_block = metrics_payload.get("metrics")
    if not isinstance(metrics_block, dict):
        raise ValueError(f"invalid OLMES detail metrics in {context}: expected object")

    primary_metric = task_config.get("primary_metric")
    if not isinstance(primary_metric, str) or not primary_metric.strip():
        raise ValueError(
            f"invalid OLMES detail primary_metric in {context}: {primary_metric!r}"
        )

    metric_values: dict[str, float | None] = {}
    for field in contract.metrics.detailed_tasks:
        if field in metrics_block:
            metric_values[field] = coerce_float(metrics_block[field])
        else:
            metric_values[field] = None

    task_hash = _require_hash(
        metrics_payload.get("task_hash"), context=context, field="task_hash"
    )
    model_hash = _require_hash(
        metrics_payload.get("model_hash"), context=context, field="model_hash"
    )

    return {
        "recipe": recipe,
        "data": contract.recipe_map[recipe],
        "params": params,
        "seed_value": seed_value,
        "seed": contract.seed_map[seed_value],
        "step": step,
        "task": task,
        "task_hash": task_hash,
        "model_hash": model_hash,
        "model_config": _canonical_json(model_config),
        "task_config": _canonical_json(task_config),
        "compute_config": _canonical_json(compute_config),
        "processing_time": _normalize_float64_field(
            metrics_payload.get("processing_time"),
            context=context,
            field="processing_time",
        ),
        "current_date": _require_hash(
            metrics_payload.get("current_date"), context=context, field="current_date"
        ),
        "num_instances": _normalize_int64_field(
            metrics_payload.get("num_instances"),
            context=context,
            field="num_instances",
        ),
        "task_idx": _normalize_int64_field(
            metrics_payload.get("task_idx"), context=context, field="task_idx"
        ),
        "primary_metric": primary_metric,
        **metric_values,
    }


def _validate_prediction_payload(
    prediction: Any,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(prediction, dict):
        raise ValueError(
            f"invalid OLMES detail prediction JSON in {context}: expected object"
        )
    missing = [key for key in _PREDICTION_REQUIRED_KEYS if key not in prediction]
    if missing:
        raise ValueError(
            f"invalid OLMES detail prediction JSON in {context}: "
            f"missing keys {missing!r}"
        )
    if not isinstance(prediction["metrics"], dict):
        raise ValueError(
            f"invalid OLMES detail prediction metrics in {context}: expected object"
        )
    if not isinstance(prediction["model_output"], list):
        raise ValueError(
            f"invalid OLMES detail prediction model_output in {context}: expected array"
        )
    return prediction


def _build_instance_row(
    prediction: dict[str, Any],
    *,
    recipe: str,
    params: str,
    seed_value: int,
    step: int,
    task: str,
    task_hash: str,
    model_hash: str,
    contract: OLMESContract,
    context: str,
) -> dict[str, object]:
    doc_id = _normalize_int64_field(
        prediction["doc_id"], context=context, field="doc_id"
    )
    label = _normalize_int64_field(prediction["label"], context=context, field="label")
    native_id, native_id_kind = _normalize_native_id(
        prediction.get("native_id"), context=context
    )
    prediction_task_hash = _require_hash(
        prediction["task_hash"], context=context, field="task_hash"
    )
    prediction_model_hash = _require_hash(
        prediction["model_hash"], context=context, field="model_hash"
    )
    if prediction_task_hash != task_hash or prediction_model_hash != model_hash:
        raise ValueError(
            f"hash mismatch in {context}, doc_id={doc_id}: "
            f"expected task_hash={task_hash!r}, model_hash={model_hash!r}"
        )

    metric_values = _extract_metric_values(
        prediction["metrics"],
        metric_names=contract.metrics.detailed_instances,
        table=contract.tables.detailed_instances,
        context=f"{context}, doc_id={doc_id}",
    )

    return {
        "recipe": recipe,
        "data": contract.recipe_map[recipe],
        "params": params,
        "seed_value": seed_value,
        "seed": contract.seed_map[seed_value],
        "step": step,
        "task": task,
        "task_hash": task_hash,
        "model_hash": model_hash,
        "doc_id": doc_id,
        "native_id": native_id,
        "native_id_kind": native_id_kind,
        "label": label,
        **metric_values,
    }


def _build_choice_rows(
    prediction: dict[str, Any],
    *,
    recipe: str,
    params: str,
    seed_value: int,
    step: int,
    task: str,
    contract: OLMESContract,
    context: str,
) -> list[dict[str, object]]:
    doc_id = _normalize_int64_field(
        prediction["doc_id"], context=context, field="doc_id"
    )
    rows: list[dict[str, object]] = []
    for choice_index, choice_metrics in enumerate(prediction["model_output"]):
        if not isinstance(choice_metrics, dict):
            raise ValueError(
                f"invalid OLMES detail choice metrics in {context}, "
                f"doc_id={doc_id}, choice_index={choice_index}: expected object"
            )
        metric_values = _extract_metric_values(
            choice_metrics,
            metric_names=contract.metrics.detailed_choices,
            table=contract.tables.detailed_choices,
            context=f"{context}, doc_id={doc_id}, choice_index={choice_index}",
        )
        rows.append(
            {
                "recipe": recipe,
                "data": contract.recipe_map[recipe],
                "params": params,
                "seed_value": seed_value,
                "seed": contract.seed_map[seed_value],
                "step": step,
                "task": task,
                "doc_id": doc_id,
                "choice_index": choice_index,
                **metric_values,
            }
        )
    return rows


def _process_checkpoint_tar(
    inner_bytes: bytes,
    *,
    recipe: str,
    params: str,
    seed_value: int,
    step: int,
    contract: OLMESContract,
    seen_task_keys: set[tuple[str, str, int, int, str]],
    seen_instance_keys: set[tuple[str, str, int, int, str, int]],
    seen_choice_keys: set[tuple[str, str, int, int, str, int, int]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    context = (
        f"recipe={recipe!r}, params={params!r}, seed_value={seed_value}, step={step}"
    )
    task_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    choice_rows: list[dict[str, object]] = []

    with tarfile.open(fileobj=io.BytesIO(inner_bytes), mode="r:gz") as inner:
        metrics_by_task, predictions_by_task = _index_checkpoint_members(
            inner, expected_step=step, context=context
        )

        for task in sorted(metrics_by_task):
            task_context = f"{context}, task={task!r}"
            metrics_member = metrics_by_task[task]
            predictions_member = predictions_by_task[task]
            metrics_file = inner.extractfile(metrics_member)
            predictions_file = inner.extractfile(predictions_member)
            if metrics_file is None or predictions_file is None:
                raise ValueError(f"missing OLMES detail task payload in {context}")

            metrics_payload = orjson.loads(metrics_file.read())
            if not isinstance(metrics_payload, dict):
                raise ValueError(
                    f"invalid OLMES detail metrics JSON in {context}: expected object"
                )

            declared_instances = _normalize_int64_field(
                metrics_payload.get("num_instances"),
                context=task_context,
                field="num_instances",
            )
            task_row = _build_task_row(
                metrics_payload,
                recipe=recipe,
                params=params,
                seed_value=seed_value,
                step=step,
                task=task,
                contract=contract,
                context=task_context,
            )
            task_key = (recipe, params, seed_value, step, task)
            if task_key in seen_task_keys:
                raise ValueError(
                    f"duplicate OLMES detail task row in {context}: "
                    f"recipe={recipe!r}, params={params!r}, "
                    f"seed_value={seed_value}, step={step}, task={task!r}"
                )
            seen_task_keys.add(task_key)
            task_rows.append(task_row)

            task_hash = str(task_row["task_hash"])
            model_hash = str(task_row["model_hash"])
            parsed_instances = 0
            for line_number, line in enumerate(predictions_file, start=1):
                if not line.strip():
                    continue
                prediction = _validate_prediction_payload(
                    orjson.loads(line),
                    context=f"{task_context}, line={line_number}",
                )
                instance_row = _build_instance_row(
                    prediction,
                    recipe=recipe,
                    params=params,
                    seed_value=seed_value,
                    step=step,
                    task=task,
                    task_hash=task_hash,
                    model_hash=model_hash,
                    contract=contract,
                    context=f"{task_context}, line={line_number}",
                )
                doc_id = int(instance_row["doc_id"])
                instance_key = (recipe, params, seed_value, step, task, doc_id)
                if instance_key in seen_instance_keys:
                    raise ValueError(
                        f"duplicate OLMES detail instance row in {task_context}: "
                        f"doc_id={doc_id}"
                    )
                seen_instance_keys.add(instance_key)
                instance_rows.append(instance_row)

                for choice_row in _build_choice_rows(
                    prediction,
                    recipe=recipe,
                    params=params,
                    seed_value=seed_value,
                    step=step,
                    task=task,
                    contract=contract,
                    context=f"{task_context}, line={line_number}",
                ):
                    choice_key = (
                        recipe,
                        params,
                        seed_value,
                        step,
                        task,
                        doc_id,
                        int(choice_row["choice_index"]),
                    )
                    if choice_key in seen_choice_keys:
                        raise ValueError(
                            f"duplicate OLMES detail choice row in {task_context}: "
                            f"doc_id={doc_id}, choice_index={choice_row['choice_index']}"
                        )
                    seen_choice_keys.add(choice_key)
                    choice_rows.append(choice_row)
                parsed_instances += 1

            if parsed_instances != declared_instances:
                raise ValueError(
                    f"instance count mismatch in {task_context}: "
                    f"declared={declared_instances}, predictions={parsed_instances}"
                )

    return task_rows, instance_rows, choice_rows


def _parse_checkpoint_member_path(
    member_name: str,
    *,
    expected_recipe: str,
) -> tuple[str, str, int, int] | None:
    match = _CHECKPOINT_MEMBER_RE.match(member_name)
    if match is None:
        return None
    recipe = match.group("recipe")
    if recipe != expected_recipe:
        raise ValueError(
            f"unexpected recipe in checkpoint member {member_name!r}: "
            f"expected {expected_recipe!r}"
        )
    params = match.group("params")
    seed_value = int(match.group("seed_value"))
    step = _normalize_step(match.group("step"), row_index=0)
    return recipe, params, seed_value, step


def stream_detail_rows(
    input_path: Path,
    recipe: str,
    *,
    contract: OLMESContract | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    int,
]:
    contract = contract or load_olmes_contract()
    _assert_all_detailed_schema_parity(contract)

    if not input_path.is_file():
        raise FileNotFoundError(f"OLMES detail archive not found: {input_path}")

    task_rows: list[dict[str, object]] = []
    instance_rows: list[dict[str, object]] = []
    choice_rows: list[dict[str, object]] = []
    seen_task_keys: set[tuple[str, str, int, int, str]] = set()
    seen_instance_keys: set[tuple[str, str, int, int, str, int]] = set()
    seen_choice_keys: set[tuple[str, str, int, int, str, int, int]] = set()
    checkpoint_count = 0

    with tarfile.open(input_path, mode="r:gz") as outer:
        for member in outer:
            if not member.isfile():
                continue
            parsed = _parse_checkpoint_member_path(member.name, expected_recipe=recipe)
            if parsed is None:
                continue
            _, params, seed_value, step = parsed
            inner_file = outer.extractfile(member)
            if inner_file is None:
                raise ValueError(
                    f"unable to read checkpoint member from {input_path}: "
                    f"{member.name!r}"
                )
            checkpoint_tasks, checkpoint_instances, checkpoint_choices = (
                _process_checkpoint_tar(
                    inner_file.read(),
                    recipe=recipe,
                    params=params,
                    seed_value=seed_value,
                    step=step,
                    contract=contract,
                    seen_task_keys=seen_task_keys,
                    seen_instance_keys=seen_instance_keys,
                    seen_choice_keys=seen_choice_keys,
                )
            )
            task_rows.extend(checkpoint_tasks)
            instance_rows.extend(checkpoint_instances)
            choice_rows.extend(checkpoint_choices)
            checkpoint_count += 1

    task_rows.sort(
        key=lambda row: tuple(
            row[name] for name in contract.tables.detailed_tasks.sort_key
        )
    )
    instance_rows.sort(
        key=lambda row: tuple(
            row[name] for name in contract.tables.detailed_instances.sort_key
        )
    )
    choice_rows.sort(
        key=lambda row: tuple(
            row[name] for name in contract.tables.detailed_choices.sort_key
        )
    )
    return task_rows, instance_rows, choice_rows, checkpoint_count


def stream_detail_task_rows(
    input_path: Path,
    recipe: str,
    *,
    contract: OLMESContract | None = None,
) -> tuple[list[dict[str, object]], int]:
    task_rows, _, _, checkpoint_count = stream_detail_rows(
        input_path, recipe, contract=contract
    )
    return task_rows, checkpoint_count


def _write_sorted_parquet_from_temp(
    temp_path: Path,
    *,
    output_path: Path,
    sort_key: tuple[str, ...],
    table: OLMESTableContract,
    columns: tuple[str, ...],
) -> int:
    if not temp_path.is_file():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty_df = _typed_output_dataframe([], table=table, columns=columns)
        empty_df.to_parquet(output_path, index=False)
        return 0

    frame = pd.read_parquet(temp_path)
    if not frame.empty:
        frame = frame.sort_values(list(sort_key), kind="mergesort")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    temp_path.unlink(missing_ok=True)
    return len(frame)


def preprocess_olmes_details(
    paths: DataDecidePaths,
    recipe: str,
    *,
    contract: OLMESContract | None = None,
    input_path: Path | None = None,
    output_tasks_path: Path | None = None,
    output_instances_path: Path | None = None,
    output_choices_path: Path | None = None,
    verbose: bool = False,
) -> OlmesDetailsPreprocessResult:
    contract = contract or load_olmes_contract()
    _assert_all_detailed_schema_parity(contract)

    resolved_input = input_path or _recipe_tar_path(paths, recipe)
    resolved_tasks = output_tasks_path or paths.olmes_details_tasks_path(recipe)
    resolved_instances = output_instances_path or paths.olmes_details_instances_path(
        recipe
    )
    resolved_choices = output_choices_path or paths.olmes_details_choices_path(recipe)

    task_columns = _output_columns(contract, "detailed_tasks")
    instance_columns = _output_columns(contract, "detailed_instances")
    choice_columns = _output_columns(contract, "detailed_choices")

    temp_tasks = resolved_tasks.with_suffix(".tmp.parquet")
    temp_instances = resolved_instances.with_suffix(".tmp.parquet")
    temp_choices = resolved_choices.with_suffix(".tmp.parquet")
    resolved_tasks.parent.mkdir(parents=True, exist_ok=True)

    task_writer: pq.ParquetWriter | None = None
    instance_writer: pq.ParquetWriter | None = None
    choice_writer: pq.ParquetWriter | None = None

    seen_task_keys: set[tuple[str, str, int, int, str]] = set()
    seen_instance_keys: set[tuple[str, str, int, int, str, int]] = set()
    seen_choice_keys: set[tuple[str, str, int, int, str, int, int]] = set()
    checkpoint_count = 0

    if not resolved_input.is_file():
        raise FileNotFoundError(f"OLMES detail archive not found: {resolved_input}")

    try:
        with tarfile.open(resolved_input, mode="r:gz") as outer:
            for member in outer:
                if not member.isfile():
                    continue
                parsed = _parse_checkpoint_member_path(
                    member.name, expected_recipe=recipe
                )
                if parsed is None:
                    continue
                _, params, seed_value, step = parsed
                inner_file = outer.extractfile(member)
                if inner_file is None:
                    raise ValueError(
                        f"unable to read checkpoint member from {resolved_input}: "
                        f"{member.name!r}"
                    )
                checkpoint_tasks, checkpoint_instances, checkpoint_choices = (
                    _process_checkpoint_tar(
                        inner_file.read(),
                        recipe=recipe,
                        params=params,
                        seed_value=seed_value,
                        step=step,
                        contract=contract,
                        seen_task_keys=seen_task_keys,
                        seen_instance_keys=seen_instance_keys,
                        seen_choice_keys=seen_choice_keys,
                    )
                )

                for rows, table, columns, writer_attr in (
                    (
                        checkpoint_tasks,
                        contract.tables.detailed_tasks,
                        task_columns,
                        "task_writer",
                    ),
                    (
                        checkpoint_instances,
                        contract.tables.detailed_instances,
                        instance_columns,
                        "instance_writer",
                    ),
                    (
                        checkpoint_choices,
                        contract.tables.detailed_choices,
                        choice_columns,
                        "choice_writer",
                    ),
                ):
                    if not rows:
                        continue
                    frame = _typed_output_dataframe(rows, table=table, columns=columns)
                    table_data = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer_attr == "task_writer":
                        if task_writer is None:
                            task_writer = pq.ParquetWriter(
                                temp_tasks, table_data.schema
                            )
                        task_writer.write_table(table_data)
                    elif writer_attr == "instance_writer":
                        if instance_writer is None:
                            instance_writer = pq.ParquetWriter(
                                temp_instances, table_data.schema
                            )
                        instance_writer.write_table(table_data)
                    else:
                        if choice_writer is None:
                            choice_writer = pq.ParquetWriter(
                                temp_choices, table_data.schema
                            )
                        choice_writer.write_table(table_data)

                checkpoint_count += 1
    finally:
        if task_writer is not None:
            task_writer.close()
        if instance_writer is not None:
            instance_writer.close()
        if choice_writer is not None:
            choice_writer.close()

    task_count = _write_sorted_parquet_from_temp(
        temp_tasks,
        output_path=resolved_tasks,
        sort_key=contract.tables.detailed_tasks.sort_key,
        table=contract.tables.detailed_tasks,
        columns=task_columns,
    )
    instance_count = _write_sorted_parquet_from_temp(
        temp_instances,
        output_path=resolved_instances,
        sort_key=contract.tables.detailed_instances.sort_key,
        table=contract.tables.detailed_instances,
        columns=instance_columns,
    )
    choice_count = _write_sorted_parquet_from_temp(
        temp_choices,
        output_path=resolved_choices,
        sort_key=contract.tables.detailed_choices.sort_key,
        table=contract.tables.detailed_choices,
        columns=choice_columns,
    )

    result = OlmesDetailsPreprocessResult(
        recipe=recipe,
        input_path=resolved_input,
        output_tasks_path=resolved_tasks,
        output_instances_path=resolved_instances,
        output_choices_path=resolved_choices,
        row_count=task_count,
        instance_count=instance_count,
        choice_count=choice_count,
        checkpoint_count=checkpoint_count,
    )
    if verbose:
        print(f"olmes-details recipe: {result.recipe}")
        print(f"olmes-details input: {result.input_path}")
        print(f"olmes-details tasks output: {result.output_tasks_path}")
        print(f"olmes-details instances output: {result.output_instances_path}")
        print(f"olmes-details choices output: {result.output_choices_path}")
        print(f"olmes-details task rows: {result.row_count}")
        print(f"olmes-details instance rows: {result.instance_count}")
        print(f"olmes-details choice rows: {result.choice_count}")
        print(f"olmes-details checkpoints: {result.checkpoint_count}")
    return result

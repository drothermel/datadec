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

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


@dataclass(frozen=True, slots=True)
class OlmesDetailsPreprocessResult:
    recipe: str
    input_path: Path
    output_path: Path
    row_count: int
    checkpoint_count: int


def _output_columns(contract: OLMESContract) -> tuple[str, ...]:
    return tuple(column.name for column in contract.tables.detailed_tasks.columns)


def _metric_columns(contract: OLMESContract) -> tuple[str, ...]:
    return contract.metrics.detailed_tasks


def _assert_detailed_tasks_schema_parity(contract: OLMESContract) -> None:
    table = contract.tables.detailed_tasks
    expected = _output_columns(contract)
    actual = tuple(column.name for column in table.columns)
    if actual != expected:
        raise AssertionError(
            "persisted OLMES detail task columns drift from contract: "
            f"expected={expected!r}, actual={actual!r}"
        )

    metric_columns = _metric_columns(contract)
    trailing = tuple(column.name for column in table.columns[-len(metric_columns) :])
    if trailing != metric_columns:
        raise AssertionError(
            "OLMES detail task metric columns drift from contract: "
            f"expected={metric_columns!r}, actual={trailing!r}"
        )


def _pandas_dtype(
    logical_type: Literal["string", "int64", "float64", "bool"],
) -> str:
    if logical_type == "string":
        return "string"
    return logical_type


def _typed_output_dataframe(
    rows: list[dict[str, object]],
    *,
    table: OLMESTableContract,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    column_types = {column.name: column.logical_type for column in table.columns}
    values_by_column = {column: [row.get(column) for row in rows] for column in columns}
    typed_columns: dict[str, pd.Series[Any]] = {}
    for column in columns:
        logical_type = column_types[column]
        dtype = _pandas_dtype(logical_type)
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
            f"invalid OLMES detail {field} in {context}: {value!r}; "
            "expected a finite float64 value"
        )
    return float(decimal_value)


def _count_jsonl_lines(file_obj: io.BufferedReader) -> int:
    count = 0
    for _ in file_obj:
        count += 1
    return count


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

    return {
        "recipe": recipe,
        "data": contract.recipe_map[recipe],
        "params": params,
        "seed_value": seed_value,
        "seed": contract.seed_map[seed_value],
        "step": step,
        "task": task,
        "task_hash": _require_hash(
            metrics_payload.get("task_hash"), context=context, field="task_hash"
        ),
        "model_hash": _require_hash(
            metrics_payload.get("model_hash"), context=context, field="model_hash"
        ),
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


def _process_checkpoint_tar(
    inner_bytes: bytes,
    *,
    recipe: str,
    params: str,
    seed_value: int,
    step: int,
    contract: OLMESContract,
    seen_keys: set[tuple[str, str, int, int, str]],
) -> list[dict[str, object]]:
    context = (
        f"recipe={recipe!r}, params={params!r}, seed_value={seed_value}, step={step}"
    )
    rows: list[dict[str, object]] = []

    with tarfile.open(fileobj=io.BytesIO(inner_bytes), mode="r:gz") as inner:
        metrics_by_task, predictions_by_task = _index_checkpoint_members(
            inner, expected_step=step, context=context
        )

        for task in sorted(metrics_by_task):
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
                context=f"{context}, task={task!r}",
                field="num_instances",
            )
            actual_instances = _count_jsonl_lines(predictions_file)
            if actual_instances != declared_instances:
                raise ValueError(
                    f"instance count mismatch in {context}, task={task!r}: "
                    f"declared={declared_instances}, predictions={actual_instances}"
                )

            row = _build_task_row(
                metrics_payload,
                recipe=recipe,
                params=params,
                seed_value=seed_value,
                step=step,
                task=task,
                contract=contract,
                context=f"{context}, task={task!r}",
            )
            primary_key = (recipe, params, seed_value, step, task)
            if primary_key in seen_keys:
                raise ValueError(
                    f"duplicate OLMES detail row in {context}: "
                    f"recipe={recipe!r}, params={params!r}, "
                    f"seed_value={seed_value}, step={step}, task={task!r}"
                )
            seen_keys.add(primary_key)
            rows.append(row)

    return rows


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


def stream_detail_task_rows(
    input_path: Path,
    recipe: str,
    *,
    contract: OLMESContract | None = None,
) -> tuple[list[dict[str, object]], int]:
    contract = contract or load_olmes_contract()
    _assert_detailed_tasks_schema_parity(contract)

    if not input_path.is_file():
        raise FileNotFoundError(f"OLMES detail archive not found: {input_path}")

    rows: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, int, int, str]] = set()
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
            checkpoint_rows = _process_checkpoint_tar(
                inner_file.read(),
                recipe=recipe,
                params=params,
                seed_value=seed_value,
                step=step,
                contract=contract,
                seen_keys=seen_keys,
            )
            rows.extend(checkpoint_rows)
            checkpoint_count += 1

    sort_key = contract.tables.detailed_tasks.sort_key
    rows.sort(key=lambda row: tuple(row[name] for name in sort_key))
    return rows, checkpoint_count


def preprocess_olmes_details(
    paths: DataDecidePaths,
    recipe: str,
    *,
    contract: OLMESContract | None = None,
    verbose: bool = False,
) -> OlmesDetailsPreprocessResult:
    contract = contract or load_olmes_contract()
    input_path = _recipe_tar_path(paths, recipe)
    output_path = paths.olmes_details_tasks_path(recipe)
    output_columns = _output_columns(contract)

    rows, checkpoint_count = stream_detail_task_rows(
        input_path, recipe, contract=contract
    )
    output_df = _typed_output_dataframe(
        rows,
        table=contract.tables.detailed_tasks,
        columns=output_columns,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)

    result = OlmesDetailsPreprocessResult(
        recipe=recipe,
        input_path=input_path,
        output_path=output_path,
        row_count=len(output_df),
        checkpoint_count=checkpoint_count,
    )
    if verbose:
        print(f"olmes-details recipe: {result.recipe}")
        print(f"olmes-details input: {result.input_path}")
        print(f"olmes-details output: {result.output_path}")
        print(f"olmes-details rows: {result.row_count}")
        print(f"olmes-details checkpoints: {result.checkpoint_count}")
    return result

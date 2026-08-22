from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datadec.paper.models import (
    AnalysisId,
    AttemptResult,
    AttemptRole,
    AttemptSpec,
    AxisScale,
    AxisSpec,
    ClaimRegistry,
    ComparisonParameterName,
    ComparisonRule,
    ContentIdentity,
    DimensionValue,
    EvidenceLevel,
    InputTableSpec,
    MeasureValue,
    NamedCount,
    PaperClaim,
    PaperValidationContract,
    PlotPoint,
    PlotSeries,
    PredicateOperator,
    RowPredicate,
    RowSelection,
    ValidationOutcome,
)

_DECISION_INPUT_ID = "new_eval_decision_accuracy"
_MEANS_INPUT_ID = "new_eval_means"
_COMPUTE_INPUT_ID = "olmes_aggregate"
_TARGET_METRIC = "primary_score"
_CONTINUOUS_METRIC = "logits_per_byte_corr"
_PAPER_ALIAS = "Correct Prob"

_DECISION_KEY_COLUMNS = ("size", "task", "target_ranking")
_MEANS_KEY_COLUMNS = ("size", "task")
_COMPUTE_KEY_COLUMNS = ("data", "params", "step")
_DECISION_COLUMNS = (
    "size",
    "task",
    "target_ranking",
    "logits_per_byte_corr",
    "logits_per_char_corr",
    "primary_score",
)
_MEANS_COLUMNS = (
    "size",
    "task",
    "primary_score",
    "logits_per_byte_corr",
    "logits_per_char_corr",
)
_DECISION_SIZES = ("4M", "60M", "150M")
_MEANS_SIZES = (*_DECISION_SIZES, "1B")
_ALL_TASKS = (
    "arc_challenge",
    "codex_humaneval",
    "gsm8k",
    "hellaswag",
    "mbpp",
    "minerva",
    "mmlu",
    "olmes_core9",
)
_TARGET_RANKINGS = (_TARGET_METRIC, _CONTINUOUS_METRIC)
_CODE_TASKS = ("mbpp", "codex_humaneval")
_MATH_TASKS = ("minerva", "gsm8k")
_PLOT_TASKS = (*_MATH_TASKS, *_CODE_TASKS)
_PLOT_SIZES = ("4M", "60M")
_PLOT_METRICS = (_TARGET_METRIC, _CONTINUOUS_METRIC)
_SUPPORTED_CLAIMS = frozenset(
    {
        "DD-0017",
        "DD-0018",
        "DD-0213",
        "DD-0221",
        "DD-0222",
        "DD-0224",
        "DD-0225",
        "DD-0226",
        "DD-0227",
        "DD-0413",
        "DD-0414",
    }
)
_POINTWISE_SENSITIVITIES = {
    "DD-0221": "dd-0221-size-pointwise",
    "DD-0222": "dd-0222-size-pointwise",
    "DD-0226": "dd-0226-size-pointwise",
}

_AGGREGATE_LIMITATION = (
    "This verifies supplied author-derived aggregate tables; it does not "
    "independently recompute evaluations."
)
_ALIAS_LIMITATION = (
    "The paper legend alias 'Correct Prob' is unresolved: the supplied aggregate "
    "column is logits_per_byte_corr, so results retain that actual metric name."
)


@dataclass(frozen=True, slots=True)
class _LoadedTable:
    spec: InputTableSpec
    sha256: str
    frame: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _ComputeEvidence:
    selection_frame: pd.DataFrame
    steps: tuple[int, int] | None
    prediction_compute: float | None
    target_compute: float | None
    percent_target_compute: float | None
    missing_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Evaluation:
    computed_value: dict[str, object]
    outcome: ValidationOutcome
    diagnostics: tuple[str, ...]
    denominator: int


def _sha256_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _json_value(value: object) -> object:
    return value.item() if isinstance(value, np.generic) else value


def _selected_key_sha256(frame: pd.DataFrame, key_columns: tuple[str, ...]) -> str:
    keys = sorted(
        tuple(_json_value(value) for value in row)
        for row in frame.loc[:, list(key_columns)].itertuples(index=False, name=None)
    )
    records = [dict(zip(key_columns, key, strict=True)) for key in keys]
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _input_spec(contract: PaperValidationContract, table_id: str) -> InputTableSpec:
    try:
        return next(item for item in contract.inputs if item.id == table_id)
    except StopIteration as error:
        raise ValueError(f"validation contract has no input {table_id}") from error


def _identity(
    input_identities: Mapping[str, ContentIdentity], table_id: str
) -> ContentIdentity:
    identity = input_identities.get(table_id)
    if identity is None:
        raise ValueError(f"input identities omit {table_id}")
    if identity.id != table_id:
        raise ValueError(f"{table_id} identity has the wrong logical input ID")
    return identity


def _validate_math_schema(
    *, path: Path, expected_columns: tuple[str, ...], table_id: str
) -> None:
    schema = pq.ParquetFile(path).schema_arrow
    if tuple(schema.names) != expected_columns:
        raise ValueError(
            f"{table_id} schema must be exactly {expected_columns!r}; "
            f"found {tuple(schema.names)!r}"
        )
    for column in expected_columns:
        field = schema.field(column)
        if column in {*_DECISION_KEY_COLUMNS, *_MEANS_KEY_COLUMNS}:
            if not (
                pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
            ):
                raise ValueError(f"{table_id}.{column} must have a string type")
        elif not pa.types.is_floating(field.type):
            raise ValueError(f"{table_id}.{column} must have a floating type")


def _load_table(
    *,
    data_root: Path,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
    table_id: str,
    exact_columns: tuple[str, ...] | None = None,
    read_columns: tuple[str, ...] | None = None,
    filters: list[tuple[str, str, list[str]]] | None = None,
) -> _LoadedTable:
    spec = _input_spec(contract, table_id)
    columns = exact_columns if read_columns is None else read_columns
    if columns is None:
        raise ValueError("table loading requires declared read columns")
    undeclared = set(columns).difference(spec.columns)
    if undeclared:
        raise ValueError(
            f"{table_id} adapter columns are undeclared: {tuple(sorted(undeclared))!r}"
        )
    path = data_root / spec.path
    if not path.is_file():
        raise FileNotFoundError(f"declared input does not exist: {path}")
    if exact_columns is not None:
        if spec.columns != exact_columns:
            raise ValueError(
                f"{table_id} configured columns differ from its frozen schema"
            )
        _validate_math_schema(
            path=path, expected_columns=exact_columns, table_id=table_id
        )
    actual_sha256 = _sha256_file(path)
    if _identity(input_identities, table_id).sha256 != actual_sha256:
        raise ValueError(f"{table_id} identity differs from the actual Parquet input")
    schema_columns = set(pq.ParquetFile(path).schema_arrow.names)
    missing_columns = tuple(sorted(set(columns).difference(schema_columns)))
    if missing_columns:
        raise ValueError(f"{table_id} is missing columns: {missing_columns!r}")
    frame = pd.read_parquet(path, columns=list(columns), filters=filters)
    if _sha256_file(path) != actual_sha256:
        raise RuntimeError(f"{table_id} changed while it was being read")
    return _LoadedTable(spec=spec, sha256=actual_sha256, frame=frame)


def _group_label(table_id: str, key: tuple[object, ...]) -> str:
    return f"{table_id}:" + "/".join(str(_json_value(value)) for value in key)


def _cube_anomalies(
    table: _LoadedTable,
    *,
    key_columns: tuple[str, ...],
    expected_keys: tuple[tuple[str, ...], ...],
    numeric_columns: tuple[str, ...],
) -> tuple[str, ...]:
    frame = table.frame
    anomalies: list[str] = []
    valid_key_rows = frame.loc[:, list(key_columns)].notna().all(axis=1)
    actual_keys = [
        tuple(str(value) for value in row)
        for row in frame.loc[valid_key_rows, list(key_columns)].itertuples(
            index=False, name=None
        )
    ]
    expected = set(expected_keys)
    actual = set(actual_keys)
    anomalies.extend(
        f"missing:{_group_label(table.spec.id, key)}"
        for key in expected_keys
        if key not in actual
    )
    anomalies.extend(
        f"unexpected:{_group_label(table.spec.id, key)}"
        for key in sorted(actual.difference(expected))
    )
    null_key_count = int((~valid_key_rows).sum())
    if null_key_count:
        anomalies.append(f"invalid:{table.spec.id}:null-key-rows={null_key_count}")
    counts: dict[tuple[str, ...], int] = {}
    for key in actual_keys:
        counts[key] = counts.get(key, 0) + 1
    anomalies.extend(
        f"duplicate:{_group_label(table.spec.id, key)}:count={count}"
        for key, count in sorted(counts.items())
        if count != 1
    )
    for row in frame.itertuples(index=False):
        key = tuple(getattr(row, column) for column in key_columns)
        for column in numeric_columns:
            value = getattr(row, column)
            if isinstance(value, bool):
                finite = False
            else:
                try:
                    finite = math.isfinite(float(value))
                except (TypeError, ValueError):
                    finite = False
            if not finite:
                anomalies.append(
                    f"nonfinite:{_group_label(table.spec.id, key)}:{column}"
                )
    return tuple(dict.fromkeys(anomalies))


def _compute_evidence(table: _LoadedTable) -> _ComputeEvidence:
    frame = table.frame
    missing: list[str] = []
    selected_frames: list[pd.DataFrame] = []
    steps: list[int] = []
    computes: list[float] = []
    for size in ("4M", "1B"):
        size_rows = frame.loc[frame["params"].eq(size)]
        if size_rows.empty:
            missing.append(f"missing:{table.spec.id}:{size}:full-final")
            continue
        numeric_steps = pd.to_numeric(size_rows["step"], errors="coerce")
        finite_steps = numeric_steps[numeric_steps.map(math.isfinite)]
        if finite_steps.empty:
            missing.append(f"nonfinite:{table.spec.id}:{size}:step")
            continue
        step = int(finite_steps.max())
        if any(value != int(value) for value in finite_steps):
            missing.append(f"invalid:{table.spec.id}:{size}:noninteger-step")
            continue
        final_rows = size_rows.loc[numeric_steps.eq(step)].copy()
        final_computes = {
            float(value)
            for value in final_rows["compute"]
            if not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0
        }
        if len(final_computes) != 1 or len(final_rows) == 0:
            missing.append(f"invalid:{table.spec.id}:{size}:full-final-compute")
            continue
        if len(final_rows["data"].dropna().unique()) != len(
            frame.loc[frame["params"].eq(size), "data"].dropna().unique()
        ):
            missing.append(f"incomplete:{table.spec.id}:{size}:full-final-recipes")
            continue
        selected_frames.append(final_rows)
        steps.append(step)
        computes.append(next(iter(final_computes)))
    selected = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else frame.iloc[0:0].copy()
    )
    if missing or len(computes) != 2:
        return _ComputeEvidence(
            selection_frame=selected,
            steps=None,
            prediction_compute=None,
            target_compute=None,
            percent_target_compute=None,
            missing_groups=tuple(missing),
        )
    percent = computes[0] / computes[1] * 100.0
    if not math.isfinite(percent):
        return _ComputeEvidence(
            selection_frame=selected,
            steps=None,
            prediction_compute=None,
            target_compute=None,
            percent_target_compute=None,
            missing_groups=(f"nonfinite:{table.spec.id}:4M-vs-1B:percent",),
        )
    return _ComputeEvidence(
        selection_frame=selected,
        steps=(steps[0], steps[1]),
        prediction_compute=computes[0],
        target_compute=computes[1],
        percent_target_compute=percent,
        missing_groups=(),
    )


def _rule(rules: Mapping[str, ComparisonRule], attempt: AttemptSpec) -> ComparisonRule:
    try:
        return rules[attempt.comparison_rule_id]
    except KeyError as error:
        raise ValueError(
            f"math/code attempt references unknown rule {attempt.comparison_rule_id}"
        ) from error


def _parameter(rule: ComparisonRule, name: ComparisonParameterName) -> float:
    return rule.parameter(name).default


def _decision_value(
    frame: pd.DataFrame,
    *,
    size: str,
    task: str,
    target_metric: str,
    predictor_metric: str,
) -> float:
    rows = frame.loc[
        frame["size"].eq(size)
        & frame["task"].eq(task)
        & frame["target_ranking"].eq(target_metric),
        predictor_metric,
    ]
    if len(rows) != 1:
        raise ValueError(
            "validated math/code cube did not provide exactly one required cell"
        )
    return float(rows.iloc[0])


def _mean_value(frame: pd.DataFrame, *, size: str, task: str, metric: str) -> float:
    rows = frame.loc[frame["size"].eq(size) & frame["task"].eq(task), metric]
    if len(rows) != 1:
        raise ValueError("validated math/code means did not provide one required cell")
    return float(rows.iloc[0])


def _mean(values: Iterable[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("mean requires at least one value")
    return math.fsum(ordered) / len(ordered)


def _component(
    *, task: str, size: str, predictor_metric: str, decision_accuracy: float
) -> dict[str, object]:
    return {
        "task": task,
        "size": size,
        "predictor_metric": predictor_metric,
        "target_metric": _TARGET_METRIC,
        "decision_accuracy": decision_accuracy,
    }


def _gain_components(
    decision: pd.DataFrame, tasks: tuple[str, ...], sizes: tuple[str, ...]
) -> list[dict[str, object]]:
    components: list[dict[str, object]] = []
    for task in tasks:
        for size in sizes:
            accuracy = _decision_value(
                decision,
                size=size,
                task=task,
                target_metric=_TARGET_METRIC,
                predictor_metric=_TARGET_METRIC,
            )
            continuous = _decision_value(
                decision,
                size=size,
                task=task,
                target_metric=_TARGET_METRIC,
                predictor_metric=_CONTINUOUS_METRIC,
            )
            components.append(
                {
                    "task": task,
                    "size": size,
                    "target_metric": _TARGET_METRIC,
                    "baseline_predictor_metric": _TARGET_METRIC,
                    "baseline_decision_accuracy": accuracy,
                    "predictor_metric": _CONTINUOUS_METRIC,
                    "decision_accuracy": continuous,
                    "proxy_gain": continuous - accuracy,
                }
            )
    return components


def _alias_payload() -> dict[str, str]:
    return {"label": _PAPER_ALIAS, "status": "unresolved"}


def _evaluate(
    *,
    attempt: AttemptSpec,
    rule: ComparisonRule,
    decision: pd.DataFrame,
    means: pd.DataFrame,
    compute: _ComputeEvidence,
    pointwise: bool = False,
) -> _Evaluation:
    claim_id = attempt.claim_id
    diagnostics = [
        f"target_metric={_TARGET_METRIC}",
        f"predictor_metric={_CONTINUOUS_METRIC}",
        f"paper_legend_alias={_PAPER_ALIAS}",
        "paper_legend_alias_status=unresolved",
    ]
    if claim_id in {"DD-0017", "DD-0018"}:
        task = _CODE_TASKS[claim_id == "DD-0018"]
        accuracy = _decision_value(
            decision,
            size="4M",
            task=task,
            target_metric=_TARGET_METRIC,
            predictor_metric=_CONTINUOUS_METRIC,
        )
        threshold = _parameter(rule, ComparisonParameterName.ACCURACY_THRESHOLD)
        maximum_percent = _parameter(
            rule, ComparisonParameterName.MAXIMUM_SCALE_PERCENT
        )
        if (
            compute.prediction_compute is None
            or compute.target_compute is None
            or compute.percent_target_compute is None
        ):
            raise ValueError("complete compute evidence is required")
        accuracy_holds = accuracy >= threshold
        compute_holds = compute.percent_target_compute <= maximum_percent
        holds = accuracy_holds and compute_holds
        computed = {
            "task": task,
            "size": "4M",
            "target_size": "1B",
            "target_metric": _TARGET_METRIC,
            "predictor_metric": _CONTINUOUS_METRIC,
            "decision_accuracy": accuracy,
            "accuracy_threshold": threshold,
            "prediction_compute": compute.prediction_compute,
            "target_compute": compute.target_compute,
            "percent_target_compute": compute.percent_target_compute,
            "maximum_scale_percent": maximum_percent,
            "accuracy_satisfied": accuracy_holds,
            "scale_budget_satisfied": compute_holds,
            "satisfied": holds,
            "paper_legend_alias": _alias_payload(),
        }
        diagnostics.extend(
            (
                f"accuracy_threshold={threshold:.17g}",
                f"maximum_scale_percent={maximum_percent:.17g}",
                f"percent_target_compute={compute.percent_target_compute:.17g}",
            )
        )
        return _Evaluation(
            computed_value=computed,
            outcome=(
                ValidationOutcome.REPRODUCED
                if holds
                else ValidationOutcome.NOT_REPRODUCED
            ),
            diagnostics=tuple(diagnostics),
            denominator=1,
        )

    if claim_id == "DD-0213":
        components = _gain_components(decision, _CODE_TASKS, _PLOT_SIZES)
        threshold = _parameter(rule, ComparisonParameterName.MARKED_GAP_MINIMUM)
        minimum_gain = min(float(item["proxy_gain"]) for item in components)
        means_components = [
            {
                "task": task,
                "size": size,
                "metric": _TARGET_METRIC,
                "mean": _mean_value(means, size=size, task=task, metric=_TARGET_METRIC),
            }
            for task in _CODE_TASKS
            for size in _PLOT_SIZES
        ]
        holds = all(float(item["proxy_gain"]) > threshold for item in components)
        return _Evaluation(
            computed_value={
                "components": components,
                "small_scale_metric_means": means_components,
                "minimum_proxy_gain": minimum_gain,
                "marked_gap_minimum": threshold,
                "all_four_proxy_gains_positive": holds,
                "paper_legend_alias": _alias_payload(),
            },
            outcome=(
                ValidationOutcome.REPRODUCED
                if holds
                else ValidationOutcome.NOT_REPRODUCED
            ),
            diagnostics=tuple(
                (*diagnostics, f"minimum_proxy_gain={minimum_gain:.17g}")
            ),
            denominator=len(components),
        )

    if claim_id in {"DD-0221", "DD-0222"}:
        tasks = _CODE_TASKS if claim_id == "DD-0221" else _MATH_TASKS
        baseline = _parameter(rule, ComparisonParameterName.CHANCE_BASELINE)
        tolerance = _parameter(rule, ComparisonParameterName.TRIVIAL_TOLERANCE)
        strong_threshold = (
            _parameter(rule, ComparisonParameterName.STRONG_BASELINE_THRESHOLD)
            if claim_id == "DD-0221"
            else None
        )
        task_components: list[dict[str, object]] = []
        point_components: list[dict[str, object]] = []
        for task in tasks:
            primary_values = tuple(
                _decision_value(
                    decision,
                    size=size,
                    task=task,
                    target_metric=_TARGET_METRIC,
                    predictor_metric=_TARGET_METRIC,
                )
                for size in _DECISION_SIZES
            )
            continuous_values = tuple(
                _decision_value(
                    decision,
                    size=size,
                    task=task,
                    target_metric=_TARGET_METRIC,
                    predictor_metric=_CONTINUOUS_METRIC,
                )
                for size in _DECISION_SIZES
            )
            task_components.append(
                {
                    "task": task,
                    "target_metric": _TARGET_METRIC,
                    f"{_TARGET_METRIC}_mean": _mean(primary_values),
                    f"{_CONTINUOUS_METRIC}_mean": _mean(continuous_values),
                }
            )
            point_components.extend(
                {
                    "task": task,
                    "size": size,
                    "target_metric": _TARGET_METRIC,
                    _TARGET_METRIC: primary,
                    _CONTINUOUS_METRIC: continuous,
                }
                for size, primary, continuous in zip(
                    _DECISION_SIZES,
                    primary_values,
                    continuous_values,
                    strict=True,
                )
            )

        def satisfies(primary: float, continuous: float) -> bool:
            primary_near = abs(primary - baseline) <= tolerance
            if strong_threshold is None:
                return primary_near and abs(continuous - baseline) <= tolerance
            return primary_near and continuous >= strong_threshold

        values_to_check = (
            (
                (float(item[_TARGET_METRIC]), float(item[_CONTINUOUS_METRIC]))
                for item in point_components
            )
            if pointwise
            else (
                (
                    float(item[f"{_TARGET_METRIC}_mean"]),
                    float(item[f"{_CONTINUOUS_METRIC}_mean"]),
                )
                for item in task_components
            )
        )
        holds = all(
            satisfies(primary, continuous) for primary, continuous in values_to_check
        )
        computed = {
            "aggregation": "size_pointwise" if pointwise else "per_task_size_mean",
            "task_components": task_components,
            "size_components": point_components,
            "chance_baseline": baseline,
            "trivial_tolerance": tolerance,
            "satisfied": holds,
            "paper_legend_alias": _alias_payload(),
        }
        if strong_threshold is not None:
            computed["strong_baseline_threshold"] = strong_threshold
        return _Evaluation(
            computed_value=computed,
            outcome=(
                ValidationOutcome.REPRODUCED
                if holds
                else ValidationOutcome.NOT_REPRODUCED
            ),
            diagnostics=tuple((*diagnostics, f"aggregation={computed['aggregation']}")),
            denominator=len(point_components) if pointwise else len(task_components),
        )

    if claim_id == "DD-0224":
        tolerance = rule.absolute_tolerance
        if tolerance is None:
            raise ValueError("DD-0224 requires an absolute-tolerance rule")
        task_means = [
            {
                "task": task,
                "target_metric": _TARGET_METRIC,
                "predictor_metric": _CONTINUOUS_METRIC,
                "decision_accuracy_mean": _mean(
                    _decision_value(
                        decision,
                        size=size,
                        task=task,
                        target_metric=_TARGET_METRIC,
                        predictor_metric=_CONTINUOUS_METRIC,
                    )
                    for size in _DECISION_SIZES
                ),
            }
            for task in _CODE_TASKS
        ]
        holds = all(
            abs(float(item["decision_accuracy_mean"]) - 0.8) <= tolerance
            for item in task_means
        )
        return _Evaluation(
            computed_value={
                "task_components": task_means,
                "approximate_target": 0.8,
                "absolute_tolerance": tolerance,
                "satisfied": holds,
                "paper_legend_alias": _alias_payload(),
            },
            outcome=(
                ValidationOutcome.APPROXIMATELY_REPRODUCED
                if holds
                else ValidationOutcome.NOT_REPRODUCED
            ),
            diagnostics=tuple((*diagnostics, f"absolute_tolerance={tolerance:.17g}")),
            denominator=len(task_means),
        )

    if claim_id == "DD-0225":
        threshold = _parameter(rule, ComparisonParameterName.STRONG_BASELINE_THRESHOLD)
        components = [
            _component(
                task=task,
                size=size,
                predictor_metric=_CONTINUOUS_METRIC,
                decision_accuracy=_decision_value(
                    decision,
                    size=size,
                    task=task,
                    target_metric=_TARGET_METRIC,
                    predictor_metric=_CONTINUOUS_METRIC,
                ),
            )
            for task in _CODE_TASKS
            for size in _DECISION_SIZES
        ]
        target_means = [
            {
                "task": task,
                "size": size,
                "metric": metric,
                "mean": _mean_value(means, size=size, task=task, metric=metric),
            }
            for task in _CODE_TASKS
            for size in _DECISION_SIZES
            for metric in (_TARGET_METRIC, _CONTINUOUS_METRIC)
        ]
        holds = all(
            float(item["decision_accuracy"]) >= threshold for item in components
        )
        return _Evaluation(
            computed_value={
                "components": components,
                "supplied_metric_means": target_means,
                "strong_baseline_threshold": threshold,
                "all_continuous_proxy_accuracies_at_least_threshold": holds,
                "paper_legend_alias": _alias_payload(),
            },
            outcome=(
                ValidationOutcome.DIRECTIONALLY_CONSISTENT
                if holds
                else ValidationOutcome.NOT_REPRODUCED
            ),
            diagnostics=tuple(diagnostics),
            denominator=len(components),
        )

    if claim_id == "DD-0226":
        threshold = _parameter(rule, ComparisonParameterName.MARKED_GAP_MINIMUM)
        components = _gain_components(
            decision, (*_CODE_TASKS, *_MATH_TASKS), _DECISION_SIZES
        )
        if pointwise:
            comparisons = []
            for size in _DECISION_SIZES:
                code = tuple(
                    float(item["proxy_gain"])
                    for item in components
                    if item["size"] == size and item["task"] in _CODE_TASKS
                )
                math_values = tuple(
                    float(item["proxy_gain"])
                    for item in components
                    if item["size"] == size and item["task"] in _MATH_TASKS
                )
                comparisons.append(
                    {
                        "size": size,
                        "minimum_code_proxy_gain": min(code),
                        "maximum_math_proxy_gain": max(math_values),
                        "difference": min(code) - max(math_values),
                    }
                )
        else:
            task_means = {
                task: _mean(
                    float(item["proxy_gain"])
                    for item in components
                    if item["task"] == task
                )
                for task in (*_CODE_TASKS, *_MATH_TASKS)
            }
            comparisons = [
                {
                    "minimum_code_proxy_gain": min(
                        task_means[task] for task in _CODE_TASKS
                    ),
                    "maximum_math_proxy_gain": max(
                        task_means[task] for task in _MATH_TASKS
                    ),
                    "difference": min(task_means[task] for task in _CODE_TASKS)
                    - max(task_means[task] for task in _MATH_TASKS),
                    "task_mean_proxy_gains": task_means,
                }
            ]
        holds = all(float(item["difference"]) > threshold for item in comparisons)
        return _Evaluation(
            computed_value={
                "aggregation": "size_pointwise" if pointwise else "per_task_size_mean",
                "components": components,
                "comparisons": comparisons,
                "marked_gap_minimum": threshold,
                "satisfied": holds,
                "paper_legend_alias": _alias_payload(),
            },
            outcome=(
                ValidationOutcome.REPRODUCED
                if holds
                else ValidationOutcome.NOT_REPRODUCED
            ),
            diagnostics=tuple(
                (
                    *diagnostics,
                    f"aggregation={'size_pointwise' if pointwise else 'per_task_size_mean'}",
                )
            ),
            denominator=len(comparisons),
        )

    if claim_id == "DD-0227":
        threshold = _parameter(rule, ComparisonParameterName.ACCURACY_THRESHOLD)
        task_components = []
        for task in _MATH_TASKS:
            values = [
                {
                    "size": size,
                    "decision_accuracy": _decision_value(
                        decision,
                        size=size,
                        task=task,
                        target_metric=_CONTINUOUS_METRIC,
                        predictor_metric=_CONTINUOUS_METRIC,
                    ),
                }
                for size in _DECISION_SIZES
            ]
            task_components.append(
                {
                    "task": task,
                    "target_metric": _CONTINUOUS_METRIC,
                    "predictor_metric": _CONTINUOUS_METRIC,
                    "size_components": values,
                    "maximum_decision_accuracy": max(
                        float(item["decision_accuracy"]) for item in values
                    ),
                }
            )
        holds = all(
            float(item["maximum_decision_accuracy"]) >= threshold
            for item in task_components
        )
        return _Evaluation(
            computed_value={
                "task_components": task_components,
                "accuracy_threshold": threshold,
                "each_math_task_reaches_threshold": holds,
                "paper_legend_alias": _alias_payload(),
            },
            outcome=(
                ValidationOutcome.REPRODUCED
                if holds
                else ValidationOutcome.NOT_REPRODUCED
            ),
            diagnostics=tuple((*diagnostics, f"accuracy_threshold={threshold:.17g}")),
            denominator=len(task_components),
        )

    if claim_id == "DD-0413":
        nontrivial = _parameter(
            rule, ComparisonParameterName.NONTRIVIAL_ACCURACY_THRESHOLD
        )
        strong = _parameter(rule, ComparisonParameterName.STRONG_BASELINE_THRESHOLD)
        components = [
            _component(
                task=task,
                size=size,
                predictor_metric=_CONTINUOUS_METRIC,
                decision_accuracy=_decision_value(
                    decision,
                    size=size,
                    task=task,
                    target_metric=_TARGET_METRIC,
                    predictor_metric=_CONTINUOUS_METRIC,
                ),
            )
            for task in _CODE_TASKS
            for size in _PLOT_SIZES
        ]
        holds = all(
            float(item["decision_accuracy"]) > nontrivial
            and float(item["decision_accuracy"]) >= strong
            for item in components
        )
        return _Evaluation(
            computed_value={
                "components": components,
                "nontrivial_accuracy_threshold": nontrivial,
                "strong_baseline_threshold": strong,
                "all_four_bars_satisfied": holds,
                "paper_legend_alias": _alias_payload(),
            },
            outcome=(
                ValidationOutcome.REPRODUCED
                if holds
                else ValidationOutcome.NOT_REPRODUCED
            ),
            diagnostics=tuple(diagnostics),
            denominator=len(components),
        )

    if claim_id == "DD-0414":
        baseline = _parameter(rule, ComparisonParameterName.CHANCE_BASELINE)
        tolerance = _parameter(rule, ComparisonParameterName.TRIVIAL_TOLERANCE)
        components = [
            _component(
                task=task,
                size=size,
                predictor_metric=_CONTINUOUS_METRIC,
                decision_accuracy=_decision_value(
                    decision,
                    size=size,
                    task=task,
                    target_metric=_TARGET_METRIC,
                    predictor_metric=_CONTINUOUS_METRIC,
                ),
            )
            for task in _MATH_TASKS
            for size in _PLOT_SIZES
        ]
        holds = all(
            abs(float(item["decision_accuracy"]) - baseline) <= tolerance
            for item in components
        )
        return _Evaluation(
            computed_value={
                "components": components,
                "chance_baseline": baseline,
                "trivial_tolerance": tolerance,
                "all_four_bars_near_baseline": holds,
                "paper_legend_alias": _alias_payload(),
            },
            outcome=(
                ValidationOutcome.REPRODUCED
                if holds
                else ValidationOutcome.NOT_REPRODUCED
            ),
            diagnostics=tuple((*diagnostics, f"trivial_tolerance={tolerance:.17g}")),
            denominator=len(components),
        )
    raise ValueError(f"no math/code implementation for {attempt.id}")


def _selection_scope(
    attempt: AttemptSpec,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    if attempt.claim_id in {"DD-0017", "DD-0018"}:
        return attempt.task_ids, ("4M",), _TARGET_METRIC
    if attempt.claim_id == "DD-0213":
        return _CODE_TASKS, _PLOT_SIZES, _TARGET_METRIC
    if attempt.claim_id == "DD-0227":
        return _MATH_TASKS, _DECISION_SIZES, _CONTINUOUS_METRIC
    if attempt.claim_id in {"DD-0413", "DD-0414"}:
        return _PLOT_TASKS, _PLOT_SIZES, _TARGET_METRIC
    return _PLOT_TASKS, _DECISION_SIZES, _TARGET_METRIC


def _decision_selection(
    table: _LoadedTable, attempt: AttemptSpec, columns: tuple[str, ...]
) -> RowSelection:
    tasks, sizes, target_metric = _selection_scope(attempt)
    selected = table.frame.loc[
        table.frame["task"].isin(tasks)
        & table.frame["size"].isin(sizes)
        & table.frame["target_ranking"].eq(target_metric)
    ]
    return RowSelection(
        logical_table_id=table.spec.id,
        columns=columns,
        predicates=(
            RowPredicate(column="size", operator=PredicateOperator.IN, value=sizes),
            RowPredicate(column="task", operator=PredicateOperator.IN, value=tasks),
            RowPredicate(
                column="target_ranking",
                operator=PredicateOperator.EQ,
                value=target_metric,
            ),
        ),
        local_parquet_sha256=table.sha256,
        selected_row_count=len(selected),
        selected_key_sha256=_selected_key_sha256(selected, _DECISION_KEY_COLUMNS),
    )


def _means_selection(
    table: _LoadedTable, attempt: AttemptSpec, columns: tuple[str, ...]
) -> RowSelection:
    tasks = _CODE_TASKS
    sizes = _PLOT_SIZES if attempt.claim_id == "DD-0213" else _DECISION_SIZES
    selected = table.frame.loc[
        table.frame["task"].isin(tasks) & table.frame["size"].isin(sizes)
    ]
    return RowSelection(
        logical_table_id=table.spec.id,
        columns=columns,
        predicates=(
            RowPredicate(column="size", operator=PredicateOperator.IN, value=sizes),
            RowPredicate(column="task", operator=PredicateOperator.IN, value=tasks),
        ),
        local_parquet_sha256=table.sha256,
        selected_row_count=len(selected),
        selected_key_sha256=_selected_key_sha256(selected, _MEANS_KEY_COLUMNS),
    )


def _compute_selection(
    table: _LoadedTable,
    evidence: _ComputeEvidence,
    columns: tuple[str, ...],
) -> RowSelection:
    recipes = tuple(
        sorted(str(value) for value in table.frame["data"].dropna().unique())
    )
    steps = () if evidence.steps is None else evidence.steps
    return RowSelection(
        logical_table_id=table.spec.id,
        columns=columns,
        predicates=(
            RowPredicate(column="data", operator=PredicateOperator.IN, value=recipes),
            RowPredicate(
                column="params",
                operator=PredicateOperator.IN,
                value=("4M", "1B"),
            ),
            RowPredicate(
                column="step",
                operator=PredicateOperator.IN,
                value=steps if steps else (0,),
            ),
        ),
        local_parquet_sha256=table.sha256,
        selected_row_count=len(evidence.selection_frame),
        selected_key_sha256=_selected_key_sha256(
            evidence.selection_frame, _COMPUTE_KEY_COLUMNS
        ),
    )


def _row_selections(
    *,
    attempt: AttemptSpec,
    decision: _LoadedTable,
    means: _LoadedTable,
    compute_table: _LoadedTable,
    compute: _ComputeEvidence,
) -> tuple[RowSelection, ...]:
    selections: list[RowSelection] = []
    for attempt_input in attempt.inputs:
        if attempt_input.table_id == _DECISION_INPUT_ID:
            selections.append(
                _decision_selection(decision, attempt, attempt_input.columns)
            )
        elif attempt_input.table_id == _MEANS_INPUT_ID:
            selections.append(_means_selection(means, attempt, attempt_input.columns))
        elif attempt_input.table_id == _COMPUTE_INPUT_ID:
            selections.append(
                _compute_selection(compute_table, compute, attempt_input.columns)
            )
        else:
            raise ValueError(
                f"math/code attempt {attempt.id} references unsupported input "
                f"{attempt_input.table_id}"
            )
    return tuple(selections)


def _plot_series(attempt: AttemptSpec, decision: pd.DataFrame) -> PlotSeries:
    if len(attempt.plot_series_ids) != 1:
        raise ValueError(f"{attempt.id} must declare exactly one plot-series ID")
    points = tuple(
        PlotPoint(
            dimensions=(
                DimensionValue(name="task", value=task),
                DimensionValue(name="size", value=size),
                DimensionValue(name="predictor_metric", value=metric),
                DimensionValue(name="target_metric", value=_TARGET_METRIC),
            ),
            measures=(
                MeasureValue(
                    name="decision_accuracy",
                    value=_decision_value(
                        decision,
                        size=size,
                        task=task,
                        target_metric=_TARGET_METRIC,
                        predictor_metric=metric,
                    ),
                ),
            ),
        )
        for task in _PLOT_TASKS
        for size in _PLOT_SIZES
        for metric in _PLOT_METRICS
    )
    return PlotSeries(
        id=attempt.plot_series_ids[0],
        figure="fig:math_and_code",
        panel=attempt.claim_id,
        semantic_kind=(
            "math/code decision accuracy using actual metric names; paper legend "
            "alias Correct Prob unresolved"
        ),
        x_axis=AxisSpec(
            measure="decision_accuracy", scale=AxisScale.LINEAR, unit="proportion"
        ),
        y_axis=AxisSpec(
            measure="decision_accuracy", scale=AxisScale.LINEAR, unit="proportion"
        ),
        dimensions=("task", "size", "predictor_metric", "target_metric"),
        measures=("decision_accuracy",),
        attempt_id=attempt.id,
        counts=(
            NamedCount(name="source_rows", value=8),
            NamedCount(name="points", value=16),
            NamedCount(name="tasks", value=4),
            NamedCount(name="sizes", value=2),
            NamedCount(name="predictor_metrics", value=2),
        ),
        points=points,
    )


def _comparison_sensitivities(
    attempt: AttemptSpec, rule: ComparisonRule
) -> tuple[tuple[str, ComparisonRule], ...]:
    sensitivities: list[tuple[str, ComparisonRule]] = []
    if rule.absolute_tolerance is not None:
        for grid_index, value in enumerate(rule.threshold_grid, start=1):
            if value == rule.absolute_tolerance:
                continue
            sensitivity_id = (
                f"{attempt.claim_id.lower()}-comparison-absolute-tolerance-grid-"
                f"{grid_index}"
            )
            sensitivities.append(
                (
                    sensitivity_id,
                    rule.model_copy(update={"absolute_tolerance": value}),
                )
            )
    for parameter_index, parameter in enumerate(rule.parameters):
        for grid_index, value in enumerate(parameter.sensitivity_grid, start=1):
            if value == parameter.default:
                continue
            sensitivity_id = (
                f"{attempt.claim_id.lower()}-comparison-"
                f"{parameter.name.value.replace('_', '-')}-grid-{grid_index}"
            )
            parameters = list(rule.parameters)
            parameters[parameter_index] = parameter.model_copy(
                update={"default": value}
            )
            sensitivities.append(
                (
                    sensitivity_id,
                    rule.model_copy(update={"parameters": tuple(parameters)}),
                )
            )
    return tuple(sensitivities)


def _not_assessable_result(
    *,
    attempt: AttemptSpec,
    claim: PaperClaim,
    rule: ComparisonRule,
    selections: tuple[RowSelection, ...],
    missing_groups: tuple[str, ...],
    attempt_id: str | None = None,
) -> AttemptResult:
    result_id = attempt.id if attempt_id is None else attempt_id
    return AttemptResult(
        attempt_id=result_id,
        claim_id=attempt.claim_id,
        role=(
            AttemptRole.DEFAULT if result_id == attempt.id else AttemptRole.SENSITIVITY
        ),
        parent_attempt_id=None if result_id == attempt.id else attempt.id,
        evidence_level=EvidenceLevel.AUTHOR_DERIVED_AGGREGATE,
        comparison_rule_id=attempt.comparison_rule_id,
        comparison_rule_version=rule.version,
        transformation_ids=attempt.transformation_ids,
        row_selections=selections,
        target_value=claim.paper_target,
        computed_value={
            "status": "not_assessable",
            "selected_row_counts": {
                selection.logical_table_id: selection.selected_row_count
                for selection in selections
            },
            "paper_legend_alias": _alias_payload(),
        },
        missing_groups=missing_groups,
        outcome=ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED,
        diagnostics=(
            "Aggregate cube validation failed for a declared attempt input.",
            "verification_scope=supplied_aggregate_tables",
            "paper_legend_alias_status=unresolved",
        ),
        limitations=(_AGGREGATE_LIMITATION, _ALIAS_LIMITATION),
    )


def _result(
    *,
    attempt: AttemptSpec,
    claim: PaperClaim,
    rule: ComparisonRule,
    selections: tuple[RowSelection, ...],
    evaluation: _Evaluation,
    series: PlotSeries | None,
    attempt_id: str | None = None,
    extra_diagnostics: tuple[str, ...] = (),
) -> AttemptResult:
    result_id = attempt.id if attempt_id is None else attempt_id
    limitations = [_AGGREGATE_LIMITATION, _ALIAS_LIMITATION]
    if attempt.claim_id == "DD-0225":
        limitations.append(
            "Aggregate data cannot establish recipe separation or a noise floor; "
            "it supports only the directional decision-accuracy comparison."
        )
    return AttemptResult(
        attempt_id=result_id,
        claim_id=attempt.claim_id,
        role=(
            AttemptRole.DEFAULT if result_id == attempt.id else AttemptRole.SENSITIVITY
        ),
        parent_attempt_id=None if result_id == attempt.id else attempt.id,
        evidence_level=EvidenceLevel.AUTHOR_DERIVED_AGGREGATE,
        comparison_rule_id=attempt.comparison_rule_id,
        comparison_rule_version=rule.version,
        transformation_ids=attempt.transformation_ids,
        row_selections=selections,
        target_value=claim.paper_target,
        computed_value=evaluation.computed_value,
        denominator=evaluation.denominator,
        outcome=evaluation.outcome,
        diagnostics=(
            *evaluation.diagnostics,
            "verification_scope=supplied_aggregate_tables",
            *(
                f"{selection.logical_table_id}_row_count={selection.selected_row_count}"
                for selection in selections
            ),
            *(
                f"{selection.logical_table_id}_sha256={selection.local_parquet_sha256}"
                for selection in selections
            ),
            *(
                f"{selection.logical_table_id}_selected_key_sha256={selection.selected_key_sha256}"
                for selection in selections
            ),
            *extra_diagnostics,
        ),
        limitations=tuple(limitations),
        plot_series_ids=()
        if series is None or result_id != attempt.id
        else (series.id,),
    )


def run_math_code_attempts(
    *,
    repository_root: Path,
    data_root: Path,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
) -> tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]]:
    """Verify math/code claims against the two supplied aggregate cubes."""
    del repository_root
    attempts = tuple(
        sorted(
            (
                item
                for item in contract.attempts
                if item.analysis_id is AnalysisId.MATH_CODE
            ),
            key=lambda item: item.id,
        )
    )
    if not attempts:
        return (), ()
    unsupported = tuple(
        item.id for item in attempts if item.claim_id not in _SUPPORTED_CLAIMS
    )
    if unsupported:
        raise ValueError(f"unsupported math/code attempts: {unsupported!r}")

    decision = _load_table(
        data_root=data_root,
        contract=contract,
        input_identities=input_identities,
        table_id=_DECISION_INPUT_ID,
        exact_columns=_DECISION_COLUMNS,
    )
    means = _load_table(
        data_root=data_root,
        contract=contract,
        input_identities=input_identities,
        table_id=_MEANS_INPUT_ID,
        exact_columns=_MEANS_COLUMNS,
    )
    compute_table = _load_table(
        data_root=data_root,
        contract=contract,
        input_identities=input_identities,
        table_id=_COMPUTE_INPUT_ID,
        read_columns=("data", "params", "step", "compute"),
        filters=[("params", "in", ["4M", "1B"])],
    )
    decision_expected = tuple(
        (size, task, target)
        for size in _DECISION_SIZES
        for task in _ALL_TASKS
        for target in _TARGET_RANKINGS
    )
    means_expected = tuple((size, task) for size in _MEANS_SIZES for task in _ALL_TASKS)
    decision_missing = _cube_anomalies(
        decision,
        key_columns=_DECISION_KEY_COLUMNS,
        expected_keys=decision_expected,
        numeric_columns=(
            _CONTINUOUS_METRIC,
            "logits_per_char_corr",
            _TARGET_METRIC,
        ),
    )
    means_missing = _cube_anomalies(
        means,
        key_columns=_MEANS_KEY_COLUMNS,
        expected_keys=means_expected,
        numeric_columns=(
            _TARGET_METRIC,
            _CONTINUOUS_METRIC,
            "logits_per_char_corr",
        ),
    )
    compute = _compute_evidence(compute_table)
    claims = {item.id: item for item in registry.claims}
    rules = {item.id: item for item in contract.comparison_rules}
    results: list[AttemptResult] = []
    series_values: list[PlotSeries] = []

    for attempt in attempts:
        if attempt.claim_id not in claims:
            raise ValueError(
                f"math/code attempt references unknown claim {attempt.claim_id}"
            )
        rule = _rule(rules, attempt)
        selections = _row_selections(
            attempt=attempt,
            decision=decision,
            means=means,
            compute_table=compute_table,
            compute=compute,
        )
        missing_groups = list(decision_missing)
        input_ids = {item.table_id for item in attempt.inputs}
        if _MEANS_INPUT_ID in input_ids:
            missing_groups.extend(means_missing)
        if _COMPUTE_INPUT_ID in input_ids:
            missing_groups.extend(compute.missing_groups)
        missing = tuple(dict.fromkeys(missing_groups))
        sensitivity_rules = _comparison_sensitivities(attempt, rule)
        pointwise_id = _POINTWISE_SENSITIVITIES.get(attempt.claim_id)
        expected_sensitivity_ids = {
            *(item[0] for item in sensitivity_rules),
            *((pointwise_id,) if pointwise_id is not None else ()),
        }
        if set(attempt.sensitivity_ids) != expected_sensitivity_ids:
            raise ValueError(
                f"{attempt.id} configured sensitivities differ from the frozen "
                f"math/code rules: configured={attempt.sensitivity_ids!r}, "
                f"expected={tuple(sorted(expected_sensitivity_ids))!r}"
            )
        claim = claims[attempt.claim_id]
        if missing:
            results.append(
                _not_assessable_result(
                    attempt=attempt,
                    claim=claim,
                    rule=rule,
                    selections=selections,
                    missing_groups=missing,
                )
            )
            for sensitivity_id in attempt.sensitivity_ids:
                results.append(
                    _not_assessable_result(
                        attempt=attempt,
                        claim=claim,
                        rule=rule,
                        selections=selections,
                        missing_groups=missing,
                        attempt_id=sensitivity_id,
                    )
                )
            continue

        plot = (
            _plot_series(attempt, decision.frame) if attempt.plot_series_ids else None
        )
        evaluation = _evaluate(
            attempt=attempt,
            rule=rule,
            decision=decision.frame,
            means=means.frame,
            compute=compute,
        )
        results.append(
            _result(
                attempt=attempt,
                claim=claim,
                rule=rule,
                selections=selections,
                evaluation=evaluation,
                series=plot,
            )
        )
        if plot is not None:
            series_values.append(plot)
        for sensitivity_id, sensitivity_rule in sensitivity_rules:
            sensitivity = _evaluate(
                attempt=attempt,
                rule=sensitivity_rule,
                decision=decision.frame,
                means=means.frame,
                compute=compute,
            )
            results.append(
                _result(
                    attempt=attempt,
                    claim=claim,
                    rule=rule,
                    selections=selections,
                    evaluation=sensitivity,
                    series=None,
                    attempt_id=sensitivity_id,
                    extra_diagnostics=("sensitivity_kind=comparison_parameter",),
                )
            )
        if pointwise_id is not None:
            sensitivity = _evaluate(
                attempt=attempt,
                rule=rule,
                decision=decision.frame,
                means=means.frame,
                compute=compute,
                pointwise=True,
            )
            results.append(
                _result(
                    attempt=attempt,
                    claim=claim,
                    rule=rule,
                    selections=selections,
                    evaluation=sensitivity,
                    series=None,
                    attempt_id=pointwise_id,
                    extra_diagnostics=("sensitivity_kind=size_pointwise",),
                )
            )

    return tuple(sorted(results, key=lambda item: item.attempt_id)), tuple(
        sorted(series_values, key=lambda item: item.id)
    )


__all__ = ["run_math_code_attempts"]

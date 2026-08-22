from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from datadec.paper.models import (
    AnalysisId,
    AttemptResult,
    AttemptRole,
    AttemptSpec,
    AxisScale,
    AxisSpec,
    CheckpointRule,
    CheckpointSelection,
    ClaimRegistry,
    ComparisonPredicate,
    ComparisonParameterName,
    ComparisonRule,
    ContentIdentity,
    DimensionValue,
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
from datadec.paper.single_scale import (
    DEFAULT_TASK_GROUPING,
    CheckpointRows,
    MetricObservation,
    PredictionRanking,
    RankedRecipe,
    SingleScaleUniverse,
    TargetRanking,
    aggregate_checkpoint,
    analyze_prediction_checkpoint,
    compare_rankings,
    observations_from_olmes_frame,
)

_OLMES_TABLE_ID = "olmes_aggregate"
_PRIMARY_METRIC = "primary_metric"
_TARGET_SIZE = "1B"
_HEADLINE_PREDICTION_SIZE = "150M"
_PAPER_STEP = 38_157
_HEADLINE_TARGET = 0.80
_AGGREGATE_SERIES_ID = "dd-0169-paper-analog"
_PER_TASK_SERIES_ID = "dd-0148-paper-analog"
_SELECTED_KEY_COLUMNS = ("params", "data", "seed", "step", "task")

_PAPER_RECIPES = (
    "C4",
    "DCLM-Baseline",
    "DCLM-Baseline (QC 10%)",
    "DCLM-Baseline (QC 20%)",
    "DCLM-Baseline (QC 7%, FW2)",
    "DCLM-Baseline (QC 7%, FW3)",
    "DCLM-Baseline (QC FW 10%)",
    "DCLM-Baseline (QC FW 3%)",
    "DCLM-Baseline 25% / Dolma 75%",
    "DCLM-Baseline 50% / Dolma 50%",
    "DCLM-Baseline 75% / Dolma 25%",
    "Dolma1.6++",
    "Dolma1.7",
    "Dolma1.7 (no Flan)",
    "Dolma1.7 (no Reddit)",
    "Dolma1.7 (no code)",
    "Dolma1.7 (no math, code)",
    "Falcon",
    "Falcon+CC",
    "Falcon+CC (QC 10%)",
    "Falcon+CC (QC 20%)",
    "Falcon+CC (QC Orig 10%)",
    "Falcon+CC (QC Tulu 10%)",
    "FineWeb-Edu",
    "FineWeb-Pro",
)
_TARGET_SEEDS = ("default", "large aux 2", "large aux 3")
_PREDICTION_SEEDS = ("default", "small aux 2", "small aux 3")
_MODEL_SIZE_ORDER = (
    "4M",
    "6M",
    "8M",
    "10M",
    "14M",
    "16M",
    "20M",
    "60M",
    "90M",
    "150M",
    "300M",
    "530M",
    "750M",
    "1B",
)
_LOGICAL_TASKS = (*DEFAULT_TASK_GROUPING.non_mmlu_tasks, "mmlu")


@dataclass(frozen=True, slots=True)
class _ComputeEquivalencePoint:
    model_size: str
    step: int
    compute: float
    decision_accuracy: float


def _parameter(rule: ComparisonRule, name: ComparisonParameterName) -> float:
    return rule.parameter(name).default


def _comparison_sensitivity_rules(
    attempt: AttemptSpec, rule: ComparisonRule
) -> tuple[tuple[str, ComparisonParameterName, float, ComparisonRule], ...]:
    sensitivities: list[tuple[str, ComparisonParameterName, float, ComparisonRule]] = []
    for parameter_index, parameter in enumerate(rule.parameters):
        for grid_index, value in enumerate(parameter.sensitivity_grid, start=1):
            if value == parameter.default:
                continue
            sensitivity_id = (
                f"{attempt.claim_id.lower()}-comparison-"
                f"{parameter.name.value.replace('_', '-')}-grid-{grid_index}"
            )
            if sensitivity_id not in attempt.sensitivity_ids:
                raise ValueError(
                    f"{attempt.id} does not declare comparison sensitivity "
                    f"{sensitivity_id}"
                )
            parameters = list(rule.parameters)
            parameters[parameter_index] = parameter.model_copy(
                update={"default": value}
            )
            sensitivities.append(
                (
                    sensitivity_id,
                    parameter.name,
                    value,
                    rule.model_copy(update={"parameters": tuple(parameters)}),
                )
            )
    return tuple(sensitivities)


def _contract_with_rule(
    contract: PaperValidationContract, rule: ComparisonRule
) -> PaperValidationContract:
    return contract.model_copy(
        update={
            "comparison_rules": tuple(
                rule if candidate.id == rule.id else candidate
                for candidate in contract.comparison_rules
            )
        }
    )


def _comparison_sensitivity_result(
    result: AttemptResult,
    *,
    sensitivity_id: str,
    parameter_name: ComparisonParameterName,
    parameter_value: float,
) -> AttemptResult:
    return result.model_copy(
        update={
            "attempt_id": sensitivity_id,
            "role": AttemptRole.SENSITIVITY,
            "parent_attempt_id": result.attempt_id,
            "diagnostics": (
                *result.diagnostics,
                f"comparison_parameter={parameter_name.value}",
                f"comparison_parameter_value={parameter_value:.17g}",
            ),
            "plot_series_ids": (),
        }
    )


def _linear_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    x_values = tuple(xs)
    y_values = tuple(ys)
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("linear slope requires paired observations")
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    denominator = math.fsum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0:
        return 0.0
    return (
        math.fsum(
            (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True)
        )
        / denominator
    )


def _rank(values: tuple[float, ...]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return tuple(ranks)


def _spearman(xs: Iterable[float], ys: Iterable[float]) -> float:
    x_values = tuple(xs)
    y_values = tuple(ys)
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Spearman correlation requires paired observations")
    x_ranks = _rank(x_values)
    y_ranks = _rank(y_values)
    x_mean = _mean(x_ranks)
    y_mean = _mean(y_ranks)
    numerator = math.fsum(
        (x - x_mean) * (y - y_mean) for x, y in zip(x_ranks, y_ranks, strict=True)
    )
    denominator = math.sqrt(
        math.fsum((value - x_mean) ** 2 for value in x_ranks)
        * math.fsum((value - y_mean) ** 2 for value in y_ranks)
    )
    return 0.0 if denominator == 0 else numerator / denominator


def _dimensions(point: PlotPoint) -> dict[str, str | int | float | bool | None]:
    return {item.name: item.value for item in point.dimensions}


def _measures(point: PlotPoint) -> dict[str, float]:
    return {item.name: item.value for item in point.measures}


def _sha256_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _mean(values: Iterable[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("at least one value is required")
    return math.fsum(ordered) / len(ordered)


def _sample_sd(values: Iterable[float]) -> float:
    ordered = tuple(sorted(values))
    if len(ordered) < 2:
        raise ValueError("sample standard deviation requires at least two values")
    mean = _mean(ordered)
    return math.sqrt(
        math.fsum((value - mean) ** 2 for value in ordered) / (len(ordered) - 1)
    )


def _log_compute_bucket_index(
    compute: float,
    *,
    target_model_compute: float,
    bin_width: float,
) -> int:
    if not math.isfinite(compute) or compute <= 0:
        raise ValueError("compute bucket assignment requires positive finite compute")
    if not math.isfinite(target_model_compute) or target_model_compute <= 0:
        raise ValueError("target model compute must be positive and finite")
    if not math.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("compute log10 bin width must be positive and finite")
    normalized_compute = compute / target_model_compute
    if normalized_compute == 0 or not math.isfinite(normalized_compute):
        raise ValueError("normalized compute must be positive and finite")
    return math.floor(math.log10(normalized_compute) / bin_width)


def _compute_equivalence_points(
    series: PlotSeries,
) -> tuple[_ComputeEquivalencePoint, ...]:
    points: list[_ComputeEquivalencePoint] = []
    for point in series.points:
        dimensions = _dimensions(point)
        measures = _measures(point)
        missing = tuple(
            name
            for name, values in (
                ("model_size", dimensions),
                ("step", dimensions),
                ("compute", measures),
                ("decision_accuracy", measures),
            )
            if name not in values
        )
        if missing:
            raise ValueError(
                f"compute-equivalence point is missing fields: {missing!r}"
            )
        model_size = dimensions["model_size"]
        step = dimensions["step"]
        if not isinstance(model_size, str) or not model_size:
            raise ValueError("compute-equivalence model size must be a nonempty string")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("compute-equivalence step must be a non-negative integer")
        points.append(
            _ComputeEquivalencePoint(
                model_size=model_size,
                step=step,
                compute=measures["compute"],
                decision_accuracy=measures["decision_accuracy"],
            )
        )
    return tuple(points)


def _compute_equivalence(
    points: Iterable[_ComputeEquivalencePoint],
    *,
    target_model_compute: float,
    bin_width: float,
    minimum_difference: float,
    preexcluded_zero_compute_count: int = 0,
) -> tuple[dict[str, object], tuple[NamedCount, ...]]:
    if not math.isfinite(minimum_difference):
        raise ValueError("minimum accuracy difference must be finite")
    if preexcluded_zero_compute_count < 0:
        raise ValueError("preexcluded zero-compute count must be non-negative")
    values = tuple(points)
    seen: set[tuple[str, int]] = set()
    positive: list[_ComputeEquivalencePoint] = []
    zero_compute_count = preexcluded_zero_compute_count
    for point in values:
        key = (point.model_size, point.step)
        if key in seen:
            raise ValueError(f"duplicate compute-equivalence checkpoint point: {key!r}")
        seen.add(key)
        if not point.model_size:
            raise ValueError("compute-equivalence model size must not be empty")
        if (
            isinstance(point.step, bool)
            or not isinstance(point.step, int)
            or point.step < 0
        ):
            raise ValueError("compute-equivalence step must be a non-negative integer")
        if not math.isfinite(point.compute):
            raise ValueError("compute-equivalence compute must be finite")
        if point.compute < 0:
            raise ValueError("compute-equivalence compute must be non-negative")
        if not math.isfinite(point.decision_accuracy):
            raise ValueError("compute-equivalence decision accuracy must be finite")
        if not 0 <= point.decision_accuracy <= 1:
            raise ValueError("compute-equivalence decision accuracy must be in [0, 1]")
        if point.compute == 0:
            zero_compute_count += 1
        else:
            positive.append(point)

    final_by_size: dict[str, _ComputeEquivalencePoint] = {}
    for point in positive:
        current = final_by_size.get(point.model_size)
        if current is None or point.step > current.step:
            final_by_size[point.model_size] = point
    intermediate_by_bucket: dict[tuple[int, str], list[_ComputeEquivalencePoint]] = {}
    for point in positive:
        if point == final_by_size[point.model_size]:
            continue
        bucket = _log_compute_bucket_index(
            point.compute,
            target_model_compute=target_model_compute,
            bin_width=bin_width,
        )
        intermediate_by_bucket.setdefault((bucket, point.model_size), []).append(point)

    model_order = {size: index for index, size in enumerate(_MODEL_SIZE_ORDER)}

    def model_key(model_size: str) -> tuple[int, str]:
        return model_order.get(model_size, len(model_order)), model_size

    matches: list[dict[str, object]] = []
    same_size_pair_count = 0
    ordered_finals = tuple(
        sorted(final_by_size.items(), key=lambda item: model_key(item[0]))
    )
    for (bucket, intermediate_size), contributing in sorted(
        intermediate_by_bucket.items(),
        key=lambda item: (item[0][0], model_key(item[0][1])),
    ):
        ordered_contributing = tuple(sorted(contributing, key=lambda point: point.step))
        intermediate_accuracy = _mean(
            point.decision_accuracy for point in ordered_contributing
        )
        intermediate_computes = tuple(point.compute for point in ordered_contributing)
        for final_size, final in ordered_finals:
            final_bucket = _log_compute_bucket_index(
                final.compute,
                target_model_compute=target_model_compute,
                bin_width=bin_width,
            )
            if final_bucket != bucket:
                continue
            if final_size == intermediate_size:
                same_size_pair_count += 1
                continue
            all_compute = (*intermediate_computes, final.compute)
            compute_minimum = min(all_compute)
            compute_maximum = max(all_compute)
            difference = intermediate_accuracy - final.decision_accuracy
            matches.append(
                {
                    "bin_index": bucket,
                    "bin_lower_edge": 10.0 ** (bucket * bin_width),
                    "bin_upper_edge": 10.0 ** ((bucket + 1) * bin_width),
                    "intermediate_model_size": intermediate_size,
                    "intermediate_steps": [
                        point.step for point in ordered_contributing
                    ],
                    "intermediate_checkpoint_count": len(ordered_contributing),
                    "final_model_size": final_size,
                    "final_step": final.step,
                    "intermediate_compute_minimum": min(intermediate_computes),
                    "intermediate_compute_maximum": max(intermediate_computes),
                    "final_compute": final.compute,
                    "compute_minimum": compute_minimum,
                    "compute_maximum": compute_maximum,
                    "compute_ratio": compute_maximum / compute_minimum,
                    "intermediate_accuracy": intermediate_accuracy,
                    "final_accuracy": final.decision_accuracy,
                    "accuracy_difference": difference,
                }
            )

    differences = tuple(float(match["accuracy_difference"]) for match in matches)
    ordered_differences = tuple(sorted(differences))
    middle = len(ordered_differences) // 2
    median_difference = (
        None
        if not ordered_differences
        else ordered_differences[middle]
        if len(ordered_differences) % 2
        else _mean(ordered_differences[middle - 1 : middle + 1])
    )
    passing_group_count = sum(
        difference >= minimum_difference for difference in differences
    )
    matched_bin_count = len({int(match["bin_index"]) for match in matches})
    satisfied = bool(matches) and passing_group_count == len(matches)
    computed_value: dict[str, object] = {
        "compute_log10_bin_width": bin_width,
        "target_model_compute": target_model_compute,
        "matched_groups": matches,
        "matched_bin_count": matched_bin_count,
        "matched_group_count": len(matches),
        "passing_group_count": passing_group_count,
        "minimum_accuracy_difference": min(differences) if differences else None,
        "mean_accuracy_difference": _mean(differences) if differences else None,
        "median_accuracy_difference": median_difference,
        "minimum_allowed_difference": minimum_difference,
        "zero_compute_checkpoint_count": zero_compute_count,
        "same_size_pair_count": same_size_pair_count,
        "satisfied": satisfied,
    }
    return computed_value, (
        NamedCount(name="zero_compute_checkpoints", value=zero_compute_count),
        NamedCount(name="same_size_pairs", value=same_size_pair_count),
    )


def _attempt(
    contract: PaperValidationContract,
    attempt_id: str,
) -> AttemptSpec:
    try:
        return next(item for item in contract.attempts if item.id == attempt_id)
    except StopIteration as error:
        raise ValueError(f"validation contract has no attempt {attempt_id}") from error


def _claim(registry: ClaimRegistry, claim_id: str) -> PaperClaim:
    try:
        return next(item for item in registry.claims if item.id == claim_id)
    except StopIteration as error:
        raise ValueError(f"claim registry has no claim {claim_id}") from error


def _comparison_rule(
    contract: PaperValidationContract,
    attempt: AttemptSpec,
) -> ComparisonRule:
    try:
        return next(
            item
            for item in contract.comparison_rules
            if item.id == attempt.comparison_rule_id
        )
    except StopIteration as error:
        raise ValueError(
            f"validation contract has no comparison rule {attempt.comparison_rule_id}"
        ) from error


def _olmes_input(contract: PaperValidationContract) -> tuple[Path, tuple[str, ...]]:
    try:
        table = next(item for item in contract.inputs if item.id == _OLMES_TABLE_ID)
    except StopIteration as error:
        raise ValueError("validation contract has no olmes_aggregate input") from error
    return Path(table.path), table.columns


def _load_observations(
    *,
    data_root: Path,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
) -> tuple[tuple[MetricObservation, ...], str, tuple[str, ...]]:
    relative_path, declared_columns = _olmes_input(contract)
    path = data_root / relative_path
    identity = input_identities.get(_OLMES_TABLE_ID)
    if identity is None:
        raise ValueError("input identities have no olmes_aggregate entry")
    if identity.id != _OLMES_TABLE_ID:
        raise ValueError("olmes_aggregate input identity has the wrong logical ID")

    digest = _sha256_file(path)
    if identity.sha256 != digest:
        raise ValueError(
            "olmes_aggregate identity differs from the configured local Parquet"
        )
    implemented_attempts = tuple(
        attempt
        for attempt in contract.attempts
        if attempt.analysis_id in {AnalysisId.SINGLE_SCALE, AnalysisId.PER_TASK}
    )
    if any(
        len(attempt.inputs) != 1 or attempt.inputs[0].table_id != _OLMES_TABLE_ID
        for attempt in implemented_attempts
    ):
        raise ValueError(
            "implemented single-scale attempts require only olmes_aggregate"
        )
    if not implemented_attempts:
        raise ValueError("validation contract has no single-scale attempts")
    required_columns = implemented_attempts[0].inputs[0].columns
    if any(
        attempt.inputs[0].columns != required_columns
        for attempt in implemented_attempts[1:]
    ):
        raise ValueError("implemented single-scale attempts require the same columns")
    if any(column not in declared_columns for column in required_columns):
        raise ValueError("headline attempt columns exceed the olmes_aggregate contract")
    schema_columns = set(pq.ParquetFile(path).schema_arrow.names)
    missing_columns = tuple(sorted(set(required_columns).difference(schema_columns)))
    if missing_columns:
        raise ValueError(
            f"olmes_aggregate is missing configured columns: {missing_columns!r}"
        )

    frame = pd.read_parquet(path, columns=list(required_columns))
    if _sha256_file(path) != digest:
        raise RuntimeError("olmes_aggregate changed while it was being read")
    missing_accuracy_count = int(frame[_PRIMARY_METRIC].isna().sum())
    if missing_accuracy_count:
        raise ValueError(
            "olmes_aggregate contains missing primary-metric accuracy: "
            f"count={missing_accuracy_count}"
        )
    observations = observations_from_olmes_frame(
        frame,
        metric_columns=(_PRIMARY_METRIC,),
    )
    return observations, digest, required_columns


def _seeds_for_size(model_size: str) -> tuple[str, ...]:
    return _TARGET_SEEDS if model_size == _TARGET_SIZE else _PREDICTION_SEEDS


def _universe(model_size: str) -> SingleScaleUniverse:
    return SingleScaleUniverse(
        model_size=model_size,
        recipes=_PAPER_RECIPES,
        seeds=_seeds_for_size(model_size),
        source_tasks=DEFAULT_TASK_GROUPING.source_tasks,
        metrics=(_PRIMARY_METRIC,),
    )


def _all_common_complete_checkpoints(
    observations: Iterable[MetricObservation],
    universe: SingleScaleUniverse,
) -> tuple[CheckpointRows, ...]:
    recipe_set = set(universe.recipes)
    seed_set = set(universe.seeds)
    task_set = set(universe.source_tasks)
    expected_keys = {
        (recipe, seed, task, _PRIMARY_METRIC)
        for recipe in universe.recipes
        for seed in universe.seeds
        for task in universe.source_tasks
    }
    by_step: dict[int, list[MetricObservation]] = {}
    for observation in observations:
        if (
            observation.model_size == universe.model_size
            and observation.recipe in recipe_set
            and observation.seed in seed_set
            and observation.source_task in task_set
            and observation.metric == _PRIMARY_METRIC
        ):
            by_step.setdefault(observation.step, []).append(observation)

    checkpoints: list[CheckpointRows] = []
    for step in sorted(by_step):
        selected = by_step[step]
        actual_keys = {
            (item.recipe, item.seed, item.source_task, item.metric) for item in selected
        }
        if actual_keys != expected_keys or len(selected) != len(actual_keys):
            continue
        computes = {item.compute for item in selected}
        if len(computes) != 1:
            raise ValueError(
                "checkpoint compute differs across the declared grid: "
                f"model_size={universe.model_size!r}, step={step}"
            )
        checkpoints.append(
            CheckpointRows(
                universe=universe,
                step=step,
                observations=tuple(
                    sorted(
                        selected,
                        key=lambda item: (
                            item.recipe,
                            item.seed,
                            item.source_task,
                            item.metric,
                        ),
                    )
                ),
                raw_row_count=len(selected),
                selected_observation_count=len(selected),
                expected_observation_count=len(expected_keys),
                actual_compute=next(iter(computes)),
            )
        )
    if not checkpoints:
        raise ValueError(
            "no common complete checkpoint exists for the declared grid: "
            f"model_size={universe.model_size!r}, expected={len(expected_keys)}"
        )
    return tuple(checkpoints)


def _available_checkpoints(
    observations: tuple[MetricObservation, ...],
) -> dict[str, tuple[CheckpointRows, ...]]:
    available_sizes = {item.model_size for item in observations}
    unknown_sizes = available_sizes.difference(_MODEL_SIZE_ORDER)
    if unknown_sizes:
        raise ValueError(
            f"olmes_aggregate contains unexpected model sizes: {tuple(sorted(unknown_sizes))!r}"
        )
    return {
        model_size: _all_common_complete_checkpoints(
            observations,
            _universe(model_size),
        )
        for model_size in _MODEL_SIZE_ORDER
        if model_size in available_sizes
    }


def _selected_key_sha256(observations: Iterable[MetricObservation]) -> str:
    records = sorted(
        (
            {
                "params": item.model_size,
                "data": item.recipe,
                "seed": item.seed,
                "step": item.step,
                "task": item.source_task,
            }
            for item in observations
        ),
        key=lambda item: tuple(item[column] for column in _SELECTED_KEY_COLUMNS),
    )
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _row_selection(
    checkpoints: Iterable[CheckpointRows],
    *,
    columns: tuple[str, ...],
    parquet_sha256: str,
) -> RowSelection:
    values = tuple(checkpoints)
    if not values:
        raise ValueError("row selection requires at least one checkpoint")
    universe = values[0].universe
    if any(item.universe != universe for item in values):
        raise ValueError("one row selection cannot span checkpoint universes")
    observations = tuple(
        observation for checkpoint in values for observation in checkpoint.observations
    )
    return RowSelection(
        logical_table_id=_OLMES_TABLE_ID,
        columns=columns,
        predicates=(
            RowPredicate(
                column="params",
                operator=PredicateOperator.EQ,
                value=universe.model_size,
            ),
            RowPredicate(
                column="data",
                operator=PredicateOperator.IN,
                value=universe.recipes,
            ),
            RowPredicate(
                column="seed",
                operator=PredicateOperator.IN,
                value=universe.seeds,
            ),
            RowPredicate(
                column="step",
                operator=PredicateOperator.IN,
                value=tuple(item.step for item in values),
            ),
            RowPredicate(
                column="task",
                operator=PredicateOperator.IN,
                value=universe.source_tasks,
            ),
        ),
        local_parquet_sha256=parquet_sha256,
        selected_row_count=len(observations),
        selected_key_sha256=_selected_key_sha256(observations),
    )


def _checkpoint_selection(
    checkpoint: CheckpointRows,
    *,
    requested_meaning: str,
    rule: CheckpointRule,
    contract: PaperValidationContract,
) -> CheckpointSelection:
    return CheckpointSelection(
        requested_meaning=requested_meaning,
        rule=rule,
        actual_step=checkpoint.step,
        completeness_dimensions=contract.checkpoint_policy.completeness_dimensions,
        expected_group_count=checkpoint.expected_observation_count,
        selected_group_count=checkpoint.selected_observation_count,
    )


def _ranked(scores: Mapping[str, float]) -> tuple[RankedRecipe, ...]:
    return tuple(
        RankedRecipe(recipe=recipe, rank=index, score=score)
        for index, (recipe, score) in enumerate(
            sorted(scores.items(), key=lambda item: (-item[1], item[0])),
            start=1,
        )
    )


def _target_ranking(checkpoint: CheckpointRows) -> TargetRanking:
    aggregate = aggregate_checkpoint(checkpoint)
    lookup = {(item.recipe, item.seed): item.score for item in aggregate}
    return TargetRanking(
        model_size=checkpoint.universe.model_size,
        step=checkpoint.step,
        metric=_PRIMARY_METRIC,
        seed_count=len(checkpoint.universe.seeds),
        scores=_ranked(
            {
                recipe: _mean(
                    lookup[(recipe, seed)] for seed in checkpoint.universe.seeds
                )
                for recipe in checkpoint.universe.recipes
            }
        ),
    )


def _headline_results(
    *,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    checkpoints: Mapping[str, tuple[CheckpointRows, ...]],
    columns: tuple[str, ...],
    parquet_sha256: str,
) -> tuple[AttemptResult, ...]:
    attempt = _attempt(contract, "dd-0011-default")
    claim = _claim(registry, "DD-0011")
    rule = _comparison_rule(contract, attempt)
    if (
        rule.predicate is not ComparisonPredicate.ABSOLUTE_TOLERANCE
        or rule.absolute_tolerance is None
    ):
        raise ValueError("DD-0011 requires a frozen absolute-tolerance rule")
    if claim.paper_target != "decision_accuracy approximately 0.80":
        raise ValueError("DD-0011 paper target changed from the implemented contract")
    if _TARGET_SIZE not in checkpoints or _HEADLINE_PREDICTION_SIZE not in checkpoints:
        raise ValueError("DD-0011 requires complete 1B and 150M OLMES grids")

    target = checkpoints[_TARGET_SIZE][-1]
    predictions = tuple(reversed(checkpoints[_HEADLINE_PREDICTION_SIZE][-3:]))
    if len(predictions) != 3:
        raise ValueError("DD-0011 requires a default and two preceding checkpoints")
    threshold_sensitivities = tuple(
        (
            f"dd-0011-comparison-absolute-tolerance-grid-{index}",
            threshold,
        )
        for index, threshold in enumerate(rule.threshold_grid, start=1)
        if threshold != rule.absolute_tolerance
    )
    expected_sensitivity_ids = (
        "dd-0011-preceding-common-complete-1",
        "dd-0011-preceding-common-complete-2",
        "dd-0011-paper-step",
        *(sensitivity_id for sensitivity_id, _ in threshold_sensitivities),
    )
    if attempt.sensitivity_ids != expected_sensitivity_ids:
        raise ValueError("DD-0011 sensitivity IDs differ from the implemented contract")
    target_ranking = _target_ranking(target)
    requested_ids = (attempt.id, *attempt.sensitivity_ids[:2])
    results: list[AttemptResult] = []
    for index, (attempt_id, prediction) in enumerate(
        zip(requested_ids, predictions, strict=True)
    ):
        analysis = analyze_prediction_checkpoint(
            prediction,
            target_ranking,
            target_compute=target.actual_compute,
        )
        summary = analysis.summaries[0]
        difference = summary.mean_accuracy - _HEADLINE_TARGET
        outcome = (
            ValidationOutcome.APPROXIMATELY_REPRODUCED
            if abs(difference) <= rule.absolute_tolerance
            else ValidationOutcome.NOT_REPRODUCED
        )
        role = AttemptRole.DEFAULT if index == 0 else AttemptRole.SENSITIVITY
        checkpoint_rule = (
            CheckpointRule.LATEST_COMMON_COMPLETE
            if role is AttemptRole.DEFAULT
            else CheckpointRule.PRECEDING_COMMON_COMPLETE
        )
        diagnostics = (
            "correct counts by prediction seed: "
            + ", ".join(
                f"{seed}={correct}/{summary.denominator_per_seed}"
                for seed, correct in zip(
                    prediction.universe.seeds,
                    summary.correct_counts,
                    strict=True,
                )
            ),
        )
        if role is AttemptRole.DEFAULT and _PAPER_STEP not in {
            item.step for item in checkpoints[_HEADLINE_PREDICTION_SIZE]
        }:
            diagnostics += (
                "Paper-reported 150M step 38157 is not available as a "
                "common-complete checkpoint in olmes_aggregate; the latest "
                f"common-complete step is {prediction.step}.",
            )
        results.append(
            AttemptResult(
                attempt_id=attempt_id,
                claim_id=claim.id,
                role=role,
                parent_attempt_id=None if role is AttemptRole.DEFAULT else attempt.id,
                comparison_rule_id=rule.id,
                comparison_rule_version=rule.version,
                transformation_ids=attempt.transformation_ids,
                row_selections=(
                    _row_selection(
                        (target,),
                        columns=columns,
                        parquet_sha256=parquet_sha256,
                    ),
                    _row_selection(
                        (prediction,),
                        columns=columns,
                        parquet_sha256=parquet_sha256,
                    ),
                ),
                checkpoint_selections=(
                    _checkpoint_selection(
                        target,
                        requested_meaning="1B final target ranking",
                        rule=CheckpointRule.LATEST_COMMON_COMPLETE,
                        contract=contract,
                    ),
                    _checkpoint_selection(
                        prediction,
                        requested_meaning=(
                            "150M final prediction ranking"
                            if role is AttemptRole.DEFAULT
                            else f"150M preceding sensitivity {index}"
                        ),
                        rule=checkpoint_rule,
                        contract=contract,
                    ),
                ),
                target_value=claim.paper_target,
                computed_value=summary.mean_accuracy,
                unrounded_difference=difference,
                seeds=prediction.universe.seeds,
                denominator=(
                    summary.denominator_per_seed * len(prediction.universe.seeds)
                ),
                target_ties=summary.target_ties,
                predicted_ties=summary.predicted_ties,
                standard_deviation=summary.sample_sd_accuracy,
                ddof=summary.ddof,
                outcome=outcome,
                diagnostics=diagnostics,
            )
        )
    default_result = results[0]
    for sensitivity_id, threshold in threshold_sensitivities:
        difference = float(default_result.unrounded_difference)
        results.append(
            default_result.model_copy(
                update={
                    "attempt_id": sensitivity_id,
                    "role": AttemptRole.SENSITIVITY,
                    "parent_attempt_id": attempt.id,
                    "computed_value": {
                        "decision_accuracy": default_result.computed_value,
                        "absolute_tolerance": threshold,
                        "satisfied": abs(difference) <= threshold,
                    },
                    "outcome": (
                        ValidationOutcome.APPROXIMATELY_REPRODUCED
                        if abs(difference) <= threshold
                        else ValidationOutcome.NOT_REPRODUCED
                    ),
                    "diagnostics": (
                        *default_result.diagnostics,
                        "Compared the default checkpoint against frozen absolute "
                        f"tolerance {threshold:.12g}.",
                    ),
                }
            )
        )
    return tuple(results)


def _plot_checkpoints(
    checkpoints: Mapping[str, tuple[CheckpointRows, ...]],
) -> tuple[CheckpointRows, ...]:
    return tuple(
        checkpoint
        for model_size in _MODEL_SIZE_ORDER
        for checkpoint in checkpoints.get(model_size, ())
        if checkpoint.actual_compute > 0
    )


def _aggregate_plot(
    *,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    checkpoints: Mapping[str, tuple[CheckpointRows, ...]],
    columns: tuple[str, ...],
    parquet_sha256: str,
) -> tuple[AttemptResult, PlotSeries]:
    attempt = _attempt(contract, "dd-0169-default")
    claim = _claim(registry, "DD-0169")
    rule = _comparison_rule(contract, attempt)
    if rule.predicate is not ComparisonPredicate.NONEMPTY_PLOT:
        raise ValueError("DD-0169 requires the frozen nonempty-plot rule")
    if attempt.plot_series_ids and attempt.plot_series_ids != (_AGGREGATE_SERIES_ID,):
        raise ValueError("DD-0169 declares an unexpected plot-series ID")

    target = checkpoints[_TARGET_SIZE][-1]
    target_ranking = _target_ranking(target)
    selected = _plot_checkpoints(checkpoints)
    points: list[PlotPoint] = []
    total_target_ties = 0
    total_predicted_ties = 0
    for checkpoint in selected:
        summary = analyze_prediction_checkpoint(
            checkpoint,
            target_ranking,
            target_compute=target.actual_compute,
        ).summaries[0]
        total_target_ties += summary.target_ties
        total_predicted_ties += summary.predicted_ties
        points.append(
            PlotPoint(
                dimensions=(
                    DimensionValue(
                        name="model_size", value=checkpoint.universe.model_size
                    ),
                    DimensionValue(name="step", value=checkpoint.step),
                ),
                measures=(
                    MeasureValue(
                        name="percent_target_compute",
                        value=summary.percent_target_compute,
                    ),
                    MeasureValue(
                        name="decision_accuracy",
                        value=summary.mean_accuracy,
                    ),
                    MeasureValue(
                        name="decision_accuracy_sd",
                        value=summary.sample_sd_accuracy,
                    ),
                    MeasureValue(name="compute", value=summary.actual_compute),
                    MeasureValue(
                        name="denominator_per_seed",
                        value=float(summary.denominator_per_seed),
                    ),
                    MeasureValue(
                        name="target_ties_total",
                        value=float(summary.target_ties),
                    ),
                    MeasureValue(
                        name="predicted_ties_total",
                        value=float(summary.predicted_ties),
                    ),
                    MeasureValue(name="seed_count", value=float(summary.seed_count)),
                ),
            )
        )
    series = PlotSeries(
        id=_AGGREGATE_SERIES_ID,
        figure="fig:accuracy_vs_compute",
        panel="single-scale",
        semantic_kind="aggregate primary-metric decision accuracy by compute",
        x_axis=AxisSpec(
            measure="percent_target_compute",
            scale=AxisScale.LOG,
            unit="percent of 1B final compute",
        ),
        y_axis=AxisSpec(
            measure="decision_accuracy",
            scale=AxisScale.LINEAR,
            unit="fraction of unordered recipe pairs correct",
        ),
        dimensions=("model_size", "step"),
        measures=(
            "percent_target_compute",
            "decision_accuracy",
            "decision_accuracy_sd",
            "compute",
            "denominator_per_seed",
            "target_ties_total",
            "predicted_ties_total",
            "seed_count",
        ),
        attempt_id=attempt.id,
        counts=(
            NamedCount(name="recipes", value=len(_PAPER_RECIPES)),
            NamedCount(name="target_seeds", value=len(_TARGET_SEEDS)),
            NamedCount(name="prediction_seeds", value=len(_PREDICTION_SEEDS)),
            NamedCount(
                name="source_tasks", value=len(DEFAULT_TASK_GROUPING.source_tasks)
            ),
            NamedCount(name="unordered_pairs_per_seed", value=300),
            NamedCount(name="points", value=len(points)),
        ),
        points=tuple(points),
    )
    zero_compute_count = sum(
        checkpoint.actual_compute == 0
        for values in checkpoints.values()
        for checkpoint in values
    )
    result = AttemptResult(
        attempt_id=attempt.id,
        claim_id=claim.id,
        role=AttemptRole.DEFAULT,
        comparison_rule_id=rule.id,
        comparison_rule_version=rule.version,
        transformation_ids=attempt.transformation_ids,
        row_selections=tuple(
            _row_selection(
                tuple(
                    item for item in selected if item.universe.model_size == model_size
                ),
                columns=columns,
                parquet_sha256=parquet_sha256,
            )
            for model_size in _MODEL_SIZE_ORDER
            if any(item.universe.model_size == model_size for item in selected)
        ),
        checkpoint_selections=tuple(
            _checkpoint_selection(
                checkpoint,
                requested_meaning="all common-complete aggregate plot points",
                rule=CheckpointRule.EXACT,
                contract=contract,
            )
            for checkpoint in selected
        ),
        target_value=claim.paper_target,
        computed_value={
            "plot_series_id": series.id,
            "point_count": len(points),
        },
        seeds=(*_PREDICTION_SEEDS, "large aux 2", "large aux 3"),
        exclusions=(
            NamedCount(
                name="non_positive_compute_checkpoints", value=zero_compute_count
            ),
        ),
        target_ties=total_target_ties,
        predicted_ties=total_predicted_ties,
        outcome=ValidationOutcome.DESCRIPTIVE_ONLY,
        diagnostics=(
            f"Persisted {len(points)} common-complete aggregate plot points.",
        ),
        limitations=(
            "The derived plot series is available for visual comparison, but no "
            "frozen semantic predicate adjudicates approximate log-linearity.",
        ),
        plot_series_ids=(series.id,),
    )
    return result, series


def _logical_task_scores(
    checkpoint: CheckpointRows,
) -> dict[tuple[str, str, str], float]:
    by_recipe_seed: dict[tuple[str, str], dict[str, float]] = {}
    for observation in checkpoint.observations:
        key = (observation.recipe, observation.seed)
        scores = by_recipe_seed.setdefault(key, {})
        scores[observation.source_task] = observation.score
    expected_tasks = set(DEFAULT_TASK_GROUPING.source_tasks)
    result: dict[tuple[str, str, str], float] = {}
    for (recipe, seed), scores in by_recipe_seed.items():
        if set(scores) != expected_tasks:
            raise ValueError("per-task checkpoint group is incomplete")
        for task in DEFAULT_TASK_GROUPING.non_mmlu_tasks:
            result[(task, recipe, seed)] = scores[task]
        result[("mmlu", recipe, seed)] = _mean(
            scores[subject] for subject in DEFAULT_TASK_GROUPING.mmlu_subjects
        )
    return result


def _per_task_target_rankings(
    checkpoint: CheckpointRows,
) -> dict[str, TargetRanking]:
    scores = _logical_task_scores(checkpoint)
    return {
        task: TargetRanking(
            model_size=checkpoint.universe.model_size,
            step=checkpoint.step,
            metric=_PRIMARY_METRIC,
            seed_count=len(checkpoint.universe.seeds),
            scores=_ranked(
                {
                    recipe: _mean(
                        scores[(task, recipe, seed)]
                        for seed in checkpoint.universe.seeds
                    )
                    for recipe in checkpoint.universe.recipes
                }
            ),
        )
        for task in _LOGICAL_TASKS
    }


def _per_task_plot(
    *,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    checkpoints: Mapping[str, tuple[CheckpointRows, ...]],
    columns: tuple[str, ...],
    parquet_sha256: str,
) -> tuple[AttemptResult, PlotSeries]:
    attempt = _attempt(contract, "dd-0148-default")
    claim = _claim(registry, "DD-0148")
    rule = _comparison_rule(contract, attempt)
    if rule.predicate is not ComparisonPredicate.NONEMPTY_PLOT:
        raise ValueError("DD-0148 requires the frozen nonempty-plot rule")
    if attempt.plot_series_ids and attempt.plot_series_ids != (_PER_TASK_SERIES_ID,):
        raise ValueError("DD-0148 declares an unexpected plot-series ID")

    target = checkpoints[_TARGET_SIZE][-1]
    targets = _per_task_target_rankings(target)
    selected = _plot_checkpoints(checkpoints)
    points: list[PlotPoint] = []
    total_target_ties = 0
    total_predicted_ties = 0
    for checkpoint in selected:
        scores = _logical_task_scores(checkpoint)
        for task in _LOGICAL_TASKS:
            seed_results = []
            for seed in checkpoint.universe.seeds:
                prediction = PredictionRanking(
                    model_size=checkpoint.universe.model_size,
                    step=checkpoint.step,
                    metric=_PRIMARY_METRIC,
                    seed=seed,
                    scores=_ranked(
                        {
                            recipe: scores[(task, recipe, seed)]
                            for recipe in checkpoint.universe.recipes
                        }
                    ),
                )
                seed_results.append(
                    compare_rankings(
                        targets[task],
                        prediction,
                        actual_compute=checkpoint.actual_compute,
                        target_compute=target.actual_compute,
                    )
                )
            accuracies = tuple(item.accuracy for item in seed_results)
            denominators = {item.denominator for item in seed_results}
            if len(denominators) != 1:
                raise ValueError("per-task denominators differ across prediction seeds")
            denominator_per_seed = next(iter(denominators))
            target_ties = sum(item.target_ties for item in seed_results)
            predicted_ties = sum(item.predicted_ties for item in seed_results)
            total_target_ties += target_ties
            total_predicted_ties += predicted_ties
            points.append(
                PlotPoint(
                    dimensions=(
                        DimensionValue(name="task", value=task),
                        DimensionValue(
                            name="model_size", value=checkpoint.universe.model_size
                        ),
                        DimensionValue(name="step", value=checkpoint.step),
                    ),
                    measures=(
                        MeasureValue(
                            name="percent_target_compute",
                            value=seed_results[0].percent_target_compute,
                        ),
                        MeasureValue(
                            name="decision_accuracy",
                            value=_mean(accuracies),
                        ),
                        MeasureValue(
                            name="decision_accuracy_sd",
                            value=_sample_sd(accuracies),
                        ),
                        MeasureValue(name="compute", value=checkpoint.actual_compute),
                        MeasureValue(
                            name="denominator_per_seed",
                            value=float(denominator_per_seed),
                        ),
                        MeasureValue(
                            name="target_ties_total",
                            value=float(target_ties),
                        ),
                        MeasureValue(
                            name="predicted_ties_total",
                            value=float(predicted_ties),
                        ),
                        MeasureValue(
                            name="seed_count",
                            value=float(len(seed_results)),
                        ),
                    ),
                )
            )
    series = PlotSeries(
        id=_PER_TASK_SERIES_ID,
        figure="fig:primary_metric_compute_vs_accuracy_per_task",
        panel="all tasks",
        semantic_kind="per-task primary-metric decision accuracy by compute",
        x_axis=AxisSpec(
            measure="percent_target_compute",
            scale=AxisScale.LOG,
            unit="percent of 1B final compute",
        ),
        y_axis=AxisSpec(
            measure="decision_accuracy",
            scale=AxisScale.LINEAR,
            unit="fraction of unordered recipe pairs correct",
        ),
        dimensions=("task", "model_size", "step"),
        measures=(
            "percent_target_compute",
            "decision_accuracy",
            "decision_accuracy_sd",
            "compute",
            "denominator_per_seed",
            "target_ties_total",
            "predicted_ties_total",
            "seed_count",
        ),
        attempt_id=attempt.id,
        counts=(
            NamedCount(name="recipes", value=len(_PAPER_RECIPES)),
            NamedCount(name="target_seeds", value=len(_TARGET_SEEDS)),
            NamedCount(name="prediction_seeds", value=len(_PREDICTION_SEEDS)),
            NamedCount(name="logical_tasks", value=len(_LOGICAL_TASKS)),
            NamedCount(name="unordered_pairs_per_seed", value=300),
            NamedCount(name="points", value=len(points)),
        ),
        points=tuple(points),
    )
    zero_compute_count = sum(
        checkpoint.actual_compute == 0
        for values in checkpoints.values()
        for checkpoint in values
    )
    result = AttemptResult(
        attempt_id=attempt.id,
        claim_id=claim.id,
        role=AttemptRole.DEFAULT,
        comparison_rule_id=rule.id,
        comparison_rule_version=rule.version,
        transformation_ids=attempt.transformation_ids,
        row_selections=tuple(
            _row_selection(
                tuple(
                    item for item in selected if item.universe.model_size == model_size
                ),
                columns=columns,
                parquet_sha256=parquet_sha256,
            )
            for model_size in _MODEL_SIZE_ORDER
            if any(item.universe.model_size == model_size for item in selected)
        ),
        checkpoint_selections=tuple(
            _checkpoint_selection(
                checkpoint,
                requested_meaning="all common-complete per-task plot points",
                rule=CheckpointRule.EXACT,
                contract=contract,
            )
            for checkpoint in selected
        ),
        target_value=claim.paper_target,
        computed_value={
            "plot_series_id": series.id,
            "point_count": len(points),
            "logical_task_count": len(_LOGICAL_TASKS),
        },
        seeds=(*_PREDICTION_SEEDS, "large aux 2", "large aux 3"),
        exclusions=(
            NamedCount(
                name="non_positive_compute_checkpoints", value=zero_compute_count
            ),
        ),
        target_ties=total_target_ties,
        predicted_ties=total_predicted_ties,
        outcome=ValidationOutcome.DESCRIPTIVE_ONLY,
        diagnostics=(f"Persisted {len(points)} common-complete per-task plot points.",),
        limitations=(
            "The derived plot series is available for visual comparison, but no "
            "frozen semantic predicate adjudicates the task-specific claims.",
        ),
        plot_series_ids=(series.id,),
    )
    return result, series


def _clone_series(
    base: PlotSeries,
    attempt: AttemptSpec,
    points: tuple[PlotPoint, ...],
) -> PlotSeries:
    if len(attempt.plot_series_ids) != 1:
        raise ValueError(f"{attempt.id} must declare exactly one plot-series ID")
    counts = tuple(count for count in base.counts if count.name != "points") + (
        NamedCount(name="points", value=len(points)),
    )
    return PlotSeries(
        id=attempt.plot_series_ids[0],
        figure=base.figure,
        panel=attempt.claim_id,
        semantic_kind=base.semantic_kind,
        x_axis=base.x_axis,
        y_axis=base.y_axis,
        dimensions=base.dimensions,
        measures=base.measures,
        attempt_id=attempt.id,
        actual_checkpoint=base.actual_checkpoint,
        counts=counts,
        points=points,
    )


def _qualitative_result(
    *,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    attempt_id: str,
    evidence: AttemptResult,
    computed_value: dict[str, object],
    outcome: ValidationOutcome,
    diagnostics: tuple[str, ...],
    missing_groups: tuple[str, ...] = (),
    plot_series_ids: tuple[str, ...] = (),
) -> AttemptResult:
    attempt = _attempt(contract, attempt_id)
    claim = _claim(registry, attempt.claim_id)
    rule = _comparison_rule(contract, attempt)
    return AttemptResult(
        attempt_id=attempt.id,
        claim_id=claim.id,
        role=AttemptRole.DEFAULT,
        comparison_rule_id=rule.id,
        comparison_rule_version=rule.version,
        transformation_ids=attempt.transformation_ids,
        row_selections=evidence.row_selections,
        checkpoint_selections=evidence.checkpoint_selections,
        target_value=claim.paper_target,
        computed_value=computed_value,
        seeds=evidence.seeds,
        denominator=evidence.denominator,
        exclusions=evidence.exclusions,
        target_ties=evidence.target_ties,
        predicted_ties=evidence.predicted_ties,
        missing_groups=missing_groups,
        outcome=outcome,
        diagnostics=diagnostics,
        limitations=(
            "Predicates and sensitivity grids are versioned in paper_validation.toml.",
            "Matched-accuracy thresholds use the minimum observed compute without interpolation.",
        ),
        plot_series_ids=plot_series_ids,
    )


def _single_scale_qualitative_attempts(
    *,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    headline: tuple[AttemptResult, ...],
    aggregate_result: AttemptResult,
    aggregate_series: PlotSeries,
) -> tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]]:
    headline_values = headline[:3]
    results: list[AttemptResult] = []
    series: list[PlotSeries] = []
    for attempt_id in ("dd-0010-default", "dd-0356-default"):
        attempt = _attempt(contract, attempt_id)
        rule = _comparison_rule(contract, attempt)
        threshold_name = (
            ComparisonParameterName.STRONG_BASELINE_THRESHOLD
            if attempt_id == "dd-0010-default"
            else ComparisonParameterName.ACCURACY_THRESHOLD
        )
        threshold = _parameter(rule, threshold_name)
        for index, evidence in enumerate(headline_values):
            result_id = attempt.id if index == 0 else attempt.sensitivity_ids[index - 1]
            value = float(evidence.computed_value)
            outcome = (
                ValidationOutcome.APPROXIMATELY_REPRODUCED
                if attempt_id == "dd-0356-default" and value >= threshold
                else ValidationOutcome.REPRODUCED
                if value >= threshold
                else ValidationOutcome.NOT_REPRODUCED
            )
            result = _qualitative_result(
                registry=registry,
                contract=contract,
                attempt_id=attempt.id,
                evidence=evidence,
                computed_value={
                    "decision_accuracy": value,
                    "threshold": threshold,
                    "satisfied": value >= threshold,
                },
                outcome=outcome,
                diagnostics=(
                    f"Observed decision accuracy {value:.12g} against frozen threshold {threshold:.12g}.",
                ),
                plot_series_ids=attempt.plot_series_ids if index == 0 else (),
            )
            if index:
                result = result.model_copy(
                    update={
                        "attempt_id": result_id,
                        "role": AttemptRole.SENSITIVITY,
                        "parent_attempt_id": attempt.id,
                        "plot_series_ids": (),
                    }
                )
            results.append(result)
        if attempt.plot_series_ids:
            selected = tuple(
                point
                for point in aggregate_series.points
                if _dimensions(point).get("model_size") == _HEADLINE_PREDICTION_SIZE
                and _dimensions(point).get("step")
                == headline_values[0].checkpoint_selections[1].actual_step
            )
            series.append(_clone_series(aggregate_series, attempt, selected))

    aggregate_points = tuple(
        point
        for point in aggregate_series.points
        if _measures(point)["percent_target_compute"] > 0
    )
    log_compute = tuple(
        math.log10(_measures(point)["percent_target_compute"])
        for point in aggregate_points
    )
    accuracies = tuple(
        _measures(point)["decision_accuracy"] for point in aggregate_points
    )
    trend_attempt = _attempt(contract, "dd-0164-default")
    trend_rule = _comparison_rule(contract, trend_attempt)
    slope = _linear_slope(log_compute, accuracies)
    correlation = _spearman(log_compute, accuracies)
    slope_minimum = _parameter(trend_rule, ComparisonParameterName.OLS_SLOPE_MINIMUM)
    correlation_minimum = _parameter(
        trend_rule, ComparisonParameterName.SPEARMAN_MINIMUM
    )
    trend_satisfied = slope > slope_minimum and correlation > correlation_minimum
    results.append(
        _qualitative_result(
            registry=registry,
            contract=contract,
            attempt_id=trend_attempt.id,
            evidence=aggregate_result,
            computed_value={
                "ols_slope_per_compute_decade": slope,
                "spearman": correlation,
                "point_count": len(aggregate_points),
                "satisfied": trend_satisfied,
            },
            outcome=(
                ValidationOutcome.REPRODUCED
                if trend_satisfied
                else ValidationOutcome.NOT_REPRODUCED
            ),
            diagnostics=(
                f"OLS slope={slope:.12g}; Spearman={correlation:.12g} over {len(aggregate_points)} points.",
            ),
        )
    )

    equivalent_attempt = _attempt(contract, "dd-0165-default")
    equivalent_rule = _comparison_rule(contract, equivalent_attempt)
    minimum_difference = _parameter(
        equivalent_rule,
        ComparisonParameterName.EQUIVALENCE_DIFFERENCE_MINIMUM,
    )
    bin_width = _parameter(
        equivalent_rule,
        ComparisonParameterName.COMPUTE_LOG10_BIN_WIDTH,
    )
    points = _compute_equivalence_points(aggregate_series)
    target_points = tuple(
        point
        for point in points
        if point.model_size == _TARGET_SIZE and point.compute > 0
    )
    if not target_points:
        raise ValueError("DD-0165 requires a positive target-model checkpoint")
    target_model_compute = max(target_points, key=lambda point: point.step).compute
    aggregate_zero_compute_count = next(
        (
            count.value
            for count in aggregate_result.exclusions
            if count.name == "non_positive_compute_checkpoints"
        ),
        0,
    )
    computed_value, exclusions = _compute_equivalence(
        points,
        target_model_compute=target_model_compute,
        bin_width=bin_width,
        minimum_difference=minimum_difference,
        preexcluded_zero_compute_count=aggregate_zero_compute_count,
    )
    matched_group_count = int(computed_value["matched_group_count"])
    matched_bin_count = int(computed_value["matched_bin_count"])
    passing_group_count = int(computed_value["passing_group_count"])
    minimum_observed = computed_value["minimum_accuracy_difference"]
    if minimum_observed is None:
        outcome = ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
        missing_groups = ("compute_bucket=cross_size_intermediate_to_final",)
        diagnostics = (
            f"No cross-size intermediate/final matches exist in fixed log10 "
            f"compute buckets of width {bin_width:.12g}.",
        )
    else:
        outcome = (
            ValidationOutcome.REPRODUCED
            if bool(computed_value["satisfied"])
            else ValidationOutcome.NOT_REPRODUCED
        )
        missing_groups = ()
        diagnostics = (
            f"Fixed log10 compute buckets of width {bin_width:.12g} matched "
            f"{matched_group_count} groups across {matched_bin_count} bins; "
            f"{passing_group_count} groups met the minimum difference and the "
            f"minimum accuracy difference was {float(minimum_observed):.12g}.",
        )
    result = _qualitative_result(
        registry=registry,
        contract=contract,
        attempt_id=equivalent_attempt.id,
        evidence=aggregate_result,
        computed_value=computed_value,
        outcome=outcome,
        diagnostics=diagnostics,
        missing_groups=missing_groups,
    )
    results.append(result.model_copy(update={"exclusions": exclusions}))
    return tuple(results), tuple(series)


def _task_curve_rows(series: PlotSeries) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            **_dimensions(point),
            **_measures(point),
            "point": point,
        }
        for point in series.points
    )


def _threshold_compute(
    rows: Iterable[dict[str, object]], task: str, threshold: float
) -> float | None:
    values = tuple(
        float(row["compute"])
        for row in rows
        if row["task"] == task
        and float(row["decision_accuracy"]) >= threshold
        and float(row["compute"]) > 0
    )
    return min(values) if values else None


def _fit_sse(xs: tuple[float, ...], ys: tuple[float, ...]) -> tuple[float, float]:
    slope = _linear_slope(xs, ys)
    intercept = _mean(ys) - slope * _mean(xs)
    return slope, math.fsum(
        (y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True)
    )


def _plateau_diagnostic(
    rows: Iterable[dict[str, object]], task: str
) -> dict[str, object]:
    by_compute: dict[float, list[float]] = {}
    for row in rows:
        if row["task"] != task or float(row["compute"]) <= 0:
            continue
        by_compute.setdefault(float(row["compute"]), []).append(
            float(row["decision_accuracy"])
        )
    curve = tuple(
        (math.log10(compute), _mean(values))
        for compute, values in sorted(by_compute.items())
    )
    if len(curve) < 4:
        return {
            "point_count": len(curve),
            "sse_improvement": 0.0,
            "early_slope": 0.0,
            "late_slope": 0.0,
            "split_index": None,
        }
    xs = tuple(item[0] for item in curve)
    ys = tuple(item[1] for item in curve)
    _, single_sse = _fit_sse(xs, ys)
    candidates: list[tuple[float, int, float, float]] = []
    for split in range(2, len(curve) - 1):
        early_slope, early_sse = _fit_sse(xs[:split], ys[:split])
        late_slope, late_sse = _fit_sse(xs[split:], ys[split:])
        candidates.append((early_sse + late_sse, split, early_slope, late_slope))
    best_sse, split, early_slope, late_slope = min(candidates)
    improvement = 0.0 if single_sse == 0 else (single_sse - best_sse) / single_sse
    return {
        "point_count": len(curve),
        "sse_improvement": improvement,
        "early_slope": early_slope,
        "late_slope": late_slope,
        "split_index": split,
    }


def _per_task_qualitative_attempts(
    *,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    evidence: AttemptResult,
    base_series: PlotSeries,
) -> tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]]:
    rows = _task_curve_rows(base_series)
    results: list[AttemptResult] = []
    series: list[PlotSeries] = []

    def add(
        attempt_id: str,
        computed: dict[str, object],
        outcome: ValidationOutcome,
        diagnostics: tuple[str, ...],
    ) -> None:
        attempt = _attempt(contract, attempt_id)
        plot_ids = attempt.plot_series_ids
        results.append(
            _qualitative_result(
                registry=registry,
                contract=contract,
                attempt_id=attempt_id,
                evidence=evidence,
                computed_value=computed,
                outcome=outcome,
                diagnostics=diagnostics,
                plot_series_ids=plot_ids,
            )
        )
        if plot_ids:
            tasks = set(attempt.task_ids)
            if "paper-olmes-tasks" in tasks:
                tasks = set(_LOGICAL_TASKS)
            selected = tuple(row["point"] for row in rows if str(row["task"]) in tasks)
            series.append(_clone_series(base_series, attempt, selected))

    threshold_attempts = {
        attempt_id: _attempt(contract, attempt_id)
        for attempt_id in (
            "dd-0051-default",
            "dd-0052-default",
            "dd-0149-default",
            "dd-0150-default",
            "dd-0166-default",
            "dd-0167-default",
            "dd-0175-default",
        )
    }
    accuracy_thresholds = {
        attempt_id: _parameter(
            _comparison_rule(contract, attempt),
            ComparisonParameterName.ACCURACY_THRESHOLD,
        )
        for attempt_id, attempt in threshold_attempts.items()
    }
    threshold_computes = {
        attempt_id: {
            task: _threshold_compute(rows, task, threshold) for task in _LOGICAL_TASKS
        }
        for attempt_id, threshold in accuracy_thresholds.items()
    }

    for attempt_id in ("dd-0051-default", "dd-0166-default"):
        attempt = threshold_attempts[attempt_id]
        rule = _comparison_rule(contract, attempt)
        available = {
            task: compute
            for task, compute in threshold_computes[attempt_id].items()
            if compute is not None
        }
        ratio = (
            max(available.values()) / min(available.values())
            if len(available) >= 2
            else 0.0
        )
        required = _parameter(rule, ComparisonParameterName.COMPUTE_RATIO_THRESHOLD)
        satisfied = ratio >= required
        add(
            attempt_id,
            {
                "accuracy_threshold": accuracy_thresholds[attempt_id],
                "threshold_compute_by_task": available,
                "missing_threshold_tasks": sorted(
                    set(_LOGICAL_TASKS).difference(available)
                ),
                "max_min_compute_ratio": ratio,
                "required_ratio": required,
                "satisfied": satisfied,
            },
            ValidationOutcome.REPRODUCED
            if satisfied
            else ValidationOutcome.NOT_REPRODUCED,
            (f"Observed max/min threshold-compute ratio {ratio:.12g}.",),
        )

    for attempt_id, comparison_tasks in (
        ("dd-0052-default", ("mmlu", "arc_challenge", "arc_easy")),
        ("dd-0167-default", ("arc_challenge", "arc_easy", "mmlu")),
    ):
        rule = _comparison_rule(contract, threshold_attempts[attempt_id])
        computes = threshold_computes[attempt_id]
        hellaswag = computes["hellaswag"]
        ratios = {
            task: (
                hellaswag / computes[task]
                if hellaswag is not None and computes[task] not in {None, 0.0}
                else 0.0
            )
            for task in comparison_tasks
        }
        directional = all(value > 1 for value in ratios.values())
        if attempt_id == "dd-0167-default":
            required = _parameter(rule, ComparisonParameterName.COMPUTE_RATIO_THRESHOLD)
            magnitude = all(value >= required for value in ratios.values())
        else:
            required = 1.0
            magnitude = directional
        outcome = (
            ValidationOutcome.REPRODUCED
            if magnitude
            else ValidationOutcome.DIRECTIONALLY_CONSISTENT
            if directional
            else ValidationOutcome.NOT_REPRODUCED
        )
        add(
            attempt_id,
            {
                "accuracy_threshold": accuracy_thresholds[attempt_id],
                "hellaswag_compute": hellaswag,
                "hellaswag_to_task_compute_ratios": ratios,
                "required_ratio": required,
                "direction_satisfied": directional,
                "magnitude_satisfied": magnitude,
            },
            outcome,
            (f"HellaSwag/task matched-accuracy compute ratios: {ratios!r}.",),
        )

    social_attempt = _attempt(contract, "dd-0053-default")
    social_rule = _comparison_rule(contract, social_attempt)
    reliability_maximum = _parameter(
        social_rule, ComparisonParameterName.LOW_RELIABILITY_MAXIMUM
    )
    social_max = max(
        float(row["decision_accuracy"]) for row in rows if row["task"] == "socialiqa"
    )
    social_satisfied = social_max < reliability_maximum
    add(
        social_attempt.id,
        {
            "maximum_decision_accuracy": social_max,
            "low_reliability_maximum": reliability_maximum,
            "satisfied": social_satisfied,
        },
        ValidationOutcome.REPRODUCED
        if social_satisfied
        else ValidationOutcome.NOT_REPRODUCED,
        (f"SocialIQA maximum observed decision accuracy={social_max:.12g}.",),
    )

    nontrivial_attempt = _attempt(contract, "dd-0142-default")
    nontrivial_rule = _comparison_rule(contract, nontrivial_attempt)
    nontrivial_threshold = _parameter(
        nontrivial_rule, ComparisonParameterName.NONTRIVIAL_ACCURACY_THRESHOLD
    )
    task_maxima = {
        task: max(
            float(row["decision_accuracy"]) for row in rows if row["task"] == task
        )
        for task in _LOGICAL_TASKS
    }
    non_boolq = {task: value for task, value in task_maxima.items() if task != "boolq"}
    nontrivial_satisfied = all(
        value > nontrivial_threshold for value in non_boolq.values()
    )
    add(
        nontrivial_attempt.id,
        {
            "nontrivial_threshold": nontrivial_threshold,
            "maximum_decision_accuracy_by_task": task_maxima,
            "satisfied": nontrivial_satisfied,
        },
        ValidationOutcome.REPRODUCED
        if nontrivial_satisfied
        else ValidationOutcome.NOT_REPRODUCED,
        (f"Checked {len(non_boolq)} non-BoolQ logical tasks.",),
    )

    small_attempt = threshold_attempts["dd-0149-default"]
    small_rule = _comparison_rule(contract, small_attempt)
    small_percent = _parameter(
        small_rule, ComparisonParameterName.MAXIMUM_SCALE_PERCENT
    )
    small_points = tuple(
        row
        for row in rows
        if row["task"] == "arc_easy"
        and float(row["percent_target_compute"]) <= small_percent
    )
    small_max = max(
        (float(row["decision_accuracy"]) for row in small_points), default=0.0
    )
    small_satisfied = small_max >= accuracy_thresholds[small_attempt.id]
    add(
        small_attempt.id,
        {
            "maximum_scale_percent": small_percent,
            "accuracy_threshold": accuracy_thresholds[small_attempt.id],
            "eligible_point_count": len(small_points),
            "maximum_eligible_accuracy": small_max,
            "satisfied": small_satisfied,
        },
        ValidationOutcome.REPRODUCED
        if small_satisfied
        else ValidationOutcome.NOT_REPRODUCED,
        (
            f"ARC Easy has {len(small_points)} points within {small_percent}% target compute.",
        ),
    )

    ratio_attempt = threshold_attempts["dd-0150-default"]
    ratio_rule = _comparison_rule(contract, ratio_attempt)
    ratio_required = _parameter(
        ratio_rule, ComparisonParameterName.COMPUTE_RATIO_THRESHOLD
    )
    arc_compute = threshold_computes[ratio_attempt.id]["arc_easy"]
    hella_compute = threshold_computes[ratio_attempt.id]["hellaswag"]
    hella_ratio = (
        hella_compute / arc_compute
        if hella_compute is not None and arc_compute not in {None, 0.0}
        else 0.0
    )
    ratio_direction = hella_ratio > 1
    ratio_magnitude = hella_ratio >= ratio_required
    add(
        ratio_attempt.id,
        {
            "accuracy_threshold": accuracy_thresholds[ratio_attempt.id],
            "arc_easy_compute": arc_compute,
            "hellaswag_compute": hella_compute,
            "compute_ratio": hella_ratio,
            "required_ratio": ratio_required,
            "direction_satisfied": ratio_direction,
            "magnitude_satisfied": ratio_magnitude,
        },
        ValidationOutcome.REPRODUCED
        if ratio_magnitude
        else ValidationOutcome.DIRECTIONALLY_CONSISTENT
        if ratio_direction
        else ValidationOutcome.NOT_REPRODUCED,
        (f"HellaSwag/ARC Easy matched-accuracy compute ratio={hella_ratio:.12g}.",),
    )

    marked_attempt = _attempt(contract, "dd-0168-default")
    marked_rule = _comparison_rule(contract, marked_attempt)
    marked_minimum = _parameter(marked_rule, ComparisonParameterName.MARKED_GAP_MINIMUM)
    highlighted = {"arc_challenge", "arc_easy", "mmlu", "hellaswag"}
    highlighted_mean = _mean(
        float(row["decision_accuracy"]) for row in rows if row["task"] in highlighted
    )
    remaining_mean = _mean(
        float(row["decision_accuracy"])
        for row in rows
        if row["task"] not in highlighted
    )
    marked_gap = highlighted_mean - remaining_mean
    marked_direction = marked_gap > 0
    marked_magnitude = marked_gap >= marked_minimum
    add(
        marked_attempt.id,
        {
            "highlighted_mean_accuracy": highlighted_mean,
            "remaining_mean_accuracy": remaining_mean,
            "gap": marked_gap,
            "required_gap": marked_minimum,
            "direction_satisfied": marked_direction,
            "magnitude_satisfied": marked_magnitude,
        },
        ValidationOutcome.REPRODUCED
        if marked_magnitude
        else ValidationOutcome.DIRECTIONALLY_CONSISTENT
        if marked_direction
        else ValidationOutcome.NOT_REPRODUCED,
        (f"Highlighted-minus-remaining mean reliability gap={marked_gap:.12g}.",),
    )

    range_attempt = _attempt(contract, "dd-0174-default")
    range_rule = _comparison_rule(contract, range_attempt)
    range_minimum = _parameter(
        range_rule, ComparisonParameterName.FIXED_COMPUTE_RANGE_MINIMUM
    )
    by_compute: dict[float, dict[str, float]] = {}
    for row in rows:
        compute = float(row["compute"])
        task = str(row["task"])
        by_compute.setdefault(compute, {})[task] = float(row["decision_accuracy"])
    ranges = {
        compute: max(values.values()) - min(values.values())
        for compute, values in by_compute.items()
        if set(values) == set(_LOGICAL_TASKS)
    }
    max_range_compute, max_range = max(
        ranges.items(), key=lambda item: (item[1], item[0]), default=(0.0, 0.0)
    )
    range_satisfied = max_range >= range_minimum
    add(
        range_attempt.id,
        {
            "fixed_compute_ranges": {
                f"{compute:.17g}": value for compute, value in sorted(ranges.items())
            },
            "maximum_range": max_range,
            "maximum_range_compute": max_range_compute,
            "required_range": range_minimum,
            "satisfied": range_satisfied,
        },
        ValidationOutcome.REPRODUCED
        if range_satisfied
        else ValidationOutcome.NOT_REPRODUCED,
        (f"Maximum complete fixed-compute task range={max_range:.12g}.",),
    )

    orders_attempt = threshold_attempts["dd-0175-default"]
    orders_rule = _comparison_rule(contract, orders_attempt)
    required_orders_ratio = _parameter(
        orders_rule, ComparisonParameterName.COMPUTE_RATIO_THRESHOLD
    )
    predictable_compute = tuple(
        float(row["compute"])
        for row in rows
        if row["task"] == "arc_easy"
        and float(row["decision_accuracy"]) >= accuracy_thresholds[orders_attempt.id]
        and float(row["compute"]) > 0
    )
    observed_orders_ratio = (
        max(predictable_compute) / min(predictable_compute)
        if predictable_compute
        else 0.0
    )
    orders_satisfied = observed_orders_ratio >= required_orders_ratio
    add(
        orders_attempt.id,
        {
            "accuracy_threshold": accuracy_thresholds[orders_attempt.id],
            "minimum_predictable_compute": min(predictable_compute, default=0.0),
            "maximum_predictable_compute": max(predictable_compute, default=0.0),
            "compute_ratio": observed_orders_ratio,
            "required_ratio": required_orders_ratio,
            "satisfied": orders_satisfied,
        },
        ValidationOutcome.REPRODUCED
        if orders_satisfied
        else ValidationOutcome.NOT_REPRODUCED,
        (f"ARC Easy predictable-compute span ratio={observed_orders_ratio:.12g}.",),
    )

    boolq_attempt = _attempt(contract, "dd-0176-default")
    boolq_rule = _comparison_rule(contract, boolq_attempt)
    boolq_threshold = _parameter(
        boolq_rule, ComparisonParameterName.NONTRIVIAL_ACCURACY_THRESHOLD
    )
    chance_baseline = _parameter(boolq_rule, ComparisonParameterName.CHANCE_BASELINE)
    trivial_tolerance = _parameter(
        boolq_rule, ComparisonParameterName.TRIVIAL_TOLERANCE
    )
    if not math.isclose(
        boolq_threshold,
        chance_baseline + trivial_tolerance,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{boolq_rule.id} nontrivial threshold must equal chance baseline plus "
            "trivial tolerance"
        )
    latest_1b = max(
        int(row["step"])
        for row in rows
        if row["task"] == "boolq" and row["model_size"] == "1B"
    )
    boolq_nontrivial = tuple(
        row
        for row in rows
        if row["task"] == "boolq" and float(row["decision_accuracy"]) > boolq_threshold
    )
    boolq_satisfied = bool(boolq_nontrivial) and all(
        row["model_size"] == "1B" and int(row["step"]) < latest_1b
        for row in boolq_nontrivial
    )
    add(
        boolq_attempt.id,
        {
            "chance_baseline": chance_baseline,
            "trivial_tolerance": trivial_tolerance,
            "nontrivial_threshold": boolq_threshold,
            "nontrivial_point_count": len(boolq_nontrivial),
            "nontrivial_points": [
                {
                    "model_size": row["model_size"],
                    "step": row["step"],
                    "decision_accuracy": row["decision_accuracy"],
                }
                for row in boolq_nontrivial
            ],
            "latest_1b_step": latest_1b,
            "satisfied": boolq_satisfied,
        },
        ValidationOutcome.REPRODUCED
        if boolq_satisfied
        else ValidationOutcome.NOT_REPRODUCED,
        (f"Found {len(boolq_nontrivial)} nontrivial BoolQ points.",),
    )

    for attempt_id, task in (
        ("dd-0177-default", "hellaswag"),
        ("dd-0178-default", "socialiqa"),
        ("dd-0179-default", "winogrande"),
    ):
        attempt = _attempt(contract, attempt_id)
        rule = _comparison_rule(contract, attempt)
        diagnostic = _plateau_diagnostic(rows, task)
        required_improvement = _parameter(
            rule, ComparisonParameterName.PLATEAU_SSE_IMPROVEMENT_MINIMUM
        )
        early_maximum = _parameter(
            rule, ComparisonParameterName.EARLY_SLOPE_ABSOLUTE_MAXIMUM
        )
        late_minimum = _parameter(rule, ComparisonParameterName.LATE_SLOPE_MINIMUM)
        satisfied = (
            float(diagnostic["sse_improvement"]) >= required_improvement
            and abs(float(diagnostic["early_slope"])) <= early_maximum
            and float(diagnostic["late_slope"]) > late_minimum
        )
        add(
            attempt_id,
            {
                **diagnostic,
                "required_sse_improvement": required_improvement,
                "early_slope_absolute_maximum": early_maximum,
                "late_slope_minimum": late_minimum,
                "satisfied": satisfied,
            },
            ValidationOutcome.REPRODUCED
            if satisfied
            else ValidationOutcome.NOT_REPRODUCED,
            (
                f"{task}: SSE improvement={float(diagnostic['sse_improvement']):.12g}, early slope={float(diagnostic['early_slope']):.12g}, late slope={float(diagnostic['late_slope']):.12g}.",
            ),
        )
    return tuple(results), tuple(series)


def run_single_scale_attempts(
    *,
    repository_root: Path,
    data_root: Path,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
) -> tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]]:
    """Run the implemented aggregate OLMES format-3 validation attempts."""
    del repository_root
    observations, digest, columns = _load_observations(
        data_root=data_root,
        contract=contract,
        input_identities=input_identities,
    )
    checkpoints = _available_checkpoints(observations)
    headline = _headline_results(
        registry=registry,
        contract=contract,
        checkpoints=checkpoints,
        columns=columns,
        parquet_sha256=digest,
    )
    plot_result, series = _aggregate_plot(
        registry=registry,
        contract=contract,
        checkpoints=checkpoints,
        columns=columns,
        parquet_sha256=digest,
    )
    qualitative, qualitative_series = _single_scale_qualitative_attempts(
        registry=registry,
        contract=contract,
        headline=headline,
        aggregate_result=plot_result,
        aggregate_series=series,
    )
    sensitivity_results: list[AttemptResult] = []
    qualitative_by_id = {result.attempt_id: result for result in qualitative}
    for attempt in contract.attempts:
        if attempt.analysis_id is not AnalysisId.SINGLE_SCALE:
            continue
        rule = _comparison_rule(contract, attempt)
        for (
            sensitivity_id,
            name,
            value,
            sensitivity_rule,
        ) in _comparison_sensitivity_rules(attempt, rule):
            rerun, _ = _single_scale_qualitative_attempts(
                registry=registry,
                contract=_contract_with_rule(contract, sensitivity_rule),
                headline=headline,
                aggregate_result=plot_result,
                aggregate_series=series,
            )
            rerun_by_id = {result.attempt_id: result for result in rerun}
            if attempt.id not in rerun_by_id or attempt.id not in qualitative_by_id:
                raise ValueError(
                    f"single-scale comparison sensitivity has no result for {attempt.id}"
                )
            sensitivity_results.append(
                _comparison_sensitivity_result(
                    rerun_by_id[attempt.id],
                    sensitivity_id=sensitivity_id,
                    parameter_name=name,
                    parameter_value=value,
                )
            )
    return tuple(
        sorted(
            (*headline, plot_result, *qualitative, *sensitivity_results),
            key=lambda item: item.attempt_id,
        )
    ), tuple(sorted((series, *qualitative_series), key=lambda item: item.id))


def run_per_task_attempts(
    *,
    repository_root: Path,
    data_root: Path,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
) -> tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]]:
    """Run the implemented per-task OLMES format-3 validation attempt."""
    del repository_root
    observations, digest, columns = _load_observations(
        data_root=data_root,
        contract=contract,
        input_identities=input_identities,
    )
    checkpoints = _available_checkpoints(observations)
    result, series = _per_task_plot(
        registry=registry,
        contract=contract,
        checkpoints=checkpoints,
        columns=columns,
        parquet_sha256=digest,
    )
    qualitative, qualitative_series = _per_task_qualitative_attempts(
        registry=registry,
        contract=contract,
        evidence=result,
        base_series=series,
    )
    sensitivity_results: list[AttemptResult] = []
    qualitative_by_id = {item.attempt_id: item for item in qualitative}
    for attempt in contract.attempts:
        if attempt.analysis_id is not AnalysisId.PER_TASK:
            continue
        rule = _comparison_rule(contract, attempt)
        for (
            sensitivity_id,
            name,
            value,
            sensitivity_rule,
        ) in _comparison_sensitivity_rules(attempt, rule):
            rerun, _ = _per_task_qualitative_attempts(
                registry=registry,
                contract=_contract_with_rule(contract, sensitivity_rule),
                evidence=result,
                base_series=series,
            )
            rerun_by_id = {item.attempt_id: item for item in rerun}
            if attempt.id not in rerun_by_id or attempt.id not in qualitative_by_id:
                raise ValueError(
                    f"per-task comparison sensitivity has no result for {attempt.id}"
                )
            sensitivity_results.append(
                _comparison_sensitivity_result(
                    rerun_by_id[attempt.id],
                    sensitivity_id=sensitivity_id,
                    parameter_name=name,
                    parameter_value=value,
                )
            )
    return tuple(
        sorted(
            (result, *qualitative, *sensitivity_results),
            key=lambda item: item.attempt_id,
        )
    ), tuple(sorted((series, *qualitative_series), key=lambda item: item.id))


__all__ = ["run_per_task_attempts", "run_single_scale_attempts"]

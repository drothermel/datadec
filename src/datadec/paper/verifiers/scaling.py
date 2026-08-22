from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from datadec.config import ScalingLawContract
from datadec.paper.contracts import load_toml_model
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
    ComparisonRule,
    ContentIdentity,
    DimensionValue,
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
from datadec.paper.scaling import (
    AccuracyAtCompute,
    CheckpointLoss,
    EvaluationPoint,
    HeldOutPrediction,
    PairDecisionAccuracy,
    PredictionCell,
    PredictionErrorSummary,
    RankingAtSize,
    ScaleCompute,
    ScalingCoordinates,
    ScalingFitError,
    ScalingTarget,
    ScalingVariant,
    SizeSubset,
    adjacent_size_crossover_counts,
    aggregate_final_losses,
    compare_stepwise_single_scale,
    construct_13_size_subsets,
    held_out_prediction,
    pair_decision_accuracy,
    prediction_errors,
    summed_compute,
)
from datadec.paper.single_scale import DEFAULT_TASK_GROUPING

_SCALING_INPUT_ID = "scaling_evaluations"
_SCALING_CONFIG_PATH = Path("configs/scaling_law.toml")
_DEFAULT_SEED = "default"
_TARGET_SIZE = "1B"
_TASK_LOSS_COLUMN = "logits_per_byte_corr"
_SCORE_COLUMN = "primary_metric"
_LOGICAL_MMLU_TASK = "mmlu"
_OLMES_MACRO_TASK = "olmes_macro"
_SOURCE_KEY_COLUMNS: tuple[str, ...] = (
    "recipe",
    "params",
    "seed",
    "step",
    "task",
)
_PLOT_SERIES_BY_CLAIM: dict[str, str] = {
    "DD-0180": "dd-0180-paper-analog",
    "DD-0368": "dd-0368-paper-analog",
    "DD-0369": "dd-0369-paper-analog",
}
_ERROR_VARIANT_BY_CLAIM: dict[str, ScalingVariant] = {
    "DD-0301": ScalingVariant.HELPER_LATE,
    "DD-0302": ScalingVariant.HELPER,
    "DD-0303": ScalingVariant.LATE,
    "DD-0304": ScalingVariant.THREE_PARAMETER_TWO_STAGE,
    "DD-0305": ScalingVariant.TWO_PARAMETER_TWO_STAGE,
    "DD-0306": ScalingVariant.FIVE_PARAMETER_ND_SINGLE_STEP,
    "DD-0307": ScalingVariant.THREE_PARAMETER_SINGLE_STEP,
    "DD-0308": ScalingVariant.FIVE_PARAMETER_ND_TWO_STAGE,
}
_ERROR_TARGETS: dict[str, tuple[float, float]] = {
    "DD-0301": (5.6, 2.6),
    "DD-0302": (6.0, 2.8),
    "DD-0303": (5.9, 2.9),
    "DD-0304": (6.5, 3.1),
    "DD-0305": (6.5, 3.2),
    "DD-0306": (42.8, 17.4),
    "DD-0307": (42.9, 42.3),
    "DD-0308": (230.8, 65.4),
}


@dataclass(frozen=True, slots=True)
class _AggregatePoint:
    recipe: str
    size_id: str
    logical_task: str
    step: int
    coordinates: ScalingCoordinates
    progress: float
    loss: float
    score: float


@dataclass(frozen=True, slots=True)
class _PreparedEvidence:
    points: tuple[_AggregatePoint, ...]
    complete_groups: frozenset[tuple[str, str]]
    missing_groups: tuple[str, ...]
    target_scores: tuple[tuple[str, str, float], ...]
    target_coordinates: ScalingCoordinates | None
    target_step: int
    target_selected_group_count: int
    single_scale_points: tuple[AccuracyAtCompute, ...]
    crossover_count: int
    crossover_comparable_pairs: int
    zero_compute_row_count: int


@dataclass(frozen=True, slots=True)
class _SubsetPrediction:
    variant: ScalingVariant
    subset: SizeSubset
    compute: float
    percent_target_compute: float
    decision: PairDecisionAccuracy
    predictions: tuple[tuple[str, float], ...]
    frontier_accuracy: float | None = None
    frontier_difference: float | None = None


@dataclass(frozen=True, slots=True)
class _Analysis:
    predictions: tuple[_SubsetPrediction, ...]
    errors: tuple[tuple[ScalingVariant, PredictionErrorSummary], ...]
    required_subset_count: int
    supported_subset_count: int
    structural_skip_count: int

    @property
    def errors_by_variant(self) -> dict[ScalingVariant, PredictionErrorSummary]:
        return dict(self.errors)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_key_sha256(frame: pd.DataFrame) -> str:
    keys = sorted(
        tuple(row)
        for row in frame.loc[:, list(_SOURCE_KEY_COLUMNS)].itertuples(
            index=False, name=None
        )
    )
    payload = json.dumps(keys, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _mean(values: Iterable[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("at least one value is required")
    return math.fsum(ordered) / len(ordered)


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a real number: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite: {value!r}")
    return result


def _non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer: {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be an integer: {value!r}") from error
    if result != value or result < 0:
        raise ValueError(f"{label} must be a non-negative integer: {value!r}")
    return result


def _load_scaling_contract(repository_root: Path) -> ScalingLawContract:
    return load_toml_model(repository_root / _SCALING_CONFIG_PATH, ScalingLawContract)


def _scaling_specs(contract: PaperValidationContract) -> tuple[AttemptSpec, ...]:
    return tuple(
        sorted(
            (
                attempt
                for attempt in contract.attempts
                if attempt.analysis_id is AnalysisId.SCALING_LAW
            ),
            key=lambda attempt: attempt.id,
        )
    )


def _resolve_input(
    contract: PaperValidationContract,
) -> tuple[InputTableSpec, tuple[str, ...]]:
    input_spec = next(
        (item for item in contract.inputs if item.id == _SCALING_INPUT_ID), None
    )
    if input_spec is None:
        raise ValueError("paper-validation contract has no scaling_evaluations input")
    required_columns = (
        *_SOURCE_KEY_COLUMNS,
        "tokens",
        "compute",
        "exact_parameter_count",
        _SCORE_COLUMN,
        _TASK_LOSS_COLUMN,
    )
    missing = tuple(sorted(set(required_columns).difference(input_spec.columns)))
    if missing:
        raise ValueError(
            f"scaling_evaluations contract is missing adapter columns: {missing!r}"
        )
    return input_spec, required_columns


def _validate_identity(
    *,
    table_id: str,
    table_path: str,
    actual_sha256: str,
    input_identities: Mapping[str, ContentIdentity],
) -> None:
    candidates = tuple(
        identity
        for key, identity in input_identities.items()
        if key in {table_id, table_path} or identity.id in {table_id, table_path}
    )
    if not candidates:
        return
    mismatches = tuple(
        identity for identity in candidates if identity.sha256 != actual_sha256
    )
    if mismatches:
        raise ValueError(
            "scaling_evaluations changed after input identity capture: "
            f"actual={actual_sha256}, recorded={tuple(i.sha256 for i in mismatches)!r}"
        )


def _select_frame(
    frame: pd.DataFrame,
    *,
    recipes: tuple[str, ...],
    models: tuple[str, ...],
    source_tasks: tuple[str, ...],
) -> pd.DataFrame:
    missing_columns = tuple(
        sorted(
            {
                *_SOURCE_KEY_COLUMNS,
                "tokens",
                "compute",
                "exact_parameter_count",
                _SCORE_COLUMN,
                _TASK_LOSS_COLUMN,
            }.difference(frame.columns)
        )
    )
    if missing_columns:
        raise ValueError(
            f"normalized scaling evaluations are missing columns: {missing_columns!r}"
        )
    null_keys = tuple(
        column for column in _SOURCE_KEY_COLUMNS if frame[column].isna().any()
    )
    if null_keys:
        raise ValueError(f"normalized scaling keys contain nulls: {null_keys!r}")
    duplicates = frame.duplicated(list(_SOURCE_KEY_COLUMNS), keep=False)
    if duplicates.any():
        duplicate_keys = tuple(
            tuple(row)
            for row in frame.loc[duplicates, list(_SOURCE_KEY_COLUMNS)]
            .drop_duplicates()
            .sort_values(list(_SOURCE_KEY_COLUMNS), kind="stable")
            .itertuples(index=False, name=None)
        )
        raise ValueError(f"duplicate normalized scaling keys: {duplicate_keys!r}")
    selected = frame[
        frame["recipe"].isin(recipes)
        & frame["params"].isin(models)
        & (frame["seed"] == _DEFAULT_SEED)
        & frame["task"].isin(source_tasks)
    ].copy()
    return selected.sort_values(list(_SOURCE_KEY_COLUMNS), kind="stable")


def _uniform_coordinates(rows: pd.DataFrame, *, label: str) -> ScalingCoordinates:
    parameter_counts = {
        _finite(value, label=f"{label} parameter count")
        for value in rows["exact_parameter_count"]
    }
    token_counts = {
        _finite(value, label=f"{label} token count") for value in rows["tokens"]
    }
    computes = {_finite(value, label=f"{label} compute") for value in rows["compute"]}
    if len(parameter_counts) != 1 or len(token_counts) != 1 or len(computes) != 1:
        raise ValueError(f"normalized scaling coordinates differ within {label}")
    return ScalingCoordinates(
        parameter_count=next(iter(parameter_counts)),
        token_count=next(iter(token_counts)),
        compute=next(iter(computes)),
    )


def _aggregate_tasks(
    rows: pd.DataFrame,
    *,
    recipe: str,
    size_id: str,
    step: int,
    coordinates: ScalingCoordinates,
    progress: float,
) -> tuple[_AggregatePoint, ...] | None:
    task_rows = rows.set_index("task")
    if not task_rows.index.is_unique:
        raise ValueError(f"duplicate task rows for {recipe}/{size_id}/{step}")
    source_tasks = set(DEFAULT_TASK_GROUPING.source_tasks)
    if set(task_rows.index) != source_tasks:
        return None
    if task_rows[[_TASK_LOSS_COLUMN, _SCORE_COLUMN]].isna().any().any():
        return None

    points: list[_AggregatePoint] = []
    logical_values: list[tuple[str, float, float]] = []
    for task in DEFAULT_TASK_GROUPING.non_mmlu_tasks:
        row = task_rows.loc[task]
        logical_values.append(
            (
                task,
                _finite(row[_TASK_LOSS_COLUMN], label=f"{task} task loss"),
                _finite(row[_SCORE_COLUMN], label=f"{task} primary metric"),
            )
        )
    mmlu_loss = _mean(
        _finite(
            task_rows.loc[task, _TASK_LOSS_COLUMN],
            label=f"{task} task loss",
        )
        for task in DEFAULT_TASK_GROUPING.mmlu_subjects
    )
    mmlu_score = _mean(
        _finite(
            task_rows.loc[task, _SCORE_COLUMN],
            label=f"{task} primary metric",
        )
        for task in DEFAULT_TASK_GROUPING.mmlu_subjects
    )
    logical_values.append((_LOGICAL_MMLU_TASK, mmlu_loss, mmlu_score))
    logical_values.sort(key=lambda value: value[0])
    macro_loss = _mean(value[1] for value in logical_values)
    macro_score = _mean(value[2] for value in logical_values)
    logical_values.append((_OLMES_MACRO_TASK, macro_loss, macro_score))
    for logical_task, loss, score in logical_values:
        points.append(
            _AggregatePoint(
                recipe=recipe,
                size_id=size_id,
                logical_task=logical_task,
                step=step,
                coordinates=coordinates,
                progress=progress,
                loss=loss,
                score=score,
            )
        )
    return tuple(points)


def _aggregate_scores_only(
    rows: pd.DataFrame,
) -> tuple[tuple[str, float], ...] | None:
    task_rows = rows.set_index("task")
    if not task_rows.index.is_unique:
        raise ValueError("duplicate task rows in primary-metric aggregate")
    if set(task_rows.index) != set(DEFAULT_TASK_GROUPING.source_tasks):
        return None
    if task_rows[_SCORE_COLUMN].isna().any():
        return None
    logical: list[tuple[str, float]] = [
        (
            task,
            _finite(task_rows.loc[task, _SCORE_COLUMN], label=f"{task} score"),
        )
        for task in DEFAULT_TASK_GROUPING.non_mmlu_tasks
    ]
    logical.append(
        (
            _LOGICAL_MMLU_TASK,
            _mean(
                _finite(task_rows.loc[task, _SCORE_COLUMN], label=f"{task} score")
                for task in DEFAULT_TASK_GROUPING.mmlu_subjects
            ),
        )
    )
    logical.sort(key=lambda value: value[0])
    logical.append((_OLMES_MACRO_TASK, _mean(score for _, score in logical)))
    return tuple(logical)


def _missing_surface_group(
    *,
    recipe: str,
    size_id: str,
    expected_steps: tuple[int, ...],
    complete_steps: set[int],
) -> str:
    incomplete = tuple(step for step in expected_steps if step not in complete_steps)
    return (
        f"recipe={recipe}|size={size_id}|seed={_DEFAULT_SEED}|"
        f"missing=task_loss_surface|incomplete_steps={','.join(map(str, incomplete))}"
    )


def _prepare_evidence(
    selected: pd.DataFrame,
    *,
    recipes: tuple[str, ...],
    models: tuple[str, ...],
) -> _PreparedEvidence:
    fit_sizes = tuple(size for size in models if size != _TARGET_SIZE)
    positive = selected[selected["compute"] > 0]
    zero_compute_row_count = len(selected) - len(positive)
    grouped = {
        (str(recipe), str(size_id), int(step)): rows
        for (recipe, size_id, step), rows in positive.groupby(
            ["recipe", "params", "step"], sort=False
        )
    }
    expected_steps_by_size = {
        size_id: tuple(
            sorted(
                {step for _, grouped_size, step in grouped if grouped_size == size_id}
            )
        )
        for size_id in fit_sizes
    }
    final_tokens_by_size: dict[str, float] = {}
    for size_id in models:
        values = positive.loc[positive["params"] == size_id, "tokens"]
        if not values.empty:
            final_tokens_by_size[size_id] = max(
                _finite(value, label=f"{size_id} tokens") for value in values
            )

    points: list[_AggregatePoint] = []
    complete_groups: set[tuple[str, str]] = set()
    missing_groups: list[str] = []
    for recipe in recipes:
        for size_id in fit_sizes:
            expected_steps = expected_steps_by_size[size_id]
            complete_steps: set[int] = set()
            if not expected_steps or size_id not in final_tokens_by_size:
                missing_groups.append(
                    f"recipe={recipe}|size={size_id}|seed={_DEFAULT_SEED}|"
                    "missing=task_loss_surface|incomplete_steps=all"
                )
                continue
            for step in expected_steps:
                rows = grouped.get((recipe, size_id, step))
                if rows is None:
                    continue
                coordinates = _uniform_coordinates(
                    rows, label=f"{recipe}/{size_id}/{step}"
                )
                progress = coordinates.token_count / final_tokens_by_size[size_id]
                aggregated = _aggregate_tasks(
                    rows,
                    recipe=recipe,
                    size_id=size_id,
                    step=step,
                    coordinates=coordinates,
                    progress=progress,
                )
                if aggregated is None:
                    continue
                points.extend(aggregated)
                complete_steps.add(step)
            if complete_steps == set(expected_steps):
                complete_groups.add((recipe, size_id))
            else:
                missing_groups.append(
                    _missing_surface_group(
                        recipe=recipe,
                        size_id=size_id,
                        expected_steps=expected_steps,
                        complete_steps=complete_steps,
                    )
                )

    target_scores: list[tuple[str, str, float]] = []
    target_coordinates: ScalingCoordinates | None = None
    target_step = 0
    target_selected_group_count = 0
    target_rows = positive[positive["params"] == _TARGET_SIZE]
    if not target_rows.empty:
        target_token_count = max(
            _finite(value, label="1B tokens") for value in target_rows["tokens"]
        )
        target_final = target_rows[target_rows["tokens"] == target_token_count]
        target_step_values = {
            _non_negative_int(value, label="1B target step")
            for value in target_final["step"]
        }
        if len(target_step_values) != 1:
            raise ValueError("held-out 1B final token count maps to multiple steps")
        target_step = next(iter(target_step_values))
        for recipe in recipes:
            rows = grouped.get((recipe, _TARGET_SIZE, target_step))
            if rows is not None:
                target_selected_group_count += int(rows[_SCORE_COLUMN].notna().sum())
            aggregated = _aggregate_scores_only(rows) if rows is not None else None
            if aggregated is None:
                missing_groups.append(
                    f"recipe={recipe}|size={_TARGET_SIZE}|seed={_DEFAULT_SEED}|"
                    f"step={target_step}|missing=target_primary_metric_tasks"
                )
                continue
            assert rows is not None
            coordinates = _uniform_coordinates(
                rows, label=f"{recipe}/{_TARGET_SIZE}/{target_step}"
            )
            if target_coordinates is None:
                target_coordinates = coordinates
            elif target_coordinates != coordinates:
                raise ValueError("held-out 1B coordinates differ across recipes")
            target_scores.extend(
                (recipe, logical_task, score) for logical_task, score in aggregated
            )
    else:
        missing_groups.extend(
            f"recipe={recipe}|size={_TARGET_SIZE}|seed={_DEFAULT_SEED}|"
            "missing=target_primary_metric_tasks"
            for recipe in recipes
        )

    target_macro = {
        recipe: score
        for recipe, task, score in target_scores
        if task == _OLMES_MACRO_TASK
    }
    single_points, rankings = _single_scale_evidence(
        positive,
        recipes=recipes,
        models=models,
        target_macro=target_macro,
    )
    crossover_count = crossover_comparable = 0
    if len(rankings) >= 2:
        crossovers = adjacent_size_crossover_counts(rankings)
        crossover_count = sum(item.crossover_count for item in crossovers)
        crossover_comparable = sum(item.comparable_pairs for item in crossovers)
    return _PreparedEvidence(
        points=tuple(
            sorted(
                points,
                key=lambda point: (
                    point.recipe,
                    point.logical_task,
                    models.index(point.size_id),
                    point.step,
                ),
            )
        ),
        complete_groups=frozenset(complete_groups),
        missing_groups=tuple(sorted(set(missing_groups))),
        target_scores=tuple(sorted(target_scores)),
        target_coordinates=target_coordinates,
        target_step=target_step,
        target_selected_group_count=target_selected_group_count,
        single_scale_points=single_points,
        crossover_count=crossover_count,
        crossover_comparable_pairs=crossover_comparable,
        zero_compute_row_count=zero_compute_row_count,
    )


def _single_scale_evidence(
    positive: pd.DataFrame,
    *,
    recipes: tuple[str, ...],
    models: tuple[str, ...],
    target_macro: dict[str, float],
) -> tuple[tuple[AccuracyAtCompute, ...], tuple[RankingAtSize, ...]]:
    if set(target_macro) != set(recipes):
        return (), ()
    target_ranks = tuple((recipe, -target_macro[recipe]) for recipe in recipes)
    grouped = {
        (str(recipe), str(size_id), int(step)): rows
        for (recipe, size_id, step), rows in positive.groupby(
            ["recipe", "params", "step"], sort=False
        )
    }
    points: list[AccuracyAtCompute] = []
    final_rankings: list[RankingAtSize] = []
    for size_id in models:
        steps = tuple(
            sorted(
                {step for _, grouped_size, step in grouped if grouped_size == size_id}
            )
        )
        if not steps:
            continue
        complete_at_size: list[tuple[int, float, tuple[tuple[str, float], ...]]] = []
        for step in steps:
            scores: list[tuple[str, float]] = []
            computes: set[float] = set()
            for recipe in recipes:
                rows = grouped.get((recipe, size_id, step))
                aggregated = _aggregate_scores_only(rows) if rows is not None else None
                if aggregated is None:
                    break
                assert rows is not None
                scores.append((recipe, dict(aggregated)[_OLMES_MACRO_TASK]))
                computes.add(
                    _uniform_coordinates(
                        rows, label=f"single/{recipe}/{size_id}/{step}"
                    ).compute
                )
            if len(scores) != len(recipes):
                continue
            if len(computes) != 1:
                raise ValueError(
                    f"single-scale compute differs across recipes for {size_id}/{step}"
                )
            decision = pair_decision_accuracy(target_ranks, scores)
            compute = next(iter(computes))
            complete_at_size.append((step, compute, tuple(scores)))
            points.append(
                AccuracyAtCompute(
                    point_id=f"single-{size_id}-{step}",
                    compute=compute,
                    accuracy=decision.accuracy,
                )
            )
        if complete_at_size:
            _, _, final_scores = complete_at_size[-1]
            final_rankings.append(RankingAtSize(size_id, final_scores))
    return tuple(points), tuple(final_rankings)


def _cell_points(
    evidence: _PreparedEvidence,
    *,
    recipe: str,
    logical_task: str,
    sizes: tuple[str, ...],
) -> tuple[_AggregatePoint, ...]:
    selected_sizes = set(sizes)
    return tuple(
        point
        for point in evidence.points
        if point.recipe == recipe
        and point.logical_task == logical_task
        and point.size_id in selected_sizes
    )


def _predict_cell(
    evidence: _PreparedEvidence,
    *,
    recipe: str,
    logical_task: str,
    sizes: tuple[str, ...],
    variant: ScalingVariant,
) -> HeldOutPrediction:
    points = _cell_points(
        evidence, recipe=recipe, logical_task=logical_task, sizes=sizes
    )
    checkpoint_losses = tuple(
        CheckpointLoss(
            size_id=point.size_id,
            coordinates=point.coordinates,
            progress=point.progress,
            loss=point.loss,
        )
        for point in points
    )
    final_losses = aggregate_final_losses(checkpoint_losses)
    evaluations = tuple(
        EvaluationPoint(
            size_id=point.size_id,
            coordinates=point.coordinates,
            progress=point.progress,
            loss=point.loss,
            score=point.score,
        )
        for point in points
    )
    target_lookup = {
        (target_recipe, target_task): score
        for target_recipe, target_task, score in evidence.target_scores
    }
    assert evidence.target_coordinates is not None
    return held_out_prediction(
        final_losses,
        evaluations,
        target=ScalingTarget(
            size_id=_TARGET_SIZE,
            coordinates=evidence.target_coordinates,
            actual_score=target_lookup[(recipe, logical_task)],
        ),
        variant=variant,
    )


def _subset_has_evidence(
    subset: SizeSubset,
    *,
    recipes: tuple[str, ...],
    complete_groups: frozenset[tuple[str, str]],
) -> bool:
    return all(
        (recipe, size_id) in complete_groups
        for recipe in recipes
        for size_id in subset.sizes
    )


def _analyze(
    evidence: _PreparedEvidence,
    *,
    recipes: tuple[str, ...],
    fit_sizes: tuple[str, ...],
) -> _Analysis:
    subsets = construct_13_size_subsets(fit_sizes)
    target_macro = {
        recipe: score
        for recipe, task, score in evidence.target_scores
        if task == _OLMES_MACRO_TASK
    }
    if evidence.target_coordinates is None or set(target_macro) != set(recipes):
        return _Analysis((), (), len(subsets), 0, 0)
    target_ranks = tuple((recipe, -target_macro[recipe]) for recipe in recipes)
    final_compute_by_size: dict[str, float] = {}
    for size_id in fit_sizes:
        values = {
            point.coordinates.compute
            for point in evidence.points
            if point.size_id == size_id and math.isclose(point.progress, 1.0)
        }
        if len(values) == 1:
            final_compute_by_size[size_id] = next(iter(values))

    predictions: list[_SubsetPrediction] = []
    supported_subsets: set[str] = set()
    structural_skips = 0
    for subset in subsets:
        if not _subset_has_evidence(
            subset, recipes=recipes, complete_groups=evidence.complete_groups
        ):
            continue
        if not set(subset.sizes).issubset(final_compute_by_size):
            continue
        supported_subsets.add(subset.subset_id)
        compute = summed_compute(
            tuple(
                ScaleCompute(size_id, final_compute_by_size[size_id])
                for size_id in subset.sizes
            )
        )
        for variant in ScalingVariant:
            predicted: list[tuple[str, float]] = []
            try:
                for recipe in recipes:
                    prediction = _predict_cell(
                        evidence,
                        recipe=recipe,
                        logical_task=_OLMES_MACRO_TASK,
                        sizes=subset.sizes,
                        variant=variant,
                    )
                    predicted.append((recipe, prediction.predicted_score))
            except ScalingFitError as error:
                from datadec.paper.scaling import FitFailureReason

                if error.failure.reason is FitFailureReason.INSUFFICIENT_OBSERVATIONS:
                    structural_skips += 1
                    continue
                raise
            decision = pair_decision_accuracy(target_ranks, predicted)
            predictions.append(
                _SubsetPrediction(
                    variant=variant,
                    subset=subset,
                    compute=compute,
                    percent_target_compute=compute
                    / evidence.target_coordinates.compute
                    * 100,
                    decision=decision,
                    predictions=tuple(predicted),
                )
            )

    multi_points = tuple(
        AccuracyAtCompute(
            point_id=f"{item.variant.value}/{item.subset.subset_id}",
            compute=item.compute,
            accuracy=item.decision.accuracy,
        )
        for item in predictions
    )
    comparisons = {
        item.multi_scale.point_id: item
        for item in compare_stepwise_single_scale(
            multi_points, evidence.single_scale_points
        )
    }
    predictions = [
        _SubsetPrediction(
            variant=item.variant,
            subset=item.subset,
            compute=item.compute,
            percent_target_compute=item.percent_target_compute,
            decision=item.decision,
            predictions=item.predictions,
            frontier_accuracy=(
                comparisons[
                    f"{item.variant.value}/{item.subset.subset_id}"
                ].single_scale.accuracy
                if f"{item.variant.value}/{item.subset.subset_id}" in comparisons
                else None
            ),
            frontier_difference=(
                comparisons[
                    f"{item.variant.value}/{item.subset.subset_id}"
                ].accuracy_difference
                if f"{item.variant.value}/{item.subset.subset_id}" in comparisons
                else None
            ),
        )
        for item in predictions
    ]

    errors: list[tuple[ScalingVariant, PredictionErrorSummary]] = []
    full_subset = next(subset for subset in subsets if subset.sizes == fit_sizes)
    if _subset_has_evidence(
        full_subset, recipes=recipes, complete_groups=evidence.complete_groups
    ):
        logical_tasks = tuple(
            sorted((*DEFAULT_TASK_GROUPING.non_mmlu_tasks, _LOGICAL_MMLU_TASK))
        )
        for variant in ScalingVariant:
            cells: list[PredictionCell] = []
            for recipe in recipes:
                for task in logical_tasks:
                    prediction = _predict_cell(
                        evidence,
                        recipe=recipe,
                        logical_task=task,
                        sizes=fit_sizes,
                        variant=variant,
                    )
                    cells.append(
                        PredictionCell(
                            cell_id=f"{recipe}|{task}",
                            predicted=prediction.predicted_score,
                            actual=prediction.target.actual_score,
                        )
                    )
            errors.append((variant, prediction_errors(cells)))
    return _Analysis(
        predictions=tuple(
            sorted(
                predictions,
                key=lambda item: (item.variant.value, item.subset.subset_id),
            )
        ),
        errors=tuple(errors),
        required_subset_count=len(subsets),
        supported_subset_count=len(supported_subsets),
        structural_skip_count=structural_skips,
    )


def _plot_series(
    *,
    spec: AttemptSpec,
    analysis: _Analysis,
    recipe_count: int,
) -> PlotSeries | None:
    series_id = _PLOT_SERIES_BY_CLAIM.get(spec.claim_id)
    if series_id is None:
        return None
    selected = analysis.predictions
    if spec.claim_id == "DD-0368":
        selected = tuple(
            item
            for item in selected
            if item.variant is ScalingVariant.FIVE_PARAMETER_ND_TWO_STAGE
        )
    elif spec.claim_id == "DD-0369":
        selected = tuple(
            item
            for item in selected
            if item.variant is ScalingVariant.THREE_PARAMETER_SINGLE_STEP
        )
    selected = tuple(item for item in selected if item.frontier_accuracy is not None)
    if not selected:
        return None
    points = tuple(
        PlotPoint(
            dimensions=(
                DimensionValue(name="variant", value=item.variant.value),
                DimensionValue(name="subset", value=item.subset.subset_id),
                DimensionValue(name="subset_kind", value=item.subset.kind),
            ),
            measures=(
                MeasureValue(
                    name="percent_target_compute", value=item.percent_target_compute
                ),
                MeasureValue(name="decision_accuracy", value=item.decision.accuracy),
                MeasureValue(
                    name="single_scale_frontier_accuracy",
                    value=item.frontier_accuracy,
                ),
                MeasureValue(
                    name="frontier_difference", value=item.frontier_difference
                ),
            ),
        )
        for item in selected
    )
    return PlotSeries(
        id=series_id,
        figure="fig:all_scaling_laws_accuracy_vs_compute_shaded",
        panel="main",
        semantic_kind="scaling-law-versus-single-scale-frontier",
        x_axis=AxisSpec(
            measure="percent_target_compute", scale=AxisScale.LOG, unit="percent"
        ),
        y_axis=AxisSpec(
            measure="decision_accuracy", scale=AxisScale.LINEAR, unit="proportion"
        ),
        dimensions=("variant", "subset", "subset_kind"),
        measures=(
            "percent_target_compute",
            "decision_accuracy",
            "single_scale_frontier_accuracy",
            "frontier_difference",
        ),
        attempt_id=spec.id,
        counts=(
            NamedCount(name="recipes", value=recipe_count),
            NamedCount(name="points", value=len(points)),
        ),
        points=points,
    )


def _common_computed_value(
    evidence: _PreparedEvidence, analysis: _Analysis
) -> dict[str, object]:
    comparable = tuple(
        item for item in analysis.predictions if item.frontier_difference is not None
    )
    return {
        "variant_count": len(ScalingVariant),
        "required_subset_count": analysis.required_subset_count,
        "supported_subset_count": analysis.supported_subset_count,
        "prediction_point_count": len(analysis.predictions),
        "frontier_comparison_count": len(comparable),
        "maximum_frontier_difference": (
            max(item.frontier_difference for item in comparable) if comparable else None
        ),
        "adjacent_crossover_count": evidence.crossover_count,
        "adjacent_crossover_comparable_pairs": evidence.crossover_comparable_pairs,
    }


def _computed_for_claim(
    claim_id: str,
    *,
    evidence: _PreparedEvidence,
    analysis: _Analysis,
) -> tuple[object, bool]:
    common = _common_computed_value(evidence, analysis)
    comparable = tuple(
        item for item in analysis.predictions if item.frontier_difference is not None
    )
    frontier_holds = bool(comparable) and all(
        item.frontier_difference <= 0 for item in comparable
    )
    if claim_id in {"DD-0013", "DD-0054", "DD-0180", "DD-0181"}:
        return ({**common, "paper_relationship_holds": frontier_holds}, frontier_holds)
    if claim_id == "DD-0119":
        baseline = {
            item.subset.subset_id: item.decision.accuracy
            for item in analysis.predictions
            if item.variant is ScalingVariant.THREE_PARAMETER_TWO_STAGE
        }
        matched = tuple(
            item for item in analysis.predictions if item.subset.subset_id in baseline
        )
        holds = bool(matched) and all(
            item.decision.accuracy <= baseline[item.subset.subset_id]
            for item in matched
        )
        return ({**common, "no_variant_exceeds_baseline": holds}, holds)
    if claim_id == "DD-0189":
        maxima = {
            variant: max(
                (
                    item.decision.accuracy
                    for item in analysis.predictions
                    if item.variant is variant
                ),
                default=-math.inf,
            )
            for variant in ScalingVariant
        }
        ordered = sorted(maxima.values(), reverse=True)
        cutoff = ordered[min(2, len(ordered) - 1)]
        holds = all(
            maxima[variant] >= cutoff
            for variant in (
                ScalingVariant.TWO_PARAMETER_TWO_STAGE,
                ScalingVariant.THREE_PARAMETER_TWO_STAGE,
            )
        )
        return (
            {
                **common,
                "maximum_decision_accuracy_by_variant": {
                    variant.value: value for variant, value in maxima.items()
                },
                "two_and_three_parameter_among_top_three": holds,
            },
            holds,
        )
    if claim_id == "DD-0192":
        maximum = max(
            (point.accuracy for point in evidence.single_scale_points), default=0.0
        )
        holds = maximum > 0.5
        return ({**common, "maximum_single_scale_accuracy": maximum}, holds)
    if claim_id in _ERROR_VARIANT_BY_CLAIM:
        variant = _ERROR_VARIANT_BY_CLAIM[claim_id]
        summary = analysis.errors_by_variant[variant]
        relative_target, absolute_target = _ERROR_TARGETS[claim_id]
        holds = (
            round(summary.mean_relative_error_percent, 1) == relative_target
            and round(summary.mean_absolute_error_percent, 1) == absolute_target
        )
        return (
            {
                "variant": variant.value,
                "relative_error_percent": summary.mean_relative_error_percent,
                "absolute_error_percent": summary.mean_absolute_error_percent,
                "cell_count": len(summary.cells),
            },
            holds,
        )
    if claim_id in {"DD-0311", "DD-0330"}:
        errors = analysis.errors_by_variant
        baseline = errors[
            ScalingVariant.THREE_PARAMETER_TWO_STAGE
        ].mean_relative_error_percent
        compared = (
            ScalingVariant.TWO_PARAMETER_TWO_STAGE,
            ScalingVariant.HELPER,
            ScalingVariant.LATE,
            ScalingVariant.HELPER_LATE,
        )
        maximum_difference = max(
            abs(errors[variant].mean_relative_error_percent - baseline)
            for variant in compared
        )
        holds = maximum_difference <= 1.0
        return (
            {
                "baseline_relative_error_percent": baseline,
                "maximum_comparable_variant_difference": maximum_difference,
            },
            holds,
        )
    if claim_id == "DD-0312":
        errors = analysis.errors_by_variant
        decision_maxima = {
            variant: max(
                item.decision.accuracy
                for item in analysis.predictions
                if item.variant is variant
            )
            for variant in ScalingVariant
        }
        high_error = sorted(
            ScalingVariant,
            key=lambda variant: errors[variant].mean_relative_error_percent,
            reverse=True,
        )[:3]
        low_decision = sorted(
            ScalingVariant,
            key=lambda variant: decision_maxima[variant],
        )[:3]
        overlap = len(set(high_error).intersection(low_decision))
        holds = overlap >= 2
        return ({"top_three_error_bottom_three_decision_overlap": overlap}, holds)
    if claim_id == "DD-0368":
        points = tuple(
            item
            for item in comparable
            if item.variant is ScalingVariant.FIVE_PARAMETER_ND_TWO_STAGE
        )
        below = sum(item.frontier_difference < 0 for item in points)
        holds = bool(points) and below > len(points) / 2
        return (
            {**common, "below_frontier_points": below, "variant_points": len(points)},
            holds,
        )
    if claim_id == "DD-0369":
        points = tuple(
            item
            for item in analysis.predictions
            if item.variant is ScalingVariant.THREE_PARAMETER_SINGLE_STEP
        )
        near_random = sum(abs(item.decision.accuracy - 0.5) <= 0.1 for item in points)
        holds = bool(points) and near_random > len(points) / 2
        return (
            {
                **common,
                "near_random_points": near_random,
                "variant_points": len(points),
            },
            holds,
        )
    return common, bool(analysis.predictions)


def _result(
    *,
    spec: AttemptSpec,
    claim: PaperClaim,
    rule: ComparisonRule,
    selection: RowSelection,
    checkpoint_selection: CheckpointSelection,
    evidence: _PreparedEvidence,
    analysis: _Analysis,
    series: PlotSeries | None,
    input_sha256: str,
) -> AttemptResult:
    incomplete = bool(evidence.missing_groups)
    if incomplete:
        computed: object = _common_computed_value(evidence, analysis)
        holds = False
        outcome = ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
    else:
        computed, holds = _computed_for_claim(
            spec.claim_id, evidence=evidence, analysis=analysis
        )
        if rule.predicate is ComparisonPredicate.NONEMPTY_PLOT:
            outcome = ValidationOutcome.DESCRIPTIVE_ONLY
        else:
            outcome = (
                ValidationOutcome.REPRODUCED
                if holds
                else ValidationOutcome.NOT_REPRODUCED
            )
    decisions = tuple(item.decision for item in analysis.predictions)
    denominator_values = {item.denominator for item in decisions}
    denominator = (
        next(iter(denominator_values)) if len(denominator_values) == 1 else None
    )
    limitations = ["Only the configured default seed is used."]
    if incomplete:
        limitations.append(
            "Full-paper scaling claims require all 25 recipes over all 13 fit "
            "sizes and the held-out 1B target; supported subset points do not "
            "complete-case the declared surface."
        )
    return AttemptResult(
        attempt_id=spec.id,
        claim_id=spec.claim_id,
        role=AttemptRole.DEFAULT if spec.default else AttemptRole.SENSITIVITY,
        parent_attempt_id=spec.parent_attempt_id,
        comparison_rule_id=spec.comparison_rule_id,
        comparison_rule_version=rule.version,
        transformation_ids=spec.transformation_ids,
        row_selections=(selection,),
        checkpoint_selections=(checkpoint_selection,),
        target_value=claim.paper_target,
        computed_value=computed,
        seeds=(_DEFAULT_SEED,),
        denominator=denominator,
        exclusions=(
            NamedCount(
                name="zero_compute_source_rows",
                value=evidence.zero_compute_row_count,
            ),
            NamedCount(
                name="structurally_unfit_variant_subsets",
                value=analysis.structural_skip_count,
            ),
        ),
        missing_groups=evidence.missing_groups,
        target_ties=sum(item.target_ties for item in decisions),
        predicted_ties=sum(item.predicted_ties for item in decisions),
        outcome=outcome,
        diagnostics=(
            f"input_sha256={input_sha256}",
            f"task_loss_column={_TASK_LOSS_COLUMN}",
            f"score_column={_SCORE_COLUMN}",
            "task_aggregation=mmlu-subject-macro-then-olmes-macro",
            "task_loss_final_window_fraction=0.10",
            f"scaling_variant_count={len(ScalingVariant)}",
            f"required_size_subset_count={analysis.required_subset_count}",
            f"supported_size_subset_count={analysis.supported_subset_count}",
            f"computed_prediction_point_count={len(analysis.predictions)}",
        ),
        limitations=tuple(limitations),
        plot_series_ids=(() if series is None else (series.id,)),
    )


def run_scaling_law_attempts(
    *,
    repository_root: Path,
    data_root: Path,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
) -> tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]]:
    """Run format-3 scaling-law attempts from normalized evaluation evidence."""
    root = Path(repository_root)
    inputs = Path(data_root)
    specs = _scaling_specs(contract)
    if not specs:
        return (), ()
    input_spec, required_columns = _resolve_input(contract)
    input_path = inputs / input_spec.path
    if not input_path.is_file():
        raise FileNotFoundError(f"missing scaling_evaluations input: {input_path}")
    input_sha256 = _sha256_file(input_path)
    _validate_identity(
        table_id=input_spec.id,
        table_path=input_spec.path,
        actual_sha256=input_sha256,
        input_identities=input_identities,
    )
    scaling_contract = _load_scaling_contract(root)
    recipes = tuple(sorted(scaling_contract.source_group_map.values()))
    models = scaling_contract.models
    if len(recipes) != 25:
        raise ValueError("paper scaling recipe universe must contain 25 recipes")
    fit_sizes = tuple(size for size in models if size != _TARGET_SIZE)
    if len(fit_sizes) != 13 or models[-1] != _TARGET_SIZE:
        raise ValueError(
            "paper scaling size universe must contain 13 ordered fit sizes then 1B"
        )
    frame = pd.read_parquet(input_path, columns=list(required_columns))
    selected = _select_frame(
        frame,
        recipes=recipes,
        models=models,
        source_tasks=DEFAULT_TASK_GROUPING.source_tasks,
    )
    selection = RowSelection(
        logical_table_id=input_spec.id,
        columns=tuple(required_columns),
        predicates=(
            RowPredicate(column="recipe", operator=PredicateOperator.IN, value=recipes),
            RowPredicate(column="params", operator=PredicateOperator.IN, value=models),
            RowPredicate(
                column="seed", operator=PredicateOperator.EQ, value=_DEFAULT_SEED
            ),
            RowPredicate(
                column="task",
                operator=PredicateOperator.IN,
                value=DEFAULT_TASK_GROUPING.source_tasks,
            ),
        ),
        local_parquet_sha256=input_sha256,
        selected_row_count=len(selected),
        selected_key_sha256=_selected_key_sha256(selected),
    )
    evidence = _prepare_evidence(selected, recipes=recipes, models=models)
    analysis = _analyze(evidence, recipes=recipes, fit_sizes=fit_sizes)
    checkpoint_selection = CheckpointSelection(
        requested_meaning="held-out 1B final primary-metric target",
        rule=CheckpointRule.EXACT,
        actual_step=evidence.target_step,
        completeness_dimensions=contract.checkpoint_policy.completeness_dimensions,
        expected_group_count=len(recipes) * len(DEFAULT_TASK_GROUPING.source_tasks),
        selected_group_count=evidence.target_selected_group_count,
    )
    claims = {claim.id: claim for claim in registry.claims}
    rules = {rule.id: rule for rule in contract.comparison_rules}
    results: list[AttemptResult] = []
    series_values: list[PlotSeries] = []
    for spec in specs:
        if spec.claim_id not in claims:
            raise ValueError(
                f"scaling attempt references unknown claim {spec.claim_id}"
            )
        if spec.comparison_rule_id not in rules:
            raise ValueError(
                f"scaling attempt references unknown rule {spec.comparison_rule_id}"
            )
        series = (
            None
            if evidence.missing_groups
            else _plot_series(spec=spec, analysis=analysis, recipe_count=len(recipes))
        )
        if series is not None:
            declared = spec.plot_series_ids
            if declared and declared != (series.id,):
                raise ValueError(
                    f"scaling plot series declaration differs for {spec.id}: "
                    f"{declared!r}"
                )
            series_values.append(series)
        results.append(
            _result(
                spec=spec,
                claim=claims[spec.claim_id],
                rule=rules[spec.comparison_rule_id],
                selection=selection,
                checkpoint_selection=checkpoint_selection,
                evidence=evidence,
                analysis=analysis,
                series=series,
                input_sha256=input_sha256,
            )
        )
    return (
        tuple(sorted(results, key=lambda result: result.attempt_id)),
        tuple(sorted(series_values, key=lambda series: series.id)),
    )


__all__ = ["run_scaling_law_attempts"]

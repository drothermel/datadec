from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from datadec.paper.models import (
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
    if "published-results" in Path(table.path).parts:
        raise ValueError("single-scale validation cannot read published-results")
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
        _attempt(contract, attempt_id)
        for attempt_id in (
            "dd-0011-default",
            "dd-0169-default",
            "dd-0148-default",
        )
    )
    if any(
        len(attempt.inputs) != 1 or attempt.inputs[0].table_id != _OLMES_TABLE_ID
        for attempt in implemented_attempts
    ):
        raise ValueError(
            "implemented single-scale attempts require only olmes_aggregate"
        )
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
    expected_sensitivity_ids = (
        "dd-0011-preceding-common-complete-1",
        "dd-0011-preceding-common-complete-2",
        "dd-0011-paper-step",
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
                target_value=_HEADLINE_TARGET,
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
        outcome=ValidationOutcome.REPRODUCED,
        diagnostics=(
            f"Persisted {len(points)} common-complete aggregate plot points.",
        ),
        limitations=(
            "The configured nonempty-plot predicate validates series availability; "
            "it does not adjudicate approximate log-linearity.",
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
        outcome=ValidationOutcome.REPRODUCED,
        diagnostics=(f"Persisted {len(points)} common-complete per-task plot points.",),
        limitations=(
            "The configured nonempty-plot predicate validates series availability; "
            "it does not adjudicate task-specific qualitative predicates.",
        ),
        plot_series_ids=(series.id,),
    )
    return result, series


def run_single_scale_attempts(
    *,
    repository_root: Path,
    data_root: Path,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
) -> tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]]:
    """Run the implemented aggregate OLMES format-2 validation attempts."""
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
    return tuple(sorted((*headline, plot_result), key=lambda item: item.attempt_id)), (
        series,
    )


def run_per_task_attempts(
    *,
    repository_root: Path,
    data_root: Path,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
) -> tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]]:
    """Run the implemented per-task OLMES format-2 validation attempt."""
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
    return (result,), (series,)


__all__ = ["run_per_task_attempts", "run_single_scale_attempts"]

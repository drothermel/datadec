from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed
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
from datadec.paper.proxy_metrics import (
    CheckpointTaskScore,
    LogicalTaskScore,
    PredictedTiePolicy,
    compare_logical_task_scores,
    latest_common_complete_noise_spread,
)
from datadec.paper.single_scale import (
    DEFAULT_TASK_GROUPING,
    CheckpointRows,
    MetricObservation,
    SingleScaleUniverse,
    observations_from_olmes_frame,
    select_common_complete_checkpoints,
    select_exact_common_complete_checkpoint,
)

_AGGREGATE_INPUT_ID = "olmes_aggregate"
_PRIMARY_METRIC = "primary_metric"
_PROXY_METRICS = (
    "acc_raw",
    "acc_per_token",
    "acc_per_char",
    "correct_prob",
    "correct_prob_per_token",
    "correct_prob_per_char",
    "margin",
    "margin_per_token",
    "margin_per_char",
    "norm_correct_prob",
    "norm_correct_prob_per_token",
    "norm_correct_prob_per_char",
    "total_prob",
    "total_prob_per_token",
    "total_prob_per_char",
)
_PAPER_METRICS = (_PRIMARY_METRIC, *_PROXY_METRICS)
_CONTINUOUS_PROXY_METRICS = tuple(
    metric for metric in _PROXY_METRICS if not metric.startswith("acc_")
)
_PER_CHARACTER_PLOT_METRICS = (
    _PRIMARY_METRIC,
    "correct_prob_per_char",
    "total_prob_per_char",
    "norm_correct_prob_per_char",
    "margin_per_char",
)
_TARGET_SEEDS = (Seed.DEFAULT.value, Seed.LARGE_AUX_2.value, Seed.LARGE_AUX_3.value)
_PREDICTION_SEEDS = (
    Seed.DEFAULT.value,
    Seed.SMALL_AUX_2.value,
    Seed.SMALL_AUX_3.value,
)
_ALL_SEEDS = tuple(seed.value for seed in Seed)
_RECIPES = tuple(recipe.value for recipe in DataRecipeName)
_MODEL_SIZES = tuple(size.value for size in ModelSizeName)
_PREDICTION_SIZES = tuple(size for size in _MODEL_SIZES if size != "1B")
_LOGICAL_TASKS = (*DEFAULT_TASK_GROUPING.non_mmlu_tasks, "mmlu")
_KEY_COLUMNS = ("params", "data", "seed", "step", "task")
_BASE_COLUMNS = (*_KEY_COLUMNS, "compute")
_THRESHOLD_ATTEMPTS = frozenset(
    {"dd-0014-default", "dd-0015-default", "dd-0016-default"}
)
_PROXY_PLOT_ATTEMPTS = frozenset({"dd-0208-default"})
_NOISE_PLOT_ATTEMPTS = frozenset(
    {
        "dd-0209-default",
        "dd-0210-default",
        "dd-0218-default",
        "dd-0219-default",
        "dd-0220-default",
    }
)
_PERCENT_COMPUTE_PATTERN = re.compile(
    r"decision_accuracy\s*>\s*(?P<accuracy>\d+(?:\.\d+)?)\s+at\s+"
    r"percent_target_compute\s*=\s*(?P<percent>\d+(?:\.\d+)?)"
)


@dataclass(frozen=True, slots=True)
class _Context:
    attempt: AttemptSpec
    claim: PaperClaim
    rule: ComparisonRule
    completeness_dimensions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Input:
    path: Path
    sha256: str
    frame: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _SeedTaskDecision:
    task: str
    metric: str
    seed: str
    accuracy: float
    denominator: int
    target_ties: int
    predicted_ties: int


@dataclass(frozen=True, slots=True)
class _TaskMetricDecision:
    task: str
    metric: str
    accuracy: float
    standard_deviation: float
    denominator: int
    target_ties: int
    predicted_ties: int


@dataclass(frozen=True, slots=True)
class _NoisePoint:
    task: str
    metric: str
    noise: float
    spread: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contexts(
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    analysis_id: AnalysisId,
) -> tuple[_Context, ...]:
    claims = {claim.id: claim for claim in registry.claims}
    rules = {rule.id: rule for rule in contract.comparison_rules}
    contexts: list[_Context] = []
    for attempt in contract.attempts:
        if attempt.analysis_id is not analysis_id:
            continue
        claim = claims.get(attempt.claim_id)
        if claim is None:
            raise ValueError(f"attempt {attempt.id} references an unknown claim")
        rule = rules.get(attempt.comparison_rule_id)
        if rule is None:
            raise ValueError(f"attempt {attempt.id} references an unknown rule")
        contexts.append(
            _Context(
                attempt=attempt,
                claim=claim,
                rule=rule,
                completeness_dimensions=contract.checkpoint_policy.completeness_dimensions,
            )
        )
    return tuple(sorted(contexts, key=lambda context: context.attempt.id))


def _resolve_values(
    values: tuple[str, ...],
    *,
    groups: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    resolved: list[str] = []
    for value in values:
        resolved.extend(groups.get(value, (value,)))
    return tuple(dict.fromkeys(resolved))


def _recipes(attempt: AttemptSpec) -> tuple[str, ...]:
    return _resolve_values(attempt.recipe_ids, groups={"paper-25-recipes": _RECIPES})


def _target_seeds(attempt: AttemptSpec) -> tuple[str, ...]:
    seeds = _resolve_values(
        attempt.seed_ids,
        groups={
            "target-three": _TARGET_SEEDS,
            "prediction-three": _PREDICTION_SEEDS,
            "all-five": _ALL_SEEDS,
        },
    )
    selected = tuple(seed for seed in seeds if seed in _TARGET_SEEDS)
    return selected or seeds


def _prediction_seeds(attempt: AttemptSpec) -> tuple[str, ...]:
    seeds = _resolve_values(
        attempt.seed_ids,
        groups={
            "target-three": _TARGET_SEEDS,
            "prediction-three": _PREDICTION_SEEDS,
            "all-five": _ALL_SEEDS,
        },
    )
    selected = tuple(seed for seed in seeds if seed in _PREDICTION_SEEDS)
    return selected or seeds


def _logical_tasks(attempt: AttemptSpec) -> tuple[str, ...]:
    return _resolve_values(
        attempt.task_ids, groups={"paper-olmes-tasks": _LOGICAL_TASKS}
    )


def _source_tasks(logical_tasks: Iterable[str]) -> tuple[str, ...]:
    values: list[str] = []
    for task in logical_tasks:
        if task == "mmlu":
            values.extend(DEFAULT_TASK_GROUPING.mmlu_subjects)
        else:
            values.append(task)
    return tuple(dict.fromkeys(values))


def _metrics(attempt: AttemptSpec) -> tuple[str, ...]:
    return _resolve_values(
        attempt.metric_ids,
        groups={
            "paper-16-metrics": _PAPER_METRICS,
            "per-token-proxies": tuple(
                metric for metric in _PROXY_METRICS if metric.endswith("_per_token")
            ),
            "per-character-proxies": tuple(
                metric for metric in _PROXY_METRICS if metric.endswith("_per_char")
            ),
        },
    )


def _model_sizes(attempt: AttemptSpec) -> tuple[str, ...]:
    return _resolve_values(
        attempt.model_sizes,
        groups={
            "prediction-sizes": _PREDICTION_SIZES,
            "paper-model-sizes": _MODEL_SIZES,
        },
    )


def _load_input(
    *,
    data_root: Path,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
    columns: tuple[str, ...],
    recipes: tuple[str, ...],
    seeds: tuple[str, ...],
    tasks: tuple[str, ...],
    model_sizes: tuple[str, ...],
) -> _Input:
    spec = next(
        (item for item in contract.inputs if item.id == _AGGREGATE_INPUT_ID), None
    )
    if spec is None:
        raise ValueError("validation contract has no olmes_aggregate input")
    undeclared = set(columns).difference(spec.columns)
    if undeclared:
        raise ValueError(
            "adapter requires undeclared aggregate columns: "
            f"{tuple(sorted(undeclared))!r}"
        )
    path = data_root / spec.path
    if not path.is_file():
        raise FileNotFoundError(f"declared aggregate input does not exist: {path}")
    actual_sha256 = _sha256_file(path)
    identity = input_identities.get(_AGGREGATE_INPUT_ID)
    if identity is None:
        raise ValueError("input identities omit olmes_aggregate")
    if identity.id != _AGGREGATE_INPUT_ID:
        raise ValueError("olmes_aggregate identity has the wrong logical input ID")
    if identity.sha256 != actual_sha256:
        raise ValueError(
            "olmes_aggregate identity differs from the actual Parquet input"
        )
    schema_columns = set(pq.ParquetFile(path).schema_arrow.names)
    missing_columns = tuple(sorted(set(columns).difference(schema_columns)))
    if missing_columns:
        raise ValueError(
            f"olmes_aggregate is missing configured columns: {missing_columns!r}"
        )
    filters: list[tuple[str, str, list[str]]] = []
    for column, values in (
        ("data", recipes),
        ("seed", seeds),
        ("task", tasks),
        ("params", model_sizes),
    ):
        if values:
            filters.append((column, "in", list(values)))
    frame = pd.read_parquet(path, columns=list(columns), filters=filters)
    if _sha256_file(path) != actual_sha256:
        raise RuntimeError("olmes_aggregate changed while it was being read")
    return _Input(path=path, sha256=actual_sha256, frame=frame)


def _selected_key_sha256(frame: pd.DataFrame) -> str:
    keys = sorted(
        tuple(value.item() if hasattr(value, "item") else value for value in row)
        for row in frame.loc[:, list(_KEY_COLUMNS)].itertuples(index=False, name=None)
    )
    records = [dict(zip(_KEY_COLUMNS, key, strict=True)) for key in keys]
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _row_selection(
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    model_size: str,
    recipes: tuple[str, ...],
    seeds: tuple[str, ...],
    source_tasks: tuple[str, ...],
    steps: tuple[int, ...],
) -> RowSelection:
    mask = (
        input_data.frame["params"].eq(model_size)
        & input_data.frame["data"].isin(recipes)
        & input_data.frame["seed"].isin(seeds)
        & input_data.frame["task"].isin(source_tasks)
        & input_data.frame["step"].isin(steps)
    )
    selected = input_data.frame.loc[mask]
    step_predicate = RowPredicate(
        column="step",
        operator=(PredicateOperator.EQ if len(steps) == 1 else PredicateOperator.IN),
        value=steps[0] if len(steps) == 1 else steps,
    )
    return RowSelection(
        logical_table_id=_AGGREGATE_INPUT_ID,
        columns=columns,
        predicates=(
            RowPredicate(
                column="params", operator=PredicateOperator.EQ, value=model_size
            ),
            RowPredicate(column="data", operator=PredicateOperator.IN, value=recipes),
            RowPredicate(column="seed", operator=PredicateOperator.IN, value=seeds),
            step_predicate,
            RowPredicate(
                column="task", operator=PredicateOperator.IN, value=source_tasks
            ),
        ),
        local_parquet_sha256=input_data.sha256,
        selected_row_count=len(selected),
        selected_key_sha256=_selected_key_sha256(selected),
    )


def _observations(
    input_data: _Input, metrics: tuple[str, ...]
) -> tuple[MetricObservation, ...]:
    return observations_from_olmes_frame(input_data.frame, metric_columns=metrics)


def _universe(
    *,
    model_size: str,
    recipes: tuple[str, ...],
    seeds: tuple[str, ...],
    source_tasks: tuple[str, ...],
    metrics: tuple[str, ...],
) -> SingleScaleUniverse:
    return SingleScaleUniverse(
        model_size=model_size,
        recipes=recipes,
        seeds=seeds,
        source_tasks=source_tasks,
        metrics=metrics,
    )


def _common_checkpoints(
    observations: tuple[MetricObservation, ...],
    universe: SingleScaleUniverse,
    *,
    preceding_count: int,
) -> tuple[CheckpointRows, ...]:
    if not any(value.model_size == universe.model_size for value in observations):
        return ()
    try:
        selected = select_common_complete_checkpoints(
            observations, universe, preceding_count=preceding_count
        )
    except ValueError as error:
        if "no common complete checkpoint" in str(error):
            return ()
        raise
    return (selected.default, *selected.preceding)


def _all_common_checkpoints(
    observations: tuple[MetricObservation, ...], universe: SingleScaleUniverse
) -> tuple[CheckpointRows, ...]:
    selected = _common_checkpoints(observations, universe, preceding_count=0)
    if not selected:
        return ()
    common = select_common_complete_checkpoints(
        observations, universe, preceding_count=0
    )
    return tuple(
        select_exact_common_complete_checkpoint(observations, universe, step=step)
        for step in common.complete_steps
    )


def _logical_values(
    checkpoint: CheckpointRows,
) -> dict[tuple[str, str, str, str], float]:
    source: dict[tuple[str, str, str, str], float] = {}
    for observation in checkpoint.observations:
        key = (
            observation.source_task,
            observation.recipe,
            observation.seed,
            observation.metric,
        )
        if key in source:
            raise ValueError(f"duplicate logical source score: {key!r}")
        source[key] = observation.score
    logical: dict[tuple[str, str, str, str], float] = {}
    for recipe in checkpoint.universe.recipes:
        for seed in checkpoint.universe.seeds:
            for metric in checkpoint.universe.metrics:
                for task in _logical_tasks_from_sources(
                    checkpoint.universe.source_tasks
                ):
                    if task == "mmlu":
                        values = tuple(
                            source[(subject, recipe, seed, metric)]
                            for subject in DEFAULT_TASK_GROUPING.mmlu_subjects
                        )
                        score = math.fsum(sorted(values)) / len(values)
                    else:
                        score = source[(task, recipe, seed, metric)]
                    logical[(task, recipe, seed, metric)] = score
    return logical


def _logical_tasks_from_sources(source_tasks: tuple[str, ...]) -> tuple[str, ...]:
    source_set = set(source_tasks)
    tasks = [
        task for task in DEFAULT_TASK_GROUPING.non_mmlu_tasks if task in source_set
    ]
    mmlu_subjects = set(DEFAULT_TASK_GROUPING.mmlu_subjects)
    present_mmlu = source_set.intersection(mmlu_subjects)
    if present_mmlu:
        if present_mmlu != mmlu_subjects:
            raise ValueError("logical MMLU score requires all configured subjects")
        tasks.append("mmlu")
    unexpected = source_set.difference(tasks).difference(mmlu_subjects)
    if unexpected:
        tasks.extend(sorted(unexpected))
    return tuple(tasks)


def _target_scores(checkpoint: CheckpointRows) -> tuple[LogicalTaskScore, ...]:
    values = _logical_values(checkpoint)
    tasks = _logical_tasks_from_sources(checkpoint.universe.source_tasks)
    return tuple(
        LogicalTaskScore(
            task=task,
            recipe=recipe,
            score=math.fsum(
                sorted(
                    values[(task, recipe, seed, _PRIMARY_METRIC)]
                    for seed in checkpoint.universe.seeds
                )
            )
            / len(checkpoint.universe.seeds),
        )
        for task in tasks
        for recipe in checkpoint.universe.recipes
    )


def _decision_rows(
    target_scores: tuple[LogicalTaskScore, ...], checkpoint: CheckpointRows
) -> tuple[_SeedTaskDecision, ...]:
    values = _logical_values(checkpoint)
    tasks = _logical_tasks_from_sources(checkpoint.universe.source_tasks)
    rows: list[_SeedTaskDecision] = []
    for metric in checkpoint.universe.metrics:
        for seed in checkpoint.universe.seeds:
            predicted = tuple(
                LogicalTaskScore(
                    task=task,
                    recipe=recipe,
                    score=values[(task, recipe, seed, metric)],
                )
                for task in tasks
                for recipe in checkpoint.universe.recipes
            )
            summary = compare_logical_task_scores(
                target_scores,
                predicted,
                logical_tasks=tasks,
                predicted_tie_policy=PredictedTiePolicy.COUNT_AS_INCORRECT,
            )
            rows.extend(
                _SeedTaskDecision(
                    task=task_summary.task,
                    metric=metric,
                    seed=seed,
                    accuracy=task_summary.result.accuracy,
                    denominator=task_summary.result.denominator,
                    target_ties=task_summary.result.target_ties,
                    predicted_ties=task_summary.result.predicted_ties,
                )
                for task_summary in summary.tasks
            )
    return tuple(rows)


def _sample_sd(values: Iterable[float]) -> float:
    ordered = tuple(sorted(values))
    if len(ordered) < 2:
        return 0.0
    mean = math.fsum(ordered) / len(ordered)
    return math.sqrt(
        math.fsum((value - mean) ** 2 for value in ordered) / (len(ordered) - 1)
    )


def _decision_aggregates(
    rows: tuple[_SeedTaskDecision, ...],
) -> tuple[_TaskMetricDecision, ...]:
    keys = sorted({(row.task, row.metric) for row in rows})
    return tuple(
        _TaskMetricDecision(
            task=task,
            metric=metric,
            accuracy=math.fsum(
                sorted(row.accuracy for row in rows if (row.task, row.metric) == key)
            )
            / sum((row.task, row.metric) == key for row in rows),
            standard_deviation=_sample_sd(
                row.accuracy for row in rows if (row.task, row.metric) == key
            ),
            denominator=sum(
                row.denominator for row in rows if (row.task, row.metric) == key
            ),
            target_ties=sum(
                row.target_ties for row in rows if (row.task, row.metric) == key
            ),
            predicted_ties=sum(
                row.predicted_ties for row in rows if (row.task, row.metric) == key
            ),
        )
        for key in keys
        for task, metric in (key,)
    )


def _checkpoint_selection(
    context: _Context,
    checkpoint: CheckpointRows,
    *,
    requested_meaning: str,
    rule: CheckpointRule,
) -> CheckpointSelection:
    return CheckpointSelection(
        requested_meaning=requested_meaning,
        rule=rule,
        actual_step=checkpoint.step,
        completeness_dimensions=context.completeness_dimensions,
        expected_group_count=checkpoint.expected_observation_count,
        selected_group_count=checkpoint.selected_observation_count,
    )


def _missing_result(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    recipes: tuple[str, ...],
    seeds: tuple[str, ...],
    source_tasks: tuple[str, ...],
    model_size: str,
    reason: str,
) -> AttemptResult:
    available_steps = tuple(
        sorted(
            int(value)
            for value in input_data.frame.loc[
                input_data.frame["params"].eq(model_size), "step"
            ].unique()
        )
    )
    steps = available_steps or (0,)
    selection = _row_selection(
        input_data,
        columns=columns,
        model_size=model_size,
        recipes=recipes,
        seeds=seeds,
        source_tasks=source_tasks,
        steps=steps,
    )
    return AttemptResult(
        attempt_id=context.attempt.id,
        claim_id=context.claim.id,
        role=AttemptRole.DEFAULT,
        comparison_rule_id=context.rule.id,
        comparison_rule_version=context.rule.version,
        transformation_ids=context.attempt.transformation_ids,
        row_selections=(selection,),
        target_value=context.claim.paper_target,
        computed_value=None,
        seeds=seeds,
        missing_groups=(reason,),
        outcome=ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED,
        diagnostics=(reason,),
        limitations=(
            "Choice-level files are reserved for formula/parity audits and were not loaded.",
        ),
    )


def _threshold_target(claim: PaperClaim) -> tuple[float, float]:
    if not isinstance(claim.paper_target, str):
        raise ValueError(f"claim {claim.id} has no encoded threshold target")
    match = _PERCENT_COMPUTE_PATTERN.search(claim.paper_target)
    if match is None:
        raise ValueError(f"claim {claim.id} has an unparseable threshold target")
    return float(match.group("accuracy")), float(match.group("percent"))


def _run_threshold_attempt(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    observations: tuple[MetricObservation, ...],
) -> AttemptResult:
    if context.rule.predicate is not ComparisonPredicate.DIRECTIONAL:
        raise ValueError(f"{context.attempt.id} requires a directional rule")
    recipes = _recipes(context.attempt)
    target_seeds = _target_seeds(context.attempt)
    prediction_seeds = _prediction_seeds(context.attempt)
    logical_tasks = _logical_tasks(context.attempt)
    source_tasks = _source_tasks(logical_tasks)
    target_universe = _universe(
        model_size="1B",
        recipes=recipes,
        seeds=target_seeds,
        source_tasks=source_tasks,
        metrics=(_PRIMARY_METRIC,),
    )
    targets = _common_checkpoints(observations, target_universe, preceding_count=0)
    if not targets:
        return _missing_result(
            context,
            input_data,
            columns=columns,
            recipes=recipes,
            seeds=target_seeds,
            source_tasks=source_tasks,
            model_size="1B",
            reason="No common-complete 1B target checkpoint exists.",
        )
    target = targets[0]
    accuracy_threshold, compute_budget_percent = _threshold_target(context.claim)
    candidates: list[CheckpointRows] = []
    for model_size in _model_sizes(context.attempt):
        if model_size == "1B":
            continue
        universe = _universe(
            model_size=model_size,
            recipes=recipes,
            seeds=prediction_seeds,
            source_tasks=source_tasks,
            metrics=_PAPER_METRICS,
        )
        candidates.extend(_all_common_checkpoints(observations, universe))
    eligible = tuple(
        checkpoint
        for checkpoint in candidates
        if checkpoint.actual_compute / target.actual_compute * 100
        <= compute_budget_percent
        and checkpoint.actual_compute > 0
    )
    if not eligible:
        return _missing_result(
            context,
            input_data,
            columns=columns,
            recipes=recipes,
            seeds=prediction_seeds,
            source_tasks=source_tasks,
            model_size=_model_sizes(context.attempt)[0],
            reason=(
                "No common-complete prediction checkpoint exists within the "
                f"{compute_budget_percent}% target-compute budget."
            ),
        )
    selected = max(
        eligible,
        key=lambda checkpoint: (
            checkpoint.actual_compute,
            _MODEL_SIZES.index(checkpoint.universe.model_size),
            checkpoint.step,
        ),
    )
    decisions = _decision_rows(_target_scores(target), selected)
    metric_scores = {
        metric: math.fsum(row.accuracy for row in decisions if row.metric == metric)
        / sum(row.metric == metric for row in decisions)
        for metric in _CONTINUOUS_PROXY_METRICS
    }
    best_metric, best_accuracy = max(
        metric_scores.items(), key=lambda item: (item[1], item[0])
    )
    best_rows = tuple(row for row in decisions if row.metric == best_metric)
    seed_means = tuple(
        math.fsum(row.accuracy for row in best_rows if row.seed == seed)
        / sum(row.seed == seed for row in best_rows)
        for seed in prediction_seeds
    )
    reproduced = best_accuracy > accuracy_threshold
    target_selection = _row_selection(
        input_data,
        columns=columns,
        model_size="1B",
        recipes=recipes,
        seeds=target_seeds,
        source_tasks=source_tasks,
        steps=(target.step,),
    )
    prediction_selection = _row_selection(
        input_data,
        columns=columns,
        model_size=selected.universe.model_size,
        recipes=recipes,
        seeds=prediction_seeds,
        source_tasks=source_tasks,
        steps=(selected.step,),
    )
    actual_percent = selected.actual_compute / target.actual_compute * 100
    return AttemptResult(
        attempt_id=context.attempt.id,
        claim_id=context.claim.id,
        role=AttemptRole.DEFAULT,
        comparison_rule_id=context.rule.id,
        comparison_rule_version=context.rule.version,
        transformation_ids=context.attempt.transformation_ids,
        row_selections=(target_selection, prediction_selection),
        checkpoint_selections=(
            _checkpoint_selection(
                context,
                target,
                requested_meaning="target final common complete",
                rule=CheckpointRule.LATEST_COMMON_COMPLETE,
            ),
            _checkpoint_selection(
                context,
                selected,
                requested_meaning=(
                    "latest common complete within "
                    f"{compute_budget_percent}% target compute"
                ),
                rule=CheckpointRule.LATEST_COMMON_COMPLETE,
            ),
        ),
        target_value=context.claim.paper_target,
        computed_value={
            "accuracy_threshold": accuracy_threshold,
            "compute_budget_percent": compute_budget_percent,
            "actual_percent_target_compute": actual_percent,
            "model_size": selected.universe.model_size,
            "step": selected.step,
            "best_continuous_proxy_metric": best_metric,
            "decision_accuracy": best_accuracy,
            "metric_decision_accuracies": dict(sorted(metric_scores.items())),
            "satisfied": reproduced,
        },
        seeds=prediction_seeds,
        denominator=sum(row.denominator for row in best_rows),
        target_ties=sum(row.target_ties for row in best_rows),
        predicted_ties=sum(row.predicted_ties for row in best_rows),
        standard_deviation=_sample_sd(seed_means),
        ddof=1,
        outcome=(
            ValidationOutcome.REPRODUCED
            if reproduced
            else ValidationOutcome.NOT_REPRODUCED
        ),
        diagnostics=(
            f"Selected {selected.universe.model_size} step {selected.step} at "
            f"{actual_percent:.12g}% of target compute.",
        ),
        limitations=(
            "The best eligible continuous proxy is selected from the metric universe declared before computation.",
            "Pre-aggregated OLMES metrics are used; choice-level formula parity is not audited here.",
        ),
    )


def _plot_series_id(context: _Context) -> str:
    expected = f"{context.claim.id.lower()}-paper-analog"
    if context.attempt.plot_series_ids and context.attempt.plot_series_ids != (
        expected,
    ):
        raise ValueError(
            f"{context.attempt.id} must declare exactly the canonical series ID {expected}"
        )
    return expected


def _run_proxy_plot_attempt(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    observations: tuple[MetricObservation, ...],
) -> tuple[AttemptResult, PlotSeries | None]:
    if context.rule.predicate is not ComparisonPredicate.NONEMPTY_PLOT:
        raise ValueError(f"{context.attempt.id} requires a nonempty-plot rule")
    recipes = _recipes(context.attempt)
    target_seeds = _target_seeds(context.attempt)
    prediction_seeds = _prediction_seeds(context.attempt)
    logical_tasks = _logical_tasks(context.attempt)
    source_tasks = _source_tasks(logical_tasks)
    target_universe = _universe(
        model_size="1B",
        recipes=recipes,
        seeds=target_seeds,
        source_tasks=source_tasks,
        metrics=(_PRIMARY_METRIC,),
    )
    targets = _common_checkpoints(observations, target_universe, preceding_count=0)
    if not targets:
        return (
            _missing_result(
                context,
                input_data,
                columns=columns,
                recipes=recipes,
                seeds=target_seeds,
                source_tasks=source_tasks,
                model_size="1B",
                reason="Proxy plot has no common-complete target checkpoint.",
            ),
            None,
        )
    target = targets[0]
    target_scores = _target_scores(target)
    checkpoints: list[CheckpointRows] = []
    for model_size in _model_sizes(context.attempt):
        if model_size == "1B":
            continue
        universe = _universe(
            model_size=model_size,
            recipes=recipes,
            seeds=prediction_seeds,
            source_tasks=source_tasks,
            metrics=_PER_CHARACTER_PLOT_METRICS,
        )
        checkpoints.extend(_all_common_checkpoints(observations, universe))
    points: list[PlotPoint] = []
    selections: list[RowSelection] = [
        _row_selection(
            input_data,
            columns=columns,
            model_size="1B",
            recipes=recipes,
            seeds=target_seeds,
            source_tasks=source_tasks,
            steps=(target.step,),
        )
    ]
    denominator = 0
    target_ties = 0
    predicted_ties = 0
    for model_size in _PREDICTION_SIZES:
        model_checkpoints = tuple(
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.universe.model_size == model_size
            and checkpoint.actual_compute > 0
        )
        if not model_checkpoints:
            continue
        selections.append(
            _row_selection(
                input_data,
                columns=columns,
                model_size=model_size,
                recipes=recipes,
                seeds=prediction_seeds,
                source_tasks=source_tasks,
                steps=tuple(checkpoint.step for checkpoint in model_checkpoints),
            )
        )
        for checkpoint in model_checkpoints:
            rows = _decision_rows(target_scores, checkpoint)
            aggregates = _decision_aggregates(rows)
            denominator += sum(row.denominator for row in rows)
            target_ties += sum(row.target_ties for row in rows)
            predicted_ties += sum(row.predicted_ties for row in rows)
            percent_compute = checkpoint.actual_compute / target.actual_compute * 100
            for aggregate in aggregates:
                points.append(
                    PlotPoint(
                        dimensions=(
                            DimensionValue(name="task", value=aggregate.task),
                            DimensionValue(name="metric", value=aggregate.metric),
                            DimensionValue(name="model_size", value=model_size),
                            DimensionValue(name="step", value=checkpoint.step),
                        ),
                        measures=(
                            MeasureValue(
                                name="percent_target_compute", value=percent_compute
                            ),
                            MeasureValue(
                                name="decision_accuracy", value=aggregate.accuracy
                            ),
                        ),
                    )
                )
    if not points:
        return (
            _missing_result(
                context,
                input_data,
                columns=columns,
                recipes=recipes,
                seeds=prediction_seeds,
                source_tasks=source_tasks,
                model_size=_model_sizes(context.attempt)[0],
                reason="Proxy plot has no common-complete prediction points.",
            ),
            None,
        )
    series_id = _plot_series_id(context)
    series = PlotSeries(
        id=series_id,
        figure="all_metrics_compute_vs_accuracy_per_task",
        panel="all_tasks",
        semantic_kind="compute_vs_decision_accuracy_per_task",
        x_axis=AxisSpec(
            measure="percent_target_compute", scale=AxisScale.LOG, unit="percent"
        ),
        y_axis=AxisSpec(
            measure="decision_accuracy", scale=AxisScale.LINEAR, unit="fraction"
        ),
        dimensions=("task", "metric", "model_size", "step"),
        measures=("percent_target_compute", "decision_accuracy"),
        attempt_id=context.attempt.id,
        counts=(
            NamedCount(name="recipes", value=len(recipes)),
            NamedCount(name="prediction_seeds", value=len(prediction_seeds)),
            NamedCount(name="logical_tasks", value=len(logical_tasks)),
            NamedCount(name="points", value=len(points)),
        ),
        points=tuple(points),
    )
    result = AttemptResult(
        attempt_id=context.attempt.id,
        claim_id=context.claim.id,
        role=AttemptRole.DEFAULT,
        comparison_rule_id=context.rule.id,
        comparison_rule_version=context.rule.version,
        transformation_ids=context.attempt.transformation_ids,
        row_selections=tuple(selections),
        checkpoint_selections=(
            _checkpoint_selection(
                context,
                target,
                requested_meaning="target final common complete",
                rule=CheckpointRule.LATEST_COMMON_COMPLETE,
            ),
        ),
        target_value=context.claim.paper_target,
        computed_value={"point_count": len(points)},
        seeds=prediction_seeds,
        denominator=denominator,
        target_ties=target_ties,
        predicted_ties=predicted_ties,
        outcome=ValidationOutcome.REPRODUCED,
        diagnostics=(f"Persisted {len(points)} finite paper-analog points.",),
        limitations=(
            "The configured nonempty-plot predicate validates the curve surface, not the paper's benefiting-task count.",
            "Pre-aggregated OLMES metrics are used; choice-level formula parity is not audited here.",
        ),
        plot_series_ids=(series_id,),
    )
    return result, series


def _noise_points(
    checkpoint: CheckpointRows,
) -> tuple[_NoisePoint, ...]:
    logical = _logical_values(checkpoint)
    tasks = _logical_tasks_from_sources(checkpoint.universe.source_tasks)
    scores = tuple(
        CheckpointTaskScore(
            model_size=checkpoint.universe.model_size,
            step=checkpoint.step,
            task=task,
            metric=metric,
            recipe=recipe,
            seed=seed,
            score=logical[(task, recipe, seed, metric)],
        )
        for task in tasks
        for metric in checkpoint.universe.metrics
        for recipe in checkpoint.universe.recipes
        for seed in checkpoint.universe.seeds
    )
    summary = latest_common_complete_noise_spread(
        scores,
        model_size=checkpoint.universe.model_size,
        expected_recipes=checkpoint.universe.recipes,
        expected_seeds=checkpoint.universe.seeds,
        expected_tasks=tasks,
        expected_metrics=checkpoint.universe.metrics,
    )
    points: list[_NoisePoint] = []
    for result in summary.results:
        points.append(
            _NoisePoint(
                task=result.task,
                metric=result.metric,
                noise=result.noise,
                spread=result.spread,
            )
        )
    return tuple(points)


def _run_noise_plot_attempt(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    observations: tuple[MetricObservation, ...],
) -> tuple[tuple[AttemptResult, ...], PlotSeries | None]:
    if context.rule.predicate is not ComparisonPredicate.NONEMPTY_PLOT:
        raise ValueError(f"{context.attempt.id} requires a nonempty-plot rule")
    recipes = _recipes(context.attempt)
    prediction_seeds = _prediction_seeds(context.attempt)
    target_seeds = _TARGET_SEEDS
    logical_tasks = _logical_tasks(context.attempt)
    source_tasks = _source_tasks(logical_tasks)
    metrics = _metrics(context.attempt)
    prediction_size = next(
        (size for size in _model_sizes(context.attempt) if size != "1B"), "150M"
    )
    target_universe = _universe(
        model_size="1B",
        recipes=recipes,
        seeds=target_seeds,
        source_tasks=source_tasks,
        metrics=(_PRIMARY_METRIC,),
    )
    prediction_universe = _universe(
        model_size=prediction_size,
        recipes=recipes,
        seeds=prediction_seeds,
        source_tasks=source_tasks,
        metrics=metrics,
    )
    targets = _common_checkpoints(observations, target_universe, preceding_count=0)
    preceding_ids = tuple(
        sensitivity_id
        for sensitivity_id in context.attempt.sensitivity_ids
        if "preceding-common-complete" in sensitivity_id
    )
    predictions = _common_checkpoints(
        observations,
        prediction_universe,
        preceding_count=len(preceding_ids),
    )
    if not targets or not predictions:
        missing_target = not targets
        return (
            (
                _missing_result(
                    context,
                    input_data,
                    columns=columns,
                    recipes=recipes,
                    seeds=target_seeds if missing_target else prediction_seeds,
                    source_tasks=source_tasks,
                    model_size="1B" if missing_target else prediction_size,
                    reason=(
                        "Noise/spread plot lacks a common-complete target checkpoint."
                        if missing_target
                        else "Noise/spread plot lacks a common-complete prediction checkpoint."
                    ),
                ),
            ),
            None,
        )
    target = targets[0]
    prediction = predictions[0]
    decisions = _decision_aggregates(_decision_rows(_target_scores(target), prediction))
    decision_lookup = {
        (decision.task, decision.metric): decision for decision in decisions
    }
    noise_points = _noise_points(prediction)
    points = tuple(
        PlotPoint(
            dimensions=(
                DimensionValue(name="task", value=point.task),
                DimensionValue(name="metric", value=point.metric),
            ),
            measures=(
                MeasureValue(name="noise", value=point.noise),
                MeasureValue(name="spread", value=point.spread),
                MeasureValue(
                    name="decision_accuracy",
                    value=decision_lookup[(point.task, point.metric)].accuracy,
                ),
            ),
        )
        for point in noise_points
    )
    if not points:
        return (
            (
                _missing_result(
                    context,
                    input_data,
                    columns=columns,
                    recipes=recipes,
                    seeds=prediction_seeds,
                    source_tasks=source_tasks,
                    model_size=prediction_size,
                    reason="Noise/spread plot has no finite points.",
                ),
            ),
            None,
        )
    series_id = _plot_series_id(context)
    series = PlotSeries(
        id=series_id,
        figure="noise_to_spread_150M",
        panel="all_tasks_metrics",
        semantic_kind="noise_spread_decision_accuracy",
        x_axis=AxisSpec(measure="noise", scale=AxisScale.LINEAR, unit="score"),
        y_axis=AxisSpec(measure="spread", scale=AxisScale.LINEAR, unit="score"),
        dimensions=("task", "metric"),
        measures=("noise", "spread", "decision_accuracy"),
        attempt_id=context.attempt.id,
        actual_checkpoint=prediction.step,
        counts=(
            NamedCount(name="recipes", value=len(recipes)),
            NamedCount(name="prediction_seeds", value=len(prediction_seeds)),
            NamedCount(name="points", value=len(points)),
        ),
        points=points,
    )
    target_selection = _row_selection(
        input_data,
        columns=columns,
        model_size="1B",
        recipes=recipes,
        seeds=target_seeds,
        source_tasks=source_tasks,
        steps=(target.step,),
    )
    prediction_selection = _row_selection(
        input_data,
        columns=columns,
        model_size=prediction_size,
        recipes=recipes,
        seeds=prediction_seeds,
        source_tasks=source_tasks,
        steps=(prediction.step,),
    )
    result = AttemptResult(
        attempt_id=context.attempt.id,
        claim_id=context.claim.id,
        role=AttemptRole.DEFAULT,
        comparison_rule_id=context.rule.id,
        comparison_rule_version=context.rule.version,
        transformation_ids=context.attempt.transformation_ids,
        row_selections=(target_selection, prediction_selection),
        checkpoint_selections=(
            _checkpoint_selection(
                context,
                target,
                requested_meaning="target final common complete",
                rule=CheckpointRule.LATEST_COMMON_COMPLETE,
            ),
            _checkpoint_selection(
                context,
                prediction,
                requested_meaning="150M final common complete",
                rule=CheckpointRule.LATEST_COMMON_COMPLETE,
            ),
        ),
        target_value=context.claim.paper_target,
        computed_value={"point_count": len(points)},
        seeds=prediction_seeds,
        denominator=sum(decision.denominator for decision in decisions),
        target_ties=sum(decision.target_ties for decision in decisions),
        predicted_ties=sum(decision.predicted_ties for decision in decisions),
        outcome=ValidationOutcome.REPRODUCED,
        diagnostics=(f"Persisted {len(points)} finite paper-analog points.",),
        limitations=(
            "The configured nonempty-plot predicate validates the plotted evidence surface, not qualitative words such as low, high, often, or align.",
            "Noise and spread use sample standard deviations (DDOF 1).",
            "The configured paper-step sensitivity is not emitted because the attempt does not declare an exact numeric paper step.",
        ),
        plot_series_ids=(series_id,),
    )
    sensitivity_results: list[AttemptResult] = []
    for sensitivity_id, sensitivity_checkpoint in zip(
        preceding_ids, predictions[1:], strict=False
    ):
        sensitivity_decisions = _decision_aggregates(
            _decision_rows(_target_scores(target), sensitivity_checkpoint)
        )
        sensitivity_noise = _noise_points(sensitivity_checkpoint)
        sensitivity_lookup = {
            (decision.task, decision.metric): decision
            for decision in sensitivity_decisions
        }
        sensitivity_selection = _row_selection(
            input_data,
            columns=columns,
            model_size=prediction_size,
            recipes=recipes,
            seeds=prediction_seeds,
            source_tasks=source_tasks,
            steps=(sensitivity_checkpoint.step,),
        )
        sensitivity_results.append(
            AttemptResult(
                attempt_id=sensitivity_id,
                claim_id=context.claim.id,
                role=AttemptRole.SENSITIVITY,
                parent_attempt_id=context.attempt.id,
                comparison_rule_id=context.rule.id,
                comparison_rule_version=context.rule.version,
                transformation_ids=context.attempt.transformation_ids,
                row_selections=(target_selection, sensitivity_selection),
                checkpoint_selections=(
                    _checkpoint_selection(
                        context,
                        target,
                        requested_meaning="target final common complete",
                        rule=CheckpointRule.LATEST_COMMON_COMPLETE,
                    ),
                    _checkpoint_selection(
                        context,
                        sensitivity_checkpoint,
                        requested_meaning="preceding 150M common complete",
                        rule=CheckpointRule.PRECEDING_COMMON_COMPLETE,
                    ),
                ),
                target_value=context.claim.paper_target,
                computed_value={
                    "point_count": len(sensitivity_noise),
                    "points": [
                        {
                            "task": point.task,
                            "metric": point.metric,
                            "noise": point.noise,
                            "spread": point.spread,
                            "decision_accuracy": sensitivity_lookup[
                                (point.task, point.metric)
                            ].accuracy,
                        }
                        for point in sensitivity_noise
                    ],
                },
                seeds=prediction_seeds,
                denominator=sum(
                    decision.denominator for decision in sensitivity_decisions
                ),
                target_ties=sum(
                    decision.target_ties for decision in sensitivity_decisions
                ),
                predicted_ties=sum(
                    decision.predicted_ties for decision in sensitivity_decisions
                ),
                outcome=ValidationOutcome.REPRODUCED,
                diagnostics=(
                    f"Computed {len(sensitivity_noise)} points at fixed preceding step {sensitivity_checkpoint.step}.",
                ),
                limitations=result.limitations,
            )
        )
    return (result, *sensitivity_results), series


def run_proxy_metrics_attempts(
    *,
    repository_root: Path,
    data_root: Path,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
) -> tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]]:
    """Run the objectively specified proxy-metric format-2 attempts."""
    del repository_root
    contexts = tuple(
        context
        for context in _contexts(registry, contract, AnalysisId.PROXY_METRICS)
        if context.attempt.id in _THRESHOLD_ATTEMPTS | _PROXY_PLOT_ATTEMPTS
    )
    if not contexts:
        return (), ()
    recipes = tuple(
        dict.fromkeys(
            recipe for context in contexts for recipe in _recipes(context.attempt)
        )
    )
    seeds = tuple(
        dict.fromkeys(
            seed
            for context in contexts
            for seed in (
                *_target_seeds(context.attempt),
                *_prediction_seeds(context.attempt),
            )
        )
    )
    logical_tasks = tuple(
        dict.fromkeys(
            task for context in contexts for task in _logical_tasks(context.attempt)
        )
    )
    source_tasks = _source_tasks(logical_tasks)
    model_sizes = tuple(
        dict.fromkeys(
            size
            for context in contexts
            for size in (*_model_sizes(context.attempt), "1B")
        )
    )
    metrics = tuple(
        dict.fromkeys(
            metric
            for context in contexts
            for metric in (
                _PAPER_METRICS
                if context.attempt.id in _THRESHOLD_ATTEMPTS
                else _PER_CHARACTER_PLOT_METRICS
            )
        )
    )
    columns = (*_BASE_COLUMNS, *metrics)
    input_data = _load_input(
        data_root=data_root,
        contract=contract,
        input_identities=input_identities,
        columns=columns,
        recipes=recipes,
        seeds=seeds,
        tasks=source_tasks,
        model_sizes=model_sizes,
    )
    observations = _observations(input_data, metrics)
    results: list[AttemptResult] = []
    series: list[PlotSeries] = []
    for context in contexts:
        if context.attempt.id in _THRESHOLD_ATTEMPTS:
            attempt_columns = (*_BASE_COLUMNS, *_PAPER_METRICS)
            results.append(
                _run_threshold_attempt(
                    context,
                    input_data,
                    columns=attempt_columns,
                    observations=observations,
                )
            )
        else:
            attempt_columns = (*_BASE_COLUMNS, *_PER_CHARACTER_PLOT_METRICS)
            result, plot = _run_proxy_plot_attempt(
                context,
                input_data,
                columns=attempt_columns,
                observations=observations,
            )
            results.append(result)
            if plot is not None:
                series.append(plot)
    return tuple(sorted(results, key=lambda result: result.attempt_id)), tuple(
        sorted(series, key=lambda plot: plot.id)
    )


def run_noise_spread_attempts(
    *,
    repository_root: Path,
    data_root: Path,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
) -> tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]]:
    """Run the objectively specified noise/spread format-2 attempts."""
    del repository_root
    contexts = tuple(
        context
        for context in _contexts(registry, contract, AnalysisId.NOISE_SPREAD)
        if context.attempt.id in _NOISE_PLOT_ATTEMPTS
    )
    if not contexts:
        return (), ()
    recipes = tuple(
        dict.fromkeys(
            recipe for context in contexts for recipe in _recipes(context.attempt)
        )
    )
    seeds = tuple(
        dict.fromkeys(
            seed
            for context in contexts
            for seed in (
                *_target_seeds(context.attempt),
                *_prediction_seeds(context.attempt),
                *_TARGET_SEEDS,
            )
        )
    )
    logical_tasks = tuple(
        dict.fromkeys(
            task for context in contexts for task in _logical_tasks(context.attempt)
        )
    )
    source_tasks = _source_tasks(logical_tasks)
    model_sizes = tuple(
        dict.fromkeys(
            size
            for context in contexts
            for size in (*_model_sizes(context.attempt), "1B")
        )
    )
    metrics = tuple(
        dict.fromkeys(
            metric for context in contexts for metric in _metrics(context.attempt)
        )
    )
    if _PRIMARY_METRIC not in metrics:
        metrics = (_PRIMARY_METRIC, *metrics)
    columns = (*_BASE_COLUMNS, *metrics)
    input_data = _load_input(
        data_root=data_root,
        contract=contract,
        input_identities=input_identities,
        columns=columns,
        recipes=recipes,
        seeds=seeds,
        tasks=source_tasks,
        model_sizes=model_sizes,
    )
    observations = _observations(input_data, metrics)
    results: list[AttemptResult] = []
    series: list[PlotSeries] = []
    for context in contexts:
        attempt_columns = tuple(
            dict.fromkeys((*_BASE_COLUMNS, _PRIMARY_METRIC, *_metrics(context.attempt)))
        )
        attempt_results, plot = _run_noise_plot_attempt(
            context,
            input_data,
            columns=attempt_columns,
            observations=observations,
        )
        results.extend(attempt_results)
        if plot is not None:
            series.append(plot)
    return tuple(sorted(results, key=lambda result: result.attempt_id)), tuple(
        sorted(series, key=lambda plot: plot.id)
    )


__all__ = ["run_noise_spread_attempts", "run_proxy_metrics_attempts"]

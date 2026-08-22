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
from datadec.paper.proxy_metrics import (
    CheckpointTaskScore,
    CrossoverTiePolicy,
    LogicalTaskScore,
    ScaleRecipeScore,
    latest_common_complete_noise_spread,
    summarize_crossovers,
)
from datadec.paper.single_scale import (
    DEFAULT_TASK_GROUPING,
    CheckpointRows,
    MetricObservation,
    SingleScaleUniverse,
    observations_from_olmes_frame,
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


@dataclass(frozen=True, slots=True)
class _CurvePoint:
    task: str
    metric: str
    model_size: str
    step: int
    compute: float
    percent_target_compute: float
    accuracy: float
    denominator: int
    target_ties: int
    predicted_ties: int


type _CurveSurface = tuple[
    CheckpointRows | None,
    tuple[CheckpointRows, ...],
    tuple[_CurvePoint, ...],
]


def _parameter(rule: ComparisonRule, name: ComparisonParameterName) -> float:
    return rule.parameter(name).default


def _comparison_sensitivity_contexts(
    context: _Context,
) -> tuple[tuple[str, ComparisonParameterName, float, _Context], ...]:
    sensitivities: list[tuple[str, ComparisonParameterName, float, _Context]] = []
    for parameter_index, parameter in enumerate(context.rule.parameters):
        for grid_index, value in enumerate(parameter.sensitivity_grid, start=1):
            if value == parameter.default:
                continue
            sensitivity_id = (
                f"{context.attempt.claim_id.lower()}-comparison-"
                f"{parameter.name.value.replace('_', '-')}-grid-{grid_index}"
            )
            if sensitivity_id not in context.attempt.sensitivity_ids:
                raise ValueError(
                    f"{context.attempt.id} does not declare comparison sensitivity "
                    f"{sensitivity_id}"
                )
            parameters = list(context.rule.parameters)
            parameters[parameter_index] = parameter.model_copy(
                update={"default": value}
            )
            sensitivities.append(
                (
                    sensitivity_id,
                    parameter.name,
                    value,
                    _Context(
                        attempt=context.attempt,
                        claim=context.claim,
                        rule=context.rule.model_copy(
                            update={"parameters": tuple(parameters)}
                        ),
                        completeness_dimensions=context.completeness_dimensions,
                    ),
                )
            )
    return tuple(sensitivities)


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


def _declared_seeds(attempt: AttemptSpec) -> tuple[str, ...]:
    return _resolve_values(
        attempt.seed_ids,
        groups={
            "target-three": _TARGET_SEEDS,
            "prediction-three": _PREDICTION_SEEDS,
            "all-five": _ALL_SEEDS,
        },
    )


def _seeds_for_model(attempt: AttemptSpec, model_size: str) -> tuple[str, ...]:
    declared = _declared_seeds(attempt)
    if set(declared) == set(_ALL_SEEDS):
        return _TARGET_SEEDS if model_size == "1B" else _PREDICTION_SEEDS
    return declared


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
    checkpoints = _all_common_checkpoints(observations, universe)
    return tuple(reversed(checkpoints[-(preceding_count + 1) :]))


def _all_common_checkpoints(
    observations: tuple[MetricObservation, ...], universe: SingleScaleUniverse
) -> tuple[CheckpointRows, ...]:
    recipe_set = set(universe.recipes)
    seed_set = set(universe.seeds)
    task_set = set(universe.source_tasks)
    metric_set = set(universe.metrics)
    expected = {
        (recipe, seed, task, metric)
        for recipe in universe.recipes
        for seed in universe.seeds
        for task in universe.source_tasks
        for metric in universe.metrics
    }
    by_step: dict[int, list[MetricObservation]] = {}
    for observation in observations:
        if (
            observation.model_size == universe.model_size
            and observation.recipe in recipe_set
            and observation.seed in seed_set
            and observation.source_task in task_set
            and observation.metric in metric_set
        ):
            by_step.setdefault(observation.step, []).append(observation)
    checkpoints: list[CheckpointRows] = []
    for step, selected in sorted(by_step.items()):
        actual = {
            (value.recipe, value.seed, value.source_task, value.metric)
            for value in selected
        }
        if actual != expected or len(selected) != len(actual):
            continue
        computes = {value.compute for value in selected}
        if len(computes) != 1:
            raise ValueError(
                "checkpoint compute differs across the declared grid: "
                f"model_size={universe.model_size!r}, step={step}"
            )
        raw_keys = {(value.recipe, value.seed, value.source_task) for value in selected}
        if len(raw_keys) != universe.expected_raw_row_count:
            raise ValueError("checkpoint raw-row grid is incomplete")
        checkpoints.append(
            CheckpointRows(
                universe=universe,
                step=step,
                observations=tuple(
                    sorted(
                        selected,
                        key=lambda value: (
                            value.recipe,
                            value.seed,
                            value.source_task,
                            value.metric,
                        ),
                    )
                ),
                raw_row_count=len(raw_keys),
                selected_observation_count=len(selected),
                expected_observation_count=len(expected),
                actual_compute=next(iter(computes)),
            )
        )
    return tuple(checkpoints)


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
    target_lookup = {(score.task, score.recipe): score.score for score in target_scores}
    rows: list[_SeedTaskDecision] = []
    for metric in checkpoint.universe.metrics:
        for seed in checkpoint.universe.seeds:
            for task in tasks:
                correct = 0
                denominator = 0
                target_ties = 0
                predicted_ties = 0
                recipes = checkpoint.universe.recipes
                for index, recipe_a in enumerate(recipes):
                    for recipe_b in recipes[index + 1 :]:
                        target_a = target_lookup[(task, recipe_a)]
                        target_b = target_lookup[(task, recipe_b)]
                        predicted_a = values[(task, recipe_a, seed, metric)]
                        predicted_b = values[(task, recipe_b, seed, metric)]
                        target_sign = (target_a > target_b) - (target_a < target_b)
                        predicted_sign = (predicted_a > predicted_b) - (
                            predicted_a < predicted_b
                        )
                        if target_sign == 0:
                            target_ties += 1
                            continue
                        denominator += 1
                        if predicted_sign == 0:
                            predicted_ties += 1
                        elif predicted_sign == target_sign:
                            correct += 1
                if denominator == 0:
                    raise ValueError("target ties exclude every recipe pair")
                rows.append(
                    _SeedTaskDecision(
                        task=task,
                        metric=metric,
                        seed=seed,
                        accuracy=correct / denominator,
                        denominator=denominator,
                        target_ties=target_ties,
                        predicted_ties=predicted_ties,
                    )
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


def _mean(values: Iterable[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("mean requires at least one value")
    return math.fsum(ordered) / len(ordered)


def _linear_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    x_values = tuple(xs)
    y_values = tuple(ys)
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
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


def _ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return tuple(ranks)


def _spearman(xs: Iterable[float], ys: Iterable[float]) -> float:
    x_values = tuple(xs)
    y_values = tuple(ys)
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    x_ranks = _ranks(x_values)
    y_ranks = _ranks(y_values)
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
        outcome=ValidationOutcome.DESCRIPTIVE_ONLY,
        diagnostics=(f"Persisted {len(points)} finite paper-analog points.",),
        limitations=(
            "The derived plot series is available for visual comparison, but no frozen semantic predicate adjudicates the paper's benefiting-task count.",
            "Pre-aggregated OLMES metrics are used; choice-level formula parity is not audited here.",
        ),
        plot_series_ids=(series_id,),
    )
    return result, series


def _curve_surface(
    context: _Context,
    observations: tuple[MetricObservation, ...],
) -> _CurveSurface:
    recipes = _recipes(context.attempt)
    target_seeds = _target_seeds(context.attempt)
    prediction_seeds = _prediction_seeds(context.attempt)
    source_tasks = _source_tasks(_logical_tasks(context.attempt))
    metrics = tuple(dict.fromkeys((_PRIMARY_METRIC, *_metrics(context.attempt))))
    target_universe = _universe(
        model_size="1B",
        recipes=recipes,
        seeds=target_seeds,
        source_tasks=source_tasks,
        metrics=(_PRIMARY_METRIC,),
    )
    targets = _common_checkpoints(observations, target_universe, preceding_count=0)
    if not targets:
        return None, (), ()
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
            metrics=metrics,
        )
        checkpoints.extend(_all_common_checkpoints(observations, universe))
    points: list[_CurvePoint] = []
    for checkpoint in checkpoints:
        for row in _decision_aggregates(_decision_rows(target_scores, checkpoint)):
            points.append(
                _CurvePoint(
                    task=row.task,
                    metric=row.metric,
                    model_size=checkpoint.universe.model_size,
                    step=checkpoint.step,
                    compute=checkpoint.actual_compute,
                    percent_target_compute=(
                        checkpoint.actual_compute / target.actual_compute * 100
                    ),
                    accuracy=row.accuracy,
                    denominator=row.denominator,
                    target_ties=row.target_ties,
                    predicted_ties=row.predicted_ties,
                )
            )
    return target, tuple(checkpoints), tuple(points)


def _curve_surface_key(context: _Context) -> tuple[tuple[str, ...], ...]:
    return (
        _recipes(context.attempt),
        _target_seeds(context.attempt),
        _prediction_seeds(context.attempt),
        _source_tasks(_logical_tasks(context.attempt)),
        tuple(dict.fromkeys((_PRIMARY_METRIC, *_metrics(context.attempt)))),
        _model_sizes(context.attempt),
    )


def _qualitative_result(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    target: CheckpointRows | None,
    checkpoints: tuple[CheckpointRows, ...],
    computed_value: dict[str, object],
    outcome: ValidationOutcome,
    diagnostics: tuple[str, ...],
) -> AttemptResult:
    recipes = _recipes(context.attempt)
    source_tasks = _source_tasks(_logical_tasks(context.attempt))
    target_seeds = _target_seeds(context.attempt)
    prediction_seeds = _prediction_seeds(context.attempt)
    selections: list[RowSelection] = []
    checkpoint_selections: list[CheckpointSelection] = []
    if target is not None:
        selections.append(
            _row_selection(
                input_data,
                columns=columns,
                model_size="1B",
                recipes=recipes,
                seeds=target_seeds,
                source_tasks=source_tasks,
                steps=(target.step,),
            )
        )
        checkpoint_selections.append(
            _checkpoint_selection(
                context,
                target,
                requested_meaning="target final common complete",
                rule=CheckpointRule.LATEST_COMMON_COMPLETE,
            )
        )
    for model_size in _MODEL_SIZES:
        selected = tuple(
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.universe.model_size == model_size
        )
        if not selected:
            continue
        selections.append(
            _row_selection(
                input_data,
                columns=columns,
                model_size=model_size,
                recipes=recipes,
                seeds=selected[0].universe.seeds,
                source_tasks=source_tasks,
                steps=tuple(checkpoint.step for checkpoint in selected),
            )
        )
        checkpoint_selections.extend(
            _checkpoint_selection(
                context,
                checkpoint,
                requested_meaning="configured common-complete curve point",
                rule=CheckpointRule.EXACT,
            )
            for checkpoint in selected
        )
    if not selections:
        raise ValueError(f"{context.attempt.id} produced no row selections")
    return AttemptResult(
        attempt_id=context.attempt.id,
        claim_id=context.claim.id,
        role=AttemptRole.DEFAULT,
        comparison_rule_id=context.rule.id,
        comparison_rule_version=context.rule.version,
        transformation_ids=context.attempt.transformation_ids,
        row_selections=tuple(selections),
        checkpoint_selections=tuple(checkpoint_selections),
        target_value=context.claim.paper_target,
        computed_value=computed_value,
        seeds=tuple(dict.fromkeys((*target_seeds, *prediction_seeds))),
        denominator=sum(
            int(value)
            for key, value in computed_value.items()
            if key == "denominator" and isinstance(value, int)
        )
        or None,
        outcome=outcome,
        diagnostics=diagnostics,
        limitations=(
            "Predicates and sensitivity grids are versioned in paper_validation.toml.",
            "Decision curves use common-complete checkpoints and no interpolation.",
        ),
    )


def _metric_family(metric: str) -> tuple[str, str]:
    if metric.endswith("_per_token"):
        return metric.removesuffix("_per_token"), "per_token"
    if metric.endswith("_per_char"):
        return metric.removesuffix("_per_char"), "per_char"
    return metric, "raw"


def _curve_correlation(
    points: tuple[_CurvePoint, ...], metric_a: str, metric_b: str, task: str
) -> float:
    lookup = {
        (point.model_size, point.step, point.metric): point.accuracy
        for point in points
        if point.task == task
    }
    keys = tuple(
        sorted(
            (model_size, step)
            for model_size, step, metric in lookup
            if metric == metric_a and (model_size, step, metric_b) in lookup
        )
    )
    return _spearman(
        (lookup[(*key, metric_a)] for key in keys),
        (lookup[(*key, metric_b)] for key in keys),
    )


def _standardized_task_vectors(
    points: tuple[_CurvePoint, ...],
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    tasks = tuple(sorted({point.task for point in points}))
    feature_sets = {
        task: {
            (point.metric, point.model_size, point.step)
            for point in points
            if point.task == task
        }
        for task in tasks
    }
    common_features = set.intersection(
        *(features for features in feature_sets.values())
    )
    features = tuple(sorted(common_features))
    vectors: list[tuple[str, tuple[float, ...]]] = []
    for task in tasks:
        lookup = {
            (point.metric, point.model_size, point.step): point.accuracy
            for point in points
            if point.task == task
        }
        values = tuple(lookup[feature] for feature in features)
        mean = _mean(values)
        scale = _sample_sd(values)
        standardized = (
            tuple(0.0 for _ in values)
            if scale == 0
            else tuple((value - mean) / scale for value in values)
        )
        vectors.append((task, standardized))
    return tuple(vectors)


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _cluster_tasks(
    points: tuple[_CurvePoint, ...],
) -> tuple[dict[str, int], float]:
    vectors = _standardized_task_vectors(points)
    if len(vectors) < 3 or not vectors[0][1]:
        return {}, 0.0
    farthest = max(
        (
            (_distance(left[1], right[1]), left, right)
            for index, left in enumerate(vectors)
            for right in vectors[index + 1 :]
        ),
        key=lambda item: (item[0], item[1][0], item[2][0]),
    )
    centers = [farthest[1][1], farthest[2][1]]
    assignments: dict[str, int] = {}
    for _ in range(100):
        updated = {
            task: min(
                range(2),
                key=lambda cluster: (_distance(vector, centers[cluster]), cluster),
            )
            for task, vector in vectors
        }
        if updated == assignments:
            break
        assignments = updated
        new_centers: list[tuple[float, ...]] = []
        for cluster in range(2):
            members = tuple(
                vector for task, vector in vectors if assignments[task] == cluster
            )
            if not members:
                return assignments, 0.0
            new_centers.append(
                tuple(
                    _mean(vector[index] for vector in members)
                    for index in range(len(members[0]))
                )
            )
        centers = new_centers
    vector_lookup = dict(vectors)
    silhouettes: list[float] = []
    for task, vector in vectors:
        own = tuple(
            other
            for other, _ in vectors
            if other != task and assignments[other] == assignments[task]
        )
        other = tuple(
            other for other, _ in vectors if assignments[other] != assignments[task]
        )
        if not own or not other:
            silhouettes.append(0.0)
            continue
        a = _mean(_distance(vector, vector_lookup[name]) for name in own)
        b = _mean(_distance(vector, vector_lookup[name]) for name in other)
        silhouettes.append(0.0 if max(a, b) == 0 else (b - a) / max(a, b))
    return assignments, _mean(silhouettes)


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


def _run_proxy_curve_qualitative_attempt(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    observations: tuple[MetricObservation, ...],
    surface: _CurveSurface | None = None,
) -> AttemptResult:
    target, checkpoints, points = (
        _curve_surface(context, observations) if surface is None else surface
    )
    if target is None or not checkpoints or not points:
        return _missing_result(
            context,
            input_data,
            columns=columns,
            recipes=_recipes(context.attempt),
            seeds=_prediction_seeds(context.attempt),
            source_tasks=_source_tasks(_logical_tasks(context.attempt)),
            model_size=_model_sizes(context.attempt)[0],
            reason="Configured qualitative curve surface has no common-complete evidence.",
        )
    outcome = ValidationOutcome.NOT_REPRODUCED
    computed: dict[str, object]
    diagnostic: str

    if context.attempt.id in {"dd-0055-default", "dd-0196-default"}:
        minimum = _parameter(
            context.rule,
            ComparisonParameterName.EQUIVALENCE_DIFFERENCE_MINIMUM,
        )
        maximum_scale = _parameter(
            context.rule, ComparisonParameterName.MAXIMUM_SCALE_PERCENT
        )
        proxy_metrics = set(_metrics(context.attempt)).difference({_PRIMARY_METRIC})
        grouped: dict[tuple[str, str, int], dict[str, float]] = {}
        for point in points:
            if point.percent_target_compute <= maximum_scale:
                grouped.setdefault((point.task, point.model_size, point.step), {})[
                    point.metric
                ] = point.accuracy
        differences = tuple(
            max(values[metric] for metric in proxy_metrics if metric in values)
            - values[_PRIMARY_METRIC]
            for values in grouped.values()
            if _PRIMARY_METRIC in values
            and any(metric in values for metric in proxy_metrics)
        )
        mean_difference = _mean(differences) if differences else -1.0
        satisfied = bool(differences) and mean_difference >= minimum
        computed = {
            "maximum_scale_percent": maximum_scale,
            "comparison_count": len(differences),
            "mean_best_proxy_minus_accuracy": mean_difference,
            "minimum_allowed_difference": minimum,
            "satisfied": satisfied,
        }
        outcome = (
            ValidationOutcome.REPRODUCED
            if satisfied
            else ValidationOutcome.NOT_REPRODUCED
        )
        diagnostic = f"Mean best-proxy minus Accuracy={mean_difference:.12g} over {len(differences)} small-scale comparisons."
    elif context.attempt.id in {"dd-0197-default", "dd-0207-default"}:
        correlations: dict[str, float] = {}
        if context.attempt.id == "dd-0197-default":
            bases = tuple(
                sorted(
                    {
                        _metric_family(metric)[0]
                        for metric in _metrics(context.attempt)
                        if _metric_family(metric)[1] != "raw"
                    }
                )
            )
            for task in _logical_tasks(context.attempt):
                for base in bases:
                    correlations[f"{task}:{base}"] = _curve_correlation(
                        points,
                        f"{base}_per_token",
                        f"{base}_per_char",
                        task,
                    )
        else:
            for task in _logical_tasks(context.attempt):
                for metric in ("norm_correct_prob", "margin"):
                    correlations[f"{task}:{metric}"] = _curve_correlation(
                        points, _PRIMARY_METRIC, metric, task
                    )
        mean_correlation = _mean(correlations.values())
        required = _parameter(context.rule, ComparisonParameterName.SPEARMAN_MINIMUM)
        satisfied = mean_correlation >= required
        outcome = (
            ValidationOutcome.REPRODUCED
            if satisfied
            else ValidationOutcome.DIRECTIONALLY_CONSISTENT
            if mean_correlation > 0
            else ValidationOutcome.NOT_REPRODUCED
        )
        computed = {
            "curve_spearman": correlations,
            "mean_spearman": mean_correlation,
            "required_spearman": required,
            "satisfied": satisfied,
        }
        diagnostic = f"Mean paired-curve Spearman={mean_correlation:.12g}."
    elif context.attempt.id == "dd-0198-default":
        task_results: dict[str, bool] = {}
        task_scores: dict[str, dict[str, float]] = {}
        for task in _logical_tasks(context.attempt):
            normalization_values: dict[str, list[float]] = {}
            for point in points:
                if point.task != task or point.metric == _PRIMARY_METRIC:
                    continue
                normalization = _metric_family(point.metric)[1]
                normalization_values.setdefault(normalization, []).append(
                    point.accuracy
                )
            means = {
                normalization: _mean(values)
                for normalization, values in normalization_values.items()
            }
            task_scores[task] = means
            task_results[task] = bool(means) and means.get("per_char", -1.0) >= max(
                means.values()
            )
        fraction = sum(task_results.values()) / len(task_results)
        required = _parameter(context.rule, ComparisonParameterName.FRACTION_THRESHOLD)
        satisfied = fraction > required
        computed = {
            "normalization_mean_accuracy_by_task": task_scores,
            "per_character_optimal_by_task": task_results,
            "fraction": fraction,
            "required_fraction_exclusive": required,
            "satisfied": satisfied,
        }
        outcome = (
            ValidationOutcome.REPRODUCED
            if satisfied
            else ValidationOutcome.NOT_REPRODUCED
        )
        diagnostic = f"Per-character normalization was optimal on {sum(task_results.values())}/{len(task_results)} tasks."
    elif context.attempt.id == "dd-0199-default":
        maximum_scale = _parameter(
            context.rule, ComparisonParameterName.MAXIMUM_SCALE_PERCENT
        )
        grouped: dict[tuple[str, str, int], dict[str, float]] = {}
        for point in points:
            if point.percent_target_compute <= maximum_scale:
                grouped.setdefault((point.task, point.model_size, point.step), {})[
                    point.metric
                ] = point.accuracy
        comparisons: list[bool] = []
        for values in grouped.values():
            raw = tuple(
                values[metric]
                for metric in ("correct_prob", "total_prob")
                if metric in values
            )
            others = tuple(
                value
                for metric, value in values.items()
                if metric not in {"correct_prob", "total_prob"}
            )
            if raw and others:
                comparisons.append(max(raw) >= max(others))
        fraction = sum(comparisons) / len(comparisons) if comparisons else 0.0
        required = _parameter(context.rule, ComparisonParameterName.FRACTION_THRESHOLD)
        satisfied = fraction > required
        computed = {
            "maximum_scale_percent": maximum_scale,
            "comparison_count": len(comparisons),
            "satisfied_count": sum(comparisons),
            "fraction": fraction,
            "required_fraction_exclusive": required,
            "satisfied": satisfied,
        }
        outcome = (
            ValidationOutcome.REPRODUCED
            if satisfied
            else ValidationOutcome.NOT_REPRODUCED
        )
        diagnostic = f"Raw likelihood metrics matched/exceeded alternatives at {sum(comparisons)}/{len(comparisons)} eligible points."
    elif context.attempt.id == "dd-0202-default":
        configured_clusters = int(
            _parameter(context.rule, ComparisonParameterName.CLUSTER_COUNT)
        )
        if configured_clusters != 2:
            raise ValueError(
                f"{context.rule.id} requires cluster_count=2 for the frozen "
                "deterministic farthest-pair algorithm"
            )
        assignments, silhouette = _cluster_tasks(points)
        required = _parameter(context.rule, ComparisonParameterName.SILHOUETTE_MINIMUM)
        clusters = len(set(assignments.values()))
        satisfied = clusters == configured_clusters and silhouette >= required
        computed = {
            "initialization": "deterministic farthest pair",
            "cluster_count": clusters,
            "required_cluster_count": configured_clusters,
            "assignments": assignments,
            "silhouette": silhouette,
            "required_silhouette": required,
            "satisfied": satisfied,
        }
        outcome = (
            ValidationOutcome.REPRODUCED
            if satisfied
            else ValidationOutcome.DIRECTIONALLY_CONSISTENT
            if clusters == configured_clusters and silhouette > 0
            else ValidationOutcome.NOT_REPRODUCED
        )
        diagnostic = f"Deterministic k=2 silhouette={silhouette:.12g}."
    elif context.attempt.id in {"dd-0203-default", "dd-0204-default"}:
        assignments, silhouette = _cluster_tasks(points)
        cluster_ids = tuple(sorted(set(assignments.values())))
        overlap_by_task: dict[str, float] = {}
        slope_by_task: dict[str, float] = {}
        for task in _logical_tasks(context.attempt):
            task_points = tuple(point for point in points if point.task == task)
            grouped: dict[tuple[str, int], list[float]] = {}
            for point in task_points:
                grouped.setdefault((point.model_size, point.step), []).append(
                    point.accuracy
                )
            overlap_by_task[task] = _mean(
                max(values) - min(values) for values in grouped.values()
            )
            ordered = sorted(
                (
                    _mean(values),
                    next(
                        point.compute
                        for point in task_points
                        if (point.model_size, point.step) == key
                    ),
                )
                for key, values in grouped.items()
            )
            slope_by_task[task] = _linear_slope(
                (math.log10(value[1]) for value in ordered if value[1] > 0),
                (value[0] for value in ordered if value[1] > 0),
            )
        cluster_overlap = {
            cluster: _mean(
                overlap_by_task[task]
                for task in assignments
                if assignments[task] == cluster
            )
            for cluster in cluster_ids
        }
        overlap_cluster = min(
            cluster_overlap, key=lambda cluster: (cluster_overlap[cluster], cluster)
        )
        selected_cluster = (
            overlap_cluster
            if context.attempt.id == "dd-0203-default"
            else next(
                (cluster for cluster in cluster_ids if cluster != overlap_cluster),
                overlap_cluster,
            )
        )
        selected_tasks = tuple(
            sorted(
                task for task in assignments if assignments[task] == selected_cluster
            )
        )
        if context.attempt.id == "dd-0203-default":
            maximum_overlap = _parameter(
                context.rule, ComparisonParameterName.OVERLAP_RANGE_MAXIMUM
            )
            slope_minimum = _parameter(
                context.rule, ComparisonParameterName.OLS_SLOPE_MINIMUM
            )
            observed_overlap = _mean(overlap_by_task[task] for task in selected_tasks)
            observed_slope = _mean(slope_by_task[task] for task in selected_tasks)
            satisfied = (
                observed_overlap <= maximum_overlap and observed_slope > slope_minimum
            )
            outcome = (
                ValidationOutcome.REPRODUCED
                if satisfied
                else ValidationOutcome.DIRECTIONALLY_CONSISTENT
                if observed_slope > slope_minimum
                else ValidationOutcome.NOT_REPRODUCED
            )
            computed = {
                "cluster_tasks": list(selected_tasks),
                "cluster_silhouette": silhouette,
                "mean_metric_range": observed_overlap,
                "maximum_metric_range": maximum_overlap,
                "mean_slope_per_decade": observed_slope,
                "satisfied": satisfied,
            }
            diagnostic = f"Overlap cluster mean range={observed_overlap:.12g}, mean slope={observed_slope:.12g}."
        else:
            flat_maximum = _parameter(
                context.rule,
                ComparisonParameterName.EARLY_SLOPE_ABSOLUTE_MAXIMUM,
            )
            convergence = _parameter(
                context.rule, ComparisonParameterName.CONVERGENCE_TOLERANCE
            )
            raw_metrics = {"correct_prob", "total_prob"}
            raw_slopes: list[float] = []
            final_gaps: list[float] = []
            initial_gaps: list[float] = []
            for task in selected_tasks:
                task_points = tuple(point for point in points if point.task == task)
                for metric in raw_metrics:
                    curve = sorted(
                        (point.compute, point.accuracy)
                        for point in task_points
                        if point.metric == metric and point.compute > 0
                    )
                    raw_slopes.append(
                        _linear_slope(
                            (math.log10(item[0]) for item in curve),
                            (item[1] for item in curve),
                        )
                    )
                by_checkpoint: dict[tuple[str, int], dict[str, float]] = {}
                compute_by_checkpoint: dict[tuple[str, int], float] = {}
                for point in task_points:
                    key = (point.model_size, point.step)
                    by_checkpoint.setdefault(key, {})[point.metric] = point.accuracy
                    compute_by_checkpoint[key] = point.compute
                ordered = tuple(
                    values
                    for key, values in sorted(
                        by_checkpoint.items(),
                        key=lambda item: (compute_by_checkpoint[item[0]], item[0]),
                    )
                )
                gaps = [
                    abs(
                        max(values.get(metric, -1.0) for metric in raw_metrics)
                        - max(
                            value
                            for metric, value in values.items()
                            if metric not in raw_metrics
                        )
                    )
                    for values in ordered
                    if any(metric in values for metric in raw_metrics)
                    and any(metric not in raw_metrics for metric in values)
                ]
                if gaps:
                    initial_gaps.append(gaps[0])
                    final_gaps.append(gaps[-1])
            maximum_raw_slope = max((abs(value) for value in raw_slopes), default=1.0)
            final_gap = _mean(final_gaps) if final_gaps else 1.0
            initial_gap = _mean(initial_gaps) if initial_gaps else 1.0
            direction = final_gap < initial_gap
            satisfied = maximum_raw_slope <= flat_maximum and final_gap <= convergence
            outcome = (
                ValidationOutcome.REPRODUCED
                if satisfied
                else ValidationOutcome.DIRECTIONALLY_CONSISTENT
                if direction
                else ValidationOutcome.NOT_REPRODUCED
            )
            computed = {
                "cluster_tasks": list(selected_tasks),
                "cluster_silhouette": silhouette,
                "maximum_absolute_raw_slope": maximum_raw_slope,
                "flat_slope_maximum": flat_maximum,
                "initial_mean_gap": initial_gap,
                "final_mean_gap": final_gap,
                "convergence_tolerance": convergence,
                "direction_satisfied": direction,
                "satisfied": satisfied,
            }
            diagnostic = f"Raw max abs slope={maximum_raw_slope:.12g}; final convergence gap={final_gap:.12g}."
    elif context.attempt.id == "dd-0205-default":
        maximum_compute = max(point.compute for point in points)
        eligible = tuple(
            point for point in points if point.compute >= maximum_compute / 10
        )
        grouped: dict[tuple[str, str, int], dict[str, float]] = {}
        for point in eligible:
            grouped.setdefault((point.task, point.model_size, point.step), {})[
                point.metric
            ] = point.accuracy
        comparisons = []
        raw_metrics = {"correct_prob", "total_prob"}
        for values in grouped.values():
            raw = tuple(
                value for metric, value in values.items() if metric in raw_metrics
            )
            others = tuple(
                value for metric, value in values.items() if metric not in raw_metrics
            )
            if raw and others:
                comparisons.append(max(others) > max(raw))
        fraction = sum(comparisons) / len(comparisons) if comparisons else 0.0
        required = _parameter(context.rule, ComparisonParameterName.FRACTION_THRESHOLD)
        satisfied = fraction > required
        computed = {
            "last_decade_minimum_compute": maximum_compute / 10,
            "comparison_count": len(comparisons),
            "overtake_count": sum(comparisons),
            "fraction": fraction,
            "required_fraction_exclusive": required,
            "satisfied": satisfied,
        }
        outcome = (
            ValidationOutcome.REPRODUCED
            if satisfied
            else ValidationOutcome.NOT_REPRODUCED
        )
        diagnostic = f"Other metrics overtook raw likelihood at {sum(comparisons)}/{len(comparisons)} last-decade points."
    elif context.attempt.id == "dd-0206-default":
        threshold = _parameter(context.rule, ComparisonParameterName.DECLINE_THRESHOLD)
        declines: list[dict[str, object]] = []
        for task in _logical_tasks(context.attempt):
            for metric in ("correct_prob", "total_prob"):
                curve = sorted(
                    (point.compute, point.accuracy, point.model_size, point.step)
                    for point in points
                    if point.task == task and point.metric == metric
                )
                for before, after in zip(curve, curve[1:], strict=False):
                    drop = before[1] - after[1]
                    if drop >= threshold:
                        declines.append(
                            {
                                "task": task,
                                "metric": metric,
                                "compute_before": before[0],
                                "compute_after": after[0],
                                "drop": drop,
                            }
                        )
        satisfied = bool(declines)
        computed = {
            "decline_threshold": threshold,
            "decline_count": len(declines),
            "declines": declines,
            "satisfied": satisfied,
        }
        outcome = (
            ValidationOutcome.REPRODUCED
            if satisfied
            else ValidationOutcome.NOT_REPRODUCED
        )
        diagnostic = f"Found {len(declines)} adjacent raw-likelihood declines at least {threshold:.12g}."
    else:
        raise ValueError(
            f"no qualitative proxy implementation for {context.attempt.id}"
        )
    return _qualitative_result(
        context,
        input_data,
        columns=columns,
        target=target,
        checkpoints=checkpoints,
        computed_value=computed,
        outcome=outcome,
        diagnostics=(diagnostic,),
    )


def _configured_noise_checkpoints(
    context: _Context,
    observations: tuple[MetricObservation, ...],
) -> tuple[CheckpointRows, ...]:
    model_size = next(
        (size for size in _model_sizes(context.attempt) if size != "1B"),
        "1B",
    )
    if context.attempt.id == "dd-0098-default":
        model_size = "1B"
    universe = _universe(
        model_size=model_size,
        recipes=_recipes(context.attempt),
        seeds=_seeds_for_model(context.attempt, model_size),
        source_tasks=_source_tasks(_logical_tasks(context.attempt)),
        metrics=tuple(dict.fromkeys((_PRIMARY_METRIC, *_metrics(context.attempt)))),
    )
    preceding_ids = tuple(
        sensitivity_id
        for sensitivity_id in context.attempt.sensitivity_ids
        if "preceding-common-complete" in sensitivity_id
    )
    return _common_checkpoints(
        observations, universe, preceding_count=len(preceding_ids)
    )


def _noise_improvement_result(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    checkpoint: CheckpointRows,
    result_id: str,
    role: AttemptRole,
) -> AttemptResult:
    points = _noise_points(checkpoint)
    by_task: dict[str, dict[str, _NoisePoint]] = {}
    for point in points:
        by_task.setdefault(point.task, {})[point.metric] = point
    task_results: dict[str, bool] = {}
    details: dict[str, dict[str, object]] = {}
    allowed_metrics = set(_metrics(context.attempt)).difference({_PRIMARY_METRIC})
    for task, metrics in sorted(by_task.items()):
        primary = metrics.get(_PRIMARY_METRIC)
        if primary is None:
            continue
        candidates = tuple(
            point for metric, point in metrics.items() if metric in allowed_metrics
        )
        improved = tuple(
            point
            for point in candidates
            if point.noise < primary.noise or point.spread > primary.spread
        )
        task_results[task] = bool(improved)
        details[task] = {
            "primary_noise": primary.noise,
            "primary_spread": primary.spread,
            "improving_metrics": sorted(point.metric for point in improved),
        }
    fraction = sum(task_results.values()) / len(task_results) if task_results else 0.0
    required = _parameter(context.rule, ComparisonParameterName.FRACTION_THRESHOLD)
    satisfied = fraction > required
    base = _qualitative_result(
        context,
        input_data,
        columns=columns,
        target=None,
        checkpoints=(checkpoint,),
        computed_value={
            "task_details": details,
            "improved_task_count": sum(task_results.values()),
            "task_count": len(task_results),
            "fraction": fraction,
            "required_fraction_exclusive": required,
            "satisfied": satisfied,
        },
        outcome=(
            ValidationOutcome.REPRODUCED
            if satisfied
            else ValidationOutcome.NOT_REPRODUCED
        ),
        diagnostics=(
            f"A configured proxy improved noise or spread on {sum(task_results.values())}/{len(task_results)} tasks at step {checkpoint.step}.",
        ),
    )
    if role is AttemptRole.DEFAULT:
        return base
    return base.model_copy(
        update={
            "attempt_id": result_id,
            "role": AttemptRole.SENSITIVITY,
            "parent_attempt_id": context.attempt.id,
        }
    )


def _run_noise_improvement_attempt(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    observations: tuple[MetricObservation, ...],
) -> tuple[AttemptResult, ...]:
    checkpoints = _configured_noise_checkpoints(context, observations)
    if not checkpoints:
        return (
            _missing_result(
                context,
                input_data,
                columns=columns,
                recipes=_recipes(context.attempt),
                seeds=_declared_seeds(context.attempt),
                source_tasks=_source_tasks(_logical_tasks(context.attempt)),
                model_size=_model_sizes(context.attempt)[0],
                reason="No common-complete checkpoint exists for the configured noise/spread comparison.",
            ),
        )
    preceding_ids = tuple(
        sensitivity_id
        for sensitivity_id in context.attempt.sensitivity_ids
        if "preceding-common-complete" in sensitivity_id
    )
    ids = (context.attempt.id, *preceding_ids)
    return tuple(
        _noise_improvement_result(
            context,
            input_data,
            columns=columns,
            checkpoint=checkpoint,
            result_id=result_id,
            role=AttemptRole.DEFAULT if index == 0 else AttemptRole.SENSITIVITY,
        )
        for index, (result_id, checkpoint) in enumerate(
            zip(ids, checkpoints, strict=False)
        )
    )


def _noise_association_result(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    observations: tuple[MetricObservation, ...],
    checkpoint: CheckpointRows,
    result_id: str,
    role: AttemptRole,
) -> AttemptResult:
    target_universe = _universe(
        model_size="1B",
        recipes=_recipes(context.attempt),
        seeds=_TARGET_SEEDS,
        source_tasks=checkpoint.universe.source_tasks,
        metrics=(_PRIMARY_METRIC,),
    )
    targets = _common_checkpoints(observations, target_universe, preceding_count=0)
    if not targets:
        raise ValueError(f"{context.attempt.id} has no common-complete 1B target")
    target = targets[0]
    decisions = _decision_aggregates(_decision_rows(_target_scores(target), checkpoint))
    decision_lookup = {(row.task, row.metric): row.accuracy for row in decisions}
    noise = _noise_points(checkpoint)
    usable = tuple(
        point
        for point in noise
        if point.noise > 0 and (point.task, point.metric) in decision_lookup
    )
    ratios = tuple(point.spread / point.noise for point in usable)
    accuracies = tuple(decision_lookup[(point.task, point.metric)] for point in usable)
    correlation = _spearman(ratios, accuracies)
    minimum = _parameter(context.rule, ComparisonParameterName.SPEARMAN_MINIMUM)
    satisfied = correlation > minimum
    base = _qualitative_result(
        context,
        input_data,
        columns=columns,
        target=target,
        checkpoints=(checkpoint,),
        computed_value={
            "association_point_count": len(usable),
            "spread_noise_ratio_spearman": correlation,
            "required_spearman_exclusive": minimum,
            "satisfied": satisfied,
        },
        outcome=(
            ValidationOutcome.REPRODUCED
            if satisfied
            else ValidationOutcome.NOT_REPRODUCED
        ),
        diagnostics=(
            f"Spread/noise ratio versus decision-accuracy Spearman={correlation:.12g} over {len(usable)} points.",
        ),
    )
    if role is AttemptRole.DEFAULT:
        return base
    return base.model_copy(
        update={
            "attempt_id": result_id,
            "role": AttemptRole.SENSITIVITY,
            "parent_attempt_id": context.attempt.id,
        }
    )


def _run_noise_association_attempt(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    observations: tuple[MetricObservation, ...],
) -> tuple[AttemptResult, ...]:
    checkpoints = _configured_noise_checkpoints(context, observations)
    if not checkpoints:
        return (
            _missing_result(
                context,
                input_data,
                columns=columns,
                recipes=_recipes(context.attempt),
                seeds=_declared_seeds(context.attempt),
                source_tasks=_source_tasks(_logical_tasks(context.attempt)),
                model_size=_model_sizes(context.attempt)[0],
                reason="No common-complete checkpoint exists for the configured association.",
            ),
        )
    preceding_ids = tuple(
        sensitivity_id
        for sensitivity_id in context.attempt.sensitivity_ids
        if "preceding-common-complete" in sensitivity_id
    )
    ids = (context.attempt.id, *preceding_ids)
    return tuple(
        _noise_association_result(
            context,
            input_data,
            columns=columns,
            observations=observations,
            checkpoint=checkpoint,
            result_id=result_id,
            role=AttemptRole.DEFAULT if index == 0 else AttemptRole.SENSITIVITY,
        )
        for index, (result_id, checkpoint) in enumerate(
            zip(ids, checkpoints, strict=False)
        )
    )


def _sd_claim_result(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    checkpoint: CheckpointRows,
    result_id: str,
    role: AttemptRole,
) -> AttemptResult:
    logical = _logical_values(checkpoint)
    tasks = _logical_tasks_from_sources(checkpoint.universe.source_tasks)
    expected_task_count = int(
        _parameter(context.rule, ComparisonParameterName.TASK_COUNT)
    )
    if len(tasks) != expected_task_count:
        raise ValueError(
            f"{context.rule.id} expected {expected_task_count} logical tasks, "
            f"found {len(tasks)}"
        )
    target_sd = _parameter(
        context.rule, ComparisonParameterName.STANDARD_DEVIATION_TARGET
    )
    tolerance = _parameter(
        context.rule, ComparisonParameterName.STANDARD_DEVIATION_TOLERANCE
    )
    task_maxima: dict[str, float] = {}
    matching_recipes: dict[str, list[str]] = {}
    for task in tasks:
        recipe_sds = {
            recipe: _sample_sd(
                logical[(task, recipe, seed, _PRIMARY_METRIC)]
                for seed in checkpoint.universe.seeds
            )
            for recipe in checkpoint.universe.recipes
        }
        task_maxima[task] = max(recipe_sds.values())
        matching_recipes[task] = sorted(
            recipe
            for recipe, value in recipe_sds.items()
            if abs(value - target_sd) <= tolerance
        )
    matching_tasks = tuple(task for task in tasks if matching_recipes[task])
    required = _parameter(
        context.rule, ComparisonParameterName.TASK_COUNT_MINIMUM_EXCLUSIVE
    )
    satisfied = len(matching_tasks) > required
    base = _qualitative_result(
        context,
        input_data,
        columns=columns,
        target=None,
        checkpoints=(checkpoint,),
        computed_value={
            "standard_deviation_target": target_sd,
            "tolerance": tolerance,
            "task_maximum_sample_sd": task_maxima,
            "matching_recipes_by_task": matching_recipes,
            "matching_task_count": len(matching_tasks),
            "expected_task_count": expected_task_count,
            "required_task_count_exclusive": required,
            "maximum_observed_sample_sd": max(task_maxima.values()),
            "satisfied": satisfied,
        },
        outcome=(
            ValidationOutcome.APPROXIMATELY_REPRODUCED
            if satisfied
            else ValidationOutcome.NOT_REPRODUCED
        ),
        diagnostics=(
            f"{len(matching_tasks)}/{len(tasks)} tasks have some recipe sample SD within {tolerance:.12g} of {target_sd:.12g}; maximum={max(task_maxima.values()):.12g}.",
        ),
    )
    if role is AttemptRole.DEFAULT:
        return base
    return base.model_copy(
        update={
            "attempt_id": result_id,
            "role": AttemptRole.SENSITIVITY,
            "parent_attempt_id": context.attempt.id,
        }
    )


def _run_sd_claim_attempt(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    observations: tuple[MetricObservation, ...],
) -> tuple[AttemptResult, ...]:
    checkpoints = _configured_noise_checkpoints(context, observations)
    if not checkpoints:
        return (
            _missing_result(
                context,
                input_data,
                columns=columns,
                recipes=_recipes(context.attempt),
                seeds=_declared_seeds(context.attempt),
                source_tasks=_source_tasks(_logical_tasks(context.attempt)),
                model_size="1B",
                reason="No common-complete 1B checkpoint exists for the configured seed-SD claim.",
            ),
        )
    preceding_ids = tuple(
        sensitivity_id
        for sensitivity_id in context.attempt.sensitivity_ids
        if "preceding-common-complete" in sensitivity_id
    )
    ids = (context.attempt.id, *preceding_ids)
    return tuple(
        _sd_claim_result(
            context,
            input_data,
            columns=columns,
            checkpoint=checkpoint,
            result_id=result_id,
            role=AttemptRole.DEFAULT if index == 0 else AttemptRole.SENSITIVITY,
        )
        for index, (result_id, checkpoint) in enumerate(
            zip(ids, checkpoints, strict=False)
        )
    )


def _run_crossover_attempt(
    context: _Context,
    input_data: _Input,
    *,
    columns: tuple[str, ...],
    observations: tuple[MetricObservation, ...],
) -> AttemptResult:
    recipes = _recipes(context.attempt)
    source_tasks = _source_tasks(_logical_tasks(context.attempt))
    checkpoints: list[CheckpointRows] = []
    for model_size in _model_sizes(context.attempt):
        seeds = _seeds_for_model(context.attempt, model_size)
        universe = _universe(
            model_size=model_size,
            recipes=recipes,
            seeds=seeds,
            source_tasks=source_tasks,
            metrics=(_PRIMARY_METRIC,),
        )
        checkpoints.extend(_all_common_checkpoints(observations, universe))
    by_compute: dict[float, dict[str, list[float]]] = {}
    for checkpoint in checkpoints:
        logical = _logical_values(checkpoint)
        tasks = _logical_tasks_from_sources(checkpoint.universe.source_tasks)
        for recipe in recipes:
            score = _mean(
                logical[(task, recipe, seed, _PRIMARY_METRIC)]
                for task in tasks
                for seed in checkpoint.universe.seeds
            )
            by_compute.setdefault(checkpoint.actual_compute, {}).setdefault(
                recipe, []
            ).append(score)
    complete = {
        compute: values
        for compute, values in by_compute.items()
        if set(values) == set(recipes)
    }
    scores = tuple(
        ScaleRecipeScore(
            scale=compute,
            recipe=recipe,
            score=_mean(values[recipe]),
        )
        for compute, values in sorted(complete.items())
        for recipe in recipes
    )
    summary = summarize_crossovers(
        scores,
        expected_recipes=recipes,
        tie_policy=CrossoverTiePolicy.BRIDGE_TIED_POINTS,
    )
    fraction = summary.pairs_with_crossover / summary.pair_count
    required = _parameter(context.rule, ComparisonParameterName.FRACTION_THRESHOLD)
    satisfied = fraction > required
    return _qualitative_result(
        context,
        input_data,
        columns=columns,
        target=None,
        checkpoints=tuple(checkpoints),
        computed_value={
            "scale_count": summary.scale_count,
            "pair_count": summary.pair_count,
            "pairs_with_crossover": summary.pairs_with_crossover,
            "crossover_count": summary.crossover_count,
            "pair_fraction": fraction,
            "required_fraction_exclusive": required,
            "tie_policy": summary.tie_policy.value,
            "satisfied": satisfied,
        },
        outcome=(
            ValidationOutcome.REPRODUCED
            if satisfied
            else ValidationOutcome.NOT_REPRODUCED
        ),
        diagnostics=(
            f"{summary.pairs_with_crossover}/{summary.pair_count} recipe pairs cross at least once across {summary.scale_count} compute levels.",
        ),
    )


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
        outcome=ValidationOutcome.DESCRIPTIVE_ONLY,
        diagnostics=(f"Persisted {len(points)} finite paper-analog points.",),
        limitations=(
            "The derived plot series is available for visual comparison, but no frozen semantic predicate adjudicates qualitative words such as low, high, often, or align.",
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
                outcome=ValidationOutcome.DESCRIPTIVE_ONLY,
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
    """Run the objectively specified proxy-metric format-3 attempts."""
    del repository_root
    contexts = _contexts(registry, contract, AnalysisId.PROXY_METRICS)
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
            (
                _PRIMARY_METRIC,
                *(
                    metric
                    for context in contexts
                    for metric in _metrics(context.attempt)
                ),
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
    curve_surfaces: dict[tuple[tuple[str, ...], ...], _CurveSurface] = {}

    def run_context(
        context: _Context,
    ) -> tuple[tuple[AttemptResult, ...], PlotSeries | None]:
        if context.attempt.id in _THRESHOLD_ATTEMPTS:
            attempt_columns = (*_BASE_COLUMNS, *_PAPER_METRICS)
            return (
                (
                    _run_threshold_attempt(
                        context,
                        input_data,
                        columns=attempt_columns,
                        observations=observations,
                    ),
                ),
                None,
            )
        if context.attempt.id in _PROXY_PLOT_ATTEMPTS:
            attempt_columns = (*_BASE_COLUMNS, *_PER_CHARACTER_PLOT_METRICS)
            result, plot = _run_proxy_plot_attempt(
                context,
                input_data,
                columns=attempt_columns,
                observations=observations,
            )
            return (result,), plot
        if context.attempt.id == "dd-0057-default":
            attempt_columns = tuple(
                dict.fromkeys(
                    (*_BASE_COLUMNS, _PRIMARY_METRIC, *_metrics(context.attempt))
                )
            )
            return (
                _run_noise_improvement_attempt(
                    context,
                    input_data,
                    columns=attempt_columns,
                    observations=observations,
                ),
                None,
            )
        attempt_columns = tuple(
            dict.fromkeys((*_BASE_COLUMNS, _PRIMARY_METRIC, *_metrics(context.attempt)))
        )
        surface_key = _curve_surface_key(context)
        surface = curve_surfaces.get(surface_key)
        if surface is None:
            surface = _curve_surface(context, observations)
            curve_surfaces[surface_key] = surface
        return (
            (
                _run_proxy_curve_qualitative_attempt(
                    context,
                    input_data,
                    columns=attempt_columns,
                    observations=observations,
                    surface=surface,
                ),
            ),
            None,
        )

    for context in contexts:
        context_results, plot = run_context(context)
        results.extend(context_results)
        if plot is not None:
            series.append(plot)
        for (
            sensitivity_id,
            name,
            value,
            sensitivity_context,
        ) in _comparison_sensitivity_contexts(context):
            sensitivity_results, _ = run_context(sensitivity_context)
            default = next(
                result
                for result in sensitivity_results
                if result.attempt_id == context.attempt.id
            )
            results.append(
                _comparison_sensitivity_result(
                    default,
                    sensitivity_id=sensitivity_id,
                    parameter_name=name,
                    parameter_value=value,
                )
            )
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
    """Run the objectively specified noise/spread format-3 attempts."""
    del repository_root
    contexts = _contexts(registry, contract, AnalysisId.NOISE_SPREAD)
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

    def run_context(
        context: _Context,
    ) -> tuple[tuple[AttemptResult, ...], PlotSeries | None]:
        attempt_columns = tuple(
            dict.fromkeys((*_BASE_COLUMNS, _PRIMARY_METRIC, *_metrics(context.attempt)))
        )
        if context.attempt.id in _NOISE_PLOT_ATTEMPTS:
            return _run_noise_plot_attempt(
                context,
                input_data,
                columns=attempt_columns,
                observations=observations,
            )
        if context.attempt.id in {"dd-0056-default", "dd-0211-default"}:
            return (
                _run_noise_association_attempt(
                    context,
                    input_data,
                    columns=attempt_columns,
                    observations=observations,
                ),
                None,
            )
        if context.attempt.id == "dd-0098-default":
            return (
                _run_sd_claim_attempt(
                    context,
                    input_data,
                    columns=attempt_columns,
                    observations=observations,
                ),
                None,
            )
        if context.attempt.id == "dd-0194-default":
            return (
                (
                    _run_crossover_attempt(
                        context,
                        input_data,
                        columns=attempt_columns,
                        observations=observations,
                    ),
                ),
                None,
            )
        if context.attempt.id == "dd-0212-default":
            return (
                _run_noise_improvement_attempt(
                    context,
                    input_data,
                    columns=attempt_columns,
                    observations=observations,
                ),
                None,
            )
        raise ValueError(f"no noise/spread implementation for {context.attempt.id}")

    for context in contexts:
        context_results, plot = run_context(context)
        results.extend(context_results)
        if plot is not None:
            series.append(plot)
        for (
            sensitivity_id,
            name,
            value,
            sensitivity_context,
        ) in _comparison_sensitivity_contexts(context):
            sensitivity_results, _ = run_context(sensitivity_context)
            default = next(
                result
                for result in sensitivity_results
                if result.attempt_id == context.attempt.id
            )
            results.append(
                _comparison_sensitivity_result(
                    default,
                    sensitivity_id=sensitivity_id,
                    parameter_name=name,
                    parameter_value=value,
                )
            )
    return tuple(sorted(results, key=lambda result: result.attempt_id)), tuple(
        sorted(series, key=lambda plot: plot.id)
    )


__all__ = ["run_noise_spread_attempts", "run_proxy_metrics_attempts"]

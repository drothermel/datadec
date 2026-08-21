from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum, UNIQUE, verify
from typing import Never

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


@verify(UNIQUE)
class ScalingVariant(StrEnum):
    THREE_PARAMETER_TWO_STAGE = "3p_two_stage"
    TWO_PARAMETER_TWO_STAGE = "2p_two_stage"
    FIVE_PARAMETER_ND_TWO_STAGE = "5p_nd_two_stage"
    THREE_PARAMETER_SINGLE_STEP = "3p_single_step"
    FIVE_PARAMETER_ND_SINGLE_STEP = "5p_nd_single_step"
    HELPER = "helper"
    LATE = "late"
    HELPER_LATE = "helper_late"


@verify(UNIQUE)
class FitFailureReason(StrEnum):
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    DEGENERATE_PREDICTOR = "degenerate_predictor"
    OPTIMIZER_FAILED = "optimizer_failed"
    ILL_CONDITIONED = "ill_conditioned"
    NON_FINITE_RESULT = "non_finite_result"


@dataclass(frozen=True, slots=True)
class OptimizerPolicy:
    """Fixed deterministic least-squares policy shared by every fit.

    Predictors are divided by their geometric mean and residuals by the response
    range (or a finite unit/magnitude fallback for a constant response). Loss
    exponents are bounded to [1e-4, 5], normalized positive amplitudes to a
    finite response-scaled range, and offsets around the observed response.
    Sigmoid slopes are negative for the paper's higher-score/lower-loss
    orientation; single-stage latent coefficients have finite [-20, 20] bounds.
    Every model uses its fixed, ordered start grid. Successful starts are ordered
    by cost, then lexicographically by parameters when costs differ by no more
    than ``cost_tie_tolerance``. The tolerances below are passed unchanged to
    ``scipy.optimize.least_squares``.
    """

    ftol: float = 1e-12
    xtol: float = 1e-12
    gtol: float = 1e-12
    max_nfev: int = 20_000
    cost_tie_tolerance: float = 1e-12
    max_condition_number: float = 1e14
    fail_on_ill_conditioned: bool = True


@dataclass(frozen=True, slots=True)
class ScalingPolicy:
    optimizer: OptimizerPolicy = OptimizerPolicy()
    final_window_fraction: float = 0.10
    complete_progress: float = 1.0
    progress_tolerance: float = 1e-12
    late_progress_threshold: float = 0.50
    helper_loss: float = 0.0
    helper_score: float = 1.0
    relative_error_floor: float = 0.0


DEFAULT_SCALING_POLICY = ScalingPolicy()


@dataclass(frozen=True, slots=True)
class ScalingCoordinates:
    parameter_count: float
    token_count: float
    compute: float


@dataclass(frozen=True, slots=True)
class CheckpointLoss:
    size_id: str
    coordinates: ScalingCoordinates
    progress: float
    loss: float


@dataclass(frozen=True, slots=True)
class FinalLoss:
    size_id: str
    coordinates: ScalingCoordinates
    loss: float
    averaged_checkpoint_count: int


@dataclass(frozen=True, slots=True)
class EvaluationPoint:
    size_id: str
    coordinates: ScalingCoordinates
    progress: float
    loss: float
    score: float


@dataclass(frozen=True, slots=True)
class StageTwoSelection:
    points: tuple[EvaluationPoint, ...]
    observed_count: int
    selected_observed_count: int
    helper_count: int


@dataclass(frozen=True, slots=True)
class FitFailure:
    reason: FitFailureReason
    variant: ScalingVariant
    stage: str
    observation_count: int
    parameter_count: int
    message: str


class ScalingFitError(RuntimeError):
    def __init__(self, failure: FitFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


@dataclass(frozen=True, slots=True)
class FitDiagnostics:
    stage: str
    observation_count: int
    parameter_count: int
    attempted_starts: int
    successful_starts: int
    cost: float
    optimality: float
    function_evaluations: int
    jacobian_rank: int
    condition_number: float


@dataclass(frozen=True, slots=True)
class ParameterFit:
    names: tuple[str, ...]
    values: tuple[float, ...]
    diagnostics: FitDiagnostics


@dataclass(frozen=True, slots=True)
class ScalingFit:
    variant: ScalingVariant
    stage_one: ParameterFit | None
    stage_two: ParameterFit | None
    combined: ParameterFit | None
    selected_stage_two_count: int
    helper_count: int


@dataclass(frozen=True, slots=True)
class ScalingTarget:
    size_id: str
    coordinates: ScalingCoordinates
    actual_score: float


@dataclass(frozen=True, slots=True)
class HeldOutPrediction:
    target: ScalingTarget
    predicted_score: float
    absolute_error_percent: float
    relative_error_percent: float
    fit: ScalingFit


@dataclass(frozen=True, slots=True)
class PredictionCell:
    cell_id: str
    predicted: float
    actual: float


@dataclass(frozen=True, slots=True)
class CellError:
    cell_id: str
    absolute_error_percent: float
    relative_error_percent: float


@dataclass(frozen=True, slots=True)
class PredictionErrorSummary:
    cells: tuple[CellError, ...]
    mean_absolute_error_percent: float
    mean_relative_error_percent: float


@dataclass(frozen=True, slots=True)
class SizeSubset:
    subset_id: str
    sizes: tuple[str, ...]
    kind: str


@dataclass(frozen=True, slots=True)
class ScaleCompute:
    size_id: str
    compute: float


@dataclass(frozen=True, slots=True)
class PairDecisionAccuracy:
    accuracy: float
    correct: int
    denominator: int
    total_pairs: int
    target_ties: int
    predicted_ties: int


@dataclass(frozen=True, slots=True)
class AccuracyAtCompute:
    point_id: str
    compute: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class FrontierComparison:
    multi_scale: AccuracyAtCompute
    single_scale: AccuracyAtCompute
    accuracy_difference: float


@dataclass(frozen=True, slots=True)
class RankingAtSize:
    size_id: str
    recipe_scores: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class AdjacentCrossoverCount:
    smaller_size_id: str
    larger_size_id: str
    crossover_count: int
    comparable_pairs: int
    tied_pairs: int


def loss_compute_3p(compute: float, a: float, alpha: float, e: float) -> float:
    """Evaluate ``L(C) = A / C**alpha + E``."""
    return a / compute**alpha + e


def loss_compute_2p(compute: float, a: float, alpha: float) -> float:
    """Evaluate ``L(C) = A / C**alpha``."""
    return a / compute**alpha


def loss_tokens_parameters_5p(
    token_count: float,
    parameter_count: float,
    a: float,
    alpha: float,
    b: float,
    beta: float,
    e: float,
) -> float:
    """Evaluate the paper's ``L(N,D)`` with N=tokens and D=parameters."""
    return a / token_count**alpha + b / parameter_count**beta + e


def accuracy_from_loss(
    loss: float, a: float, k: float, loss_midpoint: float, b: float
) -> float:
    """Evaluate ``Acc(L) = a / (1 + exp(-k(L-L0))) + b``."""
    return a * _scalar_expit(k * (loss - loss_midpoint)) + b


def accuracy_compute_single_3p(
    compute: float,
    loss_a: float,
    alpha: float,
    e: float,
    score_a: float,
    k: float,
    loss_midpoint: float,
    b: float,
) -> float:
    loss = loss_compute_3p(compute, loss_a, alpha, e)
    return accuracy_from_loss(loss, score_a, k, loss_midpoint, b)


def accuracy_tokens_parameters_single_5p(
    token_count: float,
    parameter_count: float,
    a: float,
    alpha: float,
    b: float,
    beta: float,
    e: float,
    score_a: float,
    score_b: float,
) -> float:
    """Evaluate the seven-free-parameter single-stage paper equation."""
    latent = loss_tokens_parameters_5p(
        token_count, parameter_count, a, alpha, b, beta, e
    )
    return score_a * _scalar_expit(latent) + score_b


def aggregate_final_losses(
    checkpoints: Iterable[CheckpointLoss],
    *,
    policy: ScalingPolicy = DEFAULT_SCALING_POLICY,
) -> tuple[FinalLoss, ...]:
    """Average each size's last checkpoint window and retain final coordinates."""
    _validate_policy(policy)
    groups: dict[str, list[CheckpointLoss]] = {}
    for point in checkpoints:
        _validate_checkpoint_loss(point)
        groups.setdefault(point.size_id, []).append(point)
    if not groups:
        raise ValueError("at least one checkpoint loss is required")

    result: list[FinalLoss] = []
    threshold = policy.complete_progress - policy.final_window_fraction
    for size_id in sorted(groups):
        ordered = sorted(groups[size_id], key=lambda point: point.progress)
        if len({point.progress for point in ordered}) != len(ordered):
            raise ValueError(f"duplicate checkpoint progress for size {size_id!r}")
        final = ordered[-1]
        if not math.isclose(
            final.progress,
            policy.complete_progress,
            rel_tol=0.0,
            abs_tol=policy.progress_tolerance,
        ):
            raise ValueError(f"size {size_id!r} has no complete checkpoint")
        selected = tuple(point for point in ordered if point.progress >= threshold)
        result.append(
            FinalLoss(
                size_id=size_id,
                coordinates=final.coordinates,
                loss=_mean(point.loss for point in selected),
                averaged_checkpoint_count=len(selected),
            )
        )
    return tuple(result)


def select_stage_two_points(
    points: Iterable[EvaluationPoint],
    *,
    variant: ScalingVariant,
    policy: ScalingPolicy = DEFAULT_SCALING_POLICY,
) -> StageTwoSelection:
    _validate_policy(policy)
    observed = tuple(
        sorted((_validated_evaluation(point) for point in points), key=_evaluation_key)
    )
    if not observed:
        raise ValueError("at least one evaluation point is required")
    late = variant in {ScalingVariant.LATE, ScalingVariant.HELPER_LATE}
    helper = variant in {ScalingVariant.HELPER, ScalingVariant.HELPER_LATE}
    selected = (
        tuple(
            point
            for point in observed
            if point.progress > policy.late_progress_threshold
        )
        if late
        else observed
    )
    if not selected:
        raise ValueError("stage-two policy excluded every observation")
    if helper:
        anchor = EvaluationPoint(
            size_id="__helper__",
            coordinates=selected[0].coordinates,
            progress=policy.complete_progress,
            loss=policy.helper_loss,
            score=policy.helper_score,
        )
        selected = (*selected, anchor)
    return StageTwoSelection(
        points=selected,
        observed_count=len(observed),
        selected_observed_count=len(selected) - int(helper),
        helper_count=int(helper),
    )


def fit_scaling_law(
    final_losses: Iterable[FinalLoss],
    evaluations: Iterable[EvaluationPoint],
    *,
    variant: ScalingVariant,
    policy: ScalingPolicy = DEFAULT_SCALING_POLICY,
) -> ScalingFit:
    """Fit one paper variant to a pre-grouped recipe/task prediction cell."""
    if not isinstance(variant, ScalingVariant):
        raise TypeError(f"variant must be a ScalingVariant: {variant!r}")
    _validate_policy(policy)
    losses = tuple(
        sorted(
            (_validated_final_loss(point) for point in final_losses),
            key=lambda point: point.size_id,
        )
    )
    points = tuple(
        sorted(
            (_validated_evaluation(point) for point in evaluations), key=_evaluation_key
        )
    )
    if not points:
        raise ValueError("at least one evaluation point is required")
    _validate_unique_size_progress(points)

    if variant is ScalingVariant.THREE_PARAMETER_SINGLE_STEP:
        combined = _fit_single_compute(points, variant=variant, policy=policy)
        return ScalingFit(variant, None, None, combined, len(points), 0)
    if variant is ScalingVariant.FIVE_PARAMETER_ND_SINGLE_STEP:
        combined = _fit_single_nd(points, variant=variant, policy=policy)
        return ScalingFit(variant, None, None, combined, len(points), 0)
    if not losses:
        raise ValueError("two-stage variants require final losses")
    if len({point.size_id for point in losses}) != len(losses):
        raise ValueError("final losses must contain exactly one row per size")

    if variant is ScalingVariant.TWO_PARAMETER_TWO_STAGE:
        stage_one = _fit_loss_compute(
            losses, include_e=False, variant=variant, policy=policy
        )
    elif variant is ScalingVariant.FIVE_PARAMETER_ND_TWO_STAGE:
        stage_one = _fit_loss_nd(losses, variant=variant, policy=policy)
    else:
        stage_one = _fit_loss_compute(
            losses, include_e=True, variant=variant, policy=policy
        )
    selection = select_stage_two_points(points, variant=variant, policy=policy)
    stage_two = _fit_loss_to_score(selection.points, variant=variant, policy=policy)
    return ScalingFit(
        variant=variant,
        stage_one=stage_one,
        stage_two=stage_two,
        combined=None,
        selected_stage_two_count=selection.selected_observed_count,
        helper_count=selection.helper_count,
    )


def predict_scaling_fit(fit: ScalingFit, coordinates: ScalingCoordinates) -> float:
    coordinates = _validated_coordinates(coordinates)
    if fit.combined is not None:
        values = fit.combined.values
        if fit.variant is ScalingVariant.THREE_PARAMETER_SINGLE_STEP:
            a, alpha, offset, score_a, k, score_b = values
            latent_loss = a / coordinates.compute**alpha + offset
            prediction = score_a * _scalar_expit(k * latent_loss) + score_b
        else:
            a, alpha, b, beta, e, score_a, score_b = values
            prediction = accuracy_tokens_parameters_single_5p(
                coordinates.token_count,
                coordinates.parameter_count,
                a,
                alpha,
                b,
                beta,
                e,
                score_a,
                score_b,
            )
    else:
        assert fit.stage_one is not None and fit.stage_two is not None
        stage_one = fit.stage_one.values
        if fit.variant is ScalingVariant.TWO_PARAMETER_TWO_STAGE:
            predicted_loss = loss_compute_2p(coordinates.compute, *stage_one)
        elif fit.variant is ScalingVariant.FIVE_PARAMETER_ND_TWO_STAGE:
            predicted_loss = loss_tokens_parameters_5p(
                coordinates.token_count, coordinates.parameter_count, *stage_one
            )
        else:
            predicted_loss = loss_compute_3p(coordinates.compute, *stage_one)
        prediction = accuracy_from_loss(predicted_loss, *fit.stage_two.values)
    if not math.isfinite(prediction):
        raise ValueError("scaling-law prediction is not finite")
    return prediction


def held_out_prediction(
    final_losses: Iterable[FinalLoss],
    evaluations: Iterable[EvaluationPoint],
    *,
    target: ScalingTarget,
    variant: ScalingVariant,
    policy: ScalingPolicy = DEFAULT_SCALING_POLICY,
) -> HeldOutPrediction:
    _identifier(target.size_id, label="target size_id")
    _validated_coordinates(target.coordinates)
    actual = _finite(target.actual_score, label="target actual_score")
    losses = tuple(point for point in final_losses if point.size_id != target.size_id)
    points = tuple(point for point in evaluations if point.size_id != target.size_id)
    fit = fit_scaling_law(losses, points, variant=variant, policy=policy)
    predicted = predict_scaling_fit(fit, target.coordinates)
    errors = prediction_errors(
        (PredictionCell(target.size_id, predicted, actual),), policy=policy
    )
    cell = errors.cells[0]
    return HeldOutPrediction(
        target=target,
        predicted_score=predicted,
        absolute_error_percent=cell.absolute_error_percent,
        relative_error_percent=cell.relative_error_percent,
        fit=fit,
    )


def prediction_errors(
    cells: Iterable[PredictionCell],
    *,
    policy: ScalingPolicy = DEFAULT_SCALING_POLICY,
) -> PredictionErrorSummary:
    _validate_policy(policy)
    errors: list[CellError] = []
    seen: set[str] = set()
    for cell in cells:
        cell_id = _identifier(cell.cell_id, label="cell_id")
        if cell_id in seen:
            raise ValueError(f"duplicate prediction cell: {cell_id!r}")
        seen.add(cell_id)
        predicted = _finite(cell.predicted, label=f"predicted score for {cell_id!r}")
        actual = _finite(cell.actual, label=f"actual score for {cell_id!r}")
        denominator = abs(actual)
        if denominator <= policy.relative_error_floor:
            raise ValueError(
                f"relative-error denominator for {cell_id!r} is not above "
                f"the configured floor {policy.relative_error_floor}"
            )
        absolute = abs(predicted - actual) * 100
        errors.append(
            CellError(
                cell_id=cell_id,
                absolute_error_percent=absolute,
                relative_error_percent=absolute / denominator,
            )
        )
    if not errors:
        raise ValueError("at least one prediction cell is required")
    ordered = tuple(sorted(errors, key=lambda error: error.cell_id))
    return PredictionErrorSummary(
        cells=ordered,
        mean_absolute_error_percent=_mean(
            error.absolute_error_percent for error in ordered
        ),
        mean_relative_error_percent=_mean(
            error.relative_error_percent for error in ordered
        ),
    )


def construct_13_size_subsets(sizes: Iterable[str]) -> tuple[SizeSubset, ...]:
    ordered = tuple(_identifier(size, label="size") for size in sizes)
    if len(ordered) != 13:
        raise ValueError(f"exactly 13 ordered sizes are required, got {len(ordered)}")
    if len(set(ordered)) != len(ordered):
        raise ValueError("ordered sizes must be unique")
    prefixes = tuple(
        SizeSubset(f"prefix-{count:02d}", ordered[:count], "prefix")
        for count in range(3, 14)
    )
    suffixes = tuple(
        SizeSubset(f"suffix-drop-{dropped:02d}", ordered[dropped:], "suffix")
        for dropped in range(1, 11)
    )
    return (*prefixes, *suffixes)


def summed_compute(
    costs: Iterable[ScaleCompute], *, sizes: Iterable[str] | None = None
) -> float:
    by_size: dict[str, float] = {}
    for point in costs:
        size_id = _identifier(point.size_id, label="compute size_id")
        if size_id in by_size:
            raise ValueError(f"duplicate compute size: {size_id!r}")
        compute = _finite(point.compute, label=f"compute for {size_id!r}")
        if compute <= 0:
            raise ValueError(f"compute for {size_id!r} must be positive")
        by_size[size_id] = compute
    selected = (
        tuple(by_size)
        if sizes is None
        else tuple(_identifier(size, label="selected compute size") for size in sizes)
    )
    if not selected:
        raise ValueError("at least one compute size is required")
    if len(set(selected)) != len(selected):
        raise ValueError("selected compute sizes must be unique")
    missing = tuple(size for size in selected if size not in by_size)
    if missing:
        raise ValueError(f"missing compute for sizes: {missing!r}")
    return math.fsum(sorted(by_size[size] for size in selected))


def pair_decision_accuracy(
    target_ranks: Iterable[tuple[str, float]],
    predicted_scores: Iterable[tuple[str, float]],
) -> PairDecisionAccuracy:
    ranks = _unique_values(target_ranks, label="target rank")
    predictions = _unique_values(predicted_scores, label="predicted score")
    if ranks.keys() != predictions.keys():
        raise ValueError("target rank and prediction recipe universes differ")
    recipes = tuple(sorted(ranks))
    correct = denominator = target_ties = predicted_ties = 0
    total = 0
    for index, recipe_a in enumerate(recipes):
        for recipe_b in recipes[index + 1 :]:
            total += 1
            target_sign = _comparison_sign(ranks[recipe_b], ranks[recipe_a])
            predicted_sign = _comparison_sign(
                predictions[recipe_a], predictions[recipe_b]
            )
            if target_sign == 0:
                target_ties += 1
                continue
            denominator += 1
            if predicted_sign == 0:
                predicted_ties += 1
                continue
            correct += target_sign == predicted_sign
    if denominator == 0:
        raise ValueError("target ranks contain no comparable recipe pairs")
    return PairDecisionAccuracy(
        accuracy=correct / denominator,
        correct=correct,
        denominator=denominator,
        total_pairs=total,
        target_ties=target_ties,
        predicted_ties=predicted_ties,
    )


def stepwise_frontier(
    points: Iterable[AccuracyAtCompute],
) -> tuple[AccuracyAtCompute, ...]:
    ordered = sorted(
        (_validated_accuracy_point(point) for point in points),
        key=lambda point: (point.compute, -point.accuracy, point.point_id),
    )
    if not ordered:
        raise ValueError("at least one accuracy point is required")
    result: list[AccuracyAtCompute] = []
    best_accuracy = -math.inf
    index = 0
    while index < len(ordered):
        compute = ordered[index].compute
        same_compute: list[AccuracyAtCompute] = []
        while index < len(ordered) and ordered[index].compute == compute:
            same_compute.append(ordered[index])
            index += 1
        candidate = same_compute[0]
        if candidate.accuracy > best_accuracy:
            result.append(candidate)
            best_accuracy = candidate.accuracy
    return tuple(result)


def compare_stepwise_single_scale(
    multi_scale_points: Iterable[AccuracyAtCompute],
    single_scale_points: Iterable[AccuracyAtCompute],
) -> tuple[FrontierComparison, ...]:
    frontier = stepwise_frontier(single_scale_points)
    multi = tuple(
        sorted(
            (_validated_accuracy_point(point) for point in multi_scale_points),
            key=lambda point: (point.compute, point.point_id),
        )
    )
    result: list[FrontierComparison] = []
    for point in multi:
        eligible = tuple(
            candidate for candidate in frontier if candidate.compute <= point.compute
        )
        if not eligible:
            continue
        single = eligible[-1]
        result.append(
            FrontierComparison(
                multi_scale=point,
                single_scale=single,
                accuracy_difference=point.accuracy - single.accuracy,
            )
        )
    return tuple(result)


def adjacent_size_crossover_counts(
    rankings: Iterable[RankingAtSize],
) -> tuple[AdjacentCrossoverCount, ...]:
    ordered = tuple(rankings)
    if len(ordered) < 2:
        raise ValueError("at least two ordered size rankings are required")
    parsed = tuple(
        (
            _identifier(ranking.size_id, label="ranking size_id"),
            _unique_values(ranking.recipe_scores, label="recipe score"),
        )
        for ranking in ordered
    )
    if len({size_id for size_id, _ in parsed}) != len(parsed):
        raise ValueError("ranking size IDs must be unique")
    universe = parsed[0][1].keys()
    if any(scores.keys() != universe for _, scores in parsed[1:]):
        raise ValueError("adjacent rankings have different recipe universes")
    recipes = tuple(sorted(universe))
    result: list[AdjacentCrossoverCount] = []
    for (small_id, small), (large_id, large) in zip(parsed, parsed[1:]):
        crossovers = comparable = ties = 0
        for index, recipe_a in enumerate(recipes):
            for recipe_b in recipes[index + 1 :]:
                small_sign = _comparison_sign(small[recipe_a], small[recipe_b])
                large_sign = _comparison_sign(large[recipe_a], large[recipe_b])
                if small_sign == 0 or large_sign == 0:
                    ties += 1
                    continue
                comparable += 1
                crossovers += small_sign != large_sign
        result.append(
            AdjacentCrossoverCount(
                smaller_size_id=small_id,
                larger_size_id=large_id,
                crossover_count=crossovers,
                comparable_pairs=comparable,
                tied_pairs=ties,
            )
        )
    return tuple(result)


def _fit_loss_compute(
    points: Sequence[FinalLoss],
    *,
    include_e: bool,
    variant: ScalingVariant,
    policy: ScalingPolicy,
) -> ParameterFit:
    compute = np.asarray([point.coordinates.compute for point in points], dtype=float)
    response = np.asarray([point.loss for point in points], dtype=float)
    reference = _geometric_mean_or_failure(
        compute,
        variant=variant,
        stage="stage_one",
        parameter_count=3 if include_e else 2,
    )
    x = compute / reference
    scale = _response_scale(response)
    minimum = float(np.min(response))
    maximum = float(np.max(response))
    if include_e:
        names = ("A", "alpha", "E")
        lower = np.asarray([1e-12, 1e-4, minimum - 2 * scale])
        upper = np.asarray([20 * scale, 5.0, maximum + 2 * scale])
        starts = tuple(
            np.asarray([a_fraction * scale, alpha, minimum - 0.05 * scale])
            for a_fraction in (0.25, 1.0, 4.0)
            for alpha in (0.25, 0.75, 1.5)
        )

        def model(parameters: np.ndarray) -> np.ndarray:
            a, alpha, e = parameters
            return a / x**alpha + e

    else:
        names = ("A", "alpha")
        lower = np.asarray([1e-12, 1e-4])
        upper = np.asarray([20 * max(float(np.max(np.abs(response))), scale), 5.0])
        starts = tuple(
            np.asarray([a_fraction * max(float(np.median(response)), scale), alpha])
            for a_fraction in (0.25, 1.0, 4.0)
            for alpha in (0.25, 0.75, 1.5)
        )

        def model(parameters: np.ndarray) -> np.ndarray:
            a, alpha = parameters
            return a / x**alpha

    fitted = _least_squares_fit(
        model,
        response,
        names=names,
        lower=lower,
        upper=upper,
        starts=starts,
        variant=variant,
        stage="stage_one",
        policy=policy.optimizer,
    )
    values = list(fitted.values)
    values[0] *= reference ** values[1]
    return ParameterFit(fitted.names, tuple(values), fitted.diagnostics)


def _fit_loss_nd(
    points: Sequence[FinalLoss],
    *,
    variant: ScalingVariant,
    policy: ScalingPolicy,
) -> ParameterFit:
    tokens = np.asarray(
        [point.coordinates.token_count for point in points], dtype=float
    )
    parameters = np.asarray(
        [point.coordinates.parameter_count for point in points], dtype=float
    )
    response = np.asarray([point.loss for point in points], dtype=float)
    token_reference = _geometric_mean_or_failure(
        tokens, variant=variant, stage="stage_one", parameter_count=5
    )
    parameter_reference = _geometric_mean_or_failure(
        parameters, variant=variant, stage="stage_one", parameter_count=5
    )
    n = tokens / token_reference
    d = parameters / parameter_reference
    scale = _response_scale(response)
    minimum = float(np.min(response))
    maximum = float(np.max(response))
    names = ("A", "alpha", "B", "beta", "E")
    lower = np.asarray([1e-12, 1e-4, 1e-12, 1e-4, minimum - 2 * scale])
    upper = np.asarray([20 * scale, 5.0, 20 * scale, 5.0, maximum + 2 * scale])
    starts = tuple(
        np.asarray(
            [split * scale, alpha, (1 - split) * scale, beta, minimum - 0.05 * scale]
        )
        for split in (0.25, 0.5, 0.75)
        for alpha, beta in ((0.25, 0.75), (0.75, 0.25), (0.75, 0.75), (1.5, 1.5))
    )

    def model(values: np.ndarray) -> np.ndarray:
        a, alpha, b, beta, e = values
        return a / n**alpha + b / d**beta + e

    fitted = _least_squares_fit(
        model,
        response,
        names=names,
        lower=lower,
        upper=upper,
        starts=starts,
        variant=variant,
        stage="stage_one",
        policy=policy.optimizer,
    )
    a, alpha, b, beta, e = fitted.values
    values = (
        a * token_reference**alpha,
        alpha,
        b * parameter_reference**beta,
        beta,
        e,
    )
    return ParameterFit(fitted.names, values, fitted.diagnostics)


def _fit_loss_to_score(
    points: Sequence[EvaluationPoint],
    *,
    variant: ScalingVariant,
    policy: ScalingPolicy,
) -> ParameterFit:
    loss = np.asarray([point.loss for point in points], dtype=float)
    response = np.asarray([point.score for point in points], dtype=float)
    loss_scale = _response_scale(loss)
    score_scale = _response_scale(response)
    minimum = float(np.min(response))
    maximum = float(np.max(response))
    names = ("a", "k", "L0", "b")
    lower = np.asarray(
        [
            1e-10,
            -100 / loss_scale,
            float(np.min(loss)) - 10 * loss_scale,
            minimum - 2 * score_scale,
        ]
    )
    upper = np.asarray(
        [
            2 + 4 * score_scale,
            -1e-10,
            float(np.max(loss)) + 10 * loss_scale,
            maximum + 2 * score_scale,
        ]
    )
    starts = tuple(
        np.asarray(
            [
                max(score_scale * amplitude, 0.1),
                -slope / loss_scale,
                midpoint,
                minimum - 0.05 * score_scale,
            ]
        )
        for amplitude in (0.5, 1.0)
        for slope in (0.5, 2.0, 8.0)
        for midpoint in (
            float(np.quantile(loss, 0.25)),
            float(np.median(loss)),
            float(np.quantile(loss, 0.75)),
        )
    )

    def model(values: np.ndarray) -> np.ndarray:
        a, k, midpoint, b = values
        return a * expit(k * (loss - midpoint)) + b

    return _least_squares_fit(
        model,
        response,
        names=names,
        lower=lower,
        upper=upper,
        starts=starts,
        variant=variant,
        stage="stage_two",
        policy=policy.optimizer,
    )


def _fit_single_compute(
    points: Sequence[EvaluationPoint],
    *,
    variant: ScalingVariant,
    policy: ScalingPolicy,
) -> ParameterFit:
    compute = np.asarray([point.coordinates.compute for point in points], dtype=float)
    response = np.asarray([point.score for point in points], dtype=float)
    reference = _geometric_mean_or_failure(
        compute, variant=variant, stage="combined", parameter_count=6
    )
    x = compute / reference
    score_scale = _response_scale(response)
    minimum = float(np.min(response))
    maximum = float(np.max(response))
    names = ("A", "alpha", "E_minus_L0", "a", "k", "b")
    lower = np.asarray([1e-12, 1e-4, -20.0, 1e-10, -100.0, minimum - 2 * score_scale])
    upper = np.asarray(
        [20.0, 5.0, 20.0, 2 + 4 * score_scale, -1e-10, maximum + 2 * score_scale]
    )
    starts = tuple(
        np.asarray(
            [
                a,
                alpha,
                offset,
                max(score_scale, 0.1),
                -slope,
                minimum - 0.05 * score_scale,
            ]
        )
        for a in (0.25, 1.0, 4.0)
        for alpha in (0.25, 0.75)
        for offset, slope in ((-1.0, 1.0), (0.0, 2.0), (1.0, 8.0))
    )

    def model(values: np.ndarray) -> np.ndarray:
        a, alpha, offset, score_a, k, score_b = values
        return score_a * expit(k * (a / x**alpha + offset)) + score_b

    fitted = _least_squares_fit(
        model,
        response,
        names=names,
        lower=lower,
        upper=upper,
        starts=starts,
        variant=variant,
        stage="combined",
        policy=policy.optimizer,
    )
    a, alpha, offset, score_a, k, score_b = fitted.values
    return ParameterFit(
        fitted.names,
        (a * reference**alpha, alpha, offset, score_a, k, score_b),
        fitted.diagnostics,
    )


def _fit_single_nd(
    points: Sequence[EvaluationPoint],
    *,
    variant: ScalingVariant,
    policy: ScalingPolicy,
) -> ParameterFit:
    tokens = np.asarray(
        [point.coordinates.token_count for point in points], dtype=float
    )
    parameters = np.asarray(
        [point.coordinates.parameter_count for point in points], dtype=float
    )
    response = np.asarray([point.score for point in points], dtype=float)
    token_reference = _geometric_mean_or_failure(
        tokens, variant=variant, stage="combined", parameter_count=7
    )
    parameter_reference = _geometric_mean_or_failure(
        parameters, variant=variant, stage="combined", parameter_count=7
    )
    n = tokens / token_reference
    d = parameters / parameter_reference
    score_scale = _response_scale(response)
    minimum = float(np.min(response))
    maximum = float(np.max(response))
    names = ("A", "alpha", "B", "beta", "E", "a", "b")
    lower = np.asarray(
        [-20.0, 1e-4, -20.0, 1e-4, -20.0, 1e-10, minimum - 2 * score_scale]
    )
    upper = np.asarray(
        [20.0, 5.0, 20.0, 5.0, 20.0, 2 + 4 * score_scale, maximum + 2 * score_scale]
    )
    starts = tuple(
        np.asarray(
            [
                -split,
                alpha,
                -(1 - split),
                beta,
                offset,
                max(score_scale, 0.1),
                minimum - 0.05 * score_scale,
            ]
        )
        for split in (0.25, 0.5, 0.75)
        for alpha, beta in ((0.25, 0.75), (0.75, 0.25), (0.75, 0.75))
        for offset in (-1.0, 0.0, 1.0)
    )

    def model(values: np.ndarray) -> np.ndarray:
        a, alpha, b, beta, e, score_a, score_b = values
        latent = a / n**alpha + b / d**beta + e
        return score_a * expit(latent) + score_b

    fitted = _least_squares_fit(
        model,
        response,
        names=names,
        lower=lower,
        upper=upper,
        starts=starts,
        variant=variant,
        stage="combined",
        policy=policy.optimizer,
    )
    a, alpha, b, beta, e, score_a, score_b = fitted.values
    return ParameterFit(
        fitted.names,
        (
            a * token_reference**alpha,
            alpha,
            b * parameter_reference**beta,
            beta,
            e,
            score_a,
            score_b,
        ),
        fitted.diagnostics,
    )


def _least_squares_fit(
    model: Callable[[np.ndarray], np.ndarray],
    response: np.ndarray,
    *,
    names: tuple[str, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    starts: Sequence[np.ndarray],
    variant: ScalingVariant,
    stage: str,
    policy: OptimizerPolicy,
) -> ParameterFit:
    observation_count = len(response)
    parameter_count = len(names)
    if observation_count < parameter_count:
        _raise_fit_failure(
            FitFailureReason.INSUFFICIENT_OBSERVATIONS,
            variant,
            stage,
            observation_count,
            parameter_count,
            f"{stage} requires at least {parameter_count} observations; got {observation_count}",
        )
    response_scale = _response_scale(response)
    candidates = []
    for start in starts:
        clipped = np.clip(start, lower + 1e-14, upper - 1e-14)
        result = least_squares(
            lambda values: (model(values) - response) / response_scale,
            clipped,
            bounds=(lower, upper),
            ftol=policy.ftol,
            xtol=policy.xtol,
            gtol=policy.gtol,
            max_nfev=policy.max_nfev,
            method="trf",
            x_scale="jac",
        )
        if (
            result.success
            and math.isfinite(float(result.cost))
            and np.all(np.isfinite(result.x))
        ):
            candidates.append(result)
    if not candidates:
        _raise_fit_failure(
            FitFailureReason.OPTIMIZER_FAILED,
            variant,
            stage,
            observation_count,
            parameter_count,
            f"every deterministic optimizer start failed for {stage}",
        )
    minimum_cost = min(float(candidate.cost) for candidate in candidates)
    tied = tuple(
        candidate
        for candidate in candidates
        if float(candidate.cost) <= minimum_cost + policy.cost_tie_tolerance
    )
    best = min(tied, key=lambda candidate: tuple(float(value) for value in candidate.x))
    singular_values = np.linalg.svd(best.jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(best.jac))
    condition = (
        math.inf
        if not len(singular_values) or singular_values[-1] == 0
        else float(singular_values[0] / singular_values[-1])
    )
    if policy.fail_on_ill_conditioned and (
        rank < parameter_count or condition > policy.max_condition_number
    ):
        _raise_fit_failure(
            FitFailureReason.ILL_CONDITIONED,
            variant,
            stage,
            observation_count,
            parameter_count,
            f"{stage} fit is ill-conditioned: rank={rank}/{parameter_count}, condition={condition:.6g}",
        )
    values = tuple(float(value) for value in best.x)
    if not all(math.isfinite(value) for value in values):
        _raise_fit_failure(
            FitFailureReason.NON_FINITE_RESULT,
            variant,
            stage,
            observation_count,
            parameter_count,
            f"{stage} fit produced non-finite parameters",
        )
    return ParameterFit(
        names=names,
        values=values,
        diagnostics=FitDiagnostics(
            stage=stage,
            observation_count=observation_count,
            parameter_count=parameter_count,
            attempted_starts=len(starts),
            successful_starts=len(candidates),
            cost=float(best.cost),
            optimality=float(best.optimality),
            function_evaluations=int(best.nfev),
            jacobian_rank=rank,
            condition_number=condition,
        ),
    )


def _raise_fit_failure(
    reason: FitFailureReason,
    variant: ScalingVariant,
    stage: str,
    observation_count: int,
    parameter_count: int,
    message: str,
) -> Never:
    raise ScalingFitError(
        FitFailure(reason, variant, stage, observation_count, parameter_count, message)
    )


def _response_scale(values: np.ndarray) -> float:
    span = float(np.max(values) - np.min(values))
    if span > 1e-12:
        return span
    magnitude = float(np.max(np.abs(values)))
    return max(magnitude, 1.0)


def _geometric_mean(values: np.ndarray) -> float:
    if len(set(float(value) for value in values)) < 2:
        raise ValueError("scaling predictor must contain at least two distinct values")
    return float(np.exp(np.mean(np.log(values))))


def _geometric_mean_or_failure(
    values: np.ndarray,
    *,
    variant: ScalingVariant,
    stage: str,
    parameter_count: int,
) -> float:
    try:
        return _geometric_mean(values)
    except ValueError as error:
        _raise_fit_failure(
            FitFailureReason.DEGENERATE_PREDICTOR,
            variant,
            stage,
            len(values),
            parameter_count,
            f"{stage} predictor is degenerate: {error}",
        )


def _validated_coordinates(value: ScalingCoordinates) -> ScalingCoordinates:
    parameters = _finite(value.parameter_count, label="parameter_count")
    tokens = _finite(value.token_count, label="token_count")
    compute = _finite(value.compute, label="compute")
    if parameters <= 0 or tokens <= 0 or compute <= 0:
        raise ValueError("parameter_count, token_count, and compute must be positive")
    return ScalingCoordinates(parameters, tokens, compute)


def _validate_checkpoint_loss(point: CheckpointLoss) -> None:
    _identifier(point.size_id, label="checkpoint size_id")
    _validated_coordinates(point.coordinates)
    _validate_progress(point.progress)
    _finite(point.loss, label="checkpoint loss")


def _validated_final_loss(point: FinalLoss) -> FinalLoss:
    _identifier(point.size_id, label="final-loss size_id")
    coordinates = _validated_coordinates(point.coordinates)
    loss = _finite(point.loss, label="final loss")
    if (
        isinstance(point.averaged_checkpoint_count, bool)
        or not isinstance(point.averaged_checkpoint_count, int)
        or point.averaged_checkpoint_count < 1
    ):
        raise ValueError("averaged_checkpoint_count must be a positive integer")
    return FinalLoss(point.size_id, coordinates, loss, point.averaged_checkpoint_count)


def _validated_evaluation(point: EvaluationPoint) -> EvaluationPoint:
    _identifier(point.size_id, label="evaluation size_id")
    coordinates = _validated_coordinates(point.coordinates)
    progress = _validate_progress(point.progress)
    loss = _finite(point.loss, label="evaluation loss")
    score = _finite(point.score, label="evaluation score")
    return EvaluationPoint(point.size_id, coordinates, progress, loss, score)


def _evaluation_key(point: EvaluationPoint) -> tuple[str, float, float, float]:
    return (point.size_id, point.progress, point.loss, point.score)


def _validate_progress(value: float) -> float:
    progress = _finite(value, label="progress")
    if not 0 <= progress <= 1:
        raise ValueError("progress must be between zero and one")
    return progress


def _validate_policy(policy: ScalingPolicy) -> None:
    _validate_optimizer_policy(policy.optimizer)
    if not 0 < policy.final_window_fraction <= 1:
        raise ValueError("final_window_fraction must be in (0, 1]")
    if not 0 <= policy.complete_progress <= 1:
        raise ValueError("complete_progress must be between zero and one")
    if policy.progress_tolerance < 0:
        raise ValueError("progress_tolerance must be non-negative")
    if not 0 <= policy.late_progress_threshold < 1:
        raise ValueError("late_progress_threshold must be in [0, 1)")
    if policy.relative_error_floor < 0:
        raise ValueError("relative_error_floor must be non-negative")
    _finite(policy.helper_loss, label="helper_loss")
    _finite(policy.helper_score, label="helper_score")


def _validate_optimizer_policy(policy: OptimizerPolicy) -> None:
    machine_epsilon = float(np.finfo(float).eps)
    for label, tolerance in (
        ("ftol", policy.ftol),
        ("xtol", policy.xtol),
        ("gtol", policy.gtol),
    ):
        value = _finite(tolerance, label=label)
        if value <= machine_epsilon:
            raise ValueError(f"{label} must be greater than machine epsilon")
    if isinstance(policy.max_nfev, bool) or not isinstance(policy.max_nfev, int):
        raise TypeError("max_nfev must be an integer")
    if policy.max_nfev < 1:
        raise ValueError("max_nfev must be positive")
    if _finite(policy.cost_tie_tolerance, label="cost_tie_tolerance") < 0:
        raise ValueError("cost_tie_tolerance must be non-negative")
    if _finite(policy.max_condition_number, label="max_condition_number") <= 0:
        raise ValueError("max_condition_number must be positive")


def _validate_unique_size_progress(points: Sequence[EvaluationPoint]) -> None:
    keys = tuple((point.size_id, point.progress) for point in points)
    if len(set(keys)) != len(keys):
        raise ValueError("evaluations contain duplicate size/progress rows")


def _validated_accuracy_point(point: AccuracyAtCompute) -> AccuracyAtCompute:
    point_id = _identifier(point.point_id, label="accuracy point_id")
    compute = _finite(point.compute, label=f"compute for {point_id!r}")
    accuracy = _finite(point.accuracy, label=f"accuracy for {point_id!r}")
    if compute <= 0:
        raise ValueError("accuracy-point compute must be positive")
    return AccuracyAtCompute(point_id, compute, accuracy)


def _unique_values(
    values: Iterable[tuple[str, float]], *, label: str
) -> dict[str, float]:
    result: dict[str, float] = {}
    for raw_name, raw_value in values:
        name = _identifier(raw_name, label=label)
        if name in result:
            raise ValueError(f"duplicate {label}: {name!r}")
        result[name] = _finite(raw_value, label=f"{label} for {name!r}")
    if len(result) < 2:
        raise ValueError(f"at least two {label} values are required")
    return result


def _mean(values: Iterable[float]) -> float:
    ordered = tuple(sorted(values))
    if not ordered:
        raise ValueError("at least one value is required")
    return math.fsum(ordered) / len(ordered)


def _finite(value: float | int, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite: {value!r}")
    return result


def _identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string: {value!r}")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _comparison_sign(left: float, right: float) -> int:
    return (left > right) - (left < right)


def _scalar_expit(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)

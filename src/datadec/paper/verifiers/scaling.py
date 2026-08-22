from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq

from datadec.config import DataDecideCatalog, OLMESContract, ScalingLawContract
from datadec.paper.contracts import load_toml_model
from datadec.paper.models import (
    AnalysisId,
    AttemptResult,
    AttemptRole,
    AttemptSpec,
    AxisScale,
    AxisSpec,
    CheckpointSelection,
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
from datadec.paper.single_scale import analyze_prediction_checkpoint
from datadec.paper.verifiers import single_scale as single_adapter

_CHEAP = "cheap_decisions"
_OLMES = "olmes_aggregate"
_TASK = "olmes_10_macro_avg"
_METRIC = "primary_metric"
_TARGET_SIZE = "1B"
_CHEAP_COLUMNS = (
    "task",
    "mix",
    "metric",
    "setup",
    "step_1_y",
    "step_2_y",
    "stacked_y",
    "step_1_pred",
    "step_2_pred",
    "stacked_pred",
    "abs_error_step_1",
    "abs_error_step_2",
    "abs_error_stacked",
    "rel_error_stacked",
)
_KEYS = ("task", "mix", "metric", "setup")
_FAMILIES = (
    "3_param-helper_points-step2=0.5",
    "3_param-helper_points",
    "3_param-step2=0.5",
    "3_param",
    "2_param",
    "5_param-1_step-ai2",
    "3_param-1_step",
    "5_param-ai2",
)
_TARGET_FAMILIES = _FAMILIES[:5]
_ERROR_CLAIMS = {
    "DD-0301": (_FAMILIES[0], 5.6, 2.6),
    "DD-0302": (_FAMILIES[1], 6.0, 2.8),
    "DD-0303": (_FAMILIES[2], 5.9, 2.9),
    "DD-0304": (_FAMILIES[3], 6.5, 3.1),
    "DD-0305": (_FAMILIES[4], 6.5, 3.2),
    "DD-0306": (_FAMILIES[5], 42.8, 17.4),
    "DD-0307": (_FAMILIES[6], 42.9, 42.3),
    "DD-0308": (_FAMILIES[7], 230.8, 65.4),
}
_FRONTIER_CLAIMS = {"DD-0013", "DD-0054", "DD-0180", "DD-0181", "DD-0192", "DD-0368"}
_SERIES = {
    "DD-0180": "dd-0180-paper-analog",
    "DD-0368": "dd-0368-paper-analog",
    "DD-0369": "dd-0369-paper-analog",
}


@dataclass(frozen=True, slots=True)
class _Setup:
    name: str
    family: str
    sizes: tuple[str, ...]
    subset: str
    kind: str


@dataclass(frozen=True, slots=True)
class _Pair:
    accuracy: float
    denominator: int
    target_ties: int
    predicted_ties: int


@dataclass(frozen=True, slots=True)
class _Point:
    setup: _Setup
    compute: float
    percent_compute: float
    pair: _Pair
    frontier_accuracy: float | None = None
    frontier_difference: float | None = None


@dataclass(frozen=True, slots=True)
class _Error:
    rows: int
    absolute_percent: float
    released_relative_percent: float
    paper_formula_relative_percent: float


@dataclass(frozen=True, slots=True)
class _Frontier:
    points: tuple[tuple[float, float], ...]
    selections: tuple[RowSelection, ...]
    checkpoints: tuple[CheckpointSelection, ...]
    maximum_accuracy: float | None
    missing: tuple[str, ...]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float:
    ordered = tuple(sorted(float(value) for value in values))
    if not ordered:
        raise ValueError("at least one value is required")
    return math.fsum(ordered) / len(ordered)


def _key_sha(frame: pd.DataFrame, columns: tuple[str, ...] = _KEYS) -> str:
    keys = sorted(
        tuple(
            int(value) if column == "step" else str(value)
            for column, value in zip(columns, row, strict=True)
        )
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None)
    )
    return hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest()


def _spec(contract: PaperValidationContract, table_id: str) -> InputTableSpec:
    try:
        return next(item for item in contract.inputs if item.id == table_id)
    except StopIteration as error:
        raise ValueError(f"missing configured input {table_id}") from error


def _identity(
    spec: InputTableSpec, digest: str, identities: Mapping[str, ContentIdentity]
) -> None:
    candidates = tuple(
        value
        for key, value in identities.items()
        if key in {spec.id, spec.path} or value.id in {spec.id, spec.path}
    )
    if any(value.sha256 != digest for value in candidates):
        raise ValueError(f"{spec.id} changed after input identity capture")


def _read(
    path: Path, spec: InputTableSpec, identities: Mapping[str, ContentIdentity]
) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {spec.id} input: {path}")
    schema = tuple(pq.ParquetFile(path).schema_arrow.names)
    if schema != spec.columns:
        raise ValueError(
            f"{spec.id} schema differs: expected={spec.columns!r}, actual={schema!r}"
        )
    digest = _sha(path)
    _identity(spec, digest, identities)
    frame = pd.read_parquet(path, columns=list(spec.columns))
    if _sha(path) != digest:
        raise RuntimeError(f"{spec.id} changed while being read")
    return frame, digest


def _parse_setup(value: str, sizes: tuple[str, ...]) -> _Setup | None:
    family = next(
        (
            item
            for item in _FAMILIES
            if value == item or value.startswith(f"{item}-no_")
        ),
        None,
    )
    if family is None:
        if value.startswith("3_param-intermediate"):
            return None
        raise ValueError(f"unexpected setup {value!r}")
    suffix = value[len(family) :]
    excluded = () if not suffix else tuple(suffix.removeprefix("-no_").split("_no_"))
    if len(excluded) != len(set(excluded)) or set(excluded) - set(sizes):
        raise ValueError(f"invalid setup exclusions {value!r}")
    included = tuple(size for size in sizes if size not in excluded)
    if included == sizes[: len(included)] and 3 <= len(included) <= 13:
        subset, kind = f"prefix-{len(included):02d}", "prefix"
    else:
        dropped = len(sizes) - len(included)
        if not 1 <= dropped <= 10 or included != sizes[dropped:]:
            raise ValueError(f"unsupported setup subset {value!r}")
        subset, kind = f"suffix-drop-{dropped:02d}", "suffix"
    return _Setup(value, family, included, subset, kind)


def _expected_setups(sizes: tuple[str, ...]) -> tuple[_Setup, ...]:
    subsets = tuple(
        (sizes[:count], f"prefix-{count:02d}", "prefix") for count in range(3, 14)
    ) + tuple(
        (sizes[dropped:], f"suffix-drop-{dropped:02d}", "suffix")
        for dropped in range(1, 11)
    )
    values = []
    for family in _FAMILIES:
        for included, subset, kind in subsets:
            excluded = tuple(size for size in sizes if size not in included)
            name = family + ("" if not excluded else "-no_" + "_no_".join(excluded))
            values.append(_Setup(name, family, included, subset, kind))
    return tuple(values)


def _pair(
    target: Mapping[str, float], predicted: Mapping[str, float], credit: float = 0.0
) -> _Pair:
    if target.keys() != predicted.keys() or not 0 <= credit <= 1:
        raise ValueError("invalid pairwise decision universe or tie credit")
    correct = 0.0
    denominator = target_ties = predicted_ties = 0
    for left, right in combinations(sorted(target), 2):
        target_delta = target[left] - target[right]
        predicted_delta = predicted[left] - predicted[right]
        target_sign = (target_delta > 0) - (target_delta < 0)
        predicted_sign = (predicted_delta > 0) - (predicted_delta < 0)
        if target_sign == 0:
            target_ties += 1
            continue
        denominator += 1
        if predicted_sign == 0:
            predicted_ties += 1
            correct += credit
        elif predicted_sign == target_sign:
            correct += 1
    if denominator == 0:
        raise ValueError("target ranking has no comparable pairs")
    return _Pair(correct / denominator, denominator, target_ties, predicted_ties)


def _catalog_compute(catalog: DataDecideCatalog) -> tuple[dict[str, float], float]:
    length = catalog.model_defaults.length_str.upper()
    if not length.endswith("XC") or not length[:-2].isdigit():
        raise ValueError("catalog length must be integer xC")
    multiplier = int(length[:-2]) * catalog.training.token_length_multiplier
    costs = {
        model.name: float(
            catalog.training.flops_per_token_per_parameter
            * model.exact_parameter_count
            * multiplier
            * model.training_parameter_count
        )
        for model in catalog.models
    }
    return costs, costs[_TARGET_SIZE]


def _target(
    decision: pd.DataFrame, setups: tuple[_Setup, ...], mixes: tuple[str, ...]
) -> tuple[dict[str, float] | None, tuple[str, ...], float | None]:
    reference = None
    missing = []
    for setup in (item for item in setups if item.family in _TARGET_FAMILIES):
        rows = decision[decision.setup == setup.name].sort_values("mix")
        if tuple(rows.mix.astype(str)) != mixes or rows.stacked_y.isna().any():
            missing.append(f"target_ranking:setup={setup.name}")
            continue
        values = {str(row.mix): float(row.stacked_y) for row in rows.itertuples()}
        if reference is None:
            reference = values
        elif values != reference:
            missing.append(f"target_ranking_mismatch:setup={setup.name}")
    difference = None
    if reference is not None:
        rows = decision[decision.setup == "5_param-ai2"].sort_values("mix")
        if tuple(rows.mix.astype(str)) == mixes:
            difference = max(
                abs(reference[str(row.mix)] - float(row.stacked_y))
                for row in rows.itertuples()
            )
    return reference, tuple(sorted(missing)), difference


def _points(
    decision: pd.DataFrame,
    setups: tuple[_Setup, ...],
    mixes: tuple[str, ...],
    target: Mapping[str, float] | None,
    costs: Mapping[str, float],
    target_compute: float,
    credit: float,
) -> tuple[tuple[_Point, ...], tuple[str, ...]]:
    if target is None:
        return (), ("target_ranking:missing_common_target",)
    points, missing = [], []
    for setup in setups:
        rows = decision[decision.setup == setup.name].sort_values("mix")
        if (
            tuple(rows.mix.astype(str)) != mixes
            or rows[["stacked_y", "stacked_pred"]].isna().any().any()
        ):
            missing.append(f"decision:setup={setup.name}")
            continue
        predicted = {str(row.mix): float(row.stacked_pred) for row in rows.itertuples()}
        compute = math.fsum(costs[size] for size in setup.sizes)
        points.append(
            _Point(
                setup,
                compute,
                compute / target_compute * 100,
                _pair(target, predicted, credit),
            )
        )
    return tuple(points), tuple(sorted(missing))


def _error(
    frame: pd.DataFrame, family: str, tasks: tuple[str, ...], mixes: tuple[str, ...]
) -> tuple[_Error | None, tuple[str, ...], pd.DataFrame]:
    rows = frame[(frame.setup == family) & (frame.metric == _METRIC)].sort_values(
        ["task", "mix"]
    )
    expected = {(task, mix) for task in tasks for mix in mixes}
    if {(str(row.task), str(row.mix)) for row in rows.itertuples()} != expected or len(
        rows
    ) != len(expected):
        return None, (f"prediction_error:setup={family}",), rows
    if (
        rows[["stacked_y", "stacked_pred", "abs_error_stacked", "rel_error_stacked"]]
        .isna()
        .any()
        .any()
    ):
        return None, (f"prediction_error_nonfinite:setup={family}",), rows
    actual = tuple(float(value) for value in rows.stacked_y)
    predicted = tuple(float(value) for value in rows.stacked_pred)
    if any(value == 0 for value in (*actual, *predicted)):
        return None, (f"prediction_error_zero_denominator:setup={family}",), rows
    absolute = tuple(abs(a - p) for a, p in zip(actual, predicted, strict=True))
    released = tuple(
        error / abs(p) for error, p in zip(absolute, predicted, strict=True)
    )
    paper_formula_relative = tuple(
        error / a for error, a in zip(absolute, actual, strict=True)
    )
    if any(
        not math.isclose(value, expected_value, rel_tol=1e-12, abs_tol=1e-12)
        for value, expected_value in zip(rows.abs_error_stacked, absolute, strict=True)
    ):
        raise ValueError(f"absolute error column drift for {family}")
    if any(
        not math.isclose(value, expected_value, rel_tol=1e-12, abs_tol=1e-12)
        for value, expected_value in zip(rows.rel_error_stacked, released, strict=True)
    ):
        raise ValueError(f"released relative error denominator drift for {family}")
    return (
        _Error(
            len(rows),
            _mean(absolute) * 100,
            _mean(released) * 100,
            _mean(paper_formula_relative) * 100,
        ),
        (),
        rows,
    )


def _adjudicate_error(
    *, released_display_match: bool, paper_formula_match: bool
) -> ValidationOutcome:
    if not released_display_match:
        return ValidationOutcome.NOT_REPRODUCED
    if paper_formula_match:
        return ValidationOutcome.REPRODUCED
    return ValidationOutcome.DIRECTIONALLY_CONSISTENT


def _selection(
    frame: pd.DataFrame,
    spec: InputTableSpec,
    digest: str,
    predicates: tuple[RowPredicate, ...],
) -> RowSelection:
    return RowSelection(
        logical_table_id=_CHEAP,
        columns=spec.columns,
        predicates=predicates,
        local_parquet_sha256=digest,
        selected_row_count=len(frame),
        selected_key_sha256=_key_sha(frame),
    )


def _frontier(
    data_root: Path,
    contract: PaperValidationContract,
    identities: Mapping[str, ContentIdentity],
    target_scores: Mapping[str, float],
) -> tuple[_Frontier, str]:
    spec = _spec(contract, _OLMES)
    path = data_root / spec.path
    digest = _sha(path)
    supplied = dict(identities)
    supplied.setdefault(_OLMES, ContentIdentity(id=_OLMES, sha256=digest))
    if not target_scores:
        _identity(spec, digest, identities)
        return (
            _Frontier(
                (),
                (),
                (),
                None,
                ("single_scale:missing=compatible_target_ranking",),
            ),
            digest,
        )
    observations, loaded_digest, columns = single_adapter._load_observations(
        data_root=data_root, contract=contract, input_identities=supplied
    )
    checkpoints = single_adapter._available_checkpoints(observations)
    missing = tuple(
        f"single_scale:size={size}|missing=common_complete_checkpoint"
        for size in single_adapter._MODEL_SIZE_ORDER
        if size not in checkpoints
    )
    if missing:
        return _Frontier((), (), (), None, missing), digest
    target = checkpoints[_TARGET_SIZE][-1]
    ranking = single_adapter.TargetRanking(
        model_size=_TARGET_SIZE,
        step=target.step,
        metric=_METRIC,
        seed_count=1,
        scores=single_adapter._ranked(target_scores),
    )
    selected = single_adapter._plot_checkpoints(checkpoints)
    points = []
    for checkpoint in selected:
        summary = analyze_prediction_checkpoint(
            checkpoint, ranking, target_compute=target.actual_compute
        ).summaries[0]
        points.append((summary.actual_compute, summary.mean_accuracy))
    selections = tuple(
        single_adapter._row_selection(
            tuple(item for item in selected if item.universe.model_size == size),
            columns=columns,
            parquet_sha256=loaded_digest,
        )
        for size in single_adapter._MODEL_SIZE_ORDER
    )
    checkpoint_selections = tuple(
        single_adapter._checkpoint_selection(
            item,
            requested_meaning="single-scale common-complete frontier point",
            rule=single_adapter.CheckpointRule.EXACT,
            contract=contract,
        )
        for item in selected
    )
    return _Frontier(
        tuple(points),
        selections,
        checkpoint_selections,
        max(value for _, value in points),
        (),
    ), digest


def _compare_frontier(
    points: tuple[_Point, ...], evidence: _Frontier
) -> tuple[_Point, ...]:
    ordered = sorted(evidence.points, key=lambda value: (value[0], -value[1]))
    frontier, best = [], -math.inf
    for compute, accuracy in ordered:
        if accuracy > best:
            frontier.append((compute, accuracy))
            best = accuracy
    result = []
    for point in points:
        eligible = tuple(value for value in frontier if value[0] <= point.compute)
        if not eligible:
            continue
        baseline = eligible[-1][1]
        result.append(
            _Point(
                point.setup,
                point.compute,
                point.percent_compute,
                point.pair,
                baseline,
                point.pair.accuracy - baseline,
            )
        )
    return tuple(result)


def _parameter(rule: ComparisonRule, name: ComparisonParameterName) -> float:
    return rule.parameter(name).default


def _sensitivities(
    attempt: AttemptSpec, rule: ComparisonRule
) -> tuple[tuple[str, ComparisonParameterName, float, ComparisonRule], ...]:
    values = []
    for parameter_index, parameter in enumerate(rule.parameters):
        for grid_index, value in enumerate(parameter.sensitivity_grid, 1):
            if value == parameter.default:
                continue
            identifier = f"{attempt.claim_id.lower()}-comparison-{parameter.name.value.replace('_', '-')}-grid-{grid_index}"
            if identifier not in attempt.sensitivity_ids:
                raise ValueError(f"undeclared sensitivity {identifier}")
            parameters = list(rule.parameters)
            parameters[parameter_index] = parameter.model_copy(
                update={"default": value}
            )
            values.append(
                (
                    identifier,
                    parameter.name,
                    value,
                    rule.model_copy(update={"parameters": tuple(parameters)}),
                )
            )
    if {value[0] for value in values} != set(attempt.sensitivity_ids):
        raise ValueError(f"unsupported sensitivity declaration for {attempt.id}")
    return tuple(values)


def _ranks(values: Mapping[str, float], descending: bool) -> dict[str, int]:
    return {
        name: 1
        + sum(
            other > value if descending else other < value for other in values.values()
        )
        for name, value in values.items()
    }


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return tuple(ranks)


def _spearman(xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    xr, yr = _average_ranks(xs), _average_ranks(ys)
    xm, ym = _mean(xr), _mean(yr)
    numerator = math.fsum((x - xm) * (y - ym) for x, y in zip(xr, yr, strict=True))
    denominator = math.sqrt(
        math.fsum((x - xm) ** 2 for x in xr) * math.fsum((y - ym) ** 2 for y in yr)
    )
    return numerator / denominator


def _plot(attempt: AttemptSpec, points: tuple[_Point, ...]) -> PlotSeries | None:
    identifier = _SERIES.get(attempt.claim_id)
    if identifier is None:
        return None
    if attempt.claim_id == "DD-0368":
        points = tuple(point for point in points if point.setup.family == "5_param-ai2")
    elif attempt.claim_id == "DD-0369":
        points = tuple(
            point for point in points if point.setup.family == "3_param-1_step"
        )
    if not points:
        return None
    uses_frontier = attempt.claim_id in {"DD-0180", "DD-0368"}
    measures = ("percent_target_compute", "decision_accuracy") + (
        ("single_scale_frontier_accuracy", "frontier_difference")
        if uses_frontier
        else ()
    )
    plot_points = []
    for point in sorted(
        points, key=lambda item: (item.compute, item.setup.family, item.setup.subset)
    ):
        point_measures = [
            MeasureValue(name="percent_target_compute", value=point.percent_compute),
            MeasureValue(name="decision_accuracy", value=point.pair.accuracy),
        ]
        if uses_frontier:
            if point.frontier_accuracy is None or point.frontier_difference is None:
                return None
            point_measures += [
                MeasureValue(
                    name="single_scale_frontier_accuracy", value=point.frontier_accuracy
                ),
                MeasureValue(
                    name="frontier_difference", value=point.frontier_difference
                ),
            ]
        plot_points.append(
            PlotPoint(
                dimensions=(
                    DimensionValue(name="setup_family", value=point.setup.family),
                    DimensionValue(name="subset", value=point.setup.subset),
                    DimensionValue(name="subset_kind", value=point.setup.kind),
                    DimensionValue(
                        name="included_sizes", value=",".join(point.setup.sizes)
                    ),
                ),
                measures=tuple(point_measures),
            )
        )
    return PlotSeries(
        id=identifier,
        figure="fig:all_scaling_laws_accuracy_vs_compute_shaded",
        panel="main",
        semantic_kind="author-derived scaling-law decision accuracy",
        x_axis=AxisSpec(
            measure="percent_target_compute", scale=AxisScale.LOG, unit="percent"
        ),
        y_axis=AxisSpec(
            measure="decision_accuracy", scale=AxisScale.LINEAR, unit="proportion"
        ),
        dimensions=("setup_family", "subset", "subset_kind", "included_sizes"),
        measures=measures,
        attempt_id=attempt.id,
        counts=(
            NamedCount(name="recipes", value=25),
            NamedCount(name="points", value=len(plot_points)),
        ),
        points=tuple(plot_points),
    )


def _evaluate(
    attempt: AttemptSpec,
    claim: PaperClaim,
    rule: ComparisonRule,
    *,
    points: tuple[_Point, ...],
    frontier_points: tuple[_Point, ...],
    frontier: _Frontier,
    errors: Mapping[str, _Error],
    error_missing: Mapping[str, tuple[str, ...]],
    target_missing: tuple[str, ...],
    decision_missing: tuple[str, ...],
    selections: Mapping[str, RowSelection],
    series: PlotSeries | None,
    diagnostics: tuple[str, ...],
) -> AttemptResult:
    claim_id = attempt.claim_id
    rows = []
    missing = []
    checkpoints = ()
    denominator = None
    target_ties = predicted_ties = 0
    holds = False
    outcome_override: ValidationOutcome | None = None
    result_diagnostics = list(diagnostics)
    result_limitations = [
        "Predictions and released errors are author-derived aggregates; the upstream fits were not rerun.",
        "Released rel_error_stacked uses prediction as denominator, while paper line 330 states target.",
        "Predicates and sensitivity grids are read from paper_validation.toml.",
    ]

    if claim_id in _ERROR_CLAIMS:
        family, expected_relative, expected_absolute = _ERROR_CLAIMS[claim_id]
        rows.append(selections[f"error:{family}"])
        missing += error_missing[family]
        summary = errors.get(family)
        if summary is None:
            computed: object = {"setup_family": family, "row_count": 0}
        else:
            denominator = summary.rows
            released_display_match = (
                round(summary.released_relative_percent, 1) == expected_relative
                and round(summary.absolute_percent, 1) == expected_absolute
            )
            paper_formula_match = (
                round(summary.paper_formula_relative_percent, 1) == expected_relative
            )
            outcome_override = _adjudicate_error(
                released_display_match=released_display_match,
                paper_formula_match=paper_formula_match,
            )
            computed = {
                "setup_family": family,
                "row_count": summary.rows,
                "released_relative_error_percent": summary.released_relative_percent,
                "paper_formula_relative_error_percent": summary.paper_formula_relative_percent,
                "absolute_error_percent": summary.absolute_percent,
                "displayed_released_relative_error_percent": round(
                    summary.released_relative_percent, 1
                ),
                "displayed_paper_formula_relative_error_percent": round(
                    summary.paper_formula_relative_percent, 1
                ),
                "displayed_absolute_error_percent": round(summary.absolute_percent, 1),
                "released_relative_denominator": "prediction",
                "paper_formula_relative_denominator": "target",
                "relative_error_denominator_discrepancy": True,
                "released_display_match": released_display_match,
                "paper_formula_match": paper_formula_match,
            }
            result_diagnostics.extend(
                (
                    f"released_display_match={str(released_display_match).lower()}",
                    f"paper_formula_match={str(paper_formula_match).lower()}",
                )
            )
            if released_display_match and not paper_formula_match:
                result_limitations.append(
                    "The displayed table is regenerated from the author-derived rel_error_stacked column, whose prediction denominator is inconsistent with the paper method; it is not reproduced under the paper-stated target-denominator formula."
                )
    else:
        rows.append(selections["decision"])
        missing += [*target_missing, *decision_missing]
        denominators = {point.pair.denominator for point in points}
        denominator = next(iter(denominators)) if len(denominators) == 1 else None
        target_ties = sum(point.pair.target_ties for point in points)
        predicted_ties = sum(point.pair.predicted_ties for point in points)
        if claim_id in _FRONTIER_CLAIMS:
            rows += frontier.selections
            checkpoints = frontier.checkpoints
            missing += frontier.missing
        by_family = {
            family: tuple(point for point in points if point.setup.family == family)
            for family in _FAMILIES
        }
        means = {
            family: _mean(item.pair.accuracy for item in values)
            for family, values in by_family.items()
            if values
        }
        maxima = {
            family: max(item.pair.accuracy for item in values)
            for family, values in by_family.items()
            if values
        }
        common = {
            "decision_point_count": len(points),
            "setup_family_count": len(means),
            "size_subset_count": len(points) // len(_FAMILIES),
            "mean_decision_accuracy_by_setup": means,
            "maximum_decision_accuracy_by_setup": maxima,
        }
        if claim_id in {"DD-0013", "DD-0054", "DD-0180", "DD-0181"}:
            threshold = _parameter(
                rule, ComparisonParameterName.FRONTIER_DIFFERENCE_MAXIMUM
            )
            differences = tuple(
                point.frontier_difference
                for point in frontier_points
                if point.frontier_difference is not None
            )
            holds = bool(differences) and all(
                value <= threshold for value in differences
            )
            computed = {
                **common,
                "frontier_comparison_count": len(differences),
                "maximum_frontier_difference": max(differences)
                if differences
                else None,
                "frontier_difference_maximum": threshold,
                "all_points_at_or_below_frontier": holds,
            }
        elif claim_id == "DD-0119":
            threshold = _parameter(rule, ComparisonParameterName.MARKED_GAP_MINIMUM)
            best, baseline = (
                max(maxima.values(), default=0.0),
                maxima.get("3_param", 0.0),
            )
            advantage = best - baseline
            holds = bool(maxima) and advantage <= threshold
            computed = {
                **common,
                "best_decision_accuracy": best,
                "plain_three_parameter_baseline_accuracy": baseline,
                "best_vs_baseline_advantage": advantage,
                "marked_gap_minimum": threshold,
                "material_advantage_absent": holds,
            }
        elif claim_id == "DD-0189":
            allowed = int(_parameter(rule, ComparisonParameterName.TASK_COUNT))
            ranks = _ranks(maxima, True) if maxima else {}
            holds = all(
                ranks.get(family, allowed + 1) <= allowed
                for family in ("2_param", "3_param")
            )
            computed = {
                **common,
                "maximum_accuracy_rank_by_setup": ranks,
                "maximum_allowed_rank": allowed,
                "two_and_plain_three_parameter_are_top_ranked": holds,
            }
        elif claim_id == "DD-0192":
            threshold = _parameter(rule, ComparisonParameterName.ACCURACY_THRESHOLD)
            holds = (
                frontier.maximum_accuracy is not None
                and frontier.maximum_accuracy >= threshold
            )
            computed = {
                **common,
                "maximum_single_scale_accuracy": frontier.maximum_accuracy,
                "accuracy_threshold": threshold,
                "strong_single_scale_baseline": holds,
            }
        elif claim_id in {"DD-0311", "DD-0330"}:
            rows.append(selections["comparable_errors"])
            missing += [
                item for family in _FAMILIES[:5] for item in error_missing[family]
            ]
            values = {
                family: errors[family].released_relative_percent / 100
                for family in _FAMILIES[:5]
                if family in errors
            }
            spread = max(values.values()) - min(values.values()) if values else None
            threshold = _parameter(rule, ComparisonParameterName.OVERLAP_RANGE_MAXIMUM)
            holds = spread is not None and spread <= threshold
            computed = {
                **common,
                "released_relative_error_by_setup": values,
                "comparable_error_spread": spread,
                "overlap_range_maximum": threshold,
                "comparable_error_spread_holds": holds,
            }
        elif claim_id == "DD-0312":
            rows.append(selections["all_errors"])
            missing += [item for family in _FAMILIES for item in error_missing[family]]
            values = {
                family: errors[family].released_relative_percent / 100
                for family in _FAMILIES
                if family in errors
            }
            high = tuple(
                sorted(values, key=lambda family: (-values[family], family))[:3]
            )
            low = tuple(sorted(means, key=lambda family: (means[family], family))[:3])
            correlation = (
                _spearman(
                    tuple(values[family] for family in _FAMILIES),
                    tuple(means[family] for family in _FAMILIES),
                )
                if len(values) == len(means) == 8
                else None
            )
            holds = len(high) == 3 and set(high) == set(low)
            computed = {
                **common,
                "highest_error_three": list(high),
                "lowest_mean_accuracy_three": list(low),
                "error_vs_accuracy_spearman": correlation,
                "highest_error_three_equal_lowest_accuracy_three": holds,
            }
        elif claim_id == "DD-0368":
            threshold = _parameter(
                rule, ComparisonParameterName.FRONTIER_DIFFERENCE_MAXIMUM
            )
            selected = tuple(
                point
                for point in frontier_points
                if point.setup.family == "5_param-ai2"
            )
            below = sum(
                point.frontier_difference is not None
                and point.frontier_difference <= threshold
                for point in selected
            )
            holds = bool(selected) and below == len(selected)
            computed = {
                **common,
                "five_parameter_point_count": len(selected),
                "five_parameter_points_at_or_below_frontier": below,
                "all_five_parameter_points_at_or_below_frontier": holds,
            }
        elif claim_id == "DD-0369":
            fraction = _parameter(rule, ComparisonParameterName.FRACTION_THRESHOLD)
            chance = _parameter(rule, ComparisonParameterName.CHANCE_BASELINE)
            tolerance = _parameter(rule, ComparisonParameterName.TRIVIAL_TOLERANCE)
            tie_credit = _parameter(rule, ComparisonParameterName.PREDICTED_TIE_CREDIT)
            selected = tuple(
                point for point in points if point.setup.family == "3_param-1_step"
            )
            near = sum(
                abs(point.pair.accuracy - chance) <= tolerance for point in selected
            )
            holds = bool(selected) and near / len(selected) > fraction
            computed = {
                **common,
                "single_step_point_count": len(selected),
                "near_chance_point_count": near,
                "near_chance_fraction": near / len(selected) if selected else None,
                "fraction_threshold_exclusive": fraction,
                "chance_baseline": chance,
                "trivial_tolerance": tolerance,
                "predicted_tie_credit": tie_credit,
                "strict_majority_near_chance": holds,
            }
        else:
            raise ValueError(f"unsupported scaling claim {claim_id}")

    missing_groups = tuple(sorted(set(missing)))
    outcome = (
        ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
        if missing_groups
        else outcome_override
        if outcome_override is not None
        else ValidationOutcome.REPRODUCED
        if holds
        else ValidationOutcome.NOT_REPRODUCED
    )
    return AttemptResult(
        attempt_id=attempt.id,
        claim_id=claim_id,
        role=AttemptRole.DEFAULT,
        evidence_level=EvidenceLevel.AUTHOR_DERIVED_AGGREGATE,
        comparison_rule_id=rule.id,
        comparison_rule_version=rule.version,
        transformation_ids=attempt.transformation_ids,
        row_selections=tuple(rows),
        checkpoint_selections=checkpoints,
        target_value=claim.paper_target,
        computed_value=computed,
        seeds=("default",),
        denominator=denominator,
        exclusions=(
            NamedCount(name="target_tied_pairs", value=target_ties),
            NamedCount(name="predicted_tied_pairs", value=predicted_ties),
        ),
        missing_groups=missing_groups,
        target_ties=target_ties,
        predicted_ties=predicted_ties,
        outcome=outcome,
        diagnostics=tuple(result_diagnostics),
        limitations=tuple(result_limitations),
        plot_series_ids=() if series is None or missing_groups else (series.id,),
    )


def run_scaling_law_attempts(
    *,
    repository_root: Path,
    data_root: Path,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
) -> tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]]:
    """Verify scaling-law findings from the declared released aggregates."""
    attempts = tuple(
        sorted(
            (
                item
                for item in contract.attempts
                if item.analysis_id is AnalysisId.SCALING_LAW
            ),
            key=lambda item: item.id,
        )
    )
    if not attempts:
        return (), ()
    root, inputs = Path(repository_root), Path(data_root)
    catalog = load_toml_model(root / "configs/catalog.toml", DataDecideCatalog)
    olmes_contract = load_toml_model(root / "configs/olmes.toml", OLMESContract)
    scaling_contract = load_toml_model(
        root / "configs/scaling_law.toml", ScalingLawContract
    )
    fit_sizes = tuple(size for size in scaling_contract.models if size != _TARGET_SIZE)
    mixes = tuple(sorted(scaling_contract.source_group_map))
    tasks = (*single_adapter.DEFAULT_TASK_GROUPING.non_mmlu_tasks, "mmlu", _TASK)
    if len(fit_sizes) != 13 or len(mixes) != 25 or len(tasks) != 11:
        raise ValueError("paper scaling universes drifted")

    cheap_spec = _spec(contract, _CHEAP)
    if cheap_spec.columns != _CHEAP_COLUMNS:
        raise ValueError("cheap-decisions configured schema drifted")
    frame, cheap_digest = _read(inputs / cheap_spec.path, cheap_spec, input_identities)
    if (
        frame[list(_KEYS)].isna().any().any()
        or frame.duplicated(list(_KEYS), keep=False).any()
    ):
        raise ValueError("cheap-decisions keys must be non-null and unique")
    parsed = {
        str(value): _parse_setup(str(value), fit_sizes)
        for value in frame.setup.unique()
    }
    included_names = {name for name, value in parsed.items() if value is not None}
    included = frame[frame.setup.isin(included_names)].copy()
    setups = _expected_setups(fit_sizes)
    if included_names - {setup.name for setup in setups}:
        raise ValueError("unsupported paper setup subset")
    decision = included[(included.task == _TASK) & (included.metric == _METRIC)].copy()
    target, target_missing, incompatible_difference = _target(decision, setups, mixes)
    costs, target_compute = _catalog_compute(catalog)
    if target_compute != 7.060992e20:
        raise ValueError("catalog 1B compute drifted")
    default_points, decision_missing = _points(
        decision, setups, mixes, target, costs, target_compute, 0.0
    )

    errors, error_missing, error_frames = {}, {}, {}
    for family in _FAMILIES:
        summary, missing, rows = _error(included, family, tasks, mixes)
        if summary is not None:
            errors[family] = summary
        error_missing[family], error_frames[family] = missing, rows

    selections = {
        "decision": _selection(
            decision,
            cheap_spec,
            cheap_digest,
            (
                RowPredicate(column="task", operator=PredicateOperator.EQ, value=_TASK),
                RowPredicate(column="mix", operator=PredicateOperator.IN, value=mixes),
                RowPredicate(
                    column="metric", operator=PredicateOperator.EQ, value=_METRIC
                ),
                RowPredicate(
                    column="setup",
                    operator=PredicateOperator.IN,
                    value=tuple(setup.name for setup in setups),
                ),
            ),
        )
    }
    for family in _FAMILIES:
        selections[f"error:{family}"] = _selection(
            error_frames[family],
            cheap_spec,
            cheap_digest,
            (
                RowPredicate(column="task", operator=PredicateOperator.IN, value=tasks),
                RowPredicate(column="mix", operator=PredicateOperator.IN, value=mixes),
                RowPredicate(
                    column="metric", operator=PredicateOperator.EQ, value=_METRIC
                ),
                RowPredicate(
                    column="setup", operator=PredicateOperator.EQ, value=family
                ),
            ),
        )
    comparable = pd.concat(
        tuple(error_frames[family] for family in _FAMILIES[:5]), ignore_index=True
    )
    all_errors = pd.concat(tuple(error_frames.values()), ignore_index=True)
    for name, selected, families in (
        ("comparable_errors", comparable, _FAMILIES[:5]),
        ("all_errors", all_errors, _FAMILIES),
    ):
        selections[name] = _selection(
            selected,
            cheap_spec,
            cheap_digest,
            (
                RowPredicate(column="task", operator=PredicateOperator.IN, value=tasks),
                RowPredicate(column="mix", operator=PredicateOperator.IN, value=mixes),
                RowPredicate(
                    column="metric", operator=PredicateOperator.EQ, value=_METRIC
                ),
                RowPredicate(
                    column="setup", operator=PredicateOperator.IN, value=families
                ),
            ),
        )

    frontier = _Frontier((), (), (), None, ())
    frontier_digest = "not-declared"
    if any(
        any(value.table_id == _OLMES for value in attempt.inputs)
        for attempt in attempts
    ):
        display_target = (
            {
                olmes_contract.recipe_map[scaling_contract.source_group_map[mix]]: score
                for mix, score in target.items()
            }
            if target is not None
            else {}
        )
        frontier, frontier_digest = _frontier(
            inputs, contract, input_identities, display_target
        )
    frontier_points = _compare_frontier(default_points, frontier)
    if frontier.points and len(frontier_points) != len(default_points):
        frontier = _Frontier(
            frontier.points,
            frontier.selections,
            frontier.checkpoints,
            frontier.maximum_accuracy,
            ("single_scale:missing=frontier_at_multi_scale_compute",),
        )

    claims = {claim.id: claim for claim in registry.claims}
    rules = {rule.id: rule for rule in contract.comparison_rules}
    diagnostics = (
        f"cheap_decisions_sha256={cheap_digest}",
        f"olmes_aggregate_sha256={frontier_digest}",
        f"source_row_count={len(frame)}",
        f"selected_paper_setup_row_count={len(included)}",
        f"excluded_intermediate_setup_row_count={len(frame) - len(included)}",
        f"paper_setup_count={len(setups)}",
        f"decision_point_count={len(default_points)}",
        f"catalog_target_compute={target_compute:.17g}",
        "target_ranking_owner=exact_common_stacked_y_across_five_compatible_two_stage_families",
        f"incompatible_five_parameter_target_maximum_absolute_difference={incompatible_difference}",
        "predictions=stacked_pred",
        "released_relative_error_denominator=absolute_prediction",
        "paper_line_330_relative_error_denominator=target",
    )
    results, series_values = [], []
    for attempt in attempts:
        if attempt.claim_id not in claims or attempt.comparison_rule_id not in rules:
            raise ValueError(f"invalid references for {attempt.id}")
        declared = {value.table_id for value in attempt.inputs}
        if (attempt.claim_id in _FRONTIER_CLAIMS) != (_OLMES in declared):
            raise ValueError(f"frontier declaration mismatch for {attempt.id}")
        variants = (
            (attempt.id, None, None, rules[attempt.comparison_rule_id]),
        ) + _sensitivities(attempt, rules[attempt.comparison_rule_id])
        for result_id, parameter_name, parameter_value, rule in variants:
            credit = (
                _parameter(rule, ComparisonParameterName.PREDICTED_TIE_CREDIT)
                if attempt.claim_id == "DD-0369"
                else 0.0
            )
            points = default_points
            missing = decision_missing
            if credit:
                points, sensitivity_missing = _points(
                    decision, setups, mixes, target, costs, target_compute, credit
                )
                missing = tuple(sorted(set(missing) | set(sensitivity_missing)))
            plotted_points = (
                frontier_points
                if attempt.claim_id in {"DD-0180", "DD-0368"}
                else points
            )
            series = _plot(attempt, plotted_points) if result_id == attempt.id else None
            result = _evaluate(
                attempt,
                claims[attempt.claim_id],
                rule,
                points=points,
                frontier_points=frontier_points,
                frontier=frontier,
                errors=errors,
                error_missing=error_missing,
                target_missing=target_missing,
                decision_missing=missing,
                selections=selections,
                series=series,
                diagnostics=diagnostics,
            )
            if result_id != attempt.id:
                assert parameter_name is not None and parameter_value is not None
                result = result.model_copy(
                    update={
                        "attempt_id": result_id,
                        "role": AttemptRole.SENSITIVITY,
                        "parent_attempt_id": attempt.id,
                        "plot_series_ids": (),
                        "diagnostics": (
                            *result.diagnostics,
                            f"comparison_parameter={parameter_name.value}",
                            f"comparison_parameter_value={parameter_value:.17g}",
                        ),
                    }
                )
            elif series is not None and not result.missing_groups:
                if attempt.plot_series_ids != (series.id,):
                    raise ValueError(f"plot declaration mismatch for {attempt.id}")
                series_values.append(series)
            results.append(result)
    return tuple(sorted(results, key=lambda item: item.attempt_id)), tuple(
        sorted(series_values, key=lambda item: item.id)
    )


__all__ = ["run_scaling_law_attempts"]

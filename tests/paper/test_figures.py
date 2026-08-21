from __future__ import annotations

import builtins
import math
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import pytest

from datadec.paper.figures import render_figures, render_outcome_audit_svg
from datadec.paper.models import (
    AnalysisBundle,
    AnalysisManifest,
    AttemptResult,
    AttemptRole,
    AxisScale,
    AxisSpec,
    ClaimKind,
    ContentIdentity,
    DimensionValue,
    MeasureValue,
    MetadataDiscrepancy,
    PaperTarget,
    PlotPoint,
    PlotSeries,
    RowSelection,
    ValidationOutcome,
)

_SHA = "a" * 64
_NOW = datetime(2026, 8, 21, 12, tzinfo=UTC)
_FIGURE_NAMES = (
    "compute-vs-decision",
    "per-task",
    "scaling-law",
    "proxy-metrics",
    "noise-spread",
)


def _manifest() -> AnalysisManifest:
    def identity(name: str) -> ContentIdentity:
        return ContentIdentity(id=name, sha256=_SHA)

    return AnalysisManifest(
        run_id="run-figure-test",
        started_at=_NOW,
        completed_at=_NOW,
        input_identities=(identity("olmes"),),
        targets_identity=identity("targets.json"),
        attempts_identity=identity("attempts.json"),
        plot_series_identity=identity("plot-series.json"),
    )


def _target() -> PaperTarget:
    return PaperTarget(
        claim_id="DD-0001",
        family="single_scale",
        kind=ClaimKind.EMPIRICAL_PLOT,
        source_file="docs/paper/example_paper.tex",
        line_start=10,
        line_end=11,
        source_text="Decision accuracy over compute.",
        value="increasing trend",
    )


def _selection() -> RowSelection:
    return RowSelection(
        logical_table_id="olmes",
        columns=("compute", "decision_accuracy"),
        predicates=(),
        local_parquet_sha256=_SHA,
        selected_row_count=2,
        selected_key_sha256=_SHA,
    )


def _attempt(
    attempt_id: str,
    *,
    role: AttemptRole = AttemptRole.DEFAULT,
    parent: str | None = None,
    outcome: ValidationOutcome = ValidationOutcome.REPRODUCED,
    series_ids: tuple[str, ...] = (),
) -> AttemptResult:
    return AttemptResult(
        attempt_id=attempt_id,
        claim_id="DD-0001",
        role=role,
        parent_attempt_id=parent,
        comparison_rule_id="nonempty-plot",
        comparison_rule_version=1,
        transformation_ids=("pairwise-decisions",),
        row_selections=(_selection(),),
        target_value="increasing trend",
        computed_value={"persisted": True},
        outcome=outcome,
        plot_series_ids=series_ids,
    )


def _point(
    compute: float,
    accuracy: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> PlotPoint:
    measures = [
        MeasureValue(name="compute", value=compute),
        MeasureValue(name="decision_accuracy", value=accuracy),
    ]
    if lower is not None and upper is not None:
        measures.extend(
            (
                MeasureValue(name="decision_accuracy_lower", value=lower),
                MeasureValue(name="decision_accuracy_upper", value=upper),
            )
        )
    return PlotPoint(
        dimensions=(DimensionValue(name="model_size", value="150M"),),
        measures=tuple(measures),
    )


def _noise_point(noise: float, spread: float, accuracy: float, task: str) -> PlotPoint:
    return PlotPoint(
        dimensions=(DimensionValue(name="task", value=task),),
        measures=(
            MeasureValue(name="noise", value=noise),
            MeasureValue(name="spread", value=spread),
            MeasureValue(name="decision_accuracy", value=accuracy),
        ),
    )


def _series(
    figure: str,
    *,
    series_id: str | None = None,
    points: tuple[PlotPoint, ...] | None = None,
    paper_analog: bool = True,
) -> PlotSeries:
    if figure == "noise-spread":
        return PlotSeries(
            id=series_id or "dd-0001-noise-spread-paper-analog",
            figure=figure,
            panel="150M",
            semantic_kind="noise_spread_scatter",
            x_axis=AxisSpec(measure="noise", scale=AxisScale.LOG, unit="stddev"),
            y_axis=AxisSpec(measure="spread", scale=AxisScale.LOG, unit="stddev"),
            dimensions=("task",),
            measures=("noise", "spread", "decision_accuracy"),
            attempt_id="dd-0001-default",
            points=points
            if points is not None
            else (
                _noise_point(0.001, 0.01, 0.8, "ARC Easy"),
                _noise_point(0.01, 0.05, 0.6, "HellaSwag"),
            ),
            paper_analog=paper_analog,
        )
    has_uncertainty = figure == "compute-vs-decision"
    series_points = (
        points
        if points is not None
        else (
            _point(
                0.001,
                0.65,
                lower=0.6 if has_uncertainty else None,
                upper=0.7 if has_uncertainty else None,
            ),
            _point(
                1.0,
                0.8,
                lower=0.77 if has_uncertainty else None,
                upper=0.83 if has_uncertainty else None,
            ),
        )
    )
    measures = ["compute", "decision_accuracy"]
    if has_uncertainty:
        measures.extend(("decision_accuracy_lower", "decision_accuracy_upper"))
    return PlotSeries(
        id=series_id or f"dd-0001-{figure}-paper-analog",
        figure=figure,
        panel="aggregate" if figure != "per-task" else "ARC Easy",
        semantic_kind=figure.replace("-", "_"),
        x_axis=AxisSpec(measure="compute", scale=AxisScale.LOG, unit="percent"),
        y_axis=AxisSpec(
            measure="decision_accuracy", scale=AxisScale.LINEAR, unit="ratio"
        ),
        dimensions=(
            tuple(item.name for item in series_points[0].dimensions)
            if series_points
            else ("model_size",)
        ),
        measures=tuple(measures),
        attempt_id="dd-0001-default",
        points=series_points,
        paper_analog=paper_analog,
    )


def _bundle(
    *, series: tuple[PlotSeries, ...], include_sensitivity: bool = True
) -> AnalysisBundle:
    default = _attempt("dd-0001-default", series_ids=tuple(item.id for item in series))
    attempts = [default]
    if include_sensitivity:
        attempts.append(
            _attempt(
                "dd-0001-sensitivity",
                role=AttemptRole.SENSITIVITY,
                parent=default.attempt_id,
                outcome=ValidationOutcome.NOT_REPRODUCED,
            )
        )
    return AnalysisBundle(
        manifest=_manifest(),
        targets=(_target(),),
        metadata_discrepancies=(
            MetadataDiscrepancy(
                claim_id="DD-META",
                paper_locator="paper:20",
                paper_value=1,
                metadata_source="metadata",
                metadata_value=2,
                note="not an empirical result",
            ),
        ),
        attempts=tuple(attempts),
        plot_series=series,
    )


def _assert_valid_accessible_svg(svg: bytes) -> ET.Element:
    root = ET.fromstring(svg)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.attrib["role"] == "img"
    assert root.attrib["aria-labelledby"]
    children = list(root)
    assert children[0].tag == "{http://www.w3.org/2000/svg}title"
    assert children[1].tag == "{http://www.w3.org/2000/svg}desc"
    return root


def test_paper_analog_figures_are_named_deterministic_valid_and_semantic() -> None:
    series = tuple(_series(name) for name in reversed(_FIGURE_NAMES))
    bundle = _bundle(series=series)

    rendered = render_figures(bundle)
    reordered = bundle.model_copy(update={"plot_series": tuple(reversed(series))})

    assert render_figures(reordered) == rendered
    assert tuple(name for name, _ in rendered) == (
        "outcome-audit.svg",
        "compute-vs-decision.svg",
        "noise-spread.svg",
        "per-task.svg",
        "proxy-metrics.svg",
        "scaling-law.svg",
    )
    for _, svg in rendered:
        _assert_valid_accessible_svg(svg)

    compute_svg = dict(rendered)["compute-vs-decision.svg"].decode()
    assert "compute (percent; log)" in compute_svg
    assert "decision_accuracy (ratio; linear)" in compute_svg
    assert 'data-series="dd-0001-compute-vs-decision-paper-analog"' in compute_svg
    assert 'class="uncertainty-band"' in compute_svg
    assert "model_size=150M" in compute_svg
    assert "decision_accuracy_lower=0.6" in compute_svg
    assert "Semantic kinds: compute_vs_decision" in compute_svg
    noise_svg = dict(rendered)["noise-spread.svg"].decode()
    assert 'data-color-measure="decision_accuracy"' in noise_svg
    assert "Color: decision_accuracy (0–1); values in plot-series.json." in noise_svg
    assert "ARC Easy" in noise_svg
    assert '<path class="series-line" data-series="dd-0001-noise' not in noise_svg


def test_geometry_is_finite_and_log_axes_require_positive_values() -> None:
    svg = dict(render_figures(_bundle(series=(_series("noise-spread"),))))[
        "noise-spread.svg"
    ].decode()

    for value in re.findall(r'(?:x|y|x1|x2|y1|y2|cx|cy)="([0-9.+-]+)"', svg):
        assert math.isfinite(float(value))

    zero_point = _noise_point(0.0, 0.01, 0.5, "zero")
    invalid = _series(
        "noise-spread",
        points=(zero_point, _noise_point(1.0, 0.1, 0.8, "positive")),
    )
    with pytest.raises(ValueError, match="positive on a log axis"):
        render_figures(_bundle(series=(invalid,)))


def test_outcome_audit_counts_default_primary_results_only() -> None:
    bundle = _bundle(series=())

    audit = render_outcome_audit_svg(bundle).decode()

    assert "reproduced: 1" in audit
    assert "not_reproduced: 0" in audit
    assert "metadata_discrepancy" not in audit
    assert "sensitivities and metadata discrepancies excluded" in audit


def test_empty_nonanalog_series_is_suppressed_and_rendering_reads_no_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = _series(
        "internal-audit",
        series_id="dd-0001-internal",
        points=(),
        paper_analog=False,
    )
    bundle = _bundle(series=(empty,))

    def fail_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("figure rendering must not open any input")

    monkeypatch.setattr(builtins, "open", fail_read)

    rendered = render_figures(bundle)

    assert tuple(name for name, _ in rendered) == ("outcome-audit.svg",)


def test_large_series_uses_compact_path_without_per_point_titles() -> None:
    points = tuple(_point(float(index + 1), 0.5 + index / 1000) for index in range(101))
    rendered = dict(
        render_figures(_bundle(series=(_series("per-task", points=points),)))
    )["per-task.svg"]

    svg = rendered.decode()
    assert 'data-point-count="101"' in svg
    assert "exact values are in plot-series.json" in svg
    assert "<circle" not in svg
    assert len(rendered) < 25_000


def test_curve_paths_group_nontrajectory_dimensions_and_sort_by_x() -> None:
    points = (
        PlotPoint(
            dimensions=(
                DimensionValue(name="task", value="b"),
                DimensionValue(name="model_size", value="4M"),
                DimensionValue(name="step", value=2),
            ),
            measures=(
                MeasureValue(name="compute", value=2.0),
                MeasureValue(name="decision_accuracy", value=0.6),
            ),
        ),
        PlotPoint(
            dimensions=(
                DimensionValue(name="task", value="a"),
                DimensionValue(name="model_size", value="4M"),
                DimensionValue(name="step", value=1),
            ),
            measures=(
                MeasureValue(name="compute", value=1.0),
                MeasureValue(name="decision_accuracy", value=0.5),
            ),
        ),
        PlotPoint(
            dimensions=(
                DimensionValue(name="task", value="a"),
                DimensionValue(name="model_size", value="4M"),
                DimensionValue(name="step", value=3),
            ),
            measures=(
                MeasureValue(name="compute", value=3.0),
                MeasureValue(name="decision_accuracy", value=0.7),
            ),
        ),
    )
    svg = dict(render_figures(_bundle(series=(_series("per-task", points=points),))))[
        "per-task.svg"
    ].decode()

    assert svg.count('class="series-line"') == 2
    assert 'data-subseries="task=a;model_size=4M"' in svg
    assert 'data-subseries="task=b;model_size=4M"' in svg

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from math import ceil, isfinite, log10

from datadec.paper.models import (
    PRIMARY_CLAIM_KINDS,
    AnalysisBundle,
    AttemptRole,
    AxisScale,
    PlotPoint,
    PlotSeries,
    ValidationOutcome,
)

_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_COLORS = (
    "#2166ac",
    "#b2182b",
    "#1b7837",
    "#762a83",
    "#e08214",
    "#008b8b",
    "#555555",
    "#c51b7d",
)
_PRIMARY_OUTCOMES = (
    ValidationOutcome.REPRODUCED,
    ValidationOutcome.APPROXIMATELY_REPRODUCED,
    ValidationOutcome.DIRECTIONALLY_CONSISTENT,
    ValidationOutcome.NOT_REPRODUCED,
    ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED,
)


@dataclass(frozen=True, slots=True)
class _Scale:
    minimum: float
    maximum: float
    pixel_minimum: float
    pixel_maximum: float
    logarithmic: bool

    def project(self, value: float) -> float:
        transformed = log10(value) if self.logarithmic else value
        domain_minimum = log10(self.minimum) if self.logarithmic else self.minimum
        domain_maximum = log10(self.maximum) if self.logarithmic else self.maximum
        fraction = (transformed - domain_minimum) / (domain_maximum - domain_minimum)
        projected = self.pixel_minimum + fraction * (
            self.pixel_maximum - self.pixel_minimum
        )
        if not isfinite(projected):
            raise ValueError("plot geometry must contain only finite coordinates")
        return projected


def _xml(value: object) -> str:
    return escape(str(value), quote=True)


def _number(value: float) -> str:
    if not isfinite(value):
        raise ValueError("plot geometry must contain only finite coordinates")
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError(f"figure name cannot form an output filename: {value!r}")
    return slug


def _validate_svg(svg: str) -> bytes:
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as error:
        raise ValueError("rendered figure is not valid XML") from error
    if root.tag != f"{{{_SVG_NAMESPACE}}}svg":
        raise ValueError("rendered figure root must be SVG")
    return svg.encode()


def _svg_header(
    *, title: str, description: str, width: int, height: int, identifier: str
) -> list[str]:
    title_id = f"{identifier}-title"
    description_id = f"{identifier}-description"
    return [
        f'<svg xmlns="{_SVG_NAMESPACE}" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="{title_id} {description_id}">',
        f'<title id="{title_id}">{_xml(title)}</title>',
        f'<desc id="{description_id}">{_xml(description)}</desc>',
        "<style>"
        "text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#172033}"
        ".figure-title{font-size:20px;font-weight:700}"
        ".panel-title{font-size:15px;font-weight:700}"
        ".axis-label{font-size:11px;font-weight:600}"
        ".tick{font-size:10px;fill:#536174}"
        ".legend{font-size:10px;fill:#26354a}"
        ".grid{stroke:#dce2e9;stroke-width:1}"
        ".axis{stroke:#657286;stroke-width:1.2}"
        ".series-line{fill:none;stroke-width:2}"
        ".uncertainty-band{stroke:none;opacity:.16}"
        ".series-point{stroke:#fff;stroke-width:.8}"
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _measure(point: PlotPoint, name: str) -> float:
    for measure in point.measures:
        if measure.name == name:
            return measure.value
    raise ValueError(f"plot point is missing declared measure {name!r}")


def _uncertainty_measures(series: PlotSeries) -> tuple[str, str] | None:
    y_name = series.y_axis.measure
    names = set(series.measures)
    for lower, upper in (
        (f"{y_name}_lower", f"{y_name}_upper"),
        ("y_lower", "y_upper"),
        ("lower", "upper"),
    ):
        if {lower, upper} <= names:
            return lower, upper
    return None


def _make_scale(
    values: list[float],
    *,
    axis_scale: AxisScale,
    pixel_minimum: float,
    pixel_maximum: float,
    axis_description: str,
) -> _Scale:
    if not values or any(not isfinite(value) for value in values):
        raise ValueError(f"{axis_description} values must be finite and nonempty")
    logarithmic = axis_scale is AxisScale.LOG
    if logarithmic and any(value <= 0 for value in values):
        raise ValueError(f"{axis_description} values must be positive on a log axis")
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        if logarithmic:
            minimum /= 10**0.5
            maximum *= 10**0.5
        else:
            padding = max(abs(minimum) * 0.05, 1.0)
            minimum -= padding
            maximum += padding
    return _Scale(
        minimum=minimum,
        maximum=maximum,
        pixel_minimum=pixel_minimum,
        pixel_maximum=pixel_maximum,
        logarithmic=logarithmic,
    )


def _tick_values(scale: _Scale) -> tuple[float, ...]:
    minimum = log10(scale.minimum) if scale.logarithmic else scale.minimum
    maximum = log10(scale.maximum) if scale.logarithmic else scale.maximum
    return tuple(
        10 ** (minimum + index * (maximum - minimum) / 4)
        if scale.logarithmic
        else minimum + index * (maximum - minimum) / 4
        for index in range(5)
    )


def _tick_label(value: float, logarithmic: bool) -> str:
    if logarithmic:
        exponent = log10(value)
        if abs(exponent - round(exponent)) < 1e-9:
            return f"10^{round(exponent)}"
    return f"{value:.3g}"


def _point_description(series: PlotSeries, point: PlotPoint) -> str:
    values = [f"series={series.id}"]
    values.extend(f"{item.name}={item.value}" for item in point.dimensions)
    values.extend(f"{item.name}={item.value:.12g}" for item in point.measures)
    return "; ".join(values)


def _is_noise_spread(series: PlotSeries) -> bool:
    semantic_kind = series.semantic_kind.lower()
    return "noise" in semantic_kind and "spread" in semantic_kind


def _decision_accuracy_measure(series: PlotSeries) -> str | None:
    axis_measures = {series.x_axis.measure, series.y_axis.measure}
    return next(
        (
            measure
            for measure in series.measures
            if measure not in axis_measures
            and "decision" in measure.lower()
            and "accuracy" in measure.lower()
        ),
        None,
    )


def _ratio_color(value: float, *, description: str) -> str:
    if not 0 <= value <= 1:
        raise ValueError(f"{description} must be between zero and one")
    low = (68, 1, 84)
    high = (253, 231, 37)
    channels = tuple(
        round(start + value * (end - start)) for start, end in zip(low, high)
    )
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _point_label(point: PlotPoint) -> str | None:
    preferred = next(
        (item for item in point.dimensions if item.name.lower() == "task"), None
    )
    dimension = preferred or (point.dimensions[0] if point.dimensions else None)
    return None if dimension is None else str(dimension.value)


def _render_panel(
    panel: str,
    series_group: tuple[PlotSeries, ...],
    *,
    origin_x: int,
    origin_y: int,
    panel_width: int,
    panel_height: int,
) -> list[str]:
    left = origin_x + 64
    right = origin_x + panel_width - 24
    top = origin_y + 48
    bottom = origin_y + panel_height - 58
    x_values = [
        _measure(point, series.x_axis.measure)
        for series in series_group
        for point in series.points
    ]
    y_values = [
        _measure(point, series.y_axis.measure)
        for series in series_group
        for point in series.points
    ]
    for series in series_group:
        uncertainty = _uncertainty_measures(series)
        if uncertainty is not None:
            lower, upper = uncertainty
            for point in series.points:
                lower_value = _measure(point, lower)
                center_value = _measure(point, series.y_axis.measure)
                upper_value = _measure(point, upper)
                if not lower_value <= center_value <= upper_value:
                    raise ValueError(
                        f"plot series {series.id!r} uncertainty bounds must contain "
                        "the y-axis measure"
                    )
            y_values.extend(_measure(point, lower) for point in series.points)
            y_values.extend(_measure(point, upper) for point in series.points)

    x_axis = series_group[0].x_axis
    y_axis = series_group[0].y_axis
    if any(
        series.x_axis != x_axis or series.y_axis != y_axis for series in series_group
    ):
        raise ValueError(f"plot panel {panel!r} mixes incompatible axes")
    x_scale = _make_scale(
        x_values,
        axis_scale=x_axis.scale,
        pixel_minimum=left,
        pixel_maximum=right,
        axis_description=f"panel {panel} x-axis",
    )
    y_scale = _make_scale(
        y_values,
        axis_scale=y_axis.scale,
        pixel_minimum=bottom,
        pixel_maximum=top,
        axis_description=f"panel {panel} y-axis",
    )

    lines = [
        f'<text class="panel-title" x="{origin_x + 12}" y="{origin_y + 22}">'
        f"{_xml(panel)}</text>"
    ]
    for value in _tick_values(x_scale):
        x = x_scale.project(value)
        lines.extend(
            (
                f'<line class="grid" x1="{_number(x)}" y1="{top}" '
                f'x2="{_number(x)}" y2="{bottom}"/>',
                f'<text class="tick" text-anchor="middle" x="{_number(x)}" '
                f'y="{bottom + 17}">{_xml(_tick_label(value, x_scale.logarithmic))}</text>',
            )
        )
    for value in _tick_values(y_scale):
        y = y_scale.project(value)
        lines.extend(
            (
                f'<line class="grid" x1="{left}" y1="{_number(y)}" '
                f'x2="{right}" y2="{_number(y)}"/>',
                f'<text class="tick" text-anchor="end" x="{left - 7}" '
                f'y="{_number(y + 3)}">{_xml(_tick_label(value, y_scale.logarithmic))}</text>',
            )
        )
    lines.extend(
        (
            f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
            f'<text class="axis-label" text-anchor="middle" x="{(left + right) / 2}" '
            f'y="{origin_y + panel_height - 10}">{_xml(x_axis.measure)} '
            f"({_xml(x_axis.unit)}; {_xml(x_axis.scale.value)})</text>",
            f'<text class="axis-label" text-anchor="middle" '
            f'transform="translate({origin_x + 14} {(top + bottom) / 2}) rotate(-90)">'
            f"{_xml(y_axis.measure)} ({_xml(y_axis.unit)}; {_xml(y_axis.scale.value)})</text>",
        )
    )

    for index, series in enumerate(series_group):
        color = _COLORS[index % len(_COLORS)]
        noise_spread = _is_noise_spread(series)
        color_measure = _decision_accuracy_measure(series) if noise_spread else None
        points = tuple(
            (
                x_scale.project(_measure(point, x_axis.measure)),
                y_scale.project(_measure(point, y_axis.measure)),
                point,
            )
            for point in series.points
        )
        uncertainty = _uncertainty_measures(series)
        if uncertainty is not None:
            lower, upper = uncertainty
            upper_points = [
                (
                    x_scale.project(_measure(point, x_axis.measure)),
                    y_scale.project(_measure(point, upper)),
                )
                for point in series.points
            ]
            lower_points = [
                (
                    x_scale.project(_measure(point, x_axis.measure)),
                    y_scale.project(_measure(point, lower)),
                )
                for point in reversed(series.points)
            ]
            polygon = " ".join(
                f"{_number(x)},{_number(y)}" for x, y in (*upper_points, *lower_points)
            )
            lines.append(
                f'<polygon class="uncertainty-band" data-series="{_xml(series.id)}" '
                f'fill="{color}" points="{polygon}"/>'
            )
        if not noise_spread:
            path = " ".join(
                f"{'M' if point_index == 0 else 'L'} {_number(x)} {_number(y)}"
                for point_index, (x, y, _) in enumerate(points)
            )
            lines.append(
                f'<path class="series-line" data-series="{_xml(series.id)}" '
                f'stroke="{color}" d="{path}"/>'
            )
        for x, y, point in points:
            point_color = (
                _ratio_color(
                    _measure(point, color_measure),
                    description=f"plot series {series.id!r} decision accuracy",
                )
                if color_measure is not None
                else color
            )
            color_attribute = (
                f' data-color-measure="{_xml(color_measure)}"'
                if color_measure is not None
                else ""
            )
            lines.append(
                f'<circle class="series-point" data-series="{_xml(series.id)}" '
                f'cx="{_number(x)}" cy="{_number(y)}" r="3" fill="{point_color}"'
                f"{color_attribute}>"
                f"<title>{_xml(_point_description(series, point))}</title></circle>"
            )
            if noise_spread and (label := _point_label(point)) is not None:
                lines.append(
                    f'<text class="tick" x="{_number(x + 5)}" y="{_number(y - 5)}">'
                    f"{_xml(label)}</text>"
                )
        legend_y = origin_y + 20 + 14 * index
        legend_x = origin_x + panel_width - 180
        lines.extend(
            (
                f'<line x1="{legend_x}" y1="{legend_y - 4}" '
                f'x2="{legend_x + 14}" y2="{legend_y - 4}" stroke="{color}" '
                'stroke-width="2"/>',
                f'<text class="legend" x="{legend_x + 19}" y="{legend_y}">'
                f"{_xml(series.id)}</text>",
            )
        )
        if color_measure is not None:
            lines.append(
                f'<text class="legend" x="{left}" y="{top + 14}" '
                f'data-color-measure="{_xml(color_measure)}">Point color: '
                f"{_xml(color_measure)} (0–1); exact values are in point labels.</text>"
            )
    return lines


def _render_paper_figure(
    bundle: AnalysisBundle, figure: str, series_group: tuple[PlotSeries, ...]
) -> bytes:
    panels: defaultdict[str, list[PlotSeries]] = defaultdict(list)
    for series in sorted(series_group, key=lambda item: (item.panel, item.id)):
        panels[series.panel].append(series)
    panel_names = tuple(sorted(panels))
    columns = 1 if len(panel_names) == 1 else 2
    rows = ceil(len(panel_names) / columns)
    panel_width = 500
    panel_height = 330
    width = panel_width * columns
    height = 68 + panel_height * rows
    identifier = f"figure-{_slug(figure)}"
    semantics = ", ".join(sorted({series.semantic_kind for series in series_group}))
    lines = _svg_header(
        title=figure,
        description=(
            f"Paper-analog figure rendered from persisted plot series for run "
            f"{bundle.manifest.run_id}. Panels: {', '.join(panel_names)}. "
            f"Semantic kinds: {semantics}."
        ),
        width=width,
        height=height,
        identifier=identifier,
    )
    lines.append(f'<text class="figure-title" x="24" y="34">{_xml(figure)}</text>')
    lines.append(
        f'<text class="tick" x="24" y="52">Run {_xml(bundle.manifest.run_id)}; '
        "scientific values are persisted bundle measures.</text>"
    )
    for index, panel in enumerate(panel_names):
        origin_x = (index % columns) * panel_width
        origin_y = 68 + (index // columns) * panel_height
        lines.extend(
            _render_panel(
                panel,
                tuple(panels[panel]),
                origin_x=origin_x,
                origin_y=origin_y,
                panel_width=panel_width,
                panel_height=panel_height,
            )
        )
    lines.append("</svg>")
    return _validate_svg("".join(lines))


def render_outcome_audit_svg(bundle: AnalysisBundle) -> bytes:
    """Render outcome counts for default primary attempts only."""
    primary_claim_ids = {
        target.claim_id
        for target in bundle.targets
        if target.kind in PRIMARY_CLAIM_KINDS
    }
    attempts = tuple(
        attempt
        for attempt in bundle.attempts
        if attempt.role is AttemptRole.DEFAULT
        and attempt.claim_id in primary_claim_ids
        and attempt.outcome in _PRIMARY_OUTCOMES
    )
    counts = Counter(attempt.outcome for attempt in attempts)
    maximum = max(counts.values(), default=0)
    width = 820
    height = 150 + 42 * len(_PRIMARY_OUTCOMES)
    lines = _svg_header(
        title="Paper validation outcome audit",
        description=(
            f"Outcomes for {len(attempts)} default primary results in run "
            f"{bundle.manifest.run_id}. Sensitivities and metadata discrepancies are excluded."
        ),
        width=width,
        height=height,
        identifier="outcome-audit",
    )
    lines.extend(
        (
            '<text class="figure-title" x="28" y="38">Default primary outcomes</text>',
            f'<text class="tick" x="28" y="60">Run {_xml(bundle.manifest.run_id)}; '
            "sensitivities and metadata discrepancies excluded.</text>",
        )
    )
    for index, outcome in enumerate(_PRIMARY_OUTCOMES):
        count = counts[outcome]
        y = 86 + 42 * index
        bar_width = 0 if count == 0 else max(2, count * 360 // max(1, maximum))
        lines.extend(
            (
                f'<text class="axis-label" x="28" y="{y + 17}">'
                f"{_xml(outcome.value)}: {count}</text>",
                f'<rect x="350" y="{y}" width="360" height="24" rx="3" '
                'fill="#e6eaf0"/>',
                f'<rect x="350" y="{y}" width="{bar_width}" height="24" rx="3" '
                f'fill="{_COLORS[index]}"/>',
            )
        )
    lines.append("</svg>")
    return _validate_svg("".join(lines))


def render_figures(bundle: AnalysisBundle) -> tuple[tuple[str, bytes], ...]:
    """Render named SVG bytes strictly from one completed format-2 bundle."""
    grouped: defaultdict[str, list[PlotSeries]] = defaultdict(list)
    for series in bundle.plot_series:
        if series.paper_analog and series.points:
            grouped[series.figure].append(series)

    names = {"outcome-audit.svg": "reserved outcome audit"}
    rendered: list[tuple[str, bytes]] = [
        ("outcome-audit.svg", render_outcome_audit_svg(bundle))
    ]
    for figure in sorted(grouped):
        filename = f"{_slug(figure)}.svg"
        previous = names.setdefault(filename, figure)
        if previous != figure:
            raise ValueError(
                f"figure names {previous!r} and {figure!r} map to {filename!r}"
            )
        rendered.append(
            (filename, _render_paper_figure(bundle, figure, tuple(grouped[figure])))
        )
    return tuple(rendered)


__all__ = ["render_figures", "render_outcome_audit_svg"]

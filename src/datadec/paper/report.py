from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import orjson

from datadec.paper.figures import render_figures
from datadec.paper.models import (
    AnalysisBundle,
    AttemptResult,
    AttemptRole,
    MetadataDiscrepancy,
    PaperTarget,
    ValidationOutcome,
)

_PRIMARY_OUTCOMES = (
    ValidationOutcome.REPRODUCED,
    ValidationOutcome.APPROXIMATELY_REPRODUCED,
    ValidationOutcome.DIRECTIONALLY_CONSISTENT,
    ValidationOutcome.NOT_REPRODUCED,
    ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED,
    ValidationOutcome.DESCRIPTIVE_ONLY,
)
_COMPARISON_HEADER = (
    "| Claim | Attempt | Role | Paper target | Computed result | Difference | Outcome |\n"
    "| --- | --- | --- | --- | --- | ---: | --- |\n"
)


@dataclass(frozen=True, slots=True)
class RenderedBundleOutputs:
    report: bytes
    figures: tuple[tuple[str, bytes], ...]


def _json(value: object) -> str:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode()


def _escape(value: object) -> str:
    escaped: list[str] = []
    for character in str(value):
        if character == "\n":
            escaped.append("<br>")
        elif character in r"\|`*_[]<>!":
            escaped.append(f"\\{character}")
        else:
            escaped.append(character)
    return "".join(escaped)


def _json_cell(value: object) -> str:
    return _escape(_json(value))


def _compact_json_cell(value: object, *, limit: int = 240) -> str:
    rendered = _json(value)
    if len(rendered) <= limit:
        return _escape(rendered)
    if isinstance(value, dict):
        summary = f"object keys={','.join(sorted(str(key) for key in value))}"
    elif isinstance(value, (list, tuple)):
        summary = f"{type(value).__name__} items={len(value)}"
    else:
        summary = f"{type(value).__name__}"
    return _escape(f"{summary}; {len(rendered)} JSON chars; see attempts.json")


def _attempt_sort_key(attempt: AttemptResult) -> tuple[int, str]:
    return (0 if attempt.role is AttemptRole.DEFAULT else 1, attempt.attempt_id)


def _attempts_by_claim(
    bundle: AnalysisBundle,
) -> dict[str, tuple[AttemptResult, ...]]:
    grouped: defaultdict[str, list[AttemptResult]] = defaultdict(list)
    for attempt in bundle.attempts:
        grouped[attempt.claim_id].append(attempt)
    return {
        claim_id: tuple(sorted(attempts, key=_attempt_sort_key))
        for claim_id, attempts in grouped.items()
    }


def _default_attempt(
    attempts: tuple[AttemptResult, ...],
) -> AttemptResult | None:
    return next(
        (attempt for attempt in attempts if attempt.role is AttemptRole.DEFAULT), None
    )


def _is_unassessable(attempts: tuple[AttemptResult, ...]) -> bool:
    default = _default_attempt(attempts)
    return default is None or (
        default.outcome is ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
    )


def _comparison_row(target: PaperTarget, attempt: AttemptResult) -> str:
    difference = (
        "—"
        if attempt.unrounded_difference is None
        else _escape(f"{attempt.unrounded_difference:.12g}")
    )
    return (
        f"| {_escape(target.claim_id)} | {_escape(attempt.attempt_id)} | "
        f"{attempt.role.value} | {_json_cell(target.value)} | "
        f"{_compact_json_cell(attempt.computed_value)} | {difference} | "
        f"{attempt.outcome.value} |\n"
    )


def _sample(values: tuple[str, ...], *, limit: int = 2) -> str:
    if not values:
        return "none"
    shown = "; ".join(values[:limit])
    remaining = len(values) - min(len(values), limit)
    return shown if remaining == 0 else f"{shown}; +{remaining} more"


def _method_row(target: PaperTarget, attempt: AttemptResult) -> str:
    rows = sum(selection.selected_row_count for selection in attempt.row_selections)
    checkpoint_steps = tuple(
        selection.actual_step for selection in attempt.checkpoint_selections
    )
    unique_steps = tuple(sorted(set(checkpoint_steps)))
    if not checkpoint_steps:
        checkpoints = "none"
    elif len(checkpoint_steps) <= 6:
        checkpoints = ", ".join(str(step) for step in checkpoint_steps)
    else:
        checkpoints = (
            f"{len(checkpoint_steps)} selections; {len(unique_steps)} unique steps; "
            f"range={unique_steps[0]}..{unique_steps[-1]}"
        )
    counts = (
        f"rows={rows}; denominator={attempt.denominator}; "
        f"ties={attempt.target_ties}/{attempt.predicted_ties}"
    )
    missing = (
        "none"
        if not attempt.missing_groups
        else f"{len(attempt.missing_groups)}: {_sample(attempt.missing_groups)}"
    )
    return (
        f"| {_escape(target.claim_id)} | {_escape(attempt.comparison_rule_id)} v"
        f"{attempt.comparison_rule_version} | {_escape(', '.join(attempt.transformation_ids))} | "
        f"{_escape(counts)} | {_escape(checkpoints)} | {_escape(missing)} | "
        f"{_escape(', '.join(attempt.plot_series_ids) or 'none')} |\n"
    )


def _render_family(
    family: str,
    targets: tuple[PaperTarget, ...],
    attempts_by_claim: dict[str, tuple[AttemptResult, ...]],
) -> str:
    comparisons: list[str] = []
    methods: list[str] = []
    for target in targets:
        attempts = attempts_by_claim.get(target.claim_id, ())
        for attempt in attempts:
            comparisons.append(_comparison_row(target, attempt))
        default = _default_attempt(attempts)
        if default is not None:
            methods.append(_method_row(target, default))
    return (
        f"## Family: {_escape(family)}\n\n"
        f"{_COMPARISON_HEADER}{''.join(comparisons)}\n"
        "### Compact method and selection trace\n\n"
        "Sensitivity attempts remain separate comparison rows; full predicates, "
        "diagnostics, row keys, and sensitivity parents remain in `attempts.json`.\n\n"
        "| Claim | Rule | Transformations | Counts | Checkpoints | Missing groups | Series |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        f"{''.join(methods)}\n"
    )


def _render_unassessable(
    targets: tuple[PaperTarget, ...],
    attempts_by_claim: dict[str, tuple[AttemptResult, ...]],
) -> str:
    if not targets:
        return "## Unassessable from dd_parsed\n\nNone in this bundle.\n\n"
    rows: list[str] = []
    for target in targets:
        attempts = attempts_by_claim.get(target.claim_id, ())
        default = _default_attempt(attempts)
        if default is None:
            rows.append(
                f"| {_escape(target.claim_id)} | {_escape(target.family)} | "
                f"{_json_cell(target.value)} | No attempt result persisted | — |\n"
            )
            continue
        reason_parts = (*default.diagnostics, *default.limitations)
        reason = _sample(reason_parts) if reason_parts else "No diagnostic recorded"
        missing = (
            "none"
            if not default.missing_groups
            else f"{len(default.missing_groups)}: {_sample(default.missing_groups)}"
        )
        rows.append(
            f"| {_escape(target.claim_id)} | {_escape(target.family)} | "
            f"{_json_cell(target.value)} | {_escape(reason)} | "
            f"{_escape(missing)} |\n"
        )
    return (
        "## Unassessable from dd_parsed\n\n"
        "These targets lack a default assessable result in the persisted bundle.\n\n"
        "| Claim | Family | Paper target | Recorded reason | Missing groups |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"{''.join(rows)}\n"
    )


def _metadata_row(discrepancy: MetadataDiscrepancy) -> str:
    return (
        f"| {_escape(discrepancy.claim_id)} | {_escape(discrepancy.paper_locator)} | "
        f"{_json_cell(discrepancy.paper_value)} | "
        f"{_escape(discrepancy.metadata_source)} | "
        f"{_json_cell(discrepancy.metadata_value)} | {_escape(discrepancy.note)} |\n"
    )


def _render_metadata_appendix(bundle: AnalysisBundle) -> str:
    discrepancies = tuple(
        sorted(bundle.metadata_discrepancies, key=lambda item: item.claim_id)
    )
    if not discrepancies:
        body = "None recorded.\n"
    else:
        body = (
            "| Claim | Paper locator | Paper value | Metadata source | "
            "Available value | Note |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            + "".join(_metadata_row(item) for item in discrepancies)
        )
    return (
        "## Metadata discrepancies\n\n"
        "Metadata comparisons are descriptive and are excluded from empirical "
        f"outcome counts.\n\n{body}\n"
    )


def _render_traceability(bundle: AnalysisBundle) -> str:
    manifest = bundle.manifest
    identities = "".join(
        f"| Input | {_escape(identity.id)} | `{identity.sha256}` |\n"
        for identity in sorted(manifest.input_identities, key=lambda item: item.id)
    )
    identities += "".join(
        f"| Bundle | {_escape(identity.id)} | `{identity.sha256}` |\n"
        for identity in (
            manifest.targets_identity,
            manifest.attempts_identity,
            manifest.plot_series_identity,
        )
    )
    code_trace = (
        _escape(_json(manifest.code_trace.model_dump(mode="json")))
        if manifest.code_trace is not None
        else "not recorded"
    )
    runtime_trace = (
        _escape(_json(manifest.runtime_trace.model_dump(mode="json")))
        if manifest.runtime_trace is not None
        else "not recorded"
    )
    return (
        "## Traceability appendix\n\n"
        f"- Run format: {manifest.run_format}\n"
        f"- Run ID: `{_escape(manifest.run_id)}`\n"
        f"- Started: `{manifest.started_at.isoformat()}`\n"
        f"- Completed: `{manifest.completed_at.isoformat()}`\n"
        f"- Code trace: {code_trace}\n"
        f"- Runtime trace: {runtime_trace}\n\n"
        "| Kind | Identity | SHA-256 |\n"
        "| --- | --- | --- |\n"
        f"{identities}"
    )


def render_report(bundle: AnalysisBundle) -> str:
    """Render a finding-comparison report strictly from one format-3 bundle."""
    attempts_by_claim = _attempts_by_claim(bundle)
    targets = tuple(
        sorted(bundle.targets, key=lambda item: (item.family, item.claim_id))
    )
    unassessable = tuple(
        target
        for target in targets
        if _is_unassessable(attempts_by_claim.get(target.claim_id, ()))
    )
    assessable = tuple(target for target in targets if target not in unassessable)
    families: defaultdict[str, list[PaperTarget]] = defaultdict(list)
    for target in assessable:
        families[target.family].append(target)

    default_counts = Counter(
        attempt.outcome
        for attempt in bundle.attempts
        if attempt.role is AttemptRole.DEFAULT and attempt.outcome in _PRIMARY_OUTCOMES
    )
    summary = "".join(
        f"| {outcome.value} | {default_counts[outcome]} |\n"
        for outcome in _PRIMARY_OUTCOMES
    )
    family_sections = "".join(
        _render_family(
            family,
            tuple(sorted(family_targets, key=lambda item: item.claim_id)),
            attempts_by_claim,
        )
        for family, family_targets in sorted(families.items())
    )
    return (
        "# Paper finding validation report\n\n"
        f"- Run ID: `{_escape(bundle.manifest.run_id)}`\n"
        f"- Run format: {bundle.manifest.run_format}\n"
        "- Scientific source: persisted `targets.json`, `attempts.json`, and "
        "`plot-series.json` only.\n\n"
        "## Default primary outcome summary\n\n"
        "Sensitivity attempts and metadata discrepancies are excluded.\n\n"
        "| Outcome | Count |\n"
        "| --- | ---: |\n"
        f"{summary}\n"
        f"{family_sections}"
        f"{_render_unassessable(unassessable, attempts_by_claim)}"
        f"{_render_metadata_appendix(bundle)}"
        f"{_render_traceability(bundle)}"
    )


def render_bundle_outputs(bundle: AnalysisBundle) -> RenderedBundleOutputs:
    """Render report and named SVG bytes before any output replacement begins."""
    report = render_report(bundle).encode()
    figures = render_figures(bundle)
    return RenderedBundleOutputs(report=report, figures=figures)


__all__ = ["RenderedBundleOutputs", "render_bundle_outputs", "render_report"]

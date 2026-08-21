from __future__ import annotations

import hashlib
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import orjson

from datadec.paper.models import (
    ClaimRegistry,
    EvidenceBoundary,
    Observation,
    PaperClaim,
    RunManifest,
    Verdict,
)

_CANONICAL_JSON_OPTIONS = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SORT_KEYS
_VERDICT_ORDER = (
    Verdict.REPRODUCED,
    Verdict.CONTRADICTED,
    Verdict.INTERNALLY_INCONSISTENT,
    Verdict.SOURCE_ONLY_MATCH,
    Verdict.BLOCKED_MISSING_INPUT,
    Verdict.BLOCKED_UNSPECIFIED_METHOD,
    Verdict.EXTERNAL_OR_CITATION_DEPENDENT,
    Verdict.NOT_ATTEMPTED,
    Verdict.NOT_APPLICABLE,
)
_CLAIM_TABLE_HEADER = (
    "| ID | Static claim and locator | Expected | Observed / diagnostics | Verdict | "
    "Evidence boundary | Counts | Method / policy / verifier | Blocker | Artifacts |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)
_REPRODUCED_TABLE_HEADER = (
    "| ID and locator | Expected | Observed / diagnostics | Evidence boundary | "
    "Counts | Method / policy / verifier | Artifacts |\n"
    "| --- | --- | --- | --- | --- | --- | --- |\n"
)
_COMPACT_TABLE_HEADER = "| Group | Count | Claim IDs |\n| --- | ---: | --- |\n"
_CLAIM_ID_WRAP_WIDTH = 88


def _canonical_json(value: object) -> str:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode()


def _escape_table_cell(value: str) -> str:
    escaped: list[str] = []
    for character in value:
        if character == "\n":
            escaped.append("<br>")
        elif character in r"\\|`*_[]<>!":
            escaped.append(f"\\{character}")
        else:
            escaped.append(character)
    return "".join(escaped)


def _render_code_span(value: str) -> str:
    normalized = value.replace("\n", " ")
    longest_run = 0
    current_run = 0
    for character in normalized:
        if character == "`":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    delimiter = "`" * max(1, longest_run + 1)
    padding = " " if normalized.startswith("`") or normalized.endswith("`") else ""
    return f"{delimiter}{padding}{normalized}{padding}{delimiter}"


def _render_wrapped_claim_ids(claim_ids: Iterable[str]) -> str:
    lines: list[str] = []
    current: list[str] = []
    current_width = 0
    for claim_id in sorted(claim_ids):
        escaped = _escape_table_cell(claim_id)
        separator_width = 2 if current else 0
        if (
            current
            and current_width + separator_width + len(escaped) > _CLAIM_ID_WRAP_WIDTH
        ):
            lines.append(", ".join(current))
            current = []
            current_width = 0
        current.append(escaped)
        current_width += (2 if current_width else 0) + len(escaped)
    if current:
        lines.append(", ".join(current))
    return "<br>".join(lines) if lines else "—"


def _manifest_sha256(manifest: RunManifest) -> str:
    payload = orjson.dumps(
        manifest.model_dump(mode="json"), option=_CANONICAL_JSON_OPTIONS
    )
    return hashlib.sha256(payload).hexdigest()


def _validate_identity_join(
    claim: PaperClaim,
    observation: Observation,
    *,
    field: str,
    description: str,
) -> None:
    expected = getattr(claim, field)
    actual = getattr(observation, field)
    if expected is not None and actual != expected:
        raise ValueError(
            f"observation {claim.id} {description} ID does not match its claim: "
            f"expected {expected!r}, got {actual!r}"
        )


def _validated_claim_observations(
    registry: ClaimRegistry,
    manifest: RunManifest,
    observations: Iterable[Observation],
) -> tuple[tuple[PaperClaim, Observation], ...]:
    supplied = tuple(observations)
    supplied_ids = tuple(observation.claim_id for observation in supplied)
    duplicate_ids = sorted(
        claim_id for claim_id in set(supplied_ids) if supplied_ids.count(claim_id) > 1
    )
    if duplicate_ids:
        raise ValueError(
            "duplicate observations for claim IDs: " + ", ".join(duplicate_ids)
        )

    claims_by_id = {claim.id: claim for claim in registry.claims}
    observations_by_id = {observation.claim_id: observation for observation in supplied}
    claim_ids = set(claims_by_id)
    observation_ids = set(observations_by_id)
    missing_ids = sorted(claim_ids - observation_ids)
    unknown_ids = sorted(observation_ids - claim_ids)
    if missing_ids or unknown_ids:
        raise ValueError(
            "claim and observation IDs must match exactly: "
            f"missing={missing_ids!r}, unknown={unknown_ids!r}"
        )
    if manifest.observations_identity.observation_count != len(supplied):
        raise ValueError(
            "manifest observation count does not match supplied observations: "
            f"expected {manifest.observations_identity.observation_count}, "
            f"got {len(supplied)}"
        )

    input_ids = {identity.id for identity in manifest.input_identities}
    artifact_ids = {identity.id for identity in manifest.artifact_identities}
    result: list[tuple[PaperClaim, Observation]] = []
    for claim_id in sorted(claim_ids):
        claim = claims_by_id[claim_id]
        observation = observations_by_id[claim_id]
        _validate_identity_join(
            claim, observation, field="verifier_id", description="verifier"
        )
        _validate_identity_join(
            claim, observation, field="method_id", description="method"
        )
        _validate_identity_join(
            claim, observation, field="policy_id", description="policy"
        )

        unknown_inputs = sorted(set(observation.input_ids) - input_ids)
        if unknown_inputs:
            raise ValueError(
                f"observation {claim_id} references unknown inputs: "
                + ", ".join(unknown_inputs)
            )
        unknown_artifacts = sorted(set(observation.artifact_ids) - artifact_ids)
        if unknown_artifacts:
            raise ValueError(
                f"observation {claim_id} references unknown artifacts: "
                + ", ".join(unknown_artifacts)
            )
        blocker = observation.blocker
        if blocker is not None:
            present_missing_inputs = sorted(set(blocker.missing_input_ids) & input_ids)
            if present_missing_inputs:
                raise ValueError(
                    f"observation {claim_id} marks present inputs as missing: "
                    + ", ".join(present_missing_inputs)
                )
            if blocker.unresolved_method_id is not None:
                if blocker.unresolved_method_id != claim.unresolved_method_id:
                    raise ValueError(
                        f"observation {claim_id} unresolved method ID does not match "
                        f"its claim: expected {claim.unresolved_method_id!r}, "
                        f"got {blocker.unresolved_method_id!r}"
                    )
        result.append((claim, observation))
    return tuple(result)


def _render_observed(observation: Observation) -> str:
    return (
        f"value={_canonical_json(observation.observed_value)}; "
        f"diagnostics={_canonical_json(list(observation.diagnostics))}"
    )


def _render_counts(observation: Observation) -> str:
    values: list[str] = []
    if observation.denominator is not None:
        values.append(f"denominator={observation.denominator}")
    values.extend(f"{count.name}={count.value}" for count in observation.counts)
    return "; ".join(values) if values else "—"


def _render_method(observation: Observation) -> str:
    values = (
        ("method", observation.method_id),
        (
            "provenance",
            observation.method_provenance.value
            if observation.method_provenance is not None
            else None,
        ),
        ("method reference artifact", observation.method_reference_artifact_id),
        ("policy", observation.policy_id),
        ("verifier", observation.verifier_id),
    )
    rendered = [f"{name}={value}" for name, value in values if value is not None]
    return "; ".join(rendered) if rendered else "—"


def _render_blocker(observation: Observation) -> str:
    blocker = observation.blocker
    if blocker is None:
        return "—"
    values = [f"kind={blocker.kind.value}", f"reason={blocker.reason}"]
    if blocker.missing_input_ids:
        values.append(
            f"missing inputs={_canonical_json(list(blocker.missing_input_ids))}"
        )
    if blocker.unresolved_method_id is not None:
        values.append(f"unresolved method={blocker.unresolved_method_id}")
    return "; ".join(values)


def _render_claim_row(claim: PaperClaim, observation: Observation) -> str:
    locator = f"{claim.source_file}:{claim.line_start}-{claim.line_end}"
    actual_boundary = (
        observation.actual_evidence_boundary.value
        if observation.actual_evidence_boundary is not None
        else "none"
    )
    static_claim_cell = (
        f"{_escape_table_cell(claim.text)}<br>{_escape_table_cell(locator)}"
    )
    cells = (
        _escape_table_cell(claim.id),
        static_claim_cell,
        _canonical_json(claim.expectation),
        _render_observed(observation),
        observation.verdict.value,
        f"required={claim.required_evidence_boundary.value}; actual={actual_boundary}",
        _render_counts(observation),
        _render_method(observation),
        _render_blocker(observation),
        _canonical_json(list(observation.artifact_ids)),
    )
    escaped_cells = cells[:2] + tuple(_escape_table_cell(cell) for cell in cells[2:])
    return "| " + " | ".join(escaped_cells) + " |\n"


def _render_detailed_outcomes(
    claim_observations: tuple[tuple[PaperClaim, Observation], ...],
) -> str:
    rows = "".join(
        _render_claim_row(claim, observation)
        for claim, observation in claim_observations
        if observation.verdict
        in {Verdict.CONTRADICTED, Verdict.INTERNALLY_INCONSISTENT}
    )
    if not rows:
        rows = "None in the selected run.\n"
        header = ""
    else:
        header = _CLAIM_TABLE_HEADER
    return (
        "## Known contradictions and inconsistencies\n\n"
        "**These selected-run results contradict a claim or expose an internal "
        "inconsistency and must remain visible.**\n\n"
        f"{header}{rows}"
    )


def _render_reproduced(
    claim_observations: tuple[tuple[PaperClaim, Observation], ...],
) -> str:
    rows: list[str] = []
    for claim, observation in claim_observations:
        if observation.verdict is not Verdict.REPRODUCED:
            continue
        actual_boundary = (
            observation.actual_evidence_boundary.value
            if observation.actual_evidence_boundary is not None
            else "none"
        )
        cells = (
            f"{claim.id}; {claim.source_file}:{claim.line_start}-{claim.line_end}",
            _canonical_json(claim.expectation),
            _render_observed(observation),
            f"required={claim.required_evidence_boundary.value}; actual={actual_boundary}",
            _render_counts(observation),
            _render_method(observation),
            _canonical_json(list(observation.artifact_ids)),
        )
        rows.append(
            "| " + " | ".join(_escape_table_cell(cell) for cell in cells) + " |\n"
        )
    body = (
        _REPRODUCED_TABLE_HEADER + "".join(rows)
        if rows
        else "None in the selected run.\n"
    )
    return f"## Reproduced\n\n{body}"


def _render_compact_table(
    title: str,
    introduction: str,
    grouped_claim_ids: Iterable[tuple[str, tuple[str, ...]]],
) -> str:
    rows = "".join(
        f"| {_escape_table_cell(group)} | {len(claim_ids)} | "
        f"{_render_wrapped_claim_ids(claim_ids)} |\n"
        for group, claim_ids in grouped_claim_ids
    )
    body = _COMPACT_TABLE_HEADER + rows if rows else "None in the selected run.\n"
    return f"## {title}\n\n{introduction}\n\n{body}"


def _render_source_only(
    claim_observations: tuple[tuple[PaperClaim, Observation], ...],
) -> str:
    claim_ids = tuple(
        claim.id
        for claim, observation in claim_observations
        if observation.verdict is Verdict.SOURCE_ONLY_MATCH
    )
    groups = (
        (("source or author-artifact agreement only", claim_ids),) if claim_ids else ()
    )
    return _render_compact_table(
        "Source-only matches",
        "These results are not independent reproductions. Full recorded details are "
        "in the immutable observations file identified above.",
        groups,
    )


def _render_missing_inputs(
    claim_observations: tuple[tuple[PaperClaim, Observation], ...],
) -> str:
    grouped: defaultdict[tuple[tuple[str, ...], str], list[str]] = defaultdict(list)
    for claim, observation in claim_observations:
        if observation.verdict is Verdict.BLOCKED_MISSING_INPUT:
            assert observation.blocker is not None
            grouped[
                (observation.blocker.missing_input_ids, observation.blocker.reason)
            ].append(claim.id)
    groups = tuple(
        (
            f"missing inputs={_canonical_json(list(missing_ids))}; reason={reason}",
            tuple(grouped[(missing_ids, reason)]),
        )
        for missing_ids, reason in sorted(grouped)
    )
    return _render_compact_table(
        "Blocked: missing input",
        "Claims are grouped by the stable missing input IDs and recorded blocker reason.",
        groups,
    )


def _render_unspecified_methods(
    claim_observations: tuple[tuple[PaperClaim, Observation], ...],
) -> str:
    claim_ids_by_method: defaultdict[str, list[str]] = defaultdict(list)
    reasons_by_method: defaultdict[str, set[str]] = defaultdict(set)
    for claim, observation in claim_observations:
        if observation.verdict is Verdict.BLOCKED_UNSPECIFIED_METHOD:
            assert observation.blocker is not None
            method_id = observation.blocker.unresolved_method_id
            assert method_id is not None
            claim_ids_by_method[method_id].append(claim.id)
            reasons_by_method[method_id].add(observation.blocker.reason)
    groups = tuple(
        (
            f"unresolved method={method_id}; "
            f"reason(s)={_canonical_json(sorted(reasons_by_method[method_id]))}",
            tuple(claim_ids_by_method[method_id]),
        )
        for method_id in sorted(claim_ids_by_method)
    )
    return _render_compact_table(
        "Blocked: unspecified method",
        "Claims are grouped by unresolved method ID; recorded reasons remain visible "
        "for action.",
        groups,
    )


def _render_external(
    claim_observations: tuple[tuple[PaperClaim, Observation], ...],
) -> str:
    grouped: defaultdict[tuple[str, tuple[str, ...] | str], list[str]] = defaultdict(
        list
    )
    for claim, observation in claim_observations:
        if observation.verdict is Verdict.EXTERNAL_OR_CITATION_DEPENDENT:
            assert observation.blocker is not None
            key: tuple[str, tuple[str, ...] | str]
            if claim.citation_keys:
                key = ("citation keys", tuple(sorted(claim.citation_keys)))
            else:
                key = ("blocker reason", observation.blocker.reason)
            grouped[key].append(claim.id)
    groups = tuple(
        (
            f"{kind}={_canonical_json(list(value) if isinstance(value, tuple) else value)}",
            tuple(grouped[(kind, value)]),
        )
        for kind, value in sorted(grouped, key=lambda item: (item[0], str(item[1])))
    )
    return _render_compact_table(
        "External or citation-dependent",
        "Claims are grouped by citation keys when present, otherwise by the recorded "
        "external blocker reason.",
        groups,
    )


def _render_unattempted(
    claim_observations: tuple[tuple[PaperClaim, Observation], ...],
) -> str:
    grouped: defaultdict[tuple[Verdict, str], list[str]] = defaultdict(list)
    for claim, observation in claim_observations:
        if observation.verdict in {Verdict.NOT_ATTEMPTED, Verdict.NOT_APPLICABLE}:
            assert observation.blocker is not None
            grouped[(observation.verdict, observation.blocker.reason)].append(claim.id)
    groups = tuple(
        (
            f"verdict={verdict.value}; reason={reason}",
            tuple(grouped[(verdict, reason)]),
        )
        for verdict, reason in sorted(
            grouped, key=lambda item: (item[0].value, item[1])
        )
    )
    return _render_compact_table(
        "Not attempted or not applicable",
        "Claims are grouped by verdict and recorded reason.",
        groups,
    )


def render_report(
    registry: ClaimRegistry,
    manifest: RunManifest,
    observations: Iterable[Observation],
) -> str:
    """Render one validated paper-verification run without recomputing results."""
    claim_observations = _validated_claim_observations(registry, manifest, observations)
    verdict_counts = Counter(
        observation.verdict for _, observation in claim_observations
    )
    boundary_counts = Counter(
        observation.actual_evidence_boundary for _, observation in claim_observations
    )

    summary_rows = "".join(
        f"| Verdict | {verdict.value} | {verdict_counts[verdict]} |\n"
        for verdict in _VERDICT_ORDER
        if verdict_counts[verdict]
    )
    summary_rows += "".join(
        f"| Actual evidence boundary | {boundary.value} | "
        f"{boundary_counts[boundary]} |\n"
        for boundary in EvidenceBoundary
        if boundary_counts[boundary]
    )
    if boundary_counts[None]:
        summary_rows += (
            f"| Actual evidence boundary | none | {boundary_counts[None]} |\n"
        )

    code = manifest.code_identity
    dirty_diff = code.dirty_diff_artifact_id or "—"
    groups = "\n".join(
        render_group(claim_observations)
        for render_group in (
            _render_detailed_outcomes,
            _render_reproduced,
            _render_source_only,
            _render_missing_inputs,
            _render_unspecified_methods,
            _render_external,
            _render_unattempted,
        )
    )
    return (
        "# Paper verification report\n\n"
        f"- Paper identity: {_render_code_span(manifest.paper_identity.id)}\n"
        f"- Selected run ID: {_render_code_span(manifest.run_id)}\n"
        f"- Manifest SHA256: `{_manifest_sha256(manifest)}`\n\n"
        "## Pinned run identities\n\n"
        "| Identity | ID | Digest / state |\n"
        "| --- | --- | --- |\n"
        f"| Paper | {_escape_table_cell(manifest.paper_identity.id)} | "
        f"SHA256={manifest.paper_identity.sha256} |\n"
        f"| Reproduction config | {_escape_table_cell(manifest.config_identity.id)} | "
        f"SHA256={manifest.config_identity.sha256} |\n"
        f"| Claim registry | {_escape_table_cell(manifest.claims_identity.id)} | "
        f"SHA256={manifest.claims_identity.sha256} |\n"
        f"| Code | {code.commit_sha} | tree={code.tree_state.value}; "
        f"dirty diff artifact={_escape_table_cell(dirty_diff)} |\n"
        f"| Observations | "
        f"{_escape_table_cell(manifest.observations_identity.filename)} | "
        f"SHA256={manifest.observations_identity.sha256}; "
        f"count={manifest.observations_identity.observation_count} |\n\n"
        "## Evidence and method interpretation\n\n"
        "The required evidence boundary is the static claim target; the actual "
        "evidence boundary records what this selected run reached. Method provenance "
        "records whether a method is paper-derived, upstream-informed, or "
        "artifact-derived; provenance does not by itself establish independence. "
        "A `source_only_match` confirms only source or author-artifact agreement and "
        "is not an independent reproduction. Blocked and contradicted verdicts are "
        "successful scientific outcomes, not process failures. This report renders "
        "the selected observations as recorded and does not recompute scientific "
        "results. Full per-claim details remain in the immutable selected-run "
        f"observations identity: run {_render_code_span(manifest.run_id)}, file "
        f"{_render_code_span(manifest.observations_identity.filename)}, "
        f"SHA256 {_render_code_span(manifest.observations_identity.sha256)}, "
        f"{manifest.observations_identity.byte_count} bytes.\n\n"
        "## Summary counts\n\n"
        "| Dimension | Value | Count |\n"
        "| --- | --- | ---: |\n"
        f"{summary_rows}\n"
        f"{groups}"
    )


def render_report_file(
    registry: ClaimRegistry,
    manifest: RunManifest,
    observations: Iterable[Observation],
    output_path: str | Path,
) -> None:
    """Atomically replace one report file after rendering and validation succeed."""
    report = render_report(registry, manifest, observations)
    destination = Path(output_path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(report.encode())
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = ["render_report", "render_report_file"]

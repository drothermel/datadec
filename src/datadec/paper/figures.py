from __future__ import annotations

import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable
from html import escape
from pathlib import Path

from datadec.paper.models import (
    ClaimRegistry,
    Observation,
    PaperClaim,
    RunManifest,
    Verdict,
)

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
_SUITE_CLAIM_IDS = ("DD-0269", *(f"DD-{index:04d}" for index in range(276, 290)))
_MISMATCH_PATTERN = re.compile(
    r"^(?P<field>(?:suite fact )?[a-z][a-z0-9_ ]*): expected "
    r"(?P<expected>.+?), observed (?P<observed>.+)$"
)
_SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def _escape(value: object) -> str:
    return escape(str(value), quote=True)


def _validate_observations(
    manifest: RunManifest, observations: Iterable[Observation]
) -> tuple[Observation, ...]:
    supplied = tuple(observations)
    supplied_ids = tuple(observation.claim_id for observation in supplied)
    duplicate_ids = sorted(
        claim_id for claim_id in set(supplied_ids) if supplied_ids.count(claim_id) > 1
    )
    if duplicate_ids:
        raise ValueError(
            "duplicate observations for claim IDs: " + ", ".join(duplicate_ids)
        )
    expected_count = manifest.observations_identity.observation_count
    if expected_count != len(supplied):
        raise ValueError(
            "manifest observation count does not match supplied observations: "
            f"expected {expected_count}, got {len(supplied)}"
        )
    return supplied


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
    supplied = _validate_observations(manifest, observations)
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
            if (
                blocker.unresolved_method_id is not None
                and blocker.unresolved_method_id != claim.unresolved_method_id
            ):
                raise ValueError(
                    f"observation {claim_id} unresolved method ID does not match "
                    f"its claim: expected {claim.unresolved_method_id!r}, "
                    f"got {blocker.unresolved_method_id!r}"
                )
        result.append((claim, observation))
    return tuple(result)


def _svg_header(*, title: str, description: str, width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="{_SVG_NAMESPACE}" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="figure-title figure-description">',
        f'<title id="figure-title">{_escape(title)}</title>',
        f'<desc id="figure-description">{_escape(description)}</desc>',
        "<style>"
        "text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#172033}"
        ".heading{font-size:22px;font-weight:700}"
        ".label{font-size:13px;font-weight:600}"
        ".detail{font-size:12px;fill:#48566a}"
        ".meta{font-size:11px;fill:#5f6c80}"
        "</style>",
    ]


def _validate_svg(svg: str) -> str:
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as error:
        raise ValueError("rendered figure is not valid XML") from error
    if root.tag != f"{{{_SVG_NAMESPACE}}}svg":
        raise ValueError("rendered figure root must be SVG")
    return svg


def render_verdict_summary_svg(
    manifest: RunManifest, observations: Iterable[Observation]
) -> str:
    """Render recorded verdict counts without recomputing scientific results."""
    supplied = _validate_observations(manifest, observations)
    counts = Counter(observation.verdict for observation in supplied)
    maximum = max(counts.values(), default=0)
    width = 820
    height = 116 + 34 * len(_VERDICT_ORDER) + 48
    lines = _svg_header(
        title="Paper verification verdict summary",
        description=(
            f"Verdict counts recorded by selected run {manifest.run_id}; "
            "each bar is labeled with its verdict and count."
        ),
        width=width,
        height=height,
    )
    lines.extend(
        (
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<text class="heading" x="28" y="38">Recorded verdicts</text>',
            f'<text class="detail" x="28" y="60">Selected run ID: '
            f"{_escape(manifest.run_id)}</text>",
            f'<text class="meta" x="28" y="79">Observations SHA256: '
            f"{manifest.observations_identity.sha256}</text>",
        )
    )
    colors = ("#20639b", "#b23a48", "#8f4a64", "#3d7a57", "#9a6b1f")
    for index, verdict in enumerate(_VERDICT_ORDER):
        count = counts[verdict]
        y = 103 + 34 * index
        bar_width = 0 if count == 0 else max(2, count * 360 // max(1, maximum))
        lines.extend(
            (
                f'<text class="label" x="28" y="{y + 16}">'
                f"{verdict.value}: {count}</text>",
                f'<rect x="325" y="{y}" width="360" height="22" rx="3" '
                'fill="#e6eaf0"/>',
                f'<rect x="325" y="{y}" width="{bar_width}" height="22" rx="3" '
                f'fill="{colors[index % len(colors)]}"/>',
            )
        )
    lines.extend(
        (
            f'<text class="meta" x="28" y="{height - 20}">'
            "Counts are rendered from immutable observations; bar color is redundant."
            "</text>",
            "</svg>",
        )
    )
    return _validate_svg("".join(lines))


def _strip_recorded_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _mismatch_labels(observation: Observation) -> tuple[str, ...]:
    labels: list[str] = []
    for diagnostic in observation.diagnostics:
        match = _MISMATCH_PATTERN.fullmatch(diagnostic)
        if match is None:
            continue
        field = match.group("field").removeprefix("suite fact ").replace("_", " ")
        expected = _strip_recorded_quotes(match.group("expected"))
        observed = _strip_recorded_quotes(match.group("observed"))
        labels.append(f"{field}: expected {expected}; observed {observed}")
    if not labels:
        raise ValueError(
            f"contradicted suite observation {observation.claim_id} has no recorded "
            "field mismatch diagnostic"
        )
    return tuple(labels)


def render_suite_contradictions_svg(
    registry: ClaimRegistry,
    manifest: RunManifest,
    observations: Iterable[Observation],
) -> str:
    """Render recorded suite contradictions without consulting scientific inputs."""
    claim_observations = _validated_claim_observations(registry, manifest, observations)
    observations_by_id = {
        observation.claim_id: observation for _, observation in claim_observations
    }
    contradictions = tuple(
        observation
        for claim_id in _SUITE_CLAIM_IDS
        if (observation := observations_by_id.get(claim_id)) is not None
        and observation.verdict is Verdict.CONTRADICTED
    )
    rows = tuple(
        (observation, _mismatch_labels(observation)) for observation in contradictions
    )
    width = 940
    height = 176 + (66 * len(rows) if rows else 88)
    lines = _svg_header(
        title="Recorded suite contradictions",
        description=(
            f"Suite contradictions recorded by selected run {manifest.run_id}, "
            "including the actual evidence boundary for each claim."
        ),
        width=width,
        height=height,
    )
    lines.extend(
        (
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<text class="heading" x="28" y="38">Recorded suite contradictions</text>',
            f'<text class="detail" x="28" y="60">Selected run ID: '
            f"{_escape(manifest.run_id)}</text>",
            f'<text class="meta" x="28" y="79">Observations SHA256: '
            f"{manifest.observations_identity.sha256}</text>",
        )
    )
    if not rows:
        lines.extend(
            (
                '<rect x="28" y="104" width="884" height="70" rx="6" '
                'fill="#f2f5f8" stroke="#c6cfda"/>',
                '<text class="label" x="48" y="134">No suite contradictions '
                "recorded for the selected run.</text>",
                '<text class="detail" x="48" y="156">This is an explicit '
                "empty state, not an omitted analysis.</text>",
            )
        )
    else:
        for index, (observation, mismatch_labels) in enumerate(rows):
            y = 102 + 66 * index
            boundary = observation.actual_evidence_boundary
            actual_boundary = boundary.value if boundary is not None else "none"
            lines.extend(
                (
                    f'<rect x="28" y="{y}" width="884" height="56" rx="5" '
                    f'fill="{"#fff4f2" if index % 2 == 0 else "#f8eeec"}" '
                    'stroke="#d9b5af"/>',
                    f'<text class="label" x="44" y="{y + 21}">'
                    f"{_escape(observation.claim_id)} — "
                    f"{_escape('; '.join(mismatch_labels))}</text>",
                    f'<text class="detail" x="44" y="{y + 43}">Actual evidence '
                    f"boundary: {_escape(actual_boundary)}</text>",
                )
            )
    lines.extend(
        (
            f'<text class="meta" x="28" y="{height - 20}">Mismatch labels are '
            "parsed from recorded diagnostics; no catalog or table is consulted.</text>",
            "</svg>",
        )
    )
    return _validate_svg("".join(lines))


def render_figure_files(
    registry: ClaimRegistry,
    manifest: RunManifest,
    observations: Iterable[Observation],
    output_root: str | Path,
) -> None:
    """Render both SVGs before replacing each destination file."""
    supplied = tuple(observations)
    verdict_svg = render_verdict_summary_svg(manifest, supplied)
    contradictions_svg = render_suite_contradictions_svg(registry, manifest, supplied)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    rendered = (
        (destination / "verdict-summary.svg", verdict_svg),
        (destination / "suite-contradictions.svg", contradictions_svg),
    )
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for target, svg in rendered:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=destination,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(svg.encode())
                temporary.flush()
                os.fsync(temporary.fileno())
            temporary_paths.append((temporary_path, target))
        for temporary_path, target in temporary_paths:
            os.replace(temporary_path, target)
        temporary_paths.clear()
    finally:
        for temporary_path, _ in temporary_paths:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "render_figure_files",
    "render_suite_contradictions_svg",
    "render_verdict_summary_svg",
]

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import datadec.paper.report as report_module
from datadec.paper.models import (
    ClaimRegistry,
    CodeIdentity,
    CodeTreeState,
    ContentIdentity,
    Observation,
    ObservationFileIdentity,
    PaperClaim,
    RunManifest,
    RuntimeIdentity,
    Verdict,
)
from datadec.paper.report import render_report, render_report_file

_STARTED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def _identity(identity_id: str, digit: str) -> ContentIdentity:
    return ContentIdentity(id=identity_id, sha256=digit * 64)


def _claim(claim_id: str = "claim-1", **updates: Any) -> PaperClaim:
    value: dict[str, Any] = {
        "id": claim_id,
        "source_file": "docs/paper/example.tex",
        "line_start": 10,
        "line_end": 11,
        "text": f"Static text for {claim_id}.",
        "owner": "datadec_empirical",
        "expectation_kind": "literal",
        "expectation": "expected",
        "required_evidence_boundary": "aggregate_evaluation",
    }
    value.update(updates)
    return PaperClaim.model_validate(value)


def _observation(claim_id: str = "claim-1", **updates: Any) -> Observation:
    value: dict[str, Any] = {
        "claim_id": claim_id,
        "verdict": "reproduced",
        "actual_evidence_boundary": "aggregate_evaluation",
        "observed_value": {"value": 0.75},
    }
    value.update(updates)
    return Observation.model_validate(value)


def _manifest(observation_count: int = 1) -> RunManifest:
    return RunManifest(
        run_id="selected-run",
        started_at=_STARTED_AT,
        completed_at=_STARTED_AT + timedelta(minutes=2),
        paper_identity=_identity("arxiv:2504.11393v2", "1"),
        config_identity=_identity("configs/paper_reproduction.toml", "2"),
        claims_identity=_identity("docs/paper/claims.toml", "3"),
        code_identity=CodeIdentity(
            commit_sha="4" * 40,
            tree_state=CodeTreeState.CLEAN,
        ),
        runtime_identity=RuntimeIdentity(
            python_version="3.12.5",
            implementation="CPython",
            platform="test-platform",
            dependency_lock_sha256="5" * 64,
        ),
        input_identities=(_identity("evaluation-input", "6"),),
        artifact_identities=(
            _identity("method-reference", "7"),
            _identity("result-table", "8"),
        ),
        observations_identity=ObservationFileIdentity(
            filename="observations.json",
            sha256="9" * 64,
            byte_count=123,
            observation_count=observation_count,
        ),
    )


def _all_verdict_values() -> tuple[ClaimRegistry, RunManifest, tuple[Observation, ...]]:
    verdicts = tuple(Verdict)
    claims = []
    observations = []
    for index, verdict in enumerate(reversed(verdicts), start=1):
        claim_id = f"claim-{index:02d}"
        claim_updates: dict[str, Any] = {}
        observation_updates: dict[str, Any] = {"verdict": verdict.value}
        if verdict is Verdict.REPRODUCED:
            claim_updates = {
                "text": "A | B \\ C\ncontinues.",
                "expectation": "expected | value",
                "verifier_id": "verifier-1",
                "method_id": "method-1",
                "policy_id": "policy-1",
            }
            observation_updates.update(
                {
                    "verifier_id": "verifier-1",
                    "method_id": "method-1",
                    "method_provenance": "upstream_informed",
                    "method_reference_artifact_id": "method-reference",
                    "policy_id": "policy-1",
                    "observed_value": {"z": 2, "a": [False, None]},
                    "diagnostics": ["diagnostic | value"],
                    "denominator": 20,
                    "counts": [
                        {"name": "abstentions", "value": 1},
                        {"name": "excluded", "value": 2},
                        {"name": "failed_fits", "value": 3},
                        {"name": "ties", "value": 4},
                    ],
                    "input_ids": ["evaluation-input"],
                    "artifact_ids": ["method-reference", "result-table"],
                }
            )
        elif verdict is Verdict.SOURCE_ONLY_MATCH:
            observation_updates["actual_evidence_boundary"] = "paper_or_final_artifact"
        elif verdict in {
            Verdict.CONTRADICTED,
            Verdict.INTERNALLY_INCONSISTENT,
        }:
            if verdict is Verdict.CONTRADICTED:
                claim_updates.update(
                    {
                        "text": "A | B \\ C\ncontinues.",
                        "expectation": "contradicted | expectation",
                    }
                )
            observation_updates["diagnostics"] = ["recorded conflict"]
        elif verdict is Verdict.EXTERNAL_OR_CITATION_DEPENDENT:
            claim_updates["citation_keys"] = ("z-key", "citation_*key*")
            observation_updates.update(
                {
                    "actual_evidence_boundary": None,
                    "observed_value": None,
                    "blocker": {
                        "kind": verdict.value,
                        "reason": f"recorded reason for {verdict.value}",
                    },
                }
            )
        else:
            observation_updates.update(
                {
                    "actual_evidence_boundary": None,
                    "observed_value": None,
                }
            )
            blocker_kind = verdict.value
            blocker: dict[str, Any] = {
                "kind": blocker_kind,
                "reason": f"recorded reason for {verdict.value}",
            }
            if verdict is Verdict.BLOCKED_MISSING_INPUT:
                blocker["kind"] = "missing_input"
                blocker["missing_input_ids"] = ["missing-input"]
            elif verdict is Verdict.BLOCKED_UNSPECIFIED_METHOD:
                blocker["kind"] = "unspecified_method"
                blocker["unresolved_method_id"] = "unresolved-method"
                claim_updates["unresolved_method_id"] = "unresolved-method"
            observation_updates["blocker"] = blocker
        claims.append(_claim(claim_id, **claim_updates))
        observations.append(_observation(claim_id, **observation_updates))
    return (
        ClaimRegistry(claims=tuple(reversed(claims))),
        _manifest(len(observations)),
        tuple(observations),
    )


def test_report_covers_every_outcome_and_preserves_recorded_details() -> None:
    registry, manifest, observations = _all_verdict_values()

    report = render_report(registry, manifest, observations)

    for verdict in Verdict:
        assert f"| Verdict | {verdict.value} | 1 |" in report
    for heading in (
        "## Known contradictions and inconsistencies",
        "## Reproduced",
        "## Source-only matches",
        "## Blocked: missing input",
        "## Blocked: unspecified method",
        "## External or citation-dependent",
        "## Not attempted or not applicable",
    ):
        assert heading in report
    assert (
        "source_only_match` confirms only source or author-artifact agreement" in report
    )
    assert "is not an independent reproduction" in report
    assert "successful scientific outcomes, not process failures" in report
    assert "A \\| B \\\\ C<br>continues." in report
    assert '"contradicted \\| expectation"' in report
    assert '"expected \\| value"' in report
    assert '{"a":\\[false,null\\],"z":2}' in report
    assert "diagnostic \\| value" in report
    assert (
        "denominator=20; abstentions=1; excluded=2; failed\\_fits=3; ties=4" in report
    )
    assert "method=method-1; provenance=upstream\\_informed" in report
    assert 'missing inputs=\\["missing-input"\\]' in report
    assert "unresolved method=unresolved-method" in report
    assert 'citation keys=\\["citation\\_\\*key\\*","z-key"\\]' in report
    assert 'diagnostics=\\["recorded conflict"\\]' in report
    for index in range(1, 10):
        assert f"claim-{index:02d}" in report
    for index in range(1, 7):
        assert report.count(f"claim-{index:02d}") == 1
    assert render_report(registry, manifest, reversed(observations)) == report


def test_report_identity_and_summary_header_matches_golden() -> None:
    registry = ClaimRegistry(claims=(_claim(),))

    report = render_report(registry, _manifest(), (_observation(),))

    assert report.startswith(
        """# Paper verification report

- Paper identity: `arxiv:2504.11393v2`
- Selected run ID: `selected-run`
- Manifest SHA256: `c5fd0c8d021f06b44ff526c9586a001b4fd3dffff3d7a85be1b1237b2c7d565b`

## Pinned run identities

| Identity | ID | Digest / state |
| --- | --- | --- |
| Paper | arxiv:2504.11393v2 | SHA256=1111111111111111111111111111111111111111111111111111111111111111 |
| Reproduction config | configs/paper\\_reproduction.toml | SHA256=2222222222222222222222222222222222222222222222222222222222222222 |
| Claim registry | docs/paper/claims.toml | SHA256=3333333333333333333333333333333333333333333333333333333333333333 |
| Code | 4444444444444444444444444444444444444444 | tree=clean; dirty diff artifact=— |
| Observations | observations.json | SHA256=9999999999999999999999999999999999999999999999999999999999999999; count=1 |

## Evidence and method interpretation
"""
    )
    assert (
        """## Summary counts

| Dimension | Value | Count |
| --- | --- | ---: |
| Verdict | reproduced | 1 |
| Actual evidence boundary | aggregate_evaluation | 1 |
"""
        in report
    )


@pytest.mark.parametrize(
    ("registry", "manifest", "observations", "error"),
    [
        (
            ClaimRegistry(claims=(_claim(), _claim("claim-2"))),
            _manifest(),
            (_observation(),),
            "must match exactly",
        ),
        (
            ClaimRegistry(claims=(_claim(),)),
            _manifest(2),
            (_observation(), _observation()),
            "duplicate observations",
        ),
        (
            ClaimRegistry(claims=(_claim(),)),
            _manifest(2),
            (_observation(),),
            "observation count",
        ),
        (
            ClaimRegistry(claims=(_claim(verifier_id="expected-verifier"),)),
            _manifest(),
            (_observation(verifier_id="other-verifier"),),
            "verifier ID does not match",
        ),
        (
            ClaimRegistry(claims=(_claim(unresolved_method_id="expected-method"),)),
            _manifest(),
            (
                _observation(
                    verdict="blocked_unspecified_method",
                    actual_evidence_boundary=None,
                    observed_value=None,
                    blocker={
                        "kind": "unspecified_method",
                        "reason": "method is absent",
                        "unresolved_method_id": "other-method",
                    },
                ),
            ),
            "unresolved method ID does not match",
        ),
        (
            ClaimRegistry(claims=(_claim(),)),
            _manifest(),
            (_observation(artifact_ids=["unknown-artifact"]),),
            "unknown artifacts",
        ),
    ],
)
def test_report_rejects_unresolved_joins(
    registry: ClaimRegistry,
    manifest: RunManifest,
    observations: tuple[Observation, ...],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        render_report(registry, manifest, observations)


def test_report_does_not_recompute_scientific_outcomes() -> None:
    registry = ClaimRegistry(claims=(_claim(expectation=1),))
    observation = _observation(verdict="reproduced", observed_value=999)

    report = render_report(registry, _manifest(), (observation,))

    claim_row = next(
        line for line in report.splitlines() if line.startswith("| claim-1; ")
    )
    assert "| 1 | value=999; diagnostics=\\[\\] | required=" in claim_row


def test_large_source_only_report_grows_with_ids_not_repeated_details() -> None:
    claim_count = 442
    repeated_claim_text = "repeated claim prose " + "x" * 500
    repeated_diagnostic = "repeated diagnostic " + "y" * 500
    claims = tuple(
        _claim(f"claim-{index:04d}", text=repeated_claim_text)
        for index in range(claim_count)
    )
    observations = tuple(
        _observation(
            claim.id,
            verdict="source_only_match",
            actual_evidence_boundary="paper_or_final_artifact",
            observed_value=None,
            diagnostics=(repeated_diagnostic,),
        )
        for claim in claims
    )

    large_report = render_report(
        ClaimRegistry(claims=tuple(reversed(claims))),
        _manifest(claim_count),
        tuple(reversed(observations)),
    )
    single_report = render_report(
        ClaimRegistry(claims=(claims[0],)),
        _manifest(),
        (observations[0],),
    )

    additional_id_bytes = sum(len(claim.id) for claim in claims[1:])
    assert repeated_claim_text not in large_report
    assert repeated_diagnostic not in large_report
    assert len(large_report.encode()) < 30_000
    assert len(large_report) - len(single_report) < additional_id_bytes * 3
    assert "| source or author-artifact agreement only | 442 |" in large_report


def test_large_reproduced_fact_values_remain_in_observations() -> None:
    claim = _claim("claim-1")
    observation = _observation(
        "claim-1",
        verdict="reproduced",
        actual_evidence_boundary="aggregate_evaluation",
        observed_value=[{"fact": index, "value": "x" * 100} for index in range(50)],
        diagnostics=("every recorded fact satisfies the predicate",),
    )
    manifest = _manifest()

    report = render_report(ClaimRegistry(claims=(claim,)), manifest, (observation,))

    assert "50 recorded values; full values remain" in report
    assert "every recorded fact satisfies the predicate" in report
    assert '"fact":49' not in report


def test_external_groups_fall_back_to_escaped_blocker_reason() -> None:
    claims = (_claim("claim-a"), _claim("claim-b"))
    observations = tuple(
        _observation(
            claim.id,
            verdict="external_or_citation_dependent",
            actual_evidence_boundary=None,
            observed_value=None,
            blocker={
                "kind": "external_or_citation_dependent",
                "reason": "external *artifact* | unavailable",
            },
        )
        for claim in claims
    )

    report = render_report(ClaimRegistry(claims=claims), _manifest(2), observations)

    assert (
        '| blocker reason="external \\*artifact\\* \\| unavailable" | 2 | '
        "claim-a, claim-b |" in report
    )


def test_report_file_validates_before_atomic_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "report.md"
    destination.write_text("original")
    registry = ClaimRegistry(claims=(_claim(),))
    manifest = _manifest()
    observation = _observation()

    with pytest.raises(ValueError, match="must match exactly"):
        render_report_file(
            ClaimRegistry(claims=(_claim(), _claim("claim-2"))),
            manifest,
            (observation,),
            destination,
        )
    assert destination.read_text() == "original"
    assert sorted(tmp_path.iterdir()) == [destination]

    original_replace = os.replace
    replacement_contents: list[str] = []

    def inspect_replace(source: str | Path, target: str | Path) -> None:
        assert Path(target) == destination
        assert destination.read_text() == "original"
        replacement_contents.append(Path(source).read_text())
        original_replace(source, target)

    monkeypatch.setattr(report_module.os, "replace", inspect_replace)
    render_report_file(registry, manifest, (observation,), destination)

    assert destination.read_text() == replacement_contents[0]
    assert destination.read_text() == render_report(registry, manifest, (observation,))
    assert sorted(tmp_path.iterdir()) == [destination]


def test_report_file_preserves_original_when_atomic_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "report.md"
    destination.write_text("original")

    def fail_replace(source: str | Path, target: str | Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(report_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        render_report_file(
            ClaimRegistry(claims=(_claim(),)),
            _manifest(),
            (_observation(),),
            destination,
        )

    assert destination.read_text() == "original"
    assert sorted(tmp_path.iterdir()) == [destination]

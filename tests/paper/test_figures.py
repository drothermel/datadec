from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import datadec.paper.figures as figures_module
from datadec.paper.figures import (
    render_figure_files,
    render_suite_contradictions_svg,
    render_verdict_summary_svg,
)
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

_STARTED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def _identity(identity_id: str, digit: str) -> ContentIdentity:
    return ContentIdentity(id=identity_id, sha256=digit * 64)


def _claim(claim_id: str, **updates: Any) -> PaperClaim:
    value: dict[str, Any] = {
        "id": claim_id,
        "source_file": "docs/paper/example.tex",
        "line_start": 10,
        "line_end": 11,
        "text": f"Static text for {claim_id}.",
        "owner": "datadec_empirical",
        "expectation_kind": "literal",
        "expectation": "expected",
        "required_evidence_boundary": "training_rerun",
    }
    value.update(updates)
    return PaperClaim.model_validate(value)


def _observation(claim_id: str, **updates: Any) -> Observation:
    value: dict[str, Any] = {
        "claim_id": claim_id,
        "verdict": "reproduced",
        "actual_evidence_boundary": "aggregate_evaluation",
        "observed_value": {"value": 0.75},
    }
    value.update(updates)
    return Observation.model_validate(value)


def _manifest(observation_count: int) -> RunManifest:
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
        observations_identity=ObservationFileIdentity(
            filename="observations.json",
            sha256="9" * 64,
            byte_count=123,
            observation_count=observation_count,
        ),
    )


def _suite_contradiction(claim_id: str, diagnostic: str, **updates: Any) -> Observation:
    values: dict[str, Any] = {
        "verdict": "contradicted",
        "actual_evidence_boundary": "paper_or_final_artifact",
        "observed_value": {"recorded": True},
        "diagnostics": [
            diagnostic,
            "catalog-derived evidence is below the required training-rerun boundary",
        ],
    }
    values.update(updates)
    return _observation(claim_id, **values)


def test_verdict_summary_has_exact_labeled_counts_and_run_identity() -> None:
    observations = (
        _observation("DD-0001"),
        _suite_contradiction(
            "DD-0269", "suite fact sequence_length: expected '2024', observed '2048'"
        ),
        _suite_contradiction(
            "DD-0276", "training_steps: expected 5,725, observed 5,715"
        ),
    )

    svg = render_verdict_summary_svg(_manifest(3), reversed(observations))

    assert "reproduced: 1" in svg
    assert "contradicted: 2" in svg
    for verdict in Verdict:
        expected_count = (
            1
            if verdict is Verdict.REPRODUCED
            else 2
            if verdict is Verdict.CONTRADICTED
            else 0
        )
        assert f"{verdict.value}: {expected_count}" in svg
    assert "Selected run ID: selected-run" in svg
    assert f"Observations SHA256: {'9' * 64}" in svg
    assert 'role="img"' in svg
    ET.fromstring(svg)
    assert len(svg.encode()) < 8_000


def test_suite_contradictions_are_escaped_and_in_deterministic_claim_order() -> None:
    claims = tuple(_claim(claim_id) for claim_id in ("DD-0289", "DD-0269", "other"))
    observations = (
        _suite_contradiction(
            "DD-0289", "learning_rate: expected <2.1e-03>, observed 2.2e-03 & rising"
        ),
        _observation("other"),
        _suite_contradiction(
            "DD-0269", "suite fact sequence_length: expected '2024', observed '2048'"
        ),
    )

    svg = render_suite_contradictions_svg(
        ClaimRegistry(claims=claims), _manifest(3), observations
    )
    reversed_svg = render_suite_contradictions_svg(
        ClaimRegistry(claims=tuple(reversed(claims))),
        _manifest(3),
        reversed(observations),
    )

    assert reversed_svg == svg
    assert svg.index("DD-0269") < svg.index("DD-0289")
    assert "sequence length: expected 2024; observed 2048" in svg
    assert (
        "learning rate: expected &lt;2.1e-03&gt;; observed 2.2e-03 &amp; rising" in svg
    )
    assert svg.count("Actual evidence boundary: paper_or_final_artifact") == 2
    assert "other" not in svg
    ET.fromstring(svg)
    assert len(svg.encode()) < 8_000


def test_suite_contradictions_render_explicit_empty_state() -> None:
    registry = ClaimRegistry(claims=(_claim("DD-0269"), _claim("DD-0276")))
    observations = (_observation("DD-0269"), _observation("DD-0276"))

    svg = render_suite_contradictions_svg(registry, _manifest(2), observations)

    assert "No suite contradictions recorded for the selected run." in svg
    assert "explicit empty state" in svg
    assert "Actual evidence boundary:" not in svg


def test_figures_only_reflect_recorded_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ClaimRegistry(claims=(_claim("DD-0269"),))
    first = _suite_contradiction(
        "DD-0269", "suite fact sequence_length: expected '2024', observed '2048'"
    )
    second = _suite_contradiction(
        "DD-0269", "suite fact sequence_length: expected '2024', observed '4096'"
    )

    def fail_if_config_is_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("figure rendering must not read external configuration")

    monkeypatch.setattr("datadec.config.load_catalog", fail_if_config_is_read)
    first_svg = render_suite_contradictions_svg(registry, _manifest(1), (first,))
    second_svg = render_suite_contradictions_svg(registry, _manifest(1), (second,))

    assert first_svg != second_svg
    assert "observed 2048" in first_svg
    assert "observed 4096" in second_svg


def test_figures_reject_unresolved_claim_observation_join() -> None:
    registry = ClaimRegistry(claims=(_claim("DD-0269"), _claim("DD-0276")))

    with pytest.raises(ValueError, match="must match exactly"):
        render_suite_contradictions_svg(
            registry,
            _manifest(1),
            (
                _suite_contradiction(
                    "DD-0269",
                    "suite fact sequence_length: expected '2024', observed '2048'",
                ),
            ),
        )


def test_figure_files_validate_both_before_atomic_per_file_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verdict_path = tmp_path / "verdict-summary.svg"
    suite_path = tmp_path / "suite-contradictions.svg"
    unrelated_path = tmp_path / "keep.txt"
    verdict_path.write_text("old verdict")
    suite_path.write_text("old suite")
    unrelated_path.write_text("keep")
    registry = ClaimRegistry(claims=(_claim("DD-0269"),))
    observation = _suite_contradiction(
        "DD-0269", "suite fact sequence_length: expected '2024', observed '2048'"
    )

    with pytest.raises(ValueError, match="no recorded field mismatch"):
        render_figure_files(
            registry,
            _manifest(1),
            (_suite_contradiction("DD-0269", "unparseable diagnostic"),),
            tmp_path,
        )
    assert verdict_path.read_text() == "old verdict"
    assert suite_path.read_text() == "old suite"

    original_replace = os.replace
    replacements: list[tuple[Path, Path]] = []

    def inspect_replace(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        target_path = Path(target)
        assert source_path.parent == tmp_path
        assert target_path.read_text().startswith("old ")
        ET.fromstring(source_path.read_text())
        replacements.append((source_path, target_path))
        original_replace(source_path, target_path)

    monkeypatch.setattr(figures_module.os, "replace", inspect_replace)
    render_figure_files(registry, _manifest(1), (observation,), tmp_path)

    assert [target.name for _, target in replacements] == [
        "verdict-summary.svg",
        "suite-contradictions.svg",
    ]
    assert verdict_path.read_text().startswith("<svg")
    assert suite_path.read_text().startswith("<svg")
    assert unrelated_path.read_text() == "keep"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "keep.txt",
        "suite-contradictions.svg",
        "verdict-summary.svg",
    ]

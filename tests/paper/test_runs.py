from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypedDict

import orjson
import pytest
from pydantic import ValidationError

import datadec.paper.runs as runs_module
from datadec.paper import (
    BlockerKind,
    CodeIdentity,
    CodeTreeState,
    ContentIdentity,
    Observation,
    ObservationBlocker,
    ObservationCount,
    ObservationFileIdentity,
    RunBundle,
    RunManifest,
    RuntimeIdentity,
    Verdict,
    create_run_bundle,
    load_run_bundle,
    validate_run_qualification,
)

_STARTED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


class _CreateKwargs(TypedDict):
    run_id: str
    started_at: datetime
    completed_at: datetime
    paper_identity: ContentIdentity
    config_identity: ContentIdentity
    claims_identity: ContentIdentity
    code_identity: CodeIdentity
    runtime_identity: RuntimeIdentity
    active_claim_ids: list[str]
    observations: list[Observation]
    input_identities: list[ContentIdentity]
    artifact_identities: list[ContentIdentity]


def _identity(identity_id: str, digit: str = "a") -> ContentIdentity:
    return ContentIdentity(id=identity_id, sha256=digit * 64)


def _runtime() -> RuntimeIdentity:
    return RuntimeIdentity(
        python_version="3.12.5",
        implementation="CPython",
        platform="test-platform",
        dependency_lock_sha256="e" * 64,
    )


def _observation(claim_id: str = "DD-0001", **updates: Any) -> Observation:
    value: dict[str, Any] = {
        "claim_id": claim_id,
        "verifier_id": "verify-value",
        "method_id": "paper-method",
        "method_provenance": "paper_derived",
        "policy_id": "numeric-comparison",
        "actual_evidence_boundary": "aggregate_evaluation",
        "verdict": "reproduced",
        "observed_value": {"value": 0.75},
        "diagnostics": [],
        "denominator": 12,
        "counts": [
            {"name": "excluded", "value": 2},
            {"name": "included", "value": 12},
        ],
        "input_ids": ["evaluation-table"],
        "artifact_ids": ["result-table"],
    }
    value.update(updates)
    return Observation.model_validate(value)


def _create_kwargs(run_id: str = "run-001") -> _CreateKwargs:
    return {
        "run_id": run_id,
        "started_at": _STARTED_AT,
        "completed_at": _STARTED_AT + timedelta(minutes=2),
        "paper_identity": _identity("arxiv:2504.11393v2", "1"),
        "config_identity": _identity("configs/paper_reproduction.toml", "2"),
        "claims_identity": _identity("docs/paper/claims.toml", "3"),
        "code_identity": CodeIdentity(
            commit_sha="4" * 40, tree_state=CodeTreeState.CLEAN
        ),
        "runtime_identity": _runtime(),
        "active_claim_ids": ["DD-0001", "DD-0002"],
        "observations": [_observation("DD-0002"), _observation("DD-0001")],
        "input_identities": [_identity("evaluation-table", "5")],
        "artifact_identities": [_identity("result-table", "6")],
    }


def _create_bundle(runs_root: Path, values: _CreateKwargs) -> RunBundle:
    return create_run_bundle(
        runs_root,
        run_id=values["run_id"],
        started_at=values["started_at"],
        completed_at=values["completed_at"],
        paper_identity=values["paper_identity"],
        config_identity=values["config_identity"],
        claims_identity=values["claims_identity"],
        code_identity=values["code_identity"],
        runtime_identity=values["runtime_identity"],
        active_claim_ids=values["active_claim_ids"],
        observations=values["observations"],
        input_identities=values["input_identities"],
        artifact_identities=values["artifact_identities"],
    )


def test_create_and_load_run_bundle_is_terminal_canonical_and_ordered(
    tmp_path: Path,
) -> None:
    bundle = _create_bundle(tmp_path, _create_kwargs())
    run_directory = tmp_path / "run-001"

    assert run_directory.is_dir()
    assert tuple(item.claim_id for item in bundle.observations) == (
        "DD-0001",
        "DD-0002",
    )
    manifest_raw = orjson.loads((run_directory / "manifest.json").read_bytes())
    observations_raw = orjson.loads((run_directory / "observations.json").read_bytes())
    assert set(manifest_raw) == {
        "artifact_identities",
        "claims_identity",
        "code_identity",
        "complete",
        "completed_at",
        "config_identity",
        "input_identities",
        "observations_identity",
        "paper_identity",
        "run_format",
        "run_id",
        "runtime_identity",
        "started_at",
    }
    assert set(observations_raw) == {"observations"}
    assert set(observations_raw["observations"][0]) == {
        "actual_evidence_boundary",
        "artifact_ids",
        "blocker",
        "claim_id",
        "counts",
        "denominator",
        "diagnostics",
        "input_ids",
        "method_id",
        "method_provenance",
        "method_reference_artifact_id",
        "observed_value",
        "policy_id",
        "verdict",
        "verifier_id",
    }
    assert manifest_raw["run_format"] == 1
    assert manifest_raw["complete"] is True
    assert (run_directory / "observations.json").read_bytes().endswith(b"\n")
    assert (
        load_run_bundle(tmp_path, "run-001", active_claim_ids=["DD-0002", "DD-0001"])
        == bundle
    )


def test_observation_bytes_are_deterministic_apart_from_run_manifest_identity(
    tmp_path: Path,
) -> None:
    first = _create_kwargs("run-a")
    second = _create_kwargs("run-b")
    second["started_at"] = _STARTED_AT + timedelta(days=1)
    second["completed_at"] = _STARTED_AT + timedelta(days=1, minutes=2)

    _create_bundle(tmp_path / "first", first)
    _create_bundle(tmp_path / "second", second)

    first_observations = (
        tmp_path / "first" / "run-a" / "observations.json"
    ).read_bytes()
    second_observations = (
        tmp_path / "second" / "run-b" / "observations.json"
    ).read_bytes()
    assert first_observations == second_observations

    first_manifest = orjson.loads(
        (tmp_path / "first" / "run-a" / "manifest.json").read_bytes()
    )
    second_manifest = orjson.loads(
        (tmp_path / "second" / "run-b" / "manifest.json").read_bytes()
    )
    for field in ("run_id", "started_at", "completed_at"):
        first_manifest.pop(field)
        second_manifest.pop(field)
    assert first_manifest == second_manifest


@pytest.mark.parametrize(
    ("claim_ids", "observations", "error"),
    [
        (
            ["DD-0001"],
            [_observation(), _observation()],
            "duplicate observations",
        ),
        (["DD-0001", "DD-0002"], [_observation()], "missing active claims"),
        (["DD-0001"], [_observation("DD-9999")], "unknown claims"),
    ],
)
def test_create_rejects_nonbijective_claim_observations(
    tmp_path: Path,
    claim_ids: list[str],
    observations: list[Observation],
    error: str,
) -> None:
    kwargs = _create_kwargs()
    kwargs["active_claim_ids"] = claim_ids
    kwargs["observations"] = observations

    with pytest.raises(ValueError, match=error):
        _create_bundle(tmp_path, kwargs)

    assert list(tmp_path.iterdir()) == []


def test_observation_rejects_nonfinite_json() -> None:
    with pytest.raises(ValidationError, match="finite JSON"):
        _observation(observed_value={"nested": [float("nan")]})


@pytest.mark.parametrize(
    ("verdict", "blocker", "error"),
    [
        ("blocked_missing_input", None, "matching blocker"),
        (
            "blocked_missing_input",
            ObservationBlocker(
                kind=BlockerKind.UNSPECIFIED_METHOD,
                reason="method is absent",
                unresolved_method_id="method-gap",
            ),
            "matching blocker",
        ),
        (
            "blocked_unspecified_method",
            ObservationBlocker(
                kind=BlockerKind.MISSING_INPUT,
                reason="input is absent",
                missing_input_ids=("missing.csv",),
            ),
            "matching blocker",
        ),
    ],
)
def test_blocked_verdict_requires_matching_structured_blocker(
    verdict: str, blocker: ObservationBlocker | None, error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        _observation(
            verdict=verdict,
            actual_evidence_boundary=None,
            observed_value=None,
            blocker=blocker,
        )


def test_valid_missing_input_blocker_is_explicit() -> None:
    observation = _observation(
        verdict="blocked_missing_input",
        actual_evidence_boundary=None,
        observed_value=None,
        denominator=None,
        counts=[],
        input_ids=[],
        artifact_ids=[],
        blocker={
            "kind": "missing_input",
            "reason": "the evaluation table is unavailable",
            "missing_input_ids": ["evaluation-table"],
        },
    )

    assert observation.verdict is Verdict.BLOCKED_MISSING_INPUT
    assert observation.blocker is not None
    assert observation.blocker.kind is BlockerKind.MISSING_INPUT


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        (
            {
                "verdict": "reproduced",
                "actual_evidence_boundary": "paper_or_final_artifact",
            },
            "independently recomputed",
        ),
        (
            {
                "verdict": "source_only_match",
                "actual_evidence_boundary": "aggregate_evaluation",
            },
            "source-only match",
        ),
        (
            {"actual_evidence_boundary": None},
            "actual evidence boundary",
        ),
        (
            {"observed_value": None, "diagnostics": []},
            "observed value or diagnostics",
        ),
    ],
)
def test_evidence_and_verdict_invariants(updates: dict[str, Any], error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        _observation(**updates)


def test_create_refuses_existing_run_without_changing_it(tmp_path: Path) -> None:
    _create_bundle(tmp_path, _create_kwargs())
    manifest_path = tmp_path / "run-001" / "manifest.json"
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        _create_bundle(tmp_path, _create_kwargs())

    assert manifest_path.read_bytes() == original_manifest
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run-001"]


def test_create_tolerates_stale_prior_lock_without_changing_it(tmp_path: Path) -> None:
    stale_lock = tmp_path / ".run-001.lock"
    stale_lock.write_bytes(b"prior interrupted creator")

    _create_bundle(tmp_path, _create_kwargs())

    assert (tmp_path / "run-001").is_dir()
    assert stale_lock.read_bytes() == b"prior interrupted creator"


def test_same_id_creation_collision_preserves_winner_and_cleans_loser_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_rename = runs_module._rename_no_replace
    winner_values = _create_kwargs()
    winner_values["completed_at"] += timedelta(minutes=1)
    installing_winner = False

    def install_winner_then_rename(source: Path, destination: Path) -> None:
        nonlocal installing_winner
        if not installing_winner:
            installing_winner = True
            _create_bundle(tmp_path, winner_values)
        original_rename(source, destination)

    monkeypatch.setattr(runs_module, "_rename_no_replace", install_winner_then_rename)

    with pytest.raises(FileExistsError):
        _create_bundle(tmp_path, _create_kwargs())

    winner = load_run_bundle(tmp_path, "run-001")
    assert winner.manifest.completed_at == winner_values["completed_at"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run-001"]


def test_injected_write_failure_leaves_no_final_or_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write = runs_module._write_new_file

    def fail_manifest(path: Path, contents: bytes) -> None:
        if path.name == "manifest.json":
            raise OSError("injected manifest failure")
        original_write(path, contents)

    monkeypatch.setattr(runs_module, "_write_new_file", fail_manifest)

    with pytest.raises(OSError, match="injected manifest failure"):
        _create_bundle(tmp_path, _create_kwargs())

    assert list(tmp_path.iterdir()) == []


def test_load_detects_observation_tampering(tmp_path: Path) -> None:
    _create_bundle(tmp_path, _create_kwargs())
    observations_path = tmp_path / "run-001" / "observations.json"
    observations_path.write_bytes(
        observations_path.read_bytes().replace(b"0.75", b"0.76")
    )

    with pytest.raises(ValueError, match="SHA256"):
        load_run_bundle(tmp_path, "run-001")


def test_observation_cross_references_must_exist_in_manifest(tmp_path: Path) -> None:
    kwargs = _create_kwargs()
    kwargs["observations"] = [
        _observation("DD-0001", artifact_ids=[]),
        _observation("DD-0002", input_ids=["unknown-input"]),
    ]

    with pytest.raises(ValueError, match="unknown inputs"):
        _create_bundle(tmp_path, kwargs)


def test_dirty_tree_requires_canonical_diff_artifact_and_cannot_qualify(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="canonical diff artifact"):
        CodeIdentity(commit_sha="4" * 40, tree_state=CodeTreeState.DIRTY)

    kwargs = _create_kwargs()
    kwargs["code_identity"] = CodeIdentity(
        commit_sha="4" * 40,
        tree_state=CodeTreeState.DIRTY,
        dirty_diff_artifact_id="code.diff",
    )
    kwargs["artifact_identities"] = [
        _identity("code.diff", "7"),
        _identity("result-table", "6"),
    ]
    bundle = _create_bundle(tmp_path, kwargs)

    assert bundle.manifest.code_identity.tree_state is CodeTreeState.DIRTY
    with pytest.raises(ValueError, match="clean code tree"):
        validate_run_qualification(bundle.manifest)


def test_dirty_diff_identity_must_be_a_run_artifact() -> None:
    bundle_values = _create_kwargs()
    code_identity = CodeIdentity(
        commit_sha="4" * 40,
        tree_state=CodeTreeState.DIRTY,
        dirty_diff_artifact_id="code.diff",
    )

    with pytest.raises(ValidationError, match="must appear"):
        RunManifest(
            run_id=bundle_values["run_id"],
            started_at=bundle_values["started_at"],
            completed_at=bundle_values["completed_at"],
            paper_identity=bundle_values["paper_identity"],
            config_identity=bundle_values["config_identity"],
            claims_identity=bundle_values["claims_identity"],
            code_identity=code_identity,
            runtime_identity=bundle_values["runtime_identity"],
            input_identities=tuple(bundle_values["input_identities"]),
            artifact_identities=tuple(bundle_values["artifact_identities"]),
            observations_identity=ObservationFileIdentity(
                filename="observations.json",
                sha256="8" * 64,
                byte_count=1,
                observation_count=0,
            ),
        )


def test_clean_terminal_run_qualifies(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path, _create_kwargs())

    assert validate_run_qualification(bundle.manifest) is None


def test_verdict_values_are_pinned() -> None:
    assert {verdict.value for verdict in Verdict} == {
        "reproduced",
        "contradicted",
        "internally_inconsistent",
        "source_only_match",
        "blocked_missing_input",
        "blocked_unspecified_method",
        "external_or_citation_dependent",
        "not_attempted",
        "not_applicable",
    }
    assert {state.value for state in CodeTreeState} == {"clean", "dirty"}
    assert ObservationCount(name="included", value=1).value == 1

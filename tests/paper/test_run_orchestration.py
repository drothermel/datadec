from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import datadec.paper.run as run_module
from datadec.paper.models import (
    CodeIdentity,
    CodeTreeState,
    ContentIdentity,
    MethodProvenance,
    Verdict,
)
from datadec.paper.run import (
    build_observations,
    render_repository,
    run_repository,
    validate_repository,
)
from datadec.paper.verifiers.olmes import (
    CanonicalFinalSelection,
    CheckpointDecisionSummary,
    NormalizedOlmesVerification,
)

_REPOSITORY_ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def validation() -> run_module.RepositoryValidation:
    return validate_repository(_REPOSITORY_ROOT)


def test_current_repository_validation_is_complete_and_read_only() -> None:
    status_before = run_module._git_output(
        _REPOSITORY_ROOT,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    validation = validate_repository(_REPOSITORY_ROOT)

    status_after = run_module._git_output(
        _REPOSITORY_ROOT,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    assert status_after == status_before
    assert len(validation.registry.claims) == 442
    assert len(validation.coverage.claim_ids) == 442
    assert len(validation.citations.citation_keys) == 43
    assert len(validation.suite.rows) == 14


def test_first_run_builds_one_truthful_terminal_observation_per_claim(
    validation: run_module.RepositoryValidation,
) -> None:
    observations = build_observations(validation)
    by_id = {observation.claim_id: observation for observation in observations}

    assert len(observations) == len(by_id) == 442
    assert tuple(by_id) == tuple(sorted(by_id))
    assert Counter(observation.verdict for observation in observations) == {
        Verdict.SOURCE_ONLY_MATCH: 154,
        Verdict.BLOCKED_UNSPECIFIED_METHOD: 108,
        Verdict.BLOCKED_MISSING_INPUT: 64,
        Verdict.NOT_ATTEMPTED: 62,
        Verdict.EXTERNAL_OR_CITATION_DEPENDENT: 39,
        Verdict.CONTRADICTED: 15,
    }
    assert by_id["DD-0267"].verdict is Verdict.SOURCE_ONLY_MATCH
    assert by_id["DD-0269"].verdict is Verdict.CONTRADICTED
    assert by_id["DD-0270"].verdict is Verdict.SOURCE_ONLY_MATCH
    assert by_id["DD-0272"].verdict is Verdict.BLOCKED_MISSING_INPUT
    assert by_id["DD-0273"].verdict is Verdict.BLOCKED_MISSING_INPUT
    assert by_id["DD-0276"].verdict is Verdict.CONTRADICTED
    assert by_id["DD-0289"].verdict is Verdict.CONTRADICTED
    assert by_id["DD-0268"].verdict is (Verdict.EXTERNAL_OR_CITATION_DEPENDENT)

    suite_observation = by_id["DD-0267"]
    assert (
        suite_observation.verifier_id,
        suite_observation.method_id,
        suite_observation.method_provenance,
        suite_observation.policy_id,
    ) == (
        "suite_config",
        "suite_config_comparison",
        MethodProvenance.PAPER_DERIVED,
        "suite_config_v1",
    )
    assert by_id["DD-0272"].verifier_id == "suite_config"

    source_observation = by_id["DD-0001"]
    assert (
        source_observation.verifier_id,
        source_observation.method_id,
        source_observation.policy_id,
    ) == ("source_trace", "paper_source_trace", "source_coverage_v1")

    citation_observation = by_id["DD-0268"]
    assert (
        citation_observation.verifier_id,
        citation_observation.method_id,
        citation_observation.policy_id,
    ) == (
        "citation_trace",
        "external_citation_trace",
        "citation_trace_v1",
    )

    planned_olmes = by_id["DD-0002"]
    assert planned_olmes.verdict is Verdict.NOT_ATTEMPTED
    assert planned_olmes.verifier_id == "olmes_aggregate"
    assert planned_olmes.policy_id == "olmes_v1"


def test_source_only_diagnostics_do_not_claim_independent_reproduction(
    validation: run_module.RepositoryValidation,
) -> None:
    observation = next(
        value for value in build_observations(validation) if value.claim_id == "DD-0001"
    )

    assert observation.verdict is Verdict.SOURCE_ONLY_MATCH
    assert any("source locator" in diagnostic for diagnostic in observation.diagnostics)
    assert any("SHA256" in diagnostic for diagnostic in observation.diagnostics)
    assert any(
        "not an independent reproduction" in diagnostic
        for diagnostic in observation.diagnostics
    )


def test_abstract_single_scale_claim_requires_exact_summary_then_tolerance(
    validation: run_module.RepositoryValidation,
) -> None:
    without_input = next(
        value for value in build_observations(validation) if value.claim_id == "DD-0011"
    )
    assert without_input.verdict is Verdict.BLOCKED_MISSING_INPUT
    assert without_input.blocker is not None
    assert without_input.blocker.missing_input_ids == (
        "olmes-summary:params=150M:step=38157:metric=primary_metric",
    )

    summary = CheckpointDecisionSummary(
        model_size="150M",
        step=38_157,
        metric="primary_metric",
        mean_accuracy=0.8,
        sd_accuracy=0.0,
        seed_count=3,
        ddof=1,
        sd_denominator=2,
        percent_target_compute=15.0,
        attempts=(),
    )
    verification = NormalizedOlmesVerification(
        canonical_finals=CanonicalFinalSelection(scores=(), missing=()),
        target_ranking=None,
        seed_decisions=(),
        checkpoint_summaries=(summary,),
        noise_spread=(),
        missing=(),
        facts=(),
    )

    with_input = next(
        value
        for value in build_observations(
            validation,
            olmes_verification=verification,
            olmes_input_id="normalized-olmes-input",
        )
        if value.claim_id == "DD-0011"
    )
    assert with_input.verdict is Verdict.BLOCKED_UNSPECIFIED_METHOD
    assert with_input.input_ids == ("normalized-olmes-input",)
    assert with_input.blocker is not None
    assert "no numeric tolerance" in with_input.blocker.reason


def test_run_repository_rejects_a_dirty_tree_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git_output(root: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(root)
        if arguments[0] == "status":
            return " M src/datadec/paper/run.py"
        raise AssertionError(arguments)

    monkeypatch.setattr(run_module, "_git_output", fake_git_output)
    monkeypatch.setattr(
        run_module,
        "validate_repository",
        lambda root: pytest.fail("validation must not run for dirty code"),
    )

    with pytest.raises(ValueError, match="clean Git tree"):
        run_repository(_REPOSITORY_ROOT, "dirty-run")


def test_run_repository_captures_current_file_and_runtime_identities(
    validation: run_module.RepositoryValidation,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    code_identity = CodeIdentity(commit_sha="a" * 40, tree_state=CodeTreeState.CLEAN)

    def capture_bundle(runs_root: Path, **kwargs: Any) -> SimpleNamespace:
        captured["runs_root"] = runs_root
        captured.update(kwargs)
        return SimpleNamespace(manifest=SimpleNamespace(run_id=kwargs["run_id"]))

    monkeypatch.setattr(run_module, "_clean_code_identity", lambda root: code_identity)
    monkeypatch.setattr(run_module, "validate_repository", lambda root: validation)
    monkeypatch.setattr(run_module, "create_run_bundle", capture_bundle)
    empty_verification = NormalizedOlmesVerification(
        canonical_finals=CanonicalFinalSelection(scores=(), missing=()),
        target_ranking=None,
        seed_decisions=(),
        checkpoint_summaries=(),
        noise_spread=(),
        missing=(),
        facts=(),
    )
    monkeypatch.setattr(
        run_module,
        "verify_normalized_olmes_parquet",
        lambda path, policy: empty_verification,
    )
    normalized_path = tmp_path / "normalized.parquet"
    normalized_path.write_bytes(b"normalized OLMES fixture")

    bundle = run_repository(_REPOSITORY_ROOT, "identity-run", normalized_path)

    assert bundle.manifest.run_id == "identity-run"
    assert captured["code_identity"] == code_identity
    assert captured["paper_identity"] == ContentIdentity(
        id="arxiv:2504.11393v2",
        sha256=validation.contract.paper.archive_sha256,
    )
    assert (
        captured["config_identity"].sha256
        == hashlib.sha256(
            (_REPOSITORY_ROOT / "configs/paper_reproduction.toml").read_bytes()
        ).hexdigest()
    )
    assert (
        captured["claims_identity"].sha256
        == hashlib.sha256(
            (_REPOSITORY_ROOT / "docs/paper/claims.toml").read_bytes()
        ).hexdigest()
    )
    input_ids = {
        identity.id: identity.sha256 for identity in captured["input_identities"]
    }
    assert (
        input_ids["docs/paper/example_paper.tex"]
        == hashlib.sha256(
            (_REPOSITORY_ROOT / "docs/paper/example_paper.tex").read_bytes()
        ).hexdigest()
    )
    lock_digest = hashlib.sha256(
        (_REPOSITORY_ROOT / "uv.lock").read_bytes()
    ).hexdigest()
    assert input_ids["uv.lock"] == lock_digest
    assert (
        input_ids["normalized-olmes-input"]
        == hashlib.sha256(normalized_path.read_bytes()).hexdigest()
    )
    assert captured["runtime_identity"].dependency_lock_sha256 == lock_digest


def test_render_repository_validates_current_identities_and_uses_atomic_renderer(
    validation: run_module.RepositoryValidation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = SimpleNamespace(
        complete=True,
        code_identity=CodeIdentity(
            commit_sha="a" * 40,
            tree_state=CodeTreeState.CLEAN,
        ),
        paper_identity=ContentIdentity(
            id="arxiv:2504.11393v2",
            sha256=validation.contract.paper.archive_sha256,
        ),
        config_identity=run_module._file_identity(
            _REPOSITORY_ROOT, "configs/paper_reproduction.toml"
        ),
        claims_identity=run_module._file_identity(
            _REPOSITORY_ROOT, "docs/paper/claims.toml"
        ),
    )
    bundle = SimpleNamespace(manifest=manifest, observations=())
    rendered: dict[str, Any] = {}

    monkeypatch.setattr(run_module, "validate_repository", lambda root: validation)
    monkeypatch.setattr(run_module, "load_run_bundle", lambda *args, **kwargs: bundle)

    def capture_render(*args: Any) -> None:
        rendered["args"] = args

    monkeypatch.setattr(run_module, "render_report_file", capture_render)
    monkeypatch.setattr(
        run_module,
        "render_figure_files",
        lambda *args: rendered.setdefault("figure_args", args),
    )

    output = render_repository(_REPOSITORY_ROOT, "selected-run")

    assert output == _REPOSITORY_ROOT / "docs/paper-reproduction-report.md"
    assert rendered["args"] == (
        validation.registry,
        manifest,
        (),
        output,
    )
    assert rendered["figure_args"] == (
        validation.registry,
        manifest,
        (),
        _REPOSITORY_ROOT / "docs/paper/reproduced-figures",
    )

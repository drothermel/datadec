from __future__ import annotations

import hashlib
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import datadec.paper.output_transaction as output_transaction_module
import datadec.paper.run as run_module
from datadec.paper.analysis import MMLU_SUBJECTS, OLMES_NON_MMLU_TASKS, TiePolicy
from datadec.paper.models import (
    CodeIdentity,
    CodeTreeState,
    ContentIdentity,
    EvidenceBoundary,
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
    FactRow,
    FactStatus,
    FinalCheckpoint,
    MissingDataBehavior,
    NormalizedOlmesPolicy,
    NormalizedOlmesVerification,
    OlmesTaskGrouping,
    verify_normalized_olmes,
)

_REPOSITORY_ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def validation() -> run_module.RepositoryValidation:
    return validate_repository(_REPOSITORY_ROOT)


def _olmes_policy(
    *,
    prediction_seeds: tuple[str, ...] = (
        "prediction-1",
        "prediction-2",
        "prediction-3",
    ),
) -> NormalizedOlmesPolicy:
    return NormalizedOlmesPolicy(
        recipes=("a", "b"),
        target_size="1B",
        target_seeds=("target-1", "target-2", "target-3"),
        prediction_seeds=prediction_seeds,
        target_metric_column="primary_metric",
        proxy_metric_columns=("proxy_metric",),
        task_grouping=OlmesTaskGrouping(
            non_mmlu_tasks=OLMES_NON_MMLU_TASKS,
            mmlu_subjects=MMLU_SUBJECTS,
            mmlu_task_name="mmlu",
        ),
        final_checkpoints=(
            FinalCheckpoint(model_size="1B", step=100),
            FinalCheckpoint(model_size="150M", step=50),
        ),
        noise_size="150M",
        tie_policy=TiePolicy.COUNT_AS_INCORRECT,
        attempt_ddof=1,
        within_recipe_ddof=1,
        spread_ddof=1,
        missing_data_behavior=MissingDataBehavior.RECORD,
        parameter_count_column="exact_parameter_count",
        token_count_column="tokens",
        target_compute_denominator=6_000.0,
    )


def _real_olmes_verification(
    policy: NormalizedOlmesPolicy,
    *,
    missing_primary_checkpoint_value: bool = False,
) -> NormalizedOlmesVerification:
    rows: list[dict[str, object]] = []
    for recipe_index, recipe in enumerate(policy.recipes):
        for model_size, step, seeds, parameters, tokens in (
            ("1B", 100, policy.target_seeds, 100, 10),
            ("150M", 50, policy.prediction_seeds, 10, 20),
        ):
            for seed in seeds:
                for task in (*OLMES_NON_MMLU_TASKS, *MMLU_SUBJECTS):
                    primary_value: float | None = float(recipe_index)
                    if (
                        missing_primary_checkpoint_value
                        and model_size == "150M"
                        and recipe == "a"
                        and seed == policy.prediction_seeds[0]
                        and task == OLMES_NON_MMLU_TASKS[0]
                    ):
                        primary_value = None
                    rows.append(
                        {
                            "params": model_size,
                            "data": recipe,
                            "seed": seed,
                            "step": step,
                            "task": task,
                            "exact_parameter_count": parameters,
                            "tokens": tokens,
                            "primary_metric": primary_value,
                            "proxy_metric": float(recipe_index),
                        }
                    )
    return verify_normalized_olmes(pd.DataFrame(rows), policy)


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
    assert len(validation.registry.claims) == 455
    assert len(validation.coverage.claim_ids) == 455
    assert len(validation.citations.citation_keys) == 43
    assert len(validation.suite.rows) == 14


def test_repository_validation_uses_configured_entrypoint_for_source_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_entrypoint = "docs/paper/nondefault-entrypoint.tex"
    current_entrypoint = "docs/paper/example_paper.tex"
    seen_entrypoints: dict[str, str] = {}
    original_coverage = run_module.validate_source_coverage
    original_dependencies = run_module.scan_tex_dependencies
    original_citations = run_module.validate_citations

    monkeypatch.setattr(
        run_module,
        "_paper_entrypoint",
        lambda _contract: configured_entrypoint,
    )

    def record_coverage(
        root: str | Path,
        registry: Any,
        entrypoint: str,
    ) -> Any:
        seen_entrypoints["coverage"] = entrypoint
        return original_coverage(root, registry, current_entrypoint)

    def record_dependencies(root: str | Path, entrypoint: str) -> Any:
        seen_entrypoints["dependencies"] = entrypoint
        return original_dependencies(root, current_entrypoint)

    def record_citations(root: str | Path, entrypoint: str) -> Any:
        seen_entrypoints["citations"] = entrypoint
        return original_citations(root, current_entrypoint)

    monkeypatch.setattr(run_module, "validate_source_coverage", record_coverage)
    monkeypatch.setattr(run_module, "scan_tex_dependencies", record_dependencies)
    monkeypatch.setattr(run_module, "validate_citations", record_citations)

    validate_repository(_REPOSITORY_ROOT)

    assert seen_entrypoints == {
        "coverage": configured_entrypoint,
        "dependencies": configured_entrypoint,
        "citations": configured_entrypoint,
    }


def test_first_run_builds_one_truthful_terminal_observation_per_claim(
    validation: run_module.RepositoryValidation,
) -> None:
    observations = build_observations(validation)
    by_id = {observation.claim_id: observation for observation in observations}

    assert len(observations) == len(by_id) == 455
    assert tuple(by_id) == tuple(sorted(by_id))
    assert Counter(observation.verdict for observation in observations) == {
        Verdict.SOURCE_ONLY_MATCH: 167,
        Verdict.BLOCKED_UNSPECIFIED_METHOD: 108,
        Verdict.BLOCKED_MISSING_INPUT: 65,
        Verdict.NOT_ATTEMPTED: 61,
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
    assert by_id["DD-0045"].verdict is Verdict.BLOCKED_MISSING_INPUT
    assert by_id["DD-0045"].blocker is not None
    assert by_id["DD-0045"].blocker.missing_input_ids == ("normalized-olmes-input",)


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
    policy = _olmes_policy()
    test_validation = replace(validation, olmes_policy=policy)
    without_input = next(
        value
        for value in build_observations(test_validation)
        if value.claim_id == "DD-0011"
    )
    assert without_input.verdict is Verdict.BLOCKED_MISSING_INPUT
    assert without_input.blocker is not None
    assert without_input.blocker.missing_input_ids == (
        "olmes-summary:params=150M:step=50:metric=primary_metric",
    )

    verification = _real_olmes_verification(policy)
    assert any(
        summary.model_size == "150M"
        and summary.step == 50
        and summary.metric == "primary_metric"
        for summary in verification.checkpoint_summaries
    )

    with_input = next(
        value
        for value in build_observations(
            test_validation,
            olmes_verification=verification,
            olmes_input_id="normalized-olmes-input",
        )
        if value.claim_id == "DD-0011"
    )
    assert with_input.verdict is Verdict.BLOCKED_UNSPECIFIED_METHOD
    assert with_input.input_ids == ("normalized-olmes-input",)
    assert with_input.blocker is not None
    assert "no numeric tolerance" in with_input.blocker.reason

    incomplete = _real_olmes_verification(
        policy,
        missing_primary_checkpoint_value=True,
    )
    missing_exact_summary = next(
        value
        for value in build_observations(
            test_validation,
            olmes_verification=incomplete,
            olmes_input_id="normalized-olmes-input",
        )
        if value.claim_id == "DD-0011"
    )
    assert missing_exact_summary.verdict is Verdict.BLOCKED_MISSING_INPUT
    assert missing_exact_summary.blocker is not None
    assert missing_exact_summary.blocker.missing_input_ids == (
        "olmes-summary:params=150M:step=50:metric=primary_metric",
    )


def test_exact_olmes_fact_mapping_persists_complete_fact_evidence(
    validation: run_module.RepositoryValidation,
) -> None:
    policy = _olmes_policy()
    verification = _real_olmes_verification(policy)

    observations = build_observations(
        replace(validation, olmes_policy=policy),
        olmes_verification=verification,
        olmes_input_id="normalized-olmes-input",
    )
    by_id = {observation.claim_id: observation for observation in observations}
    observation = by_id["DD-0045"]

    assert observation.verdict is Verdict.REPRODUCED
    assert observation.actual_evidence_boundary is EvidenceBoundary.AGGREGATE_EVALUATION
    assert observation.method_provenance is MethodProvenance.PAPER_DERIVED
    assert observation.input_ids == ("normalized-olmes-input",)
    assert observation.denominator == 2
    assert {count.name: count.value for count in observation.counts} == {
        "exclusions": 0,
        "predicted_ties": 0,
        "seed_count": 3,
        "target_ties": 0,
    }
    assert isinstance(observation.observed_value, list)
    assert {fact["dimensions"]["metric"] for fact in observation.observed_value} == {
        "primary_metric",
        "proxy_metric",
    }
    assert all(
        fact["denominator"] == 3
        and fact["seed_count"] == 3
        and fact["exclusions"] == 0
        and fact["target_ties"] == 0
        and fact["predicted_ties"] == 0
        for fact in observation.observed_value
    )
    for unsupported_claim_id in ("DD-0002", "DD-0012", "DD-0107"):
        assert by_id[unsupported_claim_id].verdict is Verdict.NOT_ATTEMPTED
        assert by_id[unsupported_claim_id].input_ids == ()


def test_exact_olmes_fact_mapping_ignores_unrelated_missing_stages(
    validation: run_module.RepositoryValidation,
) -> None:
    policy = _olmes_policy()
    verification = _real_olmes_verification(policy)
    unrelated_missing = FactRow(
        fact="missing_input",
        status=FactStatus.MISSING,
        dimensions=(
            ("stage", "noise_spread"),
            ("model_size", "150M"),
            ("step", "50"),
            ("metric", "primary_metric"),
            ("recipe", ""),
            ("seed", ""),
            ("missing_tasks", "task-a"),
            ("missing_recipes", ""),
        ),
        value=None,
        denominator=0,
        exclusions=0,
        target_ties=0,
        predicted_ties=0,
        seed_count=0,
        input_evidence_boundary=EvidenceBoundary.AGGREGATE_EVALUATION,
    )

    observation = next(
        value
        for value in build_observations(
            replace(
                validation,
                olmes_policy=policy,
            ),
            olmes_verification=replace(
                verification,
                facts=(*verification.facts, unrelated_missing),
            ),
            olmes_input_id="normalized-olmes-input",
        )
        if value.claim_id == "DD-0045"
    )

    assert observation.verdict is Verdict.REPRODUCED


def test_exact_olmes_fact_mapping_blocks_incomplete_input_and_contradicts_value(
    validation: run_module.RepositoryValidation,
) -> None:
    policy = _olmes_policy()
    incomplete = _real_olmes_verification(
        policy,
        missing_primary_checkpoint_value=True,
    )
    blocked = next(
        value
        for value in build_observations(
            replace(validation, olmes_policy=policy),
            olmes_verification=incomplete,
            olmes_input_id="normalized-olmes-input",
        )
        if value.claim_id == "DD-0045"
    )
    assert blocked.verdict is Verdict.BLOCKED_MISSING_INPUT
    assert blocked.input_ids == ("normalized-olmes-input",)

    two_seed_policy = _olmes_policy(prediction_seeds=("prediction-1", "prediction-2"))
    contradicted = next(
        value
        for value in build_observations(
            replace(validation, olmes_policy=two_seed_policy),
            olmes_verification=_real_olmes_verification(two_seed_policy),
            olmes_input_id="normalized-olmes-input",
        )
        if value.claim_id == "DD-0045"
    )
    assert contradicted.verdict is Verdict.CONTRADICTED
    assert {count.name: count.value for count in contradicted.counts}["seed_count"] == 2


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


def test_render_repository_validates_current_identities_and_replaces_rendered_set(
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

    def capture_report(*args: Any) -> str:
        rendered["report_args"] = args
        return "new report"

    def capture_verdict(*args: Any) -> str:
        rendered["verdict_args"] = args
        return "new verdict"

    def capture_suite(*args: Any) -> str:
        rendered["suite_args"] = args
        return "new suite"

    monkeypatch.setattr(run_module, "render_report", capture_report)
    monkeypatch.setattr(run_module, "render_verdict_summary_svg", capture_verdict)
    monkeypatch.setattr(run_module, "render_suite_contradictions_svg", capture_suite)
    monkeypatch.setattr(
        run_module,
        "replace_output_set",
        lambda outputs: rendered.setdefault("outputs", tuple(outputs)),
    )

    output = render_repository(_REPOSITORY_ROOT, "selected-run")

    assert output == _REPOSITORY_ROOT / "docs/paper-reproduction-report.md"
    assert rendered["report_args"] == (
        validation.registry,
        manifest,
        (),
    )
    assert rendered["verdict_args"] == (manifest, ())
    assert rendered["suite_args"] == (
        validation.registry,
        manifest,
        (),
    )
    assert rendered["outputs"] == (
        (output, "new report"),
        (
            _REPOSITORY_ROOT / "docs/paper/reproduced-figures/verdict-summary.svg",
            "new verdict",
        ),
        (
            _REPOSITORY_ROOT / "docs/paper/reproduced-figures/suite-contradictions.svg",
            "new suite",
        ),
    )


def _stub_render_repository(
    validation: run_module.RepositoryValidation,
    repository_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_validation = replace(validation, repository_root=repository_root)
    identities = {
        "configs/paper_reproduction.toml": ContentIdentity(
            id="configs/paper_reproduction.toml", sha256="b" * 64
        ),
        "docs/paper/claims.toml": ContentIdentity(
            id="docs/paper/claims.toml", sha256="c" * 64
        ),
    }
    manifest = SimpleNamespace(
        complete=True,
        code_identity=CodeIdentity(
            commit_sha="a" * 40,
            tree_state=CodeTreeState.CLEAN,
        ),
        paper_identity=ContentIdentity(
            id=f"arxiv:{validation.contract.paper.arxiv_id}",
            sha256=validation.contract.paper.archive_sha256,
        ),
        config_identity=identities["configs/paper_reproduction.toml"],
        claims_identity=identities["docs/paper/claims.toml"],
    )
    bundle = SimpleNamespace(manifest=manifest, observations=())
    monkeypatch.setattr(run_module, "validate_repository", lambda root: test_validation)
    monkeypatch.setattr(run_module, "load_run_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        run_module,
        "_file_identity",
        lambda root, relative_path: identities[relative_path],
    )


def _write_original_render_set(repository_root: Path) -> tuple[Path, Path, Path]:
    report_path = repository_root / "docs/paper-reproduction-report.md"
    figures_root = repository_root / "docs/paper/reproduced-figures"
    verdict_path = figures_root / "verdict-summary.svg"
    suite_path = figures_root / "suite-contradictions.svg"
    figures_root.mkdir(parents=True)
    report_path.write_text("old report")
    verdict_path.write_text("old verdict")
    suite_path.write_text("old suite")
    return report_path, verdict_path, suite_path


def test_render_repository_validates_entire_set_before_destination_changes(
    validation: run_module.RepositoryValidation,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destinations = _write_original_render_set(tmp_path)
    _stub_render_repository(validation, tmp_path, monkeypatch)
    monkeypatch.setattr(run_module, "render_report", lambda *args: "new report")
    monkeypatch.setattr(
        run_module, "render_verdict_summary_svg", lambda *args: "new verdict"
    )

    def fail_late_validation(*args: Any) -> str:
        raise ValueError("injected suite validation failure")

    monkeypatch.setattr(
        run_module, "render_suite_contradictions_svg", fail_late_validation
    )
    monkeypatch.setattr(
        run_module,
        "replace_output_set",
        lambda outputs: pytest.fail("replacement must wait for every validation"),
    )

    with pytest.raises(ValueError, match="injected suite validation failure"):
        render_repository(tmp_path, "selected-run")

    assert tuple(path.read_text() for path in destinations) == (
        "old report",
        "old verdict",
        "old suite",
    )
    assert sorted(path.name for path in tmp_path.rglob("*.*")) == [
        "paper-reproduction-report.md",
        "suite-contradictions.svg",
        "verdict-summary.svg",
    ]


def test_render_repository_restores_complete_set_after_late_replace_failure(
    validation: run_module.RepositoryValidation,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destinations = _write_original_render_set(tmp_path)
    report_path, verdict_path, suite_path = destinations
    _stub_render_repository(validation, tmp_path, monkeypatch)
    monkeypatch.setattr(run_module, "render_report", lambda *args: "new report")
    monkeypatch.setattr(
        run_module, "render_verdict_summary_svg", lambda *args: "new verdict"
    )
    monkeypatch.setattr(
        run_module, "render_suite_contradictions_svg", lambda *args: "new suite"
    )
    original_replace = os.replace
    injected = False

    def fail_third_destination(source: str | Path, target: str | Path) -> None:
        nonlocal injected
        if Path(target) == suite_path and not injected:
            injected = True
            assert report_path.read_text() == "new report"
            assert verdict_path.read_text() == "new verdict"
            raise OSError("injected late replace failure")
        original_replace(source, target)

    monkeypatch.setattr(output_transaction_module.os, "replace", fail_third_destination)

    with pytest.raises(OSError, match="injected late replace failure"):
        render_repository(tmp_path, "selected-run")

    assert injected
    assert tuple(path.read_text() for path in destinations) == (
        "old report",
        "old verdict",
        "old suite",
    )
    assert sorted(path.name for path in tmp_path.rglob("*.*")) == [
        "paper-reproduction-report.md",
        "suite-contradictions.svg",
        "verdict-summary.svg",
    ]

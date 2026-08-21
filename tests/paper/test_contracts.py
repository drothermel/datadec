from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import datadec.config as config_module
from datadec.config import config_file, load_paper_reproduction_contract
from datadec.paper import (
    ClaimOwnership,
    ClaimRegistry,
    EvidenceBoundary,
    ExpectationKind,
    MethodProvenance,
    PaperClaim,
    SourceRegion,
    load_claim_registry,
    load_repository_claim_registry,
)

_SHA256 = "a" * 64


def _claim(**updates: Any) -> dict[str, Any]:
    claim: dict[str, Any] = {
        "id": "claim-1",
        "source_file": "docs/paper/example_paper.tex",
        "line_start": 10,
        "line_end": 11,
        "text": "The paper makes a testable claim.",
        "owner": "datadec_empirical",
        "expectation_kind": "literal",
        "expectation": "expected",
        "required_evidence_boundary": "aggregate_evaluation",
        "input_refs": ["configs/olmes.toml"],
        "prerequisite_claim_ids": [],
        "paper_elements": ["section:introduction"],
        "citation_keys": [],
    }
    claim.update(updates)
    return claim


def _region(**updates: Any) -> dict[str, Any]:
    region: dict[str, Any] = {
        "id": "region-1",
        "source_file": "docs/paper/example_paper.tex",
        "line_start": 10,
        "line_end": 12,
        "kind": "prose",
        "content_sha256": _SHA256,
        "claim_ids": ["claim-1"],
    }
    region.update(updates)
    return region


def test_current_paper_reproduction_contract_pins_sources_and_boundaries() -> None:
    contract = load_paper_reproduction_contract()

    assert config_file("paper_reproduction.toml").is_file()
    assert contract.paper.arxiv_id == "2504.11393v2"
    assert contract.paper.archive_sha256 == (
        "20dc7aa3f920fe465ddf2e12d6f72fff6e8bb3567f53e34f5555a6da138542d1"
    )
    assert contract.paper.source_root == "docs/paper"
    assert contract.paper.entrypoint == "example_paper.tex"
    assert contract.contracts.model_dump() == {
        "catalog": "configs/catalog.toml",
        "sources": "configs/sources.toml",
        "olmes": "configs/olmes.toml",
        "scaling_law": "configs/scaling_law.toml",
        "published_results": "configs/published_results.toml",
        "claims_contract": "docs/paper/claims.toml",
    }
    assert {method.provenance for method in contract.methods} == {
        MethodProvenance.PAPER_DERIVED,
        MethodProvenance.ARTIFACT_DERIVED,
    }
    policy_statuses = {policy.id: policy.status for policy in contract.policies}
    assert policy_statuses["external_citation_scope"] == "settled"
    assert policy_statuses["comparison_universe"] == "unresolved"
    assert policy_statuses["statistical_fit"] == "unresolved"
    assert contract.outputs.runs_root == "data/paper-reproduction/runs"
    assert contract.outputs.report == "docs/paper-reproduction-report.md"

    with pytest.raises(ValidationError, match="frozen"):
        contract.paper.entrypoint = "other.tex"


def test_paper_models_reject_extra_fields_and_enums_are_exact() -> None:
    raw = load_paper_reproduction_contract().model_dump()
    raw["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(load_paper_reproduction_contract()).model_validate(raw)

    assert {owner.value for owner in ClaimOwnership} == {
        "datadec_empirical",
        "method_design",
        "artifact_release",
        "qualitative_interpretation",
        "external_citation",
    }
    assert {boundary.value for boundary in EvidenceBoundary} == {
        "paper_or_final_artifact",
        "author_downstream_table",
        "aggregate_evaluation",
        "instance_and_choice",
        "evaluation_rerun",
        "training_rerun",
        "corpus_construction",
    }
    assert {kind.value for kind in ExpectationKind} == {
        "literal",
        "numeric",
        "predicate",
        "citation_trace",
    }


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        (
            {"owner": "external_citation", "expectation_kind": "citation_trace"},
            "must include citation keys",
        ),
        (
            {"verifier_id": "verify-1", "unresolved_method_id": "method-gap-1"},
            "mutually exclusive",
        ),
        ({"line_start": 12, "line_end": 11}, "line_end"),
        ({"unknown": "value"}, "Extra inputs are not permitted"),
    ],
)
def test_claim_contract_rejects_invalid_persisted_claims(
    updates: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        PaperClaim.model_validate(_claim(**updates))


def test_external_citation_claim_requires_and_preserves_citation_keys() -> None:
    claim = PaperClaim.model_validate(
        _claim(
            owner="external_citation",
            expectation_kind="citation_trace",
            expectation="citation resolves to the attributed work",
            citation_keys=["brown2020language"],
        )
    )

    assert claim.citation_keys == ("brown2020language",)
    assert claim.owner is ClaimOwnership.EXTERNAL_CITATION


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"claim_ids": [], "non_claim_reason": None}, "exactly one"),
        (
            {"claim_ids": ["claim-1"], "non_claim_reason": "section heading"},
            "exactly one",
        ),
        ({"line_start": 13, "line_end": 12}, "line_end"),
    ],
)
def test_source_region_requires_claims_xor_non_claim_reason(
    updates: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        SourceRegion.model_validate(_region(**updates))

    non_claim = SourceRegion.model_validate(
        _region(claim_ids=[], non_claim_reason="formatting-only command")
    )
    assert non_claim.non_claim_reason == "formatting-only command"


@pytest.mark.parametrize(
    ("registry", "error"),
    [
        (
            {"claims": [_claim(), _claim()], "source_regions": []},
            "claim IDs must be unique",
        ),
        (
            {
                "claims": [_claim()],
                "source_regions": [_region(), _region()],
            },
            "region IDs must be unique",
        ),
        (
            {
                "claims": [_claim()],
                "source_regions": [_region(claim_ids=["missing-claim"])],
            },
            "reference unknown claims",
        ),
    ],
)
def test_claim_registry_rejects_duplicate_and_unknown_references(
    registry: dict[str, Any], error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        ClaimRegistry.model_validate(deepcopy(registry))


def test_claim_registry_loads_supplied_and_repository_paths(tmp_path: Path) -> None:
    contract_text = """
[[claims]]
id = "claim-1"
source_file = "docs/paper/example_paper.tex"
line_start = 10
line_end = 11
text = "The paper makes a testable claim."
owner = "datadec_empirical"
expectation_kind = "literal"
expectation = "expected"
required_evidence_boundary = "aggregate_evaluation"
"""
    supplied_path = tmp_path / "supplied-claims.toml"
    supplied_path.write_text(contract_text)
    repository_path = tmp_path / "repository" / "docs" / "paper"
    repository_path.mkdir(parents=True)
    (repository_path / "claims.toml").write_text(contract_text)

    supplied = load_claim_registry(supplied_path)
    repository = load_repository_claim_registry(tmp_path / "repository")

    assert supplied == repository
    assert supplied.source_regions == ()
    assert supplied.claims[0].id == "claim-1"


def test_config_file_prefers_packaged_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "datadec"
    packaged_config = package_root / "configs" / "paper_reproduction.toml"
    packaged_config.parent.mkdir(parents=True)
    packaged_config.write_text("packaged = true\n")
    source_root = tmp_path / "source-configs"
    source_root.mkdir()
    (source_root / "paper_reproduction.toml").write_text("source = true\n")
    monkeypatch.setattr(config_module, "files", lambda _package: package_root)
    monkeypatch.setattr(config_module, "_SOURCE_CONFIGS_DIR", source_root)

    assert config_file("paper_reproduction.toml") == packaged_config

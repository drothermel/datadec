from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from datadec.paper import (
    ClaimKind,
    ClaimRegistry,
    PaperClaim,
    ValidationOutcome,
    load_repository_claim_registry,
    load_validation_contract,
    validate_cross_contracts,
)

_REPOSITORY_ROOT = Path(__file__).parents[2]


def _validation_contract():
    return load_validation_contract(_REPOSITORY_ROOT / "configs/paper_validation.toml")


def _registry_for_current_contract() -> ClaimRegistry:
    claims = tuple(
        PaperClaim(
            id=attempt.claim_id,
            source_file="docs/paper/example_paper.tex",
            line_start=index,
            line_end=index,
            text=f"Synthetic claim {attempt.claim_id}",
            kind=ClaimKind.EMPIRICAL_COMPARISON,
            family="synthetic",
            paper_target=True,
            attempt_ids=(attempt.id,),
        )
        for index, attempt in enumerate(_validation_contract().attempts, start=1)
    )
    return ClaimRegistry(format_version=2, claims=claims)


def test_current_attempt_references_validate_against_exact_claim_ids() -> None:
    validate_cross_contracts(_registry_for_current_contract(), _validation_contract())


def test_cross_contract_validation_rejects_attempt_for_unknown_claim() -> None:
    contract = _validation_contract()
    bad_attempt = contract.attempts[0].model_copy(update={"claim_id": "DD-9999"})
    invalid_contract = contract.model_copy(
        update={"attempts": (bad_attempt, *contract.attempts[1:])}
    )

    with pytest.raises(ValueError, match="unknown claim"):
        validate_cross_contracts(_registry_for_current_contract(), invalid_contract)


def test_cross_contract_validation_requires_one_default_per_assessable_claim() -> None:
    contract = _validation_contract()
    nondefault = contract.attempts[0].model_copy(update={"default": False})
    invalid_contract = contract.model_copy(
        update={"attempts": (nondefault, *contract.attempts[1:])}
    )

    with pytest.raises(ValueError, match="requires one default attempt"):
        validate_cross_contracts(_registry_for_current_contract(), invalid_contract)


def test_claim_registry_rejects_unknown_method_dependency() -> None:
    claim = PaperClaim(
        id="DD-0001",
        source_file="docs/paper/example_paper.tex",
        line_start=1,
        line_end=1,
        text="Claim",
        kind=ClaimKind.METHOD_DEFINITION,
        family="method",
        supporting_outcome=ValidationOutcome.DESCRIPTIVE_ONLY,
        method_dependency_claim_ids=("DD-9999",),
    )

    with pytest.raises(ValueError, match="unknown claims"):
        ClaimRegistry(format_version=2, claims=(claim,))


def test_migrated_repository_registry_and_config_are_cross_validated() -> None:
    with (_REPOSITORY_ROOT / "docs/paper/claims.toml").open("rb") as file:
        raw = tomllib.load(file)
    if raw.get("format_version") != 2:
        pytest.skip("claims migration is integrated separately")

    registry = load_repository_claim_registry()
    contract = _validation_contract()

    validate_cross_contracts(registry, contract)
    primary = {
        ClaimKind.EMPIRICAL_NUMERIC,
        ClaimKind.EMPIRICAL_COMPARISON,
        ClaimKind.EMPIRICAL_TREND,
        ClaimKind.EMPIRICAL_PLOT,
    }
    assert len([claim for claim in registry.claims if claim.kind in primary]) == 79
    assert len(contract.attempts) == 68

from __future__ import annotations

import tomllib
from collections import Counter
from pathlib import Path

import pytest

from datadec.paper import (
    ClaimRegistry,
    PaperReproductionContract,
    VerifierId,
    load_repository_claim_registry,
    validate_static_references,
)

_REPOSITORY_ROOT = Path(__file__).parents[2]


def _load_reproduction_contract() -> PaperReproductionContract:
    with (_REPOSITORY_ROOT / "configs/paper_reproduction.toml").open("rb") as file:
        return PaperReproductionContract.model_validate(tomllib.load(file))


def test_verifier_registry_is_closed_unique_and_not_callable_paths() -> None:
    assert {verifier.value for verifier in VerifierId} == {
        "source_trace",
        "citation_trace",
        "suite_config",
        "olmes_aggregate",
        "author_artifact",
        "artifact_inventory",
        "olmes_choice",
        "scaling_law",
    }
    assert all("." not in verifier.value for verifier in VerifierId)


def test_current_static_references_are_resolved_and_intentionally_covered() -> None:
    registry = load_repository_claim_registry()
    contract = _load_reproduction_contract()

    validate_static_references(registry, contract)

    counts = Counter(claim.verifier_id for claim in registry.claims)
    assert counts == {
        "source_trace": 159,
        "citation_trace": 39,
        "suite_config": 21,
        "olmes_aggregate": 46,
        "author_artifact": 9,
        "artifact_inventory": 43,
        None: 138,
    }
    assert (
        sum(count for verifier, count in counts.items() if verifier is not None) == 317
    )
    assert not any(
        claim.verifier_id is not None and claim.unresolved_method_id is not None
        for claim in registry.claims
    )
    assert not any(
        claim.verifier_id in {VerifierId.OLMES_CHOICE, VerifierId.SCALING_LAW}
        for claim in registry.claims
    )


@pytest.mark.parametrize(
    ("field", "reference", "error"),
    [
        ("verifier_id", "missing_verifier", "unknown verifier"),
        ("method_id", "missing_method", "unknown method"),
        ("policy_id", "missing_policy", "unknown policy"),
        ("verifier_id", "package.verify", "dotted callable"),
    ],
)
def test_static_reference_validation_rejects_unregistered_ids(
    field: str,
    reference: str,
    error: str,
) -> None:
    registry = load_repository_claim_registry()
    claim = registry.claims[0].model_copy(update={field: reference})
    invalid_registry = ClaimRegistry(
        claims=(claim, *registry.claims[1:]),
        source_regions=registry.source_regions,
    )

    with pytest.raises(ValueError, match=error):
        validate_static_references(
            invalid_registry,
            _load_reproduction_contract(),
        )

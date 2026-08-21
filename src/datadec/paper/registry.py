from __future__ import annotations

from enum import UNIQUE, StrEnum, verify

from datadec.paper.models import (
    ClaimRegistry,
    MethodProvenance,
    PaperReproductionContract,
)


@verify(UNIQUE)
class VerifierId(StrEnum):
    SOURCE_TRACE = "source_trace"
    CITATION_TRACE = "citation_trace"
    SUITE_CONFIG = "suite_config"
    OLMES_AGGREGATE = "olmes_aggregate"
    AUTHOR_ARTIFACT = "author_artifact"
    ARTIFACT_INVENTORY = "artifact_inventory"
    OLMES_CHOICE = "olmes_choice"
    SCALING_LAW = "scaling_law"


def validate_static_references(
    registry: ClaimRegistry,
    contract: PaperReproductionContract,
) -> None:
    """Validate claim references against the repository-owned static registries."""
    verifier_ids = {verifier.value for verifier in VerifierId}
    method_ids = {method.id for method in contract.methods}
    policy_ids = {policy.id for policy in contract.policies}

    dotted_references = {
        reference
        for claim in registry.claims
        for reference in (claim.verifier_id, claim.method_id, claim.policy_id)
        if reference is not None and "." in reference
    }
    if dotted_references:
        dotted = ", ".join(sorted(dotted_references))
        raise ValueError(
            "paper claims must use registry IDs, not dotted callable references: "
            f"{dotted}"
        )

    unknown_verifier_ids = {
        claim.verifier_id
        for claim in registry.claims
        if claim.verifier_id is not None and claim.verifier_id not in verifier_ids
    }
    unknown_method_ids = {
        claim.method_id
        for claim in registry.claims
        if claim.method_id is not None and claim.method_id not in method_ids
    }
    unknown_policy_ids = {
        claim.policy_id
        for claim in registry.claims
        if claim.policy_id is not None and claim.policy_id not in policy_ids
    }

    failures = (
        ("verifier", unknown_verifier_ids),
        ("method", unknown_method_ids),
        ("policy", unknown_policy_ids),
    )
    for description, unknown_ids in failures:
        if unknown_ids:
            unknown = ", ".join(sorted(unknown_ids))
            raise ValueError(
                f"paper claims reference unknown {description} IDs: {unknown}"
            )


def resolve_method_provenance(
    contract: PaperReproductionContract,
    method_id: str | None,
) -> MethodProvenance | None:
    if method_id is None:
        return None
    methods = {method.id: method.provenance for method in contract.methods}
    try:
        return methods[method_id]
    except KeyError as error:  # pragma: no cover - repository validation owns this
        raise ValueError(
            f"unknown paper reproduction method ID: {method_id}"
        ) from error


__all__ = [
    "VerifierId",
    "resolve_method_provenance",
    "validate_static_references",
]

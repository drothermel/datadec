from __future__ import annotations

from datadec.paper.models import (
    PRIMARY_CLAIM_KINDS,
    ClaimRegistry,
    PaperValidationContract,
    ValidationOutcome,
)


def validate_cross_contracts(
    registry: ClaimRegistry,
    contract: PaperValidationContract,
) -> None:
    """Validate executable attempts against the source-linked claim inventory."""
    claims = {claim.id: claim for claim in registry.claims}
    attempts = {attempt.id: attempt for attempt in contract.attempts}

    declared_attempt_ids = {
        attempt_id for claim in registry.claims for attempt_id in claim.attempt_ids
    }
    configured_attempt_ids = set(attempts)
    if declared_attempt_ids != configured_attempt_ids:
        missing = sorted(declared_attempt_ids - configured_attempt_ids)
        unexpected = sorted(configured_attempt_ids - declared_attempt_ids)
        raise ValueError(
            "claim and validation attempt IDs differ: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for attempt in contract.attempts:
        claim = claims.get(attempt.claim_id)
        if claim is None:
            raise ValueError(f"attempt {attempt.id} references unknown claim")
        if attempt.id not in claim.attempt_ids:
            raise ValueError(f"attempt {attempt.id} is attached to the wrong claim")
        if claim.kind not in PRIMARY_CLAIM_KINDS:
            raise ValueError(f"nonempirical claim {claim.id} has an executable attempt")

    for claim in registry.claims:
        if claim.kind not in PRIMARY_CLAIM_KINDS:
            continue
        is_assessable = (
            claim.supporting_outcome
            is not ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
        )
        defaults = [
            attempts[attempt_id]
            for attempt_id in claim.attempt_ids
            if attempts[attempt_id].default
        ]
        if is_assessable and len(defaults) != 1:
            raise ValueError(
                f"assessable primary claim {claim.id} requires one default attempt"
            )
        if not is_assessable and claim.attempt_ids:
            raise ValueError(
                f"non-assessable primary claim {claim.id} cannot have attempts"
            )


__all__ = [
    "validate_cross_contracts",
]

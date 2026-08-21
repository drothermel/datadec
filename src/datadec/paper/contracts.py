from __future__ import annotations

import tomllib
from pathlib import Path

from datadec.paper.models import ClaimRegistry

_REPOSITORY_ROOT = Path(__file__).parents[3]
_CLAIMS_CONTRACT_PATH = Path("docs/paper/claims.toml")


def load_claim_registry(path: str | Path) -> ClaimRegistry:
    with Path(path).open("rb") as file:
        return ClaimRegistry.model_validate(tomllib.load(file))


def load_repository_claim_registry(
    repository_root: str | Path | None = None,
) -> ClaimRegistry:
    root = _REPOSITORY_ROOT if repository_root is None else Path(repository_root)
    return load_claim_registry(root / _CLAIMS_CONTRACT_PATH)


__all__ = ["load_claim_registry", "load_repository_claim_registry"]

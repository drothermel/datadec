from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TypeVar
from importlib.resources.abc import Traversable

from pydantic import BaseModel

from datadec.paper.models import ClaimRegistry, PaperValidationContract

_REPOSITORY_ROOT = Path(__file__).parents[3]
_CLAIMS_CONTRACT_PATH = Path("docs/paper/claims.toml")
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def _strict_toml_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_strict_toml_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _strict_toml_value(item) for key, item in value.items()}
    return value


def load_toml_model(path: str | Path | Traversable, model: type[_ModelT]) -> _ModelT:
    source = Path(path) if isinstance(path, (str, Path)) else path
    with source.open("rb") as file:
        raw = _strict_toml_value(tomllib.load(file))
    # TOML has no tuple or enum types; the parser already provides strict scalar
    # types, so only those representation conversions are permitted here.
    return model.model_validate(raw, strict=False)


def load_claim_registry(path: str | Path) -> ClaimRegistry:
    return load_toml_model(path, ClaimRegistry)


def load_validation_contract(
    path: str | Path | Traversable,
) -> PaperValidationContract:
    return load_toml_model(path, PaperValidationContract)


def load_repository_claim_registry(
    repository_root: str | Path | None = None,
) -> ClaimRegistry:
    root = _REPOSITORY_ROOT if repository_root is None else Path(repository_root)
    return load_claim_registry(root / _CLAIMS_CONTRACT_PATH)


__all__ = [
    "load_claim_registry",
    "load_repository_claim_registry",
    "load_toml_model",
    "load_validation_contract",
]

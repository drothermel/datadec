from __future__ import annotations

import re
from math import ceil
from pathlib import Path

from datadec.config import DataDecideCatalog
from datadec.paper.contracts import load_toml_model
from datadec.paper.models import (
    ClaimKind,
    ClaimRegistry,
    MetadataDiscrepancy,
    PaperClaim,
)

_CATALOG_PATH = Path("configs/catalog.toml")
_SEQUENCE_LENGTH_CLAIM_ID = "DD-0269"
_SUITE_ROW_CLAIM_IDS = tuple(f"DD-{index:04d}" for index in range(276, 290))
_SUITE_FIELDS = (
    "model_name",
    "batch_size",
    "hidden_dimension",
    "learning_rate",
    "model_size",
    "heads",
    "layers",
    "training_steps",
    "tokens_trained",
)
_SCALAR_METADATA_SOURCES = {
    "DD-0267": "configs/catalog.toml models",
    _SEQUENCE_LENGTH_CLAIM_ID: (
        "configs/catalog.toml and catalog-derived dd_parsed max_sequence_length"
    ),
    "DD-0270": "configs/catalog.toml model defaults and model definitions",
    "DD-0271": "configs/catalog.toml data_recipe_families",
}
_NON_HISTORICAL_NOTE = (
    "This is a descriptive metadata discrepancy, not an empirical "
    "not_reproduced outcome or evidence of the historical author training state."
)


def _paper_locator(claim: PaperClaim) -> str:
    line_span = (
        str(claim.line_start)
        if claim.line_start == claim.line_end
        else f"{claim.line_start}-{claim.line_end}"
    )
    return f"{claim.source_file}:{line_span}"


def _integer_target(claim: PaperClaim) -> int:
    target = claim.paper_target
    if type(target) is int:
        return target
    if isinstance(target, str):
        match = re.fullmatch(r".*=\s*([0-9][0-9,]*)", target)
        if match is not None:
            return int(match.group(1).replace(",", ""))
    raise ValueError(f"claim {claim.id} does not have an integer metadata target")


def _uniform_value(values: set[int]) -> int | list[int]:
    ordered = sorted(values)
    return ordered[0] if len(ordered) == 1 else ordered


def _available_scalar_metadata(
    claim_id: str,
    catalog: DataDecideCatalog,
) -> int | list[int]:
    if claim_id == "DD-0267":
        return len(catalog.models)
    if claim_id == _SEQUENCE_LENGTH_CLAIM_ID:
        return _uniform_value(
            {
                catalog.training.max_sequence_length,
                catalog.model_defaults.max_sequence_length,
            }
        )
    if claim_id == "DD-0270":
        return _uniform_value(
            {catalog.model_defaults.mlp_ratio}
            | {model.mlp_ratio for model in catalog.models}
        )
    if claim_id == "DD-0271":
        recipes = {
            recipe
            for family in catalog.data_recipe_families.values()
            for recipe in family
        }
        return len(recipes)
    raise ValueError(f"unsupported scalar metadata claim: {claim_id}")


def _parse_length_multiplier(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)xC", value, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"unsupported model length multiplier: {value!r}")
    return int(match.group(1))


def _available_suite_rows(
    catalog: DataDecideCatalog,
) -> dict[str, tuple[str, ...]]:
    training = catalog.training
    length_multiplier = _parse_length_multiplier(catalog.model_defaults.length_str)
    batch_multiple = training.gpus_per_node * training.microbatch_size
    rows: dict[str, tuple[str, ...]] = {}
    for model in catalog.models:
        batch_size = int(
            round(
                (
                    training.batch_size_coefficient
                    * (
                        model.training_parameter_count
                        / training.model_size_normalization
                    )
                    ** training.batch_size_exponent
                )
                / batch_multiple
            )
            * batch_multiple
        )
        total_tokens = (
            length_multiplier
            * training.token_length_multiplier
            * model.training_parameter_count
        )
        learning_rate = (
            training.learning_rate_base
            * (model.training_parameter_count / training.model_size_normalization)
            ** training.learning_rate_exponent
        )
        training_steps = ceil(
            total_tokens / (batch_size * training.max_sequence_length)
        )
        rows[model.name] = (
            model.name,
            f"{batch_size:,}",
            f"{model.d_model:,}",
            f"{learning_rate:.1e}",
            f"{model.exact_parameter_count / 1_000_000:.1f}M",
            f"{model.n_heads:,}",
            f"{model.n_layers:,}",
            f"{training_steps:,}",
            f"{total_tokens / 1_000_000_000:.1f}B",
        )
    return rows


def _paper_suite_row(claim: PaperClaim) -> tuple[str, ...]:
    target = claim.paper_target
    if not isinstance(target, str):
        raise ValueError(f"claim {claim.id} does not have a suite-row string target")
    fields = tuple(target.split("|"))
    if len(fields) != len(_SUITE_FIELDS):
        raise ValueError(
            f"claim {claim.id} suite-row target must contain "
            f"{len(_SUITE_FIELDS)} fields"
        )
    return fields


def _compare_scalar_claim(
    claim: PaperClaim,
    catalog: DataDecideCatalog,
) -> MetadataDiscrepancy | None:
    paper_value = _integer_target(claim)
    metadata_value = _available_scalar_metadata(claim.id, catalog)
    if paper_value == metadata_value:
        return None
    note = (
        f"The paper reports {paper_value}, while current available metadata records "
        f"{metadata_value}. {_NON_HISTORICAL_NOTE}"
    )
    return MetadataDiscrepancy(
        claim_id=claim.id,
        paper_locator=_paper_locator(claim),
        paper_value=paper_value,
        metadata_source=_SCALAR_METADATA_SOURCES[claim.id],
        metadata_value=metadata_value,
        note=note,
    )


def _compare_suite_claim(
    claim: PaperClaim,
    available_rows: dict[str, tuple[str, ...]],
) -> MetadataDiscrepancy | None:
    paper_row = _paper_suite_row(claim)
    model_name = paper_row[0]
    available_row = available_rows.get(model_name)
    if available_row is None:
        return MetadataDiscrepancy(
            claim_id=claim.id,
            paper_locator=_paper_locator(claim),
            paper_value=claim.paper_target,
            metadata_source=(
                "configs/catalog.toml and catalog-derived dd_parsed model metadata"
            ),
            metadata_value=f"model {model_name!r} is absent",
            note=f"The paper suite row has no current catalog model. {_NON_HISTORICAL_NOTE}",
        )

    differences = tuple(
        (field, paper, available)
        for field, paper, available in zip(
            _SUITE_FIELDS, paper_row, available_row, strict=True
        )
        if paper != available
    )
    if not differences:
        return None
    detail = "; ".join(
        f"{field} (paper={paper!r}, available={available!r})"
        for field, paper, available in differences
    )
    return MetadataDiscrepancy(
        claim_id=claim.id,
        paper_locator=_paper_locator(claim),
        paper_value=claim.paper_target,
        metadata_source=(
            "configs/catalog.toml and catalog-derived dd_parsed model metadata"
        ),
        metadata_value="|".join(available_row),
        note=f"Directly comparable suite fields differ: {detail}. {_NON_HISTORICAL_NOTE}",
    )


def compare_descriptive_metadata(
    repository_root: str | Path,
    registry: ClaimRegistry,
) -> tuple[MetadataDiscrepancy, ...]:
    """Compare directly supported paper descriptions with current metadata."""
    root = Path(repository_root)
    catalog = load_toml_model(root / _CATALOG_PATH, DataDecideCatalog)
    claims = {
        claim.id: claim
        for claim in registry.claims
        if claim.kind is ClaimKind.DESCRIPTIVE_METADATA
    }

    discrepancies: list[MetadataDiscrepancy] = []
    for claim_id in _SCALAR_METADATA_SOURCES:
        claim = claims.get(claim_id)
        if claim is not None:
            discrepancy = _compare_scalar_claim(claim, catalog)
            if discrepancy is not None:
                discrepancies.append(discrepancy)

    available_rows = _available_suite_rows(catalog)
    for claim_id in _SUITE_ROW_CLAIM_IDS:
        claim = claims.get(claim_id)
        if claim is not None:
            discrepancy = _compare_suite_claim(claim, available_rows)
            if discrepancy is not None:
                discrepancies.append(discrepancy)

    return tuple(sorted(discrepancies, key=lambda item: item.claim_id))


__all__ = ["compare_descriptive_metadata"]

from __future__ import annotations

from pathlib import Path

from datadec.paper import (
    ClaimRegistry,
    MetadataDiscrepancy,
    load_repository_claim_registry,
)
from datadec.paper.verifiers.metadata import compare_descriptive_metadata

_REPOSITORY_ROOT = Path(__file__).parents[2]


def _by_claim_id() -> dict[str, MetadataDiscrepancy]:
    registry = load_repository_claim_registry(_REPOSITORY_ROOT)
    return {
        item.claim_id: item
        for item in compare_descriptive_metadata(_REPOSITORY_ROOT, registry)
    }


def test_current_metadata_discrepancies_are_separate_descriptive_records() -> None:
    discrepancies = _by_claim_id()

    assert tuple(discrepancies) == (
        "DD-0269",
        *(f"DD-{index:04d}" for index in range(276, 290)),
    )
    sequence_length = discrepancies["DD-0269"]
    assert sequence_length.paper_locator == "docs/paper/example_paper.tex:489"
    assert sequence_length.paper_value == 2024
    assert sequence_length.metadata_value == 2048
    assert "dd_parsed max_sequence_length" in sequence_length.metadata_source
    assert "not an empirical not_reproduced outcome" in sequence_length.note
    assert "historical author training state" in sequence_length.note


def test_suite_rows_record_only_direct_field_differences() -> None:
    discrepancies = _by_claim_id()

    four_million = discrepancies["DD-0276"]
    assert four_million.paper_value == ("4M|32|64|1.4e-02|3.7M|8|8|5,725|0.4B")
    assert four_million.metadata_value == ("4M|32|64|1.4e-02|3.7M|8|8|5,715|0.4B")
    assert "training_steps (paper='5,725', available='5,715')" in four_million.note

    one_billion = discrepancies["DD-0289"]
    assert one_billion.metadata_value == (
        "1B|704|2,048|2.2e-03|1176.8M|16|16|69,359|100.0B"
    )
    assert "learning_rate (paper='2.1e-03', available='2.2e-03')" in (one_billion.note)
    assert "training_steps (paper='69,369', available='69,359')" in (one_billion.note)


def test_matching_and_historically_inferred_descriptions_are_omitted() -> None:
    discrepancies = _by_claim_id()

    assert set(discrepancies).isdisjoint(
        {
            "DD-0267",  # fourteen current catalog configurations
            "DD-0270",  # current catalog MLP ratio eight
            "DD-0271",  # twenty-five current catalog recipes
            "DD-0272",  # historical seeds per recipe/configuration
            "DD-0273",  # historical early-stop policy
        }
    )


def test_comparison_order_does_not_depend_on_registry_order() -> None:
    registry = load_repository_claim_registry(_REPOSITORY_ROOT)
    reversed_registry = registry.model_copy(
        update={"claims": tuple(reversed(registry.claims))}
    )

    assert compare_descriptive_metadata(
        _REPOSITORY_ROOT, reversed_registry
    ) == compare_descriptive_metadata(_REPOSITORY_ROOT, registry)


def test_repository_root_controls_available_catalog_metadata(tmp_path: Path) -> None:
    registry = load_repository_claim_registry(_REPOSITORY_ROOT)
    sequence_claim = next(claim for claim in registry.claims if claim.id == "DD-0269")
    focused_registry = ClaimRegistry(format_version=2, claims=(sequence_claim,))
    configs = tmp_path / "configs"
    configs.mkdir()
    catalog = (_REPOSITORY_ROOT / "configs/catalog.toml").read_text()
    (configs / "catalog.toml").write_text(
        catalog.replace("max_sequence_length = 2048", "max_sequence_length = 4096")
    )

    (discrepancy,) = compare_descriptive_metadata(tmp_path, focused_registry)

    assert discrepancy.claim_id == "DD-0269"
    assert discrepancy.metadata_value == 4096

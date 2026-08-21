from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath

from datadec.config import load_paper_validation_contract
from datadec.paper import (
    ClaimRegistry,
    PaperValidationContract,
    load_repository_claim_registry,
)
from datadec.paper.source import (
    derive_manuscript_source_surface,
    raw_line_slice_sha256,
    scan_tex_dependencies,
    validate_citations,
    validate_source_coverage,
)


_REPOSITORY_ROOT = Path(__file__).parents[2]

_TEX_FILES = (
    "docs/paper/example_paper.tex",
    "docs/paper/tables/data_recipes.tex",
    "docs/paper/tables/pred_error.tex",
    "docs/paper/tables/proxy_metrics.tex",
    "docs/paper/tables/suite_stats.tex",
)
_GRAPHICS_FILES = (
    "docs/paper/figures/DataDecide_Infographic.pdf",
    "docs/paper/figures/accuracy_vs_compute_with_legend_fixed.pdf",
    "docs/paper/figures/all_metrics_compute_vs_accuracy_per_task.pdf",
    "docs/paper/figures/datadecide-logo.pdf",
    "docs/paper/figures/math_code_decision_accuracy.pdf",
    "docs/paper/figures/noise_to_spread_150M.pdf",
    "docs/paper/figures/primary_metric_compute_vs_accuracy_per_task.pdf",
)
_NON_CLAIM_REGION_KINDS = {
    "affiliation_metadata",
    "author_metadata",
    "bibliography_commands",
    "bibliography_provenance",
    "bibliography_rendering",
    "contact_metadata",
    "decorative_logo",
    "document_setup",
    "document_structure",
    "keyword_metadata",
    "section_heading",
    "section_layout",
    "semantic_vocabulary",
    "table_heading",
    "table_layout",
    "title_metadata",
}


def _load_current_registry() -> tuple[PaperValidationContract, ClaimRegistry]:
    contract = load_paper_validation_contract()
    return contract, load_repository_claim_registry(_REPOSITORY_ROOT)


def test_current_source_regions_partition_independently_derived_surface() -> None:
    contract, registry = _load_current_registry()
    entrypoint = (
        PurePosixPath(contract.paper.source_root) / contract.paper.entrypoint
    ).as_posix()
    surface = derive_manuscript_source_surface(_REPOSITORY_ROOT, entrypoint)
    claim_regions = tuple(
        region for region in registry.source_regions if region.claim_ids
    )
    non_claim_regions = tuple(
        region for region in registry.source_regions if region.non_claim_reason
    )
    line_coverage = Counter(
        (region.source_file, line_number)
        for region in registry.source_regions
        for line_number in range(region.line_start, region.line_end + 1)
        if (region.source_file, line_number) in set(surface.active_tex_lines)
    )

    assert Counter(source_file for source_file, _ in surface.active_tex_lines) == {
        "docs/paper/example_paper.tex": 363,
        "docs/paper/tables/data_recipes.tex": 26,
        "docs/paper/tables/pred_error.tex": 15,
        "docs/paper/tables/proxy_metrics.tex": 18,
        "docs/paper/tables/suite_stats.tex": 20,
    }
    assert line_coverage == Counter(dict.fromkeys(surface.active_tex_lines, 1))
    assert surface.asset_files == (
        "docs/paper/example_paper.bbl",
        "docs/paper/example_paper.bib",
        *_GRAPHICS_FILES,
    )
    assert surface.excluded_implementation_files == ("docs/paper/icml2025.bst",)
    assert not any(
        region.source_file in surface.excluded_implementation_files
        for region in registry.source_regions
    )
    assert len(registry.claims) == 455
    assert len(claim_regions) == 93
    assert len(non_claim_regions) == 99
    assert Counter(
        claim_id for region in claim_regions for claim_id in region.claim_ids
    ) == Counter(claim.id for claim in registry.claims)
    assert {region.kind for region in non_claim_regions} == _NON_CLAIM_REGION_KINDS
    assert all(region.non_claim_reason for region in non_claim_regions)
    claims = {claim.id: claim for claim in registry.claims}
    assert {
        claim_id: (claims[claim_id].line_start, claims[claim_id].line_end)
        for claim_id in (
            "DD-0443",
            "DD-0444",
            "DD-0445",
            "DD-0446",
            "DD-0447",
            "DD-0448",
            "DD-0449",
            "DD-0450",
            "DD-0451",
            "DD-0452",
            "DD-0453",
            "DD-0454",
            "DD-0455",
        )
    } == {
        "DD-0443": (66, 66),
        "DD-0444": (67, 67),
        "DD-0445": (68, 68),
        "DD-0446": (69, 69),
        "DD-0447": (71, 71),
        "DD-0448": (72, 72),
        "DD-0449": (73, 73),
        "DD-0450": (158, 158),
        "DD-0451": (283, 283),
        "DD-0452": (295, 295),
        "DD-0453": (328, 328),
        "DD-0454": (495, 495),
        "DD-0455": (506, 506),
    }


def test_current_source_regions_are_hashed_and_canonically_ordered() -> None:
    _, registry = _load_current_registry()

    coverage = validate_source_coverage(_REPOSITORY_ROOT, registry)

    assert coverage.claim_ids == tuple(f"DD-{index:04d}" for index in range(1, 456))
    assert len(coverage.source_region_ids) == 192
    assert len(coverage.source_region_ids) == len(set(coverage.source_region_ids))
    assert registry.source_regions == tuple(
        sorted(
            registry.source_regions,
            key=lambda region: (
                region.source_file,
                region.line_start,
                region.line_end,
                region.id,
            ),
        )
    )
    assert all(
        region.content_sha256
        == raw_line_slice_sha256(
            _REPOSITORY_ROOT,
            region.source_file,
            region.line_start,
            region.line_end,
        )
        for region in registry.source_regions
    )


def test_current_dependencies_include_every_active_claim_source() -> None:
    contract, registry = _load_current_registry()
    entrypoint = (
        PurePosixPath(contract.paper.source_root) / contract.paper.entrypoint
    ).as_posix()

    dependencies = scan_tex_dependencies(_REPOSITORY_ROOT, entrypoint)

    assert dependencies.tex_files == _TEX_FILES
    assert dependencies.input_files == _TEX_FILES[1:]
    assert dependencies.graphics_files == _GRAPHICS_FILES
    assert dependencies.bibliography_files == ("docs/paper/example_paper.bib",)
    assert dependencies.bibliography_style_files == ("docs/paper/icml2025.bst",)
    assert dependencies.bbl_files == ("docs/paper/example_paper.bbl",)
    assert {claim.source_file for claim in registry.claims} <= (
        set(dependencies.tex_files) | set(dependencies.graphics_files)
    )

    claimed_graphics = {
        claim.source_file
        for claim in registry.claims
        if claim.source_file.endswith(".pdf")
    }
    assert set(dependencies.graphics_files) - claimed_graphics == {
        "docs/paper/figures/datadecide-logo.pdf"
    }
    assert tuple(
        region.id
        for region in registry.source_regions
        if region.source_file == "docs/paper/figures/datadecide-logo.pdf"
    ) == ("SR-N-decorative-logo",)
    assert all(
        (region.line_start, region.line_end)
        == (
            1,
            len((_REPOSITORY_ROOT / region.source_file).read_bytes().splitlines()),
        )
        for region in registry.source_regions
        if region.source_file in dependencies.graphics_files
    )


def test_current_citation_keys_resolve_in_bib_and_rendered_bibliography() -> None:
    contract, registry = _load_current_registry()
    entrypoint = (
        PurePosixPath(contract.paper.source_root) / contract.paper.entrypoint
    ).as_posix()

    dependencies = scan_tex_dependencies(_REPOSITORY_ROOT, entrypoint)
    citations = validate_citations(_REPOSITORY_ROOT, entrypoint)

    assert len(citations.citation_keys) == 43
    assert citations.citation_keys == dependencies.citation_keys
    assert set(citations.citation_keys) <= set(citations.bib_keys)
    assert set(citations.citation_keys) <= set(citations.bbl_keys)

    claims = {claim.id: claim for claim in registry.claims}
    assert claims["DD-0072"].citation_keys == ("Penedo2023TheRD",)
    assert claims["DD-0078"].citation_keys == ("commoncrawl",)
    assert (claims["DD-0231"].line_start, claims["DD-0231"].line_end) == (
        448,
        449,
    )


def test_source_region_digests_do_not_depend_on_normalized_claim_prose() -> None:
    _, registry = _load_current_registry()
    changed_claim = registry.claims[0].model_copy(
        update={"text": "A deliberately different normalized claim summary."}
    )
    changed_registry = registry.model_copy(
        update={"claims": (changed_claim, *registry.claims[1:])}
    )

    assert validate_source_coverage(
        _REPOSITORY_ROOT, changed_registry
    ) == validate_source_coverage(_REPOSITORY_ROOT, registry)

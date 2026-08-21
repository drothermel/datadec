from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from datadec.config import load_paper_reproduction_contract
from datadec.paper import (
    ClaimRegistry,
    PaperClaim,
    PaperReproductionContract,
    load_claim_registry,
)
from datadec.paper.source import (
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
    ("SR-N-bibliography-rendering", "bibliography_rendering"),
    ("SR-N-bibliography-provenance", "bibliography_provenance"),
    ("SR-N-running-title", "title_metadata"),
    ("SR-N-logo-macro", "decorative_logo"),
    ("SR-N-paper-title", "title_metadata"),
    ("SR-N-author-list", "author_metadata"),
    ("SR-N-affiliations", "affiliation_metadata"),
    ("SR-N-corresponding-author", "contact_metadata"),
    ("SR-N-keywords", "keyword_metadata"),
    ("SR-N-equal-contribution", "author_metadata"),
    ("SR-N-heading-introduction", "section_heading"),
    ("SR-N-heading-methods", "section_heading"),
    ("SR-N-heading-suite", "section_heading"),
    ("SR-N-heading-prediction-methods", "section_heading"),
    ("SR-N-heading-prediction-metrics", "section_heading"),
    ("SR-N-heading-decision-accuracy", "section_heading"),
    ("SR-N-heading-olmes", "section_heading"),
    ("SR-N-heading-proxy-metrics", "section_heading"),
    ("SR-N-heading-results", "section_heading"),
    ("SR-N-heading-compute", "section_heading"),
    ("SR-N-heading-scaling-comparison", "section_heading"),
    ("SR-N-heading-proxy-signal", "section_heading"),
    ("SR-N-heading-benchmark-predictability", "section_heading"),
    ("SR-N-heading-related-work", "section_heading"),
    ("SR-N-heading-related-prediction", "section_heading"),
    ("SR-N-heading-limitations", "section_heading"),
    ("SR-N-heading-acknowledgments", "section_heading"),
    ("SR-N-heading-impact", "section_heading"),
    ("SR-N-bibliography-commands", "bibliography_commands"),
    ("SR-N-appendix-boundary", "section_layout"),
    ("SR-N-heading-hyperparameters", "section_heading"),
    ("SR-N-heading-proxy-definitions", "section_heading"),
    ("SR-N-heading-scaling-variants", "section_heading"),
    ("SR-N-heading-baseline-fit", "section_heading"),
    ("SR-N-heading-two-parameter-fit", "section_heading"),
    ("SR-N-heading-helper-points", "section_heading"),
    ("SR-N-heading-checkpoint-filter", "section_heading"),
    ("SR-N-decorative-logo", "decorative_logo"),
    ("SR-N-table-data-recipes-heading", "table_heading"),
    ("SR-N-table-prediction-error-heading", "table_heading"),
    ("SR-N-table-proxy-metrics-heading", "table_heading"),
    ("SR-N-table-suite-stats-heading", "table_heading"),
}


def _load_current_registry() -> tuple[PaperReproductionContract, ClaimRegistry]:
    contract = load_paper_reproduction_contract()
    return contract, load_claim_registry(
        _REPOSITORY_ROOT / contract.contracts.claims_contract
    )


def _connected_claim_regions(
    claims: tuple[PaperClaim, ...],
) -> tuple[tuple[str, int, int, tuple[str, ...]], ...]:
    by_file: dict[str, list[PaperClaim]] = defaultdict(list)
    for claim in claims:
        by_file[claim.source_file].append(claim)

    connected: list[tuple[str, int, int, tuple[str, ...]]] = []
    for source_file, file_claims in sorted(by_file.items()):
        groups: list[tuple[int, int, list[str]]] = []
        for claim in sorted(
            file_claims,
            key=lambda item: (item.line_start, item.line_end, item.id),
        ):
            if not groups or claim.line_start > groups[-1][1] + 1:
                groups.append((claim.line_start, claim.line_end, [claim.id]))
                continue
            line_start, line_end, claim_ids = groups[-1]
            groups[-1] = (
                line_start,
                max(line_end, claim.line_end),
                [*claim_ids, claim.id],
            )
        connected.extend(
            (source_file, line_start, line_end, tuple(sorted(claim_ids)))
            for line_start, line_end, claim_ids in groups
        )
    return tuple(connected)


def _claim_region_id(source_file: str, line_start: int, line_end: int) -> str:
    relative = PurePosixPath(source_file).relative_to("docs/paper").with_suffix("")
    slug = re.sub(r"[^a-z0-9]+", "-", relative.as_posix().lower()).strip("-")
    return f"SR-C-{slug}-{line_start:04d}-{line_end:04d}"


def test_current_source_regions_cover_exact_connected_claim_spans() -> None:
    _, registry = _load_current_registry()
    claim_regions = tuple(
        region for region in registry.source_regions if region.claim_ids
    )
    non_claim_regions = tuple(
        region for region in registry.source_regions if region.non_claim_reason
    )
    expected_connected = _connected_claim_regions(registry.claims)

    assert len(registry.claims) == 442
    assert len(claim_regions) == len(expected_connected) == 86
    assert len(non_claim_regions) == len(_NON_CLAIM_REGION_KINDS) == 42
    assert (
        tuple(
            (
                region.source_file,
                region.line_start,
                region.line_end,
                region.claim_ids,
            )
            for region in claim_regions
        )
        == expected_connected
    )
    assert tuple(region.id for region in claim_regions) == tuple(
        _claim_region_id(source_file, line_start, line_end)
        for source_file, line_start, line_end, _ in expected_connected
    )
    assert Counter(
        claim_id for region in claim_regions for claim_id in region.claim_ids
    ) == Counter(claim.id for claim in registry.claims)
    assert {(region.id, region.kind) for region in non_claim_regions} == (
        _NON_CLAIM_REGION_KINDS
    )


def test_current_source_regions_are_hashed_and_canonically_ordered() -> None:
    _, registry = _load_current_registry()

    coverage = validate_source_coverage(_REPOSITORY_ROOT, registry)

    assert coverage.claim_ids == tuple(f"DD-{index:04d}" for index in range(1, 443))
    assert len(coverage.source_region_ids) == 128
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


def test_current_citation_keys_resolve_in_bib_and_rendered_bibliography() -> None:
    contract, _ = _load_current_registry()
    entrypoint = (
        PurePosixPath(contract.paper.source_root) / contract.paper.entrypoint
    ).as_posix()

    dependencies = scan_tex_dependencies(_REPOSITORY_ROOT, entrypoint)
    citations = validate_citations(_REPOSITORY_ROOT, entrypoint)

    assert len(citations.citation_keys) == 43
    assert citations.citation_keys == dependencies.citation_keys
    assert set(citations.citation_keys) <= set(citations.bib_keys)
    assert set(citations.citation_keys) <= set(citations.bbl_keys)


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

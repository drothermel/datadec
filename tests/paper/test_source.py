from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from datadec.paper.models import ClaimRegistry, PaperClaim, SourceRegion
from datadec.paper.source import (
    CitationReport,
    CoverageReport,
    DependencyReport,
    ManuscriptSourceSurface,
    SourceValidationError,
    derive_manuscript_source_surface,
    raw_line_slice_sha256,
    scan_tex_dependencies,
    validate_citations,
    validate_source_coverage,
)


_REPOSITORY_ROOT = Path(__file__).parents[2]
_ENTRYPOINT = "docs/paper/example_paper.tex"


def _write_paper_tree(tmp_path: Path, body: str) -> Path:
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "main.tex").write_text(body)
    (paper / "main.bbl").write_text(
        "\\begin{thebibliography}{1}\n"
        "\\bibitem{known} Known source.\n"
        "\\end{thebibliography}\n"
    )
    (paper / "refs.bib").write_text("@misc{known, title={Known}}\n")
    (paper / "plain.bst").write_text("ENTRY {} {} {}\n")
    return paper


def _claim(
    *,
    claim_id: str = "claim-1",
    line_start: int = 1,
    line_end: int = 1,
    citation_keys: tuple[str, ...] = (),
) -> PaperClaim:
    return PaperClaim.model_validate(
        {
            "id": claim_id,
            "source_file": "paper/source.tex",
            "line_start": line_start,
            "line_end": line_end,
            "text": "testable claim",
            "owner": "external_citation" if citation_keys else "datadec_empirical",
            "expectation_kind": "citation_trace" if citation_keys else "literal",
            "expectation": "alpha",
            "required_evidence_boundary": "aggregate_evaluation",
            "citation_keys": citation_keys,
        }
    )


def _region(
    source: bytes,
    *,
    region_id: str = "region-1",
    source_file: str = "paper/source.tex",
    claim_ids: tuple[str, ...] = ("claim-1",),
    line_start: int = 1,
    line_end: int = 1,
    digest: str | None = None,
) -> SourceRegion:
    lines = source.splitlines(keepends=True)
    content = b"".join(lines[line_start - 1 : line_end])
    values: dict[str, object] = {
        "id": region_id,
        "source_file": source_file,
        "line_start": line_start,
        "line_end": line_end,
        "kind": "prose",
        "content_sha256": digest or hashlib.sha256(content).hexdigest(),
        "claim_ids": claim_ids,
    }
    if not claim_ids:
        values["non_claim_reason"] = "formatting"
    return SourceRegion.model_validate(values)


def _validate_test_source_coverage(
    tmp_path: Path,
    registry: ClaimRegistry,
) -> CoverageReport:
    (tmp_path / "paper" / "source.bbl").write_text("")
    return validate_source_coverage(
        tmp_path,
        registry,
        entrypoint="paper/source.tex",
    )


def test_current_frozen_paper_dependencies_and_citations_are_complete() -> None:
    dependencies = scan_tex_dependencies(_REPOSITORY_ROOT, _ENTRYPOINT)
    citations = validate_citations(_REPOSITORY_ROOT, _ENTRYPOINT)

    assert isinstance(dependencies, DependencyReport)
    assert len(dependencies.tex_files) == 5
    assert len(dependencies.input_files) == 4
    assert len(dependencies.graphics_files) == 7
    assert dependencies.bibliography_files == ("docs/paper/example_paper.bib",)
    assert dependencies.bibliography_style_files == ("docs/paper/icml2025.bst",)
    assert dependencies.bbl_files == ("docs/paper/example_paper.bbl",)
    assert isinstance(citations, CitationReport)
    assert len(citations.citation_keys) == 43
    assert set(citations.citation_keys) <= set(citations.bib_keys)
    assert set(citations.citation_keys) <= set(citations.bbl_keys)


def test_dependency_scan_strips_comments_preserves_escaped_percent_and_stops(
    tmp_path: Path,
) -> None:
    paper = _write_paper_tree(
        tmp_path,
        r"""
% \input{missing-commented}
escaped \% remains \input{section}
\includegraphics{image.pdf} % \includegraphics{missing-commented.pdf}
\bibliography{refs}
\bibliographystyle{plain}
\citep{known}
\end{document}
\input{missing-after-end}
\citep{unknown-after-end}
""",
    )
    (paper / "section.tex").write_text("included text\n")
    (paper / "image.pdf").write_bytes(b"pdf")

    report = scan_tex_dependencies(tmp_path, "paper/main.tex")

    assert report.input_files == ("paper/section.tex",)
    assert report.graphics_files == ("paper/image.pdf",)
    assert report.citation_keys == ("known",)


def test_dependency_scan_follows_nested_literal_inputs_deterministically(
    tmp_path: Path,
) -> None:
    paper = _write_paper_tree(
        tmp_path,
        "\\input{b}\\input{a.tex}"
        "\\bibliography{refs}\\bibliographystyle{plain}\\end{document}\n",
    )
    (paper / "a.tex").write_text("A\n")
    (paper / "b.tex").write_text("\\input{nested/c}\n")
    (paper / "nested").mkdir()
    (paper / "nested" / "c.tex").write_text("C\n")

    report = scan_tex_dependencies(tmp_path, "paper/main.tex")

    assert report.tex_files == (
        "paper/a.tex",
        "paper/b.tex",
        "paper/main.tex",
        "paper/nested/c.tex",
    )
    assert report.input_files == (
        "paper/a.tex",
        "paper/b.tex",
        "paper/nested/c.tex",
    )


def test_manuscript_surface_is_derived_from_active_dependencies(
    tmp_path: Path,
) -> None:
    paper = _write_paper_tree(
        tmp_path,
        "% ignored\n"
        "claim\n"
        "\\input{section}\n"
        "\\includegraphics{image.pdf}\n"
        "\\bibliography{refs}\n"
        "\\bibliographystyle{plain}\n"
        "\\end{document}\n"
        "ignored after document\n",
    )
    (paper / "section.tex").write_text("included claim\n")
    (paper / "image.pdf").write_bytes(b"pdf")

    surface = derive_manuscript_source_surface(tmp_path, "paper/main.tex")

    assert surface == ManuscriptSourceSurface(
        active_tex_lines=(
            ("paper/main.tex", 2),
            ("paper/main.tex", 3),
            ("paper/main.tex", 4),
            ("paper/main.tex", 5),
            ("paper/main.tex", 6),
            ("paper/main.tex", 7),
            ("paper/section.tex", 1),
        ),
        asset_files=("paper/image.pdf", "paper/main.bbl", "paper/refs.bib"),
        excluded_implementation_files=("paper/plain.bst",),
    )


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("\\input{../outside}", "normalized repository-relative"),
        ("\\input{\\filename}", "literal normalized filename"),
        ("\\include{\\filename}", "outside the supported source subset"),
        ("\\includegraphics{figures/\\name.pdf}", "literal normalized filename"),
        ("\\ifdefined\\flag yes\\fi", "conditional control sequence"),
        ("\\csname generated\\endcsname", "unsupported active control sequence"),
        ("\\catcode`\\%=12", "unsupported active control sequence"),
        ("\\newcommand{\\semantic}[1]{#1}", "parameterized macro definition"),
    ],
)
def test_dependency_scan_rejects_source_that_requires_tex_interpretation(
    tmp_path: Path,
    body: str,
    error: str,
) -> None:
    _write_paper_tree(tmp_path, body)

    with pytest.raises(SourceValidationError, match=error):
        scan_tex_dependencies(tmp_path, "paper/main.tex")


def test_dependency_scan_rejects_repository_symlink_escape(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside-source.tex"
    outside.write_text("outside\n")
    paper = _write_paper_tree(repository, "\\input{linked}\n")
    (paper / "linked.tex").symlink_to(outside)

    with pytest.raises(SourceValidationError, match="escapes repository root"):
        scan_tex_dependencies(repository, "paper/main.tex")


def test_citation_validation_rejects_unknown_and_malformed_keys(tmp_path: Path) -> None:
    _write_paper_tree(
        tmp_path,
        "\\bibliography{refs}\\bibliographystyle{plain}"
        "\\citep{known, unknown}\\end{document}\n",
    )

    with pytest.raises(SourceValidationError, match=r"bib: unknown; bbl: unknown"):
        validate_citations(tmp_path, "paper/main.tex")

    (tmp_path / "paper" / "main.tex").write_text("\\citep{known,}\n")
    with pytest.raises(SourceValidationError, match="literal comma-separated"):
        validate_citations(tmp_path, "paper/main.tex")


def test_raw_line_slice_digest_preserves_original_newlines(tmp_path: Path) -> None:
    source = b"alpha\r\nbeta\ngamma"
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "source.tex").write_bytes(source)

    digest = raw_line_slice_sha256(tmp_path, "paper/source.tex", 1, 2)

    assert digest == hashlib.sha256(b"alpha\r\nbeta\n").hexdigest()


def test_source_coverage_validates_digest_mapping_and_sorted_report(
    tmp_path: Path,
) -> None:
    source = b"alpha\nbeta\n"
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "source.tex").write_bytes(source)
    registry = ClaimRegistry(
        claims=(
            _claim(claim_id="claim-b", line_start=2, line_end=2),
            _claim(claim_id="claim-a", line_start=1, line_end=1),
        ),
        source_regions=(
            _region(
                source,
                region_id="region-b",
                claim_ids=("claim-b",),
                line_start=2,
                line_end=2,
            ),
            _region(
                source,
                region_id="region-a",
                claim_ids=("claim-a",),
                line_start=1,
                line_end=1,
            ),
        ),
    )

    report = _validate_test_source_coverage(tmp_path, registry)

    assert report == CoverageReport(
        claim_ids=("claim-a", "claim-b"),
        source_region_ids=("region-a", "region-b"),
        source_files=("paper/source.tex",),
    )


def test_source_coverage_rejects_digest_mismatch(tmp_path: Path) -> None:
    source = b"alpha\n"
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "source.tex").write_bytes(source)
    registry = ClaimRegistry(
        claims=(_claim(),),
        source_regions=(_region(source, digest="0" * 64),),
    )

    with pytest.raises(SourceValidationError, match="digest mismatch"):
        _validate_test_source_coverage(tmp_path, registry)


def test_source_coverage_rejects_unpartitioned_active_tex_line(
    tmp_path: Path,
) -> None:
    source = b"claim\nactive layout\n"
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "source.tex").write_bytes(source)
    registry = ClaimRegistry(
        claims=(_claim(),),
        source_regions=(_region(source),),
    )

    with pytest.raises(SourceValidationError, match=r"exactly one.*source\.tex:2"):
        _validate_test_source_coverage(tmp_path, registry)


def test_source_coverage_requires_one_whole_file_region_per_asset(
    tmp_path: Path,
) -> None:
    source = b"claim\n\\includegraphics{image.pdf}\n"
    image = b"first\nsecond\n"
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "source.tex").write_bytes(source)
    (paper / "image.pdf").write_bytes(image)
    registry = ClaimRegistry(
        claims=(_claim(),),
        source_regions=(
            _region(source),
            _region(
                source,
                region_id="region-layout",
                claim_ids=(),
                line_start=2,
                line_end=2,
            ),
            _region(
                image,
                region_id="region-image",
                source_file="paper/image.pdf",
                claim_ids=(),
            ),
        ),
    )

    with pytest.raises(SourceValidationError, match="whole-file source region"):
        _validate_test_source_coverage(tmp_path, registry)


def test_source_coverage_binds_declared_citations_to_the_claim_span(
    tmp_path: Path,
) -> None:
    source = b"claim\n\\citep{known}\n"
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "source.tex").write_bytes(source)
    claim = _claim(citation_keys=("known",))
    registry = ClaimRegistry(
        claims=(claim,),
        source_regions=(_region(source, line_end=2),),
    )

    with pytest.raises(SourceValidationError, match="absent from its source span"):
        _validate_test_source_coverage(tmp_path, registry)

    bound_registry = registry.model_copy(
        update={"claims": (claim.model_copy(update={"line_end": 2}),)}
    )
    _validate_test_source_coverage(tmp_path, bound_registry)


def test_source_coverage_allows_claim_specific_citation_subset(
    tmp_path: Path,
) -> None:
    source = b"claim \\citep{known,other}\n"
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "source.tex").write_bytes(source)
    registry = ClaimRegistry(
        claims=(_claim(citation_keys=("known",)),),
        source_regions=(_region(source),),
    )

    _validate_test_source_coverage(tmp_path, registry)


def test_source_coverage_rejects_uncovered_claim(tmp_path: Path) -> None:
    source = b"alpha\n"
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "source.tex").write_bytes(source)
    registry = ClaimRegistry(claims=(_claim(),), source_regions=())

    with pytest.raises(SourceValidationError, match="not covered"):
        _validate_test_source_coverage(tmp_path, registry)


@pytest.mark.parametrize(
    ("source_file", "line_end", "error"),
    [
        ("paper/missing.tex", 1, "does not exist"),
        ("paper/source.tex", 2, "outside"),
    ],
)
def test_source_coverage_rejects_invalid_claim_locator(
    tmp_path: Path,
    source_file: str,
    line_end: int,
    error: str,
) -> None:
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "source.tex").write_text("alpha\n")
    claim = _claim().model_copy(
        update={"source_file": source_file, "line_end": line_end}
    )

    with pytest.raises(SourceValidationError, match=error):
        _validate_test_source_coverage(
            tmp_path, ClaimRegistry(claims=(claim,), source_regions=())
        )


def test_source_coverage_rejects_invalid_mapping_and_partial_overlap(
    tmp_path: Path,
) -> None:
    source = b"alpha\nbeta\ngamma\n"
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "source.tex").write_bytes(source)
    outside_mapping = ClaimRegistry(
        claims=(_claim(line_start=2, line_end=2),),
        source_regions=(_region(source, line_start=1, line_end=1),),
    )

    with pytest.raises(SourceValidationError, match="does not contain mapped claim"):
        _validate_test_source_coverage(tmp_path, outside_mapping)

    overlap = ClaimRegistry(
        claims=(_claim(line_start=2, line_end=2),),
        source_regions=(
            _region(source, line_start=1, line_end=2),
            _region(
                source,
                region_id="region-2",
                claim_ids=(),
                line_start=2,
                line_end=3,
            ),
        ),
    )
    with pytest.raises(SourceValidationError, match="overlap"):
        _validate_test_source_coverage(tmp_path, overlap)

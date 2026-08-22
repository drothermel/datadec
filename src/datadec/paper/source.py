from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from datadec.paper.models import ClaimRegistry, SourceRegion


class SourceValidationError(ValueError):
    """The frozen paper source cannot be validated without interpretation."""


@dataclass(frozen=True, slots=True)
class DependencyReport:
    entrypoint: str
    tex_files: tuple[str, ...]
    input_files: tuple[str, ...]
    graphics_files: tuple[str, ...]
    bibliography_files: tuple[str, ...]
    bibliography_style_files: tuple[str, ...]
    bbl_files: tuple[str, ...]
    citation_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CitationReport:
    citation_keys: tuple[str, ...]
    bib_keys: tuple[str, ...]
    bbl_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CoverageReport:
    claim_ids: tuple[str, ...]
    source_region_ids: tuple[str, ...]
    source_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ManuscriptSourceSurface:
    active_tex_lines: tuple[tuple[str, int], ...]
    asset_files: tuple[str, ...]
    excluded_implementation_files: tuple[str, ...]


_DEFAULT_MANUSCRIPT_ENTRYPOINT = "docs/paper/example_paper.tex"
_COMMAND_RE = re.compile(r"\\([A-Za-z@]+|.)")
_LITERAL_PATH_RE = re.compile(r"[A-Za-z0-9._/-]+")
_CITATION_COMMAND_RE = re.compile(r"cite(?:alp|alt|author|p|t|year|yearpar)?")
_BIB_ENTRY_RE = re.compile(
    r"(?m)^\s*@(?!comment\b|preamble\b|string\b)[A-Za-z]+\s*\{\s*([^,\s{}]+)\s*,",
    re.IGNORECASE,
)
_BBL_ENTRY_RE = re.compile(r"\\bibitem(?:\[[^]]*\])?\s*\{([^{}\s]+)\}")
_CONDITIONAL_COMMANDS = {
    "else",
    "fi",
    "if",
    "ifcase",
    "ifcat",
    "ifcsname",
    "ifdefined",
    "ifdim",
    "ifeof",
    "iffalse",
    "ifhbox",
    "ifhmode",
    "ifinner",
    "ifmmode",
    "ifnum",
    "ifodd",
    "iftrue",
    "ifvbox",
    "ifvmode",
    "ifvoid",
    "ifx",
    "unless",
}
_MACRO_DEFINITION_COMMANDS = {
    "def",
    "edef",
    "gdef",
    "newcommand",
    "providecommand",
    "renewcommand",
    "xdef",
}


def _normalized_repository_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SourceValidationError(
            f"path must be normalized repository-relative POSIX path: {value!r}"
        )
    return path


def _repository_path(repository_root: Path, value: str) -> Path:
    relative = _normalized_repository_path(value)
    root = repository_root.resolve()
    candidate = root.joinpath(*relative.parts)
    try:
        candidate.resolve().relative_to(root)
    except ValueError as error:
        raise SourceValidationError(
            f"path escapes repository root: {value!r}"
        ) from error
    return candidate


def _required_file(repository_root: Path, value: str) -> Path:
    path = _repository_path(repository_root, value)
    if not path.is_file():
        raise SourceValidationError(f"required source file does not exist: {value}")
    return path


def _raw_lines(repository_root: Path, source_file: str) -> list[bytes]:
    return (
        _required_file(repository_root, source_file)
        .read_bytes()
        .splitlines(keepends=True)
    )


def raw_line_slice_sha256(
    repository_root: str | Path,
    source_file: str,
    line_start: int,
    line_end: int,
) -> str:
    """Hash the exact bytes in an inclusive, one-indexed source line slice."""
    if line_start < 1 or line_end < line_start:
        raise SourceValidationError("line span must be one-indexed and nondecreasing")
    root = Path(repository_root)
    lines = _raw_lines(root, source_file)
    if line_end > len(lines):
        raise SourceValidationError(
            f"line span {line_start}-{line_end} is outside {source_file} "
            f"({len(lines)} lines)"
        )
    return hashlib.sha256(b"".join(lines[line_start - 1 : line_end])).hexdigest()


def _active_tex_line_numbers(
    repository_root: Path,
    source_file: str,
) -> tuple[int, ...]:
    text = _strip_unescaped_comments(
        _required_file(repository_root, source_file).read_text()
    )
    active_lines: list[int] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            active_lines.append(line_number)
        if re.search(r"\\end\s*\{document\}", line):
            break
    return tuple(active_lines)


def derive_manuscript_source_surface(
    repository_root: str | Path,
    entrypoint: str = _DEFAULT_MANUSCRIPT_ENTRYPOINT,
) -> ManuscriptSourceSurface:
    """Derive the complete semantic manuscript surface from its entrypoint."""
    root = Path(repository_root)
    dependencies = scan_tex_dependencies(root, entrypoint)
    active_tex_lines = tuple(
        (source_file, line_number)
        for source_file in dependencies.tex_files
        for line_number in _active_tex_line_numbers(root, source_file)
    )
    asset_files = tuple(
        sorted(
            {
                *dependencies.graphics_files,
                *dependencies.bibliography_files,
                *(dependencies.bbl_files if dependencies.bibliography_files else ()),
            }
        )
    )
    return ManuscriptSourceSurface(
        active_tex_lines=active_tex_lines,
        asset_files=asset_files,
        excluded_implementation_files=dependencies.bibliography_style_files,
    )


def _validate_locator(
    repository_root: Path,
    source_file: str,
    line_start: int,
    line_end: int,
    description: str,
) -> None:
    if line_start < 1 or line_end < line_start:
        raise SourceValidationError(f"invalid {description} line span")
    lines = _raw_lines(repository_root, source_file)
    if line_end > len(lines):
        raise SourceValidationError(
            f"{description} line span {line_start}-{line_end} is outside "
            f"{source_file} ({len(lines)} lines)"
        )


def _validate_nonoverlapping_regions(regions: tuple[SourceRegion, ...]) -> None:
    by_file: dict[str, list[SourceRegion]] = {}
    for region in regions:
        by_file.setdefault(region.source_file, []).append(region)

    for source_file, file_regions in by_file.items():
        ordered = sorted(
            file_regions,
            key=lambda region: (region.line_start, region.line_end, region.id),
        )
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if right.line_start > left.line_end:
                    break
                raise SourceValidationError(
                    f"source region spans overlap in {source_file}: "
                    f"{left.id} and {right.id}"
                )


def _citation_keys_in_source_span(
    repository_root: Path,
    source_file: str,
    line_start: int,
    line_end: int,
) -> tuple[str, ...]:
    lines = _raw_lines(repository_root, source_file)
    text = b"".join(lines[line_start - 1 : line_end]).decode()
    stripped = _strip_unescaped_comments(text)
    citation_keys: set[str] = set()
    for match in _COMMAND_RE.finditer(stripped):
        command = match.group(1)
        if _CITATION_COMMAND_RE.fullmatch(command):
            serialized_keys, _ = _parse_citation_keys(
                stripped,
                match.end(),
                command,
            )
            citation_keys.update(serialized_keys.split(","))
    return tuple(sorted(citation_keys))


def _validate_claim_citation_bindings(
    repository_root: Path,
    registry: ClaimRegistry,
) -> None:
    for claim in registry.claims:
        if not claim.citation_keys:
            continue
        present_keys = set(
            _citation_keys_in_source_span(
                repository_root,
                claim.source_file,
                claim.line_start,
                claim.line_end,
            )
        )
        missing_keys = sorted(set(claim.citation_keys) - present_keys)
        if missing_keys:
            raise SourceValidationError(
                f"claim {claim.id} citation keys are absent from its source span: "
                f"{', '.join(missing_keys)}"
            )


def _validate_surface_partition(
    repository_root: Path,
    regions: tuple[SourceRegion, ...],
    surface: ManuscriptSourceSurface,
) -> None:
    active_tex_lines = set(surface.active_tex_lines)
    tex_files = {source_file for source_file, _ in active_tex_lines}
    asset_files = set(surface.asset_files)
    surface_files = tex_files | asset_files

    disconnected_regions = sorted(
        region.id for region in regions if region.source_file not in surface_files
    )
    if disconnected_regions:
        raise SourceValidationError(
            "source regions are outside the active manuscript surface: "
            f"{', '.join(disconnected_regions)}"
        )

    line_coverage = {source_line: 0 for source_line in active_tex_lines}
    regions_by_file: dict[str, list[SourceRegion]] = {}
    for region in regions:
        regions_by_file.setdefault(region.source_file, []).append(region)
        if region.source_file not in tex_files:
            continue
        for line_number in range(region.line_start, region.line_end + 1):
            source_line = (region.source_file, line_number)
            if source_line in line_coverage:
                line_coverage[source_line] += 1

    missing_lines = sorted(
        source_line for source_line, count in line_coverage.items() if count == 0
    )
    duplicate_lines = sorted(
        source_line for source_line, count in line_coverage.items() if count > 1
    )
    if missing_lines or duplicate_lines:
        details: list[str] = []
        if missing_lines:
            details.append(
                "missing="
                + ", ".join(
                    f"{source_file}:{line_number}"
                    for source_file, line_number in missing_lines
                )
            )
        if duplicate_lines:
            details.append(
                "duplicate="
                + ", ".join(
                    f"{source_file}:{line_number}"
                    for source_file, line_number in duplicate_lines
                )
            )
        raise SourceValidationError(
            "active manuscript TeX lines require exactly one source region ("
            + "; ".join(details)
            + ")"
        )

    for asset_file in sorted(asset_files):
        asset_regions = regions_by_file.get(asset_file, [])
        line_count = len(_raw_lines(repository_root, asset_file))
        if len(asset_regions) != 1 or (
            asset_regions[0].line_start,
            asset_regions[0].line_end,
        ) != (1, line_count):
            raise SourceValidationError(
                f"referenced manuscript asset {asset_file} requires exactly one "
                f"whole-file source region spanning lines 1-{line_count}"
            )


def validate_source_coverage(
    repository_root: str | Path,
    registry: ClaimRegistry,
    entrypoint: str = _DEFAULT_MANUSCRIPT_ENTRYPOINT,
) -> CoverageReport:
    root = Path(repository_root)
    claims_by_id = {claim.id: claim for claim in registry.claims}

    for claim in registry.claims:
        _validate_locator(
            root,
            claim.source_file,
            claim.line_start,
            claim.line_end,
            f"claim {claim.id}",
        )

    _validate_claim_citation_bindings(root, registry)
    _validate_nonoverlapping_regions(registry.source_regions)
    covered_claim_ids: set[str] = set()
    for region in registry.source_regions:
        _validate_locator(
            root,
            region.source_file,
            region.line_start,
            region.line_end,
            f"source region {region.id}",
        )
        actual_digest = raw_line_slice_sha256(
            root,
            region.source_file,
            region.line_start,
            region.line_end,
        )
        if actual_digest != region.content_sha256:
            raise SourceValidationError(
                f"source region {region.id} digest mismatch: expected "
                f"{region.content_sha256}, found {actual_digest}"
            )

        for claim_id in region.claim_ids:
            claim = claims_by_id.get(claim_id)
            if claim is None:
                raise SourceValidationError(
                    f"source region {region.id} references unknown claim {claim_id}"
                )
            if claim.source_file != region.source_file or not (
                region.line_start <= claim.line_start
                and claim.line_end <= region.line_end
            ):
                raise SourceValidationError(
                    f"source region {region.id} does not contain mapped claim "
                    f"{claim_id}"
                )
            covered_claim_ids.add(claim_id)

    uncovered = sorted(set(claims_by_id) - covered_claim_ids)
    if uncovered:
        raise SourceValidationError(
            f"claims are not covered by claim-bearing source regions: {', '.join(uncovered)}"
        )

    surface = derive_manuscript_source_surface(root, entrypoint)
    _validate_surface_partition(root, registry.source_regions, surface)

    return CoverageReport(
        claim_ids=tuple(sorted(claims_by_id)),
        source_region_ids=tuple(
            sorted(region.id for region in registry.source_regions)
        ),
        source_files=tuple(
            sorted(
                {source_file for source_file, _ in surface.active_tex_lines}
                | set(surface.asset_files)
            )
        ),
    )


def _strip_unescaped_comments(text: str) -> str:
    stripped: list[str] = []
    for line in text.splitlines(keepends=True):
        comment_at: int | None = None
        for index, character in enumerate(line):
            if character != "%":
                continue
            preceding_backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                preceding_backslashes += 1
                cursor -= 1
            if preceding_backslashes % 2 == 0:
                comment_at = index
                break
        if comment_at is None:
            stripped.append(line)
        elif line.endswith("\n"):
            stripped.append(line[:comment_at] + "\n")
        else:
            stripped.append(line[:comment_at])
    return "".join(stripped)


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _parse_group(
    text: str,
    position: int,
    opening: str,
    closing: str,
    description: str,
) -> tuple[str, int]:
    position = _skip_whitespace(text, position)
    if position >= len(text) or text[position] != opening:
        raise SourceValidationError(f"malformed {description}: expected {opening}")
    depth = 1
    cursor = position + 1
    while cursor < len(text):
        character = text[cursor]
        if character == "\\":
            cursor += 2
            continue
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return text[position + 1 : cursor], cursor + 1
        cursor += 1
    raise SourceValidationError(f"malformed {description}: unterminated {opening}")


def _literal_path(value: str, description: str) -> str:
    if not _LITERAL_PATH_RE.fullmatch(value):
        raise SourceValidationError(
            f"{description} must use a literal normalized filename: {value!r}"
        )
    _normalized_repository_path(value)
    return value


def _source_relative_path(
    source_root: PurePosixPath,
    literal: str,
    suffix: str | None,
    description: str,
) -> str:
    value = _literal_path(literal, description)
    path = PurePosixPath(value)
    if suffix is not None and not path.suffix:
        path = path.with_suffix(suffix)
    combined = source_root / path
    return _normalized_repository_path(combined.as_posix()).as_posix()


def _has_parameterized_macro_definition(
    text: str, command: str, command_end: int
) -> bool:
    cursor = _skip_whitespace(text, command_end)
    if cursor < len(text) and text[cursor] == "*":
        cursor = _skip_whitespace(text, cursor + 1)

    if command in {"newcommand", "providecommand", "renewcommand"}:
        if cursor < len(text) and text[cursor] == "{":
            _, cursor = _parse_group(text, cursor, "{", "}", command)
        else:
            macro = _COMMAND_RE.match(text, cursor)
            if macro is None:
                raise SourceValidationError(f"malformed \\{command} definition")
            cursor = macro.end()
        cursor = _skip_whitespace(text, cursor)
        if cursor < len(text) and text[cursor] == "[":
            parameter_count, _ = _parse_group(text, cursor, "[", "]", command)
            if not parameter_count.isdigit():
                raise SourceValidationError(f"malformed \\{command} parameter count")
            return int(parameter_count) > 0
        return False

    body_at = text.find("{", cursor)
    if body_at < 0:
        raise SourceValidationError(f"malformed \\{command} definition")
    return bool(re.search(r"(?<!\\)#[1-9]", text[cursor:body_at]))


@dataclass(slots=True)
class _DependencyAccumulator:
    tex_files: set[str]
    input_files: set[str]
    graphics_files: set[str]
    bibliography_files: set[str]
    bibliography_style_files: set[str]
    citation_keys: set[str]
    visiting: set[str]


def _parse_citation_keys(text: str, position: int, command: str) -> tuple[str, int]:
    cursor = _skip_whitespace(text, position)
    for _ in range(2):
        if cursor < len(text) and text[cursor] == "[":
            _, cursor = _parse_group(text, cursor, "[", "]", f"\\{command}")
            cursor = _skip_whitespace(text, cursor)
    keys_text, cursor = _parse_group(text, cursor, "{", "}", f"\\{command}")
    keys = [key.strip() for key in keys_text.split(",")]
    if not keys or any(
        not key or not re.fullmatch(r"[A-Za-z0-9_.:+/-]+", key) for key in keys
    ):
        raise SourceValidationError(
            f"\\{command} must contain a literal comma-separated citation-key list"
        )
    return ",".join(keys), cursor


def _scan_tex_file(
    repository_root: Path,
    source_root: PurePosixPath,
    source_file: str,
    accumulator: _DependencyAccumulator,
) -> bool:
    if source_file in accumulator.visiting:
        raise SourceValidationError(f"cyclic \\input dependency: {source_file}")
    if source_file in accumulator.tex_files:
        return False
    accumulator.visiting.add(source_file)
    accumulator.tex_files.add(source_file)
    text = _strip_unescaped_comments(
        _required_file(repository_root, source_file).read_text()
    )

    try:
        for match in _COMMAND_RE.finditer(text):
            command = match.group(1)
            if (
                command in _CONDITIONAL_COMMANDS
                or command.startswith("if")
                and command != "iff"
            ):
                raise SourceValidationError(
                    f"active conditional control sequence \\{command} in {source_file}"
                )
            if command in {"catcode", "csname"}:
                raise SourceValidationError(
                    f"unsupported active control sequence \\{command} in {source_file}"
                )
            if (
                command in _MACRO_DEFINITION_COMMANDS
                and _has_parameterized_macro_definition(text, command, match.end())
            ):
                raise SourceValidationError(
                    f"semantic parameterized macro definition \\{command} in "
                    f"{source_file}"
                )
            if command == "end":
                environment, _ = _parse_group(text, match.end(), "{", "}", "\\end")
                if environment == "document":
                    return True
                continue
            if command == "include":
                raise SourceValidationError(
                    f"active \\include is outside the supported source subset in {source_file}"
                )
            if command == "input":
                literal, _ = _parse_group(text, match.end(), "{", "}", "\\input")
                dependency = _source_relative_path(
                    source_root, literal, ".tex", "\\input"
                )
                _required_file(repository_root, dependency)
                accumulator.input_files.add(dependency)
                if _scan_tex_file(
                    repository_root,
                    source_root,
                    dependency,
                    accumulator,
                ):
                    return True
                continue
            if command == "includegraphics":
                cursor = _skip_whitespace(text, match.end())
                if cursor < len(text) and text[cursor] == "*":
                    cursor = _skip_whitespace(text, cursor + 1)
                if cursor < len(text) and text[cursor] == "[":
                    _, cursor = _parse_group(
                        text, cursor, "[", "]", "\\includegraphics options"
                    )
                literal, _ = _parse_group(text, cursor, "{", "}", "\\includegraphics")
                dependency = _source_relative_path(
                    source_root, literal, None, "\\includegraphics"
                )
                _required_file(repository_root, dependency)
                accumulator.graphics_files.add(dependency)
                continue
            if command in {"bibliography", "bibliographystyle"}:
                literals, _ = _parse_group(text, match.end(), "{", "}", f"\\{command}")
                values = [value.strip() for value in literals.split(",")]
                if not values or any(not value for value in values):
                    raise SourceValidationError(f"malformed \\{command} filename list")
                for value in values:
                    suffix = ".bib" if command == "bibliography" else ".bst"
                    dependency = _source_relative_path(
                        source_root, value, suffix, f"\\{command}"
                    )
                    _required_file(repository_root, dependency)
                    target = (
                        accumulator.bibliography_files
                        if command == "bibliography"
                        else accumulator.bibliography_style_files
                    )
                    target.add(dependency)
                continue
            if _CITATION_COMMAND_RE.fullmatch(command):
                serialized_keys, _ = _parse_citation_keys(text, match.end(), command)
                accumulator.citation_keys.update(serialized_keys.split(","))
    finally:
        accumulator.visiting.remove(source_file)
    return False


def scan_tex_dependencies(
    repository_root: str | Path,
    entrypoint: str,
) -> DependencyReport:
    root = Path(repository_root)
    normalized_entrypoint = _normalized_repository_path(entrypoint).as_posix()
    _required_file(root, normalized_entrypoint)
    source_root = PurePosixPath(normalized_entrypoint).parent
    accumulator = _DependencyAccumulator(
        tex_files=set(),
        input_files=set(),
        graphics_files=set(),
        bibliography_files=set(),
        bibliography_style_files=set(),
        citation_keys=set(),
        visiting=set(),
    )
    _scan_tex_file(root, source_root, normalized_entrypoint, accumulator)

    bbl_file = PurePosixPath(normalized_entrypoint).with_suffix(".bbl").as_posix()
    _required_file(root, bbl_file)
    return DependencyReport(
        entrypoint=normalized_entrypoint,
        tex_files=tuple(sorted(accumulator.tex_files)),
        input_files=tuple(sorted(accumulator.input_files)),
        graphics_files=tuple(sorted(accumulator.graphics_files)),
        bibliography_files=tuple(sorted(accumulator.bibliography_files)),
        bibliography_style_files=tuple(sorted(accumulator.bibliography_style_files)),
        bbl_files=(bbl_file,),
        citation_keys=tuple(sorted(accumulator.citation_keys)),
    )


def _unique_keys(keys: list[str], description: str) -> tuple[str, ...]:
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise SourceValidationError(
            f"duplicate {description} keys: {', '.join(duplicates)}"
        )
    return tuple(sorted(keys))


def validate_citations(
    repository_root: str | Path,
    entrypoint: str,
) -> CitationReport:
    root = Path(repository_root)
    dependencies = scan_tex_dependencies(root, entrypoint)
    if not dependencies.bibliography_files:
        raise SourceValidationError("paper has no literal bibliography source")

    bib_keys_list: list[str] = []
    for bibliography_file in dependencies.bibliography_files:
        text = _required_file(root, bibliography_file).read_text()
        bib_keys_list.extend(_BIB_ENTRY_RE.findall(text))
    bib_keys = _unique_keys(bib_keys_list, "bibliography")

    bbl_keys_list: list[str] = []
    for bbl_file in dependencies.bbl_files:
        text = _required_file(root, bbl_file).read_text()
        bbl_keys_list.extend(_BBL_ENTRY_RE.findall(text))
    bbl_keys = _unique_keys(bbl_keys_list, "compiled bibliography")

    citation_keys = dependencies.citation_keys
    missing_from_bib = sorted(set(citation_keys) - set(bib_keys))
    missing_from_bbl = sorted(set(citation_keys) - set(bbl_keys))
    if missing_from_bib or missing_from_bbl:
        details: list[str] = []
        if missing_from_bib:
            details.append(f"bib: {', '.join(missing_from_bib)}")
        if missing_from_bbl:
            details.append(f"bbl: {', '.join(missing_from_bbl)}")
        raise SourceValidationError(
            f"unresolved active citation keys ({'; '.join(details)})"
        )

    return CitationReport(
        citation_keys=citation_keys,
        bib_keys=bib_keys,
        bbl_keys=bbl_keys,
    )


__all__ = [
    "CitationReport",
    "CoverageReport",
    "DependencyReport",
    "ManuscriptSourceSurface",
    "SourceValidationError",
    "derive_manuscript_source_surface",
    "raw_line_slice_sha256",
    "scan_tex_dependencies",
    "validate_citations",
    "validate_source_coverage",
]

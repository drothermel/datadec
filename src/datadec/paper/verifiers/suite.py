from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import UNIQUE, StrEnum, verify
from pathlib import Path

from datadec.config import DataDecideCatalog, load_catalog
from datadec.data.model_utils import create_model_config
from datadec.paper.models import EvidenceBoundary

_REPOSITORY_ROOT = Path(__file__).parents[4]
_SUITE_TABLE_PATH = Path("docs/paper/tables/suite_stats.tex")
_EXPECTED_ROW_COUNT = 14
_EXPECTED_SEQUENCE_LENGTH = 2_024
_EXPECTED_MLP_RATIO = 8
_EXPECTED_RECIPE_COUNT = 25
_EXPECTED_SEEDS_PER_CONFIGURATION = 3
_EXPECTED_SEED_ALIASES: tuple[tuple[str, int], ...] = (
    ("default", 0),
    ("small aux 2", 1),
    ("small aux 3", 2),
    ("large aux 2", 3),
    ("large aux 3", 4),
)
_HEADER = (
    "Model name & Batch size & Hidden dim. & LR & Model size & Heads & Layers & "
    r"Training steps & Tokens trained \\"
)
_INTEGER_PATTERN = r"(?:0|[1-9]\d{0,2}(?:,\d{3})*)"
_ROW_PATTERN = re.compile(
    rf"^(?P<model_name>[1-9]\d*[MB]) & "
    rf"(?P<batch_size>{_INTEGER_PATTERN}) & "
    rf"(?P<hidden_dimension>{_INTEGER_PATTERN}) & "
    r"(?P<learning_rate>\d\.\de[+-]\d{2}) & "
    r"(?P<model_size>\d+\.\dM) & "
    rf"(?P<heads>{_INTEGER_PATTERN}) & "
    rf"(?P<layers>{_INTEGER_PATTERN}) & "
    rf"(?P<training_steps>{_INTEGER_PATTERN}) & "
    r"(?P<tokens_trained>\d+\.\dB) \\\\$"
)


@verify(UNIQUE)
class SuiteField(StrEnum):
    MODEL_NAME = "model_name"
    BATCH_SIZE = "batch_size"
    HIDDEN_DIMENSION = "hidden_dimension"
    LEARNING_RATE = "learning_rate"
    MODEL_SIZE = "model_size"
    HEADS = "heads"
    LAYERS = "layers"
    TRAINING_STEPS = "training_steps"
    TOKENS_TRAINED = "tokens_trained"


@verify(UNIQUE)
class CheckStatus(StrEnum):
    MATCH = "match"
    CONTRADICTION = "contradiction"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class PaperSuiteRow:
    model_name: str
    batch_size: int
    hidden_dimension: int
    learning_rate: Decimal
    model_size_millions: Decimal
    heads: int
    layers: int
    training_steps: int
    tokens_trained_billions: Decimal

    def display(self, field: SuiteField) -> str:
        if field is SuiteField.MODEL_NAME:
            return self.model_name
        if field is SuiteField.BATCH_SIZE:
            return f"{self.batch_size:,}"
        if field is SuiteField.HIDDEN_DIMENSION:
            return f"{self.hidden_dimension:,}"
        if field is SuiteField.LEARNING_RATE:
            return f"{float(self.learning_rate):.1e}"
        if field is SuiteField.MODEL_SIZE:
            return f"{self.model_size_millions:.1f}M"
        if field is SuiteField.HEADS:
            return f"{self.heads:,}"
        if field is SuiteField.LAYERS:
            return f"{self.layers:,}"
        if field is SuiteField.TRAINING_STEPS:
            return f"{self.training_steps:,}"
        return f"{self.tokens_trained_billions:.1f}B"


@dataclass(frozen=True, slots=True)
class DerivedSuiteRow:
    model_name: str
    batch_size: int
    hidden_dimension: int
    learning_rate: float
    exact_parameter_count: int
    heads: int
    layers: int
    training_steps: int
    tokens_trained: int

    def display(self, field: SuiteField) -> str:
        if field is SuiteField.MODEL_NAME:
            return self.model_name
        if field is SuiteField.BATCH_SIZE:
            return f"{self.batch_size:,}"
        if field is SuiteField.HIDDEN_DIMENSION:
            return f"{self.hidden_dimension:,}"
        if field is SuiteField.LEARNING_RATE:
            return f"{self.learning_rate:.1e}"
        if field is SuiteField.MODEL_SIZE:
            return f"{self.exact_parameter_count / 1_000_000:.1f}M"
        if field is SuiteField.HEADS:
            return f"{self.heads:,}"
        if field is SuiteField.LAYERS:
            return f"{self.layers:,}"
        if field is SuiteField.TRAINING_STEPS:
            return f"{self.training_steps:,}"
        return f"{self.tokens_trained / 1_000_000_000:.1f}B"


@dataclass(frozen=True, slots=True)
class SuiteFieldMatch:
    field: SuiteField
    expected_display: str
    observed_display: str
    matches: bool


@dataclass(frozen=True, slots=True)
class SuiteRowVerification:
    claim_id: str
    expected: PaperSuiteRow
    observed: DerivedSuiteRow
    field_matches: tuple[SuiteFieldMatch, ...]
    available_evidence_boundary: EvidenceBoundary
    required_evidence_boundary: EvidenceBoundary

    @property
    def matches(self) -> bool:
        return all(field.matches for field in self.field_matches)

    def match_for(self, field: SuiteField) -> SuiteFieldMatch:
        return next(match for match in self.field_matches if match.field is field)


@dataclass(frozen=True, slots=True)
class SuiteFact:
    id: str
    claim_id: str | None
    expected: str
    observed: str | None
    status: CheckStatus
    available_evidence_boundary: EvidenceBoundary
    required_evidence_boundary: EvidenceBoundary
    reason: str | None = None

    @property
    def matches(self) -> bool | None:
        if self.status is CheckStatus.UNSUPPORTED:
            return None
        return self.status is CheckStatus.MATCH


@dataclass(frozen=True, slots=True)
class SuiteVerification:
    rows: tuple[SuiteRowVerification, ...]
    facts: tuple[SuiteFact, ...]

    def fact(self, fact_id: str) -> SuiteFact:
        return next(fact for fact in self.facts if fact.id == fact_id)


def _parse_integer(value: str) -> int:
    return int(value.replace(",", ""))


def _parse_row(line: str) -> PaperSuiteRow:
    match = _ROW_PATTERN.fullmatch(line)
    if match is None:
        raise ValueError(f"malformed suite table row: {line!r}")
    values = match.groupdict()
    return PaperSuiteRow(
        model_name=values["model_name"],
        batch_size=_parse_integer(values["batch_size"]),
        hidden_dimension=_parse_integer(values["hidden_dimension"]),
        learning_rate=Decimal(values["learning_rate"]),
        model_size_millions=Decimal(values["model_size"][:-1]),
        heads=_parse_integer(values["heads"]),
        layers=_parse_integer(values["layers"]),
        training_steps=_parse_integer(values["training_steps"]),
        tokens_trained_billions=Decimal(values["tokens_trained"][:-1]),
    )


def parse_suite_table(path: str | Path) -> tuple[PaperSuiteRow, ...]:
    lines = Path(path).read_text().splitlines()
    expected_line_count = _EXPECTED_ROW_COUNT + 6
    if len(lines) != expected_line_count:
        raise ValueError(
            "suite table must contain exactly "
            f"{_EXPECTED_ROW_COUNT} data rows; found {len(lines) - 6}"
        )
    expected_scaffolding = {
        0: r"\begin{tabular}{p{1cm}p{1cm}p{1cm}p{1.25cm}p{1cm}p{1cm}p{1cm}p{1cm}p{1cm}}",
        1: r"\toprule",
        2: _HEADER,
        3: r"\midrule",
        -2: r"\bottomrule",
        -1: r"\end{tabular}",
    }
    for index, expected in expected_scaffolding.items():
        if lines[index] != expected:
            raise ValueError(f"malformed suite table scaffolding at line {index + 1}")

    rows = tuple(_parse_row(line) for line in lines[4:-2])
    model_names = tuple(row.model_name for row in rows)
    if len(model_names) != len(set(model_names)):
        raise ValueError("suite table model names must be unique")
    return rows


def derive_suite_rows(
    catalog: DataDecideCatalog | None = None,
) -> tuple[DerivedSuiteRow, ...]:
    current_catalog = load_catalog() if catalog is None else catalog
    rows: list[DerivedSuiteRow] = []
    for model in current_catalog.models:
        config = create_model_config(model.name)
        rows.append(
            DerivedSuiteRow(
                model_name=model.name,
                batch_size=int(config["batch_size"]),
                hidden_dimension=model.d_model,
                learning_rate=float(config["lr_max"]),
                exact_parameter_count=model.exact_parameter_count,
                heads=model.n_heads,
                layers=model.n_layers,
                training_steps=int(config["total_steps"]),
                tokens_trained=int(config["total_tokens"]),
            )
        )
    return tuple(rows)


def compare_suite_row(
    expected: PaperSuiteRow,
    observed: DerivedSuiteRow,
    *,
    claim_id: str,
) -> SuiteRowVerification:
    field_matches = tuple(
        SuiteFieldMatch(
            field=field,
            expected_display=expected.display(field),
            observed_display=observed.display(field),
            matches=expected.display(field) == observed.display(field),
        )
        for field in SuiteField
    )
    return SuiteRowVerification(
        claim_id=claim_id,
        expected=expected,
        observed=observed,
        field_matches=field_matches,
        available_evidence_boundary=EvidenceBoundary.PAPER_OR_FINAL_ARTIFACT,
        required_evidence_boundary=EvidenceBoundary.TRAINING_RERUN,
    )


def _fact(
    *,
    fact_id: str,
    claim_id: str | None,
    expected: str,
    observed: str,
) -> SuiteFact:
    status = CheckStatus.MATCH if observed == expected else CheckStatus.CONTRADICTION
    return SuiteFact(
        id=fact_id,
        claim_id=claim_id,
        expected=expected,
        observed=observed,
        status=status,
        available_evidence_boundary=EvidenceBoundary.PAPER_OR_FINAL_ARTIFACT,
        required_evidence_boundary=EvidenceBoundary.TRAINING_RERUN,
    )


def _unsupported_fact(
    *, fact_id: str, claim_id: str, expected: str, reason: str
) -> SuiteFact:
    return SuiteFact(
        id=fact_id,
        claim_id=claim_id,
        expected=expected,
        observed=None,
        status=CheckStatus.UNSUPPORTED,
        available_evidence_boundary=EvidenceBoundary.PAPER_OR_FINAL_ARTIFACT,
        required_evidence_boundary=EvidenceBoundary.TRAINING_RERUN,
        reason=reason,
    )


def _uniform_values(values: set[int]) -> str:
    return ", ".join(str(value) for value in sorted(values))


def _suite_facts(catalog: DataDecideCatalog) -> tuple[SuiteFact, ...]:
    recipes = tuple(
        recipe
        for family_recipes in catalog.data_recipe_families.values()
        for recipe in family_recipes
    )
    sequence_lengths = {
        catalog.training.max_sequence_length,
        catalog.model_defaults.max_sequence_length,
    }
    mlp_ratios = {catalog.model_defaults.mlp_ratio} | {
        model.mlp_ratio for model in catalog.models
    }
    seed_aliases = tuple(catalog.seed_map.items())
    return (
        _fact(
            fact_id="configuration_count",
            claim_id="DD-0267",
            expected=str(_EXPECTED_ROW_COUNT),
            observed=str(len(catalog.models)),
        ),
        _fact(
            fact_id="recipe_count",
            claim_id="DD-0271",
            expected=str(_EXPECTED_RECIPE_COUNT),
            observed=str(len(set(recipes))),
        ),
        _fact(
            fact_id="sequence_length",
            claim_id="DD-0269",
            expected=str(_EXPECTED_SEQUENCE_LENGTH),
            observed=_uniform_values(sequence_lengths),
        ),
        _fact(
            fact_id="mlp_ratio",
            claim_id="DD-0270",
            expected=str(_EXPECTED_MLP_RATIO),
            observed=_uniform_values(mlp_ratios),
        ),
        _fact(
            fact_id="seed_aliases",
            claim_id=None,
            expected=repr(_EXPECTED_SEED_ALIASES),
            observed=repr(seed_aliases),
        ),
        _unsupported_fact(
            fact_id="seeds_per_recipe_configuration",
            claim_id="DD-0272",
            expected=str(_EXPECTED_SEEDS_PER_CONFIGURATION),
            reason=(
                "the catalog defines five seed aliases but does not map three aliases "
                "to every recipe/configuration pair"
            ),
        ),
        _unsupported_fact(
            fact_id="early_seed_stop_policy",
            claim_id="DD-0273",
            expected="non-default seeds below 1B stop at 25% of 1B compute",
            reason=(
                "the catalog and model schedule derivations do not encode per-run "
                "seed stopping policy or completed training-run evidence"
            ),
        ),
    )


def verify_suite(
    expected_rows: tuple[PaperSuiteRow, ...],
    catalog: DataDecideCatalog | None = None,
) -> SuiteVerification:
    current_catalog = load_catalog() if catalog is None else catalog
    observed_rows = derive_suite_rows(current_catalog)
    expected_by_name = {row.model_name: row for row in expected_rows}
    if len(expected_by_name) != len(expected_rows):
        raise ValueError("suite table model names must be unique")
    observed_names = tuple(row.model_name for row in observed_rows)
    expected_names = set(expected_by_name)
    if expected_names != set(observed_names):
        missing = sorted(set(observed_names) - expected_names)
        extra = sorted(expected_names - set(observed_names))
        raise ValueError(
            f"suite table model rows differ from catalog: missing={missing}, extra={extra}"
        )

    rows = tuple(
        compare_suite_row(
            expected_by_name[observed.model_name],
            observed,
            claim_id=f"DD-{index:04d}",
        )
        for index, observed in enumerate(observed_rows, start=276)
    )
    return SuiteVerification(rows=rows, facts=_suite_facts(current_catalog))


def verify_repository_suite(
    repository_root: str | Path | None = None,
) -> SuiteVerification:
    root = _REPOSITORY_ROOT if repository_root is None else Path(repository_root)
    return verify_suite(parse_suite_table(root / _SUITE_TABLE_PATH))


__all__ = [
    "CheckStatus",
    "DerivedSuiteRow",
    "PaperSuiteRow",
    "SuiteFact",
    "SuiteField",
    "SuiteFieldMatch",
    "SuiteRowVerification",
    "SuiteVerification",
    "compare_suite_row",
    "derive_suite_rows",
    "parse_suite_table",
    "verify_repository_suite",
    "verify_suite",
]

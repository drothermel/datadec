from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from datadec.config import load_catalog
from datadec.paper.models import EvidenceBoundary
from datadec.paper.verifiers.suite import (
    CheckStatus,
    DerivedSuiteRow,
    SuiteField,
    compare_suite_row,
    parse_suite_table,
    verify_repository_suite,
    verify_suite,
)

_REPOSITORY_ROOT = Path(__file__).parents[2]
_TABLE_PATH = _REPOSITORY_ROOT / "docs/paper/tables/suite_stats.tex"


def test_current_suite_derives_all_fourteen_catalog_rows() -> None:
    verification = verify_repository_suite()

    assert len(verification.rows) == 14
    assert tuple(row.expected.model_name for row in verification.rows) == tuple(
        model.name for model in load_catalog().models
    )
    assert tuple(row.claim_id for row in verification.rows) == tuple(
        f"DD-{index:04d}" for index in range(276, 290)
    )
    assert all(len(row.field_matches) == len(SuiteField) for row in verification.rows)


def test_current_suite_reports_every_canonical_table_contradiction() -> None:
    verification = verify_repository_suite()
    mismatches = {
        (row.expected.model_name, match.field): (
            match.expected_display,
            match.observed_display,
        )
        for row in verification.rows
        for match in row.field_matches
        if not match.matches
    }

    assert mismatches == {
        ("4M", SuiteField.TRAINING_STEPS): ("5,725", "5,715"),
        ("6M", SuiteField.TRAINING_STEPS): ("9,182", "9,172"),
        ("8M", SuiteField.TRAINING_STEPS): ("13,039", "13,029"),
        ("10M", SuiteField.TRAINING_STEPS): ("15,117", "15,107"),
        ("14M", SuiteField.TRAINING_STEPS): ("21,953", "21,943"),
        ("16M", SuiteField.TRAINING_STEPS): ("24,432", "24,422"),
        ("20M", SuiteField.TRAINING_STEPS): ("14,584", "14,574"),
        ("60M", SuiteField.TRAINING_STEPS): ("29,042", "29,032"),
        ("90M", SuiteField.TRAINING_STEPS): ("29,901", "29,891"),
        ("150M", SuiteField.TRAINING_STEPS): ("38,157", "38,147"),
        ("300M", SuiteField.TRAINING_STEPS): ("45,787", "45,777"),
        ("530M", SuiteField.TRAINING_STEPS): ("57,786", "57,766"),
        ("750M", SuiteField.TRAINING_STEPS): ("63,589", "63,579"),
        ("1B", SuiteField.LEARNING_RATE): ("2.1e-03", "2.2e-03"),
        ("1B", SuiteField.TRAINING_STEPS): ("69,369", "69,359"),
    }
    assert all(
        row.matches is all(match.matches for match in row.field_matches)
        for row in verification.rows
    )


def test_suite_scalar_facts_preserve_contradiction_and_unsupported_states() -> None:
    verification = verify_repository_suite()

    sequence_length = verification.fact("sequence_length")
    assert sequence_length.expected == "2024"
    assert sequence_length.observed == "2048"
    assert sequence_length.status is CheckStatus.CONTRADICTION
    assert sequence_length.matches is False

    assert verification.fact("configuration_count").matches is True
    assert verification.fact("recipe_count").matches is True
    assert verification.fact("mlp_ratio").matches is True
    assert verification.fact("seed_aliases").matches is True

    seed_count = verification.fact("seeds_per_recipe_configuration")
    early_stop = verification.fact("early_seed_stop_policy")
    assert seed_count.status is CheckStatus.UNSUPPORTED
    assert seed_count.observed is None
    assert seed_count.matches is None
    assert seed_count.reason is not None
    assert early_stop.status is CheckStatus.UNSUPPORTED
    assert early_stop.observed is None
    assert early_stop.matches is None
    assert early_stop.reason is not None


def test_suite_facts_never_promote_catalog_evidence_to_training_rerun() -> None:
    verification = verify_repository_suite()

    assert all(
        fact.available_evidence_boundary is EvidenceBoundary.PAPER_OR_FINAL_ARTIFACT
        for fact in verification.facts
    )
    assert all(
        fact.required_evidence_boundary is EvidenceBoundary.TRAINING_RERUN
        for fact in verification.facts
    )
    assert all(
        row.available_evidence_boundary is EvidenceBoundary.PAPER_OR_FINAL_ARTIFACT
        for row in verification.rows
    )
    assert all(
        row.required_evidence_boundary is EvidenceBoundary.TRAINING_RERUN
        for row in verification.rows
    )


def test_row_comparison_uses_field_specific_display_rounding() -> None:
    expected = parse_suite_table(_TABLE_PATH)[0]
    observed = DerivedSuiteRow(
        model_name="4M",
        batch_size=32,
        hidden_dimension=64,
        learning_rate=0.01449,
        exact_parameter_count=3_749_999,
        heads=8,
        layers=8,
        training_steps=5_725,
        tokens_trained=449_999_999,
    )

    comparison = compare_suite_row(expected, observed, claim_id="DD-0276")

    assert comparison.matches
    assert comparison.match_for(SuiteField.LEARNING_RATE).observed_display == "1.4e-02"
    assert comparison.match_for(SuiteField.MODEL_SIZE).observed_display == "3.7M"
    assert comparison.match_for(SuiteField.TOKENS_TRAINED).observed_display == "0.4B"


def test_suite_verification_is_invariant_to_table_row_permutation() -> None:
    rows = parse_suite_table(_TABLE_PATH)

    forward = verify_suite(rows)
    reversed_rows = verify_suite(tuple(reversed(rows)))

    assert reversed_rows == forward


@pytest.mark.parametrize(
    ("old", "new", "error"),
    [
        ("Model name &", "Model &", "scaffolding"),
        ("4M & 32 &", "4M | 32 &", "malformed suite table row"),
    ],
)
def test_suite_table_parser_rejects_malformed_input(
    tmp_path: Path, old: str, new: str, error: str
) -> None:
    path = tmp_path / "suite_stats.tex"
    path.write_text(_TABLE_PATH.read_text().replace(old, new, 1))

    with pytest.raises(ValueError, match=error):
        parse_suite_table(path)


def test_suite_table_parser_rejects_duplicate_rows(tmp_path: Path) -> None:
    lines = _TABLE_PATH.read_text().splitlines()
    lines[5] = lines[4]
    path = tmp_path / "suite_stats.tex"
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="model names must be unique"):
        parse_suite_table(path)


def test_suite_table_parser_rejects_missing_rows(tmp_path: Path) -> None:
    lines = _TABLE_PATH.read_text().splitlines()
    del lines[4]
    path = tmp_path / "suite_stats.tex"
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="exactly 14 data rows"):
        parse_suite_table(path)


def test_suite_verification_rejects_wrong_model_universe() -> None:
    rows = parse_suite_table(_TABLE_PATH)
    wrong_name = replace(rows[0], model_name="5M")

    with pytest.raises(ValueError, match=r"missing=\['4M'\], extra=\['5M'\]"):
        verify_suite((wrong_name, *rows[1:]))


def test_direct_suite_verification_rejects_duplicate_rows() -> None:
    rows = parse_suite_table(_TABLE_PATH)

    with pytest.raises(ValueError, match="model names must be unique"):
        verify_suite((rows[0], rows[0], *rows[2:]))

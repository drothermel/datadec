from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from datadec.config import DataDecideCatalog, load_paper_validation_contract
from datadec.paper.contracts import load_repository_claim_registry, load_toml_model
from datadec.paper.models import AttemptRole, EvidenceLevel, ValidationOutcome
from datadec.paper.verifiers import scaling

_ROOT = Path(__file__).parents[2]
_REAL_DATA = Path.home() / "drotherm/repos/datadec/data"


def _row(
    *,
    task: str = "olmes_10_macro_avg",
    mix: str = "a",
    setup: str = "3_param",
    actual: float = 0.6,
    predicted: float = 0.5,
) -> dict[str, object]:
    absolute = abs(actual - predicted)
    return {
        "task": task,
        "mix": mix,
        "metric": "primary_metric",
        "setup": setup,
        "step_1_y": actual,
        "step_2_y": actual,
        "stacked_y": actual,
        "step_1_pred": predicted,
        "step_2_pred": predicted,
        "stacked_pred": predicted,
        "abs_error_step_1": absolute,
        "abs_error_step_2": absolute,
        "abs_error_stacked": absolute,
        "rel_error_stacked": absolute / abs(predicted),
    }


def test_setup_parser_covers_all_21_subsets_and_excludes_intermediate() -> None:
    sizes = (
        "4M",
        "6M",
        "8M",
        "10M",
        "14M",
        "16M",
        "20M",
        "60M",
        "90M",
        "150M",
        "300M",
        "530M",
        "750M",
    )
    setups = scaling._expected_setups(sizes)

    assert len(setups) == 8 * 21
    assert len({item.name for item in setups}) == len(setups)
    assert Counter(item.family for item in setups) == {
        family: 21 for family in scaling._FAMILIES
    }
    assert scaling._parse_setup("3_param", sizes) == scaling._Setup(
        "3_param", "3_param", sizes, "prefix-13", "prefix"
    )
    assert scaling._parse_setup("2_param-no_4M_no_6M", sizes).subset == "suffix-drop-02"  # type: ignore[union-attr]
    assert scaling._parse_setup("3_param-intermediate", sizes) is None


def test_setup_parser_rejects_noncanonical_subsets() -> None:
    sizes = tuple(f"{value}M" for value in range(13))
    with pytest.raises(ValueError, match="unsupported setup subset"):
        scaling._parse_setup("3_param-no_1M", sizes)


def test_pair_semantics_exclude_target_ties_and_default_predicted_ties() -> None:
    target = {"a": 3.0, "b": 2.0, "c": 2.0, "d": 1.0}
    predicted = {"a": 4.0, "b": 2.0, "c": 2.0, "d": 2.0}

    strict = scaling._pair(target, predicted)
    half = scaling._pair(target, predicted, 0.5)

    assert strict.denominator == 5
    assert strict.target_ties == 1
    assert strict.predicted_ties == 2
    assert strict.accuracy == pytest.approx(3 / 5)
    assert half.accuracy == pytest.approx(4 / 5)


def test_catalog_compute_uses_configured_tokens_and_exact_parameters() -> None:
    catalog = load_toml_model(_ROOT / "configs/catalog.toml", DataDecideCatalog)
    costs, target = scaling._catalog_compute(catalog)
    model = next(item for item in catalog.models if item.name == "4M")

    assert costs["4M"] == (
        6 * model.exact_parameter_count * 100 * model.training_parameter_count
    )
    assert target == 7.060992e20


def test_subset_compute_is_per_approach_without_variant_count_multiplier() -> None:
    catalog = load_toml_model(_ROOT / "configs/catalog.toml", DataDecideCatalog)
    costs, _ = scaling._catalog_compute(catalog)
    included = ("4M", "6M", "8M")

    assert sum(costs[size] for size in included) == pytest.approx(7.38305905385728e16)
    assert sum(costs[size] for size in included) != pytest.approx(
        8 * 7.38305905385728e16
    )


def test_error_summary_persists_both_relative_denominators() -> None:
    frame = pd.DataFrame([_row(task="task", mix="a", actual=0.6, predicted=0.5)])

    summary, missing, selected = scaling._error(frame, "3_param", ("task",), ("a",))

    assert missing == ()
    assert len(selected) == 1
    assert summary is not None
    assert summary.absolute_percent == pytest.approx(10.0)
    assert summary.released_relative_percent == pytest.approx(20.0)
    assert summary.paper_formula_relative_percent == pytest.approx(100 / 6)


@pytest.mark.parametrize(
    ("released_display_match", "paper_formula_match", "expected"),
    (
        (True, False, ValidationOutcome.DIRECTIONALLY_CONSISTENT),
        (True, True, ValidationOutcome.REPRODUCED),
        (False, False, ValidationOutcome.NOT_REPRODUCED),
        (False, True, ValidationOutcome.NOT_REPRODUCED),
    ),
)
def test_error_adjudication_requires_released_and_paper_formula_matches(
    released_display_match: bool,
    paper_formula_match: bool,
    expected: ValidationOutcome,
) -> None:
    assert (
        scaling._adjudicate_error(
            released_display_match=released_display_match,
            paper_formula_match=paper_formula_match,
        )
        is expected
    )


def test_error_summary_rejects_a_released_denominator_change() -> None:
    row = _row(task="task", mix="a")
    row["rel_error_stacked"] = 1 / 6
    with pytest.raises(ValueError, match="denominator drift"):
        scaling._error(pd.DataFrame([row]), "3_param", ("task",), ("a",))


def test_common_target_requires_every_compatible_setup_to_match() -> None:
    sizes = (
        "4M",
        "6M",
        "8M",
        "10M",
        "14M",
        "16M",
        "20M",
        "60M",
        "90M",
        "150M",
        "300M",
        "530M",
        "750M",
    )
    setups = scaling._expected_setups(sizes)
    rows = []
    for setup in (item for item in setups if item.family in scaling._TARGET_FAMILIES):
        rows.extend(
            (
                _row(mix="a", setup=setup.name),
                _row(mix="b", setup=setup.name, actual=0.4),
            )
        )
    target, missing, _ = scaling._target(pd.DataFrame(rows), setups, ("a", "b"))
    assert target == {"a": 0.6, "b": 0.4}
    assert missing == ()

    rows[-1]["stacked_y"] = 0.3
    _, missing, _ = scaling._target(pd.DataFrame(rows), setups, ("a", "b"))
    assert missing == (f"target_ranking_mismatch:setup={setups[104].name}",)


def test_missing_decision_groups_are_exact_and_deterministic() -> None:
    sizes = (
        "4M",
        "6M",
        "8M",
        "10M",
        "14M",
        "16M",
        "20M",
        "60M",
        "90M",
        "150M",
        "300M",
        "530M",
        "750M",
    )
    setups = scaling._expected_setups(sizes)
    points, missing = scaling._points(
        pd.DataFrame([_row(mix="a", setup=setups[0].name)]),
        setups,
        ("a", "b"),
        {"a": 1.0, "b": 0.0},
        {size: 1.0 for size in sizes},
        100.0,
        0.0,
    )
    assert points == ()
    assert missing == tuple(sorted(missing))
    assert f"decision:setup={setups[0].name}" in missing
    assert len(missing) == 168


def test_selected_key_hash_is_order_independent() -> None:
    frame = pd.DataFrame([_row(mix="b"), _row(mix="a")], columns=scaling._CHEAP_COLUMNS)
    assert scaling._key_sha(frame) == scaling._key_sha(frame.iloc[::-1])
    assert (
        scaling._key_sha(frame)
        == hashlib.sha256(
            b'[["olmes_10_macro_avg","a","primary_metric","3_param"],["olmes_10_macro_avg","b","primary_metric","3_param"]]'
        ).hexdigest()
    )


def test_frontier_comparison_uses_only_single_scale_points_at_or_below_compute() -> (
    None
):
    setup = scaling._Setup(
        "3_param", "3_param", ("4M", "6M", "8M"), "prefix-03", "prefix"
    )
    points = (scaling._Point(setup, 10.0, 1.0, scaling._Pair(0.7, 300, 0, 0)),)
    evidence = scaling._Frontier(
        ((5.0, 0.6), (10.0, 0.65), (20.0, 0.9)), (), (), 0.9, ()
    )
    compared = scaling._compare_frontier(points, evidence)
    assert compared[0].frontier_accuracy == 0.65
    assert compared[0].frontier_difference == pytest.approx(0.05)


def test_no_scaling_attempts_do_not_read_inputs() -> None:
    contract = load_paper_validation_contract()
    contract = contract.model_copy(
        update={
            "attempts": tuple(
                attempt
                for attempt in contract.attempts
                if attempt.analysis_id.value != "scaling_law"
            )
        }
    )
    assert scaling.run_scaling_law_attempts(
        repository_root=Path("missing"),
        data_root=Path("missing"),
        registry=load_repository_claim_registry(_ROOT),
        contract=contract,
        input_identities={},
    ) == ((), ())


def test_real_scaling_aggregate_smoke() -> None:
    cheap = (
        _REAL_DATA
        / "processed/published-results/cheap_decisions_stacked_rc_pred_all.parquet"
    )
    olmes = _REAL_DATA / "processed/olmes.parquet"
    if not cheap.is_file() or not olmes.is_file():
        pytest.skip("local dd_parsed mirror is unavailable")

    results, series = scaling.run_scaling_law_attempts(
        repository_root=_ROOT,
        data_root=_REAL_DATA,
        registry=load_repository_claim_registry(_ROOT),
        contract=load_paper_validation_contract(),
        input_identities={},
    )

    assert len(results) == 24
    assert sum(result.role is AttemptRole.DEFAULT for result in results) == 20
    assert Counter((result.role, result.outcome) for result in results) == Counter(
        {
            (AttemptRole.DEFAULT, ValidationOutcome.REPRODUCED): 6,
            (AttemptRole.DEFAULT, ValidationOutcome.DIRECTIONALLY_CONSISTENT): 8,
            (AttemptRole.DEFAULT, ValidationOutcome.NOT_REPRODUCED): 6,
            (AttemptRole.SENSITIVITY, ValidationOutcome.REPRODUCED): 1,
            (AttemptRole.SENSITIVITY, ValidationOutcome.NOT_REPRODUCED): 3,
        }
    )
    assert all(
        result.evidence_level is EvidenceLevel.AUTHOR_DERIVED_AGGREGATE
        for result in results
    )
    assert tuple(result.attempt_id for result in results) == tuple(
        sorted(result.attempt_id for result in results)
    )
    assert [(value.id, len(value.points)) for value in series] == [
        ("dd-0180-paper-analog", 168),
        ("dd-0368-paper-analog", 21),
        ("dd-0369-paper-analog", 21),
    ]
    by_id = {result.attempt_id: result for result in results}
    displayed = (
        (5.6, 2.6),
        (6.0, 2.8),
        (5.9, 2.9),
        (6.5, 3.1),
        (6.5, 3.2),
        (42.8, 17.4),
        (42.9, 42.3),
        (230.8, 65.4),
    )
    paper_formula_displayed = (4.9, 5.2, 5.2, 5.6, 5.9, 28.9, 85.1, 64.4)
    for claim, expected, expected_paper_formula in zip(
        range(301, 309), displayed, paper_formula_displayed, strict=True
    ):
        result = by_id[f"dd-{claim:04d}-default"]
        assert result.outcome is ValidationOutcome.DIRECTIONALLY_CONSISTENT
        assert result.computed_value["relative_error_denominator_discrepancy"] is True
        assert result.computed_value["released_relative_denominator"] == "prediction"
        assert result.computed_value["paper_formula_relative_denominator"] == "target"
        assert result.computed_value["released_display_match"] is True
        assert result.computed_value["paper_formula_match"] is False
        assert (
            result.computed_value["displayed_released_relative_error_percent"],
            result.computed_value["displayed_absolute_error_percent"],
        ) == expected
        assert (
            result.computed_value["displayed_paper_formula_relative_error_percent"]
            == expected_paper_formula
        )
        assert "released_display_match=true" in result.diagnostics
        assert "paper_formula_match=false" in result.diagnostics
        assert any(
            "not reproduced under the paper-stated target-denominator formula"
            in limitation
            for limitation in result.limitations
        )
    assert by_id["dd-0119-default"].computed_value[
        "best_vs_baseline_advantage"
    ] == pytest.approx(0.0233333333333333)
    assert by_id["dd-0013-default"].outcome is ValidationOutcome.NOT_REPRODUCED
    assert by_id["dd-0013-default"].computed_value[
        "maximum_frontier_difference"
    ] == pytest.approx(0.05222222222222217)
    assert (
        by_id["dd-0189-default"].computed_value["maximum_accuracy_rank_by_setup"][
            "2_param"
        ]
        == 1
    )
    assert (
        by_id["dd-0189-default"].computed_value["maximum_accuracy_rank_by_setup"][
            "3_param"
        ]
        == 5
    )
    assert by_id["dd-0312-default"].computed_value[
        "error_vs_accuracy_spearman"
    ] == pytest.approx(-0.6190476190476191)
    assert by_id["dd-0369-default"].outcome is ValidationOutcome.NOT_REPRODUCED
    assert (
        by_id["dd-0369-comparison-predicted-tie-credit-grid-2"].outcome
        is ValidationOutcome.REPRODUCED
    )

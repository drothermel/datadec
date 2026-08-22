from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from datadec.paper.contracts import load_claim_registry, load_validation_contract
from datadec.paper.models import (
    AttemptRole,
    ContentIdentity,
    EvidenceLevel,
    ValidationOutcome,
)
from datadec.paper.verifiers.math_code import run_math_code_attempts

_REPOSITORY_ROOT = Path(__file__).parents[2]
_REAL_DATA_ROOT = Path("/Users/daniellerothermel/drotherm/repos/datadec/data")
_DECISION_ID = "new_eval_decision_accuracy"
_MEANS_ID = "new_eval_means"
_COMPUTE_ID = "olmes_aggregate"
_SIZES = ("4M", "60M", "150M")
_MEANS_SIZES = (*_SIZES, "1B")
_TASKS = (
    "arc_challenge",
    "codex_humaneval",
    "gsm8k",
    "hellaswag",
    "mbpp",
    "minerva",
    "mmlu",
    "olmes_core9",
)
_MATH_CODE_CLAIMS = {
    "DD-0017",
    "DD-0018",
    "DD-0213",
    "DD-0221",
    "DD-0222",
    "DD-0224",
    "DD-0225",
    "DD-0226",
    "DD-0227",
    "DD-0413",
    "DD-0414",
}


def _contracts():
    return (
        load_claim_registry(_REPOSITORY_ROOT / "docs/paper/claims.toml"),
        load_validation_contract(_REPOSITORY_ROOT / "configs/paper_validation.toml"),
    )


def _decision_frame() -> pd.DataFrame:
    rows = [
        {
            "size": size,
            "task": task,
            "target_ranking": target,
            "logits_per_byte_corr": 0.5,
            "logits_per_char_corr": 0.5,
            "primary_score": 0.5,
        }
        for size in _SIZES
        for task in _TASKS
        for target in ("primary_score", "logits_per_byte_corr")
    ]
    frame = pd.DataFrame(rows)
    primary_values = {
        ("4M", "minerva"): (0.356667, 0.526667),
        ("60M", "minerva"): (0.563333, 0.533333),
        ("150M", "minerva"): (0.483333, 0.536667),
        ("4M", "gsm8k"): (0.456667, 0.543333),
        ("60M", "gsm8k"): (0.630000, 0.553333),
        ("150M", "gsm8k"): (0.460000, 0.533333),
        ("4M", "mbpp"): (0.406667, 0.840000),
        ("60M", "mbpp"): (0.556667, 0.856667),
        ("150M", "mbpp"): (0.683333, 0.843333),
        ("4M", "codex_humaneval"): (0.383333, 0.813333),
        ("60M", "codex_humaneval"): (0.516667, 0.750000),
        ("150M", "codex_humaneval"): (0.743333, 0.840000),
    }
    continuous_values = {
        ("4M", "minerva"): 0.810000,
        ("60M", "minerva"): 0.876667,
        ("150M", "minerva"): 0.900000,
        ("4M", "gsm8k"): 0.683333,
        ("60M", "gsm8k"): 0.773333,
        ("150M", "gsm8k"): 0.766667,
        ("4M", "mbpp"): 0.950000,
        ("60M", "mbpp"): 0.960000,
        ("150M", "mbpp"): 0.953333,
        ("4M", "codex_humaneval"): 0.910000,
        ("60M", "codex_humaneval"): 0.880000,
        ("150M", "codex_humaneval"): 0.956667,
    }
    for (size, task), (accuracy, proxy) in primary_values.items():
        mask = (
            frame["size"].eq(size)
            & frame["task"].eq(task)
            & frame["target_ranking"].eq("primary_score")
        )
        frame.loc[mask, "primary_score"] = accuracy
        frame.loc[mask, "logits_per_byte_corr"] = proxy
    for (size, task), value in continuous_values.items():
        mask = (
            frame["size"].eq(size)
            & frame["task"].eq(task)
            & frame["target_ranking"].eq("logits_per_byte_corr")
        )
        frame.loc[mask, "logits_per_byte_corr"] = value
    return frame


def _means_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "size": size,
                "task": task,
                "primary_score": 0.1,
                "logits_per_byte_corr": 1.0,
                "logits_per_char_corr": 0.7,
            }
            for size in _MEANS_SIZES
            for task in _TASKS
        ]
    )


def _compute_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"data": recipe, "params": size, "step": step, "compute": compute}
            for recipe in ("recipe-a", "recipe-b")
            for size, step, compute in (
                ("4M", 10, 0.5),
                ("4M", 20, 1.0),
                ("1B", 100, 50_000.0),
                ("1B", 200, 100_000.0),
            )
        ]
    )


def _write_inputs(
    data_root: Path,
    *,
    decision: pd.DataFrame | None = None,
    means: pd.DataFrame | None = None,
    compute: pd.DataFrame | None = None,
) -> None:
    _, contract = _contracts()
    frames = {
        _DECISION_ID: _decision_frame() if decision is None else decision,
        _MEANS_ID: _means_frame() if means is None else means,
        _COMPUTE_ID: _compute_frame() if compute is None else compute,
    }
    for table_id, frame in frames.items():
        spec = next(item for item in contract.inputs if item.id == table_id)
        path = data_root / spec.path
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)


def _identities(data_root: Path) -> dict[str, ContentIdentity]:
    _, contract = _contracts()
    identities = {}
    for table_id in (_DECISION_ID, _MEANS_ID, _COMPUTE_ID):
        spec = next(item for item in contract.inputs if item.id == table_id)
        with (data_root / spec.path).open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        identities[table_id] = ContentIdentity(id=table_id, sha256=digest)
    return identities


def _run(data_root: Path):
    registry, contract = _contracts()
    return run_math_code_attempts(
        repository_root=_REPOSITORY_ROOT,
        data_root=data_root,
        registry=registry,
        contract=contract,
        input_identities=_identities(data_root),
    )


def _by_id(results):
    return {result.attempt_id: result for result in results}


def _dimension(point, name: str):
    return next(item.value for item in point.dimensions if item.name == name)


def _measure(point, name: str) -> float:
    return next(item.value for item in point.measures if item.name == name)


def test_math_code_default_predicates_and_sensitivities(tmp_path: Path) -> None:
    _write_inputs(tmp_path)

    results, series = _run(tmp_path)
    by_id = _by_id(results)

    assert len(results) == 23
    assert {result.claim_id for result in results} == _MATH_CODE_CLAIMS
    assert {result.evidence_level for result in results} == {
        EvidenceLevel.AUTHOR_DERIVED_AGGREGATE
    }
    assert {
        attempt_id: by_id[attempt_id].outcome
        for attempt_id in (
            "dd-0017-default",
            "dd-0018-default",
            "dd-0213-default",
            "dd-0221-default",
            "dd-0222-default",
            "dd-0224-default",
            "dd-0225-default",
            "dd-0226-default",
            "dd-0227-default",
            "dd-0413-default",
            "dd-0414-default",
        )
    } == {
        "dd-0017-default": ValidationOutcome.REPRODUCED,
        "dd-0018-default": ValidationOutcome.REPRODUCED,
        "dd-0213-default": ValidationOutcome.REPRODUCED,
        "dd-0221-default": ValidationOutcome.REPRODUCED,
        "dd-0222-default": ValidationOutcome.REPRODUCED,
        "dd-0224-default": ValidationOutcome.APPROXIMATELY_REPRODUCED,
        "dd-0225-default": ValidationOutcome.DIRECTIONALLY_CONSISTENT,
        "dd-0226-default": ValidationOutcome.REPRODUCED,
        "dd-0227-default": ValidationOutcome.NOT_REPRODUCED,
        "dd-0413-default": ValidationOutcome.REPRODUCED,
        "dd-0414-default": ValidationOutcome.NOT_REPRODUCED,
    }
    expected_sensitivities = {
        "dd-0017-comparison-accuracy-threshold-grid-1": ValidationOutcome.REPRODUCED,
        "dd-0017-comparison-accuracy-threshold-grid-3": ValidationOutcome.NOT_REPRODUCED,
        "dd-0018-comparison-accuracy-threshold-grid-1": ValidationOutcome.REPRODUCED,
        "dd-0018-comparison-accuracy-threshold-grid-3": ValidationOutcome.NOT_REPRODUCED,
        "dd-0221-size-pointwise": ValidationOutcome.NOT_REPRODUCED,
        "dd-0222-size-pointwise": ValidationOutcome.NOT_REPRODUCED,
        "dd-0224-comparison-absolute-tolerance-grid-1": ValidationOutcome.NOT_REPRODUCED,
        "dd-0224-comparison-absolute-tolerance-grid-3": ValidationOutcome.APPROXIMATELY_REPRODUCED,
        "dd-0226-size-pointwise": ValidationOutcome.REPRODUCED,
        "dd-0227-comparison-accuracy-threshold-grid-1": ValidationOutcome.REPRODUCED,
        "dd-0227-comparison-accuracy-threshold-grid-3": ValidationOutcome.NOT_REPRODUCED,
        "dd-0414-comparison-trivial-tolerance-grid-2": ValidationOutcome.REPRODUCED,
    }
    assert {
        sensitivity_id: by_id[sensitivity_id].outcome
        for sensitivity_id in expected_sensitivities
    } == expected_sensitivities
    assert all(
        by_id[sensitivity_id].role is AttemptRole.SENSITIVITY
        and by_id[sensitivity_id].parent_attempt_id
        == sensitivity_id.split("-comparison")[0] + "-default"
        if "-comparison" in sensitivity_id
        else by_id[sensitivity_id].parent_attempt_id
        == sensitivity_id.removesuffix("-size-pointwise") + "-default"
        for sensitivity_id in expected_sensitivities
    )

    assert by_id["dd-0017-default"].computed_value[
        "decision_accuracy"
    ] == pytest.approx(0.84)
    assert by_id["dd-0017-default"].computed_value[
        "percent_target_compute"
    ] == pytest.approx(0.001)
    assert by_id["dd-0213-default"].computed_value[
        "minimum_proxy_gain"
    ] == pytest.approx(0.233333)
    assert by_id["dd-0224-default"].computed_value["task_components"][0][
        "decision_accuracy_mean"
    ] == pytest.approx((0.84 + 0.856667) / 2)
    assert by_id["dd-0226-default"].computed_value["comparisons"][0][
        "difference"
    ] == pytest.approx(0.2616665)
    assert by_id["dd-0227-default"].computed_value["task_components"][1][
        "maximum_decision_accuracy"
    ] == pytest.approx(0.773333)
    assert (
        by_id["dd-0414-default"].computed_value["all_four_bars_near_baseline"] is False
    )
    assert len(series) == 7


def test_math_code_row_selections_match_declared_calculation_scope(
    tmp_path: Path,
) -> None:
    _write_inputs(tmp_path)
    results, _ = _run(tmp_path)
    by_id = _by_id(results)
    expected_decision_selections = {
        "dd-0017-default": (
            1,
            ("4M",),
            ("mbpp",),
            "primary_score",
            "e26fb55e177f8536ee07d8a3171d9669b3f50b91bafa08621babe0d5d8cadd71",
        ),
        "dd-0018-default": (
            1,
            ("4M",),
            ("codex_humaneval",),
            "primary_score",
            "6d55bebee2888498e084084f56690109084c269c1e725092cd3bf1f2202af773",
        ),
        "dd-0213-default": (
            4,
            ("4M", "60M"),
            ("mbpp", "codex_humaneval"),
            "primary_score",
            "1d623a5a6839f139dfa20ff0bcf5cf645ba29873b312b5096430022052c69e87",
        ),
        "dd-0221-default": (
            4,
            ("4M", "60M"),
            ("mbpp", "codex_humaneval"),
            "primary_score",
            "1d623a5a6839f139dfa20ff0bcf5cf645ba29873b312b5096430022052c69e87",
        ),
        "dd-0222-default": (
            4,
            ("4M", "60M"),
            ("minerva", "gsm8k"),
            "primary_score",
            "c3e5af7a03190016a3d4a28ae88f91f9ab9e4e0db847e635f0a14767326a2786",
        ),
        "dd-0224-default": (
            4,
            ("4M", "60M"),
            ("mbpp", "codex_humaneval"),
            "primary_score",
            "1d623a5a6839f139dfa20ff0bcf5cf645ba29873b312b5096430022052c69e87",
        ),
        "dd-0225-default": (
            4,
            ("4M", "60M"),
            ("mbpp", "codex_humaneval"),
            "primary_score",
            "1d623a5a6839f139dfa20ff0bcf5cf645ba29873b312b5096430022052c69e87",
        ),
        "dd-0226-default": (
            8,
            ("4M", "60M"),
            ("mbpp", "codex_humaneval", "minerva", "gsm8k"),
            "primary_score",
            "948a0621fe0e8a1df87e235390688dcdf38299c9a28fda256c10f091b33f980e",
        ),
        "dd-0227-default": (
            4,
            ("4M", "60M"),
            ("minerva", "gsm8k"),
            "logits_per_byte_corr",
            "ad54018c98320cba3a15b8e30c70d8a2baacd44e56525b4e6dd92f40eceb5a8b",
        ),
        "dd-0413-default": (
            4,
            ("4M", "60M"),
            ("mbpp", "codex_humaneval"),
            "primary_score",
            "1d623a5a6839f139dfa20ff0bcf5cf645ba29873b312b5096430022052c69e87",
        ),
        "dd-0414-default": (
            4,
            ("4M", "60M"),
            ("minerva", "gsm8k"),
            "primary_score",
            "c3e5af7a03190016a3d4a28ae88f91f9ab9e4e0db847e635f0a14767326a2786",
        ),
    }

    for attempt_id, (
        count,
        sizes,
        tasks,
        target,
        key_hash,
    ) in expected_decision_selections.items():
        selection = next(
            item
            for item in by_id[attempt_id].row_selections
            if item.logical_table_id == _DECISION_ID
        )
        predicates = {item.column: item for item in selection.predicates}
        assert selection.selected_row_count == count
        assert predicates["size"].value == sizes
        assert predicates["task"].value == tasks
        assert predicates["target_ranking"].value == target
        assert selection.selected_key_sha256 == key_hash
        assert "150M" not in json.dumps(by_id[attempt_id].computed_value)

    for attempt_id in ("dd-0213-default", "dd-0225-default"):
        selection = next(
            item
            for item in by_id[attempt_id].row_selections
            if item.logical_table_id == _MEANS_ID
        )
        predicates = {item.column: item for item in selection.predicates}
        assert selection.selected_row_count == 4
        assert predicates["size"].value == ("4M", "60M")
        assert predicates["task"].value == ("mbpp", "codex_humaneval")
        assert (
            selection.selected_key_sha256
            == "76493306f2749d6db194f168133b67411a2b5807877884a2072d53a2157ff881"
        )


def test_math_code_plot_semantics_order_alias_and_limitation(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    results, series = _run(tmp_path)
    by_id = _by_id(results)

    expected_series_ids = {
        "dd-0221-paper-analog",
        "dd-0222-paper-analog",
        "dd-0224-paper-analog",
        "dd-0225-paper-analog",
        "dd-0226-paper-analog",
        "dd-0413-paper-analog",
        "dd-0414-paper-analog",
    }
    assert {item.id for item in series} == expected_series_ids
    expected_scopes = {
        "dd-0221-paper-analog": (
            ("mbpp", "codex_humaneval"),
            ("primary_score", "logits_per_byte_corr"),
            8,
        ),
        "dd-0222-paper-analog": (
            ("minerva", "gsm8k"),
            ("primary_score", "logits_per_byte_corr"),
            8,
        ),
        "dd-0224-paper-analog": (
            ("mbpp", "codex_humaneval"),
            ("primary_score", "logits_per_byte_corr"),
            8,
        ),
        "dd-0225-paper-analog": (
            ("mbpp", "codex_humaneval"),
            ("primary_score", "logits_per_byte_corr"),
            8,
        ),
        "dd-0226-paper-analog": (
            ("minerva", "gsm8k", "mbpp", "codex_humaneval"),
            ("primary_score", "logits_per_byte_corr"),
            16,
        ),
        "dd-0413-paper-analog": (
            ("mbpp", "codex_humaneval"),
            ("logits_per_byte_corr",),
            4,
        ),
        "dd-0414-paper-analog": (
            ("minerva", "gsm8k"),
            ("logits_per_byte_corr",),
            4,
        ),
    }
    task_indices = {"minerva": 0.0, "gsm8k": 1.0, "mbpp": 2.0, "codex_humaneval": 3.0}
    for item in series:
        tasks, metrics, point_count = expected_scopes[item.id]
        assert item.dimensions == (
            "task",
            "size",
            "predictor_metric",
            "target_metric",
        )
        assert item.measures == ("task_index", "decision_accuracy")
        assert item.x_axis.measure == "task_index"
        assert item.y_axis.measure == "decision_accuracy"
        assert len(item.points) == point_count
        assert {_dimension(point, "task") for point in item.points} == set(tasks)
        assert {_dimension(point, "size") for point in item.points} == {"4M", "60M"}
        assert {_dimension(point, "predictor_metric") for point in item.points} == set(
            metrics
        )
        assert {_dimension(point, "target_metric") for point in item.points} == {
            "primary_score"
        }
        assert all(
            _measure(point, "task_index") == task_indices[_dimension(point, "task")]
            for point in item.points
        )
        assert {count.name: count.value for count in item.counts} == {
            "source_rows": len(tasks) * 2,
            "points": point_count,
            "tasks": len(tasks),
            "sizes": 2,
            "predictor_metrics": len(metrics),
        }
        assert "Correct Prob unresolved" in item.semantic_kind
    assert [
        _measure(point, "decision_accuracy") for point in series[0].points[:4]
    ] == pytest.approx([0.406667, 0.84, 0.556667, 0.856667])
    full_task_surfaces = [
        item.id
        for item in series
        if {_dimension(point, "task") for point in item.points} == set(task_indices)
    ]
    assert full_task_surfaces == ["dd-0226-paper-analog"]
    assert "logits_per_byte_corr" in json.dumps(by_id["dd-0225-default"].computed_value)
    assert any(
        "cannot establish recipe separation or a noise floor" in limitation
        for limitation in by_id["dd-0225-default"].limitations
    )
    assert all(
        any(
            "does not independently recompute evaluations" in value
            for value in result.limitations
        )
        for result in results
    )


@pytest.mark.parametrize("malformation", ["missing", "duplicate", "nonfinite"])
def test_math_code_malformed_decision_cells_are_not_assessable(
    tmp_path: Path, malformation: str
) -> None:
    decision = _decision_frame()
    if malformation == "missing":
        decision = decision.iloc[1:].copy()
    elif malformation == "duplicate":
        decision = pd.concat([decision, decision.iloc[[0]]], ignore_index=True)
    else:
        decision.loc[0, "logits_per_byte_corr"] = float("nan")
    _write_inputs(tmp_path, decision=decision)

    results, series = _run(tmp_path)

    assert not series
    assert all(
        result.outcome is ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
        for result in results
    )
    assert all(result.missing_groups for result in results)
    assert any(
        group.startswith(f"{malformation}:") for group in results[0].missing_groups
    )


def test_math_code_means_cube_only_blocks_declared_consumers(tmp_path: Path) -> None:
    means = pd.concat([_means_frame(), _means_frame().iloc[[0]]], ignore_index=True)
    _write_inputs(tmp_path, means=means)

    results, series = _run(tmp_path)
    by_id = _by_id(results)

    assert by_id["dd-0213-default"].outcome is (
        ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
    )
    assert by_id["dd-0225-default"].outcome is (
        ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
    )
    assert by_id["dd-0017-default"].outcome is ValidationOutcome.REPRODUCED
    assert "dd-0225-paper-analog" not in {item.id for item in series}


def test_math_code_schema_and_identity_are_fail_closed(tmp_path: Path) -> None:
    decision = _decision_frame().assign(extra=1.0)
    _write_inputs(tmp_path, decision=decision)
    registry, contract = _contracts()
    with pytest.raises(ValueError, match="schema must be exactly"):
        run_math_code_attempts(
            repository_root=_REPOSITORY_ROOT,
            data_root=tmp_path,
            registry=registry,
            contract=contract,
            input_identities=_identities(tmp_path),
        )

    _write_inputs(tmp_path)
    identities = _identities(tmp_path)
    identities[_DECISION_ID] = ContentIdentity(id=_DECISION_ID, sha256="0" * 64)
    with pytest.raises(ValueError, match="identity differs"):
        run_math_code_attempts(
            repository_root=_REPOSITORY_ROOT,
            data_root=tmp_path,
            registry=registry,
            contract=contract,
            input_identities=identities,
        )


def test_math_code_selection_hash_is_order_independent(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    first_results, _ = _run(tmp_path)
    first = _by_id(first_results)["dd-0221-default"].row_selections[0]

    _write_inputs(
        tmp_path,
        decision=_decision_frame()
        .sample(frac=1.0, random_state=7)
        .reset_index(drop=True),
    )
    second_results, _ = _run(tmp_path)
    second = _by_id(second_results)["dd-0221-default"].row_selections[0]

    assert first.selected_row_count == second.selected_row_count == 4
    assert (
        first.selected_key_sha256
        == "1d623a5a6839f139dfa20ff0bcf5cf645ba29873b312b5096430022052c69e87"
    )
    assert first.selected_key_sha256 == second.selected_key_sha256
    assert first.local_parquet_sha256 != "0" * 64
    assert len(first.local_parquet_sha256) == 64
    assert len(first.selected_key_sha256) == 64


@pytest.mark.skipif(
    not (_REAL_DATA_ROOT / "processed/olmes.parquet").is_file(),
    reason="local dd_parsed mirror is unavailable",
)
def test_math_code_real_data_smoke_has_all_defaults() -> None:
    results, series = _run(_REAL_DATA_ROOT)

    defaults = tuple(result for result in results if result.role is AttemptRole.DEFAULT)
    assert len(defaults) == 11
    assert {result.claim_id for result in defaults} == _MATH_CODE_CLAIMS
    assert {result.claim_id: result.outcome for result in defaults} == {
        "DD-0017": ValidationOutcome.REPRODUCED,
        "DD-0018": ValidationOutcome.REPRODUCED,
        "DD-0213": ValidationOutcome.REPRODUCED,
        "DD-0221": ValidationOutcome.REPRODUCED,
        "DD-0222": ValidationOutcome.REPRODUCED,
        "DD-0224": ValidationOutcome.APPROXIMATELY_REPRODUCED,
        "DD-0225": ValidationOutcome.DIRECTIONALLY_CONSISTENT,
        "DD-0226": ValidationOutcome.REPRODUCED,
        "DD-0227": ValidationOutcome.NOT_REPRODUCED,
        "DD-0413": ValidationOutcome.REPRODUCED,
        "DD-0414": ValidationOutcome.NOT_REPRODUCED,
    }
    assert {item.id: len(item.points) for item in series} == {
        "dd-0221-paper-analog": 8,
        "dd-0222-paper-analog": 8,
        "dd-0224-paper-analog": 8,
        "dd-0225-paper-analog": 8,
        "dd-0226-paper-analog": 16,
        "dd-0413-paper-analog": 4,
        "dd-0414-paper-analog": 4,
    }

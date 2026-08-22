from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from datadec.paper.single_scale import (
    DEFAULT_TASK_GROUPING,
    MMLU_SUBJECTS,
    OLMES_NON_MMLU_TASKS,
    OlmesTaskGrouping,
    SingleScaleUniverse,
    aggregate_checkpoint,
    analyze_single_scale,
    build_prediction_rankings,
    build_target_ranking,
    compare_rankings,
    observations_from_olmes_frame,
    select_common_complete_checkpoints,
    select_exact_common_complete_checkpoint,
)

FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "olmes_single_scale_regression.parquet"
)

# Derived from data/processed/olmes.parquet SHA-256
# 2141844814f8d0e9aa6d9db77c94575e932bd1a7a878fe97579865110e9d06a7 by
# selecting only the declared target/prediction seeds and the latest three
# common-complete checkpoints for each size, with the seven columns read below.
EXPECTED_RECIPES = (
    "C4",
    "DCLM-Baseline",
    "DCLM-Baseline (QC 10%)",
    "DCLM-Baseline (QC 20%)",
    "DCLM-Baseline (QC 7%, FW2)",
    "DCLM-Baseline (QC 7%, FW3)",
    "DCLM-Baseline (QC FW 10%)",
    "DCLM-Baseline (QC FW 3%)",
    "DCLM-Baseline 25% / Dolma 75%",
    "DCLM-Baseline 50% / Dolma 50%",
    "DCLM-Baseline 75% / Dolma 25%",
    "Dolma1.6++",
    "Dolma1.7",
    "Dolma1.7 (no Flan)",
    "Dolma1.7 (no Reddit)",
    "Dolma1.7 (no code)",
    "Dolma1.7 (no math, code)",
    "Falcon",
    "Falcon+CC",
    "Falcon+CC (QC 10%)",
    "Falcon+CC (QC 20%)",
    "Falcon+CC (QC Orig 10%)",
    "Falcon+CC (QC Tulu 10%)",
    "FineWeb-Edu",
    "FineWeb-Pro",
)
SMALL_GROUPING = OlmesTaskGrouping(
    non_mmlu_tasks=("arc",),
    mmlu_subjects=("mmlu-a", "mmlu-b"),
)


def _rows(
    *,
    recipes: tuple[str, ...] = ("a", "b"),
    seeds: tuple[str, ...] = ("seed-1", "seed-2"),
    step: int = 10,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for recipe_index, recipe in enumerate(recipes):
        for seed_index, seed in enumerate(seeds):
            for task in SMALL_GROUPING.source_tasks:
                if task == "arc":
                    score = float(recipe_index)
                else:
                    score = float(recipe_index + 2)
                result.append(
                    {
                        "params": "small",
                        "data": recipe,
                        "seed": seed,
                        "step": step,
                        "task": task,
                        "compute": float(step * 100),
                        "primary_metric": score + seed_index / 10,
                    }
                )
    return result


def _small_universe(
    *,
    recipes: tuple[str, ...] = ("a", "b"),
    seeds: tuple[str, ...] = ("seed-1", "seed-2"),
) -> SingleScaleUniverse:
    return SingleScaleUniverse(
        model_size="small",
        recipes=recipes,
        seeds=seeds,
        source_tasks=SMALL_GROUPING.source_tasks,
        metrics=("primary_metric",),
    )


def test_common_complete_selection_never_mixes_per_row_latest_steps() -> None:
    rows = _rows(step=10)
    rows.extend(row for row in _rows(step=20) if row["data"] == "a")
    observations = observations_from_olmes_frame(
        pd.DataFrame(rows), metric_columns=("primary_metric",)
    )

    selected = select_common_complete_checkpoints(observations, _small_universe())

    assert selected.default.step == 10
    assert selected.complete_steps == (10,)
    assert {row.step for row in selected.default.observations} == {10}
    assert selected.default.raw_row_count == 12


def test_incomplete_checkpoint_grid_fails_explicitly() -> None:
    incomplete = observations_from_olmes_frame(
        pd.DataFrame(_rows()[:-1]), metric_columns=("primary_metric",)
    )

    with pytest.raises(ValueError, match=r"checkpoint grid is incomplete.*11.*12"):
        select_exact_common_complete_checkpoint(incomplete, _small_universe(), step=10)
    with pytest.raises(ValueError, match="no common complete checkpoint"):
        select_common_complete_checkpoints(incomplete, _small_universe())


def test_two_stage_macro_aggregation_is_deterministic() -> None:
    frame = pd.DataFrame(_rows(recipes=("a",), seeds=("seed-1",)))
    observations = observations_from_olmes_frame(
        frame.sample(frac=1, random_state=11),
        metric_columns=("primary_metric",),
    )
    checkpoint = select_exact_common_complete_checkpoint(
        observations,
        _small_universe(recipes=("a",), seeds=("seed-1",)),
        step=10,
    )

    scores = aggregate_checkpoint(checkpoint, task_grouping=SMALL_GROUPING)

    assert len(scores) == 1
    assert scores[0].mmlu_score == 2.0
    assert scores[0].score == 1.0
    assert scores[0].source_task_count == 3
    assert scores[0].logical_task_count == 2


def test_target_ties_are_excluded_and_prediction_ties_are_incorrect() -> None:
    recipes = ("a", "b", "c")
    target_rows = _rows(recipes=recipes, seeds=("seed-1",))
    prediction_rows = _rows(recipes=recipes, seeds=("seed-1",))
    target_values = {"a": 1.0, "b": 1.0, "c": 2.0}
    prediction_values = {"a": 0.0, "b": 1.0, "c": 1.0}
    for row in target_rows:
        row["primary_metric"] = target_values[str(row["data"])]
    for row in prediction_rows:
        row["primary_metric"] = prediction_values[str(row["data"])]
    target_checkpoint = select_exact_common_complete_checkpoint(
        observations_from_olmes_frame(
            pd.DataFrame(target_rows), metric_columns=("primary_metric",)
        ),
        _small_universe(recipes=recipes, seeds=("seed-1",)),
        step=10,
    )
    prediction_checkpoint = select_exact_common_complete_checkpoint(
        observations_from_olmes_frame(
            pd.DataFrame(prediction_rows), metric_columns=("primary_metric",)
        ),
        _small_universe(recipes=recipes, seeds=("seed-1",)),
        step=10,
    )
    target_scores = aggregate_checkpoint(
        target_checkpoint, task_grouping=SMALL_GROUPING
    )
    prediction_scores = aggregate_checkpoint(
        prediction_checkpoint, task_grouping=SMALL_GROUPING
    )
    target = build_target_ranking(
        target_checkpoint, target_scores, metric="primary_metric"
    )
    predicted = build_prediction_rankings(prediction_checkpoint, prediction_scores)[0]

    result = compare_rankings(
        target, predicted, actual_compute=1_000.0, target_compute=2_000.0
    )

    assert result.total_pairs == 3
    assert result.target_ties == 1
    assert result.predicted_ties == 1
    assert result.denominator == 2
    assert result.correct == 1
    assert result.accuracy == 0.5
    assert result.percent_target_compute == 50.0
    assert sum(pair.excluded for pair in result.pairs) == 1
    predicted_tie = next(pair for pair in result.pairs if pair.predicted_tie)
    assert not predicted_tie.excluded
    assert not predicted_tie.correct


def test_full_dd_parsed_single_scale_regression() -> None:
    frame = pd.read_parquet(FIXTURE_PATH)
    observations = observations_from_olmes_frame(
        frame, metric_columns=("primary_metric",)
    )
    source_tasks = (*OLMES_NON_MMLU_TASKS, *MMLU_SUBJECTS)
    target_universe = SingleScaleUniverse(
        model_size="1B",
        recipes=EXPECTED_RECIPES,
        seeds=("default", "large aux 2", "large aux 3"),
        source_tasks=source_tasks,
        metrics=("primary_metric",),
    )
    prediction_universe = SingleScaleUniverse(
        model_size="150M",
        recipes=EXPECTED_RECIPES,
        seeds=("default", "small aux 2", "small aux 3"),
        source_tasks=source_tasks,
        metrics=("primary_metric",),
    )

    result = analyze_single_scale(
        observations,
        target_universe=target_universe,
        prediction_universe=prediction_universe,
        target_metric="primary_metric",
        task_grouping=DEFAULT_TASK_GROUPING,
    )

    assert len(frame) == 29_700
    assert len(EXPECTED_RECIPES) == 25
    assert len(source_tasks) == 66
    assert result.target_checkpoints.default.step == 69_369
    assert tuple(
        checkpoint.step for checkpoint in result.target_checkpoints.preceding
    ) == (67_500, 65_000)
    assert result.prediction_checkpoints.default.step == 37_500
    assert tuple(
        checkpoint.step for checkpoint in result.prediction_checkpoints.preceding
    ) == (36_250, 35_000)
    assert all(
        prediction.checkpoint.raw_row_count == 4_950
        for prediction in result.predictions
    )

    default = result.predictions[0]
    summary = default.summaries[0]
    assert len(result.target_ranking.scores) == 25
    assert len(default.rankings) == 3
    assert sum(len(attempt.pairs) for attempt in default.seed_decisions) == 900
    assert len(default.noise_spread) == 1
    assert summary.correct_counts == (234, 251, 238)
    assert summary.denominator_per_seed == 300
    assert summary.total_pairs_per_seed == 300
    assert summary.mean_accuracy == 0.8033333333333333
    assert summary.sample_sd_accuracy == 0.029627314724385286
    assert summary.ddof == 1
    assert summary.sd_denominator == 2
    assert summary.target_ties == 0
    assert summary.predicted_ties == 0
    assert summary.actual_compute == 1.3439040749568e19
    assert summary.percent_target_compute == pytest.approx(1.9029812359021125)

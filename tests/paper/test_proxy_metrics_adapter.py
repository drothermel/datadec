from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from datadec.data.ingest.enums import DataRecipeName, Seed
from datadec.paper.contracts import (
    load_repository_claim_registry,
    load_validation_contract,
)
from datadec.paper.models import (
    AttemptRole,
    ClaimRegistry,
    ContentIdentity,
    PaperValidationContract,
    ValidationOutcome,
)
from datadec.paper.single_scale import DEFAULT_TASK_GROUPING
from datadec.paper.verifiers.proxy_metrics import (
    run_noise_spread_attempts,
    run_proxy_metrics_attempts,
)

_REPOSITORY_ROOT = Path(__file__).parents[2]
_RECIPES = tuple(recipe.value for recipe in DataRecipeName)
_TARGET_SEEDS = (Seed.DEFAULT.value, Seed.LARGE_AUX_2.value, Seed.LARGE_AUX_3.value)
_PREDICTION_SEEDS = (
    Seed.DEFAULT.value,
    Seed.SMALL_AUX_2.value,
    Seed.SMALL_AUX_3.value,
)
_PROXY_METRICS = (
    "acc_raw",
    "acc_per_token",
    "acc_per_char",
    "correct_prob",
    "correct_prob_per_token",
    "correct_prob_per_char",
    "margin",
    "margin_per_token",
    "margin_per_char",
    "norm_correct_prob",
    "norm_correct_prob_per_token",
    "norm_correct_prob_per_char",
    "total_prob",
    "total_prob_per_token",
    "total_prob_per_char",
)


def _contracts(claim_id: str) -> tuple[ClaimRegistry, PaperValidationContract]:
    registry = load_repository_claim_registry(_REPOSITORY_ROOT)
    contract = load_validation_contract(
        _REPOSITORY_ROOT / "configs/paper_validation.toml"
    )
    claim = next(item for item in registry.claims if item.id == claim_id)
    attempt = next(item for item in contract.attempts if item.claim_id == claim_id)
    return (
        ClaimRegistry(format_version=2, claims=(claim,)),
        contract.model_copy(update={"attempts": (attempt,)}),
    )


def _write_aggregate(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[Path, str]:
    path = tmp_path / "processed" / "olmes.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def _identity(digest: str) -> dict[str, ContentIdentity]:
    return {"olmes_aggregate": ContentIdentity(id="olmes_aggregate", sha256=digest)}


def _metric_row(
    *,
    model_size: str,
    recipe: str,
    seed: str,
    step: int,
    task: str,
    compute: float,
    value: float,
) -> dict[str, object]:
    return {
        "params": model_size,
        "data": recipe,
        "seed": seed,
        "step": step,
        "task": task,
        "compute": compute,
        "primary_metric": value,
        **{metric: value for metric in _PROXY_METRICS},
    }


def test_threshold_attempt_uses_latest_complete_point_within_fixed_budget(
    tmp_path: Path,
) -> None:
    registry, contract = _contracts("DD-0016")
    rows = [
        _metric_row(
            model_size="1B",
            recipe=recipe,
            seed=seed,
            step=100,
            task=task,
            compute=10_000.0,
            value=index / 100,
        )
        for task in DEFAULT_TASK_GROUPING.source_tasks
        for index, recipe in enumerate(_RECIPES)
        for seed in _TARGET_SEEDS
    ]
    rows.extend(
        _metric_row(
            model_size="10M",
            recipe=recipe,
            seed=seed,
            step=10,
            task=task,
            compute=1.0,
            value=index / 100,
        )
        for task in DEFAULT_TASK_GROUPING.source_tasks
        for index, recipe in enumerate(_RECIPES)
        for seed in _PREDICTION_SEEDS
    )
    _, digest = _write_aggregate(tmp_path, rows)

    attempts, series = run_proxy_metrics_attempts(
        repository_root=_REPOSITORY_ROOT,
        data_root=tmp_path,
        registry=registry,
        contract=contract,
        input_identities=_identity(digest),
    )

    assert series == ()
    assert len(attempts) == 1
    result = attempts[0]
    assert result.attempt_id == "dd-0016-default"
    assert result.outcome is ValidationOutcome.REPRODUCED
    assert result.computed_value == {
        "accuracy_threshold": 0.8,
        "compute_budget_percent": 0.01,
        "actual_percent_target_compute": 0.01,
        "model_size": "10M",
        "step": 10,
        "best_continuous_proxy_metric": "total_prob_per_token",
        "decision_accuracy": 1.0,
        "metric_decision_accuracies": {
            metric: 1.0 for metric in sorted(_PROXY_METRICS[3:])
        },
        "satisfied": True,
    }
    assert result.denominator == 900
    assert [item.actual_step for item in result.checkpoint_selections] == [100, 10]
    assert all(item.local_parquet_sha256 == digest for item in result.row_selections)


def test_proxy_adapter_hashes_actual_input_and_does_not_require_choice_files(
    tmp_path: Path,
) -> None:
    registry, contract = _contracts("DD-0016")
    rows = [
        _metric_row(
            model_size="1B",
            recipe=recipe,
            seed=seed,
            step=100,
            task=task,
            compute=10_000.0,
            value=index / 100,
        )
        for task in DEFAULT_TASK_GROUPING.source_tasks
        for index, recipe in enumerate(_RECIPES)
        for seed in _TARGET_SEEDS
    ]
    _, digest = _write_aggregate(tmp_path, rows)

    with pytest.raises(ValueError, match="differs from the actual Parquet"):
        run_proxy_metrics_attempts(
            repository_root=_REPOSITORY_ROOT,
            data_root=tmp_path,
            registry=registry,
            contract=contract,
            input_identities=_identity("0" * 64),
        )

    assert digest != "0" * 64
    assert not (tmp_path / "processed" / "olmes-details").exists()


def test_noise_plot_uses_declared_canonical_series_id(tmp_path: Path) -> None:
    registry, contract = _contracts("DD-0209")
    attempt = contract.attempts[0].model_copy(
        update={"plot_series_ids": ("dd-0209-paper-analog",)}
    )
    contract = contract.model_copy(update={"attempts": (attempt,)})
    offsets = {
        Seed.DEFAULT.value: -0.01,
        Seed.SMALL_AUX_2.value: 0.0,
        Seed.SMALL_AUX_3.value: 0.01,
    }
    rows = [
        {
            "params": "1B",
            "data": recipe,
            "seed": seed,
            "step": 100,
            "task": "hellaswag",
            "compute": 100.0,
            "primary_metric": index / 100,
            "correct_prob": index / 100,
        }
        for index, recipe in enumerate(_RECIPES)
        for seed in _TARGET_SEEDS
    ]
    rows.extend(
        {
            "params": "150M",
            "data": recipe,
            "seed": seed,
            "step": step,
            "task": "hellaswag",
            "compute": float(step),
            "primary_metric": index / 100 + offsets[seed],
            "correct_prob": index / 100 + offsets[seed],
        }
        for step in (30, 40, 50)
        for index, recipe in enumerate(_RECIPES)
        for seed in _PREDICTION_SEEDS
    )
    _, digest = _write_aggregate(tmp_path, rows)

    attempts, series = run_noise_spread_attempts(
        repository_root=_REPOSITORY_ROOT,
        data_root=tmp_path,
        registry=registry,
        contract=contract,
        input_identities=_identity(digest),
    )

    assert len(attempts) == 3
    assert len(series) == 1
    assert [result.attempt_id for result in attempts] == [
        "dd-0209-default",
        "dd-0209-preceding-common-complete-1",
        "dd-0209-preceding-common-complete-2",
    ]
    assert attempts[0].role is AttemptRole.DEFAULT
    assert all(result.role is AttemptRole.SENSITIVITY for result in attempts[1:])
    assert [result.checkpoint_selections[1].actual_step for result in attempts] == [
        50,
        40,
        30,
    ]
    assert attempts[0].plot_series_ids == ("dd-0209-paper-analog",)
    assert attempts[0].outcome is ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
    assert series[0].id == "dd-0209-paper-analog"
    assert series[0].actual_checkpoint == 50
    assert len(series[0].points) == 1


def test_qualitative_claims_execute_instead_of_being_silently_omitted(
    tmp_path: Path,
) -> None:
    proxy_ids = (
        "DD-0055",
        "DD-0057",
        "DD-0196",
        "DD-0197",
        "DD-0198",
        "DD-0199",
        "DD-0202",
        "DD-0203",
        "DD-0204",
        "DD-0205",
        "DD-0206",
        "DD-0207",
    )
    noise_ids = ("DD-0056", "DD-0098", "DD-0194", "DD-0211", "DD-0212")

    for claim_id, runner in (
        *((claim_id, run_proxy_metrics_attempts) for claim_id in proxy_ids),
        *((claim_id, run_noise_spread_attempts) for claim_id in noise_ids),
    ):
        registry, contract = _contracts(claim_id)

        with pytest.raises(FileNotFoundError, match="aggregate input does not exist"):
            runner(
                repository_root=_REPOSITORY_ROOT,
                data_root=tmp_path,
                registry=registry,
                contract=contract,
                input_identities={},
            )


def test_small_scale_proxy_equivalence_uses_frozen_compute_and_difference_rules(
    tmp_path: Path,
) -> None:
    registry, contract = _contracts("DD-0196")
    rows = [
        _metric_row(
            model_size="1B",
            recipe=recipe,
            seed=seed,
            step=100,
            task=task,
            compute=10_000.0,
            value=index / 100,
        )
        for task in DEFAULT_TASK_GROUPING.source_tasks
        for index, recipe in enumerate(_RECIPES)
        for seed in _TARGET_SEEDS
    ]
    rows.extend(
        _metric_row(
            model_size="10M",
            recipe=recipe,
            seed=seed,
            step=10,
            task=task,
            compute=1.0,
            value=index / 100,
        )
        for task in DEFAULT_TASK_GROUPING.source_tasks
        for index, recipe in enumerate(_RECIPES)
        for seed in _PREDICTION_SEEDS
    )
    _, digest = _write_aggregate(tmp_path, rows)

    attempts, series = run_proxy_metrics_attempts(
        repository_root=_REPOSITORY_ROOT,
        data_root=tmp_path,
        registry=registry,
        contract=contract,
        input_identities=_identity(digest),
    )

    assert series == ()
    assert len(attempts) == 3
    default = next(result for result in attempts if result.role is AttemptRole.DEFAULT)
    assert default.attempt_id == "dd-0196-default"
    assert default.outcome is ValidationOutcome.REPRODUCED
    assert default.computed_value["comparison_count"] == 10
    assert default.computed_value["mean_best_proxy_minus_accuracy"] == 0.0
    sensitivities = tuple(
        result for result in attempts if result.role is AttemptRole.SENSITIVITY
    )
    assert {result.attempt_id for result in sensitivities} == {
        "dd-0196-comparison-maximum-scale-percent-grid-1",
        "dd-0196-comparison-maximum-scale-percent-grid-3",
    }
    assert all(
        result.parent_attempt_id == default.attempt_id for result in sensitivities
    )


def test_one_billion_seed_sd_claim_counts_tasks_and_reports_maximum(
    tmp_path: Path,
) -> None:
    registry, contract = _contracts("DD-0098")
    tasks = DEFAULT_TASK_GROUPING.source_tasks
    seed_offsets = dict(zip(_TARGET_SEEDS, (-0.02, 0.0, 0.02), strict=True))
    rows = [
        _metric_row(
            model_size="1B",
            recipe=recipe,
            seed=seed,
            step=100,
            task=task,
            compute=10_000.0,
            value=index / 100 + (seed_offsets[seed] if index == 0 else 0.0),
        )
        for task in tasks
        for index, recipe in enumerate(_RECIPES)
        for seed in _TARGET_SEEDS
    ]
    _, digest = _write_aggregate(tmp_path, rows)

    attempts, series = run_noise_spread_attempts(
        repository_root=_REPOSITORY_ROOT,
        data_root=tmp_path,
        registry=registry,
        contract=contract,
        input_identities=_identity(digest),
    )

    assert series == ()
    assert attempts[0].outcome is ValidationOutcome.APPROXIMATELY_REPRODUCED
    assert attempts[0].computed_value["matching_task_count"] == 10
    assert attempts[0].computed_value["maximum_observed_sample_sd"] == pytest.approx(
        0.02
    )

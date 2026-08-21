from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from datadec.config import ScalingLawContract, load_paper_validation_contract
from datadec.paper.contracts import (
    load_repository_claim_registry,
    load_toml_model,
)
from datadec.paper.models import ContentIdentity, ValidationOutcome
from datadec.paper.scaling import ScalingVariant
from datadec.paper.single_scale import DEFAULT_TASK_GROUPING
from datadec.paper.verifiers import scaling as scaling_adapter

_REPOSITORY_ROOT = Path(__file__).parents[2]


def _scaling_contract() -> ScalingLawContract:
    return load_toml_model(
        _REPOSITORY_ROOT / "configs/scaling_law.toml", ScalingLawContract
    )


def _write_partial_evaluations(data_root: Path) -> Path:
    scaling_contract = _scaling_contract()
    recipes = tuple(sorted(scaling_contract.source_group_map.values()))
    complete_fit_sizes = set(scaling_contract.models[:5])
    rows: list[dict[str, object]] = []
    for size_index, size_id in enumerate(scaling_contract.models, start=1):
        parameter_count = float(size_index * 1_000_000)
        for recipe_index, recipe in enumerate(recipes, start=1):
            for step, tokens in ((1, 90), (2, 100)):
                for task in DEFAULT_TASK_GROUPING.source_tasks:
                    is_mmlu = task in DEFAULT_TASK_GROUPING.mmlu_subjects
                    final_offset = 0.0 if step == 2 else 1.0
                    rows.append(
                        {
                            "recipe": recipe,
                            "params": size_id,
                            "seed": "default",
                            "step": step,
                            "task": task,
                            "tokens": tokens,
                            "compute": 6 * parameter_count * tokens,
                            "exact_parameter_count": parameter_count,
                            "primary_metric": (
                                0.2
                                + recipe_index / 1_000
                                + size_index / 100
                                + step / 10_000
                                + (0.2 if is_mmlu else 0.0)
                            ),
                            "logits_per_byte_corr": (
                                (3.0 if is_mmlu else 1.0) + final_offset
                                if size_id in complete_fit_sizes
                                else None
                            ),
                        }
                    )
    path = data_root / "processed/scaling-law/evaluations.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_partial_scaling_surface_records_missing_groups_and_prefix_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = _write_partial_evaluations(tmp_path)
    captured_final_losses: list[tuple[float, ...]] = []

    def fake_held_out_prediction(
        final_losses: object,
        evaluations: object,
        *,
        target: object,
        variant: ScalingVariant,
    ) -> object:
        del evaluations, variant
        losses = tuple(point.loss for point in final_losses)  # type: ignore[attr-defined]
        captured_final_losses.append(losses)
        actual_score = target.actual_score  # type: ignore[attr-defined]
        return SimpleNamespace(
            predicted_score=actual_score,
            target=SimpleNamespace(actual_score=actual_score),
        )

    monkeypatch.setattr(
        scaling_adapter, "held_out_prediction", fake_held_out_prediction
    )
    contract = load_paper_validation_contract()
    registry = load_repository_claim_registry(_REPOSITORY_ROOT)

    results, series = scaling_adapter.run_scaling_law_attempts(
        repository_root=_REPOSITORY_ROOT,
        data_root=tmp_path,
        registry=registry,
        contract=contract,
        input_identities={},
    )

    assert len(results) == 20
    assert {result.outcome for result in results} == {
        ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
    }
    assert tuple(value.id for value in series) == (
        "dd-0180-paper-analog",
        "dd-0368-paper-analog",
        "dd-0369-paper-analog",
    )
    all_variants = next(value for value in series if value.id == "dd-0180-paper-analog")
    assert len(all_variants.points) == 24
    assert {
        next(
            dimension.value
            for dimension in point.dimensions
            if dimension.name == "subset"
        )
        for point in all_variants.points
    } == {"prefix-03", "prefix-04", "prefix-05"}
    assert all(
        next(
            dimension.value
            for dimension in point.dimensions
            if dimension.name == "subset_kind"
        )
        == "prefix"
        for point in all_variants.points
    )

    result = next(value for value in results if value.claim_id == "DD-0180")
    assert result.plot_series_ids == ("dd-0180-paper-analog",)
    assert (
        "recipe=c4|size=20M|seed=default|missing=task_loss_surface|incomplete_steps=1,2"
    ) in result.missing_groups
    assert "supported_size_subset_count=3" in result.diagnostics
    assert (
        result.row_selections[0].local_parquet_sha256
        == hashlib.sha256(input_path.read_bytes()).hexdigest()
    )
    assert result.row_selections[0].selected_row_count == 46_200
    assert captured_final_losses
    # The first macro task loss is MMLU-first then OLMES-macro, and the final
    # loss averages both checkpoints in the configured final 10% window.
    assert captured_final_losses[0][0] == pytest.approx(1.7)


def test_scaling_input_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    _write_partial_evaluations(tmp_path)

    with pytest.raises(ValueError, match="changed after input identity capture"):
        scaling_adapter.run_scaling_law_attempts(
            repository_root=_REPOSITORY_ROOT,
            data_root=tmp_path,
            registry=load_repository_claim_registry(_REPOSITORY_ROOT),
            contract=load_paper_validation_contract(),
            input_identities={
                "scaling_evaluations": ContentIdentity(
                    id="scaling_evaluations", sha256="0" * 64
                )
            },
        )


def test_no_scaling_attempts_do_not_read_inputs() -> None:
    contract = load_paper_validation_contract()
    without_scaling = contract.model_copy(
        update={
            "attempts": tuple(
                attempt
                for attempt in contract.attempts
                if attempt.analysis_id.value != "scaling_law"
            )
        }
    )

    assert scaling_adapter.run_scaling_law_attempts(
        repository_root=Path("does-not-exist"),
        data_root=Path("does-not-exist"),
        registry=load_repository_claim_registry(_REPOSITORY_ROOT),
        contract=without_scaling,
        input_identities={},
    ) == ((), ())

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from datadec.paper.contracts import load_claim_registry, load_validation_contract
from datadec.paper.models import (
    AttemptRole,
    CheckpointRule,
    ContentIdentity,
    ValidationOutcome,
)
from datadec.paper.verifiers.single_scale import (
    run_per_task_attempts,
    run_single_scale_attempts,
)

_REPOSITORY_ROOT = Path(__file__).parents[2]
_FIXTURE = Path(__file__).parent / "fixtures" / "olmes_single_scale_regression.parquet"


@pytest.fixture
def adapter_inputs(tmp_path: Path) -> tuple[Path, ContentIdentity]:
    destination = tmp_path / "processed" / "olmes.parquet"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(_FIXTURE, destination)
    with destination.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    return tmp_path, ContentIdentity(id="olmes_aggregate", sha256=digest)


def _contracts():
    return (
        load_claim_registry(_REPOSITORY_ROOT / "docs/paper/claims.toml"),
        load_validation_contract(_REPOSITORY_ROOT / "configs/paper_validation.toml"),
    )


def test_single_scale_adapter_persists_headline_and_aggregate_series(
    adapter_inputs: tuple[Path, ContentIdentity],
) -> None:
    data_root, identity = adapter_inputs
    registry, contract = _contracts()

    attempts, series = run_single_scale_attempts(
        repository_root=_REPOSITORY_ROOT,
        data_root=data_root,
        registry=registry,
        contract=contract,
        input_identities={identity.id: identity},
    )

    assert tuple(item.attempt_id for item in attempts) == (
        "dd-0011-default",
        "dd-0011-preceding-common-complete-1",
        "dd-0011-preceding-common-complete-2",
        "dd-0169-default",
    )
    default = attempts[0]
    assert default.role is AttemptRole.DEFAULT
    assert default.target_value == 0.8
    assert default.computed_value == 0.8033333333333333
    assert default.outcome is ValidationOutcome.APPROXIMATELY_REPRODUCED
    assert default.denominator == 900
    assert default.standard_deviation == 0.029627314724385286
    assert default.ddof == 1
    assert default.seeds == ("default", "small aux 2", "small aux 3")
    assert default.target_ties == 0
    assert default.predicted_ties == 0
    assert tuple(item.actual_step for item in default.checkpoint_selections) == (
        69_369,
        37_500,
    )
    assert any("38157" in diagnostic for diagnostic in default.diagnostics)
    assert all(
        selection.local_parquet_sha256 == identity.sha256
        for selection in default.row_selections
    )
    assert tuple(item.computed_value for item in attempts[1:3]) == (
        0.7977777777777778,
        0.7999999999999999,
    )
    assert all(item.role is AttemptRole.SENSITIVITY for item in attempts[1:3])
    assert tuple(item.checkpoint_selections[1].rule for item in attempts[1:3]) == (
        CheckpointRule.PRECEDING_COMMON_COMPLETE,
        CheckpointRule.PRECEDING_COMMON_COMPLETE,
    )
    assert tuple(
        item.checkpoint_selections[1].actual_step for item in attempts[1:3]
    ) == (36_250, 35_000)

    assert len(series) == 1
    aggregate = series[0]
    assert aggregate.id == "dd-0169-paper-analog"
    assert aggregate.attempt_id == "dd-0169-default"
    assert len(aggregate.points) == 6
    assert {point.dimensions[0].value for point in aggregate.points} == {
        "150M",
        "1B",
    }
    assert attempts[-1].plot_series_ids == (aggregate.id,)


def test_per_task_adapter_persists_all_ten_logical_task_curves(
    adapter_inputs: tuple[Path, ContentIdentity],
) -> None:
    data_root, identity = adapter_inputs
    registry, contract = _contracts()

    attempts, series = run_per_task_attempts(
        repository_root=_REPOSITORY_ROOT,
        data_root=data_root,
        registry=registry,
        contract=contract,
        input_identities={identity.id: identity},
    )

    assert tuple(item.attempt_id for item in attempts) == ("dd-0148-default",)
    assert attempts[0].outcome is ValidationOutcome.REPRODUCED
    assert attempts[0].plot_series_ids == ("dd-0148-paper-analog",)
    assert len(series) == 1
    per_task = series[0]
    assert per_task.id == "dd-0148-paper-analog"
    assert len(per_task.points) == 60
    assert {point.dimensions[0].value for point in per_task.points} == {
        "arc_challenge",
        "arc_easy",
        "boolq",
        "csqa",
        "hellaswag",
        "mmlu",
        "openbookqa",
        "piqa",
        "socialiqa",
        "winogrande",
    }
    assert all(point.measures[0].value > 0 for point in per_task.points)


def test_adapter_rejects_an_unverified_input_identity(
    adapter_inputs: tuple[Path, ContentIdentity],
) -> None:
    data_root, _identity = adapter_inputs
    registry, contract = _contracts()

    with pytest.raises(ValueError, match="identity differs"):
        run_single_scale_attempts(
            repository_root=_REPOSITORY_ROOT,
            data_root=data_root,
            registry=registry,
            contract=contract,
            input_identities={
                "olmes_aggregate": ContentIdentity(
                    id="olmes_aggregate",
                    sha256="0" * 64,
                )
            },
        )

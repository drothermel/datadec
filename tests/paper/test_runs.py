from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import orjson
import pytest

import datadec.paper.runs as runs_module
from datadec.paper.models import (
    AttemptResult,
    AttemptRole,
    ClaimKind,
    ContentIdentity,
    PaperTarget,
    RowSelection,
    ValidationOutcome,
)
from datadec.paper.runs import create_analysis_bundle, load_analysis_bundle

_STARTED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
_INPUT_SHA256 = "a" * 64
_KEY_SHA256 = "b" * 64


def _target(claim_id: str = "DD-0001", value: float = 0.8) -> PaperTarget:
    return PaperTarget(
        claim_id=claim_id,
        family="single_scale",
        kind=ClaimKind.EMPIRICAL_NUMERIC,
        source_file="docs/paper/example_paper.tex",
        line_start=10,
        line_end=10,
        source_text="Approximately 80 percent.",
        value=value,
    )


def _attempt(
    claim_id: str = "DD-0001",
    *,
    attempt_id: str = "dd-0001-default",
    input_id: str = "olmes_aggregate",
    input_sha256: str = _INPUT_SHA256,
    target_value: float = 0.8,
) -> AttemptResult:
    return AttemptResult(
        attempt_id=attempt_id,
        claim_id=claim_id,
        role=AttemptRole.DEFAULT,
        comparison_rule_id="approximately-one-point-v1",
        comparison_rule_version=1,
        transformation_ids=("recipe-ranking-v1",),
        row_selections=(
            RowSelection(
                logical_table_id=input_id,
                columns=("data", "primary_metric"),
                predicates=(),
                local_parquet_sha256=input_sha256,
                selected_row_count=25,
                selected_key_sha256=_KEY_SHA256,
            ),
        ),
        target_value=target_value,
        computed_value=0.8033333333333333,
        unrounded_difference=0.0033333333333332993,
        denominator=900,
        outcome=ValidationOutcome.APPROXIMATELY_REPRODUCED,
    )


def _create(runs_root: Path, run_id: str = "run-001"):  # type: ignore[no-untyped-def]
    return create_analysis_bundle(
        runs_root,
        run_id=run_id,
        started_at=_STARTED_AT,
        completed_at=_STARTED_AT + timedelta(minutes=2),
        input_identities=(ContentIdentity(id="olmes_aggregate", sha256=_INPUT_SHA256),),
        targets=(_target(),),
        attempts=(_attempt(),),
        plot_series=(),
    )


def test_format_two_bundle_has_exact_canonical_files(tmp_path: Path) -> None:
    bundle = _create(tmp_path)
    run_directory = tmp_path / "run-001"

    assert {path.name for path in run_directory.iterdir()} == {
        "manifest.json",
        "targets.json",
        "attempts.json",
        "plot-series.json",
    }
    manifest = orjson.loads((run_directory / "manifest.json").read_bytes())
    assert manifest["run_format"] == 2
    assert set(manifest) == {
        "attempts_identity",
        "code_trace",
        "completed_at",
        "input_identities",
        "plot_series_identity",
        "run_format",
        "run_id",
        "runtime_trace",
        "started_at",
        "targets_identity",
    }
    assert set(orjson.loads((run_directory / "targets.json").read_bytes())) == {
        "metadata_discrepancies",
        "targets",
    }
    assert set(orjson.loads((run_directory / "attempts.json").read_bytes())) == {
        "attempts"
    }
    assert set(orjson.loads((run_directory / "plot-series.json").read_bytes())) == {
        "plot_series"
    }
    assert all(path.read_bytes().endswith(b"\n") for path in run_directory.iterdir())
    assert load_analysis_bundle(tmp_path, "run-001") == bundle


def test_payloads_are_deterministic_across_run_identity(tmp_path: Path) -> None:
    _create(tmp_path / "first", "run-a")
    _create(tmp_path / "second", "run-b")

    for filename in ("targets.json", "attempts.json", "plot-series.json"):
        assert (tmp_path / "first" / "run-a" / filename).read_bytes() == (
            tmp_path / "second" / "run-b" / filename
        ).read_bytes()


def test_create_refuses_existing_run_without_changing_it(tmp_path: Path) -> None:
    _create(tmp_path)
    original = (tmp_path / "run-001" / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        _create(tmp_path)

    assert (tmp_path / "run-001" / "manifest.json").read_bytes() == original
    assert [path.name for path in tmp_path.iterdir()] == ["run-001"]


def test_same_id_collision_preserves_winner_and_cleans_loser_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_rename = runs_module._rename_no_replace
    installing_winner = False

    def install_winner_then_rename(source: Path, destination: Path) -> None:
        nonlocal installing_winner
        if not installing_winner:
            installing_winner = True
            _create(tmp_path)
        original_rename(source, destination)

    monkeypatch.setattr(runs_module, "_rename_no_replace", install_winner_then_rename)

    with pytest.raises(FileExistsError):
        _create(tmp_path)

    assert load_analysis_bundle(tmp_path, "run-001").manifest.run_id == "run-001"
    assert [path.name for path in tmp_path.iterdir()] == ["run-001"]


def test_injected_write_failure_leaves_no_run_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write = runs_module._write_new_file

    def fail_attempts(path: Path, contents: bytes) -> None:
        if path.name == "attempts.json":
            raise OSError("injected attempts failure")
        original_write(path, contents)

    monkeypatch.setattr(runs_module, "_write_new_file", fail_attempts)

    with pytest.raises(OSError, match="injected attempts failure"):
        _create(tmp_path)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "filename", ["targets.json", "attempts.json", "plot-series.json"]
)
def test_load_detects_content_tampering(tmp_path: Path, filename: str) -> None:
    _create(tmp_path)
    path = tmp_path / "run-001" / filename
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="SHA256"):
        load_analysis_bundle(tmp_path, "run-001")


def test_load_detects_manifest_tampering(tmp_path: Path) -> None:
    _create(tmp_path)
    path = tmp_path / "run-001" / "manifest.json"
    path.write_bytes(path.read_bytes().replace(b'"run_format":2', b'"run_format":3'))

    with pytest.raises(ValueError):
        load_analysis_bundle(tmp_path, "run-001")


def test_create_rejects_unknown_input_cross_reference(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown input"):
        create_analysis_bundle(
            tmp_path,
            run_id="bad-input",
            started_at=_STARTED_AT,
            completed_at=_STARTED_AT,
            input_identities=(
                ContentIdentity(id="olmes_aggregate", sha256=_INPUT_SHA256),
            ),
            targets=(_target(),),
            attempts=(_attempt(input_id="other_table"),),
            plot_series=(),
        )
    assert list(tmp_path.iterdir()) == []


def test_create_rejects_missing_default_result(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one default"):
        create_analysis_bundle(
            tmp_path,
            run_id="incomplete",
            started_at=_STARTED_AT,
            completed_at=_STARTED_AT,
            input_identities=(
                ContentIdentity(id="olmes_aggregate", sha256=_INPUT_SHA256),
            ),
            targets=(_target(),),
            attempts=(),
            plot_series=(),
        )


def test_create_rejects_target_value_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="paper target"):
        create_analysis_bundle(
            tmp_path,
            run_id="mismatch",
            started_at=_STARTED_AT,
            completed_at=_STARTED_AT,
            input_identities=(
                ContentIdentity(id="olmes_aggregate", sha256=_INPUT_SHA256),
            ),
            targets=(_target(),),
            attempts=(_attempt(target_value=0.75),),
            plot_series=(),
        )


def test_extra_run_file_is_rejected(tmp_path: Path) -> None:
    _create(tmp_path)
    (tmp_path / "run-001" / "extra.json").write_bytes(b"{}\n")

    with pytest.raises(ValueError, match="exactly the four"):
        load_analysis_bundle(tmp_path, "run-001")

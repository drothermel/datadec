from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import orjson
from pydantic import TypeAdapter

from datadec.paper.models import (
    AnalysisBundle,
    AnalysisManifest,
    AttemptResult,
    AttemptRole,
    CodeTrace,
    ContentIdentity,
    MetadataDiscrepancy,
    PaperTarget,
    PlotSeries,
    RuntimeTrace,
    ValidationOutcome,
)

_MANIFEST_FILENAME = "manifest.json"
_TARGETS_FILENAME = "targets.json"
_ATTEMPTS_FILENAME = "attempts.json"
_PLOT_SERIES_FILENAME = "plot-series.json"
_BUNDLE_FILENAMES = frozenset(
    {
        _MANIFEST_FILENAME,
        _TARGETS_FILENAME,
        _ATTEMPTS_FILENAME,
        _PLOT_SERIES_FILENAME,
    }
)
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CANONICAL_JSON_OPTIONS = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SORT_KEYS
_TARGETS_ADAPTER = TypeAdapter(tuple[PaperTarget, ...])
_DISCREPANCIES_ADAPTER = TypeAdapter(tuple[MetadataDiscrepancy, ...])
_ATTEMPTS_ADAPTER = TypeAdapter(tuple[AttemptResult, ...])
_PLOT_SERIES_ADAPTER = TypeAdapter(tuple[PlotSeries, ...])
_ValueT = TypeVar("_ValueT")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return orjson.dumps(value, option=_CANONICAL_JSON_OPTIONS)
    except orjson.JSONEncodeError as error:
        raise ValueError("analysis bundles require finite JSON values") from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_run_id(run_id: str) -> None:
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run ID must be a safe single path component")


def _write_new_file(path: Path, contents: bytes) -> None:
    with path.open("xb") as file:
        file.write(contents)
        file.flush()
        os.fsync(file.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically install a staged directory without replacing a winner."""
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename_exclusive = library.renamex_np
        rename_exclusive.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        rename_exclusive = library.renameat2
        rename_exclusive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(
            -100, source_bytes, -100, destination_bytes, 0x00000001
        )
    elif os.name == "nt":
        os.rename(source, destination)
        return
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory rename is unsupported on this platform",
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def _remove_staging_directory(staging: Path, runs_root: Path, run_id: str) -> None:
    resolved_root = runs_root.resolve(strict=True)
    if staging.parent.resolve(strict=True) != resolved_root:
        raise RuntimeError("refusing to remove staging directory outside runs root")
    if not staging.name.startswith(f".{run_id}.staging-"):
        raise RuntimeError("refusing to remove an unexpected staging directory")
    mode = staging.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise RuntimeError(
            "refusing to recursively remove a non-directory staging path"
        )
    shutil.rmtree(staging)


def _ordered_unique(
    values: Iterable[_ValueT],
    *,
    key: Callable[[_ValueT], str],
    description: str,
) -> tuple[_ValueT, ...]:
    supplied = tuple(values)
    identities = tuple(key(value) for value in supplied)
    if len(identities) != len(set(identities)):
        raise ValueError(f"{description} must be unique")
    return tuple(sorted(supplied, key=key))


def _targets_payload(
    targets: tuple[PaperTarget, ...],
    metadata_discrepancies: tuple[MetadataDiscrepancy, ...],
) -> dict[str, object]:
    return {
        "metadata_discrepancies": [
            value.model_dump(mode="json") for value in metadata_discrepancies
        ],
        "targets": [value.model_dump(mode="json") for value in targets],
    }


def _attempts_payload(attempts: tuple[AttemptResult, ...]) -> dict[str, object]:
    return {"attempts": [value.model_dump(mode="json") for value in attempts]}


def _plot_series_payload(plot_series: tuple[PlotSeries, ...]) -> dict[str, object]:
    return {
        "plot_series": [value.model_dump(mode="json") for value in plot_series],
    }


def _validate_bundle(bundle: AnalysisBundle) -> None:
    targets = {target.claim_id: target for target in bundle.targets}
    inputs = {
        identity.id: identity.sha256 for identity in bundle.manifest.input_identities
    }
    defaults: dict[str, AttemptResult] = {}
    attempts = {attempt.attempt_id: attempt for attempt in bundle.attempts}
    for attempt in bundle.attempts:
        target = targets[attempt.claim_id]
        if attempt.target_value != target.value:
            raise ValueError(
                f"attempt {attempt.attempt_id} target differs from its paper target"
            )
        if attempt.role is AttemptRole.DEFAULT:
            if attempt.claim_id in defaults:
                raise ValueError(
                    f"multiple default results exist for claim {attempt.claim_id}"
                )
            defaults[attempt.claim_id] = attempt
        elif attempt.parent_attempt_id not in attempts:
            raise ValueError(
                f"attempt {attempt.attempt_id} references an unknown parent attempt"
            )

        not_assessable = (
            attempt.outcome is ValidationOutcome.NOT_ASSESSABLE_FROM_DD_PARSED
        )
        if not_assessable and (not attempt.missing_groups or not attempt.diagnostics):
            raise ValueError(
                "not-assessable results require missing groups and diagnostics"
            )
        for selection in attempt.row_selections:
            expected_sha256 = inputs.get(selection.logical_table_id)
            if expected_sha256 is None:
                if not_assessable and selection.selected_row_count == 0:
                    continue
                raise ValueError(
                    f"attempt {attempt.attempt_id} references unknown input "
                    f"{selection.logical_table_id}"
                )
            if selection.local_parquet_sha256 != expected_sha256:
                raise ValueError(
                    f"attempt {attempt.attempt_id} input SHA256 differs from manifest"
                )

    missing_defaults = sorted(set(targets) - defaults.keys())
    unexpected_defaults = sorted(defaults.keys() - set(targets))
    if missing_defaults or unexpected_defaults:
        raise ValueError(
            "bundle requires exactly one default result per paper target: "
            f"missing={missing_defaults}, unexpected={unexpected_defaults}"
        )


def create_analysis_bundle(
    runs_root: str | Path,
    *,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    input_identities: Iterable[ContentIdentity],
    targets: Iterable[PaperTarget],
    attempts: Iterable[AttemptResult],
    plot_series: Iterable[PlotSeries],
    metadata_discrepancies: Iterable[MetadataDiscrepancy] = (),
    code_trace: CodeTrace | None = None,
    runtime_trace: RuntimeTrace | None = None,
) -> AnalysisBundle:
    """Persist one immutable format-3 analysis bundle."""
    _validate_run_id(run_id)
    ordered_inputs = _ordered_unique(
        input_identities, key=lambda value: value.id, description="input identities"
    )
    ordered_targets = _ordered_unique(
        targets, key=lambda value: value.claim_id, description="paper targets"
    )
    ordered_discrepancies = _ordered_unique(
        metadata_discrepancies,
        key=lambda value: value.claim_id,
        description="metadata discrepancies",
    )
    ordered_attempts = _ordered_unique(
        attempts, key=lambda value: value.attempt_id, description="attempt results"
    )
    ordered_plot_series = _ordered_unique(
        plot_series, key=lambda value: value.id, description="plot series"
    )

    targets_bytes = _canonical_json_bytes(
        _targets_payload(ordered_targets, ordered_discrepancies)
    )
    attempts_bytes = _canonical_json_bytes(_attempts_payload(ordered_attempts))
    plot_series_bytes = _canonical_json_bytes(_plot_series_payload(ordered_plot_series))
    manifest = AnalysisManifest(
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        code_trace=code_trace,
        runtime_trace=runtime_trace,
        input_identities=ordered_inputs,
        targets_identity=ContentIdentity(
            id=_TARGETS_FILENAME, sha256=_sha256(targets_bytes)
        ),
        attempts_identity=ContentIdentity(
            id=_ATTEMPTS_FILENAME, sha256=_sha256(attempts_bytes)
        ),
        plot_series_identity=ContentIdentity(
            id=_PLOT_SERIES_FILENAME, sha256=_sha256(plot_series_bytes)
        ),
    )
    bundle = AnalysisBundle(
        manifest=manifest,
        targets=ordered_targets,
        metadata_discrepancies=ordered_discrepancies,
        attempts=ordered_attempts,
        plot_series=ordered_plot_series,
    )
    _validate_bundle(bundle)
    manifest_bytes = _canonical_json_bytes(manifest.model_dump(mode="json"))

    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    final_directory = root / run_id
    staging_directory: Path | None = None
    try:
        if final_directory.exists() or final_directory.is_symlink():
            raise FileExistsError(f"run already exists: {run_id}")
        staging_directory = Path(
            tempfile.mkdtemp(prefix=f".{run_id}.staging-", dir=root)
        )
        for filename, contents in (
            (_TARGETS_FILENAME, targets_bytes),
            (_ATTEMPTS_FILENAME, attempts_bytes),
            (_PLOT_SERIES_FILENAME, plot_series_bytes),
            (_MANIFEST_FILENAME, manifest_bytes),
        ):
            path = staging_directory / filename
            _write_new_file(path, contents)
            if _sha256(path.read_bytes()) != _sha256(contents):
                raise OSError(f"written {filename} failed SHA256 verification")
        _fsync_directory(staging_directory)
        _rename_no_replace(staging_directory, final_directory)
        staging_directory = None
        _fsync_directory(root)
    except BaseException:
        if staging_directory is not None and staging_directory.exists():
            _remove_staging_directory(staging_directory, root, run_id)
        raise
    return bundle


def _read_regular_file(run_directory: Path, filename: str) -> bytes:
    path = run_directory / filename
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"run file must be a non-symlink regular file: {filename}")
    return path.read_bytes()


def _load_payload(
    contents: bytes,
    *,
    expected_keys: set[str],
    description: str,
) -> dict[str, object]:
    raw = orjson.loads(contents)
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise ValueError(
            f"{description} file must contain exactly {sorted(expected_keys)}"
        )
    return raw


def load_analysis_bundle(runs_root: str | Path, run_id: str) -> AnalysisBundle:
    """Load a format-3 bundle and verify canonical bytes, identities, and links."""
    _validate_run_id(run_id)
    run_directory = Path(runs_root).resolve(strict=True) / run_id
    mode = run_directory.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError("run path must be a non-symlink directory")
    entries = {path.name for path in run_directory.iterdir()}
    if entries != _BUNDLE_FILENAMES:
        raise ValueError("run directory must contain exactly the four format-3 files")

    manifest_bytes = _read_regular_file(run_directory, _MANIFEST_FILENAME)
    manifest = AnalysisManifest.model_validate_json(manifest_bytes)
    if manifest.run_id != run_id:
        raise ValueError("manifest run ID does not match its directory")
    if manifest_bytes != _canonical_json_bytes(manifest.model_dump(mode="json")):
        raise ValueError("manifest is not canonical JSON")

    targets_bytes = _read_regular_file(run_directory, _TARGETS_FILENAME)
    attempts_bytes = _read_regular_file(run_directory, _ATTEMPTS_FILENAME)
    plot_series_bytes = _read_regular_file(run_directory, _PLOT_SERIES_FILENAME)
    for description, contents, identity in (
        ("targets", targets_bytes, manifest.targets_identity),
        ("attempts", attempts_bytes, manifest.attempts_identity),
        ("plot series", plot_series_bytes, manifest.plot_series_identity),
    ):
        if _sha256(contents) != identity.sha256:
            raise ValueError(f"{description} SHA256 does not match manifest")

    raw_targets = _load_payload(
        targets_bytes,
        expected_keys={"metadata_discrepancies", "targets"},
        description="targets",
    )
    raw_attempts = _load_payload(
        attempts_bytes, expected_keys={"attempts"}, description="attempts"
    )
    raw_plot_series = _load_payload(
        plot_series_bytes,
        expected_keys={"plot_series"},
        description="plot series",
    )
    targets = _TARGETS_ADAPTER.validate_json(orjson.dumps(raw_targets["targets"]))
    metadata_discrepancies = _DISCREPANCIES_ADAPTER.validate_json(
        orjson.dumps(raw_targets["metadata_discrepancies"])
    )
    attempts = _ATTEMPTS_ADAPTER.validate_json(orjson.dumps(raw_attempts["attempts"]))
    plot_series = _PLOT_SERIES_ADAPTER.validate_json(
        orjson.dumps(raw_plot_series["plot_series"])
    )
    for description, contents, expected in (
        (
            "targets",
            targets_bytes,
            _canonical_json_bytes(_targets_payload(targets, metadata_discrepancies)),
        ),
        (
            "attempts",
            attempts_bytes,
            _canonical_json_bytes(_attempts_payload(attempts)),
        ),
        (
            "plot series",
            plot_series_bytes,
            _canonical_json_bytes(_plot_series_payload(plot_series)),
        ),
    ):
        if contents != expected:
            raise ValueError(f"{description} file is not canonical JSON")

    if tuple(sorted(targets, key=lambda value: value.claim_id)) != targets:
        raise ValueError("paper targets must use deterministic claim-ID ordering")
    if (
        tuple(sorted(metadata_discrepancies, key=lambda value: value.claim_id))
        != metadata_discrepancies
    ):
        raise ValueError("metadata discrepancies must use claim-ID ordering")
    if tuple(sorted(attempts, key=lambda value: value.attempt_id)) != attempts:
        raise ValueError("attempt results must use deterministic attempt-ID ordering")
    if tuple(sorted(plot_series, key=lambda value: value.id)) != plot_series:
        raise ValueError("plot series must use deterministic series-ID ordering")

    bundle = AnalysisBundle(
        manifest=manifest,
        targets=targets,
        metadata_discrepancies=metadata_discrepancies,
        attempts=attempts,
        plot_series=plot_series,
    )
    _validate_bundle(bundle)
    return bundle


__all__ = ["create_analysis_bundle", "load_analysis_bundle"]

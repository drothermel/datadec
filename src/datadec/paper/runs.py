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
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import orjson

from datadec.paper.models import (
    CodeIdentity,
    CodeTreeState,
    ContentIdentity,
    Observation,
    ObservationFileIdentity,
    RunBundle,
    RunManifest,
    RuntimeIdentity,
)

_OBSERVATIONS_FILENAME = "observations.json"
_MANIFEST_FILENAME = "manifest.json"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CANONICAL_JSON_OPTIONS = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SORT_KEYS


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return orjson.dumps(value, option=_CANONICAL_JSON_OPTIONS)
    except orjson.JSONEncodeError as error:
        raise ValueError("run bundles require finite JSON values") from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_run_id(run_id: str) -> None:
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run ID must be a safe single path component")


def _validate_filename(filename: str, description: str) -> None:
    if filename in {"", ".", ".."} or Path(filename).name != filename:
        raise ValueError(f"{description} filename must be a safe bare filename")


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
    # POSIX rename replaces an existing empty directory. These platform calls add
    # the exclusion flag while retaining a single atomic namespace operation.
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


def _ordered_observations(
    observations: Iterable[Observation], active_claim_ids: Iterable[str]
) -> tuple[Observation, ...]:
    active_ids = tuple(active_claim_ids)
    if len(active_ids) != len(set(active_ids)):
        raise ValueError("active claim IDs must be unique")
    active_id_set = set(active_ids)

    supplied = tuple(observations)
    supplied_ids = tuple(observation.claim_id for observation in supplied)
    duplicate_ids = sorted(
        claim_id for claim_id in set(supplied_ids) if supplied_ids.count(claim_id) > 1
    )
    if duplicate_ids:
        raise ValueError(
            "duplicate observations for claim IDs: " + ", ".join(duplicate_ids)
        )

    supplied_id_set = set(supplied_ids)
    unknown_ids = sorted(supplied_id_set - active_id_set)
    if unknown_ids:
        raise ValueError(
            "observations reference unknown claims: " + ", ".join(unknown_ids)
        )
    missing_ids = sorted(active_id_set - supplied_id_set)
    if missing_ids:
        raise ValueError(
            "observations are missing active claims: " + ", ".join(missing_ids)
        )
    return tuple(sorted(supplied, key=lambda observation: observation.claim_id))


def _validate_cross_references(
    manifest: RunManifest, observations: tuple[Observation, ...]
) -> None:
    input_ids = {identity.id for identity in manifest.input_identities}
    artifact_ids = {identity.id for identity in manifest.artifact_identities}
    for observation in observations:
        unknown_input_ids = set(observation.input_ids) - input_ids
        if unknown_input_ids:
            unknown = ", ".join(sorted(unknown_input_ids))
            raise ValueError(
                f"observation {observation.claim_id} references unknown inputs: {unknown}"
            )
        unknown_artifact_ids = set(observation.artifact_ids) - artifact_ids
        if unknown_artifact_ids:
            unknown = ", ".join(sorted(unknown_artifact_ids))
            raise ValueError(
                f"observation {observation.claim_id} references unknown artifacts: {unknown}"
            )
        if observation.blocker is not None:
            present_missing_ids = set(observation.blocker.missing_input_ids) & input_ids
            if present_missing_ids:
                present = ", ".join(sorted(present_missing_ids))
                raise ValueError(
                    f"observation {observation.claim_id} marks present inputs as missing: "
                    f"{present}"
                )


def create_run_bundle(
    runs_root: str | Path,
    *,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    paper_identity: ContentIdentity,
    config_identity: ContentIdentity,
    claims_identity: ContentIdentity,
    code_identity: CodeIdentity,
    runtime_identity: RuntimeIdentity,
    active_claim_ids: Iterable[str],
    observations: Iterable[Observation],
    input_identities: Iterable[ContentIdentity] = (),
    artifact_identities: Iterable[ContentIdentity] = (),
    observations_filename: str = _OBSERVATIONS_FILENAME,
    manifest_filename: str = _MANIFEST_FILENAME,
) -> RunBundle:
    """Create one immutable, complete run bundle and return its validated value."""
    _validate_run_id(run_id)
    _validate_filename(observations_filename, "observations")
    _validate_filename(manifest_filename, "manifest")
    if observations_filename == manifest_filename:
        raise ValueError("observations and manifest filenames must differ")

    ordered_observations = _ordered_observations(observations, active_claim_ids)
    ordered_inputs = tuple(sorted(input_identities, key=lambda identity: identity.id))
    ordered_artifacts = tuple(
        sorted(artifact_identities, key=lambda identity: identity.id)
    )
    observations_payload = {
        "observations": [
            observation.model_dump(mode="json") for observation in ordered_observations
        ]
    }
    observations_bytes = _canonical_json_bytes(observations_payload)
    observations_identity = ObservationFileIdentity(
        filename=observations_filename,
        sha256=_sha256(observations_bytes),
        byte_count=len(observations_bytes),
        observation_count=len(ordered_observations),
    )
    manifest = RunManifest(
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        paper_identity=paper_identity,
        config_identity=config_identity,
        claims_identity=claims_identity,
        code_identity=code_identity,
        runtime_identity=runtime_identity,
        input_identities=ordered_inputs,
        artifact_identities=ordered_artifacts,
        observations_identity=observations_identity,
    )
    _validate_cross_references(manifest, ordered_observations)
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
        _write_new_file(staging_directory / observations_filename, observations_bytes)
        written_observations = (staging_directory / observations_filename).read_bytes()
        if _sha256(written_observations) != observations_identity.sha256:
            raise OSError("written observations failed SHA256 verification")
        _write_new_file(staging_directory / manifest_filename, manifest_bytes)
        written_manifest = (staging_directory / manifest_filename).read_bytes()
        if _sha256(written_manifest) != _sha256(manifest_bytes):
            raise OSError("written manifest failed SHA256 verification")
        _fsync_directory(staging_directory)

        _rename_no_replace(staging_directory, final_directory)
        staging_directory = None
        _fsync_directory(root)
    except BaseException:
        if staging_directory is not None and staging_directory.exists():
            _remove_staging_directory(staging_directory, root, run_id)
        raise

    return RunBundle(manifest=manifest, observations=ordered_observations)


def load_run_bundle(
    runs_root: str | Path,
    run_id: str,
    *,
    active_claim_ids: Iterable[str] | None = None,
    manifest_filename: str = _MANIFEST_FILENAME,
) -> RunBundle:
    """Load a complete run and validate its canonical files and references."""
    _validate_run_id(run_id)
    _validate_filename(manifest_filename, "manifest")
    run_directory = Path(runs_root).resolve(strict=True) / run_id
    mode = run_directory.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError("run path must be a non-symlink directory")

    manifest_bytes = (run_directory / manifest_filename).read_bytes()
    manifest = RunManifest.model_validate(orjson.loads(manifest_bytes))
    if manifest.run_id != run_id:
        raise ValueError("manifest run ID does not match its directory")
    if manifest_bytes != _canonical_json_bytes(manifest.model_dump(mode="json")):
        raise ValueError("manifest is not canonical JSON")

    observations_path = run_directory / manifest.observations_identity.filename
    observations_bytes = observations_path.read_bytes()
    if len(observations_bytes) != manifest.observations_identity.byte_count:
        raise ValueError("observations byte count does not match manifest")
    if _sha256(observations_bytes) != manifest.observations_identity.sha256:
        raise ValueError("observations SHA256 does not match manifest")
    raw_observations = orjson.loads(observations_bytes)
    if not isinstance(raw_observations, dict) or set(raw_observations) != {
        "observations"
    }:
        raise ValueError(
            "observations file must contain only the pinned observations key"
        )
    raw_values = raw_observations["observations"]
    if not isinstance(raw_values, list):
        raise ValueError("observations value must be a JSON array")
    observations = tuple(Observation.model_validate(value) for value in raw_values)
    if tuple(sorted(observations, key=lambda value: value.claim_id)) != observations:
        raise ValueError("observations must use deterministic claim-ID ordering")
    if len(observations) != manifest.observations_identity.observation_count:
        raise ValueError("observation count does not match manifest")
    if observations_bytes != _canonical_json_bytes(
        {"observations": [value.model_dump(mode="json") for value in observations]}
    ):
        raise ValueError("observations file is not canonical JSON")
    if active_claim_ids is not None:
        observations = _ordered_observations(observations, active_claim_ids)
    _validate_cross_references(manifest, observations)
    return RunBundle(manifest=manifest, observations=observations)


def validate_run_qualification(manifest: RunManifest) -> None:
    """Require the clean-code terminal state used for qualified results."""
    if not manifest.complete:
        raise ValueError("only terminal complete runs can qualify")
    if manifest.code_identity.tree_state is not CodeTreeState.CLEAN:
        raise ValueError("qualified runs require a clean code tree")


__all__ = [
    "create_run_bundle",
    "load_run_bundle",
    "validate_run_qualification",
]

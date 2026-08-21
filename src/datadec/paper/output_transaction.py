from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path


def _write_temporary(destination: Path, content: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=suffix,
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
    return temporary_path


def _remove_file(path: Path) -> None:
    path.unlink()


def replace_output_set(
    outputs: Iterable[tuple[Path, bytes | str]],
    *,
    exact_directories: Mapping[Path, Iterable[Path]] | None = None,
) -> None:
    """Replace rendered files and roll back completed replacements after an error.

    Every content value must already be rendered and validated. Rollback covers errors
    raised during this process; it does not provide crash consistency or a single
    atomic view to concurrent readers. Each exact directory maps to its complete
    desired immediate-file inventory; other regular files are removed. If rollback
    itself fails, the recovery backup is preserved and both errors are raised.
    """
    rendered = tuple(
        (destination, content.encode() if isinstance(content, str) else content)
        for destination, content in outputs
    )
    destinations = tuple(destination for destination, _ in rendered)
    if len(set(destinations)) != len(destinations):
        raise ValueError("output destinations must be unique")

    directory_inventories = {
        directory: tuple(inventory)
        for directory, inventory in (exact_directories or {}).items()
    }
    stale: list[Path] = []
    for directory, inventory in directory_inventories.items():
        if len(set(inventory)) != len(inventory):
            raise ValueError("exact directory destinations must be unique")
        if any(path.parent != directory for path in inventory):
            raise ValueError(
                "exact directory destinations must be immediate directory children"
            )
        rendered_children = {
            destination
            for destination in destinations
            if destination.parent == directory
        }
        if set(inventory) != rendered_children:
            raise ValueError(
                "exact directory inventory must match rendered destinations"
            )
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError("exact output directory must be a non-symlink directory")
        if directory.exists():
            for path in sorted(directory.iterdir()):
                if path.is_symlink() or not path.is_file():
                    raise ValueError(
                        "exact output directories may contain only regular files"
                    )
                if path not in rendered_children:
                    stale.append(path)

    managed_paths = (*destinations, *stale)
    for path in managed_paths:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError("managed outputs must be non-symlink regular files")

    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path | None] = {}
    try:
        for destination, content in rendered:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged.append((_write_temporary(destination, content, ".tmp"), destination))

        for path in managed_paths:
            backups[path] = (
                _write_temporary(path, path.read_bytes(), ".backup")
                if path.exists()
                else None
            )

        completed: list[Path] = []
        try:
            for temporary_path, destination in staged:
                os.replace(temporary_path, destination)
                completed.append(destination)
            for path in stale:
                _remove_file(path)
                completed.append(path)
        except Exception as error:
            rollback_errors: list[Exception] = []
            for path in reversed(completed):
                backup_path = backups[path]
                try:
                    if backup_path is None:
                        path.unlink(missing_ok=True)
                    else:
                        os.replace(backup_path, path)
                        backups[path] = None
                except Exception as rollback_error:
                    if backup_path is not None:
                        backups[path] = None
                        rollback_error.add_note(
                            f"recovery backup preserved at {backup_path}"
                        )
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise ExceptionGroup(
                    "output replacement failed and rollback was incomplete",
                    [error, *rollback_errors],
                ) from error
            raise
    finally:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)
        for backup_path in backups.values():
            if backup_path is not None:
                backup_path.unlink(missing_ok=True)


__all__ = ["replace_output_set"]

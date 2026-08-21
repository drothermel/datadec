from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
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


def replace_output_set(outputs: Iterable[tuple[Path, bytes | str]]) -> None:
    """Replace rendered files and roll back completed replacements after an error.

    Every content value must already be rendered and validated. Rollback covers errors
    raised during this process; it does not provide crash consistency or a single
    atomic view to concurrent readers. If rollback itself fails, the recovery
    backup is preserved and the original and rollback errors are both raised.
    """
    rendered = tuple(
        (destination, content.encode() if isinstance(content, str) else content)
        for destination, content in outputs
    )
    destinations = tuple(destination for destination, _ in rendered)
    if len(set(destinations)) != len(destinations):
        raise ValueError("output destinations must be unique")

    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path | None] = {}
    try:
        for destination, content in rendered:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged.append((_write_temporary(destination, content, ".tmp"), destination))

        for destination in destinations:
            backups[destination] = (
                _write_temporary(destination, destination.read_bytes(), ".backup")
                if destination.exists()
                else None
            )

        replaced: list[Path] = []
        try:
            for temporary_path, destination in staged:
                os.replace(temporary_path, destination)
                replaced.append(destination)
        except Exception as error:
            rollback_errors: list[Exception] = []
            for destination in reversed(replaced):
                backup_path = backups[destination]
                try:
                    if backup_path is None:
                        destination.unlink(missing_ok=True)
                    else:
                        os.replace(backup_path, destination)
                        backups[destination] = None
                except Exception as rollback_error:
                    if backup_path is not None:
                        backups[destination] = None
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

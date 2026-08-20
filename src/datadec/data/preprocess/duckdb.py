from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import duckdb

type DuckDbLogicalType = Literal["string", "int64", "float64", "bool"]


@dataclass(frozen=True, slots=True)
class PendingParquetExport:
    output_path: Path
    temporary_path: Path
    row_count: int


def quote_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def duckdb_type(logical_type: DuckDbLogicalType) -> str:
    return {
        "string": "VARCHAR",
        "int64": "BIGINT",
        "float64": "DOUBLE",
        "bool": "BOOLEAN",
    }[logical_type]


def remove_owned_file(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        raise ValueError(f"expected an owned file but found a directory: {path}")
    path.unlink()


def prepare_parquet_export(
    connection: duckdb.DuckDBPyConnection,
    *,
    select_sql: str,
    output_path: Path,
    key_value_metadata: dict[str, str] | None = None,
) -> PendingParquetExport:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    remove_owned_file(temporary_path)
    metadata_option = ""
    if key_value_metadata:
        metadata = ", ".join(
            f"{quote_identifier(key)}: {sql_literal(value)}"
            for key, value in key_value_metadata.items()
        )
        metadata_option = f", KV_METADATA {{{metadata}}}"
    connection.execute(
        f"""
        COPY ({select_sql})
        TO {sql_literal(temporary_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD{metadata_option})
        """
    )
    count_row = connection.execute(
        "SELECT count(*) FROM read_parquet(?)", [str(temporary_path)]
    ).fetchone()
    assert count_row is not None
    return PendingParquetExport(
        output_path=output_path,
        temporary_path=temporary_path,
        row_count=count_row[0],
    )


def replace_parquet_exports(exports: tuple[PendingParquetExport, ...]) -> None:
    backup_paths = tuple(
        export.output_path.with_name(f".{export.output_path.name}.backup.tmp")
        for export in exports
    )
    if len({export.output_path for export in exports}) != len(exports):
        raise ValueError("parquet export output paths must be unique")

    had_output: list[bool] = []
    replaced_count = 0
    preserved_backups: set[Path] = set()
    try:
        for export, backup_path in zip(exports, backup_paths, strict=True):
            remove_owned_file(backup_path)
            output_exists = (
                export.output_path.exists() or export.output_path.is_symlink()
            )
            had_output.append(output_exists)
            if output_exists:
                os.link(export.output_path, backup_path)

        try:
            for export in exports:
                os.replace(export.temporary_path, export.output_path)
                replaced_count += 1
        except BaseException as replace_error:
            rollback_errors: list[str] = []
            for index in range(replaced_count - 1, -1, -1):
                export = exports[index]
                try:
                    if had_output[index]:
                        os.replace(backup_paths[index], export.output_path)
                    else:
                        remove_owned_file(export.output_path)
                except OSError as rollback_error:
                    if had_output[index]:
                        preserved_backups.add(backup_paths[index])
                    rollback_errors.append(f"{export.output_path}: {rollback_error}")
            if rollback_errors:
                details = "; ".join(rollback_errors)
                raise RuntimeError(
                    "failed to roll back parquet exports after replacement error; "
                    f"preserved recoverable backups where available: {details}"
                ) from replace_error
            raise
    finally:
        for backup_path in backup_paths:
            if backup_path not in preserved_backups:
                remove_owned_file(backup_path)

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Protocol, cast
from typing import Literal, Sequence
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from datadec.config import (
    DatasetSource,
    DetailSource,
    PublishedResultFile,
    PublishedResultsManifest,
    SourceManifest,
    load_published_results_manifest,
    load_source_manifest,
)
from datadec.data.paths import DataDecidePaths

_GOOGLE_DRIVE_DOWNLOAD_URL = (
    "https://drive.usercontent.google.com/download?id={file_id}"
    "&export=download&confirm=t"
)
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_TIMEOUT_SECONDS = 60
_CONTENT_RANGE_PATTERN = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


@dataclass(frozen=True, slots=True)
class DownloadResult:
    source: str
    destination: Path
    status: Literal["downloaded", "reused"]


class _ParquetDataset(Protocol):
    def to_parquet(self, path: Path) -> object: ...


def _is_valid_dataset_parquet(path: Path) -> bool:
    try:
        metadata = pq.ParquetFile(path).metadata
    except (OSError, pa.ArrowInvalid):
        return False
    return metadata.num_rows > 0 and metadata.num_columns > 0


def _download_dataset_source(
    paths: DataDecidePaths,
    source: DatasetSource,
    *,
    force: bool,
) -> DownloadResult:
    destination = paths.data_dir / source.output
    if destination.exists() and not force and _is_valid_dataset_parquet(destination):
        return DownloadResult(source.id, destination, "reused")

    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = paths.data_dir / "cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    load_kwargs: dict[str, object] = {
        "split": source.split,
        "revision": source.revision,
        "cache_dir": cache_dir,
    }
    if force:
        load_kwargs["download_mode"] = "force_redownload"
    dataset = cast(_ParquetDataset, load_dataset(source.repo_id, **load_kwargs))
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        dataset.to_parquet(temporary_path)
        if not _is_valid_dataset_parquet(temporary_path):
            raise ValueError(
                f"downloaded dataset is not a non-empty Parquet: {source.id}"
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return DownloadResult(source.id, destination, "downloaded")


def resolve_olmes_detail_recipes(
    requested: Sequence[str], source: DetailSource
) -> list[str]:
    allowed = set(source.recipes)
    unknown = [
        recipe for recipe in requested if recipe != "all" and recipe not in allowed
    ]
    if unknown:
        names = ", ".join(dict.fromkeys(unknown))
        raise ValueError(f"unknown OLMES detail recipe: {names}")
    if "all" in requested:
        return list(source.recipes)
    return list(dict.fromkeys(requested))


def _download_detail_source(
    paths: DataDecidePaths,
    source: DetailSource,
    recipe: str,
    *,
    force: bool,
) -> DownloadResult:
    filename = source.filename_template.format(recipe=recipe)
    output_root = paths.data_dir / source.output_root
    destination = output_root / filename
    result_source = f"{source.id}:{recipe}"
    if destination.exists() and not force:
        return DownloadResult(result_source, destination, "reused")

    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = paths.data_dir / "cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=source.repo_id,
        repo_type=source.repo_type,
        filename=filename,
        revision=source.revision,
        cache_dir=cache_dir,
        local_dir=output_root,
        force_download=force,
    )
    return DownloadResult(result_source, destination, "downloaded")


def _published_result_destination(
    paths: DataDecidePaths, source: PublishedResultFile
) -> Path:
    relative_path = PurePosixPath(source.path)
    if source.category == "scaling_law":
        return paths.data_dir / "raw" / "scaling-law" / relative_path.name
    return (
        paths.data_dir / "reference" / "published-results" / Path(*relative_path.parts)
    )


def _response_status(response: object) -> int:
    status = getattr(response, "status", None)
    if status is not None:
        return int(status)
    getcode = getattr(response, "getcode")
    return int(getcode())


def _download_published_result_file(
    paths: DataDecidePaths,
    source: PublishedResultFile,
    *,
    force: bool,
) -> DownloadResult:
    destination = _published_result_destination(paths, source)
    result_source = f"{source.category.replace('_', '-')}:{source.path}"
    if destination.exists() and not force:
        actual_size = destination.stat().st_size
        if actual_size != source.expected_size:
            raise ValueError(
                f"existing file has unexpected size for {source.path}: "
                f"{destination} has {actual_size} bytes, expected "
                f"{source.expected_size}"
            )
        return DownloadResult(result_source, destination, "reused")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    partial_size = 0 if force or not partial.exists() else partial.stat().st_size
    if partial_size > source.expected_size:
        raise ValueError(
            f"partial file has unexpected size for {source.path}: {partial} has "
            f"{partial_size} bytes, expected at most {source.expected_size}"
        )
    if partial_size == source.expected_size and not force:
        partial.replace(destination)
        return DownloadResult(result_source, destination, "downloaded")

    request = Request(
        _GOOGLE_DRIVE_DOWNLOAD_URL.format(file_id=source.id),
        headers={"Range": f"bytes={partial_size}-"} if partial_size else {},
    )
    try:
        with urlopen(  # noqa: S310 - pinned HTTPS endpoint
            request, timeout=_DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            status = _response_status(response)
            mode = "wb"
            if partial_size:
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = _CONTENT_RANGE_PATTERN.fullmatch(content_range)
                    if (
                        match is None
                        or int(match.group(1)) != partial_size
                        or int(match.group(2)) != source.expected_size - 1
                        or int(match.group(3)) != source.expected_size
                    ):
                        raise ValueError(
                            f"invalid Content-Range for resume: {content_range!r}"
                        )
                    mode = "ab"
                elif status != 200:
                    raise ValueError(f"unexpected HTTP status {status} while resuming")
            elif status != 200:
                raise ValueError(f"unexpected HTTP status {status}")

            with partial.open(mode) as file:
                while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                    file.write(chunk)

        actual_size = partial.stat().st_size
        if actual_size != source.expected_size:
            raise ValueError(
                f"downloaded {actual_size} bytes, expected {source.expected_size}"
            )
        partial.replace(destination)
    except Exception as exc:
        raise RuntimeError(
            f"failed to download {source.path} to {destination}"
        ) from exc

    return DownloadResult(result_source, destination, "downloaded")


def download_sources(
    paths: DataDecidePaths,
    *,
    ppl: bool = False,
    olmes: bool = False,
    olmes_details: Sequence[str] = (),
    scaling_law: bool = False,
    published_results: bool = False,
    published_figures: bool = False,
    force: bool = False,
    verbose: bool = False,
    manifest: SourceManifest | None = None,
    published_results_manifest: PublishedResultsManifest | None = None,
) -> list[DownloadResult]:
    if (
        not ppl
        and not olmes
        and not olmes_details
        and not scaling_law
        and not published_results
        and not published_figures
    ):
        raise ValueError("select at least one source to download")

    manifest = manifest or load_source_manifest()
    detail_recipes = resolve_olmes_detail_recipes(olmes_details, manifest.olmes_details)
    results: list[DownloadResult] = []

    def record(result: DownloadResult) -> None:
        results.append(result)
        if verbose:
            print(f"{result.source}: {result.status} -> {result.destination}")

    if ppl:
        record(_download_dataset_source(paths, manifest.ppl, force=force))
    if olmes:
        record(_download_dataset_source(paths, manifest.olmes, force=force))
    for recipe in detail_recipes:
        record(
            _download_detail_source(paths, manifest.olmes_details, recipe, force=force)
        )

    if scaling_law or published_results or published_figures:
        drive_manifest = published_results_manifest or load_published_results_manifest()
        categories = []
        if scaling_law:
            categories.append("scaling_law")
        if published_results:
            categories.append("published_results")
        if published_figures:
            categories.append("published_figures")
        for category in categories:
            for source in drive_manifest.files:
                if source.category == category:
                    record(_download_published_result_file(paths, source, force=force))

    return results

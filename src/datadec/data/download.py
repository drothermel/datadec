from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from datadec.config import DatasetSource, DetailSource, SourceManifest
from datadec.config import load_source_manifest
from datadec.data.paths import DataDecidePaths


@dataclass(frozen=True, slots=True)
class DownloadResult:
    source: str
    destination: Path
    status: Literal["downloaded", "reused"]


def _download_dataset_source(
    paths: DataDecidePaths,
    source: DatasetSource,
    *,
    force: bool,
) -> DownloadResult:
    destination = paths.data_dir / source.output
    if destination.exists() and not force:
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
    dataset = load_dataset(source.repo_id, **load_kwargs)
    dataset.to_parquet(destination)
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


def download_sources(
    paths: DataDecidePaths,
    *,
    ppl: bool = False,
    olmes: bool = False,
    olmes_details: Sequence[str] = (),
    force: bool = False,
    verbose: bool = False,
    manifest: SourceManifest | None = None,
) -> list[DownloadResult]:
    if not ppl and not olmes and not olmes_details:
        raise ValueError("select at least one source to download")

    manifest = manifest or load_source_manifest()
    detail_recipes = resolve_olmes_detail_recipes(olmes_details, manifest.olmes_details)
    results: list[DownloadResult] = []
    if ppl:
        results.append(_download_dataset_source(paths, manifest.ppl, force=force))
    if olmes:
        results.append(_download_dataset_source(paths, manifest.olmes, force=force))
    for recipe in detail_recipes:
        results.append(
            _download_detail_source(paths, manifest.olmes_details, recipe, force=force)
        )

    if verbose:
        for result in results:
            print(f"{result.source}: {result.status} -> {result.destination}")
    return results

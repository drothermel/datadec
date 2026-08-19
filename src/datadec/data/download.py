from __future__ import annotations

import argparse
from pathlib import Path

from dr_hf import download_dataset

from datadec.config import SourceManifest, load_source_manifest
from datadec.data.paths import DEFAULT_DATA_DIR, DataDecidePaths


def download_sources(
    paths: DataDecidePaths,
    *,
    profile: str = "processing",
    force_reload: bool = False,
    verbose: bool = False,
    manifest: SourceManifest | None = None,
) -> list[Path]:
    manifest = manifest or load_source_manifest()
    outputs: list[Path] = []
    for source in manifest.sources_for_profile(profile):
        if source.output is None:
            raise ValueError(f"source {source.id!r} does not define an output")
        output = paths.get_path(source.output)
        if verbose:
            print(f">> Downloading {source.id} from {source.repo_id}")
        download_dataset(
            path=output,
            repo_id=source.repo_id,
            split=source.split,
            force_reload=force_reload,
        )
        outputs.append(output)
        if verbose:
            print(f">> Wrote to {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the source datasets used by DataDecide processing."
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Root directory for downloaded and processed data.",
    )
    parser.add_argument(
        "--profile",
        default="processing",
        help="Source profile from configs/sources.toml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download sources even when their output files already exist.",
    )
    args = parser.parse_args()
    download_sources(
        DataDecidePaths(args.data_dir),
        profile=args.profile,
        force_reload=args.force,
        verbose=True,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from unittest.mock import call, patch

import pytest

from datadec.data.download import download_sources
from datadec.data.paths import DataDecidePaths


def test_download_sources_downloads_processing_profile(tmp_path: Path) -> None:
    paths = DataDecidePaths(str(tmp_path))

    with patch("datadec.data.download.download_dataset") as download_dataset:
        outputs = download_sources(paths, force_reload=True)

    assert outputs == [paths.get_path("ppl_raw"), paths.get_path("dwn_raw")]
    assert download_dataset.call_args_list == [
        call(
            path=paths.get_path("ppl_raw"),
            repo_id="allenai/DataDecide-ppl-results",
            split="train",
            force_reload=True,
        ),
        call(
            path=paths.get_path("dwn_raw"),
            repo_id="allenai/DataDecide-eval-results",
            split="train",
            force_reload=True,
        ),
    ]


def test_download_sources_rejects_unknown_profile(tmp_path: Path) -> None:
    paths = DataDecidePaths(str(tmp_path))

    with pytest.raises(ValueError, match="unknown source profile"):
        download_sources(paths, profile="missing")

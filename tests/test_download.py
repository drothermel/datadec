from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, call, patch
from urllib.request import Request

import pytest

from datadec.config import (
    PublishedResultFile,
    PublishedResultsManifest,
    load_source_manifest,
)
from datadec.data import download
from datadec.data.download import download_sources, resolve_olmes_detail_recipes
from datadec.data.paths import DataDecidePaths
from datadec.data.pipeline import DataPipeline

PUBLISHED_RESULTS_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1weYlEOlHrA_fzT2OsRa40uLc4EKTGz1D"
)


class FakeResponse:
    def __init__(
        self,
        *chunks: bytes,
        status: int = 200,
        headers: dict[str, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._chunks = list(chunks)
        self._error = error

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        if self._error is not None:
            error = self._error
            self._error = None
            raise error
        return b""


def published_file(
    *,
    path: str = "outputs2/example.csv",
    expected_size: int = 6,
    category: str = "published_results",
    file_id: str = "drive-file-id",
) -> PublishedResultFile:
    return PublishedResultFile.model_validate(
        {
            "id": file_id,
            "path": path,
            "expected_size": expected_size,
            "category": category,
        }
    )


def test_data_paths_use_exact_root_and_raw_outputs(tmp_path: Path) -> None:
    paths = DataDecidePaths(tmp_path)

    assert paths.data_dir == tmp_path
    assert paths.get_path("ppl_raw") == tmp_path / "raw/ppl.parquet"
    assert paths.get_path("dwn_raw") == tmp_path / "raw/olmes.parquet"
    assert paths.get_path("ppl_processed") == tmp_path / "processed/ppl.parquet"
    assert paths.get_path("olmes_processed") == tmp_path / "processed/olmes.parquet"
    assert paths.get_path("ppl_parsed") == tmp_path / "ppl_eval_parsed.parquet"
    assert paths.get_path("full_eval") == tmp_path / "full_eval.parquet"
    assert paths.dataset_path("4M") == tmp_path / "datasets/dataset_4M.pkl"


def test_download_sources_preserves_ppl_olmes_then_detail_order(
    tmp_path: Path,
) -> None:
    paths = DataDecidePaths(tmp_path)
    ppl_dataset = MagicMock()
    olmes_dataset = MagicMock()
    with (
        patch(
            "datadec.data.download.load_dataset",
            side_effect=[ppl_dataset, olmes_dataset],
        ) as load_dataset,
        patch("datadec.data.download.hf_hub_download") as hf_hub_download,
    ):
        results = download_sources(
            paths,
            ppl=True,
            olmes=True,
            olmes_details=["fineweb-pro", "c4", "fineweb-pro"],
        )

    cache_dir = tmp_path / "cache/huggingface"
    assert [result.source for result in results] == [
        "ppl",
        "olmes",
        "olmes-details:fineweb-pro",
        "olmes-details:c4",
    ]
    assert [result.status for result in results] == ["downloaded"] * 4
    assert load_dataset.call_args_list == [
        call(
            "allenai/DataDecide-ppl-results",
            split="train",
            revision="c4a9fa360a0c8351e71f3ede04dd165995fab68c",
            cache_dir=cache_dir,
        ),
        call(
            "allenai/DataDecide-eval-results",
            split="train",
            revision="9919b5a0e61e57a85021263918fa82d6ceaee038",
            cache_dir=cache_dir,
        ),
    ]
    ppl_dataset.to_parquet.assert_called_once_with(tmp_path / "raw/ppl.parquet")
    olmes_dataset.to_parquet.assert_called_once_with(tmp_path / "raw/olmes.parquet")
    assert hf_hub_download.call_args_list == [
        call(
            repo_id="allenai/DataDecide-eval-instances",
            repo_type="dataset",
            filename="models/fineweb-pro.tar.gz",
            revision="23f3b2e186ca6c39026e3efa00e4af397680c075",
            cache_dir=cache_dir,
            local_dir=tmp_path / "raw/olmes-details",
            force_download=False,
        ),
        call(
            repo_id="allenai/DataDecide-eval-instances",
            repo_type="dataset",
            filename="models/c4.tar.gz",
            revision="23f3b2e186ca6c39026e3efa00e4af397680c075",
            cache_dir=cache_dir,
            local_dir=tmp_path / "raw/olmes-details",
            force_download=False,
        ),
    ]


def test_existing_outputs_are_reused_without_network_calls(tmp_path: Path) -> None:
    paths = DataDecidePaths(tmp_path)
    outputs = [
        tmp_path / "raw/ppl.parquet",
        tmp_path / "raw/olmes.parquet",
        tmp_path / "raw/olmes-details/models/c4.tar.gz",
    ]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()

    with (
        patch("datadec.data.download.load_dataset") as load_dataset,
        patch("datadec.data.download.hf_hub_download") as hf_hub_download,
    ):
        results = download_sources(paths, ppl=True, olmes=True, olmes_details=["c4"])

    assert [result.destination for result in results] == outputs
    assert [result.status for result in results] == ["reused"] * 3
    load_dataset.assert_not_called()
    hf_hub_download.assert_not_called()


def test_force_propagates_to_dataset_and_detail_downloads(tmp_path: Path) -> None:
    paths = DataDecidePaths(tmp_path)
    dataset = MagicMock()
    with (
        patch("datadec.data.download.load_dataset", return_value=dataset) as load,
        patch("datadec.data.download.hf_hub_download") as hf_download,
    ):
        download_sources(paths, ppl=True, olmes_details=["c4"], force=True)

    load.assert_called_once_with(
        "allenai/DataDecide-ppl-results",
        split="train",
        revision="c4a9fa360a0c8351e71f3ede04dd165995fab68c",
        cache_dir=tmp_path / "cache/huggingface",
        download_mode="force_redownload",
    )
    hf_download.assert_called_once_with(
        repo_id="allenai/DataDecide-eval-instances",
        repo_type="dataset",
        filename="models/c4.tar.gz",
        revision="23f3b2e186ca6c39026e3efa00e4af397680c075",
        cache_dir=tmp_path / "cache/huggingface",
        local_dir=tmp_path / "raw/olmes-details",
        force_download=True,
    )


def test_all_detail_recipes_use_config_order() -> None:
    source = load_source_manifest().olmes_details

    assert resolve_olmes_detail_recipes(["all"], source) == list(source.recipes)
    assert resolve_olmes_detail_recipes(["c4", "all", "c4"], source) == list(
        source.recipes
    )


def test_unknown_detail_recipe_is_rejected_before_download(tmp_path: Path) -> None:
    with (
        patch("datadec.data.download.load_dataset") as load_dataset,
        patch("datadec.data.download.hf_hub_download") as hf_hub_download,
    ):
        with pytest.raises(ValueError, match="unknown OLMES detail recipe: missing"):
            download_sources(
                DataDecidePaths(tmp_path), ppl=True, olmes_details=["missing"]
            )

    load_dataset.assert_not_called()
    hf_hub_download.assert_not_called()


def test_download_sources_requires_an_explicit_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="select at least one source"):
        download_sources(DataDecidePaths(tmp_path))


def test_verbose_status_identifies_source_destination_and_reuse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "raw/ppl.parquet"
    output.parent.mkdir(parents=True)
    output.touch()

    download_sources(DataDecidePaths(tmp_path), ppl=True, verbose=True)

    assert capsys.readouterr().out == f"ppl: reused -> {output}\n"


def test_pipeline_explicitly_selects_ppl_and_olmes(tmp_path: Path) -> None:
    pipeline = DataPipeline(DataDecidePaths(tmp_path))

    with patch("datadec.data.pipeline.download_sources") as download_sources_mock:
        pipeline.download_raw_data(verbose=True)

    download_sources_mock.assert_called_once_with(
        pipeline.paths, ppl=True, olmes=True, verbose=True
    )


def test_download_implementation_does_not_use_snapshot_download() -> None:
    assert "snapshot_download" not in inspect.getsource(download)


def test_published_result_destinations_preserve_category_mapping(
    tmp_path: Path,
) -> None:
    paths = DataDecidePaths(tmp_path)
    raw = published_file(path="raw_data/results_ladder.csv", category="scaling_law")
    reference = published_file(path="per_task_out/arc/figure.pdf")

    assert download._published_result_destination(paths, raw) == (
        tmp_path / "raw/scaling-law/results_ladder.csv"
    )
    assert download._published_result_destination(paths, reference) == (
        tmp_path / "reference/published-results/per_task_out/arc/figure.pdf"
    )


def test_fresh_published_result_download_streams_and_atomically_completes(
    tmp_path: Path,
) -> None:
    source = published_file()
    response = FakeResponse(b"abc", b"def")
    with patch("datadec.data.download.urlopen", return_value=response) as open_url:
        result = download._download_published_result_file(
            DataDecidePaths(tmp_path), source, force=False
        )

    destination = tmp_path / "reference/published-results/outputs2/example.csv"
    request = open_url.call_args.args[0]
    assert isinstance(request, Request)
    assert request.full_url == (
        "https://drive.usercontent.google.com/download?id=drive-file-id"
        "&export=download&confirm=t"
    )
    assert open_url.call_args.kwargs == {"timeout": download._DOWNLOAD_TIMEOUT_SECONDS}
    assert request.get_header("Range") is None
    assert destination.read_bytes() == b"abcdef"
    assert not destination.with_name("example.csv.part").exists()
    assert result == download.DownloadResult(
        "published-results:outputs2/example.csv", destination, "downloaded"
    )


def test_published_result_download_resumes_valid_partial_with_range(
    tmp_path: Path,
) -> None:
    source = published_file()
    destination = tmp_path / "reference/published-results/outputs2/example.csv"
    partial = destination.with_name("example.csv.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"abc")
    response = FakeResponse(
        b"def",
        status=206,
        headers={"Content-Range": "bytes 3-5/6"},
    )

    with patch("datadec.data.download.urlopen", return_value=response) as open_url:
        download._download_published_result_file(
            DataDecidePaths(tmp_path), source, force=False
        )

    request = open_url.call_args.args[0]
    assert request.get_header("Range") == "bytes=3-"
    assert destination.read_bytes() == b"abcdef"
    assert not partial.exists()


def test_published_result_download_restarts_when_server_ignores_range(
    tmp_path: Path,
) -> None:
    source = published_file()
    destination = tmp_path / "reference/published-results/outputs2/example.csv"
    partial = destination.with_name("example.csv.part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"old")

    with patch(
        "datadec.data.download.urlopen",
        return_value=FakeResponse(b"abcdef", status=200),
    ):
        download._download_published_result_file(
            DataDecidePaths(tmp_path), source, force=False
        )

    assert destination.read_bytes() == b"abcdef"


def test_published_result_reuses_expected_size_and_force_redownloads(
    tmp_path: Path,
) -> None:
    source = published_file()
    destination = tmp_path / "reference/published-results/outputs2/example.csv"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stored")

    with patch("datadec.data.download.urlopen") as open_url:
        reused = download._download_published_result_file(
            DataDecidePaths(tmp_path), source, force=False
        )
    open_url.assert_not_called()
    assert reused.status == "reused"

    with patch(
        "datadec.data.download.urlopen",
        return_value=FakeResponse(b"forced"),
    ):
        forced = download._download_published_result_file(
            DataDecidePaths(tmp_path), source, force=True
        )
    assert forced.status == "downloaded"
    assert destination.read_bytes() == b"forced"


def test_published_result_rejects_mismatched_complete_file(
    tmp_path: Path,
) -> None:
    source = published_file()
    destination = tmp_path / "reference/published-results/outputs2/example.csv"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"wrong")

    with patch("datadec.data.download.urlopen") as open_url:
        with pytest.raises(ValueError, match="existing file has unexpected size"):
            download._download_published_result_file(
                DataDecidePaths(tmp_path), source, force=False
            )

    open_url.assert_not_called()
    assert destination.read_bytes() == b"wrong"


def test_published_result_size_mismatch_keeps_only_partial_file(
    tmp_path: Path,
) -> None:
    source = published_file()
    with patch("datadec.data.download.urlopen", return_value=FakeResponse(b"short")):
        with pytest.raises(RuntimeError, match="failed to download") as exc_info:
            download._download_published_result_file(
                DataDecidePaths(tmp_path), source, force=False
            )

    destination = tmp_path / "reference/published-results/outputs2/example.csv"
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert not destination.exists()
    assert destination.with_name("example.csv.part").read_bytes() == b"short"


def test_published_result_failure_preserves_cause_and_completed_destination(
    tmp_path: Path,
) -> None:
    source = published_file()
    destination = tmp_path / "reference/published-results/outputs2/example.csv"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"stored")
    network_error = OSError("connection lost")
    response = FakeResponse(b"new", error=network_error)

    with patch("datadec.data.download.urlopen", return_value=response):
        with pytest.raises(RuntimeError) as exc_info:
            download._download_published_result_file(
                DataDecidePaths(tmp_path), source, force=True
            )

    assert exc_info.value.__cause__ is network_error
    assert destination.read_bytes() == b"stored"
    assert destination.with_name("example.csv.part").read_bytes() == b"new"


def test_drive_selectors_are_disjoint_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    files = (
        published_file(
            path="raw_data/second.csv",
            expected_size=1,
            category="scaling_law",
            file_id="raw-2",
        ),
        published_file(
            path="outputs2/second.csv", expected_size=1, file_id="published-2"
        ),
        published_file(
            path="raw_data/first.csv",
            expected_size=1,
            category="scaling_law",
            file_id="raw-1",
        ),
        published_file(
            path="outputs2/first.csv", expected_size=1, file_id="published-1"
        ),
    )
    manifest = PublishedResultsManifest(
        folder_url=PUBLISHED_RESULTS_FOLDER_URL, files=files
    )
    for file in files:
        destination = download._published_result_destination(
            DataDecidePaths(tmp_path), file
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"x")

    scaling = download_sources(
        DataDecidePaths(tmp_path),
        scaling_law=True,
        published_results_manifest=manifest,
    )
    published = download_sources(
        DataDecidePaths(tmp_path),
        published_results=True,
        published_results_manifest=manifest,
    )
    combined = download_sources(
        DataDecidePaths(tmp_path),
        scaling_law=True,
        published_results=True,
        published_results_manifest=manifest,
    )

    assert [result.source for result in scaling] == [
        "scaling-law:raw_data/second.csv",
        "scaling-law:raw_data/first.csv",
    ]
    assert [result.source for result in published] == [
        "published-results:outputs2/second.csv",
        "published-results:outputs2/first.csv",
    ]
    assert [result.source for result in combined] == [
        *[result.source for result in scaling],
        *[result.source for result in published],
    ]
    assert len({result.destination for result in combined}) == 4

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from datadec.config import load_source_manifest
from datadec.data import download
from datadec.data.download import download_sources, resolve_olmes_detail_recipes
from datadec.data.paths import DataDecidePaths
from datadec.data.pipeline import DataPipeline


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

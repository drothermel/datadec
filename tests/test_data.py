from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from datadec import constants as consts
from datadec.data import (
    DataDecide,
    DataFrameLoader,
    get_data_recipe_details_df,
    get_data_recipe_family,
)


class TestGetDataRecipeFamily:
    def test_finds_family_for_known_recipe(self) -> None:
        for family, names in consts.DATA_RECIPE_FAMILIES.items():
            for name in names:
                result = get_data_recipe_family(name)
                assert result == family

    def test_unknown_recipe_returns_unknown(self) -> None:
        result = get_data_recipe_family("completely_unknown_recipe")
        assert result == "unknown"

    def test_custom_families_dict(self) -> None:
        custom_families = {"test_family": ["recipe_a", "recipe_b"]}
        result = get_data_recipe_family("recipe_a", custom_families)
        assert result == "test_family"


class TestGetDataRecipeDetailsDF:
    def test_returns_dataframe(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ds_details.csv"
        csv_path.write_text("dataset,col1,col2\ntest_data,1,2\n")
        result = get_data_recipe_details_df(csv_path)
        assert isinstance(result, pd.DataFrame)
        assert "data" in result.columns
        assert "dataset" not in result.columns

    def test_renames_dataset_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ds_details.csv"
        csv_path.write_text("dataset,value\ntest,1\n")
        result = get_data_recipe_details_df(csv_path)
        assert "data" in result.columns


class TestDataFrameLoader:
    def test_init_creates_empty_cache(self) -> None:
        loader = DataFrameLoader()
        assert loader.get_cache_size() == 0

    def test_set_name_caches_dataframe(self) -> None:
        loader = DataFrameLoader()
        df = pd.DataFrame({"a": [1, 2, 3]})
        loader.set_name("test_df", df)
        assert loader.is_cached("test_df")
        assert loader.get_cache_size() == 1

    def test_load_name_returns_cached_df(self) -> None:
        loader = DataFrameLoader()
        df = pd.DataFrame({"a": [1, 2, 3]})
        loader.set_name("test_df", df)
        result = loader.load_name("test_df")
        pd.testing.assert_frame_equal(result, df)

    def test_load_name_unknown_raises(self) -> None:
        loader = DataFrameLoader()
        with pytest.raises(ValueError):
            loader.load_name("unknown_df")

    def test_clear_cache_all(self) -> None:
        loader = DataFrameLoader()
        loader.set_name("df1", pd.DataFrame({"a": [1]}))
        loader.set_name("df2", pd.DataFrame({"b": [2]}))
        loader.clear_cache()
        assert loader.get_cache_size() == 0

    def test_clear_cache_specific(self) -> None:
        loader = DataFrameLoader()
        loader.set_name("df1", pd.DataFrame({"a": [1]}))
        loader.set_name("df2", pd.DataFrame({"b": [2]}))
        loader.clear_cache("df1")
        assert not loader.is_cached("df1")
        assert loader.is_cached("df2")

    def test_cached_dataframes_property(self) -> None:
        loader = DataFrameLoader()
        loader.set_name("df1", pd.DataFrame({"a": [1]}))
        loader.set_name("df2", pd.DataFrame({"b": [2]}))
        cached = loader.cached_dataframes
        assert "df1" in cached
        assert "df2" in cached

    def test_load_path_caches_result(self, tmp_path: Path) -> None:
        parquet_path = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_parquet(parquet_path)

        loader = DataFrameLoader()
        result = loader.load_path(parquet_path, "test_df")
        assert loader.is_cached("test_df")
        pd.testing.assert_frame_equal(result, df)


class TestDataDecideSelectParams:
    @pytest.fixture
    def mock_datadecide(self) -> DataDecide:
        with patch.object(DataDecide, "__init__", lambda self, *args, **kwargs: None):
            dd = DataDecide.__new__(DataDecide)
            dd.paths = MagicMock()
            dd.pipeline = MagicMock()
            dd.loader = DataFrameLoader()
            return dd

    def test_select_params_all(self, mock_datadecide: DataDecide) -> None:
        result = mock_datadecide.select_params("all")
        assert result == consts.ALL_MODEL_SIZE_STRS

    def test_select_params_single(self, mock_datadecide: DataDecide) -> None:
        result = mock_datadecide.select_params("4M")
        assert result == ["4M"]

    def test_select_params_list(self, mock_datadecide: DataDecide) -> None:
        result = mock_datadecide.select_params(["4M", "6M"])
        assert "4M" in result
        assert "6M" in result

    def test_select_params_with_exclude(self, mock_datadecide: DataDecide) -> None:
        result = mock_datadecide.select_params("all", exclude=["4M"])
        assert "4M" not in result


class TestDataDecideSelectData:
    @pytest.fixture
    def mock_datadecide(self) -> DataDecide:
        with patch.object(DataDecide, "__init__", lambda self, *args, **kwargs: None):
            dd = DataDecide.__new__(DataDecide)
            dd.paths = MagicMock()
            dd.pipeline = MagicMock()
            dd.loader = DataFrameLoader()
            return dd

    def test_select_data_all(self, mock_datadecide: DataDecide) -> None:
        result = mock_datadecide.select_data("all")
        assert result == consts.ALL_DATA_NAMES

    def test_select_data_single(self, mock_datadecide: DataDecide) -> None:
        if len(consts.ALL_DATA_NAMES) > 0:
            first_data = consts.ALL_DATA_NAMES[0]
            result = mock_datadecide.select_data(first_data)
            assert result == [first_data]


class TestDataDecideFilterDataQuality:
    @pytest.fixture
    def mock_datadecide(self) -> DataDecide:
        with patch.object(DataDecide, "__init__", lambda self, *args, **kwargs: None):
            dd = DataDecide.__new__(DataDecide)
            dd.paths = MagicMock()
            dd.pipeline = MagicMock()
            dd.loader = DataFrameLoader()
            return dd

    def test_empty_df_returns_empty(self, mock_datadecide: DataDecide) -> None:
        empty_df = pd.DataFrame(columns=["params", "step"])
        result = mock_datadecide.filter_data_quality(empty_df)
        assert len(result) == 0

    def test_max_steps_filter(self, mock_datadecide: DataDecide) -> None:
        df = pd.DataFrame(
            {
                "params": ["4M", "4M"],
                "step": [1000, 999999999],
            }
        )
        result = mock_datadecide.filter_data_quality(df, filter_types=["max_steps"])
        assert len(result) < len(df)


class TestDataDecideAggregateResults:
    @pytest.fixture
    def mock_datadecide(self) -> DataDecide:
        with patch.object(DataDecide, "__init__", lambda self, *args, **kwargs: None):
            dd = DataDecide.__new__(DataDecide)
            dd.paths = MagicMock()
            dd.pipeline = MagicMock()
            dd.loader = DataFrameLoader()
            return dd

    def test_empty_df_returns_copy(self, mock_datadecide: DataDecide) -> None:
        empty_df = pd.DataFrame()
        result = mock_datadecide.aggregate_results(empty_df)
        assert len(result) == 0

    def test_no_aggregation_returns_copy(self, mock_datadecide: DataDecide) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = mock_datadecide.aggregate_results(df, by_seeds=False)
        pd.testing.assert_frame_equal(result, df)

    def test_return_std_returns_tuple(self, mock_datadecide: DataDecide) -> None:
        df = pd.DataFrame(
            {
                "params": ["4M", "4M"],
                "data": ["C4", "C4"],
                "step": [100, 100],
                "tokens": [1000, 1000],
                "compute": [10, 10],
                "seed": [0, 1],
                "value": [1.0, 2.0],
            }
        )
        result = mock_datadecide.aggregate_results(df, return_std=True)
        assert isinstance(result, tuple)
        assert len(result) == 2

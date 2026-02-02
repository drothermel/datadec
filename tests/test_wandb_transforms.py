from __future__ import annotations

import pandas as pd
import pytest

from datadec.wandb_eval.wandb_transforms import (
    add_datadecide_columns,
    convert_objects_and_normalize_dtypes,
    drop_wandb_constant_ignored_cols,
    extract_dataset_from_model_path,
    extract_hyperparameters,
    filter_dpo_test_runs,
    get_created_time_key,
    infer_training_method,
    map_wandb_dataset_to_datadecide,
    map_wandb_model_size_to_datadecide,
)
from datadec.wandb_eval import wandb_constants as wconsts


class TestExtractDatasetFromModelPath:
    def test_valid_path(self) -> None:
        path = "ai2-llm/DataDecide-dolma17/snapshots/abc123"
        result = extract_dataset_from_model_path(path)
        assert result == "dolma17"

    def test_path_with_model_size(self) -> None:
        path = "ai2-llm/DataDecide-c4-60M/snapshots/abc123"
        result = extract_dataset_from_model_path(path)
        assert result == "c4"

    def test_nan_returns_none(self) -> None:
        result = extract_dataset_from_model_path(pd.NA)
        assert result is None

    def test_non_matching_path(self) -> None:
        result = extract_dataset_from_model_path("some/other/path")
        assert result is None


class TestMapWandbDatasetToDatadecide:
    def test_known_dataset(self) -> None:
        for wandb_name, dd_name in wconsts.WANDB_DATASET_TO_DATADECIDE_MAPPING.items():
            result = map_wandb_dataset_to_datadecide(wandb_name)
            assert result == dd_name

    def test_unknown_dataset(self) -> None:
        result = map_wandb_dataset_to_datadecide("unknown_dataset")
        assert result is None

    def test_empty_string(self) -> None:
        result = map_wandb_dataset_to_datadecide("")
        assert result is None


class TestMapWandbModelSizeToDatadecide:
    def test_nan_returns_none(self) -> None:
        result = map_wandb_model_size_to_datadecide(pd.NA)
        assert result is None

    def test_finds_closest_match(self) -> None:
        result = map_wandb_model_size_to_datadecide(4000000)
        assert result is not None
        assert isinstance(result, str)


class TestAddDatadecideColumns:
    def test_adds_required_columns(self) -> None:
        df = pd.DataFrame(
            {
                "model_name_or_path": ["ai2-llm/DataDecide-dolma17/snapshots/abc"],
                "model_size": [4000000],
            }
        )
        result = add_datadecide_columns(df)
        assert "wandb_dataset" in result.columns
        assert "data" in result.columns
        assert "params" in result.columns

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        with pytest.raises(AssertionError):
            add_datadecide_columns(df)


class TestExtractHyperparameters:
    def test_extracts_new_format_datetime(self) -> None:
        run_name = "241201-120000_exp_DD-test-60M_100Mtx1"
        result = extract_hyperparameters(run_name)
        assert "run_datetime_rnp" in result
        assert result["run_datetime_rnp"] == "2024-12-01 12:00:00"

    def test_extracts_old_format_datetime(self) -> None:
        run_name = "2024_12_01-12_00_00_exp_DD-test-60M"
        result = extract_hyperparameters(run_name)
        assert "run_datetime_rnp" in result
        assert result["run_datetime_rnp"] == "2024-12-01 12:00:00"

    def test_extracts_model_params(self) -> None:
        run_name = "241201-120000_exp_DD-test-60M_100Mtx1"
        result = extract_hyperparameters(run_name)
        assert "params_rnp" in result
        assert result["params_rnp"] == "60M"

    def test_extracts_mtx_tokens(self) -> None:
        run_name = "241201-120000_exp_DD-test-60M_100Mtx2"
        result = extract_hyperparameters(run_name)
        assert "total_tok_rnp" in result
        assert result["total_tok_rnp"] == 100
        assert "epochs_rnp" in result
        assert result["epochs_rnp"] == 2


class TestGetCreatedTimeKey:
    def test_finds_time_key(self) -> None:
        df = pd.DataFrame({wconsts.TIME_KEYS[0]: ["2024-01-01"]})
        result = get_created_time_key(df)
        assert result == wconsts.TIME_KEYS[0]

    def test_no_time_key_raises(self) -> None:
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        with pytest.raises(AssertionError):
            get_created_time_key(df)


class TestFilterDpoTestRuns:
    def test_filters_dpo_runs(self) -> None:
        df = pd.DataFrame(
            {
                "wandb_tags": ["dpo_tune_cache", "normal_run", None],
                "value": [1, 2, 3],
            }
        )
        result = filter_dpo_test_runs(df)
        assert len(result) == 2
        assert "dpo_tune_cache" not in result["wandb_tags"].values

    def test_no_wandb_tags_returns_unchanged(self) -> None:
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = filter_dpo_test_runs(df)
        assert len(result) == 3


class TestDropWandbConstantIgnoredCols:
    def test_drops_known_cols(self) -> None:
        cols_to_include = [col for col in wconsts.ALL_DROP_COLS[:2]]
        df = pd.DataFrame({col: [1, 2] for col in cols_to_include})
        df["keep_me"] = [1, 2]
        result = drop_wandb_constant_ignored_cols(df)
        assert "keep_me" in result.columns
        for col in cols_to_include:
            assert col not in result.columns

    def test_no_cols_to_drop(self) -> None:
        df = pd.DataFrame({"keep_me": [1, 2, 3]})
        result = drop_wandb_constant_ignored_cols(df)
        assert len(result.columns) == 1


class TestConvertObjectsAndNormalizeDtypes:
    def test_converts_int_objects(self) -> None:
        df = pd.DataFrame({"col": pd.array([1, 2, 3], dtype=object)})
        result = convert_objects_and_normalize_dtypes(df)
        assert result["col"].dtype in ["Int64", "int64"]

    def test_converts_string_objects(self) -> None:
        df = pd.DataFrame({"col": pd.array(["a", "b", "c"], dtype=object)})
        result = convert_objects_and_normalize_dtypes(df)
        assert result["col"].dtype == "string"

    def test_handles_mixed_types(self) -> None:
        df = pd.DataFrame({"col": [1, "two", 3.0]})
        result = convert_objects_and_normalize_dtypes(df)
        assert "col" in result.columns


class TestInferTrainingMethod:
    def test_infers_finetune_by_default(self) -> None:
        df = pd.DataFrame({col: [None, None] for col in wconsts.CORE_DPO_HPM_COLS})
        df["other"] = [1, 2]
        result = infer_training_method(df)
        assert "method" in result.columns
        assert all(result["method"] == "finetune")

    def test_infers_dpo_when_params_present(self) -> None:
        df = pd.DataFrame({col: [1.0, None] for col in wconsts.CORE_DPO_HPM_COLS})
        result = infer_training_method(df)
        assert result.loc[0, "method"] == "dpo"
        assert result.loc[1, "method"] == "finetune"

    def test_raises_if_method_exists(self) -> None:
        df = pd.DataFrame(
            {
                "method": ["existing"],
                **{col: [None] for col in wconsts.CORE_DPO_HPM_COLS},
            }
        )
        with pytest.raises(AssertionError):
            infer_training_method(df)

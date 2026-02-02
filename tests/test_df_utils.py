from __future__ import annotations

import pandas as pd
import pytest

from datadec import constants as consts
from datadec.df_utils import (
    create_mean_std_df,
    filter_by_max_step_to_use,
    filter_olmes_rows,
    filter_ppl_rows,
    melt_for_plotting,
    print_shape,
    select_by_data_param_combos,
)


class TestPrintShape:
    def test_print_shape_verbose(self, capsys) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        print_shape(df, "Test", verbose=True)
        captured = capsys.readouterr()
        assert "Test shape: 3 rows x 2 cols" in captured.out

    def test_print_shape_not_verbose(self, capsys) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        print_shape(df, "Test", verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestFilterByMaxStepToUse:
    def test_filters_rows_exceeding_max_step(self) -> None:
        df = pd.DataFrame(
            {
                "params": ["4M", "4M", "4M"],
                "step": [1000, 5000, 10000],
            }
        )
        result = filter_by_max_step_to_use(df)
        assert len(result) == 2
        assert result["step"].max() <= consts.MAX_STEP_TO_USE["4M"]

    def test_keeps_rows_at_or_below_max_step(self) -> None:
        max_step_4m = consts.MAX_STEP_TO_USE["4M"]
        df = pd.DataFrame(
            {
                "params": ["4M", "4M"],
                "step": [max_step_4m, max_step_4m - 100],
            }
        )
        result = filter_by_max_step_to_use(df)
        assert len(result) == 2

    def test_handles_multiple_param_sizes(self) -> None:
        df = pd.DataFrame(
            {
                "params": ["4M", "1B"],
                "step": [consts.MAX_STEP_TO_USE["4M"], consts.MAX_STEP_TO_USE["1B"]],
            }
        )
        result = filter_by_max_step_to_use(df)
        assert len(result) == 2


class TestFilterPplRows:
    def test_filters_rows_with_all_nan_ppl(self) -> None:
        df = pd.DataFrame(
            {
                "pile-valppl": [1.5, None, 2.0],
                "c4_en-valppl": [2.0, None, None],
            }
        )
        result = filter_ppl_rows(df)
        assert len(result) == 2

    def test_keeps_rows_with_any_ppl(self) -> None:
        df = pd.DataFrame(
            {
                "pile-valppl": [1.5, None],
                "c4_en-valppl": [None, 2.0],
            }
        )
        result = filter_ppl_rows(df)
        assert len(result) == 2

    def test_raises_when_no_ppl_columns(self) -> None:
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        with pytest.raises(AssertionError):
            filter_ppl_rows(df)


class TestFilterOlmesRows:
    def test_filters_rows_with_all_nan_olmes(self) -> None:
        df = pd.DataFrame(
            {
                "mmlu_average_acc_raw": [0.5, None, 0.6],
                "arc_challenge_acc_raw": [0.4, None, None],
            }
        )
        result = filter_olmes_rows(df)
        assert len(result) == 2

    def test_raises_when_no_olmes_columns(self) -> None:
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        with pytest.raises(AssertionError):
            filter_olmes_rows(df)


class TestSelectByDataParamCombos:
    def test_filter_by_data_list(self) -> None:
        df = pd.DataFrame(
            {
                "data": ["C4", "Falcon", "C4"],
                "params": ["4M", "4M", "6M"],
                "value": [1, 2, 3],
            }
        )
        result = select_by_data_param_combos(df, data=["C4"])
        assert len(result) == 2
        assert all(result["data"] == "C4")

    def test_filter_by_params_list(self) -> None:
        df = pd.DataFrame(
            {
                "data": ["C4", "C4", "C4"],
                "params": ["4M", "6M", "8M"],
                "value": [1, 2, 3],
            }
        )
        result = select_by_data_param_combos(df, params=["4M", "6M"])
        assert len(result) == 2

    def test_filter_by_specific_combos(self) -> None:
        df = pd.DataFrame(
            {
                "data": ["C4", "C4", "Falcon"],
                "params": ["4M", "6M", "4M"],
                "value": [1, 2, 3],
            }
        )
        combos = [("C4", "4M"), ("Falcon", "4M")]
        result = select_by_data_param_combos(df, data_param_combos=combos)
        assert len(result) == 2

    def test_returns_all_when_no_filters(self) -> None:
        df = pd.DataFrame(
            {
                "data": ["C4", "Falcon"],
                "params": ["4M", "6M"],
            }
        )
        result = select_by_data_param_combos(df)
        assert len(result) == 2


class TestCreateMeanStdDf:
    def test_computes_mean_and_std(self) -> None:
        df = pd.DataFrame(
            {
                "params": ["4M", "4M", "4M"],
                "data": ["C4", "C4", "C4"],
                "step": [100, 100, 100],
                "tokens": [1000, 1000, 1000],
                "compute": [10, 10, 10],
                "seed": [0, 1, 2],
                "metric_value": [1.0, 2.0, 3.0],
            }
        )
        mean_df, std_df = create_mean_std_df(df)
        assert len(mean_df) == 1
        assert mean_df["metric_value"].iloc[0] == 2.0
        assert std_df["metric_value"].iloc[0] == 1.0

    def test_groups_by_mean_id_columns(self) -> None:
        df = pd.DataFrame(
            {
                "params": ["4M", "4M", "6M", "6M"],
                "data": ["C4", "C4", "C4", "C4"],
                "step": [100, 100, 100, 100],
                "tokens": [1000, 1000, 2000, 2000],
                "compute": [10, 10, 20, 20],
                "seed": [0, 1, 0, 1],
                "value": [1.0, 3.0, 5.0, 7.0],
            }
        )
        mean_df, _ = create_mean_std_df(df)
        assert len(mean_df) == 2


class TestMeltForPlotting:
    def test_melts_metrics(self) -> None:
        df = pd.DataFrame(
            {
                "params": ["4M"],
                "data": ["C4"],
                "seed": [0],
                "step": [100],
                "tokens": [1000],
                "compute": [10],
                "pile-valppl": [1.5],
                "c4_en-valppl": [2.0],
            }
        )
        result = melt_for_plotting(df, metrics=["pile-valppl", "c4_en-valppl"])
        assert "metric" in result.columns
        assert "value" in result.columns
        assert len(result) == 2

    def test_drops_na_by_default(self) -> None:
        df = pd.DataFrame(
            {
                "params": ["4M"],
                "data": ["C4"],
                "seed": [0],
                "step": [100],
                "tokens": [1000],
                "compute": [10],
                "pile-valppl": [None],
            }
        )
        result = melt_for_plotting(df, metrics=["pile-valppl"], drop_na=True)
        assert len(result) == 0

    def test_keeps_na_when_disabled(self) -> None:
        df = pd.DataFrame(
            {
                "params": ["4M"],
                "data": ["C4"],
                "seed": [0],
                "step": [100],
                "tokens": [1000],
                "compute": [10],
                "pile-valppl": [None],
            }
        )
        result = melt_for_plotting(df, metrics=["pile-valppl"], drop_na=False)
        assert len(result) == 1

    def test_excludes_seed_when_include_seeds_false(self) -> None:
        df = pd.DataFrame(
            {
                "params": ["4M"],
                "data": ["C4"],
                "seed": [0],
                "step": [100],
                "tokens": [1000],
                "compute": [10],
                "pile-valppl": [1.5],
            }
        )
        result = melt_for_plotting(df, metrics=["pile-valppl"], include_seeds=False)
        assert "seed" not in result.columns

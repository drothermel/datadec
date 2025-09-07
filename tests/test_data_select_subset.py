import pandas as pd
import pytest


def test_select_subset_data_selection(dd_instance, sample_real_data):
    """Test data type selection."""
    result = dd_instance.select_subset(sample_real_data, data="C4")

    assert isinstance(result, pd.DataFrame)
    assert len(sample_real_data) == 200  # Input: 200 rows
    assert len(result) == 42  # C4 only (from 200-row subset)
    assert (result["data"] == "C4").all()


def test_select_subset_data_list_selection(dd_instance, sample_real_data):
    """Test multiple data type selection."""
    result = dd_instance.select_subset(sample_real_data, data=["C4", "DCLM-Baseline"])

    assert isinstance(result, pd.DataFrame)
    assert len(result) > len(sample_real_data[sample_real_data["data"] == "C4"])
    assert set(result["data"].unique()).issubset({"C4", "DCLM-Baseline"})


def test_select_subset_params_selection(dd_instance, sample_real_data):
    """Test parameter selection (all our test data is 10M)."""
    result = dd_instance.select_subset(sample_real_data, params="10M")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_real_data)  # All data is 10M
    assert (result["params"] == "10M").all()


def test_select_subset_min_params_threshold(dd_instance, sample_real_data):
    """Test min_params threshold filtering."""
    result = dd_instance.select_subset(sample_real_data, min_params="10M")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_real_data)  # All data meets 10M threshold
    assert (result["params"] == "10M").all()


def test_select_subset_max_params_threshold(dd_instance, sample_real_data):
    """Test max_params threshold filtering."""
    result = dd_instance.select_subset(sample_real_data, max_params="10M")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_real_data)  # All data is at most 10M
    assert (result["params"] == "10M").all()


def test_select_subset_seed_selection_single(dd_instance, sample_real_data):
    """Test single seed selection."""
    result = dd_instance.select_subset(sample_real_data, seeds=0)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 69  # Seed 0 only (from 200-row subset)
    assert (result["seed"] == 0).all()


def test_select_subset_seed_selection_multiple(dd_instance, sample_real_data):
    """Test multiple seed selection."""
    result = dd_instance.select_subset(sample_real_data, seeds=[0, 1])

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 138  # Seeds 0,1 (from 200-row subset)
    assert set(result["seed"].unique()) == {0, 1}


def test_select_subset_step_limits(dd_instance, sample_real_data):
    """Test step range filtering."""
    result = dd_instance.select_subset(sample_real_data, step_lims=(1000, 5000))

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 59  # Steps 1000-5000 (from 200-row subset)
    assert result["step"].min() >= 1000
    assert result["step"].max() <= 5000


def test_select_subset_step_limits_min_only(dd_instance, sample_real_data):
    """Test step range filtering with min only."""
    result = dd_instance.select_subset(sample_real_data, step_lims=(10000, None))

    assert isinstance(result, pd.DataFrame)
    assert result["step"].min() >= 10000


def test_select_subset_step_limits_max_only(dd_instance, sample_real_data):
    """Test step range filtering with max only."""
    result = dd_instance.select_subset(sample_real_data, step_lims=(None, 5000))

    assert isinstance(result, pd.DataFrame)
    assert result["step"].max() <= 5000


def test_select_subset_token_limits(dd_instance, sample_real_data):
    """Test token range filtering."""
    result = dd_instance.select_subset(
        sample_real_data, token_lims=(80000000, 170000000)
    )

    assert isinstance(result, pd.DataFrame)
    assert result["tokens"].min() >= 80000000
    assert result["tokens"].max() <= 170000000


def test_select_subset_token_limits_min_only(dd_instance, sample_real_data):
    """Test token range filtering with min only."""
    result = dd_instance.select_subset(sample_real_data, token_lims=(100000, None))

    assert isinstance(result, pd.DataFrame)
    assert result["tokens"].min() >= 100000


def test_select_subset_ppl_metric_type(dd_instance, sample_real_data):
    """Test PPL metric type selection."""
    result = dd_instance.select_subset(sample_real_data, metric_type="ppl")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_real_data)
    expected_cols = 16  # 5 ID columns + ~11 PPL metrics
    assert result.shape[1] == expected_cols
    assert "params" in result.columns
    assert "pile-valppl" in result.columns


def test_select_subset_olmes_metric_type(dd_instance, sample_real_data):
    """Test OLMES metric type selection."""
    result = dd_instance.select_subset(sample_real_data, metric_type="olmes")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_real_data)
    expected_cols = 275  # 5 ID columns + 270 OLMES metrics
    assert result.shape[1] == expected_cols
    assert "params" in result.columns
    assert "arc_challenge_acc_raw" in result.columns


def test_select_subset_specific_metrics(dd_instance, sample_real_data):
    """Test specific metric selection."""
    result = dd_instance.select_subset(
        sample_real_data, metrics=["pile-valppl", "arc_challenge_acc_raw"]
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_real_data)
    expected_cols = 7  # 5 ID columns + 2 specific metrics
    assert result.shape[1] == expected_cols
    assert "pile-valppl" in result.columns
    assert "arc_challenge_acc_raw" in result.columns


def test_select_subset_specific_columns(dd_instance, sample_real_data):
    """Test specific column selection."""
    result = dd_instance.select_subset(sample_real_data, columns=["compute", "lr_max"])

    assert isinstance(result, pd.DataFrame)
    expected_cols = 7  # 5 ID columns + 2 specific columns
    assert result.shape[1] == expected_cols
    assert "compute" in result.columns
    assert "lr_max" in result.columns


def test_select_subset_no_id_columns(dd_instance, sample_real_data):
    """Test excluding ID columns."""
    result = dd_instance.select_subset(
        sample_real_data, metrics=["pile-valppl"], include_id_columns=False
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 1  # Only the metric
    assert "pile-valppl" in result.columns
    assert "params" not in result.columns


def test_select_subset_combined_selection(dd_instance, sample_real_data):
    """Test combining multiple selection criteria."""
    result = dd_instance.select_subset(
        sample_real_data,
        data="C4",
        seeds=[0, 1],
        step_lims=(1000, 10000),
        metric_type="ppl",
    )

    assert isinstance(result, pd.DataFrame)
    assert (result["data"] == "C4").all()
    assert set(result["seed"].unique()).issubset({0, 1})
    assert result["step"].min() >= 1000
    assert result["step"].max() <= 10000
    assert result.shape[1] == 16  # ID + PPL columns


def test_select_subset_verbose_output(dd_instance, sample_real_data, capsys):
    """Test verbose output for subset selection operations."""
    dd_instance.select_subset(sample_real_data, data="C4", seeds=0, verbose=True)

    captured = capsys.readouterr()
    assert "Initial subset selection shape:" in captured.out
    assert "After data/param selection shape:" in captured.out
    assert "Seeds [0] shape:" in captured.out


def test_select_subset_invalid_metric_type(dd_instance, sample_real_data):
    """Test error handling for invalid metric type."""
    with pytest.raises(AssertionError) as exc_info:
        dd_instance.select_subset(sample_real_data, metric_type="invalid")

    assert "Unknown metric_type 'invalid'" in str(exc_info.value)
    assert "Available: ['ppl', 'olmes']" in str(exc_info.value)


def test_select_subset_invalid_metric(dd_instance, sample_real_data):
    """Test error handling for invalid metric name."""
    with pytest.raises(AssertionError) as exc_info:
        dd_instance.select_subset(sample_real_data, metrics=["invalid_metric"])

    assert "Unknown metrics: ['invalid_metric']" in str(exc_info.value)
    assert "Available:" in str(exc_info.value)


def test_select_subset_empty_dataframe(dd_instance):
    """Test behavior with empty DataFrame."""
    empty_df = pd.DataFrame()
    result = dd_instance.select_subset(empty_df, data="C4")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_select_subset_data_param_combos(dd_instance, sample_real_data):
    """Test data_param_combos selection."""
    result = dd_instance.select_subset(
        sample_real_data, data_param_combos=[("C4", "10M"), ("DCLM-Baseline", "10M")]
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result["data"].unique()).issubset({"C4", "DCLM-Baseline"})
    assert (result["params"] == "10M").all()


def test_select_subset_no_column_selection_returns_all(dd_instance, sample_real_data):
    """Test that no column selection returns all columns."""
    result = dd_instance.select_subset(sample_real_data, data="C4")

    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == sample_real_data.shape[1]  # All columns preserved


def test_select_subset_preserves_original_dataframe(dd_instance, sample_real_data):
    """Test that original DataFrame is not modified."""
    original_shape = sample_real_data.shape
    result = dd_instance.select_subset(sample_real_data, data="C4", metric_type="ppl")

    # Original should be unchanged
    assert sample_real_data.shape == original_shape
    # Result should be different
    assert result.shape != original_shape

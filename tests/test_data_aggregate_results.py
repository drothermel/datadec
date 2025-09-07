import pandas as pd


def test_aggregate_results_by_seeds_mean_only(dd_instance, sample_real_data):
    """Test basic aggregation by seeds returning only means."""
    result = dd_instance.aggregate_results(
        sample_real_data, by_seeds=True, return_std=False
    )

    assert isinstance(result, pd.DataFrame)
    assert len(sample_real_data) == 200  # Input: 200 rows
    assert len(result) == 70  # Output: exactly 70 aggregated groups
    assert "seed" not in result.columns  # Seed column removed
    assert all(
        col in result.columns for col in ["params", "data", "step"]
    )  # Grouping columns preserved

    # Verify all params are 10M (from our fixed dataset)
    assert (result["params"] == "10M").all()

    # Verify we have expected data types
    expected_data_types = {
        "C4",
        "DCLM-Baseline",
        "DCLM-Baseline (QC 10%)",
        "DCLM-Baseline (QC 20%)",
        "DCLM-Baseline (QC 7%, FW2)",
    }
    assert set(result["data"].unique()) == expected_data_types


def test_aggregate_results_by_seeds_with_std(dd_instance, sample_real_data):
    """Test aggregation by seeds returning both means and stds."""
    result = dd_instance.aggregate_results(
        sample_real_data, by_seeds=True, return_std=True
    )

    assert isinstance(result, tuple)
    assert len(result) == 2

    mean_df, std_df = result
    assert isinstance(mean_df, pd.DataFrame)
    assert isinstance(std_df, pd.DataFrame)
    assert len(mean_df) == len(std_df) == 70  # Both should have exactly 70 rows
    assert mean_df.columns.tolist() == std_df.columns.tolist()
    assert "seed" not in mean_df.columns
    assert "seed" not in std_df.columns

    # Both should have same grouping structure
    assert (mean_df["params"] == "10M").all()
    assert (std_df["params"] == "10M").all()


def test_aggregate_results_no_aggregation(dd_instance, sample_real_data):
    """Test when by_seeds=False, returns original DataFrame unchanged."""
    result = dd_instance.aggregate_results(sample_real_data, by_seeds=False)

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, sample_real_data)


def test_aggregate_results_verbose_output(dd_instance, sample_real_data, capsys):
    """Test verbose output prints shape information."""
    dd_instance.aggregate_results(sample_real_data, by_seeds=True, verbose=True)

    captured = capsys.readouterr()
    assert "Before aggregation shape:" in captured.out
    assert "After aggregation (means) shape:" in captured.out
    assert "200 rows" in captured.out  # Original data
    assert "70 rows" in captured.out  # Aggregated data


def test_aggregate_results_verbose_output_with_std(
    dd_instance, sample_real_data, capsys
):
    """Test verbose output includes std information when return_std=True."""
    dd_instance.aggregate_results(
        sample_real_data, by_seeds=True, return_std=True, verbose=True
    )

    captured = capsys.readouterr()
    assert "Before aggregation shape:" in captured.out
    assert "After aggregation (means) shape:" in captured.out
    assert "After aggregation (stds) shape:" in captured.out


def test_aggregate_results_verbose_no_aggregation(
    dd_instance, sample_real_data, capsys
):
    """Test verbose output when no aggregation is performed."""
    dd_instance.aggregate_results(sample_real_data, by_seeds=False, verbose=True)

    captured = capsys.readouterr()
    assert "No aggregation shape:" in captured.out
    assert "Before aggregation" not in captured.out


def test_aggregate_results_empty_dataframe(dd_instance):
    """Test behavior with empty DataFrame."""
    empty_df = pd.DataFrame()
    result = dd_instance.aggregate_results(empty_df, by_seeds=True)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_aggregate_results_with_real_seed_groups(dd_instance, sample_real_data):
    """Test aggregation with real data having multiple seeds per group."""
    result = dd_instance.aggregate_results(sample_real_data, by_seeds=True)

    # Verify a specific group that we know has multiple seeds
    # From analysis: C4 data at step 0.0 has seeds [0, 1, 2]
    c4_step_0 = result[(result["data"] == "C4") & (result["step"] == 0.0)]
    assert len(c4_step_0) == 1  # Should be aggregated to one row

    # Verify the original had multiple seeds for this combination
    original_c4_step_0 = sample_real_data[
        (sample_real_data["data"] == "C4") & (sample_real_data["step"] == 0.0)
    ]
    assert len(original_c4_step_0) == 3  # Should have 3 seeds: 0, 1, 2
    assert set(original_c4_step_0["seed"].unique()) == {0, 1, 2}

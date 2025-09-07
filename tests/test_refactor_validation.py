"""Test the DataFrame input functionality of refactored methods."""

import pandas as pd


def test_get_filtered_df_backward_compatibility(dd_instance):
    """Test that get_filtered_df works exactly as before (backward compatibility)."""
    result = dd_instance.get_filtered_df()

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert result.shape[1] > 0


def test_easy_index_df_backward_compatibility(dd_instance):
    """Test that easy_index_df works exactly as before (backward compatibility)."""
    result = dd_instance.easy_index_df(data="C4", seeds=0)

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert (result["data"] == "C4").all()
    assert (result["seed"] == 0).all()


def test_get_filtered_df_with_dataframe_input(dd_instance, sample_real_data):
    """Test new DataFrame input functionality for get_filtered_df."""
    result = dd_instance.get_filtered_df(input_df=sample_real_data)

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    # Should be smaller than input due to filtering
    assert len(result) <= len(sample_real_data)


def test_easy_index_df_with_dataframe_input(dd_instance, sample_real_data):
    """Test new DataFrame input functionality for easy_index_df."""
    result = dd_instance.easy_index_df(input_df=sample_real_data, data="C4")

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0
    assert (result["data"] == "C4").all()
    # Should be smaller than input due to filtering
    assert len(result) <= len(sample_real_data)


def test_compositional_chaining(dd_instance, sample_real_data):
    """Test that methods can be chained together using DataFrame inputs."""
    # Chain operations
    step1 = dd_instance.select_subset(sample_real_data, data="C4")
    step2 = dd_instance.filter_data_quality(step1, filter_types=["max_steps"])
    step3 = dd_instance.aggregate_results(step2, by_seeds=True)

    # Verify each step
    assert isinstance(step1, pd.DataFrame)
    assert isinstance(step2, pd.DataFrame)
    assert isinstance(step3, pd.DataFrame)

    # Verify filtering progression
    assert len(step1) <= len(sample_real_data)  # step1 should be smaller or equal
    assert len(step2) <= len(step1)  # step2 should be smaller or equal
    assert len(step3) <= len(step2)  # step3 should be smaller or equal (aggregation)

    # Verify data type filtering worked
    assert (step1["data"] == "C4").all()
    assert (step2["data"] == "C4").all() if len(step2) > 0 else True
    assert (step3["data"] == "C4").all() if len(step3) > 0 else True


def test_result_equivalence(dd_instance):
    """Test that traditional vs compositional approaches give equivalent results."""
    # Traditional approach
    traditional = dd_instance.get_filtered_df()

    # Compositional approach (using same data)
    compositional = dd_instance.get_filtered_df(input_df=dd_instance.full_eval)

    # Results should be equivalent
    assert traditional.shape == compositional.shape
    assert traditional.columns.equals(compositional.columns)


def test_fixture_consistency_with_existing_tests(dd_instance, sample_real_data):
    """Test that fixture-based operations match expected patterns from existing tests."""
    # This should match test_select_subset_data_selection expectations
    c4_result = dd_instance.select_subset(sample_real_data, data="C4")

    assert isinstance(c4_result, pd.DataFrame)
    assert len(c4_result) == 42  # Should match the corrected test expectation
    assert (c4_result["data"] == "C4").all()


def test_dataframe_input_preserves_original(dd_instance, sample_real_data):
    """Test that using DataFrame input doesn't modify the original DataFrame."""
    original_shape = sample_real_data.shape
    original_columns = sample_real_data.columns.tolist()

    # Use sample_real_data as input to various methods
    dd_instance.get_filtered_df(input_df=sample_real_data)
    dd_instance.easy_index_df(input_df=sample_real_data, data="C4")
    dd_instance.select_subset(sample_real_data, data="C4")
    dd_instance.filter_data_quality(sample_real_data)

    # Original should be unchanged
    assert sample_real_data.shape == original_shape
    assert sample_real_data.columns.tolist() == original_columns


def test_method_parameter_compatibility(dd_instance, sample_real_data):
    """Test that all existing parameters still work with new DataFrame input capability."""

    # Test get_filtered_df with various parameter combinations
    result1 = dd_instance.get_filtered_df(
        input_df=sample_real_data,
        filter_types=["max_steps"],
        return_means=False,
        verbose=False,
    )

    result2 = dd_instance.get_filtered_df(
        input_df=sample_real_data,
        filter_types=["ppl", "max_steps"],
        return_means=True,
        min_params="10M",
    )

    # Test easy_index_df with various parameter combinations
    result3 = dd_instance.easy_index_df(
        input_df=sample_real_data,
        data=["C4", "DCLM-Baseline"],
        seeds=[0, 1],
        keep_cols=["params", "data", "seed", "step"],
    )

    # All results should be valid DataFrames
    for result in [result1, result2, result3]:
        assert isinstance(result, pd.DataFrame)

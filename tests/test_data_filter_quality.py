import pandas as pd
import pytest


def test_filter_data_quality_max_steps_only(dd_instance, sample_real_data):
    """Test max_steps filter (should not remove any rows from our test data)."""
    result = dd_instance.filter_data_quality(
        sample_real_data, filter_types=["max_steps"]
    )

    assert isinstance(result, pd.DataFrame)
    assert len(sample_real_data) == 200  # Input: 200 rows
    assert len(result) == 200  # Output: 200 rows (no filtering for our test data)
    assert result.columns.tolist() == sample_real_data.columns.tolist()


def test_filter_data_quality_ppl_only(dd_instance, sample_real_data):
    """Test ppl filter removes rows with all NaN perplexity values."""
    result = dd_instance.filter_data_quality(sample_real_data, filter_types=["ppl"])

    assert isinstance(result, pd.DataFrame)
    assert len(sample_real_data) == 200  # Input: 200 rows
    assert len(result) == 162  # Output: 162 rows (38 rows filtered)
    assert result.columns.tolist() == sample_real_data.columns.tolist()


def test_filter_data_quality_olmes_only(dd_instance, sample_real_data):
    """Test olmes filter removes rows with all NaN OLMES metric values."""
    result = dd_instance.filter_data_quality(sample_real_data, filter_types=["olmes"])

    assert isinstance(result, pd.DataFrame)
    assert len(sample_real_data) == 200  # Input: 200 rows
    assert len(result) == 173  # Output: 173 rows (27 rows filtered)
    assert result.columns.tolist() == sample_real_data.columns.tolist()


def test_filter_data_quality_combined_filters(dd_instance, sample_real_data):
    """Test combination of all three filters."""
    result = dd_instance.filter_data_quality(
        sample_real_data, filter_types=["max_steps", "ppl", "olmes"]
    )

    assert isinstance(result, pd.DataFrame)
    assert len(sample_real_data) == 200  # Input: 200 rows
    assert len(result) == 135  # Output: 135 rows (65 total rows filtered)
    assert result.columns.tolist() == sample_real_data.columns.tolist()


def test_filter_data_quality_ppl_olmes_combo(dd_instance, sample_real_data):
    """Test ppl + olmes filters (without max_steps)."""
    result = dd_instance.filter_data_quality(
        sample_real_data, filter_types=["ppl", "olmes"]
    )

    assert isinstance(result, pd.DataFrame)
    assert (
        len(result) == 135
    )  # Same result as with max_steps since max_steps doesn't filter


def test_filter_data_quality_max_steps_ppl_combo(dd_instance, sample_real_data):
    """Test max_steps + ppl filters."""
    result = dd_instance.filter_data_quality(
        sample_real_data, filter_types=["max_steps", "ppl"]
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 162  # Only ppl filtering is effective


def test_filter_data_quality_empty_filter_types(dd_instance, sample_real_data):
    """Test with empty filter_types list (should return unchanged data)."""
    result = dd_instance.filter_data_quality(sample_real_data, filter_types=[])

    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, sample_real_data)


def test_filter_data_quality_invalid_filter_type(dd_instance, sample_real_data):
    """Test error handling for invalid filter type."""
    with pytest.raises(AssertionError) as exc_info:
        dd_instance.filter_data_quality(
            sample_real_data, filter_types=["invalid_filter"]
        )

    assert "Invalid filter types: ['invalid_filter']" in str(exc_info.value)
    assert "Available: ['max_steps', 'ppl', 'olmes']" in str(exc_info.value)


def test_filter_data_quality_mixed_valid_invalid_filters(dd_instance, sample_real_data):
    """Test with mix of valid and invalid filter types."""
    with pytest.raises(AssertionError) as exc_info:
        dd_instance.filter_data_quality(
            sample_real_data, filter_types=["max_steps", "invalid", "ppl"]
        )

    assert "Invalid filter types: ['max_steps', 'invalid', 'ppl']" in str(
        exc_info.value
    )


def test_filter_data_quality_verbose_output(dd_instance, sample_real_data, capsys):
    """Test verbose output for filtering operations."""
    dd_instance.filter_data_quality(
        sample_real_data, filter_types=["ppl", "olmes"], verbose=True
    )

    captured = capsys.readouterr()
    assert "Initial shape:" in captured.out
    assert "Non-NaN perplexity shape:" in captured.out
    assert "Non-NaN OLMES shape:" in captured.out
    assert "200 rows" in captured.out  # Initial data
    assert "162 rows" in captured.out  # After ppl filter
    assert "135 rows" in captured.out  # After olmes filter


def test_filter_data_quality_verbose_no_filtering(
    dd_instance, sample_real_data, capsys
):
    """Test verbose output when no filtering occurs."""
    dd_instance.filter_data_quality(
        sample_real_data, filter_types=["max_steps"], verbose=True
    )

    captured = capsys.readouterr()
    assert "Initial shape:" in captured.out
    assert "LEQ to max step shape:" in captured.out
    assert "200 rows" in captured.out  # Same count before and after


def test_filter_data_quality_empty_dataframe(dd_instance):
    """Test behavior with empty DataFrame."""
    empty_df = pd.DataFrame()
    result = dd_instance.filter_data_quality(empty_df, filter_types=["max_steps"])

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_filter_data_quality_dataframe_copy_behavior(dd_instance, sample_real_data):
    """Test that original DataFrame is not modified."""
    original_shape = sample_real_data.shape
    result = dd_instance.filter_data_quality(
        sample_real_data, filter_types=["ppl", "olmes"]
    )

    # Original should be unchanged
    assert sample_real_data.shape == original_shape
    # Result should be different
    assert result.shape != original_shape
    assert len(result) < len(sample_real_data)

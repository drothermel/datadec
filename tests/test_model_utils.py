from __future__ import annotations

import pytest

from datadec import constants as consts
from datadec.model_utils import (
    calc_batch_size,
    calc_lr_max,
    calc_total_tokens_from_str,
    calc_warmup_tokens,
    calculate_cumulative_lr,
    create_model_config,
    get_lr_at_step,
    model_size_str_to_true_int,
    numerical_cosine_integral,
    param_to_numeric,
    round_value_by_multiple,
)


class TestRoundValueByMultiple:
    def test_rounds_up(self) -> None:
        result = round_value_by_multiple(105.0, 10)
        assert result == 110

    def test_rounds_down(self) -> None:
        result = round_value_by_multiple(104.0, 10)
        assert result == 100

    def test_exact_multiple(self) -> None:
        result = round_value_by_multiple(100.0, 10)
        assert result == 100


class TestModelSizeStrToTrueInt:
    def test_valid_model_size(self) -> None:
        result = model_size_str_to_true_int("4M")
        assert result == consts.HARDCODED_SIZE_MAPPING["4M"]

    def test_another_model_size(self) -> None:
        result = model_size_str_to_true_int("1B")
        assert result == consts.HARDCODED_SIZE_MAPPING["1B"]


class TestParamToNumeric:
    def test_millions(self) -> None:
        result = param_to_numeric("4M")
        assert result == 4e6

    def test_billions(self) -> None:
        result = param_to_numeric("1B")
        assert result == 1e9

    def test_raw_number(self) -> None:
        result = param_to_numeric("1000000")
        assert result == 1000000.0

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError):
            param_to_numeric("invalid")


class TestCalcBatchSize:
    def test_returns_positive_int(self) -> None:
        result = calc_batch_size("4M")
        assert isinstance(result, int)
        assert result > 0

    def test_larger_model_larger_batch(self) -> None:
        small_batch = calc_batch_size("4M")
        large_batch = calc_batch_size("1B")
        assert large_batch > small_batch


class TestCalcTotalTokensFromStr:
    def test_computes_tokens(self) -> None:
        result = calc_total_tokens_from_str("20xC", "4M")
        expected = (
            20 * consts.TOKEN_LEN_XC_MULTIPLIER * consts.HARDCODED_SIZE_MAPPING["4M"]
        )
        assert result == expected


class TestCalcWarmupTokens:
    def test_returns_positive_int(self) -> None:
        result = calc_warmup_tokens("4M")
        assert isinstance(result, int)
        assert result > 0


class TestCalcLrMax:
    def test_returns_positive_float(self) -> None:
        result = calc_lr_max("4M")
        assert isinstance(result, float)
        assert result > 0

    def test_smaller_model_higher_lr(self) -> None:
        small_lr = calc_lr_max("4M")
        large_lr = calc_lr_max("1B")
        assert small_lr > large_lr


class TestCreateModelConfig:
    def test_returns_dict_with_expected_keys(self) -> None:
        config = create_model_config("4M")
        assert "batch_size" in config
        assert "total_tokens" in config
        assert "warmup_tokens" in config
        assert "lr_max" in config
        assert "lr_final" in config
        assert "total_steps" in config

    def test_invalid_model_size_raises(self) -> None:
        with pytest.raises(AssertionError):
            create_model_config("invalid_size")

    def test_kwargs_override(self) -> None:
        config = create_model_config("4M", custom_param=42)
        assert config["custom_param"] == 42


class TestNumericalCosineIntegral:
    def test_zero_decay_step_returns_zero(self) -> None:
        result = numerical_cosine_integral(0.01, 0.001, 1000, 0)
        assert result == 0.0

    def test_negative_decay_step_returns_zero(self) -> None:
        result = numerical_cosine_integral(0.01, 0.001, 1000, -10)
        assert result == 0.0

    def test_positive_integral(self) -> None:
        result = numerical_cosine_integral(0.01, 0.001, 1000, 500)
        assert result > 0


class TestCalculateCumulativeLr:
    def test_zero_step_returns_zero(self) -> None:
        result = calculate_cumulative_lr(0, 0.0, 0.01, 0.001, 100, 900)
        assert result == 0.0

    def test_negative_step_returns_zero(self) -> None:
        result = calculate_cumulative_lr(-5, 0.0, 0.01, 0.001, 100, 900)
        assert result == 0.0

    def test_during_warmup(self) -> None:
        result = calculate_cumulative_lr(50, 0.0, 0.01, 0.001, 100, 900)
        assert result > 0

    def test_after_warmup(self) -> None:
        result = calculate_cumulative_lr(500, 0.0, 0.01, 0.001, 100, 900)
        assert result > 0


class TestGetLrAtStep:
    def test_at_step_zero(self) -> None:
        result = get_lr_at_step(0, 0.0, 0.01, 0.001, 100, 900)
        assert result == 0.0

    def test_during_warmup(self) -> None:
        result = get_lr_at_step(50, 0.0, 0.01, 0.001, 100, 900)
        assert 0.0 < result < 0.01

    def test_at_warmup_end(self) -> None:
        result = get_lr_at_step(100, 0.0, 0.01, 0.001, 100, 900)
        assert result == 0.01

    def test_at_decay_end(self) -> None:
        result = get_lr_at_step(1000, 0.0, 0.01, 0.001, 100, 900)
        assert result == 0.001

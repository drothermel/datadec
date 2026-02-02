from __future__ import annotations

import pytest

from datadec.validation import (
    _choices_valid,
    _select_choices,
    _validated_select,
    determine_filter_types,
    validate_filter_types,
    validate_metric_type,
    validate_metrics,
)


class TestChoicesValid:
    def test_all_is_valid(self) -> None:
        assert _choices_valid("all", ["a", "b", "c"]) is True

    def test_single_valid_choice(self) -> None:
        assert _choices_valid("a", ["a", "b", "c"]) is True

    def test_single_invalid_choice(self) -> None:
        assert _choices_valid("x", ["a", "b", "c"]) is False

    def test_list_all_valid(self) -> None:
        assert _choices_valid(["a", "b"], ["a", "b", "c"]) is True

    def test_list_some_invalid(self) -> None:
        assert _choices_valid(["a", "x"], ["a", "b", "c"]) is False

    def test_empty_list(self) -> None:
        assert _choices_valid([], ["a", "b", "c"]) is True


class TestSelectChoices:
    def test_all_returns_all_options(self) -> None:
        result = _select_choices("all", ["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_all_with_exclude(self) -> None:
        result = _select_choices("all", ["a", "b", "c"], exclude=["b"])
        assert result == ["a", "c"]

    def test_single_choice(self) -> None:
        result = _select_choices("a", ["a", "b", "c"])
        assert "a" in result
        assert len(result) == 1

    def test_list_of_choices(self) -> None:
        result = _select_choices(["a", "b"], ["a", "b", "c"])
        assert set(result) == {"a", "b"}

    def test_list_with_exclude(self) -> None:
        result = _select_choices(["a", "b"], ["a", "b", "c"], exclude=["b"])
        assert result == ["a"]


class TestValidateFilterTypes:
    def test_valid_filter_types(self) -> None:
        validate_filter_types(["max_steps"])
        validate_filter_types(["ppl"])
        validate_filter_types(["olmes"])
        validate_filter_types(["max_steps", "ppl"])

    def test_invalid_filter_type_raises(self) -> None:
        with pytest.raises(AssertionError):
            validate_filter_types(["invalid_type"])

    def test_empty_list_is_valid(self) -> None:
        validate_filter_types([])


class TestValidateMetricType:
    def test_valid_metric_types(self) -> None:
        validate_metric_type("ppl")
        validate_metric_type("olmes")

    def test_none_is_valid(self) -> None:
        validate_metric_type(None)

    def test_invalid_metric_type_raises(self) -> None:
        with pytest.raises(AssertionError):
            validate_metric_type("invalid_type")


class TestValidateMetrics:
    def test_valid_ppl_metrics(self) -> None:
        validate_metrics(["pile-valppl", "c4_en-valppl"])

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(AssertionError):
            validate_metrics(["totally_invalid_metric_name"])

    def test_with_df_cols_valid(self) -> None:
        validate_metrics(["pile-valppl"], df_cols=["pile-valppl", "other_col"])

    def test_with_df_cols_missing_raises(self) -> None:
        with pytest.raises(AssertionError):
            validate_metrics(["pile-valppl"], df_cols=["other_col"])


class TestValidatedSelect:
    def test_valid_selection(self) -> None:
        result = _validated_select("all", ["a", "b", "c"], "test")
        assert result == ["a", "b", "c"]

    def test_invalid_choice_raises(self) -> None:
        with pytest.raises(AssertionError):
            _validated_select("invalid", ["a", "b", "c"], "test")

    def test_with_exclude(self) -> None:
        result = _validated_select("all", ["a", "b", "c"], "test", exclude=["c"])
        assert result == ["a", "b"]


class TestDetermineFilterTypes:
    def test_ppl_only_metrics(self) -> None:
        ppl_metrics = ["pile-valppl", "c4_en-valppl"]
        result = determine_filter_types(ppl_metrics)
        assert "max_steps" in result
        assert "ppl" in result
        assert "olmes" not in result

    def test_olmes_only_metrics(self) -> None:
        olmes_metrics = ["mmlu_average_acc_raw", "arc_challenge_acc_raw"]
        result = determine_filter_types(olmes_metrics)
        assert "max_steps" in result
        assert "olmes" in result
        assert "ppl" not in result

    def test_mixed_metrics_returns_only_max_steps(self) -> None:
        mixed_metrics = ["pile-valppl", "mmlu_average_acc_raw"]
        result = determine_filter_types(mixed_metrics)
        assert result == ["max_steps"]

    def test_empty_metrics(self) -> None:
        result = determine_filter_types([])
        assert result == ["max_steps"]

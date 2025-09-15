import pandas as pd

from datadec.table_formatter import (
    _apply_column_formatting,
    _format_value,
    _get_column_names,
    _preprocess_data,
    _resolve_headers,
    format_coverage_table,
    format_dynamics_table,
    format_table,
    load_table_config,
)


class TestFormatTable:
    def test_format_table_with_dataframe(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "value": [1.2345, 2.6789]})
        result = format_table(df, output_format="plain")
        assert "Alice" in result
        assert "Bob" in result

    def test_format_table_with_dict_list(self):
        data = [{"name": "Alice", "value": 1.2345}, {"name": "Bob", "value": 2.6789}]
        result = format_table(data, output_format="plain")
        assert "Alice" in result
        assert "Bob" in result

    def test_format_table_with_list_of_lists(self):
        data = [["Alice", 1.2345], ["Bob", 2.6789]]
        headers = ["name", "value"]
        result = format_table(data, headers=headers, output_format="plain")
        assert "Alice" in result
        assert "Bob" in result

    def test_output_formats(self):
        data = [{"name": "Alice", "value": 123}]

        console_result = format_table(data, output_format="console")
        assert "+" in console_result or "|" in console_result

        markdown_result = format_table(data, output_format="markdown")
        assert "|" in markdown_result

        latex_result = format_table(data, output_format="latex")
        assert "tabular" in latex_result or "Alice" in latex_result

        plain_result = format_table(data, output_format="plain")
        assert "Alice" in plain_result

    def test_column_config_formatting(self):
        data = [{"lr": 0.001, "loss": 2.34567}]
        config = {
            "lr": {
                "header": "Learning Rate",
                "formatter": "scientific",
                "precision": 2,
            },
            "loss": {"header": "Loss", "formatter": "decimal", "precision": 2},
        }
        result = format_table(data, column_config=config, output_format="plain")
        assert "1.00e-03" in result
        assert "2.35" in result

    def test_empty_data(self):
        result = format_table([], output_format="plain")
        assert result == ""

    def test_none_values_handling(self):
        data = [{"name": "Alice", "value": None}]
        result = format_table(data, output_format="plain")
        assert "None" in result


class TestPreprocessData:
    def test_preprocess_dataframe(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _preprocess_data(df)
        assert result == [[1, 3], [2, 4]]

    def test_preprocess_dict_list(self):
        data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = _preprocess_data(data)
        assert result == [[1, 2], [3, 4]]

    def test_preprocess_list_of_lists(self):
        data = [[1, 2], [3, 4]]
        result = _preprocess_data(data)
        assert result == [[1, 2], [3, 4]]

    def test_preprocess_empty_data(self):
        result = _preprocess_data([])
        assert result == []


class TestGetColumnNames:
    def test_get_column_names_dataframe(self):
        df = pd.DataFrame({"col1": [1], "col2": [2]})
        result = _get_column_names(df)
        assert result == ["col1", "col2"]

    def test_get_column_names_dict_list(self):
        data = [{"col1": 1, "col2": 2}]
        result = _get_column_names(data)
        assert result == ["col1", "col2"]

    def test_get_column_names_list_of_lists(self):
        data = [[1, 2]]
        result = _get_column_names(data)
        assert result == []


class TestApplyColumnFormatting:
    def test_apply_column_formatting_with_config(self):
        data = [[0.001, 2.34567]]
        config = {
            "col1": {"formatter": "scientific", "precision": 2},
            "col2": {"formatter": "decimal", "precision": 2},
        }
        column_names = ["col1", "col2"]
        result = _apply_column_formatting(data, config, column_names)
        assert result == [["1.00e-03", "2.35"]]

    def test_apply_column_formatting_empty_data(self):
        result = _apply_column_formatting([], {}, [])
        assert result == []


class TestFormatValue:
    def test_format_value_scientific(self):
        config = {"col1": {"formatter": "scientific", "precision": 2}}
        result = _format_value(0.001, "col1", config)
        assert result == "1.00e-03"

    def test_format_value_decimal(self):
        config = {"col1": {"formatter": "decimal", "precision": 2}}
        result = _format_value(2.34567, "col1", config)
        assert result == "2.35"

    def test_format_value_integer(self):
        config = {"col1": {"formatter": "integer"}}
        result = _format_value(1234, "col1", config)
        assert result == "1,234"

    def test_format_value_comma(self):
        config = {"col1": {"formatter": "comma"}}
        result = _format_value(1234567, "col1", config)
        assert result == "1,234,567"

    def test_format_value_truncate(self):
        config = {"col1": {"formatter": "truncate", "max_length": 5}}
        result = _format_value("very long string", "col1", config)
        assert result == "very ..."

    def test_format_value_none(self):
        config = {"col1": {"formatter": "decimal", "precision": 2}}
        result = _format_value(None, "col1", config)
        assert result == "None"

    def test_format_value_no_config(self):
        result = _format_value("test", "unknown_col", {})
        assert result == "test"


class TestResolveHeaders:
    def test_resolve_headers_provided(self):
        headers = ["Custom1", "Custom2"]
        result = _resolve_headers(headers, ["col1", "col2"], {})
        assert result == ["Custom1", "Custom2"]

    def test_resolve_headers_from_config(self):
        config = {"col1": {"header": "Column 1"}, "col2": {"header": "Column 2"}}
        result = _resolve_headers(None, ["col1", "col2"], config)
        assert result == ["Column 1", "Column 2"]

    def test_resolve_headers_default(self):
        result = _resolve_headers(None, ["col1", "col2"], {})
        assert result == ["col1", "col2"]


class TestFormatDynamicsTable:
    def test_format_dynamics_table_basic(self):
        dynamics_list = [{"run_id": "run1", "max_lr": 0.001, "final_train_loss": 2.345}]
        result = format_dynamics_table(dynamics_list, output_format="plain")
        assert "run1" in result
        assert "1.00e-03" in result

    def test_format_dynamics_table_with_columns(self):
        dynamics_list = [{"run_id": "run1", "max_lr": 0.001, "final_train_loss": 2.345}]
        result = format_dynamics_table(
            dynamics_list, columns=["run_id", "max_lr"], output_format="plain"
        )
        assert "run1" in result
        assert "1.00e-03" in result
        assert "2.345" not in result

    def test_format_dynamics_table_empty(self):
        result = format_dynamics_table([])
        assert result == "No data to display"


class TestFormatCoverageTable:
    def test_format_coverage_table(self):
        df = pd.DataFrame({"col1": [1, 2, None], "col2": [1, 2, 3]})
        result = format_coverage_table(df, output_format="plain")
        assert "Column Coverage" in result
        assert "col1" in result
        assert "col2" in result
        assert "66.7" in result
        assert "100.0" in result

    def test_format_coverage_table_custom_title(self):
        df = pd.DataFrame({"col1": [1, 2]})
        result = format_coverage_table(df, title="Custom Title", output_format="plain")
        assert "Custom Title" in result


class TestIntegration:
    def test_scientific_notation_formatting(self):
        data = [{"lr": 1e-5}, {"lr": 1e-3}]
        config = {"lr": {"formatter": "scientific", "precision": 1}}
        result = format_table(data, column_config=config, output_format="plain")
        assert "1.0e-05" in result
        assert "1.0e-03" in result

    def test_mixed_data_types(self):
        data = [{"name": "experiment1", "lr": 0.001, "steps": 1000, "loss": None}]
        config = {
            "name": {"formatter": "string"},
            "lr": {"formatter": "scientific", "precision": 2},
            "steps": {"formatter": "integer"},
            "loss": {"formatter": "decimal", "precision": 3},
        }
        result = format_table(data, column_config=config, output_format="plain")
        assert "experiment1" in result
        assert "1.00e-03" in result
        assert "1,000" in result
        assert "None" in result

    def test_all_output_formats_produce_content(self):
        data = [{"name": "test", "value": 123}]
        formats = ["console", "markdown", "latex", "plain", "csv"]

        for fmt in formats:
            result = format_table(data, output_format=fmt)
            assert len(result) > 0
            assert "test" in result or "123" in result

    def test_disable_numparse_behavior(self):
        data = [{"scientific": "1.00e-03", "number": 123.456}]

        result_disabled = format_table(
            data, disable_numparse=True, output_format="plain"
        )
        assert "1.00e-03" in result_disabled

        result_enabled = format_table(
            data, disable_numparse=False, output_format="plain"
        )
        assert "scientific" in result_enabled
        assert "number" in result_enabled


class TestConfigLoading:
    def test_load_table_config_existing(self):
        config = load_table_config("wandb_analysis")
        assert "run_id" in config
        assert config["run_id"]["header"] == "Run ID"
        assert config["run_id"]["formatter"] == "truncate"

    def test_load_table_config_nonexistent(self):
        config = load_table_config("nonexistent_config")
        assert config == {}

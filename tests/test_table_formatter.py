import pandas as pd

from dr_frames import format_coverage_table, format_table
from dr_render import format_dynamics_table, load_table_config


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

        plain_result = format_table(data, output_format="plain")
        assert "Alice" in plain_result

        markdown_result = format_table(data, output_format="markdown")
        assert "|" in markdown_result

        latex_result = format_table(data, output_format="latex")
        assert "tabular" in latex_result or "Alice" in latex_result

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


class TestFormatDynamicsTable:
    def test_format_dynamics_table_basic(self):
        dynamics_list = [{"run_id": "run1", "max_lr": 0.001, "final_train_loss": 2.345}]
        result = format_dynamics_table(dynamics_list, output_format="plain")
        assert "run1" in result

    def test_format_dynamics_table_with_columns(self):
        dynamics_list = [{"run_id": "run1", "max_lr": 0.001, "final_train_loss": 2.345}]
        result = format_dynamics_table(
            dynamics_list, columns=["run_id", "max_lr"], output_format="plain"
        )
        assert "run1" in result

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

    def test_format_coverage_table_custom_title(self):
        df = pd.DataFrame({"col1": [1, 2]})
        result = format_coverage_table(df, title="Custom Title", output_format="plain")
        assert "Custom Title" in result


class TestConfigLoading:
    def test_load_table_config_nonexistent(self):
        config = load_table_config("nonexistent_config")
        assert config == {}

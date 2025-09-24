import pandas as pd
import yaml

from datadec.constants import CONFIGS_DIR

TABLE_FORMATS_DIR = CONFIGS_DIR / "table_formats"
COVERAGE_TABLE_CONFIG = {
    "index": {"header": "#", "formatter": "integer"},
    "column": {"header": "Column", "formatter": "truncate", "max_length": 35},
    "coverage": {"header": "Coverage %", "formatter": "decimal", "precision": 1},
}


def format_table(**kwargs: dict) -> str:
    return "Not Implemented"


def load_table_config(config_name: str) -> dict[str, dict]:
    config_file = TABLE_FORMATS_DIR / f"{config_name}.yaml"
    if config_file.exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    else:
        return {}


def format_dynamics_table(
    dynamics_list: list[dict],
    columns: list[str] = None,
    output_format: str = "console",
    table_style: str = "lines",
    disable_numparse: bool = True,
) -> str:
    if not dynamics_list:
        return "No data to display"
    if columns:
        filtered_data = []
        for dynamics in dynamics_list:
            filtered_row = {col: dynamics.get(col) for col in columns}
            filtered_data.append(filtered_row)
        data_to_format = filtered_data
    else:
        data_to_format = dynamics_list
    wandb_config = load_table_config("wandb_analysis")
    return format_table(
        data=data_to_format,
        output_format=output_format,
        column_config=wandb_config,
        table_style=table_style,
        disable_numparse=disable_numparse,
    )


def format_coverage_table(
    df: pd.DataFrame,
    title: str = "Column Coverage",
    output_format: str = "console",
    table_style: str = "lines",
    disable_numparse: bool = True,
) -> str:
    coverage_data = []
    for i, col in enumerate(df.columns):
        coverage = df[col].notna().sum() / len(df) * 100
        coverage_data.append({"index": i + 1, "column": col, "coverage": coverage})
    result = f"{title} ({len(df.columns)} columns):\n"
    result += format_table(
        data=coverage_data,
        output_format=output_format,
        column_config=COVERAGE_TABLE_CONFIG,
        table_style=table_style,
        disable_numparse=disable_numparse,
    )
    return result

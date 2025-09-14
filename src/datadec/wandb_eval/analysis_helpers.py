from typing import Any, Dict, List

import pandas as pd

from datadec.wandb_eval import wandb_constants as wconsts
from datadec.wandb_eval.wandb_loader import WandBDataLoader


def load_df() -> pd.DataFrame:
    loader = WandBDataLoader()
    return loader.load_runs_and_history()


def pretty_print_vals(value: Any) -> str:
    if isinstance(value, float):
        return pretty_print_floats(value)
    return str(value)


def pretty_print_floats(value: float) -> str:
    if abs(value) < 0.001 or abs(value) > 1000:
        return f"{value:.2e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def print_dynamics_summary_table(
    dynamics_list: List[Dict[str, Any]],
    columns: List[str] = None,
    column_widths: Dict[str, int] = None,
) -> None:
    if not dynamics_list:
        print("No dynamics data to display.")
        return

    if columns is None:
        columns = [
            "run_id",
            "total_steps",
            "max_lr",
            "final_lr",
            "final_train_loss",
            "max_tokens",
        ]

    default_widths = {
        "run_id": 52,
        "total_steps": 8,
        "max_lr": 10,
        "min_lr": 10,
        "initial_lr": 10,
        "final_lr": 10,
        "initial_train_loss": 10,
        "final_train_loss": 12,
        "min_train_loss": 10,
        "max_tokens": 12,
        "max_epoch": 10,
        "loss_improvement": 8,
    }

    if column_widths:
        default_widths.update(column_widths)

    header_map = {
        "run_id": "Run ID (first 50 chars)",
        "total_steps": "Steps",
        "max_lr": "Max LR",
        "min_lr": "Min LR",
        "initial_lr": "Init LR",
        "final_lr": "Final LR",
        "initial_train_loss": "Init Loss",
        "final_train_loss": "Final Loss",
        "min_train_loss": "Min Loss",
        "max_tokens": "Max Tokens",
        "max_epoch": "Max Epoch",
        "loss_improvement": "Loss Δ",
    }

    header_line = ""
    separator_line = ""
    for col in columns:
        width = default_widths.get(col, 12)
        header = header_map.get(col, col.replace("_", " ").title())
        header_line += f"{header:<{width}} "
        separator_line += "-" * width + " "

    print(header_line.rstrip())
    print(separator_line.rstrip())

    for dynamics in dynamics_list:
        row_line = ""
        for col in columns:
            width = default_widths.get(col, 12)
            value = dynamics.get(col)

            if col in wconsts.TRUNCATE_COLUMNS:
                formatted_value = (
                    value[: wconsts.TRUNCATE_LENGTH] + "..."
                    if value and len(value) > wconsts.TRUNCATE_LENGTH
                    else (value or "None")
                )
            elif col in wconsts.SCIENTIFIC_NOTATION_COLUMNS:
                formatted_value = f"{value:.2e}" if value is not None else "None"
            elif col in wconsts.THREE_DECIMAL_PLACES_COLUMNS:
                formatted_value = f"{value:.3f}" if value is not None else "None"
            elif col in wconsts.COMMA_SEPARATED_COLUMNS:
                formatted_value = f"{value:,.0f}" if value is not None else "None"
            else:
                formatted_value = str(value) if value is not None else "None"

            row_line += f"{formatted_value:<{width}} "

        print(row_line.rstrip())


def format_experimental_summary(summary: Dict[str, Any], method: str = None) -> str:
    method_label = f" {method.upper()}" if method else ""
    lines = [
        f"{method_label} Experimental Design Summary:",
        f"  Total runs: {summary['total_runs']}",
    ]
    if method:
        lines.append(f"  {method.title()} runs: {summary['method_runs']}")
    lines.extend(
        [
            f"  Complete runs: {summary['complete_runs']}",
            f"  Finished runs: {summary['finished_runs']}",
        ]
    )
    if summary.get("lr_count", 0) > 0:
        lr_list = summary["unique_learning_rates"]
        lr_display = (
            str(lr_list)
            if len(lr_list) <= 5
            else f"{lr_list[:3]}...+{len(lr_list) - 3} more"
        )
        lines.append(f"  Learning rates ({summary['lr_count']}): {lr_display}")
    if summary.get("model_size_count", 0) > 0:
        lines.append(
            f"  Model sizes ({summary['model_size_count']}): {summary['unique_model_sizes']}"
        )
    if summary.get("dataset_size_count", 0) > 0:
        lines.append(
            f"  Dataset sizes ({summary['dataset_size_count']}): {summary['unique_dataset_sizes']}"
        )
    return "\n".join(lines)


def print_dataframe_coverage(df: pd.DataFrame, title: str = "Column Coverage") -> None:
    print(f"\n{title} ({len(df.columns)} columns):")
    for i, col in enumerate(df.columns):
        coverage = df[col].notna().sum() / len(df) * 100
        print(f"  {i + 1:2d}. {col:<35} ({coverage:5.1f}% coverage)")

from __future__ import annotations

from typing import Any

import pandas as pd

from datadec.wandb_eval import wandb_constants as wconsts

DEFAULT_PANDAS_OPTIONS = {
    "display.max_columns": None,
    "display.width": 300,
    "display.max_colwidth": 500,
    "display.expand_frame_repr": True,
}

DEFAULT_COLS = [
    "run_id",
    "total_steps",
    "max_lr",
    "final_lr",
    "final_train_loss",
    "max_tokens",
]

DEFAULT_COLUMN_WIDTHS = {
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

DEFAULT_COLUMN_HEADERS = {
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


def configure_pandas_display(width: int = 300) -> None:
    for key, value in DEFAULT_PANDAS_OPTIONS.items():
        pd.set_option(key, value)


def print_dynamics_summary_table(
    dynamics_list: list[dict[str, Any]],
    columns: list[str] = [],
    column_widths: dict[str, int] = {},
) -> None:
    assert len(dynamics_list) > 0, "No dynamics data to display."
    columns = [*DEFAULT_COLS] if len(columns) == 0 else columns
    column_widths = {**DEFAULT_COLUMN_WIDTHS, **column_widths}
    header_map = {**DEFAULT_COLUMN_HEADERS}

    header_line = ""
    separator_line = ""
    for col in columns:
        width = column_widths.get(col, 12)
        header = header_map.get(col, col.replace("_", " ").title())
        header_line += f"{header:<{width}} "
        separator_line += "-" * width + " "

    print(header_line.rstrip())
    print(separator_line.rstrip())

    for dynamics in dynamics_list:
        row_line = ""
        for col in columns:
            width = column_widths.get(col, 12)
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


def print_dataframe_coverage(df: pd.DataFrame, title: str = "Column Coverage") -> None:
    print(f"\n{title} ({len(df.columns)} columns):")
    for i, col in enumerate(df.columns):
        coverage = df[col].notna().sum() / len(df) * 100
        print(f"  {i + 1:2d}. {col:<35} ({coverage:5.1f}% coverage)")


def get_step_dynamics(run_history: pd.DataFrame) -> dict[str, Any]:
    step_values = run_history["step"].dropna()
    if len(step_values) == 0:
        return {}
    return {
        "max_step": step_values.max(),
        "min_step": step_values.min(),
    }


def get_token_dynamics(run_history: pd.DataFrame) -> dict[str, Any]:
    total_tokens = run_history["total_tokens"].dropna()
    if len(total_tokens) == 0:
        return {}
    return {
        "max_tokens": total_tokens.max(),
        "min_tokens": total_tokens.min(),
    }


def get_loss_dynamics(run_history: pd.DataFrame) -> dict[str, Any]:
    train_loss = run_history["train_loss"].dropna()
    if len(train_loss) == 0:
        return {}
    return {
        "initial_train_loss": train_loss.iloc[0],
        "final_train_loss": train_loss.iloc[-1],
        "min_train_loss": train_loss.min(),
        "max_train_loss": train_loss.max(),
        "loss_improvement": train_loss.iloc[0] - train_loss.iloc[-1]
        if len(train_loss) > 1
        else None,
    }


def get_lr_dynamics(run_history: pd.DataFrame) -> dict[str, Any]:
    lr_values = run_history["learning_rate"].dropna()
    if len(lr_values) == 0:
        return {}
    return {
        "initial_lr": lr_values.iloc[0],
        "max_lr": lr_values.max(),
        "final_lr": lr_values.iloc[-1],
    }


def get_epoch_dynamics(run_history: pd.DataFrame) -> dict[str, Any]:
    epoch_values = run_history["epoch"].dropna()
    if len(epoch_values) == 0:
        return {}
    return {
        "max_epoch": epoch_values.max(),
    }


def get_run_training_dynamics(
    history_df: pd.DataFrame, run_id: str
) -> dict[str, Any] | None:
    run_history = history_df[history_df["run_id"] == run_id].copy()
    if len(run_history) == 0:
        return None
    run_history = run_history.sort_values("step")
    dynamics = {
        "run_id": run_id,
        "step_count": len(run_history),
    }
    if "step" in run_history.columns:
        dynamics.update(get_step_dynamics(run_history))
    if "learning_rate" in run_history.columns:
        dynamics.update(get_lr_dynamics(run_history))
    if "train_loss" in run_history.columns:
        dynamics.update(get_loss_dynamics(run_history))
    if "total_tokens" in run_history.columns:
        dynamics.update(get_token_dynamics(run_history))
    if "epoch" in run_history.columns:
        dynamics.update(get_epoch_dynamics(run_history))
    return dynamics


def analyze_training_progression(
    history_df: pd.DataFrame, run_ids: list[str]
) -> pd.DataFrame:
    dynamics_results = []
    for run_id in run_ids:
        dynamics = get_run_training_dynamics(history_df, run_id)
        if dynamics:
            dynamics_results.append(dynamics)
    return pd.DataFrame(dynamics_results)


def print_training_history_sample(
    run_id: str,
    history_df: pd.DataFrame,
    sample_size: int = 3,
    columns: list[str] | None = None,
) -> None:
    run_history = history_df[history_df["run_id"] == run_id].sort_values("step")
    if len(run_history) == 0:
        print(f"No training history found for run: {run_id}")
        return
    print(f"Run ID: {run_id}")
    print(f"Training progression ({len(run_history)} steps):")
    columns = wconsts.DEFAULT_COLUMNS if columns is None else columns
    available_cols = [col for col in columns if col in run_history.columns]
    print(f"\nFirst {sample_size} training steps:")
    print(run_history[available_cols].head(sample_size).to_string(index=False))
    if len(run_history) > sample_size:
        print(f"\nLast {sample_size} training steps:")
        print(run_history[available_cols].tail(sample_size).to_string(index=False))
    if "learning_rate" in run_history.columns:
        lr_values = run_history["learning_rate"].dropna()
        if len(lr_values) > 1:
            print("\nLearning rate schedule:")
            print(f"  Initial LR: {lr_values.iloc[0]:.6e}")
            print(f"  Maximum LR: {lr_values.max():.6e}")
            print(f"  Final LR: {lr_values.iloc[-1]:.6e}")
            if lr_values.iloc[0] != 0:
                print(
                    f"  LR ratio (final/initial): {lr_values.iloc[-1] / lr_values.iloc[0]:.3f}"
                )

import random
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from datadec import WandBStore

METHODS = ["dpo", "finetune"]
DEFAULT_REQUIRED_PARAMS = ["learning_rate", "model_size_m", "method"]
DEFAULT_COLUMNS = ["step", "learning_rate", "train_loss", "total_tokens", "epoch"]
SCIENTIFIC_NOTATION_COLUMNS = ["max_lr", "min_lr", "initial_lr", "final_lr"]
THREE_DECIMAL_PLACES_COLUMNS = [
    "initial_train_loss",
    "final_train_loss",
    "min_train_loss",
    "loss_improvement",
]
COMMA_SEPARATED_COLUMNS = ["max_tokens"]
TRUNCATE_COLUMNS = ["run_id"]
TRUNCATE_LENGTH = 50
RANDOM_SEED = 42


def load_runs_df() -> pd.DataFrame:
    store = WandBStore("postgresql+psycopg://localhost/wandb_test")
    return store.get_runs()


def load_history_df() -> pd.DataFrame:
    store = WandBStore("postgresql+psycopg://localhost/wandb_test")
    return store.get_history()


def load_runs_and_history_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    store = WandBStore("postgresql+psycopg://localhost/wandb_test")
    return store.get_runs(), store.get_history()


def load_random_run_sample(sample_size: int = 20) -> pd.DataFrame:
    runs_df = load_runs_df()
    random.seed(RANDOM_SEED)
    return runs_df.sample(n=min(sample_size, len(runs_df)))


def print_run_name_parsed(run_name, parsed_params):
    print(f"RUN NAME:    {run_name}")
    print("    PARSED PARAMETERS:")
    if len(parsed_params) == 0:
        print("      No parameters extracted\n")
        return
    for key, value in sorted(parsed_params.items()):
        print(f"      {key:<20}: {pretty_print_vals(value)}")
    print()


def pretty_print_vals(value: Any) -> str:
    if isinstance(value, float):
        return pretty_print_floats(value)
    return str(value)


def pretty_print_floats(value: float) -> str:
    if abs(value) < 0.001 or abs(value) > 1000:
        return f"{value:.2e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def extract_hyperparameters(run_name: str) -> Dict[str, Any]:
    """Extract comprehensive parameters from WandB run name.

    Parses structured run names like:
    2025_08_21-08_24_43_test_finetune_DD-dolma1_7-4M_main_1Mtx1_--learning_rate=5e-06
    """
    params = {}

    # 1. Date and time extraction
    date_time_match = re.search(r"^(\d{4}_\d{2}_\d{2})-(\d{2}_\d{2}_\d{2})_", run_name)
    if date_time_match:
        params["run_date"] = date_time_match.group(1)
        params["run_time"] = date_time_match.group(2)

    # 2. Experiment name (between timestamp and DD-)
    exp_name_match = re.search(
        r"^\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}_(.+?)_DD-", run_name
    )
    if exp_name_match:
        params["experiment_name"] = exp_name_match.group(1)

    # 3. Dataset family
    dataset_match = re.search(r"DD-(dolma\d+_\d+)-", run_name)
    if dataset_match:
        params["dataset_family"] = dataset_match.group(1)

    # 4. Model size
    model_match = re.search(r"dolma1_7-(\d+)M", run_name)
    if model_match:
        params["model_size_m"] = int(model_match.group(1))

    # 5. Checkpoint name
    checkpoint_match = re.search(r"-\d+M_(\w+)_\d+Mtx", run_name)
    if checkpoint_match:
        params["checkpoint"] = checkpoint_match.group(1)

    # 6. Parse Mtx format: <tokens>Mtx<epochs> (e.g., "10Mtx1" = 10M tokens × 1 epoch)
    mtx_match = re.search(r"(\d+)Mtx(\d+)", run_name)
    if mtx_match:
        dataset_tokens = int(mtx_match.group(1))
        epochs_from_name = int(mtx_match.group(2))
        params["dataset_tokens_m"] = dataset_tokens
        params["epochs_from_name"] = epochs_from_name
        params["dataset_total_m"] = dataset_tokens  # Keep for backward compatibility
        params["mtx_format"] = True
    else:
        # Legacy pattern: main_<base>Mtx<mult> (dataset scaling)
        legacy_match = re.search(r"main_(\d+)Mtx(\d+)", run_name)
        if legacy_match:
            base = int(legacy_match.group(1))
            mult = int(legacy_match.group(2))
            params["dataset_base_m"] = base
            params["dataset_mult"] = mult
            params["dataset_total_m"] = base * mult

    # 7. Extract all explicit parameters (--param=value format)
    explicit_params = re.findall(r"--(\w+)=([^\s_]+)", run_name)
    for param_name, param_value in explicit_params:
        param_key = f"explicit_{param_name}"
        # Try to convert to appropriate type
        try:
            if "." in param_value or "e" in param_value.lower():
                params[param_key] = float(param_value)
            else:
                params[param_key] = int(param_value)
        except ValueError:
            params[param_key] = param_value

    # 8. Learning rate (prioritize explicit, fallback to legacy patterns)
    if "explicit_learning_rate" in params:
        params["learning_rate"] = params["explicit_learning_rate"]
    else:
        lr_match = re.search(r"--learning_rate=([0-9\.e\-]+)", run_name)
        if lr_match:
            params["learning_rate"] = float(lr_match.group(1))

    # 9. Training method detection
    run_name_lower = run_name.lower()
    for method in METHODS:
        if method in run_name_lower:
            params["method"] = method
            break

    return params


def filter_runs_by_method(runs_df: pd.DataFrame, method: str) -> pd.DataFrame:
    runs_with_params = []
    for _, row in runs_df.iterrows():
        params = extract_hyperparameters(row["run_name"])
        if params.get("method") == method:
            combined_data = row.to_dict()
            combined_data.update(params)
            runs_with_params.append(combined_data)
    return pd.DataFrame(runs_with_params)


def get_step_dynamics(run_history: pd.DataFrame) -> Dict[str, Any]:
    step_values = run_history["step"].dropna()
    if len(step_values) == 0:
        return {}
    return {
        "max_step": step_values.max(),
        "min_step": step_values.min(),
    }


def get_token_dynamics(run_history: pd.DataFrame) -> Dict[str, Any]:
    total_tokens = run_history["total_tokens"].dropna()
    if len(total_tokens) == 0:
        return {}
    return {
        "max_tokens": total_tokens.max(),
        "min_tokens": total_tokens.min(),
        "token_progression": total_tokens.iloc[-1] - total_tokens.iloc[0]
        if len(total_tokens) > 1
        else None,
    }


def get_loss_dynamics(run_history: pd.DataFrame) -> Dict[str, Any]:
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


def get_lr_dynamics(run_history: pd.DataFrame) -> Dict[str, Any]:
    lr_values = run_history["learning_rate"].dropna()
    if len(lr_values) == 0:
        return {}
    return {
        "max_lr": lr_values.max(),
        "min_lr": lr_values.min(),
        "initial_lr": lr_values.iloc[0],
        "final_lr": lr_values.iloc[-1],
        "lr_decay_ratio": lr_values.iloc[-1] / lr_values.iloc[0]
        if lr_values.iloc[0] != 0
        else None,
    }


def get_epoch_dynamics(run_history: pd.DataFrame) -> Dict[str, Any]:
    epoch_values = run_history["epoch"].dropna()
    if len(epoch_values) == 0:
        return {}
    return {
        "max_epoch": epoch_values.max(),
    }


def get_run_training_dynamics(
    history_df: pd.DataFrame, run_id: str
) -> Optional[Dict[str, Any]]:
    run_history = history_df[history_df["run_id"] == run_id].copy()
    if len(run_history) == 0:
        return None
    run_history = run_history.sort_values("step")
    dynamics = {
        "run_id": run_id,
        "total_steps": len(run_history),
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
    history_df: pd.DataFrame, run_ids: List[str]
) -> pd.DataFrame:
    dynamics_results = []

    for run_id in run_ids:
        dynamics = get_run_training_dynamics(history_df, run_id)
        if dynamics:
            dynamics_results.append(dynamics)

    return pd.DataFrame(dynamics_results)


def get_complete_experimental_data(
    runs_df: pd.DataFrame, required_params: List[str] = None
) -> pd.DataFrame:
    if required_params is None:
        required_params = DEFAULT_REQUIRED_PARAMS
    enriched_runs = []
    for _, row in runs_df.iterrows():
        params = extract_hyperparameters(row["run_name"])
        combined_data = row.to_dict()
        combined_data.update(params)
        enriched_runs.append(combined_data)
    df = pd.DataFrame(enriched_runs)
    filters = [df["state"] == "finished"]
    for param in required_params:
        filters.append(df[param].notna())
    complete_mask = pd.concat(filters, axis=1).all(axis=1)
    return df[complete_mask]


def create_experimental_summary(
    runs_df: pd.DataFrame, method: str = None
) -> Dict[str, Any]:
    if method:
        filtered_df = filter_runs_by_method(runs_df, method)
        complete_df = get_complete_experimental_data(filtered_df)
    else:
        complete_df = get_complete_experimental_data(runs_df)
    summary = {
        "total_runs": len(runs_df),
        "method_runs": len(filtered_df) if method else len(runs_df),
        "complete_runs": len(complete_df),
        "finished_runs": (runs_df["state"] == "finished").sum(),
    }
    if len(complete_df) > 0:
        if "learning_rate" in complete_df.columns:
            summary["unique_learning_rates"] = sorted(
                complete_df["learning_rate"].unique()
            )
            summary["lr_count"] = complete_df["learning_rate"].nunique()
        if "model_size_m" in complete_df.columns:
            summary["unique_model_sizes"] = sorted(complete_df["model_size_m"].unique())
            summary["model_size_count"] = complete_df["model_size_m"].nunique()
        if "dataset_total_m" in complete_df.columns:
            summary["unique_dataset_sizes"] = sorted(
                complete_df["dataset_total_m"].unique()
            )
            summary["dataset_size_count"] = complete_df["dataset_total_m"].nunique()
        if (
            "learning_rate" in complete_df.columns
            and "model_size_m" in complete_df.columns
        ):
            coverage_table = pd.crosstab(
                complete_df["model_size_m"], complete_df["learning_rate"], margins=False
            )
            summary["model_lr_coverage"] = coverage_table.to_dict()
    return summary


def print_dataframe_coverage(df: pd.DataFrame, title: str = "Column Coverage") -> None:
    print(f"\n{title} ({len(df.columns)} columns):")
    for i, col in enumerate(df.columns):
        coverage = df[col].notna().sum() / len(df) * 100
        print(f"  {i + 1:2d}. {col:<35} ({coverage:5.1f}% coverage)")


def print_dynamics_summary_table(
    dynamics_list: List[Dict[str, Any]],
    columns: List[str] = None,
    column_widths: Dict[str, int] = None,
) -> None:
    if not dynamics_list:
        print("No dynamics data to display.")
        return

    # Default columns if none specified
    if columns is None:
        columns = [
            "run_id",
            "total_steps",
            "max_lr",
            "final_lr",
            "final_train_loss",
            "max_tokens",
        ]

    # Default column widths
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

    # Column headers and formatting
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

    # Print header
    header_line = ""
    separator_line = ""
    for col in columns:
        width = default_widths.get(col, 12)
        header = header_map.get(col, col.replace("_", " ").title())
        header_line += f"{header:<{width}} "
        separator_line += "-" * width + " "

    print(header_line.rstrip())
    print(separator_line.rstrip())

    # Print data rows
    for dynamics in dynamics_list:
        row_line = ""
        for col in columns:
            width = default_widths.get(col, 12)
            value = dynamics.get(col)

            if col in TRUNCATE_COLUMNS:
                formatted_value = (
                    value[:TRUNCATE_LENGTH] + "..."
                    if value and len(value) > TRUNCATE_LENGTH
                    else (value or "None")
                )
            elif col in SCIENTIFIC_NOTATION_COLUMNS:
                formatted_value = f"{value:.2e}" if value is not None else "None"
            elif col in THREE_DECIMAL_PLACES_COLUMNS:
                formatted_value = f"{value:.3f}" if value is not None else "None"
            elif col in COMMA_SEPARATED_COLUMNS:
                formatted_value = f"{value:,.0f}" if value is not None else "None"
            else:
                formatted_value = str(value) if value is not None else "None"

            row_line += f"{formatted_value:<{width}} "

        print(row_line.rstrip())


def print_training_history_sample(
    run_id: str,
    history_df: pd.DataFrame,
    sample_size: int = 3,
    columns: List[str] = None,
) -> None:
    run_history = history_df[history_df["run_id"] == run_id].sort_values("step")
    if len(run_history) == 0:
        print(f"No training history found for run: {run_id}")
        return
    print(f"Run ID: {run_id}")
    print(f"Training progression ({len(run_history)} steps):")
    columns = DEFAULT_COLUMNS if columns is None else columns
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

import pandas as pd
from rich.console import Console

from datadec.console_components import InfoBlock, MetricPanel, SectionRule, TitlePanel
from datadec.table_formatter import format_table, load_table_config
from datadec.fancy_table import create_learning_rate_table
from datadec.wandb_eval import analysis_helpers
from datadec.wandb_eval import wandb_transforms as transforms
from datadec.wandb_eval.wandb_loader import WandBDataLoader

analysis_helpers.configure_pandas_display()

METRICS = {
    "olmes_acc": "OLMES Accuracy",
    "pile_perplexity": "Pile Perplexity",
    "arc_challenge": "ARC Challenge",
    "mmlu_acc": "MMLU Accuracy",
    "train_loss": "Train Loss",
}
GROUP_SWEEPS_BY = ["model_size_m", "dataset_total_m"]
TABLE_CONFIG = load_table_config("wandb_analysis")
TABLE_STYLE = "zebra"  # None or "lines" or "zebra"


def create_lr_sweep_table_config(display_df, metric_key):
    config = TABLE_CONFIG.copy()
    for col in display_df.columns:
        if col.startswith("LR_"):
            lr_value = float(col.replace("LR_", ""))
            config[col] = {
                "header": f"{lr_value:.0e}",
                "formatter": "decimal",
                "precision": 1 if metric_key == "pile_perplexity" else 3,
            }
    return config


def get_param_data_from_run_names(runs_df: pd.DataFrame) -> pd.DataFrame:
    param_data = []
    for _, row in runs_df.iterrows():
        params = transforms.extract_hyperparameters(row["run_name"])
        converted_params = {}
        for key, value in params.items():
            if key.endswith("_rnp"):
                base_key = key[:-4]
                if base_key == "lr":
                    converted_params["learning_rate"] = value
                elif base_key == "params":
                    if isinstance(value, str) and value.endswith("M"):
                        converted_params["model_size_m"] = int(value[:-1])
                elif base_key == "total_tok":
                    converted_params["dataset_total_m"] = value
                elif base_key == "method":
                    converted_params["method"] = value
                else:
                    converted_params[base_key] = value
            else:
                converted_params[key] = value
        converted_params.update(
            {
                "run_id": row["run_id"],
                "run_name": row["run_name"],
                "state": row["state"],
                "olmes_acc": row.get("pretrain_eval_olmes_10_macro_avg_acc_raw", None),
                "pile_perplexity": row.get(
                    "pretrain_eval/pile-validation/Perplexity", None
                ),
                "arc_challenge": row.get("pretrain_eval_arc_challenge_acc_raw", None),
                "mmlu_acc": row.get("pretrain_eval_mmlu_acc_raw", None),
                "train_loss": row.get("train_loss", None),
            }
        )
        param_data.append(converted_params)
    return pd.DataFrame(param_data)


def build_metric_lr_sweep_df(
    params_df: pd.DataFrame, lr_sweep_groups: pd.DataFrame, metric_key: str
) -> pd.DataFrame | None:
    metric_data = params_df[params_df[metric_key].notna()]
    if len(metric_data) == 0:
        return None

    performance_rows = []
    for (model_size, dataset_tokens), group in lr_sweep_groups:
        group_metric = group[group[metric_key].notna()]
        lrs_in_group = sorted(group_metric["learning_rate"].unique())
        if len(group_metric) < 2 or len(lrs_in_group) < 2:
            continue

        row_data = {
            "Model_Size": f"{int(model_size)}M",
            "Dataset_Tokens": f"{int(dataset_tokens)}M",
        }
        for lr in lrs_in_group:
            lr_runs = group_metric[group_metric["learning_rate"] == lr]
            if len(lr_runs) > 0:
                if metric_key == "pile_perplexity":
                    value = lr_runs[metric_key].mean()
                    row_data[f"LR_{lr:.0e}"] = f"{value:.1f}"
                else:
                    value = lr_runs[metric_key].mean()
                    row_data[f"LR_{lr:.0e}"] = f"{value:.3f}"
        performance_rows.append(row_data)
    return pd.DataFrame(performance_rows)


def main():
    console = Console()

    console.print(TitlePanel("SWEEP PERFORMANCE TABLES"))

    loader = WandBDataLoader()
    runs_df, _ = loader.load_runs_and_history()
    params_df = get_param_data_from_run_names(runs_df)
    finetune_df = params_df[
        (params_df["method"] == "finetune")
        & (params_df["state"] == "finished")
        & params_df["learning_rate"].notna()
        & params_df["model_size_m"].notna()
        & params_df["dataset_total_m"].notna()
    ]
    console.print(InfoBlock(f"Finetune runs with complete data: {len(finetune_df)}"))
    console.print()

    availability_data = []
    for metric_key, metric_name in METRICS.items():
        available_count = finetune_df[metric_key].notna().sum()
        availability_data.append(
            {
                "metric": metric_name,
                "available": available_count,
                "total": len(finetune_df),
                "coverage": f"{available_count / len(finetune_df) * 100:.1f}%",
            }
        )

    table = format_table(
        availability_data,
        column_config=TABLE_CONFIG,
        title="Data Availability Summary",
        table_style=TABLE_STYLE,
    )
    console.print(table)
    console.print()

    console.print(SectionRule("LEARNING RATE SWEEPS BY METRIC"))

    lr_sweep_groups = finetune_df.groupby(GROUP_SWEEPS_BY)
    for metric_key, metric_name in METRICS.items():
        perf_df = build_metric_lr_sweep_df(finetune_df, lr_sweep_groups, metric_key)
        if perf_df is None or len(perf_df) == 0:
            console.print(f"No complete LR sweep data available for {metric_name}")
            continue

        perf_df["_sort_model"] = perf_df["Model_Size"].str.replace("M", "").astype(int)
        perf_df["_sort_dataset"] = (
            perf_df["Dataset_Tokens"].str.replace("M", "").astype(int)
        )
        perf_df = perf_df.sort_values(["_sort_model", "_sort_dataset"])
        display_df = perf_df.drop(columns=["_sort_model", "_sort_dataset"])

        # Use FancyTable for better visualization of learning rate sweeps
        table = create_learning_rate_table(
            display_df,
            title=f"{metric_name} Learning Rate Sweep"
        )

        best_values_lines = []
        if metric_key != "pile_perplexity":  # Higher is better
            best_values_lines.append("📈 Best values:")
            for _, row in display_df.iterrows():
                lr_cols = [col for col in display_df.columns if col.startswith("LR_")]
                if lr_cols:
                    best_lr = max(
                        lr_cols,
                        key=lambda x: float(row[x]) if pd.notna(row[x]) else -1,
                    )
                    best_values_lines.append(
                        f"  {row['Model_Size']}, {row['Dataset_Tokens']} tokens → {best_lr}: {row[best_lr]}"
                    )
        else:
            best_values_lines.append("📉 Best values:")
            for _, row in display_df.iterrows():
                lr_cols = [col for col in display_df.columns if col.startswith("LR_")]
                if lr_cols:
                    best_lr = min(
                        lr_cols,
                        key=lambda x: float(row[x])
                        if pd.notna(row[x])
                        else float("inf"),
                    )
                    best_values_lines.append(
                        f"  {row['Model_Size']}, {row['Dataset_Tokens']} tokens → {best_lr}: {row[best_lr]}"
                    )

        console.print(MetricPanel(metric_name, table, "", "\n".join(best_values_lines)))


if __name__ == "__main__":
    main()

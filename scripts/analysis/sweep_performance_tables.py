from dataclasses import dataclass

import pandas as pd
from rich.console import Console

from datadec.console_components import (
    InfoBlock,
    SectionRule,
    SectionTitlePanel,
    TitlePanel,
    create_hyperparameter_sweep_table,
)
from dr_frames import format_table
from dr_render import load_table_config
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


@dataclass
class SweepResults:
    finetune_count: int
    availability_summary: list[dict]
    lr_sweeps: dict[str, pd.DataFrame]


def _get_param_data_from_run_names(runs_df: pd.DataFrame) -> pd.DataFrame:
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


def _build_metric_lr_sweep_df(
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


def _load_and_filter_data() -> pd.DataFrame:
    loader = WandBDataLoader()
    runs_df, _ = loader.load_runs_and_history()
    params_df = _get_param_data_from_run_names(runs_df)
    finetune_df = params_df[
        (params_df["method"] == "finetune")
        & (params_df["state"] == "finished")
        & params_df["learning_rate"].notna()
        & params_df["model_size_m"].notna()
        & params_df["dataset_total_m"].notna()
    ]
    return finetune_df


def _create_availability_summary(finetune_df: pd.DataFrame) -> list[dict]:
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
    return availability_data


def _build_all_lr_sweeps(finetune_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lr_sweeps = {}
    lr_sweep_groups = finetune_df.groupby(GROUP_SWEEPS_BY)
    for metric_key, _metric_name in METRICS.items():
        perf_df = _build_metric_lr_sweep_df(finetune_df, lr_sweep_groups, metric_key)
        if perf_df is not None and len(perf_df) > 0:
            perf_df["_sort_model"] = (
                perf_df["Model_Size"].str.replace("M", "").astype(int)
            )
            perf_df["_sort_dataset"] = (
                perf_df["Dataset_Tokens"].str.replace("M", "").astype(int)
            )
            perf_df = perf_df.sort_values(["_sort_model", "_sort_dataset"])
            display_df = perf_df.drop(columns=["_sort_model", "_sort_dataset"])
            lr_sweeps[metric_key] = display_df
    return lr_sweeps


def calculate_sweep_data() -> SweepResults:
    finetune_df = _load_and_filter_data()
    availability_summary = _create_availability_summary(finetune_df)
    lr_sweeps = _build_all_lr_sweeps(finetune_df)
    return SweepResults(
        finetune_count=len(finetune_df),
        availability_summary=availability_summary,
        lr_sweeps=lr_sweeps,
    )


def _display_title(console: Console):
    console.print(TitlePanel("Sweep Performance Tables: LR Sweep by Metric Type"))


def _display_availability_summary(
    console: Console, availability_data: list[dict], finetune_count: int
):
    console.print(InfoBlock(f"Finetune runs with complete data: {finetune_count}"))
    console.print()
    table = format_table(
        availability_data,
        column_config=TABLE_CONFIG,
        title="Data Availability Summary",
        table_style=TABLE_STYLE,
    )
    console.print(table)
    console.print()


def _display_lr_sweep_section(console: Console, lr_sweeps: dict[str, pd.DataFrame]):
    console.print(SectionRule("LEARNING RATE SWEEPS BY METRIC"))
    for metric_key, display_df in lr_sweeps.items():
        metric_name = METRICS[metric_key]
        lr_columns = [col for col in display_df.columns if col.startswith("LR_")]
        other_columns = [col for col in display_df.columns if not col.startswith("LR_")]
        table, info_block = create_hyperparameter_sweep_table(
            data=display_df,
            fixed_section={"title": "Training Setup", "columns": other_columns},
            swept_section={
                "title": "Learning Rate",
                "columns": lr_columns,
                "display_transform": lambda col: col.replace("LR_", ""),
            },
            optimization="min" if metric_key == "pile_perplexity" else "max",
            best_performance={
                "enabled": False,
                "title": "Best Perf",
                "column_names": ["LR", "Value"],
            },
            highlight_threshold=0.01,
        )
        console.print(SectionTitlePanel(metric_name))
        console.print(table)
        console.print(info_block)
    for metric_key, metric_name in METRICS.items():
        if metric_key not in lr_sweeps:
            console.print(f"No complete LR sweep data available for {metric_name}")


def display_sweep_results(results: SweepResults, console: Console):
    _display_title(console)
    _display_availability_summary(
        console, results.availability_summary, results.finetune_count
    )
    _display_lr_sweep_section(console, results.lr_sweeps)


def main():
    console = Console()
    results = calculate_sweep_data()
    display_sweep_results(results, console)


if __name__ == "__main__":
    main()

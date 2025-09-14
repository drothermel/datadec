import pandas as pd

from datadec.wandb_eval import analysis_helpers
from datadec.wandb_eval import wandb_transforms as transforms
from datadec.wandb_eval.wandb_loader import WandBDataLoader

analysis_helpers.configure_pandas_display()


def main():
    print("=== SWEEP PERFORMANCE TABLES ===\n")

    loader = WandBDataLoader()
    runs_df, _ = loader.load_runs_and_history()

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

    df = pd.DataFrame(param_data)

    finetune_df = df[
        (df["method"] == "finetune")
        & (df["state"] == "finished")
        & df["learning_rate"].notna()
        & df["model_size_m"].notna()
        & df["dataset_total_m"].notna()
    ]

    print(f"Finetune runs with complete data: {len(finetune_df)}")

    metrics = {
        "olmes_acc": "OLMES Accuracy",
        "pile_perplexity": "Pile Perplexity",
        "arc_challenge": "ARC Challenge",
        "mmlu_acc": "MMLU Accuracy",
        "train_loss": "Train Loss",
    }

    print("\n" + "=" * 80)
    print("LEARNING RATE SWEEP TABLES")
    print("=" * 80)
    print("Format: Rows = (Model Size, Dataset Tokens), Columns = Learning Rates")

    lr_sweep_groups = finetune_df.groupby(["model_size_m", "dataset_total_m"])

    for metric_key, metric_name in metrics.items():
        metric_data = finetune_df[finetune_df[metric_key].notna()]
        if len(metric_data) == 0:
            continue

        print(f"\n{'-' * 60}")
        print(f"METRIC: {metric_name}")
        print(f"{'-' * 60}")

        performance_rows = []

        for (model_size, dataset_tokens), group in lr_sweep_groups:
            group_metric = group[group[metric_key].notna()]
            if len(group_metric) < 2:  # Need at least 2 LRs for a sweep
                continue

            lrs_in_group = sorted(group_metric["learning_rate"].unique())
            if len(lrs_in_group) < 2:
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

        if performance_rows:
            perf_df = pd.DataFrame(performance_rows)

            perf_df["_sort_model"] = (
                perf_df["Model_Size"].str.replace("M", "").astype(int)
            )
            perf_df["_sort_dataset"] = (
                perf_df["Dataset_Tokens"].str.replace("M", "").astype(int)
            )
            perf_df = perf_df.sort_values(["_sort_model", "_sort_dataset"])

            display_df = perf_df.drop(columns=["_sort_model", "_sort_dataset"])
            print(display_df.to_string(index=False))

            if metric_key != "pile_perplexity":  # Higher is better
                print(f"\n📈 Best {metric_name} values:")
                for _, row in display_df.iterrows():
                    lr_cols = [
                        col for col in display_df.columns if col.startswith("LR_")
                    ]
                    if lr_cols:
                        best_lr = max(
                            lr_cols,
                            key=lambda x: float(row[x]) if pd.notna(row[x]) else -1,
                        )
                        print(
                            f"  {row['Model_Size']}, {row['Dataset_Tokens']} tokens → {best_lr}: {row[best_lr]}"
                        )
            else:
                print(f"\n📉 Best {metric_name} values:")
                for _, row in display_df.iterrows():
                    lr_cols = [
                        col for col in display_df.columns if col.startswith("LR_")
                    ]
                    if lr_cols:
                        best_lr = min(
                            lr_cols,
                            key=lambda x: float(row[x])
                            if pd.notna(row[x])
                            else float("inf"),
                        )
                        print(
                            f"  {row['Model_Size']}, {row['Dataset_Tokens']} tokens → {best_lr}: {row[best_lr]}"
                        )
        else:
            print(f"No complete LR sweep data available for {metric_name}")

    print(f"\n\n{'=' * 80}")
    print("DATA AVAILABILITY SUMMARY")
    print(f"{'=' * 80}")

    for metric_key, metric_name in metrics.items():
        available_count = finetune_df[metric_key].notna().sum()
        print(
            f"{metric_name}: {available_count}/{len(finetune_df)} runs ({available_count / len(finetune_df) * 100:.1f}%)"
        )


if __name__ == "__main__":
    main()

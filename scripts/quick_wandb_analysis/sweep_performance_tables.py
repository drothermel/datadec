#!/usr/bin/env python3

from datadec import WandBStore
import pandas as pd
import re


def extract_hyperparameters(run_name):
    """Extract key hyperparameters from run name"""
    params = {}

    # Learning rate
    lr_match = re.search(r"--learning_rate=([0-9\.e\-]+)", run_name)
    if lr_match:
        params["learning_rate"] = float(lr_match.group(1))

    # Model size (in millions)
    model_match = re.search(r"dolma1_7-(\d+)M", run_name)
    if model_match:
        params["model_size_m"] = int(model_match.group(1))

    # Dataset tokens pattern
    token_match = re.search(r"main_(\d+)Mtx(\d+)", run_name)
    if token_match:
        base = int(token_match.group(1))
        mult = int(token_match.group(2))
        params["dataset_total_m"] = base * mult

    # Training method
    if "dpo" in run_name.lower():
        params["method"] = "dpo"
    elif "finetune" in run_name.lower():
        params["method"] = "finetune"

    return params


def main():
    print("=== SWEEP PERFORMANCE TABLES ===\n")

    store = WandBStore("postgresql+psycopg://localhost/wandb_test")
    runs_df = store.get_runs()

    # Extract hyperparameters and combine with metadata
    param_data = []
    for _, row in runs_df.iterrows():
        params = extract_hyperparameters(row["run_name"])
        params.update(
            {
                "run_id": row["run_id"],
                "run_name": row["run_name"],
                "state": row["state"],
                # Key evaluation metrics
                "olmes_acc": row.get("pretrain_eval_olmes_10_macro_avg_acc_raw", None),
                "pile_perplexity": row.get(
                    "pretrain_eval/pile-validation/Perplexity", None
                ),
                "arc_challenge": row.get("pretrain_eval_arc_challenge_acc_raw", None),
                "mmlu_acc": row.get("pretrain_eval_mmlu_acc_raw", None),
                "train_loss": row.get("train_loss", None),
            }
        )
        param_data.append(params)

    df = pd.DataFrame(param_data)

    # Filter to finished finetune runs with complete data
    finetune_df = df[
        (df["method"] == "finetune")
        & (df["state"] == "finished")
        & df["learning_rate"].notna()
        & df["model_size_m"].notna()
        & df["dataset_total_m"].notna()
    ]

    print(f"Finetune runs with complete data: {len(finetune_df)}")

    # Available metrics to display
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

    # Get unique combinations for LR sweeps
    lr_sweep_groups = finetune_df.groupby(["model_size_m", "dataset_total_m"])

    for metric_key, metric_name in metrics.items():
        # Check if this metric has data
        metric_data = finetune_df[finetune_df[metric_key].notna()]
        if len(metric_data) == 0:
            continue

        print(f"\n{'-' * 60}")
        print(f"METRIC: {metric_name}")
        print(f"{'-' * 60}")

        # Build the performance matrix
        performance_rows = []

        for (model_size, dataset_tokens), group in lr_sweep_groups:
            # Only include groups with multiple LRs and this metric data
            group_metric = group[group[metric_key].notna()]
            if len(group_metric) < 2:  # Need at least 2 LRs for a sweep
                continue

            # Get unique LRs for this group
            lrs_in_group = sorted(group_metric["learning_rate"].unique())
            if len(lrs_in_group) < 2:
                continue

            # Build row data
            row_data = {
                "Model_Size": f"{int(model_size)}M",
                "Dataset_Tokens": f"{int(dataset_tokens)}M",
            }

            # Add performance values for each LR
            for lr in lrs_in_group:
                lr_runs = group_metric[group_metric["learning_rate"] == lr]
                if len(lr_runs) > 0:
                    # Take mean if multiple runs, otherwise single value
                    if metric_key == "pile_perplexity":
                        # For perplexity, lower is better - show with 1 decimal
                        value = lr_runs[metric_key].mean()
                        row_data[f"LR_{lr:.0e}"] = f"{value:.1f}"
                    else:
                        # For accuracy metrics, higher is better - show with 3 decimals
                        value = lr_runs[metric_key].mean()
                        row_data[f"LR_{lr:.0e}"] = f"{value:.3f}"

            performance_rows.append(row_data)

        if performance_rows:
            # Create DataFrame and display
            perf_df = pd.DataFrame(performance_rows)

            # Sort by model size then dataset size
            perf_df["_sort_model"] = (
                perf_df["Model_Size"].str.replace("M", "").astype(int)
            )
            perf_df["_sort_dataset"] = (
                perf_df["Dataset_Tokens"].str.replace("M", "").astype(int)
            )
            perf_df = perf_df.sort_values(["_sort_model", "_sort_dataset"])

            # Drop sort columns and display
            display_df = perf_df.drop(columns=["_sort_model", "_sort_dataset"])
            print(display_df.to_string(index=False))

            # Show best performing conditions
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
            else:  # Lower is better for perplexity
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

    # Quick summary of data availability
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

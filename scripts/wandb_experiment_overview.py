#!/usr/bin/env python3

import pandas as pd

from datadec import analysis_helpers


def main():
    print("=== WandB Experiment Overview ===\n")

    runs_df = analysis_helpers.load_runs_df()

    param_data = []
    for _, row in runs_df.iterrows():
        params = analysis_helpers.extract_hyperparameters(row["run_name"])
        params.update(
            {
                "run_id": row["run_id"],
                "state": row["state"],
                "metadata_model_size": row.get("model_size", None),
                "metadata_lr": row.get("learning_rate", None),
            }
        )
        param_data.append(params)

    df = pd.DataFrame(param_data)

    print(f"Total runs: {len(df)}")
    print(f"Finished runs: {(df['state'] == 'finished').sum()}")
    print(f"Runs with parsed hyperparams: {df['learning_rate'].notna().sum()}")

    print("\n" + "=" * 60)
    print("1. MODEL SIZE × LEARNING RATE SWEEP")
    print("=" * 60)
    complete_df = df[
        df["model_size_m"].notna()
        & df["learning_rate"].notna()
        & (df["state"] == "finished")
    ]
    if len(complete_df) > 0:
        model_lr_table = pd.crosstab(
            complete_df["model_size_m"], complete_df["learning_rate"], margins=True
        )
        print("\nFinished runs by Model Size × Learning Rate:")
        print(f"({len(complete_df)} total finished runs with both params)")
        print(model_lr_table)

        print("\nModel size coverage:")
        for size in sorted(complete_df["model_size_m"].unique()):
            count = (complete_df["model_size_m"] == size).sum()
            lr_range = complete_df[complete_df["model_size_m"] == size]["learning_rate"]
            print(
                f"  {size}M: {count} runs, LR range {lr_range.min():.0e} to {lr_range.max():.0e}"
            )

    print("\n" + "=" * 60)
    print("2. DATASET SCALING EXPERIMENTS")
    print("=" * 60)
    dataset_df = df[df["dataset_total_m"].notna() & (df["state"] == "finished")]
    if len(dataset_df) > 0:
        print("\nFinished runs by Dataset Pattern:")
        print(f"({len(dataset_df)} total finished runs with dataset info)")
        dataset_pattern_table = pd.crosstab(
            dataset_df["dataset_base_m"], dataset_df["dataset_mult"], margins=True
        )
        print(dataset_pattern_table)
        print("\nTotal dataset sizes:")
        dataset_totals = dataset_df["dataset_total_m"].value_counts().sort_index()
        for total, count in dataset_totals.items():
            print(f"  {total}M tokens: {count} runs")
        if "model_size_m" in dataset_df.columns:
            model_dataset_df = dataset_df[dataset_df["model_size_m"].notna()]
            if len(model_dataset_df) > 0:
                print("\nModel Size × Dataset Size:")
                model_dataset_table = pd.crosstab(
                    model_dataset_df["model_size_m"],
                    model_dataset_df["dataset_total_m"],
                    margins=True,
                )
                print(model_dataset_table)

    print("\n" + "=" * 60)
    print("3. TRAINING METHOD COMPARISON")
    print("=" * 60)
    method_df = df[df["method"].notna() & (df["state"] == "finished")]
    if len(method_df) > 0:
        print("\nFinished runs by Training Method:")
        print(f"({len(method_df)} total finished runs with method info)")
        method_counts = method_df["method"].value_counts()
        for method, count in method_counts.items():
            print(f"  {method}: {count} runs")
        if "model_size_m" in method_df.columns:
            method_model_df = method_df[method_df["model_size_m"].notna()]
            if len(method_model_df) > 0:
                print("\nMethod × Model Size:")
                method_model_table = pd.crosstab(
                    method_model_df["method"],
                    method_model_df["model_size_m"],
                    margins=True,
                )
                print(method_model_table)

    print("\n" + "=" * 60)
    print("4. DATA COMPLETENESS SUMMARY")
    print("=" * 60)
    print("\nHyperparameter extraction success rates:")
    total_runs = len(df)
    print(
        f"  Learning rate: {df['learning_rate'].notna().sum()}/{total_runs} ({df['learning_rate'].notna().sum() / total_runs * 100:.1f}%)"
    )
    print(
        f"  Model size: {df['model_size_m'].notna().sum()}/{total_runs} ({df['model_size_m'].notna().sum() / total_runs * 100:.1f}%)"
    )
    print(
        f"  Dataset info: {df['dataset_total_m'].notna().sum()}/{total_runs} ({df['dataset_total_m'].notna().sum() / total_runs * 100:.1f}%)"
    )
    print(
        f"  Training method: {df['method'].notna().sum()}/{total_runs} ({df['method'].notna().sum() / total_runs * 100:.1f}%)"
    )

    finished_runs = (df["state"] == "finished").sum()
    print("\nOverall data quality:")
    print(
        f"  Finished runs: {finished_runs}/{total_runs} ({finished_runs / total_runs * 100:.1f}%)"
    )
    print(
        f"  Complete model+LR+finished: {len(complete_df)}/{total_runs} ({len(complete_df) / total_runs * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()

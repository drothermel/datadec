#!/usr/bin/env python3

from datadec import WandBStore
import pandas as pd
import re
from collections import defaultdict


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
        params["dataset_base_m"] = base
        params["dataset_mult"] = mult
        params["dataset_total_m"] = base * mult

    # Training method
    if "dpo" in run_name.lower():
        params["method"] = "dpo"
    elif "finetune" in run_name.lower():
        params["method"] = "finetune"

    # Special experiment markers
    if "--max_train_samples=" in run_name:
        sample_match = re.search(r"--max_train_samples=(\d+)", run_name)
        if sample_match:
            params["max_train_samples"] = int(sample_match.group(1))

    return params


def create_experiment_key(row, fixed_params):
    """Create a key for grouping experiments with same fixed parameters"""
    key_parts = []
    for param in fixed_params:
        value = row.get(param, "None")
        key_parts.append(f"{param}={value}")
    return "__".join(key_parts)


def analyze_sweep_dimension(df, swept_param, fixed_params):
    """Analyze sweeps for a specific dimension"""
    print(f"\n{'=' * 60}")
    print(f"ANALYZING {swept_param.upper()} SWEEPS")
    print(f"Fixed parameters: {fixed_params}")
    print(f"Swept parameter: {swept_param}")
    print(f"{'=' * 60}")

    # Filter to runs that have the swept parameter
    valid_df = df[df[swept_param].notna()]
    print(f"\nRuns with {swept_param} data: {len(valid_df)}/{len(df)}")

    if len(valid_df) == 0:
        print("No data available for this sweep analysis")
        return

    # Group by fixed parameters
    sweep_groups = defaultdict(list)

    for _, row in valid_df.iterrows():
        # Check if all fixed params are available
        if all(pd.notna(row.get(param, None)) for param in fixed_params):
            key = create_experiment_key(row, fixed_params)
            sweep_groups[key].append(row)

    print(f"\nExperimental groups found: {len(sweep_groups)}")

    # Analyze each sweep group
    valid_sweeps = []
    for i, (group_key, runs) in enumerate(sweep_groups.items()):
        swept_values = [run[swept_param] for run in runs]
        unique_swept_values = sorted(set(swept_values))

        # Only consider groups with multiple swept values (actual sweeps)
        if len(unique_swept_values) > 1:
            finished_count = sum(1 for run in runs if run.get("state") == "finished")

            print(f"\n  Group {i + 1}: {group_key}")
            print(f"    Total runs: {len(runs)}")
            print(f"    Finished runs: {finished_count}")
            print(f"    {swept_param} values: {unique_swept_values}")
            print(
                f"    Value counts: {pd.Series(swept_values).value_counts().to_dict()}"
            )

            if (
                finished_count >= 2
            ):  # At least 2 finished runs for meaningful comparison
                valid_sweeps.append(
                    {
                        "group_key": group_key,
                        "runs": runs,
                        "finished_runs": finished_count,
                        "swept_values": unique_swept_values,
                        "total_runs": len(runs),
                    }
                )

    print(f"\nValid sweep groups (≥2 finished runs): {len(valid_sweeps)}")

    return valid_sweeps


def main():
    print("=== EXPERIMENTAL SWEEP CLUSTERING ANALYSIS ===\n")

    store = WandBStore("postgresql+psycopg://localhost/wandb_test")
    runs_df = store.get_runs()

    # Extract hyperparameters
    param_data = []
    for _, row in runs_df.iterrows():
        params = extract_hyperparameters(row["run_name"])
        params.update(
            {
                "run_id": row["run_id"],
                "run_name": row["run_name"],
                "state": row["state"],
                "metadata_model_size": row.get("model_size", None),
                "metadata_lr": row.get("learning_rate", None),
            }
        )
        param_data.append(params)

    df = pd.DataFrame(param_data)

    print(f"Total runs: {len(df)}")
    print(f"Finished runs: {(df['state'] == 'finished').sum()}")

    # First split by method (major experimental paradigm difference)
    print(f"\n{'=' * 80}")
    print("SPLITTING BY TRAINING METHOD")
    print(f"{'=' * 80}")

    method_counts = df[df["method"].notna()]["method"].value_counts()
    print("\nTraining method distribution:")
    for method, count in method_counts.items():
        finished = df[(df["method"] == method) & (df["state"] == "finished")].shape[0]
        print(f"  {method}: {count} total, {finished} finished")

    # Analyze each method separately
    for method in ["finetune", "dpo"]:
        method_df = df[(df["method"] == method) & (df["state"] == "finished")]

        if len(method_df) == 0:
            continue

        print(f"\n{'=' * 80}")
        print(f"ANALYZING {method.upper()} EXPERIMENTS")
        print(f"{'=' * 80}")
        print(f"Finished {method} runs: {len(method_df)}")

        # 1. Learning Rate Sweeps (fix everything else, vary LR)
        lr_sweeps = analyze_sweep_dimension(
            method_df,
            swept_param="learning_rate",
            fixed_params=["model_size_m", "dataset_total_m"],
        )

        # 2. Model Size Sweeps (fix everything else, vary model size)
        model_sweeps = analyze_sweep_dimension(
            method_df,
            swept_param="model_size_m",
            fixed_params=["learning_rate", "dataset_total_m"],
        )

        # 3. Dataset Size Sweeps (fix everything else, vary dataset size)
        dataset_sweeps = analyze_sweep_dimension(
            method_df,
            swept_param="dataset_total_m",
            fixed_params=["model_size_m", "learning_rate"],
        )

        print(f"\n{'-' * 60}")
        print(f"SUMMARY FOR {method.upper()} METHOD:")
        print(f"{'-' * 60}")
        print(f"  Valid Learning Rate sweeps: {len(lr_sweeps) if lr_sweeps else 0}")
        print(f"  Valid Model Size sweeps: {len(model_sweeps) if model_sweeps else 0}")
        print(
            f"  Valid Dataset Size sweeps: {len(dataset_sweeps) if dataset_sweeps else 0}"
        )

        # Show most promising sweeps
        all_sweeps = []
        if lr_sweeps:
            for sweep in lr_sweeps:
                all_sweeps.append(("LR", sweep))
        if model_sweeps:
            for sweep in model_sweeps:
                all_sweeps.append(("Model", sweep))
        if dataset_sweeps:
            for sweep in dataset_sweeps:
                all_sweeps.append(("Dataset", sweep))

        if all_sweeps:
            # Sort by number of finished runs
            all_sweeps.sort(key=lambda x: x[1]["finished_runs"], reverse=True)

            print(f"\n  Top 3 most complete sweeps for {method}:")
            for i, (sweep_type, sweep) in enumerate(all_sweeps[:3]):
                print(
                    f"    {i + 1}. {sweep_type} sweep: {sweep['finished_runs']} finished runs"
                )
                print(f"       Fixed conditions: {sweep['group_key']}")
                print(f"       Values: {sweep['swept_values']}")

    # Look for special experiment types (data efficiency, etc.)
    print(f"\n{'=' * 80}")
    print("SPECIAL EXPERIMENT TYPES")
    print(f"{'=' * 80}")

    # Data efficiency experiments
    data_eff_df = df[df["max_train_samples"].notna() & (df["state"] == "finished")]
    if len(data_eff_df) > 0:
        print(f"\nData efficiency experiments: {len(data_eff_df)} finished runs")
        sample_counts = data_eff_df["max_train_samples"].value_counts().sort_index()
        print(f"Training sample sizes: {dict(sample_counts)}")

        # Group by other parameters to see the sweep structure
        if "model_size_m" in data_eff_df.columns:
            model_sample_crosstab = pd.crosstab(
                data_eff_df["model_size_m"], data_eff_df["max_train_samples"]
            )
            print("Model Size × Training Samples:")
            print(model_sample_crosstab)


if __name__ == "__main__":
    main()

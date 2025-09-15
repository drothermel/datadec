from collections import defaultdict

import pandas as pd

from datadec.wandb_eval import analysis_helpers
from datadec.wandb_eval import wandb_transforms as transforms
from datadec.wandb_eval.wandb_loader import WandBDataLoader

analysis_helpers.configure_pandas_display()


def create_experiment_key(row, fixed_params):
    key_parts = []
    for param in fixed_params:
        value = row.get(param, "None")
        key_parts.append(f"{param}={value}")
    return "__".join(key_parts)


def analyze_sweep_dimension(df, swept_param, fixed_params):
    print(f"\n{'=' * 60}")
    print(f"ANALYZING {swept_param.upper()} SWEEPS")
    print(f"Fixed parameters: {fixed_params}")
    print(f"Swept parameter: {swept_param}")
    print(f"{'=' * 60}")

    valid_df = df[df[swept_param].notna()]
    print(f"\nRuns with {swept_param} data: {len(valid_df)}/{len(df)}")

    if len(valid_df) == 0:
        print("No data available for this sweep analysis")
        return

    sweep_groups = defaultdict(list)

    for _, row in valid_df.iterrows():
        if all(pd.notna(row.get(param, None)) for param in fixed_params):
            key = create_experiment_key(row, fixed_params)
            sweep_groups[key].append(row)

    print(f"\nExperimental groups found: {len(sweep_groups)}")

    valid_sweeps = []
    for i, (group_key, runs) in enumerate(sweep_groups.items()):
        swept_values = [run[swept_param] for run in runs]
        unique_swept_values = sorted(set(swept_values))

        if len(unique_swept_values) > 1:
            finished_count = sum(1 for run in runs if run.get("state") == "finished")

            print(f"\n  Group {i + 1}: {group_key}")
            print(f"    Total runs: {len(runs)}")
            print(f"    Finished runs: {finished_count}")
            print(f"    {swept_param} values: {unique_swept_values}")
            print(
                f"    Value counts: {pd.Series(swept_values).value_counts().to_dict()}"
            )

            if finished_count >= 2:
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
                elif base_key == "data":
                    converted_params["data"] = value
                elif base_key == "params":
                    if isinstance(value, str) and value.endswith("M"):
                        converted_params["model_size_m"] = int(value[:-1])
                elif base_key == "total_tok":
                    converted_params["dataset_total_m"] = value
                else:
                    converted_params[base_key] = value
            else:
                converted_params[key] = value

        converted_params.update(
            {
                "run_id": row["run_id"],
                "run_name": row["run_name"],
                "state": row["state"],
                "metadata_model_size": row.get("model_size", None),
                "metadata_lr": row.get("learning_rate", None),
            }
        )
        param_data.append(converted_params)

    df = pd.DataFrame(param_data)

    print(f"Total runs: {len(df)}")
    print(f"Finished runs: {(df['state'] == 'finished').sum()}")

    print(f"\n{'=' * 80}")
    print("SPLITTING BY TRAINING METHOD")
    print(f"{'=' * 80}")

    method_counts = df[df["method"].notna()]["method"].value_counts()
    print("\nTraining method distribution:")
    for method, count in method_counts.items():
        finished = df[(df["method"] == method) & (df["state"] == "finished")].shape[0]
        print(f"  {method}: {count} total, {finished} finished")

    for method in ["finetune", "dpo"]:
        method_df = df[(df["method"] == method) & (df["state"] == "finished")]

        if len(method_df) == 0:
            continue

        print(f"\n{'=' * 80}")
        print(f"ANALYZING {method.upper()} EXPERIMENTS")
        print(f"{'=' * 80}")
        print(f"Finished {method} runs: {len(method_df)}")

        lr_sweeps = analyze_sweep_dimension(
            method_df,
            swept_param="learning_rate",
            fixed_params=["model_size_m", "dataset_total_m"],
        )

        model_sweeps = analyze_sweep_dimension(
            method_df,
            swept_param="model_size_m",
            fixed_params=["learning_rate", "dataset_total_m"],
        )

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
            all_sweeps.sort(key=lambda x: x[1]["finished_runs"], reverse=True)

            print(f"\n  Top 3 most complete sweeps for {method}:")
            for i, (sweep_type, sweep) in enumerate(all_sweeps[:3]):
                print(
                    f"    {i + 1}. {sweep_type} sweep: {sweep['finished_runs']} finished runs"
                )
                print(f"       Fixed conditions: {sweep['group_key']}")
                print(f"       Values: {sweep['swept_values']}")

    print(f"\n{'=' * 80}")
    print("SPECIAL EXPERIMENT TYPES")
    print(f"{'=' * 80}")

    data_eff_df = df[df["max_train_samples"].notna() & (df["state"] == "finished")]
    if len(data_eff_df) > 0:
        print(f"\nData efficiency experiments: {len(data_eff_df)} finished runs")
        sample_counts = data_eff_df["max_train_samples"].value_counts().sort_index()
        print(f"Training sample sizes: {dict(sample_counts)}")

        if "model_size_m" in data_eff_df.columns:
            model_sample_crosstab = pd.crosstab(
                data_eff_df["model_size_m"], data_eff_df["max_train_samples"]
            )
            print("Model Size × Training Samples:")
            print(model_sample_crosstab)


if __name__ == "__main__":
    main()

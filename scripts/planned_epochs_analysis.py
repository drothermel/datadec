#!/usr/bin/env python3

import pandas as pd

from datadec import analysis_helpers

COMPARISON_COLS = [
    "run_id",
    "method",
    "planned_epochs_meta",
    "expected_steps",
    "actual_steps",
    "step_ratio",
]


def calculate_expected_steps(
    planned_epochs: float, dataset_tokens: int = None, batch_size: int = None
) -> dict:
    """Calculate expected training steps based on epochs and dataset info."""
    calculations = {}

    if dataset_tokens and batch_size:
        # Simple calculation: dataset_tokens / batch_size * epochs
        steps_per_epoch = (
            dataset_tokens * 1_000_000 / batch_size
        )  # Convert M tokens to tokens
        expected_steps = int(steps_per_epoch * planned_epochs)
        calculations["expected_steps"] = expected_steps
        calculations["steps_per_epoch"] = int(steps_per_epoch)
        calculations["calculation_method"] = "dataset_tokens / batch_size * epochs"

    return calculations


def main():
    print("=== PLANNED EPOCHS ANALYSIS ===\n")

    runs_df, history_df = analysis_helpers.load_runs_and_history_df()
    planned_epoch_runs = runs_df[runs_df["num_train_epochs"].notna()].copy()
    print(f"Total runs in database: {len(runs_df)}")
    print(f"Runs with planned epochs in metadata: {len(planned_epoch_runs)}")
    if len(planned_epoch_runs) == 0:
        print("No runs found with planned epochs!")
        return
    print("\nPlanned epoch distribution:")
    epoch_counts = planned_epoch_runs["num_train_epochs"].value_counts().sort_index()
    for epochs, count in epoch_counts.items():
        print(f"  {epochs} epochs: {count} runs")

    print(f"\n{'=' * 80}")
    print("ANALYZING EPOCH PARAMETERS FROM RUN NAMES")
    print(f"{'=' * 80}")
    analysis_data = []
    for _, row in planned_epoch_runs.iterrows():
        params = analysis_helpers.extract_hyperparameters(row["run_name"])
        run_history = history_df[history_df["run_id"] == row["run_id"]]
        actual_steps = len(run_history) if len(run_history) > 0 else None
        analysis_record = {
            "run_id": row["run_id"],
            "run_name": row["run_name"],
            "planned_epochs_meta": row["num_train_epochs"],
            "method": params.get("method"),
            "state": row["state"],
            "actual_steps": actual_steps,
            "epochs_from_name": params.get("epochs_from_name"),
            "dataset_tokens_from_name": params.get("dataset_tokens_m"),
            "mtx_format": params.get("mtx_format", False),
        }
        if params.get("dataset_tokens_m") and "batch_size_from_name" in params:
            expectations = calculate_expected_steps(
                row["num_train_epochs"],
                params["dataset_tokens_m"],
                params["batch_size_from_name"],
            )
            analysis_record.update(expectations)

        analysis_data.append(analysis_record)
    analysis_df = pd.DataFrame(analysis_data)

    print("\nParameter extraction summary:")
    print(
        f"  Epochs found in run names: {analysis_df.get('epochs_from_name', pd.Series()).notna().sum()}"
    )
    print(
        f"  Max steps found in run names: {analysis_df.get('max_steps_from_name', pd.Series()).notna().sum()}"
    )
    print(
        f"  Batch size found in run names: {analysis_df.get('batch_size_from_name', pd.Series()).notna().sum()}"
    )
    print(
        f"  Dataset tokens found in run names: {analysis_df.get('dataset_tokens_from_name', pd.Series()).notna().sum()}"
    )
    print(f"  Runs with training history: {analysis_df['actual_steps'].notna().sum()}")
    print(
        f"  Expected steps calculated: {analysis_df.get('expected_steps', pd.Series()).notna().sum()}"
    )

    if "mtx_format" in analysis_df.columns:
        mtx_count = analysis_df["mtx_format"].sum()
        print("\nMtx format usage:")
        print(f"  Runs using Mtx format: {mtx_count}")

    if "epochs_from_name" in analysis_df.columns:
        name_vs_meta_epochs = analysis_df[analysis_df["epochs_from_name"].notna()]
        if len(name_vs_meta_epochs) > 0:
            print(f"\n{'=' * 60}")
            print("EPOCHS: METADATA vs RUN NAME COMPARISON")
            print(f"{'=' * 60}")

            consistent = (
                name_vs_meta_epochs["planned_epochs_meta"]
                == name_vs_meta_epochs["epochs_from_name"]
            ).sum()
            print(
                f"Runs with consistent epoch values: {consistent}/{len(name_vs_meta_epochs)}"
            )

            if consistent < len(name_vs_meta_epochs):
                inconsistent = name_vs_meta_epochs[
                    name_vs_meta_epochs["planned_epochs_meta"]
                    != name_vs_meta_epochs["epochs_from_name"]
                ]
                print("\nInconsistent epoch values found:")
                for _, row in inconsistent.head(5).iterrows():
                    print(
                        f"  Metadata: {row['planned_epochs_meta']}, Name: {row['epochs_from_name']} | {row['run_name'][:60]}..."
                    )

    if "expected_steps" in analysis_df.columns:
        calculable_runs = analysis_df[
            analysis_df["expected_steps"].notna() & analysis_df["actual_steps"].notna()
        ]
    else:
        calculable_runs = pd.DataFrame()
    if len(calculable_runs) > 0:
        print(f"\n{'=' * 60}")
        print("EXPECTED vs ACTUAL STEPS ANALYSIS")
        print(f"{'=' * 60}")
        calculable_runs = calculable_runs.copy()
        calculable_runs["step_ratio"] = (
            calculable_runs["actual_steps"] / calculable_runs["expected_steps"]
        )
        calculable_runs["step_difference"] = (
            calculable_runs["actual_steps"] - calculable_runs["expected_steps"]
        )
        print(f"Runs with calculable expected steps: {len(calculable_runs)}")
        print(
            f"Average step ratio (actual/expected): {calculable_runs['step_ratio'].mean():.3f}"
        )
        print(
            f"Step ratio range: {calculable_runs['step_ratio'].min():.3f} to {calculable_runs['step_ratio'].max():.3f}"
        )
        display_data = calculable_runs[COMPARISON_COLS].to_dict("records")

        custom_widths = {"run_id": 40, "step_ratio": 10}
        analysis_helpers.print_dynamics_summary_table(
            display_data, columns=COMPARISON_COLS, column_widths=custom_widths
        )

        well_matched = calculable_runs[
            (calculable_runs["step_ratio"] >= 0.9)
            & (calculable_runs["step_ratio"] <= 1.1)
        ]
        print(
            f"\nRuns with good step prediction (90-110% of expected): {len(well_matched)}"
        )
        if len(well_matched) < len(calculable_runs):
            poorly_matched = calculable_runs[
                (calculable_runs["step_ratio"] < 0.9)
                | (calculable_runs["step_ratio"] > 1.1)
            ]
            print(f"Runs with poor step prediction: {len(poorly_matched)}")

    print(f"\n{'=' * 60}")
    print("ANALYSIS BY TRAINING METHOD")
    print(f"{'=' * 60}")

    agg_dict = {
        "run_id": "count",
        "planned_epochs_meta": ["mean", "min", "max"],
        "actual_steps": lambda x: x.notna().sum(),
    }
    if "expected_steps" in analysis_df.columns:
        agg_dict["expected_steps"] = lambda x: x.notna().sum()
    method_summary = analysis_df.groupby("method").agg(agg_dict).round(2)

    print("\nSummary by method:")
    print(method_summary)
    for method in analysis_df["method"].dropna().unique():
        method_runs = analysis_df[analysis_df["method"] == method]
        print(
            f"\n{method.upper()} runs with planned epochs ({len(method_runs)} total):"
        )
        sample_runs = method_runs.head(3)
        for _, row in sample_runs.iterrows():
            epochs = row["planned_epochs_meta"]
            steps = (
                row["actual_steps"] if pd.notna(row["actual_steps"]) else "No history"
            )
            print(f"  {epochs} epochs | {steps} steps | {row['run_name'][:60]}...")


if __name__ == "__main__":
    main()

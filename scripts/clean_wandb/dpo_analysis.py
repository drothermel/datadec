#!/usr/bin/env python3

import pandas as pd

from datadec import analysis_helpers

TRAIN_SUMMARY_COLUMNS = [
    "run_id",
    "total_steps",
    "max_lr",
    "min_train_loss",
    "initial_train_loss",
    "final_train_loss",
    "loss_improvement",
]
DISPLAY_COLS = [
    "run_name",
    "state",
    "learning_rate",
    "model_size_m",
    "dataset_total_m",
    "method",
]


def main():
    print("=== DPO TRAINING ANALYSIS ===\n")

    runs_df, history_df = analysis_helpers.load_runs_and_history_df()
    dpo_runs_df = analysis_helpers.filter_runs_by_method(runs_df, "dpo")
    print(f"Total runs in database: {len(runs_df)}")
    print(f"DPO runs found: {len(dpo_runs_df)}")
    if len(dpo_runs_df) == 0:
        print("No DPO runs found!")
        return
    dpo_summary = analysis_helpers.create_experimental_summary(runs_df, method="dpo")

    print("DPO runs by state:")
    print(dpo_runs_df["state"].value_counts())
    print(
        f"\n{analysis_helpers.format_experimental_summary(dpo_summary, method='dpo')}"
    )
    print(f"\n{'=' * 60}")
    print("DPO RUN DETAILS")
    print(f"{'=' * 60}")

    available_cols = [col for col in DISPLAY_COLS if col in dpo_runs_df.columns]
    if len(available_cols) > 0:
        print("\nDPO runs with extracted parameters:")
        for idx, (_, row) in enumerate(dpo_runs_df.iterrows(), 1):
            print(f"\n{idx}. Run: {row['run_name'][:80]}...")
            for col in available_cols[1:]:
                if col in row and pd.notna(row[col]):
                    print(f"   {col}: {row[col]}")

    print(f"\n{'=' * 60}")
    print("DPO TRAINING DYNAMICS")
    print(f"{'=' * 60}")
    dpo_run_ids = dpo_runs_df["run_id"].tolist()
    available_history_runs = history_df["run_id"].unique()
    dpo_runs_with_history = [
        rid for rid in dpo_run_ids if rid in available_history_runs
    ]
    print(f"\nDPO runs with training history: {len(dpo_runs_with_history)}")
    if len(dpo_runs_with_history) == 0:
        print("No DPO runs found with training history data.")
        return
    dpo_dynamics_df = analysis_helpers.analyze_training_progression(
        history_df, dpo_runs_with_history
    )
    if len(dpo_dynamics_df) > 0:
        print(f"\nTraining dynamics summary ({len(dpo_dynamics_df)} runs):")
        dynamics_list = dpo_dynamics_df.to_dict("records")
        column_widths = {"run_id": 32}
        analysis_helpers.print_dynamics_summary_table(
            dynamics_list,
            columns=TRAIN_SUMMARY_COLUMNS,
            column_widths=column_widths,
        )
    if len(dpo_runs_with_history) > 0:
        print(f"\n{'=' * 60}")
        print("DETAILED DPO RUN ANALYSIS (FIRST RUN)")
        print(f"{'=' * 60}")
        sample_run_id = dpo_runs_with_history[0]
        analysis_helpers.print_training_history_sample(sample_run_id, history_df)


if __name__ == "__main__":
    main()

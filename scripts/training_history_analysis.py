import random

from datadec import analysis_helpers

RANDOM_SEED = 42

TRAIN_SUMMARY_COLUMNS = [
    "run_id",
    "total_steps",
    "max_lr",
    "final_lr",
    "max_epoch",
    "final_train_loss",
    "max_tokens",
]


def main():
    print("=== TRAINING HISTORY ANALYSIS ===\n")

    history_df = analysis_helpers.load_history_df()
    available_runs = history_df["run_id"].unique()
    random.seed(RANDOM_SEED)

    print(f"\nRuns with training history: {len(available_runs)}")
    print(f"Total history records: {len(history_df)}")
    print(f"History data shape: {history_df.shape}")
    analysis_helpers.print_dataframe_coverage(history_df, "All history columns")

    sample_runs = random.sample(list(available_runs), min(5, len(available_runs)))
    dynamics_results = analysis_helpers.analyze_training_progression(
        history_df, sample_runs
    )
    assert len(dynamics_results) > 0, "No dynamics results found"

    print(f"\nAnalyzing {len(sample_runs)} sample runs:")
    print("=" * 120)
    dynamics_list = dynamics_results.to_dict("records")
    columns = TRAIN_SUMMARY_COLUMNS
    analysis_helpers.print_dynamics_summary_table(dynamics_list, columns=columns)
    print(f"\n{'=' * 60}")
    print("DETAILED ANALYSIS OF FIRST SAMPLE RUN")
    print(f"{'=' * 60}")
    sample_run_id = dynamics_results.iloc[0]["run_id"]
    analysis_helpers.print_training_history_sample(sample_run_id, history_df)


if __name__ == "__main__":
    main()

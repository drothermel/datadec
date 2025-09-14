import pandas as pd

from datadec.wandb_eval import analysis_helpers
from datadec.wandb_eval import wandb_transforms as transforms
from datadec.wandb_eval.wandb_loader import WandBDataLoader

analysis_helpers.configure_pandas_display()


def main():
    print("=== EPOCHS ANALYSIS ===\n")

    loader = WandBDataLoader()
    runs_df, history_df = loader.load_runs_and_history()

    print(f"Total runs in database: {len(runs_df)}")
    print(f"Total history records: {len(history_df)}")

    runs_with_epochs_meta = runs_df[runs_df["num_train_epochs"].notna()]
    print(f"\nRuns with epochs in metadata: {len(runs_with_epochs_meta)}")

    if len(runs_with_epochs_meta) > 0:
        print(
            f"Unique epoch values in metadata: {sorted(runs_with_epochs_meta['num_train_epochs'].unique())}"
        )

    history_with_epochs = history_df[history_df["epoch"].notna()]
    runs_with_epochs_history = history_with_epochs["run_id"].unique()

    print(f"Runs with epochs in training history: {len(runs_with_epochs_history)}")
    print(f"History records with epoch data: {len(history_with_epochs)}")

    if len(runs_with_epochs_history) > 0:
        epoch_stats = (
            history_with_epochs.groupby("run_id")["epoch"]
            .agg(["min", "max", "count"])
            .reset_index()
        )
        epoch_stats.columns = ["run_id", "min_epoch", "max_epoch", "epoch_records"]

        print("\nEpoch range summary:")
        print(f"  Min epoch across all runs: {epoch_stats['min_epoch'].min():.3f}")
        print(f"  Max epoch across all runs: {epoch_stats['max_epoch'].max():.3f}")
        print(f"  Average records per run: {epoch_stats['epoch_records'].mean():.1f}")

        max_epochs_distribution = epoch_stats["max_epoch"].value_counts().sort_index()
        print("\nDistribution of maximum epochs reached:")
        for epoch_val, count in max_epochs_distribution.head(10).items():
            print(f"  {epoch_val:.3f}: {count} runs")
        if len(max_epochs_distribution) > 10:
            print(f"  ... and {len(max_epochs_distribution) - 10} more unique values")

        print(f"\n{'=' * 60}")
        print("DETAILED ANALYSIS OF RUNS WITH EPOCH DATA")
        print(f"{'=' * 60}")

        sample_runs = list(runs_with_epochs_history)[:5]

        if len(sample_runs) > 0:
            dynamics_results = analysis_helpers.analyze_training_progression(
                history_df, sample_runs
            )

            if len(dynamics_results) > 0:
                print(
                    f"\nTraining dynamics for runs with epoch data ({len(dynamics_results)} runs):"
                )

                dynamics_list = dynamics_results.to_dict("records")
                columns = [
                    "run_id",
                    "total_steps",
                    "max_epoch",
                    "max_lr",
                    "final_train_loss",
                    "loss_improvement",
                ]
                analysis_helpers.print_dynamics_summary_table(
                    dynamics_list, columns=columns
                )

                print(f"\n{'=' * 60}")
                print("DETAILED EPOCH PROGRESSION (FIRST RUN)")
                print(f"{'=' * 60}")

                sample_run_id = sample_runs[0]
                analysis_helpers.print_training_history_sample(
                    sample_run_id, history_df
                )

                run_history = history_df[
                    history_df["run_id"] == sample_run_id
                ].sort_values("step")
                if (
                    "epoch" in run_history.columns
                    and run_history["epoch"].notna().any()
                ):
                    epoch_values = run_history["epoch"].dropna()
                    print("\nEpoch progression details:")
                    print(f"  Total epoch records: {len(epoch_values)}")
                    print(
                        f"  Epoch range: {epoch_values.min():.3f} to {epoch_values.max():.3f}"
                    )
                    print("  Epoch increment pattern (first 10 steps):")

                    epoch_sample = run_history[["step", "epoch"]].dropna().head(10)
                    if len(epoch_sample) > 0:
                        print(epoch_sample.to_string(index=False))

        print(f"\n{'=' * 60}")
        print("EPOCH DATA BY TRAINING METHOD")
        print(f"{'=' * 60}")

        runs_with_epoch_methods = []
        for run_id in runs_with_epochs_history:
            run_row = runs_df[runs_df["run_id"] == run_id]
            if len(run_row) > 0:
                run_name = run_row["run_name"].iloc[0]
                params = transforms.extract_hyperparameters(run_name)
                method = None
                if "method_rnp" in params:
                    method = params["method_rnp"]
                if method:
                    epoch_data = epoch_stats[epoch_stats["run_id"] == run_id].iloc[0]
                    runs_with_epoch_methods.append(
                        {
                            "run_id": run_id,
                            "method": method,
                            "max_epoch": epoch_data["max_epoch"],
                            "epoch_records": epoch_data["epoch_records"],
                        }
                    )

        if runs_with_epoch_methods:
            method_epoch_df = pd.DataFrame(runs_with_epoch_methods)
            method_summary = (
                method_epoch_df.groupby("method")
                .agg(
                    {
                        "run_id": "count",
                        "max_epoch": ["mean", "min", "max"],
                        "epoch_records": "mean",
                    }
                )
                .round(3)
            )

            print("\nEpoch statistics by training method:")
            print(method_summary)

            method_counts = method_epoch_df["method"].value_counts()
            print("\nRuns with epoch data by method:")
            for method, count in method_counts.items():
                print(f"  {method}: {count} runs")

    else:
        print("\nNo runs found with epoch data in training history.")


if __name__ == "__main__":
    main()

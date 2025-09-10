import pandas as pd

from datadec import analysis_helpers

PARAMS_TO_DROP = [
    "run_date_rnp",
    "run_time_rnp",
]

DYNAMICS_TO_KEEP = [
    "max_step",
    "max_tokens",
    "max_epoch",
    "max_lr",
    "final_train_loss",
    "min_train_loss",
]
# parsing.HISTORY_DROP for cols to drop


def create_unified_parsed_df(sample_runs):
    all_runs_data = []
    runs_df, history_df = analysis_helpers.load_df()

    for _, row in sample_runs.iterrows():
        run_name = row["run_name"]
        run_id = row["run_id"]

        # Existing parsing
        parsed_params = analysis_helpers.extract_hyperparameters(run_name)

        # Add training dynamics
        dynamics = analysis_helpers.get_run_training_dynamics(history_df, run_id)

        run_data = {"run_name": run_name}
        run_data.update(
            {k: v for k, v in parsed_params.items() if k not in PARAMS_TO_DROP}
        )

        if dynamics:
            filtered_dynamics = {
                k: v for k, v in dynamics.items() if k in DYNAMICS_TO_KEEP
            }
            run_data.update(filtered_dynamics)

        all_runs_data.append(run_data)

    return pd.DataFrame(all_runs_data)


def main():
    print("=== RUN NAME PARSING SAMPLE ===\n")
    sample_runs = analysis_helpers.load_random_run_sample(20)
    print(f"Showing parsing results for {len(sample_runs)} random runs:\n")

    unified_df = create_unified_parsed_df(sample_runs)
    print(unified_df)
    print(f"\nDataFrame shape: {unified_df.shape}")


if __name__ == "__main__":
    main()

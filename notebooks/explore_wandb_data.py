# %%



# %%

%load_ext autoreload
%autoreload 2




































from datadec import wandb_store
from datadec.wandb_eval import wandb_constants as wconsts
from datadec import analysis_helpers
import pandas as pd

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

HISTORY_NaN_OR_CONST = [
    'training_step', # verified: NaN
#    'runtime', # verified: 0
]
HISTORY_DPO_ONLY = [
    "rewards/chosen",
    "rewards/margin",
    "rewards/average",
    "rewards/accuracy",
    "rewards/rejected"
]
HISTORY_DROP = [
    *HISTORY_NaN_OR_CONST,
    *HISTORY_DPO_ONLY,
    "run_name",
    "total_tokens_including_padding", # keep total tokens instead
    "per_device_tps",
    "per_device_tps_including_padding",
]
# run_id == run_name
HISTORY_KEEP = [
    "project",
    "run_id",
    # x axis
    "timestamp",
    "step",
    "epoch",
    "total_tokens",
    "learning_rate",
    # y axis
    "logps/chosen",
    "logps/rejected",
    "train_loss",
]

# %%
def filter_early_test_runs(df):
    key = "created_at" if "created_at" in df.columns else "timestamp"
    return df[df[key] >= "2025-08-21"]

# %%

runs_df, history_df = analysis_helpers.load_df()
print("Runs Df:",runs_df.shape)
print("History Df:",history_df.shape)
# %%
hist_cols = set(history_df.columns)
my_hist_cols = set(HISTORY_KEEP+HISTORY_DROP)
print("History cols: ", len(hist_cols))
print("My history cols: ", len(my_hist_cols))
print("Difference: ", hist_cols - my_hist_cols)
print("Difference: ", my_hist_cols - hist_cols)
print(hist_cols)
print(my_hist_cols)

#%%

for col in runs_df.columns:
    try:
        unique = runs_df[col].unique()
        if len(unique) <= 2:
            print(col, len(unique), unique[:5])
    except:
        print(col, runs_df[col])


#%%

print("Runs initial: ", runs_df.shape)
runs_df = filter_early_test_runs(runs_df)
print("Runs after filtering: ", runs_df.shape)
print("History initial: ", history_df.shape)
history_df = filter_early_test_runs(history_df)
print("History after filtering: ", history_df.shape)



print()
runs_rids = set(runs_df["run_id"].unique())
history_rids = set(history_df["run_id"].unique())
print("Runs rids: ", len(runs_rids))
print("History rids: ", len(history_rids))
print("runs not history: ", len(runs_rids - history_rids))
print("history not runs: ", len(history_rids - runs_rids))

# Max step -> step
# Max training_step
# Max tokens -> total_tokens
# Max lr -> learning_rate
# Max epoch -> epoch
# Min train loss
# Final train loss

# %%
runs_df[wconsts.CORE_RUN_FIELDS + ['training_step']]
# %%
print("History: ", history_df.shape)
history_df[list(wconsts.CORE_HISTORY_FIELDS)]
# %%

#%%
runs_df.dtypes
# %%
history_df.dtypes
# %%
metadata_cols = runs_df.columns
history_cols = history_df.columns
for col in metadata_cols:
    print(col)
for col in history_cols:
    print(col)
# %%


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
    # print(unified_df)
    # print(f"\nDataFrame shape: {unified_df.shape}")
    return unified_df


# %%
unified_df = main()
unified_df

# %%
if __name__ == "__main__":
    unified_df = main()
    print(unified_df)

# %%

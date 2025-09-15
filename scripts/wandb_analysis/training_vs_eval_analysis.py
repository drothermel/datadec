import pandas as pd

from datadec.wandb_eval import analysis_helpers
from datadec.wandb_eval.wandb_loader import WandBDataLoader

analysis_helpers.configure_pandas_display()

loader = WandBDataLoader()
runs_df, history_df = loader.load_runs_and_history()

print("=== TRAINING vs EVAL-ONLY RUN ANALYSIS ===\n")

print("1. TRAINING INDICATORS ANALYSIS")
print("=" * 50)

training_indicators = [
    "max_train_steps",
    "num_train_epochs",
    "learning_rate",
    "warmup_ratio",
    "gradient_accumulation_steps",
    "per_device_train_batch_size",
]

print("\nA. Training Configuration Presence:")
print("-" * 40)
for indicator in training_indicators:
    if indicator in runs_df.columns:
        has_values = runs_df[indicator].notna().sum()
        non_zero_values = (
            (runs_df[indicator] > 0).sum()
            if runs_df[indicator].dtype in ["float64", "int64"]
            else has_values
        )
        print(
            f"  {indicator}: {has_values} runs have values, {non_zero_values} are > 0"
        )

print("\nB. Training History Presence:")
print("-" * 40)
runs_with_history = set(history_df["run_id"].unique())
runs_without_history = set(runs_df["run_id"]) - runs_with_history

print(f"Runs WITH training history: {len(runs_with_history)}")
print(f"Runs WITHOUT training history: {len(runs_without_history)}")
print(f"History coverage: {len(runs_with_history) / len(runs_df) * 100:.1f}%")

if len(runs_without_history) > 0:
    print("\nSample runs WITHOUT training history:")
    no_history_runs = runs_df[runs_df["run_id"].isin(list(runs_without_history))]
    for i, (_, run) in enumerate(no_history_runs.head(5).iterrows()):
        print(f"  {i + 1}. {run['run_name'][:80]}")

print("\n2. RUN TYPE CLASSIFICATION")
print("=" * 50)


def classify_run_type(row, has_history):
    name = str(row["run_name"]).lower()

    has_training_config = any(
        [
            pd.notna(row.get("max_train_steps", None))
            and row.get("max_train_steps", 0) > 0,
            pd.notna(row.get("num_train_epochs", None))
            and row.get("num_train_epochs", 0) > 0,
            pd.notna(row.get("learning_rate", None))
            and row.get("learning_rate", 0) > 0,
        ]
    )

    if "eval" in name and not any(x in name for x in ["finetune", "dpo", "train"]):
        return "Eval-Only"
    elif has_history and has_training_config:
        return "Training"
    elif has_history and not has_training_config:
        return "Training (minimal config)"
    elif not has_history and has_training_config:
        return "Training (config only)"
    elif not has_history and not has_training_config:
        return "Eval-Only"
    else:
        return "Unknown"


run_types = []
for _, row in runs_df.iterrows():
    has_history = row["run_id"] in runs_with_history
    run_type = classify_run_type(row, has_history)
    run_types.append(run_type)

runs_df["inferred_type"] = run_types

print("\nA. Inferred Run Type Distribution:")
print("-" * 40)
type_counts = pd.Series(run_types).value_counts()
for run_type, count in type_counts.items():
    print(f"  {run_type}: {count} runs ({count / len(runs_df) * 100:.1f}%)")

print("\n3. DETAILED ANALYSIS BY TYPE")
print("=" * 50)

print("\nA. Training Runs Analysis:")
print("-" * 40)
training_runs = runs_df[runs_df["inferred_type"].str.contains("Training")]
if len(training_runs) > 0:
    print(f"Total training runs: {len(training_runs)}")
    print(
        f"Mean training steps (where available): {training_runs['max_train_steps'].mean():.0f}"
    )
    print(
        f"Training run success rate: {(training_runs['state'] == 'finished').sum()}/{len(training_runs)} ({(training_runs['state'] == 'finished').sum() / len(training_runs) * 100:.1f}%)"
    )

print("\nB. Eval-Only Runs Analysis:")
print("-" * 40)
eval_runs = runs_df[runs_df["inferred_type"] == "Eval-Only"]
if len(eval_runs) > 0:
    print(f"Total eval-only runs: {len(eval_runs)}")
    print(
        f"Eval-only success rate: {(eval_runs['state'] == 'finished').sum()}/{len(eval_runs)} ({(eval_runs['state'] == 'finished').sum() / len(eval_runs) * 100:.1f}%)"
    )
    print("\nSample eval-only run names:")
    for i, name in enumerate(eval_runs["run_name"].head(5)):
        print(f"  {i + 1}. {name[:80]}")

print("\n4. TRAINING STEPS ANALYSIS")
print("=" * 50)

print("\nA. Training Steps Distribution:")
print("-" * 40)
if "max_train_steps" in runs_df.columns:
    steps_data = runs_df[
        runs_df["max_train_steps"].notna() & (runs_df["max_train_steps"] > 0)
    ]["max_train_steps"]
    if len(steps_data) > 0:
        print(f"Runs with training steps configured: {len(steps_data)}")
        print(f"Mean steps: {steps_data.mean():.0f}")
        print(f"Median steps: {steps_data.median():.0f}")
        print(f"Range: {steps_data.min():.0f} to {steps_data.max():.0f}")

        step_ranges = pd.cut(
            steps_data,
            bins=[0, 1000, 5000, 10000, float("inf")],
            labels=["≤1K", "1K-5K", "5K-10K", ">10K"],
        )
        print("\nStep ranges distribution:")
        for range_name, count in step_ranges.value_counts().items():
            print(f"  {range_name}: {count} runs")

print("\nB. Actual Training Length (from history):")
print("-" * 40)
if len(history_df) > 0:
    actual_steps = history_df.groupby("run_id")["step"].max()
    print(f"Runs with actual training history: {len(actual_steps)}")
    print(f"Mean actual steps: {actual_steps.mean():.0f}")
    print(f"Median actual steps: {actual_steps.median():.0f}")
    print(f"Range: {actual_steps.min():.0f} to {actual_steps.max():.0f}")

print("\n5. RUN NAME PATTERN ANALYSIS")
print("=" * 50)

print("\nA. Name Pattern Classification:")
print("-" * 40)
name_patterns = {}
for name in runs_df["run_name"]:
    name_lower = str(name).lower()
    if "eval" in name_lower and not any(
        x in name_lower for x in ["finetune", "dpo", "train"]
    ):
        pattern = "eval-only"
    elif "test" in name_lower:
        pattern = "test"
    elif "finetune" in name_lower:
        pattern = "finetune"
    elif "dpo" in name_lower:
        pattern = "dpo"
    else:
        pattern = "other"

    name_patterns[pattern] = name_patterns.get(pattern, 0) + 1

for pattern, count in sorted(name_patterns.items()):
    print(f"  {pattern}: {count} runs")

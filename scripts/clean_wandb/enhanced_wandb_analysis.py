#!/usr/bin/env python3

from datadec import WandBStore
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 100)

store = WandBStore("postgresql+psycopg://localhost/wandb_test")

print("=== ENHANCED WandB Data Analysis for Plotting Strategy ===\n")

runs_df = store.get_runs()
history_df = store.get_history()

print("=" * 60)
print("1. SCALING DIMENSION ANALYSIS")
print("=" * 60)

print("\nA. Model Size Scaling Pattern:")
print("-" * 40)
# Handle mixed data types in model_size column
model_size_clean = pd.to_numeric(runs_df["model_size"], errors='coerce')
model_sizes = model_size_clean.value_counts().sort_index()
print("Model sizes (parameters) and run counts:")
for size, count in model_sizes.items():
    if pd.notna(size):
        print(f"  {size:,.0f} parameters: {count} runs")

print("\nB. Learning Rate Distribution Analysis:")
print("-" * 40)
lr_data = runs_df[runs_df["learning_rate"] > 0]["learning_rate"]
if len(lr_data) > 0:
    print(f"Learning rates range: {lr_data.min():.2e} to {lr_data.max():.2e}")
    print(f"Unique learning rates: {lr_data.nunique()}")

    print("\nLearning rate groups:")
    lr_groups = lr_data.apply(lambda x: f"{x:.0e}" if x >= 1e-5 else f"{x:.1e}")
    lr_group_counts = lr_groups.value_counts().sort_index()
    for lr_group, count in lr_group_counts.head(10).items():
        print(f"  ~{lr_group}: {count} runs")

print("\n" + "=" * 60)
print("2. TRAINING METHOD ANALYSIS")
print("=" * 60)

print("\nA. Run Type Distribution:")
print("-" * 40)
run_types = []
for name in runs_df["run_name"]:
    if "dpo" in name.lower():
        run_types.append("DPO")
    elif "finetune" in name.lower():
        run_types.append("Finetune")
    elif "test" in name.lower():
        run_types.append("Test")
    else:
        run_types.append("Other")

run_type_counts = pd.Series(run_types).value_counts()
for rtype, count in run_type_counts.items():
    print(f"  {rtype}: {count} runs")

print("\nB. Model Size vs Training Method:")
print("-" * 40)
runs_with_types = runs_df.copy()
runs_with_types["run_type"] = run_types

# Cross-tabulation of model size and run type
# Use cleaned numeric model_size
runs_with_types["model_size_clean"] = pd.to_numeric(runs_with_types["model_size"], errors='coerce')
crosstab = pd.crosstab(runs_with_types["model_size_clean"], runs_with_types["run_type"])
print(crosstab)

print("\n" + "=" * 60)
print("3. EVALUATION METRICS ANALYSIS")
print("=" * 60)

eval_cols = [
    col for col in runs_df.columns if "pretrain_eval" in col or "oe_eval" in col
]
print(f"\nTotal evaluation metrics available: {len(eval_cols)}")

print("\nA. Key Evaluation Categories:")
print("-" * 40)
eval_categories = {}
for col in eval_cols:
    if "perplexity" in col.lower():
        category = "Perplexity"
    elif any(
        task in col.lower()
        for task in ["arc", "boolq", "piqa", "hellaswag", "csqa", "mmlu"]
    ):
        category = "Reasoning Tasks"
    elif "olmes" in col.lower():
        category = "OLMES Suite"
    else:
        category = "Other"

    eval_categories[category] = eval_categories.get(category, 0) + 1

for category, count in sorted(eval_categories.items()):
    print(f"  {category}: {count} metrics")

print("\nB. Sample Key Metrics (with data):")
print("-" * 40)
key_metrics = [
    "pretrain_eval_olmes_10_macro_avg_acc_raw",
    "pretrain_eval/pile-validation/Perplexity",
    "pretrain_eval_arc_challenge_acc_raw",
    "pretrain_eval_mmlu_acc_raw",
]

for metric in key_metrics:
    if metric in runs_df.columns:
        non_null_count = runs_df[metric].notna().sum()
        if non_null_count > 0:
            mean_val = runs_df[metric].mean()
            print(f"  {metric}: {non_null_count} runs, mean={mean_val:.3f}")

print("\n" + "=" * 60)
print("4. TRAINING DYNAMICS ANALYSIS")
print("=" * 60)

print("\nA. History Data Structure:")
print("-" * 40)
print(f"Total history records: {len(history_df)}")
print(f"Runs with history: {history_df['run_id'].nunique()}")

history_metrics = [
    col
    for col in history_df.columns
    if col not in ["run_id", "run_name", "project", "step", "timestamp"]
]
print(f"Training metrics available: {len(history_metrics)}")
print("Key training metrics:")
for metric in history_metrics[:10]:
    non_null_count = history_df[metric].notna().sum()
    print(f"  {metric}: {non_null_count} records")

print("\nB. Training Length Distribution:")
print("-" * 40)
steps_per_run = history_df.groupby("run_id")["step"].max()
print(f"Mean training steps: {steps_per_run.mean():.0f}")
print(f"Median training steps: {steps_per_run.median():.0f}")
print(f"Max training steps: {steps_per_run.max()}")
print(f"Min training steps: {steps_per_run.min()}")

print("\n" + "=" * 60)
print("5. EXPERIMENTAL DESIGN PATTERNS")
print("=" * 60)

print("\nA. Learning Rate vs Model Size Relationships:")
print("-" * 40)
# Clean numeric conversion for analysis
lr_model_df = runs_df[runs_df["learning_rate"] > 0].copy()
lr_model_df["model_size_clean"] = pd.to_numeric(lr_model_df["model_size"], errors='coerce')
lr_model_analysis = lr_model_df[["model_size_clean", "learning_rate"]].dropna()
if len(lr_model_analysis) > 0:
    for model_size in sorted(lr_model_analysis["model_size_clean"].unique()):
        model_runs = lr_model_analysis[lr_model_analysis["model_size_clean"] == model_size]
        lr_range = f"{model_runs['learning_rate'].min():.1e} to {model_runs['learning_rate'].max():.1e}"
        print(f"  {model_size:,.0f} params: {len(model_runs)} runs, LR range {lr_range}")

print("\nB. Experimental Coverage Analysis:")
print("-" * 40)
finished_runs = runs_df[runs_df["state"] == "finished"]
print(
    f"Successfully completed runs: {len(finished_runs)}/{len(runs_df)} ({len(finished_runs) / len(runs_df) * 100:.1f}%)"
)

print("State distribution:")
for state, count in runs_df["state"].value_counts().items():
    print(f"  {state}: {count} runs")

print("\n" + "=" * 60)
print("6. DATA QUALITY ASSESSMENT")
print("=" * 60)

print("\nA. Data Completeness for Key Dimensions:")
print("-" * 40)
key_fields = ["model_size", "learning_rate", "train_loss"]
for field in key_fields:
    if field in runs_df.columns:
        completeness = runs_df[field].notna().sum() / len(runs_df) * 100
        print(f"  {field}: {completeness:.1f}% complete")

print("\nB. Evaluation Data Coverage:")
print("-" * 40)
runs_with_eval = 0
for _, run in runs_df.iterrows():
    if any(pd.notna(run[col]) for col in eval_cols):
        runs_with_eval += 1

print(
    f"Runs with evaluation data: {runs_with_eval}/{len(runs_df)} ({runs_with_eval / len(runs_df) * 100:.1f}%)"
)

print("\nC. Training History Coverage:")
print("-" * 40)
runs_with_history = history_df["run_id"].nunique()
print(
    f"Runs with training history: {runs_with_history}/{len(runs_df)} ({runs_with_history / len(runs_df) * 100:.1f}%)"
)

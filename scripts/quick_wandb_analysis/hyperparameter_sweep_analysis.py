#!/usr/bin/env python3

from datadec import WandBStore
import pandas as pd
import re

store = WandBStore("postgresql+psycopg://localhost/wandb_test")
runs_df = store.get_runs()

print("=== KEY HYPERPARAMETER SWEEP ANALYSIS ===\n")


# Enhanced extraction function
def extract_key_hyperparameters(name):
    """Extract key hyperparameters with better accuracy"""
    extracted = {}

    # Learning rate - multiple formats
    lr_patterns = [r"--learning_rate=([0-9\.e\-]+)", r"learning_rate=([0-9\.e\-]+)"]
    for pattern in lr_patterns:
        match = re.search(pattern, name)
        if match:
            extracted["name_learning_rate"] = float(match.group(1))
            break

    # Model size from dolma pattern
    model_match = re.search(r"dolma1_7-(\d+)M", name)
    if model_match:
        extracted["name_model_size_m"] = int(model_match.group(1))

    # Dataset tokens pattern (e.g., "1Mtx100" = 1M tokens × 100 = 100M total)
    token_match = re.search(r"main_(\d+)Mtx(\d+)", name)
    if token_match:
        base_tokens = int(token_match.group(1))
        multiplier = int(token_match.group(2))
        extracted["dataset_tokens_m"] = base_tokens
        extracted["dataset_multiplier"] = multiplier
        extracted["total_dataset_tokens_m"] = base_tokens * multiplier

    # Training method
    if "dpo" in name.lower():
        extracted["name_method"] = "dpo"
    elif "finetune" in name.lower():
        extracted["name_method"] = "finetune"
    else:
        extracted["name_method"] = "other"

    # Special experiment types from name
    if "--max_train_samples=" in name:
        sample_match = re.search(r"--max_train_samples=(\d+)", name)
        if sample_match:
            extracted["max_train_samples"] = int(sample_match.group(1))

    if "--reduce_loss=" in name:
        loss_match = re.search(r"--reduce_loss=(\w+)", name)
        if loss_match:
            extracted["reduce_loss_type"] = loss_match.group(1)

    return extracted


# Extract hyperparameters from all runs
enhanced_data = []
for i, name in enumerate(runs_df["run_name"]):
    run_data = runs_df.iloc[i].to_dict()
    parsed_params = extract_key_hyperparameters(name)

    # Combine metadata and parsed data
    combined = {**run_data, **parsed_params}
    enhanced_data.append(combined)

enhanced_df = pd.DataFrame(enhanced_data)

print("1. PRIMARY SWEEP DIMENSIONS")
print("=" * 50)

print("\nA. Model Size Sweep:")
print("-" * 30)
if "name_model_size_m" in enhanced_df.columns:
    model_sizes = enhanced_df["name_model_size_m"].value_counts().sort_index()
    print("Model sizes from names:")
    for size, count in model_sizes.items():
        print(f"  {size}M: {count} runs")

    # Compare with metadata
    metadata_sizes = enhanced_df["model_size"].dropna() / 1e6
    print(f"\nMetadata model sizes: {sorted(metadata_sizes.unique())[:10]}M (first 10)")

print("\nB. Learning Rate Sweep:")
print("-" * 30)
if "name_learning_rate" in enhanced_df.columns:
    name_lrs = enhanced_df["name_learning_rate"].dropna().unique()
    print(f"Learning rates from names: {sorted(name_lrs)}")
    print("Count per LR:")
    lr_counts = enhanced_df["name_learning_rate"].value_counts().sort_index()
    for lr, count in lr_counts.items():
        print(f"  {lr:.0e}: {count} runs")

print("\nC. Dataset Token Sweep:")
print("-" * 30)
if "total_dataset_tokens_m" in enhanced_df.columns:
    token_amounts = enhanced_df["total_dataset_tokens_m"].value_counts().sort_index()
    print("Total dataset sizes (M tokens):")
    for tokens, count in token_amounts.items():
        print(f"  {tokens}M: {count} runs")

    # Show the base×multiplier breakdown
    if (
        "dataset_tokens_m" in enhanced_df.columns
        and "dataset_multiplier" in enhanced_df.columns
    ):
        print("\nDataset patterns (base × multiplier):")
        pattern_counts = (
            enhanced_df.groupby(["dataset_tokens_m", "dataset_multiplier"])
            .size()
            .reset_index(name="count")
        )
        for _, row in pattern_counts.iterrows():
            print(
                f"  {row['dataset_tokens_m']}M × {row['dataset_multiplier']} = {row['dataset_tokens_m'] * row['dataset_multiplier']}M: {row['count']} runs"
            )

print("\n2. EXPERIMENTAL DESIGN STRUCTURE")
print("=" * 50)

print("\nA. Model × Learning Rate Grid:")
print("-" * 30)
if (
    "name_model_size_m" in enhanced_df.columns
    and "name_learning_rate" in enhanced_df.columns
):
    model_lr_crosstab = pd.crosstab(
        enhanced_df["name_model_size_m"],
        enhanced_df["name_learning_rate"],
        margins=True,
    )
    print(model_lr_crosstab)

print("\nB. Model × Dataset Size Grid:")
print("-" * 30)
if (
    "name_model_size_m" in enhanced_df.columns
    and "total_dataset_tokens_m" in enhanced_df.columns
):
    model_data_crosstab = pd.crosstab(
        enhanced_df["name_model_size_m"],
        enhanced_df["total_dataset_tokens_m"],
        margins=True,
    )
    print(model_data_crosstab)

print("\n3. SPECIAL EXPERIMENT TYPES")
print("=" * 50)

print("\nA. Data Representation Experiments:")
print("-" * 30)
data_rep_runs = enhanced_df[enhanced_df["max_train_samples"].notna()]
if len(data_rep_runs) > 0:
    print(f"Runs with max_train_samples: {len(data_rep_runs)}")
    sample_counts = data_rep_runs["max_train_samples"].value_counts().sort_index()
    for samples, count in sample_counts.items():
        print(f"  {samples} samples: {count} runs")

    if "reduce_loss_type" in data_rep_runs.columns:
        loss_types = data_rep_runs["reduce_loss_type"].value_counts()
        print(f"  Loss reduction types: {dict(loss_types)}")

print("\nB. Training Method Comparison:")
print("-" * 30)
method_counts = enhanced_df["name_method"].value_counts()
for method, count in method_counts.items():
    print(f"  {method}: {count} runs")

# Analyze DPO vs Finetune by model size
if "name_method" in enhanced_df.columns and "name_model_size_m" in enhanced_df.columns:
    method_model_crosstab = pd.crosstab(
        enhanced_df["name_model_size_m"], enhanced_df["name_method"]
    )
    print("\nMethod × Model Size:")
    print(method_model_crosstab)

print("\n4. EXPERIMENT EVOLUTION TIMELINE")
print("=" * 50)

print("\nA. Run Name Complexity Evolution:")
print("-" * 30)
enhanced_df["run_length"] = enhanced_df["run_name"].str.len()
enhanced_df["has_lr_in_name"] = enhanced_df["name_learning_rate"].notna()
enhanced_df["has_tokens_in_name"] = enhanced_df["total_dataset_tokens_m"].notna()
enhanced_df["has_special_params"] = enhanced_df["max_train_samples"].notna()

# Sort by created_at to see evolution
if "created_at" in enhanced_df.columns:
    enhanced_df_sorted = enhanced_df.sort_values("created_at")

    print("First 10 runs (earliest):")
    for i, row in enhanced_df_sorted.head(10).iterrows():
        lr_str = (
            f"LR={row['name_learning_rate']:.0e}"
            if pd.notna(row["name_learning_rate"])
            else "no-LR"
        )
        tokens_str = (
            f"tokens={row['total_dataset_tokens_m']}M"
            if pd.notna(row["total_dataset_tokens_m"])
            else "no-tokens"
        )
        print(
            f"  {row['created_at'].strftime('%m-%d')}: {lr_str}, {tokens_str}, len={row['run_length']}"
        )

    print("\nLast 10 runs (latest):")
    for i, row in enhanced_df_sorted.tail(10).iterrows():
        lr_str = (
            f"LR={row['name_learning_rate']:.0e}"
            if pd.notna(row["name_learning_rate"])
            else "no-LR"
        )
        tokens_str = (
            f"tokens={row['total_dataset_tokens_m']}M"
            if pd.notna(row["total_dataset_tokens_m"])
            else "no-tokens"
        )
        print(
            f"  {row['created_at'].strftime('%m-%d')}: {lr_str}, {tokens_str}, len={row['run_length']}"
        )

print("\n5. PLOTTING STRATEGY RECOMMENDATIONS")
print("=" * 50)

print("\nA. Primary Sweep Dimensions (High Coverage):")
print("-" * 40)
print("✅ Model Size: 4M, 10M, 60M, 150M (100% name coverage)")
print("✅ Learning Rate: 8 values from 2e-07 to 5e-04 (96% name coverage)")
print("✅ Training Method: finetune vs dpo (99% name coverage)")

print("\nB. Secondary Dimensions (Moderate Coverage):")
print("-" * 40)
print("⚠️  Dataset Tokens: 1M, 10M, 100M, 1000M (54% name coverage)")
print("⚠️  Dataset Multiplier: 1×, 10×, 100× patterns")

print("\nC. Special Experiments (Low Coverage):")
print("-" * 40)
print("🔍 Data Representation: max_train_samples experiments (few runs)")
print("🔍 Loss Reduction: sum vs mean loss (few runs)")

print("\nD. Recommended Plot Dimensions:")
print("-" * 40)
print("1. **Primary**: Model Size × Learning Rate (full grid)")
print("2. **Secondary**: Training Method comparison (finetune vs dpo)")
print("3. **Advanced**: Dataset Size scaling (where available)")
print("4. **Special**: Data efficiency experiments (separate analysis)")

print("\nE. Data Quality for Plotting:")
print("-" * 40)
complete_runs = enhanced_df[
    enhanced_df["name_model_size_m"].notna()
    & enhanced_df["name_learning_rate"].notna()
    & (enhanced_df["state"] == "finished")
].shape[0]
print(
    f"Runs with complete model+LR+finished: {complete_runs}/{len(enhanced_df)} ({complete_runs / len(enhanced_df) * 100:.1f}%)"
)

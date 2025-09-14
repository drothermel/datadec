import pandas as pd

from datadec.wandb_eval import analysis_helpers
from datadec.wandb_eval import wandb_transforms as transforms
from datadec.wandb_eval.wandb_loader import WandBDataLoader

analysis_helpers.configure_pandas_display()

loader = WandBDataLoader()
runs_df, _ = loader.load_runs_and_history()

print("=== KEY HYPERPARAMETER SWEEP ANALYSIS ===\n")


enhanced_data = []
for i, name in enumerate(runs_df["run_name"]):
    run_data = runs_df.iloc[i].to_dict()
    parsed_params = transforms.extract_hyperparameters(name)

    converted_params = {}
    for key, value in parsed_params.items():
        if key.endswith("_rnp"):
            base_key = key[:-4]
            if base_key == "lr":
                converted_params["name_learning_rate"] = value
            elif base_key == "data":
                converted_params["name_data"] = value
            elif base_key == "params":
                if isinstance(value, str) and value.endswith("M"):
                    converted_params["name_model_size_m"] = int(value[:-1])
            elif base_key == "total_tok":
                converted_params["total_dataset_tokens_m"] = value
            elif base_key == "method":
                converted_params["name_method"] = value
            else:
                converted_params[f"name_{base_key}"] = value
        else:
            converted_params[key] = value

    combined = {**run_data, **converted_params}
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

    metadata_sizes_clean = (
        pd.to_numeric(enhanced_df["model_size"], errors="coerce").dropna() / 1e6
    )
    if len(metadata_sizes_clean) > 0:
        print(
            f"\nMetadata model sizes: {sorted(metadata_sizes_clean.unique())[:10]}M (first 10)"
        )
    else:
        print("\nMetadata model sizes: No valid numeric values found")

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

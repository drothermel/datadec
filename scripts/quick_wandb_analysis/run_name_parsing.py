#!/usr/bin/env python3

from datadec import WandBStore
import pandas as pd
import re
from collections import defaultdict

store = WandBStore("postgresql+psycopg://localhost/wandb_test")
runs_df = store.get_runs()

print("=== RUN NAME HYPERPARAMETER PARSING ANALYSIS ===\n")

# Get all run names
run_names = runs_df["run_name"].tolist()
print(f"Total runs to analyze: {len(run_names)}")

print("\n1. RUN NAME PATTERN ANALYSIS")
print("=" * 60)

print("\nA. Sample Run Names (showing evolution):")
print("-" * 50)
for i, name in enumerate(run_names[:15]):
    print(f"{i + 1:2}. {name}")

print("\n\nB. Run Name Length Distribution:")
print("-" * 50)
name_lengths = [len(name) for name in run_names]
print(f"Mean length: {sum(name_lengths) / len(name_lengths):.0f} characters")
print(f"Range: {min(name_lengths)} to {max(name_lengths)} characters")

# Length groups
short_names = [name for name in run_names if len(name) < 80]
medium_names = [name for name in run_names if 80 <= len(name) < 120]
long_names = [name for name in run_names if len(name) >= 120]

print(f"Short names (<80 chars): {len(short_names)} runs")
print(f"Medium names (80-120 chars): {len(medium_names)} runs")
print(f"Long names (≥120 chars): {len(long_names)} runs")

print("\n2. HYPERPARAMETER EXTRACTION PATTERNS")
print("=" * 60)

# Define extraction patterns
patterns = {
    "learning_rate": [
        r"--learning_rate=([0-9\.e\-]+)",
        r"_lr=([0-9\.e\-]+)",
        r"learning_rate=([0-9\.e\-]+)",
    ],
    "model_size": [
        r"(\d+)M(?:_|\b)",
        r"dolma1_7-(\d+)M",
        r"-(\d+\.\d+)M",
        r"(\d+\.\d+)M",
    ],
    "dataset_tokens": [r"(\d+)Mtx(\d+)", r"main_(\d+)Mtx(\d+)", r"(\d+)Mt[x_](\d+)"],
    "batch_size": [
        r"--per_device_train_batch_size=(\d+)",
        r"batch_size=(\d+)",
        r"bs=(\d+)",
    ],
    "max_train_samples": [r"--max_train_samples=(\d+)", r"max_samples=(\d+)"],
    "reduce_loss": [r"--reduce_loss=(\w+)", r"reduce_loss=(\w+)"],
    "training_method": [r"(dpo|finetune)", r"_(dpo|ft)_", r"(DPO|Finetune)"],
    "dataset_name": [r"DD-([^_\-]+)", r"dolma([^_\-]*)", r"(dclm[^_\-]*)"],
}


def extract_hyperparameters(name):
    """Extract hyperparameters from run name using patterns"""
    extracted = {}

    for param, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                if param == "dataset_tokens":
                    # Special handling for token patterns like "1Mtx100"
                    extracted[f"{param}_amount"] = match.group(1)
                    extracted[f"{param}_multiplier"] = match.group(2)
                elif param in ["learning_rate", "model_size"]:
                    extracted[param] = match.group(1)
                else:
                    extracted[param] = match.group(1)
                break  # Use first match

    return extracted


print("\nA. Hyperparameter Extraction Results:")
print("-" * 50)

# Extract hyperparameters from all run names
extracted_params = []
for name in run_names:
    params = extract_hyperparameters(name)
    params["run_name"] = name
    extracted_params.append(params)

# Analyze extraction success rates
extraction_stats = defaultdict(int)
for params in extracted_params:
    for key in params:
        if key != "run_name" and params[key]:
            extraction_stats[key] += 1

print("Extraction success rates:")
for param, count in sorted(extraction_stats.items()):
    percentage = count / len(run_names) * 100
    print(f"  {param}: {count}/{len(run_names)} ({percentage:.1f}%)")

print("\n\nB. Extracted Learning Rates:")
print("-" * 50)
extracted_lrs = [
    params.get("learning_rate")
    for params in extracted_params
    if params.get("learning_rate")
]
unique_extracted_lrs = list(set(extracted_lrs))[:15]  # Show first 15
print(f"Unique learning rates found in names: {len(set(extracted_lrs))}")
print(
    f"Sample values: {sorted(unique_extracted_lrs, key=lambda x: float(x) if x else 0)}"
)

print("\n\nC. Extracted Model Sizes:")
print("-" * 50)
extracted_sizes = [
    params.get("model_size") for params in extracted_params if params.get("model_size")
]
unique_sizes = sorted(set(extracted_sizes), key=lambda x: float(x) if x else 0)
print(f"Model sizes found in names: {unique_sizes}")

print("\n\nD. Dataset Token Patterns:")
print("-" * 50)
token_patterns = [
    (params.get("dataset_tokens_amount"), params.get("dataset_tokens_multiplier"))
    for params in extracted_params
    if params.get("dataset_tokens_amount") and params.get("dataset_tokens_multiplier")
]
unique_token_patterns = list(set(token_patterns))
print(f"Token patterns found: {len(unique_token_patterns)}")
for amount, mult in sorted(unique_token_patterns):
    print(f"  {amount}M × {mult}")

print("\n3. COMPARISON WITH METADATA")
print("=" * 60)

print("\nA. Learning Rate Comparison:")
print("-" * 50)
# Compare extracted LRs with metadata LRs
lr_comparison_data = []
for i, params in enumerate(extracted_params):
    run_data = runs_df.iloc[i]
    extracted_lr = params.get("learning_rate")
    metadata_lr = run_data.get("learning_rate", None)

    if extracted_lr and pd.notna(metadata_lr):
        try:
            extracted_float = float(extracted_lr)
            if (
                abs(extracted_float - metadata_lr) < 1e-10 or metadata_lr == 0.0
            ):  # Account for zero LR issue
                lr_comparison_data.append("match")
            else:
                lr_comparison_data.append("mismatch")
        except:
            lr_comparison_data.append("parse_error")
    elif extracted_lr and pd.isna(metadata_lr):
        lr_comparison_data.append("name_only")
    elif not extracted_lr and pd.notna(metadata_lr):
        lr_comparison_data.append("metadata_only")
    else:
        lr_comparison_data.append("neither")

lr_comparison_counts = pd.Series(lr_comparison_data).value_counts()
print("Learning rate extraction vs metadata:")
for category, count in lr_comparison_counts.items():
    print(f"  {category}: {count} runs")

print("\n\nB. Model Size Comparison:")
print("-" * 50)
size_comparison_data = []
for i, params in enumerate(extracted_params):
    run_data = runs_df.iloc[i]
    extracted_size = params.get("model_size")
    metadata_size = run_data.get("model_size", None)

    if extracted_size and pd.notna(metadata_size):
        try:
            extracted_float = float(extracted_size) * 1e6  # Convert M to actual number
            if abs(extracted_float - metadata_size) / metadata_size < 0.01:  # Within 1%
                size_comparison_data.append("match")
            else:
                size_comparison_data.append("mismatch")
        except:
            size_comparison_data.append("parse_error")
    elif extracted_size:
        size_comparison_data.append("name_only")
    elif pd.notna(metadata_size):
        size_comparison_data.append("metadata_only")
    else:
        size_comparison_data.append("neither")

size_comparison_counts = pd.Series(size_comparison_data).value_counts()
print("Model size extraction vs metadata:")
for category, count in size_comparison_counts.items():
    print(f"  {category}: {count} runs")

print("\n4. PARSING RECOMMENDATIONS")
print("=" * 60)

print("\nA. Most Reliable Extractions:")
print("-" * 50)
reliable_params = []
for param, count in extraction_stats.items():
    success_rate = count / len(run_names)
    if success_rate > 0.5:  # More than 50% success rate
        reliable_params.append((param, success_rate))

reliable_params.sort(key=lambda x: x[1], reverse=True)
for param, rate in reliable_params:
    print(f"  {param}: {rate * 100:.1f}% success rate")

print("\nB. Recommended Parsing Strategy:")
print("-" * 50)
print("1. Use run name parsing for:")
for param, rate in reliable_params[:5]:
    print(f"   - {param} ({rate * 100:.0f}% reliable)")

print("\n2. Use metadata for:")
unreliable_params = [
    param for param, rate in extraction_stats.items() if rate / len(run_names) < 0.3
]
print("   - Parameters with <30% name extraction success")
print("   - Cross-validation with extracted values")

print("\n3. Hybrid approach:")
print("   - Extract from names where possible")
print("   - Fall back to metadata for missing values")
print("   - Cross-validate extracted vs metadata values")

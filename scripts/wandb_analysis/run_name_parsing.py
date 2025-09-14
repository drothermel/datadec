from collections import defaultdict

import pandas as pd

from datadec.wandb_eval import analysis_helpers
from datadec.wandb_eval import wandb_transforms as transforms
from datadec.wandb_eval.wandb_loader import WandBDataLoader

analysis_helpers.configure_pandas_display()

loader = WandBDataLoader()
runs_df, _ = loader.load_runs_and_history()

print("=== RUN NAME HYPERPARAMETER PARSING ANALYSIS ===\n")

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

short_names = [name for name in run_names if len(name) < 80]
medium_names = [name for name in run_names if 80 <= len(name) < 120]
long_names = [name for name in run_names if len(name) >= 120]

print(f"Short names (<80 chars): {len(short_names)} runs")
print(f"Medium names (80-120 chars): {len(medium_names)} runs")
print(f"Long names (≥120 chars): {len(long_names)} runs")

print("\n2. HYPERPARAMETER EXTRACTION PATTERNS")
print("=" * 60)


print("\nA. Hyperparameter Extraction Results:")
print("-" * 50)

extracted_params = []
for name in run_names:
    params = transforms.extract_hyperparameters(name)
    converted_params = {"run_name": name}
    for key, value in params.items():
        if key.endswith("_rnp"):
            base_key = key[:-4]
            if base_key == "lr":
                converted_params["learning_rate"] = value
            elif base_key == "params":
                if isinstance(value, str) and value.endswith("M"):
                    converted_params["model_size"] = value[:-1]
            elif base_key == "total_tok":
                converted_params["dataset_tokens_amount"] = str(value)
                converted_params["dataset_tokens_multiplier"] = "1"
            elif base_key == "method":
                converted_params["training_method"] = value
            else:
                converted_params[base_key] = value
        else:
            converted_params[key] = value

    extracted_params.append(converted_params)

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
unique_extracted_lrs = list(set(extracted_lrs))[:15]
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
lr_comparison_data = []
for i, params in enumerate(extracted_params):
    run_data = runs_df.iloc[i]
    extracted_lr = params.get("learning_rate")
    metadata_lr = run_data.get("learning_rate", None)

    if extracted_lr and pd.notna(metadata_lr):
        try:
            extracted_float = float(extracted_lr)
            if abs(extracted_float - metadata_lr) < 1e-10 or metadata_lr == 0.0:
                lr_comparison_data.append("match")
            else:
                lr_comparison_data.append("mismatch")
        except Exception:
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
            extracted_float = float(extracted_size) * 1e6
            if abs(extracted_float - metadata_size) / metadata_size < 0.01:
                size_comparison_data.append("match")
            else:
                size_comparison_data.append("mismatch")
        except Exception:
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
    if success_rate > 0.5:
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

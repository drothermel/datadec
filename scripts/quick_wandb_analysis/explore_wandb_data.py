#!/usr/bin/env python3

from datadec import WandBStore
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 100)

store = WandBStore("postgresql+psycopg://localhost/wandb_test")

print("=== WandB Data Exploration ===\n")

print("1. BASIC DATABASE STATS")
print("-" * 40)
runs_df = store.get_runs()
history_df = store.get_history()

print(f"Total runs in database: {len(runs_df)}")
print(f"Total history records: {len(history_df)}")
print()

print("2. RUN STATES DISTRIBUTION")
print("-" * 40)
if not runs_df.empty:
    state_counts = runs_df["state"].value_counts()
    for state, count in state_counts.items():
        print(f"{state}: {count}")
print()

print("3. SAMPLE RUN DATA STRUCTURE")
print("-" * 40)
if not runs_df.empty:
    print("Core columns in runs table:")
    print(f"Columns: {list(runs_df.columns)}")
    print()

    print("First run sample:")
    first_run = runs_df.iloc[0]
    print(f"Run ID: {first_run['run_id']}")
    print(f"Run Name: {first_run['run_name']}")
    print(f"State: {first_run['state']}")
    print(f"Created: {first_run['created_at']}")
    print()

print("4. RUN NAMING PATTERNS")
print("-" * 40)
if not runs_df.empty:
    run_names = runs_df["run_name"].tolist()
    print("Sample run names:")
    for name in run_names[:10]:
        print(f"  {name}")
    print()

    # Extract patterns from run names
    print("Run name patterns analysis:")
    has_finetune = sum(1 for name in run_names if "finetune" in name)
    has_dpo = sum(1 for name in run_names if "dpo" in name)
    has_test = sum(1 for name in run_names if "test" in name)
    print(f"Finetune runs: {has_finetune}")
    print(f"DPO runs: {has_dpo}")
    print(f"Test runs: {has_test}")
    print()

print("5. MODEL SIZE ANALYSIS")
print("-" * 40)
if "model_size" in runs_df.columns:
    model_sizes = runs_df["model_size"].value_counts().sort_index()
    print("Model sizes distribution:")
    for size, count in model_sizes.items():
        print(f"  {size}: {count} runs")
    print()

print("6. DATASET ANALYSIS")
print("-" * 40)
if "dataset_name" in runs_df.columns:
    datasets = runs_df["dataset_name"].value_counts()
    print("Dataset distribution:")
    for dataset, count in datasets.items():
        print(f"  {dataset}: {count} runs")
    print()

print("7. LEARNING RATE PATTERNS")
print("-" * 40)
if "learning_rate" in runs_df.columns:
    lr_values = runs_df["learning_rate"].value_counts().sort_index()
    print("Learning rate distribution:")
    for lr, count in lr_values.items():
        print(f"  {lr}: {count} runs")
    print()

print("8. HISTORY DATA SAMPLE")
print("-" * 40)
if not history_df.empty:
    print("History columns:")
    print(f"Columns: {list(history_df.columns)}")
    print()

    sample_run_history = history_df[
        history_df["run_id"] == history_df["run_id"].iloc[0]
    ]
    print(f"Sample run has {len(sample_run_history)} history records")
    if len(sample_run_history) > 0:
        print("Sample history record:")
        print(f"Step: {sample_run_history.iloc[0]['step']}")
        print(
            f"Available metrics: {[col for col in sample_run_history.columns if col not in ['run_id', 'run_name', 'project', 'step', 'timestamp']]}"
        )

    print("\nHistory records per run distribution:")
    history_counts = history_df.groupby("run_id").size()
    print(f"Mean history records per run: {history_counts.mean():.1f}")
    print(f"Min history records: {history_counts.min()}")
    print(f"Max history records: {history_counts.max()}")

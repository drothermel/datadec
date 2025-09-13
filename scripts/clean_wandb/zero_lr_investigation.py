#!/usr/bin/env python3

from datadec import WandBStore

store = WandBStore("postgresql+psycopg://localhost/wandb_test")
runs_df = store.get_runs()
history_df = store.get_history()

print("=== ZERO LEARNING RATE RUNS INVESTIGATION ===\n")

# Get zero LR runs
zero_lr_runs = runs_df[runs_df["learning_rate"] == 0.0]
print(f"Total runs with learning_rate = 0.0: {len(zero_lr_runs)}")
print(f"Percentage of total: {len(zero_lr_runs) / len(runs_df) * 100:.1f}%")

print("\n1. ZERO LR RUN CHARACTERISTICS")
print("=" * 50)

print("\nA. Run States:")
print("-" * 30)
state_counts = zero_lr_runs["state"].value_counts()
for state, count in state_counts.items():
    print(f"  {state}: {count} runs")

print("\nB. Run Names (sample):")
print("-" * 30)
for i, name in enumerate(zero_lr_runs["run_name"].head(10)):
    print(f"  {i + 1}. {name}")

print("\nC. Training Configuration Analysis:")
print("-" * 30)
training_config_fields = [
    "num_train_epochs",
    "max_train_steps",
    "warmup_ratio",
    "gradient_accumulation_steps",
    "per_device_train_batch_size",
]

for field in training_config_fields:
    if field in zero_lr_runs.columns:
        has_values = zero_lr_runs[field].notna().sum()
        if zero_lr_runs[field].dtype in ["float64", "int64"]:
            non_zero = (zero_lr_runs[field] > 0).sum()
            mean_val = zero_lr_runs[field].mean() if has_values > 0 else 0
            print(
                f"  {field}: {has_values} have values, {non_zero} are > 0, mean={mean_val:.2f}"
            )
        else:
            print(f"  {field}: {has_values} have values")

print("\n2. TRAINING HISTORY ANALYSIS")
print("=" * 50)

# Check if zero LR runs have training history
zero_lr_run_ids = set(zero_lr_runs["run_id"])
zero_lr_history = history_df[history_df["run_id"].isin(zero_lr_run_ids)]

print(
    f"\nZero LR runs with training history: {zero_lr_history['run_id'].nunique()}/{len(zero_lr_runs)}"
)

if len(zero_lr_history) > 0:
    print(f"Total history records for zero LR runs: {len(zero_lr_history)}")

    print("\nA. Training Steps Analysis:")
    print("-" * 30)
    steps_per_run = zero_lr_history.groupby("run_id")["step"].max()
    print(f"  Runs with history: {len(steps_per_run)}")
    print(f"  Mean steps: {steps_per_run.mean():.0f}")
    print(f"  Step range: {steps_per_run.min():.0f} to {steps_per_run.max():.0f}")

    print("\nB. Learning Rate in History (should reveal true LR):")
    print("-" * 30)
    if "learning_rate" in zero_lr_history.columns:
        history_lrs = zero_lr_history["learning_rate"].dropna()
        if len(history_lrs) > 0:
            unique_lrs = history_lrs.unique()
            print(f"  Unique LR values in history: {len(unique_lrs)}")
            print(
                f"  LR range in history: {history_lrs.min():.2e} to {history_lrs.max():.2e}"
            )
            print(
                f"  Non-zero LRs in history: {(history_lrs > 0).sum()}/{len(history_lrs)}"
            )

            if len(unique_lrs) <= 10:
                print(f"  Actual LR values found: {sorted(unique_lrs)}")
        else:
            print("  No learning rate values in training history")

    print("\nC. Training Loss Analysis:")
    print("-" * 30)
    if "train_loss" in zero_lr_history.columns:
        loss_data = zero_lr_history["train_loss"].dropna()
        if len(loss_data) > 0:
            print(f"  Training loss records: {len(loss_data)}")
            print(f"  Loss range: {loss_data.min():.3f} to {loss_data.max():.3f}")

            # Check if loss is decreasing (sign of actual training)
            loss_by_run = zero_lr_history.groupby("run_id")["train_loss"].apply(list)
            decreasing_count = 0
            for run_id, losses in loss_by_run.items():
                if len(losses) > 1:
                    if losses[-1] < losses[0]:  # Final loss < initial loss
                        decreasing_count += 1

            print(
                f"  Runs with decreasing loss: {decreasing_count}/{len(loss_by_run)} (indicates actual training)"
            )
        else:
            print("  No training loss data found")

print("\n3. COMPARISON WITH NON-ZERO LR RUNS")
print("=" * 50)

non_zero_lr_runs = runs_df[runs_df["learning_rate"] > 0]
print(f"Non-zero LR runs: {len(non_zero_lr_runs)}")

print("\nA. Success Rate Comparison:")
print("-" * 30)
zero_lr_success = (zero_lr_runs["state"] == "finished").sum() / len(zero_lr_runs) * 100
non_zero_lr_success = (
    (non_zero_lr_runs["state"] == "finished").sum() / len(non_zero_lr_runs) * 100
)
print(f"  Zero LR success rate: {zero_lr_success:.1f}%")
print(f"  Non-zero LR success rate: {non_zero_lr_success:.1f}%")

print("\nB. Training Configuration Completeness:")
print("-" * 30)
zero_lr_config_complete = zero_lr_runs["num_train_epochs"].notna().sum()
non_zero_lr_config_complete = non_zero_lr_runs["num_train_epochs"].notna().sum()
print(
    f"  Zero LR with training config: {zero_lr_config_complete}/{len(zero_lr_runs)} ({zero_lr_config_complete / len(zero_lr_runs) * 100:.1f}%)"
)
print(
    f"  Non-zero LR with training config: {non_zero_lr_config_complete}/{len(non_zero_lr_runs)} ({non_zero_lr_config_complete / len(non_zero_lr_runs) * 100:.1f}%)"
)

print("\n4. DETAILED INVESTIGATION OF SPECIFIC ZERO LR RUNS")
print("=" * 50)

# Look at a few specific examples
print("\nA. Sample Zero LR Run Details:")
print("-" * 30)
for i, (_, run) in enumerate(zero_lr_runs.head(3).iterrows()):
    print(f"\nRun {i + 1}: {run['run_name']}")
    print(f"  State: {run['state']}")
    print(f"  Model size: {run.get('model_size', 'N/A')}")
    print(f"  Num epochs: {run.get('num_train_epochs', 'N/A')}")
    print(f"  Max steps: {run.get('max_train_steps', 'N/A')}")

    # Check if this run has history
    run_history = history_df[history_df["run_id"] == run["run_id"]]
    if len(run_history) > 0:
        print(f"  Has training history: Yes ({len(run_history)} records)")
        if "learning_rate" in run_history.columns:
            actual_lrs = run_history["learning_rate"].dropna().unique()
            if len(actual_lrs) > 0:
                print(f"  Actual LRs in history: {actual_lrs}")
        if "train_loss" in run_history.columns:
            losses = run_history["train_loss"].dropna()
            if len(losses) > 1:
                print(
                    f"  Loss progression: {losses.iloc[0]:.3f} → {losses.iloc[-1]:.3f}"
                )
    else:
        print("  Has training history: No")

print("\n" + "=" * 60)
print("CONCLUSION: ZERO LR RUNS ANALYSIS")
print("=" * 60)

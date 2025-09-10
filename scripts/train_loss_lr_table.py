#!/usr/bin/env python3

import pandas as pd

from datadec import analysis_helpers


def main():
    print("=== TRAIN LOSS LEARNING RATE TABLE ===\n")
    runs_df = analysis_helpers.load_runs_df()

    df = analysis_helpers.filter_runs_by_method(runs_df, "finetune")

    if "train_loss" not in df.columns:
        df["train_loss"] = df["run_id"].map(runs_df.set_index("run_id")["train_loss"])
    print(f"Total finetune runs: {len(df)}")
    print(f"Finished runs: {(df['state'] == 'finished').sum()}")
    print(f"With learning_rate: {df['learning_rate'].notna().sum()}")
    print(f"With model_size_m: {df['model_size_m'].notna().sum()}")
    print(f"With dataset_total_m: {df['dataset_total_m'].notna().sum()}")
    print(f"With train_loss: {df['train_loss'].notna().sum()}")

    complete_df = df[
        (df["state"] == "finished")
        & df["learning_rate"].notna()
        & df["model_size_m"].notna()
        & df["dataset_total_m"].notna()
        & df["train_loss"].notna()
    ]

    print(
        f"\nComplete data (finished + all params + train_loss): {len(complete_df)} runs"
    )

    if len(complete_df) == 0:
        print("No complete data available!")
        return

    print("\nSample of complete data:")
    print(
        complete_df[
            ["model_size_m", "dataset_total_m", "learning_rate", "train_loss"]
        ].head(10)
    )

    print(f"\n{'=' * 80}")
    print("TRAIN LOSS BY LEARNING RATE")
    print(f"{'=' * 80}")
    print(f"\nUnique model sizes: {sorted(complete_df['model_size_m'].unique())}")
    print(f"Unique dataset sizes: {sorted(complete_df['dataset_total_m'].unique())}")
    print(f"Unique learning rates: {sorted(complete_df['learning_rate'].unique())}")

    table_data = []
    for model_size, dataset_size in complete_df.groupby(
        ["model_size_m", "dataset_total_m"]
    ).groups.keys():
        subset = complete_df[
            (complete_df["model_size_m"] == model_size)
            & (complete_df["dataset_total_m"] == dataset_size)
        ]
        unique_lrs = subset["learning_rate"].nunique()
        if unique_lrs < 2:
            continue
        row = {
            "Model_Size": f"{int(model_size)}M",
            "Dataset_Tokens": f"{int(dataset_size)}M",
            "Num_LRs": unique_lrs,
        }
        for lr in sorted(subset["learning_rate"].unique()):
            lr_subset = subset[subset["learning_rate"] == lr]
            if len(lr_subset) > 0:
                avg_loss = lr_subset["train_loss"].mean()
                row[f"LR_{lr:.0e}"] = f"{avg_loss:.3f}"
        table_data.append(row)

    if table_data:
        table_df = pd.DataFrame(table_data)
        table_df["_sort_model"] = (
            table_df["Model_Size"].str.replace("M", "").astype(int)
        )
        table_df["_sort_dataset"] = (
            table_df["Dataset_Tokens"].str.replace("M", "").astype(int)
        )
        table_df = table_df.sort_values(["_sort_model", "_sort_dataset"])
        display_df = table_df.drop(columns=["_sort_model", "_sort_dataset"])
        print("\nTrain Loss Learning Rate Sweep Table:")
        print(display_df.to_string(index=False))
        print("\n📊 Learning Rate Effect Analysis:")
        for _, row in display_df.iterrows():
            lr_cols = [col for col in display_df.columns if col.startswith("LR_")]
            if len(lr_cols) >= 2:
                values = [float(row[col]) for col in lr_cols if pd.notna(row[col])]
                if len(values) >= 2:
                    min_loss = min(values)
                    max_loss = max(values)
                    effect_size = max_loss - min_loss
                    print(
                        f"  {row['Model_Size']}, {row['Dataset_Tokens']}: Δ = {effect_size:.3f} (range: {min_loss:.3f} to {max_loss:.3f})"
                    )
    else:
        print("No learning rate sweep data found!")

    print(f"\n{'=' * 80}")
    print("FILTERING BREAKDOWN")
    print(f"{'=' * 80}")
    print(f"1. Total finetune runs: {len(df)}")
    print(f"2. Finished: {(df['state'] == 'finished').sum()}")
    print(
        f"3. + learning_rate: {(df['state'] == 'finished') & df['learning_rate'].notna()}.sum()"
    )
    print(
        f"4. + model_size_m: {((df['state'] == 'finished') & df['learning_rate'].notna() & df['model_size_m'].notna()).sum()}"
    )
    print(
        f"5. + dataset_total_m: {((df['state'] == 'finished') & df['learning_rate'].notna() & df['model_size_m'].notna() & df['dataset_total_m'].notna()).sum()}"
    )
    print(f"6. + train_loss: {len(complete_df)} (final)")
    print("\nFiltered out runs:")
    no_lr = df[(df["state"] == "finished") & df["learning_rate"].isna()]
    print(f"- No learning rate: {len(no_lr)}")
    if len(no_lr) > 0:
        print(f"  Sample: {no_lr['run_name'].iloc[0][:80]}...")
    no_train_loss = df[
        (df["state"] == "finished")
        & df["learning_rate"].notna()
        & df["model_size_m"].notna()
        & df["dataset_total_m"].notna()
        & df["train_loss"].isna()
    ]
    print(f"- No train loss: {len(no_train_loss)}")
    if len(no_train_loss) > 0:
        print(f"  Sample: {no_train_loss['run_name'].iloc[0][:80]}...")


if __name__ == "__main__":
    main()

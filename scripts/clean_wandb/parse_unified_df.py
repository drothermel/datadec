from datadec.wandb_eval import analysis_helpers
from datadec.wandb_eval import wandb_constants as wconsts
from datadec.wandb_eval.parsing import parse_and_clean_runs_df

HISTORY_POSTPROCESSED_TO_KEEP = [
    "max_step",
    "max_tokens",
    "max_epoch",
    "max_lr",
    "final_train_loss",
    "min_train_loss",
]


def main():
    runs_df, _ = analysis_helpers.load_df()

    print("=== SEQUENTIAL FILTERING AND CATEGORIZATION ===\n")
    print(f"1. Initial runs: {len(runs_df)}")

    result = parse_and_clean_runs_df(runs_df)
    filtered_df = result["filtered_df"]

    print(f"   After filtering early test runs: {len(filtered_df)}")
    print(f"   Columns dropped: {len(result['cols_dropped'])} {result['cols_dropped']}")

    print("\n2. Semantic categorization:")
    print(f"   OE eval columns: {len(result['oe_cols'])}")
    print(f"   Pretrain eval columns: {len(result['pretrain_cols'])}")

    print("\n3. Key-based semantic groups:")
    semantic_total = 0
    for cat in wconsts.KEY_SETS.keys():
        if result.get(cat):
            semantic_total += len(result[cat])
            print(f"   {cat}: {len(result[cat])}")

    print("\n4. After semantic categorization:")
    print(f"   Object columns remaining: {len(result['object_cols'])}")
    print(
        f"   Total semantically categorized: {semantic_total + len(result['oe_cols']) + len(result['pretrain_cols'])}"
    )

    print("\n5. Constant/NaN filtering on remaining columns:")
    print(f"   All NaN columns: {len(result['all_nan_cols'])}")
    print(f"   All constant columns: {len(result['all_constant_cols'])}")
    print(f"   Constant or NaN columns: {len(result['constant_or_nan_cols'])}")
    print(f"   Truly uncategorized: {len(result['truly_uncategorized'])}")

    print("\n6. Key-based categorization details:")
    for cat in wconsts.KEY_SETS.keys():
        if result[cat]:
            print(f"   {cat}: {len(result[cat])} {result[cat]}")

    if result["truly_uncategorized"]:
        print(
            f"\n7. Truly uncategorized columns analysis ({len(result['truly_uncategorized'])}):"
        )
        for i, col in enumerate(result["truly_uncategorized"], 1):
            if col in filtered_df.columns:
                col_data = filtered_df[col]
                n_unique = col_data.nunique(dropna=False)
                pct_non_nan = (col_data.notna().sum() / len(col_data)) * 100
                sample_values = col_data.dropna().unique()[:5].tolist()
                if len(sample_values) == 0:
                    sample_values = ["All NaN"]
                print(
                    f"   {i:2d}. {col:<35} | unique: {n_unique:3d} | non-NaN: {pct_non_nan:5.1f}% | sample: {sample_values}"
                )
            else:
                print(f"   {i:2d}. {col:<35} | NOT FOUND IN DATAFRAME")

    print("\n=== WANDB TAGS ===")
    if "wandb_tags" in filtered_df.columns:
        tags_data = filtered_df["wandb_tags"].dropna()
        if len(tags_data) > 0:
            unique_tags = tags_data.unique()
            print(f"Unique wandb_tags ({len(unique_tags)}):")
            for tag in sorted(unique_tags):
                print(f"   {tag}")
        else:
            print("No wandb_tags data available")
    else:
        print("wandb_tags column not found")

    print("\n=== ALL-NAN COLUMNS ===")
    if result["all_nan_cols"]:
        print(f"All-NaN columns ({len(result['all_nan_cols'])}):")
        for col in result["all_nan_cols"]:
            print(f"   {col}")
    else:
        print("No all-NaN columns")

    print("\n=== CONSTANT/NAN COLUMNS WITH VALUES ===")
    if result["constant_or_nan_cols"]:
        print(f"Constant or NaN columns ({len(result['constant_or_nan_cols'])}):")
        for col in result["constant_or_nan_cols"]:
            if col in filtered_df.columns:
                col_data = filtered_df[col]
                unique_values = col_data.dropna().unique()
                if len(unique_values) > 0:
                    constant_value = unique_values[0]
                    print(f"   {col:<35} = {constant_value}")
                else:
                    print(f"   {col:<35} = [All NaN]")
            else:
                print(f"   {col:<35} = [Not found]")
    else:
        print("No constant/NaN columns")

    print("\n=== REBUILT DATAFRAME ANALYSIS ===")
    from datadec.wandb_eval.parsing import rebuild_run_df

    rebuilt_df = rebuild_run_df(filtered_df, result)
    print(f"Rebuilt DataFrame shape: {rebuilt_df.shape}")
    print(f"Columns: {list(rebuilt_df.columns)}")

    print("\nNaN Analysis for each column:")
    for col in rebuilt_df.columns:
        total_rows = len(rebuilt_df)
        nan_count = rebuilt_df[col].isna().sum()
        nan_pct = (nan_count / total_rows) * 100
        print(f"  {col:<25} | NaNs: {nan_count:3d}/{total_rows} ({nan_pct:5.1f}%)")


if __name__ == "__main__":
    main()

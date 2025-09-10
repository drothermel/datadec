import pandas as pd

from datadec.wandb_eval import analysis_helpers, parsing
from datadec.wandb_eval import wandb_constants as wconsts

PARAMS_TO_DROP = [
    "run_date_rnp",
    "run_time_rnp",
]

DYNAMICS_TO_KEEP = [
    "max_step",
    "max_tokens",
    "max_epoch",
    "max_lr",
    "final_train_loss",
    "min_train_loss",
]
KEY_SETS = {
    "id_cols": wconsts.ID_KEYS,
    "status_cols": wconsts.STATUS_KEYS,
    "pretrain_hpm_cols": wconsts.PRETRAIN_HPM_KEYS,
    "core_hpm_cols": wconsts.CORE_HPM_KEYS,
    "dpo_hpm_cols": wconsts.DPO_HPM_KEYS,
    "eval_setting_cols": wconsts.EVAL_SETTING_KEYS,
    "x_axis_cols": wconsts.X_AXIS_KEYS,
    "summary_metrics_cols": wconsts.SUMMARY_METRICS_KEYS,
}


def extract_unverified_parsed_params(sample_runs):
    all_params = []
    for _, row in sample_runs.iterrows():
        run_name = row["run_name"]
        parsed_params = analysis_helpers.extract_hyperparameters(run_name)
        # Add training dynamics, need history df for this to wrok
        # dynamics = analysis_helpers.get_run_training_dynamics(history_df, run_id)
        run_data = {"run_name": run_name}
        run_data.update(
            {k: v for k, v in parsed_params.items() if k not in PARAMS_TO_DROP}
        )
        """
        if dynamics:
            filtered_dynamics = {
                k: v for k, v in dynamics.items() if k in DYNAMICS_TO_KEEP
            }
            run_data.update(filtered_dynamics)
        """
        all_params.append(run_data)
    return pd.DataFrame(all_params)


def categorize_columns_by_key_sets(all_cols):
    categorized = {name: [] for name in KEY_SETS.keys()}
    remaining_cols = list(all_cols)

    for category_name, key_list in KEY_SETS.items():
        matched_cols = [col for col in remaining_cols if col in key_list]
        categorized[category_name] = matched_cols
        remaining_cols = [col for col in remaining_cols if col not in matched_cols]

    categorized["truly_uncategorized"] = remaining_cols
    return categorized


def analyze_category_overlap(df, category_cols, category_name):
    if len(category_cols) < 2:
        return None

    overlaps = []
    for i, col1 in enumerate(category_cols):
        for j, col2 in enumerate(category_cols[i + 1 :], i + 1):
            if col1 in df.columns and col2 in df.columns:
                col1_data = df[col1]
                col2_data = df[col2]

                both_non_null = col1_data.notna() & col2_data.notna()
                if both_non_null.sum() > 0:
                    matching_values = (col1_data == col2_data) & both_non_null
                    overlap_pct = (matching_values.sum() / both_non_null.sum()) * 100
                    overlaps.append(
                        {
                            "col1": col1,
                            "col2": col2,
                            "overlap_pct": overlap_pct,
                            "shared_rows": both_non_null.sum(),
                        }
                    )

    return overlaps


def create_unified_parsed_df(runs_df):
    filtered_df = parsing.filter_early_test_runs(runs_df)
    cols_to_drop = [col for col in parsing.ALL_DROP_COLS if col in filtered_df.columns]
    if cols_to_drop:
        filtered_df = filtered_df.drop(columns=cols_to_drop)
    object_cols, nonobject_cols = parsing.split_obj_vs_nonobj_cols(filtered_df)
    nonobj_df = filtered_df[nonobject_cols]
    col_categories = parsing.filter_constant_and_nanconstant_cols(nonobj_df)
    pretrain_cols = parsing.filter_pretrain_metric_cols(
        filtered_df[col_categories["other"]]
    )
    uncategorized_cols = [
        col for col in col_categories["other"] if col not in pretrain_cols
    ]

    key_categorized = categorize_columns_by_key_sets(uncategorized_cols)

    return {
        "filtered_df": filtered_df,
        "object_cols": object_cols,
        "all_nan_cols": col_categories["all_nan"],
        "all_constant_cols": col_categories["all_constant"],
        "constant_or_nan_cols": col_categories["constant_or_nan"],
        "pretrain_cols": pretrain_cols,
        "uncategorized_cols": uncategorized_cols,
        "cols_dropped": cols_to_drop,
        **key_categorized,
    }


def main():
    runs_df, history_df = analysis_helpers.load_df()

    print("=== SEQUENTIAL FILTERING AND CATEGORIZATION ===\n")
    print(f"1. Initial runs: {len(runs_df)}")

    result = create_unified_parsed_df(runs_df)
    filtered_df = result["filtered_df"]

    print(f"   After filtering early test runs: {len(filtered_df)}")
    print(f"   Columns dropped: {len(result['cols_dropped'])} {result['cols_dropped']}")

    print("\n2. Column type split:")
    print(f"   Object columns: {len(result['object_cols'])}")
    print(
        f"   Non-object columns: {len(result['all_nan_cols']) + len(result['all_constant_cols']) + len(result['constant_or_nan_cols']) + len(result['pretrain_cols']) + len(result['uncategorized_cols'])}"
    )

    print("\n3. Non-object column categorization:")
    print(f"   All NaN columns: {len(result['all_nan_cols'])}")
    print(f"   All constant columns: {len(result['all_constant_cols'])}")
    print(f"   Constant or NaN columns: {len(result['constant_or_nan_cols'])}")
    print(f"   Pretrain eval columns: {len(result['pretrain_cols'])}")
    print(f"   Uncategorized useful columns: {len(result['uncategorized_cols'])}")

    print("\n4. Key-based categorization:")
    overlap_categories = [
        "core_hpm_cols",
        "dpo_hpm_cols",
        "x_axis_cols",
        "summary_metrics_cols",
        "all_constant_cols",
        "constant_or_nan_cols",
    ]

    for cat in KEY_SETS.keys():
        if result[cat]:
            print(f"   {cat}: {len(result[cat])} {result[cat]}")

            if cat in overlap_categories and len(result[cat]) > 1:
                overlaps = analyze_category_overlap(filtered_df, result[cat], cat)
                if overlaps:
                    print(f"     Overlap analysis for {cat}:")
                    for overlap in overlaps:
                        print(
                            f"       {overlap['col1']} ↔ {overlap['col2']}: {overlap['overlap_pct']:.1f}% overlap ({overlap['shared_rows']} rows)"
                        )
                else:
                    print(f"     No overlaps computed for {cat}")

    if result["truly_uncategorized"]:
        print(
            f"\n5. Truly uncategorized columns analysis ({len(result['truly_uncategorized'])}):"
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

    print("\n=== CATEGORY SAMPLES ===")
    if result["object_cols"]:
        print(f"Object columns (first 5): {result['object_cols'][:5]}")
    if result["all_nan_cols"]:
        print(f"All NaN columns (first 5): {result['all_nan_cols'][:5]}")
    if result["all_constant_cols"]:
        print(f"All constant columns (first 5): {result['all_constant_cols'][:5]}")
    if result["constant_or_nan_cols"]:
        print(
            f"Constant or NaN columns (first 5): {result['constant_or_nan_cols'][:5]}"
        )
    if result["pretrain_cols"]:
        print(f"Pretrain columns (first 5): {result['pretrain_cols'][:5]}")

    print(f"\n[Final] Filtered df shape: {filtered_df.shape}")


if __name__ == "__main__":
    main()

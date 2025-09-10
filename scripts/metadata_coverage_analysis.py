#!/usr/bin/env python3

import pandas as pd

from datadec import analysis_helpers
from datadec.constants import OLMES_METRICS, PPL_NAME_MAP

KEY_HP_COLUMNS = [
    "learning_rate",
    "model_size",
    "num_train_epochs",
    "max_train_steps",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "warmup_ratio",
    "weight_decay",
    "train_loss",
]

# Extended OLMES task list including discovered task names from actual data
EXTENDED_OLMES_TASKS = [
    # Original OLMES tasks
    "mmlu_average",
    "arc_challenge",
    "arc_easy",
    "boolq",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
    # Additional discovered tasks
    "mmlu",  # Individual MMLU task metrics
    "olmes_10_macro_avg",  # Aggregate OLMES metrics
]


def get_expected_olmes_columns():
    return [f"pretrain_eval_{metric}" for metric in OLMES_METRICS]


def get_expected_ppl_columns():
    return [col.replace("eval/", "pretrain_eval/") for col in PPL_NAME_MAP.keys()]


def get_columns_missing_in_last_week(runs_df):
    if "created_at" not in runs_df.columns:
        return []

    runs_df = runs_df.copy()
    runs_df["week"] = pd.to_datetime(runs_df["created_at"]).dt.to_period("W")
    weeks = sorted(runs_df["week"].unique())
    last_wk_runs = runs_df[runs_df["week"] == weeks[-1]]

    missing_columns = []
    for col in KEY_HP_COLUMNS:
        if col in runs_df.columns:
            last_week_coverage = (
                last_wk_runs[col].notna().sum() / len(last_wk_runs) * 100
            )
            if last_week_coverage == 0:
                missing_columns.append(col)

    return missing_columns


def analyze_week_metadata_evolution(runs_df):
    if "created_at" not in runs_df.columns:
        return None

    runs_df["week"] = pd.to_datetime(runs_df["created_at"]).dt.to_period("W")
    weeks = sorted(runs_df["week"].unique())
    first_wk_runs = runs_df[runs_df["week"] == weeks[0]]
    last_wk_runs = runs_df[runs_df["week"] == weeks[-1]]
    comparison = []
    for col in KEY_HP_COLUMNS:
        if col in runs_df.columns:
            week1_coverage = first_wk_runs[col].notna().sum() / len(first_wk_runs) * 100
            week4_coverage = last_wk_runs[col].notna().sum() / len(last_wk_runs) * 100
            comparison.append(
                {
                    "column": col,
                    "week1_coverage": week1_coverage,
                    "week4_coverage": week4_coverage,
                }
            )
    return {
        "weeks": (weeks[0], weeks[-1]),
        "week_counts": (len(first_wk_runs), len(last_wk_runs)),
        "comparison": comparison,
    }


def analyze_general_metadata_coverage(runs_df):
    column_coverage = []
    for col in runs_df.columns:
        non_null_count = runs_df[col].notna().sum()
        coverage_pct = non_null_count / len(runs_df) * 100
        try:
            sample_values = runs_df[runs_df[col].notna()][col].unique()
            sample_str = (
                str(sample_values[:3])[:50] + "..."
                if len(sample_values) > 3
                else str(sample_values)
            )
        except (TypeError, ValueError):
            non_null_data = runs_df[runs_df[col].notna()][col]
            sample_str = f"<complex_type: {type(non_null_data.iloc[0]).__name__}>"
        column_coverage.append(
            {
                "column": col,
                "non_null_count": non_null_count,
                "coverage_pct": coverage_pct,
                "sample_values": sample_str,
            }
        )
    coverage_df = pd.DataFrame(column_coverage).sort_values(
        "coverage_pct", ascending=False
    )
    return {"coverage_df": coverage_df}


def analyze_olmes_metrics(runs_df):
    expected_olmes_columns = get_expected_olmes_columns()
    present_olmes_columns = [
        col for col in expected_olmes_columns if col in runs_df.columns
    ]
    missing_olmes_columns = [
        col for col in expected_olmes_columns if col not in runs_df.columns
    ]
    return {
        "present_columns": present_olmes_columns,
        "missing_columns": missing_olmes_columns,
    }


def analyze_perplexity_metrics(runs_df):
    expected_ppl_columns = get_expected_ppl_columns()
    present_ppl_columns = [
        col for col in expected_ppl_columns if col in runs_df.columns
    ]
    missing_ppl_columns = [
        col for col in expected_ppl_columns if col not in runs_df.columns
    ]
    return {
        "present_columns": present_ppl_columns,
        "missing_columns": missing_ppl_columns,
    }


def analyze_remaining_eval_metrics(runs_df, olmes_columns, ppl_columns):
    non_olmes_non_ppl_cols = [
        col
        for col in runs_df.columns
        if col.startswith("pretrain_eval")
        and col not in olmes_columns
        and col not in ppl_columns
    ]
    return {
        "non_olmes_non_ppl_cols": non_olmes_non_ppl_cols,
    }


def analyze_remaining_column_patterns(remaining_columns):
    clean_names = []
    for col in remaining_columns:
        clean_name = col.replace("pretrain_eval_", "").replace("pretrain_eval/", "")
        if not clean_name.endswith("-validation/Perplexity"):
            clean_names.append(clean_name)

    olmes_task_matches = []
    olmes_task_non_matches = []
    extracted_metrics = set()

    for name in clean_names:
        matched_task = False

        for task in EXTENDED_OLMES_TASKS:
            if name.startswith(task + "_"):
                metric_part = name[len(task) + 1 :]
                extracted_metrics.add(metric_part)
                olmes_task_matches.append(name)
                matched_task = True
                break

        if not matched_task:
            olmes_task_non_matches.append(name)

    return {
        "total_remaining": len(clean_names),
        "olmes_task_matches": olmes_task_matches,
        "olmes_task_count": len(olmes_task_matches),
        "olmes_task_non_matches": olmes_task_non_matches,
        "olmes_task_non_matches_count": len(olmes_task_non_matches),
        "extracted_metrics": sorted(list(extracted_metrics)),
    }


def main():
    print("=== METADATA COVERAGE ANALYSIS ===\n")

    runs_df = analysis_helpers.load_runs_df()

    print(f"Total runs: {len(runs_df)}")

    if "created_at" in runs_df.columns:
        runs_df = runs_df.sort_values("created_at")
        print(
            f"Date range: {runs_df['created_at'].min()} to {runs_df['created_at'].max()}"
        )

    week4_missing_columns = get_columns_missing_in_last_week(runs_df)
    olmes_analysis = analyze_olmes_metrics(runs_df)
    ppl_analysis = analyze_perplexity_metrics(runs_df)
    remaining_analysis = analyze_remaining_eval_metrics(
        runs_df,
        get_expected_olmes_columns(),
        get_expected_ppl_columns(),
    )
    week4_available_remaining = [
        col
        for col in remaining_analysis["non_olmes_non_ppl_cols"]
        if col not in week4_missing_columns
    ]
    pattern_analysis = analyze_remaining_column_patterns(week4_available_remaining)

    print(f"\nColumns missing in Week 4: {week4_missing_columns}")
    print(
        f"\nPretrain_eval OLMES metrics: {len(olmes_analysis['present_columns'])} present, {len(olmes_analysis['missing_columns'])} missing"
    )
    print(
        f"Pretrain_eval PPL metrics: {len(ppl_analysis['present_columns'])} present, {len(ppl_analysis['missing_columns'])} missing"
    )
    print(
        f"\nPattern analysis of {pattern_analysis['total_remaining']} remaining columns:"
    )
    print(
        f"  OLMES {{task}}_<metric> pattern: {pattern_analysis['olmes_task_count']} columns"
    )
    print(
        f"  Non-OLMES task pattern: {pattern_analysis['olmes_task_non_matches_count']} columns"
    )
    if pattern_analysis["extracted_metrics"]:
        print(
            f"\nExtracted metric patterns from OLMES tasks ({len(pattern_analysis['extracted_metrics'])} unique):"
        )
        for metric in pattern_analysis["extracted_metrics"][:15]:
            print(f"  {metric}")
        if len(pattern_analysis["extracted_metrics"]) > 15:
            print(f"  ... and {len(pattern_analysis['extracted_metrics']) - 15} more")
    if pattern_analysis["olmes_task_non_matches_count"] > 0:
        print("\nColumns not starting with OLMES tasks:")
        for name in sorted(pattern_analysis["olmes_task_non_matches"])[:20]:
            print(f"  {name}")
        if pattern_analysis["olmes_task_non_matches_count"] > 20:
            print(
                f"  ... and {pattern_analysis['olmes_task_non_matches_count'] - 20} more"
            )


if __name__ == "__main__":
    main()

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from datadec.model_utils import param_to_numeric


ParamResults = Dict[str, Dict[str, Any]]
ParamInfo = Dict[str, Dict[str, Any]]
PerformanceTable = pd.DataFrame


def display_param_performance_table(
    param_results: ParamResults,
    results_df: Optional[pd.DataFrame] = None,
    param_info: Optional[ParamInfo] = None,
    sort_by_size: bool = True,
) -> PerformanceTable:
    if param_info is not None:
        param_counts = {param: info["n_points"] for param, info in param_info.items()}
    elif results_df is not None:
        param_counts = {
            param: len(results_df[results_df["params"] == param])
            for param in param_results.keys()
        }
    else:
        raise ValueError("Either param_info or results_df must be provided")

    # Prepare data for table
    table_data = []
    for param in param_results.keys():
        results = param_results[param]
        n_points = param_counts[param]
        param_size = param_to_numeric(param)

        # Format size string
        if param_size >= 1e9:
            f"{param_size / 1e9:.1f}B"
        else:
            f"{param_size / 1e6:.0f}M"

        row = {
            "Parameter": param,
            "N Points": n_points,
            "Decision Acc": results["decision_accuracy"],
            "Kendall's Tau": results.get("kendall_tau", np.nan),
            "NDCG@3": results.get("ndcg_3", results.get("ndcg@3", np.nan)),
            "NDCG@5": results.get("ndcg_5", results.get("ndcg@5", np.nan)),
            "NDCG@10": results.get("ndcg_10", results.get("ndcg@10", np.nan)),
            "NDCG@All": results.get("ndcg_all", np.nan),
            "Correlation": results["correlation"],
            "MSE": results["mse"],
            "RMSE": results["rmse"],
            "Std True": results["std_true"],
            "NMSE": results["nmse"],
            "NRMSE": results["nrmse"],
            "ABC Raw": results.get("abc_area_between_curves", np.nan),
            "ABC Per Progress": results.get("abc_area_per_progress", np.nan),
            "ABC Normalized": results.get("abc_area_normalized_combined", np.nan),
        }
        table_data.append(row)

    # Create DataFrame
    table_df = pd.DataFrame(table_data)

    # Sort by parameter size if requested
    if sort_by_size:
        table_df["_sort_key"] = table_df["Parameter"].apply(param_to_numeric)
        table_df = table_df.sort_values("_sort_key").drop("_sort_key", axis=1)

    return table_df


def print_param_performance_table(
    param_results: ParamResults,
    results_df: Optional[pd.DataFrame] = None,
    param_info: Optional[ParamInfo] = None,
    sort_by_size: bool = True,
) -> None:
    """
    Print a nicely formatted table of performance metrics by parameter.

    Args:
        param_results: Dictionary of parameter -> evaluation results from full_eval()
        results_df: DataFrame with 'params' column for counting data points (legacy)
        param_info: Dictionary of parameter -> info dict with 'n_points' (preferred)
        sort_by_size: If True, sort parameters by model size (default: True)
    """
    table_df = display_param_performance_table(
        param_results, results_df, param_info, sort_by_size
    )

    print("\n" + "=" * 120)
    print(
        "PARAMETER PERFORMANCE SUMMARY (Sorted by Size)"
        if sort_by_size
        else "PARAMETER PERFORMANCE SUMMARY"
    )
    print("=" * 120)

    # Print header - updated to include new metrics
    header = f"{'Param':<8} {'N':<6} {'Dec Acc':<8} {'K-Tau':<8} {'NDCG@3':<8} {'NDCG@5':<8} {'NDCG@10':<8} {'NDCG@All':<8} {'Corr':<8} {'RMSE':<8} {'NRMSE':<8}"
    print(header)
    print("-" * 140)

    # Helper function to format values for table display
    def format_table_value(value, decimals=3, width=8):
        if pd.isna(value) or np.isnan(value):
            return f"{'NaN':<{width}}"
        else:
            formatted = f"{value:.{decimals}f}"
            return f"{formatted:<{width}}"

    # Print data rows
    for _, row in table_df.iterrows():
        kendall_tau_key = "Kendall's Tau"
        data_row = (
            f"{row['Parameter']:<8} {row['N Points']:<6} "
            f"{format_table_value(row['Decision Acc'])} {format_table_value(row[kendall_tau_key])} "
            f"{format_table_value(row['NDCG@3'])} {format_table_value(row['NDCG@5'])} {format_table_value(row['NDCG@10'])} {format_table_value(row['NDCG@All'])} "
            f"{format_table_value(row['Correlation'])} {format_table_value(row['RMSE'], 1)} {format_table_value(row['NRMSE'])}"
        )
        print(data_row)

    print("=" * 120)


def display_overall_performance(overall_results: Dict[str, Any]) -> None:
    """
    Display overall performance metrics in a clean format.

    Args:
        overall_results: Results dictionary from full_eval()
    """
    print("\n" + "=" * 60)
    print("OVERALL BASELINE PERFORMANCE")
    print("=" * 60)

    # Helper function to format values, showing NaN clearly
    def format_value(value, decimals=3, is_int=False):
        if np.isnan(value):
            return "NaN"
        elif is_int:
            return f"{value:.0f}"
        else:
            return f"{value:.{decimals}f}"

    # Core metrics
    print(f"Decision Accuracy:    {format_value(overall_results['decision_accuracy'])}")
    print(
        f"Kendall's Tau:        {format_value(overall_results.get('kendall_tau', np.nan))}"
    )
    print(f"Correlation:          {format_value(overall_results['correlation'])}")
    print(f"RMSE:                 {format_value(overall_results['rmse'], 1)}")
    print(f"NRMSE:                {format_value(overall_results['nrmse'])}")

    # NDCG metrics
    print(
        f"NDCG@3:               {format_value(overall_results.get('ndcg_3', overall_results.get('ndcg@3', np.nan)))}"
    )
    print(
        f"NDCG@5:               {format_value(overall_results.get('ndcg_5', overall_results.get('ndcg@5', np.nan)))}"
    )
    print(
        f"NDCG@10:              {format_value(overall_results.get('ndcg_10', overall_results.get('ndcg@10', np.nan)))}"
    )
    print(
        f"NDCG@All:             {format_value(overall_results.get('ndcg_all', np.nan))}"
    )

    # Error metrics with normalization factor
    print("\nDetailed Error Metrics:")
    print(f"MSE:                  {format_value(overall_results['mse'], 1)}")
    print(f"NMSE:                 {format_value(overall_results['nmse'])}")
    print(
        f"Std(true):            {format_value(overall_results['std_true'], 1)}  (normalization factor for NRMSE)"
    )

    # Within-param metrics if available
    if "within_param_decision_accuracy" in overall_results:
        print("\nWithin-Parameter Metrics:")
        print(
            f"Within-Param Dec Acc: {overall_results['within_param_decision_accuracy']:.3f}"
        )
        print(
            f"Within-Param NDCG@5:  {overall_results.get('within_param_ndcg_5', 0.0):.3f}"
        )

    # ABC metrics if available
    if any(key.startswith("abc_") for key in overall_results.keys()):
        print("\nArea Between Curves Metrics:")
        print(
            f"ABC Raw:              {format_value(overall_results.get('abc_area_between_curves', np.nan), 1)}"
        )
        print(
            f"ABC Per Progress:     {format_value(overall_results.get('abc_area_per_progress', np.nan))}"
        )
        print(
            f"ABC Normalized:       {format_value(overall_results.get('abc_area_normalized_combined', np.nan))}"
        )

    print("=" * 60)


def display_steps_in_df(df: pd.DataFrame) -> None:
    return (
        df.groupby(["params", "data"])
        .agg(
            step_count=("step", "count"),
            max_step=("step", "max"),
            steps=("step", "unique"),
            total_steps=("total_steps", "first"),
            warmup_steps=("warmup_steps", "first"),
            early_window_end_step=("early_window_end_step", "first"),
            model_size=("model_size", "first"),
        )
        .sort_values(by="model_size", ascending=False)
        .drop(columns=["model_size"])
    )

from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import datadec.constants
from ft_pred.metrics.utils import compute_comprehensive_metrics
from ft_pred.analysis.core import get_ordered_models
from ft_pred.analysis.validation import apply_max_step_filter


def calculate_confidence_intervals(
    data: pd.Series, confidence: float = 0.95
) -> Tuple[float, float]:
    assert 0 < confidence < 1, "Confidence level must be between 0 and 1"
    assert len(data) > 0, "Data series cannot be empty"

    data_clean = data.dropna()
    assert len(data_clean) > 0, "No valid data points after removing NaN values"

    alpha = 1 - confidence
    degrees_freedom = len(data_clean) - 1

    mean = data_clean.mean()
    sem = stats.sem(data_clean)

    margin_error = stats.t.ppf(1 - alpha / 2, degrees_freedom) * sem

    return (mean - margin_error, mean + margin_error)


def detect_overfitting_patterns(
    train_r2: pd.Series, eval_r2: pd.Series
) -> Dict[str, Any]:
    assert len(train_r2) == len(eval_r2), (
        "Train and eval R² series must have same length"
    )
    assert len(train_r2) > 0, "Cannot analyze empty data series"

    valid_mask = ~(train_r2.isna() | eval_r2.isna())
    train_clean = train_r2[valid_mask]
    eval_clean = eval_r2[valid_mask]

    if len(train_clean) == 0:
        return {
            "status": "insufficient_data",
            "n_valid_pairs": 0,
            "overfitting_fraction": 0.0,
            "mean_gap": 0.0,
            "correlation": 0.0,
        }

    r2_gap = train_clean - eval_clean
    threshold = max(0.05, r2_gap.std() * 2)
    overfitting_mask = r2_gap > threshold

    correlation = train_clean.corr(eval_clean)

    return {
        "status": "analyzed",
        "n_valid_pairs": len(train_clean),
        "overfitting_fraction": overfitting_mask.mean(),
        "mean_gap": r2_gap.mean(),
        "max_gap": r2_gap.max(),
        "correlation": correlation,
        "potential_data_leakage": (eval_clean > train_clean).mean() > 0.1,
    }


def summarize_performance_by_group(
    results_df: pd.DataFrame, group_col: str
) -> pd.DataFrame:
    assert group_col in results_df.columns, (
        f"Group column '{group_col}' not found in DataFrame"
    )
    assert "eval_r2" in results_df.columns, (
        "Missing eval_r2 column for performance summary"
    )

    summary = (
        results_df.groupby(group_col)["eval_r2"]
        .agg(
            [
                ("mean", "mean"),
                ("std", "std"),
                ("min", "min"),
                ("max", "max"),
                ("count", "count"),
            ]
        )
        .reset_index()
    )

    for _, row in summary.iterrows():
        group_data = results_df[results_df[group_col] == row[group_col]]["eval_r2"]
        ci_low, ci_high = calculate_confidence_intervals(group_data)
        summary.loc[summary[group_col] == row[group_col], "ci_low"] = ci_low
        summary.loc[summary[group_col] == row[group_col], "ci_high"] = ci_high

    return summary.sort_values("mean", ascending=False)


def analyze_seed_data_density(
    seed_data: pd.DataFrame, target_metric: str, analysis_mode: str = "dual_filter"
) -> pd.DataFrame:
    assert target_metric in seed_data.columns, (
        f"Target metric '{target_metric}' not found in data"
    )
    assert analysis_mode in ["dual_filter", "eval_focused"], (
        f"Invalid analysis mode: {analysis_mode}"
    )

    first_recipe = sorted(seed_data["data"].unique())[0]
    max_steps = datadec.constants.MAX_STEP_TO_USE
    ppl_metrics = datadec.constants.PPL_TYPES
    eval_metrics = [
        col
        for col in seed_data.columns
        if any(metric in col for metric in datadec.constants.METRIC_NAMES)
    ]

    density_results = []

    for params in get_ordered_models(seed_data):
        max_step = max_steps.get(params, float("inf"))
        recipe_data = seed_data[
            (seed_data["params"] == params) & (seed_data["data"] == first_recipe)
        ]
        available_seeds = sorted(recipe_data["seed"].unique())

        seed_data_by_seed = {}
        max_total_count = 0

        for seed in range(5):
            if seed in available_seeds:
                seed_data_subset = apply_max_step_filter(
                    recipe_data[recipe_data["seed"] == seed], max_step
                )
                seed_data_by_seed[seed] = seed_data_subset
                max_total_count = max(max_total_count, len(seed_data_subset))

        for seed in range(5):
            if seed in seed_data_by_seed:
                seed_data_subset = seed_data_by_seed[seed]

                if analysis_mode == "dual_filter":
                    ppl_all_nan = seed_data_subset[ppl_metrics].isna().all(axis=1)
                    eval_any_val = seed_data_subset[eval_metrics].notna().any(axis=1)
                    eval_all_nan = seed_data_subset[eval_metrics].isna().all(axis=1)
                    ppl_any_val = seed_data_subset[ppl_metrics].notna().any(axis=1)

                    filtered_ppl_subset = seed_data_subset[
                        ~(ppl_all_nan & eval_any_val)
                    ]
                    filtered_eval_subset = seed_data_subset[
                        ~(eval_all_nan & ppl_any_val)
                    ]

                    ppl_nonnan_count = len(
                        filtered_ppl_subset[filtered_ppl_subset[target_metric].notna()]
                    )
                    eval_nonnan_count = len(
                        filtered_eval_subset[
                            filtered_eval_subset[target_metric].notna()
                        ]
                    )

                    for filter_type, nonnan_count in [
                        ("ppl_focused", ppl_nonnan_count),
                        ("eval_focused", eval_nonnan_count),
                    ]:
                        density_results.append(
                            {
                                "model_size": params,
                                "filter_type": filter_type,
                                "seed": f"seed_{seed}_nonnan",
                                "recipe": first_recipe,
                                "non_nan_points": nonnan_count,
                                "total_points": len(seed_data_subset),
                                "max_step_cap": max_step,
                            }
                        )
                else:
                    eval_all_nan = seed_data_subset[eval_metrics].isna().all(axis=1)
                    ppl_any_val = seed_data_subset[ppl_metrics].notna().any(axis=1)

                    filtered_subset = seed_data_subset[~(eval_all_nan & ppl_any_val)]
                    nonnan_count = len(
                        filtered_subset[filtered_subset[target_metric].notna()]
                    )

                    density_results.append(
                        {
                            "model_size": params,
                            "seed": f"seed_{seed}_nonnan",
                            "recipe": first_recipe,
                            "non_nan_points": nonnan_count,
                            "total_points": len(seed_data_subset),
                            "max_step_cap": max_step,
                        }
                    )
            else:
                if analysis_mode == "dual_filter":
                    for filter_type in ["ppl_focused", "eval_focused"]:
                        density_results.append(
                            {
                                "model_size": params,
                                "filter_type": filter_type,
                                "seed": f"seed_{seed}_nonnan",
                                "recipe": first_recipe,
                                "non_nan_points": 0,
                                "total_points": 0,
                                "max_step_cap": max_step,
                            }
                        )
                else:
                    density_results.append(
                        {
                            "model_size": params,
                            "seed": f"seed_{seed}_nonnan",
                            "recipe": first_recipe,
                            "non_nan_points": 0,
                            "total_points": 0,
                            "max_step_cap": max_step,
                        }
                    )

        if analysis_mode == "dual_filter":
            for filter_type in ["ppl_focused", "eval_focused"]:
                density_results.append(
                    {
                        "model_size": params,
                        "filter_type": filter_type,
                        "seed": "max_total",
                        "recipe": first_recipe,
                        "non_nan_points": max_total_count,
                        "total_points": max_total_count,
                        "max_step_cap": max_step,
                    }
                )
        else:
            density_results.append(
                {
                    "model_size": params,
                    "seed": "max_total",
                    "recipe": first_recipe,
                    "non_nan_points": max_total_count,
                    "total_points": max_total_count,
                    "max_step_cap": max_step,
                }
            )

    return pd.DataFrame(density_results)


def analyze_seed_data_density_filtered(seed_data: pd.DataFrame) -> pd.DataFrame:
    return analyze_seed_data_density(seed_data, "pile-valppl", "dual_filter")


def analyze_mmlu_eval_focused_density(seed_data: pd.DataFrame) -> pd.DataFrame:
    return analyze_seed_data_density(
        seed_data, "mmlu_average_correct_prob", "eval_focused"
    )


def analyze_prediction_horizons(
    predictions, targets, target_percentages, actual_percentages
):
    return analyze_horizon_performance(
        None, targets, predictions, np.array(target_percentages), actual_percentages
    )


def analyze_horizon_performance(
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    predictions: np.ndarray,
    target_percentages_actual: np.ndarray,
    target_percentages_list: List[float],
    std_predictions: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    horizon_metrics = {}

    for target_pct in target_percentages_list:
        tolerance = 2.0
        mask = np.abs(target_percentages_actual - target_pct) < tolerance

        if np.sum(mask) > 0:
            pct_true = y_eval[mask]
            pct_pred = predictions[mask]

            if len(pct_true) > 1:
                metrics_result = compute_comprehensive_metrics(pct_pred, pct_true)
                r2 = metrics_result["r2"]
                mse = metrics_result["mse"]
                rmse = metrics_result["rmse"]

                metrics = {
                    "r2": r2,
                    "mse": mse,
                    "rmse": rmse,
                    "n_samples": len(pct_true),
                    "mean_actual": np.mean(pct_true),
                    "mean_predicted": np.mean(pct_pred),
                    "std_actual": np.std(pct_true),
                    "std_predicted": np.std(pct_pred),
                }

                if std_predictions is not None:
                    pct_std = std_predictions[mask]
                    metrics["uncertainty_mean"] = np.mean(pct_std)
                    metrics["uncertainty_std"] = np.std(pct_std)

                horizon_metrics[target_pct] = metrics

    return horizon_metrics


def analyze_degradation_patterns(
    horizon_metrics: Dict[float, Dict[str, Any]],
) -> Dict[str, Any]:
    assert len(horizon_metrics) >= 3, (
        f"Need at least 3 horizons for degradation analysis, got {len(horizon_metrics)}"
    )

    horizons = sorted(horizon_metrics.keys())
    r2_values = [horizon_metrics[h]["r2"] for h in horizons]
    mse_values = [horizon_metrics[h]["mse"] for h in horizons]

    r2_slope, r2_intercept, r2_corr, r2_pvalue, r2_stderr = stats.linregress(
        horizons, r2_values
    )
    mse_slope, mse_intercept, mse_corr, mse_pvalue, mse_stderr = stats.linregress(
        horizons, mse_values
    )

    degradation_analysis = {
        "r2_degradation": {
            "slope": r2_slope,
            "intercept": r2_intercept,
            "correlation": r2_corr,
            "p_value": r2_pvalue,
            "std_error": r2_stderr,
            "degradation_per_10pct": r2_slope * 10,
        },
        "mse_degradation": {
            "slope": mse_slope,
            "intercept": mse_intercept,
            "correlation": mse_corr,
            "p_value": mse_pvalue,
            "std_error": mse_stderr,
            "increase_per_10pct": mse_slope * 10,
        },
        "optimal_horizon": horizons[np.argmax(r2_values)],
        "worst_horizon": horizons[np.argmin(r2_values)],
        "horizon_range": {
            "best_r2": max(r2_values),
            "worst_r2": min(r2_values),
            "r2_range": max(r2_values) - min(r2_values),
        },
    }

    return degradation_analysis

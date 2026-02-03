from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

def _calc_mse(preds: np.ndarray, true_vals: np.ndarray) -> float:
    return float(np.mean((preds - true_vals) ** 2))


def _calc_rmse(preds: np.ndarray, true_vals: np.ndarray) -> float:
    return float(np.sqrt(_calc_mse(preds, true_vals)))


def _calc_r2(preds: np.ndarray, true_vals: np.ndarray) -> float:
    ss_res = float(np.sum((true_vals - preds) ** 2))
    ss_tot = float(np.sum((true_vals - np.mean(true_vals)) ** 2))
    return float('nan') if ss_tot == 0 else 1.0 - ss_res / ss_tot



def apply_seed_validation_1b(df: pd.DataFrame, min_steps: int = 20) -> pd.DataFrame:
    assert "params" in df.columns, (
        "DataFrame must have 'params' column for seed validation"
    )
    assert "seed" in df.columns, "DataFrame must have 'seed' column for seed validation"
    assert "step" in df.columns, "DataFrame must have 'step' column for seed validation"

    mask_1b = df["params"] == "1B"
    df_1b = df[mask_1b]

    if len(df_1b) > 0:
        valid_1b_seeds = df_1b.groupby("seed")["step"].max()
        valid_seeds = valid_1b_seeds[valid_1b_seeds >= min_steps].index
        df.loc[mask_1b, "keep"] = df.loc[mask_1b, "seed"].isin(valid_seeds)
        df = df[~mask_1b | df.get("keep", True)]
        df = df.drop(columns=["keep"], errors="ignore")

    return df


def apply_max_step_filter(data: pd.DataFrame, max_step: float) -> pd.DataFrame:
    return data[data["step"] <= max_step] if max_step != float("inf") else data


def analyze_sweep_results(results_df: pd.DataFrame) -> None:
    print("\n=== Sweep Results Analysis ===")
    print(f"Total successful experiments: {len(results_df)}")

    valid_results = results_df.dropna(subset=["eval_r2"])
    print(f"Experiments with valid eval R²: {len(valid_results)}")

    if len(valid_results) == 0:
        print("No valid results to analyze!")
        return

    best_eval_idx = valid_results["eval_r2"].idxmax()
    best_result = valid_results.loc[best_eval_idx]
    print(f"\nBest Eval R²: {best_result['eval_r2']:.4f}")
    print(f"Model type: {best_result.get('model_type', 'elasticnet')}")
    print(f"Holdout param: {best_result['holdout_param']}")
    if best_result.get("model_type") == "gpssm":
        print(f"Sequence length: {best_result['sequence_length']}")
    else:
        print(f"Features: {best_result.get('features', 'N/A')}")

    if best_result.get("model_type") in ["gp", "gpssm"]:
        print(
            f"Alpha: {best_result['alpha']}, Kernel type: {best_result.get('kernel_type', 'rbf')}"
        )
        print(f"N restarts: {best_result.get('n_restarts_optimizer', 9)}")
        if "eval_std_mean" in best_result:
            print(f"Eval uncertainty (std mean): {best_result['eval_std_mean']:.4f}")
        if "train_log_marginal_likelihood" in best_result:
            print(
                f"Log marginal likelihood: {best_result['train_log_marginal_likelihood']:.4f}"
            )
    elif best_result.get("model_type") == "elasticnet":
        print(
            f"Alpha: {best_result['alpha']}, L1 ratio: {best_result.get('l1_ratio', 0.5)}"
        )

    print(f"Train R²: {best_result['train_r2']:.4f}")
    print(f"Train/Eval gap: {best_result['train_r2'] - best_result['eval_r2']:.4f}")

    print("\n=== Performance by Holdout Parameter ===")
    holdout_performance = (
        valid_results.groupby("holdout_param")
        .agg({"eval_r2": ["mean", "std", "min", "max", "count"], "train_r2": ["mean"]})
        .round(4)
    )
    holdout_performance.columns = [
        "eval_r2_mean",
        "eval_r2_std",
        "eval_r2_min",
        "eval_r2_max",
        "n_experiments",
        "train_r2_mean",
    ]
    holdout_performance = holdout_performance.sort_values(
        "eval_r2_mean", ascending=False
    )
    print(holdout_performance.to_string())

    if "features" in valid_results.columns:
        print("\n=== Performance by Feature Combination ===")
        feature_performance = (
            valid_results.groupby("features")
            .agg({"eval_r2": ["mean", "std", "min", "max", "count"]})
            .round(4)
        )
        feature_performance.columns = [
            "eval_r2_mean",
            "eval_r2_std",
            "eval_r2_min",
            "eval_r2_max",
            "n_experiments",
        ]
        feature_performance = feature_performance.sort_values(
            "eval_r2_mean", ascending=False
        )
        print(feature_performance.to_string())

    if "sequence_length" in valid_results.columns:
        print("\n=== Performance by Sequence Length ===")
        sequence_performance = (
            valid_results.groupby("sequence_length")
            .agg({"eval_r2": ["mean", "std", "min", "max", "count"]})
            .round(4)
        )
        sequence_performance.columns = [
            "eval_r2_mean",
            "eval_r2_std",
            "eval_r2_min",
            "eval_r2_max",
            "n_experiments",
        ]
        sequence_performance = sequence_performance.sort_values(
            "eval_r2_mean", ascending=False
        )
        print(sequence_performance.to_string())

    print("\n=== Overfitting Analysis (Train-Eval Gap) ===")
    valid_results["train_eval_gap"] = (
        valid_results["train_r2"] - valid_results["eval_r2"]
    )

    if "l1_ratio" in valid_results.columns:
        gap_analysis = (
            valid_results.groupby(["alpha", "l1_ratio"])
            .agg({"train_eval_gap": ["mean", "std"], "eval_r2": ["mean"]})
            .round(4)
        )
        gap_analysis.columns = ["gap_mean", "gap_std", "eval_r2_mean"]
        gap_analysis = gap_analysis.sort_values("gap_mean")
        print(gap_analysis.to_string())
    else:
        gap_analysis = (
            valid_results.groupby("alpha")
            .agg({"train_eval_gap": ["mean", "std"], "eval_r2": ["mean"]})
            .round(4)
        )
        gap_analysis.columns = ["gap_mean", "gap_std", "eval_r2_mean"]
        gap_analysis = gap_analysis.sort_values("gap_mean")
        print(gap_analysis.to_string())

    if "eval_std_mean" in valid_results.columns:
        print("\n=== GP/GPSSM Uncertainty Analysis ===")
        gp_gpssm_results = valid_results[
            valid_results.get("model_type").isin(["gp", "gpssm"])
        ]
        if len(gp_gpssm_results) > 0:
            uncertainty_stats = gp_gpssm_results["eval_std_mean"].describe()
            print("Eval prediction uncertainty (std) statistics:")
            print(f"  Mean: {uncertainty_stats['mean']:.4f}")
            print(f"  Std: {uncertainty_stats['std']:.4f}")
            print(f"  Min: {uncertainty_stats['min']:.4f}")
            print(f"  Max: {uncertainty_stats['max']:.4f}")

            if "train_log_marginal_likelihood" in gp_gpssm_results.columns:
                lml_stats = gp_gpssm_results["train_log_marginal_likelihood"].describe()
                print("\nLog marginal likelihood statistics:")
                print(f"  Mean: {lml_stats['mean']:.2f}")
                print(f"  Std: {lml_stats['std']:.2f}")
                print(f"  Min: {lml_stats['min']:.2f}")
                print(f"  Max: {lml_stats['max']:.2f}")


def compare_models_across_horizons(
    models_results: Dict[str, Dict[float, Dict[str, Any]]],
) -> pd.DataFrame:
    comparison_data = []

    for model_name, horizon_metrics in models_results.items():
        for horizon, metrics in horizon_metrics.items():
            comparison_data.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "r2": metrics["r2"],
                    "mse": metrics["mse"],
                    "rmse": metrics["rmse"],
                    "n_samples": metrics["n_samples"],
                    "uncertainty_mean": metrics.get("uncertainty_mean", None),
                }
            )

    comparison_df = pd.DataFrame(comparison_data)

    if len(comparison_df) > 0:
        pivot_r2 = comparison_df.pivot(index="horizon", columns="model", values="r2")
        pivot_mse = comparison_df.pivot(index="horizon", columns="model", values="mse")

        comparison_df.attrs["pivot_r2"] = pivot_r2
        comparison_df.attrs["pivot_mse"] = pivot_mse

    return comparison_df


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
        tolerance = 2.0  # Hardcode since removed from config
        mask = np.abs(target_percentages_actual - target_pct) < tolerance

        if np.sum(mask) > 0:
            pct_true = y_eval[mask]
            pct_pred = predictions[mask]

            if len(pct_true) > 1:
                r2 = _calc_r2(pct_pred, pct_true)
                mse = _calc_mse(pct_pred, pct_true)
                rmse = _calc_rmse(pct_pred, pct_true)

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


def analyze_cv_folds(
    X: pd.DataFrame, groups: pd.Series, cv_folds: int = 3
) -> Dict[str, Any]:
    assert len(X) == len(groups), "X and groups must have same length"
    assert cv_folds > 1, "cv_folds must be greater than 1"

    fold_info = []

    unique_groups = list(pd.unique(groups))
    folds = [[] for _ in range(cv_folds)]
    for idx, group in enumerate(unique_groups):
        folds[idx % cv_folds].append(group)

    for fold_idx, val_groups in enumerate(folds):
        val_groups = set(val_groups)
        val_mask = groups.isin(val_groups)
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])

        fold_info.append(
            {
                "fold": fold_idx,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "train_groups": len(train_groups),
                "val_groups": len(val_groups),
                "group_overlap": len(train_groups.intersection(val_groups)),
            }
        )

    fold_df = pd.DataFrame(fold_info)

    analysis = {
        "n_folds": cv_folds,
        "fold_details": fold_df,
        "total_groups": len(set(groups)),
        "avg_train_samples": fold_df["train_samples"].mean(),
        "avg_val_samples": fold_df["val_samples"].mean(),
        "group_leakage": fold_df["group_overlap"].sum() > 0,
    }

    return analysis

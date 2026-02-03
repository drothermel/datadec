from typing import List
import pandas as pd

AVAILABLE_TARGET_TYPES = ["best_perf_ppl", "final_perf"]


def calculate_best_perf_ppl_target(group: pd.DataFrame, metric: str) -> float:
    return float(group[metric].min())


def calculate_final_perf_target(group: pd.DataFrame, metric: str) -> float:
    group_sorted = group.sort_values("step")
    return float(group_sorted[metric].iloc[-1])


def extract_targets(
    filtered_df: pd.DataFrame, metric: str, target_type: str
) -> pd.DataFrame:
    assert target_type in AVAILABLE_TARGET_TYPES, (
        f"Unknown target type '{target_type}'. Available types: {AVAILABLE_TARGET_TYPES}"
    )

    def compute_group_target(group: pd.DataFrame) -> float:
        if target_type == "best_perf_ppl":
            return calculate_best_perf_ppl_target(group, metric)
        elif target_type == "final_perf":
            return calculate_final_perf_target(group, metric)
        raise ValueError(f"Unsupported target type: {target_type}")

    target_df = (
        filtered_df.groupby(["params", "data"])
        .apply(compute_group_target, include_groups=False)
        .reset_index()
    )
    target_df = target_df.rename(columns={0: f"target_{target_type}_{metric}"})

    return target_df

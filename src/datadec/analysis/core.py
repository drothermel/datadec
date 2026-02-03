from typing import Dict, List, Any
import pandas as pd
from datadec.model_utils import param_to_numeric


def create_analysis_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    assert len(results) > 0, "Results list cannot be empty"

    flattened_results = []
    for result in results:
        assert isinstance(result, dict), "Each result must be a dictionary"
        flat_result = {**result}

        if "train" in result and "metrics" in result["train"]:
            for key, value in result["train"]["metrics"].items():
                flat_result[f"train_{key}"] = value
            flat_result["n_train_samples"] = result["train"]["n_samples"]

        if "eval" in result and "metrics" in result["eval"]:
            for key, value in result["eval"]["metrics"].items():
                flat_result[f"eval_{key}"] = value
            flat_result["n_eval_samples"] = result["eval"]["n_samples"]

        flattened_results.append(flat_result)

    return pd.DataFrame(flattened_results)


def convert_model_size_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    assert "params" in df.columns, (
        "DataFrame must have 'params' column for model size conversion"
    )

    df = df.copy()
    df["model_size_numeric"] = df["params"].apply(param_to_numeric)
    return df


def get_ordered_models(data: pd.DataFrame) -> List[str]:
    model_col = "params" if "params" in data.columns else "model_size"
    return sorted(data[model_col].unique(), key=param_to_numeric)


def get_recipe_order(
    df: pd.DataFrame, metric: str = "pile-valppl", ascending: bool = True
) -> List[str]:
    assert "data" in df.columns, "DataFrame must have 'data' column for recipe ordering"
    assert metric in df.columns, (
        f"DataFrame must have '{metric}' column for recipe ordering"
    )

    df = df.dropna(subset=[metric])
    assert len(df) > 0, (
        f"No valid data found for metric '{metric}' after dropping NaN values"
    )

    recipe_performance = (
        df.groupby("data")[metric].mean().sort_values(ascending=ascending)
    )
    return list(recipe_performance.index)

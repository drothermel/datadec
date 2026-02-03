from typing import List, Tuple
import pandas as pd

from datadec import DataDecide
from dr_plotter.figure import FigureManager
from dr_plotter.figure_config import FigureConfig
from datadec.data_filtering import get_filtered_data
from datadec.analysis import extract_targets
from datadec.analysis import (
    convert_model_size_to_numeric,
    apply_seed_validation_1b,
    get_recipe_order,
)


def identify_proxy_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    proxy_pairs = []

    task_bases = [
        "arc_challenge",
        "arc_easy",
        "boolq",
        "csqa",
        "hellaswag",
        "openbookqa",
        "piqa",
        "socialiqa",
        "winogrande",
    ]

    for task in task_bases:
        choice_col = f"{task}_correct_choice"
        prob_col = f"{task}_correct_prob"

        if choice_col in df.columns and prob_col in df.columns:
            proxy_pairs.append((choice_col, prob_col))

    if (
        "mmlu_average_correct_choice" in df.columns
        and "mmlu_average_correct_prob" in df.columns
    ):
        proxy_pairs.append(("mmlu_average_correct_choice", "mmlu_average_correct_prob"))

    return proxy_pairs


def main() -> None:
    dd = DataDecide()

    exclude_params = ["750M"]
    filtered_df, validated_params, validated_data = get_filtered_data(
        dd, "all", "all", "pile-valppl", exclude_params=exclude_params
    )

    filtered_df = apply_seed_validation_1b(filtered_df)

    target_df = extract_targets(filtered_df, "pile-valppl", "final_perf")

    final_data = target_df.rename(
        columns={"target_final_perf_pile-valppl": "pile-valppl"}
    )
    final_data = convert_model_size_to_numeric(final_data)
    recipe_order = get_recipe_order(final_data)

    selected_recipes = [
        recipe_order[0],
        recipe_order[len(recipe_order) // 2],
        recipe_order[-1],
    ]

    final_data_filtered = final_data[final_data["data"].isin(selected_recipes)]

    proxy_pairs = identify_proxy_pairs(final_data_filtered)

    if not proxy_pairs:
        print("No proxy pairs found")
        return

    n_proxies = len(proxy_pairs)
    with FigureManager(
        figure=FigureConfig(rows=2, cols=n_proxies, figsize=(n_proxies * 6, 12))
    ) as fm:
        for i, (choice_col, prob_col) in enumerate(proxy_pairs):
            choice_data = final_data_filtered.dropna(subset=[choice_col])
            prob_data = final_data_filtered.dropna(subset=[prob_col])

            if len(choice_data) > 0:
                fm.plot(
                    "line",
                    0,
                    i,
                    choice_data,
                    x="model_size_numeric",
                    y=choice_col,
                    hue_by="data",
                    title=f"{choice_col} vs Model Size",
                )
                fm.get_axes(0, i).set_xscale("log")

            if len(prob_data) > 0:
                fm.plot(
                    "line",
                    1,
                    i,
                    prob_data,
                    x="model_size_numeric",
                    y=prob_col,
                    hue_by="data",
                    title=f"{prob_col} vs Model Size",
                )
                fm.get_axes(1, i).set_xscale("log")

        fm.finalize_layout()
        fm.fig.savefig(
            "output/metric_proxy_comparison.png", dpi=300, bbox_inches="tight"
        )

    print("Generated: output/metric_proxy_comparison.png")


if __name__ == "__main__":
    main()

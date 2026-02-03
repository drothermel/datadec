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


def melt_olmes_tasks(df: pd.DataFrame) -> pd.DataFrame:
    task_columns = []

    if "mmlu_average_correct_prob" in df.columns:
        task_columns.append("mmlu_average_correct_prob")

    non_mmlu_tasks = [
        "arc_challenge_correct_prob",
        "arc_easy_correct_prob",
        "boolq_correct_prob",
        "csqa_correct_prob",
        "hellaswag_correct_prob",
        "openbookqa_correct_prob",
        "piqa_correct_prob",
        "socialiqa_correct_prob",
        "winogrande_correct_prob",
    ]

    available_tasks = [col for col in non_mmlu_tasks if col in df.columns]
    task_columns.extend(available_tasks)

    if not task_columns:
        return pd.DataFrame()

    melted = df.melt(
        id_vars=["params", "data", "model_size_numeric"],
        value_vars=task_columns,
        var_name="task",
        value_name="task_score",
    )

    melted["task"] = (
        melted["task"]
        .str.replace("_correct_prob", "")
        .str.replace("mmlu_average", "mmlu_avg")
    )

    return melted.dropna(subset=["task_score"])


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

    olmes_data = melt_olmes_tasks(final_data)

    if len(olmes_data) == 0:
        print("No OLMES task data found")
        return

    with FigureManager(figure=FigureConfig(rows=1, cols=3, figsize=(18, 6))) as fm:
        for i, recipe in enumerate(selected_recipes):
            recipe_data = olmes_data[olmes_data["data"] == recipe]

            fm.plot(
                "line",
                0,
                i,
                recipe_data,
                x="model_size_numeric",
                y="task_score",
                hue_by="task",
                title=f"OLMES Tasks Scaling: {recipe}",
            )
            fm.get_axes(0, i).set_xscale("log")

        fm.finalize_layout()
        fm.fig.savefig("output/olmes_tasks_scaling.png", dpi=300, bbox_inches="tight")

    print("Generated: output/olmes_tasks_scaling.png")


if __name__ == "__main__":
    main()

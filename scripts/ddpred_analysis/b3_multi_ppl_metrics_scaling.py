import pandas as pd
import datadec.constants

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


def melt_ppl_metrics(df: pd.DataFrame) -> pd.DataFrame:
    ppl_metrics = datadec.constants.PPL_TYPES
    available_ppl_metrics = [col for col in ppl_metrics if col in df.columns]

    melted = df.melt(
        id_vars=["params", "data", "model_size_numeric"],
        value_vars=available_ppl_metrics,
        var_name="metric",
        value_name="metric_value",
    )

    return melted.dropna(subset=["metric_value"])


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

    ppl_metrics_data = melt_ppl_metrics(final_data)

    with FigureManager(figure=FigureConfig(rows=1, cols=3, figsize=(18, 6))) as fm:
        for i, recipe in enumerate(selected_recipes):
            recipe_data = ppl_metrics_data[ppl_metrics_data["data"] == recipe]

            fm.plot(
                "line",
                0,
                i,
                recipe_data,
                x="model_size_numeric",
                y="metric_value",
                hue_by="metric",
                title=f"PPL Metrics Scaling: {recipe}",
            )
            fm.get_axes(0, i).set_xscale("log")

        fm.finalize_layout()
        fm.fig.savefig(
            "output/multi_ppl_metrics_scaling.png", dpi=300, bbox_inches="tight"
        )

    print("Generated: output/multi_ppl_metrics_scaling.png")


if __name__ == "__main__":
    main()

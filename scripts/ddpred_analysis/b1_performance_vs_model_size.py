import pandas as pd
from typing import Tuple, List

from datadec import DataDecide
from datadec.model_utils import param_to_numeric
from dr_plotter.figure import FigureManager
from dr_plotter.figure_config import FigureConfig
from dr_plotter.legend_manager import LegendConfig, LegendStrategy
from ft_pred.core.data_filtering import get_filtered_data
from ft_pred.core.data_preparation import extract_targets
from ft_pred.analysis import (
    convert_model_size_to_numeric,
    apply_seed_validation_1b,
)


def split_recipes_by_1b_performance(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Get 1B performance for each recipe
    data_1b = df[df["params"] == "1B"]
    recipe_performance = data_1b.groupby("data")["pile-valppl"].mean().sort_values()

    n_recipes = len(recipe_performance)
    split_point = n_recipes // 2

    better_recipes = list(recipe_performance.index[:split_point])  # Lower PPL = better
    worse_recipes = list(recipe_performance.index[split_point:])

    return better_recipes, worse_recipes


def create_recipe_group_plot(
    data: pd.DataFrame,
    recipes: List[str],
    group_name: str,
    model_sizes_ordered: List[str],
) -> None:
    group_data = data[data["data"].isin(recipes)].copy()

    with FigureManager(
        figure=FigureConfig(rows=1, cols=1, figsize=(16, 8)),
        legend=LegendConfig(
            strategy=LegendStrategy.FIGURE_BELOW,
            ncol=6,
            layout_bottom_margin=0.15,
            bbox_y_offset=0.02,
        ),
    ) as fm:
        fm.plot(
            "line",
            0,
            0,
            group_data,
            x="model_size_numeric",
            y="pile-valppl",
            hue_by="data",
            title=f"PPL vs Model Size - {group_name} Recipes",
        )

        ax = fm.get_axes(0, 0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Model Size (log scale)")
        ax.set_ylabel("Pile Validation PPL (log scale)")

        fm.fig.suptitle(
            f"Performance Scaling by Model Size: {group_name} Performing Recipes",
            fontsize=16,
            y=0.95,
        )

        fm.finalize_layout()
        output_name = (
            f"output/performance_vs_model_size_{group_name.lower()}_recipes.png"
        )
        fm.fig.savefig(output_name, dpi=300, bbox_inches="tight")

    print(f"Generated: {output_name}")


def main() -> None:
    dd = DataDecide()

    exclude_params = ["750M", "4M", "6M", "8M", "10M"]
    filtered_df, validated_params, validated_data = get_filtered_data(
        dd, "all", "all", "pile-valppl", exclude_params=exclude_params
    )

    filtered_df = apply_seed_validation_1b(filtered_df)

    target_df = extract_targets(filtered_df, "pile-valppl", "final_perf")

    final_ppl_data = target_df.rename(
        columns={"target_final_perf_pile-valppl": "pile-valppl"}
    )
    final_ppl_data = convert_model_size_to_numeric(final_ppl_data)
    final_ppl_data = final_ppl_data.dropna(subset=["pile-valppl"])

    model_sizes_ordered = sorted(
        final_ppl_data["params"].unique(), key=param_to_numeric
    )

    better_recipes, worse_recipes = split_recipes_by_1b_performance(final_ppl_data)

    create_recipe_group_plot(
        final_ppl_data, better_recipes, "Better", model_sizes_ordered
    )
    create_recipe_group_plot(
        final_ppl_data, worse_recipes, "Worse", model_sizes_ordered
    )


if __name__ == "__main__":
    main()

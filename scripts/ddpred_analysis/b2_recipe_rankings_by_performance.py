import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

from datadec import DataDecide
from datadec.model_utils import param_to_numeric
from dr_plotter.figure import FigureManager
from dr_plotter.figure_config import FigureConfig
from dr_plotter.legend_manager import LegendConfig, LegendStrategy
from dr_plotter.theme import Theme, PlotStyles
from ft_pred.core.data_filtering import get_filtered_data
from ft_pred.core.data_preparation import extract_targets
from ft_pred.analysis import (
    convert_model_size_to_numeric,
    apply_seed_validation_1b,
)


def add_recipe_rankings(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.dropna(subset=["pile-valppl"])

    recipe_performance = df.groupby("data")["pile-valppl"].mean().sort_values()
    recipe_to_rank = {
        recipe: rank + 1 for rank, recipe in enumerate(recipe_performance.index)
    }

    recipe_names = list(recipe_performance.index)

    df["recipe_rank"] = df["data"].map(recipe_to_rank)
    df["recipe_name"] = df["data"]
    return df, recipe_names


def create_model_size_colormap(df: pd.DataFrame) -> Theme:
    model_sizes = sorted(df["params"].unique(), key=param_to_numeric)
    n_sizes = len(model_sizes)

    colors = []
    for i in range(n_sizes):
        ratio = i / max(1, n_sizes - 1)
        colors.append(plt.cm.viridis(ratio))

    size_to_color = dict(zip(model_sizes, colors))

    theme = Theme(
        name="model_size_gradient", plot_styles=PlotStyles(color_mapping=size_to_color)
    )

    return theme


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
    final_ppl_data, recipe_names = add_recipe_rankings(final_ppl_data)

    model_sizes_ordered = sorted(
        final_ppl_data["params"].unique(), key=param_to_numeric
    )
    final_ppl_data["params"] = pd.Categorical(
        final_ppl_data["params"], categories=model_sizes_ordered, ordered=True
    )

    model_size_theme = create_model_size_colormap(final_ppl_data)

    with FigureManager(
        figure=FigureConfig(rows=1, cols=1, figsize=(16, 8)),
        theme=model_size_theme,
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
            final_ppl_data,
            x="recipe_rank",
            y="pile-valppl",
            hue_by="params",
            title="Recipe Rankings by Final PPL Performance",
        )

        ax = fm.get_axes(0, 0)
        ax.set_xticks(range(1, len(recipe_names) + 1))
        ax.set_xticklabels(recipe_names, rotation=45, ha="right")
        ax.set_xlabel("Recipe (sorted by performance)")

        fm.finalize_layout()
        fm.fig.savefig(
            "output/recipe_rankings_by_performance.png", dpi=300, bbox_inches="tight"
        )

    print("Generated: output/recipe_rankings_by_performance.png")


if __name__ == "__main__":
    main()

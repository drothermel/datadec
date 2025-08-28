"""
Plot Seeds Analysis Script

Requires dr_plotter integration:
    uv add "datadec[plotting]"

Generates overlay plots showing perplexity metrics across different seeds.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt

from datadec import DataDecide
from datadec.constants import PPL_TYPES
from datadec.script_utils import select_params, select_data
from datadec.plotting_utils import safe_import_plotting

# Import plotting components at module level
FigureManager, FigureConfig, (LegendConfig, LegendStrategy) = safe_import_plotting()


def load_data():
    dd = DataDecide(data_dir="../outputs/example_data", verbose=True)
    df = dd.load_dataframe("full_eval")
    return df


def process_params(param):
    """Process parameter input using script utilities."""
    if param is None:
        return select_params("all")  # All params, properly sorted
    elif isinstance(param, list):
        return select_params(param)  # Validate and sort
    else:
        return select_params([param])  # Single param as list


def process_recipes(recipe):
    """Process recipe input using script utilities."""
    if recipe is None:
        return select_data("all")  # All data recipes
    elif isinstance(recipe, list):
        return select_data(recipe)  # Validate list
    else:
        return select_data([recipe])  # Single recipe as list


def plot_seeds(df, num_cols, metrics, recipe=None, param=None):
    params = process_params(param)
    recipes = process_recipes(recipe)

    combinations = [(p, r, m) for p in params for r in recipes for m in metrics]
    num_rows = math.ceil(len(combinations) / num_cols)

    with FigureManager(
        figure=FigureConfig(
            rows=num_rows, cols=num_cols, figsize=(num_cols * 6, num_rows * 4)
        )
    ) as fm:
        for i, (param_val, recipe_val, metric) in enumerate(combinations):
            row = i // num_cols
            col = i % num_cols

            subset = df[(df["params"] == param_val) & (df["data"] == recipe_val)]

            fm.plot(
                "line",
                row,
                col,
                subset,
                x="tokens",
                y=metric,
                hue_by="seed",
                title=f"{param_val} | {recipe_val} | {metric}",
            )
    plt.show()


def plot_overlay(df, metrics, num_cols, recipe=None, param=None, save_dir=None):
    params = process_params(param)
    recipes = process_recipes(recipe)

    # Create data_param combinations
    data_param_combos = [(d, p) for d in recipes for p in params]
    num_rows = math.ceil(len(data_param_combos) / num_cols)

    # Melt the dataframe to have all metrics in one column
    filtered_df = df[(df["params"].isin(params)) & (df["data"].isin(recipes))].copy()
    melted_df = filtered_df.melt(
        id_vars=["params", "data", "seed", "tokens"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    melted_df = melted_df.dropna(subset=["value"])

    # Create x_labels and y_labels arrays
    x_labels = [
        ["Tokens"] * num_cols if row == num_rows - 1 else [None] * num_cols
        for row in range(num_rows)
    ]
    y_labels = [
        ["Perplexity" if col == 0 else None for col in range(num_cols)]
        for row in range(num_rows)
    ]

    with FigureManager(
        figure=FigureConfig(
            rows=num_rows,
            cols=num_cols,
            figsize=(num_cols * 6, num_rows * 6),
            x_labels=x_labels,
            y_labels=y_labels,
            tight_layout_pad=1.0,
        ),
        legend=LegendConfig(
            strategy=LegendStrategy.GROUPED_BY_CHANNEL,
            channel_titles={"hue": "Val Dataset (for PPL calc)", "alpha": "Seed"},
            layout_bottom_margin=0.22,
            bbox_y_offset=0.22,
            ncol=4,
            two_legend_left_x=0.1,
            two_legend_right_x=0.5,
        ),
    ) as fm:
        for i, (data_val, param_val) in enumerate(data_param_combos):
            row = i // num_cols
            col = i % num_cols

            subset = melted_df[
                (melted_df["data"] == data_val) & (melted_df["params"] == param_val)
            ]

            fm.plot(
                "line",
                row,
                col,
                subset,
                x="tokens",
                y="value",
                hue_by="metric",
                alpha_by="seed",
                title=f"{data_val} | {param_val}",
            )

        # Set ylim for all subplots based on recipe
        recipe_ylims = {
            "C4": 225,
            "DCLM-Baseline": 60,
            "Dolma1.7": 60,
            "FineWeb-Edu": 100,
        }
        ylim_max = recipe_ylims.get(recipe, 225)  # Default to 225 if recipe not found

        for ax in fm.fig.axes:
            ax.set_ylim(0, ylim_max)
            ax.set_xlim(0, 3.1e10)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f"{recipe}_ppl_metrics_per_seed.png"
        filepath = save_path / filename
        fm.fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {filepath}")
    else:
        plt.show()


def main():
    df = load_data()
    # Use script utilities for validation and selection
    recipes = select_data(["C4", "DCLM-Baseline", "Dolma1.7", "FineWeb-Edu"])
    params = select_params(["90M", "150M", "300M"])

    for recipe in recipes:
        print(f"Generating plot for recipe: {recipe}")
        plot_overlay(
            df,
            PPL_TYPES,
            num_cols=3,
            recipe=recipe,
            param=params,
            save_dir="/Users/daniellerothermel/drotherm/repos/datadec/outputs/plots",
        )


if __name__ == "__main__":
    main()

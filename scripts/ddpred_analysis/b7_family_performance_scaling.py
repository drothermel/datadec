import pandas as pd
import datadec.constants

from datadec import DataDecide
from dr_plotter.figure import FigureManager
from dr_plotter.figure_config import FigureConfig
from dr_plotter.legend_manager import LegendConfig, LegendStrategy
from ft_pred.core.data_filtering import get_filtered_data
from ft_pred.core.data_preparation import extract_targets
from ft_pred.analysis import (
    convert_model_size_to_numeric,
    apply_seed_validation_1b,
)


def add_recipe_families(df: pd.DataFrame) -> pd.DataFrame:
    families = datadec.constants.DATA_RECIPE_FAMILIES

    recipe_to_family = {}
    for family, recipes in families.items():
        for recipe in recipes:
            recipe_to_family[recipe] = family

    df = df.copy()
    df["family"] = df["data"].map(recipe_to_family)
    df = df.dropna(subset=["family"])

    return df


def prepare_family_data(filtered_df: pd.DataFrame) -> pd.DataFrame:
    target_df = extract_targets(filtered_df, "pile-valppl", "final_perf")
    final_data = target_df.rename(
        columns={"target_final_perf_pile-valppl": "pile-valppl"}
    )
    final_data = convert_model_size_to_numeric(final_data)
    final_data = add_recipe_families(final_data)
    return final_data


def create_family_plots(data: pd.DataFrame, family_name: str) -> None:
    family_data = data[data["family"] == family_name]

    if len(family_data) == 0:
        print(f"No data found for family: {family_name}")
        return

    ppl_data = family_data.dropna(subset=["pile-valppl"])
    mmlu_data = family_data.dropna(subset=["mmlu_average_correct_prob"])

    with FigureManager(
        figure=FigureConfig(
            rows=1, cols=2, figsize=(16, 8), subplot_kwargs={"sharey": False}
        ),
        legend=LegendConfig(
            strategy=LegendStrategy.FIGURE_BELOW,
            ncol=min(len(family_data["data"].unique()), 8),
            layout_bottom_margin=0.12,
            bbox_y_offset=0.02,
        ),
    ) as fm:
        if len(ppl_data) > 0:
            fm.plot(
                "line",
                0,
                0,
                ppl_data,
                x="model_size_numeric",
                y="pile-valppl",
                hue_by="data",
                title=f"PPL vs Model Size - {family_name.upper()}",
            )
            ax0 = fm.get_axes(0, 0)
            ax0.set_xscale("log")
            ax0.set_yscale("log")
            ax0.set_xlabel("Model Size (log scale)")
            ax0.set_ylabel("Pile Validation PPL (log scale)")

        if len(mmlu_data) > 0:
            fm.plot(
                "line",
                0,
                1,
                mmlu_data,
                x="model_size_numeric",
                y="mmlu_average_correct_prob",
                hue_by="data",
                title=f"MMLU vs Model Size - {family_name.upper()}",
            )
            ax1 = fm.get_axes(0, 1)
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.set_xlabel("Model Size (log scale)")
            ax1.set_ylabel("MMLU Average Correct Prob (log scale)")

        fm.fig.suptitle(
            f"Performance Scaling by Model Size: {family_name.upper()} Family",
            fontsize=16,
            y=0.95,
        )

        fm.finalize_layout()
        fm.fig.savefig(
            f"output/family_performance_{family_name}.png", dpi=300, bbox_inches="tight"
        )

    print(f"Generated: output/family_performance_{family_name}.png")


def main() -> None:
    dd = DataDecide()

    exclude_params = ["750M"]
    filtered_df, validated_params, validated_data = get_filtered_data(
        dd, "all", "all", "pile-valppl", exclude_params=exclude_params
    )

    filtered_df = apply_seed_validation_1b(filtered_df)

    end_data = prepare_family_data(filtered_df)

    families = datadec.constants.DATA_RECIPE_FAMILIES.keys()

    for family in families:
        create_family_plots(end_data, family)


if __name__ == "__main__":
    main()

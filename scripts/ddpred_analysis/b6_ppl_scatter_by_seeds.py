import pandas as pd

from datadec import DataDecide
from dr_plotter.figure import FigureManager
from dr_plotter.figure_config import FigureConfig
from dr_plotter.legend_manager import LegendConfig, LegendStrategy
from datadec.data_filtering import get_filtered_data
from datadec.analysis import extract_targets
from datadec.analysis import (
    convert_model_size_to_numeric,
    apply_seed_validation_1b,
)


def prepare_scatter_data_with_seeds(filtered_df: pd.DataFrame) -> pd.DataFrame:
    target_df = extract_targets(filtered_df, "pile-valppl", "final_perf")

    seed_data = []
    for (params, data), group in filtered_df.groupby(["params", "data"]):
        target_value = target_df[
            (target_df["params"] == params) & (target_df["data"] == data)
        ]["target_final_perf_pile-valppl"].iloc[0]

        for seed in group["seed"].unique():
            seed_data.append(
                {
                    "params": params,
                    "data": data,
                    "seed": seed,
                    "pile-valppl": target_value,
                }
            )

    result = pd.DataFrame(seed_data)
    return result.dropna(subset=["pile-valppl"])


def main() -> None:
    dd = DataDecide()

    exclude_params = ["750M"]
    filtered_df, validated_params, validated_data = get_filtered_data(
        dd, "all", "all", "pile-valppl", exclude_params=exclude_params
    )

    filtered_df = apply_seed_validation_1b(filtered_df)

    end_scatter_data = prepare_scatter_data_with_seeds(filtered_df)
    end_scatter_data = convert_model_size_to_numeric(end_scatter_data)

    with FigureManager(
        figure=FigureConfig(rows=1, cols=1, figsize=(20, 12)),
        legend=LegendConfig(
            strategy=LegendStrategy.FIGURE_BELOW,
            ncol=8,
            layout_bottom_margin=0.15,
            bbox_y_offset=0.02,
        ),
    ) as fm:
        fm.plot(
            "scatter",
            0,
            0,
            end_scatter_data,
            x="model_size_numeric",
            y="pile-valppl",
            hue_by="data",
            s=60,
            alpha=0.7,
            title="PPL vs Model Size (End Training): Individual Seeds",
        )
        ax = fm.get_axes(0, 0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Model Size (log scale)")
        ax.set_ylabel("Pile Validation PPL (log scale)")

        fm.fig.suptitle(
            "Performance Scaling by Model Size: Each point represents an individual seed",
            fontsize=18,
            y=0.95,
        )

        fm.finalize_layout()
        fm.fig.savefig("output/ppl_scatter_by_seeds.png", dpi=300, bbox_inches="tight")

    print("Generated: output/ppl_scatter_by_seeds.png")


if __name__ == "__main__":
    main()

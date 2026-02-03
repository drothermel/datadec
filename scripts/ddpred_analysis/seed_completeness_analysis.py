from typing import Dict, Any, List, Callable
import pandas as pd
from datadec import DataDecide
import datadec.constants
from datadec.analysis import (
    analyze_seed_data_density_filtered,
    analyze_mmlu_eval_focused_density,
    get_ordered_models,
    apply_max_step_filter,
)
from dr_plotter.figure import FigureManager
from dr_plotter.figure_config import FigureConfig
from dr_plotter.legend_manager import LegendConfig, LegendStrategy
from dr_plotter.theme import Theme, AxesStyles, HEATMAP_THEME


def get_categorical_models(
    data: pd.DataFrame, column: str = "model_size"
) -> pd.DataFrame:
    available_models = get_ordered_models(data)
    data_copy = data.copy()
    data_copy[column] = pd.Categorical(
        data_copy[column], categories=available_models, ordered=True
    )
    return data_copy


def process_seed_data_by_model_recipe(
    seed_data: pd.DataFrame, processor_func: Callable, recipe_filter: str = None
) -> List[Dict[str, Any]]:
    results = []
    max_steps = datadec.constants.MAX_STEP_TO_USE
    available_models = get_ordered_models(seed_data)

    for model_size in available_models:
        if model_size not in max_steps:
            continue

        max_step = max_steps[model_size]
        model_data = seed_data[seed_data["params"] == model_size]

        recipes = (
            [recipe_filter] if recipe_filter else sorted(model_data["data"].unique())
        )

        for recipe in recipes:
            recipe_data = model_data[model_data["data"] == recipe]

            for seed in sorted(recipe_data["seed"].unique()):
                seed_data_subset = apply_max_step_filter(
                    recipe_data[recipe_data["seed"] == seed], max_step
                )

                if len(seed_data_subset) > 0:
                    result = processor_func(
                        seed_data_subset, model_size, recipe, seed, max_step
                    )
                    if result:
                        results.extend(result if isinstance(result, list) else [result])

    return results


def load_seed_level_data(dd: DataDecide) -> pd.DataFrame:
    return dd.full_eval


def assess_seed_completeness(
    seed_data: pd.DataFrame, max_steps: Dict[str, int]
) -> pd.DataFrame:
    def process_completion(subset, model_size, recipe, seed, max_step):
        combo_data = seed_data[
            (seed_data["params"] == model_size) & (seed_data["data"] == recipe)
        ]
        seeds_reaching_max = combo_data[combo_data["step"] == max_step][
            "seed"
        ].nunique()
        total_seeds = combo_data["seed"].nunique()
        max_step_reached = combo_data["step"].max() if len(combo_data) > 0 else 0

        return {
            "model_size": model_size,
            "recipe": recipe,
            "seeds_reaching_max": seeds_reaching_max,
            "total_seeds": total_seeds,
            "completeness_ratio": seeds_reaching_max / total_seeds
            if total_seeds > 0
            else 0,
            "max_step_target": max_step,
            "max_step_reached": max_step_reached,
            "step_completion_ratio": max_step_reached / max_step if max_step > 0 else 0,
        }

    # Get unique combinations first
    unique_combinations = (
        seed_data.groupby(["params", "data"]).first().reset_index()[["params", "data"]]
    )

    completeness_results = []
    for _, row in unique_combinations.iterrows():
        model_size, recipe = row["params"], row["data"]
        if model_size in max_steps:
            max_step = max_steps[model_size]
            combo_data = seed_data[
                (seed_data["params"] == model_size) & (seed_data["data"] == recipe)
            ]
            seeds_reaching_max = combo_data[combo_data["step"] == max_step][
                "seed"
            ].nunique()
            total_seeds = combo_data["seed"].nunique()
            max_step_reached = combo_data["step"].max() if len(combo_data) > 0 else 0

            completeness_results.append(
                {
                    "model_size": model_size,
                    "recipe": recipe,
                    "seeds_reaching_max": seeds_reaching_max,
                    "total_seeds": total_seeds,
                    "completeness_ratio": seeds_reaching_max / total_seeds
                    if total_seeds > 0
                    else 0,
                    "max_step_target": max_step,
                    "max_step_reached": max_step_reached,
                    "step_completion_ratio": max_step_reached / max_step
                    if max_step > 0
                    else 0,
                }
            )

    return pd.DataFrame(completeness_results)


def create_completeness_heatmap(completeness_df: pd.DataFrame) -> None:
    sorted_df = get_categorical_models(completeness_df)

    with FigureManager(figure=FigureConfig(rows=1, cols=1, figsize=(15, 12))) as fm:
        fm.plot(
            "heatmap",
            0,
            0,
            sorted_df,
            x="model_size",
            y="recipe",
            values="seeds_reaching_max",
            cmap="PRGn",
            title="Seed Completeness: Seeds Reaching MAX_STEP_TO_USE",
        )
        fm.finalize_layout()
        fm.fig.savefig(
            "output/seed_completeness_heatmap.png", dpi=300, bbox_inches="tight"
        )
        print("Generated: output/seed_completeness_heatmap.png")


def create_individual_seed_progress_heatmap(seed_data: pd.DataFrame) -> None:
    def process_progress(subset, model_size, recipe, seed, max_step):
        max_step_reached = subset["step"].max()
        steps_reached = min(max_step_reached, max_step)
        steps_reached_k = int(round(steps_reached / 1000))
        completion_ratio = steps_reached / max_step

        return {
            "model_seed": f"{model_size}_s{seed}",
            "recipe": recipe,
            "steps_reached_k": steps_reached_k,
            "steps_reached": steps_reached,
            "max_possible_steps": max_step,
            "completion_ratio": completion_ratio,
            "model_size": model_size,
            "seed": seed,
        }

    progress_data = process_seed_data_by_model_recipe(seed_data, process_progress)
    progress_df = pd.DataFrame(progress_data)

    if len(progress_df) == 0:
        print("No seed progress data found")
        return

    # Create ordered model_seed categories
    available_models = get_ordered_models(seed_data)
    max_steps = datadec.constants.MAX_STEP_TO_USE
    model_seed_order = [
        combo
        for model_size in available_models
        if model_size in max_steps
        for combo in sorted(
            [
                c
                for c in progress_df["model_seed"].unique()
                if c.startswith(f"{model_size}_")
            ]
        )
    ]

    progress_df["model_seed"] = pd.Categorical(
        progress_df["model_seed"], categories=model_seed_order, ordered=True
    )
    progress_df["steps_reached_k"] = progress_df["steps_reached_k"].astype(int)

    custom_theme = Theme(
        name="integer_heatmap",
        parent=HEATMAP_THEME,
        axes_styles=AxesStyles(xlabel_pos="bottom", cell_text_format="int"),
    )

    with FigureManager(figure=FigureConfig(rows=1, cols=1, figsize=(30, 16))) as fm:
        fm.plot(
            "heatmap",
            0,
            0,
            progress_df,
            x="model_seed",
            y="recipe",
            values="steps_reached_k",
            cmap="viridis",
            theme=custom_theme,
            title="Individual Seed Training Progress: Shows how far each seed progressed in training\n(Thousand Steps Reached, Capped at MAX_STEP_TO_USE per model size)",
        )
        fm.finalize_layout()
        fm.fig.savefig(
            "output/individual_seed_progress_heatmap.png", dpi=300, bbox_inches="tight"
        )
        print("Generated: output/individual_seed_progress_heatmap.png")


def create_individual_seed_nonnan_heatmap(
    seed_data: pd.DataFrame, target_metric: str, output_filename: str, title: str
) -> None:
    def process_nonnan(subset, model_size, recipe, seed, max_step):
        nonnan_count = subset[target_metric].notna().sum()
        return {
            "model_seed": f"{model_size}_s{seed}",
            "recipe": recipe,
            "nonnan_count": int(nonnan_count),
            "model_size": model_size,
            "seed": seed,
        }

    nonnan_data = process_seed_data_by_model_recipe(seed_data, process_nonnan)
    nonnan_df = pd.DataFrame(nonnan_data)

    if len(nonnan_df) == 0:
        print(f"No seed non-NaN data found for {target_metric}")
        return

    # Create ordered model_seed categories
    available_models = get_ordered_models(seed_data)
    max_steps = datadec.constants.MAX_STEP_TO_USE
    model_seed_order = [
        combo
        for model_size in available_models
        if model_size in max_steps
        for combo in sorted(
            [
                c
                for c in nonnan_df["model_seed"].unique()
                if c.startswith(f"{model_size}_")
            ]
        )
    ]

    nonnan_df["model_seed"] = pd.Categorical(
        nonnan_df["model_seed"], categories=model_seed_order, ordered=True
    )

    custom_theme = Theme(
        name="integer_heatmap",
        parent=HEATMAP_THEME,
        axes_styles=AxesStyles(xlabel_pos="bottom", cell_text_format="int"),
    )

    with FigureManager(figure=FigureConfig(rows=1, cols=1, figsize=(30, 16))) as fm:
        fm.plot(
            "heatmap",
            0,
            0,
            nonnan_df,
            x="model_seed",
            y="recipe",
            values="nonnan_count",
            cmap="viridis",
            theme=custom_theme,
            title=title,
        )
        fm.finalize_layout()
        fm.fig.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"Generated: {output_filename}")


def create_individual_seed_nonnan_heatmap_ppl(seed_data: pd.DataFrame) -> None:
    create_individual_seed_nonnan_heatmap(
        seed_data,
        "pile-valppl",
        "output/individual_seed_nonnan_heatmap_ppl.png",
        "Individual Seed Non-NaN Data Points: pile-valppl metric\n(Count of non-NaN pile-valppl values per seed, capped at MAX_STEP_TO_USE)",
    )


def create_individual_seed_nonnan_heatmap_mmlu(seed_data: pd.DataFrame) -> None:
    create_individual_seed_nonnan_heatmap(
        seed_data,
        "mmlu_average_correct_prob",
        "output/individual_seed_nonnan_heatmap_mmlu.png",
        "Individual Seed Non-NaN Data Points: MMLU average correct prob metric\n(Count of non-NaN MMLU evaluations per seed, capped at MAX_STEP_TO_USE)",
    )


def create_focused_visualization(
    density_df: pd.DataFrame, filter_type: str, title: str, output_filename: str
) -> None:
    sorted_df = get_categorical_models(density_df)
    filtered_data = (
        sorted_df[sorted_df["filter_type"] == filter_type]
        if "filter_type" in sorted_df.columns
        else sorted_df
    )

    with FigureManager(
        figure=FigureConfig(rows=1, cols=1, figsize=(16, 6)),
        legend=LegendConfig(strategy=LegendStrategy.FIGURE_BELOW),
    ) as fm:
        fm.fig.suptitle(title, fontsize=16)

        if len(filtered_data) > 0:
            recipe_name = filtered_data.iloc[0]["recipe"]
            fm.plot(
                "bar",
                0,
                0,
                filtered_data,
                x="model_size",
                y="non_nan_points",
                hue_by="seed",
                title=f"{filter_type.replace('_', '-').title()}: First bar - count including NaNs, other bars - count per seed without NaNs (Recipe: {recipe_name})",
            )

        fm.finalize_layout()
        fm.fig.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"Generated: {output_filename}")


def create_ppl_focused_visualization(density_df: pd.DataFrame) -> None:
    create_focused_visualization(
        density_df,
        "ppl_focused",
        "pile-valppl Seed Density: PPL-Focused Analysis",
        "output/ppl_focused_seed_density.png",
    )


def create_mmlu_eval_focused_visualization(density_df: pd.DataFrame) -> None:
    create_focused_visualization(
        density_df,
        "eval_focused",
        "MMLU Average Correct Prob Seed Density: Eval-Focused Analysis",
        "output/mmlu_eval_focused_seed_density.png",
    )


def identify_missing_data_patterns(completeness_df: pd.DataFrame) -> pd.DataFrame:
    incomplete_combinations = completeness_df[
        completeness_df["seeds_reaching_max"] < 3
    ].copy()

    if len(incomplete_combinations) == 0:
        return pd.DataFrame()

    incomplete_combinations["missing_seeds"] = (
        3 - incomplete_combinations["seeds_reaching_max"]
    )
    incomplete_combinations["gap_severity"] = incomplete_combinations[
        "seeds_reaching_max"
    ].map({0: "Critical", 1: "Severe", 2: "Moderate"})

    return incomplete_combinations[
        [
            "model_size",
            "recipe",
            "seeds_reaching_max",
            "total_seeds",
            "missing_seeds",
            "gap_severity",
            "max_step_target",
            "max_step_reached",
            "step_completion_ratio",
        ]
    ]


def main() -> None:
    dd = DataDecide()
    seed_data = load_seed_level_data(dd)
    print(f"Loaded seed-level data: {seed_data.shape}")

    max_steps = datadec.constants.MAX_STEP_TO_USE

    # Analysis
    completeness_df = assess_seed_completeness(seed_data, max_steps)
    print(f"Assessed completeness for {len(completeness_df)} model×recipe combinations")

    ppl_density_df = analyze_seed_data_density_filtered(seed_data)
    print(
        f"Analyzed PPL-focused seed data density for {len(ppl_density_df)} combinations"
    )

    mmlu_density_df = analyze_mmlu_eval_focused_density(seed_data)
    print(
        f"Analyzed MMLU eval-focused seed data density for {len(mmlu_density_df)} combinations"
    )

    # Visualizations
    create_completeness_heatmap(completeness_df)
    create_individual_seed_progress_heatmap(seed_data)
    create_individual_seed_nonnan_heatmap_ppl(seed_data)
    create_individual_seed_nonnan_heatmap_mmlu(seed_data)
    create_ppl_focused_visualization(ppl_density_df)
    create_mmlu_eval_focused_visualization(mmlu_density_df)

    # Analysis and reporting
    missing_data_df = identify_missing_data_patterns(completeness_df)
    print(f"Identified {len(missing_data_df)} incomplete combinations")

    completion_rate = completeness_df["completeness_ratio"].mean()
    print(f"Overall completion rate: {completion_rate:.1%}")

    print("\n=== Phase A.1 Seed Completeness Analysis Complete ===")
    print("Generated deliverables:")
    print("  - 5 dr_plotter visualizations")


if __name__ == "__main__":
    main()

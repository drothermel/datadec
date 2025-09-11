from __future__ import annotations

import click
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from datadec.wandb_eval.wandb_constants import PRETRAIN_POSTTRAIN_DF_PATH
from datadec.wandb_eval.parsing import create_and_save_pretrain_posttrain_df


def load_pretrain_posttrain_df() -> pd.DataFrame:
    path = Path(PRETRAIN_POSTTRAIN_DF_PATH)

    if not path.exists():
        print(f"DataFrame not found at {path}. Creating it now...")
        df = create_and_save_pretrain_posttrain_df()
        return df
    else:
        print(f"Loading existing DataFrame from {path}...")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        else:
            return pd.read_pickle(path)


def filter_runs_by_tag_and_variation(
    df: pd.DataFrame, tag_filter: str, vary_by: str
) -> pd.DataFrame:
    # Filter runs that contain the specified tag
    if "wandb_tags" in df.columns:
        tag_mask = df["wandb_tags"].fillna("").str.contains(tag_filter, na=False)
        filtered_df = df[tag_mask].copy()
    else:
        print("Warning: wandb_tags column not found, using all data")
        filtered_df = df.copy()

    print(f"Found {len(filtered_df)} total rows with tag '{tag_filter}'")

    # Show what values we're varying by
    if vary_by in filtered_df.columns:
        unique_values = filtered_df[vary_by].dropna().unique()
        print(f"Varying by {vary_by}: {sorted(unique_values)}")
        return filtered_df[filtered_df[vary_by].notna()]
    else:
        print(f"Warning: {vary_by} column not found")
        return filtered_df


def plot_scaling_curve(
    df: pd.DataFrame, metric: str, vary_by: str, save_path: str = None
):
    # Check if metric exists
    if metric not in df.columns:
        available_metrics = [
            col
            for col in df.columns
            if any(task in col for task in ["csqa", "piqa", "boolq", "arc"])
        ]
        print(
            f"Metric {metric} not found. Available metrics: {available_metrics[:10]}..."
        )
        return

    plt.figure(figsize=(12, 8))

    # Get unique values for the variation dimension
    vary_values = sorted(df[vary_by].dropna().unique())
    colors = plt.cm.Set1(range(len(vary_values)))

    for i, vary_val in enumerate(vary_values):
        subset = df[df[vary_by] == vary_val]

        # Only plot points where the metric has values (non-NaN)
        metric_data = subset[subset[metric].notna()]

        if len(metric_data) > 0:
            plt.scatter(
                metric_data["cumulative_tokens"],
                metric_data[metric],
                color=colors[i],
                label=f"{vary_by}={vary_val}",
                alpha=0.7,
                s=50,
            )

            print(f"{vary_by}={vary_val}: {len(metric_data)} data points")

    plt.xlabel("Cumulative Tokens")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(
        f"Scaling Curve: {metric} vs Cumulative Tokens\n(Pretraining + Finetuning)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")

    # Add annotation about the data
    total_points = len(df[df[metric].notna()])
    plt.figtext(0.02, 0.02, f"Total evaluation points: {total_points}", fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


@click.command()
@click.option(
    "--tag-filter",
    default="100mt",
    help='Filter runs by this tag (e.g., "100mt", "dolma")',
)
@click.option(
    "--vary-by", default="params", help='Dimension to vary by (e.g., "params", "data")'
)
@click.option(
    "--metric",
    default="csqa_acc_raw",
    help='Metric to plot (e.g., "csqa_acc_raw", "piqa_primary_score")',
)
@click.option("--save-path", help="Path to save the plot (optional)")
@click.option(
    "--recreate", is_flag=True, help="Force recreate the DataFrame even if it exists"
)
def main(tag_filter: str, vary_by: str, metric: str, save_path: str, recreate: bool):
    print("=== Pretraining + Finetuning Scaling Curve Analysis ===")

    # Load or create the DataFrame
    if recreate:
        print("Recreating DataFrame...")
        df = create_and_save_pretrain_posttrain_df()
    else:
        df = load_pretrain_posttrain_df()

    print(f"DataFrame shape: {df.shape}")

    # Filter and prepare data
    filtered_df = filter_runs_by_tag_and_variation(df, tag_filter, vary_by)

    if len(filtered_df) == 0:
        print(f"No data found with tag '{tag_filter}' and column '{vary_by}'")
        return

    # Create the plot
    plot_scaling_curve(filtered_df, metric, vary_by, save_path)

    print("✅ Scaling curve plot completed!")
    print("This plot shows:")
    print("  - Continuous scaling curves from pretraining through finetuning")
    print(
        "  - Each point represents an evaluation at the correct cumulative token count"
    )
    print("  - Gaps between pretraining end and finetuning evaluation are realistic")


if __name__ == "__main__":
    main()

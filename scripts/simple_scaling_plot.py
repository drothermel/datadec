from __future__ import annotations

import pandas as pd
from dr_plotter import FigureManager

from datadec.data import DataDecide
from datadec.wandb_eval.wandb_constants import PRETRAIN_POSTTRAIN_DF_PATH


def main():
    print("=== Simple Scaling Plot ===")

    # Load both datasets
    print("Loading pretraining data...")
    dd = DataDecide()
    pretraining_df = dd.full_eval

    print("Loading WandB finetuning data...")
    wandb_df = pd.read_pickle(PRETRAIN_POSTTRAIN_DF_PATH)

    # Filter pretraining data to only include combinations that exist in WandB
    wandb_combinations = wandb_df[["params", "data"]].dropna().drop_duplicates()
    filtered_pretraining = pretraining_df.merge(
        wandb_combinations, on=["params", "data"], how="inner"
    )

    print(f"Filtered pretraining data: {len(filtered_pretraining)} rows")
    print(f"Unique (params, data) combinations: {len(wandb_combinations)}")

    # Select a few specific model sizes to keep it simple
    model_sizes = ["4M", "60M", "150M"]
    metric = "csqa_acc_raw"

    with FigureManager() as fm:
        # Plot pretraining progression lines
        for model_size in model_sizes:
            pretrain_subset = filtered_pretraining[
                (filtered_pretraining["params"] == model_size)
                & (filtered_pretraining["data"] == "Dolma1.7")
                & (filtered_pretraining[metric].notna())
            ].copy()

            if len(pretrain_subset) > 0:
                print(
                    f"Plotting {len(pretrain_subset)} pretraining points for {model_size}"
                )
                fm.plot(
                    pretrain_subset,
                    "line",
                    x="tokens",
                    y=metric,
                    label=f"{model_size} pretraining",
                    linewidth=2,
                )

        # Plot finetuning final points as scatter
        for model_size in model_sizes:
            finetune_subset = wandb_df[
                (wandb_df["params"] == model_size)
                & (wandb_df["data"] == "Dolma1.7")
                & (wandb_df[metric].notna())
            ].copy()

            if len(finetune_subset) > 0:
                print(
                    f"Plotting {len(finetune_subset)} finetuning points for {model_size}"
                )
                fm.plot(
                    finetune_subset,
                    "scatter",
                    x="cumulative_tokens",
                    y=metric,
                    label=f"{model_size} finetuning",
                    s=100,
                    alpha=0.8,
                )

    fm.finalize(
        xlabel="Tokens",
        ylabel="CSQA Accuracy",
        title="Scaling Curves: Pretraining → Finetuning",
    )

    print("✓ Plot completed!")


if __name__ == "__main__":
    main()

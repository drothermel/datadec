import matplotlib.pyplot as plt
import pandas as pd

from datadec.wandb_eval.wandb_constants import PRETRAIN_POSTTRAIN_DF_PATH


def main():
    print("=== Super Simple Scaling Plot ===")

    # Load the integrated DataFrame (already has corrected pretraining data)
    print("Loading integrated WandB + pretraining data...")
    wandb_df = pd.read_pickle(PRETRAIN_POSTTRAIN_DF_PATH)

    print(f"Loaded integrated data: {wandb_df.shape}")

    # Select specific hyperparameters to get distinct runs
    model_size = "4M"
    metric = "csqa_acc_raw"
    dataset = "Dolma1.7"

    plt.figure(figsize=(12, 8))

    # Get a few specific learning rates to compare
    lr_subset = wandb_df[
        (wandb_df["params"] == model_size)
        & (wandb_df["data"] == dataset)
        & (wandb_df[metric].notna())
        & (
            wandb_df["num_train_epochs"] == 1
        )  # Fix epochs to isolate learning rate effect
    ]

    unique_lrs = sorted(lr_subset["learning_rate"].unique())[
        :3
    ]  # Take first 3 learning rates
    print(f"Plotting learning rates: {unique_lrs}")

    # Plot each learning rate as a separate complete series (pretraining + finetuning)
    for i, lr in enumerate(unique_lrs):
        color = f"C{i}"

        # Get all data for this specific learning rate (both pretraining and finetuning)
        run_data = wandb_df[
            (wandb_df["params"] == model_size)
            & (wandb_df["data"] == dataset)
            & (wandb_df["learning_rate"] == lr)
            & (wandb_df["num_train_epochs"] == 1)
        ].copy()

        if len(run_data) == 0:
            continue

        # Split into pretraining and finetuning phases
        pretrain_data = run_data[run_data["pretraining_phase"] == True].copy()
        finetune_data = run_data[
            (run_data["pretraining_phase"] == False) & (run_data[metric].notna())
        ].copy()

        # Create label with distinguishing hyperparameters
        if len(finetune_data) > 0:
            point = finetune_data.iloc[0]
            # Include the key hyperparameters that distinguish runs
            label_parts = []
            if "learning_rate" in point:
                label_parts.append(f"lr={point['learning_rate']:.2e}")
            if "seed" in point and pd.notna(point["seed"]):
                label_parts.append(f"seed={int(point['seed'])}")
            if "num_train_epochs" in point and pd.notna(point["num_train_epochs"]):
                label_parts.append(f"epochs={int(point['num_train_epochs'])}")

            run_label = ", ".join(label_parts)
        else:
            run_label = f"lr={lr:.2e}"

        # Track if we've added a label yet
        label_added = False

        # Plot pretraining progression line
        if len(pretrain_data) > 0:
            pretrain_data = pretrain_data.sort_values("cumulative_tokens")
            pretrain_metric_data = pretrain_data[pretrain_data[metric].notna()]

            if len(pretrain_metric_data) > 0:
                print(f"{run_label}: {len(pretrain_metric_data)} pretraining points")
                plt.plot(
                    pretrain_metric_data["cumulative_tokens"],
                    pretrain_metric_data[metric],
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                    label=run_label,
                )
                label_added = True

        # Plot finetuning final point
        if len(finetune_data) > 0:
            point = finetune_data.iloc[0]
            print(
                f"{run_label}: final point {point[metric]:.3f} at {point['cumulative_tokens']:,.0f} tokens"
            )
            plt.scatter(
                point["cumulative_tokens"],
                point[metric],
                color=color,
                s=100,
                alpha=0.8,
                marker="o",
                edgecolor="white",
                linewidth=1,
                label=run_label if not label_added else None,
            )

    plt.xlabel("Tokens")
    plt.ylabel("CSQA Accuracy")
    plt.title(f"Scaling Curves: Pretraining → Finetuning ({dataset})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")

    print("Showing plot...")
    plt.show()

    print("✓ Plot completed!")


if __name__ == "__main__":
    main()

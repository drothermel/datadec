import sys

from datadec.wandb_eval import analysis_helpers
from datadec.wandb_eval import wandb_constants as wconsts
from datadec.wandb_eval.wandb_loader import FilterConfig, WandBDataLoader


def main() -> None:
    loader = WandBDataLoader()
    config = FilterConfig(
        method="finetune",
        completed_only=False,
        recent_only=False,
        column_groups=wconsts.INITIAL_SFT_GROUPS,
    )
    filtered_df, history_df = loader.load_data(config)
    print()
    print(f"Total runs after filtering: {len(filtered_df)}")
    print(f"Training methods: {filtered_df['method'].value_counts().to_dict()}")
    print(f"Run states: {filtered_df['state'].value_counts().to_dict()}")
    print(f"History records: {len(history_df)}")
    print()
    print("First rows of filtered data:")
    print(filtered_df.head(20))
    print()
    print("First rows of history data:")
    print(history_df.head(20))


if __name__ == "__main__":
    width = 300 if len(sys.argv) == 1 else int(sys.argv[1])
    analysis_helpers.configure_pandas_display(width)
    main()

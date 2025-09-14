# %%
%load_ext autoreload
%autoreload 2
import sys
from pathlib import Path

import pandas as pd

# Configure pandas to show all columns in interactive output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 500)  # Limit column content width for readability

sys.path.append(str(Path(__file__).parent.parent / "src"))

from datadec.wandb_eval import wandb_constants as wconsts
from datadec.wandb_eval.wandb_loader import FilterConfig, WandBDataLoader
# %%


def main():
    loader = WandBDataLoader()

    config = FilterConfig(
        method="finetune",
        completed_only=False,
        recent_only=False,
        column_groups=wconsts.INITIAL_SFT_GROUPS,
    )

    filtered_df = loader.load_data(config)
    print(f"\nTotal runs after filtering: {len(filtered_df)}")
    print(f"Training methods: {filtered_df['method'].value_counts().to_dict()}")
    if "state" in filtered_df.columns:
        print(f"Run states: {filtered_df['state'].value_counts().to_dict()}")
    return filtered_df


# %%
filtered_df = main()

# %%
filtered_df.head()

# %%
filtered_df["num_train_epochs"].unique()

# %%
if __name__ == "__main__":
    filtered_df = main()


# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%

# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%

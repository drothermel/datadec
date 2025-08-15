# %%
# type: ignore
%load_ext autoreload
%autoreload 2

from pprint import pprint

from datadec import DataDecide
from datadec.paths import DataDecidePaths

# %%
data_dir = "../tmp_data_v3"
dd = DataDecide(
    data_dir=data_dir,
    # recompute_from=None,
    #recompute_from="merge",
    recompute_from="all",
    verbose=True,
)
# %%
pprint(dd.paths.available_dataframes)
full_eval = dd.loader.load_name("full_eval")
full_eval.head(20)

# %%
pprint(list(dd.full_eval['mmlu_moral_disputes']))
# %%
dd.full_eval[
    (dd.full_eval["params"] == "10M")
    & (dd.full_eval["data"] == "C4")
    & (dd.full_eval["seed"] == 0)
]

# %%

# %%
dd.full_eval[dd.full_eval["model_size"] == "1B"].head(20)

# %%

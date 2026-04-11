from __future__ import annotations

from dr_hf import HFLocation

PPL_EVAL_DATASET = HFLocation(
    org="allenai",
    repo_name="DataDecide-ppl-results",
)

DWN_EVAL_DATASET = HFLocation(
    org="allenai",
    repo_name="DataDecide-eval-results",
)

DWN_INSTANCE_DATASET = HFLocation(
    org="allenai",
    repo_name="DataDecide-eval-instances",
)

HF_DATASET_LOCATIONS: dict[str, HFLocation] = {
    "ppl_eval_ds": PPL_EVAL_DATASET,
    "dwn_eval_ds": DWN_EVAL_DATASET,
    "dwn_instance_ds": DWN_INSTANCE_DATASET,
}

HF_DATASET_SPLIT = "train"

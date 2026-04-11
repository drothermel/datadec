from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from datadec import model_utils
from datadec.ingest.enums import ModelSizeName


class ModelDetails(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    params: ModelSizeName

    default_seed: int
    length_str: str
    lr_warmup_start: float

    d_model: int
    n_heads: int
    n_layers: int
    mlp_ratio: int

    weight_tying: bool
    alibi: bool
    rope: bool
    flash_attention: bool
    attention_dropout: float
    attention_layer_norm: bool
    include_bias: bool
    layer_norm_type: str
    layer_norm_with_affine: bool
    layer_norm_eps: float
    bias_for_layer_norm: bool
    attention_layer_norm_with_affine: bool
    activation_type: str
    residual_dropout: float
    embedding_dropout: float

    max_sequence_length: int
    vocab_size: int
    embedding_size: int
    eos_token_id: int
    pad_token_id: int

    init_device: str
    init_fn: str
    init_std: float
    init_cutoff_factor: int

    params_numeric: float
    true_model_size: int
    batch_size: int
    total_tokens: int
    warmup_tokens: int
    lr_max: float
    lr_final: float
    total_steps: int
    total_seqs: int
    warmup_perc: float
    warmup_steps: int
    lr_decay_tokens: int
    lr_decay_steps: int

    tokens_per_step: int
    compute_per_step: float


class ModelRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    details_by_size: dict[ModelSizeName, ModelDetails]

    def __getitem__(self, key: ModelSizeName) -> ModelDetails:
        return self.details_by_size[key]

    def __contains__(self, key: object) -> bool:
        return key in self.details_by_size

    def __iter__(self):
        return iter(self.details_by_size.values())

    def __len__(self) -> int:
        return len(self.details_by_size)


def _empirical_step_ratios_from_dwn(
    dwn_df: pd.DataFrame,
) -> dict[str, tuple[int, float]]:
    grouped = dwn_df.groupby("params").agg(
        max_step=("step", "max"),
        max_tokens=("tokens", "max"),
        max_compute=("compute", "max"),
    )
    ratios: dict[str, tuple[int, float]] = {}
    for params_str, row in grouped.iterrows():
        max_step = int(row["max_step"])
        tokens_per_step = int(row["max_tokens"]) // max_step
        compute_per_step = float(row["max_compute"]) / max_step
        ratios[str(params_str)] = (tokens_per_step, compute_per_step)
    return ratios


def load_model_registry(dwn_df: pd.DataFrame) -> ModelRegistry:
    ratios = _empirical_step_ratios_from_dwn(dwn_df)
    configs = model_utils.create_all_model_configs()
    details_by_size: dict[ModelSizeName, ModelDetails] = {}
    for size_str, config in configs.items():
        if size_str not in ratios:
            raise ValueError(
                f"no empirical step ratios for params={size_str!r} in dwn data"
            )
        tokens_per_step, compute_per_step = ratios[size_str]
        details_by_size[ModelSizeName(size_str)] = ModelDetails.model_validate(
            {
                "params": size_str,
                "tokens_per_step": tokens_per_step,
                "compute_per_step": compute_per_step,
                **config,
            }
        )
    return ModelRegistry(details_by_size=details_by_size)

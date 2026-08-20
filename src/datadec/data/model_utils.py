from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from datadec.data import constants as consts


@dataclass(frozen=True, slots=True)
class ModelSchedule:
    params: str
    nominal_parameter_count: int
    training_parameter_count: int
    exact_parameter_count: int
    tokens_per_step: int
    flops_per_token_per_parameter: int

    def tokens_at_step(self, step: int) -> int:
        return step * self.tokens_per_step

    def compute_at_step(self, step: int) -> float:
        return float(
            self.tokens_at_step(step)
            * self.exact_parameter_count
            * self.flops_per_token_per_parameter
        )


MODEL_DETAIL_COLUMNS: tuple[str, ...] = (
    "default_seed",
    "length_str",
    "lr_warmup_start",
    "d_model",
    "n_heads",
    "n_layers",
    "mlp_ratio",
    "weight_tying",
    "alibi",
    "rope",
    "flash_attention",
    "attention_dropout",
    "attention_layer_norm",
    "include_bias",
    "layer_norm_type",
    "layer_norm_with_affine",
    "layer_norm_eps",
    "bias_for_layer_norm",
    "attention_layer_norm_with_affine",
    "activation_type",
    "residual_dropout",
    "embedding_dropout",
    "max_sequence_length",
    "vocab_size",
    "embedding_size",
    "eos_token_id",
    "pad_token_id",
    "init_device",
    "init_fn",
    "init_std",
    "init_cutoff_factor",
    "nominal_parameter_count",
    "training_parameter_count",
    "exact_parameter_count",
    "batch_size",
    "total_tokens",
    "warmup_tokens",
    "lr_max",
    "lr_final",
    "total_steps",
    "total_seqs",
    "warmup_perc",
    "warmup_steps",
    "lr_decay_tokens",
    "lr_decay_steps",
    "tokens_per_step",
    "compute_per_step",
)


def round_value_by_multiple(value: float, multiple: int) -> int:
    return int(round(value / multiple) * multiple)


def model_size_str_to_training_parameter_count(size_str: str) -> int:
    return consts.TRAINING_PARAMETER_COUNTS[size_str]


def param_to_numeric(param_str: str) -> float:
    if param_str.endswith("M"):
        return float(param_str[:-1]) * 1e6
    elif param_str.endswith("B"):
        return float(param_str[:-1]) * 1e9
    else:
        try:
            return float(param_str)
        except ValueError:
            raise ValueError(f"Cannot parse parameter string: {param_str}")


def calc_batch_size(model_size_str: str) -> int:
    assert consts.MAX_SEQ_LEN == 2_048
    model_size = model_size_str_to_training_parameter_count(model_size_str)
    batch_size = (
        consts.BS_COEFFICIENT
        * (model_size / consts.MODEL_SIZE_NORM_VALUE) ** consts.BS_EXPONENT
    )
    rounding_size = consts.GPUS_PER_NODE * consts.MICROBATCH_SIZE
    return round_value_by_multiple(batch_size, rounding_size)


def calc_total_tokens_from_str(length_str: str, model_size_str: str) -> int:
    model_size = model_size_str_to_training_parameter_count(model_size_str)
    length_in_tokens, length_unit = consts.NUMBER_UNIT_RE.match(
        length_str.strip().upper()
    ).groups()  # type: ignore
    assert length_unit == "XC"
    return int(length_in_tokens) * consts.TOKEN_LEN_XC_MULTIPLIER * model_size


def calc_warmup_tokens(model_size_str: str) -> int:
    model_size = model_size_str_to_training_parameter_count(model_size_str)
    batch_size = calc_batch_size(model_size_str)
    return round(model_size / (batch_size / consts.MAX_SEQ_LEN))


def calc_lr_max(model_size_str: str) -> float:
    model_size = model_size_str_to_training_parameter_count(model_size_str)
    return (
        consts.LR_MAX_BASE
        * (model_size / consts.MODEL_SIZE_NORM_VALUE) ** consts.LR_EXPONENT
    )


def calc_tokens_per_step(batch_size: int) -> int:
    return batch_size * consts.MAX_SEQ_LEN


def calc_compute(tokens: int, model_size_str: str) -> float:
    return float(
        tokens
        * consts.EXACT_PARAMETER_COUNTS[model_size_str]
        * consts.FLOPS_PER_TOKEN_PER_PARAMETER
    )


def create_model_schedules() -> tuple[ModelSchedule, ...]:
    return tuple(
        ModelSchedule(
            params=model_size,
            nominal_parameter_count=consts.NOMINAL_PARAMETER_COUNTS[model_size],
            training_parameter_count=consts.TRAINING_PARAMETER_COUNTS[model_size],
            exact_parameter_count=consts.EXACT_PARAMETER_COUNTS[model_size],
            tokens_per_step=calc_tokens_per_step(calc_batch_size(model_size)),
            flops_per_token_per_parameter=consts.FLOPS_PER_TOKEN_PER_PARAMETER,
        )
        for model_size in consts.ALL_MODEL_SIZE_STRS
    )


@cache
def create_persisted_model_details(model_size_str: str) -> dict[str, object]:
    config = create_model_config(model_size_str)
    config.pop(consts.PARAM_NUMERIC_COL)
    schedule = next(
        schedule
        for schedule in create_model_schedules()
        if schedule.params == model_size_str
    )
    config["tokens_per_step"] = schedule.tokens_per_step
    config["compute_per_step"] = schedule.compute_at_step(1)
    actual = tuple(config)
    if actual != MODEL_DETAIL_COLUMNS:
        raise AssertionError(
            "persisted model detail columns drift from model config: "
            f"expected={MODEL_DETAIL_COLUMNS!r}, actual={actual!r}"
        )
    return config


def checkpoint_enrichment(model_size_str: str, step: int) -> dict[str, object]:
    if step < 0:
        raise ValueError(f"checkpoint step must be non-negative: {step}")
    details = create_persisted_model_details(model_size_str)
    schedule = next(
        schedule
        for schedule in create_model_schedules()
        if schedule.params == model_size_str
    )
    lr_warmup_start = float(details["lr_warmup_start"])
    lr_max = float(details["lr_max"])
    lr_final = float(details["lr_final"])
    warmup_steps = int(details["warmup_steps"])
    lr_decay_steps = int(details["lr_decay_steps"])
    return {
        "tokens": schedule.tokens_at_step(step),
        "compute": schedule.compute_at_step(step),
        **details,
        "lr_at_step": get_lr_at_step(
            step=step,
            lr_warmup_start=lr_warmup_start,
            lr_max=lr_max,
            lr_final=lr_final,
            warmup_steps=warmup_steps,
            lr_decay_steps=lr_decay_steps,
        ),
        "cumulative_lr": calculate_cumulative_lr(
            step=step,
            lr_warmup_start=lr_warmup_start,
            lr_max=lr_max,
            lr_final=lr_final,
            warmup_steps=warmup_steps,
            lr_decay_steps=lr_decay_steps,
        ),
    }


def calc_total_steps_from_tokens(total_tokens: int, batch_size: int) -> int:
    return int(math.ceil(total_tokens / calc_tokens_per_step(batch_size)))


def calc_total_seqs_from_tokens(total_tokens: int) -> int:
    return int(round(total_tokens / consts.MAX_SEQ_LEN))


def create_model_config(model_size_str: str, **kwargs: Any) -> dict[str, Any]:
    assert model_size_str in consts.ALL_MODEL_SIZE_STRS, (
        f"Unknown model size '{model_size_str}'. Available: {consts.ALL_MODEL_SIZE_STRS}"
    )
    config = consts.MODEL_CONFIG_BASE.copy()
    config.update(consts.MODEL_SHAPES[model_size_str])

    config[consts.PARAM_NUMERIC_COL] = param_to_numeric(model_size_str)
    config["nominal_parameter_count"] = consts.NOMINAL_PARAMETER_COUNTS[
        model_size_str
    ]
    config["training_parameter_count"] = consts.TRAINING_PARAMETER_COUNTS[
        model_size_str
    ]
    config["exact_parameter_count"] = consts.EXACT_PARAMETER_COUNTS[model_size_str]
    config["batch_size"] = calc_batch_size(model_size_str)
    length_str = config["length_str"]
    assert isinstance(length_str, str)
    config["total_tokens"] = calc_total_tokens_from_str(
        length_str, model_size_str
    )
    config["warmup_tokens"] = calc_warmup_tokens(model_size_str)
    config["lr_max"] = calc_lr_max(model_size_str)
    config["lr_final"] = consts.LR_FINAL_RATIO * config["lr_max"]
    config["total_steps"] = calc_total_steps_from_tokens(
        config["total_tokens"], config["batch_size"]
    )
    config["total_seqs"] = calc_total_seqs_from_tokens(config["total_tokens"])
    config["warmup_perc"] = config["warmup_tokens"] / config["total_tokens"]
    config["warmup_steps"] = calc_total_steps_from_tokens(
        config["warmup_tokens"], config["batch_size"]
    )
    config["lr_decay_tokens"] = config["total_tokens"] - config["warmup_tokens"]
    config["lr_decay_steps"] = config["total_steps"] - config["warmup_steps"]

    config.update(kwargs)
    return config


def create_all_model_configs() -> dict[str, dict[str, Any]]:
    return {
        model_size: create_model_config(model_size)
        for model_size in consts.ALL_MODEL_SIZE_STRS
    }


def get_model_details_df() -> pd.DataFrame:
    configs = create_all_model_configs()
    return (
        pd.DataFrame.from_dict(configs, orient="index")
        .reset_index()
        .rename(columns={"index": "params"})
    )


def numerical_cosine_integral(
    lr_max: float, lr_final: float, lr_decay_steps: int, decay_step: int
) -> float:
    if decay_step <= 0:
        return 0.0

    t_values = np.linspace(0, decay_step, int(decay_step) + 1)
    lr_values = lr_final + 0.5 * (lr_max - lr_final) * (
        1 + np.cos(np.pi * t_values / lr_decay_steps)
    )

    return float(np.trapezoid(lr_values, t_values))


def calculate_cumulative_lr(
    step: int,
    lr_warmup_start: float,
    lr_max: float,
    lr_final: float,
    warmup_steps: int,
    lr_decay_steps: int,
) -> float:
    if step <= 0:
        return 0.0
    cumulative_lr = 0.0
    if step <= warmup_steps:
        t = step
        cumulative_lr = lr_warmup_start * t + (lr_max - lr_warmup_start) * t**2 / (
            2 * warmup_steps
        )
    else:
        t = warmup_steps
        warmup_cumulative = lr_warmup_start * t + (lr_max - lr_warmup_start) * t**2 / (
            2 * warmup_steps
        )
        decay_step = min(step - warmup_steps, lr_decay_steps)
        if decay_step > 0:
            decay_cumulative = numerical_cosine_integral(
                lr_max, lr_final, lr_decay_steps, decay_step
            )
            cumulative_lr = warmup_cumulative + decay_cumulative
        else:
            cumulative_lr = warmup_cumulative
    return cumulative_lr


def get_lr_at_step(
    step: int,
    lr_warmup_start: float,
    lr_max: float,
    lr_final: float,
    warmup_steps: int,
    lr_decay_steps: int,
) -> float:
    if step <= warmup_steps:
        return lr_warmup_start + (lr_max - lr_warmup_start) * step / warmup_steps
    else:
        decay_step = min(step - warmup_steps, lr_decay_steps)
        if decay_step >= lr_decay_steps:
            return lr_final
        return lr_final + 0.5 * (lr_max - lr_final) * (
            1 + np.cos(np.pi * decay_step / lr_decay_steps)
        )

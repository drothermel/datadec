from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from datadec import constants as consts


def round_value_by_multiple(value: float, multiple: int) -> int:
    return int(round(value / multiple) * multiple)


def model_size_str_to_true_int(size_str: str) -> int:
    return consts.HARDCODED_SIZE_MAPPING[size_str]


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
    model_size = model_size_str_to_true_int(model_size_str)
    batch_size = (
        consts.BS_COEFFICIENT
        * (model_size / consts.MODEL_SIZE_NORM_VALUE) ** consts.BS_EXPONENT
    )
    rounding_size = consts.GPUS_PER_NODE * consts.MICROBATCH_SIZE
    return round_value_by_multiple(batch_size, rounding_size)


def calc_total_tokens_from_str(length_str: str, model_size_str: str) -> int:
    model_size = model_size_str_to_true_int(model_size_str)
    length_in_tokens, length_unit = consts.NUMBER_UNIT_RE.match(
        length_str.strip().upper()
    ).groups()  # type: ignore
    assert length_unit == "XC"
    return int(length_in_tokens) * consts.TOKEN_LEN_XC_MULTIPLIER * model_size


def calc_warmup_tokens(model_size_str: str) -> int:
    model_size = model_size_str_to_true_int(model_size_str)
    batch_size = calc_batch_size(model_size_str)
    return round(model_size / (batch_size / consts.MAX_SEQ_LEN))


def calc_lr_max(model_size_str: str) -> float:
    model_size = model_size_str_to_true_int(model_size_str)
    return (
        consts.LR_MAX_BASE
        * (model_size / consts.MODEL_SIZE_NORM_VALUE) ** consts.LR_EXPONENT
    )


def calc_tokens_per_step(batch_size: int) -> int:
    return batch_size * consts.MAX_SEQ_LEN


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
    config["true_model_size"] = consts.HARDCODED_SIZE_MAPPING[model_size_str]
    config["batch_size"] = calc_batch_size(model_size_str)
    config["total_tokens"] = calc_total_tokens_from_str(
        config["length_str"], model_size_str
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

    return np.trapz(lr_values, t_values)


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

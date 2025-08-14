import math
from typing import Dict, Any

import numpy as np
import pandas as pd

from datadec import constants as consts


def calculate_batch_size(model_size_str: str) -> int:
    model_size = parse_model_size_str(model_size_str)

    raw_batch_size = (consts.MODEL_SIZE_NORM_VALUE / model_size) ** consts.BS_EXPONENT

    rounded_batch_size = round(raw_batch_size / 64) * 64

    samples_per_node = rounded_batch_size / consts.GPUS_PER_NODE
    samples_per_node = max(samples_per_node, consts.MICROBATCH_SIZE)

    return int(samples_per_node * consts.GPUS_PER_NODE)


def calculate_lr_max(model_size_str: str) -> float:
    model_size = parse_model_size_str(model_size_str)

    lr_max = (
        consts.LR_MAX_BASE
        * (model_size / consts.MODEL_SIZE_NORM_VALUE) ** consts.LR_EXPONENT
    )

    return lr_max


def calculate_warmup_tokens(model_size_str: str) -> int:
    model_size = parse_model_size_str(model_size_str)

    warmup_tokens = int(375e6 * (model_size / 108_000_000) ** (1 / 3))

    return warmup_tokens


def parse_model_size_str(size_str: str) -> int:
    if size_str in consts.HARDCODED_SIZE_MAPPING:
        return consts.HARDCODED_SIZE_MAPPING[size_str]

    match = consts.NUMBER_UNIT_RE.match(size_str)
    number, unit = int(match.group(1)), match.group(2).upper()

    unit_multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    return int(number * unit_multiplier[unit])


def parse_token_length_str(length_str: str, model_size_str: str) -> int:
    if length_str == "5xC":
        model_size = parse_model_size_str(model_size_str)
        return 5 * model_size

    match = consts.NUMBER_UNIT_RE.match(length_str)
    number, unit = int(match.group(1)), match.group(2).upper()

    unit_multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    return int(number * unit_multiplier[unit])


def create_model_config(model_size_str: str, **kwargs) -> Dict[str, Any]:
    config = consts.MODEL_CONFIG_BASE.copy()

    if model_size_str in consts.MODEL_SHAPES:
        config.update(consts.MODEL_SHAPES[model_size_str])

    config["lr_max"] = calculate_lr_max(model_size_str)
    config["batch_size"] = calculate_batch_size(model_size_str)
    config["warmup_tokens"] = calculate_warmup_tokens(model_size_str)

    config["tokens"] = parse_token_length_str(config["length_str"], model_size_str)
    config["lr_final"] = config["lr_max"] * consts.LR_FINAL_RATIO
    config["lr_warmup_steps"] = int(
        config["warmup_tokens"] / (config["batch_size"] * consts.MAX_SEQ_LEN)
    )

    config.update(kwargs)
    return config


def create_all_model_configs() -> Dict[str, Dict[str, Any]]:
    return {
        model_size: create_model_config(model_size)
        for model_size in consts.MODEL_SHAPES.keys()
    }


def param_to_numeric(param_str: str) -> float:
    if isinstance(param_str, (int, float)):
        return float(param_str)

    param_str = str(param_str).upper()

    if param_str in consts.HARDCODED_SIZE_MAPPING:
        return float(consts.HARDCODED_SIZE_MAPPING[param_str])

    match = consts.NUMBER_UNIT_RE.match(param_str)
    number = float(match.group(1))
    unit = match.group(2)

    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    return number * multipliers.get(unit, 1)


def get_model_details_df() -> pd.DataFrame:
    configs = create_all_model_configs()
    return (
        pd.DataFrame.from_dict(configs, orient="index")
        .reset_index()
        .rename(columns={"index": "params"})
    )


def get_lr_at_step(
    step: int,
    lr_warmup_steps: int,
    lr_max: float,
    lr_final: float,
    total_steps: int,
) -> float:
    if step <= lr_warmup_steps:
        return lr_max * step / lr_warmup_steps

    cosine_progress = (step - lr_warmup_steps) / (total_steps - lr_warmup_steps)
    cosine_progress = min(cosine_progress, 1.0)

    cosine_factor = 0.5 * (1 + math.cos(math.pi * cosine_progress))
    return lr_final + (lr_max - lr_final) * cosine_factor


def numerical_cosine_integral(
    lr_warmup_steps: int,
    lr_max: float,
    lr_final: float,
    total_steps: int,
    num_points: int = 10000,
) -> float:
    steps = np.linspace(lr_warmup_steps, total_steps, num_points)

    cosine_progress = (steps - lr_warmup_steps) / (total_steps - lr_warmup_steps)
    cosine_factor = 0.5 * (1 + np.cos(np.pi * cosine_progress))
    lr_values = lr_final + (lr_max - lr_final) * cosine_factor

    return float(np.trapz(lr_values, steps))


def calculate_cumulative_lr(
    step: int,
    lr_warmup_steps: int,
    lr_max: float,
    lr_final: float,
    total_steps: int,
) -> float:
    if step <= lr_warmup_steps:
        return 0.5 * lr_max * step**2 / lr_warmup_steps

    warmup_area = 0.5 * lr_max * lr_warmup_steps

    cosine_start = lr_warmup_steps
    cosine_end = min(step, total_steps)
    cosine_length = cosine_end - cosine_start

    if cosine_length <= 0:
        return warmup_area

    total_cosine_length = total_steps - lr_warmup_steps

    progress_end = cosine_length / total_cosine_length

    linear_component = lr_final * cosine_length

    cosine_component = (
        (lr_max - lr_final)
        * total_cosine_length
        / math.pi
        * (math.sin(math.pi * progress_end))
    )

    cosine_area = linear_component + cosine_component

    return warmup_area + cosine_area


def add_lr_cols(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    model_configs = create_all_model_configs()

    lr_data = []
    for _, row in result.iterrows():
        params = row["params"]
        step = row["step"]

        if params in model_configs:
            config = model_configs[params]

            lr_at_step = get_lr_at_step(
                step,
                config["lr_warmup_steps"],
                config["lr_max"],
                config["lr_final"],
                config["tokens"] // (config["batch_size"] * consts.MAX_SEQ_LEN),
            )

            cumulative_lr = calculate_cumulative_lr(
                step,
                config["lr_warmup_steps"],
                config["lr_max"],
                config["lr_final"],
                config["tokens"] // (config["batch_size"] * consts.MAX_SEQ_LEN),
            )

            lr_data.append(
                {
                    "lr_warmup_start": config.get("lr_warmup_start", 0.0),
                    "lr_max": config["lr_max"],
                    "lr_final": config["lr_final"],
                    "lr_at_step": lr_at_step,
                    "cumulative_lr": cumulative_lr,
                }
            )
        else:
            lr_data.append(
                {
                    "lr_warmup_start": np.nan,
                    "lr_max": np.nan,
                    "lr_final": np.nan,
                    "lr_at_step": np.nan,
                    "cumulative_lr": np.nan,
                }
            )

    lr_df = pd.DataFrame(lr_data)
    result = pd.concat([result, lr_df], axis=1)

    return result

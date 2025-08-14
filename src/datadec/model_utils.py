"""Model configuration and calculation utilities for DataDecide.

This module contains utility functions for model configuration generation,
training parameter calculations, and string parsing related to model sizes.
"""

import math
from typing import Dict, Any

import pandas as pd

from datadec import constants as consts


def calculate_batch_size(model_size_str: str) -> int:
    """Calculate batch size for a given model size.

    Args:
        model_size_str: Model size string (e.g., "1B", "300M")

    Returns:
        Calculated batch size for the model
    """
    assert consts.MAX_SEQ_LEN == 2_048
    model_size = parse_model_size_str(model_size_str)
    batch_size = 160 * (model_size / consts.MODEL_SIZE_NORM_VALUE) ** consts.BS_EXPONENT
    rounding_size = consts.GPUS_PER_NODE * consts.MICROBATCH_SIZE
    batch_size /= rounding_size
    batch_size = round(batch_size)
    batch_size *= rounding_size
    return batch_size


def calculate_lr_max(model_size_str: str) -> float:
    """Calculate maximum learning rate for a given model size.

    Args:
        model_size_str: Model size string (e.g., "1B", "300M")

    Returns:
        Maximum learning rate for the model
    """
    model_size = parse_model_size_str(model_size_str)
    return (
        consts.LR_MAX_BASE
        * (model_size / consts.MODEL_SIZE_NORM_VALUE) ** consts.LR_EXPONENT
    )


def calculate_warmup_tokens(model_size_str: str) -> int:
    """Calculate warmup tokens for a given model size.

    Args:
        model_size_str: Model size string (e.g., "1B", "300M")

    Returns:
        Number of warmup tokens for the model
    """
    model_size = parse_model_size_str(model_size_str)
    bs = calculate_batch_size(model_size_str)
    # model_size / bs = num_warmup_steps
    # (model_size / bs) * max_seq_len = num_warmup_tokens
    return round(model_size / (bs / consts.MAX_SEQ_LEN))


def parse_model_size_str(size_str: str) -> int:
    """Parse model size string to get parameter count.

    Args:
        size_str: Model size string (e.g., "1B", "300M")

    Returns:
        Number of parameters as integer
    """
    return consts.HARDCODED_SIZE_MAPPING[size_str]


def parse_token_length_str(length_str: str, model_size_str: str) -> int:
    """Parse token length string to get total tokens.

    Args:
        length_str: Length specification (e.g., "5xC")
        model_size_str: Model size string (e.g., "1B", "300M")

    Returns:
        Total number of tokens
    """
    model_size = parse_model_size_str(model_size_str)
    length_in_tokens, length_unit = consts.NUMBER_UNIT_RE.match(
        length_str.strip().upper()
    ).groups()  # type: ignore
    assert length_unit == "XC"
    length_in_tokens = int(length_in_tokens)
    return length_in_tokens * 20 * model_size


def create_model_config(model_size_str: str, **kwargs) -> Dict[str, Any]:
    """Create a complete model configuration dictionary.

    Args:
        model_size_str: Model size string (e.g., "1B", "300M")
        **kwargs: Additional configuration parameters to override defaults

    Returns:
        Complete model configuration dictionary
    """
    mc = {
        **consts.MODEL_CONFIG_BASE,
        "params": model_size_str,
        "model_size_str": model_size_str,
        **kwargs,
    }
    mc["model_size"] = int(parse_model_size_str(mc["model_size_str"]))
    mc["batch_size"] = int(calculate_batch_size(mc["model_size_str"]))
    mc["lr_max"] = calculate_lr_max(mc["model_size_str"])
    mc["lr_final"] = consts.LR_FINAL_RATIO * mc["lr_max"]
    mc["warmup_tokens"] = int(calculate_warmup_tokens(mc["model_size_str"]))
    mc["total_tokens"] = int(
        parse_token_length_str(mc["length_str"], mc["model_size_str"])
    )
    mc["lr_decay_tokens"] = int(mc["total_tokens"] - mc["warmup_tokens"])
    mc["total_seqs"] = int(round(mc["total_tokens"] / consts.MAX_SEQ_LEN))
    mc["total_steps"] = int(
        math.ceil(mc["total_tokens"] / (mc["batch_size"] * consts.MAX_SEQ_LEN))
    )
    mc["warmup_perc"] = mc["warmup_tokens"] / mc["total_tokens"]
    mc["warmup_steps"] = int(
        math.ceil(mc["warmup_tokens"] / (mc["batch_size"] * consts.MAX_SEQ_LEN))
    )
    mc["lr_decay_steps"] = int(mc["total_steps"] - mc["warmup_steps"])
    mc["theoretical_tokens_per_step"] = int(
        round(consts.MAX_SEQ_LEN * mc["batch_size"])
    )
    mc["10_perc_lrdecay_steps"] = int(round(mc["lr_decay_steps"] * 0.1))
    mc["early_window_10p_end_step"] = mc["warmup_steps"] + mc["10_perc_lrdecay_steps"]
    mc["early_window_perc"] = mc["early_window_10p_end_step"] / mc["total_steps"]
    return mc


def create_all_model_configs() -> Dict[str, Dict[str, Any]]:
    """Create configuration dictionaries for all model sizes.

    Returns:
        Dictionary mapping model size strings to configuration dictionaries
    """
    model_configs = {}
    for param_str, cfg in consts.MODEL_SHAPES.items():
        model_configs[param_str] = create_model_config(param_str, **cfg)
    return model_configs


def get_model_details_df() -> pd.DataFrame:
    """Get DataFrame with model configuration details.

    Returns:
        DataFrame with model configurations as rows and config parameters as columns
    """
    model_configs = create_all_model_configs()
    return pd.DataFrame(model_configs).T.infer_objects()

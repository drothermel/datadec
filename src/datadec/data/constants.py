from __future__ import annotations

import re
from itertools import product

from datadec.config import load_catalog

MODEL_DETAILS_DF_NAME = "model_details"

THOUSAND = 1000
MILLION = 1e6
BILLION = 1e9

_CATALOG = load_catalog()

ALL_MODEL_SIZE_STRS: list[str] = [model.name for model in _CATALOG.models]

MAX_SEQ_LEN = _CATALOG.training.max_sequence_length
TOKEN_LEN_XC_MULTIPLIER = _CATALOG.training.token_length_multiplier
MODEL_SIZE_NORM_VALUE = _CATALOG.training.model_size_normalization
LR_EXPONENT = _CATALOG.training.learning_rate_exponent
LR_MAX_BASE = _CATALOG.training.learning_rate_base
LR_FINAL_RATIO = _CATALOG.training.final_learning_rate_ratio
BS_COEFFICIENT = _CATALOG.training.batch_size_coefficient
BS_EXPONENT = _CATALOG.training.batch_size_exponent
GPUS_PER_NODE = _CATALOG.training.gpus_per_node
MICROBATCH_SIZE = _CATALOG.training.microbatch_size
FLOPS_PER_TOKEN_PER_PARAMETER = (
    _CATALOG.training.flops_per_token_per_parameter
)

MODEL_SHAPES: dict[str, dict[str, int]] = {
    model.name: {
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "n_layers": model.n_layers,
        "mlp_ratio": model.mlp_ratio,
    }
    for model in _CATALOG.models
}
NOMINAL_PARAMETER_COUNTS: dict[str, int] = {
    model.name: model.nominal_parameter_count for model in _CATALOG.models
}
TRAINING_PARAMETER_COUNTS: dict[str, int] = {
    model.name: model.training_parameter_count for model in _CATALOG.models
}
EXACT_PARAMETER_COUNTS: dict[str, int] = {
    model.name: model.exact_parameter_count for model in _CATALOG.models
}
MAX_STEP_TO_USE: dict[str, int] = {
    model.name: model.max_step for model in _CATALOG.models
}
MODEL_CONFIG_BASE: dict[str, bool | float | int | str] = (
    _CATALOG.model_defaults.model_dump()
)

NUMBER_UNIT_RE = re.compile(r"^([0-9]+)([a-zA-Z]+)$")

DATA_RECIPE_FAMILIES = _CATALOG.data_recipe_families
ALL_DATA_NAMES: list[str] = [
    name for family in DATA_RECIPE_FAMILIES.values() for name in family
]
SEED_MAP = _CATALOG.seed_map
PPL_NAME_MAP = _CATALOG.perplexity_name_map
PPL_TYPES: list[str] = list(PPL_NAME_MAP.values())
MMLU_TASKS = _CATALOG.mmlu_tasks
OLMES_TASKS = _CATALOG.olmes_tasks
METRIC_NAMES = _CATALOG.metric_names
DROP_METRICS = _CATALOG.drop_metrics
OLMES_METRICS: list[str] = [
    f"{task}_{metric_type}" for task, metric_type in product(OLMES_TASKS, METRIC_NAMES)
]
ALL_KNOWN_METRICS: set[str] = set(PPL_TYPES) | set(OLMES_METRICS)

PARAM_NUMERIC_COL = "params_numeric"

FULL_ID_COLUMNS: list[str] = [
    "params",
    "data",
    "seed",
    "step",
    "tokens",
    "compute",
]
MEAN_ID_COLUMNS: list[str] = [
    "params",
    "data",
    "step",
    "tokens",
    "compute",
]
KEY_COLUMNS: list[str] = ["params", "data", "seed", "step"]
STEP_TOK_COMP_COLS: list[str] = ["params", "step", "tokens", "compute"]
DWN_DROP_COLS: list[str] = ["chinchilla", "tokens", "compute"]
PPL_DROP_COLS: list[str] = ["__index_level_0__"]
FINAL_PREFIX_COLS: list[str] = FULL_ID_COLUMNS + [
    "total_steps",
    "warmup_steps",
    "lr_max",
    "batch_size",
]
LR_INPUT_COLS: list[str] = [
    "step",
    "lr_warmup_start",
    "lr_max",
    "lr_final",
    "warmup_steps",
    "lr_decay_steps",
]
LR_OUTPUT_COLS: list[str] = ["lr_at_step", "cumulative_lr"]
PREFIX_COLS_WITH_LR: list[str] = FINAL_PREFIX_COLS + LR_OUTPUT_COLS

BASE_RECIPES = _CATALOG.recipe_groups.base_recipes
BASE_AND_QC = _CATALOG.recipe_groups.base_and_quality_control
RECIPES_WITHOUT_ABLATIONS = _CATALOG.recipe_groups.without_ablations
CUSTOM_RECIPE_FAMILIES = _CATALOG.recipe_groups.custom_families
PPL_PERFORMANCE_RECIPE_CHUNKS = _CATALOG.recipe_groups.perplexity_performance
OLMES_PERFORMANCE_RECIPE_CHUNKS = _CATALOG.recipe_groups.olmes_performance

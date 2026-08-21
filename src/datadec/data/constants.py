from __future__ import annotations

import re

from datadec.config import load_catalog

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
FLOPS_PER_TOKEN_PER_PARAMETER = _CATALOG.training.flops_per_token_per_parameter

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
MODEL_CONFIG_BASE: dict[str, bool | float | int | str] = (
    _CATALOG.model_defaults.model_dump()
)

NUMBER_UNIT_RE = re.compile(r"^([0-9]+)([a-zA-Z]+)$")

METRIC_NAMES = _CATALOG.metric_names
DROP_METRICS = _CATALOG.drop_metrics

PARAM_NUMERIC_COL = "params_numeric"

BASE_RECIPES = _CATALOG.recipe_groups.base_recipes
BASE_AND_QC = _CATALOG.recipe_groups.base_and_quality_control
RECIPES_WITHOUT_ABLATIONS = _CATALOG.recipe_groups.without_ablations
CUSTOM_RECIPE_FAMILIES = _CATALOG.recipe_groups.custom_families
PPL_PERFORMANCE_RECIPE_CHUNKS = _CATALOG.recipe_groups.perplexity_performance
OLMES_PERFORMANCE_RECIPE_CHUNKS = _CATALOG.recipe_groups.olmes_performance

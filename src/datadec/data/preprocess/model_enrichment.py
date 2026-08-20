from __future__ import annotations

from typing import Literal

import duckdb
import pyarrow as pa

from datadec.data.model_utils import (
    MODEL_DETAIL_COLUMNS,
    checkpoint_enrichment,
)
from datadec.data.preprocess.duckdb import quote_identifier

LogicalType = Literal["string", "int64", "float64", "bool"]

MODEL_DETAIL_TYPES: tuple[tuple[str, LogicalType], ...] = (
    ("default_seed", "int64"),
    ("length_str", "string"),
    ("lr_warmup_start", "float64"),
    ("d_model", "int64"),
    ("n_heads", "int64"),
    ("n_layers", "int64"),
    ("mlp_ratio", "int64"),
    ("weight_tying", "bool"),
    ("alibi", "bool"),
    ("rope", "bool"),
    ("flash_attention", "bool"),
    ("attention_dropout", "float64"),
    ("attention_layer_norm", "bool"),
    ("include_bias", "bool"),
    ("layer_norm_type", "string"),
    ("layer_norm_with_affine", "bool"),
    ("layer_norm_eps", "float64"),
    ("bias_for_layer_norm", "bool"),
    ("attention_layer_norm_with_affine", "bool"),
    ("activation_type", "string"),
    ("residual_dropout", "float64"),
    ("embedding_dropout", "float64"),
    ("max_sequence_length", "int64"),
    ("vocab_size", "int64"),
    ("embedding_size", "int64"),
    ("eos_token_id", "int64"),
    ("pad_token_id", "int64"),
    ("init_device", "string"),
    ("init_fn", "string"),
    ("init_std", "float64"),
    ("init_cutoff_factor", "int64"),
    ("nominal_parameter_count", "int64"),
    ("training_parameter_count", "int64"),
    ("exact_parameter_count", "int64"),
    ("batch_size", "int64"),
    ("total_tokens", "int64"),
    ("warmup_tokens", "int64"),
    ("lr_max", "float64"),
    ("lr_final", "float64"),
    ("total_steps", "int64"),
    ("total_seqs", "int64"),
    ("warmup_perc", "float64"),
    ("warmup_steps", "int64"),
    ("lr_decay_tokens", "int64"),
    ("lr_decay_steps", "int64"),
    ("tokens_per_step", "int64"),
    ("compute_per_step", "float64"),
)

CHECKPOINT_ENRICHMENT_TYPES: tuple[tuple[str, LogicalType], ...] = (
    ("tokens", "int64"),
    ("compute", "float64"),
    *MODEL_DETAIL_TYPES,
    ("lr_at_step", "float64"),
    ("cumulative_lr", "float64"),
)
CHECKPOINT_ENRICHMENT_COLUMNS: tuple[str, ...] = tuple(
    name for name, _ in CHECKPOINT_ENRICHMENT_TYPES
)

_ARROW_TYPES: dict[LogicalType, pa.DataType] = {
    "string": pa.string(),
    "int64": pa.int64(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
}


def create_model_enrichment_table(
    connection: duckdb.DuckDBPyConnection,
    *,
    checkpoint_select_sql: str,
) -> None:
    checkpoints = connection.execute(
        f"""
        SELECT DISTINCT
            CAST(params AS VARCHAR) AS params,
            CAST(step AS BIGINT) AS step
        FROM ({checkpoint_select_sql})
        ORDER BY params, step
        """
    ).fetchall()
    rows = [
        {
            "params": str(params),
            "step": int(step),
            **checkpoint_enrichment(str(params), int(step)),
        }
        for params, step in checkpoints
    ]
    schema = pa.schema(
        [pa.field("params", pa.string(), nullable=False),
         pa.field("step", pa.int64(), nullable=False)]
        + [
            pa.field(name, _ARROW_TYPES[logical_type], nullable=False)
            for name, logical_type in CHECKPOINT_ENRICHMENT_TYPES
        ]
    )
    relation_name = "_model_enrichment_arrow"
    connection.register(relation_name, pa.Table.from_pylist(rows, schema=schema))
    try:
        connection.execute(
            f"CREATE TEMP TABLE _model_enrichment AS SELECT * FROM {relation_name}"
        )
        connection.execute(
            "CREATE UNIQUE INDEX _model_enrichment_key "
            "ON _model_enrichment (params, step)"
        )
    finally:
        connection.unregister(relation_name)


def enrichment_select_expressions(*, table_alias: str) -> str:
    return ", ".join(
        f"{table_alias}.{quote_identifier(column)} AS {quote_identifier(column)}"
        for column in CHECKPOINT_ENRICHMENT_COLUMNS
    )


assert tuple(name for name, _ in MODEL_DETAIL_TYPES) == MODEL_DETAIL_COLUMNS

__all__ = [
    "CHECKPOINT_ENRICHMENT_COLUMNS",
    "CHECKPOINT_ENRICHMENT_TYPES",
    "MODEL_DETAIL_TYPES",
    "create_model_enrichment_table",
    "enrichment_select_expressions",
]

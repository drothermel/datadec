from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd

from datadec.data.paths import DataDecidePaths

if TYPE_CHECKING:
    from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed
    from datadec.data.ingest.metrics import PerplexityMetrics

PPL_IDENTITY_COLUMNS: tuple[str, ...] = ("params", "data", "seed", "step")
PPL_METRIC_COLUMNS: tuple[str, ...] = (
    "wikitext_103_valppl",
    "pile_valppl",
    "c4_en_valppl",
    "m2d2_s2orc_valppl",
    "ice_valppl",
    "dolma_wiki_valppl",
    "dolma_stack_valppl",
    "dolma_reddit_valppl",
    "dolma_pes2o_valppl",
    "dolma_common_crawl_valppl",
    "dolma_books_valppl",
)
PPL_OUTPUT_COLUMNS: tuple[str, ...] = PPL_IDENTITY_COLUMNS + PPL_METRIC_COLUMNS

PplRunKey: TypeAlias = tuple["ModelSizeName", "DataRecipeName", "Seed"]
PplRowsByKey: TypeAlias = dict[
    PplRunKey,
    dict[int, "PerplexityMetrics"],
]

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


@dataclass(frozen=True, slots=True)
class PplPreprocessResult:
    input_path: Path
    output_path: Path
    checkpoint_count: int
    training_run_count: int


def group_perplexity_rows(ppl_df: pd.DataFrame) -> PplRowsByKey:
    from datadec.data.ingest.enums import DataRecipeName, ModelSizeName, Seed
    from datadec.data.ingest.metrics import PerplexityMetrics

    _assert_metric_field_parity(PerplexityMetrics)
    grouped: PplRowsByKey = defaultdict(dict)
    for row_index, record in enumerate(ppl_df.to_dict(orient="records")):
        run_key = (
            ModelSizeName(record["params"]),
            DataRecipeName(record["data"]),
            Seed(record["seed"]),
        )
        step = _normalize_step(record["step"], row_index=row_index)
        metrics = PerplexityMetrics.model_validate(record)
        if step in grouped[run_key]:
            params, data, seed = run_key
            raise ValueError(
                f"duplicate PPL checkpoint at row {row_index}: "
                f"params={params.value!r}, data={data.value!r}, "
                f"seed={seed.value!r}, step={step}"
            )
        grouped[run_key][step] = metrics
    return dict(grouped)


def flatten_perplexity_rows(grouped: PplRowsByKey) -> pd.DataFrame:
    from datadec.data.ingest.metrics import PerplexityMetrics

    _assert_metric_field_parity(PerplexityMetrics)
    rows: list[dict[str, object]] = []
    for run_key in sorted(grouped, key=lambda key: tuple(item.value for item in key)):
        params, data, seed = run_key
        for step in sorted(grouped[run_key]):
            metrics = grouped[run_key][step]
            rows.append(
                {
                    "params": params.value,
                    "data": data.value,
                    "seed": seed.value,
                    "step": step,
                    **{field: getattr(metrics, field) for field in PPL_METRIC_COLUMNS},
                }
            )
    return _typed_output_dataframe(rows)


def preprocess_ppl(
    paths: DataDecidePaths,
    *,
    verbose: bool = False,
) -> PplPreprocessResult:
    input_path = paths.get_path("ppl_raw")
    output_path = paths.get_path("ppl_processed")

    ppl_df = pd.read_parquet(input_path)
    grouped = group_perplexity_rows(ppl_df)
    output_df = flatten_perplexity_rows(grouped)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)

    result = PplPreprocessResult(
        input_path=input_path,
        output_path=output_path,
        checkpoint_count=len(output_df),
        training_run_count=len(grouped),
    )
    if verbose:
        print(f"ppl input: {result.input_path}")
        print(f"ppl output: {result.output_path}")
        print(f"ppl checkpoints: {result.checkpoint_count}")
        print(f"ppl training runs: {result.training_run_count}")
    return result


def _assert_metric_field_parity(metrics_type: type[PerplexityMetrics]) -> None:
    actual = tuple(metrics_type.model_fields)
    if actual != PPL_METRIC_COLUMNS:
        raise AssertionError(
            "persisted PPL metric columns drift from PerplexityMetrics: "
            f"expected={PPL_METRIC_COLUMNS!r}, actual={actual!r}"
        )


def _normalize_step(value: Any, *, row_index: int) -> int:
    invalid = isinstance(value, (bool, np.bool_)) or value is None or value is pd.NA
    if not invalid:
        try:
            if not isinstance(
                value,
                (str, int, float, Decimal, np.integer, np.floating),
            ):
                raise InvalidOperation
            text = str(value).strip()
            if not text:
                raise InvalidOperation
            decimal_value = Decimal(text)
            invalid = (
                not decimal_value.is_finite()
                or decimal_value != decimal_value.to_integral_value()
            )
        except (InvalidOperation, ValueError):
            invalid = True
    if invalid:
        raise ValueError(
            f"invalid PPL step at row {row_index}: {value!r}; "
            "expected a finite integral int64 value"
        )

    step = int(decimal_value)
    if not _INT64_MIN <= step <= _INT64_MAX:
        raise ValueError(
            f"invalid PPL step at row {row_index}: {value!r}; "
            "expected a finite integral int64 value"
        )
    return step


def _typed_output_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    values_by_column = {
        column: [row[column] for row in rows] for column in PPL_OUTPUT_COLUMNS
    }
    columns: dict[str, pd.Series[Any]] = {
        "params": pd.Series(values_by_column["params"], dtype="string"),
        "data": pd.Series(values_by_column["data"], dtype="string"),
        "seed": pd.Series(values_by_column["seed"], dtype="string"),
        "step": pd.Series(values_by_column["step"], dtype="int64"),
    }
    columns.update(
        {
            field: pd.Series(values_by_column[field], dtype="float64")
            for field in PPL_METRIC_COLUMNS
        }
    )
    return pd.DataFrame(columns, columns=PPL_OUTPUT_COLUMNS)

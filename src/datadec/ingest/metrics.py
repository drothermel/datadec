from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from datadec import constants as consts
from datadec.ingest.coerce import coerce_float

RAW_PPL_TO_FIELD: dict[str, str] = {
    "eval/wikitext_103-validation/Perplexity": "wikitext_103_valppl",
    "eval/pile-validation/Perplexity": "pile_valppl",
    "eval/c4_en-validation/Perplexity": "c4_en_valppl",
    "eval/m2d2_s2orc-validation/Perplexity": "m2d2_s2orc_valppl",
    "eval/ice-validation/Perplexity": "ice_valppl",
    "eval/dolma_wiki-validation/Perplexity": "dolma_wiki_valppl",
    "eval/dolma_stack-validation/Perplexity": "dolma_stack_valppl",
    "eval/dolma_reddit-validation/Perplexity": "dolma_reddit_valppl",
    "eval/dolma_pes2o-validation/Perplexity": "dolma_pes2o_valppl",
    "eval/dolma_common-crawl-validation/Perplexity": "dolma_common_crawl_valppl",
    "eval/dolma_books-validation/Perplexity": "dolma_books_valppl",
}

DROPPED_TASK_METRIC_NAMES: frozenset[str] = frozenset(consts.DROP_METRICS)


class PerplexityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    wikitext_103_valppl: float | None = None
    pile_valppl: float | None = None
    c4_en_valppl: float | None = None
    m2d2_s2orc_valppl: float | None = None
    ice_valppl: float | None = None
    dolma_wiki_valppl: float | None = None
    dolma_stack_valppl: float | None = None
    dolma_reddit_valppl: float | None = None
    dolma_pes2o_valppl: float | None = None
    dolma_common_crawl_valppl: float | None = None
    dolma_books_valppl: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _remap_raw_columns(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if any(key in RAW_PPL_TO_FIELD for key in data):
            return {
                RAW_PPL_TO_FIELD.get(key, key): coerce_float(value)
                for key, value in data.items()
                if key in RAW_PPL_TO_FIELD
            }
        return {key: coerce_float(value) for key, value in data.items()}


class TaskEvalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    correct_choice: float | None = None
    acc_raw: float | None = None
    acc_per_token: float | None = None
    acc_per_char: float | None = None
    acc_per_byte: float | None = None
    acc_uncond: float | None = None
    no_answer: float | None = None
    sum_logits_corr: float | None = None
    logits_per_token_corr: float | None = None
    logits_per_char_corr: float | None = None
    bits_per_byte_corr: float | None = None
    correct_prob: float | None = None
    correct_prob_per_token: float | None = None
    correct_prob_per_char: float | None = None
    margin: float | None = None
    margin_per_token: float | None = None
    margin_per_char: float | None = None
    total_prob: float | None = None
    total_prob_per_token: float | None = None
    total_prob_per_char: float | None = None
    uncond_correct_prob: float | None = None
    uncond_correct_prob_per_token: float | None = None
    uncond_correct_prob_per_char: float | None = None
    uncond_total_prob: float | None = None
    norm_correct_prob: float | None = None
    norm_correct_prob_per_token: float | None = None
    norm_correct_prob_per_char: float | None = None
    primary_metric: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _drop_known_ignored_metrics(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return {
            key: coerce_float(value)
            for key, value in data.items()
            if key not in DROPPED_TASK_METRIC_NAMES
        }

    def average_with(self, others: list["TaskEvalMetrics"]) -> "TaskEvalMetrics":
        group: list[TaskEvalMetrics] = [self, *others]
        averaged: dict[str, float | None] = {}
        for field_name in type(self).model_fields:
            present = [
                getattr(metrics, field_name)
                for metrics in group
                if getattr(metrics, field_name) is not None
            ]
            averaged[field_name] = (
                sum(present) / len(present) if len(present) > 0 else None
            )
        return type(self).model_validate(averaged)


def average_task_metrics(
    metrics_list: list[TaskEvalMetrics],
) -> TaskEvalMetrics | None:
    if len(metrics_list) == 0:
        return None
    head, *tail = metrics_list
    return head.average_with(tail)


TASK_METRIC_FIELDS: list[str] = list(TaskEvalMetrics.model_fields.keys())
_EXPECTED_TASK_METRICS: set[str] = set(consts.METRIC_NAMES)
assert set(TASK_METRIC_FIELDS) == _EXPECTED_TASK_METRICS, (
    f"TaskEvalMetrics fields drift from consts.METRIC_NAMES: "
    f"missing={_EXPECTED_TASK_METRICS - set(TASK_METRIC_FIELDS)} "
    f"extra={set(TASK_METRIC_FIELDS) - _EXPECTED_TASK_METRICS}"
)


PPL_METRIC_FIELDS: list[str] = list(PerplexityMetrics.model_fields.keys())
_EXPECTED_PPL_METRICS: set[str] = set(RAW_PPL_TO_FIELD.values())
assert set(PPL_METRIC_FIELDS) == _EXPECTED_PPL_METRICS, (
    f"PerplexityMetrics fields drift from RAW_PPL_TO_FIELD: "
    f"missing={_EXPECTED_PPL_METRICS - set(PPL_METRIC_FIELDS)} "
    f"extra={set(PPL_METRIC_FIELDS) - _EXPECTED_PPL_METRICS}"
)

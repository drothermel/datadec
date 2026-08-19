from datadec.data.preprocess.ppl import (
    PPL_OUTPUT_COLUMNS,
    PplPreprocessResult,
    flatten_perplexity_rows,
    group_perplexity_rows,
    preprocess_ppl,
)

_OLMES_EXPORTS = {
    "OlmesDetailsPreprocessResult",
    "OlmesPreprocessResult",
    "flatten_olmes_rows",
    "group_olmes_rows",
    "preprocess_olmes",
    "preprocess_olmes_details",
}

__all__ = [
    *sorted(_OLMES_EXPORTS),
    "PPL_OUTPUT_COLUMNS",
    "PplPreprocessResult",
    "flatten_perplexity_rows",
    "group_perplexity_rows",
    "preprocess_ppl",
]


def __getattr__(name: str):
    if name in _OLMES_EXPORTS:
        from datadec.data.preprocess import olmes as olmes_module

        return getattr(olmes_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

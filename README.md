# DataDecide

Library and scripts for downloading and preprocessing DataDecide evaluation artifacts locally.

## Data layout

Artifacts live under `data/` by default:

| Path | Description |
|------|-------------|
| `raw/ppl.parquet` | Raw perplexity export |
| `raw/olmes.parquet` | Raw aggregate OLMES export |
| `raw/olmes-details/models/{recipe}.tar.gz` | Per-recipe OLMES detail archives |
| `raw/scaling-law/{file}.csv` | Three raw scaling-law CSV inputs |
| `reference/published-results/{source path}` | Published result artifacts with their Drive-relative paths |
| `processed/ppl.parquet` | Typed PPL output |
| `processed/olmes.parquet` | Typed aggregate OLMES output |
| `processed/olmes-details/{recipe}/tasks.parquet` | Detail task summaries |
| `processed/olmes-details/{recipe}/instances.parquet` | Per-instance detail rows |
| `processed/olmes-details/{recipe}/choices.parquet` | Per-choice detail rows |

Table schemas are declared in [`configs/olmes.toml`](configs/olmes.toml).

## Download vs preprocess

Download and preprocess are separate steps. Preprocess scripts read local files only and never trigger downloads.

```bash
# Download selected sources
uv run python scripts/download.py --ppl --olmes --olmes-details dolma1.7-no-math-no-code

# Download the three raw scaling-law CSVs (2.86 GB)
uv run python scripts/download.py --scaling-law

# Download the other 131 published artifacts (9.06 GB)
uv run python scripts/download.py --published-results

# Preprocess aggregate sources
uv run python scripts/preprocess_ppl.py
uv run python scripts/preprocess_olmes.py

# Preprocess OLMES detail archives (tasks + instances + choices)
uv run python scripts/preprocess_olmes_details.py --recipe dolma1.7-no-math-no-code
```

Select both Drive-backed options to reconstruct all 134 files (11.92 GB). The
downloads use a pinned inventory from the public Drive folder rather than
crawling its current contents.

All preprocess CLIs accept `--data-dir` (default: repo `data/`). Override paths explicitly when needed:

```bash
uv run python scripts/preprocess_olmes.py --input path/to/raw.parquet --output path/to/out.parquet

uv run python scripts/preprocess_olmes_details.py \
  --recipe dolma1.7-no-math-no-code \
  --input path/to/recipe.tar.gz \
  --output-tasks path/to/tasks.parquet \
  --output-instances path/to/instances.parquet \
  --output-choices path/to/choices.parquet
```

## OLMES detail preprocessing

Detail preprocessing streams one checkpoint at a time through the recipe archive, writing three contract-typed parquet files per recipe:

- **tasks** — task-level metrics and config JSON
- **instances** — one row per `(recipe, params, seed_value, step, task, doc_id)`; heterogeneous native IDs normalized to nullable `native_id` + `native_id_kind`
- **choices** — one row per choice index in each instance's `model_output`

Nullable byte/unconditional fields remain null when absent in the source checkpoint.

## Verification

**Default tests** (`uv run pytest`) use small tar fixtures and run in about a second. They cover schema mapping, nullability, CLI wiring, and verification logic — but not full live archives.

**Manual verification** is for representative recipes after download + preprocess. This checks:

- task / instance / choice counts and primary-key uniqueness
- cross-source parity on the 482 overlapping checkpoints between aggregate and detail for a recipe such as `dolma1.7-no-math-no-code`
- reconstruction of task metrics from instance rows

`bits_per_byte_corr` is declared non-reconstructible from the detail slice in `configs/olmes.toml` and is excluded from reconstruction checks.

```bash
uv run python scripts/preprocess_olmes_details.py --recipe dolma1.7-no-math-no-code
uv run python scripts/verify_olmes_details.py --recipe dolma1.7-no-math-no-code
```

Full-recipe detail preprocessing and verification can take a long time and require multi-GB local data; they are intentionally excluded from the default test suite.

## Development

```bash
uv run ruff check src scripts tests
uv run ty src
uv run pytest
```

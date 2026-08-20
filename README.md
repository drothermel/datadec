# DataDecide

Library and scripts for downloading and preprocessing DataDecide evaluation artifacts locally.

## Data layout

Artifacts live under `data/` by default:

| Path | Description |
|------|-------------|
| `raw/ppl.parquet` | Raw perplexity export |
| `raw/olmes.parquet` | Raw aggregate OLMES export |
| `raw/olmes-details/models/{recipe}.tar.gz` | Per-recipe OLMES detail archives |
| `raw/scaling-law/results_ladder_5xC_seeds.csv` | Google Drive: main raw scaling-law results |
| `raw/scaling-law/results_ladder_5xC_small_seed_extras.csv` | Google Drive: additional small-model seed results |
| `raw/scaling-law/results_ladder_5xC_small_seeds_extra_real.csv` | Google Drive: additional 6M-16M model results |
| `reference/published-results/{source path}` | Google Drive: 131 published result artifacts preserving their source-relative paths |
| `processed/ppl.parquet` | Typed PPL output |
| `processed/olmes.parquet` | Typed aggregate OLMES output |
| `processed/scaling-law/evaluations.parquet` | Typed, precedence-resolved scaling-law task evaluations |
| `processed/scaling-law/checkpoint-losses.parquet` | Typed, reconciled scaling-law checkpoint losses and throughput |
| `processed/olmes-details/{recipe}/tasks.parquet` | Detail task summaries |
| `processed/olmes-details/{recipe}/instances.parquet` | Per-instance detail rows |
| `processed/olmes-details/{recipe}/choices.parquet` | Per-choice detail rows |

OLMES table schemas are declared in [`configs/olmes.toml`](configs/olmes.toml).
Scaling-law source precedence, aliases, seed policy, and table schemas are
declared in [`configs/scaling_law.toml`](configs/scaling_law.toml).
Model definitions and training constants are declared in
[`configs/catalog.toml`](configs/catalog.toml). Each model distinguishes its
nominal parameter count (the size label), training parameter count (the value
used to scale batch size and learning-rate schedules), and exact architectural
parameter count (the value used for FLOP estimates). The configured
`flops_per_token_per_parameter` constant owns the compute multiplier.

Every checkpoint-bearing processed table except the detail instance and choice
tables carries the canonical checkpoint derivations directly, without a later
join: tokens, exact-parameter FLOP compute, model architecture/training details,
`lr_at_step`, and `cumulative_lr`. These fields are present in PPL, aggregate
OLMES, both scaling-law tables, and OLMES detail tasks. Instances and choices
remain evaluation-detail tables keyed to their parent task checkpoint.

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
uv run python scripts/preprocess_scaling_law.py

# Preprocess OLMES detail archives (tasks + instances + choices)
uv run python scripts/preprocess_olmes_details.py --recipe dolma1.7-no-math-no-code
```

Scaling-law preprocessing requires all three local raw CSVs. It validates the
fixed source schema, excludes the recorded blank/6198 legacy-seed rows and
invalid source groups, resolves the historical `baseline` alias and source
overlaps by configured policy, and derives corrected token/compute schedules
from the pinned batch sizes and exact model parameter counts. It writes both
output tables only after both temporary parquet files validate successfully.
It does not download or upload data.

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
uv run python scripts/verify_preprocessed_derivations.py
```

The derivation verifier covers PPL, aggregate OLMES, scaling-law evaluations,
scaling-law checkpoint losses, and OLMES detail task outputs. It excludes the
instance and choice tables. It checks every available raw token/compute value
against the canonical schedule and reports when a raw source instead encodes
nominal-parameter compute. No current preprocessing source records learning-rate
schedule values, so LR derivations can be checked for internal consistency but
not independently confirmed against raw evidence.

The 2026-08-19 full-data validation produced zero token, exact-compute,
model-detail, or LR contradictions in all five processed outputs: 22,709 PPL
rows; 1,410,750 aggregate OLMES rows; 1,788,996 scaling-law evaluation rows;
27,106 scaling-law checkpoint-loss rows; and 35,772 OLMES detail task rows.
The aggregate OLMES raw source also had zero token or exact-compute
contradictions across 1,410,750 rows. Of 2,245,848 raw Google Drive scaling-law
rows, 489,258 contained token and compute evidence: their token values all
matched, their compute values all matched nominal-parameter compute, and all
therefore differed from the standardized exact-parameter compute. Embedded
OLMES detail model configuration had zero contradictions across 35,772 task
rows. Because the raw Google Drive distinction is intentional evidence rather
than a standardized-output failure, the verifier reports those 489,258 raw
exact-compute differences and exits nonzero.

Measured full local preprocessing wall times for that validation were 0.78s
for PPL, 30.58s for aggregate OLMES, 228.94s for scaling-law, and 863.36s for
the 542-checkpoint OLMES detail archive. The detail run wrote 20,423,644
instance rows and 74,384,622 choice rows in addition to its task rows.

Full-recipe detail preprocessing and verification can take a long time and require multi-GB local data; they are intentionally excluded from the default test suite.

## Development

```bash
uv run ruff check src scripts tests
uv run ty check src
uv run pytest
```

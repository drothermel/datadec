# Data Sources Documentation

This document maps out all data sources used by the by-tomorrow-app visualization backend, their origins, and which repos were used to produce them.

## Quick Reference

| Data File | Rows | Source | Processing Repo |
|-----------|------|--------|-----------------|
| `mean_eval_melted.parquet` | 1.8M | AllenAI DataDecide | `datadec`, `dr_hf` |
| `combined_plotting_data_matched.pkl` | 149 | Your wandb experiments | `dr_wandb`, `ft-scaling` |

## Data Sources Overview

### 1. AllenAI DataDecide (Public HuggingFace)

The foundation data comes from AllenAI's DataDecide project, which contains evaluation results for language models trained on various data recipes.

**Public Datasets:**
- `allenai/DataDecide-ppl-results` (22,709 rows) - Perplexity evaluations
- `allenai/DataDecide-eval-results` (1,410,750 rows) - Downstream task evaluations
- `allenai/DataDecide-eval-instances` - Instance-level evaluation data

**What it contains:**
- Model sizes: 4M, 6M, 8M, 10M, 14M, 16M, 20M, 60M, 90M, 150M, 300M, 530M, 750M, 1B
- Data recipes: Dolma1.7, DCLM-Baseline, C4, Falcon, FineWeb, and variants
- Metrics: Perplexity (pile-valppl, wikitext_103-valppl, etc.) and downstream tasks (MMLU, ARC, HellaSwag, etc.)

### 2. Your Private HuggingFace Dataset (`drotherm/dd_parsed`)

A collection of processed data and your own experiment results.

| File | Rows | Origin | Description |
|------|------|--------|-------------|
| `train_results.parquet` | 1.4M | AllenAI | Reparsed eval-results for easier use |
| `macro_avg.parquet` | 235K | AllenAI | Aggregated evaluation results |
| `scaling_law_true.parquet` | 275 | AllenAI (Google Drive) | Ground truth scaling law data |
| `scaling_law_pred_one_step.parquet` | 1,650 | AllenAI (Google Drive) | One-step scaling predictions |
| `scaling_law_pred_two_step.parquet` | 6,600 | AllenAI (Google Drive) | Two-step scaling predictions |
| `wandb_runs_*.parquet` | 495 each | **Your experiments** | Fine-tuning experiment metadata |
| `wandb_history.parquet` | 495 | **Your experiments** | Training history from wandb |

### 3. Local by-tomorrow-app Data (`python-backend/initial_data/`)

These are the files actually served by the FastAPI backend.

#### `mean_eval_melted.parquet` (20MB, 1.8M rows)

**What it is:** The AllenAI DataDecide ppl-results and eval-results merged, averaged across seeds, and melted into long format.

**Schema:**
```
params   | data      | step  | tokens | compute | metric              | value
---------|-----------|-------|--------|---------|---------------------|-------
60M      | Dolma1.7  | 5000  | ...    | ...     | pile-valppl         | 12.34
60M      | Dolma1.7  | 5000  | ...    | ...     | arc_challenge_acc   | 0.45
```

**Processing pipeline:**
1. Download from `allenai/DataDecide-ppl-results` and `allenai/DataDecide-eval-results`
2. Parse and rename columns (via `dr_hf` or `datadec`)
3. Merge ppl and downstream results on (params, data, seed, step)
4. Add tokens and compute columns
5. Average across seeds → "mean_eval"
6. Melt to long format → "melted"

**Repos involved:**
- `dr_hf` - HuggingFace dataset utilities and downloads
- `datadec` - Column mappings, metric names, data recipe definitions

#### `combined_plotting_data_matched.pkl` (9.5MB, 149 rows)

**What it is:** Your fine-tuning experiment results matched against pre-training baselines for comparison plots.

**Schema:** Wide format with 923 columns including:
- `run_id`, `timestamp` - Wandb run identifiers
- `comparison_model_size`, `ckpt_data`, `ckpt_params`, `ckpt_steps` - Checkpoint info
- Many metric columns for fine-tuning vs pre-training comparison

**Processing pipeline:**
1. Download fine-tuning runs from wandb (`ml-moe/ft-scaling` project)
2. Match against corresponding pre-training checkpoints
3. Combine into comparison dataset

**Repos involved:**
- `dr_wandb` - Downloads experiment data from Weights & Biases
- `ft-scaling` - Processing and matching logic

## Related Repos

| Repo | Purpose |
|------|---------|
| `datadec` | Main data processing library for DataDecide data. Constants, parsing, filtering. |
| `dr_hf` | HuggingFace utilities including DataDecide dataset downloading |
| `dr_wandb` | CLI tool for downloading wandb experiment data |
| `dr_duck` | MotherDuck/DuckDB connectors for cloud data access (not yet integrated) |
| `ft-scaling` | Fine-tuning scaling analysis and W&B parsing |

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SOURCE DATA                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  AllenAI HuggingFace              AllenAI Google Drive    Your Wandb     │
│  ├── DataDecide-ppl-results       ├── scaling_law_true    ml-moe/        │
│  ├── DataDecide-eval-results      ├── scaling_law_pred    ft-scaling     │
│  └── DataDecide-eval-instances    └── ...                                │
│                                                                          │
└───────────────┬───────────────────────────┬──────────────────┬──────────┘
                │                           │                  │
                ▼                           ▼                  ▼
┌───────────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐
│       dr_hf/datadec       │  │     ft-scaling      │  │    dr_wandb      │
│  Download & parse AllenAI │  │  Parse scaling law  │  │  Download runs   │
└───────────────┬───────────┘  └──────────┬──────────┘  └────────┬─────────┘
                │                         │                      │
                ▼                         ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    drotherm/dd_parsed (HuggingFace)                      │
│  ├── train_results.parquet      (reparsed AllenAI eval)                  │
│  ├── macro_avg.parquet          (aggregated results)                     │
│  ├── scaling_law_*.parquet      (from Google Drive)                      │
│  └── wandb_*.parquet            (your fine-tuning experiments)           │
└─────────────────────────────────────────────────────────────────────────┘
                │
                │ Further processing
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              by-tomorrow-app/python-backend/initial_data/                │
│  ├── mean_eval_melted.parquet   (merged + averaged + melted AllenAI)     │
│  └── combined_plotting_data_matched.pkl  (FT vs PT comparison)           │
└─────────────────────────────────────────────────────────────────────────┘
                │
                │ Served by FastAPI
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      api.bytomorrow.app                                  │
│  GET /api/visualize/data       → mean_eval_melted.parquet                │
│  GET /api/visualize/data/ft    → combined_plotting_data_matched.pkl      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Future Options

1. **MotherDuck Integration** - The `dr_duck` repo provides a MotherDuck connector that could query HuggingFace datasets directly, eliminating the need for local data files.

2. **Direct HuggingFace Serving** - Could serve data directly from `drotherm/dd_parsed` instead of committing files to git.

3. **Database Migration** - Move processed data to Railway PostgreSQL for SQL-based querying.

## Notes

- The local parquet/pkl files are cached in memory at FastAPI startup (as of Feb 2025 fix)
- Data files are ~30MB total, loaded once into ~100-150MB of RAM
- The `mean_eval_melted` format is optimized for the frontend's filtering needs (by metric, params, data recipe)

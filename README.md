# DataDecide

A Python library for downloading and processing DataDecide datasets from Hugging Face. Provides tools for analyzing ML training experiment results across different model sizes, data recipes, and hyperparameters.

## Quick Start

```bash
# Install
git clone <repository-url>
cd datadec
uv sync && uv add -e .
```

```python
from datadec import DataDecide

# Initialize and process data
dd = DataDecide(data_dir="./data")

# Access datasets
full_eval = dd.full_eval      # Full evaluation results
mean_eval = dd.mean_eval      # Averaged across seeds

# Get analysis-ready DataFrame
analysis_df = dd.get_analysis_df(min_params="10M", add_lr_cols=True)

# Load intermediate data
raw_data = dd.load_dataframe("ppl_raw")
```

## Architecture

**Core Components:**
- **`DataDecide`** - Main interface for dataset access and analysis
- **`DataDecidePaths`** - Centralized path management (see `paths.dataframes` dict)
- **`DataPipeline`** - ETL processing with granular recomputation
- **`DataFrameLoader`** - Lazy loading and caching

**Data Pipeline:**
1. **Download** - Raw datasets from Hugging Face
2. **Metrics Expansion** - Expand JSON metrics (slow: 2-5 min)
3. **Parsing** - Clean and standardize formats
4. **Merging** - Combine perplexity + downstream evaluations
5. **Aggregation** - Statistics across random seeds

## Available Data

Access via `dd.load_dataframe(name)`:
- **Raw:** `ppl_raw`, `dwn_raw`
- **Processed:** `full_eval`, `mean_eval`, `std_eval`
- **Intermediate:** `dwn_metrics_expanded`, `ppl_parsed`, `dwn_parsed`

## Key Features

- **Model sizes:** 4M to 1B parameters
- **Data recipes:** dolma17, c4, fineweb, falcon, dclm, etc.
- **Evaluation tasks:** MMLU, ARC, BoolQ, HellaSwag, perplexity
- **Learning rate schedules:** Warmup + cosine annealing
- **Granular recomputation:** Restart from any pipeline stage

## Performance

- **Slow step:** Metrics expansion (2-5 min) saved as intermediate file
- **Caching:** DataFrames cached in memory after first load
- **Recomputation:** Use `recompute_from="metrics_expand"` to skip download
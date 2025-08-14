# DataDecide

A Python library for downloading and processing DataDecide datasets from Hugging Face. Provides tools for analyzing ML training experiment results across different model sizes, data recipes, and hyperparameters.

## Quick Start

```bash
# Install
git clone <repository-url>
cd datadec
uv sync && uv add -e .

# Demo all features
uv run python scripts/download_data.py
```

```python
from datadec import DataDecide

# Initialize and process data
dd = DataDecide(data_dir="./test_data")

# Access datasets
full_eval = dd.full_eval      # Full evaluation results
mean_eval = dd.mean_eval      # Averaged across seeds

# Get filtered analysis DataFrame
analysis_df = dd.get_filtered_df(min_params="10M")

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

**Static Resources:**
- Dataset metadata: `src/datadec/data/dataset_features.csv` (bundled with package)

## Key Features

- **Model sizes:** 4M to 1B parameters
- **Data recipes:** dolma17, c4, fineweb, falcon, dclm, etc.
- **Evaluation tasks:** MMLU, ARC, BoolQ, HellaSwag, perplexity
- **Learning rate schedules:** Warmup + cosine annealing
- **Granular recomputation:** Restart from any pipeline stage

## Demo Script

The comprehensive demo script showcases all library features:

```bash
# Basic demo with cached data (fast)
uv run python scripts/download_data.py

# Full pipeline from scratch (~2 minutes)  
uv run python scripts/download_data.py --recompute_from all

# Filter to specific model sizes
uv run python scripts/download_data.py --min_params 300M --model_size 1B

# Interactive exploration mode
uv run python scripts/download_data.py --explore --data_recipe "Dolma1.7"

# See all options
uv run python scripts/download_data.py --help
```

## Performance

- **Slow step:** Metrics expansion (2-5 min) saved as intermediate file
- **Caching:** DataFrames cached in memory after first load  
- **Recomputation:** Use `recompute_from="metrics_expand"` to skip download
# DataDecide

DataDecide is a Python library for downloading, processing, and analyzing machine learning experiment data, specifically focusing on language model evaluation results.

> For project-wide context and onboarding, see the [Agent Guide](../dr_ref/docs/guides/AGENT_GUIDE_datadec.md).

## Features

-   **Data Pipeline:** A multi-stage pipeline that downloads raw data from Hugging Face, processes it, and enriches it with additional details.
-   **Easy Data Access:** A simple interface to load and access various dataframes, including raw data, parsed data, and aggregated results.
-   **Advanced Filtering:** Multiple filter types including perplexity (`ppl`), OLMES metrics (`olmes`), and training steps (`max_steps`) with composable combinations.
-   **Scripting Utilities:** Powerful parameter and data selection with `"all"` keyword, exclusion lists, and intelligent validation for reproducible analysis scripts.
-   **Native Plotting:** Production-ready scaling analysis plots using dr_plotter integration.
-   **WandB Integration:** Download and store ML experiment data from Weights & Biases with PostgreSQL backend and incremental updates.

## Getting Started

### Installation

To install the necessary dependencies, run:

```bash
uv sync
source .venv/bin/activate
```

To get dr_plotter:
```bash
uv sync --all-extras

# To update the github version
uv lock --upgrade-package dr_plotter
```


### Usage

The main entry point to the library is the `DataDecide` class. Here's how to use it:

#### Basic Usage

```python
from datadec import DataDecide

# Initialize the DataDecide class, which will run the data processing pipeline
dd = DataDecide(data_dir="./data")

# Access the full evaluation dataframe
full_eval_df = dd.full_eval

# Example of easy indexing
indexed_df = dd.easy_index_df(
    df_name="full_eval",
    data="C4",
    params="10M",
    seed=0,
)

print(indexed_df.head())
```

#### Advanced Filtering

```python
# Filter data with multiple criteria
filtered_df = dd.get_filtered_df(
    filter_types=["ppl", "max_steps"],  # Remove NaN perplexity + apply step limits
    min_params="150M",                  # Only models 150M and larger
    verbose=True                        # Show filtering progress
)

# Filter by specific combinations only
olmes_only_df = dd.get_filtered_df(
    filter_types=["olmes"],            # Keep only rows with OLMES metrics
    return_means=False                 # Get individual seed results
)
```

#### Scripting Utilities

```python
from datadec.script_utils import select_params, select_data

# Flexible parameter selection
params = select_params(["150M", "1B"])                    # Specific models
all_params = select_params("all")                          # All available (sorted)  
large_models = select_params("all", exclude=["4M", "6M"]) # All except smallest

# Data recipe selection  
data_recipes = select_data(["C4", "Dolma1.7"])           # Specific datasets
limited_data = select_data("all", exclude=["C4"])         # All except C4

print(f"Selected {len(params)} models: {params}")
print(f"Selected {len(data_recipes)} datasets: {data_recipes}")
```

### Plotting

Generate scaling analysis plots using the native dr_plotter integration:

```python
# Run the production plotting system
python scripts/plot_scaling_analysis.py

# Generates 7 different plot configurations in plots/test_plotting/
```

### WandB Integration

Download experiment data from Weights & Biases projects with PostgreSQL storage and bulk optimizations:

#### Full Database Download (Optimized)
```bash
# Initial download - uses bulk API optimization (5-6x faster)
python scripts/wandb_download.py --entity your-entity --project your-project --force-refresh

# With parquet export
python scripts/wandb_download.py --entity your-entity --project your-project --force-refresh --output-dir ./wandb_data/
```

#### Incremental Updates (Daily Use)
```bash
# Fast incremental updates - only downloads new/changed runs
python scripts/wandb_download.py --entity your-entity --project your-project

# Custom database connection
python scripts/wandb_download.py --entity your-entity --project your-project --database-url postgresql+psycopg://localhost/custom_db
```

#### Tag Synchronization
```bash
# Sync tags only (very fast)
python scripts/wandb_download.py --entity your-entity --project your-project --sync-tags-only

# Download + sync tags in one operation
python scripts/wandb_download.py --entity your-entity --project your-project --also-sync-tags

# Tag sync modes: recent (default), all, finished
python scripts/wandb_download.py --entity your-entity --project your-project --sync-tags-only --sync-tags-mode all
```

#### Performance Features
- **Bulk API optimization**: 5-6x faster full downloads using bulk history retrieval
- **Incremental updates**: Smart detection of new/unfinished runs
- **Progress reporting**: Real-time progress for all operations
- **Batch processing**: Efficient tag updates with 50-run batches
- **Automatic fallbacks**: Individual downloads if bulk operations fail

See [docs/wandb_integration.md](docs/wandb_integration.md) for complete setup and usage guide.

The `notebooks/explore_data.py` file provides a more detailed example of how to use the library.

## Data

This library uses the following Hugging Face datasets:

-   [allenai/DataDecide-ppl-results](https://huggingface.co/datasets/allenai/DataDecide-ppl-results): Perplexity evaluation results.
-   [allenai/DataDecide-eval-results](https://huggingface.co/datasets/allenai/DataDecide-eval-results): Downstream task evaluation results.

The data processing pipeline downloads these datasets and stores them in the `data_dir` specified during the `DataDecide` initialization.  Then does some filtering, parsing, merging, and pulling in external information about hpms and other training settings.

## Project Structure

```
├── src/datadec/           # Main library code
│   ├── data.py           # Main DataDecide class
│   ├── df_utils.py       # DataFrame utilities and filtering
│   ├── script_utils.py   # Parameter/data selection utilities
│   ├── wandb_store.py    # WandB PostgreSQL storage backend
│   ├── wandb_downloader.py # WandB download logic with incremental updates
│   └── ...              # Pipeline, parsing, constants, etc.
├── scripts/               # Utilities and analysis scripts
│   ├── plot_scaling_analysis.py  # Production plotting system
│   ├── wandb_download.py  # WandB data download CLI script
│   └── legacy_deprecated/ # Archived legacy code
├── docs/                  # Documentation and reports
│   ├── processes/         # Templates and guides
│   ├── wandb_integration.md # WandB system setup and usage guide
│   └── reports/          # Project documentation
├── plots/                 # Generated visualizations
└── notebooks/            # Analysis notebooks
```

## Development

See `docs/processes/reporting_guide.md` for project documentation standards and `CLAUDE.md` for development setup.

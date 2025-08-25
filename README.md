# DataDecide

DataDecide is a Python library for downloading, processing, and analyzing machine learning experiment data, specifically focusing on language model evaluation results.

## Features

-   **Data Pipeline:** A multi-stage pipeline that downloads raw data from Hugging Face, processes it, and enriches it with additional details.
-   **Easy Data Access:** A simple interface to load and access various dataframes, including raw data, parsed data, and aggregated results.
-   **Flexible Filtering:** Methods to easily filter and index the data based on model parameters, datasets, and other criteria.
-   **Native Plotting:** Production-ready scaling analysis plots using dr_plotter integration.

## Getting Started

### Installation

To install the necessary dependencies, run:

```bash
uv sync
source .venv/bin/activate
```

### Usage

The main entry point to the library is the `DataDecide` class. Here's a simple example of how to use it:

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

### Plotting

Generate scaling analysis plots using the native dr_plotter integration:

```python
# Run the production plotting system
python scripts/plot_scaling_analysis.py

# Generates 7 different plot configurations in plots/test_plotting/
```

The `notebooks/explore_data.py` file provides a more detailed example of how to use the library.

## Data

This library uses the following Hugging Face datasets:

-   [allenai/DataDecide-ppl-results](https://huggingface.co/datasets/allenai/DataDecide-ppl-results): Perplexity evaluation results.
-   [allenai/DataDecide-eval-results](https://huggingface.co/datasets/allenai/DataDecide-eval-results): Downstream task evaluation results.

The data processing pipeline downloads these datasets and stores them in the `data_dir` specified during the `DataDecide` initialization.  Then does some filtering, parsing, merging, and pulling in external information about hpms and other training settings.

## Project Structure

```
├── src/datadec/           # Main library code
├── scripts/               # Utilities and analysis scripts
│   ├── plot_scaling_analysis.py  # Production plotting system
│   └── legacy_deprecated/ # Archived legacy code
├── docs/                  # Documentation and reports
│   ├── processes/         # Templates and guides
│   └── reports/          # Project documentation
├── plots/                 # Generated visualizations
└── notebooks/            # Analysis notebooks
```

## Development

See `docs/processes/reporting_guide.md` for project documentation standards and `CLAUDE.md` for development setup.

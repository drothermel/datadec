# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`datadec` is a Python library for downloading and processing DataDecide datasets from Hugging Face. The library provides tools for analyzing ML training experiment results across different model sizes, data recipes, and hyperparameters.

## Commands

### Environment Setup
- `uv sync` - Install/update dependencies
- `uv run python` - Run Python with proper environment

### Development Scripts
- `uv run python scripts/download_data.py --data_dir ./data --force_reload` - Download and process DataDecide datasets

### Package Installation
The package can be installed in development mode:
```bash
uv add -e .
```

## Architecture

### Core Components

**datadec.data**: Main data processing module
- `DataDecide` class: Central interface for dataset management and analysis
- `DataDecidePaths`: File path management for datasets and processed files
- `DataDecideDefaults`: Configuration constants, model architectures, and data recipe mappings
- `prep_base_df()`: High-level function to prepare analysis-ready dataframes

**datadec.model_utils**: Model configuration and learning rate utilities  
- Model parameter calculations and configuration generation
- `add_lr_cols()`: Adds learning rate schedule columns to dataframes
- Learning rate schedule functions for cosine annealing and warmup

### Data Pipeline Architecture

1. **Raw Data Download**: Downloads perplexity and downstream eval datasets from Hugging Face
2. **Parsing**: Converts raw evaluation data into structured formats with standardized columns
3. **Aggregation**: Creates full evaluation datasets by merging perplexity and downstream results
4. **Statistics**: Generates mean/std datasets across random seeds

### Key Data Structures

- **Model Sizes**: 4M to 1B parameter models with predefined architectures
- **Data Recipes**: Organized into families (dolma17, c4, fineweb, falcon, dclm, etc.)
- **Evaluation Tasks**: MMLU, ARC, BoolQ, HellaSwag, and perplexity metrics
- **Step Mapping**: Converts training steps to tokens and compute for analysis

### Dependencies

Core ML/data libraries:
- `datasets` - Hugging Face datasets library
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `pyarrow` - Efficient data serialization
- `huggingface_hub` - HF model/dataset access

## File Structure

```
src/datadec/
├── __init__.py          # Package initialization
├── data.py              # Main data processing classes and functions
├── model_utils.py       # Model configuration and learning rate utilities
├── df_utils.py          # DataFrame manipulation utilities
├── data_utils.py        # Dataset-specific utilities
├── pipeline.py          # ETL processing pipeline
├── loader.py            # DataFrame loading and caching
└── paths.py             # File path management

scripts/
└── download_data.py     # CLI script for dataset download and processing
```

## Usage Patterns

The typical workflow involves:

1. **Initialize DataDecide**: `dd = DataDecide(data_dir="./data")`
2. **Access processed data**: Use properties like `dd.full_eval_ds`, `dd.mean_eval_ds`
3. **Filter and analyze**: Use helper methods for filtering by model size, data recipe, etc.
4. **Add features**: Use `datadec.model_utils.add_lr_cols()` to add learning rate schedules

The library handles caching of processed datasets automatically, with `force_reload` option to refresh data.
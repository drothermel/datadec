# WandB Integration Guide

This guide covers the complete WandB (Weights & Biases) integration system for downloading and storing ML experiment data.

## Overview

The WandB integration provides:
- **PostgreSQL backend** with JSONB storage for flexible experiment data
- **Incremental downloads** - only fetches new/unfinished runs
- **Streaming architecture** - writes data immediately as it downloads
- **Web-ready SQL backend** - perfect for building web frontends
- **Comprehensive testing** - 22 tests with realistic PostgreSQL environment

## Quick Start

### Prerequisites

1. **PostgreSQL running** (via Docker, OrbStack, or local installation)
2. **WandB authentication** - run `wandb login` 
3. **Python dependencies** - installed via `uv sync`

### Basic Usage

```bash
# Download all runs from a WandB project to PostgreSQL
python scripts/wandb_download.py \
  --entity ml-moe \
  --project ft-scaling \
  --database-url postgresql+psycopg://localhost/wandb

# Create database first if needed
createdb wandb
```

### With Parquet Export

```bash
# Download to PostgreSQL and export parquet files
python scripts/wandb_download.py \
  --entity ml-moe \
  --project ft-scaling \
  --database-url postgresql+psycopg://localhost/wandb \
  --output-dir ./wandb_data/
```

## Key Features

### Incremental Downloads

The system automatically detects what's already been downloaded:

```bash
# First run: downloads all 239 runs
python scripts/wandb_download.py --entity ml-moe --project ft-scaling --database-url postgresql+psycopg://localhost/wandb

# Second run: only downloads new runs + updates unfinished ones
# Output: "Found 61 already downloaded runs, Total runs to process: 178"
```

### Smart Run Detection

- **New runs**: Not in database → download everything
- **Unfinished runs**: State ≠ "finished" → re-download to get latest data  
- **Finished runs**: State = "finished" → skip (never changes)

### Data Storage Schema

**Runs Table** (`wandb_runs`):
```sql
CREATE TABLE wandb_runs (
    run_id TEXT PRIMARY KEY,
    run_name TEXT,
    state TEXT,
    project TEXT, 
    entity TEXT,
    created_at TIMESTAMP,
    runtime INTEGER,
    raw_data JSONB  -- All config/summary data
);
```

**History Table** (`wandb_history`):
```sql
CREATE TABLE wandb_history (
    id INTEGER PRIMARY KEY,
    run_id TEXT REFERENCES wandb_runs(run_id),
    step INTEGER,
    timestamp TIMESTAMP,
    metrics JSONB  -- All step metrics
);
```

## Library Usage

### Direct Python API

```python
from datadec import WandBStore, WandBDownloader

# Initialize storage
store = WandBStore("postgresql+psycopg://localhost/wandb")

# Create downloader
downloader = WandBDownloader(store)

# Download project data
stats = downloader.download_project("ml-moe", "ft-scaling")
print(f"Downloaded {stats['new_runs']} new runs")

# Query stored data
runs_df = store.get_runs(project="ft-scaling", state="finished")
history_df = store.get_history(project="ft-scaling")

# Export to parquet
store.export_to_parquet("./export/")
```

### Progress Callbacks

```python
def progress_callback(run_idx, total_runs, run_name):
    print(f"Processing {run_idx}/{total_runs}: {run_name}")

stats = downloader.download_project(
    "ml-moe", "ft-scaling", 
    progress_callback=progress_callback
)
```

## Configuration Options

### CLI Arguments

- `--entity`: WandB entity (username or team name)
- `--project`: WandB project name  
- `--database-url`: PostgreSQL connection string
- `--output-dir`: Optional parquet export directory
- `--force-refresh`: Re-download all data ignoring cache

### Connection Strings

Use the `postgresql+psycopg://` format for psycopg3:

```bash
# Local PostgreSQL
postgresql+psycopg://localhost/wandb

# With authentication  
postgresql+psycopg://user:password@localhost/wandb

# Custom port
postgresql+psycopg://localhost:5433/wandb
```

## Architecture Details

### Why PostgreSQL + JSONB?

The system uses PostgreSQL with JSONB columns because:

1. **Complex WandB objects** - Config/summary data contains nested dictionaries that break parquet serialization
2. **Flexible schema** - JSONB handles arbitrary WandB object structures without predefined schema
3. **Web-ready queries** - SQL backend perfect for building web frontends
4. **Performance** - JSONB provides efficient storage and querying of JSON data
5. **Streaming writes** - Direct database inserts eliminate file rewrite bottlenecks

### Data Flow

1. **API Query** → `wandb.Api().runs(f"{entity}/{project}")`
2. **Incremental Logic** → Compare with existing database state  
3. **Run Download** → `run.scan_history()` for complete, non-downsampled data
4. **Data Processing** → Separate core fields from flexible JSONB data
5. **Database Storage** → Stream writes to PostgreSQL as data arrives
6. **Progress Tracking** → Real-time feedback via callbacks

### Error Handling

- **WandB Authentication**: Clear error messages with instructions
- **Database Connection**: Helpful PostgreSQL connection diagnostics  
- **Data Type Conversion**: Automatic Unix timestamp → PostgreSQL TIMESTAMP
- **Partial Downloads**: Interruptions resume exactly where they left off

## Testing

Run the comprehensive test suite:

```bash
# All WandB integration tests (22 tests)
uv run pytest tests/test_wandb_store.py tests/test_wandb_downloader.py -v

# Just storage tests (10 tests)  
uv run pytest tests/test_wandb_store.py -v

# Just downloader tests (12 tests)
uv run pytest tests/test_wandb_downloader.py -v
```

Tests use **pytest-postgresql** for realistic testing with actual PostgreSQL instances.

## Troubleshooting

### WandB Authentication

```bash
# Login to WandB
wandb login

# Or set environment variable
export WANDB_API_KEY=your_api_key
```

### PostgreSQL Setup

```bash
# Check if PostgreSQL is running
pg_isready -h localhost -p 5432

# Create database
createdb wandb

# Connect and explore
psql wandb -c "SELECT COUNT(*) FROM wandb_runs;"
```

### Common Issues

**"database does not exist"**: Create the database first with `createdb`

**"No module named psycopg2"**: Use correct connection string `postgresql+psycopg://`

**Timestamp conversion errors**: Fixed in v1.0 - Unix timestamps automatically converted

## Performance Notes

- **Download Speed**: ~90 runs/minute typical (WandB API rate limited)
- **Database Performance**: PostgreSQL handles concurrent writes efficiently  
- **Memory Usage**: Streaming design keeps memory usage low
- **Storage Efficiency**: JSONB compression reduces storage vs raw JSON

## Integration with Web Frontends

The PostgreSQL backend is designed for web applications:

```sql
-- Example queries for web frontend
SELECT run_name, state, raw_data->>'accuracy' as accuracy 
FROM wandb_runs 
WHERE project = 'ft-scaling' AND state = 'finished'
ORDER BY CAST(raw_data->>'accuracy' AS FLOAT) DESC;

-- History data for plotting
SELECT step, metrics->>'loss' as loss, metrics->>'accuracy' as accuracy
FROM wandb_history 
WHERE run_id = 'specific_run_id'
ORDER BY step;
```

This provides the foundation for building dashboards, plotting tools, and analysis interfaces that query experiment data directly from PostgreSQL.
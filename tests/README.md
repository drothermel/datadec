# Visual Regression Tests

## Overview

Comprehensive visual regression tests for the datadec plotting system. These tests capture **ALL** visual details of the 7 plot configurations to ensure absolutely no changes occur during refactoring.

## Test Coverage

### 🎯 **Data Point Values**
- **Actual X/Y coordinates** for every line in every plot
- **Finite data point counting** (excludes NaN/infinite values)
- **Data range verification** (min/max values)
- **Data sanity checks** (positive tokens, reasonable perplexity values)

### 🏷️ **Legend Ordering & Content**
- **Exact text ordering** in legends (`texts_ordered`)
- **Handle property ordering** (`handle_properties_ordered`)
- **Cross-validation** between line labels and legend entries
- **Legend titles** for grouped legends (Config 4: "Data", "Params")
- **Multi-legend detection** for configs with grouped legends

### 🎨 **Visual Properties**
- **Line colors, styles, widths** for every line
- **Figure sizes, DPI, positioning**
- **Axis limits, scales, labels, ticks**
- **Subplot sharing behavior** (`sharey`, `sharex_per_row`)
- **Legend positioning and styling**

### 📊 **Configuration-Specific Tests**

#### Config 1: Params as lines, Data as subplots
- ✅ Legend contains all params in order: `["10M", "20M", "60M", "90M"]`
- ✅ Each line has distinct colors from multi-color sequence
- ✅ Y-axis labels only on leftmost subplot (sharey fix)

#### Config 2: Data as lines, Params as subplots  
- ✅ Data ordering preservation from test_data list
- ✅ Cross-validation: line labels match legend entries exactly
- ✅ All 5 data recipes appear in correct order

#### Config 3: MMLU metric
- ✅ Different figure size from Config 1 (default vs explicit)
- ✅ Y-axis values in reasonable MMLU range (0-1)
- ✅ Same structural properties as Config 1

#### Config 4: Multi-metric comparison
- ✅ Grouped legends with titles "Data" and "Params"
- ✅ Multiple line styles due to `style_col="params"`
- ✅ Different Y-limits for different metrics (sharey=False)
- ✅ 12 lines per axis (4 data × 3 params)

#### Config 5: Single data, Purple colormap
- ✅ Single subplot structure
- ✅ Purple colormap colors (different from multi-color sequences)
- ✅ All params represented with distinct colors

#### Config 6: Stacked params lines
- ✅ 2×5 grid layout (2 metrics × 5 data)
- ✅ Proper stacked subplot positioning
- ✅ 4 lines per axis (params)

#### Config 7: Stacked data lines (reduced data set)
- ✅ **NO** 50/50 recipe present (excluded by design)
- ✅ **Exactly** 4 data recipes: `["Dolma1.7", "25%/75%", "75%/25%", "DCLM-Baseline"]`
- ✅ **Exactly** 4 colors (not 5) due to reduced data set  
- ✅ Shared X-axis ranges within each row (`sharex_per_row=True`)
- ✅ 2×5 grid layout (2 metrics × 5 params)

## Key Features

### 🔬 **Rigorous Data Capture**
```python
# Every line's data points are captured
'xdata_values': [1e8, 2e8, 5e8, 1e9, 2e9, ...],  # Actual token values
'ydata_values': [45.2, 42.1, 38.7, 35.4, ...],   # Actual perplexity values
'finite_data_count': 156,  # Number of valid points
```

### 📋 **Legend Verification**
```python
# Exact ordering captured
'texts_ordered': ['10M', '20M', '60M', '90M'],
'handle_properties_ordered': [
    {'color': 'darkred', 'linestyle': '-', ...},
    {'color': 'lightcoral', 'linestyle': '-', ...},
    ...
]
```

### 🎨 **Color & Style Validation**
- Multi-color sequences: `["darkred", "lightcoral", "plum", "lightblue", "darkblue"]`
- Line style sequences: `["-", "--", "-.", ":"]`
- Cross-config consistency verification

### 🔍 **Layout Verification**
- Subplot grid positioning (2×5, 1×4, etc.)
- Axis sharing behavior
- Figure-level vs axis-level legends
- Margin and spacing consistency

## Usage

```bash
# Run all visual regression tests
uv run pytest tests/test_visual_regression.py -v

# Run specific configuration
uv run pytest tests/test_visual_regression.py::TestConfig7::test_config7_visual_properties -v

# Capture any visual changes during refactoring
uv run pytest tests/test_visual_regression.py --tb=short
```

## Failure Examples

If any visual property changes during refactoring, tests will fail with specific details:

```
AssertionError: Line label 'data=Dolma1.7, params=10M' not found in legend
AssertionError: Config 7 should have exactly 4 colors, got 5
AssertionError: Excluded recipe DCLM-Baseline 50% / Dolma 50% should not be present in Config 7
```

These tests establish the **exact baseline** for your plotting system and will catch any unintended changes during refactoring.
# Plotting System Evolution Log

This document chronicles the detailed evolution of the datadec plotting system from initial monolithic implementation to final object-oriented architecture with comprehensive visual configurations.

## Timeline Overview

**Branch**: `08-18-plotting`  
**Base Commit**: `39dc4fb` - Add plotting functionality with cross-subplot color coordination  
**Final Commit**: `209139d` - add final version of config 6 and 7  
**Total Commits**: 12 major iterations  

---

## Phase 1: Initial Implementation (Aug 18, 2025)

### Commit `39dc4fb` - "Add plotting functionality with cross-subplot color coordination"

**Major Changes:**
- Created initial `src/datadec/plotting.py` with monolithic functions
- Added `plot_scaling_curves()` and `plot_model_comparison()` functions
- Integrated dr_plotter's FigureManager for color coordination
- Created `scripts/test_plotting.py` with 5 initial configurations

**Files Added:**
```
src/datadec/plotting.py           (235 lines) - Main plotting functions
scripts/test_plotting.py          (163 lines) - Test configurations  
plots/test_plotting/config*.png   (5 files)  - Initial plot outputs
pyproject.toml                    (+5 lines) - Added dr_plotter dependency
```

**Key Features Implemented:**
- Log/log scaling with proper axis labels
- Cross-subplot color coordination via dr_plotter
- Support for different line/subplot column mappings
- 5 test configurations demonstrating flexibility

**Configurations Created:**
1. **Config 1**: params as lines, data as subplots
2. **Config 2**: data as lines, params as subplots  
3. **Config 3**: MMLU metric visualization
4. **Config 4**: Multi-metric comparison
5. **Config 5**: Single data recipe, multiple parameters

---

## Phase 2: Object-Oriented Refactoring (Aug 20, 2025)

### Commit `add480e` - "orig plotting setup"
- Preserved original plotting.py as backup

### Commit `ba7e44d` - "add updated drplotter"
- Updated dr_plotter dependency for enhanced features
- Modified `pyproject.toml` and `uv.lock`

### Commit `7b64714` - "implement new plotting style and make it work"

**Major Architectural Changes:**
- **DELETED**: Monolithic `src/datadec/plotting.py` (275 lines removed)
- **CREATED**: Object-oriented plotting module structure

**New Module Structure:**
```
src/datadec/plotting/
├── __init__.py              (22 lines) - Public API exports
├── base.py                  (249 lines) - BasePlotBuilder abstract class
├── scaling.py               (186 lines) - ScalingPlotBuilder implementation  
└── model_comparison.py      (205 lines) - ModelComparisonBuilder implementation
```

**Builder Pattern Implementation:**
- `BasePlotBuilder` - Abstract base with common functionality
- `ScalingPlotBuilder` - Fluent interface for scaling curve plots
- `ModelComparisonBuilder` - Multi-metric comparison plots
- Method chaining: `.with_params()`, `.with_data()`, `.configure()`, `.build()`

**Script Updates:**
- `scripts/test_plotting.py` (136 lines modified) - Converted to builder pattern
- Added `fix_sharey_labels()` helper function
- Updated all 5 configurations to use new API

**Visual Improvements:**
- Enhanced subplot layouts with single-row arrangements
- Improved figure sizing (25x5 for Config 1, 20x5 for Config 2)
- Added sharey=True for consistent y-axis scaling

---

## Phase 3: Legend Enhancement (Aug 21, 2025)

### Commit `9a82b5b` - "add a shared legend function"

**New Features:**
- Added `add_unified_legend_below()` helper function
- Automatic legend positioning below subplots
- Configurable legend columns and spacing
- Individual subplot legend removal

### Commit `6e7c848` - "add the ability to select num cols in legend"

**Enhancements:**
- Enhanced legend column control
- Improved spacing calculations
- Better legend positioning algorithms
- Applied to all 4 configurations

### Commit `225b3c2` - "add a way to make a grouped legend, see config 4"

**Major Feature Addition:**
- **NEW**: `add_grouped_legends_below()` function (189 lines)
- Dual legend system for Config 4:
  - **Left Legend**: Data colors with "Data" title
  - **Right Legend**: Parameter line styles with "Params" title
- Sophisticated legend parsing and positioning
- Support for both `line_col` and `style_col` groupings

**Technical Implementation:**
- Legend text parsing: `"data=DCLM-Baseline, params=10M"`
- Custom Line2D element creation for legend handles
- Adaptive positioning based on legend content
- Cross-validation between line labels and legend entries

---

## Phase 4: Data Ordering & Color Enhancement (Aug 21, 2025)

### Commit `fbf772a` - "fix the selection of params and ordering cfg 5"

**Bug Fixes:**
- Fixed parameter ordering in Config 5
- Corrected color sequence application
- Enhanced `param_to_numeric()` integration

### Commit `6ab7a45` - "update data ordering to work for color maps"

**Critical Data Ordering Fixes:**
- **DataFrame Sorting**: Added explicit sorting by `test_data` order
- **Ordering Preservation**: Prevents DataFrame.unique() from scrambling order
- **Cross-Config Consistency**: Applied to Configs 2, 3, 4, 5

**Technical Implementation:**
```python
# Create mapping of data values to their position in test_data list
data_order_map = {data_val: i for i, data_val in enumerate(test_data)}
df_sorted["_temp_data_order"] = df_sorted["data"].map(data_order_map)
df_sorted = df_sorted.sort_values("_temp_data_order").drop(columns=["_temp_data_order"])
```

### Commit `c3934cd` - "more auto set the num cols"

**Automation Improvements:**
- Dynamic `ncols` calculation: `len(test_data)` or `len(test_params)`
- Automatic figure sizing based on subplot count
- Reduced hardcoded values

### Commit `009efe0` - "better colormap"

**Multi-Color Sequence Implementation:**
- **NEW**: `multi_color_sequence` parameter in builders
- **Enhanced BasePlotBuilder**: Added `_create_multi_color_colormap()` method
- **Custom Color Progressions**: `["darkred", "lightcoral", "plum", "lightblue", "darkblue"]`
- **Flexible Color Range**: `color_range_min`/`color_range_max` parameters

**Technical Details:**
- `LinearSegmentedColormap.from_list()` for custom sequences
- Index-based color interpolation vs numeric value interpolation
- Support for both named colormaps ("Purples") and custom sequences

---

## Phase 5: Advanced Features & Bug Fixes (Aug 21, 2025)

### Commit `3f7b2f9` - "enable 50/50 recipe"

**Multi-Color Sequence Bug Fixes:**
- **Fixed**: `scaling.py` condition check for multi-color sequences
- **Enhanced**: `base.py` parameter validation and application
- **Verified**: All configs now properly use custom color progressions

**Line Style Integration:**
- Updated StyleEngine integration with dr_plotter
- Fixed `BASE_THEME.styles['linestyle_cycle']` override
- Proper line style sequence application

### Commit `43a119d` - "everything is formatted the way we want"

**Config 4 Refinements:**
- Perfected grouped legend implementation
- Optimized legend spacing and positioning
- Enhanced multi-metric visualization layout

---

## Phase 6: Stacked Subplot Implementation (Aug 21, 2025)

### Commit `209139d` - "add final version of config 6 and 7"

**Major Feature Addition: Stacked Subplots**

**NEW: Config 6 - Stacked Params Lines**
- **Layout**: 2×5 grid (2 metrics × 5 data recipes)
- **Metrics**: `["pile-valppl", "mmlu_average_correct_prob"]`
- **Line Mapping**: params as lines (colors), data as subplots
- **Features**: `stacked_subplots=True`, unified legend below

**NEW: Config 7 - Stacked Data Lines**
- **Layout**: 2×5 grid (2 metrics × 5 parameters)  
- **Parameters**: `["20M", "60M", "90M", "300M", "1B"]`
- **Data Filtering**: Excludes 50/50 recipe (4 data recipes total)
- **Features**: `sharex_per_row=True`, 4-color progression

**ModelComparisonBuilder Enhancements:**
- **NEW**: `stacked_subplots` parameter
- **NEW**: `_plot_stacked_mode()` method
- **NEW**: `_share_y_axis_per_row()` and `_share_x_axis_per_row()` methods
- **Enhanced**: Grid-based subplot layout management
- **Advanced**: Per-row axis range calculation and sharing

**Technical Implementation:**
```python
# Stacked subplot creation
for metric_idx, metric in enumerate(self.metrics):
    for subplot_idx, subplot_val in enumerate(subplot_values):
        row = metric_idx  # Metrics as rows
        col = subplot_idx  # Subplot values as columns
```

---

## Final Architecture Summary

### Module Structure (862 lines total)
```
src/datadec/plotting/
├── __init__.py              (22 lines)  - Public API
├── base.py                  (293 lines) - BasePlotBuilder + utilities  
├── scaling.py               (186 lines) - ScalingPlotBuilder
└── model_comparison.py      (361 lines) - ModelComparisonBuilder + stacked mode
```

### Helper Functions (571 lines total)
```
scripts/test_plotting.py:
├── fix_sharey_labels()           (17 lines)  - Y-label management
├── add_unified_legend_below()    (126 lines) - Single legend system
└── add_grouped_legends_below()   (282 lines) - Dual legend system
```

### Configuration Portfolio
1. **Config 1**: params lines, data subplots, 5×1 layout, sharey
2. **Config 2**: data lines, params subplots, 4×1 layout, sharey  
3. **Config 3**: MMLU metric, same as Config 1 structure
4. **Config 4**: multi-metric, grouped legends, line styles
5. **Config 5**: single data, purple colormap, sorted params
6. **Config 6**: stacked 2×5, params lines, dual metrics
7. **Config 7**: stacked 2×5, data lines, reduced data set, sharex per row

---

## Key Innovations Achieved

### 1. **Object-Oriented Builder Pattern**
- Fluent interface design
- Method chaining capabilities  
- Separation of concerns
- Extensible architecture

### 2. **Advanced Color Management**
- Multi-color sequence support
- Custom colormap creation
- Index-based color interpolation
- Cross-subplot color coordination

### 3. **Sophisticated Legend System**
- Unified legends below subplots
- Grouped legends for multi-dimensional data
- Automatic ordering preservation
- Cross-validation between lines and legends

### 4. **Data Ordering Preservation**
- Explicit DataFrame sorting
- Prevention of unique() scrambling
- Consistent cross-configuration ordering
- Configurable filter sequences

### 5. **Stacked Subplot Architecture**
- Multi-metric visualization
- Per-row axis sharing
- Grid-based layout management
- Advanced subplot coordination

### 6. **Comprehensive Helper Ecosystem**
- Y-axis label management for sharey
- Legend positioning and styling
- Automatic layout adjustments
- Configuration-specific customizations

---

## Lines of Code Evolution

| Phase | Plotting Code | Test Script | Total | Delta |
|-------|---------------|-------------|-------|-------|
| Initial | 235 | 163 | 398 | +398 |
| OOP Refactor | 662 | 299 | 961 | +563 |
| Legend System | 662 | 571 | 1233 | +272 |
| Final | 862 | 817 | 1679 | +446 |

**Total Growth**: 1,281 lines of sophisticated plotting infrastructure

---

This evolution represents a complete transformation from basic plotting functions to a comprehensive, extensible, and visually sophisticated plotting system with advanced layout management, color coordination, and legend systems.
# Coding Principles - Learnings from DataDec Development

This document captures key coding principles extracted from refactoring experiences, particularly from the `supervised_ft_data_loader.py` evolution. These principles align with DR methodology and should guide future development.

## 1. Conciseness Over Verbosity

**Principle**: Use shorter, clear names rather than overly descriptive ones. Eliminate unnecessary complexity in straightforward operations.

**Before**:
```python
def load_supervised_finetuning_data(db_connection: str, config: Optional[FilterConfig] = None):
    # Long function name that's unnecessarily verbose
```

**After**:
```python
def load_sft_data(runs_df: pd.DataFrame, config: FilterConfig):
    # Concise, clear, and still understandable
```

**Guideline**: If the context makes the meaning clear, prefer brevity. Functions should have names that are scannable and memorable.

---

## 2. Use Assertions for Preconditions

**Principle**: Use `assert` statements for validating expected conditions. Reserve exceptions for actual error handling, not validation.

**Before**:
```python
if missing_params:
    raise ValueError(
        f"Required parameters missing from data: {missing_params}. "
        f"Available columns: {list(runs_df.columns)}"
    )
```

**After**:
```python
def validate_runs_df(runs_df: pd.DataFrame, required_params: List[str]):
    assert "method" not in runs_df.columns, "Method column already exists"
    assert all(col in runs_df.columns for col in required_params), (
        "Required parameters missing"
    )
```

**Guideline**: Assertions document expectations and fail fast. They're for things that "should never happen" in correct usage, not for user input validation.

---

## 3. Extract Magic Values to Named Constants

**Principle**: Replace inline literals and lists with named constants that express intent.

**Before**:
```python
def get_experimental_summary(runs_df: pd.DataFrame) -> ExperimentalSummary:
    experimental_grid = runs_df.groupby(
        ["model_size", "learning_rate"]  # Magic list inline
    ).size()
```

**After**:
```python
GROUP_COLS = ["model_size", "learning_rate"]  # Named constant at top level

def get_experimental_summary(runs_df: pd.DataFrame) -> ExperimentalSummary:
    experimental_grid = runs_df.groupby(GROUP_COLS).size()
```

**Guideline**: If a literal value appears more than once or has semantic meaning, extract it to a named constant.

---

## 4. Namespace Imports for Related Constants

**Principle**: Use namespace imports to provide context and avoid cluttering the local namespace.

**Before**:
```python
from datadec.wandb_eval.wandb_constants import (
    DEFAULT_DB_CONNECTION,
    EARLIEST_GOOD_RUN_DATE,
    METHODS,
    TIME_KEYS,
)
```

**After**:
```python
from datadec.wandb_eval import wandb_constants as wconsts

# Usage: wconsts.METHODS, wconsts.TIME_KEYS, etc.
```

**Guideline**: When importing many related constants from a module, use namespace import to provide clear context and reduce import noise.

---

## 5. Leverage Existing Codebase Structure

**Principle**: Use existing architectural patterns and constants rather than recreating or hardcoding.

**Before**:
```python
def infer_training_method(runs_df: pd.DataFrame) -> pd.DataFrame:
    dpo_params = ['dpo_beta', 'dpo_loss_type']  # Hardcoded list
    has_dpo_params = runs_df[dpo_params].notna().any(axis=1)
```

**After**:
```python
def infer_training_method(runs_df: pd.DataFrame) -> pd.DataFrame:
    dpo_params = wconsts.KEY_SETS["dpo_hpm_cols"]  # Use existing structure
    has_dpo_params = runs_df[dpo_params].notna().any(axis=1)
```

**Guideline**: Before hardcoding values, check if the codebase already has the structure you need. Build on existing patterns.

---

## 6. Atomic Function Responsibility

**Principle**: Each function should have a single, clear purpose. Separate validation, processing, and formatting concerns.

**Before**:
```python
def load_supervised_finetuning_data(db_connection: str, config: Optional[FilterConfig] = None):
    # Database connection + validation + filtering + error handling all mixed
    wandb_store = WandBStore(db_connection)
    runs_df = wandb_store.get_runs()

    if "method" not in runs_df.columns:
        runs_df = infer_training_method(runs_df)

    if config.completed_only and "state" not in runs_df.columns:
        raise ValueError("Completion filtering requested but 'state' column missing.")

    # ... more mixed concerns
```

**After**:
```python
def validate_runs_df(runs_df: pd.DataFrame, required_params: List[str]):
    # Only validation
    assert "method" not in runs_df.columns, "Method column already exists"
    assert "state" in runs_df.columns, "State column missing"

def load_sft_data(runs_df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    # Only filtering logic
    validate_runs_df(runs_df, config.required_params)
    runs_df = infer_training_method(runs_df)
    if config.method is not None:
        runs_df = runs_df[runs_df["method"] == config.method]
    # ... other filtering
```

**Guideline**: If a function does multiple distinct things, split it. Each function should be testable in isolation.

---

## 7. Explicit Contracts Over Implicit Assumptions

**Principle**: Make function expectations crystal clear through validation and clear interfaces.

**Before**:
```python
def load_supervised_finetuning_data(db_connection: str, config: Optional[FilterConfig] = None):
    # Implicit assumptions about what db_connection contains
    # Unclear what happens with None config
    if config is None:
        config = FilterConfig()
```

**After**:
```python
def validate_runs_df(runs_df: pd.DataFrame, required_params: List[str]):
    # Explicit contract about what runs_df should/shouldn't contain
    assert "method" not in runs_df.columns, "Method column already exists"
    assert "state" in runs_df.columns, "State column missing"

def load_sft_data(runs_df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    # Clear input types, no optional parameters with unclear defaults
    validate_runs_df(runs_df, config.required_params)
```

**Guideline**: Functions should validate their inputs and make their expectations explicit. Avoid optional parameters that hide complexity.

---

## Application Guidelines

When reviewing code for these principles:

1. **Scan for long function names** → Check if they can be shortened while remaining clear
2. **Look for `if/raise` patterns** → Consider if they should be assertions instead
3. **Find repeated literals** → Extract to named constants
4. **Count individual imports** → Consider namespace imports for related items
5. **Check for hardcoded lists** → Look for existing constants to reuse
6. **Identify multi-concern functions** → Split into atomic responsibilities
7. **Find implicit assumptions** → Add explicit validation

These principles support the DR methodology goals of clarity, minimalism, and failing fast while creating more maintainable and self-documenting code.
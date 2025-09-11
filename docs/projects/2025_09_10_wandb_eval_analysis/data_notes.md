# WandB Evaluation Data Analysis - Data Notes

## Pretrain Column Overlap Analysis

### Overview
During semantic categorization analysis on 2025-09-10, discovered significant overlap between pretrain metrics from different sources.

### Key Findings

**High Overlap Cases (90.5% match rate):**
- `pretrain_steps` ↔ `model_pretrain_steps` 
- `pretrain_compute` ↔ `model_pretrain_compute`

### Non-Matching Cases Analysis

**Divergent Cases (2 out of 21 shared rows):**
```
Run: 250904-215942_test_match_60M_f...
  Created: 2025-09-05 
  State: running
  pretrain_steps: 15000 vs model_pretrain_steps: "29901"
  pretrain_compute: 2887188480000000000 vs model_pretrain_compute: "5755321516032000000"
```

**Pattern Observations:**
- Non-matching values are substantially different (~2x), not just formatting differences
- Both divergent runs are from "test_match_60M" experiment group
- Both runs are in "running" state vs completed runs in matching cases
- Values suggest different data sources rather than duplicate tracking

### Data Source Hypothesis
- `pretrain_*` columns: likely from experiment configuration
- `model_pretrain_*` columns: likely from model metadata/checkpoint info
- 90.5% overlap indicates most experiments have consistent settings
- 10% divergence could indicate:
  - Configuration errors
  - Model loading inconsistencies  
  - Legitimate differences in tracking sources

### Recommendation
**Keep both column sets** - the divergent cases may be valuable for detecting data quality issues or configuration inconsistencies in experiments.

### Action Items
- [ ] Follow up with collaborator about the source and meaning of these discrepancies
- [ ] Consider adding validation rules to flag when these values diverge significantly
- [ ] Investigate if "running" state runs have different data collection patterns

## Missing Tags Investigation

### Issue Discovered
User reported seeing more tags in WandB UI (like "4M", "10mtx1") than in our parsed data (only 4 tags).

### Root Cause Identified
**WandB Tag Synchronization Issue**: Our download logic only re-downloads unfinished runs, missing tags added to completed runs post-finish.

**Evidence:**
- Older runs (<2025-09-01): 100.0% have tags (126/126)
- Recent runs (>=2025-09-01): Only 35.1% have tags (73/208)
- Download logic in `wandb_downloader.py:35-42` excludes finished runs

### Solution Implemented
Force refresh download with `force_refresh=True` to re-download all runs and capture updated tag data.

**Code Location**: `wandb_downloader.py` - `download_project()` method

### Bug Resolution
**Critical Bug Found**: `wandb_downloader.py` was not capturing `run.tags` - only getting incomplete tags from `run.summary._json_dict`.

**Fix Applied**: Added proper tag capture in `download_run_data()`:
```python
if run.tags:
    run_data["wandb_tags"] = ",".join(run.tags)
```

**Verification**: Test run now correctly shows `10mtx1,4M,finetune` instead of just `finetune`.

### Lesson Learned  
WandB tags require explicit capture from `run.tags` API field, not just summary data. Post-completion tag additions also require periodic refreshes for complete synchronization.

## Timeout vs Non-Timeout Run Analysis

### Dataset Overview

**Total runs analyzed**: 331 (after filtering early test runs)  
**Columns in rebuilt DataFrame**: 25 (ID + status + 18 parsed tag columns)  
**Data quality**: Excellent - no missing core metadata

### Status Column Analysis

**Complete Data Coverage:**
- `run_id`, `project`, `entity`, `state`, `created_at`: **0% NaN** - perfect coverage
- `runtime`: **0% NaN** but all values = 0 (data quality issue - see notes below)
- `timeout`: **40.8% NaN** (135/331 runs) - expected pattern

**Timeout Configuration Pattern:**
- **Single timeout value**: 1800 seconds (30 minutes) for all timed runs
- **196 runs (59.2%)**: Configured with timeout
- **135 runs (40.8%)**: No timeout limit (NaN)

### Run State Distribution

**Timeout runs (196 total)**:
- `finished`: 133 (67.9%)
- `running`: 49 (25.0%) ⚠️ Higher rate of active runs
- `crashed`: 8 (4.1%)
- `failed`: 6 (3.1%)

**No-timeout runs (135 total)**:
- `finished`: 116 (85.9%) ✅ Higher success rate
- `running`: 5 (3.7%)
- `crashed`: 13 (9.6%)
- `failed`: 1 (0.7%)

### Experimental Focus Differences

**No-timeout runs favor established datasets:**
- `dclm-dolma`: 89.6% vs 42.9% (**-46.8% difference**)
- `dolma-qc`: 68.9% vs 35.7% (**-33.2% difference**)

**Timeout runs favor smaller model experiments:**
- `60M`: 14.3% vs 0.0% (**+14.3% difference**)
- `10M`: 12.2% vs 0.0% (**+12.2% difference**)
- Token multiplier tags (`100mt`, `10mtx1`, `10mtx10`, `1mtx1`, `1mtx10`, `1mtx100`): 8.2% vs 0.0% (**+8.2% difference each**)

### Strategic Interpretation

**Timeout runs = Exploratory experiments:**
- Shorter 30-minute time limits
- Focus on smaller models (10M, 60M parameters)
- Testing various token multipliers
- Higher rate of active/running experiments
- Lower completion rate (67.9%)

**No-timeout runs = Production experiments:**
- No time constraints
- Focus on established datasets (dclm-dolma, dolma-qc)
- Higher completion rate (85.9%)
- More comprehensive training runs

### Operational Insights

1. **Two execution contexts**: Different compute environments or experiment types
2. **Risk/reward trade-off**: Timeout runs allow faster iteration but lower success rates
3. **Dataset maturity**: Established datasets get longer, unconstrained runs

### Data Quality Issues

**Runtime Field:**
- **All runs show runtime = 0** - suggests field tracks different metric or gets updated post-completion
- **Validation needed**: History merge will provide actual training progression time

### Next Steps for Validation
- Merge with history data to validate run progression
- Filter runs that didn't progress meaningfully
- Cross-validate completion status with actual training steps

### Tag Analysis Summary

**18 distinct tag categories extracted** from complex tag strings:
- Model sizes: 4M, 10M, 60M, 150M
- Token multipliers: 1mtx1, 1mtx10, 1mtx100, 10mtx1, 10mtx10, 100mt
- Datasets: dclm-dolma, dolma-qc
- Experiment types: finetune, dpo_tune_cache, hidden
- Model matching: match_10M, match_60M, match_150M

**Tag parsing reliability**: 100% coverage - no NaN values in boolean tag columns

**Implementation**: `src/datadec/wandb_eval/parsing.py:285` - Added status columns to rebuild_run_df function

---
*Analysis conducted: 2025-09-10*  
*Data processed through: `parse_and_clean_runs_df()` pipeline*  
*Tag synchronization issue discovered and resolved: 2025-09-10*  
*Timeout analysis completed: 2025-09-10*
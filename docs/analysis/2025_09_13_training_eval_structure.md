# Training vs Evaluation Structure Analysis - 2025-09-13

**Source**: `scripts/clean_wandb/training_vs_eval_analysis.py`

## Training vs Evaluation Run Classification

### Run Type Distribution
- **Training runs**: 321 (94.1%) - have actual training progression data
- **Training (config only)**: 18 (5.3%) - have training config but no history
- **Eval-only runs**: 2 (0.6%) - minimal evaluation-only experiments

### Training Configuration Coverage
- **Core training parameters well-covered**:
  - `num_train_epochs`: 206/341 runs (60.4%)
  - `learning_rate`: 339/341 runs (99.4%)
  - `warmup_ratio`: 206/341 runs (60.4%)
  - `per_device_train_batch_size`: 206/341 runs (60.4%)

### Training History Data Quality
- **94.1% coverage**: 321/341 runs have training progression data
- **High completion rate**: 255/339 training runs finished successfully (75.2%)
- **Rich temporal data**: 20,191 training step records across all runs

### Training Length Analysis
**Configured vs Actual Training Steps**:
- **Configured steps**: Only 4 runs specify max_train_steps (800 steps average)
- **Actual training length**: Mean 3,509 steps, median 1,000 steps
- **Wide range**: 100 to 14,600 actual training steps
- **Configuration gap**: Most runs use epochs rather than step limits

## Evaluation Data Integration Requirements

### DataDecide vs WandB Evaluation Alignment
Based on previous analysis, need to map:
- **Pretraining evaluations** (DataDecide): mmlu_average, arc_challenge, etc.
- **Posttraining evaluations** (WandB oe_metrics): JSON structure with overlapping tasks

### Run Name Pattern Analysis
- **Test runs dominate**: 335/341 runs tagged as "test" experiments
- **Production naming**: Need systematic naming convention for production runs
- **Experiment tracking**: Clear separation between test and production experimental families

## Design Implications for SupervisedFinetuningData

### Training Progression Data Access
1. **94.1% coverage** ensures rich training progression data availability
2. **Step-based temporal alignment** needed (not epoch-based)
3. **Variable training lengths** require flexible progression handling
4. **High success rate** (75%) enables reliable completed-run filtering

### Integration with DataDecide
1. **Training runs need lineage tracking** to pretraining checkpoints
2. **Evaluation alignment required** between pretrain and posttrain formats
3. **Rich temporal data** available for continuous pretraining → posttraining curves
4. **Run completion filtering** essential for reliable analysis

### Schema Validation Requirements
- **Training configuration consistency** needed across 60% of runs
- **History data validation** required for 94% of runs with progression data
- **Run status filtering** critical (only use 75% finished runs for analysis)
- **Flexible step/epoch handling** for different training length specifications
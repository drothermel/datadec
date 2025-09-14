# Training Progression Data Patterns Analysis - 2025-09-13

**Source**: `scripts/clean_wandb/training_history_analysis.py`

## Training History Data Structure

### Data Volume & Coverage
- **321 runs** have training progression data (94.1% coverage)
- **20,191 total training records** across all runs
- **20 columns** of training metrics per step
- **Dense temporal sampling**: Average 63 steps per run (median much lower)

### Core Training Metrics (100% Coverage)
1. **run_id, run_name, project** - Run identification
2. **step, timestamp** - Temporal tracking
3. **train_loss** - Training objective
4. **learning_rate** - Dynamic LR schedules

### Training Efficiency Metrics (57.4% Coverage)
- **total_tokens** - Cumulative token processing
- **per_device_tps** - Throughput measurements
- **total_tokens_including_padding** - Actual compute usage
- **per_device_tps_including_padding** - True efficiency metrics

### Method-Specific Metrics (42.6% Coverage)
- **epoch** - Epoch-based training tracking
- **logps/chosen, logps/rejected** - DPO/preference learning metrics
- **rewards/chosen, rewards/margin, rewards/accuracy** - Reward modeling metrics
- **training_step** - Alternative step counting

## Training Progression Patterns

### Sample Run Analysis (10M Model, 100M Tokens)
- **Training length**: 1,000 steps (100-step increments)
- **Token consumption**: 91.3M total tokens processed
- **Learning rate schedule**: 4.68e-05 → 4.43e-06 (cosine decay, 0.095 final ratio)
- **Loss progression**: 2.23 → 2.17 (steady improvement)

### Key Temporal Alignment Features
- **Step-based tracking** (not epoch-based for most runs)
- **Cumulative token counting** enables alignment with pretraining
- **Dynamic learning rate schedules** (cosine decay common)
- **Dense sampling** enables detailed progression analysis

## Data Quality Assessment

### Coverage by Metric Type
- **Universal metrics**: run_id, step, train_loss, learning_rate (100%)
- **Efficiency metrics**: total_tokens, throughput (57.4% - supervised FT only)
- **Method-specific**: DPO rewards, epoch tracking (42.6% - method-dependent)

### Missing Data Patterns
- **Epoch tracking**: Only 42.6% use epochs (step-based training dominant)
- **Token counting**: Missing in 42.6% (likely DPO runs with different tracking)
- **Method separation**: Clear split between supervised FT vs DPO metric availability

## Design Implications for SupervisedFinetuningData

### Training Progression Data Access
1. **Dense temporal data** available for detailed progression analysis
2. **Step-based indexing** preferred over epoch-based
3. **Token accumulation tracking** enables pretraining → posttraining alignment
4. **Learning rate schedules** captured for optimization analysis

### Schema Validation Requirements
- **Required fields**: run_id, step, train_loss, learning_rate (100% reliable)
- **Optional fields**: total_tokens, throughput metrics (57% coverage)
- **Method-specific fields**: DPO metrics only for DPO runs (42% coverage)
- **Missing value handling**: Clear semantic differences between supervised FT vs DPO

### Integration Capabilities
- **Token-based alignment** with pretraining progression
- **Rich training dynamics** for scaling law analysis
- **Method-specific handling** confirmed necessary (supervised FT vs DPO)
- **Quality filtering** based on metric availability and run completion status

This confirms the need for type-specific handlers - supervised finetuning and DPO have fundamentally different training progression schemas.
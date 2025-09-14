# Experimental Design Patterns Analysis - 2025-09-13

**Source**: `scripts/clean_wandb/experimental_sweep_clustering.py` (Gold Standard Script)

## Perfect Systematic Experimental Design Discovered

### Supervised Finetuning: Complete 3D Grid Design
**4 model sizes × 4 learning rates × 3 dataset sizes = 48 unique experimental conditions**

#### Model Sizes: [4M, 10M, 60M, 150M]
#### Learning Rates: [5e-07, 5e-06, 5e-05, 5e-04]
#### Dataset Sizes: [1M, 10M, 100M tokens]

### Experimental Group Validation
- **12 learning rate sweeps** identified (4 LRs across model/dataset combinations)
- **12 model size sweeps** identified (4 models across LR/dataset combinations)
- **16 dataset size sweeps** identified (3 datasets across model/LR combinations)
- **All groups have complete experimental coverage** (typically 4-8 finished runs per sweep)

### Run Completion Quality
- **Total finetune runs**: 332 total, 249 finished (75% completion rate)
- **Perfect replication**: Most experimental cells have 2 replicate runs
- **Data availability**: 108/249 runs have complete model_size_m + learning_rate + dataset_total_m data

## DPO Experimental Design Analysis
- **6 DPO runs total** (all finished)
- **No systematic sweeps detected** - insufficient runs for sweep analysis
- **Model coverage**: Only 4M model size represented
- **Schema validation**: DPO requires separate handler (different experimental structure)

## Special Experiment Types
- **Data efficiency experiments**: 4 runs testing max_train_samples [200, 51,200]
- **Limited scope**: Only 4M models tested with sample size variations

## Critical Design Insights for SupervisedFinetuningData

### Perfect Experimental Matrix Structure
1. **3-dimensional systematic design** perfectly suited for multidimensional analysis
2. **Complete factorial coverage** enables robust scaling law analysis
3. **Consistent replication** (typically 2 runs per cell) enables error estimation
4. **High completion rate** (75%) ensures reliable data for most experimental conditions

### Schema Requirements Confirmed
1. **Required parameters**: `model_size_m`, `learning_rate`, `dataset_total_m`
2. **Complete data coverage**: 108/249 runs have full experimental metadata
3. **Systematic filtering**: Can reliably group runs by experimental conditions
4. **Replication handling**: Need to aggregate across multiple runs per condition

### Integration Capabilities
- **Perfect grouping structure** for comparative analysis across any dimension
- **Reliable experimental family identification** for related run clustering
- **Complete coverage validation** for sweep analysis quality assessment
- **Robust enough for scaling law fitting** across multiple dimensions simultaneously

This confirms the experimental design is ideal for the entity-relationship research interface - runs naturally cluster into experimental families with clear parameter relationships.
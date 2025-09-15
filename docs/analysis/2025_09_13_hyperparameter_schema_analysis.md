# Hyperparameter Schema Analysis - 2025-09-13

**Source**: `scripts/clean_wandb/hyperparameter_sweep_analysis.py`

## Key Schema Discoveries

### Primary Hyperparameters (Required for SupervisedFinetuningData)
- **Model Size**: 4M, 10M, 60M, 150M (systematic sweep)
- **Learning Rate**: 8 values from 2e-07 to 5e-04 (systematic sweep)
- **Training Method**: 'finetune' (332 runs), 'dpo' (6 runs), 'other' (3 runs)
- **Dataset Tokens**: 1M, 10M, 100M patterns (partial coverage)

### Data Quality Assessment
- **Complete runs**: Only 108/341 (31.7%) have complete model+LR+finished status
- **Missing value patterns**: Significant gaps in dataset token information
- **Schema validation needed**: Wide variation in metadata completeness

### Experimental Design Structure
- **Perfect grid design**: Model Size × Learning Rate shows systematic 4×8 matrix
- **Dataset multipliers**: 1×, 10×, 100× scaling patterns
- **Special experiments**: max_train_samples variations (200 to 140,000 samples)

### Training Method Separation
- **Supervised finetuning**: 332 runs with clean parameter space
- **DPO**: 6 runs, only 4M model size, different hyperparameter requirements
- **Schema implications**: Confirms need for type-specific handlers

### Critical Design Insights
1. **Required parameters**: model_size, learning_rate, training_method
2. **Optional parameters**: dataset_tokens, max_train_samples, loss_reduction
3. **Type-specific validation**: DPO runs have different schema requirements
4. **Data completeness**: Need robust missing value handling

This confirms the hybrid architecture approach - supervised finetuning and DPO need separate handlers with distinct schemas.
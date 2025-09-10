# WandB Data Analysis Scripts

**Exploratory analysis scripts** used to understand the WandB experimental data structure. 

## Main Analysis Script

Use `../wandb_experiment_overview.py` for the clean, production overview.

## Exploratory Scripts (Historical)

- `explore_wandb_data.py` - Initial database exploration
- `enhanced_wandb_analysis.py` - Detailed metric and dimension analysis  
- `training_vs_eval_analysis.py` - Training vs eval-only run classification
- `zero_lr_investigation.py` - Investigation of zero learning rate logging issue
- `run_name_parsing.py` - Run name hyperparameter extraction patterns
- `hyperparameter_sweep_analysis.py` - Complete sweep structure analysis

## Key Findings

- **239 total runs**, 188 finished (78.7% success rate)
- **4 model sizes**: 4M, 10M, 60M, 150M parameters  
- **8 learning rates**: 2e-07 to 5e-04 (systematic sweep)
- **3 dataset sizes**: 1M, 10M, 100M tokens
- **2 training methods**: finetune (182 runs), DPO (6 runs)
- **Complete data**: 108 runs with model+LR+finished status

## Usage

All scripts connect to: `postgresql+psycopg://localhost/wandb_test`
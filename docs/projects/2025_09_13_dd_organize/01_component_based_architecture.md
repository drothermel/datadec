# Option 2: Component-Based Architecture

## Core Concept

Separate managers for orthogonal concerns (training, evaluation, metadata) unified under a single research-focused API. Each manager handles its domain optimally while collaborating through well-defined interfaces.

## Architecture Overview

```python
class TrainingProgressionManager:
    """Handles temporal training data across all training types"""
    def __init__(self, dd: DataDecide, wandb_store: WandBStore):
        self.pretraining_backend = dd
        self.posttraining_backend = wandb_store

    def get_training_curves(self, run_type: str, run_ids: List[str]):
        if run_type == "pretraining":
            return self.pretraining_backend.full_eval
        else:
            return self.posttraining_backend.get_history(run_ids)

    def get_continuous_progression(self, final_run_id: str):
        # Returns pretraining -> posttraining sequence
        links = self._resolve_run_lineage(final_run_id)
        return self._merge_temporal_sequences(links)

class EvaluationManager:
    """Handles all evaluation data across training types"""
    def get_evaluations(self, run_ids: List[str], eval_type: str = "all"):
        # Unified interface for all evaluation types
        if eval_type in ["pretraining", "all"]:
            pretrain_evals = self._get_pretrain_evaluations(run_ids)
        if eval_type in ["posttraining", "all"]:
            posttrain_evals = self._parse_oe_evaluations(run_ids)
        return self._standardize_eval_format(pretrain_evals, posttrain_evals)

class RunRelationshipManager:
    """Manages relationships and lineage between runs"""
    def get_run_lineage(self, run_id: str) -> Dict[str, Any]:
        # Maps pretraining checkpoint -> posttraining runs
        return {"base_run": run_id, "derived_runs": [...]}

    def get_comparable_runs(self, run_id: str) -> List[str]:
        # Finds runs suitable for comparison (same base model, different HPMs)

class DataDecideNext:
    """Single entry point maintaining DataDecide's familiar interface"""
    def __init__(self, data_dir: str, db_connection: str):
        self.dd = DataDecide(data_dir)
        self.wandb = WandBStore(db_connection)

        # Specialized managers handle domain complexity
        self.training = TrainingProgressionManager(self.dd, self.wandb)
        self.evaluations = EvaluationManager()
        self.relationships = RunRelationshipManager()

    def get_scaling_curves(self, include_posttraining: bool = True):
        # High-level research interface
        base_curves = self.training.get_training_curves("pretraining", None)
        if include_posttraining:
            linked_curves = self.training.get_continuous_progressions()
            return self._merge_scaling_data(base_curves, linked_curves)
        return base_curves
```

## Key Design Principles

### Atomic Responsibility
Each manager has a single, clear purpose:
- **TrainingProgressionManager**: Temporal training sequences
- **EvaluationManager**: Assessment snapshots
- **RunRelationshipManager**: Run lineage and comparisons

### Familiar Entry Point
Maintains DataDecide-style simplicity at the top level while hiding complexity in specialized managers.

### Optimized Backends
Each domain uses its optimal data structure without compromising other domains.

### Easy Coordination
Managers collaborate through well-defined interfaces, avoiding tight coupling.

## Pros
- **Atomic responsibility**: Each manager has single, clear purpose
- **Familiar entry point**: Maintains DataDecide-style simplicity
- **Optimized backends**: Each domain uses its optimal data structure
- **Easy coordination**: Managers collaborate through well-defined interfaces

## Cons
- Three managers to coordinate (potential orchestration complexity)
- Run relationships handled separately from run data

## Implementation Flow

### 1. Building Core Data Structures from Initial Sources

**Initialization Phase:**
```python
dd_next = DataDecideNext("./data", "postgresql://localhost/wandb")
```

**Internal Setup Process:**
- **DataDecide instance**: Loads pretraining data using existing pipeline
- **WandBStore instance**: Connects to PostgreSQL, accesses runs/history tables
- **TrainingProgressionManager**:
  - Creates run type mapping (pretraining IDs vs. WandB run IDs)
  - Builds temporal alignment between sparse pretraining checkpoints and dense posttraining steps
  - Caches frequently accessed progression data
- **EvaluationManager**:
  - Parses pretraining evaluation columns from DataDecide
  - Extracts and standardizes oe_metrics JSON from WandB runs
  - Creates unified evaluation schema across both data sources
- **RunRelationshipManager**:
  - Maps WandB model_name_or_path back to pretraining (params, data) combinations
  - Builds lineage graph: pretraining checkpoints → derived posttraining runs
  - Identifies comparable runs within the same experimental family

### 2. Filtering Down to a Specific Sweep for Plotting

**Example: Learning Rate Sweep for 150M Models on Dolma1.7**
```python
sweep_data = dd_next.get_sweep_analysis(
    base_params="150M",
    base_data="Dolma1.7",
    posttraining_method="supervised_ft",
    learning_rates=[1e-5, 3e-5, 1e-4, 3e-4]
)
```

**Internal Processing:**
1. **RunRelationshipManager** identifies base pretraining runs matching (150M, Dolma1.7)
2. **TrainingProgressionManager** finds all posttraining runs derived from those checkpoints
3. **Filter by criteria**: Keep only supervised finetuning runs with specified learning rates
4. **Relationship linking**: Groups pretraining base → posttraining derivatives for continuous curves

### 3. Getting Data Ready to Plot

**Continuous Scaling Curves:**
```python
plot_data = dd_next.get_continuous_curves(sweep_run_ids)
```

**Data Assembly Process:**
1. **TrainingProgressionManager** handles temporal merging:
   - Loads pretraining progression (sparse checkpoints)
   - Loads posttraining progression (dense steps)
   - Aligns timeline: cumulative_tokens = pretraining_tokens + posttraining_tokens
   - Creates seamless curves with proper step/token/compute alignment

2. **EvaluationManager** standardizes metrics:
   - Maps pretraining eval columns to standard names
   - Parses oe_metrics JSON to matching column format
   - Ensures comparable evaluation metrics across training phases

3. **Final DataFrame Structure:**
```python
# Result: Clean, plottable DataFrame
columns = [
    'run_id', 'run_type', 'lineage_id',           # Identity
    'params', 'data', 'learning_rate',             # Experimental variables
    'cumulative_tokens', 'cumulative_compute',     # X-axis options
    'train_loss', 'mmlu_average', 'arc_challenge', # Y-axis metrics
    'pretraining_phase'                             # Plotting aesthetics
]
```

**Usage for Plotting:**
```python
# Ready for dr_plotter - each row is a training step, properly linked
with FigureManager(legend_strategy="figure_below") as fm:
    fm.plot("line", 0, 0, plot_data,
            x="cumulative_tokens", y="mmlu_average",
            hue_by="learning_rate", style_by="pretraining_phase")
```
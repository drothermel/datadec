# Option 3: Entity-Relationship Architecture

## Core Concept

Runs as first-class entities with explicit relationships, leveraging domain-driven design principles. Training runs are objects that know their own data access patterns and relationships to other runs.

## Architecture Overview

```python
class TrainingRun:
    """Base entity representing any training run"""
    def __init__(self, run_id: str, run_type: str):
        self.run_id = run_id
        self.run_type = run_type  # 'pretraining', 'supervised_ft', 'dpo'
        self.parent_run_id: Optional[str] = None
        self.child_run_ids: List[str] = []

    def get_training_progression(self) -> pd.DataFrame:
        # Polymorphic - each subtype handles its own data optimally
        raise NotImplementedError

    def get_evaluations(self) -> pd.DataFrame:
        # Type-specific evaluation loading
        raise NotImplementedError

    def get_full_lineage(self) -> 'TrainingLineage':
        # Returns complete pretraining -> posttraining chain

class PretrainingRun(TrainingRun):
    def __init__(self, run_id: str, dd: DataDecide):
        super().__init__(run_id, "pretraining")
        self.data_source = dd

    def get_training_progression(self):
        return self.data_source.select_subset(/* filters for this run */)

class PosttrainingRun(TrainingRun):
    def __init__(self, run_id: str, wandb_store: WandBStore, parent_run_id: str):
        super().__init__(run_id, "posttraining")
        self.parent_run_id = parent_run_id
        self.data_source = wandb_store

class TrainingLineage:
    """Represents a complete training sequence: pretraining -> posttraining(s)"""
    def __init__(self, base_run: PretrainingRun):
        self.base_run = base_run
        self.derived_runs: List[PosttrainingRun] = []

    def get_continuous_curves(self) -> pd.DataFrame:
        # Seamlessly merges pretraining + posttraining progression
        base_progression = self.base_run.get_training_progression()
        for derived_run in self.derived_runs:
            derived_progression = derived_run.get_training_progression()
            base_progression = self._append_continuation(base_progression, derived_progression)
        return base_progression

class RunRepository:
    """Factory and relationship manager for training runs"""
    def __init__(self, dd: DataDecide, wandb_store: WandBStore):
        self.pretraining_source = dd
        self.posttraining_source = wandb_store
        self._run_cache: Dict[str, TrainingRun] = {}

    def get_run(self, run_id: str) -> TrainingRun:
        if run_id not in self._run_cache:
            run_type = self._identify_run_type(run_id)
            if run_type == "pretraining":
                self._run_cache[run_id] = PretrainingRun(run_id, self.pretraining_source)
            else:
                parent_id = self._resolve_parent_run(run_id)
                self._run_cache[run_id] = PosttrainingRun(run_id, self.posttraining_source, parent_id)
        return self._run_cache[run_id]

    def get_lineage(self, run_id: str) -> TrainingLineage:
        # Builds complete training lineage for analysis

class DataDecideEvolved:
    def __init__(self, data_dir: str, db_connection: str):
        self.repository = RunRepository(DataDecide(data_dir), WandBStore(db_connection))

    def get_scaling_analysis(self, final_run_ids: List[str]):
        lineages = [self.repository.get_lineage(run_id) for run_id in final_run_ids]
        return [lineage.get_continuous_curves() for lineage in lineages]
```

## Key Design Principles

### First-Class Entities
Training runs are objects with their own data access methods and relationship management.

### Explicit Relationships
Run lineage is explicitly modeled rather than inferred from metadata.

### Type Safety
Each run type is optimized for its specific data patterns and access requirements.

### Natural Mental Model
Matches how researchers think about experiments (base runs -> derived runs).

## Pros
- **Explicit relationships**: Run lineage is first-class, not inferred
- **Natural mental model**: Matches researcher thinking about experiments
- **Type safety**: Each run type optimized for its data patterns
- **Powerful queries**: Easy to find related runs, compare lineages

## Cons
- **Complex object relationships**: Potential circular references, memory management
- **Higher abstraction**: More concepts for researchers to learn
- **Relationship management overhead**: Explicit linking requires maintenance

## Implementation Flow

### 1. Building Core Data Structures from Initial Sources

**Initialization Phase:**
```python
dd_evolved = DataDecideEvolved("./data", "postgresql://localhost/wandb")
```

**Internal Setup Process:**
- **RunRepository initialization**:
  - Scans DataDecide data to catalog all pretraining runs
  - Scans WandBStore to catalog all posttraining runs
  - Creates run type mapping and parent-child relationships
  - Builds comprehensive run registry without loading actual data (lazy loading)

- **Relationship graph construction**:
  - Maps WandB model_name_or_path to corresponding pretraining (params, data) combinations
  - Creates parent_run_id → [child_run_ids] mapping
  - Identifies run families for experimental comparison

- **TrainingRun object creation** (lazy):
  - PretrainingRun objects know how to access DataDecide data for their specific run
  - PosttrainingRun objects know how to access WandB data and link to parent run
  - Each run type optimized for its own data access patterns

### 2. Filtering Down to a Specific Sweep for Plotting

**Example: Learning Rate Sweep for 150M Models on Dolma1.7**
```python
# Find base lineages matching criteria
base_lineages = dd_evolved.repository.find_lineages(
    base_params="150M",
    base_data="Dolma1.7"
)

# Filter derived runs within each lineage
sweep_lineages = []
for lineage in base_lineages:
    matching_runs = lineage.filter_derived_runs(
        posttraining_method="supervised_ft",
        learning_rates=[1e-5, 3e-5, 1e-4, 3e-4]
    )
    if matching_runs:
        sweep_lineages.append(lineage)
```

**Internal Processing:**
1. **RunRepository** queries the relationship graph:
   - Finds all PretrainingRun objects matching base criteria
   - For each base run, retrieves associated TrainingLineage
   - Each TrainingLineage knows its derived PosttrainingRun objects

2. **TrainingLineage filtering**:
   - Each lineage filters its derived_runs based on posttraining criteria
   - Returns only lineages that have matching posttraining experiments
   - Maintains explicit parent-child relationships throughout

3. **Run-level filtering**:
   - Individual TrainingRun objects can filter their own data
   - PosttrainingRun objects filter based on hyperparameter criteria
   - PretrainingRun objects provide base progression data

### 3. Getting Data Ready to Plot

**Continuous Scaling Analysis:**
```python
plot_data_list = []
for lineage in sweep_lineages:
    lineage_curves = lineage.get_continuous_curves()
    plot_data_list.append(lineage_curves)

# Combine all lineages for comprehensive plotting
combined_plot_data = pd.concat(plot_data_list, ignore_index=True)
```

**Data Assembly Process:**
1. **TrainingLineage orchestration**:
   - Each TrainingLineage manages its own pretraining → posttraining continuity
   - Base PretrainingRun provides progression data using DataDecide interface
   - Each PosttrainingRun provides its progression data using WandB interface
   - Lineage handles temporal alignment and token/compute accumulation

2. **TrainingRun-level data access**:
   - **PretrainingRun.get_training_progression()**:
     - Uses DataDecide's optimized filtering for the specific run
     - Returns sparse checkpoint data with evaluations
   - **PosttrainingRun.get_training_progression()**:
     - Queries WandB history for dense step-by-step training data
     - Includes parsed oe_metrics evaluations at final step
   - **PosttrainingRun.get_evaluations()**:
     - Parses oe_metrics JSON for standardized evaluation format

3. **Final DataFrame Assembly**:
```python
# Each TrainingLineage produces a DataFrame segment
# Structure optimized for continuous plotting
columns = [
    'lineage_id', 'run_id', 'run_type',              # Lineage tracking
    'params', 'data',                                 # Base experimental variables
    'learning_rate', 'posttraining_method',          # Derived experimental variables
    'cumulative_tokens', 'cumulative_compute',       # Continuous X-axis
    'train_loss', 'mmlu_average', 'arc_challenge',   # Y-axis metrics
    'checkpoint_type'                                 # pretraining vs posttraining
]
```

**Usage for Plotting:**
```python
# Each row represents a training checkpoint with full lineage context
with FigureManager(legend_strategy="figure_below") as fm:
    fm.plot("line", 0, 0, combined_plot_data,
            x="cumulative_tokens", y="mmlu_average",
            hue_by="learning_rate", style_by="checkpoint_type",
            facet_by="lineage_id")  # Easy to separate different base models
```

## Key Implementation Advantages

1. **Natural object relationships**: TrainingLineage explicitly models the pretraining → posttraining relationship
2. **Optimized data access**: Each TrainingRun type uses its optimal backend (DataDecide vs WandB)
3. **Lazy loading**: Run objects created without loading data until needed
4. **Clear lineage tracking**: Easy to trace any posttraining run back to its pretraining base
5. **Type-specific optimization**: PretrainingRun and PosttrainingRun handle their data access patterns optimally
# Hybrid Architecture: Type-Specific Data Handlers + Entity-Relationship Research Interface

## Core Design Insight

After analyzing the component-based and entity-relationship architectures, we identified that **different phases of the research workflow have different optimization needs**:

- **Phase 1 (Data Ingestion)**: Component-based architecture excels due to heterogeneous data sources and processing needs
- **Phase 2 (Research Analysis)**: Entity-relationship architecture excels for exploring relationships and experimental comparisons

## Hybrid Architecture Overview

```python
class DataDecideNext:
    def __init__(self, data_dir: str, db_connection: str):
        # Phase 1: Type-specific data handlers (component-based)
        self.supervised_ft = SupervisedFinetuningData(db_connection)
        self.dpo_ft = DPOFinetuningData(db_connection)
        self.pretraining_evals = PretrainingEvaluations(data_dir)
        self.posttraining_evals = PosttrainingOEEvaluations(db_connection)

        # Phase 2: Entity-based research interface
        self.repository = RunRepository(self.supervised_ft, self.dpo_ft, ...)

    # Research-focused methods use entity interface
    def explore_available_sweeps(self) -> Dict[str, List[TrainingLineage]]:
        return self.repository.catalog_experimental_families()
```

## Key Design Principles

### 1. Type-Specific Schemas vs Generic Blob Storage

**Problem with current approach**: Everything dumped into one metadata table
- DPO hyperparameters mixed with supervised finetuning hyperparameters
- Missing values are ambiguous: "Not applicable" vs "Missing data" vs "Default value"
- No schema validation or expectation management

**Type-specific solution**: Each training/evaluation type gets its own handler with explicit schema

```python
class SupervisedFinetuningData:
    expected_hparams = ['learning_rate', 'batch_size', 'num_epochs', 'max_seq_length']
    optional_hparams = ['warmup_ratio', 'weight_decay']

    def load_runs(self) -> pd.DataFrame:
        # Returns clean DataFrame with expected columns
        # Clear errors for missing required hparams
        # NaN for missing optional hparams

class DPOFinetuningData:
    expected_hparams = ['dpo_beta', 'dpo_loss_type', 'reference_model_path']
    optional_hparams = ['dpo_label_smoothing', 'dpo_gamma_beta_ratio']

    def load_runs(self) -> pd.DataFrame:
        # Different schema, different validation
```

### 2. Explicit Evaluation Mappings

**Challenge**: Two evaluation systems with overlapping but not identical task sets and metrics

```python
class PretrainingEvaluations:
    standard_tasks = ['mmlu_average', 'arc_challenge', 'hellaswag', 'winogrande']

class PosttrainingOEEvaluations:
    standard_tasks = ['csqa', 'piqa', 'boolq', 'arc_easy', 'arc_challenge']
    task_mapping = {'arc_challenge': 'arc_challenge'}  # Explicit mapping

    def standardize_to_pretraining_format(self) -> pd.DataFrame:
        # Handles name mapping and metric alignment
```

### 3. Phase Separation for Optimal Design

**Phase 1: Data Processing** (Component-based strengths)
- Different data sources need specialized handling
- Different temporal patterns (sparse vs dense) need different processing
- Different formats (structured vs JSON) need different parsers
- Type-specific schema validation and cleaning

**Phase 2: Research Interface** (Entity-relationship strengths)
- "What can I compare?" → TrainingLineage objects make relationships explicit
- "What's varying in this experiment?" → Run objects know their parameters
- "What sweeps exist?" → Repository queries relationship graph naturally
- "How do I group related runs?" → Lineage objects group pretraining → posttraining families

## Benefits of This Approach

### Schema Validation & Data Quality
- Each class knows what it expects (required vs optional parameters)
- Clear missing data semantics: Missing required vs optional vs not-applicable
- Type-specific validation catches data quality issues early

### Maintainability & Extension
- Add new training/evaluation types without touching existing code
- Type-specific optimization for each data source
- Clear boundaries between different concerns

### Research Workflow Support
- Natural entity relationships for complex research queries
- Familiar DataDecide-style interface at top level
- Explicit lineage tracking for pretraining → posttraining relationships

## Implementation Strategy

### Build from Concrete to Abstract

Rather than building the full architecture upfront, we'll **start with specialized data handlers** to:

1. **Surface actual data constraints** we haven't discovered yet
2. **Reveal edge cases** that would break higher-level abstractions
3. **Clarify integration points** between different data types
4. **Validate assumptions** about what data actually exists and how it's structured
5. **Inform better entity relationships** once we understand real data patterns

### Proposed Implementation Order

1. **SupervisedFinetuningData** - Start with most complete dataset
2. **PretrainingEvaluations** - Leverage existing DataDecide knowledge
3. **PosttrainingOEEvaluations** - Figure out mapping/alignment challenges
4. **DPOFinetuningData** - Handle smaller, different dataset
5. **Integration analysis** - Make informed decisions about entity relationships

Each handler should be **self-contained and testable** - load its data, validate its schema, surface problems clearly.

### Design Philosophy Alignment

This approach follows the DR methodology:

- **Clarity through structure**: Each data type has its own optimized handler
- **Architectural courage**: Clean separation rather than generic blob approach
- **Fail fast**: Problems surface at handler level, not after complex orchestration
- **Minimize friction**: Researchers work with clean, validated data schemas

## Next Steps

Begin implementation with `SupervisedFinetuningData` handler to examine actual WandB data structure and validate our assumptions about the hyperparameter landscape.
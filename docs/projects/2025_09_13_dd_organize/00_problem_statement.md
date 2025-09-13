# Problem Statement: Forced Unification of Fundamentally Different Data Types

## The Core Issue

We've identified a **classic data architecture anti-pattern**: forcing conceptually distinct data structures into a single unified format, which paradoxically makes analysis harder rather than easier.

The current system **conflates different data dimensions** that have fundamentally different:
- **Temporal characteristics** (sparse checkpoints vs. dense logging)
- **Semantic meaning** (training progression vs. evaluation snapshots vs. run metadata)
- **Access patterns** (time-series analysis vs. cross-sectional comparison vs. metadata filtering)

## Specific Data Type Conflicts

### 1. Training Runs (temporal sequences with different granularities)
- **Pretraining**: Sparse checkpoints with evaluations, clean DataDecide structure
- **Post-training**: Dense step-by-step logs, messy WandB metadata structure
- **Different training types**: Each has distinct hyperparameter spaces and settings

### 2. Evaluations (assessment snapshots at different temporal points)
- **Pretraining evals**: Checkpoint-aligned, well-structured
- **Post-training evals**: Potentially decoupled from training steps, stored in oe_metrics
- **Chat evals**: Possibly missing/inconsistent

### 3. Metadata (static run characteristics)
- **Pretraining**: Clean (data recipe, model size, cumulative metrics)
- **Post-training**: Messy WandB nested structures with many variables

## The Analysis Goals Being Hindered

The desired analysis capabilities:
- **Plot pretraining + post-training together** → but they're at different temporal granularities
- **Compare within/across post-training types** → but they're buried in a massive unified table
- **Match pretraining to paired post-training** → but the relationship is obscured by forced denormalization
- **Intuitive data access** → but everything is flattened into one unwieldy structure

## The Fundamental Design Flaw

Instead of recognizing that **different data types need different storage patterns optimized for their specific access patterns**, the current system forces everything into a single "god table" that serves no use case well.

**The result**: More complex to query, harder to plot, and conceptually confusing - the opposite of what a good data architecture should provide.

## Solution Requirements

The new system should:
1. **Separate concerns**: Different data types stored in structures optimized for their access patterns
2. **Preserve relationships**: Clear linking between pretraining runs and their post-training derivatives
3. **Enable flexible analysis**: Easy to combine data when needed, easy to analyze separately when appropriate
4. **Maintain intuitive access**: DataDecide-style interface that makes complex operations simple
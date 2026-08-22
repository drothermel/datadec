# Potential projects — LR schedule / WSD on DataDecide

One document per candidate workshop-paper project, derived from the idea inventory in
[../refs/lr-schedule-wsd-synthesis.md](../refs/lr-schedule-wsd-synthesis.md). Each project is
written to stand alone so it can be evaluated on its own merits; cross-project dependencies are
called out explicitly where they exist. IDs (A1, C2, …) refer to the synthesis inventory.

Each document has the same three parts:

1. **What the project involves** — the core experiment plus the optional directions.
2. **Doability and impact** — an overall doability take, then per-direction workshop-paper impact.
3. **Infrastructure build sequence** — what to build, in what order, if we proceed.

| Project | Core question | Training required | Standalone paper? |
|---------|---------------|-------------------|-------------------|
| [A — Annealed readouts](a-annealed-readouts.md) | How much of DataDecide's reported ranking is a cosine-schedule artifact, and can it be corrected for the cost of evals? | short decay branches | **Yes — strongest candidate** |
| [B — WSD retrain suite](b-wsd-retrain-suite.md) | What does a DataDecide-subset with a proper stable phase + decay branches enable, and is it worth keeping the cluster warm for? | full retrain (subset) | Resource paper; better as an enabler |
| [C — Geometry & comparability](c-geometry-comparability.md) | Are cross-recipe metric comparisons well-defined, and does basin membership predict when recipe effects hold? | none | Yes, evals-only |
| [D — Token-level decomposition](d-token-level-decomposition.md) | Which tokens respond to LR decay, does that track epistemic uncertainty, and do recipes differ in how tokens migrate over training? | needs branches (from A or B) | Highest ceiling; depends on A/B |
| [Track C — Recipe featurization](recipe-featurization.md) | What is actually in the DataDecide recipes, and which measurable data properties explain which task-level differences? | none | Yes, GPU-free |

## Tracks (from the published-data-analysis synthesis)

One document per candidate workshop-paper project, derived from
[../refs/research-trajectory-synthesis.md](../refs/research-trajectory-synthesis.md).
Each is written to stand alone so projects can be compared and chosen
independently; shared infrastructure is restated per document rather than
factored out. Idea IDs (`A1`, `B3`, …) match the synthesis.

| Doc | Project | Compute | Initial read |
|-----|---------|---------|--------------|
| [irt-reanalysis.md](irt-reanalysis.md) | Track B — Psychometric reanalysis of DataDecide | T0 | Strongest standalone bet |
| [trajectory-movement.md](trajectory-movement.md) | Track A — Drift/diffusion in eval trajectories | T0 | Strong; gated on checkpoint spacing |

Track C (dataset featurization) is [recipe-featurization.md](recipe-featurization.md),
listed in the table above; its idea map is [../dataset-analysis-idea-map.md](../dataset-analysis-idea-map.md).

Track D (annealing-confound correction) is merged into Project A,
[a-annealed-readouts.md](a-annealed-readouts.md): its MPL correction is A5/A2, its
merging route is A1, its ranking-stability analysis is A6, and its durable-movement
operator is A-opt-7.

Track E (token/item-level movement) is merged into Project D,
[d-token-level-decomposition.md](d-token-level-decomposition.md), as its Stage 1
(observational) half; Project D's original core is Stage 2 (causal).

Track F (FLAME-MoE routing dynamics) is no longer standalone: its routing-flip
core (survey, ingest, F1, F3, commitment timing, routing-vs-eval, scale ladder)
is a follow-up inside [trajectory-movement.md](trajectory-movement.md), and F2
(flips by token entropy) is a Stage 1 optional direction in
[d-token-level-decomposition.md](d-token-level-decomposition.md). Each restates its own
prerequisites.

Compute tiers: **T0** = analysis of published tables only; **T1** = forward
passes with existing checkpoints; **T1+** = checkpoint merging plus re-running
evals. Nothing here trains a model.

Resolved gate checks and open questions (with the code used to answer them)
are logged in [open-questions-answered.md](open-questions-answered.md).

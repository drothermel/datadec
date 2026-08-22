# Potential projects — DataDecide

One document per candidate workshop-paper project. Each is written to stand alone so it can be
evaluated on its own merits: shared infrastructure is restated in every document that needs it
rather than factored out, and the only cross-document mentions are short coordination notes
("X specifies the same runner; reuse it if it exists"). IDs inside each document carry that
document's prefix (`ANN-1`, `TRJ-opt-2`, …); each document's header maps its IDs back to the
synthesis inventory it was derived from.

Each document has the same three parts:

1. **What the project involves** — the core experiment plus the optional directions.
2. **Doability and impact** — an overall doability take, then per-direction workshop-paper impact.
3. **Infrastructure build sequence** — what to build, in what order, if we proceed.

| Project | Core question | Training required | Compute tier | Standalone paper? |
|---------|---------------|-------------------|--------------|-------------------|
| [Annealed readouts](annealed-readouts.md) (`ANN`) | How much of DataDecide's reported ranking is a cosine-schedule artifact, and can it be corrected for the cost of evals? | short decay branches | T0 core; T1+/T2 for proxies and branches | **Yes — strongest candidate** |
| [WSD retrain suite](wsd-suite.md) (`WSD`) | What does a DataDecide-subset with a proper stable phase + decay branches enable, and is it worth keeping the cluster warm for? | full retrain (subset) | T3 | Resource paper; better as an enabler |
| [Loss-landscape geometry](landscape-geometry.md) (`GEO`) | Are cross-recipe metric comparisons well-defined, and does basin membership predict when recipe effects hold? | none | T1 | Yes, evals-only |
| [Token-level movement](token-movement.md) (`TOK`) | Stage 1: where does movement between checkpoints live, and does it concentrate on high-entropy tokens? Stage 2: which tokens respond to LR decay, does that track epistemic uncertainty, and do recipes differ in how tokens migrate? | Stage 1 none; Stage 2 decay branches | T0/T1 then T2 | Stage 1 standalone if its headline holds; Stage 2 highest ceiling |
| [Trajectory drift/diffusion](trajectory-statistics.md) (`TRJ`) | What lives inside the checkpoint-to-checkpoint "noise" term: directional drift vs. mean-reverting diffusion, and does diffusion track the learning rate? | none | T0 | Strong; checkpoint spacing confirmed adequate |
| [IRT reanalysis](irt-reanalysis.md) (`IRT`) | Do recipes differ along one latent axis or many, and which items behave differently across recipes at matched ability? | none | T0 | Strongest standalone bet |
| [Recipe featurization](recipe-featurization.md) (`REC`) | What is actually in the DataDecide recipes, and which measurable data properties explain which task-level differences? | none | T0/T1 | Yes, GPU-free |

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short decay
branches from existing checkpoints; **T3** = new pretraining runs.

Source inventories: [../refs/lr-schedule-wsd-synthesis.md](../refs/lr-schedule-wsd-synthesis.md)
(ANN, WSD, GEO, TOK Stage 2), [../refs/research-trajectory-synthesis.md](../refs/research-trajectory-synthesis.md)
(TRJ, IRT, TOK Stage 1), and [../dataset-analysis-idea-map.md](../dataset-analysis-idea-map.md)
(REC).

Resolved gate checks and open questions (with the code used to answer them) are logged in
[../open-questions-answered.md](../open-questions-answered.md). The recipe-featurization
literature review (plan and process) lives in [../litreview/](../litreview/).

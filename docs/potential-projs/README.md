# Potential projects

One document per candidate workshop-paper project, derived from
[../refs/research-trajectory-synthesis.md](../refs/research-trajectory-synthesis.md).
Each is written to stand alone so projects can be compared and chosen
independently; shared infrastructure is restated per document rather than
factored out. Idea IDs (`A1`, `B3`, …) match the synthesis.

| Doc | Project | Compute | Initial read |
|-----|---------|---------|--------------|
| [irt-reanalysis.md](irt-reanalysis.md) | Track B — Psychometric reanalysis of DataDecide | T0 | Strongest standalone bet |
| [trajectory-movement.md](trajectory-movement.md) | Track A — Drift/diffusion in eval trajectories | T0 | Strong; gated on checkpoint spacing |
| [schedule-confound.md](schedule-confound.md) | Track D — Annealing-confound correction | T0 / T1+ | Good section; standalone only if merging works |
| [token-item-movement.md](token-item-movement.md) | Track E — Token/item-level movement | T0 / T1 | Best single figure; needs inference harness |
| [moe-routing.md](moe-routing.md) | Track F — FLAME-MoE routing dynamics | T0 after ingest | Novel but decoupled from DataDecide |

Track C (dataset featurization) is considered separately and intentionally
omitted here.

Compute tiers: **T0** = analysis of published tables only; **T1** = forward
passes with existing checkpoints; **T1+** = checkpoint merging plus re-running
evals. Nothing here trains a model.

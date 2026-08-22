# Track D — Correcting the annealing confound in DataDecide evals

**One-line pitch.** DataDecide models were trained with cosine schedules, so
every intermediate checkpoint is evaluated mid-schedule with high residual
learning rate. In river-valley terms the evals measure "position along the
river plus current height up the wall," and the wall term is schedule
artifact. Two routes to an annealed readout without retraining: an analytic
correction from the multi-power law on loss curves (T0), and checkpoint
merging as an annealing proxy with re-evaluation (T1+). Either lets us ask
how much of each recipe's apparent level and ranking is schedule artifact.

**Compute tier.** D1/D4 are T0 (loss curves + LR schedule already in
`processed/scaling-law/checkpoint-losses.parquet` and `lr_at_step` /
`cumulative_lr` on every checkpoint row). D2/D3 are T1+ (load HF
checkpoints, merge, run evals).

## 1. What the project involves

### Core (T0)

- **D1 — Multi-power-law correction.** Fit the multi-power law (loss as a
  power law in cumulative LR plus decay-induced drop terms) to each run's
  loss curve and schedule. For each checkpoint, predict the loss a
  hypothetical short decay would reach. Report corrected loss curves per
  recipe × scale × seed and the size of the correction as a function of
  position in the schedule.
- **D4 — Does the confound cancel?** DataDecide observed that
  intermediate-checkpoint *decisions* match compute-equivalent final
  checkpoints. Using D1, quantify where rankings are preserved under
  correction and where they flip; separate "levels distorted, ranks
  preserved" from "ranks distorted." Caveat from the source discussion: the
  decay phase itself makes progress along the river, so the correction is
  not a pure reveal; branch length is a parameter to sweep analytically.

### Optional directions (T1+)

- **D2 — Checkpoint merging as annealing proxy.** WSM-style merging of a
  sliding window of recent checkpoints with weights from an emulated decay
  curve. Established on stable-phase (WSD) checkpoints; the open question is
  whether it works on *cosine* mid-run checkpoints where LR varies inside
  the merge window. Validate against (a) D1's predicted loss drop and (b)
  each run's own fully decayed final checkpoint. If it works, retrofits
  annealed downstream evals onto all of DataDecide for eval cost.
- **D3 — Durable-movement operator.** Define durable movement as change that
  persists under the schedule-neutralizing transform: compare merged(t) vs.
  merged(t+k). Decomposes Signal-and-Noise "noise" into measurement noise,
  wall oscillation, and unresolved drift. Requires D2 plus per-token or
  per-item comparison between checkpoints.

## 2. Doability and impact

**Doability: D1/D4 high, D2/D3 medium.**

- D1 needs loss curves at reasonable resolution for every run; verify the
  scaling-law checkpoint-loss table covers all recipes and seeds or only a
  subset. The MPL was fit on runs with explicit schedules; fitting it to
  cosine runs is in its stated scope but its extrapolation to "hypothetical
  decay from here" on these specific runs is unvalidated. A held-out check
  (predict the final decayed loss from mid-run checkpoints) is mandatory.
- D2 requires a checkpoint loader, a merge implementation, and an
  OLMES-equivalent eval harness; the eval cost is one eval per merged
  checkpoint, so scope to a subset of recipes and steps. The method itself
  is unproven on cosine checkpoints, which is both the risk and the
  contribution.

**Impact per direction:**

| Direction | Impact | Why |
|-----------|--------|-----|
| D1 MPL correction | Medium | Useful, loss-only, and the MPL is someone else's method; a figure rather than a headline. |
| D4 cancellation analysis | **Medium-high** | Directly answers "how much should anyone trust intermediate-checkpoint rankings," which every DataDecide user cares about. Standalone short paper if the answer is surprising. |
| D2 merging on cosine | **High if it works** | A new, cheap way to get annealed evals from existing suites; clean negative result is also publishable but less exciting. |
| D3 durable movement | Medium-high | Conceptually strong; depends entirely on D2 and on a token/item-level harness. |

**Likely paper shape.** As a standalone workshop paper this needs D2: "D1 +
D4" alone is a strong *section* in a trajectory or IRT paper rather than a
paper. With D2 validated, "annealed evals for DataDecide without training"
is a clear methods contribution with D4 as the motivating analysis.

## 3. Infrastructure sequence

1. **Loss-curve accessor.** Ordered loss + `lr_at_step` + `cumulative_lr`
   series per run from the scaling-law checkpoint-loss table, with coverage
   report across recipes × scales × seeds.
2. **MPL fitting module (D1).** Parametric fit with uncertainty; held-out
   validation predicting final decayed loss from truncated curves;
   hypothetical-decay prediction at each checkpoint for a sweep of decay
   lengths.
3. **Ranking-stability analysis (D4).** Pairwise recipe decisions under raw
   vs. corrected loss, per scale and per position in schedule; flip rates
   and their confidence under the seed noise.
4. **Checkpoint loader (T1+).** Pull DataDecide HF checkpoints for a chosen
   subset; cache locally.
5. **Merge module (D2).** Sliding-window weighted average with emulated
   decay weights; expert-agnostic (dense models only).
6. **Eval harness.** OLMES-equivalent task runner producing the same table
   schema as the processed OLMES tables so merged-model results slot into
   existing accessors.
7. **Validation + D3.** Compare merged vs. D1 predictions vs. final
   checkpoints; then merged(t) vs. merged(t+k) comparisons.

Steps 1–3 are cheap and should be done regardless of whether the T1+ half
is pursued; they also tell you whether the T1+ half is worth it.

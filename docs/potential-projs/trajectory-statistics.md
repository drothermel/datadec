# Trajectory drift/diffusion — in eval trajectories (the Signal-and-Noise dual)

**One-line pitch.** Signal-and-Noise (Heineman et al.) treats
checkpoint-to-checkpoint variability as a nuisance term to suppress. Invert
it: treat inter-checkpoint change as the object and decompose it into
directional, cumulative *drift* (learning) and mean-reverting *diffusion*
(oscillation). The result is a "movement SNR" table over benchmarks,
recipes, and scales, plus a zero-training test of the river-valley
hypothesis using the cosine schedule as a within-run LR sweep.

**Compute tier.** T0. Inputs are the aggregate OLMES and perplexity
trajectories already in `data/processed/`, each row carrying `lr_at_step`,
`cumulative_lr`, tokens, and FLOPs.

## 1. What the project involves

### Core

- **TRJ-6 — Noise floor.** Before measuring movement, estimate the variance
  that is not movement: (a) pooled seed variance across 25 recipes × 3 seeds
  at fixed scale, with a heteroscedasticity test across recipes; (b)
  trajectory-as-replicate (late-window variance within a run), corrected for
  within-window drift; (c) item bootstrap for benchmark-composition
  uncertainty (uses per-instance tables). Output: per-metric, per-scale
  noise floors that every later claim is tested against.
- **TRJ-1 — Drift/diffusion decomposition.** For each (benchmark, metric,
  recipe, scale, seed) series: increment autocorrelation, sign-consistency
  of increments, variance-vs-lag scaling (diffusion ∝ lag, drift ∝ lag²).
  Fit a simple drift+diffusion model (e.g. local linear trend with AR(1)
  noise, or Ornstein–Uhlenbeck around a trend) and report drift rate,
  diffusion scale, and their ratio. Apply to accuracy, likelihood margins,
  and perplexity metrics.
- **TRJ-4 — Re-derive Signal-and-Noise.** Predict and confirm that continuous
  metrics have a higher drift-to-diffusion ratio than accuracy, and that
  "filter noisy subtasks" is recovered as "drop low-ratio tasks." This is the
  sanity check that the decomposition measures what it claims.

### Optional directions

- **TRJ-2 — River-valley LR test.** Diffusion magnitude should scale with the
  current learning rate (wall oscillation) while drift should not (progress
  along the river). The cosine schedule supplies a monotone LR sweep inside
  every run; regress per-window diffusion and drift on `lr_at_step`.
- **TRJ-3 — Recipe signatures at matched loss.** Pair checkpoints across
  recipes at equal loss and compare drift/diffusion signatures. Any
  benchmark where recipes differ at matched loss is "pretraining shapes
  models beyond final performance" in public data.
- **TRJ-5 — Resolution transfer.** If DataDecide's checkpoint spacing is too
  coarse to separate diffusion from drift, fit the diffusion component on
  the denser OLMo trajectories in the Signal-and-Noise release and transfer
  to DataDecide's grid.
- **TRJ-7 — Scale ladder.** Same decomposition across the DataDecide size
  ladder; does the drift-to-diffusion ratio improve with scale, and does it
  do so uniformly across tasks?

### Follow-up: routing dynamics as a movement channel (FLAME-MoE)

*Moved from the former standalone MoE-routing doc (`moe-routing.md`). Its F2
direction (flips by token entropy) is TOK-obs-5 in Token-level movement, `token-movement.md`, Stage 1. *

**One-line pitch.** Dense-model movement between checkpoints is continuous
drift that has to be extracted from KL or CKA. Mixture-of-experts models
expose a *categorical* channel — per-token expert assignments — and
FLAME-MoE (38M–1.7B active, 64 experts, top-8) releases checkpoints, routing
logs, and evals. Apply the drift/diffusion lens to routing flips: flips that
revert are wall oscillation, flips that persist are river movement, and
per-layer saturation curves are cumulative-commitment plots. A dense control
from DataDecide at matched active parameters separates "MoE" from "small."

**Compute tier.** T0 after ingest, if the released routing logs cover the
checkpoints and tokens needed. T1 if routing must be recomputed from
checkpoints over a probe corpus.

Core:

- **Ingest FLAME-MoE artifacts.** Routing logs (per token, per layer, top-k
  expert ids across checkpoints), eval results, checkpoint metadata and
  schedules. Verify format and coverage first; this determines the tier.
- **TRJ-moe-1 — Routing-flip drift/diffusion.** For each layer and checkpoint pair,
  compute per-token assignment flip rates; separate reverting flips (t →
  t+1 → back at t+2) from persistent ones; compute router saturation
  (overlap of top-k at step t with top-k at the final checkpoint) per layer
  over training. Fit the same drift/diffusion decomposition used for dense
  eval trajectories.
- **TRJ-moe-3 — Dense control ladder.** Run the dense drift/diffusion decomposition
  on DataDecide models at matched active parameter counts so each MoE
  finding has an "is this MoE or just small models" comparison.

Optional:

- **Commitment timing.** Per-layer saturation timestamps as a "commitment
  clock": deeper layers are reported to saturate faster; quantify, and
  relate to the schedule (`lr_at_step` equivalent for FLAME-MoE runs).
- **Routing vs. eval movement.** Does routing-flip mass predict eval-metric
  movement between the same checkpoints, and does it do so better than the
  dense proxies?
- **Scale ladder.** Repeat across the seven FLAME-MoE sizes.

## 2. Doability and impact

**Doability: high, with one gate.** The gate is temporal resolution: if
saves are thousands of steps apart, adjacent-checkpoint increments may be
drift-dominated and the diffusion estimate degenerate. Check spacing per
scale first; TRJ-5 is the fallback and adds an ingest. Everything else is
standard time-series estimation on small series, and the noise-floor
module is reusable by any other project.

Secondary risks: (i) three seeds is thin for per-recipe variance — pool and
state the limitation; (ii) drift is not constant over a cosine run, so the
model must allow a time-varying trend (windowed fits).

**Routing follow-up doability: medium, dominated by ingest uncertainty.**
Everything hinges on
what the released routing logs actually contain: which checkpoints, how many
tokens, whether token identities are recoverable (needed for TOK-obs-5 and for
per-token flip tracking across checkpoints rather than aggregate
histograms). If logs are aggregate-only, per-token flip tracking requires
recomputing routing from checkpoints (T1, and a new model-loading path
distinct from DataDecide's). Also a separate suite with its own training
recipe and data, so it does not share DataDecide's recipe axis; the
"recipe" question cannot be asked here.

**Impact per direction:**

| Direction | Impact | Why |
|-----------|--------|-----|
| TRJ-6 noise floor | **Medium-high** | Nobody publishes it; every later DataDecide paper would cite it. Not a headline alone. |
| TRJ-1 movement-SNR table | **High** | The central artifact; extends a NeurIPS 2025 framework from the same group with a new marginal. |
| TRJ-4 re-derivation | Medium (supporting) | Validation; reviewers expect it. |
| TRJ-2 LR test | **High** | A zero-training river-valley test is a clean, quotable figure; risk that LR and training progress are confounded within a single cosine run (both monotone), mitigated by comparing across scales with different schedule lengths. |
| TRJ-3 matched-loss signatures | **High if positive** | Direct thesis demonstration; risk of null at these scales. |
| TRJ-5 resolution transfer | Low (enabling) | Only matters if the gate fails. |
| TRJ-7 scale ladder | Medium | Natural figure, limited novelty. |
| *Routing follow-up:* TRJ-moe-1 routing drift/diffusion + saturation | **Medium-high** | Novel framing of existing logs; router saturation is known, reverting-vs-persistent decomposition is not. |
| *Routing follow-up:* TRJ-moe-3 dense control | Medium (supporting) | Required for credibility; not a result alone. |
| *Routing follow-up:* Commitment timing | Medium | Largely confirms published observations with better statistics. |
| *Routing follow-up:* Routing vs. eval movement | Medium-high | Practical: routing as a cheap, high-signal movement detector. |
| *Routing follow-up:* Scale ladder | Medium | Good figure, limited novelty. |

**Likely paper shape.** TRJ-6 + TRJ-1 + TRJ-4 as the core ("what lives inside the
noise term"), TRJ-2 as the headline figure, TRJ-3 as the high-variance bonus.

**Routing follow-up paper shape.** TRJ-moe-1 + TRJ-moe-3 as the core, TOK-obs-5 (Token-level movement, Stage 1) as the
headline if the logs support it. A real workshop paper, but one that stands
apart from the DataDecide line: it shares methodology with the trajectory
project, not data or story. Worth doing if the drift/diffusion machinery
already exists and the ingest proves cheap; otherwise defer.

## 3. Infrastructure sequence

1. **Trajectory accessor.** Thin view over the processed OLMES and PPL
   tables returning ordered series per (recipe, scale, seed, task, metric)
   with step, tokens, FLOPs, `lr_at_step`. Assert monotone steps and report
   spacing statistics per scale (the gate check).
2. **Noise-floor module (TRJ-6).** Pooled variance, heteroscedasticity test,
   windowed replicate estimate, item bootstrap over per-instance tables.
   Tested on synthetic series with known variance.
3. **Decomposition module (TRJ-1).** Increment statistics, variance-vs-lag
   fits, windowed drift+diffusion model; returns per-series parameters with
   uncertainty. Tested on simulated drift+OU series.
4. **Aggregation and SNR table (TRJ-1, TRJ-4).** Per benchmark × metric × scale
   summaries; comparison of metric families.
5. **LR regression (TRJ-2).** Per-window diffusion/drift vs. `lr_at_step`,
   across scales.
6. **Matched-loss pairing utility (TRJ-3).** Nearest-loss checkpoint per
   recipe × seed with tolerance reporting; signature comparison.
7. **Optional: Signal-and-Noise ingest (TRJ-5)** following the repo's
   download → preprocess → typed parquet pattern, scoped to the OLMo dense
   trajectories.

Steps 1–4 form the minimum paper; 5 and 6 are independent add-ons.

Routing follow-up sequence (after the steps above):

1. **Artifact survey.** Inspect FLAME-MoE release: routing-log schema,
   checkpoint coverage, token recoverability, eval table format. Decide
   T0 vs. T1 from this alone.
2. **Ingest.** Download → preprocess → typed parquet following the repo's
   existing pattern; routing logs as a long table (checkpoint, layer, token
   id/position, expert ids); evals into the same trajectory schema used for
   DataDecide so the dense accessors work unchanged.
3. **Flip and saturation metrics (TRJ-moe-1).** Per-layer flip rates with
   reverting/persistent split; saturation vs. final checkpoint; feed into
   the drift/diffusion decomposition module (shared with the trajectory
   project; build it there if it does not exist yet).
4. **Dense control (TRJ-moe-3).** Run the same decomposition on DataDecide models
   at matched active parameters using the existing processed tables.
5. **Optional: routing recomputation path** (MoE checkpoint loader +
   forward hooks) only if the logs are insufficient.

---

## 4. External assessments

Dated, attributed notes from external review conversations, recorded for consolidation — not
decisions. Only notes about this project are kept here.

### 2026-08-21 — two "top-N by workshop-paper likelihood × speed" rankings

- **Reviewer 1, ranked #1 of 3.** "Pure T0 on data you already have parsed. The methods are
  standard small-series time-series estimation. Critically, it's robust to outcome: the TRJ-6
  noise floor plus the movement-SNR table (TRJ-1) plus the Signal-and-Noise re-derivation
  (TRJ-4) form a paper even if the exciting bits (TRJ-2's LR test, TRJ-3's matched-loss
  signatures) come up null. The one gate — checkpoint spacing — can be checked in an afternoon
  before you commit. And the noise-floor module is the single most reusable artifact in the
  whole portfolio; every other project's credibility section needs it."
- **Reviewer 2, ranked #2 of 5.** "TRJ-6 + TRJ-1 + TRJ-4, TRJ-2 as headline. Also pure T0,
  robust to outcome, and the noise-floor module (TRJ-6) is a public good every other project
  cites. One gate to check on day one: checkpoint spacing. The TRJ-2 river-valley LR test is
  the quotable figure if it works, but the paper stands without it."
- **Downstream use (reviewer 1).** Results here "will tell you which recipes and steps are
  most schedule-sensitive, making the branch grid [for annealed readouts] smaller and better
  targeted."
- **Pairing (reviewer 2).** The annealed-readouts T0 half (MPL correction + ranking-flip
  analysis) could be paired with this paper, since "they share the trajectory accessor and the
  noise floor."
- Checkpoint-spacing gate: resolved (see `docs/open-questions-answered.md`).

- **Overlap (reviewer 1).** TRJ-3 matched-loss signatures and IRT recipe-DIF "are close enough
  that reviewers will ask why you need both" — decide which instrument makes the
  beyond-final-performance claim first.
- **n = 25 (reviewer 2).** TRJ-3 "currently treat[s] 25 recipes as exchangeable"; prefer
  within-family comparisons along a measured dose (the family-contrast framing from recipe
  featurization).
- **Feasibility role (reviewer 1).** "The noise-floor module isn't just reusable hygiene; it's
  the thing that determines whether the whole program has publishable effect sizes. That's
  another argument for [this project] first: it's partly a feasibility study for everything
  else."

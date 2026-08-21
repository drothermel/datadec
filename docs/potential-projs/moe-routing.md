# Track F — Routing dynamics as a movement channel (FLAME-MoE)

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

## 1. What the project involves

### Core

- **Ingest FLAME-MoE artifacts.** Routing logs (per token, per layer, top-k
  expert ids across checkpoints), eval results, checkpoint metadata and
  schedules. Verify format and coverage first; this determines the tier.
- **F1 — Routing-flip drift/diffusion.** For each layer and checkpoint pair,
  compute per-token assignment flip rates; separate reverting flips (t →
  t+1 → back at t+2) from persistent ones; compute router saturation
  (overlap of top-k at step t with top-k at the final checkpoint) per layer
  over training. Fit the same drift/diffusion decomposition used for dense
  eval trajectories.
- **F3 — Dense control ladder.** Run the dense drift/diffusion decomposition
  on DataDecide models at matched active parameter counts so each MoE
  finding has an "is this MoE or just small models" comparison.

### Optional directions

- **F2 — Flips by token entropy.** Bucket tokens by reference-model entropy
  and test whether high-entropy tokens keep flipping experts long after
  low-entropy tokens' routes have frozen. Needs a reference-model scoring
  pass over the logged tokens (T1).
- **Commitment timing.** Per-layer saturation timestamps as a "commitment
  clock": deeper layers are reported to saturate faster; quantify, and
  relate to the schedule (`lr_at_step` equivalent for FLAME-MoE runs).
- **Routing vs. eval movement.** Does routing-flip mass predict eval-metric
  movement between the same checkpoints, and does it do so better than the
  dense proxies?
- **Scale ladder.** Repeat across the seven FLAME-MoE sizes.

## 2. Doability and impact

**Doability: medium, dominated by ingest uncertainty.** Everything hinges on
what the released routing logs actually contain: which checkpoints, how many
tokens, whether token identities are recoverable (needed for F2 and for
per-token flip tracking across checkpoints rather than aggregate
histograms). If logs are aggregate-only, per-token flip tracking requires
recomputing routing from checkpoints (T1, and a new model-loading path
distinct from DataDecide's). Also a separate suite with its own training
recipe and data, so it does not share DataDecide's recipe axis; the
"recipe" question cannot be asked here.

**Impact per direction:**

| Direction | Impact | Why |
|-----------|--------|-----|
| F1 routing drift/diffusion + saturation | **Medium-high** | Novel framing of existing logs; router saturation is known, reverting-vs-persistent decomposition is not. |
| F3 dense control | Medium (supporting) | Required for credibility; not a result alone. |
| F2 flips by entropy | **High if positive** | Sharp, unasked question; the MoE counterpart to the token-level movement project's headline figure. |
| Commitment timing | Medium | Largely confirms published observations with better statistics. |
| Routing vs. eval movement | Medium-high | Practical: routing as a cheap, high-signal movement detector. |
| Scale ladder | Medium | Good figure, limited novelty. |

**Likely paper shape.** F1 + F3 as the core, F2 as the headline if the logs
support it. A real workshop paper, but one that stands apart from the
DataDecide line: it shares methodology with the trajectory project, not
data or story. Worth doing if the drift/diffusion machinery already exists
and the ingest proves cheap; otherwise defer.

## 3. Infrastructure sequence

1. **Artifact survey.** Inspect FLAME-MoE release: routing-log schema,
   checkpoint coverage, token recoverability, eval table format. Decide
   T0 vs. T1 from this alone.
2. **Ingest.** Download → preprocess → typed parquet following the repo's
   existing pattern; routing logs as a long table (checkpoint, layer, token
   id/position, expert ids); evals into the same trajectory schema used for
   DataDecide so the dense accessors work unchanged.
3. **Flip and saturation metrics (F1).** Per-layer flip rates with
   reverting/persistent split; saturation vs. final checkpoint; feed into
   the drift/diffusion decomposition module (shared with the trajectory
   project; build it there if it does not exist yet).
4. **Dense control (F3).** Run the same decomposition on DataDecide models
   at matched active parameters using the existing processed tables.
5. **Optional: reference-model scoring** of logged tokens and bucket
   analysis (F2).
6. **Optional: routing recomputation path** (MoE checkpoint loader +
   forward hooks) only if the logs are insufficient.

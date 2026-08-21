# Track E — Where does movement live? Token- and item-level change between checkpoints

**One-line pitch.** Benchmark-level trajectories hide what actually changes
between adjacent checkpoints. Measure movement at finer granularity — which
benchmark items flip, which tokens' predictions move, which layers'
representations drift — and test the concrete hypothesis that mid-schedule
movement concentrates on high-entropy ("hillside") tokens while low-entropy
tokens carry the durable drift. If true, it connects Signal-and-Noise's
engineering framework to the river-valley mechanism in one figure and yields
a principled recipe for low-noise evals.

**Compute tier.** E2 (item flips) is T0 — the per-instance OLMES tables are
already parsed. E1/E3/E4 are T1: forward passes with existing DataDecide
checkpoints over a fixed probe corpus plus a reference model for token
entropy. No training.

## 1. What the project involves

### Core

- **E2 — Item flip rates (T0).** From `instances.parquet`, compute per
  adjacent-checkpoint pair the fraction of items whose correctness flips,
  split into up-flips and down-flips; compare to the net accuracy change.
  Quantifies how much flat accuracy hides item-level churn, per task, scale,
  and recipe.
- **E1 — Per-token KL between adjacent checkpoints (T1).** Fixed probe
  corpus (held-out slices matching the PPL eval corpora are a natural
  choice); compute KL(p_t ‖ p_{t+1}) per token; summarize distribution and
  its evolution over training and with `lr_at_step`.
- **Token-entropy bucketing.** Score probe tokens with a reference model's
  conditional entropy (a larger open model, or an ensemble of DataDecide
  final checkpoints to approximate the aleatoric floor). Bucket tokens by
  entropy; cache scores once.
- **E4 — KL by entropy bucket.** Test whether adjacent-checkpoint KL is
  concentrated in high-entropy buckets mid-schedule and whether low-entropy
  tokens show monotone, cumulative change. Compare across recipes at matched
  loss.

### Optional directions

- **E3 — Layerwise representation drift.** CKA or linear-map residual per
  layer between adjacent checkpoints on the probe corpus; where in depth
  movement lives and whether that changes over the schedule.
- **Durable vs. transient movement.** KL(t, t+k) as a function of k:
  transient movement cancels, durable movement accumulates. (The causal
  version via checkpoint merging belongs to the schedule-confound project.)
- **Principled low-noise eval construction.** If E4 holds, weight benchmark
  items by the determinism of their answer tokens and test whether the
  resulting aggregate has lower seed-to-seed variance than subtask filtering.
- **Cross-recipe token migration.** For the same held-out tokens, do
  different pretraining corpora show different rates at which tokens move
  from "still changing" to "settled"?

## 2. Doability and impact

**Doability: E2 high; E1/E4 medium.** E2 is a join and a groupby on existing
tables. The T1 pieces need a checkpoint loader, a probe-corpus pipeline, a
reference-model scoring pass, and careful caching; per-token outputs over
many checkpoints get large, so store per-token KL summaries by bucket rather
than raw logits. Scope by selecting a subset of recipes (e.g. extremes on
some axis) and all checkpoints at one or two scales. Choice of reference
model is the main judgment call and should be ablated (one large model vs.
DataDecide-final ensemble).

**Impact per direction:**

| Direction | Impact | Why |
|-----------|--------|-----|
| E2 item flips | Medium | Known phenomenon (churn) in a new suite; good supporting figure for any DataDecide paper, weak standalone. |
| E1 per-token KL | Medium (enabling) | Instrument, not a result. |
| E4 KL by entropy bucket | **High** | The single most interesting figure in the program if it holds; a mechanism-level link between eval noise and landscape geometry. Clean null is informative but less publishable. |
| E3 layerwise drift | Medium | Nice descriptive result; crowded literature on representation dynamics. |
| Low-noise eval construction | **High if E4 holds** | Practical deliverable the Signal-and-Noise authors would care about. |
| Cross-recipe migration | Medium-high | Thesis-relevant; needs matched-loss pairing and more checkpoints. |

**Likely paper shape.** E2 + E1 + E4 with the eval-construction corollary.
Standalone workshop paper if E4 is positive; otherwise the T0 part (E2)
folds into the trajectory or IRT project and the T1 harness is reused later.

## 3. Infrastructure sequence

1. **Item-flip analysis (E2).** Adjacent-pair joins on `instances.parquet`;
   per task × scale × recipe flip-rate tables; confidence via the pooled
   seed variance. Pure T0, do first.
2. **Probe corpus builder.** Fixed, versioned token sequences with a
   manifest; sized for one forward pass per checkpoint to be cheap.
3. **Checkpoint loader.** Pull selected DataDecide HF checkpoints; cache.
4. **Reference-model scoring.** Per-token conditional entropy from the
   chosen reference(s); cache per-token scores and bucket assignments keyed
   by probe-corpus version.
5. **Per-token KL runner (E1).** For each adjacent checkpoint pair, compute
   per-token KL and store per-bucket summaries (mean, quantiles, mass
   share), plus optional full per-token arrays for a small subset.
6. **Bucket analysis (E4).** KL share by bucket vs. step and `lr_at_step`;
   cross-recipe comparison at matched loss.
7. **Optional: CKA/linear-map drift (E3)** using hidden states from the same
   forward passes — capture activations in step 5 to avoid a second pass.
8. **Optional: weighted-eval construction** and variance comparison.

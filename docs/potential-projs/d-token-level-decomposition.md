# Project D — Token-level river/wall decomposition under LR decay

**Working title:** *Which tokens does the cooldown fix? A causal per-token measurement of
river-valley dynamics across pretraining recipes.*

**One-line pitch.** The river-valley account says deterministic tokens form the river and
uncertain tokens the walls, and that the stable phase learns the former while decay learns the
latter. The mapping has only been measured statically. Decay branches give a *causal* per-token
measurement — how much each token's loss drops under decay — and repeating it along training
yields each token's migration from decay-responsive to decay-inert. Crossing that with an
epistemic/aleatoric split tests the field's implied but unchecked hypothesis, and asking whether
recipes differ in migration dynamics for the same tokens is the recipe question.

Inventory IDs: D1–D6.

**Dependency:** Stage 2 requires decay branches with per-token logging — from Project A's A3
branches or Project B's suite. It cannot start before one of those exists, but Stage 1 and the
cheap static pieces (D1, D2, D5) can.

**Structure.** Two stages. *Stage 1 (observational; T0/T1 on existing checkpoints)* is the
former Track E doc (`token-item-movement.md`, now merged here): measure movement at token and
item granularity between released checkpoints and test the entropy-bucket hypothesis
statistically. *Stage 2 (causal; T2 on branches)* is the original Project D core. If Stage 1's
headline (E4) holds, it both de-risks and motivates Stage 2; if it does not, the branch compute
is saved.

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T2** = short decay branches.

---

## 1. What the project involves

### Stage 1 — observational core (Track E)

Benchmark-level trajectories hide what actually changes between adjacent checkpoints. Measure
movement at finer granularity — which benchmark items flip, which tokens' predictions move,
which layers' representations drift — and test the concrete hypothesis that mid-schedule
movement concentrates on high-entropy ("hillside") tokens while low-entropy tokens carry the
durable drift. If true, it connects Signal-and-Noise's engineering framework to the
river-valley mechanism in one figure and yields a principled recipe for low-noise evals.

- **E2 — Item flip rates (T0).** From `instances.parquet`, compute per adjacent-checkpoint
  pair the fraction of items whose correctness flips, split into up-flips and down-flips;
  compare to the net accuracy change. Quantifies how much flat accuracy hides item-level
  churn, per task, scale, and recipe. Per-instance tables exist for all 25 recipes and 66
  tasks at 150M–1B with 3 seeds (1 seed below 150M; see `open-questions-answered.md`).
- **E1 — Per-token KL between adjacent checkpoints (T1).** Fixed probe corpus (held-out
  slices matching the PPL eval corpora are a natural choice); compute KL(p_t ‖ p_{t+1}) per
  token; summarize distribution and its evolution over training and with `lr_at_step`.
- **Token-entropy bucketing (T1).** Score probe tokens with a reference model's conditional
  entropy (a larger open model, or an ensemble of DataDecide final checkpoints to approximate
  the aleatoric floor). Bucket tokens by entropy; cache scores once.
- **E4 — KL by entropy bucket (T1).** Test whether adjacent-checkpoint KL is concentrated in
  high-entropy buckets mid-schedule and whether low-entropy tokens show monotone, cumulative
  change. Compare across recipes at matched loss.

### Stage 1 — optional directions

- **E3 — Layerwise representation drift (T1).** CKA or linear-map residual per layer between
  adjacent checkpoints on the probe corpus; where in depth movement lives and whether that
  changes over the schedule.
- **Durable vs. transient movement (T1).** KL(t, t+k) as a function of k: transient movement
  cancels, durable movement accumulates. (The causal version via checkpoint merging belongs
  to Project A, `a-annealed-readouts.md`, A-opt-7.)
- **Principled low-noise eval construction.** If E4 holds, weight benchmark items by the
  determinism of their answer tokens and test whether the resulting aggregate has lower
  seed-to-seed variance than subtask filtering.
- **F2 — Flips by token entropy (FLAME-MoE) (T1).** Bucket tokens by reference-model entropy
  and test whether high-entropy tokens keep flipping experts long after low-entropy tokens'
  routes have frozen. Needs a reference-model scoring pass over the logged tokens.
  *Moved from the former standalone Track F doc (`moe-routing.md`); the routing-flip core
  (F1/F3) lives in Track A. Prerequisite, restated here:*
  - **Ingest FLAME-MoE artifacts.** Routing logs (per token, per layer, top-k expert ids
    across checkpoints), eval results, checkpoint metadata and schedules. Verify format and
    coverage first; this determines the tier. FLAME-MoE (38M–1.7B active, 64 experts, top-8)
    releases checkpoints, routing logs, and evals.
  - Doability is dominated by ingest uncertainty. Everything hinges on what the released
    routing logs actually contain: which checkpoints, how many tokens, whether token
    identities are recoverable (needed for F2 and for per-token flip tracking across
    checkpoints rather than aggregate histograms). If logs are aggregate-only, per-token flip
    tracking requires recomputing routing from checkpoints (T1, and a new model-loading path
    distinct from DataDecide's). Also a separate suite with its own training recipe and data,
    so it does not share DataDecide's recipe axis; the "recipe" question cannot be asked here.

### Stage 2 — causal core (T2)

1. **Fixed held-out token set.** A stratified held-out corpus (across domains, sized for
   per-token statistics) on which every checkpoint and every branch endpoint logs per-token
   loss. Shared with Projects A and B.
2. **Per-token decay-responsiveness (D3).** For each branch, the per-token loss drop from
   branch start to branch endpoint. High drop = wall token at that point in training; ~zero
   drop = already at the river.
3. **Migration over training (D4).** Repeat D3 at successive branch points to get each token's
   responsiveness trajectory: when it migrates from responsive to inert and how fast. Compare
   trajectories across recipes for the *same* held-out tokens. Observational (Stage 1)
   version: for the same held-out tokens, do different pretraining corpora show different
   rates at which tokens move from "still changing" to "settled"?
4. **Epistemic/aleatoric decomposition (D2).** Estimate a per-token aleatoric floor from an
   ensemble (DataDecide's 3 seeds, or a much larger reference model); epistemic = current
   uncertainty minus floor. Test whether decay-responsiveness tracks epistemic-but-not-aleatoric
   uncertainty, and whether recipes differ in epistemic-drainage schedules rather than
   aleatoric floors.

### Stage 2 — optional directions

- **D-opt-1: Static determinism profile (D1).** Per-token reference-model entropy on the
  held-out set and on each corpus; a dataset-level "% deterministic" statistic. Cheap; also
  the input to Track C's D4 (`recipe-featurization.md`).
- **D-opt-2: Loss-trajectory taxonomy on raw checkpoints (D5).** Rho-1-style buckets
  (persistently high/low, descending, fluctuating) across DataDecide's existing checkpoints.
  No branches needed; previews the dynamics, though it conflates wall oscillation with river
  progress.
- **D-opt-3: Domain and token-type breakdown.** Aggregate responsiveness by domain, POS,
  frequency band, and position-in-sequence to give the buckets interpretable structure.
- **D-opt-4: Bridge to post-training token regimes (D6).** Compare the wall bucket with the
  high-entropy "forking tokens" that carry most of RLVR's effect. Needs a post-training run
  (Project A's A-opt-3 or Project B's B3).
- **D-opt-5: Toy replication.** Reproduce Wen et al.'s bigram-language toy with varying
  determinism profiles and run the same branch protocol; gives a controlled sanity check of the
  measurement before interpreting real-data results.

---

## 2. Doability and impact

### Overall doability: **medium**, gated on branches

- Once branches with per-token logging exist, D3/D4 are pure analysis over logged arrays —
  cheap. The gating cost is entirely in Projects A/B.
- D2 is the delicate part: aleatoric estimation from 3 seeds is crude, and a larger reference
  model's entropy is a different quantity from the small model's aleatoric floor. Expect to
  report both and be explicit about the limitation.
- Signal risk: at 150M the per-token decay effect may be small relative to per-token noise,
  requiring aggregation over many tokens per bucket. The held-out set must be sized for this
  up front.
- The static pieces (D-opt-1, D-opt-2) can start immediately and de-risk the rest.
- Stage 1: E2 high; E1/E4 medium. E2 is a join and a groupby on existing tables. The T1
  pieces need a checkpoint loader, a probe-corpus pipeline, a reference-model scoring pass,
  and careful caching; per-token outputs over many checkpoints get large, so store per-token
  KL summaries by bucket rather than raw logits. Scope by selecting a subset of recipes (e.g.
  extremes on some axis) and all checkpoints at one or two scales. Choice of reference model
  is the main judgment call and should be ablated (one large model vs. DataDecide-final
  ensemble).

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| E2 item flips | Medium | Known phenomenon (churn) in a new suite; good supporting figure for any DataDecide paper, weak standalone. |
| E1 per-token KL | Medium (enabling) | Instrument, not a result. |
| E4 KL by entropy bucket | **High** | The single most interesting figure in the program if it holds; a mechanism-level link between eval noise and landscape geometry. Clean null is informative but less publishable. |
| E3 layerwise drift | Medium | Nice descriptive result; crowded literature on representation dynamics. |
| Low-noise eval construction | **High if E4 holds** | Practical deliverable the Signal-and-Noise authors would care about. |
| Cross-recipe migration (observational D4) | Medium-high | Thesis-relevant; needs matched-loss pairing and more checkpoints. |
| F2 flips by entropy (FLAME-MoE) | **High if positive** | Sharp, unasked question; the MoE counterpart to E4. |
| Stage 2 core (D3, D4, D2) | **High — highest ceiling in the programme** | The first causal token-level validation of the river-valley mechanism, plus a recipe-comparison result nobody has run. Strong enough for a main venue if the signal is clean. Workshop version can ship with D3/D4 alone and a partial D2. |
| D-opt-1 determinism profile | Medium | Cheap descriptive statistic; its value is as a predictor (Track C, `recipe-featurization.md`) and as a sanity check. |
| D-opt-2 raw taxonomy (D5) | Low–Medium | Descriptive; good as a warm-up figure in Project A's paper. |
| D-opt-3 breakdowns | Medium | Turns the result from a histogram into an interpretable story; low cost once D3 exists. |
| D-opt-4 RLVR bridge | High, speculative | A striking link if it holds; depends on a post-training run and on the wall bucket being well-defined. |
| D-opt-5 toy replication | Medium | Mostly credibility; reviewers will ask. Cheap. |

**Recommended scope:** Start Stage 1 (E2 first), D-opt-1 and D-opt-2 now (no dependencies).
Run the Stage 2 core as soon as Project A's first branch grid lands; add D-opt-3 and D-opt-5
for the paper. Hold D-opt-4 for a follow-up.

**Stage 1 paper shape.** E2 + E1 + E4 with the eval-construction corollary. Standalone
workshop paper if E4 is positive; otherwise the T0 part (E2) folds into the trajectory or IRT
project and the T1 harness is reused by Stage 2.

---

## 3. Infrastructure build sequence

0. **Item-flip analysis (E2).** Adjacent-pair joins on `instances.parquet`; per task × scale
   × recipe flip-rate tables; confidence via the pooled seed variance. Pure T0, do first.
1. **Held-out token set / probe corpus design.** Fixed, versioned token sequences with a
   manifest; size and stratification chosen for per-token statistics (target: enough tokens
   per domain × entropy-bucket cell to estimate mean loss drop within a tolerance), and sized
   for one forward pass per checkpoint to be cheap. Freeze it and share it with Projects A
   and B before any branch runs.
2. **Per-token loss logging in the eval harness.** Standard output for every checkpoint variant
   (shared infra; Project A step 3). Store as compact arrays keyed by (checkpoint variant,
   held-out set version).
3. **Reference-model scoring (D1/D2).** Per-token entropy from a strong reference model and
   per-token loss from each DataDecide seed on the held-out set; compute ensemble-based
   aleatoric estimates.
4. **Raw-checkpoint trajectory taxonomy (D-opt-2).** Classify tokens by loss trajectory across
   existing checkpoints from the logged arrays. Needs only steps 1–2.
4a. **Per-token KL runner (E1).** For each adjacent checkpoint pair, compute per-token KL and
   store per-bucket summaries (mean, quantiles, mass share), plus optional full per-token
   arrays for a small subset. Capture hidden states in the same forward passes for E3.
4b. **Bucket analysis (E4).** KL share by bucket vs. step and `lr_at_step`; cross-recipe
   comparison at matched loss.
4c. *(Optional)* **CKA/linear-map drift (E3)** from the activations captured in 4a;
   **weighted-eval construction** and variance comparison; **F2** (FLAME-MoE artifact survey,
   ingest as a long table of (checkpoint, layer, token id/position, expert ids),
   reference-model scoring of logged tokens, routing recomputation only if logs are
   insufficient).
5. **Branch-pair differencing (D3).** Given (branch start, branch endpoint) pairs from the
   results store, compute per-token drops and aggregate by bucket/domain/type.
6. **Migration analysis (D4).** Assemble per-token responsiveness across branch points into
   trajectories; fit migration times; compare across recipes.
7. **Decomposition tests (D2 joins).** Regress responsiveness on epistemic and aleatoric
   estimates; compare recipes' drainage schedules and floors.
8. *(Optional)* **Toy-language harness** (D-opt-5): small bigram-language generator with a
   determinism dial, trained with the same branch runner.

Steps 0–4c (Stage 1) have no dependency on branches and should be done early; steps 5–7
(Stage 2) are analysis over artifacts produced by Projects A/B.

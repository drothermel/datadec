# Token-level movement — observational (existing checkpoints) → causal (under LR decay)

**Working title:** *Which tokens does the cooldown fix? A causal per-token measurement of
river-valley dynamics across pretraining recipes.*

**One-line pitch.** The river-valley account says deterministic tokens form the river and
uncertain tokens the walls, and that the stable phase learns the former while decay learns the
latter. The mapping has only been measured statically. Decay branches give a *causal* per-token
measurement — how much each token's loss drops under decay — and repeating it along training
yields each token's migration from decay-responsive to decay-inert. Crossing that with an
epistemic/aleatoric split tests the field's implied but unchecked hypothesis, and asking whether
recipes differ in migration dynamics for the same tokens is the recipe question.

IDs: TOK-1–TOK-6 (TOK-1–TOK-6 in the LR-schedule synthesis); Stage 1 items TOK-obs-1–5 (TOK-obs-1–TOK-obs-4 and TOK-obs-5 in the published-data-analysis synthesis).

**Dependency:** Stage 2 requires decay branches with per-token logging, specified in §1
(Stage 2 core, step 0). It cannot start before those runs exist, but Stage 1 and the cheap
static pieces (TOK-1, TOK-2, TOK-5) can.

**Structure.** Two stages. *Stage 1 (observational; T0/T1 on existing checkpoints)* is the
former token/item-movement doc (`token-item-movement.md`, now merged here): measure movement at token and
item granularity between released checkpoints and test the entropy-bucket hypothesis
statistically. *Stage 2 (causal; T2 on branches)* is the original causal core. If Stage 1's
headline (TOK-obs-4) holds, it both de-risks and motivates Stage 2; if it does not, the branch compute
is saved.

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T2** = short decay branches.

---

## 1. What the project involves

### Stage 1 — observational core

Benchmark-level trajectories hide what actually changes between adjacent checkpoints. Measure
movement at finer granularity — which benchmark items flip, which tokens' predictions move,
which layers' representations drift — and test the concrete hypothesis that mid-schedule
movement concentrates on high-entropy ("hillside") tokens while low-entropy tokens carry the
durable drift. If true, it connects Signal-and-Noise's engineering framework to the
river-valley mechanism in one figure and yields a principled recipe for low-noise evals.

- **TOK-obs-2 — Item flip rates (T0).** From `instances.parquet`, compute per adjacent-checkpoint
  pair the fraction of items whose correctness flips, split into up-flips and down-flips;
  compare to the net accuracy change. Quantifies how much flat accuracy hides item-level
  churn, per task, scale, and recipe. Per-instance tables exist for all 25 recipes and 66
  tasks at 150M–1B with 3 seeds (1 seed below 150M; see `docs/open-questions-answered.md`).
- **TOK-obs-1 — Per-token KL between adjacent checkpoints (T1).** Fixed probe corpus (held-out
  slices matching the PPL eval corpora are a natural choice); compute KL(p_t ‖ p_{t+1}) per
  token; summarize distribution and its evolution over training and with `lr_at_step`.
- **Token-entropy bucketing (T1).** Score probe tokens with a reference model's conditional
  entropy (a larger open model, or an ensemble of DataDecide final checkpoints to approximate
  the aleatoric floor). Bucket tokens by entropy; cache scores once.
- **TOK-obs-4 — KL by entropy bucket (T1).** Test whether adjacent-checkpoint KL is concentrated in
  high-entropy buckets mid-schedule and whether low-entropy tokens show monotone, cumulative
  change. Compare across recipes at matched loss.

### Stage 1 — optional directions

- **TOK-obs-3 — Layerwise representation drift (T1).** CKA or linear-map residual per layer between
  adjacent checkpoints on the probe corpus; where in depth movement lives and whether that
  changes over the schedule.
- **Durable vs. transient movement (T1).** KL(t, t+k) as a function of k: transient movement
  cancels, durable movement accumulates. Causal version: compare schedule-neutralized
  (merged or branch-annealed) checkpoints at t and t+k; movement that survives the
  neutralizing transform is durable by construction.
- **Principled low-noise eval construction.** If TOK-obs-4 holds, weight benchmark items by the
  determinism of their answer tokens and test whether the resulting aggregate has lower
  seed-to-seed variance than subtask filtering.
- **TOK-obs-5 — Flips by token entropy (FLAME-MoE) (T1).** Bucket tokens by reference-model entropy
  and test whether high-entropy tokens keep flipping experts long after low-entropy tokens'
  routes have frozen. Needs a reference-model scoring pass over the logged tokens.
  *Moved from the former standalone MoE-routing doc (`moe-routing.md`); the routing-flip
  core (TRJ-moe-1/3) lives in Trajectory drift/diffusion. Prerequisite, restated here:*
  - **Ingest FLAME-MoE artifacts.** Routing logs (per token, per layer, top-k expert ids
    across checkpoints), eval results, checkpoint metadata and schedules. Verify format and
    coverage first; this determines the tier. FLAME-MoE (38M–1.7B active, 64 experts, top-8)
    releases checkpoints, routing logs, and evals.
  - Doability is dominated by ingest uncertainty. Everything hinges on what the released
    routing logs actually contain: which checkpoints, how many tokens, whether token
    identities are recoverable (needed for TOK-obs-5 and for per-token flip tracking across
    checkpoints rather than aggregate histograms). If logs are aggregate-only, per-token flip
    tracking requires recomputing routing from checkpoints (T1, and a new model-loading path
    distinct from DataDecide's). Also a separate suite with its own training recipe and data,
    so it does not share DataDecide's recipe axis; the "recipe" question cannot be asked here.

### Stage 2 — causal core (T2)

0. **Decay branches with per-token logging (T2).** For a chosen set of recipes × checkpoint
   steps × (1–2) sizes × 3 seeds, resume training from the existing DataDecide checkpoint
   with a fresh decay (linear-to-zero or 1-sqrt; ~10% of elapsed tokens as the default
   length) on the recipe's own data; log curves and per-token losses on the held-out set at
   branch start and endpoint, save endpoint weights, and run the full DataDecide eval suite
   on the endpoint. Parameterise decay shape and length from day one. These are short
   continued-training runs at 150M–300M, well within a small cluster budget. *Annealed readouts
   proposes the same branch grid for a different question; if both proceed, run the grid
   once.*
1. **Fixed held-out token set.** Choose a held-out token set once and freeze it: fixed, versioned token
   sequences with a manifest; stratified across domains and across the DataDecide leaf
   corpora; sized so that each domain × entropy-bucket cell has enough tokens to estimate
   mean per-token loss drop within a set tolerance, while keeping one forward pass per
   checkpoint cheap. Per-token loss on it is a standard output of the eval harness for every
   checkpoint variant (raw checkpoints, merged checkpoints, branch starts and endpoints),
   stored as compact arrays keyed by (checkpoint variant, held-out-set version). Branch
   endpoints also save their weights. Cheap to add now; expensive to retrofit later because
   it would mean re-running branches. *An identical spec appears in Annealed readouts, WSD retrain suite, Token-level
   movement, MoE movement, MoE recipe suite, and Functional featurization; keep them in sync.*
2. **Per-token decay-responsiveness (TOK-3).** For each branch, the per-token loss drop from
   branch start to branch endpoint. High drop = wall token at that point in training; ~zero
   drop = already at the river.
3. **Migration over training (TOK-4).** Repeat TOK-3 at successive branch points to get each token's
   responsiveness trajectory: when it migrates from responsive to inert and how fast. Compare
   trajectories across recipes for the *same* held-out tokens. Observational (Stage 1)
   version: for the same held-out tokens, do different pretraining corpora show different
   rates at which tokens move from "still changing" to "settled"?
4. **Epistemic/aleatoric decomposition (TOK-2).** Estimate a per-token aleatoric floor from an
   ensemble (DataDecide's 3 seeds, or a much larger reference model); epistemic = current
   uncertainty minus floor. Test whether decay-responsiveness tracks epistemic-but-not-aleatoric
   uncertainty, and whether recipes differ in epistemic-drainage schedules rather than
   aleatoric floors.

### Stage 2 — optional directions

- **TOK-opt-1: Static determinism profile (TOK-1).** Per-token reference-model entropy on the
  held-out set and on each corpus; a dataset-level "% deterministic" statistic. Cheap; also
  usable as a corpus-level feature.
- **TOK-opt-2: Loss-trajectory taxonomy on raw checkpoints (TOK-5).** Rho-1-style buckets
  (persistently high/low, descending, fluctuating) across DataDecide's existing checkpoints.
  No branches needed; previews the dynamics, though it conflates wall oscillation with river
  progress.
- **TOK-opt-3: Domain and token-type breakdown.** Aggregate responsiveness by domain, POS,
  frequency band, and position-in-sequence to give the buckets interpretable structure.
- **TOK-opt-4: Bridge to post-training token regimes (TOK-6).** Compare the wall bucket with the
  high-entropy "forking tokens" that carry most of RLVR's effect. Needs the earlier
  post-training protocol re-run from branch endpoints and the matching raw checkpoints.
- **TOK-opt-5: Toy replication.** Reproduce Wen et al.'s bigram-language toy with varying
  determinism profiles and run the same branch protocol; gives a controlled sanity check of the
  measurement before interpreting real-data results.

---

## 2. Doability and impact

### Overall doability: **medium**, gated on branches

- Once branches with per-token logging exist, TOK-3/TOK-4 are pure analysis over logged arrays —
  cheap. The gating cost is entirely the branch runs (Stage 2 step 0).
- TOK-2 is the delicate part: aleatoric estimation from 3 seeds is crude, and a larger reference
  model's entropy is a different quantity from the small model's aleatoric floor. Expect to
  report both and be explicit about the limitation.
- Signal risk: at 150M the per-token decay effect may be small relative to per-token noise,
  requiring aggregation over many tokens per bucket. The held-out set must be sized for this
  up front.
- The static pieces (TOK-opt-1, TOK-opt-2) can start immediately and de-risk the rest.
- Stage 1: TOK-obs-2 high; TOK-obs-1/TOK-obs-4 medium. TOK-obs-2 is a join and a groupby on existing tables. The T1
  pieces need a checkpoint loader, a probe-corpus pipeline, a reference-model scoring pass,
  and careful caching; per-token outputs over many checkpoints get large, so store per-token
  KL summaries by bucket rather than raw logits. Scope by selecting a subset of recipes (e.g.
  extremes on some axis) and all checkpoints at one or two scales. Choice of reference model
  is the main judgment call and should be ablated (one large model vs. DataDecide-final
  ensemble).

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| TOK-obs-2 item flips | Medium | Known phenomenon (churn) in a new suite; good supporting figure for any DataDecide paper, weak standalone. |
| TOK-obs-1 per-token KL | Medium (enabling) | Instrument, not a result. |
| TOK-obs-4 KL by entropy bucket | **High** | The single most interesting figure in the program if it holds; a mechanism-level link between eval noise and landscape geometry. Clean null is informative but less publishable. |
| TOK-obs-3 layerwise drift | Medium | Nice descriptive result; crowded literature on representation dynamics. |
| Low-noise eval construction | **High if TOK-obs-4 holds** | Practical deliverable the Signal-and-Noise authors would care about. |
| Cross-recipe migration (observational TOK-4) | Medium-high | Thesis-relevant; needs matched-loss pairing and more checkpoints. |
| TOK-obs-5 flips by entropy (FLAME-MoE) | **High if positive** | Sharp, unasked question; the MoE counterpart to TOK-obs-4. |
| Stage 2 core (TOK-3, TOK-4, TOK-2) | **High — highest ceiling in the programme** | The first causal token-level validation of the river-valley mechanism, plus a recipe-comparison result nobody has run. Strong enough for a main venue if the signal is clean. Workshop version can ship with TOK-3/TOK-4 alone and a partial TOK-2. |
| TOK-opt-1 determinism profile | Medium | Cheap descriptive statistic; its value is as a corpus-level predictor and as a sanity check. |
| TOK-opt-2 raw taxonomy (TOK-5) | Low–Medium | Descriptive; good as a warm-up figure. |
| TOK-opt-3 breakdowns | Medium | Turns the result from a histogram into an interpretable story; low cost once TOK-3 exists. |
| TOK-opt-4 RLVR bridge | High, speculative | A striking link if it holds; depends on a post-training run and on the wall bucket being well-defined. |
| TOK-opt-5 toy replication | Medium | Mostly credibility; reviewers will ask. Cheap. |

**Recommended scope:** Start Stage 1 (TOK-obs-2 first), TOK-opt-1 and TOK-opt-2 now (no dependencies).
Run the Stage 2 core as soon as the first branch grid lands; add TOK-opt-3 and TOK-opt-5
for the paper. Hold TOK-opt-4 for a follow-up.

**Stage 1 paper shape.** TOK-obs-2 + TOK-obs-1 + TOK-obs-4 with the eval-construction corollary. Standalone
workshop paper if TOK-obs-4 is positive; otherwise the T0 part (TOK-obs-2) folds into the trajectory or IRT
project and the T1 harness is reused by Stage 2.

---

## 3. Infrastructure build sequence

0. **Item-flip analysis (TOK-obs-2).** Adjacent-pair joins on `instances.parquet`; per task × scale
   × recipe flip-rate tables; confidence via the pooled seed variance. Pure T0, do first.
1. **Held-out token set / probe corpus design.** Choose a held-out token set once and freeze it: fixed, versioned token
   sequences with a manifest; stratified across domains and across the DataDecide leaf
   corpora; sized so that each domain × entropy-bucket cell has enough tokens to estimate
   mean per-token loss drop within a set tolerance, while keeping one forward pass per
   checkpoint cheap. Per-token loss on it is a standard output of the eval harness for every
   checkpoint variant (raw checkpoints, merged checkpoints, branch starts and endpoints),
   stored as compact arrays keyed by (checkpoint variant, held-out-set version). Branch
   endpoints also save their weights. Cheap to add now; expensive to retrofit later because
   it would mean re-running branches. *An identical spec appears in Annealed readouts, WSD retrain suite, Token-level
   movement, MoE movement, MoE recipe suite, and Functional featurization; keep them in sync.*
   Freeze it before any branch runs.
2. **Checkpoint + eval harness with per-token loss logging.** Load any (recipe, size, seed,
   step) DataDecide checkpoint; run the DataDecide task suite and perplexity evals; store
   results keyed by that tuple plus a `variant` field (`raw`, `merged:<cfg>`,
   `branch:<cfg>`), in the same table schema as the processed OLMES tables. Per-token loss
   on the held-out set is a standard output for every variant, stored as compact arrays
   keyed by (checkpoint variant, held-out-set version).
3. **Reference-model scoring (TOK-1/TOK-2).** Per-token entropy from a strong reference model and
   per-token loss from each DataDecide seed on the held-out set; compute ensemble-based
   aleatoric estimates.
4. **Raw-checkpoint trajectory taxonomy (TOK-opt-2).** Classify tokens by loss trajectory across
   existing checkpoints from the logged arrays. Needs only steps 1–2.
4a. **Per-token KL runner (TOK-obs-1).** For each adjacent checkpoint pair, compute per-token KL and
   store per-bucket summaries (mean, quantiles, mass share), plus optional full per-token
   arrays for a small subset. Capture hidden states in the same forward passes for TOK-obs-3.
4b. **Bucket analysis (TOK-obs-4).** KL share by bucket vs. step and `lr_at_step`; cross-recipe
   comparison at matched loss.
4c. *(Optional)* **CKA/linear-map drift (TOK-obs-3)** from the activations captured in 4a;
   **weighted-eval construction** and variance comparison; **TOK-obs-5** (FLAME-MoE artifact survey,
   ingest as a long table of (checkpoint, layer, token id/position, expert ids),
   reference-model scoring of logged tokens, routing recomputation only if logs are
   insufficient).
4d. **Decay-branch runner (Stage 2 step 0).** Resume a checkpoint with configurable decay
   shape/length on the recipe's own data stream; log curves and per-token losses; save
   endpoint weights; hand the endpoint to the eval harness as a `branch:<cfg>` variant.
5. **Branch-pair differencing (TOK-3).** Given (branch start, branch endpoint) pairs from the
   results store, compute per-token drops and aggregate by bucket/domain/type.
6. **Migration analysis (TOK-4).** Assemble per-token responsiveness across branch points into
   trajectories; fit migration times; compare across recipes.
7. **Decomposition tests (TOK-2 joins).** Regress responsiveness on epistemic and aleatoric
   estimates; compare recipes' drainage schedules and floors.
8. *(Optional)* **Toy-language harness** (TOK-opt-5): small bigram-language generator with a
   determinism dial, trained with the same branch runner.

Steps 0–4c (Stage 1) have no dependency on branches and should be done early; steps 5–7
(Stage 2) are analysis over artifacts produced by the branch runner (4d).

---

## 4. External assessments

Dated, attributed notes from external review conversations, recorded for consolidation — not
decisions. Only notes about this project are kept here.

### 2026-08-21 — two "top-N by workshop-paper likelihood × speed" lists

- **Ranked #5 in a top-5 list (Stage 1 only).** "The TOK-obs-2 → TOK-obs-4 ladder. TOK-obs-2
  (item flips) is a groupby on existing tables and slots into [the IRT or trajectory papers] as
  a figure. TOK-obs-4 (KL concentrated in high-entropy tokens) is the highest-impact single
  figure available at T1-light cost — one forward-pass campaign, no training. It's the only
  entry here with real null risk, which is why it's fifth, but the harness it builds
  (checkpoint loader, probe corpus, reference scorer) is exactly what [annealed readouts,
  landscape geometry, and Stage 2 here] need next."
- Neither stage is in the top-3 list.

- **Structure.** "The observational/causal pairing is your best long-game
  structure. [Stage 1] now, [Stage 2] later once the branch runner exists. If TOK-obs-4 holds
  observationally, it both de-risks and motivates the causal follow-up; if it doesn't, you
  saved yourself the branch compute. That's the cleanest dependency chain in the set, and
  worth preserving explicitly." (This is why the doc is split into two stages.)

### 2026-08-21 — on MoE routing as a per-token commitment channel

- TOK-obs-5 extended: "routing-commitment timing as a per-token version of [the migration
  analysis, TOK-4], observable from existing checkpoints with *no decay branches needed* —
  the categorical channel makes 'committed vs. still moving' directly legible instead of
  inferred from KL." If routing is mostly token-identity clustering fixed early (unverified
  claim, attributed to the OpenMoE analysis), then "*deviations* from that (context-dependent
  routing, late reassignments) mark exactly the tokens [the] entropy-bucket hypothesis cares
  about." Adjacent work on load-balance phases across OLMoE/OpenMoE checkpoints is said to be
  aggregate-level only (unverified). (Full discussion in
  `docs/potential-projs/moe-partitions.md`.)

### 2026-08-21 — on TOK-obs-5 as the MoE twin of TOK-obs-4

- TOK-obs-5 ("high-entropy tokens keep flipping after low-entropy tokens' routes freeze") "is
  the MoE twin of the dense program's single most interesting figure [TOK-obs-4]. The two
  would make each other far more credible if they land together." Caveat for any routing
  claim: "regress assignments on token ID, frequency band, and position first, and make the
  taxonomy claim only about the residual structure" — the reference-entropy scorer "is
  exactly the right covariate set for this"; and the load-balancing objective "actively
  pushes routing toward uniformity," confounding observed assignments. (Full discussion in
  `docs/potential-projs/moe-partitions.md`.)

### 2026-08-21 — on endogenous non-stationarity

- "Even under iid data, the effective distribution is data weighted by current gradient
  magnitude, so as easy/deterministic tokens saturate, the learning signal automatically
  migrates toward harder tokens. Every model runs an implicit self-curriculum; your
  Rho-1-style loss-trajectory taxonomy [TOK-opt-2] and the river/wall token migration
  [TOK-4] are measurements of exactly this. Loss-of-plasticity in 'stationary' pretraining
  stops being paradoxical under this lens: from the gradient's perspective, pretraining was
  never stationary." (Full discussion in `docs/topics/nonstationarity-accounting.md`.)

### 2026-08-21 — positions in three ranked lists (full lists in `docs/portfolio-rankings.md`)

- **6–12-month flagship list: Tier 1, #1** as the mechanism and thesis halves of "the unified
  causal program": "causal per-token decay-responsiveness [TOK-3], crossed with the
  epistemic/aleatoric split [TOK-2] and the entropy-bucket observational result [TOK-obs-4]
  — the first causal token-level test of the river/wall picture"; "cross-recipe migration
  dynamics [TOK-4]… with each corpus's determinism profile as the predictor." The "vibes"
  figure: "a heatmap of per-token decay-responsiveness vs. entropy bucket vs. training
  position, with recipes overlaid."
- **Workshop-sized list: #8** (Stage 1): "the highest-ceiling figure in the dense program…
  Eighth because it's the first entry needing a real compute campaign *and* the first with
  genuine null risk: a clean null… folds back into [the drift/diffusion paper] as a section.
  The item-flip piece [TOK-obs-2] de-risks it and ships early regardless."
- **Full-conference list: #9**, "Which Tokens Does the Cooldown Fix?" (Stage 2): "*Speed:*
  slow — gated on [the annealed-readouts] branch grid existing with per-token logging… real
  iteration before the signal is trusted. Pivot exists (descriptive taxonomy + partial
  decomposition) but is a step down. **Expected impact: high**… **Ceiling: very high** —
  clean signal here is main-venue-strong and the mechanistic anchor for the whole program."

### 2026-08-18 — prior art for TOK-opt-4 (from the Research Trajectory page)

- RLVR characterized as "predominantly support-preserving, entropy-reducing reweighting" (Wu
  & Choi, ICML 2025 AI-for-Math workshop) that "often improve[s] pass@k at small k but
  fail[s] to expand the base model's reasoning boundary at large k" (Yue et al., NeurIPS
  2025), with counterpoints (*The Invisible Leash*, arXiv 2507.14843; arXiv 2506.14245).
  These are the post-training token regimes the wall bucket would be compared against.
  Full list in `docs/topics/pretraining-to-posttraining.md`.

### 2026-08-18 — "did the model move in distribution space?" (from the Research Trajectory page)

- On the earlier "SFT did nothing" result: "'SFT did nothing' almost certainly means benchmark
  accuracy didn't move — but did the model move in distribution space? NLL on held-out
  reasoning traces, KL from the base model, calibration, sample diversity, pass@k at very
  large k are all continuous and much lower-variance than accuracy. Two possibilities, both
  publishable: either the models genuinely don't move even in likelihood space… or they *do*
  move and pretraining recipes differ in *how much*." The Stage 1 instruments (per-token
  KL, item flips) are the same measurements applied between pretraining checkpoints; the
  post-training version is the earlier project's data reread through them. Full discussion
  in `docs/topics/posttraining-experiment-design.md`.

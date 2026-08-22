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

**Dependency:** requires decay branches with per-token logging — from Project A's A3 branches
or Project B's suite. It cannot start before one of those exists, but the cheap static pieces
(D1, D2, D5) can.

---

## 1. What the project involves

### Core experiment

1. **Fixed held-out token set.** A stratified held-out corpus (across domains, sized for
   per-token statistics) on which every checkpoint and every branch endpoint logs per-token
   loss. Shared with Projects A and B.
2. **Per-token decay-responsiveness (D3).** For each branch, the per-token loss drop from
   branch start to branch endpoint. High drop = wall token at that point in training; ~zero
   drop = already at the river.
3. **Migration over training (D4).** Repeat D3 at successive branch points to get each token's
   responsiveness trajectory: when it migrates from responsive to inert and how fast. Compare
   trajectories across recipes for the *same* held-out tokens.
4. **Epistemic/aleatoric decomposition (D2).** Estimate a per-token aleatoric floor from an
   ensemble (DataDecide's 3 seeds, or a much larger reference model); epistemic = current
   uncertainty minus floor. Test whether decay-responsiveness tracks epistemic-but-not-aleatoric
   uncertainty, and whether recipes differ in epistemic-drainage schedules rather than
   aleatoric floors.

### Optional directions

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

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| Core (D3, D4, D2) | **High — highest ceiling in the programme** | The first causal token-level validation of the river-valley mechanism, plus a recipe-comparison result nobody has run. Strong enough for a main venue if the signal is clean. Workshop version can ship with D3/D4 alone and a partial D2. |
| D-opt-1 determinism profile | Medium | Cheap descriptive statistic; its value is as a predictor (Track C, `recipe-featurization.md`) and as a sanity check. |
| D-opt-2 raw taxonomy (D5) | Low–Medium | Descriptive; good as a warm-up figure in Project A's paper. |
| D-opt-3 breakdowns | Medium | Turns the result from a histogram into an interpretable story; low cost once D3 exists. |
| D-opt-4 RLVR bridge | High, speculative | A striking link if it holds; depends on a post-training run and on the wall bucket being well-defined. |
| D-opt-5 toy replication | Medium | Mostly credibility; reviewers will ask. Cheap. |

**Recommended scope:** Start D-opt-1 and D-opt-2 now (no dependencies). Run the core as soon as
Project A's first branch grid lands; add D-opt-3 and D-opt-5 for the paper. Hold D-opt-4 for a
follow-up.

---

## 3. Infrastructure build sequence

1. **Held-out token set design.** Size and stratification chosen for per-token statistics
   (target: enough tokens per domain × entropy-bucket cell to estimate mean loss drop within a
   tolerance). Freeze it and share it with Projects A and B before any branch runs.
2. **Per-token loss logging in the eval harness.** Standard output for every checkpoint variant
   (shared infra; Project A step 3). Store as compact arrays keyed by (checkpoint variant,
   held-out set version).
3. **Reference-model scoring (D1/D2).** Per-token entropy from a strong reference model and
   per-token loss from each DataDecide seed on the held-out set; compute ensemble-based
   aleatoric estimates.
4. **Raw-checkpoint trajectory taxonomy (D-opt-2).** Classify tokens by loss trajectory across
   existing checkpoints from the logged arrays. Needs only steps 1–2.
5. **Branch-pair differencing (D3).** Given (branch start, branch endpoint) pairs from the
   results store, compute per-token drops and aggregate by bucket/domain/type.
6. **Migration analysis (D4).** Assemble per-token responsiveness across branch points into
   trajectories; fit migration times; compare across recipes.
7. **Decomposition tests (D2 joins).** Regress responsiveness on epistemic and aleatoric
   estimates; compare recipes' drainage schedules and floors.
8. *(Optional)* **Toy-language harness** (D-opt-5): small bigram-language generator with a
   determinism dial, trained with the same branch runner.

Steps 1–4 have no dependency on branches and should be done early; steps 5–7 are analysis
over artifacts produced by Projects A/B.

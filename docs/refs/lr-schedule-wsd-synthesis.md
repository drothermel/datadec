---
title: 'LR Schedule / WSD — Idea Synthesis'
---

Structured extraction of the ideas in [lr-schedule-wsd-subset.md](lr-schedule-wsd-subset.md).
The goal is an inventory of everything we could try on this topic, how the pieces relate
(builds-on vs. alternative), which hypotheses each piece tests, and what infrastructure is
shared across paths — so we can pick a first move and build support for several paths at once.

Section references like `§WSD` point at the four dropdowns in the subset:
`§MPL` (Plasticity + Multi-Power Laws), `§RV` (Loss Basins and River-Valley),
`§WSD` (WSD and Annealing Effects + Dataset Features), `§TOK` (Token bucket mapping).

---

## 1. The problem statement

**The confound.** DataDecide models were trained OLMo-style with cosine schedules. Every
intermediate checkpoint — and the final one, for any budget short of the full run — sits
mid-schedule with high residual LR. In river-valley terms (`§RV`, `§WSD`) the eval on such a
checkpoint measures

> position along the river (durable progress) **+** current distance up the wall (transient,
> schedule-dependent)

and the second term is noise relative to every question we care about: recipe ranking, loss
levels, proxy-metric validity, and post-training (which is fine-tuning from a point high on the
wall).

**The central question.** How much of what DataDecide reports — and of what we measured in the
earlier post-training project — is a schedule artifact, and what does the picture look like once
the wall component is removed?

**The conceptual backbone (river-valley, `§RV`).** The stable/high-LR phase drives progress
along the river; the decay phase drives progress in the wall direction (and, per the theory,
*also* some along-river progress — so decay is not a pure "reveal"). The multi-power law's
decay-induced-loss-drop term is the phenomenological model of exactly this descent. Wen et al.
attribute the geometry to data: deterministic tokens form the river, uncertain tokens the walls.
That last claim is what makes the geometry plausibly *recipe-dependent* and connects schedules
to dataset features.

One caveat that any design must address: DataDecide found that intermediate-checkpoint
*decisions* (pairwise recipe rankings) matched compute-equivalent final checkpoints. So the
confound may partially cancel for rankings while still distorting levels and post-training.
"When does it cancel" is itself a result.

---

## 2. Idea inventory

Each idea has an ID used in the dependency map (§3) and the hypothesis table (§4).
Cost is rough: **evals-only** (no training), **small-train** (short decay branches from
existing checkpoints), **retrain** (new stable-phase runs).

### A. Retrofitting annealed readouts onto the existing DataDecide checkpoints

These are *alternatives to each other* as ways to remove the wall component without retraining.
They can also be run together and cross-validated (which is itself an experiment, A4).

| ID | Idea | Cost | Source |
|----|------|------|--------|
| **A1** | **Checkpoint merging as pseudo-annealing.** Merge a sliding window of recent checkpoints with weights from an emulated decay curve (WSM). Validated on *stable-phase* runs (MiniCPM-style WSD, Nemotron 3 uses it for mid-run readouts). The open question is whether it works on *cosine* mid-run checkpoints, where LR varies within the merge window. If it works even approximately, "annealed" evals can be retrofitted onto all of DataDecide for the cost of evals. | evals-only | `§WSD` |
| **A2** | **Multi-power-law analytic correction.** Fit the MPL (power law in cumulative LR + decay-drop terms) to each recipe's loss curve and predict the loss after a hypothetical decay at each checkpoint. Gives corrected *loss* only — no downstream metrics — but directly quantifies how much each recipe's apparent ranking is schedule artifact. | evals-only (curve fitting) | `§MPL`, `§WSD` |
| **A3** | **Short decay branches resumed from existing cosine checkpoints.** Resume a checkpoint with a fresh, short decay (e.g. 1-sqrt or linear-to-zero, ~10% of elapsed tokens) and eval the endpoint. This is the Hägele / MiniCPM protocol applied to a non-stable starting point; it is the "small extensions of training from each checkpoint" option. Branch length is a parameter, not a constant (river-valley predicts the decay also advances along the river). | small-train | `§WSD` |
| **A4** | **Cross-validate A1/A2 against A3.** Use A3 branches as ground truth on a subset of (recipe, checkpoint) pairs; measure how well A1 merges and A2 corrections reproduce them. If A1 or A2 is faithful, the cheap method scales to the whole grid. | small-train (subset) + evals | derived |

### B. New training: a WSD-branch suite

| ID | Idea | Cost | Source |
|----|------|------|--------|
| **B1** | **DataDecide-with-WSD-branches.** Retrain a subset of recipes (150M–300M is called out as affordable) with a constant-LR stable phase, then launch fixed-length decay branches at many points along the stable trajectory. This is the dataset the field's own methodology papers (Hägele et al.) say should exist and no open multi-recipe suite provides. Every downstream idea in C, D, E becomes cleaner with it; A-ideas become validation targets for it. | retrain | `§WSD` |
| **B2** | **Branch-length sweep.** Within B1, vary decay length at a fixed branch point to measure how much of the annealed gain is "reveal" (wall descent) vs. continued along-river progress. Needed to choose a canonical branch length for everything else. | retrain (part of B1) | `§WSD` |
| **B3** | **Post-training from annealed vs. un-annealed checkpoints.** Re-run the earlier post-training protocol from B1 branch endpoints (or A3 endpoints) vs. the matching stable/cosine checkpoints. Tests directly whether the "post-training did nothing" wall was a schedule artifact. | small-train + post-training | `§WSD` (implied by the confound argument) |

### C. Geometry and comparability measurements

These are *diagnostics*, not standalone experiments; they attach to checkpoints from A or B
(and to the raw DataDecide checkpoints).

| ID | Idea | Cost | Source |
|----|------|------|--------|
| **C1** | **Interpolation "river test".** Loss along the linear interpolation between two stable-phase checkpoints is convex/unimodal (valley cross-section); between two decay-phase checkpoints it is smooth and monotone. Apply to pairs of DataDecide cosine checkpoints to estimate where each sits relative to the river; apply to B1 checkpoints as a sanity check of the theory. | evals-only | `§RV` |
| **C2** | **Pairwise interpolation barrier as a validity covariate.** Log the barrier (raw and permutation-aligned) between every pair of checkpoints being compared, and report recipe comparisons (ICL-ability, task vectors, proxy metrics) *conditional on* barrier height. "Effects hold only within low-barrier pairs" and "effects hold across basins" are both findings. Nobody has connected basin tests to metric validity. | evals-only | `§RV` |
| **C3** | **Curve-collapse comparability.** "Scaling with Collapse": well-tuned runs' loss curves collapse onto a shared shape. Test whether DataDecide recipes' curves collapse, as a weight-free criterion for "traveling the same river". Cheap, uses only logged losses. | evals-only (curves) | `§RV` |
| **C4** | **River-valley visualization.** Plot the landscape in (pre-cooldown→final direction, local Adam-step direction) coordinates for a few recipes. Mostly explanatory; useful for figures and for checking C1 qualitatively. | evals-only | `§RV` |
| **C5** | **Basin-membership tests between recipes.** Linear mode connectivity (with and without Git Re-Basin alignment) between different recipes at matched loss / matched compute. Determines whether cross-recipe mechanism-level comparisons are well-defined at all. Note that re-basin works poorly early in training. | evals-only | `§RV` |

### D. Token-level decomposition (river vs. wall at the token level)

| ID | Idea | Cost | Source |
|----|------|------|--------|
| **D1** | **Static determinism profile.** Score every token of a fixed held-out set (and of each corpus) with a strong reference model's conditional entropy; characterize each dataset by its per-token entropy distribution ("% deterministic tokens" as a threshold statistic). Cheap, reference-model only. | evals-only | `§TOK`, `§WSD` |
| **D2** | **Epistemic / aleatoric decomposition.** Estimate aleatoric uncertainty per token via an ensemble (or a much larger reference model); epistemic = current-model uncertainty minus aleatoric floor. Separates the fixed data property (true hillside) from distance-not-yet-traveled. | evals-only (+ ensemble of checkpoints/seeds) | `§TOK` |
| **D3** | **Per-token decay-responsiveness from branches.** For each branch (A3 or B1), measure per-token loss drop between branch start and end on the held-out set: decay-responsive = wall, decay-inert = already at the river. This is a *causal* per-token bucket measurement nobody has run. | requires A3 or B1 + per-token logging | `§TOK` |
| **D4** | **Bucket migration over training.** Repeat D3 at successive branch points to get each token's trajectory from decay-responsive to decay-inert, the migration rate, and whether different recipes produce different migration dynamics for the *same* held-out tokens. | requires D3 at multiple branch points | `§TOK` |
| **D5** | **Rho-1-style loss-trajectory taxonomy on raw checkpoints.** Classify tokens by loss trajectory across existing DataDecide checkpoints (persistently-high, persistently-low, descending, fluctuating). A token-bucket-over-time view that needs no branches — a cheap precursor to D4, though it conflates wall oscillation with river progress. | evals-only | `§TOK` |
| **D6** | **Bridge to post-training token regimes.** Compare the wall bucket (D3) with the high-entropy "forking tokens" that carry most of RLVR's effect. Suggestive, unexplored; depends on B3 or an RL pass. | post-training | `§TOK` |

### E. Dataset featurization

| ID | Idea | Cost | Source |
|----|------|------|--------|
| **E1** | **Intrinsic corpus statistics** for the 25 DataDecide corpora: WIMBD-style (duplication, contamination, domain composition, lengths), compression ratio / entropy-law, Zipf / burstiness / type-token stats (the Chan et al. properties causally tied to ICL emergence). | data pipeline | `§WSD` |
| **E2** | **Model-mediated features**: perplexity-correlation profiles across public LLMs; RegMix / data-mixing-law style proxies. Predict well but featurize as "mixture weights over domains" and don't say what property mattered. | evals-only | `§WSD` |
| **E3** | **Dataset-similarity embeddings**: Task2Vec alignment / diversity coefficient. Known negative result: similarity alone does not explain LM performance. | evals-only | `§WSD` |
| **E4** | **Regress the DataDecide outcome table on features.** 25 corpora × (~300 pairwise decisions, per-task breakdowns) is a supervised problem nobody has run. Ask which feature family predicts outcomes, and whether intrinsic features recover model-mediated ones. Should be run on *annealed* outcomes (A/B) as well as raw ones. | analysis | `§WSD` |
| **E5** | **Determinism profile → landscape geometry.** Test whether D1 (cheap, static) predicts annealing behaviour: decay gain (A3/B1), interpolation signature (C1), per-token migration (D4). This ties the WSD suite and the featurization question into one design. | requires D1 + A3/B1 | `§WSD`, `§TOK` |

### F. Schedule-as-predictor / loss-curve functionals

| ID | Idea | Cost | Source |
|----|------|------|--------|
| **F1** | **Fit the MPL per recipe** and compare fitted parameters across recipes. If recipes differ mainly in the along-river term vs. the decay-drop term, that is a compact, schedule-aware recipe signature. Prerequisite for A2. | curve fitting | `§MPL` |
| **F2** | **"When does the confound cancel" test.** Using A2/A3/B1-corrected values, compare pairwise recipe decisions at intermediate vs. final checkpoints to DataDecide's published finding that they match. Identify which recipes/tasks flip once annealed. | analysis on A/B outputs | `§WSD` |
| **F3** | **Loss-curve features → future trainability (plasticity bridge).** Cheap training statistics (curvature, feature rank, dead units, weight norm) as predictors of how much a checkpoint gains from decay or from post-training. Lower priority; the connective-tissue idea from `§MPL`. | evals-only + diagnostics | `§MPL` |

---

## 3. How the ideas relate

### Builds-on dependencies

```
                    raw DataDecide checkpoints + loss curves
                                    │
        ┌──────────────┬────────────┼──────────────┬───────────────┐
        ▼              ▼            ▼              ▼               ▼
       A1            F1 ──► A2     C1/C3/C5       D1/D5           E1/E2/E3
   (merge)         (MPL fit/    (geometry on     (static token    (features)
                    correct)     raw ckpts)       views)              │
        │              │                                              │
        └──────┬───────┘                                              │
               ▼                                                      │
              A3  ◄── ground truth for A4 (validate A1/A2)            │
       (decay branches from cosine ckpts)                             │
               │                                                      │
               ├──► D3 ──► D4 (per-token responsiveness, migration)   │
               ├──► B3 (post-training from annealed)                  │
               └──► F2 (does the confound cancel?) ◄──────────────────┤
                                                                      │
              B1/B2 (WSD retrain suite) — cleaner replacement for A3  │
               │    as the source of branches for D3/D4/B3/F2          │
               ▼                                                      ▼
              E5 (determinism profile ─► geometry) ◄── needs D1 + branches
              E4 (features ─► annealed outcome table) ◄── needs E1-3 + A/B
```

### Alternatives (pick one, or run both as a comparison)

- **A1 vs. A2 vs. A3** — three ways to get annealed readouts from existing checkpoints. A4
  is the comparison. A3 is the most trustworthy and most expensive; A1 gives downstream
  metrics cheaply if it works on cosine; A2 gives loss only but is nearly free.
- **A3 vs. B1** — branches from cosine checkpoints vs. branches from a proper stable phase.
  B1 is the clean version; A3 is the fast version and tells us whether B1 is worth it.
- **D5 vs. D3/D4** — token buckets from raw loss trajectories vs. from decay branches. D5 is
  free but confounded; D3/D4 is the causal measurement.
- **E1 vs. E2 vs. E3** — three featurization families; E4 compares them.
- **C1 vs. C3** — weight-based vs. curve-based "same river" tests.

### Complements (strictly additive once the base exists)

- C2 and C5 attach to *any* cross-recipe comparison and make it better-posed.
- B2 is a parameter sweep inside B1, not a separate experiment.
- D2 upgrades D3/D4 from "responsive vs. inert" to "epistemic vs. aleatoric".

---

## 4. Hypotheses and the experiments that test them

| # | Hypothesis | Tested by | Prediction if true |
|---|-----------|-----------|--------------------|
| H1 | Un-annealed DataDecide evals mix durable progress with a schedule-dependent wall term large enough to matter. | A3 (or B1) vs. raw evals | Annealed-vs-raw gap is large and varies by recipe and checkpoint. |
| H2 | The confound cancels for pairwise recipe *rankings* but not for levels or post-training. | F2, B3 | Rankings mostly stable under annealing; levels shift; post-training outcomes change. |
| H3 | Checkpoint merging approximates a true anneal even on cosine mid-run checkpoints. | A4 (A1 vs. A3) | Merged-model evals track branch-endpoint evals within noise. |
| H4 | The MPL fitted on a recipe's raw curve predicts its branch-endpoint loss. | A4 (A2 vs. A3) | Small residuals; residual structure tells us where the MPL breaks. |
| H5 | The interpolation signature discriminates stable-phase from decayed checkpoints on real DataDecide runs. | C1 on raw vs. A3/B1 checkpoints | Convex/unimodal between raw checkpoints; monotone between branch endpoints. |
| H6 | Cross-recipe metric comparisons are only well-defined within a basin. | C2, C5 | Recipe effects on proxy metrics / ICL are stronger or only present in low-barrier pairs. |
| H7 | Decay-responsiveness is a per-token property that tracks epistemic-but-not-aleatoric uncertainty. | D2 + D3 | Loss drop under decay correlates with epistemic, not aleatoric, uncertainty. |
| H8 | Recipes differ in epistemic-drainage *schedules*, not aleatoric floors. | D4 across recipes | Same held-out tokens migrate at recipe-dependent rates; aleatoric estimates agree across recipes. |
| H9 | A dataset's determinism profile predicts its annealing behaviour (how much the decay reveals). | E5 | D1 statistics predict A3/B1 decay gain and C1 curvature across recipes. |
| H10 | Intrinsic corpus features predict the (annealed) DataDecide outcome table at least as well as model-mediated features. | E4 | Comparable or better held-out R² from E1 vs. E2. |
| H11 | Post-training "did nothing" partly because it started high on the wall. | B3 | Post-training gains are larger / more consistent from annealed checkpoints. |
| H12 | The tokens that form the valley walls are the tokens where RLVR does its work. | D6 | Overlap between D3 wall bucket and high-entropy forking tokens exceeds chance. |

---

## 5. Shared infrastructure

What to build so that several paths are supported at once. Roughly ordered by how many ideas
depend on it.

1. **Checkpoint + eval harness over DataDecide** — load any (recipe, size, seed, step)
   checkpoint; run the DataDecide task suite and perplexity evals; store results keyed by the
   same tuple. Needed by every A/B/C/D idea. Already partly exists in this repo's data tooling.
2. **Per-token loss logging on a fixed held-out token set** — the same tokens, every
   checkpoint, every branch endpoint, every recipe. This single artifact powers D1–D5, E5, and
   H7–H9. Design the held-out set once (sized for per-token statistics, stratified across
   domains) and never change it.
3. **Decay-branch runner** — resume any checkpoint with a configurable decay (shape, length),
   log curves and per-token losses, eval the endpoint. Serves A3 and, later, B1/B2 with the
   same code. Branch length and shape are first-class parameters.
4. **Checkpoint-merging tool** — sliding-window weighted averaging with an emulated decay
   curve (WSM). Evals-only; serves A1/A4.
5. **Loss-curve fitting** — MPL fit per recipe from logged training curves; predicted decay
   drop at arbitrary steps. Serves F1/A2/A4/F2.
6. **Interpolation tooling** — loss along linear paths between two checkpoints, optional
   permutation alignment (Git Re-Basin). Serves C1/C2/C5.
7. **Reference-model scoring** — per-token entropy from a strong reference model and from an
   ensemble of DataDecide seeds/checkpoints on the held-out set. Serves D1/D2.
8. **Corpus statistics pipeline** — WIMBD-style and compression/Zipf statistics over the 25
   corpora. Serves E1/E4. Independent of everything else; parallelizable.
9. **Stable-phase training config** — WSD recipe for a DataDecide-subset retrain (B1). Reuses
   item 3 for branches. Only needed once A3 results justify it.

---

## 6. Sequencing considerations (for discussion, not a decision)

- **Cheapest high-information first move:** A2 + F1 (curve fitting, no GPU) and C1/C3 (a
  handful of evals) answer "how big is the confound in loss terms, and does the geometry look
  river-valley-like on these runs" before any training.
- **The pivotal experiment is A3 on a small grid** (a few recipes × a few checkpoints × 1–2
  branch lengths). It simultaneously gives ground truth for A4, the first D3 measurement, the
  first B3 starting points, and the evidence needed to decide whether B1 is worth the retrain.
- **A1 rides along with A3** at essentially zero extra cost, and if H3 holds it unlocks
  annealed evals for the full grid immediately.
- **E1 is embarrassingly parallel** with everything above and has no GPU dependency.
- **B1 is the commitment decision.** Defer until A3 shows (a) the confound matters (H1) and
  (b) cosine-resumed branches are noisy or ambiguous enough that a proper stable phase is needed.
- **Build item 2 (per-token logging) before running A3**, even for the first small grid —
  re-running branches to add per-token logging later is the most likely avoidable rework.

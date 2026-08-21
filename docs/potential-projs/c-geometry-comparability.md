# Project C — Loss-landscape geometry and cross-recipe comparability

**Working title:** *When are two pretraining recipes comparable? Basin membership as a covariate
of proxy-metric validity on DataDecide.*

**One-line pitch.** Metric-level comparisons across models (proxy metrics, ICL curves, task
vectors) implicitly assume the models sit in comparable regions of the loss landscape. Nobody
has tested that assumption on a controlled multi-recipe suite. We log pairwise interpolation
barriers between every DataDecide checkpoint pair we compare, apply the river-valley
interpolation "river test", and report whether recipe effects hold within low-barrier pairs,
across basins, or not at all.

Inventory IDs: C1–C5.

---

## 1. What the project involves

### Core experiment

1. **Pairwise interpolation barriers (C2, C5).** For pairs of DataDecide checkpoints —
   same recipe across seeds, same recipe across steps, different recipes at matched compute and
   at matched loss — compute loss along the linear interpolation path, raw and after
   permutation alignment (Git Re-Basin style). Record barrier height and path shape.
2. **River test (C1).** Classify each interpolation path by shape: convex/unimodal (a valley
   cross-section — both endpoints oscillating on the walls) vs. smooth monotone (both near the
   river). Apply across checkpoint steps to see whether DataDecide's cosine checkpoints behave
   as the theory predicts as the schedule decays.
3. **Conditional recipe comparisons.** Take DataDecide's existing pairwise recipe decisions and
   proxy-metric correlations and stratify them by barrier height. Report whether recipe
   effects (on accuracy, on proxy-metric validity) are concentrated in low-barrier pairs,
   uniform, or absent.

### Optional directions

- **C-opt-1: Curve-collapse comparability (C3).** Test whether DataDecide recipes' loss curves
  collapse onto a shared shape ("Scaling with Collapse"), giving a weight-free comparability
  criterion. Compare its verdict with the barrier-based one.
- **C-opt-2: River-valley visualisation (C4).** Plot a few recipes in the (pre-cooldown→final
  direction, local Adam-step direction) coordinate system. Explanatory figure.
- **C-opt-3: Barriers on annealed variants.** Repeat the barrier measurements on Project A's
  branch endpoints or merged checkpoints. Tests whether annealing brings recipes *into* the
  same basin (barriers fall) or reveals that they were never in it.
- **C-opt-4: Feature-space connectivity.** Layerwise linear feature connectivity in activation
  space, not just loss, for pairs that look connected in loss. Sharper but more expensive.
- **C-opt-5: Seed-split timing.** Using DataDecide's 3 seeds, estimate when (if ever) sibling
  runs become linearly connected, as a proxy for basin commitment time per recipe.

---

## 2. Doability and impact

### Overall doability: **high** (evals-only), with an analysis-quality risk

- No training. Interpolation requires forward passes at, say, 10–20 points per pair on a
  modest eval set; the number of pairs is the cost driver and is fully under our control.
- Permutation alignment for transformers is the fiddliest piece; raw barriers alone are a
  valid first pass, and the literature notes re-basin helps only marginally and poorly early in
  training — which is itself worth reporting.
- Main risk is interpretive: the likely result is that *all* cross-recipe barriers are high
  (independently trained models are rarely connected without alignment), in which case the
  stratification has no low-barrier stratum across recipes and the paper's claim becomes "only
  within-recipe (seed/step) comparisons are in-basin." That is still a useful, citable finding,
  but it is a weaker story than "recipe effects are conditional on basin."
- Mitigation: include the matched-loss-across-steps comparison (same recipe, different steps),
  which will have a real spread of barriers, so the conditional analysis has variation to
  exploit regardless of the cross-recipe outcome.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| Core (C1, C2, C5) | **Medium–High** | "Metric validity requires basin membership" is an unclaimed framing; applying it to DataDecide's proxy-metric claims is concrete and self-contained. Risk of a degenerate stratification (see above). |
| C-opt-1 curve collapse | Medium | Cheap and a nice contrast (weights vs. curves); strengthens the core rather than standing alone. |
| C-opt-2 visualisation | Low | Figure-only. |
| C-opt-3 barriers on annealed variants | **High** (conditional on Project A) | "Does annealing make recipes comparable?" is a crisp question that links C to A and gives the paper a causal knob. |
| C-opt-4 feature connectivity | Medium | More expensive, more convincing; a follow-up unless the loss-space result is ambiguous. |
| C-opt-5 seed-split timing | Medium | Touches the critical-period thread; a good secondary figure and cheap given 3 seeds. |

**Recommended scope:** Core + C-opt-1 + C-opt-5 as an evals-only paper; add C-opt-3 if Project
A's branches exist by then, which would likely raise it to the strongest version.

---

## 3. Infrastructure build sequence

1. **Checkpoint + eval harness** (shared with Project A, step 2). Needed to load arbitrary
   checkpoint pairs and evaluate on a fixed eval set.
2. **Interpolation tool.** Given two checkpoints, evaluate loss (and optionally task metrics)
   at N points on the linear path; emit barrier height, path shape classification, and the raw
   curve. Batch over a pair list.
3. **Permutation alignment.** Git Re-Basin-style weight matching for the DataDecide
   architecture; plug into the interpolation tool as an optional preprocessing step. Can lag
   step 2 — raw barriers are useful immediately.
4. **Pair-selection logic.** Enumerate pairs by type (seed-seed, step-step matched loss,
   recipe-recipe matched compute, recipe-recipe matched loss) from the results store, with
   budget caps per type.
5. **Conditional-analysis layer.** Join barrier results to DataDecide's pairwise decisions
   and proxy-metric tables; stratify, bootstrap over seeds, produce the core figures.
6. *(Optional)* **Curve-collapse analysis** (C-opt-1) from logged curves only — can be done
   first, in parallel with steps 1–2, as it needs no checkpoints.
7. *(Optional)* **Variant support**: accept `merged:*` / `branch:*` checkpoints from Project A
   as endpoints (C-opt-3) — free if the results store convention is shared.

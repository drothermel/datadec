# Loss-landscape geometry — and cross-recipe comparability

**Program pillars served:** how (when are comparisons well-defined), apex (identifiability-aware comparison). (Program: `README.md` → Program.)

**Working title:** *When are two pretraining recipes comparable? Basin membership as a covariate
of proxy-metric validity on DataDecide.*

**One-line pitch.** Metric-level comparisons across models (proxy metrics, ICL curves, task
vectors) implicitly assume the models sit in comparable regions of the loss landscape. Nobody
has tested that assumption on a controlled multi-recipe suite. We log pairwise interpolation
barriers between every DataDecide checkpoint pair we compare, apply the river-valley
interpolation "river test", and report whether recipe effects hold within low-barrier pairs,
across basins, or not at all.

IDs: GEO-1–GEO-5 (GEO-1–GEO-5 in the LR-schedule synthesis).

---

## 1. What the project involves

### Core experiment

1. **Pairwise interpolation barriers (GEO-2, GEO-5).** For pairs of DataDecide checkpoints —
   same recipe across seeds, same recipe across steps, different recipes at matched compute and
   at matched loss — compute loss along the linear interpolation path, raw and after
   permutation alignment (Git Re-Basin style). Record barrier height and path shape.
2. **River test (GEO-1).** Classify each interpolation path by shape: convex/unimodal (a valley
   cross-section — both endpoints oscillating on the walls) vs. smooth monotone (both near the
   river). Apply across checkpoint steps to see whether DataDecide's cosine checkpoints behave
   as the theory predicts as the schedule decays.
3. **Conditional recipe comparisons.** Take DataDecide's existing pairwise recipe decisions and
   proxy-metric correlations and stratify them by barrier height. Report whether recipe
   effects (on accuracy, on proxy-metric validity) are concentrated in low-barrier pairs,
   uniform, or absent.

### Optional directions

- **GEO-opt-1: Curve-collapse comparability (GEO-3).** Test whether DataDecide recipes' loss curves
  collapse onto a shared shape ("Scaling with Collapse"), giving a weight-free comparability
  criterion. Compare its verdict with the barrier-based one.
- **GEO-opt-2: River-valley visualisation (GEO-4).** Plot a few recipes in the (pre-cooldown→final
  direction, local Adam-step direction) coordinate system. Explanatory figure.
- **GEO-opt-3: Barriers on annealed variants.** Repeat the barrier measurements on annealed
  variants of the checkpoints: endpoints of short decay branches resumed from existing
  checkpoints (~10% of elapsed tokens, linear-to-zero or 1-sqrt, on the recipe's own data),
  or decay-weighted sliding-window merges of preceding checkpoints. Tests whether annealing
  brings recipes *into* the same basin (barriers fall) or reveals that they were never in it.
  *Annealed readouts produces exactly these variants; if they exist, reuse them.*
- **GEO-opt-4: Feature-space connectivity.** Layerwise linear feature connectivity in activation
  space, not just loss, for pairs that look connected in loss. Sharper but more expensive.
- **GEO-opt-5: Seed-split timing.** Using DataDecide's 3 seeds, estimate when (if ever) sibling
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
| Core (GEO-1, GEO-2, GEO-5) | **Medium–High** | "Metric validity requires basin membership" is an unclaimed framing; applying it to DataDecide's proxy-metric claims is concrete and self-contained. Risk of a degenerate stratification (see above). |
| GEO-opt-1 curve collapse | Medium | Cheap and a nice contrast (weights vs. curves); strengthens the core rather than standing alone. |
| GEO-opt-2 visualisation | Low | Figure-only. |
| GEO-opt-3 barriers on annealed variants | **High** (conditional on annealed variants existing) | "Does annealing make recipes comparable?" is a crisp question that links geometry to the schedule and gives the paper a causal knob. |
| GEO-opt-4 feature connectivity | Medium | More expensive, more convincing; a follow-up unless the loss-space result is ambiguous. |
| GEO-opt-5 seed-split timing | Medium | Touches the critical-period thread; a good secondary figure and cheap given 3 seeds. |

**Recommended scope:** Core + GEO-opt-1 + GEO-opt-5 as an evals-only paper; add GEO-opt-3 if
annealed variants exist by then, which would likely raise it to the strongest version.

---

## 3. Infrastructure build sequence

1. **Checkpoint + eval harness.** Load any (recipe, size, seed, step) DataDecide checkpoint; run
   the DataDecide task suite and perplexity evals; store results keyed by that tuple plus a
   `variant` field (`raw`, `merged:<cfg>`, `branch:<cfg>`), in the same table schema as the
   processed OLMES tables so results slot into existing accessors.
   Needed to load arbitrary checkpoint pairs and evaluate on a fixed eval set. *Annealed readouts,
   WSD retrain suite, and Token-level movement specify the same harness; build it once.*
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
6. *(Optional)* **Curve-collapse analysis** (GEO-opt-1) from logged curves only — can be done
   first, in parallel with steps 1–2, as it needs no checkpoints.
7. *(Optional)* **Variant support**: accept `merged:*` / `branch:*` checkpoints as
   interpolation endpoints (GEO-opt-3) — free given the `variant` convention in step 1.

---

## 4. External assessments

Dated, attributed notes from external review conversations, recorded for consolidation — not
decisions. Only notes about this project are kept here.

### 2026-08-21 — two "top-N by workshop-paper likelihood × speed" lists

- **Alternate #5 in a top-5 list.** "If you want a lower-variance fifth slot, [the]
  raw-barrier core is the alternate — evals-only, but your own risk analysis (all
  cross-recipe barriers high → degenerate stratification) is real."
- Not in the top-3 list.

### 2026-08-21 — on stage-dependent data value

- The interpolation/barrier tooling here has a further use: testing "whether late-injected
  [data] components land the model somewhere geometrically different than early-injected
  ones. That last question, component timing → landscape position, is as far as I can tell
  completely unoccupied." (Full discussion in `docs/potential-projs/functional-featurization.md`.)

### 2026-08-21 — on short-branch landscape probes

- GEO-opt-5 (seed-split timing) is "the free observational cousin" of a twin-branch probe:
  "spawn two children from the same checkpoint with different SGD noise/data order, train
  both, measure the interpolation barrier between them… Barrier-between-siblings is a 'have
  we committed yet' statistic, and the checkpoint time at which it collapses is a commitment
  clock… the branch version makes it causal and controllable." (Unverified claims: Frankle et
  al.'s linear-mode-connectivity work trains children to completion; short-child variants
  exist but are not standardized; not run across data recipes.)
- Cheap single-checkpoint complements mentioned: a scalable critical-sharpness statistic and
  perturbation-resilience / basin-emergence measures — "useful as covariates."
- Proposed as one of four probes in a "checkpoint tomography" battery; see
  `docs/topics/staging/checkpoint-tomography.md`.

### 2026-08-21 — positions in three ranked lists (full lists in `docs/portfolio-rankings.md`)

- **6–12-month flagship list: Tier 3 (component)**: "standalone drops (the 'all
  cross-recipe barriers are high' degenerate outcome is too likely), but [GEO-opt-3] — does
  annealing collapse barriers between recipes — is a genuinely great section inside [the
  flagship], with a causal knob."
- **Workshop-sized list:** not included. **Full-conference list:** not included as a
  standalone.

### 2026-08-18 — a control to add to pair selection (from the Research Trajectory page)

- For recipe–recipe pairs: "equal loss at different token counts vs. equal tokens at
  different loss are different controls, and you'll want both." The core already
  distinguishes matched-compute from matched-loss pairs; keep both in every stratified
  analysis rather than collapsing to one.

### 2026-08-18 — origin of this project (from the Research Trajectory page)

**Danielle-flagged project seed** (the `→` note on the Notion toggle): "Treat basin
membership as a covariate of proxy-metric validity: test whether 'metrics are comparable iff
linear mode connectivity (two models are in the same basin).' Report your elicitability
comparisons *conditional on* barrier height. If recipe effects on ICL-ability only hold
within low-barrier pairs, that's a finding. If they hold across basins, that's a stronger
one."

Question posed: metrics may only be comparable between models in the same basin — is there
a way to tell whether a model is in the valley vs. climbing the mountains, and whether two
models share a basin?

- "You're comparing observables (ICL curves, task vectors, plasticity statistics) across
  models at matched loss. Loss is a scalar projection of position in parameter space, where
  two models with equal loss can sit in different valleys or in the same valley but at
  different points along the river vs. up the wall. The basin question is therefore the
  question of *when your comparisons are well-defined*."
- "There's no settled scalar measure of 'same basin' or 'on the river.' What exists is a
  toolkit of pairwise tests: interpolation barrier (raw and permutation-aligned), the
  convex-vs-monotone interpolation signature, feature-space connectivity, curve-collapse —
  each partial." (These are GEO-2/GEO-5, GEO-1, GEO-opt-4, GEO-opt-1 respectively.)
- "Nobody has connected either literature to *metric validity*: no paper says 'ICL scores /
  task vectors / plasticity statistics are comparable iff models pass test X.' For your
  design this suggests a cheap, high-value addition: log the pairwise interpolation barrier
  (with and without alignment) between every pair of checkpoints you compare, and report
  your elicitability comparisons *conditional on* barrier height."
- Precedent for the core question: Juneja et al. (ICLR 2023) — models in different
  linearly-connected basins "implement *different generalization strategies*… despite
  similar in-distribution accuracy."
- Caveats inherited from the literature: re-basin "often reduce[s] barriers only marginally
  and work[s] poorly early in training, with no unified theory of when they succeed"; the
  2026 neuron-identifiability line may eventually give a principled comparability
  criterion. Paper list in `docs/topics/reference/landscape-literature.md`.

### 2026-08-18 — GEO-opt-5 as a critical-period measurement (from the Research Trajectory page)

- The onset of linear connectivity between sibling runs (GEO-opt-5, seed-split timing) is
  proposed as one of four events — with the Fisher-trace peak, the Achille critical period,
  and induction-head / ICL emergence — that "are all claimed to live in the same early
  window — but no one has measured them *together* on one set of runs to check whether
  they're the same event." A timed-deficit sibling-seed study is staged in
  `docs/potential-projs/intervention-grid.md`; GEO-opt-5 on DataDecide's three seeds is
  its free observational version.

### 2026-08-22 — a ready test from the reinit literature pass

- Layer-wise LMC (arXiv 2307.06966) reports that middle layers own the loss barrier and
  per-layer perturbations are near-barrier-free; nobody has reset an embedding layer and
  measured the barrier back to the pre-reset solution (gap G3 in
  `docs/topics/reference/reinit-and-transfer-literature.md`). The interpolation tool here is the
  instrument; PolyPythias (arXiv 2503.09543; 50 runs, 14M–410M, ~7k checkpoints) is a
  ready substrate with seeds.

### 2026-08-18 — raw vs. aligned barriers, and functional identifiability tests (from the Research Trajectory page)

- Interpret the *difference* between raw and permutation-aligned barriers: "raw-barrier-
  high/aligned-barrier-low means 'same solution class, different parameterization'
  (benign), while aligned-barrier-high means genuine solution-class divergence (the real
  scar)." Report both and their gap, not just one.
- Weight-free complements to the interpolation tool: linear-map residuals between
  representations (Roeder, Metz & Kingma 2021) and model stitching (Lenc & Vedaldi 2015;
  Bansal, Nakkiran & Barak 2021) as ground truth, CKA as the scalable proxy (it "can be
  dominated by a few directions and disagree with stitching"). GEO-opt-4's feature-space
  connectivity is the layerwise version of this. See
  `docs/topics/reference/identifiability-literature.md`.

### 2026-08-22 — GEO-opt-6 (cross-listed from embedding-reset dynamics)

- **GEO-opt-6: Is an interface reset basin-preserving?** Reset a model's input embeddings,
  recover briefly, and measure the barrier (raw and permutation-aligned) and stitching
  residual back to the pre-reset model; body reset of matched parameter count as the
  contrast. Layer-wise LMC (arXiv 2307.06966) predicts interface resets are
  near-barrier-free. PolyPythias as substrate. Primary home: `embedding-reset-dynamics.md`
  RESET-opt-1.
## 5. Related work and positioning

*Purpose: the paper-facing synthesis — the prior-art landscape, this project's
position in it, and what each closest neighbor lacks. Unlike §4 (a dated intake
log, which grows by appending new entries **above this section**), §5 is a
current-state statement: rewrite it as understanding changes. Positioning claims
are Danielle's to make; agent-supplied literature claims anywhere in this document
are unverified leads, not established facts.*

**Status: raw material assembled from repository records (2026-08-24); positioning not
yet written.**

**Where the raw material lives:**

- `../topics/reference/landscape-literature.md` — the primary accumulator: the river-valley
  papers and their interpolation-based "river test", the linear-mode-connectivity /
  re-basin line, feature connectivity, cross-task linearity, neuron identifiability, and the
  comparability precedent.
- `../topics/reference/identifiability-literature.md` — the frame in which basin
  distinctness is residual non-identifiability, the raw-vs-aligned barrier reading, and the
  weight-free functional tests (linear-map residuals, stitching, CKA).
- §4 of this doc — the 2026-08-18 origin entry (Danielle's project seed and the posed
  question), the raw-vs-aligned interpretation entry, the pair-selection control, and the
  2026-08-22 reinit-pass entry.
- `../topics/staging/checkpoint-tomography.md` — the twin-branch probe as the causal version
  of GEO-opt-5, the decay-branch instrument, and the SGLD/LLC probe, with the record's own
  "prior-art check to do" note.
- `../topics/reference/reinit-and-transfer-literature.md` — gap G3 (interface reset and the
  barrier back to the pre-reset solution), the layer-wise-LMC prediction, and PolyPythias as
  substrate; cross-listed here as GEO-opt-6.
- `../topics/reference/critical-periods.md` — the "critical period = the window before basin
  commitment" reading that links GEO-opt-5 to `intervention-grid.md`.
- `../topics/reference/ntk-literature.md` — eNTK readouts named there as candidates "for the
  ladder / GEO" (spectrum, effective rank, kernel velocity, kernel–target alignment).
- `../portfolio-rankings.md` — GEO's placement (Tier 3 component; GEO-opt-3 folded into the
  flagship) with the stated reason.

**Starting inventory for the synthesis** (assembled at intake 2026-08-24; detail in the
dated §4 entries and the topic files):

- **River-valley / valley-position line.** Wen et al., *Understanding Warmup-Stable-Decay
  Learning Rates: A River Valley Loss Landscape View* (arXiv 2410.05192) — the canonical
  statement and the source of the interpolation signature (convex/unimodal between
  stable-phase checkpoints vs. smooth monotone between decay-phase ones), "currently the
  closest thing to a 'river test'", plus the toy-bigram validation and the ~0.39 Spearman
  correlation between token-level uncertainty and local sharpness; *Training Dynamics of the
  Cooldown Stage in WSD* (the pre-cooldown→final vs. local-Adam-step coordinates, GEO-opt-2's
  figure); *Scaling with Collapse* (arXiv 2509.25087) — curve collapse as a weight-free
  comparability criterion (GEO-opt-1); the multi-power law (Luo et al., 2503.12811) reading
  the decay-induced drop as descent from the walls to the river.
  (`landscape-literature.md`.)
- **Basin identification / mode connectivity.** Frankle et al., *Linear Mode Connectivity and
  the Lottery Ticket Hypothesis* (same-run-early-split models are linearly connected — the
  precedent for GEO-opt-5); Entezari et al. on permutation invariance; Ainsworth et al., *Git
  Re-Basin* (independently trained models connected only after permutation alignment — GEO's
  alignment step); *Unveiling LMC of Re-Basin from a Neuron Distribution Perspective*
  (re-basin "often reduce[s] barriers only marginally and work[s] poorly early in training,
  with no unified theory of when they succeed" — the source of this doc's stated analysis
  risk); *Going Beyond LMC: Layerwise Linear Feature Connectivity* (GEO-opt-4); *On the
  Emergence of Cross-Task Linearity in the Pretraining-Finetuning Paradigm* and *Model soups*
  (Wortsman et al.) as why merging works only within a basin; *Beyond Structural Symmetries:
  LMC via Neuron Identifiability* (2026). (`landscape-literature.md`.)
- **The comparability precedent named in §4.** The 2026-08-18 entry names Juneja et al.
  (ICLR 2023), *Linear Connectivity Reveals Generalization Strategies*, as the precedent for
  the core question: models in different linearly-connected basins "implement *different
  generalization strategies*… despite similar in-distribution accuracy" — recorded there as
  "the strongest existing evidence" that same-metric-value-different-basin can mean different
  mechanisms.
- **The gap statement on record** (attributed to the 2026-08-18 Research Trajectory entry,
  agent-supplied and unverified): "Nobody has connected either literature to *metric
  validity*: no paper says 'ICL scores / task vectors / plasticity statistics are comparable
  iff models pass test X.'" The same entry supplies the design consequence — log raw and
  aligned barriers for every compared pair and report comparisons conditional on barrier
  height.
- **Identifiability reading of the barrier pair.** Raw-high/aligned-low = "same solution
  class, different parameterization" (benign); aligned-high = genuine solution-class
  divergence. Weight-free complements: Roeder, Metz & Kingma 2021 (linear-map residuals);
  model stitching (Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021) as ground truth; CKA
  (Kornblith et al.) as the scalable proxy that "can be dominated by a few directions and
  disagree with stitching." (`identifiability-literature.md`, §4 entry of 2026-08-18.)
- **Reset-side and substrate items.** Layer-wise LMC (arXiv 2307.06966) — middle layers own
  the barrier, per-layer perturbations near-barrier-free — with `reinit-and-transfer-
  literature.md`'s gap G3 ("nobody has reset an interface and measured the barrier to the
  pre-reset solution") as GEO-opt-6's stated opening; PolyPythias (van der Wal et al., ICLR
  2025, arXiv 2503.09543; 50 runs, 14M–410M, ~7k checkpoints) as the many-seed substrate;
  also listed there: LMC of MoEs (2509.11348), *Landscaping LMC* (2406.16300), *The Butterfly
  Effect* (2506.13234, motivating many seeds).
- **Adjacent instruments for the same statistic** (from `checkpoint-tomography.md`): Frankle-
  style twin branches as the causal version of GEO-opt-5, with the caveats recorded there
  (children trained to completion; short-child variants not standardized; mostly pre-LLM-scale
  vision work; not run across data recipes); the devinterp/SGLD local-learning-coefficient
  probe; single-checkpoint critical-sharpness and basin-emergence statistics as covariates.
  That file also records a prior-art check still to do over the devinterp and WSD-followup
  communities' 2025–26 output.
- **Provenance caveat.** Almost all of the above entered the repo through the 2026-08-18
  Research Trajectory conversations and the 2026-08-22 reinit pass; both source files carry
  the standing header that related-work claims are unverified unless an identifier is given,
  and the only GEO-tagged row in `../litreview/citation-verification-ledger.md` is 2407.17465
  (u-µP, Danielle-supplied), still unverified.

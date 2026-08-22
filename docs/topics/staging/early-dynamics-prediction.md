# Early dynamics predict model performance — forecasting DataDecide outcomes from the first 10% of training

**DataDecide link.** This proposal runs entirely on the DataDecide suite (25 recipes × 14
sizes × 3 seeds, per-checkpoint perplexity + OLMES) — the same data every `T0`/`T1` project
in `potential-projs/` uses — so if promoted it serves the program's "how" pillar
(measurement at small scale / early time) and the "independent variable" pillar (recipe
effects). The recipe-family CV scheme below is reusable by any leave-recipe-out design on
DataDecide.

**Kind:** staging. Candidate exits: a standalone project doc (it is a complete proposal with a
draft paper), or absorption into tiny-scale measurement (`TINY`) / trajectory statistics
(`TRJ`) as a prediction arm. Gate: Danielle's decision on whether this July-2025 direction is
still live, and a check of its overlap with the published loss-to-loss / task-scaling-law
work already listed in `../reference/loss-curve-forecasting.md`.

Source: Danielle's July 2025 LaTeX draft (local copy
`../../refs/2025-07-early-dynamics-predict-model-performance.pdf`) and an external review of
it plus her Gemini refinement notes (notes not in this repo). Intake 2026-08-22. Feedback
claims (e.g. "warm-up % alone explains ~20% of variance") are **unverified**.
---

## 2025-07 — The proposal (from the draft)

**Motivation.** Scaling-law extrapolation from small models is best practice for choosing
designs but "not always accurate" (cites Li et al. 2025 *(mis)fitting*; Lourie, Hu & Cho 2025
*Scaling laws are unreliable for downstream tasks*). Alternative: use metrics from the
**first period of training** to predict later performance. Target claim: "XXX% accuracy of
predicting the performance ranking of models trained across DataDecide recipes on IID and
OOD datasets and the OLMES set of downstream tasks at the final epoch of training, using
metrics from just the first 10% of training."

**Data.** All 25 DataDecide recipes × 14 sizes (4M–1B) × 3 seeds; per-checkpoint
perplexity on 11 validation splits (wikitext-103, pile, m2d2 s2orc, ice, dolma wiki / stack
/ reddit / pes2o / common-crawl / books, c4 en) and OLMES downstream tasks (MMLU, HellaSwag,
ARC-C/E, PIQA, CSQA, SocialIQA, OpenBookQA, BoolQ, WinoGrande).

**Problem setup.** Given M training setups, train for the first S₀ steps; use those metrics
to predict performance at step S. Downstream target = CORRECT PROB (average probability of
correct continuations), chosen because DataDecide found it gives larger spread and less
noise at small scale than accuracy (cf. Schaeffer et al. on jumpy discrete metrics). Also
predict the ranking of models at S.

**Evaluation.** Absolute and relative prediction error on CORRECT PROB (following DataDecide)
and **decision accuracy** = fraction of model pairs whose predicted and actual ordering
agree (Eq. 1); TODO on rescaling to [0,1].

**Predictors.** Linear regression first; possibly GBDTs; GPs for interpretability.

**Experiments.** Generalization across recipes: leave-one-out CV at single-recipe and
recipe-family level. Generalization across scale: expanding-window validation (train on
4M–20M, validate on all larger; expand). §4.1 in-domain prediction (same task early → late;
correlation first). §4.2 all-to-all (predict MMLU from all early evaluations). §4.3 open
design choices: held-out sets (larger scales? random datasets?); what predicts what —
early→late; ppl→downstream; downstream→ppl; early ppl + downstream → OOD ppl + downstream
late; all early → all late simultaneously.

**Future work.** Predict finetune performance from pretraining + early finetune metrics
(SFT and DPO; math/code/instruction-following evals). Extended early metrics: curvature (max
Hessian eigenvalue, Hessian-trace approximation — lower curvature ↔ better optimization and
finetuning?), feature stability (less forgetting, maybe less plasticity), gradient and
activation variance (less interference during acquisition).

**Related work named.** Loss curve prediction (Brandfonbrener et al. 2024, loss-to-loss);
early-metric performance estimation for NAS (Ru et al. 2021); loss-landscape metrics.

## 2025-07 — External review of the plan (near-verbatim, condensed)

Framed as six themes for a workshop-first, conference-extension plan within a 30–50 h budget.

1. *Scope and positioning.* Add a tiny **scaling-law extrapolation baseline** (Kaplan-style
   power law fit on ≤60M models, extrapolated, scored with the same ranking metrics) — "lets
   reviewers calibrate how much your early-dynamics features buy over today's common
   practice." Drop the CNN for the workshop version; keep one interpretable (GBDT) and one
   probabilistic (sparse GP) model. One-question framing: "Can we replace 90% of training
   with 10% of training plus a predictor, at comparable accuracy?"
2. *Budget.* Feature-matrix explosion (>10⁵ rows balloons GP runtimes): subsample to ~8
   representative recipes, keep all sizes (≤ 8 × 14 × 3 ≈ 336 runs). Unequal "10%" cost
   across scales: redefine S₀ as "first 2B tokens or 10% of training, whichever is smaller."
3. *Features.* Leakage: encode progress only as a fraction of total steps; drop
   absolute-step features when the target step is fixed. Also stash the raw down-sampled
   curve (length 100) and let the GBDT decide. Add LR-schedule scalars (warm-up length %,
   initial LR).
4. *Validation and statistics.* Add a **LORO × LLSO** stress split (train on small models of
   24 recipes, predict large models of the held-out recipe). Block-bootstrap CIs over model
   pairs for ranking accuracy. Benjamini–Hochberg FDR ≤ 5% across the many metrics.
5. *Risk register.* Data-licence ambiguity in some Dolma/CC subsets (restrict release;
   list licences); GP O(n³) (GPyTorch variational sparse GP, ≤1,000 inducing points);
   hyper-parameter blow-up (fix one canonical GBDT config, don't retune per fold);
   "correlation ≠ causation" (reserve a page for mechanistic interpretation, e.g. fast
   curvature drop ↔ flat minima).
6. *Manuscript checklist.* Baseline table with scaling-law extrapolation leftmost; one
   predicted-vs-actual figure for held-out large models with GP bands; feature-importance
   bars by category; reproducibility statement + Colab; compute-budget appendix.

## Relation to existing docs

- `../reference/loss-curve-forecasting.md` holds the loss-to-loss / multi-power-law /
  task-scaling-law references this proposal positions against.
- `../../potential-projs/tiny-scale-measurement.md` (`TINY`) asks how far down the scale
  ladder decision signal survives — the early-*time* analogue of the same question is this
  proposal; natural cross-listing.
- `../../potential-projs/trajectory-statistics.md` (`TRJ`) and
  `../../potential-projs/annealed-readouts.md` (`ANN`) supply the caveat that late-training
  rankings are partly a cosine-tail artifact, which bounds what "predicting the final
  ranking" can mean.

## 2025-07 — Second review: GBDT v0 design details (near-verbatim, condensed)

One of several similar reviews Danielle requested of the same plan; recorded where it adds
to the first. Focused on the GBDT-based proof of concept (GP comparison later). The plan it
reviewed already had S₀ = min(10% of training, 2B tokens), curve-fit features (linear /
exponential / power-law, endpoints, variance), LightGBM + basic CV.

*A. Quick wins by pipeline stage.* Store relative progress w.r.t. the LR schedule (% warm-up
done, % cosine decay done) as a phase feature. Add rolling-slope features (5-point window)
and a noise-scale estimate (≈ E[(loss_t − loss_{t−1})²]). Add "effective context length" =
min(seq_len, tokens_seen/steps) per checkpoint. Use a pairwise-ranking objective
(`lambdarank`) for ranking tasks instead of hand-built binary pairs — yields NDCG /
Kendall τ directly and handles ties. Add a stratified 10% shuffle split that balances
size buckets and holds out unseen **seeds** as the fast sanity check and the single HPO
validation split. Report Spearman ρ / Kendall τ and calibration (ECE) alongside decision
accuracy. Interpretability: SHAP TreeExplainer per fold, mean |SHAP| aggregated.

*B. Risks and cheap mitigations.* **Seed leakage** — when holding out recipes or sizes, the
same seed appears in train and test; always split on (recipe, size, seed). **Pair class
imbalance** (most pairs tie or differ marginally) — `lambdarank` or weight pairs by
|Δmetric|. **Heteroscedastic noise** (tiny models far noisier) — log-transform and z-score
within size bucket; pass size as a numeric feature. **Featurization cost** (>10k examples ×
repeated fits) — cache fits keyed by (recipe, size, seed, metric) (`joblib.Memory`).
**Non-IID recipes** (Dolma resamplings vs. FineWeb-Edu) — hold out recipe *families* in one
fold and report separately, so the model can't latch onto recipe identity.

*C. TODO fill-ins.* Modelling-choices list: code-token mix ratio → math (Gadre et al. 2024);
warm-up length → calibration (Mao et al. 2024); curriculum ordering → convergence (Zhang et
al. 2025) — all unverified attributions. GBDT spec: LightGBM, 512 leaves, LR 0.05, early
stopping 50 rounds on validation NDCG; lambdarank for pairwise, MSE for scalars. Recipe
features: total tokens; % code / % CC-derived / % social-media; mean document length;
duplicate-rate estimate. Target-timestep sampling: K = 3 fractional positions {33%, 66%,
100%} of the remaining training after S₀. Early-curve downsampling: 16 equispaced points in
log-time per metric. Fit visualisation: one subplot per fit type overlaid.

*Timeline claim.* With cached fits, full LOOCV over 25 recipes ≈ 3 CPU-hours; expanding
window (≤60M → rest) ≈ 2 more. Everything is post-processing — no extra GPU jobs.

## 2025-07 — Third response: a recipe-family scheme for leave-family-out CV

Reusable beyond this proposal (any DataDecide leave-recipe-family-out design: `REC`, `IRT`,
`ANN`). Verify membership against the repo's authoritative recipe list
(`named_data_mixes.py`) before use.

**Eight folds by shared provenance** (hold out the family, train on the rest):
1. Dolma 1.7 family — original + the four ablations (no code, no math/code, no Reddit, no
   Flan);
2. Dolma 1.6++;
3. C4;
4. FineWeb — Pro + Edu;
5. Falcon base (RefinedWeb only);
6. Falcon + CC quality-filtered variants (QC 10%, QC 20%, QC Orig 10%, QC Tulu 10%);
7. DCLM-Baseline + QC variants (QC 7% FW2, QC 7% FW3, QC FW 3%, QC FW 10%, QC 10%, QC 20%);
8. λ-mixtures of DCLM and Dolma (25 / 50 / 75%).

Rationale: each family shares crawl / curation pipeline, so "the generalisation gap you
measure [is] meaningful rather than arbitrary"; the structure also suits a later GP
comparison (kernels benefit from knowing test points are an unseen domain). Implementation:
a `family_map` dict; per fold, train = runs whose recipe ∉ family, test = runs ∈ family; the
rest of the LOOCV loop unchanged.

Variants: **6 folds** — merge Dolma 1.6++ into Dolma; merge FineWeb into Falcon+CC (all
quality-filtered CC). **4 coarse folds** — (Dolma 1.7 + 1.6++), (CC-derived: C4 + Falcon +
Falcon-CC), (FineWeb), (DCLM + mixes); "use the coarse scheme only if runtime becomes a
bottleneck; it hides interesting variation."

*Intake note.* The response's size claim ("the smallest family … FineWeb … ≥ 84 runs") is
wrong as written: three families are single recipes (Dolma 1.6++, C4, Falcon base → 14 × 3 =
42 runs each). Test folds that small are fine for a held-out domain but their ranking
metrics will be noisy — pair with the bootstrap CIs from the first review.

## 2025-07 — Fourth and fifth responses: singleton folds; expanding-window over sizes

**Danielle's pushback on the family scheme.** Dolma 1.6++, C4, and Falcon base folds hold a
single recipe; and the Dolma 1.7 ablations would plausibly look different from the base —
"would you still argue for the 8-fold approach?"

**Response (near-verbatim, condensed).** Keeps 8 folds for the first pass: every held-out
fold is "a recognisably different data-provenance shift" and still yields ≈ 42 evaluation
runs (14 sizes × 3 seeds), "enough to compute rank-correlation or decision-accuracy with low
variance" (claims NDCG std ≈ 0.01 on pilot runs — unverified). Singletons test distinct
shifts: Dolma 1.6++ = same lineage, more sources (positive heterogeneity shift); C4 =
older, lightly filtered crawl; Falcon = strict dedup/filter (implicit data-quality prior).
Grouping the Dolma 1.7 ablations with the base is "fine for v0": they differ by *removing*
one slice over an identical crawl/dedup/topic mix, so the shift is "missing a modality"
rather than an unseen pipeline; promote an ablation to its own fold later if the model
struggles on it. Collapse to 6 folds (C4 + Dolma 1.6++; Falcon + Falcon-CC) only if
wall-time matters or singleton-fold metrics are unstable (e.g. Spearman ρ ±0.15). Split to
10 folds (base + each ablation) if the question is which missing slice hurts predictability
most. Staged plan: run 8, inspect per-fold variance, then decide.

**Danielle's second question.** Given sizes 4M–1B and — as she understood it then — "for all
but 1B, 2 seeds are only run until 25% whereas only 1 seed makes the full run," how to do
the expanding window over model sizes?

**Response (near-verbatim, condensed).** Treat truncated seeds as right-censored. (1) Build a
*canonical-seed* dataset: one row per (recipe, size) using only the seed that ran to 100% →
25 × 14 = 350 rows per metric; early-window features averaged across all seeds where they
exist; target = the canonical seed's true final metric. (2) Expanding windows: size ladder
[4, 6, 8, 10, 14, 16, 20, 60, 90, 150, 300, 530, 750, 1000]M; S_max ∈ {20, 60, 90, 150, 300,
530, 750}; train = all rows with size ≤ S_max across all recipes; test = size > S_max,
**reported per test size** to draw "performance vs. held-out size." Start at 20M because
sub-10M models rarely expose sharp optimisation issues and mainly inflate the train set.
(3) 1B's three full seeds: treat as independent test rows (v0) or average. (4) Truncated
seeds: reuse as features, skip as labels until a "predict 25% → 25%" or censored-regression
phase. (5) Sketch: mark `full_run = tokens_trained ≥ max_tokens`; pick the canonical row per
group; loop over windows; per-size metrics. Headline shape: "with only models up to 20M we
already predict the 300M model within ±Z%, and adding 60M cuts that error in half."

*Intake note (2026-08-22).* The seed-truncation premise should be re-checked before reuse:
`../../open-questions-answered.md` (2026-08-21) records the aggregate OLMES table as 25
recipes × **3 seeds at every size**, and instance-level tables with 3 seeds at 150M–1B and 1
seed below 150M (and 750M's aggregate table truncated at step 26,250 while its instance
table runs to 63,599). If the three-seeds-to-completion reading holds, the canonical-seed
construction is unnecessary and the expanding window can use all seeds as rows — with the
(recipe, size, seed) split rule from the second review.

## 2025-07 — Sixth and seventh responses: inner validation split; family × size stress test

**Inner validation split (Danielle asked how the seed finding changes the earlier
"hold out unseen seeds" suggestion).** Response (condensed): with one canonical row per
(recipe, size) there is no seed axis to stratify on, so switch to **recipe-grouped,
size-stratified** splits: `group_id = recipe-size` (keeps the 1B triple together);
stratify on size buckets tiny {4, 6, 8} / small {10, 14, 16, 20} / medium {60, 90} / large
{150, 300} / huge {530, 750, 1000}; `StratifiedGroupShuffleSplit(test_size=0.10,
random_state=42)`; one 90/10 split suffices for LightGBM early stopping. Nest the *same*
splitter inside every outer fold (inside the 7-family train set for LOFO; inside the ≤ S
rows for each expanding window — even ≤ 20M has 25 × 6 = 150 rows). 1B extra seeds: keep as
three rows sharing a `group_id`, or average. Properties: no leakage between rows sharing
training dynamics; every validation slice spans the size ladder; one splitter reused
everywhere. *(Same caveat as above: if three seeds run to completion at every size, group
on (recipe, size) still, but rows per group become 3 everywhere.)*

**Family × size stress test (Danielle asked for splits that train on some recipe families
+ smaller sizes and test on larger models from held-out recipes).** Response (condensed):
a single-axis scheme over the eight families 𝔽 with 𝕊small = {4…60}M and 𝕊large =
{90…1000}M. For each held-out family F*: **train** = family ≠ F* and size ∈ 𝕊small; **val** =
10% stratified-group split inside train; **test-in-fam** = family = F* and size ∈ 𝕊large;
**test-out-fam (optional)** = family ≠ F* and size ∈ 𝕊large — the out-of-family column
separates size-extrapolation error from new-distribution error. Counts per fold
(canonical-seed dataset): train ≈ 7 × 8 × 25 ≈ 1,400 rows (the response's arithmetic
double-counts — 7 families already contain all non-held-out recipes, so it is
(25 − |F*|) × 8 rows ≈ 130–190); test-in-fam 6 × |F*| (× seeds) → 6–54 rows depending on
family size. Refinements after v0: repeat with 𝕊small ≤ 20M then ≤ 60M to draw a
"how much size coverage before predictions stabilise" curve; train on three wildly
different families (Dolma, FineWeb, DCLM) and test on the other five; survival-GBDT
extension for truncated seeds if that premise holds.

*Intake note.* The row counts quoted in this response overstate the training set by ~7×;
corrected above. Singleton-family test sets at 6 large sizes × 1 seed = 6 rows are too small
for stable ranking metrics — report them with CIs or pool singleton families for this
stress test.

## Open questions

- Is this still live (July 2025 draft; DataDecide scaling-law baselines have since been
  published by the DataDecide team — the "does early dynamics beat scaling-law
  extrapolation" comparison may now have a published answer to position against).
- Seed coverage: reconcile the July-2025 "only one seed runs to completion" premise with
  the 2026-08-21 finding of 3 seeds at every size in the aggregate table.
- Whether the target should be the annealed (`ANN`) readout rather than the raw final
  checkpoint.
- Promote, absorb into `TINY` as an option, or archive.

**Waiting on:** Danielle's status call.

# Early dynamics predict model performance — forecasting DataDecide outcomes from the first 10% of training

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

## Open questions

- Is this still live (July 2025 draft; DataDecide scaling-law baselines have since been
  published by the DataDecide team — the "does early dynamics beat scaling-law
  extrapolation" comparison may now have a published answer to position against).
- Whether the target should be the annealed (`ANN`) readout rather than the raw final
  checkpoint.
- Promote, absorb into `TINY` as an option, or archive.

**Waiting on:** Danielle's status call.

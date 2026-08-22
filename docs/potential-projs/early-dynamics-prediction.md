# Early-dynamics prediction — forecasting DataDecide outcomes from the first 10% of training

> **Draft scaffolding (2026-08-22).** Promoted from the staging topic
> `early-dynamics-prediction`. §1–§3 are synthesized from Danielle's July 2025 draft (local
> copy `../refs/2025-07-early-dynamics-predict-model-performance.pdf`), her implementation
> state, and the design decisions settled in the 2025-07 review threads; §4 is the dated
> record. Treat §1–§3 as provisional until this note is removed.

**Program pillars served:** how — measurement at small scale and *early time*: the
early-time analogue of tiny-scale measurement's early-scale question; independent variable
— recipe effects as they appear in early dynamics. (Program: `README.md` → Program.)

**One-line pitch.** Predict where a DataDecide run ends — final (or annealed) perplexity and
downstream correct-prob, and the *ranking* of recipes — from features of the first ≤ 10% of
training (or ≤ 2B tokens), with GBDTs over curve-shape features; test generalisation across
model scale (train ≤ 20M → predict ≥ 60M, expanding) and across recipe families; compare
against scaling-law extrapolation and naive continuation baselines.

IDs: EDP-1–EDP-4, EDP-opt-1–EDP-opt-5.

**Paper goal.** Workshop: the thin vertical slice — one CV axis (model scale), two targets
(Pile-validation perplexity; MMLU correct prob), baselines vs. GBDT, SHAP. Main conference:
add leave-recipe-family-out and the family × size stress test, the any-step predictor, the
all-metrics → MMLU setting, and a GP comparison with uncertainty.

Compute tiers: **T0** = analysis of published tables only — the entire core; **T1** only if
extended early metrics (curvature, gradient statistics) are added.

---

## 1. What the project involves

### Core experiment (EDP-1–EDP-3)

**Data.** DataDecide: 25 recipes × 14 sizes (4M–1B) × seeds; per-checkpoint perplexity on
11 validation splits and OLMES downstream tasks. Seed coverage to be reconciled: the
July-2025 premise ("only one seed runs to completion below 1B") conflicts with the
2026-08-21 check in `../open-questions-answered.md` (3 seeds at every size in the aggregate
table; instance tables 3 seeds at ≥ 150M). If all seeds complete, every seed is a row and
splits are on (recipe, size, seed).

**Setup.** Early window S₀ = min(10% of training, 2B tokens); target at step S (final) —
CORRECT PROB for downstream tasks (larger spread, less noise than accuracy at small scale),
log perplexity for validation splits. Targets transformed (log / logit), never z-scored per
size; inverted for reporting.

**Features (per metric curve, three windows: warm-up, early LR-decay, full early).** Per
window: step count, first/last value, first–last difference and slope, four fit types
(log-log, lin-lin, log-lin, lin-log) each with slope and one goodness-of-fit statistic (R²
or std err; p-values and CI bounds dropped as deterministic twins), window mean and std;
optionally the 16 log-spaced interpolated points and 12 rolling OLS slopes. Plus
architecture/schedule scalars (`d_model`, `n_layers`, `n_heads`, `mlp_ratio`, LR
schedule, batch, total tokens — collinear with size under fixed per-size configs) and
recipe properties (to be taken from `recipe-featurization.md`'s measured set rather than
hand-assigned percentages). Transforms: `log` for perplexity levels, logit for
correct-prob, signed-log for differences, `log1p` for counts; slopes and R² untransformed.
Per-size z-scoring is **not** used on the scale-generalisation axis (held-out sizes have no
training statistics; global z-scoring is a no-op for trees) — `log(size)` carries the size
trend.

**EDP-1 — Thin vertical slice (workshop core).** Scale axis only: train ≤ 20M, test per
held-out size, then expand (≤ 60M, ≤ 90M, …). Targets: Pile-val perplexity, MMLU correct
prob. Models: LightGBM regressor (starter params; tune `num_leaves`, `min_data_in_leaf`,
`learning_rate`, `feature_fraction` once on a grouped, size-stratified 10% slice of the
training sizes). Baselines: train-set mean; repeat-last-value; two-point linear
extrapolation; log-log power-law extrapolation; random ranking; a Kaplan-style
scaling-law fit on ≤ 60M models. Metrics: ρ/τ (ordering), MAE/MAPE or NRMSE (magnitude),
pairwise decision accuracy (binary), ECE with hand-rolled quantile bins (calibration);
bootstrap CIs over pairs. SHAP per fold, mean |SHAP| with cross-fold std.

**EDP-2 — Ranking head and recipe-family generalisation.** `lambdarank` head on the same
features with queries = one *size* (recipes as the ranked items); leave-recipe-family-out
CV over the eight provenance families (Dolma 1.7 + ablations; Dolma 1.6++; C4; FineWeb
Pro+Edu; Falcon; Falcon+CC QC; DCLM + QC; λ-mixtures), with singleton-family folds
reported with CIs; the family × size stress test (train ≤ 60M on 7 families → test ≥ 90M
on the held-out family, with an in-family/out-of-family split of the error).

**EDP-3 — Any-step and all-metrics settings.** (a) One model predicting a metric at any
target step: long format with `τ_target` as a feature, features fixed at S₀ (never masked
at τ), group = run; ranking metrics computed within τ and size. (b) All early metrics →
final MMLU: 28 features per metric curve + recipe properties; direct regression, and the
*oracle* variant feeding true final values of the other metrics as the actual upper
bound; permutation test as leakage guard.

**EDP-4 — Target choice: annealed vs. raw.** Because late-training rankings are partly a
cosine-tail artefact (`annealed-readouts.md`, `trajectory-statistics.md`), repeat EDP-1
with the annealed readout as the target where available and report which is more
predictable from early dynamics.

### Optional directions

- **EDP-opt-1 — GP comparison.** Sparse / multi-output GP on the validated
  low-dimensional feature subset; uncertainty bands; active-learning extension (which run
  to continue next).
- **EDP-opt-2 — Extended early metrics.** Curvature (top Hessian eigenvalue, trace
  estimate), feature stability, gradient/activation variance — requires checkpoints (T1).
- **EDP-opt-3 — Finetune-outcome prediction.** Predict SFT/DPO outcomes from pretraining
  early dynamics + early finetune metrics (the post-training bridge).
- **EDP-opt-4 — Early-window sweep.** Performance vs. early-window fraction / absolute
  tokens; where does predictability saturate.
- **EDP-opt-5 — Cost-aware framing.** "Replace 90% of training with 10% + a predictor":
  F1-of-decision vs. compute saved, per size.

## 2. Doability and impact

### Overall doability: **high** (T0; pipeline already partly built in July 2025)

Feature extraction, the pruned 67-column schema, transforms, and the LightGBM setup exist;
all experiments are post-processing of published tables. Risks: seed-coverage premise;
small per-size test folds (noisy ranking metrics — CIs mandatory); the published
DataDecide scaling-law baselines must be positioned against honestly; collinearity of
schedule features with size on the held-out-size axis; rabbit-holing on features (the
third review's minimal-slice advice).

### Per-direction impact

- **EDP-1.** Workshop paper on its own if early dynamics beat scaling-law extrapolation on
  held-out sizes; the SHAP story (which window, which fit) is the interpretable payload.
- **EDP-2.** Turns it into a recipe-selection result — the DataDecide question proper.
- **EDP-3/4.** Main-conference material; EDP-4 connects to the annealing-confound line.
- **EDP-opt-1/3.** Highest ceiling (uncertainty-driven run selection; post-training
  prediction).

## 3. Infrastructure build sequence

1. **Loaders and canonical frame** — long (run, metric, tokens, value) + metadata; pinned
   DataDecide table versions; resolve seed coverage from `../open-questions-answered.md`.
2. **Feature pipeline** — windows, fit statistics, transforms, pruning; cached Parquet
   with a hash-tagged filename; `dyn_*` / `stat_*` namespaces; recipe properties from
   `REC`.
3. **Splitters** — expanding-size windows; grouped, size-stratified inner split; recipe
   families; family × size stress split.
4. **Baselines + LightGBM heads** (regression, `lambdarank`) with the starter params and
   one-time tuning; evaluation script emitting ρ/τ, MAE/MAPE/NRMSE, decision accuracy,
   ECE, NDCG, bootstrap CIs; SHAP aggregation.
5. **Deliverables** — baseline-vs-GBDT table per target; predicted-vs-true scatter for
   held-out sizes; SHAP bar chart; runtime note.
6. *(Optional)* any-step long format; all-metrics → MMLU; GP baseline.

---

## 4. External assessments and origin notes

Dated notes from Danielle's July 2025 draft and the review threads this doc was promoted
from — recorded for consolidation, not decisions. Figures, attributions, and ballparks
quoted from the reviews are unverified; Danielle's prompts are logged verbatim in
`../danielle-inputs.md`.

### 2025 (undated, pre-July) — Lineage: the CIFAR-10 loss-slope study

An analysis of Danielle's notes from an earlier period (the original notes export is lost;
only the response survives, and its date is inferred from the tools it references — 4o vs.
o3, Heptabase — so roughly spring 2025) lists an "ExpMan + loss-slope study" as one of
seven active tracks: an "empirical study of loss-curve linearity vs. accuracy across sweeps
on CIFAR-10. Focus on early-epoch metrics as predictors and on regression slope/R²
diagnostics." The response's summary of early findings: "validation-loss slope (epoch
0–20) anticorrelates with final accuracy; RMSE strongest single predictor," with advice to
"confirm on ImageNet-subset to rule out CIFAR artefacts," "report adjusted-R² and
confidence intervals," and "consider mixed-effects model to separate optimiser vs
augmentation contributions."

Intake note (added 2026-08-22): the "loss-curve linearity → better training" hypothesis
the CIFAR study tested came from Danielle's advisor; her check on whether it is an
established belief is recorded in `../topics/reference/loss-curve-forecasting.md` —
**neither search found literature support for it**, so the linearity/R² features in EDP
inherit an untested premise and must be framed as a hypothesis under test, not a
consensus. This is
the CNN-sweep precursor of EDP — the same question (early-window
curve-shape features → final performance) asked on Danielle's own `dr_exp`/`deconCNN`
CIFAR-10 runs before the July 2025 DataDecide draft. The "early findings" quoted above are
the response's paraphrase of notes that no longer exist and are unverified; treat them as
a pointer that a CIFAR-scale pilot of the slope/R² features was run, not as a result.
The "RMSE strongest single predictor" line is also ambiguous (RMSE of what fit?) and
matches the later residual-vs-RMSE confusion flagged in the 2025-07 review thread.

### Origin notes — moved from `topics/staging/early-dynamics-prediction.md`

### 2025-07 — The proposal (from the draft)

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

### 2025-07 — External review of the plan (near-verbatim, condensed)

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

### Relation to existing docs

- `../topics/reference/loss-curve-forecasting.md` holds the loss-to-loss / multi-power-law /
  task-scaling-law references this proposal positions against.
- `tiny-scale-measurement.md` (`TINY`) asks how far down the scale
  ladder decision signal survives — the early-*time* analogue of the same question is this
  proposal; natural cross-listing.
- `trajectory-statistics.md` (`TRJ`) and
  `annealed-readouts.md` (`ANN`) supply the caveat that late-training
  rankings are partly a cosine-tail artifact, which bounds what "predicting the final
  ranking" can mean.

### 2025-07 — Second review: GBDT v0 design details (near-verbatim, condensed)

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

### 2025-07 — Third response: a recipe-family scheme for leave-family-out CV

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

### 2025-07 — Fourth and fifth responses: singleton folds; expanding-window over sizes

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
`../open-questions-answered.md` (2026-08-21) records the aggregate OLMES table as 25
recipes × **3 seeds at every size**, and instance-level tables with 3 seeds at 150M–1B and 1
seed below 150M (and 750M's aggregate table truncated at step 26,250 while its instance
table runs to 63,599). If the three-seeds-to-completion reading holds, the canonical-seed
construction is unnecessary and the expanding window can use all seeds as rows — with the
(recipe, size, seed) split rule from the second review.

### 2025-07 — Sixth and seventh responses: inner validation split; family × size stress test

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

### 2025-07 — Eighth and ninth responses: featurization

**Danielle's questions.** Explain "log-transform and z-score within each model-size bucket"
(what and why); why those five recipe features and are they sufficient; any more
featurization thoughts. Then: perplexity is obviously log-transformed, but should
correct_prob (already on [0, 1]) be transformed too?

**Per-size normalisation (condensed).** For each early-window scalar m and size s: m_log =
f(m); μ_s, σ_s computed **on training rows only** within that fold; m̂ = (m_log − μ_s)/σ_s;
store (μ_s, σ_s) and apply to val/test. Why: mean *and* variance shrink with size
(heteroscedasticity); LightGBM's axis-aligned splits would spend depth separating sizes
before learning curve shape; per-bucket z-scoring "puts every size on roughly the same
numeric footing, so subsequent splits encode the dynamic behaviour you actually care
about," and SHAP then reflects shape rather than "8M has higher raw loss than 750M."
Re-add `size_log = log(params_M)` as its own feature so the removed global trend returns as
a learnable interaction.

**Transforms by metric.** Perplexity (1, ∞): log. **Correct prob (0, 1): logit**,
`log((m + ε)/(1 − m + ε))`, ε ≈ 1e-4 — spreads the squeezed tail near 1, makes increments
additive (Δlogit ≈ log odds ratio), near-Gaussian noise so the z-score has sensible
variance; arcsin √ is the alternative but less interpretable. Since early correct_prob sits
≈ 0.01–0.05 rising toward ~0.2–0.3, logit linearises small absolute gains. Mask exact 0/1
before logit. Rule of thumb: bounded + relative differences matter → logit; positive reals
→ log; then z-score.

**Why the five recipe features** (total tokens; % code; % CC-derived; % social media; mean
document length; duplicate rate): "five orthogonal axes: quantity, domain, quality,
structure, redundancy"; one scalar each; sufficient for a v0 whose goal is to demonstrate
signal without overfitting. Backlog: tokenizer diversity (unique BPE tokens / 1k docs);
unigram entropy; quality-classifier score moments; rolling slope/curvature at several
radii; spectral density / AR(1) of loss deltas; LR-phase one-hot or continuous cosine
position; log grad-norm and param-norm; Fisher-trace / Hessian-norm estimates;
log(batch_tokens), grad-accumulation, tokens-per-update × LR; LM-based quality scores;
recipe-family one-hot or learned embedding (only if families are not held out).

**Hygiene.** Separate `dyn_*` vs. `stat_*` column namespaces for regex ablations; immutable
feature-derivation code with a hash-tagged artefact (`features_v0_<sha>.parquet`); cache
the bucket μ/σ dict in the artefact so the later GP baseline normalises identically;
ablation waterfall: early-curve only → + `size_log` → + static stats.

*Intake note.* The static recipe features overlap heavily with the recipe-featurization
project (`REC`) — if this proposal is promoted, its "static" feature set should be the
measured recipe properties from `REC`, not hand-assigned percentages, which would also make
the two projects share one artefact.

### 2025-07 — Tenth response: transform the targets too?

**Danielle's question.** Apply the same transformation to the prediction targets?

**Response (condensed).** Yes — the same monotone transform (log for perplexity, logit for
correct_prob) on the training targets; invert (exp / sigmoid) before reporting errors on the
original scale. **Do not z-score targets per size bucket**: the learner needs the global
shift with size ("for the same early slope, bigger models end lower"), which lives in the
bucket mean shift; supply `log(size_params)` as an input feature instead. Why transform
targets: per-size heteroscedasticity (final-perplexity variance shrinks ~10× from 4M to
1B) so squared error weights sizes evenly; skew/heavy tails from a few hard recipes;
relationships become approximately additive (multiplicative for perplexity, odds-ratio for
probabilities). Ranking objectives (`lambdarank`) are invariant to monotone transforms — no
inverse needed for NDCG/τ. Sanity ballparks quoted (unverified): log-perplexity RMSE ≈ 0.30
→ ≤ 0.05 in 200 trees, abs-% error < 6% on held-out sizes; logit-correct_prob RMSE ≈ 0.35
→ ≤ 0.07, Spearman ρ > 0.85. GPs may prefer raw targets later (they model heteroscedastic
noise differently).

### 2025-07 — Eleventh response: training method details

**Danielle's questions.** Spearman/Kendall and ECE — how and why? Is `lambdarank` different
from LambdaMART for the regression target? Why 16 log-spaced points? Are the quoted LightGBM
settings the right starting point?

**Response (condensed).**
- *Metrics.* Spearman ρ / Kendall τ between predicted and true final metric: order matters
  for pruning; invariant to monotone transforms; comparable across perplexity and
  correct_prob. ECE for regression: bin `y_pred` into ~15 equal-count bins, compare mean
  prediction vs. mean truth per bin, ECE = Σ |gap| × bin fraction — exposes systematic
  optimism for large models / pessimism for small ones. "ρ/τ tell you ordering quality;
  ECE tells you numerical trustworthiness."
- *Objectives.* `lambdarank` (LightGBM's LambdaMART): pairwise lambdas from the NDCG
  gradient, pairs sampled on the fly (no O(n²) materialisation), ties get zero pair weight;
  the label can simply be the target value. Pointwise regression (MSE) when the numeric
  value is needed (calibration, curves). Hand-rolled binary win/loss is dominated by
  `lambdarank`. Train two heads off the same feature matrix.
- *16 log-time-spaced points.* Trees need a rectangular matrix (no variable-length
  sequences); log spacing samples densely where curves change fast and sparsely where they
  flatten, and makes feature importance interpretable ("point 7 ≈ 30k tokens"). Recipe:
  `grid = exp(linspace(log(0.01·T), log(T), 16))`, `np.interp`; 16 × 11 metrics = 176
  features.
- *Starter hyper-parameters.* `num_leaves=512`, `learning_rate=0.05`,
  `feature_fraction=0.9`, `bagging_fraction=0.8, bagging_freq=1`, `min_data_in_leaf=20`,
  `early_stopping_rounds=50` on validation NDCG (ranker) or RMSE (regressor),
  `n_estimators` large. Later: small Optuna sweep over {num_leaves, lr, min_data_in_leaf};
  monotone constraint (final perplexity decreasing in size); CatBoost as a Phase-2 baseline
  for high-cardinality categoricals.

*Intake note.* `num_leaves=512` with `min_data_in_leaf=20` on ~130–350 training rows (see
the corrected counts above) cannot actually grow 512 leaves — effective depth will be
bounded by `min_data_in_leaf`; treat the setting as harmless rather than tuned. The
"16 × 11 metrics" count also predates the choice to subsample recipes/metrics.

### 2025-07 — Responses 12–15: metric suite, ECE binning, lambdarank inputs, two-head code

**Metric suite (Danielle asked whether only ρ and ECE should be reported, vs. the standard
absolute/relative error and decision accuracy).** Response: report four complementary
dimensions — *ordering* (Spearman ρ / Kendall τ; NDCG@k if a ranking head exists),
*magnitude* (MAE or RMSE on the original scale; MAPE), *binary decision* (decision accuracy =
pairwise winner hit-rate, computed once from regression or ranker outputs), *calibration*
(ECE on binned residuals — two predictors with equal MAE can differ wildly in bias). ρ and
ECE were called out first as "the two most sensitive indicators of trouble," not as
replacements. Per-fold reporting table: ρ, τ, MAE, MAPE, decision acc., ECE, NDCG@10, plus
one baseline row (global mean, or "final = last early value"). Slide deck: ρ, MAE, decision
acc., ECE; full table in an appendix. Decision accuracy on log-space targets is identical
(sign of differences unchanged).

**ECE binning.** Scikit-learn's `calibration_error` is for classification probabilities;
for regression targets you implement the binning yourself (equal-count quantile bins,
e.g. 15; `np.digitize`; Σ |mean pred − mean truth| × bin fraction) or use a small library
(`netcal` was named). ECE varies a lot with the binning scheme, so control bin count and
strategy explicitly.

**lambdarank inputs.** Keep the same one-row-per-run `X`; `LGBMRanker` takes items plus a
**group/query vector** (e.g. one query per recipe — all sizes × seeds ranked together), and
forms the within-group pairs and NDCG-gradient lambdas internally. Relevance label = any
monotone transform with higher-is-better (e.g. −log final perplexity). LightGBM's
`"lambdarank"` objective *is* LambdaMART (LambdaRank + boosted trees). Explicit pairs are
only needed for an external "A beats B" classifier, which is discouraged.

**Two-head sketch.** One feature matrix (`dyn_*`, `stat_*` columns); `y = log(final
perplexity)` (logit for correct_prob); `qid` = recipe code; size-bucket-stratified,
qid-grouped 90/10 split; `LGBMRegressor(objective='regression', metric='rmse')` and
`LGBMRanker(objective='lambdarank', metric='ndcg')` with shared params `num_leaves=512,
learning_rate=0.05, min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.8,
bagging_freq=1`, `n_estimators=10_000`, `early_stopping_rounds=50`; ranker gets
`group=lengths(qid_train)`, `group_eval=[lengths(qid_val)]`; evaluate MAE/MAPE/ρ on the
inverted scale, NDCG on the ranker, decision accuracy pairwise.

*Intake notes.* Two defects in the final code sketch: (1) its `decision_accuracy` compares
`argsort` arrays positionally (fraction of items at the same rank), which is not the
pairwise hit-rate defined earlier — use the `itertools.combinations` version from the
metric-suite response; (2) `lgb.engine._eval_function.NDCG` is not a real LightGBM API —
compute NDCG with `sklearn.metrics.ndcg_score` per group or via `ranker.evals_result_`.
Also, one query per recipe ranks *sizes* against each other, which is trivially easy (bigger
is better); the decision the project cares about is ranking *recipes* at a fixed size, so
the query should be per size (or per size × seed), with recipes as items.

### 2025-07 — Responses 16–18: SHAP, rolling slopes, three feature clarifications

**SHAP per fold, mean |SHAP| across folds.** TreeExplainer gives exact Shapley attributions
for tree ensembles cheaply; compute once per CV fold on that fold's validation/test rows;
per-feature importance = mean |SHAP| over rows, then mean (and std, as error bars) over
folds — absolute value removes sign cancellation, fold-averaging is a stability check.
Deliverables: top-k bar plot colour-coded dynamic vs. static; cross-fold std; a sentence
like "power-law exponent at ~30k tokens and log(size) explain most of the variance." Uses:
debugging (leaning on an artefact such as a recipe-ID one-hot), feature selection,
quantitative rather than eyeballed argument. (`feature_perturbation="interventional"`,
`check_additivity=False` in the sketch.)

**Rolling slopes.** Linear OLS only — one slope per 5-point window over the 16 log-spaced
points (12 windows per metric): on a short stretch the curve is near-linear so the slope ≈
local derivative; closed-form, tree-friendly, no intercept (redundant with the raw points).
Axis choice: log perplexity vs. log tokens; logit(correct_prob) vs. log tokens. Theil–Sen
if robustness is wanted. Revisit 2nd-order coefficients only if SHAP shows slopes <1%
importance *and* there are visible non-linear bumps within windows.

**Three clarifications (Danielle's questions: is "% of schedule" cumulative LR? can the
noise-scale estimate be built from sparse eval curves? is effective context length
relevant with a uniform 2,024-token sequence length?).** (1) `%_warmup_done` and
`%_decay_done` are fractions of the schedule completed at the checkpoint (not integrated
LR): `clip(tokens/warm_tokens, 0, 1)` and `clip((tokens − warm)/(total − warm), 0, 1)`,
stored per sampled point. (2) Noise scale: with only sparse evaluation checkpoints, use the
variance of first differences across the 16 points (`mean(diff(log_loss)²)`) as a crude
"jitter" proxy, or table it — note it as an approximation to be replaced by Fisher-trace
estimates if full logs appear. (The response's "ρ ≈ −0.4 on ResNet-CIFAR pilots" claim is
unverified.) (3) Effective context length = tokens_seen/steps is constant (= seq_len) when
every run uses the same sequence length and batch is counted in sequences — **drop** it;
only relevant with variable or curriculum sequence lengths.

*Intake note.* With DataDecide's uniform schedule per size, `%_decay_done` at a given
*fraction of training* is the same for all recipes within a size; it varies across sizes
only through the token budgets, so it is nearly collinear with `size_log` plus the
sampling-grid position. Expect it to add little unless the early window is defined in
absolute tokens (S₀ = min(2B, 10%)), in which case it does carry size information.

### 2025-07 — Responses 19–20: any-step targets; all-metrics → MMLU

**Danielle's two remaining settings.** (1) One model predicting the target metric at *any*
step — the target step must be featurized as an input. (2) Use all evaluation measures
(perplexity splits + downstream tasks at each checkpoint) across the early window to predict
MMLU at the end, "as somewhat of an upper bound." Stay close to the existing setup.

**First response (before the clarification; condensed).** Separate LightGBM per horizon
{25, 50, 75, 100%} (multi-target LightGBM as an alternative); for MMLU, direct regression on
all early metrics + static stats with a logit target, a stacked "upper bound" (predict each
metric's final value, then Ridge → MMLU), multi-task later; permutation test (shuffle MMLU,
expect R² ≈ 0) as a leakage guard. Quoted "pilot" ballparks (ρ ~0.75/0.85/0.92 at
25/50/75%; MMLU 0.55–0.70) are unverified and not from this data.

**Second response (after the clarification; condensed).**
- *Any-step predictor:* reshape long — one row per (run_id, τ_target) with τ_target =
  tokens_seen/tokens_total as a numeric feature, `y_target` = log/logit of the metric at τ;
  same LightGBM regressor; group-aware split with **group = run_id** (all τ rows of a run
  stay together), stratified on a coarse τ bucket; at inference feed the early-window
  features plus the τ you want. ~25 × 14 × 4 ≈ 1,400 rows per metric.
- *All-metrics → MMLU:* for every logged evaluation metric, 16 log-spaced points + 12
  rolling slopes (≈ 28 × #metrics + static features); target = logit(MMLU_final); direct
  regression first, stacked meta-learner (Ridge over per-metric final-value predictions) as
  the "upper bound" flavour; reuse the recipe-family and expanding-size outer folds with
  the group-aware inner split; report MAE/MAPE after inverse-logit, ρ, decision accuracy,
  ECE. Code glue: a shared feature builder, a `targets` dict with a `predict_any_step`
  flag, one training loop over targets × folds.

*Intake notes.* (1) The long-format sketch says to **mask features beyond τ_target** ("later
slots = NaN") — that contradicts the early-window premise: features must stay fixed at the
early window S₀ regardless of τ_target, otherwise predicting τ = 0.75 uses data up to 75%
of training. Only τ_target itself should vary across rows. (2) With one row per (run, τ),
the decision-accuracy and ranking metrics must be computed *within* a τ (and within a
size), or they mix trivially-ordered horizons. (3) The "stacked upper bound" is not an upper
bound in any strict sense — it is a second estimator; the actual upper bound for
"all early metrics → MMLU" is the *oracle* variant that feeds the true final values of the
other metrics, which is also the cleaner diagnostic (how much of MMLU is explained by final
perplexity/correct_prob at all).

### 2025-07 — Third distinct review: "thin vertical slice first"

Another review of the same plan (not a duplicate of the earlier two). Condensed; where it
contradicts the earlier reviews, the disagreement is noted rather than resolved.

- *Scope.* Full LOOCV over 25 × 14 × 3 "will generate >1,000 folds" — pick **one
  generalisation axis** for v0 (leave-one-recipe *or* expanding sizes) and one regression
  target (e.g. Pile-validation perplexity at end of training); add ranking, other metrics,
  and MMLU later. Minimal feature subset first: initial value, value at 10% tokens,
  power-law exponent, log(size); expand only if weak. *(Disagrees with the second review's
  16-point grid + 12 rolling slopes from the start — here those are "reconsider/postpone:
  high dimensional, may over-fit." Also postpones the recipe-composition features unless
  already computed.)*
- *Data checks.* Exclude warm-up artefacts (first ~0.5% of tokens or <10× batch) from curve
  fits, or model warm-up separately. **Seed leakage if seeds are averaged before the
  train/val split** — average after splitting. Clip correct_prob to [0.02, 0.98] before
  logit. Two sanity plots before modelling: early-vs-final scatter with r; per-metric
  cross-seed variance.
- *Features to add.* Slope over the last 20% of the early window (plateau detection);
  relative improvement (value_end − value_start)/value_start.
- *GBDT details.* `feature_fraction`, `bagging_fraction`, `max_bin=255`;
  `min_gain_to_split=0.01` against noise splits; LightGBM GPU if tables are large;
  **consider one LambdaMART model instead of separate regressor + ranker** when ordering is
  all that is needed *(disagrees with the two-head design)*.
- *Baselines.* Repeat-last-value extrapolation; two-point linear extrapolation; random
  ranking (decision accuracy should be 50%). Metrics: normalised RMSE (RMSE/range) for
  cross-metric comparability; Spearman ρ with bootstrap CIs.
- *Reproducibility.* Cache the design matrix; fixed LightGBM/numpy seeds logged in the
  notebook header; target ≤ 30 min per single-fold end-to-end run.
- *Deliverable.* One page: setup (one CV axis); baseline-vs-GBDT NRMSE and ρ on Pile-val
  perplexity; predicted-vs-true scatter; top-5 SHAP features.
- *Risks.* Curve-fit failures on non-monotonic curves (catch warnings, fall back to median
  early value); memory (native categorical handling for recipe/size rather than one-hot);
  time-to-first-result (follow the minimal slice; defer ranking and MMLU).
- *GP look-ahead.* Sparse / multi-output GPs (GPyTorch, GPflow) handle 50–100k rows, not
  millions; reuse the validated low-dimensional feature subset; GP uncertainty enables an
  active-learning extension for the conference version.

*Intake note.* The ">1,000 folds" and ">1M rows" / "GB-scale design matrix" framings
overstate the canonical dataset (≈ 350 rows per metric, 8 family folds + 7 size windows);
the structural advice (one axis, one target, minimal features, naive baselines, seed
handling) stands regardless.

### 2025-07 — Responses 21–22 (third-review thread): targets, baselines, implementation plan

**Danielle's decisions/questions.** Likes the narrow slice; why Pile perplexity rather than
a downstream metric — expand slightly to one perplexity + one downstream task? **Chooses
generalisation across model scale** as the single CV axis. Asks for detail on the simple
baselines, then for a from-scratch step list with code.

**Why Pile perplexity first (condensed).** Continuous, monotone, low cross-seed variance
(claimed 2–4% vs. ≥8% for MMLU at small scale), one value per checkpoint, cheap, and
correlated with many downstream tasks after log-transform — it validates the whole pipeline
in under an hour before touching noisier downstream metrics. Extension: add
`mmlu_cp_final` (correct prob, not accuracy) with the same logit/clip handling; predict it
from early MMLU curves, or cross-metric via stacking (predicted late Pile perplexity as one
extra feature). Scale axis: train ≤ 20M → test ≥ 60M, then expand. Report RMSE, ρ, decision
accuracy; binomial CIs for MMLU.

**Simple baselines (all stateless).** Train-set mean (detects leakage / unlearnable task);
repeat-last-value of the early window ("no further improvement"); two-point linear
extrapolation in tokens; log-log power-law extrapolation `y = a·T^b` fit on the early
window; random ranking (decision accuracy ≈ 50%). Decision accuracy via
`itertools.combinations` pairwise hit-rate — "a model with lower RMSE can still mis-rank the
best run."

**Implementation plan (structure only; code in the conversation).** `src/{io, features,
baselines, splitters, train, eval, plots}.py` + `meeting_run.py`. Steps: load curves to a
long (run, metric, tokens, value) frame + metadata; early-window mask `tokens ≤ min(2e9,
0.10·T_total)`; EDA log-log line plots; `summarise_one` features — start/end value,
relative improvement, log-log OLS exponent, trend slope, variance of first differences, 16
log-grid interpolated values; `scale_split` at 20M; baselines; LightGBM regressor wrapper
(starter params); RMSE + decision accuracy; loop over {pile_ppl, mmlu_cp}; scatter and
SHAP bar plots; pack deliverables. Post-meeting: replace the train-as-validation stub with
a real 10% held-out split; add the two extrapolation baselines; sweep early-window
fractions.

*Intake notes.* The sketch fits with `val = train` (acknowledged as a stub — early stopping
then does nothing); `tokens_last` is referenced but not stored; the "<25 min on one A100 for
~3k rows" and "peak RAM 6 GB" figures are placeholders, not measurements; the "ρ ≈ 0.7–0.9,
Table 3 of your notes" claim references the unrecoverable Gemini notes. The baseline set
(mean / repeat-last / linear / power-law / random) is the durable content here and aligns
with the oracle-ladder habit of reporting a floor row.

### 2025-07 — Implementation state: extracted features and training setup (Danielle)

**Feature schema** extracted per (model size, dataset) pair, 131 columns:
- *Three early windows* — `warmup_*`, `early_lr_decay_*`, `full_early_*` — each with: number
  of steps; first/last value; first–last difference and slope; for each of four fit types
  (`xlogylog`, `xlinylin`, `xlogylin`, `xlinylog`) the slope, R², p-value, RMSE, CI
  lower/upper, std err; window mean and std. (35 columns per window.)
- *Architecture / schedule*: `d_model`, `n_layers`, `n_heads`, `mlp_ratio`, `warmup_perc`,
  `warmup_steps`, `lr_decay_steps`, `lr_max`, `lr_final`, `batch_size`, `total_steps`,
  `total_tokens`, `total_tokens_billions`.
- *Recipe-level* (hand-assigned per dataset): `pct_code`, `pct_common_crawl`,
  `pct_social_media`, `mean_doc_length_tokens`, `duplicate_rate_pct`,
  `quality_filter_strength`, `is_mixed_dataset`, `num_sources_mixed`,
  `educational_content_score`.

**Training setup.** LightGBM regressor with the merged starter params (`num_leaves=512`,
`learning_rate=0.05`, `min_data_in_leaf=20`, `feature_fraction=0.9`,
`bagging_fraction=0.8`, `bagging_freq=1`, `max_bin=255`, `min_gain_to_split=0.01`,
`device_type='cuda'`, seed to set), `n_estimators=10_000`, `early_stopping_rounds=50` on a
validation RMSE. Generalisation axis: expanding window over model sizes; a train/eval split
exists, with an optional 10% validation slice inside train for hyper-parameter tuning.

**Her open questions (2025-07).** Which GBDT hyper-parameters to tune, and is tuning once
then reusing them fine? Which features get "z-score per model-size bucket" after
`log(ppl + 1e-8)` for perplexity-scale features? Anything else needed to prepare features
for the GBDT?

*Intake notes.* (1) The per-window fit statistics (p-value, CI bounds, std err) are
deterministic functions of slope/R²/n and of each other — heavily redundant for a tree
model; harmless, but SHAP importance will be diluted across them. (2) Several schedule
columns (`warmup_perc`, `warmup_steps`, `lr_decay_steps`, `total_steps`, `total_tokens`,
`total_tokens_billions`) are functions of model size under DataDecide's fixed per-size
configs, so they are collinear with `d_model`/`n_layers` — fine for trees, but they make
"size" identifiable many ways, which matters for the held-out-size design. (3) The
recipe-level columns are hand-assigned estimates; `REC` would replace them with measured
properties.

#### 2025-07 — Answers: what to tune, what to normalise, what to prune

**Hyper-parameters (condensed).** Tune `num_leaves` (64–1024), `min_data_in_leaf`
(10–200), `learning_rate` (0.01–0.20, log), `feature_fraction` (0.6–1.0) lightly; keep
`bagging_fraction=0.8`, `max_depth=-1`; `n_estimators` large with early stopping; add
`lambda_l1/l2` only if overfitting persists. Tune **once** on a 10% validation split of the
training-size bucket (≤ 20M) and reuse unless the feature count doubles, categorical splits
are added, or the target transform changes. ~100 Optuna trials, <30 min on GPU.

**Normalisation per size bucket.** YES for raw metric levels (`*_first_val`, `*_last_val`,
`*_window_mean`, after log / logit), differences and slopes, and variance/RMSE/std-err
(they inherit the metric's scale). NO for counts (`*_num_steps`, `total_steps`,
`total_tokens` — take `log1p` instead), hyper-parameters (`lr_max`, `batch_size`),
booleans (0/1), and 0–1 composition features (optional). Pattern: `StandardScaler` fit per
`model_size` group over the selected columns, applied *after* log/logit of the raw curves.

**Other prep.** `log1p` heavy-tailed counts; real `np.nan` for missing (−1/Unknown for
categoricals); remove perfectly collinear twins; store `tokens_last` for the extrapolation
baselines; persist the design matrix to Parquet; fix seeds (`random_state=42`).
Pre-train checklist: no unexpected NaNs; roughly Gaussian transformed distributions;
~120–140 columns after pruning; early stopping actually triggers (best_iteration < 4,000,
else lower the LR); validation RMSE ≲ the extrapolation baselines before touching the test
set.

**`log1p`.** ln(1 + x): finite at 0, numerically precise for tiny x, no hand-tuned ε —
replace `log(x + 1e-8)` with `log1p(x)` for counts that can be zero.

**Pruning the fit statistics.** Per window and fit type keep **slope** plus **one**
goodness-of-fit scalar (R² *or* `std_err`); drop `p_value` (function of R² and n), `rmse`
(rescaling of `std_err`), `ci_lower/upper` (slope ± t·std_err). With fixed n per window the
dropped columns are affine functions of the kept ones; trees waste depth choosing among
twins. Sanity check: far fewer |corr| > 0.95 off-diagonals after pruning.

*Intake notes.* (1) The Optuna objective sketch uses a plain random `train_test_split`
inside training — it should use the grouped, size-stratified split already designed, or
tuning will leak across (recipe, size) rows. (2) "Perplexity counts → `log1p`" is a slip:
perplexity is ≥ 1 and never zero, so plain `log` is correct there; `log1p` is for genuine
counts. (3) With per-size z-scoring *and* the tuned model reused across expanding windows,
the bucket statistics for held-out (larger) sizes must come from the training sizes only
or from a size-extrapolated fit — a per-test-size scaler fit on test rows is leakage; the
earlier reviews' "store (μ_s, σ_s) from training rows" rule does not directly cover unseen
sizes, and this gap is unresolved.

#### 2025-07 — R² vs. RMSE: are they really redundant?

**Danielle:** "I was under the impression that residuals (R²) and RMSE give you different
information that you should interpret differently and can sometimes point in different
directions. Is this false?"

**Response (condensed).** Within one fit on the same n points with fixed TSS, RMSE and R² are
monotone functions of the same RSS (R² = 1 − n·RMSE²/TSS), so they rank models identically
and a tree gains nothing from both; `std_err` is RMSE rescaled. They diverge only when
comparing across data sets / target scales (same RMSE, very different R² when TSS differs)
or when R² is computed against a different baseline. Recommends slope + one residual
statistic (R² for unit-free fit quality; RMSE for metric units; `std_err` for slope
uncertainty).

*Intake note — the response's conclusion does not follow for the feature matrix.* The
monotone link holds *within one row* (one run × window), but the GBDT compares features
*across rows*, and TSS (the variance of the metric inside the window) differs per run and
per size. Across rows R² is scale-free ("how straight") while RMSE carries the metric's
scale ("how far off, in log-perplexity units") — exactly the cross-data-set case the
response itself lists as divergent. So Danielle's intuition is right in this context: the
two are not redundant features, and which one matters is an empirical question (SHAP will
say). The pruning of `p_value` and `ci_lower/upper` still stands, since those are
row-wise functions of (slope, std_err, n) with n fixed per window — and `std_err` vs. RMSE
*are* redundant across rows only if Σ(xᵢ − x̄)² is constant per window, which it is for a
fixed sampling grid.

#### 2025-07 — Pruned feature set and Danielle's normalisation plan

**Pruned set (67 columns).** Per window (`warmup_`, `early_lr_decay_`, `full_early_`):
`num_steps`, `first_val`, `last_val`, `val_first_last_diff`, `val_first_last_slope`, and for
each of the four fit types `_slope` + `_r_squared`, plus `window_mean`, `window_std` (15 per
window). Architecture/schedule: `d_model`, `n_layers`, `n_heads`, `mlp_ratio`,
`warmup_perc`, `warmup_steps`, `lr_decay_steps`, `lr_max`, `lr_final`, `batch_size`,
`total_steps`, `total_tokens`, `total_tokens_billions`. Recipe: `pct_code`,
`pct_common_crawl`, `pct_social_media`, `mean_doc_length_tokens`, `duplicate_rate_pct`,
`quality_filter_strength`, `is_mixed_dataset`, `num_sources_mixed`,
`educational_content_score`. All normalisation switched to `log1p`; Pile perplexity only
for now.

**Her proposed treatment.** (1) 0/1-encode `is_mixed_dataset`. (2) Leave unchanged:
`lr_max`, `lr_final`, `batch_size`, `d_model`, `n_layers`, `n_heads`, `mlp_ratio`,
`pct_code`, `pct_common_crawl`, `pct_social_media`, `duplicate_rate_pct`. (3) `log1p` only:
`total_steps`, `total_tokens`, `total_tokens_billions`, `mean_doc_length_tokens`,
`warmup_steps`, `lr_decay_steps`, `full_early_num_steps`, `early_lr_decay_num_steps`,
`warmup_num_steps`. (4) Everything else on the perplexity scale: `log1p`, then per-size
bucket z-score. "Is this correct?"

*Intake notes (before the answer).* Unassigned by her list: `warmup_perc`,
`quality_filter_strength`, `num_sources_mixed`, `educational_content_score` (all plain
scalars — leave unchanged). Not on the perplexity scale despite being "the rest": every
`*_slope` and `*_r_squared` column (slopes are in transformed-axis units already; R² is
unit-free) — these should not get `log1p` (slopes are signed), and R² arguably needs no
bucket normalisation; `*_val_first_last_diff/slope` are differences of raw perplexities and
are signed too. `total_tokens` and `total_tokens_billions` are the same feature (drop one).
The perplexity-vs-`log1p` slip from the previous answer has propagated: for perplexity
levels (`first_val`, `last_val`, `window_mean`) plain `log` is the natural transform;
`log1p` is harmless for ppl ≫ 1 but muddles the log-log slope semantics if applied before
fitting.

**Answer (condensed).** "Almost perfect" with two tweaks. (1) `is_mixed_dataset` as 0/1
numeric. (2) Leave the listed hyper-parameters and percentages as is; optionally `log10`
the learning rates if they span >1 order of magnitude. (3) `log1p` the nine count-like
columns. (4) Perplexity-scale statistics: raw levels (`*_first_val`, `*_last_val`,
`*_window_mean`) → `log1p` → bucket z-score; `*_val_first_last_diff` → **signed log**
`sign(x)·log1p(|x|)` → bucket z-score; `*_slope` → no transform, bucket z-score; `*_r_squared`
→ nothing; `*_window_std` → `log1p` → bucket z-score. Checklist: no inf/NaN beyond allowed;
≤ 5% of off-diagonal |corr| > 0.95; roughly bell-shaped transformed columns; only
`is_mixed_dataset` in `categorical_feature` if marked.

*Intake notes on the answer.* It resolves three of the four items flagged above (slopes
untransformed; R² untouched; signed diff) but still does not assign `warmup_perc`,
`quality_filter_strength`, `num_sources_mixed`, `educational_content_score`, nor drop the
duplicate `total_tokens_billions`; and it keeps `log1p` for perplexity levels (harmless
for ppl ≫ 1, but then the `xlogylog` slopes — fit on `log` — and the `log1p`-transformed
levels are on slightly different transforms; unify to plain `log` for perplexity).

#### 2025-07 — Z-scoring implementation, and the unseen-bucket problem (resolved)

**Danielle's `zscore_by_param`** (groupby `params` = model size; per-column mean/std;
`transform` with `x.name` as the group key). **Answer:** correct as written; tighten with
`std(ddof=0)`, `.replace(0, 1)` against constant columns, precompute group stats once, pass
`means/stds` as lambda defaults. Then a `BucketZScaler` fit-on-train / transform-anywhere
class (caches per-bucket μ, σ; raises on unseen buckets; save/load).

**Danielle's realisation.** "Since I'm going to be evaluating generalization across bucket
sizes then my eval set will be all unseen buckets … it seems this type of scaling will
inherently make this type of generalization harder." **Answer:** yes — per-bucket z-scoring
inserts a systematic shift when test rows are normalised with another bucket's (or no)
statistics, and trees read that shift as a feature change. Three options: **A** global
z-score (μ, σ over all training rows); **B** log-only, no z-score; **C** fit μ(size),
σ(size) as smooth functions of log(size) on training buckets and extrapolate (linear
regression per column). Recommended default: A, then B if generalisation is poor; sanity
checks — scaled-column histograms on the new bucket centred near 0; unseen-bucket rows
still hit top splits; val-vs-test RMSE gap ≤ ~10%. (`StandardScaler` usage note appended.)

*Intake note.* For LightGBM, option A is a per-feature affine map and therefore a no-op
for tree splits — A and B give identical models; the choice only matters once the GP
baseline enters. The real decision is per-bucket (rank-changing) normalisation vs. none:
under the scale-generalisation axis, drop it (B) and carry `size_log`, or use C if
size-normalised inputs are wanted. This closes the "per-size z-scoring under held-out
sizes" open question below.

### Open questions (carried from staging)

- Is this still live (July 2025 draft; DataDecide scaling-law baselines have since been
  published by the DataDecide team — the "does early dynamics beat scaling-law
  extrapolation" comparison may now have a published answer to position against).
- Seed coverage: reconcile the July-2025 "only one seed runs to completion" premise with
  the 2026-08-21 finding of 3 seeds at every size in the aggregate table.
- Whether the target should be the annealed (`ANN`) readout rather than the raw final
  checkpoint.
- Unresolved disagreements across reviews: full 16-point + slopes feature set vs. a
  four-feature minimal slice first; two heads vs. one LambdaMART. (Decided in the third
  thread: CV axis = model scale; targets = Pile-val perplexity + MMLU correct prob.)
- Per-size z-scoring under held-out sizes — resolved 2025-07: drop per-bucket
  normalisation for the scale axis (log-only + `size_log`), or extrapolate μ(size), σ(size);
  global z-scoring is a no-op for trees.
- Any-step setting: confirm the feature window stays fixed at S₀ (see intake note); decide
  τ grid (K = 3 {33, 66, 100%} per the first review, or {25, 50, 75, 100%}).
- Source the static recipe features from `REC`'s measured properties.

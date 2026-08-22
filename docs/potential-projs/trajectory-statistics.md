# Trajectory drift/diffusion — in eval trajectories (the Signal-and-Noise dual)

**Program pillars served:** how (the noise floor and movement SNR), mechanism. (Program: `README.md` → Program.)

**One-line pitch.** Signal-and-Noise (Heineman et al.) treats
checkpoint-to-checkpoint variability as a nuisance term to suppress. Invert
it: treat inter-checkpoint change as the object and decompose it into
directional, cumulative *drift* (learning) and mean-reverting *diffusion*
(oscillation). The result is a "movement SNR" table over benchmarks,
recipes, and scales, plus a zero-training test of the river-valley
hypothesis using the cosine schedule as a within-run LR sweep.

**Compute tier.** T0. Inputs are the aggregate OLMES and perplexity
trajectories already in `data/processed/`, each row carrying `lr_at_step`,
`cumulative_lr`, tokens, and FLOPs.

## 1. What the project involves

### Core

- **TRJ-6 — Noise floor.** Before measuring movement, estimate the variance
  that is not movement: (a) pooled seed variance across 25 recipes × 3 seeds
  at fixed scale, with a heteroscedasticity test across recipes; (b)
  trajectory-as-replicate (late-window variance within a run), corrected for
  within-window drift; (c) item bootstrap for benchmark-composition
  uncertainty (uses per-instance tables). Output: per-metric, per-scale
  noise floors that every later claim is tested against.
- **TRJ-1 — Drift/diffusion decomposition.** For each (benchmark, metric,
  recipe, scale, seed) series: increment autocorrelation, sign-consistency
  of increments, variance-vs-lag scaling (diffusion ∝ lag, drift ∝ lag²).
  Fit a simple drift+diffusion model (e.g. local linear trend with AR(1)
  noise, or Ornstein–Uhlenbeck around a trend) and report drift rate,
  diffusion scale, and their ratio. Apply to accuracy, likelihood margins,
  and perplexity metrics.
- **TRJ-4 — Re-derive Signal-and-Noise.** Predict and confirm that continuous
  metrics have a higher drift-to-diffusion ratio than accuracy, and that
  "filter noisy subtasks" is recovered as "drop low-ratio tasks." This is the
  sanity check that the decomposition measures what it claims.

### Optional directions

- **TRJ-2 — River-valley LR test.** Diffusion magnitude should scale with the
  current learning rate (wall oscillation) while drift should not (progress
  along the river). The cosine schedule supplies a monotone LR sweep inside
  every run; regress per-window diffusion and drift on `lr_at_step`.
- **TRJ-3 — Recipe signatures at matched loss.** Pair checkpoints across
  recipes at equal loss and compare drift/diffusion signatures. Any
  benchmark where recipes differ at matched loss is "pretraining shapes
  models beyond final performance" in public data.
- **TRJ-5 — Resolution transfer.** If DataDecide's checkpoint spacing is too
  coarse to separate diffusion from drift, fit the diffusion component on
  the denser OLMo trajectories in the Signal-and-Noise release and transfer
  to DataDecide's grid.
- **TRJ-7 — Scale ladder.** Same decomposition across the DataDecide size
  ladder; does the drift-to-diffusion ratio improve with scale, and does it
  do so uniformly across tasks?

### Follow-up: routing dynamics as a movement channel (FLAME-MoE)

*Moved from the former standalone MoE-routing doc (`moe-routing.md`). Its F2
direction (flips by token entropy) is TOK-obs-5 in Token-level movement, `token-movement.md`, Stage 1. *

**One-line pitch.** Dense-model movement between checkpoints is continuous
drift that has to be extracted from KL or CKA. Mixture-of-experts models
expose a *categorical* channel — per-token expert assignments — and
FLAME-MoE (38M–1.7B active, 64 experts, top-8) releases checkpoints, routing
logs, and evals. Apply the drift/diffusion lens to routing flips: flips that
revert are wall oscillation, flips that persist are river movement, and
per-layer saturation curves are cumulative-commitment plots. A dense control
from DataDecide at matched active parameters separates "MoE" from "small."

**Compute tier.** T0 after ingest, if the released routing logs cover the
checkpoints and tokens needed. T1 if routing must be recomputed from
checkpoints over a probe corpus.

Core:

- **Ingest FLAME-MoE artifacts.** Routing logs (per token, per layer, top-k
  expert ids across checkpoints), eval results, checkpoint metadata and
  schedules. Verify format and coverage first; this determines the tier.
- **TRJ-moe-1 — Routing-flip drift/diffusion.** For each layer and checkpoint pair,
  compute per-token assignment flip rates; separate reverting flips (t →
  t+1 → back at t+2) from persistent ones; compute router saturation
  (overlap of top-k at step t with top-k at the final checkpoint) per layer
  over training. Fit the same drift/diffusion decomposition used for dense
  eval trajectories.
- **TRJ-moe-3 — Dense control ladder.** Run the dense drift/diffusion decomposition
  on DataDecide models at matched active parameter counts so each MoE
  finding has an "is this MoE or just small models" comparison.

Optional:

- **Commitment timing.** Per-layer saturation timestamps as a "commitment
  clock": deeper layers are reported to saturate faster; quantify, and
  relate to the schedule (`lr_at_step` equivalent for FLAME-MoE runs).
- **Routing vs. eval movement.** Does routing-flip mass predict eval-metric
  movement between the same checkpoints, and does it do so better than the
  dense proxies?
- **Scale ladder.** Repeat across the seven FLAME-MoE sizes.

## 2. Doability and impact

**Doability: high, with one gate.** The gate is temporal resolution: if
saves are thousands of steps apart, adjacent-checkpoint increments may be
drift-dominated and the diffusion estimate degenerate. Check spacing per
scale first; TRJ-5 is the fallback and adds an ingest. Everything else is
standard time-series estimation on small series, and the noise-floor
module is reusable by any other project.

Secondary risks: (i) three seeds is thin for per-recipe variance — pool and
state the limitation; (ii) drift is not constant over a cosine run, so the
model must allow a time-varying trend (windowed fits).

**Routing follow-up doability: medium, dominated by ingest uncertainty.**
Everything hinges on
what the released routing logs actually contain: which checkpoints, how many
tokens, whether token identities are recoverable (needed for TOK-obs-5 and for
per-token flip tracking across checkpoints rather than aggregate
histograms). If logs are aggregate-only, per-token flip tracking requires
recomputing routing from checkpoints (T1, and a new model-loading path
distinct from DataDecide's). Also a separate suite with its own training
recipe and data, so it does not share DataDecide's recipe axis; the
"recipe" question cannot be asked here.

**Impact per direction:**

| Direction | Impact | Why |
|-----------|--------|-----|
| TRJ-6 noise floor | **Medium-high** | Nobody publishes it; every later DataDecide paper would cite it. Not a headline alone. |
| TRJ-1 movement-SNR table | **High** | The central artifact; extends a NeurIPS 2025 framework from the same group with a new marginal. |
| TRJ-4 re-derivation | Medium (supporting) | Validation; reviewers expect it. |
| TRJ-2 LR test | **High** | A zero-training river-valley test is a clean, quotable figure; risk that LR and training progress are confounded within a single cosine run (both monotone), mitigated by comparing across scales with different schedule lengths. |
| TRJ-3 matched-loss signatures | **High if positive** | Direct thesis demonstration; risk of null at these scales. |
| TRJ-5 resolution transfer | Low (enabling) | Only matters if the gate fails. |
| TRJ-7 scale ladder | Medium | Natural figure, limited novelty. |
| *Routing follow-up:* TRJ-moe-1 routing drift/diffusion + saturation | **Medium-high** | Novel framing of existing logs; router saturation is known, reverting-vs-persistent decomposition is not. |
| *Routing follow-up:* TRJ-moe-3 dense control | Medium (supporting) | Required for credibility; not a result alone. |
| *Routing follow-up:* Commitment timing | Medium | Largely confirms published observations with better statistics. |
| *Routing follow-up:* Routing vs. eval movement | Medium-high | Practical: routing as a cheap, high-signal movement detector. |
| *Routing follow-up:* Scale ladder | Medium | Good figure, limited novelty. |

**Likely paper shape.** TRJ-6 + TRJ-1 + TRJ-4 as the core ("what lives inside the
noise term"), TRJ-2 as the headline figure, TRJ-3 as the high-variance bonus.

**Routing follow-up paper shape.** TRJ-moe-1 + TRJ-moe-3 as the core, TOK-obs-5 (Token-level movement, Stage 1) as the
headline if the logs support it. A real workshop paper, but one that stands
apart from the DataDecide line: it shares methodology with the trajectory
project, not data or story. Worth doing if the drift/diffusion machinery
already exists and the ingest proves cheap; otherwise defer.

## 3. Infrastructure sequence

1. **Trajectory accessor.** Thin view over the processed OLMES and PPL
   tables returning ordered series per (recipe, scale, seed, task, metric)
   with step, tokens, FLOPs, `lr_at_step`. Assert monotone steps and report
   spacing statistics per scale (the gate check).
2. **Noise-floor module (TRJ-6).** Pooled variance, heteroscedasticity test,
   windowed replicate estimate, item bootstrap over per-instance tables.
   Tested on synthetic series with known variance.
3. **Decomposition module (TRJ-1).** Increment statistics, variance-vs-lag
   fits, windowed drift+diffusion model; returns per-series parameters with
   uncertainty. Tested on simulated drift+OU series.
4. **Aggregation and SNR table (TRJ-1, TRJ-4).** Per benchmark × metric × scale
   summaries; comparison of metric families.
5. **LR regression (TRJ-2).** Per-window diffusion/drift vs. `lr_at_step`,
   across scales.
6. **Matched-loss pairing utility (TRJ-3).** Nearest-loss checkpoint per
   recipe × seed with tolerance reporting; signature comparison.
7. **Optional: Signal-and-Noise ingest (TRJ-5)** following the repo's
   download → preprocess → typed parquet pattern, scoped to the OLMo dense
   trajectories.

Steps 1–4 form the minimum paper; 5 and 6 are independent add-ons.

Routing follow-up sequence (after the steps above):

1. **Artifact survey.** Inspect FLAME-MoE release: routing-log schema,
   checkpoint coverage, token recoverability, eval table format. Decide
   T0 vs. T1 from this alone.
2. **Ingest.** Download → preprocess → typed parquet following the repo's
   existing pattern; routing logs as a long table (checkpoint, layer, token
   id/position, expert ids); evals into the same trajectory schema used for
   DataDecide so the dense accessors work unchanged.
3. **Flip and saturation metrics (TRJ-moe-1).** Per-layer flip rates with
   reverting/persistent split; saturation vs. final checkpoint; feed into
   the drift/diffusion decomposition module (shared with the trajectory
   project; build it there if it does not exist yet).
4. **Dense control (TRJ-moe-3).** Run the same decomposition on DataDecide models
   at matched active parameters using the existing processed tables.
5. **Optional: routing recomputation path** (MoE checkpoint loader +
   forward hooks) only if the logs are insufficient.

---

## 4. External assessments

Dated, attributed notes from external review conversations, recorded for consolidation — not
decisions. Only notes about this project are kept here.

### 2026-08-22 — the spread-to-noise correlation is TRJ in embryo; "meaningful crossing" = drift-attributable

From Danielle's reproduction of the DataDecide paper (summary in
`../topics/reference/datadecide-data-pipeline.md`): across 160 task/metric observations
the Spearman association between predictability and the spread-to-noise ratio was 0.798
(stable ~0.80 across adjacent checkpoints). The response's reading: this is the
movement-SNR thesis as a single correlation; the drift/diffusion decomposition is the
version that explains *why* — which variance components are mean-reverting and how the
ratio evolves within runs rather than across the surface. Good for TRJ-1/TRJ-6 (the
phenomenon is confirmed strong); framing pressure: the paper must deliver more than the
correlation the original authors already report. Second use: the reproduction's 15,523
crossings (all 300 pairs) vs. Danielle's bump plots showing a stable jittery ordering —
the noise-aware recount should define a meaningful crossing as one attributable to drift,
not diffusion, which makes TRJ's decomposition the tool for the IRT / data-card crossover
finding (see `irt-reanalysis.md` §4).

### 2026-08-22 — small-scale density and a retrain substrate

From a conversation on the data layer (record in
`../topics/reference/datadecide-data-pipeline.md`). Two points for this doc: (a) at
4M–8M the released runs have 5–10 checkpoints, so the drift/diffusion decomposition
cannot be fit per run there — dynamics claims live at the dense-save scales, small scales
get endpoint statistics only (this sharpens the 2026-08-21 spacing answer, not reverses
it); (b) a small heavily-instrumented many-seed retrain of a few recipes at the 2–4
smallest scales ("DataDecide-dense": dense checkpoints, true training loss, executed LR
schedule, data-order manifest, per-token probe losses) would restore density there and
give TRJ-6 real many-seed noise floors instead of the 3-seed pooled version. Not a
prerequisite for the T0 work; a candidate infrastructure investment shared with the
tiny-scale and recipe-featurization projects. Also noted: training loss appears in the
scaling-law checkpoint-loss table only for 150M–1B and sparsely; held-out CE is the
working "loss" for matched-loss pairing (TRJ-3), which is arguably the better definition
across recipes anyway.

### 2026-08-21 — two "top-N by workshop-paper likelihood × speed" lists

- **Ranked #1 in a top-3 list.** "Pure T0 on data you already have parsed. The methods are
  standard small-series time-series estimation. Critically, it's robust to outcome: the TRJ-6
  noise floor plus the movement-SNR table (TRJ-1) plus the Signal-and-Noise re-derivation
  (TRJ-4) form a paper even if the exciting bits (TRJ-2's LR test, TRJ-3's matched-loss
  signatures) come up null. The one gate — checkpoint spacing — can be checked in an afternoon
  before you commit. And the noise-floor module is the single most reusable artifact in the
  whole portfolio; every other project's credibility section needs it."
- **Ranked #2 in a top-5 list.** "TRJ-6 + TRJ-1 + TRJ-4, TRJ-2 as headline. Also pure T0,
  robust to outcome, and the noise-floor module (TRJ-6) is a public good every other project
  cites. One gate to check on day one: checkpoint spacing. The TRJ-2 river-valley LR test is
  the quotable figure if it works, but the paper stands without it."
- **Downstream use.** Results here "will tell you which recipes and steps are
  most schedule-sensitive, making the branch grid [for annealed readouts] smaller and better
  targeted."
- **Pairing.** The annealed-readouts T0 half (MPL correction + ranking-flip
  analysis) could be paired with this paper, since "they share the trajectory accessor and the
  noise floor."
- Checkpoint-spacing gate: resolved (see `docs/open-questions-answered.md`).

- **Overlap.** TRJ-3 matched-loss signatures and IRT recipe-DIF "are close enough
  that reviewers will ask why you need both" — decide which instrument makes the
  beyond-final-performance claim first.
- **n = 25.** TRJ-3 "currently treat[s] 25 recipes as exchangeable"; prefer
  within-family comparisons along a measured dose (the family-contrast framing from recipe
  featurization).
- **Feasibility role.** "The noise-floor module isn't just reusable hygiene; it's
  the thing that determines whether the whole program has publishable effect sizes. That's
  another argument for [this project] first: it's partly a feasibility study for everything
  else."

### 2026-08-21 — on the routing follow-up and MoE releases

- No public multi-recipe MoE suite exists: "FLAME-MoE is seven models from 38M to 1.7B active
  parameters — a *scale* ladder, one data recipe. OLMoE is one recipe… OpenMoE is one recipe.
  [The 2025–26 open-weights MoE wave] is open-*weights*, closed-data." (Unverified claims.)
  The recipe question cannot be asked with these artifacts — consistent with the follow-up's
  own caveat.
- Reframing: the follow-up "becomes 'the routing instrument' chapter of the same
  data-measurement program rather than a separate suite with a separate story" — MoE models
  "write part of [data-driven] structure down explicitly" where dense models force inference
  from KL/CKA/per-token loss. Cross-suite option: OLMoE vs. FLAME-MoE vs. OpenMoE "all have
  checkpoints and known data… enough to ask whether expert-specialization structure tracks
  corpus composition across independent training setups." The artifact survey remains step
  one. (Full discussion in `docs/potential-projs/moe-partitions.md`.)

### 2026-08-21 — on an MoE dual of the drift/diffusion decomposition

- In an MoE, change between checkpoints "decomposes *architecturally* into two channels:
  **rerouting** (same experts, different assignments) and **rewriting** (same assignments,
  different experts). You can compute this decomposition exactly: hold routing fixed at
  checkpoint t while using checkpoint t+1's experts, and vice versa, and attribute the output
  delta. That's the MoE dual of your drift/diffusion decomposition, and it's *causal by
  construction* rather than inferred from time-series statistics." Conjectured phenomenology:
  early training rerouting-dominated, late training rewriting-dominated, per-layer crossover
  as a commitment clock. Frozen-router branches as the causal control. TRJ-moe-1's
  reverting/persistent flip split "slots directly in here." (Full discussion in
  `docs/potential-projs/moe-partitions.md`.)

### 2026-08-21 — on tiny-scale measurement

- "The noise-floor module [TRJ-6] tells you the minimum detectable effect per scale, and the
  drift/diffusion SNR table [TRJ-1] tells you which metrics carry signal down there" — the
  instruments for the 10–150M range, where "most benchmark items sit at chance, accuracy is
  quantized into a few reachable values, and seed variance swamps treatment effects." (Full
  discussion in `docs/potential-projs/tiny-scale-measurement.md`.)

### 2026-08-21 — positions in three ranked lists (full lists in `docs/portfolio-rankings.md`)

- **6–12-month flagship list: Tier 3 (component)**: "under the workshop lens this was my #1;
  under this lens it's the measurement/noise-floor section of whichever flagship you pick,
  plus the churn figures. Its standalone ceiling is a good-but-forgettable methods paper."
- **Workshop-sized list: #3**: "Also pure T0… robust to outcome — the movement-SNR table is
  the artifact regardless of what it shows, and the noise floor is a citable public good on
  its own… Slightly slower than IRT only because the windowed drift+diffusion modeling and
  its validation on synthetic series involve more judgment calls."
- **Full-conference list: #3**, "Anatomy of the Noise Term" (TRJ-6 + TRJ-1 + TRJ-4 + TRJ-2 +
  item churn): "*Speed:* T0, but slower than [IRT] because the windowed drift+diffusion
  modeling requires real iteration and the LR test has a known confound… Pivot is clear: the
  SNR table + noise floors stand without the river-valley claim. **Expected impact:
  medium-high.** **Ceiling: high** — a clean zero-training confirmation of river-valley
  structure in public data is a widely quotable figure." Overlap warning with the IRT paper;
  a scoop race.

### 2026-08-18 — a control to add to TRJ-3's matched-loss pairing (from the Research Trajectory page)

- "Matched-loss pairs have a hidden confound: equal loss at different token counts vs. equal
  tokens at different loss are different controls, and you'll want both, since 'recipe A
  reaches this loss faster' and 'recipe A has better [signature] at this loss' are separable
  claims." The pairing utility should emit both pair types.

### 2026-08-18 — a control to add to the noise floor (from the Research Trajectory page)

- The movement-microscope design's null distribution for post-training movement includes,
  "crucially, *continued pretraining on the same pretraining data* for the same token
  budget as the post-training would use." TRJ-6's floors (seed variance, windowed
  replicates, item bootstrap) are the between-checkpoint analogue; the token-exposure
  control is the piece to add when the floor is used to judge any intervention that adds
  training. "Movement that doesn't exceed seed-noise-plus-token-exposure isn't movement."
  See `docs/potential-projs/movement-microscope.md`.

### 2026-08-18 — origin of this project (from the Research Trajectory page)

Question posed: start by looking for changes between the provided pretraining checkpoints,
referencing the Signal-and-Noise paper.

- "Your proposed project is literally its dual. Heineman et al. define signal as a
  benchmark's ability to separate better models from worse models, and noise as a
  benchmark's sensitivity to random variability between training steps — i.e., in their
  framework, *checkpoint-to-checkpoint movement is the nuisance term*… Your project inverts
  the roles: treat inter-checkpoint change as the *object*, and ask what structure lives
  inside what they averaged away as noise. Same data, opposite marginal." Their release:
  "900K evaluation results on 465 open-weight models, including evaluations across
  intermediate checkpoints of OLMo, the DataDecide suite, and the ladder models."
- Stage 0 as first stated (TRJ-1, TRJ-4, TRJ-2, TRJ-3): "'Variability between training
  steps' conflates two processes with opposite meanings: stochastic jitter (the checkpoint
  oscillating on the valley walls — order-free, mean-reverting) and genuine learning
  (progress along the river — directional, cumulative). These are separable from a
  trajectory alone: autocorrelation structure, sign-consistency of increments, variance
  growth with lag (diffusion grows linearly in lag; drift grows quadratically)… you'd
  predict continuous metrics have high drift-to-diffusion ratio, accuracy metrics low — and
  any benchmark where *recipes differ in drift/diffusion signature at matched loss* is your
  thesis phenomenon appearing in data someone else already published… diffusion magnitude
  should scale with the current LR (walls), drift should not (river), and the cosine
  schedule's gradual decay gives you a natural LR sweep *within* each trajectory."
- Origin of TRJ-5: "DataDecide checkpoints are relatively sparse, so adjacent-pair
  statistics at small k may be dominated by drift… the OLMo trajectories in the same
  release are denser and better for the diffusion estimates, then transfer the fitted
  decomposition." (Spacing since measured: ~1,000–1,300 steps at 8M–530M; see
  `docs/open-questions-answered.md`.)
- Role in the program: "the drift/diffusion fits are the noise floor for any future
  post-training movement claim."

### 2026-08-18 — origin of the routing follow-up (from the Research Trajectory page)

- "Dense-model movement between checkpoints is continuous drift you have to dig out of KL
  and CKA; MoE gives you *categorical* movement — per-token expert-assignment flips… Your
  drift/diffusion decomposition applies directly: routing flips that revert are wall
  oscillation, flips that persist are river movement, and saturation curves are literally
  cumulative-commitment plots." Router saturation (OLMoE) is the field's existing metric;
  FLAME-MoE's released routing logs across checkpoints mean "your Stage-0 (zero-training
  analysis of public trajectories) extends to MoE immediately." Paper list in
  `docs/topics/reference/moe-literature.md`.

### 2026-08-18 — origin of TRJ-6's three components (from the Research Trajectory page)

Question posed: will evaluating the same checkpoint with different random seeds give the
same or different results — because 3 seeds is a small n for averages?

- "Same checkpoint, different eval seeds: mostly deterministic." For loglikelihood /
  rank-classification evals "the only randomness is floating-point non-associativity from
  batching/kernel scheduling… typically negligible"; generation-based evals vary but are
  not most of the suite; configuration variance (few-shot demo choice, order, template) "is
  large, but it's *systematic*, not random: it shifts all models coherently, so it's a bias
  axis to sweep, not noise to average." "Almost all of the variance you care about isn't in
  the eval — it's in *training* (seed, data order, init)."
- The three mitigations that became TRJ-6: "**pool**: assume (and then test) that
  seed-variance is roughly homoscedastic across recipes at fixed scale — 25 recipes × 3
  seeds gives you ~50 degrees of freedom for a *shared* variance estimate… a recipe whose
  seeds diverge more is a finding, not a nuisance"; "**use the trajectory as replicate** —
  Signal and Noise's own trick… their windowed noise estimate assumes the drift within the
  window is negligible; your decomposition checks and corrects that assumption";
  "**bootstrap over items** for the orthogonal axis — benchmark-composition uncertainty
  from a single run." "With those three, n=3 stops being a blocker for anything except
  recipe-specific variance claims."

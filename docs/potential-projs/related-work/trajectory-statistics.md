# trajectory statistics — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`trajectory-statistics.md`](../trajectory-statistics.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

High-recall corpus for trajectory drift/diffusion (`TRJ-1`–`TRJ-7`, `TRJ-moe-*`). Every
item is on record in this repository and cites its source. **Nothing here is verified**:
the accumulators are agent-generated, the citation ledger states none of its rows has
been checked (the Signal-and-Noise identifier in particular is a Claude-added,
hallucination-prone row), and the in-repo reproduction numbers carry Danielle's explicit
caveat about agent-written verification code.

**The direct dual and the evaluation-methodology frame**

- **Heineman et al., *Signal and Noise: A Framework for Reducing Uncertainty in Language
  Model Evaluation*** (NeurIPS 2025 per the record; 2508.13144 per the ledger, unverified
  and Claude-added) — signal = a benchmark's ability to separate better from worse
  models; noise = sensitivity to random variability between training steps; interventions
  = continuous metrics beat accuracy on both, and filtering noisy subtasks improves
  aggregate reliability. Release: ~900K evaluation results on 465 open-weight models
  including OLMo intermediate checkpoints, DataDecide, and the ladder runs. This project
  is recorded as its dual — same data, opposite marginal — and TRJ-4 re-derives its
  metric-choice finding. Their windowed noise estimate assumes within-window drift is
  negligible; TRJ-1 checks and corrects that assumption. (source:
  docs/topics/reference/evaluation-methodology-literature.md;
  docs/potential-projs/trajectory-statistics.md §4 origin;
  docs/litreview/citation-verification-ledger.md)
- **OLMES, *A Standard for Language Model Evaluations*** (no ID on record) — the eval
  standard behind the processed tables, and the citation for the record that
  re-evaluating a fixed checkpoint with new seeds buys nothing (loglikelihood evals are
  effectively deterministic; generation evals are the exception; few-shot configuration
  variance is "a bias axis to sweep, not noise to average"). This is what fixes TRJ-6's
  three components. (source:
  docs/topics/reference/evaluation-methodology-literature.md;
  docs/potential-projs/trajectory-statistics.md §4 2026-08-18)
- **DataDecide**, Magnusson et al. (Ai2, ICML 2025; 2504.11393) — the suite whose
  trajectories are analyzed: 25 corpora, sizes to 1B, 3 seeds, 100B tokens; single-150M
  ranking predicts the 1B best dataset ~80% of the time; continuous likelihood metrics as
  low-noise proxies (the finding TRJ-4 should recover as "drop low-ratio tasks").
  (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/litreview/citation-verification-ledger.md)
- **Patel et al. 2026, *Forecasting Downstream Performance of LLMs With Proxy Metrics***
  (2605.18607) — 80 proxy metrics from token-level statistics × weighting schemes; on
  DataDecide, frequency-weighted top-5 accuracy ranks the 25 corpora at ~1e-5 of target
  compute; along OLMo-3-7B, proxies extrapolate downstream accuracy over an 18× horizon
  at half the RMSE of loss baselines. A competing account of which observables carry
  trajectory signal. Version 2 of that review fabricated the author list; prefer v1's
  bibliography. (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md)
- **Metric-family context from the same review** — NeuNeu (2601.19831,
  accuracy-trajectory extrapolation with token-level validation losses); FLP (2410.08527);
  model ladders (2412.04403); observational scaling laws (2405.10938); Ye et al. BIG-bench
  predictability (2305.14947); Pechi et al. small-scale break below ~2.2e15 FLOPs
  (2305.17266); Informedness over accuracy/F1 (2401.03831); generative→NLU reformulation
  (2506.03592). All bear on which metric families should show high drift-to-diffusion.
  (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md)
- **Emergence framing** — Wei et al. (2206.07682) vs. Schaeffer et al. mirage
  (2304.15004); proxy tasks for emergent abilities (2412.07111). Relevant to whether a
  low drift-to-diffusion ratio on accuracy is a property of the metric or the model.
  (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md)

**The river-valley claim TRJ-2 tests without training**

- **Wen et al., *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss
  Landscape View*** (2410.05192) — the canonical statement: high-LR phase drives
  along-river progress, decay drives the mountain direction; the interpolation signature
  (convex/unimodal between stable-phase checkpoints, smooth monotone between decay-phase
  ones) is "the closest thing to a river test"; they attribute the valley geometry to
  token determinism, making it plausibly recipe-dependent. TRJ-2's prediction —
  diffusion scales with `lr_at_step`, drift does not — is this claim in trajectory
  statistics. (source: docs/topics/reference/landscape-literature.md;
  docs/topics/reference/token-level-literature.md)
- **Wen et al.'s own validation** — a toy bigram language of varying token determinism
  reproduces the geometry; stable phase learns deterministic tokens, decay phase learns
  stochastic ones; Spearman ≈0.39 between token uncertainty and local sharpness on real
  data; and the constant-LR branch-and-interpolate experiment. (source:
  docs/topics/reference/landscape-literature.md;
  docs/topics/staging/checkpoint-tomography.md)
- ***Scaling with Collapse: Efficient and Predictable Training of LLM Families***
  (2509.25087) — well-tuned runs' loss curves collapse onto a shared shape: a curves-only
  cross-run comparability criterion needing no weight access, and a natural companion to
  a curves-only drift/diffusion analysis. (source:
  docs/topics/reference/landscape-literature.md)
- ***Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning Rate
  Scheduler*** (no ID on record) — visualizes the landscape in pre-cooldown→final vs.
  local-Adam-step coordinates. (source:
  docs/topics/reference/landscape-literature.md)
- **Multi-power law, Luo et al.** (2503.12811) — the decay-drop term read in river-valley
  language as descending from the walls to the river; the analytic counterpart of the
  drift/diffusion split, and the reason "loss" conflates along-river progress with
  distance-from-river. (source: docs/topics/reference/landscape-literature.md;
  docs/topics/reference/loss-curve-forecasting.md)
- **Tissue et al.** (2408.11029) — annealing-area term in the loss law: the LR trajectory
  affects realized loss beyond total tokens, i.e. an explicit schedule term in the
  observable TRJ regresses against `lr_at_step`. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **Hägele et al.** (2405.18392) and **MiniCPM** (2404.06395) — the WSD methodology whose
  stable/decay split is the phenomenon TRJ-2 detects inside a cosine run; MiniCPM's
  decay-phase statistics (gradient norm falls, consecutive-update cosine turns positive)
  are the update-space analogue of "drift dominates, diffusion collapses". (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **The recorded TRJ-2 confound** — LR and training progress are both monotone within a
  single cosine run, mitigated by comparing across scales with different schedule
  lengths. (source: docs/potential-projs/trajectory-statistics.md §2, §4 2026-08-21)

**Boundary conditions on matched-loss comparison (TRJ-3)**

- **Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic
  Datasets*** (no ID on record) — train loss at floor and test loss at chance while the
  generalizing circuit assembles invisibly. (source:
  docs/topics/reference/grokking-and-hidden-progress.md)
- **Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability*** (no
  ID on record) — the mechanism, and the argument that the field needed non-loss
  observables. Together these make matched-loss pairs "a necessary but provably
  insufficient control": two checkpoints matched on train *and* test loss can sit at
  different points of hidden circuit maturity. (source:
  docs/topics/reference/grokking-and-hidden-progress.md)
- **Nakkiran et al., *Deep Double Descent*** (no ID on record) — epoch-wise double
  descent means capability is not monotone in training loss along a run: the
  non-monotonicity warning on any matched-loss pairing. (source:
  docs/topics/reference/grokking-and-hidden-progress.md)
- ***What Can Grokking Teach Us About Learning Under Non-Stationarity*** (no ID on
  record) — warm-starting gap and grokking as one delayed-generalization phenomenon with
  effective LR as the shared knob. (source:
  docs/topics/reference/grokking-and-hidden-progress.md)
- **The equal-loss vs. equal-tokens control (2026-08-18 §4)** — matched-loss pairs have a
  hidden confound; "recipe A reaches this loss faster" and "recipe A has a better
  signature at this loss" are separable claims, so the pairing utility should emit both
  pair types. (source: docs/potential-projs/trajectory-statistics.md §4)
- **Basin comparability toolkit** — Frankle et al. LMC; Entezari et al. permutation
  invariance; Ainsworth et al. *Git Re-Basin*; layerwise linear feature connectivity;
  cross-task linearity in pretraining–finetuning; model soups (2203.05482); *Beyond
  Structural Symmetries: LMC via Neuron Identifiability* (2026); and Juneja et al.,
  *Linear Connectivity Reveals Generalization Strategies* (ICLR 2023) — models in
  different basins implement different generalization strategies at similar
  in-distribution accuracy. The recorded thought: nobody has connected either literature
  to *metric validity*. This is the strongest version of "matched loss may not mean
  comparable". (source: docs/topics/reference/landscape-literature.md)
- **Held-out CE as the working "loss" for pairing (in-repo)** — training loss appears in
  the scaling-law checkpoint-loss table only for 150M–1B and sparsely, so held-out CE is
  the working matched-loss variable and is arguably the better definition across recipes.
  (source: docs/potential-projs/trajectory-statistics.md §4 2026-08-22;
  docs/topics/reference/datadecide-data-pipeline.md)

**Noise-floor prior art and the overlap with IRT (TRJ-6)**

- **Signal and Noise's trajectory-as-replicate trick** (above) — the windowed noise
  estimate TRJ-6(b) reuses and corrects for within-window drift. (source:
  docs/potential-projs/trajectory-statistics.md §4 2026-08-18)
- **IRT accumulator** — Lalor et al., *Building an Evaluation Scale Using Item Response
  Theory*; Rodriguez et al., *Evaluation Examples Are Not Equally Informative* (ACL
  2021); Polo et al., *tinyBenchmarks* (ICML 2024, IRT-selected ~100-item subsets
  preserving rankings); *metabench*. Recorded overlap: TRJ-3's matched-loss signatures
  and recipe-DIF are "close enough that reviewers will ask why you need both", and the
  binary-vs-continuous IRT comparison would replicate Signal-and-Noise's metric-choice
  finding inside one framework — i.e. a second route to TRJ-4. Also the local-independence
  caution (shared passages, contamination). (source:
  docs/topics/reference/irt-literature.md;
  docs/potential-projs/trajectory-statistics.md §4 2026-08-21)
- **Per-instance coverage fact (in-repo)** — item bootstrap, IRT dimensionality and
  recipe-DIF are fully supported at 150M–1B with 3 seeds; below 150M there is one seed
  per cell, so seed-variance estimates are unavailable and pooled-across-recipe floors
  must be used. This bounds TRJ-6(a) and TRJ-6(c). (source:
  docs/open-questions-answered.md 2026-08-21)
- **Checkpoint-spacing table (in-repo, the resolved gate)** — ~1,000–1,300-step spacing
  from 8M to 530M with 30–40 checkpoints at 150M–530M; 1B coarsest (~2,500 steps, 27
  points); sub-10M sizes have too few points to fit per-run. TRJ-5 is therefore not a
  prerequisite, only an optional robustness check. (source:
  docs/open-questions-answered.md 2026-08-21;
  docs/potential-projs/trajectory-statistics.md §4)
- **750M truncation (in-repo)** — `olmes.parquet` stops at step 26,250 (22 checkpoints)
  while the instance table reaches 63,599 (54 checkpoints), so 750M trajectory analysis
  must be rebuilt from instances. (source: docs/open-questions-answered.md)
- **The movement-microscope null-distribution control** — the token-exposure control
  (continued pretraining on the same data for the same token budget) is the piece to add
  when TRJ-6's floors are used to judge any intervention that adds training: "movement
  that doesn't exceed seed-noise-plus-token-exposure isn't movement." (source:
  docs/potential-projs/trajectory-statistics.md §4 2026-08-18;
  docs/potential-projs/movement-microscope.md)
- **DataDecide-dense** (staging, no external ID) — the many-seed retrain substrate that
  would give TRJ-6 real many-seed floors instead of the 3-seed pooled version, and
  restore density below 20M; not a prerequisite for the T0 work. (source:
  docs/topics/staging/datadecide-dense.md;
  docs/potential-projs/trajectory-statistics.md §4 2026-08-22)

**In-repo empirical context (reproduction records, not literature)**

- **Spread-to-noise correlation** — Spearman ≈0.798 between predictability and the
  spread-to-noise ratio across 160 task/metric observations, stable ~0.80 across adjacent
  checkpoints. Read in §4 as "the movement-SNR thesis as a single correlation", with the
  framing pressure that the paper must deliver more than the correlation the original
  authors already report. Carries Danielle's provenance caveat. (source:
  docs/potential-projs/trajectory-statistics.md §4 2026-08-22;
  docs/topics/reference/datadecide-data-pipeline.md)
- **The 15,523-crossings recount** — all 300 pairs cross at least once, against
  Danielle's bump plots showing a stable jittery ordering; the noise-aware recount should
  define a meaningful crossing as drift-attributable, not diffusion — which makes TRJ's
  decomposition the tool for the IRT / data-card crossover finding. Seed SD ~0.02 for
  some recipes on 7/10 tasks, max 0.111. Same provenance caveat. (source:
  docs/potential-projs/trajectory-statistics.md §4 2026-08-22;
  docs/topics/reference/datadecide-data-pipeline.md;
  docs/potential-projs/irt-reanalysis.md §4)
- **The earliest written statement of the decomposition** — idea-map I1: separate each
  benchmark×recipe×scale trajectory into directional drift and mean-reverting diffusion
  via autocorrelation, increment sign-consistency and variance-vs-lag scaling (diffusion
  ∝ lag, drift ∝ lag²); tests, feeds, and the sparse-checkpoint design note. N1 is the
  noise-floor item, I2 the IRT item. (source:
  docs/dataset-analysis-idea-map.md §L2)

**Routing follow-up prior art (TRJ-moe-1, TRJ-moe-3)**

- **FLAME-MoE, *A Transparent End-to-End Research Platform for Mixture-of-Experts
  Language Models*** (no ID on record) — "DataDecide-for-MoE": seven decoder-only models
  38M–1.7B active, 64 experts, top-8, with open code, data, checkpoints, routing logs and
  eval results; training traces show expert specialization emerging early and
  intensifying, co-activation sparse and stable, routing converging quickly early. The
  substrate for the whole follow-up. (source:
  docs/topics/reference/moe-literature.md;
  docs/potential-projs/trajectory-statistics.md §4 2026-08-18)
- **OLMoE, *Open Mixture-of-Experts Language Models*** (no ID on record) — router
  saturation defined as average overlap between top-k experts at step t versus at
  convergence, rising sharply within the first few thousand steps, deeper layers
  saturating faster. The field's existing metric that TRJ-moe-1's reverting-vs-persistent
  split extends. (source: docs/topics/reference/moe-literature.md)
- ***Three Phases of Expert Routing: How Load Balance Evolves During Mixture-of-Experts
  Training*** (no ID on record) — an early balance-prioritizing phase, a stabilization
  phase where experts specialize, and a late relaxation phase trading balance for
  quality; non-monotone and invisible to post-hoc analysis of converged models, with
  annealing checkpoints confirming the phases are pretraining-specific. The closest
  existing "routing dynamics over training" result. (source:
  docs/topics/reference/moe-literature.md)
- ***Continual Pre-training of MoEs: How Robust Is Your Router?*** (no ID on record) —
  routing changes most in early layers; the no-replay condition shows the most dramatic
  early-layer reorganization and the most forgetting. (source:
  docs/topics/reference/moe-literature.md)
- ***The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not
  Necessarily Domain Expertise*** (no ID on record) — specialization resists human
  interpretation; expert overlap between different models answering the same question is
  no higher than between different questions, so independently trained MoEs pick
  unrelated specializations; routers are linear maps so hidden-state similarity explains
  expert-usage similarity; load-balancing loss provably suppresses shared hidden
  directions. The comparability warning for any cross-run routing statistic. (source:
  docs/topics/reference/moe-literature.md)
- **OpenMoE** (no ID on record) — reported (unverified) to decide token-to-expert
  assignments early and keep them fixed; the origin of the frozen-routing hypothesis and
  a boundary case for reverting-vs-persistent flips. (source:
  docs/topics/reference/nonstationarity-accounting.md;
  docs/potential-projs/trajectory-statistics.md §4 2026-08-21)
- **The recorded gap** — no public multi-recipe MoE suite exists: FLAME-MoE is a scale
  ladder on one recipe, OLMoE one recipe, OpenMoE one recipe, and the 2025–26
  open-weights wave is closed-data, so the recipe question cannot be asked with these
  artifacts. Recorded as unverified claims. (source:
  docs/topics/reference/moe-literature.md;
  docs/potential-projs/trajectory-statistics.md §4)
- **The reroute-vs-rewrite decomposition** (repo design, no external ID) — hold routing
  fixed at checkpoint t while using t+1's experts and vice versa, attributing the output
  delta: the MoE dual of drift/diffusion, causal by construction rather than inferred
  from time-series statistics; conjectured early-rerouting/late-rewriting phenomenology
  with a per-layer crossover as a commitment clock, and frozen-router branches as the
  causal control. TRJ-moe-1's flip split slots into it. (source:
  docs/potential-projs/trajectory-statistics.md §4 2026-08-21;
  docs/potential-projs/moe-partitions.md)
- **MoE comparability warning for dense tools** — expert permutations extend the symmetry
  group, so interpolation barriers, checkpoint merging and stitching all need an
  expert-alignment step, and re-basin for MoE is immature. Also: MoE knobs are
  folklore-tuned at large scale and may be mis-set at 20–50M active; routing discreteness
  plausibly adds eval variance, making the noise-floor stage more necessary, not less;
  keep a dense control ladder at matched active parameters (TRJ-moe-3). (source:
  docs/topics/reference/moe-literature.md)
- **MoE design-space rows in the ledger, tagged for the MoE projects** — Model soups
  2203.05482; RouteLLM 2406.18665; MIMONets 2312.02829; MatFormer 2310.07707;
  Weight-Ensembling MoE 2402.00433; Higher Layers Need More LoRA Experts 2402.08562;
  MixLoRA 2404.15159; SwitchHead 2312.07987; Mixture-of-Depths 2404.02258; Soft MoE
  2308.00951. Peripheral to TRJ but on record for the routing follow-up's design space.
  (source: docs/litreview/citation-verification-ledger.md)

**Program frame and adjacent instruments**

- **Non-stationarity accounting** (repo reference topic) — the LR schedule as *exogenous*
  non-stationarity and the drift/diffusion split as one of its accounting instruments;
  the endogenous counterpart is the model's own implicit self-curriculum (as easy tokens
  saturate, learning signal migrates to harder ones), which the Rho-1-style taxonomy and
  river/wall token migration measure. Also the ITER record (Igl et al., ICLR 2021:
  transient non-stationarity permanently scars the representation) and the Achille 2019 /
  Ash & Adams 2020 / Igl 2021 "three communities, one claim" reading. (source:
  docs/topics/reference/nonstationarity-accounting.md)
- **Rho-1, *Not All Tokens Are What You Need for Pretraining*** (no ID on record) —
  token loss-trajectory taxonomy across checkpoints; the token-level analogue of the
  drift/diffusion split. (source: docs/topics/reference/token-level-literature.md)
- ***Token-Level Uncertainty-Aware Objective for Language Model Post-Training*** (no ID
  on record) — epistemic vs. aleatoric token uncertainty; epistemic drains faster for
  low-aleatoric tokens, so a token's apparent bucket migrates over training. The
  token-level statement of "drift vs. transient position". (source:
  docs/topics/reference/token-level-literature.md)
- **Checkpoint tomography** (staging, no external ID) — the hot branch (constant-LR
  continuation) is explicitly the *causal* version of TRJ's diffusion width, the twin
  branch the causal version of basin commitment; the battery's per-checkpoint statistics
  are the outcome variables a movement study would regress on. (source:
  docs/topics/staging/checkpoint-tomography.md)
- **Frankle et al. LMC and the sibling-barrier commitment clock; devinterp / local
  learning coefficient (Lau, Murfet et al.; Timaeus); the critical-sharpness statistic;
  the basin-emergence line; PolyPythias (2503.09543)** — the branch- and geometry-side
  instruments that measure the same commitment/movement questions with training rather
  than time-series statistics. (source:
  docs/topics/staging/checkpoint-tomography.md;
  docs/potential-projs/embedding-reset-dynamics.md)
- **Annealed readouts' durable-movement operator (ANN-opt-7)** — durable movement defined
  as change that persists under the schedule-neutralizing transform (merge or short decay
  branch at t and t+k, compare the *annealed* models), with the Signal-and-Noise noise
  term decomposing into measurement noise + wall oscillation + unresolved drift. The
  causal counterpart of TRJ-1's statistical decomposition, and the reason the two
  projects are recorded as pairing (shared trajectory accessor and noise floor). (source:
  docs/potential-projs/annealed-readouts.md §4 2026-08-18;
  docs/refs/research-trajectory-pre-to-post-training.md;
  docs/potential-projs/trajectory-statistics.md §4 2026-08-21)
- **Learning-curve extrapolation lineage** — Domhan, Springenberg & Hutter 2015; Klein et
  al. 2017; Baker et al. 2017; LC-PFN (2310.20447); Ding et al. (2412.15554);
  neural capacitance (2201.04194); zero-cost NAS proxies (AZ-NAS 2403.19232; FreeREA).
  The other tradition of fitting statistics to partial curves; recorded for the
  early-dynamics project but the same series-modeling machinery. Also the recorded
  finding that no literature supports the "more linear loss curves = better training"
  hypothesis. (source: docs/topics/reference/loss-curve-forecasting.md;
  docs/topics/reference/nas-literature.md; docs/past-projects/loss-slope-prediction.md)
- **Estimation and calibration methods** (repo reference topic) — bootstrap intervals
  (resample programs/items, never individual outcomes), block bootstrap when batches are
  correlated, analytic SE, empirical Bernstein bounds, Wilson/Jeffreys for binary,
  conformal prediction; the toolbox TRJ-6's item bootstrap and confidence intervals draw
  on. (source: docs/topics/reference/estimation-and-calibration-methods.md)

**Standing caveats to carry**

- The ledger's Signal-and-Noise identifier 2508.13144 is a Claude-added row, flagged as
  hallucination-prone in last digits and title–ID pairing; nothing in the ledger is
  verified. (source: docs/litreview/citation-verification-ledger.md)
- The 2026-08-22 spread-to-noise and crossings entries carry Danielle's explicit caveat
  that the numbers come from agent-written verification code she has not read, debugged,
  run, or analyzed — flags for where to look, not findings. (source:
  docs/potential-projs/trajectory-statistics.md §4;
  docs/topics/reference/datadecide-data-pipeline.md)
- The MoE "no public multi-recipe suite" claims and the OpenMoE frozen-routing report are
  marked unverified in their own files. (source:
  docs/topics/reference/moe-literature.md;
  docs/topics/reference/nonstationarity-accounting.md)

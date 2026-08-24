# token movement — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`token-movement.md`](../token-movement.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

High-recall corpus for TOK (token-level movement, observational Stage 1 + causal Stage 2).
Every paper, method, or named prior-art item on record anywhere in this repository that is
*possibly* relevant to per-token movement, entropy buckets, decay-responsiveness, and the
recipe question. Grouped by theme; one line per item with its repo source. Err toward
inclusion. No positioning claims.

**The anchor: the river-valley account and its static validation**

- **Wen et al., Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss
  Landscape View** (arXiv 2410.05192) — the canonical statement: deterministic tokens
  contribute the river direction, uncertain/ambiguous tokens create the steep hillsides;
  the stable phase learns the former and decay the latter; the toy bigram language (origin
  of TOK-opt-5); correlational real-data validation (Spearman ~0.39 between token-level
  uncertainty and local sharpness); interpolation as the "river test" (convex/unimodal
  between stable-phase pairs, smooth monotone between decay-phase pairs); WSD-S built from
  resuming decayed checkpoints — (source:
  docs/topics/reference/landscape-literature.md; token-level-literature.md;
  docs/topics/staging/checkpoint-tomography.md).
- **The recorded verdict on the gap** — "the mapping has been made statically, and there
  are fragments of the dynamics, but the full 'watch the bucket assignment evolve over
  training' study doesn't exist"; Wen et al. treat the mapping as a fixed data property —
  (source: docs/topics/reference/token-level-literature.md).
- **Valley geometry as a data property** — the river-valley hypothesis attributes the
  geometry itself to data properties, making a corpus's "determinism profile" (cheap with
  any reference model) a candidate predictor of landscape geometry, i.e. TOK-opt-1's
  statistic doubles as a corpus feature — (source:
  docs/topics/reference/landscape-literature.md, 2026-08-18 WSD/featurization entry).
- **Training Dynamics of the Cooldown Stage in WSD** (no ID on record) — plots the
  landscape in pre-cooldown→final vs. local-Adam-step coordinates, noting a clear river-
  valley visualization had been lacking — (source:
  docs/topics/reference/landscape-literature.md).
- **Scaling with Collapse: Efficient and Predictable Training of LLM Families** (arXiv
  2509.25087) — well-tuned runs' loss curves collapse onto a shared shape; a cross-run
  comparability criterion from curves alone, relevant to TOK's matched-loss pairing —
  (source: docs/topics/reference/landscape-literature.md).
- **The multi-power law** (Luo et al., arXiv 2503.12811) — its "decay-induced loss drop"
  term is, in river-valley language, descending from the walls to the river: the scalar
  version of TOK-3's per-token drop — (source:
  docs/topics/reference/landscape-literature.md).
- **Hägele et al. / MiniCPM annealing-branch methodology** (no IDs on record) — the same
  branch-and-decay instrument used for scaling-law fitting; establishes "branch + decay +
  measure the loss drop," with the per-token profile as the unclaimed part — (source:
  docs/topics/staging/checkpoint-tomography.md).

**Uncertainty decomposition (TOK-2)**

- **Token-Level Uncertainty-Aware Objective for Language Model Post-Training** (no ID on
  record) — epistemic vs. aleatoric per token; epistemic drains faster for low-aleatoric
  examples; recorded as "the closest existing measurement" of the migration TOK-4 tracks,
  "though it never connects to landscape geometry" — (source:
  docs/topics/reference/token-level-literature.md).
- **The aleatoric/epistemic reading of the river-valley picture** — aleatoric is the true
  hillside (fixed data property); epistemic is distance-not-yet-traveled along the river;
  "the mapping changing over training is largely the epistemic component collapsing at
  token-dependent rates" — the interpretive frame behind TOK-2 — (source:
  docs/potential-projs/token-movement.md §4; token-level-literature.md).

**Loss-trajectory taxonomies and token-selection / reweighting objectives (TOK-opt-2,
TOK-opt-6) — the training-side mirror; from a SciSpace review, characterizations
agent-generated and identifiers unverified**

- **Rho-1 / selective LM, Not All Tokens Are What You Need for Pretraining** (arXiv
  2404.07965) — reference-model excess-loss scoring, train on the top tokens (15B
  OpenWebMath, 80B general); its loss-trajectory categories (persistently high/low,
  descending, fluctuating/ascending) are "literally a token-bucket-over-time taxonomy…
  with no landscape interpretation attached" — origin of TOK-opt-2 and half of TOK-opt-6
  — (source: docs/topics/reference/token-level-literature.md;
  training-objective-alternatives-literature.md).
- **MiLe** (Findings NAACL 2024, no ID on record) — scale loss by predictive entropy;
  468M–6.7B on the Pile; the entropy-based prescription whose descriptive counterpart is
  TOK-obs-4 — (source: docs/topics/reference/training-objective-alternatives-literature.md).
- **ESLM** (arXiv 2505.19893) — value-at-risk thresholding on per-token loss per batch,
  recovering CVaR minimization; GPT-2 pretraining FLOP savings — (source: same).
- **VCORE** (arXiv 2510.27462) — closed-form Gibbs weights from a one-backward probe,
  variance-controlled gradient utility — (source: same).
- **TALR** (arXiv 2509.20758) — w ∝ p(x)^(1/τ), a curriculum downweighting hard tokens —
  (source: same).
- **RFT** (arXiv 2412.14780) — reasoning vs. boilerplate tokens by relative loss —
  (source: same).
- **IR-DRO** (arXiv 2402.14270) — keep moderately-high-loss samples, drop the highest as
  noise — (source: same).
- **Power-Law Decay Loss** (arXiv 2505.16900) — frequency/information-based reweighting —
  (source: same).
- **Multi-token prediction** (Gloeckle et al., arXiv 2404.19737) — implicitly upweights
  "choice-point" tokens; the choice-point notion is the wall bucket by another name;
  gains on code, up to 13B — (source: same).
- **MTP curricula** (arXiv 2505.22757) — (source: same).
- **Patch-level training** (arXiv 2407.12665) — predict K-token patches, ~50% cost
  reduction at matched loss; changes what "a token" is for movement accounting — (source:
  same).
- **"Filling the mutual-information gap"** (arXiv 2511.00198) — (source: same).
- **Beyond Log Likelihood** (Li et al., arXiv 2510.00526) — the f_α(p) = (1−p^α)/α family
  with NLL at α→0 and a *model-capability continuum* (prior-leaning objectives win at the
  model-strong end, NLL at the model-weak end); flagged as the single most
  decision-relevant entry and a DataDecide-shaped size-ladder claim — (source: same).
- **MixCE** (arXiv 2305.16958) — forward + reverse CE — (source: same).
- **Strictly proper scoring rules** (Shao et al., arXiv 2405.18906) — Brier/spherical at
  token level with smoothing — (source: same).
- **Focal / Lovász / Dice CV-inspired losses** (Cambrin et al., arXiv 2409.13641) —
  (source: same).
- **Contrastive token learning for degeneration** (arXiv 2205.02517) — (source: same).
- **ScaleGrad** (2021, no ID on record) — gradient edits favoring novel tokens — (source:
  same).
- **Velocitune** (arXiv 2411.14318) — weight domains by learning velocity; the domain-
  level analogue of TOK-4's migration rate — (source: same).
- **tDRO** (arXiv 2408.10613); **XDoGE** (arXiv 2512.10545); **online sample reweighting**
  (Zhao et al. 2024) — domain-level reweighting family — (source: same).
- **Concept-level objectives** (Iyer et al., arXiv 2601.11791) — surface forms of one
  concept count as correct; changes the token-identity assumption behind per-token KL —
  (source: same).
- **UL2 mixture-of-denoisers**, **SpacTor-T5** (arXiv 2401.13160), **continuous-paragraph-
  denoise diffusion LMs**, **RTS/SLM structural objectives** (arXiv 2309.08272) — (source:
  same).
- **LLM-JEPA** (arXiv 2509.14252); **Focused Transformer contrastive KV training** (arXiv
  2307.03170) — embedding-space objectives — (source: same).
- **Missing canon flagged by the review's intake note:** label smoothing / confidence
  penalty; unlikelihood (Welleck et al. 2019); DeepSeek-V3 MTP at scale (arXiv 2412.19437);
  fill-in-the-middle (arXiv 2207.14255); instruction-loss masking vs. loss-over-
  instructions (arXiv 2405.14394 — the fine-tuning-side "which tokens count" question);
  z-loss and auxiliary losses; latent reasoning objectives; Byte Latent Transformer —
  (source: docs/topics/reference/training-objective-alternatives-literature.md).

**Post-training token regimes (TOK-opt-4, the RLVR bridge)**

- **Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM
  Reasoning** (no ID on record) — a small high-entropy minority carries most of RLVR's
  effect; the clearest statement of "the wall bucket resurfacing as the locus of
  post-training" — (source: docs/topics/reference/token-level-literature.md).
- **Revisiting Entropy in Reinforcement Learning for Large Reasoning Models** (no ID on
  record) — masking RLVR updates to different token regimes produces qualitatively
  different dynamics, some driving stability and some collapse — (source: same).
- **Wu & Choi, On the Limits of RLVR: Support, Entropy, and the Illusion of Reasoning**
  (AI for Math Workshop, ICML 2025) — RLVR as support-preserving, entropy-reducing
  reweighting — (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/potential-projs/token-movement.md §4).
- **Yue et al., Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?**
  (NeurIPS 2025 oral; arXiv 2504.13837) — pass@k improves at small k, boundary unchanged
  at large k — (source: same).
- **The Invisible Leash** (arXiv 2507.14843) and **RLVR Implicitly Incentivizes Correct
  Reasoning in Base LLMs** (arXiv 2506.14245, ICLR 2026) — the recorded counterpoints —
  (source: same).
- **The recorded bridge claim** — "the tokens that form the valley walls in pretraining
  look like the same tokens where RL does its work," a bridge the entry says "nobody has
  drawn explicitly" — (source: docs/topics/reference/token-level-literature.md;
  docs/potential-projs/token-movement.md §4).

**Eval-side frame and the item-flip measure (TOK-obs-2, the low-noise-eval corollary)**

- **Signal and Noise: A Framework for Reducing Uncertainty in LM Evaluation** (Heineman
  et al., NeurIPS 2025 per the record) — signal = separating better from worse models,
  noise = sensitivity to step-to-step variability; continuous metrics beat accuracy;
  filtering noisy subtasks improves aggregate reliability; ~900K results on 465 models
  including DataDecide — the framework TOK-obs-4's dividend attaches to, against their
  *empirical* subtask filtering — (source:
  docs/topics/reference/evaluation-methodology-literature.md).
- **"Where eval variance actually lives"** — re-evaluating a fixed checkpoint with new
  seeds buys nothing for OLMES-style loglikelihood evals; the variance is in training
  (seed, data order, init); few-shot configuration variance is "a bias axis to sweep, not
  noise to average" — sets the baseline TOK-obs-2's flip rates are compared to — (source:
  same).
- **OLMES: A Standard for Language Model Evaluations** (no ID on record) — the eval
  standard behind the processed tables TOK-obs-2 joins — (source: same).
- **"The churn literature's measure"** (no citation on record) — prediction-flip rate as
  the attribution for TOK-obs-2; §2 rates it "known phenomenon in a new suite" — (source:
  docs/potential-projs/token-movement.md §2, §4).
- **Per-instance eval coverage** — `instances.parquet` for all 25 recipes × 66 tasks at
  150M–1B with 3 seeds (1 seed below 150M); the fact that makes TOK-obs-2 pure T0 —
  (source: docs/open-questions-answered.md, 2026-08-21 entry).
- **750M table truncation and seed coverage** — the aggregate OLMES table is truncated at
  step 26,250 while the instance table runs to 63,599; "750M only has 1 seed that trains
  fully" (hedged, open) — caveats for any 750M slice of TOK-obs-2 — (source:
  docs/open-questions-answered.md).

**Layerwise drift instruments (TOK-obs-3)**

- **CKA** (Kornblith et al., no ID on record) — the scalable representation-similarity
  proxy; recorded caveat that it can be dominated by a few directions and disagree with
  stitching; §2 calls the representation-dynamics literature crowded — (source:
  docs/potential-projs/landscape-geometry.md §5 → identifiability-literature.md;
  docs/potential-projs/token-movement.md §2).
- **Linear-map residuals** (Roeder, Metz & Kingma 2021) — the weight-free complement —
  (source: docs/potential-projs/landscape-geometry.md §5).
- **Model stitching** (Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021) — the
  behavioral ground truth for functional interchangeability at a depth — (source:
  docs/topics/reference/reinit-and-transfer-literature.md).
- **Representation-plasticity timeline in LLMs** (arXiv 2410.06225) — when representations
  stop moving; the depth-and-stage counterpart to TOK-obs-3 — (source:
  docs/topics/reference/reinit-and-transfer-literature.md; plasticity-adjacent).

**Durable-vs-transient movement and the branch instruments (Stage 2)**

- **Frankle et al., Linear Mode Connectivity and the Lottery Ticket Hypothesis** — twin
  branches from a shared checkpoint; barrier-between-siblings as a basin-commitment clock;
  the instrument the "durable movement" arm's schedule-neutralized comparison borrows —
  (source: docs/topics/staging/checkpoint-tomography.md;
  docs/topics/reference/landscape-literature.md).
- **The devinterp / SGLD local learning coefficient** (Lau, Murfet et al., Timaeus; no ID
  on record) — short noisy continued training measuring local degeneracy, tracked across
  Pythia-style checkpoint sequences and shown to detect developmental transitions; the
  off-the-shelf "point at movement" statistic — (source:
  docs/topics/staging/checkpoint-tomography.md).
- **Critical-sharpness and basin-emergence single-checkpoint statistics** (no IDs on
  record) — progressive sharpening at scale, applied even to data-mixing decisions;
  covariates for TOK's bucket analyses — (source:
  docs/topics/staging/checkpoint-tomography.md).
- **The checkpoint-tomography battery** — decay branch → wall height (total *and per-token
  profile*, which is exactly TOK-3); hot branch → diffusion width; twin branches → sibling
  barrier; data-shifted branch → component responsiveness; recorded as unclaimed as a
  battery, with a prior-art check still owed over the devinterp and WSD-followup
  communities' 2025–26 output — (source: docs/topics/staging/checkpoint-tomography.md).
- **The shared held-out-token-set spec** — an identical spec appears in Annealed readouts,
  WSD retrain suite, Token-level movement, MoE movement, MoE recipe suite, and Functional
  featurization; must be frozen before any branch runs — (source:
  docs/potential-projs/token-movement.md §1, §3).

**MoE twin (TOK-obs-5)**

- **FLAME-MoE: A Transparent End-to-End Research Platform for MoE LMs** (no ID on record)
  — 38M–1.7B active, 64 experts, top-8; released code, data, checkpoints, routing logs,
  evals; traces show specialization emerging early and intensifying, co-activation sparse
  and stable, routing converging quickly — the substrate for TOK-obs-5 — (source:
  docs/topics/reference/moe-literature.md).
- **OLMoE: Open Mixture-of-Experts Language Models** (no ID on record) — router saturation
  as top-k overlap at step t vs. convergence, rising sharply within the first few thousand
  steps, deeper layers saturating faster; the "routes freeze" side of TOK-obs-5 — (source:
  same).
- **Three Phases of Expert Routing** (no ID on record) — early balance-prioritizing,
  stabilization/specialization, late relaxation; non-monotone and invisible to post-hoc
  analysis; annealing checkpoints confirm phases are pretraining-specific — (source: same).
- **Continual Pre-training of MoEs: How Robust Is Your Router?** (no ID on record) —
  routing changes most in early layers; no-replay shows the most reorganization and most
  forgetting — (source: same).
- **The Myth of Expert Specialization in MoEs** (no ID on record) — routing reflects
  representation geometry, not domain expertise; routers are linear maps so hidden-state
  similarity explains expert-usage similarity; load-balancing loss provably suppresses
  shared hidden directions — the confound behind the "regress on token ID, frequency band,
  and position first" caveat — (source: same; docs/potential-projs/token-movement.md §4).
- **The OpenMoE token-identity-clustering claim** — routing mostly fixed early by token
  identity; recorded as *unverified*; if true, deviations (context-dependent routing, late
  reassignments) mark exactly the tokens TOK's entropy-bucket hypothesis cares about —
  (source: docs/potential-projs/token-movement.md §4;
  docs/topics/reference/nonstationarity-accounting.md).
- **Expert permutation as a textbook non-identifiable latent** — breaks dense comparability
  tools (interpolation barriers, checkpoint merging, stitching all need expert alignment;
  re-basin for MoE immature) — (source: docs/topics/reference/moe-literature.md).
- **Mixture of Parrots** (Jelassi et al., ICLR 2025) — experts buy memorization not
  reasoning; supports a per-token-by-frequency-band decomposition of MoE gains — (source:
  same).
- **FLAME-MoE routing-log contents** — open item: which checkpoints, how many tokens,
  whether token identities are recoverable; gates TOK-obs-5 entirely and decides T0 vs. T1
  — (source: docs/open-questions-answered.md "Open — not yet checked").

**Non-stationarity framing (TOK-opt-2, TOK-4)**

- **The endogenous self-curriculum** — even under iid data the effective distribution is
  data weighted by current gradient magnitude, so as easy/deterministic tokens saturate
  the signal migrates to harder tokens; "your Rho-1-style loss-trajectory taxonomy and the
  river/wall token migration are measurements of exactly this"; loss-of-plasticity in
  "stationary" pretraining stops being paradoxical — (source:
  docs/topics/reference/nonstationarity-accounting.md).
- **The exogenous/endogenous split and the "accounting" program** — LR schedule, data-order
  drift, midtraining as exogenous; routing and the gradient-weighted distribution as
  endogenous; the missing contribution is per-source accounting in comparable units —
  (source: same).
- **LR decay as controlled non-stationarity** — the schedule is itself the treatment TOK's
  Stage 2 manipulates — (source: same, 2026-08-18 entry).
- **ITER** (Igl et al., ICLR 2021; arXiv 2006.05826) — transient non-stationarity
  permanently scars the representation; the third statement of the thesis phenomenon
  alongside Achille 2019 and Ash & Adams 2020 — (source: same;
  docs/topics/reference/reinit-and-transfer-literature.md).

**Post-training twins of TOK's core figures (shared with MIC)**

- **Movement-microscope Stage 3** — per-token KL sliced by determinism/entropy buckets with
  the *base model* as the reference point: the same instrument as TOK-obs-4 one stage later
  — (source: docs/potential-projs/movement-microscope.md; token-movement.md §4).
- **Movement-microscope Stage 4** — post-train all 25 recipes and compare movement profiles
  at matched final loss; the post-training counterpart of TOK-4's cross-recipe migration —
  (source: same).
- **"Did the model move in distribution space?"** — NLL on held-out traces, KL from base,
  calibration, sample diversity, pass@k at very large k as continuous low-variance
  alternatives to accuracy; the same trick DataDecide used, applied one stage later —
  (source: docs/potential-projs/token-movement.md §4;
  docs/topics/reference/pretraining-to-posttraining.md).
- **DataDecide** (Magnusson et al., ICML 2025; arXiv 2504.11393) — the checkpoint suite,
  the recipe axis, and the "continuous likelihood metrics as small-scale proxies" precedent
  — (source: docs/topics/reference/pretraining-to-posttraining.md).

**Program-position records (not literature, but on record for this project)**

- **Portfolio positions** — Tier 1 #1 on the 6–12-month flagship list (mechanism + thesis
  halves of "the unified causal program"; "vibes" figure = heatmap of per-token
  decay-responsiveness vs. entropy bucket vs. training position with recipes overlaid); #8
  workshop-sized (Stage 1; "highest-ceiling figure in the dense program," first with
  genuine null risk); #9 full-conference ("Which Tokens Does the Cooldown Fix?", expected
  impact high, ceiling very high) — (source: docs/portfolio-rankings.md;
  docs/potential-projs/token-movement.md §4).
- **Ranked #5 in a top-5 workshop-likelihood × speed list (Stage 1 only)** — TOK-obs-2 is a
  groupby; TOK-obs-4 is "the highest-impact single figure available at T1-light cost";
  the harness it builds is what annealed readouts, landscape geometry, and Stage 2 need
  next — (source: docs/potential-projs/token-movement.md §4).
- **Litreview reading row** — `../litreview/recipe-featurization-litreview-plan.md` row C is
  the reading row for this cluster — (source: docs/potential-projs/token-movement.md §5).

**Provenance caveats**

- **Citation-verification ledger: ~27 rows tagged `TOK, dense`** — 2205.02517, 2207.14255,
  2305.16958, 2307.03170, 2309.08272, 2401.13160, 2402.14270, 2404.07965, 2404.19737,
  2405.14394, 2405.18906, 2407.12665, 2408.10613, 2409.13641, 2411.14318, 2412.14780,
  2412.19437, 2505.16900, 2505.19893, 2505.22757, 2509.14252, 2509.20758, 2510.00526,
  2510.27462, 2511.00198, 2512.10545, 2601.11791 — all from the
  `training-objective-alternatives-literature` intake; **agent-supplied or Claude-added,
  none verified**; Claude-added rows are flagged as hallucination-prone in the last digits
  and in title–ID pairing — (source: docs/litreview/citation-verification-ledger.md).
- **SciSpace CE-alternatives review quality flags** — RLHF-adjacent methods the prompt
  excluded were included anyway (ASPO, λ-GRPO, UFT, sequence-level CPO, GRACE; dropped in
  intake); numbers are as the agent reported, unverified — (source:
  docs/topics/reference/training-objective-alternatives-literature.md).
- **Artifacts on disk** — the SciSpace CE-alternatives bundle (report md + LaTeX/PDF, 10
  parsed papers with figures, 9 further full texts, 37 mostly off-topic downloads, 15
  search CSVs, `INDEX.md` with the missing-canon list) at
  `~/drotherm/data/convo-artifacts/2026/scispace-alts-to-CE-loss-llm-pretraining-agent-
  artifacts-zip_.../` — (source: same).

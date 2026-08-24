# movement microscope — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`movement-microscope.md`](../movement-microscope.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

High-recall corpus for MIC (movement microscope). Every paper, method, or named
prior-art item on record anywhere in this repository that is *possibly* relevant to
measuring post-training movement at small scale. Grouped by theme; one line per item
with its repo source. Inclusion here is not endorsement: many entries are
agent-generated and unverified, and several are marginal by design. No positioning
claims — inventory only.

**The substrate and the original hypothesis**

- **DataDecide: How to Predict Best Pretraining Data with Small Experiments**
  (Magnusson et al., Ai2, ICML 2025; arXiv 2504.11393) — the 25-recipe / ≤1B / 3-seed
  checkpoint set MIC post-trains, and the source of the "continuous likelihood metrics
  beat accuracy at small scale" move MIC applies one stage later — (source:
  docs/topics/reference/pretraining-to-posttraining.md).
- **Tulu / Tulu 3** — the SFT datasets used in the earlier no-movement project;
  MIC-opt-1 re-measures exactly those runs — (source:
  docs/topics/reference/pretraining-to-posttraining.md, Danielle's verbatim first-hand
  account).
- **The 2025 DataDecide post-training project itself** (unwritten past-project record) —
  the negative result MIC exists to re-measure; listed as a candidate past-project
  write-up — (source: docs/past-projects/README.md).

**"Pretraining shapes post-training beyond final loss" — the closest experimental
neighbors (MIC-4)**

- **Similar Models Learn Differently: Final-Window Pretraining Shapes Post-Training
  Beyond SFT** (arXiv 2607.25063) — recorded as "closest to your exact experimental
  design"; models that look alike after SFT diverge under identical post-training
  depending on late-pretraining interventions; explicitly cites Dohare's plasticity paper
  — (source: docs/topics/reference/pretraining-to-posttraining.md).
- **Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining** (Zhao,
  Meterez et al., COLM 2025; arXiv 2504.07912) — from-scratch training on controlled
  pretraining mixtures then PPO/GRPO/EI across scales; also argues controlled small-model
  proxies yield real RL insight, the premise of MIC's whole scale choice — (source:
  docs/topics/reference/pretraining-to-posttraining.md).
- **Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data**
  (Akter et al., NVIDIA, ICLR 2026; arXiv 2510.03264) — broad reasoning data best in
  pretraining, curated long-CoT best in SFT; the data-placement contrast MIC-4's recipe
  axis sits inside — (source: docs/topics/reference/pretraining-to-posttraining.md).
- **Early Data Exposure Improves Robustness to Subsequent Fine-Tuning** (Feng et al.,
  arXiv 2605.12705) — moving target-domain data into pretraining improves retention after
  fine-tuning even at matched immediate post-training performance — (source:
  docs/topics/reference/pretraining-to-posttraining.md).
- **The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data** (Baek et al.,
  arXiv 2603.16177) — early domain exposure more durable than late; repetition schedule
  decides generalize/overfit/forget — (source:
  docs/topics/reference/pretraining-to-posttraining.md).
- **Understanding Reasoning from Pretraining to Post-Training** (Shen et al., arXiv
  2607.16097) — lower pretraining loss predicts higher post-RL pass@1 at fixed RL
  compute; recorded as *in tension* with the original hypothesis, so a contrast case for
  MIC-4's matched-loss framing — (source:
  docs/topics/reference/pretraining-to-posttraining.md).
- **Pre-/mid-training/RL interplay study (arXiv 2512.07783)** — the forward-run version
  of the same question (midtraining *for* a downstream suite), pointer only — (source:
  docs/topics/reference/pretraining-to-posttraining.md → targeted-pretraining-
  midtraining-literature.md).

**The "no movement" / underpowered-post-training neighbors (MIC-opt-1, MIC-opt-3)**

- **A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to
  Reproducibility** (Hochlehnert et al., COLM 2025; arXiv 2504.07086) — RL on
  distillation-based models yields little significant gain; small benchmarks give
  unstable estimates, multi-seed essential; MIC-opt-3 proposes the post-training-
  *experiment* power-analysis counterpart to its evaluation variance analysis — (source:
  docs/topics/reference/pretraining-to-posttraining.md; also flagged in
  docs/topics/reference/evaluation-methodology-literature.md as the corrective to the
  existence-proof genre).
- **Spurious Rewards: Rethinking Training Signals in RLVR** (Shao et al., ICML 2026;
  arXiv 2506.10947) — GRPO with random rewards moves MATH-500 by 21.4 points on
  Qwen2.5-Math-7B and generally fails outside Qwen; the "elicitation in disguise" case
  MIC's elicitation-controlled readout must exclude — (source:
  docs/topics/reference/pretraining-to-posttraining.md;
  docs/topics/reference/evaluation-methodology-literature.md).
- **Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?** (Yue et al.,
  NeurIPS 2025 oral; arXiv 2504.13837) — RLVR improves pass@k at small k but not the
  reasoning boundary at large k; pass@k-at-large-k is one of MIC-opt-1's proposed
  distribution-space readouts — (source:
  docs/topics/reference/pretraining-to-posttraining.md).
- **On the Limits of RLVR: Support, Entropy, and the Illusion of Reasoning** (Wu & Choi,
  AI-for-Math workshop, ICML 2025) — RLVR as "predominantly support-preserving,
  entropy-reducing reweighting"; the token-regime characterization MIC-3's bucket slice
  echoes — (source: docs/topics/reference/pretraining-to-posttraining.md).
- **The Invisible Leash** (arXiv 2507.14843) — counterpoint to Yue et al. — (source:
  docs/topics/reference/pretraining-to-posttraining.md).
- **RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs** (arXiv 2506.14245, ICLR
  2026) — the second counterpoint — (source:
  docs/topics/reference/pretraining-to-posttraining.md).
- **Small-scale SFT degradation** (Chen et al., arXiv 2505.17988) — SFT on Qwen2.5-1.5B
  *reduces* MATH-500 accuracy 23.8%→18.4% while eliciting reasoning-style behavior; the
  negative-movement case a movement instrument should register — (source:
  docs/topics/reference/pretraining-to-posttraining.md).
- **Through the Valley: Path to Effective Long CoT Training for Small Language Models**
  (Luo et al., EMNLP 2025; arXiv 2506.07712) — "Long CoT Degradation" via error
  accumulation at small scale — (source:
  docs/topics/reference/pretraining-to-posttraining.md).

**Within-reach-task and synthetic-testbed precedents (MIC-2, "lower the task")**

- **TinyZero** (no ID on record) — "RL visibly works at 0.5–3B on countdown and simple
  arithmetic"; the template for MIC-2's within-reach synthetic tasks — (source:
  docs/topics/reference/pretraining-to-posttraining.md; docs/potential-projs/movement-
  microscope.md §4, 2026-08-18 entry).
- **Provable Benefits of RLVR over SFT for Reasoning Models: Learning to Backtrack
  Efficiently** (no ID on record) — graph-pathfinding synthetic testbed where "a seed
  costs minutes"; causal rather than correlational, offered as the fully-synthetic style
  MIC could borrow — (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/potential-projs/movement-microscope.md §4).
- **Task-shaped low-budget SFT recipe ordering** (task-shaped SFT → distillation → DPO on
  verifiable pairs; LoRA/QLoRA; data selection over volume, arXiv 2503.01807) — the
  method menu offered for moving metrics on small models; retained in the record only for
  its ordering, consistent with MIC-2's guaranteed-effect step; agent-supplied and mostly
  unchecked — (source: docs/topics/reference/pretraining-to-posttraining.md, response
  condensation + intake notes).

**Distillation arm (MIC-2's KL-to-teacher axis, MIC-opt-5) — all from Danielle's SciSpace
six-question review; characterizations are the agent's and identifiers unverified**

- **MiniLLM / reverse KL** (ICLR 2024, no ID on record) — mode-seeking reverse KL, policy
  gradient with teacher-mixed sampling (α=0.2) and length normalization, plus an explicit
  LM loss; the default objective MIC fixes for the KL-to-teacher readout — (source:
  docs/topics/reference/distillation-literature.md).
- **Concrete Score Matching / JSD, skew-KL, α-β, TV variants** (Kim et al., arXiv
  2509.25837) — the skew/JSD family MIC's §4 entry names as the alternative default —
  (source: docs/topics/reference/distillation-literature.md).
- **ToDi** (arXiv 2505.16297) — token-wise divergence control; per-token distillation
  loss is directly the per-token movement axis MIC-3 slices — (source:
  docs/topics/reference/distillation-literature.md).
- **BiLD** (arXiv 2406.13555) — objective variant in the same family — (source:
  docs/topics/reference/distillation-literature.md).
- **DSKD** (arXiv 2406.17328) — equal-weight CE + KD at T=2.0; the CE/KD combination
  choice MIC's harness must fix — (source:
  docs/topics/reference/distillation-literature.md).
- **ULD cross-tokenizer loss** (arXiv 2402.12030) — cross-tokenizer distillation loss —
  (source: docs/topics/reference/distillation-literature.md).
- **Multi-level OT distillation** (arXiv 2412.14528) — cross-tokenizer variant — (source:
  docs/topics/reference/distillation-literature.md).
- **Minitron** (arXiv 2407.14679) — 2:1 teacher/student ratio anchor for MIC-opt-5's size
  choice — (source: docs/topics/reference/distillation-literature.md).
- **ADPA** (arXiv 2502.17927) — ~26:1 ratio, advantage-guided KD weighted against SFT,
  62.7% AlpacaEval win rate for a 1.8B student; the alignment-distillation end of the
  teacher question — (source: docs/topics/reference/distillation-literature.md).
- **Pre-training-distillation design-space study** (Peng et al., arXiv 2410.16215) —
  moderate 2–4:1 ratios best in general — (source:
  docs/topics/reference/distillation-literature.md).
- **DDK** (arXiv 2407.16154) — domain-aligned mid-size teachers can beat bigger ones —
  (source: docs/topics/reference/distillation-literature.md).
- **Distillation scaling laws** (Busbridge et al., arXiv 2502.08606) — flagged as the
  *missing anchor*: no distillation scaling law is cited by the review; MIC's §4 lists it
  as missing canon to cite — (source: docs/topics/reference/distillation-literature.md;
  docs/potential-projs/movement-microscope.md §4).
- **On-policy GKD** (Agarwal et al., arXiv 2306.13649) — missing canon flagged for the
  objective question — (source: docs/topics/reference/distillation-literature.md).
- **DistiLLM** (arXiv 2402.03898) — missing canon flagged alongside GKD — (source:
  docs/topics/reference/distillation-literature.md).
- **Bui et al.** (arXiv 2404.19319) — the only logit-vs-token-repetition evidence:
  BERT-scale, KD beats scratch by 1.3–2.4 points only when data-limited; the basis for
  MIC's requirement of a *from-scratch control at matched token budget* in MIC-2 —
  (source: docs/topics/reference/distillation-literature.md; movement-microscope.md §4).
- **GLMD** (arXiv 2306.06625) — the counterexample where a heavily resourced from-scratch
  2B beat a 10B→2B distilled model (85.9 vs. 85.3 SuperGLUE) — (source:
  docs/topics/reference/distillation-literature.md).
- **"Well-read students learn better"** (Turc et al. 2019, no ID on record) — pretrain
  the student before distilling; relevant to MIC-opt-5's student setup — (source:
  docs/topics/reference/distillation-literature.md).
- **MiniPLM** (arXiv 2410.17215) — pre-training distillation yielding reusable bases; the
  "distil from the base teacher" arm of MIC-opt-5 — (source:
  docs/topics/reference/distillation-literature.md).
- **"Revealing the power of post-training for SLMs via KD"** (arXiv 2509.26497) — the
  "distil from the post-trained teacher" arm — (source:
  docs/topics/reference/distillation-literature.md).
- **DCKD** (no ID on record) — alignment-distillation entry alongside ADPA — (source:
  docs/topics/reference/distillation-literature.md).
- **The review's own "no controlled sequential-vs-direct comparison exists" finding** —
  the record MIC-opt-5 rests on; the review restates question 5 with preference-
  distillation results and argues flexibility vs. efficiency; explicitly under-evidenced
  (questions 1, 3, 6) — (source: docs/topics/reference/distillation-literature.md;
  docs/potential-projs/movement-microscope.md §4).
- **Sequence-level KD = training on teacher samples** — pointer to the synthetic-data
  accumulator, relevant if MIC-2's distillation arm goes sequence-level — (source:
  docs/topics/reference/distillation-literature.md header, → synthetic-data-
  literature.md).
- **Muennighoff et al. token repetition (≤4 epochs ≈ free)** — the token-repetition side
  of the logit-vs-token question, kept in the sibling file — (source:
  docs/topics/reference/distillation-literature.md → synthetic-data-literature.md).

**Instrument and noise-floor lineage (MIC-1)**

- **Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation**
  (Heineman et al., NeurIPS 2025 per the record) — signal/noise definitions, noise as
  step-to-step wander, continuous metrics beating accuracy, noisy-subtask filtering; the
  ~900K-result release covers 465 models including DataDecide; the framework MIC-1's noise
  floor extends to post-training — (source:
  docs/topics/reference/evaluation-methodology-literature.md).
- **"Where eval variance actually lives"** (2026-08-18 entry) — for OLMES-style
  loglikelihood evals, re-evaluating a fixed checkpoint with new seeds buys nothing;
  variance lives in training (seed, data order, init); bounds what MIC-1 can measure and
  dictates that its null distribution be built from *training* replicates — (source:
  docs/topics/reference/evaluation-methodology-literature.md).
- **OLMES: A Standard for Language Model Evaluations** (no ID on record) — the eval
  standard the DataDecide tables and MIC's harness follow — (source:
  docs/topics/reference/evaluation-methodology-literature.md).
- **The continued-pretraining token-exposure control** — "movement that doesn't exceed
  seed-noise-plus-token-exposure isn't movement"; recorded as the piece to add to any
  trajectory noise floor judging an intervention that adds training — (source:
  docs/potential-projs/trajectory-statistics.md §4, 2026-08-18 entry).
- **P1–P4 idea-map restatement of the microscope** — the compact statement of the four
  stages with the metric list (per-token KL, likelihood margins, accuracy, layerwise CKA,
  ΔW norm/effective rank) and their dependencies on M1/N1/I1/F2 — (source:
  docs/dataset-analysis-idea-map.md).

**Plasticity / diagnostic-panel lineage for the movement metrics (MIC-1, MIC-3)**

- **Lyle et al., Understanding Plasticity in Neural Networks** (ICML 2023; arXiv
  2303.01486) — plasticity loss tied to loss-landscape curvature, often without saturated
  units; the source of the curvature/feature-rank/dead-units/weight-norm panel MIC's
  metrics draw on — (source: docs/topics/reference/plasticity.md).
- **Lyle et al., Disentangling the Causes of Plasticity Loss** (arXiv 2402.18762) —
  follow-up — (source: docs/topics/reference/plasticity.md).
- **Dohare et al., Loss of plasticity in deep continual learning** (Nature 2024; earlier
  arXiv 2306.13812) — continual backpropagation, dormant-unit reinitialization; cited by
  the Final-Window paper — (source: docs/topics/reference/plasticity.md).
- **Achille, Rovere & Soatto, Critical Learning Periods in Deep Networks** (ICLR 2019) —
  Information Plasticity with the **Fisher trace** as diagnostic; the record adds Fisher
  trace to MIC's panel and notes Lyle's curvature finding is nearly a rediscovery in
  different coordinates — (source: docs/topics/reference/critical-periods.md;
  docs/topics/reference/plasticity.md).
- **Can Scale Save Us From Plasticity Loss in LLMs?** (Hernandez-Garcia, Figliolia,
  Millidge; arXiv 2606.24752) — plasticity loss at 5M–314M, sublinear scaling law,
  continual *and stationary*; the LLM-scale evidence that a 150M model's "state of the
  learner" is a real variable in MIC-4 — (source: docs/topics/reference/plasticity.md).
- **"Plasticity-style statistics as an RL-ability proxy"** (curvature, feature rank, NLL
  on gold traces, pass@k at large k, entropy at decision points) — the proxy-metric
  contribution MIC-opt-3's power analysis is paired with — (source:
  docs/potential-projs/movement-microscope.md §4; docs/potential-projs/tiny-scale-
  measurement.md §4).

**Representation-drift and weight-space instruments (MIC-1, MIC-3)**

- **CKA** (Kornblith et al., no ID on record) — the scalable representation-similarity
  proxy for MIC's per-layer drift metric; recorded caveat that it "can be dominated by a
  few directions and disagree with stitching" — (source:
  docs/potential-projs/landscape-geometry.md §5 → identifiability-literature.md).
- **Linear-map residuals** (Roeder, Metz & Kingma 2021) — the weight-free complement to
  CKA MIC-1 lists alongside it — (source: docs/potential-projs/landscape-geometry.md §5).
- **Model stitching** (Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021) — ground truth
  for functional interchangeability at a depth; the record calls stitching "literally
  your embedding-reset experiment as a measurement" — usable as a movement readout —
  (source: docs/topics/reference/reinit-and-transfer-literature.md;
  docs/potential-projs/landscape-geometry.md §5).
- **Editing Models with Task Arithmetic** (Ilharco et al., ICLR 2023; arXiv 2212.04089) —
  the weight-space task vector ΔW = finetuned − pretrained; MIC-3's "project ΔW onto
  interpretable axes" is arithmetic on exactly this object — (source:
  docs/topics/reference/task-vectors.md).
- **On Task Vectors and Gradients** (Zhou et al., arXiv 2508.16082) — a one-epoch task
  vector is exactly the negative scaled gradient; first-epoch gradient dominates the
  finetuning trajectory in norm and direction — a cheap surrogate for MIC's ΔW readouts —
  (source: docs/topics/reference/task-vectors.md).
- **Transporting Task Vectors across Different Architectures** (Rinaldi et al., ICML
  2026; arXiv 2602.12952) — functional (activation-based) task identity via orthogonal
  Procrustes; relevant if MIC compares movement directions across sizes — (source:
  docs/topics/reference/task-vectors.md).
- **Task Vector Quantization for Memory-Efficient Model Merging** (Kim et al., arXiv
  2503.06921) — task vectors have narrow weight range, 4-bit quantizable; the storage
  trick for saving many post-trained endpoints — (source:
  docs/topics/reference/task-vectors.md).
- **On the Emergence of Cross-Task Linearity in the Pretraining-Finetuning Paradigm** and
  **Model soups** (Wortsman et al.) — why task arithmetic and merging work only within a
  basin; the precondition for treating ΔW directions as comparable across MIC's 25
  recipes — (source: docs/topics/reference/landscape-literature.md;
  docs/topics/reference/task-vectors.md).
- **Linear Connectivity Reveals Generalization Strategies** (Juneja et al., ICLR 2023) —
  fine-tuned models cluster into distinct basins implementing *different* generalization
  strategies at similar in-distribution accuracy; the strongest recorded evidence that
  "flat accuracy" can hide mechanism differences, i.e. MIC's premise from the basin side
  — (source: docs/topics/reference/landscape-literature.md).

**Elicitation controls (MIC-opt-2) and the capability-vs-accessibility split**

- **Eliciting Fine-Tuned Transformer Capabilities via Inference-Time Techniques** (no ID
  on record) — formal argument that SFT-acquired capabilities can be approximated by the
  base model via ICL without parameter updates; the strongest form of "movement may be
  elicitation" — (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/potential-projs/icl-elicitability.md).
- **Fine-Tuned In-Context Learners for Efficient Adaptation** (Bornschein, Lyle, Pascanu
  et al., no ID on record) — prompt-based methods excel few-shot but plateau as data
  grows; the plasticity group moving into ICL-vs-fine-tuning; supplies the elicitation
  ceiling MIC-opt-2 nets movement against — (source:
  docs/topics/reference/pretraining-to-posttraining.md; docs/topics/reference/
  plasticity.md).
- **The Power of Scale for Parameter-Efficient Prompt Tuning** (Lester, Al-Rfou &
  Constant 2021) — prompt tuning matches full fine-tuning only above ~10B and lags at
  small sizes; the recorded headwind for any elicitation-based null at 150M; flagged
  "from memory; verify" — (source: docs/potential-projs/elicitation-gain.md §5).
- **Predictive 𝒱-information** (Xu et al., arXiv 2002.10689) — the measurement language
  shared with ELI for "capability under a declared wrapper family"; a candidate formal
  frame for elicitation-controlled movement — (source:
  docs/potential-projs/elicitation-gain.md §5).
- **pass@k estimator** (arXiv 2107.03374) and estimation toolkit (block bootstrap,
  Wilson/Jeffreys, calibrate-after-selection, split conformal — Lei et al. 1604.04173;
  Angelopoulos et al. 2208.02814) — the interval machinery for MIC-opt-1's pass@k-at-
  large-k readout and MIC-opt-3's power analysis — (source:
  docs/potential-projs/elicitation-gain.md §5).
- **ELI-3 (elicitation-gain project) as MIC-opt-2's instrument** — a fixed outer
  optimizer measuring pre/post extractability (ΔS, stability, iterations-to-threshold) on
  the same SFT checkpoints — (source: docs/potential-projs/movement-microscope.md §1
  coordination note; docs/potential-projs/elicitation-gain.md).
- **TinyStories** (Eldan & Li 2023), **the phi-1 line**, **small-model DSL/semantic-
  parsing work**, **Distilling step-by-step** (Hsieh et al. 2023) — small-model leads
  recorded as from-memory and unverified; relevant to MIC-2's "what can a 150M model be
  made to learn at all" — (source: docs/potential-projs/elicitation-gain.md §5).

**Token-bucket slice (MIC-3) — shared instruments with TOK**

- **Wen et al., Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss
  Landscape View** (arXiv 2410.05192) — deterministic tokens as river, uncertain tokens
  as walls; the entropy buckets MIC-3 slices per-token KL by — (source:
  docs/topics/reference/landscape-literature.md; token-level-literature.md).
- **Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM
  Reasoning** (no ID on record) — a small high-entropy minority carries most of RLVR's
  effect; the "forking token" result MIC-3 asks whether small-model SFT echoes — (source:
  docs/topics/reference/token-level-literature.md).
- **Revisiting Entropy in Reinforcement Learning for Large Reasoning Models** (no ID on
  record) — masking RLVR updates to different token regimes gives qualitatively different
  dynamics — (source: docs/topics/reference/token-level-literature.md).
- **Rho-1, Not All Tokens Are What You Need for Pretraining** (arXiv 2404.07965) —
  loss-trajectory token taxonomy; the reference-model excess-loss scorer MIC's held-out
  set could reuse — (source: docs/topics/reference/token-level-literature.md;
  training-objective-alternatives-literature.md).
- **Token-Level Uncertainty-Aware Objective for Language Model Post-Training** (no ID on
  record) — epistemic vs. aleatoric per token; the decomposition MIC-3's entropy buckets
  implicitly assume — (source: docs/topics/reference/token-level-literature.md).

**MoE variant (MIC-opt-4): routing flips as the categorical movement channel**

- **FLAME-MoE: A Transparent End-to-End Research Platform for MoE LMs** (no ID on record)
  — 38M–1.7B active, 64 experts, top-8, released checkpoints, routing logs, evals;
  "DataDecide-for-MoE at exactly your target scale" — (source:
  docs/topics/reference/moe-literature.md).
- **OLMoE: Open Mixture-of-Experts Language Models** (no ID on record) — router
  saturation as top-k overlap vs. convergence; deeper layers saturate first — a
  commitment clock analogous to MIC's movement decay — (source:
  docs/topics/reference/moe-literature.md).
- **Three Phases of Expert Routing: How Load Balance Evolves During MoE Training** (no ID
  on record) — early balance-prioritizing, stabilization/specialization, late relaxation;
  annealing checkpoints confirm the phases are pretraining-specific and *stable during
  fine-tuning* — directly bears on whether MIC-opt-4 would see routing movement at all —
  (source: docs/topics/reference/moe-literature.md).
- **Continual Pre-training of MoEs: How Robust Is Your Router?** (no ID on record) —
  routing changes most in early layers; no-replay shows the most reorganization and most
  forgetting — (source: docs/topics/reference/moe-literature.md).
- **The Myth of Expert Specialization in MoEs** (no ID on record) — routing reflects
  representation geometry, not domain expertise; load-balancing loss provably suppresses
  shared hidden directions — the confound for any routing-movement claim — (source:
  docs/topics/reference/moe-literature.md).
- **Mixture of Parrots** (Jelassi et al., ICLR 2025) — experts buy memorization, not
  reasoning; the template for decomposing gains by eval type, and a mechanism for a
  reasoning-side null — (source: docs/topics/reference/moe-literature.md).
- **Recorded MoE warnings for MIC** — expert permutation breaks dense comparability tools
  (interpolation barriers, checkpoint merging, stitching all need expert alignment;
  re-basin for MoE is immature); MoE knobs are folklore-tuned at large scale and may be
  mis-set at 20–50M active; routing discreteness plausibly adds eval variance, so "the
  noise-floor stage isn't skippable here, it's more necessary" — (source:
  docs/topics/reference/moe-literature.md).
- **Slicing-and-Dicing MoE sweep** (arXiv 2605.11689; Danielle third author) — ~2,000
  MoEs; final checkpoints confirmed available, no intermediates; a possible MIC-opt-4
  substrate — (source: docs/open-questions-answered.md, 2026-08-21 entry).

**Non-stationarity framing (why "movement" is the right object)**

- **ITER / Transient Non-Stationarity and Generalisation in Deep RL** (Igl et al., ICLR
  2021; arXiv 2006.05826) — transient non-stationarities permanently scar the latent
  representation; distilling into a fresh network launders the trajectory and students
  generalize better than teachers; the one reset that separates *function* from
  *trajectory* and the distillation control MIC-2/MIC-opt-5 sit next to — (source:
  docs/topics/reference/reinit-and-transfer-literature.md;
  docs/topics/reference/nonstationarity-accounting.md).
- **The endogenous self-curriculum reading** — even under iid data the effective
  distribution is data weighted by current gradient magnitude, so movement migrates toward
  harder tokens; makes MIC-3's token-bucket slice a measurement of a mechanism rather than
  a description — (source: docs/topics/reference/nonstationarity-accounting.md).
- **Achille 2019 / Ash & Adams 2020 / Igl 2021 as "three communities, one claim"** — the
  training path leaves damage the final loss doesn't show; the general form of MIC-4's
  matched-final-loss premise — (source:
  docs/topics/reference/nonstationarity-accounting.md).
- **Ash & Adams, On Warm-Starting Neural Network Training** (NeurIPS 2020) and **DASH**
  (NeurIPS 2024) — warm-started models generalize worse at matched training loss;
  DASH argues non-stationarity-motivated plasticity fixes fail in stationary settings;
  relevant to whether MIC's continued-pretraining control is truly a null — (source:
  docs/topics/reference/plasticity.md).

**Method / design-discipline notes that are not papers but are on record**

- **Tuning-response curves instead of matched-budget comparisons** — flat curve = mature
  paradigm, steep curve = headroom; proposed as the honest form of any "our intervention
  worked" claim — (source: docs/potential-projs/movement-microscope.md §4, 2026-08-18
  research-hypothesis entry; docs/research-hypothesis.md).
- **Demonstration hygiene checklist** — pre-specified settings, effect sizes with
  confidence bounds across seeds, replication in a second model family, honest reporting
  of settings searched, a mechanism readout explaining why a ceiling was exceeded —
  (source: docs/potential-projs/movement-microscope.md §4; docs/research-hypothesis.md).
- **A meta-analysis of "how often does the incumbent's advantage survive serious
  re-tuning"** — named as publishable on its own, adjacent to MIC-opt-3 — (source:
  docs/potential-projs/movement-microscope.md §4).
- **Family diversity from the last window** — apply controlled late-window continued
  pretraining to OLMo, Pythia, SmolLM, Llama, Qwen checkpoints instead of pretraining five
  families; rests on the Final Window paper's claim — (source:
  docs/potential-projs/movement-microscope.md §4).
- **The asymmetric design** — full seeded sweep only where cheap, then two or three
  confirmation runs testing a *ranking* the cheap tier predicted — (source:
  docs/potential-projs/movement-microscope.md §4; docs/potential-projs/tiny-scale-
  measurement.md §4).
- **ICL as the gradient-free post-training stage** — makes "seeds" one forward pass and
  sidesteps the elicitation threshold; proxy candidate is the ICL curve (loss on the k-th
  demo vs. k) — (source: docs/potential-projs/movement-microscope.md §4 →
  docs/potential-projs/icl-elicitability.md).

**Provenance items and unresolved records**

- **FollowIR** (arXiv 2403.15246; Kyle Lo co-author) — recorded as an external search's
  *guess* at the AI2 "dataset that moves specific-task metrics," and explicitly **not a
  match** (it is a JHU-led retrieval benchmark); the real question is open and resolves by
  asking the contact — gates MIC-opt-1's post-training data choice — (source:
  docs/topics/reference/pretraining-to-posttraining.md intake notes;
  docs/open-questions-answered.md "Open — not yet checked").
- **"Olmo-3.1-32B-Instruct" card** — cited in the same response for an SFT → DPO →
  verifiable-reward RL pipeline; recorded as **looking fabricated** — (source:
  docs/topics/reference/pretraining-to-posttraining.md intake notes).
- **Citation-verification ledger: ~17 rows tagged `MIC`** — 2306.06625, 2306.13649,
  2402.03898, 2402.12030, 2404.19319, 2406.13555, 2406.17328, 2407.14679, 2407.16154,
  2410.16215, 2410.17215, 2412.14528, 2502.08606, 2502.17927, 2505.16297, 2509.25837,
  2509.26497 — all from the `distillation-literature` intake, all **agent-supplied and
  unverified**; the same sessions produced swapped bibliography entries and a fabricated
  author list — (source: docs/litreview/citation-verification-ledger.md).
- **Distillation review quality flags** — 16 duplicate reference entries ([49]–[60]),
  "Yeongmin et al." is a first name, half the downloaded PDFs off-topic; questions 1, 3,
  and 6 under-evidenced — (source: docs/topics/reference/distillation-literature.md).
- **Artifacts on disk** — the SciSpace distillation bundle (report md + LaTeX/PDF with 10
  figures, 34 PDFs, 11 search CSVs, `INDEX.md` listing the canonical papers the review
  missed) at `~/drotherm/data/convo-artifacts/2026/scispace-llm-distillation-agent-
  artifacts-zip_.../` — (source: docs/topics/reference/distillation-literature.md).

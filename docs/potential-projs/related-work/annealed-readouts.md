# annealed readouts — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`annealed-readouts.md`](../annealed-readouts.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

High-recall corpus for annealed readouts (`ANN-1`–`ANN-6` and the ANN-opt-* arms).
Every item is on record in this repository and cites its source. **Nothing here is
verified**: the annealing and midtraining accumulators are agent-generated, the citation
ledger's header states that none of its 208 rows has been checked, and the in-repo
reproduction numbers carry Danielle's explicit caveat that they come from agent-written
verification code she has not read, debugged, run, or analyzed.

**The confound and its three workarounds (the 2026-08-18 framing this project is built on)**

- **The confound as first stated** (repo record, not literature): DataDecide's models are
  cosine-trained, so every intermediate checkpoint sits mid-schedule with high residual
  LR; in river-valley terms evals measure "position along river + current distance up the
  wall", and post-training from such a checkpoint inherits it. (source:
  docs/topics/reference/schedules-and-annealing-literature.md 2026-08-18;
  docs/potential-projs/annealed-readouts.md §4 origin entry)
- **Hägele et al. 2024, *Scaling Laws and Compute-Optimal Training Beyond Fixed Training
  Durations*** (2405.18392) — stable phase plus cheap decay branches as the methodology;
  the (1-sqrt) cooldown vs. linear is the citation for ANN-3's decay shape and
  ANN-opt-1's sweep; branch reuse is the cost model. (source:
  docs/topics/reference/schedules-and-annealing-literature.md, entries 1 and 2)
- **MiniCPM** (2404.06395) — the ~10%-decay template ANN-3 defaults to; and the
  decay-phase gradient statistics (weights move less than in the stable phase while loss
  falls faster; gradient norm diminishes; consecutive-update cosine turns predominantly
  positive) proposed as a mechanism-level companion readout logged per branch: a branch
  whose updates are not yet consistently aligned has not "read out" yet. Also "loss drop
  in decay ≈ a 5× larger model" and decay-branch reuse at linear rather than quadratic
  cost. (source: docs/potential-projs/annealed-readouts.md §4 2026-08-22;
  docs/topics/reference/schedules-and-annealing-literature.md, second entry)
- **Llama 3 "Annealing Data"** (no ID on record) — 8B GSM8K +24.0 / MATH +6.4, 405B
  negligible; final 40B tokens at 30% new / 70% default with LR linearly to zero, used as
  a data-valuation protocol. The scale-attenuation result bears directly on whether an
  annealed-vs-raw gap at 150M–300M will look like the published ones. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **Blakeney et al., Databricks *Does your data spark joy?*** (no ID on record) — 7B/1T
  end-of-training domain upsampling (MMLU +6.90, GSM8K +8.26, HumanEval +6.17 pp); 10–20%
  of training as the trade-off point, i.e. the budget heuristic behind the default branch
  length. (source: docs/topics/reference/schedules-and-annealing-literature.md)
- **OLMo 2 / Dolmino** (2501.00656) — mid-training mix, LR decayed to zero across the
  phase, annealing used as a data-evaluation tool (30/70): the open reproducible template
  for what ANN-3 emulates. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **OLMo** (no ID on record) — the training setup DataDecide inherits, hence the
  provenance of the cosine schedule being audited. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **WSM, *Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM
  Pre-training*** (no ID on record) — the direct method behind ANN-1: merge recent
  checkpoints with weights from an emulated decay curve to get an annealed model without
  altering the live LR; WSM-merged models reported to closely mirror a true anneal at
  intermediate stages of long runs. Validated on stable-phase runs, which is precisely
  the gap ANN-1 tests. (source:
  docs/topics/reference/schedules-and-annealing-literature.md;
  docs/refs/research-trajectory-pre-to-post-training.md)
- **Nemotron 3 Super** (no ID on record) — sliding-window checkpoint merging as a
  production mid-run readout, ~16% of total pretraining FLOPs saved; the deployment-scale
  evidence that merging-as-readout is already in use. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **The open question, Danielle's seed 2** (repo record): does merging-as-annealing-proxy
  work on *cosine* mid-run checkpoints, where LR varies inside the merge window, rather
  than only on stable-phase ones. This is ANN-1's contribution and its risk. (source:
  docs/potential-projs/annealed-readouts.md §4 origin entry;
  docs/topics/reference/schedules-and-annealing-literature.md)
- **Model soups (Wortsman et al.)** (2203.05482) and the landscape record that
  souping/task arithmetic only work within a basin — the precondition ANN-1's merge
  window inherits. (source: docs/topics/reference/landscape-literature.md;
  docs/litreview/citation-verification-ledger.md)
- **Branch-and-Merge** (2407.08699) — merging models fine-tuned on data subsets yields
  smaller but higher-quality weight changes with less forgetting; filed as adjacent to
  ANN-opt-7's merging angle. (source:
  docs/topics/reference/schedules-and-annealing-literature.md, third entry;
  docs/litreview/citation-verification-ledger.md)

**Analytic correction: the multi-power law and the loss→accuracy chain (ANN-5/ANN-2/ANN-6)**

- **Kairong Luo et al., *A Multi-Power Law for Loss Curve Prediction Across Learning Rate
  Schedules*** (2503.12811, ICLR 2025) — power law on the sum of learning rates plus
  extra power-law terms for the decay-induced loss drop; fitted on a few runs it
  extrapolates to unseen schedules and discovers a WSD-like schedule beating cosine. The
  ANN-5 anchor. Recorded risks: it was fit on runs with explicit schedules, so fitting to
  cosine runs is in scope but the extrapolation to "hypothetical decay from here" on
  these runs is unvalidated — hence the mandatory held-out check. (source:
  docs/topics/reference/loss-curve-forecasting.md;
  docs/potential-projs/annealed-readouts.md §2, §4)
- **The MPL's own river-valley reading** — the paper flags the river-valley conjecture as
  the landscape framework its schedule-dependent terms implicitly model; the
  "decay-induced loss drop" is descent from the walls to the river. This is the bridge
  between ANN-2 and the mechanism story. (source:
  docs/potential-projs/annealed-readouts.md §4 2026-08-18;
  docs/topics/reference/landscape-literature.md)
- **Tissue et al. 2024** (2408.11029) — annealing-area term in the loss-vs-compute law;
  the LR trajectory affects realized loss beyond total tokens. A second analytic
  correction family. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **Second LR-annealing scaling-law citation** (2508.01483) — paired everywhere with
  Tissue ("forward area" vs. "annealing area"; annealing "momentum": LR changes reflected
  in loss with a delay growing with annealing slope; 10–20% annealing ratio). Unknown
  paper; on the open-ID check list as a possible mis-ID. (source:
  docs/topics/reference/schedules-and-annealing-literature.md;
  docs/litreview/citation-verification-ledger.md)
- **Gadre et al. 2024, *Language models scale reliably with over-training and on
  downstream tasks*** (2403.08540 in one accumulator) — downstream accuracy as an
  exponential function of training loss; the loss→accuracy link ANN-6 rides on. Noted
  elsewhere as holding on average but varying by task (104 models, 11M–6.9B). (source:
  docs/topics/reference/loss-curve-forecasting.md;
  docs/topics/reference/small-scale-evaluation-metrics-literature.md)
- **FLP two-stage, Yangyi Chen et al., *Scaling Laws for Predicting Downstream
  Performance in LLMs*** (2410.08527) — FLOPs → pretraining loss → downstream
  performance; 5–10% error at 7B/13B per the metrics accumulator. (source:
  docs/topics/reference/loss-curve-forecasting.md;
  docs/topics/reference/small-scale-evaluation-metrics-literature.md)
- **Model ladders, Bhagia et al., *Establishing Task Scaling Laws via Compute-Efficient
  Model Ladders*** (2412.04403) — compute → task NLL → accuracy; 1% of target compute,
  within 2 points on some tasks; N and D beat FLOPs in overtrained regimes. The ledger
  notes a bibliography swap in which `bhagia2024scaling` points at 2410.08527 instead.
  (source: docs/topics/reference/loss-curve-forecasting.md;
  docs/topics/reference/small-scale-evaluation-metrics-literature.md;
  docs/litreview/citation-verification-ledger.md)
- **The emergence caveat for flips on task metrics** — hard accuracy metrics can look
  emergent, showing no progress above chance until loss crosses a threshold, exactly
  where the loss→accuracy mapping is fragile; ANN-6 measures flips in that regime.
  (source: docs/topics/reference/loss-curve-forecasting.md;
  docs/potential-projs/annealed-readouts.md §4 2026-08-18)
- **Nakkiran et al., *Deep Double Descent*** (no ID on record) — epoch-wise double
  descent means capability is not even monotone in training loss along a run: the
  boundary condition on the whole prediction-law thread and on reading a corrected loss
  as a corrected capability. (source:
  docs/topics/reference/grokking-and-hidden-progress.md;
  docs/topics/reference/loss-curve-forecasting.md)
- **Emergence framing pair: Wei et al.** (2206.07682) vs. **Schaeffer et al. mirage**
  (2304.15004) — the metric-choice reading of apparent emergence, relevant to whether a
  "flip" is a measurement artifact. (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md)
- **Patel et al. 2026, *Forecasting Downstream Performance of LLMs With Proxy Metrics***
  (2605.18607) — 80 proxy metrics from one forward pass over expert trajectories; on
  DataDecide, frequency-weighted top-5 accuracy ranks the 25 corpora for the 1B target
  with decision accuracy >0.85 at ~1e-5 of target compute, and along OLMo-3-7B proxies
  extrapolate downstream accuracy over an 18× compute horizon at roughly half the RMSE of
  loss-based baselines. Directly adjacent: a competing route to the same
  decision-accuracy question ANN-6 audits. Version 2 of the review fabricated its author
  list; prefer v1's bibliography. (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md)
- **Other forecasting rows from the same review** — NeuNeu "Neural Neural Scaling Laws"
  (2601.19831); observational scaling laws (2405.10938); Krajewski et al. (2512.08894,
  direct power law for log-accuracy beating the two-stage loss→accuracy route); Ye et al.
  BIG-bench predictability (2305.14947); context-aware scaling (2510.14919); Pechi et al.
  small-scale break below ~2.2e15 FLOPs (2305.17266). All bear on how much a corrected
  loss can be expected to carry. (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md)
- **Learning-curve extrapolation lineage** — Domhan, Springenberg & Hutter 2015; Klein et
  al. 2017 (LC-Net); Baker et al. 2017; LC-PFN (2310.20447); Ding et al. (2412.15554,
  architecture-aware neural ODE). Recorded for the early-dynamics project but the same
  "fit a curve family, extrapolate" machinery as ANN-5. (source:
  docs/topics/reference/loss-curve-forecasting.md)

**Ground truth: the decay branch as an instrument (ANN-3, ANN-opt-1)**

- **Wen et al., river valley** (2410.05192) — the canonical statement; the interpolation
  signature as the closest thing to a river test; they validate by branching a constant-LR
  run at 20B tokens, decaying 5B and interpolating, and their WSD-S variant resumes from
  decayed checkpoints. Also: valley geometry attributed to token determinism, so the
  wall height is plausibly recipe-dependent — the link to the determinism-features seed.
  (source: docs/topics/reference/landscape-literature.md;
  docs/topics/staging/checkpoint-tomography.md)
- **The tomography verdict on branching** — "'branch + decay + measure the loss drop' is
  established; the drop is your height-above-river statistic. What's not established:
  doing it on cosine mid-run checkpoints, and treating the per-token profile of the drop
  as the statistic rather than the scalar." The sharpest statement on record of what
  ANN-3 adds. (source: docs/topics/staging/checkpoint-tomography.md)
- **The four/five-probe battery** (staging, no external ID) — decay branch → wall height;
  hot branch → diffusion width; twin branches → sibling barrier / basin commitment;
  data-shifted branch → component responsiveness; plus a reset branch. ANN-3's runner is
  the shared instrument. (source: docs/topics/staging/checkpoint-tomography.md)
- **Frankle et al., *Linear Mode Connectivity and the Lottery Ticket Hypothesis*** (no ID
  on record) — the twin-branch probe and the commitment clock; caveats that the originals
  train children to completion, are mostly pre-LLM vision work, and have never been run
  across data recipes. (source: docs/topics/staging/checkpoint-tomography.md;
  docs/topics/reference/landscape-literature.md)
- **Devinterp / local learning coefficient (Lau, Murfet et al.; Timaeus)** (no ID on
  record) — short SGLD chains estimate local degeneracy, tracked across Pythia-style
  checkpoint sequences and reported to detect developmental transitions; named as one of
  two communities most likely to have partial versions of the battery in flight. (source:
  docs/topics/staging/checkpoint-tomography.md)
- **Critical-sharpness statistic across public checkpoints; the basin-emergence line**
  (no IDs on record) — single-checkpoint geometry probes as covariates rather than
  movement measures. (source: docs/topics/staging/checkpoint-tomography.md)
- **PolyPythias** (2503.09543) — the many-seed substrate named in the reset-probe entry;
  relevant if branch statistics need seed power ANN's 3 seeds cannot supply. (source:
  docs/topics/staging/checkpoint-tomography.md)
- **Scaling with Collapse** (2509.25087) — curve collapse as a weight-free cross-run
  comparability criterion; a cheap check on whether corrected curves belong to one
  family. (source: docs/topics/reference/landscape-literature.md)
- ***Training Dynamics of the Cooldown Stage in WSD*** (no ID on record) — the cooldown
  visualized in pre-cooldown→final vs. local-Adam-step coordinates. (source:
  docs/topics/reference/landscape-literature.md)

**Basin and comparability machinery behind merging (ANN-1's assumptions)**

- **Frankle et al. LMC; Entezari et al. permutation invariance; Ainsworth et al. *Git
  Re-Basin*; *Unveiling Linear Mode Connectivity of Re-Basin from Neuron Distribution
  Perspective*; *Going Beyond Linear Mode Connectivity: Layerwise Linear Feature
  Connectivity*; *On the Emergence of Cross-Task Linearity in the Pretraining-Finetuning
  Paradigm*; *Beyond Structural Symmetries: LMC via Neuron Identifiability* (2026)** (no
  IDs on record) — the toolkit of pairwise basin tests, the finding that re-basin methods
  often reduce barriers only marginally and work poorly early in training, and the reason
  merging works only within a basin. (source:
  docs/topics/reference/landscape-literature.md)
- **Juneja et al., *Linear Connectivity Reveals Generalization Strategies*** (ICLR 2023,
  no ID on record) — models in different basins implement different generalization
  strategies despite similar in-distribution accuracy; the strongest recorded evidence
  that "same metric value, different basin" can mean different mechanisms. (source:
  docs/topics/reference/landscape-literature.md)
- **MoE merging caution (PART-4)** — checkpoint merging needs expert matching first or it
  averages mismatched experts into mush; ANN-1 is therefore dense-only as specified.
  (source: docs/topics/reference/moe-literature.md;
  docs/potential-projs/annealed-readouts.md §4 2026-08-18)

**The evaluation-side neighbor and the noise floor (ANN-opt-6, ANN-opt-7)**

- **Heineman et al., *Signal and Noise: A Framework for Reducing Uncertainty in Language
  Model Evaluation*** (NeurIPS 2025 per the record; 2508.13144 per the ledger, a
  Claude-added and hallucination-prone row) — signal vs. noise, continuous metrics beating
  accuracy, filtering noisy subtasks; release of ~900K results on 465 models including
  OLMo checkpoints, DataDecide and the ladders. The "noise" term ANN-opt-7's
  durable-movement operator proposes to decompose into measurement noise, wall
  oscillation and unresolved drift; its authors are also flagged as the obvious people to
  run the T0 reanalysis. (source:
  docs/topics/reference/evaluation-methodology-literature.md;
  docs/potential-projs/annealed-readouts.md §4 2026-08-21;
  docs/litreview/citation-verification-ledger.md)
- **OLMES, *A Standard for Language Model Evaluations*** (no ID on record) — the eval
  standard the results store mirrors; the basis for the recorded fact that re-evaluating
  a fixed checkpoint with new seeds buys nothing, so the variance of interest is in
  training. (source: docs/topics/reference/evaluation-methodology-literature.md)
- **DataDecide**, Magnusson et al. (2504.11393, Ai2 ICML 2025) — the suite under audit:
  25 corpora, sizes to 1B, 3 seeds; single-150M ranking predicts the 1B best dataset ~80%
  of the time; continuous likelihood metrics make MMLU/ARC/HellaSwag/MBPP/HumanEval >80%
  predictable at 0.01% of compute. Its published intermediate-vs-final claim is what
  ANN-6 re-tests. (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/topics/reference/datadecide-data-pipeline.md)
- **The compute-matched verifier record (in-repo, not literature)** — the DataDecide
  claim that an intermediate checkpoint predicts rankings as accurately as a
  compute-matched final checkpoint was classified `not_reproduced`, but the verifier
  required exact floating-point compute equality and found zero matched pairs; the
  validator itself recommended `not_assessable`. Design consequences recorded for ANN-4:
  tolerance in log-compute space, predeclared, with match distances reported, or
  interpolation along each run's compute axis; sweep the tolerance; add a
  "predicate liveness" guard. Carries Danielle's provenance caveat. (source:
  docs/potential-projs/annealed-readouts.md §4 2026-08-22;
  docs/topics/reference/datadecide-data-pipeline.md;
  src/datadec/paper/verifiers/single_scale.py ~line 1341 on `main`)
- **Checkpoint-spacing and coverage facts (in-repo)** — ~1,000–1,300-step spacing from 8M
  to 530M with 30–40 checkpoints at 150M–530M; 1B coarsest at ~2,500; the 750M aggregate
  table truncated at step 26,250 while the instance table runs to 63,599; per-instance
  coverage supports seed-variance work only at 150M–1B. These bound the branch grid and
  the flip analysis. (source: docs/open-questions-answered.md)
- **Spread-to-noise and crossings (in-repo reproduction)** — Spearman ≈0.798 between
  predictability and the spread-to-noise ratio across 160 task/metric observations, and
  15,523 crossings across all 300 pairs; both under Danielle's caveat about
  agent-written verification code. Context for how much of the flip signal is noise.
  (source: docs/topics/reference/datadecide-data-pipeline.md;
  docs/potential-projs/trajectory-statistics.md §4)

**Hidden progress and the anti-grokking reading of the branch**

- **Power et al., *Grokking*** and **Nanda et al., *Progress Measures for Grokking via
  Mechanistic Interpretability*** (no IDs on record) — train loss at floor and test loss
  at chance while the generalizing circuit assembles; the field needed non-loss
  observables. Two checkpoints matched on train *and* test loss can sit at different
  points of circuit maturity, making matched-loss pairs a necessary-but-provably-
  insufficient control. (source:
  docs/topics/reference/grokking-and-hidden-progress.md)
- **The anti-grokking-instrument reading** — grokking plateaus as travel along the river
  that loss cannot see, so the decay branch is a probe that *reveals* accumulated hidden
  river progress. The specific paper on high-LR plateaus accelerating final convergence is
  not named in the record. (source:
  docs/topics/reference/grokking-and-hidden-progress.md;
  docs/potential-projs/annealed-readouts.md §4 2026-08-18)
- ***What Can Grokking Teach Us About Learning Under Non-Stationarity*** (no ID on
  record) — the warm-starting gap and grokking analyzed as one delayed-generalization
  phenomenon with effective LR as the shared knob. (source:
  docs/topics/reference/grokking-and-hidden-progress.md)

**Post-training flanks (ANN-opt-3, ANN-opt-8)**

- ***Similar Models Learn Differently: Final-Window Pretraining Shapes Post-Training
  Beyond SFT*** (2607.25063) — the closest published design; models similar after SFT
  diverge under identical post-training depending on late-pretraining interventions. The
  basis for ANN-opt-8's cross-family late-window design (OLMo, Pythia, SmolLM, Llama,
  Qwen with the decay-branch runner as instrument). (source:
  docs/topics/reference/pretraining-to-posttraining.md;
  docs/potential-projs/annealed-readouts.md §4;
  docs/potential-projs/movement-microscope.md)
- ***Echo Chamber*** (2504.07912) — controlled-mixture pretraining then PPO/GRPO/Expert
  Iteration across scales; small controlled proxies as valid RL testbeds. (source:
  docs/topics/reference/pretraining-to-posttraining.md)
- ***Front-Loading Reasoning*** (2510.03264); ***Early Data Exposure*** (2605.12705);
  ***The Finetuner's Fallacy*** (2603.16177); ***Understanding Reasoning from Pretraining
  to Post-Training*** (2607.16097) — the pretraining-choices→post-training cluster, the
  last in recorded tension with the beyond-final-loss hypothesis. (source:
  docs/topics/reference/pretraining-to-posttraining.md)
- **The "post-training did nothing" cluster** — *A Sober Look* (2504.07086); *Spurious
  Rewards* (2506.10947); Yue et al. (2504.13837); Wu & Choi *On the Limits of RLVR*;
  counterpoints *The Invisible Leash* (2507.14843) and 2506.14245. The hindsight reading
  of the earlier negative result ANN-opt-3 revisits, with the three named causes
  (elicitation ceiling, Qwen confound, benchmark noise without multi-seed evaluation).
  (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/potential-projs/annealed-readouts.md §4 2026-08-18)
- **Small-model post-training cautions** — Chen et al. (2505.17988) and *Through the
  Valley* (2506.07712). (source:
  docs/topics/reference/pretraining-to-posttraining.md)
- **Tulu / Tulu 3** (no ID on record) — the SFT data of the earlier project;
  the account of record is Danielle's own. **FollowIR** (2403.15246) is a respondent's
  guess at "the AI2 dataset by Kyle" that the intake note rejects; the question is open.
  (source: docs/topics/reference/pretraining-to-posttraining.md)
- ***Provable Benefits of RLVR over SFT…*** and **TinyZero** (no IDs on record) — cheap
  causal RL testbeds noted in the experiment-design discussion. (source:
  docs/topics/reference/pretraining-to-posttraining.md)
- ***Eliciting Fine-Tuned Transformer Capabilities via Inference-Time Techniques*** and
  Bornschein, Lyle, Pascanu et al., *Fine-Tuned In-Context Learners* (no IDs on record) —
  ICL vs. fine-tuning as access routes to the same capabilities; relevant to whether a
  post-annealing gain is elicitation. (source:
  docs/topics/reference/pretraining-to-posttraining.md)

**Changed-mixture branches (ANN-3 with a shifted mixture = controlled midtraining)**

- **Pre-/mid-training/RL interplay study** (2512.07783) — surfaced by Danielle's SciSpace
  midtraining review; its fixed-compute comparison is a framing precedent. The review is
  noted to have surfaced little else LM-specific and to have missed the LM
  midtraining/annealing canon (DAPT/TAPT, Dolmino, OctoThinker, DSIR/DoReMi). (source:
  docs/topics/reference/targeted-pretraining-midtraining-literature.md;
  docs/potential-projs/annealed-readouts.md §4 2026-08-22; docs/topics/README.md)
- **Additional rows from the same review tagged ANN** — 2506.20512 (mid-training data
  line incl. OctoThinker; Claude-added row) and 2306.12070 (task-robust minimax
  pretraining). (source: docs/litreview/citation-verification-ledger.md)
- **The decay-data "what and when" set, inherited from the annealing accumulator** — TREC
  (2509.25380, receptivity valley, TRECs predictable from AdamW's EMA timescale); PDPC
  (2501.13126); AutoScale (2407.20177); Data Mixing Laws (2403.16952); UtiliMax/MEDU
  (2501.11747); Parmar et al. two-stage CPT (no ID); temperature-sampling mixture cooldown
  (2410.04579); FineWeb-Edu (2406.17557); Phi-4 (2412.08905); Nemotron-CC (2412.02595);
  YuLan-Mini (2412.17743); contamination survey (2503.17793); the rewriting cluster
  SwallowCode/Math (2505.02881), ProX (2409.17115), FinerWeb-10BT (2501.07314); curricula
  (2506.11300, 2508.15475). Each is a candidate mixture for a changed-mixture branch.
  (source: docs/topics/reference/schedules-and-annealing-literature.md;
  docs/topics/staging/rewritten-anneal-slice.md)

**Token-level readouts on branch endpoints (ANN-opt-4, and the per-token profile idea)**

- **Rho-1, *Not All Tokens Are What You Need for Pretraining*** (no ID on record) —
  classifies tokens by loss trajectory across checkpoints (persistently-high,
  persistently-low, descending, fluctuating/ascending), finding only a minority descend
  late; literally the taxonomy ANN-opt-4 reproduces, built for data selection with no
  landscape interpretation. (source: docs/topics/reference/token-level-literature.md)
- ***Token-Level Uncertainty-Aware Objective for Language Model Post-Training*** (no ID
  on record) — epistemic vs. aleatoric token uncertainty; epistemic drains faster for
  low-aleatoric tokens, so a token's apparent bucket migrates over training. The closest
  existing measurement of bucket migration, never connected to landscape geometry.
  (source: docs/topics/reference/token-level-literature.md)
- **Wen et al.'s token mechanism** (2410.05192) — toy bigram language reproducing the
  geometry; stable phase learns deterministic tokens, decay phase learns stochastic ones;
  Spearman ≈0.39 between token uncertainty and local sharpness on real data. The
  prediction a per-token branch profile tests. (source:
  docs/topics/reference/token-level-literature.md;
  docs/topics/reference/landscape-literature.md)
- **RLVR token-regime results** — *Revisiting Entropy in RL for Large Reasoning Models*
  and *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL* (no IDs on
  record) — the wall-token bucket resurfacing as the locus of post-training; a suggestive
  bridge for ANN-opt-3. (source: docs/topics/reference/token-level-literature.md)

**Program frame**

- **Non-stationarity accounting** (repo reference topic) — the LR schedule as *exogenous*
  non-stationarity that pretraining launders through the schedule and pipeline; annealed
  readouts is the schedule's accounting instrument. Includes the ITER record (Igl et al.,
  ICLR 2021: transient non-stationarity permanently scars the representation) and the
  Achille 2019 / Ash & Adams 2020 / Igl 2021 "three communities, one claim" reading.
  (source: docs/topics/reference/nonstationarity-accounting.md)
- **IRT accumulator** (Lalor et al.; Rodriguez et al. ACL 2021; tinyBenchmarks, ICML
  2024; metabench) — the binary-vs-continuous IRT comparison would replicate Signal and
  Noise's metric-choice finding inside one framework, and recipe-DIF is the other
  instrument for the same beyond-final-performance claim ANN-6 touches. (source:
  docs/topics/reference/irt-literature.md)

**Standing caveats to carry**

- Every annealing-literature entry is agent-generated and marked unverified in its own
  file; three separate agent answers to the same annealing-data question exist and only
  the Oct-2025 survey is judged usable. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- The 2026-08-22 compute-matched entry carries Danielle's explicit caveat that the
  reproduction numbers come from agent-written verification code she has not read,
  debugged, run, or analyzed, and its "most important finding" framing was withdrawn as a
  verifier bug on our side. (source:
  docs/potential-projs/annealed-readouts.md §4;
  docs/topics/reference/datadecide-data-pipeline.md)
- The ledger's Signal-and-Noise identifier 2508.13144 is a Claude-added row, flagged as
  hallucination-prone in last digits and title–ID pairing. (source:
  docs/litreview/citation-verification-ledger.md)
- Do-not-re-flag term collisions: Annealed-RLVR 2509.23629; RLHFuse; the 2020 "data
  annealing" paper 2004.13833; and the instruction-tuning/continual-learning drift set
  (2411.11266, 2403.01244, 2508.03571, 2312.11508, 2406.08811, 2310.05492, 2312.10793,
  2501.00237, 2405.17830, 2308.08747, 2410.10210, 2505.05427, 2412.06724). (source:
  docs/topics/reference/schedules-and-annealing-literature.md;
  docs/litreview/citation-verification-ledger.md)

# tiny scale measurement — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`tiny-scale-measurement.md`](../tiny-scale-measurement.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*Recall corpus for `tiny-scale-measurement.md` (TINY). Highest-recall inventory of every
paper, method, or named prior-art item on record anywhere in this repository that is
possibly relevant to this project. Errs toward inclusion; one line per item with its repo
source. Nothing here is verified — the SciSpace reviews, novelty checks, and NBLM-style
tables are agent-generated records and the ledger marks every arXiv ID unverified. No
positioning claims; inventory and attribution only.*

**The incumbent proxy family (the thing TINY-1's method axis must contain)**

- **Patel, Reddy, Mosbach & Bahdanau 2026, *Forecasting Downstream Performance of LLMs
  With Proxy Metrics*** (arXiv 2605.18607) — 80 proxies = 10 token-level statistics
  (cross-entropy, top-k accuracy k∈{1,2,3,5}, entropy, rank, reciprocal rank, margin,
  wrong-confidence) × 8 weightings from one forward pass over expert-written solution
  trajectories; RankSVM leave-2-tasks-out ρ 0.81 vs. 0.36 for FineWeb cross-entropy and
  0.33 for rBridge; ranks DataDecide's 25 corpora for the 1B target at decision accuracy
  > 0.85 at ~10⁻⁵ of target compute; also forecasts along OLMo-3-7B at ~half the RMSE of
  loss baselines. Named the incumbent TINY-1 must beat and the source of the
  expert-trajectory trick. Mila/McGill + ServiceNow (one review version fabricated an
  AI2 author list) — (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md;
  docs/potential-projs/tiny-scale-measurement.md §4, 2026-08-22).
- **"rBridge" (Koh & Liang 2026, no ID)** — the expert-reweighted-NLL baseline Patel et al.
  report at ρ 0.33; the ledger flags it unverifiable and likely fabricated — (source:
  docs/litreview/citation-verification-ledger.md, row `(no ID)` tagged TINY).

**Measurement-intervention precedent (the frame TINY-1 extends downward)**

- **Heineman et al., *Signal and Noise: A Framework for Reducing Uncertainty in Language
  Model Evaluation*** (NeurIPS 2025 per the record; ledger ID 2508.13144, Claude-added) —
  signal = separating better from worse models, noise = sensitivity to step-to-step
  variability; continuous metrics beat accuracy, noisy-subtask filtering helps; released
  ~900K results on 465 models including DataDecide, OLMo intermediate checkpoints, and the
  ladder runs. The stated basis for "measurement interventions improve decision accuracy"
  — (source: docs/topics/reference/evaluation-methodology-literature.md;
  docs/potential-projs/tiny-scale-measurement.md §4).
- **OLMES, *A Standard for Language Model Evaluations*** (NAACL Findings 2025 per the
  record) — the standardized loglikelihood harness whose gold-span per-character correct
  probability is TINY-opt-5's gold-span case, and whose formats were standardized because
  of small-model format sensitivity — (source:
  docs/topics/reference/evaluation-methodology-literature.md;
  docs/topics/reference/datadecide-data-pipeline.md).
- **The fixed-checkpoint variance rule** — re-evaluating a fixed checkpoint with new seeds
  buys nothing (inference nondeterminism negligible for loglikelihood evals; few-shot
  configuration variance is a systematic bias axis to sweep, not noise to average), so
  TINY-3's minimum-detectable-effect estimates must come from training-side replicates and
  item bootstraps — (source: docs/topics/reference/evaluation-methodology-literature.md,
  2026-08-18; docs/potential-projs/tiny-scale-measurement.md §4).
- **Pooling / trajectory-window / item-bootstrap recipe for n=3 seeds** — pool seed variance
  across 25 recipes at fixed scale (~50 df), use a window of late checkpoints within one run
  as replicates (Signal-and-Noise's own trick), and bootstrap over items for benchmark-
  composition uncertainty — (source: docs/refs/research-trajectory-pre-to-post-training.md).
- **DataDecide itself (Magnusson et al., Ai2, ICML 2025, arXiv 2504.11393)** — 25 corpora,
  ≤1B, 3 seeds, 150M ranking predicts the 1B best dataset ~80% of the time, beating 8
  scaling-law baselines; continuous likelihood proxies make benchmarks >80% predictable at
  0.01% compute. The premise TINY extends downward — (source:
  docs/topics/reference/pretraining-to-posttraining.md, 2026-08-18;
  docs/litreview/citation-verification-ledger.md).

**The which-loss axis (TINY-opt-5): loss-replacement metrics**

- **LongPPL / key-token perplexity (Fang et al., 2410.23771)** — perplexity computed only on
  "key tokens" selected by a long-context-influence score from a reference model; Pearson
  −0.96 with long-context accuracy where ordinary PPL is uninformative. The published
  template for reference-model token selection — (source:
  docs/topics/reference/loss-alternative-metrics-literature.md; SciSpace review, unverified).
- **PPLqa (Friedland et al., 2411.15320)** — |PPL(prompt+response) − PPL(response)| as an
  unsupervised reference-free response-quality score — (source:
  docs/topics/reference/loss-alternative-metrics-literature.md).
- **Rho-1 / "Not all tokens are what you need" (2404.07965)** — training-time reference-model
  token selector whose scorer equally defines an evaluation subset — (source:
  docs/topics/reference/loss-alternative-metrics-literature.md).
- **Bits per byte / per character; Biderman et al., *Lessons from the trenches*
  (2405.14782)** — the standard tokenizer-independent normalization; also ByteFlow,
  SuperBPE, MrT5, and the Script Tax BPC study as users — (source:
  docs/topics/reference/loss-alternative-metrics-literature.md).
- **Paloma per-domain BPB protocol (2312.10523)** — the per-domain bits-per-byte reporting
  form recorded as missing canon for this question; shared with DCARD — (source:
  docs/topics/reference/loss-alternative-metrics-literature.md;
  docs/litreview/citation-verification-ledger.md, Claude-added).
- **Marginal likelihood over tokenizations (Cao & Rimell, EMNLP 2021)** — sum over valid
  segmentations instead of the canonical one; larger gains out of domain — (source:
  docs/topics/reference/loss-alternative-metrics-literature.md).
- **Vieira et al. (2412.03719)** — exact token-level → character-level LM conversion giving
  character-string probabilities and tokenization-marginal perplexity — (source: same).
- **Takahashi et al. 2019** — why per-character and per-word perplexities are
  incommensurable — (source: same).
- **Brown et al. 1992 (1.75 bits/char)** — the information-theoretic anchor — (source: same).
- **Delétang et al., *Language Modeling Is Compression* (2309.10668)** — the canonical
  compression statement the SciSpace review omitted — (source: same, missing-canon note).
- **Information capacity (Yuan et al., 2511.08066)** and **entropy-estimation modelling
  (Badger et al., 2511.10618)** — compression-based scores including tokenizer efficiency —
  (source: same).
- **Diff-eRank (Wei et al., 2401.17139)**, **Matrix Nuclear-Norm (Li et al., 2410.10672)**,
  **hybrid (Vo, 2410.14480)** — representation-side hidden-state readouts; the §4 entry
  rules these out as a TINY method and routes them to GEO/TOK — (source:
  docs/topics/reference/loss-alternative-metrics-literature.md;
  docs/potential-projs/tiny-scale-measurement.md §4, 2026-08-22).
- **Semantic-distance scoring of predictions (O'Neill et al. 2019)** — score wrong
  predictions by embedding distance to target rather than 0/1 — (source: same).
- **The exp-relation finding: `correct_prob = exp(sum_logits_corr)`** (checked on two
  released rows) — as *ranking* metrics they are one metric, so TINY-opt-5's candidate list
  must not count them twice; per-char variants are per-character geometric means, not
  probability ÷ length — (source: docs/topics/reference/datadecide-data-pipeline.md, OLMES
  metric-column entry; docs/potential-projs/tiny-scale-measurement.md §4, 2026-08-22).
- **`uncond_correct_prob` (Danielle-proposed continuous proxy)** — the correct option's
  likelihood with the answer-string prior removed; endorsed in conversation without
  evidence; cheap to add to the TINY-opt-5 sweep once its definition is pinned from the
  evaluation code — (source: docs/potential-projs/tiny-scale-measurement.md §4;
  docs/topics/reference/datadecide-data-pipeline.md).
- **DataDecide's own metric hierarchy from the reproduction** — length-normalized correct
  probability wins everywhere (816/830 pairwise near target; per-character won 9/10 tasks);
  margin tracks accuracy at ρ 0.360 vs. Normalized Correct Probability at 0.916; raw
  likelihood and margin each have documentable failure modes. Agent-written verification
  code Danielle has not read or rerun — flags, not findings — (source:
  docs/topics/reference/datadecide-data-pipeline.md; docs/potential-projs/irt-reanalysis.md §4).

**Downstream forecasting, scaling laws, and the loss→accuracy link**

- **Gadre et al. 2403.08540** — perplexity→downstream power law holds on average but varies
  by task; 104 models 11M–6.9B; downstream accuracy as an exponential function of training
  loss — (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md;
  docs/topics/reference/loss-curve-forecasting.md).
- **FLP two-stage loss→performance (Chen et al., 2410.08527)** — FLOPs → pretraining loss →
  downstream performance; 5–10% error at 7B/13B — (source: both files above).
- **Model ladders (Bhagia et al., 2412.04403)** — compute → task NLL → accuracy at 1% of
  target compute, within 2 points on some tasks; N and D beat FLOPs when overtrained —
  (source: both files above).
- **Observational scaling laws (2405.10938)** — ~80 public models, low-dimensional capability
  space, emergence as sigmoids — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **NeuNeu, "Neural Neural Scaling Laws" (2601.19831)** — accuracy-trajectory extrapolation +
  token-level validation losses; 2.04% MAE on 66 tasks vs. 3.29% for logistic fits;
  zero-shot to unseen families — (source: same).
- **Ye et al., BIG-bench predictability (2305.14947)** — MLP predictor, >95% R²; a
  "small-bench" 3× smaller than BBH is equally informative — (source: same).
- **Pechi et al. (2305.17266)** — small-scale break below ~2.2e15 FLOPs; a direct bound on
  how far down decision signal survives — (source: same).
- **ADO (2410.11820)** — small proxy models often fail to predict larger ones; the
  counterweight to the whole tiny-proxy thesis — (source: same).
- **Krajewski et al. (2512.08894)** — a direct power law for log-accuracy at fixed
  tokens-per-parameter beats the two-stage loss→accuracy route — (source: same).
- **Ali et al. (2310.08754)** — tokenizer metrics uncorrelated with downstream — (source: same).
- **Dudy et al. 2020** — soft-match accuracy — (source: same).
- **Kaplan; Chinchilla; Ivgi et al. 2022 (finetuning scaling laws need R² ≥ 0.95);
  data-constrained scaling (2305.16264); quality-aware Q (2510.03313); effective tokens =
  diversity × syntheticity (2410.03083, r = 0.83 over 200 models 25M–1.5B); context-aware
  scaling (2510.14919); hyperparameter scaling (2505.13738)** — the scaling-law flank of the
  downstream-prediction question — (source: same).
- **Loss-to-loss scaling determined by data and tokenizer, not architecture (2502.12120)** —
  bears on whether a tiny-scale metric transfers across suites — (source: same).
- **Knowledge capacity 2 bits/parameter (2404.05405)**; **repeated-data double descent
  (2205.10487)** — capacity/repetition bounds at tiny parameter counts — (source: same).
- **Data-mixture / selection laws: AutoScale 2407.20177; UtiliMax / MEDU 2501.11747; D-CPT
  2406.01375; BiMix 2405.14908; data mixing laws 2403.16952; optimal-mixture laws
  2507.09404** — the selection setting DataDecide's corpora ranking sits in — (source: same).
- **Multi-power law for loss curves (Luo et al., 2503.12811, ICLR 2025)** — predicts the full
  loss curve across LR schedules from a functional of the schedule; the analytic backbone if
  TINY reads loss trajectories — (source: docs/topics/reference/loss-curve-forecasting.md).
- **Double descent (Nakkiran et al.; also 2407.09845, 2203.07337)** — capability not monotone
  in training loss along a run: a boundary condition on any loss-based proxy — (source:
  docs/topics/reference/loss-curve-forecasting.md; grokking-and-hidden-progress.md pointer).
- **Early stabilization of larger models (2410.11451)** — larger LMs stabilize within the
  first 20% of training while smaller ones converge slower and less stably; directly about
  the small-scale noise regime — (source: docs/topics/reference/loss-curve-forecasting.md).
- **AdaGC (2502.11034)** — removes loss spikes while preserving convergence patterns;
  relevant to cleaning small-scale curves — (source: same).
- **Learning-curve extrapolation line: Domhan, Springenberg & Hutter 2015; Klein et al. 2017
  (LC-Net); Baker et al. 2017; LC-PFN (Adriaensen et al., NeurIPS 2023, 2310.20447); Ding et
  al. 2024 (2412.15554)** — the "fit early curve, predict the end" ancestry; LC-PFN named as
  the modern extrapolation baseline — (source: docs/topics/reference/loss-curve-forecasting.md).
- **Zero-cost NAS proxies (ICLR blog-track 2022 survey; AZ-NAS CVPR 2024, 2403.19232;
  FreeREA WACV 2023); neural capacitance (Jiang et al., 2201.04194)** — the init-time end of
  the cheap-predictor spectrum — (source: docs/topics/reference/loss-curve-forecasting.md;
  docs/topics/reference/nas-literature.md).

**Emergence, and the mirage debate**

- **Wei et al. (2206.07682)** — the emergent-abilities claim — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **Schaeffer et al., *Are Emergent Abilities a Mirage?* (2304.15004)** — metric choice
  manufactures discontinuities; the record notes it swapped metrics ad hoc without a
  principled framework — (source: same; docs/potential-projs/irt-reanalysis.md §4).
- **Proxy tasks for emergent abilities (2412.07111)** — (source: same accumulator).

**Small-model benchmarks and eval-quality flank**

- **SLM-Bench (2508.15478)** — 15 SLMs, 9 tasks, 11 metrics including energy — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **SLM survey (2409.15790)**; **SLMs on code (2507.03160)** — (source: same).
- **ReTraceQA (2510.09351)** — answer-only metrics overstate SLM reasoning by up to 25%; 24%
  of flawed traces still score correct — a validity warning for tiny-scale accuracy —
  (source: same).
- **Generative→NLU reformulation for 35× cheaper evaluation (2506.03592)** — (source: same).
- **Informedness over accuracy/F1 (2401.03831)** — chance-corrected scoring, directly
  relevant where most items sit at chance — (source: same).
- **Reference-based metrics failing for modern models (2310.13800)** — (source: same).
- **Probes and uncertainty: linear probes on activations (Pacchiardi 2025, poor on math);
  conformal probes (Ashok & May 2025); conformal set size as a benchmark axis (2401.12794);
  self-evaluation EQT (2501.11721); Kadavath et al. calibration (2207.05221)** — candidate
  continuous readouts where accuracy is degenerate — (source: same).
- **Learned predictors: Schellaert et al. (2305.12415, DeBERTa assessors predicting
  per-instance success); ProxyLM (2406.09334, 37× speedup); lineage-regularized matrix
  factorization (2504.19811); FamiCom (2406.11243, familiarity × complexity, ρ 0.848)** —
  (source: same).
- **Contamination: time-travel detection (2308.08493); C2LEVA (2412.04947)** — (source: same).

**Many-seed substrates and scaling results in the 5M–410M range**

- **PolyPythias (ICLR 2025; arXiv 2503.09543)** — 50 pretraining runs, 9 seeds × 5 sizes
  (14M–410M), ~7,000 checkpoints: a released many-seed substrate in exactly TINY's range —
  (source: docs/topics/reference/reinit-and-transfer-literature.md;
  docs/potential-projs/tiny-scale-measurement.md §4, 2026-08-22).
- ***Can Scale Save Us From Plasticity Loss in LLMs?* (arXiv 2606.24752)** — plasticity loss
  at 5M–314M follows a sublinear scaling law in continual and stationary settings — (source:
  same).
- ***The Butterfly Effect* (arXiv 2506.13234)** — trajectories highly sensitive to initial
  conditions; the seed-count argument for replicate-heavy design — (source: same).
- **Spectral collapse (2509.22335); activation design (2509.22562); calibrated partial resets
  (2607.24996); plasticity-loss survey in RL (2411.04832); plasticity injection (Nikishin et
  al., 2305.15555); Reset & Distill (2403.05066); *When Does Re-initialization Work?* (Zaidi
  et al., 2206.10011, >15,000 runs)** — the plasticity/reset flank a tiny-scale substrate can
  test with real replicates — (source: docs/topics/reference/reinit-and-transfer-literature.md).
- ***Loss Curves Hide Stepwise Token Learning* (2606.29858)** — aggregate curves conceal
  item-level structure; the same argument TINY makes about accuracy — (source: same).

**Post-training / RL prior art for TINY-opt-2 and the RL options**

- **Chen et al. (2505.17988)** — small-scale SFT on Qwen2.5-1.5B lowers MATH-500 from 23.8%
  to 18.4% while eliciting reasoning-style behavior — (source:
  docs/topics/reference/pretraining-to-posttraining.md;
  docs/potential-projs/tiny-scale-measurement.md §4, 2026-08-18).
- **Luo et al., *Through the Valley* (EMNLP 2025; 2506.07712)** — Long-CoT degradation in
  small LMs attributed to error accumulation — (source: same).
- **Shao et al., *Spurious Rewards* (ICML 2026; 2506.10947)** — GRPO with random rewards gains
  21.4 points on Qwen2.5-Math-7B; comparable spurious rewards fail outside Qwen families —
  (source: same).
- **Hochlehnert et al., *A Sober Look at Progress in LM Reasoning* (COLM 2025; 2504.07086)** —
  RL on distillation-based models yields little significant gain; "small benchmarks produce
  unstable estimates, making multiple seed runs essential" — the replicate-heavy design's
  direct citation — (source: same).
- **Yue et al. (NeurIPS 2025 oral; 2504.13837)** — RLVR improves pass@k at small k but does
  not expand the base model's reasoning boundary at large k — (source: same).
- **Wu & Choi, *On the Limits of RLVR* (ICML 2025 AI for Math workshop)** — RLVR as
  support-preserving, entropy-reducing reweighting; counterpoints *The Invisible Leash*
  (2507.14843) and *RLVR Implicitly Incentivizes Correct Reasoning* (2506.14245, ICLR 2026) —
  (source: same).
- **TinyZero** — RL visibly works at 0.5–3B on countdown and simple arithmetic; the
  "RL works small on synthetic families" evidence for TINY-opt-2 — (source: same, 2026-08-18).
- ***Provable Benefits of RLVR over SFT for Reasoning Models: Learning to Backtrack
  Efficiently*** — graph-pathfinding synthetic testbed where "a seed costs minutes";
  causal rather than correlational — (source: same).
- **Echo Chamber (Zhao, Meterez et al., COLM 2025; 2504.07912)** — trains from scratch on
  controlled pretraining mixtures then compares PPO/GRPO/Expert Iteration across scales;
  argues controlled small-model proxies yield real RL insight — (source: same).
- ***Similar Models Learn Differently* (2607.25063)**, ***Front-Loading Reasoning* (Akter et
  al., ICLR 2026; 2510.03264)**, ***Early Data Exposure…* (Feng et al., 2605.12705)**,
  ***The Finetuner's Fallacy* (Baek et al., 2603.16177)**, **Shen et al. (2607.16097)** — the
  pretraining→post-training line the TINY options descend from — (source: same).
- **FollowIR (2403.15246)** — recorded as a *mismatched* guess at the AI2 "dataset that moves
  metrics" and explicitly flagged as not a finding — (source: same, intake note;
  docs/open-questions-answered.md, open item).
- **Bornschein, Lyle, Pascanu et al., *Fine-Tuned In-Context Learners…*** and ***Eliciting
  Fine-Tuned Transformer Capabilities via Inference-Time Techniques*** — ICL vs. fine-tuning
  as access routes to the same capabilities; feeds the gradient-free proxy candidate —
  (source: same, 2026-08-18; docs/potential-projs/icl-elicitability.md).
- **Lester, Al-Rfou & Constant 2021, *The Power of Scale for Parameter-Efficient Prompt
  Tuning*** — prompt tuning matches full fine-tuning only above ~10B and lags at small sizes;
  the known headwind for tiny-scale elicitation ("from memory; verify") — (source:
  docs/potential-projs/elicitation-gain.md §4/§5).

**Folk precedents named in the origin responses (external text, unverified)**

- **TinyStories (Eldan & Li 2023)** — coherent generation at 1–30M params when the
  distribution is narrowed; the capability-per-parameter-under-narrowing precedent for
  TINY-opt-1 — (source: docs/potential-projs/tiny-scale-measurement.md §4, 2026-08-21;
  docs/potential-projs/elicitation-gain.md §4, listed from memory and unverified).
- **The BabyLM line** — same axis, cited as evidence that rigorous science at small scale has
  an audience — (source: docs/potential-projs/tiny-scale-measurement.md §4, 2026-08-21).
- **The phi-1 line (textbook-quality data for small code models)**; **small-model DSL /
  semantic-parsing work**; **"Distilling step-by-step" (Hsieh et al. 2023)** — tiny-specialist
  prior art listed as from-memory and unverified — (source:
  docs/potential-projs/elicitation-gain.md §4).
- **The nanoGPT-speedrun / modded-nanoGPT culture** — the strongest existing evidence for
  TINY-opt-4's "design decisions are tuned for larger models" thesis (different optimizers,
  schedules, architectural details win at GPT-2 scale, with large cumulative gains) —
  (source: docs/potential-projs/tiny-scale-measurement.md §4, 2026-08-21;
  docs/refs/research-trajectory-pre-to-post-training.md).
- **"Small-scale-proxy work (tiny models to predict large-model training instabilities)"** —
  named without citation in the origin response — (source:
  docs/potential-projs/tiny-scale-measurement.md §4, 2026-08-21).

**Parametrization / HP transfer (bears on TINY-opt-4 and any dense substrate)**

- **u-µP: Unit-Scaled Maximal Update Parametrization (Blake et al., arXiv 2407.17465;
  Danielle-supplied PDF)** — µP + Unit Scaling; independent HP search (9 runs vs. 339);
  LR optimum found at width 256 transfers to 4096; out-of-the-box FP8; embedding LR rule
  c_emb = 1/√fan-out. Recorded consequence: DataDecide uses per-size hand-set
  hyperparameters, i.e. is *not* µP-parametrized, so cross-size comparisons carry the
  "was the small model's LR optimal" confound — the core TINY-opt-4 question — (source:
  docs/topics/reference/parametrization-and-hp-transfer.md).
- **µP / µTransfer (Yang et al., 2203.03466); Unit Scaling (Blake et al., 2303.11257);
  depth-µP / CompleteP** — listed as "related and absent", Claude-added and unverified —
  (source: same).

**Ladder-as-instrument and identifiability**

- **Platonic Representation Hypothesis (Huh et al. 2024)** — read as the empirical claim that
  identifiability improves with scale; if true, small-scale path-dependence washes out and
  TINY's effects have a scale ceiling. Makes the ladder a measurement rather than only an
  external-validity check — (source: docs/topics/reference/identifiability-literature.md;
  docs/potential-projs/tiny-scale-measurement.md §4, 2026-08-18;
  docs/refs/research-trajectory-pre-to-post-training.md).
- **Roeder, Metz & Kingma 2021, *On Linear Identifiability of Learned Representations*;
  model stitching (Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021); CKA (Kornblith et
  al.); Git Re-Basin; Entezari et al.; Juneja et al. (strategy-distinct basins); Frankle et
  al. (commitment events); Fort et al., *Deep Learning vs. Kernel Learning*** — the
  functional-identifiability toolkit the ladder measurement sits inside — (source:
  docs/topics/reference/identifiability-literature.md;
  docs/refs/research-trajectory-pre-to-post-training.md).

**Estimation, intervals, and ranking metrics TINY-3 needs**

- **Wilson / Jeffreys / Clopper–Pearson / Beta-posterior intervals; Hoeffding; empirical
  Bernstein; block and cluster-robust bootstrap; confidence sequences; jackknife** — the
  interval menu, with the explicit note that `IRT`, `TINY`, `ANN-opt-6` need small-n pass-rate
  interval choices and "calibrate after selection" whenever a best recipe/checkpoint is
  chosen from many — (source: docs/topics/reference/estimation-and-calibration-methods.md,
  first entry and relevance map).
- **Split conformal (Lei et al., 1604.04173) and conformal risk control (Angelopoulos et al.,
  2208.02814)** — Danielle's flagged cross-project tool; calibrated intervals on cheap→
  expensive forecasts — (source: same).
- **Unbiased pass@k estimator (Codex paper, 2107.03374)** — for any pass@k readout in the RL
  options — (source: same).
- **Weighted Kendall τ (Vigna 2015, 1404.3325); rank-biased overlap (Webber, Moffat & Zobel
  2010, TOIS); Blest / Wroclaw ν family; NDCG@K; top-k hit; regret** — the right metric
  families when pairwise decision accuracy saturates (adjacent-pair accuracy named as the
  honest stratified version); both weighted τ and RBO are Claude-added and unverified —
  (source: docs/topics/reference/estimation-and-calibration-methods.md, last entry).

**IRT machinery TINY-2 inherits**

- **Lalor et al., *Building an Evaluation Scale Using Item Response Theory*; Rodriguez et al.
  (ACL 2021); Polo et al., *tinyBenchmarks* (ICML 2024, IRT-selected ~100-item subsets
  preserving rankings); *metabench*** — the benchmark-compression IRT line TINY-2's derived
  eval inherits via `irt-reanalysis.md`; attributions unverified — (source:
  docs/topics/reference/irt-literature.md; docs/refs/research-trajectory-pre-to-post-training.md).
- **The IRT local-independence caution and the binary-vs-margin caution** — shared-passage
  items and contamination violate local independence; binary IRT discards the margin
  information carrying most small-scale signal — (source: docs/topics/reference/irt-literature.md).
- **Staged difficulty recipe (smoothed pass rate → Rasch → many-facet logistic mixed model →
  2PL) with leave-one-facet-out validation** — the method order for a first fit, facets being
  recipe/size/seed/step — (source: docs/topics/reference/estimation-and-calibration-methods.md,
  third entry; docs/potential-projs/irt-reanalysis.md §4).

**Substrate thread (not literature, but the record TINY's options rest on)**

- **DataDecide-dense (+WSD)** — a few recipes × the 2–4 smallest scales × 10+ seeds with dense
  checkpointing and full logging (training loss, executed LR schedule, data-order manifest,
  per-token held-out losses on a frozen probe set); first result is the reproduction-gap
  measurement; hypothesis that annealed readouts improve measurement SNR most at small scales
  where wall oscillation is proportionally largest. Not started; design doc gated — (source:
  docs/topics/staging/datadecide-dense.md; docs/topics/reference/datadecide-data-pipeline.md).
- **Regularization inputs for many-epoch tiny runs: Xue et al. 2305.13230; Muennighoff et al.
  2305.16264** — cited for the frozen regularization spec at the smallest scales — (source:
  docs/topics/staging/datadecide-dense.md;
  docs/topics/reference/regularization-literature.md).
- **Hägele et al. WSD/branch cost guidance** — cited for branch cost ("~10% per branch
  point"), unverified — (source: docs/topics/reference/datadecide-data-pipeline.md).
- **Coverage facts the design rests on** — per-instance tables at 4M/20M/60M/90M with one seed
  each and 150M–1B with three; aggregate tables at all 14 sizes with three seeds; 4M–8M runs
  have 5–10 checkpoints; the 750M aggregate table is truncated — (source:
  docs/open-questions-answered.md).

**In-program neighbors that supply or consume TINY's instruments**

- **`elicitation-gain.md` (ELI-1)** — the within-reach existence test under an oracle
  interface at every DataDecide size; the coordination note says reuse its result before
  designing TINY-opt-1 — (source: docs/potential-projs/tiny-scale-measurement.md §1;
  docs/potential-projs/elicitation-gain.md).
- **`irt-reanalysis.md`** — the fit TINY-2's derived eval is a byproduct of; the
  decision-reliability frontier is cross-listed as sub B of P1 — (source:
  docs/potential-projs/irt-reanalysis.md §4; docs/portfolio-rankings.md).
- **`movement-microscope.md`** — the asymmetric design, proxy-metric-as-contribution, and
  power-analysis-for-post-training texts absorbed into TINY's options — (source:
  docs/potential-projs/tiny-scale-measurement.md §4, 2026-08-22).
- **`icl-elicitability.md`** — in-context learning curves (loss on the k-th demo vs. k) and
  round-trip reconstruction through a natural-language bottleneck as graded capability probes
  — (source: docs/potential-projs/tiny-scale-measurement.md §4, 2026-08-18).
- **`trajectory-statistics.md` / drift-diffusion** — the SNR table for which metrics carry
  signal at tiny scale, and the noise-aware crossing definition — (source:
  docs/potential-projs/tiny-scale-measurement.md §4;
  docs/potential-projs/irt-reanalysis.md §4).
- **`datadecide-data-card.md` (DCARD)** — the cleaned tables and metric-definition pinning
  every TINY-opt-5 metric claim depends on — (source: docs/potential-projs/datadecide-data-card.md).
- **Portfolio placements** — decision-reliability frontier workshop-sized #5 and P1 sub B;
  realized-exposure audit workshop-sized #7 and P2 sub B; the whole program full-conference
  #6, "Measuring Learning Where Benchmarks Can't See" (expected impact medium, ceiling
  medium-high) — (source: docs/portfolio-rankings.md;
  docs/potential-projs/tiny-scale-measurement.md §4).

**Provenance caveats carried from the records**

- The two SciSpace reviews seeding the metric accumulators are agent-generated: version 2
  fabricated the seed paper's author list, and its bibliography has swapped or fabricated
  entries (`gao2021framework`→The Pile; `luo2025scaling`→WizardCoder; `xie2023finpythia`→
  PIXIU; `chang2024effective`→2402.04177; `bhagia2024scaling`→2410.08527; Ruan et al. listed
  twice). Prefer version 1's bibliography — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, intake notes).
- The loss-alternative review cites one non-credible source ("The Shannon Paradox … 0.36 bits
  per character", Zenodo) twice as state of the art; ignore — (source:
  docs/topics/reference/loss-alternative-metrics-literature.md).
- Every arXiv ID above traces to `docs/litreview/citation-verification-ledger.md`, where rows
  are marked *agent-supplied* or *Claude-added* and **nothing is verified**.

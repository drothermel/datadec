# early dynamics prediction — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`early-dynamics-prediction.md`](../early-dynamics-prediction.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*Recall corpus for EDP (early-dynamics prediction). Highest-recall enumeration of every
paper, method, or named prior-art item on record anywhere in this repository that could
plausibly bear on EDP. One line per item; inclusion is cheap, so marginal items are kept.
Every item carries its repo source. **Nothing here is verified** — most entered through
agent-generated records (SciSpace reviews, Perplexity searches, memory-sourced intake
notes); those are marked in-line. No positioning or novelty claims are made here.*

**Learning-curve extrapolation — the named direct-ancestor line**

- **Domhan, Springenberg & Hutter 2015, "Speeding up automatic hyperparameter optimization
  of DNNs by extrapolation of learning curves" (IJCAI)** (no ID on record) — ensemble of
  parametric curve families fit to a partial curve to predict the asymptote; CIFAR-10 CNNs
  among the testbeds; the record's "direct ancestor of fit-something-to-early-epochs";
  reported RMSE 0.25/0.19/0.11 after 10/40/60 epochs (agent-reported, unverified) —
  (source: docs/topics/reference/loss-curve-forecasting.md; docs/topics/reference/nas-literature.md).
- **Klein et al. 2017, LC-Net (Bayesian NN over learning curves)** (no ID) — regress final
  accuracy on early partial curve + config features; named in the EDP related-work gate —
  (source: docs/topics/reference/loss-curve-forecasting.md; docs/potential-projs/early-dynamics-prediction.md §4 gate).
- **Baker et al. 2017, "Accelerating NAS using performance prediction"** (no ID) — regressor
  over early partial-curve features plus configuration/architecture features; the record
  calls it "the closest methodological match to the loss-slope study and EDP";
  memory-sourced, flagged "verify" — (source: docs/topics/reference/loss-curve-forecasting.md;
  docs/topics/reference/nas-literature.md).
- **LC-PFN — Adriaensen et al., NeurIPS 2023** (arXiv 2310.20447) — transformer pretrained on
  millions of synthetic curves, Bayesian extrapolation in one forward pass, "10,000× faster
  than MCMC"; the record names it "the modern extrapolation baseline any early-window
  predictor should be compared against" — (source: docs/topics/reference/loss-curve-forecasting.md).
- **Ding et al. 2024, architecture-aware neural-ODE learning-curve prediction** (arXiv
  2412.15554) — conditions curve prediction on the configuration (MLPs, CNNs); the record
  reads EDP's recipe-conditioning as the same move on a different axis —
  (source: docs/topics/reference/loss-curve-forecasting.md).
- **Neural capacitance — Jiang et al.** (arXiv 2201.04194; Nat. Commun. 2024) — line-graph
  "capacitance" metric from early-training synaptic dynamics, validated on CIFAR-10 model
  selection; the record's "non-curve early-dynamics predictor, closest in spirit to
  mathematical properties of early training" — (source: docs/topics/reference/loss-curve-forecasting.md;
  docs/potential-projs/early-dynamics-prediction.md §4 gate).
- **An OpenReview paper on selective prediction from intermediate-checkpoint instability**
  (no ID) — prediction instability across training used to decide what to abstain on;
  logged as tangential (instance-level, not run-level) —
  (source: docs/topics/reference/loss-curve-forecasting.md).

**Zero-cost / training-free NAS proxies (the init-time end of the same spectrum)**

- **TE-NAS (Chen et al., ICLR 2021)** (no ID) — ranks architectures by NTK condition number
  and number of linear regions at init; memory-sourced and unverified —
  (source: docs/topics/reference/nas-literature.md).
- **NASWOT (Mellor et al., ICML 2021)** (no ID) — activation-overlap score at init;
  memory-sourced — (source: docs/topics/reference/nas-literature.md).
- **Abdelfattah et al. 2021, "Zero-Cost Proxies for Lightweight NAS" (ICLR)** (no ID) —
  synflow / snip / grasp / jacob_cov / fisher compared; found weak-to-moderate rank
  predictors — (source: docs/topics/reference/nas-literature.md).
- **AZ-NAS (CVPR 2024)** (arXiv 2403.19232) — assembling proxies raises rank correlation —
  (source: docs/topics/reference/loss-curve-forecasting.md; nas-literature.md).
- **FreeREA (WACV 2023)** (no ID) — training-free evolutionary NAS —
  (source: docs/topics/reference/loss-curve-forecasting.md).
- **ICLR blog-track 2022 zero-cost-proxy survey** (no ID) — scores at init or after minimal
  training — (source: docs/topics/reference/loss-curve-forecasting.md).
- **Rank-collapse caution (design constraint, not a paper)** — zero-cost proxies' rank
  correlations collapse *within the top of the search space*, so EDP's ranking/decision
  metrics should be evaluated on the top-k subset as well as the full population —
  (source: docs/topics/reference/nas-literature.md intake note; docs/potential-projs/early-dynamics-prediction.md §4 gate).
- **NAS field-survey items carried in the same accumulator** (all blog/survey-sourced,
  low relevance but on record): Elsken et al. NAS survey; predictor-based NAS with
  attention-enhanced path encodings (2025 Sci. Reports); EENAS (PMLR v189); Auto-Meta
  (arXiv 1806.06927); NAHAS; LLMatic; CE-NAS (NeurIPS 2024); causal zero-shot NAS
  (OpenReview 3s6aE1LeiR); Ci et al. (ICCV 2021, larger search spaces hurt); NAS-Bench-360;
  NATS-Bench; AutoML 2025 "NAS Unseen Data" competition —
  (source: docs/topics/reference/nas-literature.md).

**LM-scale training-time forecasting (the newest and closest line on record)**

- **Patel, Reddy, Mosbach & Bahdanau 2026, "Forecasting Downstream Performance of LLMs With
  Proxy Metrics"** (arXiv 2605.18607, Danielle-supplied ID; Mila/McGill + ServiceNow) — 80
  token-level proxy metrics (10 core statistics × 8 weighting schemes) from one forward pass
  over expert-written solution trajectories; RankSVM leave-2-tasks-out ρ = 0.81 vs. 0.36 for
  FineWeb cross-entropy and 0.33 for rBridge; **DataDecide corpus ranking at ~10⁻⁵ of target
  compute with decision accuracy > 0.85**; **training-time forecasting of downstream accuracy
  along the OLMo-3-7B trajectory over an 18× compute horizon at ~half the RMSE of loss
  baselines** (one architecture/scale). Sits directly on EDP's question —
  (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md;
  docs/potential-projs/early-dynamics-prediction.md §4 2026-08-22 entry). *SciSpace-agent
  record; version 2 of that review fabricated the author list.*
- **NeuNeu, "Neural Neural Scaling Laws"** (arXiv 2601.19831) — scaling-law prediction as
  time-series extrapolation over observed accuracy trajectories plus token-level validation
  losses; 2.04% MAE on 66 tasks vs. 3.29% for logistic fits; zero-shot to unseen families.
  The §4 entry places it "in the related-work gate beside the LC-PFN / Domhan / Klein line"
  — (source: docs/potential-projs/early-dynamics-prediction.md §4;
  docs/topics/reference/small-scale-evaluation-metrics-literature.md). *Agent-sourced.*
- **rBridge (expert-reweighted NLL; "Koh & Liang 2026", no ID)** — the comparison baseline
  Patel et al. beat; the ledger records it as unverifiable —
  (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md intake note).
- **Ye et al., BIG-bench predictability** (arXiv 2305.14947) — MLP predictor, >95% R²;
  "small-bench" 3× smaller than BBH equally informative — (source: same file).
- **Schellaert et al.** (arXiv 2305.12415) — DeBERTa "assessors" predicting per-instance
  success — (source: same file).
- **ProxyLM** (arXiv 2406.09334) — small proxy models for multilingual performance, 37×
  speedup — (source: same file).
- **Lineage-regularized matrix factorization** (arXiv 2504.19811) — model ancestry as a prior
  for performance prediction — (source: same file).
- **FamiCom** (arXiv 2406.11243) — familiarity × complexity, ρ 0.848 with end-task
  performance — (source: same file).

**Loss-curve shape, scaling laws, and loss→accuracy links (what EDP predicts against)**

- **Multi-power law for loss-curve prediction — Luo et al.** (arXiv 2503.12811, ICLR 2025) —
  predicts the full loss curve at every step across LR schedules from a power law on the sum
  of LRs plus decay terms; extrapolates to unseen schedules; discovers a WSD-like schedule —
  (source: docs/topics/reference/loss-curve-forecasting.md; landscape-literature.md).
- **FLP two-stage scaling law — Yangyi Chen et al.** (arXiv 2410.08527) — FLOPs → pretraining
  loss → downstream performance; 5–10% error at 7B/13B —
  (source: docs/topics/reference/loss-curve-forecasting.md; small-scale-evaluation-metrics-literature.md).
- **Gadre et al. 2024, "Language models scale reliably with over-training and on downstream
  tasks"** (arXiv 2403.08540) — downstream accuracy as an exponential of training loss; 104
  models 11M–6.9B; holds on average, varies by task —
  (source: docs/topics/reference/loss-curve-forecasting.md; small-scale-evaluation-metrics-literature.md).
- **Bhagia et al., compute-efficient model ladders / task scaling laws** (arXiv 2412.04403) —
  compute → task NLL → accuracy; 1% of target compute, within 2 points on some tasks; N and D
  beat FLOPs in overtrained regimes — (source: same two files).
- **Observational scaling laws** (arXiv 2405.10938) — ~80 public models, low-dimensional
  capability space, emergence as sigmoids — (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **Krajewski et al.** (arXiv 2512.08894) — direct power law for log-accuracy at fixed
  tokens-per-parameter beats the two-stage loss→accuracy route — (source: same file).
- **Kaplan-style power law / Chinchilla** (no IDs on record) — the scaling-law extrapolation
  baseline the reviews require EDP to run (Kaplan-style fit on ≤60M models, extrapolated,
  scored with the same ranking metrics) — (source: docs/potential-projs/early-dynamics-prediction.md
  §4 first review; small-scale-evaluation-metrics-literature.md).
- **Li et al. 2025, *(mis)fitting* scaling laws** (no ID) — cited in the July-2025 draft's
  motivation that scaling-law extrapolation is "not always accurate" —
  (source: docs/potential-projs/early-dynamics-prediction.md §4 2025-07 proposal).
- **Lourie, Hu & Cho 2025, *Scaling laws are unreliable for downstream tasks*** (no ID) —
  the second motivating citation of the July-2025 draft — (source: same).
- **Pechi et al.** (arXiv 2305.17266) — small-scale break below ~2.2e15 FLOPs, a bound on how
  small the training rungs can be — (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **Ivgi et al. 2022** (no ID) — finetuning scaling laws need R² ≥ 0.95 — (source: same file).
- **Data-constrained scaling** (arXiv 2305.16264); **quality-aware Q** (arXiv 2510.03313);
  **effective tokens = diversity × syntheticity** (arXiv 2410.03083, r = 0.83 over 200 models
  25M–1.5B); **context-aware scaling** (arXiv 2510.14919); **hyperparameter scaling** (arXiv
  2505.13738) — all candidate covariate/target framings for EDP's static features —
  (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **Loss-to-loss scaling determined by data and tokenizer, not architecture** (arXiv
  2502.12120) — (source: same file).
- **Brandfonbrener et al. 2024, loss-to-loss prediction** (no ID) — named in the July-2025
  draft's own "Related work named" list — (source: docs/potential-projs/early-dynamics-prediction.md §4).
- **Knowledge capacity 2 bits/parameter** (arXiv 2404.05405); **repeated-data double descent**
  (arXiv 2205.10487) — (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md).

**Boundary conditions on any loss → capability prediction law**

- **Nakkiran et al., *Deep Double Descent*** (no ID) — epoch-wise double descent means
  capability is not monotone in training loss along a single run; the record calls this "a
  boundary condition on the whole prediction-law thread" —
  (source: docs/topics/reference/grokking-and-hidden-progress.md; loss-curve-forecasting.md).
- **Power et al., *Grokking*** (no ID) — train loss at floor and test loss flat while the
  generalizing circuit assembles invisibly; the maximal demonstration that final/interim loss
  is an insufficient statistic — (source: docs/topics/reference/grokking-and-hidden-progress.md).
- **Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability*** (no ID)
  — non-loss observables needed to see hidden progress — (source: same file).
- **Wei et al., emergent abilities** (arXiv 2206.07682) vs. **Schaeffer et al., "mirage"**
  (arXiv 2304.15004) — the jumpy-discrete-metric argument behind EDP's choice of CORRECT PROB
  over accuracy as the downstream target — (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md;
  docs/potential-projs/early-dynamics-prediction.md §4 2025-07 proposal, which cites
  Schaeffer et al. directly).
- **Proxy tasks for emergent abilities** (arXiv 2412.07111) — (source: small-scale-evaluation-metrics-literature.md).
- **Additional double-descent / landscape citations surfaced by the linearity searches**
  (arXiv 2407.09845, 2203.07337, IEEE 10222624; loss-landscape convergence vs. sample size
  arXiv 2409.11995) — (source: docs/topics/reference/loss-curve-forecasting.md, answer 2).

**The linearity premise, its provenance, and the CNN-scale pilot**

- **Advisor-supplied hypothesis: "more linear loss curves indicate better training"** (no
  paper) — **neither of two literature searches found any support**; the record's honest
  summary is that no established belief about linearity exists and the documented shape
  priors are power-law / multi-power-law decay, with smoothness as a separate stability
  heuristic; EDP's linearity/R² features inherit an untested premise —
  (source: docs/topics/reference/loss-curve-forecasting.md, two undated ~2025 answers;
  docs/past-projects/loss-slope-prediction.md).
- **Perplexity answer 1 (blog/StackExchange-sourced)** — argues smoothness, not linearity;
  one paper-shaped citation, an ECCV 2022 paper on enforcing smoothness in *learned
  optimizers* (ecva.net 136830533, unverified) — (source: loss-curve-forecasting.md).
- **Perplexity answer 2 ("academic" mode)** — the record marks it as *manufacturing* support:
  a fast-adversarial-training paper (IEEE 10376811, "ConvergeSmooth"); loss-*function*
  smoothness papers (arXiv 2208.04075; IEEE 10255658) conflated with loss-*curve* smoothness;
  an L2-regularized CNN-LSTM (IEEE 10872755); applied hate-speech / air-quality / geophysics
  papers; SGD "noise equilibria"; AdaGC (arXiv 2502.11034, removes loss spikes) — (source: same).
- **arXiv 2410.11451** — larger LMs stabilize early (within the first 20% of epochs) while
  smaller ones converge slower and less stably; flagged as one of only two citations from
  those searches worth following for EDP ("how early is early enough") — (source: same).
- **AdaGC** (arXiv 2502.11034) — loss-spike handling, flagged as relevant to *cleaning*
  early-window features — (source: same).
- **Loss-slope prediction study (Danielle, 2025-06; CIFAR-10, ~34 runs × 25 epochs)** (no ID,
  internal) — EDP's CNN-scale lineage: 4-epoch validation-loss slope is the best single
  predictor of final accuracy (|r| = 0.71), decaying to 0.36 for the full-curve slope; higher
  R² (linearity) ↔ *lower* accuracy (r ≈ −0.4, "linearity indexes slowness"); direction
  resolved from the slope-bin table (fastest early decline → highest accuracy, monotone);
  confounded with a mid-ladder LR change and treats seeds as independent —
  (source: docs/past-projects/loss-slope-prediction.md; docs/potential-projs/early-dynamics-prediction.md §4).
- **CNN deconstruction ladder (`deconCNN`, 2025)** (no ID, internal) — the substrate the
  loss-slope study ran on; if revived, the early-features → final-accuracy question can be
  asked per rung, making it a CNN-scale replication setting for EDP —
  (source: docs/past-projects/cnn-deconstruction-ladder.md).
- **Recipe-ablation papers the CNN lineage should be positioned against**: **"Bag of Tricks"**
  (arXiv 1812.01187); **"Revisiting ResNets"** (arXiv 2103.07579); **"ResNet strikes back"**
  (arXiv 2110.00476) — named in EDP's own related-work gate and in the ladder record —
  (source: docs/potential-projs/early-dynamics-prediction.md §4 gate;
  docs/past-projects/cnn-deconstruction-ladder.md). *Memory-sourced, unverified.*
- **Li et al. 2018 loss-surface visualizations** (no ID) — the landscape-metric comparison
  point named for the CNN ladder's with/without-skip and with/without-BN rungs; flagged
  "unverified here, check before citing" — (source: docs/past-projects/cnn-deconstruction-ladder.md).

**The suite, its own baselines, and DataDecide-adjacent consumers**

- **Magnusson et al., *DataDecide: How to Predict Best Pretraining Data with Small
  Experiments*** (Ai2, ICML 2025, arXiv 2504.11393) — the object of study: 25 recipes × 14
  sizes × 3 seeds, per-checkpoint perplexity on 11 validation splits + OLMES tasks; also the
  source of the CORRECT-PROB-over-accuracy choice and of the *published scaling-law
  baselines* EDP's open questions flag as possibly having already answered the "does early
  dynamics beat scaling-law extrapolation" comparison —
  (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/potential-projs/datadecide-data-card.md; docs/potential-projs/early-dynamics-prediction.md §4 open questions).
- **OLMES, *A Standard for Language Model Evaluations*** (no ID) — the downstream task suite
  and evaluation standard EDP's targets come from —
  (source: docs/topics/reference/evaluation-methodology-literature.md).
- **Heineman et al., *Signal and Noise: A Framework for Reducing Uncertainty in LM
  Evaluation*** (NeurIPS 2025 per the record, no ID) — signal/noise decomposition, continuous
  (perplexity-type) metrics beating accuracy on both, ~900K eval results on 465 models
  including DataDecide and the model ladders; directly relevant to EDP's noisy small-fold
  ranking metrics — (source: docs/topics/reference/evaluation-methodology-literature.md).
- **u-µP: Unit-Scaled Maximal Update Parametrization (Blake et al.)** (arXiv 2407.17465,
  Danielle-supplied) — the only ledger row tagged as feeding EDP besides the SciSpace block;
  the accompanying note that **DataDecide is not µP-parametrized**, so cross-size LR is a
  confound on the scale-generalisation axis — (source: docs/litreview/citation-verification-ledger.md;
  docs/topics/reference/parametrization-and-hp-transfer.md).

**Ranking / calibration metric methodology (EDP's evaluation layer)**

- **Weighted Kendall τ — Vigna 2015** (arXiv 1404.3325) — hyperbolic top-weighted rank
  correlation (`scipy.stats.weightedtau`), proposed because pairwise decision accuracy is
  (τ+1)/2 and saturates on 25-recipe rankings; *Claude-added to the ledger, unverified* —
  (source: docs/topics/reference/estimation-and-calibration-methods.md;
  docs/potential-projs/early-dynamics-prediction.md §4 2026-08-22 metric entry).
- **Rank-biased overlap — Webber, Moffat & Zobel 2010 (TOIS)** (no arXiv ID) — top-weighted
  overlap with a stated emphasis parameter; *Claude-added, unverified* — (source: same).
- **NDCG / LambdaMART / `lambdarank`** (no IDs) — LightGBM's pairwise NDCG-gradient objective
  as the ranking head; NDCG@10 with rank-derived gains; MAP/P@K judged unnatural here —
  (source: docs/topics/reference/estimation-and-calibration-methods.md;
  docs/potential-projs/early-dynamics-prediction.md §4 responses 11–15).
- **Top-k hit, regret, and decision accuracy stratified by true-rank gap** (methods, no
  papers) — the rest of the saturation fix — (source: same two files).
- **ECE for regression via hand-rolled equal-count quantile bins** (`netcal` named;
  scikit-learn's `calibration_error` is classification-only) — (source: docs/potential-projs/early-dynamics-prediction.md §4).
- **SHAP TreeExplainer** (no ID) — exact Shapley attributions per fold, mean |SHAP| with
  cross-fold std as the interpretable payload — (source: same).
- **Benjamini–Hochberg FDR ≤ 5%; block-bootstrap CIs over model pairs; binomial CIs for
  MMLU** (methods) — the first review's statistical hygiene — (source: same).
- **Kadavath et al. 2207.05221 (calibration); conformal probes (Ashok & May 2025); conformal
  set size as a benchmark axis (2401.12794); self-evaluation EQT 2501.11721; linear probes
  on activations (Pacchiardi 2025)** — the uncertainty/calibration flank of the same
  SciSpace review — (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **Conformal prediction and conformal risk control** — Danielle's cross-project tool flag,
  potentially applicable to EDP's uncertainty story —
  (source: docs/topics/reference/estimation-and-calibration-methods.md).

**Models and estimators EDP names as its own methods (prior art for the method, not the claim)**

- **LightGBM regressor and `LGBMRanker`/LambdaMART** (no ID) — the two-head design; starter
  params (`num_leaves=512`, lr 0.05, `min_data_in_leaf=20`, …) with the intake note that 512
  leaves cannot grow on ~130–350 rows — (source: docs/potential-projs/early-dynamics-prediction.md §4).
- **CatBoost** (no ID) — named as a Phase-2 baseline for high-cardinality categoricals —
  (source: same).
- **Sparse / multi-output GPs (GPyTorch variational sparse GP, ≤1,000 inducing points;
  GPflow)** (no IDs) — EDP-opt-1's uncertainty model and the active-learning extension —
  (source: same).
- **Theil–Sen regression** (no ID) — robust alternative for the rolling-slope features —
  (source: same).
- **Optuna** (no ID) — the ~100-trial tuning sweep, with the intake note that its sketch
  leaks by using a plain random split — (source: same).

**Early-dynamics signals beyond the loss curve (EDP-opt-2's feature backlog)**

- **Fort et al. 2020, *Deep Learning versus Kernel Learning*** (arXiv 2010.15110) — the eNTK
  moves most in the first few epochs then the network becomes near-linear-in-parameters; the
  ntk file calls this "exactly the kind of early-dynamics signal EDP … care[s] about";
  unverified — (source: docs/topics/reference/ntk-literature.md).
- **Jacot et al. 2018, NTK** (arXiv 1806.07572) — the one identifier-bearing citation in the
  eNTK overview; the kernel-regression limit behind condition-number-as-trainability —
  (source: docs/topics/reference/ntk-literature.md).
- **Candidate eNTK readouts** (methods, no papers): top-k spectrum and effective rank at
  checkpoints; kernel velocity ‖Θ_t − Θ_{t−1}‖_F/‖Θ_t‖_F; kernel–target alignment (CKA of Θ
  with yyᵀ); on a fixed probe set of a few hundred examples —
  (source: docs/topics/reference/ntk-literature.md intake note).
- **Unchecked eNTK IDs carried in the same overview** (arXiv 2104.03093, 2305.14585,
  2406.18800, 2502.02870; Tensor Programs II; `neural_tangents.empirical_ntk_fn`;
  `torch.func` NTK tutorial) — (source: same).
- **Curvature readouts — top Hessian eigenvalue, Hessian-trace approximation** (methods) —
  EDP-opt-2's extended early metrics; the July-2025 draft's hypothesis that lower curvature ↔
  better optimization and finetuning — (source: docs/potential-projs/early-dynamics-prediction.md §1, §4).
- **Fisher trace / Fisher-information diagnostics** (Achille–Soatto's Information Plasticity;
  Task2Vec as the same formalism pointed at data) — named in EDP's feature backlog as the
  replacement for the crude "jitter" noise-scale proxy, and the bridge to the critical-period
  line — (source: docs/topics/reference/critical-periods.md; docs/potential-projs/early-dynamics-prediction.md §4 responses 16–18).
- **Plasticity statistics as candidate early features — Lyle et al., *Understanding
  Plasticity in Neural Networks*** (ICML 2023, arXiv 2303.01486) and ***Disentangling the
  Causes of Plasticity Loss*** (arXiv 2402.18762) — curvature "comes closest" as the cheap
  training statistic that forecasts future trainability; feature rank, dead units, weight
  norm as the rest of the panel — (source: docs/topics/reference/plasticity.md).
- **Dohare et al., *Loss of plasticity in deep continual learning*** (Nature 2024; arXiv
  2306.13812) — the panel's other source — (source: same).
- **Gradient/activation variance, feature stability** (methods) — the July-2025 draft's
  "extended early metrics" future-work list — (source: docs/potential-projs/early-dynamics-prediction.md §4).

**Recipe-side features and data-mixture prediction (EDP's static-feature half)**

- **Recipe featurization (`REC`, internal project)** — the record's instruction that EDP's
  static recipe features should be `REC`'s *measured* properties, not the hand-assigned
  percentages (`pct_code`, `pct_common_crawl`, `pct_social_media`, `duplicate_rate_pct`,
  `quality_filter_strength`, `educational_content_score`) currently in the 131/67-column
  schema — (source: docs/potential-projs/early-dynamics-prediction.md §4 intake notes).
- **RegMix; DoReMi; data mixing laws (arXiv 2403.16952); AutoScale (2407.20177); UtiliMax /
  MEDU (2501.11747); D-CPT law (2406.01375); ADO (2410.11820 — small proxy models often fail
  to predict larger ones); BiMix (2405.14908); optimal-mixture laws (2507.09404); PDPC
  (2501.13126)** — the data-mixture prediction flank; ADO's negative result is the sharpest
  caution for EDP's cross-scale claim — (source: docs/topics/reference/data-featurization-literature.md;
  small-scale-evaluation-metrics-literature.md; schedules-and-annealing-literature.md).
- **Task2Vec (Achille et al.)** (no ID) — Fisher-embedding dataset representation behind
  alignment coefficients; a candidate recipe feature —
  (source: docs/topics/reference/critical-periods.md; data-featurization-literature.md).
- **WIMBD, compression, Zipf/burstiness intrinsic corpus features** (no IDs) — the intrinsic
  family of recipe features — (source: docs/topics/reference/data-featurization-literature.md).
- **Determinism profile as a recipe feature** — Wen et al.'s river-valley mechanism makes a
  corpus's token-determinism profile a candidate predictor of *landscape geometry*, hence of
  annealing behaviour — (source: docs/topics/reference/landscape-literature.md 2026-08-18 entry).
- **Unverified attributions from the second review's TODO list**: code-token mix ratio → math
  (Gadre et al. 2024); warm-up length → calibration (Mao et al. 2024); curriculum ordering →
  convergence (Zhang et al. 2025) — flagged in-repo as unverified —
  (source: docs/potential-projs/early-dynamics-prediction.md §4 second review).
- **Large-scale curriculum-learning study (Zhang et al. 2025) and influence-driven curricula
  (arXiv 2508.15475)** — order effects that would confound early-window features —
  (source: docs/topics/reference/schedules-and-annealing-literature.md).

**Annealing / schedule confounds on the target (EDP-4)**

- **Wen et al., *Understanding WSD Learning Rates: A River Valley Loss Landscape View***
  (arXiv 2410.05192) — the decay-phase drop is descent from the walls to the river; late
  rankings are partly a cosine-tail artefact, which bounds what "predicting the final
  ranking" means — (source: docs/topics/reference/landscape-literature.md;
  docs/potential-projs/early-dynamics-prediction.md §4 relation-to-docs).
- **Hägele et al. 2024, *Scaling Laws and Compute-Optimal Training Beyond Fixed Training
  Durations*** (arXiv 2405.18392) — stable-phase + decay-branch methodology, the annealed
  readout EDP-4 would use as its alternative target —
  (source: docs/topics/reference/schedules-and-annealing-literature.md).
- **MiniCPM** (arXiv 2404.06395) — gradient dynamics across the decay phase — (source: same).
- **Tissue et al.** (arXiv 2408.11029) and **arXiv 2508.01483** — LR-annealing scaling laws
  adding an "annealing area" term — (source: same).
- **TREC, Training Re-evaluation Curves — Bergsma et al.** (arXiv 2509.25380) — receptivity
  valley for anneal placement; a stage-dependent readout adjacent to EDP's early window —
  (source: same).
- **Blakeney et al., *Does your data spark joy?*** (no ID) — domain upsampling at the end of
  training — (source: same).
- **Annealed readouts (`ANN`) and trajectory statistics (`TRJ`), internal** — the caveat that
  late-training rankings are partly a cosine-tail artefact; EDP-4 repeats EDP-1 with the
  annealed readout as target — (source: docs/potential-projs/early-dynamics-prediction.md §1, §4).

**Cross-project analogues inside the repo (the same question at other axes)**

- **Tiny-scale measurement (`TINY`)** — how far *down the scale ladder* decision signal
  survives; EDP is the early-*time* analogue, flagged for cross-listing —
  (source: docs/potential-projs/early-dynamics-prediction.md §4 relation-to-docs).
- **IRT reanalysis (`IRT`)** — shares the leave-recipe-family-out fold scheme and the
  decision-reliability framing — (source: same; docs/topics/reference/irt-literature.md).
- **Generalization-and-OOD question sequence (2025-01-04)** — Danielle's Q3 "predicting the
  performance conditioned on the method" is recorded as the early trace of EDP's question;
  the response never answered that half (it listed DG survey 2103.03097, CVPR 2024 evaluation
  protocols, NeurIPS 2021 calibration↔OOD, NeurIPS 2021 OOD theory, arXiv 2405.19703) —
  (source: docs/topics/reference/generalization-and-ood-literature.md). *All unverified,
  ChatGPT-sourced.*
- **Shen et al., *Understanding Reasoning from Pretraining to Post-Training*** (arXiv
  2607.16097) — lower pretraining loss strongly predicts higher post-RL pass@1 at fixed RL
  compute; the closest published statement on EDP-opt-3's finetune-outcome prediction —
  (source: docs/topics/reference/pretraining-to-posttraining.md).

**Peripheral items on record with a plausible EDP hook**

- **Ali et al.** (arXiv 2310.08754) — tokenizer metrics uncorrelated with downstream —
  (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **Dudy et al. 2020** (no ID) — soft-match accuracy — (source: same).
- **SLM-Bench (2508.15478); SLM survey (2409.15790); SLMs on code (2507.03160); ReTraceQA
  (2510.09351); generative→NLU reformulation (2506.03592); Informedness over accuracy/F1
  (2401.03831); reference-based metrics failing for modern models (2310.13800)** — the
  small-model benchmark flank of the same review — (source: same).
- **Contamination: time-travel detection (2308.08493); C2LEVA (2412.04947); contamination
  survey (2503.17793)** — (source: same file; schedules-and-annealing-literature.md).
- **Loss-replacement metrics (LongPPL, bits per byte, tokenization-marginal likelihood,
  Diff-eRank / nuclear norm)** — alternative early-window signals if perplexity is a poor
  feature — (source: docs/topics/reference/loss-alternative-metrics-literature.md).
- **Rho-1 loss-trajectory taxonomy** (no ID) — per-token loss-trajectory classes; the
  token-level version of EDP's curve-shape features —
  (source: docs/topics/reference/token-level-literature.md; nonstationarity-accounting.md).
- **Melis, Dyer & Blunsom 2018** (no ID) — conclusions inverting under equalized tuning
  budgets; the reason EDP's baselines must be tuned as seriously as its GBDT —
  (source: docs/topics/reference/evaluation-methodology-literature.md).
- **Hochlehnert et al., *A Sober Look…*** (arXiv 2504.07086) — small benchmarks produce
  unstable estimates, multiple seeds essential; the seed-handling discipline EDP's per-size
  folds need — (source: docs/topics/reference/pretraining-to-posttraining.md).
- **PolyPythias** (van der Wal et al., ICLR 2025, arXiv 2503.09543; 50 runs, 14M–410M, ~7k
  checkpoints) — a many-seed substrate with dense checkpoints if EDP ever needs more seed
  coverage than DataDecide gives — (source: docs/topics/reference/reinit-and-transfer-literature.md).
- **Smooth Scaling Laws Hide Stepwise Token Learning** (arXiv 2606.29858) — aggregate curves
  hiding per-token step changes; a caution on curve-shape features —
  (source: docs/topics/reference/reinit-and-transfer-literature.md).
- **Critical periods in LM finetuning** (TACL, doi:10.1162/tacl_a_00725) — stage-dependence of
  intervention effects, the mechanism-side reason early windows might carry signal —
  (source: docs/topics/reference/reinit-and-transfer-literature.md; staging/checkpoint-tomography.md).

**Provenance caveats for this corpus**

- The `small-scale-evaluation-metrics-literature.md` block entered through a **SciSpace deep
  review** whose version 2 fabricated the seed paper's author list and carries swapped or
  fabricated bibliography entries (`gao2021framework`, `luo2025scaling`, `xie2023finpythia`,
  `chang2024effective`, `bhagia2024scaling` all point at the wrong paper); the v1
  bibliography is the cleaner citation source. Every ID in that block is ledgered as
  agent-supplied and unverified — (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md
  intake notes; docs/litreview/citation-verification-ledger.md).
- The `nas-literature.md` proxies and the Baker/Klein rows are **memory-sourced** by the
  intake agent, not retrieved — (source: those files' own headers/intake notes).
- The linearity searches are **Perplexity-sourced**, and the record explicitly judges answer 2
  to have manufactured support — (source: docs/topics/reference/loss-curve-forecasting.md).
- The **related-work gate** in EDP §4 (added 2026-08-22) states the standing requirement:
  verify and position against the learning-curve-extrapolation, early-dynamics-proxy, and
  zero-cost-NAS lines before any write-up — (source: docs/potential-projs/early-dynamics-prediction.md §4).

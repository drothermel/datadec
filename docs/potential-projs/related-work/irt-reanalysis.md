# irt reanalysis — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`irt-reanalysis.md`](../irt-reanalysis.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*Recall corpus for `irt-reanalysis.md` (IRT). Highest-recall inventory of every paper,
method, or named prior-art item on record anywhere in this repository that is possibly
relevant to this project. Errs toward inclusion; one line per item with its repo source.
Nothing here is verified — SciSpace reviews, novelty checks, and NBLM-style tables are
agent-generated records and the ledger marks every arXiv ID unverified. No positioning
claims; inventory and attribution only.*

**The IRT-for-NLP-evaluation line (the direct prior art)**

- **Lalor et al., *Building an Evaluation Scale Using Item Response Theory*** (no ID on
  record) — the seed of the "fit IRT to diverse converged models" line — (source:
  docs/topics/reference/irt-literature.md;
  docs/refs/research-trajectory-pre-to-post-training.md).
- **Rodriguez et al., *Evaluation Examples Are Not Equally Informative: How Should That
  Change NLP Leaderboards?*** (ACL 2021; no ID on record) — named in §4 as one of the two
  accepted main-venue IRT precedents — (source: docs/topics/reference/irt-literature.md;
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-21).
- **Polo et al., *tinyBenchmarks: Evaluating LLMs with Fewer Examples*** (ICML 2024; no ID on
  record) — IRT-selected ~100-item subsets that preserve full-benchmark rankings; the second
  named main-venue precedent — (source: same).
- ***metabench — A Sparse Benchmark of Reasoning and Knowledge in Large Language Models***
  (no ID on record) — the fourth member of the compression line — (source:
  docs/topics/reference/irt-literature.md).
- **Recorded characterization of all four**: they fit IRT to *diverse converged models* to
  compress benchmarks; the pattern read in the accepted ones is "IRT plus a claim or payoff,
  never IRT as reanalysis." Attributions unverified — (source: same;
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-21).
- **Cautions recorded with the prior art**: local independence is violated by shared-passage
  items and by contamination (fit diagnostics flag both — IRT-6); binary IRT discards the
  margin information carrying most small-scale signal, so fit both response models —
  (source: docs/topics/reference/irt-literature.md).

**Emergence-as-measurement (Claim 1 of the full-conference version)**

- **Schaeffer et al., *Are Emergent Abilities a Mirage?* (2304.15004)** — metric choice
  manufactures discontinuities; the §4 entry's characterization is that it swapped metrics ad
  hoc without a principled framework, which is the gap IRT's link function fills — (source:
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-21;
  docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **Wei et al. (2206.07682)** — the emergent-abilities claim being decomposed — (source: same).
- **Proxy tasks for emergent abilities (2412.07111)** — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **Gadre et al. 2403.08540 / Gadre et al. 2024** — downstream accuracy as an exponential
  function of training loss; the loss→accuracy link IRT-4 would replace with a per-item
  distribution — (source: docs/topics/reference/loss-curve-forecasting.md;
  docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **Chen et al., FLP two-stage loss→performance (2410.08527)** — FLOPs → pretraining loss →
  downstream performance — (source: docs/topics/reference/loss-curve-forecasting.md).
- **Bhagia et al., model ladders (2412.04403)** — compute → task NLL → accuracy — (source: same).
- **The recorded caveat**: hard accuracy metrics look emergent, showing no progress above
  chance until the loss crosses a threshold, which is where the loss-to-accuracy mapping gets
  fragile — the fragility IRT would formalize — (source:
  docs/topics/reference/loss-curve-forecasting.md;
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-18).
- **Double descent (Nakkiran et al.; 2407.09845; 2203.07337)** — capability is not monotone in
  training loss along a single run: a boundary condition on smooth-θ-growth stories — (source:
  docs/topics/reference/loss-curve-forecasting.md; grokking-and-hidden-progress.md pointer).
- **Multi-power law for loss curves (Luo et al., 2503.12811, ICLR 2025); NeuNeu (2601.19831);
  observational scaling laws (2405.10938, emergence as sigmoids)** — the forecasting flank
  around the same link — (source: docs/topics/reference/loss-curve-forecasting.md;
  docs/topics/reference/small-scale-evaluation-metrics-literature.md).

**The metric-choice precedent (IRT-5's target result)**

- **Heineman et al., *Signal and Noise: A Framework for Reducing Uncertainty in Language
  Model Evaluation*** (NeurIPS 2025 per the record; ledger 2508.13144, Claude-added) —
  signal/noise decomposition, continuous metrics beating accuracy, noisy-subtask filtering;
  ~900K results on 465 models including DataDecide and OLMo intermediates. IRT-5 is stated as
  replicating this inside one framework; IRT-2 notes IRT *estimates* the item weights their
  filtering sets to 0/1 — (source: docs/topics/reference/evaluation-methodology-literature.md;
  docs/potential-projs/irt-reanalysis.md §1/§4).
- **OLMES, *A Standard for Language Model Evaluations*** (NAACL Findings 2025 per the record)
  — the harness; the source of the metric columns the response variable is chosen from —
  (source: docs/topics/reference/evaluation-methodology-literature.md;
  docs/topics/reference/datadecide-data-pipeline.md).
- **The fixed-checkpoint variance rule** — for loglikelihood evals, re-evaluating a fixed
  checkpoint with new seeds buys nothing; configuration variance is a bias axis to sweep —
  (source: docs/topics/reference/evaluation-methodology-literature.md, 2026-08-18).
- **Pooling, trajectory-window replicates, and item bootstrap as the answer to n=3 seeds** —
  pooled variance across 25 recipes at fixed scale; late-checkpoint windows as replicates
  (Signal-and-Noise's own trick); item bootstrap for benchmark-composition uncertainty —
  (source: docs/refs/research-trajectory-pre-to-post-training.md).
- **DataDecide (Magnusson et al., Ai2, ICML 2025, arXiv 2504.11393)** — the suite supplying
  the response matrix; 25 corpora, ≤1B, 3 seeds, 150M→1B ~80%, continuous proxies at 0.01%
  compute — (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/litreview/citation-verification-ledger.md).

**The proxy frontier θ is compared against (the decision-reliability sub)**

- **Patel, Reddy, Mosbach & Bahdanau 2026 (arXiv 2605.18607)** — RankSVM over 80 token-level
  proxies computed on expert solutions; ρ 0.81 vs. 0.36 for cross-entropy and 0.33 for
  rBridge; ranks DataDecide corpora at 10⁻⁵ of target compute. The §4 entry records the
  granularity mismatch (their statistics are per-token; `choices.parquet` is per-choice) and
  that their entropy-/frequency-weighted winners echo the per-character normalization result.
  Per the SciSpace review, unverified — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md;
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-22).
- **rBridge ("Koh & Liang 2026", no ID)** — their expert-reweighted-NLL baseline; the ledger
  flags it unverifiable and likely fabricated — (source:
  docs/litreview/citation-verification-ledger.md).
- **The reproduced DataDecide continuous proxies as incumbent baselines** — "how much further
  does θ push past the best continuous proxy"; per-character normalization winning 9/10 tasks
  is a design input for the continuous-response variant. Agent-written verification code
  Danielle has not read or rerun — flags, not findings — (source:
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-22;
  docs/topics/reference/datadecide-data-pipeline.md).
- **Pechi et al. (2305.17266)** small-scale break below ~2.2e15 FLOPs; **ADO (2410.11820)**
  small proxies often fail to predict larger models — the bounds on how far down the frontier
  can go — (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **The frontier design brief from Danielle's ≤1%-compute skepticism** — normalize by target
  compute but report iso-compute *bands* with cell composition explicit; separate the
  size-ladder axis from the within-run axis; evaluate metric selection out-of-cell; "best
  proxy achieved 0.874" is a post-hoc max with winner's-curse inflation; noise floors are
  per-task-per-recipe objects (0.02 typical, 0.111 max, BoolQ-driven) — (source:
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-22).

**Response-variable evidence and the margin demotion (IRT-5, IRT-9)**

- **Reproduction finding: margin tracks accuracy at mean ρ 0.360 (negative on ARC Challenge,
  PIQA, WinoGrande) while Normalized Correct Probability tracks at 0.916**; per-character
  normalization won 9/10 tasks. Consequence recorded: per-character Normalized Correct
  Probability becomes the primary continuous response and margin becomes an object of study.
  Proposed mechanism: plausible-by-construction distractors rise faster than the correct
  answer on near-minimal-pair tasks — (source:
  docs/potential-projs/irt-reanalysis.md §4; docs/topics/reference/datadecide-data-pipeline.md;
  agent-written verification code, unread by Danielle).
- **Metric hierarchy from the pipeline** — "raw Correct/Total Prob dominates at most small
  scales" failed at 2.38% vs. a >50% predicate; raw-plateau/penalized-converge failed on both
  halves; normalized metrics won 816/830 near target; the gap grew with compute (0.023 →
  0.134), so response-model comparison is a permanent methodological question rather than a
  small-scale workaround. A definition-matching pass against the paper's released analysis
  code is required first — (source: same).
- **The OLMES metric-column reconstruction** — five scoring rules (raw, unconditional-
  normalized, per-byte, per-char, per-token) with continuous companions `correct_prob*`,
  `norm_correct_prob*`, `total_prob*`, `margin*`, `bits_per_byte_corr`; only
  `correct_prob = exp(sum_logits_corr)` is checked on data; per-char scoring is a
  per-character geometric mean; `bits_per_byte_corr` is the one aggregate not rebuildable
  from instance details — (source: docs/topics/reference/datadecide-data-pipeline.md;
  docs/potential-projs/datadecide-data-card.md §4).
- **Loss-replacement family as candidate response variables** — LongPPL key-token perplexity
  (2410.23771); PPLqa (2411.15320); Rho-1 (2404.07965); bits-per-byte (Biderman et al.
  2405.14782; Paloma 2312.10523); tokenization-marginal likelihood (Cao & Rimell 2021;
  Vieira et al. 2412.03719; Takahashi et al. 2019); Diff-eRank (2401.17139) and Matrix
  Nuclear-Norm (2410.10672) as the hidden-state family — (source:
  docs/topics/reference/loss-alternative-metrics-literature.md; SciSpace review, unverified).

**Psychometric method precedents on record (methods, not citations)**

- **The staged difficulty recipe** — smoothed pass rate (α = 0.5, d = −logit p̂) → Rasch/1PL →
  many-facet Rasch / logistic mixed model (`correct ~ 1 + (1|sample) + (1|model) + (1|prompt)
  + (1|model:prompt)`) → only then 2PL. The right order for a first fit here, with facets
  recipe/size/seed/step — (source:
  docs/topics/reference/estimation-and-calibration-methods.md, third entry;
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-22).
- **The per-item diagnostic split** — overall difficulty, discrimination, facet sensitivity,
  "random-looking" items — the table IRT-8's autopsy should produce for every task, not just
  BoolQ — (source: same).
- **Leave-one-facet-out validation** — fit difficulty without one recipe or one size, test
  whether it predicts the held-out facet; the cheap generalization check for IRT-3's
  recipe-DIF claims — (source: same).
- **Respondent-count caveat** — 2PL is noisy at 32 respondents; does not bind for a
  thousands-strong checkpoint pool but applies to any per-recipe or per-size subfit —
  (source: same).
- **Response-level choice when a generation is scored by a whole test suite** — all-pass
  (Rasch, primary) vs. passed-count (beta-binomial, secondary) vs. per-test with a mandatory
  generation-level random effect u_ij (diagnostic only); the average pass fraction measures
  test-suite density — (source:
  docs/topics/reference/estimation-and-calibration-methods.md, continuation entry).
- **DIF test menu named in §1** — Mantel–Haenszel, logistic-regression DIF, multi-group IRT
  with anchor items, with multiple-comparison control — (source:
  docs/potential-projs/irt-reanalysis.md §1/§3).
- **Fitting tools named** — `py-irt`, `girth`, a small PyTorch/NumPyro 2PL; `mirt`-equivalent
  absent from `statsmodels`; marginal-ML or VI rather than MCMC at this matrix size —
  (source: docs/potential-projs/irt-reanalysis.md §2).

**IRT-11: which item representation explains difficulty (training-free scoring)**

- **Cluster R² / correlation ratio η², with adjusted R² / ω²** and a fixed clustering rule
  across embeddings — (source: docs/topics/reference/estimation-and-calibration-methods.md,
  second entry).
- **NMI / V-measure / ARI / purity** when difficulty is binned — (source: same).
- **kNN difficulty smoothness against a shuffled-label baseline** — the response's preferred
  primary metric because it scores the embedding rather than embedding + clustering —
  (source: same).
- **Pairwise distance–difficulty Spearman (Mantel-style)** — (source: same).
- **Two Claude-added intake cautions** — the shuffle null must respect benchmark block
  structure (within-benchmark shuffle), and pairwise Spearman over n² dependent pairs needs a
  permutation p-value — (source: same).
- **Empirical test clustering by the pass/fail vector across the respondent grid** — the
  test-level analogue with the pass/fail vector as the embedding; the same η²/kNN metrics
  score candidate groupings — (source: same, DSPy/GEPA continuation; GEPA 2507.19457
  ledgered, agent-supplied).

**Ranking metrics for the decision-reliability sub**

- **Weighted Kendall τ (Vigna 2015, 1404.3325; `scipy.stats.weightedtau`)** — top-weighted
  permutation correlation; Claude-added and unverified — (source:
  docs/topics/reference/estimation-and-calibration-methods.md, last entry).
- **Rank-biased overlap (Webber, Moffat & Zobel 2010, TOIS)** — persistence parameter sets
  top-heaviness; Claude-added and unverified — (source: same).
- **Blest's weighted rank correlation / the Wroclaw ν family; NDCG@K; MAP@K; P@K / R@K** —
  the response's recsys menu, with the intake correction that MAP/P@K need a
  relevant/irrelevant split and NDCG's log discount is too mild for the 1↔2 swap — (source: same).
- **Top-1 / top-k hit and regret** — regret named as the decision-theoretic quantity
  DataDecide actually cares about, immune to the baselines-look-fine problem — (source: same).
- **Decision accuracy is (τ+1)/2 up to ties** — so pairwise saturation is identical to τ
  saturation; report stratified by true-rank gap (adjacent pairs; pairs within the top 5) —
  (source: same).

**Format intervention (IRT-10) prior art**

- **"The Hidden Cost of Structure" (RANLP 2025; 11 models)** — constraints often *help* base
  models and degrade instruction-tuned models on generation; DataDecide checkpoints are base
  models, making this a direct expected-direction prior — (source:
  docs/topics/reference/structured-output-literature.md;
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-22).
- **"Quantifying the Impact of Structured Output Format on LLMs' Reasoning Performance"
  (EACL Findings 2026)** — the sign depends on model, task, schema, and prompt; the reason to
  report the intervention per checkpoint rather than pooled — (source: same).
- **Structured Output Benchmark (2604.25359); ExtractBench (2602.12247); JSONSchemaBench
  (2501.10868); "When Correct Isn't Usable" (2605.02363); LLMStructBench (2602.14743);
  clinical SLM extraction (2507.01810); SLOT (EMNLP Industry 2025); GLiNER2 (2507.18546);
  ScrapeGraphAI-100k (2602.15189); VAREX (2603.15118); Schema RL (2502.18878); RL-Struct
  (2512.00319); schema key wording (2604.14862); PA-Tool (2510.07248)** — the wider
  format-vs-content literature; thirteen of these IDs are agent-supplied with very recent
  numbers whose title–ID pairing is explicitly flagged for verification — (source:
  docs/topics/reference/structured-output-literature.md;
  docs/litreview/citation-verification-ledger.md).
- **The prompt facet as the other side of IRT-10** — per-item prompt sensitivity is a direct
  estimate of "format-limited" vs. "hard" — (source:
  docs/topics/reference/estimation-and-calibration-methods.md, intake note).

**BoolQ autopsy (IRT-8) — the diagnostic apparatus on record**

- **The two-fingerprint test** — "genuinely hard" gives high difficulty, normal
  discrimination, smooth sub-threshold margins; "broken as measurement" gives near-zero
  discrimination *and* large local-independence violations. Item parameters alone do not
  separate them; the aggregate variance structure does — (source:
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-22).
- **The variance arithmetic** — independent guessing at chance over ~3,200 items gives a seed
  SD of ~0.008–0.01; the observed SD up to 0.111 is >10× that floor, producible only under
  strong within-checkpoint response correlation — (source: same;
  docs/topics/reference/datadecide-data-pipeline.md).
- **Prior-tracking / acquiescence bias** — BoolQ is two-choice with a yes-heavy (~60%+,
  unverified) label distribution, so "chance" is ambiguous between 50% and the majority rate;
  prior-tracking predicts the observed phenomenology, "too hard" predicts a flat quiet 50% —
  (source: docs/potential-projs/irt-reanalysis.md §4).
- **Three discriminating tests** — residual inter-item correlation / local dependence;
  regression of per-checkpoint accuracy on predicted-yes fraction; whether margins are
  structured by label rather than content — (source: same).
- **The BoolQ twist** — the "nontrivial only at intermediate 1B checkpoints" claim strongly
  failed (108 nontrivial >0.55 decision-accuracy points, 85 below 1B across eight sizes,
  final 1B 0.7867): predictable while possibly measuring nothing, the sharpest illustration
  that decision accuracy alone is an insufficient validity criterion — (source: same;
  docs/potential-projs/datadecide-data-card.md §4).
- **The two-cluster null** — silhouette 0.207 vs. 0.25 default (reproduced at 0.15); two
  proxy-curve *shape* clusters do not imply two ability dimensions, since a strictly
  one-dimensional model produces different aggregate shapes when tasks differ in
  item-difficulty distributions (Group A: ARC Easy, BoolQ, CSQA, PIQA, SocialIQA; Group B:
  ARC Challenge, HellaSwag, MMLU, OpenBookQA, WinoGrande) — (source: same).
- **Noise-aware crossings** — 15,523 crossovers with all 300 recipe pairs crossing at least
  once; Danielle's bump plots show "super super super consistent" ordering, so ~50 crossings
  per pair is jitter. Adopt: a meaningful crossing exceeds the per-task seed-noise floor and
  persists for k consecutive checkpoints (drift, not diffusion). Crossover density vs. compute
  is a companion statistic to the frontier — (source: same;
  docs/potential-projs/trajectory-statistics.md pointer).
- **The motivation-section pattern** — every ambiguous or failed reproduction traces to a
  metric with a bad functional form (margin), a statistic without a noise model (crossovers,
  seed-SD phrasing), or a threshold without principled basis (silhouette, 0.90 cutoffs) — the
  three things likelihoods, error bars, and model comparison replace — (source: same).

**Continual-learning / plasticity pivot (Claim 2)**

- **The stated hook** — forgetting metrics conflate item difficulty with ability change; task
  orderings are not on a common scale; measurement-invariance testing is the formal criterion
  for "is this benchmark measuring the same thing before and after training on B"; DIF-over-
  time localizes interference. Offered as a recognized methodological hole; **no
  CL-measurement papers are named anywhere in the repo record** — (source:
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-21, external response, unverified).
- **Adjacent plasticity records that could seed it** — *Can Scale Save Us From Plasticity Loss
  in LLMs?* (2606.24752); plasticity injection (Nikishin et al., 2305.15555); Reset & Distill
  (2403.05066); *When Does Re-initialization Work?* (Zaidi et al., 2206.10011); the RL
  plasticity-loss survey (2411.04832); representation-plasticity timeline in LLMs
  (2410.06225) — (source: docs/topics/reference/reinit-and-transfer-literature.md;
  docs/topics/reference/plasticity.md).
- **PolyPythias (2503.09543)** — 9 seeds × 5 sizes, 14M–410M, ~7,000 checkpoints: an
  additional respondent population if the CL pivot needs many models × checkpoints — (source:
  docs/topics/reference/reinit-and-transfer-literature.md).

**Estimation and interval machinery**

- **Wilson / Jeffreys / Clopper–Pearson / Beta posterior for binary pass rates; Hoeffding and
  empirical Bernstein for conservative floors; bootstrap over programs/items; block bootstrap;
  confidence sequences; jackknife** — with the explicit note that `IRT` needs interval choices
  for small-n pass rates and "calibrate after selection" whenever a best recipe or checkpoint
  is chosen from many — (source:
  docs/topics/reference/estimation-and-calibration-methods.md, first entry and relevance map).
- **Split conformal (Lei et al., 1604.04173); conformal risk control (Angelopoulos et al.,
  2208.02814); unbiased pass@k (Codex, 2107.03374)** — Danielle's flagged cross-project tools
  — (source: same).
- **Bayesian shrinkage / hierarchical priors over many items** — trades unbiasedness for MSE;
  report as model-based — (source: same).

**The matched-ability control**

- **The matched-loss caution carried to matched-θ** — "equal loss at different token counts vs.
  equal tokens at different loss are different controls"; report DIF at matched θ both with and
  without conditioning on compute — (source:
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-18).
- **n = 25 exchangeability warning** — IRT-3 currently treats 25 recipes as exchangeable;
  prefer within-family comparisons along a measured dose (the family-contrast framing from
  recipe featurization) — (source: docs/potential-projs/irt-reanalysis.md §4, 2026-08-21).
- **Overlap warning with drift/diffusion** — recipe-DIF and matched-loss drift/diffusion
  signatures are close enough that reviewers will ask why both are needed — (source: same;
  docs/portfolio-rankings.md).
- **Null asymmetry** — IRT-1's null is a genuine substantive claim; a null IRT-3 at 150M is
  ambiguous between "recipes are one-dimensional" and "these scales are too small to see it" —
  (source: docs/potential-projs/irt-reanalysis.md §4).

**Origin record and program placement**

- **The six Danielle-flagged project seeds** (θ as movement metric; DIF as the formalized
  matched-loss comparison; ICCs against compute for per-item emergence; dimensionality;
  binary-vs-continuous replicating Signal and Noise; DIF items clustering by token-determinism
  buckets or domain) mapping onto IRT-1–IRT-5 and IRT-7 — (source:
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-18;
  docs/refs/research-trajectory-pre-to-post-training.md).
- **`dataset-analysis-idea-map.md` item I2** — the earliest written form of the five IRT
  sub-results, with L2 ("measurement instruments over released results") as the layer —
  (source: docs/dataset-analysis-idea-map.md).
- **Per-instance coverage gate: resolved** — all 25 recipes × 66 tasks, 3 seeds at 150M–1B,
  1 seed below 150M; per-instance tables at 4M/20M/60M/90M — (source:
  docs/open-questions-answered.md).
- **Rankings** — Tier 2 #4 on the flagship list ("do it anyway in months 1–2 as the fast
  insurance paper"); workshop-sized #2; full-conference #1, "The Psychometrics of Pretraining
  Suites"; P1 in the four-main-conference list, recommended as the primary starting effort;
  scoop risk medium-high — (source: docs/portfolio-rankings.md;
  docs/potential-projs/irt-reanalysis.md §4).
- **What upgrades workshop → full paper** (external response, unverified): validation of the
  measurement model itself — invariance testing across training, simulation studies showing
  the decomposition recovers ground truth, robustness to the response-model choice (IRT-5) —
  "psychometrics reviewers do this by default; ML papers using IRT mostly skip it" — (source:
  docs/potential-projs/irt-reanalysis.md §4, 2026-08-21).

**Cross-listed neighbors that consume or feed the fit**

- **`tiny-scale-measurement.md` (TINY-2)** — the tiny-scale eval as a *derived artifact* of the
  IRT fit (select items whose difficulty brackets the tiny-model θ range, score with likelihood
  margins, report θ with standard errors); "the effective test length of MMLU-style suites at
  10M is close to zero" — (source: docs/potential-projs/irt-reanalysis.md §4, 2026-08-21;
  docs/potential-projs/tiny-scale-measurement.md).
- **`elicitation-gain.md`** — IRT-10 is cross-listed there as the first concrete instance of
  the elicitation thesis (apparent capability floors that are measurement floors) — (source:
  docs/potential-projs/irt-reanalysis.md §1/§4; docs/potential-projs/elicitation-gain.md).
- **`datadecide-data-card.md` (DCARD)** — supplies the cleaned tables, the metric-definition
  pinning, and the coverage ledger the fit rests on — (source:
  docs/potential-projs/datadecide-data-card.md).
- **`trajectory-statistics.md`** — the drift/diffusion instrument the crossings definition and
  the overlap warning both reference — (source: docs/potential-projs/irt-reanalysis.md §4).
- **The per-example comparison workbench (marimo + Altair)** — consumes the per-item
  probability columns; depends on DCARD-1(e) pinning definitions first — (source:
  docs/topics/reference/experiment-tooling.md;
  docs/topics/reference/datadecide-data-pipeline.md).

**Provenance caveats carried from the records**

- All reproduction numbers cited above come from agent-written verification code Danielle has
  not personally read, debugged, run, or analyzed — flags for where to look first, not
  findings (her statement) — (source: docs/topics/reference/datadecide-data-pipeline.md).
- The SciSpace review seeding the proxy-metric accumulator fabricated the seed paper's author
  list in version 2 and has swapped bibliography entries; prefer version 1 — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, intake notes).
- The structured-output pass is agent-generated with thirteen recent-number IDs flagged for
  title–ID verification — (source: docs/topics/reference/structured-output-literature.md).
- Every arXiv ID above traces to `docs/litreview/citation-verification-ledger.md`, where rows
  are marked *agent-supplied* or *Claude-added* and **nothing is verified**.

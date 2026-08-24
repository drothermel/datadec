# recipe featurization — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`recipe-featurization.md`](../recipe-featurization.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

High-recall corpus of every paper, method, or named prior-art item on record in this
repository that is possibly relevant to recipe featurization (`REC`). Assembled
2026-08-24 from the project doc's §1–§5, the theme accumulators in `docs/topics/`, the
litreview files, the idea map, and the program notes. **Recall is the point** — items
appear here on the strength of one repo mention, not on a judgment that they matter.
Every line names its repo source. Nothing here is verified against the primary record
unless the source says so; agent-generated records (SciSpace reviews, novelty checks,
NBLM tables) are flagged in-line.

**Model-mediated data valuation and mixing laws (the "predicts best, explains least"
family; REC-3's baselines)**

- **Improving Pretraining Data Using Perplexity Correlations** (no ID) — losses of 90
  public LLMs over tens of thousands of web domains as dataset features; the direct
  model-mediated baseline REC-3 measures intrinsic features against — (source:
  docs/topics/reference/data-featurization-literature.md).
- **RegMix / Data Mixture as Regression for Language Model Pre-training** (no ID) —
  mixture selection as regression from small proxy runs, matching DoReMi at ~10% compute;
  baseline for REC-3 and a named outcome-side suite ("RegMix's 1000+ small models") for
  REC-11 — (source: docs/topics/reference/data-featurization-literature.md; §5 of the
  project doc).
- **DoReMi / Optimizing Data Mixtures Speeds Up Language Model Pretraining** (no ID) —
  mixture-weight optimization via excess loss; both a REC-3 baseline and the source of the
  excess-loss machinery REC-2 reuses — (source:
  docs/topics/reference/data-featurization-literature.md; docs/dataset-analysis-idea-map.md F2).
- **Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling
  Performance** (2403.16952) — parametric law over domain proportions nested with scaling
  laws; predicts unseen mixtures; REC-3 baseline and named in the "crowded" half of the
  crowded-vs-open map — (source: docs/topics/reference/schedules-and-annealing-literature.md;
  docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **BiMix / A Bivariate Data Mixing Law** (2405.14908) — joint law over domain proportion
  and data volume; the second parametric-law baseline — (source:
  docs/topics/reference/data-featurization-literature.md;
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, SciSpace-sourced,
  unverified).
- **AutoScale** (2407.20177) — optimal domain mix changes with scale (HQ sources saturate,
  diverse CC keeps paying); a scale-dependence caution for any single-scale REC-3 fit —
  (source: docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **UtiliMax / MEDU** (2501.11747) — portfolio optimization over ablation-estimated or
  LLM-estimated per-source utility; size-aware heuristics as strong baselines; adjacent to
  REC-5's per-source marginal effects — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **PDPC** (2501.13126) — perplexity difference between a weak and a strong model as a
  per-sample "when should this be learned" score; a reference-model-difference feature
  cousin of REC-2's determinism profile — (source:
  docs/topics/reference/schedules-and-annealing-literature.md; flagged there as
  misattributed in one of the three annealing answers; unverified).
- **D-CPT law** (2406.01375) — continued-pretraining mixture law — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, SciSpace-sourced,
  unverified).
- **ADO** (2410.11820) — small proxy models often fail to predict larger ones; a direct
  caution for REC-11's cross-suite transfer and for proxy-scale conclusions — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified).
- **Optimal-mixture laws** (2507.09404) — further mixture-law entry — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified).
- **Loss-to-loss scaling determined by data and tokenizer, not architecture**
  (2502.12120) — an argument that data identity is the load-bearing variable; supports
  featurizing data rather than architecture — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified).
- **DSIR, DsDm** (no IDs) — targeted data selection, named in the midtraining topic's
  missing-canon list; the selection-side counterpart to REC's measurement framing —
  (source: docs/topics/reference/targeted-pretraining-midtraining-literature.md,
  Claude-added intake note, unverified).

**Dataset-similarity embeddings and diversity coefficients (REC-1 and REC-3 baselines)**

- **Task2Vec / Task Embedding for Meta-Learning** (Achille et al.; no ID) — the
  Fisher-embedding dataset representation underlying the alignment and diversity
  coefficients; REC-1 computes the diversity coefficient from it — (source:
  docs/topics/reference/data-featurization-literature.md;
  docs/topics/reference/critical-periods.md).
- **Quantifying the Importance of Data Alignment in Downstream Model Performance**
  (no ID) — alignment coefficients between two datasets, with controlled interventional
  experiments; the similarity-family baseline for REC-3 — (source:
  docs/topics/reference/data-featurization-literature.md).
- **Miranda et al., Beyond Scale: The Diversity Coefficient as a Data Quality Metric**
  (no ID) — expected Task2Vec distance between batches as a quality metric; the concrete
  REC-1 diversity feature — (source: docs/topics/reference/data-featurization-literature.md).
- **Data Similarity is Not Enough to Explain Language Model Performance** (no ID) — the
  recorded negative result the similarity family has to respect; a boundary condition on
  REC-3's similarity baselines — (source:
  docs/topics/reference/data-featurization-literature.md).

**Intrinsic corpus statistics (REC-1's core; "closest to your instinct, least developed")**

- **WIMBD / What's In My Big Data?** (Elazar et al.; no ID) — corpus-level duplication,
  contamination, domain composition, length distributions at trillion-token scale; the
  template REC-1 follows and the "WIMBD sequel that hasn't been written" the crowded-vs-open
  map points at — (source: docs/topics/reference/data-featurization-literature.md;
  docs/potential-projs/recipe-featurization.md §5).
- **Compression-based corpus measures** (gzip ratio, entropy-law style; no ID) — scalar
  complexity/redundancy features in REC-1 — (source:
  docs/topics/reference/data-featurization-literature.md;
  docs/dataset-analysis-idea-map.md F1).
- **Zipf / burstiness / type-token statistics** (no ID) — the intrinsic statistics named as
  "exactly the properties Chan et al. showed cause ICL emergence"; REC-1 features with a
  causal precedent — (source: docs/topics/reference/data-featurization-literature.md).
- **Chan et al. 2022, Data distributional properties drive emergent in-context learning in
  transformers** (no ID) — burstiness, class-distribution skew and within-class variation
  determine whether ICL emerges at all, often at similar training loss; the record calls
  this "the one place intrinsic data statistics have been causally tied to a capability" —
  (source: docs/topics/reference/icl-literature.md;
  docs/topics/reference/data-featurization-literature.md).
- **Raventós et al., Pretraining task diversity and the emergence of non-Bayesian
  in-context learning for regression** (no ID) — a task-diversity threshold; the second
  data-property→capability result, with a flagged citation gap on the source page —
  (source: docs/topics/reference/icl-literature.md).
- **Near-duplicate / exact-match corpus infrastructure — MinHash, SimHash, Bloom filters,
  sketches, n-gram indices, the WIMBD / infini-gram line** (no IDs) — the prerequisite for
  the task-side query features and for duplication/contamination statistics — (source:
  docs/topics/reference/data-featurization-literature.md; cross-ref
  docs/topics/reference/retrieval-storage-tooling.md).
- **Contamination survey** (2503.17793) — 1–45% contamination across benchmarks, inflation
  up to 14% C-Eval / 7% HellaSwag; relevant to REC-1's contamination-against-eval-suite
  feature — (source: docs/topics/reference/schedules-and-annealing-literature.md,
  respondent's numbers, unverified).
- **Time-travel contamination detection** (2308.08493) and **C2LEVA** (2412.04947) —
  contamination-detection methods usable for the REC-1 contamination feature — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, SciSpace-sourced,
  unverified).

**Determinism profile, token-level uncertainty, and river/wall geometry (REC-2, REC-4)**

- **Wen et al.** (2410.05192) — the river-valley toy bigram language; stable phase learns
  deterministic tokens, decay learns stochastic ones; Spearman ~0.39 between token-level
  uncertainty and local sharpness. The record notes the mapping was made *within* one
  distribution and "never comparatively across corpora" — which is REC-2's move — (source:
  docs/topics/reference/token-level-literature.md;
  docs/topics/reference/landscape-literature.md).
- **Rho-1 / Not All Tokens Are What You Need for Pretraining** (2404.07965) —
  reference-model excess-loss token scoring plus a loss-trajectory taxonomy
  (persistently-high/low, descending, fluctuating); the machinery REC-2 reuses and a
  token-bucket-over-time precedent — (source: docs/topics/reference/token-level-literature.md;
  docs/topics/reference/training-objective-alternatives-literature.md).
- **Token-Level Uncertainty-Aware Objective for Language Model Post-Training** (no ID) —
  epistemic vs. aleatoric token uncertainty; epistemic drains faster for low-aleatoric
  examples; the closest existing measurement of bucket migration, with no landscape link —
  (source: docs/topics/reference/token-level-literature.md).
- **Revisiting Entropy in Reinforcement Learning for Large Reasoning Models** (no ID) —
  masking RLVR updates by token regime changes dynamics; the wall-bucket resurfacing
  post-training — (source: docs/topics/reference/token-level-literature.md).
- **Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement
  Learning for LLM Reasoning** (no ID) — a small high-entropy minority carries most of
  RLVR's effect; the same suggestive bridge — (source:
  docs/topics/reference/token-level-literature.md).
- **MiLe** (no ID, Findings NAACL 2024), **TALR** (2509.20758), **RFT** (2412.14780),
  **IR-DRO** (2402.14270), **Power-Law Decay Loss** (2505.16900), **ESLM** (2505.19893),
  **VCORE** (2510.27462), **ScaleGrad** (no ID), **Velocitune** (2411.14318), **tDRO**
  (2408.10613), **XDoGE** (2512.10545) — the token/domain reweighting family; the
  training-side mirror of per-token entropy featurization and a menu of entropy-derived
  scores REC-2 could report — (source:
  docs/topics/reference/training-objective-alternatives-literature.md, SciSpace-sourced,
  unverified).
- **Multi-token prediction** (2404.19737), **patch-level training** (2407.12665),
  **Beyond Log Likelihood / model-capability continuum** (2510.00526) — objective-side
  entries; the capability-continuum result is flagged as a DataDecide-shaped size-ladder
  claim testable if a retrain substrate gets an objective arm (REC-10 adjacency) —
  (source: docs/topics/reference/training-objective-alternatives-literature.md, unverified).

**Annealing, schedules, and decay-branch prior art (REC-4, REC-6, REC-7)**

- **Hägele et al.** (2405.18392) — constant LR + short cooldown matches cosine; (1−√)
  cooldown; decay-branch reuse as the cost model. The methodological argument REC-4's
  premise rests on — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **MiniCPM** (2404.06395) — WSD origin; ~10% decay completes convergence; decay-phase
  gradient statistics (norm falls, consecutive-update cosine turns positive) — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **Llama 3 "Annealing Data"** (no ID) — 8B GSM8K +24 / MATH +6.4, 405B negligible;
  final-40B 30/70 anneal used as a data-valuation instrument; the canonical late-HQ-data
  result and its scale attenuation — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **Blakeney et al., "Does your data spark joy?"** (no ID) — 7B end-of-training domain
  upsampling; MMLU +6.90 / GSM8K +8.26 / HumanEval +6.17 pp; 10–20% of training as the
  budget trade-off point; late upweighting as data valuation — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **OLMo 2 / Dolmino** (2501.00656) — mid-training mix targeting weak spots, LR to zero;
  the open reproducible annealing template — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **OLMo** (no ID) — the training setup DataDecide inherits, i.e. the source of the
  cosine-tail confound REC-4 quantifies — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **TREC / Training Re-evaluation Curves** (Bergsma et al., 2509.25380) — a receptivity
  valley before the end; identical HQ amounts do best near the TREC minimum; predictable
  from AdamW's EMA timescale; claims to explain Llama-3-405B's null — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, flagged "verify before
  building on").
- **Tissue et al.** (2408.11029) — annealing-area term added to the loss-vs-compute law —
  (source: docs/topics/reference/schedules-and-annealing-literature.md).
- **Second LR-annealing scaling-law citation** (2508.01483) — paired everywhere with
  Tissue; the topic explicitly flags it as possibly a mis-ID and asks whether it is a
  follow-up — (source: docs/topics/reference/schedules-and-annealing-literature.md;
  docs/litreview/citation-verification-ledger.md).
- **Multi-power law** (Luo et al., 2503.12811) — loss as a power law in cumulative LR plus
  decay-drop terms; the analytic schedule correction REC-7 fits per recipe and REC-4's
  minimum target — (source: docs/topics/reference/schedules-and-annealing-literature.md;
  docs/topics/reference/loss-curve-forecasting.md).
- **WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging** (no ID) — merged
  checkpoints mirror a true anneal at intermediate stages; the pseudo-anneal REC-4/REC-6
  could use without training — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **Nemotron 3 Super** (no ID) — sliding-window checkpoint merging for quality readouts,
  ~16% FLOP savings — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **Parmar et al. 2024 continued-pretraining recipe** (no ID) — two-stage CPT, switch at
  LR ≈ η_max/5, stay distribution-adjacent — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Large-scale curriculum study** (Zhang et al., 2506.11300) — 0.5–1B models; easy→hard
  ordering by compression ratio / lexical diversity / readability gives lasting gains up to
  +3.5%, with ordering disentangled from selection. Directly relevant to REC-9/REC-10's
  order-effect question and uses intrinsic-statistic difficulty measures — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Influence-driven curricula** (2508.15475) — rank by gradient-similarity influence;
  >10 pp over random in low-resource pretraining — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Temperature sampling vs. scalarization on imbalanced mixtures** (2410.04579) — a
  *mixture-level* cooldown distinct from LR cooldown; relevant to how a realized mixture
  differs from a nominal one (REC-9) — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Branch-and-Merge** (2407.08699) — merging models fine-tuned on data subsets; smaller,
  higher-quality weight changes — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Curriculum cluster** (2405.07490, 2406.19853, 2411.02337, ADCL 2505.08364) — largely
  post-training curricula, kept as leads — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **"Data annealing" for informal language** (2004.13833) — a 2020 BERT-era formal→informal
  gradual-mixing paper; recorded as a *terminology collision*, not prior art — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **Annealed-RLVR** (2509.23629) and **RLHFuse** (no ID) — recorded term collisions
  (RL "heating"; simulated annealing for pipeline scheduling); do not re-flag — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).

**Quality filtering, rewriting, and the syntheticity axis (REC-1 feature; REC-11 target)**

- **FineWeb-Edu** (2406.17557) — educational-value classifier (460k Llama-3-70B-Instruct
  annotations, ~82% F1), 1.3T of 15T tokens, quoted +12% MMLU / +24% ARC; a singleton
  recipe in DataDecide and the canonical classifier-filtering ablation — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **DCLM and FineWeb quality-filtering ablations** (no IDs) — named in the crowded half of
  the crowded-vs-open map and as REC-11's outcome-side suites — (source:
  docs/potential-projs/recipe-featurization.md §4/§5).
- **Phi-4** (2412.08905) — synthetic data throughout and especially late; decontamination
  incl. MinHash fuzzy + semantic; post-cutoff AMC as contamination-proof eval — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Nemotron-CC** (2412.02595) — ensemble quality classifiers plus synthetic rephrasing of
  high-quality segments; a rephrased-corpus suite REC-11 would need for the syntheticity
  feature to vary — (source: docs/topics/reference/schedules-and-annealing-literature.md).
- **SwallowCode / SwallowMath** (2505.02881) — "transform-and-retain" rewriting (+17.0
  HumanEval, +12.4 GSM8K); rewriting rather than selecting as the data upgrade —
  (source: docs/topics/staging/rewritten-anneal-slice.md;
  docs/topics/reference/schedules-and-annealing-literature.md).
- **ProX** (2409.17115) — a 0.3B model emits per-document refinement programs — (source:
  docs/topics/staging/rewritten-anneal-slice.md).
- **FinerWeb-10BT** (2501.07314) — line-level LLM filtering (GPT-4o-mini labels → DeBERTa)
  — (source: docs/topics/staging/rewritten-anneal-slice.md).
- **YuLan-Mini** (2412.17743) — context extension during annealing; topic-based recall and
  cross-lingual synthetic generation — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **"Effective tokens = diversity × syntheticity" (teacher-measured)** (2410.03083) —
  r = 0.83 over 200 models 25M–1.5B; the concrete syntheticity feature proposed for REC-1 —
  (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md;
  docs/potential-projs/recipe-featurization.md §4; SciSpace-sourced, unverified).
- **Diversity of synthetic data and its effect on training** (2410.15226) — the diversity
  half of that construct — (source: docs/topics/reference/synthetic-data-literature.md,
  SciSpace-sourced, unverified).
- **BeyondWeb** (2508.10975) — trillion-scale rephrasing "lessons"; the rephrased-corpus
  suite REC-11 would ingest — (source:
  docs/topics/reference/synthetic-data-literature.md, unverified).
- **WRAP / Rephrasing the Web** (2401.16380) — listed as canon to check for before relying
  on the SciSpace bundle; Claude-added ID — (source:
  docs/topics/reference/synthetic-data-literature.md;
  docs/litreview/citation-verification-ledger.md, unverified).
- **Scaling data-constrained LMs** (2305.16264) — repetition up to ~4 epochs ≈ free; the
  repeated-data boundary any REC-10 retrain arm inherits — (source:
  docs/topics/reference/synthetic-data-literature.md, unverified).
- **A Tale of Tails / model collapse as a change of scaling laws** (2402.07043) — the
  overfitting failure mode for recipes built on generated text — (source:
  docs/topics/reference/synthetic-data-literature.md, unverified).
- **Beyond collapse — scaling with synthesized data requires verification** (2406.07515)
  and **How to synthesize text without collapse** (2412.14689) — the remedies paired with
  the collapse result — (source: docs/topics/reference/synthetic-data-literature.md,
  unverified).
- **Curse of recursion** (2305.17493) and **accumulate-don't-replace** (2404.01413) —
  listed as missing collapse canon; Claude-added IDs, hallucination-prone — (source:
  docs/topics/reference/synthetic-data-literature.md;
  docs/litreview/citation-verification-ledger.md).
- **Self-Instruct** (2212.10560), **GLAN** (2402.13064), **MAmmoTH2**,
  **Instruct-SkillMix** (no IDs) — instruction-synthesis methods; peripheral to REC but on
  the same axis — (source: docs/topics/reference/synthetic-data-literature.md, unverified).
- **Synthetic-data surveys** (2406.15126; 2410.12896) and **best practices / lessons**
  (2404.07503) — taxonomy of generation, curation, augmentation vs. synthesis, evaluation
  pollution — (source: docs/topics/reference/synthetic-data-literature.md, unverified).
- **When scaling meets LLM finetuning** (2402.17193), **quality-aware scaling with a
  quality parameter Q** (2510.03313) — quality as a scaling-law parameter, i.e. the
  reduced-form version of what REC-1 tries to measure directly — (source:
  docs/topics/reference/synthetic-data-literature.md, unverified).
- **Two unidentified key PDFs in the synthetic bundle** (2405.03548, 2510.01631) —
  unsummarized; flagged for later identification — (source:
  docs/topics/reference/synthetic-data-literature.md;
  docs/litreview/citation-verification-ledger.md, Claude-added).

**Data attribution and per-source effects (REC-5; the task-side/perturbation family)**

- **Influence functions, datamodels, TRAK, TracIn-style influence, representer points**
  (no IDs) — example-level attribution named as precise but brutally expensive at
  pretraining scale; the granularity REC-5 sidesteps by working at leaf-corpus share —
  (source: docs/potential-projs/recipe-featurization.md §4 undated entry;
  docs/potential-projs/functional-featurization.md §4).
- **Membership inference / extraction tests** (no IDs) — "does the model behave as if it
  saw it"; part of the behavioral-attribution family in the unsourced task-side response —
  (source: docs/potential-projs/recipe-featurization.md §4, explicitly unsourced).
- **Task-side query features — exact/near-duplicate search, entity and template retrieval,
  embedding retrieval with aggregate similarity, cluster coverage, semantic proximity
  mass** (no IDs) — Danielle's origin statement of the "datasets are unknowable" objection
  and its retrieval answer; the feature-construction recipe for the similarity family —
  (source: docs/potential-projs/recipe-featurization.md §4, recorded as unsourced;
  docs/topics/reference/data-featurization-literature.md).
- **Perturbation sequences as system identification** (no ID) — a task probe set × dataset
  perturbations → outcome vectors; the record notes DataDecide's 25 recipes *are* the
  perturbation set — (source: docs/potential-projs/recipe-featurization.md §4, unsourced).
- **Domain ablations, domain upsampling, end-of-training data valuation** (no IDs) —
  subdomain E of the dedicated lit review, serving REC-5 — (source:
  docs/litreview/recipe-featurization-litreview-plan.md).

**Scaling suites, proxy metrics, and evaluation noise (REC-3's outcome side; REC-11)**

- **DataDecide** (no ID in the topics; the suite itself) — 25 corpora × 14 scales × 3
  seeds; the supervised problem REC sets up; also the suite whose intermediate-checkpoint
  *decisions* were reported to match compute-equivalent final checkpoints (the partial
  cancellation REC-4 tests) — (source:
  docs/topics/reference/schedules-and-annealing-literature.md;
  docs/topics/reference/datadecide-data-pipeline.md).
- **Patel et al., Forecasting Downstream Performance of LLMs With Proxy Metrics**
  (2605.18607) — 80 proxy metrics from one forward pass over expert trajectories; ranks the
  25 DataDecide corpora for a 1B target with decision accuracy > 0.85 at ~10⁻⁵ of target
  compute. A direct consumer of DataDecide and the strongest recorded model-mediated
  competitor to REC-3's feature→outcome map — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md; the topic flags
  version 2 of the review as having *fabricated the author list*, so treat the surrounding
  characterization with care).
- **Heineman et al., Signal and Noise** (no ID) — signal/noise framework over 465
  open-weight models incl. DataDecide and the model-ladder runs; the evaluation-noise floor
  any n=25 feature→outcome claim is measured against — (source:
  docs/topics/reference/evaluation-methodology-literature.md).
- **Model ladders** (2412.04403) — 1% of target compute, within 2 points on some tasks;
  N and D beat FLOPs in overtrained regimes — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified; the topic
  notes a bibliography entry mis-pointing this ID).
- **Gadre et al.** (2403.08540) — perplexity→downstream power law holds on average but
  varies by task, over 104 models 11M–6.9B; the recorded reason cross-entropy fails as a
  selection signal — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified).
- **FLP two-stage loss→performance** (2410.08527) — 5–10% error at 7B/13B — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified).
- **Observational scaling laws** (2405.10938) — ~80 public models, low-dimensional
  capability space; the methodological cousin of regressing outcomes on corpus features —
  (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified).
- **NeuNeu / Neural Neural Scaling Laws** (2601.19831), **Ye et al. BIG-bench
  predictability** (2305.14947), **Schellaert et al. assessors** (2305.12415), **ProxyLM**
  (2406.09334), **lineage-regularized matrix factorization** (2504.19811), **FamiCom**
  (2406.11243) — the learned-predictor flank REC-3 is measured against — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, SciSpace-sourced,
  unverified).
- **Krajewski et al.** (2512.08894) — direct power law for log-accuracy at fixed
  tokens-per-parameter beats the two-stage loss→accuracy route — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified).
- **Pechi et al.** (2305.17266) — small-scale break below ~2.2e15 FLOPs; a scale-floor
  caution for reading DataDecide's smallest sizes — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified).
- **Ali et al.** (2310.08754) — tokenizer metrics uncorrelated with downstream; a negative
  result on one class of intrinsic feature — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified).
- **Knowledge capacity 2 bits/parameter** (2404.05405) and **repeated-data double descent**
  (2205.10487) — capacity and repetition boundaries on what a corpus can deliver —
  (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified).
- **Kaplan; Chinchilla; hyperparameter scaling (2505.13738); context-aware scaling
  (2510.14919)** — background scaling-law entries in the same accumulator — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified).
- **Wei et al. emergence (2206.07682) vs. Schaeffer et al. mirage (2304.15004);
  proxy tasks for emergent abilities (2412.07111)** — the emergence framing behind
  "which task families show recipe differences at all" — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, unverified).
- **OLMES** (no ID) — the eval standard behind DataDecide's task suite and the per-instance
  tables REC-c consumes — (source:
  docs/topics/reference/evaluation-methodology-literature.md;
  docs/open-questions-answered.md).
- **rBridge (expert-reweighted NLL)** (no ID; "Koh & Liang 2026" flagged as unverifiable
  in the ledger) — a proxy-metric baseline in the Patel comparison — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md).

**Generalization / OOD framing (the unanswered half REC occupies)**

- **Generalizing to Unseen Domains: A Survey on Domain Generalization** (2103.03097) —
  (source: docs/topics/reference/generalization-and-ood-literature.md; the whole file is
  flagged unverified, links carry `utm_source=chatgpt.com`).
- **Rethinking the Evaluation Protocol of Domain Generalization** (CVPR 2024, no ID) —
  OOD evaluation-protocol cautions — (source:
  docs/topics/reference/generalization-and-ood-literature.md, unverified).
- **On Calibration and Out-of-domain Generalization** (NeurIPS 2021, no ID) — multi-domain
  calibration as a robustness indicator — (source:
  docs/topics/reference/generalization-and-ood-literature.md, unverified).
- **Towards a Theoretical Framework of Out-of-Distribution Generalization** (NeurIPS 2021,
  no ID) — (source: docs/topics/reference/generalization-and-ood-literature.md, unverified).
- **Towards a Better Evaluation of Out-of-Domain Generalization** (2405.19703) —
  average accuracy approximates true OOD performance poorly; worst-case-across-domains
  proposed. Relevant to whether REC-3 should target means or worst-case task families —
  (source: docs/topics/reference/generalization-and-ood-literature.md, unverified).
- **Towards the Generalization of Contrastive Self-Supervised Learning** (2111.00743) and
  the SSL-generalization strand (MI frameworks; "Rationality implies generalization";
  a Stanford thesis) — the 2025-01 question sequence whose unanswered half ("predict
  performance conditioned on the method") the record says REC and EDP now occupy —
  (source: docs/topics/reference/generalization-and-ood-literature.md, explicitly
  unverified; Danielle's own flag: "I'm not sure I trust the citations").

**Critical periods, plasticity, and data-order effects (REC-9/REC-10 background)**

- **Achille, Rovere & Soatto, Critical Learning Periods in Deep Networks** (ICLR 2019,
  no ID) — early stimulus deficits permanently impair; Information Plasticity via the
  Fisher trace; also the origin of the Task2Vec formalism REC-1 uses — (source:
  docs/topics/reference/critical-periods.md).
- **Igl et al., ITER** (ICLR 2021, no ID) — transient non-stationarity permanently scars
  the representation; grouped with Achille and Ash & Adams as "three communities, one
  claim" — (source: docs/topics/reference/nonstationarity-accounting.md).
- **Ash & Adams, On Warm-Starting Neural Network Training** (NeurIPS 2020, no ID) and
  **DASH** (NeurIPS 2024, no ID) — the warm-starting gap and its stationary-setting theory;
  background for why data order and early exposure can matter beyond endpoint loss —
  (source: docs/topics/reference/plasticity.md).
- **Dohare et al., Loss of plasticity in deep continual learning** (Nature 2024;
  2306.13812); **Lyle et al.** (2303.01486; 2402.18762) — plasticity loss and its link to
  loss-landscape curvature; the cheap training statistics (curvature, feature rank, dead
  units, weight norm) — (source: docs/topics/reference/plasticity.md).
- **The LLM-scale data-placement echo** (no IDs) — 2025–26 results that early exposure
  shapes models more durably than late data, final-window effects, pretraining safety
  behaviors resisting post-training removal; "critical-period phenomenology at scale,
  mostly published without the connection drawn" — (source:
  docs/topics/reference/critical-periods.md).
- **Endogenous non-stationarity / the implicit self-curriculum** (no ID) — even under iid
  data the effective distribution is data weighted by current gradient magnitude, so the
  learning signal migrates to harder tokens; the theoretical frame behind REC-9's drift
  question — (source: docs/topics/reference/nonstationarity-accounting.md).

**Repository-internal records that behave like prior art**

- **The label≠token-share correction** (no ID) — DCLM/Dolma 25/50/75 recipes are 43/69/87%
  DCLM by tokens; scoop risk rated "medium — the label correction is discoverable by anyone
  who looks, but nobody seems to be looking" — (source:
  docs/potential-projs/recipe-featurization.md §1.1, §5;
  ~/drotherm/data/.claude/datadec/2026-08-19/2013-dclm25-dolma75-training-data.md as cited
  there).
- **The crowded-vs-open map** (2026-08-21 external assessment; agent-generated, unverified)
  — crowded: WIMBD-style descriptive statistics, quality-filtering ablations (DCLM,
  FineWeb), mixture optimization (DoReMi, RegMix, mixing laws); open: features predicting
  *dynamics* rather than endpoints, cross-suite transfer of feature→outcome maps,
  midtraining/annealing data ("folklore plus internal lab knowledge" post-MiniCPM),
  post-training data, and the measurement layer itself — (source:
  docs/potential-projs/recipe-featurization.md §4/§5).
- **The dedicated lit review's subdomain partition and cross-cutting gaps** — A dataset
  featurization / intrinsic statistics; B model-mediated valuation and mixing laws;
  C token-level uncertainty and landscape geometry; D scaling suites, proxy metrics,
  evaluation noise, annealing protocols; E data attribution and per-source effects; gaps
  G1 small-n methodology, G2 QC-variant / filter-strength, G3 data-card precedent.
  Post-training amplification scoped out. 75 seed rows (63 exact, 12 title-only), 229
  queries, 48 citation-graph seeds; the final review lives outside this repo at
  `~/drotherm/data/.claude/datadec/2026-08-21/1412-recipe-featurization-litreview/` —
  (source: docs/litreview/recipe-featurization-litreview-plan.md;
  docs/litreview/recipe-featurization-litreview-process.md).
- **Citation-verification ledger, `Feeds: REC` rows** — every REC-feeding ID entered via
  the 2026-08-22 SciSpace intake and is in `synthetic-data-literature`; the ledger states
  "nothing here is verified" and separates *agent-supplied* from *Claude-added* (the latter
  "hallucination-prone in the last digits and in title–ID pairing") — (source:
  docs/litreview/citation-verification-ledger.md).
- **Idea-map lineage** — REC derives from L0 (ground truth about the data), F1 (intrinsic
  statistics), F2 (determinism profile), F3 (feature→outcome); the map names L0 and F2 as
  hubs unlocking three or more downstream ideas, and lists the three-way determinism
  cross-check (F2 static entropy · C1 decay-responsiveness · Rho-1 loss-trajectory classes)
  where "the three should agree if river-valley is right" — (source:
  docs/dataset-analysis-idea-map.md).
- **Ranked-list attributions** — workshop list #1 (REC-a + REC-5) and #7 (realized-exposure
  audit + order-effect experiment); full-conference #2 "What Is Actually in DataDecide";
  P2 in the four-main-conference list — (source: docs/portfolio-rankings.md;
  docs/potential-projs/recipe-featurization.md §4).
- **DataDecide-dense staging doc** — the retrain substrate REC-10's order-effect arm would
  run on (a few recipes × 2–4 smallest scales × 10+ seeds × {cosine, WSD} ×
  {sequential, stratified}) — (source: docs/topics/staging/datadecide-dense.md).
- **Rewritten-vs-selected anneal slice staging doc** — connects REC's syntheticity feature
  to the rewriting cluster; the open question is whether *upgrading* a slice beats
  *selecting* one at matched tokens — (source: docs/topics/staging/rewritten-anneal-slice.md).
- **Per-instance coverage gate** — instance-level OLMES results exist for all 25 recipes ×
  66 tasks but only 9 of 14 sizes, with 3 seeds only at 150M–1B; bounds which scales REC-3
  can use — (source: docs/open-questions-answered.md, 2026-08-21).
- **DCARD (datadecide-data-card.md)** — the data-card / validation-report /
  provenance-ledger material split out of REC on 2026-08-22, with REC-a as its primary
  composition input — (source: docs/potential-projs/recipe-featurization.md §4;
  docs/potential-projs/datadecide-data-card.md).
- **Data-pipeline external readings** — the still-missing analysis-side pieces (coverage
  census, checkpoint-spacing statistics, REC-a manifest module, response-matrix builder)
  and the note that `configs/dataset_features.csv` is absent at head, so the doc's
  reference may be stale — (source: docs/topics/reference/datadecide-data-pipeline.md).

# Recipe featurization — what is actually in the DataDecide recipes

**Program pillars served:** data (what is actually in the recipes; features → dynamics). (Program: `README.md` → Program.)

Status: proposal, 2026-08-21. Derived from [../dataset-analysis-idea-map.md](../dataset-analysis-idea-map.md) (idea-map ideas L0, F1, F2, F3) and the recipe ground-truth dig in
`~/drotherm/data/.claude/datadec/2026-08-19/2013-dclm25-dolma75-training-data.md`.

**One-line pitch.** DataDecide is a supervised problem nobody has set up: 25 pretraining corpora with measured outcomes across 14 scales, 3 seeds, and many tasks. Measure what is actually in each corpus, compute intrinsic and theory-motivated features, and ask which data properties explain which task-level differences.

---

## 1. What the project involves

### 1.1 Core (non-optional)

**REC-a · Ground truth for every recipe.** Regenerate the exact shard manifest for each of the 25 recipes from `named_data_mixes.py` (OLMo `DataDecide` branch), sum realized token masses from `allenai/DataDecide-data-recipes` file sizes, and flatten each recipe to its leaf corpora (DCLM CC, Dolma CC head/middle/tail, RefinedWeb, StarCoder, C4, Reddit, peS2o, …). Output: a per-recipe composition table in realized tokens. Already done for one recipe; the 25-recipe version is mechanical.

Known finding this immediately surfaces: mixture labels are shard-file fractions, not token fractions — the DCLM/Dolma 25/50/75 recipes are 43/69/87% DCLM by tokens. Any consumer treating labels as mixture weights is wrong, including datadec's own `configs/dataset_features.csv`.

**REC-b · Corpus sampler.** Sample tokens per recipe from the actual shards (stratified by leaf corpus so per-source statistics are possible), optionally detokenize with the `gpt-neox-olmo-dolma-v1_5` tokenizer.

**REC-c · Outcome table.** The DataDecide eval results datadec already ingests, organized as recipe × scale × seed × step × task, with both accuracy and continuous (likelihood/margin) metrics; plus derived pairwise decisions and per-task rankings.

**REC-d · Analysis framing that respects n = 25.** The recipes form families — Dolma 1.7 + 4 ablations, DCLM-Baseline + 6 QC variants, Falcon/Falcon+CC + 4 variants, 3 DCLM/Dolma mixes, and singletons (C4, FineWeb-Pro, FineWeb-Edu, Dolma 1.6++). Within-family contrasts are one-factor interventions with measured outcomes; the analysis leans on those rather than on a free-form 25-point regression. Validation is leave-one-family-out.

### 1.2 Optional directions

**REC-1 · Intrinsic-statistics features.** WIMBD-style corpus statistics (duplication rate, length distributions, source/domain composition from REC-a), contamination against the eval suite, compression ratio, Zipf/burstiness/type-token statistics, Task2Vec diversity coefficient. Cheap, mostly CPU, well-precedented individually; never assembled across these 25 corpora.

**REC-2 · Determinism profile.** Per-token conditional entropy from a reference model (and an ensemble or larger model to separate the aleatoric floor), summarized per recipe as curves over entropy threshold × context length, computed before and after dedup. The river-valley-motivated feature: predicted to relate to landscape geometry and annealing behavior, not just performance. Reuses DoReMi/Rho-1 excess-loss machinery.

**REC-3 · Feature → outcome study.** Which features explain which task-family outcomes; comparison against model-mediated baselines (perplexity-correlation features, RegMix-style mixture weights computed on *realized* shares) and similarity-embedding baselines (Task2Vec alignment). Family contrasts as dose-response evidence (e.g., DCLM QC 3→7→10→20% sweep vs. code/knowledge/commonsense task groups).

**REC-4 · Determinism profile → annealing behavior.** Does REC-2 predict how much each recipe's apparent ranking is schedule artifact? Targets come from the A-layer of the idea map: multi-power-law loss correction (analytic, T0) at minimum; checkpoint-merging pseudo-anneals or short decay branches if available. Further geometry targets once they exist: measured decay gain from short decay branches, interpolation-path curvature between checkpoint pairs, or per-token migration rates (when tokens stop responding to decay) — each produced by a separate project if pursued. This is the bridge to the causal token-bucket work and the part with a mechanism story.

**REC-5 · Per-source attribution via the mixes and ablations.** Use the three mixes (continuous DCLM share: 0/43/69/87/100%) and the Dolma ablations as a designed experiment over leaf-corpus shares, estimating per-task marginal effects of DCLM-CC, StarCoder, Reddit, Flan, math corpora. Purely from REC-a + REC-c; no new features required.

*REC-6–REC-8 ported from the former dataset-featurization doc (`e-dataset-featurization.md`, now removed):*

**REC-6 · Predict annealed outcomes.** Re-run REC-3 against annealed outcome values (evals of decay-branch endpoints or decay-weighted checkpoint merges, if another project produces them) instead of raw ones; report whether feature importance shifts once the schedule artifact is removed.

**REC-7 · Curve-parameter targets.** Fit the multi-power law (loss as a power law in cumulative LR plus decay-drop terms) to each recipe's released loss curve — T0, using `processed/scaling-law/checkpoint-losses.parquet` with `lr_at_step` / `cumulative_lr` — and use the per-recipe fitted parameters as the regression target: do intrinsic features predict the along-river term, the decay term, or neither?

**REC-8 · Per-task feature maps.** Which features predict which tasks; whether code / math / knowledge tasks load on different intrinsic statistics.

**REC-9 · Realized-exposure audit (time-resolved REC-a).** Reconstruct the actual token-stream order for each run from the OLMo training configuration (deterministic ordering) and plot realized source composition as a function of training position, per scale. Two failure modes: the run's time-averaged mixture deviating from nominal (bias) and within-run compositional drift (non-stationarity — an unintended curriculum). If drift is material, small-scale recipe comparisons are partially confounded with data order, and the confound shrinks with scale.

**REC-10 · Order-effect intervention.** Retrain 10–50M models on the same recipe with stratified vs. sequential sampling, ten-plus seeds; measure the order effect directly against a noise floor. Outcome-robust: material drift is a confound finding about a widely used suite; no drift is a validation. A stratified-sampling data loader is the concrete artifact.

**REC-11 · Cross-suite transfer.** Re-fit the REC-3 feature→outcome map on other controlled suites (DCLM runs, FineWeb ablations, RegMix's small models) and test whether relationships fit on DataDecide transfer; pooling suites attacks n = 25. Prefer dynamics targets (schedule sensitivity, emergence timing, noise levels) over endpoints.

---

## 2. Doability and impact

Calibration: "impact" is judged against workshop-paper expectations — a contained, defensible result that people working on pretraining data would find worth reading and citing.

### 2.1 Overall doability — high

- All inputs are public and already reachable: manifests regenerate deterministically, shards are on HF, outcomes are in datadec. No training, no external dependencies.
- Compute is small: REC-1 is CPU over samples; REC-2 is reference-model inference (~20–50M tokens × 25 recipes × a few context lengths, hours-to-a-day on one GPU).
- Timeline with REC-a/REC-b/REC-c built: 3–5 focused weeks to a workshop draft.

Risks are statistical, not engineering:

| Risk | Severity | Mitigation |
|---|---|---|
| Effective n well below 25 (families, collinear features) | high | family-contrast design, few pre-registered features, leave-one-family-out validation, rank-based targets |
| Outcomes are unannealed cosine checkpoints | medium | use rankings (reported stable) as primary targets; REC-4 quantifies the artifact explicitly |
| Determinism is relative to reference model and context | medium | report curves, not a scalar; two reference models; pre/post-dedup |
| Model-mediated baselines likely predict better | medium | frame as measurement + explanation, not prediction R² |
| Pairwise decisions are not independent | low | treat them as descriptive, not as test statistics |

### 2.2 Per-direction assessment

| Direction | Doability | Workshop impact | Why |
|---|---|---|---|
| **Core REC-a (realized composition + label correction)** | very high; ~1 week | **medium-high, and the most likely to be cited** | First measured data card for a widely used suite; the label≠token-share correction is concrete and affects anyone using the mixes |
| **REC-1 intrinsic statistics** | high | low-medium alone | Well-trodden descriptors; value comes only when joined to outcomes (REC-3) |
| **REC-2 determinism profile** | high; main GPU cost | medium; highest variance | Novel, theory-linked feature; a clean correlation with task-family gains or annealing response would be discussed. Weak signal at this scale is plausible |
| **REC-3 feature→outcome** | medium (statistics) | medium with family framing; low as a regression paper | Within-family dose-response statements per task family are workshop-sized; a 25-point R² is not |
| **REC-4 determinism→annealing** | medium; depends on A-layer targets | medium-high if positive | Only direction with a mechanism story; sets up the causal follow-on. Needs at least the analytic correction, ideally branches |
| **REC-5 per-source attribution** | high | medium | Clean designed-experiment reading of existing results; interpretable per-task marginal effects of code/Reddit/Flan/DCLM-CC. Limited by few levels per factor |
| **REC-6 annealed targets** | medium; depends on annealed outcomes existing | medium | Makes the paper schedule-aware; a clean "feature importance shifts once you anneal" result would be notable. |
| **REC-7 MPL-parameter targets** | high; the MPL fit is T0 | medium | Compact targets with a physical interpretation; the fit itself is cheap. |
| **REC-8 per-task maps** | high; cheap once REC-3 exists | medium | Interpretable and practically useful; cheap once REC-3 exists. |
| **REC-9 realized-exposure audit** | high; analysis over REC-a machinery + OLMo data order | medium-high | Corrects "the recipe" as an intervention; a candidate explanation for small-scale mispredictions. |
| **REC-10 order-effect intervention** | medium; tiny training, many seeds | medium-high (→ high if the effect is large) | Two-sided result; "proxy-scale data decisions are confounded with data order" would be disruptive. |
| **REC-11 cross-suite transfer** | medium; multi-suite ingest | medium-high | Fixes n = 25 by pooling; irreducibly correlational. |

### 2.3 Recommended framing

"What is actually in the DataDecide recipes, and which measurable data properties explain which task-level differences." Measurement first (REC-a), family contrasts and per-source attribution as evidence (REC-5, REC-3), determinism profile as the novel feature (REC-2), and REC-4 as the forward-looking section. Avoid a headline that is a prediction score. Drop REC-1 to a supporting role.

Expected outcome bands:
- **Floor (high confidence):** data card + label correction + per-source attribution. Publishable as a short paper; useful regardless of REC-2/REC-4.
- **Target:** floor + determinism profiles with an interpretable relationship to task-family outcomes or to annealing response.
- **Ceiling:** REC-4 shows the profile predicts schedule-artifact magnitude across recipes, motivating the WSD-branch / token-migration project.

---

## 3. Proposed infrastructure sequence

Each step produces something usable on its own and is reused by later per-token and branch-based analyses.

1. **Recipe manifest + composition module (REC-a).** Exec `named_data_mixes.py` in a pinned copy, emit per-recipe shard lists and per-leaf-corpus realized token counts from HF `paths-info` sizes; persist as a versioned table in datadec. Replace `configs/dataset_features.csv` with the measured table. *Reused by:* everything below, probe-corpus construction.
2. **Shard sampler (REC-b).** Given a recipe and a budget, sample uint16 token windows stratified by leaf corpus from `allenai/DataDecide-data-recipes`, with optional detokenization and a deterministic seed; cache locally. *Reused by:* REC-1, REC-2, probe-corpus construction for per-token analyses.
3. **Outcome table with full structure (REC-c).** Extend the existing ingest to a tidy recipe × scale × seed × step × task frame with accuracy and continuous metrics, plus helpers for rankings, pairwise decisions, task families, and leave-one-family-out splits. Confirm per-item availability while here (gates the IRT project, not this one). *Reused by:* every analysis over the outcome table.
4. **Intrinsic feature extractors (REC-1).** Pure functions over sampled text/tokens: duplication, lengths, compression ratio, Zipf/burstiness/TTR, diversity coefficient. *Reused by:* REC-3.
5. **Reference-model token scorer (REC-2).** Per-token entropy/logprob from one or two open reference models at several context lengths; emits per-recipe profile curves. *Reused by:* any entropy-bucketed per-token analysis — the single most shared piece after the sampler.
6. **Feature–outcome analysis (REC-3, REC-5).** Family-contrast and dose-response analyses, baselines, figures. Analysis code, not infrastructure.
7. **Data-order reconstruction (REC-9).** From the OLMo `DataDecide` configs, rebuild the per-run token-stream order and join to the REC-a manifest to emit realized composition per training window and per scale.
8. **Stratified-sampling loader + tiny retrain harness (REC-10).** 10–50M runs, many seeds, deterministic order, stratified vs. sequential arms.
9. **Multi-suite ingest (REC-11)**, following the repo's download → preprocess → typed parquet pattern.
10. **Annealing targets (REC-4), only if continuing.** Multi-power-law correction over released loss curves (analytic); then, if the A-layer is pursued, checkpoint-merging utility and decay-branch harness feed REC-4 as additional targets.

Steps 1–3 are the commit-worthy foundation and take roughly the first week; 4–5 are a second week; 6 is the paper. Step 7 is deferred and belongs to the annealing project.

---

## 4. Decisions needed before starting

- **Reference model(s) for REC-2** and the context lengths to report.
- **Sampling budget per recipe** (trade: per-leaf-corpus resolution vs. GPU time for REC-2).
- **Primary outcome targets:** rankings vs. levels; which task families; which scales (likely 150M–1B where signal exists).
- **Whether REC-4 is in scope for the first paper** or deferred to the annealing project.

---

## 4. External assessments

Dated, attributed notes from external review conversations, recorded for consolidation — not
decisions. Only notes about this project are kept here.

### 2026-08-21 — two "top-N by workshop-paper likelihood × speed" lists

- **Ranked #3 in a top-3 list (the floor only).** "REC-a data card + label correction + REC-5
  per-source attribution. The composition ground-truth work is partially done, mechanical to
  extend, and the shard-fraction ≠ token-fraction correction is concrete, surprising, and
  affects everyone using DataDecide's mixes — including your own repo's config. That's the
  'most likely to be cited' claim in your own assessment, and I agree with it. REC-5 (using the
  DCLM-share sweep and Dolma ablations as a designed experiment) needs no new features at all,
  just REC-a joined to outcomes you've already ingested. You can defer the determinism profile
  (REC-2) and the underpowered regression to a second iteration."
- **Ranked #3 in a top-5 list (the floor only).** "REC-a + REC-5. The realized-composition
  data card plus the label≠token-share correction is concrete, already one-25th done, affects
  everyone using the suite, and is your most citable single finding across all documents.
  Per-source attribution (REC-5) needs only REC-a + the outcome table. This is the highest
  certainty-per-week item on the list; the determinism profile (REC-2) is the stretch goal,
  not the core."

- **n = 25.** "The family-contrast/dose-response framing is the right answer and
  should back-propagate everywhere recipes are compared" — i.e. REC-d is a program-level
  design pattern, not just this project's.

### 2026-08-21 — on generalizing beyond DataDecide (full response)

"Yes, with an important qualification: the space is real, but it's unevenly occupied, and the
crowded parts are exactly the ones that are easiest to do. So agreement, but the value depends
on picking the right sub-region."

**Where it's crowded.** "Descriptive corpus statistics as an end in themselves — WIMBD
basically planted that flag for the big public pretraining corpora (duplication,
contamination, toxicity, length distributions across C4, Pile, Dolma, etc.).
Quality-filtering ablations are also well-tilled: DCLM and FineWeb are essentially large
empirical programs on 'which filter produces which outcome.' And mixture optimization (DoReMi,
RegMix, data mixing laws) covers the 'predict the best weights' version. A paper whose
contribution is 'we computed features on big datasets' or 'we found that dedup and quality
filtering matter' lands in a saturated area."

**The structural bottleneck, and why DataDecide mattered.** "Featurization is cheap;
*outcomes* are expensive. The whole reason your featurizing proposal works is that someone
else paid for 25 recipes × 14 scales × 3 seeds. Outside DataDecide, your options for the
outcome side are: other controlled suites (DCLM's runs, FineWeb ablations, RegMix's 1000+
small models — these exist and are underused as supervised problems in exactly your sense),
observational model zoos (hundreds of open models, but confounded provenance — data,
architecture, and scale all vary at once), or training your own (compute-bound). This is the
real constraint on the generalized version of the idea, and it's worth being clear-eyed that
'featurize datasets and predict impacts' is really 'assemble (corpus, outcome) pairs,' which
is a data-collection problem more than an analysis problem."

**Where the genuinely open space is.**

- *Features that predict dynamics, not endpoints.* "Nearly all existing work maps features →
  final benchmark scores. Your determinism-profile → annealing-response idea generalizes:
  which data properties predict schedule sensitivity, emergence timing, noise levels,
  forgetting rates, susceptibility to churn. Nobody has a public map of that, and it's
  theory-motivated rather than fishing."
- *Cross-suite transfer of feature→outcome maps.* "Does a relationship fit on DataDecide hold
  on the DCLM pool or FineWeb ablations? A feature that predicts across suites is a finding;
  one that doesn't is an important negative about how contingent all these results are. This
  is cheap once the extractors exist and directly attacks the n=25 problem by pooling suites."
- *Midtraining/annealing data.* "This is the most practically hot and least publicly
  systematized area. Everyone post-MiniCPM knows you put 'high quality' data in the decay
  phase; what 'high quality' measurably *is*, and whether the right decay data is a function
  of the stable-phase data, is mostly folklore plus internal lab knowledge. A controlled
  featurization-plus-ablation study here would get read by practitioners immediately. The
  catch: fewer public controlled suites exist, so you'd likely need some training compute —
  though decay branches are cheap, which is exactly the [annealed-readouts / WSD-suite]
  machinery."
- *Post-training data.* "Also underdeveloped publicly, but different in character:
  instruction/preference/RLVR datasets are small enough that per-example influence methods
  (datamodels-style) become feasible, and the interesting features are things like difficulty
  distributions, response-length confounds, and diversity — not Zipf exponents. It's arguably
  a different field with the same slogan, and the confound structure (data interacts strongly
  with the base model) makes clean claims harder."
- *The measurement layer itself.* "Your [realized-composition] finding — labels ≠ realized
  token shares — almost certainly generalizes. Public datasets' stated compositions vs.
  measured compositions is unglamorous, highly citable, and nobody's job. A 'measured data
  cards' effort across the major public corpora is the WIMBD sequel that hasn't been written."

**One honest caution.** "The frontier labs do versions of this internally at scales and with
outcome data you can't match, and don't publish. That means the academic contribution has to
be either the *public artifact* (measured features + outcomes anyone can build on), the
*theory link* (determinism/geometry-style mechanistic features, which labs have less incentive
to care about), or the *dynamics angle*. Pure predictive performance — 'our features predict
benchmark scores with R²=X' — is a race you lose to people with a thousand internal
ablations."

**Framing.** "Agree, and I'd frame the generalized program as 'data measurement → training
dynamics,' not 'data features → benchmark scores.' The DataDecide work then becomes the first
instrument-validation study in a program rather than the whole program — which is also a much
better story for a thesis or grant than a single-suite reanalysis."

---

### 2026-08-21 — on MoE releases as extra outcome data

- Against using MoE releases to enlarge the REC-3 outcome table: "it adds architecture as a
  confound without adding recipe variation, and n stays tiny. One model per data point, with
  architecture, tokenizer, and scale all varying between releases, is a worse supervised
  problem than DataDecide, not a better one."
- For: expert assignment as "a data fingerprint" — "does the expert decomposition recover your
  intrinsic features (domain composition, frequency bands, determinism profile)?" This
  "strengthens the case for building [the] reference-model scorer and corpus-feature
  extractors first (they're what routing gets joined *to*)." (Full discussion in
  `docs/potential-projs/moe-partitions.md`.)

### 2026-08-21 — on a time-resolved REC-a (per-window realized mixture)

- Question raised: DataDecide uses a small fraction of each recipe's corpora, so "unless they
  use stratified sampling throughout training they are likely getting real nonstationarity
  or not really hitting the percentages that they expect."
- Response: "This is the time-resolved extension of [REC-a] (labels ≠ realized token shares,
  now per-window rather than in aggregate). It's checkable: OLMo-style training logs data
  order deterministically, so you can reconstruct the realized mixture per window for every
  DataDecide run and measure the nonstationarity directly. If it's substantial, it (a) is a
  standalone audit paper in the same vein as [REC-a] but with dynamics implications, (b)
  confounds every timing/curriculum claim built on these suites… and (c) motivates a concrete
  artifact: a stratified-sampling data loader as the fix." Listed as an open gate in
  `docs/open-questions-answered.md`.

- Refinement: "Two distinct failure modes: the run's time-averaged mixture deviating from
  nominal (a bias), and within-run compositional drift (non-stationarity — effectively an
  unintended curriculum)… plot realized composition as a function of training position, per
  scale. If there's material drift, every small-scale recipe comparison in the suite is
  partially confounded with data *order*, and the confound shrinks with scale — which would
  itself be a candidate explanation for why small-scale decisions sometimes mispredict
  large-scale ones." Interventional follow-up: "retrain 10–50M models with stratified vs.
  sequential sampling of the same recipe, n=10 seeds, and measure the order effect directly."

### 2026-08-21 — positions in three ranked lists (full lists in `docs/portfolio-rankings.md`)

- **6–12-month flagship list:** the determinism-profile data link is "folded in" to the
  flagship; cross-suite features→dynamics is **Tier 2, #3** ("irreducibly correlational…
  its best headline… actually belongs inside [the flagship] as the data-link section"); the
  data card is **Tier 3 (component)** — "indispensable hygiene and your most citable single
  table; not a main-conference paper. Ship it early as a short paper or blog-plus-artifact."
- **Workshop-sized list: #1** (REC-a + REC-5): "Fastest by a wide margin… the headline
  finding… is *already in hand*… Zero outcome risk… Two to three weeks to a draft." Also
  **#7**, the realized-exposure audit + order-effect experiment: "Outcome-robust in a useful
  way: material drift is a confound finding about a widely used suite; no drift is a
  validation people will cite defensively."
- **Full-conference list: #2**, "What Is Actually in DataDecide" (REC-a + label correction +
  REC-5 + the realized-exposure/order-effect audit with the stratified-sampling
  intervention): "*Speed:* the core is mechanical and half-started; the only training is
  10–50M reruns with many seeds; no outcome risk anywhere. **Expected impact: medium-high**…
  **Ceiling: medium-high**; it caps out unless the order effect is large, in which case
  'small-scale data decisions are confounded with data order' becomes a genuinely disruptive
  claim." A scoop race.

### 2026-08-21 — position in the "four main-conference projects from two workshop subs" list (full list in `docs/portfolio-rankings.md`)

- **P2 — What's Actually in the Data: Composition, Order, and Small-Scale Validity.** Sub A:
  REC-a + label correction + REC-5. Sub B: the realized-exposure audit. Main paper: both plus
  "stratified vs. sequential sampling reruns at 10–50M with many seeds." "**Speed: second.**…
  Minimal outcome risk anywhere. **Scoop risk: medium** — the label correction is
  discoverable by anyone who looks, but nobody seems to be looking. **Expected impact:
  medium-high**… **Ceiling: medium-high**, jumping to high if the order effect is large."
  Recommendation: run sub A "in the background from week one — it's mechanical, half-done,
  and its manifest module is [the project's] foundation anyway."

### 2026-08-18 — origin of this project (from the Research Trajectory page)

**Danielle-flagged project seeds** (the `→` notes on the Notion toggle):
1. "Predict performance differences from dataset features."
2. "Does merging-as-annealing-proxy work on cosine mid-run checkpoints rather than just
   stable-phase ones?"
3. "Does a dataset's 'determinism profile' predict landscape geometry?"

Seed 1 is REC-3; seed 3 is REC-2 → REC-4. Question posed: how does the field currently
quantify differences between datasets? They become approximately black boxes because they
are so large — but we analyze trained models, which actually are black boxes, so there must
be things we can do for datasets as well.

- The three feature families as first laid out — model-mediated (perplexity correlations,
  RegMix, DoReMi, mixing laws: "don't tell you *what property* of the data mattered"),
  similarity embeddings (Task2Vec alignment, diversity coefficient; "data similarity alone
  is not enough"), intrinsic statistics (WIMBD, compression, Zipf/burstiness: "closest to
  your instinct, least developed") — in `docs/topics/reference/data-featurization-literature.md`.
- The causal anchor for intrinsic features: Zipf/burstiness/type-token statistics "are
  exactly the properties Chan et al. showed *cause* ICL emergence in small transformers…
  the one place intrinsic data statistics have been causally tied to a capability."
- The supervised-problem framing as first stated: "DataDecide hands you a supervised problem
  — 25 corpora with measured outcomes (and ~300 pairwise decisions, plus per-task
  breakdowns). No one has systematically featurized those 25 corpora… and asked which
  features predict the outcome table, or whether intrinsic features match model-mediated
  ones."
- REC-4's motivation as first stated: "a dataset's 'determinism profile' (cheap to estimate
  with any reference model) is a candidate feature predicting not just performance but
  *landscape geometry*, i.e., annealing behavior. That would tie your WSD-branch suite and
  your featurization question into one design."

### 2026-08-18 — origin of REC-2's design (from the Research Trajectory page)

Question posed: for the deterministic axis, can it be measured across datasets (e.g. percent
deterministic tokens)? Has it been done?

- "Score every token with a strong reference model's conditional entropy (or an ensemble, to
  separate the aleatoric floor from the reference model's own ignorance), and characterize a
  corpus by its distribution of per-token entropy — from which '% deterministic tokens'
  falls out as a threshold statistic." Existing machinery: DoReMi's excess loss and Rho-1's
  reference-model excess-loss scoring are "per-token *epistemic* measurements… the machinery
  is exactly what you'd need, pointed at a different goal." The "code has far lower
  conditional entropy than web text" folk fact is "a two-dataset determinism comparison,
  made informally." Wen et al. computed token uncertainty within one distribution, "never
  comparatively across corpora."
- "So the specific study — determinism profiles of, say, the 25 DataDecide corpora,
  correlated with valley geometry, decay-responsiveness, and annealing behavior — is open,
  and it's cheap: it's inference-only over corpus samples."
- The two design cautions that became REC-2's curve-not-scalar and pre/post-dedup
  requirements: "determinism is *relative* — to the reference model's capacity and to
  context length (a token deterministic given 2k context may be uncertain given 128) — so
  report profiles as curves over entropy thresholds and context lengths rather than a
  single percentage; and dedup interacts with it, since repeated text is trivially
  deterministic, so you'd want the profile computed before and after dedup to separate
  'structured domain' from 'duplicated corpus.'"

### 2026-08-22 — the validation report and a coverage/abnormality ledger as data-card components

Danielle had an agent reproduce the DataDecide paper's claims from the processed tables
(`docs/paper-validation-report.md` on `main`: 27 reproduced + 3 approximately reproduced
claim records, with the distinctions claim-record vs. independent discovery, strict vs.
approximate thresholds, and "0.02 seed SD occurs for some recipes" vs. "global maximum").
Two additions to the data-card scope from the response: (1) the claim-by-claim
validation report is a first-class component — which published claims reproduce from
the cleaned tables, with operationalizations pinned — and it de-risks every downstream
analysis (they run on tables that reproduced the headline 0.8033 150M→1B result). The
report should distinguish "claim reproduces" from "claim's operationalization is
informative" (the crossover count is the example). (2) Danielle's "there are definitely
some dataset abnormalities, like 750M only has 1 seed that trains fully I think"
(unverified; the 750M aggregate-table truncation is already in
`../open-questions-answered.md`) → an automated **coverage and abnormality ledger**
(recipe × size × seed × step cells present, early-terminated, known-issue), with every
downstream analysis declaring exclusion rules against it, and published numbers whose
support runs through thin cells flagged. Provenance list now: labels≠token shares,
nominal-vs-exact compute, unrecoverable LR, possibly-absent training loss, incomplete
seed replication. Candidate program framing sentence from the response: the original
paper's statistics are computed without a noise model and the portfolio recomputes them
with one — "DataDecide with error bars."

### 2026-08-22 — own-mixture held-out CE as a reconstructed training-loss analog

Follow-on from the same conversation: for each recipe, hold out a sample of its own
mixture drawn via the REC-a manifest/sampler and forward-pass the released checkpoints
over it. This gives an own-mixture held-out cross-entropy at checkpoint cadence — the
closest well-defined analog of training loss (minus batch noise and the moving-mixture
confound) — and as a by-product the cross-loss matrix (every recipe's model on every
recipe's mixture) that REC's similarity features want. Candidate fourth provenance-ledger
entry: training loss is absent from the released artifacts except sparsely at 150M–1B in
the scaling-law ladder CSVs; whether the authors could supply more is unconfirmed
(Danielle is checking). The response's broader thesis candidate: "DataDecide is an eval
suite being used as a training-dynamics suite; here is what it takes to make that valid."

### 2026-08-22 — The data-card thesis as a pattern of three divergences

From a conversation reviewing the `datadec` repository state (record in
`../topics/reference/datadecide-data-pipeline.md`). The data-card / composition paper
this doc's REC-a feeds has three independent, already-found cases where the suite's
self-description and its ground truth diverge: (1) mixture labels are shard-file
fractions, not token shares (this doc, §1); (2) the raw scaling-law exports encode
nominal-parameter rather than exact-parameter compute (caught by
`verify_preprocessed_derivations.py`); (3) learning-rate schedules are not recoverable
from any published artifact — Danielle's derivations come from the OLMo repo, issues,
Drive docs, and the paper, with the authors unable to confirm details of the sweep. The
response's framing: "the pattern is the paper," and each downstream analysis paper cites
the data card for its cleaned inputs. Action it implies for REC: write the LR-provenance
narrative into the data-card outline now, while the search trail is reconstructible.
Coverage fact settled the same day: OLMES detail tables are processed and published
(private HF dataset) for all 25 recipes.

### Undated (~2026) — Danielle's framing of the "datasets are unknowable" objection, and the retrieval / perturbation answer

Her statement (verbatim, from speech; external conversation, intake 2026-08-22):

> In a somewhat tangential direction, but related to the data-to-side dataset itself, one
> of the things that I thought was really cool about the data-to-side dataset is that it
> might provide a way to get some, what's it called, some quantitative feedback on the
> impact of different datasets that are used for pretraining. And I initially was really
> excited about this, and then the response from everyone was basically, it's impossible to
> understand what's inside the data sets, and so you can't make any claims about
> differences between data sets, and so you can't make any claims about how differences
> between datasets impact things. And so there's really no future in being able to
> investigate any questions around things like this. And I see that point. I think that's
> fair. But like many problems where people say this is impossible, it just kind of sits in
> my head and pops back up periodically because it just seems like it shouldn't be
> impossible, because what's impossible, really? And so one of the questions that I have is
> whether people try to analyze these huge datasets in terms of their match to specific
> corpora or tasks by doing some type of retrieval from the dataset on things that are
> relevant to that task, basically. So like the most direct form of retrieval would be some
> form of exact string match to questions or code or a corpora or something like that. But
> I think something that would still perhaps give you interesting aggregate information
> about the dataset would be all kinds of different ways of doing that, whether it's like
> querying by entity or embedding it and then doing similarity search or chunking your
> task space into things and querying that way, querying by question, querying by relevant
> documents for your task, things like that, to get some sense of what, not just what the
> metadata tags are for the different components of these datasets, but actually when you
> perturb it in some way, what is the outcome and then what do the outcome of a sequence
> of perturbations tell you about the thing itself?

("data-to-side" is the transcription of "DataDecide.") This is the origin statement for
the project's *task-conditioned* feature family: rather than describing a corpus by
metadata tags, **query it from the task side** — exact string match against questions /
code / corpora, entity queries, embedding similarity search, queries over a chunked task
space, by question, by task-relevant documents — and treat the dataset as something you
learn about by **perturbing it and reading the outcome sequence**. The objection it
answers is the one that ended the earlier enthusiasm: "you can't know what's inside, so
you can't make claims about dataset differences."

The response's method families (condensed; no citations given, so unverified as a survey):
(1) *direct corpus querying* — exact / near-duplicate search (n-gram, MinHash/SimHash,
Bloom filters, sketches) for leakage and memorization; entity- and template-based
retrieval (package / function names, error strings, `groupby(`) for domain coverage and
long-tail presence; embedding retrieval of task items against the corpus with aggregate
similarity distributions, cluster coverage, topical density ("semantic proximity mass");
defensible claims are density / near-duplicate-count / enrichment claims, not "X% of the
task is in the data." (2) *Corpus profiling without retrieval* — token and language
statistics, code/text ratio, library-mention counts, topic clustering on samples, dataset
maps via reference-model loss. (3) *Behavioral attribution* — A/B proxy training on
controlled slices, with DataDecide's recipes as natural experiments; per-example /
per-slice influence (TracIn-style, representer points) given checkpoints; membership
inference / extraction tests ("does the model behave as if it saw it"). (4) *Perturbation
sequences as system identification* — a task probe set × a set of dataset perturbations
(remove / add / reweight / swap slices) → outcome vectors → "task T is sensitive to slice S,
not T"; where training data is fixed, proxy perturbations via tiny probe models on sampled
subsets, or elicitation wrappers as the perturbation with the model as the instrument.
Claim types it says are falsifiable without omniscience: coverage, sensitivity,
attribution. Low-budget plan: tight auto-graded task suite → task query set (entities,
templates, paraphrases, doc snippets) → per-recipe sample of 1–10M tokens → retrieval /
proximity statistics → tiny proxy models on the samples → correlate proxy deltas with
proximity statistics ("predict task lift from measurable dataset signals").

Intake note: families (1)–(3) are already in this doc's §1 and in
`../topics/reference/data-featurization-literature.md` (model-mediated, similarity,
intrinsic families; WIMBD-style querying; perplexity correlations; influence). What this
conversation adds is (a) Danielle's verbatim statement of the objection and why she does
not accept it — useful in the paper's motivation, since the retrieval-side features are
the direct rebuttal; (b) the explicit **task-side query set** as a feature-construction
recipe (entities, templates, paraphrased questions, relevant documents, chunked task
space), which makes REC's similarity features concrete for the OLMES tasks; and (c) the
**perturbation-sequence / system-identification** framing, which is REC-2-style
leave-slice-out attribution restated as an active design — DataDecide's 25 recipes *are*
the perturbation set, and the outcome vector over sizes × tasks is the readout. None of
the response's claims are sourced; the relevant literature is in the reference topic.

# IRT reanalysis — a psychometric reanalysis of DataDecide

**Program pillars served:** how (a calibrated ability instrument), apex (measurement net of elicitation; emergence as measurement). (Program: `README.md` → Program.)

**One-line pitch.** DataDecide's per-instance eval results form a matrix of
(model × checkpoint) rows by item columns, where the rows are *structured*
(recipe × scale × seed × step) rather than arbitrary models. Item response
theory, previously used only to compress benchmarks over diverse converged
models, becomes a measurement instrument: a lower-noise ability score, a
formal matched-ability recipe comparison, per-item emergence curves, and a
direct test of whether recipes differ along one axis or many.

**Compute tier.** T0 throughout. The response matrix already exists in this
repo (`data/processed/olmes-details/{recipe}/instances.parquet` for binary
correctness, `choices.parquet` for per-choice likelihoods).

## 1. What the project involves

### Core (required for any paper)

- **Build the response matrix.** Rows keyed by (recipe, scale, seed, step);
  columns by (task, item). Two variants: binary correctness, and continuous
  likelihood margin (correct-choice logprob minus best-incorrect, from
  `choices.parquet`).
- **IRT-1 — Dimensionality check.** Fit 1-factor and k-factor IRT (2PL / graded
  or continuous-response models); compare by held-out log-likelihood,
  eigenstructure of the tetrachoric/residual correlation matrix, and
  item-fit diagnostics. Outcomes:
  - One θ fits well → recipes at these scales differ mostly along a single
    axis; matched-loss ≈ matched-everything; deflates the "beyond final
    performance" hypothesis at this scale. *Still a result.*
  - Multiple dimensions needed → the factor structure is the answer to "what
    do recipes change besides final performance."
- **IRT-5 — Binary vs. margin response model.** Fit both; compare item
  discrimination estimates, θ precision (standard errors), and trajectory
  smoothness. Replicates Signal-and-Noise's "continuous metrics carry more
  signal" finding inside one framework, and decides the response model for
  the rest of the paper.

### Optional directions

- **IRT-2 — θ(t) as a movement metric.** Compare signal-to-noise of θ
  trajectories vs. raw accuracy trajectories (seed-to-seed variance vs.
  between-recipe spread, both pooled across recipes). IRT estimates the item
  weights that Signal-and-Noise's subtask filtering sets to 0/1.
- **IRT-3 — Recipe-DIF.** Differential item functioning with recipe as group:
  items whose characteristic curves differ across recipes at matched θ.
  Standard DIF tests (Mantel–Haenszel, logistic regression DIF, or
  multi-group IRT with anchor items). This is the psychometric statement of
  "pretraining data shapes models beyond final performance."
- **IRT-4 — Per-item emergence.** Item characteristic curves plotted against
  compute (FLOPs already on the checkpoint rows) rather than θ: the compute
  at which each item crosses 50% gives a *distribution* of emergence points
  instead of a benchmark-level claim.
- **IRT-6 — Local-independence diagnostics.** Residual dependence flags
  shared-passage items and contamination; useful item-filtering byproduct.
- **IRT-7 — Explain DIF items.** Cluster recipe-DIF items by task/domain. (The
  token-determinism clustering from the synthesis needs a reference model;
  out of scope for this T0 project unless it is already built elsewhere.)
- **IRT-8 — BoolQ autopsy: hard or broken as measurement?** Separate "universally
  too hard" from "prior-tracking / response style" with the variance structure rather
  than item parameters: residual inter-item dependence within checkpoints (IRT-6
  machinery), regression of per-checkpoint accuracy on predicted-yes fraction, and
  whether margins are structured by label rather than content. Addendum: test whether
  BoolQ's nontrivial *decision* accuracy survives controlling for prior alignment — a
  benchmark that is predictable while measuring nothing.
- **IRT-9 — Margin decomposition.** Using per-choice likelihoods in `choices.parquet`,
  plot correct-prob and best-incorrect-prob trajectories separately per task to test the
  mechanism for margin's negative correlation with accuracy on strong-distractor tasks;
  establishes per-character Normalized Correct Probability as the continuous response
  for IRT-5 and margin as an object of study.
- **IRT-10 — BoolQ format intervention.** Re-score BoolQ on a checkpoint subset under
  alternative formats (cloze vs. MCQ presentation, label-balanced subsets, flipped label
  order; forward passes only) to split small-scale failure into format artifact vs.
  genuine difficulty. First concrete instance of the elicitation thesis (cross-listed in
  `elicitation-gain.md`).

## 2. Doability and impact

**Doability: high.** Inputs exist, methods are mature (`py-irt`, `girth`, or
a small PyTorch/NumPyro 2PL; `mirt`-equivalent via `statsmodels` is absent
but a custom EM/VI fit is a few hundred lines). Main engineering risks:

- Matrix size: hundreds of (model, checkpoint) rows × tens of thousands of
  items. Fine for marginal-ML or VI fits; avoid MCMC for the full matrix.
- Coverage: confirm per-instance details exist for all recipe × scale × seed
  cells, not a subset. Partial coverage bounds IRT-3 and IRT-4.
- Many items near chance at small scales give flat ICCs and poorly identified
  parameters; item filtering (IRT-6) must be principled and reported.

**Impact per direction (workshop-paper lens):**

| Direction | Impact | Why |
|-----------|--------|-----|
| IRT-1 dimensionality | **High** | Either outcome is a headline claim about what pretraining recipes change; no one has asked it of a controlled suite. |
| IRT-5 binary vs. margin | **Medium-high** | Strong methodological result; replicates a known finding in a new framework, which reviewers like but do not find surprising. |
| IRT-2 θ as movement metric | **Medium-high** | Practical payoff ("use θ, not accuracy, at small scale"); depends on the SNR gap being large. |
| IRT-3 recipe-DIF | **High if positive** | The cleanest public-data demonstration of the thesis; risk that DIF is sparse or dominated by contamination artifacts at these scales. |
| IRT-4 per-item emergence | **Medium** | Nice figure and connects to proxy-metric literature; less novel on its own. |
| IRT-6 diagnostics | Low (supporting) | Necessary hygiene, not a result. |
| IRT-7 DIF clustering | Medium | Depends on IRT-3; domain clustering alone is modest. |
| IRT-8 BoolQ autopsy | High | Case study that makes the psychometrics concrete; reviewers' "too hard" reading is testable and plausibly wrong. |
| IRT-9 margin decomposition | Medium–High | Cheap, decisive, self-contained metric finding; fixes the response-model choice. |
| IRT-10 format intervention | Medium | Turns the autopsy into an intervention; T1 forward passes. |

**Likely paper shape.** IRT-1 + IRT-5 + IRT-2 as the core ("a psychometric
reanalysis of DataDecide"), IRT-3 as the headline if it lands, IRT-4 as a figure.
A defensible workshop paper exists even if IRT-3 is null, provided IRT-1 is
reported honestly in either direction.

## 3. Infrastructure sequence

1. **Response-matrix builder.** Load per-recipe `instances.parquet` and
   `choices.parquet`; produce two wide matrices (binary, margin) plus a
   row-metadata table (recipe, scale, seed, step, tokens, FLOPs,
   `lr_at_step`) and an item-metadata table (task, item id). Cache as
   parquet. Verify coverage per cell and report gaps.
2. **IRT fitting module.** 2PL binary and a continuous-response model on
   margins; 1-factor and k-factor; marginal-ML or VI; returns item
   parameters, θ with standard errors, fit statistics, residual matrix.
   Unit tests on synthetic matrices with known parameters.
3. **Model-comparison + diagnostics.** Held-out likelihood, dimensionality
   statistics, item-fit and local-dependence flags (IRT-1, IRT-6).
4. **θ trajectories + SNR.** Join θ back to row metadata; compute pooled
   seed variance vs. recipe spread for θ and for accuracy (IRT-2).
5. **DIF module.** Multi-group fits with anchor items or MH/logistic DIF per
   item, with multiple-comparison control (IRT-3).
6. **ICC-vs-compute plotting and emergence extraction** (IRT-4).
7. **Report notebook** assembling figures; everything above is T0 and
   re-runnable from cached matrices.

Steps 1–3 are the minimum for the core paper; 4–6 are independent of each
other and can be picked by impact.

---

## 4. External assessments

Dated, attributed notes from external review conversations, recorded for consolidation — not
decisions. Only notes about this project are kept here.

### 2026-08-22 — scoring item embeddings against IRT difficulty without a trained predictor

Danielle's framing (from a separate conversation; record in
`../topics/reference/estimation-and-calibration-methods.md`, second entry): once the
response matrix gives per-item difficulty, compare several ways of *embedding the items*
by how much difficulty structure each recovers — with correlation-type metrics, not a
trained predictor, because the object of interest is the embedding approach. Metric stack
proposed: cluster R² / η² (with adjusted R², same clustering and cluster-count rule across
embeddings); NMI / V-measure / ARI if difficulty is binned; and, preferred because they
skip the clustering step, kNN difficulty smoothness relative to a shuffled-label baseline
and Spearman between pairwise embedding distance and pairwise difficulty difference. This
generalizes IRT-7 (cluster DIF items by task/domain) into "which item representation
explains difficulty and DIF" — a candidate named direction once IRT-1 exists, not added as
an ID yet. Caveat from intake: the shuffle null should respect benchmark block structure,
and pairwise correlations need permutation p-values.

### 2026-08-22 — frontier baseline from the literature: token-level proxies over expert trajectories

From Danielle's SciSpace literature review on small-scale evaluation metrics
(record in `../topics/reference/small-scale-evaluation-metrics-literature.md`; arXiv 2605.18607, Patel, Reddy, Mosbach & Bahdanau — Mila/ServiceNow, not AI2 as one review version claimed). Patel et al. 2026 rank models with a RankSVM over 80 token-level proxies
computed on expert solutions (Spearman 0.81 vs. 0.36 for cross-entropy; 0.33 for
rBridge) and rank DataDecide corpora at 10⁻⁵ of target compute. For the
decision-reliability frontier this is the strongest published proxy to beat alongside
the reproduced DataDecide continuous metrics: the question becomes "how much further
does θ-based measurement push past proxy-over-expert-trajectory", and the honest
comparison needs their proxies reimplemented on the instance tables (their statistics
are per-token; `choices.parquet` has per-choice likelihoods, which is coarser — note
the mismatch). Also relevant to the response-model choice: their finding that
entropy- and frequency-weighted variants dominate echoes the per-character
normalization result here. Unverified beyond the agent summaries.

### 2026-08-22 — margin demoted, the metric hierarchy, the two-cluster null, and a BoolQ twist

*Provenance caveat: the reproduction numbers cited here come from agent-written verification code that Danielle has not yet personally read, debugged, run, or analyzed; treat them as flags for where to look first, not as findings (her statement in `../topics/reference/datadecide-data-pipeline.md`).*

From the "directionally consistent" and "not reproduced" batches of Danielle's
reproduction of the DataDecide paper (numbers as reported in
`../topics/reference/datadecide-data-pipeline.md`; not re-checked here).

- **Continuous response variable.** Margin (correct-choice logprob minus best incorrect)
  was the penciled-in continuous response for the IRT fits. The reproduction finds
  margin tracks accuracy at mean ρ 0.360 (negative on ARC Challenge, PIQA, WinoGrande)
  while Normalized Correct Probability tracks at 0.916, and per-character normalization
  won 9/10 tasks earlier. Design consequence: **per-character Normalized Correct
  Probability is the primary continuous response; margin becomes an object of study.**
  Proposed mechanism for the negative correlations: as models improve, all choices get
  more probable, and plausible-by-construction distractors can rise faster than the
  correct answer even while the ranking improves — the three negative tasks are the
  near-minimal-pair / strong-distractor ones. Decisive cheap test: `choices.parquet`
  has per-choice likelihoods, so decompose margin into correct-prob and
  best-incorrect-prob trajectories per task; if best-incorrect rises faster on the
  negative tasks, "margin is an anti-signal on plausible-distractor tasks; use
  normalized correct probability" is a self-contained finding (links to the
  metric-functional-form / emergence-mirage line, unverified).
- **Metric hierarchy from the pipeline.** Raw Correct/Total Prob "dominates at most
  small scales" failed at 2.38% vs. a >50% predicate, robustly across compute bands
  (3.30% / 2.38% / 1.77% up to 0.1% / 1% / 10%); the raw-plateau-penalized-converge
  claim failed on both halves (max raw slope 0.0914; gap grew 0.023 → 0.134). Together
  with normalized metrics winning 816/830 near target: length-normalized correct
  probability everywhere, at every scale; raw likelihood and margin each have
  documentable failure modes. This is the frontier sub's empirical foundation, and #6's
  diverging gap says metric choice matters *more* with compute on those tasks — the
  response-model comparison is a permanent methodological question, not a small-scale
  workaround. Caveat before any public framing: the three strong failures live in
  proxy-metric territory where a definitional mismatch (what "Total Prob" is, normalized
  how, over which span) could manufacture divergence; a definition-matching pass
  against the paper's released analysis code is required first.
- **Two task clusters (silhouette 0.207 vs. 0.25 default; reproduced at 0.15).** A sharper
  null than either the paper or the validator used: two proxy-curve *shape* clusters do
  not imply two ability dimensions. A strictly one-dimensional model produces
  qualitatively different aggregate shapes if tasks differ in item-difficulty
  distributions — difficulty mass near the small-model θ range gives early smooth rises
  (Group A: ARC Easy, BoolQ, CSQA, PIQA, SocialIQA), mass above the range gives
  flat-then-rise (Group B: ARC Challenge, HellaSwag, MMLU, OpenBookQA, WinoGrande). The
  dimensionality test (IRT-2-style) is the instrument: fit one factor, ask whether the
  A/B split is reconstructable from difficulty distributions alone or whether a second
  factor's loadings recover it. Either outcome is clean (deflationary or confirmatory).
  BoolQ landing in the "easy-shaped" cluster while sitting at noisy chance is a further
  hint its curve shape is prior-tracking artifact.
- **BoolQ twist.** The "nontrivial only at intermediate 1B checkpoints" claim strongly
  failed: 108 nontrivial (>0.55) decision-accuracy points, 85 below 1B across eight
  sizes, final 1B at 0.7867. Predictive signal without measurement validity is what
  prior-tracking predicts: if a recipe's data statistics induce a scale-stable
  yes-prior, small-scale BoolQ decisions predict 1B BoolQ decisions through a channel
  that never touches comprehension. Addendum to the autopsy: regress BoolQ decision
  accuracy on prior-alignment features (predicted-yes fraction) and see whether the
  predictive signal survives. If not, the benchmark is simultaneously "predictable" and
  measuring nothing — the sharpest illustration that decision accuracy alone is an
  insufficient validity criterion.
- Motivation-section pattern across all batches: every ambiguous or failed reproduction
  traces to a metric with a bad functional form (margin), a statistic without a noise
  model (crossovers, seed-SD phrasing), or a threshold without principled basis
  (silhouette, 0.90 cutoffs) — the three things likelihoods, error bars, and model
  comparison replace.

### 2026-08-22 — BoolQ as the diagnostic case study; noise-aware crossings; the frontier's design brief

*Provenance caveat: the reproduction numbers cited here come from agent-written verification code that Danielle has not yet personally read, debugged, run, or analyzed; treat them as flags for where to look first, not as findings (her statement in `../topics/reference/datadecide-data-pipeline.md`).*

Prompted by Danielle's claim-by-claim reproduction of the DataDecide paper on the
processed tables (`docs/paper-validation-report.md` on `main`; the summary she pasted is
recorded in `../topics/reference/datadecide-data-pipeline.md`). Her question, verbatim:

> boolq is basically always sitting at random noise and has VERY high variance. and it
> makes me wonder whether its really so hard or whether something about the task
> formatting, etc is adversarial especially to small models. is that a question that fits
> somewhere in our 4 project design?

And her follow-up:

> would the "Broken as measurement" result also come from the task just being universally
> too hard for this scale of models? because thats what reviewers have all concluded so
> its unclear that IRT would distinguish this?

**BoolQ autopsy (fits IRT-3/IRT-7 and the diagnostics module).** Two hypotheses leave
different fingerprints. Genuinely hard: high item difficulty, normal discrimination, θ
must climb far before accuracy moves, margins drift smoothly sub-threshold. Broken as
measurement: discrimination near zero *and* large local-independence violations — within
a checkpoint, responses correlate across items beyond what θ explains. The correct
answer to the follow-up: item parameters alone do **not** separate the two (a
universally-too-hard task also gives near-zero discrimination); the **aggregate variance
structure** does. Independent guessing at chance over ~3,200 items gives a seed SD of
roughly 0.008–0.01; the observed SD up to 0.111 is >10× that floor and is only producible
if responses are strongly correlated within a checkpoint — the model answers whole
swaths the same way, and which way it leans swings across seeds and steps. BoolQ is
two-choice with an imbalanced (yes-heavy, ~60%+, unverified) label distribution, so
"chance" is ambiguous between 50% and the majority-class rate, and prior-tracking
(response style / acquiescence bias) predicts exactly the observed phenomenology; "too
hard" predicts a flat, quiet 50%. Discriminating tests: (a) residual inter-item
correlation / local-dependence structure; (b) regress per-checkpoint accuracy on the
model's predicted-yes fraction — if accuracy variance is explained by answer-base-rate
variance, the task measures the yes-prior; (c) whether margins are structured by label
rather than content. Residual caveat: "too hard" and "prior-tracks *because* too hard"
can both be true; a **format intervention** (cloze vs. MCQ presentation, label-balanced
subsets, flipped label order, on a checkpoint subset; forward passes only) separates
"irrecoverable at this scale" from "recoverable under better elicitation." Constructive
corollary either way: does the margin metric on BoolQ retain signal that accuracy
destroys, or should BoolQ be dropped from small-scale suites? Both are citable
recommendations; the generalization is "a principled diagnostic for when a benchmark
measures response style rather than ability." The format study is the first concrete
instance of the elicitation thesis (apparent capability floors that are measurement
floors; see `elicitation-gain.md`).

**Crossings.** The reproduction counted 15,523 crossovers with all 300 recipe pairs
crossing at least once. Danielle: her bump plots showed the ordering "super super super
consistent however you slice it … basically two lines that are the same and are just
jittery." The response agreed her bump plot is the better evidence: a crossing statistic
with no persistence or magnitude threshold is near-guaranteed under stable ordering +
noise, and ~50 crossings per pair is the signature of jitter, not regime change.
Noise-aware definition to adopt: a *meaningful* crossing exceeds the per-task seed-noise
floor and persists for k consecutive checkpoints (in drift/diffusion terms, attributable
to drift, not diffusion — see `trajectory-statistics.md`); recount, then ask what
annealing does to the survivors. Expected outcome: the count collapses to dozens in
genuinely close pairs; "the paper's crossover claim is technically true but mostly
measurement noise" is a data-card / IRT finding. Crossover density vs. compute is a
companion statistic to the decision-reliability frontier.

**Frontier design brief (from Danielle's ≤1%-compute skepticism).** Her point:

> I'm a bit skeptical about "<= 1% compute" metrics because most of the model sizes don't
> provide anywhere near that level of granularity if we're normalizing within size, and if
> we're normalizing by 1B compute full training then that seems strange.

Response: normalizing by target (1B-final) compute is defensible for the decision framing
(a fully-trained 4M model is ~0.01% of a 1B run), but the low-compute region is populated
by very few (size, step) cells, so "at 0.009% of compute" means "at the one or two cells
near 0.009%"; iso-compute points conflate a fully-trained tiny model with an early-stopped
larger one; and "the *best* continuous proxy achieved 0.874" is a post-hoc max over a
metric family with winner's-curse inflation. The qualitative conclusion (continuous ≫
accuracy at low compute; the 816/830 pairwise result) survives; the extreme-low-compute
numbers are fragile. Brief for the frontier sub: report iso-compute bands with cell
composition explicit; separate the size-ladder axis from the within-run axis rather than
collapsing to one compute scalar; evaluate metric selection out-of-cell. The reproduced
proxy numbers are the incumbent baselines the frontier must beat ("how much further does
θ push past the best continuous proxy"); per-character normalization winning 9/10 tasks is
a design input for the continuous-response variant (operate on per-character-normalized
margins, or ablate the normalization family). Noise floors must be per-task-per-recipe
objects, not global scalars (0.02 typical, 0.111 max, BoolQ-driven).

### 2026-08-21 — two "top-N by workshop-paper likelihood × speed" lists

- **Ranked #2 in a top-3 list.** "Also pure T0, and IRT-1 (dimensionality) is the rare
  design where both outcomes are headline claims: 'one axis suffices' deflates the
  beyond-final-performance thesis; 'multiple axes needed' is the thesis. IRT-5 gives you a
  guaranteed medium-strength methodological result as a floor. The methods are mature and the
  engineering risk (matrix size, near-chance items) is well-understood. Main caveat: verify
  per-instance coverage across all cells before starting."
- **Ranked #1 in a top-5 list.** "IRT-1 + IRT-5, IRT-2 optional. Pure T0, data already parsed,
  mature methods, and — critically — IRT-1 is a headline in either direction ('recipes are
  one-dimensional at these scales' is as publishable as 'here's the factor structure'). Fastest
  path to a paper with no outcome risk. Main real work is the response-matrix builder and a VI
  2PL fit."
- Per-instance coverage gate: resolved — all 25 recipes × 66 tasks, 3 seeds at 150M–1B, 1 seed
  below 150M (see `docs/open-questions-answered.md`).

- **Overlap.** "IRT's recipe-DIF (IRT-3) and drift/diffusion's matched-loss
  signatures are close enough that reviewers will ask why you need both" — decide which
  instrument makes the beyond-final-performance claim first.
- **Nulls.** IRT-1's null is "a genuine substantive claim"; a null IRT-3 at 150M
  "is ambiguous between 'recipes are one-dimensional' and 'these scales are too small to see
  it,' and reviewers will pick the boring interpretation."
- **n = 25.** IRT-3 "currently treat[s] 25 recipes as exchangeable"; prefer
  within-family comparisons along a measured dose (the family-contrast framing from recipe
  featurization).

### 2026-08-21 — on a full-conference ("strong") version

Question posed: is there a path to turn IRT-on-datasets into an acceptance-worthy full
conference paper — if not in NLP, then in a continual-learning / plasticity space where the
pitch is rigor and science rather than large-lab adoption?

- **Precedent and pattern.** "IRT-for-NLP-evaluation papers have gotten into main venues
  before (Rodriguez et al.'s leaderboard IRT at ACL 2021, tinyBenchmarks at ICML 2024). The
  pattern in every accepted one: IRT plus a claim or payoff, never IRT as reanalysis."
  (Unverified attributions.)
- **Claim 1 — "emergence is a measurement phenomenon."** "Schaeffer et al.'s 'mirage' paper
  showed metric choice manufactures discontinuities, but it lacked a principled framework —
  it swapped metrics ad hoc. IRT is the discipline that has spent 70 years on exactly this
  problem: latent ability θ, item difficulty distributions, and link functions jointly
  determine when smooth latent growth produces discontinuous observed scores. The paper: fit
  longitudinal IRT to checkpoint sequences (DataDecide, Pythia, OLMo), show θ(compute) is
  smooth where accuracy jumps, and decompose each claimed emergent ability into
  latent-growth vs. item-threshold-distribution components, with per-item emergence points
  (IRT-4) as the mechanism figure. That's a headline claim about a topic main venues actively
  care about, with the rigor pitch as the differentiator rather than the whole pitch."
- **Claim 2 — the continual-learning / plasticity pivot** ("genuinely good and underrated").
  "CL's measurement situation is a mess — forgetting metrics conflate item difficulty with
  ability change, task orderings aren't on a common scale, and the community knows this. IRT
  gives you: forgetting as θ decline on a vertically-scaled common metric; DIF-over-time as a
  localization tool for which items a new task interferes with (interference structure, not
  just a scalar); and measurement invariance testing as the formal criterion for 'is this
  benchmark measuring the same thing before and after training on B.' CoLLAs and TMLR would
  take this readily, and it's plausibly strong enough for a main venue because it fixes a
  recognized methodological hole rather than adding a method. Bonus: CL experiments are
  small-scale by nature, so the 'population of respondents' IRT needs (many models ×
  checkpoints) is cheap to generate."
- **Claim 3 — dimensionality / matched ability (IRT-1).** "Real but I'd rank it third for
  full-conference purposes — 'recipes differ along k axes' needs the DIF result [IRT-3] to
  land to be exciting, and that's your highest-variance component."
- **What upgrades workshop → full paper.** "Validation of the measurement model itself —
  invariance testing across training, simulation studies showing the decomposition recovers
  ground truth, robustness to the response-model choice (IRT-5). Psychometrics reviewers do
  this by default; ML papers using IRT mostly skip it. Doing it properly is both the rigor
  pitch and the extra 4 months of work that justifies the venue."

### 2026-08-21 — on tiny-scale evaluation as a derived artifact

- "Item difficulty/discrimination parameters tell you directly which items have any
  discriminating power in the 10–150M ability range (most don't — the effective test length
  of MMLU-style suites at 10M is close to zero), so 'an eval that works at tiny scale' is a
  *derived artifact* of the IRT fit: select items whose difficulty brackets the tiny-model θ
  range, score with likelihood margins, report θ with standard errors instead of accuracy."
- Candidate paper, "the decision-reliability frontier": "how far down does reliable decision
  signal survive, as a function of measurement method — decision accuracy vs. compute curves
  where the treatment is the measurement stack (accuracy vs. margins vs. θ vs. IRT-selected
  item subsets)." Mostly analysis over existing tables plus modest tiny-model evals. Note: the
  per-instance tables cover 4M/20M/60M/90M with one seed each (see
  `docs/open-questions-answered.md`).
- IRT as curriculum: "'IRT-scheduled RL for small models' — pick tasks whose difficulty puts
  the model's pass rate in the informative band, advance the ladder as θ moves." (Full
  discussion in `docs/potential-projs/tiny-scale-measurement.md`.)

### 2026-08-21 — positions in three ranked lists (full lists in `docs/portfolio-rankings.md`)

- **6–12-month flagship list: Tier 2, #4**: "The problem isn't quality… it's that this
  project doesn't scale with time. The 10-week version and the 10-month version are nearly
  the same paper… Reweighted: do it anyway in months 1–2 as the fast insurance paper… then
  let its dimensionality answer inform how you frame [the flagship's] recipe claims. As the
  flagship bet: no."
- **Workshop-sized list: #2**: "Pure T0… the safest bet in the portfolio on outcome… The only
  real work is the response-matrix builder and principled item filtering."
- **Full-conference list: #1**, "The Psychometrics of Pretraining Suites" (IRT-1 + IRT-5 +
  IRT-3 + the derived tiny-scale instrument): "*Fastest because:* entirely T0, every
  component is either-way publishable… if DIF is sparse, the measurement-instrument
  contribution carries the paper. Best venue fit: ACL/EMNLP or NeurIPS D&B. **Expected
  impact: medium-high**… **Ceiling: high** — if recipes are one-dimensional at these scales,
  'matched loss = matched everything' is a quotable negative result." Overlap warning with
  the drift/diffusion paper ("a cynical reviewer could ask why they aren't one paper — have
  the answer ready"); a scoop race.

### 2026-08-21 — position in the "four main-conference projects from two workshop subs" list (full list in `docs/portfolio-rankings.md`)

- **P1 — The Psychometrics of Pretraining Evaluation.** Sub A: IRT-1 + IRT-5. Sub B: the
  decision-reliability frontier. Main paper: "a measurement framework for checkpoint suites —
  latent structure, optimal response models, recipe-DIF [IRT-3], and the derived instrument
  for small-scale evaluation." "**Speed: fastest.**… **Scoop risk: medium-high**… speed is
  the defense. **Expected impact: medium-high. Ceiling: high.**" Recommended as **the primary
  starting effort**: "its outputs (θ, item parameters, noise-aware measurement) are inputs to
  [the data/order project's] intervention analysis and [the schedule audit's]
  flip-significance testing, and shipping it first blunts the scoop exposure where the race
  is tightest."

### 2026-08-18 — prior art for the emergence-as-measurement claim (from the Research Trajectory page)

- The loss→accuracy literature already notes the fragility IRT would formalize: "hard accuracy
  metrics can look emergent, showing no progress above chance until the loss crosses a
  threshold, which is where the loss-to-accuracy mapping gets fragile" (contrasting the
  exponential link of Gadre et al. 2024, the FLP pipeline of Chen et al., and Bhagia et al.'s
  model ladders). See `docs/topics/reference/loss-curve-forecasting.md`.

### 2026-08-18 — a control for matched-ability comparisons (from the Research Trajectory page)

- The matched-loss caution applies to matched-θ recipe comparisons (IRT-3) too: "equal loss
  at different token counts vs. equal tokens at different loss are different controls" —
  report DIF at matched θ both with and without conditioning on compute.

### 2026-08-18 — origin of this project (from the Research Trajectory page)

**Danielle-flagged project seeds** (the `→` notes on the Notion toggle):
1. Prior work "fit IRT to *diverse converged models* to compress benchmarks"; "the
   DataDecide setting adds is *structure in the model axis*… which converts IRT from a
   compression tool into a measurement instrument for your program."
2. "**Use 'ability' from IRT as your movement metric**: it should be substantially
   lower-noise than accuracy. Test Signal-to-Noise approach of 'switching to better-signal
   metrics and filtering noisy subtasks': is SNR(θ trajectories) > SNR(accuracy
   trajectories) on their own released data?"
3. "**Differential item functioning (DIF) is your matched-loss comparison**, formalized. A
   measure of whether 'items behave differently across model groups *at the same ability
   level*.' Use recipes as groups and then you have the psychometric statement of
   'pretraining data shapes models beyond final performance.' Items exhibiting recipe-DIF
   are the loci where recipes differ irreducibly — and the natural follow-up is whether DIF
   items cluster by your token-determinism buckets or by domain."
4. "**Item characteristic curves against compute give per-item emergence points.** Then
   emergence becomes a distribution providing a per-item version of the loss-to-accuracy
   mapping for the proxy-metric track."
5. "**IRT provides insight over latent dimensionality.** Classical IRT assumes one latent
   dimension; if a single θ fits the DataDecide matrix well, that's evidence recipes at
   these scales differ mostly along one axis (and matched-loss ≈ matched-everything,
   deflating part of your hypothesis at this scale!); if it demands multiple dimensions,
   the factor structure *is* the answer to 'what do recipes change besides final
   performance.'"
6. "Binary IRT discards the margin information that carries most small-scale signal, so fit
   both binary and continuous-response variants (on likelihood margins) and compare. **The
   comparison itself replicates Signal and Noise's metric-choice finding inside a single
   framework.**"

These map onto IRT-2 (seed 2), IRT-3 and IRT-7 (seed 3), IRT-4 (seed 4), IRT-1 (seed 5),
IRT-5 (seed 6). Question posed: the large set of evaluated models and tasks seems like the
ideal setting for IRT and other approaches that analyze benchmarks using a set of models'
predictions; DataDecide publishes per-task eval results and perplexity evals for a range
of corpora.

- "The setup is ideal: hundreds of model×checkpoint rows, thousands of item columns, binary
  (and margin) outcomes — precisely the data shape item response theory was built for."
  Prior art (Lalor; Rodriguez; tinyBenchmarks; metabench) compressed benchmarks over
  diverse converged models; see `docs/topics/reference/irt-literature.md`.
- "Dimensionality is the honest first check, and it's substantive rather than procedural…
  Either outcome is a result."
- Project shape as first stated: "another all-inference, public-data, weeks-scale artifact:
  'a psychometric reanalysis of DataDecide'… produces the calibrated instrument (θ
  trajectories, item bank with difficulty/discrimination/DIF flags) that every later stage
  would rather use than raw accuracy."

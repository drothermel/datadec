# IRT reanalysis — a psychometric reanalysis of DataDecide

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

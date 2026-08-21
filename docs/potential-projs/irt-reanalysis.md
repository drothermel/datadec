# Track B — A psychometric reanalysis of DataDecide (IRT)

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
- **B1 — Dimensionality check.** Fit 1-factor and k-factor IRT (2PL / graded
  or continuous-response models); compare by held-out log-likelihood,
  eigenstructure of the tetrachoric/residual correlation matrix, and
  item-fit diagnostics. Outcomes:
  - One θ fits well → recipes at these scales differ mostly along a single
    axis; matched-loss ≈ matched-everything; deflates the "beyond final
    performance" hypothesis at this scale. *Still a result.*
  - Multiple dimensions needed → the factor structure is the answer to "what
    do recipes change besides final performance."
- **B5 — Binary vs. margin response model.** Fit both; compare item
  discrimination estimates, θ precision (standard errors), and trajectory
  smoothness. Replicates Signal-and-Noise's "continuous metrics carry more
  signal" finding inside one framework, and decides the response model for
  the rest of the paper.

### Optional directions

- **B2 — θ(t) as a movement metric.** Compare signal-to-noise of θ
  trajectories vs. raw accuracy trajectories (seed-to-seed variance vs.
  between-recipe spread, both pooled across recipes). IRT estimates the item
  weights that Signal-and-Noise's subtask filtering sets to 0/1.
- **B3 — Recipe-DIF.** Differential item functioning with recipe as group:
  items whose characteristic curves differ across recipes at matched θ.
  Standard DIF tests (Mantel–Haenszel, logistic regression DIF, or
  multi-group IRT with anchor items). This is the psychometric statement of
  "pretraining data shapes models beyond final performance."
- **B4 — Per-item emergence.** Item characteristic curves plotted against
  compute (FLOPs already on the checkpoint rows) rather than θ: the compute
  at which each item crosses 50% gives a *distribution* of emergence points
  instead of a benchmark-level claim.
- **B6 — Local-independence diagnostics.** Residual dependence flags
  shared-passage items and contamination; useful item-filtering byproduct.
- **B7 — Explain DIF items.** Cluster recipe-DIF items by task/domain. (The
  token-determinism clustering from the synthesis needs a reference model;
  out of scope for this T0 project unless it is already built elsewhere.)

## 2. Doability and impact

**Doability: high.** Inputs exist, methods are mature (`py-irt`, `girth`, or
a small PyTorch/NumPyro 2PL; `mirt`-equivalent via `statsmodels` is absent
but a custom EM/VI fit is a few hundred lines). Main engineering risks:

- Matrix size: hundreds of (model, checkpoint) rows × tens of thousands of
  items. Fine for marginal-ML or VI fits; avoid MCMC for the full matrix.
- Coverage: confirm per-instance details exist for all recipe × scale × seed
  cells, not a subset. Partial coverage bounds B3 and B4.
- Many items near chance at small scales give flat ICCs and poorly identified
  parameters; item filtering (B6) must be principled and reported.

**Impact per direction (workshop-paper lens):**

| Direction | Impact | Why |
|-----------|--------|-----|
| B1 dimensionality | **High** | Either outcome is a headline claim about what pretraining recipes change; no one has asked it of a controlled suite. |
| B5 binary vs. margin | **Medium-high** | Strong methodological result; replicates a known finding in a new framework, which reviewers like but do not find surprising. |
| B2 θ as movement metric | **Medium-high** | Practical payoff ("use θ, not accuracy, at small scale"); depends on the SNR gap being large. |
| B3 recipe-DIF | **High if positive** | The cleanest public-data demonstration of the thesis; risk that DIF is sparse or dominated by contamination artifacts at these scales. |
| B4 per-item emergence | **Medium** | Nice figure and connects to proxy-metric literature; less novel on its own. |
| B6 diagnostics | Low (supporting) | Necessary hygiene, not a result. |
| B7 DIF clustering | Medium | Depends on B3; domain clustering alone is modest. |

**Likely paper shape.** B1 + B5 + B2 as the core ("a psychometric
reanalysis of DataDecide"), B3 as the headline if it lands, B4 as a figure.
A defensible workshop paper exists even if B3 is null, provided B1 is
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
   statistics, item-fit and local-dependence flags (B1, B6).
4. **θ trajectories + SNR.** Join θ back to row metadata; compute pooled
   seed variance vs. recipe spread for θ and for accuracy (B2).
5. **DIF module.** Multi-group fits with anchor items or MH/logistic DIF per
   item, with multiple-comparison control (B3).
6. **ICC-vs-compute plotting and emergence extraction** (B4).
7. **Report notebook** assembling figures; everything above is T0 and
   re-runnable from cached matrices.

Steps 1–3 are the minimum for the core paper; 4–6 are independent of each
other and can be picked by impact.

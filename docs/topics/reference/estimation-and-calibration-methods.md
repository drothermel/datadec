# Estimation and calibration methods — reference topic

**Kind:** reference (a methods toolkit, not a literature survey). Entries are dated and
condensed from conversations; formulas are standard, citations unverified unless marked.

Why it matters here: nearly every project estimates a per-cell quantity (a pass rate, a
ranking, a seed-noise floor, a forecast) from a fixed sample budget and wants honest
intervals. **Danielle's flag (2026-08-22): conformal prediction is a potential tool /
analysis method to keep in view across projects**, not only for the code-generation task
that prompted the conversation below. Program-level pointer in
`../../potential-projs/README.md`.

---

## Undated (intake 2026-08-22) — estimating per-(model, docstring) performance at fixed samples

**Context.** Danielle's task: LLMs write a function from a docstring + signature; performance
= fraction of predefined tests passed; temperature 0 still gives diverse outputs, so
multiple samples are needed. Question: how to improve accuracy of the estimate at a fixed
number of samples per (model, docstring) pair. Four turns (estimators → bootstrap →
non-bootstrap intervals → conformal prediction → conformal + cheap estimators across
docstrings/models). Danielle: "even though this is in the context of a specific task, I
think this whole line of questioning is super relevant to many of the projects."

### Estimand and estimator discipline

- Estimand: μ = E[S], S ∈ [0,1] the fraction of tests passed by one completion from a fixed
  (model, exact prompt, decoding config, model version/date). Changing any of those changes
  the distribution; sampling from another needs importance weights.
- With black-box samples and no side information the sample mean is already the right
  estimator; gains come only from (1) variance reduction, (2) side information,
  (3) better independence, (4) changing the estimand.
- **Use the fractional score, not all-pass** — binary all-pass has much higher variance
  when full success is rare. Record the whole per-test vector per completion, but the
  independent unit is the *program*, never the n×m individual test outcomes.
- **Deduplicate with multiplicities**: canonicalize (body extraction, AST-normalize, strip
  comments, optionally hash the test-pass vector), test each unique program once, weight by
  observed frequency. Same estimator, cheaper when tests are expensive, and the mode
  structure (few dominant modes vs. many low-probability ones) becomes visible.
- **Stratification** if logprobs or branch structure are available: strata by first token /
  first line / algorithm skeleton / parse category; μ = Σ p_h μ_h with n_h ∝ p_h σ_h. Strong
  when a few branches dominate with different pass rates.
- **Control variates** from cheap correlated signals (compile/type-check, visible-test
  score, lint, length, logprob, self-repair success): μ̂_CV = mean(S_i − β(X_i − μ_X)),
  β ≈ Cov(S,X)/Var(X). Only helps when μ_X is known from a much larger cheap sample;
  cross-fit, or it overfits on the same small n.
- **Bayesian shrinkage** with a prior fitted over many docstrings (Beta–Binomial for
  all-pass; beta / logit-normal hierarchical for fractional): trades unbiasedness for MSE;
  report as model-based.
- **pass@k**: use the unbiased estimator 1 − C(n−c,k)/C(n,k) (Codex paper, arXiv
  2107.03374), not the plug-in.
- **Temperature-0 samples may not be iid** — batching, backend routing, model version,
  nondeterministic kernels. Record version/fingerprint, don't mix snapshots, randomize
  request order across models, use block methods when collected in batches.
- **Paired designs** for comparisons (same docstrings, counts, seeds, tests,
  post-processing; compare per-docstring differences) — reduces variance of differences,
  not of absolute estimates.

### Interval methods and how they compare

Bootstrap is for intervals, not for a better point estimate. Resample *programs*, never
individual test outcomes; with dedup, resample by observed frequency. Percentile bootstrap
is fine at moderate n; at tiny n, or with many exact 0/1 scores, it looks precise while
missing unobserved modes (a 5% perfect-solution mode absent from 10 samples cannot be
recovered). Block bootstrap over API calls when batches may be correlated. For pass@k,
bootstrap the unbiased estimator.

| Method | Target | Best for | Limitation |
|---|---|---|---|
| Analytic SE / t-interval, σ̂/√n | μ̂ | plain means at n ≳ 30 | approximate normality |
| Bootstrap over completions | μ̂ | skewed / multimodal score distributions | unstable at small n, near 0/1; no coverage for unseen modes |
| Hoeffding, ±√(log(2/δ)/2n) | μ (bound) | distribution-free worst case on [0,1] | very conservative (±0.136 at n=100) |
| Empirical Bernstein | μ (bound) | variance-aware conservative bound | still conservative |
| Wilson / Jeffreys / Clopper–Pearson / Beta posterior | p (binary) | all-pass rates, small n, c near 0 or n | binary only |
| Bayesian posterior | belief about μ | small n with historical prior | prior/model dependent |
| Jackknife | μ̂ | smooth non-trivial estimators (pass@k, CV estimators) | adds nothing for a plain mean |
| Batch-level SE / cluster-robust | μ̂ | correlated completions, batched generation | needs enough batches |
| Confidence sequences (anytime-valid) | μ | "sample until the interval is narrow enough" | wider, more complex |

Reporting stack suggested: mean + analytic SE + bootstrap CI (+ empirical Bernstein if a
conservative bound is wanted); Wilson/Jeffreys for binary; batch-level SE or block bootstrap
when not iid. If bootstrap and analytic SE disagree a lot, that itself signals small n,
discreteness, or non-iid samples.

### Conformal prediction: a different target

- Bootstrap and the methods above give uncertainty in the **mean**; conformal prediction
  gives a **prediction interval for the next observation** (S_{n+1}) with finite-sample
  marginal coverage under exchangeability. For one fixed (model, docstring) with no
  predictor it reduces to finite-sample-corrected empirical quantiles — valid, often
  uninformative at small n.
- It becomes useful when there is a predictor to calibrate or many units to calibrate
  across: (A) next-completion prediction; (B) **calibrating cheap predictors of hidden
  performance** (visible tests, compile, lint, logprob, judge scores) with split conformal
  on residuals (Lei et al., arXiv 1604.04173); (C) **calibrating across docstrings** — a
  model predicts μ_d from features, conformalize |μ_d − μ̂_d|; the guarantee is marginal
  over future docstrings, not per-docstring.
- **Conformal risk control** (Angelopoulos et al., arXiv 2208.02814): calibrate a threshold
  so a decision rule (accept / reject / sample more / route to a stronger model / ask a
  human) has bounded expected loss — often more operationally useful than an interval.
- Exchangeability is the assumption to check: same model version, prompt, decoding,
  post-processing; batch effects need grouped/block conformal.

### Conformal + cheap estimators across docstrings and models (the design that was sketched)

1. **Unit and data.** Historical tuples (docstring, model, prompt, decoding, cheap features
   X, expensive target Y = high-sample μ or pass@k).
2. **Cheap features.** Reframe Danielle's "can a model reconstruct the original function
   from the docstring?" as **"does the docstring determine the behaviour the tests
   check?"** — exact reconstruction is too strict when many implementations are valid.
   Ask a judge for structured fields (ambiguity 0–5, missing edge cases, constraints /
   output format / error behaviour specified, examples present, number of plausible
   interpretations, predicted success probability, suggested clarifications) plus
   signature features, compile rate from 1–3 samples, weak/generated-test pass rate,
   sample diversity, model/prompt identity. The judge need only be *correlated*; conformal
   supplies the reliability.
3. **Split conformal regression** f̂(X) → μ with calibrated residual quantile →
   [f̂ − q, f̂ + q] ∩ [0,1]; borrows strength from past docstrings instead of bootstrapping a
   tiny n.
4. **Conformalized shrinkage at fixed budget** (the most direct answer to the original
   question): T = w·S̄_n + (1−w)·f̂(X), or a meta-estimator ĝ(X, S̄_n, σ̂_n, …); on
   calibration docstrings *simulate the same n*, compare T to the high-sample μ,
   conformalize |μ − T|. Beats bootstrap when features are informative; cost is
   dependence on calibration ≈ deployment distribution.
5. **Risk-controlled policies**: threshold τ on predicted risk; calibrate so accepted items
   have e.g. ≤5% expected hidden-test failure.
6. **Prompt repair from the judge**: assumption-explication step; auditor writes edge-case
   tests before codegen; calibrated routing table (high predicted success → simple prompt;
   high ambiguity → list assumptions; low sufficiency → clarify or skip; high sample
   variance → sample more + select; model-specific weakness → stronger model). Evaluate
   routing policies on calibration docstrings; don't trust the judge.
7. **Multiple models / prompts: calibrate after selection.** Picking the best predicted
   model or prompt is optimistic; run the full selection rule on calibration docstrings
   and calibrate the selected item's error or risk. Mondrian / grouped conformal per model,
   block by docstring when a docstring appears with many models.
8. What it does not give: per-docstring guarantees, robustness to distribution shift,
   validity across model-version or prompt changes without recalibration, anything without
   labelled calibration data, correctness if calibration leaks hidden tests.

**Relevance map (Claude-added, to be confirmed by Danielle).**
- `TLC` / `ELI`: the census and optimizer harness are literally this task — fractional
  score as the response, dedup with multiplicities, bootstrap over programs, block
  structure over OpenRouter batches, conformalized shrinkage with cheap features,
  docstring-adequacy judge as a prompt-repair and routing instrument.
- `IRT`, `TINY`, `ANN-opt-6`: interval choice for small-n pass rates (Wilson/Jeffreys vs.
  bootstrap; empirical Bernstein for conservative floors); "calibrate after selection"
  whenever a best recipe/checkpoint is chosen from many.
- `EDP`, `REC`, `FUNC`: any cheap-predictor → expensive-target forecast (early curve →
  final accuracy; recipe features → rank) is a split-conformal / conformal-risk-control
  problem — calibrated intervals on forecasts are the honest version of "predictability".
- `DCARD`: the estimand-discipline list (version, decoding, post-processing) is the
  provenance ledger's checklist restated for evaluation.

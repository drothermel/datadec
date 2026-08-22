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

## Undated (intake 2026-08-22) — training-free metrics for "does an item embedding explain IRT difficulty?"

**Danielle's setup (from speech).** Tasks × methods evaluation results → per-task
difficulty via something like IRT → several ways of embedding the tasks → she wants to
compare *embedding approaches* by how well they predict difficulty, using correlation-type
metrics that don't require training a predictor; concretely, if embeddings are clustered,
how to score cluster labels against difficulty.

**Response (condensed).** Two cases plus two metrics that skip clustering.

*Continuous difficulty, cluster labels.* Variance explained by clusters = correlation ratio
η² = Σ_c n_c(d̄_c − d̄)² / Σ_i(d_i − d̄)², equivalently cluster R² = 1 − Σ_i(d_i − d̄_{c_i})² /
Σ_i(d_i − d̄)². A cluster-mean lookup, so descriptive rather than a trained predictor; it
rises with cluster count, so report adjusted R² / ω² too, and fix the clustering method and
cluster-count selection across embeddings. Companions: Spearman between difficulty and
cluster-mean difficulty; within-cluster difficulty variance.

*Binned difficulty (easy/medium/hard).* Treat bins as classes, clusters as predictions:
NMI or V-measure as headline, ARI as robustness check, purity for intuition.

*Skip the clustering — it throws information away.*
- **kNN difficulty smoothness**: mean over tasks of the mean |d_i − d_j| over the k nearest
  embedding neighbours; compare with a shuffled-label baseline and report
  1 − kNN error / shuffled kNN error. Evaluates the embedding itself rather than embedding
  + clustering algorithm — the response's preferred primary metric.
- **Pairwise distance correlation**: Spearman between ‖x_i − x_j‖ and |d_i − d_j| over task
  pairs (a Mantel-style test).

Suggested table per embedding: cluster R², adjusted R², kNN smoothness, pairwise Spearman,
NMI if binned. Headline framing: "how much difficulty structure is recoverable from
embedding geometry without supervised training."

**Intake notes.**
- The kNN-smoothness and pairwise-distance metrics need a null with the right structure:
  shuffling labels preserves the embedding geometry but not any task-family block
  structure; if tasks come from a few benchmarks, a within-benchmark shuffle is the
  honest baseline (Claude-added).
- Pairwise Spearman over n² pairs has dependent pairs; use a permutation p-value, not the
  nominal one (Claude-added).
- Relevance: `IRT` (item difficulty from the DataDecide response matrix, then which item
  representation — text embedding, task-format features, token-level statistics —
  explains it; extends IRT-7's clustering of DIF items); `FUNC` / `REC` (the same
  training-free stack scores a *recipe* or *chunk* featurization against a target without
  a learned predictor).

## Undated (intake 2026-08-22) — first-pass item difficulty from a small respondent pool (4 models × 8 prompts)

**Danielle's setup.** A dataset of task samples, each evaluated by four language models
under eight prompts (32 responses per item); wants a first stab at IRT-style difficulty.

**Response (condensed).** Items = task samples; respondents = model–prompt combinations;
response = correctness.
1. *Baseline:* smoothed pass rate p̂_i = (s_i + α)/(n_i + 2α), α = 0.5, difficulty
   d_i = −logit(p̂_i); bootstrap over model–prompt combinations for intervals.
2. *Rasch / 1PL:* P(y_ij = 1) = σ(θ_j − b_i); corrects pass rate for respondent strength.
3. *Many-facet Rasch / logistic mixed model* — the recommended "serious first stab":
   logit P = μ + α_model + β_prompt + γ_model:prompt − b_sample, or
   `correct ~ 1 + (1|sample) + (1|model) + (1|prompt) + (1|model:prompt)`; difficulty =
   −(sample effect). A Bayesian fit gives posterior intervals per item.
4. *Hold off on 2PL/3PL*: with 32 respondents, item discrimination is noisy without
   regularization; 3PL only with a reason to model guessing.
5. *Difficulty is not one thing* — track per item: overall difficulty, discrimination
   (separates strong from weak systems), prompt sensitivity, model-family sensitivity
   (DIF), and "random-looking" (noisy/ambiguous). "Hard for everyone" ≠ "prompt-fragile".
6. *Validation:* leave-one-model-out and leave-one-prompt-out (does estimated difficulty
   predict the held-out system's outcomes?); bootstrap over respondents; monotonicity of
   ability vs. items solved; 0/32 and 32/32 items are only "very hard/easy relative to
   this pool".
7. Report per item: pass rate, difficulty, interval width, prompt sensitivity, model
   sensitivity, a note.

**Intake notes.**
- The respondent pool is the binding constraint here; DataDecide's IRT (`IRT`) has the
  opposite shape — thousands of checkpoint-respondents, so 2PL is affordable there and the
  many-facet decomposition becomes recipe × size × seed × step facets rather than
  model × prompt.
- The prompt facet is the item-format intervention of IRT-10 seen from the other side:
  prompt sensitivity per item is a direct estimate of "format-limited" vs. "hard".
- Which dataset this is was not stated; 4 models × 8 prompts matches the TLC/ELI census
  shape (`TLC`).

### Continuation — when each generation is scored by a whole test suite

**Danielle's follow-up.** The samples are code-generation instructions; each generation
runs an entire test set, so the response could be binary all-pass, the average pass
fraction, or one response per test case. How does that change the approach, and how to
reason about which to use?

**Response (condensed).** Two measurement layers now (instruction-level, test-case-level),
and the three choices estimate *different constructs*:

| Response | Model | "Difficulty" means | Role |
|---|---|---|---|
| all-pass z_ij | Rasch / many-facet on z | difficulty of a fully correct solution | **primary** (matches user-facing outcome and pass/fail benchmarks); coarse — 19/20 = 0/20 |
| passed-count c_ij ~ Binomial(K_i, p_ij), logit p = θ_j − b_i | beta-binomial / overdispersed (tests within a generation are correlated) | difficulty of partial behavioural coverage | **secondary**; separates "almost solved" from "fundamentally failed" when full-pass rates are sparse |
| per-test y_ijk, logit P = θ_j − b_i − t_ik + u_ij | hierarchical; **u_ij generation-level random effect is mandatory** | which requirements / edge cases are hard | **diagnostic** only; never naive independent items |

Why the average pass fraction is dangerous on its own: it measures **test-suite density**
— 90 easy format tests + 10 logic tests vs. 10 edge-case tests — and one bug fails 50
correlated tests. Fixes: group tests by requirement and average group scores; weight tasks
equally, not tests; hierarchical nesting; report full-pass and partial side by side.

Per-instruction report proposed: full-pass difficulty; partial-credit difficulty; mean
test-pass rate; mean test-pass rate *given not full pass* (near-miss vs. catastrophic);
failure concentration (variance of per-test pass rates within the task); prompt
sensitivity; model sensitivity; interval width. Illustration: three tasks at 10% full-pass
with 85% / 50% / 20% mean test-pass are three different kinds of hard.

Decision questions: does 90% of tests passed count as mostly correct? (no → full-pass
primary); are tests balanced across requirements? (no → requirement-group aggregation);
do tasks have different test counts? (yes → count outcome per task-generation, equal
task weights, or nesting).

**Intake notes.**
- This is the same "the program is the independent unit" rule from the first entry,
  now inside the response model: u_ij is the formal version.
- `TLC` chose the fractional test score as its response (2026-07-11 §4, "fractional test
  pass rate as a second signal"); this turn argues full-pass primary + fractional
  secondary, with the density caveat. Noted in TLC §4 as a design tension, not a change.
- The conditional near-miss score (pass rate given not full pass) is also a natural
  *optimizer* signal for TLC-2/ELI: it orders failures, which all-pass cannot.

### Second continuation — the evaluations as signal for automated prompt optimization (DSPy / GEPA)

**Danielle's follow-ups.** (1) How does the design change when the evaluations drive an
automated prompt optimizer like DSPy + GEPA, at one level initially and eventually two,
where per-test feedback may help and difficulty could select or characterize the most
useful tasks and tests? (2) "You talk about grouping by requirement, but I don't actually
have a way to group the tests by requirement, do I?"

**Response (condensed; full project-facing version in
`../../potential-projs/text-latent-code-autoencoder.md` §4 2026-08-22).** Objective becomes
example *utility*, not difficulty: medium-difficulty, high prompt-sensitivity, diverse
interpretable failure modes, clear feedback, low flakiness and cost. Three pools (reflective
trainset / Pareto valset / locked full-pass holdout). Metric = hybrid scalar + structured
text feedback (compile, n/N, failed groups, ≤4 representative failures, one-line advice).
Tiered tests (smoke / diagnostic / full). One score per task-generation so test count does
not weight tasks. IRT estimates (task and test difficulty, discrimination, sensitivities,
flakiness, redundancy) feed stratified batch construction and task/test clustering. One
level first; per-predictor feedback for two. Without requirement labels: full-pass +
per-task pass rate + representative failures now; **cluster tests by their pass/fail vector
across the respondent grid** (co-failing tests ≈ one bug), label clusters later; cluster on
test content if available; emit requirement metadata if tests are generated going forward.

**Intake notes.**
- The empirical test clustering is the test-level analogue of the item-embedding question
  in the second entry, with the pass/fail vector as the embedding — and the same η² / kNN
  metrics apply to scoring candidate test groupings against held-out generations
  (Claude-added).
- "Optimization metric educational, final metric uncompromising" is the calibrate-
  after-selection rule from the first entry in optimizer form: the holdout is what the
  selection rule is scored on.
- GEPA 2507.19457 added to the ledger (agent-supplied).

## Undated (intake 2026-08-22) — ranking metrics when pairwise decision accuracy saturates

**Danielle's prompt (verbatim).**

> I am predicting the ranking of a list of things (~25 items) and one metric that is
> relevant is "decision accuracy" which would be the average pairwise prediction accuracy.
> however, even bad baselines do great on this metric. so then I'd like to instead use a
> metric that captures "correct rank predicted" or somehting like this to capture that
> swapping element 1 and 2 is substantial even if 1 and 2 are both ranked higher than all
> the rest correctly so there's a high decision accruacy.
>
> My loose memory is that NDCG or something like that is a ranking metric that captures
> osmething like this, but it weights the values at the top of the list heavier or
> somethign like this? Waht woud be good metrics for me to consider

**Response (condensed).** NDCG@K (DCG = Σ rel_i / log₂(i+1), normalized by the ideal
DCG; top-heavy via the log discount; graded relevance); MAP@K (precision at each relevant
position — needs a binary "relevant" notion); weighted rank correlations (Blest's
weighted rank correlation; the top-weighted ν family with a tunable emphasis parameter);
Precision@K / Recall@K; Kendall τ and Spearman ρ noted as position-uniform, with weighted
versions existing. Recommendation: NDCG@K (K ≈ 5–10) primary, MAP@K secondary, combine
with P@K. Sources are recsys blog posts plus one weighted-rank-correlation paper.

**Intake notes.**

- The response answered a recsys question; Danielle's is a *full-permutation* question
  (all ~25 items have a true rank; nothing is "irrelevant"). That changes the menu:
  - MAP@K and P@K/R@K need a relevant/irrelevant split and are the wrong tools unless she
    defines "top-k recipes" as the relevant set — then P@K is just "how many of the true
    top-k did I put in my top-k," which is coarse but interpretable.
  - NDCG with *rank-derived gains* (gain = 25 − true_rank, or exponential) works, but its
    log discount is mild: swapping the true #1 and #2 costs only the difference between
    1/log₂2 and 1/log₂3 on two items — small against a sum over 25. It will not make the
    1↔2 swap "substantial" unless gains are steep (2^rel − 1 style).
  - **Kendall τ and Spearman ρ** are the baseline permutation metrics; they are *not*
    what saturates — decision accuracy *is* (τ+1)/2 up to ties, so any pairwise-accuracy
    saturation is identical for τ. The problem is position-uniform weighting, not the
    statistic.
  - The right families for "top swaps matter more" on a permutation are
    **top-weighted rank correlations**: weighted Kendall τ (Vigna 2015, hyperbolic
    weights; `scipy.stats.weightedtau`), **rank-biased overlap** (Webber, Moffat & Zobel
    2010; persistence parameter p sets how top-heavy), and Blest's / the Wroclaw ν family
    the response named. These keep the whole permutation and let her choose the emphasis
    curve explicitly.
  - Two plain, reportable complements: **top-1 / top-k hit** (did the true best recipe
    land in the predicted top-k) and **regret** (true metric of the predicted #1 minus
    true metric of the true #1) — the latter is the decision-theoretic quantity DataDecide
    actually cares about and is immune to the "baselines look fine" problem because it is
    in metric units.
- Why bad baselines score well: with ~25 items and a few large effect sizes, most pairs
  are easy; pairwise accuracy is dominated by far-apart pairs. Report it stratified by
  true-rank gap (adjacent pairs only; pairs within the top 5) before replacing it —
  adjacent-pair accuracy is often the honest version of the same metric.
- This is EDP's metric suite question revisited (`../../potential-projs/early-dynamics-
  prediction.md`, 2025-07 responses 12–15 already list ρ/τ + NDCG@10); the additions
  there should be weighted τ / RBO and regret, not MAP.
- Weighted τ (Vigna 2015, 1404.3325) and RBO (Webber et al. 2010, TOIS) are
  Claude-added, unverified.


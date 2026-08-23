# Evaluation methodology and paradigm comparison — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: the research hypothesis (`../../research-hypothesis.md`) rests on the
claim that pipeline-stage comparisons are confounded by unequal tuning history and by
uncontrolled elicitation; these are the precedents for that claim and for the
existence-proof alternative.

---

## 2026-08-18 — precedents (from the Research Trajectory page)

- Melis, Dyer & Blunsom 2018, *On the State of the Art of Evaluation in Neural Language
  Models* — "showed LSTM-vs-transformer conclusions inverted under equalized tuning
  budgets." The pre-2021 ancestor of the "headline phenomenon is a tuning artifact"
  finding-shape.
- Sara Hooker, *The Hardware Lottery* — "research directions win not on merit but because
  the surrounding stack co-evolved with them, and a decade of co-adaptation can't be
  replayed for the challenger inside any single experiment's budget." Danielle's version
  points at the software stack (init schemes, optimizer settings, warmup/decay conventions,
  curricula, eval formats).
- *Position: Lifetime Tuning Is Incompatible with Continual Reinforcement Learning* — "the
  same complaint from the RL side."
- Existence proofs as paradigm evidence: AlexNet (Krizhevsky et al.), GPT-3 (*Language Models
  are Few-Shot Learners*), DeepSeek-R1-zero — "existence proofs, not controlled
  comparisons." Within the arc: shrink-and-perturb; continual backprop ("Sutton's own gloss
  was that continual backprop at least shows the problem can be solved").
- Failure mode of the existence-proof genre: the 2024–2025 RLVR literature, corrected by
  *A Sober Look…* (seeds) and *Spurious Rewards* (elicitation in disguise) — see
  `pretraining-to-posttraining.md`.

---

## 2026-08-18 — Signal and Noise (from the Research Trajectory page)

- Heineman et al., *Signal and Noise: A Framework for Reducing Uncertainty in Language Model
  Evaluation* (NeurIPS 2025 per the discussion) — signal = a benchmark's ability to
  separate better from worse models; noise = sensitivity to random variability between
  training steps; interventions: continuous (perplexity-type) metrics beat accuracy on
  both; filtering noisy subtasks improves aggregate reliability. Release: ~900K evaluation
  results on 465 open-weight models including OLMo intermediate checkpoints, DataDecide,
  and the model-ladder runs. The trajectory drift/diffusion project is its dual.

---

## 2026-08-18 — where eval variance actually lives (from the Research Trajectory page)

For OLMES-style loglikelihood evaluation, re-evaluating a fixed checkpoint with new seeds
buys nothing: inference nondeterminism is negligible, generation-based evals are the
exception, and few-shot configuration variance is "a bias axis to sweep, not noise to
average." The variance of interest is in training (seed, data order, init) — consistent
with Signal and Noise's definition of noise as step-to-step wander. Cited: OLMES (*A
Standard for Language Model Evaluations*).

## 2026-08-23 — Danielle's model-divergence hypothesis and behavioral-distance designs (early-2026 conversation, historical)

From chunk 3 of the ChatGPT re-eval conversation (~Jan–Mar 2026; transcript at
`~/drotherm/data/convo-artifacts/2026/2026-08-23-prompt-opt-reeval-aha/`; project
context in the TLC doc §4). Respondent claims unverified.

**The hypothesis (Danielle, verbatim):** "I've heard frequently from researchers that
all the model provider's models have collapsed to almost identical solutions.  My
results on simple fxn generation seems to suggest almost the opposite and makes me
wonder whether for the specialized task of coding older models were actually *more*
similar because their gains came mainly from general LLM improvements but now we're
making great progess in the coding realm because of the ability to verify outputs for
rewards or dataset generation, etc so different model families may actually have gotten
*further* from each other in the coding task space over time."

Note this is grounded in her own data: the early-2026 TLC baseline runs on simple
function generation showed large cross-model behavioral differences, against the
folk-wisdom "collapse" claim.

**Behavioral-distance measurements** (respondent's menu; all run on fixed task sets at
fixed decode settings):

- A. Agreement/correlation on per-item success across models; cluster models
  (dendrogram/heatmap). Collapse predicts high correlation, tight clusters.
- B. Error-mode distance: label failures (not-code / parse / type / wrong-algorithm /
  off-by-one / timeout / contract violation), compare distributions across models (JS
  or Wasserstein divergence). Equal pass@1 with different failure signatures is the
  strong anti-collapse evidence.
- C. Output diversity under controlled sampling: N samples per prompt per model;
  unique AST shapes / CFG-ish fingerprints, edit-distance distributions, solution
  families after normalization (formatting, alpha-renaming).
- D. Prompt transfer as behavioral probe: optimize on A, evaluate on B; collapse
  predicts high transfer, small gaps.
- E. Time axis (her hypothesis directly): 2–3 generations per family, repeat A–D, test
  whether pairwise distances shrink or grow over time — OpenRouter's historical model
  sequences as the asset. Respondent's prior: convergence on easy items, divergence on
  hard/edge-case items.

**Workshop-sized slices** (respondent, on request): H1 behavioral clustering (200–500
tasks × 6–10 models, pass/fail correlations); H2 failure-mode divergence ("equal mean,
different tails" — same runs plus failure taxonomy; recommended single pick, least
sensitive to baseline accuracy levels); H3 transfer matrix (3 models, optimize-on ×
eval-on deltas, transfer-gap metric). Concrete starter: ~300 generated tasks, 6 models
(3 families × old/new), log pass/fail + failure taxonomy + normalized solution
fingerprint; plot success-correlation matrix, failure-distribution JS divergence,
per-model solution-family entropy.

Status: a freestanding project candidate (uses the TLC synthetic library as testbed but
is not TLC); promotion to staging/project doc is a pending intake decision.

## 2026-08-23 — Decision quality under constraints as a divergence probe (chunk 5 of the early-2026 conversation)

Continuation of the divergence-hypothesis entry above; same historical conversation and
provenance caveats.

**Danielle's idea (near-verbatim):** tasks "with clear constraints that make one of the
options clearly optimal given your constraint, but where an agent might not choose that
option even if they technically solve the task" — e.g., performance-critical sorting of
a huge float list, choose among three implementation sketches (bubble / merge / radix)
and complete the chosen one. Verify implementation correctness with tests; verify
choice optimality "probably directly via ast, but definitely via timing." Generation
strategy: work backwards from the curated "true statements" of software engineering
(interview-prep books and sites) → small set of clearly ranked choice options →
templated task scenarios + verification methods.

**Prior-art answer (respondent, unverified):** no widely adopted benchmark has
choose-among-sketches-then-implement as its core unit, but efficiency-beyond-
correctness is an active crowded area: ENAMEL (eff@k, expert references, stress test
generators), EvalPerf/DPE (profiling against reference solutions at distinct efficiency
tiers), EffiBench (NeurIPS D&B, efficiency vs. human canonical solutions on
LeetCode-style tasks), and DS-1000 (2211.11501 — execution-tested data-science code
across pandas/numpy/matplotlib/sklearn/scipy/torch with surface-form constraints; the
direct precedent for the chunk-4 data-manipulation family, complemented by a generator's
controllability and anti-memorization). Danielle's pushback that this is "a pretty hard
pitch" for a full paper accepted as correct for a generic efficiency benchmark.

**The defensible framing (respondent's recommendation):** decision quality under
constraints *as a probe for model divergence and reliability* — merging this thread
with Danielle's divergence hypothesis. Unit of evaluation: three separable skills via a
two-stage protocol (structured pick-and-justify, then implement): (A) choice quality,
(B) implementation correctness, (C) constraint adherence (profiling/static checks).
Unique hook a generator enables: constraint-varying families where the *right choice
changes with the constraint* (memory cap → in-place wins; stability required → stable
sort; latency-critical → vectorization), testing whether models track constraints.
Candidate full-paper claim set: similar-on-correctness ≠ similar-on-decision-quality;
decision quality more variance-prone and family-sensitive; prompt optimization improves
correctness more than decision quality (or vice versa by regime); "good coders, bad
planners" as a persistent per-model profile.

**De-risking (Danielle flagged high risk of unconcludable outcomes; respondent's
answer):** named failure modes — ceiling, floor, spec ambiguity, timing noise, and the
choice-vs-implementation-difficulty confound. Mitigations: the A/B/C separation itself;
low-risk families (asymptotic slam-dunks verified by scaling regime or hard timeout
rather than microbenchmarks; API-constraint compliance via AST/import checks — pandas
vectorization, pydantic validators, torch-not-numpy; memory caps via OS limits);
ceiling knobs (trap options, conflicting constraints requiring prioritization, 3→5
options), floor knobs (fill-in-missing-lines scaffolds, partial starters), timing knobs
(coarse scaling tests, timeouts, median-of-5, pinned threads). Hypotheses designed to
be interpretable under nulls: H1 choice separates families more than correctness (null
= "constraint-following is commoditized"); H2 optimization improves implementation more
than choice (either direction tells a story); H3 relative seed-stability of choice vs.
implementation ("planning stable, execution noisy" or the reverse). Pilot kill-test:
30 tasks (10 asymptotic / 10 API-constraint / 10 vectorization-vs-loop) × 6 models × 5
seeds.

**Intake note (Claude-added):** the two-stage structured pick is exposed to
option-position and label bias (models preferentially pick option A / the first or last
listed); the respondent never mentions randomizing option order and identity across
samples, which this design needs from day one.

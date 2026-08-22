# Evaluation methodology and paradigm comparison — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: the research hypothesis (`../research-hypothesis.md`) rests on the
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

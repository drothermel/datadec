# Model behavioral divergence — "cheap models aren't collapsed"

**Kind:** staging (promotion candidate). Promoted from the divergence thread in
`../reference/evaluation-methodology-literature.md` by decision 2026-08-23. Uses the TLC
harness and task infrastructure as its testbed but is not TLC — it is a freestanding
measurement program about models, not representations.

**Provenance.** Danielle's hypothesis from the February-2026 conversation (transcript at
`~/drotherm/data/convo-artifacts/2026/2026-08-23-prompt-opt-reeval-aha/`, chunks 3 and
16–17; dated entries in `../reference/evaluation-methodology-literature.md`). An
implementation-ready workshop-paper spec ("Behavioral non-collapse in cheap coding
LLMs," originally aimed at the ICLR 2026 Re-Align Challenge track) was written on
2026-02-08 and never executed — no version was submitted or close. The spec is
preserved verbatim in the transcript (chunk 17).

## The hypothesis (Danielle, verbatim)

> "I've heard frequently from researchers that all the model provider's models have
> collapsed to almost identical solutions. My results on simple fxn generation seems to
> suggest almost the opposite and makes me wonder whether for the specialized task of
> coding older models were actually *more* similar because their gains came mainly from
> general LLM improvements but now we're making great progess in the coding realm
> because of the ability to verify outputs for rewards or dataset generation, etc so
> different model families may actually have gotten *further* from each other in the
> coding task space over time."

Grounded in her own data: the early-2026 TLC baseline runs showed large cross-model
behavioral differences on simple function generation, against the folk "collapse" claim.

## What exists, ready to use

- **The paper spec** (2026-02-08): thesis — similar mean accuracy, divergent tail risk,
  failure-mode signatures, and success-conditioned solution-strategy distributions,
  persisting across specs and prompt regimes. H1 tail/failure-mode non-collapse; H2
  success-conditioned strategy diversity ("even when they succeed, they differ" —
  divergence in the equivalence-class interior); H3 persistence across ≥2 prompt
  regimes and ≥2 task families (fallback claim if strict contracts collapse behavior:
  collapse on core correctness, divergence in robustness dimensions). Four figures:
  mean-vs-tail scatter; failure-signature heatmap + clustering; success-conditioned
  behavioral-fingerprint heatmap (strategy features headline, style features
  supporting); regime stability. Behavioral signature s_m over waterfall categories
  with JS/TV/Wasserstein distances; P(catastrophic) tail metric; per-sample logging
  schema.
- **Distance designs** (chunk 3): agreement/correlation clustering; failure-mode
  distance (condition on failure); output diversity under controlled sampling (AST/
  fingerprint solution families — note fixed temperature is not behaviorally
  equivalent across families); prompt transfer as behavioral probe; the time-axis
  study (2–3 generations per family — her hypothesis directly).
- **Adjacent probe:** decision quality under constraints (same reference file,
  chunks 5–7) as the sharpest single divergence instrument if the program expands.
- **Data/infrastructure:** the TLC variance runs (FizzBuzz ×100; synthetic families ×
  providers × cost tiers × DG-* prompt regimes) already have the multi-level sampling
  structure. **No AST-metrics code exists** (corrected 2026-08-23) — the strategy-
  feature extraction must be built; Danielle has code-processing codebases to point to.

## Time-sensitive note

The time-axis arm's unique asset — historical model sequences still served on
OpenRouter — erodes as providers deprecate old models. If this promotes, a cheap
snapshot run (harvest outputs from currently available older checkpoints on the fixed
task set) should happen first, ahead of any analysis or writing.

## Gate

Danielle decides whether this competes for time (against TLC and the rest of the
portfolio). Exits: promote to a project doc with its own prefix (the spec is close to
§1-ready), or absorb the durable pieces into TLC-0's measurement suite and the
evaluation-methodology accumulator and delete.

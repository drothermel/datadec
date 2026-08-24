# Convergent intake findings — cross-cutting threads that recur across independent sources

**Kind:** reference (standing accumulator for findings that surface independently in
multiple intake sources; one dated section per synthesis pass). Each thread lists its
independent witnesses with pointers to the accumulator entries holding the detail.
Everything here is agent-distilled from unverified sources — convergence raises
confidence but is not verification, and shared upstream sources can manufacture
false convergence (several witnesses below are NotebookLM syntheses that could
share training-data priors).

---

## 2026-08-24 — three threads from the post-batch link run

The 2026-08-24 intake run (4 conversations + 12 NotebookLM notebooks; bundle
`2026-08-24-notion-lit-reviews/`) produced three findings that recurred across
sources that were not talking to each other.

**1. Cheap improvement loops amplify; they don't instill (mode collapse as the
shared failure).** Independent witnesses: the SFT-vs-RL atomic-skill study (SFT
induces jagged over-specialization; RL preserves balanced profiles —
`pretraining-to-posttraining.md`); the iterative-DPO study (matches RL reasoning
gains cheaply but "amplifies existing reasoning patterns rather than instilling
novel self-reflection" — same file); SEAL self-edit search (mode collapse without
explicit novelty pressure — `prompt-optimization-landscape.md`); execution-grounded
automated AI research 2601.14525 (evolutionary search sample-efficient, RL updates
mode-collapse — same file); and the standalone-LLM-optimizer verdict (frontier
models mode-collapse as lone search algorithms; hybrids with classical state
management win — same file). Working statement: *optimization pressure applied
through the model's own outputs selects among existing behaviors; genuine novelty
needs either external structure (evolutionary search, classical state, verifiers)
or explicit diversity pressure.* Program relevance: C2's optimizer-vs-bandit
decomposition (what does the generator actually add?), TLC-opt loop design, and
the data-autophagy warning from the self-improvement survey.

**2. Over-training is not free — four independent costs.** Witnesses:
catastrophic overtraining ("Overtrained LMs Are Harder to Fine-Tune" — parameter-
transformation sensitivity grows with token budget; `plasticity.md`); weight-decay
plasticity (pretraining HPs chosen on pretraining loss alone yield worse
post-trained models — same file); Scaling Laws for Precision (quantization
degradation increases with over-training — `schedules-and-annealing-literature.md`,
with the PTQ-robustness paper's LR-decay counter-attribution flagged as a
read-together pair); and the CPT "loss potential" framing (initial loss level
bounds adaptability — same file). Working statement: *the token budget trades
final pretraining loss against downstream malleability (fine-tunability,
quantizability, adaptability), and the trade is invisible to loss-only HP
selection.* Program relevance: directly the pretraining-choices→post-training-
outcomes pillar; joins B2's sparsity-hurts-reasoning-unrecoverably result
(`moe-literature.md`) as pretraining decisions post-training cannot undo.

**3. The representation surface changes what a frozen model can do.** Witnesses:
LANTERN's cross-language repair hypothesis (translating a stuck bug into the
model's strong language beats iterating deeper in place —
`prompt-optimization-landscape.md`); PseudoEval (pseudocode-for-docstring swaps
isolate different bottlenecks per language — `code-benchmarks-landscape.md`); the
NL-vs-NNL ablation (decoder accuracy collapses when lexical surface is randomized
at constant logic — `generalization-and-ood-literature.md`); NL-in-the-Middle's
representation-ordering effect and CoT-vs-pipeline gap
(`nl-bottleneck-prior-art.md`); and the Perplexity Paradox (token-level importance
is surface-statistics-confounded; the signature is the load-bearing NL —
`code-compression-literature.md`). Working statement: *for frozen models,
choosing the representation surface is an optimization move of the same order as
choosing the search algorithm.* Program relevance: TLC's NL-likeness ladder and
cross-decoder falsifiers are direct probes of this thread; it is the closest
thing the intake record has to independent support for the representation
program's premise (with the usual caveat that supportive readings were
agent-selected).

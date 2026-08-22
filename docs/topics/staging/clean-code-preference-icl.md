# Clean-code preference via in-context / lightweight adaptation — a small practical test

**Kind:** staging. Candidate exits: a small standalone project doc (practical tooling with a
measurement angle; program pillars served: none directly — though it is a hands-on instance
of the elicitation-vs-weight-update question), or absorption into ICL elicitability as an
applied probe. Gate: Danielle actually running the v0.

Source: an external conversation (undated; intake 2026-08-22) following the ICL scaling
thread in `../reference/icl-literature.md`. The responses were thin; Danielle's statements
carry the content.
---

## Undated — Danielle's framing

**Decision.** Table the ICL related-work investigation for now — "my goal is to test this
approach out practically for myself in a small-scale way first" — while noting it "directly
links to some very interesting and current research that could be helpful for my future
research directions."

**Goal.** A very simple, practically useful setup for adapting coding models (via in-context
prompting and possibly lightweight tuning) to her coding preferences, which doubles as a
probe of "how far I can expect models that have been in-context prompted a bit to be able
to go in terms of just following these practices" — both a preparation process and a
limitations study.

**What she cares about.** Simplicity; semantically meaningful naming and grouping so the
code is easy to read; no "random additional things" such as if/else checks that simplify to
a boolean — a bundle of coding best practices to enforce in the model's expectations from
the start (acknowledging part of this is having a codebase that already follows them).

**Automated feedback (high priority; agent feedback judged too complex for now).**
1. Clear test cases: work with individual functions with a clear purpose, each with tests
   verifying the implementation meets expectations — pass/fail signal.
2. Length of the model's implementation vs. her reference implementation as a rough
   efficiency / verbosity proxy.
Response added: a consistency check — same prompt under slight rephrasings; does the code
still pass, and how much does it vary?

**Dataset construction idea.** Take her "utterly clean" examples and farm them out to many
models — half asked to make them *cleaner*, half to make them *substantially less clean* by
deliberately inserting clean-code violations (e.g. replacing a succinct construct with a
verbose one). Filter out anything that doesn't run; then a quick manual labelling from bad
to good. Use as **paired good/bad examples** rather than scalar rewards — she was reaching
for the name; the response supplied "preference learning / pairwise ranking" (the specific
thing she likely meant: DPO-style preference pairs, or pairwise comparisons for a
reward-free preference objective).

## Sketch of a v0 (synthesized from the above; not yet agreed)

- Corpus: N small functions with tests (her own reference implementations = "clean").
- Perturbation: k models × {cleaner, dirtier} rewrites → run tests → keep passing variants.
- Labels: manual bad→good ordering per function (pairs or a short ranking).
- Conditions to compare on held-out functions: zero-shot; few-shot with clean examples;
  few-shot with *contrastive* pairs (bad→good shown explicitly); optional lightweight
  tuning (LoRA/DPO on the pairs) for the same budget.
- Metrics: test pass rate; length ratio vs. reference; a clean-code rubric score (manual on
  a sample, or a judge model calibrated against her labels); consistency across
  rephrasings.

## Open questions

- Whether to treat the rubric ("clean") as a learned judge or keep it manual.
- Which models (API vs. local) and whether any weight updates are in scope for v0.
- Relation to the text-latent code autoencoder (`../../potential-projs/text-latent-code-autoencoder.md`):
  the style field there is exactly what this rubric measures.

**Waiting on:** Danielle running or refining the v0.

# Clean-code preference via in-context / lightweight adaptation — a small practical test

**Kind:** staging. Candidate exits: a small standalone project doc (practical tooling with a
measurement angle; program pillars served: none directly — though it is a hands-on instance
of the elicitation-vs-weight-update question), or absorption into ICL elicitability as an
applied probe. Gate: Danielle actually running the v0; a prior-art pass on code-style preference
datasets before any paper framing.

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

## Undated — Paper potential and prior-art check (same conversation)

**Danielle.** Realised the setup "is something that I could see doing as a preparation for a
workshop paper as well"; asked whether this specific approach — creating small,
high-quality coding datasets of good-vs-bad practice — has been done and published
recently.

**Response (thin; condensed; unverified).** Named **AgentPack** (arXiv 2509.21891): a
dataset of code changes co-authored by AI agents (Claude Code, Codex) aimed at code-editing
models; and **KODCODE**: a synthetic dataset of coding questions, solutions, and test cases
validated by self-verification. Claimed: a small, carefully labelled style/best-practice
dataset "hasn't been overdone" and could anchor a workshop paper. No search for
code-style / readability preference datasets specifically was evidenced.

*Intake note.* The two named datasets are about correctness and agent edits, not style
preference; the prior-art question that actually matters — code *readability/quality*
preference pairs, LLM-as-judge for code style, clean-code rubrics, code-review datasets —
was not searched. Add to the gate: a real literature check before any paper framing.

## Undated — Experimental structure: N in-context + M interactive examples; context-fraction axis

**Danielle's design.** Start with her own functions ("the thing that I want the model to
do is improve data in this specific domain"); easiest first: **parsing functions and
DataFrame-manipulation functions** — many of them, well contained. Prompt = task
description + N in-context examples; then M held-out examples worked *interactively*:
the agent produces code, it is tested, feedback is returned. Performance on the M examples
evaluates the initial prompt — but "there's not really something built into that setup
that would provide an evaluation of the benefit of the active learning steps." Response
(thin): measure held-out performance before vs. after the interactive rounds; and keep a
separate final test set seen only after all interactive steps to measure generalisation.

**Danielle's additional axis.** The fraction of the context window in use. With short
examples and a small fraction used, she would not expect much difference between the
first and last interactive example; to *test how interactive learning changes performance
over time* (rather than prepare an agent for her own interactive use), she would throw all
examples at the agent, and then must weigh "more examples versus a longer context over
which to remember." Response (content-free): agrees it's a trade-off.

*Intake note.* This pins the experiment's two distinct goals and their designs:
(a) *tool preparation* — find the N-shot prompt that generalises best, evaluated on
held-out functions; (b) *interactive-learning curve* — fix the prompt, stream M examples
with test feedback, and measure per-step performance on fresh items, i.e. the ICL learning
curve from `../reference/icl-literature.md` with test feedback as the "label." For (b) the
confound she names is context length: per-step improvement is entangled with growing
context, so the clean comparison is matched total context with different splits of
(unique examples × feedback rounds), which is the same (unique × repetition) factorial
proposed for ICL-elicitability.

## Open questions

- Which of the two goals (tool preparation vs. interactive-learning curve) v0 targets —
  they need different evaluation designs; start with (a).
- Whether to treat the rubric ("clean") as a learned judge or keep it manual.
- Which models (API vs. local) and whether any weight updates are in scope for v0.
- Relation to the text-latent code autoencoder (`../../potential-projs/text-latent-code-autoencoder.md`):
  the style field there is exactly what this rubric measures.

**Waiting on:** Danielle running or refining the v0; a prior-art pass on code-style
preference datasets if the workshop-paper framing is pursued.

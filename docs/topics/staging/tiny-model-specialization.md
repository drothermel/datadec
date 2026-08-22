# Tiny-model specialization via the outer layer — "the 10M pandas specialist"

**Kind:** staging. Candidate exits: a standalone project doc (a small, verifiable
specialization study outside DataDecide's pillars), absorption into
`../../potential-projs/text-latent-code-autoencoder.md` (the harness-optimization machinery
is the same) or `../../potential-projs/tiny-scale-measurement.md` (the within-reach-tasks
question), or a cross-listed optional direction on both. Gate: Danielle picks an experiment
shape (wrapper-only / wrapper + soft prompt / wrapper + LoRA) and an evaluation; a prior-art
pass on prompt-tuning-at-small-scale and tiny specialist models (below) before any paper
framing.

Source: an external conversation (undated, ~2026; intake 2026-08-22) that followed the TLC
draft PDF (not on file here; it is the draft behind `text-latent-code-autoencoder.md`, with
its harness $H_\theta$, verifiable success objective $J(\theta)$, and "waterfall" feasibility
metrics code-only → compiles → runs → passes tests). Danielle's statement carries the
content; the response is a reasonable but unsourced taxonomy. A literature review was
requested as the next step in that conversation.
---

## Undated — Danielle's hypothesis (verbatim, from speech)

> So recently I've been thinking about the ideas that are in the PDF that I just provided,
> um, and I'm probably gonna submit a very small portion um to a workshop in the near
> future. Um but as I thought about them more, it made me wonder whether it would be
> possible, like, what types of very low budget approaches might exist for fitting the
> models to specific tasks and specifically, how, like, what I, okay, I have a hypothesis
> that you could fit much smaller models than we usually um consider plausible to fairly
> specific tasks, potentially even just by optimizing the outer layer, um not even
> post-training the weights specifically. Um And I would be curious, A, whether that's true,
> B, if it is, what the breakdown is in terms of like, how specific of the tasks are we
> talking about, how much data would you need? Can you do it just as an external thing
> versus fitting weights? Um And then, how, if you are able to fit the models to very
> specific types of tasks, then how much does that destroy their general performance
> ability? Um because The really small models, like even 10 million, didn't really perform
> reasonably, even on the most simplified multiple choice benchmarks that were being used
> specifically to try to give them a shot. And so I do wonder, like, if, for example, I
> wanted to take a 10 million parameter model and make it really good at writing, I don't
> know, pandas code, right? Like, just the best pandas coder ever, then how plausible is
> that? And is there a way where an external LLM could fit a function that takes a standard
> human input and converts it into something that will actually allow instructions to guide
> the tiny LLM that is really great at writing pandas code? And so I'm curious whether you
> think this is a plausible idea and whether you think this is something that, if done as
> experiments, would get research interest. And then after you answer that question, I'll
> probably ask you to do a small literature review.

The hypothesis, restated: (H) models far smaller than usually considered plausible (~10M)
can be fit to **fairly specific tasks** by optimizing only an **outer layer** — an external
function, possibly fit by a large LLM, that rewrites standard human input into whatever
form lets instructions steer the tiny model — without post-training its weights. Questions:
(A) is it true; (B) the breakdown — task specificity, data needed, external-only vs.
weight-fitting; (C) how much specialization destroys general ability; and whether the
experiments would draw research interest. Context facts: DataDecide's smallest models
(~10M) did not perform reasonably even on the simplified multiple-choice benchmarks built
to give them a chance (see `../../open-questions-answered.md` and
`../../potential-projs/tiny-scale-measurement.md`); a small portion of the TLC draft is
headed for a workshop.

## Undated — Response (condensed)

**Framing.** The PDF already treats the optimized object as *interfaces and bottlenecks*
(prompts, latent formats, decomposition, verifiers, budgets) rather than weights; this
hypothesis is the same move applied to a tiny executor.

**(A) Three "outer layers," increasing in power:** (1) prompting / I/O shaping only — moves
format-discipline and feasibility failures (the waterfall gates) a lot; (2) a learned
wrapper or adapter that leaves the base untouched — linear probe / logistic head on frozen
representations (classification, routing, scoring), soft prompts / prefix tuning, or a
separate small **input rewriter** mapping user intent → tiny-model-friendly prompt/latent;
(3) parameter-efficient tuning (LoRA / IA3 / BitFit / last-layer-only) — cheap, usually
beats pure prompting when the base lacks the capability. Reality check: "outer-layer-only
is unlikely unless the 10M model already has a lot of latent code competence … the wrapper
can't conjure missing algorithmic/programming skill; it can mostly (a) route, (b) constrain,
(c) retrieve, (d) translate into what the base already knows."

**(B) Specificity / data / external-vs-weights.** Spectrum: very narrow, high-structure tasks
(classification, schema filling, field extraction, format conversion, ranking, choosing
among options) → frozen + head or prompt tuning with very little data; medium (small domain
code patterns, templated transformations, API-call synthesis) → retrieval + templates or
PEFT; broad capability (general codegen, novel composition, hard reasoning) → "10M won't get
there with a head." Data rules of thumb (unsourced): linear head tens–few thousand; soft
prompts hundreds–tens of thousands; LoRA/light SFT thousands–hundreds of thousands, with
movement at low thousands if the task is narrow and the eval tight. Decomposition proposed:
external teacher/optimizer (big model) generates compressed specs / plans / latents and
proposes harness edits (the draft's LLM-as-optimizer loop); tiny specialist emits the cheap
step (pandas code) from a highly structured prompt; retrieval + unit tests as verifiers so
rewards are automatic. "Spend expensive tokens only on the optimizer/teacher; keep the
specialized executor tiny."

**(C) Generality loss.** Wrapper-only: nothing is destroyed — the model is just not used
generally; out-of-domain inputs get funnelled into "pandas-land," which is a feature ("make
a specialist behave like a tool"). Weight-touching PEFT: forgetting is real but manageable
(mixed training with a general slice, KL to base outputs, a general-sanity eval suite).
Feasibility fixes (format, compilation, runtime) improve success without capability
changes that damage generality.

**The 10M pandas case.** Without weight updates, the plausible path is big model → restricted
DSL / structured spec → tiny model maps DSL → pandas within a narrow distribution →
verifiers — "the natural-language-bottleneck framing in your draft." With LoRA /
last-layer: "surprisingly competent at a tight subset (joins, groupby/agg, pivot,
resampling)" given standardized specs, eval-close training data, unit-test checking. "Best
pandas coder ever" is beyond 10M unless "ever" is inside a constrained distribution.

**Research interest.** Yes, if measurable and verifiable (the draft's setup) and if it shows
surprising efficiency trade-offs ("with no weight updates, optimizer LLM + harness + tiny
executor hits X% pass@k at < $Y, and here is where it breaks"). Pitch: "specialization
without fine-tuning" / "with < 1% parameter updates," emphasizing the budgeted interface
and automatic verification. Asked her to choose among wrapper-only / wrapper + soft prompt
/ wrapper + LoRA, and the evaluation.

## Intake notes

- The response is sensible and matches the TLC machinery, but it is unsourced, and it
  skips the one empirical result that bears most directly on (H): **prompt tuning and soft
  prompts are known to be weak at small scale** — Lester, Al-Rfou & Constant 2021 ("The
  Power of Scale for Parameter-Efficient Prompt Tuning") found prompt tuning only matches
  full fine-tuning above ~10B parameters and lags badly at small sizes; the same scale
  dependence is reported for in-context learning generally (see
  `../../potential-projs/icl-elicitability.md`). So the "learned wrapper without touching
  weights" tier has a known headwind exactly in the regime (H) targets. That is not fatal
  — the *external rewriter* variant is different from soft prompts, since the rewriter's
  capacity lives in the big model — but it means the honest version of (H) is "an
  external, large-model-fit interface can make a tiny model useful on a narrow task," not
  "a tiny model can be prompt-tuned into competence." From memory; verify.
- The tiny-specialist prior art the literature review should start from (from memory,
  unverified): TinyStories (Eldan & Li 2023 — coherent generation at 1–30M params when the
  distribution is narrowed); the phi-1 line (textbook-quality data for small code models);
  small-model DSL/semantic-parsing work (text-to-SQL with small seq2seq models; "neural
  program synthesis with DSLs"); distillation-to-small-specialists (e.g. "Distilling
  step-by-step," Hsieh et al. 2023); and the frozen-model probing literature for the
  head-on-frozen-reps tier. The big-model-as-rewriter idea is close to "LLM as a compiler
  to a restricted DSL executed by a small model" — check whether that exact shape exists.
- **Connection to the repo's existing questions.** (i) TINY's "within-reach tasks" and MIC's
  "guaranteed-effect calibration" are the same question as (A): *does the 10M model have any
  latent competence to unlock?* The DataDecide 10M models' failure on simplified MC is
  evidence that, for general tasks, there is nothing to unlock; (H) bets that a narrow
  enough distribution changes that. A clean first experiment is therefore a **capability
  existence test**: take the narrowest pandas subset (one-line groupby/agg with fixed
  column names), and check whether *any* outer layer — including an oracle DSL — yields
  non-trivial pass rates from a frozen 10M DataDecide model before building the optimizer
  loop. If nothing moves under an oracle interface, the answer to (A) is no at 10M and the
  study shifts to "what is the smallest size at which it becomes yes" — which is a
  scale-sweep TINY already wants. (ii) The waterfall feasibility gates and pass@k machinery
  are TLC's; (H) is TLC with the decoder replaced by a tiny frozen model and the latent
  replaced by a DSL — worth stating as a TLC optional direction even if staged separately.
  (iii) (C) for the wrapper-only case is answered by construction; for the PEFT case it is
  a small instance of MIC's movement-decomposition question.
- The "how much data" rules of thumb should not be quoted; they are unsourced folklore.

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

## Undated — Wrapper-only, and the two questions it turns into (Danielle, verbatim from speech)

> I'm definitely most interested in the wrapper only. And I guess it feels like there are
> two different types of questions that I have here. One type of question is purely like
> um analytical scientific, right? I don't necessarily have a use case for why it would be
> useful for a 10 million model, 10 million parameter model to be able to execute very
> specific code tasks based on a wrapper and an expensive model giving it very, very
> specific instructions. But from a trying to understand what these models at different
> sizes can do perspective, I think it would be very interesting to know, like, as you use
> the same strength external um like optimizer prompter agent with the different models
> ranging from 1 billion all the way down, at what point do you really stop being able to
> get any type of reasonable output on a given set of tasks if you fix your training data
> and your budget? Um I think that just would be such an interesting test and thing to
> understand. And while, I don't know, I also think that if it really were possible. to do
> that, then you could also see what the same spectrum looked like after post-training on a
> data set that wasn't particularly tuned to let you see movement on a specific type of
> task, where for a fixed model size, the post-training really would make no difference in
> terms of the model performance. But if you use this type of fitting loop on a variety of
> very specific types of tasks, then you started to be able to see performance movements
> between the end of pre-training and the end of a post-training round. I feel like that
> could be a really interesting way to extract model differences on challenging tasks, even
> in the scenario where the models are so small that it's very difficult to prompt them to
> show their capabilities. And you could argue, like, I mean, I don't think you could ever
> formally phrase it that way, because you could argue that the, like, much smarter outer
> model is, in fact, eliciting the behavior it wants by pushing the buttons of the smaller
> model. So it may be unfair to say the smaller model has the ability to perform a certain
> task, but I still think it'd be interesting.

Decisions and questions fixed by this: **wrapper-only** (no weight updates, no learned
adapters). Two questions: (Q1, scientific) with the *same* external optimizer agent, fixed
training data, and fixed budget, at what model size (1B → 4M, the DataDecide sweep) does
reasonable output on a fixed task set stop being extractable — a **cliff curve**; (Q2)
whether the same fitting loop can **reveal pre- vs. post-training movement** at a fixed
small size where direct prompting shows none, i.e. use the loop as a more sensitive
instrument for model differences on challenging tasks. Her own caveat: the outer model may
be "pushing the buttons" of the small one, so "the small model has the ability" can never
be formally claimed.

### Response (condensed)

**Object of measurement:** *elicitation capacity under a constrained interface* — given a
fixed outer optimizer and fixed budgets, how much verifiable behavior can be extracted from
a base model by choosing and iterating on the harness alone (the draft's $H_	heta$ →
$J(	heta)$ loop). The fairness concern is answered by declaring the outer model **part of
the instrument, like a microscope**: the claim is "the base model contains behavior that a
fixed microscope with bounded effort can elicit," not "the base model can do X."

**Q1 → a curve:** x = base size, y = best achieved success under wrapper-only optimization,
at fixed wrapper budget (outer tokens, trials, wall-clock) and fixed task suite. Expect
smooth degradation on some tasks and genuine cliffs on others (where planning /
algorithmic depth is needed).

**Q2 → three quantities direct prompting misses:** post-training may leave zero-shot flat
while increasing (i) the *volume* of wrappers that work, (ii) *stability* across wrappers,
(iii) *sample efficiency* of the outer optimizer.

**"Wrapper-only" rules.** Allowed: prompt templates; formatting constraints / DSLs; staged
prompting (plan → code → verify → repair); external verifiers (tests, type check,
execution); retrieval from a fixed corpus. Not allowed: gradient updates; learned adapters
or soft prompts; **the outer model writing the final answer** — it may propose wrapper
changes and critique, but the answer channel must be the base model.

**Design.** Tasks (2–3, each with a waterfall verifier): pandas unit-test suite with hidden
tests; JSON-schema / structured extraction (valid + exact match); tiny algorithmic coding.
Base models: the DataDecide sweep (one family, many sizes — "architecture confounds kill
interpretability"). Fixed budget per problem: max outer-model calls (e.g. 5), max outer
tokens, max base-model samples, fixed decoding. Metrics per size × {pre, post}: $S_0$
(single generic wrapper, no optimization), $S_{opt}$ (best under budget), $\Delta S = S_{opt}
- S_0$ ("elicitation gain"), iterations-to-threshold (e.g. 30%), stability across seeds /
wrapper inits, and success-vs-iterations AUC as a single number. Hoped-for story: small
models have low $S_0$ but some show surprising $\Delta S$; at some size $\Delta S$ collapses
(the cliff); post-training leaves $S_0$ flat but changes $\Delta S$ or stability.

**Controls against "the outer model is doing the work":** (1) *outer-model-only* — the outer
model attempts the task directly under the same token budget; if it solves everything
cheaply the result is uninteresting, if it cannot but wrapper + small model can, the small
model contributes; (2) *shuffled / adversarial wrapper* — equally long but semantically
wrong or random DSL mappings; collapse shows the gain is targeted elicitation, not tokens.
Offered next: a 1–2-page workshop skeleton (contributions, experimental matrix, 3–4
figures) and a literature review anchored on elicitation, prompt/program optimization,
wrapper pipelines with verifiers, and capability-vs-extractability.

### Intake notes on this turn

- This turn moves the topic from "tiny pandas specialist" to **elicitation as a measurement
  instrument across scale and across pre/post-training** — which is the program's own
  framing: `../../potential-projs/icl-elicitability.md` (elicitation vs. weight update as
  access routes to the same capability) and `../../potential-projs/movement-microscope.md`
  ("lower the task instead of raising the model"; the instrument / detection-limit
  framing; the guaranteed-effect calibration). The response's "microscope" answer to the
  fairness worry is the same answer MIC already gives. At post-processing this should be
  cross-listed as an optional direction on both, whatever the staging decision.
- The cliff curve (Q1) is the same object as TLC's cliff structure
  (`../../potential-projs/text-latent-code-autoencoder.md`: bisection for critical ratios,
  binomial variance, SE = 0.5/√n) — the x-axis is model size instead of latent budget, and
  the same bisection-over-sizes / seed-variance machinery applies. Reuse it rather than
  re-deriving.
- The controls are right but incomplete. Outer-model-only is necessary; add a **fixed-outer,
  swapped-executor** control — the same optimized wrapper applied to a *different* small
  model (or the same size from another recipe) — to show the wrapper encodes something
  about *this* executor rather than a generic task decomposition; and a **wrapper-transfer**
  measurement (optimize on size $s$, evaluate on $s'$) which doubles as a recipe/size
  similarity readout. The "answer channel must be the base model" rule also needs a
  leakage check: the outer model's critiques must not contain the answer (token-overlap
  audit between outer outputs and final answers).
- Q2 connects to the earlier no-movement result in a specific way: the hypothesis is that
  Tulu-style SFT changed *extractability* ($\Delta S$, stability, iterations-to-threshold)
  without changing $S_0$. That is testable on the same DataDecide sizes and the same SFT
  runs, if the checkpoints from the earlier project still exist — worth checking before
  any new training.
- Budget note: the per-problem budget (5 outer calls × N problems × 14 sizes × 2 training
  states × seeds × 2–3 tasks) is dominated by outer-model tokens; the design should fix the
  outer model to one cheap, deterministic-decoding model and report cost per curve, per the
  project-approach principle "cost on every plot."

## Undated — Closing summary of the conversation (response; condensed to what is not above)

Danielle asked for "a summary of the topics that we discussed that would make it easy for
me to know what type of design decisions we kind of agreed on, what the related works are
that are linked to it." The response's list of **agreed design decisions** (recorded as
the conversation's settled set, pending her review here):

1. Wrapper-only: outer model = optimizer / elicitation engine that edits the wrapper and
   critiques; inner small model = the only component that emits the final answer.
2. Tasks with an automatic verifier and a waterfall of failure points (code-only →
   compiles → runs → passes tests) — the TLC draft's feasibility metric and success
   definition (its Eq. 4–5).
3. Outer model as microscope: fix optimizer (model + prompt policy) and budgets; report best
   achievable success per base model under equal elicitation effort.
4. Report elicitation gain $\Delta S = S_{opt} - S_0$, not raw score; the target plot is
   $\Delta S$ vs. base size, optionally pre vs. post-training.
5. Controls: outer-model-only under the same budget; sham wrapper of equal complexity.

Research questions as it left them: Q1 size-to-extractability curves (smooth vs. cliff by
task type); Q2 wrapper optimization as a microscope for post-training differences
(stability, sample efficiency, larger $\Delta S$) — "post-training didn't help *without
elicitation effort*"; Q3 wrapper-only specialization funnels inputs but does not destroy
generality, weight-tuning can (out of scope).

**Facts about the TLC draft it surfaced** (useful because the PDF is not on file here):
objective $J(\theta)$ over verifiable success with success = feasibility waterfall ×
semantic correctness (Eq. 4–5); harness parameters $\theta$ = prompts, templates, latent
format, stage decomposition, tool use, memory, sampling; LLM-as-optimizer loop whose
actions are prompt diffs / added verifier stages / sampling changes, reward = mean success
over batches (Eq. 7); a latent-format axis **COMP-NL vs. COMP-SHORT** (human-readable vs.
machine-oriented shorthand) probing the natural-language bottleneck; related-work stubs
naming RL / evolutionary prompt optimization and prompt compression, "language bottleneck
models," semantic-compression two-step pipelines, and AlphaCodium. Fit proposed: either a
new section of the TLC paper ("using the optimizer loop to elicit competence across base
sizes") or a sibling paper with the same formalism and pandas tasks as the reconstructed
object.

**Related-work map** (as given; identifiers where present, unverified): DataDecide (the
controlled data × scale suite); FollowIR (arXiv 2403.15246) — still the mismatched guess
from the post-training thread, see `../reference/pretraining-to-posttraining.md`; Tülu 3
(Ai2 open post-training stack — the baseline family for "generic post-training may not
move targeted metrics"); **AlphaCodium** (arXiv 2401.08500) — multi-stage, test-based
"flow engineering" that raises pass@k with no weight change, the concrete exemplar of
wrapper-only mattering for code; the prompt/harness-optimization cluster (OPRO / COPRO /
evolutionary — already in TLC §4); language-bottleneck and semantic-compression work.

Intake note: the summary adds nothing to the design beyond what the two turns above
record, except the TLC-draft internals and AlphaCodium as the named wrapper-only precedent;
AlphaCodium should also be cited in TLC's prior-art gate. The "agreed" list is the
response's reading; the only explicit decision Danielle made in the conversation is
wrapper-only.

## Intake notes (first turn)

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

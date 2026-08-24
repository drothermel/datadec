# Elicitation gain across scale — wrapper-only extractable competence as a measurement instrument

> **Draft scaffolding (2026-08-22).** Promoted from the staging topic
> `tiny-model-specialization`. §1–§3 are synthesized from the source conversation and not yet
> reviewed by Danielle; §4 is the dated discussion record (her statements verbatim). Treat
> §1–§3 as provisional until this note is removed.

**Program pillars served:** how (elicitation under fixed effort as a calibrated instrument;
the outer optimizer declared part of the microscope), apex (post-training movement detected
through extractability rather than zero-shot score). Also the DataDecide-scale sibling of
the non-pillar `TLC` harness work. (Program: `README.md` → Program.)

**One-line pitch.** Fix a strong outer model as an *optimizer of the interface* (prompts,
DSL, staging, verifiers — never the answer channel) with a fixed budget, and measure how
much verifiable task success can be extracted from frozen base models as a function of
model size (DataDecide 4M → 1B) and training state (end of pretraining vs. after
post-training). The readout is elicitation gain ΔS = S_opt − S_0, not raw score. Two
questions: where is the size cliff for extractable competence, and does post-training
that looks like "no movement" under direct prompting change extractability.

IDs: ELI-1–ELI-3 (core), ELI-opt-1–ELI-opt-4.

**Paper goal.** Workshop paper from ELI-1 + ELI-2 on one task family (size-to-extractability
curves with controls); main-conference with ELI-3 (pre/post extractability on the earlier
project's SFT checkpoints or new ones) and two or three task families.

Compute tiers: **API** for the outer model; **T1** forward passes with existing DataDecide
checkpoints for the executor. No training in the core.

---

## 1. What the project involves

### Core experiment (ELI-1–ELI-3)

**Setup shared by all three.** *Executor:* a frozen DataDecide checkpoint (final, or
post-trained variant). *Outer model:* one fixed, cheap, deterministic-decoding LLM that may
(i) propose edits to the wrapper and (ii) critique executor outputs against verifier
feedback, but may never emit the final answer. *Wrapper* θ: prompt template, formatting
constraints / a restricted DSL for the input, staged prompting (plan → code → verify →
repair), retrieval from a fixed small corpus, sampling settings. *Verifier:* automatic, with
a feasibility waterfall (format-only → parses/compiles → runs → passes hidden tests) and a
semantic success bit. *Budget per problem:* max outer calls (e.g. 5), max outer tokens,
max executor samples, fixed decoding — identical for every executor. *Metrics per executor:*
S_0 (one generic wrapper, no optimization), S_opt (best under budget), ΔS = S_opt − S_0,
iterations-to-threshold, stability across seeds / wrapper initializations, and
success-vs-iterations AUC; cost (outer tokens, executor tokens, wall-clock) on every plot.

- **ELI-1 — Capability existence test.** Before any optimizer loop: take the narrowest task
  slice (e.g. one-line pandas `groupby`/`agg` with fixed column names; or JSON-schema field
  extraction) and an *oracle* interface (hand-written DSL → prompt, best-known template),
  and measure S under that oracle for every DataDecide size with seed replicates. Output:
  the smallest size at which any outer layer yields non-trivial success. If nothing moves at
  the smallest sizes even under an oracle, the cliff is above them and ELI-2 starts there.
- **ELI-2 — Size-to-extractability curve.** Run the fixed outer optimizer at fixed budget on
  2–3 task families (pandas unit-test suite; structured extraction; tiny algorithmic coding)
  for every size in the sweep. Plot S_0, S_opt, ΔS vs. size; classify each task family as
  smooth degradation vs. cliff; locate cliffs by bisection over sizes with binomial error
  bars (SE = 0.5/√n at the cliff). Controls run at every size: *outer-model-only* under the
  same token budget; *sham wrapper* (equal length/complexity, semantically wrong or random
  DSL mapping); *swapped executor* (the wrapper optimized for size s applied to size s′ or to
  the same size from another recipe); an *answer-leak audit* (token overlap between outer
  critiques and final answers).
- **ELI-3 — Extractability before vs. after post-training.** At fixed sizes, repeat ELI-2 on
  pre- and post-trained checkpoints (the earlier project's Tulu/Tulu-3 SFT runs if the
  checkpoints exist; otherwise new light SFT). Hypothesis under test: SFT that leaves S_0
  flat changes ΔS, stability, or iterations-to-threshold. This is the "post-training didn't
  help *without elicitation effort*" reframing of the no-movement result.

### Optional directions

- **ELI-opt-1 — Wrapper transfer as a similarity readout.** Optimize on size s (or recipe
  r), evaluate on s′ (r′) without re-optimization; the transfer matrix is a cheap
  model-similarity measure comparable to the cross-recipe geometry statistics elsewhere in
  the program.
- **ELI-opt-2 — The DSL axis.** Vary the interface language from natural language to a
  machine-oriented shorthand (the TLC COMP-NL vs. COMP-SHORT axis) and measure ΔS as a
  function of interface form at each size.
- **ELI-opt-3 — The specialist framing.** The original "10M pandas specialist": at the
  smallest size above the cliff, how narrow must the task distribution be for a
  large-model-fit interface to make the tiny executor useful as a tool; report the
  generality funnel (out-of-domain inputs get mangled) rather than forgetting.
- **ELI-opt-4 — PEFT tier comparison.** Add soft-prompt and LoRA/last-layer arms to the
  same budget accounting to place wrapper-only on the spectrum; out of scope for the core
  by decision (wrapper-only).

## 2. Doability and impact

### Overall doability: **high** — inference only; cost is outer-model tokens

Everything runs on existing DataDecide checkpoints (T1) plus API calls for the outer
model. The budget is dominated by outer tokens: per-problem budget × problems × 14 sizes ×
training states × seeds × task families. Fixing one cheap outer model and reporting cost
per curve keeps it bounded; the ELI-1 existence test prunes sizes before the loop runs. The
harness is TLC's optimizer loop with the decoder swapped for a small frozen model; the
verifier suite is the waterfall already specified there.

Known headwinds: prompt tuning and in-context methods are weak at small scale (Lester et al.
2021; the ICL-elicitability literature), so the honest claim is about an *external,
large-model-fit* interface, not about the tiny model's own promptability. The "outer model
did the work" objection is handled by construction (answer channel rule + controls), not by
argument.

### Per-direction impact

| Direction | Impact | Why |
|---|---|---|
| ELI-1 existence test | Medium | Cheap, decisive; tells the program whether 4–20M models have anything to elicit on narrow tasks — directly informs TINY's within-reach question |
| ELI-2 cliff curves | High | A new, interpretable curve family ("extractable capability vs. size at fixed effort") with defensible controls; workshop-ready alone |
| ELI-3 pre/post extractability | High | Turns the earlier null result into a measurable claim; the instrument MIC and ICL want |
| ELI-opt-1 transfer | Medium | Cheap byproduct; cross-recipe similarity from the interface side |
| ELI-opt-2 DSL axis | Medium | Ties to TLC's bottleneck question |
| ELI-opt-3 specialist | Low–medium | Fun framing; scientifically subsumed by ELI-2 |
| ELI-opt-4 PEFT arms | Medium | Needed for a main-conference version; deliberately excluded from the core |

## 3. Infrastructure build sequence

1. **Verifier suite** (shared with TLC and the clean-code ICL topic; restated here, keep in
   sync): hidden-test pandas problems (graded by test pass; waterfall gates logged), a
   JSON-schema extraction set (valid + exact match), a tiny algorithmic set; seeds and
   per-problem records.
2. **Executor harness:** load any DataDecide checkpoint (HF), fixed decoding, batched
   sampling, token accounting.
3. **Wrapper + outer loop:** TLC's LLM-as-optimizer loop (actions = prompt diffs, DSL
   changes, added verifier stages, sampling changes; reward = mean success) with the
   answer-channel rule enforced and the budget meter; sham-wrapper generator; leak audit.
4. **ELI-1 runs** (oracle interface, all sizes, seeds) → choose the size range.
5. **ELI-2 runs** with controls; cliff bisection; plots with cost.
6. **ELI-3** once pre/post checkpoint pairs are located or produced.

**Cross-listed (2026-08-22).** `irt-reanalysis.md` IRT-10 — the BoolQ format intervention
(cloze vs. MCQ, label-balanced subsets, flipped label order on a checkpoint subset) — is
the first concrete instance of the elicitation thesis: an apparent capability floor that
may be a measurement floor. Keep its design consistent with the ELI-2 controls.

## 4. External assessments and origin notes

Dated notes from the source conversation (undated, ~2026; intake 2026-08-22), moved from
`topics/staging/tiny-model-specialization.md` — recorded for consolidation, not decisions.
The TLC draft PDF the conversation was conducted against is not on file; its internals as
surfaced are recorded in `text-latent-code-autoencoder.md` §4. Related-work claims in the
responses are unverified.

### 2026-08-22 — 𝒱-information as the shared measurement language with TLC (pointer)

TLC §4 (2026-08-22, extractable information) records predictive 𝒱-information (Xu et al.
2002.10689) — information usable by a declared extractor family — as the analysis layer
for "what is recoverable from a representation". The same object, roles swapped, is this
project's "extractable competence": here the probed variable is the model and the wrapper
family is 𝒱, so ELI's competence-vs-size curves are I_𝒱(model → task) for a declared 𝒱.
One probe harness can serve both; the declaration of 𝒱 (wrappers, prompts, budget) is the
thing both must publish.

### 2026-08-22 — structured-output literature as external evidence for the wrapper-vs-skill split

Danielle, in a separate conversation (record in
`../topics/reference/structured-output-literature.md`): "I suspect that the skill of
adhering to a very specific output format is a different skill than solving many specific
tasks and that it might be unnecessarily limiting to require that the same LLM does both."
That is this project's premise stated for a systems audience. Findings there that bear on
ELI (all unverified): "The Hidden Cost of Structure" (RANLP 2025) reports base models
often *benefit* from constrained decoding while instruction-tuned models degrade on
generation — a direct prediction for the pre/post-training axis (ELI-2), and DataDecide
checkpoints are base models; several 2026 benchmarks separate schema validity from value
accuracy, which is the feasibility waterfall (format-only → parses → runs → passes) with
off-the-shelf instruments; SLOT (EMNLP Industry 2025) shows a 1B fine-tuned
post-processing structurer, i.e. the wrapper can itself be a tiny model rather than a
prompt — a candidate wrapper class for ELI-1 beyond templates and decoders. The "valid
JSON but wrong value" failure is what the waterfall's "parses" gate must not be confused
with success.

### 2026-08-22 — the system-level prompt-optimization cluster is the outer loop's positioning set

From Danielle's SciSpace deep review of prompt optimization (record in
`../topics/reference/prompt-optimization-landscape.md`). ELI's outer model optimizes the
*interface* — prompts, templates, staging, verifiers, sampling — not only the prompt
text; the published cluster that does the same is: **Trace / OptoPrime** (2406.16218;
workflow as a graph, an LLM optimizer over heterogeneous parameters including
hyperparameters and code), **LLM-AutoDiff** (2501.16673; textual gradients through
multi-stage pipelines), **SPRIG** (2410.14826; system-prompt components by genetic
search, transferring across 47 tasks), **RePrompt** (2406.11132; agent instructions
refined from trajectories), and Lin et al.'s compound-AI optimization survey
(2410.16392) whose target list — prompt components, sampling parameters, tool specs,
orchestration — is the ELI action space. Differences to state: these optimize a
pipeline around a strong model for task performance; ELI fixes the outer optimizer and
the budget and reads the *executor's* extractable competence as the measurement. The
review omits APE, OPRO, ProTeGi, TextGrad, DSPy/MIPROv2, and GEPA; add those from the
bundle's 648-row search table when the gate runs. Gate still parked.

### 2026-08-22 — EPiC as the nearest published outer-optimizer with budget accounting

From a SciSpace paper summary (record in
`../topics/reference/prompt-compression-and-optimization-literature.md`). **EPiC** (Saluja
et al. 2024, arXiv 2408.11198) evolves code-generation prompts against test pass rate
with fitness-weighted selection and either an LLM mutator or a cheaper synonym mutator
(the cheap one wins on cost), and reports cost via **ATSP** — additional tokens per solved
problem relative to a baseline. Two uses here: (1) ATSP is the closest published analog
of ELI's fixed-budget elicitation accounting — adopt it or contrast explicitly against
ΔS-at-budget and iterations-to-threshold; (2) EPiC optimizes the prompt for a strong
model, whereas ELI fixes a strong *outer* model and optimizes the interface for a weak
frozen executor — the positioning sentence for the prior-art gate. PCRL and
Nano-Capsulator (same record) are prompt *compression* with transfer across models;
relevant to ELI only as evidence that discrete, gradient-free interface optimization
against black-box executors works. Gate still parked.

### Undated — Danielle's hypothesis (verbatim, from speech)

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
to give them a chance (see `../open-questions-answered.md` and
`tiny-scale-measurement.md`); a small portion of the TLC draft is
headed for a workshop.

### Undated — Response (condensed)

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

### Undated — Wrapper-only, and the two questions it turns into (Danielle, verbatim from speech)

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

#### Response (condensed)

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

#### Intake notes on this turn

- This turn moves the topic from "tiny pandas specialist" to **elicitation as a measurement
  instrument across scale and across pre/post-training** — which is the program's own
  framing: `icl-elicitability.md` (elicitation vs. weight update as
  access routes to the same capability) and `movement-microscope.md`
  ("lower the task instead of raising the model"; the instrument / detection-limit
  framing; the guaranteed-effect calibration). The response's "microscope" answer to the
  fairness worry is the same answer MIC already gives. At post-processing this should be
  cross-listed as an optional direction on both, whatever the staging decision.
- The cliff curve (Q1) is the same object as TLC's cliff structure
  (`text-latent-code-autoencoder.md`: bisection for critical ratios,
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

### Undated — Closing summary of the conversation (response; condensed to what is not above)

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
from the post-training thread, see `../topics/reference/pretraining-to-posttraining.md`; Tülu 3
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

### 2026-08-22 — estimation toolkit shared with TLC

The per-docstring estimation conversation recorded in
`../topics/reference/estimation-and-calibration-methods.md` applies to this project's
harness unchanged (fractional score, program as the unit, block bootstrap over provider
batches, Wilson/Jeffreys at small n, calibrate-after-selection when the best wrapper is
chosen). One ELI-specific use: a split-conformal predictor from cheap wrapper signals to
the expensive elicited-competence target gives calibrated intervals on "how much is
extractable" per model size — Danielle flags conformal prediction as a cross-project tool.

### Intake notes (first turn)

- The response is sensible and matches the TLC machinery, but it is unsourced, and it
  skips the one empirical result that bears most directly on (H): **prompt tuning and soft
  prompts are known to be weak at small scale** — Lester, Al-Rfou & Constant 2021 ("The
  Power of Scale for Parameter-Efficient Prompt Tuning") found prompt tuning only matches
  full fine-tuning above ~10B parameters and lags badly at small sizes; the same scale
  dependence is reported for in-context learning generally (see
  `icl-elicitability.md`). So the "learned wrapper without touching
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

## 5. Related work and positioning

*Purpose: the paper-facing synthesis — the prior-art landscape, this project's
position in it, and what each closest neighbor lacks. Unlike §4 (a dated intake
log, which grows by appending new entries **above this section**), §5 is a
current-state statement: rewrite it as understanding changes. Positioning claims
are Danielle's to make; agent-supplied literature claims anywhere in this document
are unverified leads, not established facts.*

**Status: not yet drafted.** Raw material: the dated entries in §4, the theme
accumulators under `../topics/reference/` (index: `../topics/README.md`), and
`../litreview/citation-verification-ledger.md` (citation provenance; nothing there
is verified).

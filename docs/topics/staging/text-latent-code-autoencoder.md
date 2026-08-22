# Text-latent code autoencoder — frozen LLMs as encoder/decoder, prompts as the only learned object

**Kind:** staging. Candidate exits: a standalone project doc (representation learning /
program synthesis; program pillars served: none), or a parked idea. Gate: pin down the bottleneck constraint on the latent (see
"The open question" below) before any promotion decision.

Source: an external conversation dated 2026-07-11 (intake 2026-08-22). Danielle's opening
prompt was not pasted; the setup below is the respondent's playback of it. Related-work claims in the quoted
material (round-trip correctness, DSPy, OPRO, TextGrad, LLMLingua, gist tokens) are
**unverified** — treat as leads, not facts.
---

## 2026-07-11 — the idea as played back

> You want representation-space benefits (latent dynamics, style manipulation) without weight
> updates, so you'd build an autoencoder where encoder and decoder are both frozen frontier
> LLMs behind APIs, the latent is *text* (hence natively consumable by any LLM), and the only
> learnable object is the harness — the prompts — optimized by an LLM outer loop against
> round-trip reconstruction, scored by test pass rate.

## The assessment (near-verbatim)

**Bottleneck.** "Without some constraint on the latent representation — length budget, schema
restrictions, forbidding code tokens — the degenerate solution is just the code itself, which
doesn't buy you structure. The interesting problem is really about what constraint makes the
latent meaningful and useful. This feels like a rate-distortion question." "The LLM optimizer
will try to cheat by smuggling the source through any opening I leave, so the constraints have
to be verifiable, not just suggested in the prompt."

**What is actually being built.** "You're not creating embeddings in the vector sense; you're
designing a discrete, human-readable intermediate representation. That's actually a strength
for some of your goals — style transfer, editing, interpretability all work better when the
latent is editable and legible — but it breaks others like interpolation or nearest-neighbor
search that rely on geometric structure." Connects to literate programming and pseudo-code
as IR; encoder extracts specs, decoder synthesizes programs.

**Objective tension.** "The test-pass objective is elegant but it only preserves function;
style is exactly what it's free to throw away" — variable names, idioms, structure are the
first things an optimizer under a length budget discards. Fixes: a style-preservation loss
term, or a factored latent (functional spec field + style field, each with its own
objective), which would also enable style transfer by swapping fields. "That's probably the
biggest gap between the current setup and your stated goals."

**Forward model in latent space.** "An LLM predicting the next latent from the current one,
but the real case for this hinges on whether compression and abstraction make prediction
cheaper than just prompting on raw code."

**Strengths.** Training-free; works with closed models; the round-trip + tests signal is
clean, self-supervised, and scalable; human-readable latents give interpretability and
editability — "you can actually intervene in latent space using natural language."

**Practical costs.** Each optimization step encodes, decodes, and tests across batches, and
LLM-based optimizers compound this; API nondeterminism, temperature sensitivity, and model
version drift; the latent space is implicitly defined by the (model, prompt) pair, so
switching models changes it entirely; comparability across samples depends on format
constraints; the optimizer can overfit prompts to the dev set; test-pass evaluation is
noisy and expensive and prompt optimization plateaus — budget for variance.

**Prior work named (unverified).** Round-trip correctness as an eval metric (attributed to
DeepMind, ICML 2024); LLM-as-optimizer frameworks (DSPy, OPRO, TextGrad); prompt compression
(LLMLingua); gist tokens (skipped as it involves training). Claimed novelty: "treating the
optimized intermediate representation itself as a learned latent space."

## 2026-07-11 — Point 1, the degenerate solution (discussed point by point)

**Danielle's position.** Pure copy-through is not what frozen LLMs do: "if you're using LMs as
the encoder and the decoder, they're actually incredibly strongly biased away from just
copying the input as the output." Much of the code will land in the representation, and
"that's not necessarily a problem depending on what the goal is." Bottlenecks were
introduced classically because nothing else biased the system away from identity; here one
could instead "optimize for pass rate ... but then also add other optimization targets" —
e.g. a contrastive setup making "functionally equivalent, but surface form distinct code
samples match to the same thing." "Assuming that this would behave the same way as the
previous version of ... autoencoders is actually a really large assumption." A smaller
representation is ultimately desirable for practical reasons (storage, cost), but she does
not want to design around a failure mode before there is evidence it is the biggest blocker:
"this domain, while structurally similar to the domain that we're used to, isn't actually the
same. And I think that exploring how same or different it is is actually a big part of why
this would be an interesting question."

**Response — pushback, then concession (near-verbatim).** "The bias away from copying is a
default, not a constraint, and the encoder LLM isn't the optimizer — the outer loop is. LLMs
follow 'transcribe the source verbatim in a fenced block' extremely well, that prompt is
simple and discoverable, and under pure pass-rate it's the global optimum." Predicted
equilibrium is not full copying but "the sneakier version: the optimizer converges on 'write a
spec, plus include the tricky lines verbatim,' and you get pass rate from the smuggled
fragments while the rest provides the appearance of abstraction. Your unoptimized
observations don't tell you much about the optimized equilibrium."

On the contrastive idea: "it's not just 'another target,' it's the anti-degeneracy
mechanism. A verbatim copy is maximally surface-sensitive, so an objective demanding that
surface-distinct, functionally equivalent programs map to the same representation
structurally rules out copying rather than merely discouraging it. ... a bottleneck says
'discard some bits, I don't care which,' while your invariance term says 'discard these
bits.' The two losses brace each other — pass rate prevents collapse-to-generic, invariance
prevents copying. Tests even give you a cheap equivalence oracle for generating pairs."
Operational cost noted: text latents need a similarity metric (LLM judge or embedding
distance) for the invariance term. Revised claim: "you don't need a bottleneck, you need at
least one objective that makes copying non-optimal — and you've named one."

**Sharpened research question.** "What you're really asking is whether prompt-space
optimization is even powerful enough to reach the pathological optima that gradient descent
reliably finds. The classical degeneracy results implicitly assume an optimizer that can
realize nearly any function. Yours is an LLM proposing discrete prompt edits — a weak,
heavily biased search over a tiny subspace. Whether the smuggling equilibrium is reachable
under that optimizer within a realistic budget is genuinely open, and either answer is a
finding ... Designing the constraint in from day one forecloses exactly that experiment."

**Instrument, don't design.** "Log a copy-detection signal (n-gram overlap or
longest-common-substring between input and representation) as the optimizer runs. It costs
nothing, doesn't shape the solution, and converts 'worry about degeneracy' into 'watch a
number.'"

## 2026-07-11 — Point 2, the objective vs. the style goal

**Danielle's position.** The pass-rate objective targets the general-purpose case; style
transfer would build on it. Sketch: a two-part representation (function F, style S) where
"the score ... was based on regenerating with just the functional part versus regenerating
with the functional and style part" — surface-form reconstruction metrics on decode(F+S),
test passing only on decode(F), and possibly "one more loss element, which would be keeping
functionality out of the style representation." Then "ideally, you would be able to swap out
the style representations," and if the optimal representation is human-readable text, "you
would be able to actually just write language into the style portion." No reason to believe
pure optimization yields readable text, but "the general structure makes space for different
types of modifications of the optimization surface." The optimizer is not a semi-random
search: as an LLM it "would have its own priors about what would work ... and would have the
ability to look at the outputs and adapt its prompts accordingly" — so playing with
components may be far more effective than in "a pre-LLM standard autoencoder setup where it
was notoriously very difficult to tune." This is motivation plus illustrative examples of
plausibility and usefulness, not loss design.

**Response (near-verbatim; detail beyond the motivation level, kept as a lead).** The third
term is load-bearing, not optional: "With only the two terms — decode(F) passes tests,
decode(F+S) reconstructs surface — the optimal S is the verbatim code ... The smuggling
problem from point 1 doesn't disappear in the factored setup; it relocates to whichever
channel has no constraint." A function-contaminated S "means a style swap silently swaps
behavior too." Strongest form: "make the swap itself the training objective. Decode(F_a +
S_b) must pass a's tests while exhibiting b's surface characteristics — cross-decoding as
loss, the cycle-consistency trick from classical style transfer ported to prompt space."
Metrics: no ground truth for "a's function in b's style," so style-feature agreement (naming
conventions, comment density, idiom usage) or an LLM judge; "raw edit distance actively
rewards verbatim S," so the reconstruction metric choice is where tuning tension
concentrates. Style is not fully independent of content; some swaps may not transfer
sensibly. (Unverified lineage pointer: MUNIT / cycle-consistency style transfer.)

On the optimizer: "The deep reason classical disentanglement was brutal is that gradient
descent has no concept of 'style' — the factorization had to emerge from blind architectural
pressure, and you tuned β by superstition. Here you can name the factorization ... a
frontier model lands near the right split at step zero. Optimization becomes refinement from
a semantically correct initialization, not search for an unnamed direction. And balancing
the losses becomes conversational." On readability, mild pushback: "prose is closer to the
default than you're granting ... drift toward opaque encodings requires the optimizer to
actively discover them — the same reachability question from point 1. Hand-editable style
fields are probably nearly free."

**Motivation-level summary (agreed).** "The architecture creates space for these variations
(factored representations, swappable components, steerable optimization) in a way the
classical setup never did, and ... an LLM optimizer with semantic priors makes exploring
that space tractable. ... the fact that we could sketch three plausible designs in one
exchange is itself evidence for the plausibility argument."

## Open questions

- Bottleneck: now framed as *optional* and itself an experimental variable — does
  prompt-space optimization reach the smuggling equilibrium at all? Track with a
  copy-detection metric rather than preventing by construction.
- Contrastive / invariance objective: how to generate surface-distinct functionally
  equivalent pairs (LLM rewrites + test oracle), and what similarity metric on text latents.
- Factored (F, S) schema: the third "function stays out of S" term and its metric;
  cross-decoding as objective; readability of S as nearly free vs. needing pressure.
  (Loss design deferred — motivation stage; Points 3+ pending.)

**Waiting on:** the remaining points of the point-by-point discussion; a promotion decision.

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

## 2026-07-11 — Point 3, "this isn't quite an embedding"

**Danielle's position.** "Embedding" is a deliberately handwavy parallel, but LLM input/output
space "probably is actually a lot more structured than just, like, arbitrary whatever" — the
inputs and outputs are token embeddings, so "in some sense they are vectors, though the
vector space is probably not the same shape as, like, a VAE constrained to a unit ball."
Earlier-era embedding-space design existed "to make it practical to actually learn a
non-collapsed space that had different things that clustered near or far away from each
other, whereas by using an LLM ... we're kind of getting the space for free. And the question
is, is this actually a useful space? But at least when used along with LLMs, then it's
clearly in some sense a useful space, and we have tools that are able to extract the
usefulness from the space." Not convinced it is "fundamentally worse"; suspects "digging into
the math and theory of it all, there would be some things that were surprisingly similar
that wouldn't seem so on their surface."

**Response (near-verbatim; concession).** "The right distinction isn't geometry versus no
geometry — it's explicit versus borrowed geometry. Text in isolation has no metric, but text
consumed by an LLM inherits the model's internal semantic space, and interpretability work
(the linear representation hypothesis, SAE features) suggests that space is far more
structured — even surprisingly linear." "Here the space is pretrained; you're not learning a
geometry, you're choosing an addressing scheme into one that already exists." Lineage
(unverified pointers): discrete latent variable models — "VQ-VAE showed that discrete codes
indexing a continuous learned codebook match continuous latents; your setup is that pattern
with text tokens as compositional discrete codes and the frozen LLM as a gigantic pretrained
codebook"; an information-geometry reading where the distance between two representations
is the divergence between the decoder conditionals they induce. "So the parallel isn't
handwavy so much as unformalized."

What remains: "access cost, not existence. Vector geometry is explicit and free ... Here
every geometric operation is mediated by an inference call: interpolation becomes 'blend
these two,' arithmetic becomes 'this, minus X, plus Y' — functional analogs that work but are
stochastic, priced per token, and back a 'metric' with no guarantees (asymmetric, no
triangle inequality)."

**Cheap empirical signature.** "Run the representations through an off-the-shelf embedder,
and if you're right, they should form a better-organized vector space than embeddings of
the raw code — clustering by function rather than surface. Cheap to check, and it converts
this whole dispute into a measurement."

## 2026-07-11 — Point 4, prior art to position against

**Claude's initial map (unverified).** Round-trip correctness (attributed to Allamanis et al.,
DeepMind) — the same code→NL→code loop with tests, used purely as an LLM evaluation metric;
LLM-driven prompt optimization (DSPy, OPRO, TextGrad); prompt compression (LLMLingua). "None
of them treat the optimized intermediate representation itself as the object of interest."

**Danielle's additions (from memory; titles to be resolved).** A nearer cluster in two
pieces. (1) In the coding space: natural-language intermediates for *translation between
programming languages* ("a few different natural language for translation papers") and for
*debugging* — "translating code into natural language, making hypotheses about what might be
going wrong, and then doing rollouts in natural language before moving back into code"; she
recalls "LDB" or similar in the debugging paper's title. (2) A "natural language bottleneck"
paper — not code, but grading student work: bottleneck the student work, use the intermediate
for scoring, found useful. In both, "they didn't really optimize the bottleneck. They just
tried a few different prompts." Summary: "each of the individual pieces has definitely been
done ... but I haven't really seen anything that does all of them together."

**Response (near-verbatim; unverified attributions).** "The useful cut is what each line does
with the intermediate. In the code work (explain-then-translate for cross-language
translation, the self-debugging line where the model narrates hypotheses in NL before
touching code), the intermediate is ephemeral scaffolding: generated, consumed once,
discarded ... The bottleneck-for-grading paper — which is essentially the
concept-bottleneck-model lineage with free text in place of a concept vector — does treat
the representation as the object, but statically ... And the DSPy/OPRO line has the loop but
aims it at task instructions rather than representations. So your gap statement gets crisp:
**everyone uses NL intermediates, some even treat them as representations, but nobody learns
them.**"

Two upshots: "this literature is feasibility evidence, not just positioning: unoptimized
intermediates already carry enough information to translate, debug, and grade. So the bet
isn't 'can natural language hold the information' — that's settled — it's 'does optimizing
the channel buy abstraction and control.'" And the differentiating claims: "that an
optimized representation beats a sensibly hand-prompted one (otherwise it's
explain-then-translate with extra steps), and that one representation serves multiple
downstream uses (otherwise it's scaffolding, not a representation). Those two comparisons
are the entire defense of the gap."

## 2026-07-11 — Two interest categories: representation exploration vs. beating lossless compression

**Danielle's framing.** Her own interest: "exploring what types of representations you can
make and what you can do with them, and how you can control them." Left alone she would
study whether a representation can be *broadly effective* — robustness across task types
with code input, and across models: "could you actually optimize the encoder prompt such
that you get a representation that you could pass into any level of model and get high
quality output from it, or even could you hyper-optimize for a specific model? ... the
smallest representation that would give a high pass rate on a very specific model." The
lab's chosen direction instead: **can lossy functional compression beat lossless
compression** — this approach as the lossy stage, existing lossless algorithms applied to the
intermediate, exploiting that for code "we care about preserving functionality not surface
form," so the lossy portion only needs to reconstruct functionality. "It uses the same
structure, and it does give us a very crisp optimization metric, but it also feels kind of
qualitatively different ... at the same time, I think that by pursuing [that] goal, I can
learn things about my goal."

**Response (near-verbatim).**

*What the compression framing buys.* "A brutal scalar with unambiguous baselines — you
either beat zstd (and the stronger baseline, LLM-plus-arithmetic-coding à la 'Language
Modeling Is Compression') or you don't. It's also theoretically clean: lossless is bounded by
the entropy of the surface string, while [this] scheme targets the entropy of the functional
equivalence class, and the gap between those bounds is a real quantity." Callback to Point 1:
"under a bits objective, verbatim smuggling stops being a cheat and becomes a priced
tradeoff — copying the tricky lines now costs bits. Compression is the one objective that
contains its own anti-degeneracy term; the bottleneck you resisted designing in by hand
becomes the research question itself." Application: semantic compression of code where only
behavior matters (fitting more code into a context window).

*What it costs.* "The objective is actively hostile to everything you personally find
interesting — readability, editability, factored structure, cross-model portability are all
bits, and bits are the enemy, so the optimal artifact is dense, cryptic, and
decoder-specific." Guarantee semantics differ: "gzip never lies, while lossy-functional
compression is test-relative (silent behavior change outside the suite) and stochastic and
expensive at decompression time." Headline risk: "equivalence classes are so large that 'beat
gzip' may read as unsurprising, with the real interest buried in the details."

*Why they are the same project.* "It's one rate–distortion plane viewed from two axes: [one]
pins the distortion criterion (pass tests) and minimizes rate; you want to roughly fix rate
and explore the distortion axis — which properties survive, which can be steered. And you've
already converged more than your framing suggests: your 'smallest representation with high
pass rate on one specific model' is [the compression] question, verbatim. [The compression]
metric doubles as your instrument, too — bits saved relative to lossless is exactly the
information content of surface-given-function, i.e., a measurement of how much of code is
style, the very quantity your factored-representation ideas would manipulate."

*Where they diverge.* "The artifact. [The compression-optimal] representation and yours are
different objects, so qualitative findings about [one] (what it looks like, whether it's
readable or structured) won't transfer. What transfers ~100% is the machinery — harness,
optimizer loop, test-based evaluation, equivalence-pair generation — plus positioning shifts
from the NL-intermediates literature to the LLM-compression one." Sequencing: the compression
project builds all the infrastructure and earns credibility on a crisp claim; its natural
ablations — bits vs. readability, model-specific vs. portable — are the representation
research agenda.

Baseline note (unverified): "beat lossless" should be defined against LLM arithmetic coding
(Delétang et al., "Language Modeling Is Compression"; LLMZip), not only gzip/zstd.

## 2026-07-11 — Compression project: implementation state and initial experiment design

**State as of 2026-07-11 (Danielle).** A solid implementation of the inner loop exists:
encode → decode → test evaluation → compression evaluation of the intermediate, runnable
against different models. The lab's decision is to use **HumanEval exclusively** — "such an
easy task that it's been a little bit hard to find models that perform poorly on it even with
fairly low budgets." Current scheme: decoder uses the original HumanEval prompt (generate a
function matching this description); encoder baseline is the "describe this code" prompt from
a HumanEval-explain-style paper ("obviously a pretty bad prompt for our goal, but it was the
baseline"). The system swaps prompts and sweeps a **length budget** ("do this in less than n
characters") to get a distribution over length vs. correctness; going over budget is not
counted as failure — the budget instruction is "a way that a prompt can encourage different
levels of trade-off between correctness and compression." Ready to start optimization of the
**encoder prompt only**, with DSPy-derived algorithms (not DSPy itself — "not optimal for my
setup"), starting from COPRO (sample prompts, score, resample).

**Three concerns she raised.** (1) Most HumanEval problems pass 100% (some 0%) — bad examples
for prompt optimization; subset to non-saturated problems, which is needed anyway since
optimizing on the full eval set is too expensive and improper. (2) Variance is high even at
temperature 0; the right number of samples per optimization round is unknown. (3) "I know for
sure that frontier language models can optimize these prompts really well. I'm not sure that
the structure that COPRO is using would produce that" — so a mini-test of the outer-loop LLM
on a simpler, more direct optimization problem (e.g. tune the number of samples until an
estimate converges) before using it on the real problem. Aside on the compression goal
itself: "in the theoretical sense, it should be possible. But, practically, I think that
that's maybe not what I would assume LLM-style optimization would be best at."

**Response (near-verbatim).**

*Steps 1 and 2 are one experiment.* "Before optimizing anything, run one generous census:
baseline prompt, all 164 problems, 10+ samples each, at each budget tier. That single
artifact gives you the per-problem difficulty profiles for subsetting, the empirical variance
floor for sample-size math, and your baseline Pareto curve." On subsetting: "beware
regression to the mean (a problem measured at 60% with five samples may be a true 100%, so
characterize with more samples than feels necessary), audit the 0% problems before
discarding them (some will be harness or test bugs, some are genuine headroom worth keeping a
few of), and freeze the held-out set now."

*Sample count is a power calculation, not a search.* "You need the standard error of a score
difference between candidate prompts to be smaller than the gaps you care about detecting —
early rounds have big gaps and tolerate few samples, late rounds don't." Multipliers: paired
evaluation on identical problems; caching encoder outputs; adaptive allocation — "screen
candidates cheaply, spend samples on survivors, one high-n validation of the final top-k
(racing/successive-halving logic)."

*The pilot instinct is right; the testbed is wrong.* The tune-n-samples toy "doesn't
exercise the one skill you actually doubt — proposing good prompt edits from feedback; it's
scalar estimation, a different genre — and a strong model may just do the power calculation
analytically and stop." Better pilot: "a miniature of the real loop: tightest budget tier
(where even strong models fail — your headroom problem solves itself under compression
pressure), 10–15 problems, low n, cheap inner models, strong model driving with full
transcripts." Also: "playing optimizer yourself for two or three rounds first: whatever
information you need to improve the prompt is exactly what the automated loop must be
shown."

*Quirks to expect.* "Under noise, LLM optimizers confidently narrate causal stories about
score differences that are pure noise — so your variance measurement determines what you're
allowed to show it. Surface only differences above the noise floor, or print explicit
intervals ('64% ± 9')." Proposal mode collapse ("COPRO's known failure: later candidates
become paraphrases of the incumbent"); overfitting to the training subset — validate on
held-out, early-stop the optimization itself.

*Structure the real run as arms; close one design gap first.* Arms: COPRO as floor;
OPRO-style trajectory-in-context; a diagnosis-driven arm that sees failure transcripts
(ProTeGi/TextGrad-style, or fully agentic with the harness as a tool) — equal budgets, shared
validation; "that comparison is a finding in itself, and it's your point-1 reachability
question in production form." The gap: "decide what scalar the optimizer sees. You measure
real length but only request it softly, so the honest objective is the measured (pass,
length) pair — optimize per budget tier or pick an explicit scalarization." And decide
pass@1 vs. pass@k now — best-of-k needs a decode-side selector, which changes variance and
the compression claim.

*Strategic flag.* gzip/zstd are weak on tiny inputs (HumanEval functions are 5–15 lines);
the convincing baseline is LLM + arithmetic coding, and bit accounting must decide how to
count the shared decoder prompt.

## 2026-07-11 — Sequencing the first measurements; the cliff structure

**Danielle's decisions and observations.** pass@1 is the metric — cost-wise the only one that
makes sense, and "language models are so good that, at this point, what are we doing if
we're not talking about pass@1, for simple coding tasks anyways." Budget is now a
**compression ratio against the input code** rather than a fixed character count: the input
is the HumanEval solution with the docstring and comments stripped (the docstring "is
basically the prompt for the original HumanEval task"); the target ratio is expressed as a
token count computed per sample. Open question on how much **extraction effort** to spend
normalizing LM output — "you could consider extraction to be part of the decompression
algorithm." Empirically, with the cheapest OpenRouter models: ~95% pass rate (worst ~80%)
down to a threshold below which models "will just say, I can't do that, that's too small" —
"a very step-function jump between feasible and zero percent," with feasible meaning very
high pass rate. Sequencing questions: intensive sampling for the statistics vs. a coarse
then fine budget sweep; and optimizer order — each additional optimizer is implementation
time instead of results, so start with something easy to implement, analyze it, and build
intuition.

**Response (near-verbatim).**

*The cliff dissolves the sequencing dilemma.* "With pass/fail outcomes, variance isn't a
separate thing to measure — it's fully determined by the pass probability: σ² = p(1−p),
maximal at p = 0.5, near zero at the extremes. Your step function means almost every
(problem, ratio) point sits at an extreme, so global intensive sampling would spend most of
its budget confirming 0s and 100s. So the order is: coarse ratio sweep at tiny n (2–3) just
to bracket each problem's cliff, then concentrate samples only inside the brackets." Closed
form for n: "worst-case SE is 0.5/√n, so n = 25 buys ±10 points per estimate, n = 100 buys ±5
— needed only near cliffs."

*Measure cliffs, not curves.* "The informative per-problem statistic is the critical ratio —
the threshold where reconstruction starts failing — found by bisection at log cost rather
than a dense grid. The distribution of critical ratios across problems is your landscape
map: it tells you where tiers should sit and which problems carry training signal."

*Tier placement.* "One mid tier where baseline passes ~30–60% (maximum headroom and
sensitivity), one aggressive tier below most cliffs where baseline is ~0%. The zero tier
isn't wasted, because your zeros are refusals — compliance failures, not capability failures
— and compliance is the first thing prompt optimization fixes." Corollary: "the cliff is
prompt-dependent, so the baseline prompt's cliff will move the moment optimization starts —
map it coarsely for the paper, but do fine mapping under a reasonable prompt."

*Split the zero now.* "Log each sample as {refused, extraction/format failure, doesn't run,
tests fail, pass} rather than binary. Nearly free, it tells you how much of the cliff is
refusal versus genuine information insufficiency, and that taxonomy is precisely what the
diagnosis-driven optimizer arm will consume later."

*Extraction-as-decompression is principled under two conditions.* "Deterministic and frozen:
string munging is part of the codec spec; anything model-based becomes decoder capability
leaking into the harness and muddies attribution. Expect some early optimizer 'gains' to be
format-compliance wins — legitimate, but the taxonomy makes them visible as such."

*Simplest-first is right, and it's not throwaway.* "Blind propose–evaluate–select is a
required arm of the eventual optimizer comparison — the null that diagnosis-driven methods
must beat — so building it first is collecting results, not deferring them ... its second job
is generating the failure transcripts that define what a smarter optimizer needs to see."
Flag: "token ratio is the right knob now, but the claim is ultimately in bits, so log
zstd-compressed sizes of representation and original alongside — token orderings don't
always survive the conversion" (code compresses well; dense natural language may not).

*Composition.* "Aggressive tier × 10–15 bracket-mapped problems × simple optimizer at low n is
exactly the miniature pilot from before — your first real optimization run and the pipeline
shakedown in one."

## 2026-07-11 — Baseline vs. seed: where to start the optimization

**Danielle's decision point.** The lab's constraints: HumanEval (agreed suboptimal — "the
original code is so short that, what are we even really measuring?") and the original
HumanEval-explain "describe this code" prompts as the baseline. Her reading: optimization
should therefore start *from* those prompts. But "the task was to describe in natural
language and then reconstruct. That's not actually what we're doing, so it seems strange to
start an optimization for compression from a 'describe this function' prompt instead of from
a 'provide a representation that would let another model reconstruct' prompt or 'compress
this function as much as possible' prompt." Two worries: it seems silly; and "by starting
with a prompt that doesn't have anything to do with optimization, I'm worried that it'll
bias the outer LLM towards exploring things in the space of the initial prompt, as opposed to
... the types of prompts that actually might optimize performance."

**Response (near-verbatim).**

*Disentangle two roles.* "A baseline is a comparison point — the number your final result is
reported against. A seed is an optimizer hyperparameter. [The constraint] mandated the
first; the second [says] nothing, and the optimized-vs-baseline comparison is equally valid
wherever the trajectory started."

*The attribution argument for seeding from the baseline.* "If you seed with your own
task-aligned prompt, the final score is part your manual prompt engineering, part the
optimizer's search, and you can't decompose them. Seeded from the naive baseline, the entire
delta belongs to the loop: 'we took prior work's prompt and optimization took it from X to Y'
is the attribution-clean headline. ... though [a] misaligned seed burns early budget
rediscovering the task, which means it understates the method's ceiling. Those are two
different measurements: delta-from-baseline measures what optimization contributes;
aligned-seed measures what the method can achieve."

*Anchoring lives mostly in the meta-prompt, not the seed.* "Proposals condition on
incumbents, and COPRO-style loops drift into paraphrases of whatever they're shown. But the
optimizer's goal comes from the meta-prompt you write ... if it plainly states 'propose
encoder prompts that maximize reconstruction pass rate under budget R; scores below,' the
loop knows description-quality isn't the objective no matter what the incumbent looks like."
Second escape force at the aggressive tier: the baseline scores ~0 via refusals, pushing
proposals out of description-space — "with the caveat that a flat-zero incumbent provides no
differential signal, so whether the optimizer makes the conceptual leap or flails is itself
worth watching."

*Make initialization an arm rather than an anxiety.* "Run both seeds, equal budgets, shared
validation. Convergence to similar prompts and scores means your search escapes its basin —
the reachability question from point 1, answered empirically. Divergence means
initialization sensitivity is real, which is both a reportable finding and the justification
for the diagnosis-driven arm later."

## Open questions

- Bottleneck: now framed as *optional* and itself an experimental variable — does
  prompt-space optimization reach the smuggling equilibrium at all? Track with a
  copy-detection metric rather than preventing by construction.
- Contrastive / invariance objective: how to generate surface-distinct functionally
  equivalent pairs (LLM rewrites + test oracle), and what similarity metric on text latents.
- Factored (F, S) schema: the third "function stays out of S" term and its metric;
  cross-decoding as objective; readability of S as nearly free vs. needing pressure.
  (Loss design deferred — motivation stage.)
- Borrowed geometry: formalize the discrete-latent / information-geometry reading; run the
  embed-the-representations check (function-clustering vs. raw-code embeddings). 
- Prior-art map needs a real literature pass: resolve the NL-for-code-translation papers,
  the LDB-like debugging paper, the NL-bottleneck-for-grading paper, round-trip correctness,
  concept bottleneck models. Differentiating claims: optimized > hand-prompted baseline;
  one representation serves multiple downstream uses.
- Two framings on one rate–distortion plane: compression (fix distortion = tests, minimize
  bits; baseline = LLM arithmetic coding) vs. representation exploration (fix rate, explore
  controllable distortion; cross-model portability vs. per-model hyper-optimization). Decide
  how the compression project's ablations are designed to also answer the representation
  questions.
- Compression experiment design: census run (all problems × budget tiers × ≥10 samples) →
  subset + frozen held-out + power calculation; miniature-loop pilot instead of a toy
  optimizer test; optimizer arms (COPRO / OPRO-trajectory / diagnosis-driven); the scalar the
  optimizer sees (per-tier vs. scalarization); pass@1 vs. pass@k; baseline = LLM arithmetic
  coding with honest bit accounting. Decided: pass@1; ratio-vs-stripped-input budget.
- Cliff-based measurement plan: bisection for per-problem critical ratios; tiers at ~30–60%
  and ~0% (refusal) baseline; five-way outcome taxonomy; frozen deterministic extraction;
  blind propose–evaluate–select first; log zstd bytes alongside tokens.
- Initialization: primary arm seeds from the mandated HumanEval-explain baseline
  (attribution-clean delta); second arm seeds from a task-aligned prompt (method ceiling /
  initialization-sensitivity check); the true objective is stated in the meta-prompt either
  way.

**Waiting on:** the remaining points of the point-by-point discussion; a promotion decision.

# Text-latent code autoencoder — optimizable text representations of code with frozen frontier LLMs

> **Draft scaffolding (2026-08-22).** Promoted from a staging topic. §1 and §3 are a
> lightly edited version of the structured write-up produced at the end of the 2026-07-11
> discussion (first-person voice is Danielle's); §2 is synthesized scaffolding not yet
> reviewed by Danielle; §4 is the dated discussion record. Treat §2 as provisional until this
> note is removed.

**Program pillars served:** none — this project sits outside the DataDecide program and its
three pillars; it gets its own topic proposal if pursued as a thesis direction. (Program:
`README.md` → Program.)

**One-line pitch.** An autoencoder whose encoder and decoder are frozen frontier LLMs behind
APIs, whose latent is *text*, whose reconstruction signal is the test pass rate of the decoded
program, and whose only learnable object is the harness (prompts) optimized by an LLM outer
loop. Three layers (decided 2026-08-22): a **measurement suite** (TLC-0) that scores any
representation for retained behavioural information and leaked implementation detail
relative to a declared extractor family; the **compression project** (TLC-1–TLC-3) — beat
lossless compression of code by composing lossy *functional* compression with standard
lossless compression of the intermediate; and the **representation program**
(TLC-opt-1–TLC-opt-4) — what representations can be induced this way, how robust and
controllable they are, and whether function and style can be factored. The compression
project and the representation program are **separate projects** sharing the harness and
the measurement suite; they are not stages of one paper. Danielle's primary research
interest is the representation program; the compression project is the track being pursued
for the thesis.

IDs: TLC-0 (measurement suite), TLC-1–TLC-3 (compression phases), TLC-opt-1–TLC-opt-4
(representation program).

**Paper goal.** A crisp compression result (TLC-1–TLC-3) is the compression project's
paper — workshop-sized if the claim holds only against raw-source zstd-class baselines,
main-conference if it holds against the strongest fixed-prior baselines (trained
dictionaries, AST stream codecs, LLM arithmetic coding) with honest bit accounting. The
representation program (TLC-opt-*) is its own project with its own paper: what a
search-optimized text representation of code retains, leaks, and can be made to do. TLC-0
is the instrument both report with; it may also carry a measurement paper on its own.

Compute tiers: **API** = frontier/cheap-model inference via OpenRouter only; no training
anywhere in this project.

---

## 1. What the project involves

### Motivation and core idea

Learned representations changed what was possible in other domains: given a good latent
space, you can forward-model in that space, factor out attributes, and manipulate objects by
manipulating their representations. I want those affordances for code — forward modeling of
program evolution in latent space, capturing qualitative aspects of coding style separately
from function to enable style transfer, and similar.

Two constraints shape everything. First, the capability worth leveraging lives in frontier
closed models, accessible only through APIs. Second, I want no weight updates anywhere —
training is expensive, slow, and hard, and it forfeits access to the true frontier. This
looks like an impasse: even a perfect vector embedding of code would be useless to a frontier
model, which consumes language, not vectors.

The resolution is to notice what actually makes something an embedding: a representation of
an object in a different (usually compressed) form, learned against a reconstruction
objective. Nothing in that definition requires a vector. So the proposal is an autoencoder in
which the encoder is a frozen frontier LLM plus a harness that maps a code sample to a text
representation; the decoder is a frozen LLM plus harness that maps the representation back to
code; the reconstruction signal is the test pass rate of the decoded program against the
original's tests; and the only learnable object is the harness — in the simplest case, the
prompts — optimized by an LLM in an outer loop. The resulting representation is natively
consumable by any LLM, because it is text, and plausibly controllable through
natural-language instructions to the encoder.

### What the representation is (and is not)

This is not an embedding in the classical sense: it is discrete, human-readable, and carries
no explicit metric — no distances, interpolation, or arithmetic for free. But the geometry is
borrowed rather than absent. A text representation consumed by an LLM inherits the model's
internal semantic space, which interpretability work (linear representation results, SAE
features) suggests is far more structured than arbitrary. The relevant lineage is discrete
latent variable models: VQ-VAE showed discrete codes indexing a learned codebook can match
continuous latents, and this setup is that pattern with text tokens as compositional
discrete codes and the frozen LLM as an enormous pretrained codebook. Where classical
VAE-era engineering existed to learn a non-collapsed, well-organized space from scratch,
here the space is pretrained and the question is whether a text addressing scheme into it is
useful.

What remains true is that access costs differ: vector geometry is explicit and free, while
every geometric operation here is mediated by an inference call — functional analogs (blend
these two; this, minus X, plus Y) that work but are stochastic, priced per token, and backed
by no metric guarantees. A cheap empirical probe exists: re-embed the text representations
with an off-the-shelf embedding model. If the structured-space intuition is right, they
should form a better-organized vector space than embeddings of the raw code — clustering by
function rather than surface.

### Why the latent is natural language — a prediction, not a requirement

*(Adopted into §1 by decision 2026-08-23; provenance and full derivation in the §4
entries for the February-2026 conversation, chunks 12–14.)*

The most common objection to this design is: if the objective is a behavior-preserving
compressed representation, why should the latent be natural language rather than any
compressed form? The answer is a constraint, not a taste:

> **Because the encoder and decoder are black-box, pretrained LLMs, the intermediate
> representation must live in a region of representation space where those models
> already have strong priors.**

("Representation space" here means input token sequences with training-distribution
support, not embedding space.) With no retraining, no tokenizer changes, and
general-purpose decoders, the only shared, high-capacity, semantically grounded channel
is natural language plus code; a custom DSL, symbolic encoding, or novel markup has no
prior support and fails at decode time.

**Crucially, this is a falsifiable prediction, not a design requirement.** The project
does not constrain the representation to natural language; it predicts that the
best-performing intermediate representations will be some form of human-interpretable
text, because of the training data — and therefore treats NL-instruction inductive
biases in early experiments as probability-of-success maximization, not dogma. Two
experiments test the prediction directly: the abstraction-vs-compatibility ladder
(increasingly aggressive / less NL-like representations at matched budget; predicted
inverted-U — the COMP-NL vs. COMP-SHORT contrast is already a two-point sample), and
cross-decoder transfer of the representation (shared priors are what force NL; an
idiosyncratic code learned for one decoder pair that fails to transfer *confirms* the
mechanism). Cross-decoder decodability is accordingly an emergent evaluation axis, not
an assumption: hierarchy = constraint (high-prior regions) → prediction (NL-like) →
emergent benefit (cross-model transfer).

The working vocabulary that goes with this framing: equivalence is
**property-indexed** — "we define program equivalence relative to a set of observable
properties (functional behavior, complexity class, and effects), and treat programs
equivalent under these properties as interchangeable for downstream learning and
analysis." The pipeline performs **behavior-preserving normalization** — "rather than
preserving exact program structure, we abstract programs into equivalence classes
defined by behavioral properties and operate on representative implementations to
reduce representational variance." (Not "canonicalization," which formally requires a
unique, deterministic, total, idempotent representative.) Closest formal ancestor:
Miao & Blunsom 2016 (1609.07317), whose background-LM prior on a latent summary
sentence is the trained-model counterpart of the frozen-prior constraint above.

### Degeneracy: an open question, not a design constraint

Classical autoencoders require bottlenecks because an unconstrained optimizer reliably finds
the identity map. Two things differ here. The encoder's default behavior is strongly biased
away from copying — initial experiments confirm paraphrase, not copy, though substantial
surface content does flow into the representation, which may or may not matter depending on
the goal. And the optimizer is not gradient descent: it is an LLM proposing discrete prompt
edits, a weak, heavily prior-laden search over a small subspace. Whether smuggling
equilibria — "write a spec, plus include the tricky lines verbatim" — are even reachable
under such an optimizer at realistic budgets is genuinely open, and either answer is a
finding: convergence there means prompt optimization is stronger than it looks; failure to
converge means this domain provides anti-degeneracy for free. Designing constraints in from
day one would foreclose exactly that experiment.

Policy: instrument rather than design. Copy-detection signals (n-gram overlap, longest
common substring between source and representation) are logged on every run, converting the
worry into a number to watch. Two further notes. The compression objective below prices
smuggling naturally — verbatim fragments cost bits — so the compression project contains its
own anti-degeneracy pressure. And for the broader program, the principled alternative to a
blind bottleneck is an invariance objective: requiring surface-distinct, functionally
equivalent programs to map to the same representation structurally rules out copying rather
than merely discouraging it, and tests provide a cheap equivalence oracle for generating
such pairs.

### Three layers: one instrument, two separate projects

**The measurement suite (TLC-0): what a representation retains and what it leaks.** Length
and Shannon mutual information cannot tell compression-by-abstraction ("returns the number
of unique elements") from compression-by-minification (`def f(x):return len(set(x))`);
both are short and behaviour-preserving, only the second carries implementation identity.
The instrument is predictive 𝒱-information: information usable by a *declared* extractor
family 𝒱 (frozen LMs, prompted probes, decoders, judges), so that the encoder and decoder
already knowing programming is a property of the measurement rather than something the
representation must carry. Two quantities per representation Z, against context C
(signature, imports, type hints, template) and behavioural facts B (test signature, edge
cases, exceptions, mutation, algebraic properties): **retention** I_𝒱(Z→B | C) normalized
by I_𝒱(X→B | C), and **leakage** I_𝒱(Z→S | B, C) for implementation facts S after
controlling for behaviour. Instantiated as decoder-pass probes (K decodes, pass
probability), question–answer probes, a contrastive game with behaviour-defined positives
and four negative types, and MDL probes, always with control tasks, and always on the
condition matrix ∅ / signature / Z / minified X / X / oracle spec. The minified-source
column is the control that makes "abstraction, not shortening" a measured claim. TLC-0 is
the decodable-information-bottleneck stance made operational and is what both projects
below report with (§4 2026-08-22, extractable information; measurement bibliography).

**The representation program (TLC-opt-\*): fix rate loosely, explore the distortion and
property axis.** This is a separate project from the compression project below. What kinds of representations can be induced, and what can be done with
them? Concretely: robustness of a single representation across downstream tasks that take
code as input (TLC-opt-1); robustness across decoder models — one representation any model
decodes well — versus hyper-specialization, the smallest representation a specific model
decodes at high pass rate (TLC-opt-2); controllability through natural-language instruction
to the encoder (TLC-opt-3); and the flagship future direction, factored representations
(TLC-opt-4) — separate function and style components trained with reconstruction on the
function field, surface reconstruction on the combined fields, an anti-leakage term keeping
function out of the style field, and ideally cross-decoding as a direct objective (decode
F_a + S_b must pass a's tests while exhibiting b's surface characteristics). That would make
style transfer a field swap, with the further possibility of optimizing the style field
toward human-readable, hand-editable text. The deep advantage over classical disentanglement
is that the factorization can be named in the prompt rather than emerging from blind
architectural pressure, and loss balancing becomes conversational rather than hyperparameter
search.

**The compression project (TLC-1–TLC-3): fix distortion, minimize rate.** A separate
project. The claim: beat lossless compression by composing lossy functional compression
(this system) with standard lossless compression of the intermediate representation.
Because the domain is code, we care about preserving functionality rather than surface
form, so the lossy stage targets the entropy of the functional equivalence class rather
than the entropy of the surface string — and the gap between those two bounds is a real
quantity. In information-bottleneck terms the objective is min I(X;R) − β I(R;Y_T) with the
test-suite signature Y_T as the relevance variable, operationalized as rate = description
length of the representation under a declared LM, distortion = −log of the smoothed
K-sample pass probability, plus a leakage penalty (§4 2026-08-22, IB reading). The metric
is a brutal scalar with unambiguous baselines, organized as a layered per-sample suite:
representation-length controls; standalone compressors up to the slow ceilings (PPMd, PAQ,
cmix, NNCP); fixed-prior baselines — trained dictionaries per representation, JSZap-style
AST stream codecs, nearest-reference deltas; and LLM arithmetic coding — with prior
fairness and selector cost paid, every baseline labelled by its correctness level
(byte-exact → source-equivalent → runtime-equivalent → test-equivalent), and the headline
stated against the strongest fixed-prior baseline (§4 2026-08-22, baseline suite;
classical precedents). Acknowledged costs: the objective is hostile
to readability, editability, and portability, since all are bits; the guarantee is
test-relative rather than absolute, and decompression is stochastic and expensive; and
prompt-space optimization is not obviously the tool best suited to squeezing bits, which
makes the project ambitious in a specific way.

**Sequencing.** The compression project runs first. What carries that: the harness,
optimizer loop, test-based evaluation, and equivalence tooling transfer nearly completely;
its claims are crisper and its risk lower; and its natural ablations (bits versus
readability, model-specific versus portable) are representation-program questions arriving
as analysis. What no longer carries it: the earlier argument that the compression metric
*is* the representation program's instrument — bits saved relative to lossless as the
information content of surface-given-function. With TLC-0 that quantity is measured
directly, as leakage relative to a declared extractor family, on any representation,
without first winning a compression race; so the representation program does not depend
on the compression result, only on the shared harness and TLC-0. The caution stands: the
compression-optimal representation and the representation program's artifacts are
different objects, so qualitative findings about one will not transfer — the machinery
and the measurements will.

### Positioning against prior art

Related-work attributions below are unverified leads pending a literature pass; see §4.

Four nearby clusters, distinguished by what each does with the intermediate.
Natural-language intermediates in code — explain-then-translate for cross-language
translation, self-debugging lines where the model reasons in NL before returning to code —
treat the intermediate as ephemeral scaffolding: generated, consumed once, never measured or
optimized. NL-bottleneck representations (Language Bottleneck Models, Berthon & van der Schaar 2025,
arXiv 2506.16982 — a frozen decoder grading student responses from an NL summary;
effectively the concept-bottleneck lineage with free text) treat the representation as
the object but statically — a few hand-written prompts, no optimization loop, and a
non-verifiable target (Danielle's read, 2026-08-22). LLM-driven prompt optimization
(DSPy/COPRO, OPRO, ProTeGi, TextGrad) has the loop but aims it at task instructions rather
than representations. Round-trip correctness work uses the exact code-to-NL-to-code loop
with tests, but purely as an evaluation metric for models. The LLM-as-compressor literature
supplies the compression baselines.

The gap statement: everyone uses NL intermediates, some treat them as representations,
nobody learns them. This literature is also feasibility evidence, not just positioning —
unoptimized intermediates already carry enough information to translate, debug, and grade,
so the open bet is whether optimizing the channel buys abstraction and control. The
differentiating claims to defend: an optimized representation beats a sensibly hand-prompted
one (else this is explain-then-translate with extra steps), and — for the broader program —
one representation serves multiple downstream uses (else it is scaffolding, not a
representation).

### Current status (as of 2026-07-11)

The inner loop is implemented: encode → deterministic extraction → decode → test evaluation →
compression accounting, model-agnostic via OpenRouter. Fixed choices: HumanEval as the
dataset; the HumanEval-Explain "describe this code" prompt as the baseline. The budget is
expressed as a compression ratio against the source with docstring and comments stripped
(the docstring being effectively the original task prompt), converted to a per-sample token
target. Budgets are soft — overage is scored rather than failed — so budget-pressure phrasing
functions as a knob on the correctness–compression tradeoff. Extraction is deterministic
string processing, frozen as part of the codec specification; anything model-based would
leak decoder capability into the harness and muddy attribution. HumanEval's `check()` has
been split into independent per-test outcomes (pending a two-minute audit that the split
tests are order-independent, with no shared mutable state).

Observed landscape: the cheapest OpenRouter models sit at roughly 80–95%+ pass at generous
ratios, with a sharp step-function cliff to zero below a threshold ratio — and the zeros are
refusals ("that's too small"), i.e., compliance failures rather than capability failures.
This step structure drives the census design in §3.

### TLC-0 candidate metrics from the February-2026 conversation (findable, not decided)

*(Consolidated by decision 2026-08-23. These are candidates for the measurement suite,
recorded here so they are findable when TLC-0 implementation resumes — none is adopted
into the suite's agreed design yet. Derivations in §4, chunks 10–14.)*

- **IR-distance** between source f and reconstruction f̂ (bytecode n-grams, CFG
  fingerprints, def-use chains, normalized opcode sequences, or runtime traces on
  canonical inputs): detects algorithm drift ("passes tests but changed strategy") and
  directly operationalizes implementation leakage — IR-close reconstruction means the
  description carried implementation, not just behavior.
- **Q(d) = pass_rate(d) − λ·variance(d) − µ·cost(d)** — a grounded scalar ordering on
  representations requiring no training.
- **S_multi(d)** — mean pass probability across M decoders; cross-decoder agreement as
  the shared-prior-space probe (the NL prediction restated as a per-representation
  metric).
- **Equivalence-class geometry:** class consistency, intra-class variance,
  representative quality, abstraction error; plus per-test outcome-vector similarity
  (cosine/Jaccard/KL over per-test pass probabilities) as the soft, in-expectation
  equivalence label — extends the validated fractional-test-pass machinery; never
  pass-all-or-none.
- **Graded property oracles:** P = {tests} → {tests, complexity class} → {tests,
  complexity, purity/effects}; complexity properties are free on synthetic items
  (provenance) and tool-assisted elsewhere (BigO(Bench)-style inference, noisy).
- **Deterministic-canonicalizer conditions** (surface canonicalization → structural
  rewrites) as the style-only information floor the NL bottleneck must beat; also
  separates stylistic from semantic variance when applied to decoder outputs.

## 2. Doability and impact

### Overall doability: **high** for Phases 1–2 (inner loop exists; API-only), **medium** for the headline claim

The harness, per-test evaluation, and budget sweep already run. The census (TLC-1) and the
simple optimizer (TLC-2) are bounded API spend on cheap models. Risks to the headline:
HumanEval functions are tiny, so zstd-class baselines are weak on them and the claim must be
made against LLM arithmetic coding with explicit bit accounting (including how the shared
decoder prompt is counted); the guarantee is test-relative; OpenRouter provider
nondeterminism must be pinned; and prompt-space optimization may plateau before the bits
claim clears the strong baseline.

### Per-direction impact

- **TLC-0 (measurement suite).** Retention and leakage numbers against a declared 𝒱 are
  the figures every other direction reports; the minified-source control and the
  contrastive game with behavioural negatives are publishable as a measurement
  contribution on their own if the representation program is the paper.
- **TLC-1–TLC-3 (compression).** A clean "beat lossless" result on code with a
  functional-equivalence distortion is a workshop paper on its own; the optimizer comparison
  (TLC-3) adds a reachability finding about prompt-space search that reviewers in the
  prompt-optimization line will care about.
- **TLC-opt-1/2 (cross-task, cross-model robustness).** Tests whether the representation is
  a representation rather than scaffolding; the per-model hyper-optimization variant is the
  compression question restated.
- **TLC-opt-3 (controllability).** Cheap once the harness exists; mostly demonstrations.
- **TLC-opt-4 (factored function/style).** Highest ceiling and highest risk; the third
  anti-leakage term is load-bearing and its metric is unresolved (§4).

## 3. Infrastructure build sequence

### TLC-0 — Layer 0: the measurement suite

Built alongside the census and before any optimizer claim. Components: (a) a declared,
frozen extractor family 𝒱 per experiment — model, prompt set, K — published with every
number; (b) behavioural targets B as an information profile (test signature, edge cases,
exceptions, mutation/side effects, algebraic properties, complexity facts) and
implementation targets S (identifier choice, AST shape, library choice, formatting,
exact-source reconstruction probability log p_D(x | r)); (c) the condition matrix ∅ /
signature only / Z / minified X / X / oracle spec, with retention and leakage reported per
condition; (d) probes: decoder-pass (K decodes → smoothed pass probability),
question–answer, the contrastive game (positives = different implementation, same tests;
negatives = similar code / different behaviour, same description / different edge case,
same tests / hidden difference) run as Z→B and Z→S, and MDL probes; (e) control tasks —
shuffled behaviour labels, random implementation identities — from day one. Data:
HumanEval/MBPP as the sandbox; CodeNet's many-implementation clusters with I/O tests as the
source of behaviour-equivalent positives. Every census representation and every optimizer
candidate gets TLC-0 numbers; the optimizer is not allowed to report without them.

### TLC-1 — Phase 1: the census

Purpose: establish where optimization has signal and what sampling buys, before optimizing
anything. Binary outcomes make variance mean-determined (σ² = p(1−p)), so intensive sampling
everywhere would spend most of its budget confirming zeros and ones. Structure: a coarse
ratio sweep at n = 2–3 per point to bracket each problem's cliff, using bisection to locate
critical ratios at logarithmic cost, then concentrated high-n sampling only near the cliffs.
Worst-case standard error is 0.5/√n — n = 25 buys roughly ±10 points, n = 100 buys ±5 — and
is needed only where pass probabilities are interior. The distribution of critical ratios
across problems is the landscape map and sets tier placement: a mid tier where the baseline
passes roughly 30–60% (maximum sensitivity), and an aggressive tier below most cliffs where
the baseline scores near zero but the zeros are refusal-driven — compliance headroom that
prompt optimization should recover first. Cliffs are prompt-dependent, so coarse mapping
happens under the baseline for the record and fine mapping under a reasonable prompt.

Deliverables:

1. **Power table.** Minimum detectable score difference versus samples per candidate, plus
   kill thresholds for racing (stop paying for a candidate whose interval cannot reach the
   incumbent).
2. **Encoder/decoder variance decomposition.** k encodings × m decodings on a few problems,
   determining sample allocation and whether caching one encoder output per candidate is
   statistically sound or a lucky-draw bias. Problem-to-problem heterogeneity likely
   dominates the aggregate, which is the formal justification for paired evaluation.
3. **Stratified training subset.** Problems whose cliffs sit near the operating tiers, plus a
   few refusal-zeros retained for compliance signal; harness and test bugs quarantined and
   fixed before optimization (or the optimizer learns to route around them); held-out set
   frozen now. Beware regression to the mean when selecting on noisy estimates —
   characterize with more samples than feels necessary.
4. **Failure-taxonomy base rates per tier.** Every sample logged as one of {refused,
   extraction/format failure, does not run, tests fail, pass}. This predicts how much early
   optimizer gain will be compliance recovery and is the raw material for the
   diagnosis-driven optimizer later.
5. **Frozen baseline Pareto curve with confidence bands.** The reference all later results
   report against, and the noise floor determining which score differences the LLM
   optimizer is allowed to be shown.
6. **Stationarity check.** Run-to-run drift on identical inputs; decide provider pinning.
   OpenRouter serves same-name models at different quantizations across providers, and
   temperature-zero is not deterministic — without pinning, every cross-day comparison
   inherits that confound.

**Metrics discipline.** Binary all-tests pass@1 is the confirmatory top-line metric — the
claim is functional equivalence, and pass@1 is the only practical and meaningful choice at
current model quality. The per-test fraction is shaping and screening: a within-problem
ordering signal for racing, and denser optimizer feedback where binary is flat (the
aggressive tier), where "passes 70%, failing empty-input cases" is actionable and "0%" is
not. It is never a cross-problem aggregate — HumanEval problems have few, uneven, correlated
tests, so the fraction is coarse and not comparable across problems — and never the sole
optimization target, since optimizing it directly Goodharts toward main-path-only
representations while stubs pass odd tests by luck. Standard reward-shaping discipline:
guide and screen with the fraction, confirm and report on binary. Log per-test identity
vectors rather than counts — they enable McNemar-style paired comparisons on exactly which
tests flipped between candidates, and they feed diagnosis. Decompose as P(runnable) ×
E[fraction | runnable] so compliance failures do not pollute the quality signal; expect
zero-inflated bimodality rather than a smooth gradient. Alongside token counts, log
zstd-compressed byte sizes of both representation and stripped source — the claim is
ultimately in bits, and token orderings do not always survive the conversion.

### TLC-2 — Phase 2: simple optimizer

A COPRO-style propose–evaluate–select loop over the encoder prompt only. Concrete incumbent
(2026-08-22): DSPy with GEPA — metric returns a scalar plus structured text feedback
(compile status, n/N, failed groups, ≤4 representative failures); three pools (reflective
trainset, Pareto valset, locked full-pass holdout); tasks selected for utility (medium
difficulty, high prompt sensitivity, diverse failure modes), not difficulty; one level
(encoder) first, both levels with per-predictor feedback later. Reported scores are
calibrated under the full selection rule, on the holdout, never per candidate. This arm is
permanent, not throwaway: it is the null that any smarter optimizer must beat, and its
failure transcripts are the design input for the diagnosis arm. Before automating, a few
human-as-optimizer rounds — whatever information I need in order to improve the prompt by
hand is exactly what the automated loop must be shown.

**Initialization as an arm, not an anxiety.** A baseline (the comparison point for reported
results) and a seed (an optimizer hyperparameter) are different roles; the fixed baseline
constrains the first and is silent on the second. Two arms: the primary seeds from the
baseline prompt, which is attribution-clean — the entire delta from baseline to optimized
belongs to the loop ("we took prior work's prompt and optimization took it from X to Y").
The secondary seeds from a task-aligned prompt ("produce a representation from which another
model can reconstruct the function"), measuring the method's ceiling and initialization
sensitivity, since a misaligned seed burns early budget rediscovering the task. Convergence
of the two arms to similar prompts and scores means the search escapes its initialization
basin — the reachability question from §1 in production form; divergence is a reportable
finding and the justification for smarter optimizers. The anchoring risk lives mostly in the
meta-prompt, not the seed: the meta-prompt states the true objective (maximize
reconstruction pass rate under budget R) regardless of what the incumbent prompt looks like,
and at the aggressive tier the baseline's own refusal record pushes proposals out of
description-space — with the caveat that a flat-zero incumbent provides no differential
signal, so whether the optimizer makes the conceptual leap there is itself worth watching.

**Run hygiene.** Paired evaluation on identical problem sets; encoder-output caching as
licensed by the variance decomposition; adaptive allocation (screen candidates cheaply,
spend samples on survivors, one high-n validation of the final top-k). Show the optimizer
only differences above the noise floor, or print explicit intervals — LLM optimizers
confidently narrate causal stories about pure noise. Watch for proposal mode collapse (later
candidates degenerating into paraphrases of the incumbent) and for overfitting to the
training subset (periodic held-out validation; early-stop the optimization itself). Decide
the scalarization of (pass, measured length) explicitly per tier rather than presenting a
wobbling multi-objective.

**Pilot.** Aggressive tier × 10–15 bracket-mapped problems × low n × cheap inner models × a
strong outer model with full transcripts. This is simultaneously the first real
optimization run, the pipeline shakedown, and the cheap venue for A/B-ing outer-loop
optimizer models.

### TLC-3 — Phase 3: optimizer comparison

Arms at equal budget on the shared frozen validation set: blind propose–select (the Phase 2
null); trajectory-in-context (OPRO-style, conditioning on the history of prompt–score
pairs); and diagnosis-driven (consuming the failure taxonomy, per-test vectors, and failure
transcripts — ProTeGi/TextGrad-flavored, or fully agentic with the harness as a tool). The
comparison is itself a finding: how much feedback structure does prompt-space search need?
It is also the degeneracy/reachability question made operational — whether richer feedback
changes which equilibria the search can reach.

### Standing instrumentation and decisions

Always on: copy-detection metrics between source and representation, superseded as the
leakage measure by TLC-0's I_𝒱(Z→S | B) once the suite runs; per-sample failure taxonomy;
per-test identity vectors (the full vector, since full-pass, fractional, near-miss, and
requirement-grouped scores are all derived from it); token and compressed-byte accounting
under the layered baseline suite (`code_to_test` separated from scored `payload`, selector
bytes, per-representation dictionaries); provider and version pinning per the stationarity
check. Decided: pass@1; extraction frozen as deterministic
codec spec; soft budgets scored by measured length; held-out set frozen at census time.
Open: per-tier scalarization of the (pass, length) objective; how the eventual bits-level
claim frames its guarantee (test-relative equivalence, stochastic decompression) against
lossless baselines.

---

## 4. External assessments and origin notes

Dated notes from the external conversation this doc was promoted from, recorded for
consolidation — not decisions. Related-work claims in quoted text are unverified unless a
citation is given; Danielle's prompts are logged verbatim in `../danielle-inputs.md`.

### 2026-08-22 — the information-bottleneck reading, and proxies for the two mutual informations

Danielle, thinking through representation learning as an information bottleneck, asked
for the objective (min I(X;Z) − β I(Z;Y); constrained form min I(X;Z) s.t. I(Z;Y) ≥ C;
sufficient → minimal sufficient statistic; Tishby–Pereira–Bialek physics/0004057), confirmed
the "removes distractor information" intuition with its caveat (irrelevance is relative to
Y *and the training distribution*; spurious correlations that predict Y survive), and then
put the project to it verbatim: "I'm interested in using an encoder-decoder setup to encode
a Python program into a representation and then to decode the representation into a
function that passes the same tests as the original function, aiming for a proxy for
behavioral equivalence. I'm looking for a proxy for mutual information if the encoder and
decoder are frozen pretrained language models and the intermediate representation is
natural language. I'm not even sure how I'd measure the 'mutual info' between the input
function and the output function, let alone how I'd do that between language and
function." Response, condensed; the formal choices are worth adopting as the project's
statement of its objective.

- **Reframe the relevance variable.** Not I(X; X̂) — that rewards source identity — but
  I(R; Y_T) where X = source, R = the NL representation, X̂ = decoded function, and
  Y_T(X) = (X(t₁), …, X(t_m)) the *test-suite signature*. Objective: min I(X;R) − β I(R;Y_T).
- **Why exact MI is the wrong target here.** MI is a property of random variables under a
  distribution, not of one program and one sentence; neural IB work uses variational
  bounds (Deep VIB, arXiv 1612.00410) for the same reason. So proxies:
  - *I(X;R)* → description length of R: token/byte length, or better −log p_LM(r). With a
    deterministic encoder and discrete R, I(X;R) = H(R), so description length is the
    operational stand-in — this is the rate axis §1 already uses.
  - *I(R;Y_T)* → behavioral reconstruction: sample K decodes from the frozen decoder,
    q̂_T(x,r) = (1 + #{x̂_k passes T_x}) / (2 + K), loss −log q̂_T. The smoothing avoids
    infinite loss at zero passes. This is the pass-rate / pass@k proxy (Codex 2107.03374).
  - *Leakage* → a term that penalizes copying the implementation into R: copy rate,
    identifier overlap, and most interestingly **log p_D(x | r)** — if the frozen decoder
    can reconstruct the exact original source from r, r carries too much implementation
    detail for a behavioral abstraction. This is the formal version of the copy-detection
    signals §1 ("instrument rather than design") already logs.
  - *MI-like relevance* → InfoNCE over a batch: s(r_i, y_j) = how well r_i predicts
    behavior y_j (decode from r_i, run against tests/signature of x_j); log N − L_NCE as a
    contrastive lower-bound proxy for I(R;Y_T) (CPC, arXiv 1807.03748). Requires negatives
    whose signatures differ — functions with the same test behaviour are positives, which
    is the invariance-objective idea in §1 arriving from the other direction.
- **Proposed scalar:** Behavioral Bottleneck Score = log q̂_T(x,r) − λ·L_NL(r) −
  γ·Leak(x,r), higher is better.
- **Key design choice:** the target variable must be behaviour under a sufficiently rich
  test distribution, otherwise the bottleneck rewards "the exact code is: …".

What this adds to the doc: §1's rate–distortion framing and the degeneracy policy are one
objective — rate = −log p_LM(r), distortion = −log q̂_T, and leakage = log p_D(x|r) prices
smuggling in the decoder's own units rather than by n-gram overlap. Two cautions
(Claude-added): the per-sample estimate q̂_T is exactly the small-n pass-rate estimation
problem recorded in the estimation toolkit (`../topics/reference/
estimation-and-calibration-methods.md`), so K and its interval belong in the census power
table; and −log p_LM(r) depends on which LM prices the representation — it should be the
decoder, or a fixed public LM declared as a prior, for the same fairness reason as the
dictionary rule in the baseline suite.

### 2026-08-22 — extractable information: 𝒱-information, the behavioral retention ratio, and the leakage split

Danielle's follow-up (verbatim core): "in practice models often take a function and just
strip out white space and flatten etc to shorten so while length of representation can be
a proxy, there is a real concept of no behavioral information being lost even if the
representation can't literally be run (because I guess our encoder already knows
'programming' so that part doesn't have to be preserved in the representation). So I'm
looking for alternative measures that get at the 'extractable info' overlap between the x
from X and z from Z that could be useful from an analysis perspective even if not directly
optimizable. It could use a separate LM and prompting setup (or set of them) to aim to
measure some type of baseline, etc?" Response, condensed; adopt as the analysis layer.

- **The confound named**: compression by abstraction ("returns the number of unique
  elements") vs. compression by minification (`def f(x):return len(set(x))`) — both short,
  both behaviour-preserving, only the second carries implementation identity. Length and
  Shannon MI cannot separate them; what is wanted is *agent-relative, extractable*
  information.
- **Formalism: predictive 𝒱-information** (Xu et al., arXiv 2002.10689). For an
  extractor family 𝒱, H_𝒱(B | Z) = inf_{v∈𝒱} E[−log v(B | Z)] and I_𝒱(Z → B) = H_𝒱(B) −
  H_𝒱(B | Z): how much Z reduces a *chosen LM/probe family's* uncertainty about
  behaviour B. The pretrained LM's programming knowledge is part of 𝒱 — so "the encoder
  already knows programming" becomes a feature of the measurement rather than a bug, and
  Z need only carry the task-specific facts the prior lacks.
- **Behavioral retention ratio**: Retention_𝒱(Z) = I_𝒱(Z → B) / I_𝒱(X → B), with the
  denominator measured by giving the same extractor the source — because even full source
  does not yield perfect extraction for a bounded probe. Conditional form I_𝒱(Z → B | C)
  with C = signature, imports, type hints, template: what Z adds beyond the coding prior
  and the visible interface.
- **Information profile rather than one B**: input–output behaviour; edge cases;
  exceptions / mutation / side effects; algorithm class; complexity class; and two
  *leakage* targets — names/formatting/exact AST, and "which implementation among
  equivalent variants". Define S = implementation-specific details and measure
  **I_𝒱(Z → S | B)**: after controlling for behaviour, how much does Z still reveal about
  the original implementation? A good abstraction: high I_𝒱(Z → B), low I_𝒱(Z → S | B);
  minified source scores high on both — exactly the distinction wanted.
- **Behavioral bottleneck index**: BBI(Z) = I_𝒱(Z→B | C)/I_𝒱(X→B | C) − γ ·
  I_𝒱(Z→S | B,C)/I_𝒱(X→S | B,C).
- **Four instantiations of 𝒱**: (i) *decoder-pass* — K decodes, −log q_Z as a behavioral
  conditional description length (pass@k, Codex 2107.03374); (ii) *question–answer* —
  generate semantic questions ("what does this return on …?", "can it mutate its
  argument?", "what happens on an empty list?", "is the output sorted?") and score an LM's
  answers given X, Z, minified X, signature only, or nothing — H_𝒱(B | ·) without Z being
  runnable; (iii) *contrastive* — one Z against N candidate behaviours (semantic
  information) or N behaviourally-equivalent candidate sources (leakage); (iv) *MDL probe*
  — a small probe predicting behavioural labels from Z, reported as online codelength not
  accuracy (Voita & Titov, EMNLP 2020).
- **Experimental matrix**: conditions ∅ / signature only / Z / minified X / X / oracle
  docstring or spec; per condition report H_𝒱(B | ·) and H_𝒱(S | B, ·). Target sentence
  of the kind: "the NL representation preserves 92% of extractable behavioural
  information and 18% of extractable implementation identity; minified code preserves
  98% and 93%."

What this adds: the analysis layer the plot needs beside the rate axis. The minified-code
column is the control that makes "abstraction, not shortening" a measured claim rather
than an assertion; the leakage term I_𝒱(Z → S | B) is the principled successor to §1's
n-gram copy detection and to the previous note's log p_D(x | r). Two Claude-added notes:
𝒱 must be declared and frozen per experiment (model, prompt set, K), since every number
here is relative to it; and the same machinery is ELI's measurement of "extractable
competence" with the roles swapped (there the model is the thing probed and the wrapper is
𝒱), so the two projects can share one probe harness.

### 2026-08-22 — the measurement bibliography and five analysis directions

Danielle asked for a deep dive on measuring shared information between language and code
representations, on relatedness estimators for a task or model, and on bottleneck analysis
or optimization approaches — noting "I also really like the contrastive directions in
general." Full record in `../topics/reference/nl-bottleneck-prior-art.md` (measurement
entry, 2026-08-22; unverified). What to take into the plan:

- Two citations that name the project's analysis stance: **Decodable Information
  Bottleneck** (IB relative to a predictive family — the theory behind "the decoder
  already knows programming") and **ContraCode** (contrastive invariance to
  semantics-preserving transforms — the published form of §1's invariance objective).
  Saxe et al. is the caution against claiming compression ⇒ generalization.
- Five directions, and where each lands here: (1) conditional usable-information benchmark
  and (2) behavioural contrastive evaluation with four negative types → the *analysis
  layer* of TLC-1, to be built before any optimizer claim (the response's own
  recommendation); (3) decoder-relative bottleneck score → the objective statement from
  the IB note; (4) representation-geometry analysis (CKA/RSA) over X, Z, X̂, B, traces →
  optional, the "does Z collapse equivalents and separate near-misses" figure; (5) search
  over bottleneck *formats* with a three-axis Pareto frontier → already §1's program; the
  hunch that the best Z is a behavioural contract rather than a docstring is the
  hypothesis the census should be able to refute.
- Data: CodeNet supplies many-implementation / same-problem clusters with I/O tests — the
  source of behaviour-equivalent positives the contrastive game needs; HumanEval/MBPP stay
  the small sandbox. Control tasks (shuffled behaviour labels, random implementation
  identities) belong in the probe suite from day one.

### 2026-08-22 — the correctness ladder and the classical code-compression precedents (Pro-mode search)

A four-turn conversation (record in `../topics/reference/code-compression-literature.md`,
2026-08-22 taxonomy/search entry; attributions unverified) did what the SciSpace passes
did not: it found the classical *code → representation → regenerated code* literature.
Three things to carry into this doc.

1. **A five-level correctness ladder** — byte-exact → source-equivalent (tokens/AST) →
   runtime-equivalent → test-equivalent → intent-equivalent — with Python's reflective
   features (`inspect`, `getattr`, `-O`/`-OO`) as the reason "runtime-equivalent" is not
   safely reachable by transformation alone. TLC's guarantee sits at *test-equivalent*;
   the plot should label every baseline by its ladder level, and leaderboards are per
   level, not pooled.
2. **Lossless competitors on the rate axis**: syntax-directed compression (Katajainen et
   al. 1986; Evans, guided parsing), **JSZap** (JavaScript AST as production / identifier /
   literal streams, ~10% under gzip), Stork–Haldar–Franz adaptive AST compression, Pugh's
   class-file compression, and Boffa et al. 2025 for corpus-scale Python. These are the
   strongest *lossless* baselines on the rate axis and belong in the suite (§4 baseline
   note) at Layer 4 — a production-stream AST codec is what "AST compact" should become.
3. **The honest framing of the lossy side**: program reducers (C-Reduce, HDD/Picireny,
   Perses) are "lossy program minimizers whose correctness is an oracle"; TLC is the same
   object with an NL representation, a stochastic regeneration step, and the same test
   oracle — which is the sentence for §2 that neither "autoencoder" nor "compressor"
   gets right on its own. Library learning (Stitch, BABBLE, LILO, Leroy) is the
   abstraction-invention reading of the same rate–distortion plane.

The search's gap statement matches §1's: no Python-specific system produces a measured
compact semantic representation and regenerates α- or test-equivalent Python with a
reconstruction metric. Seeds added to the litreview plan, subdomain C.

### 2026-08-22 — the lossless-baseline suite for the compression-vs-correctness plot (three turns)

Danielle's framing, verbatim in part: she wants to test "standard lossless compression
algorithms on fairly short code samples, like the HumanEval ground truth examples, one by
one," see what works best, then move to longer samples; she cares only about per-sample
regimes because she is "explicitly trying to plot 'compression versus correctness' on a
per sample level (where correctness is the result of doing other transforms before
running unit tests)"; and she wants every method including slow ones, Python-specific
performance, and transform stacks that "really try to push the limit." Tool record in
`../topics/reference/code-compression-literature.md` (2026-08-22 baseline-suite entry).
What the conversation fixes for TLC-1's baseline accounting:

- **Two regimes only**: standalone per-sample, and per-sample with a *fixed external
  prior* (dictionary, codebook, trained model, reference corpus, LM). Solid archives,
  tar+compressor, and concatenation are a corpus question and are out.
- **Scoring rule**: the x-axis is **compressed bytes of the tested candidate**, not a
  ratio — a transform that shortens code can have a worse ratio and a better description
  length. Report raw length, compressed length, and compressed length relative to the
  original ground-truth bytes; wrappers (`.xz`, `.gz`, `.7z`, ZPAQ) dominate tiny samples,
  so use raw streams / memory-to-memory APIs.
- **Fairness for priors**: the prior is shared by encoder and decoder and not trained on
  the evaluated sample or its variants; don't charge a fixed public prior, do charge or
  amortize any prior trained per benchmark / task family / transform family / model; an
  oracle min over methods is a legitimate envelope only beside a realistic score that adds
  the method/dictionary selector (≥1 byte matters at HumanEval size). This is the same
  rule the project already applies to the shared decoder prompt.
- **Source-lossless vs. semantics-preserving** labels: `tokenize.untokenize` round-trips
  tokens, not spacing; `ast.unparse` round-trips the AST, not the source. Anything that
  strips comments, renames locals, or unparses is a *test-preserving transform* and is
  scored as such — exactly the status of this project's representations.
- **Transform stacks, not compressor stacks**: compressing already-entropy-coded output
  adds overhead; what helps is minify / tokenize / alpha-rename / canonical AST /
  reference-diff *before* one compressor. Expected standalone winners for short Python:
  PPMd high order, paq8px / cmix / NNCP as slow ceilings; Brotli q11 / raw LZMA2 / zstd-22
  as practical points. Expected fixed-prior winners: Python-token codec + zstd dictionary
  (sweep 256 B–128 KB; small dictionaries may win), compact AST codec + trained model,
  nearest-reference token/AST diff. Grammar-based arithmetic coding over AST symbols and
  code-LM arithmetic coding are the custom ceilings — the latter is the "LLM-as-compressor"
  baseline §1 already names.
- **Controls**: the byte lengths of each *representation* before compression (raw,
  minified, AST-unparsed, token codec, AST codec), so a win is attributable to the
  compressor or to the representation.

Consequence for the plan: the "zstd and friends" baseline in §1 becomes a layered suite
(Layer 0 controls; Layer 1 standalone compressors; Layer 2 dictionaries trained on
separate Python *in the same representation*; Layers 3–4 token and AST codecs; Layer 5
bytecode as a CPython-pinned side experiment; Layer 6 reference-diff and LM arithmetic
coding). The headline claim must be stated against the strongest fixed-prior baseline
with the selector cost paid, not against zstd-22 on raw source. The `python-minifier`
and `pyppmd` packages are the named Python entry points. All tool claims are the
respondent's and unverified.

*Implementation turn (same conversation).* The harness shape that came back, kept because
it is the right shape for TLC-1's bit accounting: separate `code_to_test` (what the unit
tests run) from `payload` (the bytes whose description length is scored) — identical for
raw source, different for token/AST/bytecode/reference-diff methods; one result row per
(sample, representation, compressor) with raw payload bytes, compressed bytes, selector
bytes, score bytes, correctness, seconds, and a free-text `side_info` naming every
uncounted prior; dictionaries trained **per representation** on a disjoint split; external
slow tools through stdout/file wrappers with a fixed input filename so archive metadata
does not vary by sample; reference-delta with an explicit oracle label when the nearest
reference is chosen by search. Implementation order: raw source → canonical/minified
source → per-representation zstd dictionaries → token codec → compact AST → slow external
tools → reference delta → oracle min with selector cost. The agent-written, never-run
skeleton is archived at
`~/drotherm/data/convo-artifacts/2026/2026-08-22-lossless-baseline-harness/` (with
`INDEX.md`); API details there are unverified.

### 2026-08-22 — code-compression related work

From Danielle's SciSpace deep review of code compression, run with this paper as the
stated context (record in `../topics/reference/code-compression-literature.md`; 173-file
bundle with `INDEX.md`). Landscape for the related-work section: "code compression" in
the literature means learned embeddings (CuBERT, ContraCode, autoencoders),
prompt/context compression for code LLMs (ICAE, gist tokens, 500xCompressor,
CodePromptZip, LongCodeZip, Stingy Context), model compression (Compressor, LORD), LLMs
as entropy coders, library learning (Leroy — the contrast case for "compressing
functions"), and semantic compression / precise rewriting (Cummins et al.). Girish et
al.'s rate–distortion framework for black-box prompt compression (2407.15504) supplies
the formal axis for this paper's length-vs-reconstruction curve. Standard comparisons to
match: compression ratio in tokens, reconstruction fidelity (exact match / AST
similarity) *and* functional retention (pass@k), against gzip / truncation / selective
context baselines. One title-level false alarm resolved the same day: Maveli, Vergari &
Cohen, "Can LLMs Compress (and Decompress)?" asks coding LLMs to forward- and
reverse-predict four lossless compressors as a code-understanding probe — Danielle read
it; not related. Prior-art gate still parked.

### 2026-08-22 — where the optimizer loop sits in the prompt-optimization landscape

From Danielle's SciSpace deep review of prompt optimization, run with this paper as the
stated context (record in `../topics/reference/prompt-optimization-landscape.md`). The
LLM-as-optimizer loop over harness parameters θ is, in the review's taxonomy, a
meta-prompting / generation-refinement optimizer with execution feedback, applied at
system level (prompts + latent format + stages + sampling) — nearest published
relatives Trace/OptoPrime, LLM-AutoDiff, and Prochemy/EPiC on the code side; the
compound-AI survey (2410.16392) supplies the vocabulary for "optimize the whole
pipeline." Two items for the draft: (1) the review lists *prompt length vs.
performance* as an open problem, which is this paper's rate–distortion framing stated
from the other side — quotable positioning; (2) the related-work section should anchor
on APE / OPRO / ProTeGi / TextGrad / DSPy-MIPROv2 / GEPA, none of which the review
names — pull them from the bundle's 648-row table. Prior-art gate still parked.

### 2026-08-22 — task-set choice against the benchmark landscape

From Danielle's SciSpace deep review of coding benchmarks (record in
`../topics/reference/code-benchmarks-landscape.md`). Two points for the census and the
ICBINB evaluation: (1) HumanEval/MBPP are the reporting standard but carry the heaviest
contamination and toy-function critiques; LiveCodeBench-style dated problems or
HumanEvalPack subtasks (HumanEvalExplain is this project's loop, see the note below) are
the cleaner task sets, and MBPP+/HumanEval+ test suites should be used where the
originals are kept for comparability; (2) the "code reasoning / execution prediction"
benchmark family (CRUXEval-like; LiveCodeBench's execution and output-prediction tracks)
is the nearest neighbour to reconstruct-from-description evaluation and a candidate
source of "semantic contract" probes beyond pass@1.

### 2026-08-22 — prior-art verdicts and the ICBINB grounding bundle

Danielle ran a SciSpace prior-art search on this project's mechanism (brief, verdicts,
and artifacts recorded in `../topics/reference/nl-bottleneck-prior-art.md`; the 100-file
bundle with its own `INDEX.md` at
`~/drotherm/data/convo-artifacts/2026/scispace-nl-latents-rw-agent-artifacts-zip_cc4d31ce-8970-4a5a-9cd1-248b327a0b06_1787423020/`). Dec 2025 verdict over 378 papers: "no publication matches — appears novel", best
partials Nano-Capsulator (4/6: no program encoding, no reconstruction), EPiC (3.5/6),
RLPrompt, Latent Programmer (discrete trained latents), Sentence Bottleneck AE (vector
bottleneck + frozen decoder), ICAE; Jan 2026 update over 36 new papers: unchanged. A
third pass grounded each stage of the ICBINB draft in citations and baseline prompts:
AlphaCodium / AceCoder / Structured CoT (direct generation), Nano-Capsulator + Midolo
guidelines (compression), Misu et al. Dafny prompt styles + SelfEvolve (reconstruction),
Prochemy / EPiC / RL4QE / MCTS-OPS (LLM-as-optimizer), plus a meta-prompt with
JSON execution feedback and T/top-p starting points. For the prior-art gate this
supersedes the earlier stub list (LLMLingua, gist tokens, DSPy/OPRO/TextGrad remain
valid adds). Two gate items the bundle does not settle: **GenDLN** (ACL SRW 2025;
evolutionary joint prompt optimization over stacked frozen LLMs — dismissed by the agent
in one line, structurally the closest thing to the encoder+decoder harness search) needs
Danielle's own read; and the agent's bibliography has identifier slips to fix before
citing. Her ICBINB draft PDF is in the bundle.

### 2026-08-22 — HumanEvalExplain is a ready-made TLC-1 task set

From a SciSpace search Danielle ran for "all papers that report performance results on
the HumanEvalExplain subtask of the HumanEvalPack" with the goal of "a comparison plot of
correctness versus explanation length based on different forms of applying models (eg
different prompt formats, etc) and different models used" (record and artifacts in
`../topics/reference/humanevalexplain-results.md`). HumanEvalExplain (OctoPack, arXiv
2308.07124) is exactly this project's reconstruction loop: explain a function in natural
language, regenerate it from the explanation alone, score pass@1 — 164 problems × 6
languages, standardized harness. Three facts for the plan: (1) only three papers report
it, none reports explanation length or varies prompt format, so the published numbers
(GPT-4 52.1 avg / 64.6 Python; best open 6.7B models 46–51; base models 0) are the
*no-length-pressure endpoint* of the rate–distortion curve and nothing more; (2) the
correctness-vs-length plot therefore has to be generated — run the harness with logged
explanations under a length sweep, which is TLC-1's census on a free task set; (3) the
0% base-model result is a clean prior for ELI-style questions (the explain→regenerate
loop is unavailable without instruction tuning under direct prompting). Prior-art gate:
cite OctoPack as the benchmark precedent for reconstruct-from-text evaluation of code.

### 2026-08-22 — prior-art additions for the gate: PCRL, Nano-Capsulator, EPiC

From a SciSpace paper summary (record and on-disk PDFs/search archives in
`../topics/reference/prompt-compression-and-optimization-literature.md`). Three papers to
add beside LLMLingua and gist tokens: **PCRL** (Jung & Kim 2024; RL-trained extractive
token deletion, black-box compatible, ~25% compression), **Nano-Capsulator** (Zhou et al.
2024, arXiv 2402.18700; a generator rewrites long prompts into natural-language capsules
under a semantic-preservation loss × downstream-utility reward with a hard length cutoff —
structurally the closest objective to TLC's, differing in target: TLC reconstructs the
source, Capsule preserves downstream utility), and **EPiC** (Saluja et al. 2024, arXiv
2408.11198; evolutionary prompt search for code generation with the ATSP
tokens-per-solved-problem metric — the published analog of this project's optimizer-cost
accounting). The SciSpace search CSVs on disk are a seeded candidate list for the full
prior-art pass (still parked).

### 2026-08-22 — evaluations as optimizer signal: DSPy + GEPA, task/test selection, no requirement labels

Two further turns in the estimation conversation (record in
`../topics/reference/estimation-and-calibration-methods.md`, third entry, second
continuation) put the difficulty machinery in service of the optimizer. Danielle's
framing: automated prompt optimization with something like DSPy and GEPA, "at two levels,
or maybe even just at one level initially, but eventually at two levels" (i.e. encoder and
decoder prompts); per-test feedback "might actually be very helpful"; difficulty lets her
cluster and select or characterize the most useful tasks and test cases. Response's design,
condensed; DSPy/GEPA API claims are the respondent's and unverified:

- **Objective shifts** from "which tasks are hard" to "which tasks and tests yield the
  most useful optimization signal without overfitting the prompt". Difficulty is one input
  to *example utility*; the best optimization tasks are medium-difficulty with high prompt
  sensitivity, diverse interpretable failure modes, and clear feedback — not the hardest.
  Suggested mix: ~10–15% smoke, 45–60% near-boundary, 20–30% hard-but-solvable, 10–20%
  edge/failure-mode tasks; never all-hard (repetitive feedback, no gradient).
- **Three pools**: optimization trainset (GEPA reflects on it; rich feedback), valset
  (Pareto-score tracking; DSPy reuses train if absent — don't), locked holdout scored by
  full hidden-suite pass/fail only.
- **Metric returns score + text feedback** (GEPA's design: reflect on trajectories,
  propose prompt edits, combine lessons on a Pareto frontier; predictor-level feedback via
  `pred_name`/`pred_trace`). Score: hybrid, e.g. 0.5·full_pass + 0.4·test-pass rate
  (requirement-grouped if possible) + 0.1·compile/runtime; feedback: compile status, n/N,
  failed groups, ≤4 representative failures (input / expected / got / likely issue),
  one-line advice. Final metric uncompromising; optimization metric educational.
- **Tiered tests**: smoke on every candidate; diagnostic subset on most; full suite on
  validation/final candidates. Store raw per-test outcomes; never let a 100-test task
  outweigh a 5-test task — one score per task-generation.
- **IRT becomes a sampling/curriculum tool**: per-task full-pass and partial difficulty,
  prompt sensitivity (within the model to be optimized), model sensitivity, failure-mode
  entropy, feedback clarity, cost; per-test pass rate, discrimination, flakiness,
  redundancy. Task-level clusters (algorithmic pattern, I/O format, edge-case profile,
  failure modes) for stratified batches; test-level clusters for diagnostic coverage.
- **One level first** (generation prompt), then two (plan/generate + repair-from-feedback,
  or here encoder + decoder); two-level attribution is harder — did the generator improve
  or did the other prompt compensate? Per-predictor feedback is what makes the
  two-level case tractable.
- **No requirement labels?** (Danielle: "I don't actually have a way to group the tests
  by requirement, do I?") Start without: full-pass primary, per-task test-pass rate as
  partial credit, representative failed tests as feedback. Then **cluster tests
  empirically by their pass/fail vector across the model × prompt grid** — tests that fail
  together are one bug counted many times; cluster, then label clusters with an LLM or by
  hand later. If test content is available, cluster on test name / body / assertion /
  error message. If tests are generated going forward, emit requirement metadata with
  them.

What this changes for the plan: TLC-1's census already yields the model × prompt grid the
empirical test clustering needs; the "simple optimizer" of TLC-2 gets a concrete
incumbent (DSPy/GEPA with score + feedback) and the three-pool split becomes a standing
instrumentation decision. GEPA's arXiv ID (2507.19457, unverified) closes one entry of the
prior-art gate's missing-anchor list.

### 2026-08-22 — full-pass vs. fractional test score as the response: a design tension

A follow-up in the estimation conversation (record in
`../topics/reference/estimation-and-calibration-methods.md`, third entry, continuation)
argues that for code generation the **all-pass outcome should be the primary difficulty
and success measure**, the fractional test score a secondary partial-credit measure, and
per-test outcomes a diagnostic layer with a generation-level random effect. Its case
against fractional-as-primary: the average tracks test-suite density (many near-duplicate
easy tests vs. a few edge cases) and one bug fails many correlated tests. This project
chose the fractional score as a signal precisely because all-pass is too coarse for an
optimizer (2026-07-11). Both can be true: report full-pass as the headline and the
fractional score — or better, a requirement-grouped score, and the near-miss score
"test-pass rate given not full pass" — as the optimizer's ordering signal. Whether
HumanEvalPack-style suites are balanced enough for the raw fraction to be honest is a
census question; log per-test outcomes so the choice can be revisited. Recorded as a
tension, not a decision.

### 2026-08-22 — item difficulty from the census matrix (pointer)

If the census is run as several models × several prompts per problem, the matrix supports a
first-pass item-difficulty fit (smoothed pass rate → Rasch → many-facet mixed model with
model, prompt, and model:prompt effects; per-item prompt-sensitivity vs. hard-for-everyone
split; leave-one-model/prompt-out validation). Recipe and caveats in
`../topics/reference/estimation-and-calibration-methods.md` (third entry). This is the
"stratified training subset" step of the large-n phase (§4 2026-07-11) made formal.

### 2026-08-22 — estimation design for the census and optimizer loop (from a per-docstring estimation conversation)

A four-turn conversation on estimating a (model, docstring) pass fraction at fixed samples
(full record in `../topics/reference/estimation-and-calibration-methods.md`) is, in effect,
TLC-1's measurement design. Items to carry into the census spec: fractional test score as
the response (already chosen, §4 2026-07-11) with the *program* as the independent unit;
canonicalize and deduplicate completions but keep multiplicities; bootstrap over
completions (block bootstrap over API calls, since OpenRouter temp-0 is not iid); Wilson or
Jeffreys for all-pass rates at small n; empirical Bernstein when a conservative floor is
wanted; paired designs for candidate-vs-incumbent comparisons. Two additions beyond the
noise-floor plan: (1) **conformalized shrinkage** — combine the small-n mean with a cheap
predictor (compile rate from 1–3 samples, visible-test score, sample diversity, a
docstring-adequacy judge) and calibrate the residual against high-n estimates on held-out
problems, simulating the same budget; (2) **calibrate after selection** — the optimizer
picks the best candidate from many, so its reported score must be calibrated under the
full selection rule, not per candidate. The docstring-adequacy judge ("does the docstring
determine the behaviour the tests check?", structured ambiguity fields) doubles as a
prompt-repair and routing instrument. Danielle flags conformal prediction as a
cross-project tool.

### 2026-08-22 — a second novelty check returns the opposite verdict: "Language Bottleneck Models"

A separate novelty query (Danielle's prompt not kept; the response plays it back as "a
system that compresses code into a natural-language latent representation using an
optimization loop (RL/search) with a frozen LLM decoder") concludes **"equivalent method
already published", confidence high** — the reverse of the Dec-2025 SciSpace verdict ("no
publication matches … appears novel"). All attributions below are the response's and
unverified; the full record is in
`../topics/reference/nl-bottleneck-prior-art.md` (2026-08-22 second-check entry).

The one item that is genuinely new to the record and needs Danielle's read, beside GenDLN:
**Language Bottleneck Models** (Berthon & van der Schaar, arXiv 2506.16982, June 2025) —
encoder LLM emits a short NL summary, a *frozen* LLM decodes from the summary alone, the
encoder is trained with group-relative policy optimization against the frozen decoder's
performance; primary domain is knowledge tracing (summarizing a student history), framed
as a general "inverse problems with frozen LLMs" recipe. If the characterization holds,
LBM is the named framework TLC must position against. The response's own fine print
concedes the gap: LBM trains the encoder's weights and optimizes one side, whereas TLC
optimizes *both* prompts by search with both models frozen — which the response calls
"less common … an implementation detail or a minor variation." That judgment is the
respondent's, and it is the same adjacent-question substitution the SciSpace sessions made
(weight-trained encoder ≈ search-only harness). The substantive claim to defend is the one
already in §1: whether a search-optimized representation beats a hand-prompted one, and
whether it serves more than reconstruction.

Also surfaced: "OverLang — Teaching LLMs to Speak in Pseudocode" (an `agents4agents.ai`
PDF, no arXiv record given; RL-trained pseudocode shorthand decodable by other models —
credibility unknown, treat as a lead only); APRIL (2509.25196) and Proof2Silicon
(2509.06239), RL prompt optimization for frozen code generators — the decoder half only;
Concept Bottleneck LLMs (2412.07992) — classification through text concepts, not
reconstruction; "Semantic Compression with LLMs" (Gilbert et al., already on file) and
Sentence Bottleneck autoencoders (already on file, soft-vector bottleneck). Net effect on
the plan: LBM joins GenDLN as a must-read gate item; nothing else changes the gap statement.

**Danielle's read (same day, from memory of the paper):** LBM is very relevant and is
definitely prior work — but it uses the bottleneck to grade student responses (a
non-verifiable target, unlike tests on code), and tries only about three prompts rather
than running an optimization loop. So it supplies neither a baseline, a method, nor
comparative results for TLC. Position it as the nearest named framework in §2 related
work; the "minor variation" judgment does not survive the comparison. The LBM half of
gate item 1 is closed; GenDLN remains.

### Origin notes — moved from `topics/staging/text-latent-code-autoencoder.md`

### 2026-07-11 — the idea as played back

> You want representation-space benefits (latent dynamics, style manipulation) without weight
> updates, so you'd build an autoencoder where encoder and decoder are both frozen frontier
> LLMs behind APIs, the latent is *text* (hence natively consumable by any LLM), and the only
> learnable object is the harness — the prompts — optimized by an LLM outer loop against
> round-trip reconstruction, scored by test pass rate.

### The assessment (near-verbatim)

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

### 2026-07-11 — Point 1, the degenerate solution (discussed point by point)

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

### 2026-07-11 — Point 2, the objective vs. the style goal

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

### 2026-07-11 — Point 3, "this isn't quite an embedding"

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

### 2026-07-11 — Point 4, prior art to position against

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

### 2026-07-11 — Two interest categories: representation exploration vs. beating lossless compression

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

### 2026-07-11 — Compression project: implementation state and initial experiment design

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

### 2026-07-11 — Sequencing the first measurements; the cliff structure

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

### 2026-07-11 — Baseline vs. seed: where to start the optimization

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

### 2026-07-11 — Recap of conclusions; what the large-n phase must output

**Danielle's recap (confirmed accurate).** Many design decisions become arms of the
experiment. First need: "a regime where there is the potential for optimization, and a
statistical understanding of how many samples we need to get any type of signal on the
behavior of a given prompt." So: coarse sweep → find cliffs → choose those regions for large
sample collection. Then a simple optimizer (sample prompts, evaluate in sequence), run from
both the baseline and the task-targeted seed to see whether there is a difference, plus "the
experience of doing a human-in-the-loop version where I am the optimizer," to design the
first real optimization approach that tries to move the metrics rather than validate the
loop. Open question she raised: beyond the power calculation (samples needed to detect the
expected difference), what other actionable conclusions should the large-n sweeps yield?

**Response (near-verbatim).**

*Refinement.* "'Variance estimation' is subtly the wrong frame for binary outcomes.
Per-problem variance is fully determined by the pass probability — p(1−p) — so the large-n
phase isn't measuring variance as an independent quantity. It's estimating pass
probabilities precisely where they're informative, and locating where the stochasticity
enters."

*What the large-n phase should output.*
1. "The power table — minimum detectable difference versus samples-per-candidate, plus the
   kill thresholds for racing (when a candidate's interval can't reach the incumbent, stop
   paying for it)."
2. "An encoder-vs-decoder variance decomposition. Run k encodings × m decodings on a few
   problems and see which stage carries the noise. This sets your sample allocation and
   answers whether caching one encoder output per candidate is statistically sound or a
   lucky-draw bias. Problem-to-problem heterogeneity will likely dominate the aggregate —
   that's the formal justification for paired evaluation."
3. "The stratified training subset: problems whose cliffs sit near the operating tiers, a
   few refusal-zeros kept for compliance signal, broken problems quarantined — some 0%s will
   be harness or test bugs, and those get fixed before optimizing, or the optimizer learns to
   route around them. Held-out set frozen."
4. "Failure-taxonomy base rates per tier: refusal vs extraction vs wrong-code. This predicts
   how much early 'gain' will be compliance and is the raw material for the diagnosis arm."
5. "The frozen baseline Pareto curve with confidence bands — the reference all later results
   report against, and the noise floor determining which score differences you're allowed to
   show the LLM optimizer."
6. "A stationarity check. You're on OpenRouter's cheapest models: the same model name can be
   served by different providers at different quantizations, and temp-0 isn't
   deterministic. Measure run-to-run drift on identical inputs and decide whether to pin
   providers — otherwise every cross-day comparison inherits that confound."

*Sharpening.* "The simple-optimizer runs aren't just loop validation — they're a permanent
arm (the null every smarter optimizer must beat) and the source of the failure transcripts
that design the diagnosis arm. Nothing there is throwaway."

**Next requested artifact (2026-07-11).** A structured write-up of the motivation, the goal,
Danielle's interests, the lab's compression interest, and the plan for pursuing the
compression goal first — the natural seed for this topic's project doc if promoted.

### 2026-07-11 — Fractional test pass rate as a second signal

**Danielle's observation.** The binary argument made her realize she also has per-test pass
percentage: "binary pass rate is the relevant top-line metric but it feels like test pass rate
is useful signal too? And that had variance right?"

**Response (near-verbatim).**

*Its variance is a real free parameter.* "A fraction on [0,1] isn't mean-determined the way a
Bernoulli is: at the same mean, its variance can be anything from 0 up to the Bernoulli
bound p(1−p). Two useful corollaries: your binary power table is the conservative envelope
(the fractional metric can only be cheaper to estimate at a given mean), and the fractional
variance becomes one more census output — with the shape mattering more than the number."

*Signal density where binary is flat.* "This directly patches the hole ... at the aggressive
tier — a flat-zero incumbent giving the optimizer no differential signal. Binary can't
distinguish 'preserves 80% of behavior' from 'gibberish'; the fraction can. So it's the
natural screening metric for racing ... and much richer optimizer feedback — 'passes 70% of
tests, failing empty-input cases' is actionable where '0%' isn't."

*Shaping, not the objective.* "The claim is functional equivalence, which is all-tests.
Optimizing the fraction directly Goodharts toward representations that nail the main path
and systematically drop edge cases — high partial credit, zero binary. Also watch the cheap
floor: a stub with the right signature passes the odd test by luck. Standard reward-shaping
discipline: guide and screen with the fraction, confirm and report on binary."

*Expect zero-inflated bimodality.* "Refusals, extraction failures, and signature errors all
land at 0; working code lands near 1. Whether the middle is populated is empirical — the
census answers it free if you log per-test outcomes (which tests, not just how many) ...
decompose into P(runnable) × E[fraction | runnable] so compliance failures don't pollute the
quality signal."

*Harness check first.* "Stock HumanEval `check()` is sequential asserts that abort on first
failure — if your fraction comes from that, it's 'position of first failing assert'
(censored), not independent per-test outcomes. Verify the asserts run individually before
trusting the metric."

### 2026-07-11 — Per-test outcomes confirmed; dividends and caveat

**Danielle:** the HumanEval `check()` function has already been split into its individual
asserts, so per-test outcomes are real.

**Response (near-verbatim).** "You have per-test identity, not just counts — that's the
diagnosis arm's raw material. ... 'all candidates at this tier fail the empty-input and
negative-number cases' is exactly the transcript-level signal the smarter optimizer
consumes, and it's also how you distinguish 'representation dropped an edge case' from
'decoder misread the format.' Log per-test pass/fail vectors, not just the fraction — the
vectors also enable McNemar-style paired comparisons between candidates on exactly which
tests flipped." Caveat: "test counts per problem are small and wildly uneven. HumanEval
averages ~7–8 asserts, some problems have 3, and the tests weren't designed as independent
probes — several often hammer the same behavior. So the fraction is coarse ..., correlated
within a problem, and not comparable across problems as an absolute number. Treat it as a
within-problem ordering signal for racing and feedback, not as a quantity to average across
the suite — the cross-problem aggregate belongs to binary." Micro-check: "make sure the split
tests are order-independent (no shared mutable state from the original sequential block)."
With that, "the census design is fully specified."

### 2026-08-22 — Draft internals and a wrapper-only sibling (from a later conversation)

A later conversation (undated, ~2026; recorded in `elicitation-gain.md`)
was conducted against the TLC draft PDF and surfaced these facts about the draft as written:
objective $J(\theta)$ over verifiable success, success = feasibility waterfall (code-only →
compiles → runs → passes tests) × semantic correctness (Eq. 4–5); harness parameters
$\theta$ = prompts, templates, latent format, stage decomposition, tool use, memory,
sampling; LLM-as-optimizer actions = prompt diffs / added verifier stages / sampling
changes, reward = mean batch success (Eq. 7); latent-format axis COMP-NL vs. COMP-SHORT;
related-work stubs for RL/evolutionary prompt optimization and compression, language
bottleneck models, semantic compression, AlphaCodium (arXiv 2401.08500 — add to the
prior-art gate as the wrapper-only code precedent). The conversation proposed a sibling
use of the same formalism: fix the optimizer loop and budget, swap the decoder for a tiny
frozen DataDecide model, and measure elicitation gain $\Delta S$ vs. base size and pre/post
training — now the project `elicitation-gain.md` (`ELI`), which reuses this harness; keep the verifier
suite and optimizer loop in sync between the two.

### Open questions (running)

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
- Large-n phase deliverables (six, above): power table; encoder/decoder variance
  decomposition; stratified train subset + frozen held-out; failure-taxonomy base rates;
  baseline Pareto curve with bands; OpenRouter provider stationarity check.
- Fractional test pass rate: screening/shaping signal only (binary remains the reported
  objective); log per-test outcomes; decompose P(runnable) × E[fraction | runnable]; verify
  asserts run independently (done — `check()` split); log per-test vectors for McNemar-style
  pairing; fraction is within-problem only; audit order-independence of the split tests.

### 2026-08-22 — Prompt condition is part of the harness contract

HumanEval's original prompt is the raw stub (completion mode); MBPP's original prompt is
already a few-shot NL task + visible asserts; both have named instruct variants in current
harnesses, and pass rates are not comparable across the raw/instruct choice. For the TLC-0
condition matrix, record per run whether signature/oracle-spec conditions use the raw
stub or an instruct wrapper, and pin the harness commit. MBPP's prompt leaks test cases by
design, which matters for leakage accounting. Details and sources:
`../topics/reference/code-benchmarks-landscape.md` (prompt-format entry).

### 2026-08-22 — HumanEval derivative ecosystem as prior art and controls

From the HumanEval-afterlife survey in `../topics/reference/code-benchmarks-landscape.md`:
**ShortenDoc** (docstring compression on HumanEval/EvoEval; ~30% compression often keeps
or improves pass@1) is the nearest existing work to the compression project's NL-side
question and should join gate 1 of the litreview plan; **ReCode** and **NLPerturbator**
are the semantics-preserving perturbation baselines the TLC-0 control tasks are compared
against; **CL-HumanEval**'s name-stripping is a ready "signature without hints" condition
for the condition matrix; **EvoEval** supplies harder/creative variants if HumanEval
saturates. No flagship code→docstring benchmark was found (HumanEvalExplain remains the
loop baseline). The overlap literature's three-matrix scheme (prompt↔prompt, code↔code,
prompt↔code) is the same bookkeeping as TLC-0's leakage accounting; within-HumanEval item
redundancy is unstudied and is a small analysis worth doing before per-item claims.


### 2026-08-23 — The Cho pitch (early 2026) and the cheap-model re-evaluation angle (historical conversation, intake 2026-08-23)

**Timeline (Danielle's correction at intake):** the conversation and pitch date from
roughly January–March 2026, not from intake time; time-sensitive advice in the
conversation (deadlines, "this week") is historical. A workshop paper was eventually
submitted and rejected — Danielle's own assessment: the rejection was fair, the paper
was written very, very fast. The project's scope was subsequently redirected for a
period; as of 2026-08-23 Danielle is returning to the original version of the
direction. This conversation is therefore the *ancestor* of the current TLC framing:
this doc and the three-layer TLC structure (decided 2026-08-22) are the return to and
refinement of what was pitched here.

**The pitch (Danielle, early 2026):** "I just pitched my advisor on the pdf contents as
a general thesis direction to move in (starting with a well scoped workshop paper which
is a small subset of what's in that doc)." The pitch document — "LLM-as-Optimizer of
Natural Language Bottleneck Model" (Rothermel*, Li*, Cho), targeting the AI with
Recursive Self-Improvement Workshop @ ICLR 2026 — is preserved at
`~/drotherm/data/convo-artifacts/2026/2026-08-23-prompt-opt-reeval-aha/cho-proj-pitch-llm-as-optimizer-of-nl-bottleneck-model.pdf`
(full conversation transcript beside it). Draft state at pitch time: formal setup done
(harness config θ=(θ_E,θ_D) over prompts/templates/format/stages/tool
use/sampling; Feas/Succ waterfall, Eqs. 4–5; objective J(θ), Eq. 6; LLM-as-optimizer
policy π_opt with harness-editing actions and batch-mean-Succ reward, Eq. 7); manual
baseline experiments done (FizzBuzz ×100-sample variance; String Rules / Stateful /
Simple Algorithms families at difficulty 3–5; gpt-5-nano, gpt-5.1-codex-mini,
gemini-2.5-flash-lite, gemini-2.5-flash, haiku-4.5, gpt-oss-120b; temperature 0.2,
top-p 0.95; OpenRouter + Docker isolation); the optimizer loop itself not yet run —
"later this week, before the workshop deadline."

**Danielle's motivating curiosities beyond the pitch itself (near-verbatim, from the
early-2026 conversation):**

- Using OPRO+ as related work, "wondering how accurate the conclusions are currently,
  when optimizing prompts, tasks, etc."
- "has posttraining in some cases robbed llms of previously existing general
  optimization skills because they've been partially collapsed to game metrics/fit
  benchmarks?"
- The OpenRouter cheap-model menu (the pitch's Table 5: coding-capable,
  latest-of-their-line models from $0.02/M input) "just seems like this amazing
  opportunity not to evaluate the models on your standard terminalbench one shot
  monstrosities, but actually to evaluate them on fairly well scoped tasks that we'd
  expect them to succeed at and then measure the variance within model and across
  models + extend to understanding how these 'super cheap' modern options compare to
  the 1-3 affordable-SOTAish options that were generally published on in for previous
  methods."

**Response (ChatGPT, condensed; unverified):** the load-bearing reframe is that *the
feasible experimental regime changed* — cheap models make distributional evaluation
affordable (within-model variance, across-model variance, and their interaction with
optimizer loops), where 2022–2024 papers reported point results. Recasts the question
decision-theoretically: not "can method X improve task Y on model Z" but "when is
prompt optimization worth doing at all given model strength, cost, variance, and
transfer" — expected gain vs. search cost plus stability. Operationalizes the
posttraining-collapse hypothesis two ways: (a) optimizer competence vs. task competence
(models chosen so direct-solve is decent; a collapsed model shows flat/noisy improvement
curves and poor edit validity despite good direct performance); (b) transfer (optimize
on A, evaluate on B — collapsed-to-quirks optimization transfers poorly). Proposes a
workshop-slice with headline claim "under realistic budgets, prompt optimization
exhibits a phase transition: below a model-dependent capability threshold loops
stagnate; above it, gains appear but are dominated by variance reduction and contract
enforcement rather than clever prompting" — RQs: budgeted value / mean-shift vs.
variance-reduction attribution through the Feas waterfall / transfer matrix / optimizer
vs. task competence; three headline figures (ROI curves vs. $, waterfall attribution
stacked bars, optimize-on × evaluate-on transfer heatmap). Flags OpenRouter model
identity/versioning drift as both a reproducibility hazard (log model string, provider,
date, sampling params, request IDs) and a research question ("how stable are
conclusions under model drift?").

Intake still in progress (more chunks of the historical conversation to come); this
entry may be extended. Sibling record of the re-eval framing:
`../topics/reference/prompt-optimization-landscape.md` (2026-08-23 entry).

### 2026-08-23 — Chunk 3 of the early-2026 conversation: library leverage, open-ended tasks, curriculum

Same historical conversation as the entry above. Chunk 3 turns to the synthetic
function-generation library (the "pretty robust library for generating testable python
functions" built for the workshop paper, needing multi-language extension) and what
else it unlocks. Danielle's reactions carry the signal:

- **Open-ended task templates — enthusiastic yes** ("would be really cool to add to my
  synthetic dataset library!!"). The respondent's design note (verbatim in the
  transcript) proposes three starter templates that stay closed-world-verifiable while
  opening the implementation space: multi-function micro-libraries (4–8 related
  functions sharing invariants, e.g. a tiny `stats` module with cross-function
  invariant tests like `quantile(0.5) == median`), stateful data structures (LRU cache
  with TTL, event emitter, rate limiter — sequential scripts + invariant checks +
  randomized operation sequences), and repair/refactor tasks (broken code + tests;
  avoids cold-start variance, ideal for measuring optimizer improvement trajectories).
  Design principles: deterministic verification, multiple reasonable decompositions,
  surface area over algorithmic novelty, failure-mode richness, composable difficulty
  knobs. These extend the existing String Rules / Stateful / Simple Algorithms
  families upward in openness.
- **Curriculum for the LLM-as-optimizer** — Danielle's observation: curriculum learning
  over difficulty knobs "would even be plausible as an approach to making the
  LLM-as-optimizer approach stronger… I suspect it isn't a standard component of all
  approaches?" Respondent concurs it is not standard (adjacent momentum in
  budget-aware/sequential evaluation-allocation methods, e.g. a cost-aware
  evolutionary prompt-optimization paper at PMLR v293, zehle25a — no arXiv ID given);
  implementable inside the harness as adaptive task scheduling (start search on easy
  instances, introduce harder ones, measure sample efficiency and transfer). This is a
  candidate TLC optimizer-loop contribution the difficulty ladder already supports.
- **Cross-language semantic equivalence** — Danielle's strongest reaction
  ("!!!!! I hadn't even considered this"): same spec → implementations in multiple
  languages, each verified by language-native tests; measures polyglot robustness,
  cross-language prompt-optimization transfer, language-specific failure signatures.
  Her follow-up on unsupervised-MT-for-code answered with TransCoder (2006.03511,
  2020) and "Leveraging Automated Unit Tests for Unsupervised Code Translation"
  (2110.06773, 2021) — the latter's generate-then-test-filter loop closely matches the
  library's spec+verify structure; the library can emit supervised translation pairs
  cleaner than mined corpora. (Both IDs agent-supplied, ledger rows added.)
- **Post-training/distillation data generation** — aligns with her stated interest in
  "how far we can push small/weak models": the library can emit failure cases with
  minimal counterexamples, self-correction traces, and unit tests from specs as
  reusable assets for RL/DPO/verifier training.
- **The orphaned variance sweep** — the large performance/variance run done for the
  workshop paper never fit in it. Respondent's homes: arXiv companion evaluation
  report (recommended: workshop paper = tight claim, companion = the sweep),
  repro-style writeup, artifact report living with code+logs, or supplement. As a
  standalone reliability-workshop framing: "point estimates are misleading:
  within-model variance and early-stage feasibility failures dominate observed
  differences; reliability ranking flips under realistic budgets."

Chunk 3 also contains Danielle's model-divergence hypothesis and three
behavioral-distance experiment designs — recorded in
`../topics/reference/evaluation-methodology-literature.md` (2026-08-23 entry), since it
is a freestanding measurement question rather than TLC-specific.

### 2026-08-23 — Dataset strategy: synthetic-library lessons and the HumanEvalPlus cutover (current state, Danielle at intake)

Current-state update from Danielle during the transcript intake (this postdates the
early-2026 conversation and supersedes its assumption that the synthetic families are
ready to use as-is):

**Synthetic-library lessons learned** (near-verbatim): "designing the families and
their difficulty levels was very non-trivial and a variety of issues came up such as
ending up with 'different' functions by sampling different settings that ended up being
identical, it being very difficult to maintain, some functions basically simplifying to
identity, and the difficulty measures being very uneven across and within families.
so the synthetic functions would need some work to use." Four distinct defect classes:
(1) sampled-setting collisions producing duplicate functions, (2) high maintenance
burden, (3) degenerate samples that simplify to identity, (4) design-time difficulty
ladders not matching empirical difficulty, unevenly across and within families.

**The cutover:** a core reviewer pushback was "why use synthetic functions at all, why
not use standard datasets," which the advisor firmly agreed with; the project currently
uses HumanEvalPlus.

**Danielle's assessment of HumanEvalPlus** (first-hand, from running it): "much worse
than my synthetic functions in that it has a dramatic variance of difficulty, some
samples have been solved by every model I've tried on them, down to the cheapest, some
are broken in the official release, they are SUPER short so compression is kinda
meaningless." The shortness point is structural for the compression track: items must
be long enough for a description-length bottleneck to bind.

**Current interest:** "combining synthetic and real world, reporting on humaneval
because people assume it must be good, but also on better real world datasets, etc."

Cross-references: first-hand HumanEvalPlus defects also recorded in
`../topics/reference/code-benchmarks-landscape.md` (2026-08-23 note); the 2026-08-22
HumanEval-derivative intake (same file) is the candidate menu for "better real world
datasets"; empirical difficulty calibration of synthetic families connects to the IRT
program (IRT-11: which item representation explains difficulty).

### 2026-08-23 — Chunk 4 of the early-2026 conversation: data-manipulation and tiny-ML task families

Same historical conversation. Danielle's proposal (her idea, from writing "a ton of
research code"): **data-manipulation tasks** — model may use any library from a
whitelisted set (pandas, numpy, polars, no library); accomplish a transformation
expressible as filtering → transforming columns → aggregating → writing; optionally
produce a plot with a specified plotting library judged by pixel distance against the
spec-matching plot — "all of those ideas have a ton of different ways you could
implement them but you could describe the target succinctly and make it easily
testable." And **tiny-ML tasks** — scikit-learn vs scipy vs pytorch with a
deterministic target (possibly deliberately unspecified how to achieve determinism):
build a tiny model to spec, train a fixed number of steps on CPU, evaluate, report
scores; she flags cross-library evaluation identity as the tricky part.

Respondent (condensed, endorsing): the combo is "open-ended implementation space +
tight verification." Key design guidance —

- **Task DSL + sandboxed execution:** an internal task representation generating input
  dataset, target transformation, and evaluation oracle; the NL prompt is one
  *rendering* of the DSL. Data-manipulation tasks as compositions of a small operator
  set (filter / select-rename / mutate / join / groupby-aggregate / sort-limit /
  write). "Relational algebra benchmarks with modern dataframe libraries."
- **Canonicalized evaluation, not string equality:** schema (names/dtypes, possibly
  relaxed), values (exact for ints/strings, tolerance for floats), row-set equality
  with ordering enforced only when specified.
- **Plots:** pixel distance is fragile (fonts, anti-aliasing, versions, backends);
  start by comparing the underlying plotted data or a canonical spec (Vega-Lite JSON /
  captured matplotlib calls); pixel distance later with Agg backend, fixed
  font/DPI/figure size, pinned versions, tolerance.
- **Tiny-ML determinism by construction:** closed-form targets (normal-equation
  ridge, PCA-via-SVD) or fixed-init fixed-order full-batch iterative methods;
  single-thread (OMP_NUM_THREADS=1), fixed dtype, no shuffling/dropout; cross-library
  comparison on metrics within tolerance, parameters only when the algorithm is
  forced. Starter: synthetic data from a known linear model → ridge/logreg → held-out
  metrics.
- **Pitfalls:** library-trivia benchmark (mitigate with minimal docs in prompt or
  restricted operator set); hidden nondeterminism; overly strict matching; spec
  ambiguity (include schema previews, 3–5 example rows, explicit invariants).
- **Roadmap:** v0 dataframe transforms only (8–12 operator templates, pandas + pure
  Python); v1 polars/numpy + one non-pixel plot task; v2 pixel plots + tiny-ML.
- Ties to the earlier threads: library choice and idiom become behavioral-distance
  fingerprints; optimizers measurably reduce schema/ordering/missing-value/style
  errors; cheap per-task cost supports proper variance estimation.

**Intake note (Claude-added observation, for the walkthrough):** the DSL design bears
directly on the dataset-strategy problems recorded above — a canonical operator-tree
representation gives (a) dedup and identity-degeneracy detection at the semantic level
rather than post-hoc, (b) a maintainable generator, (c) difficulty as measurable
properties of the operator tree, and (d) a natural ground-truth description d_s whose
length the budget B = b1·|d_s| + b2 can bind against — while "succinct spec, rich
implementation" is exactly the asymmetry the compression bottleneck needs and
HumanEvalPlus lacks.

### 2026-08-23 — Chunk 5 pointer: decision quality, efficiency prior art, and Danielle's agent-tooling goals

Chunk 5 of the historical conversation mostly belongs to the divergence thread —
Danielle's ranked-choice "decision quality under constraints" idea, its efficiency-
benchmark prior art (ENAMEL / EvalPerf / EffiBench / DS-1000), the probe-for-divergence
framing, and the de-risked A/B/C design are recorded in
`../topics/reference/evaluation-methodology-literature.md` and
`../topics/reference/code-benchmarks-landscape.md` (2026-08-23 entries). TLC-relevant
residue: DS-1000 (2211.11501) is the direct precedent to position the chunk-4
data-manipulation family against (a generator adds controllability, anti-memorization,
and variance-at-scale over its fixed items); pydantic contract tasks (implement
BaseModels + validators against payload suites asserting accept/reject behavior and
error paths) are a candidate task family aligned with Danielle's stated tooling goal —
"I'd REALLLY love to fit my coding agents to be pros at pydantic," then pandas, altair,
matplotlib, torch; her stack is a Docker-sandboxed pydantic-ai agent, uv, pytest, with
synthetic tests currently plain input/output pairs.

### 2026-08-23 — Chunk 9 of the early-2026 conversation: canonicalization as a deterministic sibling of the autoencoder

Danielle's idea (voice-dictated; her framing): alongside function → language → function,
consider **function → normalized function variant** — provably functionality-preserving
transformations that take code to a much more standard form, beyond what
linters/autoformatters do for human consumption ("standardization that [is] not
necessarily beneficial… for humans… but that [is] functionality preserving, like
provably"). The encoder/decoder need not be an LLM at all — a deterministic
canonicalizer. Her extension: compile to a lower-level language (C, MIPS, LLVM-style)
where much more aggressive normalization/optimization is possible, then convert back to
a standardized Python variant. Motivations she names: **stability for highly variable
pipelines** ("it shouldn't matter whether I put a comment above or below a certain
line… but I suspect it does"); a prediction that normalization/minification produces
measurable, *non-uniform-across-families* performance changes (interesting on its own);
and **shrinking the space wrapper-style optimizers search over** ("fewer variations
that you're optimizing over"). She links it to constraint consistency, to
compressing/stylistically-modifying code, and to CompilerGym as a task setting.

Respondent (condensed; identifiers unverified): this is quotienting out program
symmetries. Prior art: **SymC** (code symmetries, invariance/equivariance to
semantics-preserving rewrites; PMLR v235 pei24b, no arXiv ID given);
**ProgramTransformer** (semantics-preserving transformation catalog — variable
renaming, independent-statement swaps, loop exchange, dead-code insertion; ScienceDirect,
no ID); **LibCST** (lossless CST parsing + codemods, since Python's AST cannot
round-trip formatting/comments); **CompilerGym** (compiler pass sequencing as
sequential decision-making — "rewrites as an action space"); LLM+compiler feedback
work (2403.14714). On compile-down-and-back: Python → IR → Python is not a clean
inverse (decompilation underdetermined); keep the IR as a canonical *view* (CFG/SSA/
bytecode) rather than a round-trip. Three-level decomposition: L1 surface
canonicalization (formatting, imports, comment/docstring normalization); L2
structure-changing semantics-preserving rewrites (alpha-renaming, statement
reordering, desugaring, boolean/commutative normalization) — where cross-family
differences likely appear; L3 canonical IR for invariance-style reasoning. Suggested
first experiment: L1 + a few L2 operators, quantify per-model sensitivity to
semantics-preserving transformations; canonicalization then motivated as the
stabilizing intervention. Paper-claim shape: canonicalization reduces spurious
variance and improves robustness/edit-agent reliability/optimizer transfer,
non-uniformly across families.

**TLC ties (Claude-added):** (a) TLC-0's condition matrix already contains "minified
X" — this thread generalizes that single condition into a family of deterministic
encoders at graded aggressiveness, enriching the 𝒱-information-retention comparisons;
(b) canonicalizing decoder *outputs* before comparison would separate stylistic from
semantic variance in the existing FizzBuzz/variance measurements; (c) the
"fewer variations to optimize over" motivation is exactly the wrapper-only sibling's
regime (2026-08-22 entry above); (d) transformation-generated equivalence-class
variants give the consistency axis (evaluation-methodology entry, chunk-8) its
code-side perturbations, complementing prompt-side rephrasings.

### 2026-08-23 — Chunk 10 of the early-2026 conversation: the canonicalization deep-dive (representation ladder, oracle-relative equivalence, IR-distance)

A long adversarial back-and-forth extending the chunk-9 thread (Danielle repeatedly
"stopping halfway through" to push back; the exchange ends mid-conversation — her
final message is answered in the next chunk). The technically durable outputs:

- **The Python representation ladder** (respondent): source text → CST (lossless
  w.r.t. text; LibCST/RedBaron) → AST (lossless w.r.t. executable semantics modulo
  introspection) → lowered/desugared AST with explicit control flow (non-standard but
  conceptually key: fewer degrees of freedom, still structured) → bytecode
  (many-to-one from AST) → CFG/SSA/trace views → native. Terminology settled:
  CPython (the C-implemented interpreter everyone uses) vs Cython (an opt-in separate
  language); general Python is not compilable to other languages because of dynamic
  typing/attributes, monkey-patching, reflection, eval/exec — compilable subsets
  (Numba, Cython, TorchScript, JAX traces) work by removing those features.
- **Test-oracle-relative equivalence — paper-ready formalization TLC should adopt
  (Claude-added flag):** three equivalence notions — textual, observational (full
  reflective indistinguishability), behavioral (same outputs/effects on inputs of
  interest). "Lossless is relative to the observer": under an execution-based
  black-box harness that never inspects reflective details, AST round-tripping and
  AST-level normalization are lossless. This is exactly TLC's C_s (Eq. 2) notion; the
  formalization pre-answers the reviewer objection to any normalization step in the
  pipeline. Caveat retained: the guarantee is as strong as the test set — true of the
  whole execution-based-eval paradigm.
- **Where Danielle's compile-down-and-back argument landed** after several rounds:
  one-to-many is not a blocker when any single canonical representative suffices
  (conceded — how decompilers/superoptimizers work); the real blockers are
  compositionality (IR chunk boundaries don't align with Python structure; meaning is
  context-dependent, so chunk-caching explodes), stability under iteration (naive
  decompiler-style round trips drift — structure oscillates, abstractions dissolve),
  and environmental equivalence (dependency versions, FP backends, threading). AST
  normalization is idempotent (normalize∘normalize = normalize); lower-IR round trips
  generally have no stable normal form.
- **Rewrite-theory boundary** (the math-vs-search resolution): a rewrite system is
  "math" when sound + confluent + terminating; AST normalization rules are convergent
  by construction; semantic-idiom collapsing ("these are both sorting") makes rules
  non-local/overlapping/non-confluent — classification is where the search lives,
  amortized offline into the ruleset. Behavioral equivalence is undecidable, so any
  canonicalizer is incomplete, heuristic, or oracle-backed; the honest claim is
  "policy-driven semantic normalization under a distributional prior." E-graphs /
  equality saturation named as the formalism sitting exactly between rewriting and
  search (no citation given; the standard reference is the egg line of work —
  Claude-added pointer, unverified).
- **The strongest salvage — IR-distance, don't round-trip:** compile *both* sides of
  the autoencoder down to a common view (bytecode n-grams, CFG fingerprints, def-use
  chains, normalized opcode sequences, call-graph patterns, or runtime traces on
  canonical inputs) and compare there. Uses: detect reconstruction drift ("passes
  tests but changed algorithm"), cluster solutions into algorithm families, rerank
  samples (minimal IR-distance subject to passing tests), regenerate under a
  preserve-structure constraint — "a semantic regularizer for generation." Proposed
  v0: 200 tasks × 6 models × 20 seeds; does IR-distance explain output variance
  beyond pass/fail; intervention = IR-distance-guided reranking. **(Claude-added:)**
  IR-distance between f and f̂ also directly operationalizes TLC-0's
  implementation-leakage measurement — a reconstruction that is IR-close to the
  source demonstrates the description carried implementation, not just behavior.
- **Danielle's refined hypothesis** (final message, not yet answered — near-verbatim):
  not that a C round trip helps LLMs across the board ("that would be a wild
  hypothesis"); rather that "there is an impact to having a consistent output format
  for each type of operation in Python… you dramatically reduce the state space,
  because then it is easier to compress components… it allows you to identify things
  that are in fact the same operation, and if you are a weak coding model… you can
  just track that whenever you see something like this, you've got to make sure to do
  this other thing." The identification/merging is best done "in a language where you
  have strict typing and a lot of other constraints… before then bringing it back" to
  Python. Note the explicitly compression-shaped statement of the benefit — this ties
  canonicalization to the thesis track's MDL framing, and the small/weak-model clause
  ties it to her elicitation interests.

### 2026-08-23 — Chunk 11 of the early-2026 conversation: the resolution — property-indexed equivalence and faithful abstraction

The canonicalization argument resolves, and this chunk contains what is probably the
conversation's core "aha." Danielle's marker, on being shown "you're not asking for
reversibility — you're asking for faithful abstraction… re-emitting Python is just
choosing a representative of the equivalence class": **"I feel so seen lol, this is
what I was trying to say at the beginning, but finally understand how to say now an
hour later!"**

**The precise hypothesis (respondent's faithful restatement, accepted):** for a large
class of real-world Python, forcing programs through a more constrained intermediate
language and re-emitting Python in a consistent form *reduces representational
entropy* in a way that improves downstream learning, recognition, and error
correction — especially for weaker models. About state-space reduction /
regularization / pattern alignment / compressibility / error predictability, not
semantic perfection. The intermediate language is a *regularizer, not a semantic
mirror* (kin: SSA, desugared core languages, theorem-prover IRs, BPE — "lossy but
regularizing"). Real risks: over-regularization (collapsing distinctions that are
signal — vectorized-vs-loop, stable-vs-unstable, precision choices), model mismatch
(weak models gain, strong models may not — itself an interesting result), translation
noise (tests + filters catch). Minimal test: Python → restricted typed IR → fixed
ugly-but-consistent Python style; measure pass@k, seed variance, error
predictability, transfer.

**Property-indexed equivalence (Danielle's formulation, formalized):** her framing —
define the properties of a code chunk that matter (big-O category, optionally its
constant, side effects on inputs, the output set) and prove equivalence on those
properties, algorithms-class style. Formalized: f ~_P g iff ∀p∈P, p(f)=p(g) —
observational equivalence under a projection; compound property sets like
{functional correctness, big-O, purity} are valid equivalence relations; the missing
word is **abstraction** (abstract interpretation / cost semantics are the named
frameworks). Under this: normalization = moving within a class; **compression =
collapsing class variance**; decoding = choosing a representative.

**Terminology decisions (Danielle's, recorded for adoption):** *canonicalization*
rejected with its precise definition (unique + idempotent + total representative per
equivalence class — soundness, uniqueness, idempotence; her instinct that ML usage is
aspirational was confirmed: entity resolution is approximate clustering +
representative selection, not canonicalization). Adopted vocabulary, in her order of
preference: **"abstracting programs into equivalence classes"** (favorite),
**"behavior-preserving normalization"** (the invariant named explicitly, no
uniqueness claim). Rejected: "representation regularization" (regularization
overloaded), "semantic X" (precision varies uncomfortably), "idiom standardization"
(linguistic baggage). Her keeper sentences, verbatim from the conversation:

> "We define program equivalence relative to a set of observable properties
> (functional behavior, complexity class, and effects), and treat programs equivalent
> under these properties as interchangeable for downstream learning and analysis."

> "Rather than preserving exact program structure, we abstract programs into
> equivalence classes defined by behavioral properties and operate on representative
> implementations to reduce representational variance."

North-star sentence (respondent's, thesis-shaped): "We study how abstracting programs
into equivalence classes under behavioral properties can reduce representational
variance and improve model reliability."

**The tokenizer thread:** her structural-tokenization idea (learned vocabulary over
Python with primitives larger than byte pairs; synthetic structure markers replacing
indentation; "templated tokenizing" — templates × tokenization) named as *program
vocabulary induction / grammar-aware structural tokenization*: identify frequent
normalized subtrees, replace with typed macro tokens (e.g. SORT_LIST(xs),
MAP_FILTER(xs, f, p)), define decoding rules back to Python; reversibility required
only up to a behaviorally valid representative. Caveats: BPE-style greedy local
merges fail on hierarchical structure (operate on parsed structure, not raw text);
tokens must be typed (arity/roles/constraints). Adjacent prior art named without IDs:
code2vec/code2seq, AST path vocabularies, neural grammar induction, e-graphs with
learned costs.

**TLC ties (Claude-added):** (a) property-indexed equivalence generalizes TLC's C_s —
the current oracle is P = {test-suite behavior}; adding complexity-class (BigO(Bench)
tooling, chunk 7) and purity/effect properties gives a graded family of stricter
success criteria for the same pipeline; (b) "compression = collapsing class variance"
states the compression thesis in the new vocabulary — the NL bottleneck is an
abstraction map, and the typed-macro-token vocabulary is a *deterministic,
enumerable* competitor to the NL latent (a baseline the compression project should
eventually beat or absorb — note it is close kin to the chunk-4 task-DSL operators);
(c) tension to manage: normalization destroys exactly the behavioral fingerprints the
divergence instrument (evaluation-methodology thread) wants to measure — normalize
inside the optimizer/autoencoder loop, do *not* normalize the divergence
measurements; (d) these keeper sentences and the vocabulary are candidates for
adoption into this doc's §1/objective statement — walkthrough decision.

### 2026-08-23 — Chunk 12 of the early-2026 conversation: the distribution constraint, the d-unification, and the obviousness check

The debrief after the resolution. Three durable results:

**1. The distribution-compatibility constraint (Danielle's realization, near-verbatim):**
black-box LLMs are "trained over a distribution of text, and you really can't change
that distribution… to have a high likelihood of improving the system, we almost
certainly need something that will return whatever tokenization, encoding, abstraction
that we produce into something that looks close enough to the distribution of Python
that the language models are able to operate over it… at least well enough that it
doesn't wipe out any gains from the effect of the abstraction." Respondent's
compression: you are designing "a preimage that must land back in the model's existing
token space" — reversibility is needed relative to the *training distribution*, not
just the abstraction. The induced experiment: x-axis = normalization/abstraction
aggressiveness ladder (raw → Black-only → desugar+rename → "ugly consistent"), y-axis
= performance/variance; predicted inverted-U — "that curve is the paper."
**(Claude-added:)** this constraint applies equally to TLC's NL latent: the
description d must lie in the decoder's expected natural-language distribution, and
aggressive structured/compressed descriptions risk falling off-distribution — the
pitch's COMP-NL vs COMP-SHORT contrast (strictness of the "Natural Language"
definition in the two-stage setup) is already a two-point sample of exactly this
ladder, so the U-curve experiment extends existing machinery rather than requiring new
scaffolding.

**2. The d-unification (Danielle's, verbatim — arguably the conversation's second
"aha"):** "my goal with the (f -> enc -> d -> dec -> f) setting is to produce 'd' a
'behavior-preserving normalization' of the code input 'f' to 'reduce representational
variance while retraining functional equivalence under a defined observation model.'
I just realized that my goal in both of these settings is literally identical, down to
trying produce that single specific entity! … one lived in the world of 'Variational
Autoencoders' and 'ELBO bounds' and 'RL-style optimization' and the other lived in the
world of linters because they are two very very different approaches (on the surface)
to accomplish the same thing." Respondent's alignment table: rule-based arm
(human-designed invariants, deterministic, partial-but-interpretable coverage) vs.
learned arm (objective + verifier, statistical, broad-but-opaque coverage) — same
target object d. This formally merges the chunk-9 "deterministic sibling" into TLC:
not a competitor project but the rule-based arm of the same objective, sharing
evaluation protocol and failure modes (over-regularization vs. distribution shift).

**3. The obviousness check (her "bubble bursting" question, answered honestly):**
the ingredients are standard in their home communities — abstract interpretation
(map programs into an abstract domain capturing the properties you care about),
e-graphs / equality saturation (equivalence classes as first-class objects + cost-based
representative extraction; third mention in this conversation — firmly litreview
material), denoising autoencoders (BART, 1910.13461, as many-surface-forms → regular
form). The cross-paradigm unification itself — one target object, two mechanism
families, one shared evaluation — is not standard ML furniture; expected reception is
"interesting — show me the curve," not "duh." Pitch sentence: "treat code
transformations, learned autoencoders, and prompt/agent optimization as different ways
of selecting representatives from program equivalence classes defined by behavioral
properties — and measure the abstraction/compatibility trade-off empirically."

**Additional keepers:** the canonicalization definition's strength registered
("deterministic, total, and idempotent… an incredibly strong statement lol") — her
one-line journey summary: "I thought I was arguing for canonicalization, but I was
actually arguing for equivalence-class abstraction with representative selection under
behavioral constraints." Terminology cheat sheet recorded in her notes (transcript,
chunk 12). New metric vocabulary that TLC-0 can absorb: **class consistency** (do two
surface forms land in the same class), **intra-class variance**, **representative
quality** (does representative choice change downstream success), **abstraction
error** (when collapsing variance destroys signal). The disambiguation line: "by
canonical, we mean canonical in the space of behavioral abstractions, not canonical
syntax." The "one knob + one plot" experiment template (name the knob, name the
y-axis, hold all else fixed, 3–5 settings × 2–3 models × enough seeds, one figure) —
matches her stated aspiration to convert intuitions into crisp single-question
experiments; walkthrough candidate alongside the other principles.

### 2026-08-23 — Chunk 13 of the early-2026 conversation: why the latent must be natural language

The conversation's most direct payoff for TLC's core design. Chunk 13 also closes the
compile-down question (LLVM front/middle/back-end mental model confirmed; binaries
decompile only to heuristic pseudocode — Ghidra/Hex-Rays-style — because compilation
is intentionally many-to-one; no faithful source recovery) and endorses Danielle's
"collect them" structuring idea (below). But the center is this:

**The advisor's objection, and the answer (early 2026).** The advisor's strong
pushback on the pitch was: if the objective is a behavior-preserving compressed
representation optimized on a behavioral reconstruction loss, *why does the latent
have to be natural language* — it could be any compressed representation. Danielle
could feel but not articulate the answer; the distribution-compatibility constraint
from chunk 12 is it. The keystone sentence:

> "Because the encoder and decoder are black-box, pretrained LLMs, the intermediate
> representation must live in a region of representation space where those models
> already have strong priors."

Formal sibling: "Given fixed, pretrained encoder–decoder models, any intermediate
representation must reside within a region of input space with sufficient prior
support to allow reliable encoding and decoding." Unpacked: with no retraining, no
tokenizer changes, and multiple general-purpose decoders, the only shared,
high-capacity, semantically grounded channel is natural language plus code — a custom
DSL, symbolic graph encoding, dense vector-in-text, or novel markup has no prior
support and dies at decode time. Danielle's own refinement of the claim: **NL is a
prediction, not a requirement** — "I'm not arguing that we need to constrain the
intermediate representation to natural language. I'm predicting that the best
performing intermediate representations will be some form of human interpretable text
(due to the training data)" — which then justifies NL-instruction inductive biases in
initial experiments as probability-of-success maximization, not dogma. The
respondent's strengthening lever: the argument is strongest under the **cross-decoder
robustness** requirement (d must decode across model families, not just the encoder's
partner) — a single fixed decoder pair might learn an idiosyncratic code; shared
priors are what force NL. Note this makes "transfer of d across decoders" a *test of
the NL-necessity claim itself*, connecting to the conversation's transfer-matrix
thread. Precision caveat for the paper: "representation space" here means input token
sequences with training-distribution support, not embedding/latent space.

She drafted a written version of the argument for the advisor at the time
(Context / Method / Hypothesis / prediction-not-requirement / inductive-bias
justification — full draft and the correctness review preserved in the transcript,
chunk 13). Historical status: drafted the evening before sending, ~4 days before the
workshop deadline; outcome not recorded in this conversation.

**(Claude-added:)** this argument belongs in this doc's core rationale (§1), not just
here in §4 — it answers the project's most predictable objection, converts the NL
choice from taste into a falsifiable constraint-driven prediction, and names its own
killer experiment (NL vs. non-NL vs. NL-likeness-ladder d under fixed decoders — the
chunk-12 inverted-U, plus cross-decoder transfer of d). Walkthrough item.

**The "collect them" idea (hers):** the abstraction pattern appears across algorithmic
domains, shaped by each domain's history and admissible assumptions; a structuring
activity is to collect instantiations, extract the portable pieces from the
domain-specific ones, and accumulate a comparison/reuse suite. Respondent's domain
map: PL (abstract interpretation, SSA/IRs, e-graphs, superoptimization), algorithms
(complexity classes, reducibility), ML (denoising AEs, VAEs, contrastive invariance),
RL (state abstraction, MDP homomorphisms), formal methods (bisimulation, trace
equivalence), databases (query normalization, logical-vs-physical plans). Practice:
a two-column notebook — object type × properties defining equivalence, plus how the
representative is chosen. **(Claude-added:)** this is a natural seed for a new
reference accumulator (cross-domain abstraction mechanisms) if the thread stays
live — walkthrough item.

### 2026-08-23 — Chunk 14 of the early-2026 conversation: cross-decoder positioning, the closing experiment suite, bisimulation, and representation scoring

**Cross-decoder decodability — settled positioning (Danielle's decision, early 2026):**
she recognized it is *not* a standard autoencoder assumption (encoder and decoder are
normally co-adapted; latents are decoder-relative) and chose not to include it as a
requirement — but the system should be designed to permit swapping cheaply, because
swappability strengthens the method and analysis. Respondent's hierarchy, adopted:
(1) primary constraint — black-box pretrained models force d into high-prior regions;
(2) prediction — those regions are NL-like; (3) secondary emergent benefit /
evaluation axis — cross-model decodability. Boundary sentence for
reviewers: "We don't assume decoder interchangeability in the autoencoder sense; we
treat cross-model decodability as an empirical property of the learned representation
that may increase its utility." Also the memorable framing: "You are not learning a
latent from scratch; you're discovering a representation that already aligns with
existing priors."

**The requested closing experiment suite (consolidating the conversation's
hypotheses):** E0 — fix the observation model (P1 tests; optional P2 side-effect-free
via sandbox + static checks; optional P3 complexity proxy via timeout/scaling
buckets). E1 — the distribution-compatibility curve (five-rung input-normalization
ladder: raw / Black / +alpha-renaming / +desugaring / ugly-consistent; ~200 tasks ×
N seeds; success + CI vs. rung). E2 — d̂ format comparison at matched budget:
freeform NL summary vs. structured-NL JSON (purpose/inputs/invariants/examples) vs.
invented pseudo-DSL; the NL-necessity test. E3 — cross-decoder transfer matrix
(2–3 decoders × formats); tests "d lives in shared prior space." E4 —
equivalence-class variance compression: structural distance (AST features, opcode
n-grams, CFG fingerprints) among test-passing solutions, before vs. after
normalization/representation intervention. E5 (optional) — planner-vs-implementer
sketch tasks. Recommended first wave: E1 + E2 + E3.

**Bisimulation as the formal frame (her connection, from prior exposure to
friends' POMDP/procedural-env work):** the recursive bisimulation metric (reward
difference + discounted Wasserstein distance over next-state class distributions)
maps onto the project as program↔state, test suite↔reward, execution↔transitions,
nuisance syntax↔pixel noise, normalization↔state abstraction, NL d̂↔learned state
embedding; contrastive learning = approximate bisimulation (collapse nuisance, keep
task-relevant differences). Grounding sentence offered: "our formulation can be
viewed as a form of behavioral bisimulation… analogous to state abstraction in RL."
Her take: bisimulation is a drop-in *definition* upgrade; contrastive learning is the
tractable *mechanism* — "quick learnings." **(Claude-added caveat:)** the mapping is
weaker than the respondent claims — programs here are single-shot, not sequential
state-action processes, so "bisimulation" largely degenerates to observational/
contextual equivalence (the established PL term); useful as intuition-bridge and for
borrowing RL abstraction-quality tools, but a PL reviewer may read "behavioral
bisimulation" as overdressed. All uncited; no ledger rows.

**Representation scoring (her question — the missing piece she named: "a good way to
evaluate the representations themselves"):** respondent's grounded menu — scalar
quality Q(d) = S(d) − λ·V(d) − µ·C(d) (decoding success, seed variance, token cost;
needs no training); **cross-decoder agreement** S_multi(d) = mean pass probability
across M decoders (operationalizes shared-prior-space directly — note this is the
NL-necessity claim restated as a per-representation metric); contrastive variants
(pairwise ranking scorer g(d); InfoNCE embedder with inter/intra-class distance ratio
or silhouette-style indices); then optionally train a small scorer to predict Q from
text features — a learned ordering with a grounded target. Equivalence labels between
representations must be **soft/in-expectation**, not pass-all-or-none: pass-rate
similarity y = 1 − |p(d_i) − p(d_j)| to start, upgraded to per-test outcome-vector
similarity (cosine/Jaccard/KL over per-test pass probabilities) — "they fail and
succeed on the same things." **(Claude-added:)** the outcome-vector version is the
meaningful one (pass-rate similarity alone conflates different behaviors with equal
rates), and it extends machinery this project already validated — the 2026-07-11
fractional-test-pass-rate and per-test-outcomes entries above; Q(d) and S_multi are
direct candidates for TLC-0's representation-quality axis alongside IR-distance
(chunk 10) and the class-variance metrics (chunk 12).

### 2026-08-23 — Chunk 15 of the early-2026 conversation: framing adoption, the bandit baseline, and the conversation's own design digest

**Optimizer-loop baseline decision (early 2026):** the advisor's suggested loop —
sample actions, keep any that beat the current reward, stop after n non-improving
steps or when improvements peter off — adopted as the main baseline (simple, matches
common practice), with Danielle's recognition that it is a greedy multi-armed-bandit /
hill-climbing variant and her commitment that bandit formulations (ε-greedy / UCB /
Thompson) should at least be *tried*, appendix-level. Note the convergence with the
chunk-3 curriculum thread: bandit machinery is budget-allocation machinery, and the
cost-aware evolutionary prompt-optimization citation (PMLR zehle25a) already sits in
exactly that space. (Also note the pitch's §7 named "a simple ES based optimization
approach" as the intended next step — the greedy loop is the even-simpler successor.)

**Framing-adoption decision ("lens, not commitment"):** her worry — "it feels pretty
substantial to shift our framing so dramatically to a much more formal version
overnight" — answered with a rule she accepted: workshop main text stays operational
(test-based equivalence, executable definitions); each formal lens gets exactly one
bridge sentence (greedy loop "can be viewed as a greedy multi-armed-bandit-style
optimization"; equivalence "closely related to behavioral abstraction and
bisimulation"; ranking "can be interpreted as a contrastive objective"); explicit
losses, bandit baselines, and formalisms live in the appendix; the full formal
unification is the follow-up paper. Decision table (main / appendix / future-work per
concept) in the transcript. Sorting rule: changes-how-you-interpret-results → main
text; changes-how-you-might-implement-later → appendix/future work.

**The conference-paper horizon (recorded so it can be shelved):** a unification +
systematic-measurement paper — formalize the observation model, treat intermediate
representations as elements of an abstract space scored by invariance, recoverability,
and downstream utility; compare abstraction mechanisms head-to-head (heuristic
normalization, greedy/bandit search, contrastive objectives, learned encoders); the
central object of study is the abstraction–compatibility trade-off; contributions are
variance-reduction, robustness/transfer, and success/failure conditions per mechanism
— explicitly *not* a new algorithm. Venue sketch (respondent's, generic but
reasonable): ICLR/NeurIPS under a representation-learning framing, ACL/EMNLP Findings
under a code-LLM framing, PL+ML crossover or D&B under a measurement framing.

**The conversation's own design-decision digest (next-day request):** the respondent
produced a consolidated digest — shared foundations (property-indexed equivalence;
behavior-preserving normalization as umbrella term; CST/AST for rewriting vs. IR for
comparison; the black-box distribution constraint; the one-plot trade-off) and eight
project modules (workshop core; greedy-vs-bandit optimization; representation
evaluation; the L0–L4 normalization ladder; the macro-token vocabulary — with one new
detail: **start with a hand-designed macro set of 10–30 constructs before any learned
vocabulary**; task/benchmark building blocks; behavioral distance; venue/packaging).
The digest is preserved verbatim in the transcript (chunk 15); its content is already
distributed across this doc's chunk entries and the evaluation-methodology/
benchmarks-landscape entries, which serve as the durable, cross-referenced version.

### 2026-08-23 — Chunks 16–17 (session of 2026-02-08): the variance data becomes the divergence paper; conversation ends

The conversation's final session (two days after the pitch-week chunks; the timeline
is now precise — chunks 1–15 on 2026-02-06, four days before the Recursive
Self-Improvement workshop deadline of 2026-02-10; chunks 16–17 on 2026-02-08). TLC
facts surfaced: the respondent's recap confirms the draft's variance instrumentation
end-to-end (C_s; decoder-only vs. full-reconstruction pass distributions; the
waterfall; FizzBuzz ×100; 3 synthetic families × 10 specs × 10 generations; provider/
cost-tier design; DG-* prompt regimes as a contract-entropy axis; one-stage vs.
two-stage; COMP-NL/COMP-SHORT; b1 ∈ [0.5, 1.0] budget sweeps). In chunk 16 Danielle described having
an AST code-metrics analysis over generations (control-flow condition presence,
solution complexity, variable-name/line/total lengths, comments, docstrings);
**corrected at intake (2026-08-23): no AST-metrics code exists** — she has
code-processing codebases to point to instead (pointers to follow). She was also planning a *second* workshop submission
at the time: contradictions within (model, spec) pairs under a structured latent, for
the Logical Reasoning workshop — a TLC-adjacent plan whose outcome is unrecorded.
The variance-paper spec itself ("Behavioral non-collapse in cheap coding LLMs",
targeted at the Re-Align Challenge track) lives with the divergence thread:
`../topics/reference/evaluation-methodology-literature.md`, chunks-16–17 addendum;
full verbatim spec in the transcript bundle, chunk 17.

### 2026-08-23 — The actual submission, its reviews, the post-mortem presentation, and the Miao & Blunsom anchor (project history, early 2026)

Documented at intake from four artifacts Danielle preserved together (all now in the
convo-artifacts bundle): the submitted paper, the OpenReview reviews, her 2026-02-25
advisor presentation, and the advisor-recommended reference. She marks the
presentation plus that reference as the turning point of the discussions around this
project.

**The submission.** "Prompt Optimization for Behavioral Code Compression: Bandits vs
LLM-in-the-Loop Search," submitted to the **ICLR 2026 Workshop on Lifelong Agents
(LLA)** — a third venue, distinct from both the pitch's target (Recursive
Self-Improvement, deadline 2026-02-10) and the 2026-02-08 plan (Re-Align Challenge).
Content: frozen black-box encoder–decoder; encoder-prompt selection as a multi-armed
bandit over a structured template (3 slots × 3 phrasings incl. omission → 27 arms;
expanded grid 8,064 arms with |A| ≫ N); UCB1 vs. Thompson sampling vs. LLM-in-the-loop
optimizer; headline metric the **compression–correctness frontier** S_m(c; N) —
fraction of tasks with ≥1 passing reconstruction of realized IR length ≤ c within
budget N (N = 90), summarized by frontier AUC; training on Stateful Algorithms, |D| =
50, medium difficulty; OOD with frozen selected prompts on HumanEval++, a harder
bit-operations family, cross-language Java/Rust, and Gemini-Flash-Lite decoder
transfer; 3 seeds, 95% CIs. Findings: in the enumerable grid all three methods find
near-identical frontiers and the oracle gap stays open (the budget/feasibility
landscape, not exploration policy, is the limiting factor); in the large space the
LLM optimizer shows modest but consistent sample-efficiency gains; transfer is
strongly domain-dependent (bit-ops partially reverses the ordering; Gemini transfer
drops further — decoder-specific contracts). Note the chunk-15 decision inverted en
route: bandits went from appendix candidate to co-headline. Writing context recorded
for interpretation: the deadline slipped 12 hours, her collaborator fell ill, and
Danielle wrote the entire paper herself across two consecutive nights.

**Reviews (verbatim in the bundle's `openreview-reviews.md`).** R1 = 7 (accept): clean
frontier metric, careful regime separation, honest conclusions; wants formal
rate–distortion grounding and links to best-arm identification. R2 = 5: synthetic
scale (50 functions), heavily templated prompt space, modest gains unexplained,
unit-test-only oracle may overestimate equivalence, wants seeds/ablations/decoder
sensitivity. R3 = 5: **novelty — GEPA (2507.19457) and the SOTA prompt-optimization
line undiscussed** (the omission the litreview plan exists to prevent; GEPA was
already the doc's named incumbent); synthetic data; narrow model pool (gpt-5-nano +
claude-haiku-4.5 in the loop). Area Chair recommended **Accept (Poster)** ("fits the
workshop theme to some extent"); **Program Chairs rejected** — an AC-overridden
reject, consistent with theme-fit/capacity rather than technical assessment.

**The 2026-02-25 presentation** ("Code Compression Results + DQE Discussion," 1:1) —
franker than the paper: "Basically no differentiation between the 3 methods attempted
on our 'training' setup, even in large search space where N_arms ≪ T"; generalization
experiments differentiate but were not thoroughly analyzed — "not clear patterns."
Per-stage retro with next steps: (1) difficulty calibration and task sampling
distributions matter — better difficulty labeling and validation (true uniqueness,
test coverage); (2) refine the prompt search space so methods *can* differentiate;
(3) reconsider reward design and the history the LLM-as-optimizer receives; (4) dig
into whether generalization-only differentiation was a design decision or an
implementation quirk. Overall: need more task and model diversity for general
conclusions. **The DQE slide is a research-statement formulation worth preserving
(near-verbatim):** "I'm interested in how to build ML-style modular systems out of
frozen pretrained LLMs… where the trainable object is the system… where we optimize
using verifiable feedback. My goal is to understand and improve **reliability**,
**cost-efficiency**, and **broad utility** (like OODG) by designing the right
benchmarks & optimizing intermediate representations → with special interest in using
small/cheap models."

**The advisor-recommended anchor: Miao & Blunsom 2016, "Language as a Latent
Variable: Discrete Generative Models for Sentence Compression" (1609.07317; PDF in
the bundle, metadata verified against the PDF).** Auto-encoding sentence compression
(ASC): a latent *summary sentence* is drawn from a background language model, and the
observed sentence is reconstructed conditioned on it; discrete VAE trained with
REINFORCE; combined with a supervised compression model for semi-supervised learning.
This is the closest formal ancestor of TLC's setup. **(Claude-added:)** the
background-LM prior on the latent is the trainable-era implementation of exactly the
distribution-compatibility constraint from the February conversation — Miao & Blunsom
enforce NL-ness of the latent with an explicit LM prior in the objective, where TLC
gets it implicitly from the frozen models' pretrained priors, with the reconstruction
likelihood replaced by a test-suite oracle. That is a one-sentence lineage for the
related-work section, and the paper belongs in TLC litreview gate 1 alongside GenDLN
and ShortenDoc.

**Framing provenance of the submitted paper (Danielle at intake, 2026-08-23,
near-verbatim):** the compression framing used in the submission was the advisor's;
"as I was writing I came to the conclusion that I guess we should use the compression
framing my advisor dictated but I didn't really understand it and really I understood
my previous framing much better but I hadn't thought it through sufficiently to
either do both or do mine in a way that wasn't a clear 'I'm rejecting your
instructions' way. so I tried very hard to figure out how to do his with no sleep in
very very little time." A separate conversation from the writing period documents
this framing confusion — intake to follow. The DQE statement (README program-level
note, 2026-02-25) was her attempt at her own framing; it was not adopted in that
discussion.

### 2026-08-23 — The writing-sprint conversation, chunk 1 (metric origin; historical, 2026-02-16)

Second conversation intake (verbatim companion transcript:
`~/drotherm/data/convo-artifacts/2026/2026-08-23-prompt-opt-reeval-aha/writing-conversation-transcript.md`).
This is the conversation Danielle used while writing the LLA submission during the
two-night sprint; her intake framing: it documents the pieces she was confused about
while implementing the advisor's compression framing over her own better-understood
one.

**Chunk 1 — the origin of the paper's headline metric.** Three exchanges:

1. Her opening question: with prompt selection framed as a finite-arm MAB (UCB1 /
   Thompson vs. LLM-in-the-loop), what does "maximizing reward" mean — best-arm-at-end
   evaluated on a test set, or cumulative reward/regret over matched train time? And is
   it fair that the LLM optimizer isn't told how it will be evaluated? Answer: these
   are two different problems (best-arm identification vs. cumulative regret) with
   different exploration profiles; UCB1/TS are implicitly cumulative-regret algorithms;
   **telling the agent the evaluation metric is problem specification, not privileged
   leakage** (unfair = asymmetric horizon knowledge or test-distribution details);
   LLM-as-policy-class framing; the semantic prior makes it structured bandits / side
   information, which should be said, not hidden.
2. Her disclosure of the real confusion: the paper's pitch is compression, the reward
   is hit-budget-and-pass-tests, she had imagined **x = instructed budget** vs. success
   while the advisor wanted **x = realized IR cost** (a Pareto front), and "I just
   don't even at this point know what I'm claiming… all of the hyperparameters
   directly impact which methods you'd expect to be better." Answer: three conflated
   axes (search procedure / search objective / compression tradeoff); the end goal is
   the **frontier under a fixed evaluation budget**, not reward; internal scalar reward
   (1[pass ∧ cost ≤ B]) is fine as the optimization signal but not the evaluation; and
   the two-story split — **Story A** (fair comparison on the same finite arm space)
   vs. **Story B** (scaling: LLM-optimizer on spaces bandits can't enumerate) — which
   is exactly the 27-arm / 8,064-arm structure the submitted paper ended up with.
3. Her deepest confusion ("you are describing the same thing as my advisor, but I
   also did not understand, when he explained it, how we would collapse what we're
   doing into that"): is the instructed budget another arm? how do specs-per-pool and
   samples-per-spec collapse? Answer — the unlock: **budget is a conditioning
   variable, not an arm**; the frontier is computed from realized (cost, success)
   outcomes, so any budget-cap view is recoverable from logs by filtering
   (SuccessRate(B) = E[1(success ∧ cost ≤ B)]); one evaluation = one
   encode→decode→test execution, making pooling hyperparameters collapse into
   evaluation-budget accounting; per-spec envelopes averaged across specs; F(c) =
   fraction of specs solvable under cost c. The closing "drop-in paragraph" is
   near-verbatim the submitted paper's frontier definition (S_m(c; N), Eq. 1).

**Intake notes (Claude-added):** (a) the instructed-budget vs. realized-cost
distinction established here survives into current TLC (B = b1·|d_s| + b2 is the
instruction; frontiers are on realized cost); (b) R1's later review asked for exactly
the formal grounding this exchange deferred — "how does this relate to constrained
optimization or best-arm identification?" — the reviewer independently found the seam
("don't try to force regret/BAI framing" was the right deadline call, and the question
is still open for the follow-up); (c) her "all of the hyperparameters directly impact
which methods you'd expect to be better" is an honest statement that the
method-comparison claim was under-determined — which the results then confirmed
(no differentiation in-distribution).

### 2026-08-23 — The writing-sprint conversation, chunk 2 (framing decision and Eq. 1 exact form; 2026-02-16)

(Date corrected at intake: the writing-sprint conversation is from **2026-02-16**;
source link in the companion transcript header.) Four exchanges:

- **The framing question — and the clearest early statement of the original vision.**
  Danielle (near-verbatim): "Initially, I had a very different framing for this
  project, and in that framing, I was pretty confident that the LLM as optimizer would
  perform better. Now, the paper's framing has changed so much that it's no longer
  clear…" Her novelty claim, from her own substantial lit review: the *combination* —
  black-box LLM encoder–decoder code compression + behavioral-equivalence
  reconstruction + compression Pareto frontier — did not exist. Her real motivation:
  "using LLMs as optimizers with systems that have LLM components… to fit them in the
  same way that you would fit a trained model with weights… you can actually replicate
  many different types of algorithms and explore a lot of interesting concepts related
  to representation learning, identifiability, continual learning… I wanted to put out
  a paper that included each of these pieces to stake a claim." (This is the DQE
  research statement of 2026-02-25, nine days earlier and near-verbatim; "framing C"
  below is its packaging.) The dilemma: the LLM optimizer might only win in settings
  "clearly skewed towards it."
- **The three framings.** A: problem + evaluation protocol + baseline suite (robust to
  mixed results); B: regime characterization with a-priori-defined regimes ("re-label
  skewed as the intended deployment regime" — large space / low budget is realistic,
  not skewed); C: the platform paper ("training-by-querying LLM systems") — her true
  interest. Decision rule taken: **A + one paragraph of B**, which is precisely the
  submitted paper's shape; C was deferred (and resurfaced as the DQE statement).
  Workshop-safe vocabulary list (introduce/characterize/competitive; never
  SOTA/consistently-outperforms/optimal) — the paper follows it, and R1's "does not
  overclaim… honest conclusions" praise is the direct payoff.
- **Metric details.** AUC-vs-evaluations chosen as primary sample-efficiency summary
  (robust, no threshold; C_max by e.g. 95th-percentile cost) with min-cost@target as
  optional secondary under pre-registered targets. Her proposed Figure-1 definition,
  corrected from "best success achievable" to per-spec-then-average with "best
  *observed* success": S_{m,s}(c;N) = max y_i over attempts on spec s with cost ≤ c;
  S_m(c;N) = mean over specs — **the exact Eq. 1 and near-verbatim Figure-1 caption of
  the submitted paper originate here.**
- **The coverage-vs-probability decision under time pressure.** Her question: can the
  binary ever-succeeded envelope stand in for the pass-probability version, given
  repeated spec sampling — "I'm not necessarily optimizing for this to get accepted.
  I just want to submit something plausible." Answer: yes, if described as *coverage*
  ("at least one passing reconstruction… differs from estimating the per-attempt
  success probability"), and if N counts total decode+test attempts, not pools. The
  submitted paper adopted the coverage wording ("finds at least one passing
  reconstruction"). **(Claude-added:)** R2's later critique ("relies solely on
  unit-test pass rates… stronger statistical rigor, more seeds") presses on exactly
  the probability-vs-coverage seam accepted here; the pass-probability frontier
  (the deferred "optional enhancement") is a known upgrade for any revival, and the
  per-test outcome-vector machinery recorded in the TLC-0 candidate metrics is its
  natural implementation.

### 2026-08-23 — Current state: the compression thesis's origin, the decoupled-sampling insight, and the optimizer infrastructure (Danielle at intake)

Three current-state additions from the intake dialogue (not the historical
transcripts):

**Why the compression track demands a strong optimizer — provenance of the goal.**
Danielle's resolution of the framing confusion (near-verbatim): in later conversations
she asked "but if our goal is compression why would we ever setup a system like this
one?? aren't there better ways?" and ultimately realized the advisor "was certain that
by combining lossless and lossy compression we could beat all lossless compression at
least on a pareto frontier level. which I agree is true in theory, but I feel like in
practice claiming that all we have to do to write an easy straightforward paper is to
**beat LOSSLESS COMPRESSION** seems kinda crazy. but it is the task we now have, which
means we actually do need a very well tuned complex prompt optimization algorithm
because the baseline is a crazy hard bar that has been aggressively tuned for
decades." This is the origin of TLC-1–3's stated goal (this doc's one-line pitch) —
now recorded with her difficulty assessment attached: the lossless bar is what forces
the optimizer sophistication, which in turn is why she "ended up doing prompt
optimization when that's not really what I thought I was doing."

**The unarticulated benefit of the original fixed-arm design — decoupled sampling.**
Her articulation (near-verbatim): the initial design "took a very complicated pipeline
and made it so you could do a substantial amount of sampling up front (the
infrastructurally hard part) while still being able to do 'optimization' because the
search space is basically fixed." By contrast, live prompt-optimization loops need a
distributed system: the inner step is prompt → remote inference → result → remote
inference → result → compute-heavy parsing → sandboxed evaluation, and "when you add a
complex optimization loop around that its actually kinda a nightmare" — she has spent
more than a month getting to first optimization results. **(Claude-added:)** this is a
real architectural axis worth keeping explicit — fixed-space designs permit
sample-once/optimize-offline (and offline re-analysis under any metric, as the
February frontier-from-logs collapse exploited), while open-space optimizers buy
expressiveness at the cost of durable-workflow infrastructure; the two-regime paper
structure was also, implicitly, a comparison of these infrastructure classes.

**Optimizer infrastructure status (verified locally 2026-08-23).** The live
optimization harness is "basically fully implemented" for **COPRO, MIPROv2, GEPA, and
a Codex direct prompt proposer** on her dr-* stack — confirmed against the whetstone-ai
README (COPRO and GEPA platform-wired through `submit_optim_run` with sandboxes;
MIPROv2 and Codex-direct live via the in-process harness; Codex-direct is the only
tool-using optimizer, macOS-only). All repos public, local, versioned, and active
(whetstone-ai 0.1.13 and whetstone-envs 0.2.4 committed 2026-08-23). Design principle
(hers): "agents will often mess with things they shouldn't so I tried to pull each
level of primitives out so that we could freeze them during exp running without
concern + update different pieces with clear versioning over time." The stack, with
her one-line descriptions:

- `dr-serialize` (0.1.2) — canonical serialization; core to giving runs stable IDs
- `dr-wire` (0.1.0) — HTTP helper
- `dr-providers` (0.3.2) — LLM provider access: param validation (no silent
  failures), unified API
- `dr-graph` (0.1.3) — the graph definition that *is* the inner loop being optimized
- `dr-store` (0.2.6) — storage primitives of different shapes
- `dr-exec` (0.1.14) — CPU jobs with different parallelization types
- `dr-platform` (0.2.7) — durable workflow primitives built on DBOS
- `whetstone-ai` (0.1.13) — the optimization algorithms + eval engine (bootstrap CIs,
  power analysis, anchor calibration over persisted evidence)
- `whetstone-envs` (0.2.4) — simpler test environments; **the minigrid-based env is
  one Danielle "probably want[s] to spec out into a small project"** (flagged; her
  decision)
- `dr-code` (0.2.0) — code operations

**Missing piece:** the `code-comp` repo (not yet written) that composes whetstone-ai
and dr-code into the actual TLC experiments. The February submission did *not* use
this stack; the next round will.

### 2026-08-23 — The writing-sprint conversation, chunk 3 (AUC mechanics and the OOD protocol; 2026-02-16)

**Provenance note (Danielle at intake, near-verbatim):** "my ultimate goal was that
basically nothing was copied verbatim from this convo into the paper, I honestly can't
remember in the rush in the end whether I got to rewriting all the sections or not.
but regardless, even if I was to submit to exactly the same venue with the same
results I'd do a substantial rewrite." (Spot-check at intake: the submitted paper's
§3.2 matches the drafted generalization paragraph in structure and content but not
wording — consistent with rewriting, at least there.)

Chunk 3 content: (1) the AUC computation recipe — the envelope *is* a curve (a step
function); trapezoid integration over a cost grid, normalized by C_max (fixed cap or
95th-percentile of observed costs), np.trapz; gotchas: N counts total decode+test
attempts (not pools), per-seed AUC then average with error bars. (2) Her
generalization-experiment notes, corrected into the paper's §3.2 protocol — the two
methodological saves: **c_0 chosen from training data only** (no test tuning; the
paper adopted this phrasing) and **sweep the post-hoc cost cap computed from realized
costs, not the instructed budget** (the February collapse principle re-applied at
test time); plus per-method prompt-selection symmetry (same train score rule for
bandits and the LLM optimizer) and k=3 decodes with coverage/pass@k-under-cap.
(3) Drafted generalization-intro and Figure-2 caption templates. **(Claude-added:)**
the drafted Figure-2 plan (AUC(N) vs. evaluations) partially changed en route — the
submitted paper's Figure 3(b) plots cumulative success rate vs. evaluation number
instead, with normalized frontier AUC appearing as the OOD summary panel (Figure 2,
right); and the np.trapz-level question arriving in the same session as the framing
decision is a fair snapshot of the sprint's working conditions.

### 2026-08-23 — The writing-sprint conversation, chunk 4 (the working document and the final calibration pass; 2026-02-16)

Her consolidated Notion-style working document — the intermediate artifact between the
conversation and the paper (framing bullets, goal, Figure 1/2 definitions, the
generalization protocol, with the chunk-3 drafts embedded wholesale as asides) — and
the assistant's four final corrections, all of which visibly landed in the submitted
paper: (1) reserve "Pareto" for prose; the math object is the empirical
compression–correctness frontier / coverage–cost curve (the paper's usage matches);
(2) drop the "selects the minimum-cost IR" framing — the frontier is a property of
the attempt *set*, not a selection (the paper's Figure-1 text uses the set-of-attempts
form); (3) the cleaned per-spec formula (the paper's Eq. 1 exactly); (4) the
evaluation-accounting sentence — "one evaluation = one end-to-end encode→decode→test
execution; all methods compared at the same total N" — flagged as preventing "80% of
fairness questions" (the paper: budget "measured in number of end-to-end
reconstructions and test-suite executions"). Nice-to-haves: specify C_max and the
train-only c_0 rule.

**(Claude-added:)** the working document itself is provenance-relevant: it embedded
assistant drafts verbatim as asides, i.e., the verbatim material lived in the *notes*
layer, with the intended rewrite happening at notes→paper — which is exactly the step
her intake note says she can't remember completing everywhere. The transcript
preserves the document with explicit placeholders for the re-embedded chunk-3
material (byte-identical text already in the same file), noted inline.

### 2026-08-23 — The writing-sprint conversation, chunk 5 (the section rewrites — and the provenance answer; 2026-02-16)

The rewrite-from-scratch plan collapsed under time ("there's no way I can rewrite from
scratch fast enough. I'm going to ask you to take the section we have and give me a
better one and then I'm going to edit that down") — and this chunk substantially
answers the provenance question from her intake note. Comparing the drafts here
against the submitted PDF:

- **Abstract — mixed provenance, mostly hers.** Lineage: assistant long draft → her
  compact rewrite in her own words ("will this approx work?") → assistant polish →
  submitted abstract ≈ the polish of *her* text. The base voice is hers.
- **Background & Methods — assistant rewrite, edited.** Her old §2 was the *pitch
  formalism* (θ = (θ_E, θ_D) harness configurations, the Feas waterfall with the
  budget term, Succ, J(θ)); she asked for it to be "completely rewritten," and the
  replacement swapped the harness-configuration formalism for the arm/frontier
  formalism. **This exchange is the pinpointable moment the system-fitting framing
  left the paper** — the deleted formalism is precisely what this project doc's
  current §1/§3 restored and extended in July–August. The submitted §2 follows the
  replacement's structure with her edits; "per-spec envelope" became "per-function
  solvability indicator," picked from the alternatives list she requested.
- **Related Works — shipped closest to the draft.** Base text hers; the three
  assistant contrast sentences appear in the submitted paper nearly verbatim ("In
  contrast, we study a fully black-box LLM encoder–decoder and evaluate
  reconstructions by unit-test behavioral equivalence under an explicit compression
  cost"; "Our work differs in focusing on behavioral reconstruction…"; "We complement
  this line of work…"). For any resubmission this is the first section to rewrite —
  doubly so since R3's GEPA objection means it needs restructuring anyway.
- **Experiments — assistant full rewrite; the submitted §3 follows its skeleton**
  (benchmarks / system / two-regime structure / metrics / generalization), with her
  numbers and edits. The old section's commented-out fragments preserved two
  artifacts worth noting: the variance-experiment text (FizzBuzz ×100, 10×10
  sampling) that never fit the compression paper — the orphaned sweep of chunk 3 of
  the main conversation — and a commented-out figure sketch ("off-diagonal failures
  highlight that seemingly well formed compressions actually encode model-specific
  contracts instead of portable semantics") that is the encoder×decoder transfer-
  matrix idea, drafted in February and still unrun.
- The "truth checks" pattern continued (uniform vs. adaptive spec sampling; does the
  LLM optimizer pick from A or propose new prompts) — the assistant's
  don't-accidentally-lie discipline, which the honest final paper reflects.

**Net provenance answer (for her records):** the submitted paper is a mix — abstract
mostly hers, methods/experiments assistant-skeleton with her edits, related works
closest to assistant text, with the three contrast sentences shipping near-verbatim.
Her stated bar ("nothing copied verbatim") was met in some sections and not in others;
her planned substantial rewrite for any resubmission stands, with Related Works first.

### 2026-08-23 — The writing-sprint conversation, chunk 6 (style and citations; 2026-02-16)

Two lighter exchanges, both of which landed in the submitted paper. (1) Her style
check — "is this 'tons of little paragraphs' style standard or are you copying my
collaborator's writing style?" — answered with the three-variant taxonomy (paragraph
headers: scannable, deadline-friendly, choppy; subsections with prose: polished;
hybrid recommended: 3–5 subsections with ≤ 1–3 paragraph heads each, ≤ 10 total) plus
a hybrid-style conversion of the Experiments section; the submitted paper's §3 uses
the hybrid structure. (2) The citation check for the algorithms paragraph: UCB1 →
Auer, Cesa-Bianchi & Fischer 2002; Thompson sampling → Thompson 1933; the LLM
optimizer needs no citation as her method but should nod to OPRO — all three landed
(both canonical citations appear in the submitted references, and the paper's §2
compares the strategies in exactly this three-part structure). The uniform-vs-adaptive
spec-sampling truth check recurs a third time (as a footnote to resolve); the
submitted paper states specs are sampled uniformly at random.

### 2026-08-23 — The writing-sprint conversation, chunk 7 (figure-agent instructions, intro, and the appendix's archaeological layer; 2026-02-16)

Four exchanges. (1) **Delegated-agent instruction writing:** her plotting agent ("my
floating language model") had produced Figure 1 conditioned on instructed budget = 50;
the fix was an explicit instruction block — logging schema, budget-vs-realized-cost
rules, per-spec-then-average order of operations, "critical checks," and the key
clause "Do not condition on the instructed budget field at all… Budget only affects
what the model tried; it is not the x-axis." An early instance of the
bound-the-delegated-agent's-task discipline, and the instructed-vs-realized confusion
appearing one level down, in the tooling. (2) Three placeholder-intro options
(page-limit constrained). (3) Cite-once-plus-at-definition rule for the bandit
citations. (4) **The appendix rewrite — with the submission's model roles stated
plainly: gpt-5-nano as both encoder and decoder, claude-haiku-4.5 as the
LLM-as-optimizer, gemini-2.5-flash-lite for model-transfer generalization.**

**The old appendix is preserved verbatim in the bundle**
(`lla-old-appendix-pre-rewrite.tex`) because its comments are TLC primary source: the
complete pitch-era prompt suite — DG-MINIMAL/SIGNATURE/STRICT,
**ANTI-CHEATING-CONSTRAINTS** (the anti-leakage clause set), ENC-BASIC/FAITHFUL/
STRUCTURED, and the full text of **COMP-NL and COMP-SHORT** (the two-stage compression
prompts this doc's earlier entries discuss abstractly — here is their exact text),
with the two-step pipeline explicitly attributed to Gilbert 2023 (semantic
compression, 2304.12512) as its source; plus the 8-category failure classification
(a finer-grained taxonomy than the paper's waterfall: adds latent-contains-code,
signature/output-type, restrictions-not-followed), the FizzBuzz contract-variation
figure caption, the B1/B2/B3 = 140/280/560-char budget tiers, the cross-model
reconstruction sketch, and the harness-fitting formalism paragraph (θ ∈ Θ covering
"prompts, templates, latent format, generation decomposition into stages, tool use,
memory, and sampling hyperparameters" — the pitch's H_θ, surviving here in the
appendix after being cut from the main text in chunk 5). Two rewrite notes from the
assistant: the "prices gathered on February 5, 2026" claim flagged as a liability, and
the recurring balanced-vs-uniform sampling truth check. **Correction (verified against
the submitted PDF's appendix, pages 7–11):** the pricing table did *not* survive into
the submission — the liability advice was taken; the Feb-5-dated Table 5 exists in the
*pitch*, not the submitted paper. Likewise the old formal-definitions appendix (H_θ)
did not survive: the submitted appendix is the chunk-7 rewrite with real values, so
the harness formalism exited the submission entirely — main text in chunk 5, appendix
here.

**Verified submitted-appendix facts worth recording:** (a) the enumerable 27-arm
space is Task × Goal × Constraint with three values each (token-named:
noop_task/summarize_code/generate_nl_sketch × noop_goal/as_detailed_as_possible/
detailed_for_llm_reconstruction × noop_constraint/omit_unnecessary_details/
high_level) — a 3-slot calibration subset of the full 6-slot clause space; (b) the
LLM-in-the-loop optimizer uses a **descriptor-token action protocol** — Haiku 4.5
must return only structured descriptor tokens (no JSON, no prose; unrecognized tokens
ignored; a spelling alias DESCRIBE_BEHAVIOR→DESCRIBE_BEHAVIOUR; unparseable turn ⇒
resample the previous prompt), conditioned on a 20-turn history with score history
and current rendered prompt — i.e., the optimizer's action space was itself
grammar-constrained, a small structured-output design worth remembering for
code-comp; (c) prompt selection for OOD used budget=200, **c_0 = 129 chars**, N=50 on
the 27-arm calibration space, and the per-method best-prompt table shows
coverage@c_0 of only 0.073/0.034/0.030 (Thompson/UCB1/LLM-Opt) — tiny absolute
coverage at the operating point, with Thompson's selection nominally best, another
face of the no-differentiation finding.

### 2026-08-23 — The writing-sprint conversation, chunk 8 (title, keywords, 7:25am; 2026-02-16/17)

Three short exchanges closing out the sprint's documented portion. (1) **The submitted
title is the assistant's recommended pick verbatim** — she asked for something better
than "Multi-Armed Bandits can Optimize Code Summarization Prompts" (note the old
title's summarization framing and its implicit bandits-win claim); from twelve options
the assistant bet on "Prompt Optimization for Behavioral Code Compression: Bandits vs
LLM-in-the-Loop Search," which is the submission's title exactly. Two unused options
worth remembering for the program's future vocabulary: "Training Without Weights" and
"Compression Without Gradients" — the framing-C banner phrases, shelved with framing
C. (2) OpenReview keywords. (3) Her intake note timestamps the moment: "I think this
is when I realized the deadline was really extended to 6pm not 6am, it was 7:25am, I
was very tired" — followed by the sanity check "does it make sense for 3 slots with 3
discrete phrasings each to have 27 arms?" (3³ = 27; consistent with the submitted
appendix's Table 6). The sprint's working conditions, in one exchange: verifying
single-digit exponentiation at 7:25am while discovering eleven unexpected extra hours.

### 2026-08-23 — The writing-sprint conversation, chunk 9 (the figures land, the noise question, the submission decision; ends 07:33 on 2026-02-16)

Danielle's timeline correction at intake: the deadline was originally 7am (not 6am as
she recalled at chunk 8), and this exchange ends at exactly 07:33 on 2026-02-16 —
i.e., 30+ minutes past the original deadline, portal still open, extension to 6pm just
discovered. She attached the then-current Figures 1 & 2: near-overlapping AUC-vs-N
curves (~0.03 → ~0.3 over N = 5→50, LLM optimizer slightly above from N ≈ 20,
overlapping bands). Her verdict: "basically the least interesting they could possibly
be..."

Three exchanges: (1) legibility triage — difference-plot insets Δ(c), Δ(N)
(the single best move), scalar-in-caption, y-zoom, caption metadata, multi-N
frontiers, and **the regime experiment ("this is the real story"): expand the arm
space at fixed N so bandits' exploration thins while the semantic prior degrades
gracefully.** (2) Her "but it is noise right, the band shows that its noise?" —
answered correctly: bands are uncertainty, not noise; overlapping CIs are a weak
eyeball test; the paired per-run difference curve with its CI is the fast decisive
check; conservative-vs-supported phrasings for either outcome. (3) Her submission
decision (verbatim): "I put them in the figure slots, get the generalization results
based on the data (running now) into the fig 3 slot, remove any claims of comparison
and submit, then iterate after?" — endorsed with the descriptive-not-comparative
wording list, the limitations sentence as a planned-extension hook, and the
still-answers-a-question checklist.

**(Claude-added:)** the submitted paper's final shape is this plan *plus the regime
experiment executed inside the unexpected extension*: the 8,064-arm large-space
regime (paper §3.3, Figure 3) is the chunk-9 "real story" advice made real — likely
run between 07:33 and the 6pm extended deadline — and "modest but consistent" is the
supported-phrasing branch of the noise answer. Also note the OOD results were still
*running* at submission-decision time ("data (running now) into the fig 3 slot"),
which is the concrete referent of the 2026-02-25 presentation's "differentiation in
generalization exps, but not enough time to thoroughly analyze." The paired
difference-curve analysis recommended here was never run — it remains the cheapest
outstanding item for any revival: the logs exist, and it would settle "modest but
consistent" vs. noise definitively.

### 2026-08-23 — The writing-sprint conversation, chunk 10 (the final polish loop; 2026-02-16, post-07:33)

Five exchanges over an intermediate draft that no longer exists (Danielle's note: she
attached a draft PDF she no longer has — the submitted PDF is the nearest surviving
state). The abstract and intro were softened per the remove-comparisons decision
(comparative clauses → "Across settings we observe a consistent cost–performance
tradeoff"; "using semantic priors" → "conditioned on the search history"); the
submitted abstract matches the softened version closely. The 5-page triage: figure
plan settled as Fig 1 = frontier + AUC panels, Fig 2 = generalization grid (her
placeholder was deliberate, for length estimation — a small miscommunication the
checklist misread as an error); the §2 collapse advice ("keep the encoder→IR→decoder
definition in 2–4 sentences; cut the feasibility waterfall to the appendix") — the
submitted §2 matches the tight replacement's content. The glaring-issues checklist:
leaked footnote instruction, Table ?? refs, "RELATED WORKS"→"RELATED WORK", typos —
and, notably, **item 8 flagged the "expanded-space settings" sentence as overreach
since only 27 arms existed at that point**, confirming the chunk-9 inference: the
8,064-arm regime did not exist at 07:33 and was added later in the extension day.
**(Claude-added:)** cross-checking the submitted PDF: the checklist items demonstrably
got fixed in the main text (no placeholders, no leaked footnote, uniform-sampling
sentence present), while the caption-level typo class survived ("intructed character
budgets," "neumeartion pass rates" in the submitted Figure 1 caption) — the last
edits under the least review. The final submitted figure numbering (Fig 1 two-panel,
Fig 2 OOD grid, Fig 3 large-space regime) is this chunk's plan plus the
afternoon's regime experiment appended as Figure 3.

### 2026-08-23 — The writing-sprint conversation, chunk 11 (09:09 on 2026-02-16 — the ownership pass)

The most important chunk of the writing conversation for the provenance record, and
for the project's history. At 09:09 — post-all-nighter, deadline crisis over — she
reset the terms (verbatim): "I decided to assume that the portal will remain open
until 6 and also to aim to make a version of this draft I feel good about, send it to
my advisor for his thoughts, but make it clear that I'm not convinced we should
submit and instead should probably aim for a different deadline with some more thought
given to different aspects of the exploration/design/framing. Then, I'm working
through the draft to write it in my own words based on what I actually want to say" —
with the assistant demoted to diction support ("my diction tanks when I'm exhausted").

What the ownership pass produced:

- **Her equivalence-class articulation, in her own words, at 9am:** "there are a few
  properties that define equivalence for me, then I can find a representation of the
  equivalence class. and the equivalence class is defined by behavior but we can't
  measure 'behavior' in the same way you can't take an expectation, so instead we
  have to estimate it like taking the empirical estimator of an expectation. and
  thats what we're using the test suite as." This is the property-indexed-equivalence
  framing from the 2026-02-06 conversation (chunk 11 there), resurfacing as her
  working vocabulary ten days later while writing — direct evidence the "aha"
  formalization had stuck and was hers.
- **The coining-ethics exchange:** her check "'behavioral code compression' isn't
  really a thing is it?" followed by the sharper "plenty of people use test passing…
  so it seems like this isn't really a term thats ours to coin?" — resolved as
  don't-coin: "code compression under a behavioral equivalence criterion," **which is
  the submitted abstract's opening phrase.** The same instinct as the chunk-6 style
  check and the attribution practice: verify what you're inheriting or claiming
  before adopting it.
- **Abstract provenance, final revision:** her "new-and-probably-not-improved-but-
  more-me abstract" plus the minimal fixes (criterion-not-constraint; "post hoc" cap;
  the parallel transfer list "across languages, across models, and on HumanEval++")
  matches the submitted abstract nearly line for line. This supersedes the chunk-5
  finding: the submitted abstract's last ancestor is the 9am ownership rewrite, not
  the chunk-5 polish — the abstract is more hers than the chunk-5 lineage suggested.
- Word decisions: "verifiable surrogate" chosen; her pretension check on "empirical
  estimator" ("pretentious and pedantic, or just specific?") answered honestly
  (accurate but heavier than the intro needs); "When I get clearer results the last
  sentence will become a claim instead of an observation" — the upgrade that never
  happened before the decision date.

**(Claude-added — open question:)** at 09:09 her stated position was
not-convinced-we-should-submit, aim for a different deadline; the paper was
nevertheless submitted to LLA that day. How the position changed between 09:09 and
submission is not recorded in this conversation.

### 2026-08-23 — The writing-sprint conversation, chunk 12 (the ownership pass reaches Methods; 2026-02-16) — and her retrospective verdict

**Her retrospective verdict (at intake, 2026-08-23 — resolving the chunk-11 open
question):** "I actually am really glad I did submit, because up until this point I
thought I was incapable of writing a workshop paper, and after submitting (and now,
looking back) I think I did an absolutely spectacular job considering the dramatic
time constraints and the complexity of the required infra + the confusion of trying
to move from one framing to another on no sleep... so I think future me would ideally
come into the deadline push more prepared but I think submitting isn't a bad choice
as long as you believe your results are true (if not good)."

**Scope correction (Danielle, at intake, correcting the intake agent's reading):**
the results did *not* exist before the push — "I only got the experimental setup
running sometime in that 42 hour period, so I got all the results during that
period." The two-night window therefore contained: standing up the experimental
infrastructure, collecting every result in the paper, the framing translation, and
the writing. (Her Feb-16 statement "I have a working implementation of everything and
a lot of samples collected," quoted in the writing-conversation chunk-2 entry, was
made from *inside* that window.)

**Chunk-12 content — the ownership pass extending into Methods, with traced
landings:** her restructure plan (collapse Tasks + Enc-Dec + Budget into one setup
paragraph; prompt search and algorithms next; keep the coverage curve compact; AUC
deferred to Experiments) is the submitted §2's structure. Her "why is code
compression potentially useful" instinct became the submitted intro's opening
(agent-assisted development / repeatedly conditioning models on large codebases —
the LLM-cost use case from the offered options, in edited form). The rate–distortion
question — "the whole cost/something tradeoff curve for compression has its own
standard language doesn't it?" — landed as the submitted discussion's
"rate–distortion-style tradeoff" (the recommended borrowing-without-overcommitting
phrasing), with the constrained form (min rate s.t. distortion ≤ ε; ε = 0 = pass all
tests) as the underlying objective statement. And the notation micro-negotiation —
her "T is of f, so making it T_i feels like its burying the lead," functions-vs-
programs ambiguity, p_i rejected for probability collision, f kept with a
parenthetical — produced the **submitted §2's opening Task sentence, which is her
sentence essentially verbatim** ({(f_i, T_{f_i})}; "a short intermediate
representation of f_i that still decodes into a function f'_i which passes the test
suite T_{f_i}"). Micro-conventions settled: LLM spelled out at first use; T_{f_i}
subscripting throughout the submitted paper.

**(Claude-added:)** with chunks 11–12 the provenance ledger flips for the paper's
front matter: abstract, intro opener, and the §2 problem statement are hers (9am+
ownership rewrites with copyediting), while the mid-paper machinery (metrics
prose, experiments skeleton) remains assistant-scaffolded with her edits, and Related
Works remains the closest-to-draft section. Her rate–distortion instinct here is also
the seed R1's review later asked to be grown ("is this formally analogous to a
rate–distortion curve?") — the borrowed-style phrasing was the honest deadline
choice, and the formalization remains the open follow-up.

### 2026-08-23 — The writing-sprint conversation, chunk 13 (the ownership pass deepens; 2026-02-16)

Fourteen exchanges; the richest chunk for tracing submitted-§2 phrases and for the
project's intellectual record:

- **"From my perspective we are training/learning, thats why I think the whole 'llm
  as optim' is cool. but I realize we haven't really presented data to back this up
  yet"** — the framing-C conviction restated mid-rewrite, with the claim-calibration
  awareness attached. Resolution: "outer-loop learning problem," which is the
  submitted §2's phrase verbatim. Likewise **"interaction trace (previous prompts and
  observed outcomes)"** — her deliberate choice ("'trace' ties it to RL/Sequential
  decision making, which is my goal"), landing verbatim in the submitted §2. The RL
  vocabulary in the paper is intentional positioning, not accident.
- **Her combination-novelty stance, stated as taste:** choosing the positioning
  sentence that "makes it clear we're not claiming to be first for any of these
  things individually, but does unify a collection of different topics which is more
  some of the novelty" — landing (shortened) as the submitted intro's "We relate our
  setup to prior work on prompt optimization, test-driven synthesis/repair, and code
  representation learning in §5."
- **Precision instincts, each landing:** "overreachy" check on "produce the shortest"
  → "seek"; black-box must bind to the LLMs, not the pipeline → the submitted
  "pipeline, built from black-box LLMs"; the parallel-list grammar objection. One
  counter-datapoint: the suggested "Background & Methods" → "Methods" rename was
  *not* taken — the submitted paper keeps "Background & Methods." Advice was
  filtered, not swallowed.
- **The ownership moment, in February's own words:** "sigh, I'm actually feeling more
  positively towards this paper, probably because I feel more ownership again, the
  words were yours but alot of the thoughts really were mine. but I also don't want
  to want to push for a 6pm deadline. sigh" — the February original of her intake
  retrospective, plus the not-embarrassed submission checklist and stake-in-the-
  ground framing that preceded the submit decision.
- **Her results diagnosis (verbatim):** "I can't have two plots that look basically
  identical (despite supposedly plotting different things) that show that everything
  I tried (1) behaved the same as each other and (2) did exactly what you'd expect
  them to do. but the idea of digging into the code and trying to figure out how I
  can salvage that is the daunting piece. the writing is fun." The salvage menu
  (difference plots; early-N frontiers; heterogeneity/tail plots; the expanded-arm
  regime) — of which **B and D were executed before submission** (the appendix's
  multi-T frontier panels, Figure 4; and the large-space regime at 8,064 arms —
  beyond the suggested 243), while **A (difference plots with CIs) and C
  (per-spec heterogeneity/tails) were never run** and join the paired-difference
  analysis from chunk 9 as cheap outstanding items: the min-cost-to-first-success
  distribution per spec (option C) is also a natural bridge to the behavioral-
  divergence instrument's tail-risk metrics.

### 2026-08-23 — The writing-sprint conversation, chunk 14 (the final exchanges; 2026-02-16, 12:54 and 14:32)

Three exchanges (Danielle at intake: "these were the last two"), all executing salvage
option B from chunk 13 — the early-N' frontiers that became the submitted appendix's
Figure 4 multi-T panels (T = 5/10/25/50/90):

- **Early-N' frontiers defined:** recompute S_m(c;N') using only the first N'
  evaluations — a *prefix of the interaction trace* — for N' < N, overlaid on the
  same coverage-vs-cost-cap axes. The point: methods that converge to similar
  N=50 performance can differ in how quickly they get there; this makes the
  AUC-vs-evals story visible on the frontier axes themselves. Requires no
  re-running — only re-aggregation from logs, which is what made it executable
  inside the deadline extension.
- **The prefix-not-resample clarification (her 2:32pm question, the conversation's
  last):** no resampling needed — "early-N'" means "what the method had discovered
  after N' attempts," not "what it would discover if rerun with budget N'." Filter
  by attempt index t < N'. **The bias warning:** filtering by cost instead changes
  the definition from "after N' evaluations" to "among low-cost attempts" and
  biases the curve — cost belongs only on the x-axis inside the frontier
  computation. Worth keeping as a standing analysis rule for any TLC-0 frontier
  reimplementation.
- **Bucketing guidance:** S_m(c;N) is exactly a step function at observed unique
  costs; plotted smoothness is visualization only. Rule of thumb: bucket width
  giving ~30–60 x-points across the plotted range (step 10–20 chars for the
  ~0–450-char range of the submitted Figure 1a). Don't bucket-then-average to
  manufacture step-ness; compute at a fine grid and downsample the plot, not the
  metric.
- **Budget fairness, disentangled:** two budgets exist. The evaluation budget N
  (attempt count) **must** be matched across methods — the key fairness
  constraint. The instructed IR budget B (the "keep under X chars" instruction)
  may be ignored on the frontier axes provided its distribution is not
  systematically different across methods: either hold B fixed, use an identical
  schedule, or make B part of a shared arm space (then "arms include budget
  phrasing" and fairness comes from identical arm access). This is the
  clean statement of B-as-conditioning-variable that the pitch's budget-tier
  design (140/280/560) gestured at.

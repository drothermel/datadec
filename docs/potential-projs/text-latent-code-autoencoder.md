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


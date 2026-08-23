# Evaluation methodology and paradigm comparison — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: the research hypothesis (`../../research-hypothesis.md`) rests on the
claim that pipeline-stage comparisons are confounded by unequal tuning history and by
uncontrolled elicitation; these are the precedents for that claim and for the
existence-proof alternative.

---

## 2026-08-18 — precedents (from the Research Trajectory page)

- Melis, Dyer & Blunsom 2018, *On the State of the Art of Evaluation in Neural Language
  Models* — "showed LSTM-vs-transformer conclusions inverted under equalized tuning
  budgets." The pre-2021 ancestor of the "headline phenomenon is a tuning artifact"
  finding-shape.
- Sara Hooker, *The Hardware Lottery* — "research directions win not on merit but because
  the surrounding stack co-evolved with them, and a decade of co-adaptation can't be
  replayed for the challenger inside any single experiment's budget." Danielle's version
  points at the software stack (init schemes, optimizer settings, warmup/decay conventions,
  curricula, eval formats).
- *Position: Lifetime Tuning Is Incompatible with Continual Reinforcement Learning* — "the
  same complaint from the RL side."
- Existence proofs as paradigm evidence: AlexNet (Krizhevsky et al.), GPT-3 (*Language Models
  are Few-Shot Learners*), DeepSeek-R1-zero — "existence proofs, not controlled
  comparisons." Within the arc: shrink-and-perturb; continual backprop ("Sutton's own gloss
  was that continual backprop at least shows the problem can be solved").
- Failure mode of the existence-proof genre: the 2024–2025 RLVR literature, corrected by
  *A Sober Look…* (seeds) and *Spurious Rewards* (elicitation in disguise) — see
  `pretraining-to-posttraining.md`.

---

## 2026-08-18 — Signal and Noise (from the Research Trajectory page)

- Heineman et al., *Signal and Noise: A Framework for Reducing Uncertainty in Language Model
  Evaluation* (NeurIPS 2025 per the discussion) — signal = a benchmark's ability to
  separate better from worse models; noise = sensitivity to random variability between
  training steps; interventions: continuous (perplexity-type) metrics beat accuracy on
  both; filtering noisy subtasks improves aggregate reliability. Release: ~900K evaluation
  results on 465 open-weight models including OLMo intermediate checkpoints, DataDecide,
  and the model-ladder runs. The trajectory drift/diffusion project is its dual.

---

## 2026-08-18 — where eval variance actually lives (from the Research Trajectory page)

For OLMES-style loglikelihood evaluation, re-evaluating a fixed checkpoint with new seeds
buys nothing: inference nondeterminism is negligible, generation-based evals are the
exception, and few-shot configuration variance is "a bias axis to sweep, not noise to
average." The variance of interest is in training (seed, data order, init) — consistent
with Signal and Noise's definition of noise as step-to-step wander. Cited: OLMES (*A
Standard for Language Model Evaluations*).

## 2026-08-23 — Danielle's model-divergence hypothesis and behavioral-distance designs (early-2026 conversation, historical)

From chunk 3 of the ChatGPT re-eval conversation (~Jan–Mar 2026; transcript at
`~/drotherm/data/convo-artifacts/2026/2026-08-23-prompt-opt-reeval-aha/`; project
context in the TLC doc §4). Respondent claims unverified.

**The hypothesis (Danielle, verbatim):** "I've heard frequently from researchers that
all the model provider's models have collapsed to almost identical solutions.  My
results on simple fxn generation seems to suggest almost the opposite and makes me
wonder whether for the specialized task of coding older models were actually *more*
similar because their gains came mainly from general LLM improvements but now we're
making great progess in the coding realm because of the ability to verify outputs for
rewards or dataset generation, etc so different model families may actually have gotten
*further* from each other in the coding task space over time."

Note this is grounded in her own data: the early-2026 TLC baseline runs on simple
function generation showed large cross-model behavioral differences, against the
folk-wisdom "collapse" claim.

**Behavioral-distance measurements** (respondent's menu; all run on fixed task sets at
fixed decode settings):

- A. Agreement/correlation on per-item success across models; cluster models
  (dendrogram/heatmap). Collapse predicts high correlation, tight clusters.
- B. Error-mode distance: label failures (not-code / parse / type / wrong-algorithm /
  off-by-one / timeout / contract violation), compare distributions across models (JS
  or Wasserstein divergence). Equal pass@1 with different failure signatures is the
  strong anti-collapse evidence.
- C. Output diversity under controlled sampling: N samples per prompt per model;
  unique AST shapes / CFG-ish fingerprints, edit-distance distributions, solution
  families after normalization (formatting, alpha-renaming).
- D. Prompt transfer as behavioral probe: optimize on A, evaluate on B; collapse
  predicts high transfer, small gaps.
- E. Time axis (her hypothesis directly): 2–3 generations per family, repeat A–D, test
  whether pairwise distances shrink or grow over time — OpenRouter's historical model
  sequences as the asset. Respondent's prior: convergence on easy items, divergence on
  hard/edge-case items.

**Workshop-sized slices** (respondent, on request): H1 behavioral clustering (200–500
tasks × 6–10 models, pass/fail correlations); H2 failure-mode divergence ("equal mean,
different tails" — same runs plus failure taxonomy; recommended single pick, least
sensitive to baseline accuracy levels); H3 transfer matrix (3 models, optimize-on ×
eval-on deltas, transfer-gap metric). Concrete starter: ~300 generated tasks, 6 models
(3 families × old/new), log pass/fail + failure taxonomy + normalized solution
fingerprint; plot success-correlation matrix, failure-distribution JS divergence,
per-model solution-family entropy.

Status: a freestanding project candidate (uses the TLC synthetic library as testbed but
is not TLC); promotion to staging/project doc is a pending intake decision.

## 2026-08-23 — Decision quality under constraints as a divergence probe (chunk 5 of the early-2026 conversation)

Continuation of the divergence-hypothesis entry above; same historical conversation and
provenance caveats.

**Danielle's idea (near-verbatim):** tasks "with clear constraints that make one of the
options clearly optimal given your constraint, but where an agent might not choose that
option even if they technically solve the task" — e.g., performance-critical sorting of
a huge float list, choose among three implementation sketches (bubble / merge / radix)
and complete the chosen one. Verify implementation correctness with tests; verify
choice optimality "probably directly via ast, but definitely via timing." Generation
strategy: work backwards from the curated "true statements" of software engineering
(interview-prep books and sites) → small set of clearly ranked choice options →
templated task scenarios + verification methods.

**Prior-art answer (respondent, unverified):** no widely adopted benchmark has
choose-among-sketches-then-implement as its core unit, but efficiency-beyond-
correctness is an active crowded area: ENAMEL (eff@k, expert references, stress test
generators), EvalPerf/DPE (profiling against reference solutions at distinct efficiency
tiers), EffiBench (NeurIPS D&B, efficiency vs. human canonical solutions on
LeetCode-style tasks), and DS-1000 (2211.11501 — execution-tested data-science code
across pandas/numpy/matplotlib/sklearn/scipy/torch with surface-form constraints; the
direct precedent for the chunk-4 data-manipulation family, complemented by a generator's
controllability and anti-memorization). Danielle's pushback that this is "a pretty hard
pitch" for a full paper accepted as correct for a generic efficiency benchmark.

**The defensible framing (respondent's recommendation):** decision quality under
constraints *as a probe for model divergence and reliability* — merging this thread
with Danielle's divergence hypothesis. Unit of evaluation: three separable skills via a
two-stage protocol (structured pick-and-justify, then implement): (A) choice quality,
(B) implementation correctness, (C) constraint adherence (profiling/static checks).
Unique hook a generator enables: constraint-varying families where the *right choice
changes with the constraint* (memory cap → in-place wins; stability required → stable
sort; latency-critical → vectorization), testing whether models track constraints.
Candidate full-paper claim set: similar-on-correctness ≠ similar-on-decision-quality;
decision quality more variance-prone and family-sensitive; prompt optimization improves
correctness more than decision quality (or vice versa by regime); "good coders, bad
planners" as a persistent per-model profile.

**De-risking (Danielle flagged high risk of unconcludable outcomes; respondent's
answer):** named failure modes — ceiling, floor, spec ambiguity, timing noise, and the
choice-vs-implementation-difficulty confound. Mitigations: the A/B/C separation itself;
low-risk families (asymptotic slam-dunks verified by scaling regime or hard timeout
rather than microbenchmarks; API-constraint compliance via AST/import checks — pandas
vectorization, pydantic validators, torch-not-numpy; memory caps via OS limits);
ceiling knobs (trap options, conflicting constraints requiring prioritization, 3→5
options), floor knobs (fill-in-missing-lines scaffolds, partial starters), timing knobs
(coarse scaling tests, timeouts, median-of-5, pinned threads). Hypotheses designed to
be interpretable under nulls: H1 choice separates families more than correctness (null
= "constraint-following is commoditized"); H2 optimization improves implementation more
than choice (either direction tells a story); H3 relative seed-stability of choice vs.
implementation ("planning stable, execution noisy" or the reverse). Pilot kill-test:
30 tasks (10 asymptotic / 10 API-constraint / 10 vectorization-vs-loop) × 6 models × 5
seeds.

**Intake note (Claude-added):** the two-stage structured pick is exposed to
option-position and label bias (models preferentially pick option A / the first or last
listed); the respondent never mentions randomizing option order and identity across
samples, which this design needs from day one.

**Chunk-6 addendum — the bare-MCQ variant:** Danielle, surprised no standard eval is
just the multiple-choice asymptotic pick ("input size is huge; O(n²) vs O(n log n) vs
O(n) — which do you choose?") as a probe of "the agent's ability to explicitly
retrieve and understanding about core foundational programming concepts." Respondent:
MCQ complexity questions exist inside general-knowledge benchmarks (MMLU CS topics) but
never became a standard *coding* eval because bare MCQ is gameable as recall and the
field shifted to execution-based evals; the nearest existing benchmark is
**BigO(Bench)** (Facebook Research — complexity prediction from code plus
complexity-controlled generation, validated by profiling/curve-fitting; no arXiv ID
given). What would make the MCQ version scientifically useful rather than trivia:
scenario constraints that make the answer spec-dependent (memory vs time caps,
stability, input-distribution assumptions like nearly-sorted or small integer range);
a *commitment test* pairing the pick with implementation or scaffold-completion
(separating knows-the-concept from can-execute-it — the same A/B separation as above);
and generated anti-memorization variants rotating surface form over fixed underlying
principles (sorting under assumptions, kth-element selection, streaming distinct
counts, join strategies). Offered v0: ~50 items across 5 "truth families," each with
MCQ + optional scaffolded implementation + deterministic verifier.

**Chunk-7 addendum — mining BigO(Bench) into the choose-then-implement format, and the
analysis-first paper shape.** Two developments (same historical conversation;
respondent claims unverified):

- *Practical note:* BigO(Bench)'s last author is one of Danielle's friends and
  favorite previous collaborators — a natural contact if the mining direction ever
  becomes real. Danielle explicitly not interested in building the bare-MCQ version.
- *Mining designs.* Per the respondent, BigO(Bench) ships ~1.19M solutions labeled
  with time/space complexity (inferred by its dynamic profiling/regression framework)
  plus runtime coefficients. Option A (low hassle): for each problem, sample 3 real
  solutions from different complexity classes, mask 10–30% into sketches (loop body /
  update / helper; skeleton intact — completion, not synthesis, to reduce the
  choice-vs-implementation-difficulty confound), prompt choose-and-fill under stated
  constraints ("n up to 10^6, <1s"; "256MB cap"; "streaming, single pass"). Verify:
  problem's own tests (correctness), option's complexity label (choice), and re-run
  the complexity-inference framework on the completed code to catch silent algorithm
  mutation (drifting back to the slow strategy — itself a measurable phenomenon:
  choice→implementation consistency). Option B (more novel): cluster solutions by
  AST/control-flow fingerprints into *strategy families*, one representative sketch
  per cluster. Label-noise caveat: dynamic inference is noisy at small n and for
  large-constant implementations — start with large-margin class separations (O(n²)
  vs O(n log n) vs O(n)), optionally require clear profiling-coefficient separation.
- *Danielle's framing principle (near-verbatim):* "I would prefer a setting where the
  paper relies on some other key thing, and the release of a benchmark is a
  byproduct… I've done this interesting, very carefully designed analysis on this
  specific set of questions, and I created this benchmark as a way to do that. And
  then I introduced a, like, not super novel, but semi-novel approach that targeted
  one of the biggest failure modes that I discovered in my analysis." Respondent's
  compression: benchmark = instrument; paper = discovery + explanation + targeted
  fix. Seven-step shape: question → instrument → non-obvious analysis → dominant
  failure mode → small principled intervention (example: a constraint-conformance
  repair loop — static/dynamic checker flags forbidden AST patterns or scaling
  violations, model patches while preserving the chosen strategy) → show the fix
  works across models → artifact release as byproduct. Four publishable result
  shapes: families differ on decision quality (divergence evidence); similar choice
  but different implementation reliability (planner-vs-implementer); divergence only
  under mixed tradeoffs; intervention reduces violations even when raw correctness
  is flat. **Candidate standing principle for `project-approach-principles.md`
  (pending Danielle's decision at the walkthrough):** instrument → analysis →
  targeted fix, benchmark as byproduct.
- *Intake note (Claude-added):* mining real competitive-programming solutions
  reintroduces the contamination/memorization risk the synthetic generator was
  praised for avoiding — masked completion of solutions models may have seen verbatim
  partially measures recall. Surface-form rewriting (renaming, restructuring-
  preserving-strategy) or regenerating implementations against the labels would be
  needed; the respondent does not raise this here despite raising anti-memorization
  earlier in the same conversation.

**Chunk-8 addendum — constraint consistency as a fourth measurement axis, and the
workshop-fitting exercise.** Danielle's practice (near-verbatim): "taking a loosely
plausible workshop subject and then trying to figure out how I could fit my ideas… by
changing the framing as a way to get better at thinking about the first versions of
the ideas from different perspectives." Applied here to the ICLR 2026 LLM logical-
reasoning workshop (sites.google.com/view/iclr-2026-llmreasoning — historical; that
workshop has passed). The respondent's translation of her portfolio into
logic-workshop language surfaced one genuinely new measurable: **constraint
consistency across logically related prompt variants** — same spec rephrased, spec
plus an added constraint, spec as choose-among-sketches — with contradiction defined
as violating an implied or stated commitment ("claims it chose O(n log n) but
implements O(n²)"; "says stability required but uses unstable approach"). Framings:
spec = premises, program = proof witness; a unit-test harness / complexity profiler /
type checker / pydantic validator functions as the CFP's "external logical solver."
Sketch submission: 30–80 tasks × 3–5 related variants; feasibility, success, and
consistency rate; intervention = verifier loop where the model proposes solution +
structured commitments, the verifier returns a failing constraint or counterexample,
and the model patches under a don't-violate-prior-commitments rule.

This gives the accumulating instrument four distinct axes worth naming: variance
(same prompt, different seeds), consistency (related variants, systematic sensitivity),
divergence (across models/families/vintages), and decision quality (within-task choice
under constraints). (Claude-added synthesis.)

On the meta-question — workshops as testbed for adjacent areas and for practicing
half-baked-idea-to-tiny-paper conversion — respondent's cautions: the two-stage
tiny-then-expanded process is not a universal ICLR-workshop feature (verify per CFP);
review quality varies; dual-submission rules vary; and the "workshop farm" risk of
fragmenting the thesis narrative. Crisp rule offered, candidate standing principle for
`project-approach-principles.md` (pending walkthrough decision, alongside the chunk-7
benchmark-as-byproduct candidate): **every workshop submission should either become a
module of the eventual big paper or decisively kill a direction quickly** — plus the
1-home-workshop + 1-stretch-workshop pattern, each tiny paper = one clean hypothesis,
one killer figure, one minimal intervention.

**Chunk-9 note:** the consistency axis extends to code-side perturbations — Danielle's
canonicalization thread (recorded in the TLC doc §4, chunk-9 entry) supplies
semantics-preserving transformation operators as the generator of "logically related
variants" on the code side (invariance to program symmetries), complementing the
prompt-side rephrasings above; canonicalization is then the natural intervention arm.

**Chunks 16–17 addendum (session of 2026-02-08) — the divergence hypothesis becomes a
paper spec.** Two days after the main conversation, Danielle returned to package the
existing TLC baseline data as a variance-analysis workshop paper (ICLR 2026 workshops
with deadlines ≥ Feb 10). Workshop scan (respondent, web-backed, historical): CAO
drift-monitoring workshop judged the cleanest fit but deadline too tight (Feb 11);
Logical Reasoning (Feb 20 / Mar 20 two-round) already earmarked by Danielle for a
*separate planned submission* — "contradictions within (model, spec) pairs if we use a
structured latent"; **Re-Align Challenge track (Feb 26) chosen** — behavioral
distributions treated as representations, alignment/divergence measured across
providers. Three storylines offered; Danielle picked C, her divergence hypothesis:
"cheap models aren't collapsed — they're diverse in behavioral geometry."

Evidence requirements settled for the claim: operationalize collapse as (small)
per-spec outcome correlation, failure-mode distribution distance, and within-spec
variance; behavioral signature vector s_m over waterfall categories with JS/TV/
Wasserstein distances; explicit tail-risk metric P(catastrophic) = P(not-code ∪
compile ∪ runtime); robustness across ≥2 prompt regimes and ≥2 task families. Her
addition (the chunk-16 question, confirmed load-bearing): **two kinds of collapse** —
outcome collapse vs. behavioral/solution collapse — with her existing AST code-metrics
analysis (control-flow conditions, complexity, naming/line/total lengths, comments,
docstrings) testing the second, **conditioned on success** ("even when they succeed,
they differ" — divergence in the equivalence-class interior). Feature split for
credibility: structural/strategy features headline (control-flow shape,
algorithm-family proxies, library usage, complexity, mutation patterns, exception
handling); stylistic features supporting only. Final deliverable: a complete
implementation-ready spec — "Behavioral non-collapse in cheap coding LLMs" — with
thesis, H1 (tail/failure-mode non-collapse), H2 (success-conditioned strategy
diversity), H3 (persistence across specs/regimes; fallback: collapse under strict
contracts but divergence under underspecification), four figures (mean-vs-tail
scatter; failure-signature heatmap+clustering; success-conditioned behavioral
fingerprint heatmap; regime stability), per-sample logging schema, and paper outline —
handed off at the time to a separate implementation-planning agent (verbatim spec in
the transcript, chunk 17). **Outcome of the Re-Align submission and the Logical
Reasoning plan not recorded in this conversation — open intake question.**

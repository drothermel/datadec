# MoE analysis program — router as taxonomy, reroute-vs-rewrite, interventional data typing

**Status:** topic (staging). Candidate exits: one or more new `potential-projs/` docs (sweep
reanalysis; reroute-vs-rewrite over training; interventional MoE data typing), or a program
framing that wraps the other MoE topics
([moe-routing-as-data-instrument.md](moe-routing-as-data-instrument.md),
[moe-recipe-suite.md](moe-recipe-suite.md)). Overlaps with both; dedup when deciding.

**Question posed (Danielle, 2026-08-21).** What would it look like to ask similar styles of
questions around MoEs — richer outcome channels, routing decisions to categorize tokens,
released MoE models and datasets as additional "free" compute artifacts to combine with or
compare to the dense options? Plus the Slicing-and-Dicing codebase (Danielle third author,
https://arxiv.org/abs/2605.11689) with hyperparameter guidance at small sizes, and a stated
preference for interesting analysis over grid sweeps. See
[../danielle-inputs.md](../danielle-inputs.md).

Related-work claims below are unverified (see [README.md](README.md)).

---

## 2026-08-21 — Response

"Your own headline finding ('most config choices don't affect final quality') is exactly the
kind of result that begs the follow-up you're describing."

### The core conceptual gift: MoEs make the featurization bidirectional and nearly free

"In the dense program, 'functional featurization' required you to *construct* the instrument:
branch, perturb the mixture, measure a response vector assembled from KL, barriers, per-token
losses. The MoE gives you two things the dense model doesn't:"

*A learned, categorical token taxonomy at every forward pass.* "The router is a data
featurizer that the training process built for you. Every token gets a discrete type
assignment (top-k expert set, per layer), plus a confidence (logit margin), and this taxonomy
was discovered by gradient descent under the actual training objective rather than imposed by
you or borrowed from a reference model. Your 'identify the types of the chunks' question
becomes, in part, 'read the router.' The immediate scientific questions are whether this
taxonomy is *real* — stable across seeds, across your sweep's configurations, across training
time — and whether it aligns with the featurizations from the dense program (entropy buckets,
domains, determinism). If routing clusters recover the determinism profile without ever
seeing a reference model, that's a striking convergence result; if they cut across it, you've
found a second axis of data typology that the dense program couldn't see."

*A factorized locus of change.* "This is, I think, the deepest structural difference. In a
dense model, 'what changed between checkpoints' is a formless question — you attack it with
KL, CKA, drift/diffusion. In an MoE, change decomposes *architecturally* into two channels:
**rerouting** (same experts, different assignments — the function changed because tokens go
elsewhere) and **rewriting** (same assignments, different experts — the function changed in
place). You can compute this decomposition exactly: hold routing fixed at checkpoint t while
using checkpoint t+1's experts, and vice versa, and attribute the output delta. That's the
MoE dual of your drift/diffusion decomposition, and it's *causal by construction* rather than
inferred from time-series statistics. Nobody has run this decomposition over training on an
open suite. My guess at the phenomenology — early training is rerouting-dominated, late
training rewriting-dominated, with the crossover per layer being a 'commitment clock' — but
the point is that any answer is a paper-shaped answer, in the same either-way-safe genre as
your IRT dimensionality question."

### How the branch-probe design gets better in MoEs

"Take the ε-mixture-perturbation design [see functional-featurization.md] and re-run it
mentally on an MoE. The response vector gains channels that are qualitatively sharper than
anything dense:"

- **Expert-level update attribution.** "When you continue training on type-X-enriched data,
  the sparsity localizes the gradient: only the experts that type-X tokens route to get
  meaningful updates. Data attribution — which in dense models requires influence-function
  machinery — partially falls out of the architecture. 'Which parameters does this data move'
  has a first-order answer you can read off the routing table, and the interesting
  measurement is the *residual*: how much of type-X's effect lands outside its own experts
  (via the shared expert, attention, and router updates). That residual is a direct measure
  of how modular the model's knowledge actually is."
- **Reroute-vs-rewrite response.** "For each (data type × stage) cell, does the treatment
  mostly reassign tokens or mostly rewrite experts in place? A crisp hypothesis: early-stage
  treatments reroute, late-stage treatments rewrite, and the stage at which a data type can
  no longer trigger rerouting is a *plasticity boundary* — measured categorically, not
  through effective-rank proxies."
- **Modular plasticity.** "The continual-learning angle gets an MoE-specific twist: plasticity
  may be a per-expert resource rather than a global one. Does new-distribution data get
  absorbed by colonizing under-utilized experts (cheap, non-interfering) or by overwriting
  committed ones (expensive, interfering)? This connects to a practical lever the field
  actually uses — expert resetting/addition for domain adaptation — but nobody has measured
  the underlying absorption dynamics during ordinary pretraining. A 'which data types
  colonize vs. overwrite, at which stages' map is both a science result and a midtraining
  design guide."

"And the river-valley reading is natural: router saturation is basin commitment made visible,
persistent routing flips are river movement in a categorical channel, reverting flips are
wall oscillation ([the routing follow-up's TRJ-moe-1], which slots directly in here), and the
[TOK-obs-5] hypothesis — high-entropy tokens keep flipping after low-entropy tokens' routes
freeze — is the MoE twin of the dense program's single most interesting figure. The two would
make each other far more credible if they land together."

### Where Danielle's specific position is an unfair advantage

*The released artifacts.* "OLMoE (fully open, intermediate checkpoints, routing analyzable),
FLAME-MoE, and the closed-weights-open-final models (Mixtral, DeepSeek-style) give
observational breadth. But observational MoE routing analysis is a moderately crowded area —
expert-specialization papers exist for most released models. The free artifacts are the
*validation set*, not the contribution."

*The codebase + hpm guidance.* "This is what converts you from routing-analysis paper #12 into
something else: you can run **interventional** MoE experiments at will — seeds, decay
branches, mixture perturbations, frozen-router controls — at scales where you already know
the hyperparameters are right. The single most valuable control your infra enables, which
released models can never give you: *train with data as the treatment and everything else
identical*, or *freeze the router mid-run and continue training* to cleanly separate the
rerouting and rewriting channels causally rather than by post-hoc decomposition. Nobody doing
observational analysis on Mixtral can run that experiment."

*The sweep itself, reread as data.* "This is the one I'd push hardest, because it turns your
existing NeurIPS submission into the first act of the analysis paper you actually want to
write. You have on the order of 2,000 trained MoEs spanning expert count, granularity,
shared-expert size, and load-balancing — many reaching *similar final loss* by your own
findings. That is a matched-loss comparison across *architectures*, the exact structural move
as DataDecide's matched-loss comparison across *recipes*. The question it enables: **is the
token taxonomy a property of the data or of the architecture?** Concretely: match tokens'
expert assignments across configs (expert-matching across models is the MoE version of Git
Re-Basin, and granularity changes make it a soft/hierarchical matching problem — does a
64-expert model's partition refine a 16-expert model's, like a nested clustering?). If the
discovered partition is largely invariant across your sweep, you've shown the data imposes
its own decomposition and the architecture merely allocates resolution to it — which would
*explain* your paper's finding that config choices barely matter: they're all approximating
the same underlying factorization at different granularities. That's the 'more interesting
analysis' sequel in its purest form: the grid sweep found the invariance in outcomes; the
follow-up finds the mechanism behind the invariance. And if the partitions *aren't*
invariant — same loss, genuinely different factorizations — that's a
same-basin/identifiability result with teeth: quality-equivalent models that are internally
non-equivalent, and then the follow-on question is whether the factorization differences
predict *anything* downstream (finetuning behavior, continual-learning interference,
robustness) that final loss doesn't."

### The two big caveats, which the sweep uniquely defuses

"First: routing is known to be partly shallow. Analyses of released MoEs repeatedly find
assignments that correlate with token identity and frequency more than semantics, especially
in early layers. So 'router as taxonomy' must survive the trivial-explanation gauntlet —
regress assignments on token ID, frequency band, and position first, and make the taxonomy
claim only about the residual structure. Interestingly, your dense program's
reference-entropy scorer is exactly the right covariate set for this."

"Second, and more fundamental: **the load-balancing objective distorts the measurement
instrument.** The auxiliary loss actively pushes routing toward uniformity, so observed
assignments confound 'where the data wants to go' with 'where the balancer forced it to go.'
Any routing-as-featurization paper that ignores this is measuring the artifact. Here your
sweep is again the asset rather than the bystander — you varied load-balancing mechanisms and
found they barely affect quality. The analysis-level follow-up: do they affect the
*taxonomy*? If the discovered partition is stable across balancing mechanisms (and across
dropless vs. dropping), routing-as-measurement is validated in a way no single-model analysis
can achieve; if the taxonomy shifts while quality doesn't, that's itself a finding about how
underdetermined the factorization is. Either way, you're the only group positioned to run the
check, because you're the only one holding a balancing-mechanism sweep at matched
everything-else."

### Suggested sequence

1. "The **sweep-reanalysis paper** (taxonomy invariance across configs, seeds, and balancing
   mechanisms, with the shallow-routing controls) — it's T0-flavored, reuses artifacts you
   own, directly extends your submission's story, and its methods (cross-model expert
   matching) are the tooling everything else needs."
2. "The **reroute-vs-rewrite decomposition over training** on OLMoE/FLAME plus a few of your
   own runs with dense checkpointing — the MoE dual of drift/diffusion, with frozen-router
   branches as the causal control your infra makes cheap."
3. "The **interventional data-typing study**: ε-mixture perturbation branches at 2–3 stages
   with the expert-level response vector — this is the MoE version of the
   functional-featurization program and the most novel, but it should inherit a frozen
   branch-schedule policy and response-vector spec from the first two."

**Contrast with the dense DataDecide line.** "That program's moat is careful measurement on
someone else's public artifact; this program's moat is that *you own the training stack and
the sweep*. The former is faster to a first paper; the latter compounds — every intervention
you can run that observational groups can't widens the gap. Given that your stated ambition
is moving from grid sweeps to analysis with a mechanism story, the MoE line is the better
long-term home, and the dense T0 papers (IRT, drift/diffusion, the data card) are best
treated as the fast, low-risk publications that build the shared instruments — noise floors,
entropy scorer, matched-loss methodology — that the MoE program then imports."

---

## 2026-08-21 — Artifact status

Final checkpoints for the entire Slicing-and-Dicing sweep exist and will be uploaded to
Hugging Face soon; the collaborator's upcoming experiments may provide intermediate
checkpoints/logs for some runs. Details in
[../open-questions-answered.md](../open-questions-answered.md). This unblocks the
sweep-reanalysis step (final-checkpoint analyses only; over-training analyses need the new
runs).

---

## 2026-08-21 — Tiny-MoE floor study (from the small-scale discussion)

"Where [the sweep's] laws break as active scale shrinks": per-expert data starvation
(budget/E tokens), a router too small to learn a useful partition, routing shallowness
worsening as capacity drops. "Does a 5M-active, 500M-total MoE beat a 5M dense model, and does
its routing learn anything non-trivial" — "the taxonomy-realness question acquires a scale
axis, and the failure mode of routing at tiny scale is informative about what routing is
doing at normal scale." Credible because of the sweep's validated small-scale
hyperparameters. Full text in
[small-scale-measurement-science.md](small-scale-measurement-science.md).

# MoE partitions — is the token taxonomy a property of the data or the architecture?

> **Draft scaffolding (2026-08-21).** This doc was promoted from a topic. The quoted material in
> §4 is external text; the core steps, doability notes, impact ratings, and infrastructure
> sequence in §1–§3 are synthesized scaffolding not yet reviewed by Danielle. Treat them as
> provisional until this note is removed.

**One-line pitch.** The router of a trained MoE is a learned, categorical token taxonomy.
The Slicing-and-Dicing sweep (~2,000 MoEs varying expert count, granularity, shared experts,
and load-balancing, many at similar final loss) is a matched-loss comparison across
*architectures*. Match expert assignments across configs and ask whether the discovered
partition is invariant (the data imposes its factorization; architecture sets resolution —
which would explain the sweep's own finding that config choices barely matter) or not
(quality-equivalent models that are internally non-equivalent).

IDs: PART-1–PART-6, PART-opt-1–PART-opt-4.

**Artifacts.** All final checkpoints of the sweep exist and are being uploaded to Hugging
Face; no intermediate checkpoints from the original sweep (see
`docs/open-questions-answered.md`). A working MoE pretraining repo with validated
small-scale hyperparameters exists.

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short
continued-training branches; **T3** = new pretraining runs.

---

## 1. What the project involves

### Core experiment

1. **Checkpoint access (T0 setup).** Load any sweep checkpoint from the repo's model code;
   record each run's config (expert count, granularity, shared-expert size, balancing
   mechanism, seed) and final loss.
2. **Routing logs over a fixed probe corpus (T1).** One forward pass per checkpoint over a
   fixed, versioned probe corpus; log per-token, per-layer top-k expert ids and router
   logit margins.
3. **Shallow-routing controls (PART-3).** Regress assignments on token ID, frequency band,
   and position first; make every taxonomy claim about the residual structure only. The
   reference-model entropy scorer (per-token conditional entropy) is the covariate set.
4. **Cross-model expert matching (PART-4).** Match experts across configs — the MoE version
   of Git Re-Basin; across granularities it is a soft/hierarchical matching problem (does a
   64-expert model's partition refine a 16-expert model's, like a nested clustering?).
5. **Invariance tests (PART-5).** Partition agreement across expert count, granularity,
   seeds, and load-balancing mechanism (and dropless vs. dropping), at matched final loss.
   The balancing-mechanism arm is the validation that routing-as-measurement is not a
   load-balancing artifact.
6. **Alignment with dense featurizations (PART-6).** Does the partition recover entropy
   buckets, domains, frequency bands, and the determinism profile? Convergence is a result;
   cutting across them is a second axis of data typology.

### Optional directions

- **PART-opt-1: Cross-suite.** OLMoE vs. FLAME-MoE vs. OpenMoE (checkpoints and known data):
  does expert-specialization structure track corpus composition across independent setups?
- **PART-opt-2: Do factorization differences predict anything downstream** (fine-tuning
  behaviour, continual-learning interference, robustness) that final loss does not?
- **PART-opt-3: The two "why" analyses for the sweep paper itself.** (a) Decompose the
  total-parameter gain at extreme sparsity by eval type (memorization-heavy vs.
  reasoning-heavy tasks, or per-token by frequency band). (b) Does routing
  entropy/specialization stay constant at the optimal expert size across total-parameter
  counts? Rebuttal-sized if checkpoints and eval infra exist.
- **PART-opt-4: Routing as a data fingerprint.** Read a featurization of the corpus *out of*
  the router: which intrinsic features does the expert decomposition recover, and do
  deviations from token-ID routing (context-dependent routing, late reassignments) mark the
  high-entropy tokens?

---

## 2. Doability and impact

### Overall doability: **medium-high**; no training, one methods unknown

- Artifacts confirmed; forward passes over ~2,000 checkpoints on a modest probe corpus are
  the main compute.
- The genuine methods work is soft/hierarchical expert matching across granularities; raw
  per-config routing statistics are useful before matching exists.
- Caveat 1: routing is partly shallow (token identity/frequency) — hence PART-3 first.
- Caveat 2: the load-balancing objective distorts observed assignments; the sweep's
  balancing-mechanism variation is the only known way to test this at matched
  everything-else.
- Both outcomes of PART-5 are strong; no pivot needed.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| Core (PART-3–PART-6) | **High** | Mechanistically explains the sweep's headline; introduces a reusable cross-MoE comparison method; either outcome is a claim about MoEs generally. |
| PART-opt-1 cross-suite | Medium-high | Validation set for the taxonomy-realness claim. |
| PART-opt-2 downstream | High if positive | Turns an identifiability result into a practical one. |
| PART-opt-3 "why" analyses | Medium (rebuttal value) | Days of work if infra exists; explains the sweep's own findings. |
| PART-opt-4 fingerprint | Medium-high | "Read the router" as a data-measurement instrument; crowded observationally, uncrowded when joined to measured corpus features. |

---

## 3. Infrastructure build sequence

1. **Sweep checkpoint loader** from the Slicing-and-Dicing repo; config and final-loss
   table.
2. **Probe corpus** (fixed, versioned, manifest; stratified by domain and leaf corpus).
3. **Routing-log runner**: per-token, per-layer top-k ids and margins per checkpoint, stored
   as long tables keyed by (run, layer, token position).
4. **Reference-model scorer** for per-token entropy; token-ID/frequency/position covariates.
5. **Expert-matching module** (hard matching first; soft/hierarchical second).
6. **Invariance and alignment analyses** (PART-5, PART-6); figures.
7. *(Optional)* OLMoE/FLAME-MoE/OpenMoE ingest for PART-opt-1.


---

## 4. External assessments

Dated, attributed-by-date notes from external review conversations, recorded for
consolidation — not decisions. Only notes about this project are kept here. Related-work
claims in quoted text are unverified.

### 2026-08-21 — origin: the sweep reread as data, and the routing-as-taxonomy argument

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

### 2026-08-21 — on MoE releases as artifacts, and routing as a data fingerprint

**What MoE releases actually give you.** "Your featurization program needs two things:
**treatment variation** (many datasets/recipes) and **outcomes** (what the models learned).
DataDecide's gift was the treatment axis — 25 recipes, everything else controlled. MoE
releases give you almost none of that: FLAME-MoE is seven models from 38M to 1.7B active
parameters — a *scale* ladder, one data recipe. OLMoE is one recipe (a mix building on Dolma
and DataComp-Baseline, released with open data, code, logs, and intermediate checkpoints).
OpenMoE is one recipe. And the big 2025–26 MoE wave — Llama 4, DeepSeek V4, Qwen 3.6, Kimi
K2.6, gpt-oss, Command A+ — is open-*weights*, closed-data: no treatment axis and often no
checkpoints. So no public multi-recipe MoE suite exists. The recipe question — the core of
your thesis — cannot be asked with these artifacts."

"What they *do* give you is a much richer **outcome channel**. Dense models give you loss and
eval scores; MoE models additionally give you per-token, per-layer categorical routing
decisions and emergent expert specialization. That's a learned, discrete decomposition of the
data, produced for free as a training byproduct."

**The reframing: routing as a data fingerprint.** "The OpenMoE analysis found that routing
decisions are predominantly based on token IDs with minimal context relevance, and
token-to-expert assignments are determined early in pretraining and remain largely fixed. If
routing is mostly a token-identity/frequency phenomenon locked in early, then **expert
assignment is essentially a data fingerprint** — a clustering of the corpus that the model
committed to based on early data statistics. That inverts the direction of your program in a
productive way: instead of 'featurize the data, predict the model,' you get 'read the
model's routing, recover a featurization of the data.' Concretely:"

- "Does the expert decomposition recover your intrinsic features (domain composition,
  frequency bands, determinism profile)? If routing ≈ token-ID clustering, it should correlate
  strongly with cheap corpus statistics — and *deviations* from that (context-dependent
  routing, late reassignments) mark exactly the tokens your entropy-bucket hypothesis cares
  about."
- "Routing-commitment timing as a per-token version of your migration analysis, observable
  from existing checkpoints with *no decay branches needed* — the categorical channel makes
  'committed vs. still moving' directly legible instead of inferred from KL. There's already
  adjacent work treating this dynamically — e.g. a recent paper tracking a three-phase
  load-balance trajectory (surge, stabilization, relaxation) across OLMoE and OpenMoE
  checkpoints — but it's aggregate-level; the per-token, data-linked version is open."
- "Cross-suite: OLMoE (largely DCLM-based) vs. FLAME-MoE vs. OpenMoE all have checkpoints and
  known data. Three points isn't a recipe axis, but it's enough to ask whether
  expert-specialization structure tracks corpus composition across independent training
  setups."

**Where the framing is wrong.** "If the motivation is 'add MoE to get more free outcome data
for the featurization regression,' I'd say no — it adds architecture as a confound without
adding recipe variation, and n stays tiny. One model per data point, with architecture,
tokenizer, and scale all varying between releases, is a worse supervised problem than
DataDecide, not a better one."

"If the motivation is 'MoE routing is a new *instrument* for reading data structure out of
models' — yes, and I think that's genuinely underexplored. It slots into your program as a
measurement-layer contribution (like the drift/diffusion decomposition or IRT θ), not as an
outcome-table expansion. The honest version of the pitch: dense models force you to infer
data-driven structure from continuous proxies (KL, CKA, per-token loss); MoE models write
part of that structure down explicitly, and nobody has systematically joined routing
statistics to measured corpus features."

**Ordering implication.** "This strengthens the case for building your reference-model scorer
and corpus-feature extractors first (they're what routing gets joined *to*), and it makes
[the routing follow-up] less of an orphan — it becomes 'the routing instrument' chapter of
the same data-measurement program rather than a separate suite with a separate story. The
ingest-uncertainty gate still applies, though: whether FLAME-MoE's released routing logs
support per-token tracking across checkpoints determines whether this is a T0 join or a T1
recomputation, and that survey should still be step one."

### 2026-08-21 — the two "why" analyses for the sweep paper

**What the paper found (as summarized in the response).** "2,000+ runs up to 6.6B total,
exhaustively varying expert count, granularity, heterogeneous sizing, shared experts, load
balancing, with the findings being that total params always help even at 128× ratios, optimal
expert size depends only on active params, and most other knobs are second-order. The review
trajectory you describe ('weak accept/weak reject, lacking interesting analysis') is the
classic empirical-sweep review: solid, exhaustive, descriptive — reviewers want a 'why.'"

**Two "why" analyses sitting on the existing grid.**

- *Why does total capacity keep helping at extreme sparsity?* "The standing hypothesis in the
  literature (the 'mixture of parrots' line) is that expert capacity buys memorization more
  than reasoning. Your grid is the ideal testbed: decompose the total-param gain by eval type
  (tail-knowledge/memorization-heavy vs. reasoning-heavy tasks, or per-token by frequency
  band on held-out data). If the 128×-ratio gains concentrate on memorization-flavored items,
  you've explained your own headline finding. If the checkpoints and eval infra exist, this
  is potentially days of work, not months — plausibly rebuttal material."
- *Why does optimal expert size depend only on active params?* "That's a striking invariance
  with no stated mechanism. Routing analysis across the grid is the natural probe: does
  routing entropy/specialization structure stay constant at the optimum across total-param
  counts? Does granularity trade off against per-expert specialization in a way that's
  visible in assignment statistics? This is exactly the routing-as-observable channel
  [see PART-opt-4 and §4 below], applied to architecture variation instead of
  data variation."

### 2026-08-21 — positions in ranked lists (full lists in `docs/portfolio-rankings.md`)

- Workshop-sized list **#6**; full-conference list **#4, "One Partition, Many Architectures"** (expected impact high; ceiling high); sub A of **P4** in the four-project list (scoop risk low — "the sweep is the moat"). Quoted entries:

- Sweep reanalysis: **workshop-sized #6** ("Either outcome is strong… mid-list only because
  of practical unknowns: whether the sweep saved checkpoints… If checkpoints exist, this
  could move up two slots" — they do) and **full-conference #4, "One Partition, Many
  Architectures"** ("the soft/hierarchical matching method needs genuine development and
  validation — that's the iteration cost. Both outcomes strong; no pivot needed. **Expected
  impact: high**… **Ceiling: high**"). Protected from scooping by the sweep.

**P4 — What MoE Configurations Actually Change: Partitions and Movement.** Sub A: taxonomy
invariance across the sweep. Sub B: reroute-vs-rewrite over training. Main paper: "a unified
account of what varies and what's invariant across quality-matched MoEs — the partition, its
resolution-refinement across granularities, where training movement lives, and the
frozen-router/thaw causal arm." "**Speed: fourth**… **Scoop risk: low** — the sweep is the
moat… **Expected impact: high. Ceiling: high**, with an option on very-high if the causal arm
shows suppressed routing adaptivity costs quality." The recommended day-one action — confirm the sweep's checkpoints survive — is done.

### 2026-08-22 — functional task identity across architectures (citation supplied by Danielle)

- Theseus (Rinaldi et al., ICML 2026; arXiv 2602.12952) transports task updates "across
  heterogeneous-width models" by characterizing "a task update by the functional effect it
  induces on intermediate representations" and solving "a functional matching problem on
  observed activations… after aligning representation spaces via orthogonal Procrustes
  analysis." The same move — define identity functionally on activations, not
  parametrically — is what PART-4's expert matching across expert counts and granularities
  needs; Procrustes alignment of expert output spaces is a concrete starting point. See
  `docs/topics/task-vectors.md`.

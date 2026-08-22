# Functional featurization — data types defined by training response

> **Draft scaffolding (2026-08-21).** This doc was promoted from a topic. The quoted material in
> §4 is external text; the core steps, doability notes, impact ratings, and infrastructure
> sequence in §1–§3 are synthesized scaffolding not yet reviewed by Danielle. Treat them as
> provisional until this note is removed.

**One-line pitch.** Replace *intrinsic* featurization (properties of the text) with
*functional* featurization (properties of the text's effect on a model, conditioned on where
the model is in training). Perturb the data stream from an intermediate checkpoint, log a
response vector, and type chunks by how they move the model at each stage. The output is a
component × stage × measurement tensor — the measured version of every lab's midtraining
stage table — and the generalizing claim is which measurable corpus properties predict a
component's response curve.

IDs: FUNC-1–FUNC-5, FUNC-opt-1–FUNC-opt-5.

**Dependencies.** A decay-branch runner, the held-out token set with per-token logging, the
reference-model scorer, and a frozen branch schedule policy. The response-vector logging
spec must be frozen *before* any branch runs by any project, because annealing branches can
double as the first cells of this tensor only if they log the right things.

**Confound to check first.** If DataDecide's realized per-window source mixture drifts from
nominal, every run has an implicit curriculum that confounds stage-dependent claims (open
gate in `docs/open-questions-answered.md`).

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short
continued-training branches; **T3** = new pretraining runs.

---

## 1. What the project involves

### Core experiment (T2)

1. **A-priori type taxonomy for the pilot (FUNC-1).** Code, math, high-determinism prose,
   high-entropy dialogue, instruction-formatted, synthetic reasoning traces; chunked
   candidate pool, stratified by leaf corpus; functional features (entropy profile,
   instruction-likeness, reasoning density) attached to every chunk. Include the pair most
   likely to separate (e.g. high-determinism code vs. high-entropy dialogue) at an early vs.
   a late-stable stage to learn the effect size first.
2. **Local mixture-perturbation branches (FUNC-2).** From checkpoints at 2–3 (later 4–6)
   stages: replace an ε fraction of the ongoing stream with type X, vary ε, estimate the
   derivative of the trajectory with respect to composition at that operating point (the
   "data Jacobian"); the ε=0 continuation is the built-in baseline. Pure-type far-field
   branches are the large-dose anchor of the dose-response curve, not the primary
   measurement.
3. **Durable-movement filter (FUNC-3).** Every treatment branch gets a schedule-neutralizing
   readout at its endpoint (short decay or checkpoint merge) to separate durable from
   transient response; the annealing project is the control arm (schedule as treatment,
   data fixed).
4. **Response vector (FUNC-4).** *Frozen minimum, logged at every branch endpoint:* per-token
   loss on the held-out set, and saved endpoint weights. *Readouts computed later from
   those:* Δ per-token loss by entropy bucket and domain; Δθ from an IRT fit; weight-space
   direction of the branch projected onto the run's river direction vs. orthogonal;
   interpolation barrier to the untreated continuation at matched tokens; representation
   effective rank. *Requires an extra run:* learning speed on a fixed probe task fine-tuned
   from the endpoint.
5. **Surrogate ladder (FUNC-5).** Single-step gradient alignment of type-X batches against
   held-out losses (cheapest) → short ε-branches (middle) → full branches (ground truth);
   validate each tier against the next on a subset, then scale the cheapest validated tier.
   Calibrate branch length against readout SNR early.

### Optional directions

- **FUNC-opt-1: Discovered taxonomy.** Cluster chunks by response vector; the rank of the
  response tensor (chunk × stage × measurement) is the headline question. Long-game
  motivation rather than a first-paper pitch at DataDecide scale.
- **FUNC-opt-2: Plasticity cost by data type.** Which types spend plasticity and which
  preserve or restore it, by stage.
- **FUNC-opt-3: Component × timing map (U_c(t)).** Inject one component per branch at 4–6
  points along a base recipe; per-token response, task-family deltas, durable-vs-transient
  split; the measured stage table.
- **FUNC-opt-4: MoE variant.** Expert-level update attribution (how much of type-X's effect
  lands outside its own experts), reroute-vs-rewrite response per (type × stage), modular
  plasticity (colonize under-used experts vs. overwrite committed ones).
- **FUNC-opt-5: Intrinsic features → response profile.** The featurization regression with
  n = chunks rather than n = 25 recipes.

---

## 2. Doability and impact

### Overall doability: **medium-low** — second-act by construction

- Consumes the branch runner, logging harness, held-out set, reference scorer, and ideally
  the annealing results that fix the branch schedule policy.
- Cost: 1/16 of run length per cell caps the taxonomy at ~6–10 types × 4–6 stages at
  DataDecide scale; shorter branches are plausible with per-token readouts; the surrogate
  ladder is what makes the full tensor affordable (core, not mitigation).
- Biggest scientific risk: at 150M–1B, stage × type interactions may be small relative to
  the generic shift transient and seed noise. The local ε design protects; a null under it
  is a real but less exciting result. Fully powered versions belong at 10–50M with many
  seeds.
- Curriculum-learning prior among reviewers: the defense is that this measures
  stage-dependent data value with causal probes, agnostic about whether any curriculum
  beats random.
- Literature positioning should be stated narrowly: recipe × stage × response profile on an
  open suite.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| Core (FUNC-1–5) | **High if effects exist; ceiling the highest in the portfolio** | "How many functional kinds of data are there, and when does each act" reframes data curation. |
| FUNC-5 surrogate ladder alone | Medium-high | "When do gradient surrogates predict short-horizon training response, as a function of stage" stands alone as a methods paper. |
| FUNC-opt-1 discovered taxonomy | Very high (conditional) | Any rank answer is interesting; underpowered at small grids. |
| FUNC-opt-2 plasticity cost | High | Cited by pretraining-data and continual-learning communities. |
| FUNC-opt-3 U_c(t) map | High | First public map of its kind even at 5 components × 4 timings. |
| FUNC-opt-4 MoE variant | High | Sparsity localizes attribution; the residual measures modularity. |
| FUNC-opt-5 features → response | Medium-high | Fixes n=25 by moving supervision to chunk level. |

---

## 3. Infrastructure build sequence

1. **Held-out token set.** Choose a held-out token set once and freeze it: fixed, versioned token
   sequences with a manifest; stratified across domains and across the DataDecide leaf
   corpora; sized so that each domain × entropy-bucket cell has enough tokens to estimate
   mean per-token loss drop within a set tolerance, while keeping one forward pass per
   checkpoint cheap. Per-token loss on it is a standard output of the eval harness for every
   checkpoint variant (raw checkpoints, merged checkpoints, branch starts and endpoints),
   stored as compact arrays keyed by (checkpoint variant, held-out-set version). Branch
   endpoints also save their weights. Cheap to add now; expensive to retrofit later because
   it would mean re-running branches. *An identical spec appears in Annealed readouts, WSD retrain suite, Token-level
   movement, MoE movement, MoE recipe suite, and Functional featurization; keep them in sync.*
2. **Results store + eval harness.** Load any checkpoint; run the eval suite and perplexity evals; store results
   keyed by (run, step) plus a `variant` field (`raw`, `merged:<cfg>`, `branch:<cfg>`), in the
   same table schema as the processed OLMES tables so results slot into existing accessors.
3. **Decay-branch / continuation runner** with a mixture-perturbation option (replace an ε
   fraction of the stream with a chunk pool), configurable decay at the endpoint, per-token
   logging, endpoint weights saved.
4. **Chunk pool + taxonomy tooling**: chunking, leaf-corpus stratification, functional
   features (reference-model entropy profile, instruction-likeness, reasoning density).
5. **Gradient-alignment surrogate** (single-step influence of a batch on held-out losses).
6. **Response-vector readouts** from saved weights: bucketed Δloss, IRT Δθ, river-direction
   projection, barrier to untreated continuation, effective rank; probe-task fine-tune
   harness.
7. **Pilot** (two types × two stages × seeds) → calibrate branch length and surrogate tiers
   → scale.


---

## 4. External assessments

Dated, attributed-by-date notes from external review conversations, recorded for
consolidation — not decisions. Only notes about this project are kept here. Related-work
claims in quoted text are unverified.

### 2026-08-21 — origin: response to "is that a thing?" (the origin of this topic)

**Question posed (Danielle).** Not high-level descriptors like "quality"; instead chunk the
datasets, identify different types of contributions, look at how they are distributed, and
how they move metrics or the loss landscape when applied at different stages of training. See
[../danielle-inputs.md](../danielle-inputs.md).

"Yes, it's a thing — but only in pieces. Nobody has assembled the version you're describing,
and the assembled version has a name-able shape: **the value of a data component as a
function of training time**, call it U_c(t)."

**Decomposition into contribution types.** "This exists but is mostly coarse and
provenance-based. The standard axis is source/domain labels (what DoReMi and RegMix reweight).
More interesting recent work builds *learned* taxonomies — WebOrganizer-style topic×format
classification of web data, embedding-cluster decompositions used for mixture optimization,
Skill-It's skill graphs. And there are *functional* decompositions closer to your instinct:
instruction-like text hiding in web crawl, reasoning-dense passages, and your own
determinism/entropy profile is exactly a functional decomposition at token granularity. So
the 'chunk and identify types of contribution' step has tooling, but almost everyone stops at
domains because that's what's labeled."

**Component → metric attribution.** "Also exists in two flavors: mixture-level (RegMix, data
mixing laws — regress outcomes on mixture weights, coarse but scalable) and example-level
(influence functions, datamodels — precise but brutally expensive at pretrain scale). The
middle granularity you're describing — cluster/component-level causal contribution — is
comparatively empty, and it's the tractable one: hundreds of components instead of billions
of examples."

**The stage/timing axis.** "This is where the field is most active *and* most anecdotal.
Multi-stage pretraining is now standard practice — early pretraining on diverse noisy
corpora, then shifting the distribution toward more challenging, higher-quality data as the
learning rate anneals, which the mid-training literature explicitly frames as
curriculum-style scheduling. Production models publish their stage tables — HyperCLOVA X, for
example, reports code going from 12% to 25% and math from 8.6% to 25.3% across four stages,
with instruction data appearing only late. And there's emerging work on the schedule×data
interaction directly — one recent paper's title is literally about how learning rate decay
wastes your best data, in the context of the now-standard practice of shifting to
high-quality mixtures at mid-training. There's even evidence for effects invisible at the
endpoint: a recent study of pretraining data ordering found curriculum effects that don't
show up in final performance alone — ordering shapes when and which linguistic capabilities
emerge and reorganizes embedding-space structure."

**The gap.** "Every existing timing result has one of three limitations. Either the component
axis is crude ('high quality data' — exactly the ambiguous descriptor you're rejecting), or
the timing axis is crude (two stages: before and after midtrain), or the outcome is crude
(final benchmark score, no dynamics, no landscape). The practice you see in stage tables like
HyperCLOVA's is real engineering knowledge, but it's *folklore encoded as percentages* —
nobody has published the underlying map: for a measured decomposition of data into
components, what is each component's marginal effect as a function of *when* it's
introduced, on *which* metrics, and on the geometry of where the model ends up?"

**Why the existing machinery is well-positioned.** "The decay-branch runner is the missing
instrument. A branch at step t with component c injected (or ablated) at some mixing weight is
a causal probe of U_c(t) — it's a factorial component × injection-time experiment where each
cell costs ~10% of a training run, not a full run. Your docs already flagged this and
deferred it ([WSD-opt-4], 'MiniCPM-style mixed-in decay data — scope creep risk'). Under this
framing it's not scope creep; it's the point. And the per-token logging plus reference-model
scoring gives you the mechanism-level outcome: which *tokens* each component moves, whether
components act on the wall (transient, schedule-dependent) or the river (durable), and — via
the interpolation/barrier tooling from [landscape geometry] — whether late-injected
components land the model somewhere geometrically different than early-injected ones. That
last question, component timing → landscape position, is as far as I can tell completely
unoccupied."

**Concrete study sketch.** "Take one base recipe; decompose a candidate pool into ~10–20
components by a learned taxonomy crossed with functional features (entropy profile,
instruction-likeness, reasoning density) rather than source labels; run branches at 4–6
points along training, each injecting one component; measure per-token response, task-family
deltas, and durable-vs-transient split. Output is a component × timing × outcome tensor — the
measured version of every lab's stage table. Even a small version (5 components × 4 timings)
would be the first public map of its kind, and it composes with everything else: the
DataDecide featurization becomes 'which measurable properties of a component predict its
U_c(t) curve,' which is the generalizing claim that turns a map into a theory."

**Caution.** "Curriculum learning proper has a long history of weak-at-scale results, and
reviewers carry that prior. The defense is that you're not proposing a curriculum — you're
*measuring* stage-dependent data value with causal probes, which is upstream of (and agnostic
about) whether any curriculum beats random. The mid-training practice of the entire field is
the proof that stage-dependence exists; what's missing is the measurement, and that's the
paper."

---

### 2026-08-21 — response to the combined prompt (beyond DataDecide + chunk types + 1/16-run branches)

The combined prompt is in [../danielle-inputs.md](../danielle-inputs.md). The three
entries above are its step-by-step precursors.

**Where it sits relative to existing literature.** "The nearest neighbors are data attribution
(influence functions, datamodels/TRAK) and mixture optimization (DoReMi, RegMix). But both
differ in ways that leave your space open. Attribution methods are retrospective — 'which
training points caused this behavior in the final model' — and mostly stage-blind. Mixture
methods are prospective but operate at coarse source granularity with a single scalar objective
(loss or a downstream metric). What you're describing is prospective, stage-conditioned, and
with a *rich* response vector: not 'does this data lower loss' but 'which tokens' losses move,
in which direction in weight space, with what effect on geometry and plasticity.' Skill-It
touches stage-ordering but with predefined skills. The stage × type interaction map, measured
with landscape-aware readouts, is genuinely unclaimed territory. The critical-period and
plasticity-loss literatures (Achille et al.; Dohare/Lyle-style loss of plasticity) predict that
this map should be strongly non-uniform in the stage axis — and nobody has drawn it on an open
suite."

**The design, sharpened.** "Your 1/16-run branch is what I'd call the *far-field* probe:
continue on pure (or heavily enriched) type-X data and measure the response. It's the right
pilot, but it has a known confound — any distribution shift produces a generic transient (loss
spike, wall excursion) that will dominate short branches regardless of content. Three
refinements make it a real instrument:"

- *The local version.* "Instead of pure-type continuation, perturb the mixture: replace an ε
  fraction of the ongoing stream with type X, vary ε, and estimate the derivative of the
  trajectory with respect to composition at the current operating point. That's a 'data
  Jacobian' of training — the response to type X *from here* — and it's far better behaved
  than the far-field branch because the baseline (ε=0, continue on original mixture) is built
  in. Pure-type branches then become the large-dose anchor for the dose-response curve rather
  than the primary measurement."
- *The durable-movement filter.* "Your own river-valley framing predicts that a lot of
  short-branch response is wall excursion that a decay would erase. So every treatment branch
  needs the schedule-neutralizing readout you already designed for the annealing project — a
  short decay or checkpoint-merge at the branch endpoint — to separate durable from transient
  response. This is a genuine synergy, not just infrastructure reuse: the annealing project is
  literally the control arm of this design (same branch machinery, schedule as the treatment,
  data held fixed; here data is the treatment and the schedule policy is fixed). The (data ×
  schedule) factorial is the full experiment, and your existing work determines the schedule
  policy for the branches."
- *The response vector.* "This is where the idea either becomes a measurement science or stays
  a vibe. Concretely, per (type, stage, seed) cell I'd log: Δ per-token loss on the fixed
  held-out set, bucketed by reference-model entropy and domain; Δθ from the IRT fit rather
  than raw accuracy (lower noise, and item-level DIF of the *treatment* is a lovely readout);
  the weight-space direction of the branch — its projection onto the run's river direction
  (pre-cooldown→final) vs. orthogonal components; the interpolation barrier between the
  treated endpoint and the untreated continuation at matched tokens (did this data kick you
  toward a different basin, or move you along the same one?); and one or two plasticity
  measures — representation effective rank, and learning speed on a fixed probe task
  fine-tuned from the endpoint."

**Plasticity cost by data type.** "That last one deserves emphasis because I think it's the
most under-asked question in your whole space: **which data types spend plasticity, and which
preserve or restore it?** Continual-learning work establishes that plasticity is a depletable
resource; nobody has attributed its depletion to data types during pretraining, or asked
whether the annealing-phase 'high-quality data' convention works partly because certain types
are plasticity-cheap late. A stage × type map of plasticity cost would be cited by both the
pretraining-data and continual-learning communities."

**The discovered-taxonomy version.** "There are two routes, and the second is the more novel
one. Route one: a priori types (code, math, high-determinism prose, high-entropy dialogue,
instruction-formatted, synthetic reasoning traces) — necessary for the pilot, interpretable,
but inherits human categories. Route two: *define* type by response — cluster chunks by their
response vectors, so two chunks are the same type iff they move models the same way at the
same stage. The full object is a response tensor (chunk × stage × measurement), and the
scientific question is its rank: if it's low-rank, there are only k functional data types, and
the factors are the answer to 'what does data actually vary in, functionally.' That's the
datamodels idea lifted to chunk granularity and stage-conditioning, and it's a headline-shaped
question because *any* answer is interesting — k=2 ('river data and wall data exist and that's
it') would be a strong claim; high rank with stage-dependent structure would be the opposite
strong claim. Note the pleasing duality with your IRT project: there you featurize models by
their response to items; here you featurize data by models' response to it. Same bilinear
skeleton."

**Cost, and the surrogate ladder.** "1/16 of run length per cell is expensive once you multiply
types × stages × seeds — at DataDecide scale it's feasible but it caps the taxonomy at maybe
6–10 types × 4–6 stages. Two mitigations. First, branches can likely be much shorter than 1/16
if the readout is per-token loss deltas (high sensitivity) rather than benchmark movement
(noisy); calibrate branch length against readout SNR early, the way your branch-length sweep
calibrates decay length. Second — and this is the methods contribution hiding inside the
project — build a surrogate ladder: single-step gradient alignment (LESS-style influence of
type-X batches on the held-out losses) as the cheapest tier, short mixture-perturbation
branches as the middle tier, full branches as ground truth. Validate each tier against the
next on a subset, then scale the cheapest validated tier over the full taxonomy. 'When do
gradient surrogates predict short-horizon training response, as a function of stage' is a
publishable question by itself, and it's exactly the kind of result that makes the expensive
tensor tractable for anyone else."

**Relation to the existing portfolio.** "This direction subsumes rather than competes.
[Token-level movement Stage 2] (per-token decay responsiveness) is this design with schedule
as the only treatment; [Stage 1's] entropy bucketing supplies the readout coordinates;
[landscape geometry's] barriers become one component of the response vector rather than a
standalone paper; the featurizing project's intrinsic statistics get recast as *predictors* of
functional type — the interesting regression is no longer features → final benchmark outcomes
(n=25, underpowered) but features → response profile (n = however many chunks you probe,
hundreds or thousands). That quietly fixes your n=25 problem: the supervised structure moves
from recipe-level to chunk-level, where you control n."

**Sequencing.** "This is a second-act program, not a first paper. It needs the branch runner,
the eval/logging harness, the held-out token set, the reference scorer, and ideally the
annealing results to fix the branch schedule policy — which is precisely the infrastructure
your T0 papers build and de-risk. What I'd change about the current plan in light of it:
freeze the response-vector spec *now*, before any branch runs, because the annealing branches
can double as the first cells of this tensor for free if they log the right things —
per-token losses, endpoint weights for barrier/direction analysis, and the probe-task
fine-tune. Retrofitting any of those means re-running branches, which is the one mistake your
documents already warn about in three places."

**Biggest scientific risk.** "At 150M–1B, stage × type interactions may be small relative to
the generic shift transient and seed noise, and the far-field/local distinction is what
protects you — if the ε-derivative design still shows nothing, that null is itself a real
result ('short-horizon data response is type-blind at small scale'), but it's a less exciting
one. I'd want the pilot to deliberately include the pair most likely to separate (e.g.,
high-determinism code vs. high-entropy dialogue, at an early vs. late-stable stage) so you
learn the effect size before committing to the full tensor."

### 2026-08-21 — pushback recorded in the planning session

- The response vector as listed is six instruments with six validation burdens. The only
  irreversible logging decision is **saving endpoint weights** (plus per-token losses on the
  shared held-out set); every readout except probe-task fine-tune speed can be recomputed post
  hoc from saved weights. Freeze that minimum, treat the rest as later readouts.
- The response-tensor rank headline is over-promised at DataDecide scale (6–10 types × 4–6
  stages × 3 seeds, plus the shift transient); keep it as long-game motivation, not a
  first-paper pitch.
- Landscape geometry's standalone question (do recipe effects hold only within low-barrier
  pairs?) runs on existing checkpoints and should not be absorbed here.
- Literature positioning should be stated narrowly: stage-conditioned data influence has
  precedent in online/curriculum selection and the critical-period work; the unclaimed part
  is specifically recipe × stage × response-profile on an open suite.
- The surrogate ladder is core, not a mitigation — it is the only way the full tensor is
  affordable.

---

### 2026-08-21 — the MoE version of the branch-probe design


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

### 2026-08-21 — positions in ranked lists (full lists in `docs/portfolio-rankings.md`)

**Deliberately cut** from the workshop-sized list ("remain your strongest *eventual* papers
but are second-act by construction"). **Full-conference #10, "The Functional Types of
Pretraining Data"**: "*Speed:* slowest by design — it consumes the frozen branch policy
from [annealed readouts], the response-vector spec, and the powered tiny-scale substrate… its
central risk… has only a modest pivot (the surrogate-validation study stands alone as a
methods paper). **Expected impact: high** if the effects exist. **Ceiling: the highest on
the list**… it's the paper the previous nine were quietly building toward."

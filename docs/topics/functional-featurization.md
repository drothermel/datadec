# Functional featurization — data types defined by training response

**Status:** topic (staging). Not yet a project doc. Candidate exits: its own project, or the
second act of token-level movement / annealed readouts.

**Idea in one line.** Replace *intrinsic* featurization (properties of the text) with
*functional* featurization (properties of the text's effect on a model, conditioned on where
the model is in training). A chunk's type is defined by its response profile; quality stops
being a descriptor and becomes an operator.

---

## 2026-08-21 — Response to "is that a thing?" (the origin of this topic)

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

## 2026-08-21 — Response to the combined prompt (beyond DataDecide + chunk types + 1/16-run branches)

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

### Same date — pushback recorded in the planning session

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

## 2026-08-21 — Confound to check first

If DataDecide's realized per-window source mixture drifts from nominal (small sampling
fractions, no per-window stratification), "every run has an implicit, unintended curriculum"
— which "confounds every timing/curriculum claim built on these suites including your own
U_c(t) program." Open gate in [../open-questions-answered.md](../open-questions-answered.md).

---

## 2026-08-21 — Position in the ranked lists (full lists in `../portfolio-rankings.md`)

**Deliberately cut** from the workshop-sized list ("remain your strongest *eventual* papers
but are second-act by construction"). **Full-conference #10, "The Functional Types of
Pretraining Data"**: "*Speed:* slowest by design — it consumes the frozen branch policy
from [annealed readouts], the response-vector spec, and the powered tiny-scale substrate… its
central risk… has only a modest pivot (the surrogate-validation study stands alone as a
methods paper). **Expected impact: high** if the effects exist. **Ceiling: the highest on
the list**… it's the paper the previous nine were quietly building toward."

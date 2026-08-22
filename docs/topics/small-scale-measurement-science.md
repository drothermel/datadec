# Small-scale measurement science — rigorous LM-training experiments at ≤150M

**Status:** topic (staging). Candidate exits: a lab-level program framing ("measurement
science of language-model training at academic scale") that wraps the existing projects; and
concrete project candidates — a per-window realized-mixture audit (see
[../open-questions-answered.md](../open-questions-answered.md), open items), a
stratified-sampling data loader, and RL / elicitation / multi-model experiments on synthetic
task families with replicates.

**Question posed (Danielle, 2026-08-21).** How far can models be pushed at really small
scales (≤150M, e.g. in DataDecide); how to even measure a training or hyperparameter-fitting
signal there; the suspicion that DataDecide's small per-recipe sampling fraction produces
real nonstationarity or misses the intended percentages unless sampling is stratified
throughout training; and whether this is a space that lets academic labs, dynamics-focused
science, and local model runners benefit even if big labs don't care — with elicitation, RL,
multi-component systems, and experiments with confidence intervals all feasible because the
models run so fast. See [../danielle-inputs.md](../danielle-inputs.md).

Related-work claims below are unverified (see [README.md](README.md)).

---

## 2026-08-21 — Response (item 3 of a three-item answer, plus its synthesis)

"This space is real and growing — BabyLM, the nanoGPT-speedrun culture, small-scale-proxy
work (using tiny models to predict large-model training instabilities), and DataDecide itself
are all evidence that 'rigorous science at 150M' has an audience even where big labs shrug.
But I want to sharpen what the comparative advantage actually is, because 'small models are
cheap' isn't it — everyone knows that. The advantage is that cheapness converts into
statistical practices the field otherwise never gets: 20+ seeds, factorial designs, power
analysis, preregistered comparisons, real confidence intervals. Nearly every big-lab training
paper is n=1 per configuration. A lab whose identity is 'we run LM training experiments to
wet-lab evidentiary standards' is differentiated, and the measurement problem you flag — how
do you even detect a training or HP-fitting signal at 150M — is the research question, not
the obstacle. That's [the trajectory noise floor], [IRT's] θ-precision, and the probe battery
[see checkpoint-tomography.md], unified: the instruments exist precisely because raw metrics
are too noisy at this scale, and small scale is where you can afford the replicates to
validate the instruments."

**The stratified-sampling observation is a paper on its own.** "You're right about the
mechanism: if a recipe specifies source proportions but sampling draws a small fraction of
each corpus without per-window stratification, the realized mixture in any given training
window can drift from nominal — so every run has an implicit, unintended curriculum, and
'the recipe' as an intervention is not what anyone thinks it is. This is the time-resolved
extension of [the realized-composition finding] (labels ≠ realized token shares, now
per-window rather than in aggregate). It's checkable: OLMo-style training logs data order
deterministically, so you can reconstruct the realized mixture per window for every
DataDecide run and measure the nonstationarity directly. If it's substantial, it (a) is a
standalone audit paper in the same vein as [the data card] but with dynamics implications,
(b) confounds every timing/curriculum claim built on these suites including your own U_c(t)
program — so you need to know regardless — and (c) motivates a concrete artifact: a
stratified-sampling data loader as the fix, which is exactly the kind of contribution the
small-scale-science community adopts."

**RL / elicitation / multi-component systems at ≤150M.** "Viable, with one honest constraint
— the capability floor means questions must be about dynamics, optimization, measurement,
and data, not capabilities. RLVR on natural tasks mostly won't elicit anything at 150M; on
synthetic/formal task families (algorithmic tasks, controlled grammars, arithmetic curricula)
it works and gives you the first-ever RL training experiments with error bars. Same for
populations of interacting small models. The standard reviewer attack is external validity
('does any of this hold at 7B?'), and the mitigations are: use suites with scale ladders so
every claim gets a does-it-trend-with-scale check, and frame claims as being about the
training process rather than the artifact."

**Synthesis: one lab thesis.** "Your three items aren't three directions — they're one lab
thesis. The small-scale platform generates the model populations that IRT requires as
respondents; the MoE repo [see moe-analysis-program.md] is a validated apparatus already
sitting in that scale range with a categorical observable dense models lack; and the probe
battery + noise-floor work is the shared instrument suite. 'Measurement science of
language-model training at academic scale' is a coherent identity that big labs structurally
won't compete with — not because they can't, but because n=20-seed experiments on 150M models
will never be their incentive. It's yours if you want it."

---

## 2026-08-21 — Response (to the expanded version of the question: 10M–150M, elicitation and post-training, dense vs. MoE at tiny scale)

"When I line it up against everything above, it isn't really a fourth program. It's two
things at once: the *practical payoff* of your measurement program, and the *experimental
substrate* the other programs need anyway."

**The measurement question is the deep one.** "At 10–50M, the standard eval stack collapses:
most benchmark items sit at chance, accuracy is quantized into a few reachable values, and
seed variance swamps treatment effects. So 'how do you even measure a training or hpm
signal' is not a preliminary annoyance — it's the scientific question, and it's exactly what
your T0 portfolio produces instruments for. The IRT project is the clearest case: item
difficulty/discrimination parameters tell you directly which items have any discriminating
power in the 10–150M ability range (most don't — the effective test length of MMLU-style
suites at 10M is close to zero), so 'an eval that works at tiny scale' is a *derived
artifact* of the IRT fit: select items whose difficulty brackets the tiny-model θ range, score
with likelihood margins, report θ with standard errors instead of accuracy. Likewise the
noise-floor module tells you the minimum detectable effect per scale, and the
drift/diffusion SNR table tells you which metrics carry signal down there."

**The decision-reliability frontier (a paper).** "DataDecide's own premise is that 150M
decisions predict 1B rankings; Signal-and-Noise showed measurement interventions improve
decision accuracy. The tiny-scale extension is 'how far down does reliable decision signal
survive, as a function of measurement method' — decision accuracy vs. compute curves where
the treatment is the measurement stack (accuracy vs. margins vs. θ vs. IRT-selected item
subsets). If IRT-based measurement moves the reliable-decision floor from 150M to 30M, that's
a headline with an obvious constituency (everyone doing academic-compute data/hpm work), and
it's mostly analysis over tables you already have plus modest tiny-model evals."

**The non-stationarity catch: two failure modes and an intervention.** "A 10M-scale run
consumes a small percent of most of these corpora, so the *realized exposure* of that
specific run — not the recipe's nominal composition, not even the shard-level token masses
you already corrected — depends entirely on shard ordering and sampling implementation. Two
distinct failure modes: the run's time-averaged mixture deviating from nominal (a bias), and
within-run compositional drift (non-stationarity — effectively an unintended curriculum).
Your [realized-composition] manifest machinery makes both auditable: reconstruct the actual
token stream order for the small-scale runs and plot realized composition as a function of
training position, per scale. If there's material drift, every small-scale recipe comparison
in the suite is partially confounded with data *order*, and the confound shrinks with scale —
which would itself be a candidate explanation for why small-scale decisions sometimes
mispredict large-scale ones. And the interventional follow-up is cheap at exactly this
scale: retrain 10–50M models with stratified vs. sequential sampling of the same recipe, n=10
seeds, and measure the order effect directly. Note the connection to your plasticity thread:
order effects *are* critical-period phenomena, and small models with proportionally huge
compositional drift are where they'd bite hardest."

**Tiny MoEs: the sweep's natural downward extrapolation.** "You found total-parameter
benefits persisting to 128:1 ratios and optimal expert size depending only on active params
— both invite the question of where those laws break as active scale shrinks. There are
concrete mechanisms that should impose a floor: each expert sees roughly budget/E tokens, so
at tiny scale experts fall below their own critical data threshold; the router itself is a
small model that has to learn a useful partition, and the routing-shallowness problem
(assignments collapsing to token-ID/frequency clustering) plausibly worsens as capacity
drops. So the question 'does a 5M-active, 500M-total MoE beat a 5M dense model, and does its
routing learn anything non-trivial' has real stakes for the local-model audience *and* feeds
the MoE analysis program: the taxonomy-realness question acquires a scale axis, and the
failure mode of routing at tiny scale is informative about what routing is doing at normal
scale. Your hpm guidance from the sweep is what makes this credible — the classic failure of
tiny-scale comparisons is that one arm is mistuned, and you're one of few groups holding
validated small-scale MoE hyperparameters."

**Elicitation and post-training: the specialist frontier, with IRT as curriculum.** "The
honest prior art here is TinyStories and the BabyLM line on one side (restrict the task
distribution and small models become shockingly capable — capability was never purely a
parameter question, it's parameters *relative to distribution breadth*) and the speedrun
community (modded-nanoGPT etc.) on the other, which is the strongest existing evidence for
your 'design decisions are tuned for larger models' thesis — at GPT-2 scale they've found
different optimizers, schedules, and architectural details win, with enormous cumulative
gains. The open space isn't 'can tiny models do things' but the *systematic* version: the
capability-per-parameter frontier as a function of distribution narrowness, measured
properly."

"For RL specifically, the binding constraint is reward signal: a base model with ~0% pass
rate on a task yields no gradient. The fix is ability-matched task ladders — and notice this
is your IRT machinery again, now as *curriculum design*: pick tasks whose difficulty puts the
model's pass rate in the informative band, advance the ladder as θ moves. 'IRT-scheduled RL
for small models' is a cute, self-contained methods paper, and it's the kind of thing that
only works because the models are fast enough to run the full loop hundreds of times. Same
for multi-component systems: at 10–50M you can afford real factorial designs over system
configurations with confidence intervals, which nobody at 7B can."

**How it composes: tiny models as the *Drosophila* of the program.** "Every design above
that's compute-gated at DataDecide scale — the (data × schedule) factorial, the
ε-perturbation response tensor, the stage × type plasticity map, the reroute-vs-rewrite
causal controls — becomes fully powered at 10–50M: twenty seeds instead of three, full
factorials instead of corner samples, preregistered analyses with the noise floor known in
advance. So the sequencing story is: the T0 measurement papers build the instruments; the
tiny-scale program is where the interventional science gets run *properly first*; the
DataDecide/MoE-scale versions become confirmation at 2–3 points on a scale ladder rather
than underpowered first attempts."

"The one risk to manage explicitly is external validity, and the right response is to be
selective about *which* questions you ask down there. Questions about **training dynamics
and mechanisms** — order effects, plasticity depletion, schedule artifacts,
rerouting-vs-rewriting, response-to-data-type — are plausibly scale-portable, and a scale
ladder can check. Questions about **capability emergence** mostly aren't, and a tiny-scale
program that drifts toward 'can we make a 10M model reason' will produce results nobody can
use."

**Two fastest wins (plus one medium-term).** "The decision-reliability frontier paper
(measurement methods vs. minimum reliable scale — nearly pure analysis, direct extension of
work you're doing anyway) and the realized-exposure audit with the stratified-sampling
intervention (concrete, cheap, potentially explains an existing anomaly in a widely used
suite, and seeds the order-effects/plasticity thread). The tiny-MoE floor study is the best
medium-term one because it's the unique intersection of your sweep, your infrastructure, and
a question with both a scientific and a local-model audience. And all three quietly recruit
the same shared instruments — which at this point is the clearest sign the portfolio has
converged: every new direction you've raised has turned out to be a new consumer of the same
five or six pieces of measurement infrastructure, which is exactly what a coherent research
program looks like from the inside."

---

## 2026-08-21 — Positions in the ranked lists (full lists in `../portfolio-rankings.md`)

- Decision-reliability frontier: **workshop-sized #5** ("robust to outcome (the frontier is
  the artifact, wherever it sits)… its clock effectively starts after [IRT]").
- Realized-exposure audit + order-effect experiment: **workshop-sized #7**; also part of
  **full-conference #2**.
- The whole tiny-scale program: **full-conference #6, "Measuring Learning Where Benchmarks
  Can't See"** (CoLLAs or ICLR): "the most iteration-heavy entry — getting *any* clean
  RL/posttraining signal at 10–50M may take several design loops… The frontier arm is the
  safe core; the RL arm is the differentiator. **Expected impact: medium** — big-lab
  reviewers may shrug. **Ceiling: medium-high**; if IRT-scheduled reward shaping demonstrably
  unlocks RL at scales where naive reward gives zero gradient, the local-model and academic
  communities adopt it."

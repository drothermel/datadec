# Tiny-scale measurement — how small can you measure, and what can you do down there?

> **Draft scaffolding (2026-08-21).** This doc was promoted from a topic. The quoted material in
> §4 is external text; the core steps, doability notes, impact ratings, and infrastructure
> sequence in §1–§3 are synthesized scaffolding not yet reviewed by Danielle. Treat them as
> provisional until this note is removed.

**One-line pitch.** At 10M–150M the standard eval stack collapses: most items sit at chance,
accuracy is quantized, seed variance swamps treatment effects. "How do you detect a training
or hyperparameter signal at all" is the research question. Measure the decision-reliability
frontier — how far down the scale ladder reliable decision signal survives as a function of
measurement method — derive an eval that works at tiny scale from the IRT fit, and use the
resulting fast, replicated substrate for elicitation, post-training, and RL experiments with
confidence intervals.

IDs: TINY-1–TINY-3, TINY-opt-1–TINY-opt-4.

**Scope rule.** Ask dynamics / mechanism / measurement questions (plausibly scale-portable,
checkable on a scale ladder), not capability-emergence questions.

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short
continued-training branches; **T3** = new pretraining runs.

---

## 1. What the project involves

### Core experiment (T0 / T1)

1. **Decision-reliability frontier (TINY-1).** Decision accuracy vs. compute curves over
   DataDecide's size ladder where the *treatment is the measurement method*: accuracy vs.
   likelihood margins vs. IRT θ vs. IRT-selected item subsets. The frontier — the smallest
   scale at which decisions reliably predict 1B rankings, per method — is the artifact.
2. **IRT-derived tiny-scale eval (TINY-2).** From a fitted item-response model on the
   per-instance tables: select items whose difficulty brackets the tiny-model θ range, score
   with likelihood margins, report θ with standard errors instead of accuracy. Report the
   effective test length of standard suites at each size.
3. **Minimum detectable effect per scale (TINY-3).** Pooled seed variance and
   trajectory-window replicates per metric and size; which metrics carry signal at 10–90M.

### Optional directions

- **TINY-opt-1: Capability-per-parameter under distribution narrowing.** The systematic
  version of TinyStories/BabyLM-style results: capability as a function of distribution
  breadth, measured with TINY-2 instruments.
- **TINY-opt-2: IRT-scheduled RL.** On synthetic/formal task families (algorithmic tasks,
  controlled grammars, arithmetic curricula): pick tasks whose difficulty puts the pass rate
  in the informative band and advance the ladder as θ moves. RL training experiments with
  error bars.
- **TINY-opt-3: Multi-component systems.** Factorial designs over system configurations with
  confidence intervals.
- **TINY-opt-4: Design decisions tuned for larger models.** Which optimizer/schedule/
  architecture defaults are wrong at 10–50M (speedrun-style evidence), measured against the
  noise floor.

---

## 2. Doability and impact

### Overall doability: TINY-1–3 **high** (analysis over existing tables); options **medium** (iteration-heavy)

- Per-instance tables exist at 4M/20M/60M/90M (one seed each) and 150M–1B (three seeds);
  aggregate tables at all 14 sizes with three seeds (see
  `docs/open-questions-answered.md`). Seed-variance estimates below 150M need pooled
  floors.
- The compelling TINY-1 wants the IRT fit first; a simpler-metrics version can be written
  standalone.
- RL/post-training at 10–50M: the binding constraint is reward signal (~0% pass rate gives
  no gradient); natural tasks mostly will not elicit; synthetic families will. Several
  design loops likely before clean signal.
- External validity is the standard reviewer attack; mitigations are the scale ladder and
  claims about the training process rather than the artifact.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| TINY-1 frontier | **Medium-high** | Robust to outcome; clear constituency in academic-compute work. |
| TINY-2 derived eval | Medium-high | A practical instrument people adopt. |
| TINY-3 MDE per scale | Medium (supporting) | Every later tiny-scale claim needs it. |
| TINY-opt-1 narrowing | Medium | Systematizes folklore; risk of drifting toward capability claims. |
| TINY-opt-2 IRT-scheduled RL | Medium-high if it works | "Unlocks RL at scales where naive reward gives zero gradient." |
| TINY-opt-3 factorials | Medium | Methodological demonstration. |
| TINY-opt-4 defaults | Medium | Practical for local-model users. |

---

## 3. Infrastructure build sequence

1. **Outcome table with full structure** (recipe × scale × seed × step × task; accuracy and
   continuous metrics; pairwise-decision helpers).
2. **Response-matrix builder + IRT fit** (binary and margin models; θ with standard errors;
   item parameters). Tested on synthetic matrices.
3. **Noise-floor module** (pooled seed variance, windowed replicates, item bootstrap) per
   metric and size.
4. **Frontier analysis (TINY-1)** and **item-selection / derived-eval construction (TINY-2)**.
5. **Tiny training harness** (10–50M, many seeds, deterministic data order) for the optional
   directions; synthetic task-family generators and an RL loop for TINY-opt-2.


---

## 4. External assessments

Dated, attributed-by-date notes from external review conversations, recorded for
consolidation — not decisions. Only notes about this project are kept here. Related-work
claims in quoted text are unverified.

### 2026-08-21 — origin: small-scale science (two responses)

*First response:*

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
respondents; the MoE repo [see moe-partitions.md / moe-movement.md] is a validated apparatus already
sitting in that scale range with a categorical observable dense models lack; and the probe
battery + noise-floor work is the shared instrument suite. 'Measurement science of
language-model training at academic scale' is a coherent identity that big labs structurally
won't compete with — not because they can't, but because n=20-seed experiments on 150M models
will never be their incentive. It's yours if you want it."

---

*Second response:*

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

### 2026-08-21 — positions in ranked lists (full lists in `docs/portfolio-rankings.md`)

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

- The decision-reliability frontier is sub B of **P1** and the realized-exposure audit is sub
  B of **P2** in the "four main-conference projects" list (`../portfolio-rankings.md`).

### 2026-08-18 — prior art for the post-training / RL options (from the Research Trajectory page)

- Small-model post-training can regress: "small-scale SFT on Qwen2.5-1.5B can reduce MATH-500
  accuracy from 23.8% to 18.4% even while eliciting reasoning-style behaviors" (Chen et al.,
  arXiv 2505.17988); Long-CoT degradation attributed to error accumulation (*Through the
  Valley*, EMNLP 2025). Spurious-reward gains are Qwen-specific (Shao et al., ICML 2026).
  "Small benchmarks produce unstable estimates, making multiple seed runs essential"
  (Hochlehnert et al., COLM 2025) — the replicate-heavy design here is the response. Full
  list in `docs/topics/pretraining-to-posttraining.md`.

### 2026-08-18 — design inputs for the post-training / RL options (from the Research Trajectory page)

- **Within-reach tasks (TINY-opt-2).** "The 'no movement' is a property of the model–task
  pair, not the model… design verifiable tasks whose difficulty sits just above the base
  models' zero-shot ability. Then recipe effects on post-training become measurable at
  sweepable scale." Open question it creates: "do recipe effects on within-reach tasks
  predict recipe effects on out-of-reach tasks at larger scale?" — "only needs a couple of
  larger validation runs, not a factorial sweep."
- **Proxy metric as the contribution.** "A continuous, low-variance predictor of
  'RL-ability' measured on the base model (NLL on gold reasoning traces, pass@k at large k,
  entropy at decision points, even plasticity-style statistics like curvature or feature
  rank)… Validating a proxy needs far fewer runs than detecting an intervention effect."
- **Power analysis for post-training experiments.** "How many seeds does a claimed RLVR
  delta actually require at 150M vs 1B, and how much of the published small-scale
  literature clears that bar?" — mostly reanalysis of public results plus modest runs.
- **Asymmetric design.** Full sweep with seeds only where cheap; expensive budget on "two or
  three confirmation runs testing a *ranking* the cheap tier predicted." Full discussion in
  `docs/topics/posttraining-experiment-design.md`.

### 2026-08-18 — a gradient-free proxy candidate (from the Research Trajectory page)

- In-context learning curves — "loss on the k-th demo as a function of k — a continuous,
  smooth, per-token metric, no benchmark thresholds" — measured on existing checkpoints,
  averaged over prompt orderings and samples, as a candidate predictor of post-training
  movement; and round-trip reconstruction fidelity through a natural-language bottleneck as
  "a capability probe — one that's graded rather than thresholded, unlike pass@1." Full
  discussion in `docs/topics/icl-as-posttraining.md`.

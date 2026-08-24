# Tiny-scale measurement — how small can you measure, and what can you do down there?

**Program pillars served:** how (measurement where benchmarks can't see), apex (elicitation and post-training at tiny scale). (Program: `README.md` → Program.)

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

IDs: TINY-1–TINY-3, TINY-opt-1–TINY-opt-5.

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
- *Coordination note (2026-08-22):* `elicitation-gain.md` (`ELI`) runs the within-reach
  question as an existence test under an oracle interface at every DataDecide size (ELI-1);
  reuse its result before designing TINY-opt-1.
- **TINY-opt-4: Design decisions tuned for larger models.** Which optimizer/schedule/
  architecture defaults are wrong at 10–50M (speedrun-style evidence), measured against the
  noise floor.
- **TINY-opt-5: The which-loss axis.** Decision-accuracy-vs-compute curves on the same
  checkpoints with the *response variable* as the treatment: per-token vs. per-byte
  normalization, gold-span vs. whole-sequence likelihood (OLMES per-character correct
  probability), token-selected likelihood (LongPPL-style key tokens; Patel et al.'s
  expert-trajectory reweighting), and tokenization-marginal likelihood. T0 on released
  tables; the literature hook is in
  `../topics/reference/loss-alternative-metrics-literature.md` (2026-08-22 §4 note).

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
| TINY-opt-5 which-loss axis | Medium–High | Cheap; directly answers "which loss should a small-scale suite report"; feeds IRT's response-model choice and DCARD's table conventions. |

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

### 2026-08-22 — two small points for the which-loss axis from the OLMES metric walkthrough

From Danielle's metric-column conversation (record:
`../topics/reference/datadecide-data-pipeline.md`, last entry). (1) `correct_prob` and
`sum_logits_corr` are exp-related (checked on released rows), so as *ranking* metrics they
are one metric; they differ only for regression-style prediction, where the log form is
the better-behaved one — TINY-opt-5's candidate list should not count them twice.
(2) Danielle proposed `uncond_correct_prob` — the correct option's likelihood with the
answer-string prior removed — as an additional continuous proxy candidate; the response
endorsed it on the paper's "continuous beats discrete at small scale" finding, but no
evidence was offered. It is cheap to add to the TINY-opt-5 sweep once its exact definition
is pinned from the evaluation code (the conversation's definition is a guess; see the
DCARD note on metric provenance). The per-char variants are per-character geometric
means, not probability ÷ length, which matters for how they are interpreted on the axis.

### 2026-08-22 — the "which loss" axis: what the loss-alternative literature offers

From Danielle's SciSpace review of CE/NLL alternatives for evaluation (record in
`../topics/reference/loss-alternative-metrics-literature.md`). TINY's measurement-method
axis should enumerate the loss *variants*, not just loss vs. accuracy: (a) per-token vs.
per-byte normalization (constant tokenizer across DataDecide makes this a no-op within
the suite, decisive across suites); (b) gold-span vs. whole-sequence likelihood (OLMES
per-character correct-probability is the gold-span case); (c) token-selected likelihood
— LongPPL's key-token perplexity (2410.23771) is the published template for "compute
loss only on tokens a reference model says are informative", and Patel et al.'s
expert-trajectory reweighting is the same move with a strong model as the selector; (d)
tokenization-marginal likelihood (Cao & Rimell 2021; Vieira et al. 2412.03719) as a
robustness check. Candidate TINY-opt: decision-accuracy-vs-compute curves for each
variant on the same checkpoints, which is the frontier with the response variable as
the treatment. Representation-side metrics (Diff-eRank) are a separate, hidden-state
readout — possible GEO/TOK interest, not a TINY method.

### 2026-08-22 — the incumbent proxy for small-scale decisions: Patel et al. 2026

From Danielle's SciSpace literature review on small-scale evaluation metrics
(record in `../topics/reference/small-scale-evaluation-metrics-literature.md`; arXiv 2605.18607, Patel, Reddy, Mosbach & Bahdanau — Mila/ServiceNow, not AI2 as one review version claimed). "Forecasting Downstream Performance of LLMs With Proxy Metrics" builds
80 proxies (10 token-level statistics × 8 weightings) from one forward pass over
expert-written solutions and, on DataDecide's 25 corpora, reports decision accuracy
> 0.85 for the 1B target at ~10⁻⁵ of target compute using frequency-weighted top-5
accuracy — i.e. it already claims the "how far down does decision signal survive"
result TINY-1 is built to measure, from the same tables. Consequences: (1) TINY's
method axis must include their proxy family as the incumbent, not only DataDecide's
own continuous metrics; (2) their result is at the corpus-ranking level for one
target; TINY's replicate/seed and per-task analyses, and the IRT θ comparison, are the
parts they do not cover; (3) the expert-trajectory trick (score the candidate on a
strong model's solution) is the natural proxy to add to the decision-reliability
frontier. Unverified beyond the two agent summaries; read the paper first.

### 2026-08-22 — WSD arms in the substrate

Danielle would add WSD arms to DataDecide-dense if it is built at all (verbatim in
`../topics/reference/datadecide-data-pipeline.md`). Hypothesis this enables for TINY:
annealed readouts improve measurement SNR most at small scales, where wall oscillation
is proportionally largest relative to signal — if true, WSD + branches is what makes
10M-scale experimentation measurable, a finding only a controlled small-scale suite can
produce. Candidate TINY-opt direction; unverified reasoning from the response.

### 2026-08-22 — the "DataDecide-dense" substrate

From a conversation on the data layer (record in
`../topics/reference/datadecide-data-pipeline.md`). Three needs were observed to
triangulate on one artifact: this project's need for a powered small-scale substrate,
the order-effect reruns in recipe featurization, and the 5–10-checkpoint sparsity of the
released 4M–8M runs. Proposed object: a few recipes × the 2–4 smallest scales × 10+
seeds with dense checkpointing and full logging (training loss, executed LR schedule,
data-order manifest, per-token held-out losses on a frozen probe set), emitting the
results store's existing schema. Its first result is the reproduction-gap measurement
(faithful rerun vs. published checkpoint, relative to seed variance), which licenses
treating retrained variants as commensurable with the released suite. Scope caution from
the response: it is a substrate for TINY, REC's interventional arm, and MPL/LR
validation — not a larger retraining suite. Design doc (recipes, scales, seeds, cadence,
logging spec) suggested as worth writing now; not started.

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
  list in `docs/topics/reference/pretraining-to-posttraining.md`.

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
  `docs/potential-projs/movement-microscope.md`.

### 2026-08-18 — a gradient-free proxy candidate (from the Research Trajectory page)

- In-context learning curves — "loss on the k-th demo as a function of k — a continuous,
  smooth, per-token metric, no benchmark thresholds" — measured on existing checkpoints,
  averaged over prompt orderings and samples, as a candidate predictor of post-training
  movement; and round-trip reconstruction fidelity through a natural-language bottleneck as
  "a capability probe — one that's graded rather than thresholded, unlike pass@1." Full
  discussion in `docs/potential-projs/icl-elicitability.md`.

### 2026-08-18 — the scale ladder as an identifiability-vs-scale instrument (from the Research Trajectory page)

- "The Platonic Representation Hypothesis (Huh et al. 2024)… is an empirical claim that
  *identifiability improves with scale*. If true, path-dependence — and with it critical
  periods, recipe effects on elicitability, all of it — should *wash out* as models grow,
  which would mean your small-scale effects have a scale ceiling; if false, the scars
  persist and pretraining-recipe choices matter at the frontier. Your ladder design (tiny
  transformers → DataDecide → sparse large confirmations) is, almost accidentally, the
  right instrument for measuring how fast the underdetermination closes with scale." This
  sharpens the external-validity rule: the ladder is not just a check, it is a measurement.
  See `docs/topics/reference/identifiability-literature.md`.

### 2026-08-22 — substrate and scaling result from the reinit literature pass

- PolyPythias (ICLR 2025; arXiv 2503.09543): 50 pretraining runs, 9 seeds × 5 sizes
  (14M–410M), ~7,000 checkpoints — a released many-seed substrate in exactly this scale
  range. *Can Scale Save Us From Plasticity Loss in LLMs?* (arXiv 2606.24752): plasticity
  loss at 5M–314M follows a sublinear scaling law in both continual and stationary
  settings. *The Butterfly Effect* (arXiv 2506.13234): trajectories are highly sensitive to
  initial conditions — a seed-count argument. See
  `docs/topics/reference/reinit-and-transfer-literature.md`.

### 2026-08-18 — what can and cannot be averaged at a fixed checkpoint (from the Research Trajectory page)

- "You can't buy variance reduction on a fixed checkpoint by re-evaling with new seeds";
  configuration variance (demos, order, template) is systematic and should be swept as a
  bias axis. TINY-3's minimum-detectable-effect estimates must therefore come from
  training-side replicates (seeds, trajectory windows) and item bootstraps, not eval
  reruns. See `docs/topics/reference/evaluation-methodology-literature.md`.

### 2026-08-22 — absorbed from the post-training experiment-design topic (now deleted)

Optional directions added to this project: a continuous, low-variance predictor of
"RL-ability" measured on the base model (NLL on gold reasoning traces, pass@k at large k,
entropy at decision points, plasticity-style statistics) — "validating a proxy needs far
fewer runs than detecting an intervention effect"; a power analysis for post-training
experiments (how many seeds a claimed RLVR delta requires at 150M vs. 1B); tuning-response
curves (performance vs. search budget per paradigm) as the falsifiable replacement for
matched-budget comparisons; fully synthetic testbeds where a seed costs minutes. The
"asymmetric design": full sweep with seeds only where cheap, then two or three confirmation
runs testing a *ranking* the cheap tier predicted. Original text preserved in
`potential-projs/movement-microscope.md` §4.
## 5. Related work and positioning

*Purpose: the paper-facing synthesis — the prior-art landscape, this project's
position in it, and what each closest neighbor lacks. Unlike §4 (a dated intake
log, which grows by appending new entries **above this section**), §5 is a
current-state statement: rewrite it as understanding changes. Positioning claims
are Danielle's to make; agent-supplied literature claims anywhere in this document
are unverified leads, not established facts.*

**Status: raw material assembled from repository records (2026-08-24); positioning not yet
written.** All identifiers below are unverified; several come from agent-generated SciSpace
reviews whose bibliographies contain swapped and fabricated entries.

**The load-bearing items and the role each plays:**

- **Patel, Reddy, Mosbach & Bahdanau 2026, *Forecasting Downstream Performance of LLMs With
  Proxy Metrics*** (arXiv 2605.18607) — *the incumbent*. 80 token-level proxies over
  expert-written solution trajectories rank DataDecide's 25 corpora for the 1B target at
  decision accuracy > 0.85 at ~10⁻⁵ of target compute, from the same tables TINY-1 uses.
  Their proxy family must sit on TINY-1's method axis, and their expert-trajectory trick is
  the natural addition to it. Per the SciSpace review, unverified.
- **Heineman et al., *Signal and Noise*** (NeurIPS 2025 per the record; ledger 2508.13144) —
  *the measurement-intervention precedent*: continuous metrics beat accuracy, noisy subtasks
  filtered, ~900K results over 465 models including DataDecide.
- **OLMES, *A Standard for Language Model Evaluations*** — *the harness and the metric
  definitions*; its per-character correct probability is TINY-opt-5's gold-span case.
- **DataDecide (Magnusson et al., ICML 2025, 2504.11393)** — *the premise TINY extends
  downward*: 150M decisions predict 1B rankings ~80% of the time; continuous proxies at
  0.01% compute.
- **LongPPL key-token perplexity (Fang et al., 2410.23771)** — *metric precedent* for
  TINY-opt-5: the published template for computing loss only on reference-model-selected
  tokens. Companions on the same axis: bits-per-byte (Biderman et al. 2405.14782; Paloma's
  per-domain protocol 2312.10523) and tokenization-marginal likelihood (Cao & Rimell 2021;
  Vieira et al. 2412.03719).
- **ADO (2410.11820) and Pechi et al. (2305.17266)** — *contrast cases*: small proxies often
  fail to predict larger models, and a small-scale break below ~2.2e15 FLOPs — the bounds a
  frontier measurement has to sit against.
- **PolyPythias (2503.09543)** — *the many-seed substrate in range* (9 seeds × 5 sizes,
  14M–410M, ~7,000 checkpoints), alongside sublinear plasticity-loss scaling at 5M–314M
  (2606.24752) and *The Butterfly Effect* (2506.13234) as the seed-count argument.
- **Hochlehnert et al., *A Sober Look…*** (COLM 2025; 2504.07086) — *the replicate-design
  citation*: small benchmarks give unstable estimates, so multiple seeds are essential;
  with Chen et al. 2505.17988 and Shao et al. 2506.10947 as the small-model post-training
  cautions behind the RL options.
- **The fixed-checkpoint variance rule** — re-evaluating a fixed checkpoint with new seeds
  buys nothing, so TINY-3's minimum detectable effects come from training-side replicates
  and item bootstraps, not eval reruns.
- **u-µP (Blake et al., 2407.17465)** — *the TINY-opt-4 hook*: DataDecide is not
  µP-parametrized, so its cross-size comparisons carry a "was the small model's LR optimal"
  confound.
- **TinyStories, the BabyLM line, and the nanoGPT-speedrun culture** — *folk precedents*
  named in the 2026-08-21 origin responses for TINY-opt-1 and TINY-opt-4; external text,
  no citations on record.

Full inventory (every possibly relevant item on record, grouped by theme) in
`related-work/tiny-scale-measurement.md`. Main accumulators:
`../topics/reference/small-scale-evaluation-metrics-literature.md`,
`../topics/reference/loss-alternative-metrics-literature.md`,
`../topics/reference/evaluation-methodology-literature.md`,
`../topics/reference/pretraining-to-posttraining.md`,
`../topics/reference/reinit-and-transfer-literature.md`,
`../topics/reference/parametrization-and-hp-transfer.md`,
`../topics/reference/identifiability-literature.md`; provenance in
`../litreview/citation-verification-ledger.md`.

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

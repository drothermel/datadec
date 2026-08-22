# LR schedules, annealing branches, and checkpoint merging — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: annealed readouts (ANN) and the WSD suite (WSD) are built on the
stable-phase-plus-decay-branch methodology and on checkpoint merging as a decay proxy; the
same machinery is the branch runner for token-level movement and functional featurization.

---

## 2026-08-18 — the annealing confound and its workarounds (from the Research Trajectory page)

**The confound.** "DataDecide's models were trained OLMo-style with cosine schedules, so every
intermediate checkpoint sits mid-schedule with high residual LR — in river-valley terms,
evals on those checkpoints measure 'position along river + current distance up the wall,'
and the wall component is schedule-dependent noise relative to the question you care about.
Post-training from such checkpoints inherits the confound (you're fine-tuning from a point
high on the wall)."

**Stable phase + decay branches**
- Hägele et al. 2024, *Scaling Laws and Compute-Optimal Training Beyond Fixed Training
  Durations* — "made essentially your argument as a methodology proposal: constant LR +
  short cooldown matches cosine, so scaling-law and data experiments should use
  stable-phase runs with cheap decay branches instead of retraining full cosine runs per
  budget."
- MiniCPM (*Unveiling the Potential of Small Language Models with Scalable Training
  Strategies*) — "established the practical template — a stable-phase checkpoint plus a
  fixed-length decay matches full cosine baselines, with ~10% decay completing convergence
  and new data mixed in strictly during decay." "The stable-checkpoint-plus-1-sqrt-decay-
  resume protocol is now standard in careful small-scale studies."
- Llama 3 (*The Llama 3 Herd of Models*) — annealing-based data-quality assessments;
  Blakeney et al., *Does your data spark joy? Performance gains from domain upsampling at
  the end of training*. "So 'annealing branches as the correct eval' is validated practice
  — but no *open, multi-recipe suite* has it."
- OLMo (*OLMo: Accelerating the Science of Language Models*) — the training setup
  DataDecide inherits.

**Checkpoint merging as pseudo-annealing**
- WSM (*WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training*)
  — "merging recent checkpoints with weights derived from an emulated decay curve provides a
  robust annealed model without ever altering the live learning rate… WSM-merged models
  consistently and closely mirror the results of a true LR anneal at intermediate stages of
  long runs."
- Nemotron 3 (*Nemotron 3 Super…*) — "applies sliding-window checkpoint merging to get
  stronger quality readouts without dedicated decay runs, estimating savings of ~16% of
  total pretraining FLOPs."
- Open question: "whether merging-as-annealing-proxy works on *cosine* mid-run checkpoints
  (varying LR within the merge window) rather than stable-phase ones."

**Analytic correction**
- The multi-power law (Luo et al., arXiv 2503.12811) "predicts the loss drop from a
  hypothetical decay given the trajectory so far, so you could correct unannealed loss
  curves analytically. It won't give you downstream metrics, but it quantifies how much
  each recipe's apparent ranking is schedule artifact."

**Caveats**
- "DataDecide found intermediate-checkpoint *decisions* matched compute-equivalent final
  checkpoints — so the confound may partially cancel for rankings even while distorting
  levels and post-training. Testing when it cancels vs. doesn't is exactly what your WSD
  branches would settle."
- "The river-valley theory predicts the decay phase itself makes progress along the river —
  so annealed evals aren't a pure 'reveal' either; branch length becomes a parameter to
  control."

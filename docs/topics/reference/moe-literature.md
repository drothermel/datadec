# Mixture-of-experts training dynamics — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: routing is the categorical observable behind MoE movement (MOVE), the
learned token taxonomy behind MoE partitions (PART), the readout of the MoE recipe suite
(MSUITE), and the routing follow-up in trajectory drift/diffusion; expert permutation is
the textbook non-identifiable latent for the identifiability framing.

---

## 2026-08-22 — Mixture of Parrots (Jelassi et al., ICLR 2025): experts buy memorization, not reasoning

**Danielle's prompt (verbatim):**

> I want you to summarize this paper's key findings.
> https://proceedings.iclr.cc/paper_files/paper/2025/file/5bc3356e0fa1753fff7e8d6628e71b22-Paper-Conference.pdf

Artifacts: `~/drotherm/data/convo-artifacts/2026/scispace-summarize-mixture-of-parrots/` — the paper's full text as markdown and the agent summary (truncated at
its start); `INDEX.md` inside.

**Claims (from the agent summary and the paper abstract in the bundle):**

- *Theory:* there exist graph problems (e.g. path connectivity) that no number of
  experts of a given width can solve, while a dense model of slightly larger width
  solves them easily; conversely, for memorization, total parameter count (not active)
  determines storage capacity, so a sparse model memorizes N sequences with fewer active
  parameters by spreading storage across experts.
- *Synthetic:* on graph problems more experts do not help but wider dense models do; on
  closed-book retrieval (memorizing associations) MoEs beat dense models at equal active
  parameters and improve with expert count.
- *Pretrained (65B tokens, FineWeb-edu / Cosmopedia / Wikipedia):* on world-knowledge
  benchmarks (TriviaQA, Natural Questions) MoEs with ~18M active parameters nearly match
  much larger dense models and scale with expert count; on GSM8K / MATH / ARC, adding
  experts gives no consistent gain while adding active parameters does.
- The title references the "stochastic parrots" critique: expert scaling makes better
  parrots, not better thinkers.

**Why it is here.** This is the concrete citation for "the 'mixture of parrots' line"
already invoked in `../../potential-projs/moe-partitions.md` §4 (the "why does total
capacity keep helping at extreme sparsity?" analysis): decompose total-parameter gains
by eval type (memorization-heavy vs. reasoning-heavy; per-token by frequency band). The
paper's own dense-vs-MoE pretraining series at fixed active parameters is the template
for that decomposition, and its theorem gives the reasoning-side null a mechanism
(width-bounded expert circuits) rather than just an empirical pattern. Also relevant to
MSUITE's "does the data choose the experts" question: if experts are storage, recipe
differences should show up as *what* gets stored, which is the routing-taxonomy reading.

**Intake notes.** Summary figures as the agent reported; the full text is on disk for
verification. The pasted summary is missing its first two sections (truncated at
source, not in transcription).

## 2026-08-18 — MoE as a new observable channel at 20–50M active (from the Research Trajectory page)

Prompt context (Danielle): MoE models do something at 20–50M active parameters, so the
program should look all the way down to that scale.

**Suites and metrics**
- FLAME-MoE (*A Transparent End-to-End Research Platform for Mixture-of-Experts Language
  Models*) — "essentially DataDecide-for-MoE at exactly your target scale — seven
  decoder-only MoE models from 38M to 1.7B active parameters, 64 experts per layer, top-8
  gating, with full openness: code, data, checkpoints, routing logs, and evaluation results
  — and their training traces already show expert specialization emerging early and
  intensifying, co-activation staying sparse and stable, and routing behaviors converging
  quickly during early pretraining."
- OLMoE (*Open Mixture-of-Experts Language Models*) — router saturation "defined as the
  average overlap between the top-k experts selected per token at step t versus at
  convergence, rising sharply within the first few thousand steps, with deeper layers
  saturating faster than shallower ones."
- *Three Phases of Expert Routing: How Load Balance Evolves During Mixture-of-Experts
  Training* — "an early balance-prioritizing phase, a stabilization phase where experts
  specialize, and a late relaxation phase trading balance for quality — a non-monotone
  trajectory invisible to post-hoc analysis of converged models, with annealing checkpoints
  confirming the phases are pretraining-specific and stable during fine-tuning."
- *Continual Pre-training of MoEs: How Robust Is Your Router?* — "routing decisions change
  most in early layers, with the no-replay condition showing the most dramatic early-layer
  routing reorganization and the most forgetting — suggesting early-layer routing changes
  may be a key mechanism of catastrophic forgetting in MoEs."
- *The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily
  Domain Expertise* — "specialization patterns in pretrained MoEs resist human
  interpretation, with expert overlap between different models answering the same question
  no higher than between entirely different questions — i.e., independently trained MoEs
  select *unrelated* specialization solutions… routers are linear maps, so hidden-state
  similarity is necessary and sufficient to explain expert-usage similarity —
  specialization is a property of the representation space, not the routing architecture —
  and load-balancing loss provably suppresses shared hidden directions, explaining
  specialization collapse under less diverse data."

**Thoughts**
- "MoE doesn't just make tiny models capable — it adds a *new observable channel* that's
  almost custom-built for your movement microscope… per-token expert-assignment flips."
- "Expert assignment is a textbook non-identifiable latent: the objective is invariant to
  expert permutation, so *which* expert specializes in what is pure trajectory-selected
  symmetry breaking." The Myth paper's cross-model overlap result is "about as clean an
  existence proof of solution-class underdetermination as the field has produced."
- The warning: "the symmetry group now includes expert permutations, which breaks your
  dense-model comparability tools — naive interpolation barriers, checkpoint merging (your
  annealing-proxy trick needs expert matching first or it averages mismatched experts into
  mush), and stitching all require an expert-alignment step, and re-basin methods for MoE
  are immature… 'how to quotient MoE symmetries for checkpoint comparison' is an open gap."
- Practical cautions: MoE knobs (aux-loss coefficients, top-k, expert count, capacity
  factors) "are all folklore-tuned at large scale and may be mis-set for 20–50M active, so
  your regime-mismatch argument applies to your *own* baseline"; noise worsens as scale
  shrinks and routing discreteness plausibly adds eval variance, so the noise-floor stage
  "isn't skippable here, it's more necessary"; keep a dense control ladder at matched
  active parameters.

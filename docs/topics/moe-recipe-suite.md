# Multi-recipe MoE mini-suite — data varied at fixed architecture, routing as the readout

**Status:** topic (staging). Candidate exits: a new `potential-projs/` doc (an MoE sibling of
the WSD retrain suite, with routing/expert-specialization as the outcome channel); plus a
separate, non-project item — rebuttal-sized analyses for the existing Slicing-and-Dicing
paper.

**Context (Danielle, 2026-08-21).** The Slicing-and-Dicing MoE paper (Danielle third author)
ran its grid on current compute across a range of model sizes with a working repo; the paper
is trending weak-accept/weak-reject at NeurIPS and is "lacking interesting analysis." Question:
would using that repo for additional experiments in that range be useful? Interest is in
analysis rather than full empirical grid sweeps. See
[../danielle-inputs.md](../danielle-inputs.md).

Related-work claims below are unverified (see [README.md](README.md)). The response refers
to an earlier "reweighting" of directions (its "#2") that has not been captured here.

---

## 2026-08-21 — Response (section 2 of a longer answer; section 1 not captured)

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
  [see moe-routing-as-data-instrument.md], applied to architecture variation instead of
  data variation."

**The strategic point: the repo is the apparatus for a multi-recipe MoE mini-suite.** "You
have working MoE pretraining at the right scale range, validated configs, and now a
principled default architecture (the paper's own finding: fix expert size by active params,
use dropless routing, ignore the second-order knobs). Slicing-and-Dicing varied architecture
at fixed data; the follow-up varies data at fixed architecture and reads out routing/expert
specialization. Together they're a clean factorial story, and the follow-up is the
analysis-heavy paper you say you actually want — the sweep paper earns you the right to hold
architecture fixed. I'd now upgrade that direction relative to my earlier reweighting,
because its main risk (standing up MoE training infra) just evaporated."

---

## 2026-08-21 — Related: the sweep reread as data

A separate response proposes using the Slicing-and-Dicing sweep itself as a matched-loss
comparison across architectures — "is the token taxonomy a property of the data or of the
architecture?" via cross-model expert matching — as the first-act "sweep-reanalysis paper,"
with the balancing-mechanism sweep as the validation that routing-as-measurement is not a
load-balancing artifact. Recorded in full in
[moe-analysis-program.md](moe-analysis-program.md).

---

## 2026-08-21 — Artifact status

Final checkpoints for the full sweep exist and are headed to Hugging Face; intermediate
checkpoints/logs are likely obtainable for some of the collaborator's upcoming experiments.
Details in [../open-questions-answered.md](../open-questions-answered.md).

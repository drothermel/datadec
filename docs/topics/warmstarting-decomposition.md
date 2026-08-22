# Warm-starting decomposition — why did Ash & Adams' gap happen, and what fixed it?

**Kind:** staging. Candidate exits: a project doc (reproduce the chunked-CIFAR warm-starting
gap with many seeds, then a factorial over modernization axes with the plasticity diagnostic
panel at matched training loss; deliverable is a decomposition of the gap; bridge experiment
with a tiny transformer asking whether warm-starting damages elicitability); or absorption
into tiny-scale measurement / the plasticity reference topic. Proposed as "Chapter 1" of a
retrospective narrative whose later chapter is matched-loss ICL.

**Question posed (Danielle, 2026-08-18).** Start the retrospective with the question that
started it all: Ash & Adams' warm-starting + shrink-and-perturb paper found training from
scratch beats warm-starting. Why didn't it work then, and why don't we hit these problems
now? Confident the plasticity research has the answers, but has not seen a clear
reproduction of the original data with a breakdown of specific causes. See
[../danielle-inputs.md](../danielle-inputs.md).

Related-work claims below are unverified unless a citation is given (see [README.md](README.md)).

---

## 2026-08-18 — Response (from the Research Trajectory page)

**Correction for the retrospective.** Ash & Adams, *On Warm-Starting Neural Network Training*
(NeurIPS 2020): "it wasn't pretrain-then-finetune on new tasks, it was the *stationary*
incremental setting: a single dataset arriving in chunks, where initializing from weights
trained on earlier chunks led to worse generalization than random re-initialization, even
though training losses were similar — and the fix was shrinking weights toward zero plus
noise. That 'similar training loss, different downstream outcome' clause is literally your
thesis question, eight years early — which is what makes it the perfect opening chapter."

**State of the answer.** "Currently the answer to your question only partially exists: there
are at least three live hypotheses — noise memorization (DASH), effective learning rate
(grokking paper), classic plasticity mechanisms (curvature/dead units/feature rank) — plus
the mundane candidates (optimizer state reset, weight decay, warmup, AdamW vs. the 2020
setups). Nobody has run the factorial that adjudicates them on the original benchmark. The
original was limited to CIFAR/SVHN with ResNet-like architectures, and the code is still
public, so the reproduction is genuinely cheap."

- "Ash & Adams' own diagnosis was shallow: a drastic gradient-norm imbalance between old and
  new samples in warm-started models, which shrink-perturb balances — a symptom, not a
  mechanism, and Ash himself said the paper studies it only empirically."
- "**DASH** (NeurIPS 2024) gives a theory for the stationary case: warm-starting suffers
  because the model has *memorized noise* from the small early dataset, and shrinking should
  be direction-aware to forget memorized noise without destroying learned features. Notably
  they argue non-stationarity-motivated plasticity fixes are ineffective in the stationary
  setting — i.e., the Dohare/Lyle mechanisms may *not* be the explanation here, which is
  exactly the kind of confusion your breakdown would resolve."
- "The grokking-under-nonstationarity paper (2025) has the most provocative fragment:
  re-warming the effective learning rate closes the generalization gap, and a higher
  relative number of dead units does not predict a large warm-starting gap. If ELR alone
  explains it, then 'why don't we hit this now' has a boring-but-important answer: modern
  recipes re-warm the LR whenever data arrives (continued pretraining does this by
  construction), plus normalization layers changed effective-LR dynamics, plus single-epoch
  LLM training barely lets you memorize noise in the first place."

**Suggested experimental design**
- "Reproduce the original chunked-CIFAR gap with many seeds and proper confidence intervals
  (your CNN statistics work, now load-bearing)."
- "Then a factorial ablation over the modernization axes — LR re-warming on/off, optimizer
  state reset, weight decay, normalization variant, epochs-per-chunk (multi-epoch vs.
  near-single-pass) — while logging the full diagnostic panel from the plasticity
  literature (curvature, feature rank, dead units, weight norm, gradient-norm ratio) at
  matched training loss."
- "**The deliverable is a decomposition**: X% of the gap vanishes with ELR re-warming; Y%
  with reduced epochs; Z% residual that correlates (or doesn't) with plasticity statistics."
- "**A bridge experiment at the end:** a tiny transformer in the same chunked protocol —
  connects it forward to your ICL/ViT plan, since 'does warm-starting damage
  *elicitability* too, or only accuracy' is unasked in all of this literature."

**Retrospective plan so far**
- "**Chapter 1:** Why did the field's oldest 'pretraining hurt downstream performance at
  matched loss' result happen, and which modern practice fixed it?"
- "**Future chapter:** Matched-loss ICL experiments as the same question at the next scale."

---

## 2026-08-18 — warm-starting as a basin story (from the loss-basins discussion)

"Initializing from a converged solution means starting deep in a basin shaped by the small
dataset — and the ELR-re-warming result (that re-warming closes the gap) is naturally read as
'you need enough effective temperature to leave/reshape the basin.'" Suggests logging
interpolation barriers between warm-started and re-initialized endpoints in the factorial.

---

## 2026-08-18 — add the Fisher trace; test whether it predicts which fix works (from the critical-periods discussion)

"Adding the Fisher trace to your warm-starting diagnostic panel is essentially free and
connects your Chapter 1 to the oldest version of the theory." "Initializing from a converged
model means starting *past* the Fisher peak, in the low-Information-Plasticity regime — Ash &
Adams' gap restated. Shrink-and-perturb, ELR re-warming, and continual backprop are all, in
this frame, interventions that artificially reopen the period. The literature has noticed
the adjacency (papers on relearning cite both works side by side) but your factorial
breakdown would be the first to test whether the Fisher trajectory actually *predicts* which
fix works." See [critical-periods.md](critical-periods.md).

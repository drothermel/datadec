# Critical-period timing study — are the critical period, the Fisher peak, basin commitment, and ICL emergence one event?

**Kind:** staging. Candidate exits: a project doc (small-scale sibling-seed runs with timed
deficits, logging Fisher trace, pairwise interpolation barriers, and ICL curves per
checkpoint); or absorption into landscape geometry (GEO-opt-5 seed-split timing), the
warm-starting decomposition, and the ICL-as-post-training topic. This is the "K1" item of
the original idea map.

**Context (Danielle, 2026-08-18).** Asked how Soatto & Achille's critical-period work
interacts with the plasticity / river-valley / basin threads. See
[../danielle-inputs.md](../danielle-inputs.md); literature in [critical-periods.md](critical-periods.md).

Related-work claims below are unverified unless a citation is given (see [README.md](README.md)).

---

## 2026-08-18 — the testable synthesis (from the Research Trajectory page)

"The critical period, the Fisher-trace peak, basin commitment (onset of linear connectivity
between sibling runs), and the emergence of induction heads / ICL are all claimed to live in
the same early window — but no one has measured them *together* on one set of runs to check
whether they're the same event. Your small-scale tiers are exactly where that's affordable:
run sibling seeds with timed deficits (the Achille protocol), log Fisher trace, pairwise
interpolation barriers, and ICL curves per checkpoint, and ask whether deficit sensitivity
for *elicitability* closes at the same time it closes for accuracy. If it does,
'pretraining shapes post-training beyond final loss' gets a mechanism with a fifty-year
lineage back to the kitten experiments; if it doesn't, you've separated the critical period
for capability from the critical period for elicitability — which would be a genuinely new
distinction."

"Either way, it's a fitting closing loop for the retrospective: the blurred-kitten paper you
started from turns out to contain the earliest version of your plasticity thread, your
featurization thread, and your basin thread all at once."

**Instruments it shares with existing projects:** pairwise interpolation barriers
(landscape geometry GEO-2, GEO-opt-5 seed-split timing); Fisher trace alongside the
plasticity diagnostic panel (warm-starting decomposition); ICL curves and induction-head
strength (ICL-as-post-training protocol statistics 1–2); the many-seed tiny-scale substrate
(tiny-scale measurement).

---

## 2026-08-18 — reproducing Achille et al. directly, then deconstructing it (from the Research Trajectory page)

Question posed (Danielle): if the goal were the same reproduce-then-deconstruct treatment
applied directly to the 2019 critical-periods paper, what would it look like? Reproducing
and deconstructing these two core papers from the current-era perspective seems like a
great way to start, with each continuation thread tying into one or both, and the ultimate
goal being to reconcile them in the modern era.

"The parallel reproduction works, and it's actually cleaner than the warm-starting one in
some ways because the original already comes with a mechanism claim to test."

**The original protocol, restated as a template.** "Train on CIFAR with a *deficit*
(blurred/downsampled images) for a window of epochs, remove it, train to convergence, and
plot final performance as a function of deficit onset and duration — the dose-response
curve that mirrors the animal experiments. Plus the two controls that made the paper:
high-level deficits like vertical flipping leave no permanent damage, and the Fisher trace
tracks the sensitive window. The reproduction is trivially affordable at your scale, and
your CNN statistics framework slots in directly — the original's dose-response curves
were, in classic 2018 style, thin on seeds, so even the pure replication with proper
confidence bands on the *shape* of the sensitivity window is a contribution."

**Deconstruction axes — "whether the critical period is a fact about *networks* or about
*training recipes*."**
- (a) "**Effective-learning-rate artifact** — the sharpest modern challenge. The original
  used fixed decay schedules, so 'the period closed' is confounded with 'the LR dropped.'
  Test: after deficit removal, re-warm the LR (or use a WSD schedule where the deficit ends
  during the stable phase) and see how much of the permanence evaporates — the exact analog
  of the ELR result that closed the warm-starting gap, and the river-valley reading is that
  a still-high LR should let the run climb out and find a different valley."
- (b) "**Basin commitment** — run sibling seeds from shared init, with and without deficit,
  and measure pairwise interpolation barriers over time. The critical-period claim becomes
  geometric: the period closes when the deficit run's basin diverges irreversibly from the
  control's. This turns 'decisive early transient' from metaphor into a measurable event
  time."
- (c) "**The Fisher peak** — the original's own mechanism, now one hypothesis among several
  rather than the conclusion; log it alongside the Dohare/Lyle panel (dead units, feature
  rank, curvature) and see which diagnostic's timeline actually predicts the sensitivity
  window."
- (d) "**Optimizer/normalization modernization** — does the window survive AdamW, warmup,
  BatchNorm→LayerNorm, ViT vs. CNN? Nobody has systematically checked whether critical
  periods are an SGD-era phenomenon."

**The forward bridge — elicitability as an outcome column.** "Run the deficit protocol on
your small transformers and measure ICL curves and induction-head strength alongside
accuracy. The pointed question: is there a critical period for *elicitability* distinct
from the one for performance — i.e., can a deficit window leave final loss fully recovered
while permanently flattening the in-context learning curve? Given that induction-head
formation is a known sharp phase transition early in training, there's a specific,
falsifiable prediction: deficits spanning that transition should damage ICL
disproportionately… a critical period for elicitability is 'pretraining shapes
post-training beyond final loss' with a mechanism and a *timestamp*; no such period would
dissociate capability formation from elicitability formation, which none of the existing
literature can currently distinguish."

The unified design with the warm-starting reproduction is in
[warmstarting-decomposition.md](warmstarting-decomposition.md).

---

## 2026-08-18 — the "money figure" this study produces (from the practical-plan discussion)

"A shared training-time axis on which you plot, for the same architecture and dataset, (a)
the deficit-sensitivity window, (b) the warm-start-damage window, (c) the non-stationarity
memory-effect window, and (d) the panel events — Fisher-trace peak, basin-commitment time
(interpolation barriers between sibling seeds), representation-similarity divergence."
Alignment states the retrospective's thesis; non-alignment "dissociates phenomena the
vocabulary would otherwise merge." The identifiability layer is the same runs read as
"interventions inside the sensitive window change *which solution class* you land in…
interventions after it change only parameters within the class." Full plan in
[warmstarting-decomposition.md](warmstarting-decomposition.md).

---

## 2026-08-18 — the falsifiable core claim and the four-instrument panel (from the CRL-foundations discussion)

"The critical period is an *identifiability phase transition*": before commitment,
interventions select among solution classes (sibling divergence under *aligned* barriers,
high local learning coefficient, failed stitching to controls, permanent damage); after
commitment, interventions perturb within a class (alignment and stitching recover, damage
is transient and retunable away). Panel additions: the local learning coefficient
(degeneracy), model stitching / linear-map residuals (functional identifiability), and
barriers in both raw and permutation-aligned flavors — "whether LLC drops, Fisher peaks, LMC
onset, and deficit-window closure coincide is exactly your money figure, now with four
independently-motivated instruments instead of two." Per-paper sub-claims in
[identifiability-literature.md](identifiability-literature.md).

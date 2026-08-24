# Intervention grid — early-window interventions, permanent damage, and when the solution is chosen

> **Draft scaffolding (2026-08-22).** Promoted from a staging topic. The quoted material in §4
> is external text; §1–§3 are synthesized scaffolding not yet reviewed by Danielle. Treat them as
> provisional until this note is removed.

**Program pillars served:** mechanism (every scar as a non-stationarity source; every fix as a
stabilizer), how (many seeds, one harness, a diagnostic panel), apex (real path-dependence
vs. measurement artifact; identifiability as the formalism). (Program: `README.md` → Program.)

**One-line pitch.** Achille et al.'s critical periods, Ash & Adams' warm-starting gap, and
Igl et al.'s ITER memory effect are the same experiment: apply an intervention schedule to
training, vary onset and duration, measure permanent damage, log a diagnostic panel.
Rothermel et al. 2021 is the control cell — an apparent history effect that vanished under
fair tuning. Build one harness where each paper is a config, reproduce the known answers
with confidence intervals, decompose each gap across modernization axes, and read the
panel as an identifiability phase transition: before commitment interventions select the
solution class; after it they perturb within it.

IDs: GRID-1–GRID-5 (Stage 1), GRID-6–GRID-8 (Stage 2), GRID-opt-1–GRID-opt-5.

**Paper goal.** Main-conference or TMLR-shaped: the unified grid with the alignment figure
(Stage 1 + Stage 2) is the thesis's experimental spine; a workshop-sized first paper is the
warm-start reproduction + factorial decomposition alone.

**Structure.** *Stage 1* — harness, reproductions, factorial decomposition (CNNs, then an LM
diagonal). *Stage 2* — the measurement layer on the same runs: commitment clocks,
identifiability tests, the alignment figure.

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short
continued-training branches or fine-tunes; **T3** = new pretraining runs.

---

## 1. What the project involves

### Stage 1 — harness, reproductions, decomposition (T3 at CIFAR / tiny-LM scale)

1. **One harness (GRID-1).** Interventions defined on the data stream — data poverty
   (chunked data, warm-start), input corruption (blur/downsample deficit), distribution
   drift (Igl-style supervised variants, label noise) — with onset and duration as
   parameters; diagnostic panel defined on any network: curvature, feature rank, dead units,
   weight norm, gradient-norm ratio, Fisher trace. Modality-agnostic by construction.
2. **Known-answer reproductions (GRID-2).** Ash & Adams' chunked-CIFAR ResNet cell and
   Achille's blur-deficit cell, many seeds, confidence bands on the gap magnitude and the
   sensitivity-window shape. Gates everything downstream.
3. **Factorial decomposition (GRID-3).** Modernization axes — LR re-warming on/off,
   optimizer-state reset, weight decay, normalization variant, epochs-per-chunk, AdamW vs.
   SGD-era setups, ViT vs. CNN — at matched training loss. Deliverable: X% of the gap
   vanishes with effective-LR re-warming, Y% with reduced epochs, Z% residual and whether
   it correlates with the panel.
4. **Period-reopening interventions (GRID-4).** Shrink-and-perturb, ELR re-warming,
   continual-backprop resets, and distill-into-fresh-network (ITER), each as an arm; the
   distillation arm with an undamaged-teacher control splits "geometry damage" from
   "knowledge damage."
5. **Optimum displacement (GRID-5).** For each stage, how far the warm-start regime's tuned
   optimum sits from the from-scratch default, knob by knob.

### Stage 2 — the measurement layer (same runs)

6. **Commitment clocks (GRID-6).** Sibling seeds from shared init with and without
   intervention; pairwise interpolation barriers over time, raw and permutation-aligned
   (report the difference); Fisher-trace peak; local learning coefficient; representation
   similarity (CKA as proxy, stitching / linear-map residuals as ground truth).
7. **The alignment figure (GRID-7).** On one training-time axis: the deficit-sensitivity
   window, the warm-start-damage window, the non-stationarity memory-effect window, and the
   panel events. Alignment or dissociation is the result either way.
8. **Identifiability reading (GRID-8).** Test the claim: inside the sensitive window,
   aligned barriers and representation distance to control siblings stay permanently
   elevated and stitching fails (class selection); after it, alignment and stitching
   recover and damage is retunable away (within-class perturbation).

### Optional directions

- **GRID-opt-1: LM diagonal.** The warm-start/data-poverty cell and one deficit cell on
  small transformers, with ICL curves and induction-head strength as LM-only outcome
  columns; pre-registered divergence hypotheses.
- **GRID-opt-2: Critical period for elicitability.** Deficits spanning the induction-head
  transition (cross-listed with ICL-opt-3).
- **GRID-opt-3: Tuning-response curves.** Performance vs. search budget per paradigm as the
  falsifiable replacement for matched-budget comparisons.
- **GRID-opt-4: Re-tuning meta-analysis.** How often has an incumbent's advantage survived
  serious re-tuning (Rothermel 2021; Melis 2018; ELR; the Qwen-RLVR corrections)?
- **GRID-opt-5: MoE arm.** Router saturation as a fourth commitment clock.

---

## 2. Doability and impact

### Overall doability: **high** — the harness exists; the cost is seeds and verification

- A harness is already implemented; code, configs, and first plots compress. What does
  not: GPU hours for seeds (n≈10 per cell) and verification against the known answers.
- Pin replication targets narrowly; budget seed counts from a power target at the start;
  run the complete grid in CNNs first and a chosen diagonal in LMs.
- Prior claims to respect: DASH argues non-stationarity-motivated plasticity fixes are
  ineffective in the stationary setting; the grokking-under-non-stationarity result
  attributes the gap to effective LR. The factorial adjudicates.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| GRID-2 reproductions with CIs | Medium | The originals were thin on seeds; confidence bands on the window shape are a contribution. |
| GRID-3 decomposition | **High** | Adjudicates three live hypotheses on the original benchmark; nobody has. |
| GRID-4 distillation arm | High | Splits geometry damage from knowledge damage — none of the founding papers could. |
| GRID-6/7 alignment figure | **High** | "Three literatures, one sensitive period, with candidate mechanisms timestamped" — or a discovered dissociation. |
| GRID-8 identifiability | High | Turns "damage happens" into "damage = trajectory-induced selection among underdetermined solutions." |
| GRID-opt-1 LM diagonal | High | The LM data-poverty cell is the DataDecide question in miniature. |

---

## 3. Infrastructure build sequence

1. **Intervention schedule layer** on the existing harness (what / onset / duration) and the
   diagnostic panel as standard logging; run manifest with config hashes.
2. **Known-answer reproductions** (GRID-2) as the acceptance test.
3. **Factorial runner** over modernization axes; seed budgeting from a power target.
4. **Sibling-seed + interpolation tooling** (raw and aligned barriers; Git Re-Basin-style
   matching; stitching / linear-map residuals; CKA), Fisher trace, LLC estimator.
5. **Period-reopening arms** including distill-into-fresh-network with the undamaged-teacher
   control.
6. **Alignment-figure assembly** and identifiability tests.
7. *(Optional)* LM diagonal on small transformers with ICL/induction-head outcomes; MoE arm.


---

## 4. External assessments and origin notes

Dated notes from external conversations and the staging topic this doc was promoted from,
recorded for consolidation — not decisions. Related-work claims in quoted text are
unverified unless a citation is given.

### Origin notes — moved from `topics/staging/warmstarting-decomposition.md`

## 2026-08-18 — Response (from the Research Trajectory page)

**Correction for the thesis framing.** Ash & Adams, *On Warm-Starting Neural Network Training*
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

**Thesis chapter plan (as first proposed)**
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
fix works." See [critical-periods.md](../topics/reference/critical-periods.md).

---

## 2026-08-18 — the unification: one early-window intervention grid (from the Research Trajectory page)

"**Ash & Adams and Achille et al. are *the same experiment*.** Both apply an early-window
distribution intervention and measure permanent damage: warm-starting's 'train on 50% of
the data first' is just a data-poverty deficit, and blur is a data-corruption deficit. So
rather than two reproductions, you can build one intervention grid — intervene(*what*:
blur, subset, label noise, mixture shift; *when*: onset; *how long*: duration) ×
recipe(*LR schedule, optimizer, architecture*) × measure(*final accuracy, diagnostic
panel, basin divergence*) — in which each classic paper is a single cell. Every result in
both papers becomes a special case, and the modern fixes (shrink-perturb, ELR re-warming,
continual-backprop resets) become 'period-reopening interventions' whose success or failure
per cell is itself mechanism evidence. That grid *is* the thesis's experimental
spine, and it's honest to your history: you're not retrofitting a story, you're showing the
two papers that bracketed your entry into the field were probing one phenomenon from two
sides."

**Sequencing.** "Do the warm-start reproduction first anyway. It's the smaller grid, it
debugs the shared infrastructure (seed management, diagnostic logging,
interpolation-barrier tooling), and its factorial results tell you which axes are live
before you commit the critical-period grid, which is the bigger design." The critical-period
half (reproduction template, deconstruction axes, elicitability column) is in
[critical-period-timing-study.md](intervention-grid.md).

---

## 2026-08-18 — the third founding cell, and a distill-into-fresh-network arm (from the Research Trajectory page)

- ITER (Igl et al., ICLR 2021) gives the intervention grid "its third founding cell": RL's
  transient policy non-stationarity permanently scarring the representation, alongside
  Achille's deficits and Ash & Adams' data poverty.
- "It hands your experiment grid its sharpest diagnostic arm. Across the deficit/warm-start
  factorial, add a distill-into-fresh-network condition alongside the weight-perturbation
  fixes. The readout: for each intervention and each outcome (accuracy, ICL curves, the
  diagnostic panel), does distillation erase the damage? Where it does, the scar was
  trajectory-borne and basin-bound — and your interpolation-barrier measurements should show
  the student in a different, cleaner basin. Where it doesn't — where even a fresh student
  inherits the impairment through the teacher's *outputs* — the deficit corrupted the
  function itself, not just the weights. That single contrast cleanly splits every
  permanence result in the grid into 'geometry damage' vs. 'knowledge damage,' which none
  of the three founding papers could distinguish."
- Control: "distillation quality is a confound (an imperfect student underperforms for
  boring reasons), so the ITER arm needs the control the original paper used — distill an
  *undamaged* teacher too, and measure damage relative to that baseline rather than to the
  teacher."

---

## 2026-08-18 — the fourth founding cell: the control (from the Research Trajectory page)

"Achille, Ash & Adams, and Igl each claimed a real, permanent training-history effect. Your
paper (*Don't Sweep your Learning Rate under the Rug*, arXiv 2107.12460) is the founding
example of the opposite outcome: an apparent history effect (frozen pretrained structure
'sufficing') that vanished under fair measurement. Those are precisely the two hypotheses
every cell of your proposed grid must distinguish — real scar vs. measurement artifact —
and the thesis now has a founding paper for *each side of the dichotomy*, one of
which you wrote." Deconstruction axis (a) — "is the period just the LR schedule closing?" —
"is structurally identical" to the 2021 move.

---

## 2026-08-18 — the grid as the smallest instance of "re-tune the regime-mismatched knobs" (from the research-hypothesis discussion)

"The warm-start factorial becomes the oldest, smallest instance of the second half [of the
hypothesis]: re-tune the regime-mismatched knobs and watch how much 'impossibility'
evaporates." Supporting measurement to add: *optimum displacement* — "how far the
warm-start regime's tuned optimum sits from the from-scratch default in hyperparameter
space," knob by knob. See [../research-hypothesis.md](../research-hypothesis.md).

---

## 2026-08-18 — the practical plan: one harness, one figure, six months (from the Research Trajectory page)

Question posed (Danielle): replicate the three foundational papers (Achille et al.; Ash &
Adams; Igl et al.) and break each into the grid to form a shared vocabulary and conclusion
space; extend toward identifiability (the CRL direction); run both CNN vision models as in
the originals and small LMs on language tasks, in parallel, to see where they diverge and to
keep flexibility for scaling. How might it look under the goal of convincing results
rapidly? (The response also contained adviser-management advice; disregarded at Danielle's
direction and not recorded here.)

"The idea is right — this is the correct foundation, and the CNN+LM parallelism is what
gives it legs. But as specified it's a 3-papers × full-factorial × 2-modalities object…
the unification itself, taken seriously, is what compresses the work."

**Build one harness, not three replications.** "All three papers are the same experiment:
apply an *intervention schedule* to training (data poverty for Ash & Adams, input
corruption for Achille, distribution drift for Igl), vary onset/duration, measure permanent
damage, log a diagnostic panel. So the deliverable is a single framework where each paper
is a config file… Conveniently, Igl's repo already contains supervised CIFAR variants
of their claim (label noise, dataset-size interventions) — so you can drop the RL machinery
entirely and keep all three phenomena in the supervised setting, same backbone, same data.
That's a 3× cost reduction with no loss of the vocabulary claim." (Unverified: that Igl's
repo contains supervised CIFAR variants.)

**Design backwards from the one figure.** "A shared training-time axis on which you plot,
for the same architecture and dataset, (a) the deficit-sensitivity window, (b) the
warm-start-damage window, (c) the non-stationarity memory-effect window, and (d) the panel
events — Fisher-trace peak, basin-commitment time (interpolation barriers between sibling
seeds), representation-similarity divergence. If those windows and events align, that
single figure states the thesis claim: three literatures, one sensitive
period, with candidate mechanisms timestamped. If they *don't* align, that's arguably
better — it dissociates phenomena the vocabulary would otherwise merge, and it's a
discovery."

**The identifiability extension costs analysis, not compute.** "The CRL tie-in should be a
measurement layer on the *same runs*, not new experiments: sibling seeds from shared init,
pairwise interpolation barriers (raw and permutation-aligned), and representation
similarity over time, interpreted as 'when does the solution become determined.' The
operational claim: interventions inside the sensitive window change *which solution class*
you land in (barriers and representation distance to control siblings stay permanently
elevated); interventions after it change only parameters within the class (alignment
recovers them)… Save the formal CRL machinery for the writeup's framing; don't let it
generate experiments yet."

**Stagger the modalities; don't parallelize them.** "Make every grid cell modality-agnostic
by construction (interventions defined on the data stream, panel defined on any network),
run the *complete* grid in CNNs where training is nearly free, then run a chosen
*diagonal* in small LMs — I'd pick the warm-start/data-poverty cell and one deficit cell,
with ICL curves added as the LM-only outcome column. And pre-register the divergence
hypotheses before the LM runs… Note the LM data-poverty cell is *literally your DataDecide
question* at miniature scale."

**Two flags.** "Pin the replication targets narrowly (Ash & Adams' chunked-CIFAR ResNet cell
and Achille's blur-deficit cell — not their full papers)… and budget the statistics up front
— decide seed counts per cell from a power target at month zero rather than discovering
mid-grid that the interesting effects need n=15."

---

## 2026-08-18 — with the harness already implemented: what compresses and what doesn't (from the Research Trajectory page)

Context (Danielle): the harness is already implemented, and with her agentic coding setup
an initial version of the replications plus a few variations with preliminary analysis is
plausible in ~2.5 weeks of focus. (The response's adviser-related remarks are disregarded
at her direction.)

- "What compresses: code, configs, plumbing, first plots. What doesn't: wall-clock GPU time
  for seeds (the CIs are the product, and n=10 seeds per cell takes the same hours
  regardless of how fast the code was written), and *verification*."
- The known-answer replication is "the *acceptance test that gates everything*": the
  harness must reproduce Ash & Adams' published gap magnitude and Achille's
  sensitivity-window shape before any variation runs.
- "Spend the speed dividend on *seeds and the panel, not on variations*… the money figure
  needs depth: many seeds per cell so the window boundaries and panel-event timestamps have
  error bars, because 'the Fisher peak aligns with basin commitment' is only a claim if both
  events have confidence intervals. A tight grid with n=10 beats a sprawling grid with
  n=3."
- Define "done" per milestone in writing: "replication matches published effect within CI;
  figure has error bands; predictions graded."

### Origin notes — moved from `topics/staging/critical-period-timing-study.md`

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

"Either way, it's a fitting closing loop for the thesis: the blurred-kitten paper you
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
[warmstarting-decomposition.md](intervention-grid.md).

---

## 2026-08-18 — the "money figure" this study produces (from the practical-plan discussion)

"A shared training-time axis on which you plot, for the same architecture and dataset, (a)
the deficit-sensitivity window, (b) the warm-start-damage window, (c) the non-stationarity
memory-effect window, and (d) the panel events — Fisher-trace peak, basin-commitment time
(interpolation barriers between sibling seeds), representation-similarity divergence."
Alignment states the thesis claim; non-alignment "dissociates phenomena the
vocabulary would otherwise merge." The identifiability layer is the same runs read as
"interventions inside the sensitive window change *which solution class* you land in…
interventions after it change only parameters within the class." Full plan in
[warmstarting-decomposition.md](intervention-grid.md).

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
[identifiability-literature.md](../topics/reference/identifiability-literature.md).

---

## 2026-08-18 — a fourth commitment clock (from the MoE discussion)

"Router-saturation timestamps join Fisher trace, LLC, and LMC onset as a fourth commitment
clock in your money figure — with the advantage of being the cheapest to compute and the
only one that's exactly zero/one per token." Requires an MoE arm of the grid; see
`moe-movement.md`.
## 5. Related work and positioning

*Purpose: the paper-facing synthesis — the prior-art landscape, this project's
position in it, and what each closest neighbor lacks. Unlike §4 (a dated intake
log, which grows by appending new entries **above this section**), §5 is a
current-state statement: rewrite it as understanding changes. Positioning claims
are Danielle's to make; agent-supplied literature claims anywhere in this document
are unverified leads, not established facts.*

**Status: not yet drafted.** Raw material: the dated entries in §4, the theme
accumulators under `../topics/reference/` (index: `../topics/README.md`), and
`../litreview/citation-verification-ledger.md` (citation provenance; nothing there
is verified).

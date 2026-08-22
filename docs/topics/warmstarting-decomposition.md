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
per cell is itself mechanism evidence. That grid *is* the retrospective's experimental
spine, and it's honest to your history: you're not retrofitting a story, you're showing the
two papers that bracketed your entry into the field were probing one phenomenon from two
sides."

**Sequencing.** "Do the warm-start reproduction first anyway. It's the smaller grid, it
debugs the shared infrastructure (seed management, diagnostic logging,
interpolation-barrier tooling), and its factorial results tell you which axes are live
before you commit the critical-period grid, which is the bigger design." The critical-period
half (reproduction template, deconstruction axes, elicitability column) is in
[critical-period-timing-study.md](critical-period-timing-study.md).

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
and your retrospective now has a founding paper for *each side of the dichotomy*, one of
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
rapidly, given an adviser who wants evidence of independent-researcher ability rather than
small publishable bundles?

"The idea is right — this is the correct foundation, and the CNN+LM parallelism is what
gives it legs. But as specified it's a 3-papers × full-factorial × 2-modalities object,
which is the scope-explosion reflex again… the unification itself, taken seriously, is what
compresses the work."

**Build one harness, not three replications.** "All three papers are the same experiment:
apply an *intervention schedule* to training (data poverty for Ash & Adams, input
corruption for Achille, distribution drift for Igl), vary onset/duration, measure permanent
damage, log a diagnostic panel. So the deliverable is a single framework where each paper
is a config file… three replications demonstrate diligence; one framework that *subsumes*
three literatures demonstrates exactly the taste and abstraction ability your adviser is
asking for evidence of. Conveniently, Igl's repo already contains supervised CIFAR variants
of their claim (label noise, dataset-size interventions) — so you can drop the RL machinery
entirely and keep all three phenomena in the supervised setting, same backbone, same data.
That's a 3× cost reduction with no loss of the vocabulary claim." (Unverified: that Igl's
repo contains supervised CIFAR variants.)

**Design backwards from the one figure.** "A shared training-time axis on which you plot,
for the same architecture and dataset, (a) the deficit-sensitivity window, (b) the
warm-start-damage window, (c) the non-stationarity memory-effect window, and (d) the panel
events — Fisher-trace peak, basin-commitment time (interpolation barriers between sibling
seeds), representation-similarity divergence. If those windows and events align, that
single figure states the thesis of the retrospective: three literatures, one sensitive
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

**Six-month structure (pre-committed falsifiable checkpoints).** "Month 2 — harness
validated by replicating the warm-start gap with CIs (a known answer, so it's a pure
execution test); month 4 — the alignment figure, with your written predictions from month 1
graded against it; month 6 — the LM diagonal and the identifiability layer, plus the
long-form manuscript that unifies it (TMLR-shaped…)."

**Two flags.** "Pin the replication targets narrowly (Ash & Adams' chunked-CIFAR ResNet cell
and Achille's blur-deficit cell — not their full papers)… and budget the statistics up front
— decide seed counts per cell from a power target at month zero rather than discovering
mid-grid that the interesting effects need n=15."

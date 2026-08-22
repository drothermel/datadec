# DataDecide-dense (+WSD) — a small, heavily instrumented, many-seed retrain substrate

**Kind:** staging. Candidate exit: a standalone project doc (resource + validation
substrate) or absorption as the shared pilot of `../../potential-projs/wsd-suite.md`,
`tiny-scale-measurement.md`, and `recipe-featurization.md` (order-effect arm). Gate: a
design doc (recipes, scales, seeds, cadence, logging spec, pilot sequence) and a decision
to train at all.

Source: a 2026-08-22 conversation on the data layer (record in
`../reference/datadecide-data-pipeline.md`). Danielle's statements: the smallest sizes
have 4–10 released checkpoints, "conveniently also the scales that it would be very doable
to retrain some examples"; and "if I was going to do DataDecide-dense I'd want to also do
WSD … the value of having the smallscale wsd becomes much higher than the cost." She judged
the idea "much more plausible" after the conversation (post-processing, 2026-08-22).

---

## Why three projects point at it

1. **Temporal resolution.** Released 4M–8M runs have 5–10 checkpoints
   (`../../open-questions-answered.md`, 2026-08-21 spacing table); per-run drift/diffusion
   fits and early-curve prediction are infeasible there. Forward-pass reconstruction
   changes *what* is measured per checkpoint; only retraining restores density.
2. **Order effects.** REC's stratified-vs-sequential reruns were already planned at these
   scales; make sampling strategy an arm.
3. **Tiny-scale measurement.** TINY needs a powered substrate with many seeds; the
   released 3 seeds are a pooled compromise.

Plus: ground-truth validation of the LR derivations and the MPL; the training-config
parity check that WSD/MSUITE list as step 1, measured as a reproduction-gap result
(faithful rerun vs. published checkpoint, relative to seed variance); and with WSD arms,
cosine twins with matched data order (WSD-opt-3) done small first.

## Design constraints gathered so far

- **Grid:** a few recipes spanning the outcome range plus one within-family pair × the
  2–4 smallest scales × 10+ seeds; schedule ∈ {cosine reproduction, WSD} ×
  sampling strategy ∈ {sequential, stratified}.
- **Logged, everything the released suite lacks:** true training loss; the LR schedule
  as executed; the realized data-order manifest; per-token held-out losses on a frozen
  probe set; dense checkpoints (log-spaced early, uniform late).
- **Pilot-first sequence:** cosine reproduction of one recipe (parity gate) → its WSD
  twin → 2–3 branch points with a small decay length/shape sweep on that run → freeze the
  spec → fan out. Do not skip the pilot because reruns are cheap; it protects suite
  homogeneity.
- **Tuning parity:** WSD's stable-phase LR is not "reuse the cosine peak"; a small LR
  sweep at one scale, a stated transfer rule, sensitivity reported.
- **Branch data consumption:** pin whether decay branches consume the continuation of the
  parent's data stream (keeps twins exactly matched) or fresh/replayed data.
- **Schema:** emit the results store's existing schema (`variant` field) so it slots into
  existing accessors.
- **Scope line at scale, not schedule:** 150M+ stable-phase runs or many more recipes is
  the separate WSD resource-paper decision.

- **Regularization recipe (added 2026-08-22):** at the smallest scales the fixed corpus
  will be seen for multiple epochs; fix the regularization recipe (dropout and its
  onset, weight decay, any expert dropout / z-loss for MoE arms) in the frozen spec and
  record epochs per run, citing Xue et al. 2305.13230 and Muennighoff et al. 2305.16264
  (`../reference/regularization-literature.md`).

## Hypotheses only this substrate can test

- Annealed readouts improve measurement SNR most at small scales, where wall oscillation
  is proportionally largest ("WSD + branches is what makes 10M-scale experimentation
  measurable"). Unverified reasoning.
- Data-order sensitivity differs between stable-phase and decaying-LR training (the
  schedule × sampling interaction).

## Intake notes

- Cost claims from the response ("branches add ~10% per branch point"; small-scale GPU
  delta "noise") are unverified.
- Not started; sequenced after the pure-T0 work (IRT matrix builder, 2PL fit) per the
  conversation, with the design doc written first.
- Design option to record when the doc is written (2026-08-22): parametrize the retrain
  with u-µP (`../reference/parametrization-and-hp-transfer.md`) to get width transfer and
  a ~9-run independent HP sweep per recipe, at the cost of departing from DataDecide's
  own per-size hand-set hyperparameters — a comparability trade-off, undecided.


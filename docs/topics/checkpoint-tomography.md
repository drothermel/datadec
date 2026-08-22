# Checkpoint tomography — short-branch probes as a landscape measurement suite

**Status:** topic (staging). Candidate exits: a flagship "instrument paper" framing that wraps
annealed readouts, landscape geometry, token-level movement, and functional featurization; or
absorption of individual probes into those projects' §4s.

**Question posed (Danielle, 2026-08-21).** Something in direct relation to river-valley / loss
landscape / same-vs-different-basin ideas: take an intermediate checkpoint, continue
pretraining for ~1/16 of the run length or less, then measure a statistic that points at
movement. Does that exist? See [../danielle-inputs.md](../danielle-inputs.md).

Related-work claims below are unverified (see [README.md](README.md)).

---

## 2026-08-21 — Response

"Yes — this exact paradigm ('cheap continuation from a checkpoint, then a statistic that
localizes you in the landscape') exists, but as three or four separate instruments in
separate literatures that nobody has unified or applied systematically along training ×
across data."

**1. The decay branch as a wall-height meter.** "This is the most direct hit, and it's in the
river-valley paper itself. Wen et al. verify their theory by branching off a constant-LR run
at 20B tokens, decaying for 5B tokens, and then interpolating between checkpoints:
stable-phase pairs show a valley cross-section (bouncing between walls), decay-phase pairs
show smooth monotone curves — and their WSD-S variant explicitly builds a training procedure
out of resuming from decayed checkpoints. The Hägele et al. / MiniCPM annealing-branch
methodology is the same instrument used for scaling-law fitting. So 'branch + decay + measure
the loss drop' is established; the drop is your height-above-river statistic. What's not
established: doing it on cosine mid-run checkpoints ([annealed readouts'] open question), and
treating the per-token profile of the drop as the statistic rather than the scalar."

**2. The twin-branch instability probe as a basin-commitment meter.** "Frankle et al.'s
linear-mode-connectivity work is the canonical version: spawn two children from the same
checkpoint with different SGD noise/data order, train both, measure the interpolation
barrier between them. Early checkpoints produce disconnected children (high barrier — the run
hasn't committed to a basin); past a critical point, children stay linearly connected.
Barrier-between-siblings is a 'have we committed yet' statistic, and the checkpoint time at
which it collapses is a commitment clock. Caveats: the original trains children to completion
(short-child variants exist but aren't standardized), it's mostly pre-LLM-scale vision work,
and nobody has run it across data recipes. [Landscape geometry's seed-split timing] is the
free observational cousin; the branch version makes it causal and controllable."

**3. Stochastic-sampling probes as degeneracy meters.** "The singular-learning-theory /
devinterp line (Timaeus) estimates the local learning coefficient by running short SGLD
chains around a checkpoint — literally brief noisy continued training whose stationary
behavior measures local landscape degeneracy/effective dimensionality. They've tracked it
across Pythia-style checkpoint sequences and shown it detects developmental transitions.
This is far cheaper than 1/16 of a run and is the closest thing to an off-the-shelf 'point at
movement' statistic, though it measures local geometry rather than valley position."

**4. Non-continuation geometry statistics** (cheap complements). "Recent work measures a
scalable critical-sharpness statistic across public pretraining and mid-training
checkpoints, showing progressive sharpening at scale, and even applies it to data-mixing
decisions; and the basin-emergence line shows LLMs become progressively more resilient to
random parameter perturbations as scale grows, creating expansive stability regions, with
pretraining forming a basic-capability basin and fine-tuning forming specific-capability
basins inside it. These are single-checkpoint probes — useful as covariates, but they don't
involve training movement."

**What doesn't exist — the battery.** "Nobody has assembled these into a standardized probe
battery run at many points along training, with data as the treatment. Concretely, at
checkpoint t, spend ≤1/16 budget on four branches and get four statistics:"

- "decay branch → wall height (total + per-token profile);"
- "hot branch (constant LR continuation) → diffusion width, the wall-oscillation amplitude,
  from the variance of the continued trajectory;"
- "twin branches (two short children, different data order) → sibling barrier = basin
  commitment;"
- "data-shifted branch (inject component c) → responsiveness to c, which is exactly the
  U_c(t) probe [see functional-featurization.md]."

"That's checkpoint tomography: a four-number (or four-curve) landscape signature per
checkpoint, causal by construction, whose cost you control. Each individual probe has
precedent — that's a feature, not a problem, since the methods are pre-validated — but the
battery, its calibration (how short can the branches be before the statistics degrade — the
1/16 question is itself an experiment, and Frankle-style probes plausibly need much less),
and its application across DataDecide recipes are all unclaimed."

**How it composes with the portfolio.** "The statistics are the outcome variables the
featurization program was missing ('which corpus properties predict wall height / commitment
time / component responsiveness as functions of t'), the decay branch is [annealed
readouts'] runner, the twin probe is [landscape geometry] made causal, the injection probe is
the U_c(t) study, and the per-token profiles are [token-level movement]."

**Prior-art check to do.** "The devinterp group and the WSD-followup literature are the two
communities most likely to have partial versions in flight, so a targeted pass over their
2025–26 output is worth an afternoon. But based on what's visible, 'short-branch probes as a
landscape measurement suite for pretraining checkpoints, with data recipes as the independent
variable' is open — and it's a better flagship framing than any single project in the
original documents, because it's an instrument paper whose validation experiments are
themselves the results."

---

## 2026-08-22 — candidate fifth probe

A reset branch (reset an interface or block, continue briefly, measure recovery cost and the
barrier to the pre-reset model) has the same shape as the four probes above and shares the
runner. Staged in [reset-response-stage-probe.md](reset-response-stage-probe.md).


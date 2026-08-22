# Post-training experiment design — alternatives to the sweep-and-seeds approach

**Kind:** staging. Candidate exits: one or more project docs — a post-training power-analysis /
proxy-metric paper (T0 reanalysis + modest runs), a "did SFT move the model in distribution
space?" reanalysis of the earlier negative result, a within-reach-task post-training study at
tiny scale (overlaps tiny-scale measurement TINY-opt-2), or a late-window cross-family
intervention study. Or absorption into the tiny-scale, annealed-readouts, and token-movement
§4s.

**Context (Danielle, 2026-08-18).** Skeptical of returning to the post-training project
because iteration is slow; real differences need more seeds; everything is model-specific, so
debugging a small-scale issue needs a large sweep; limiting to existing clean pretraining
sweeps makes family effects near-impossible to test; and the DataDecide models are tiny with
the "no movement during SFT" issue. But not wanting to abandon the direction. See
[../danielle-inputs.md](../danielle-inputs.md).

Related-work claims below are unverified unless a citation is given (see [README.md](README.md)).

---

## 2026-08-18 — Response (from the Research Trajectory page)

**The tension.** "The noise floor of your measurements scales with the number of seeds you
can afford, while the generality of any finding scales with the number of model families you
test — and both multiply against slow iteration. These also interact: a clean single-family
result might just be another family artifact, so even a successful sweep has uncertain
external validity."

**Consider a different question.** "'Why can't anyone measure whether X improves Y at
affordable scale?' The measurement-and-proxy angle is both the least blocked and, right now,
probably the most needed."

**Make the proxy metric the contribution, not the sweep.** "DataDecide's insight —
continuous likelihood beats accuracy at small scale — hasn't really been transplanted to
post-training. A continuous, low-variance predictor of 'RL-ability' measured on the base
model (NLL on gold reasoning traces, pass@k at large k, entropy at decision points, even
plasticity-style statistics like curvature or feature rank) would let *everyone else* escape
the seed problem. Validating a proxy needs far fewer runs than detecting an intervention
effect, because you're fitting a correlation across existing variation rather than powering
a comparison."

**Turn the blocker into the object of study.** "The Sober Look paper did variance analysis
for evaluation; nobody has done the equivalent power analysis for post-training
*experiments* — how many seeds does a claimed RLVR delta actually require at 150M vs 1B, and
how much of the published small-scale literature clears that bar? That's mostly re-analysis
of public results plus a modest number of your own runs, and it directly legitimizes your
negative result."

**Fully synthetic testbeds.** "The Echo Chamber and graph-pathfinding style of work (*Provable
Benefits of RLVR over SFT for Reasoning Models: Learning to Backtrack Efficiently*) shows you
can study pretraining-conditioned post-training mechanistically in settings where a seed
costs minutes. Findings are less directly transferable, but they're *causal*, and the field
currently has correlational LLM results and toy theory with little in between."

**Interrogate whether "no movement" was real or a metric artifact** ("the one I'd push
hardest"). "'SFT did nothing' almost certainly means benchmark accuracy didn't move — but did
the model move in distribution space? NLL on held-out reasoning traces, KL from the base
model, calibration, sample diversity, pass@k at very large k are all continuous and much
lower-variance than accuracy. Two possibilities, both publishable: either the models
genuinely don't move even in likelihood space (a stronger, stranger negative result than
'accuracy flat'), or they *do* move and pretraining recipes differ in *how much* — in which
case your original question is answerable at DataDecide scale after all, with the accuracy
threshold reframed as the thing that was hiding it. This is the same trick DataDecide itself
used to make MMLU predictable at 150M, just applied one stage later."

**Lower the task instead of raising the model.** "The 'no movement' is a property of the
model–task pair, not the model. TinyZero-style results show RL visibly works at 0.5–3B on
countdown and simple arithmetic; you can design verifiable tasks whose difficulty sits just
above the base models' zero-shot ability. Then recipe effects on post-training become
measurable at sweepable scale. The open question this creates — do recipe effects on
within-reach tasks predict recipe effects on out-of-reach tasks at larger scale? — is itself
a gap nobody has addressed, and it only needs a couple of larger validation runs, not a
factorial sweep."

**Get family diversity from the last window, not from scratch.** "You can't pretrain five
families, but you can take OLMo, Pythia, SmolLM, Llama, and Qwen checkpoints and apply
controlled *late-window* continued pretraining — same intervention, same tokens, different
lineages. The Final Window paper's claim (*Similar Models Learn Differently*) is that this
window disproportionately shapes post-training behavior, which if true means most of the
family-effect question is testable at annealing cost rather than pretraining cost. If the
claim is false at your scales, that's also a finding."

**The asymmetric design ties these together.**
- "Full sweep with seeds only where it's cheap (small models, continuous metrics, easy
  tasks)."
- "Then spend the expensive budget on two or three confirmation runs testing a *ranking* the
  cheap tier predicted — a much lower-powered, therefore affordable, test than estimating
  effect sizes."

---

## 2026-08-18 — a gradient-free variant

Treating in-context learning as the post-training stage makes "seeds" one forward pass and
sidesteps the elicitation threshold; the proxy candidate is the ICL curve (loss on the k-th
demo vs. k) on existing checkpoints. Recorded in [icl-as-posttraining.md](icl-as-posttraining.md).

---

## 2026-08-18 — tuning-response curves, demonstration hygiene, and a meta-analysis (from the research-hypothesis discussion)

- Replace matched-budget comparisons with *tuning-response curves*: "performance as a
  function of search budget for each paradigm… a mature, communally-exhausted paradigm
  should show a flat curve… an under-explored paradigm with real headroom should show a
  steep, still-rising curve."
- Demonstration hygiene for existence proofs: "pre-specified settings, effect sizes with
  confidence bounds across seeds, replication in at least a second model family, honest
  reporting of how many settings were searched, and… a mechanism readout from your
  diagnostic panel explaining *why* the ceiling was exceeded there."
- A publishable piece on its own: "a modest meta-analysis of 'how often does the
  incumbent's advantage survive serious re-tuning'" over the field's natural experiments.
See [../research-hypothesis.md](../research-hypothesis.md).

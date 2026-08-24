# Movement microscope — measure post-training movement before testing whether it happened

> **Draft scaffolding (2026-08-22).** Promoted from a staging topic. The quoted material in §4
> is external text; §1–§3 are synthesized scaffolding not yet reviewed by Danielle. Treat them as
> provisional until this note is removed.

**Program pillars served:** how (sensitivity as the object of study; noise floors and
dose-response calibration), apex (movement net of elicitation; the earlier negative result
re-measured), mechanism (where movement lives by layer and token). (Program: `README.md` →
Program.)

**One-line pitch.** Instead of asking whether a post-training intervention worked, build
the instrument that shows what movement looks like in small models at all: a null
distribution of movement (including continued pretraining on the same data), guaranteed-
effect interventions to calibrate each metric's sensitivity, a decomposition of movement by
layer, token bucket, and direction, and only then the recipe question — recipe-dependent
movement profiles at matched final loss, below the elicitation threshold.

IDs: MIC-1–MIC-4 (the four stages), MIC-opt-1–MIC-opt-5.

**Paper goal.** Workshop-sized from Stages 1–3 alone (a noise-floor atlas, a metric
dose-response benchmark, a movement decomposition); main-conference with Stage 4 across the
25 recipes.

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short
continued-training branches or fine-tunes; **T3** = new pretraining runs.

---

## 1. What the project involves

### Core experiment

1. **Noise floor (MIC-1, T2-light).** One DataDecide recipe; the null distribution of
   movement under different seeds, data orderings, trivially perturbed hyperparameters,
   and continued pretraining on the same pretraining data for the same token budget. Every
   candidate metric evaluated against it: per-token KL from the base model, likelihood
   margins on task data, benchmark accuracy, per-layer representation drift (CKA /
   linear-map residuals), weight-space statistics (norm and effective rank of ΔW per
   layer). Movement that doesn't exceed seed-noise-plus-token-exposure isn't movement.
2. **Calibrated sensitivity (MIC-2).** Guaranteed-effect interventions — memorize a narrow
   distribution; distill from a much larger teacher (KL-to-teacher as a ground-truth
   movement axis); within-reach synthetic tasks — giving each metric a dose-response curve
   and a detection limit per scale.
3. **Decomposition (MIC-3).** Where movement lives when it happens: which layers, which
   tokens (per-token KL sliced by reference-model entropy buckets), which direction
   (projection of ΔW / activation shifts onto: toward the fine-tuning distribution, toward
   the teacher, orthogonal).
4. **Recipe movement profiles (MIC-4).** Post-train all 25 recipes identically; compare
   movement profiles — amount, layers, token classes — at matched final loss.

### Optional directions

- **MIC-opt-1: "Did the earlier SFT move the model?"** Re-measure the earlier
  post-training-did-nothing result in distribution space (NLL on held-out traces, KL from
  base, calibration, sample diversity, pass@k at large k).
- **MIC-opt-2: Elicitation-controlled readout.** Report each movement measure raw and net of
  the tuned elicitation ceiling (joint with ICL elicitability).
- **MIC-opt-3: Post-training power analysis.** How many seeds does a claimed RLVR/SFT delta
  require at 150M vs. 1B, and how much of the small-scale literature clears the bar —
  reanalysis of public results plus modest runs.
- *Coordination note (2026-08-22):* `elicitation-gain.md` (`ELI`) measures pre/post
  extractability (ΔS, stability, iterations-to-threshold) with a fixed outer optimizer —
  the elicitation-controlled readout MIC-opt-2 wants, on the same SFT checkpoints if they
  exist.
- **MIC-opt-4: MoE variant.** Routing flips as the categorical movement channel.
- **MIC-opt-5: Sequential vs. direct distillation.** Distil one 150M-class student from a
  base teacher and then post-train it, versus distilling the same student from the
  teacher's post-trained sibling; read both with the microscope. The six-question
  distillation review found no published controlled comparison; it is the apex question
  with a teacher in the loop. Default objective reverse KL / skew KL with an explicit LM
  term; include a from-scratch control at matched token budget. Hook in
  `../topics/reference/distillation-literature.md`.

---

## 2. Doability and impact

### Overall doability: **high** — inference plus tiny fine-tunes on one GPU

- Stage 1–3 are publishable regardless of Stage 4's outcome; nulls become points on a
  detection-limit curve.
- Risks: reference-model choice for entropy buckets (ablate); distillation quality as a
  confound in MIC-2 (control with an undamaged/no-op distillation); the 150M capability
  floor (use within-reach tasks).
- Relation to token-level movement: same instruments (per-token KL, bucket slicing) applied
  to post-training deltas rather than adjacent pretraining checkpoints; share the held-out
  token set and reference scorer.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| MIC-1 noise-floor atlas | Medium-high | "The single most valuable artifact nobody builds." |
| MIC-2 dose-response benchmark | Medium-high | Turns "did nothing" into "nothing down to detection limit X on instrument Y." |
| MIC-3 decomposition | **High** | The token-bucket slice connects movement to the landscape story; pure inference. |
| MIC-4 recipe profiles | **High if positive** | The original thesis demonstrated below the elicitation threshold. |
| MIC-opt-3 power analysis | Medium-high | Legitimizes the earlier negative result; publishable alone. |
| MIC-opt-5 sequential vs. direct distillation | High, if the microscope is built | Unclaimed comparison; two small distillation runs; reuses the Stage-1/2 readouts. |

---

## 3. Infrastructure build sequence

1. **Post-training harness** for 150M-class checkpoints (SFT / distillation / synthetic-task
   RL), deterministic data order, seed control.
2. **Held-out token set + reference-model scorer** (shared spec with token-level movement).
3. **Movement metrics**: per-token KL vs. base, likelihood margins, per-layer CKA /
   linear-map residuals, ΔW norm and effective rank.
4. **Null-distribution runs** (MIC-1) and **guaranteed-effect runs** (MIC-2).
5. **Decomposition analyses** (MIC-3); then the 25-recipe campaign (MIC-4).


---

## 4. External assessments and origin notes

Dated notes from external conversations and the staging topic this doc was promoted from,
recorded for consolidation — not decisions. Related-work claims in quoted text are
unverified unless a citation is given.

### 2026-08-22 — distillation as the post-training arm: what the literature settles and what it leaves open

From Danielle's SciSpace review of LLM distillation (record in
`../topics/reference/distillation-literature.md`). Three things for the harness: (1) the
objective question is settled enough to fix a default — reverse KL (MiniLLM) or a
skew/JSD variant for the KL-to-teacher readout, with an explicit CE/LM term; (2) the
review confirms there is no published controlled comparison of "distil from the base
teacher, then post-train the student" vs. "distil from the post-trained teacher" — the
MIC-opt design that runs both on the same student and reads movement is unclaimed
territory, and it is the apex question in miniature; (3) the evidence that KD beats
scratch only in data-limited regimes (BERT-scale) means the distillation-quality confound
listed in §2 needs a from-scratch control at the same token budget, not only a no-op
distillation. Missing canon to cite before any of this is written: distillation scaling
laws (2502.08606), on-policy GKD (2306.13649), DistiLLM (2402.03898).

### Origin notes — moved from `topics/staging/movement-microscope.md`

## 2026-08-18 — Response (from the Research Trajectory page)

"The unpressured version of this project inverts the usual order: instead of asking 'did
our intervention work?' (a hypothesis test), ask 'what does movement even look like in
these models?' (an instrument-building question)."

**Stage 1: Measure the noise floor before measuring anything else.** "Take one DataDecide
recipe, and produce the *null distribution* of movement — same model post-trained with
different seeds, different data orderings, trivially perturbed hyperparameters, and,
crucially, *continued pretraining on the same pretraining data* for the same token budget
as the post-training would use. Every candidate metric gets evaluated against this:
per-token KL from the base model, likelihood margins on task data, benchmark accuracy,
representation drift per layer (CKA / linear-map residuals), weight-space statistics (norm
and effective rank of ΔW per layer). Movement that doesn't exceed
seed-noise-plus-token-exposure isn't movement… the unpressured researcher finds out which
metrics have floors low enough to see *anything* at 150M."

**Stage 2: Use interventions with guaranteed effects to calibrate sensitivity.** "Fine-tune
on a narrow distribution until it's memorized (maximal movement, trivially verifiable);
distill from a much larger teacher (movement toward a known target — and the KL-to-teacher
gives you a ground-truth movement axis); train on within-reach synthetic tasks
(TinyZero-style countdown, formatting tasks, style transfer — things 150M models
demonstrably learn). This gives each metric a dose-response curve: 'per-token KL detects
1k-example SFT at 20σ; MMLU detects nothing until 8B parameters.' Now 'our post-training
did nothing' becomes a calibrated statement — nothing *down to detection limit X on
instrument Y*."

**Stage 3: Decompose movement by location and by token.** "Which layers (representation
drift profile — small-model SFT plausibly moves only late layers, which would itself explain
benchmark inertness), which tokens (per-token KL sliced by the determinism/entropy buckets
— does SFT at small scale only move the high-entropy 'hillside' tokens, echoing the RLVR
forking-token result?), and which direction (project ΔW or activation shifts onto
interpretable axes: toward the fine-tuning distribution, toward the teacher, orthogonal to
both). The token-bucket slice is the one I'd bet on being interesting: it connects movement
measurement to the landscape story, and it's pure inference over checkpoints you already
have."

**Stage 4: Only now, the recipe question.** "Post-train all 25 recipes identically and
compare *movement profiles* — not outcomes. Even if no recipe's accuracy budges, recipes
may differ in how much distributional movement the same SFT produces, in which layers, on
which token classes. Recipe-dependent movement profiles at matched final loss would be your
original thesis, demonstrated below the elicitation threshold — the regime everyone else
abandoned as unmeasurable."

**Why the unpressured design is also the faster one.** "The pressured version of this
project treats null results as failures and races toward the setting where effects appear;
the unpressured version treats sensitivity itself as the object of study, so every null is
a data point on a detection-limit curve… Stages 1–3 produce publishable artifacts (a
noise-floor atlas, a metric dose-response benchmark, a movement decomposition) regardless
of what Stage 4 finds — whereas the pressured design's value is hostage to one hypothesis.
And the whole thing is inference-plus-tiny-fine-tunes on models that fit on one GPU."

### Origin notes — moved from `topics/staging/posttraining-experiment-design.md` (the "did the model move" and power-analysis items)

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
demo vs. k) on existing checkpoints. Recorded in [icl-as-posttraining.md](icl-elicitability.md).

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

---

## 2026-08-18 — the instrument-first ordering

The movement-microscope design ([movement-microscope.md](movement-microscope.md)) is the
fully worked version of "interrogate whether 'no movement' was real or a metric artifact":
noise floor (including a continued-pretraining control) → guaranteed-effect calibration
(memorization, distillation with KL-to-teacher as a ground-truth axis, within-reach tasks)
→ decomposition by layer, token bucket, and direction → recipe movement profiles.

### Origin — Danielle's first-hand account (added 2026-08-22)

The original observation, in her words, is recorded verbatim in
`../topics/reference/pretraining-to-posttraining.md` (the "first-hand account" entry): SFT
on Tulu / Tulu 3 over DataDecide models produced no movement on any task from multiple
choice through HumanEval; an AI2 contact said this is more common than expected and that a
dataset built to move specific-task metrics by fine-tuning was in progress (co-author
"Kyle"). Whether that dataset was released is an open item to resolve by asking the contact
— an external search guessed FollowIR, which is a retrieval benchmark and not a match.
## 5. Related work and positioning

*Purpose: the paper-facing synthesis — the prior-art landscape, this project's
position in it, and what each closest neighbor lacks. Unlike §4 (a dated intake
log, which grows by appending new entries **above this section**), §5 is a
current-state statement: rewrite it as understanding changes. Positioning claims
are Danielle's to make; agent-supplied literature claims anywhere in this document
are unverified leads, not established facts.*

**Status: raw material assembled from repository records (2026-08-24); curated cut
2026-08-24; positioning not yet written.**

**The load-bearing items and the role each plays:**

- **DataDecide** (Magnusson et al., Ai2, ICML 2025; arXiv 2504.11393) — the substrate
  (25 recipes, ≤1B, 3 seeds) and the source of the "continuous metrics beat accuracy at
  small scale" move MIC applies one stage later.
- **Similar Models Learn Differently: Final-Window Pretraining Shapes Post-Training
  Beyond SFT** (arXiv 2607.25063) — recorded as *closest to MIC-4's exact experimental
  design*: identical post-training, divergence traced to late-pretraining data.
- **Echo Chamber** (Zhao, Meterez et al., COLM 2025; arXiv 2504.07912) — the controlled-
  mixture-then-RL precedent, and the argument that small controlled proxies yield real RL
  insight (MIC's scale premise).
- **Understanding Reasoning from Pretraining to Post-Training** (arXiv 2607.16097) — the
  contrast case: pretraining loss carries strong predictive signal, in tension with the
  matched-final-loss framing of MIC-4.
- **A Sober Look at Progress in Language Model Reasoning** (Hochlehnert et al., COLM
  2025; arXiv 2504.07086) — the variance-analysis precedent MIC-opt-3 proposes the
  post-training-*experiment* power-analysis counterpart of.
- **Signal and Noise** (Heineman et al., NeurIPS 2025 per the record) — the noise-floor
  vocabulary MIC-1 extends to post-training; paired with the repo's "eval variance lives
  in training, not in re-evaluating a fixed checkpoint" note, which bounds what MIC-1 can
  measure.
- **Spurious Rewards** (Shao et al., arXiv 2506.10947) and **Yue et al.** (arXiv
  2504.13837) / **Wu & Choi** (ICML 2025 AI-for-Math) — the elicitation-in-disguise and
  support-preserving-reweighting cases MIC-opt-2's elicitation-controlled readout must
  exclude; counterpoints on record (2507.14843, 2506.14245).
- **TinyZero** and **Provable Benefits of RLVR over SFT (learning to backtrack)** — the
  within-reach-task and fast-synthetic-testbed precedents for MIC-2's guaranteed-effect
  interventions.
- **The distillation arm** — reverse KL (MiniLLM) / skew-JSD as the fixed default with an
  explicit LM term, **Bui et al.** (2404.19319) forcing a from-scratch control at matched
  token budget, and the review's finding that **no controlled sequential-vs-direct
  comparison exists** (MIC-opt-5's premise). All from a SciSpace six-question review:
  agent-generated, identifiers unverified; missing canon flagged (2502.08606, 2306.13649,
  2402.03898).
- **Plasticity diagnostic panel** — Lyle (2303.01486, 2402.18762), Dohare (Nature 2024 /
  2306.13812), Achille–Rovere–Soatto Fisher trace, and *Can Scale Save Us* (2606.24752)
  — the metric lineage MIC's movement panel draws on.
- **Unresolved provenance:** the FollowIR guess (2403.15246) is on record as *not* the
  AI2 dataset behind MIC-opt-1, and the "Olmo-3.1-32B-Instruct" citation is flagged as
  looking fabricated.

Full enumeration with sources: `related-work/movement-microscope.md`. Main accumulators:
`../topics/reference/pretraining-to-posttraining.md` (primary),
`distillation-literature.md`, `evaluation-methodology-literature.md`, `plasticity.md`,
`critical-periods.md`; ledger rows tagged `MIC` in
`../litreview/citation-verification-ledger.md`; open item in `../open-questions-answered.md`.

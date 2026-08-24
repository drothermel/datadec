# ICL elicitability — in-context learning as the calibrated post-training stage

> **Draft scaffolding (2026-08-22).** Promoted from a staging topic. The quoted material in §4
> is external text; §1–§3 are synthesized scaffolding not yet reviewed by Danielle. Treat them as
> provisional until this note is removed.

**Program pillars served:** how (elicitation as a calibrated instrument and strong null), apex
(the capability-vs-accessibility decomposition; "how much of what the body carries can the
interface reach"). (Program: `README.md` → Program.)

**One-line pitch.** Treat in-context learning as a gradient-free post-training stage: an ICL
"training run" costs one forward pass, so seeds (prompt orderings, samples) are cheap, and
the outcome is a continuous per-token curve with no benchmark thresholds. Measure ICL
curves on existing checkpoints at matched loss across recipes, validate the protocol's
statistics, and use the tuned elicitation ceiling as the null that any weight-update claim
must beat.

IDs: ICL-1–ICL-5, ICL-opt-1–ICL-opt-7.

**Paper goal.** Workshop-sized first paper from ICL-1–ICL-3 on DataDecide checkpoints
(recipe differences in ICL-ability at matched loss, either way); a main-conference version
adds the two-tier vision/LM design and the protocol-validation study.

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short
continued-training branches or fine-tunes; **T3** = new pretraining runs.

---

## 1. What the project involves

### Core experiment

1. **ICL curves on existing checkpoints (ICL-1, T1).** For recipes × sizes × checkpoints:
   per-token loss on the k-th in-context demonstration as a function of k, on held-out task
   families; averaged over prompt orderings and samples. Report slope and asymptote with
   confidence bands.
2. **Matched-loss recipe comparison (ICL-2).** Pair checkpoints across recipes at equal
   pretraining loss *and* at equal tokens (both controls), and compare ICL curves. Report
   comparisons conditional on interpolation-barrier height between the paired checkpoints.
3. **Protocol validation (ICL-3).** Statistics in order of robustness: in-context loss
   curves; induction-head strength (prefix-matching / copying scores) per checkpoint;
   task-recognition vs. task-learning decomposition (shuffled-label and format-only
   controls); task-vector geometry (norm, direction stability across orderings,
   transferability); ICL–GD similarity only with the untrained-model control. Decide how
   many orderings/samples a stable estimate needs.
4. **Elicitation ceiling (ICL-4).** For each model–task, the tuned elicitation ceiling
   (prompt/format/demo search under a reported budget) as the strong null model; report
   capability estimates both raw and elicitation-controlled, with the difference as the
   accessibility delta.
5. **Transfer test (ICL-5).** Does ICL slope/asymptote at 150M–1B predict post-training
   movement at larger scale (the proxy claim)?

### Optional directions

- **ICL-opt-1: Two-tier design.** Tiny ViT / sequence-transformer ICL tier (Omniglot/CIFAR
  style) with many seeds and matched-loss checkpoint pairs to define the protocol; one
  DataDecide confirmation tier.
- **ICL-opt-2: Code-autoencoder probe.** Round-trip reconstruction fidelity through a
  natural-language bottleneck as a graded capability probe.
- **ICL-opt-3: Critical period for elicitability.** Deficit windows spanning the
  induction-head transition should damage ICL disproportionately; run on the intervention
  grid's small transformers.
- **ICL-opt-4: Function vs. trajectory.** Does an ICL-flattened model's distilled student
  recover its ICL?
- **ICL-opt-5: How much of the body can a frozen interface reach?** Reframe the
  frozen/finetuned gap (Rothermel et al. 2021) as an elicitation-ceiling measurement with
  modern probes (cross-listed from the frozen-body audit topic, gap G6).
- **ICL-opt-6: Learned task vectors** instead of extracted ones for the geometry statistic;
  tasks chosen within the rank limitation.
- *Coordination note (2026-08-22):* `elicitation-gain.md` (`ELI`) is the optimizer-driven
  counterpart of the tuned elicitation ceiling: a fixed outer model edits the interface under
  budget; its ΔS across DataDecide sizes is the same null this doc wants, obtained by search
  rather than by hand-tuned prompts.
- **ICL-opt-7: Repetition as step size — the (unique examples × repetitions) factorial.**
  Treat the number of *unique* demonstrations as the data axis and how often each is
  repeated in the prompt as the step-size axis; at matched total context, compare splits
  of (unique × repetitions) on the per-token ICL curve. If repetition shifts the curve the
  way a larger LR shifts a training curve (faster early movement, different plateau or
  instability), ICL has a separable step-size axis; if it only adds tokens, it does not.
  Same design separates "more examples help" from "longer context hurts." Candidate ICL
  x-axes to log alongside: cumulative demonstration tokens, query-loss trajectory,
  attention mass on demonstrations vs. query, task-vector norm growth, prompt-order
  entropy. (Danielle-flagged seed, 2026-08-22; origin in
  `../topics/reference/icl-literature.md`.)

---

## 2. Doability and impact

### Overall doability: **high** for the core (inference only), **medium** for validation

- Inputs exist (DataDecide checkpoints; held-out task families to build). Each cell is
  forward passes; the cost driver is orderings × samples × checkpoints.
- Risks: ICL at 150M may be weak on natural tasks (use synthetic/structured task families
  with an ability-matched ladder); the ICL-mechanism assumption (induction heads) must be
  stated; matched-loss pairs need both controls and basin conditioning.
- Either-way outcomes: recipe differences at matched loss support the apex claim; none is
  a deflation at this scale, still reportable.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| ICL-1 + ICL-2 | **High** | "Pretraining shapes the *learner*, not just the *snapshot* — demonstrated without a single gradient update." |
| ICL-3 validation | Medium-high | The measurement theory every later elicitation-controlled claim needs. |
| ICL-4 ceiling | **High (apex)** | Turns elicitation from confound into control condition. |
| ICL-5 transfer | High if positive | The proxy metric the field is missing. |
| ICL-opt-1 two-tier | Medium-high | Full statistical rigor at the cheap tier; a transfer negative is also publishable. |
| ICL-opt-3 / opt-4 | High, conditional | Connect to the intervention grid's critical-period and distillation arms. |
| ICL-opt-5 | Medium-high | Directly downstream of Danielle's 2021 paper. |

---

## 3. Infrastructure build sequence

1. **Task-family bank** for ICL (structured/synthetic families with difficulty ladders;
   held-out splits), versioned.
2. **ICL-curve runner**: per-token loss over demonstration position for any checkpoint;
   orderings and sampling controls; cached per (checkpoint, family, ordering seed).
3. **Protocol statistics**: induction-head scores; task-recognition/learning controls;
   task-vector extraction/learning; ICL–GD similarity with the untrained control.
4. **Matched-loss pairing utility** (equal-loss and equal-token pairs; barrier logging).
5. **Elicitation-ceiling search** with budget accounting.
6. *(Optional)* Tiny-transformer ICL tier on the intervention-grid harness; code-autoencoder
   probe; distillation arm hookup.


---

## 4. External assessments and origin notes

Dated notes from external conversations and the staging topic this doc was promoted from,
recorded for consolidation — not decisions. Related-work claims in quoted text are
unverified unless a citation is given.

### Origin notes — moved from `topics/staging/icl-as-posttraining.md`

## 2026-08-18 — Response (from the Research Trajectory page)

**ICL as the post-training stage — "this directly attacks your two blockers."**
- "Iteration: an ICL 'training run' costs one forward pass, so seeds become cheap — you can
  average over prompt orderings and samples the way you never could over SFT runs."
- "Elicitation threshold: there's a real line of work arguing ICL and fine-tuning are two
  access routes to the same latent capabilities — e.g., a formal argument that capabilities
  acquired through SFT can be approximated by the base model via in-context learning without
  parameter updates (*Eliciting Fine-Tuned Transformer Capabilities via Inference-Time
  Techniques*), and empirical comparisons of when each paradigm wins: prompt-based methods
  excel in few-shot scenarios but their effectiveness plateaus as more data becomes
  available (*Fine-Tuned In-Context Learners for Efficient Adaptation*, Bornschein, Lyle,
  Pascanu et al. — the plasticity crowd literally moved into 'ICL vs fine-tuning as
  adaptation' territory)."

**The unvalidated proxy: ICL-ability as a cheap predictor of finetunability across
pretraining recipes.**
- "Take your existing DataDecide-style checkpoints."
- "Measure in-context learning curves (loss on the k-th demo as a function of k — a
  continuous, smooth, per-token metric, no benchmark thresholds)."
- "Ask whether recipe differences show up there even where SFT accuracy was flat."
- "If ICL slope at 150M–1B predicts post-training movement at larger scale, you've built
  exactly the proxy metric the field is missing, using the models you already have."
- "And if models differ in ICL-ability while being identical in final loss, that's your
  original thesis — pretraining shapes the *learner*, not just the *snapshot* — demonstrated
  without a single gradient update."

**The code autoencoder (an earlier project of Danielle's).**
- "Mechanically, prompt optimization *is* elicitation pushed to its ceiling — you're
  measuring what a frozen model can be made to do, which is the same quantity the ICL idea
  probes, just optimized rather than sampled."
- "Your reconstruction loss is another instance of your recurring move: a continuous, cheap,
  differentiable-ish signal (round-trip fidelity through the natural-language bottleneck)
  standing in for a fuzzy capability ('does the model *understand* this code')."
- "**How compressible code is into natural language *for a given model pair* is a property
  of their shared representations**, so reconstruction fidelity could itself serve as a
  capability probe — one that's graded rather than thresholded, unlike pass@1."

**Link between the two.** "The autoencoder's encoder/decoder prompts are 'trained' parameters
living in text space, so you're doing gradient-free adaptation where a 'seed' is a sampling
temperature draw. Your whole trajectory — loss curves, proxy metrics, elicitation — keeps
circling one question: *what cheap continuous observable reveals latent capability?*"

---

## 2026-08-18 — Combining vision and language: a two-tier design (from the Research Trajectory page)

**Question posed (Danielle).** Earlier work looked at plasticity in CNNs and at which results
could and could not get confidence bounds. Is there a way to keep both small-scale vision
experiments and larger-scale LLM experiments, combining the two spaces — possibly with vision
transformers instead of CNNs for clean comparisons?

**"Pretraining choices shape elicitability, holding final performance constant — this
doesn't require language at all.** ICL is the perfect 'post-training' stand-in because it's
gradient-free, continuous (per-token loss on the k-th in-context example), and cheap enough
to run with the many seeds your CNN experience taught you that you need for confidence
bounds." Prior art (Chan et al. 2022; Raventós et al.) in `../topics/reference/icl-literature.md`: "that
literature is your hypothesis, already demonstrated in miniature — but it's framed as 'when
does ICL emerge,' not as 'ICL-ability as a measurable functional of pretraining recipe that
predicts adaptation at larger scale.' That reframing is your gap."

**Proposed sequence**
1. "**Tiny-transformer vision/synthetic tier (weeks, full statistical rigor).** Adapt your
   CNN pipeline to small ViTs or sequence-transformers on Omniglot/CIFAR-style ICL tasks à la
   Chan. Vary pretraining recipe (data distribution properties, ordering, mixture), train
   many seeds, and — critically — construct **matched-loss checkpoint pairs**: select
   checkpoints across recipes at equal pretraining loss, then measure ICL curves. This is
   the clean version of your dead project: same question, but the outcome variable moves,
   and you can afford the seeds."
2. "**Define the invariant measurement, not the invariant model.** The bridge between tiers
   is a protocol: ICL curve slope/asymptote at matched loss, with a power analysis (your old
   confidence-bound work is directly reusable here — it becomes methodology, not a detour).
   The deliverable of tier 1 is 'here is a low-variance elicitability metric and here's how
   many seeds it needs.'"
3. "**DataDecide tier (one replication, not a sweep).** Reproduce your pipeline on a handful
   of DataDecide recipes and measure the *same* protocol — ICL loss curves on held-out tasks,
   matched-loss comparisons. You're not powering a new discovery here; you're testing
   whether the mapping found in tier 1 transfers. A confirmation test needs far less compute
   than an exploration sweep."

**Two cautions**
- "ViT-ICL and LLM-ICL plausibly share mechanism (induction-head-like circuits) but that's
  an assumption of the design — worth stating as such, and honestly, finding that the
  mapping *doesn't* transfer would also be a publishable answer to 'can small vision
  experiments inform LLM decisions.'"
- "Matched-loss pairs have a hidden confound: equal loss at different token counts vs. equal
  tokens at different loss are different controls, and you'll want both, since 'recipe A
  reaches this loss faster' and 'recipe A has better ICL at this loss' are separable
  claims."

---

## 2026-08-18 — bridge from the warm-starting decomposition (intervention grid)

The warm-starting decomposition's final experiment — "a tiny transformer in the same chunked
protocol" — asks whether "warm-starting damage[s] *elicitability* too, or only accuracy,"
which "is unasked in all of this literature." Matched-loss ICL is proposed as the next
chapter of the same question. See [warmstarting-decomposition.md](intervention-grid.md).

---

## 2026-08-18 — Measurement protocol: statistics in rough order of robustness (from the Research Trajectory page)

1. "*In-context loss curves* — per-token loss as a function of context position / number of
   demonstrations. Olsson et al.'s original 'ICL score' was literally the loss difference
   between an early and late token position. Smooth, continuous, seed-cheap — the direct
   descendant of your loss-curve-features thesis."
2. "*Induction-head strength* (prefix-matching/copying scores on synthetic sequences) — the
   mechanistic correlate of ICL emergence, measurable per-checkpoint, so you can watch
   elicitability *develop* along the pretraining trajectory."
3. "*Task-recognition vs. task-learning decomposition* — shuffled-label and format-only
   controls separate 'the demos told the model which task' from 'the model learned the
   mapping.' Matters because pretraining recipes plausibly affect these two components
   differently."
4. "*Task-vector geometry*" — norm (how much learning happened), direction stability across
   demo orderings (cheap variance), transferability across prompts (generalization), and
   whether differently-pretrained base models produce differently-structured task vectors at
   matched loss.
5. "*ICL–GD similarity*, last, and only with the adversarial controls" (the untrained-model
   control that the founding papers skipped).

Paper references for each in [icl-literature.md](../topics/reference/icl-literature.md).

---

## 2026-08-22 — Danielle-flagged seed: ICL learning curves and their x-axes (origin of ICL-opt-7)

> I previously found it very surprising, the comparison between fine-tuning and in-context
> learning … the idea that by consuming more tokens, a model is moving towards a,
> quote-unquote, more trained state, kind of, is a parallel to fine-tuning … is that
> something that is also investigated in the prompt-tuning space in terms of … how many
> examples … or how different choices of prompts impact that, like, quote-unquote learning
> curve.

> I see the number of examples in context learning as being more similar to either a
> compute metric or a number of tokens metric … one [analogy] might be … how often you
> repeat examples in your prompt, in that like if you considered n to be the number of
> unique examples, and then your step size or your learning rate or whatever was like how
> often each of the examples was repeated, then you're arguing maybe you're taking a bigger
> step on each example.

Recorded as ICL-opt-7. The surrounding conversation's responses were content-free; the
paper leads it produced (many-shot ICL scaling, *In-Context Learning with Long-Context
Models*, NAACL 2025) are in the ICL literature topic with reliability flags.

## 2026-08-22 — refinements to protocol statistic #4 (task-vector geometry), from Danielle's citations

- Extracted ICL task vectors fail on high-rank mappings (Dong et al., arXiv 2506.09048) —
  choose ICL tasks whose mapping rank is within reach, or inject multiple vectors.
- Learned task vectors (Yang et al., ICLR 2026) are more accurate and position/layer
  flexible than extracted ones, and come with a mechanistic account (OV circuits, key
  heads, linear propagation) — a candidate for the "task-vector geometry" measurement that
  is less extraction-method-dependent. See `../topics/reference/task-vectors.md`.

---

## 2026-08-18 — report comparisons conditional on basin membership (from the loss-basins discussion)

Danielle-flagged seed on the loss-basins toggle, applied to this project: "log the pairwise
interpolation barrier (with and without alignment) between every pair of checkpoints you
compare, and report your elicitability comparisons *conditional on* barrier height. If
recipe effects on ICL-ability only hold within low-barrier pairs, that's a finding; if they
hold across basins, that's a stronger one." Also: "mechanism-level metrics (task vectors,
GD-similarity scores) may not be comparable across basins at all" (Juneja et al., ICLR
2023).

---

## 2026-08-18 — a falsifiable prediction: a critical period for elicitability (from the critical-periods discussion)

"Is there a critical period for *elicitability* distinct from the one for performance —
i.e., can a deficit window leave final loss fully recovered while permanently flattening
the in-context learning curve? Given that induction-head formation is a known sharp phase
transition early in training, there's a specific, falsifiable prediction: deficits spanning
that transition should damage ICL disproportionately." Protocol statistics 1–2 (ICL loss
curves, induction-head strength) are the outcome columns. See
[critical-period-timing-study.md](intervention-grid.md).

---

## 2026-08-18 — does elicitability live in the function or the trajectory? (from the ITER discussion)

"Whether an ICL-flattened model's distilled student recovers its ICL is exactly the test of
whether elicitability lives in the function or in the trajectory." The distill-into-fresh-
network arm is specified in [warmstarting-decomposition.md](intervention-grid.md).

---

## 2026-08-18 — the project as "quantifying the gap your 2021 paper discovered"

"ICL-as-posttraining is, delightfully, *you finally switching sides*: betting that
gradient-free elicitation is measurable enough to be the instrument… your planned
ICL-elicitability protocol is best framed as *quantifying the gap your 2021 paper discovered*:
how much of what the body carries can the interface reach?" (Rothermel et al. 2021, arXiv
2107.12460: frozen variants lag full fine-tuning; transfer through the body is real.)

---

## 2026-08-18 — reframed as instrument calibration and the strong null (from the research-hypothesis discussion)

"The ICL/elicitation protocol isn't an outcome measure anymore — it's *instrument
calibration*… so that the later experiments can report 'capability change net of
elicitation change.'" The tuned elicitation ceiling becomes "the *strong null model*, and
every demonstration [that a weight update exceeds it] is a one-sided test against the
strongest available null." Both readouts — raw and elicitation-controlled — are reported,
with their difference as the capability-vs-accessibility decomposition. See
[../research-hypothesis.md](../research-hypothesis.md).
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

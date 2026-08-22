# ICL as the post-training stage — gradient-free elicitation probes across recipes

**Kind:** staging. Candidate exits: a project doc ("ICL-ability as a cheap predictor of
finetunability across pretraining recipes": in-context learning curves on existing DataDecide
checkpoints, validated against post-training movement at larger scale), possibly joined with
the code-autoencoder reconstruction-fidelity probe; or absorption into tiny-scale measurement
(proxy metrics) and the post-training experiment-design topic.

**Danielle-flagged project seeds** (the `→` notes on the Notion toggle; these mark what she
considers especially relevant to defining a project):

1. "ICL-ability as a cheap predictor of finetunability across pretraining recipes."
2. "How compressible code is into natural language *for a given model pair* is a property of
   their shared representations, so reconstruction fidelity could itself serve as a
   capability probe, one that's graded rather than thresholded, unlike pass@1."

**Question posed (Danielle, 2026-08-18).** Could in-context learning be treated as the
post-training stage, and features extracted from it or from elicitation? See
[../danielle-inputs.md](../danielle-inputs.md).

Related-work claims below are unverified unless a citation is given (see [README.md](README.md)).

---

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
bounds." Prior art (Chan et al. 2022; Raventós et al.) in `icl-literature.md`: "that
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

## 2026-08-18 — bridge from the warm-starting retrospective

The warm-starting decomposition's final experiment — "a tiny transformer in the same chunked
protocol" — asks whether "warm-starting damage[s] *elicitability* too, or only accuracy,"
which "is unasked in all of this literature." Matched-loss ICL is proposed as the next
chapter of the same question. See [warmstarting-decomposition.md](warmstarting-decomposition.md).

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

Paper references for each in [icl-literature.md](icl-literature.md).

---

## 2026-08-22 — refinements to protocol statistic #4 (task-vector geometry), from Danielle's citations

- Extracted ICL task vectors fail on high-rank mappings (Dong et al., arXiv 2506.09048) —
  choose ICL tasks whose mapping rank is within reach, or inject multiple vectors.
- Learned task vectors (Yang et al., ICLR 2026) are more accurate and position/layer
  flexible than extracted ones, and come with a mechanistic account (OV circuits, key
  heads, linear propagation) — a candidate for the "task-vector geometry" measurement that
  is less extraction-method-dependent. See `task-vectors.md`.

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
[critical-period-timing-study.md](critical-period-timing-study.md).

---

## 2026-08-18 — does elicitability live in the function or the trajectory? (from the ITER discussion)

"Whether an ICL-flattened model's distilled student recovers its ICL is exactly the test of
whether elicitability lives in the function or in the trajectory." The distill-into-fresh-
network arm is specified in [warmstarting-decomposition.md](warmstarting-decomposition.md).

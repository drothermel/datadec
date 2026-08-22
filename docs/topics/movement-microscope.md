# Movement microscope — measure post-training movement before testing whether it happened

**Kind:** staging. Candidate exits: a project doc (four-stage instrument-building study on
DataDecide checkpoints: noise floor → calibrated sensitivity → decomposition → recipe
movement profiles); or absorption into token-level movement (as its post-training arm) and
the post-training experiment-design topic. This is the idea map's P1–P4.

**Question posed (Danielle, 2026-08-18).** For identifying movement of the pretrained
DataDecide models at small scale (beyond the proxy metrics they already identified), how
would a researcher with no external pressure explore the space? See
[../danielle-inputs.md](../danielle-inputs.md).

Related-work claims below are unverified unless a citation is given (see [README.md](README.md)).

---

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

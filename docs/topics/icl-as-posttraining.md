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

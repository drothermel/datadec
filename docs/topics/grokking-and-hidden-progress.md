# Grokking, double descent, and hidden progress — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: the thesis premise is "final loss is an insufficient statistic for
latent capability"; grokking is its maximal demonstration, and double descent is the
warning that capability is not even monotone in loss. Both bear on every matched-loss
comparison and on the validity regime of loss-curve forecasting.

---

## 2026-08-18 — the existence proofs the premise needed (from the Research Trajectory page)

**Grokking**
- Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*
  — "train loss is at floor and test loss is flat at chance for thousands of steps while the
  model is… *gradually assembling the generalizing circuit underneath*, invisible to both
  losses, until it snaps into place."
- Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability* — showed
  the mechanism; "the field needed non-loss observables (circuit-formation metrics,
  weight-norm trajectories, Fourier-structure probes) to see progress that loss curves
  hide, which is precisely the role your ICL curves, Fisher traces, and decay-branch
  responses play at your scales."
- "A danger in your matched-loss design: two checkpoints matched on both train *and* test
  loss can sit at radically different points of hidden circuit maturity — pre-grok and
  mid-grok — and will respond completely differently to further training, fine-tuning, or
  elicitation. So grokking says matched-loss pairs are a *necessary but provably
  insufficient* control, which is an argument *for* your diagnostic panel rather than
  against your design: the panel exists to catch exactly this."

**Double descent**
- Nakkiran et al., *Deep Double Descent: Where Bigger Models and More Data Hurt* —
  "epoch-wise double descent means capability isn't even monotone in training loss along a
  single run, which is a boundary condition on the whole prediction-law thread — the
  multi-power law and loss-to-accuracy links assume away non-monotonicity that demonstrably
  occurs in certain regimes, and knowing *which* regimes is part of your 'when are proxy
  metrics valid' question."

**Bridges to the rest of the program**
- *What Can Grokking Teach Us About Learning Under Non-Stationarity* — "the warm-starting
  gap and grokking are being analyzed as the same delayed-generalization phenomenon, with
  effective learning rate as the shared control knob."
- River-valley physics: "work showing plateaus at higher learning rates can
  counterintuitively accelerate final convergence during decay by optimally initializing
  the slow river modes — grokking plateaus reinterpreted as travel along the river that
  loss can't see, which unifies 'hidden progress' with your token-bucket/decay-branch
  machinery: the decay branch is a probe that *reveals* accumulated-but-hidden river
  progress, i.e., an operationalized anti-grokking instrument." (Specific paper not named;
  unverified.)

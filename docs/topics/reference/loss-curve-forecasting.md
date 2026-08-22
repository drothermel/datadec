# Loss-curve forecasting and loss→accuracy mapping — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: the multi-power law is the analytic annealing correction in annealed
readouts (ANN-5/ANN-2) and a regression target in recipe featurization (REC-7); the
loss→downstream-accuracy mapping is what the decision-flip analysis and the IRT
emergence-as-measurement claim both depend on.

---

## 2026-08-18 — the LLM loss-curve-prediction thread (from the Research Trajectory page)

**Papers**

- Kairong Luo et al., *A Multi-Power Law for Loss Curve Prediction Across Learning Rate
  Schedules* (arXiv 2503.12811, ICLR 2025). "Predicts the full pretraining loss curve at every
  intermediate step across LR schedules, using a power law on the sum of learning rates plus
  extra power-law terms for the decay-induced loss drop; fitted on a few runs, it
  extrapolates to unseen schedules and even discovers a schedule beating cosine (resembling
  WSD)."
- Yangyi Chen et al., *Scaling Laws for Predicting Downstream Performance in LLMs* (arXiv
  2410.08527). "The two-stage 'FLP' pipeline: FLOPs → pretraining loss → downstream
  performance."
- Samir Yitzhak Gadre et al. 2024, *Language models scale reliably with over-training and on
  downstream tasks*. "Downstream accuracy is predicted as an exponential function of training
  loss."
- Akshita Bhagia et al., *Establishing Task Scaling Laws via Compute-Efficient Model Ladders*.
  "Maps compute → task NLL → accuracy."

**Thoughts**

- "In the scaling-law literature, the latent quantity is final loss or benchmark accuracy,
  predicted from early/partial loss curves or small-model runs." Methodological flavor:
  "phenomenological — fit a parametric form, extrapolate, and mostly stay agnostic about
  mechanism." Regime: "a single stationary distribution where the only 'non-stationarity' is
  the LR schedule."
- "The pretraining answer [to 'what low-dimensional summary of training dynamics forecasts a
  capability'] is 'a surprisingly simple functional of the LR schedule' (multi-power law)
  plus a sigmoid/exponential link to accuracy — with the caveat that hard accuracy metrics
  can look emergent, showing no progress above chance until the loss crosses a threshold,
  which is where the loss-to-accuracy mapping gets fragile."
- "The multi-power law's decay term is essentially modeling how the optimizer's response to
  the schedule shapes the curve — a dynamics question the plasticity people would recognize."

---

## 2026-08-18 — double descent as a boundary condition (from the grokking discussion)

"Epoch-wise double descent (Nakkiran et al.) means capability isn't even monotone in
training loss along a single run, which is a boundary condition on the whole prediction-law
thread — the multi-power law and loss-to-accuracy links assume away non-monotonicity that
demonstrably occurs in certain regimes, and knowing *which* regimes is part of your 'when
are proxy metrics valid' question." See `grokking-and-hidden-progress.md`.

## Undated (~2025) — "Do more linear loss curves indicate better training?" — answer 1 of 2 (condensed)

Danielle's question (the motivating question of the CIFAR-10 loss-slope study, EDP's
lineage — `../../potential-projs/early-dynamics-prediction.md` §4):

> Is there a belief in machine learning research that "more linear loss curves indicate
> better training"? If so, what is it based on?

**Answer 1 (Perplexity-style, blog/StackExchange-sourced).** Asserts "a widely held belief
… that smoother, more linear loss curves generally indicate better training quality and
stability," then argues entirely about **smoothness**, not linearity:

- Smooth, gradually decreasing curves are preferred to jagged/oscillating ones; a "good fit"
  is train and validation loss decreasing to a stable point with a small gap.
- Gradient *flow* is monotone, so oscillation in practice signals discrete step sizes
  overshooting → a preference for configurations that descend smoothly.
- Smooth curves are read as: stable dynamics, well-tuned hyperparameters, clean data,
  effective optimization. Jagged curves as: LR too high, small batches (gradient variance),
  data problems (outliers, NaNs, bad shuffling), model instability.
- One paper-shaped citation: an ECCV 2022 paper on enforcing smoothness in *learned
  optimizers* (ecva.net 136830533; unverified), offered as evidence that "smooth
  optimization behavior is … functionally superior."
- Practical guidelines: lower LR, larger batch, EMA smoothing for visualization, early
  stopping on smooth convergence.
- Nuances: cyclic LR schedules produce intentional oscillation; some oscillation "might
  indicate healthy exploration"; validation curves are noisier than training curves.

Intake note: the answer never engages with *linearity*. "Linear loss curve" (loss
decreasing at constant rate in epochs, or in log-steps) is a shape claim; "smooth" is a
noise claim — and typical healthy curves are smooth *and* strongly non-linear (fast early
drop, then power-law/plateau). So the response answers a different question than the one
asked, and its "well-established belief" is the smoothness folklore, sourced to tutorials
and forum threads. The actual literature relevant to the linearity question is the
loss-curve *shape* literature above (power laws, the multi-power law, the broken-power-law
scaling-law fits) and the "linear-in-log-steps means nothing special" reading that follows
from them; see answer 2 below when it is added, and the EDP intake notes for why the
slope/R² features from the CIFAR study were ambiguous.

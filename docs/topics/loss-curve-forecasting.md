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

# Loss-curve forecasting and loss→accuracy mapping — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: the multi-power law is the analytic annealing correction in annealed
readouts (ANN-5/ANN-2) and a regression target in recipe featurization (REC-7); the
loss→downstream-accuracy mapping is what the decision-flip analysis and the IRT
emergence-as-measurement claim both depend on.

---

## 2026-08-22 — pointer: proxy-metric forecasting literature

The downstream-*accuracy* forecasting line (Patel et al. 2026 proxy metrics over expert
trajectories; NeuNeu 2601.19831; FLP 2410.08527; model ladders 2412.04403; observational
scaling laws 2405.10938) is accumulated in `small-scale-evaluation-metrics-literature.md`;
this file stays with loss-curve shape and extrapolation.

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
lineage — `../../potential-projs/early-dynamics-prediction.md` §4). The hypothesis being
checked — that more linear loss curves indicate better training — came from her advisor;
this query was Danielle testing whether it is an established belief and what it rests on:

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
from them; see answer 2 below, and the EDP intake notes for why the
slope/R² features from the CIFAR study were ambiguous.

## Undated (~2025) — same question, answer 2 of 2 (condensed)

**Answer 2 (Perplexity "academic" mode; paper-shaped citations).** Same headline — "a
widespread belief … that smoother, more linear loss curves generally indicate better
training quality and stability" — and the same substitution of smoothness for linearity,
now dressed in citations:

- *Stability*: a fast-adversarial-training paper (IEEE 10376811) where catastrophic
  overfitting co-occurs with "loss convergence outliers," motivating a "ConvergeSmooth"
  method that bounds loss differences between adjacent epochs.
- *Optimization theory*: "smooth loss functions are associated with better convergence
  properties" (arXiv 2208.04075; IEEE 10255658) — a claim about the smoothness of the
  **loss function** (Lipschitz gradients), not of the loss **curve**.
- *Regularization*: an L2-regularized CNN-LSTM whose "curve looks smooth" (IEEE 10872755);
  three applied papers (hate-speech LSTM, air-quality prediction, a geophysics
  resistivity-curve paper) cited as evidence that smooth curves mean good generalization.
- *Noise*: SGD noise driving parameters toward "noise equilibria" (a Semantic Scholar
  entry); larger LMs "stabilize early in training within the first 20% of epochs" while
  smaller ones converge "slower and less stable" (arXiv 2410.11451).
- *Counterarguments*: double descent (arXiv 2407.09845, 2203.07337; IEEE 10222624); loss
  landscape convergence depends on sample size and local geometry (arXiv 2409.11995); AdaGC
  (arXiv 2502.11034) removes loss spikes while "maintaining natural convergence patterns."
- Conclusion: "the current consensus favors controlled smoothness and stability over strict
  linearity."

Intake note: answer 2 is worse than answer 1 in a specific way — it manufactures support.
Its "theoretical foundations" conflate loss-*function* smoothness with loss-*curve*
smoothness; its "empirical evidence" is a set of minor applied papers whose authors
remarked that their curve looked smooth; a geophysics paper about resistivity–phase curves
is cited under "medical imaging." None of the cited work tests the proposition that *linear*
loss curves indicate better training, and the conclusion quietly concedes the point
("controlled smoothness … over strict linearity"). Net: neither answer found a literature
basis for the linearity hypothesis, and both defaulted to the smoothness heuristic. The
honest summary for the project is: **no established belief about linearity exists in the
literature; the well-documented shape priors are power-law / multi-power-law decay, with
smoothness as a separate stability heuristic.** **Flag:** the loss-slope study's premise — the advisor-supplied hypothesis that more
linear loss curves indicate better training — has no literature support that either search
could find. Any write-up of that study (or of EDP's linearity/R² features) must present
linearity as a hypothesis the study tests, not an accepted belief it builds on, and should
say so explicitly in the motivation rather than citing smoothness folklore as if it were
evidence for linearity. The only citations worth following for EDP are arXiv 2410.11451
(early stabilization of larger models — relevant to "how early is early enough") and AdaGC
2502.11034 (loss-spike handling, relevant to cleaning early-window features); both
unverified.

## Undated (~2025) — Related work for early-curve-linearity → final performance (two search rounds)

Danielle's question (verbatim):

> I'm interested in using the linearity of the early epochs of a loss curve to predict
> final performance of a model. Specifically when training cnn based vision models of cifar
> 10. What related work exists for this?

**Round 1** (general web search) found nothing: tutorials, a high-school journal
hyperparameter study, and a StackOverflow thread; concluded "no established body of work
… formally uses this property to predict final model performance." Zero content beyond
the (correct) observation that practitioners read early curves informally.

**Round 2** (academic search) found the actual literature. Identifier-bearing citations,
unverified but plausible:

| Line | Work | What it does | Why it matters here |
|---|---|---|---|
| Learning-curve extrapolation | Domhan, Springenberg & Hutter 2015 (IJCAI; "Speeding up automatic hyperparameter optimization of DNNs by extrapolation of learning curves") | Fits an ensemble of parametric curve families (power law, sigmoidal, …) to a partial curve, predicts the asymptote; CIFAR-10 CNNs among the testbeds; RMSE 0.25 / 0.19 / 0.11 after 10 / 40 / 60 epochs (per the response) | The direct ancestor of "fit something to early epochs, predict the end." Their features are the fitted-asymptote, not slope/linearity of a linear-in-log-x fit — that is the differentiator, such as it is |
| | Klein et al. 2017 (LC-Net / Bayesian NN over curves); Baker et al. 2017 ("Accelerating NAS using performance prediction") | Regress final accuracy on early partial curve + config/architecture features | Baker et al. is the closest methodological match to the loss-slope study and EDP (regressor over early-curve features + configuration features) — not cited by either round; from memory, verify |
| | LC-PFN — Adriaensen et al., NeurIPS 2023 (arXiv 2310.20447) | Transformer pre-trained on millions of synthetic curves; Bayesian extrapolation in one forward pass; "10,000× faster than MCMC" | The modern extrapolation baseline any early-window predictor should be compared against |
| | Ding et al. 2024 (arXiv 2412.15554) | Architecture-aware neural ODE for learning-curve prediction (MLPs and CNNs) | Shows the trend toward conditioning curve prediction on the configuration — EDP's recipe-conditioning is the same move on a different axis |
| Early-dynamics proxies | Neural capacitance — Jiang et al. (arXiv 2201.04194; Nat. Commun. 2024) | Converts the net to a line graph on edges, derives a "capacitance" metric from early-training synaptic dynamics; validated on CIFAR-10 model selection | A non-curve early-dynamics predictor; closest in spirit to "mathematical properties of early training" |
| Zero-cost proxies | ICLR blog-track 2022 survey; AZ-NAS, CVPR 2024 (arXiv 2403.19232); FreeREA, WACV 2023 | Scores at init or after minimal training; assembling proxies raises rank correlation | See `nas-literature.md` — the init-time end of the same spectrum |
| Training dynamics → reliability | an OpenReview paper on selective prediction from intermediate-checkpoint instability | Uses prediction instability across training to decide what to abstain on | Tangential; instance-level rather than run-level |

Round 2's gap statement: none "specifically isolate and leverage the linearity
characteristic of early loss curves as a primary predictive feature"; it proposes
quantifying linearity, correlating it with final accuracy across architectures, building
lightweight predictors, and combining with zero-cost proxies — and notes the observation-window
trade-off (Domhan), noise in early epochs (LC-PFN's motivation), and architecture dependence
(Ding).

Intake note: round 2 is the useful one and its reading list is the EDP related-work core,
together with the NAS-side proxies in `nas-literature.md`. Two corrections to its gap
claim: (1) "linearity" as a feature is not a gap in any interesting sense — it is one
particular curve-family choice inside the extrapolation literature (a linear-in-log-x fit
is the log-law member of Domhan's family), so the honest framing is "we use a deliberately
crude fit family and ask how early its slope carries signal," not "nobody has looked at
linearity"; (2) the pilot result in `../../past-projects/loss-slope-prediction.md` already shows
early *slope* predicts and high R² indexes slowness, which is a point against linearity
per se being the useful feature. The real differentiators for EDP remain the axis varied
(data recipe, across scales) and the target (downstream benchmark scores, ranking/decision
metrics), not the feature.

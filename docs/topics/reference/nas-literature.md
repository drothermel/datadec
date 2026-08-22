# Neural architecture search — reference topic

**Kind:** reference (standing accumulator). Entries are dated and quoted close to verbatim;
related-work claims are unverified unless a citation with an identifier is given.

Why it matters here: not NAS as a method, but the **performance-estimation** half of NAS —
training-free / zero-cost proxies and learning-curve extrapolation — which is the closest
established prior-art line for "predict a configuration's final performance from cheap
early signals" (`../../potential-projs/early-dynamics-prediction.md`, the CNN loss-slope
study in `../../past-projects/cnn-deconstruction-ladder.md`, and the eNTK readouts in
`ntk-literature.md`). Any EDP related-work section has to position against this
literature.

---

## Undated (~2025) — "What is the current state of NAS?" (Perplexity survey; condensed)

Danielle's question (verbatim): "What is the current state of neural architecture search
subfield in machine learning research"

**Framing.** Three components (Elsken et al. survey, automl.org): search space, search
strategy, performance estimation — the last "often the most computationally expensive."

**Recent directions listed.**
- *Efficiency*: predictor-based NAS (attention-enhanced path encodings; a 2025 Sci. Reports
  paper); **training-free metrics** (TE-NAS, UT Austin — "measurements grounded in deep
  learning mathematical theories" to rank architectures without training); efficient
  evolutionary search with proxies (EENAS, PMLR v189).
- *Applications*: NLP/transformers; meta-learning (Auto-Meta, arXiv 1806.06927);
  hardware-aware joint search (NAHAS, OpenReview); federated/personalized NAS.
- *Trends*: LLM-driven architecture generation (LLMatic, via a YouTube talk);
  carbon-efficient NAS (CE-NAS, NeurIPS 2024 — RL scheduling against carbon intensity,
  "up to 7.22×" emissions reduction); **zero-shot NAS** ("causal zero-shot NAS … 8 GPU
  seconds on CIFAR-10", OpenReview `3s6aE1LeiR`); quantum-assisted NAS (sourced to a
  magazine — discard).
- *Challenges*: compute (even after ENAS/DARTS); generalization beyond CIFAR/ImageNet;
  larger search spaces can hurt existing methods (Ci et al., ICCV 2021); benchmark
  operation diversity.
- *Benchmarks*: NAS-Bench-360 (diverse domains), NATS-Bench (topology + size spaces,
  extends NAS-Bench-201), the AutoML 2025 "NAS Unseen Data" competition.

## Intake notes

- Survey-grade and mostly blog-sourced (byteplus, LinkedIn, dotcommagazine); the paper-shaped
  citations (NAS-Bench-360, NATS-Bench, CE-NAS, TE-NAS, Ci et al.) are plausible real papers
  but unchecked. It does not mention DARTS-family pathologies, weight-sharing rank
  correlation problems, or the "random search is a strong baseline" results — the parts of
  the field most relevant to trusting any cheap proxy.
- **The thread to pull for EDP.** Zero-cost / training-free NAS proxies are literally
  "predict final accuracy of a configuration from an untrained or barely-trained network":
  TE-NAS (Chen et al., ICLR 2021) scores architectures by the **NTK condition number** and
  the number of linear regions at init; NASWOT (Mellor et al., ICML 2021) by activation
  overlap at init; Abdelfattah et al. (ICLR 2021, "Zero-Cost Proxies for Lightweight NAS")
  compare synflow / snip / grasp / jacob_cov / fisher and find them weak-to-moderate rank
  predictors; **learning-curve extrapolation** (Domhan et al. 2015; Klein et al. 2017
  LC-Net; Baker et al. 2017 "Accelerating NAS using performance prediction" — which fits
  regressors on *early partial training curves* plus architecture features to predict final
  accuracy) is the same method as the loss-slope study and EDP, applied to architectures
  instead of data recipes. All of these are from memory and unverified; together with the learning-curve-extrapolation
  table in `loss-curve-forecasting.md` they are the related-work list an EDP paper must cite
  and differentiate from. The differentiator is
  the axis being varied (data recipe / pretraining choices vs. architecture) and the
  target (downstream benchmark scores across scales vs. in-distribution accuracy).
- A caution these papers established that applies to EDP: zero-cost proxies' rank
  correlations often collapse within the top of the search space — good at separating bad
  from fine, poor at ranking among good. The ranking-head / decision-accuracy metrics in
  EDP should be evaluated on the *top-k* subset as well as the full population for the
  same reason.

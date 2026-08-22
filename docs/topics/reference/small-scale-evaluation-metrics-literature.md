# Evaluation metrics for language models at small scale — literature reference

**Kind:** reference (accumulator for the proxy-metric / downstream-forecasting literature
the DataDecide measurement projects position against: TINY, IRT's decision-reliability
frontier, EDP, DCARD). Entries are dated. Paper characterizations are a SciSpace agent's;
identifiers unverified unless Danielle-supplied. Sibling: `loss-curve-forecasting.md`
(curve extrapolation), `evaluation-methodology-literature.md`.

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-eval-of-llms-agent-artifacts-zip_e11a0b3d-f220-45af-9d38-6a50581427b3_1787424353/` — the seed paper PDF and its cropped figures, both review
rounds (markdown, LaTeX, PDF), the intermediate insight extraction, 32 downloaded
full-text PDFs, and ~30 search CSVs. **`INDEX.md` inside the folder is the file-level
index.**

---

## 2026-08-22 — pointer: loss-replacement metrics

Metrics that replace the loss itself (token-selected / reweighted NLL such as LongPPL,
bits per byte, tokenization-marginal likelihood, representation-side readouts) are
accumulated in `loss-alternative-metrics-literature.md`; this file stays with proxies
that *predict downstream accuracy*.

## 2026-08-22 — SciSpace deep review seeded on Patel et al. 2026 (two versions)

**Danielle's prompts (verbatim):**

> Please do a literature review of papers related to metrics for evaluation of language
> models at small scale, especially for downstream tasks. One example of a related paper
> would be https://arxiv.org/abs/2605.18607 which you can start with.

> I want you to significantly expand sections 3.4, 3.5, 5.2, 5.3, 5.4, 6.2, 6.3, 7.2, 7.3,
> 7.4, 7.5, 7.6, which more details about the specific experiments and key takeaways. Also
> create a comparison table summarizing the most important papers with their methods,
> evaluation metrics, key results, followup questions, potential weaknesses, and any other
> crucial information.

**The seed paper (Danielle-supplied ID: arXiv 2605.18607v1, 18 May 2026, "Forecasting
Downstream Performance of LLMs With Proxy Metrics" — Arkil Patel, Siva Reddy, Marius
Mosbach, Dzmitry Bahdanau; Mila/McGill and ServiceNow Research; PDF in the bundle).** As described across the two versions:
a library of 80 proxy metrics = 10 token-level core statistics (cross-entropy, top-k
accuracy for k∈{1,2,3,5}, entropy, rank, reciprocal rank, margin, wrong-confidence) × 8
weighting schemes, all computed from a single forward pass of the candidate model over
*expert-written solution trajectories* — so the proxy applies to models that cannot yet
solve the task. Three uses: (1) *model ranking* — 18 reasoning models from six base
families and six post-training recipes (0.6B–70B) on AIME 2025, HMMT, GPQA, USACO,
MMLU-Pro, SuperGPQA, expert trajectories from three open-weight reasoning models; a
linear RankSVM over proxies reaches leave-2-tasks-out Spearman ρ = 0.81 vs. 0.36 for
FineWeb cross-entropy and 0.33 for rBridge (expert-reweighted NLL); an oracle 3-sparse
proxy reaches 0.88. (2) *Pretraining data selection on DataDecide* — frequency-weighted
top-5 accuracy ranks the 25 corpora for the 1B target with decision accuracy > 0.85 at
~10⁻⁵ of target compute, "pushing the Pareto frontier" past DataDecide's own proxies.
(3) *Training-time forecasting* — along the OLMo-3-7B trajectory, proxies extrapolate
downstream accuracy over an 18× compute horizon at roughly half the RMSE of loss-based
baselines. Stated limitation: forecasting shown on one architecture/scale.

**The landscape the two versions give (condensed, by theme).**

- *Why cross-entropy fails as a selection signal:* token-averaged over the training
  distribution, blind to which tokens matter; Gadre et al. 2403.08540 (perplexity→downstream
  power law holds on average, varies by task; 104 models 11M–6.9B); Dudy et al. 2020
  (soft-match accuracy); tokenizer metrics uncorrelated with downstream (Ali et al.
  2310.08754); Krajewski et al. 2512.08894 (direct power law for log-accuracy at fixed
  tokens-per-parameter beats the two-stage loss→accuracy route).
- *Learned / neural predictors:* NeuNeu "Neural Neural Scaling Laws" 2601.19831
  (accuracy-trajectory extrapolation + token-level validation losses; 2.04% MAE on 66
  tasks vs. 3.29% logistic fits; zero-shot to unseen families); Ye et al. BIG-bench
  predictability 2305.14947 (MLP, >95% R², "small-bench" 3× smaller than BBH equally
  informative); Schellaert et al. 2305.12415 (DeBERTa assessors predicting per-instance
  success); ProxyLM 2406.09334 (small proxies for multilingual performance, 37× speedup);
  lineage-regularized matrix factorization 2504.19811 (model ancestry as a prior);
  FamiCom 2406.11243 (familiarity × complexity, ρ 0.848 with end-task performance).
- *Probes and uncertainty:* linear probes on activations predict correctness before
  generation (Pacchiardi 2025; poor on math); conformal probes (Ashok & May 2025);
  conformal set size as a benchmark axis (2401.12794); self-evaluation EQT 2501.11721;
  Kadavath et al. 2207.05221 calibration.
- *Small-model benchmarks:* SLM-Bench 2508.15478 (15 SLMs, 9 tasks, 11 metrics incl.
  energy); SLM survey 2409.15790; SLMs on code 2507.03160; ReTraceQA 2510.09351
  (answer-only metrics overstate SLM reasoning by up to 25%; 24% of flawed traces still
  correct); generative→NLU reformulation for 35× cheaper evaluation 2506.03592;
  Informedness over accuracy/F1 (2401.03831); reference-based metrics failing for modern
  models (2310.13800).
- *Scaling laws for downstream prediction:* Kaplan; Chinchilla; small-scale break below
  ~2.2e15 FLOPs (Pechi et al. 2305.17266); finetuning scaling laws need R² ≥ 0.95
  (Ivgi et al. 2022); data-constrained (2305.16264); quality-aware Q (2510.03313);
  effective tokens = diversity × syntheticity (2410.03083, r = 0.83 over 200 models
  25M–1.5B); FLP two-stage loss→performance 2410.08527 (5–10% error at 7B/13B); model
  ladders 2412.04403 (1% of target compute, within 2 points on some tasks; N and D beat
  FLOPs in overtrained regimes); context-aware 2510.14919; observational scaling laws
  2405.10938 (~80 public models, low-dimensional capability space, emergence as sigmoids);
  hyperparameter scaling 2505.13738.
- *Data mixture / selection laws:* AutoScale 2407.20177; UtiliMax / MEDU 2501.11747;
  D-CPT law 2406.01375; ADO 2410.11820 (small proxy models often fail to predict larger
  ones); BiMix 2405.14908; data mixing laws 2403.16952; optimal-mixture laws 2507.09404;
  loss-to-loss scaling determined by data and tokenizer, not architecture (2502.12120);
  knowledge capacity 2 bits/parameter (2404.05405); repeated-data double descent
  (2205.10487).
- *Emergence:* Wei et al. 2206.07682 vs. Schaeffer et al. mirage 2304.15004; proxy tasks
  for emergent abilities 2412.07111.
- *Contamination:* time-travel detection 2308.08493; C2LEVA 2412.04947.

**Intake notes.**

- **Version 2 fabricated the seed paper's author list** ("Patel, Magnusson, Groeneveld,
  Walsh, Soldaini, Tafjord, … Hajishirzi" — the DataDecide/OLMo team, plausible given
  the content and wrong). The PDF in the bundle gives Patel, Reddy, Mosbach, Bahdanau;
  version 1 had it right. A clean example of content-driven author hallucination.
- **v2's bibliography has swapped or fabricated entries** that the table and text cite as
  if correct: `gao2021framework` is cited as lm-evaluation-harness but the entry is The
  Pile; `luo2025scaling` cited as code-LLM scaling laws but the entry is WizardCoder;
  `xie2023finpythia` cited as FinPythia DACP but the entry is PIXIU; `chang2024effective`
  points to 2402.04177 (a different downstream-scaling paper) instead of 2410.03083;
  `bhagia2024scaling` points to 2410.08527 (the FLP paper) instead of 2412.04403;
  `koh2026rbridge` ("Koh & Liang 2026", no ID) is unverifiable; Ruan et al. appears twice
  in the table; several entries are "arXiv preprint" with no identifier; the LaTeX
  references figures under `figures/` that were not supplied. The v1 bibliography is
  cleaner (arXiv IDs throughout) — prefer it as the citation source.
- v2 did not do what was asked: rather than expanding v1's numbered sections 3.4, 3.5,
  5.2–5.4, 6.2–6.3, 7.2–7.6, it renumbered and restructured the review (new sections on
  contamination, calibration, cross-lingual, agentic), so the requested expansions cannot
  be checked against the original numbering. The comparison table was delivered
  (34 rows, with "Insuff. ev." flags). v1's section-level content survives mostly in
  condensed form.
- Some v2 claims are paraphrases of the seed paper's own framing ("pushes the Pareto
  frontier on DataDecide", "half the error") and should be re-read in the paper before
  being repeated.
- Where this bears on the repo: the seed paper is a direct consumer of DataDecide and
  the incumbent for the TINY / IRT decision-reliability comparison — recorded in those
  docs' §4.

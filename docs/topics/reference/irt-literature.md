# Item response theory for LM evaluation — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: the IRT reanalysis project turns IRT from a benchmark-compression tool
into a measurement instrument by exploiting structure in the model axis (recipe × scale ×
seed × step); the tiny-scale eval and the decision-reliability frontier are derived from
the same fit.

---

## 2026-08-18 — prior art, and what the DataDecide setting adds (from the Research Trajectory page)

**Prior art — "fit IRT to *diverse converged models* to compress benchmarks"**
- Lalor et al., *Building an Evaluation Scale Using Item Response Theory*.
- Rodriguez et al., *Evaluation Examples Are Not Equally Informative: How Should That Change
  NLP Leaderboards?* (ACL 2021).
- Polo et al., *tinyBenchmarks: Evaluating LLMs with Fewer Examples* (ICML 2024) —
  "IRT-selected ~100-item subsets that preserve full-benchmark rankings."
- *metabench — A Sparse Benchmark of Reasoning and Knowledge in Large Language Models*.

**What DataDecide adds.** "Hundreds of model×checkpoint rows, thousands of item columns,
binary (and margin) outcomes — precisely the data shape item response theory was built for…
the rows aren't arbitrary models, they're organized trajectories (recipe × scale × seed ×
step), which converts IRT from a compression tool into a measurement instrument."

**Cautions recorded at the time.** "IRT's local-independence assumption is violated by
shared-passage items and by contamination (fit diagnostics will flag these — also useful);
binary IRT discards the margin information that carries most small-scale signal, so fit
both binary and continuous-response variants (on likelihood margins) and compare — the
comparison itself replicating Signal and Noise's metric-choice finding inside a single
framework."

**Later additions (from the full-conference discussion, recorded in the IRT project §4):**
Rodriguez et al. (ACL 2021) and tinyBenchmarks (ICML 2024) as accepted main-venue IRT
papers; the pattern "IRT plus a claim or payoff, never IRT as reanalysis."

## 2026-08-24 — NotebookLM evaluation notebook: the 2024–2026 meta-evaluation cluster (IDs recovered)

Danielle supplied a NotebookLM notebook on LLM evaluation (bundle:
`nblm-llm-evaluation-notebook.md` in the 2026-08-24 intake bundle; data table +
two synthesis reports over 16 paper sources). Three of the sources are the
program's own foundations (DataDecide, Signal-and-Noise, model ladders) — those
rows restate what `datadecide-data-pipeline.md` and the ledger already hold. The
new material is the meta-evaluation cluster around this project's
benchmark-compression prior art, **with arXiv IDs supplied by the notebook's
report 2** (agent-supplied, unverified; NotebookLM inaccuracy caveat applies):

- **PSN-IRT / "Lost in Benchmarks?" 2505.15055** (Hongli Zhou 2025) —
  pseudo-Siamese neural IRT with a 4PL formulation, two pathways (model / item);
  used to *diagnose benchmark quality*, not just compress — the closest new
  neighbor to this project's IRT-as-measurement-instrument move, though still on
  diverse converged models rather than a structured recipe × scale × seed × step
  model axis. Fisher-information selection beats random for preference alignment.
- **EffiEval 2508.09662** (Yaoning Wang 2025) — capability-coverage maximization
  via a Model Utility Index; Kendall's τ > 0.9 with 5% of data;
  performance-independent selection as the fairness argument.
- **Benchmark² (no ID; Qi Qian 2025)** — benchmark meta-metrics: Capability
  Alignment Deviation (within-family hierarchy inversions), Discriminability
  Score, Cross-Benchmark Ranking Consistency, Benchmark Quality Score; 35% of
  data at 0.93 ranking consistency. Per-benchmark scores in the table (OmniMath
  DS 0.79 vs ARC DS 0.11 with CAD 0.87 — the alignment-vs-discriminability
  trade-off; SIQA flagged low-quality across families). Note: report 1 quotes
  ARC "DS = 0.03" against the table's 0.11 — internal inconsistency, unresolved.
- **ONEBench 2412.06745** (Ghosh 2024) — sample-level pooled benchmarking with
  Plackett-Luce rank aggregation; robust to >95% missing measurements; the
  aggregation-theory flank (vs Elo / Bradley-Terry / Borda, all shown less
  robust).
- **Federiakin 2501.17200** — psychometric reanalysis of the HF leaderboards:
  factor analysis, g-factor claim, anti-guessing normalization raising
  McDonald's ω 0.579 → 0.789, anchor items across leaderboard versions,
  DIF/measurement-invariance as an open problem; the naive-average-masks-
  plateau argument. The closest methodological cousin to IRT-as-reanalysis on
  the leaderboard axis.
- **ResampledBench 2504.09979** (2025) — farthest-point sampling in feature
  space; 1% of data at >0.96 rank correlation (27 VLMs).
- **SparseEval 2602.07909** (Taolin Zhang 2026) — sparse-optimization
  evaluation; report 2 lists it as a source but the table does not row it.
- **Amortized model-based evaluation (no ID; Sang Truong 2024)** — adaptive
  testing with Fisher-information acquisition (53% avg query reduction);
  *amortized calibration* (content-based difficulty prediction, constant- rather
  than linear-cost item calibration) and a *conditional question generator*
  (difficulty-targeted, PPO-trained, 10× better difficulty targeting than SFT)
  — the generative-item-bank flank.
- **EvaLearn (no ID; Shihan Dou, NeurIPS 2025)** — sequential problem-solving
  evaluation measuring learning capability (slope, first-attempt vs post-warmup
  accuracy); a different "ability" axis than static θ.
- **ADeLe (no ID)** — cognitive-rubric item-demand features for interpretable
  capability profiles; content-side complement to fitted item parameters.
- Detail upgrades on known items: tinyBenchmarks' **gp-IRT** estimator (convex
  combination of raw data and IRT predictions; robust to specialized-model
  distribution shift where correctness-clustering fails) and its coverage
  limitation (46/57 MMLU subtasks; metabench 37/57); stratified random sampling
  needs ~400+ samples/task to match IRT anchors.

Report 2's framing worth keeping (Federiakin-derived, agent-synthesized):
Representativism vs a causal theory of measurement; construct validity /
Evidence-Centered Design as the missing discipline; fit indices (RMSEA/CFI/TLI)
as the test of whether an aggregate mirrors structure or averages noise. All
unverified; IDs to confirm at verification.

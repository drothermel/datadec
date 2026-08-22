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

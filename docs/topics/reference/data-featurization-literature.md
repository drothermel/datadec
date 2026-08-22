# Dataset featurization — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: recipe featurization (REC) compares intrinsic, model-mediated, and
similarity-based feature families against DataDecide's outcome table; functional
featurization (FUNC) is the proposed fourth family.

---

## 2026-08-18 — how the field quantifies differences between datasets (from the Research Trajectory page)

Prompt context (Danielle): datasets become approximately black boxes because they are so
large, but we analyze trained models, which actually are black boxes — so there must be
things we can do for datasets as well. "Your framing — datasets aren't black boxes, just
obscured by size — matches where the field is, and the current toolkit splits into three
families."

**1. Model-mediated features** ("dominant, works best, philosophically unsatisfying for you")
- *Improving Pretraining Data Using Perplexity Correlations* — "uses losses of 90 public LLMs
  on tens of thousands of web domains as features, exploiting the observation that losses
  on pretraining texts correlate with downstream benchmark performance."
- RegMix (*Data Mixture as Regression for Language Model Pre-training*) — "formulates mixture
  selection as a regression task from small proxy runs, matching or beating DoReMi with ~10%
  of the compute." DoReMi (*Optimizing Data Mixtures Speeds Up Language Model Pretraining*).
- *Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance*;
  BiMix (*A Bivariate Data Mixing Law*) — "fit parametric laws jointly over domain
  proportion and data volume."
- "These predict outcomes well but featurize the dataset as 'mixture weights over named
  domains' — they don't tell you *what property* of the data mattered."

**2. Dataset-similarity embeddings**
- Task2Vec (*Task Embedding for Meta-Learning*); *Quantifying the Importance of Data
  Alignment in Downstream Model Performance* — alignment coefficients that "quantify
  similarity between two datasets and, in controlled interventional experiments, measure
  alignment's impact on downstream performance."
- Miranda et al., *Beyond Scale: The Diversity Coefficient as a Data Quality Metric* —
  "expected Task2Vec distance between batches — which correlates with downstream quality."
- Negative result to respect: *Data Similarity is Not Enough to Explain Language Model
  Performance*.

**3. Intrinsic corpus statistics** ("closest to your instinct, least developed")
- WIMBD (Elazar et al., *What's In My Big Data?*) — "corpus-level statistics — duplication,
  contamination, domain composition, length distributions — at trillion-token scale,
  proving the 'obscured by size' problem is tractable."
- Compression-based measures (gzip ratio, entropy-law style) as scalar
  complexity/redundancy features.
- Zipf / burstiness / type-token statistics — "exactly the properties Chan et al. showed
  *cause* ICL emergence in small transformers. That last link matters: the one place
  intrinsic data statistics have been causally tied to a capability is your [ICL]
  setting."

**Open questions recorded at the time**
- "No one has systematically featurized those 25 corpora (WIMBD statistics, diversity
  coefficient, compression ratios, perplexity-correlation profiles) and asked which features
  predict the outcome table, or whether intrinsic features match model-mediated ones."
- "Does a dataset's 'determinism profile' predict landscape geometry?" — the river-valley
  hypothesis "attributes the valley geometry itself to data properties — deterministic
  tokens forming the river, uncertain tokens the walls — meaning a dataset's 'determinism
  profile' (cheap to estimate with any reference model) is a candidate feature predicting
  not just performance but *landscape geometry*, i.e., annealing behavior."

---

## 2026-08-18 — the determinism profile as a feature family member

Per-token reference-model entropy as a corpus descriptor. Related machinery: DoReMi's
"excess loss" and Rho-1's reference-model excess-loss scoring (per-token epistemic
measurements used for domain weighting and token selection respectively); compression-based
statistics as the crude aggregate version; Task2Vec (Achille et al.) as the Fisher-embedding
formalism that the alignment and diversity coefficients build on. Design cautions (relative
to reference model and context length → curves; pre/post dedup) recorded in
`../../potential-projs/recipe-featurization.md` §4.

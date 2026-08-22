# Potential project: featurizing the DataDecide recipes

Status: proposal, 2026-08-21. Derived from [../dataset-analysis-idea-map.md](../dataset-analysis-idea-map.md) (ideas L0, F1, F2, F3) and the recipe ground-truth dig in
`~/drotherm/data/.claude/datadec/2026-08-19/2013-dclm25-dolma75-training-data.md`.

**One-line pitch.** DataDecide is a supervised problem nobody has set up: 25 pretraining corpora with measured outcomes across 14 scales, 3 seeds, and many tasks. Measure what is actually in each corpus, compute intrinsic and theory-motivated features, and ask which data properties explain which task-level differences.

---

## 1. What the project involves

### 1.1 Core (non-optional)

**C-a · Ground truth for every recipe.** Regenerate the exact shard manifest for each of the 25 recipes from `named_data_mixes.py` (OLMo `DataDecide` branch), sum realized token masses from `allenai/DataDecide-data-recipes` file sizes, and flatten each recipe to its leaf corpora (DCLM CC, Dolma CC head/middle/tail, RefinedWeb, StarCoder, C4, Reddit, peS2o, …). Output: a per-recipe composition table in realized tokens. Already done for one recipe; the 25-recipe version is mechanical.

Known finding this immediately surfaces: mixture labels are shard-file fractions, not token fractions — the DCLM/Dolma 25/50/75 recipes are 43/69/87% DCLM by tokens. Any consumer treating labels as mixture weights is wrong, including datadec's own `configs/dataset_features.csv`.

**C-b · Corpus sampler.** Sample tokens per recipe from the actual shards (stratified by leaf corpus so per-source statistics are possible), optionally detokenize with the `gpt-neox-olmo-dolma-v1_5` tokenizer.

**C-c · Outcome table.** The DataDecide eval results datadec already ingests, organized as recipe × scale × seed × step × task, with both accuracy and continuous (likelihood/margin) metrics; plus derived pairwise decisions and per-task rankings.

**C-d · Analysis framing that respects n = 25.** The recipes form families — Dolma 1.7 + 4 ablations, DCLM-Baseline + 6 QC variants, Falcon/Falcon+CC + 4 variants, 3 DCLM/Dolma mixes, and singletons (C4, FineWeb-Pro, FineWeb-Edu, Dolma 1.6++). Within-family contrasts are one-factor interventions with measured outcomes; the analysis leans on those rather than on a free-form 25-point regression. Validation is leave-one-family-out.

### 1.2 Optional directions

**D1 · Intrinsic-statistics features (F1).** WIMBD-style corpus statistics (duplication rate, length distributions, source/domain composition from C-a), contamination against the eval suite, compression ratio, Zipf/burstiness/type-token statistics, Task2Vec diversity coefficient. Cheap, mostly CPU, well-precedented individually; never assembled across these 25 corpora.

**D2 · Determinism profile (F2).** Per-token conditional entropy from a reference model (and an ensemble or larger model to separate the aleatoric floor), summarized per recipe as curves over entropy threshold × context length, computed before and after dedup. The river-valley-motivated feature: predicted to relate to landscape geometry and annealing behavior, not just performance. Reuses DoReMi/Rho-1 excess-loss machinery.

**D3 · Feature → outcome study (F3).** Which features explain which task-family outcomes; comparison against model-mediated baselines (perplexity-correlation features, RegMix-style mixture weights computed on *realized* shares) and similarity-embedding baselines (Task2Vec alignment). Family contrasts as dose-response evidence (e.g., DCLM QC 3→7→10→20% sweep vs. code/knowledge/commonsense task groups).

**D4 · Determinism profile → annealing behavior.** Does D2 predict how much each recipe's apparent ranking is schedule artifact? Targets come from the A-layer of the idea map: multi-power-law loss correction (analytic, T0) at minimum; checkpoint-merging pseudo-anneals or short decay branches if available. Further geometry targets once they exist: measured decay gain from short decay branches, interpolation-path curvature between checkpoint pairs, or per-token migration rates (when tokens stop responding to decay) — each produced by a separate project if pursued. This is the bridge to the causal token-bucket work (C1) and the part with a mechanism story.

**D5 · Per-source attribution via the mixes and ablations.** Use the three mixes (continuous DCLM share: 0/43/69/87/100%) and the Dolma ablations as a designed experiment over leaf-corpus shares, estimating per-task marginal effects of DCLM-CC, StarCoder, Reddit, Flan, math corpora. Purely from C-a + C-c; no new features required.

*D6–D8 ported from the former Project E doc (`e-dataset-featurization.md`, now removed):*

**D6 · Predict annealed outcomes.** Re-run D3 against annealed outcome values (evals of decay-branch endpoints or decay-weighted checkpoint merges, if another project produces them) instead of raw ones; report whether feature importance shifts once the schedule artifact is removed.

**D7 · Curve-parameter targets.** Fit the multi-power law (loss as a power law in cumulative LR plus decay-drop terms) to each recipe's released loss curve — T0, using `processed/scaling-law/checkpoint-losses.parquet` with `lr_at_step` / `cumulative_lr` — and use the per-recipe fitted parameters as the regression target: do intrinsic features predict the along-river term, the decay term, or neither?

**D8 · Per-task feature maps.** Which features predict which tasks; whether code / math / knowledge tasks load on different intrinsic statistics.

---

## 2. Doability and impact

Calibration: "impact" is judged against workshop-paper expectations — a contained, defensible result that people working on pretraining data would find worth reading and citing.

### 2.1 Overall doability — high

- All inputs are public and already reachable: manifests regenerate deterministically, shards are on HF, outcomes are in datadec. No training, no external dependencies.
- Compute is small: D1 is CPU over samples; D2 is reference-model inference (~20–50M tokens × 25 recipes × a few context lengths, hours-to-a-day on one GPU).
- Timeline with C-a/C-b/C-c built: 3–5 focused weeks to a workshop draft.

Risks are statistical, not engineering:

| Risk | Severity | Mitigation |
|---|---|---|
| Effective n well below 25 (families, collinear features) | high | family-contrast design, few pre-registered features, leave-one-family-out validation, rank-based targets |
| Outcomes are unannealed cosine checkpoints | medium | use rankings (reported stable) as primary targets; D4 quantifies the artifact explicitly |
| Determinism is relative to reference model and context | medium | report curves, not a scalar; two reference models; pre/post-dedup |
| Model-mediated baselines likely predict better | medium | frame as measurement + explanation, not prediction R² |
| Pairwise decisions are not independent | low | treat them as descriptive, not as test statistics |

### 2.2 Per-direction assessment

| Direction | Doability | Workshop impact | Why |
|---|---|---|---|
| **Core C-a (realized composition + label correction)** | very high; ~1 week | **medium-high, and the most likely to be cited** | First measured data card for a widely used suite; the label≠token-share correction is concrete and affects anyone using the mixes |
| **D1 intrinsic statistics** | high | low-medium alone | Well-trodden descriptors; value comes only when joined to outcomes (D3) |
| **D2 determinism profile** | high; main GPU cost | medium; highest variance | Novel, theory-linked feature; a clean correlation with task-family gains or annealing response would be discussed. Weak signal at this scale is plausible |
| **D3 feature→outcome** | medium (statistics) | medium with family framing; low as a regression paper | Within-family dose-response statements per task family are workshop-sized; a 25-point R² is not |
| **D4 determinism→annealing** | medium; depends on A-layer targets | medium-high if positive | Only direction with a mechanism story; sets up the causal follow-on. Needs at least the analytic correction, ideally branches |
| **D5 per-source attribution** | high | medium | Clean designed-experiment reading of existing results; interpretable per-task marginal effects of code/Reddit/Flan/DCLM-CC. Limited by few levels per factor |
| **D6 annealed targets** | medium; depends on annealed outcomes existing | medium | Makes the paper schedule-aware; a clean "feature importance shifts once you anneal" result would be notable. |
| **D7 MPL-parameter targets** | high; the MPL fit is T0 | medium | Compact targets with a physical interpretation; the fit itself is cheap. |
| **D8 per-task maps** | high; cheap once D3 exists | medium | Interpretable and practically useful; cheap once D3 exists. |

### 2.3 Recommended framing

"What is actually in the DataDecide recipes, and which measurable data properties explain which task-level differences." Measurement first (C-a), family contrasts and per-source attribution as evidence (D5, D3), determinism profile as the novel feature (D2), and D4 as the forward-looking section. Avoid a headline that is a prediction score. Drop D1 to a supporting role.

Expected outcome bands:
- **Floor (high confidence):** data card + label correction + per-source attribution. Publishable as a short paper; useful regardless of D2/D4.
- **Target:** floor + determinism profiles with an interpretable relationship to task-family outcomes or to annealing response.
- **Ceiling:** D4 shows the profile predicts schedule-artifact magnitude across recipes, motivating the WSD-branch / token-migration project.

---

## 3. Proposed infrastructure sequence

Each step produces something usable on its own and is reused by later per-token and branch-based analyses.

1. **Recipe manifest + composition module (C-a).** Exec `named_data_mixes.py` in a pinned copy, emit per-recipe shard lists and per-leaf-corpus realized token counts from HF `paths-info` sizes; persist as a versioned table in datadec. Replace `configs/dataset_features.csv` with the measured table. *Reused by:* everything below, probe-corpus construction.
2. **Shard sampler (C-b).** Given a recipe and a budget, sample uint16 token windows stratified by leaf corpus from `allenai/DataDecide-data-recipes`, with optional detokenization and a deterministic seed; cache locally. *Reused by:* D1, D2, probe-corpus construction for per-token analyses.
3. **Outcome table with full structure (C-c).** Extend the existing ingest to a tidy recipe × scale × seed × step × task frame with accuracy and continuous metrics, plus helpers for rankings, pairwise decisions, task families, and leave-one-family-out splits. Confirm per-item availability while here (gates the IRT project, not this one). *Reused by:* every analysis over the outcome table.
4. **Intrinsic feature extractors (D1).** Pure functions over sampled text/tokens: duplication, lengths, compression ratio, Zipf/burstiness/TTR, diversity coefficient. *Reused by:* D3.
5. **Reference-model token scorer (D2).** Per-token entropy/logprob from one or two open reference models at several context lengths; emits per-recipe profile curves. *Reused by:* any entropy-bucketed per-token analysis — the single most shared piece after the sampler.
6. **Feature–outcome analysis (D3, D5).** Family-contrast and dose-response analyses, baselines, figures. Analysis code, not infrastructure.
7. **Annealing targets (D4), only if continuing.** Multi-power-law correction over released loss curves (analytic); then, if the A-layer is pursued, checkpoint-merging utility and decay-branch harness feed D4 as additional targets.

Steps 1–3 are the commit-worthy foundation and take roughly the first week; 4–5 are a second week; 6 is the paper. Step 7 is deferred and belongs to the annealing project.

---

## 4. Decisions needed before starting

- **Reference model(s) for D2** and the context lengths to report.
- **Sampling budget per recipe** (trade: per-leaf-corpus resolution vs. GPU time for D2).
- **Primary outcome targets:** rankings vs. levels; which task families; which scales (likely 150M–1B where signal exists).
- **Whether D4 is in scope for the first paper** or deferred to the annealing project.

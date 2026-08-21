# Project E — Featurizing the DataDecide corpora

**Working title:** *What property of the data mattered? Intrinsic corpus statistics as
predictors of DataDecide outcomes and annealing behaviour.*

**One-line pitch.** DataDecide hands us a supervised problem: 25 corpora with a measured outcome
table (~300 pairwise decisions, per-task breakdowns, full curves). Nobody has systematically
featurized those corpora with intrinsic statistics (WIMBD-style composition, compression ratio,
Zipf/burstiness, determinism profile) and asked which features predict the outcomes, whether
intrinsic features recover the model-mediated ones, and — the schedule link — whether a
dataset's determinism profile predicts how much LR decay reveals.

Inventory IDs: E1–E5 (E5 depends on Project A or B; D1 is shared with Project D).

---

## 1. What the project involves

### Core experiment

1. **Intrinsic corpus statistics (E1)** for each of the 25 corpora: duplication and
   near-duplication rates, domain/source composition, document-length distributions,
   contamination against the eval suite, compression ratio (gzip / entropy-law style),
   Zipf exponent, burstiness, type-token statistics.
2. **Determinism profile (D1).** Per-token reference-model entropy distribution per corpus;
   threshold statistics such as "% deterministic tokens".
3. **Model-mediated baselines (E2).** Perplexity-correlation profiles (loss of several public
   LLMs on each corpus) and, where cheap, RegMix-style proxies — the strong-but-opaque baseline.
4. **Outcome regression (E4).** Predict DataDecide's outcome table (per-task final metrics,
   pairwise decisions, curve parameters) from each feature family; compare held-out predictive
   power; test whether intrinsic features recover the model-mediated ones.

### Optional directions

- **E-opt-1: Dataset-similarity embeddings (E3).** Task2Vec alignment and diversity
  coefficient as a third family. Known negative result (similarity alone doesn't explain LM
  performance), so its role is as a baseline to beat.
- **E-opt-2: Predict annealed outcomes.** Re-run E4 against Project A's annealed values
  instead of raw ones; report whether feature importance shifts once the schedule artifact is
  removed.
- **E-opt-3: Determinism profile → landscape geometry (E5).** Test whether the determinism
  profile predicts decay gain (Project A branches), interpolation-path curvature (Project C),
  or per-token migration rates (Project D). The schedule-specific payoff of the project.
- **E-opt-4: Per-task feature maps.** Which features predict which tasks; whether code /
  math / knowledge tasks load on different intrinsic statistics.
- **E-opt-5: Curve-parameter targets.** Use Project A's per-recipe MPL parameters (A5) as the
  regression target: do intrinsic features predict the along-river term, the decay term, or
  neither?

---

## 2. Doability and impact

### Overall doability: **high for the pipeline, low-powered for the headline**

- GPU-free (except reference-model scoring for the determinism profile and the
  perplexity-correlation baseline, both modest). Embarrassingly parallel across corpora. No
  dependency on any other project for the core.
- The structural limitation is **n = 25**. With 25 corpora and dozens of candidate features,
  any regression is underpowered and overfitting is the default failure mode. The honest
  analyses are: strong regularisation, leave-one-out, pairwise-decision targets (which give
  ~300 rows but are not independent), and a clear statement that this is hypothesis-generating.
- Some recipes are near-duplicates of each other (same source, different filtering), which
  helps: within-family contrasts isolate a single intervention and are the most interpretable
  results available.
- Expected headline: model-mediated features win; intrinsic features partially recover them;
  a few intrinsic statistics (likely duplication and determinism-related) carry most of the
  intrinsic signal. Useful, not surprising.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| Core (E1, D1, E2, E4) | **Medium** | A "reanalysis of DataDecide" paper with a complete feature table is useful and citable; the small-n caveat limits how strong any claim can be. |
| E-opt-1 similarity embeddings | Low–Medium | Completes the baseline set; unlikely to change the story. |
| E-opt-2 annealed targets | Medium | Makes the paper schedule-aware; a clean "feature importance shifts once you anneal" result would be notable. Depends on Project A. |
| E-opt-3 determinism → geometry (E5) | **High** (conditional) | The one direction that produces a mechanistic claim: data determinism predicts annealing behaviour. Depends on A/C/D artifacts. |
| E-opt-4 per-task maps | Medium | Interpretable and practically useful; cheap once E4 exists. |
| E-opt-5 MPL-parameter targets | Medium | Compact targets with a physical interpretation; depends on Project A's A5, which is itself cheap. |

**Recommended scope:** Run the core now as a background/parallel effort — it is the lowest-
coordination project in the set. Treat E-opt-3 as the paper's ambition and E-opt-5 as the
cheap schedule link that does not need branches. Pitch as a supporting paper unless E-opt-3
lands.

---

## 3. Infrastructure build sequence

1. **Corpus access layer.** Streaming access to each of the 25 DataDecide corpora (or the
   published samples, with a note on representativeness), with a common document iterator.
2. **Intrinsic-statistics pipeline (E1).** Per-corpus jobs for duplication, composition,
   lengths, contamination, compression, Zipf/burstiness/type-token. Output: one feature table,
   corpora × features, with provenance.
3. **Reference-model scoring (D1; shared with Project D step 3).** Per-token entropy over a
   sample of each corpus; summarise into determinism-profile features.
4. **Model-mediated baselines (E2).** Public-LLM loss per corpus sample; optional RegMix-style
   fit.
5. **Outcome-table ingestion.** Load DataDecide's per-task results, pairwise decisions, and
   curves into the shared results store (shared with Project A step 2).
6. **Regression/analysis layer (E4).** Regularised regression and leave-one-out evaluation per
   feature family; within-family recipe contrasts; per-task maps (E-opt-4).
7. *(Optional)* **Variant targets.** Accept annealed outcomes (`branch:*`, `merged:*`) and MPL
   parameters from Project A as alternative targets (E-opt-2, E-opt-5); accept geometry
   measurements from Projects C/D for E5 (E-opt-3).

Steps 1–4 are independent of every other project and can start immediately.

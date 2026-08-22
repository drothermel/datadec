# Experiment-tracking and run-infrastructure tooling — reference topic

**Kind:** reference (standing accumulator for tool comparisons around running and logging
many small training runs — tracking, sweeps, artifact storage). Retrieval/corpus storage
tooling lives separately in `retrieval-storage-tooling.md`. Entries are dated; vendor
claims quoted from responses are unverified marketing-grade statements.

Why it matters here: several project docs plan "many small runs × seeds × sweeps" (GEO,
TINY, EDP, the CNN deconstruction below), and Danielle's own `dr_exp` / `dr_results`
stack (see `project-approach-principles.md`, 2025 seven-track record) was built to serve
exactly that pattern. This file keeps the tool-choice reasoning so it is not re-derived.

---

## Undated (Perplexity, ~late 2024 / early 2025) — Logger options for a CNN-deconstruction study

**Danielle's project statement (verbatim — the useful part of this entry):**

> I want to slowly deconstruct the improvements in architecture and training procedure that
> were introduced into CNN based vision models from today all the way back to the earliest
> "deep" models. For each step I want to train on CIFAR 10 and ImageNet for a minimal length
> of time to do fast iteration for comparison (not aiming for super competitive results),
> and I want to track key metrics about the optimization landscape as I go. This will
> involve making significant numbers of iterative changes and running a substantial number
> of runs (multiple seeds per configuration + simple hpm sweeps).

This is the design statement behind the `deconCNN` / `dr_exp` "Engineering Journey" track (past-project record: `../../past-projects/cnn-deconstruction-ladder.md`)
and the CIFAR-10 loss-slope study (both in the 2025 seven-track record in
`project-approach-principles.md`; the slope study is EDP's lineage, see
`../../potential-projs/early-dynamics-prediction.md` §4). The shape — an ablation ladder
backwards through the history of a model family, cheap runs, seeds + small sweeps, and
landscape metrics tracked alongside accuracy — is the same shape as the DataDecide-era
docs `../../potential-projs/landscape-geometry.md` and `../../potential-projs/tiny-scale-measurement.md`.

**Response (condensed).** Asked for top logger setups *excluding TensorBoard and W&B*, it
listed, with sources almost entirely from Neptune's own comparison pages:

- *Neptune.ai* — managed tracking, offline/async logging, flexible metadata, side-by-side
  comparison, usage-based pricing.
- *MLflow* — open-source, self-hosted tracking server, cloud integrations, simple REST
  API; "basic visualization capabilities compared to commercial tools."
- *ClearML* — experiment versioning, automatic environment capture (code, containers,
  notebooks), offline mode with local cache sync.
- *Sacred + Omniboard* — research-focused; MongoDB-backed metadata, web UI, CLI/Python API
  for batch operations on large run sets.
- Pairings suggested: Neptune for cloud-agnostic managed; MLflow + DVC for open-source
  reproducibility; ClearML for offline-first.

Intake note: the response is a vendor-page summary (every citation is neptune.ai or one
netguru listicle) and carries no evaluation against the stated requirements — high run
counts, seeds, sweeps, custom landscape metrics, fast iteration. Its one relevant axis is
*offline/async logging + batch query API*, which is what the requirement actually demands.
Danielle subsequently built her own (`dr_exp` jobs DB + `dr_results` + Supabase sync, per
the seven-track record), which is the decision of record; this entry preserves the
requirement statement, not a tool recommendation.

## Undated (intake 2026-08-22) — Qualitative per-example comparison of pre-computed token probabilities, ending in marimo (three turns)

**Danielle's requirement, as it narrowed across the turns.** She has a collection of
models trained on different data recipes, evaluated on standard QA tasks, with output
probabilities per token/char. Turn 1 asked broadly for Python ecosystems to (1) learn from
the checkpoint itself, (2) learn about the data recipe, and (3) visualize questions and
predictions individually and comparatively. Turn 2 redirected: she will generally have
*only the probabilities for a few tokens per model–example pair* — no model in the loop —
and wants to "qualitatively compare models and learn from individual examples between
models"; "this might be more in the visualization space than the ML space." Turn 3 fixed
the surface: the analysis happens in a **marimo notebook**. The operative requirement is
therefore a notebook-native workbench over a long table of
(model/recipe, example, candidate, token-level log-probs) — exactly the per-item OLMES
outputs described in `datadecide-data-pipeline.md` — with example navigation,
side-by-side model panels, and selection that flows back into Python.

**Responses (condensed).**

- *Turn 1 (model-in-the-loop tooling, mostly off-target after turn 2).* Internals:
  TransformerLens, nnsight, SAE Lens, TorchLens. Data recipes: Data-Juicer, the Pythia
  suite, Data Portraits (membership sketches). Prediction visualization: BertViz, Ecco,
  Captum `LLMAttribution`, a "Token Visualizer" repo, eli5's log-prob highlighting.
  Comparison: lm-evaluation-harness, Zeno (slice-based), W&B, Google's LLM Comparator,
  LIT. Only Ecco/eli5-style log-prob highlighting, Zeno's slice view, and LLM
  Comparator's side-by-side rationale view are relevant once the model is out of the loop.
- *Turn 2 (pure visualization).* Plotly Dash (callbacks + crossfiltering), Streamlit,
  Altair/Vega-Lite (declarative, linked brushing, faceting by model), Panel/HoloViews
  (`hv.link_selections`), matplotlib + ipywidgets, Bokeh. Recommended Dash with faceted
  heatmaps, hover tooltips, and crossfiltering. The code sketches are generic scaffolds
  (a vocabulary×position heatmap is the wrong shape for "a few tokens per pair"; see
  note below).
- *Turn 3 (marimo).* `mo.ui.altair_chart` — chart selections come back as a filtered
  DataFrame (`chart.value`); `mo.ui.plotly` — selection supported for scatter and a few
  chart types, with a rendering-bug caveat (marimo issue 5326 cited); matplotlib/seaborn
  static; HoloViews/Panel usable but less integrated. UI: dropdown/slider/radio,
  `mo.ui.tabs`, grids, accordion, markdown, reactive re-execution. Recommendation: Altair
  via `mo.ui.altair_chart` as primary, Plotly for scatter-type selection.

**Intake notes.**

- The decision of record is *marimo + Altair*; the reactive-cell model plus
  `altair_chart.value` gives the example-navigation → panel-update → selection-back loop
  without callback code. Everything in turn 2 that is a standalone server (Dash,
  Streamlit, Bokeh server, Panel server) is superseded by turn 3.
- The data shape the sketches assume (dense vocab×position matrices) does not match the
  requirement (a handful of candidate continuations per example, each with per-token
  log-probs, per model). The natural Altair shape is a long table with one row per
  (model, example, candidate, token) and faceting by model with candidate on one axis;
  per-item comparative views then reduce to small multiples + a linked table of the
  example text. Nothing in the response engages with this; it is design work, not
  tooling choice.
- For DataDecide specifically the table already exists in per-item OLMES outputs
  (`correct_prob`, `sum_logits_corr`, per-candidate logits — see
  `datadecide-data-pipeline.md`, OLMES metric entry and DCARD-1(e)); the workbench is a
  consumer of DCARD's column definitions, not a separate pipeline.
- Turn-1 tools worth keeping as pointers for *other* projects: Data Portraits (membership
  testing against The Pile/The Stack — relevant to contamination checks), Zeno (slice
  analysis, relevant to IRT-style per-item work), LLM Comparator (rationale-grouped
  side-by-side). All citations response-supplied and unverified; several are listicles or
  vendor pages.
- No project ID; this is infrastructure supporting DCARD / IRT / TINY per-item analysis.


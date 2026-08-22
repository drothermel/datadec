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

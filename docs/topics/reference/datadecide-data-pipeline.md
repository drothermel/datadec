# DataDecide data-processing pipeline — reference topic

**Kind:** reference (accumulator for external readings of the `datadec` repository's data
layer — what is built, what is verified, and what analysis-side pieces are still missing).
The repository's own `README.md`, `configs/*.toml`, and verification scripts are the
authoritative description; this file records how outside conversations read that state
against the planning docs, and which of their claims checked out. Entries are dated.

Why it matters here: several project docs (IRT reanalysis, trajectory statistics,
annealed readouts, recipe featurization, early-dynamics prediction) assume specific tables
exist. A conversation that maps the repo against a plan is a cheap audit of those
assumptions.

---

## 2026-08-22 — Repo read against a three-paper plan (conversation undated, ~2026-08)

**Danielle's prompt (verbatim):**

> are you able to look at this github repo: https://github.com/danielle-rothermel/datadec
> it should be public?

The response read the public repo and revised its earlier plan, which had budgeted "two
weeks of foundation work" for items it labelled P1–P3 / A1–A2 / B5 / C-a. Those labels
come from the preceding part of the conversation, which is not on file; from content they
map to: P1 ≈ IRT reanalysis with recipe-DIF (`../../potential-projs/irt-reanalysis.md`,
IRT-3/IRT-7), P2 ≈ a data-card / recipe-composition paper
(`../../potential-projs/recipe-featurization.md`, REC-a manifest module = "C-a"),
P3 ≈ multi-power-law fitting with the LR-aware correction (`../../potential-projs/trajectory-statistics.md` /
`annealed-readouts.md`), A1/A2 ≈ drift/diffusion statistics with an LR regression,
B5 ≈ a likelihood-margin (continuous) IRT response matrix. Treat the mapping as inferred.

**What the response said is already built** (condensed; each checked against the repo):

- Per-instance layer: `instances.parquet` and `choices.parquet` per recipe, primary-key
  uniqueness checks, cross-source parity on 482 overlapping checkpoints between aggregate
  and detail tables, reconstruction of task metrics from instance rows. *Checks out*
  (`README.md` "Verification"). The choices table is the input for a likelihood-margin
  response matrix, so binary and continuous IRT variants both have defined inputs.
- Checkpoint derivations denormalized onto every checkpoint-bearing processed table
  except instances/choices: tokens, exact-parameter FLOPs, `lr_at_step`, `cumulative_lr`.
  *Checks out* (`README.md`; `configs/catalog.toml` nominal / training / exact parameter
  counts, `flops_per_token_per_parameter`).
- Scaling-law checkpoint-losses table: 27,106 rows, validated. *Checks out.*
- The 2026-08-19 zero-contradiction validation across five outputs, and the verifier's
  explicit detection of raw sources encoding nominal-parameter compute. *Checks out.* The
  response's framing — this provenance discipline is itself evidence for the
  data-card-paper thesis (the suite's raw scaling-law exports encode nominal-parameter
  compute) — is a fair reading of the README.
- HF publishing pipeline wired (`scripts/publish.py`, `configs/publishing.toml`),
  giving the data-card paper a natural release artifact and strengthening a
  NeurIPS D&B / resource-track fit.

**Caveat the response carried forward (correct):** LR derivations are internally
consistent but not independently confirmed, because no raw source records schedule
values (`README.md`). `lr_at_step` is load-bearing for the LR regression and the MPL
correction, so one external spot-check against the OLMo DataDecide-branch config code is
recommended before a paper leans on it.

**Open question the response posed — answered from the repo:** "do detail archives
exist upstream for all 25 recipes, and have you pulled them?" `configs/sources.toml`
declares `olmes_details` archives for all 25 recipes (c4 … fineweb-pro). Whether all 25
have been downloaded and processed is local machine state, not recorded in the repo; the
README's worked example is one recipe (`dolma1.7-no-math-no-code`: 542 checkpoints,
20.4M instance rows, 74M choice rows, ~14 min). The response's extrapolation — ~500M
instance rows and ~1.9B choice rows across the suite — follows arithmetically and
motivates its design point: the response-matrix builder should aggregate to (row × item)
matrices at build time rather than materializing anything at choice granularity.

**What the response listed as still missing (priority order):**

1. Coverage census: process or inventory `olmes-details` for all 25 recipes; emit the
   recipe × scale × seed × step coverage report (gate for recipe-DIF).
2. Checkpoint-spacing statistics per scale (gate for drift/diffusion work).
3. Manifest/composition module (REC-a) inside the repo next to `catalog.toml`, replacing
   `configs/dataset_features.csv`.
4. Thin analysis-side accessors: ordered trajectory views, response-matrix builder.

Revised recommendation: start the IRT work this week on the one processed recipe (fit a
2PL to shake out the module) while the other archives process in the background.

**Intake notes.**

- The response's factual claims about the repo all check against `README.md` and
  `configs/` at head `14cec12`; no errors found. Its numbers are quoted from the README
  rather than recomputed.
- `configs/dataset_features.csv` does not exist in the repo at this head; the only
  references are in `recipe-featurization.md` (which says it is wrong and should be
  replaced by the measured manifest table). Either it was removed already or the
  planning doc's reference is stale — worth a one-line check when REC-a starts.
- The "two weeks of foundation work" estimate that the response retracts was its own
  earlier estimate, made before it read the repo; the retraction is the useful part.
- The P/A/B/C labels belong to a plan document not on file. If that plan surfaces, it
  should be routed to the relevant project docs' §4 and this entry's mapping corrected.

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

---

## 2026-08-22 — Coverage closed; LR-derivation provenance (same conversation, next turn)

**Danielle's statement (verbatim):**

> the detailed parses exist for these and they've all been pulled, processed and uploaded
> to my private huggingface dataset. I don't remember exactly what I finally landed on for
> the lr derivations but I did my best to get the info and these estimates are the closest
> I could get. I think I scoured the repo, the issues, the drive docs, the paper and maybe
> even asked the paper authors (who generally didn't know a ton of the details because
> this was a huge sweep and they published what they did know...).

Attached: a screenshot of the private HF dataset (`drotherm`, head commit `9f28329`,
"Publish OLMES details for fineweb-pro", verified), listing all 25 recipe folders from `c4`
to `fineweb-pro`, each with a "Publish OLMES details for …" commit about a day old.

**Repo facts this settles:** the per-instance/choice layer exists, is processed, and is
published for all 25 recipes (private). The coverage census is no longer a gate — it is
an appendix table generated from held data. The LR schedule values in the processed
tables are best-effort derivations assembled from the OLMo repo, GitHub issues, the Drive
docs, the paper, and (probably) author correspondence; the authors themselves did not
know all details of the sweep. The README's "internally consistent but not independently
confirmed" caveat is the accurate statement of their status.

**Response, condensed to what survives:**

- Most portfolio analyses depend on the schedule's *shape*, not absolute values: a
  cosine schedule is a monotone transform of position post-warmup, so a
  diffusion-vs-`lr_at_step` regression is robust to a slightly wrong peak LR or
  parameterization; what would break it is a wrong schedule *family*, which the OLMo
  config code pins down. Cross-scale comparisons with different peak LRs per size are
  the exposed case; defense is a sensitivity sweep over plausible peak/warmup bounds.
- **The MPL fit doubles as the missing independent check:** the multi-power law is
  parameterized in cumulative LR and decay structure, so materially wrong derived
  schedules would fit observed loss curves systematically poorly or need per-run
  parameter contortions. Good fits across recipes × scales with shared structure are
  affirmative evidence the schedules are right "in every way the loss dynamics can see."
  Paper framing: "no raw source records schedule values; we derive them from the
  training configuration and validate via loss-curve dynamics."
- Cheap extra check: HF checkpoint directories sometimes embed the training config yaml
  or optimizer state; any DataDecide checkpoint that does gives a direct (scale, step)
  spot-check.
- The provenance trail is *content* for the data-card paper: "LR schedules are not
  recoverable from any published artifact and the authors could not confirm them" sits
  beside the label≠token-share correction and the nominal-vs-exact compute discrepancy —
  three independent cases where the suite's self-description and its ground truth
  diverge; "the pattern is the paper." Downstream papers cite the data card for cleaned
  inputs, making best-effort derivations the canonical ones.
- Updated sequence: (1) response-matrix builder against the full detail set, aggregating
  to (row × item) at build time; (2) 2PL/margin fits on one recipe end-to-end, then the
  suite; (3) in parallel, checkpoint-spacing statistics — the only unchecked gate left in
  the top three projects; (4) write the LR provenance narrative into the data-card
  outline now, including the author correspondence. Flip the private HF dataset public
  when the data-card paper ships; the publishing pipeline is the distribution mechanism.

**Intake notes.**

- The response says "24 recipes visible in frame … with the remainder below the fold."
  The screenshot shows all 25 (`c4` through `fineweb-pro`, matching `configs/sources.toml`
  exactly); nothing is below the fold. Harmless miscount, conclusion unaffected.
- "Cosine schedule" is the response's assumption about the DataDecide schedule family;
  the repo's `catalog.toml` / schedule code is the authority on what was actually
  derived — check before repeating the monotone-transform argument in a paper.
- The MPL-as-validation argument is sound as stated but one-sided: a good fit rules out
  schedule errors the loss dynamics can see, not errors invisible to them (e.g. a
  uniformly scaled peak LR absorbed into fitted coefficients). Still the best available
  check; record as such.
- Danielle's "maybe even asked the paper authors" is hedged; the data-card narrative
  should only claim author correspondence if the emails can be found.

---

## 2026-08-22 — No training loss from the authors; sparse small-scale checkpoints; the "DataDecide-dense" retrain idea (same conversation, two turns)

**Danielle (verbatim, turn 1):**

> ok, the one thing the authors said they could not provide was actual loss metrics across
> the runs. we only have the perplexity measures and accuracy measures for each checkpoint.
> does that become an issue?

**Danielle (verbatim, turn 2):**

> lets not treat "the authors could not provide training loss" as a fact yet, I have to
> double check to be sure I'm not misremembering. but reconstructing the missing quantity
> sounds like a good idea. also, especially the much smaller models have very few
> checkpoints (like 4-10) which was an issue back when I was trying to predict final values
> from early curve sections. I'm not sure if it will be an issue here but those are
> conveniently also the scales that it would be very doable to retrain some examples.
> thoughts?

**Status of the "no training loss" claim:** unconfirmed by Danielle's own account. One
repo fact bears on it: `processed/scaling-law/checkpoint-losses.parquet` carries a
`train_cross_entropy` column plus four validation cross-entropies (c4_en, dolma common
crawl, pile, wikitext-103), but they are populated for only 16% of rows — none at any size
≤ 90M, 7–22% of rows at 150M–750M, 79% at 1B (22 of 25 recipes appear). So some
training-loss values *were* released (in the scaling-law ladder CSVs), just not at the
checkpoint cadence or the scales where they would matter most. The exact finding for the
data card ("not logged" vs. "logged but not released" vs. "released partially") is still
Danielle's to pin down.

**Response 1, condensed:** held-out cross-entropy (what the PPL tables are) substitutes
for training loss almost everywhere and is better for matched-loss pairing across
recipes, since training loss is measured on each recipe's own mixture. Where it bites:
MPL fitting — conceptually fine on held-out CE, but the PPL table is thin per run, so fit
with partial pooling across seeds and scales, check what the scaling-law checkpoint-loss
table actually records (see above), and use the planned truncated-curve validation as
go/no-go; and any forensics needing per-batch training loss (spike detection, data-order
effects in the original runs) is unavailable. Recovery option: reconstruct own-mixture
held-out CE at checkpoint cadence by forward-passing released checkpoints over a held-out
sample of each recipe's mixture drawn with the REC-a manifest/sampler; this is the closest
analog of training loss, is embarrassingly parallel, and yields the cross-loss matrix
(every recipe's model on every recipe's mixture) as a by-product. Meta-point: "no training
loss" would be a fourth provenance-ledger entry, and the data-card thesis becomes
"DataDecide is an eval suite being used as a training-dynamics suite; here is what it
takes to make that valid" — every reconstruction (LR via MPL, own-mixture CE, realized
composition) is a contribution under it.

**Response 2, condensed:** verify first, because the data card must state the finding
exactly. Sparse checkpoints and forward-pass reconstruction fix *different* problems:
reconstruction changes *what* is measured per checkpoint; it cannot create temporal
resolution. With 4–10 checkpoints, the drift/diffusion decomposition is dead at those
scales (dynamics claims live at dense-save scales; small scales get endpoint statistics),
adjacent-checkpoint item flips stop meaning "churn," and per-run MPL fits are
impossible (pooling becomes the method); IRT and the decision-reliability frontier are
mostly fine. Hence **DataDecide-dense**: a few recipes (spanning the outcome range plus one
within-family pair) × the 2–4 smallest scales × many seeds (10+), dense checkpointing, and
everything the original suite lacks logged — true training loss, the LR schedule as
executed, the realized data-order manifest, per-token held-out losses on a frozen probe
set. One campaign discharges five obligations: temporal resolution at small scale; the
P2 order-effect experiment if sampling strategy is an arm; ground-truth validation of the
LR derivations and the MPL; real many-seed noise floors; and the training-config parity
check that the suite docs already list as the gate for any training work. The parity gap
(faithful reproduction vs. published run, relative to seed variance) is a reproducibility
result in its own right. Cautions: freeze the logging spec (cadence: log-spaced early,
uniform late; probe corpus; per-token outputs; manifest format) before the first real run
and emit the results store's existing schema; keep scope disciplined — it is P2's
interventional arm, P3's validation arm, and the tiny-scale substrate, not a larger
retraining project. Sequence it after the pure-T0 work is moving, but write the design doc
now.

**Intake notes.**

- Response 2 calls checkpoint-spacing statistics "the only unchecked gate left." That
  gate was answered 2026-08-21 in `../../open-questions-answered.md` (4M: 6 steps, 6M: 5,
  8M: 10; ~1,000–1,300-step spacing from 8M to 530M; 1B coarsest at ~2,500). Danielle's
  "4–10" recollection matches the table. The potential-projs README's TRJ row says
  "checkpoint spacing confirmed adequate" — true for ≥ 8M, and the sub-10M exception is
  exactly what this turn is about.
- The 750M aggregate OLMES table is truncated at step 26,250 while the instance table is
  complete (same file); any per-scale density claim should use the instance-derived view.
- "Cosine" as the schedule family is consistent with `wsd-suite.md` ("its original cosine"
  schedule); not re-verified here.
- Response 1's "ppl table ~23k rows → a few dozen points per run" is arithmetic on the
  README's 22,709 PPL rows and the spacing table; consistent.
- Literature claims in the responses (MPL developed on dense train/val logs; cooldown
  drops conventionally shown on validation loss) are unverified.
- The "Project B" referenced in response 2 maps to the WSD / MoE recipe suites (both list
  "training config parity" as step 1); inferred.

---

## 2026-08-22 — WSD arms in DataDecide-dense (same conversation, next turn)

**Danielle (verbatim):**

> if I was going to do DataDecide-dense I'd want to also do WSD. because if we're investing
> in getting to the point that we can train and doing a grid then the value of having the
> smallscale wsd becomes much higher than the cost

**Response, condensed:** agrees, and argues WSD arms transform rather than pad the
mini-suite. (1) Cosine twins with matched data order — same recipe, same realized token
stream, only the schedule differs — is the cleanest experiment in the annealing program
and only possible in runs you control; it converts the assumption that cosine-resumed
decay branches / checkpoint merges approximate a true anneal into a measurement, before
the 150M–300M branch budget is spent on released checkpoints (this is WSD-opt-3 done
small first). (2) Every intermediate readout becomes a proper annealed readout, enabling
the hypothesis that annealed readouts improve measurement SNR most at small scales, where
wall oscillation is proportionally largest — "WSD + branches is what makes 10M-scale
experimentation measurable at all." (3) Schedule × sampling-strategy × seed gives the
order-effect × schedule interaction (does data-order sensitivity differ between
stable-phase and decaying-LR training?) for free.

Where the cost hides (design, not GPUs): **tuning parity** — WSD's stable-phase LR is not
"reuse the cosine peak"; budget a small LR sweep at one scale, transfer with a stated
rule, report sensitivity, or every twin comparison is confounded. **Pilot-first
sequence** — cosine reproduction of one recipe (parity gate) → its WSD twin → two or three
branch points with a small decay length/shape sweep on that run → freeze the spec → fan
out; do not skip the pilot because reruns are cheap, since what it protects is suite
homogeneity, not compute. **Define what branches consume** — the continuation of the
parent's data stream (natural; keeps twins exactly matched) vs. fresh/replayed data; pin
it in the spec because it interacts with the order-effects arm and MiniCPM-style
decay-data experiments.

Portfolio effect: DataDecide-dense-WSD becomes the shared interventional substrate (REC's
order-effect arm, the annealing validation arm, TINY's measurement substrate) and is the
WSD suite's pilot entered deliberately. Hold the scope line at *scale*, not schedule:
150M+ stable-phase runs or many more recipes is the separate resource-paper decision.
The branch runner and results-store `variant` conventions move earlier on the critical
path; building them against 10M-scale runs is the cheapest place to get them right. The
mini-suite itself (true loss, true LR, realized order, twins, annealed readouts) is a
releasable object.

**Intake notes.**

- The cost figures ("branches add ~10% per branch point"; Hägele et al. guidance
  "calibrated at larger scales") are unverified response claims.
- "B-opt-3" in the response = WSD-opt-3 in `../../potential-projs/wsd-suite.md`; the
  quoted phrase "makes Project A's method validatable" is a paraphrase of that doc's
  impact table, not a quote.
- Danielle's statement is the decision-relevant content: the WSD arm is conditional on
  doing DataDecide-dense at all, which is itself unstarted.

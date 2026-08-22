# DataDecide data card — what is actually knowable about the suite, with a validated release

> **Draft scaffolding (2026-08-22).** Promoted from notes accumulated in
> `recipe-featurization.md` §4 and `../topics/reference/datadecide-data-pipeline.md` during
> a 2026-08-22 conversation. §1–§3 are synthesized from that record and not yet reviewed by
> Danielle; §4 is the dated discussion record (her statements verbatim). Treat §1–§3 as
> provisional until this note is removed. **All reproduction numbers cited anywhere in this
> doc come from agent-written verification code Danielle has not yet personally read,
> debugged, run, or analyzed — flags, not findings.**

**Program pillars served:** none directly; this is the release artifact every
DataDecide-facing project (IRT, TRJ, ANN, REC, TINY, EDP, ELI) cites for its cleaned
inputs. (Program: `README.md` → Program.)

**One-line pitch.** DataDecide is an evaluation suite being used as a training-dynamics
suite. Document, from a validated processed release, exactly what the suite's published
artifacts do and do not support: a provenance ledger of every place the suite's
self-description diverges from its ground truth, a claim-by-claim reproduction of the
paper on the cleaned tables, a coverage and abnormality ledger, and the reconstructions
(LR schedules, realized composition, own-mixture loss) that make dynamics analyses valid.

IDs: DCARD-1–DCARD-4 (core), DCARD-opt-1–DCARD-opt-3.

**Paper goal.** Resource / datasets-and-benchmarks track (NeurIPS D&B or equivalent), with
the processed HF dataset flipped public at submission. Candidate framing sentence for the
slice of the program it anchors: *DataDecide with error bars*.

Compute tiers: **T0** for the ledgers and reproduction; **T1** forward passes for
DCARD-opt-1.

---

## 1. What the project involves

### Core

- **DCARD-1 — Provenance ledger.** One entry per divergence between the suite's
  self-description and its ground truth, each with evidence and the repo's correction:
  (a) mixture labels are shard-file fractions, not token shares (the DCLM/Dolma 25/50/75
  recipes are 43/69/87% DCLM by tokens); (b) raw scaling-law exports encode
  nominal-parameter rather than exact-parameter compute (caught by
  `verify_preprocessed_derivations.py`); (c) learning-rate schedules are not recorded in
  any published artifact — derived from the OLMo repo, issues, Drive docs, and the paper,
  authors unable to confirm sweep details; (d) training loss is absent at checkpoint
  cadence (present only sparsely at 150M–1B in the scaling-law ladder CSVs; what the
  authors can supply is unconfirmed); (e) incomplete seed replication at some sizes
  (750M recollection, unverified; the 750M aggregate-table truncation is established).
- **DCARD-2 — Claim-by-claim reproduction.** The paper's claims as frozen predicates
  over the processed tables, each classified *reproduced* / *approximately* /
  *directionally consistent* / *fails on cleaned data* / *depends on definitional
  choices the paper did not pin down* / *not assessable*, with sensitivity analyses around
  thresholds and atomic decomposition of conjunctive claims. Current agent-generated
  state: headline results reproduce (0.8033 150M→1B decision accuracy; compute-reliability
  trend; task-difficulty spread; spread-to-noise ρ 0.798); the qualitative metric-family
  and curve-shape narrative largely does not (raw-likelihood dominance at small scales at
  2.38% vs. >50%; raw-plateau / penalized-converge fails both halves; BoolQ-only-at-1B
  fails; margin tracks accuracy at 0.360 vs. Norm Correct Prob 0.916). Required before
  any failure is framed as a contradiction: a definition-matching pass against the paper's
  released analysis code.
- **DCARD-3 — Coverage and abnormality ledger.** Automated recipe × size × seed × step
  table of cells present, early-terminated, and known-issue; every downstream analysis
  declares exclusion rules against it; published numbers whose support runs through thin
  cells inherit a flag.
- **DCARD-4 — Validated release.** The processed tables (PPL, aggregate OLMES, scaling-law
  evaluations and checkpoint losses, OLMES detail tasks/instances/choices for all 25
  recipes) with the verifier outputs, published from the existing pipeline.

### Optional directions

- **DCARD-opt-1 — Own-mixture held-out CE.** For each recipe, a held-out sample of its own
  mixture via the manifest/sampler (REC-a/REC-b), forward-passed over released checkpoints:
  the closest well-defined analog of training loss at checkpoint cadence, plus the
  cross-loss matrix (every model on every mixture) as a by-product.
- **DCARD-opt-2 — LR derivation validated by dynamics.** Multi-power-law fits across
  recipes × scales with shared structure as affirmative evidence the derived schedules
  are right in every way the loss dynamics can see; a sensitivity sweep over plausible
  peak/warmup; a spot-check of any checkpoint directory that embeds its training config.
  (Shared with `annealed-readouts.md`.)
- **DCARD-opt-3 — Reproduction-methodology framework.** Frozen predicates, threshold
  sensitivity, atomic decomposition, the `not_assessable` category, and a
  predicate-liveness guard (comparison set non-empty, size reported) as a reusable
  contribution to reproduction practice.

## 2. Doability and impact

### Overall doability: **high** (T0 over tables that exist; the reproduction code exists and needs Danielle's own pass)

| Direction | Workshop-paper impact | Notes |
|---|---|---|
| DCARD-1 provenance ledger | Medium–High | Five entries already; the pattern is the paper. |
| DCARD-2 reproduction | High | Only after Danielle personally validates the verifier; definition-matching pass is the gate for any "fails" framing. |
| DCARD-3 coverage ledger | Medium | Hygiene that every other project needs; low standalone value. |
| DCARD-4 release | High for a resource track | The publishing pipeline exists; the decision is flipping the dataset public. |
| DCARD-opt-1 own-mixture CE | Medium–High | Reconstructs the missing quantity; feeds ANN and REC. |
| DCARD-opt-2 LR validation | Medium | Converts a private caveat into paper material. |
| DCARD-opt-3 methods framework | Medium | Response's judgment: workshop-sized on its own; unproven. |

**Likely paper shape.** DCARD-1 + DCARD-3 + DCARD-4 as the resource paper; DCARD-2 as its
validation section; the thesis sentence "an eval suite used as a training-dynamics
suite, and what it takes to make that valid."

## 3. Infrastructure sequence

1. **Danielle's own pass over the verifier** (`src/datadec/paper/verifiers/` on `main`):
   read, debug, rerun; fix the degenerate compute-matching predicate (bucketed in
   log-compute space or interpolated; tolerance predeclared and swept); add the liveness
   guard.
2. **Coverage/abnormality ledger (DCARD-3)** from the existing tables, including the
   instance-derived view for 750M.
3. **Provenance ledger (DCARD-1)** as a versioned document beside `configs/catalog.toml`,
   with the manifest/composition module (REC-a) supplying entry (a).
4. **Definition-matching pass** for the proxy-metric claims against the paper's released
   analysis code.
5. **Own-mixture CE (DCARD-opt-1)** once REC-b exists.
6. **Release (DCARD-4):** flip the HF dataset public with the paper.

Shared infrastructure (keep in sync with `recipe-featurization.md` REC-a/REC-b and
`annealed-readouts.md` ANN-4 matcher): the manifest/composition module, the shard sampler,
and the compute-/loss-matched pairing utility.

---

## 4. External assessments

### 2026-08-22 — a published downstream consumer of DataDecide to track

From Danielle's SciSpace literature review on small-scale evaluation metrics
(record in `../topics/reference/small-scale-evaluation-metrics-literature.md`; Danielle-supplied ID arXiv 2605.18607; author list unresolved across the two review versions). Patel et al. 2026 use the 25 DataDecide corpora and the 1B target
rankings as a data-selection benchmark and report beating the suite's own proxies
(decision accuracy > 0.85 at ~10⁻⁵ target compute). For the data card this is (1) a
consumer whose numbers can be re-derived from the validated tables — a second
reproduction target after the original paper, and a check on whether their ground-truth
ranking used the nominal-compute or label-as-token-share assumptions this card
corrects; (2) evidence for the "eval suite used as a decision benchmark" framing.
Unverified beyond the agent summaries.

### 2026-08-22 — the validation section's thesis and its three-way classification

*Provenance caveat: the reproduction numbers cited here come from agent-written verification code that Danielle has not yet personally read, debugged, run, or analyzed; treat them as flags for where to look first, not as findings (her statement in `../topics/reference/datadecide-data-pipeline.md`).*

After the "directionally consistent" (4) and "not reproduced" (6, one misclassified as
not assessable) batches of the reproduction: the paper's quantitative headline results
reproduce (0.80 decision accuracy, compute-reliability trend, task-difficulty spread,
spread-to-noise), while its qualitative narrative about metric families and curve
shapes largely does not (raw-likelihood dominance at small scales, raw-plateau /
penalized-converge, BoolQ-only-at-1B, SocialIQA plateau shape; SocialIQA "low
reliability" a threshold quibble at 0.8233 vs. 0.80). Charitable framing for the data
card: the decision-making core survives independent reproduction; the descriptive
glosses don't. Required before any claim is framed as a contradiction: a three-way
distinction — *fails on cleaned data* / *depends on definitional choices the paper did
not pin down* / *not assessable* — and a definition-matching pass against the paper's
released analysis code for the proxy-metric failures, since the pipeline's own
choices (source precedence, legacy-seed exclusion, schema normalization) and unpinned
metric definitions could manufacture divergence. If they survive, major findings; if
not, the finding is that the claims are irreproducible-as-stated because
operationalizations were never published. Validation-methodology elements worth
showcasing: frozen predicates, sensitivity analyses around thresholds, atomic
decomposition of conjunctive claims (the Norm Correct Prob 0.916 / Margin 0.360 split
is the model case), the `not_assessable` category, and a predicate-liveness guard (see
`annealed-readouts.md` §4). The response rated this framework a workshop-sized
contribution to reproduction practice in its own right (its judgment, not a decision).

### 2026-08-22 — the validation report and a coverage/abnormality ledger as data-card components

*Provenance caveat: the reproduction numbers cited here come from agent-written verification code that Danielle has not yet personally read, debugged, run, or analyzed; treat them as flags for where to look first, not as findings (her statement in `../topics/reference/datadecide-data-pipeline.md`).*

Danielle had an agent reproduce the DataDecide paper's claims from the processed tables
(`docs/paper-validation-report.md` on `main`: 27 reproduced + 3 approximately reproduced
claim records, with the distinctions claim-record vs. independent discovery, strict vs.
approximate thresholds, and "0.02 seed SD occurs for some recipes" vs. "global maximum").
Two additions to the data-card scope from the response: (1) the claim-by-claim
validation report is a first-class component — which published claims reproduce from
the cleaned tables, with operationalizations pinned — and it de-risks every downstream
analysis (they run on tables that reproduced the headline 0.8033 150M→1B result). The
report should distinguish "claim reproduces" from "claim's operationalization is
informative" (the crossover count is the example). (2) Danielle's "there are definitely
some dataset abnormalities, like 750M only has 1 seed that trains fully I think"
(unverified; the 750M aggregate-table truncation is already in
`../open-questions-answered.md`) → an automated **coverage and abnormality ledger**
(recipe × size × seed × step cells present, early-terminated, known-issue), with every
downstream analysis declaring exclusion rules against it, and published numbers whose
support runs through thin cells flagged. Provenance list now: labels≠token shares,
nominal-vs-exact compute, unrecoverable LR, possibly-absent training loss, incomplete
seed replication. Candidate program framing sentence from the response: the original
paper's statistics are computed without a noise model and the portfolio recomputes them
with one — "DataDecide with error bars."

### 2026-08-22 — own-mixture held-out CE as a reconstructed training-loss analog

Follow-on from the same conversation: for each recipe, hold out a sample of its own
mixture drawn via the REC-a manifest/sampler and forward-pass the released checkpoints
over it. This gives an own-mixture held-out cross-entropy at checkpoint cadence — the
closest well-defined analog of training loss (minus batch noise and the moving-mixture
confound) — and as a by-product the cross-loss matrix (every recipe's model on every
recipe's mixture) that REC's similarity features want. Candidate fourth provenance-ledger
entry: training loss is absent from the released artifacts except sparsely at 150M–1B in
the scaling-law ladder CSVs; whether the authors could supply more is unconfirmed
(Danielle is checking). The response's broader thesis candidate: "DataDecide is an eval
suite being used as a training-dynamics suite; here is what it takes to make that valid."

### 2026-08-22 — The data-card thesis as a pattern of three divergences

From a conversation reviewing the `datadec` repository state (record in
`../topics/reference/datadecide-data-pipeline.md`). The data-card / composition paper
this doc's REC-a feeds has three independent, already-found cases where the suite's
self-description and its ground truth diverge: (1) mixture labels are shard-file
fractions, not token shares (this doc, §1); (2) the raw scaling-law exports encode
nominal-parameter rather than exact-parameter compute (caught by
`verify_preprocessed_derivations.py`); (3) learning-rate schedules are not recoverable
from any published artifact — Danielle's derivations come from the OLMo repo, issues,
Drive docs, and the paper, with the authors unable to confirm details of the sweep. The
response's framing: "the pattern is the paper," and each downstream analysis paper cites
the data card for its cleaned inputs. Action it implies for REC: write the LR-provenance
narrative into the data-card outline now, while the search trail is reconstructible.
Coverage fact settled the same day: OLMES detail tables are processed and published
(private HF dataset) for all 25 recipes.


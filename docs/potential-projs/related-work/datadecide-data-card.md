# datadecide data card — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`datadecide-data-card.md`](../datadecide-data-card.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*Recall corpus for `datadecide-data-card.md` (DCARD). Highest-recall inventory of every
paper, method, protocol, or named prior-art item on record anywhere in this repository that
is possibly relevant to this project. Errs toward inclusion; one line per item with its repo
source. The literature thread here is genuinely thin — much of what follows is
record/protocol material rather than citations, included because recall is the point.
Nothing is verified; all reproduction numbers come from agent-written verification code
Danielle has not personally read, debugged, run, or analyzed. No positioning claims.*

**The suite's own artifacts (the object of the card)**

- **Magnusson et al., *DataDecide: How to Predict Best Pretraining Data with Small
  Experiments*** (Ai2, ICML 2025, arXiv 2504.11393; also a COLM PDF) — 25 corpora varying
  source/dedup/filtering up to 100B tokens, sizes to 1B, 3 seeds; ranking at 150M predicts
  the 1B best dataset ~80% of the time, beating 8 scaling-law baselines; continuous
  likelihood proxies make MMLU/ARC/HellaSwag/MBPP/HumanEval >80% predictable at 0.01%
  compute. The object of DCARD-2's reproduction and of every provenance-ledger entry; the
  ledger records its metric definitions as cited from the HTML version (agent-supplied) —
  (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/topics/reference/datadecide-data-pipeline.md;
  docs/litreview/citation-verification-ledger.md, row 2504.11393 tagged DCARD, TINY).
- **OLMES, *A Standard for Language Model Evaluations*** (NAACL Findings 2025 per the record)
  — the evaluation standard DCARD-1(e) proposes to pin every column definition against;
  formats were standardized because of small-model format sensitivity — (source:
  docs/topics/reference/datadecide-data-pipeline.md;
  docs/topics/reference/evaluation-methodology-literature.md).
- **The EleutherAI multiple-choice-normalization blog post** — cited in the metric-column
  reconstruction as one of the definitional sources — (source:
  docs/topics/reference/datadecide-data-pipeline.md).
- **arXiv 2407.21072** — a length-normalization / multiple-choice-scoring paper cited for
  `acc_raw` vs. the per-length accuracies; the record explicitly does not know its title.
  Agent-supplied, unverified — (source: docs/litreview/citation-verification-ledger.md, row
  2407.21072 tagged DCARD; docs/topics/reference/datadecide-data-pipeline.md).
- **The oe-eval / OLMES implementation** — named as the only place several column definitions
  are actually answerable from (not the paper); the repo's `TaskEvalMetrics` docstrings should
  cite it — (source: docs/potential-projs/datadecide-data-card.md §4;
  docs/topics/reference/datadecide-data-pipeline.md).

**Reporting-protocol citations**

- **Paloma per-domain bits-per-byte protocol (2312.10523)** — recorded as the form to carry
  into the released PPL tables (which are per-token cross-entropy under one tokenizer) so
  they are comparable to other suites and tokenizer-free models, with the tokenizer and
  byte-count convention stated in the card. Claude-added in the ledger, unverified — (source:
  docs/potential-projs/datadecide-data-card.md §4, 2026-08-22;
  docs/topics/reference/loss-alternative-metrics-literature.md, missing-canon list;
  docs/litreview/citation-verification-ledger.md).
- **Bits per byte / per character generally — Biderman et al., *Lessons from the trenches*
  (2405.14782)** — the standard tokenizer-independent normalization the BPB recommendation
  rests on; also ByteFlow, SuperBPE, MrT5, and the Script Tax BPC study as users — (source:
  docs/topics/reference/loss-alternative-metrics-literature.md).
- **Tokenization-marginal likelihood (Cao & Rimell, EMNLP 2021; Vieira et al. 2412.03719;
  Takahashi et al. 2019 on incommensurable per-char/per-word perplexities)** — the robustness
  layer above BPB if the card wants a tokenizer-free statement — (source: same).
- **Compute-unit conventions** — petaFLOP/s-day (OpenAI's "AI and Compute" unit,
  ≈ 8.64 × 10¹⁹ FLOP) vs. plain scientific-notation FLOP (Epoch AI convention); the card
  (DCARD-4) should state the convention: FLOP via 6·N·D with which N (nominal vs. exact
  parameter count). Includes the Claude-added correction that the PyArrow failure was int64
  overflow, not float64 overflow — (source:
  docs/topics/reference/datadecide-data-pipeline.md, compute-units entry).

**The published downstream consumer**

- **Patel, Reddy, Mosbach & Bahdanau 2026, *Forecasting Downstream Performance of LLMs With
  Proxy Metrics*** (arXiv 2605.18607; Mila/McGill + ServiceNow) — uses the 25 DataDecide
  corpora and the 1B target rankings as a data-selection benchmark and reports beating the
  suite's own proxies (decision accuracy > 0.85 at ~10⁻⁵ target compute). Recorded as (1) a
  second reproduction target whose numbers can be re-derived from the validated tables and a
  check on whether their ground-truth ranking inherited the nominal-compute or
  label-as-token-share assumptions this card corrects, and (2) evidence for the "eval suite
  used as a decision benchmark" framing. Per the SciSpace review, unverified beyond agent
  summaries; one review version fabricated the author list — (source:
  docs/potential-projs/datadecide-data-card.md §4, 2026-08-22;
  docs/topics/reference/small-scale-evaluation-metrics-literature.md).

**The noise-model precedent behind "DataDecide with error bars"**

- **Heineman et al., *Signal and Noise*** (ledger ID 2508.13144, Claude-added) — appears in
  the repo as the noise-model precedent behind the recorded program framing sentence: the
  original paper's statistics are computed without a noise model and the portfolio recomputes
  them with one — (source: docs/potential-projs/datadecide-data-card.md §4;
  docs/topics/reference/evaluation-methodology-literature.md).
- **The fixed-checkpoint variance rule** — for OLMES-style loglikelihood evaluation,
  re-evaluating a fixed checkpoint with new seeds buys nothing; the variance of interest is in
  training (seed, data order, init). Constrains what error bars the card can honestly carry —
  (source: docs/topics/reference/evaluation-methodology-literature.md, 2026-08-18).

**The provenance-ledger entries (DCARD-1) — the divergences themselves**

- **(a) Mixture labels are shard-file fractions, not token shares** — the DCLM/Dolma 25/50/75
  recipes are 43/69/87% DCLM by tokens; REC-a's manifest/composition module supplies the
  evidence — (source: docs/potential-projs/datadecide-data-card.md §1/§4;
  docs/topics/reference/datadecide-data-pipeline.md).
- **(b) Raw scaling-law exports encode nominal- rather than exact-parameter compute** — caught
  by `verify_preprocessed_derivations.py` — (source: same).
- **(c) Learning-rate schedules are recorded in no published artifact** — derived from the OLMo
  repo, issues, Drive docs, and the paper; the authors could not confirm sweep details. Action
  recorded: write the LR-provenance narrative now, while the search trail is reconstructible —
  (source: same).
- **(d) Training loss absent at checkpoint cadence** — present only sparsely at 150M–1B in the
  scaling-law ladder CSVs; what the authors can supply is unconfirmed — (source: same;
  docs/open-questions-answered.md, open item "Training-loss availability").
- **(e) Incomplete seed replication at some sizes** — the 750M recollection ("only 1 seed that
  trains fully") is unverified; the 750M aggregate-table truncation at step 26,250 while the
  instance table is not truncated is established — (source: same;
  docs/open-questions-answered.md, 2026-08-21).
- **(e′) Metric definitions pinned to the evaluation code** — one line per column with the
  formula, the scoring rule it belongs to, and whether the aggregate is rebuildable from
  instance details; `configs/olmes.toml` already records
  `not_reproducible_from_details = ["bits_per_byte_corr"]`. Danielle's own reading —
  `predicted_index_*`, `correct_choice`, and the `uncond_*` columns are per-item building
  blocks, not reportable metrics — is the first row of that table — (source:
  docs/potential-projs/datadecide-data-card.md §4, 2026-08-22;
  docs/topics/reference/datadecide-data-pipeline.md).

**The metric-column reconstruction (what is settled vs. guessed)**

- **Settled on data**: `correct_prob = exp(sum_logits_corr)` (checked on two released rows:
  −33.2023 → 9.71e-6; −34.7956 → 5.74e-6) — identical rankings, different magnitudes for
  regression — (source: docs/topics/reference/datadecide-data-pipeline.md).
- **Family structure (response's claim)**: five scoring rules — raw sum of log-probs,
  unconditional-normalized, per-byte, per-char, per-token — each yielding a
  `predicted_index_*` and an `acc_*`, with continuous companions `correct_prob*`,
  `norm_correct_prob*`, `total_prob*` (read by the paper as domain exposure), `margin*`, and
  `bits_per_byte_corr` — (source: same).
- **`norm_correct_prob`** is a per-item ratio P(correct | ctx) / Σ_options P(option | ctx)
  averaged over items — *not* the ratio of aggregate `correct_prob` to `total_prob` —
  (source: same).
- **Open definitional questions (answerable only from code)**: whether `uncond_correct_prob`
  is a probability difference or the lm-eval/OLMES log-ratio; whether `correct_choice` is
  binary or the gold index (the repo schema types it float); that
  `correct_prob_per_char = exp(sum_logits_corr / chars)`, a per-character geometric mean, not
  probability ÷ length; the bpb byte-count and log-base conventions (the bpb sanity check in
  the conversation was circular) — (source: same).
- **Danielle's proposed column `uncond_correct_prob`** as an additional continuous proxy
  candidate (response endorsed on the paper's "continuous beats discrete at small scale"
  finding, no evidence offered) — (source: docs/potential-projs/tiny-scale-measurement.md §4;
  docs/topics/reference/datadecide-data-pipeline.md).
- **Repo facts**: `src/datadec/data/ingest/metrics.py` is the typed column schema
  (`TaskEvalMetrics`; `correct_choice: float`); `configs/olmes.toml` lists reproducible
  aggregate columns; no metric-definition document exists in the repo — (source: same).
- **The metric-column table is also the input for the per-example comparison workbench**
  (marimo + Altair), which depends on DCARD-1(e) landing first — (source:
  docs/topics/reference/experiment-tooling.md).

**DCARD-2: the reproduction record (agent-generated; flags, not findings)**

- **Reproduced headline results**: 0.8033 150M→1B decision accuracy; the compute-reliability
  trend; task-difficulty spread; spread-to-noise ρ 0.798 — (source:
  docs/potential-projs/datadecide-data-card.md §1/§4;
  docs/topics/reference/datadecide-data-pipeline.md).
- **Failed / directionally-inconsistent qualitative claims**: raw-likelihood dominance at
  small scales (2.38% vs. a >50% predicate, robust across compute bands 3.30% / 2.38% /
  1.77%); raw-plateau / penalized-converge failing both halves (max raw slope 0.0914; gap
  growing 0.023 → 0.134); BoolQ-nontrivial-only-at-1B strongly failing (108 nontrivial points,
  85 below 1B, final 1B 0.7867); SocialIQA plateau shape; SocialIQA "low reliability" a
  threshold quibble at 0.8233 vs. 0.80; margin tracking accuracy at 0.360 vs. Normalized
  Correct Probability at 0.916 — (source: same).
- **The three-way classification required before any failure is framed as a contradiction** —
  *fails on cleaned data* / *depends on definitional choices the paper did not pin down* /
  *not assessable* — plus a definition-matching pass against the paper's released analysis
  code, since the pipeline's own choices (source precedence, legacy-seed exclusion, schema
  normalization) and unpinned metric definitions could manufacture divergence — (source: same).
- **The validation report on `main`** (`docs/paper-validation-report.md`): 27 reproduced + 3
  approximately reproduced claim records, with the distinctions claim-record vs. independent
  discovery, strict vs. approximate thresholds, and "0.02 seed SD occurs for some recipes" vs.
  "global maximum" — (source: docs/potential-projs/datadecide-data-card.md §4;
  docs/topics/reference/datadecide-data-pipeline.md).
- **"Claim reproduces" vs. "claim's operationalization is informative"** — the crossover count
  is the named example: 15,523 crossovers with all 300 recipe pairs crossing at least once is
  near-guaranteed under stable ordering plus noise; the noise-aware redefinition (exceeds the
  per-task seed-noise floor and persists k consecutive checkpoints) is the fix — (source:
  docs/potential-projs/datadecide-data-card.md §4; docs/potential-projs/irt-reanalysis.md §4).
- **The degenerate compute-matching predicate** — to be bucketed in log-compute space or
  interpolated, with tolerance predeclared and swept; plus a predicate-liveness guard
  (comparison set non-empty, size reported) — (source:
  docs/potential-projs/datadecide-data-card.md §3/§1; docs/potential-projs/annealed-readouts.md §4).
- **The two-cluster silhouette finding (0.207 vs. 0.25 default; reproduced at 0.15)** and the
  sharper null that curve-shape clusters do not imply ability dimensions — a threshold-
  without-principled-basis example the card can use — (source:
  docs/potential-projs/irt-reanalysis.md §4).

**DCARD-opt-3: reproduction-practice methodology**

- **The framework elements** — frozen predicates, sensitivity analyses around thresholds,
  atomic decomposition of conjunctive claims (the Norm Correct Prob 0.916 / Margin 0.360 split
  is the model case), the `not_assessable` category, and the predicate-liveness guard. Rated
  workshop-sized by the response (its judgment, not a decision) — (source:
  docs/potential-projs/datadecide-data-card.md §1/§4).
- **Explicit gap on record: no data-card / datasheet / reproduction-practice literature is
  named anywhere in the repository**, and the resource / NeurIPS D&B framing is recorded as a
  venue judgment rather than a positioning against named resource papers. These are the two
  gaps a literature pass would fill first — (source:
  docs/potential-projs/datadecide-data-card.md §5;
  docs/topics/reference/datadecide-data-pipeline.md; docs/portfolio-rankings.md).

**Coverage facts the ledgers rest on**

- **Per-instance coverage** — all 25 recipes × 66 tasks; 3 seeds at 150M–1B, 1 seed below
  150M; per-instance tables at 4M/20M/60M/90M — (source: docs/open-questions-answered.md,
  2026-08-21).
- **Checkpoint spacing** — the 4M–8M runs have 5–10 checkpoints — (source: same;
  docs/topics/staging/datadecide-dense.md).
- **The 750M aggregate table truncated at step 26,250 while the instance table is not** —
  motivates the instance-derived view for 750M in DCARD-3 — (source:
  docs/open-questions-answered.md; docs/potential-projs/datadecide-data-card.md §3).
- **OLMES detail tables processed and published (private HF dataset) for all 25 recipes** —
  the release DCARD-4 flips public — (source:
  docs/topics/reference/datadecide-data-pipeline.md, 2026-08-22).
- **Open items** — loss-curve coverage in the scaling-law table; per-window realized mixture;
  training-loss availability; 750M seed coverage — (source: docs/open-questions-answered.md,
  "Open — not yet checked").

**DCARD-opt-1 and DCARD-opt-2 inputs**

- **Own-mixture held-out CE** — for each recipe, hold out a sample of its own mixture drawn via
  the REC-a manifest/sampler and forward-pass the released checkpoints: the closest
  well-defined analog of training loss at checkpoint cadence (minus batch noise and the
  moving-mixture confound), with the cross-loss matrix (every model on every mixture) as a
  by-product — (source: docs/potential-projs/datadecide-data-card.md §1/§4).
- **Multi-power law (Luo et al., arXiv 2503.12811, ICLR 2025)** — predicts the full pretraining
  loss curve at every intermediate step across LR schedules from a power law on the sum of
  learning rates plus decay terms; DCARD-opt-2 uses cross-recipe × cross-scale MPL fits with
  shared structure as affirmative evidence the derived LR schedules are right in every way the
  loss dynamics can see — (source: docs/topics/reference/loss-curve-forecasting.md;
  docs/potential-projs/datadecide-data-card.md §1; docs/potential-projs/annealed-readouts.md).
- **DataDecide-dense (+WSD)** — the ground-truth validation substrate for the LR derivations
  and the MPL, and the reproduction-gap measurement (faithful rerun vs. published checkpoint,
  relative to seed variance); not started, design doc gated — (source:
  docs/topics/staging/datadecide-dense.md; docs/topics/reference/datadecide-data-pipeline.md).
- **u-µP (Blake et al., arXiv 2407.17465, Danielle-supplied)** — the record notes DataDecide
  uses per-size hand-set hyperparameters, i.e. it is *not* µP-parametrized, so its cross-size
  comparisons carry the "was the small model's LR optimal" confound — a recipe-confound entry
  the card can carry — (source: docs/topics/reference/parametrization-and-hp-transfer.md).
- **Regularization inputs for many-epoch small-scale runs — Xue et al. 2305.13230; Muennighoff
  et al. 2305.16264 (data-constrained scaling)** — cited for fixing the regularization recipe
  and recording epochs per run — (source: docs/topics/staging/datadecide-dense.md;
  docs/topics/reference/regularization-literature.md).

**The wider proxy-metric landscape DCARD's tables feed (all agent-supplied, unverified)**

- **Gadre et al. 2403.08540; FLP 2410.08527; model ladders 2412.04403; observational scaling
  laws 2405.10938; NeuNeu 2601.19831; Ye et al. 2305.14947; Pechi et al. 2305.17266; ADO
  2410.11820; Krajewski et al. 2512.08894; Ali et al. 2310.08754** — the downstream-prediction
  literature the released tables are consumed by — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md; the ledger tags these
  rows TINY, IRT, EDP, DCARD).
- **Data mixture / selection laws: AutoScale 2407.20177; UtiliMax / MEDU 2501.11747; D-CPT
  2406.01375; BiMix 2405.14908; data mixing laws 2403.16952; optimal-mixture laws 2507.09404;
  effective tokens = diversity × syntheticity 2410.03083; quality-aware Q 2510.03313** — the
  data-selection setting the 25-corpus ranking sits in — (source: same).
- **Contamination: time-travel detection 2308.08493; C2LEVA 2412.04947** — relevant to a
  coverage/abnormality ledger that wants to flag validity as well as presence — (source: same).
- **Emergence: Wei et al. 2206.07682 vs. Schaeffer et al. 2304.15004; proxy tasks 2412.07111**
  — the reading under which the card's metric-family findings are interpreted — (source: same).
- **Loss-replacement family (LongPPL 2410.23771; PPLqa 2411.15320; Rho-1 2404.07965;
  Diff-eRank 2401.17139; Matrix Nuclear-Norm 2410.10672; information capacity 2511.08066;
  entropy-estimation 2511.10618; Delétang et al. 2309.10668)** — the ledger tags all of these
  as feeding TINY, DCARD, IRT — (source:
  docs/topics/reference/loss-alternative-metrics-literature.md;
  docs/litreview/citation-verification-ledger.md).

**Estimand and provenance discipline (method, not literature)**

- **The estimand-discipline checklist** — model version/fingerprint, exact prompt, decoding
  config, post-processing; don't mix snapshots; block methods for batched collection —
  explicitly mapped to DCARD as "the provenance ledger's checklist restated for evaluation" —
  (source: docs/topics/reference/estimation-and-calibration-methods.md, relevance map).

**Program placement and framing sentences on record**

- **"DataDecide with error bars"** — the candidate framing sentence for the slice of the
  program this anchors; **"an eval suite used as a training-dynamics suite, and what it takes
  to make that valid"** — the thesis sentence; **"the pattern is the paper"** — the response's
  framing of the three-divergence pattern — (source:
  docs/potential-projs/datadecide-data-card.md §1/§4;
  docs/topics/reference/datadecide-data-pipeline.md).
- **Rankings** — workshop-sized #1 as the REC-a data card; a Tier 3 component; "measured data
  cards" named as indispensable hygiene and the most citable single table; full-conference #2,
  "What Is Actually in DataDecide" — (source: docs/portfolio-rankings.md).
- **Consumer list** — every DataDecide-facing project (IRT, TRJ, ANN, REC, TINY, EDP, ELI)
  cites this card for its cleaned inputs — (source:
  docs/potential-projs/datadecide-data-card.md header; docs/potential-projs/README.md).
- **Shared infrastructure** — the manifest/composition module and shard sampler (REC-a/REC-b),
  the compute-/loss-matched pairing utility, and ANN-4's matcher — (source:
  docs/potential-projs/datadecide-data-card.md §3).

**Provenance caveats carried from the records**

- Every reproduction number above comes from agent-written verification code Danielle has not
  personally read, debugged, run, or analyzed — flags for where to look first, not findings
  (her statement) — (source: docs/topics/reference/datadecide-data-pipeline.md;
  docs/potential-projs/datadecide-data-card.md header).
- The metric-column reconstruction is a cited-browsing conversation with several
  plausible-but-wrong claims explicitly listed; it is not a substitute for a
  metric-definition document — (source: docs/topics/reference/datadecide-data-pipeline.md).
- The SciSpace review behind the Patel et al. entry fabricated the author list in one version
  and has swapped bibliography entries; prefer version 1 — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md, intake notes).
- Every arXiv ID above traces to `docs/litreview/citation-verification-ledger.md`, where rows
  are marked *agent-supplied* or *Claude-added* and **nothing is verified**.

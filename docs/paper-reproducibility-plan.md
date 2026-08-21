# DataDecide paper finding validation plan

Status: active hard-cutover redesign

Revised: August 21, 2026

## Goal

Use the postprocessed DataDecide data published as `drotherm/dd_parsed` to make
our best independent attempt at recomputing every empirical number,
comparison, trend, and plot reported in the DataDecide paper.

The paper supplies the target claim and method description. `dd_parsed`
supplies the analysis evidence. For every assessable finding, report the
paper's value or relationship beside our computed result, the exact rows and
operational choices used, and whether the result agrees.

This is analytical validation of reported findings. It is not an attempt to
recreate the historical training environment, prove release completeness, or
retrain the models.

## Hard scope boundaries

### Training is permanently out of scope

Do not pretrain, continue training, fine-tune, reconstruct training corpora, or
plan a future training rerun. A claim that appears to require training must
first be reconsidered: the likely task is to analyze existing results or to
classify a training-design statement as descriptive metadata. If it genuinely
cannot be assessed without training, record `not_assessable_from_dd_parsed`.
Training is not a missing input to acquire.

### Checkpoints are optional read-only evidence

Released model checkpoints may be downloaded only to resolve a specific,
valuable uncertainty through metadata inspection, inference, or evaluation.
Checkpoint use must never include parameter updates. Evaluation reruns are a
targeted fallback after confirming that the required result cannot be derived
from `dd_parsed`; they are not a prerequisite for the main workflow.

### Exact historical provenance is not a gate

Do not require an exact author-repository commit, historical training manifest,
paper-final checkpoint number, clean Git tree, or pinned `dd_parsed` revision
before computing a result. Record available identities for traceability, but
never convert their absence into a scientific blocker.

The current author repository and this repository's current catalog are not
assumed to match the configurations used for the paper. Differences may be
useful metadata notes, but they do not contradict an empirical finding unless
the provided analysis data itself directly falsifies that finding.

### Author-produced result tables are targets, not evidence

Files under `published-results/` may help extract a reported target or diagnose
a mismatch. They must not be used as the computational input for claiming that
the same result was independently reproduced. Independent computations use the
lower-level aggregate, scaling-law, task, instance, or choice tables.

Enforce this separation structurally: static claim-target extraction may read
paper result tables, but validation modules and run orchestration must not open
or import them. Computation receives normalized target values through the claim
contract. Any debugging comparison with `published-results` is a separate
manual command whose output cannot enter a validation run.

## Canonical analysis inputs

The Hugging Face publication contract at `configs/publishing.toml` defines the
`dd_parsed` layout. Local development uses the corresponding processed files
under `data/processed/`; authenticated remote loading must expose the same
logical tables.

- `ppl.parquet`: checkpoint-level perplexity and model-loss evidence.
- `olmes.parquet`: aggregate checkpoint/task evaluation metrics.
- `scaling-law/evaluations.parquet`: normalized scaling-law evaluations.
- `scaling-law/checkpoint-losses.parquet`: normalized checkpoint losses.
- `olmes-details/<recipe>/tasks.parquet`: task-level evaluation details.
- `olmes-details/<recipe>/instances.parquet`: instance-level observations.
- `olmes-details/<recipe>/choices.parquet`: answer-choice likelihood evidence.
- `published-results/**`: paper targets and debugging references only.

The local processed mirror is sufficient for the first implementation waves.
Remote access and checkpoint downloads are not initial blockers.

## Claim taxonomy

Retain stable paper locators and atomic claim IDs, but classify claims by what
this exercise can meaningfully validate.

### Primary validation targets

- `empirical_numeric`: an exact or approximate reported number.
- `empirical_comparison`: an ordering, difference, equivalence, crossover, or
  directional relationship.
- `empirical_trend`: a relationship over scale, compute, task, metric, or
  checkpoint.
- `empirical_plot`: the data and semantic content of a paper figure or panel.

### Supporting claims

- `method_definition`: a formula, aggregation, selection, or plotting rule
  needed to compute primary targets.
- `descriptive_metadata`: model, sequence-length, suite-design, recipe, release,
  or training-description statements. Compare with `dd_parsed` metadata when
  useful, but keep discrepancies separate from empirical finding outcomes.
- `external_background`: literature attribution or contextual framing. Preserve
  the citation trace; do not independently audit the cited work.
- `normative_or_future`: recommendations, impact statements, limitations, or
  future-work proposals. Do not force these into empirical verdicts.

The main report centers primary validation targets. Supporting claims appear
only where they affect interpretation or document a useful discrepancy.

## Attempt contract

Each empirical claim may have one or more named attempts. An attempt records:

- stable claim ID and paper locator;
- verbatim or normalized paper target;
- `dd_parsed` table and columns used;
- identities of the actual inputs used, recorded as local Parquet hashes and,
  when available, the remote dataset revision; identity capture is traceability
  metadata and never a historical-provenance gate;
- row-selection rules, including recipes, tasks, metrics, seeds, sizes, and
  checkpoints;
- transformation and aggregation order;
- our computed value, relationship, or machine-readable plot series;
- comparison rule appropriate to the paper's precision or wording;
- outcome;
- sensitivity results for consequential ambiguous choices; and
- concise diagnostics and limitations.

The default attempt is the best paper-faithful interpretation that can be
computed from `dd_parsed`. Additional attempts test reasonable alternatives;
they do not overwrite or hide the default.

## Outcomes

- `reproduced`: our result matches an exact target at the precision reported or
  satisfies the reported comparison.
- `approximately_reproduced`: our result supports an explicitly approximate
  number, trend, or visual relationship.
- `directionally_consistent`: the reported direction or ordering holds, but the
  magnitude or stronger wording does not fully match.
- `not_reproduced`: available `dd_parsed` evidence directly disagrees with the
  empirical target under the best attempt and reasonable sensitivities.
- `not_assessable_from_dd_parsed`: the required observation is absent or cannot
  be derived without prohibited training or a materially different experiment.
- `metadata_discrepancy`: paper description and available metadata differ; this
  is not an empirical finding failure.
- `descriptive_only`: no empirical comparison is appropriate.
- `external_or_background`: retained for traceability but outside validation.

Do not use missing exact provenance, an absent historical checkpoint number,
or current repository drift as a `not_reproduced` result.

## Default operational choices

### Checkpoints

When the paper says `final` but its exact table step is absent, use the latest
single checkpoint that has a complete Cartesian grid across the entire
predeclared comparison universe for that attempt: every required recipe, seed,
task, and metric must use the same step. Record the actual step and completeness
counts. Never select a different `latest` step per row or choose a step after
examining the result.

For final-checkpoint claims, always compute the two preceding common complete
checkpoints as fixed sensitivities when they exist, plus the exact paper step if
it exists. These sensitivities are selected before results are examined, not
only when the default appears material.

Never silently replace one model size, task, metric, recipe, or seed group with
another.

### Reported precision

For exact tabulated numbers, compare after applying the paper's displayed
precision and also report the unrounded difference. For words such as
`approximately`, `roughly`, or `comparable`, define and version a transparent
best-attempt predicate and its sensitivity thresholds before examining the
computed result. Ambiguity calls for a predeclared operationalized attempt, not
an automatic blocker or a post hoc threshold.

### Seeds and uncertainty

Follow the paper's stated seed grouping when those rows exist. If only a subset
is available, compute the available-data result and disclose the difference.
Report denominators, exclusions, ties, missing groups, and the standard
deviation convention. Default sample-standard-deviation choices remain
repository operationalizations when the paper is silent.

### Tasks and metrics

Follow the paper's OLMES task grouping: average MMLU subjects into one task,
then macro-average MMLU with the nine non-MMLU tasks. Use the task-specific
primary metric and the paper's proxy-metric definitions. Instance and choice
tables are preferred when a claim depends on option likelihoods, item
selection, normalization, or tie behavior.

### Comparisons and ties

Use all unordered recipe pairs in the stated comparison universe. Target ties
are outside a two-class winner comparison. Record predicted ties explicitly;
the default paper-faithful attempt counts them as incorrect, with an exclusion
sensitivity when material.

## Implementation phases

### Phase 0: hard-cut the contracts

1. Replace provenance-first evidence-boundary and blocker semantics with the
   claim taxonomy, attempt contract, and outcomes above.
2. Remove training-rerun, corpus-reconstruction, release-manifest, clean-tree,
   and exact-checkpoint qualification requirements from the analysis path.
3. Preserve paper locators, active-source coverage, data schema validation, and
   deterministic calculations where they remain useful.
4. Delete superseded report, figure, CLI, and test paths in the same cutover;
   do not retain competing scientific interpretations.

Exit condition: a default attempt can report a valid empirical result from the
current local `dd_parsed` mirror without historical provenance gates.

### Phase 1: triage and map the inventory

1. Reclassify all claims into the taxonomy above.
2. Identify the empirical findings and paper plot semantics that are primary
   validation targets.
3. Map each primary target to the lowest-level sufficient `dd_parsed` tables.
4. Identify method dependencies, expected values, comparison rules, and
   sensitivities.
5. Mark descriptive, external, normative, and training-only statements without
   turning them into future data-acquisition work.

Exit condition: every primary target has an executable attempt specification or
a concrete reason it is not assessable from `dd_parsed`.

### Phase 2: reproduce headline single-scale findings

1. Build mean 1B target rankings over the stated target seeds.
2. Build single-scale recipe rankings for every available prediction size,
   checkpoint, seed, task, and metric.
3. Compute pairwise decision accuracy, denominators, ties, exclusions, seed
   means, and uncertainty.
4. Validate the abstract claim that a 150M single-scale ranking gets
   approximately 80% of 1B pairwise decisions correct using the latest
   available 150M checkpoint.
5. Recreate aggregate and per-task compute-versus-decision-accuracy paper plots.

The current local data already yields `0.803333...` for the 150M primary-metric
best attempt at step 37,500 over three prediction seeds. The first cutover test
must classify this as approximately reproduced rather than blocked by the
paper table's absent step 38,157.

Exit condition: headline single-scale numbers, comparisons, and plot series are
computed and compared with the paper.

### Phase 3: reproduce metric, task, and noise findings

1. Compare primary accuracy with all continuous proxy metrics.
2. Recompute task-specific predictability and compute relationships.
3. Recompute seed-attempt uncertainty, within-recipe noise, across-recipe
   spread, crossover, and ranking-stability analyses.
4. Validate claims about easy, hard, insensitive, or predictable tasks using
   explicit operational predicates and sensitivities.
5. Recreate the corresponding paper panels and caption comparisons.

Exit condition: every single-scale, task, metric, and noise finding has a
best-attempt outcome.

### Phase 4: reproduce multi-scale and scaling-law findings

1. Implement the paper's eight fit variants from normalized evaluations and
   checkpoint losses.
2. Recompute helper-point and early-checkpoint-filtering variants.
3. Recompute relative and absolute 1B prediction error.
4. Compare multi-scale and single-scale decision accuracy over compute.
5. Recreate scaling-law figures and qualitative comparisons with transparent
   fit settings and sensitivity analyses.

Use a reasonable repository-owned optimizer configuration when the paper is
silent. Inspect pinned author source only when it materially clarifies a method;
record that influence, but do not import or execute author analysis code.

Exit condition: every scaling-law number, comparison, and plot has a computed
attempt or is genuinely not assessable from `dd_parsed`.

### Phase 5: targeted checkpoint evaluation, only if earned

For a high-value empirical claim still not assessable, determine whether a
released checkpoint plus a bounded evaluation can clarify the uncertainty.
Record the exact claim, expected new evidence, models, tasks, storage, and
compute before downloading. Skip the evaluation when it merely improves
historical provenance or would duplicate available `dd_parsed` evidence.

Checkpoint-evaluation results are supplemental and remain separate from the
primary `dd_parsed` attempt. They cannot upgrade
`not_assessable_from_dd_parsed` into a `dd_parsed` reproduction; instead report
a separately labeled checkpoint-evaluation finding.

Exit condition: checkpoint use is limited to specific unresolved findings and
contains no training.

### Phase 6: report and review

1. Generate a compact report organized by empirical finding family.
2. Show paper target, our result, difference, outcome, method, and sensitivity
   side by side.
3. Generate small paper-analog plots plus a summary of outcomes.
4. Keep metadata discrepancies in a separate appendix.
5. Perform one full adversarial review focused on analytical correctness,
   selection bias, accidental use of author results, overstatement, and plot
   semantics.
6. Remediate confirmed correctness defects once and regenerate outputs.

Exit condition: a reader can tell which paper findings replicate from
`dd_parsed`, which are only directionally consistent, which do not replicate,
and which cannot be assessed without leaving scope.

### Phase 7: correct the reusable skill

Update `verify-paper-claims` in its existing dotfiles PR only after this revised
workflow operates end to end. The skill must prioritize analytical replication
from available evidence, make best attempts under transparent
operationalizations, and treat training as an explicit optional scope boundary
that is prohibited for this DataDecide effort.

Run a new independent cold-agent test against a held-out analytical fixture.
Do not preserve provenance-first blocker behavior from the superseded workflow.

## Deliverables

- this canonical plan;
- a reclassified static claim inventory;
- versioned attempt and comparison contracts;
- repository-owned `dd_parsed` analyses;
- machine-readable plot series;
- small paper-analog figure outputs;
- a compact finding-comparison report;
- tests that pin calculations and outcome semantics;
- one reviewed DataDecide PR with a single final analysis path; and
- a corrected, cold-tested dotfiles skill PR.

## Validation requirements

The completed system must prove that:

- no code path trains or updates model parameters;
- every attempt records the identities of the `dd_parsed` files it actually
  read without requiring an unavailable historical revision;
- primary results are computed from lower-level `dd_parsed` tables rather than
  copied from `published-results`, and computation code cannot access those
  author-result paths;
- reported row selections and aggregations are deterministic and inspectable;
- final-checkpoint selection uses one predeclared common complete step across
  the comparison universe and always records fixed preceding-step sensitivity;
- seeds, denominators, exclusions, missing groups, and ties remain visible;
- paper targets and our results are stored separately;
- approximate and qualitative outcomes use predicates frozen before results are
  examined;
- metadata discrepancies do not masquerade as empirical contradictions;
- paper-analog plots are backed by machine-readable series; and
- generated reports do not recompute scientific results.

## Delivery strategy

Revise DataDecide PR #43 in place rather than adding a competing validation
stack. Publish coherent checkpoints regularly while the hard cutover proceeds.
Do not merge the current provenance-first report as the final result.

Revise dotfiles PR #81 only after the corrected DataDecide workflow is reviewed.
The two repositories remain separate PRs because the skill has a distinct
owner and validation path.

# Paper Validation Contract

This document is the shared implementation contract for the analytical
validation workflow described in `paper-reproducibility-plan.md`. It fixes the
persisted vocabulary and dependency boundaries used by the claim registry,
analyses, run bundles, report, and figures.

## Scientific boundary

Validation computations may read only normalized `dd_parsed` inputs declared
by the validation config. They must not import or open `published-results`.
Paper targets are static claim-contract values. Author result tables may be
consulted only by a separate manual target-extraction or debugging tool whose
output cannot enter a validation run.

No validation module exposes training, parameter updates, corpus reconstruction,
or training-rerun planning. Released-checkpoint evaluations, if separately
authorized later, are supplemental and cannot change a primary `dd_parsed`
outcome.

## Closed vocabulary

`ClaimKind` has exactly these values:

- `empirical_numeric`
- `empirical_comparison`
- `empirical_trend`
- `empirical_plot`
- `method_definition`
- `descriptive_metadata`
- `external_background`
- `normative_or_future`

`ValidationOutcome` has exactly these values:

- `reproduced`
- `approximately_reproduced`
- `directionally_consistent`
- `not_reproduced`
- `not_assessable_from_dd_parsed`
- `metadata_discrepancy`
- `descriptive_only`
- `external_or_background`

The provenance-first vocabulary is deleted, including evidence boundaries,
method provenance, blocker kinds, blocker verdicts, source-only matches,
qualification status, and clean-tree state.

## Static contracts

`docs/paper/claims.toml` remains the canonical source-linked inventory. Each
claim contains its stable source fields plus:

- `kind: ClaimKind`
- `family: str`
- `paper_target: JSON scalar or null`
- `attempt_ids: tuple[str, ...]`
- `method_dependency_claim_ids: tuple[str, ...]`
- `citation_keys: tuple[str, ...]`
- `supporting_outcome: ValidationOutcome | null`
- `non_assessable_reason: str | null`

Primary empirical claims have at least one attempt ID unless their only outcome
is `not_assessable_from_dd_parsed`. Supporting claims have no executable attempt
unless an empirical claim references them as a method dependency. Source-region
records and source locators remain unchanged.

`configs/paper_validation.toml` is the only executable paper-validation config.
It replaces `configs/paper_reproduction.toml` without a compatibility alias. It
contains input table/column declarations, typed attempt specifications,
versioned comparison rules, checkpoint policy, fixed sensitivity policy,
analysis policies, and output paths. Predicates and their threshold grids are
fixed in this file before computation.

Each `AttemptSpec` contains:

- stable attempt ID and claim ID;
- `default: bool` and optional parent attempt ID;
- closed analysis ID;
- logical input tables and columns;
- declared recipe, seed, task, metric, size, and checkpoint universe;
- ordered transformation IDs;
- versioned comparison-rule ID;
- fixed sensitivity IDs; and
- plot-series IDs it is expected to produce.

Cross-contract validation requires unique IDs, exact claim/attempt references,
one default attempt per assessable primary claim, existing method-dependency
claims, and no executable attempt for a nonempirical claim.

## Persisted result models

Persisted/config boundaries use frozen Pydantic models. Internal calculation
values use frozen slotted dataclasses.

`PaperTarget` stores claim ID, family, kind, source locator/text, and the
normalized paper target. `AttemptResult` stores target and computed result in
separate fields and includes:

- attempt and claim IDs, default/sensitivity role, and parent attempt;
- comparison-rule ID and version;
- ordered transformations;
- row selections and actual input identities;
- checkpoint selections and completeness counts;
- computed JSON value and unrounded difference where numeric;
- seeds, denominator, exclusions, missing groups, target ties, predicted ties,
  SD, and DDOF;
- outcome, diagnostics, limitations, and plot-series IDs.

`RowSelection` identifies the logical table, columns, typed predicates, local
Parquet SHA-256, optional remote dataset revision, selected row count, and a
canonical selected-key SHA-256. Input identity is traceability metadata and
never a qualification gate.

`CheckpointSelection` records requested meaning, the rule
`exact | latest_common_complete | preceding_common_complete`, actual step,
declared completeness dimensions, expected group count, and selected group
count. A final-checkpoint default uses one common complete step across the
entire declared comparison universe. Its fixed sensitivities are the two
preceding common complete steps when present and the paper step when present.

`MetadataDiscrepancy` is separate from empirical attempts. It records the paper
locator/value, available metadata source/value, and an explanatory note.

`PlotSeries` is the only scientific plotting input. It records a stable series
ID, figure/panel, semantic kind, axes/scales/units, dimensions, measures,
attempt ID, actual checkpoint, counts, and ordered finite points. Empty
paper-analog series are prohibited.

## Run format 2

One immutable run directory contains exactly:

```text
data/paper-validation/runs/<run-id>/
  manifest.json
  targets.json
  attempts.json
  plot-series.json
```

`AnalysisManifest` records `run_format = 2`, run ID/timestamps, optional
code/runtime trace, actual input identities, and content identities for the
other three bundle files. It contains no scientific qualification state.
Bundle creation retains canonical JSON, unique staging directories, atomic
no-replace finalization, deterministic ordering, tamper detection, and
same-run-ID collision behavior. Loading validates all cross-references.

The generated views are:

```text
docs/paper-validation-report.md
docs/paper/validation-figures/*.svg
```

Report and figure rendering read only a completed run bundle. They never reopen
scientific inputs or recompute findings. Paper-analog figures use only persisted
`PlotSeries`; audit summaries count only default primary outcomes. Metadata
discrepancies never enter empirical outcome counts.

## Canonical API and CLI

The public workflow is:

```python
validate_repository(root, data_dir) -> ValidationSurface
run_validation(root, run_id, data_dir) -> AnalysisBundle
load_analysis_bundle(runs_root, run_id) -> AnalysisBundle
render_validation(root, run_id) -> RenderedOutputs
```

The sole CLI entrypoint is `scripts/validate_paper_findings.py` with `validate`,
`run`, and `render` commands. The prior verification CLI and generated blocker
report/figures are deleted in the same cutover.

## First required analytical slice

The first end-to-end slice is the headline single-scale result and its
compute-versus-decision series. It must:

- form the 1B target from three target seeds at common complete step 69,369;
- select 150M common complete step 37,500 over all 25 recipes, three prediction
  seeds, and 66 source tasks;
- macro-average 57 MMLU subjects into one task, then average it equally with the
  nine non-MMLU tasks;
- compare all 300 unordered recipe pairs per prediction seed;
- produce correct counts 234, 251, and 238, mean `0.8033333333333333`, and
  sample SD `0.029627314724385286`;
- classify the approximately-80-percent target as
  `approximately_reproduced`; and
- persist fixed preceding-step sensitivities at 36,250 and 35,000.

The old exact-step 38,157 refusal, clean-tree gate, training/corpus/release
blockers, and `published-results` import path must be absent.

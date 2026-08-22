# Paper Validation Contract

This document is the shared implementation contract for the analytical
validation workflow described in `paper-reproducibility-plan.md`. It fixes the
persisted vocabulary and dependency boundaries used by the claim registry,
analyses, run bundles, report, and figures.

## Scientific boundary

Validation computations may read only normalized `dd_parsed` inputs declared
by the validation config. This includes explicitly declared normalized tables
under `published-results/`; unrestricted subtree discovery is prohibited.
Paper targets remain static claim-contract values rather than being inferred
from the validation inputs.

Evidence strength is persisted separately from agreement. `lower_level_rows`
means the attempt recomputes the finding from normalized evaluation, task,
instance, choice, loss, or checkpoint rows. `author_derived_aggregate` means
the attempt verifies downstream selections, arithmetic, comparisons, or plot
semantics from supplied predictions or aggregates without claiming to rerun
the upstream fit or evaluation. An attempt that reads either evidence level is
reported at the less independent `author_derived_aggregate` level.

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

`EvidenceLevel` has exactly these values:

- `lower_level_rows`
- `author_derived_aggregate`

The provenance-first vocabulary is deleted, including the former evidence
boundary hierarchy, method provenance, blocker kinds, blocker verdicts,
source-only matches, qualification status, and clean-tree state. The two-value
evidence-level field above is scientific interpretation metadata, not a
qualification gate.

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

Each `InputTableSpec` declares exactly one evidence level. Lower-level and
author-derived tables may coexist in the config, but every physical path is
listed explicitly and receives its own content identity.

Each `AttemptSpec` contains:

- stable attempt ID and claim ID;
- `default: bool` and optional parent attempt ID;
- closed analysis ID;
- evidence level, equal to the least-independent declared input;
- logical input tables and columns;
- declared recipe, seed, task, metric, size, and checkpoint universe;
- ordered transformation IDs;
- versioned comparison-rule ID;
- fixed sensitivity IDs; and
- plot-series IDs it is expected to produce.

Default attempt IDs are `<claim-id-lowercase>-default` (for example,
`dd-0011-default`). Closed analysis IDs are `single_scale`, `per_task`,
`proxy_metrics`, `noise_spread`, `scaling_law`, and `math_code`. A claim that
is assessable in principle has its default specification even before its
analysis is implemented; orchestration records
`not_assessable_from_dd_parsed` only for a declared missing evidence surface,
never as a synonym for unfinished code.

Cross-contract validation requires unique IDs, exact claim/attempt references,
one default attempt per assessable primary claim, existing method-dependency
claims, and no executable attempt for a nonempirical claim.

### Frozen qualitative comparison parameters

Every qualitative rule composes named `ComparisonParameter` values. Each value
stores one default and a sorted sensitivity grid containing that default;
duplicate parameter names and mixtures of typed parameters with the earlier
predicate-specific numeric fields are invalid. Verifiers read these values from
the parsed rule and do not carry analytical thresholds as code constants.

The single-scale and per-task defaults are: predictable accuracy at least
`0.80` (sensitivities `0.75`, `0.85`); strong-baseline accuracy at least
`0.75`; trivial accuracy within `0.05` of `0.50`; small scale at most `1%` of
target compute (sensitivities `0.1%`, `10%`); compute-equivalent accuracy
difference at least `-1/300`; matched accuracy at the minimum observed compute
reaching `0.80`, without interpolation; directional cheaper comparisons for
ordinary wording and a `10x` ratio for “much” or “substantially” (sensitivities
`3x`, `30x`); positive OLS slope per log10-compute decade and positive
Spearman correlation for “more compute”; maximum/minimum task threshold-compute
ratio at least `10`; low reliability below `0.80`; nontrivial accuracy above
`0.55`; markedly lower reliability gap at least `0.05`; fixed-compute task
range at least `0.20`; and a compute ratio at least `100000` while accuracy
remains at least `0.80` for the five-orders claim. BoolQ nontrivial points must
be above `0.55` and occur only at 1B intermediate checkpoints.

The compute-equivalence default uses half-open log10-compute buckets of width
`0.10` decade, with fixed `0.05` and `0.20` sensitivities. Edges are anchored
at integer multiples of the width after normalizing by target-model compute.
Within a `(bucket, intermediate size)`, checkpoint accuracy is averaged
arithmetically; it is compared with every different-size final checkpoint in
that bucket. Zero compute and same-size pairs are excluded and counted, invalid
compute fails validation, and interpolation is prohibited. Every matched group
must have intermediate-minus-final accuracy at least `-1/300`.

Plateau-then-rise claims use the best deterministic two-segment fit over
log10 compute. The two-segment SSE must improve on one segment by at least
`20%` (sensitivities `10%`, `30%`), the absolute early slope must be at most
`0.02` accuracy per decade, and the late slope must be positive.

Proxy and noise defaults are: small-scale proxy-minus-accuracy difference at
least `-1/300`; curve Spearman at least `0.90`; “most” and “strict majority”
strictly above `0.50` (sensitivities `1/3`, `2/3`, and, where configured,
`0.80`); maximum proxy overlap range `0.05`; flat absolute slope at most
`0.02`; convergence gap at most `0.05`; any adjacent decline at least `1/300`;
and positive Spearman association between decision accuracy and the
spread/noise ratio. Noise or spread must improve on a strict majority of tasks.
The 1B seed-SD claim requires more than five of the ten logical tasks to have
some recipe sample SD within `0.01` of `0.02`, and records the maximum observed
SD. Frequent crossover requires more than half of recipe pairs to cross
(sensitivities `1/3`, `2/3`).

The two-trend-type claim standardizes each task curve, initializes deterministic
`k=2` clustering from the farthest task pair, and requires mean silhouette at
least `0.25` (sensitivities `0.15`, `0.35`). The initialization has no random
seed or observed-outcome tuning step.

Scaling-law defaults use the eight paper setup families and their 21 declared
size subsets from the supplied cheap-decisions table. Prediction-error checks
average the 275 primary-metric task-by-recipe rows per base setup, multiply by
100, and round to one decimal. Relative error is computed twice: once from the
supplied `rel_error_stacked` column and once as the paper states,
`abs(target - prediction) / target`. Both values and the denominator mismatch
are persisted. Decision accuracy uses one validated common target ranking,
excludes target ties, and counts predicted ties as incorrect by default with a
half-credit sensitivity where declared.

Math/code defaults use the complete supplied decision-accuracy and means cubes
for MBPP, HumanEval, Minerva, and GSM8K. Thresholds are fixed at `0.80` with
`0.75` and `0.85` sensitivities; near-random uses `0.50` with default tolerance
`0.05` and `0.10` sensitivity; approximate-0.80 uses default tolerance `0.05`
with `0.025` and `0.10` sensitivities; and material gain is `0.05`. Plots use
the table's actual `logits_per_byte_corr` metric name and disclose that the
paper caption calls the series `Correct Prob`.

## Persisted result models

Persisted/config boundaries use frozen Pydantic models. Internal calculation
values use frozen slotted dataclasses.

`PaperTarget` stores claim ID, family, kind, source locator/text, and the
normalized paper target. `AttemptResult` stores target and computed result in
separate fields and includes:

- attempt and claim IDs, default/sensitivity role, and parent attempt;
- evidence level, matching its specification and selected inputs;
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

## Run format 3

One immutable run directory contains exactly:

```text
data/paper-validation/runs/<run-id>/
  manifest.json
  targets.json
  attempts.json
  plot-series.json
```

`AnalysisManifest` records `run_format = 3`, run ID/timestamps, optional
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
discrepancies never enter empirical outcome counts. Every finding displays its
evidence level. Author-derived results use language such as "verified against
an author-derived aggregate" and never "independently reproduced."

## Canonical API and CLI

The public workflow is:

```python
validate_repository(root, data_dir) -> ValidationSurface
run_validation(root, run_id, data_dir) -> AnalysisBundle
load_analysis_bundle(runs_root, run_id) -> AnalysisBundle
render_validation(root, run_id) -> RenderedOutputs
```

Analysis adapters expose one consistent boundary:

```python
run_<analysis_id>_attempts(
    *,
    repository_root: Path,
    data_root: Path,
    registry: ClaimRegistry,
    contract: PaperValidationContract,
    input_identities: Mapping[str, ContentIdentity],
) -> tuple[tuple[AttemptResult, ...], tuple[PlotSeries, ...]]
```

Adapters return only results for their closed analysis ID, ordered by attempt
ID, followed by ordered plot-series IDs. They translate deterministic internal
calculation values into persisted models but do not create run manifests,
write files, or inspect other analysis families. Missing eligible evidence is
an explicit `not_assessable_from_dd_parsed` result with a zero-row selection
and concrete missing groups; unfinished implementation is never serialized as
that outcome.

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

The old exact-step 38,157 refusal, clean-tree gate, and
training/corpus/release blockers must be absent.

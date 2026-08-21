# DataDecide paper reproducibility plan

Status: implemented and qualified in DataDecide; reusable skill submitted

Created: August 21, 2026

Implementation record: DataDecide PR #43 records the qualified workflow and
selected run `20260821T1458-remediated`. The extracted `verify-paper-claims`
skill is submitted separately in dotfiles PR #81. Repository-only skill
validation and an independent cold-agent fixture test passed. Live harness
reconciliation is intentionally deferred until that dotfiles PR merges because
the canonical main checkout contains unrelated user edits.

## Goal

Build an auditable, repeatable account of every active claim in the DataDecide
paper and determine the strongest evidence this repository can provide for each
claim.

After the workflow is implemented and validated here, extract its reusable
parts into a cross-repository agent skill so future paper-verification efforts
can follow the demonstrated process without inheriting DataDecide-specific
assumptions.

The scope includes prose findings, comparisons, exact numerical statements,
methods and dataset descriptions, equations, tables, figure data, captions,
release claims, and limitations. The result must distinguish independent
recomputation from matching an author-produced artifact or implementing a
method detail learned from the author source. Missing inputs and underspecified
methods are expected outcomes, not verification failures to hide or work
around.

## Current repository position

This repository already owns a strong normalized evidence layer:

- pinned source inventories and download hashes;
- raw and processed aggregate PPL and OLMES results;
- processed OLMES task, instance, and choice results for all 25 recipes;
- processed scaling-law evaluations and checkpoint losses;
- 51 processed author-produced result tables;
- canonical model, training, token, and compute derivations; and
- cross-source and derivation verification.

It does not currently own the paper's scientific analysis. In particular,
there is no current-repository implementation of target ranking, pairwise
decision accuracy, the paper's scaling-law fits, noise-versus-spread analysis,
or the paper plots.

The official `allenai/DataDecide` repository contains relevant author code
under `single_scale/`, `scaling_laws/`, and `viz/`. Its current `main` resolves
to commit `68abc496587935b7211e9894206e78f15d535832`, dated June 17, 2025. This
is a candidate methodological reference, not yet proof that the commit exactly
corresponds to arXiv v2, dated July 13, 2025. It will not be a package,
submodule, vendored tree, or executable dependency of this repository.

One known paper-versus-repository contradiction already demonstrates the need
for explicit verdicts: the paper's hyperparameter caption says sequence length
`2024`, while the current catalog, implementation assertions, aggregate data,
and detail data say `2048`.

## Decision 1: reimplement the required analysis locally

The verification implementation will be owned by this repository and use its
existing normalized data contracts and environment. Implement only the
transformations, aggregations, fits, and plot-data generation required by the
inventoried paper claims. Do not port the upstream repository as a general
library or reproduce unused infrastructure.

The paper is the primary method specification. Record the upstream repository
URL, exact reference commit, license, and relevant file or function paths in
`configs/paper_reproduction.toml` when the author source helps interpret an
omitted or ambiguous detail. This freezes the provenance of that interpretation
without making the author repository an executable dependency.

Each verifier records one of these method-provenance categories:

- `paper_derived`: implemented from the paper and independently checked;
- `upstream_informed`: repository-owned implementation of a detail that the
  paper omitted but the pinned author source made explicit;
- `artifact_derived`: computed from an author-produced downstream table rather
  than lower-level observations.

The default is `paper_derived`; use `upstream_informed` only when necessary.
Do not copy or mechanically adapt author source. If a required operation cannot
be independently reimplemented, record it as blocked and revisit that scope
explicitly.

Executing the author repository is not part of the planned verification
workflow. This plan maintains one analysis path, and disagreements remain
visible rather than being resolved by running the original scripts.

## Decision 2: canonical reproduction configuration

Create `configs/paper_reproduction.toml` as the versioned contract for repeating
the audit. It should record decisions rather than duplicate data already owned
by `configs/sources.toml`, `configs/catalog.toml`,
`configs/scaling_law.toml`, or `configs/published_results.toml`.

The configuration should cover:

### Source identity

- paper arXiv identifier, revision, source URL, and archive checksum;
- when author source informs a method, its repository URL, exact reference
  commit, license, and relevant source paths;
- qualified runtime identity, including Python version and implementation,
  operating system, architecture, and required native tools or libraries;
- references to the existing source, catalog, OLMES, scaling-law, and
  published-result contracts; and
- expected input table names and schemas.

### Comparison universe

- included and excluded data recipes;
- included tasks and task aggregation groups;
- target model size and target checkpoint selection;
- target seeds and prediction seeds;
- target and proxy metrics;
- recipe-pair construction and ordering;
- tie and abstention behavior;
- missing-value and incomplete-pair behavior; and
- any historical seed or recipe aliases.

### Compute and checkpoint policy

- parameter-count field used for compute;
- token and FLOP definitions;
- percent-of-target-compute denominator;
- final-checkpoint definition;
- intermediate-checkpoint inclusion;
- early-seed stopping policy; and
- scaling-law model-size subsets.

### Statistical and fit policy

- aggregation order across instances, tasks, recipes, and seeds;
- standard-deviation convention and uncertainty display;
- scaling-law variants;
- optimizer, initialization, bounds, helper points, and failure policy;
- treatment of crossovers, ties, and failed fits;
- qualitative-claim predicates; and
- absolute and relative numeric tolerances.

### Output policy

- expected claim registry path;
- per-run observation and manifest locations;
- generated result and plot locations;
- compact report output at `docs/paper-reproduction-report.md` for version
  control;
- small reproduced figure outputs under `docs/paper/reproduced-figures/` for
  version control;
- large or intermediate outputs that remain under ignored `data/`; and
- run-manifest fields, including code and input identities.

The configuration loader must reject incomplete policies needed by an enabled
analysis. It must not silently supply a plausible default where the paper and
available method evidence leave behavior unresolved.

## Decision 3: claim registry and verdict model

Create a machine-readable static claim specification. The likely location is
`docs/paper/claims.toml`, with per-run observations and a generated human-facing
report stored separately. The paper source remains unchanged.

Each claim needs:

- a stable claim ID;
- an active-document locator and exact claim text;
- claim ownership: DataDecide empirical, method/design, artifact/release,
  qualitative interpretation, or external citation;
- expected value, relationship, or qualitative predicate;
- required evidence boundary;
- verifier and input references;
- method-provenance category and upstream reference when applicable;
- comparison policy reference when known;
- expected tolerance when the paper defines one;
- known prerequisite or external-owner references;
- generated table or figure references; and
- any unresolved method question that prevents verification.

The static claim specification never owns observed values or verdicts. Each run
writes an immutable observation set and run manifest under
`data/paper-reproduction/runs/`. Generated reports join the static claim
definitions to one explicitly selected run. A rerun creates a new observation
set rather than updating claim definitions or a previous run.

Evidence boundary and verdict are independent.

### Evidence boundaries

1. paper-source or final-artifact match;
2. regeneration from author-produced downstream tables;
3. independent recomputation from aggregate evaluation rows;
4. independent recomputation from instance and choice rows;
5. evaluation rerun from released checkpoints and pinned task data;
6. training rerun; and
7. corpus construction from source documents and recipe operations.

### Verdicts

- reproduced;
- contradicted by current repository evidence;
- internally inconsistent;
- source-only match;
- blocked by missing input;
- blocked by unspecified method;
- externally owned or citation-dependent;
- not attempted; and
- not applicable at the requested evidence boundary.

### External-citation coverage

External literature claims receive citation-trace coverage, not independent
verification of the cited work. For each external claim:

- retain its exact paper text and location;
- record every citation key attached to it;
- verify that each key resolves to a bibliography entry and identify a stable
  DOI, arXiv record, or publisher page when available; and
- preserve the mapping from the attributed proposition to the cited work.

Unless the paper also makes the proposition as a DataDecide-owned empirical
claim, its verdict remains `externally owned or citation-dependent`. This scope
does not assess whether the cited work's evidence is correct or reproducible.

Every observation and generated verdict records the paper revision,
current-repository code identity, method-provenance category, upstream reference
when it informed the method, input identities, denominator, exclusions, and
comparison policy. Current-repository code identity is either a clean tree at
the recorded commit or the commit plus a hash of the captured dirty diff. Full
qualification should use a clean tree. A downstream author table containing the
paper number may support a source-only match but cannot by itself support
independent reproduction.

## Architecture and ownership

The intended dependency flow is:

```text
paper source + paper_reproduction.toml
                 |
                 v
          claim registry
                 |
                 v
     repository-owned datadec verifiers
                 |
                 v
        observations + run manifests
                 |
                 v
       generated report and plots
```

The pinned upstream reference informs explicit method-provenance records but is
not on the runtime dependency path.

The eventual reusable skill is durable agent configuration owned by the
`dotfiles` repository, not this repository. DataDecide retains the concrete
claim schema, comparison configuration, implementations, tests, and results;
the skill retains only portable workflow guidance and any genuinely generic,
validated helpers.

Proposed ownership boundaries:

- `configs/paper_reproduction.toml`: canonical study and comparison contract;
- `docs/paper/claims.toml`: canonical static claim definitions and expected
  evidence;
- `src/datadec/paper/`: independent calculations and claim verification;
- `scripts/verify_paper_claims.py`: orchestration and report generation;
- `data/paper-reproduction/runs/`: immutable per-run observations and manifests;
- `data/paper-reproduction/`: other large inputs and intermediates;
- `docs/paper-reproduction-report.md`: generated compact versioned status
  report; and
- `docs/paper/reproduced-figures/`: small versioned figure outputs.

Exact names may change during implementation, but paper and method provenance,
configuration, verification, and generated-output responsibilities must remain
separate.

## Work plan

### Phase 0: freeze source and method provenance

1. Record the exact paper revision, source URL, and archive checksum.
2. Treat the current upstream commit as an optional methodological reference,
   not an input that every verifier must use.
3. Only for a method question the paper and current artifacts cannot resolve,
   identify the exact upstream commit and source path that clarifies it; do not
   import or execute the source.
4. Record discrepancies between the paper and reference source as unresolved
   method questions rather than silently choosing one.
5. Qualify the current repository runtime used for verification, including the
   Python interpreter, operating system, architecture, and required native
   dependencies.

Exit condition: the paper, current repository, and every upstream-informed
method interpretation have immutable, reviewable identities and provenance.

### Phase 1: inventory every active claim

1. Traverse the active compiled TeX tree, expanding macros and active inputs.
2. Exclude comments and dead historical table variants.
3. Atomize prose, caption, table-cell, equation, release, and limitation
   assertions.
4. Describe every plotted series, facet, aggregation, uncertainty band, axis,
   and caption assertion.
5. Tag external literature claims separately from DataDecide-owned claims and
   create the defined citation trace for each one.
6. Check that every active claim-bearing region maps to one or more claim IDs.
7. Record the static definitions in `docs/paper/claims.toml` without assigning
   run verdicts.

Exit condition: there are no unclassified active assertions, tables, or paper
figures.

### Phase 2: define the comparison contract and evidence map

1. Map each claim to paper definitions, current data contracts, available
   author artifacts, relevant pinned source references, and the minimum useful
   evidence boundary.
2. Resolve comparison universe, aggregation, seeds, checkpoints, ties,
   abstentions, missing values, uncertainty, and qualitative predicates.
3. Record settled behavior in `configs/paper_reproduction.toml`.
4. Record genuinely unresolved behavior explicitly; do not infer it in code.
5. Add comparison-policy and verifier references to the static claim
   definitions.
6. Validate the contract against every active claim, existing table schemas,
   and canonical model configuration.

Exit condition: every enabled analysis has a complete, reviewable method
contract or an explicit unresolved blocker, and every claim has an evidence
map. Phases 1 and 2 may iterate until both completeness checks converge.

### Phase 3: implement repository-owned verification

Implement repository-owned calculations for:

- model-grid, seed, checkpoint, token, parameter, and compute claims;
- task metrics, macro averages, and target rankings;
- recipe-pair construction and decision accuracy;
- single-scale compute-versus-decision curves;
- proxy-metric comparisons;
- run-to-run noise and between-recipe spread;
- scaling-law variants and prediction-error tables; and
- paper figure data and qualitative comparison predicates.

Each verifier emits expected and observed values, tolerance, denominator,
exclusions, input identities, method provenance, and verdict. Small explicit
fixtures establish behavior; full-data runs establish paper evidence. Tests
must exercise the mathematical or data contract directly rather than treating
agreement with an upstream output as sufficient proof.

Exit condition: every claim that is answerable from current normalized data has
an independent verdict.

### Phase 4: evaluate deeper evidence gaps

For claims not answerable from current normalized data, separately assess:

- availability and completeness of released models and checkpoints;
- evaluation reruns using released checkpoints and pinned task datasets;
- recipe availability and declared composition;
- document membership, mixing proportions, filters, and classifier artifacts;
- exact training data order and checkpoint provenance; and
- feasibility of training reruns.

The published tokenized recipe collection is approximately 19.3 TB, so a full
download is not a prerequisite for earlier phases. Begin with metadata,
manifests, mapping code, and representative checks. Expand only when the
evidence requirement justifies the storage and compute cost.

Exit condition: every remaining claim names the exact missing artifact,
underspecified rule, external owner, or deferred cost boundary.

### Phase 5: synthesize and review

1. Generate the compact human-readable report by joining the static claim
   specification to one explicitly selected immutable run.
2. Summarize reproduced, contradicted, inconsistent, source-only, blocked, and
   external claims separately.
3. Distinguish current local availability from repository-supported download
   capability and remote availability.
4. Perform one full adversarial review of behavior, contracts, evidence
   boundaries, failure handling, and the main scientific conclusions.
5. Remediate in-scope correctness defects once, then regenerate the report.
6. Version the compact report and small reproduced figure outputs; retain large
   observations, intermediates, and plot data under ignored `data/`.

Exit condition: every headline conclusion links to atomic claims, executable
evidence, and explicit provenance.

### Phase 6: extract and validate the reusable skill

Begin this phase only after the compact report and adversarial review establish
that the workflow has operated end to end. Use the completed implementation,
tests, report, review findings, and skill-extraction ledger as source material.

1. Review the ledger and retain only lessons supported by an executed workflow,
   test, observed failure, or review finding. Do not turn untested preferences
   or DataDecide-specific values into general rules.
2. Separate portable workflow from repository policy. The skill may describe
   contracts, evidence boundaries, coverage checks, provenance, and completion
   criteria, but it must discover repository-specific paths, schemas, commands,
   and output locations from the target project.
3. Create an authored skill named `verify-paper-claims` under
   `dotfiles/agents/skills/verify-paper-claims/`. Give it an automatically
   discoverable description for claim-level paper reproduction and exclude
   ordinary paper summaries, literature reviews, and formatting work. Candidate
   description: "Inventory and independently verify a research paper's claims
   against repository data and code. Use for claim-level reproducibility audits;
   do not use for ordinary summaries, literature reviews, or paper formatting."
4. Keep `SKILL.md` focused on the shared workflow, non-obvious invariants, and
   checkable completion criteria. Add focused `references/` only for substantial
   reusable contracts such as claim schemas, evidence/verdict semantics, or
   figure and citation handling.
5. Include a skill-local script only when the implementation has demonstrated
   that the operation is deterministic, portable, repeated across the workflow,
   and free of DataDecide-specific assumptions. Otherwise direct agents to use
   or create repository-owned commands.
6. Add the skill to `dotfiles/agents/skills/sources.json` as an authored skill
   with exposure `on`. Record the qualifying DataDecide commit or PR in its
   provenance note, then regenerate the dotfiles-managed exposure metadata.
7. Validate structure with the skill-creator `quick_validate.py`, run dotfiles'
   focused exposure checks and `mise run check`, apply the managed skill links,
   and confirm the live skill resolves to the dotfiles-owned source.
8. Run an independent cold-agent forward test in an isolated workspace against
   a held-out small paper/repository fixture. Confirm that it inventories all
   claim-bearing regions, separates static claims from run observations,
   distinguishes evidence boundary from verdict, exposes missing inputs, and
   avoids author-code dependence. Also test a plain paper-summary request to
   confirm the skill's invocation boundary is not overbroad.
9. Revise only for failures observed in validation, rerun the structural and
   behavioral checks, and publish the skill through a separate dotfiles PR.

Exit condition: a fresh agent can apply the skill to a new repository without
DataDecide knowledge, preserves the verified evidence boundaries, and produces
an auditable claim-level workflow rather than a prose-only checklist.

## Figure and qualitative-claim policy

Paper plots are verified primarily through their data and semantics, not exact
pixels. For each figure, verify:

- source observations;
- filtering and aggregation;
- series and facet membership;
- uncertainty computation;
- axis variables and transforms;
- labels and legend meaning; and
- every factual caption statement.

Pixel or PDF equality may be recorded as author-artifact evidence but is not
required for independent reproduction because rendering libraries and fonts can
change without changing the scientific result.

Qualitative phrases such as "roughly log-linear," "as good as," "most small
scales," "frequently," or "markedly less reliable" require a predicate in the
comparison configuration. If the paper and available method evidence do not
define one, the report must label the independent verdict as operationalized by
this repository or blocked by an unspecified criterion.

## Validation requirements

The finished system must prove:

- the paper identity is exact, and any upstream-informed method records its
  reference identity, license, and relevant source path without creating an
  upstream runtime dependency;
- the comparison configuration is schema-valid and complete for enabled
  analyses;
- every active paper assertion has a claim ID;
- every external literature claim has a complete citation trace and remains
  explicitly citation-dependent;
- the claim specification contains no mutable run verdicts;
- every verifier is owned and executed by this repository;
- no verifier imports or executes author analysis functions;
- method provenance distinguishes paper-derived, upstream-informed,
  and artifact-derived behavior;
- no verifier contains copied or mechanically adapted author source;
- input hashes or immutable revisions are present in every full-data run;
- current-repository code identity is a clean commit or includes a captured
  dirty-diff hash, with clean commits required for final qualification;
- denominators, exclusions, ties, abstentions, and failed fits are visible;
- generated observations are deterministic under stable inputs;
- figures are backed by machine-readable numerical series; and
- contradictions are not collapsed into missing-data blockers.

## Delivery shape

Use the fewest coherent PRs that preserve real review boundaries. The likely
shape is:

1. source and method provenance, comparison contract, and complete claim
   inventory;
2. repository-owned current-data verifiers and generated report;
3. optional external-data, evaluation-rerun, or corpus-reconstruction work only
   where the gap audit justifies it.

The first PR should not claim scientific reproduction. It establishes the
versioned contracts and complete audit surface on which later verdicts depend.
After the DataDecide workflow is qualified, deliver `verify-paper-claims` in a
separate dotfiles PR because durable agent configuration has a different
repository owner and validation path.

## Settled scope decisions

1. Defer identifying the exact upstream reference commit unless a method must
   be `upstream_informed`. Omit the reference entirely if every method is
   `paper_derived`.
2. Give external literature claims citation-trace coverage only. A full audit
   of cited papers is outside this effort.
3. Version the compact reproduction report and small reproduced figure outputs.
   Keep large observations, plot data, and intermediate artifacts under
   ignored `data/`.
4. Create the reusable skill only after the workflow completes end to end, and
   expose it automatically from its canonical dotfiles owner.

## Skill-extraction ledger

Update this ledger at each phase exit. Promote a candidate into the final skill
only after the named evidence gate has been exercised; record new candidates
when implementation or review reveals a reusable decision that is not already
captured.

| Candidate lesson | Evidence gate | Status |
| --- | --- | --- |
| Cover every active prose, equation, table, figure, caption, method, release, and limitation assertion. | Independent active-source traversal partitions 455 claims across 192 exact claim/nonclaim regions. | Promoted |
| Keep static claim definitions separate from immutable run observations and generated verdicts. | The qualified run and generated report join separate hashed layers; registry remediation remained a distinct identity rather than a false same-spec rerun. | Promoted |
| Track evidence boundary, method provenance, and verdict as separate dimensions. | The qualified report contains reproduced, contradicted, source-only, blocked, external, and not-attempted outcomes across four actual evidence boundaries. | Promoted |
| Require explicit comparison policy and expose unresolved methods instead of supplying plausible defaults. | 108 real claims terminate as method-blocked, and the exact missing DD-0011 checkpoint remains input-blocked rather than using a nearby checkpoint. | Promoted |
| Verify figure data and semantics before treating pixel equality as scientific evidence. | The workflow inventoried all paper-figure semantics but lacked inputs to regenerate them; only audit-summary figures were rendered. The reusable blocker rule was cold-tested, but scientific figure regeneration was not demonstrated. | Promoted as a blocking safeguard only |
| Treat external claims as citation traces unless their underlying evidence is explicitly in scope. | All 39 external claims retain source-bound citation traces and external verdicts. | Promoted |
| Derive source coverage independently from the claim registry and bind citations to each claim span. | Full review exposed circular coverage and global citation-union checks; remediation added an independent active surface and per-claim validation. | Promoted |
| Mark reproduced only when a persisted fact directly satisfies an explicit claim predicate. | Full-data replay produced 4,128 complete summaries, but only DD-0045 had a direct fact mapping; related facts did not upgrade the other claims. | Promoted |
| Render compact outputs from selected persisted observations without scientific recomputation. | Review replay reproduced the report and both summary figures byte-for-byte; detailed DD-0045 facts remain in immutable observations. | Promoted |
| Apply shared manuscript-wide method statements to every claim they govern. | The cold-agent fixture exposed this ambiguity for a three-seed statement; the skill was revised before submission. | Promoted |

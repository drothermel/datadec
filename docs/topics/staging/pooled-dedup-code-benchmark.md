# Pooled + deduplicated code benchmark — spec-out placeholder

**Kind:** staging (placeholder by decision, 2026-08-24). Danielle asked for the
placeholder now and the spec later — do not expand this document until she
initiates the spec.

**Provenance.** Line of work 3 ("Building New Benchmarks by Pooling &
Deduplicating Old Ones") from her code-datasets lineage/overlap note (bundle:
`~/drotherm/data/convo-artifacts/2026/2026-08-24-notion-lit-reviews/code-datasets-lineage-overlap-note.md`;
routed entry: `../reference/code-benchmarks-landscape.md`, 2026-08-24 lineage
entry). Her 6-step component-analysis pipeline (normalized-hash exact edges →
MinHash/LSH lexical edges → token/AST clone edges → embedding edges on
prompts/docstrings → connected components / community detection → human review of
borderline clusters) is the half-spec already on record, and ContextBench's
pool-4,497 → dedup-3,100 → select-1,136 construction is the named precedent.

**Constraint to honor at spec time:** benchmark-as-byproduct
(`../reference/project-approach-principles.md`, principle 1) — the spec must name
the question the pooled/deduplicated suite instruments and the finding it must
enable before any construction; a cleaned benchmark alone is not the deliverable.
Natural candidates on record: the overlap/lineage structure itself as the finding
(component analysis of the benchmark ecosystem), and the TLC task-suite rehab as
the first consumer.

**Gate.** Danielle decides whether and when to spec it. Exits: a real staging spec
(then the normal promotion path), or absorption into the TLC dataset-rehab work,
or deletion.

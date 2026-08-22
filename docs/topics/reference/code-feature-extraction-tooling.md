# Code feature extraction from Python source — reference topic

**Kind:** reference (accumulator for tooling that turns source code into named,
interpretable features: lexical, syntactic, complexity, structural, rule-match, type,
flow, query, symbol, accounting, statistical). Entries are dated. Tool-capability claims
below are as stated in the response; none re-verified here.

Why it matters here: code is the working domain of `../../potential-projs/text-latent-code-autoencoder.md`
(TLC: compression of code, functional vs. stylistic factors), the clean-code preference
test (`../staging/clean-code-preference-icl.md`: automated feedback beyond tests + length
ratio), and the verifier suite in `../../potential-projs/elicitation-gain.md` (ELI).
Any of these wants a fixed-schema, no-training feature extractor for Python strings.

---

## 2026-07-03 — Deep-research report from Danielle's "Report 2: Code Feature Extraction" brief (intake 2026-08-22)

**Danielle's prompt (verbatim, at intake):**

> great! ok, a completely different topic, I did a deep research prompt based on the
> attached doc. and got the answering report.

**The brief (Danielle's own document; on file at
`../../refs/deep-research/2026-07-03-report2-code-features-brief.md`).** Report 2 of a
series governed by two companion documents (Deep Research Session Guidelines; Text and
Code Analysis Deep Research Guidelines — not on file). Its contract, condensed:

- Objective: a *landscape survey* of methods, libraries, and research artifacts that
  take source code (Python among the inputs; not Python-exclusive) and produce named
  features. Describe what exists — input, output, what it is built on, maintenance
  status with dated evidence. **No recommendations, rankings, tiers, workflows, or
  difficulty assessments. No tables.**
- Scope rules: small fixed-schema pretrained models in; large generative models and
  learned dense representations (neural code embeddings) out; manual recipes out;
  stdlib modules are entries. Adjacent reports own mixed prose+code handling (Report 1,
  inferred) and similarity/clone detection (Report 3, inferred).
- Malformed-input behavior is a first-order descriptive fact per entry (fails /
  degrades / partial / sub-syntactic), as are parse requirement, snippet-vs-file-vs-project
  granularity, and accepted Python versions.
- Eleven feature families as a floor: lexical/token; syntactic parse structure; size and
  complexity; structural/design metrics; rule-match static analysis; type-related;
  control/data-flow; structured representations and query systems; symbol indexing and
  cross-reference; code accounting and language ID; statistical models with interpretable
  outputs (n-gram naturalness, defect-prediction feature sets).
- Matrix accounting: input granularity × parse requirement × output shape, every cell
  populated or declared empty; the snippet row is the axis of greatest uncertainty.
- Search must reach beyond PyPI (crates.io, npm, Maven, Go, OCaml/Haskell archives,
  system packages; SE/program-analysis/MSR/security literature; IDE engines, compiler
  front-ends, commercial platforms), expand from hubs, and state any narrowing.
- Deliverables: the report; a considered-and-excluded list with the boundary rule per
  exclusion; the matrix-accounting section; a search-log summary.

**The response (on file at
`../../refs/deep-research/2026-07-03-code-feature-extraction-report.md`),
condensed to the surviving inventory.** It organizes by layer rather than by the brief's
eleven families:

- *Stdlib baseline:* `tokenize` (token stream, `exact_type` for operators), `ast`
  (line/column offsets), `symtable` (compiler scopes), `pyclbr` (class/function browsing
  without import).
- *Concrete / error-tolerant / incremental parsers:* LibCST (lossless, metadata, scope
  analysis, codemods), parso (round-trip, error recovery, multi-version), tree-sitter
  (incremental, syntax-error robust).
- *Inference and type signals:* astroid (inference; powers Pylint), Pyright, mypy
  (incremental, daemon, plugins, stubgen), Jedi (references, goto, completion).
- *Rule and query engines:* Ruff (900+ rules, cache, autofix), Pylint, Semgrep (pattern +
  data-flow; cross-file), CodeQL (code as queryable database), Joern (code property
  graphs, Scala query language).
- *Accounting / identification / indexing:* Linguist, enry (shebang, modeline, Bayesian
  strategies), Universal Ctags (Python parser; JSON/xref output), `cloc`, Tokei, Radon
  (cyclomatic, Halstead, raw, maintainability index).
- *Type-inference research line (named for later):* Type4Py, Typify, PyTy; datasets
  ManyTypes4Py (5,382 projects, 869k+ annotations), CrossDomainTypes4Py, PyTyDefects
  (2,766 type-error/fix pairs); corpora The Stack, The Heap.
- *Commercial overlays named:* SciTools Understand, CodeScene.
- *Excluded by the response:* srcML (language list C/C++/C#/Java, no Python), clone
  detectors, execution/profiling, decompilers, large code-generation models.

**Intake notes — the response does not deliver the brief.**

- It reports a conflict between "the prompt" (asking about text features, embeddings,
  multimodal methods, PyTorch/TF/HF) and the attached brief, and resolves it toward the
  brief. The prompt Danielle actually sent is not on file; if it was a generic deep-research
  template, the conflict is real and the brief won.
- Despite that, it violates the brief's form rules: it recommends ("preferable when…",
  "highest-confidence implementation conclusion"), defines small/medium/large budget
  profiles, gives resource estimates and a Gantt timeline, and ships code, pre-commit,
  and GitHub Actions recipes — all explicitly out of scope (no recommendations, no
  workflows, no difficulty assessment). It omits three of four required deliverables: no
  considered-and-excluded list with boundary rules (only a short paragraph), no
  matrix-accounting section, no search-log summary. Malformed-input behavior — the
  brief's first-order fact — is mentioned only in passing for parso and tree-sitter.
- Several of the eleven families are thin or absent: structural/design metrics
  (coupling/cohesion, call/import graphs), control/data-flow as a family (only via
  Semgrep/CodeQL/Joern), statistical naturalness models (n-gram perplexity; the
  code-naturalness and defect-prediction feature literature) — entirely missing. Search
  scope narrowed to tools "a Python practitioner can use today" without the brief's
  required statement of narrowing.
- Citations are opaque tool-call references (`turn19view0` etc.), not resolvable sources;
  every capability and dataset figure above is therefore unverified. The "ruff v0.15.0",
  Python "3.14" matrix, and dated claims were not checked.
- Net: the tool inventory is a reasonable seed list; the report is not the survey the
  brief specified. Re-running the brief (or delegating the eleven families to separate
  passes with the matrix as the output contract) is the path to the actual deliverable.

# Deep Research Brief — Report 2: Code Feature Extraction

This brief is read alongside two attached documents: the Deep Research Session Guidelines (which govern how the survey is conducted and written) and the Text and Code Analysis Deep Research Guidelines (which define what counts as an entry). Where this brief is silent, those documents govern.

## Objective

Produce a landscape survey of existing methods, libraries, and research artifacts for extracting named, interpretable features from source code. The inputs of primary interest are Python source strings, but the tooling landscape is not Python's alone: analyzers written in any language, multi-language analysis platforms, and language-agnostic structural tools all qualify as entries whenever Python is among the inputs they can process. An entry need not be Python-native, pip-installable, or Python-exclusive to belong to this survey.

The survey describes what exists: for each entry, what it is, what it takes as input, what it produces, what it is built on, and its maintenance status with evidence. The survey does not recommend, rank, tier, or assemble entries into a workflow, and it does not assess difficulty of use.

Per the method-boundaries document, small pretrained and adaptable models with fixed output schemas are in scope; large generative models and learned dense representations (including neural code embeddings) are out of scope; manual recipes are out of scope, but shipped standard-library modules are entries on normal terms.

## Subject Definition

An entry belongs to this landscape if it takes source code as input and produces one or more named features as output: scalar metrics, counts and ratios, categorical tags or rule matches, symbol tables, extracted spans, or structural representations such as trees and graphs. The defining property is that each output has a name and a meaning traceable to the method that produced it.

Two adjacent landscapes are covered by separate reports and are out of scope here: methods for handling strings that interleave prose and code (including locating or extracting code blocks from mixed text — this report's inputs are already known to be code), and methods for measuring similarity, overlap, or duplication between code fragments, including clone detection. If an entry primarily belongs to one of those landscapes, it is excluded here with a note; if it primarily belongs here and incidentally touches those, it is described here with its full capability stated factually.

Behavior on malformed input is a descriptive fact of first-order interest in this landscape. For each entry, where documentation or the reference implementation makes it determinable, the description states what the entry does with code that does not parse: whether it fails, degrades, recovers with partial output, or operates below the syntax level entirely. Related intrinsic facts include whether an entry requires a successful parse, whether it operates on single self-contained snippets or requires surrounding project context (imports resolvable, multiple files, an installed environment), and which Python language versions its parser accepts.

Multi-language entries are described in their full generality — what languages they cover and what they produce across them — not solely through the lens of their Python support, per the domain guidelines' cross-ecosystem coverage rule.

## Coverage: Feature Families

The survey must address each of the following families. Families are defined by what is extracted, not by which tools extract it. Within each family, the specific entries are determined by the landscape itself; the family names below carry no implication about which entries matter or how many exist. If research reveals a family boundary drawn here is wrong, or a family that belongs and is missing, restructure or extend — the list is a floor, not a taxonomy mandate.

1. Lexical and token-level features: tokenizers and the features computed from token streams — token counts and class distributions, literal and operator statistics, comment and docstring extraction, identifier-name analysis (casing conventions, dictionary-word content, length statistics).
2. Syntactic parsing and parse-derived structure: abstract and concrete syntax tree parsers, error-tolerant and incremental parsers, and the structural features computed from parse output — node-type distributions, tree depth and shape, construction frequencies.
3. Size and complexity metrics: line-counting conventions and their variants, cyclomatic and cognitive complexity, Halstead measures, maintainability-index-style composites, nesting and branching statistics.
4. Structural and design metrics: coupling and cohesion measures, inheritance and composition statistics, call-graph and import-graph derived features, module- and function-granularity architecture measures.
5. Rule-match features from static analysis: linters, security scanners, style checkers, and pattern-matching engines, viewed through the property that their rule matches are named categorical outputs; the rule vocabularies themselves are part of what is described.
6. Type-related features: type inference engines, type checkers, and annotation-analysis tools, producing inferred types, error counts and categories, and annotation-coverage measures.
7. Control-flow and data-flow analysis: control-flow graph builders, def-use and reaching-definitions analyzers, program-dependence and code-property-graph platforms, and the queryable features their representations expose.
8. Structured code representations and query systems: tools whose product is a structured, queryable representation of code (markup documents, relational or graph databases, symbol indexes) from which features are obtained by query.
9. Symbol indexing and cross-reference extraction: tag generators, language-server-based extractors, and code-intelligence indexers producing symbol tables, definition-reference relations, and scope structure.
10. Code accounting and identification: language identification for code snippets, line-classification tools (code, comment, blank), and per-language counting systems.
11. Statistical models of code with interpretable outputs: n-gram and other transparent language models of source code producing perplexity or naturalness scores, and research feature sets from the code-naturalness and defect-prediction literature, where outputs are named scalars.

Execution-based analysis — measuring code by running it — is out of scope; the survey covers static analysis of code as text and structure. Dynamic-analysis tools whose static components independently qualify are described for those static components, with the boundary noted.

## Coverage: Matrix Accounting

The survey must account for the full matrix defined by these dimensions:

- Input granularity: self-contained snippet or single function; single complete file; multi-file project with resolvable imports.
- Parse requirement: operates on raw text without parsing; requires a successful parse; parses with error recovery.
- Output shape: scalar metric; counts and ratios; categorical tags or rule matches; symbol tables and span relations; tree or graph structure.

Every plausible cell is either populated with entries or explicitly declared empty or sparse. Declared gaps are findings, not failures. Pay specific attention to the snippet-granularity row: which entries genuinely operate on isolated, possibly incomplete fragments versus assuming a file or project is the axis of greatest uncertainty in this landscape.

## Search Strategy Requirements

The search must not be confined to one ecosystem or source type. Code analysis has decades of investment outside the Python packaging world, and reaching it is a primary requirement of this survey, not an extension.

- Search across package ecosystems and distribution channels: PyPI, crates.io, npm, Maven Central, Go modules, OCaml and Haskell package archives, and system package managers — many mature analyzers ship as standalone binaries or system packages rather than language-ecosystem packages.
- Search the research literature and its artifacts: software-engineering and program-analysis venues for metrics, structural analysis, and code-naturalness work; mining-software-repositories literature for feature-extraction artifacts; security-analysis literature for rule engines and graph platforms. Research code attached to papers is an entry on equal terms with packaged libraries.
- Search the industrial tooling world: IDE analysis engines and their exposed APIs, code-intelligence platforms, compiler front-ends usable as libraries, and commercial static-analysis platforms — commercial status is a fact to report, not an exclusion rule.
- Expand from every entry found: its stated alternatives and comparisons, reverse dependencies, plugin and rule ecosystems, papers citing it, curated lists containing it, and the intermediate representations or grammar libraries it shares with other tools. Treat well-known entries as hubs for reaching less-visible ones, not as endpoints.
- Deliberately phrase queries for non-Python framings: multi-language static analysis, language-agnostic parsing, source-code metrics for other languages where the tool also handles Python.
- State explicitly in the report any narrowing that was applied (by language, ecosystem, era, or venue) and why.

Verify recency directly: release dates from the package index or repository, archive status, changelog activity. Date every such claim. Prefer primary sources (documentation, papers, repository READMEs) over aggregator articles. Where performance or accuracy numbers are reported, attribute them as self-reported or independently measured, with the source.

## Deliverables

1. The report itself, following the general guidelines' style rules (executive summary, compact table of contents, prose-first, no tables, TTS-readable). Length is whatever the landscape requires; depth proportional to centrality, never below the descriptive schema floor for any entry.
2. A considered-and-excluded list: every candidate encountered during research that was excluded, each with the specific boundary rule that excluded it. Neural code-embedding methods, clone detectors deferred to the similarity report, and mixed-text tooling deferred to the text-with-code report all belong here, named — their absence from the survey body must be visibly deliberate.
3. A matrix accounting section: for each cell of the input-granularity × parse-requirement × output-shape matrix, either pointers to the entries that populate it or an explicit statement that it is empty or sparse.
4. A search log summary: which ecosystems and venues were searched, which expansion moves were used, and any narrowing applied.

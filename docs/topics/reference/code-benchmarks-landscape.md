# LLM coding benchmarks and datasets — landscape reference

**Kind:** reference (accumulator for the code-evaluation benchmark landscape as it bears on
task-set choice for TLC, ELI's verifier suite, and the clean-code staging test). Entries
are dated. Characterizations are the SciSpace agent's readings of survey papers;
identifiers unverified.

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-coding-datasets-and-benchmarks-agent-artifacts-zip_bdc06926-5b5a-465b-b70b-9af7aa5a4fcd_1787423366/` — two report drafts, `enriched_top30.json` (the 30 cited papers
with extracted benchmark/focus/strengths fields), and ~25 search CSVs, two of which carry
per-paper extracted columns for benchmarks mentioned, task types, and strengths/limits.
**`INDEX.md` inside the folder is the file-level index**, including a list of well-known
benchmarks the report does not cover.

---

## 2026-08-22 — SciSpace deep review (undated, ~early 2026)

**Danielle's prompt (verbatim):**

> What are the most frequently used coding datasets/benchmarks for LLMs as of late 2025
> and 2026? What tasks do they focus on and how are they used. What are the general
> strengths and weaknesses of these datasets/benchmarks? How do they relate to very recent
> coding datasets/benchmarks released mid 2025 to early 2026?

Scope answer (verbatim):

> I want to get a general understanding of the coding benchmark space, primarily from the
> perspective of NLP with LLMs use cases as an ML researcher, but I do want to include any
> highly relevant topics from more of the CS program synthesis side of things. Then, I
> want to understand what the possible focus areas are, aiming for qualitative insights and
> breadth more than depth.

**The map the report gives (condensed).**

- *Task taxonomy:* NL→PL (generation), PL→PL (translation, refactoring, repair), PL→NL
  (summarization, documentation), NL→NL; program-synthesis framing: deductive
  (spec-based) vs. inductive (example-based) vs. NL-based.
- *Foundational function-level:* HumanEval (+, -XL, mHumanEval, HumanEvalPack), MBPP (+);
  strengths: standardized execution-based pass@k; weaknesses: standalone toy functions,
  contamination, narrow domains. Multilingual/multitask: MultiPL-E, xCodeEval (11
  languages, 7 tasks, ExecEval), CodeXGLUE. Domain-specific: DS-1000, FullStack Bench (16
  languages), DOMAINEVAL (six domains; up to ~69-point cross-domain gaps), VerilogEval /
  RTLLM.
- *2025–26 directions:* repository-level with evolution awareness (HumanEvo — future-context
  leakage vs. missing context), context requirements (MRG-Bench), dependency
  understanding (DependEval), executable repo tasks (REPOEXEC); dynamic / contamination-
  resistant (LiveCodeBench; SWE-MERA from live GitHub issues); multi-language agentic repo
  eval (SWE-PolyBench, with the Goodhart worry about Python-bug-fix overfitting);
  LLM-generated benchmarks (AutoCodeBench, 20 languages — validity question of LLMs
  evaluating LLMs); long-context code understanding (LONGCODEU, >36.5K tokens); holistic
  infrastructure (BEHELM 2026: robustness, interpretability, fairness, efficiency).
- *Possible focus areas* (her stated goal): function- vs. repository-level; code
  understanding and reasoning (execution prediction, semantics) as distinct from
  generation; translation/transformation/repair; test generation and completion;
  real-world SE lifecycle tasks; agentic evaluation (process, tool use, iteration, not
  just final code).
- *Cross-cutting weaknesses:* contamination arms race; benchmark–reality gap; pass@k
  blind to quality/efficiency/security and to debugging effort; domain and language
  imbalance; evolution/context handling; inconsistent construction methodology.

**Intake notes.**

- Coverage is a reading of ~30 survey and benchmark papers from one combined query, not
  a usage census. Benchmarks absent or only indirect: SWE-bench (Verified), BigCodeBench,
  EvalPlus, APPS, CodeContests, CRUXEval, Aider polyglot, Terminal-Bench; HumanEvalPack's
  Fix/Explain subtasks named once (see `humanevalexplain-results.md`). "Most frequently
  used" was asserted, not measured.
- For this repo's purposes the useful outputs are the contamination and
  construction-methodology critiques (relevant to choosing TLC / ELI task sets — prefer
  LiveCodeBench-style dated problems or HumanEvalPack subtasks with logged outputs over
  raw HumanEval) and the "code reasoning as distinct from generation" line, which is the
  nearest benchmark family to the explain→regenerate protocol.
- Two near-duplicate report drafts exist in the bundle; the pasted one is the 9-section
  final. The 30 reference numbers map 1:1 onto `enriched_top30.json`.

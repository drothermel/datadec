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

## Undated (intake 2026-08-22) — How HumanEval and MBPP are actually prompted (two turns)

**Danielle's prompts.** (1) The Codex paper seems to "just literally pass in the stub"; she
suspects modern practice adds an instruction somewhere (system or otherwise). (2) The same
question for MBPP.

**Response — HumanEval (condensed).** Original: prompt = header + signature + docstring,
completion-mode with stop sequences (`\ndef`, `\nclass`, …); `openai/human-eval` passes
`problems[task_id]["prompt"]` and stores only the completion — no instruction in the
benchmark. Modern instruct variants: OpenAI `simple-evals` prepends "Read the following
function signature and docstring, and fully implement the function described. Your
response should only contain the code for this function." before the stub;
lm-evaluation-harness separates `humaneval` from `humaneval_instruct`; BigCode harness
keeps plain HumanEval as zero-shot completion and has InstructHumanEval with chat
role/template tokens. Rule offered: raw prompt for base models; for chat models report
which condition was used — raw, instruct wrapper, or a named variant — because the wrapper
is a different evaluation condition.

**Response — MBPP (condensed).** Original (Austin et al.) was already instruction-shaped:
few-shot, NL task description + "Your code should satisfy these tests:" + visible asserts,
then the model writes the function; few-shot exemplars in the same format were prepended
(omitted from the paper figure). The original-style template as reproduced in
lm-evaluation-harness issue 2644 (Jan 2025): `You are an expert Python programmer, and
here is your task: {prompt} Your code should pass these tests: … [BEGIN]` with `[DONE]`
as the end delimiter. BigCode instead builds an InCoder-style docstring prompt —
`"""\n{description}\n{test_example}\n"""\n` — a harness choice, not the paper's format.
lm-evaluation-harness has since added a separate "MBPP Instruct" task. Net: HumanEval
original is raw completion; MBPP original is already few-shot instruction + tests; both now
have named instruct variants.

**Intake notes.**

- Directly relevant to the TLC decoder pass (`../../potential-projs/text-latent-code-
  autoencoder.md`): the prompt condition is part of the harness contract. The decoder
  reconstructs code from a representation, so its prompt is a *new* format regardless;
  but the oracle-spec / signature conditions in the TLC-0 condition matrix should state
  whether they use the raw HumanEval stub or an instruct wrapper, since pass rates are
  not comparable across that choice. Record the condition per run, do not pick one
  globally.
- MBPP's visible asserts are in the prompt by design — so for TLC's leakage accounting
  and for ELI's verifier suite, MBPP leaks test cases into the task statement while
  HumanEval leaks only the docstring examples (which are also doctest-style asserts, just
  fewer). Neither benchmark's prompt is test-free; the "hidden test" distinction is about
  the *held-out* split, not the prompt.
- The `[BEGIN]`/`[DONE]` delimiters and the "only the code" instruction are exactly the
  structured-output concern in `structured-output-literature.md`: extraction failures are
  an evaluation-condition artifact and should be counted separately from solution
  failures (the format-vs-content split).
- Claims are from response-cited GitHub READMEs/issues (unverified; harness behaviour
  drifts by version — pin the harness commit when recording a condition). EvalPlus
  (HumanEval+/MBPP+) prompt handling was not covered and is the variant most likely to be
  used in 2025–26 papers.

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

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

## Undated (intake 2026-08-22) — HumanEval's derivative ecosystem, and cross-benchmark overlap/dedup studies (two turns)

**Danielle's prompts.** (1) HumanEval is so widely adopted that many works must build on
it — test augmentation (HumanEval+), extracting the comments or just the stub, using the
snippets for docstring creation, LM- or program-based augmentation — "please explore what
has been done" across tasks, benchmarks, datasets, extensions, and name other test-backed
code datasets with the same kind of afterlife. (2) Are there attempts to catalog or
deduplicate across major coding datasets/benchmarks? "A substantial portion of the data
samples are very very similar functions or compositions of similar functions" — how has
this been studied, what are the conclusions and impact, how would one categorize/cluster.

**Response 1 — HumanEval as a seed dataset (condensed, all IDs response-supplied).**

- *Harder / less gameable:* HumanEval+ / EvalPlus 2305.01210 (~80× tests, LLM + mutation
  input generation); EvoEval 2403.19114 (LLM-evolved into 7 benchmarks / 828 problems:
  difficult, creative, subtle, combine, tool-use, verbose, concise); HumanEval Pro
  2412.21199 (self-invoking: solve base, then a harder problem that uses it);
  contamination-resistant branch — HumanEval_T (template tasks by combinatorial test
  design), DyCodeEval (dynamic prompt variants + DyPass metric), HumanEvalNext (manual
  revision; 2551 vs 1325 assertions) — cited together as 2412.01526 (which paper that ID
  actually is, unchecked).
- *Multilingual:* HumanEval-X 2303.17568 (hand-written C++/Java/JS/Go, 820 samples);
  MultiPL-E 2208.08227 (transpiled to 18 languages; also MBPP); MBXP / Multilingual
  HumanEval; HumanEval-XL 2402.16694 (23 NLs × 12 PLs); mHumanEval (200+ NLs);
  CL-HumanEval (strips function names, variable names, execution examples to isolate
  cross-lingual transfer); HumanEval.jl.
- *Task reformulations on the same problems:* HumanEvalPack (Fix / Explain / Synthesize
  in 6 languages; `fixdocs` variant); InstructHumanEval (docstring → instruction, with a
  no-context setting); HumanEval Infilling (single-line, multi-line, random-span).
  **On docstring generation specifically: no major HumanEval-derived benchmark whose
  headline task is code → docstring was found**; HumanEval-X's README notes the fields
  can be recombined for summarization.
- *Prompt / docstring / comment manipulation:* ReCode 2212.10264 (30+ semantics-preserving
  transformations over docstrings, names, syntax, format; HumanEval + MBPP); NLPerturbator
  / HumanEval-R 2406.19783; HumanEvalComm 2406.00215 (762 ambiguous/inconsistent/
  incomplete descriptions; Communication Rate, Good Question Rate); substrate studies —
  unit-test generation under random comments / animal names / partial docstrings
  (2404.03114; incorrect comments hurt most), LLM docstring reformulation (little
  change), **ShortenDoc** (docstring compression on HumanEval and EvoEval; ~30%
  compression often preserves or improves pass@1).
- *New modalities / domains:* HumanEval-V 2410.12381 (visual context required); Qiskit
  HumanEval 2406.14712; a bio-image-analysis HumanEval-style set (57 prompts).
- *Other families with afterlives:* MBPP (MBPP+ 35× tests, MultiPL-E/MBXP, MBPP Pro);
  SWE-bench (Lite, Verified 500, Multilingual 300, Multimodal 517); BigCodeBench
  (1,140 tasks, Hard ~150, Complete/Instruct, Lite Pro); reactions to HumanEval's limits
  — LiveCodeBench 2403.07974, NaturalCodeBench, CoderEval (230 Py + 230 Java, six
  dependency levels), HumanEvo.

**Response 2 — cataloguing and overlap (condensed).** No master deduplicated map exists;
three lines of work:

- *Benchmark-of-benchmarks audits:* How2Bench 2501.10711 audits 274 code benchmarks —
  62% did not deduplicate or did not say; 81.8% of 2023–24 benchmarks did not address
  contamination; 18% later served as sources for newer benchmarks (lineage propagates
  overlap).
- *Cross-benchmark leakage:* LessLeak-Bench 2502.06215 — 83 SE benchmarks vs
  pretraining corpora, MinHash+LSH then manual verification, 1.7T comparisons; average
  leakage modest (Py 4.8%, Java 2.8%, C/C++ 0.7%) but QuixBugs 100%, BigCloneBench
  55.7%, APPS 10.8%, SWE-bench-Verified 10.6%; on APPS StarCoder-7B 4.4% pass@1 on leaked
  vs 0.9% on non-leaked items. CodeSearchNet vs five downstream sets 2401.07930
  (SourcererCC clone detection, Jaccard fingerprints, duplicate graph; CodeTrans 22.8%,
  Python-150 15.0%, TLC 13.8% — "TLC" there is a code-summarization dataset, not this
  project's acronym); LoRA/prefix tuning more susceptible to leakage than full FT.
  HumanEval/MBPP contamination 2403.04811 — solution-level overlap: HumanEval 12.2% in
  the Pile / 18.9% in The Stack, MBPP 3.6% / 20.8%; StarCoderBase-15.5B 72% on the
  top-10%-most-similar MBPP items vs 22% on the bottom 10%; decontamination narrows the
  StarCoderBase–Pythia gap from 23.8 to 13.9 points. Substring Levenshtein + AST Dolos.
- *Dedup-by-construction:* ContextBench 2602.05892 (4,497 pooled issue tasks → 3,100
  unique via metadata + embedding near-dup + manual review → 1,136); CrossCodeEval (repos
  chosen disjoint from The Stack); StarCoder2 / The Stack v2 2402.19173 (MinHash+LSH,
  5-grams, Jaccard 0.7; Kaggle notebooks shrank 78%).
- *Background:* DéjàVu (~85M unique of 428M GitHub files, ~70% clones; ACM
  10.1145/3133908); Allamanis — metrics inflated up to 100% on duplicated corpora.
- *Proposed clustering scheme:* clone depth (Type 1–4) × granularity (sample / function
  / file / repo / issue) × representation view (prompt, code-lexical, AST, semantic);
  build a multi-layer graph (exact hash, MinHash/LSH, token/AST clone, embedding edges),
  connected components / community detection, human review of borderline clusters; for
  HumanEval-like sets keep **three overlap matrices — prompt↔prompt, code↔code,
  prompt↔code** — since they leak independently.

**Intake notes.**

- *TLC prior art, by relevance:* ShortenDoc is the closest existing thing to the
  compression project's NL-side question (how much docstring can you remove before
  pass@1 moves) and is not yet in `nl-bottleneck-prior-art.md` or the TLC litreview
  plan; it belongs in gate 1 alongside GenDLN. ReCode and NLPerturbator are the
  perturbation-robustness baselines the TLC-0 control tasks should be compared against
  (semantics-preserving prompt transforms with a measured pass-rate delta). CL-HumanEval's
  name-stripping is a ready-made "signature without hints" condition for the condition
  matrix. HumanEvalPack/Explain is already the loop baseline (`humanevalexplain-results.md`).
- *The docstring-generation gap is a real finding:* the response searched for it and found
  no flagship benchmark; `humanevalexplain-results.md` and TLC's Explain baseline are the
  nearest things. Worth a targeted second search before relying on "nobody has done it."
- *Overlap for TLC-0 and IRT:* the three-matrix recommendation maps directly onto the
  leakage accounting (prompt↔code overlap between the representation and the source is
  exactly I(Z→S)); for IRT-style per-item work, near-duplicate items violate local
  independence and should be clustered before fitting — the clone-depth × view scheme is
  a usable pre-processing spec. HumanEval's within-benchmark redundancy ("compositions of
  similar functions") was Danielle's actual question and is **not answered**: every study
  cited measures benchmark↔corpus or benchmark↔benchmark overlap, not within-benchmark
  item similarity. That is a small, doable analysis on 164 items (AST clone + docstring
  embedding clustering) and would feed IRT-11 directly.
- *Response errors/cautions:* 2412.01526 is used for three different contamination-
  resistant benchmarks — at most one is right; 2404.03114's identity is unchecked;
  response-cited "v1/v3" HTML versions mean numbers may be from superseded drafts. All
  percentages are quoted from the response, unverified.

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

## 2026-08-23 — First-hand HumanEvalPlus defects (Danielle, from TLC baseline runs)

Danielle-supplied ground truth (not an external claim), from running HumanEvalPlus in
the TLC reconstruction pipeline after the project cut over to it from the synthetic
families: dramatic variance of difficulty across items; some items solved by every
model tried, down to the cheapest; some items broken in the official release; items
"SUPER short," making description-length compression nearly meaningless for the TLC
bottleneck. Context and the synthetic-library lessons that preceded the cutover:
`../../potential-projs/text-latent-code-autoencoder.md` §4, 2026-08-23 dataset-strategy
entry.

## 2026-08-23 — Efficiency-beyond-correctness benchmarks and DS-1000 (chunk 5 of the early-2026 conversation)

From the historical re-eval conversation (respondent's claims, unverified; only DS-1000
came with an arXiv ID). The efficiency-eval cluster the decision-quality idea must
position against: **ENAMEL** — eff@k metric, expert reference solutions, strong stress-
test generators, distinguishes suboptimal algorithms from suboptimal implementations;
**EvalPerf / DPE** — profiles candidate code against reference solutions at distinct
efficiency levels; **EffiBench** — NeurIPS Datasets & Benchmarks, efficiency relative
to human canonical solutions on LeetCode-style tasks. **DS-1000** (2211.11501) —
execution-based evaluation of real-world data-science code generation
(pandas/numpy/matplotlib/sklearn/scipy/torch), with surface-form constraints (e.g.
must-use-vectorized-ops) that already implement a form of constraint adherence beyond
pass@1; the direct precedent for the data-manipulation task family (TLC doc §4,
chunk-4 entry). Also mentioned: Pydantic ships "Pydantic Evals" as an eval framework
(ecosystem note, not a benchmark paper). Decision-quality framing and de-risking:
`evaluation-methodology-literature.md`, 2026-08-23 chunk-5 entry.

**Chunk-6 addition:** **BigO(Bench)** (Facebook Research; no ID given) — complexity
prediction (given code, predict time/space complexity) and complexity-controlled
generation (generate code meeting a complexity constraint), validated by
profiling/curve-fitting. The nearest prior art to the asymptotic-choice slice of the
decision-quality probe; differs in unit (predict/generate under constraint, not
choose-then-commit under a scenario).

## 2026-08-24 — Danielle's April-2026 task-family taxonomy and evaluation-confound notes (first-hand)

From her 2026-04-17 reflection notes (bundle:
`reflection-april-2026-code-comp-nl-latents.md`, 2026-08-24 Notion batch; some
passages quote an assistant; no arXiv IDs on record — the Notion paper DB holds the
links). Six one-turn task families with representatives: context→missing-code
(RepoBench, RepoCoder, CrossCodeEval); buggy/unsafe→patch (Defects4J, BugsInPy,
QuixBugs, HumanEvalFix, Vul4J/ManyVuls4J/VJBench, HEJ-Robust); code+goal→new code
(JMigBench Java 8→11, SWE-Refactor, CodeTaste, PyCommits/Coeditor, CoEdPilot);
codebase→tests (MultiFileTest — multi-file, three languages); code→answer/trace
(CRUXEval, REval, CodeMMLU, R2-Eval / real-world-context reasoning,
reasoning-quality via "Beyond Output Correctness"); code→label/rank/retrieve
(CodeSearchNet, SecVulEval statement-level C/C++). Candidate suite for the
code-comp task set enumerated in the source. Evaluation-confound flank:
**PartialOrderEval** (in "More Than a Score") — the prompt-detail→pass@1 ladder
(0.280→0.921→0.860 non-monotone at full detail); **Fault in our Stars** (3,566
prompts across 9 benchmarks; benchmark prompt quality is itself a confound);
**ChatGPT non-determinism** (temp 0 ≠ deterministic; 829 problems); HumanExtension
auxiliary-function oracle (pass@1 triples with leaked ground-truth structure).
AVATAR flagged for Java↔Python. All unverified pending the paper-DB cleanup.

# Code compression — related-work landscape (learned embeddings → LLM-based)

**Kind:** reference (accumulator for the code-compression literature TLC positions
against: what "compressing code" has meant, the standard comparisons, and the
approaches close enough to be baselines). Entries are dated. Characterizations and
figures are the SciSpace agent's; identifiers unverified. Siblings:
`prompt-compression-and-optimization-literature.md`, `nl-bottleneck-prior-art.md`,
`humanevalexplain-results.md`.

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-code-compression-agent-artifacts-zip_ae5ad3b7-95bd-4c9c-8caf-394ff76ac5fe_1787423736/` — one report, ~170 search tables (eleven merged corpora with
extracted methodology / findings / evaluation / "frozen LLM encoder-decoder strategy"
columns; ~100 deep-search tables grouped by query stem). **`INDEX.md` inside the folder
is the file-level index**, with a "papers to pull first" list and a warning that the
report's in-text citation numbers are broken.

---

## 2026-08-22 — SciSpace deep review (undated, ~early 2026)

**Danielle's prompt (verbatim):**

> What is the related work on code compression? Both classic approaches that involve
> learning an embedding that can be reconstructed and more recent approaches that somehow
> use LLMs as components in the compression and reconstruction process?

Scope answer (verbatim):

> I'm currently writing a paper on compressing functions using an frozen pretrained black
> box LLMs as encoders and decoders, with instructions to compress the input function and
> then reconstruct it in a way that preserves functionality not surface form. Then, I'm
> interested in a broad overview of the landscape of code compression, both the
> foundational approaches and more recent approaches. Both training models and using LLMs.
> I want to understand the landscape, the types of comparisons that are done, the common
> methods, the standard positioning, the framing of why this would be useful, etc. And I
> want to understand that approaches that are similar to the one that we're proposing to
> understand such that they might be baselines or related work.

**The map the report gives (condensed; citations by paper name because the report's
numbering is broken).**

- *Senses of "code compression" the literature actually uses:* (a) learned embeddings of
  code — autoencoders (Saletta et al. 2021 on Java), contextual embeddings (Kanade et al.
  CuBERT 2020), contrastive (Jain et al. ContraCode 2020), hierarchy-compressed
  transformers (Zhang et al. EMSE 2025), embedding-dimension analyses (Rabin et al. 2020;
  Ding et al. 2022 — do pretrained code embeddings help?); (b) *prompt/context*
  compression for code LLMs — ICAE (Ge et al.), gist tokens (Mu et al. 2304.08467),
  500xCompressor (2408.03094), query-guided compressor (Cao et al. 2406.02376),
  CodePromptZip (He et al. 2502.14925), LongCodeZip (Shi et al.), docstring compression
  (Yang et al. 2410.22793), Ostby "Stingy Context" 18:1 hierarchical compression for
  auto-coding, Johnson "Perplexity Paradox" (code compresses better than math; TAAC);
  (c) *model* compression of code LLMs — Compressor "3 MB" (Shi et al. ASE 2022, 160×),
  LORD low-rank (2309.14021), structural pruning (2412.15921); (d) LLMs as entropy
  coders (Tsai, "Revisiting data compression with language modeling"); (e) compression by
  *abstraction* — library learning, Leroy for imperative languages (Bellur et al.
  2410.06438, 1.04×); (f) *semantic compression* — "Semantic Compression with LLMs"
  (2304.12512), Ong et al. layered contextual pruning; precise rewriting via LLMs
  (Cummins et al. "Don't transform the code, code the transforms", 2410.08806).
- *Theory:* Girish et al. rate–distortion framework for black-box prompt compression
  (2407.15504) — distortion-rate as a linear program, a large gap between current methods
  and the optimum, query-aware variable-rate adaptation; Hinton & Zemel MDL/autoencoders;
  VQ regularization.
- *Standard comparisons:* compression ratio (tokens/bytes), reconstruction quality (exact
  match, BLEU, edit distance, AST similarity), downstream-task retention (CodeXGLUE
  tasks; HumanEval/MBPP pass@k), efficiency (latency, memory), OOD generalization;
  baselines gzip/bzip2, truncation, random/token pruning, LLMLingua-style selective
  context; model baselines CodeBERT / GraphCodeBERT / CodeT5(+).
- *Why-useful framings in use:* context-window fitting, storage/transmission, edge
  deployment, retrieval, analysis.
- *Evaluation gaps the report names:* little rigorous functional testing of
  "functionality-preserving" claims; Python/Java-centric; efficiency rarely reported.

**Maveli, Vergari & Cohen, "Can LLMs Compress (and Decompress)? Evaluating Code
Understanding and Execution via Invertibility"** (SciSpace
https://scispace.com/paper/QfvwlJrWh6EJ). Flagged at intake as a possible near-neighbour
of TLC from its title; **Danielle read it (2026-08-22): it is not.** Her summary: "its
using coding llms to try to forward predict and reverse predict the effect of four
lossless compression models" — i.e. the LLM is asked to *execute* compression algorithms
(and their inverses) on inputs as a code-understanding probe; the compression is the
task, not the method. The agent's 18/100 relevance score was right. Recorded so it is not
re-flagged.

**Intake notes.**

- The report's paradigm 3, "functionality-preserving approaches with frozen LLMs," is
  Danielle's prompt reflected back: the papers filed under it are prompt/context
  compression and library learning, not code autoencoders. Its summary claims ("4x–18x
  with functionality preservation") are unsupported by what it cites.
- In-text citation numbers are reused across sections ([1]/[2]/[3] each point at three
  different papers); resolve by name.
- All ratios and percentages are as the agent reported; unverified.
- Positioning takeaway for TLC: the literature's "compression" is almost entirely
  lossless-or-embedding (a) or context-budget (b); functional-equivalence reconstruction
  through a *text* latent is represented in this bundle by nothing closer than
  HumanEvalExplain (`humanevalexplain-results.md`). Rate–distortion (Girish) supplies the axis
  vocabulary; library learning (Leroy) is the contrast case for "compressing functions."

## 2026-08-22 — Lossless-baseline suite for per-sample Python (three-turn conversation; tools, not papers)

Danielle's prompts (core): test standard lossless compressors on HumanEval ground-truth
samples one by one; only per-sample regimes matter for her compression-vs-correctness plot;
then "focus on all methods, even quite slow ones … consider how well methods perform on
python not just on general text … consider stacking multiple methods or preprocessing to
really try to push the limit." Project-facing consequences are in
`../../potential-projs/text-latent-code-autoencoder.md` §4 (2026-08-22, baseline suite).
Tool inventory as given (unverified):

| Family | Methods | Notes |
|---|---|---|
| Baselines | raw bytes; zlib/DEFLATE 1/6/9; **Zopfli** raw deflate | Zopfli = slow best-DEFLATE point |
| Practical | **zstd** 3/9/19/22; **Brotli** q9/q11; raw **LZMA2** / xz 9e | raw streams, not containers |
| Dictionary | zstd trained dictionary (COVER/fastCover via `python-zstandard`; sweep 256 B–128 KB); zlib `deflateSetDictionary`; Brotli shared dictionary (RFC 9841) | the centerpiece for short code; train on a separate corpus in the same representation |
| BWT | bzip2; **libbsc** | different family, unlikely to win |
| Text modeling | **PPMd** (7-Zip; `pyppmd` Variant H, order 2–64) | strong on short text/code |
| Context mixing | **paq8px**, ZPAQ, **cmix** (≥32 GB RAM) | slow ceilings |
| Neural | **NNCP** (Transformer-based) | research ceiling; code-LM + arithmetic coding is the custom version (Delétang et al. 2309.10668 for the principle) |
| Python-aware transforms | `python-minifier`; `tokenize` token codecs (single- and multi-stream); alpha-renaming of locals; `ast.parse` → compact preorder AST stream; grammar-based arithmetic coder over AST symbols; CPython bytecode (version-pinned; `marshal` not stable) ; nearest-reference byte/token/AST diff | stacks of transform → one compressor; never compressor → compressor |

Rules the conversation set: per-sample only (no solid/tar/concatenation); x-axis =
compressed bytes, with raw, compressed, and relative-to-ground-truth lengths reported;
prior fairness (shared, not trained on the target; charge per-family priors; selector
cost in any oracle-vs-realistic pair); source-lossless vs. test-preserving labels;
representation byte lengths as controls.

*Implementation turn.* A Python harness skeleton (dataclasses for representation variants,
compressors, and result rows; raw DEFLATE / raw LZMA2 / zstd ± dictionary / Brotli / PPMd;
`ast.unparse`, token-canonical, compact-AST, and `python-minifier` variants;
per-representation dictionary training; external-tool wrappers; `bsdiff4` reference delta)
is archived, never run, at
`~/drotherm/data/convo-artifacts/2026/2026-08-22-lossless-baseline-harness/` — see its
`INDEX.md` for caveats. Named packages: `zstandard`, `brotli` (vs. `brotlipy` for custom
dictionaries — both import as `brotli`), `pyppmd`, `python-minifier`, `bsdiff4`, `xdelta3`;
Python 3.14's stdlib `compression.zstd` noted.

## 2026-08-22 — General compression taxonomy, the code-correctness ladder, and a Pro-mode related-work search (four turns)

Danielle's four prompts: (1) an in-depth overview of lossless and lossy compression
families clustered by type, purpose, and assumption, including application-specific
codecs; (2) her reading of the code case — lossless is de facto 100% correct so the
question is how far ratio can be pushed; non-code lossy methods should be terrible under
test-pass; unlimited compression time should buy ratio; (3) describe how a Pro-mode search
would find papers with *code → representation → regenerated code* and a measured
correctness criterion (exact program flow modulo names/docstrings, or test equivalence),
since her own searching found only embeddings and prompt compression; (4) execute it,
"focusing ruthlessly" on "does this method actually produce a smaller representation from
code and reconstruct code/behavior with a measured correctness criterion?", no empirical
suite. All attributions are the respondent's; nothing verified here.

**Turn 1 — taxonomy (kept as a two-line summary).** Compression = model/transform + coder.
Clusters by exploited assumption: entropy coding (Huffman, arithmetic/range, ANS
1311.2540); dictionary/LZ (DEFLATE RFC 1951, LZ4/Snappy, zstd RFC 8878 incl. trained
dictionaries, Brotli RFC 7932, LZMA); decorrelation (delta, PNG filters, BWT);
context modeling (PPM, PAQ/cmix, LM + arithmetic coding); delta/dedup (VCDIFF RFC 3284,
rsync, Git packfiles, content-defined chunking); schema-aware (minifiers, AST, columnar);
transform + quantization (JPEG, AAC, AV1/HEVC/VVC); predictive/residual (DPCM, LPC/CELP,
motion compensation); perceptual; learned (JPEG AI, neural audio codecs SoundStream
2107.03312 / EnCodec, video coding for machines); scientific error-bounded (zfp, SZ). Ten
assumptions that make compression work, ending with "learned prior" — the one TLC's
decoder supplies.

**Turn 2 — the correctness ladder for code** (the useful artefact of this conversation):

| Level | What must survive | Typical method | Risk |
|---|---|---|---|
| byte-exact | the `.py` bytes | gzip/zstd/xz/Brotli | none |
| source-equivalent | tokens/AST, not formatting/comments | AST/token serialization | tools expecting exact text |
| runtime-equivalent | behaviour under intended use | strip comments/docstrings, bytecode | Python reflection (`inspect`, `getattr`, `-O`/`-OO` semantics) |
| test-equivalent | the test suite | test-guided minimization / regeneration | overfits tests |
| intent-equivalent | "roughly the same task" | LLM regeneration from a spec | unreliable without validation |

Confirmations with caveats: non-code lossy compression is "the wrong mental model" — code
is a brittle symbolic object, one character flips `authorized`; code-aware losses
(comments, formatting, local renaming, dead code, AST/bytecode, test-guided regeneration)
are the real spectrum, each with Python-specific observability traps. Slower lossless does
buy ratio with diminishing returns; unlimited compute does not beat the information
content; model/dictionary cost must be accounted; corpus-level beats file-level.

**Turn 3 — the search plan.** Inclusion: code → compact representation → reconstructed
code; code → smaller equivalent code; code → AST/IR/bytecode → regenerated; code → latent →
decoded; code → library + residuals; code → reduced program preserving tests. Exclusion:
embeddings without decoders, prompt compression, summarization, model compression,
retrieval, embedded instruction-cache compression. Ten search categories (A strong
lossless baselines; B syntax-directed compression; C tree/grammar compression; D
α-equivalent representations; E minification/obfuscation; F program reduction; G
superoptimization/equality saturation; H library learning/MDL; I neural code autoencoders;
J LLM semantic compression), five passes, an evidence-table schema (input language,
representation, decoder?, reconstruction target, correctness metric, compression metric,
does the model count, times, artifact, Python relevance), and **separate leaderboards per
correctness class** rather than one.

**Turn 4 — results, by cluster (the related-work map).**

*Direct hits: code → compressed representation → reconstruct code/execution.*
- Katajainen, Penttonen & Teuhola 1986, "Syntax-directed Compression of Program Files" —
  parse tree + symbol table; 50–60% gain.
- Evans, "Compression via Guided Parsing" — parser-action stream under CFG predictions;
  functionally equivalent reconstruction.
- **JSZap** (Burtscher, Livshits, Sinha, Zorn; MSR) — JavaScript AST as three streams
  (productions, identifiers, literals); ~10% smaller than gzip; reconstructs code, not
  formatting. *Nearest modern source-language precedent.*
- Stork, Haldar & Franz, "Generic Adaptive Syntax-Directed Compression for Mobile Code" —
  grammar-parameterized AST compression with PPM-style modeling + arithmetic coding;
  5–50% smaller than the best Java-specific scheme.
- Franz & Kistler, Slim Binaries / adaptive syntax-tree compression — >2× denser than Java
  bytecode.
- Ernst, Evans, Fraser, Lucco & Proebsting 1997, "Code Compression" (PLDI) — wire
  representation ~21% of SPARC code for gcc.
- Evans & Fraser, "Bytecode Compression via Profiled Grammar Rewriting" — lcc bytecode
  199 KB → 58 KB (+11 KB interpreter).
- Pugh, "Compressing Java Class Files" — 17–41% of gzipped class files; drops debug
  attributes (lossy w.r.t. metadata, semantics-preserving w.r.t. execution); Pack200 as the
  deployed, since-removed standard.

*Exact lossless baseline.* Boffa et al. 2025, "On the compressibility of large-scale
source code datasets" (JSS) — C/C++/Java/JS/**Python** from Software Heritage; 78 TiB → ~3
TiB (~4%) with context-aware corpus compression; the citation for "exact source
compression is not gzip".

*Test-/property-preserving reduction (the "passes the same tests" match).* C-Reduce
(Regehr et al. PLDI 2012; outputs >25× smaller than prior reducers), Hierarchical Delta
Debugging (Misherghi & Su), Picireny (Python HDD over ANTLR grammars), Perses (syntax-
guided, ICSE 2018), C-Vise (Python port of C-Reduce), J-Reduce (Java bytecode). Framing:
"lossy program minimizers whose correctness is an oracle" — no decoder, the reduced
program *is* the representation.

*Semantics-preserving smaller code at IR/assembly level.* Massalin 1987 superoptimizer;
STOKE 1211.0557 (stochastic search, test + formal verification); Souper 1711.04422 (SMT-
backed LLVM IR; 4.4% smaller Clang); egg / equality saturation (e-graphs + extraction).
Maps to Python only on restricted subsets.

*Library learning / abstraction invention (the ML-adjacent cluster).* DreamCoder
2006.08381 (background); **Stitch** (Bowers et al.; compressivity metric, e.g. 806 → 604);
**BABBLE** 2212.04596 (e-graphs + anti-unification); **LILO** 2310.19791 (LLM synthesis +
Stitch + documentation); **Leroy** 2410.06438 (imperative / Python subset; ~1.04×, slight
expansion including the library — already on file as TLC's contrast case).

*Background only.* Embedded instruction-stream compression (Lekatsas & Wolf; Lin, Xie &
Wolf LZW for VLIW) — why "code compression" searches are dominated by it. LLM cluster:
KoLMogorov Test 2503.13992 (shortest program that outputs a sequence — compression as
code generation); "Semantic Compression with LLMs" (Gilbert et al., on file); LLMZip and
LM-as-compressor as probability models, not code reconstruction.

*Python tooling, not citations.* stdlib compression modules; `python-minifier`;
`pyminifier`/`pyminifier3`; `ast`; **LibCST** (lossless CST — the exact-source
serialization option); `compileall` / `.pyc`; `zipapp`.

**The gap the search claims (near-verbatim):** no obvious standard Python-specific system
takes arbitrary Python source, produces a measured compact semantic representation, and
regenerates α-equivalent or test-equivalent Python with a reconstruction metric; closest
precedents are syntax-directed/AST compression, JSZap, property-preserving program
reduction, and learned library abstraction over DSLs or Python subsets.

**Intake notes.**
- This is the first search in the record that found the *classical* precedents
  (1986–2010 syntax-directed compression, JSZap, Pack200). The SciSpace passes did not.
  They belong in TLC §2 beside Leroy: TLC's intermediate is natural language rather than
  a production stream, and its decoder is stochastic, but the "AST as three entropy-coded
  streams" design is the strongest *lossless* competitor on the rate axis.
- Program reduction is the right citation family for TLC's test-relative guarantee — the
  honest sentence is that TLC is a reducer whose "reduced program" is NL and whose oracle
  is the same test suite, plus a regeneration step.
- The evidence-table schema from turn 3 is the one the TLC litreview plan should adopt
  for subdomain C.
- Eight new arXiv IDs to the ledger; the 1980s–2000s items have no arXiv IDs and need DOI
  or venue checks instead.

## 2026-08-24 — Danielle's April-2026 compression notes (first-hand)

From her 2026-04-17 reflection (bundle:
`reflection-april-2026-code-comp-nl-latents.md`; no IDs on record yet). Method
self-labeling for the TLC/code-comp approach: **black-box** (no model internals) vs
white-box compression; **abstractive** (rewrite) vs extractive, with Cmprsr as the
case for abstraction. **LongCodeZip**: embedding models over-index on surface form
(weak code RAG); compression can *beat* uncompressed source on RepoQA; a 500M model
predicts pack-the-knapsack perplexities for Claude Sonnet — small models match
large-model distributions on some code tasks. **CodePromptZip** (2502.14925,
ledgered): AST-based removal by structural impact, variable names kept as
human-intent carriers; ablates dropping token types against downstream performance
— the existing answer to her token-level-perturbation question. **TRAAC / Think
Right**: adaptive attentive compression as an outer loop. **ShortenDoc / Less is
More** (docstring compression): motivation for task-direct decodings.
**LM-CC / Rethinking Code Complexity**: token-entropy unit boundaries; rewrites
lowering the metric improve downstream pass@1 at constant cyclomatic complexity;
plus the atoms-of-confusion perplexity-spike line ("How do Humans and LLMs Process
Confusing Code?") — the LLM-aware rewriting flank adjacent to behavior-preserving
normalization. All unverified.

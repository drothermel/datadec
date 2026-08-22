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

**The one paper that matters most here.** Maveli, Vergari & Cohen, *"Can LLMs Compress (and
Decompress)? Evaluating Code Understanding and Execution via Invertibility"*
(SciSpace https://scispace.com/paper/QfvwlJrWh6EJ; no DOI/arXiv in the tables) — an
invertibility evaluation of code through an LLM compress→decompress cycle. The agent's
relevance tag: 18/100, "Low Relevance … compression is not the central contribution."
For TLC it is the nearest published protocol, and it was **not surfaced by the
prior-art search that declared the mechanism novel** (`nl-bottleneck-prior-art.md`).
Must be read before any novelty claim; likely a baseline or the primary related work.

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
  through a *text* latent is represented only by Maveli et al. and by HumanEvalExplain
  (`humanevalexplain-results.md`). Rate–distortion (Girish) supplies the axis
  vocabulary; library learning (Leroy) is the contrast case for "compressing functions."

# Natural-language-bottleneck code autoencoder — prior-art search record

**Kind:** reference (the prior-art and literature-grounding record for
`../../potential-projs/text-latent-code-autoencoder.md`, TLC). Entries are dated. Verdicts
and paper characterizations are the SciSpace agent's; nothing re-read here except as
noted. Danielle calls this bundle "some of the most important citations … for my future
work".

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-nl-latents-rw-agent-artifacts-zip_cc4d31ce-8970-4a5a-9cd1-248b327a0b06_1787423020/` — 100 files; **`INDEX.md` inside the folder is the file-level
index** (verdict docs, ICBINB grounding docs, her own ICLR-2026 ICBINB draft PDF and the
agent's read of it, deep dives on Latent Programmer / Sentence Bottleneck / ICAE,
candidate JSONs, and ~60 search CSVs grouped by query family).

---

## 2025-12-10 / 2026-01-30 / ICBINB pass — three agent sessions (intake 2026-08-22)

**Danielle's brief (verbatim core).** "Your task is to perform an exhaustive conceptual and
literature-level search for research that matches *any* of the following ideas…": a
program is transformed into a short natural-language latent ("language bottleneck") by a
programmatic transform, a prompt, or both; a frozen LLM decodes it to reconstruct the
program, reproduce its behavior via tests, or emulate its semantics; encoder and decoder
prompts/templates are optimized without weight updates (RL, evolutionary, prompt search,
any gradient-free method); the goal is a compressed, semantically meaningful
representation of code supporting reconstruction, equivalence tests, clustering,
hierarchical planning. Required for a match: language bottleneck as explicit latent;
frozen decoder; search-only optimization of both sides; semantic compression of code (not
summarization or embedding learning). Scope answers to the agent's clarifying questions:

> 1. Prioritize the exact mechanism (frozen LLM in decoder and encoder where the harness
> for the LLMs is what is optimized for the best text latent) 2. include results from
> anywhere this pattern exists. 3. Focus on preprints and published content across all
> years and venues including workshops and arxiv 4. include all optimization methods even
> if they aren't explicitly framed that way as long as they don't involve updating LLM
> weights

**Verdict (Dec 2025, 378 papers):** "No publication matches the described method —
appears novel." Six-component rubric (NL latent; frozen decoder; program encoding;
prompt/template optimization; gradient-free; reconstruction goal). Best partial:
Nano-Capsulator 4/6 (no program encoding, no reconstruction); EPiC 3.5/6 (no autoencoder);
RLPrompt 3/6; Latent Programmer (discrete token latents, trained), Sentence Bottleneck AE
(vector bottleneck, frozen decoder), ICAE (embedding memory slots), SPELL, PCRL as medium
partials. Top-10 list with identifiers in `FINAL_VERDICT.md`.

**Update (Jan 2026, +36 papers Nov 2025–Jan 2026):** still novel; best new match 2/6
("Conversion of Neural Networks into Logic Flows"; "Exploring Reasoning Reward Model for
Agents"). No new citations.

**ICBINB grounding pass (421 papers):** per-component citations and baseline prompts for
the workshop paper — direct generation (AlphaCodium 2401.08500; AceCoder
10.1145/3675395; Structured CoT), compression (Nano-Capsulator; Midolo et al. prompt
guidelines), reconstruction (Misu et al. 2024 Dafny, 10.1145/3643763, three prompt
styles; SelfEvolve 2306.02907), LLM-as-optimizer (Prochemy 2503.11085; EPiC; RL4QE;
MCTS-OPS 2508.05995); a meta-prompt skeleton with JSON-formatted execution feedback;
hyperparameter starting points (T=0 generation/reconstruction, 0.2 compression, 0.7
optimizer; top-p 0.95); datasets HumanEval/MBPP + her synthetic set; a suggested
related-work narrative and novelty framing ("first to optimize encoder and decoder jointly
… first to study reconstruction brittleness systematically"). Fourteen papers with
code/artifacts listed in `REPRODUCIBLE_PAPERS_WITH_CODE.md`.

**Intake notes.**

- **GenDLN** (ACL SRW 2025, DOI 10.18653/v1/2025.acl-srw.92) is the only paper the agent
  scored as a three-keyword "high priority" hit — evolutionary *joint* prompt optimization
  over *stacked* frozen LLMs — and it was dismissed in one line ("not program synthesis, no
  autoencoding"). Structurally, a stacked-LLM joint prompt search is the encoder+decoder
  harness search; it deserves Danielle's own read before the novelty claim is written.
  Unverified beyond the abstract in `high_priority_papers.json`.
- The "95% / 96%" confidence figures are the agent's self-assessment; the "research is
  moving away from your mechanism" trend claim is an artifact of a 36-paper keyword sample.
- Identifier slips in the ICBINB bibliography (EPiC given as both 2408.11198 and
  2410.14321; AlphaCodium as 2401.08500 and 2401.19489; Midolo "2024" with a 2601 arXiv
  number; Nano-Capsulator's first author is Chuang, contradicting the sibling SciSpace
  summary's "Zhou") — check before citing.
- Her own draft is in the bundle (`iclr2026_conference.pdf`, "ICBINB: Code Synthesis and
  Reconstruction", [C]-tagged abstract: high entropy in interfaces/implementations;
  failure to reconstruct through a constrained NL bottleneck, especially cross-model;
  "portable semantic contracts"; evaluation-driven contract/harness optimization). That is
  the same draft whose internals are recorded in the TLC doc §4 (Eq. 4–5, 7; COMP-NL vs
  COMP-SHORT).
- Title-level false alarm, resolved: Maveli, Vergari & Cohen, "Can LLMs Compress (and
  Decompress)?" (surfaced 2026-08-22 from the code-compression bundle) has LLMs
  forward/reverse-predict four lossless compressors as a code-understanding probe;
  Danielle read it — unrelated. The verdict stands as the agent gave it; GenDLN remains
  the one item still needing her read.
- The "six novel aspects" and "proceed with confidence" sections are agent
  editorializing, dropped; the rubric and partial list are kept because they are the
  related-work skeleton.

## 2026-08-22 — second novelty check (different agent, prompt not kept): "equivalent method already published"

Danielle passed only the answer. The response plays her prompt back as: code →
natural-language latent → frozen LLM decoder, optimized by RL/search; it refers to the
idea as "CodeVLAE" (not a term on file here). Verdict: **"Equivalent Method Already
Published … Confidence: High"** — the opposite of the Dec-2025 verdict above. Nothing
below is verified; the response cites by link, not by bibliography.

**Claimed matches, as the response gave them.**

| Item | Response's characterization | Status in this record |
|---|---|---|
| **Language Bottleneck Models** — Berthon & van der Schaar, arXiv 2506.16982 (June 2025) | Optimizable encoder LLM/prompt → short NL summary → *frozen* LLM decoder; encoder trained by group-relative policy optimization to maximize frozen-decoder performance; domain knowledge tracing, offered as a general framework for "inverse problems" with frozen LLMs | **New.** Not in any SciSpace pass. Needs Danielle's read (gate item alongside GenDLN). Weight-trained encoder, one side optimized, no code, prediction rather than reconstruction — on the Dec-2025 rubric that is roughly 3–4/6, but it is the *named* framework |
| **OverLang** — "Teaching LLMs to Speak in Pseudocode for Efficient Compression", `agents4agents.ai/media/OverLang.pdf` | RL-trained "pseudo-language" shorthand for code, decodable by other (possibly frozen) models, "semantic compression" | **New; provenance weak.** No arXiv ID or venue given; the host is not a publisher. Lead only until located |
| "Semantic Compression With Large Language Models" (academia.edu link) | LLMs compress text/code to short representations preserving functional equivalence; manual prompting, no joint optimization | Already on file (Gilbert et al. 2023, 2304.12512 in `code-compression-literature.md`) |
| APRIL 2509.25196; Proof2Silicon 2509.06239 | RL optimizes prompts for frozen LLM code/hardware synthesis | New IDs; the response itself classes them as "the decoder half" |
| Concept Bottleneck LLMs 2412.07992 | Text concepts as bottleneck for classification | New ID; interpretability lineage, not reconstruction |
| Sentence Bottleneck Autoencoders (ACL 2021.emnlp-main.137) | Frozen transformers, soft-vector bottleneck | Already on file (2109.00055) |
| "Text Bottleneck Models (TBM)" — a Hugging Face *search-results* link; "Vision-to-Language Tokenizers" IEEE 10657022 | Discrete text as latent; frozen LLM reading a latent | Not papers as cited (one is a query URL); tangential |

**Response's own novelty-gap paragraph (near-verbatim):** if the system strictly uses
only prompt search for *both* encoder and decoder simultaneously, "this specific
dual-prompt optimization configuration is less common than training a small encoder
network or optimizing just one side. However, it would likely be considered an
implementation detail or a minor variation of the LBM framework rather than a
fundamentally new invention."

**Intake notes.**
- The "High confidence / equivalent method" verdict rests on relaxing two of Danielle's
  four required properties (search-only optimization of *both* sides; code as the
  domain). Same failure class as the SciSpace sessions: adjacent-question substitution
  reported as a match. Recorded, not adjudicated.
- The two verdicts are not contradictory on the facts — both agree no paper does all of
  code + frozen decoder + search-only + both sides — they differ on whether that residual
  is "novel" or "a minor variation". That is the framing question TLC §1 already takes on
  (optimized vs. hand-prompted representation; multi-use representation).
- LBM's 2506 date post-dates the Dec-2025 SciSpace search window only by its own
  absence from that search, not by publication — it should have been found; the keyword
  families in the bundle (`INDEX.md` §5) did not include "language bottleneck model".
- Four new arXiv IDs added to `../../litreview/citation-verification-ledger.md`
  (agent-supplied, unverified).
- **Danielle's read of LBM (2026-08-22, from memory):** very relevant and definitely
  prior work, but the bottleneck grades student responses (non-verifiable target) and the
  paper tries only ~3 prompts — no optimization loop. Gives TLC neither a baseline, a
  method, nor comparative results. Cite as the nearest named framework; the agent's
  "minor variation of LBM" verdict does not hold. Her recollection, not a re-read;
  confirm the prompt count when writing §2.

## Undated (intake 2026-08-22) — measurement literature for the bottleneck: usable information, probing, contrastive code semantics, NL intermediates

**Danielle's prompt (verbatim core).** "I also really like the contrastive directions in
general. Please now do a deep dive into related work that looks at both trying to measure
relatedness or shared info content of language vs code representations, on methods that
try to estimate that relatedness for code or language or embeddings for a given task or
model, and general analysis or bottleneck approaches that might be relevant to either the
analysis or optimization of the bottleneck setup that I've described. Present the related
work and then highlight the 5 most plausible/interesting directions." Notation carried
from the previous turn: X source, Z NL bottleneck, B behavioural facts, S implementation
facts, C given context; target I_𝒱(Z→B | C) minus I_𝒱(Z→S | B,C). All attributions the
respondent's, unverified; this is a *measurement* bibliography, distinct from the
mechanism prior art above.

**Seven clusters.**
1. *Bottleneck theory.* IB (physics/0004057); Deep VIB (1612.00410) as the "you optimize
   bounds, not MI" caution; **Decodable Information Bottleneck** (Dubois et al., NeurIPS
   2020) — IB relative to a predictive family, "almost exactly your setup"; Saxe et al.
   (ICLR 2018) against the IB-theory-of-deep-learning claims — treat IB as a design
   language, not a generalization theorem; rationale extraction (Lei, Barzilay & Jaakkola
   1606.04155; later IB-style rationale objectives) as the text-input, label-output cousin.
2. *Usable information and probing.* 𝒱-information (2002.10689); **conditional probing**
   (Hewitt et al., EMNLP 2021) — information beyond a baseline, here C = signature /
   imports / type hints / "Python knowledge"; probing-as-MI (Pimentel et al., ACL 2020);
   **control tasks** (Hewitt & Liang) — here shuffled behaviour labels, random
   implementation identities, impossible labels; MDL probing (Voita & Titov). Metrics
   given: BehaviorRetained_𝒱(Z) = [H_𝒱(B|C) − H_𝒱(B|Z,C)] / [H_𝒱(B|C) − H_𝒱(B|X,C)];
   Leakage_𝒱(Z) = H_𝒱(S|B,C) − H_𝒱(S|Z,B,C).
3. *NL–code alignment models.* CodeSearchNet 1909.09436 (docstring/code pairs — often
   underspecified relative to behaviour); CodeBERT 2002.08155 (NL–PL probing); GraphCodeBERT
   2009.08366 (data flow as a middle ground between surface and behaviour); CodeT5
   2109.00859; UniXcoder (comments + AST + contrastive). Limitation: trained on
   comments/docstrings, not behavioural equivalence — instruments, not behaviour scores.
4. *Contrastive code semantics* (Danielle's stated interest). CPC/InfoNCE 1807.03748 (a
   lower-bound-like proxy, not MI); **ContraCode** 2007.04973 — identify functionally
   similar variants among distractors via semantics-preserving transforms, "directly
   attacks the minification/formatting problem"; Corder 2009.02731; CoCoSoDa 2204.03293
   and CodeRetriever for NL–code contrastive search. Next step: make labels behavioural —
   positives "different implementation, same tests/properties", hard negatives "similar
   tokens, different edge-case behaviour".
5. *What code models encode.* Troshin & Chirkova 2202.08975 (syntax, identifiers,
   namespaces yes; semantic equivalence poorly); Naik et al. 2207.07706 (RSA on CodeBERT /
   CodeNet: form-based patterns unless fine-tuned on semantic tasks); SVCCA 1706.05806,
   CKA; task-transfer scores LEEP 2002.12462, LogME, Task2Vec ("does Z make this
   behavioural task easy for a light probe?").
6. *Behavioural evaluation.* HumanEval 2107.03374; MBPP 2108.07732; **CodeNet** 2105.12655
   (millions of accepted solutions with I/O tests — many-implementation / same-problem
   clusters); CodeBLEU 2009.10297 and CodeBERTScore as side metrics only; execution-trace
   representations (dynamic program embeddings 1711.07163; FuzzPretrain); LLM-as-judge
   survey 2411.15594 and CodeJudge as members of 𝒱, never ground truth.
7. *NL as intermediate representation.* Intermediate-language study 2407.05411 (NL often the
   most effective intermediate; intermediate correctness only weakly correlates with
   final generation — NL intermediates help without being faithful); NL-Debugging
   2505.15356 (translate buggy code to NL, refine, regenerate — TLC's loop for repair);
   "equivalent representations" of code (OpenReview RMaB6cn07S; comments, pseudocode,
   flowcharts); CoT faithfulness (Turpin et al. 2305.04388) — evaluate what can be
   extracted from Z, not whether Z reads as correct.

**Five suggested directions.** (1) Conditional usable-information benchmark: conditions
C / C+Z / C+minified X / C+X / C+oracle spec; B = test outputs, edge cases, exceptions,
mutation, algebraic properties, complexity facts; S = implementation identity,
identifiers, AST shape, library choice, formatting, exact-source reconstruction.
(2) Behavioural contrastive evaluation with four negative types — same behaviour /
different implementation; similar code / different behaviour; same description /
different edge case; same tests / hidden behavioural difference — run as Z→B (retention)
and Z→S (leakage) games. (3) Decoder-relative behavioural bottleneck score DBS(z) =
log q_D(z,x) − λ(−log p_LM(z)) − γ Leak(z,x) with tests + fuzzing + generated edge cases.
(4) Representation-geometry analysis (CKA/RSA/SVCCA) across X, Z, X̂, B, traces,
AST/data-flow — does Z cluster by implementation, algorithm family, problem statement,
I/O behaviour, or lexical features; collapse-equivalents / separate-near-misses test;
LogME-style probe-ease. (5) Search over bottleneck *formats* — docstring, pseudocode, I/O
examples, edge-case list, pre/postconditions, invariants, algorithm sketch, complexity +
side effects + exceptions — and draw the Pareto frontier of behaviour retained vs.
description length vs. leakage; hunch: the best Z is a compact behavioural contract, not
a docstring. Recommended first prototype: (1) + (2).

**Intake notes.**
- Cluster 4 plus CodeNet answer the "where do behaviour-equivalent positives come from"
  question that InfoNCE needed in the previous turn.
- Direction 5 overlaps the existing TLC §1 optimizer and the "two research programs"
  framing; directions 1–3 are new analysis machinery; direction 4 is the item-embedding
  question from the estimation toolkit applied to representations.
- Twenty-odd IDs added to the ledger; the ones without arXiv numbers in the response
  (Decodable IB, Saxe, Hewitt conditional probing, Pimentel, Hewitt & Liang, UniXcoder,
  CodeRetriever, CKA, LogME, Task2Vec, CodeBERTScore, FuzzPretrain, CodeJudge) need venue
  or ID lookup.

## 2026-08-23 — Miao & Blunsom 2016 (advisor-recommended, PDF in hand)

"Language as a Latent Variable: Discrete Generative Models for Sentence Compression,"
arXiv 1609.07317 (EMNLP 2016; Oxford/DeepMind) — recommended by the advisor during
the February 2026 feedback rounds; Danielle marks it among the artifacts that turned
the discussion. ASC: latent summary sentence drawn from a background language model;
observed sentence reconstructed conditioned on it; discrete VAE via REINFORCE;
semi-supervised by marginalizing the latent text. The closest formal ancestor of the
TLC setup — the explicit LM prior on the latent is the trained-model counterpart of
the frozen-model distribution-compatibility constraint. Metadata verified against the
PDF (bundle: `miao-blunsom-2016-language-as-a-latent-variable.pdf`). Candidate
gate-1 must-read; details in the TLC doc §4 (submission-record entry).

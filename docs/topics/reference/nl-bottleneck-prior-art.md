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

*Identity resolved 2026-08-24: this check is the **Perplexity**-generated Notion
"Lit Review" page created **2026-02-04 02:26** — part of the five-check Feb-2026
set; verbatim source copy now in the 2026-08-24 bundle
(`lit-review-perplexity-novelty-check.md`), including the Sources section this
entry lacked. See the 2026-08-24 Perplexity entry below for what the full source
adds.*

Danielle passed only the answer. The response plays her prompt back as: code →
natural-language latent → frozen LLM decoder, optimized by RL/search; it refers to the
idea as "CodeVLAE" (not a term on file here). Verdict: **"Equivalent Method Already
Published … Confidence: High"** — the opposite of the Dec-2025 verdict above. Nothing
below is verified; the response cites by link, not by bibliography.

**Claimed matches, as the response gave them.**

| Item | Response's characterization | Status in this record |
|---|---|---|
| **Language Bottleneck Models** — Berthon & van der Schaar, arXiv 2506.16982 (June 2025) | Optimizable encoder LLM/prompt → short NL summary → *frozen* LLM decoder; encoder trained by group-relative policy optimization to maximize frozen-decoder performance; domain knowledge tracing, offered as a general framework for "inverse problems" with frozen LLMs | **New.** Not in any SciSpace pass. Needs Danielle's read (gate item alongside GenDLN). Weight-trained encoder, one side optimized, no code, prediction rather than reconstruction — on the Dec-2025 rubric that is roughly 3–4/6, but it is the *named* framework. **2026-08-24 update:** also the "definitive architectural precedent" (~90% overlap claim) of Danielle's 2026-02-04 Gemini novelty check — a two-independent-check headline match (the other: the 2026-02-04 Perplexity check, identified 2026-08-24; both from the same Feb-4 session), still unread |
| **OverLang** — "Teaching LLMs to Speak in Pseudocode for Efficient Compression", `agents4agents.ai/media/OverLang.pdf` | RL-trained "pseudo-language" shorthand for code, decodable by other (possibly frozen) models, "semantic compression" | **New; provenance weak.** No arXiv ID or venue given; the host is not a publisher. Lead only until located. **2026-08-24:** source located — the 2026-02-04 Perplexity check; its "Paper:" link conflates OverLang with the Semantic-Compression page, so provenance remains weak |
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

*Provenance addendum (2026-08-24):* Miao & Blunsom already appears in Danielle's own
2026-02-04 Claude novelty check (component-feasibility list, entry below) — the paper
was in the project's orbit before the advisor's recommendation.

## 2026-08-24 — The 2026-02-03 ChatGPT novelty check (Danielle's Notion "Lit Review" page)

Verbatim copy with resolved citation links:
`~/drotherm/data/convo-artifacts/2026/2026-08-24-notion-lit-reviews/lit-review-nl-bottleneck-prior-art.md`.
ChatGPT-generated two days before the pitch (Notion Date field 2025-12-11);
characterizations are the generating agent's. Verdict: **"published in parts — Yes,"**
against this record's Dec-2025 SciSpace verdict of "appears novel" (378 papers,
six-component rubric) — per Danielle's note (2026-08-24, Consensus entry below), the
verdicts track how each review interpreted the prompt, and hence which literature it
considered.

Its prior-art map, relative to what this record already holds:

- **NL-Debugging (2505.15356)** and **PlanSearch (2409.03733)** as the two
  independent 2024/25 convergences on NL-intermediate + frozen decoder + search —
  both already ledgered via the submitted bibliography; this review is where they
  entered the project.
- **New here — RTC (Allamanis et al., ICML 2024, 2402.08699):** round-trip
  correctness (code → NL summary → code, check semantic equivalence) as an
  *unsupervised evaluation*, explicitly treating the NL description as a compression
  of behavior. Directly TLC-0-relevant (the reconstruction-metric family); added to
  the litreview plan row A.
- **New here — secondary strands:** Self-Debug (Chen et al., 2304.05128) and
  Reflexion (Shinn et al., NeurIPS 2023) — single-model NL-reasoning loops, no
  separated encoder/decoder, no external search over the representation; Yuan et al.
  (Sci. Rep. 15:37300, 2025) — RL-trained query rephraser for a fixed code
  generator; CodePlan (Bairi et al., FSE 2024, 2309.12499) — fine-tuned pseudocode
  planning, blurring encoder/decoder roles.
- **The gap statement** (the pitch's novelty positioning, verbatim in the bundle
  copy): prior art optimizes the encoder/latent heavily but treats the decoder
  prompt as given; no work searches decoder phrasing or tunes a separate frozen
  encoder/decoder pair in tandem.
- **Red-flagged in source** (Notion red highlight, kept unverified): "Wei et al.
  (NeurIPS 2019)" (ledgered as 1910.05923 via the submitted bibliography) and
  "RL4Prompt" (name unresolved; possibly RLPrompt, which row A of the litreview plan
  already lists).

## 2026-08-24 — The 2026-02-04 Consensus novelty check (Danielle's Notion "Lit Review" page #2)

Verbatim copy with resolved citation links:
`~/drotherm/data/convo-artifacts/2026/2026-08-24-notion-lit-reviews/lit-review-consensus-novelty-check.md`.
Consensus-generated one day after the ChatGPT check; characterizations are the
generating agent's. Verdict: **"appears novel," 90–95% confidence**, under a strict
four-component conjunction (explicit NL-bottleneck latent; frozen LLM decoder;
gradient-free encoder/decoder optimization; semantic code compression goal).

**Danielle's meta-note on the whole novelty-check corpus (verbatim, 2026-08-24):**

> "my reading of these different lit reviews is that each interpreted my prompt
> differently and therefore the 'novelty' conclusion varied substantially based on
> what they thought I was proposing and therefore what related literature they
> considered."

Read every verdict in this record through that lens. Concretely here: this review's
closest matches are the prompt/context-compression cluster (SelfCP 2405.17052,
Semantic Compression 2304.12512, LLMLingua 2310.05736, ICAE 2307.06945, NL in the
Middle 2507.08627, Hidden CoT 2409.08561, CodePromptZip 2502.14925, Latent
Programmer) — it never surfaced NL-Debugging or PlanSearch, and the ChatGPT review
never surfaced the compression cluster. Largely disjoint literatures, opposite
verdicts. Its uniform miss-axis across all near-matches is the optimization method
(learned connectors / instruction tuning / fine-tuning, never gradient-free search
over a frozen pair), which restates the pitch's claimed gap from the other side.

Related-but-missing tier: FunSearch (Nature 2023; LLM as generator/mutator, no
bottleneck), LLaMEA 2405.20132, SCoT 2310.10698, compressed-hierarchy code model
(EMSE 2025, no arXiv). Survey-toggle standouts worth pulling forward: **CETBench
2506.04019** (code-equivalence benchmark via program transformations — relevant to
property-indexed equivalence and behavior-preserving normalization) and **Generating
Equivalent Representations of Code by Self-Reflection, 2410.03351**. Authorship
discrepancy for the parked verification pass: 2507.08627 "Wong et al." (ledger, from
the submitted bibliography) vs "Tai, Nie, Golab & Wong, CASCON 2025" (this review).

## 2026-08-24 — The 2026-02-04 Claude novelty check (Danielle's Notion "Lit Review" page #3)

Verbatim copy with resolved citation links:
`~/drotherm/data/convo-artifacts/2026/2026-08-24-notion-lit-reviews/lit-review-claude-novelty-check.md`.
Claude-generated, created four minutes before the Consensus page; characterizations
are the generating agent's. Verdict: **"appears novel," 85–90%**, by explicit
novelty-of-combination over a four-component rubric with a component-feasibility
matrix. The broadest of the three novelty checks: it surfaced the union of the
ChatGPT review's cluster (PlanSearch, RTC, Gilbert) and the Consensus review's
compression cluster (ICAE, gist tokens 2304.08467) plus FunSearch, Latent
Programmer, and Miao & Blunsom — supporting Danielle's prompt-interpretation note
with a third data point: broadest reading, per-component search, combination
verdict.

New to this record from its lists: **Proto-tokens** (Kuratov et al., ACL 2025,
2502.13063 — frozen-LLM reconstruction of ~1.5k tokens from 1–2 trained embeddings;
evidence that reconstruction capacity does not require an NL latent — directly
relevant to the NL-necessity falsifiers), **CodeT** (2207.10397), **EvoPrompt**
(2309.08532 — distinct from EvoPrompting 2302.14838 already in plan row A),
**de Bruin et al., Autoencoders as Tools for Program Synthesis** (2108.07129 —
program VAE with gradient-free evolutionary search over a *neural* latent; the
non-NL-latent counterpart of the TLC setup), **Self-Planning** (2303.06689),
**Tree of Thoughts** (2305.10601), and the DreamCoder/AutoDoc lineage
(NL-as-documentation vs NL-as-representation distinction). Its closing
recommendations (Gilbert-then-differentiate; RTC as eval-not-learning; FunSearch
minus bottleneck; prompt-opt canon then novel application) are recognizably the
submitted paper's related-work skeleton.

## 2026-08-24 — The 2026-02-04 Gemini novelty check (Danielle's Notion "Lit Review" page #4)

Verbatim copy with resolved citation links:
`~/drotherm/data/convo-artifacts/2026/2026-08-24-notion-lit-reviews/lit-review-gemini-novelty-check.md`.
Gemini-generated, 12 minutes after the Consensus page; characterizations are the
generating agent's. **Fourth distinct verdict: "Partially Novel"** — the
architecture (encoder → NL bottleneck → frozen decoder, gradient-free) is
"established" in LBM 2506.16982 with GRPO 2402.03300; the novelty is the
code-compression application (functional-equivalence rewards, MDL-style objective).
A fourth prompt interpretation selecting a fourth literature, per Danielle's note.

New to this record: **GRPO 2402.03300** (LBM's optimizer — group-relative scoring
of candidate summaries against a frozen decoder, no value network); **SPAE
2306.17842** (frozen-LLM autoencoding of images via *lexical tokens* — an existing
non-prose rung for the NL-likeness axis; the review's "SPAE for Code" analogy);
**LINT** (Assessing the Interpretability of Programmatic Policies with LLMs, no ID
in source — explain→regenerate formalized as an interpretability score; "LINT
provides the metric; the user proposes the optimization loop"); **CodeCloak
2404.09066** (DRL prompt manipulation to *prevent* reconstruction — the adversarial
dual, relevant to the leakage/anti-cheating thread); **RLPrompt 2205.12548** (row
A's bare "RLPrompt" now has its ID); CompLLM 2509.19228; Zip2Zip 2506.01084;
ReFIne 2510.09062. Its three-objectives contrast (LBM interpretability/prediction;
CyclePrompt 2402.08756 generation quality; the proposal compression) is a usable
positioning frame; CyclePrompt is already in the submitted bibliography, likely via
this review.

## 2026-08-24 — The 2026-02-04 Perplexity novelty check: the "second novelty check" identified, and its sources recovered

Verbatim copy:
`~/drotherm/data/convo-artifacts/2026/2026-08-24-notion-lit-reviews/lit-review-perplexity-novelty-check.md`.
The Notion pull identifies the 2026-08-22 "second novelty check" entry above as
Perplexity-generated, 2026-02-04 — so the Feb-2026 set is **five checks in ~7
hours with five distinct verdicts** (ChatGPT "published in parts"; Claude "novel
85–90% by combination"; Consensus "novel 90–95%"; Gemini "partially novel";
Perplexity "equivalent method already published"). The strongest evidence yet for
Danielle's prompt-interpretation note: verdict tracks reading, not literature.

What the full source adds over the August answer-only intake:

- **The Sources section** with IDs the August copy lacked: TBM 2310.19660 (Ludan
  et al. — text bottleneck for interpretable classification) and FAST 2501.09747
  (Pertsch et al. — VLA action tokenization; the frozen-LLM-interprets-latent
  example in its component table). Sentence Bottleneck 2109.00055 and CB-LLM
  2412.07992 confirmed as row-A items.
- **An ID conflict for verification:** its source row links the
  Semantic-Compression page to arXiv **2406.01989**, while the ledger's Gilbert =
  2304.12512 (from the submitted bibliography). One of the two pairings is wrong;
  flagged in the ledger.
- **The Novelty Gap concession, now in full:** simultaneous prompt-search
  optimization of BOTH encoder and decoder prompts is "less common than training a
  small encoder network or optimizing just one side" — but the review deems it "an
  implementation detail or a minor variation of the LBM framework." This is the
  one check that names the pitch's exact configuration and dismisses it; the
  positioning statement should engage this dismissal head-on.

## 2026-08-24 — The 2026-02-04 Undermind novelty check (Danielle's Notion "Lit Review" page #6, with 14 sub-pages)

Verbatim multi-file packet:
`~/drotherm/data/convo-artifacts/2026/2026-08-24-notion-lit-reviews/undermind-prompt-compression-review/`.
The sixth check of the Feb-2026 session (3 minutes after Perplexity) and the
deepest on the prompt-compression flank: 21 core papers, four analysis dimensions,
four timeline eras (detail routed to
`prompt-compression-and-optimization-literature.md`, 2026-08-24 entry). Verdict —
effectively the friendliest deep check: every architectural ingredient is
established *for prompts* (discrete text bottlenecks, frozen decoders,
gradient-free/RL encoder optimization: PCRL, TACO-RL, GPT-C, Cmprsr-via-GRPO), but
"none of the papers compress external structured objects (programs, policies,
proofs) into NL codes whose semantics are defined solely via the LLM's decoding" —
fidelity is task performance, never formal equivalence, and its Recommendations
sub-page concludes the field "appears poised for direct extensions to your target
setting" (replace long-prompt with program/trace; PCRL/TACO-RL-style search;
behavioral fidelity). PCRL's objective (divergence between the frozen model's
output distributions under original vs. compressed prompt) is the nearest formal
cousin of behavioral reconstruction. Contrast pair for positioning: Undermind's
"poised for extension" vs. Perplexity's "minor variation of LBM" — the two deep
checks agree on the ingredients and disagree on whether the extension is a
contribution.

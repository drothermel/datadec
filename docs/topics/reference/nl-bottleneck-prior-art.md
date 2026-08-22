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
- The "six novel aspects" and "proceed with confidence" sections are agent
  editorializing, dropped; the rubric and partial list are kept because they are the
  related-work skeleton.

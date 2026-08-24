# Literature review plan: text-latent code autoencoder (TLC)

Status: approved 2026-08-22 ("yes, TLC is [live] so lets write it"). Companion to
[text-latent-code-autoencoder.md](../potential-projs/text-latent-code-autoencoder.md);
the ICBINB draft is in the prior-art bundle (`iclr2026_conference.pdf`). Method and
infrastructure follow [recipe-featurization-litreview-plan.md](recipe-featurization-litreview-plan.md)
(candidates contract, quality axes, three-workflow orchestration, invariants) and are not
repeated here; this file fixes TLC's subdomains, seeds, gate items, and differences.

Goal: a verified, tiered related-work base for the TLC paper — strong enough to write
the novelty sentence and the baselines section — plus resolution of the specific items
the 2026-08-22 intake left open. Unlike REC's review, this one has a submission on the
other end, so it is bounded: core + supporting tiers fully carded; peripheral tier listed
only.

## Subdomains

| Id | Subdomain | Serves | Seeds already on file |
|---|---|---|---|
| A | Prior art for the mechanism: text/NL bottlenecks with frozen decoders, program autoencoders, language-as-latent | novelty claim; §2 related work | Nano-Capsulator 2402.18700; Latent Programmer 2012.00377; Sentence Bottleneck AE 2109.00055; ICAE 2307.06945; RLPrompt; SPELL 2310.01260; PCRL; **GenDLN** (ACL SRW 2025 — the one unread three-keyword hit); EvoPrompting 2302.14838; **Language Bottleneck Models** 2506.16982 (Berthon & van der Schaar — second novelty check's headline match, unread); Concept Bottleneck LLMs 2412.07992; OverLang (agents4agents PDF, provenance weak); **RTC** 2402.08699 (Allamanis et al., ICML 2024 — round-trip correctness as unsupervised eval; from the 2026-02-03 Notion novelty check, TLC-0-relevant) (`nl-bottleneck-prior-art.md`, bundle `INDEX.md` §4–5) |
| B | Prompt / harness optimization — LLM-as-optimizer and system-level | the optimizer loop; baselines | Canon the reviews missed: APE, OPRO, ProTeGi/APO, TextGrad, DSPy/MIPROv2, GEPA, EvoPrompt; present: Promptbreeder 2309.16797, PromptWizard 2405.18369, Trace/OptoPrime 2406.16218, LLM-AutoDiff 2501.16673, SPRIG 2410.14826, Prochemy 2503.11085, EPiC 2408.11198, MCTS-OPS 2508.05995, RL4QE, SelfEvolve 2306.02907 (`prompt-optimization-landscape.md`, 648-row table in the bundle) |
| C | Code compression and rate–distortion framing | positioning; the length-vs-reconstruction curve | Girish et al. 2407.15504 (rate–distortion for black-box prompt compression); LLMLingua; gist tokens 2304.08467; 500xCompressor 2408.03094; CodePromptZip 2502.14925; LongCodeZip; Leroy 2410.06438 (library learning as the contrast case); Cummins et al. 2410.08806; Delétang "LM is compression" 2309.10668; **classical precedents (2026-08-22 search)**: Katajainen–Penttonen–Teuhola 1986 syntax-directed compression, Evans guided parsing, JSZap (MSR), Stork–Haldar–Franz adaptive AST compression, Ernst et al. 1997 "Code Compression", Evans & Fraser bytecode grammar rewriting, Pugh class files / Pack200; Boffa et al. 2025 corpus compressibility (incl. Python); program reduction C-Reduce / HDD / Picireny / Perses; STOKE 1211.0557, Souper 1711.04422, egg; Stitch, BABBLE 2212.04596, LILO 2310.19791, DreamCoder 2006.08381; KoLMogorov Test 2503.13992; LibCST as the exact-source serialization. Use the evidence-table schema recorded in `code-compression-literature.md` (turn 3) and separate leaderboards per correctness level (`code-compression-literature.md`; `prompt-compression-and-optimization-literature.md`) |
| D | Evaluation protocol and task sets for reconstruct-from-description | TLC-1 census design; metrics | OctoPack / HumanEvalExplain 2308.07124 (the explain→regenerate protocol; only three papers report it); WaveCoder 2312.14187; Szalontai 2405.19032; LiveCodeBench 2403.07974; HumanEval+/MBPP+ (EvalPlus); CRUXEval; AlphaCodium 2401.08500; Misu et al. Dafny 10.1145/3643763; code-feature extraction tooling for latent-length and structure measures (`humanevalexplain-results.md`, `code-benchmarks-landscape.md`, `code-feature-extraction-tooling.md`) |
| E | Elicitation and wrapper-only competence (shared with ELI) | the "apparent capability floors are measurement floors" framing | AlphaCodium; prompt tuning at small scale; ELI's own gate list (`../potential-projs/elicitation-gain.md`) |
| F | Measurement of the bottleneck: usable information, probing, contrastive code semantics, representation similarity, NL intermediates | the analysis layer; the retention/leakage metrics; positives/negatives for the contrastive game | IB physics/0004057; Deep VIB 1612.00410; Decodable IB (Dubois et al. NeurIPS 2020); Saxe et al. ICLR 2018; 𝒱-information 2002.10689; conditional probing (Hewitt et al. EMNLP 2021); control tasks (Hewitt & Liang); MDL probing (Voita & Titov); ContraCode 2007.04973; Corder 2009.02731; CoCoSoDa 2204.03293; CodeBERT 2002.08155; GraphCodeBERT 2009.08366; CodeT5 2109.00859; UniXcoder; Troshin & Chirkova 2202.08975; Naik et al. 2207.07706; SVCCA 1706.05806; CKA; LEEP 2002.12462 / LogME / Task2Vec; CodeNet 2105.12655; execution-trace embeddings 1711.07163; intermediate-language study 2407.05411; NL-Debugging 2505.15356; Turpin et al. 2305.04388 (`nl-bottleneck-prior-art.md`, measurement entry) |

Out of scope: clone detection / code similarity (Report 3 of Danielle's deep-research
series), mixed prose+code handling (Report 1), neural code embeddings as representations.

## Gate items (resolve before writing §2 of the paper)

1. **Read GenDLN** (ACL SRW 2025): stacked-LLM joint prompt optimization is structurally
   the encoder+decoder harness search; decide whether it is prior art, related work, or
   neither. Danielle's read, not an agent's. *Language Bottleneck Models* (2506.16982)
   — resolved 2026-08-22 from Danielle's memory of the paper: prior work and the nearest
   named framework, but a non-verifiable grading target and only ~3 prompts, no
   optimization loop → cite in §2, not a baseline/method/comparison. Confirm the prompt
   count against the PDF when writing. **ShortenDoc** (added 2026-08-22; no identifier
   yet — docstring compression on HumanEval/EvoEval, ~30% compression keeps or improves
   pass@1) is the closest existing work to the compression project's NL-side question:
   locate it, then decide prior art vs. related work the same way.
2. **Fix the identifier slips** in the ICBINB bibliography (EPiC 2408.11198 not
   2410.14321; AlphaCodium 2401.08500 not 2401.19489; Midolo "2024" carries a 2601
   arXiv number; Nano-Capsulator first author Chuang); locate or discard OverLang. Ledger:
   [citation-verification-ledger.md](citation-verification-ledger.md).
3. **Supply the prompt-optimization anchors** (APE, OPRO, ProTeGi, TextGrad,
   DSPy/MIPROv2, GEPA) — none appear in any SciSpace output; verify IDs and position the
   optimizer loop against them (meta-prompting with execution feedback, system-level).
   GEPA arrived with an ID on 2026-08-22 (2507.19457, unverified; `prompt-optimization-landscape.md`).
4. **Decide the benchmark precedent sentence**: HumanEvalExplain is the reconstruct-from-
   explanation protocol; state what TLC adds (length pressure, optimizer loop,
   cross-model encode/decode) and whether TLC-1 runs on HumanEvalPack as a free task set.
5. **Rate–distortion framing**: cite Girish et al. for the axis; decide whether the
   paper reports a distortion-rate curve or a fixed-budget point.
6. **Closed false alarms, do not reopen**: Maveli–Vergari–Cohen "Can LLMs Compress (and
   Decompress)?" (LLMs predicting lossless compressors; Danielle read it — unrelated).

## Artifacts and locations

- Working packet: `~/drotherm/data/.claude/datadec/<YYYY-MM-DD>/<HHMM>-tlc-litreview/`
  with the same `candidates.jsonl` contract, `briefs/<A–E>.md`, `cards/`, `synthesis/`,
  `quality.jsonl`, `run-log.md` as the REC plan.
- Seed sources, all on disk and indexed: the five 2026-08-22 SciSpace bundles under
  `~/drotherm/data/convo-artifacts/2026/` (`scispace-nl-latents-rw-…` — 100 files incl.
  the ICBINB draft; `scispace-code-compression-…` — 173 files; `scispace-prompt-
  optimization-…`; `scispace-prompt-compression-method-papers-…`;
  `scispace-humanevalexplain-results-…`; plus `scispace-coding-datasets-and-benchmarks-…`).
  Seed extraction reads their merged CSVs and `INDEX.md` files rather than re-searching.
- Final review: `docs/potential-projs/text-latent-code-autoencoder-litreview.md`, with
  the citation list keyed to corpus paper IDs; the paper's §2 is written from it.

## Differences from the REC plan

- Bounded by the submission: Workflow 3 runs at most three rounds per subdomain, and
  the completeness critic's follow-up is limited to subdomain A (novelty) and B
  (baselines).
- Identity verification is mandatory for every paper that will appear in the paper's
  bibliography, and it runs *first* on the ledger rows tagged TLC.
- Subdomain E is shared with ELI; its cards are written once and cross-filed.
- Quality scoring is skipped for peripheral-tier papers.

## Invariants

As in the REC plan: identifier-first dedup; `title-only` papers never cited as fact;
one serial corpus writer; no Notion or Paperpile mutation; every worker Opus, fresh,
self-contained, read-only except the designated writer; the orchestrating session reviews
between workflows.

## 2026-08-23 addendum — must-reads from the February-2026 conversation intake (Danielle-approved)

Added by decision at the 2026-08-23 intake walkthrough ("strong yes on miao + Blunsom
and the rest of the must-reads"). Fit noted per item; identifiers from the intake
ledger rows (origins recorded there).

- **Miao & Blunsom 2016, 1609.07317** — *gate 1 must-read, PDF in hand* (bundle:
  `~/drotherm/data/convo-artifacts/2026/2026-08-23-prompt-opt-reeval-aha/`). ASC:
  discrete-VAE sentence compression with an NL latent under a background-LM prior —
  the closest formal ancestor of the TLC setup; advisor-recommended; the LM prior is
  the trained-model counterpart of the frozen-model distribution-compatibility
  argument.
- **DS-1000, 2211.11501** — required before building the data-manipulation task
  family (subdomain: tasks/benchmarks); the precedent a generator must visibly
  improve on (controllability, anti-memorization, scale).
- **BigO(Bench)** (Facebook Research; no ID recorded) — required before any
  complexity-property oracle or choose-then-implement mining; ~1.19M
  complexity-labeled solutions + dynamic inference tooling; friendly last author.
- **e-graphs / equality saturation** (egg line of work; Claude-attributed, verify) —
  the formalism between rewriting and search; recommended three times in the
  February conversation; relevant to canonicalization and macro-token vocabulary.
- **TransCoder 2006.03511 + "Leveraging Automated Unit Tests for Unsupervised Code
  Translation" 2110.06773** — before the cross-language equivalence direction; the
  latter's generate-then-test-filter loop matches the library's spec+verify shape.
- **SymC (PMLR v235 pei24b) + ProgramTransformer** (no IDs recorded) —
  semantics-preserving-transformation prior art for the canonicalization/invariance
  thread.
- **Novelty check: curriculum in prompt optimization** — verify the respondent's
  claim that curriculum/adaptive task scheduling is not a standard component of
  prompt-optimization methods (adjacent: cost-aware evolutionary prompt opt, PMLR
  v293 zehle25a) before it appears as a contribution claim anywhere.
- **Bisimulation metrics (skim level)** — for tool-borrowing only (abstraction-
  quality measures); the intake caveat stands: in single-shot settings this
  degenerates to observational/contextual equivalence — do not adopt as paper
  framing without PL review.

Reminder this addendum exists to serve: Reviewer 3 of the LLA submission dinged the
paper for not discussing GEPA and the prompt-optimization SOTA — the exact omission
gate 1 is designed to prevent.

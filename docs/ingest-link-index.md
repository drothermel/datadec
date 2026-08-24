# Ingest link index

One line per external source ingested into this repo: the link Danielle supplied, what
the source is, and where its content was routed. Lookback aid — the verbatim captures
live in the convo-artifacts bundles; the routing summaries live in
`danielle-inputs.md`. **Standing intake step:** every link-based ingest adds a row here
in the same commit.

Format: `- [Title](url) — what it is → routed: targets.` Grouped by ingest date,
newest group first. Notion view/query parameters are stripped; the page ID in the URL
is the durable part.

## 2026-08-24 — ChatGPT source conversations (retro-provenance)

- [HumanEval-family tooling and leaderboards (ChatGPT, 2026-04-09)](https://chatgpt.com/c/69d7f822-f760-832e-9bde-06213ac73301)
  — one-turn conversation (same evening as the ecosystem/overlap one, ~1h earlier;
  dated from ID timestamp, inferred); tooling map: BigCode Evaluation Harness,
  EvalPlus leaderboard, Big Code Models Leaderboard, Awesome-Code-Benchmark,
  BigCodeBench → routed: new dated entry in
  `topics/reference/code-benchmarks-landscape.md` (resource pointers; REval new
  name); no ledger rows (repos/leaderboards, not citations).

- [HumanEval derivative ecosystem + cross-benchmark overlap/dedup (ChatGPT, 2026-04-09)](https://chatgpt.com/c/69d807bb-7c84-8333-9a18-c182475985ca)
  — two-turn conversation, first ingested 2026-08-22 as an unlinked paste; link + full
  verbatim supplied 2026-08-24; conversation dated from its ID timestamp (inferred).
  Turn 2 = source of the code-datasets lineage/overlap Notion note → routed (08-22):
  `topics/reference/code-benchmarks-landscape.md` (now dated, with provenance note),
  TLC §4 prior-art pointer, 21 ledger rows.

## 2026-08-24 — NotebookLM notebooks

- [Intermediate-representations / invertibility notebook (NotebookLM, undated)](https://notebook.google.com/notebook/beaf7015-f9ee-43e8-80a2-bd0089c5402d)
  — five TLC-core papers: NL in the Middle (content lands for the aggregate
  5.5★ item), NL-Debugging detail, RTCE bijection benchmark, Proof2Silicon/
  PREFACE, the Perplexity Paradox → routed:
  `topics/reference/nl-bottleneck-prior-art.md` (main entry),
  `topics/reference/code-compression-literature.md` (Perplexity Paradox),
  TLC §4 entry + §5 NL-in-the-Middle enrichment + recall corpus (5 lines),
  4 ledger rows + NL-Debugging row note.

- [AI4SE notebook: prompt learning, translation/repair, compression, probing (NotebookLM, undated)](https://notebook.google.com/notebook/04e09363-31e7-4bf5-acb3-a4fddbcc795e)
  — 8 sources: ShortenDoc full detail + method-name dependency, Prochemy
  detail, PromptCS, Fluorine restart-beats-repair, LANTERN cross-language
  repair, RepE correctness probing, TransAgent, summarization-metrics study →
  routed: `topics/reference/code-compression-literature.md` (ShortenDoc +
  metric findings), `topics/reference/prompt-optimization-landscape.md`
  (Prochemy/PromptCS/feedback paradox/LANTERN/RepE/TransAgent), TLC recall
  corpus (6 lines), ShortenDoc ledger-row note + 6 new rows.

- [Tokenization / vocabulary-scaling notebook (NotebookLM, undated)](https://notebook.google.com/notebook/1d9de7a7-815a-4a65-a933-1abde72c1ff4)
  — 11 sources: Hayou √d-rule (embedding LR in the large-vocab regime),
  byte-level cross-tokenizer distillation, BPE-dropout unification, mT5/ByT5
  morphology probing, Llama 3 tokenizer, VocabTailor, Sennrich BPE, mT5 (+2
  off-topic sources flagged) → routed:
  `topics/reference/parametrization-and-hp-transfer.md` (LVP/√d-rule entry,
  embedding-reset relevance), `topics/reference/reinit-and-transfer-literature.md`
  (tokenizer/vocabulary flank: BLD, BPE-dropout, VocabTailor, probing), 8
  ledger rows (2 Claude-added canonical IDs).

- [LLM-driven-optimization notebook (NotebookLM)](https://notebook.google.com/notebook/5439c587-5feb-432d-a779-a23c454ebf25)
  — 25 sources
  on LLMs + classical optimization/EAs/search (standalone-fails-hybrids-win
  verdict; Centaur; evolutionary-beats-RL idea search; MCTS hybrids;
  agent-scaffold optimization; AlphaEvolve 2506.13131 ID fill) → routed:
  `topics/reference/prompt-optimization-landscape.md` (main entry extending the
  LLM-as-optimizer taxonomy), 22 ledger rows (7 with IDs) + AlphaEvolve ID fill.

- [World-models notebook (NotebookLM, undated)](https://notebook.google.com/notebook/ecbd81fa-0e71-42bf-b86d-b1b0f3d32478)
  — two papers: LLMs as text-based world models (From Word to World) and the
  no-model-free-shortcut theory (Richens, plausibly 2506.01622) + convergence
  report → routed: **new accumulator**
  `topics/reference/world-models-literature.md` (+ topics README row), 2 ledger
  rows (one Claude-added ID).

- [Reasoning-mechanisms notebook (NotebookLM, undated, 2025–26-era)](https://notebook.google.com/notebook/19a583a2-ee35-40b7-bc03-cb0710e594bd)
  — four papers: encoder-vs-decoder causal reasoning (NL-vs-NNL ablation), OCR
  generalization/hallucination duality, SFT-vs-RL atomic-skill profiles,
  procedural-knowledge influence functions → routed:
  `topics/reference/generalization-and-ood-literature.md` (main entry),
  `topics/reference/pretraining-to-posttraining.md` (SFT-vs-RL +
  procedural-knowledge, recipe relevance), 4 ledger rows (one Claude-added ID:
  2411.12580).

- [Pretraining optimization-dynamics notebook (NotebookLM, undated)](https://notebook.google.com/notebook/611a0dd1-628d-4b4b-8742-bc6d12767de0)
  — 11 pretraining papers (Power Lines, Step Law, CompleteP, MPL, CPT dynamics,
  mid-training survey, catastrophic overtraining, WD-improves-plasticity,
  LoRA-LR-matters, IB-at-LLM-scale, PTQ robustness) + an MPL/PTQ synthesis
  report → routed: `topics/reference/schedules-and-annealing-literature.md`
  (main entry: PTQ-spikes-at-decay, sqrt-cube decay, CPT, mid-training, IB,
  LoRA-LR), `topics/reference/parametrization-and-hp-transfer.md` (CompleteP
  detail, Power Lines, Step Law), `topics/reference/plasticity.md` (the
  LM-plasticity pair), wsd-suite recall corpus (6 lines), 10 no-ID ledger rows.

- [NL-latents / TLC source-collection notebook (NotebookLM, undated)](https://notebook.google.com/notebook/8d1031d0-06f1-4437-af7e-0cd819fb9695)
  — synthesis over her curated 24-source TLC paper collection (all but three
  names already on record with IDs) + a LILO/library-induction report → routed:
  `topics/reference/nl-bottleneck-prior-art.md` (aggregate-synthesis entry:
  idea-space > token-space, frozen LLMs as universal decompressors; LILO/Stitch/
  AutoDoc depth), TLC §4 entry + recall-corpus lines, 3 no-ID ledger rows
  (LAPS, O'Connor & Andreas, Self-consistency).

- [LLM evaluation / meta-evaluation notebook (NotebookLM, undated, ≥2026 sources)](https://notebook.google.com/notebook/2f70b7f0-4156-4e90-ab47-a34dac85ddc1)
  — data table + two reports over 16 paper sources (incl. DataDecide,
  Signal-and-Noise, model ladders — the program's own foundations); new
  meta-evaluation cluster with IDs (PSN-IRT, EffiEval, ONEBench, Federiakin,
  ResampledBench, SparseEval) → routed: `topics/reference/irt-literature.md`
  (main entry), `topics/reference/evaluation-methodology-literature.md`
  (proxies/perturbation/predictability), irt-reanalysis §5 bullet + recall
  corpus, 14 ledger rows.

- [Continual Learning notebook (NotebookLM, assembled ≥2024-10)](https://notebook.google.com/notebook/040273e8-020f-4926-b7ab-6af42dee3505)
  — data table + source list + two synthesis reports over CoLLAs 2022/23 talks
  (Sutton CBP, Lyle plasticity, Van Roy CCRL, Bing Liu CL/OOD, Rish scale,
  Aljundi pretrained-model CL, Larochelle mobilization), a Harrison
  learned-optimization talk, and her own Roam daily-pages export (2024-10-02);
  first report transcript-garbled (flagged) → routed:
  `topics/reference/plasticity.md` (main entry, cluster by cluster),
  `topics/reference/nonstationarity-accounting.md` (drift taxonomy +
  stream-evaluation vocabulary), 15 no-ID ledger rows.

## 2026-08-24 — Perplexity source conversations

- [LLMs as optimizers; classical ML component slots; toy RL with minimal translation (Perplexity task, undated)](https://www.perplexity.ai/computer/tasks/ddb5d88b-0e76-4664-bdd5-c61b94d23e17)
  — one turn; taxonomy of genuine LLM-as-optimizer work (OPRO, FunSearch,
  Eureka, LLAMBO), component-slot substitutions (ReEvo, LLM-SR, DICL), dual-LLM
  structures nearest her autoencoder framing (Matryoshka, ACING), and the
  toy-RL survey with the outer-loop-vs-inner-loop conclusion → routed:
  `topics/reference/prompt-optimization-landscape.md` (full entry), TLC recall
  corpus (4 lines), minigrid staging placeholder (prior-art pointer only), 11
  no-ID ledger rows.

- [HumanEval derivatives; task-suite fit for compression; cross-dataset composition (Perplexity task, undated ~early April 2026)](https://www.perplexity.ai/computer/tasks/e7748d8a-10b8-4ccf-bc34-f83c840ba770)
  — three turns; turn 1 repeats the 2026-04-09 ChatGPT ecosystem question verbatim
  on a second platform; turns 2–3 are TLC planning material (ClassEval as
  compression testbed; the composition idea, called unexplored; Gilbert ERE/SRE)
  → routed: `topics/reference/code-benchmarks-landscape.md` (full three-turn
  entry), `topics/reference/code-compression-literature.md` (metrics + theory
  anchors), TLC §4 entry + §5 Gilbert enrichment + recall corpus section, 34
  ledger rows + 4 row updates (ENAMEL/EffiBench ID fills, 2412.01526 untangling,
  Gilbert ERE/SRE note).

## 2026-08-24 — Notion lit-review batch

Bundle: `~/drotherm/data/convo-artifacts/2026/2026-08-24-notion-lit-reviews/`
(INDEX.md maps files). Common routing targets abbreviated: TLC =
`potential-projs/text-latent-code-autoencoder.md`; NBPA =
`topics/reference/nl-bottleneck-prior-art.md`; ledger =
`litreview/citation-verification-ledger.md`; plan = `litreview/tlc-litreview-plan.md`.

Review pages (the six Feb-2026 novelty checks + the SciSpace copy):

- [Lit Review — ChatGPT novelty check](https://app.notion.com/p/Lit-Review-2fcde135cd1f80358304da2a8f39ede3)
  — 2026-02-03 ChatGPT check, "published in parts" → routed: TLC §4, NBPA, ledger, plan.
- [Lit Review — Consensus novelty check](https://app.notion.com/p/Lit-Review-2fdde135cd1f8021a3f7d2f93d64bdbc)
  — 2026-02-04 Consensus check, "appears novel" 90–95% → routed: TLC §4, NBPA, ledger;
  carries Danielle's prompt-interpretation note.
- [Lit Review — Claude novelty check](https://app.notion.com/p/Lit-Review-2fdde135cd1f808d9386c7142796f3f7)
  — 2026-02-04 Claude check, "novel 85–90% by combination," broadest literature →
  routed: TLC §4, NBPA (+ Miao & Blunsom provenance),
  `topics/reference/prompt-optimization-landscape.md`, ledger.
- [Lit Review — Gemini novelty check](https://app.notion.com/p/Lit-Review-2fdde135cd1f8010b843eddc15898351)
  — 2026-02-04 Gemini check, "partially novel," LBM as architectural precedent →
  routed: TLC §4/§5, NBPA (+ LBM table row), plan, ledger.
- [Lit Review — Perplexity novelty check](https://app.notion.com/p/Lit-Review-2fdde135cd1f808eabd7fd9b74efdc11)
  — 2026-02-04 Perplexity check, "already published"; identity of the formerly
  anonymous 2026-08-22 second novelty check → routed: NBPA (identity addendum + new
  entry + OverLang note), TLC §4/§5, ledger.
- [Lit Review — Undermind prompt-compression review](https://app.notion.com/p/Lit-Review-2fdde135cd1f80d697d7de94ef4a8b53)
  — 2026-02-04 Undermind 21-paper map + 14 sub-pages, "poised for direct extensions" →
  routed: `topics/reference/prompt-compression-and-optimization-literature.md`, NBPA
  (Undermind-vs-Perplexity contrast pair), TLC §4/§5, ledger.
- [Lit Review — SciSpace (Dec-2025 verdict copy)](https://app.notion.com/p/Lit-Review-2fdde135cd1f804ab0d2e05f44ed95c9)
  — Notion copy of the Dec-2025 SciSpace verdict + Prior Work Supplemental sub-page →
  routed: NBPA (identity addendum settling both scored tables' derivation), TLC §4,
  ledger (PCRL/SAPS ID recoveries).

NBLM summary-table layer (six-column distillations; Source Notes of the reviews):

- [ChatGPT Lit Review Summary Table](https://app.notion.com/p/ChatGPT-Lit-Review-Summary-Table-2fcde135cd1f8064bc7ce06ccae10988)
  — 8 rows, no new papers → routed: NBPA table-layer entry, TLC §4.
- [Claude Lit Review Summary Table](https://app.notion.com/p/Claude-Lit-Review-Summary-Table-2fdde135cd1f80879b75e5490b71facf)
  — 15 rows; gap + tiered-overlap schema (best §5 template) → routed: same.
- [Consensus Lit Review Summary Table](https://app.notion.com/p/Consensus-Lit-Review-Summary-Table-2fdde135cd1f8016b7d5ce46a8b2bec4)
  — 8 rows; prior-art-justification schema → routed: same.
- [Gemini Lit Review Summary Table](https://app.notion.com/p/Gemini-Lit-Review-Summary-Table-2fdde135cd1f80868b03ed296a6e81c2)
  — 7 rows; component-matrix schema → routed: same.
- [Perplexity Lit Review Summary Table](https://app.notion.com/p/Perplexity-Lit-Review-Summary-Table-2fdde135cd1f80cf9cffff9a5d41c73b)
  — 4 rows; OverLang "(blog)" provenance → routed: same + NBPA OverLang note.
- [Undermind Lit Review Summary Table](https://app.notion.com/p/Undermind-Lit-Review-Summary-Table-2fdde135cd1f809e8006d712c35e325e)
  — 8 rows; content mismatch — actually the Dec-2025 rubric's scored list → routed:
  same + mislabel flag in NBPA.
- [SciSpace Lit Review Summary Table](https://app.notion.com/p/SciSpace-Lit-Review-Summary-Table-2fdde135cd1f802ba656c26d12197e35)
  — 18 rows × 12 cols; GPT-C 4.5/6 top score → routed: same + GPT-C elevation in TLC
  §5, ledger (SAPS row).

Chain endpoint and post-batch pages:

- [NL Latents Lit Review Aggregate Summary](https://app.notion.com/p/NL-Latents-Lit-Review-Aggregate-Summary-2fdde135cd1f80469c7ee9293dcbfedc)
  — merged cross-review table, chain endpoint; closest thing to a completed §5
  comparison table → routed: TLC §4/§5 (pointed as capstone), NBPA, ledger.
- [Reflection on NBLM of Code Comp / NL Latents](https://app.notion.com/p/Reflection-on-NBLM-of-Code-Comp-NL-Latents-3c6de135cd1f80678d98de65ef823bed)
  — Danielle's own April-2026 working notes (raw capture 2026-04-17) → routed: TLC
  §4/§5 (PartialOrderEval, cross-model transfer trio),
  `topics/reference/code-compression-literature.md`,
  `topics/reference/code-benchmarks-landscape.md`, ledger (12 no-ID rows).
- [Code Datasets Lineage, Overlap, and Component Analysis](https://app.notion.com/p/Code-Datasets-Lineage-Overlap-and-Component-Analysis-33dde135cd1f804b93e2d120e8f0501f)
  — Danielle's curated lineage/overlap note; her 6-step dedup pipeline → routed:
  `topics/reference/code-benchmarks-landscape.md`, TLC recall corpus,
  `topics/staging/pooled-dedup-code-benchmark.md` (placeholder), ledger.

## 2026-08-22 — MAQA Next Steps

- [MAQA Next Steps](https://app.notion.com/p/3c1de135cd1f815ea18ad1c9776077ca) —
  Notion page of conversation excerpts, pasted one chunk at a time → routed: see the
  per-chunk sections in `danielle-inputs.md` ("MAQA Next Steps (Notion page) — intake
  from 2026-08-22") and the topic accumulators/staging docs they name.

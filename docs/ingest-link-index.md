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

- [HumanEval derivative ecosystem + cross-benchmark overlap/dedup (ChatGPT, 2026-04-09)](https://chatgpt.com/c/69d807bb-7c84-8333-9a18-c182475985ca)
  — two-turn conversation, first ingested 2026-08-22 as an unlinked paste; link + full
  verbatim supplied 2026-08-24; conversation dated from its ID timestamp (inferred).
  Turn 2 = source of the code-datasets lineage/overlap Notion note → routed (08-22):
  `topics/reference/code-benchmarks-landscape.md` (now dated, with provenance note),
  TLC §4 prior-art pointer, 21 ledger rows.

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

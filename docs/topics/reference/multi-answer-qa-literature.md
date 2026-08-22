# Multi-answer QA literature — QAMPARI / QUEST / RoMQA and what came after

**Kind:** reference — a standing accumulator for multi-answer (list) question answering:
datasets, methods, closed-book vs. corpus-grounded results, and evaluation caveats.
**Danielle interest flag:** she worked on this task ~2022 (corpus-only setting: raw
Wikipedia pages, no parsed knowledge base; datasets QAMPARI, QUEST, RoMQA) and wants to know
where the line went and how frontier models do with and especially without a corpus.

Source: excerpts from the Notion page "MAQA Next Steps" (conversation dated 2026-08-16; intake
2026-08-22). The respondent ran a browsing deep-search; numbers and citations below are its
claims and remain **unverified** here. Related: `../staging/wiki-qa-sharding.md`.
---

## 2026-08-16 — Deep-search summary (report body pasted separately)

**Danielle's framing.** Multi-answer QA: one question, a full list of answers assumed
correct ("what movies did Alfred Hitchcock write?"), simple or complex (filtering). Her
subset: access to the raw corpus (Wikipedia pages) but not a knowledge base. Questions: what
research has continued with these datasets or ones like them; current state; and rigorous
demonstrations of how effective frontier models are "without a corpus or with a corpus, but
especially without, because I think that would be the comparison point."

**Headline conclusions (near-verbatim; all figures unverified).**

- The line "has divided into exhaustive list generation, set-compositional retrieval, and
  broader fan-out research"; QAMPARI, QUEST, and RoMQA "now serve somewhat different branches
  of that taxonomy."
- Closed-book frontier models "still struggle to enumerate complete sets. On MoNaCo
  (arXiv 2508.11133), GPT-4o's recall falls to 27.6% for lists containing 101–500 items and
  2.5% above 500."
- "Corpus access demonstrably helps when retrieval explicitly seeks complementary evidence.
  On RI2VER (Findings of ACL 2025), QAMPARI F1 rises from 24.59 for closed-book GPT-4o to
  40.70 for a retrieval-and-verification system. RoMQA rises from 12.20 to 19.24."
- "Naive RAG is not reliably helpful. On MoNaCo, GPT-4o scored 48.98 closed-book, 37.28
  with top-20 BM25 passages, and 58.67 with oracle evidence. The problem is therefore not
  whether a corpus is valuable, but whether the system can find and manage the right
  evidence."
- "Retrieval completeness remains poor. The 2026 RVR system (arXiv 2602.18425) reaches 68.70
  ordinary Recall@100 on QAMPARI, but only 33.70% complete recall."
- "QUEST-style logical performance is fragile. Controlled experiments in *Reproducing Complex
  Set-Compositional Information Retrieval* (arXiv 2605.03824) show methods scoring around
  0.42 Recall@100 on QUEST collapsing below 0.02 on the more controlled LIMIT+ benchmark."
- **Evaluation caveat:** "Several later papers use QAMPARI's easier 'F1-5' protocol, which
  caps recall credit at five answers. Those results should not be treated as evidence of
  exhaustive enumeration." Recall — not precision — is what collapses as lists grow.

Also covered in the full report: FanOutQA, TANQ, WideSearch, LOFT, LLatrieval, LOGICOL,
evaluation contamination, answer-set incompleteness, and a proposed design for a rigorous
modern closed-book-vs-corpus comparison.

**Method note from the respondent.** Treat the three datasets as a citation lineage rather
than searching "multi-answer QA"; keep closed-book / retrieved-corpus / oracle-evidence
results separate; watch for papers that quietly change the task (capped recall).

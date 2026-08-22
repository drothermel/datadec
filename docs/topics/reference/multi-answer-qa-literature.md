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

## 2026-08-16 — The full report

Verbatim copy: [`../../refs/multi-answer-qa-state-of-research-2026.md`](../../refs/multi-answer-qa-state-of-research-2026.md)
("Exhaustive Multi-Answer Question Answering over Unstructured Corpora", with a 16-entry
annotated bibliography). Distilled here; figures remain the report's unverified claims.

**Task definition and evidence conditions.** Set prediction A(q) = {e : e satisfies all
constraints in q}; differs from open-domain QA in coverage, distributed evidence, and
interacting constraints. Four evidence conditions to keep separate: closed-book (parametric
memory — not "no external knowledge", since Wikipedia is in training data), retrieved corpus,
oracle evidence, structured-KB oracle.

**The three datasets (original numbers).** QAMPARI (GeM 2023): ≥5 answers, ~13 avg; best
trained system 32.8 F1; text-davinci-003 closed-book 13.8; BM25 reader 18.8; oracle passage
selection 62.4 on dev. QUEST (ACL 2023): 3,357 set-operation queries, ≤20 entities; T5-Large
dual encoder beat BM25 but mean complete recall@100 only 0.142. RoMQA (Findings EMNLP 2023):
100-candidate setting BART+retrieval 63.8 F1 / 37.9 robust F1, gold evidence 95.0/83.4;
open generation GPT-3 few-shot 4.4 F1 / 0.4 robust.

**How the line continued.**
- *Coverage and verification over relevance ranking:* Joint Passage Ranking (EMNLP 2021);
  LLatrieval (NAACL 2024; ALCE-QAMPARI, F1-5 cap); RI2VER (Findings ACL 2025; inter-passage
  verification, evaluates QAMPARI and RoMQA directly); RVR (arXiv 2602.18425; 25.9M-passage
  Aug-2021 index; QAMPARI complete recall@100 33.70% vs. ordinary 68.70%; zero-shot QUEST
  6.02% / 30.53%). The objective shifts to "select a small set of passages whose *union*
  covers the answer set."
- *Closed-book prompting:* Mallen et al. (Findings NAACL 2024) — knowledge-aware demo
  selection gives small gains; GPT-3.5 EM-F1 ≈15–16 QAMPARI, ≈6 QUEST.
- *Long context:* LOFT (Findings NAACL 2025) — Gemini 1.5 Pro at 128K: QAMPARI multi-target
  recall 0.61 vs. 0.57 specialized; QUEST 0.30 vs. 0.54; relevant docs capped at 5/3.
- *Logic in dense retrieval:* Does Dense Retrieval Understand Boolean Logic? (Findings EMNLP
  2024); LOGICOL (EMNLP 2025); Reproducing Complex Set-Compositional IR (SIGIR 2026): 12
  retrievers + 4 reasoning methods; strong retrievers >0.41 R@100 on QUEST vs. ~0.20 BM25;
  ~0.42 on QUEST → <0.02 on LIMIT+ where lexical ≈0.96 — "some apparent logical competence
  on QUEST comes from semantic or lexical shortcuts."

**Closed-book vs. corpus (within-row comparisons only).** QAMPARI 2023: 13.8 / 18.8–32.8 /
62.4 oracle. QAMPARI in RI2VER: GPT-4o 24.59 / RI2VER-Llama-3.1-70B 40.70. RoMQA in RI2VER:
12.20 / 19.24. MoNaCo: GPT-4o 48.98 / BM25 top-20 37.28 / oracle 58.67. "The best evidence
for corpus value is the oracle gap, not 'RAG' as a label"; closed-book scores are not a clean
reasoning lower bound (memorized facts, contamination).

**Successors.** MoNaCo (TACL 2025; arXiv 2508.11133): 1,315 human-written decomposed
questions, 43.3 unique pages avg (vs. ~13 QAMPARI, 10.5 QUEST); 8,549 intermediate list
questions avg 16.2 answers; closed-book o3 61.18, GPT-5 60.11, Gemini 2.5 Pro 59.11, Claude 4
Opus 55.03; fully correct on only 38.7%; recall 61–66% (2–20 items) → 27.6% (101–500) → 2.5%
(>500). FanOutQA (ACL 2024): 1,034 questions, ≥5 articles each; models <50%, open-book
humans ~85%; large evidence contexts hurt several models. TANQ (TACL 2025): lists → attributed
tables; best baseline 60.7 F1, 12.3 below human. WideSearch (ICLR 2026): 200 web research
tasks filtered to need tools; best agent ~5%. Also ALCE, AmbigQA, GRANOLA-QA, MulTiple.

**Supported claims.** Much stronger closed-book than GPT-3 era; "often know correct items but
stop too early" (recall degrades faster than precision); corpus access can give large gains;
retrieval quality is decisive; logical/set generalization fragile; real exhaustive research
far from solved. **Not supported:** reliable full-set enumeration from memory; long context
replacing retrieval at Wikipedia scale; F1-5 as exhaustiveness evidence; ordinary R@K as
full-coverage evidence; uncontaminated closed-book generalization on old Wikipedia questions;
cross-dataset F1 ranking.

**Evaluation hazards.** Answer-set incompleteness (QAMPARI's manual extension kept rankings
stable, but precision is less secure than recall); metric drift (macro set F1 vs. ALCE F1-5;
RoMQA P@10; LOFT caps); contamination and changing model knowledge; retrieval and generation
entangled — report retrieval coverage, reader coverage given gold, and final set accuracy
separately.

**Recommended study design.** Hidden, freshly authored eval set over a frozen dated
Wikipedia; same frontier models across arms: closed-book; fixed-corpus RAG (reproducible
index + budget); coverage-aware RAG (iterative, conditioned on verified entities/evidence);
oracle evidence with stated context budget; optional structured-KB oracle. Measures: macro
set P/R/F1; complete-set accuracy and complete-recall@K / MRecall@K; evidence recall and
citation entailment per answer; RoMQA-style worst-case over constraint variants;
abstention/calibration, latency, tokens, retrieval calls, cost. Stratify by set size,
evidence pages, operator type, compositional depth, temporal freshness, entity popularity.
Publish exact canonical and alias-expanded metrics; do not cap recall at five. Suite: the
three originals + MoNaCo + LIMIT+ (+ TANQ, WideSearch).

**Bottom line (quoted).** "Frontier models are useful components for exhaustive multi-answer
QA, but neither closed-book generation nor naive RAG is a solved system. High-recall
evidence acquisition, constraint-faithful verification, and completeness-aware evaluation
remain the critical research problems."

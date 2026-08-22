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

## 2026-08-16 — Cleaner datasets: verification over enumeration

**Danielle's question.** Back then, incomplete gold sets mattered little at hundreds of
answers but "if we're looking at doing recall over eight, but only six are actually listed in
the ground truth, then that can dramatically swing the accuracy of the evaluation." Gold
evidence passages were built by question-entity/answer-entity co-occurrence or BM25 hits
containing the answer — many false positives "that could definitely be filtered out fairly
inexpensively now, given how strong our reader models are." What work has improved QA
dataset quality so evaluations differentiate methods more precisely?

**Response (near-verbatim, condensed; citations unverified).** Core shift: "strong models are
more useful as pointwise verifiers than as one-shot authors of a supposedly exhaustive gold
set. Generate candidates broadly; verify each candidate narrowly; escalate disagreements to
humans."

*The original concern was substantial.* QAMPARI's ExtendedSet study (12 min/question, 200
questions) added a median 2 / mean 3.13 / up to 16 answers per question; expanded sets
raised system precision ~5–6 points; rankings stayed stable but the paper acknowledged
ExtendedSet was still incomplete. Its NLI check removed 70% of co-occurrence false positives
at the cost of 7.5% of correct alignments.

*What improved (problem → practice → evidence).* Missing answers → pool candidates from
retrievers/KGs/LMs/submitted systems, adjudicate each (DREAM/BRIDGE). Unsupported answer
strings → judge (question, answer, passage) entailment/extractability/scope (ObliQA-MP,
GaRAGe). Alias/paraphrase errors → semantic, question-aware equivalence (Kamalloo et al.,
PEDANTS, LongRecall). LLM-authored gold lists omit answers → elicit the predicate, check
membership pointwise (*Judging Is Not Enumerating*). Long answers → atomic nuggets
(AutoNuggetizer). Coarse diagnosis → annotate answer, evidence, attribution, sufficiency,
decomposition separately (ExpertQA, GaRAGe, MoNaCo, TANQ).

1. **Pooling and repair — DREAM/BRIDGE (arXiv 2602.06526, ICLR 2026).** Two LLM agents
   initialized with opposing positions; agreement accepted, persistent disagreement to
   humans. Reported 95.2% labeling accuracy, 3.5% human review, 29,824 missing relevant
   chunks recovered (428% of the 6,976 original gold chunks); missing labels changed
   retriever comparisons. List-QA analog: pool candidates from heterogeneous systems → search
   for evidence for *and against* each → independent verifiers judge (question, candidate,
   evidence) → escalate disagreements (not merely low confidence) → re-pool when a novel
   system family is evaluated. "Modernized TREC pooling, made substantially cheaper by
   model-based triage."
2. **Enumeration ≠ verification — *Judging Is Not Enumerating* (arXiv 2608.01000).** On
   tasks with mechanically complete truth sets, models judged membership far better than they
   enumerated; asked for the predicate rather than the extension they approached 0.99 F1
   while enumerations stayed badly incomplete. Bad: "generate every movie … use that as gold."
   Better: formalize constraints, broad candidate pool, test each. Best: compile the
   predicate to a KB query, materialize on a dated snapshot, verify each result in text.
   "Self-verification is not enough if the same model first created an incomplete roster."
3. **Evidence validation — ObliQA-MP (NLLP 2025).** GPT-4.1 classified pairs as directly
   answer-bearing / indirectly supportive / not connected: 20.46% of 31,037 previously
   accepted passages were not connected; only 2,976 of 13,191 candidate multi-passage
   questions survived. Still an automatic precision filter — audit stratified samples
   (accepted, rejected, high-confidence, verifier disagreements, both-entities-but-unsupported).
   GaRAGe (Findings ACL 2025): 2,366 questions, >35k individually annotated grounding passages,
   includes insufficient-evidence cases for abstention.
4. **Answer matching.** Kamalloo et al. (ACL 2023): >half of NQ-Open lexical failures were
   semantically equivalent; manual evaluation raised InstructGPT ~60%. PEDANTS (Findings
   EMNLP 2024); LongRecall (arXiv 2508.15085). For multi-answer: bipartite matching between
   predicted and gold entities, canonical IDs where available, semantic verifier otherwise —
   no alias penalties, no double credit.
5. **Nuggets.** The Great Nugget Recall (arXiv 2504.15068); a 2026 QAMPARI reproduction (GeM
   2026) found it ranks systems well but automatic nugget creation omits required entities
   (recall inflated) and automatic assignment is stricter on aliases (~85% of disagreements
   were automatic rejections humans accepted). Recommendation: human-curated nuggets,
   automatic assignment.
6. **Richer supervision per example.** question; intensional constraints; canonical answer
   entities; aliases; supporting claims per answer; direct evidence spans; additional
   reasoning evidence; source and corpus version; ambiguity and temporal scope; known hard
   negatives. Models: ExpertQA (NAACL 2024; 2,177 expert questions, 32 fields), GaRAGe, TANQ,
   MoNaCo.

*Recommended construction pipeline for a new Wikipedia list-QA dataset.* (1) Closed world:
dated snapshot; gold = answers supported by that snapshot. (2) Store the intensional
specification (type, relation, interval, exclusions, intersections, aggregation). (3)
Overcomplete candidate pool (Wikidata, tables/categories, BM25, dense, link-graph expansion,
multiple LMs, several QA systems) — optimize for recall. (4) Pointwise structured verdicts:
qualifies? canonical identity? direct passage? entailment? temporal qualification? (5)
Adversarial evidence judgments (co-occurring but wrong relation / historical / negated /
attributed elsewhere / mere mention). (6) Independent verifier families + human adjudication
of disagreements and a sample of agreements. (7) Canonicalize by entity ID, redirects,
aliases, granularity rules. (8) Pool system outputs before freezing; versioned supplemental
judgment pool afterwards. (9) Measure annotation coverage: unjudged candidates, marginal
yield per proposal method, capture–recapture estimates of residual incompleteness. (10)
Evaluate four stages separately: candidate-answer recall; evidence-retrieval recall;
correctness given gold evidence; final set P/R and complete-set accuracy.

"The remaining hard problem is not the cost of verifying a proposed answer ... It is ensuring
that the proposal pool is sufficiently diverse that missing answers have an opportunity to
be verified at all."

## 2026-08-17 — Second validation pass on the state of the field

A second deep-search response to Danielle's restated goals (three-paper arc recorded in
`../staging/maqa-oracle-ladder.md`). Near-verbatim, condensed; claims unverified.

- **"F1 is still surprisingly low" — confirmed.** MoNaCo: frontier LLMs at most 61.2% F1,
  "hampered by low recall and hallucinations"; across 15 frontier LLMs (GPT-5, o3, Claude Opus
  4, Gemini 2.5 Pro, DeepSeek-R1, …) the top performer (o3) answered only 38.7% perfectly.
  QAMPARI 2026 retrieval: off-the-shelf retrievers Recall@100 52–62%, MRecall@100 17–26;
  in-domain fine-tuning only to the mid-60s. Set-based retrieval: best average nDCG@10 of
  0.346 on complex-retrieval benchmarks; dense retrievers still fail to represent logical
  relations.
- **Date correction.** MoNaCo is arXiv 2025 / TACL 2026 — not contemporaneous with QAMPARI
  (2022), RoMQA (2022), QUEST (2023); possibly conflated with Break/QDMR (2020). It "is now
  arguably the flagship benchmark for exactly your setting."
- **Tracks.** "Exhaustive / set-based / fan-out" is roughly right but not the field's labels:
  fan-out from FanOutQA (ACL 2024); set operations descend from QUEST (SetBERT and
  follow-ups); exhaustive/complete-recall from QAMPARI. Add a fourth: **agentic /
  deep-research search** — RVR benchmarks against agentic search (LLM alternating reasoning
  and tool calls) and beats it by 10% relative in complete recall on QAMPARI. Newer datasets:
  DeepAmbigQA (2025; answer completeness under ambiguity); conflict-aware MAQA benchmarks
  (all valid answers plus detection of conflicting pairs); RI2VER evaluating on RoMQA and
  QAMPARI.
- **Dataset-quality complaints remain live and mostly unfixed for these datasets.** QUEST's
  authors acknowledge Wikipedia categories have imperfect recall, so false positives may be
  wrongly penalized; RoMQA gold evidence has known coverage gaps. The general problem is
  documented (lexical-matching failures largely from incomplete gold lists; generative
  outputs harder to match); the field's response is LLM-as-judge, which fixes the
  formatting-over-correctness complaint but adds a failure mode: when the gold reference
  conflicts with the judge's parametric knowledge, reference adherence degrades and prompt
  mitigations don't fix it. **No published cleaned QAMPARI/QUEST/RoMQA was found.**
- **The entity-graph approach "held up" — it is now GraphRAG.** Microsoft GraphRAG (LLM-
  induced graph + community summaries); HippoRAG (KG + personalized PageRank from query
  seeds); LightRAG (dual-level graph index); GFM-RAG, HopRAG, RAPTOR, Think-on-Graph. These
  build schemaless graphs by LLM or OpenIE extraction, so LLM extraction has largely replaced
  string matching and surface-form resolution (co-occurrence graphs still noted as noisy).
  Caveat: most GraphRAG work evaluates on 2–4-hop single-answer benchmarks (HotpotQA,
  MuSiQue, 2Wiki), not exhaustive multi-answer sets; graph-based exhaustive retrieval on
  QAMPARI/QUEST/RoMQA/MoNaCo "appears genuinely underexplored" — RVR is retriever-side
  verification, not traversal.

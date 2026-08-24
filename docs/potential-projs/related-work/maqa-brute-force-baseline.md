# maqa brute force baseline — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`maqa-brute-force-baseline.md`](../maqa-brute-force-baseline.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*High-recall corpus of every possibly-relevant paper, method, benchmark, or named
prior-art item on record anywhere in this repository for `MAQA`. Err toward
inclusion; one line each; the curated cut lives in §5 of
`../maqa-brute-force-baseline.md`. **Provenance warning:** essentially the entire
inventory descends from agent-generated records — a browsing deep-search report
(`docs/refs/multi-answer-qa-state-of-research-2026.md`), its distillation
(`multi-answer-qa-literature.md`), and respondent answers in the project doc's §4.
The citation-verification ledger contains **no MAQA rows**, so every ID and number
below is unverified.*

**The three original datasets (the citation lineage the whole project revisits)**

- **QAMPARI — Amouyal et al., "An Open-domain QA Benchmark for Questions with Many
  Answers from Multiple Paragraphs"** (no arXiv ID on record; GeM 2023,
  aclanthology 2023.gem-1.9) — the primary dataset for MAQA-1/MAQA-2; ≥5 answers,
  ~13 avg, 2,000 dev/test + >60k train, Aug-2021 Wikipedia; best original system
  32.8 F1, davinci-003 closed-book 13.8, BM25 reader 18.8, PIG oracle passage
  selection 62.4 dev — the numbers the ceiling ladder is calibrated against
  (source: docs/refs/multi-answer-qa-state-of-research-2026.md;
  docs/topics/reference/multi-answer-qa-literature.md).
- **QAMPARI ExtendedSet study** (inside the QAMPARI paper; no separate ID) — 200
  questions re-annotated at 12 min/question, median +2 / mean +3.13 / up to +16
  answers; precision rose ~5–6 points, rankings stable; its NLI check removed 70%
  of co-occurrence false positives at 7.5% of correct alignments — the direct
  precedent and the diagnostic set for MAQA-1 rung 10 (gold incompleteness)
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **QUEST — Malaviya et al., "A Retrieval Dataset of Entity-Seeking Queries with
  Implicit Set Operations"** (no ID; ACL 2023, 2023.acl-long.784) — 3,357 queries,
  union/intersection/difference, ≤20 entities; T5-Large dual encoder beat BM25 but
  mean complete recall@100 only 0.142; named as a frozen transfer smoke-test slice
  (category intersections, unlike QAMPARI's relation composition)
  (source: docs/refs/multi-answer-qa-state-of-research-2026.md; §4 2026-08-17).
- **RoMQA — Zhong et al., "A Benchmark for Robust, Multi-evidence, Multi-answer
  QA"** (no ID; Findings EMNLP 2023, 2023.findings-emnlp.470) — 100-candidate
  setting BART+retrieval 63.8 F1 / 37.9 robust, gold evidence 95.0/83.4, GPT-3
  open generation 4.4/0.4; the constraint-cluster contrast case and the second
  candidate transfer slice; also the source of the P@10 metric hazard
  (source: docs/refs/multi-answer-qa-state-of-research-2026.md).

**Coverage-and-verification retrieval line (the ladder's retrieval and
evidence-budget rungs, 4–5)**

- **Joint Passage Ranking** (no ID; EMNLP 2021, 2021.emnlp-main.560) — explicitly
  reranked passages to cover *new* answers rather than relevance; the earliest
  recorded ancestor of the oracle set-cover / diversity rung
  (source: docs/refs/multi-answer-qa-state-of-research-2026.md §3.1).
- **LLatrieval: LLM-Verified Retrieval for Verifiable Generation** (no ID; NAACL
  2024, 2024.naacl-long.305) — LLM checks whether retrieved passages suffice and
  iteratively retrieves more; evaluated on ALCE-QAMPARI **under the F1-5 cap**, so
  it is both a coverage-aware-retrieval precedent and a metric-hazard example
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **RI2VER — Dhole et al., "Inter-Passage Verification for Open-Domain QA"** (no
  ID; Findings ACL 2025, 2025.findings-acl.354) — the cleanest recorded
  closed-book-vs-corpus row on the actual datasets: QAMPARI 24.59 (GPT-4o) → 40.70
  (Llama-3.1-70B system); RoMQA 12.20 → 19.24; the comparison point for MAQA-1's
  step-0 closed-book baseline (source:
  docs/topics/reference/multi-answer-qa-literature.md).
- **RVR: Retrieve-Verify-Retrieve for Comprehensive QA** (arXiv 2602.18425, 2026) —
  conditions each retrieval round on already-verified evidence over a 25.9M-passage
  Aug-2021 index; QAMPARI complete recall@100 **33.70%** vs. ordinary Recall@100
  68.70%; zero-shot QUEST 6.02% / 30.53%; the exact ordinary-vs-complete-recall
  distinction the ladder's all-answers-covered@k metric makes; also benchmarks
  against agentic search and beats it ~10% relative in complete recall
  (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17 pass).
- **Agentic / deep-research search as a fourth track** (no ID; named only
  generically — LLMs alternating reasoning and tool calls) — recorded as a track
  MAQA's brute-force floor is implicitly contrasted with (source:
  docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).

**Closed-book, long-context, and set-logic flanks (rung 0 and reader ceilings)**

- **Mallen et al., "Crafting In-context Examples according to LMs' Parametric
  Knowledge"** (no ID; Findings NAACL 2024, 2024.findings-naacl.133) — repurposed
  QAMPARI/QUEST as closed-book multi-answer generation; knowledge-aware demo
  selection gave small gains, GPT-3.5 EM-F1 ≈15–16 QAMPARI / ≈6 QUEST; evidence
  that prompting does not fix coverage (source: docs/refs/… §3.2).
- **LOFT: Can Long-Context LMs Subsume Retrieval, RAG, SQL, and More?** (no ID;
  Findings NAACL 2025, 2025.findings-naacl.374) — Gemini 1.5 Pro at 128K: QAMPARI
  multi-target recall 0.61 vs. 0.57 specialized, QUEST 0.30 vs. 0.54; **relevant
  docs capped at 5 (QAMPARI) / 3 (QUEST)** — a document-cap hazard the ladder's
  reader-input arm must avoid (source: docs/refs/… §3.3).
- **Does Dense Retrieval Understand Boolean Logic?** (no ID; Findings EMNLP 2024,
  2024.findings-emnlp.156) — conjunction/disjunction/negation in dense retrieval;
  background for QAMPARI's intersection/composition question types
  (source: docs/refs/… §3.4).
- **LOGICOL: Logically Informed Contrastive Learning for Set-Compositional
  Retrieval — Zhao et al.** (no ID; EMNLP 2025, 2025.emnlp-main.608) — logic-aware
  contrastive training improving QUEST; relevant to MAQA-opt-4's set-operation
  handling (source: docs/refs/… §3.4).
- **Reproducing Complex Set-Compositional Information Retrieval** (arXiv
  2605.03824; SIGIR 2026) — 12 retrievers × 4 reasoning methods; ~0.42 R@100 on
  QUEST collapsing below 0.02 on LIMIT+ where lexical ≈0.96; the recorded warning
  that apparent logical competence is lexical/semantic shortcut — a caution for any
  ladder conclusion drawn from one dataset (source:
  docs/topics/reference/multi-answer-qa-literature.md).
- **LIMIT+** (no ID; introduced by the above) — controlled logical-generalization
  benchmark, named in the report's recommended suite
  (source: docs/refs/… §8).

**Successor and adjacent benchmarks (MAQA-opt-5 and the paper-time suite)**

- **MoNaCo — Wolfson et al., "More Natural and Complex Questions for Reasoning
  Across Dozens of Documents"** (arXiv 2508.11133; TACL) — MAQA-opt-5's second
  dataset; 1,315 human-written decomposed questions, 43.3 unique pages avg vs. ~13
  QAMPARI / 10.5 QUEST; 8,549 intermediate list questions avg 16.2 answers;
  closed-book o3 61.18 / GPT-5 60.11 / Gemini 2.5 Pro 59.11 / Claude 4 Opus 55.03,
  fully correct on only 38.7%; recall 61–66% (2–20 items) → 27.6% (101–500) → 2.5%
  (>500); GPT-4o 48.98 closed-book / 37.28 BM25 top-20 / 58.67 oracle — the
  naive-RAG-hurts row and the closed-book-as-separate-setting precedent behind
  MAQA-1 rung 0. Date correction on record: 2025 arXiv / TACL 2026, **not** the
  original era (source: docs/topics/reference/multi-answer-qa-literature.md;
  §4 2026-08-17 feedback).
- **FanOutQA — Zhu et al.** (no ID; ACL 2024, 2024.acl-short.2) — 1,034 questions,
  7,305 decompositions, ≥5 articles each, 4,121 articles; models <50% vs. open-book
  humans ~85%; several models got *worse* with large evidence contexts — the
  context-overload result bearing on the reader-ceiling rung (source: docs/refs/… §5).
- **TANQ: An Open Domain Dataset of Table Answered Questions — Wang et al.** (no
  ID; TACL 2025, 2025.tacl-1.23) — QAMPARI-style lists extended to attributed
  tables, 1,395 entries, 6.7 rows × 4 columns avg, best baseline 60.7 F1 (12.3
  below human); a richer-supervision model for the annotation schema
  (source: docs/refs/… §5).
- **WideSearch: Benchmarking Agentic Broad Info-Seeking — Shao et al.** (arXiv
  2508.07999; ICLR 2026) — 200 web-research tasks filtered to require tools; best
  agent ~5% success; external-validity stress test for exhaustive research
  (source: docs/refs/… §5).
- **ALCE — Gao et al., "Enabling LLMs to Generate Text with Citations"** (arXiv
  2305.14627; EMNLP 2023) — the citation-grounded QAMPARI adaptation whose F1-5
  five-answer recall cap is the single most-flagged evaluation hazard in the record
  (source: docs/refs/… §5, §7).
- **AmbigQA** (no ID; EMNLP 2020, 2020.emnlp-main.466) — multiplicity from ambiguity
  rather than an exhaustive set constraint; a contrast case for what MAQA is *not*
  measuring (source: docs/refs/… §5).
- **GRANOLA-QA** (no ID; ACL 2024, 2024.acl-long.365) — valid answers at multiple
  granularities; bears on the answer-normalization / granularity rules in rung 1
  (source: docs/refs/… §5).
- **MulTiple** (no ID; OpenReview qvxjSXiBlLF) — 17,580 time-sensitive multi-answer
  instances; temporal/KB-oriented, relevant to QAMPARI's "ever" and date-restriction
  adjudication strata (source: docs/refs/… §5).
- **DeepAmbigQA** (no ID; 2025) — answer completeness under ambiguity; named in the
  2026-08-17 pass as a newer dataset in the line
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **Conflict-aware MAQA benchmarks** (no ID; unnamed family) — all valid answers
  plus detection of conflicting answer pairs; recorded as an adjacent newer track
  (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).
- **Break / QDMR (2020)** (no ID) — mentioned only as the work MoNaCo may have been
  conflated with in the date correction; listed for completeness
  (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).

**Dataset-cleaning, verification, and gold-incompleteness thread (MAQA-1's noise
audit and MAQA-opt-1)**

- **DREAM/BRIDGE** (arXiv 2602.06526; ICLR 2026) — two LLM agents initialized with
  *opposing* positions; agreement accepted, persistent disagreement escalated to
  humans; 95.2% labeling accuracy, 3.5% human review, 29,824 recovered missing
  relevant chunks (428% of the 6,976 original gold chunks) and missing labels
  changed retriever comparisons — the pooling-and-repair template behind
  MAQA-opt-1's verification-first re-annotation
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **Judging Is Not Enumerating** (arXiv 2608.01000) — models judge set membership
  far better than they enumerate; asked for the *predicate* rather than the
  extension they approach ≈0.99 F1 while enumerations stay badly incomplete; the
  argument for pointwise verification over LLM-authored gold lists, and for
  "self-verification is not enough if the same model authored the roster"
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **ObliQA-MP** (no arXiv ID; NLLP 2025) — GPT-4.1 classified passage pairs as
  directly answer-bearing / indirectly supportive / not connected: **20.46% of
  31,037 previously accepted passages were not connected**; only 2,976 of 13,191
  candidate multi-passage questions survived — the closest precedent for auditing
  QAMPARI's co-occurrence/BM25-derived gold evidence
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **GaRAGe** (no ID; Findings ACL 2025) — 2,366 questions, >35k individually
  annotated grounding passages, includes insufficient-evidence cases for
  abstention; a supervision-schema model and an abstention-evaluation precedent
  for MAQA-opt-4 (source: docs/topics/reference/multi-answer-qa-literature.md).
- **Kamalloo et al., answer-matching evaluation** (no ID; ACL 2023) — more than
  half of NQ-Open lexical failures were semantically equivalent; manual evaluation
  raised InstructGPT ~60% — the direct evidence base for the metric self-test rung
  and the ~50-line answer normalizer
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **PEDANTS** (no ID; Findings EMNLP 2024) — answer-matching / equivalence judging;
  candidate evaluator for the lenient-judge arm of rung 0 and the evaluator rung
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **LongRecall** (arXiv 2508.15085) — recall-oriented answer matching for long/
  multi-part answers; candidate for bipartite predicted↔gold entity matching
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **The Great Nugget Recall** (arXiv 2504.15068) and **its 2026 QAMPARI
  reproduction** (no ID; GeM 2026) — nugget-based evaluation ranks systems well,
  but automatic nugget creation omits required entities (inflating recall) and
  automatic assignment is stricter on aliases (~85% of disagreements were automatic
  rejections humans accepted); recommendation on record: human-curated nuggets with
  automatic assignment — directly bears on MAQA-1's official-vs-audited gap
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **AutoNuggetizer** (no ID) — atomic-nugget decomposition for long answers, named
  as the practice for the "long answers" problem
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **ExpertQA** (no ID; NAACL 2024) — 2,177 expert questions across 32 fields;
  cited as a model for annotating answer, evidence, attribution, sufficiency, and
  decomposition separately (source: docs/topics/reference/multi-answer-qa-literature.md).
- **LLM-as-judge as the field's response to formatting-over-correctness** (no ID;
  generic) — recorded with its own failure mode: when gold conflicts with the
  judge's parametric knowledge, reference adherence degrades and prompt mitigations
  do not fix it — a risk for the lenient-judge arm
  (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).
- **QUEST authors' own caveat** (no ID) — Wikipedia categories have imperfect
  recall, so false positives may be wrongly penalized; **RoMQA gold evidence has
  known coverage gaps** — the standing dataset-quality complaints, unfixed
  (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).
- **"No published cleaned QAMPARI/QUEST/RoMQA was found"** (no ID; a negative
  finding of the 2026-08-17 deep-search pass) — the recorded basis for MAQA-opt-1
  as a standalone resource contribution
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **Capture–recapture estimates of residual incompleteness** (no ID; method named
  in the recommended construction pipeline, step 9) — the annotation-coverage
  measure MAQA-opt-1 adopts
  (source: docs/topics/reference/multi-answer-qa-literature.md).
- **TREC-style pooling** (no ID; named as the ancestor — "modernized TREC pooling,
  made substantially cheaper by model-based triage") — the historical frame for
  MAQA-opt-1's heterogeneous candidate pool
  (source: docs/topics/reference/multi-answer-qa-literature.md).

**Graph / GraphRAG flank (MAQA-opt-3, and the anchor-link vs. string-match ablation)**

- **Microsoft GraphRAG** (no ID) — LLM-induced graph plus community summaries;
  named as what "the entity-graph approach" became; uses hierarchical Leiden for
  navigation (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17;
  cross-recorded in docs/potential-projs/wiki-qa-sharding.md §4).
- **HippoRAG** (no ID) — KG plus personalized PageRank seeded from query entities;
  the closest published analogue to Danielle's original entity-graph traversal
  (source: docs/topics/reference/multi-answer-qa-literature.md;
  docs/potential-projs/wiki-qa-sharding.md §4).
- **LightRAG** (no ID) — dual-level graph index
  (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).
- **GFM-RAG** (no ID) — graph-foundation-model RAG, named in the same list
  (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).
- **HopRAG** (no ID) — hop-structured retrieval, named in the same list
  (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).
- **RAPTOR** (no ID) — hierarchical/recursive summarization tree retrieval
  (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).
- **Think-on-Graph** (no ID) — LLM traversal over a knowledge graph
  (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).
- **The recorded GraphRAG gap claim** (no ID; an agent claim repeated twice) — most
  GraphRAG work evaluates on 2–4-hop *single-answer* benchmarks (HotpotQA, MuSiQue,
  2Wiki), and graph-based exhaustive retrieval on QAMPARI/QUEST/RoMQA/MoNaCo
  "appears genuinely underexplored"; RVR is retriever-side verification, not
  traversal (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17;
  docs/potential-projs/maqa-brute-force-baseline.md §4 2026-08-17 feedback).
- **OpenIE / co-occurrence entity graphs** (no ID; generic) — the schemaless
  extraction route GraphRAG systems use; co-occurrence graphs noted as noisy — the
  contrast for the anchor-link vs. string-matched-mention graph ablation
  (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).
- **Multi-hop single-answer benchmarks named as the GraphRAG evaluation habit:**
  HotpotQA, MuSiQue, 2WikiMultiHopQA (no IDs) — the benchmarks MAQA-opt-3 would
  *not* use; also usable as workload sources per the sibling SHARD doc
  (source: docs/topics/reference/multi-answer-qa-literature.md;
  docs/potential-projs/wiki-qa-sharding.md §1/§4).

**Entity linking prior art (MAQA-opt-2; all respondent claims, unverified)**

- **ReLiK** (no ID; Findings ACL 2024, SapienzaNLP) — retriever–reader linker,
  one transformer pass per chunk, retrieves candidates from a pre-encoded KB and
  resolves all spans jointly; claimed SOTA in/out-of-domain and up to 40× faster;
  no supplied mention boundaries; can retrieve entities whose exact alias is absent
  — the lead candidate for replacing string matching, and one of the three graph
  options for MAQA-1's reachability rung
  (source: docs/topics/reference/entity-linking-at-scale.md).
- **ReFinED** (no ID; NAACL Industry 2022, Amazon) — mention detection +
  fine-grained typing + disambiguation for all mentions in one forward pass, >60×
  faster than contemporaries, links to Wikipedia or >30M Wikidata entities; caveat
  on record: prepackaged entity sets lag the snapshot
  (source: docs/topics/reference/entity-linking-at-scale.md).
- **BELA** (no ID; Meta, 97 languages) — mention detection, passage-level encoding,
  kNN against an entity index, disambiguation, NIL prediction; first multilingual
  comparison; a bake-off arm (source: docs/topics/reference/entity-linking-at-scale.md).
- **`entity-linkings`** (no ID; EACL 2026 demo, NAIST) — unified framework with
  interchangeable candidate retrievers/rerankers (prior- and BM25-based indexes,
  trainable retrieval) — the recorded vehicle for a *controlled* bake-off
  (source: docs/topics/reference/entity-linking-at-scale.md).
- **The "every mention" problem** (no ID; a documented phenomenon, not a paper) —
  Wikipedia-derived training links only the first useful mention, leaves repetitions
  unlinked, underlinks common concepts, never links pronouns; "a linker can have
  good benchmark scores while still missing repeated or low-salience mentions" —
  the reason MAQA-1 rung 5 must compare anchor-link edges against string-matched
  mentions (source: docs/topics/reference/entity-linking-at-scale.md).
- **Alias-prior linking / longest-match title+redirect lookup** (no ID; the
  no-ML baselines) — the simplest-heuristic rungs of MAQA-1 step 2, with alias
  top-M for M ∈ {1, 4, 16, 64} (source:
  docs/potential-projs/maqa-brute-force-baseline.md §1/§4).

**Retrieval and indexing methods named as MAQA's floor-system components**

- **BM25 / Pyserini (Lucene)** (no ID) — the reference lexical baseline for the
  floor system and rung 4; the record warns "'uses BM25' does not guarantee
  baseline equivalence" — validate any other engine against Pyserini
  (source: docs/potential-projs/maqa-brute-force-baseline.md §3;
  docs/topics/reference/retrieval-storage-tooling.md).
- **Reciprocal Rank Fusion (RRF)** (no ID) — the fusion rule for BM25 +
  entity-postings (+ dense) candidates in rung 4 and the MAQA-2 brute-force variant;
  noted that raw BM25 and cosine are not on the same scale
  (source: docs/potential-projs/maqa-brute-force-baseline.md §1;
  docs/potential-projs/wiki-qa-sharding.md §4).
- **Entity-sparse retrieval as a sparse dot product** (no ID; the reframing of
  Danielle's earlier HDF5 entity–page index) — A_{e,d} weights, question as a sparse
  entity vector, IDF modifier, Qdrant `entities` named sparse vector; the MAQA-opt
  entity-sparse index in §3 step 6
  (source: docs/topics/reference/retrieval-storage-tooling.md).
- **WAND / block-max WAND top-k algorithms** (no ID) — named as the reason a search
  engine beats a database join on the head-entity explosion Danielle hit in her
  original system (source: docs/topics/reference/retrieval-storage-tooling.md).
- **CSR + memory-mapped postings; LMDB; RocksDB; Roaring bitmaps** (no IDs) —
  the storage substrate options for the entity→entity graph and exact set operations
  in MAQA-opt (source: docs/topics/reference/retrieval-storage-tooling.md).
- **LanceDB / Qdrant / Vespa / OpenSearch / Milvus / ColBERT-PLAID** (no IDs) —
  the engine survey behind the hybrid-experiment choice named in §3 step 3
  (source: docs/potential-projs/wiki-qa-sharding.md §4;
  docs/topics/reference/retrieval-storage-tooling.md).
- **Maximum coverage / greedy set cover; ILP for exact small-pool values** (no ID;
  classical, named as method) — the oracle budgeted coverage U_{N,K} and the
  minimum-passages-per-question headline plot
  (source: docs/potential-projs/maqa-brute-force-baseline.md §4 2026-08-16 item 5).
- **MRecall-style all-answers-covered@k** (no ID; named as a metric family) — the
  per-question exhaustiveness metric that averages hide; also "complete-set accuracy
  and complete-recall@K/MRecall@K" in the recommended study design
  (source: docs/potential-projs/maqa-brute-force-baseline.md §4;
  docs/refs/multi-answer-qa-state-of-research-2026.md §8).

**Recorded evaluation hazards and protocol precedents (rungs 0, 9, 10)**

- **The F1-5 capped-recall protocol** (ALCE/LLatrieval) — "should not be treated as
  evidence of exhaustive enumeration"; the report's design section says explicitly
  "do not cap recall at five"
  (source: docs/topics/reference/multi-answer-qa-literature.md; docs/refs/… §7–§8).
- **RoMQA P@10 and LOFT document caps** — the two other recorded metric-drift cases
  (source: docs/refs/multi-answer-qa-state-of-research-2026.md §7).
- **Contamination of Aug-2021-Wikipedia-derived benchmarks** — the reason closed-book
  is rung 0 rather than a baseline; "closed-book scores are not a clean lower bound
  on reasoning"; MoNaCo evaluates closed-book as a separate setting
  (source: docs/refs/… §4, §7; docs/potential-projs/maqa-brute-force-baseline.md §4).
- **Retrieval–generation entanglement** — report retrieval coverage, reader coverage
  given gold, and final set accuracy *separately*; the report calls this its own
  prescription and it is the shape of the ladder
  (source: docs/refs/… §7; docs/topics/reference/multi-answer-qa-literature.md).
- **The four evidence conditions** (closed-book / retrieved corpus / oracle evidence
  / structured-KB oracle) — the taxonomy the ladder's arms instantiate
  (source: docs/refs/… §1).
- **The report's recommended five-arm study design** (closed-book; fixed-corpus RAG;
  coverage-aware RAG; oracle evidence; optional structured-KB oracle, with
  stratification by set size, evidence pages, operator type, compositional depth,
  temporal freshness, entity popularity) — the external template MAQA-1's funnel
  parallels (source: docs/refs/… §8).

**Methodology precedents from the repo's own principles record**

- **Bounded dataset-noise work / error-driven annotation** (no ID; methodology, not
  literature) — noise-rate scalars with error bars rather than a "clean dataset";
  union of generic baselines (BM25, dense, closed-book LLM) as sampling frame, never
  the proposed method; **system-blind judging** so "the sampling frame only determines
  coverage, not direction"; freeze and hash `qampari-dev-clean-v1` before the method
  exists; residual bias stated as a lower bound on noise
  (source: docs/topics/reference/project-approach-principles.md, 2026-08-17).
- **The cleaning-and-tooling rabbit-hole guardrail** (no ID) — "the failure mode for
  someone with your instincts isn't over-indexing on QAMPARI; it's the
  cleaning-and-tooling rabbit hole wearing the costume of rigor"; touch the official
  test set rarely; a 50-line normalizer with a test file of known-equivalent pairs
  "is an afternoon" (source: docs/topics/reference/project-approach-principles.md).
- **Sequencing: ship the floor number first, ceilings in the same breath** (no ID)
  — the delivery principle governing MAQA-1 vs. MAQA-2 ordering
  (source: docs/topics/reference/project-approach-principles.md, 2026-08-17).
- **Precision control / knowing when to stop; abstention calibration** (no ID;
  raised as feedback, not a citation) — "with modern LLMs the highest-risk component
  has arguably shifted from 'can it extract answers from clean evidence' to 'does it
  know when to stop'"; QAMPARI's own paper found over-prediction was a real failure
  mode — the origin of MAQA-opt-4
  (source: docs/potential-projs/maqa-brute-force-baseline.md §4 2026-08-17).

**Repo-internal cross-references (not literature, but where more material sits)**

- `docs/topics/reference/multi-answer-qa-literature.md` — the primary accumulator;
  every dataset row, the cleaner-datasets entry, and the 2026-08-17 validation pass.
- `docs/refs/multi-answer-qa-state-of-research-2026.md` — the verbatim external
  deep-search report with a 16-entry annotated bibliography and venue links; header
  states citations are unverified (agent-produced).
- `docs/topics/reference/entity-linking-at-scale.md` — the MAQA-opt-2 flank.
- `docs/topics/reference/retrieval-storage-tooling.md` — index/storage substrate.
- `docs/topics/reference/project-approach-principles.md` — methodology source.
- `docs/potential-projs/wiki-qa-sharding.md` §3–§4 — the engine-choice rationale
  MAQA §3 step 3 defers to, plus the GraphRAG/QA-graph prior-art paragraph.
- `docs/danielle-inputs.md` ("MAQA Next Steps" intake, 2026-08-22) — verbatim prompts.
- `docs/litreview/citation-verification-ledger.md` — **no MAQA rows**; nothing above
  is verified.
- `docs/potential-projs/README.md` and `docs/topics/README.md` §"Multi-answer QA and
  retrieval (MAQA / SHARD)" — program placement (non-pillar; alternative topic
  proposal deferred 2026-08-22).

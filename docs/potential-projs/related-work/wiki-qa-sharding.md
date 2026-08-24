# wiki qa sharding — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`wiki-qa-sharding.md`](../wiki-qa-sharding.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*High-recall corpus of every possibly-relevant paper, method, tool, dataset, or
named prior-art item on record anywhere in this repository for `SHARD`. Err toward
inclusion; one line each. **Provenance warning:** the literature-bearing material is
one dated conversation record (§4 of the project doc, 2026-08-16 "Turn wiki corpus
into shards"), whose own paragraph on QA prior art is headed "unverified
attributions", plus tooling and linking accumulators that carry the same flag. The
citation-verification ledger contains **no SHARD rows**; there is no theme
accumulator devoted to sharding or graph partitioning. Every attribution below is
unverified.*

**Partitioning formulation and algorithm families**

- **Weighted balanced k-way graph partitioning** (no ID; the formulation, not a
  paper) — vertices = articles with storage size s_v and load q_v, edge weights
  w_uv = value of co-placement, minimize cut subject to (1+ε)·Σs_v/k capacity; the
  core objective of SHARD-2; explicitly *not* unconstrained min-cut, whose trivial
  optimum is one giant shard plus leftovers (source:
  docs/potential-projs/wiki-qa-sharding.md §4, 2026-08-16).
- **Minimum bisection** (no ID) — the two-shard special case; named as the
  textbook anchor for the formulation (source: wiki-qa-sharding.md §4).
- **Multilevel heuristics (coarsen → partition → uncoarsen/refine)** (no ID) — the
  practical method family, NP-hardness of the exact problem noted
  (source: wiki-qa-sharding.md §4).
- **METIS** (no ID) — named multilevel partitioner; a SHARD-2 baseline condition
  ("hyperlink-only balanced METIS/KaMinPar") (source: wiki-qa-sharding.md §1/§4).
- **KaHIP** (no ID) — named multilevel partitioner in the same family
  (source: wiki-qa-sharding.md §4).
- **KaMinPar** (no ID) — the partitioner actually named in the SHARD-3 build
  sequence for hyperlink-only balanced k-way
  (source: wiki-qa-sharding.md §1/§3/§4).
- **Hypergraph partitioning with the connectivity objective Σ_q f_q(λ_q − 1)** (no
  ID) — each query's evidence set H_q is one hyperedge, λ_q = shards touched; the
  headline method of the project (source: wiki-qa-sharding.md §1/§4).
- **Mt-KaHyPar** (no ID) — the named hypergraph-partitioning tool implementing the
  connectivity objective; the primary experimental condition
  (source: wiki-qa-sharding.md §1/§3/§4).
- **Spectral / normalized cut** (no ID) — recorded as mathematically insightful
  but expensive at scale; a rejected-with-reason alternative
  (source: wiki-qa-sharding.md §4).
- **Leiden / Louvain community detection** (no ID) — topical communities with
  wildly uneven sizes, needing splitting/packing to meet capacity; the record notes
  **GraphRAG uses hierarchical Leiden for navigation, not physical sharding** — the
  sharpest recorded contrast case for the project's placement framing
  (source: wiki-qa-sharding.md §4).
- **Vertex cuts / replication, cited to PowerGraph** (no ID) — the technique for
  Wikipedia's hub vertices; adds storage and update cost; the ancestor of the
  project's "bounded boundary replication" condition
  (source: wiki-qa-sharding.md §4).
- **Streaming partitioning** (no ID) — lower partition quality but supports updates;
  the method behind SHARD-opt-2 (monthly snapshots)
  (source: wiki-qa-sharding.md §1/§4).
- **Hub downweighting / degree normalization / IDF-like edge weighting** (no ID;
  method) — the affinity-weighting policy so generic links do not dominate
  co-retrieval and evidence-path counts (source: wiki-qa-sharding.md §1/§4).
- **Article-level vs. passage-level granularity** (no ID; design argument) —
  partition at article level, chunk into passages inside shards; passage-level gives
  marginally better cuts but makes single-article questions touch multiple shards
  (source: wiki-qa-sharding.md §4).

**QA-over-graph prior art (the entry's own "unverified attributions" list)**

- **Learning to Retrieve Reasoning Paths over the Wikipedia Graph — Asai et al.**
  (no ID; ICLR 2020) — graph structure used for retrieval/reasoning rather than
  physical placement; the closest named QA-side neighbor
  (source: wiki-qa-sharding.md §4).
- **Multi-step entity-centric retrieval — Das et al.** (no ID; 2019) — iterative
  entity-centric retrieval; the lineage of SHARD-opt-1's routing signal
  (source: wiki-qa-sharding.md §4).
- **CogQA — Ding et al.** (no ID; 2019) — cognitive-graph multi-hop QA; same
  "graph for reasoning, not placement" characterization
  (source: wiki-qa-sharding.md §4).
- **Multi-hop Dense Retrieval — Xiong et al.** (no ID; ICLR) — dense multi-hop
  retrieval; recorded in the same list (source: wiki-qa-sharding.md §4).
- **HippoRAG** (no ID) — entity/passage graphs with personalized PageRank from
  query seeds; the closest published system to a graph-routed retrieval stack, and
  cross-listed in the MAQA GraphRAG record
  (source: wiki-qa-sharding.md §4; docs/topics/reference/multi-answer-qa-literature.md).
- **Microsoft GraphRAG** (no ID) — LLM-induced graph + community summaries; on
  record for hierarchical Leiden navigation and as part of the GraphRAG lineage
  (source: wiki-qa-sharding.md §4;
  docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).
- **LightRAG, GFM-RAG, HopRAG, RAPTOR, Think-on-Graph** (no IDs) — the rest of the
  GraphRAG lineage recorded in the sibling accumulator; relevant to SHARD only as
  graph-use-without-placement contrast cases
  (source: docs/topics/reference/multi-answer-qa-literature.md 2026-08-17).

**Systems / routing neighbors**

- **"Unleashing Graph Partitioning for Large-Scale Nearest Neighbor Search"** (no
  ID; PVLDB) — named as the closest thing on record to graph partitioning paired
  with an explicit router; the nearest systems-side neighbor to SHARD-3
  (source: wiki-qa-sharding.md §4).
- **Workload-aware database placement with partitioning, balancing, and
  replication — Golab et al.** (no ID) — recorded as existing prior art on the
  database side, but not in the Wikipedia-QA combination
  (source: wiki-qa-sharding.md §4).
- **The gap claim itself** (no ID; attributed, not asserted) — "physically shard all
  of Wikipedia by observed QA evidence locality and evaluate end-to-end QA latency
  is not a saturated research problem"; §2 flags it unverified and the open-questions
  list carries "verify the gap claim and the tool/paper citations above"
  (source: wiki-qa-sharding.md §2/§4).
- **The routability caveat** (no ID; design argument, recorded as the headline risk)
  — "co-location only pays if queries route to few shards; if first-stage search
  fans out everywhere, placement helps document fetch and multi-hop expansion only"
  (source: wiki-qa-sharding.md §2/§4).

**Datasets usable as workload hyperedges**

- **HotpotQA** (no ID; ~113k supporting-document sets) — the largest named source of
  QA evidence hyperedges for SHARD-1; also named in the MAQA record as a GraphRAG
  evaluation habit (source: wiki-qa-sharding.md §1/§4;
  docs/topics/reference/multi-answer-qa-literature.md).
- **WikiHop** (no ID) — named as a supporting-document-set dataset
  (source: wiki-qa-sharding.md §4).
- **2WikiMultiHopQA** (no ID) — explicit reasoning paths, so hyperedges carry
  structure (source: wiki-qa-sharding.md §1/§4).
- **MuSiQue** (no ID; 25k questions, 2–4 hops) — named workload source
  (source: wiki-qa-sharding.md §1/§4).
- **QAMPARI evidence sets** (no ID; GeM 2023) — the multi-answer workload for
  SHARD-opt-3's question of whether list-QA evidence is sharding-friendly at all
  (source: wiki-qa-sharding.md §1;
  docs/topics/reference/multi-answer-qa-literature.md).
- **QUEST, RoMQA, MoNaCo** (no IDs; ACL 2023 / Findings EMNLP 2023 / arXiv
  2508.11133) — the rest of the multi-answer workload family on record; MoNaCo's
  43.3 unique pages per question is the most extreme fan-out figure recorded, the
  worst case for evidence-set locality
  (source: docs/topics/reference/multi-answer-qa-literature.md;
  docs/refs/multi-answer-qa-state-of-research-2026.md).
- **Wikimedia clickstream** (no ID; dataset) — an optional affinity layer, flagged
  as a weaker proxy since "browsing locality ≠ QA locality"
  (source: wiki-qa-sharding.md §1/§4).
- **Co-retrieval traces** (no ID; signal, not dataset) — the affinity signal to
  weight up once representative traffic exists (source: wiki-qa-sharding.md §4).

**Corpus and link-graph artifacts surveyed (sizes/schemas unverified)**

- **`wikimedia/structured-wikipedia` Parquet dataset** (no ID; HF) — the chosen
  snapshot-aligned corpus + graph source; links sit recursively inside
  sections/paragraphs, lists, infobox fields, references, each with target `url` and
  anchor `text`; the recommendation is to derive `nodes.parquet`/`edges.parquet` from
  it rather than maintain two ingestion pipelines
  (source: wiki-qa-sharding.md §1/§3/§4).
- **Wikipedia SQL dump graph tables** — `page`, `linktarget`, `pagelinks`,
  `redirect` ≈ 11.0 GB (no ID) — the authoritative graph, kept only as a
  completeness check; `categorylinks.sql.gz` 2.51 GB named separately
  (source: wiki-qa-sharding.md §4).
- **HotpotQA Wikipedia corpus with links** (no ID; HF `ParthMandaliya/hotpotqa-wiki`)
  — Oct-2017 snapshot, 5.49M articles, explicit `links` column, clean text, 26.3 GB
  Parquet; a ready-made alternative, flagged as old
  (source: wiki-qa-sharding.md §4).
- **USearchWiki** (no ID; HF `unum-cloud/USearchWiki`) — Aug 2025, multilingual,
  text + graph metadata + embeddings, 943 GB; flagged "overkill"
  (source: wiki-qa-sharding.md §4).
- **SNAP enwiki-2013 edge list** (no ID) — flagged very old
  (source: wiki-qa-sharding.md §4).
- **WikiLinkGraphs** (arXiv 1902.04298) — annual snapshots to 2018, nine languages;
  the one ready-made link-graph artifact on record with an arXiv ID
  (source: wiki-qa-sharding.md §4).
- **Kiwix ZIM builds** (no ID) — `mini` 12 GB, `nopic` 49 GB, `maxi` 115 GB, plus
  topical/popularity subsets (top-selection intro-only 316 MB, top no-images 2.1 GB,
  top-1M no-images 16 GB, physics 304 MB); rendered HTML, useful as bounded
  collections, "awkward as canonical QA-ingestion input"
  (source: wiki-qa-sharding.md §4).
- **`pages-articles.xml.bz2` (25.55 GB) / multistream + index (26.67 GB + 284 MB) /
  `pages-meta-current` (46.31 GB)** (no IDs) — the wikitext route, with the recorded
  caveat that a nominally 26 GB download can need well over 100 GB of working space
  (source: wiki-qa-sharding.md §4).
- **"MediaWiki Content Current" export** (no ID) — 45.68 GB across 19 page-ID-range
  shards of ~1.2–2.8 GB; the concrete example of official sharding being contiguous
  page-ID ranges, "not random, topical, or graph-coherent subsets" — the negative
  result motivating the project's own partitioning (source: wiki-qa-sharding.md §4).
- **`all-titles-in-ns0.gz` (109 MB), `page.sql.gz` (2.41 GB), `redirect.sql.gz`
  (185 MB), `page_props.sql.gz` (458 MB)** (no IDs) — the metadata tier of the
  budget ladder (source: wiki-qa-sharding.md §4).
- **Wikidata truthy RDF (43.25 GB) / full JSON (102.67 GB)** (no ID) — the KB tier;
  Wikidata relations, redirects, and disambiguation are listed as affinity signals
  (source: wiki-qa-sharding.md §4).
- **Deterministic hash sampling for a representative subset** (no ID; method) —
  `retain if hash(snapshot_date, page_id) mod 100 < 10` for stable nested
  1/5/10/20% samples, since no official representative random subset exists
  (source: wiki-qa-sharding.md §4).

**Retrieval / storage engines and index methods (the serving flank, SHARD-3)**

- **Pyserini** (no ID; Lucene BM25 + Faiss, client-side hybrid) — "has not really
  been displaced as a research baseline"; the reference BM25 in the build sequence,
  and the record's rule that "'uses BM25' does not guarantee baseline equivalence"
  (source: wiki-qa-sharding.md §4; docs/topics/reference/retrieval-storage-tooling.md).
- **LanceDB** (no ID) — the chosen single-machine stack: native BM25 FTS, disk-
  oriented `IVF_HNSW_FLAT/SQ/PQ`, `IVF_PQ`, `IVF_RQ` (RaBitQ-style), exhaustive
  search for ground truth, RRF hybrid, reranker adapters, true multivector MaxSim,
  scalar indexes for `graph_partition`; demonstrated FTS over 41M Wikipedia docs;
  OSS is single-process (~10–50 QPS per the vendor) with no graph engine
  (source: wiki-qa-sharding.md §4; docs/topics/reference/retrieval-storage-tooling.md).
- **Qdrant** (no ID) — the serving-first alternative: named dense + BM25-sparse
  (+ ColBERT) vectors on one point, IDF modifier, top-level RRF so fusion merges
  globally rather than per shard, `query_points_groups` by `article_id`, payload
  indexes created *before* bulk ingestion, custom shard keys suited only to few
  low-cardinality partitions — hence the recorded rule to use `graph_community` as
  **metadata, not the physical shard key**, initially, because "querying the wrong
  subset causes catastrophic recall loss"
  (source: wiki-qa-sharding.md §4).
- **Vespa** (no ID) — field-aware BM25, sparse WAND, filtered HNSW, multi-stage
  phased ranking incl. ONNX cross-encoder over global top results; "most capable if
  you want to experiment with retrieval architecture"; the migration target if
  concurrency, ranking expressiveness, or HA demand it
  (source: wiki-qa-sharding.md §4).
- **OpenSearch** (no ID) — Lucene BM25, Lucene/Faiss HNSW and IVF, hybrid pipelines
  with normalization/fusion, on-disk vector search and quantization; "safer
  organizational choice"; also the answer if exact **global BM25 across shards**
  matters, since RRF merges rankings but does not reproduce global BM25 — a direct
  constraint on manual sharding (source: wiki-qa-sharding.md §4;
  docs/topics/reference/retrieval-storage-tooling.md).
- **Milvus** (no ID) — native BM25 since 2.5, weighted/RRF hybrid
  (source: wiki-qa-sharding.md §4).
- **ColBERT / PLAID** (no ID) — late-interaction multivector reranking, usually
  after first-stage retrieval; available natively in LanceDB (MaxSim) and Qdrant
  (third named vector with HNSW `m=0`) (source: wiki-qa-sharding.md §4).
- **DuckDB** (no ID) — the analytical workbench: nodes/edges construction, degree
  distributions, PageRank inputs, community statistics, train/dev/test slices,
  duplicate/leakage detection, evaluation as a SQL join; FTS extension is "a
  credible lightweight BM25 implementation"; **DuckDB VSS** is experimental (whole
  HNSW must fit in RAM, not governed by `memory_limit`, `FLOAT` only, incomplete WAL
  recovery) and must not be the sole Wikipedia-scale vector index
  (source: docs/topics/reference/retrieval-storage-tooling.md; wiki-qa-sharding.md §4).
- **Lance format + DuckDB's Lance extension** (no ID) — the recorded design where
  DuckDB and LanceDB become "two execution interfaces over the same Lance data"
  (source: docs/topics/reference/retrieval-storage-tooling.md).
- **Reciprocal Rank Fusion (RRF)** (no ID) — the default fusion rule (500–1,000
  candidates each), with the note that raw BM25 and cosine are not on the same scale
  and weighted fusion should be tuned once held-out QA data exists
  (source: wiki-qa-sharding.md §4).
- **Article-level grouping / diversification before the reader** (no ID; design
  rule) — "otherwise the top 20 results can become 15 nearly identical passages from
  one long article"; the multi-answer requirement that ties SHARD-3 to SHARD-opt-3
  (source: wiki-qa-sharding.md §4).
- **Quantization arithmetic on record** (no ID) — 40M × 768-d ≈ 123 GB float32 /
  61 GB float16 / 31 GB int8 before HNSW overhead; measure ANN recall against exact
  Faiss on a sample (source: wiki-qa-sharding.md §4).

**Entity-index substrate (SHARD-opt-1 routing signal)**

- **Entity–page retrieval reframed as sparse retrieval** (no ID) — A_{e,d} = w(e,d),
  question as a sparse entity vector, s(d) = Σ_e q_e A_{e,d} × IDF(e); "not
  inherently a graph-database query … a sparse dot product, implemented physically
  as a set of inverted posting lists"
  (source: docs/topics/reference/retrieval-storage-tooling.md).
- **Qdrant sparse vector named `entities`** (no ID) — edge-quality weights (Wikidata
  subject 1.0, infobox 0.9, lead link 0.8, body link 0.5, unlinked NER mention 0.1,
  navigation/template ~0), IDF modifier, cold/disk tier — the named implementation
  for SHARD-opt-1 (source: docs/topics/reference/retrieval-storage-tooling.md;
  wiki-qa-sharding.md §4).
- **Lucene / Tantivy entity-ID-as-term** (no ID) — each entity ID as an exact
  pretokenized term in an `entity_field`, query as weighted disjunction; less
  convenient for arbitrary float edge weights
  (source: docs/topics/reference/retrieval-storage-tooling.md).
- **CSR + memory mapping** (no ID) — `entity_offsets: uint64[E+1]`, `page_ids:
  uint32[M]`, `edge_weights: float16|uint8[M]`, `document_freqs: uint32[E]`, with a
  transpose for analysis; the mmap CSR neighbor-expansion file in SHARD-opt-1
  (source: docs/topics/reference/retrieval-storage-tooling.md).
- **LMDB** (no ID) — memory-mapped, zero-copy, larger-than-RAM, cheap concurrent
  readers, no compaction; "the closest off-the-shelf equivalent" to the custom HDF5
  store Danielle originally built
  (source: docs/topics/reference/retrieval-storage-tooling.md).
- **RocksDB** (no ID) — LSM, block cache, compression, Bloom filters, batched
  `MultiGet`; the update-frequent alternative
  (source: docs/topics/reference/retrieval-storage-tooling.md).
- **HDF5 / h5py** (no ID) — Danielle's original implementation; recorded as
  "defensible but not the best default", awkward for per-entity variable-length
  records; if used, mirror CSR with four or five large contiguous numeric datasets
  (source: docs/topics/reference/retrieval-storage-tooling.md).
- **WAND / block-max WAND** (no ID) — threshold/top-k algorithms, "one of the
  reasons a search engine can outperform an ordinary database join even when both
  ultimately use inverted data" — the answer to the head-entity join blow-up
  (source: docs/topics/reference/retrieval-storage-tooling.md).
- **Roaring bitmaps** (no ID) — exact set operations over the highest-degree
  postings (source: docs/topics/reference/retrieval-storage-tooling.md;
  wiki-qa-sharding.md §4).
- **Head-list truncation, rare-first query planning, IDF weighting, edge-quality
  weights** (no IDs; the five head-entity strategies) — the recorded mitigations for
  candidate explosion on frequent entities
  (source: docs/topics/reference/retrieval-storage-tooling.md).
- **LanceDB `LABEL_LIST` index (`array_has_any/all`)** (no ID) — positioned for
  lower-cardinality labels; explicitly *not* a fit for millions of weighted entity
  values (source: docs/topics/reference/retrieval-storage-tooling.md).
- **NetworkX / igraph** (no IDs) — where graph algorithms live given LanceDB has no
  graph engine; "compute graph partitions and graph features offline, then store the
  results as indexed columns" (source: wiki-qa-sharding.md §4).
- **Personalized PageRank / community lookup / score propagation** (no IDs; methods)
  — named as the graph operations that must live in application code
  (source: wiki-qa-sharding.md §4).

**Entity linking behind the mention layer (feeds SHARD-opt-1's `entities` vector)**

- **ReLiK** (no ID; Findings ACL 2024, SapienzaNLP) — retriever–reader linker, one
  pass per chunk, claimed up to 40× faster; the lead candidate for producing the
  mention table (source: docs/topics/reference/entity-linking-at-scale.md).
- **ReFinED** (no ID; NAACL Industry 2022, Amazon) — all mentions in one forward
  pass, >60× faster than contemporaries, Wikipedia or >30M Wikidata entities
  (source: docs/topics/reference/entity-linking-at-scale.md).
- **BELA** (no ID; Meta, 97 languages) — passage-level encoding, kNN against an
  entity index, NIL prediction (source: docs/topics/reference/entity-linking-at-scale.md).
- **`entity-linkings`** (no ID; EACL 2026 demo, NAIST) — unified framework with
  interchangeable retrievers/rerankers for a controlled bake-off
  (source: docs/topics/reference/entity-linking-at-scale.md).
- **The four-tier cascade and three-source mention union** (no ID; protocol) —
  deterministic alias priors → passage linker in GPU batches → ambiguous-case
  resolver over ~0.1–1% of cases → document consistency pass; the cost structure
  that makes exhaustive linking over Wikipedia affordable
  (source: docs/topics/reference/entity-linking-at-scale.md).
- **Page-level aggregate table** (no ID; schema) — page_id, entity_id,
  mention_count, max/mean_confidence, first_position, linked_in_title,
  linked_in_lead, distinct_surface_forms, with edge weight
  w(e,d) = log(1 + mention count) × confidence × positional salience — the explicit
  input to the `entities` sparse vector
  (source: docs/topics/reference/entity-linking-at-scale.md; wiki-qa-sharding.md §4).

**Evaluation design on record (SHARD-3's metric table)**

- **Central objective** (no ID) — "expected number of physical shards required to
  recover a high-recall evidence set, subject to storage and QPS balance"
  (source: wiki-qa-sharding.md §1/§4).
- **Metric list** (no ID) — % gold evidence sets within one / two shards; mean and
  p95 shards touched; router recall vs. shards searched; Recall@k; answer and
  supporting-fact F1; p50/p95 end-to-end latency; storage/QPS/compute imbalance;
  replication factor and update amplification (source: wiki-qa-sharding.md §1/§4).
- **Train/held-out protocol** (no ID) — partition on training queries, evaluate on
  held-out questions; the guard against fitting the partition to the eval workload
  (source: wiki-qa-sharding.md §1/§4).
- **Baseline conditions** (no IDs) — hash/random; title/category; hyperlink-only
  balanced k-way; semantic-kNN; hybrid; workload hypergraph; workload hypergraph +
  bounded boundary replication (source: wiki-qa-sharding.md §1/§4).

**Repo-internal cross-references (where more material sits)**

- `docs/potential-projs/wiki-qa-sharding.md` §4, "2026-08-16 — Turn wiki corpus into
  shards" — the only literature-bearing record; its QA paragraph is explicitly
  headed "unverified attributions".
- `docs/topics/reference/retrieval-storage-tooling.md` — the tooling flank (DuckDB
  vs. LanceDB, entity–page sparse reframing, CSR/mmap, LMDB/RocksDB, head-entity
  strategies); product claims unverified.
- `docs/topics/reference/entity-linking-at-scale.md` — the linking stack behind
  SHARD-opt-1.
- `docs/topics/reference/multi-answer-qa-literature.md` and
  `docs/refs/multi-answer-qa-state-of-research-2026.md` — the workload side for
  SHARD-opt-3; agent-produced deep-search report, unverified.
- `docs/potential-projs/maqa-brute-force-baseline.md` — the sibling project whose
  §3 defers to this doc's engine-choice rationale, and whose GraphRAG record
  overlaps this one's graph flank.
- `docs/danielle-inputs.md` ("MAQA Next Steps" intake, 2026-08-22, lines ~1229–1345
  and the 2026-08-22 post-intake decisions) — Danielle's verbatim prompts and the
  decision splitting `SHARD` out as a post-PhD / "engineering break" project.
- `docs/litreview/citation-verification-ledger.md` — **no SHARD rows**; nothing
  above is verified.
- `docs/potential-projs/README.md` and `docs/topics/README.md` §"Multi-answer QA and
  retrieval (MAQA / SHARD)" — program placement (non-pillar; alternative topic
  proposal deferred 2026-08-22).

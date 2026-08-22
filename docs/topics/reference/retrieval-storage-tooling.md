# Retrieval and analytics storage tooling — DuckDB, LanceDB, Qdrant, Pyserini

**Kind:** reference — a standing accumulator for tooling comparisons around corpus storage,
analytical scans, BM25, vector/ANN indexes, and hybrid retrieval. Project-specific
application lives in `../staging/wiki-qa-sharding.md`.

Source: excerpts from the Notion page "MAQA Next Steps" (conversation dated 2026-08-16; intake
2026-08-22). Product claims are the respondent's and **unverified**; feature sets move fast.
---

## 2026-08-16 — DuckDB vs. LanceDB

**Danielle's question.** How does DuckDB compare to LanceDB — for the Wikipedia use case,
but "more generally is the interesting part of the question."

**Framing (near-verbatim).** "DuckDB and LanceDB overlap, but they are optimized around
different query shapes. DuckDB starts from 'scan, join, aggregate, transform.' LanceDB starts
from 'given this query, return the nearest or most relevant rows.'" DuckDB's Lance extension
can now read/write Lance datasets and run Lance vector, FTS, and hybrid search — "DuckDB and
LanceDB no longer necessarily imply separate storage layers. They can be two execution
interfaces over the same Lance data."

**General comparison (condensed).**
- *Identity:* embedded analytical SQL database vs. embedded retrieval database + AI-oriented
  table format.
- *Best query shape:* many-row scans/joins/aggregates/windows vs. top-k relevant rows.
- *Storage:* native DuckDB file or direct Parquet queries vs. Lance columnar datasets with
  retrieval indexes.
- *SQL:* broad relational SQL vs. useful but narrower SQL/filtering.
- *Vector search:* exact SQL scans + experimental HNSW (VSS) vs. first-class exact and
  approximate retrieval with IVF / IVF-HNSW and flat/PQ/SQ quantization.
- *BM25:* DuckDB FTS extension (Okapi BM25; stemmer, stopwords, normalization, k₁/b
  configurable) vs. native Lance FTS (configurable tokenization, stemming, stopwords, phrase
  positions, fuzzy, retrieval-time filtering) integrated into hybrid search.
- *Hybrid:* manual SQL combination or via the Lance extension vs. built-in dense + FTS
  hybrid and reranking.
- *Filtering:* analytical predicates vs. retrieval-aware pre/post-filtering with scalar
  indexes.
- *Joins / feature engineering:* DuckDB's major strength; not LanceDB's centre.
- *Versioning:* transactions (external formats govern dataset versions) vs. dataset
  versions, time travel, tags, cleanup.
- *Serving:* embedded, mostly single-process writes vs. embedded OSS with distributed
  Enterprise options.
- *Operational bias:* batch analytics and research vs. repeated low-latency retrieval.
- DuckDB handles larger-than-memory analytics by spilling to disk — good for corpus
  construction and evaluation on a storage-constrained machine.

**The key vector-search distinction.** DuckDB VSS: still experimental; persistent HNSW
behind an experimental flag; WAL recovery for persistent custom indexes incomplete; the
whole HNSW index must fit in RAM and is not governed by `memory_limit`; checkpointing
serializes the whole index; `FLOAT` only; DuckDB's docs advise against persistent VSS
indexes in production. Rule of thumb: ~1M moderate embeddings on a laptop → DuckDB VSS may
be fine; tens of millions of passages with repeated queries → LanceDB (disk-oriented IVF /
IVF-HNSW, PQ/SQ, exact scans for ANN-recall measurement); production online service with
replicas → Qdrant or another server-oriented system.

**BM25.** Both real. DuckDB FTS is "a credible lightweight BM25 implementation" for a static
corpus; LanceDB wins when BM25 is routinely combined with dense retrieval, metadata
prefiltering before top-k is needed, indexes should be colocated with embeddings,
indexed-vs-unindexed-row accounting matters, or one API for FTS/ANN/hybrid is wanted. For
comparability with classic open-domain QA baselines, validate either against
Lucene/Pyserini — "'uses BM25' does not guarantee baseline equivalence."

**DuckDB-alone pattern.** Reasonable when offline, immutable corpus, BM25-primary, infrequent
dense queries, HNSW fits in memory, workload is join/analysis-heavy. "BM25 first, dense
rerank" without a global ANN index: FTS top-1000 lexical candidates → exact cosine over only
those → top-100. Works when BM25 candidate recall is high; fails on passages only dense
retrieval would find.

**Combined design.** Canonical working corpus as a Lance dataset; DuckDB + Lance extension
for joins/scans/statistics/evaluation/transforms; LanceDB for BM25/ANN/hybrid/filtered
top-k; small Parquet tables retained for the link graph and evaluations. Evaluation becomes
a join: `SELECT method, k, avg(found_gold) FROM retrieval_results JOIN gold_evidence USING
(question_id, passage_id) GROUP BY method, k`.

**Recommendation (quoted).** "DuckDB is probably the best workbench for the project; LanceDB
is the more appropriate retrieval engine. The combination is more compelling than choosing
one exclusively." Don't make DuckDB VSS the sole Wikipedia-scale vector index; move to
Qdrant when concurrent serving, replication, and distributed operation dominate.

See also the engine survey (Pyserini, LanceDB, Vespa, OpenSearch, Qdrant, Milvus,
ColBERT/PLAID) and the LanceDB and Qdrant deep-dives recorded in
`../staging/wiki-qa-sharding.md`.

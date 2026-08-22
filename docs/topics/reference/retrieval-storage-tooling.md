# Retrieval and analytics storage tooling — DuckDB, LanceDB, Qdrant, Pyserini

**Kind:** reference — a standing accumulator for tooling comparisons around corpus storage,
analytical scans, BM25, vector/ANN indexes, and hybrid retrieval. Project-specific
application lives in `../../potential-projs/wiki-qa-sharding.md`.

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
`../../potential-projs/wiki-qa-sharding.md`.

## 2026-08-16 — Entity–page candidate fetch at scale: beyond a hand-built HDF5 index

**Danielle's context.** An earlier research project extracted entity IDs from every Wikipedia
page, built a graph over entity IDs (not links), and fetched candidates from question
entities — "a wildly heuristic approach." Frequent entities made relational joins infeasible,
so she implemented her own indices with typed integer IDs and concluded that an HDF5/h5py
store with smart batched disk reads was the way to make queries feasible. Question: is there
a better way around "everything doesn't fit in memory, something must be streamed from disk,
and I can't assume sharding will work" than a custom HDF5 structure?

**Response (near-verbatim, condensed).** "HDF5 is a defensible implementation, but it is not
the best default. What you independently arrived at is essentially the physical design of
an inverted index or compressed sparse matrix" — compact integer IDs, contiguous postings
per entity, in-memory offsets, batched disk reads, no giant joins. "The more fundamental
complication is that faster disk access does not eliminate the candidate explosion caused
by very frequent entities. That requires scoring and query-planning changes as well as
better storage."

*Reframe as sparse retrieval.* Entity–page matrix A_{e,d} = w(e,d); question = sparse entity
vector q; s(d) = Σ_e q_e A_{e,d} (× IDF(e)). "This is not inherently a graph-database query.
It is a sparse dot product, implemented physically as a set of inverted posting lists."

*Best modern fit — Qdrant sparse vector.* A named sparse vector `entities` per page/passage
whose dimensions are entity IDs and whose values are edge weights (subject 1.0, lead link
0.6, body mention 0.2…); the question gets the same representation weighted by linker
confidence; `query_points(using="entities")`. Sparse index is exact, inverted-index-like,
can sit in a cold/disk tier, IDF modifier supplies document frequencies. "You never ask a
relational database to join a question's entities against a giant entity–page table. Qdrant
walks the relevant postings and maintains a top-k score accumulator." Fuse entity-sparse +
text BM25 + dense — "likely stronger than using the entity graph as the sole candidate
generator."

*Lucene or Tantivy.* Treat each entity ID as an exact pretokenized term in an `entity_field`;
query = weighted disjunction. Gets compressed postings, skip structures, DF statistics,
top-k scoring, segment storage, page-cache integration, early skipping, and textual BM25 in
the same index. Lucene is less convenient for arbitrary float edge weights; Qdrant's
explicit sparse vectors handle weighted edges more naturally.

*Direct control — CSR + memory mapping.* `entity_offsets: uint64[E+1]`, `page_ids:
uint32[M]`, `edge_weights: float16|uint8[M]`, `document_freqs: uint32[E]`; size ≈ 8(E+1) +
4M + sizeof(w)·M; `numpy.memmap` / Rust mmap; build the transpose (`page_offsets`,
`entity_ids`) for analysis/expansion/rebuilds. Batched queries: resolve offsets → order
ranges by file offset → read nearby postings together → accumulate in a reusable
integer-key hash table → keep top-k → let the page cache retain hot postings. "Probably
faster and more predictable than variable-length HDF5 objects."

*Where HDF5 becomes awkward.* One dataset per entity (metadata overhead), variable-length
records (indirection), chunks misaligned with posting boundaries, whole-chunk reads for
small slices, per-call overhead for many tiny accesses. If using HDF5: four or five large
contiguous numeric datasets mirroring CSR; no per-entity objects, object dtypes, pickle, or
vlen arrays — at which point a raw mmap file is simpler.

*Key-value middle ground.* LMDB (static/read-dominated: memory-mapped, zero-copy,
larger-than-RAM, cheap concurrent readers, no compaction — "the closest off-the-shelf
equivalent to what you were trying to obtain with HDF5"); RocksDB (frequent updates: LSM,
block cache, compression, Bloom filters, batched `MultiGet` with parallel block reads).
Neither supplies top-k sparse scoring — you still implement union/intersection/accumulation.

*The head-entity problem remains.* No representation makes "return five million pages and
join" cheap; avoid fully decoding or scoring that list: (1) IDF weighting; (2) edge-quality
weights (Wikidata subject 1.0, infobox 0.9, lead link 0.8, body link 0.5, unlinked NER
mention 0.1, navigation/template ~0); (3) head-list truncation — top-L (e.g. 10,000)
salient pages per common entity hot, full tail cold; (4) rare-first query planning (the
sparse analogue of choosing the most selective predicate); (5) threshold / top-k algorithms
(WAND, block-max WAND) — "one of the reasons a search engine can outperform an ordinary
database join even when both ultimately use inverted data"; (6) don't rely on entities alone
— a frequent entity is a small boost, the actual evidence comes from a rare entity, BM25, or
dense match.

*Entity graph vs. entity–page index.* Keep two structures: entity→page sparse index and
entity→entity adjacency (second CSR: `neighbor_offsets`, `neighbor_ids`,
`neighbor_weights`). At question time: link entities → 20–100 strongest neighbors each,
downweighted → expanded sparse question vector → entity-to-page retrieval. Never expand a
high-degree entity's full neighborhood; prune by conditional probability / PMI / normalized
co-occurrence / learned relevance with a fixed per-entity expansion budget. Roaring bitmaps
for exact set operations over the highest-degree postings.

*Where DuckDB and LanceDB fit.* DuckDB builds the index (dense IDs, DFs, edge weights, sorted
(entity_id, page_id), offsets, degree distributions, truncation/weighting policy
evaluation) but is not where online entity queries run. LanceDB's `LABEL_LIST` index
(`array_has_any/all`) is positioned for lower-cardinality labels; millions of entity values
with weighted top-k are a better fit for Qdrant sparse vectors or a real inverted index.

**Bottom line (quoted).** "Your earlier conclusion was directionally right: compact IDs,
contiguous arrays and batched disk reads are indeed the solution. The part I would change
is implementing candidate retrieval yourself on top of HDF5. A sparse-vector engine or
text-search inverted index already embodies that data structure and, more importantly,
supplies the top-k algorithms needed to avoid fully processing the huge posting lists that
caused the original problem."

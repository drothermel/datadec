# Wikipedia QA sharding — co-locating what is retrieved together

> **Draft scaffolding (2026-08-22).** Promoted from the staging topic `wiki-qa-sharding`.
> §1–§3 are synthesized from the 2026-08-16 discussion; §4 is the dated record. Treat §1–§3
> as provisional until this note is removed. Flagged by Danielle as a post-PhD or
> "engineering break" paper rather than part of the current arc.

**Program pillars served:** none — outside the DataDecide program; systems/retrieval.
(Program: `README.md` → Program.)

**One-line pitch.** Physically shard a full Wikipedia snapshot so that the evidence sets QA
queries actually touch co-locate — workload-aware hypergraph partitioning over a
hyperlink + semantic + co-retrieval affinity graph, with a compact global router and bounded
boundary replication — and evaluate end-to-end: shards touched per query, router recall,
retrieval recall, answer F1, latency, and balance, against hash/category/hyperlink-only
baselines.

IDs: SHARD-1–SHARD-3, SHARD-opt-1–SHARD-opt-3.

**Paper goal.** A systems/IR paper (workshop or a data-systems venue): "shard all of
Wikipedia by observed QA evidence locality and measure end-to-end QA latency and recall" —
claimed unsaturated (unverified). Doable solo as an engineering-focused project.

Compute tiers: **CPU** = partitioning, indexing, evaluation on one large machine; **GPU-light**
= embeddings for the semantic kNN graph and dense index.

---

## 1. What the project involves

### Core experiment (SHARD-1–SHARD-3)

**SHARD-1 — Data and graph.** One dated English Wikipedia snapshot (pinned directory,
checksums). Derive `nodes.parquet` / `edges.parquet` from the Structured Wikipedia Parquet
dataset (flatten recursive `links`, keep internal links, strip fragments, normalize, join to
page IDs, resolve redirects; retain anchor text and section path), with the SQL
`pagelinks` tables as a completeness check. Article-level vertices sized by storage
footprint; passages chunked inside shards. Build affinity layers: hyperlinks (lead/body vs.
navigation), embedding-kNN edges, QA supporting-document sets (HotpotQA, 2WikiMultiHopQA,
MuSiQue, QAMPARI evidence) as hyperedges, optional clickstream; hub downweighting.

**SHARD-2 — Partitioning conditions.** Same snapshot: hash/random; title/category;
hyperlink-only balanced k-way (METIS/KaMinPar); semantic-kNN partition; hybrid; workload
hypergraph (Mt-KaHyPar, connectivity objective Σ f_q(λ_q − 1)); workload hypergraph +
bounded boundary replication. Partition on training queries, evaluate on held-out.

**SHARD-3 — Serving and evaluation.** A global routing layer (titles/aliases, shard
centroids, small lexical index, entity→shard map) over shard-local stores (text,
lexical/vector indexes, local adjacency) with boundary support. Single-machine stack:
canonical Lance/Parquet corpus; DuckDB for construction and evaluation; LanceDB (or
Qdrant if serving) with `graph_partition` as an indexed filter column. Metrics: % gold
evidence sets within one / two shards; mean and p95 shards touched; router recall vs.
shards searched; Recall@k; answer and supporting-fact F1; p50/p95 latency; storage/QPS
imbalance; replication factor. Central objective: expected number of shards required to
recover a high-recall evidence set under balance constraints.

### Optional directions

- **SHARD-opt-1 — Entity-sparse candidate fetch as a routing signal** (Qdrant `entities`
  sparse vector; head-entity strategies; mmap CSR neighbor expansion).
- **SHARD-opt-2 — Streaming/update partitioning** for monthly snapshots.
- **SHARD-opt-3 — Multi-answer workload** — evaluate whether list-QA evidence (many
  articles per question) is sharding-friendly at all, connecting to `maqa-brute-force-baseline.md`.

## 2. Doability and impact

### Overall doability: **high** (engineering-bound, one large machine), impact **medium**

All inputs are public; the hard parts are data plumbing and a careful evaluation design. The
headline risk is the "placement only pays if first-stage retrieval is routable" point: if
queries fan out to all shards, placement helps fetch and expansion only — the router is the
load-bearing component and must be evaluated with the partition. Claimed literature gap is
unverified.

### Per-direction impact

- **SHARD-1–3.** A clean systems result with a reusable sharded-Wikipedia artifact.
- **SHARD-opt-3.** The interesting cross-over with MAQA if multi-answer evidence turns out to
  cluster.

## 3. Infrastructure build sequence

1. Snapshot pull (Structured Wikipedia Parquet + SQL graph tables ≈ 11 GB) with a dated
   directory and checksums.
2. DuckDB pipeline: nodes/edges, degree statistics, hub weights, QA hyperedges from
   benchmark supporting sets.
3. Partitioners: KaMinPar / Mt-KaHyPar wrappers; condition registry.
4. Section-level passage corpus with `graph_partition`; Pyserini BM25 reference; LanceDB
   hybrid index (migrate to Qdrant/Vespa only if serving demands it).
5. Router + evaluation harness producing the metric table above, with cost per condition.

---

## 4. External assessments and origin notes

Dated notes from the external conversation this doc was promoted from, recorded for
consolidation — not decisions. Tool and literature claims are unverified. Tooling
comparisons live in `../topics/reference/retrieval-storage-tooling.md`; entity linking in
`../topics/reference/entity-linking-at-scale.md`.

### Origin notes — moved from `topics/staging/wiki-qa-sharding.md`

### 2026-08-16 — Turn wiki corpus into shards

**Danielle's question.** How to store a full Wikipedia corpus sharded such that "the shards
co-locate things that are likely to be accessed together when doing question-answer tasks."
First idea: build the page-link graph and apply a graph algorithm (min-cut-like) that "parses
the graph into subsets with the goal of cutting the fewest edges." Wants both the
foundational graph-algorithm options and how this has been applied to Wikipedia or other
large corpora for QA in the literature.

**Response (near-verbatim, condensed).**

*Formulation.* Not unconstrained min-cut (whose trivial optimum is one giant shard plus
leftovers) but **weighted, balanced k-way graph partitioning**: vertices = articles with
storage size s_v and load q_v; edge weights w_uv = value of co-placement; minimize cut weight
subject to per-shard capacity (1+ε)·Σs_v/k (and optionally a QPS constraint). Two shards =
minimum bisection. NP-hard; practical systems use multilevel heuristics (coarsen → partition →
uncoarsen/refine): METIS, KaHIP, KaMinPar. For QA, go one step further: **workload-aware
hypergraph partitioning with limited replication** — a query's evidence set H_q is one
hyperedge; with λ_q = shards touched, minimize Σ_q f_q(λ_q − 1) (connectivity objective;
tools: Mt-KaHyPar).

*Algorithm families.* Balanced graph partitioning (strong baseline; pairwise only);
spectral/normalized cut (math insight; expensive at scale); Leiden/Louvain (topical
communities, but wildly uneven sizes — need splitting/packing to meet capacity; GraphRAG uses
hierarchical Leiden for navigation, not physical sharding); hypergraph partitioning (best
match; needs access traces or evidence sets); vertex cuts / replication (for Wikipedia's
hubs; adds storage/update cost — cf. PowerGraph); streaming partitioning (for updates; lower
quality).

*Progression.* (1) weighted hyperlink graph; (2) add semantic-neighbor edges from an
embedding kNN graph; (3) add real co-retrieval / QA evidence sets as hyperedges; (4) weight
workload evidence up as representative traffic accumulates.

*Affinity signals.* Article hyperlinks (lead/body vs. template/navigation); semantic
nearest neighbors; QA supporting-document sets; co-retrieval traces; Wikimedia clickstream
(weaker proxy — browsing locality ≠ QA locality); Wikidata relations, redirects,
disambiguation; article structure (passages of one article: very high affinity). Downweight
hubs (degree normalization / IDF-like); let co-retrieval and evidence-path counts dominate
generic links once traffic exists.

*Granularity.* Partition at **article** level, chunk into passages inside shards (stable
identities, cheap article-local expansion, less fragmentation, easier updates); split only
exceptionally large articles into section bundles with high-weight cohesion edges.
Passage-level partitioning gives marginally better cuts but makes single-article questions
touch multiple shards.

*Retrieval architecture matters as much as placement.* Co-location only pays if queries
route to few shards; if first-stage search fans out everywhere, placement helps document
fetch and multi-hop expansion only. Design: a compact global routing layer (titles/aliases,
shard centroids, small lexical index, entity→shard map); shard-local storage (text,
lexical/vector indexes, local adjacency, entity metadata); boundary support (replicated
boundary articles, global title/redirect table, remote-edge pointers, overflow path).
Related: graph partitioning for large-scale ANN with an explicit router (PVLDB, "Unleashing
Graph Partitioning for Large-Scale Nearest Neighbor Search"). Replication: compact hub
metadata everywhere; full content only for hot boundary pages; budget by measured reduction
in shards touched.

*What the QA literature has done (unverified attributions).* Graph structure used for
retrieval/reasoning, rarely for physical placement: Learning to Retrieve Reasoning Paths over
the Wikipedia Graph (Asai et al., ICLR 2020); multi-step entity-centric retrieval (Das et
al., 2019); CogQA (Ding et al., 2019); Multi-hop Dense Retrieval (Xiong et al., ICLR);
HippoRAG (entity/passage graphs, personalized PageRank). Datasets with supporting-document
sets usable as hyperedges: HotpotQA (~113k), WikiHop, 2WikiMultiHopQA (explicit reasoning
paths), MuSiQue (25k, 2–4 hops). Workload-aware database placement with partitioning,
balancing, and replication exists (Golab et al.) but not this Wikipedia-QA combination.
Claimed gap: "physically shard all of Wikipedia by observed QA evidence locality and
evaluate end-to-end QA latency is not a saturated research problem."

*Proposed experiment.* Same snapshot; conditions: hash/random; title/category; hyperlink-only
balanced METIS/KaMinPar; semantic kNN partition; hybrid; workload hypergraph; workload
hypergraph + bounded boundary replication. Partition on training queries, evaluate on
held-out questions. Metrics: % gold evidence sets within one shard / two shards; mean and
p95 shards touched per query; router recall vs. shards searched; retrieval Recall@k; answer
and supporting-fact F1; p50/p95 end-to-end latency; storage/QPS/compute imbalance;
replication factor and update amplification. Central objective: "Expected number of physical
shards required to recover a high-recall evidence set, subject to storage and QPS balance."

### 2026-08-16 — Downloading Wikipedia under a storage budget

**Danielle's question.** With limited storage, where to download Wikipedia data, at what
granularity (metadata vs. text), what sizes, and whether one can download a percentage /
some shards or must download everything and split locally. One date only (recent or
historical).

**Response (near-verbatim, condensed; sizes as stated for August 2026 enwiki, unverified).**
Headline: "some official datasets are physically sharded and individual shards can be
downloaded. But those shards are generally contiguous page-ID ranges — not random, topical,
or graph-coherent subsets."

*Main options (compressed sizes).* Titles only: `all-titles-in-ns0.gz` 109 MB (`all-titles.gz`
382 MB, all namespaces). Page metadata: `page.sql.gz` 2.41 GB + `redirect.sql.gz` 185 MB +
`page_props.sql.gz` 458 MB. **Link graph:** `page` + `linktarget` + `pagelinks` + `redirect`
≈ 11.0 GB (no bodies); + `categorylinks.sql.gz` 2.51 GB. Current article wikitext:
`pages-articles.xml.bz2` 25.55 GB; multistream + index 26.67 GB + 284 MB (independently
decompressible streams, pieces ~50–625 MB). All public namespaces: `pages-meta-current`
46.31 GB. New "MediaWiki Content Current" export: 45.68 GB across 19 page-ID-range shards of
~1.2–2.8 GB. Kiwix ZIM (rendered HTML): `mini` 12 GB (intros + infoboxes), `nopic` 49 GB,
`maxi` 115 GB; topical/popularity subsets (top selection intro-only 316 MB, top no-images
2.1 GB, top-1M no-images 16 GB, physics 304 MB, …). **Structured Wikipedia Parquet**
(huggingface `wikimedia/structured-wikipedia`): sharded, streamable; abstracts, descriptions,
QIDs, sections, infoboxes, tables, citations, lists — DuckDB/Polars can scan selected
columns/files remotely. Wikidata: truthy RDF 43.25 GB / full JSON 102.67 GB.

*Wikitext caveat.* `pages-articles` is wikitext (templates, infobox markup, refs, Lua-dependent
rendering), not clean prose; stream the bz2 into an extractor and write compressed
JSONL/Parquet — "a nominally 26 GB download can require well over 100 GB of working space"
if fully decompressed.

*Four meanings of "10%".* (1) 10% of physical files — yes, but page-ID ranges correlate with
creation time, so not representative, and links point outside the subset. (2) Representative
random 10% — no official download; stream once and `retain if hash(snapshot_date, page_id)
mod 100 < 10` for stable, nested 1/5/10/20% samples (possibly filterable remotely over
Parquet). (3) Topical subset — Kiwix publishes them; fine for bounded collections, awkward as
canonical QA-ingestion input. (4) **Graph-coherent subset — not provided; download the ~11 GB
graph tables, pick seeds, build the subset (categories, link neighborhoods, PageRank,
communities, or the QA workload), then fetch chosen articles by ID/title. "That is likely the
appropriate route for the sharding project."**

*Budget ladder.* <1 GB: titles/metadata, Kiwix topical, remote Parquet scans. ~3 GB: Kiwix
top no-images or a few XML shards. ~12–15 GB: Kiwix `mini` or the link graph without text.
~27 GB: `pages-articles` (minimum for complete current wikitext). ~40 GB: `pages-articles` +
core link graph (~36.6 GB). ~50 GB: Kiwix `nopic` or all-namespace XML. ≥100 GB: Kiwix with
images or full Wikidata (plus much more working space).

*Reproducible single date.* Use a dated directory (e.g. `enwiki/20260801/`), not `latest`;
download all tables from the same snapshot; keep checksums and filenames; record whether
raw wikitext, rendered HTML, or an extractor was used. Server holds recent months only
(Feb–Aug 2026 at the time); older snapshots via Internet Archive per Wikimedia's
"Finding older xml dumps" guidance. One date ⇒ no need for the revision-history dump.

### 2026-08-16 — Link graph from Structured Wikipedia

**Danielle's question.** Is there something like the Hugging Face Structured Wikipedia
dataset for the link graph too?

**Response (near-verbatim, condensed; unverified).** The official `wikimedia/structured-wikipedia`
dataset "already contains the hyperlink information, although it does not expose a separate,
flat `edges.parquet` table" — links sit recursively inside sections/paragraphs, lists, infobox
fields, references, and other components, each generally with target `url` and anchor `text`
(schema: enterprise.wikimedia.com data dictionary).

*Recommended derivation.* Two tables: `nodes.parquet` (page_id, title, url, wikidata_qid,
redirect/status) and `edges.parquet` (source_page_id, target_page_id, target_url,
anchor_text, section_path, component_type, occurrence_count). Process: recursively flatten
every `links` field → keep internal `en.wikipedia.org/wiki/...` links → strip fragments
(optionally keep as `target_section`) → normalize encoding/capitalization → join target URLs
to article URLs for IDs → resolve redirects → keep occurrences or aggregate to weighted
edges. Advantage over `pagelinks.sql.gz`: anchor text and approximate location survive, so
lead-paragraph links can be weighted differently from navigation lists, infoboxes,
citations, or "See also" — directly useful for the affinity weighting above.

*Ready-made alternatives.* HotpotQA Wikipedia corpus with links (HF
`ParthMandaliya/hotpotqa-wiki`; Oct-2017 snapshot; 5.49M articles, explicit `links` column,
clean text, 26.3 GB Parquet — old). USearchWiki (HF `unum-cloud/USearchWiki`; Aug 2025,
multilingual, text + graph metadata + embeddings; 943 GB — overkill). SNAP enwiki-2013 edge
list (very old). WikiLinkGraphs (arXiv 1902.04298; annual snapshots to 2018, nine languages).

*If completeness matters most.* The authoritative graph is still the SQL dumps (`page`,
`linktarget`, `pagelinks`, `redirect`; ~11 GB). Recommendation: "extract the graph directly
from Structured Wikipedia. You get a snapshot-aligned corpus and graph, link context for
QA-aware weighting, and avoid maintaining two substantially different ingestion pipelines.
Use the SQL graph only as a completeness check."

### 2026-08-16 — Index stack: BM25 to vectors at Wikipedia scale

**Danielle's question.** Pyserini used to be the best way to implement indices from BM25 to
vector indexes; what are the newer, better options that still work at Wikipedia scale?

**Response (near-verbatim, condensed; tool claims unverified).** "Pyserini has not really
been displaced as a research baseline ... What has improved is the set of unified engines
that can serve BM25, vectors, filtering, hybrid fusion, and reranking from one corpus."
Short recommendation: research baseline — Pyserini; single-machine experimentation —
LanceDB; serious hybrid serving / complex ranking — Vespa; conventional distributed search
ops — OpenSearch; vector-centric with some lexical — Qdrant or Milvus.

*Options.* Pyserini (Lucene BM25 + Faiss; client-side hybrid). LanceDB (native BM25,
disk-oriented vector indexes, built-in hybrid/reranking; demonstrated FTS over 41M Wikipedia
docs; operationally light). Vespa (field-aware BM25, sparse WAND, filtered HNSW, dense,
metadata features, multi-stage phased ranking incl. ONNX cross-encoder over global top
results — "most capable if you want to experiment with retrieval architecture"). OpenSearch
(Lucene BM25, Lucene/Faiss HNSW and IVF, hybrid pipelines with normalization/fusion; recent
on-disk vector search and quantization; "safer organizational choice"). Qdrant (BM25 as
sparse vectors, HNSW + quantization, dense+sparse+multivector). Milvus (native BM25 since 2.5,
weighted/RRF hybrid). ColBERT/PLAID (late-interaction multivectors, usually after first-stage
retrieval).

*Proposed implementation.* One canonical Parquet corpus; indexes are disposable derived
artifacts. Passage schema: passage_id, article_id, title, section_heading, section_path,
text, wikidata_qid, **graph_partition**, outgoing_page_ids, embedding, snapshot_date. Use
sections / paragraph groups rather than fixed overlapping token windows; prepend
`Title > Section > Subsection` when embedding and indexing. Four retrieval profiles over the
same passage IDs: BM25 (title/heading/body fields with boosts); dense (one quantized
embedding per passage); hybrid (500–1,000 candidates each, fused by RRF — raw BM25 + cosine
are not on the same scale; tune a weighted fusion once held-out QA data exists); reranked
hybrid (top 50–200 via cross-encoder or ColBERT-style). For multi-answer QA, group or
diversify results by article before the reader, "otherwise the top 20 results can become 15
nearly identical passages from one long article."

*Scale.* ~7M article records are easy; passages are the problem (tens of millions of
vectors). 40M × 768-d ≈ 123 GB float32 / 61 GB float16 / 31 GB int8 before HNSW overhead.
Quantize; consider 384-d; measure ANN recall against exact Faiss on a sample; don't store
multiple embedding models in the serving index; consider article-level retrieval then
section-level scoring if storage is tight.

*Specific choice.* Structured Wikipedia → section-level Parquet; Pyserini BM25 as the
reference baseline; LanceDB for rapid BM25+dense hybrid experiments; Vespa once
distributed serving, graph-partition filters, richer ranking features, or online reranking
are needed (skip LanceDB and go straight to Vespa if the goal is already a durable
multi-node service).

### 2026-08-16 — LanceDB alone: pros, cons, how far it goes

**Danielle's question.** Pros and cons of building on LanceDB, and how far could it carry
retrieval on its own?

**Response (near-verbatim, condensed; product claims unverified, as of Aug 2026).** "For a
mostly static, single-date Wikipedia snapshot — LanceDB is unusually well matched. You could
use it alone for essentially the entire experimental retrieval stack ... The main boundary is
operational rather than retrieval quality: OSS LanceDB is an embedded, single-process
system, not a distributed search service."

*What it gives.* One canonical store (passage_id, article_id, title, section, text,
outgoing_links, graph_partition, metadata, dense_vector, colbert_vectors) in the columnar
Lance format with projection, random access, versioning — data, embeddings and indexes stay
aligned, no separate Lucene doc IDs / Faiss row numbers / metadata DB; build once, compact,
pin the version, run read-only. Native BM25 FTS (configurable tokenization, English
stemming/stopwords, optional positional phrase index at a storage cost, fuzzy/prefix,
multiple text columns, pre/post-filtering, array fields); suggested lexical field `title +
title + section_heading + text` as cheap title boosting. ANN choices: `IVF_HNSW_FLAT`,
`IVF_HNSW_SQ` (recommended general balance), `IVF_HNSW_PQ`, `IVF_PQ`, `IVF_RQ`
(RaBitQ-style, storage-constrained), plus exhaustive search for ground truth — "measuring
whether poor QA recall comes from the embedding model or the approximate index." Hybrid
search with RRF by default; reranker adapters (local cross-encoders, ColBERT, hosted,
custom); true multivector MaxSim search (a ColBERT-like index, not just top-100 reranking);
scalar indexes for filters such as graph_partition, category, wikidata_type, is_list,
is_disambiguation, page_rank_bucket — filters applied before retrieval by default.

*Where it is weaker.* Lexical ranking less expressive than Lucene/Vespa (BM25 parameter
control, BM25F-style field weighting, analyzers/synonyms, score explanation, ranking
profiles) — keep a small Pyserini comparison and investigate tokenization/field
construction if they diverge. OSS is single-process: no distributed execution, replication,
failover, horizontal read scaling, HA, or automatic maintenance; vendor characterizes OSS as
~10–50 QPS. Maintenance is manual (`optimize()`, compaction) — irrelevant for a frozen
snapshot. No graph engine: neighbor expansion, personalized PageRank, community lookup, score
propagation, and cross-shard routing live in application code or NetworkX/igraph —
"compute graph partitions and graph features offline, then store the results as indexed
columns." Manual sharding becomes application logic (choose partitions, query concurrently,
merge, handle local BM25 statistics — RRF merges rankings but does not reproduce global
BM25; if exact global BM25 across shards matters, OpenSearch/Vespa).

*How far.* Yes to: full ingestion, section/passage storage, BM25 baseline (validated against
Pyserini), dense exact on samples, dense ANN over tens of millions of passages, quantized
indexes, hybrid, cross-encoder reranking, ColBERT/MaxSim, metadata and graph-partition
filtering, offline QA evaluation (excellent), small research search service (good). Not
alone: graph traversal (app code), high-QPS multi-node, HA/replication; complex learned
ranking possible but Vespa stronger. "LanceDB could carry this project through corpus
construction, retrieval research and a credible single-machine demonstrator." Migrate only
when query concurrency exceeds one machine, sophisticated lexical/ranking behavior is
needed, or automatic distributed sharding/HA is required.

### 2026-08-16 — A Qdrant-based solution

**Danielle's question.** What would a solution with Qdrant look like?

**Response (near-verbatim, condensed; product claims unverified).** Qdrant as the *serving*
index: one point per passage (~100–250 words, bounded by section/paragraph), with multiple
named representations on the same point — dense embedding, BM25 sparse vector (IDF
modifier so Qdrant maintains document-frequency statistics), optional ColBERT multivector
for late-interaction reranking — and a payload (passage_id, article_id, title, section, text
or reference, char offsets, wikidata_qid, snapshot, graph_community, page_rank,
is_disambiguation). One versioned collection per snapshot (e.g. `enwiki_2026_08_01_e5_v1`)
with an atomic alias `enwiki_current`; Parquet remains the canonical corpus. Create payload
indexes (article_id, wikidata_qid, snapshot, graph_community, is_disambiguation) *before*
bulk ingestion because they affect the filtered vector index.

*First stage.* Dense and sparse prefetch (~500 each) → RRF as the top-level query (on a
distributed collection fusion must be top-level so it merges globally rather than per
shard) → `query_points_groups` by `article_id` (e.g. 100 articles × best 3 passages) — "you
can retrieve 100 distinct articles while retaining the best two or three passages from
each," which is what multi-answer QA needs. Weighted RRF or learned fusion once a dev set
exists.

*Reranking.* Simpler: top 100–300 fused passages → external cross-encoder (comparable to
IR research systems). Native: third named vector `colbert` with HNSW `m=0` (rerank-only),
pipeline BM25 500 + dense 500 → RRF 200 → MaxSim → top 30; multivectors cost considerably
more storage.

*Link graph.* Compute communities, PageRank, redirects, adjacency outside Qdrant; store
`graph_community`, `page_rank`, maybe a compact neighbor list in payload; use for routing,
filtering, expansion, reranking. Two-stage expansion: unrestricted hybrid → top 10–30
article IDs → one-hop neighbors from an external adjacency table → search/rerank passages in
that article set → merge. **Use `graph_community` as metadata, not the physical shard key,
initially** — "querying the wrong subset causes catastrophic recall loss"; custom shard keys
suit few partitions with nearly every query specifying its partition (Qdrant recommends
low-cardinality shard keys; physical shards carry overhead).

*Storage.* Original dense vectors on disk; HNSW in RAM if affordable; scalar/binary
quantization after measuring recall loss (quantized in memory, originals on disk for
rescoring an oversampled set); passage text in payload only if convenience justifies
duplication; 2–4 shards per machine to start, then benchmark.

*How far.* Qdrant alone: sparse BM25-like, dense ANN, hybrid fusion, metadata/subset filters,
article grouping, ColBERT-style reranking, quantization, disk-backed storage, sharding,
replication, online serving. External: parsing and passage construction, embedding
generation, link-graph construction and algorithms, cross-encoder/LLM reranking, canonical
storage and analytical scans, QA evaluation. "I would favor Qdrant when the endpoint is a
continuously available, horizontally scalable retrieval service. I would favor LanceDB when
the corpus is mainly an offline research artifact that I want to inspect, transform, and
version like a columnar dataset." First implementation: one collection, one point per
passage, `dense + bm25`, RRF, article grouping, external cross-encoder, communities as
payload only — "leaving each additional component independently measurable."

### 2026-08-16 — Workbench vs. engine: DuckDB + LanceDB over one Lance dataset

General comparison recorded in `../topics/reference/retrieval-storage-tooling.md`. Project
application (near-verbatim, condensed): DuckDB for building, inspecting, partitioning, and
evaluating the corpus — parse/normalize Parquet shards; join passages with metadata,
redirects, Wikidata IDs, categories; construct and inspect the link-edge table; degree
distributions, PageRank inputs, community statistics; materialize train/dev/test slices;
detect duplicate passages and answer leakage; evaluate runs by joining results to gold
evidence (recall@k, answer recall, MRR, per-question slices); compare LanceDB/Qdrant/Pyserini
runs in SQL. LanceDB for the online portion — one row per passage; FTS over title/text; ANN
over the embedding; scalar indexes on article_id, snapshot, graph community; retrieve, fuse,
rerank, fetch scattered rows. Canonical corpus as one Lance dataset read by both via DuckDB's
Lance extension; small Parquet tables for the link graph and evaluations. Don't use DuckDB
VSS as the sole Wikipedia-scale vector index.

### 2026-08-16 — Entity-based candidate fetch: the stack to build now

How the entity mentions themselves get produced (replacing string matching with a
ReLiK/ReFinED cascade plus lexical proposals and document propagation) is recorded in
`../topics/reference/entity-linking-at-scale.md`; its page-level aggregate table is the input to
the `entities` sparse vector below.

Context and the general analysis (sparse-retrieval reframing, CSR/mmap, LMDB/RocksDB,
head-entity strategies) are in `../topics/reference/retrieval-storage-tooling.md`. Danielle's
earlier project fetched candidates from question entities over an entity-ID graph and hit
join blow-ups on frequent entities.

**Stack proposed for this project (near-verbatim).** Canonical pages and passages: Lance or
Parquet. Corpus construction, entity statistics, evaluation: DuckDB. Entity-to-page
retrieval: Qdrant sparse vector named `entities` (edge-quality weights, IDF modifier, cold
tier). Text retrieval: a separate Qdrant BM25 sparse vector. Dense retrieval: Qdrant dense
vector. Entity-to-entity graph: static memory-mapped CSR with a per-entity expansion budget.
Optional exact set operations: Roaring bitmaps for the highest-degree entities.
Cross-encoder or reader after fusion of entity-sparse + BM25 + dense. Fewest components:
Qdrant (entity sparse + BM25 + dense) + one mmap CSR file for neighbor expansion + Parquet
source data.

### Open questions (carried from staging)

- What the surrounding MAQA system needs (latency target, shard count, update
  cadence, whether first-stage retrieval is routable) — determines whether placement matters
  at all.
- Verify the gap claim and the tool/paper citations above.
- Whether this is a project in itself or infrastructure for another one (the entity-based
  candidate fetch revives Danielle's earlier MAQA retrieval approach; decide whether the
  entity graph is a retrieval signal here or its own project).
- Data plan (revised): derive nodes/edges from Structured Wikipedia Parquet (snapshot-aligned
  corpus + graph with link context); SQL `pagelinks` only as a completeness check; pinned
  dated snapshot. Verify the dataset's link schema and whether it is sharded such that a
  graph-coherent subset can be fetched without the full download.
- Index plan: Pyserini BM25 reference → LanceDB as the single-machine stack (one table,
  `graph_partition` as an indexed filter column; graph features computed offline) → Vespa
  only if concurrency, ranking expressiveness, or HA demand it (Qdrant is the alternative
  serving-first choice: named dense+sparse(+ColBERT) vectors, RRF, article grouping,
  community as payload not shard key); DuckDB (+ Lance extension) as the analytical
  workbench over the same Lance dataset; section-level passages with title/section prefix; RRF first;
  article-level diversification for multi-answer readers.
## 5. Related work and positioning

*Purpose: the paper-facing synthesis — the prior-art landscape, this project's
position in it, and what each closest neighbor lacks. Unlike §4 (a dated intake
log, which grows by appending new entries **above this section**), §5 is a
current-state statement: rewrite it as understanding changes. Positioning claims
are Danielle's to make; agent-supplied literature claims anywhere in this document
are unverified leads, not established facts.*

**Status: little related-work material on record (2026-08-24); positioning not yet
written.** What exists is one dated conversation record — the 2026-08-16 "Turn wiki
corpus into shards" entry in §4 — whose literature attributions the entry itself marks
unverified; there is no theme accumulator devoted to sharding or graph partitioning, and
no ledger rows for `SHARD`.

**Where the raw material lives:**

- §4 of this document, "2026-08-16 — Turn wiki corpus into shards" — the only
  literature-bearing record: partitioning algorithm families, the QA-graph paragraph
  (explicitly headed "unverified attributions"), and the claimed gap.
- `../topics/reference/retrieval-storage-tooling.md` — the tooling flank (DuckDB vs.
  LanceDB, the entity–page sparse-retrieval reframing, CSR/mmap, LMDB/RocksDB,
  head-entity strategies); product claims marked unverified. The engine survey
  (Pyserini, LanceDB, Vespa, OpenSearch, Qdrant, Milvus, ColBERT/PLAID) and the LanceDB
  and Qdrant deep-dives live in §4 of this document, which that file points back to.
- `../topics/reference/entity-linking-at-scale.md` — the linking stack behind SHARD-opt-1
  (ReLiK, ReFinED, BELA, `entity-linkings`); respondent's claims, unverified.
- `../topics/reference/multi-answer-qa-literature.md` and
  `../refs/multi-answer-qa-state-of-research-2026.md` — the workload side, relevant to
  SHARD-opt-3's question of whether list-QA evidence is sharding-friendly at all;
  agent-produced deep-search report, unverified.
- `../danielle-inputs.md` ("MAQA Next Steps" intake, 2026-08-22) — Danielle's verbatim
  framing prompt, and the 2026-08-22 decision splitting `SHARD` out as a post-PhD /
  "engineering break" project.
- `../litreview/citation-verification-ledger.md` — no `SHARD` rows; nothing below has
  been verified.

**Starting inventory for the synthesis** (everything here is from the 2026-08-16 response
and carries its unverified flag):

- **Partitioning formulation and tools named:** weighted balanced k-way graph partitioning
  (not unconstrained min-cut, whose trivial optimum is one giant shard) with multilevel
  heuristics — METIS, KaHIP, KaMinPar; hypergraph partitioning with the connectivity
  objective Σ f_q(λ_q − 1) via Mt-KaHyPar; spectral/normalized cut; Leiden/Louvain
  (noted: GraphRAG uses hierarchical Leiden for navigation, not physical sharding);
  vertex cuts / replication, cited to PowerGraph; streaming partitioning for updates.
- **QA-graph prior art (the entry's own "unverified attributions" list):** Learning to
  Retrieve Reasoning Paths over the Wikipedia Graph (Asai et al., ICLR 2020); multi-step
  entity-centric retrieval (Das et al., 2019); CogQA (Ding et al., 2019); Multi-hop Dense
  Retrieval (Xiong et al., ICLR); HippoRAG (entity/passage graphs with personalized
  PageRank). The entry characterizes all of these as using graph structure for
  retrieval/reasoning rather than physical placement.
- **Routing/ANN neighbor:** "Unleashing Graph Partitioning for Large-Scale Nearest
  Neighbor Search" (PVLDB), named as the closest thing on record to graph partitioning
  with an explicit router.
- **Database-placement neighbor:** workload-aware placement with partitioning, balancing,
  and replication, cited to Golab et al. — recorded as existing, but not in the
  Wikipedia-QA combination.
- **Datasets usable as workload hyperedges:** HotpotQA (~113k), WikiHop,
  2WikiMultiHopQA (explicit reasoning paths), MuSiQue (25k, 2–4 hops), plus QAMPARI
  evidence sets (§1, SHARD-1).
- **Corpus/graph artifacts surveyed** (2026-08-16 storage and link-graph entries, sizes
  and schemas unverified): the `wikimedia/structured-wikipedia` Parquet dataset; the SQL
  dump graph tables (`page`, `linktarget`, `pagelinks`, `redirect`, ≈11 GB) as the
  completeness check; and the ready-made alternatives — HotpotQA Wikipedia corpus with
  links (HF `ParthMandaliya/hotpotqa-wiki`, Oct-2017), USearchWiki (HF
  `unum-cloud/USearchWiki`, Aug 2025), SNAP enwiki-2013, WikiLinkGraphs (arXiv 1902.04298).
- **The gap claim itself, attributed not asserted:** the 2026-08-16 entry states that to
  "physically shard all of Wikipedia by observed QA evidence locality and evaluate
  end-to-end QA latency is not a saturated research problem." §2 of this document flags
  it as unverified, and the §4 open questions list "verify the gap claim and the tool/paper
  citations above" as outstanding.

# Wikipedia sharding for QA — co-locating what is retrieved together

**Kind:** staging. Candidate exits: a standalone project doc (systems / retrieval; program
pillars served: none), or absorption into a larger QA-infrastructure project if one emerges.
Gate: a literature check of the claimed gap ("physically shard all of Wikipedia by observed
QA evidence locality and evaluate end-to-end QA latency" is unsaturated) and of the named
tools/papers.

Source: excerpts from the Notion page "MAQA Next Steps" (MAQA = multi-answer question
answering; literature in `../reference/multi-answer-qa-literature.md`) (conversation dated 2026-08-16; intake
2026-08-22). The respondent browsed while answering, but every citation and tool claim below
is still **unverified** here — treat as leads.
---

## 2026-08-16 — Turn wiki corpus into shards

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

## 2026-08-16 — Downloading Wikipedia under a storage budget

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

## 2026-08-16 — Link graph from Structured Wikipedia

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

## 2026-08-16 — Index stack: BM25 to vectors at Wikipedia scale

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

## 2026-08-16 — LanceDB alone: pros, cons, how far it goes

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

## Open questions

- What the surrounding MAQA system needs (latency target, shard count, update
  cadence, whether first-stage retrieval is routable) — determines whether placement matters
  at all.
- Verify the gap claim and the tool/paper citations above.
- Whether this is a project in itself or infrastructure for another one.
- Data plan (revised): derive nodes/edges from Structured Wikipedia Parquet (snapshot-aligned
  corpus + graph with link context); SQL `pagelinks` only as a completeness check; pinned
  dated snapshot. Verify the dataset's link schema and whether it is sharded such that a
  graph-coherent subset can be fetched without the full download.
- Index plan: Pyserini BM25 reference → LanceDB as the single-machine stack (one table,
  `graph_partition` as an indexed filter column; graph features computed offline) → Vespa
  only if concurrency, ranking expressiveness, or HA demand it; section-level passages with title/section prefix; RRF first;
  article-level diversification for multi-answer readers.

**Waiting on:** further excerpts from the MAQA Next Steps page; a promotion decision.

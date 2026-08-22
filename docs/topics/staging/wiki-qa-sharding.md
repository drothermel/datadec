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

## Open questions

- What the surrounding MAQA system needs (latency target, shard count, update
  cadence, whether first-stage retrieval is routable) — determines whether placement matters
  at all.
- Verify the gap claim and the tool/paper citations above.
- Whether this is a project in itself or infrastructure for another one.
- Data plan: graph tables (~11 GB) first for subset construction; `pages-articles` multistream
  or Structured Wikipedia Parquet for text; pinned dated snapshot.

**Waiting on:** further excerpts from the MAQA Next Steps page; a promotion decision.

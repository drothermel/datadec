# Wikipedia sharding for QA — co-locating what is retrieved together

**Kind:** staging. Candidate exits: a standalone project doc (systems / retrieval; program
pillars served: none), or absorption into a larger QA-infrastructure project if one emerges.
Gate: a literature check of the claimed gap ("physically shard all of Wikipedia by observed
QA evidence locality and evaluate end-to-end QA latency" is unsaturated) and of the named
tools/papers.

Source: excerpts from the Notion page "MAQA Next Steps" (conversation dated 2026-08-16; intake
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

## Open questions

- What "MAQA" is and what the surrounding system needs (latency target, shard count, update
  cadence, whether first-stage retrieval is routable) — determines whether placement matters
  at all.
- Verify the gap claim and the tool/paper citations above.
- Whether this is a project in itself or infrastructure for another one.

**Waiting on:** further excerpts from the MAQA Next Steps page; a promotion decision.

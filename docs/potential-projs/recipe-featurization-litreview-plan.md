# Literature review plan: recipe featurization

Status: approved design, 2026-08-21. Companion to [recipe-featurization.md](recipe-featurization.md).

Goal: a comprehensive, quality-annotated review of the research subdomains behind the recipe-featurization project — foundational papers plus late-2025/2026 work — with every paper verified, cataloged in the local paper corpus, and tiered by relevance, reception, and rigor. No cap on paper count; the set is bounded only by relevance and by search exhaustion.

## Subdomains (confirmed)

| Id | Subdomain | Serves |
|---|---|---|
| A | Dataset featurization & intrinsic corpus statistics (WIMBD, diversity coefficient, compression/entropy measures, Task2Vec alignment, similarity-is-not-enough results) | D1, D3 |
| B | Model-mediated data valuation & mixing laws (DoReMi, RegMix, data-mixing laws, BiMix, perplexity correlations) | D3 baselines |
| C | Token-level uncertainty & loss-landscape geometry (river-valley/WSD, epistemic–aleatoric token decomposition, Rho-1 token trajectories, forking/high-entropy tokens) | D2, D4 |
| D | Scaling suites, proxy metrics, evaluation noise & annealing protocols (DataDecide, model ladders, Signal-and-Noise, Hägele/MiniCPM cooldowns, checkpoint merging, multi-power law) | C-c, D4 |
| E | Data attribution & per-source effects (domain ablations, domain upsampling, end-of-training data valuation) | D5 |

Post-training amplification work (Echo Chamber, RLVR support results) is out of scope for this review and belongs to a separate one.

## Artifacts and locations

- Working packet: `~/drotherm/data/.claude/datadec/<YYYY-MM-DD>/<HHMM>-recipe-featurization-litreview/`
  - `candidates.jsonl` — the single canonical exchange contract (schema below)
  - `briefs/<A–E>.md`, `plan.md`, `cards/<paper-id>.md`, `synthesis/<A–E>.md`, `quality.jsonl`, `run-log.md`
- Final review: `docs/potential-projs/recipe-featurization-litreview.md` (this repository), with a `.html` rendering beside it in the packet
- Paper corpus: `~/drotherm/data/papers/` via `paper-corpus add` and `paper-corpus enrich` only
- `paperpile-import.txt` emitted for later manual import; no Paperpile or Notion mutation in this run

### `candidates.jsonl` contract

One JSON object per line:

```json
{
  "key": "arxiv:2504.11393 | doi:10.xxxx/… ",
  "title": "…",
  "authors": ["…"],
  "year": 2025,
  "first_posted": "2025-04-15",
  "venue": "ICML 2025 | arXiv | OpenReview:ICLR 2026 …",
  "subdomains": ["D"],
  "role": "foundational | recent | method | negative-result | dataset | survey",
  "relevance_to": ["C-a", "D2"],
  "relevance_tier": "core | supporting | peripheral | null",
  "one_line_claim": "…",
  "why_it_matters": "…",
  "source_of_lead": "existing-corpus | citation-graph | query | venue | critic",
  "identity_status": "exact | title-only",
  "paper_id": "paper-<uuidv7> | null",
  "quality": {"external": {…}, "rigor": {…}, "contestation": "…"}
}
```

Dedup key is the normalized identifier; title normalization generates candidates but never merges on its own (same rule as `paper-corpus`).

## Quality assessment

Two independent axes plus relevance tier, reported side by side and never collapsed into one score.

### External reception (age-aware)

| Signal | Source | Notes |
|---|---|---|
| Citation velocity (cites/month since first posting) | OpenAlex `cited_by_count`, Semantic Scholar `citationCount`, `influentialCitationCount` | reported with an age band |
| Cohort percentile of velocity | OpenAlex works in same topic and posting month | only for papers ≥6 months old |
| Who cites it | OpenAlex/S2 citing works ∩ review's foundational set; number of subdomains citing | few right cites beat many random ones |
| Venue and review evidence | acceptance, oral/spotlight; OpenReview reviewer scores and decision text where exposed | strongest signal under 12 months |
| Artifact uptake | code/data released; GitHub stars, HF downloads normalized by age | weak, honest |
| Author prior | OpenAlex author metrics | shown separately; most biased signal |

Papers under 6 months old are labeled `too-new-for-external` rather than scored low.

### Internal rigor (from the deep-read card, quoted evidence required)

- controlled comparison vs. observational design
- seeds and variance reported; scale and compute stated
- effect sizes with uncertainty
- mechanism ablations; negative results reported
- artifacts sufficient to reproduce
- **transfer to our setting**: assumptions DataDecide violates (annealed endpoints, token-share mixture labels, >1B scale, instruction-tuned bases)

### Contestation (from synthesis)

Per cited claim: confirmed / contested / unreplicated by later work, with the citing papers named.

## Orchestration

Three Claude Workflows in sequence; the orchestrating session reviews between them. All workers are Opus, fresh per task, self-contained prompts, read-only except one designated corpus writer.

### Workflow 1 — Scope (~6 agents)

1. **Seed extraction** (1): harvest featurization/mixing/token-uncertainty/annealing/DataDecide papers from `phd@08-19-convos:docs/referenced-papers.md` and the research-trajectory transcripts into `candidates.jsonl` (`source_of_lead: existing-corpus`).
2. **Subdomain briefs** (5, parallel): per subdomain, ≤1 page — question answered for the project, expected canonical papers, search vocabulary (terms, venues, authors), adjacent subdomains, and exclusion rules.
3. **Scope judge** (1): overlaps, gaps, and a per-subdomain recency emphasis. Orchestrator reviews before Workflow 2.

### Workflow 2 — Plan (~3 agents)

Per pair of subdomains, a **search plan**: exact query strings per source (arXiv listing API, OpenAlex citation graph, Semantic Scholar, OpenReview venue pages for ICLR 2026 / COLM 2025 / NeurIPS 2025 / ICML 2026 and relevant workshops), backward/forward citation seeds, and exclusion rules. A planner merges them into `plan.md` with the round structure and logging requirements.

### Workflow 3 — Execute (pipelined per subdomain, no cross-subdomain barriers)

1. **Multi-modal sweep** — three blind finders per subdomain per round: *citation graph* (forward/backward from current seeds), *query* (plan strings; a separate pass restricted to 2025-09 → present), *venue* (accepted lists and workshop proceedings).
2. **Dedup in code** against everything seen, identifier-first.
3. **Identity verification** (1 agent per ~15 candidates): resolve to exact arXiv ID/DOI, confirm title/authors/year against the primary record, set `identity_status`. `title-only` papers are never cited as fact.
4. **Loop-until-dry**: verified papers become next-round seeds; a subdomain stops after two consecutive rounds with nothing new; hard stop at 5 rounds, logged if hit.
5. **Relevance tiering** (2 independent adjudicators per paper, disagreement → a third): core / supporting / peripheral, with `relevance_to` directions. Nothing is dropped for relevance unless both adjudicators mark it outside all five subdomains.
6. **Quality scoring** (1 agent per ~20 papers): the external-reception lookups above, written to `quality.jsonl`.
7. **Deep-read cards**: full cards (setup, data/scale, citable claim, effect sizes, transfer assumptions, rigor rubric) for core and supporting tiers, 5–8 papers per agent, reading full text from `paper-corpus enrich` artifacts or arXiv HTML; short cards for peripheral.
8. **Corpus ingestion** (1 serial writer): `paper-corpus add` when absent, then `paper-corpus enrich <run-id> --paper-id <id>`; write `paper_id` back into `candidates.jsonl`; record the phd working-tree state (branch, HEAD, dirty-file digest) in `run-log.md`. No `notion apply`, no Paperpile import.
9. **Synthesis** (1 per subdomain, then 1 integrator): lineage → late-2025/2026 frontier → contestation → gaps mapped to C-a/D1–D5; integrator produces the review with a per-direction known / contested / never-done table and a citation list keyed to corpus paper IDs.
10. **Completeness critic** (1): unswept modalities, claims resting on `title-only` papers, directions with <3 recent papers, subdomains that hit the round cap. One bounded follow-up round, then stop.

## Invariants

- No paper enters synthesis without `identity_status: exact` and a retrieved primary record.
- Recency is a dedicated date-filtered sweep, not a side effect of general search.
- Dedup is against *seen*, not *accepted*.
- Cards quote setups and effect sizes; abstracts are not evidence.
- Corpus writes are append-only (`add`, `enrich`); the phd repository and Notion/Paperpile are untouched.
- Every `log()` line reports papers seen / verified / carded and tokens spent.

## Expected scale

A few hundred verified papers; roughly 6–10M output tokens across the three workflows. Workflow 1 is cheap and is the checkpoint for adjusting the partition before the expensive phases.

## Next steps after this run (not in scope)

Notion Research Papers synchronization and Paperpile import of `paperpile-import.txt`, once the phd branch's concurrent work lands.

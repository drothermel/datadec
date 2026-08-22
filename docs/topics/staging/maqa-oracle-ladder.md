# MAQA oracle ladder — decomposing multi-answer QA into measurable ceilings on QAMPARI

**Kind:** staging. Candidate exit: a standalone project doc (multi-answer QA; program pillars
served: none — though "measurement science at academic scale" is the same spirit). Gate: a
decision to pursue MAQA again, and confirmation of the QAMPARI release details assumed below
(official dump + chunking, metrics, enriched 200-question subset, construction metadata).

Source: excerpts from the Notion page "MAQA Next Steps" (conversation dated 2026-08-16; intake
2026-08-22). Literature context in `../reference/multi-answer-qa-literature.md`;
infrastructure in `wiki-qa-sharding.md`, `../reference/retrieval-storage-tooling.md`,
`../reference/entity-linking-at-scale.md`. Dataset facts quoted below are the respondent's
and **unverified**.
---

## 2026-08-16 — Danielle's goal (the seed)

Lesson from the first attempt: "there are so many pieces that interact, and there are kind
of infinitely complex ways to try to address each of the pieces." What she wants: "start
with the simplest of heuristic baselines and decompose the different pieces of the problem
as much as possible to try to understand how well different pieces work, especially as you
scale them up or down." Pieces she named: question entity linking; retrieval; the maximum
possible retrieval under a given assumption (e.g. the entity universe = Wikipedia entity KB
or normalized page titles gives an upper bound — "we can't predict entities that are not
included in that set"); caps along the way (an evidence-passage cap bounds some questions
below 100% even with perfect, fully diverse evidence); whole-Wikipedia retrieval checked for
presence of any gold evidence passage / the correct answer, to learn "whether we can handle
complex questions with a single round of retrieval"; reading given the full gold evidence
set, together or individually, and "whether our evaluation metric actually considers correct
answers correct." "Each of those things are things that can be analyzed before trying to
hook up the whole system together."

## 2026-08-16 — The oracle ladder (response, near-verbatim, condensed)

"Formalize it as an oracle ladder: at each stage, compare the simplest heuristic with an
oracle replacement for that stage. That gives you empirical ceilings without assuming the
components are independent." QAMPARI: ≥5 entity answers per question, usually 1–2 evidence
passages per answer, passages ~100 words; official metrics P/R/F1, recall ≥ 0.8, F1 ≥ 0.5.

**Ceilings (measure each conditional intervention directly; do not multiply).**
Answer-universe (gold answers representable in the entity vocabulary?) → corpus (snapshot
contains usable evidence?) → chunking (passage construction preserves it?) → candidate pool
(any useful passage in top N?) → evidence budget (could an oracle choose K from the pool to
cover the answers?) → reader (recovers answers from perfect evidence?) → evaluator
(recognizes known-correct answers and aliases?) → end-to-end. Illustrative funnel for one
question: 8 annotated → 7 representable → 6 with evidence in snapshot → 5 in top-1,000 pool →
3 covered by actual top-10 → 5 coverable by oracle 10-of-pool → 4 extracted from gold
evidence → 3 accepted by the evaluator.

**1. Answer universe and normalization (first, everything depends on it).** Progressive
universes: exact normalized titles → + redirects → + Wikidata labels → + Wikidata aliases →
+ observed anchor strings. Per gold answer record gold string, normalized string, page ID,
QID, mapping source, candidate count, ambiguity. Measure mapping coverage, unique-mapping
coverage, ambiguity rate, coverage by answer frequency / entity type / question type
(simple, composition, intersection), string- vs. entity-ID equivalence. Simplest normalizer:
NFKC → casefold → whitespace → punctuation → strip leading articles → optionally strip
parenthetical disambiguator; keep every intermediate form. This is a ceiling only for an
entity-closed system. *Evaluator self-test:* feed canonical gold answers, supplied aliases,
titles, redirects, Wikidata labels — recall must be 100% on canonical gold strings.

**2. Question entity linking.** Separate mention detection / candidate generation /
disambiguation. Baselines: longest exact title match; title + redirects; alias top-1 prior;
alias top-M (M ∈ {1, 4, 16, 64}); gold spans + heuristic ID; gold entity IDs. Report span
recall, candidate recall@M, top-1 accuracy given candidate recall, whole-question entity-set
exact match, and — crucially — downstream retrieval coverage with gold vs. predicted
entities. Use QAMPARI construction metadata as the oracle if exposed; otherwise annotate a
diagnostic subset.

**3. Corpus and chunking.** Start from the released QAMPARI dump and chunking. Three
availability signals per answer: exact gold-passage presence (precise, brittle), gold entity
occurs in a passage (robust, relation unproven), answer alias occurs lexically (optimistic
extractability). Then a small grid: passage size {128, 256, 512} × overlap {0, 64} × unit
{paragraph, section, page}; measure gold evidence retained in one chunk, split across
boundaries, answers per passage, index size, latency.

**4. Retrieval.** Three simple systems: BM25 over the question; entity-postings retrieval
from predicted entities; BM25 + entity via RRF. Dense only after these are stable. Depths
N ∈ {10, 50, 100, 500, 1,000, 5,000}; store top-5,000 with raw scores once. Per N: exact
evidence recall |{a : E_q(a) ∩ R_N ≠ ∅}| / |G_q|; answer-bearing coverage (alias occurs);
entity-bearing coverage (linked IDs); threshold success at recall ≥ 0.25 / 0.5 / 0.8 / 1.0;
coverage curve with area under recall-vs-log N as the summary.

**5. Candidate generation vs. ranking vs. diversity.** Oracle budgeted coverage
U_{N,K}(q) = max over S ⊆ R_N, |S| ≤ K of |∪_{p∈S} C(p)| / |G_q| — a maximum-coverage problem;
greedy suffices for diagnostics, ILP for exact values on small pools. Compare actual top-K,
greedy diversity selection, pool oracle, global gold. Gaps: global gold − pool oracle =
candidate-generation loss; pool oracle − diversity result = selection loss; diversity −
actual top-K = ranking/redundancy loss. Values: N ∈ {100, 1,000, 5,000}; K ∈ {5, 10, 20, 50,
100}; per-article cap {1, 2, 4, ∞}; plus diagnostic K ∈ {0.5, 1, 2, 4} × |G_q|.

**6. Trivial diversity baseline.** Round-robin by article (one per article, then a second,
until K); compare to unrestricted top-K, strict one-per-article, two-per-article, greedy
oracle — tells whether the problem is repeated chunks or missing articles.

**7. Single-round vs. iterative.** Ladder: question only → + gold question entities → + one
oracle answer entity → + entities from one oracle evidence passage → two rounds with system
outputs → two rounds with oracle outputs. Differences isolate query-formulation loss, the
need for multi-stage discovery, and error propagation. Graph expansion: hops {0, 1, 2} ×
neighbors {10, 100, 1,000} per seed (two hops uncapped "will mostly measure graph
explosion").

**8. Reader ceilings.** Four inputs: one gold passage at a time; all gold independently
(map-and-union bound); all gold jointly (aggregation/global filtering); retrieved. Non-neural
readers first: article title as answer; every gold-vocabulary alias in the passage; every
linked entity; entity-linked filtered by expected type — poor precision, useful
extractability ceilings (alias present but extractor misses = extraction problem; extractor
finds but reader rejects = semantic filtering problem). Model readers: independent + union;
independent + score aggregation; joint over first K; joint over all gold; contexts {2k, 8k,
32k}; passages {1, 5, 20, all gold}. Conditional metrics: recall given alias present; recall
given exact gold evidence; unsupported-prediction rate; duplicate rate; accuracy vs. evidence
position; accuracy when a filter/intersection must be applied ("answer occurrence" and
"reader success" stay separate).

**9. Output-budget ceiling.** RecallCeiling_L = min(L, |G_q|) / |G_q|; report under L ∈ {5, 10,
20, 50, 100, ∞}; record token truncation separately from answer-count truncation.

**10. Gold incompleteness.** Use the 200 enriched test questions as a separate diagnostic
set. Report official closed-world precision; precision on the enriched subset; sampled
adjudicated precision of off-gold predictions; entity-ID precision after alias resolution;
novel valid answers per question. Adjudicate stratified samples of apparent false positives
(high-confidence novel, lexically present but unsupported, alias collisions, temporal/filter
boundary, annotation omissions) — especially for "ever", date restrictions, intersections,
composition.

**Minimal experiment matrix (not a factorial).** Entity universe {titles, +redirects,
+Wikidata aliases}; question EL {longest match, top-1 prior, top-16, gold IDs}; retriever
{BM25, entity sparse, RRF hybrid}; pool {100, 1,000, 5,000}; selector {top-K, article
round-robin, oracle greedy}; budget {5, 20, 100}; reader {alias extraction, entity
extraction, one model reader}; reader input {individual gold, joint gold, retrieved};
evaluator {official string, normalized alias, canonical ID}; output cap {10, 50, ∞}. Then
only material interactions: candidate cap × depth; passage length × budget; pool depth ×
selector; context length × evidence count; evaluator normalization × reader precision.

**Organization.** Freeze stage interfaces; persist every intermediate (questions,
question_entity_candidates/predictions, gold_answer_entities, passage_answer_coverage,
retrieval_runs_top_5000, evidence_selections, reader_predictions, normalized_predictions,
evaluation_matches) so any stage can be re-evaluated, swapped for its oracle, or re-scored
without rerunning upstream. Macro over questions as primary (so 50-answer questions don't
dominate), micro alongside; stratify by |G_q|, evidence count, question type, answer
frequency, entity ambiguity, evidence redundancy, representability. Report as a funnel: KB
coverage → evidence availability → pool coverage → budgeted oracle coverage → selected
coverage → gold-evidence reader recall → retrieved-evidence reader recall →
evaluator-recognized recall and precision — "whether the next unit of complexity should go
into entity linking, retrieval breadth, diversity, iterative search, reading, or simply
fixing the evaluator."

## Open questions

- Confirm QAMPARI artifacts: released dump + chunked corpus, construction metadata (source
  entities/relations), the 200-question enriched subset, official metric scripts.
- Whether this becomes the first MAQA paper on its own (a diagnostic/funnel paper) or the
  measurement layer under a system paper.
- Relationship to the cleaner-dataset pipeline in the literature topic (verification-first
  gold sets) — same project or a follow-on.

**Waiting on:** further excerpts; a promotion decision.

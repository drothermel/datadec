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

## 2026-08-17 — Danielle's three-paper arc (program framing for MAQA)

Her own statement of the project, paraphrased from the prompt (verbatim in
`../../danielle-inputs.md`):

1. **Revisit the original approach** — same shape, same-era datasets (QAMPARI, QUEST, RoMQA;
   MoNaCo named too), applying what she has learned since to make faster/better progress.
2. **A variant that fits the current landscape**, staying moderately close to the brute-force
   design, as the target first paper (workshop likely) — "a heuristic baseline from someone
   who doesn't care about people thinking the approach is cool and just wants to see how far
   we can get with simple clean approaches."
3. **One or two papers on more interesting approaches targeting specific failure points.**

The original brute-force design she describes: identify all entity mentions in questions and
Wikipedia passages; split Wikipedia into 100-word chunks; build a graph with entity mentions
as edges; traverse for candidate fetch; cluster evidence by entity; rerank per entity for a
subset to read per entity. Her view: "building a knowledge base like that from Wikipedia and
doing more exhaustive retrieval has actually held up in the era where LLMs are super strong
but correctness is the issue," while string matching, surface-form entity resolution, long
tails, and the infra constraints they imposed were the frustrating part. Her standing
dataset-quality complaints: missing answers in exhaustive gold sets; heuristically selected,
often incorrect evidence passages; scoring that penalized answer formatting more than
correctness.

**Feedback received (near-verbatim).** "The overall arc — [revisit], then an
unglamorous-but-clean brute-force baseline, then targeted failure-point papers — is coherent
and well-timed. The 'simple exhaustive baseline that nobody bothered to run properly' niche
is exactly the kind of thing workshop reviewers tend to appreciate right now, especially
given how much of the GraphRAG literature avoids multi-answer benchmarks." Two plan notes:
(1) contamination — include a closed-book baseline to separate "retrieval helped" from "the
model already knew" (MoNaCo evaluates closed-book as a separate setting); (2) the datasets are
frozen against specific Wikipedia dumps — pin the corpus version and decide whether to report
on original dumps vs. current Wikipedia, for comparability and for the eval-cleanliness
story. Also: a re-annotation / eval audit of QAMPARI/QUEST/RoMQA with modern tooling "could
itself be a credible workshop contribution folded into" paper 1; graph-based exhaustive
retrieval evaluated on QAMPARI/QUEST/RoMQA/MoNaCo "appears genuinely underexplored" — a
favorable gap for paper 2. Date correction: MoNaCo is 2025 (TACL 2026), not from the original
era.

## 2026-08-17 — A second, leaner version of the ladder

Same decomposition question posed again (prompt identical to the 2026-08-16 seed); this
response is shorter and adds a few distinct ideas. Near-verbatim, condensed.

**Framing.** "Your component ceilings won't compose multiplicatively (errors correlate —
questions with hard entity linking also tend to have hard retrieval), so treat them as
brackets, not a factorization. The most useful artifact is a table per question of 'which
stage first made 100% impossible,' which gives you a loss attribution you can track as you
swap components."

**Measurements in dependency order.**
0. **Closed-book baseline first** — contamination control and "embarrassingly often, a
   strong baseline on 2022 Wikipedia-derived datasets"; run with both the official metric
   and a lenient judge — the gap is the first estimate of eval noise.
1. **Answer-vocabulary ceiling.** Titles vs. titles + redirects ("redirects are the poor
   man's alias table"); fraction of gold answers resolving to a page, per question and
   aggregate; failures are informative (dates, quantities, non-entity answers).
2. **Question entity linking.** Longest-match string lookup against the title + redirect
   table, no ML. Metrics: does the linked set include the seed entity from QAMPARI's
   construction metadata; and, more operationally, for what fraction of questions does at
   least one linked entity's page or immediate neighborhood contain ≥1 gold answer —
   "whether linking failures actually matter downstream or get rescued by retrieval."
3. **Single-round retrieval ceiling.** BM25 over 100-word chunks, k ∈ {10, 50, 100, 200,
   1000}: gold-evidence-passage recall and answer-presence recall. Make **answer-presence
   primary** (gold evidence was heuristically selected) and gold-evidence secondary;
   "answer-presence overstates … gold-evidence understates … The two together bracket true
   retrieval quality." Report per-question all-answers-covered@k (MRecall-style) since
   averages hide the exhaustiveness failure.
4. **Budget/diversity ceiling.** Answer→passage bipartite map (answer string appears in
   passage) → greedy set cover per question → minimum passages to cover all answers. "The
   distribution of that number is one of the most decision-relevant plots you'll produce —
   it directly tells you what reading budget k makes 100% recall even theoretically
   possible." Then oracle-at-budget-k.
5. **Graph reachability ceiling** (for the brute-force design). From linked question
   entities, what fraction of gold answers lie within 1 hop / 2 hops? Compare two graphs:
   edges from **wikitext anchor links** (pre-annotated mentions in the raw dump — no NER, no
   resolution, nearly free) vs. edges from **string-matched mentions** (the old approach).
   "The delta between those two tells you how much your old entity-resolution pain was
   actually buying you. This is a nice, cheap ablation with a clear story."
6. **Reading ceiling.** Gold evidence (or oracle set-cover passages) to the LLM,
   all-concatenated vs. per-passage with union + dedupe; official metric plus LLM judge or
   manual audit on a sample. Yields gold-context F1 (ceiling of any retrieve-then-read
   system), the concat-vs-independent gap (whether reading needs clustering/parallelizing at
   all), and the official-vs-audited gap (eval noise, decomposable into formatting/alias
   failures vs. genuinely missing gold).
7. **Metric self-test.** Run the gold answers through the metric after surface perturbations
   (redirect aliases, "The X" vs "X", diacritics, reformatted dates); "any score below 100%
   is pure metric artifact, quantified."

**Simplest end-to-end floor.** string-link → BM25@k → read → normalize via redirect table →
dedupe. "The ceiling ladder is the paper, with the baseline as its floor."

**Cross-reference to the 2026-08-16 version.** The two ladders agree on structure; distinct
contributions here are the per-question first-failing-stage table, closed-book as step 0,
answer-presence as the primary retrieval signal, the minimum-set-cover distribution as a
headline plot, and the anchor-link vs. string-match graph ablation. The earlier version is
more complete on the experiment matrix, the N/K/L value grids, output-budget and
gold-incompleteness protocols, and persisted intermediates.

## 2026-08-17 — Problem definition, solution shape, impact hypotheses (Danielle)

Applying principle 1 of `../reference/project-approach-principles.md`:

**Problem.** Exhaustive QA for QAMPARI-style questions — entity-centric, few hops max, some
simple set operations, generally 4–10 answers — over an unannotated knowledge corpus;
success = F1 on the cleanest possible dataset.

**Solution space.** (1) prepare the corpus; (2) get evidence from it; (3) choose which
evidence to use; (4) use the evidence to answer — entity-centric: form an entity set and
answer from it.

**What will impact outcomes most (hypothesis list).**
- a complete but precise entity set to select answers from;
- evidence chunks small enough that all chunks needed for multi-hop questions fit usefully in
  the answering mechanism's input;
- diversity of retrieved evidence for long-tail answers;
- normalization of entity mentions so they are linkable and match the expected answer form;
- an answering mechanism that is accurate given clear, correct evidence;
- some way to limit evidence explosion under time and infra constraints.

**Plan under principles 3 and 4.** Investigate the scale of QAMPARI's issues beyond the
original paper's own investigation (entity string matching, evidence representativeness,
answer-set completeness), then quickly use modern tools to improve the dev set and the
evaluation metric in a best-effort way. Start with QAMPARI for historical reasons.

**Feedback specific to this plan (near-verbatim).** Each hypothesis maps to a ceiling
measurement — entity-set completeness → answer-vocabulary ceiling; chunk size/budget →
set-cover distribution; normalization → metric self-test; answering given clean evidence →
gold-context reading ceiling — annotate them as such. The list is recall-heavy: "F1 on
exhaustive QA punishes overgeneration hard, and with modern LLMs the highest-risk component
has arguably shifted from 'can it extract answers from clean evidence' (mostly solved) to
'does it know when to stop' — inclusion thresholding and abstention calibration"; QAMPARI's
paper found over-prediction was a real failure mode — **add precision control as a
first-class impact axis**, and promote eval fidelity into the list. The entity-set
commitment makes non-entity answers (dates, quantities, works without pages) structurally
impossible — state it as "the entity-set assumption caps recall at X%, we accept that."
Three early distribution plots: answers per question; minimum passages per question (set
cover); entity-mention frequency. Clean-set hygiene: time-box, version (e.g.
`qampari-dev-clean-v1`), deterministic and publishable pipeline, official and cleaned
numbers side by side; the noise-vs-difficulty audit is "plausibly the most citable single
result." Keep a small frozen QUEST or RoMQA slice as a transfer smoke test — QAMPARI is
Wikidata/table-derived (relation-composition heavy, uniform answer style), unlike QUEST's
category intersections and RoMQA's constraint clusters. Decide kill criteria for the
entity-centric shape up front; put cost on every plot — "the F1-vs-compute curve is the
contribution."

## Open questions

- Confirm QAMPARI artifacts: released dump + chunked corpus, construction metadata (source
  entities/relations), the 200-question enriched subset, official metric scripts.
- Where the oracle ladder sits in the three-paper arc: it is the natural spine of paper 1
  (revisit + eval audit) and the measurement layer for paper 2 (clean brute-force baseline).
- Which graph for paper 1's reachability ceiling: anchor links (free) vs. string-matched
  mentions vs. the ReLiK-style linker from `../reference/entity-linking-at-scale.md`.
- Relationship to the cleaner-dataset pipeline in the literature topic (verification-first
  gold sets) — same project or a follow-on.

- Kill criteria for the entity-centric shape (undecided); precision-control / stopping
  mechanism as an explicit impact axis.

**Waiting on:** further excerpts; a promotion decision.

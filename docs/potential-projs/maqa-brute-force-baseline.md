# MAQA brute-force baseline — exhaustive multi-answer QA over raw Wikipedia, decomposed into ceilings

> **Draft scaffolding (2026-08-22).** Promoted from the staging topic `maqa-oracle-ladder`.
> §1–§3 are synthesized from the 2026-08-16/17 discussions and Danielle's own problem
> statement; §4 is the dated record. Treat §1–§3 as provisional until this note is removed.

**Program pillars served:** none — outside the DataDecide program; gets its own topic
proposal if pursued (pending). Its spirit — measure ceilings before building systems — matches
the "how" pillar's measurement-science stance. (Program: `README.md` → Program.)

**One-line pitch.** Revisit exhaustive multi-answer QA (QAMPARI-style: entity-centric, few
hops, simple set operations, 4–10 answers) over an unannotated Wikipedia corpus with the
simplest clean brute-force design — entity-centric candidate fetch, evidence selection,
reading — and make the *oracle ladder* the paper: per-stage ceilings (answer universe,
corpus, chunking, retrieval pool, evidence budget, reader, evaluator) with simplest-heuristic
baselines and oracle swaps, per-question loss attribution, and cost on every plot.

IDs: MAQA-1–MAQA-4 (papers/phases), MAQA-opt-1–MAQA-opt-5.

**Paper goal.** Paper 1 (workshop): the revisit + eval audit — the ceiling ladder with the dumb
floor system as its floor, official and cleaned numbers side by side, noise-vs-difficulty
audit of QAMPARI. Paper 2 (workshop or main conference): the clean brute-force baseline
updated for the current landscape, F1-vs-compute as the contribution. Papers 3+: targeted
failure-point methods. Thesis-scale if MAQA becomes an alternative topic proposal.

Compute tiers: **CPU/API** = corpus processing, BM25, and API reads; **GPU-light** = dense
embeddings and an entity linker over Wikipedia once.

---

## 1. What the project involves

### Problem and solution shape (Danielle, 2026-08-17)

Exhaustive QA for QAMPARI-style questions over raw Wikipedia; success = F1 on the cleanest
possible dataset. Solution space: prepare corpus → get evidence → choose evidence → answer,
entity-centric (form an entity set, answer from it). Impact hypotheses: a complete but
precise entity set; evidence chunks small enough that multi-hop evidence fits the answerer;
evidence diversity for the long tail; mention normalization that is linkable and matches
answer form; an answerer accurate on clean evidence; bounded evidence explosion. Added on
feedback: **precision control / knowing when to stop** and **eval fidelity** as first-class
axes; the entity-set commitment stated as a measured cap ("caps recall at X%; accepted").

### Core experiment: the oracle ladder on QAMPARI (MAQA-1)

Start with QAMPARI only (released dump + chunking; official metrics). Run, in dependency
order, each with the simplest heuristic and an oracle swap:

0. Closed-book baseline (contamination control), scored officially and with a lenient judge.
1. Answer-universe ceiling: titles → +redirects → +Wikidata labels/aliases → +anchor strings;
   evaluator self-test on canonical gold strings (must be 100%).
2. Question entity linking: longest-match title/redirect lookup; alias priors with M ∈ {1,
   4, 16, 64}; gold spans; gold IDs — with downstream retrieval coverage as the metric that
   matters.
3. Corpus/chunking: exact gold-passage presence, entity presence, alias presence; small
   size/overlap/unit grid.
4. Single-round retrieval: BM25, entity-postings, RRF hybrid; N ∈ {10…5,000} stored once;
   exact-evidence recall, answer-presence (primary), entity-bearing coverage, threshold
   success, area under recall-vs-log N; all-answers-covered@k.
5. Budget/diversity: greedy set cover → minimum passages per question (headline plot);
   oracle-at-budget U_{N,K}; loss decomposition into candidate-generation / selection /
   ranking; article round-robin as the trivial diversifier.
6. Graph reachability: gold answers within 1–2 hops of linked question entities, comparing
   anchor-link edges (free) vs. string-matched mentions vs. a neural linker.
7. Single-round vs. iterative ladder (gold question entities; oracle answer entity; oracle
   evidence entities; two rounds system vs. oracle).
8. Reader ceilings: one gold passage / all gold independent / all gold joint / retrieved;
   non-neural extractors first; conditional metrics incl. unsupported-prediction rate.
9. Output-budget ceiling min(L, |G|)/|G|; metric self-test under surface perturbations.
10. Gold incompleteness: the 200 enriched test questions as a diagnostic set; adjudicated
    precision of off-gold predictions.

Deliverable: a per-question "which stage first made 100% impossible" table and a funnel
report (KB coverage → evidence availability → pool → budgeted oracle → selected → gold-evidence
reader → retrieved-evidence reader → evaluator-recognized P/R), each with cost.

### Bounded dataset-noise work (part of MAQA-1)

Error-driven annotation of a small subset using a union of generic baselines (BM25, dense,
closed-book LLM) as the sampling frame — never the proposed method; system-blind judging;
freeze and hash `qampari-dev-clean-v1` before the method exists; report noise-rate scalars
and error bars, official and cleaned numbers side by side; a ~50-line answer normalizer with
a test file of known-equivalent pairs. State the residual bias (lower bound on noise) as a
limitation. Time-boxed.

### The floor system (MAQA-2)

string-link → BM25@k → read → normalize via redirect table → dedupe. Shipped first; the
ladder explains it stage by stage. Then the updated brute-force variant: entity-sparse +
BM25 + dense candidates with RRF, article grouping, per-entity evidence clustering and
reranking, reader with explicit inclusion thresholding; F1-vs-compute curves.

### Optional directions

- **MAQA-opt-1 — Verification-first re-annotation of QAMPARI/QUEST/RoMQA** (pool candidates
  from heterogeneous systems; pointwise structured verdicts; adversarial evidence
  judgments; independent verifier families; capture–recapture estimates of residual
  incompleteness). No cleaned versions of these datasets exist; a credible standalone
  contribution.
- **MAQA-opt-2 — Neural entity linking for every mention** (ReLiK/ReFinED cascade with
  lexical proposals and document propagation) replacing string matching; bake-off on 5–20k
  chunks first.
- **MAQA-opt-3 — Graph-based exhaustive retrieval evaluated on multi-answer benchmarks**
  (the GraphRAG lineage never evaluates on QAMPARI/QUEST/RoMQA/MoNaCo) — paper 3 candidate.
- **MAQA-opt-4 — Failure-point methods**: inclusion thresholding / abstention calibration;
  coverage-aware iterative retrieval conditioned on verified entities; set-operation
  handling.
- **MAQA-opt-5 — MoNaCo as the second dataset** once the design stabilizes (paper-time only;
  no transfer slices during iteration).

## 2. Doability and impact

### Overall doability: **high** for MAQA-1/2 (CPU/API; QAMPARI-scale), **medium** for paper-3 methods

Everything in the ladder is computable from the released QAMPARI corpus with BM25 and API
reads; the only GPU pass is optional (dense embeddings, neural linker). Risks: QAMPARI
artifacts (construction metadata, enriched subset, metric scripts) may not be as assumed;
contamination of 2022 Wikipedia answers in frontier models; the cleaning-and-tooling rabbit
hole — mitigated by the time-box and "front-load findings, defer scaffolding."

### Per-direction impact

- **MAQA-1 (ladder + audit).** Workshop paper; the "how much of QAMPARI's headline difficulty
  is dataset noise vs. task difficulty" number is the most citable result.
- **MAQA-2 (clean brute force).** Workshop or main conference depending on how far it gets;
  the F1-vs-compute curve is the claim.
- **MAQA-opt-1.** Standalone resource paper if done properly; otherwise folded into MAQA-1.
- **MAQA-opt-3/4.** Main-conference ceiling; depends on what the ladder says is the binding
  stage.

## 3. Infrastructure build sequence

1. **Corpus + metric loaders** for QAMPARI (dump, chunks, gold evidence, official scorer);
   pinned snapshot; persisted intermediates (questions, entity candidates/predictions, gold
   answer entities, passage–answer coverage, top-5,000 retrieval runs, evidence selections,
   reader predictions, normalized predictions, evaluation matches).
2. **Title/redirect/alias tables** and the answer normalizer with its test pairs.
3. **BM25 index** (Pyserini reference; LanceDB for hybrid experiments — see
   `wiki-qa-sharding.md` §3 for the engine choice rationale, duplicated there).
4. **Ceiling scripts** 0–10 with the per-question first-failing-stage table and funnel
   report; cost logged per run.
5. **Floor system** and the error-driven annotation tooling; `qampari-dev-clean-v1` frozen.
6. *(Optional)* entity-sparse index (Qdrant `entities` vector), neural linker cascade,
   graph-neighbor CSR.

---

## 4. External assessments and origin notes

Dated notes from the external conversations this doc was promoted from, recorded for
consolidation — not decisions. Related-work and dataset claims in quoted text are
unverified unless a citation is given; Danielle's prompts are logged verbatim in
`../danielle-inputs.md`. Literature lives in `../topics/reference/multi-answer-qa-literature.md`;
methodology in `../topics/reference/project-approach-principles.md`.

### Origin notes — moved from `topics/staging/maqa-oracle-ladder.md`

### 2026-08-16 — Danielle's goal (the seed)

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

### 2026-08-16 — The oracle ladder (response, near-verbatim, condensed)

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

### 2026-08-17 — Danielle's three-paper arc (program framing for MAQA)

Her own statement of the project, paraphrased from the prompt (verbatim in
`../danielle-inputs.md`):

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

### 2026-08-17 — A second, leaner version of the ladder

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

### 2026-08-17 — Problem definition, solution shape, impact hypotheses (Danielle)

Applying principle 1 of `../topics/reference/project-approach-principles.md`:

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

### Open questions (carried from staging)

- Confirm QAMPARI artifacts: released dump + chunked corpus, construction metadata (source
  entities/relations), the 200-question enriched subset, official metric scripts.
- Where the oracle ladder sits in the three-paper arc: it is the natural spine of paper 1
  (revisit + eval audit) and the measurement layer for paper 2 (clean brute-force baseline).
- Which graph for paper 1's reachability ceiling: anchor links (free) vs. string-matched
  mentions vs. the ReLiK-style linker from `../topics/reference/entity-linking-at-scale.md`.
- Relationship to the cleaner-dataset pipeline in the literature topic (verification-first
  gold sets) — same project or a follow-on.

- Bounded dataset-noise work for QAMPARI (per `../topics/reference/project-approach-principles.md`,
  2026-08-17): error-driven annotation of a small subset (baseline's top evidence that is not
  gold → hand-check; predicted-but-marked-wrong answers → hand-check), reported as noise-rate
  scalars with error bars rather than a "clean dataset"; a ~50-line answer normalizer with a
  test file of known-equivalent pairs. No sibling-dataset transfer slice during iteration.
  Sampling frame = union of generic baselines (BM25, dense, closed-book LLM), never the
  proposed method; system-blind judging; freeze and hash `qampari-dev-clean-v1` before the
  proposed method runs. Deliver the floor number first, ceilings in the same report.
- Kill criteria for the entity-centric shape (undecided); precision-control / stopping
  mechanism as an explicit impact axis.
## 5. Related work and positioning

*Purpose: the paper-facing synthesis — the prior-art landscape, this project's
position in it, and what each closest neighbor lacks. Unlike §4 (a dated intake
log, which grows by appending new entries **above this section**), §5 is a
current-state statement: rewrite it as understanding changes. Positioning claims
are Danielle's to make; agent-supplied literature claims anywhere in this document
are unverified leads, not established facts.*

**Status: raw material assembled from repository records (2026-08-24); positioning not
yet written.**

**Where the raw material lives:**

- `../topics/reference/multi-answer-qa-literature.md` — the primary literature record
  for this project: the QAMPARI/QUEST/RoMQA lineage and its successors, closed-book vs.
  retrieved vs. oracle result rows, the F1-5 capped-recall caveat, the
  dataset-cleaning/verification thread, and the 2026-08-17 second validation pass
  (GraphRAG lineage, MoNaCo date correction, "no published cleaned QAMPARI").
- `../refs/multi-answer-qa-state-of-research-2026.md` — the verbatim external deep-search
  report ("Exhaustive Multi-Answer Question Answering over Unstructured Corpora",
  2026-08-16) the accumulator distils, including a 16-entry annotated primary-source
  bibliography with venue links. Header states citations are unverified; it was produced
  by a browsing deep-search assistant, so treat every number as an agent claim.
- `../topics/reference/entity-linking-at-scale.md` — the linking flank: retriever–reader
  linkers, the "every mention" problem, the four-tier cascade, and the bake-off protocol
  behind MAQA-opt-2 (respondent's claims, unverified).
- `../topics/reference/project-approach-principles.md` — the methodology source for the
  bounded dataset-noise work and the error-driven-annotation sampling frame.
- `../danielle-inputs.md` ("MAQA Next Steps" intake, 2026-08-22) — Danielle's verbatim
  prompts behind the state-of-field and cleaner-datasets responses.
- `../litreview/citation-verification-ledger.md` — provenance ledger; it currently
  contains **no MAQA rows**, so nothing in the inventory below has been verified.

**Starting inventory for the synthesis** (all figures are the deep-search report's
unverified claims unless noted; detail in the accumulator and in §4):

- **The three original datasets** (report §2): QAMPARI (Amouyal et al., GeM 2023 — ≥5
  answers, ~13 average, Aug-2021 Wikipedia; best original system 32.8 F1, davinci-003
  closed-book 13.8, BM25 reader 18.8, oracle passage selection 62.4 dev); QUEST
  (Malaviya et al., ACL 2023 — 3,357 set-operation queries, mean complete recall@100 only
  0.142); RoMQA (Zhong et al., Findings EMNLP 2023 — BART+retrieval 63.8 F1 / 37.9 robust,
  gold evidence 95.0/83.4, GPT-3 open generation 4.4/0.4).
- **Coverage-and-verification successors** — the branch closest to the ladder's retrieval
  and evidence-budget rungs: Joint Passage Ranking (EMNLP 2021); LLatrieval (NAACL 2024,
  on ALCE-QAMPARI under the F1-5 cap); RI2VER (Dhole et al., Findings ACL 2025 —
  inter-passage verification; the cleanest closed-book-vs-corpus row on QAMPARI, 24.59 →
  40.70, and RoMQA 12.20 → 19.24); RVR (arXiv 2602.18425, 2026 — 68.70 ordinary Recall@100
  on QAMPARI but 33.70% *complete* recall). The accumulator states the objective has
  shifted to selecting passages whose union covers the answer set.
- **Closed-book, long-context, and logic flanks:** Mallen et al. (Findings NAACL 2024,
  knowledge-aware demo selection); LOFT (Findings NAACL 2025 — Gemini 1.5 Pro at 128K,
  QAMPARI 0.61 vs. 0.57, QUEST 0.30 vs. 0.54, with capped relevant docs); Does Dense
  Retrieval Understand Boolean Logic? (Findings EMNLP 2024); LOGICOL (EMNLP 2025);
  Reproducing Complex Set-Compositional IR (arXiv 2605.03824, SIGIR 2026 — ~0.42 R@100 on
  QUEST collapsing below 0.02 on LIMIT+).
- **Successor benchmarks:** MoNaCo (Wolfson et al., arXiv 2508.11133; TACL — the
  2026-08-17 pass calls it "arguably the flagship benchmark for exactly your setting" and
  corrects its date to 2025/2026, not the original era; recall 61–66% at 2–20 items →
  27.6% at 101–500 → 2.5% above 500; GPT-4o 48.98 closed-book / 37.28 BM25 top-20 / 58.67
  oracle; o3 fully correct on 38.7%); FanOutQA (ACL 2024); TANQ (TACL 2025); WideSearch
  (arXiv 2508.07999, ICLR 2026); plus ALCE (arXiv 2305.14627), AmbigQA, GRANOLA-QA,
  MulTiple, DeepAmbigQA, LIMIT+.
- **The eval-audit thread** (the accumulator's 2026-08-16 "Cleaner datasets" entry) —
  material for MAQA-1's noise audit and MAQA-opt-1: QAMPARI's own ExtendedSet study
  (200 questions, median +2 / mean +3.13 answers; NLI check removed 70% of co-occurrence
  false positives at 7.5% of correct alignments); DREAM/BRIDGE (arXiv 2602.06526, ICLR
  2026 — opposed-agent pooling, 95.2% labeling accuracy, 3.5% human review, 29,824
  recovered chunks); *Judging Is Not Enumerating* (arXiv 2608.01000 — membership judging
  ≈0.99 F1 vs. badly incomplete enumeration); ObliQA-MP (NLLP 2025 — 20.46% of accepted
  passages not connected); GaRAGe (Findings ACL 2025); answer-matching work (Kamalloo et
  al. ACL 2023, PEDANTS Findings EMNLP 2024, LongRecall arXiv 2508.15085); nugget work
  (The Great Nugget Recall arXiv 2504.15068 and a 2026 QAMPARI reproduction, GeM 2026);
  ExpertQA (NAACL 2024). The 2026-08-17 pass records that **no published cleaned
  QAMPARI/QUEST/RoMQA was found**.
- **The graph flank** (bearing on MAQA-opt-3): the 2026-08-17 pass says the entity-graph
  approach "is now GraphRAG" and names Microsoft GraphRAG, HippoRAG, LightRAG, GFM-RAG,
  HopRAG, RAPTOR, Think-on-Graph — with the caveat, recorded as a claim, that these
  evaluate on 2–4-hop single-answer benchmarks (HotpotQA, MuSiQue, 2Wiki) and that
  graph-based exhaustive retrieval on QAMPARI/QUEST/RoMQA/MoNaCo "appears genuinely
  underexplored"; the §4 feedback of 2026-08-17 repeats this as a favorable gap.
- **Linking prior art for MAQA-opt-2** (`entity-linking-at-scale.md`, unverified): ReLiK
  (Findings ACL 2024), ReFinED (NAACL Industry 2022), BELA (Meta, 97 languages), and
  `entity-linkings` (EACL 2026 demo) as the bake-off field.
- **Recorded evaluation hazards to engage:** the F1-5 recall cap (ALCE/LLatrieval),
  RoMQA P@10, LOFT's document caps, contamination of Aug-2021-Wikipedia-derived
  benchmarks, and the entanglement of retrieval and generation — the report's own
  prescription (report retrieval coverage, reader-given-gold, and final set accuracy
  separately) is the shape of the ladder in §1.

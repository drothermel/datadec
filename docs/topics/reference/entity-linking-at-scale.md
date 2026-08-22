# Entity linking at corpus scale — every mention, any corpus

**Kind:** reference — accumulator for exhaustive entity linking over large corpora (not just
Wikipedia's sparse first-mention links), corpus-agnostic, with the cost structure of modern
retriever–reader linkers.

Source: excerpts from the Notion page "MAQA Next Steps" (conversation dated 2026-08-16; intake
2026-08-22). System names, numbers, and repository claims are the respondent's and
**unverified**. Downstream use: `../../potential-projs/wiki-qa-sharding.md` (entity–page index).
---

## 2026-08-16 — Replacing string-match linking

**Danielle's context.** Her earlier pipeline acted on Wikipedia text chunks and linked
entities by string matching because (a) she wanted *every* mention on a page, not just the
first hyperlink, and (b) the approach had to "plausibly work with any arbitrary corpora
without Wikipedia's linking." String matching was "very not fun," though cheaper than running
an expensive model over all of Wikipedia. Is there a better option now?

**Response (near-verbatim, condensed).** "The best replacement is not a frontier LLM over
every mention. It is an end-to-end or retriever–reader entity linker that performs one
transformer pass per chunk, retrieves a short list of candidate entities from a pre-encoded
KB, and resolves all mentions jointly." Candidates: **ReLiK** and **ReFinED** (English),
**BELA** (multilingual). Caveat: off-the-shelf mention detectors inherit Wikipedia's sparse
linking convention, so combine neural detection with a high-recall lexical path and
within-document propagation.

*What changed.* Old: enumerate every plausible string → alias lookup → huge candidate sets →
heuristics. New: encode a chunk once → detect spans → retrieve candidates from a pre-encoded
KB → resolve all spans against them in one reader pass. Entity descriptions encoded once; no
cross-encoder pass per mention–candidate pair.

*Systems.* **ReLiK** (Findings ACL 2024; SapienzaNLP): retriever reads the whole chunk and
retrieves entities that might appear anywhere in it; reader consumes text + retrieved
entities and in one pass decides spans and identities; reported SOTA in/out-of-domain and up
to 40× faster inference; no supplied mention boundaries; context-sensitive candidates; can
retrieve entities whose exact alias is absent (abbreviations, nicknames); entity index
buildable from target-KB descriptions. **ReFinED** (NAACL Industry 2022; Amazon): mention
detection + fine-grained typing + disambiguation for all mentions in one forward pass; >60×
faster than contemporaries; links to Wikipedia or >30M Wikidata entities; fine-tuning
supported and recommended for domains with different mention conventions; risk: prepackaged
entity sets/descriptions lag the snapshot and adding entities is less convenient than the
zero-shot story — test the repo workflow before a full run. **BELA** (Meta; 97 languages):
mention detection, one passage-level encoding, kNN against an entity index, disambiguation,
NIL prediction; first multilingual comparison. Also **entity-linkings** (EACL 2026 demo;
NAIST): unified framework with interchangeable candidate retrievers/rerankers (prior- and
BM25-based indexes, trainable retrieval) — useful for a controlled bake-off.

*The "every mention" problem.* Wikipedia-derived training data links the first useful
mention, leaves repetitions unlinked, underlinks common concepts, never links pronouns, and
varies by editor — "a linker can have good benchmark scores while still missing repeated or
low-salience mentions." Use a union of three mention sources: (1) neural detection
(ReLiK/ReFinED); (2) recall-oriented lexical enumeration — keep the alias matcher but
demote it to proposing spans and candidates (aliases from titles, redirects, Wikidata
labels/aliases, anchor-text statistics, derived acronyms, punctuation/possessive
normalization, corpus-learned aliases; compiled multi-pattern matcher; false matches
acceptable at this stage); (3) within-document propagation in tiers — exact repeated
normalized surface (very high), known alias of a linked entity (high), compatible acronym
(medium), nominal/coreferential expression via a coreference model (lower), pronoun (keep
as a coreference edge, not an ordinary mention).

*Scalable cascade.* Tier 1 deterministic easy cases: alias prior P(e|m) = count(m→e) /
Σ count(m→e') ≈ 1, consistent with page context, not a common word, no strong competitor.
Tier 2 passage-level linker once per chunk in large GPU batches — cost per token/chunk, not
per string candidate. Tier 3 ambiguous-case resolver (stronger cross-encoder or an LLM) only
when the top two candidates are close, neural and lexical disagree, low-confidence NIL,
mention matters downstream, or rare entity with weak evidence — "a frontier model over
perhaps 0.1–1% of the hardest cases." Tier 4 document consistency (CPU): propagate
high-confidence links, reconcile conflicting IDs, page-level coherence, record unresolved
ambiguity rather than forcing an ID.

*Chunking.* Sections/paragraphs as units; ~256–512 model tokens; 50–100 token overlap;
prepend page title and section heading (`[DOCUMENT TITLE] … [SECTION] … [TEXT] …`) — "can
materially improve linking without a larger model"; dedupe predictions in overlaps; for
other corpora prepend whatever metadata exists.

*Entity index.* Per entity: canonical label, aliases, short description, lead paragraph,
instance-of types, notable type, country/language — not the full page. Two candidate
indexes: lexical alias index (`normalized alias → [(entity_id, prior, source)]`) and dense
entity-description index (Qdrant/LanceDB/Faiss; the linker's native index may be preferable
initially since its retriever was trained in that embedding space).

*Output more than one hard ID.* Mention table: document_id, chunk_id, start/end_char,
surface, entity_id, link_score, mention_score, nil_score, candidate_entity_ids,
candidate_scores, method, propagated_from, model_version, kb_snapshot. Page-level aggregate:
page_id, entity_id, mention_count, max/mean_confidence, first_position, linked_in_title,
linked_in_lead, distinct_surface_forms — so thresholds and the graph can be rebuilt without
relinking. Edge weight e.g. w(e,d) = log(1 + mention count) × confidence × positional
salience.

*Bake-off first.* 5,000–20,000 chunks: ReLiK small/base; ReFinED Wikipedia/Wikidata; BELA;
lexical aliases + priors; lexical + neural union. Evaluation set exhaustively annotated, not
inherited from hyperlinks. Measure separately: mention-detection recall; disambiguation
accuracy given correct span; end-to-end mention F1; repeated-mention recall; head vs. tail
accuracy; NIL P/R; candidate recall@k; cost and tokens/s; accuracy outside Wikipedia-style
prose.

*Likely implementation.* Lexical span proposals + ReLiK passage predictions +
already-linked document aliases → union + confidence calibration → ambiguous-case reranker →
document consistency pass → mention table + page/entity aggregate. "Its [string matching's]
ideal role is now inexpensive, high-recall span and candidate proposal; the neural model
handles context, ambiguity and NIL rejection, and a document-level pass fills in repeated
mentions that sparse Wikipedia supervision tends to miss."

# Structured output as a separate skill; tiny structurer models — reference topic

**Kind:** reference (standing accumulator). Entries are dated; paper characterizations are
the responding agent's and unverified unless marked. Danielle's interest: "using really
small LLMs as tiny focused components of larger systems," with structured output as the
core design question.

Why it matters here: the hypothesis that *adhering to a format* is a different skill from
*solving the task* is the same claim ELI makes about wrapper-only competence
(`../../potential-projs/elicitation-gain.md`) and IRT-10's BoolQ format intervention
(`../../potential-projs/irt-reanalysis.md`); this literature is its external evidence base.

---

## Undated (~mid-2026; intake 2026-08-22) — two questions on structured output for small models

**Danielle's prompt (verbatim core).** "I'm interested in using really small LLMs as tiny
focused components of larger systems. And one core question in that type of system design
is the question of structured output. Intuitively I suspect that the skill of adhering to a
very specific output format is a different skill than solving many specific tasks and that
it might be unnecessarily limiting to require that the same LLM does both task solving and
output structuring." (1) How does enforcing structured output affect task performance,
especially for older/smaller models? (2) How small can a model be whose sole job is to
structure information that is easily extractable from a text blob, and what are the
approaches? 2025–2026 papers, most recent preferred.

**Response (condensed).** "Structured output" is four things: asking for JSON, JSON mode,
schema-constrained decoding, and a schema-native trained extractor — with different
effects. Skill decomposition offered: semantic task skill → grounding (copy/normalize
values) → schema skill (which fields/arrays/nulls) → serialization (valid JSON/XML/YAML).
Serialization can be improved nearly independently but can interfere with the semantic
task when enforced at decode time.

*(1) Effect of enforcing structure.*
- "The Hidden Cost of Structure" (RANLP 2025; 11 models): base models often *benefit*
  from constraints; instruction-tuned models keep classification performance but degrade
  on generation; constraints push models off preferred NL continuations into
  lower-confidence structured ones; schema complexity and examples matter.
- "Quantifying the Impact of Structured Output Format on LLMs' Reasoning Performance"
  (EACL Findings 2026): effect is positive, negative, or neutral by instruction, reasoning
  type, and format — the question is per (model, task, schema, prompt).
- **Valid format ≠ correct content**: Structured Output Benchmark (2604.25359) —
  near-perfect schema compliance with much lower value accuracy; ExtractBench (2602.12247)
  — valid JSON while failing extraction; very broad schemas cause outright validity
  failures; provider structured-output modes reject complex schemas or reduce accuracy on
  deep ones.
- JSONSchemaBench (2501.10868): constrained-decoding frameworks improve adherence,
  sometimes downstream performance and speed; coverage varies by framework and schema.
- "When Correct Isn't Usable" (2605.02363): 7–9B models solve math but fail to emit
  usable JSON under naive prompting; constrained decoding fixes syntax at large latency
  cost and sometimes lower task performance.
- LLMStructBench (2602.14743; 22 models): prompting improves structural validity for
  small models but shifts errors into wrong values; no monotonic size → reliability
  relation; key omissions fade around mid-scale, wrong values remain the bottleneck.
- Clinical SLM extraction (2507.01810): JSON most parseable; targeted prompts help; some
  3–4B models reach high parseability; long documents degrade.
- **SLOT** (EMNLP Industry 2025): a lightweight fine-tuned *post-processing structurer* for
  other models' outputs — Llama-3.2-1B reaches high schema accuracy with SFT + constrained
  decoding; Mistral-7B 99.5% schema / 94.0% content similarity. The strongest published
  version of Danielle's design intuition.

*(2) How small can the structurer be.* No universal bound; depends on schema variability,
document length, value ambiguity, nulls, multilinguality, copy vs. infer. Evidence:
- Narrowest case may need no LLM: span tagger / NER / GLiNER-style extractor (GLiNER2,
  2507.18546) + code assembles JSON.
- ~0.5B: NuExtract-tiny-v1.5 (Qwen2.5-0.5B fine-tune, text + JSON template → JSON);
  NuExtract 2.0 at 2B/4B/8B, multimodal, typed fields.
- 1–2B as a realistic sweet spot: SLOT at 1B; ScrapeGraphAI-100k (2602.15189) — a 1.7B
  fine-tune narrows the gap to a large MoE reference on schema/key metrics; VAREX
  (2603.15118) — sub-4B models struggle more with compliance than extraction, but
  extraction-specific fine-tuning at 2B gives a large gain.
- 3–8B when schemas vary a lot, documents are long, nulls matter, multilingual/multimodal.
- Large deep enterprise schemas defeat frontier models unless decomposed.

*Approaches.* (a) Avoid generative JSON: text → span/value extractor → deterministic
assembler; require `{field, evidence span}` so values are verifiably in the source.
(b) Constrained decoding (Outlines, Guidance, XGrammar, llama.cpp grammars, provider
APIs) only for the final serialization step. (c) Train a formatter as a post-processor
(SLOT pattern: input = source + upstream free-form answer/candidates). (d) Distill from a
teacher: generate (blob, schema, JSON), validate against schema, source-grounding checks,
hard negatives (missing/ambiguous fields, distractors), fine-tune 0.5–2B, constrained
decoding at inference. (e) Schema-aware RL / verifier training: Schema Reinforcement
Learning (2502.18878), RL-Struct (2512.00319; dense rule-based rewards from the schema).
(f) Optimize the schema itself: key wording as an instruction channel (2604.14862);
PA-Tool (2510.07248) adapts tool/schema names to small models' pretrained patterns.
(g) Decompose large schemas: small field groups, shallow objects, one array type at a
time, per-sub-object validation and retry.

*Recommended architecture.* solver/extractor → tiny structurer → constrained decoder →
validator (schema, types, enums, nulls, source support) → deterministic merger. For pure
copy-extraction, start with a span extractor or 0.5–2B specialized extractor plus
evidence-per-field and JSON assembled in code.

**Intake notes.**
- Reading order the response gives: SLOT, Hidden Cost of Structure, LLMStructBench,
  ExtractBench, VAREX, ScrapeGraphAI-100k, NuExtract, Schema-RL / RL-Struct.
- Thirteen arXiv IDs added to the ledger (agent-supplied); several are 2602–2605 numbers,
  i.e. very recent — title–ID pairing is the thing to verify.
- The "base models benefit, instruction-tuned models degrade on generation" split (Hidden
  Cost of Structure) is a direct, testable prediction for ELI's pre/post-training axis and
  for IRT-10: DataDecide checkpoints are base models.
- "Valid format ≠ correct content" is the feasibility-waterfall distinction ELI already
  draws (format-only → parses → runs → passes); this literature supplies benchmarks that
  measure the two layers separately.
- Danielle's systems-design interest (tiny LLMs as focused components) has no project doc;
  it sits here until she says otherwise.

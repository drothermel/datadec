# Pretraining and midtraining toward a target task suite — literature reference

**Kind:** reference (accumulator for the "train the base model *for* a downstream suite"
literature: task-aware pretraining objectives, intermediate / continued pretraining,
midtraining and annealing data, and the midtraining ↔ post-training interplay). Entries
are dated. Characterizations are a SciSpace agent's; identifiers unverified. Siblings:
`pretraining-to-posttraining.md` (pretraining choices → post-training outcomes),
`schedules-and-annealing-literature.md` (the decay-phase mechanics),
`data-featurization-literature.md`.

Why it matters here: the apex question is how pretraining data shapes what post-training
can do; "midtraining to target a suite" is that question run forward as an intervention,
and it is the framing ANN's branch experiments, FUNC's stage-dependent data value, and
REC's annealing-data line all sit inside.

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-pretraining-task-analysis-agent-artifacts-zip_73fb24c0-fa4f-4ea3-b380-6dbe92b7e173_1787424452/` — the report (md + LaTeX/PDF), insight extraction, six
downloaded PDFs with cropped figures, and five search CSVs. **`INDEX.md` inside the folder
is the file-level index** and lists the canonical LM papers the review missed.

---

## 2026-08-22 — SciSpace deep review (undated, ~2026)

**Danielle's prompt (verbatim):**

> Please do a literature review about pretraining and midtraining to target a specific
> task or suite of downstream tasks.

**What survives from the report (LM-relevant only; the rest was graph prompting, vision
middleware, control transformers, ASR, and domain foundation models).**

- **Zhang et al. 2025, "On the Interplay of Pre-Training, Mid-Training, and RL on
  Reasoning Language Models" (2512.07783; PDF in the bundle).** Controlled study: under a
  fixed compute budget, midtraining on task-relevant data moves the model's competence
  boundary more efficiently than RL-only post-training when pretraining leaves headroom,
  and improves contextual generalization when pretraining was already sufficient.
  Practical reading: with limited compute, spend on midtraining data before extended RL.
  The directly relevant paper for the apex question.
- **Dery et al. 2021, "Should We Be Pre-training? An Argument for End-task Aware
  Training"** — jointly training the end task with auxiliary objectives during the
  intermediate phase beats task-agnostic continued pretraining, especially low-resource;
  gains depend on intermediate↔end-task alignment.
- **van der Goot 2023, MaChAmp at SemEval-2023** — intermediate training on a large
  uncurated multi-task collection gives broad modest gains; a well-matched single
  transfer task gives larger gains on its target. Curated-single vs. diverse-many is the
  operative tradeoff.
- **Qiu et al. 2021 (EMNLP), further pretraining for diverse dialogue tasks** — different
  downstream dialogue tasks want different further-pretraining objectives; selective, not
  universal, gains.
- **Gan et al. 2023, joint domain-specific pretraining with data enhancement** —
  reconstruct the continued-pretraining corpus around hard downstream examples (+5% avg
  over BERT on scientific NER/classification); costs generality.
- **Luo et al. 2021, meta-learning for downstream-aware pretraining** — put downstream
  task distribution signal into the pretraining objective to optimize for fast
  adaptation.
- **Task-robust minimax pretraining (2306.12070)** — minimize the worst-case risk over
  representative upstream tasks rather than the average; +1.8 GLUE avg, +9.2 CoLA;
  worst-case framing is a useful contrast for "target a suite" (optimize the min over the
  suite, not the mean).
- The report's synthesis: targeted objectives / midtraining / modular adaptation beat
  generic pretrain-then-finetune when aligned with the target suite, with gains from
  1–5% to 10–40%, paid for in compute, generality, and task-specificity.

**Intake notes.**

- The review answered a broader question than asked (any modality, any adaptation
  method) and missed the LM literature that the word "midtraining" now denotes: DAPT/TAPT
  (Gururangan et al. 2020), STILTs / intermediate-task transfer, the annealing-data line
  (MiniCPM, Llama 3, OLMo 2 / Dolmino, SmolLM, Nemotron), OctoThinker (2506.20512,
  midtraining that makes RL scale), Phi-style targeted synthetic pretraining, targeted
  data selection (DSIR, DoReMi, DsDm), and Blakeney et al. on late upweighting. Listed in
  the bundle `INDEX.md`; they are the related-work skeleton, not the report.
- Only one full text of real relevance was downloaded (2512.07783); the LaTeX version's
  figures are from the six downloaded papers, five of which are off-topic.
- All numbers as the agent reported; unverified.

# Topics — staging and reference docs outside the project list

Two subfolders make the kind visible in the tree: [`staging/`](staging/) holds promotion
candidates; [`reference/`](reference/) holds informational accumulators.

Two kinds of topic doc:

- **Staging** topics hold ideas without a project home and have the two exits described
  below (promoted to a project, or absorbed into project §4s; deleted either way). Any
  research idea qualifies — the collection is wide, not DataDecide-specific.
- **Reference** topics are standing accumulators — paper references and thoughts on a theme
  (plasticity, loss-curve forecasting, …) that many projects draw on. They are never
  "resolved"; they just grow, dated entry by dated entry.

Scratch notes collected from external conversations that do not clearly belong to an existing
project in [../potential-projs/](../potential-projs/) or to the program-level notes in its
README. Entries use the same convention as the project docs' §4: dated, attributed, quoted
close to verbatim, no decisions.

Unlike project docs, topic docs may reference projects freely — they are staging, not plans.

**Caveat on quoted related-work claims.** Statements in quoted external text about what does or
does not exist in the literature ("nobody has…", "this is unclaimed", specific paper
attributions) are unverified: the responding agents did not run literature searches and have no
reliable knowledge of the current landscape. Treat them as leads to check, not facts. The ideas
stand on their own.

**Danielle-flagged seeds.** On the Research Trajectory Notion page, `→` lines in a toggle's
title are Danielle's own flags for points she considers especially relevant to defining a
project. Wherever such a toggle is routed, those lines are quoted verbatim under a
"Danielle-flagged project seeds" heading near the top, and logged in `../danielle-inputs.md`.

**Routing rule for a new excerpt:** clearly about one existing project → that doc's §4;
clearly program-level → the potential-projs README; neither → a topic doc here (create on first
need).

**Lifecycle:** a topic doc has two exits — it matures into a new `potential-projs/` doc (same
three-section template, prefixed IDs, §4 seeded from its entries), or its entries are absorbed
into existing projects' §4s. Either way the topic file is deleted at that point, so this
directory only holds live, unresolved material.

## Staging (promotion candidates)

| Staging topic | What it is waiting on |
|---|---|
| [checkpoint-tomography.md](staging/checkpoint-tomography.md) | A prior-art pass (devinterp / WSD follow-up literature); then a decision on whether the five-probe battery (decay, hot, twin, data-shifted, reset) becomes the flagship instrument framing or is absorbed into annealed readouts, landscape geometry, token-level movement, functional featurization, and embedding-reset dynamics |
| [frozen-body-transfer-audit.md](staging/frozen-body-transfer-audit.md) | A forward-citation sweep of arXiv 2107.12460; then a promotion decision (gaps G5/G6). G6 is cross-listed as ICL-opt-5 in ICL elicitability |
| [clean-code-preference-icl.md](staging/clean-code-preference-icl.md) | Small practical test of adapting coding models to Danielle's clean-code preferences via in-context (and maybe lightweight) adaptation: tests + length ratio as automated feedback, model-generated cleaner/dirtier variants hand-labelled into preference pairs; possible workshop paper. Gate: Danielle runs a v0; prior-art pass on code-style preference datasets |

Promoted on 2026-08-22 (text moved into the new project docs' §4): `icl-as-posttraining` →
`../potential-projs/icl-elicitability.md`; `warmstarting-decomposition` +
`critical-period-timing-study` → `../potential-projs/intervention-grid.md`;
`movement-microscope` + `posttraining-experiment-design` →
`../potential-projs/movement-microscope.md` (with items also absorbed into tiny-scale
measurement and annealed readouts); the four reset topics →
`../potential-projs/embedding-reset-dynamics.md`; `reset-response-stage-probe` →
`checkpoint-tomography.md`.

## Reference (informational accumulators)

| Reference topic | Theme |
|---|---|
| [plasticity.md](reference/plasticity.md) | Continual-learning plasticity: papers, cheap training statistics, links to response vectors, modular plasticity, critical periods |
| [loss-curve-forecasting.md](reference/loss-curve-forecasting.md) | Multi-power law, loss→downstream-accuracy mappings, emergence-as-threshold caveat |
| [pretraining-to-posttraining.md](reference/pretraining-to-posttraining.md) | DataDecide; pretraining choices → post-training outcomes at matched loss; the "post-training did nothing" literature; hindsight reading of the earlier project |
| [icl-literature.md](reference/icl-literature.md) | Emergence of in-context learning as a function of pretraining data properties; ICL mechanism assumptions |
| [task-vectors.md](reference/task-vectors.md) | Weight-space task vectors / task arithmetic / merging (Ilharco; Zhou; Theseus; quantization) and activation-space ICL task vectors (Dong; Yang); the two senses and their gradient link |
| [landscape-literature.md](reference/landscape-literature.md) | River-valley picture and its measurements; linear mode connectivity, re-basin, feature connectivity, cross-task linearity, neuron identifiability; the comparability-across-basins precedent |
| [schedules-and-annealing-literature.md](reference/schedules-and-annealing-literature.md) | Stable-phase + decay-branch methodology (Hägele, MiniCPM, Llama 3 annealing, Blakeney), checkpoint merging (WSM, Nemotron 3), MPL correction, the cancellation and branch-length caveats |
| [data-featurization-literature.md](reference/data-featurization-literature.md) | Model-mediated (perplexity correlations, RegMix, DoReMi, mixing laws), similarity (Task2Vec, alignment, diversity coefficient), intrinsic (WIMBD, compression, Zipf/burstiness) feature families |
| [token-level-literature.md](reference/token-level-literature.md) | River/wall token mapping (Wen toy + Spearman), epistemic/aleatoric decomposition, Rho-1 loss-trajectory taxonomy, RLVR entropy / forking-token results |
| [critical-periods.md](reference/critical-periods.md) | Achille–Soatto critical learning periods and Information Plasticity (Fisher trace); how it anchors plasticity, warm-starting, basins, Task2Vec, and LLM data-placement results |
| [reinit-and-transfer-literature.md](reference/reinit-and-transfer-literature.md) | Embedding-reset / vocab-swap lineage (Artetxe, vocab-swap study, WECHSEL/FVT/FOCUS/ZeTT, tokenizer-change costs), frozen-transformer transfer, ITER; basin-preserving vs. basin-determining resets. **Danielle interest flag:** wants to know where this literature went and whether she can contribute — candidate for a targeted lit pass and possibly a staging topic |
| [multi-answer-qa-literature.md](reference/multi-answer-qa-literature.md) | Multi-answer (list) QA: QAMPARI / QUEST / RoMQA lineage, successors (MoNaCo, RI2VER, RVR, FanOutQA, …), closed-book vs. corpus vs. oracle results, F1-5 capped-recall caveat. **Danielle interest flag:** her ~2022 task; wants the current state and the frontier-model closed-book comparison |
| [retrieval-storage-tooling.md](reference/retrieval-storage-tooling.md) | Tooling comparisons for corpus storage + retrieval: DuckDB vs. LanceDB (query shapes, VSS limits, FTS, Lance extension, combined design), with pointers to the engine survey and Qdrant/LanceDB deep-dives |
| [entity-linking-at-scale.md](reference/entity-linking-at-scale.md) | Exhaustive, corpus-agnostic entity linking: retriever–reader linkers (ReLiK, ReFinED, BELA), the every-mention problem and three-source union, four-tier cascade, chunking, entity index, mention-table schema, bake-off protocol |
| [project-approach-principles.md](reference/project-approach-principles.md) | Danielle's four principles for starting a research project (problem/solution-shape/impact list; intuition first; scope dataset noise and make a clean set; single dataset with reproducibility + "why"), with feedback and two additions (kill criteria; cost on every plot); plus a 2025 note-taking workflow review and the seven-track record of that period |
| [grokking-and-hidden-progress.md](reference/grokking-and-hidden-progress.md) | Grokking (Power; Nanda progress measures), epoch-wise double descent (Nakkiran), grokking-under-non-stationarity, river-valley plateau reading; matched loss as necessary-but-insufficient |
| [identifiability-literature.md](reference/identifiability-literature.md) | CRL / nonlinear ICA identifiability (Schölkopf; Hyvärinen; iVAE), neuron identifiability, Platonic Representation Hypothesis; path-dependence ⇔ non-identifiability; interventions as identification |
| [evaluation-methodology-literature.md](reference/evaluation-methodology-literature.md) | Melis 2018, the hardware lottery, lifetime-tuning position paper, existence proofs as paradigm evidence; precedents for the research hypothesis |
| [moe-literature.md](reference/moe-literature.md) | FLAME-MoE, OLMoE router saturation, three phases of load balance, router robustness under continual pretraining, the Myth of Expert Specialization; expert permutation as a non-identifiable latent; MoE comparability warning |
| [irt-literature.md](reference/irt-literature.md) | IRT for NLP benchmarks (Lalor; Rodriguez; tinyBenchmarks; metabench); what structure in the model axis adds; local-independence and binary-vs-margin cautions |
| [workshop-deadlines.md](reference/workshop-deadlines.md) | Venue/timing accumulator: workshop-paper deadline waves for ML/RL/NLP conferences (NeurIPS ~Aug 29, AAAI-27 Nov 20, EACL Dec 15, COLING Jan 5, NAACL Feb 5), with ARR-commitment dates and the Sep 4 2026 ACL workshop-*proposal* deadline; dated, unverified, refresh per query |
| [generalization-and-ood-literature.md](reference/generalization-and-ood-literature.md) | Generalization formalisms across supervised/SSL (SSL bounds, MI frameworks, gap comparisons) and OOD measurement (DG survey, evaluation protocols, calibration, worst-case metrics); 2025-01 question sequence ending in "predict performance conditioned on the method" — the unanswered half that recipe-featurization / EDP now occupy. Citations unverified |
| [reference/nonstationarity-accounting.md](reference/nonstationarity-accounting.md) | The program's mechanism pillar: exogenous/endogenous non-stationarity sources and what each stabilizer suppresses |

# Topics — staging and reference docs outside the project list

Two kinds of topic doc:

- **Staging** topics hold ideas without a project home and have the two exits described
  below (promoted to a project, or absorbed into project §4s; deleted either way).
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

| Staging topic | What it is waiting on |
|---|---|
| [checkpoint-tomography.md](checkpoint-tomography.md) | A prior-art pass (devinterp / WSD follow-up literature) and a decision on whether the four-probe battery becomes the flagship instrument framing wrapping annealed readouts, landscape geometry, token-level movement, and functional featurization |
| [posttraining-experiment-design.md](posttraining-experiment-design.md) | Whether any of its designs (post-training power analysis / RL-ability proxy; "did SFT move the model in distribution space"; within-reach-task post-training; late-window cross-family intervention) becomes a project or is absorbed into TINY / ANN / TOK |
| [icl-as-posttraining.md](icl-as-posttraining.md) | Whether ICL curves on existing checkpoints (plus the code-autoencoder reconstruction probe) become a proxy-metric project, or fold into TINY / the post-training design topic |
| [warmstarting-decomposition.md](warmstarting-decomposition.md) | Whether the Ash & Adams reproduction + factorial becomes a project, now framed as one cell of a unified early-window intervention grid shared with the critical-period reproduction — "Chapter 1" and the retrospective's experimental spine; now carries a six-month one-harness plan with month-2/4/6 checkpoints |
| [critical-period-timing-study.md](critical-period-timing-study.md) | Whether the sibling-seeds-with-timed-deficits study (Fisher trace + barriers + ICL curves measured together) becomes a project or folds into GEO-opt-5 / warm-starting / ICL |
| [reset-recovery-dynamics.md](reset-recovery-dynamics.md) | Promotion decision (gaps G1/G2/G10): embedding-reset recovery curve vs. scale/stage/seed; input-vs-output asymmetry; init-in-the-limit |
| [interface-reset-basin-test.md](interface-reset-basin-test.md) | Promotion decision (gap G3): reset an interface, measure the barrier to the pre-reset solution — or add as GEO-opt-6 |
| [reset-and-plasticity.md](reset-and-plasticity.md) | Promotion decision (gaps G4/G9): does an interface reset restore plasticity; which layers need resetting; plasticity-injection diagnostic |
| [frozen-body-transfer-audit.md](frozen-body-transfer-audit.md) | A forward-citation sweep of arXiv 2107.12460, then a promotion decision (gaps G5/G6): LR-tuning re-audit with the reservoir null; elicitation-ceiling measurement of what a frozen interface can reach |
| [reset-response-stage-probe.md](reset-response-stage-probe.md) | Whether checkpoint tomography is promoted and adds this as its fifth probe (gap G7) |
| [reset-effects-many-seed-lm.md](reset-effects-many-seed-lm.md) | Whether the Zaidi-style many-seed / tuned-regularization replication stands alone or becomes a requirement in the other reset topics (gap G8) |
| [movement-microscope.md](movement-microscope.md) | Whether the four-stage post-training movement study (noise floor → calibrated sensitivity → decomposition → recipe movement profiles) becomes a project or folds into token-level movement / post-training design |
| [nonstationarity-accounting.md](nonstationarity-accounting.md) | Whether "non-stationarity accounting" becomes the thesis-level framing (see candidate program framings in the potential-projs README); its case study has moved to the MoE movement project |

Absorbed on 2026-08-21 into project docs: `moe-analysis-program` and
`moe-routing-as-data-instrument` → `moe-partitions.md` / `moe-movement.md`;
`moe-recipe-suite` → `moe-recipe-suite.md`; `small-scale-measurement-science` →
`tiny-scale-measurement.md`; `functional-featurization` → `functional-featurization.md`;
`beyond-datadecide-data-measurement` → `recipe-featurization.md` §4 and REC-11.

| Reference topic | Theme |
|---|---|
| [plasticity.md](plasticity.md) | Continual-learning plasticity: papers, cheap training statistics, links to response vectors, modular plasticity, critical periods |
| [loss-curve-forecasting.md](loss-curve-forecasting.md) | Multi-power law, loss→downstream-accuracy mappings, emergence-as-threshold caveat |
| [pretraining-to-posttraining.md](pretraining-to-posttraining.md) | DataDecide; pretraining choices → post-training outcomes at matched loss; the "post-training did nothing" literature; retrospective on the earlier project |
| [icl-literature.md](icl-literature.md) | Emergence of in-context learning as a function of pretraining data properties; ICL mechanism assumptions |
| [task-vectors.md](task-vectors.md) | Weight-space task vectors / task arithmetic / merging (Ilharco; Zhou; Theseus; quantization) and activation-space ICL task vectors (Dong; Yang); the two senses and their gradient link |
| [landscape-literature.md](landscape-literature.md) | River-valley picture and its measurements; linear mode connectivity, re-basin, feature connectivity, cross-task linearity, neuron identifiability; the comparability-across-basins precedent |
| [schedules-and-annealing-literature.md](schedules-and-annealing-literature.md) | Stable-phase + decay-branch methodology (Hägele, MiniCPM, Llama 3 annealing, Blakeney), checkpoint merging (WSM, Nemotron 3), MPL correction, the cancellation and branch-length caveats |
| [data-featurization-literature.md](data-featurization-literature.md) | Model-mediated (perplexity correlations, RegMix, DoReMi, mixing laws), similarity (Task2Vec, alignment, diversity coefficient), intrinsic (WIMBD, compression, Zipf/burstiness) feature families |
| [token-level-literature.md](token-level-literature.md) | River/wall token mapping (Wen toy + Spearman), epistemic/aleatoric decomposition, Rho-1 loss-trajectory taxonomy, RLVR entropy / forking-token results |
| [critical-periods.md](critical-periods.md) | Achille–Soatto critical learning periods and Information Plasticity (Fisher trace); how it anchors plasticity, warm-starting, basins, Task2Vec, and LLM data-placement results |
| [reinit-and-transfer-literature.md](reinit-and-transfer-literature.md) | Embedding-reset / vocab-swap lineage (Artetxe, vocab-swap study, WECHSEL/FVT/FOCUS/ZeTT, tokenizer-change costs), frozen-transformer transfer, ITER; basin-preserving vs. basin-determining resets. **Danielle interest flag:** wants to know where this literature went and whether she can contribute — candidate for a targeted lit pass and possibly a staging topic |
| [grokking-and-hidden-progress.md](grokking-and-hidden-progress.md) | Grokking (Power; Nanda progress measures), epoch-wise double descent (Nakkiran), grokking-under-non-stationarity, river-valley plateau reading; matched loss as necessary-but-insufficient |
| [identifiability-literature.md](identifiability-literature.md) | CRL / nonlinear ICA identifiability (Schölkopf; Hyvärinen; iVAE), neuron identifiability, Platonic Representation Hypothesis; path-dependence ⇔ non-identifiability; interventions as identification |
| [evaluation-methodology-literature.md](evaluation-methodology-literature.md) | Melis 2018, the hardware lottery, lifetime-tuning position paper, existence proofs as paradigm evidence; precedents for the research hypothesis |
| [moe-literature.md](moe-literature.md) | FLAME-MoE, OLMoE router saturation, three phases of load balance, router robustness under continual pretraining, the Myth of Expert Specialization; expert permutation as a non-identifiable latent; MoE comparability warning |

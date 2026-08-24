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
| [datadecide-dense.md](staging/datadecide-dense.md) | A small, heavily instrumented, many-seed retrain of a few recipes at the 2–4 smallest scales with cosine and WSD arms: restores checkpoint density where the released suite has 5–10 points, doubles as REC's order-effect arm, TINY's substrate, the LR/MPL ground-truth validation, and the WSD suite's pilot. Gate: a design doc and a decision to train |
| [rewritten-anneal-slice.md](staging/rewritten-anneal-slice.md) | Rewritten (SwallowCode / ProX / FinerWeb-style) vs. selected anneal data as a decay-branch arm: separates per-document quality from mixture shift in WSD-opt-4. Gate: verify the four lead papers; decide whether a rewritten slice is cheap enough at the smallest scales to be an arm |
| [model-behavioral-divergence.md](staging/model-behavioral-divergence.md) | "Cheap models aren't collapsed": behavioral non-collapse across providers measured on outcome signatures, tail risk, and success-conditioned solution-strategy diversity; implementation-ready Feb-2026 spec, never executed. Gate: Danielle decides it competes for time; if yes, the time-sensitive OpenRouter old-model snapshot run goes first |
| [whetstone-minigrid-env.md](staging/whetstone-minigrid-env.md) | Placeholder only (2026-08-24, by decision): the whetstone-envs minigrid env flagged as a candidate small project; spec-out deferred until Danielle initiates it |
| [pooled-dedup-code-benchmark.md](staging/pooled-dedup-code-benchmark.md) | Placeholder only (2026-08-24, by decision): pooling + deduplicating existing code benchmarks into a clean suite, per her lineage/overlap note's line of work 3 and 6-step pipeline; spec must satisfy benchmark-as-byproduct; spec-out deferred until Danielle initiates it |

Promoted on 2026-08-22 (text moved into the new project docs' §4): `icl-as-posttraining` →
`../potential-projs/icl-elicitability.md`; `warmstarting-decomposition` +
`critical-period-timing-study` → `../potential-projs/intervention-grid.md`;
`movement-microscope` + `posttraining-experiment-design` →
`../potential-projs/movement-microscope.md` (with items also absorbed into tiny-scale
measurement and annealed readouts); the four reset topics →
`../potential-projs/embedding-reset-dynamics.md`; `reset-response-stage-probe` →
`checkpoint-tomography.md`.

## Reference (informational accumulators)

Grouped by theme (2026-08-22); each topic sits in one group even when several programs draw on it.

### Program and process

| Reference topic | Theme |
|---|---|
| [project-approach-principles.md](reference/project-approach-principles.md) | Danielle's four principles for starting a research project (problem/solution-shape/impact list; intuition first; scope dataset noise and make a clean set; single dataset with reproducibility + "why"), with feedback and two additions (kill criteria; cost on every plot); plus a 2025 note-taking workflow review and the seven-track record of that period |
| [reference/nonstationarity-accounting.md](reference/nonstationarity-accounting.md) | The program's mechanism pillar: exogenous/endogenous non-stationarity sources and what each stabilizer suppresses |
| [workshop-deadlines.md](reference/workshop-deadlines.md) | Venue/timing accumulator: workshop-paper deadline waves for ML/RL/NLP conferences (NeurIPS ~Aug 29, AAAI-27 Nov 20, EACL Dec 15, COLING Jan 5, NAACL Feb 5), with ARR-commitment dates and the Sep 4 2026 ACL workshop-*proposal* deadline; dated, unverified, refresh per query |
| [experiment-tooling.md](reference/experiment-tooling.md) | Experiment-tracking / run-infrastructure tool comparisons for many-small-runs studies; qualitative per-example comparison workbench over pre-computed token probabilities (decision of record: marimo + Altair via `mo.ui.altair_chart`; Dash/Streamlit/Panel/Bokeh surveyed and superseded); holds Danielle's CNN-deconstruction (`deconCNN`) requirement statement — ablation ladder back through CNN history on CIFAR-10/ImageNet with seeds, sweeps, and landscape metrics — and a vendor-sourced logger comparison (Neptune, MLflow, ClearML, Sacred) |
| [retrieval-storage-tooling.md](reference/retrieval-storage-tooling.md) | Tooling comparisons for corpus storage + retrieval: DuckDB vs. LanceDB (query shapes, VSS limits, FTS, Lance extension, combined design), with pointers to the engine survey and Qdrant/LanceDB deep-dives |

### Evaluation, metrics, and forecasting

| Reference topic | Theme |
|---|---|
| [evaluation-methodology-literature.md](reference/evaluation-methodology-literature.md) | Melis 2018, the hardware lottery, lifetime-tuning position paper, existence proofs as paradigm evidence; precedents for the research hypothesis |
| [estimation-and-calibration-methods.md](reference/estimation-and-calibration-methods.md) | Ranking metrics when pairwise decision accuracy saturates (top-weighted τ / RBO / regret over NDCG-MAP for full permutations; stratify by rank gap); methods toolkit for per-cell estimates at fixed samples: estimand discipline, fractional vs. binary scores, dedup with multiplicities, stratification, control variates, shrinkage; interval-method comparison (analytic, bootstrap, Hoeffding/Bernstein, Wilson/Jeffreys, batch-level, confidence sequences); **conformal prediction and conformal risk control** with cheap predictors across docstrings/models — Danielle's cross-project tool flag |
| [small-scale-evaluation-metrics-literature.md](reference/small-scale-evaluation-metrics-literature.md) | Proxy-metric and downstream-forecasting literature seeded on Patel et al. 2026 (arXiv 2605.18607: 80 token-level proxies over expert trajectories; ρ 0.81 vs 0.36 for CE; DataDecide corpus ranking at 10⁻⁵ target compute; 18× training-time extrapolation on OLMo-3-7B) plus neural predictors, probes, SLM benchmarks, downstream scaling laws, mixture laws, emergence, contamination; intake note on v2's swapped bibliography entries and fabricated seed-paper author list (real: Patel, Reddy, Mosbach, Bahdanau); bundle `INDEX.md` on disk |
| [loss-alternative-metrics-literature.md](reference/loss-alternative-metrics-literature.md) | Loss-replacement evaluation metrics: token-selected / reweighted NLL (LongPPL, PPLqa), tokenization-independent likelihoods (bits per byte, marginal likelihood over tokenizations, token→character LM conversion), representation-side readouts (Diff-eRank, nuclear norm), compression scores; intake note that half the review is training objectives, one source is a crank, and Paloma / DataDecide continuous metrics / Patel proxies are missing; bundle `INDEX.md` on disk |
| [loss-curve-forecasting.md](reference/loss-curve-forecasting.md) | Multi-power law, loss→downstream-accuracy mappings, emergence-as-threshold caveat |
| [irt-literature.md](reference/irt-literature.md) | IRT for NLP benchmarks (Lalor; Rodriguez; tinyBenchmarks; metabench); what structure in the model axis adds; local-independence and binary-vs-margin cautions |
| [humanevalexplain-results.md](reference/humanevalexplain-results.md) | All published HumanEvalExplain (explain→regenerate) pass@1 results — only three papers (OctoPack, WaveCoder, Szalontai 2024), ~23 models × 6 languages; base models score 0; no paper reports explanation length, so the benchmark is a ready-made TLC-1 task set rather than a source for the correctness-vs-length plot; transcription-shift error in the WaveCoder GPT-4/WizardCoder rows flagged |
| [code-benchmarks-landscape.md](reference/code-benchmarks-landscape.md) | HumanEval derivative ecosystem (EvalPlus, EvoEval, HumanEval Pro, -X/-XL/MultiPL-E, HumanEvalPack, ReCode, ShortenDoc, HumanEvalComm, -V; no code→docstring flagship benchmark) and cross-benchmark overlap/dedup studies (How2Bench, LessLeak-Bench, HumanEval/MBPP contamination, ContextBench; three-matrix overlap scheme; within-benchmark redundancy unstudied); HumanEval/MBPP prompt-format conventions (original stub vs. instruct wrappers; MBPP already few-shot NL + asserts; harness variants); coding benchmark landscape from a SciSpace deep review (~30 survey/benchmark papers): task taxonomy, HumanEval/MBPP family and extensions, multilingual and domain-specific suites, 2025–26 directions (repo-level, evolution-aware, dynamic/contamination-free, LLM-generated, long-context, BEHELM), focus areas, cross-cutting weaknesses; intake note lists well-known benchmarks the review missed; bundle `INDEX.md` on disk |

### Data, recipes, and the DataDecide suite

| Reference topic | Theme |
|---|---|
| [datadecide-data-pipeline.md](reference/datadecide-data-pipeline.md) | External readings of the `datadec` data layer against the planning docs: what is built and verified (instance/choice tables, denormalized checkpoint derivations, 2026-08-19 validation, HF publishing), the LR-derivation caveat, and the still-missing analysis-side pieces (coverage census, checkpoint-spacing stats, REC-a manifest module, response-matrix builder); claims checked against the repo |
| [data-featurization-literature.md](reference/data-featurization-literature.md) | Model-mediated (perplexity correlations, RegMix, DoReMi, mixing laws), similarity (Task2Vec, alignment, diversity coefficient), intrinsic (WIMBD, compression, Zipf/burstiness) feature families |
| [synthetic-data-literature.md](reference/synthetic-data-literature.md) | Synthetic / rephrased training data: scaling (BeyondWeb, data-constrained scaling, quality parameter Q, diversity), rewriting and instruction synthesis (WRAP, Self-Instruct, GLAN, MAmmoTH2, surveys), model collapse and its avoidance (Tale of Tails, verification, synthesizing without collapse); the SciSpace LaTeX review is the deliverable, the pasted per-paper summaries cover 4 of 14 papers; bundle `INDEX.md` with arXiv→title map on disk |
| [targeted-pretraining-midtraining-literature.md](reference/targeted-pretraining-midtraining-literature.md) | Pretraining / midtraining toward a target task suite: the interplay of pre-, mid-training and RL (2512.07783), end-task-aware training, curated vs. uncurated intermediate training, task-robust minimax pretraining; intake note that the SciSpace review drifted across modalities and missed the LM midtraining/annealing-data canon (DAPT/TAPT, OLMo 2 Dolmino, OctoThinker, DSIR/DoReMi); bundle `INDEX.md` on disk |
| [pretraining-to-posttraining.md](reference/pretraining-to-posttraining.md) | DataDecide; pretraining choices → post-training outcomes at matched loss; the "post-training did nothing" literature; hindsight reading of the earlier project |
| [schedules-and-annealing-literature.md](reference/schedules-and-annealing-literature.md) | Stable-phase + decay-branch methodology (Hägele, MiniCPM, Llama 3 annealing, Blakeney), checkpoint merging (WSM, Nemotron 3), MPL correction, the cancellation and branch-length caveats |

### Training dynamics and mechanism

| Reference topic | Theme |
|---|---|
| [landscape-literature.md](reference/landscape-literature.md) | River-valley picture and its measurements; linear mode connectivity, re-basin, feature connectivity, cross-task linearity, neuron identifiability; the comparability-across-basins precedent |
| [token-level-literature.md](reference/token-level-literature.md) | River/wall token mapping (Wen toy + Spearman), epistemic/aleatoric decomposition, Rho-1 loss-trajectory taxonomy, RLVR entropy / forking-token results |
| [plasticity.md](reference/plasticity.md) | Continual-learning plasticity: papers, cheap training statistics, links to response vectors, modular plasticity, critical periods |
| [critical-periods.md](reference/critical-periods.md) | Achille–Soatto critical learning periods and Information Plasticity (Fisher trace); how it anchors plasticity, warm-starting, basins, Task2Vec, and LLM data-placement results |
| [grokking-and-hidden-progress.md](reference/grokking-and-hidden-progress.md) | Grokking (Power; Nanda progress measures), epoch-wise double descent (Nakkiran), grokking-under-non-stationarity, river-valley plateau reading; matched loss as necessary-but-insufficient |
| [identifiability-literature.md](reference/identifiability-literature.md) | CRL / nonlinear ICA identifiability (Schölkopf; Hyvärinen; iVAE), neuron identifiability, Platonic Representation Hypothesis; path-dependence ⇔ non-identifiability; interventions as identification |
| [reinit-and-transfer-literature.md](reference/reinit-and-transfer-literature.md) | Embedding-reset / vocab-swap lineage (Artetxe, vocab-swap study, WECHSEL/FVT/FOCUS/ZeTT, tokenizer-change costs), frozen-transformer transfer, ITER; basin-preserving vs. basin-determining resets. **Danielle interest flag:** wants to know where this literature went and whether she can contribute — candidate for a targeted lit pass and possibly a staging topic |
| [generalization-and-ood-literature.md](reference/generalization-and-ood-literature.md) | Generalization formalisms across supervised/SSL (SSL bounds, MI frameworks, gap comparisons) and OOD measurement (DG survey, evaluation protocols, calibration, worst-case metrics); 2025-01 question sequence ending in "predict performance conditioned on the method" — the unanswered half that recipe-featurization / EDP now occupy. Citations unverified |
| [world-models-literature.md](reference/world-models-literature.md) | LLMs as text-based world models (From Word to World: fidelity/scaling/agent-utility framework; structured saturates ~20K trajectories, open-ended non-saturating) and the no-model-free-shortcut theory (Richens: general agents provably contain extractable world models; myopic agents don't); model-based flank of the LLM-in-classic-RL thread; minigrid spec-out background |
| [parametrization-and-hp-transfer.md](reference/parametrization-and-hp-transfer.md) | µP-family parametrizations, width/depth hyperparameter transfer, low-precision numerics; u-µP (2407.17465) summary + Figure 1 read — independent HP search (9 vs. 339 runs), embedding-LR rule 1/√fan-out, cast-only FP8; DataDecide is not µP-parametrized (cross-size LR confound) |
| [ntk-literature.md](reference/ntk-literature.md) | Empirical NTK as a dynamics readout: definition, gradient-flow propagator, spectrum-as-learning-modes, condition number, eNTK evolution = feature learning, computation via JVPs; link to the interactive Perplexity app; intake note on what the tutorial omits for measurement use (early kernel motion, multi-output shape, sampling, kernel–target alignment) and candidate readouts for GEO / the CNN ladder |
| [task-vectors.md](reference/task-vectors.md) | Weight-space task vectors / task arithmetic / merging (Ilharco; Zhou; Theseus; quantization) and activation-space ICL task vectors (Dong; Yang); the two senses and their gradient link |
| [icl-literature.md](reference/icl-literature.md) | Emergence of in-context learning as a function of pretraining data properties; ICL mechanism assumptions |

### Objectives, distillation, and regularization

| Reference topic | Theme |
|---|---|
| [training-objective-alternatives-literature.md](reference/training-objective-alternatives-literature.md) | Training-side CE alternatives: token reweighting / selection (Rho-1, MiLe, ESLM, TALR, VCORE, Velocitune), probability families and proper scoring rules ("Beyond Log Likelihood" capability continuum, MixCE, Brier/spherical, focal), beyond-next-token (multi-token, patch-level, concept-level, UL2/SpacTor, LLM-JEPA); RLHF-adjacent leakage removed; missing canon listed; bundle `INDEX.md` on disk |
| [distillation-literature.md](reference/distillation-literature.md) | LLM distillation against Danielle's six questions (teacher/student scaling; forward vs. reverse KL and loss combination; logit vs. token repetition; KD vs. scratch; pre- vs. post-trained teachers; sequential vs. direct workflows); evidence quality per question noted — distillation scaling laws (2502.08606), on-policy GKD, DistiLLM missing; question 6 has no direct evidence; bundle `INDEX.md` on disk |
| [regularization-literature.md](reference/regularization-literature.md) | Regularizers for transformer / MoE LMs on repeated data: general (dropout, weight decay, flooding, augmentation, weight/spectral norm, R-Drop), transformer-specific (LayerDrop, UniDrop, relaxed attention), MoE-specific (load balancing, expert/gating dropout, Dirichlet-prior router shaping, z-loss), dedup/reweighting; intake note that the review missed the token-crisis multi-epoch paper (2305.13230: dropout works, MoE overfits repeats more) and Switch/ST-MoE expert dropout; bundle `INDEX.md` on disk |

### Architecture

| Reference topic | Theme |
|---|---|
| [moe-literature.md](reference/moe-literature.md) | Ensemble→MoE design space (routing unit / sharing / selection; placement table; 13-paper 2022+ reading list); FLAME-MoE, OLMoE router saturation, three phases of load balance, router robustness under continual pretraining, the Myth of Expert Specialization; expert permutation as a non-identifiable latent; MoE comparability warning |
| [layer-looping-literature.md](reference/layer-looping-literature.md) | Layer looping / recursive depth / cross-layer weight tying (Universal Transformers, ALBERT → Relaxed Recursive Transformers, Sparse UT, MoEUT, dynamic layer tying, retrofitted recurrence, recurrent-depth latent reasoning; programmable-computer and in-context-GD theory; adaptive halting); no bundle; intake note on broken citation numbering and the missing 2025 LLM-scale canon (Mixture-of-Recursions, Saunshi et al., Ouro) |
| [nas-literature.md](reference/nas-literature.md) | NAS state-of-field survey (search space / strategy / performance estimation; training-free and zero-shot proxies; benchmarks NAS-Bench-360, NATS-Bench; LLM-driven and carbon-efficient NAS). Kept for the performance-estimation half: zero-cost proxies (TE-NAS via NTK condition number, NASWOT, Abdelfattah et al.) and learning-curve extrapolation (Domhan; Baker) are the prior-art line EDP must position against; unverified |

### Code, latents, and the TLC / ELI program

| Reference topic | Theme |
|---|---|
| [structured-output-literature.md](reference/structured-output-literature.md) | Format adherence as a skill distinct from task solving: constrained-decoding effects by model type (Hidden Cost of Structure; EACL 2026), valid-format-≠-correct-content benchmarks (Structured Output Benchmark, ExtractBench, LLMStructBench, JSONSchemaBench, VAREX), tiny structurers (SLOT 1B post-processor, NuExtract 0.5B–8B, GLiNER2), approaches (span-extract + assemble, constrained decoding last, distillation, schema RL, schema wording, decomposition); Danielle's tiny-components-of-systems interest |
| [nl-bottleneck-prior-art.md](reference/nl-bottleneck-prior-art.md) | TLC prior-art record: Danielle's search brief, the Dec-2025 "appears novel" verdict and Jan-2026 update (six-component rubric; Nano-Capsulator / EPiC / RLPrompt / Latent Programmer / Sentence Bottleneck / ICAE partials), the ICBINB literature-grounding pass (per-stage citations, baseline prompts, optimizer meta-prompt); pointer to the 100-file bundle and its `INDEX.md`; GenDLN flagged for a human read; identifier slips listed |
| [code-compression-literature.md](reference/code-compression-literature.md) | Code-compression landscape from a SciSpace deep review: the six senses of "code compression" in use (learned embeddings, prompt/context compression, model compression, LLM entropy coding, library learning, semantic compression), Girish rate–distortion theory, standard comparisons and baselines; records that Maveli–Vergari–Cohen "Can LLMs Compress (and Decompress)?" is a title-level false alarm (LLMs predicting lossless compressors; Danielle read it); bundle `INDEX.md` on disk |
| [prompt-compression-and-optimization-literature.md](reference/prompt-compression-and-optimization-literature.md) | PCRL (RL token-level extractive compression), Nano-Capsulator (NL capsule prompts, utility reward × semantic loss), EPiC (evolutionary prompt search for code with the ATSP cost metric); prior art for TLC's compression objective and ELI's budgeted outer optimizer; pointer to the SciSpace search archives on disk |
| [prompt-optimization-landscape.md](reference/prompt-optimization-landscape.md) | Automatic prompt optimization landscape from a SciSpace deep review: method families (evolutionary, meta-prompting with critics, gradient-inspired textual updates, learned per-query generators, hybrids), system-level optimization (SPRIG, Trace/OptoPrime, LLM-AutoDiff, RePrompt, compound-AI survey target list), execution-driven code methods, feedback taxonomy, open problems; intake note: canonical anchors (APE, OPRO, ProTeGi, TextGrad, DSPy/MIPROv2, GEPA) missing from the review; bundle `INDEX.md` on disk |
| [code-feature-extraction-tooling.md](reference/code-feature-extraction-tooling.md) | Named-feature extraction from Python source (stdlib `tokenize`/`ast`/`symtable`, LibCST/parso/tree-sitter, astroid/Pyright/mypy/Jedi, Ruff/Semgrep/CodeQL/Joern, cloc/Tokei/Radon/ctags; type-inference datasets) from a 2026-07-03 deep-research run against Danielle's Report-2 brief; intake note: the report violated the brief's no-recommendation/no-workflow rules and omitted the exclusion list, matrix accounting, and search log — seed inventory only, unverified |

### Multi-answer QA and retrieval (MAQA / SHARD)

| Reference topic | Theme |
|---|---|
| [multi-answer-qa-literature.md](reference/multi-answer-qa-literature.md) | Multi-answer (list) QA: QAMPARI / QUEST / RoMQA lineage, successors (MoNaCo, RI2VER, RVR, FanOutQA, …), closed-book vs. corpus vs. oracle results, F1-5 capped-recall caveat. **Danielle interest flag:** her ~2022 task; wants the current state and the frontier-model closed-book comparison |
| [entity-linking-at-scale.md](reference/entity-linking-at-scale.md) | Exhaustive, corpus-agnostic entity linking: retriever–reader linkers (ReLiK, ReFinED, BELA), the every-mention problem and three-source union, four-tier cascade, chunking, entity index, mention-table schema, bake-off protocol |

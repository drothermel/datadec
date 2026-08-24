# elicitation gain — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`elicitation-gain.md`](../elicitation-gain.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*Highest-recall corpus for `ELI` (ELI-1–ELI-3, ELI-opt-1–ELI-opt-4). Every item is on
record in this repository. Most of this cluster arrived via SciSpace deep reviews, agent
summaries, or Feb-2026 novelty checks — those are **agent-generated and unverified**, and
said so in-line. `docs/litreview/citation-verification-ledger.md` marks ~40 rows as
feeding TLC/ELI and states that nothing in it is verified. No positioning claims are made.*

**System-level prompt / interface optimization — the §4 entry calls this "the outer loop's
positioning set"**

- **Trace / OptoPrime** (arXiv 2406.16218) — workflows as graphs; an LLM optimizer over
  heterogeneous parameters including prompts, hyperparameters, and code; the structural
  match to ELI's wrapper θ — (source:
  docs/topics/reference/prompt-optimization-landscape.md; SciSpace review, unverified;
  ledgered Feeds=TLC,ELI).
- **LLM-AutoDiff** (arXiv 2501.16673) — textual gradients through multi-stage pipelines;
  the multi-component analogue of ELI's staged prompting — (source: same; unverified).
- **SPRIG** (arXiv 2410.14826) — system-prompt components by edit-based genetic search; one
  optimized system prompt ≈ task-specific prompts across 47 tasks; the transfer claim
  ELI-opt-1's wrapper-transfer matrix tests in the size direction — (source: same;
  unverified).
- **RePrompt** (arXiv 2406.11132) — agent planning instructions refined from interaction
  trajectories — (source: same; unverified).
- **Lin et al., survey of LLM-based optimization of compound AI systems** (arXiv 2410.16392)
  — its target list (prompt components, sampling parameters, retrieval configs, tool specs,
  orchestration logic) is called "the ELI action space" — (source: same; unverified).

**Broader automatic-prompt-optimization families (same SciSpace review; all unverified)**

- **Promptbreeder** (2309.16797) — self-referential evolution that evolves the mutation
  prompts too; the review's foundational anchor — (source:
  docs/topics/reference/prompt-optimization-landscape.md).
- **PhaseEvo** (2402.11347) and **AMPO** (2410.08696) — evolutionary/population branch.
- **PromptWizard** (2405.18369) — meta-prompting / generation-refinement with a critic.
- **Tang et al., "LLMs as prompt optimizers ≈ gradient optimizers"** (2402.17564) — update
  direction + update method decomposition.
- **Learning from Contrastive Prompts** (2409.15199) — success-vs-failure contrastive
  feedback, one of the review's four feedback types.
- **Dual-Phase accelerated prompt optimization** (2406.13443).
- **QPO** (2408.10504) — query-dependent learned prompt generators via offline RL.
- **Davari et al.** (2507.09839) — reinforcement + diversification + migration for black-box
  LLMs.
- **Wan et al., "Teach better or show smarter"** (2406.15708) — joint instruction × exemplar
  optimization as a settled finding; directly relevant to whether ELI's wrapper should
  co-optimize demonstrations.
- **Feedback taxonomy on record** (validation accuracy; LLM-as-critic; execution feedback;
  contrastive) — ELI's verifier waterfall is the execution-feedback branch — (source: same).
- **Open problems the review lists** — evaluation-protocol inconsistency, cross-model
  transfer of optimized prompts, sample efficiency, validation-set overfitting, prompt
  brittleness, prompt length vs. performance, integration with fine-tuning — (source: same).

**Missing canon recovered later (each with the record's provenance flag)**

- **OPRO, *Large Language Models as Optimizers*** (2309.03409) — arrived with an ID in the
  2026-08-23 re-eval conversation; also cited in the Feb-2026 Claude novelty check —
  (source: docs/topics/reference/prompt-optimization-landscape.md; unverified).
- ***The Limitations of Small-Scale LLMs as Optimizers*** ("Revisiting OPRO"; 2405.10276;
  ACL Findings 2024) — small open models (LLaMA-2, Mistral 7B) are weak *as optimizers*,
  with strong direct-instruction baselines recommended in that regime. Bears directly on
  ELI's choice of a single fixed outer model — (source: same; Danielle named the title, the
  ID is the respondent's; unverified).
- **LEO** (2403.02054) — LLM-based evolutionary optimizer with elitism and
  anti-hallucination guardrails — (source: same; unverified).
- **GEPA, *Reflective Prompt Evolution Can Outperform Reinforcement Learning*** (2507.19457)
  — samples trajectories, reflects on failures in text, proposes prompt edits, combines
  lessons across a Pareto frontier; in DSPy the metric may return a scalar *plus textual
  feedback* at whole-program or predictor level. The nearest published shape of ELI's
  critique channel — (source: docs/topics/reference/prompt-optimization-landscape.md;
  docs/topics/reference/estimation-and-calibration-methods.md:276; unverified).
- **EvoPrompt** (2309.08532; Guo et al. — distinct from EvoPrompting 2302.14838, Chen et al.)
  — surfaced via the 2026-02-04 Claude novelty check as component-feasibility evidence for
  gradient-free prompt optimization — (source:
  docs/topics/reference/prompt-optimization-landscape.md;
  docs/topics/reference/nl-bottleneck-prior-art.md; agent-generated novelty check,
  unverified).
- **PromptBridge** (2512.01420) — cross-model prompt transfer; the published analogue of
  ELI-opt-1's swapped-executor / transfer matrix — (source: same; unverified).
- **APO systematic survey** (2502.16923) and **automatic prompt engineering survey**
  (2502.11560) — offered as scaffolding for a fair comparison protocol — (source: same).
- **Still-missing anchors named in the record but never supplied with IDs: APE, ProTeGi/APO,
  TextGrad, DSPy/MIPROv2, RLPrompt** — (source:
  docs/topics/reference/prompt-optimization-landscape.md intake notes;
  docs/litreview/tlc-litreview-plan.md row B).
- **MCTS-OPS** (2508.05995), **RL4QE** (no ID), **SelfEvolve** (2306.02907) — the
  LLM-as-optimizer items from the ICBINB grounding pass — (source:
  docs/topics/reference/nl-bottleneck-prior-art.md;
  docs/litreview/tlc-litreview-plan.md row B).

**Budgeted, gradient-free optimization against black-box executors (the accounting problem)**

- **EPiC** (Saluja et al. 2024, arXiv 2408.11198) — evolutionary prompt optimization for
  code generation with explicit cost accounting: initial-evaluation phase, LLM-built
  population, fitness = test pass ratio, fitness-weighted selection, LLM-mutator vs. cheap
  WordNet/GloVe synonym mutator (**the cheap one wins on cost** — a design data point for
  ELI's outer loop), population 5–8; new metric **ATSP** = additional tokens per solved
  problem, reported 20k vs. Reflexion 38k / LDB 196k / LATS 275k on HumanEval+. §4 calls it
  "the nearest published outer-optimizer with budget accounting" and the closest analog of
  ΔS-at-budget — (source:
  docs/topics/reference/prompt-compression-and-optimization-literature.md;
  docs/potential-projs/elicitation-gain.md §4 2026-08-22; SciSpace summary, unverified; the
  ledger also records an identifier slip, EPiC given as 2410.14321 elsewhere).
- **PCRL** (Jung & Kim 2024, IEEE Access, DOI 10.1109/ACCESS.2024.3403426) — discrete
  token-level prompt compression as sequence labeling; policy on frozen DistilBERT trained
  by SCST; **no gradient access to the target LM and no labels, so black-box-API
  compatible**; policies trained on small LMs transfer to LLaMA-2-7B, Falcon-7B,
  FLAN-T5-XXL, GPT-3.5-Turbo. Evidence that discrete gradient-free interface optimization
  against black-box executors works, and that optimizers fit on small models transfer —
  (source: docs/topics/reference/prompt-compression-and-optimization-literature.md;
  unverified).
- **Nano-Capsulator** (Zhou/Chuang et al. 2024, arXiv 2402.18700) — abstractive NL "Capsule
  Prompts" transferring across LLMs including APIs; semantic-preservation loss × utility
  reward with a hard length cutoff. The abstractive counterpart to PCRL's extractive
  compression — (source: same; first-author attribution disputed in the record; unverified).
- **Prochemy** (2503.11085) — execution-driven prompt refinement, the verifiable/code branch
  — (source: docs/topics/reference/prompt-optimization-landscape.md; unverified).
- **The 21-paper Undermind prompt-compression map** (2026-02-04 novelty check;
  agent-generated, unverified): RL/gradient-free subset **PCRL, TACO-RL, PIS, GPT-C,
  Cmprsr (GRPO), LLM-DCP, LanguaShrink**; training-free search **Style-Compress, DSPC,
  SCOPE, PartPrompt, EHPC, AttnComp**; foundational/adjacent **Selective Context
  (2310.06201), LongLLMLingua (2310.06839), Fei et al. 2312.09571, LLMLingua-2, R2C, CPC,
  *Fundamental Limits of Prompt Compression: A Rate-Distortion Framework for Black-Box
  LLMs*, CompressionAttack** — relevant to ELI as the population of gradient-free
  interface-editing methods with budget/length as the controlled resource — (source:
  docs/topics/reference/prompt-compression-and-optimization-literature.md 2026-08-24 entry).
- **Girish et al., rate–distortion framework for black-box prompt compression** (2407.15504)
  — the formal length-vs-performance frontier; possibly the same paper as the item above,
  flagged "confirm same paper at verification" — (source:
  docs/litreview/tlc-litreview-plan.md row C;
  docs/litreview/citation-verification-ledger.md).

**Wrapper-only competence — the named precedent**

- **AlphaCodium** (arXiv 2401.08500) — multi-stage, test-based "flow engineering" raising
  pass@k with **no weight change**; §4's closing-summary calls it "the named wrapper-only
  precedent" and the concrete exemplar of wrapper-only mattering for code. The ledger notes
  a competing ID 2401.19489, resolved to 2401.08500 — (source:
  docs/potential-projs/elicitation-gain.md §4;
  docs/topics/reference/nl-bottleneck-prior-art.md;
  docs/litreview/tlc-litreview-plan.md rows D/E; agent-supplied, unverified).
- **AceCoder** (DOI 10.1145/3675395), **Structured CoT**, **Self-Planning** (2303.06689),
  **Tree of Thoughts** (2305.10601), **CodeT** (2207.10397 — code generation with generated
  tests) — the direct-generation / staging / self-verification wrapper family from the
  ICBINB grounding pass and the Feb-2026 novelty check — (source:
  docs/topics/reference/nl-bottleneck-prior-art.md; agent-generated, unverified).
- **Misu et al. 2024, Dafny reconstruction** (DOI 10.1145/3643763; three prompt styles) —
  prompt-condition-as-variable precedent — (source:
  docs/litreview/tlc-litreview-plan.md row D).
- **GenDLN** (ACL SRW 2025, DOI 10.18653/v1/2025.acl-srw.92) — evolutionary *joint* prompt
  optimization over *stacked frozen LLMs*; flagged as needing Danielle's own read; the
  stacked-frozen-LLM shape is structurally an outer optimizer over a multi-model interface —
  (source: docs/topics/reference/nl-bottleneck-prior-art.md;
  docs/litreview/tlc-litreview-plan.md gate item 1).
- **Language Bottleneck Models** (Berthon & van der Schaar, arXiv 2506.16982) — optimizable
  encoder prompt → short NL summary → *frozen* LLM decoder, encoder trained by
  group-relative policy optimization to maximize frozen-decoder performance; Danielle's read:
  prior work and the nearest named framework, cite in §2 — (source:
  docs/topics/reference/nl-bottleneck-prior-art.md:121).
- **Proto-tokens** (Kuratov et al., ACL 2025, 2502.13063) — frozen-LLM reconstruction of
  ~1.5k tokens from 1–2 trained embeddings; the extreme point of "how few tuned parameters
  does an interface need" — (source: docs/topics/reference/nl-bottleneck-prior-art.md).

**Structured output as a skill distinct from task solving (external evidence for the
premise; all agent-characterized and unverified; thirteen IDs ledgered ELI/IRT)**

- **"The Hidden Cost of Structure"** (RANLP 2025; 11 models; no arXiv ID) — **base models
  often *benefit* from constrained decoding while instruction-tuned models degrade on
  generation**; recorded as a direct, testable prediction for ELI's pre/post axis, since
  DataDecide checkpoints are base models — (source:
  docs/topics/reference/structured-output-literature.md;
  docs/potential-projs/elicitation-gain.md §4 2026-08-22).
- **"Quantifying the Impact of Structured Output Format on LLMs' Reasoning Performance"**
  (EACL Findings 2026; no ID) — the effect is positive, negative, or neutral per (model,
  task, schema, prompt) — (source: docs/topics/reference/structured-output-literature.md).
- **SLOT** (EMNLP Industry 2025; no ID) — a lightweight fine-tuned **post-processing
  structurer**: Llama-3.2-1B reaches high schema accuracy; Mistral-7B 99.5% schema / 94.0%
  content similarity. Named "the strongest published version of Danielle's design
  intuition," and a candidate wrapper class for ELI-1 in which the *wrapper is itself a tiny
  model* rather than a prompt — (source: same).
- **Structured Output Benchmark** (2604.25359) — near-perfect schema compliance with much
  lower value accuracy; the format-vs-content split the feasibility waterfall encodes.
- **ExtractBench** (2602.12247) — valid JSON while failing extraction; broad schemas cause
  outright validity failures.
- **LLMStructBench** (2602.14743; 22 models) — prompting improves structural validity for
  small models but shifts errors into wrong values; **no monotonic size → reliability
  relation**, which bears on ELI-2's cliff shape.
- **JSONSchemaBench** (2501.10868) — constrained-decoding frameworks; coverage varies by
  framework and schema.
- **VAREX** (2603.15118) — sub-4B models struggle more with *compliance* than extraction;
  extraction-specific fine-tuning at 2B gives a large gain.
- **"When Correct Isn't Usable"** (2605.02363) — 7–9B models solve math but fail to emit
  usable JSON under naive prompting; constrained decoding fixes syntax at latency cost and
  sometimes lower task performance.
- **Clinical SLM extraction** (2507.01810) — JSON most parseable; targeted prompts help;
  some 3–4B models reach high parseability; long documents degrade.
- **Tiny structurers:** NuExtract-tiny-v1.5 (Qwen2.5-0.5B fine-tune) and NuExtract 2.0
  (2B/4B/8B) (no IDs); **GLiNER2** (2507.18546 — span-tagger/NER route needing no LLM);
  **ScrapeGraphAI-100k** (2602.15189 — a 1.7B fine-tune narrowing the gap to a large MoE).
- **Schema-side methods:** Schema Reinforcement Learning (2502.18878); RL-Struct
  (2512.00319, dense rule-based schema rewards); schema key wording as an instruction
  channel (2604.14862); PA-Tool (2510.07248 — adapting tool/schema names to small models'
  pretrained patterns; the closest published "fit the interface to this executor" move).
- **Constrained-decoding tooling named:** Outlines, Guidance, XGrammar, llama.cpp grammars,
  provider structured-output APIs — the allowed "formatting constraints" tier of the wrapper.
- **Recommended architecture on record:** solver/extractor → tiny structurer → constrained
  decoder → validator → deterministic merger — (all from
  docs/topics/reference/structured-output-literature.md; ledger rows marked ELI, IRT).

**Known headwinds and small-model specialization precedents (mostly "from memory,
unverified")**

- **Lester, Al-Rfou & Constant 2021, *The Power of Scale for Parameter-Efficient Prompt
  Tuning*** (no ID on record) — prompt tuning matches full fine-tuning only above ~10B and
  lags badly at small sizes; §2 cites it as the reason the honest ELI claim is about an
  external large-model-fit interface, not the tiny model's own promptability — (source:
  docs/potential-projs/elicitation-gain.md §2 and §4 intake notes; flagged "from memory;
  verify").
- **TinyStories (Eldan & Li 2023)** — coherent generation at 1–30M params when the
  distribution is narrowed; the existence evidence for ELI-opt-3's specialist funnel; also
  named with the **BabyLM line** as the honest prior art for capability-per-parameter under
  distribution narrowing — (source: docs/potential-projs/elicitation-gain.md §4;
  docs/potential-projs/tiny-scale-measurement.md:261,492; unverified).
- **The phi-1 line** (textbook-quality data for small code models; no ID) — (source: same;
  also docs/topics/reference/synthetic-data-literature.md:52,81).
- **"Distilling step-by-step" (Hsieh et al. 2023)** (no ID) — distillation-to-small-
  specialists — (source: docs/potential-projs/elicitation-gain.md §4; unverified).
- **Small-model DSL / semantic-parsing work; text-to-SQL with small seq2seq models; "neural
  program synthesis with DSLs"** (no papers named) — the shape closest to "LLM as a compiler
  to a restricted DSL executed by a small model," recorded as a check-whether-this-exists
  item — (source: docs/potential-projs/elicitation-gain.md §4; unverified).
- **Frozen-model probing literature** (no papers) — the head-on-frozen-representations tier
  of the outer-layer taxonomy — (source: same).
- **PEFT tiers named for ELI-opt-4: LoRA, IA3, BitFit, last-layer-only, soft prompts, prefix
  tuning** (no IDs) — the spectrum against which wrapper-only is placed — (source: same).

**Task sets and verifier suites (shared with TLC)**

- **DS-1000** (2211.11501) — execution-based evaluation of real-world data-science code
  (**pandas**/numpy/matplotlib/sklearn/scipy/torch) with surface-form constraints such as
  must-use-vectorized-ops; **the direct precedent for ELI's pandas task family** — (source:
  docs/topics/reference/code-benchmarks-landscape.md 2026-08-23 chunk-5 entry;
  docs/topics/reference/evaluation-methodology-literature.md:127; unverified).
- **HumanEval+ / EvalPlus** (2305.01210), **MBPP/MBPP+**, **EvoEval** (2403.19114),
  **HumanEval Pro** (2412.21199), **BigCodeBench**, **LiveCodeBench** (2403.07974),
  **CoderEval**, **NaturalCodeBench** — the candidate verifier task pool — (source:
  docs/topics/reference/code-benchmarks-landscape.md; ledgered TLC/ELI/IRT; unverified).
- **Prompt-condition finding that governs ELI's S_0 definition:** HumanEval original is raw
  completion (stub only); MBPP original is *already* few-shot instruction + visible asserts;
  both now have named instruct variants. "Neither benchmark's prompt is test-free." So
  "one generic wrapper" is itself a benchmark-specific choice to record per run — (source:
  docs/topics/reference/code-benchmarks-landscape.md, "How HumanEval and MBPP are actually
  prompted").
- **ReCode** (2212.10264 — 30+ semantics-preserving transformations over docstrings, names,
  syntax, format) and **NLPerturbator / HumanEval-R** (2406.19783) — off-the-shelf
  machinery for the *sham wrapper* and prompt-perturbation controls — (source: same).
- **HumanEvalComm** (2406.00215 — ambiguous/incomplete descriptions with Communication Rate
  and Good Question Rate) — the interface-quality-as-variable precedent — (source: same).
- **ShortenDoc** (no ID; docstring compression on HumanEval/EvoEval; ~30% compression often
  preserves or improves pass@1) — evidence that shortening the interface need not cost
  success — (source: docs/topics/reference/code-benchmarks-landscape.md;
  docs/litreview/tlc-litreview-plan.md gate item 1).
- **Contamination / overlap audits bounding any task set:** How2Bench (2501.10711),
  LessLeak-Bench (2502.06215), HumanEval/MBPP contamination in the Pile/The Stack
  (2403.04811), CodeSearchNet duplication (2401.07930), ContextBench (2602.05892),
  StarCoder2/The Stack v2 (2402.19173) — (source:
  docs/topics/reference/code-benchmarks-landscape.md; ledgered TLC/ELI/IRT; unverified).
- **Danielle's first-hand HumanEvalPlus defects** (from TLC baseline runs, not an external
  claim): dramatic item-difficulty variance, items solved by every model tried, some items
  broken in the official release, many items "SUPER short" — a direct warning for ELI-1's
  narrowest-slice selection — (source: docs/topics/reference/code-benchmarks-landscape.md
  2026-08-23).
- **Efficiency-beyond-correctness cluster: ENAMEL (eff@k), EvalPerf / DPE, EffiBench,
  BigO(Bench)** (no IDs) — alternative success definitions beyond the pass/fail waterfall —
  (source: docs/topics/reference/code-benchmarks-landscape.md;
  docs/topics/reference/evaluation-methodology-literature.md).

**Measurement language and estimation toolkit**

- **Xu et al., *A Theory of Usable Information Under Computational Constraints*
  (𝒱-information)** (arXiv 2002.10689) — ELI's competence-vs-size curves recorded as
  I_𝒱(model → task) for a declared wrapper family 𝒱; the declaration of 𝒱 (wrappers,
  prompts, budget) is what must be published — (source:
  docs/potential-projs/elicitation-gain.md §4 2026-08-22; ledgered TLC, ELI; unverified).
- **Codex / HumanEval unbiased pass@k estimator** (2107.03374) — (source:
  docs/topics/reference/estimation-and-calibration-methods.md; ledgered TLC/ELI/EDP).
- **Lei et al., split conformal** (1604.04173) and **Angelopoulos et al., Conformal Risk
  Control** (2208.02814) — the split-conformal predictor from cheap wrapper signals to the
  expensive elicited-competence target; **Danielle flags conformal prediction as a
  cross-project tool** — (source: docs/topics/reference/estimation-and-calibration-methods.md;
  docs/potential-projs/elicitation-gain.md §4 2026-08-22).
- **Estimation discipline on record:** fractional score with the program as the unit, block
  bootstrap over provider batches, Wilson/Jeffreys/Clopper–Pearson at small n, empirical
  Bernstein for conservative floors, **calibrate-after-selection** when the best wrapper is
  chosen (the direct hazard for S_opt), Mondrian/grouped conformal per model — (source:
  docs/topics/reference/estimation-and-calibration-methods.md).
- **Cliff-bisection machinery** — bisection over the axis with binomial error bars,
  SE = 0.5/√n at the cliff, inherited from TLC's critical-ratio search with model size
  substituted for latent budget — (source: docs/potential-projs/elicitation-gain.md §4
  intake notes; docs/potential-projs/text-latent-code-autoencoder.md).
- **Re-eval harness design (respondent's, unverified but adopted as a checklist):** forced
  separation of task/template/optimizer/model/budget/evaluation; a strict optimizer
  interface (`initialize`/`propose`/`update`); **budget-first execution with a hard stop —
  "most papers quietly cheat on budget"**; full search-trajectory logging; train/val/test
  splits for the prompt search itself; cross-model transfer as a first-class experiment.
  Candidate claims it could test: gains collapse under realistic budgets; improvements are
  largely prompt-length creep — (source:
  docs/topics/reference/prompt-optimization-landscape.md 2026-08-23 entry).

**Program context: checkpoints, post-training, and the no-movement result behind ELI-3**

- **DataDecide (Magnusson et al., arXiv 2504.11393)** — the controlled data × scale suite;
  the executor family (4M → 1B) — (source:
  docs/topics/reference/pretraining-to-posttraining.md;
  docs/litreview/citation-verification-ledger.md).
- **Tülu / Tülu 3** (Ai2 open post-training stack; no ID) — the SFT family behind the
  "post-training moved nothing" first-hand account and ELI-3's pre/post pairs — (source:
  docs/topics/reference/pretraining-to-posttraining.md;
  docs/potential-projs/elicitation-gain.md §4).
- **FollowIR** (arXiv 2403.15246) — recorded as a **mismatched guess** at the AI2
  "fine-tuning dataset that moves specific-task metrics" (co-author "Kyle"); it is a
  retrieval benchmark and not a match; resolution is a question to the contact, and it gates
  ELI-3's post-training data choice — (source:
  docs/topics/reference/pretraining-to-posttraining.md; docs/open-questions-answered.md:178).
- **The "post-training elicits rather than adds" arc:** Yue et al. 2504.13837 (pass@k
  limits); Wu & Choi, *On the Limits of RLVR* (support-preserving, entropy-reducing
  reweighting); counterpoints *The Invisible Leash* (2507.14843) and *RLVR Implicitly
  Incentivizes Correct Reasoning in Base LLMs* (2506.14245) — the field-level version of
  ELI-3's hypothesis — (source: docs/topics/reference/pretraining-to-posttraining.md).
- **Hochlehnert et al., *A Sober Look at Progress in LM Reasoning*** (COLM 2025; 2504.07086)
  and **Shao et al., *Spurious Rewards*** (ICML 2026; 2506.10947) — benchmark noise and
  elicitation variance masquerading as training effects; the reason ELI reports ΔS with
  seed replicates rather than raw score — (source:
  docs/topics/reference/pretraining-to-posttraining.md; docs/research-hypothesis.md:61-62).
- **Synthetic-data post-training options for ELI-3 when no task-specific SFT set exists** —
  (source: docs/topics/reference/synthetic-data-literature.md:14).
- **DataDecide's smallest models (~10M) did not perform reasonably even on the simplified
  multiple-choice benchmarks built to give them a chance** — the repo's own context fact
  motivating the ELI-1 existence test — (source: docs/potential-projs/elicitation-gain.md §4;
  docs/open-questions-answered.md; docs/potential-projs/tiny-scale-measurement.md).

**Elicitation-as-instrument framing (program-level, not literature)**

- **The research hypothesis** — elicitation promoted from confound to instrument; the tuned
  elicitation ceiling as the strong null; both raw and elicitation-controlled readouts with
  their difference as the capability-vs-accessibility decomposition; tuning-budget accounting
  — (source: docs/research-hypothesis.md).
- **The "outer model as microscope" answer to the fairness objection** — the claim is "the
  base model contains behavior that a fixed microscope with bounded effort can elicit," not
  "the base model can do X"; Danielle's own caveat that the outer model may be "pushing the
  buttons" — (source: docs/potential-projs/elicitation-gain.md §4 verbatim + response).
- **Controls on record:** outer-model-only under the same budget; sham/adversarial wrapper of
  equal length or complexity; **fixed-outer swapped-executor**; **wrapper transfer** (optimize
  on s, evaluate on s′) as a similarity readout; **answer-leak audit** (token overlap between
  outer critiques and final answers) — (source: docs/potential-projs/elicitation-gain.md §4
  intake notes).
- **Tuning-response curves** (performance vs. search budget per paradigm) and demonstration
  hygiene for existence proofs — (source: docs/potential-projs/movement-microscope.md).

**In-program neighbors (positioning targets, not literature)**

- **`text-latent-code-autoencoder.md` (`TLC`)** — the harness ELI reuses (objective J(θ),
  feasibility waterfall × semantic correctness Eq. 4–5, LLM-as-optimizer loop Eq. 7, the
  COMP-NL vs. COMP-SHORT latent-format axis that becomes ELI-opt-2's DSL axis), plus the
  cliff-bisection machinery — (source: docs/potential-projs/elicitation-gain.md §4;
  docs/potential-projs/text-latent-code-autoencoder.md:1562,1574).
- **`icl-elicitability.md` (`ICL`)** — the hand-tuned counterpart of the same null —
  (source: docs/potential-projs/icl-elicitability.md:71-74).
- **`movement-microscope.md` (`MIC`)** — the same microscope/detection-limit framing and the
  guaranteed-effect calibration — (source: docs/potential-projs/movement-microscope.md:63).
- **`irt-reanalysis.md` IRT-10** — the BoolQ format intervention (cloze vs. MCQ,
  label-balanced subsets, flipped label order) as the first concrete instance of the
  elicitation thesis; keep its design consistent with ELI-2's controls — (source:
  docs/potential-projs/irt-reanalysis.md:76,320;
  docs/potential-projs/elicitation-gain.md:131-134).
- **`tiny-scale-measurement.md` (`TINY`)** — ELI-1 is the within-reach question run as an
  existence test under an oracle interface at every size — (source:
  docs/potential-projs/tiny-scale-measurement.md:55-56).
- **`clean-code-preference-icl.md`** — shares the verifier suite (functions with tests,
  length ratio vs. reference) — (source: docs/potential-projs/elicitation-gain.md §3).

**Provenance and hygiene notes**

- Artifacts on disk backing the prompt-optimization and prompt-compression clusters (final
  reports, ~25 search CSVs, a 648-row deep-search table, three PDFs) are indexed by
  `INDEX.md` in each bundle — the recovery path for the missing anchors — (source:
  docs/topics/reference/prompt-optimization-landscape.md;
  docs/topics/reference/prompt-compression-and-optimization-literature.md).
- Known identifier slips to fix before citing: EPiC as 2408.11198 vs. 2410.14321;
  AlphaCodium as 2401.08500 vs. 2401.19489; Midolo "2024" carrying a 2601 number;
  Nano-Capsulator's first author (Chuang vs. Zhou). The SciSpace review also has [6] under
  two names and a malformed comparison-table header — (source:
  docs/litreview/tlc-litreview-plan.md gate item 2;
  docs/topics/reference/nl-bottleneck-prior-art.md;
  docs/topics/reference/prompt-optimization-landscape.md).
- The "how much data" rules of thumb in §4 (linear head tens–few thousand; soft prompts
  hundreds–tens of thousands; LoRA thousands–hundreds of thousands) are explicitly recorded
  as **unsourced folklore, not to be quoted** — (source:
  docs/potential-projs/elicitation-gain.md §4 intake notes).

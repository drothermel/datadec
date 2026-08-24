# Prompt optimization — landscape reference (LLM-as-optimizer and system-level)

**Kind:** reference (accumulator for the automatic-prompt-optimization subfield as it bears
on TLC's optimizer loop and ELI's outer optimizer of the interface). Entries are dated.
Characterizations are the SciSpace agent's; identifiers unverified. Sibling:
`prompt-compression-and-optimization-literature.md` (PCRL, Nano-Capsulator, EPiC in
detail) and `nl-bottleneck-prior-art.md` (the TLC prior-art verdicts).

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-prompt-optimization-agent-artifacts-zip_97f96bdd-5f03-456a-8061-f2e8204d51c2_1787423566/` — final report, an earlier draft, the insight-extraction
pass, and ~25 search CSVs, two merged with per-paper extracted method-category /
contribution / domain / limitations columns and one 648-row deep-search table.
**`INDEX.md` inside the folder is the file-level index**, including the canonical works
the review omits.

---

## 2026-08-22 — SciSpace deep review (undated, ~early 2026)

**Danielle's prompt (verbatim):**

> What is the current state of the Prompt Optimization subfield? What are the related
> works that are foundational, and what are the strongest recent themes? Please focus
> especially on approaches that use LLMs to do the prompt optimization and on approaches
> that optimize not just the prompt but also other aspects of the system (for example
> sampling hpms, tool use, etc). My goal is to get a general understanding of the subfield
> before diving deeper into specifics, so prioritize breadth and qualitative comparisons.

Scope answers (verbatim):

> Please include as broad a set of methods as possible, I'd say focus mostly on things like
> prompt optimization, so RAG systems are a bit further afield for this deep research. I'm
> interested across broad domains, but especially in the realm of verifiable tasks like
> code generation. And focus on recent developments but provide a brief intro on the
> historical perspective.
>
> For context, I'm currently writing a paper on compressing functions using an frozen
> pretrained black box LLMs as encoders and decoders, with instructions to compress the
> input function and then reconstruct it in a way that preserves functionality not surface
> form. We're using an external LLM to optimize the compression prompt. So I'm interested
> in things related to this but more interested in getting a broad understanding of the
> landscape.

**The map the report gives (condensed).**

- *History:* manual heuristics (pre-2023) → Promptbreeder's self-referential evolution
  (2023; evolves the mutation prompts too) → diversification (meta-prompting,
  gradient-inspired, system-level, code-specific; 2024–25) → compound-system
  optimization (2025–26).
- *Method families and representatives:* evolutionary / population (Promptbreeder
  2309.16797, PhaseEvo 2402.11347, AMPO 2410.08696); meta-prompting /
  generation-refinement with a critic (PromptWizard 2405.18369; Tang et al. "LLMs as
  prompt optimizers ≈ gradient optimizers", update direction + update method, 2402.17564;
  Learning from Contrastive Prompts 2409.15199); gradient-inspired textual updates
  (LLM-AutoDiff 2501.16673 — textual gradients through multi-component workflows;
  Dual-Phase accelerated 2406.13443); query-dependent learned generators (QPO
  2408.10504, offline RL); hybrids (Davari et al. reinforcement + diversification +
  migration for black-box LLMs, 2507.09839).
- *Joint instruction × exemplar optimization* as a settled finding (Wan et al. "Teach
  better or show smarter", 2406.15708).
- *System-level:* SPRIG (2410.14826) edit-based genetic search over system-prompt
  components — one optimized system prompt ≈ task-specific prompts across 47 tasks;
  Trace / OptoPrime (2406.16218) — workflows as graphs, an LLM optimizer over
  heterogeneous parameters incl. prompts, hyperparameters, and code; LLM-AutoDiff;
  RePrompt (2406.11132) for agent planning instructions from interaction histories; Lin
  et al. survey of LLM-based optimization of compound AI systems (2410.16392) listing
  targets: prompt components, sampling parameters, retrieval configs, tool specs,
  orchestration logic.
- *Verifiable / code:* Prochemy (2503.11085, execution-driven refinement), EPiC
  (2408.11198, cost-aware evolutionary), PromSec (security + functionality loop), Midolo
  et al. empirical prompt guidelines, Shashikala et al. accuracy + diversity objectives.
- *Feedback taxonomy:* validation accuracy; LLM-as-critic; execution feedback;
  contrastive (success vs. failure prompts).
- *Open problems the report lists:* evaluation protocol inconsistency; cross-model
  transfer of optimized prompts; sample efficiency; overfitting to validation sets;
  prompt brittleness; prompt length vs. performance ("prompt compression" as an open
  problem — the link to TLC); integration with fine-tuning.

**Intake notes.**

- For a "foundational works" question the review's omissions are severe: no APE, OPRO,
  ProTeGi/APO, TextGrad, DSPy/MIPROv2, GEPA, EvoPrompt, or RLPrompt. Its foundations are
  Promptbreeder plus surveys. The 648-row deep-search table on disk is the place to
  recover the missing anchors; the sibling prior-art bundle already has RLPrompt and
  SPELL.
- The most useful content for this repo is the system-level cluster — Trace/OptoPrime,
  LLM-AutoDiff, SPRIG, and the compound-AI survey's target list — because "optimize the
  harness, not just the prompt" is exactly the ELI outer loop and TLC's θ (prompts,
  templates, latent format, stage decomposition, tool use, sampling). These are the
  positioning references.
- The "implications for function compression" section is the agent restating
  Danielle's own context; nothing new, dropped except the prompt-length-vs-performance
  framing.
- Citation hygiene: [6] appears under two names; [5] in two roles; the comparison
  table's header is malformed. All identifiers as produced.

## 2026-08-22 — GEPA and DSPy as the optimizer incumbent (from the estimation conversation)

Surfaced with identifiers in a conversation on using test results as optimizer signal
(record: `estimation-and-calibration-methods.md`, third entry): **GEPA** — "Reflective
Prompt Evolution Can Outperform Reinforcement Learning", arXiv 2507.19457 — samples
trajectories, reflects on failures in text, proposes prompt edits, and combines lessons
across a Pareto frontier; in DSPy the metric may return a scalar plus textual feedback, at
whole-program or predictor level (`pred_name`, `pred_trace`), with a trainset for
reflection and a valset for Pareto tracking (`dspy.ai` optimizer docs). This is the first
of the missing-anchor list (APE, OPRO, ProTeGi, TextGrad, DSPy/MIPROv2, GEPA) to arrive
with an ID; the rest still need supplying. All claims are the respondent's, unverified.

## 2026-08-23 — Re-evaluating 2022–2024 findings under modern models (ChatGPT conversation, early 2026; intake in progress)

A long ChatGPT conversation Danielle describes as a real "aha" moment. **The
conversation dates from roughly January–March 2026** (Danielle's correction at intake);
time-sensitive content — deadlines, venue timing, model pricing — reflects that period.
The workshop paper it fed was submitted and rejected, and Danielle is returning to the
direction as of 2026-08-23 (project history in the TLC doc §4). Arriving in chunks,
full verbatim transcript at
`~/drotherm/data/convo-artifacts/2026/2026-08-23-prompt-opt-reeval-aha/` (source link in
its `INDEX.md`). This entry summarizes chunk 1 and may be extended. All respondent
claims and identifiers unverified.

**Danielle's motivating observation:** current cheap LLMs are often more powerful than
the best LLM of two years ago — so which 2022–2024 prompt-optimization findings (OPRO,
Revisiting OPRO, LLM-based evolutionary optimizers) actually hold up now? She initially
framed this as "purely analysis" suited to a repro/blog-post track, unsure it counts as
a research paper.

**Identifiers supplied** (all agent-supplied; ledger rows added). This delivers the
second and third of the missing-anchor list (OPRO now has an ID; APE, ProTeGi, TextGrad,
DSPy/MIPROv2 still needed):

- **OPRO** — "Large Language Models as Optimizers", 2309.03409 (2023).
- **Revisiting OPRO** — "The Limitations of Small-Scale LLMs as Optimizers",
  2405.10276; per the respondent an ACL Findings 2024 paper (2024.findings-acl.100):
  small open models (LLaMA-2, Mistral 7B) are weak as optimizers; strong
  direct-instruction baselines recommended in that regime. Danielle named the title;
  the ID is the respondent's. The respondent calls it "a template" for her idea, with
  the axis changed from small-vs-big to modern-cheap-vs-old-SOTA.
- **LEO** — "Large Language Model-Based Evolutionary Optimizer: Reasoning with
  elitism", 2403.02054 (Danielle-named title, respondent ID); population-based
  numerical optimizer, guardrails against hallucination.
- APO systematic survey 2502.16923; automatic prompt engineering survey 2502.11560
  (both 2025, offered as scaffolding for a fair comparison protocol).
- **PromptBridge** 2512.01420 — cross-model prompt transfer (2025).

**Venue map for replication/re-eval work** (respondent's, unverified): MLRC/ReproML
(reproml.org); ReScience C (replication journal); TMLR reproducibility certifications;
mainline venues when framed as evaluation protocol + findings (Findings-style tracks);
NeurIPS Datasets & Benchmarks if packaged as an evaluation suite.

**The re-eval harness sketch** (respondent's design, condensed): "MLflow +
lm-eval-harness + Optuna, narrowly scoped to prompt optimization." Six pillars —
(1) forced separation of task / template / optimizer / model / budget / evaluation;
(2) a strict optimizer interface (`initialize` / `propose` / `update`) every method must
fit, enabling comparison of search behavior, not just final accuracy; (3) budget-first
execution (tokens/$/steps/wall-clock as first-class config, hard stop on exhaustion —
"most papers quietly cheat on budget"); (4) full search-trajectory logging (every
candidate prompt, per-step cost and score, seed variance, edit distance);
(5) train/val/test splits for the prompt search itself, reporting only on test
(prompt-overfitting control); (6) cross-model transfer as a first-class experiment
(optimize on A, evaluate on A/B/C). v1 scope: 3–5 tasks, 3–4 optimizers (random,
hill-climb, OPRO-style, evolutionary), 4–6 models, YAML configs, one command per
figure, reproducibility manifest (model version, API params, commit, cost, seed).

**Candidate claims the instrument could test** (respondent's list, useful as hypothesis
menu): gains collapse under realistic budgets; search trajectories differ qualitatively
across model families; improvements largely from prompt length creep; prompts optimized
on model A do/don't transfer under controls.

**Positioning** (respondent, condensed): a benchmark counts as research when it is a
measurement framework / diagnostic lens / standardization contribution / belief-revising
findings / adopted artifact — "the paper is about the methodology and the findings; the
library is the evidence and artifact." Suggested arc: (1) measurement + first
disconfirmation, (2) mechanism paper on why the failures happen (why does OPRO plateau;
is the optimizer doing search or writing longer prompts), (3) optional constrained
improvement (budget-aware optimizer, transfer-robust representation, stopping
criterion). Explicit anti-scope-creep advice: small, clean, convincing — not the
definitive benchmark.

**Chunk 2 (2026-08-23):** the conversation's target project is TLC — Danielle attached
the pitch "LLM-as-Optimizer of Natural Language Bottleneck Model" (Rothermel*, Li*,
Cho; Recursive Self-Improvement Workshop @ ICLR 2026) plus its Table 5 OpenRouter
pricing menu. The re-eval framing merged with TLC's harness-optimizer formulation:
cheap models make distributional (variance-first) evaluation of optimizer loops
affordable; the respondent proposed a budgeted re-eval workshop slice with a
phase-transition claim and a transfer matrix, and operationalized Danielle's
posttraining-collapse hypothesis (optimizer competence vs. task competence; transfer).
Full detail in the TLC project doc §4 (2026-08-23 entry); transcript and pitch PDF in
the convo-artifacts bundle. Chunk 2 supplied no new identifiers, but the pitch's own
bibliography settles two earlier attribution conflicts (see ledger).

## 2026-08-24 — the Feb-2026 Claude novelty check already held part of the missing-anchor list

Danielle's 2026-02-04 Claude-generated novelty check (bundle:
`~/drotherm/data/convo-artifacts/2026/2026-08-24-notion-lit-reviews/lit-review-claude-novelty-check.md`)
cites OPRO 2309.03409, EvoPrompt 2309.08532, and PromptBreeder 2309.16797 as
component-feasibility evidence for gradient-free prompt optimization — three of the
anchors the August 2026 SciSpace reviews were faulted for omitting were already in
the project's February record. EvoPrompt (Guo et al.) is distinct from EvoPrompting
2302.14838 (Chen et al., neural-architecture prompts) — both are now ledgered.

## 2026-08-24 — the cross-model prompt-transfer cluster (Danielle's April-2026 notes, first-hand)

From her 2026-04-17 reflection (bundle:
`reflection-april-2026-code-comp-nl-latents.md`; no IDs on record yet). Three
papers converging on prompt model-specificity: **PromptBridge** (cross-model
prompt transfer; five coding benchmarks; "model drifting" with 50–70% relative
loss on xCodeEval for some within-family transfers); **discrete GA prompt
optimization for secure Python code** (JSS 232:112682, Feb 2026 — abstract pasted
verbatim in the source; security-specific mutations; "prompts optimized on one LLM
showed lack of transferability to others"); **Tuning LLM-based Code Optimization
via Meta-Prompting** (industrial; five codebases, 366h runtime benchmarking, up to
19.06% improvement; same transferability finding). Plus **Guidelines to Prompt
LLMs for Code Generation** (10 empirical guidelines) and her flag to skim
Prochemy/Prompt Alchemy (2503.11085, already row B). Bearing: optimizer outputs
are model-specific — evidence for the cross-decoder-transfer falsifier and a
design constraint for any shared-arm-space comparison. Unverified.

## Undated (Perplexity; intake 2026-08-24) — LLM-as-optimizer taxonomy and toy-RL-with-minimal-translation survey (one turn)

Perplexity task supplied with link (bundle:
`perplexity-llm-as-optimizer-toy-rl-convo.md`); undated (task IDs carry no
timestamp; her prompt's dual-LLM-autoencoder framing places it after TLC ideation,
and the response cites 2025 work). Her question: which works genuinely deserve
"LLM as an optimizer" in a broad sense; which put a black-box LLM in a classical
ML component slot; and what exists on LLMs solving classic toy optimization/RL
problems (CartPole, minigrid, mountain-car) with minimal state↔language
translation. **No arXiv IDs anywhere in the response** — author-year only;
identifiers to recover later. Agent-supplied, unverified.

**True optimizers (the loop is the point):** OPRO 2309.03409 (already canonical
here; read as "warm-start guesser that improves in context," strongest when the
LLM has solution-structure priors); **FunSearch** (Nature; search in *program*
space, LLM as learned mutation operator inside an evolutionary loop; cap set +
bin-packing results; **AlphaEvolve** 2025 follow-up adds RL fine-tuning of the
mutation operator); **Eureka** (Ma et al. 2023 — LLM optimizes the *reward
function*, RL optimizes the policy: LLM-outer/RL-inner nesting; beats human
reward engineers on 83% of 29 envs); **LLAMBO** (Liu et al. 2024 — LLM replaces
both surrogate and acquisition function in Bayesian optimization, the
Gaussian-process slot).

**Classical component slots:** **ReEvo** (Ye et al. 2024 — FunSearch structure on
NP-hard combinatorial problems + a reflection step, "verbal gradients");
**LLM-SR** (Shojaee et al. 2024 — equation skeletons as programs + numerical
parameter fitting; LLM injects structure priors into symbolic regression);
**DICL** (Benechehab et al. 2024 — LLM in the transition-model slot of
model-based RL; disentangling multivariate continuous observations for
in-context prediction). **Closest to the dual-LLM autoencoder framing:**
**Matryoshka** (small white-box LLM guiding a large black-box LLM via decomposed
task representations) and **ACING** (actor-critic over instruction optimization
for black-box LLMs); the generator-evaluator dual-LLM pattern is pervasive but
never branded as an autoencoder.

**Toy RL with minimal translation (sparse, mixed):** **Sensory-Motor Control
with LLMs via Iterative Prompting** (2025 — the closest match to her setting:
minimal labeling of the observation vector, LLM writes a Python control function,
performance fed back; o1-class models solve CartPole reliably); **Bosio &
Mueller 2024** (FunSearch structure for pendulum swing-up / ball-in-cup;
interpretable program policies); **Atari-GPT** (Waytowich et al. 2024 — negative
result: multimodal LLMs fail as direct per-timestep policies).

**The synthesized claim worth keeping:** LLMs work as *outer-loop optimizers*
(propose, mutate, refine programs/rewards against a fast external evaluator) and
fail as *inner-loop controllers* (per-timestep reactive decisions); the apparent
exception (sensory-motor control) conforms once you notice the LLM writes an
offline controller rather than acting as one. Recommended general framework:
FunSearch/Eureka/ReEvo — LLM as mutation operator over programs with an external
evaluator supplying the loss signal. Relevant context for the whetstone-envs
minigrid spec-out (staging placeholder) and for the C2 optimizer-vs-bandit
decomposition (OPRO's "warm-start guesser" reading).

## 2026-08-24 — NotebookLM LLM-driven-optimization notebook (25 sources; extends the LLM-as-optimizer entry)

Danielle supplied a NotebookLM notebook over 25 LLM-optimization sources
(bundle: `nblm-llm-optimization-notebook.md`; **link not yet supplied** — fill
in the ingest-link-index row when available; items 12–20 carry arXiv IDs,
the rest titles only; agent-generated, unverified; NotebookLM caveat). The
direct extension of the undated-Perplexity LLM-as-optimizer taxonomy above,
adding a **standalone-vs-hybrid axis** to its outer-loop-vs-inner-loop
conclusion:

- **Headline aggregate verdicts.** LLMs are largely ineffective as standalone
  direct optimizers or planners on long horizons — small models are beaten by
  zero-shot CoT (Revisiting OPRO 2405.10276, already on record: few-shot CoT
  dramatically beats iterative OPRO for 7B/13B models); frontier models suffer
  mode collapse, hallucination, and **poor optimization state-tracking** (no
  density estimation — they re-enter known-infeasible/OOM regions despite the
  trial history in context). **Hybrids win**: LLM semantic reasoning + classical
  state management.
- **Centaur** (from the "Can LLMs Beat Classical HPO?" autoresearch study, no
  ID) — the flagship hybrid: CMA-ES's (µ, σ, C) state is shared with the LLM
  for optimizer-informed proposals; best overall on a 14-parameter HPO
  benchmark where pure LLMs lose to TPE/CMA-ES. Scale note: 0.8B suffices for
  the hybrid, while unconstrained code editing needs 27B+/frontier (Gemini 3.1
  Flash-Lite fails 87–94% of edits). "Classical methods find the path; LLMs
  suggest where to look next."
- **Search-structure results relevant to C2/TLC-opt:** Execution-Grounded
  Automated AI Research (2601.14525) — evolutionary search over LLM-generated
  research ideas is sample-efficient (48.0% → 69.4%) while RL updates
  mode-collapse; SEAL self-edit search — archive helps but mode
  collapse/homogenization requires explicit novelty pressure; EvoPrompt-style
  multimodal evolutionary prompt optimization elicits emergent strategies from
  20 labeled examples; **HAPO** — unit-level attribution + UCB selection,
  +13.28% over zero-shot CoT at far fewer API calls than OPRO/APE (a
  budget-frugality datum for the eval-matched-budget framing).
- **MCTS hybrids:** SCULPT (constraint-pruned MCTS, 89.9% MATH / 97.8% GSM8K on
  GPT-5.2, >34% of branches pruned), JudgeFlow (block-level failure
  attribution), CogMCTS 2512.08609.
- **Agent-scaffold optimization:** Artemis/Evolving Excellence (+13.6%
  competitive programming, −36.9% token cost), ReCreate (meta-agent edits
  scaffolds from execution traces; >5% over human-designed scaffolds) — the
  scaffold-as-search-space framing.
- **RL-adjacent (minigrid/world-models context):** Language Feedback Models —
  small offline models distilling LLM feedback for imitation learning on
  ALFWorld/ScienceWorld, beating direct LLM-expert imitation cheaply; Modeling
  Capabilities for Sequential Decision Making — LLMs fail at direct policy
  output on unfamiliar dynamics but excel as **reward models** via AI feedback
  (converges with the outer-loop-optimizer conclusion); planning survey — LLMs
  as heuristic guides for classical planners, not standalone planners.
- **ID fills and metadata-only items:** AlphaEvolve **2506.13131** (fills the
  earlier no-ID row); LLM evolutionary optimizer with elitism 2403.02054; LLMs
  as particle swarm optimizers 2504.09247; test-time compute-optimal scaling as
  an optimizable graph 2511.00086 (Agent-REINFORCE textual gradients);
  diffusion LLMs for offline BBO 2601.14446 (bidirectional modeling beats
  autoregressive for design spaces); phylogenetic-tree code evolution
  2601.14523. Surveys: LLMs-for-evolutionary-optimization, AutoML-in-the-age-
  of-LLMs, GI-for-LLM-code (PSB2; helps small models, saturates at GPT-4),
  LLMize (OPRO-style numerical optimization; LLM proposes the annealing
  cooling rate).

## 2026-08-24 — AI4SE prompt-opt / repair-loop findings (NBLM AI4SE notebook)

Companion to the compression-side entry in `code-compression-literature.md`
(same date; bundle `nblm-ai4se-code-notebook.md`; no IDs; agent-generated,
unverified):

- **Prochemy, framework detail** (2503.11085, already on record as a name):
  execution-driven prompt refinement — mutation (LLM rewrites the prompt) →
  evaluation (Pass@1 on a contamination-free training set with
  complexity-weighted scoring) → selection, with early stopping. Plug-and-play
  over CoT/LDB/Self-Collaboration; +4.04% zero-shot average; Prochemy+LDB
  reaches 96.3% HumanEval on GPT-4o; **works on o1-mini too** (LiveCodeBench
  40.6 → 44.6) — reasoning models still benefit from optimized instructions.
- **PromptCS** — the *continuous*-prompt contrast case: frozen LLM + trained
  prompt agent (BiLSTM encoder + MLP over meaningless learnable tokens)
  matches or beats task fine-tuning on summarization at a fraction of cost
  (67h vs 211h for 7B), works from ~100 samples. Soft prompts as the
  non-NL-latent counterpart to NL prompt search (cf. gist tokens /
  Proto-tokens on the compression side).
- **The feedback paradox (Fluorine)** — an optimizer-loop design datum:
  feeding fuzzer-generated counterexamples back into the prompt *confuses*
  the model (large textual repr, random values); **restart-from-scratch beats
  conversational repair by 7–21%** on real-world C/Go→Rust translation
  (success collapses past ~100 LoC). Contrast: TransAgent and LANTERN find
  *localized* execution state (aligned CFG-block variable values) does help —
  the fault line is raw-global vs localized feedback, not feedback per se.
- **LANTERN's cross-language repair hypothesis** — when a bug resists repair
  in its native language, translating it into the model's strong language and
  back beats iterating deeper in place (Rust +22% Pass@10; harder bugs become
  solvable). A striking representation-choice datum: changing the linguistic
  surface of the same problem changes what the frozen model can do — adjacent
  to TLC's cross-decoder and NL-likeness questions.
- **RepE code-correctness probing** — LLMs encode an extractable correctness
  direction (LAT: PCA over paired correct/incorrect hidden states at the last
  token); projection scores beat log-likelihood and verbalized confidence for
  ranking candidates, +21.3% Pass@1 on HumanEval (vs RankEF's +17.7% at 172
  GPU-hours; the PCA fit takes ~3.75s); strong OOD generalization. A
  test-free inner-loop verifier signal — relevant wherever candidate ranking
  under budget matters (C2's eval-matched framing; white-box only.)
- **TransAgent** — four-agent translation with CFG-block execution alignment
  (95.8–100% block-mapping accuracy; contamination-free post-2023-08
  benchmark; stable in LoC/complexity).

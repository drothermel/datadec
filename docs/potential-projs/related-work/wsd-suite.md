# wsd suite — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`wsd-suite.md`](../wsd-suite.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

High-recall corpus for the WSD retrain suite (`WSD-1`–`WSD-3` and the
WSD-opt-* arms). Every item below is on record somewhere in this repository and is
listed with its repo source. **Nothing here is verified**: the annealing accumulator
and the SciSpace/browsing reports that feed it are agent-generated, and the citation
ledger's own header says no identifier in it has been checked. Inclusion is
deliberately generous — a line is cheap.

**Stable-phase + decay-branch methodology (the lineage the §4 origin entry names as
the reason this dataset should exist)**

- **Hägele et al. 2024, *Scaling Laws and Compute-Optimal Training Beyond Fixed
  Training Durations*** (2405.18392) — the methodology proposal this suite executes:
  constant LR + short cooldown matches cosine, so scaling-law and data experiments
  should use stable-phase runs with cheap decay branches rather than a full cosine run
  per budget; also the (1-sqrt) cooldown shape and branch reuse as the cost model
  behind WSD-opt-1. (source: docs/topics/reference/schedules-and-annealing-literature.md;
  docs/potential-projs/wsd-suite.md §4/§5)
- **MiniCPM, *Unveiling the Potential of Small Language Models with Scalable Training
  Strategies*** (2404.06395) — WSD origin and the practical template the core design
  copies: ~10% decay completes convergence, new data mixed in strictly during decay
  (the basis of WSD-opt-4); the stable-checkpoint-plus-1-sqrt-decay-resume protocol
  described as standard in careful small-scale studies; decay-phase gradient statistics
  (weights move less, loss falls faster, gradient norm falls, consecutive-update cosine
  turns positive) as a per-branch instrument; "loss drop in decay ≈ a 5× larger model";
  decay-branch reuse for data–model scaling laws at linear rather than quadratic cost.
  (source: docs/topics/reference/schedules-and-annealing-literature.md, entries 1 and 2;
  docs/potential-projs/annealed-readouts.md §4 2026-08-22)
- **OLMo, *Accelerating the Science of Language Models*** (no ID on record) — the
  training setup DataDecide inherits and therefore the parity target of infrastructure
  step 1. (source: docs/topics/reference/schedules-and-annealing-literature.md)
- **Wen et al., *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss
  Landscape View*** (2410.05192) — the mechanism reading of the stable/decay split the
  whole suite is described in; the interpolation signature as "the closest thing to a
  river test"; their own validation branches a constant-LR run at 20B tokens, decays 5B
  and interpolates; the WSD-S variant resumes from decayed checkpoints, which is the
  precedent for WSD-opt-3's cosine-vs-WSD twins; and the claim that valley geometry is
  data-property-dependent (deterministic tokens = river, uncertain tokens = walls), i.e.
  plausibly recipe-dependent. (source: docs/topics/reference/landscape-literature.md;
  docs/topics/staging/checkpoint-tomography.md; docs/topics/reference/token-level-literature.md)
- **Wen et al.'s toy-bigram validation** (same paper) — a synthetic language of varying
  token determinism reproduces the river-valley geometry; stable phase learns
  deterministic tokens, decay phase learns stochastic ones; on real data a Spearman
  ≈0.39 between token-level uncertainty and local sharpness. Relevant because branch
  endpoints on different recipes are the multi-recipe version of that experiment.
  (source: docs/topics/reference/landscape-literature.md 2026-08-18;
  docs/topics/reference/token-level-literature.md)
- ***Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning Rate
  Scheduler*** (no ID on record) — plots the landscape in pre-cooldown→final vs.
  local-Adam-step coordinates, noting a clear river-valley visualization had been
  lacking; a figure precedent for what a decay branch looks like geometrically.
  (source: docs/topics/reference/landscape-literature.md;
  docs/refs/research-trajectory-pre-to-post-training.md)
- ***Scaling with Collapse: Efficient and Predictable Training of LLM Families***
  (2509.25087) — well-tuned runs' loss curves collapse onto a shared shape; a
  curves-only, weight-free cross-run comparability criterion, i.e. a candidate check
  that the WSD twins and the released cosine runs are "the same river".
  (source: docs/topics/reference/landscape-literature.md)
- **Tissue et al. 2024, LR-annealing scaling law** (2408.11029) — adds a term linear in
  "annealing area" to the loss-vs-compute law: the LR trajectory affects realized loss
  beyond total tokens, which is what makes branch length a designed parameter rather
  than a detail. (source: docs/topics/reference/schedules-and-annealing-literature.md)
- **Second LR-annealing scaling-law citation** (2508.01483) — paired everywhere with
  Tissue ("forward area" vs. "annealing area"; annealing "momentum" — LR changes
  reflected in loss with a delay growing with annealing slope; 10–20% annealing ratio).
  Flagged in the accumulator as an unknown paper, possibly a mis-ID; on the open-ID
  check list. (source: docs/topics/reference/schedules-and-annealing-literature.md
  working list; docs/litreview/citation-verification-ledger.md)
- **Multi-power law, Kairong Luo et al.** (2503.12811, ICLR 2025) — power law on the
  sum of learning rates plus decay-drop terms; fitted on a few runs it extrapolates to
  unseen schedules and discovers a WSD-like schedule beating cosine. For this suite it
  is both the analytic prediction of which recipes are most decay-sensitive (the §2
  suggestion for choosing the first recipe pair) and a fit the suite could
  ground-truth. (source: docs/topics/reference/loss-curve-forecasting.md;
  docs/potential-projs/wsd-suite.md §2)

**Checkpoint merging as the eval-cost alternative to branches (the sibling method the
suite would validate against)**

- **WSM, *Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM
  Pre-training*** (no ID on record) — merging recent checkpoints with weights from an
  emulated decay curve gives an annealed model without altering the live LR; merged
  models reported to closely mirror a true anneal at intermediate stages of long runs.
  Validated on stable-phase runs — which is exactly what this suite would produce.
  (source: docs/topics/reference/schedules-and-annealing-literature.md;
  docs/refs/research-trajectory-pre-to-post-training.md)
- **Nemotron 3 Super** (no ID on record) — sliding-window checkpoint merging as a
  production mid-run readout, ~16% of total pretraining FLOPs saved. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **Branch-and-Merge** (2407.08699) — merge models fine-tuned on data subsets; smaller
  but higher-quality weight changes, less forgetting; filed as adjacent to the merging
  angle. (source: docs/litreview/citation-verification-ledger.md;
  docs/topics/reference/schedules-and-annealing-literature.md third entry)
- **Model soups (Wortsman et al.)** (2203.05482) — weight-space ensembling; the
  landscape file records that souping/task arithmetic work only within a basin, which
  is the precondition any merge-based decay proxy on this suite inherits. (source:
  docs/topics/reference/landscape-literature.md;
  docs/litreview/citation-verification-ledger.md moe row)

**Decay-phase data: what to anneal on and when (the WSD-opt-4 flank)**

- **Llama 3 "Annealing Data"** (no ID on record; Meta PDF) — upsample small HQ code/math
  near the end; 8B GSM8K +24.0 / MATH +6.4, 405B negligible; final 40B tokens at 30%
  new / 70% default with LR linearly to 0, used as a data-valuation protocol. The
  canonical late-HQ result and its scale attenuation. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **Blakeney et al., Databricks *Does your data spark joy?*** (no ID on record) — 7B /
  1T tokens; end-of-training domain upsampling → MMLU +6.90, GSM8K +8.26, HumanEval
  +6.17 pp; 10–20% of training as the general-vs-targeted trade-off point. The budget
  heuristic behind the default branch length. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **OLMo 2 / Dolmino** (2501.00656) — mid-training "Dolmino mix" targeting weak spots,
  LR decayed to zero across the phase; annealing as a data-evaluation tool (30/70); the
  open reproducible template for a released decay recipe. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **TREC (Training Re-evaluation Curves), Bergsma et al.** (2509.25380) — re-evaluate
  each training batch with the final weights; the curve dips in a "receptivity valley"
  before the end, identical HQ amounts do best near the TREC minimum, and TRECs are
  predictable from AdamW's implicit EMA timescale; claims to explain Llama-3-405B's
  null. Directly bears on *where* to place branch points. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **PDPC** (2501.13126) — Perplexity Difference between a weak and a strong model ranks
  samples by when they should be learned; high-PD deferred; offline corpus arrangement;
  +8.1% avg MMLU/CMMLU on 3B / 1T. A "when should this data arrive" predictor. Note the
  accumulator records this was misattributed in one report. (source:
  docs/topics/reference/schedules-and-annealing-literature.md, entries 1 and 2)
- **AutoScale** (2407.20177) — optimal domain mix changes with scale; HQ sources
  dominate small budgets then saturate, diverse CC keeps paying at large budgets; fit at
  small budgets and extrapolate. Bears on whether a decay mixture chosen at 150M
  transfers. (source: docs/topics/reference/schedules-and-annealing-literature.md)
- **Data Mixing Laws, Ye et al.** (2403.16952) — validation loss vs. domain proportions
  in an exponential form nested with scaling laws; predicts unseen mixtures. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **UtiliMax / MEDU** (2501.11747) — size-aware heuristics as strong baselines;
  portfolio optimization over ablation- or LLM-estimated utility; claimed up to 200×
  compute savings vs. brute ablations. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **Parmar et al. 2024 continued-pretraining recipe** (no ID on record) — two-stage CPT:
  general blend up-weighted for HQ sources, then switch once LR ≈ η_max/5 to
  QA/targeted data; cosine decay from original η_min, no warmup, stay
  distribution-adjacent. A schedule-coupled data-switch precedent for the branch runner.
  (source: docs/topics/reference/schedules-and-annealing-literature.md)
- **FineWeb / FineWeb-Edu** (2406.17557) — educational-value classifier
  (Snowflake-arctic-embed-m + linear head, 460k Llama-3-70B-Instruct annotations, keep
  ≥3, ~82% F1), 1.3T of 15T tokens, quoted +12% MMLU / +24% ARC, ~6,000 H100 hours to
  classify. Candidate source of a "selected" decay slice. The record flags that the
  claim "edu filtering works best during annealing" is a respondent inference FineWeb-Edu
  never tested. (source: docs/topics/reference/schedules-and-annealing-literature.md,
  second entry)
- **Phi-4** (2412.08905) — synthetic data throughout and especially late; multi-agent
  generation, self-revision, rejection sampling, execution/proof verification;
  decontamination incl. MinHash + semantic; post-cutoff AMC as contamination-proof eval.
  (source: docs/topics/reference/schedules-and-annealing-literature.md, second entry)
- **YuLan-Mini** (2412.17743) — context extension 4K→32K *during* annealing at constant
  token batch; topic-based recall and cross-lingual synthetic generation. A different
  use of the decay window than WSD-opt-4's. (source:
  docs/topics/reference/schedules-and-annealing-literature.md, second entry)
- **Nemotron-CC** (2412.02595) — ensemble quality classifiers (Mistral-based,
  Nemotron-340B, DCLM) plus synthetic rephrasing of HQ segments; sits in both the
  selection and the rewriting camps. (source:
  docs/topics/reference/schedules-and-annealing-literature.md;
  docs/topics/staging/rewritten-anneal-slice.md)
- **Benchmark-contamination survey** (2503.17793) — 1–45% contamination across
  benchmarks, inflation up to 14% C-Eval / 7% HellaSwag (report's numbers). A hygiene
  constraint on any released suite and its eval tables. (source:
  docs/topics/reference/schedules-and-annealing-literature.md, second entry)
- **Temperature sampling vs. scalarization on imbalanced mixtures** (2410.04579) —
  temperature sampling has lower gradient variance, converges faster, overfits more;
  proposes a *mixture-level* "cooldown" (heavy upsampling early, reduced later), which
  is a distinct axis from LR cooldown and a possible confound in WSD-opt-4. (source:
  docs/topics/reference/schedules-and-annealing-literature.md, third entry)
- **Large-scale curriculum-learning study, Zhang et al. 2025** (2506.11300) — 0.5–1B
  models; easy→hard warmup by compression ratio / lexical diversity / readability
  improves early and mid convergence with lasting gains up to +3.5%, ordering
  disentangled from selection. Bears on whether branch effects are ordering effects.
  (source: docs/topics/reference/schedules-and-annealing-literature.md)
- **Influence-driven curricula** (2508.15475) — rank by gradient-similarity influence;
  >10 pp over random in low-resource pretraining. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- **Curriculum cluster carried as leads** (2405.07490; 2406.19853; 2411.02337; ADCL
  2505.08364 "difficulty shift") — largely post-training, kept only as ordering leads.
  (source: docs/litreview/citation-verification-ledger.md;
  docs/topics/reference/schedules-and-annealing-literature.md third entry)

**The rewritten-vs-selected anneal slice (staged variant of WSD-opt-4, 2026-08-22)**

- **SwallowCode / SwallowMath** (2505.02881) — "transform-and-retain" refinement of
  Python and math; quoted +17.0 HumanEval / +12.4 GSM8K. The staging note records these
  numbers are for *full pretraining* on the rewritten corpus, not an anneal slice, so the
  anneal-slice version is untested on the record. (source:
  docs/topics/staging/rewritten-anneal-slice.md;
  docs/topics/reference/schedules-and-annealing-literature.md)
- **ProX, "programming every example"** (2409.17115) — a 0.3B model emits per-document
  refinement programs; the cheapest route to producing a rewritten slice at DataDecide's
  smallest scales. (source: docs/topics/staging/rewritten-anneal-slice.md)
- **FinerWeb-10BT** (2501.07314) — line-level LLM filtering (GPT-4o-mini labels →
  DeBERTa; "25% faster to target"). (source:
  docs/topics/staging/rewritten-anneal-slice.md)
- The staging doc's own framing: selection changes the mixture, rewriting keeps the
  mixture and changes per-document quality, so the pair separates the two — the confound
  WSD-opt-4 as written leaves entangled. Also connects a rewritten slice to the
  syntheticity feature in recipe featurization. (source:
  docs/topics/staging/rewritten-anneal-slice.md)

**Post-training from branch endpoints (WSD-opt-2 / WSD-3)**

- ***Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining***, Zhao,
  Meterez et al. (COLM 2025; 2504.07912) — trains models from scratch on controlled
  pretraining mixtures then compares PPO, GRPO and Expert Iteration across scales, and
  argues controlled small-model proxies yield real insight into RL behavior. Named in §4
  as "the nearest existing design" to post-training from this suite's branch endpoints.
  (source: docs/potential-projs/wsd-suite.md §4 2026-08-18;
  docs/topics/reference/pretraining-to-posttraining.md)
- ***Similar Models Learn Differently: Final-Window Pretraining Shapes Post-Training
  Beyond SFT*** (2607.25063) — models that look similar after SFT diverge under identical
  post-training depending on late-pretraining interventions; the late-window version of
  WSD-opt-2, and cited as explicitly citing Dohare's plasticity paper. (source:
  docs/topics/reference/pretraining-to-posttraining.md; docs/potential-projs/wsd-suite.md §4)
- ***Front-Loading Reasoning*** , Akter et al. (NVIDIA, ICLR 2026; 2510.03264) — in
  controlled 8B experiments broad diverse reasoning data helps most in pretraining while
  SFT benefits from a smaller curated long-CoT set. (source:
  docs/topics/reference/pretraining-to-posttraining.md)
- ***Early Data Exposure Improves Robustness to Subsequent Fine-Tuning***, Feng et al.
  (2605.12705) — moving target-domain data into pretraining improves retention after
  fine-tuning even at similar immediate post-training performance. (source:
  docs/topics/reference/pretraining-to-posttraining.md)
- ***The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data***, Baek et al.
  (2603.16177) — early domain exposure can be more durable than the same data only in
  fine-tuning; the repetition schedule decides generalize/overfit/forget. (source:
  docs/topics/reference/pretraining-to-posttraining.md)
- ***Understanding Reasoning from Pretraining to Post-Training***, Shen et al.
  (2607.16097) — lower pretraining loss strongly predicts higher post-RL pass@1 at fixed
  RL compute; the local RL-improvement slope grows with log pretraining tokens. Recorded
  as in tension with the beyond-final-loss hypothesis. (source:
  docs/topics/reference/pretraining-to-posttraining.md)
- **The "post-training did nothing" cluster** — *A Sober Look at Progress in Language
  Model Reasoning* (2504.07086); *Spurious Rewards* (2506.10947); Yue et al. *Does RL
  Really Incentivize Reasoning Capacity…* (2504.13837); Wu & Choi *On the Limits of
  RLVR*; counterpoints *The Invisible Leash* (2507.14843) and *RLVR Implicitly
  Incentivizes Correct Reasoning* (2506.14245). These reframe the earlier negative result
  that WSD-opt-2 would retest from genuinely stable-phase starting points. (source:
  docs/topics/reference/pretraining-to-posttraining.md)
- **Small-model post-training cautions** — Chen et al. (2505.17988; small-scale SFT on
  Qwen2.5-1.5B *reduces* MATH-500 from 23.8% to 18.4% while eliciting reasoning style);
  Luo et al. *Through the Valley* (2506.07712; Long CoT Degradation from error
  accumulation). Relevant because branch endpoints here are 150M–300M. (source:
  docs/topics/reference/pretraining-to-posttraining.md)
- **DataDecide**, Magnusson et al. (Ai2, ICML 2025; 2504.11393) — the suite this project
  retrains a subset of: 25 corpora, sizes to 1B, 3 seeds, 100B tokens; single-150M
  ranking predicts the 1B best dataset ~80% of the time; continuous likelihood metrics as
  low-noise proxies. (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/litreview/citation-verification-ledger.md)
- **Tulu / Tulu 3** (no ID on record) — the SFT data of the earlier project whose null
  result WSD-opt-2 revisits; Danielle's first-hand account is the fact of record.
  (source: docs/topics/reference/pretraining-to-posttraining.md, undated ~2026 entry)
- **FollowIR** (2403.15246) — offered by a respondent as the "AI2 dataset by Kyle"; the
  intake note calls the identification a guess, not a finding, and leaves the question
  open. Listed only so it is not re-derived. (source:
  docs/topics/reference/pretraining-to-posttraining.md)

**The MoE sibling suite (2026-08-21 §4 note)**

- **Slicing-and-Dicing MoE repo** (no ID on record) — a working MoE pretraining repo at
  the relevant scale range with validated configs and a principled default architecture
  (fix expert size by active params, dropless routing, ignore second-order knobs);
  removes the "standing up MoE infra" risk for a counterpart of this suite. (source:
  docs/potential-projs/wsd-suite.md §4 2026-08-21; docs/potential-projs/moe-recipe-suite.md)
- **FLAME-MoE** (no ID on record) — "DataDecide-for-MoE": seven models 38M–1.7B active,
  64 experts, top-8, open code/data/checkpoints/routing logs/evals; early-emerging expert
  specialization. (source: docs/topics/reference/moe-literature.md)
- **OLMoE, *Open Mixture-of-Experts Language Models*** (no ID on record) — router
  saturation as the field's existing dynamics metric; deeper layers saturate faster.
  (source: docs/topics/reference/moe-literature.md)
- **The recorded gap**: no public *multi-recipe* MoE suite exists — FLAME-MoE is a scale
  ladder on one recipe, OLMoE one recipe, OpenMoE one recipe, and the 2025–26 open-weights
  wave is closed-data. Recorded as unverified claims. (source:
  docs/topics/reference/moe-literature.md; docs/potential-projs/trajectory-statistics.md §4)
- **MoE merging caution** — checkpoint merging on MoE averages mismatched experts into
  mush without an expert-alignment step; re-basin for MoE is immature. Constrains any
  merge-based readout on an MoE counterpart. (source:
  docs/topics/reference/moe-literature.md)

**The small-scale pilot substrate (DataDecide-dense, 2026-08-22 §4)**

- **DataDecide-dense** (staging doc, no external ID) — a many-seed, densely-checkpointed,
  fully-logged retrain of a few recipes at the 2–4 smallest scales, with cosine and WSD
  arms; Danielle's own statement that WSD is what makes it worth the cost. Records the
  design cautions this suite inherits: tuning parity for the stable-phase LR (a small
  sweep at one scale, a stated transfer rule, sensitivity reported); a pilot-first
  sequence (parity reproduction → WSD twin → 2–3 branch points with a decay length/shape
  sweep → freeze the spec → fan out); pinning whether branches consume the parent's data
  stream or fresh/replayed data; emitting the results store's `variant` schema.
  (source: docs/topics/staging/datadecide-dense.md;
  docs/topics/reference/datadecide-data-pipeline.md 2026-08-22)
- **Regularization citations for multi-epoch small-scale runs** — Xue et al. 2305.13230
  and Muennighoff et al. 2305.16264, cited in the dense-substrate spec because at the
  smallest scales the fixed corpus is seen for multiple epochs. (source:
  docs/topics/staging/datadecide-dense.md;
  docs/topics/reference/regularization-literature.md)
- **u-µP / parametrization-and-HP-transfer option** (no ID on record) — a recorded,
  undecided design option for the retrain: width transfer plus a ~9-run independent HP
  sweep per recipe, at the cost of departing from DataDecide's per-size hand-set
  hyperparameters. (source: docs/topics/staging/datadecide-dense.md;
  docs/topics/reference/parametrization-and-hp-transfer.md)
- **Checkpoint-spacing table** (repo measurement, not literature) — released spacing is
  ~1,000–1,300 steps from 8M to 530M with 30–40 checkpoints at 150M–530M, and 5–12
  points below 20M; the quantitative case for retraining at small scale rather than
  reusing releases. (source: docs/open-questions-answered.md 2026-08-21)

**Branch-as-instrument framings the suite would supply substrate for**

- **Checkpoint tomography** (staging, no external ID) — the decay branch as one of four
  (later five) short-branch probes: decay branch → wall height; hot branch → diffusion
  width; twin branches → sibling barrier / basin commitment; data-shifted branch →
  component responsiveness U_c(t); plus a reset branch. Contains the sharpest statement
  on record of what is established ("branch + decay + measure the loss drop") versus not
  (doing it on cosine mid-run checkpoints; per-token profile as the statistic). A WSD
  suite makes the first of those a non-issue. (source:
  docs/topics/staging/checkpoint-tomography.md)
- **Frankle et al., *Linear Mode Connectivity and the Lottery Ticket Hypothesis*** (no ID
  on record) — the twin-branch instability probe; spawn two children from a checkpoint,
  measure the interpolation barrier; the step at which it collapses is a commitment
  clock. Caveats recorded: originals train children to completion, mostly pre-LLM vision
  work, never run across data recipes. (source:
  docs/topics/staging/checkpoint-tomography.md; docs/topics/reference/landscape-literature.md)
- **Devinterp / local learning coefficient (Lau, Murfet et al.; Timaeus)** (no ID on
  record) — short SGLD chains around a checkpoint estimate local degeneracy; tracked
  across Pythia-style checkpoint sequences and reported to detect developmental
  transitions; the cheapest "point at movement" statistic and a named prior-art community
  to check. (source: docs/topics/staging/checkpoint-tomography.md)
- **Critical-sharpness statistic across public pretraining/mid-training checkpoints** and
  **the basin-emergence line** (no IDs on record) — single-checkpoint geometry probes
  offered as covariates alongside branch statistics. (source:
  docs/topics/staging/checkpoint-tomography.md)
- **PolyPythias** (2503.09543) — 50 runs, 9 seeds × 5 sizes, ~7k checkpoints; the
  many-seed substrate named for the reset-branch probe and, more broadly, the existence
  proof that multi-seed released suites get built. (source:
  docs/topics/staging/checkpoint-tomography.md; docs/potential-projs/embedding-reset-dynamics.md)
- **Critical periods (TACL doi:10.1162/tacl_a_00725)** — data-side stage interventions;
  paired in the tomography note with weight-side resets, and the landscape file's reading
  that the critical period is the window before basin commitment, which a WSD stable
  phase would let one probe at any point. (source:
  docs/topics/staging/checkpoint-tomography.md; docs/topics/reference/critical-periods.md;
  docs/topics/reference/landscape-literature.md)

**Stage-dependent data value (the 2026-08-21 reframing of WSD-opt-4)**

- The §4 note that a branch at step t with a component injected at some mixing weight is
  "a causal probe of [the component's value as a function of training time] — a factorial
  component × injection-time experiment where each cell costs ~10% of a training run, not
  a full run", i.e. WSD-opt-4 is not scope creep under that framing. Full discussion in
  the functional-featurization doc. (source: docs/potential-projs/wsd-suite.md §4
  2026-08-21; docs/potential-projs/functional-featurization.md)
- **TREC's receptivity valley** (2509.25380, above) is recorded as the candidate
  explanation for stage-dependent chunk effects, i.e. the theory a branch grid tests.
  (source: docs/topics/reference/schedules-and-annealing-literature.md)

**Adjacent methodology this suite's outputs would feed or be judged against**

- **Heineman et al., *Signal and Noise*** (NeurIPS 2025 per the record; 2508.13144 per
  the ledger, a Claude-added and therefore hallucination-prone row) — the noise framework
  and the ~900K-result release including OLMo, DataDecide and ladder checkpoints; the
  seed-noise floor any "the branch endpoint differs" claim is tested against. (source:
  docs/topics/reference/evaluation-methodology-literature.md;
  docs/litreview/citation-verification-ledger.md)
- **OLMES, *A Standard for Language Model Evaluations*** (no ID on record) — the eval
  standard whose table schema the results store mirrors; also the basis for the recorded
  finding that re-evaluating a fixed checkpoint with new seeds buys nothing. (source:
  docs/topics/reference/evaluation-methodology-literature.md)
- **Rho-1, *Not All Tokens Are What You Need for Pretraining*** (no ID on record) —
  loss-trajectory token taxonomy across checkpoints; the per-token readout the frozen
  held-out set exists to support on branch starts and endpoints. (source:
  docs/topics/reference/token-level-literature.md)
- **Token-level uncertainty decomposition** (*Token-Level Uncertainty-Aware Objective for
  Language Model Post-Training*, no ID on record) — epistemic vs. aleatoric token
  uncertainty, with epistemic draining faster for low-aleatoric tokens; the
  interpretation frame for per-token decay response on branch endpoints. (source:
  docs/topics/reference/token-level-literature.md)
- **Grokking (Power et al.) and progress measures (Nanda et al.)** (no IDs on record) —
  the decay branch read as an "anti-grokking instrument" that reveals accumulated hidden
  river progress, and the warning that matched-loss pairs are a necessary-but-insufficient
  control. (source: docs/topics/reference/grokking-and-hidden-progress.md)
- **Nakkiran et al., deep double descent** (no ID on record) — capability is not monotone
  in training loss along a run, a boundary condition on reading branch-endpoint loss
  drops as progress. (source: docs/topics/reference/grokking-and-hidden-progress.md)
- **Pre-/mid-training/RL interplay study** (2512.07783) — from Danielle's SciSpace
  midtraining review; the fixed-compute comparison is a framing precedent for a
  changed-mixture decay branch. The review is noted to have surfaced little else
  LM-specific and to have missed the LM midtraining canon. (source:
  docs/topics/reference/targeted-pretraining-midtraining-literature.md;
  docs/topics/README.md)
- **Additional midtraining-review rows tagged for this cluster**: 2506.20512
  (mid-training data line incl. OctoThinker; Claude-added row) and 2306.12070
  (task-robust minimax pretraining). (source:
  docs/litreview/citation-verification-ledger.md)

**Standing caveats to carry**

- The entire annealing accumulator is agent-generated and marked unverified in its own
  file; three separate agent answers to the same annealing-data question are on file and
  only the Oct-2025 survey is judged usable. (source:
  docs/topics/reference/schedules-and-annealing-literature.md)
- Explicit **do-not-re-flag collision list**: instruction-tuning / continual-learning
  papers (VersaTune 2411.11266, O-LoRA, Self-Synthesized Rehearsal 2403.01244, KILO
  2508.03571, LIFT 2312.11508, Mixture-of-Skills 2406.08811, instruction-mix studies
  2310.05492 / 2312.10793, plus forgetting rows 2501.00237, 2405.17830, 2308.08747,
  2410.10210); Annealed-RLVR 2509.23629 (SFT "heating" inside RLVR); RLHFuse (simulated
  annealing for pipeline scheduling); the 2020 "data annealing" paper 2004.13833
  (formal→informal BERT). None of these is the pretraining→annealing transition. (source:
  docs/topics/reference/schedules-and-annealing-literature.md working list and third
  entry; docs/litreview/citation-verification-ledger.md)
- Known drift in the second annealing report: PDPC misattributed to the YuLan-Mini team
  with a Phi-4 link; the "web 50→20% / synthetic 5→35%" proportions are illustrative with
  no source; "edu filtering works best during annealing" is a respondent inference;
  several sources are a Reddit thread and a blog mirror. (source:
  docs/topics/reference/schedules-and-annealing-literature.md, second entry)

**NBLM pretraining-dynamics notebook additions (intake 2026-08-24;
agent-generated, no IDs supplied):**

- **MPL-optimized schedules + "sqrt-cube" decay** — the MPL's gradient-based
  schedule search beats cosine and WSD; symbolic regression finds
  η_t ≈ η_max·(1−τ)^1.5 as the efficient WSD decay shape — a concrete candidate
  arm for schedule comparisons — (source:
  `../../topics/reference/schedules-and-annealing-literature.md`, 2026-08-24
  NBLM entry)
- **PTQ robustness as a consequence-of-anneal axis** (no ID) — quantization
  error spikes exactly at LR decay; stable phase PTQ-flat; souping/LAWA
  mitigate — a new measurable the anneal trades against — (source: same)
- **Power Lines** (no ID) — optimal AdamW timescale as a power law of D/N;
  B_opt/B_crit scale with D alone — fixes-by-formula for sweep design —
  (source: `../../topics/reference/parametrization-and-hp-transfer.md`,
  2026-08-24 entry)
- **Step Law** (no ID) — (LR, BS) loss landscape strictly convex; optimal LR
  joint in (N, D), optimal BS in D — same role — (source: same)
- **CPT learning-dynamics scaling law** (no ID) — decouples distribution shift
  from LR annealing; replay ratio and peak LR targetable — relevant to
  anneal-phase data-swap arms — (source:
  `../../topics/reference/schedules-and-annealing-literature.md`, same entry)
- **Mid-Training survey** (no ID) — taxonomy of the stage the anneal-slice arms
  live in (data curation + WSD annealing + long-context) — (source: same)

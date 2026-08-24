# intervention grid — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`intervention-grid.md`](../intervention-grid.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*Recall corpus for GRID (intervention grid). Highest-recall enumeration of every paper,
method, or named prior-art item on record anywhere in this repository that could plausibly
bear on GRID. One line per item; recall is the point, so marginal items are kept. Every item
carries its repo source. **Nothing here is verified** — most entered through the 2026-08-18
Research Trajectory conversations, the 2026-08-22 reinit literature pass, and other
agent-generated records; those are marked in-line. No positioning or novelty claims.*

**The founding cells — the three claimed history effects, and the control**

- **Achille, Rovere & Soatto, *Critical Learning Periods in Deep Networks*** (ICLR 2019, no
  ID on record) — blur/downsample deficit early in training causes permanent impairment
  depending on deficit onset and length; mechanism = Information Plasticity, measured by the
  Fisher trace (rises early, then decreases); the vertical-flip high-level-deficit control
  shows no permanent damage. GRID-2's second reproduction cell and GRID-4's template —
  (source: docs/topics/reference/critical-periods.md; docs/potential-projs/intervention-grid.md §4).
- **Ash & Adams, *On Warm-Starting Neural Network Training*** (NeurIPS 2020, no ID) —
  *stationary* incremental setting: a single dataset arriving in chunks, warm-started models
  generalize worse than re-initialized ones at similar training loss; shrink-and-perturb is
  the fix; the diagnosis (gradient-norm imbalance between old and new samples) is recorded as
  "a symptom, not a mechanism." GRID-2's first reproduction cell —
  (source: docs/topics/reference/plasticity.md; docs/potential-projs/intervention-grid.md §4).
- **Igl, Farquhar, Luketina, Böhmer & Whiteson, ITER / *Transient Non-Stationarity and
  Generalisation in Deep RL*** (ICLR 2021, arXiv 2006.05826) — a memory effect where
  transient non-stationarities permanently impact the latent representation; fixed by
  periodically distilling into a freshly initialised network; ITER's students generalize
  *better* than their teachers. GRID's third founding cell and the source of GRID-4's
  distill-into-fresh-network arm —
  (source: docs/topics/reference/reinit-and-transfer-literature.md; nonstationarity-accounting.md;
  docs/potential-projs/intervention-grid.md §4).
- **Rothermel, Li, Rocktäschel & Foerster, *Don't Sweep your Learning Rate under the Rug***
  (2021, arXiv 2107.12460; ICML 2021 SSL workshop — the one repo citation confirmed as
  Danielle's own) — an apparent history effect (frozen pretrained structure "sufficing") that
  vanished under fair LR tuning; recorded as "the founding example of the opposite outcome"
  and the fourth cell — (source: docs/topics/reference/reinit-and-transfer-literature.md;
  docs/potential-projs/intervention-grid.md §4).
- **Lu et al. 2021, *Pretrained Transformers as Universal Computation Engines*** (arXiv
  2103.05247; "Frozen Pretrained Transformer") — the claim 2107.12460 rebutted; the
  incumbent side of the control cell — (source: docs/topics/reference/reinit-and-transfer-literature.md).
- **Igl's own supervised CIFAR variants (label noise, dataset-size interventions)** — the
  §4 record claims the repo contains them, letting all three phenomena run in the supervised
  setting with one backbone ("a 3× cost reduction with no loss of the vocabulary claim").
  *Explicitly flagged unverified in the doc* —
  (source: docs/potential-projs/intervention-grid.md §4 practical-plan entry).
- **Critical Learning Periods for Multisensory Integration in Deep Networks** (no ID) —
  critical periods arise from "complex, unstable early transient dynamics which are decisive
  of the final performance and learned representations" —
  (source: docs/topics/reference/critical-periods.md).
- **Critical periods in LM finetuning** (TACL, doi:10.1162/tacl_a_00725) — the LM-side
  critical-period result; the closest existing work to GRID-opt-1's LM diagonal —
  (source: docs/topics/reference/reinit-and-transfer-literature.md; staging/checkpoint-tomography.md).

**Competing explanations of the warm-start gap (what GRID-3's factorial adjudicates)**

- **DASH: *Warm-Starting Neural Network Training in Stationary Settings without Loss of
  Plasticity*** (NeurIPS 2024, no ID) — theory for the stationary case: the model memorized
  *noise* from the small early dataset, so shrinking should be direction-aware; argues
  non-stationarity-motivated plasticity fixes are **ineffective** in the stationary setting,
  i.e. Dohare/Lyle mechanisms may not explain this gap —
  (source: docs/topics/reference/plasticity.md; docs/potential-projs/intervention-grid.md §4).
- ***What Can Grokking Teach Us About Learning Under Non-Stationarity*** (2025, no ID) —
  re-warming the **effective learning rate** closes the generalization gap, and a higher
  relative number of dead units does *not* predict a large warm-starting gap; the sharpest
  live hypothesis and the ELR arm of GRID-3 —
  (source: docs/topics/reference/plasticity.md; grokking-and-hidden-progress.md;
  docs/potential-projs/intervention-grid.md §4).
- **The mundane candidates** (no papers): optimizer-state reset, weight decay, warmup, AdamW
  vs. 2020-era setups, normalization variant, epochs-per-chunk, ViT vs. CNN — the
  modernization axes of GRID-3 — (source: docs/potential-projs/intervention-grid.md §1, §4).
- **The accounting reading**: modern recipes re-warm the LR whenever data arrives (continued
  pretraining does this by construction), normalization changed effective-LR dynamics, and
  single-epoch LLM training barely lets you memorize noise — three stabilizers whose effect
  on a once-famous gap has never been decomposed —
  (source: docs/topics/reference/nonstationarity-accounting.md).
- **The record's own gap statement**: "Nobody has run the factorial that adjudicates them on
  the original benchmark" (agent-supplied, unverified) —
  (source: docs/topics/reference/plasticity.md; docs/potential-projs/intervention-grid.md §4).

**The plasticity diagnostic panel (the standard logging layer)**

- **Dohare et al., *Loss of plasticity in deep continual learning*** (Nature 2024; earlier
  arXiv "Maintaining Plasticity…", 2306.13812) — plasticity loss until networks learn no
  better than a shallow net; **continual backpropagation** (selective reinit of
  dormant/unuseful units); the incremental-CIFAR experiments are in their codebase —
  (source: docs/topics/reference/plasticity.md; critical-periods.md).
- **Lyle et al., *Understanding Plasticity in Neural Networks*** (ICML 2023, arXiv 2303.01486)
  — plasticity loss deeply connected to loss-landscape **curvature**, often without saturated
  units; "no single statistic — curvature comes closest" —
  (source: docs/topics/reference/plasticity.md).
- **Lyle et al., *Disentangling the Causes of Plasticity Loss*** (arXiv 2402.18762) —
  follow-up — (source: same).
- **Hernandez-Garcia, Figliolia & Millidge, *Can Scale Save Us From Plasticity Loss in
  LLMs?*** (arXiv 2606.24752, 2026) — 5M–314M; sublinear scaling law; scale delays but does
  not prevent plasticity loss, in continual *and stationary* settings; bears directly on
  GRID-opt-1's LM diagonal — (source: docs/topics/reference/plasticity.md;
  reinit-and-transfer-literature.md).
- **Nikishin et al., *Deep RL with Plasticity Injection*** (arXiv 2305.15555) — "the most
  directly borrowable instrument": if injection helps, plasticity was the binding constraint;
  a candidate GRID-4 arm and a panel diagnostic —
  (source: docs/topics/reference/plasticity.md; reinit-and-transfer-literature.md).
- **Zaidi, Berariu, Kim, Bornschein, Clopath, Teh & Pascanu, *When Does Re-initialization
  Work?*** (arXiv 2206.10011) — >15,000 vision models: reinit helps without other
  regularization, little once regularization is tuned, significantly under label noise — the
  "is the effect a tuning artefact" precedent for every GRID arm —
  (source: docs/topics/reference/plasticity.md; reinit-and-transfer-literature.md).
- **Panel statistics themselves** (methods, no papers): curvature, feature rank, dead units,
  weight norm, gradient-norm ratio, Fisher trace — logged **at matched training loss** —
  (source: docs/topics/reference/plasticity.md; docs/potential-projs/intervention-grid.md §1).
- **Achille–Soatto Information Plasticity as the panel's ancestor** — "Information Plasticity
  decreases and the network can no longer adapt" (2017–2019) *is* plasticity loss five years
  before Dohare, with the Fisher trace as diagnostic; since Fisher ≈ Hessian for log-loss,
  Lyle's curvature finding is "nearly a rediscovery in different coordinates" —
  (source: docs/topics/reference/critical-periods.md; plasticity.md).
- **Task2Vec (Achille et al.)** (no ID) — the same Fisher formalism pointed at data;
  potential deficit-characterization tool — (source: docs/topics/reference/critical-periods.md).
- **Reset-family additions from the 2026-08-22 pass**: spectral collapse (arXiv 2509.22335);
  activation-function design (arXiv 2509.22562); calibrated partial resets (arXiv 2607.24996);
  plasticity-loss survey in RL (arXiv 2411.04832 — last-layer resets standard, the
  "concentrated in last layers" belief weakly evidenced); Fisher-guided selective forgetting
  (arXiv 2502.00802); Reset & Distill (Ahn et al., arXiv 2403.05066); representation-plasticity
  timeline in LLMs (arXiv 2410.06225) —
  (source: docs/topics/reference/plasticity.md; reinit-and-transfer-literature.md).

**Period-reopening interventions (GRID-4's arms)**

- **Shrink-and-perturb** (Ash & Adams' own fix, no separate ID) — also cited in the repo as an
  existence proof in the "paradigm evidence" sense —
  (source: docs/topics/reference/plasticity.md; evaluation-methodology-literature.md).
- **Effective-LR re-warming** (from the grokking-under-non-stationarity paper) —
  (source: docs/topics/reference/plasticity.md).
- **Continual Backprop: SGD with Persistent Randomness** (no ID) — named as one of the
  interventions that "artificially reopen the period"; Sutton's gloss was that it "at least
  shows the problem can be solved" — (source: docs/topics/reference/critical-periods.md;
  evaluation-methodology-literature.md).
- **Distill-into-fresh-network (ITER)** — with the **undamaged-teacher control** the original
  used, so damage is measured relative to a clean distillation rather than to the teacher;
  the contrast that splits "geometry damage" from "knowledge damage" —
  (source: docs/potential-projs/intervention-grid.md §4 2026-08-18 ITER entry).
- **The basin reading of resets**: interface resets are basin-*preserving*, early-training
  deficits are basin-*determining*; distillation into a fresh network is the one reset that
  leaves the basin by construction. Recorded as an interpretive frame with **no paper behind
  it** (gap G3) — (source: docs/topics/reference/landscape-literature.md;
  reinit-and-transfer-literature.md).

**Commitment clocks and their precedents (GRID-6's instruments)**

- **Frankle et al., *Linear Mode Connectivity and the Lottery Ticket Hypothesis*** (no ID) —
  sibling runs from a shared init become linearly connected only after a critical number of
  steps; the canonical basin-commitment clock —
  (source: docs/topics/reference/landscape-literature.md; identifiability-literature.md;
  staging/checkpoint-tomography.md).
- **Twin-branch probe (checkpoint tomography instrument 2)** — spawn two children from the
  same checkpoint with different SGD noise/data order, train both, measure the barrier; the
  causal, controllable version of the sibling clock. Caveats recorded: the original trains
  children to completion, short-child variants are not standardized, it is mostly pre-LLM
  vision work, and nobody has run it across data recipes —
  (source: docs/topics/staging/checkpoint-tomography.md).
- **Fisher-trace peak** (Achille et al.) — the original mechanism, now one hypothesis among
  several; "adding the Fisher trace to the warm-starting panel is essentially free" —
  (source: docs/topics/reference/critical-periods.md; docs/potential-projs/intervention-grid.md §4).
- **Local learning coefficient — Watanabe (singular learning theory); Lau, Murfet et al.'s
  estimator; devinterp/Timaeus SGLD probes** (no IDs) — a per-checkpoint scalar measuring
  degeneracy of the solution neighbourhood whose jumps track training phase transitions;
  "degeneracy is the local face of non-identifiability" —
  (source: docs/topics/reference/identifiability-literature.md; staging/checkpoint-tomography.md).
- **Developmental interpretability: *Differentiation and Specialization of Attention Heads via
  the Refined LLC*; *Loss Landscape Degeneracy and Stagewise Development in Transformers***
  (no IDs) — (source: docs/topics/reference/identifiability-literature.md).
- **Fort et al., *Deep Learning versus Kernel Learning*** (arXiv 2010.15110) — rapid early NTK
  rotation then stabilization, listed as a commitment-event precedent —
  (source: docs/topics/reference/identifiability-literature.md; ntk-literature.md).
- **Git Re-Basin (Ainsworth et al.) and Entezari et al.** (no IDs) — permutation alignment; the
  raw-vs-aligned barrier *difference* is the informative quantity (raw-high/aligned-low =
  same solution class, different parameterization; aligned-high = genuine divergence) —
  (source: docs/topics/reference/identifiability-literature.md; landscape-literature.md).
- **REPAIR (activation renormalization)** (no ID) — part of the symmetry-quotienting toolkit —
  (source: docs/topics/reference/identifiability-literature.md).
- **Unveiling LMC of Re-Basin from a Neuron Distribution Perspective** (no ID) — re-basin
  often reduces barriers only marginally and works poorly early in training, with no unified
  theory of when it succeeds — a direct caution on GRID-6's aligned-barrier readings —
  (source: docs/topics/reference/landscape-literature.md).
- **Layer-wise LMC** (arXiv 2307.06966) — middle layers own the barrier; per-layer
  perturbations near-barrier-free — (source: docs/topics/reference/reinit-and-transfer-literature.md).
- **Router saturation as a fourth commitment clock — OLMoE** (no ID) — average overlap between
  top-k experts at step t and at convergence, rising sharply within the first few thousand
  steps, deeper layers saturating faster; "the cheapest to compute and the only one that's
  exactly zero/one per token" (GRID-opt-5) —
  (source: docs/topics/reference/moe-literature.md; docs/potential-projs/intervention-grid.md §4).
- **Three Phases of Expert Routing** (no ID) — balance-prioritizing → specialization →
  relaxation, a non-monotone trajectory invisible post hoc; the MoE-arm phase structure —
  (source: docs/topics/reference/moe-literature.md).
- **Continual Pre-training of MoEs: How Robust Is Your Router?** (no ID) — early-layer routing
  reorganization as a possible forgetting mechanism; the MoE version of a warm-start scar —
  (source: docs/topics/reference/moe-literature.md).
- **Single-checkpoint complements** (methods): a scalable critical-sharpness statistic
  (progressive sharpening at scale, applied to data-mixing decisions) and
  perturbation-resilience / basin-emergence measures — "useful as covariates" —
  (source: docs/topics/staging/checkpoint-tomography.md).

**Functional-identifiability tests (GRID-8's Stage-2 readouts)**

- **Roeder, Metz & Kingma 2021, *On Linear Identifiability of Learned Representations*** (no
  ID) — fit the optimal linear map between two models' representations; the residual is the
  identifiability gap — (source: docs/topics/reference/identifiability-literature.md).
- **Model stitching — Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021** (no IDs) — if a
  trained adapter lets A's bottom half drive B's top half at low penalty they are functionally
  interchangeable at that depth; treated as ground truth —
  (source: docs/topics/reference/identifiability-literature.md; reinit-and-transfer-literature.md).
- **CKA (Kornblith et al.)** (no ID) — the scalable proxy, with the caveat that it can be
  dominated by a few directions and disagree with stitching —
  (source: docs/topics/reference/identifiability-literature.md).

**The identifiability frame (GRID-8's formalism)**

- **Hyvärinen & Pajunen 1999, *Nonlinear ICA: Existence and Uniqueness Results*** (no ID) —
  latents are fundamentally non-identifiable from observational i.i.d. data alone —
  (source: docs/topics/reference/identifiability-literature.md).
- **Hyvärinen's time-contrastive learning / auxiliary-variable nonlinear ICA; Khemakhem et
  al., iVAE (2020)** (no IDs) — identifiability restored by auxiliary variables,
  non-stationarity, or multiple environments, up to a residual group — (source: same).
- **Schölkopf et al., *Toward Causal Representation Learning*** (2021, no ID) — (source: same).
- **Brehmer et al. 2022, *Weakly Supervised Causal Representation Learning***; **Ahuja et al.,
  *Interventional CRL***; **von Kügelgen et al. 2021** (no IDs) — interventions buy
  identifiability; the sparse-mechanism-shift principle — (source: same).
- **Sussmann 1992** (no ID) — single-hidden-layer uniqueness up to permutation/sign; plus ReLU
  positive rescalings and attention-head permutations as the known symmetry group —
  (source: same).
- ***Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability***
  (2026, no ID) — consistent assignment of features to neurons across seeds, with
  symmetry-breaking mechanisms — (source: docs/topics/reference/identifiability-literature.md;
  landscape-literature.md).
- **Huh et al. 2024, *Position: The Platonic Representation Hypothesis*** (no ID) —
  representations of independently trained models converge with scale; read in-repo as "an
  empirical claim that identifiability improves with scale," hence a scale ceiling on
  path-dependence — (source: docs/topics/reference/identifiability-literature.md).
- **The honesty note on record**: CRL identifies latents *in data* from interventions *on
  data*, whereas GRID identifies structure *in training dynamics* from interventions *on
  training* — "same epistemic logic, different object" — (source: same).
- **Assembled core claim**: the critical period as an *identifiability phase transition*,
  with a per-paper sub-claim for Achille, Ash & Adams, and Igl —
  (source: docs/topics/reference/identifiability-literature.md).
- **MoE expert assignment as the concrete non-identifiable latent; *The Myth of Expert
  Specialization in MoEs*** (no ID) — independently trained MoEs select unrelated
  specialization solutions, "about as clean an existence proof of solution-class
  underdetermination as the field has produced"; the symmetry group now includes expert
  permutations, breaking dense comparability tools —
  (source: docs/topics/reference/identifiability-literature.md; moe-literature.md).

**Behavioral evidence that solution classes are real**

- **Juneja et al., *Linear Connectivity Reveals Generalization Strategies*** (ICLR 2023, no
  ID) — fine-tuned models cluster into distinct linearly-connected basins that implement
  *different generalization strategies* (heuristic vs. syntax-sensitive on NLI) at similar
  in-distribution accuracy; recorded as "the NLP existence proof" —
  (source: docs/topics/reference/landscape-literature.md; identifiability-literature.md).
- **Implicit-bias / simplicity-bias and shortcut-learning literatures** (no IDs) — SGD selects
  non-uniformly; shortcut learning as selection pathology; "what's missing is *when*
  selection happens" — (source: docs/topics/reference/identifiability-literature.md).
- **Model soups (Wortsman et al.)** and ***On the Emergence of Cross-Task Linearity in the
  Pretraining-Finetuning Paradigm*** (no IDs) — merging works only within a basin; the
  positive-space evidence that basin membership is functionally load-bearing —
  (source: docs/topics/reference/landscape-literature.md).

**Non-stationarity accounting (the mechanism-pillar frame around the grid)**

- **The exogenous/endogenous split** (framework, no paper) — exogenous: LR schedule,
  data-order and realized-composition drift, midtraining; endogenous: routing, and the
  implicit self-curriculum by which gradient magnitude reweights the effective distribution.
  "From the gradient's perspective, pretraining was never stationary" —
  (source: docs/topics/reference/nonstationarity-accounting.md).
- **"Three communities, one claim"** — Achille 2019 (vision), Ash & Adams 2020 (supervised),
  Igl 2021 (RL) state the same claim in a two-year window, "none citing the others' framing"
  (agent-supplied, unverified) — (source: docs/topics/reference/nonstationarity-accounting.md).
- **Fixes read as stabilizers** — ITER removes the non-stationarity's *history* rather than
  suppressing the non-stationarity; balancing loss, EMA, decay, warmup as the general family;
  the open question of what suppression costs — (source: same).

**Re-tuning precedents (GRID-opt-4's meta-analysis, and the program frame)**

- **Melis, Dyer & Blunsom 2018, *On the State of the Art of Evaluation in Neural Language
  Models*** (no ID) — LSTM-vs-transformer conclusions inverted under equalized tuning
  budgets; "the pre-2021 ancestor of the headline-phenomenon-is-a-tuning-artifact
  finding-shape" — (source: docs/topics/reference/evaluation-methodology-literature.md;
  docs/research-hypothesis.md).
- **Sara Hooker, *The Hardware Lottery*** (no ID) — directions win because the surrounding
  stack co-evolved; Danielle's software-stack version underlies the
  matched-budget-is-impossible argument — (source: same two files).
- ***Position: Lifetime Tuning Is Incompatible with Continual Reinforcement Learning*** (no
  ID) — the same complaint from the RL side — (source: docs/topics/reference/evaluation-methodology-literature.md).
- **Hochlehnert et al., *A Sober Look at Progress in LM Reasoning*** (COLM 2025, arXiv
  2504.07086) — RL on distillation-based models yields little significant gain; small
  benchmarks give unstable estimates, multi-seed essential —
  (source: docs/topics/reference/pretraining-to-posttraining.md).
- **Shao et al., *Spurious Rewards: Rethinking Training Signals in RLVR*** (ICML 2026, arXiv
  2506.10947) — random rewards improve MATH-500 by 21.4 points on Qwen2.5-Math-7B, failing
  outside Qwen families; the elicitation-variance existence proof — (source: same).
- **Yue et al.** (NeurIPS 2025 oral, arXiv 2504.13837) — RLVR improves pass@k at small k but
  does not expand the base model's reasoning boundary at large k; plus *On the Limits of
  RLVR* (no ID), *The Invisible Leash* (arXiv 2507.14843), *RLVR Implicitly Incentivizes
  Correct Reasoning* (arXiv 2506.14245) — (source: same).
- **Existence proofs as paradigm evidence**: AlexNet (Krizhevsky et al.); GPT-3 (*Language
  Models are Few-Shot Learners*); DeepSeek-R1-zero; shrink-and-perturb; continual backprop —
  the genre GRID's demonstrations belong to, with the RLVR corrections as its failure mode —
  (source: docs/topics/reference/evaluation-methodology-literature.md; docs/research-hypothesis.md).
- **Optimum displacement and historical base rates** (measurements, no papers) — how far the
  warm-start regime's tuned optimum sits from the from-scratch default, knob by knob
  (GRID-5); and a meta-analysis of how often an incumbent's advantage survives serious
  re-tuning — (source: docs/research-hypothesis.md).
- **Tuning-response curves** (method, no paper) — performance vs. search budget per paradigm
  as the falsifiable replacement for matched-budget comparisons (GRID-opt-3) —
  (source: docs/research-hypothesis.md Refinement 1).

**LM-diagonal outcome columns (GRID-opt-1 / GRID-opt-2)**

- **Olsson et al., *In-context Learning and Induction Heads*** (no ID) — induction-head
  formation as a known sharp early phase transition; the original ICL score was the loss
  difference between token positions; the LM-only outcome column —
  (source: docs/potential-projs/icl-elicitability.md).
- **The falsifiable prediction on record**: deficits spanning the induction-head transition
  should damage ICL disproportionately; a critical period for *elicitability* distinct from
  the one for performance would be a new distinction, and its absence would dissociate
  capability formation from elicitability formation —
  (source: docs/potential-projs/intervention-grid.md §4; docs/potential-projs/icl-elicitability.md).
- **Bornschein, Lyle, Pascanu et al., *Fine-Tuned In-Context Learners for Efficient
  Adaptation*** (no ID) — the plasticity group moving into ICL-vs-fine-tuning; recorded as
  the direct descendant of 2107.12460 —
  (source: docs/topics/reference/plasticity.md; reinit-and-transfer-literature.md).
- **DataDecide (Magnusson et al., arXiv 2504.11393)** — the LM data-poverty cell is "literally
  the DataDecide question at miniature scale" —
  (source: docs/potential-projs/intervention-grid.md §4; pretraining-to-posttraining.md).
- **LLM-scale echo of critical periods** — 2025–2026 data-placement results (early exposure
  shaping models more durably than late data, final-window effects, safety behaviors from
  pretraining resisting post-training removal) described as "critical-period phenomenology at
  scale, mostly published without the connection drawn" (no papers named, unverified) —
  (source: docs/topics/reference/critical-periods.md).

**Substrate, statistics, and hygiene**

- **PolyPythias (van der Wal, Lesci, Muller-Eberstein, Saphra, Schoelkopf, Zuidema,
  Biderman)** (ICLR 2025, arXiv 2503.09543) — 50 pretraining runs, 9 seeds × 5 sizes
  (14M–410M), ~7,000 checkpoints: "this is the substrate" for many-seed LM work —
  (source: docs/topics/reference/reinit-and-transfer-literature.md).
- ***The Butterfly Effect*** (arXiv 2506.13234) — trajectories highly sensitive to initial
  conditions, so reset/intervention studies need many seeds; the power-budgeting argument —
  (source: docs/topics/reference/reinit-and-transfer-literature.md).
- **Seed budgeting from a power target at month zero; n≈10 per cell beats a sprawling grid at
  n=3; "replication matches published effect within CI" as the written definition of done** —
  (source: docs/potential-projs/intervention-grid.md §4 practical-plan and harness entries).
- **Heineman et al., *Signal and Noise*** (no ID) — signal/noise framework for evaluation
  uncertainty; continuous metrics beat accuracy on both —
  (source: docs/topics/reference/evaluation-methodology-literature.md).
- **Where eval variance lives** — for OLMES-style loglikelihood evals, re-evaluating a fixed
  checkpoint with new seeds buys nothing; the variance of interest is in training (seed, data
  order, init) — (source: docs/topics/reference/evaluation-methodology-literature.md).
- **Demonstration hygiene** (standard, no paper): pre-specified settings, effect sizes with
  confidence bounds across seeds, replication in a second model family, honest reporting of
  how many settings were searched, and a mechanism readout from the panel —
  (source: docs/research-hypothesis.md Refinement 2).
- **CNN deconstruction ladder (`deconCNN`, 2025, internal)** — the CIFAR ablation-ladder
  lineage and its own recipe-ablation reading list ("Bag of Tricks" 1812.01187, "Revisiting
  ResNets" 2103.07579, "ResNet strikes back" 2110.00476; Li et al. 2018 loss-surface
  visualizations), plus the lesson that budget is a hidden ablation axis and a mid-ladder LR
  change confounded the rungs — (source: docs/past-projects/cnn-deconstruction-ladder.md).
- **Loss-slope prediction (2025, internal)** — the CNN statistics work the §4 entries call
  "now load-bearing" for the reproduction CIs — (source: docs/past-projects/loss-slope-prediction.md).

**Adjacent instruments the grid could borrow (checkpoint tomography)**

- **Decay branch as a wall-height meter** — Wen et al. branch off a constant-LR run at 20B
  tokens, decay for 5B, then interpolate; "branch + decay + measure the loss drop" is
  established — (source: docs/topics/staging/checkpoint-tomography.md;
  docs/topics/reference/landscape-literature.md).
- **Hot branch (constant-LR continuation) → diffusion width; data-shifted branch → component
  responsiveness; reset branch → recovery cost and barrier to the pre-reset model** — the
  rest of the five-probe battery, all short continuations sharing one runner —
  (source: docs/topics/staging/checkpoint-tomography.md).
- **The record's own prior-art check to do** — a targeted pass over the devinterp and
  WSD-followup communities' 2025–26 output — (source: same).

**Provenance caveats for this corpus**

- Almost everything above entered through the **2026-08-18 Research Trajectory
  conversations** and the **2026-08-22 reinit literature pass** (an Opus subagent working
  from abstracts and paper pages, no full-PDF reads and no forward-citation sweep). Both
  source files carry a standing header that related-work claims are unverified unless an
  identifier is given — (source: docs/topics/README.md; docs/topics/reference/reinit-and-transfer-literature.md).
- Two "nobody has" claims are explicitly agent-supplied: that the literature "has noticed the
  adjacency (papers on relearning cite both works side by side)" but no factorial tests
  whether the Fisher trajectory predicts which fix works; and that the three founding papers
  state one claim with "none citing the others' framing" —
  (source: docs/topics/reference/critical-periods.md; nonstationarity-accounting.md).
- The claim that Igl's repo contains supervised CIFAR variants is flagged unverified in the
  doc itself — (source: docs/potential-projs/intervention-grid.md §4).
- Only **arXiv 2107.12460** (Danielle's own paper) is recorded as confirmed; ITER's
  **2006.05826** and Lu et al.'s **2103.05247** were retrieved in the reinit pass —
  (source: docs/topics/reference/reinit-and-transfer-literature.md).

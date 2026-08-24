# moe partitions — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`moe-partitions.md`](../moe-partitions.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*Purpose: the high-recall corpus for PART (is the token taxonomy a property of the data or
the architecture?). Every item is on record somewhere in this repository; nothing is
verified and nothing here is a positioning claim. Err toward inclusion. The curated core
lives in `../moe-partitions.md` §5.*

**Closest prior art on expert specialization and its (non-)invariance**

- **The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not
  Necessarily Domain Expertise** (no ID on record) — specialization patterns resist human
  interpretation; expert overlap between different models answering the same question is no
  higher than between entirely different questions (independently trained MoEs pick
  *unrelated* specialization solutions); routers are linear maps, so hidden-state similarity
  is necessary and sufficient to explain expert-usage similarity; load-balancing loss
  provably suppresses shared hidden directions, explaining specialization collapse under less
  diverse data. The record names it as prior art for PART-5 (it predicts non-invariance
  across independent runs) and for the load-balancing caveat — (source:
  docs/topics/reference/moe-literature.md; docs/topics/reference/identifiability-literature.md;
  docs/potential-projs/moe-partitions.md §4 2026-08-18; unverified intake quote).
- **Jelassi et al., Mixture of Parrots: Experts Improve Memorization More Than Reasoning**
  (ICLR 2025; no arXiv ID on record, full text on disk) — theory: width-bounded experts
  provably cannot solve certain graph problems a slightly wider dense model solves; memory
  capacity tracks total not active parameters; synthetic: more experts help closed-book
  retrieval, not graph problems; pretrained (65B tokens, FineWeb-edu/Cosmopedia/Wikipedia):
  gains on TriviaQA/NQ with expert count, none on GSM8K/MATH/ARC. Supplies the mechanism and
  the reproduction template for PART-opt-3(a) on the sweep's 128× ratios — (source:
  docs/topics/reference/moe-literature.md 2026-08-22; docs/potential-projs/moe-partitions.md
  §4; SciSpace bundle, unverified beyond the agent summary and abstract).
- **OpenMoE analysis** (no ID on record) — routing predominantly token-ID-driven with minimal
  context relevance, assignments fixed early; the direct source of the shallow-routing
  caveat PART-3 controls for and of the "routing as a data fingerprint" reframing behind
  PART-opt-4 — (source: docs/potential-projs/moe-partitions.md §4 2026-08-21;
  docs/topics/reference/nonstationarity-accounting.md).
- **OLMoE — Open Mixture-of-Experts Language Models** (no ID on record) — router saturation
  as the field's existing commitment metric; fully open with intermediate checkpoints and
  analyzable routing, named as observational breadth / validation set for PART-opt-1 —
  (source: docs/topics/reference/moe-literature.md).
- **Three Phases of Expert Routing: How Load Balance Evolves During MoE Training** (no ID on
  record) — surge/stabilization/relaxation across OLMoE and OpenMoE checkpoints; the record
  calls it aggregate-level, so a per-token, data-linked partition analysis is the contrast —
  (source: docs/topics/reference/moe-literature.md; docs/potential-projs/moe-partitions.md §4).
- **Continual Pre-training of MoEs: How Robust Is Your Router?** (no ID on record) — routing
  changes most in early layers; evidence that the discovered partition is not fixed under
  distribution shift, relevant to PART-5's stability claim and PART-opt-2's downstream
  question — (source: docs/topics/reference/moe-literature.md).
- **FLAME-MoE: A Transparent End-to-End Research Platform for MoE Language Models** (no ID on
  record) — 38M–1.7B active, 64 experts, top-8, open code/data/checkpoints/routing
  logs/evals; traces show expert specialization emerging early and intensifying and
  co-activation sparse and stable; PART-opt-1's cross-suite validation point — (source:
  docs/topics/reference/moe-literature.md).
- **The observational-crowding claim** (no citations) — "observational MoE routing analysis
  is a moderately crowded area — expert-specialization papers exist for most released
  models"; the free artifacts are said to be the validation set, not the contribution —
  (source: docs/potential-projs/moe-partitions.md §4 2026-08-21, unverified review text).

**The matching method: permutation alignment, re-basin, and functional identity (PART-4)**

- **Ainsworth et al., Git Re-Basin: Merging Models modulo Permutation Symmetries** (no ID on
  record) — weight matching; independently trained models connected only after permutation
  alignment; the record calls cross-model expert matching "the MoE version of Git Re-Basin" —
  (source: docs/topics/reference/landscape-literature.md; docs/potential-projs/moe-partitions.md §4).
- **Entezari et al., The Role of Permutation Invariance in LMC of Neural Networks** (no ID on
  record) — the conjecture that permutation is the symmetry to quotient; the antecedent of
  PART-5's "same loss, same solution?" question — (source: docs/topics/reference/landscape-literature.md).
- **REPAIR (activation renormalization)** (no ID on record) — the third quotienting tool;
  relevant if matched experts need activation statistics repaired before comparison —
  (source: docs/topics/reference/identifiability-literature.md §3).
- **Unveiling LMC of Re-Basin from Neuron Distribution Perspective** (no ID on record) —
  re-basin methods often reduce barriers only marginally and work poorly early in training,
  with no unified theory of when they succeed; the honest caution for a hard-matching
  baseline — (source: docs/topics/reference/landscape-literature.md).
- **Beyond Structural Symmetries: LMC via Neuron Identifiability** (2026; no ID on record) —
  consistent feature-to-neuron assignment across seeds, with symmetry-breaking mechanisms
  characterizing which functions neurons can implement; the closest existing frame for "is
  the same partition being rediscovered?" — (source: docs/topics/reference/landscape-literature.md;
  docs/topics/reference/identifiability-literature.md).
- **Theseus — Rinaldi, Panariello, Salici, Porrello, Calderara, *Transporting Task Vectors
  across Different Architectures without Training*** (ICML 2026; arXiv 2602.12952;
  Danielle-supplied, not under the agent-unverified caveat) — characterizes a task update by
  its functional effect on intermediate representations, solves a functional matching problem
  on observed activations after aligning representation spaces via orthogonal Procrustes,
  closed-form and geometry-preserving; named as exactly the move PART-4 needs across expert
  counts and granularities — (source: docs/topics/reference/task-vectors.md;
  docs/potential-projs/moe-partitions.md §4 2026-08-22).
- **Roeder, Metz & Kingma, On Linear Identifiability of Learned Representations** (2021; no
  ID on record) — fit the optimal linear map between two models' representations; residual =
  identifiability gap; a matching-free invariance test — (source:
  docs/topics/reference/identifiability-literature.md §4).
- **Model stitching — Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021** (no IDs on
  record) — functional interchangeability at a depth via a trained adapter; an alternative
  operationalization of "the same factorization" — (source:
  docs/topics/reference/identifiability-literature.md §4).
- **CKA (Kornblith et al.)** (no ID on record) — scalable proxy with the caution that it can
  be dominated by a few directions and disagree with stitching — (source:
  docs/topics/reference/identifiability-literature.md §4).
- **LMC of MoEs** (arXiv 2509.11348) — the connectivity-side MoE record; the nearest work on
  whether two MoEs are in comparable regions — (source:
  docs/topics/reference/reinit-and-transfer-literature.md §(d)).
- **The Butterfly Effect** (arXiv 2506.13234) — trajectories highly sensitive to initial
  conditions, so seed-comparison studies need many seeds; a design caution for PART-5's seed
  arm — (source: docs/topics/reference/reinit-and-transfer-literature.md §(d)).
- **Landscaping LMC** (arXiv 2406.16300); **Layer-wise LMC** (arXiv 2307.06966, per-layer
  barriers insignificant vs. full-model, middle layers create barriers) — the rest of the
  connectivity toolkit a per-layer partition comparison could borrow — (source:
  docs/topics/reference/reinit-and-transfer-literature.md §(d)).
- **Going Beyond LMC: Layerwise Linear Feature Connectivity** (no ID on record) —
  connectivity in activation space rather than loss space; the space PART-4's functional
  matching would operate in — (source: docs/topics/reference/landscape-literature.md).
- **Juneja et al., Linear Connectivity Reveals Generalization Strategies** (ICLR 2023; no ID
  on record) — fine-tuned models cluster into distinct linearly-connected basins implementing
  *different generalization strategies* at similar in-distribution accuracy; the record's
  strongest existing evidence that quality-equivalent models can be internally non-equivalent
  — exactly PART-5's second outcome, and the precedent for PART-opt-2 — (source:
  docs/topics/reference/landscape-literature.md).
- **On the Emergence of Cross-Task Linearity in the Pretraining–Finetuning Paradigm** (no ID
  on record) — models fine-tuned from a common checkpoint stay in a shared linear regime,
  which is why task arithmetic and souping work *only* within a basin — (source:
  docs/topics/reference/landscape-literature.md).
- **Model soups (Wortsman et al.)** (2203.05482) — weight-space ensembling; the merging move
  whose MoE version needs expert matching first — (source:
  docs/topics/reference/landscape-literature.md; docs/topics/reference/moe-literature.md).
- **Ilharco et al., Editing Models with Task Arithmetic** (ICLR 2023; arXiv 2212.04089;
  Danielle-supplied) — weight-space task vectors and their arithmetic; the weight-space
  notion of identity that Theseus's functional notion is contrasted with — (source:
  docs/topics/reference/task-vectors.md).
- **Zhou et al., On Task Vectors and Gradients** (arXiv 2508.16082; Danielle-supplied) —
  one-epoch task vector ≈ negative scaled gradient, first-epoch gradient dominates; the
  early-dynamics claim behind cheap functional descriptors — (source:
  docs/topics/reference/task-vectors.md).
- **Kim et al., Task Vector Quantization for Memory-Efficient Model Merging** (arXiv
  2503.06921; Danielle-supplied) — compressing task updates; adjacent to storing per-config
  expert descriptors at sweep scale — (source: docs/topics/reference/task-vectors.md).

**Identifiability framing (the language PART's question is posed in)**

- **Expert assignment as a textbook non-identifiable latent** (no ID; framing entry) — the
  objective is invariant to expert permutation, so which expert specializes in what is
  trajectory-selected symmetry breaking; the Myth paper's cross-model overlap result is
  called "about as clean an existence proof of solution-class underdetermination as the field
  has produced" — (source: docs/topics/reference/identifiability-literature.md 2026-08-18).
- **Schölkopf et al., Toward Causal Representation Learning** (2021; no ID on record) — the
  CRL frame; interventions buy identifiability, which is why PART-5's balancing-mechanism arm
  is treated as an identification strategy — (source: docs/topics/reference/identifiability-literature.md).
- **Hyvärinen & Pajunen (1999), Nonlinear ICA: Existence and Uniqueness Results** (no ID) —
  latents fundamentally non-identifiable from observational i.i.d. data alone; the formal
  statement of PART's null — (source: docs/topics/reference/identifiability-literature.md).
- **Khemakhem et al., iVAE (2020)** (no ID on record) and Hyvärinen's time-contrastive /
  auxiliary-variable nonlinear ICA — identifiability restored by auxiliary variables,
  non-stationarity, or multiple environments, up to a residual group; the sweep's many
  configs are the "multiple environments" analogue — (source:
  docs/topics/reference/identifiability-literature.md).
- **Brehmer et al. (2022), Weakly Supervised CRL; Ahuja et al., Interventional CRL; von
  Kügelgen et al. (2021), Data Augmentations Provably Isolate Content from Style; the
  sparse-mechanism-shift principle** (no IDs on record) — the interventional-identifiability
  family, with the record's honesty note that PART identifies structure in training dynamics,
  not in data — (source: docs/topics/reference/identifiability-literature.md §2).
- **Huh et al. (2024), Position: The Platonic Representation Hypothesis** (no ID on record) —
  representations of independently trained models converge with scale; the conjecture that
  partition invariance should *increase* with scale, i.e. PART's result may be scale-bounded
  — (source: docs/topics/reference/identifiability-literature.md).
- **Sussmann 1992 (uniqueness up to permutation/sign); ReLU positive rescalings; attention-
  head permutations** (no IDs) — the enumerated known symmetry group that must be quotiented
  before residual variation counts as real — (source:
  docs/topics/reference/identifiability-literature.md §3).
- **Implicit-bias / simplicity-bias literatures; shortcut learning as selection pathology**
  (no IDs) — the selection principle behind "which of the equivalent partitions do we get" —
  (source: docs/topics/reference/identifiability-literature.md §5).
- **Local learning coefficient (Watanabe; Lau, Murfet et al.); Differentiation and
  Specialization of Attention Heads via the Refined LLC; Loss Landscape Degeneracy and
  Stagewise Development in Transformers** (no IDs on record) — degeneracy as the local face
  of non-identifiability, and dense precedents for measured specialization — (source:
  docs/topics/reference/identifiability-literature.md §5).

**The load-balancing confound (PART-5's validation arm and PART-3's controls)**

- **Load-balancing auxiliary loss** (Shazeer et al. 2017; Switch, Fedus et al. 2022) (no IDs
  on record; 1701.06538 for Sparsely-Gated MoE) — pushes routing toward uniformity, so
  observed assignments confound "where the data wants to go" with "where the balancer forced
  it"; the record calls this the most fundamental caveat, and says the sweep's
  balancing-mechanism variation is the only known way to test it at matched everything-else —
  (source: docs/topics/reference/regularization-literature.md;
  docs/potential-projs/moe-partitions.md §4).
- **ST-MoE router z-loss (Zoph et al. 2022)** (no ID on record) — the second router-shaping
  objective in the inventory — (source: docs/topics/reference/regularization-literature.md).
- **MoEC / cluster-level expert dropout** (2207.09094) and **Elbayad et al. (Findings ACL
  2023)** — gating dropout, conditional routing, curriculum; further distortions of the
  measured partition — (source: docs/topics/reference/regularization-literature.md,
  SciSpace-agent record, unverified).
- **Dirichlet-prior shaping of router outputs for upcycled MoEs** (2510.01185) — an explicit
  prior on the routing distribution — (source: docs/topics/reference/regularization-literature.md).
- **Switch expert dropout; StableMoE; DeepSeek's auxiliary-loss-free balancing; OLMoE's
  stability recipe; Gating Dropout (2205.14336)** — listed as the missing MoE canon on the
  balancing side; each is a distinct distortion regime the sweep's arms may or may not span —
  (source: docs/topics/reference/regularization-literature.md intake notes).
- **Expert-Choice routing; GLaM** (no IDs on record) — named-but-unlisted selection variants
  that change what a "partition" even is — (source: docs/topics/reference/moe-literature.md).

**Architectural vocabulary for the granularity / refinement axis (agent-generated design-
space record; identifiers agent-supplied or Claude-added and unverified per the ledger;
author pairs flagged fabrication-prone)**

- **DeepSeekMoE** (2401.06066) — shared + routed experts; the always-on path changes what the
  routed partition covers — (source: docs/topics/reference/moe-literature.md).
- **Scaling Laws for Fine-Grained MoE** (2402.07871) — the granularity axis itself; the
  nested-refinement question (does a 64-expert partition refine a 16-expert one?) is posed
  against it — (source: docs/topics/reference/moe-literature.md).
- **From Sparse to Soft MoE** (2308.00951) — soft selection dissolves the categorical
  partition; the limiting contrast case — (source: docs/topics/reference/moe-literature.md).
- **Mixtral of Experts** (2401.04088) — open-weights, closed-data; observational-only
  artifact — (source: docs/topics/reference/moe-literature.md).
- **Mixture-of-Depths** (2404.02258); **SwitchHead** (2312.07987); **Mixture of Attention
  Heads** (2210.05144) — routing over other internal units, i.e. other partitions of the same
  data — (source: docs/topics/reference/moe-literature.md).
- **Sparsely-Gated MoE** (1701.06538) — the origin of the routed-FFN object — (source:
  docs/topics/reference/moe-literature.md).
- **Branch-Train-Merge** (2208.03306) / **Branch-Train-MiX** (2403.07816, Claude-added) /
  **RouteLLM** (2406.18665) — partitions imposed rather than discovered (routing learned
  post-hoc over frozen experts); the record's noted fourth axis, *when* routing is learned —
  (source: docs/topics/reference/moe-literature.md intake notes).
- **BatchEnsemble** (2002.06715), **MIMONets** (2312.02829), **MatFormer** (2310.07707),
  **AdapterFusion** (2005.00247), **Mixture of LoRA Experts** (2404.13628), **MixLoRA**
  (2404.15159), **Higher Layers Need More LoRA Experts** (2402.08562), **Weight-Ensembling
  MoE** (2402.00433), **model soups** (2203.05482), **"hydra"/shared-trunk** (2209.14375,
  identity unchecked) — the rest of the placement table; the vocabulary for what unit a
  partition is over — (source: docs/topics/reference/moe-literature.md;
  docs/litreview/citation-verification-ledger.md rows 331–351).
- **MoEUT** (2405.16039), **Sparse Universal Transformer** (2310.07096) — MoE inside a shared
  looped block; a partition that is reused across iterations — (source:
  docs/topics/reference/layer-looping-literature.md; SciSpace-agent record, unverified).

**What the partition would be aligned against (PART-6) and downstream (PART-opt-2)**

- **Wen et al., Understanding WSD LRs: A River Valley Loss Landscape View** (arXiv
  2410.05192) — deterministic tokens form the river, uncertain tokens the walls; the
  "determinism profile" PART-6 asks whether routing recovers, and the toy bigram validation
  plus the ~0.39 Spearman token-uncertainty/sharpness correlation — (source:
  docs/topics/reference/landscape-literature.md; docs/topics/reference/token-level-literature.md).
- **Token-Level Uncertainty-Aware Objective for LM Post-Training** (no ID on record) —
  epistemic vs. aleatoric token uncertainty; the reference-model entropy scorer PART-3 uses
  as its covariate set — (source: docs/topics/reference/token-level-literature.md).
- **Rho-1, Not All Tokens Are What You Need for Pretraining** (no ID on record) — token
  loss-trajectory taxonomy; a competing data typology the partition may or may not recover —
  (source: docs/topics/reference/token-level-literature.md).
- **Task2Vec (Achille et al.)** (no ID on record) — Fisher-embedding dataset representation;
  the intrinsic-featurization family the routing partition is asked to align with — (source:
  docs/topics/reference/critical-periods.md; docs/topics/reference/data-featurization-literature.md).
- **WIMBD, compression, Zipf/burstiness; perplexity correlations, RegMix, DoReMi, mixing
  laws; alignment and diversity coefficients** (no IDs on record) — the intrinsic and
  model-mediated feature families PART-6's alignment analysis draws its features from —
  (source: docs/topics/reference/data-featurization-literature.md).
- **Achille, Rovere & Soatto, Critical Learning Periods in Deep Networks** (ICLR 2019; no ID)
  — the window before commitment; the reading under which the partition is chosen early and
  fixed — (source: docs/topics/reference/critical-periods.md).
- **Frankle et al., LMC and the Lottery Ticket Hypothesis** (no ID on record) — sibling runs
  linearly connected only after a critical number of steps; the timing precedent for when a
  partition becomes comparable — (source: docs/topics/reference/landscape-literature.md).
- **Plasticity-loss line (Dohare et al., Nature 2024 / arXiv 2306.13812; Lyle et al.
  2303.01486 and 2402.18762)** — the "modular plasticity" reading of MoE partitions: does a
  different factorization leave differently plastic experts? PART-opt-2's downstream question
  — (source: docs/topics/reference/plasticity.md).

**Artifacts, gates, and program placement (context, not literature)**

- **Slicing-and-Dicing MoE sweep** (arXiv 2605.11689; Danielle third author) — ~2,000 runs up
  to 6.6B total varying expert count, granularity, heterogeneous sizing, shared experts, load
  balancing; findings: total parameters always help even at 128× ratios, optimal expert size
  depends only on active parameters, other knobs second-order; this is the matched-loss
  comparison across architectures PART is built on, and the two "why" analyses target it —
  (source: docs/potential-projs/moe-partitions.md §4; docs/danielle-inputs.md).
- **Sweep checkpoint availability** (no ID) — all final checkpoints exist and are being
  uploaded to Hugging Face; no intermediate checkpoints from the original sweep, so
  training-dynamics analyses need the collaborator's new experiments or own runs — (source:
  docs/open-questions-answered.md 2026-08-21).
- **The suite landscape and the artifact gap** (no IDs) — FLAME-MoE / OLMoE / OpenMoE each
  one data recipe; the 2025–26 open-weights wave (Llama 4, DeepSeek V4, Qwen 3.6, Kimi K2.6,
  gpt-oss, Command A+) open-weights and closed-data; the record's conclusion that no public
  multi-recipe MoE suite exists — (source: docs/topics/reference/moe-literature.md;
  docs/potential-projs/moe-partitions.md §4 2026-08-21; unverified).
- **Signal and Noise** (no ID on record) — eval noise worsens as scale shrinks and routing
  discreteness plausibly adds variance; the noise-floor caution on any cross-config partition
  comparison — (source: docs/topics/reference/moe-literature.md).
- **DataDecide** (no ID on record) — the matched-loss-across-recipes structural move PART
  mirrors across architectures, and the dense control ladder — (source:
  docs/topics/reference/moe-literature.md; docs/potential-projs/moe-partitions.md §4).
- **ANN-1 MoE caution (sibling record)** (no ID) — checkpoint merging needs expert matching
  first or it averages mismatched experts into mush; PART-4 is the prerequisite — (source:
  docs/potential-projs/annealed-readouts.md §4 2026-08-18).
- **Portfolio placement** (no ID) — workshop-sized #6; full-conference #4 "One Partition,
  Many Architectures" (expected high / ceiling high); P4 sub A, scoop risk low ("the sweep is
  the moat") — (source: docs/portfolio-rankings.md; docs/potential-projs/moe-partitions.md §4).

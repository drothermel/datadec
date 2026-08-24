# moe movement — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`moe-movement.md`](../moe-movement.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*Purpose: the high-recall corpus for MOVE (reroute-vs-rewrite; the frozen-routing
case study). Every item is on record somewhere in this repository; nothing here is
verified, and nothing here is a positioning claim. Err toward inclusion — one line is
cheap. Grouped by theme; the curated core lives in `../moe-movement.md` §5.*

**Commitment-clock and routing-dynamics precedents (the metrics MOVE-2 is defined
against)**

- **OLMoE — Open Mixture-of-Experts Language Models** (no ID on record) — supplies
  "router saturation" (average overlap of per-token top-k experts at step t vs. at
  convergence), rising sharply in the first few thousand steps, deeper layers saturating
  faster; MOVE-2's saturation metric is this metric, and OLMoE's open intermediate
  checkpoints are a Stage-1 substrate — (source: docs/topics/reference/moe-literature.md).
- **Three Phases of Expert Routing: How Load Balance Evolves During MoE Training** (no ID
  on record) — early balance-prioritizing phase, stabilization/specialization phase, late
  relaxation phase trading balance for quality; non-monotone, invisible post-hoc; annealing
  checkpoints said to confirm the phases are pretraining-specific and "stable during
  fine-tuning"; the record calls this aggregate-level, so the per-token version is the
  contrast — (source: docs/topics/reference/moe-literature.md;
  docs/potential-projs/moe-movement.md §4 2026-08-18).
- **OpenMoE analysis** (no ID on record) — routing decisions predominantly driven by token
  ID with minimal context relevance; token-to-expert assignments determined early in
  pretraining and largely fixed thereafter; this is the empirical seed of the whole
  frozen-routing hypothesis and of MOVE-2's "is the freeze intrinsic" question —
  (source: docs/topics/reference/nonstationarity-accounting.md;
  docs/potential-projs/moe-movement.md §4).
- **Continual Pre-training of MoEs: How Robust Is Your Router?** (no ID on record) —
  routing changes most in early layers under continual pretraining, no-replay showing the
  most reorganization and the most forgetting; named the MoE-native warm-start precedent
  and a candidate mechanism of catastrophic forgetting in MoEs; relevant to MOVE-5's reset
  test and MOVE-6's thaw — (source: docs/topics/reference/moe-literature.md).
- **The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not
  Necessarily Domain Expertise** (no ID on record) — independently trained MoEs pick
  unrelated specialization solutions; routers are linear maps so hidden-state similarity is
  necessary and sufficient for expert-usage similarity; load-balancing loss provably
  suppresses shared hidden directions; matters to MOVE because it predicts what a MOVE-5
  router reset should converge to and gives the balancer a mechanism — (source:
  docs/topics/reference/moe-literature.md; docs/topics/reference/identifiability-literature.md).
- **FLAME-MoE: A Transparent End-to-End Research Platform for MoE Language Models** (no ID
  on record) — 38M–1.7B active, 64 experts/layer, top-8, open code/data/checkpoints/routing
  logs/evals; training traces reported to show expert specialization emerging early and
  intensifying, co-activation sparse and stable, routing converging quickly early; the
  primary candidate Stage-1 substrate and the artifact-survey target of §3 step 1 —
  (source: docs/topics/reference/moe-literature.md;
  docs/potential-projs/trajectory-statistics.md §"Follow-up").
- **Hash routing results** (no citation on record) — fixed, content-free, random assignment
  performing surprisingly close to learned routing; invoked in the 2026-08-21 §4 entry as
  evidence that a never-adaptive router is nearly sufficient, i.e. as the argument against
  needing routing freedom at all — (source: docs/potential-projs/moe-movement.md §4).
- **Slicing-and-Dicing MoE sweep** (arXiv 2605.11689; Danielle third author) — ~2,000 MoEs
  varying expert count, granularity, shared experts, load balancing; its own finding that
  load-balancing mechanism barely affects final quality is the second piece of evidence the
  frozen-routing hypothesis is argued against, and its balancing-mechanism arms are half of
  MOVE-4 for free; all final checkpoints exist, no intermediate checkpoints — (source:
  docs/open-questions-answered.md 2026-08-21; docs/potential-projs/moe-partitions.md §4).

**Comparability, symmetry, and the swap-evaluation obstacle (what MOVE-1 inherits)**

- **Expert permutation as a non-identifiable latent** (no ID; framing entry) — the MoE
  objective is invariant to expert permutation, so which expert specializes in what is
  trajectory-selected symmetry breaking; the symmetry group MOVE-1's cross-checkpoint swap
  must respect — (source: docs/topics/reference/identifiability-literature.md 2026-08-18).
- **The MoE comparability warning** (no ID; framing entry) — interpolation barriers,
  checkpoint merging, and stitching all need an expert-alignment step; "re-basin methods for
  MoE are immature"; "how to quotient MoE symmetries for checkpoint comparison" is called an
  open gap — directly gates whether MOVE-1's (router_t, experts_t+1) swap is meaningful —
  (source: docs/topics/reference/moe-literature.md; docs/potential-projs/moe-movement.md §4).
- **Entezari et al., The Role of Permutation Invariance in LMC** (no ID on record) — the
  dense conjecture that models are connected modulo permutation; antecedent for the
  "quotient the symmetry first" move — (source: docs/topics/reference/landscape-literature.md).
- **Ainsworth et al., Git Re-Basin: Merging Models modulo Permutation Symmetries** (no ID on
  record) — weight matching; independently trained models connected only after permutation
  alignment; named as the dense antecedent of MoE expert matching — (source:
  docs/topics/reference/landscape-literature.md; docs/topics/reference/identifiability-literature.md).
- **REPAIR (activation renormalization)** (no ID on record) — third member of the
  quotienting toolkit; relevant if swapped checkpoints need activation statistics repaired —
  (source: docs/topics/reference/identifiability-literature.md §3).
- **Unveiling LMC of Re-Basin from Neuron Distribution Perspective** (no ID on record) —
  re-basin methods often reduce barriers only marginally and work poorly early in training,
  with no unified theory of when they succeed; a caution for early-training MOVE-1 swaps —
  (source: docs/topics/reference/landscape-literature.md).
- **Beyond Structural Symmetries: LMC via Neuron Identifiability** (2026; no ID on record) —
  consistent assignment of features to neurons across seeds; the "principled answer to when
  two models' internal metrics are comparable" the record wants — (source:
  docs/topics/reference/landscape-literature.md; docs/topics/reference/identifiability-literature.md).
- **LMC of MoEs** (arXiv 2509.11348) — the connectivity-side MoE record in the
  reset/landscape sub-thread; the nearest existing work on interpolating MoE checkpoints —
  (source: docs/topics/reference/reinit-and-transfer-literature.md §(d)).
- **The Butterfly Effect** (arXiv 2506.13234) — trajectories highly sensitive to initial
  conditions, so reset studies need many seeds; a direct design caution for MOVE-5 —
  (source: docs/topics/reference/reinit-and-transfer-literature.md §(d)).
- **Landscaping LMC** (arXiv 2406.16300) — connectivity-toolkit entry in the same sub-thread
  — (source: docs/topics/reference/reinit-and-transfer-literature.md §(d)).
- **Layer-wise LMC** (arXiv 2307.06966) — per-layer barriers insignificant relative to the
  full-model barrier, middle layers create barriers; relevant to MOVE-1's per-layer
  attribution of the swap delta — (source: docs/topics/reference/reinit-and-transfer-literature.md §(d)).
- **Going Beyond LMC: Layerwise Linear Feature Connectivity** (no ID on record) —
  connectivity in activation space rather than loss space; the functional-comparison flavour
  MOVE-1's output-delta attribution uses — (source: docs/topics/reference/landscape-literature.md).
- **Theseus, Rinaldi et al., Transporting Task Vectors across Different Architectures
  without Training** (ICML 2026; arXiv 2602.12952; Danielle-supplied, so not under the
  agent-unverified caveat) — task identity defined functionally on activations, solved by
  orthogonal Procrustes alignment; the concrete alternative to parametric matching if
  MOVE-1's swap needs an alignment step — (source: docs/topics/reference/task-vectors.md;
  docs/potential-projs/moe-partitions.md §4 2026-08-22).
- **Roeder, Metz & Kingma, On Linear Identifiability of Learned Representations** (2021; no
  ID on record) — fit the optimal linear map between two models' representations, residual =
  identifiability gap; a functional comparison usable across swapped checkpoints — (source:
  docs/topics/reference/identifiability-literature.md §4).
- **Model stitching — Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021** (no IDs on
  record) — adapter-mediated interchangeability at a depth; the record explicitly notes
  stitching needs an expert-alignment step in MoEs — (source:
  docs/topics/reference/identifiability-literature.md §4).
- **CKA (Kornblith et al.)** (no ID on record) — the scalable proxy the record says to use
  with stitching/linear-map residuals as ground truth; the dense "formless" comparison MOVE-1
  is contrasted with — (source: docs/topics/reference/identifiability-literature.md §4).

**Non-stationarity, two-timescale, and stabilizer framings (Stage 2's conceptual frame)**

- **The exogenous/endogenous non-stationarity accounting frame** (no ID; canonical program
  text) — routing is the clearest endogenous case; every stabilizer (balancing loss, EMA,
  decay, warmup) is to be accounted for by what it suppresses and what suppression costs;
  Stage 2 instantiates this — (source: docs/topics/reference/nonstationarity-accounting.md).
- **The target-network analogy from RL and GANs** (no citations on record) — deliberately
  slowing one timescale so the other converges against a quasi-stationary target; the record
  reads MoE aux-loss / z-loss / jitter / capacity factors / router-LR reduction as the same
  move; MOVE-opt-1 turns it into a design axis — (source:
  docs/potential-projs/moe-movement.md §4 2026-08-21).
- **Igl et al., ITER (ICLR 2021)** (no ID on record) — "transient non-stationarities
  permanently impact the latent representation"; in accounting terms a stabilizer that
  removes the non-stationarity's history rather than the non-stationarity; the RL-side
  statement of the scar phenomenon MOVE-5 tests for in routing — (source:
  docs/topics/reference/nonstationarity-accounting.md 2026-08-18).
- **Achille, Rovere & Soatto, Critical Learning Periods in Deep Networks** (ICLR 2019; no ID
  on record) — early deficits permanently impair; Information Plasticity via the Fisher
  trace; the leading member of the "dense networks also commit early" alternative hypothesis
  the record holds honestly against the frozen-routing story — (source:
  docs/topics/reference/critical-periods.md; docs/potential-projs/moe-movement.md §4).
- **Critical Learning Periods for Multisensory Integration in Deep Networks** (no ID on
  record) — critical periods arise from unstable early transient dynamics decisive of final
  representations; same alternative-hypothesis cluster — (source:
  docs/topics/reference/critical-periods.md).
- **Ash & Adams, warm-starting gap** (no ID on record) — early data poverty permanently
  impairs; shrink-perturb partially reopens selection; the supervised-learning member of the
  same triple, and the template for MOVE-5's "perturb and continue" design — (source:
  docs/topics/reference/identifiability-literature.md; docs/topics/reference/critical-periods.md).
- **Frankle et al., Linear Mode Connectivity and the Lottery Ticket Hypothesis** (no ID on
  record) — sibling runs become linearly connected only after a critical number of steps; a
  commitment-event precedent, and lottery-ticket structure is named in the
  early-commitment-is-intrinsic alternative — (source:
  docs/topics/reference/identifiability-literature.md §5; docs/topics/reference/landscape-literature.md).
- **Fort et al., Deep Learning vs. Kernel Learning** (no ID on record) — NTK rapid early
  rotation then stabilization; another commitment clock alongside router saturation —
  (source: docs/topics/reference/identifiability-literature.md §5).
- **Local learning coefficient / singular learning theory (Watanabe; Lau, Murfet et al.)**
  (no IDs on record) — per-checkpoint degeneracy scalar; the record lists LLC alongside
  Fisher trace and LMC onset as commitment clocks that router saturation joins as a fourth —
  (source: docs/topics/reference/identifiability-literature.md §5;
  docs/potential-projs/moe-movement.md §4 2026-08-18).
- **Differentiation and Specialization of Attention Heads via the Refined LLC; Loss
  Landscape Degeneracy and Stagewise Development in Transformers** (no IDs on record) —
  developmental-interpretability precedents for staged specialization over training, the
  dense analogue of expert specialization phases — (source:
  docs/topics/reference/identifiability-literature.md §5).
- **Dohare et al., Loss of plasticity in deep continual learning** (Nature 2024; arXiv
  2306.13812) — the continual-learning plasticity anchor behind reading each expert as a
  continual-learning system under covariate shift (MOVE-3) — (source:
  docs/topics/reference/plasticity.md).
- **Lyle et al., Understanding Plasticity in Neural Networks** (arXiv 2303.01486) and
  **Disentangling the Causes of Plasticity Loss** (arXiv 2402.18762) — plasticity loss tied
  to landscape curvature; the "modular plasticity" reading of the MoE projects — (source:
  docs/topics/reference/plasticity.md).
- **Hernandez-Garcia, Figliolia, Millidge, Can Scale Save Us From Plasticity Loss in LLMs?**
  (arXiv 2606.24752) — plasticity loss in continual *and stationary* settings, sublinear
  scaling law; supports the record's claim that "stationary" pretraining is not stationary
  from the gradient's perspective — (source: docs/topics/reference/plasticity.md;
  docs/topics/reference/reinit-and-transfer-literature.md §(c)).
- **DASH** (no ID on record) — non-stationarity-motivated plasticity fixes ineffective in
  the stationary incremental setting; a caution that the exogenous/endogenous split governs
  which fixes apply — (source: docs/topics/reference/nonstationarity-accounting.md).
- **Plasticity injection (Nikishin et al.; arXiv 2305.15555)** — "the most directly
  borrowable instrument": if injection helps, plasticity was the binding constraint; the
  design template for MOVE-5/MOVE-6 read as plasticity interventions on the router —
  (source: docs/topics/reference/reinit-and-transfer-literature.md §(c)).
- **Reset & Distill (Ahn et al.; arXiv 2403.05066)** — reset-based continual-learning method;
  reset-family prior art for MOVE-5 — (source: docs/topics/reference/reinit-and-transfer-literature.md §(c)).
- **When Does Re-initialization Work? (Zaidi et al.; arXiv 2206.10011)** — >15,000 vision
  models; reinit helps without other regularization, little once regularization is tuned;
  the "your reset result may be a regularization result" caution for MOVE-5 — (source:
  docs/topics/reference/reinit-and-transfer-literature.md §(c)).
- **Plasticity-loss survey in RL (arXiv 2411.04832)** — last-layer resets standard, the
  "concentrated in last layers" belief weakly evidenced; partial-reset design prior art —
  (source: docs/topics/reference/reinit-and-transfer-literature.md §(c)).
- **Calibrated partial resets (arXiv 2607.24996); spectral collapse (arXiv 2509.22335);
  activation design (arXiv 2509.22562)** — the rest of the reset-method flank a router reset
  could borrow from — (source: docs/topics/reference/reinit-and-transfer-literature.md §(c)).
- **Fisher-guided selective forgetting (arXiv 2502.00802)** — targeted-forgetting instrument
  adjacent to a mid-run router perturbation — (source:
  docs/topics/reference/reinit-and-transfer-literature.md §(d)).
- **Representation-plasticity timeline in LLMs (arXiv 2410.06225)** — when representations
  stop moving; the dense timeline against which router-freeze timing is read — (source:
  docs/topics/reference/reinit-and-transfer-literature.md §(d)).
- **PolyPythias (van der Wal et al.; ICLR 2025; arXiv 2503.09543)** — 50 pretraining runs,
  9 seeds × 5 sizes, ~7,000 checkpoints; the many-seed substrate the record calls "this is
  the substrate" for reset studies — (source: docs/topics/reference/reinit-and-transfer-literature.md §(e)).
- **Continual Backprop: SGD with Persistent Randomness** (no ID on record) — an intervention
  that "artificially reopens" the critical period; the dense analogue of a routing thaw
  (MOVE-6) — (source: docs/topics/reference/critical-periods.md).

**Schedule / annealing structure (MOVE-6 is said to be structurally identical to LR decay)**

- **Wen et al., Understanding WSD Learning Rates: A River Valley Loss Landscape View**
  (arXiv 2410.05192) — stable phase drives progress along the river, decay drives the
  mountain direction; deterministic tokens form the river and uncertain tokens the walls;
  MOVE-2's reverting-vs-persistent flip split is the categorical translation of wall
  oscillation vs. river movement, and MOVE-6 is framed as the router's constraint schedule
  in the LR schedule's role — (source: docs/topics/reference/landscape-literature.md;
  docs/topics/reference/token-level-literature.md).
- **Training Dynamics of the Cooldown Stage in WSD** (no ID on record) — landscape plotted in
  global-direction vs. local-Adam-direction coordinates; the visualization precedent for a
  thaw-window analysis — (source: docs/topics/reference/landscape-literature.md).
- **Multi-power law (Luo et al.; arXiv 2503.12811)** — the decay-induced loss-drop term as
  descending from the walls to the river; the quantitative frame for "did anything move when
  we annealed the suppressor" — (source: docs/topics/reference/landscape-literature.md).
- **Scaling with Collapse (arXiv 2509.25087)** — well-tuned runs' loss curves collapse onto
  a shared shape; a cross-run comparability criterion from curves alone, useful when MOVE-4's
  arms must be compared without weight access — (source: docs/topics/reference/landscape-literature.md).
- **Rho-1, Not All Tokens Are What You Need for Pretraining** (no ID on record) — token
  loss-trajectory taxonomy (persistently-high/low, descending, fluctuating); the dense
  taxonomy MOVE-opt-3's entropy-bucketed flip analysis parallels — (source:
  docs/topics/reference/token-level-literature.md).
- **Token-Level Uncertainty-Aware Objective for LM Post-Training** (no ID on record) —
  epistemic vs. aleatoric split; epistemic uncertainty drains faster for low-aleatoric
  tokens; the reference-model entropy scorer MOVE-opt-3 needs is this construct — (source:
  docs/topics/reference/token-level-literature.md).

**MoE architecture vocabulary — the design-space reading list (agent-generated; all
identifiers unverified per the ledger, author pairs flagged fabrication-prone)**

- **Sparsely-Gated MoE, Shazeer et al.** (1701.06538) — the sparse-FFN-MoE origin and the
  load-balancing auxiliary loss whose weight MOVE-4 varies — (source:
  docs/topics/reference/moe-literature.md; docs/litreview/citation-verification-ledger.md).
- **Switch Transformer / ST-MoE (Fedus et al. 2022; Zoph et al. 2022)** (no IDs on record) —
  higher expert dropout and the router z-loss; the two stability knobs MOVE-4 treats as arms
  — (source: docs/topics/reference/regularization-literature.md;
  docs/potential-projs/moe-movement.md §5).
- **DeepSeekMoE** (2401.06066) — shared + routed experts; changes what "same experts" means
  in the reroute/rewrite split — (source: docs/topics/reference/moe-literature.md).
- **Mixtral of Experts** (2401.04088) — open-weights MoE named as observational-only
  (no checkpoints/data), i.e. an artifact MOVE-1 cannot use — (source:
  docs/topics/reference/moe-literature.md).
- **From Sparse to Soft Mixtures of Experts** (2308.00951) — soft selection; the limiting
  case where the categorical reroute channel disappears — (source:
  docs/topics/reference/moe-literature.md).
- **Mixture-of-Depths** (2404.02258) — routing token–layer participation; a different
  routing unit whose commitment clock would be a separate object — (source:
  docs/topics/reference/moe-literature.md).
- **SwitchHead** (2312.07987) — MoE attention; extends where rerouting can occur beyond the
  FFN — (source: docs/topics/reference/moe-literature.md).
- **Scaling Laws for Fine-Grained MoE** (2402.07871) — the granularity axis; per-layer
  crossover timing plausibly depends on it — (source: docs/topics/reference/moe-literature.md).
- **Branch-Train-Merge** (2208.03306) and **Branch-Train-MiX** (2403.07816, Claude-added) —
  routing learned post-hoc over frozen dense experts; the record's noted fourth axis (*when*
  routing is learned) and the extreme case of a never-adapting router — (source:
  docs/topics/reference/moe-literature.md intake notes).
- **Model soups** (2203.05482) — weight-space ensembling; the merging move the MoE record
  says averages mismatched experts into mush without alignment — (source:
  docs/topics/reference/moe-literature.md; docs/potential-projs/annealed-readouts.md §4).
- **RouteLLM** (2406.18665) — query routing between deployed LLMs; a fixed, externally
  learned router as a contrast to a jointly trained one — (source: docs/topics/reference/moe-literature.md).
- **MIMONets** (2312.02829), **BatchEnsemble** (2002.06715), **MatFormer** (2310.07707),
  **AdapterFusion** (2005.00247), **Mixture of LoRA Experts** (2404.13628), **MixLoRA**
  (2404.15159), **Higher Layers Need More LoRA Experts** (2402.08562), **Weight-Ensembling
  MoE** (2402.00433), **Mixture of Attention Heads** (2210.05144), **"hydra"/shared-trunk**
  (2209.14375, identity unchecked) — the rest of the ensemble→MoE placement table; vocabulary
  for "what is an expert" and therefore what "same experts, different assignments" means —
  (source: docs/topics/reference/moe-literature.md;
  docs/litreview/citation-verification-ledger.md rows 331–351, agent-supplied, unverified).
- **GLaM; Expert-Choice routing; StableMoE; DeepSeek auxiliary-loss-free balancing; OLMoE's
  stability recipe** (no IDs on record) — named-but-unlisted stability/selection variants;
  each is a candidate MOVE-4 arm — (source: docs/topics/reference/moe-literature.md;
  docs/topics/reference/regularization-literature.md intake notes).
- **MoEUT** (2405.16039) and **Sparse Universal Transformer** (2310.07096) — MoE inside a
  shared/looped block; the record notes these sit at the MoE/layer-looping intersection, so
  "which expert at which iteration" is a further routing channel — (source:
  docs/topics/reference/layer-looping-literature.md; unverified SciSpace-agent record).

**Regularizers and stability knobs (the "stability apparatus" MOVE-4/MOVE-6 manipulate;
SciSpace-agent record, unverified, several citations flagged off-target)**

- **Load-balancing auxiliary loss** (Shazeer 2017; Switch, Fedus 2022) — the primary
  suppressor MOVE-6 anneals to zero — (source: docs/topics/reference/regularization-literature.md).
- **ST-MoE router z-loss** (Zoph et al. 2022) — the second MOVE-4 arm — (source:
  docs/topics/reference/regularization-literature.md).
- **Gating Dropout** (2205.14336) — listed as missing MoE canon; a stochastic routing
  perturbation adjacent to MOVE-5's router perturbation — (source:
  docs/topics/reference/regularization-literature.md intake notes; ledger row 130).
- **MoEC, cluster-level expert dropout** (2207.09094) and **Elbayad et al. (Findings ACL
  2023)** — gating dropout, conditional routing, curriculum against MoE overfitting; further
  router-side stochasticity precedents — (source: docs/topics/reference/regularization-literature.md).
- **Dirichlet-prior shaping of router outputs for upcycled MoEs** (2510.01185) — an explicit
  prior on the routing distribution; a suppressor of a different shape — (source:
  docs/topics/reference/regularization-literature.md).
- **Flooding (Ishida et al., ICML 2020)**, dropout (Srivastava 2014), weight decay/AdamW
  (Loshchilov & Hutter 2019), LayerDrop (Fan, Grave & Joulin, ICLR 2020), UniDrop (NAACL
  2021) (no arXiv IDs on record for most) — the general/transformer regularizer inventory a
  MOVE intervention arm must hold fixed — (source: docs/topics/reference/regularization-literature.md).

**Substrate, artifacts, and measurement cautions**

- **Signal and Noise: A Framework for Reducing Uncertainty in LM Evaluation** (no ID on
  record) — noise worsens as scale shrinks; routing discreteness plausibly adds eval
  variance, making the noise-floor stage "more necessary" for MoE work at 20–50M active —
  (source: docs/topics/reference/moe-literature.md 2026-08-18;
  docs/refs/research-trajectory-pre-to-post-training.md).
- **DataDecide: How to Predict Best Pretraining Data with Small Experiments** (no ID on
  record) — supplies the dense control ladder at matched active parameters for MOVE-opt-4
  "for free" — (source: docs/topics/reference/moe-literature.md;
  docs/potential-projs/trajectory-statistics.md).
- **Folklore-tuned MoE knobs caution** (no ID) — aux-loss coefficients, top-k, expert count,
  capacity factors tuned at large scale and possibly mis-set for 20–50M active, so the
  regime-mismatch critique applies to MOVE's own baseline — (source:
  docs/topics/reference/moe-literature.md).
- **FLAME-MoE routing-log gate** (no ID) — which checkpoints, how many tokens, whether token
  identities are recoverable; still open, and it decides T0 vs. T1 for Stage 1 — (source:
  docs/open-questions-answered.md "Open — not yet checked").
- **Slicing-and-Dicing checkpoint answer** (no ID) — final checkpoints exist, no
  intermediates from the original sweep, so "over training" analyses need new experiments or
  own runs — (source: docs/open-questions-answered.md 2026-08-21).
- **TRJ-moe-1 / TRJ-moe-3 (sibling project records)** (no IDs) — routing-flip drift/diffusion
  with the reverting/persistent split, and the dense control ladder; the same metrics as
  MOVE-2/MOVE-opt-4 recorded under trajectory statistics — (source:
  docs/potential-projs/trajectory-statistics.md §"Follow-up").
- **TOK-obs-5 (sibling project record)** (no ID) — flips by token entropy on FLAME-MoE,
  described as the MoE twin of the dense entropy-bucket figure; identical to MOVE-opt-3 —
  (source: docs/potential-projs/token-movement.md).
- **ANN-1 MoE caution (sibling project record)** (no ID) — checkpoint merging is dense-only
  as specified because an MoE variant needs the PART-4 expert-alignment step first —
  (source: docs/potential-projs/annealed-readouts.md §4 2026-08-18).
- **Nemotron 3 sliding-window checkpoint merging** (no ID on record) — production mid-run
  merged readouts (~16% FLOP savings), an MoE-hybrid model; the merging practice that the
  expert-alignment caution applies to — (source: docs/refs/research-trajectory-pre-to-post-training.md).
- **Held-out token set spec** (no ID; internal) — the frozen, versioned, domain- and
  entropy-stratified probe set shared with Annealed readouts, WSD retrain suite, Token-level
  movement, MoE recipe suite, and Functional featurization; MOVE-1's swap evaluation is
  defined on it — (source: docs/potential-projs/moe-movement.md §3 step 4).

**Ranked-list and program placement (context, not literature)**

- **Portfolio placement** (no ID) — reroute-vs-rewrite at workshop #9 / full-conference #7
  ("Reroute or Rewrite? Where Training Moves an MoE"); frozen-routing at workshop #10 /
  full-conference #8 ("Does MoE Training Suppress Its Own Non-Stationarity?"), #8 said to
  need #7's machinery; both are P4 sub B plus its causal arm — (source:
  docs/portfolio-rankings.md; docs/potential-projs/moe-movement.md §4 2026-08-21).

**NBLM MoE-notebook additions (intake 2026-08-24; agent-generated; canon IDs
Claude-added, cluster items no-ID):**

- **The MoE scaling-law cluster** — Efficiency Leverage (activation ratio as
  primary driver), comprehensive joint law (G_opt≈7, S_opt≈0.31), unified
  routed laws (2202.01169; S-BASE robustness), holistic shape laws (optimal
  band widens with scale — proxy-scale shape sensitivity caution), joint
  memory-aware law (MoE memory-optimality), parameters-vs-FLOPs (reading
  comprehension favors density), fine-grained laws (2402.07871), 50B+
  empirics (softmax/Top-k ordering sensitivity) — (source:
  `../../topics/reference/moe-literature.md`, 2026-08-24 NBLM entry)
- **B2, optimal sparsity for reasoning** (no ID) — sparsity helps memorization
  monotonically but hurts reasoning at scale; NOT recoverable by GRPO or
  test-time compute; convergent with Mixture of Parrots — (source: same)
- **Capacity-aware inference** (no ID) — test-time expert load up to 7×
  average despite training load losses; the train-vs-inference routing
  distribution gap — (source: same)
- **Specialization evidence** — OLMoE 2409.02060 from-scratch routing highly
  domain-specialized vs upcycled Mixtral's redundancy; A4 survey's >99%
  expert-similarity collapse without regularization (OMoE/MoDE fixes); HMoE
  hard-token→large-expert routing — (source: same)
- **ST-MoE 2202.08906 fine-tuning protocols** — sparse models need different
  fine-tuning hyperparameters (smaller batch, higher LR) — a comparability
  datum for dense-vs-MoE pipeline comparisons — (source: same)

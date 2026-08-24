# landscape geometry — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`landscape-geometry.md`](../landscape-geometry.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*Recall corpus for GEO (loss-landscape geometry and cross-recipe comparability). Highest-recall
enumeration of every paper, method, or named prior-art item on record anywhere in this
repository that could plausibly bear on GEO. One line per item; recall is the point, so
marginal items are kept. Every item carries its repo source. **Nothing here is verified** —
almost all of it entered through the 2026-08-18 Research Trajectory conversations and the
2026-08-22 reinit literature pass; the only GEO-tagged row in the citation ledger is
2407.17465 (Danielle-supplied, still unverified). No positioning or novelty claims.*

**River-valley picture and the "river test" (GEO-1)**

- **Wen et al., *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss
  Landscape View*** (arXiv 2410.05192) — the canonical statement: pretraining loss as a deep
  valley with a river at the bottom; the sustained high-LR phase drives progress along the
  river, the decay phase in the mountain direction. **Their interpolation signature is
  GEO-1's instrument**: loss between two stable-phase checkpoints is convex and unimodal (a
  valley cross-section), between two decay-phase checkpoints it decays smoothly and
  monotonically — "currently the closest thing to a 'river test'" —
  (source: docs/topics/reference/landscape-literature.md; docs/potential-projs/landscape-geometry.md §5).
- **Wen et al.'s own validation of the token mechanism** — a toy bigram language (cities with
  name distributions of varying determinism) reproducing the river-valley geometry; the
  stable phase learns deterministic tokens, the decay phase the stochastic ones; on real data
  a **Spearman correlation ≈ 0.39 between token-level uncertainty and local sharpness** —
  (source: docs/topics/reference/landscape-literature.md; token-level-literature.md).
- **Valley geometry as a *data* property** — deterministic tokens form the river, uncertain
  tokens the walls, so a corpus's determinism profile is a candidate predictor of landscape
  geometry, i.e. the valley shape is plausibly recipe-dependent (the mechanism behind GEO's
  cross-recipe question) — (source: docs/topics/reference/landscape-literature.md 2026-08-18 entry).
- ***Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning Rate Scheduler***
  (no ID) — plots the landscape in coordinates defined by the global pre-cooldown→final
  direction vs. the local Adam-step direction, noting a clear river-valley visualization had
  been lacking; GEO-opt-2's figure —
  (source: docs/topics/reference/landscape-literature.md).
- ***Scaling with Collapse: Efficient and Predictable Training of LLM Families*** (arXiv
  2509.25087) — well-tuned runs' loss curves collapse onto a shared shape, "a possible
  cross-run comparability criterion built from curves alone, no weight access needed";
  GEO-opt-1's method and the weight-free contrast to the barrier verdict —
  (source: docs/topics/reference/landscape-literature.md).
- **Multi-power law — Luo et al.** (arXiv 2503.12811, ICLR 2025) — predicts the full loss
  curve across LR schedules; its "decay-induced loss drop" term reads as descending from
  oscillating on the walls down to the river —
  (source: docs/topics/reference/landscape-literature.md; loss-curve-forecasting.md).
- **Grokking-plateau reading of the river** (paper not named, unverified) — plateaus at higher
  LR can accelerate final convergence during decay by optimally initializing the slow river
  modes; the decay branch as "an operationalized anti-grokking instrument" —
  (source: docs/topics/reference/grokking-and-hidden-progress.md).

**Basin identification, mode connectivity, and re-basin (GEO-2, GEO-5)**

- **The operational definition on record**: linear mode connectivity — interpolate two weight
  sets; if performance along the path stays comparable to the endpoints, they probably lie in
  the same basin — (source: docs/topics/reference/landscape-literature.md).
- **Frankle et al., *Linear Mode Connectivity and the Lottery Ticket Hypothesis*** (no ID) —
  same-run-early-split models are linearly connected; the precedent for GEO-opt-5's
  seed-split timing — (source: docs/topics/reference/landscape-literature.md;
  identifiability-literature.md; staging/checkpoint-tomography.md).
- **Entezari et al., *The Role of Permutation Invariance in Linear Mode Connectivity***
  (conjecture, no ID) — (source: docs/topics/reference/landscape-literature.md).
- **Ainsworth et al., *Git Re-Basin: Merging Models modulo Permutation Symmetries*** (no ID) —
  independently trained models are connected *only after* permutation alignment; GEO's
  alignment step — (source: docs/topics/reference/landscape-literature.md).
- ***Unveiling Linear Mode Connectivity of Re-Basin from a Neuron Distribution Perspective***
  (no ID) — re-basin methods "often reduce barriers only marginally and work poorly early in
  training, with no unified theory of when they succeed"; **the source of GEO's stated
  analysis risk** — (source: docs/topics/reference/landscape-literature.md;
  docs/potential-projs/landscape-geometry.md §2, §4).
- ***Going Beyond Linear Mode Connectivity: The Layerwise Linear Feature Connectivity*** (no
  ID) — connectivity in activation space, not just loss; GEO-opt-4 —
  (source: docs/topics/reference/landscape-literature.md).
- ***On the Emergence of Cross-Task Linearity in the Pretraining-Finetuning Paradigm*** (no
  ID) — models fine-tuned from a common checkpoint stay in a shared linear regime — "precisely
  why task arithmetic and model souping work, and why they *only* work within a basin" —
  (source: docs/topics/reference/landscape-literature.md).
- ***Model soups*** (Wortsman et al., no ID) — the merging result that depends on the same
  basin condition — (source: docs/topics/reference/landscape-literature.md).
- ***Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability***
  (2026, no ID) — explains basin structure via consistent feature-to-neuron assignment across
  seeds; "would, if it matures, give a principled answer to 'when are two models' internal
  metrics comparable'" — (source: docs/topics/reference/landscape-literature.md;
  identifiability-literature.md).
- **Layer-wise LMC** (arXiv 2307.06966) — per-layer barriers are insignificant relative to the
  full-model barrier and **middle layers create the barrier**, predicting interface-only
  perturbations are near-barrier-free; the ready test behind GEO-opt-6 —
  (source: docs/topics/reference/reinit-and-transfer-literature.md;
  docs/potential-projs/landscape-geometry.md §4 2026-08-22 entry).
- **LMC of MoEs** (arXiv 2509.11348) — (source: docs/topics/reference/reinit-and-transfer-literature.md).
- ***Landscaping LMC*** (arXiv 2406.16300) — (source: same).
- ***The Butterfly Effect*** (arXiv 2506.13234) — trajectories highly sensitive to initial
  conditions, motivating many seeds for any barrier study — (source: same).
- **REPAIR (activation renormalization)** (no ID) — part of the symmetry-quotienting toolkit
  alongside Git Re-Basin — (source: docs/topics/reference/identifiability-literature.md).
- **MoE comparability warning** — the symmetry group includes expert permutations, so naive
  interpolation barriers, checkpoint merging, and stitching all need an expert-alignment step
  and MoE re-basin is immature; "how to quotient MoE symmetries for checkpoint comparison" is
  recorded as an open gap — (source: docs/topics/reference/moe-literature.md).

**The comparability precedent and the gap statement**

- **Juneja et al., *Linear Connectivity Reveals Generalization Strategies*** (ICLR 2023, no
  ID) — fine-tuned models cluster into distinct linearly-connected basins; models in
  *different* basins implement *different generalization strategies* (heuristic vs.
  syntax-sensitive on NLI) despite similar in-distribution accuracy. The record calls it "the
  strongest existing evidence" that same-metric-value-different-basin can mean different
  mechanisms — (source: docs/topics/reference/landscape-literature.md;
  docs/potential-projs/landscape-geometry.md §4).
- **The recorded gap statement** (2026-08-18 Research Trajectory, agent-supplied, unverified):
  "Nobody has connected either literature to *metric validity*: no paper says 'ICL scores /
  task vectors / plasticity statistics are comparable iff models pass test X.'" Its design
  consequence — log raw and aligned barriers for every compared pair, report comparisons
  conditional on barrier height — (source: docs/topics/reference/landscape-literature.md;
  docs/potential-projs/landscape-geometry.md §4).
- **Danielle's own project seed** (verbatim, the `→` Notion flag): "Treat basin membership as
  a covariate of proxy-metric validity: test whether 'metrics are comparable iff linear mode
  connectivity'… If recipe effects on ICL-ability only hold within low-barrier pairs, that's a
  finding. If they hold across basins, that's a stronger one." —
  (source: docs/potential-projs/landscape-geometry.md §4; docs/danielle-inputs.md).
- **"There's no settled scalar measure of 'same basin' or 'on the river'"** — what exists is a
  toolkit of partial pairwise tests: interpolation barrier (raw and aligned), the
  convex-vs-monotone signature, feature-space connectivity, curve-collapse —
  (source: docs/topics/reference/landscape-literature.md).

**Identifiability reading of the barrier pair**

- **The raw-vs-aligned interpretation**: raw-barrier-high/aligned-barrier-low = "same solution
  class, different parameterization" (benign); aligned-barrier-high = genuine solution-class
  divergence (the real scar). Report both and their gap —
  (source: docs/topics/reference/identifiability-literature.md;
  docs/potential-projs/landscape-geometry.md §4 2026-08-18 entry).
- **Basin distinctness as residual non-identifiability** — permutation alignment, Git
  Re-Basin, and the 2026 neuron-identifiability framework quotient out the *known* symmetry
  group so that whatever variation remains is real underdetermination —
  (source: docs/topics/reference/landscape-literature.md; identifiability-literature.md).
- **Sussmann 1992** (no ID) — single-hidden-layer uniqueness up to permutation/sign; ReLU
  positive rescalings; attention-head permutations — the symmetry group to quotient —
  (source: docs/topics/reference/identifiability-literature.md).
- **Roeder, Metz & Kingma 2021, *On Linear Identifiability of Learned Representations*** (no
  ID) — fit the optimal linear map between two models' representations; the residual is the
  identifiability gap; the weight-free complement to interpolation —
  (source: docs/topics/reference/identifiability-literature.md;
  docs/potential-projs/landscape-geometry.md §4).
- **Model stitching — Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021** (no IDs) — a
  trained adapter letting A's bottom half drive B's top half at low penalty means functional
  interchangeability at that depth; ground truth for functional comparability —
  (source: docs/topics/reference/identifiability-literature.md; reinit-and-transfer-literature.md).
- **CKA (Kornblith et al.)** (no ID) — the scalable proxy, which "can be dominated by a few
  directions and disagree with stitching" —
  (source: docs/topics/reference/identifiability-literature.md).
- **Huh et al. 2024, *Platonic Representation Hypothesis*** (no ID) — representations of
  independently trained models converge with scale; read as "identifiability improves with
  scale," which would predict cross-recipe barriers shrink as models grow —
  (source: docs/topics/reference/identifiability-literature.md).
- **Hyvärinen & Pajunen 1999; Khemakhem et al. iVAE 2020; Schölkopf et al., *Toward Causal
  Representation Learning*; Brehmer et al. 2022; Ahuja et al.; von Kügelgen et al. 2021** (no
  IDs) — the CRL/nonlinear-ICA formalism in which basin distinctness is residual
  non-identifiability — (source: docs/topics/reference/identifiability-literature.md).
- **Implicit-bias / simplicity-bias and shortcut-learning literatures** (no IDs) — the
  selection principle that decides which basin a run lands in —
  (source: docs/topics/reference/identifiability-literature.md).

**Commitment timing (GEO-opt-5) and its causal cousin**

- **Twin-branch probe** (checkpoint tomography instrument 2) — spawn two children from one
  checkpoint with different SGD noise/data order, train both, measure the barrier;
  "barrier-between-siblings is a 'have we committed yet' statistic, and the checkpoint time at
  which it collapses is a commitment clock… the branch version makes it causal and
  controllable," with GEO-opt-5 as "the free observational cousin." Caveats on record: the
  original trains children to completion, short-child variants are not standardized, mostly
  pre-LLM-scale vision work, and nobody has run it across data recipes —
  (source: docs/topics/staging/checkpoint-tomography.md; docs/potential-projs/landscape-geometry.md §4).
- **The critical period as "the window before basin commitment"** — combining critical periods
  with the LMC fact that checkpoints become linearly connected only after early training
  stabilizes; the river is chosen early while Fisher information is high —
  (source: docs/topics/reference/critical-periods.md; landscape-literature.md).
- **The four-events question** — Fisher-trace peak, Achille's critical period, LMC onset
  between sibling runs, and induction-head/ICL emergence "are all claimed to live in the same
  early window — but no one has measured them *together* on one set of runs"; GEO-opt-5 is
  the observational half, `intervention-grid.md` the interventional one —
  (source: docs/potential-projs/landscape-geometry.md §4; docs/potential-projs/intervention-grid.md §4).
- **Local learning coefficient / devinterp SGLD probe** (Watanabe; Lau, Murfet et al.'s
  estimator; Timaeus) — short SGLD chains around a checkpoint measure local degeneracy;
  tracked across Pythia-style checkpoint sequences and shown to detect developmental
  transitions; "far cheaper than 1/16 of a run" —
  (source: docs/topics/staging/checkpoint-tomography.md; identifiability-literature.md).
- **Fort et al., *Deep Learning versus Kernel Learning*** (arXiv 2010.15110) — rapid early NTK
  rotation then stabilization; a commitment-event precedent and an eNTK-side clock —
  (source: docs/topics/reference/identifiability-literature.md; ntk-literature.md).
- **Single-checkpoint geometry statistics** (no IDs) — a scalable critical-sharpness statistic
  showing progressive sharpening at scale (even applied to data-mixing decisions), and the
  basin-emergence line showing LLMs become progressively more resilient to random parameter
  perturbations with scale, with pretraining forming a basic-capability basin and fine-tuning
  specific-capability basins inside it — "useful as covariates" —
  (source: docs/topics/staging/checkpoint-tomography.md; docs/potential-projs/landscape-geometry.md §4).

**Annealing, decay branches, and GEO-opt-3**

- **The decay branch as a wall-height meter** — Wen et al. branch off a constant-LR run at 20B
  tokens, decay for 5B tokens, then interpolate between checkpoints; their WSD-S variant
  builds a training procedure out of resuming from decayed checkpoints. "Branch + decay +
  measure the loss drop" is established; what is not is doing it on cosine mid-run checkpoints
  and treating the per-token profile as the statistic —
  (source: docs/topics/staging/checkpoint-tomography.md).
- **Hägele et al. 2024, *Scaling Laws and Compute-Optimal Training Beyond Fixed Training
  Durations*** (arXiv 2405.18392) — the stable-phase + decay-branch methodology used for
  scaling-law fitting; the same instrument as GEO-opt-3's branch variants —
  (source: docs/topics/reference/schedules-and-annealing-literature.md; staging/checkpoint-tomography.md).
- **MiniCPM** (arXiv 2404.06395) — gradient dynamics across the decay phase; annealing-branch
  methodology — (source: docs/topics/reference/schedules-and-annealing-literature.md).
- **Checkpoint merging as pseudo-annealing — WSM; Nemotron 3** (no IDs) — decay-weighted
  sliding-window merges of preceding checkpoints, exactly the `merged:<cfg>` variants
  GEO-opt-3 would use as interpolation endpoints —
  (source: docs/topics/reference/schedules-and-annealing-literature.md;
  docs/potential-projs/landscape-geometry.md §1).
- **Tissue et al.** (arXiv 2408.11029) and **arXiv 2508.01483** — LR-annealing scaling laws
  with an annealing-area term — (source: docs/topics/reference/schedules-and-annealing-literature.md).
- **TREC — Bergsma et al.** (arXiv 2509.25380) — training re-evaluation curves; place the
  anneal at the receptivity valley — (source: same).
- **Blakeney et al., *Does your data spark joy?*** (no ID) — domain upsampling at the end of
  training — (source: same).
- **Annealed readouts (`ANN`, internal)** — produces exactly the annealed variants GEO-opt-3
  needs; "if they exist, reuse them"; portfolio note that GEO-opt-3 is folded into the
  flagship as a section with a causal knob —
  (source: docs/potential-projs/landscape-geometry.md §1, §4; docs/portfolio-rankings.md).

**Reset-side items and substrate (GEO-opt-6)**

- **Gap G3 — "Is an interface reset basin-preserving?"** — layer-wise LMC predicts yes; nobody
  has reset an interface and measured the barrier back to the pre-reset solution; recorded as
  "the single best-shaped question for Danielle's program." The interpolation tool here is the
  instrument — (source: docs/topics/reference/reinit-and-transfer-literature.md;
  docs/potential-projs/landscape-geometry.md §4; docs/potential-projs/embedding-reset-dynamics.md RESET-opt-1).
- **PolyPythias (van der Wal et al., ICLR 2025)** (arXiv 2503.09543) — 50 runs, 9 seeds × 5
  sizes (14M–410M), ~7,000 checkpoints; "this is the substrate" for many-seed barrier work —
  (source: docs/topics/reference/reinit-and-transfer-literature.md;
  docs/potential-projs/landscape-geometry.md §4).
- **ITER (Igl et al., ICLR 2021, arXiv 2006.05826)** — distillation into a fresh network is
  "the one reset that leaves the basin by construction"; the contrast case for
  basin-preserving resets — (source: docs/topics/reference/landscape-literature.md;
  reinit-and-transfer-literature.md).
- **Artetxe et al. 2020; the vocab-swap study (OpenReview MsjB2ohCJO1); WECHSEL; FVT; FOCUS;
  ZeTT (arXiv 2405.07883); OMP transplantation (2506.06607); MATT (2510.21954); *Beyond
  Initialization Loss* (2608.03494); Dagan, Synnaeve & Rozière (2402.01035)** — the
  interface-reset literature whose recovered models GEO-opt-6 would measure barriers against —
  (source: docs/topics/reference/reinit-and-transfer-literature.md).
- **Zaidi et al., *When Does Re-initialization Work?*** (arXiv 2206.10011); **plasticity
  injection** (arXiv 2305.15555) — reset-family context for the body-reset contrast arm —
  (source: same).

**Metric-validity consumers — what GEO's stratification would condition**

- **Task vectors / task arithmetic (Ilharco; Zhou; Theseus; quantization) and activation-space
  ICL task vectors (Dong; Yang)** (no IDs) — the weight-space observables whose comparability
  the basin condition governs — (source: docs/topics/reference/task-vectors.md).
- **ICL curves and induction-head strength (Olsson et al.)** (no ID) — the elicitability
  observables Danielle's seed proposes to report conditional on barrier height —
  (source: docs/potential-projs/icl-elicitability.md; docs/potential-projs/landscape-geometry.md §4).
- **Plasticity statistics (Lyle's curvature/feature-rank/dead-unit panel; Dohare 2306.13812)**
  — the third class of "mechanism-level metrics" the record says may not be comparable across
  basins — (source: docs/topics/reference/plasticity.md; landscape-literature.md).
- **DataDecide (Magnusson et al., arXiv 2504.11393)** — the suite supplying the pairwise recipe
  decisions and proxy-metric correlations GEO stratifies by barrier height —
  (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/potential-projs/datadecide-data-card.md).
- **Patel, Reddy, Mosbach & Bahdanau 2026** (arXiv 2605.18607) — a published consumer of the
  DataDecide proxy-metric claims that GEO's conditional analysis would be applied to;
  SciSpace-recorded, unverified —
  (source: docs/topics/reference/small-scale-evaluation-metrics-literature.md).
- **Grokking / double descent as matched-loss cautions** — two checkpoints matched on train
  *and* test loss can sit at different points of hidden circuit maturity, so matched-loss
  pairs are "necessary but provably insufficient"; epoch-wise double descent means capability
  is not monotone in loss — the non-geometric half of the same comparability worry —
  (source: docs/topics/reference/grokking-and-hidden-progress.md).

**Readouts and covariates GEO could add**

- **eNTK readouts named for GEO** (methods, no papers) — top-k spectrum and effective rank at
  checkpoints; kernel velocity ‖Θ_t − Θ_{t−1}‖_F/‖Θ_t‖_F; kernel–target alignment; on a fixed
  probe set of a few hundred examples — (source: docs/topics/reference/ntk-literature.md).
- **Jacot et al. 2018** (arXiv 1806.07572) — the NTK limit; condition number as a trainability
  heuristic; unchecked companion IDs in the same overview (2104.03093, 2305.14585, 2406.18800,
  2502.02870) — (source: docs/topics/reference/ntk-literature.md).
- **u-µP (Blake et al.)** (arXiv 2407.17465, Danielle-supplied) — the only GEO-tagged ledger
  row; the accompanying note that **DataDecide is not µP-parametrized**, so cross-size LR is
  a confound in any cross-scale barrier comparison —
  (source: docs/litreview/citation-verification-ledger.md;
  docs/topics/reference/parametrization-and-hp-transfer.md).
- **Pair-selection control on record** — "equal loss at different token counts vs. equal tokens
  at different loss are different controls, and you'll want both"; keep matched-compute and
  matched-loss pairs distinct in every stratified analysis —
  (source: docs/potential-projs/landscape-geometry.md §4 2026-08-18 entry).
- **Stage-dependent data value hook** — the barrier tooling can test "whether late-injected
  components land the model somewhere geometrically different than early-injected ones";
  component timing → landscape position is recorded as "completely unoccupied" (agent-supplied)
  — (source: docs/potential-projs/landscape-geometry.md §4; functional-featurization.md).
- **Li et al. 2018 loss-surface visualizations** (no ID) — the landscape-metric comparison
  point for skip/BN ablations, flagged "unverified here, check before citing" —
  (source: docs/past-projects/cnn-deconstruction-ladder.md).

**Portfolio placement and risk on record (not positioning)**

- **Portfolio rankings**: GEO is **Tier 3 (component)** on the 6–12-month flagship list —
  "standalone drops (the 'all cross-recipe barriers are high' degenerate outcome is too
  likely), but [GEO-opt-3] — does annealing collapse barriers between recipes — is a genuinely
  great section inside [the flagship], with a causal knob"; not in the workshop-sized or
  full-conference lists — (source: docs/portfolio-rankings.md;
  docs/potential-projs/landscape-geometry.md §4).
- **Alternate #5 in a top-5 workshop list**: "the raw-barrier core is the alternate —
  evals-only, but your own risk analysis (all cross-recipe barriers high → degenerate
  stratification) is real" — (source: docs/potential-projs/landscape-geometry.md §4).

**Provenance caveats for this corpus**

- Almost every item above entered the repo through the **2026-08-18 Research Trajectory
  conversations** and the **2026-08-22 reinit literature pass**; both source files carry the
  standing header that related-work claims are unverified unless an identifier is given —
  (source: docs/topics/README.md; docs/potential-projs/landscape-geometry.md §5).
- The **only GEO-tagged row** in `docs/litreview/citation-verification-ledger.md` is
  **2407.17465** (u-µP, Danielle-supplied), still unverified — (source: that ledger).
- `docs/topics/staging/checkpoint-tomography.md` records its own **prior-art check still to
  do** over the devinterp and WSD-followup communities' 2025–26 output — (source: that file).

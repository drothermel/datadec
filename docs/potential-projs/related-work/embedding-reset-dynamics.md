# embedding reset dynamics — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`embedding-reset-dynamics.md`](../embedding-reset-dynamics.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

High-recall corpus for RESET (embedding-reset dynamics). Every paper, method, or named
prior-art item on record anywhere in this repository that is *possibly* relevant to
recovery curves, input/output asymmetry, basin tests, and the plasticity arm. Grouped by
theme; one line per item with its repo source. Err toward inclusion. No positioning
claims. The bulk of this corpus came from a single 2026-08-22 Opus-subagent literature
pass whose verdicts rest on abstracts only — no full-PDF reads, no forward-citation sweep
(full report: `~/drotherm/data/.claude/datadec/2026-08-22/0031-reinit-transfer-litpass.md`).

**The lineage the record says landed on Danielle's early result**

- **Artetxe et al. 2020, On the Cross-lingual Transferability of Monolingual
  Representations** (no ID on record) — transferred a monolingual model to new languages
  by retraining *only* the embedding layer against a frozen body; "your hypothesis in
  cross-lingual form" — (source:
  docs/topics/reference/reinit-and-transfer-literature.md).
- **How to Do a Vocab Swap? A Study of Embedding Replacement for Pre-trained
  Transformers** (OpenReview MsjB2ohCJO1; **no confirmed venue**) — embeddings for a
  swapped vocabulary are re-learnable even from naive initialization, even across source
  languages, with smart inits helping — (source: same).
- **Lu et al. 2021, Pretrained Transformers as Universal Computation Engines** (arXiv
  2103.05247; "Frozen Pretrained Transformer" is the model name) — froze the body and
  swapped input/output layers for non-language tasks; the claim Danielle's 2021 paper
  rebuts — (source: same).
- **Chinese LLaMA-2 (Efficient and Effective Text Encoding for Chinese LLaMA and
  Alpaca)** (no ID on record) — vocabulary extension plus continued pretraining as
  standard industrial practice — (source: same).
- **Korean SOLAR / Phi-2 (Efficient and Effective Vocabulary Expansion Towards
  Multilingual LLMs)** (no ID on record) — same practice — (source: same).
- **ALLaM, Arabic bilingual LLaMA** (arXiv 2407.15390) — same practice; **mechanism not
  confirmed** by the verification pass — (source: same).

**Danielle's own entry (verified: her paper) and the frozen-body thread (G5/G6)**

- **Rothermel, Li, Rocktäschel & Foerster, *Don't Sweep your Learning Rate under the Rug:
  A Closer Look at Cross-modal Transfer of Pretrained Transformers*** (arXiv 2107.12460,
  ICML 2021 SSL workshop; **confirmed as stated**) — Lu et al.'s frozen-vs-scratch claim
  was an LR-tuning artifact; with tuning, pretrained transformers match or beat scratch
  but *only under full fine-tuning*, frozen variants often greatly lagging; CIFAR10-LRA
  positive result; reported with error bounds across 3 seeds. The load-bearing rebuttal —
  (source: docs/topics/reference/reinit-and-transfer-literature.md;
  docs/danielle-inputs.md Toggle 14).
- **The recorded "under-absorbed" claim** — no 2022–2026 paper systematically re-audits
  frozen-vs-finetuned claims for LR-tuning asymmetry (gap G5), resting on keyword absence
  rather than a forward-citation sweep of 2107.12460 — (source: same;
  docs/topics/staging/frozen-body-transfer-audit.md).
- **Frozen-LM multimodal** (arXiv 2106.13884) — frozen-body transfer claim in the same
  structural family — (source: docs/topics/reference/reinit-and-transfer-literature.md).
- **X-Fusion** (ICCV 2025; arXiv 2504.20996) — modern frozen-body claim named as an audit
  target for G5 — (source: same; docs/topics/staging/frozen-body-transfer-audit.md).
- **Decoding PDEs** (arXiv 2510.05278) — frozen-body PDE adaptation, same audit target —
  (source: same).
- **Frozen in Time** (arXiv 2508.18130) — frozen *random* dynamics as a reservoir; "a live
  confound that most frozen-body transfer papers do not rule out"; supplies the reservoir
  null for G5 — (source: same).
- **Fine-Tuned In-Context Learners for Efficient Adaptation** (Bornschein, Lyle, Pascanu
  et al., no ID on record) — the modern literature landing on the 2021 dividing line
  (prompt-based excels few-shot, plateaus as data grows); "the direct descendant" of
  Rothermel et al. — (source: docs/topics/reference/reinit-and-transfer-literature.md;
  plasticity.md).

**Initialization methods (RESET-4, G10)**

- **WECHSEL** (no ID on record) — cross-lingual embedding initialization — (source:
  docs/topics/reference/reinit-and-transfer-literature.md).
- **FVT, Fast Vocabulary Transfer for Language Model Compression** (no ID; **method
  confirmed, exact citation not**) — (source: same).
- **FOCUS** (no ID on record) — new-token embeddings as sparse combinations of
  semantically overlapping old tokens — (source: same).
- **ZeTT, Zero-Shot Tokenizer Transfer** (Minixhofer, Ponti, Vulić; NeurIPS 2024; arXiv
  2405.07883) — a hypernetwork predicting embedding matrices for arbitrary tokenizers;
  finds prior init heuristics near-chance in true zero-shot transfer — (source: same).
- **OMP tokenizer transplantation** (Goddard & Fernandes Neto; arXiv 2506.06607) —
  training-free; beats WECHSEL/FOCUS/ZeTT on zero-shot preservation — (source: same).
- **MATT** (Haltiuk & Smywinski-Pohl; arXiv 2510.21954) — distills source→target attention
  patterns as warm-up; the first method treating the *body's dynamics*, not just embedding
  geometry, as the target — the one init method that is not purely an interface
  intervention — (source: same).
- **Teaching Old Tokenizers New Words** (Purason et al.; EACL 2026 Findings; arXiv
  2512.03989) — (source: same).
- **Dobler & de Melo, academic-budget adaptation** (arXiv 2408.15793) — (source: same).
- **Convex-hull initialization** (arXiv 2407.05841) — (source: same).
- **Learned Embedding Propagation** (arXiv 2412.21140) — CPT of embeddings "redistributes
  existing language knowledge among new tokens"; a mechanism claim for what recovery
  actually does — (source: same).

**The recovery-cost estimates RESET-1 exists to reconcile (G1, G2)**

- **Beyond Initialization Loss** (arXiv 2608.03494, 2026) — >20 init strategies on a 30B
  MoE; best init gives a 6× CPT reduction; **input and output embeddings have distinct
  optimal inits** (the RESET-2 observation, reported as a tuning finding with no
  mechanism); init loss/BPB unreliable as convergence predictors while ~50 CPT steps are
  reliable (the basis for G10); the "500 steps" end of the estimate range — (source:
  docs/topics/reference/reinit-and-transfer-literature.md;
  docs/potential-projs/embedding-reset-dynamics.md §4).
- **EEVE** (arXiv 2402.14714) — proficiency within 2B tokens, "explicitly contra…
  trillions of training tokens"; the middle of the estimate range — (source: same).
- **Dagan, Synnaeve & Rozière, Getting the Most Out of Your Tokenizer for Pre-training and
  Domain Adaptation** (ICML 2024; arXiv 2402.01035) — the ">50B tokens to recover" reading
  is a **recorded misreading**, corrected to a statement about when tokenizer
  specialization *pays off*; the seeded framing inverts it — (source: same).
- **The pass's summary statement** — continued-pretraining cost estimates "range across
  three orders of magnitude (500 steps to >50B tokens) with no controlled study
  reconciling them" — the sentence RESET-1 is built from — (source: same).

**Reset methods, plasticity, and the regularization control (RESET-3, opt-2, opt-3)**

- **Zaidi, Berariu, Kim, Bornschein, Clopath, Teh, Pascanu, *When Does Re-initialization
  Work?*** (arXiv 2206.10011) — >15,000 vision models; reinit helps without other
  regularization, little once regularization is tuned, significantly under label noise;
  the control RESET-3 exists to run in LMs (gap G8) — (source:
  docs/topics/reference/reinit-and-transfer-literature.md; plasticity.md).
- **Nikishin et al., *Deep RL with Plasticity Injection*** (arXiv 2305.15555) — "the most
  directly borrowable *instrument*": if injection helps, plasticity was the binding
  constraint; the RESET-opt-2 diagnostic — (source: same).
- **Reset & Distill** (Ahn et al.; arXiv 2403.05066) — reset-based continual-learning
  method — (source: same).
- **Plasticity-loss survey in RL** (arXiv 2411.04832) — last-layer resets are standard;
  the "plasticity loss is concentrated in the last layers" belief is recorded as *belief,
  not evidence* — the opening for RESET-opt-3 (gap G9) — (source: same).
- **Can Scale Save Us From Plasticity Loss in LLMs?** (Hernandez-Garcia, Figliolia,
  Millidge; arXiv 2606.24752, June 2026) — 5M–314M; plasticity loss follows a sublinear
  scaling law; scale delays but does not prevent it, in continual **and stationary**
  settings; the checkpoints-with-measured-plasticity-loss precondition RESET-opt-2 needs —
  (source: same; docs/topics/reference/plasticity.md).
- **Calibrated partial resets** (arXiv 2607.24996) — the closest existing work to
  RESET-opt-3's layer-wise ablation — (source: same).
- **Spectral collapse** (arXiv 2509.22335) — a plasticity-loss mechanism candidate —
  (source: same).
- **Activation-function design** (arXiv 2509.22562) — plasticity-preserving architecture
  choice — (source: same).
- **Fisher-guided selective forgetting** (arXiv 2502.00802) — targeted parameter
  forgetting; the Fisher-weighted cousin of a layer-wise reset — (source: same).
- **ITER** (Igl, Farquhar, Luketina, Böhmer & Whiteson, ICLR 2021; arXiv 2006.05826) —
  repeatedly distilling the current policy into a freshly initialised network; "the only
  one that separates *function* from *trajectory*"; students generalize better than
  teachers, evidence the damage lives in parameter-space history not the function; the
  one reset that leaves the basin by construction — the contrast case for the
  basin-preserving reading — (source:
  docs/topics/reference/reinit-and-transfer-literature.md; landscape-literature.md;
  nonstationarity-accounting.md; docs/danielle-inputs.md Toggle 14).
- **Continual Backprop: SGD with Persistent Randomness** (no ID on record) — selective
  reinitialization of dormant/unuseful units; the "least-used units" arm of RESET-opt-3
  and, in the critical-period frame, an intervention that artificially reopens the period
  — (source: docs/topics/reference/critical-periods.md;
  docs/topics/reference/plasticity.md).
- **Dohare et al., *Loss of plasticity in deep continual learning*** (Nature 2024; earlier
  arXiv 2306.13812) — the continual-backprop source and half of the "Dohare/Lyle panel"
  RESET-opt-2 runs — (source: docs/topics/reference/plasticity.md).
- **Lyle et al., *Understanding Plasticity in Neural Networks*** (ICML 2023; arXiv
  2303.01486) — plasticity loss tied to loss-landscape curvature, often without saturated
  units; supplies curvature, feature rank, dead units, weight norm to the panel — (source:
  same).
- **Lyle et al., *Disentangling the Causes of Plasticity Loss*** (arXiv 2402.18762) —
  follow-up — (source: same).
- **Achille, Rovere & Soatto, *Critical Learning Periods in Deep Networks*** (ICLR 2019) —
  Information Plasticity with the **Fisher trace** as diagnostic; predates Dohare by five
  years; the Fisher trace is the fifth member of the panel — (source:
  docs/topics/reference/critical-periods.md; plasticity.md).
- **Critical Learning Periods for Multisensory Integration in Deep Networks** (no ID on
  record) — critical periods arise from unstable early transient dynamics decisive of
  final performance and representations — (source:
  docs/topics/reference/critical-periods.md).
- **Ash & Adams, *On Warm-Starting Neural Network Training*** (NeurIPS 2020) — warm-started
  models generalize worse than re-initialized ones at similar training loss;
  shrink-and-perturb as the fix; gradient-norm imbalance diagnosed as "a symptom, not a
  mechanism" — the reset family's other canonical control — (source:
  docs/topics/reference/plasticity.md).
- **DASH: Warm-Starting NN Training in Stationary Settings without Loss of Plasticity**
  (NeurIPS 2024) — theory for the stationary case (memorized noise; shrinking should be
  direction-aware) and the claim that non-stationarity-motivated plasticity fixes are
  *ineffective* in stationary settings — a caution for RESET-opt-2's framing — (source:
  same).
- **What Can Grokking Teach Us About Learning Under Non-Stationarity** (2025, no ID on
  record) — re-warming the effective LR closes the generalization gap; dead-unit counts do
  not predict the warm-starting gap; bears directly on RESET-1's re-warm-vs-not arm —
  (source: same).

**Basin instruments (RESET-opt-1 / GEO-opt-6, gap G3)**

- **Layer-wise LMC** (arXiv 2307.06966) — per-layer barriers insignificant relative to the
  full-model barrier; **middle layers create the barrier** — the prediction that interface
  resets are near-barrier-free; the stated opening of RESET-opt-1 — (source:
  docs/topics/reference/reinit-and-transfer-literature.md;
  docs/potential-projs/landscape-geometry.md §5).
- **LMC of MoEs** (arXiv 2509.11348) — closest adjacent work, and the MoE-arm instrument —
  (source: same).
- **Landscaping LMC** (arXiv 2406.16300) — (source: same).
- **The Butterfly Effect** (arXiv 2506.13234) — trajectories highly sensitive to initial
  conditions, so reset studies need many seeds; the reason RESET-3 is a requirement rather
  than an option — (source: same).
- **Model stitching** (Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021) — recorded as
  "*literally your embedding-reset experiment* as a measurement rather than a method" —
  (source: docs/topics/reference/reinit-and-transfer-literature.md).
- **Frankle et al., *Linear Mode Connectivity and the Lottery Ticket Hypothesis*** —
  same-run-early-split models are linearly connected; the canonical barrier protocol and
  the commitment-clock reading — (source:
  docs/topics/reference/landscape-literature.md; docs/topics/staging/checkpoint-tomography.md).
- **Entezari et al., *The Role of Permutation Invariance in LMC*** and **Ainsworth et al.,
  *Git Re-Basin*** — independently trained models connect only after permutation
  alignment; the alignment step behind RESET-opt-1's "raw and aligned" barrier pair —
  (source: docs/topics/reference/landscape-literature.md).
- **Unveiling LMC of Re-Basin from a Neuron Distribution Perspective** (no ID on record) —
  re-basin methods often reduce barriers only marginally and work poorly early in
  training; the stated analysis risk for any barrier-based reset test — (source: same).
- **Going Beyond LMC: Layerwise Linear Feature Connectivity** (no ID on record) —
  connectivity in activation space, not just loss; the feature-space half of RESET-opt-1's
  CKA/stitching measurement — (source: same).
- **Juneja et al., *Linear Connectivity Reveals Generalization Strategies*** (ICLR 2023) —
  models in different basins implement different generalization strategies at similar
  in-distribution accuracy; the comparability precedent for "recovered model ≈ pre-reset
  model?" — (source: same).
- **On the Emergence of Cross-Task Linearity in the Pretraining-Finetuning Paradigm** and
  **Model soups** (Wortsman et al.) — why merging works only within a basin — (source:
  same).
- **Beyond Structural Symmetries: LMC via Neuron Identifiability** (2026, no ID on record)
  — basin structure via consistent feature-to-neuron assignment across seeds — (source:
  same).
- **The interpretive frame with no paper behind it** — "interface resets are
  basin-preserving; early-training deficits are basin-*determining*. Same operation,
  opposite regimes." The pass records its absence *as* gap G3, "the single best-shaped
  question for Danielle's program" — (source:
  docs/topics/reference/reinit-and-transfer-literature.md; landscape-literature.md;
  docs/potential-projs/embedding-reset-dynamics.md §2 table).
- **CKA** (Kornblith et al., no ID on record) — the scalable proxy for feature
  connectivity, with the recorded caveat that it can be dominated by a few directions and
  disagree with stitching — (source: docs/potential-projs/landscape-geometry.md §5).
- **Roeder, Metz & Kingma 2021** (linear-map residuals) — weight-free complement to the
  barrier pair — (source: same).
- **The identifiability reading of the barrier pair** — raw-high/aligned-low = same
  solution class, different parameterization (benign); aligned-high = genuine solution-
  class divergence — the decision rule RESET-opt-1's readout needs — (source: same).

**Substrate, stage probe, and timing (RESET-1, RESET-opt-4, gap G7)**

- **PolyPythias** (van der Wal, Lesci, Müller-Eberstein, Saphra, Schoelkopf, Zuidema,
  Biderman; ICLR 2025; arXiv 2503.09543) — 50 pretraining runs, 9 new seeds × 5 sizes
  (14M–410M), ~7,000 checkpoints; "this is the substrate" — (source:
  docs/topics/reference/reinit-and-transfer-literature.md;
  docs/potential-projs/landscape-geometry.md §5).
- **Critical periods in LM finetuning** (TACL, doi:10.1162/tacl_a_00725) — intervenes on
  *data*; the data-side counterpart to the weight-side reset probe; the closest work for
  G7 — (source: docs/topics/reference/reinit-and-transfer-literature.md;
  docs/topics/staging/checkpoint-tomography.md).
- **Smooth Scaling Laws Hide Stepwise Token Learning** (arXiv 2606.29858) — stepwise
  per-token acquisition hidden under smooth aggregate curves; a caution for reading
  recovery curves as smooth — (source:
  docs/topics/reference/reinit-and-transfer-literature.md).
- **Representation-plasticity timeline in LLMs** (arXiv 2410.06225) — when representations
  stop moving; closest work for G6 and a covariate for reset-stage choice — (source: same;
  docs/topics/staging/frozen-body-transfer-audit.md).
- **The checkpoint-tomography battery** — decay, hot, twin, and data-shifted branches, with
  the **reset branch as the candidate fifth probe** (reset an interface or block, continue
  briefly, measure recovery cost and barrier to the pre-reset model); shares the runner;
  "the 1/16 question is itself an experiment" — (source:
  docs/topics/staging/checkpoint-tomography.md).
- **The devinterp / SGLD local learning coefficient** (Lau, Murfet et al., Timaeus) — a
  per-checkpoint degeneracy scalar tracked across Pythia-style checkpoint sequences,
  detecting developmental transitions; a stage covariate for RESET-opt-4 — (source: same).
- **Critical-sharpness and basin-emergence single-checkpoint statistics** (no IDs on
  record) — progressive sharpening at scale; LLMs becoming progressively more resilient to
  random parameter perturbations, with pretraining forming a basic-capability basin and
  fine-tuning forming specific-capability basins inside it — directly relevant to whether
  an interface reset stays inside — (source: same).
- **The critical period as the window before basin commitment** — checkpoints become
  linearly connected only after early training stabilizes; "the river is chosen early,
  while Fisher information is high" — the timing frame RESET-opt-4 measures — (source:
  docs/topics/reference/landscape-literature.md; critical-periods.md).
- **The LLM-scale critical-period echo** — 2025–2026 data-placement results (early exposure
  more durable than late; final-window effects; safety behaviors resisting post-training
  removal) as critical-period phenomenology published without the connection drawn —
  (source: docs/topics/reference/critical-periods.md).
- **Task2Vec** (Achille et al., no ID on record) — the Fisher-embedding dataset
  representation; same formalism pointed at data; the featurization cousin of the Fisher
  trace in the panel — (source: docs/topics/reference/critical-periods.md).

**MoE arm and storage**

- **Slicing-and-Dicing MoE sweep** (arXiv 2605.11689; Danielle third author) — the repo
  named as the MoE arm's apparatus; final checkpoints confirmed to exist with an HF upload
  pending, **no intermediate checkpoints**, so training-dynamics analyses on the sweep
  itself are unavailable — (source: docs/potential-projs/embedding-reset-dynamics.md
  substrate note; docs/open-questions-answered.md 2026-08-21 entry).
- **Task Vector Quantization for Memory-Efficient Model Merging** (Kim et al., arXiv
  2503.06921) — task vectors have a narrow weight range and quantize to ~4 bits, with
  Residual TVQ decomposing into base + offset; the concrete method behind RESET-opt-5's
  "save branch endpoints as quantized deltas from the reset point" — (source:
  docs/topics/reference/task-vectors.md).
- **Editing Models with Task Arithmetic** (Ilharco et al., ICLR 2023; arXiv 2212.04089) —
  the weight-space delta object a reset-and-recover run produces — (source: same).

**Non-stationarity framing**

- **The three-communities-one-claim record** — Achille 2019 (vision: early deficits
  permanently impair), Ash & Adams 2020 (supervised: early data poverty permanently
  impairs), Igl 2021 (RL: transient policy non-stationarity permanently scars the
  representation), none citing the others' framing; the general form of "a reset is an
  intervention on training history" — (source:
  docs/topics/reference/nonstationarity-accounting.md).
- **Resets as stabilizers that remove non-stationarity's *history* rather than the
  non-stationarity itself** (ITER's role in the accounting frame) — (source: same).
- **The warm-starting gap as a stabilizer story** — LR re-warming, normalization, and
  single-pass training as stabilizers whose effect on a once-famous gap has never been
  decomposed; directly relevant to RESET-1's re-warm-vs-not reporting — (source: same).

**The gap list this project is built from (G1, G2, G3, G4, G8, G9, G10; siblings G5–G7)**

- **G1 recovery-cost curve** (high confidence) — no controlled curve vs. scale, stage,
  seed; cost: small runs — (source:
  docs/topics/reference/reinit-and-transfer-literature.md;
  docs/potential-projs/embedding-reset-dynamics.md §4).
- **G2 input-vs-output asymmetry explained** (high) — 2608.03494 observes it as a tuning
  finding; nobody isolates head-only vs. input-only vs. both; **weight tying is an
  uncontrolled confound** — (source: same).
- **G3 is an interface reset basin-preserving?** (high) — instruments and reset methods
  both exist, never combined; "the single best-shaped question for Danielle's program" —
  (source: same).
- **G4 does a reset restore plasticity?** (high) — plasticity literature resets *body*
  layers, tokenizer literature resets *interfaces* and never asks — (source: same).
- **G5 LR-tuning asymmetry as an unaudited confound** and **G6 how much of the body a
  frozen interface can reach** (both medium-high; split into
  `topics/staging/frozen-body-transfer-audit.md`; both rest on keyword absence and need the
  2107.12460 forward-citation sweep first) — (source:
  docs/topics/staging/frozen-body-transfer-audit.md).
- **G7 reset-response as a critical-period instrument** (medium) — absorbed as checkpoint
  tomography's fifth probe; the cross-listing for RESET-opt-4 — (source:
  docs/topics/staging/checkpoint-tomography.md).
- **G8 many-seed replication of reset effects in LMs** (medium) — 2206.10011's "disappears
  under tuned regularization" unchecked in LMs; PolyPythias makes the seed dimension free;
  the design is a factorial of reset interventions × regularization/optimizer settings ×
  seeds, "the exhaust-the-boring-explanations discipline applied to resets" — (source:
  docs/potential-projs/embedding-reset-dynamics.md §4).
- **G9 which layers actually need resetting** (medium) — no LM-side layer-wise reset
  ablation exists — (source: same).
- **G10 does init quality matter once you train long enough** (lower confidence — may
  exist) — the init literature optimizes init loss/BPB, which 2608.03494 shows is an
  unreliable convergence predictor; no explicit convergence-crossover study found —
  (source: same).

**Provenance caveats**

- **The whole 2026-08-22 pass is subagent-retrieved** — verdicts rest on abstracts and
  paper pages, no full PDFs, **no forward-citation sweep**; search skewed to recent arXiv
  HTML so 2021–2022 workshop work is under-represented; several 2026 IDs have had little
  scrutiny — (source: docs/topics/reference/reinit-and-transfer-literature.md, the
  subagent's own caveats).
- **Verification status of the seeded claims** — all verified except: Dagan et al.
  (2402.01035, reading inverted), ALLaM (2407.15390, mechanism not confirmed), FVT (method
  confirmed, citation not), the Vocab Swap study (OpenReview only, no confirmed venue);
  Lu et al.'s title corrected to *Pretrained Transformers as Universal Computation
  Engines*; Rothermel et al. confirmed — (source: same).
- **No `RESET`-tagged rows in the citation ledger** — the reinit pass predates the
  2026-08-22 SciSpace intake batch that the ledger covers, so these IDs carry the pass's
  own caveats rather than a ledger row — (source:
  docs/litreview/citation-verification-ledger.md, header and scope).
- **Danielle's verbatim origin prompt** for the lineage question (the hypothesis, the
  preliminary result, the ITER follow-up) is Toggle 14 — (source: docs/danielle-inputs.md).

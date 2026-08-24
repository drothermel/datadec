# icl elicitability — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`icl-elicitability.md`](../icl-elicitability.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*Highest-recall corpus for `ICL` (ICL-1–ICL-5, ICL-opt-1–ICL-opt-7). Every item below
is on record somewhere in this repository; inclusion here is not a relevance judgment.
Many entries come from agent-generated intake (SciSpace reviews, novelty checks, Notion
trajectory responses) and are marked unverified in-line. No positioning claims are made.*

**Emergence of ICL as a function of pretraining data (the "thesis in miniature" cluster)**

- **Chan et al. 2022, *Data distributional properties drive emergent in-context learning
  in transformers*** (no ID on record) — small transformers on Omniglot-style image
  sequences; whether ICL emerges *at all* depends on burstiness, class-distribution skew,
  and within-class variation, often at similar ordinary training loss; the direct model
  for ICL-opt-1's tiny vision/sequence tier — (source:
  docs/topics/reference/icl-literature.md; docs/potential-projs/icl-elicitability.md §4
  2026-08-18 two-tier entry; docs/refs/research-trajectory-pre-to-post-training.md).
- **Raventós et al., *Pretraining task diversity and the emergence of non-Bayesian
  in-context learning for regression*** (no ID) — task-diversity threshold for
  linear-regression ICL; the analogous "when does ICL emerge" result in the regression
  setting; the topic file flags a citation gap on the Notion source page — (source:
  docs/topics/reference/icl-literature.md).
- Cross-listed appearance of the same Chan et al. entry as a data-featurization anchor —
  (source: docs/potential-projs/recipe-featurization.md:495).

**ICL as gradient descent — the founding arc**

- **von Oswald et al., *Transformers learn in-context by gradient descent*** (no ID) —
  constructed K/Q/V matrices make one linear self-attention step identical to
  gradient-induced dynamics; the origin of the ICL–GD similarity statistic (protocol
  statistic #5) — (source: docs/topics/reference/icl-literature.md).
- **Akyürek et al., *What learning algorithm is in-context learning? Investigations with
  linear models*** (no ID) — extends the construction to ridge regression — (source:
  docs/topics/reference/icl-literature.md).
- **Dai et al., *Why Can GPT Learn In-Context? LMs Implicitly Perform Gradient Descent as
  Meta-Optimizers*** (no ID) — kernel view, ICL as implicit finetuning; defines concrete
  similarity statistics between attention-induced hidden-state updates and real
  fine-tuning gradients; recorded as "the most directly usable" statistic source —
  (source: docs/topics/reference/icl-literature.md;
  docs/refs/research-trajectory-pre-to-post-training.md).
- **Ahn et al., *Transformers learn to implement preconditioned gradient descent for
  in-context learning*** (no ID) — the preconditioned-GD refinement — (source:
  docs/topics/reference/icl-literature.md).

**Debunkings and controls the ICL–GD statistics require**

- ***In-context Learning and Gradient Descent Revisited*** (no ID) — on realistic NLP
  tasks: problematic metrics, insufficient baselines, and *untrained* models reaching
  comparable ICL–GD similarity despite no ICL; the origin of the mandatory
  untrained-model control in ICL-3 — (source: docs/topics/reference/icl-literature.md).
- ***The Initialization Determines Whether In-Context Learning Is Gradient Descent***
  (no ID) — whether ICL corresponds to GD at all depends on initialization; identified in
  tractable linear-self-attention settings — (source:
  docs/topics/reference/icl-literature.md).

**Extractable objects: task / function / state vectors (protocol statistic #4)**

- **Hendel et al., *In-Context Learning Creates Task Vectors*** (no ID) — a demonstration
  set compressed into a transportable hidden-state vector; the base object for norm,
  direction-stability, and transferability statistics — (source:
  docs/topics/reference/icl-literature.md; docs/topics/reference/task-vectors.md).
- **Todd et al., *Function Vectors in Large Language Models*** (no ID) — the sibling
  extractable object — (source: docs/topics/reference/icl-literature.md).
- ***In-Context Learning State Vector with Inner and Momentum Optimization*** (no ID) — a
  "state vector" of the ICL processing state in attention activations, explicitly compared
  with GD-trained parameters and refined with optimizer-style tricks; the closest
  activation-space analogue of a training-run state — (source:
  docs/topics/reference/icl-literature.md).
- ***Learning Task Representations from In-Context Learning*** (no ID) — ICL tasks as a
  learned weighted sum of all attention heads — (source:
  docs/topics/reference/icl-literature.md).
- **Dong, Jiang, Zhu, Ning, *Understanding Task Vectors in In-Context Learning: Emergence,
  Functionality, and Limitations*** (arXiv 2506.09048; Danielle-supplied ID, not under the
  unverified-agent caveat) — Linear Combination Conjecture; emergence in linear
  transformers on triplet-formatted prompts; *predicted and confirmed failure on high-rank
  mappings*, which constrains which ICL task families ICL-3 may use, or forces multi-vector
  injection — (source: docs/topics/reference/task-vectors.md;
  docs/topics/reference/icl-literature.md; icl-elicitability.md §4 2026-08-22).
- **Yang, Cho, Ding, Inoue, *Task Vectors, Learned Not Extracted: Performance Gains and
  Mechanistic Insight*** (ICLR 2026; arXiv 2509.24169; Danielle-supplied) — directly
  trained LTVs beat extracted TVs and act at arbitrary layers/positions; mechanism via
  attention-head OV circuits and a few "key heads"; propagation largely linear (early
  rotation, later scaling). The basis of ICL-opt-6 (learned rather than extracted vectors
  as the geometry statistic) — (source: docs/topics/reference/task-vectors.md).

**Weight-space task vectors (the other sense; the bridge to gradient-space readouts)**

- **Ilharco et al., *Editing Models with Task Arithmetic*** (ICLR 2023; arXiv 2212.04089;
  Danielle-supplied) — task vector = fine-tuned minus pretrained weights; negation and
  addition; analogy composition. The weight-space counterpart against which
  activation-space ICL vectors are defined — (source: docs/topics/reference/task-vectors.md).
- **Zhou et al., *On Task Vectors and Gradients*** (arXiv 2508.16082) — a one-epoch task
  vector is exactly the negative gradient scaled by the LR, with a bounded second-order
  error multi-epoch; the first-epoch gradient dominates the trajectory. Supplies the
  formal link between "ICL does something gradient-like" and weight-space movement —
  (source: docs/topics/reference/task-vectors.md).
- **Rinaldi et al., *Transporting Task Vectors across Different Architectures without
  Training* (Theseus)** (ICML 2026; arXiv 2602.12952) — functional (activation-level)
  characterization of a task update plus orthogonal-Procrustes alignment; relevant if ICL
  task-vector comparisons must cross model widths or recipes — (source:
  docs/topics/reference/task-vectors.md).
- **Kim et al., *Task Vector Quantization for Memory-Efficient Model Merging*** (arXiv
  2503.06921) — narrow weight range of task vectors permits 4-bit storage; practical for
  storing many checkpoint deltas cheaply — (source: docs/topics/reference/task-vectors.md).

**Measurement statistics (protocol statistics 1–3)**

- **Olsson et al., *In-context Learning and Induction Heads*** (no ID) — the original "ICL
  score" as the loss difference between an early and a late token position (protocol
  statistic #1), plus prefix-matching / copying scores on synthetic sequences as the
  mechanistic correlate measurable per checkpoint (statistic #2). Also the sharp-phase-
  transition claim ICL-opt-3's critical-period prediction rests on — (source:
  docs/topics/reference/icl-literature.md; icl-elicitability.md §4 2026-08-18 protocol
  entry; docs/potential-projs/intervention-grid.md:371,440;
  docs/refs/research-trajectory-pre-to-post-training.md).
- ***What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task
  Learning*** (no ID) — shuffled-label and format-only controls separating "the demos told
  the model which task" from "the model learned the mapping" (statistic #3); the
  decomposition recipes may affect asymmetrically — (source:
  docs/topics/reference/icl-literature.md).

**ICL vs. fine-tuning as two access routes (the elicitation-threshold cluster)**

- ***Eliciting Fine-Tuned Transformer Capabilities via Inference-Time Techniques*** (no ID;
  agent-sourced, unverified) — formal argument that SFT-acquired capabilities can be
  approximated by the base model in-context without parameter updates; the strongest
  stated version of the elicitation-ceiling premise — (source:
  docs/topics/reference/pretraining-to-posttraining.md;
  docs/refs/research-trajectory-pre-to-post-training.md).
- **Bornschein, Lyle, Pascanu et al., *Fine-Tuned In-Context Learners for Efficient
  Adaptation*** (no ID; agent-sourced, unverified) — prompt-based methods excel few-shot
  but plateau as data grows while fine-tuning keeps going; recorded as the plasticity
  group moving into ICL-vs-fine-tuning territory and as "the direct descendant" of
  Rothermel et al. 2021 — (source: docs/topics/reference/plasticity.md:59-63;
  docs/topics/reference/pretraining-to-posttraining.md:126-133;
  docs/topics/reference/reinit-and-transfer-literature.md:111-112).

**Lightweight-adaptation hybrids and the soft-prompt continuum**

- ***Context Tuning for In-Context Optimization*** (arXiv 2507.04221, 2025; agent-supplied,
  unverified) — optimize the context/prompt rather than the parameters; frozen weights,
  improved few-shot adaptation — (source: docs/topics/reference/icl-literature.md).
- ***You Only Fine-tune Once: Many-Shot In-Context Fine-Tuning for LLMs*** (title as given,
  no ID; unverified) — one lightweight fine-tuning run, then rely on ICL for later tasks —
  (source: docs/topics/reference/icl-literature.md).
- **"Industry blog posts" on LoRA adapter + in-context examples** — *no source named*;
  recorded explicitly as unsupported and to be replaced by a real source before any use —
  (source: docs/topics/reference/icl-literature.md).
- **Lester, Al-Rfou & Constant 2021, *The Power of Scale for Parameter-Efficient Prompt
  Tuning*** (no ID on record; flagged "from memory; verify") — prompt tuning matches full
  fine-tuning only above ~10B and lags badly at small sizes; the known headwind for any
  small-scale elicitation claim on DataDecide sizes — (source:
  docs/potential-projs/elicitation-gain.md §4 intake notes, which cites
  icl-elicitability.md for the general small-scale ICL weakness).
- **Soft-prompt / learned-token steering lineages — prompt tuning (Lester et al.), prefix
  tuning, P-tuning** (no IDs) — Danielle-flagged lead framing NL-ICL vs. tuned tokens as a
  continuum of "how many tuned parameters does elicitation get"; the response naming
  "certain recent research papers" is content-free and named nothing — (source:
  docs/topics/reference/icl-literature.md, undated follow-up; unverified).
- **Candidate lit-pass targets recorded but unsearched: gist/compression tokens, learned
  tool/format tokens, optimized non-readable prompt strings from discrete prompt
  optimization** (no IDs) — (source: docs/topics/reference/icl-literature.md).

**ICL scaling, many-shot, and the x-axis question (ICL-opt-7) — weak provenance**

- ***Bayesian Scaling Laws for In-Context Learning*** (arXiv 2410.16531, late 2024;
  agent-supplied, the only ID in this cluster) — accuracy-vs-shots as a scaling law —
  (source: docs/topics/reference/icl-literature.md).
- ***Scaling Laws for Many-Shot In-Context Learning with Self-Generated Annotations***
  (March 2025; no ID; unverified) — (source: docs/topics/reference/icl-literature.md).
- ***MachineLearningLM: Scaling Many-Shot ICL via Continued Pretraining*** (Sept 2025; no
  ID; unverified) — the continued-pretraining-for-ICL branch — (source:
  docs/topics/reference/icl-literature.md).
- ***Prompt Design and Repetition Strategies in In-Context Learning*** ("Hashimoto et al.,
  2025") — **the topic's reliability note says this title has the shape of an invented
  title matching Danielle's own repetition idea**; recorded only so it is not
  re-discovered as real — (source: docs/topics/reference/icl-literature.md).
- **Named researcher list (Alvarez-Melis "MIT", Percy Liang, Jacob Andreas, Tatsunori
  Hashimoto)** — **explicitly flagged as fabricated or wrong attributions** and not to be
  reused; rebuild from a real citation graph — (source:
  docs/topics/reference/icl-literature.md).
- ***In-Context Learning with Long-Context Models*** (NAACL 2025,
  aclanthology 2025.naacl-long.605; agent-supplied, unverified) — performance improves with
  more examples to a limit, then diminishing returns; the real lead for separating "more
  examples help" from "longer context hurts" (the matched-context factorial in ICL-opt-7) —
  (source: docs/topics/reference/icl-literature.md; icl-elicitability.md §4 2026-08-22).
- ***Efficient Prompting via Dynamic In-Context Learning*** (no venue, no ID; unverified,
  recorded as "adjacent at best") — adapt the number of examples to balance performance and
  cost — (source: docs/topics/reference/icl-literature.md).
- **Needle-in-a-haystack / long-context-usage benchmarking** (no named papers) — Danielle's
  framing that production context benchmarks do not predict how well an agent uses its
  context; recorded as the gap motivating the many-shot vs. long-context question —
  (source: docs/topics/reference/icl-literature.md).
- **GPT-3's n-shot curves; power-law fits to ICL accuracy vs. shots; ICL-vs-fine-tuning
  matched-budget comparisons; demonstration-ordering and calibration effects on the curve**
  (no IDs) — Danielle's own candidate lit-pass list, marked "mine, unverified" — (source:
  docs/topics/reference/icl-literature.md).
- **Demonstration repetition / duplication studies; many-shot ICL with repeated vs. unique
  examples; ICL-as-implicit-GD analyses as repetition evidence** (no IDs; "mine,
  unverified") — the direct prior-art slot for the (unique × repetitions) factorial —
  (source: docs/topics/reference/icl-literature.md).
- **GPT-3, *Language Models are Few-Shot Learners*** (no ID) — cited in the trajectory
  record as one of the field's canonical *existence proofs* (alongside AlexNet and
  R1-zero), the argument form the elicitation-ceiling program adopts — (source:
  docs/refs/research-trajectory-pre-to-post-training.md:323).

**Basin comparability — the constraint on mechanism-level ICL statistics**

- **Juneja et al., *Linear Connectivity Reveals Generalization Strategies*** (ICLR 2023;
  no ID) — basin-separated generalization strategies; the record's basis for the claim that
  task vectors and GD-similarity scores may not be comparable across basins, hence
  barrier-conditioned reporting in ICL-2 — (source: icl-elicitability.md §4 2026-08-18
  loss-basins entry; docs/topics/reference/landscape-literature.md:74;
  docs/potential-projs/landscape-geometry.md:169,191).
- **Frankle et al., *Linear Mode Connectivity and the Lottery Ticket Hypothesis***; **Entezari
  et al., *The Role of Permutation Invariance in Linear Mode Connectivity***; ***Unveiling
  Linear Mode Connectivity of Re-Basin from Neuron Distribution Perspective***; ***Going
  Beyond Linear Mode Connectivity: Layerwise Linear Feature Connectivity***; ***Beyond
  Structural Symmetries: LMC via Neuron Identifiability*** (2026) (no IDs) — the
  barrier/alignment toolkit ICL-2's matched-loss pairing must log — (source:
  docs/topics/reference/landscape-literature.md:51-68).
- ***On the Emergence of Cross-Task Linearity in the Pretraining-Finetuning Paradigm*** and
  **Wortsman et al., *Model soups*** (no IDs) — models fine-tuned from a common checkpoint
  stay in a shared linear regime, which is why task arithmetic and souping work *only*
  within a basin — (source: docs/topics/reference/task-vectors.md:88-94;
  docs/topics/reference/landscape-literature.md).

**Critical periods and the elicitability-window prediction (ICL-opt-3)**

- **Achille, Rovere & Soatto, *Critical Learning Periods in Deep Networks*** (ICLR 2019;
  no ID) — deficit windows permanently impair a skill; Information Plasticity measured by
  the Fisher information of the weights; the protocol ICL-opt-3 borrows for the deficit
  design — (source: docs/topics/reference/critical-periods.md;
  docs/potential-projs/intervention-grid.md).
- ***Critical Learning Periods for Multisensory Integration in Deep Networks*** (no ID) —
  critical periods from unstable early transient dynamics — (source:
  docs/topics/reference/critical-periods.md).
- **Task2Vec (Achille et al.)** (no ID) — the Fisher-embedding dataset representation; the
  same formalism pointed at data; listed as a shared checkpoint-pair instrument for the
  elicitability-critical-period experiment — (source: docs/topics/reference/critical-periods.md;
  docs/dataset-analysis-idea-map.md:171-172).
- ***Continual Backprop: SGD with Persistent Randomness*** (no ID) — an intervention that
  "artificially reopens" the critical period; also cited in the trajectory record as an
  existence-proof-style paper — (source: docs/topics/reference/critical-periods.md;
  docs/refs/research-trajectory-pre-to-post-training.md:323).
- **Ash & Adams, *On Warm-Starting Neural Network Training*** (NeurIPS 2020; no ID) and
  ***DASH*** (NeurIPS 2024; no ID) — the warm-starting gap and its stationary-case theory;
  the bridge entry frames matched-loss ICL as the next chapter of "does warm-starting damage
  *elicitability* too, or only accuracy" — (source: docs/topics/reference/plasticity.md:67-78;
  icl-elicitability.md §4 2026-08-18 warm-starting bridge;
  docs/potential-projs/intervention-grid.md:185).
- ***What Can Grokking Teach Us About Learning Under Non-Stationarity*** (2025; no ID) —
  effective-LR re-warming closes the generalization gap — (source:
  docs/topics/reference/plasticity.md:79-81).
- **Repeated-data double descent disproportionately damaging induction heads** (no paper
  named) — recorded twice as a mechanism claim connecting data repetition to the ICL
  circuit; relevant to ICL-opt-7's repetition axis — (source:
  docs/potential-projs/moe-recipe-suite.md:120,243;
  docs/topics/reference/regularization-literature.md:67).

**Danielle's own prior work and the frozen-interface reframing (ICL-opt-5)**

- **Rothermel et al. 2021, *Don't Sweep your Learning Rate under the Rug: A Closer Look at
  Cross-modal Transfer of Pretrained Transformers*** (arXiv 2107.12460) — frozen variants
  lag full fine-tuning under proper LR tuning; transfer through the body is real. The §4
  entry frames the whole project as "quantifying the gap your 2021 paper discovered" —
  (source: icl-elicitability.md §4 2026-08-18;
  docs/topics/staging/frozen-body-transfer-audit.md;
  docs/refs/research-trajectory-pre-to-post-training.md:286).
- **G6 closest-work citation, arXiv 2410.06225** (no title recorded) — the second-closest
  paper found in the reinit/transfer sweep for "frozen/finetuned gap as an
  elicitation-ceiling measurement"; verdict rests on abstracts, no forward-citation sweep —
  (source: docs/topics/staging/frozen-body-transfer-audit.md).
- **Modern frozen-body claims flagged as structurally identical to the 2021 setup and
  unaudited for LR-tuning asymmetry (G5): X-Fusion (arXiv 2504.20996), PDE adaptation
  (arXiv 2510.05278), frozen time-series transformers (arXiv 2508.18130 — also the source
  of the *reservoir null*: does a randomly initialized frozen body do as well?)** — the
  frozen-interface baselines an elicitation-ceiling measurement would inherit — (source:
  docs/topics/staging/frozen-body-transfer-audit.md).
- **Modern elicitation instruments named for the G6 measurement: linear probes across
  depth, CKA, sparse autoencoders, plasticity injection as diagnostic** (no papers) —
  (source: docs/topics/staging/frozen-body-transfer-audit.md).

**The code-autoencoder probe as a graded capability measure (ICL-opt-2)**

- **Round-trip reconstruction fidelity through a natural-language bottleneck** — recorded
  as a graded rather than thresholded capability probe, contrasted with pass@1; the claim
  "how compressible code is into natural language for a given model pair is a property of
  their shared representations" — (source: icl-elicitability.md §4 2026-08-18 code-autoencoder
  entry; docs/potential-projs/tiny-scale-measurement.md:360;
  docs/refs/research-trajectory-pre-to-post-training.md:71).
- **The TLC project (`text-latent-code-autoencoder.md`) and its prior-art accumulator
  `nl-bottleneck-prior-art.md`** — the full prior-art corpus for that probe, incl.
  Language Bottleneck Models (2506.16982), Proto-tokens (2502.13063 — frozen-LLM
  reconstruction from 1–2 trained embeddings, evidence reconstruction does not require an
  NL latent), GenDLN (ACL SRW 2025, DOI 10.18653/v1/2025.acl-srw.92) — (source:
  docs/topics/reference/nl-bottleneck-prior-art.md).

**Practical small-scale ICL test sharing ICL-opt-7's confound**

- **The clean-code-preference staging design** (N in-context examples + M interactive
  examples with test feedback; context-fraction axis) — its interactive-learning-curve arm
  has exactly the (unique examples × repetitions/rounds) vs. growing-context confound;
  the clean comparison is matched total context — (source:
  docs/topics/staging/clean-code-preference-icl.md).
- **AgentPack** (arXiv 2509.21891; agent-supplied, unverified) and **KODCODE** (no ID) —
  named as prior code datasets in that thread; the intake note says both are about
  correctness and agent edits, *not* style preference, and that the real prior-art question
  was not searched — (source: docs/topics/staging/clean-code-preference-icl.md).
- **Preference learning / pairwise ranking, DPO-style preference pairs** (no papers named)
  — the named objective family for the contrastive-pairs ICL condition — (source:
  docs/topics/staging/clean-code-preference-icl.md).

**Elicitation-controlled measurement, the strong null, and eval-variance context**

- **The research-hypothesis framing** — elicitation promoted from confound to instrument;
  the tuned elicitation ceiling as the *strong null model*; both readouts (raw and
  elicitation-controlled) reported with their difference as the capability-vs-accessibility
  decomposition; tuning-budget accounting and the anti-circularity caveat — (source:
  docs/research-hypothesis.md; icl-elicitability.md §4 2026-08-18 final entry).
- **Tuning-response curves** (performance vs. search budget per paradigm; flat = exhausted,
  steep = headroom) and **demonstration hygiene** for existence proofs — (source:
  docs/potential-projs/movement-microscope.md:270-285).
- ***Spurious Rewards*** (Shao et al., ICML 2026; arXiv 2506.10947) and **Yue et al.
  2504.13837** (pass@k limits) — named as existence proofs that elicitation variance
  masquerades as training effects — (source: docs/research-hypothesis.md:61-62;
  docs/potential-projs/tiny-scale-measurement.md).
- **Hochlehnert et al., *A Sober Look at Progress in LM Reasoning*** (COLM 2025; arXiv
  2504.07086) — multiple seed runs essential; benchmark noise swamps effect sizes —
  (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/potential-projs/tiny-scale-measurement.md).
- ***Signal and Noise: A Framework for Reducing Uncertainty in LM Evaluation*** (no ID) —
  noise as the model's own step-to-step wander rather than measurement error; the source of
  the finding that configuration variance (few-shot demo choice, order, template) is a
  *systematic bias axis to sweep, not noise to average*, which directly governs how ICL-1's
  ordering/sample averaging is interpreted — (source:
  docs/topics/reference/evaluation-methodology-literature.md;
  docs/potential-projs/trajectory-statistics.md:388; docs/dataset-analysis-idea-map.md:99).
- **OLMES** (*A Standard for Language Model Evaluations*) — the eval standard DataDecide
  uses; loglikelihood/rank-classification evals have no sampling nondeterminism — (source:
  docs/refs/research-trajectory-pre-to-post-training.md:426).

**Scale-ceiling and emergence context for the transfer claim (ICL-5)**

- **Huh et al. 2024, *The Platonic Representation Hypothesis*** (no ID) — read as an
  empirical claim that identifiability improves with scale; if true, recipe effects on
  elicitability should wash out as models grow, which is exactly ICL-5's risk — (source:
  docs/potential-projs/tiny-scale-measurement.md:492+).
- **Wei et al. 2206.07682 (emergent abilities)** vs. **Schaeffer et al. 2304.15004
  (emergence as a mirage)** — the threshold-vs-continuous-metric debate the ICL curve's
  "no benchmark thresholds" pitch sits inside — (source:
  docs/potential-projs/tiny-scale-measurement.md).
- **PolyPythias** (ICLR 2025; arXiv 2503.09543 — 9 seeds × 5 sizes, 14M–410M, ~7,000
  checkpoints), ***Can Scale Save Us From Plasticity Loss in LLMs?*** (2606.24752),
  ***The Butterfly Effect*** (2506.13234) — many-seed checkpoint substrates in range for a
  seed-hungry ICL protocol — (source: docs/potential-projs/tiny-scale-measurement.md;
  docs/topics/reference/reinit-and-transfer-literature.md).
- **DataDecide (Magnusson et al., arXiv 2504.11393)** — the checkpoint suite ICL-1/ICL-2
  run on — (source: docs/topics/reference/pretraining-to-posttraining.md;
  docs/litreview/citation-verification-ledger.md).
- **TinyStories (Eldan & Li 2023) and the BabyLM line** (no IDs; from-memory, unverified) —
  capability-per-parameter under distribution narrowing; the honest prior art if ICL at
  150M requires narrowed synthetic task families — (source:
  docs/potential-projs/tiny-scale-measurement.md:261,492;
  docs/potential-projs/elicitation-gain.md §4).

**Pretraining→post-training context for the proxy claim (ICL-5)**

- **Shen et al., *Understanding Reasoning from Pretraining to Post-Training*** (arXiv
  2607.16097, 2026) — lower pretraining loss strongly predicts higher post-RL pass@1 at
  fixed RL compute; recorded as *in tension* with the recipe-beyond-loss hypothesis —
  (source: docs/topics/reference/pretraining-to-posttraining.md:72-76).
- **Akter et al., *Front-Loading Reasoning*** (NVIDIA, ICLR 2026; arXiv 2510.03264);
  **Feng et al., *Early Data Exposure Improves Robustness to Subsequent Fine-Tuning***
  (arXiv 2605.12705); **Baek et al., *The Finetuner's Fallacy*** (arXiv 2603.16177) — the
  data-placement results read as critical-period phenomenology at scale — (source:
  docs/topics/reference/pretraining-to-posttraining.md:62-71,137-143).
- **Yue et al. 2504.13837; Wu & Choi, *On the Limits of RLVR*; *The Invisible Leash*
  (2507.14843); *RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs* (2506.14245)**
  — the "post-training elicits rather than adds" arc that motivates elicitation-controlled
  readouts — (source: docs/topics/reference/pretraining-to-posttraining.md:88-95).
- **Chen et al. 2505.17988** (small-scale SFT lowering MATH-500 while eliciting reasoning
  style) and **Luo et al., *Through the Valley*** (EMNLP 2025; 2506.07712) — small-model
  post-training pathologies bounding what ICL-5 could predict — (source:
  docs/topics/reference/pretraining-to-posttraining.md:96-100).
- **Tülu / Tülu 3** (no ID) — the SFT stack behind the earlier no-movement result the proxy
  is meant to see through — (source: docs/topics/reference/pretraining-to-posttraining.md).

**In-program neighbors (not literature; positioning targets)**

- **`elicitation-gain.md` (`ELI`)** — the optimizer-driven counterpart of the tuned
  elicitation ceiling; same null obtained by search rather than hand-tuned prompts —
  (source: docs/potential-projs/icl-elicitability.md:71-74;
  docs/potential-projs/elicitation-gain.md:382,616).
- **`movement-microscope.md` (`MIC`)** — the ICL curve as the gradient-free proxy candidate;
  the microscope/detection-limit framing — (source:
  docs/potential-projs/movement-microscope.md:265-270).
- **`intervention-grid.md` (`GRID`)** — GRID-opt-2 critical period for elicitability
  (cross-listed as ICL-opt-3); the distill-into-fresh-network arm behind ICL-opt-4 —
  (source: docs/potential-projs/intervention-grid.md:79-80,376-379,436-445).
- **`tiny-scale-measurement.md` (`TINY`)** — the within-reach-task and gradient-free-proxy
  entries — (source: docs/potential-projs/tiny-scale-measurement.md:340-365).
- **`landscape-geometry.md`** — the barrier-conditioning instrument for ICL-2 — (source:
  docs/potential-projs/landscape-geometry.md:169,191).
- **`irt-reanalysis.md` IRT-10** — the BoolQ format intervention as the first concrete
  instance of "apparent capability floors may be measurement floors" — (source:
  docs/potential-projs/irt-reanalysis.md:76,320).
- **`recipe-featurization.md`** — elicitation wrappers as a perturbation with the model as
  instrument — (source: docs/potential-projs/recipe-featurization.md:424).

**Provenance note**

- The ICL-topic citations largely arrived from Danielle or the Notion Research Trajectory
  page rather than the 2026-08-22 SciSpace batch, so most are **not ledgered**; the
  exceptions above (2506.09048, 2509.24169, 2212.04089, 2508.16082, 2602.12952,
  2503.06921) came with IDs from Danielle and are explicitly exempted from the
  unverified-agent-claim caveat. Nothing in
  `docs/litreview/citation-verification-ledger.md` is verified — (source:
  docs/topics/reference/task-vectors.md header;
  docs/litreview/citation-verification-ledger.md header;
  docs/potential-projs/icl-elicitability.md §5).

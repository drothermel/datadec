# functional featurization — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`functional-featurization.md`](../functional-featurization.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

High-recall corpus of every paper, method, or named prior-art item on record in this
repository that is possibly relevant to functional featurization (`FUNC`). Assembled
2026-08-24 from the project doc's §1–§5, the theme accumulators in `docs/topics/`, the
staging docs, the citation ledger, and the program notes. **Recall is the point** — items
appear on the strength of one repo mention. Every line names its repo source. Agent-
generated records (SciSpace reviews, novelty checks, NBLM tables) are flagged in-line and
all their identifiers are unverified.

**Nearest-neighbour framing as recorded (the 2026-08-21 external assessment; unverified)**

- **Influence functions / datamodels / TRAK** (no IDs) — named as the nearest neighbour on
  the attribution side, "retrospective… and mostly stage-blind" and "precise but brutally
  expensive at pretrain scale"; FUNC-opt-1 is described as "the datamodels idea lifted to
  chunk granularity and stage-conditioning" — (source:
  docs/potential-projs/functional-featurization.md §4, 2026-08-21).
- **DoReMi** (no ID) — mixture optimization at source granularity; named nearest neighbour
  on the mixture side and in the missing-canon list of targeted data selection — (source:
  docs/potential-projs/functional-featurization.md §4;
  docs/topics/reference/targeted-pretraining-midtraining-literature.md).
- **RegMix** (no ID) — the same, "prospective but operate at coarse source granularity with
  a single scalar objective" — (source:
  docs/potential-projs/functional-featurization.md §4).
- **Skill-It** (no ID) — skill graphs; "touches stage-ordering but with predefined skills";
  the closest recorded prior art on the ordering axis — (source:
  docs/potential-projs/functional-featurization.md §4).
- **WebOrganizer-style topic × format classification of web data** (no ID) — a *learned*
  taxonomy of web data; the component-decomposition precedent FUNC-1 departs from —
  (source: docs/potential-projs/functional-featurization.md §4).
- **Embedding-cluster decompositions used for mixture optimization** (no ID) — the other
  learned-taxonomy family — (source: docs/potential-projs/functional-featurization.md §4).
- **Functional decompositions closer to FUNC's instinct** (no IDs) — instruction-like text
  hiding in web crawl; reasoning-dense passages; the determinism/entropy profile at token
  granularity — (source: docs/potential-projs/functional-featurization.md §4).
- **LESS-style single-step influence of a batch on held-out losses** (no ID) — named as
  the cheapest tier of the FUNC-5 surrogate ladder — (source:
  docs/potential-projs/functional-featurization.md §4).
- **DSIR, DsDm** (no IDs) — targeted data selection, in the midtraining missing-canon list
  — (source: docs/topics/reference/targeted-pretraining-midtraining-literature.md,
  Claude-added intake note, unverified).

**Midtraining as intervention — the stage axis stated as a training recipe**

- **Zhang et al. 2025, On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning
  Language Models** (2512.07783) — under fixed compute, midtraining on task-relevant data
  moves the competence boundary more efficiently than RL-only post-training when
  pretraining leaves headroom; the one carried result and the closest published statement
  that *where in training* data lands is decision-relevant — (source:
  docs/topics/reference/targeted-pretraining-midtraining-literature.md; SciSpace-sourced,
  unverified).
- **Dery et al. 2021, Should We Be Pre-training? An Argument for End-task Aware Training**
  (no ID) — joint end-task + auxiliary training in the intermediate phase beats
  task-agnostic continued pretraining; gains depend on intermediate↔end-task alignment —
  (source: docs/topics/reference/targeted-pretraining-midtraining-literature.md,
  unverified).
- **van der Goot 2023, MaChAmp at SemEval-2023** (no ID) — diverse-many uncurated
  intermediate training gives broad modest gains; a well-matched single transfer task gives
  larger targeted gains. The curated-single vs. diverse-many tradeoff is FUNC's type axis
  in coarse form — (source:
  docs/topics/reference/targeted-pretraining-midtraining-literature.md, unverified).
- **Qiu et al. 2021 (EMNLP), further pretraining for diverse dialogue tasks** (no ID) —
  different downstream tasks want different further-pretraining objectives; selective, not
  universal, gains — (source:
  docs/topics/reference/targeted-pretraining-midtraining-literature.md, unverified).
- **Gan et al. 2023, joint domain-specific pretraining with data enhancement** (no ID) —
  reconstruct the continued-pretraining corpus around hard downstream examples (+5% avg);
  costs generality — (source:
  docs/topics/reference/targeted-pretraining-midtraining-literature.md, unverified).
- **Luo et al. 2021, meta-learning for downstream-aware pretraining** (no ID) — downstream
  task distribution signal put into the pretraining objective — (source:
  docs/topics/reference/targeted-pretraining-midtraining-literature.md, unverified).
- **Task-robust minimax pretraining** (2306.12070) — minimize worst-case rather than
  average risk over representative upstream tasks (+1.8 GLUE avg, +9.2 CoLA); a contrast
  framing for "target a suite" — (source:
  docs/topics/reference/targeted-pretraining-midtraining-literature.md, unverified).
- **DAPT / TAPT (Gururangan et al. 2020)** and **STILTs / intermediate-task transfer**
  (no IDs) — the LM canon the SciSpace review missed; named as the actual related-work
  skeleton — (source:
  docs/topics/reference/targeted-pretraining-midtraining-literature.md, Claude-added,
  unverified).
- **OctoThinker** (2506.20512) — "midtraining that makes RL scale"; in the same
  missing-canon list and flagged in the ledger as Claude-added — (source:
  docs/topics/reference/targeted-pretraining-midtraining-literature.md;
  docs/litreview/citation-verification-ledger.md).
- **Phi-style targeted synthetic pretraining** (no ID) — same list — (source:
  docs/topics/reference/targeted-pretraining-midtraining-literature.md).
- **HyperCLOVA X stage table** (no ID) — code 12%→25%, math 8.6%→25.3% across four stages,
  instruction data only late; described in the record as "folklore encoded as percentages"
  and as exactly the object FUNC proposes to measure — (source:
  docs/potential-projs/functional-featurization.md §4, named without citation).
- **Unnamed paper on learning-rate decay wasting your best data** (no ID) — cited only by
  title-gist in the §4 entry; the schedule × data interaction stated directly — (source:
  docs/potential-projs/functional-featurization.md §4, uncited).
- **Unnamed pretraining-data-ordering study** (no ID) — curriculum effects that do not show
  up in final performance; ordering shapes when capabilities emerge and reorganizes
  embedding-space structure. The single most on-point recorded result for FUNC's premise,
  and it is uncited — (source: docs/potential-projs/functional-featurization.md §4).

**Annealing-data line (the record says related work should anchor here)**

- **Llama 3 "Annealing Data"** (no ID) — 8B GSM8K +24 / MATH +6.4, 405B negligible;
  final-40B 30/70 anneal used explicitly as a data-valuation instrument — i.e. a
  single-cell, coarse-component version of the U_c(t) map — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **Blakeney et al., "Does your data spark joy?"** (no ID) — 7B end-of-training domain
  upsampling; MMLU +6.90 / GSM8K +8.26 / HumanEval +6.17 pp; 10–20% of training as the
  budget point; the recorded late-upweighting data-valuation result — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **OLMo 2 / Dolmino** (2501.00656) — mid-training mix targeting weak spots, LR to zero;
  annealing as a data-evaluation tool (30/70) — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **MiniCPM** (2404.06395) — WSD origin; ~10% decay completes convergence, new data mixed
  in strictly during decay; decay-phase gradient statistics (norm falls, consecutive-update
  cosine turns positive) as a candidate branch instrument — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **Hägele et al.** (2405.18392) — constant LR + cooldown matches cosine; (1−√) cooldown;
  decay-branch reuse as the cost model that makes FUNC's branch grid affordable — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **TREC / Training Re-evaluation Curves** (Bergsma et al., 2509.25380) — a "receptivity
  valley" where identical HQ amounts do best, predictable from AdamW's EMA timescale; the
  topic names it as "a candidate explanation for stage-dependent chunk effects," i.e. the
  closest published mechanism for FUNC's stage axis. Flagged: verify before building on —
  (source: docs/topics/reference/schedules-and-annealing-literature.md).
- **Tissue et al.** (2408.11029) — annealing-area term in the loss law; LR trajectory
  affects realized loss beyond total tokens — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **Second LR-annealing scaling-law citation** (2508.01483) — paired with Tissue; flagged
  as possibly a mis-ID — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).
- **Multi-power law** (Luo et al., 2503.12811) — predicts the loss drop from a hypothetical
  decay; the analytic version of the schedule-neutralizing readout FUNC-3 does empirically
  — (source: docs/topics/reference/schedules-and-annealing-literature.md).
- **WSM (checkpoint merging as decay-free schedule)** and **Nemotron 3** (no IDs) —
  merged checkpoints mirror a true anneal; the cheap version of FUNC-3's durable-movement
  filter — (source: docs/topics/reference/schedules-and-annealing-literature.md).
- **PDPC** (2501.13126), **AutoScale** (2407.20177), **Data Mixing Laws** (2403.16952),
  **UtiliMax / MEDU** (2501.11747) — "what to anneal on and when"; scale-dependent
  composition; the mixture-level answers FUNC's chunk-level design contrasts with —
  (source: docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Parmar et al. 2024 continued-pretraining recipe** (no ID) — two-stage CPT, switch at
  LR ≈ η_max/5, stay distribution-adjacent; a stated stage policy — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Large-scale curriculum study** (Zhang et al., 2506.11300) — 0.5–1B models; easy→hard by
  compression ratio / lexical diversity / readability, lasting gains up to +3.5%, ordering
  disentangled from selection. The strongest recorded evidence that ordering has a
  measurable effect at LM scale — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Influence-driven curricula** (2508.15475) — rank by gradient-similarity influence,
  >10 pp over random in low-resource pretraining; a gradient-surrogate curriculum, adjacent
  to FUNC-5's cheapest tier — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Temperature sampling vs. scalarization / "mixture-level cooldown"** (2410.04579) —
  heavy upsampling early, reduced later; mixture-level scheduling distinct from LR cooldown
  — (source: docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Curriculum cluster** (2405.07490, 2406.19853, 2411.02337, ADCL 2505.08364 "difficulty
  shift") — largely post-training; kept as leads because the reviewer prior about curriculum
  learning is a stated risk for FUNC — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Rewriting cluster** — SwallowCode/SwallowMath (2505.02881), ProX (2409.17115),
  FinerWeb-10BT (2501.07314), Nemotron-CC (2412.02595) — "upgraded rather than selected"
  data; a candidate chunk *type* for FUNC-1 and the subject of a staging doc — (source:
  docs/topics/staging/rewritten-anneal-slice.md;
  docs/topics/reference/schedules-and-annealing-literature.md).
- **FineWeb-Edu** (2406.17557) and **Phi-4** (2412.08905) — classifier-filtered and
  synthetic anneal data at scale — (source:
  docs/topics/reference/schedules-and-annealing-literature.md, unverified).
- **Term collisions recorded, do not re-flag:** Annealed-RLVR (2509.23629), RLHFuse,
  "data annealing" (2004.13833), VersaTune (2411.11266), O-LoRA, Self-Synthesized Rehearsal
  (2403.01244), KILO (2508.03571), LIFT, Mixture-of-Skills (2406.08811), instruction-mix
  studies (2310.05492 / 2312.10793) — (source:
  docs/topics/reference/schedules-and-annealing-literature.md).

**Task vectors — FUNC-4's direction readout and FUNC-5's surrogate ladder
(Danielle-supplied IDs, explicitly *not* under the unverified-claims caveat)**

- **Ilharco et al., Editing Models with Task Arithmetic** (ICLR 2023; 2212.04089) — the
  fine-tuned-minus-pretrained delta as a composable direction; negation and addition give
  the readout for whether two data types' effects add, cancel, or interfere — (source:
  docs/topics/reference/task-vectors.md; docs/potential-projs/functional-featurization.md §4).
- **Zhou et al., On Task Vectors and Gradients** (2508.16082) — a one-epoch task vector
  "is exactly equivalent to the negative gradient of the loss, scaled by the learning
  rate," and "the first-epoch gradient dominates the finetuning trajectory in both norm and
  direction"; direct support for FUNC-5's cheapest tier standing in for short branches —
  (source: docs/topics/reference/task-vectors.md).
- **Kim et al., Task Vector Quantization for Memory-Efficient Model Merging** (2503.06921)
  — task vectors have a narrow weight range, enabling 4-bit quantization; makes saving
  branch endpoint weights for every cell cheap — (source:
  docs/topics/reference/task-vectors.md).
- **Rinaldi et al., Theseus — Transporting Task Vectors across Different Architectures
  without Training** (ICML 2026; 2602.12952) — task-vector transport as functional matching
  on observed activations via orthogonal Procrustes; "task identity defined functionally
  rather than parametrically" is FUNC's own move stated for architectures — (source:
  docs/topics/reference/task-vectors.md).
- **Dong et al., Understanding Task Vectors in In-Context Learning** (2506.09048) — the
  Linear Combination Conjecture; predicted failure on high-rank mappings. Relevant to
  FUNC-opt-1's rank question by analogy — (source:
  docs/topics/reference/task-vectors.md).
- **Yang et al., Task Vectors, Learned Not Extracted** (ICLR 2026; 2509.24169) — trained
  task vectors beat extracted ones; TV propagation is largely linear (early TVs rotated,
  later scaled) — (source: docs/topics/reference/task-vectors.md).
- **On the Emergence of Cross-Task Linearity in the Pretraining-Finetuning Paradigm**
  (no ID) and **Model soups** (no ID) — why task arithmetic and merging work *only within a
  basin*; the boundary condition on FUNC-4's weight-space readouts — (source:
  docs/topics/reference/task-vectors.md; docs/topics/reference/landscape-literature.md).

**Plasticity, critical periods, and the predicted stage structure (FUNC-opt-2)**

- **Achille, Rovere & Soatto, Critical Learning Periods in Deep Networks** (ICLR 2019,
  no ID) — early deficits permanently impair; Information Plasticity measured by the Fisher
  trace, rising then falling; named as predicting a strongly non-uniform stage axis —
  (source: docs/topics/reference/critical-periods.md;
  docs/potential-projs/functional-featurization.md §4).
- **Critical Learning Periods for Multisensory Integration in Deep Networks** (no ID) —
  critical periods arise from unstable early transient dynamics decisive of final
  performance — (source: docs/topics/reference/critical-periods.md).
- **Dohare et al., Loss of plasticity in deep continual learning** (Nature 2024;
  2306.13812) — plasticity as a depletable resource; continual backpropagation as the fix —
  (source: docs/topics/reference/plasticity.md).
- **Lyle et al., Understanding Plasticity in Neural Networks** (2303.01486) and
  **Disentangling the Causes of Plasticity Loss** (2402.18762) — plasticity loss tied to
  loss-landscape curvature; the search for cheap training statistics (curvature, feature
  rank, dead units, weight norm) that the plasticity topic proposes as FUNC-4
  response-vector components — (source: docs/topics/reference/plasticity.md).
- **Hernandez-Garcia, Figliolia, Millidge, Can Scale Save Us From Plasticity Loss in Large
  Language Models?** (2606.24752) — 5M–314M, sublinear scaling law; scale delays but does
  not prevent plasticity loss, in continual *and stationary* settings — (source:
  docs/topics/reference/plasticity.md).
- **Nikishin et al., Deep RL with Plasticity Injection** (2305.15555) — a diagnostic: if
  injection helps, plasticity was binding — (source: docs/topics/reference/plasticity.md).
- **Continual Backprop: SGD with Persistent Randomness** (no ID) — an intervention that
  "artificially reopens" the critical period — (source:
  docs/topics/reference/critical-periods.md).
- **Ash & Adams, On Warm-Starting Neural Network Training** (NeurIPS 2020, no ID) —
  warm-started models generalize worse at similar training loss; shrink-and-perturb; read
  in the record as starting past the Fisher peak — (source:
  docs/topics/reference/plasticity.md; docs/topics/reference/critical-periods.md).
- **DASH** (NeurIPS 2024, no ID) — stationary-setting theory; argues
  non-stationarity-motivated plasticity fixes are ineffective in stationary settings, i.e.
  Dohare/Lyle mechanisms may not be the explanation — (source:
  docs/topics/reference/plasticity.md).
- **What Can Grokking Teach Us About Learning Under Non-Stationarity** (2025, no ID) —
  effective-LR re-warming closes the gap; dead units do not predict it — (source:
  docs/topics/reference/plasticity.md).
- **Igl et al., ITER** (ICLR 2021, no ID) — transient non-stationarity permanently scars
  the latent representation; grouped with Achille and Ash & Adams as the same claim from
  three subfields — (source: docs/topics/reference/nonstationarity-accounting.md).
- **Zaidi et al., When Does Re-initialization Work?** (2206.10011) — >15,000 vision models;
  reinit helps without regularization, little once tuned — (source:
  docs/topics/reference/plasticity.md).
- **Spectral collapse** (2509.22335), **activation-function design** (2509.22562),
  **calibrated partial resets** (2607.24996), **plasticity-loss survey in RL** (2411.04832),
  **Fisher-guided selective forgetting** (2502.00802) — the reinit-pass additions;
  candidate plasticity readouts — (source: docs/topics/reference/plasticity.md).
- **Bornschein, Lyle, Pascanu et al., Fine-Tuned In-Context Learners for Efficient
  Adaptation** (no ID) — the plasticity group moving into ICL-vs-fine-tuning adaptation —
  (source: docs/topics/reference/plasticity.md).
- **The LLM-scale critical-period echo** (no IDs) — 2025–26 results on early exposure
  shaping models more durably than late data, final-window effects, and pretraining safety
  behaviors resisting post-training removal, "mostly published without the connection
  drawn" — (source: docs/topics/reference/critical-periods.md).

**Token-level readout coordinates for the response vector (FUNC-4)**

- **Wen et al.** (2410.05192) — river/wall geometry; stable phase learns deterministic
  tokens, decay learns stochastic ones; the durable-vs-transient framing FUNC-3 formalizes
  — (source: docs/topics/reference/token-level-literature.md).
- **Rho-1 / Not All Tokens Are What You Need** (2404.07965) — loss-trajectory taxonomy of
  tokens across checkpoints; a token-bucket-over-time precedent and the reference-model
  scoring FUNC-1's functional features reuse — (source:
  docs/topics/reference/token-level-literature.md;
  docs/topics/reference/training-objective-alternatives-literature.md).
- **Token-Level Uncertainty-Aware Objective for Language Model Post-Training** (no ID) —
  epistemic vs. aleatoric token uncertainty and its drain over training; the entropy-bucket
  coordinates for Δ per-token loss — (source:
  docs/topics/reference/token-level-literature.md).
- **Revisiting Entropy in RL for Large Reasoning Models** (no ID) and **Beyond the 80/20
  Rule** (no ID) — token-regime masking changes dynamics; a high-entropy minority carries
  most of RLVR's effect — the wall bucket as the locus of post-training — (source:
  docs/topics/reference/token-level-literature.md).
- **Token-reweighting objectives** — MiLe, TALR (2509.20758), RFT (2412.14780), IR-DRO
  (2402.14270), Power-Law Decay Loss (2505.16900), ESLM (2505.19893), VCORE (2510.27462),
  Velocitune (2411.14318), tDRO (2408.10613), XDoGE (2512.10545) — the training-side mirror
  of "which tokens does this data move" — (source:
  docs/topics/reference/training-objective-alternatives-literature.md, SciSpace-sourced,
  unverified).
- **Beyond Log Likelihood / model-capability continuum** (2510.00526) — the right objective
  depends on where on the size ladder you are; flagged as the most decision-relevant entry
  in that topic and a DataDecide-shaped claim — (source:
  docs/topics/reference/training-objective-alternatives-literature.md, unverified).

**MoE variant (FUNC-opt-4)**

- **Mixture of Parrots** (Jelassi et al., ICLR 2025; no ID in the topic) — experts buy
  memorization, not reasoning; if experts are storage, what a type-X treatment can move is
  bounded — (source: docs/topics/reference/moe-literature.md).
- **OLMoE-style expert-specialization traces** (no ID) — 38M–1.7B active, 64 experts/layer,
  top-8; specialization emerges early; top-k overlap between step t and final as a routing
  stability statistic — the categorical version of FUNC's plasticity boundary — (source:
  docs/topics/reference/moe-literature.md).
- **Router-phase study ("balance-prioritizing → stabilization")** (no ID) — training phases
  in routing; a stage structure to align FUNC's stages against — (source:
  docs/topics/reference/moe-literature.md).
- **Expert-overlap / expert-usage-similarity work** (no ID) — whether representation
  similarity explains expert-usage similarity — (source:
  docs/topics/reference/moe-literature.md).
- **Expert resetting / addition for domain adaptation** (no ID) — the practical lever the
  modular-plasticity question sits under; recorded with no citation — (source:
  docs/potential-projs/functional-featurization.md §4, 2026-08-21 MoE entry).
- **Krajewski et al. granularity axis; DeepSeekMoE shared+routed; Expert-Choice routing;
  Soft MoE; Mixture of Attention Heads (2210.05144)** — the MoE design space FUNC-opt-4
  would instantiate in — (source: docs/topics/reference/moe-literature.md).

**Non-stationarity framing (the program's mechanism pillar)**

- **Exogenous / endogenous non-stationarity split** (no ID) — LR schedule, data-order and
  realized-composition drift, and midtraining as "institutionalized distribution shift that
  the field adopted empirically without a theory of when shift helps"; endogenously, even
  under iid data the effective distribution is data weighted by current gradient magnitude,
  so "every model runs an implicit self-curriculum." This is the theoretical statement of
  why FUNC's stage axis should be non-trivial — (source:
  docs/topics/reference/nonstationarity-accounting.md).
- **The generic-shift-transient confound** (no ID) — any distribution shift produces a loss
  spike / wall excursion that dominates short branches regardless of content; the recorded
  reason for the local ε design over pure-type far-field branches — (source:
  docs/potential-projs/functional-featurization.md §4).

**Featurization families FUNC is positioned against as a proposed fourth**

- **The three-family toolkit** — model-mediated (perplexity correlations, RegMix, DoReMi,
  Data Mixing Laws, BiMix), similarity embeddings (Task2Vec, alignment coefficients,
  Miranda et al.'s diversity coefficient, with *Data Similarity is Not Enough* as the
  negative result), and intrinsic statistics (WIMBD, compression/entropy, Zipf/burstiness/
  type-token) — FUNC-opt-5 regresses these onto response profiles with n = chunks —
  (source: docs/topics/reference/data-featurization-literature.md).
- **Chan et al. 2022** (no ID) — data distributional properties drive emergent ICL; the one
  causal tie from intrinsic statistics to a capability, and the precedent for typing data
  by what it does — (source: docs/topics/reference/icl-literature.md).
- **Determinism profile / reference-model entropy as a functional decomposition at token
  granularity** (no ID) — REC-2's feature, reused as a FUNC-1 chunk feature — (source:
  docs/topics/reference/data-featurization-literature.md;
  docs/potential-projs/recipe-featurization.md).
- **"Effective tokens = diversity × syntheticity"** (2410.03083) — a teacher-measured
  corpus property; the ledger records one
  `small-scale-evaluation-metrics-literature` row feeding FUNC — (source:
  docs/topics/reference/small-scale-evaluation-metrics-literature.md;
  docs/litreview/citation-verification-ledger.md, unverified).
- **Xie et al. 2023 FinPythia-6.9B DACP** (no ID; the ledger's one non-midtraining
  `Feeds: FUNC` row, and it is flagged as a *swapped bibliography entry* — the v2 bib
  points at PIXIU) — domain-adaptive continued pretraining — (source:
  docs/litreview/citation-verification-ledger.md;
  docs/topics/reference/small-scale-evaluation-metrics-literature.md).

**Repository-internal records that behave like prior art or constraints**

- **The recorded narrow framing** — "stage-conditioned data influence has precedent in
  online/curriculum selection and the critical-period work; the unclaimed part is
  specifically recipe × stage × response-profile on an open suite" — (source:
  docs/potential-projs/functional-featurization.md §4, 2026-08-21 pushback entry).
- **The curriculum-learning reviewer prior** — "a long history of weak-at-scale results";
  the recorded defense is that FUNC measures stage-dependent data value rather than
  proposing a curriculum — (source:
  docs/potential-projs/functional-featurization.md §2, §4).
- **The realized-per-window-mixture gate** — if DataDecide's realized composition drifts
  from nominal, every run has an implicit curriculum confounding stage-dependent claims;
  named in the open-questions log as gating "every timing/curriculum claim built on
  DataDecide (including the stage-dependent data-value ideas in
  `functional-featurization.md`)" — (source: docs/open-questions-answered.md;
  docs/potential-projs/functional-featurization.md preamble).
- **The shared held-out-token-set spec** — an identical spec appears in Annealed readouts,
  WSD retrain suite, Token-level movement, MoE movement, MoE recipe suite, and FUNC; the
  irreversible logging decision is saving endpoint weights plus per-token losses —
  (source: docs/potential-projs/functional-featurization.md §3, §4 pushback).
- **WSD-opt-4 ("MiniCPM-style mixed-in decay data — scope creep risk")** — the deferred
  item the FUNC framing reclassifies as the point rather than creep — (source:
  docs/potential-projs/functional-featurization.md §4;
  docs/potential-projs/wsd-suite.md).
- **Checkpoint-tomography staging doc** — names the U_c(t) probe and the diagnostic-panel
  statistics as shared machinery — (source: docs/topics/staging/checkpoint-tomography.md).
- **DataDecide-dense staging doc** — the many-seed tiny-scale substrate the record says
  fully powered stage × type versions belong on — (source:
  docs/topics/staging/datadecide-dense.md).
- **The IRT duality** — "there you featurize models by their response to items; here you
  featurize data by models' response to it. Same bilinear skeleton"; Δθ from an IRT fit is
  a FUNC-4 readout — (source: docs/potential-projs/functional-featurization.md §4;
  docs/topics/reference/irt-literature.md).
- **Ranked-list attribution** — cut from the workshop-sized list as "second-act by
  construction"; full-conference #10, "The Functional Types of Pretraining Data," with the
  highest ceiling on that list and only a modest pivot (the surrogate-validation study
  stands alone as a methods paper) — (source: docs/portfolio-rankings.md;
  docs/potential-projs/functional-featurization.md §4).
- **Citation-verification ledger, `Feeds: FUNC` rows** — all from
  `targeted-pretraining-midtraining-literature` (2512.07783, 2306.12070 agent-supplied;
  2506.20512 Claude-added) plus the flagged FinPythia row; the ledger states nothing there
  is verified — (source: docs/litreview/citation-verification-ledger.md).

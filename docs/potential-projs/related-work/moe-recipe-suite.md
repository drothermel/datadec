# moe recipe suite — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`moe-recipe-suite.md`](../moe-recipe-suite.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*Purpose: the high-recall corpus for MSUITE (small MoEs on 4–6 DataDecide recipes at fixed
architecture, routing as the readout). Every item is on record somewhere in this
repository; nothing is verified and nothing here is a positioning claim. Err toward
inclusion. The curated core lives in `../moe-recipe-suite.md` §5.*

**Repeated-data and multi-epoch training (the training-config-parity skeleton)**

- **Xue et al., To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis**
  (2305.13230) — multi-epoch training of dense *and* MoE LMs; dropout switched on late is the
  regularizer that works; MoE models overfit repeated data *more* than dense ones. Directly
  gates the MSUITE spec since 4–6 recipes at small scale implies multi-epoch exposure —
  (source: docs/topics/reference/regularization-literature.md intake note;
  docs/potential-projs/moe-recipe-suite.md §4 2026-08-22; Claude-added ledger row 131,
  unverified).
- **Muennighoff et al., data-constrained scaling** (2305.16264) — repetition up to ~4 epochs
  nearly free, then diminishing; sets the epoch budget MSUITE's matched-epoch reading
  depends on — (source: docs/topics/reference/regularization-literature.md; ledger row 132,
  Claude-added, unverified).
- **Hernandez et al., repeated-data double descent** (2205.10487) — disproportionately damages
  induction heads; the mechanism reason repetition could change routing specialization rather
  than just loss — (source: docs/topics/reference/regularization-literature.md; ledger row 129).
- **Deduplication line the SciSpace report defaulted to** — exact dedup (Lee et al., ACL
  2022; Kandpal et al., ICML 2022), SoftDedup commonness reweighting (2407.06654),
  ClusterClip balanced sampling (2402.14526), semantic dedup, entropy filtering (SoK 2025),
  Carlini et al. 2021 extraction. The intake note says this section answers a different
  question (remove the repeats) and does not apply to MSUITE — recorded so it is not
  re-searched — (source: docs/topics/reference/regularization-literature.md; ledger rows
  135–136; agent-supplied, unverified).
- **Synthetic / rephrased data scaling (BeyondWeb; quality parameter Q; WRAP; Tale of
  Tails)** (no IDs on record) — the sibling accumulator where repeated-data scaling also
  sits, flagged as a neighbour of the regularization file — (source:
  docs/topics/reference/synthetic-data-literature.md;
  docs/topics/reference/regularization-literature.md header).

**MoE-specific regularizers and stability recipes (SciSpace-agent record; unverified, and
the intake notes flag several off-target citations)**

- **Load-balancing auxiliary loss** (Shazeer et al. 2017; Switch, Fedus et al. 2022) — the
  baseline stabilizer that must be recorded in the frozen spec — (source:
  docs/topics/reference/regularization-literature.md).
- **ST-MoE router z-loss (Zoph et al. 2022)** — plus ST-MoE's finding that sparse models
  overfit in fine-tuning, listed as missing canon — (source:
  docs/topics/reference/regularization-literature.md).
- **Switch Transformer expert dropout** (no ID on record) — higher dropout inside experts than
  in dense layers; listed as missing on the MoE side and named in the §4 entry — (source:
  docs/topics/reference/regularization-literature.md intake notes).
- **MoEC, cluster-level expert dropout** (2207.09094) — expert-level dropout variant —
  (source: docs/topics/reference/regularization-literature.md; ledger row 133,
  agent-supplied).
- **Elbayad et al. (Findings ACL 2023)** (no ID on record) — gating dropout, conditional
  routing, and curriculum to fix MoE overfitting on low-resource languages in multilingual MT
  — (source: docs/topics/reference/regularization-literature.md).
- **Gating Dropout** (2205.14336) — listed as missing MoE canon; a routing-side stochastic
  regularizer — (source: docs/topics/reference/regularization-literature.md; ledger row 130).
- **Dirichlet-prior shaping of router outputs for upcycled MoEs** (2510.01185) — a prior on
  the routing distribution — (source: docs/topics/reference/regularization-literature.md;
  ledger row 137).
- **StableMoE; DeepSeek's auxiliary-loss-free balancing; OLMoE's stability recipe** (no IDs
  on record) — the rest of the missing-canon list on the MoE stability side — (source:
  docs/topics/reference/regularization-literature.md intake notes).
- **Intra-/cross-layer expert-specialization regularizers (Hu et al., unpublished)** — named
  in the report; flagged as an unpublished manuscript — (source:
  docs/topics/reference/regularization-literature.md).
- **General regularizers the matched dense/MoE recipe must fix:** dropout (Srivastava et al.
  2014); weight decay / AdamW (Loshchilov & Hutter 2019); L1; batch norm (Ioffe & Szegedy
  2015); early stopping; data augmentation (Hernández-García & König 2018; mixup); gradient
  clipping; **flooding** (Ishida et al., ICML 2020); weight normalization (Salimans & Kingma
  2016 — explicitly requested by Danielle, present only in the companion URL list); spectral
  norm; R-Drop; stochastic depth; DropBlock; label smoothing (Müller et al. 2019) — (source:
  docs/topics/reference/regularization-literature.md).
- **Transformer-specific:** attention dropout; LayerDrop / structured dropout (Fan, Grave &
  Joulin, ICLR 2020); UniDrop (NAACL 2021); layer norm + label smoothing (cited to Liu et al.
  2020); relaxed attention (2209.09735, ledger row 134) — (source:
  docs/topics/reference/regularization-literature.md).

**The readout precedents MSUITE-4 reports against**

- **OLMoE — Open Mixture-of-Experts Language Models** (no ID on record) — router saturation
  (top-k overlap at step t vs. convergence; deeper layers saturate faster) as the field's
  existing commitment metric, and an external validation point with open intermediate
  checkpoints — (source: docs/topics/reference/moe-literature.md).
- **Three Phases of Expert Routing: How Load Balance Evolves During MoE Training** (no ID on
  record) — surge / stabilization / relaxation, non-monotone, said to be stable during
  fine-tuning; the trajectory shape MSUITE's per-recipe commitment curves would be compared
  to — (source: docs/topics/reference/moe-literature.md).
- **OpenMoE analysis** (no ID on record) — routing dominated by token ID with minimal context
  relevance, assignments fixed early; MSUITE-4 tests whether this holds across recipes —
  (source: docs/topics/reference/moe-literature.md; docs/portfolio-rankings.md Tier 1 #2).
- **The Myth of Expert Specialization in MoEs** (no ID on record) — cross-model specialization
  overlap no higher than chance; routing reflects representation geometry; load-balancing
  loss suppresses shared hidden directions, "explaining specialization collapse under less
  diverse data" — the last clause is a direct prediction about recipe diversity for MSUITE —
  (source: docs/topics/reference/moe-literature.md).
- **Continual Pre-training of MoEs: How Robust Is Your Router?** (no ID on record) — routing
  changes most in early layers under shift; relevant if MSUITE adds decay branches
  (MSUITE-opt-1) — (source: docs/topics/reference/moe-literature.md).
- **Jelassi et al., Mixture of Parrots** (ICLR 2025; no arXiv ID on record) — if experts are
  storage, recipe differences should show up as *what* gets stored; the record names this as
  directly relevant to "does the data choose the experts" — (source:
  docs/topics/reference/moe-literature.md 2026-08-22, unverified beyond agent summary).

**Substrate, external validation points, and the artifact gap**

- **FLAME-MoE: A Transparent End-to-End Research Platform for MoE Language Models** (no ID on
  record) — 38M–1.7B active, 64 experts, top-8, full openness (code, data, checkpoints,
  routing logs, evals); originally the config template ("start from FLAME-MoE's validated
  configs rather than tuning fresh"), later superseded by the sweep's own defaults; now an
  external validation point — (source: docs/topics/reference/moe-literature.md;
  docs/potential-projs/moe-recipe-suite.md §4 2026-08-18).
- **FLAME-MoE routing-log gate** (no ID) — which checkpoints, how many tokens, whether token
  identities are recoverable; still open — (source: docs/open-questions-answered.md).
- **Slicing-and-Dicing MoE sweep** (arXiv 2605.11689; Danielle third author) — ~2,000 runs;
  total parameters always help even at 128× ratios, optimal expert size depends only on
  active parameters, other knobs second-order; supplies MSUITE-2's fixed architecture (fix
  expert size by active params, dropless routing, ignore second-order knobs) and removes the
  "standing up MoE infra" risk — (source: docs/potential-projs/moe-recipe-suite.md §4
  2026-08-21; docs/potential-projs/moe-partitions.md §4; docs/danielle-inputs.md).
- **Slicing-and-Dicing checkpoint availability** (no ID) — final checkpoints exist; no
  intermediates from the original sweep — (source: docs/open-questions-answered.md 2026-08-21).
- **DataDecide: How to Predict Best Pretraining Data with Small Experiments** (no ID on
  record) — the 25-recipe treatment axis MSUITE mirrors on the MoE side, the token
  budget/tokenizer to match, and the free dense control ladder for MSUITE-opt-3 — (source:
  docs/topics/reference/moe-literature.md; docs/portfolio-rankings.md).
- **Signal and Noise: A Framework for Reducing Uncertainty in LM Evaluation** (no ID on
  record) — noise worsens as scale shrinks, and routing discreteness plausibly adds eval
  variance, so the noise-floor stage is "more necessary" here — (source:
  docs/topics/reference/moe-literature.md 2026-08-18).
- **The artifact-gap claim** (no IDs) — FLAME-MoE is a scale ladder on one recipe; OLMoE one
  recipe (Dolma + DataComp-Baseline); OpenMoE one recipe; the 2025–26 open-weights wave
  (Llama 4, DeepSeek V4, Qwen 3.6, Kimi K2.6, gpt-oss, Command A+) is closed-data — hence "no
  public multi-recipe MoE suite exists," the gap MSUITE fills — (source:
  docs/topics/reference/moe-literature.md; docs/potential-projs/moe-partitions.md §4;
  docs/potential-projs/trajectory-statistics.md §4; unverified).
- **Folklore-tuned MoE knobs caution** (no ID) — aux-loss coefficients, top-k, expert count,
  capacity factors tuned at large scale, possibly mis-set for 20–50M active — (source:
  docs/topics/reference/moe-literature.md).
- **Held-out token set spec** (no ID; internal) — frozen, versioned, domain- and
  entropy-stratified; shared verbatim with Annealed readouts, WSD retrain suite, Token-level
  movement, MoE movement, and Functional featurization — (source:
  docs/potential-projs/moe-recipe-suite.md §3 step 4).
- **DataDecide-dense staging substrate** (no ID; internal) — the sibling that carries the same
  regularization-recipe decision (dropout onset, weight decay, expert dropout / z-loss,
  epochs recorded in the frozen spec) and the same training-config parity check as step 1 —
  (source: docs/topics/staging/datadecide-dense.md).

**Architecture vocabulary for the fixed-architecture choice (agent-generated design-space
record; identifiers agent-supplied/Claude-added and unverified per the ledger; author pairs
flagged fabrication-prone)**

- **Scaling Laws for Fine-Grained MoE** (2402.07871) — the granularity axis, named in the
  intake note as "already the MSUITE design knob" — (source:
  docs/topics/reference/moe-literature.md).
- **DeepSeekMoE** (2401.06066) — shared + routed experts, the hybrid the fixed config chooses
  for or against — (source: docs/topics/reference/moe-literature.md).
- **Sparsely-Gated MoE** (1701.06538); **Switch / ST-MoE** (no IDs on record); **GLaM**
  (no ID) — the sparse-FFN-MoE lineage the config sits in — (source:
  docs/topics/reference/moe-literature.md).
- **From Sparse to Soft MoE** (2308.00951); **Mixtral** (2401.04088);
  **Mixture-of-Depths** (2404.02258); **SwitchHead** (2312.07987); **Mixture of Attention
  Heads** (2210.05144); **Expert-Choice routing** (named, not listed) — alternative selection
  types and routing units — (source: docs/topics/reference/moe-literature.md).
- **Branch-Train-Merge** (2208.03306) / **Branch-Train-MiX** (2403.07816, Claude-added) /
  **RouteLLM** (2406.18665) — routing learned post-hoc over frozen experts; the record's noted
  fourth axis (*when* routing is learned), which a data-as-treatment suite holds fixed —
  (source: docs/topics/reference/moe-literature.md intake notes).
- **Model soups** (2203.05482), **Weight-Ensembling MoE** (2402.00433), **BatchEnsemble**
  (2002.06715), **MIMONets** (2312.02829), **MatFormer** (2310.07707), **AdapterFusion**
  (2005.00247), **Mixture of LoRA Experts** (2404.13628), **MixLoRA** (2404.15159), **Higher
  Layers Need More LoRA Experts** (2402.08562), **"hydra"/shared-trunk** (2209.14375,
  identity unchecked) — the rest of the placement table — (source:
  docs/topics/reference/moe-literature.md; docs/litreview/citation-verification-ledger.md
  rows 331–351).
- **MoEUT** (2405.16039), **Sparse Universal Transformer** (2310.07096) — MoE inside a shared
  looped block; an architecture family adjacent to the MSUITE config space — (source:
  docs/topics/reference/layer-looping-literature.md; SciSpace-agent record, unverified).

**Mechanism framing and the branch/plasticity extensions (MSUITE-opt-1)**

- **The non-stationarity accounting frame** (no ID; canonical program text) — routing as the
  clearest endogenous non-stationarity; per-expert input drift as a logged scalar; the pillar
  MSUITE's routing readouts feed — (source: docs/topics/reference/nonstationarity-accounting.md).
- **Wen et al., Understanding WSD LRs: A River Valley Loss Landscape View** (arXiv
  2410.05192) — the decay-branch machinery MSUITE-opt-1 borrows, and the determinism-profile
  claim (deterministic tokens = river, uncertain = walls) that MSUITE-4 tests routing
  commitment against — (source: docs/topics/reference/landscape-literature.md;
  docs/topics/reference/token-level-literature.md).
- **Token-Level Uncertainty-Aware Objective for LM Post-Training** (no ID on record) —
  epistemic/aleatoric split; the reference-model entropy scorer that defines MSUITE's
  entropy buckets — (source: docs/topics/reference/token-level-literature.md).
- **Rho-1, Not All Tokens Are What You Need for Pretraining** (no ID on record) — the
  loss-trajectory token taxonomy the routing taxonomy would be compared with — (source:
  docs/topics/reference/token-level-literature.md).
- **Task2Vec (Achille et al.); WIMBD; perplexity correlations, RegMix, DoReMi, mixing laws;
  diversity coefficient** (no IDs on record) — the corpus-feature families the routing
  partition is joined to in MSUITE-4's alignment readout — (source:
  docs/topics/reference/data-featurization-literature.md; docs/topics/reference/critical-periods.md).
- **Plasticity line — Dohare et al. (Nature 2024 / 2306.13812); Lyle et al. (2303.01486,
  2402.18762); Hernandez-Garcia et al. (2606.24752)** — the "modular plasticity" reading of
  the MoE projects and the claim that plasticity loss appears in stationary settings too;
  relevant to reading per-expert continual learning across recipes — (source:
  docs/topics/reference/plasticity.md).
- **Achille, Rovere & Soatto, Critical Learning Periods** (ICLR 2019; no ID on record) — the
  early-window framing under which recipe differences would be locked into the partition —
  (source: docs/topics/reference/critical-periods.md).
- **µP-family parametrization / HP-transfer caution** (no ID; u-µP 2407.17465 recorded) —
  the record flags that recipe suites including MSUITE assume small-scale-tuned HPs transfer,
  and that DataDecide is not µP-parametrized (cross-size LR confound) — (source:
  docs/topics/reference/parametrization-and-hp-transfer.md).

**Sibling-project and program records**

- **WSD retrain suite §4** (no ID; internal) — records this suite as its MoE counterpart, with
  the "standing up MoE training infra" risk already removed — (source:
  docs/potential-projs/wsd-suite.md §4 2026-08-21).
- **Trajectory statistics / token-movement routing follow-up (TRJ-moe-1/3, TOK-obs-5)** (no
  IDs; internal) — the FLAME-MoE routing follow-up this suite is said to rescue from orphan
  status, and the per-token flip metrics MSUITE-4 reuses — (source:
  docs/potential-projs/trajectory-statistics.md; docs/potential-projs/token-movement.md).
- **Tiny-MoE floor (MSUITE-opt-2) rationale** (no IDs) — per-expert data starvation
  (budget/E tokens), a router too small to learn a partition, worsening routing shallowness;
  deliberately cut from the workshop-sized list because fair dense-baseline tuning at 5M
  active is "a tarpit" and the outcome is fragile — (source:
  docs/potential-projs/moe-recipe-suite.md §4; docs/portfolio-rankings.md).
- **Portfolio placement** (no ID) — 6–12-month flagship list Tier 1 #2, "Does the data choose
  the experts?"; rated slightly below #1 on probability-of-strong-outcome but comparable on
  ceiling; the entry predates the Slicing-and-Dicing discussion so it assumes a FLAME-MoE-
  style config — (source: docs/portfolio-rankings.md; docs/potential-projs/moe-recipe-suite.md §4).

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

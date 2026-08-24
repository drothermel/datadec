# LR schedules, annealing branches, and checkpoint merging — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: annealed readouts (ANN) and the WSD suite (WSD) are built on the
stable-phase-plus-decay-branch methodology and on checkpoint merging as a decay proxy; the
same machinery is the branch runner for token-level movement and functional featurization.

---

## Working list (maintained; last revised 2026-08-22) — nothing below is verified

Keepers, with the role each plays for `ANN` / `WSD` / `FUNC`:

- **Llama 3** "Annealing Data" (8B GSM8K +24 / MATH +6.4; 405B negligible; 30/70 final-40B
  anneal as data valuation) — the canonical late-HQ-data result and its scale attenuation.
- **Databricks "Does your data spark joy?"** (7B end-of-training domain upsampling; 10–20%
  of training is the trade-off point) — the budget heuristic.
- **OLMo 2** (2501.00656; Dolmino mid-training mix, LR to zero) — the open reproducible
  template.
- **TREC** (Bergsma et al. 2509.25380) — place the anneal at the receptivity valley, not
  automatically at the end; predictable from AdamW's EMA timescale.
- **Tissue et al. 2408.11029** (annealing-area term in the loss law) and **Hägele et al.
  2405.18392** (WSD/cooldown scaling laws; (1-sqrt) cooldown; branch reuse) — the
  LR-side theory and the cost model for decay branches.
- **MiniCPM 2404.06395** — WSD origin; decay-phase gradient statistics (norm falls,
  consecutive-update cosine positive) as a candidate ANN instrument.
- **PDPC 2501.13126**, **AutoScale 2407.20177**, **Data Mixing Laws 2403.16952**,
  **UtiliMax/MEDU 2501.11747** — what to anneal on and when; scale-dependent composition.
- **Rewriting cluster** — SwallowCode/Math 2505.02881, ProX 2409.17115, FinerWeb
  2501.07314, Nemotron-CC 2412.02595 — anneal-grade data by upgrading rather than
  selecting; staged as `../staging/rewritten-anneal-slice.md`.
- **FineWeb-Edu 2406.17557**, **Phi-4 2412.08905** — classifier-filtered and synthetic
  anneal data at scale; details in the second entry.

Known drift and term collisions (do not re-flag): instruction-tuning / continual-learning
papers (VersaTune, O-LoRA, SSR, KILO, LIFT, MoS); Annealed-RLVR 2509.23629 (RL "heating");
RLHFuse (simulated annealing for scheduling); "data annealing" 2004.13833 (2020,
formal→informal BERT); the "web 50→20%, synthetic 5→35%" proportions (illustrative, no
source); "edu filtering works best during annealing" (respondent inference, untested).
Open ID check: 2508.01483 (paired with Tissue; may be a mis-ID).

---

## 2026-08-18 — the annealing confound and its workarounds (from the Research Trajectory page)

**The confound.** "DataDecide's models were trained OLMo-style with cosine schedules, so every
intermediate checkpoint sits mid-schedule with high residual LR — in river-valley terms,
evals on those checkpoints measure 'position along river + current distance up the wall,'
and the wall component is schedule-dependent noise relative to the question you care about.
Post-training from such checkpoints inherits the confound (you're fine-tuning from a point
high on the wall)."

**Stable phase + decay branches**
- Hägele et al. 2024, *Scaling Laws and Compute-Optimal Training Beyond Fixed Training
  Durations* — "made essentially your argument as a methodology proposal: constant LR +
  short cooldown matches cosine, so scaling-law and data experiments should use
  stable-phase runs with cheap decay branches instead of retraining full cosine runs per
  budget."
- MiniCPM (*Unveiling the Potential of Small Language Models with Scalable Training
  Strategies*) — "established the practical template — a stable-phase checkpoint plus a
  fixed-length decay matches full cosine baselines, with ~10% decay completing convergence
  and new data mixed in strictly during decay." "The stable-checkpoint-plus-1-sqrt-decay-
  resume protocol is now standard in careful small-scale studies."
- Llama 3 (*The Llama 3 Herd of Models*) — annealing-based data-quality assessments;
  Blakeney et al., *Does your data spark joy? Performance gains from domain upsampling at
  the end of training*. "So 'annealing branches as the correct eval' is validated practice
  — but no *open, multi-recipe suite* has it."
- OLMo (*OLMo: Accelerating the Science of Language Models*) — the training setup
  DataDecide inherits.

**Checkpoint merging as pseudo-annealing**
- WSM (*WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training*)
  — "merging recent checkpoints with weights derived from an emulated decay curve provides a
  robust annealed model without ever altering the live learning rate… WSM-merged models
  consistently and closely mirror the results of a true LR anneal at intermediate stages of
  long runs."
- Nemotron 3 (*Nemotron 3 Super…*) — "applies sliding-window checkpoint merging to get
  stronger quality readouts without dedicated decay runs, estimating savings of ~16% of
  total pretraining FLOPs."
- Open question: "whether merging-as-annealing-proxy works on *cosine* mid-run checkpoints
  (varying LR within the merge window) rather than stable-phase ones."

**Analytic correction**
- The multi-power law (Luo et al., arXiv 2503.12811) "predicts the loss drop from a
  hypothetical decay given the trajectory so far, so you could correct unannealed loss
  curves analytically. It won't give you downstream metrics, but it quantifies how much
  each recipe's apparent ranking is schedule artifact."

**Caveats**
- "DataDecide found intermediate-checkpoint *decisions* matched compute-equivalent final
  checkpoints — so the confound may partially cancel for rankings even while distorting
  levels and post-training. Testing when it cancels vs. doesn't is exactly what your WSD
  branches would settle."
- "The river-valley theory predicts the decay phase itself makes progress along the river —
  so annealed evals aren't a pure 'reveal' either; branch length becomes a parameter to
  control."

## 2025-10 (undated conversation; intake 2026-08-22) — Annealing-data literature survey

**Danielle's question.** Understand recent research on data quality as it affects LLM
annealing — especially changing data from pretraining to the annealing stage: "how the order
and stage of introducing data impacts how well the model fits it, with an understanding that
different labs use their highest quality data during the annealing phase because this
somehow improves the overall results." Asked for an extensive pass over the last ~year of
arXiv work, precise representation of paper contents, and a forward look.

**Response (browsing survey scoped "current as of Oct 7, 2025"; near-verbatim, condensed; all
figures and attributions unverified here).** "Annealing" = late-stage, low-LR training on a
higher-quality or targeted mixture (a.k.a. mid-training, domain upsampling).

*Evidence that late high-quality data moves the needle.*
- Llama 3 "Annealing Data": upsample small amounts of HQ code/math near the end; 8B:
  GSM8K +24.0%, MATH +6.4%; 405B: negligible gains (diminishing returns at scale); benchmark
  train sets excluded; annealing used to value datasets — final 40B tokens, 30% new / 70%
  default, LR linearly to 0.
- Databricks "Does your data spark joy?": 7B / 1T tokens; end-of-training domain
  upsampling (swap part of CC for targeted HQ domains) → MMLU +6.90, GSM8K +8.26, HumanEval
  +6.17 pp; Llama-2-7B-like at ~half the FLOPs; 10–20% of training devoted to upsampling is
  the best general-vs-targeted trade-off.
- OLMo 2 (arXiv 2501.00656): mid-training "Dolmino mix" targeting weak spots (math), LR
  decayed to zero across the phase; annealing as a data-evaluation tool (30/70).
- Takeaway: late small HQ slices give material gains at small/medium scale when LR is
  already low and the mixture doesn't over-shift; attenuates at very large scale.

*When to insert HQ data — not necessarily the end.*
- **TREC** (Training Re-evaluation Curves; Bergsma et al. 2025, arXiv 2509.25380):
  re-evaluate each training batch with the final weights; the curve dips in a "valley"
  before the end (esp. step-drop LR); identical HQ amounts placed in different 10% segments
  do best near the TREC minimum; TRECs are predictable in advance from AdamW's implicit EMA
  timescale; claims to explain why Llama-3-405B didn't benefit from GSM8K annealing.
- LR-annealing scaling law (Tissue et al. 2024, arXiv 2408.11029): adds a term linear in
  "annealing area" to the loss-vs-compute law — LR trajectory affects realized loss beyond
  total tokens.
- Continued-pretraining recipe (Parmar et al. 2024): two-stage CPT — general blend
  up-weighted for HQ sources, then switch once LR ≈ η_max/5 to QA/targeted data; cosine
  decay from the original η_min, no warmup; stay distribution-adjacent.

*What to anneal on.*
- AutoScale (arXiv 2407.20177): optimal domain mix changes with scale — HQ sources
  (Wikipedia, papers) dominate small budgets then saturate; diverse CC keeps paying at large
  budgets; fit a surrogate at small budgets and extrapolate.
- Data Mixing Laws (Ye et al., arXiv 2403.16952): validation loss vs. domain proportions via
  an exponential form nested with scaling laws; predict unseen mixtures; avoid forgetting in
  continual training.
- UtiliMax / MEDU (arXiv 2501.11747): size-aware heuristics are strong baselines;
  portfolio optimization over ablation-estimated (UtiliMax) or LLM-estimated (MEDU) utility;
  claimed up to 200× compute savings vs. brute ablations.
- PDPC (arXiv 2501.13126): Perplexity Difference between a weak and a strong model ranks
  samples by when they should be learned; high-PD deferred; offline corpus arrangement;
  +8.1% avg MMLU/CMMLU on a 3B / 1T run.

*Curriculum and order beyond mixtures.* Large-scale CL study (Zhang et al. 2025, arXiv
2506.11300): 0.5–1B models; easy→hard warmup by compression ratio / lexical diversity /
readability improves early/mid convergence with lasting gains up to +3.5%, ordering
disentangled from selection. Influence-driven curricula (arXiv 2508.15475): rank by
gradient-similarity influence; >10 pp over random in low-resource pretraining.

*Emerging principles (quoted list).* (1) late low-LR phases are the time for
high-information slices if not saturated — +6–8 pp at 7B, may vanish at 400B; (2) place the
anneal where the model is most receptive (TREC), not automatically at the end; (3) keep a
stabilizing anchor (~70% base mix), decay LR to zero across the anneal; (4) budget ~10–20%
of tokens; (5) composition is scale-dependent; (6) don't contaminate evals.

*Mental model.* Optimization lens — LR annealing makes AdamW a long-horizon EMA over
updates, so data near the TREC valley has disproportionate influence on final parameters;
data lens — "high-quality" is scale- and stage-dependent; scheduling lens — heuristics work,
predictive methods (TREC / PDPC / mixing laws) do better.

*Forward look.* Optimizer-aware curricula (predict-then-place); scale-adaptive, possibly
continuous annealing rather than one late block; model-centric difficulty metrics (PD,
influence, loss attribution); unified data × LR scaling laws; contamination and provenance
hygiene.

**Relevance here.** Directly informs the annealed-readouts (`ANN`) confound story (what the
DataDecide cosine tail is doing to rankings), the WSD suite's decay-branch data choice, and
functional featurization (`FUNC`) — TREC's "receptivity valley" is a candidate explanation
for stage-dependent chunk effects. Verify TREC (2509.25380) and PDPC before building on
them.

## Undated (~2025; intake 2026-08-22) — Annealing-data report, second answer to the same question

**Danielle's prompt (verbatim).** "I want to better understand the recent research around
data quality as it affects LLM annealing, especially changing data from pre-training to the
annealing stage. Please find one or two recent papers and visualize their results in an
interactive educational app/document to guide me through understanding this content. I'm
a phd student so target your level of depth to that."

**Response.** A long browsing report ("based on analysis of key papers from 2024–2025,
including Llama 3, MiniCPM, FineWeb, and Phi-4") with inline links; the interactive app it
refers to at the end was not passed. Substantial overlap with the Oct-2025 survey entry
above (Llama 3 annealing numbers, 10–20% budget, scale attenuation); only what is new or
different is kept here. **All figures and attributions are the respondent's and
unverified**; the report also carries a large amount of unsourced editorial (e.g. "20–40%
improvements", "50–70% less compute", "3–5× infrastructure returns", "10–20 experimental
runs", "2–3% from (1-sqrt) over linear") that is dropped rather than re-checked.

*Papers it leans on (links as given).*
- **Hägele et al. 2024, arXiv 2405.18392** ("scaling laws and compute-optimal training
  beyond fixed training durations"): WSD/constant-LR + cooldown; the (1-sqrt) cooldown
  f = 1 − √((n − (N − N_decay))/N_decay) reported as beating linear decay; cooldown as a
  way to reuse one run for many endpoints. *The one source here not already on file.*
- **MiniCPM, arXiv 2404.06395**: gradient dynamics across the decay phase — weights move
  less than in the stable phase, loss drops faster, gradient norm falls, cosine similarity
  between consecutive updates turns predominantly positive (consistent directed progress
  vs. exploratory); "loss drop in decay ≈ a 5× larger model"; decay-branch reuse for
  data–model scaling laws at linear rather than quadratic cost. Report also attributes to
  it a "curvature increases / first-order directional derivative decays with the LR,
  second-order only slightly up" description (the link given for that is Hägele).
- **Llama 3** (Meta PDF mirror): 8B GSM8K +24%, MATH +6.4%, 405B negligible; final-40B
  annealing as a data-valuation tool — same facts as the survey above.
- **FineWeb / FineWeb-Edu, arXiv 2406.17557**: educational-value classifier
  (Snowflake-arctic-embed-m + linear head, 460k Llama-3-70B-Instruct annotations, keep
  score ≥3, ~82% F1), 1.3T tokens from 15T, quoted +12% MMLU / +24% ARC; ~6,000 H100 hours
  to classify; heuristic filter ablations (duplicate-line fraction, short-line proportion).
- **Phi-4, arXiv 2412.08905**: synthetic data throughout and especially late; multi-agent
  generation, self-revision, rejection sampling, execution/proof verification; multiple
  epochs over synthetic data; decontamination incl. MinHash fuzzy + semantic; post-cutoff
  AMC-10/12 as contamination-proof evaluation. "One million math problems × eight verified
  solutions ≈ 30B tokens" is cited to a *Phi-4-mini-flash-reasoning* model card, not the
  paper.
- **Nemotron-CC, arXiv 2412.02595**: ensemble quality classifiers (Mistral-based,
  Nemotron-340B, DCLM) and synthetic rephrasing of high-quality segments.
- **YuLan-Mini, arXiv 2412.17743**: context extension 4K→32K during annealing at constant
  token batch; topic-based recall and cross-lingual synthetic generation.
- Contamination survey, arXiv 2503.17793: 1–45% contamination across benchmarks,
  inflation up to 14% C-Eval / 7% HellaSwag (report's numbers).

*Errors and drift.*
- **PDPC is misattributed** to "the YuLan-Mini team" with a Phi-4 link; PDPC is arXiv
  2501.13126 (survey entry above) and the "+8.1% MMLU/CMMLU" belongs to it. The expansion
  "Preference Data-aware Preference Curriculum" is the respondent's; PD = perplexity
  difference.
- The "data-quality gradient" proportions (web 50%→20%, synthetic 5%→35%, math 8%→20%)
  are presented as a cross-model pattern but carry a Llama 3 link; no model reports them
  in that form. Treat as illustrative, not data.
- The "annealing should begin when validation loss plateaus … monitor gradient norm and
  curvature" guideline, the tiered filtering thresholds by model size, and the
  environmental/"democratizing" sections are editorial.
- Sources include a Reddit thread and a blog mirror of the Llama 3 PDF; cite the papers.

**Relevance here.** Two new things beyond the survey entry: (1) Hägele et al. 2405.18392 is
the citation for the decay-shape choice in the WSD suite (`WSD`) and for decay-branch reuse
as the cost model behind annealed readouts (`ANN`); (2) MiniCPM's decay-phase gradient
statistics (norm falls, consecutive-update cosine turns positive) are a concrete
measurement the decay-branch runner can reproduce at DataDecide scale — a candidate ANN
instrument, noted in `../../potential-projs/annealed-readouts.md` §4. Both unverified
against the papers.

## Undated (~2025; intake 2026-08-22) — Annealing-data question, third answer (two versions)

**Danielle's prompt (verbatim).** "I want to better understand the recent research around
data quality as it affects LLM annealing, especially changing data from pre-training to the
annealing stage."

**Version 1.** A citation-dense browsing answer that drifts from annealing-data into
instruction tuning, continual learning and forgetting mitigation (VersaTune 2411.11266,
O-LoRA, Self-Synthesized Rehearsal 2403.01244, KILO 2508.03571, LIFT, Mixture-of-Skills
2406.08811, instruction-mix studies 2310.05492 / 2312.10793) — none of which is the
pretraining→annealing transition. Kept as leads, all unverified:
- A second LR-annealing scaling-law citation, **arXiv 2508.01483**, paired everywhere with
  Tissue et al. 2408.11029 ("forward area" vs. "annealing area"; annealing "momentum" —
  LR changes reflected in loss with a delay that grows with annealing slope; 10–20%
  annealing ratio). Unknown paper; check whether it is a follow-up or a mis-ID.
- **Rewriting rather than filtering**: SwallowCode / SwallowMath (2505.02881; refined
  Python +17.0 HumanEval, math +12.4 GSM8K, "transform-and-retain"); ProX "programming
  every example" (2409.17115; a 0.3B model emits per-document refinement programs);
  FinerWeb-10BT line-level filtering (2501.07314; GPT-4o-mini labels → DeBERTa; "25%
  faster to target"). Relevant to what an annealing slice *is* — upgraded data, not just
  selected data.
- Temperature sampling vs. scalarization on imbalanced mixtures (2410.04579): temperature
  sampling has lower gradient variance, converges faster, overfits more; proposed
  "cooldown" = heavy upsampling early, reduce later. Mixture-level cooldown, distinct
  from LR cooldown.
- Branch-and-Merge (2407.08699): merge models fine-tuned on data subsets; smaller but
  higher-quality weight changes, less forgetting — adjacent to ANN-opt-7's merging angle.
- Curriculum cluster (2405.07490, 2406.19853, 2411.02337, ADCL 2505.08364 "difficulty
  shift"), largely post-training; and the 2020 "data annealing" paper for informal
  language (2004.13833 / Findings EMNLP 2020) — an earlier, unrelated use of the term
  (formal→informal gradual mixing for BERT), worth knowing exists for terminology.
- Claim to watch: "educational filtering becomes particularly effective when applied
  during the annealing phase rather than throughout pre-training" is asserted with a
  FineWeb citation; FineWeb-Edu did not test that. Respondent's inference.

**Version 2.** Short and almost entirely term collisions: Annealed-RLVR (2509.23629; an SFT
"heating" phase inserted into RL with verifiable rewards — post-training, not LR/data
annealing), RLHFuse (NSDI '25; *simulated annealing* for RLHF pipeline scheduling —
unrelated), cosine-to-zero over the full duration (cited to 2408.11029), re-warm/re-decay
for continued pretraining and "stricter filtering / upsampling of core sources in the
final phase" cited to two Raschka newsletter posts. Nothing to keep beyond the collision
list.

**Intake note.** Three answers to this question are now on file (Oct-2025 survey; the
interactive-app report; this pair). The survey entry remains the usable one; this pair adds
only the rewriting cluster, 2508.01483, and the mixture-cooldown idea. Search drift into
fine-tuning is the dominant failure here, as with the SciSpace batch.

## 2026-08-24 — NotebookLM pretraining-dynamics notebook (11 papers; main routing entry)

Danielle supplied a NotebookLM notebook over eleven 2024–2025 pretraining papers
(bundle: `nblm-pretraining-dynamics-notebook.md` in the 2026-08-24 intake
bundle; **no arXiv IDs supplied anywhere** — agent-generated, unverified;
NotebookLM inaccuracy caveat). The MPL (Luo 2503.12811), river-valley picture
(Wen), and CompleteP-as-name were already on record; the rest is new. Companion
entries: HP-scaling-law cluster in `parametrization-and-hp-transfer.md`,
plasticity/overtraining pair in `plasticity.md` (same date).

Schedule-side material:

- **PTQ robustness vs training dynamics (paper 10; OLMo/SmolLM3 trajectories to
  32B).** Quantization error diverges from validation loss **abruptly when the
  LR decays** — the stable phase of WSD is PTQ-flat regardless of token count,
  so the prior data-scale-degrades-quantization claim (Kumar et al. 2024) is
  confounded by schedule. Mitigations: keep LR larger longer (WSD), LAWA weight
  averaging, and **model souping along the decay trajectory** (the soup is more
  PTQ-robust than any ingredient checkpoint). Report framing: the decay phase's
  loss-reduction term settles the model into sharper river-valley minima
  (λ_max vs 2/η, Edge of Stability), trading final loss against low-bit
  deployability. A new consequence-of-anneal axis for the WSD/anneal program:
  the anneal buys loss but costs quantization robustness.
- **MPL report detail beyond the existing record:** the S1 cumulative-LR-sum
  formulation (S1 beats token count as the loss predictor across schedules);
  Theorem-1 link of the α/β exponents to Hessian-spectrum and noise-covariance
  decay; the MPL-optimized schedule beating cosine *and* WSD, with symbolic
  regression finding a **"sqrt-cube" decay** η_t ≈ η_max·(1−τ)^1.5 as the
  efficient WSD decay shape (LAMBADA +2.17%, HellaSwag +0.62%); exponential-
  vs-power-law decay-term disagreement (Momentum Law's exponential form yields
  collapsed optimized schedules); coefficient stability degrading at high peak
  LR as an open problem.
- **CPT learning dynamics (paper 3).** A continual-pretraining scaling law
  decoupling distribution shift from LR annealing; CPT loss as a transfer
  curve; initial "loss potential" dictates downstream adaptability; replay
  ratio and peak LR mathematically targetable for the in-domain/out-of-domain
  balance.
- **Mid-Training survey (paper 6).** First taxonomy of the mid-training stage:
  data distribution (knowledge-dense curated sets beat raw scale late),
  multi-stage LR annealing (WSD), long-context extension (ABF / YaRN).
- **Learning is Forgetting (paper 5).** Information Bottleneck operationalized
  at LLM scale via soft-entropy MI estimators on OLMo2/C4/Tulu: two-phase
  expand-then-compress trajectory; proximity to the compression bound predicts
  downstream performance — a candidate trajectory statistic.
- **LoRA-LR-matters (paper 4).** Claimed advantages of LoRA variants
  (PiSSA/MiLoRA/DoRA/Init[AB]) are largely LR-tuning artifacts; optimally tuned,
  all converge; initialization sets the max Hessian eigenvalue and hence the
  usable LR — an elicitation-tuning-equity datum for the pipeline-comparison
  thread.

## 2026-08-24 — Scaling Laws for Precision: the other side of the PTQ dispute (NBLM refactoring notebook)

One outlier source in the eleventh NotebookLM notebook (bundle:
`nblm-refactoring-selfimprovement-notebook.md`): **"Scaling Laws for
Precision"** — recognizably the "Kumar et al. 2024" position that the
PTQ-robustness paper in today's pretraining notebook argued is
schedule-confounded. Both sides now on record: 465 OLMo-style pretraining
sweeps on Dolma across 3–16-bit precision; low-precision training reduces
*effective parameter count*; **quantization degradation increases with
over-training on data** (the data-scale claim the PTQ paper attributes to LR
decay instead); compute-optimal pretraining stabilizes around 7–8 bits.
Note the overtraining connection: yet another cost of over-training joining
catastrophic overtraining/plasticity (same-day plasticity entry). ID
plausibly 2411.04330 (Claude-added, inferred). Verification should read the
two papers together.

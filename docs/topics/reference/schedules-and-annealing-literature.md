# LR schedules, annealing branches, and checkpoint merging — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: annealed readouts (ANN) and the WSD suite (WSD) are built on the
stable-phase-plus-decay-branch methodology and on checkpoint merging as a decay proxy; the
same machinery is the branch runner for token-level movement and functional featurization.

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

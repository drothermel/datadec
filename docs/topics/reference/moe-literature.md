# Mixture-of-experts training dynamics — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: routing is the categorical observable behind MoE movement (MOVE), the
learned token taxonomy behind MoE partitions (PART), the readout of the MoE recipe suite
(MSUITE), and the routing follow-up in trajectory drift/diffusion; expert permutation is
the textbook non-identifiable latent for the identifiability framing.

---

## Undated (intake 2026-08-22) — The design space between dense ensembles and sparse MoE (three turns)

**Danielle's prompts.** (1) MoE gives enormous numbers of paths through one model; output
ensembling of completely different dense models combines information another way; she
assumes there is a family in between "a token goes through all of a few different dense
models" and "a token is routed to a subset of feedforward layers inside one model" — what
lies in between? (2) Place each listed family on the three axes. (3) A 10–15 paper reading
list, at least one representative per cell, 2022 onward, with title, first/last author,
arXiv PDF.

**Response 1 — three knobs, not a line.** Routing unit (what is selected), parameter
sharing (how much of the network is common), selection type (fixed / soft / sparse).
Families named in order from the ensemble end to the sparse-FFN end: independent dense
ensemble → gated whole-model mixture (classical MoE, hierarchical/deep MoE) → shared-trunk
multi-head / "hydra" → packed single-network ensembles (BatchEnsemble: shared base +
rank-one per-member factors; MIMO: subnetworks ensembled in one pass) → adapter/LoRA
mixtures (AdapterFusion; Mixture of LoRA Experts) → routing over other internal units
(Mixture of Attention Heads, SwitchHead, Mixture-of-Depths) → soft or hybrid MoE (Soft
MoE, DeepSeekMoE shared+routed experts, Expert-Choice routing) → sparse FFN-level MoE
(Sparsely-Gated MoE, Switch, GLaM). Training-side bridges listed separately:
Branch-Train-Merge, model soups, Branch-Train-MiX. Claimed takeaway: the interpolation
parameter is *routing granularity*, not "ensemble vs MoE."

**Response 2 — placement table (condensed).**

| Family | Unit | Sharing | Selection |
|---|---|---|---|
| Independent dense ensemble | whole model | none | fixed average |
| Gated ensemble / classical MoE | whole model | none–low | soft (sometimes sparse) |
| Hierarchical / deep MoE | whole model or branch | low | soft or sparse, multi-stage |
| Shared-trunk multi-branch | branch / top | medium–high | fixed or soft |
| BatchEnsemble | implicit member | very high | fixed |
| MIMO / packed subnets | implicit subnetwork | very high | fixed |
| AdapterFusion | adapters | very high | soft |
| Mixture of LoRA Experts | LoRA branches | very high | soft–sparse |
| MoA heads / SwitchHead | attention heads / projections | high | sparse or structured |
| Mixture-of-Depths | token–layer participation | very high | sparse top-k |
| Hybrid shared+routed (DeepSeekMoE) | FFN experts | medium–high | sparse + always-on path |
| Soft MoE | FFN experts | medium–high | soft |
| Sparse Transformer MoE | FFN experts | high outside experts | sparse top-k |

Summary offered: ensemble → MoE moves unit smaller, sharing larger, selection
fixed → soft → sparse; the intermediate methods each move mostly one axis (BatchEnsemble
sharing; gated ensembles selection; adapter/LoRA mixtures granularity; head/depth routing
which unit is conditional). "A cube, not a line."

**Response 3 — reading list (13, all arXiv, response-supplied, unverified).** Model soups
2203.05482 (Wortsman … Schmidt); RouteLLM 2406.18665 (Ong … Stoica); MIMONets 2312.02829
(Menet … Rahimi); MatFormer 2310.07707 (Devvrit … Jain); Weight-Ensembling MoE for
multi-task merging 2402.00433 (Tang … Tao); Higher Layers Need More LoRA Experts
2402.08562 (Gao … Subrahmanian); MixLoRA 2404.15159 (Li … Tang); SwitchHead 2312.07987
(Csordás … Schmidhuber); Mixture-of-Depths 2404.02258 (Raposo … Santoro); From Sparse to
Soft MoE 2308.00951 (Puigcerver … Houlsby); DeepSeekMoE 2401.06066 (Dai … Liang);
Mixtral 2401.04088 (Jiang … El Sayed); Scaling Laws for Fine-Grained MoE 2402.07871
(Krajewski … Jaszczur). Suggested reading order in four groups: ensemble/share/routing
intuition → partial expertization → what else can be routed + soft/sparse → modern sparse
MoE and granularity. The response's own caveat: the space is "not a literal Cartesian
grid with universally agreed cells."

**Intake notes.**

- The three-axis frame is useful and should be reused when MOVE/PART/MSUITE discuss what
  "an expert" is; the granularity axis (Krajewski et al.) is already the MSUITE design
  knob. Also note the frame omits a fourth axis that matters for this repo's observables:
  *when* the routing decision is learned (jointly from scratch vs. post-hoc over frozen
  dense models, as in BTM/BTX/RouteLLM) — the training-side bridges were listed but not
  placed, and they differ from the architectural families exactly on that axis.
- Reading-list substitutions relative to response 1: RouteLLM stands in for "gated
  whole-model mixture" (it routes *queries between deployed LLMs* by preference data —
  a different object than a jointly trained gated ensemble); MIMONets stands in for
  MIMO/BatchEnsemble (2022+ constraint; the originals are 2020); MatFormer stands in for
  shared-trunk/hydra (nested, not branched — elastic inference rather than diversity);
  WEMoE stands in for the model-merging bridge. None of the 2022+ papers directly covers
  the "gated ensemble of whole dense models" cell as originally described; Branch-Train-
  MiX (2403.07816, *Claude-added*) would be the closer fit and was named in response 1
  but dropped from the list. Cells with no representative at all: hierarchical/deep MoE
  over full experts; Mixture of Attention Heads (2210.05144, cited in response 1 only);
  Expert-Choice routing (named, not listed).
- Author pairs are as the response stated them; first/last author for large-team papers
  (Mixtral, DeepSeekMoE, MatFormer) are the kind of detail these sessions have fabricated
  before — verify before citing.
- No project ID. Relevant to MOVE/PART/MSUITE as vocabulary and to the
  "tiny LLMs as focused components of larger systems" thread (not yet filed) as the
  architectural side of composing small models.

## 2026-08-22 — Mixture of Parrots (Jelassi et al., ICLR 2025): experts buy memorization, not reasoning

**Danielle's prompt (verbatim):**

> I want you to summarize this paper's key findings.
> https://proceedings.iclr.cc/paper_files/paper/2025/file/5bc3356e0fa1753fff7e8d6628e71b22-Paper-Conference.pdf

Artifacts: `~/drotherm/data/convo-artifacts/2026/scispace-summarize-mixture-of-parrots/` — the paper's full text as markdown and the agent summary (truncated at
its start); `INDEX.md` inside.

**Claims (from the agent summary and the paper abstract in the bundle):**

- *Theory:* there exist graph problems (e.g. path connectivity) that no number of
  experts of a given width can solve, while a dense model of slightly larger width
  solves them easily; conversely, for memorization, total parameter count (not active)
  determines storage capacity, so a sparse model memorizes N sequences with fewer active
  parameters by spreading storage across experts.
- *Synthetic:* on graph problems more experts do not help but wider dense models do; on
  closed-book retrieval (memorizing associations) MoEs beat dense models at equal active
  parameters and improve with expert count.
- *Pretrained (65B tokens, FineWeb-edu / Cosmopedia / Wikipedia):* on world-knowledge
  benchmarks (TriviaQA, Natural Questions) MoEs with ~18M active parameters nearly match
  much larger dense models and scale with expert count; on GSM8K / MATH / ARC, adding
  experts gives no consistent gain while adding active parameters does.
- The title references the "stochastic parrots" critique: expert scaling makes better
  parrots, not better thinkers.

**Why it is here.** This is the concrete citation for "the 'mixture of parrots' line"
already invoked in `../../potential-projs/moe-partitions.md` §4 (the "why does total
capacity keep helping at extreme sparsity?" analysis): decompose total-parameter gains
by eval type (memorization-heavy vs. reasoning-heavy; per-token by frequency band). The
paper's own dense-vs-MoE pretraining series at fixed active parameters is the template
for that decomposition, and its theorem gives the reasoning-side null a mechanism
(width-bounded expert circuits) rather than just an empirical pattern. Also relevant to
MSUITE's "does the data choose the experts" question: if experts are storage, recipe
differences should show up as *what* gets stored, which is the routing-taxonomy reading.

**Intake notes.** Summary figures as the agent reported; the full text is on disk for
verification. The pasted summary is missing its first two sections (truncated at
source, not in transcription).

## 2026-08-18 — MoE as a new observable channel at 20–50M active (from the Research Trajectory page)

Prompt context (Danielle): MoE models do something at 20–50M active parameters, so the
program should look all the way down to that scale.

**Suites and metrics**
- FLAME-MoE (*A Transparent End-to-End Research Platform for Mixture-of-Experts Language
  Models*) — "essentially DataDecide-for-MoE at exactly your target scale — seven
  decoder-only MoE models from 38M to 1.7B active parameters, 64 experts per layer, top-8
  gating, with full openness: code, data, checkpoints, routing logs, and evaluation results
  — and their training traces already show expert specialization emerging early and
  intensifying, co-activation staying sparse and stable, and routing behaviors converging
  quickly during early pretraining."
- OLMoE (*Open Mixture-of-Experts Language Models*) — router saturation "defined as the
  average overlap between the top-k experts selected per token at step t versus at
  convergence, rising sharply within the first few thousand steps, with deeper layers
  saturating faster than shallower ones."
- *Three Phases of Expert Routing: How Load Balance Evolves During Mixture-of-Experts
  Training* — "an early balance-prioritizing phase, a stabilization phase where experts
  specialize, and a late relaxation phase trading balance for quality — a non-monotone
  trajectory invisible to post-hoc analysis of converged models, with annealing checkpoints
  confirming the phases are pretraining-specific and stable during fine-tuning."
- *Continual Pre-training of MoEs: How Robust Is Your Router?* — "routing decisions change
  most in early layers, with the no-replay condition showing the most dramatic early-layer
  routing reorganization and the most forgetting — suggesting early-layer routing changes
  may be a key mechanism of catastrophic forgetting in MoEs."
- *The Myth of Expert Specialization in MoEs: Why Routing Reflects Geometry, Not Necessarily
  Domain Expertise* — "specialization patterns in pretrained MoEs resist human
  interpretation, with expert overlap between different models answering the same question
  no higher than between entirely different questions — i.e., independently trained MoEs
  select *unrelated* specialization solutions… routers are linear maps, so hidden-state
  similarity is necessary and sufficient to explain expert-usage similarity —
  specialization is a property of the representation space, not the routing architecture —
  and load-balancing loss provably suppresses shared hidden directions, explaining
  specialization collapse under less diverse data."

**Thoughts**
- "MoE doesn't just make tiny models capable — it adds a *new observable channel* that's
  almost custom-built for your movement microscope… per-token expert-assignment flips."
- "Expert assignment is a textbook non-identifiable latent: the objective is invariant to
  expert permutation, so *which* expert specializes in what is pure trajectory-selected
  symmetry breaking." The Myth paper's cross-model overlap result is "about as clean an
  existence proof of solution-class underdetermination as the field has produced."
- The warning: "the symmetry group now includes expert permutations, which breaks your
  dense-model comparability tools — naive interpolation barriers, checkpoint merging (your
  annealing-proxy trick needs expert matching first or it averages mismatched experts into
  mush), and stitching all require an expert-alignment step, and re-basin methods for MoE
  are immature… 'how to quotient MoE symmetries for checkpoint comparison' is an open gap."
- Practical cautions: MoE knobs (aux-loss coefficients, top-k, expert count, capacity
  factors) "are all folklore-tuned at large scale and may be mis-set for 20–50M active, so
  your regime-mismatch argument applies to your *own* baseline"; noise worsens as scale
  shrinks and routing discreteness plausibly adds eval variance, so the noise-floor stage
  "isn't skippable here, it's more necessary"; keep a dense control ladder at matched
  active parameters.

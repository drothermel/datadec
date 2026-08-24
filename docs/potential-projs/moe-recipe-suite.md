# MoE recipe suite — data varied at fixed architecture, routing as the readout

**Program pillars served:** data (treatment variation on the MoE side), how (substrate). (Program: `README.md` → Program.)

> **Draft scaffolding (2026-08-21).** This doc was promoted from a topic. The quoted material in
> §4 is external text; the core steps, doability notes, impact ratings, and infrastructure
> sequence in §1–§3 are synthesized scaffolding not yet reviewed by Danielle. Treat them as
> provisional until this note is removed.

**One-line pitch.** Train small MoE models on 4–6 DataDecide recipes spanning the outcome
range, with the architecture fixed to the Slicing-and-Dicing sweep's validated defaults (fix
expert size by active params, dropless routing, second-order knobs ignored), dense
checkpoints, routing logged. Ask whether different corpora produce different expert
decompositions, whether routing-commitment timing tracks the corpus determinism profile, and
whether token-ID-dominated routing holds across recipes. The missing artifact on the MoE
side is treatment variation; this is the MoE analogue of DataDecide at pilot scale.

IDs: MSUITE-1–MSUITE-4, MSUITE-opt-1–MSUITE-opt-4.

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short
continued-training branches; **T3** = new pretraining runs.

---

## 1. What the project involves

### Core experiment (T3)

1. **Choose the subset.** 4–6 recipes spanning the outcome range (best, worst, hard to rank),
   ~40–100M active parameters, 2–3 seeds; match DataDecide's token budget and tokenizer so
   results are comparable to the dense suite.
2. **Fixed architecture.** The sweep's principled defaults; one config for every recipe.
3. **Dense checkpointing with routing logged** on a fixed held-out token set (per-token,
   per-layer top-k ids and margins) and per-token losses.
4. **Readouts (MSUITE-4).** Expert decomposition per recipe; routing-commitment timing per
   token and layer; shallow-routing controls (token ID, frequency, position); alignment
   with the corpus determinism profile and intrinsic features; FLAME-MoE / OLMoE as
   external validation points.

### Optional directions

- **MSUITE-opt-1: Decay branches off the runs.** The same branch machinery as the dense
  annealing work, giving annealed readouts and per-token decay responsiveness for MoEs.
- **MSUITE-opt-2: Tiny-MoE floor.** Where the sweep's laws break as active scale shrinks:
  per-expert data starvation (budget/E tokens), a router too small to learn a partition,
  routing shallowness worsening. "Does a 5M-active, 500M-total MoE beat a 5M dense model, and
  does its routing learn anything non-trivial?" Outcome-fragile; fair dense-baseline tuning
  is the tarpit.
- **MSUITE-opt-3: Dense twins.** A dense run per recipe at matched active parameters with
  identical data order, so every routing finding has a dense control.
- **MSUITE-opt-4: Release** as a checkpoint suite with routing logs and eval tables.

---

## 2. Doability and impact

### Overall doability: **medium** — compute- and operations-bound

- Operationally the heaviest new-training item after the WSD retrain suite; MoE training at
  tiny scale can be finicky; expert specialization with many experts at small scale may be
  weak.
- The main historical risk — standing up MoE training infra — is removed: a working repo
  with validated small-scale hyperparameters exists.
- Composes with the dense branch work: same branch runner, held-out token set, reference
  scorer, results store.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| Core (MSUITE-1–4) | **High** (ceiling comparable to the dense flagship) | Resource contribution + mechanism result; "nobody has this." |
| MSUITE-opt-1 branches | High | Makes the suite the MoE arm of the annealing/token-movement program. |
| MSUITE-opt-2 tiny floor | Low–Medium as a paper | Real audience; "MoE advantage fades gradually" is hard to publish. |
| MSUITE-opt-3 dense twins | Medium (supporting) | Required for credibility. |
| MSUITE-opt-4 release | Medium | Cited as a resource if adopted. |

---

## 3. Infrastructure build sequence

1. **Training config parity.** Reproduce one DataDecide recipe on the MoE stack at the
   chosen config; confirm loss/eval parity expectations.
2. **Recipe data access** via the DataDecide shard manifests (realized token shares, not
   labels).
3. **Dense checkpointing + routing logging** as standard run outputs.
4. **Held-out token set.** Choose a held-out token set once and freeze it: fixed, versioned token
   sequences with a manifest; stratified across domains and across the DataDecide leaf
   corpora; sized so that each domain × entropy-bucket cell has enough tokens to estimate
   mean per-token loss drop within a set tolerance, while keeping one forward pass per
   checkpoint cheap. Per-token loss on it is a standard output of the eval harness for every
   checkpoint variant (raw checkpoints, merged checkpoints, branch starts and endpoints),
   stored as compact arrays keyed by (checkpoint variant, held-out-set version). Branch
   endpoints also save their weights. Cheap to add now; expensive to retrofit later because
   it would mean re-running branches. *An identical spec appears in Annealed readouts, WSD retrain suite, Token-level
   movement, MoE movement, MoE recipe suite, and Functional featurization; keep them in sync.*
5. **Results store + eval harness.** Load any checkpoint; run the eval suite and perplexity evals; store results
   keyed by (run, step) plus a `variant` field (`raw`, `merged:<cfg>`, `branch:<cfg>`), in the
   same table schema as the processed OLMES tables so results slot into existing accessors.
6. **Pilot** one recipe, one seed, end to end; then scale out; then MSUITE-opt-3 twins.
7. *(Optional)* Decay-branch runner hookup (MSUITE-opt-1); packaging for release.


---

## 4. External assessments

Dated, attributed-by-date notes from external review conversations, recorded for
consolidation — not decisions. Only notes about this project are kept here. Related-work
claims in quoted text are unverified.

### 2026-08-22 — regularization is a stated design choice, not a default

From Danielle's SciSpace review of regularization for MoE LMs on repeated data (record
in `../topics/reference/regularization-literature.md`). If the suite trains 4–6 recipes
at small scale, multi-epoch exposure is likely, and the literature says MoE models
overfit repeated data more than dense ones (Xue et al. 2305.13230 — dropout, switched on
late, is the regularizer that works; Switch/ST-MoE use higher expert dropout and a
router z-loss; Hernandez et al. 2205.10487 show repeated-data double descent hitting
induction heads). Consequence for the training-config parity step: the dense
reproduction and the MoE arm need an explicit, matched regularization recipe (dropout
schedule, expert dropout, z-loss, epochs) recorded in the spec, and "does the data
choose the experts" should be read at matched epoch count, since repetition changes
routing specialization. The review itself lists the general and MoE-specific options;
its repeated-data section is about deduplication and does not apply.

### 2026-08-21 — origin: the Slicing-and-Dicing repo as apparatus

**The strategic point: the repo is the apparatus for a multi-recipe MoE mini-suite.** "You
have working MoE pretraining at the right scale range, validated configs, and now a
principled default architecture (the paper's own finding: fix expert size by active params,
use dropless routing, ignore the second-order knobs). Slicing-and-Dicing varied architecture
at fixed data; the follow-up varies data at fixed architecture and reads out routing/expert
specialization. Together they're a clean factorial story, and the follow-up is the
analysis-heavy paper you say you actually want — the sweep paper earns you the right to hold
architecture fixed. I'd now upgrade that direction relative to my earlier reweighting,
because its main risk (standing up MoE training infra) just evaporated."

---

### 2026-08-21 — the tiny-MoE floor (from the small-scale discussion)

**Tiny MoEs: the sweep's natural downward extrapolation.** "You found total-parameter
benefits persisting to 128:1 ratios and optimal expert size depending only on active params
— both invite the question of where those laws break as active scale shrinks. There are
concrete mechanisms that should impose a floor: each expert sees roughly budget/E tokens, so
at tiny scale experts fall below their own critical data threshold; the router itself is a
small model that has to learn a useful partition, and the routing-shallowness problem
(assignments collapsing to token-ID/frequency clustering) plausibly worsens as capacity
drops. So the question 'does a 5M-active, 500M-total MoE beat a 5M dense model, and does its
routing learn anything non-trivial' has real stakes for the local-model audience *and* feeds
the MoE analysis program: the taxonomy-realness question acquires a scale axis, and the
failure mode of routing at tiny scale is informative about what routing is doing at normal
scale. Your hpm guidance from the sweep is what makes this credible — the classic failure of
tiny-scale comparisons is that one arm is mistuned, and you're one of few groups holding
validated small-scale MoE hyperparameters."

### 2026-08-21 — an MoE sibling of the WSD suite

- The WSD retrain suite's §4 records the same point: a follow-up that "varies data at fixed architecture and reads out routing/expert specialization" would be the MoE counterpart of that suite, with its main risk already removed.

### 2026-08-21 — positions in ranked lists (full lists in `docs/portfolio-rankings.md`)

The 6–12-month flagship list names this as **Tier 1, #2, "Does the data choose the
experts?"** — the "reweighting" previously referenced here. "Train small MoE models
(FLAME-MoE-style config, ~40–100M active) on 4–6 DataDecide recipes spanning the outcome
range, 2–3 seeds, dense checkpoints with routing logged… Nobody has this. It's simultaneously
a resource contribution (the MoE analogue of DataDecide, at pilot scale), a mechanism result…
and it rescues [the routing follow-up] from orphan status… Risks: operationally the heaviest
new-training item after [the WSD suite], MoE training at tiny scale can be finicky, and
expert specialization at 64-experts/small-scale may be weak. I'd rate it slightly below #1
on probability-of-strong-outcome but comparable on ceiling." Offered as the
background-cluster track "if you're willing to accept MoE training risk for a shot at a
second novel paper." (That list predates the Slicing-and-Dicing discussion, so it assumes a
FLAME-MoE-style config rather than the sweep's validated defaults.)
- Tiny-MoE floor: **deliberately cut** from the workshop-sized list ("fair dense-baseline tuning at 5M-active is a tarpit and it's outcome-fragile").

### 2026-08-18 — origin and cautions (from the Research Trajectory page)

- FLAME-MoE (38M–1.7B active, 64 experts, top-8, full openness) as "DataDecide-for-MoE" at
  the target scale; "start from FLAME-MoE's validated configs rather than tuning fresh"
  (MoE knobs are folklore-tuned at large scale and "may be mis-set for 20–50M active").
  (The sweep's own validated defaults, confirmed later, supersede this.)
- "Tiny-scale noise: Signal-and-Noise found noise worsens as scale shrinks, and routing
  discreteness plausibly adds eval variance — so the noise-floor stage isn't skippable
  here, it's more necessary."
- "Keep a dense control ladder at matched active params… DataDecide's small dense models
  give it to you for free" (MSUITE-opt-3).
- Endgame: "matched-loss recipe comparisons run on FLAME-MoE-style models where the
  *routing fingerprint* is part of the movement profile — and if pretraining recipes at
  matched loss produce measurably different routing-commitment schedules that predict
  post-training or elicitation behavior, that's your thesis phenomenon with its mechanism
  visible to the naked eye."
## 5. Related work and positioning

*Purpose: the paper-facing synthesis — the prior-art landscape, this project's
position in it, and what each closest neighbor lacks. Unlike §4 (a dated intake
log, which grows by appending new entries **above this section**), §5 is a
current-state statement: rewrite it as understanding changes. Positioning claims
are Danielle's to make; agent-supplied literature claims anywhere in this document
are unverified leads, not established facts.*

**Status: raw material assembled from repository records (2026-08-24); positioning not
yet written. The full high-recall inventory now lives in
`related-work/moe-recipe-suite.md`; what follows is the load-bearing core.**

- **The repeated-data skeleton the spec depends on:** Xue et al. 2305.13230 (*To Repeat or
  Not To Repeat*: multi-epoch training of dense and MoE LMs; dropout switched on late is the
  regularizer that works; MoE models overfit repeated data more than dense ones),
  Muennighoff et al. 2305.16264 (repetition to ~4 epochs nearly free), Hernandez et al.
  2205.10487 (repeated-data double descent damaging induction heads). Consequence on record:
  the dense reproduction and the MoE arm need an explicit matched regularization recipe, and
  "does the data choose the experts" must be read at matched epoch count. These three are
  Claude-added ledger rows, unverified.
- **The MoE regularizer inventory behind the frozen spec** (SciSpace-agent record,
  unverified, several citations flagged off-target): load-balancing auxiliary loss (Shazeer
  2017; Switch, Fedus 2022), ST-MoE router z-loss (Zoph 2022), expert / cluster-level expert
  dropout (MoEC 2207.09094; Elbayad et al., Findings ACL 2023), Dirichlet-prior router
  shaping for upcycled MoEs (2510.01185); listed as missing: Switch expert dropout, ST-MoE's
  fine-tuning-overfitting finding, Gating Dropout 2205.14336, StableMoE, DeepSeek's
  auxiliary-loss-free balancing, OLMoE's stability recipe.
- **The readout precedents MSUITE-4 reports against:** OLMoE router saturation as the field's
  existing commitment metric; the three-phase load-balance trajectory; the OpenMoE token-ID-
  dominated routing finding, which MSUITE-4 tests across recipes; *The Myth of Expert
  Specialization* — whose claim that load-balancing loss suppresses shared hidden directions
  "explaining specialization collapse under less diverse data" is a direct prediction about
  recipe diversity. All quoted from the 2026-08-18 intake, unverified.
- **The mechanism prior for what recipes would change:** Jelassi et al., *Mixture of Parrots*
  (ICLR 2025) — if experts are storage rather than reasoning capacity, recipe differences
  should appear as *what* gets stored, which is the routing-taxonomy reading.
- **The external validation points and the gap:** FLAME-MoE (38M–1.7B active, 64 experts,
  top-8, full openness) and OLMoE serve as validation, not treatment variation; the record's
  claim across §4 and `../portfolio-rankings.md` is that no public multi-recipe MoE suite
  exists. FLAME-MoE's routing-log contents remain an open gate
  (`../open-questions-answered.md`).
- **The fixed-architecture prior:** the Slicing-and-Dicing sweep's own findings (total
  parameters always help even at 128× ratios; optimal expert size depends only on active
  parameters; other knobs second-order) are the defaults MSUITE-2 adopts, superseding the
  earlier "start from FLAME-MoE's configs" advice.
- **Measurement cautions on record:** Signal-and-Noise's finding that eval noise worsens as
  scale shrinks, with routing discreteness plausibly adding variance (so the noise-floor
  stage is "more necessary"), and the caution that MoE knobs are folklore-tuned at large
  scale and may be mis-set at 20–50M active.

All characterizations above are quoted from agent-generated or external-review intake and
are unverified; provenance for every identifier is in
`../litreview/citation-verification-ledger.md`. Full inventory:
`related-work/moe-recipe-suite.md`. Main accumulators:
`../topics/reference/regularization-literature.md`,
`../topics/reference/moe-literature.md`,
`../topics/reference/nonstationarity-accounting.md`, `../topics/reference/plasticity.md`,
plus `../topics/staging/datadecide-dense.md` for the shared regularization decision.

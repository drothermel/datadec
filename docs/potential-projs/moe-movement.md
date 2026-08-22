# MoE movement — reroute vs. rewrite, and whether the stability apparatus freezes the router

**One-line pitch.** In an MoE, change between checkpoints decomposes architecturally into
**rerouting** (same experts, different assignments) and **rewriting** (same assignments,
different experts), computable exactly by swapping routers and experts across adjacent
checkpoints — a causal-by-construction dual of the dense drift/diffusion decomposition.
Stage 1 measures this over training; Stage 2 asks whether early routing commitment is
imposed by the stability apparatus (balancing loss, z-loss, router LR) or intrinsic, via
reset, thaw, and timescale interventions.

IDs: MOVE-1–MOVE-3 (Stage 1), MOVE-4–MOVE-6 (Stage 2), MOVE-opt-1–MOVE-opt-4.

**Structure.** *Stage 1 (descriptive; T1 on existing checkpoints plus a few own runs with
dense checkpointing)*. *Stage 2 (causal; T3 interventions on the MoE training stack)*. Stage
2 needs Stage 1's machinery; Stage 1 stands alone.

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short
continued-training branches; **T3** = new pretraining runs.

---

## 1. What the project involves

### Stage 1 — descriptive core

1. **Reroute-vs-rewrite decomposition (MOVE-1).** For adjacent checkpoints t, t+1 on a fixed
   probe corpus: evaluate (router_t, experts_t+1) and (router_t+1, experts_t) alongside the
   two real checkpoints; attribute the output delta (per-token loss, logits) to rerouting
   vs. rewriting, per layer. Conjectured phenomenology: early training rerouting-dominated,
   late training rewriting-dominated, with a per-layer crossover.
2. **Commitment clocks (MOVE-2).** Per layer: per-token assignment flip rates split into
   reverting (t → t+1 → back at t+2) and persistent flips; router saturation (overlap of
   top-k at t with the final checkpoint). Reverting flips read as wall oscillation,
   persistent flips as river movement, saturation as basin commitment.
3. **Per-expert input drift (MOVE-3).** Divergence between the distribution of tokens routed
   to expert i at t vs. t+k, from routing logs alone — "how much continual learning is this
   expert experiencing," per layer, per config. A standardized diagnostic.

### Stage 2 — causal core

4. **Commitment timing vs. the stability apparatus (MOVE-4).** Across balancing-mechanism
   arms (plus arms varying aux-loss weight, z-loss, router LR multiplier), measure MOVE-2
   curves. Invariant to the knobs → commitment is intrinsic (the self-reinforcing
   "commitment ratchet"); tracks them → the freeze is imposed.
5. **The reset test (MOVE-5).** Mid-training, reinitialize or heavily perturb the router and
   continue. Three outcomes: re-converges to the same partition (data-driven attractor);
   different partition at equal loss (underdetermined; the freeze is dynamical); loss
   improves (the early partition was bad and training was locked into it).
6. **Annealing the suppressor (MOVE-6).** Decay the balancing loss to zero after warmup, or
   raise router LR late; does routing resume moving, and is the movement drift (persistent)
   or chatter (reverting)? Structurally identical to an LR-decay experiment with the
   router's constraint schedule in the role of the learning-rate schedule.

### Optional directions

- **MOVE-opt-1: Two-timescale as a design axis.** If the freeze is imposed and costly:
  router LR schedules decoupled from expert schedules; periodic "routing thaw" windows.
- **MOVE-opt-2: Frozen-router branches.** Freeze the router mid-run and continue training
  as the clean causal separation of the two channels.
- **MOVE-opt-3: Flips by token entropy.** Bucket probe tokens by reference-model entropy;
  do high-entropy tokens keep flipping experts after low-entropy tokens' routes freeze?
  Needs a reference-model scoring pass.
- **MOVE-opt-4: Dense control.** Run the dense drift/diffusion decomposition on dense models
  at matched active parameters so each MoE finding has an "is this MoE or just small"
  comparison.

---

## 2. Doability and impact

### Overall doability: Stage 1 **medium** (ingest-gated), Stage 2 **medium** (sequential)

- Stage 1 on released suites hinges on what the routing logs contain (which checkpoints,
  token recoverability); if aggregate-only, routing must be recomputed from checkpoints (T1
  with a new model-loading path). Own runs with dense checkpointing on the MoE stack remove
  that uncertainty but put training wall-clock on the critical path.
- Stage 2 is inherently sequential (pilot → grid) and its most exciting outcome (frozen
  routing costs loss) is the least likely; the other two reset outcomes are still findings.
- Routing is partly shallow; all claims need token-ID/frequency/position covariates.
- Any answer to MOVE-1 is paper-shaped.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| MOVE-1 decomposition | **High** | New, exact decomposition of training movement; a conceptual tool others will apply. |
| MOVE-2 commitment clocks | Medium-high | Router saturation is known; the reverting/persistent split is not. |
| MOVE-3 input drift | Medium-high | A new standard diagnostic; the MoE analogue of a realized-composition audit. |
| MOVE-4 timing vs. knobs | High either way | Decides imposed vs. intrinsic commitment; half extractable from existing sweep arms. |
| MOVE-5 reset test | **Very high if outcome 3** | "Frozen routing costs loss" would change how MoEs are trained. |
| MOVE-6 thaw | Medium-high | The causal version of "did we remove the incentive or was there never one." |
| MOVE-opt-1 timescale knob | High (conditional) | Converts the diagnosis into a knob industry would use. |
| MOVE-opt-3 flips by entropy | High if positive | The MoE twin of the dense entropy-bucket figure. |

---

## 3. Infrastructure build sequence

1. **Artifact survey** of released MoE suites (FLAME-MoE, OLMoE, OpenMoE): routing-log
   schema, checkpoint coverage, token recoverability. Decides T0 vs. T1 for Stage 1.
2. **Ingest** of usable suites into long routing tables (checkpoint, layer, token
   id/position, expert ids), evals into the standard trajectory schema.
3. **Own dense-checkpoint runs** on the MoE training stack (validated small-scale
   hyperparameters), with routing logged on the held-out token set.
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
5. **Swap-evaluation runner (MOVE-1)** and **flip/saturation/input-drift metrics (MOVE-2,
   MOVE-3)**.
6. **Intervention arms (Stage 2):** balancing/z-loss/router-LR arms, router-reset
   procedure, thaw schedules; pilot on one config before committing the grid.
7. **Reference-model scorer** for MOVE-opt-3; dense baselines for MOVE-opt-4.


---

## 4. External assessments

Dated, attributed-by-date notes from external review conversations, recorded for
consolidation — not decisions. Only notes about this project are kept here. Related-work
claims in quoted text are unverified.

### 2026-08-21 — origin: the factorized locus of change

*A factorized locus of change.* "This is, I think, the deepest structural difference. In a
dense model, 'what changed between checkpoints' is a formless question — you attack it with
KL, CKA, drift/diffusion. In an MoE, change decomposes *architecturally* into two channels:
**rerouting** (same experts, different assignments — the function changed because tokens go
elsewhere) and **rewriting** (same assignments, different experts — the function changed in
place). You can compute this decomposition exactly: hold routing fixed at checkpoint t while
using checkpoint t+1's experts, and vice versa, and attribute the output delta. That's the
MoE dual of your drift/diffusion decomposition, and it's *causal by construction* rather than
inferred from time-series statistics. Nobody has run this decomposition over training on an
open suite. My guess at the phenomenology — early training is rerouting-dominated, late
training rewriting-dominated, with the crossover per layer being a 'commitment clock' — but
the point is that any answer is a paper-shaped answer, in the same either-way-safe genre as
your IRT dimensionality question."

### 2026-08-21 — origin: the frozen-routing hypothesis (non-stationarity discussion)

### The mechanism: routing change is a distribution shift for the experts

"Each expert's effective training distribution isn't the data — it's the data as gated by the
router. So an MoE under perfectly stationary data is internally a continual-learning system:
if the router moves, every downstream expert experiences covariate shift; and if experts
move, the router's optimal assignment changes. That's a coupled two-timescale system, and the
failure mode of such systems is well known from RL and GANs — co-adaptation chatter,
oscillation, collapse. The field's stability toolkit for MoEs (auxiliary balancing losses,
router z-loss, jitter, capacity factors, sometimes explicit router LR reduction or freezing)
reads naturally as the same move RL makes with target networks: deliberately slow one
timescale so the other can converge against a quasi-stationary target. Under that reading,
early routing commitment isn't a discovered property of MoEs — it's the intended effect of
the stability apparatus, and OpenMoE-style findings are observations of a system we tuned
into that regime. Your 'removed the ability or incentive for routing to update' is exactly
the target-network trade: stability purchased by freezing the component whose adaptation
creates the non-stationarity."

"Two pieces of existing evidence cut interestingly here. Hash-routing results (fixed,
content-free, random assignment performing surprisingly close to learned routing) show that a
completely frozen, never-adaptive router is nearly sufficient — consistent with a world where
learned routers are allowed to adapt only briefly before the dynamics pin them, and where
most of routing's value is 'a consistent partition exists' rather than 'the partition is
smart or keeps improving.' And your own paper's finding that load-balancing mechanism barely
affects final quality is provocative in this light: either the balancing apparatus isn't
what's freezing routing, or routing freedom just doesn't matter for loss — and those two
readings predict different things under intervention."

**The alternative hypothesis, held honestly.** "Early commitment might have nothing to do
with the stability apparatus. Dense networks also commit early — critical-period effects,
early symmetry breaking, lottery-ticket structure all appear in the first fraction of
training with no router in sight. Token-ID routing may simply be the easy local optimum for a
small softmax router, reached fast and self-reinforcing (experts specialize to their tokens,
which raises the cost of reassignment — a commitment ratchet that needs no external
suppression). The virtue of your position is that this is directly decidable."

### Why this is unusually testable

"The discriminating experiments are all small, and several reuse the reroute-vs-rewrite
machinery [MOVE-1, MOVE-2]:"

- **Commitment timing vs. the stability apparatus.** "Across your sweep's
  balancing-mechanism arms (plus new arms varying aux-loss weight, z-loss, router LR
  multiplier), measure routing-commitment curves — saturation timing, flip rates, the
  persistent/reverting split. If commitment timing is invariant to the stability knobs, your
  hypothesis is mostly wrong and the ratchet story wins. If it tracks them, you've shown the
  freeze is imposed, not intrinsic. Either outcome is a real result, and you may be able to
  extract half of it from checkpoints you already have."
- **The reset test.** "Mid-training, reinitialize or heavily perturb the router and continue.
  Three distinguishable outcomes: it re-converges to the same partition (commitment is
  data-driven — the partition is an attractor); it converges to a different partition at
  equal loss (partitions are underdetermined and the freeze is dynamical — many equivalent
  factorizations, we just keep the first one found); or loss improves (the early partition
  was actually bad and we've been locked into it — the strongest version of your 'optimized
  away the incentive' story). This is cheap at your scales and directly answers whether
  frozen routing costs anything."
- **Annealing the suppressor.** "Decay the balancing loss to zero after warmup, or raise
  router LR late in training, and watch whether routing resumes moving and whether that
  movement is drift (persistent reassignments that stick) or chatter (reverting flips). This
  is the causal version of 'did we remove the incentive or was there never one' — and note
  it's structurally identical to your LR-decay/river-valley experiments, with the router's
  constraint schedule playing the role of the learning-rate schedule."
- **Two-timescale as a design axis.** "If the freeze is imposed and costly, the constructive
  follow-up is treating router-vs-expert timescale separation as an explicit hyperparameter —
  router LR schedules decoupled from expert schedules, periodic 'routing thaw' windows.
  That's the paper that would make industry care despite the small scale, because it
  converts the diagnosis into a knob."

**The standard diagnostic.** "Per-expert input drift — the divergence between the
distribution of tokens routed to expert i at time t versus t+k, computable from routing logs
alone. That's the internal non-stationarity made into a logged scalar, the MoE analog of your
realized-composition audit. Expert-level 'how much continual learning is this expert
experiencing' curves, per layer, per config, would be a genuinely new standard diagnostic."

### 2026-08-21 — routing as a per-token commitment channel

**The reframing: routing as a data fingerprint.** "The OpenMoE analysis found that routing
decisions are predominantly based on token IDs with minimal context relevance, and
token-to-expert assignments are determined early in pretraining and remain largely fixed. If
routing is mostly a token-identity/frequency phenomenon locked in early, then **expert
assignment is essentially a data fingerprint** — a clustering of the corpus that the model
committed to based on early data statistics. That inverts the direction of your program in a
productive way: instead of 'featurize the data, predict the model,' you get 'read the
model's routing, recover a featurization of the data.' Concretely:"

- "Does the expert decomposition recover your intrinsic features (domain composition,
  frequency bands, determinism profile)? If routing ≈ token-ID clustering, it should correlate
  strongly with cheap corpus statistics — and *deviations* from that (context-dependent
  routing, late reassignments) mark exactly the tokens your entropy-bucket hypothesis cares
  about."
- "Routing-commitment timing as a per-token version of your migration analysis, observable
  from existing checkpoints with *no decay branches needed* — the categorical channel makes
  'committed vs. still moving' directly legible instead of inferred from KL. There's already
  adjacent work treating this dynamically — e.g. a recent paper tracking a three-phase
  load-balance trajectory (surge, stabilization, relaxation) across OLMoE and OpenMoE
  checkpoints — but it's aggregate-level; the per-token, data-linked version is open."
- "Cross-suite: OLMoE (largely DCLM-based) vs. FLAME-MoE vs. OpenMoE all have checkpoints and
  known data. Three points isn't a recipe axis, but it's enough to ask whether
  expert-specialization structure tracks corpus composition across independent training
  setups."

### 2026-08-21 — positions in ranked lists (full lists in `docs/portfolio-rankings.md`)

- Reroute-vs-rewrite: workshop-sized **#9**, full-conference **#7, "Reroute or Rewrite? Where Training Moves an MoE"** (expected high; ceiling high); sub B of **P4**.
- Frozen-routing case study: workshop-sized **#10**, full-conference **#8, "Does MoE Training Suppress Its Own Non-Stationarity?"** (expected medium-high; ceiling very high); the causal arm of **P4**. Quoted entries:

- Reroute vs. rewrite: **workshop-sized #9** ("Ninth on logistics, not on merit") and
  **full-conference #7, "Reroute or Rewrite? Where Training Moves an MoE"** ("iteration risk
  is logistical rather than scientific. **Expected impact: high**… **Ceiling: high.**").

The frozen-routing case study is **workshop-sized #10** ("Last only because every experiment
is a new training intervention… the most sequential of the ten. If the reset test lands on
outcome three — frozen routing costs loss — this retroactively becomes the most important
paper on the list") and **full-conference #8, "Does MoE Training Suppress Its Own
Non-Stationarity?"** ("inherently sequential (pilot → grid) and its most exciting outcome…
is the least likely one… **Expected impact: medium-high.** **Ceiling: very high** — if
suppressed routing adaptivity measurably costs quality, this changes how people train MoEs,
full stop"). Overlaps with the reroute-vs-rewrite paper ("#8 genuinely needs #7's
machinery").

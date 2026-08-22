# Non-stationarity accounting — the hidden variable across schedule, data-order, plasticity, and MoE routing

**Status:** topic (staging). Candidate exits: a thesis-level framing ("which non-stationarities
does training secretly depend on") that wraps annealed readouts, the realized-exposure
audit, the plasticity thread, and the MoE program; and a concrete first case study (the
frozen-routing hypothesis) with four small experiments and one standard diagnostic
(per-expert input drift).

**Question posed (Danielle, 2026-08-21).** On non-stationarity in MoE training: if
token-to-expert assignments are decided early and stay fixed (as reported for OpenMoE —
unverified), then routing changes would themselves induce non-stationarity and instability —
so has the training setup been optimized to remove that instability, with the side effect of
removing the ability or incentive for routing to update over training? Not confident enough
in that specific hypothesis to build a program around it, but non-stationarity seems like a
large subthread playing out across many training settings, not just the continual-learning
and RL spaces that discuss it explicitly. See
[../danielle-inputs.md](../danielle-inputs.md).

Related-work claims below are unverified (see [README.md](README.md)).

---

## 2026-08-21 — Response

"The hypothesis has a clean mechanism at its core… the sharp version is both more plausible
and more testable than you're giving it credit for."

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
machinery [see moe-analysis-program.md]:"

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

### The broader thesis: non-stationarity as the hidden variable

"Once you look for it, essentially every thread in your portfolio is a non-stationarity
thread wearing different clothes, and it's worth splitting the phenomenon into two kinds.
**Exogenous** non-stationarity is imposed by the training setup: the LR schedule (deliberate
non-stationarity in the optimizer, which is what the whole river-valley/annealing program
manages), data-order and realized-composition drift (your tiny-scale audit — accidental
non-stationarity in the data), and midtraining/multi-stage pretraining, which is nothing but
institutionalized distribution shift that the field adopted empirically without a theory of
when shift helps. **Endogenous** non-stationarity is generated by the model's own state:
routing is the clearest case, but dense models have a subtler one — even under iid data, the
effective distribution is data weighted by current gradient magnitude, so as
easy/deterministic tokens saturate, the learning signal automatically migrates toward harder
tokens. Every model runs an implicit self-curriculum; your Rho-1-style loss-trajectory
taxonomy and the river/wall token migration are measurements of exactly this.
Loss-of-plasticity in 'stationary' pretraining stops being paradoxical under this lens: from
the gradient's perspective, pretraining was never stationary."

"Continual learning and RL discuss non-stationarity explicitly because there it's undeniable;
pretraining has the same dynamics but launders them through the schedule, the data pipeline,
and (in MoEs) the architecture, so they get managed by folklore-level stabilizers rather than
theory. What's missing from the field — and what your instrument-building program is
unusually positioned to supply — is the **accounting**: how much non-stationarity does each
source inject, in comparable units (per-token gradient-distribution shift, per-expert input
drift, realized-composition drift), what does each stabilizer (balancing loss, EMA, decay,
warmup) actually suppress, and what does suppression cost. My genuine suspicion, aligned with
yours, is that some suppressed non-stationarity is load-bearing — the routing-thaw and reset
experiments are where that suspicion first becomes falsifiable cheaply."

**On the meta-question.** "You're right not to build a program around the specific
frozen-routing hypothesis — it's one falsifiable claim, and the reset test might kill it in a
week. Build around the accounting framework, and let the routing hypothesis be its first case
study. It slots in rather than sprawling: the reroute-vs-rewrite decomposition already
separates the two internal channels of change, the tiny-scale line gives you the powered
experimental substrate (routing resets with twenty seeds are trivial at 10M active), the
sweep gives you the stability-apparatus variation for free, and 'which non-stationarities
does training secretly depend on' is a thesis-shaped question that unifies the schedule
work, the data-order work, the plasticity work, and the MoE work under one roof — which,
given how consistently your last four directions have converged on shared instruments, might
just be the name of the thing you've been circling."

---

## 2026-08-21 — Position in the ranked lists (full lists in `../portfolio-rankings.md`)

The frozen-routing case study is **workshop-sized #10** ("Last only because every experiment
is a new training intervention… the most sequential of the ten. If the reset test lands on
outcome three — frozen routing costs loss — this retroactively becomes the most important
paper on the list") and **full-conference #8, "Does MoE Training Suppress Its Own
Non-Stationarity?"** ("inherently sequential (pilot → grid) and its most exciting outcome…
is the least likely one… **Expected impact: medium-high.** **Ceiling: very high** — if
suppressed routing adaptivity measurably costs quality, this changes how people train MoEs,
full stop"). Overlaps with the reroute-vs-rewrite paper ("#8 genuinely needs #7's
machinery").

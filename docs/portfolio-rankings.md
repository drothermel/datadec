# Portfolio rankings

Ranked lists of project directions produced in external conversations, each under a stated
objective. Kept whole here because the lists only make sense side by side; the entries about
any single project are also copied into that project's §4 (or topic doc), so each project doc
still stands alone. Recorded for consolidation, not decisions.

Related-work claims in quoted text are unverified (see `potential-projs/README.md`).

> **Scope of these rankings (added 2026-08-22).** All four lists were produced on 2026-08-21,
> before most of the portfolio existed in its current form. They rank a **small subset** of
> the full candidate list: the seven original projects plus a few directions named in
> conversation. Not ranked anywhere below: the five promoted project docs in their current
> form (PART, MOVE, MSUITE, TINY, FUNC were ranked only as conversation-stage ideas), the
> twelve staging topics in `topics/` (reset/transfer gaps, movement microscope,
> ICL-as-post-training, post-training experiment design, warm-starting grid,
> critical-period timing study, checkpoint tomography), and the research-hypothesis
> program (`research-hypothesis.md`). Treat the lists as a dated snapshot, not a current
> prioritization.

## Crosswalk — every project's position in each list

One row per project (current names); columns are the four lists below. "—" means not
listed; "component" means explicitly demoted to a section of another paper. Labels are the
lists' own (expected impact / ceiling; scoop risk).

| Project | 6–12-month flagship list | Workshop-sized top 10 | Full-conference top 10 (expected / ceiling) | Four projects from two subs (scoop risk) |
|---|---|---|---|---|
| Annealed readouts (ANN) | Tier 1 #1 (with TOK + REC determinism link; GEO-opt-3 folded in) | #4 (T0 half) | #5 "How Much of Your Checkpoint Suite Is Schedule Artifact?" (high / high) | P3, speed 3 (**high**) |
| Token-level movement (TOK) | Tier 1 #1 (mechanism + thesis halves) | #8 (Stage 1) | #9 "Which Tokens Does the Cooldown Fix?" (high / very high) | — |
| Recipe featurization (REC) | determinism link folded into #1; cross-suite Tier 2 #3; data card Tier 3 component | #1 (data card), #7 (realized-exposure audit) | #2 "What Is Actually in DataDecide" (med-high / med-high→high) | P2, speed 2 (medium) |
| IRT reanalysis (IRT) | Tier 2 #4 (months 1–2 insurance paper) | #2 | #1 "The Psychometrics of Pretraining Suites" (med-high / high) | P1, speed 1 (med-high); recommended start |
| Trajectory drift/diffusion (TRJ) | Tier 3 component | #3 | #3 "Anatomy of the Noise Term" (med-high / high) | — |
| Loss-landscape geometry (GEO) | Tier 3 component (GEO-opt-3 inside #1) | — | — | — |
| WSD retrain suite (WSD) | Tier 2 #5 (background enabler) | — | — | — |
| MoE partitions (PART) | (sweep not yet discussed) | #6 | #4 "One Partition, Many Architectures" (high / high) | P4 sub A, speed 4 (**low**) |
| MoE movement (MOVE) | — | #9 (Stage 1), #10 (Stage 2) | #7 "Reroute or Rewrite?" (high / high); #8 "Does MoE Training Suppress Its Own Non-Stationarity?" (med-high / very high) | P4 sub B + causal arm |
| MoE recipe suite (MSUITE) | Tier 1 #2 "Does the data choose the experts?" | tiny-MoE floor cut | — | — |
| Tiny-scale measurement (TINY) | — | #5 (decision-reliability frontier) | #6 "Measuring Learning Where Benchmarks Can't See" (medium / med-high) | P1 sub B (frontier) |
| Functional featurization (FUNC) | — | cut (second-act) | #10 "The Functional Types of Pretraining Data" (high / highest) | — |

Recurring observations across the lists: speed and ceiling are close to inverted; speed and
scoop risk run together (the T0 DataDecide papers are races; the sweep- and stack-dependent
work is protected); and every new direction consumes the same shared instruments (noise
floor, IRT θ, held-out token set + per-token logging, reference scorer, branch runner,
results store).

---

---

## 2026-08-21 — Objective: strongest outcome from a 6–12-month effort; "really strong" main-conference paper possible depending on results

Prompt (Danielle): "reweight the different project directions (including additional ones you
might define) based on strongest project outcome from a 6-12 month long effort. main
conference paper thats really strong vibes possible depending on how the results come out."

"Switching the objective from 'workshop paper, fast' to 'strong main-conference paper, 6–12
months' inverts most of the weighting logic. Three things change: variance becomes
affordable (you no longer need the floor to be publishable in 6 weeks, so
high-ceiling/high-risk directions rise); infrastructure stops being a penalty (branches,
retraining, harnesses are exactly what 6–12 months buys); and the bar shifts from
'defensible' to 'memorable' — main-conference-strong papers have causal evidence, a
mechanism, or a definitive artifact, plus one figure people screenshot. Pure T0 reanalyses
mostly cap out below that bar no matter how long you polish them."

### Tier 1 — flagship candidates

**1. The unified causal program: "Which tokens does the cooldown fix, and does the data
decide?"** (annealed readouts + token-level movement + the determinism features from recipe
featurization). "This was scattered across four documents; assembled, it's the strongest
thing in the portfolio and it's shaped like a main-conference paper:"

- "Practical hook: the decision-flip audit — which of DataDecide's ~300 recipe decisions
  survive annealing. Everyone using the suite cares."
- "Method: validated checkpoint merging on cosine checkpoints as cheap annealed evals, with
  real decay branches as ground truth."
- "Mechanism: causal per-token decay-responsiveness, crossed with the epistemic/aleatoric
  split and the entropy-bucket observational result — the first causal token-level test of
  the river/wall picture."
- "Thesis: cross-recipe migration dynamics for the same held-out tokens, with each corpus's
  determinism profile as the predictor of its schedule sensitivity. That closes the loop:
  data property → token dynamics → eval artifact."

"Your own assessment called [the causal core] 'highest ceiling in the programme, strong
enough for a main venue if the signal is clean' — and the 6–12 month frame removes its only
gate (branches). Crucially, it degrades gracefully: if the token-level mechanism is noisy at
150M, the audit + merging-validation half is still a solid paper; if it's clean, you have the
'vibes' figure — a heatmap of per-token decay-responsiveness vs. entropy bucket vs. training
position, with recipes overlaid. Budget one scale step up (~1B) for the core grid to preempt
the inevitable 'does this hold beyond toy scale' review."

**2. New direction: a multi-recipe MoE mini-suite — "Does the data choose the experts?"**
"This didn't exist in any document… the missing artifact is treatment variation on the MoE
side. Train small MoE models (FLAME-MoE-style config, ~40–100M active) on 4–6 DataDecide
recipes spanning the outcome range, 2–3 seeds, dense checkpoints with routing logged. Then
ask: do different corpora produce different expert decompositions? Does routing-commitment
timing (per-token, per-layer) track the corpus determinism profile? Do the token-ID-dominated
routing findings from OpenMoE hold across recipes, or is context-dependence itself
recipe-dependent?"

"Nobody has this. It's simultaneously a resource contribution (the MoE analogue of
DataDecide, at pilot scale), a mechanism result (routing as a categorical, directly
observable version of your token-migration story), and it rescues [the routing follow-up]
from orphan status — FLAME-MoE and OLMoE become external validation points rather than the
whole dataset. Risks: operationally the heaviest new-training item after [the WSD suite],
MoE training at tiny scale can be finicky, and expert specialization at 64-experts/small-scale
may be weak. I'd rate it slightly below #1 on probability-of-strong-outcome but comparable on
ceiling, and it composes with #1 (same branch runner, same held-out token set, same
reference scorer — you could even branch the MoE runs)."

### Tier 2 — strong papers, but capped ceiling or unfavorable time-scaling

**3. Cross-suite features→dynamics** (DataDecide + FineWeb ablations + DCLM runs + RegMix
models, predicting dynamics — schedule sensitivity, emergence timing, noise levels — not
endpoints, validated by cross-suite transfer). "The 6–12 month frame is what makes the
multi-suite ingest feasible and fixes n=25. Main-conference plausible, and the practically
hottest topic here. But it's irreducibly correlational, and its best headline ('determinism
profile predicts annealing response across suites') actually belongs inside #1 as the
data-link section. As a standalone it's a very good paper whose reviews will all say 'but
why.'"

**4. IRT with recipe-DIF as the headline.** "The problem isn't quality — [dimensionality +
DIF] could be excellent — it's that this project doesn't scale with time. The 10-week version
and the 10-month version are nearly the same paper; extra months buy polish, not evidence,
because it's observational on a fixed dataset. And the DIF-null risk at these scales is real.
Reweighted: do it anyway in months 1–2 as the fast insurance paper (workshop or short
main-conference submission), then let its dimensionality answer inform how you frame #1's
recipe claims. As the flagship bet: no."

**5. The WSD suite as a datasets-track resource + the post-training result.** "Feasible in
this window and DataDecide-WSD would be cited for years. But resource papers live or die on
adoption, the operational risk (data-order/eval parity) is the highest in the set, and its
headline analyses are subsets of #1. Correct role: run the stable phases as background
cluster utilization feeding #1's branch grid, and let 'release the suite' be a second paper
if the runs are clean — not the primary bet."

### Tier 3 — demoted from papers to components

- "**Drift/diffusion** — under the workshop lens this was my #1; under this lens it's the
  measurement/noise-floor section of whichever flagship you pick, plus the churn figures. Its
  standalone ceiling is a good-but-forgettable methods paper."
- "**Barriers** — standalone drops (the 'all cross-recipe barriers are high' degenerate
  outcome is too likely), but [barriers on annealed variants] — does annealing collapse
  barriers between recipes — is a genuinely great section inside #1, with a causal knob."
- "**Measured data cards** — indispensable hygiene and your most citable single table; not a
  main-conference paper. Ship it early as a short paper or blog-plus-artifact so it accrues
  citations while the flagship cooks."
- "**Pure FLAME-MoE reanalysis** — absorbed into #2."

### Portfolio recommendation

"If I had to commit: #1 as the flagship, with the determinism-profile data link and
[barriers-on-annealed-variants] folded in; IRT in months 1–2 as the cheap early paper and
feasibility probe (its noise-floor and coverage checks de-risk everything downstream); #2 or
[the WSD suite's] stable phases as the background-cluster track, choosing #2 if you're
willing to accept MoE training risk for a shot at a second novel paper, [the WSD suite] if
you want the safer enabler. That gives you one high-ceiling causal paper, one fast
publication, and one compounding artifact — and every piece of infrastructure (branch
runner, held-out token set, reference scorer, results store) is shared across all three."

"One honest caveat on 'strong vibes': the field's attention in this area currently rewards
causal, mechanism-level claims about training dynamics and artifacts people can build on,
and punishes reanalysis. That's exactly why the reweighting concentrates value into #1 and
#2 — they're the only two directions in the entire portfolio where the strongest plausible
outcome is a result nobody else can currently produce."

---

## 2026-08-21 — Objective: ten clearly separate workshop-paper-sized contributions, fastest-to-come-together first (shared infra set aside)

Prompt (Danielle): "pull out the 10 strongest clearly separate workshop paper sized
contributions, ordered with highest likelihood to come together quickly first + your
justification for why? Put aside the question of shared infra for now."

"Ordering logic: (1) does the input data exist today, (2) is the result robust to outcome
(either-way publishable) or does it need a specific finding to land, (3) how much new
machinery or training does it need."

**1. The DataDecide data card: realized composition, label correction, and per-source
attribution.** "Fastest by a wide margin: one recipe is already done, the other 24 are
mechanical, and the headline finding — mixture labels are shard fractions, not token
fractions, so 25/50/75 is really 43/69/87 — is *already in hand*. Per-source attribution via
the mix ladder and ablation families is a join against existing eval tables. Zero outcome
risk (measurement papers can't be null), and it's your most citable single artifact because
it corrects everyone currently using the suite. Two to three weeks to a draft."

**2. IRT dimensionality of DataDecide.** "Pure T0 over already-parsed per-instance tables;
methods are mature; the fit is a few hundred lines of VI. Crucially, it's the safest bet in
the portfolio on outcome: 'recipes vary along one axis at these scales' and 'here is the
factor structure' are both headline claims. The only real work is the response-matrix
builder and principled item filtering. Main risk is a coverage gap in the per-instance tables
— checkable in a day."

**3. Drift/diffusion decomposition of eval trajectories.** "Also pure T0, standard
time-series estimation on small series, robust to outcome — the movement-SNR table is the
artifact regardless of what it shows, and the noise floor is a citable public good on its
own. One gate (checkpoint spacing) checkable on day one, with a stated fallback. Slightly
slower than IRT only because the windowed drift+diffusion modeling and its validation on
synthetic series involve more judgment calls."

**4. How much of DataDecide is schedule artifact? MPL correction + decision-flip analysis.**
"T0, cheap, and answers the question every DataDecide user has ('can I trust
intermediate-checkpoint rankings?'). It ranks below the previous two for two reasons: your
own analysis correctly flagged that without ground-truth branches this is section-shaped and
needs careful framing to stand alone as a workshop paper, and the MPL's extrapolation to
hypothetical decays on these cosine runs is unvalidated — the mandatory held-out check
(predict final decayed loss from truncated curves) could partially fail and cap the claims.
Fast, but with a real methodological gate."

**5. The decision-reliability frontier: how small can you measure?** "Decision-accuracy-
vs-compute curves at 10–150M where the *treatment is the measurement method* (accuracy vs.
margins vs. θ vs. IRT-selected items). Mostly analysis over existing DataDecide tables;
robust to outcome (the frontier is the artifact, wherever it sits); clear constituency in
academic-compute work. It sits fifth rather than higher because its strongest version
consumes outputs from #2 and #3 — it can be written standalone with simpler metrics, but the
compelling version wants the IRT machinery first, so its clock effectively starts after #2."

**6. Is the token taxonomy a property of the data or the architecture? Sweep reanalysis.**
"Cross-config and cross-balancing-mechanism expert-matching over checkpoints you already own,
with token-ID/frequency controls for routing shallowness. Either outcome is strong:
invariance explains your paper's 'config choices barely matter' finding mechanistically;
non-invariance is a quality-equivalent-but-internally-different identifiability result. It's
mid-list only because of practical unknowns: whether the sweep saved checkpoints (or enough
of them) and the genuine methods work in soft/hierarchical expert-matching across
granularities. If checkpoints exist, this could move up two slots; it's also the sequel that
most directly compounds your NeurIPS submission." *(Checkpoints since confirmed to exist —
see `open-questions-answered.md`.)*

**7. Realized-exposure audit + the order-effect experiment at small scale.** "The audit half
(reconstruct actual token-stream composition vs. training position for small-scale runs) is
fast and builds directly on #1's manifest machinery. The interventional half — stratified vs.
sequential sampling at 10–50M, ten-plus seeds — is cheap training but *is* training, with
tuning-parity and pipeline-reconstruction fiddliness. Outcome-robust in a useful way:
material drift is a confound finding about a widely used suite; no drift is a validation
people will cite defensively. Strengthens further if #3's noise floors exist to test the
order effect against."

**8. Where does movement live? KL by entropy bucket.** "The highest-ceiling figure in the
dense program — a mechanism-level link between eval noise and landscape geometry, with a
practical low-noise-eval corollary. Eighth because it's the first entry needing a real
compute campaign (checkpoint loader, probe corpus, reference-model scoring, forward passes
over many checkpoints) *and* the first with genuine null risk: a clean null is informative
but folds back into #3 as a section rather than standing alone. The item-flip piece
de-risks it and ships early regardless."

**9. Reroute vs. rewrite: an architectural decomposition of training movement in MoEs.**
"Exact, causal-by-construction decomposition (swap routers and experts across adjacent
checkpoints) that nobody has run over training on an open suite; commitment-clock
phenomenology per layer; any answer is paper-shaped. Ninth on logistics, not on merit: it
needs either the FLAME/OLMoE ingest (format and checkpoint-coverage uncertainty) or fresh
runs of your own with dense checkpointing. With your training stack the fresh-run path is
credible but puts training wall-clock on the critical path."

**10. Does the stability apparatus freeze the router? Commitment curves + the reset test.**
"The most conceptually loaded entry — it's the first falsifiable case study of the
non-stationarity accounting thesis — and cleanly three-outcome-decidable (same partition /
different partition at equal loss / loss improves). Last only because every experiment is a
new training intervention (aux-loss arms, router-LR arms, mid-run resets, thaw schedules),
and the design space needs a pilot before the grid is committed. Cheap per-run at your
scales, but the most sequential of the ten. If the reset test lands on outcome three —
frozen routing costs loss — this retroactively becomes the most important paper on the list."

"Two things deliberately cut, for the record: the **tiny-MoE floor study** (real audience,
but fair dense-baseline tuning at 5M-active is a tarpit and it's outcome-fragile — a mushy
'MoE advantage fades gradually' result is hard to publish), and the **full annealing branch
grid / ε-perturbation functional featurization**, which remain your strongest *eventual*
papers but are second-act by construction. Also worth saying: the top four are
simultaneously your fastest papers and the field's likeliest to be scooped by the
DataDecide/Ai2 authors themselves — speed is the moat there, whereas for #6, #9, and #10 the
moat is that you own the training stack, so they tolerate a slower clock."

---

## 2026-08-21 — Objective: ten weak-accept-or-better full-conference submissions, ranked by speed to produce (including iteration/pivot time), labelled by expected impact and impact ceiling

Prompt (Danielle): "give me a list of 10 weak accept tier or higher full conference
submission projects (NeurIPS, ICLR, ACL, CoLLAs, etc) with as much distinction between them
as possible… Rank them by speed to produce (which includes the iteration time needed if its a
less clear cut outcome or if there isn't an obvious pivot if the results don't go our way)
but clearly label each by the 'expected impact' and the 'impact ceiling.'"

"Calibrating to your stated bar — Slicing-and-Dicing-level, i.e. a real systematic
contribution that lands weak-accept-or-better — a full submission needs roughly 2–3 mutually
reinforcing results plus a robustness story, so most of these compose several workshop-sized
pieces into one arc."

**1. "The Psychometrics of Pretraining Suites"** — IRT dimensionality + binary-vs-margin
response models + recipe-DIF + the derived low-noise/tiny-scale measurement instrument, on
DataDecide. "*Fastest because:* entirely T0, every component is either-way publishable
(dimensionality answer, DIF present or absent), and the pivot structure is built in — if DIF
is sparse, the measurement-instrument contribution carries the paper. Best venue fit:
ACL/EMNLP or NeurIPS D&B. **Expected impact: medium-high** (a new measurement framework
applied to a suite everyone uses). **Ceiling: high** — if recipes are one-dimensional at
these scales, 'matched loss = matched everything' is a quotable negative result about the
entire small-scale data-selection enterprise."

**2. "What Is Actually in DataDecide"** — realized composition + label correction +
per-source dose-response attribution + realized-exposure/order-effect audit with the
stratified-sampling intervention at small scale. "*Speed:* the core is mechanical and
half-started; the only training is 10–50M reruns with many seeds; no outcome risk anywhere
(measurement + a two-sided intervention result). **Expected impact: medium-high** —
corrections to a widely used artifact get cited steadily rather than loudly. **Ceiling:
medium-high**; it caps out unless the order effect is large, in which case 'small-scale data
decisions are confounded with data order' becomes a genuinely disruptive claim about
proxy-scale experimentation."

**3. "Anatomy of the Noise Term"** — noise floors + drift/diffusion decomposition +
re-derivation of Signal-and-Noise + the within-run LR test of river-valley + item-level
churn. "*Speed:* T0, but slower than #1 because the windowed drift+diffusion modeling
requires real iteration and the LR test has a known confound (LR and progress co-monotone
within a run) whose mitigation — cross-scale schedule-length comparison — may or may not
convince reviewers on the first pass. Pivot is clear: the SNR table + noise floors stand
without the river-valley claim. **Expected impact: medium-high.** **Ceiling: high** — a
clean zero-training confirmation of river-valley structure in public data is a widely
quotable figure."

**4. "One Partition, Many Architectures"** — expert-matching across the 2,000-run sweep:
taxonomy invariance across expert count, granularity, and balancing mechanism, with
shallow-routing controls, plus the nested-refinement question across granularities.
"*Speed:* no new training if checkpoints survive (the gating unknown), but the
soft/hierarchical matching method needs genuine development and validation — that's the
iteration cost. Both outcomes strong; no pivot needed. **Expected impact: high** — it
mechanistically explains your own paper's headline and introduces a reusable cross-MoE
comparison method. **Ceiling: high**; 'the data imposes its factorization and architecture
only sets its resolution' is a statement about MoEs generally, not about one suite."
*(Checkpoints since confirmed to exist.)*

**5. "How Much of Your Checkpoint Suite Is Schedule Artifact?"** — the full annealed-readouts
project: MPL correction + checkpoint merging on cosine checkpoints + ground-truth decay
branches + the ~300-decision flip analysis against a seed-noise floor. "*Speed:* medium —
branches at 150M–300M are cheap but the eval-harness/parity work is real, and the paper is
unusually outcome-robust (merging works → methods contribution; fails → ground-truth audit
still stands). Iteration risk concentrates in MPL fit quality, which only weakens one arm.
**Expected impact: high** — audits a suite everyone uses and potentially delivers 'annealed
evals for free.' **Ceiling: high**, especially if a meaningful fraction of published
decisions flip."

**6. "Measuring Learning Where Benchmarks Can't See"** — the tiny-scale program as one
paper: decision-reliability frontier by measurement method + elicitation/
capability-per-parameter under distribution narrowing + IRT-scheduled RL as the constructive
demonstration. CoLLAs or ICLR. "*Speed:* mid-list despite fast run cycles because it's the
most iteration-heavy entry — getting *any* clean RL/posttraining signal at 10–50M may take
several design loops, and the elicitation arm needs careful scoping to avoid the 'can tiny
models reason' trap. The frontier arm is the safe core; the RL arm is the differentiator.
**Expected impact: medium** — big-lab reviewers may shrug. **Ceiling: medium-high**; if
IRT-scheduled reward shaping demonstrably unlocks RL at scales where naive reward gives zero
gradient, the local-model and academic communities adopt it."

**7. "Reroute or Rewrite? Where Training Moves an MoE"** — the exact router/expert swap
decomposition over training (own runs + OLMoE/FLAME validation) + commitment clocks per
layer + the causal arm: frozen-router and router-thaw branches. "*Speed:* needs fresh runs
with dense checkpointing on your stack plus an ingest with known format risk; the
decomposition itself is either-way publishable, so iteration risk is logistical rather than
scientific. **Expected impact: high** — a new, exact decomposition of training movement is a
conceptual tool others will apply. **Ceiling: high.**"

**8. "Does MoE Training Suppress Its Own Non-Stationarity?"** — commitment curves vs. the
stability apparatus + the router-reset test + balancing-loss annealing/thaw schedules +
per-expert input-drift as a standardized diagnostic; the two-timescale design knob if the
freeze proves costly. "*Speed:* slower than #7 because it's inherently sequential (pilot →
grid) and its most exciting outcome (frozen routing costs loss) is the least likely one; the
hedge is that the reset test's other two outcomes are still real findings, just quieter.
**Expected impact: medium-high.** **Ceiling: very high** — if suppressed routing adaptivity
measurably costs quality, this changes how people train MoEs, full stop."

**9. "Which Tokens Does the Cooldown Fix?"** — the full token-level-movement causal stage:
per-token decay-responsiveness + migration trajectories across branch points + the
epistemic/aleatoric split + cross-recipe comparison, with the toy-language sanity check.
"*Speed:* slow — gated on #5's branch grid existing with per-token logging, and the delicate
aleatoric estimation plus per-token SNR at 150M means real iteration before the signal is
trusted. Pivot exists (descriptive taxonomy + partial decomposition) but is a step down.
**Expected impact: high** — the first causal token-level test of the river-valley mechanism.
**Ceiling: very high** — clean signal here is main-venue-strong and the mechanistic anchor
for the whole program."

**10. "The Functional Types of Pretraining Data"** — the ε-mixture-perturbation response
tensor at small scale: data typed by response profile across training stages, rank of the
tensor, plasticity cost per type, surrogate-ladder validation (gradient proxies → short
branches → full branches). "*Speed:* slowest by design — it consumes the frozen branch
policy from #5, the response-vector spec, and the powered tiny-scale substrate from #6, and
its central risk (stage × type effects small relative to shift transients and seed noise at
these scales) has only a modest pivot (the surrogate-validation study stands alone as a
methods paper). **Expected impact: high** if the effects exist. **Ceiling: the highest on the
list** — a low-rank answer to 'how many functional kinds of data are there, and when does
each act' is the kind of result that reframes data curation, and it's the paper the previous
nine were quietly building toward."

"Three honest notes on the ranking. First, the overlap you predicted concentrates in two
places: #1/#3 share the DataDecide-measurement identity (they're separable because one is a
latent-variable framework and one is a dynamics framework, but a cynical reviewer could ask
why they aren't one paper — have the answer ready), and #7/#8 share the MoE-dynamics identity
(separable as descriptive-decomposition vs. causal-intervention, and #8 genuinely needs #7's
machinery). Second, the speed ordering and the ceiling ordering are close to inverted — your
fastest papers are your safest and your slowest are your biggest — which is the correct
shape for a portfolio, and argues for running one from the top third, one from the middle,
and starting the long-lead training for one from the bottom third concurrently rather than
executing the list in order. Third, scoop risk is also inverted: #1–#3 and #5 are races
(public data, obvious-in-retrospect questions, and the Ai2 authors adjacent to all of them),
while #4 and #7–#10 are protected by your stack and sweep — so if anything, the ranking
understates the urgency of the top and overstates the urgency of the bottom."

---

## 2026-08-21 — Objective: four strongest/most plausible weak-accept-or-better main-conference projects, each built from two non-archival workshop sub-contributions; ranked by speed, labelled by scoop risk, expected impact, impact ceiling; plus a recommended starting point

Prompt (Danielle): "identify a set of four strongest/most plausible projects, aiming to be
feasible and likely to be at least at the weak accept level+ projects for a main conf like
NeurIPS, ICLR, ACL, etc, that have a core main paper project built off of two workshop
submission sub-contributions (non archival of course). Again rank them by speed to produce,
label them with the likelihood to be scooped soon, and with the expected impact and impact
ceiling estimates. then recommend a place to start (this can be based on shared infra)."

"Here are the four, each structured as (workshop sub A + workshop sub B → main paper), chosen
for maximum separation: measurement, data, schedule, architecture."

**P1 — The Psychometrics of Pretraining Evaluation.** "*Sub A:* IRT dimensionality +
binary-vs-margin response models on DataDecide. *Sub B:* the decision-reliability frontier —
how far down the scale ladder reliable decision signal survives as a function of measurement
method (accuracy vs. margins vs. θ vs. IRT-selected items). *Main paper:* a measurement
framework for checkpoint suites — latent structure, optimal response models, recipe-DIF, and
the derived instrument for small-scale evaluation, validated on DataDecide end to end. The
two subs are genuinely independent (one asks 'what's the structure,' one asks 'what can you
detect with it') and the main paper is their product. **Speed: fastest.** All T0, both subs
either-way publishable, sub B mostly consumes sub A's outputs so the pipeline is serial but
short. **Scoop risk: medium-high** — public data, and the Ai2/Signal-and-Noise group is one
good idea away from it; speed is the defense. **Expected impact: medium-high. Ceiling:
high** (a one-dimensionality result reframes small-scale data selection; the instrument gets
adopted)."

**P2 — What's Actually in the Data: Composition, Order, and Small-Scale Validity.** "*Sub
A:* the data card — realized composition, the label≠token-share correction, per-source
dose-response attribution. *Sub B:* the realized-exposure audit — compositional drift as a
function of training position at small scale, quantified per scale. *Main paper:* both plus
the causal arm — stratified vs. sequential sampling reruns at 10–50M with many seeds,
testing whether data order confounds small-scale recipe decisions, with the
drift-shrinks-with-scale account of proxy-scale mispredictions. **Speed: second.** Sub A is
half-done, sub B is analysis over sub A's machinery; only the main paper's intervention
involves training, and it's tiny. Minimal outcome risk anywhere. **Scoop risk: medium** —
the label correction is discoverable by anyone who looks, but nobody seems to be looking,
and the order-effect experiment isn't obvious until you've done the audit. **Expected
impact: medium-high** (steady citations from every suite user). **Ceiling: medium-high**,
jumping to high if the order effect is large — 'proxy-scale data decisions are confounded
with data order' would be genuinely disruptive."

**P3 — Auditing the Schedule: Annealed Readouts for Cosine-Trained Suites.** "*Sub A:* MPL
correction + decision-flip analysis (the T0 half, with the held-out validation predicting
final decayed loss). *Sub B:* checkpoint merging on cosine checkpoints, validated against a
small pilot grid of ground-truth decay branches. *Main paper:* the full [annealed-readouts
project] — proxy validation at scale, the ~300-decision flip analysis against seed-noise
floors, branch-length sweep, canonical annealed re-release of the suite's decisions.
**Speed: third.** Sub A is fast; sub B introduces the eval-parity harness and first
branches, which is where wall-clock and fiddliness live. Outcome-robust at every level
(merging works → method; fails → ground-truth audit). **Scoop risk: highest of the four** —
it's the most obvious-in-retrospect question about DataDecide, WSM/merging papers are
circling it, and the Ai2 authors could run it internally with better access. Sub A should
ship early partly as a flag-plant. **Expected impact: high. Ceiling: high** ('annealed evals
for free' + flipped published decisions)."

**P4 — What MoE Configurations Actually Change: Partitions and Movement.** "*Sub A:*
taxonomy invariance — expert-matching across the sweep's configs and balancing mechanisms,
with token-ID/frequency shallow-routing controls. *Sub B:* reroute-vs-rewrite — the exact
router/expert swap decomposition over training on OLMoE/FLAME plus one or two of your own
densely-checkpointed runs. *Main paper:* a unified account of what varies and what's
invariant across quality-matched MoEs — the partition, its resolution-refinement across
granularities, where training movement lives, and the frozen-router/thaw causal arm
connecting to the stability-apparatus question. This is the direct sequel to your NeurIPS
submission and mechanistically explains its own headline finding. **Speed: fourth** —
checkpoint-survival audit, matching-method development, possibly fresh runs. **Scoop risk:
low** — the sweep is the moat; nobody else holds balancing-mechanism variation at matched
everything-else. **Expected impact: high. Ceiling: high**, with an option on very-high if
the causal arm shows suppressed routing adaptivity costs quality." *(Checkpoints since
confirmed to exist.)*

| | Speed | Scoop risk | Expected | Ceiling |
|---|---|---|---|---|
| P1 Psychometrics | 1 | Med-high | Med-high | High |
| P2 Data/order | 2 | Medium | Med-high | Med-high (→High) |
| P3 Schedule audit | 3 | **High** | High | High |
| P4 MoE partitions | 4 | **Low** | High | High (→Very high) |

"Note the portfolio shape: speed and scoop risk run together (P1–P3 are races on public data;
P4 is protected), which is exactly why you shouldn't execute serially."

**Where to start.** "The infra argument is decisive here, because P1, P2, and P3 sit on one
foundation: the DataDecide access layer — outcome tables with full (recipe × scale × seed ×
step × task) structure, the per-instance coverage check, the trajectory accessor with
spacing statistics, and the manifest/composition module. That's roughly two weeks of work
that simultaneously *starts* three projects and *runs their gates* (per-instance coverage
gates P1's DIF, checkpoint spacing gates nothing here but feeds P3's analysis, loss-curve
coverage gates P3's MPL). So concretely: **start P1 as the primary effort** — it's the
fastest to two subs, its outputs (θ, item parameters, noise-aware measurement) are inputs to
P2's intervention analysis and P3's flip-significance testing, and shipping it first blunts
the scoop exposure where the race is tightest. Run **P2's sub A in the background** from week
one — it's mechanical, half-done, and its manifest module is P2's foundation anyway. And
spend **one day, immediately, on the P4 checkpoint audit**: whether the sweep's checkpoints
survive is a zero-compute fact that determines whether your lowest-scoop-risk, highest-moat
project is cheap or expensive — and if they're at risk of deletion, that's the single most
urgent action item in this entire conversation." *(Done: final checkpoints confirmed; see
`open-questions-answered.md`.)*


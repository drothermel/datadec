# Potential projects — DataDecide

One document per candidate workshop-paper project. Each is written to stand alone so it can be
evaluated on its own merits: shared infrastructure is restated in every document that needs it
rather than factored out, and the only cross-document mentions are short coordination notes
("X specifies the same runner; reuse it if it exists"). IDs inside each document carry that
document's prefix (`ANN-1`, `TRJ-opt-2`, …); each document's header maps its IDs back to the
synthesis inventory it was derived from.

Each document has the same three parts:

1. **What the project involves** — the core experiment plus the optional directions.
2. **Doability and impact** — an overall doability take, then per-direction workshop-paper impact.
3. **Infrastructure build sequence** — what to build, in what order, if we proceed.

| Project | Core question | Training required | Compute tier | Standalone paper? |
|---------|---------------|-------------------|--------------|-------------------|
| [Annealed readouts](annealed-readouts.md) (`ANN`) | How much of DataDecide's reported ranking is a cosine-schedule artifact, and can it be corrected for the cost of evals? | short decay branches | T0 core; T1+/T2 for proxies and branches | **Yes — strongest candidate** |
| [WSD retrain suite](wsd-suite.md) (`WSD`) | What does a DataDecide-subset with a proper stable phase + decay branches enable, and is it worth keeping the cluster warm for? | full retrain (subset) | T3 | Resource paper; better as an enabler |
| [Loss-landscape geometry](landscape-geometry.md) (`GEO`) | Are cross-recipe metric comparisons well-defined, and does basin membership predict when recipe effects hold? | none | T1 | Yes, evals-only |
| [Token-level movement](token-movement.md) (`TOK`) | Stage 1: where does movement between checkpoints live, and does it concentrate on high-entropy tokens? Stage 2: which tokens respond to LR decay, does that track epistemic uncertainty, and do recipes differ in how tokens migrate? | Stage 1 none; Stage 2 decay branches | T0/T1 then T2 | Stage 1 standalone if its headline holds; Stage 2 highest ceiling |
| [Trajectory drift/diffusion](trajectory-statistics.md) (`TRJ`) | What lives inside the checkpoint-to-checkpoint "noise" term: directional drift vs. mean-reverting diffusion, and does diffusion track the learning rate? | none | T0 | Strong; checkpoint spacing confirmed adequate |
| [IRT reanalysis](irt-reanalysis.md) (`IRT`) | Do recipes differ along one latent axis or many, and which items behave differently across recipes at matched ability? | none | T0 | Strongest standalone bet |
| [Recipe featurization](recipe-featurization.md) (`REC`) | What is actually in the DataDecide recipes, and which measurable data properties explain which task-level differences? | none | T0/T1 | Yes, GPU-free |

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short decay
branches from existing checkpoints; **T3** = new pretraining runs.

Source inventories: [../refs/lr-schedule-wsd-synthesis.md](../refs/lr-schedule-wsd-synthesis.md)
(ANN, WSD, GEO, TOK Stage 2), [../refs/research-trajectory-synthesis.md](../refs/research-trajectory-synthesis.md)
(TRJ, IRT, TOK Stage 1), and [../dataset-analysis-idea-map.md](../dataset-analysis-idea-map.md)
(REC).

Resolved gate checks and open questions (with the code used to answer them) are logged in
[../open-questions-answered.md](../open-questions-answered.md). The recipe-featurization
literature review (plan and process) lives in [../litreview/](../litreview/). Ideas that do not yet
belong to a project are staged in [../topics/](../topics/).

## Program-level notes

Dated, attributed observations from external review conversations about the portfolio as a
whole — recorded for consolidation, not decisions. Project-specific notes live in each
document's §4.

### 2026-08-21 — "general thoughts" from two external reviews

- **One thesis.** *Reviewer 1:* "Almost everything here tests either 'pretraining data shapes
  models beyond final performance' (recipe-DIF, matched-loss drift/diffusion signatures,
  token-migration comparison, feature→outcome regression) or 'the cosine schedule contaminates
  intermediate readouts' (all of annealing, the LR test, annealed-target regression). That's
  a strength for a research program — but it means you should decide which instrument gets to
  make each claim first, or you'll end up with three papers each half-claiming the same
  result." *Reviewer 2:* "Nearly everything reduces to: DataDecide's numbers conflate durable
  progress with schedule/measurement artifact; separate them, and ask what recipes change once
  you do. IRT separates ability from item noise; drift/diffusion separates trend from
  oscillation; annealing separates river from wall; token bucketing localizes where each lives.
  That's a strength — the papers cite each other and the intro writes itself — but it means
  you should consciously sequence them as one arc (T0 audit papers → branch grid → causal
  token-level) rather than treating them as a menu."
- **Single point of failure.** *Reviewer 1:* "Everything rests on one suite, 25 recipes, 3
  seeds, small scales. The recurring risk rows — '3 seeds is thin,' 'n=25 underpowered,'
  'signal may be inside seed variance,' 'results may not hold at scale' — are all the same
  risk restated. The noise-floor module isn't just reusable hygiene; it's the thing that
  determines whether the whole program has publishable effect sizes. That's another argument
  for [trajectory drift/diffusion] first: it's partly a feasibility study for everything
  else."
- **Nulls: informative vs. merely reportable.** *Reviewer 1:* "Be a little skeptical of
  'either outcome is a result.' It's true for [IRT dimensionality], where the null is a
  genuine substantive claim. It's shakier for others — a null recipe-DIF at 150M is ambiguous
  between 'recipes are one-dimensional' and 'these scales are too small to see it,' and
  reviewers will pick the boring interpretation. When you write, distinguish nulls that are
  informative from nulls that are merely reportable." *Reviewer 2:* "Your impact tables have a
  tell: the highest-rated items are disproportionately 'high if positive.' The portfolio
  hedges this well because every recommended core is result-either-way — keep that discipline.
  The corollary is that the noise-floor and coverage work isn't hygiene, it's what lets you
  claim a null is a finding rather than underpowering."
- **Two global decisions, made once.** *Reviewer 2:* "The reference model (and context
  lengths) for entropy scoring appears in four projects — ablate it once, freeze it. Same for
  the held-out probe/token set: [annealed readouts, WSD suite, token-level movement] all say
  it's expensive to retrofit. These are program-level decisions masquerading as per-project
  ones."
- **Clock / competition.** *Reviewer 2:* "The Signal-and-Noise / DataDecide authors are the
  obvious people to do [the drift/diffusion and annealing-T0 reanalyses] themselves — these
  are reanalyses of their own artifacts. That argues for shipping the T0 papers fast (they're
  your speed advantage) and for considering the authors as collaborators on the branch-grid
  sequel, where your infrastructure investment is the moat."
- **n = 25 is the one honest ceiling.** *Reviewer 2:* "The family-contrast/dose-response
  framing [in recipe featurization] is the right answer and should back-propagate everywhere
  recipes are compared — including IRT DIF and matched-loss signatures, which currently treat
  25 recipes as exchangeable. Anywhere a table says 'compare across recipes,' ask whether it
  can instead say 'compare within a family along a measured dose.'"
- **Gates audit.** *Reviewer 2:* "Several projects hinge on cheap empirical checks nobody has
  run: per-instance coverage across all recipe×scale×seed cells (gates IRT and DIF),
  checkpoint spacing per scale (gates drift/diffusion), loss-curve coverage in the
  scaling-law table (gates the MPL fit), FLAME-MoE log contents (gates the routing follow-up
  entirely). One week of coverage reports would re-rank this whole portfolio with data instead
  of guesses." Status: the first two are answered and the last two are open — see
  [../open-questions-answered.md](../open-questions-answered.md).
- **Observational → causal pairing.** *Reviewer 1:* "Your best long-game structure. [Stage 1
  of token-level movement] now, [Stage 2] later once the branch runner exists. If the
  entropy-bucket result holds observationally, it both de-risks and motivates the causal
  follow-up; if it doesn't, you saved yourself the branch compute. That's the cleanest
  dependency chain in the set, and worth preserving explicitly."


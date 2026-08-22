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
| [MoE partitions](moe-partitions.md) (`PART`) | Is the token taxonomy a property of the data or the architecture? Expert-matching across the Slicing-and-Dicing sweep with shallow-routing and load-balancing controls | none (sweep checkpoints exist) | T1 | Yes — either outcome strong |
| [MoE movement](moe-movement.md) (`MOVE`) | Stage 1: does training move an MoE by rerouting or rewriting, per layer over time? Stage 2: does the stability apparatus freeze the router, and does that cost loss? | Stage 1 own dense-checkpoint runs; Stage 2 interventions | T1 then T3 | Stage 1 standalone; Stage 2 very high ceiling |
| [MoE recipe suite](moe-recipe-suite.md) (`MSUITE`) | Does the data choose the experts? Small MoEs on 4–6 DataDecide recipes at fixed architecture, routing logged | full retrain (subset) | T3 | Resource + mechanism paper |
| [Tiny-scale measurement](tiny-scale-measurement.md) (`TINY`) | How far down the scale ladder does reliable decision signal survive, as a function of measurement method; what can be done at 10–50M with replicates? | none for core; small for options | T0/T1 | Yes; options iteration-heavy |
| [Functional featurization](functional-featurization.md) (`FUNC`) | What are the functional types of pretraining data — chunks typed by how they move a model at each training stage? | mixture-perturbation branches | T2 | Second-act; highest ceiling |
| [ICL elicitability](icl-elicitability.md) (`ICL`) | Does ICL-ability differ across recipes at matched loss, and can the tuned elicitation ceiling serve as the null that weight-update claims must beat? | none | T1 | Workshop from the core; main-conference with the two-tier design |
| [Intervention grid](intervention-grid.md) (`GRID`) | Are critical periods, the warm-starting gap, and ITER's memory effect one sensitive period — and is it an identifiability phase transition? One harness, many seeds, a diagnostic panel | CIFAR-scale CNNs, then an LM diagonal | T3 (cheap) | Main-conference / TMLR; workshop from the warm-start decomposition alone |
| [Movement microscope](movement-microscope.md) (`MIC`) | What does post-training movement look like at 150M — noise floor, calibrated sensitivity, decomposition — and do recipes differ in movement profile at matched loss? | tiny fine-tunes | T2 | Workshop from Stages 1–3; main-conference with Stage 4 |
| [Embedding-reset dynamics](embedding-reset-dynamics.md) (`RESET`) | How fast does an LM recover from an embedding reset as a function of scale, stage, and seed; why are input and output resets different; is the reset basin-preserving; does it restore plasticity? | small continued-training runs on PolyPythias | T2 | Workshop from the recovery curves; main-conference with the basin and plasticity arms |

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short decay
branches from existing checkpoints; **T3** = new pretraining runs.

Source material: the Research Trajectory Notion page is mirrored in
[../refs/research-trajectory-pre-to-post-training.md](../refs/research-trajectory-pre-to-post-training.md)
(re-pulled 2026-08-21; see [../refs/README.md](../refs/README.md)). The three earlier synthesis
documents that the project IDs cite (LR-schedule/WSD synthesis; published-data-analysis
synthesis; dataset-analysis subset) were removed from `refs/` on 2026-08-21 and remain in git
history and on the closed PR 45/46 branches.

Resolved gate checks and open questions (with the code used to answer them) are logged in
[../open-questions-answered.md](../open-questions-answered.md). The recipe-featurization
literature review (plan and process) lives in [../litreview/](../litreview/). Danielle's organizing
hypothesis, in her words with agreed refinements, is [../research-hypothesis.md](../research-hypothesis.md). Ideas that do not yet
belong to a project are staged in [../topics/](../topics/). Ranked lists of directions
under different objectives are in [../portfolio-rankings.md](../portfolio-rankings.md). Danielle's own prompts from the source
conversations are logged verbatim in [../danielle-inputs.md](../danielle-inputs.md).

## Program-level notes

Dated, attributed observations from external review conversations about the portfolio as a
whole — recorded for consolidation, not decisions. Project-specific notes live in each
document's §4.

**Caveat on quoted related-work claims.** Statements in quoted external text about what does or
does not exist in the literature ("nobody has…", "this is unclaimed", specific paper
attributions) are unverified: the responding agents did not run literature searches and have no
reliable knowledge of the current landscape. Treat them as leads to check, not facts. The ideas
stand on their own. This applies equally to every §4 section.

### 2026-08-21 — general thoughts on the portfolio

- **One thesis.** One framing: "Almost everything here tests either 'pretraining data shapes
  models beyond final performance' (recipe-DIF, matched-loss drift/diffusion signatures,
  token-migration comparison, feature→outcome regression) or 'the cosine schedule contaminates
  intermediate readouts' (all of annealing, the LR test, annealed-target regression). That's
  a strength for a research program — but it means you should decide which instrument gets to
  make each claim first, or you'll end up with three papers each half-claiming the same
  result." Another: "Nearly everything reduces to: DataDecide's numbers conflate durable
  progress with schedule/measurement artifact; separate them, and ask what recipes change once
  you do. IRT separates ability from item noise; drift/diffusion separates trend from
  oscillation; annealing separates river from wall; token bucketing localizes where each lives.
  That's a strength — the papers cite each other and the intro writes itself — but it means
  you should consciously sequence them as one arc (T0 audit papers → branch grid → causal
  token-level) rather than treating them as a menu."
- **Single point of failure.** "Everything rests on one suite, 25 recipes, 3
  seeds, small scales. The recurring risk rows — '3 seeds is thin,' 'n=25 underpowered,'
  'signal may be inside seed variance,' 'results may not hold at scale' — are all the same
  risk restated. The noise-floor module isn't just reusable hygiene; it's the thing that
  determines whether the whole program has publishable effect sizes. That's another argument
  for [trajectory drift/diffusion] first: it's partly a feasibility study for everything
  else."
- **Nulls: informative vs. merely reportable.** "Be a little skeptical of
  'either outcome is a result.' It's true for [IRT dimensionality], where the null is a
  genuine substantive claim. It's shakier for others — a null recipe-DIF at 150M is ambiguous
  between 'recipes are one-dimensional' and 'these scales are too small to see it,' and
  reviewers will pick the boring interpretation. When you write, distinguish nulls that are
  informative from nulls that are merely reportable." Also: "Your impact tables have a
  tell: the highest-rated items are disproportionately 'high if positive.' The portfolio
  hedges this well because every recommended core is result-either-way — keep that discipline.
  The corollary is that the noise-floor and coverage work isn't hygiene, it's what lets you
  claim a null is a finding rather than underpowering."
- **Two global decisions, made once.** "The reference model (and context
  lengths) for entropy scoring appears in four projects — ablate it once, freeze it. Same for
  the held-out probe/token set: [annealed readouts, WSD suite, token-level movement] all say
  it's expensive to retrofit. These are program-level decisions masquerading as per-project
  ones."
- **Clock / competition.** "The Signal-and-Noise / DataDecide authors are the
  obvious people to do [the drift/diffusion and annealing-T0 reanalyses] themselves — these
  are reanalyses of their own artifacts. That argues for shipping the T0 papers fast (they're
  your speed advantage) and for considering the authors as collaborators on the branch-grid
  sequel, where your infrastructure investment is the moat."
- **n = 25 is the one honest ceiling.** "The family-contrast/dose-response
  framing [in recipe featurization] is the right answer and should back-propagate everywhere
  recipes are compared — including IRT DIF and matched-loss signatures, which currently treat
  25 recipes as exchangeable. Anywhere a table says 'compare across recipes,' ask whether it
  can instead say 'compare within a family along a measured dose.'"
- **Gates audit.** "Several projects hinge on cheap empirical checks nobody has
  run: per-instance coverage across all recipe×scale×seed cells (gates IRT and DIF),
  checkpoint spacing per scale (gates drift/diffusion), loss-curve coverage in the
  scaling-law table (gates the MPL fit), FLAME-MoE log contents (gates the routing follow-up
  entirely). One week of coverage reports would re-rank this whole portfolio with data instead
  of guesses." Status: the first two are answered and the last two are open — see
  [../open-questions-answered.md](../open-questions-answered.md).
- **Observational → causal pairing.** "Your best long-game structure. [Stage 1
  of token-level movement] now, [Stage 2] later once the branch runner exists. If the
  entropy-bucket result holds observationally, it both de-risks and motivates the causal
  follow-up; if it doesn't, you saved yourself the branch compute. That's the cleanest
  dependency chain in the set, and worth preserving explicitly."

### 2026-08-21 — "one lab thesis": measurement science of LM training at academic scale

- "The advantage [of small scale] is that cheapness converts into statistical practices the
  field otherwise never gets: 20+ seeds, factorial designs, power analysis, preregistered
  comparisons, real confidence intervals. Nearly every big-lab training paper is n=1 per
  configuration. A lab whose identity is 'we run LM training experiments to wet-lab
  evidentiary standards' is differentiated, and the measurement problem… — how do you even
  detect a training or HP-fitting signal at 150M — is the research question, not the
  obstacle."
- "The small-scale platform generates the model populations that IRT requires as respondents;
  the MoE repo is a validated apparatus already sitting in that scale range with a
  categorical observable dense models lack; and the probe battery + noise-floor work is the
  shared instrument suite. 'Measurement science of language-model training at academic
  scale' is a coherent identity that big labs structurally won't compete with — not because
  they can't, but because n=20-seed experiments on 150M models will never be their
  incentive." Full text in [../tiny-scale-measurement.md](tiny-scale-measurement.md).
- Tiny models as the program's *Drosophila*: "Every design… that's compute-gated at
  DataDecide scale — the (data × schedule) factorial, the ε-perturbation response tensor, the
  stage × type plasticity map, the reroute-vs-rewrite causal controls — becomes fully powered
  at 10–50M: twenty seeds instead of three, full factorials instead of corner samples,
  preregistered analyses with the noise floor known in advance. So the sequencing story is:
  the T0 measurement papers build the instruments; the tiny-scale program is where the
  interventional science gets run *properly first*; the DataDecide/MoE-scale versions become
  confirmation at 2–3 points on a scale ladder rather than underpowered first attempts."
  External-validity rule: ask dynamics/mechanism questions (scale-portable, checkable on a
  ladder), not capability-emergence questions.
- Convergence signal: "every new direction you've raised has turned out to be a new consumer
  of the same five or six pieces of measurement infrastructure, which is exactly what a
  coherent research program looks like from the inside."

### 2026-08-21 — non-stationarity as the hidden variable

- "Essentially every thread in your portfolio is a non-stationarity thread wearing different
  clothes." Exogenous (imposed by the setup): the LR schedule (the annealing program),
  data-order / realized-composition drift (the realized-exposure audit), midtraining /
  multi-stage pretraining. Endogenous (generated by the model's own state): MoE routing; in
  dense models, the gradient-weighted effective distribution that migrates toward harder
  tokens (the token-movement measurements).
- "What's missing from the field… is the accounting: how much non-stationarity does each
  source inject, in comparable units (per-token gradient-distribution shift, per-expert input
  drift, realized-composition drift), what does each stabilizer (balancing loss, EMA, decay,
  warmup) actually suppress, and what does suppression cost." Suggested as the thesis-shaped
  question unifying the schedule, data-order, plasticity, and MoE work, with the
  frozen-routing hypothesis as its first case study. Full text in
  [../topics/nonstationarity-accounting.md](../topics/reference/nonstationarity-accounting.md).

### 2026-08-21 — portfolio-shape observations from the three ranked lists

- Changing the objective from "workshop, fast" to "strong main-conference, 6–12 months"
  "inverts most of the weighting logic… variance becomes affordable… infrastructure stops
  being a penalty… the bar shifts from 'defensible' to 'memorable.' Pure T0 reanalyses mostly
  cap out below that bar no matter how long you polish them."
- "The speed ordering and the ceiling ordering are close to inverted — your fastest papers
  are your safest and your slowest are your biggest — which is the correct shape for a
  portfolio, and argues for running one from the top third, one from the middle, and
  starting the long-lead training for one from the bottom third concurrently."
- "Scoop risk is also inverted: [the T0 DataDecide papers] are races (public data,
  obvious-in-retrospect questions, and the Ai2 authors adjacent to all of them), while [the
  sweep- and stack-dependent MoE and branch work] are protected by your stack and sweep."
- Full lists: [../portfolio-rankings.md](../portfolio-rankings.md).
- From the "four projects from two workshop subs each" list: "P1, P2, and P3 sit on one
  foundation: the DataDecide access layer — outcome tables with full (recipe × scale × seed ×
  step × task) structure, the per-instance coverage check, the trajectory accessor with
  spacing statistics, and the manifest/composition module. That's roughly two weeks of work
  that simultaneously *starts* three projects and *runs their gates*." Recommended start: IRT
  as primary, the data card in the background, and (now done) confirming the MoE sweep
  checkpoints exist.

### Program (decided 2026-08-22): one question, three pillars

**The question** (apex; canonical text in [../research-hypothesis.md](../research-hypothesis.md)):
claimed training-history effects — critical periods, warm-start scars, non-stationarity
memory, recipe effects on elicitability — are confounded with regime-mismatched defaults and
uncontrolled elicitation. Build the measurement framework (calibrated elicitation,
replicated small-scale interventions, identifiability-aware comparisons) that separates
real path-dependence from measurement artifact, and use it to find where weight updates
beat tuned elicitation and how the balance shifts with scale. Its one-line readout is the
**capability delta vs. accessibility delta** decomposition.

**Pillars** (lenses on how to answer it; each project declares which it serves):

| Pillar | Role | Canonical text |
|---|---|---|
| **Measurement science at academic scale** (*how*) | Replicates, confidence intervals, calibrated instruments, tiny models as the program's Drosophila, the scale ladder as an identifiability-vs-scale measurement | [../topics/reference/evaluation-methodology-literature.md](../topics/reference/evaluation-methodology-literature.md), [tiny-scale-measurement.md](tiny-scale-measurement.md) §4 |
| **Non-stationarity accounting** (*mechanism*) | Every scar is a non-stationarity source (exogenous: schedule, data order, midtraining; endogenous: routing, the gradient-weighted self-curriculum); every fix is a stabilizer; account for how much each injects and what each suppresses | [../topics/reference/nonstationarity-accounting.md](../topics/reference/nonstationarity-accounting.md) |
| **Data measurement → training dynamics** (*independent variable*) | Featurize data by what it does to training, not by endpoint scores; DataDecide is the first instrument-validation suite | [recipe-featurization.md](recipe-featurization.md) §4 (first entry) |

The four framings previously listed here as candidates (data measurement → dynamics;
measurement science at academic scale; non-stationarity accounting; elicitation-controlled
evaluation) are not discarded: the fourth is the apex question and the other three are the
pillars above.

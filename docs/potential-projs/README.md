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
literature review (plan and process) lives in [../litreview/](../litreview/). Ideas that do not yet
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
  [../topics/nonstationarity-accounting.md](../topics/nonstationarity-accounting.md).

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

### Candidate program framings (recorded, not chosen)

Three names have been proposed for the program as a whole. Each is a different emphasis over
the same projects.

| Framing | One-line pitch | Naturally wraps |
|---|---|---|
| **Data measurement → training dynamics** | Featurize data by what it does to training (schedule sensitivity, emergence timing, noise, forgetting), not by endpoint scores; DataDecide is the first instrument-validation study | REC, FUNC, ANN, TOK, PART |
| **Measurement science of LM training at academic scale** | Cheapness converts into replicates, factorial designs, and confidence intervals; the measurement problem at 10–150M is the research question; tiny models as the program's Drosophila | TRJ, IRT, TINY, ANN, MSUITE, FUNC |
| **Non-stationarity accounting** | Every thread is a non-stationarity thread (exogenous: schedule, data order, midtraining; endogenous: routing, gradient-weighted self-curriculum); account for how much each source injects and what each stabilizer suppresses | ANN, REC-9/10, TOK, MOVE, FUNC |

Full text for each: `recipe-featurization.md` §4 (first),
`tiny-scale-measurement.md` §4 (second),
`../topics/nonstationarity-accounting.md` (third).

### 2026-08-18 — the unifying question behind the plasticity and scaling-law threads (from the Research Trajectory page)

- "Both are doing the same epistemic move: treating the loss curve (or statistics derivable
  during training) as a *measurable signal that predicts a latent capability you actually
  care about*… If you want a 'loss-curve features → success' framing that unifies them, the
  shared question is: what low-dimensional summary of training dynamics is sufficient to
  forecast a capability? The plasticity answer so far is 'no single statistic — curvature
  comes closest' (Lyle), while the pretraining answer is 'a surprisingly simple functional of
  the LR schedule' (multi-power law) plus a sigmoid/exponential link to accuracy."
- Core differences worth keeping in view: target of prediction (a property of the *learner*
  — future adaptability — vs. a property of the *outcome*); methodological flavor
  (mechanistic/causal interventions with many cheap seeds vs. phenomenological fits where
  prediction substitutes for experimentation). Reference topics:
  [../topics/plasticity.md](../topics/plasticity.md),
  [../topics/loss-curve-forecasting.md](../topics/loss-curve-forecasting.md).

### 2026-08-18 — the original hypothesis and where the field has gone (from the Research Trajectory page)

- "The field is converging on exactly your framing — final pretraining loss is an
  insufficient statistic for downstream success, and the open question is what *else* about
  the training trajectory (data order, late-window exposure, the 'state of the learner' in
  plasticity terms) predicts post-training outcomes." Tension to keep in view: Shen et al.
  (arXiv 2607.16097) find "lower pretraining loss strongly predicts higher post-RL pass@1 at
  fixed RL compute… though not all of it."
- DataDecide's own headline (Magnusson et al., ICML 2025): ranking at 150M predicts the best
  1B recipe ~80% of the time; continuous likelihood proxies make benchmarks >80% predictable
  at 0.01% of the compute — "the 'at low scale, accuracy is noise, so find a smoother
  observable' move" that the measurement projects extend. Reference topic:
  [../topics/pretraining-to-posttraining.md](../topics/pretraining-to-posttraining.md).

### 2026-08-18 — the seeds × families × iteration tension, and the asymmetric design (from the Research Trajectory page)

- "The noise floor of your measurements scales with the number of seeds you can afford,
  while the generality of any finding scales with the number of model families you test —
  and both multiply against slow iteration… a clean single-family result might just be
  another family artifact, so even a successful sweep has uncertain external validity."
- "The measurement-and-proxy angle is both the least blocked and, right now, probably the
  most needed." Validating a proxy "needs far fewer runs than detecting an intervention
  effect, because you're fitting a correlation across existing variation rather than
  powering a comparison."
- Asymmetric design: "Full sweep with seeds only where it's cheap (small models, continuous
  metrics, easy tasks). Then spend the expensive budget on two or three confirmation runs
  testing a *ranking* the cheap tier predicted — a much lower-powered, therefore
  affordable, test than estimating effect sizes." Staging topic:
  [../topics/posttraining-experiment-design.md](../topics/posttraining-experiment-design.md).
- Same-date restatement of the recurring question, from the ICL discussion: "Your whole
  trajectory — loss curves, proxy metrics, elicitation — keeps circling one question: *what
  cheap continuous observable reveals latent capability?*" Staging topic:
  [../topics/icl-as-posttraining.md](../topics/icl-as-posttraining.md).

### 2026-08-18 — matched-loss comparisons need two controls; define the invariant measurement, not the invariant model (from the Research Trajectory page)

- "Matched-loss pairs have a hidden confound: equal loss at different token counts vs. equal
  tokens at different loss are different controls, and you'll want both, since 'recipe A
  reaches this loss faster' and 'recipe A has better ICL at this loss' are separable
  claims." Applies to every matched-loss comparison in the portfolio (TRJ-3, IRT-3, GEO pair
  selection, TOK-4, the sweep reread as data in PART).
- Third control, from the loss-basins discussion: matched loss can hide different mixes of
  along-river progress (durable) and distance-from-river (transient), and different basins
  altogether — "the basin question is therefore the question of *when your comparisons are
  well-defined*." Log interpolation barriers between compared checkpoints and report
  conditional on barrier height (the landscape-geometry project's core).
- Cross-scale / cross-modality design principle: "Define the invariant measurement, not the
  invariant model. The bridge between tiers is a protocol… The deliverable of tier 1 is
  'here is a low-variance elicitability metric and here's how many seeds it needs.'" A
  confirmation test at the expensive tier "needs far less compute than an exploration
  sweep." Staging topic: [../topics/icl-as-posttraining.md](../topics/icl-as-posttraining.md).

### 2026-08-18 — a retrospective narrative: the thesis question, eight years early (from the Research Trajectory page)

- Ash & Adams (NeurIPS 2020): warm-started models "led to worse generalization than random
  re-initialization, even though training losses were similar… That 'similar training loss,
  different downstream outcome' clause is literally your thesis question, eight years early
  — which is what makes it the perfect opening chapter."
- Chapter plan: "**Chapter 1:** Why did the field's oldest 'pretraining hurt downstream
  performance at matched loss' result happen, and which modern practice fixed it? **Future
  chapter:** Matched-loss ICL experiments as the same question at the next scale." Staging
  topic: [../topics/warmstarting-decomposition.md](../topics/warmstarting-decomposition.md).
- From the ICL-analysis discussion: "the field's best ICL measurement is a loss curve over
  context position, fit with power laws, complete with emergence thresholds and proxy-metric
  debates — your pretraining-scale question recapitulated inside a single forward pass."
  Reference topic: [../topics/icl-literature.md](../topics/icl-literature.md).

### 2026-08-18 — the deepest root: critical learning periods (from the Research Trajectory page)

- Achille, Rovere & Soatto's critical-period / Information-Plasticity line "is arguably the
  deepest root of your whole tree": it is the origin of the plasticity literature (Fisher
  trace as diagnostic; Lyle's curvature finding "nearly a rediscovery in different
  coordinates"), it restates the warm-starting gap (starting past the Fisher peak), it
  grounds the basin story (the critical period as "the window before the run commits to a
  basin/valley"), it supplies the featurization tool (Task2Vec is the same Fisher
  formalism), and the LLM-scale data-placement results are "critical-period phenomenology at
  scale, mostly published without the connection drawn."
- Closing loop for the retrospective narrative: "the blurred-kitten paper you started from
  turns out to contain the earliest version of your plasticity thread, your featurization
  thread, and your basin thread all at once." Reference topic:
  [../topics/critical-periods.md](../topics/critical-periods.md); staged study:
  [../topics/critical-period-timing-study.md](../topics/critical-period-timing-study.md).
- The retrospective's experimental spine and one-sentence pitch (same date): one
  early-window intervention grid — intervene(what, when, how long) × recipe × measure — "in
  which each classic paper is a single cell," with modern fixes as "period-reopening
  interventions." "Two foundational anomalies from 2019–2020 showed that *when* a network
  learns something determines *what* it can become, the modern field keeps rediscovering
  this at LLM scale without the connection, and the thesis builds the measurement framework
  that reconciles them." Staging topics:
  [../topics/warmstarting-decomposition.md](../topics/warmstarting-decomposition.md),
  [../topics/critical-period-timing-study.md](../topics/critical-period-timing-study.md).


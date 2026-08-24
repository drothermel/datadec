# Annealed readouts — on existing DataDecide checkpoints

**Program pillars served:** how (schedule-neutralized instruments), mechanism (the LR schedule as exogenous non-stationarity). (Program: `README.md` → Program.)

**Working title:** *How much of DataDecide is a schedule artifact? Retrofitting annealed
evaluations onto cosine-trained checkpoint suites.*

**One-line pitch.** DataDecide's checkpoints are all evaluated mid-cosine-schedule, so every
reported number conflates durable progress with a schedule-dependent "distance up the wall"
term. We measure that term with short decay branches, test two eval-cost proxies for it
(checkpoint merging, multi-power-law correction), and report which of DataDecide's ~300 pairwise
recipe decisions survive annealing.

IDs: ANN-1–ANN-6 (ANN-1–ANN-6 in the LR-schedule synthesis, with the former F1/F2 folded in as ANN-5/ANN-6).

---

## 1. What the project involves

### Core experiment

1. **Fit the multi-power law per recipe (ANN-5). [T0]** Using DataDecide's logged training curves,
   fit the MPL (power law in cumulative LR plus decay-drop terms) for each recipe × size × seed.
   Output: fitted parameters and a predicted "loss if decayed now" at every checkpoint (ANN-2).
   No GPU. The fitted decay term also selects the branch grid: branch where the predicted
   decay gain is largest or most recipe-divergent. Loss curves and the LR schedule are already
   in `processed/scaling-law/checkpoint-losses.parquet` and `lr_at_step` / `cumulative_lr` on
   every checkpoint row. The decay phase itself makes progress along the river, so the
   correction is not a pure reveal; branch length is a parameter to sweep analytically.
2. **Run short decay branches from a grid of existing checkpoints (ANN-3). [T2]** For a chosen set of
   recipes × checkpoint steps × (1–2) sizes × 3 seeds, resume training with a fresh decay
   (linear-to-zero or 1-sqrt; ~10% of elapsed tokens as the default length) on the recipe's
   own data, then run the full DataDecide eval suite on the endpoint. This is the ground-truth
   annealed readout.
3. **Compute checkpoint merges on the same grid (ANN-1). [T1+]** Sliding-window weighted average of the
   checkpoints preceding each branch point, with weights derived from an emulated decay curve
   (WSM). Eval with the same suite.
4. **Validate the proxies (ANN-4). [analysis]** Compare ANN-1 (merge) and ANN-2 (MPL) against ANN-3 (branch) on the
   grid: loss agreement, task-metric agreement, and — most importantly — agreement on pairwise
   recipe decisions.
5. **Decision-flip analysis (ANN-6). [T0 under ANN-2; analysis otherwise]** Recompute DataDecide's
   pairwise recipe decisions on the
   annealed values and report which flip, at which steps, on which tasks. This is the headline
   figure. Compare flips under each proxy, and test DataDecide's published claim that
   intermediate-checkpoint decisions already match final ones. Quantify where rankings are
   preserved under correction and where they flip; separate "levels distorted, ranks
   preserved" from "ranks distorted."

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short
decay branches.

### Optional directions

- **ANN-opt-1: Branch-length and branch-shape sweep.** At a fixed branch point, vary decay length
  (e.g. 2%, 5%, 10%, 20% of elapsed tokens) and shape (linear, 1-sqrt, cosine tail). Separates
  "reveal" (wall descent) from continued along-river progress and picks a canonical branch.
- **ANN-opt-2: Merge-window sensitivity (ANN-1 detail).** Vary window length and weight curve for
  the merge; characterise where merging breaks on cosine checkpoints (it is only validated on
  stable-phase runs).
- **ANN-opt-3: Post-training from annealed vs. raw checkpoints.** Re-run the earlier
  post-training protocol from ANN-3 endpoints and the matching raw checkpoints. Tests whether the
  previous "post-training did nothing" result was a wall artifact.
- **ANN-opt-4: Token-loss-trajectory taxonomy on raw checkpoints.** Rho-1-style
  classification of held-out tokens by loss trajectory across checkpoints. A cheap descriptive
  sub-result — the static version of per-token decay-responsiveness — that previews the
  causal token-level follow-on; no branches needed.
- **ANN-opt-5: Size scaling of the confound.** Repeat the core grid at a second model size to
  test whether the annealed-vs-raw gap and the flip rate shrink or grow with scale.
- **ANN-opt-6: Seed-noise floor.** Use the 3 seeds to put confidence intervals on every
  annealed-vs-raw difference; report the fraction of flips that exceed seed noise. This is less
  an option than a requirement for credibility, but it shapes how big the grid must be.
- **ANN-opt-7: Durable-movement operator.** Define durable movement as change
  that persists under the schedule-neutralizing transform: compare merged(t) vs. merged(t+k).
  Decomposes Signal-and-Noise "noise" into measurement noise, wall oscillation, and
  unresolved drift. Requires ANN-1 plus per-token or per-item comparison between checkpoints.

---

## 2. Doability and impact

### Overall doability: **high**

- Everything except ANN-3 is evals-only or curve fitting. ANN-3 is short continued-training runs at
  150M–300M, well within a small cluster budget.
- The paper is robust to outcome: "merging works on cosine checkpoints" is a methods
  contribution; "it doesn't, here is the ground truth" is still a useful negative result; and
  the decision-flip analysis is informative either way.
- Main risk is **noise**: at 150M with 3 seeds, the annealed-vs-raw differences on some tasks
  may sit inside seed variance. Mitigation: choose recipes with large known gaps, use
  continuous-likelihood metrics (which DataDecide already shows are the low-noise ones), and
  report flips against a seed-noise floor (ANN-opt-6).
- Secondary risk: the MPL may fit cosine curves poorly at small scale or need per-recipe
  tuning; this only weakens ANN-2, not the paper. The MPL was fit on runs with explicit
  schedules; fitting it to cosine runs is in its stated scope but its extrapolation to
  "hypothetical decay from here" on these specific runs is unvalidated. A held-out check
  (predict the final decayed loss from mid-run checkpoints) is mandatory. Verify the
  scaling-law checkpoint-loss table covers all recipes and seeds or only a subset.
- Merging is unproven on cosine mid-run checkpoints where LR varies inside the merge window,
  which is both the risk and the contribution. A clean negative result is also publishable
  but less exciting.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| Core (ANN-1–ANN-6) | **High** | Directly audits a widely used public suite; the decision-flip figure is self-contained and quotable; methods contribution if merging holds. |
| ANN-opt-1 branch sweep | Medium | Needed for a defensible canonical branch length; modest standalone interest but strengthens every claim. |
| ANN-opt-2 merge sensitivity | Medium | Makes the "retrofit annealed evals for free" claim precise; mostly a supporting figure. |
| ANN-opt-3 post-training | **High, higher variance** | A positive result ("post-training gains appear once you anneal") is the most compelling story available and closes the loop on the earlier project; a null is still reportable but harder to sell. Adds post-training infrastructure cost. |
| ANN-opt-4 token taxonomy | Low–Medium | Nice descriptive figure; its real value is de-risking the causal token-level follow-on. |
| ANN-opt-5 size scaling | Medium | Standard reviewer ask; doubles branch compute. |
| ANN-opt-6 seed-noise floor | Required | Not an option in practice — without it the flip analysis is not credible. |
| ANN-opt-7 durable movement | Medium-high | Conceptually strong; depends entirely on ANN-1 and on a token/item-level harness. |

**Recommended scope for a first workshop paper:** Core + ANN-opt-1 (small) + ANN-opt-6, with
ANN-opt-3 as a stretch if the branch grid finishes early. Defer ANN-opt-5 to a reviewer-response
or follow-up.

**Fallback if branches do not happen.** The T0 half (ANN-5 + ANN-2 + ANN-6 on corrected loss) is a
strong *section* in a trajectory or IRT paper rather than a paper. As a standalone workshop
paper this needs ANN-1 validated: "annealed evals for DataDecide without training" is a clear
methods contribution with the flip analysis as the motivating analysis.

---

## 3. Infrastructure build sequence

Ordered so each step is usable on its own and later steps reuse earlier ones.

1. **Curve ingestion + MPL fitting (ANN-5/ANN-2).** Load DataDecide training curves per recipe/size/
   seed; fit the MPL; emit predicted decay gain per checkpoint. Pure Python, no GPU. Output
   also drives branch-grid selection. Include a held-out validation predicting final decayed
   loss from truncated curves, and a hypothetical-decay prediction at each checkpoint for a
   sweep of decay lengths; emit a coverage report across recipes × scales × seeds.
   *Deliverable: per-recipe parameter table + predicted annealed-loss curves.*
2. **Checkpoint + eval harness.** Load any (recipe, size, seed, step) DataDecide checkpoint;
   run the DataDecide task suite and perplexity evals; store results keyed by that tuple plus
   a `variant` field (`raw`, `merged:<cfg>`, `branch:<cfg>`), producing the same table schema
   as the processed OLMES tables so merged-model results slot into existing accessors. Builds
   on this repo's existing data tooling. *Everything downstream writes into this store.*
3. **Fixed held-out token set + per-token loss logging.** Choose a held-out token set once and freeze it: fixed, versioned token
   sequences with a manifest; stratified across domains and across the DataDecide leaf
   corpora; sized so that each domain × entropy-bucket cell has enough tokens to estimate
   mean per-token loss drop within a set tolerance, while keeping one forward pass per
   checkpoint cheap. Per-token loss on it is a standard output of the eval harness for every
   checkpoint variant (raw checkpoints, merged checkpoints, branch starts and endpoints),
   stored as compact arrays keyed by (checkpoint variant, held-out-set version). Branch
   endpoints also save their weights. Cheap to add now; expensive to retrofit later because
   it would mean re-running branches. *An identical spec appears in Annealed readouts, WSD retrain suite, Token-level
   movement, MoE movement, MoE recipe suite, and Functional featurization; keep them in sync.*
   Enables ANN-opt-4 and per-token analyses of branch endpoints.
4. **Checkpoint-merging tool (ANN-1).** Sliding-window weighted averaging with configurable
   window and weight curve; expert-agnostic (dense models only); outputs a checkpoint that the
   eval harness treats as a variant. Evals-only, so it can run on the full DataDecide grid
   immediately.
5. **Decay-branch runner (ANN-3).** Resume a checkpoint with configurable decay shape/length on
   the recipe's own data stream; log curves and per-token losses; hand the endpoint to the eval
   harness. Parameterise shape and length from day one (ANN-opt-1).
6. **Analysis layer (ANN-4/ANN-6).** Pairwise-decision recomputation from the results store;
   proxy-vs-branch agreement metrics; seed-noise confidence intervals; flip tables and figures.
7. *(Optional, ANN-opt-3)* **Post-training harness hookup.** Point the existing post-training
   protocol at `branch:*` variants and record outputs in the same store.

Steps 1, 2, and 4 can proceed in parallel; 3 should be finished before 5 starts; 6 is
incremental as data arrives. Step 1 plus the T0 half of step 6 (flip analysis on corrected
loss) are cheap and should be done regardless of whether the branch/merge half is pursued;
they also tell you whether that half is worth it.

---

## 4. External assessments

Dated, attributed notes from external review conversations, recorded for consolidation — not
decisions. Only notes about this project are kept here.

### 2026-08-22 — midtraining literature pointer

Danielle's SciSpace review of pretraining/midtraining toward a target suite (record in
`../topics/reference/targeted-pretraining-midtraining-literature.md`) surfaced the
midtraining ↔ RL interplay study (2512.07783) and little else LM-specific; the
annealing-data canon this project's decay branches relate to (MiniCPM, Llama 3 annealing,
OLMo 2 / Dolmino, Blakeney et al.) remains in
`../topics/reference/schedules-and-annealing-literature.md`. Relevance here: ANN-3 decay
branches with a *changed* data mixture are the controlled version of "midtraining to
target a suite"; the interplay paper's fixed-compute comparison is a framing precedent.

### 2026-08-22 — the intermediate-vs-final compute-matched claim is unassessed, not refuted

*Provenance caveat: the reproduction numbers cited here come from agent-written verification code that Danielle has not yet personally read, debugged, run, or analyzed; treat them as flags for where to look first, not as findings (her statement in `../topics/reference/datadecide-data-pipeline.md`).*

Danielle's reproduction of the DataDecide paper (record in
`../topics/reference/datadecide-data-pipeline.md`) classified "an intermediate checkpoint
predicts rankings as accurately as a final checkpoint with equal compute" as
`not_reproduced`, but the verifier required exact floating-point equality of compute and
found zero matched pairs; the `-1.0` "minimum difference" was an empty-set sentinel. The
validator itself recommended `not_assessable` pending a predeclared approximate-matching
or interpolation rule (`src/datadec/paper/verifiers/single_scale.py` ~line 1341 on
`main`). Danielle: "no human would try to compute match with floats, let alone integers,
this would be a bucketed comparison" — a verifier bug on our side, not a finding about
the paper; the response withdrew its "most important finding in all three batches"
framing. What survives for this project: (1) the fixed matcher is shared infrastructure
— matched-loss pairing (`trajectory-statistics.md` TRJ-3) and compute-matched pairing are
the same tool with a different matching variable; design decisions: tolerance in
log-compute space (a ratio, not an absolute difference), predeclared, match distances
reported; or interpolation along each run's compute axis, which sidesteps bucketing and
may be cleaner given full trajectories; sweep the tolerance ("holds at 5% but not 1%"
would itself say how sharp the equivalence is). (2) The analysis enters ANN-4 as
"reproduce on cleaned data under a defensible matching rule, then re-examine under
annealing correction" — the normal and arguably stronger two-act shape. (3) Methods
lesson for the validation framework: a predicate can be formally frozen but physically
degenerate (exact equality on a continuous quantity), distinct from threshold
sensitivity; add a "predicate liveness" guard — verify the comparison set is non-empty
and report its size with every result.

### 2026-08-21 — two "top-N by workshop-paper likelihood × speed" lists

- **Deliberately left out of a top-3 list.** "It has the highest ceiling of the
  eval-adjacent work, but its paper-worthy version requires the branch runs, an eval harness,
  and a checkpoint loader — real infrastructure and real wall-clock. It's the right second
  wave project, and [trajectory drift/diffusion and IRT] results will tell you which recipes
  and steps are most schedule-sensitive, making the branch grid smaller and better targeted."
- **Ranked #4 in a top-5 list (T0 half only).** "MPL correction + 'which rankings flip' is
  cheap, and 'how much should anyone trust intermediate-checkpoint rankings' is the question
  every DataDecide user has. Your own doc rightly says [the T0 half] alone is a section, not a
  paper — so either pair it with [the trajectory drift/diffusion paper] (they share the
  trajectory accessor and the noise floor) or treat it as the mandatory prelude to the branch
  grid."
- **Grid targeting.** Use the T0 results from the trajectory and IRT projects
  to choose schedule-sensitive recipes and steps before branching, in addition to the MPL-fit
  selection in ANN-5.

- **Nulls.** The merging-on-cosine result (ANN-1) and the flip analysis (ANN-6)
  are listed among the "shakier" either-outcome claims: "distinguish nulls that are
  informative from nulls that are merely reportable."
- **Competition.** The Signal-and-Noise / DataDecide authors "are the obvious
  people to do [the T0 reanalysis] themselves"; ship the T0 half fast and consider them as
  collaborators on the branch grid, "where your infrastructure investment is the moat."

### 2026-08-21 — on short-branch landscape probes

- The decay branch (ANN-3) read as "a wall-height meter": "'branch + decay + measure the loss
  drop' is established; the drop is your height-above-river statistic. What's not
  established: doing it on cosine mid-run checkpoints, and treating the per-token profile of
  the drop as the statistic rather than the scalar." (Unverified claim: Wen et al. validate
  the river-valley theory by branching a constant-LR run and interpolating; their WSD-S
  variant resumes from decayed checkpoints.)
- Proposed as one of four probes in a "checkpoint tomography" battery; see
  `docs/topics/staging/checkpoint-tomography.md`.

### 2026-08-21 — positions in three ranked lists (full lists in `docs/portfolio-rankings.md`)

- **6–12-month flagship list: Tier 1, #1** as the practical-hook and method halves of "the
  unified causal program" (with token-level movement and the determinism features): "the
  decision-flip audit… Everyone using the suite cares"; "validated checkpoint merging on
  cosine checkpoints as cheap annealed evals, with real decay branches as ground truth."
  "Degrades gracefully: if the token-level mechanism is noisy at 150M, the audit +
  merging-validation half is still a solid paper." Advice: "budget one scale step up (~1B)
  for the core grid." Barriers-on-annealed-variants "folded in."
- **Workshop-sized list: #4** (T0 half only): "T0, cheap, and answers the question every
  DataDecide user has… section-shaped [without branches]… the MPL's extrapolation… is
  unvalidated — the mandatory held-out check could partially fail and cap the claims. Fast,
  but with a real methodological gate."
- **Full-conference list: #5**, "How Much of Your Checkpoint Suite Is Schedule Artifact?" —
  "*Speed:* medium… unusually outcome-robust (merging works → methods contribution; fails →
  ground-truth audit still stands)… **Expected impact: high**… **Ceiling: high**, especially
  if a meaningful fraction of published decisions flip." Flagged as a scoop race (public
  data, Ai2 authors adjacent).

### 2026-08-21 — position in the "four main-conference projects from two workshop subs" list (full list in `docs/portfolio-rankings.md`)

- **P3 — Auditing the Schedule: Annealed Readouts for Cosine-Trained Suites.** Sub A: ANN-5
  + ANN-2 + ANN-6 (the T0 half, with the held-out validation). Sub B: ANN-1 merging
  validated against a small pilot grid of ANN-3 branches. Main paper: the full project
  including ANN-opt-1 and "canonical annealed re-release of the suite's decisions."
  "**Speed: third.**… Outcome-robust at every level. **Scoop risk: highest of the four** —
  it's the most obvious-in-retrospect question about DataDecide, WSM/merging papers are
  circling it, and the Ai2 authors could run it internally with better access. Sub A should
  ship early partly as a flag-plant. **Expected impact: high. Ceiling: high.**"

### 2026-08-18 — prior art for ANN-5 and a caveat for ANN-6 (from the Research Trajectory page)

- The multi-power law is Kairong Luo et al., arXiv 2503.12811 (ICLR 2025): "a power law on
  the sum of learning rates plus extra power-law terms for the decay-induced loss drop;
  fitted on a few runs, it extrapolates to unseen schedules and even discovers a schedule
  beating cosine (resembling WSD)."
- Caveat for flips measured on task metrics: "hard accuracy metrics can look emergent,
  showing no progress above chance until the loss crosses a threshold, which is where the
  loss-to-accuracy mapping gets fragile." Loss→accuracy links in use: exponential (Gadre et
  al. 2024), FLP two-stage (Chen et al., arXiv 2410.08527), model ladders (Bhagia et al.).
  See `docs/topics/reference/loss-curve-forecasting.md`.

### 2026-08-18 — prior art and a hindsight reading for ANN-opt-3 (from the Research Trajectory page)

- The earlier "post-training did nothing" result, read in hindsight: tested at scales where
  "(a) post-training gains are largely elicitation of capabilities your models didn't yet
  have, (b) the Qwen confound was silently inflating the literature's baseline
  expectations, and (c) benchmark noise swamps effect sizes without multi-seed evaluation."
  Any rerun from annealed endpoints needs multi-seed evaluation and a capability-floor
  check before the "wall artifact" reading can be tested.
- Closest published design: *Similar Models Learn Differently: Final-Window Pretraining
  Shapes Post-Training Beyond SFT* (arXiv 2607.25063) — "models that look similar after SFT
  diverge under identical post-training depending on late-pretraining data interventions."
  Full list in `docs/topics/reference/pretraining-to-posttraining.md`.

### 2026-08-18 — a late-window cross-family design adjacent to ANN-opt-3 (from the Research Trajectory page)

- "Take OLMo, Pythia, SmolLM, Llama, and Qwen checkpoints and apply controlled *late-window*
  continued pretraining — same intervention, same tokens, different lineages. The Final
  Window paper's claim (*Similar Models Learn Differently*) is that this window
  disproportionately shapes post-training behavior, which if true means most of the
  family-effect question is testable at annealing cost rather than pretraining cost. If the
  claim is false at your scales, that's also a finding." The decay-branch runner is the same
  instrument with the schedule (rather than data) as the late-window intervention. Full
  discussion in `docs/potential-projs/movement-microscope.md`.

### 2026-08-18 — the river-valley reading of the MPL decay term (from the Research Trajectory page)

- "The multi-power law paper itself flags this. Its authors note the river-valley conjecture
  as the landscape framework their schedule-dependent loss terms are implicitly modeling.
  The 'decay-induced loss drop' term is, in river-valley language, descending from
  oscillating on the walls down to the river… So 'loss' during training conflates
  along-river progress (durable) with distance-from-river (transient) — exactly the two
  components your matched-loss design needs to distinguish. Two recipes matched on loss
  could be matched on totally different mixes of the two." Wen et al. (arXiv 2410.05192) is
  the canonical statement; their interpolation signature is "the closest thing to a 'river
  test.'" Paper list in `docs/topics/reference/landscape-literature.md`.

### 2026-08-18 — origin of this project (from the Research Trajectory page)

**Danielle-flagged project seeds** (the `→` notes on the Notion toggle):
1. "Predict performance differences from dataset features."
2. "Does merging-as-annealing-proxy work on cosine mid-run checkpoints rather than just
   stable-phase ones?"
3. "Does a dataset's 'determinism profile' predict landscape geometry?"

Seed 2 is ANN-1's open question; seed 3 is the data link that the flagship ranking folds
into this project. Question posed: DataDecide's models would be more useful trained with a
WSD schedule, because then annealing tests would be comparable along the trajectory; their
evals show un-annealed or partially annealed performance, which skews all results including
post-training. What are the options — small extensions of training from each checkpoint, or
redoing some pretraining with WSD?

- The confound, as first stated: "every intermediate checkpoint sits mid-schedule with high
  residual LR — in river-valley terms, evals on those checkpoints measure 'position along
  river + current distance up the wall'… Post-training from such checkpoints inherits the
  confound (you're fine-tuning from a point high on the wall)."
- The two cheap fixes on existing checkpoints (now ANN-1 merging and ANN-5/ANN-2 MPL):
  "if [merging] works even approximately you can retrofit 'annealed' evals onto all of
  DataDecide for the cost of evals"; the MPL "quantifies how much each recipe's apparent
  ranking is schedule artifact."
- Two caveats that shaped ANN-6 and ANN-opt-1: intermediate-checkpoint *decisions* may
  already match final ones ("the confound may partially cancel for rankings even while
  distorting levels and post-training"), and "the decay phase itself makes progress along
  the river — so annealed evals aren't a pure 'reveal' either; branch length becomes a
  parameter to control."
- Prior art (Hägele 2024; MiniCPM; Llama 3 annealing data valuation; Blakeney; WSM; Nemotron
  3) in `docs/topics/reference/schedules-and-annealing-literature.md`: "'annealing branches as the
  correct eval' is validated practice — but no *open, multi-recipe suite* has it."

### 2026-08-18 — the decay branch as an "anti-grokking instrument" (from the Research Trajectory page)

- "Grokking plateaus reinterpreted as travel along the river that loss can't see, which
  unifies 'hidden progress' with your token-bucket/decay-branch machinery: the decay branch
  is a probe that *reveals* accumulated-but-hidden river progress, i.e., an operationalized
  anti-grokking instrument." Matched-loss pairs are "a *necessary but provably insufficient*
  control" (two checkpoints at equal loss can differ in hidden circuit maturity). See
  `docs/topics/reference/grokking-and-hidden-progress.md`.

### 2026-08-18 — origin of ANN-opt-7, the durable-movement operator (from the Research Trajectory page)

- "Compare KL(t, t+k) as a function of k (transient movement cancels; durable movement
  accumulates), and — the stronger instrument — apply checkpoint merging or a short decay
  branch at t and t+k and compare the *annealed* models: movement that survives annealing
  is river movement by construction. That gives you an operational definition the field
  currently lacks: **durable movement = change that persists under the
  schedule-neutralizing transform**, with the Signal-and-Noise 'noise' then decomposing
  into (measurement noise) + (wall oscillation) + (drift you're too coarse to see), each
  separately estimated… the durable-movement operator is the instrument your matched-loss
  comparisons have needed all along."

### 2026-08-18 — merging on MoE checkpoints needs expert matching first (from the Research Trajectory page)

- "Checkpoint merging (your annealing-proxy trick) needs expert matching first or it
  averages mismatched experts into mush" — ANN-1 is dense-only as specified; an MoE variant
  requires the expert-alignment step from MoE partitions (PART-4). See
  `docs/topics/reference/moe-literature.md`.

### 2026-08-22 — decay-phase gradient statistics as an instrument (from a second annealing-data report)

A browsing report on annealing data quality (record in
`../topics/reference/schedules-and-annealing-literature.md`, second entry; figures
unverified) attributes to MiniCPM (arXiv 2404.06395) a description of the decay phase:
weights move less than in the stable phase while loss falls faster, gradient norm
diminishes, and the cosine similarity between consecutive updates turns predominantly
positive — directed descent into a basin rather than exploration. If that holds, the
decay-branch runner can log the same three statistics per branch at DataDecide scale, which
gives a mechanism-level companion to the ranking-stability readouts: a branch whose updates
are not yet consistently aligned has not "read out" yet. Also the citation for the decay
shape: Hägele et al. 2405.18392 (1-sqrt cooldown vs. linear). Keep in sync with the WSD
suite's decay-shape note.

### 2026-08-22 — absorbed from the post-training experiment-design topic: ANN-opt-8

- **ANN-opt-8: Late-window cross-family intervention.** Take OLMo, Pythia, SmolLM, Llama,
  and Qwen checkpoints and apply controlled late-window continued pretraining — same
  intervention, same tokens, different lineages — with the decay-branch runner as the
  instrument. If the final window disproportionately shapes post-training behavior
  (*Similar Models Learn Differently*), most of the family-effect question is testable at
  annealing cost; if not at these scales, that is also a finding.
## 5. Related work and positioning

*Purpose: the paper-facing synthesis — the prior-art landscape, this project's
position in it, and what each closest neighbor lacks. Unlike §4 (a dated intake
log, which grows by appending new entries **above this section**), §5 is a
current-state statement: rewrite it as understanding changes. Positioning claims
are Danielle's to make; agent-supplied literature claims anywhere in this document
are unverified leads, not established facts.*

**Status: raw material assembled from repository records (2026-08-24); positioning not
yet written. The full inventory — every possibly-relevant item on record, with sources —
now lives in `related-work/annealed-readouts.md`; this section keeps only the
load-bearing core.**

- **Kairong Luo et al., multi-power law (arXiv 2503.12811, ICLR 2025)** — the ANN-5/ANN-2
  anchor: a power law on the sum of learning rates plus decay-drop terms, extrapolating
  to unseen schedules. Its extrapolation to "hypothetical decay from here" on cosine runs
  is unvalidated, which is why the held-out check is mandatory.
- **WSM (checkpoint merging as pseudo-annealing) and Nemotron 3's sliding-window
  merging** — the direct method behind ANN-1 and the evidence it is already used in
  production, both validated only on stable-phase runs. Danielle's seed 2 — does it hold
  on *cosine* mid-run checkpoints — is the contribution and the risk.
- **Hägele et al. (arXiv 2405.18392) and MiniCPM (arXiv 2404.06395)** — the branch
  methodology ANN-3 executes: (1-sqrt) vs. linear cooldown as the shape citation, ~10%
  decay as the default length, and MiniCPM's decay-phase gradient statistics as a
  mechanism-level companion readout per branch.
- **Wen et al. (arXiv 2410.05192)** — the river-valley reading that makes the branch a
  wall-height meter; their interpolation signature is the nearest thing to a river test,
  and they validate by branching a constant-LR run and interpolating.
- **Heineman et al., *Signal and Noise*** (NeurIPS 2025 per the record; 2508.13144 per
  the ledger, a Claude-added row) — the noise term ANN-opt-7's durable-movement operator
  decomposes, and the seed-noise floor ANN-opt-6 tests flips against.
- **DataDecide (arXiv 2504.11393)** — the suite under audit and the source of the
  intermediate-vs-final compute-matched claim ANN-6 re-tests; the repo's own reproduction
  of that claim is `not_assessable`, not refuted (verifier bug).
- **Loss→accuracy links the flip analysis rides on:** Gadre et al. 2024 (exponential
  link), FLP (2410.08527), model ladders (2412.04403) — with the recorded caveat that
  hard accuracy metrics can look emergent and the mapping is fragile at threshold, and
  Nakkiran et al. double descent as the non-monotonicity boundary condition.
- **Similar Models Learn Differently (arXiv 2607.25063), Echo Chamber (arXiv 2504.07912),
  and the "post-training did nothing" cluster (A Sober Look 2504.07086; Spurious Rewards
  2506.10947; Yue et al. 2504.13837)** — the flank for ANN-opt-3/ANN-opt-8 and the
  hindsight reading of the earlier negative result.
- **Grokking (Power et al.) and progress measures (Nanda et al.)** — the branch as an
  anti-grokking instrument, and the argument that matched-loss pairs are necessary but
  provably insufficient.

All identifiers are agent-supplied and unverified; ANN-1 is dense-only because merging
MoE checkpoints needs expert matching first. The 2026-08-22 compute-matched entry
carries Danielle's caveat that its numbers come from agent-written verification code she
has not read or run.

Full inventory: `related-work/annealed-readouts.md`. Main accumulators:
`../topics/reference/schedules-and-annealing-literature.md`,
`../topics/reference/loss-curve-forecasting.md`,
`../topics/reference/landscape-literature.md`,
`../topics/reference/pretraining-to-posttraining.md`,
`../topics/reference/grokking-and-hidden-progress.md`,
`../topics/staging/checkpoint-tomography.md`,
`../litreview/citation-verification-ledger.md`.

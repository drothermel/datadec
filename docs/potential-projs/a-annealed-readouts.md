# Project A — Annealed readouts on existing DataDecide checkpoints

**Working title:** *How much of DataDecide is a schedule artifact? Retrofitting annealed
evaluations onto cosine-trained checkpoint suites.*

**One-line pitch.** DataDecide's checkpoints are all evaluated mid-cosine-schedule, so every
reported number conflates durable progress with a schedule-dependent "distance up the wall"
term. We measure that term with short decay branches, test two eval-cost proxies for it
(checkpoint merging, multi-power-law correction), and report which of DataDecide's ~300 pairwise
recipe decisions survive annealing.

Inventory IDs: A1–A6 (with the former F1/F2 folded in as A5/A6), optional B3 and D5.

---

## 1. What the project involves

### Core experiment

1. **Fit the multi-power law per recipe (A5). [T0]** Using DataDecide's logged training curves,
   fit the MPL (power law in cumulative LR plus decay-drop terms) for each recipe × size × seed.
   Output: fitted parameters and a predicted "loss if decayed now" at every checkpoint (A2).
   No GPU. The fitted decay term also selects the branch grid: branch where the predicted
   decay gain is largest or most recipe-divergent. Loss curves and the LR schedule are already
   in `processed/scaling-law/checkpoint-losses.parquet` and `lr_at_step` / `cumulative_lr` on
   every checkpoint row. The decay phase itself makes progress along the river, so the
   correction is not a pure reveal; branch length is a parameter to sweep analytically.
2. **Run short decay branches from a grid of existing checkpoints (A3). [T2]** For a chosen set of
   recipes × checkpoint steps × (1–2) sizes × 3 seeds, resume training with a fresh decay
   (linear-to-zero or 1-sqrt; ~10% of elapsed tokens as the default length) on the recipe's
   own data, then run the full DataDecide eval suite on the endpoint. This is the ground-truth
   annealed readout.
3. **Compute checkpoint merges on the same grid (A1). [T1+]** Sliding-window weighted average of the
   checkpoints preceding each branch point, with weights derived from an emulated decay curve
   (WSM). Eval with the same suite.
4. **Validate the proxies (A4). [analysis]** Compare A1 (merge) and A2 (MPL) against A3 (branch) on the
   grid: loss agreement, task-metric agreement, and — most importantly — agreement on pairwise
   recipe decisions.
5. **Decision-flip analysis (A6). [T0 under A2; analysis otherwise]** Recompute DataDecide's
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

- **A-opt-1: Branch-length and branch-shape sweep.** At a fixed branch point, vary decay length
  (e.g. 2%, 5%, 10%, 20% of elapsed tokens) and shape (linear, 1-sqrt, cosine tail). Separates
  "reveal" (wall descent) from continued along-river progress and picks a canonical branch.
- **A-opt-2: Merge-window sensitivity (A1 detail).** Vary window length and weight curve for
  the merge; characterise where merging breaks on cosine checkpoints (it is only validated on
  stable-phase runs).
- **A-opt-3: Post-training from annealed vs. raw checkpoints.** Re-run the earlier
  post-training protocol from A3 endpoints and the matching raw checkpoints. Tests whether the
  previous "post-training did nothing" result was a wall artifact.
- **A-opt-4: Token-loss-trajectory taxonomy on raw checkpoints (D5).** Rho-1-style
  classification of held-out tokens by loss trajectory across checkpoints. A cheap descriptive
  sub-result — the static version of per-token decay-responsiveness — that previews the
  causal token-level follow-on; no branches needed.
- **A-opt-5: Size scaling of the confound.** Repeat the core grid at a second model size to
  test whether the annealed-vs-raw gap and the flip rate shrink or grow with scale.
- **A-opt-6: Seed-noise floor.** Use the 3 seeds to put confidence intervals on every
  annealed-vs-raw difference; report the fraction of flips that exceed seed noise. This is less
  an option than a requirement for credibility, but it shapes how big the grid must be.
- **A-opt-7: Durable-movement operator.** Define durable movement as change
  that persists under the schedule-neutralizing transform: compare merged(t) vs. merged(t+k).
  Decomposes Signal-and-Noise "noise" into measurement noise, wall oscillation, and
  unresolved drift. Requires A1 plus per-token or per-item comparison between checkpoints.

---

## 2. Doability and impact

### Overall doability: **high**

- Everything except A3 is evals-only or curve fitting. A3 is short continued-training runs at
  150M–300M, well within a small cluster budget.
- The paper is robust to outcome: "merging works on cosine checkpoints" is a methods
  contribution; "it doesn't, here is the ground truth" is still a useful negative result; and
  the decision-flip analysis is informative either way.
- Main risk is **noise**: at 150M with 3 seeds, the annealed-vs-raw differences on some tasks
  may sit inside seed variance. Mitigation: choose recipes with large known gaps, use
  continuous-likelihood metrics (which DataDecide already shows are the low-noise ones), and
  report flips against a seed-noise floor (A-opt-6).
- Secondary risk: the MPL may fit cosine curves poorly at small scale or need per-recipe
  tuning; this only weakens A2, not the paper. The MPL was fit on runs with explicit
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
| Core (A1–A6) | **High** | Directly audits a widely used public suite; the decision-flip figure is self-contained and quotable; methods contribution if merging holds. |
| A-opt-1 branch sweep | Medium | Needed for a defensible canonical branch length; modest standalone interest but strengthens every claim. |
| A-opt-2 merge sensitivity | Medium | Makes the "retrofit annealed evals for free" claim precise; mostly a supporting figure. |
| A-opt-3 post-training | **High, higher variance** | A positive result ("post-training gains appear once you anneal") is the most compelling story available and closes the loop on the earlier project; a null is still reportable but harder to sell. Adds post-training infrastructure cost. |
| A-opt-4 token taxonomy (D5) | Low–Medium | Nice descriptive figure; its real value is de-risking the causal token-level follow-on. |
| A-opt-5 size scaling | Medium | Standard reviewer ask; doubles branch compute. |
| A-opt-6 seed-noise floor | Required | Not an option in practice — without it the flip analysis is not credible. |
| A-opt-7 durable movement | Medium-high | Conceptually strong; depends entirely on A1 and on a token/item-level harness. |

**Recommended scope for a first workshop paper:** Core + A-opt-1 (small) + A-opt-6, with
A-opt-3 as a stretch if the branch grid finishes early. Defer A-opt-5 to a reviewer-response
or follow-up.

**Fallback if branches do not happen.** The T0 half (A5 + A2 + A6 on corrected loss) is a
strong *section* in a trajectory or IRT paper rather than a paper. As a standalone workshop
paper this needs A1 validated: "annealed evals for DataDecide without training" is a clear
methods contribution with the flip analysis as the motivating analysis.

---

## 3. Infrastructure build sequence

Ordered so each step is usable on its own and later steps reuse earlier ones.

1. **Curve ingestion + MPL fitting (A5/A2).** Load DataDecide training curves per recipe/size/
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
   it would mean re-running branches. *An identical spec appears in Project A and Project D;
   keep them in sync.*
   Enables A-opt-4 and per-token analyses of branch endpoints.
4. **Checkpoint-merging tool (A1).** Sliding-window weighted averaging with configurable
   window and weight curve; expert-agnostic (dense models only); outputs a checkpoint that the
   eval harness treats as a variant. Evals-only, so it can run on the full DataDecide grid
   immediately.
5. **Decay-branch runner (A3).** Resume a checkpoint with configurable decay shape/length on
   the recipe's own data stream; log curves and per-token losses; hand the endpoint to the eval
   harness. Parameterise shape and length from day one (A-opt-1).
6. **Analysis layer (A4/A6).** Pairwise-decision recomputation from the results store;
   proxy-vs-branch agreement metrics; seed-noise confidence intervals; flip tables and figures.
7. *(Optional, A-opt-3)* **Post-training harness hookup.** Point the existing post-training
   protocol at `branch:*` variants and record outputs in the same store.

Steps 1, 2, and 4 can proceed in parallel; 3 should be finished before 5 starts; 6 is
incremental as data arrives. Step 1 plus the T0 half of step 6 (flip analysis on corrected
loss) are cheap and should be done regardless of whether the branch/merge half is pursued;
they also tell you whether that half is worth it.

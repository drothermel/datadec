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

1. **Fit the multi-power law per recipe (A5).** Using DataDecide's logged training curves,
   fit the MPL (power law in cumulative LR plus decay-drop terms) for each recipe × size × seed.
   Output: fitted parameters and a predicted "loss if decayed now" at every checkpoint (A2).
   No GPU. The fitted decay term also selects the branch grid: branch where the predicted
   decay gain is largest or most recipe-divergent.
2. **Run short decay branches from a grid of existing checkpoints (A3).** For a chosen set of
   recipes × checkpoint steps × (1–2) sizes × 3 seeds, resume training with a fresh decay
   (linear-to-zero or 1-sqrt; ~10% of elapsed tokens as the default length) on the recipe's
   own data, then run the full DataDecide eval suite on the endpoint. This is the ground-truth
   annealed readout.
3. **Compute checkpoint merges on the same grid (A1).** Sliding-window weighted average of the
   checkpoints preceding each branch point, with weights derived from an emulated decay curve
   (WSM). Eval with the same suite.
4. **Validate the proxies (A4).** Compare A1 (merge) and A2 (MPL) against A3 (branch) on the
   grid: loss agreement, task-metric agreement, and — most importantly — agreement on pairwise
   recipe decisions.
5. **Decision-flip analysis (A6).** Recompute DataDecide's pairwise recipe decisions on the
   annealed values and report which flip, at which steps, on which tasks. This is the headline
   figure. Compare flips under each proxy, and test DataDecide's published claim that
   intermediate-checkpoint decisions already match final ones.

### Optional directions

- **A-opt-1: Branch-length and branch-shape sweep.** At a fixed branch point, vary decay length
  (e.g. 2%, 5%, 10%, 20% of elapsed tokens) and shape (linear, 1-sqrt, cosine tail). Separates
  "reveal" (wall descent) from continued along-river progress and picks a canonical branch.
- **A-opt-2: Merge-window sensitivity (A1 detail).** Vary window length and weight curve for
  the merge; characterise where merging breaks on cosine checkpoints (it is only validated on
  stable-phase runs).
- **A-opt-3: Post-training from annealed vs. raw checkpoints (B3).** Re-run the earlier
  post-training protocol from A3 endpoints and the matching raw checkpoints. Tests whether the
  previous "post-training did nothing" result was a wall artifact.
- **A-opt-4: Token-loss-trajectory taxonomy on raw checkpoints (D5).** Rho-1-style
  classification of held-out tokens by loss trajectory across checkpoints. A cheap descriptive
  sub-result that previews Project D; no branches needed.
- **A-opt-5: Size scaling of the confound.** Repeat the core grid at a second model size to
  test whether the annealed-vs-raw gap and the flip rate shrink or grow with scale.
- **A-opt-6: Seed-noise floor.** Use the 3 seeds to put confidence intervals on every
  annealed-vs-raw difference; report the fraction of flips that exceed seed noise. This is less
  an option than a requirement for credibility, but it shapes how big the grid must be.

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
  tuning; this only weakens A2, not the paper.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| Core (A1–A6) | **High** | Directly audits a widely used public suite; the decision-flip figure is self-contained and quotable; methods contribution if merging holds. |
| A-opt-1 branch sweep | Medium | Needed for a defensible canonical branch length; modest standalone interest but strengthens every claim. |
| A-opt-2 merge sensitivity | Medium | Makes the "retrofit annealed evals for free" claim precise; mostly a supporting figure. |
| A-opt-3 post-training (B3) | **High, higher variance** | A positive result ("post-training gains appear once you anneal") is the most compelling story available and closes the loop on the earlier project; a null is still reportable but harder to sell. Adds post-training infrastructure cost. |
| A-opt-4 token taxonomy (D5) | Low–Medium | Nice descriptive figure; its real value is de-risking Project D. |
| A-opt-5 size scaling | Medium | Standard reviewer ask; doubles branch compute. |
| A-opt-6 seed-noise floor | Required | Not an option in practice — without it the flip analysis is not credible. |

**Recommended scope for a first workshop paper:** Core + A-opt-1 (small) + A-opt-6, with
A-opt-3 as a stretch if the branch grid finishes early. Defer A-opt-5 to a reviewer-response
or follow-up.

---

## 3. Infrastructure build sequence

Ordered so each step is usable on its own and later steps reuse earlier ones.

1. **Curve ingestion + MPL fitting (A5/A2).** Load DataDecide training curves per recipe/size/
   seed; fit the MPL; emit predicted decay gain per checkpoint. Pure Python, no GPU. Output
   also drives branch-grid selection. *Deliverable: per-recipe parameter table + predicted
   annealed-loss curves.*
2. **Checkpoint + eval harness.** Load any (recipe, size, seed, step) DataDecide checkpoint;
   run the DataDecide task suite and perplexity evals; store results keyed by that tuple plus
   a `variant` field (`raw`, `merged:<cfg>`, `branch:<cfg>`). Builds on this repo's existing
   data tooling. *Everything downstream writes into this store.*
3. **Fixed held-out token set + per-token loss logging.** Choose a held-out token set once
   (stratified across domains, sized for per-token statistics) and make per-token loss on it a
   standard output of the eval harness. Cheap to add now; expensive to retrofit later because
   it would mean re-running branches. Enables A-opt-4 and hands Project D its core artifact.
4. **Checkpoint-merging tool (A1).** Sliding-window weighted averaging with configurable
   window and weight curve; outputs a checkpoint that the eval harness treats as a variant.
   Evals-only, so it can run on the full DataDecide grid immediately.
5. **Decay-branch runner (A3).** Resume a checkpoint with configurable decay shape/length on
   the recipe's own data stream; log curves and per-token losses; hand the endpoint to the eval
   harness. Parameterise shape and length from day one (A-opt-1). This same runner serves
   Project B's branches later.
6. **Analysis layer (A4/A6).** Pairwise-decision recomputation from the results store;
   proxy-vs-branch agreement metrics; seed-noise confidence intervals; flip tables and figures.
7. *(Optional, A-opt-3)* **Post-training harness hookup.** Point the existing post-training
   protocol at `branch:*` variants and record outputs in the same store.

Steps 1, 2, and 4 can proceed in parallel; 3 should be finished before 5 starts; 6 is
incremental as data arrives.

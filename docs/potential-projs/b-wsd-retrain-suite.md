# Project B — A WSD-branch retrain suite for DataDecide recipes

**Working title:** *DataDecide-WSD: stable-phase pretraining with dense decay branches across
data recipes.*

**One-line pitch.** Retrain a subset of DataDecide recipes with a constant-LR stable phase and
launch fixed-length decay branches at many points along it. This is the dataset the field's own
methodology papers (Hägele et al.; the MiniCPM protocol) say should exist: an open, multi-recipe
suite where every intermediate readout is a proper annealed one.

Inventory IDs: B1, B2, B3.

This project is also being considered as **background cluster utilisation** — runs that keep the
cluster warm while attention is on evals-only work — so the doability section covers that
framing separately.

---

## 1. What the project involves

### Core experiment

1. **Choose the subset.** A handful of recipes spanning the known outcome range (best, worst,
   and a few that DataDecide finds hard to rank), at 1–2 sizes in the 150M–300M range, 3 seeds.
   Match DataDecide's token budget and architecture so results are directly comparable.
2. **Stable-phase training.** Constant LR after warmup (WSD "S" phase) to the full budget,
   saving checkpoints densely.
3. **Decay branches.** At a regular schedule of branch points (e.g. every 5–10% of the budget),
   launch a fixed-length decay (default ~10% of elapsed tokens, linear-to-zero or 1-sqrt) on
   the recipe's own data. Eval each branch endpoint with the full DataDecide suite, and log
   per-token losses on the fixed held-out token set (specified in §3, step 5).
4. **Release as a checkpoint suite** with curves, branch endpoints, and eval tables.

### Optional directions

- **B-opt-1: Branch-length sweep (B2).** At a few branch points, run several decay lengths
  and shapes to quantify reveal vs. continued progress and choose the canonical branch. Small
  relative to the stable phase.
- **B-opt-2: Post-training from branch endpoints (B3).** Post-train from annealed endpoints
  vs. matched stable checkpoints; the cleanest version of the "was post-training starting
  from the wall?" question.
- **B-opt-3: Cosine twins.** For one or two recipes, also train a cosine run with identical
  data order, so the suite contains a matched cosine-vs-WSD comparison. Makes decay
  branches resumed from cosine checkpoints directly validatable against a true stable phase.
- **B-opt-4: Mixed-in decay data.** MiniCPM-style: introduce high-quality data only during the
  decay. Tests whether decay-phase data interacts with recipe. Scope creep risk; note and defer.
- **B-opt-5: Extend to more recipes / a larger size** once the pipeline is proven — the
  "keep the cluster warm" mode.

---

## 2. Doability and impact

### Overall doability: **medium** (compute-bound, not idea-bound)

- Technically routine: OLMo-style training with a different schedule is a config change plus
  a branch launcher. The risk is operational — getting data order, tokenisation, and eval
  parity with DataDecide right, and the wall-clock cost of full-budget stable runs.
- Rough scale: a 150M run at DataDecide's budget is small; the branches add ~10% × number of
  branch points on top. Five recipes × 2 sizes × 3 seeds × (1 + 0.1 × 10 branches) is a
  meaningful but not unreasonable cluster allocation.
- Depends on nothing else. Every project that consumes annealed checkpoints or per-token
  branch logs benefits from it.

### As background cluster utilisation

This is the natural use of idle cluster time *provided* two things are fixed first:

1. **The schedule design is frozen** — branch spacing, decay length, checkpoint cadence,
   held-out token set. Changing these mid-way wastes the early runs. B-opt-1 on the first
   recipe should settle them.
2. **Per-token logging and the results store exist** (§3, steps 3 and 5), so runs produce
   the artifact downstream analyses consume rather than something to re-process later.

Given those, the stable-phase runs are fire-and-forget and branch launches can be scripted off
checkpoint arrival. Start with a recipe pair spanning the outcome range, or — if a
multi-power-law fit of the released loss curves is available — the pair it predicts to be most
decay-sensitive.

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| Core (B1) | **Medium as a paper, High as an enabler** | A resource contribution; workshops accept these but they are cited more than they are celebrated. Its value is that every other project becomes cleaner. Pair with a headline analysis (A6 or B3 on the suite) to make it a paper. |
| B-opt-1 branch sweep | Medium | Settles a methodological parameter the community mostly guesses at; a good figure, not a paper. |
| B-opt-2 post-training (B3) | **High** | Clean test of the personal-narrative hypothesis; stronger than the version resumed from cosine checkpoints because the starting checkpoints are genuinely stable-phase. |
| B-opt-3 cosine twins | Medium–High | Turns "does cosine-resumed branching approximate a true anneal" from an assumption into a measurement; directly validates decay branches resumed from cosine checkpoints. |
| B-opt-4 mixed decay data | Low (for now) | Interesting but a different paper; would dilute the suite's comparability. |
| B-opt-5 extension | n/a | Pure resource growth; only matters if the suite is being used. |

**Recommended framing:** do not pitch B1 alone as the workshop paper. Pitch it as the
infrastructure behind either the annealed-readouts question (cosine twins, B-opt-3) or the
post-training result (B-opt-2), and run the stable phases in the background while evals-only
work proceeds.

---

## 3. Infrastructure build sequence

1. **Training config parity.** Reproduce one DataDecide recipe with its original cosine
   schedule and confirm loss curve and eval parity with the published checkpoints. Without this
   the suite is not comparable.
2. **WSD config + dense checkpointing.** Swap in constant LR after warmup; set checkpoint
   cadence to the branch-point spacing.
3. **Decay-branch runner.** Resume from a checkpoint with configurable decay shape/length
   (linear-to-zero or 1-sqrt; length as a fraction of elapsed tokens) on the same data
   stream; log curves and per-token losses on the held-out set at start and endpoint; save
   endpoint weights; hand the endpoint to the eval harness as a `branch:<cfg>` variant.
   Parameterise shape and length from day one. *Project A specifies the same runner; if it
   already exists, reuse it unchanged.*
4. **Branch scheduler.** Watch for stable-phase checkpoint arrival and launch branches
   automatically; record provenance (parent step, decay config) in the results store.
5. **Results store + eval harness + held-out token set.** Load any (recipe, size, seed, step) DataDecide checkpoint; run
   the DataDecide task suite and perplexity evals; store results keyed by that tuple plus a
   `variant` field (`raw`, `merged:<cfg>`, `branch:<cfg>`), in the same table schema as the
   processed OLMES tables so results slot into existing accessors.
   All endpoints are evaluated with the DataDecide suite. Choose a held-out token set once and freeze it: fixed, versioned token
   sequences with a manifest; stratified across domains and across the DataDecide leaf
   corpora; sized so that each domain × entropy-bucket cell has enough tokens to estimate
   mean per-token loss drop within a set tolerance, while keeping one forward pass per
   checkpoint cheap. Per-token loss on it is a standard output of the eval harness for every
   checkpoint variant (raw checkpoints, merged checkpoints, branch starts and endpoints),
   stored as compact arrays keyed by (checkpoint variant, held-out-set version). Branch
   endpoints also save their weights. Cheap to add now; expensive to retrofit later because
   it would mean re-running branches. *An identical spec appears in Project A, Project B, and
   Project D; keep them in sync.*
6. **Pilot on one recipe, one size, one seed** end to end, including B-opt-1 on two branch
   points, to freeze the schedule design.
7. **Scale out** to the chosen recipe subset; then B-opt-3 cosine twins; then B-opt-5 as idle
   capacity allows.
8. **Packaging** for release: checkpoint naming, manifest, eval tables, curve dumps.

Step 1 is the gating item and is worth doing early regardless of whether B proceeds, since
it also validates the training stack for any decay branches resumed from released checkpoints.

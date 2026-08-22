# Open questions — answered

Running log of gate checks and open questions from the potential-project docs
(`docs/potential-projs/` and the `dataset-analysis` idea map) that have been resolved against local data. Each
entry records the date, the answer, and the exact code used, so the check can
be re-run when `data/processed/` changes.

Add new entries at the bottom; do not rewrite old ones — amend with a dated
note if an answer changes.

---

## 2026-08-21 — Per-instance eval coverage (gates IRT reanalysis, the trajectory noise-floor item bootstrap, and TOK-obs-2 item flips)

**Question.** Do instance-level OLMES results exist for every recipe × scale ×
seed cell, or only a subset? (Raised in `irt-reanalysis.md` §2,
`recipe-featurization.md` step 3, and `dataset-analysis-idea-map.md` §7.)

**Answer.** Instance-level results exist at
`data/processed/olmes-details/{recipe}/instances.parquet` for **all 25 recipes
and all 66 tasks**, but only for **9 of the 14 sizes**, and seed coverage
differs by size:

| params | seeds | distinct steps | max step |
|---|---|---|---|
| 4M | 1 | 8 | 5745 |
| 20M | 1 | 14 | 14594 |
| 60M | 1 | 27 | 29062 |
| 90M | 1 | 25 | 29901 |
| 150M | 3 | 34 | 38750 |
| 300M | 3 | 41 | 46250 |
| 530M | 3 | 50 | 57786 |
| 750M | 3 | 54 | 63599 |
| 1B | 3 | 31 | 69369 |

Missing sizes: 6M, 8M, 10M, 14M, 16M. Columns include binary correctness
(`acc_raw`, `acc_per_token`, `acc_per_char`, `acc_per_byte`, `acc_uncond`) and
continuous margins (`sum_logits_corr`, `logits_per_*_corr`), so both the binary
and margin IRT response models (IRT-1/IRT-5) are buildable without the separate
`choices.parquet`.

**Consequence.** IRT dimensionality (IRT-1), recipe-DIF (IRT-3), and the item
bootstrap in the noise-floor module are fully supported at **150M–1B (3
seeds)**. Below 150M there is one seed per cell, so seed-variance estimates
are unavailable there and pooled-across-recipe floors must be used instead.

## 2026-08-21 — Checkpoint spacing (gates Trajectory drift/diffusion; decides whether TRJ-5 resolution transfer is needed)

**Question.** Are DataDecide checkpoints dense enough that adjacent-checkpoint
increments can separate diffusion from drift, or is transfer from the denser
OLMo trajectories (TRJ-5) required? (Raised in `trajectory-statistics.md` §2 and
`dataset-analysis-idea-map.md` §7.)

**Answer.** From the aggregate `data/processed/olmes.parquet` (25 recipes × 3
seeds at every size):

| params | distinct steps | max step | ≈ spacing |
|---|---|---|---|
| 4M | 6 | 5725 | ~1000 |
| 6M | 5 | 9182 | ~1800 |
| 8M | 10 | 13039 | ~1300 |
| 10M | 12 | 15117 | ~1250 |
| 14M | 16 | 20000 | ~1250 |
| 16M | 19 | 24432 | ~1250 |
| 20M | 13 | 14584 | ~1100 |
| 60M | 23 | 29042 | ~1250 |
| 90M | 22 | 29901 | ~1350 |
| 150M | 31 | 37500 | ~1200 |
| 300M | 37 | 45000 | ~1200 |
| 530M | 42 | 51250 | ~1200 |
| 750M | 22 | 26250 | ~1200 (truncated — see below) |
| 1B | 27 | 69369 | ~2500 |

Spacing is roughly uniform at ~1,000–1,300 steps for everything from 8M to
530M, with 30–40 checkpoints per run at 150M–530M. That is enough points per
series for windowed drift+diffusion fits; **TRJ-5 (OLMo resolution transfer) is
not needed as a prerequisite** and can stay an optional robustness check. 1B
is the coarsest (~2,500-step spacing, 27 points) and the sub-10M sizes have
too few points to fit per-run.

## 2026-08-21 — 750M aggregate table is truncated; the instance table is not

**Observation.** In `olmes.parquet` the 750M rows stop at step **26,250** (22
checkpoints), well short of the run length. The instance table for 750M goes
to step **63,599** with 54 checkpoints. So the aggregate table is missing
roughly the second half of the 750M trajectory, while the instance-level data
is complete.

**Consequence.** Any 750M trajectory analysis should be computed from
`olmes-details/*/instances.parquet` (aggregating instances → task accuracy
ourselves) rather than from `olmes.parquet`, and the aggregate ingest for 750M
should be investigated and fixed. This is also the cheapest way to rebuild
aggregates for all sizes from a single source.

### Code used for the three entries above

Run from the repo root:

```bash
uv run python - <<'PY'
import duckdb, glob
fs = sorted(glob.glob('data/processed/olmes-details/*/instances.parquet'))
print(len(fs), 'instance files')
c = duckdb.connect()
print([r[0] for r in c.sql(f"describe select * from '{fs[0]}'").fetchall()])
print(c.sql("""
    select params, count(distinct seed) seeds, count(distinct step) steps,
           min(step), max(step), count(distinct task) tasks
    from read_parquet('data/processed/olmes-details/*/instances.parquet')
    group by 1 order by 1
""").fetchall())
print('--- olmes.parquet spacing by params:')
print(c.sql("""
    select params, count(distinct step) steps, min(step), max(step),
           count(distinct data) recipes, count(distinct seed) seeds
    from 'data/processed/olmes.parquet'
    group by 1 order by 1
""").fetchall())
PY
```

## 2026-08-21 — Slicing-and-Dicing MoE sweep: final checkpoints are available

**Question.** Are the trained models from the Slicing-and-Dicing MoE sweep
(https://arxiv.org/abs/2605.11689; Danielle third author) retained and obtainable? Gates the
sweep-reanalysis idea (taxonomy invariance across configs / seeds / balancing mechanisms via
cross-model expert matching) and the use of the sweep as a matched-loss comparison across
architectures — see `docs/topics/moe-analysis-program.md` and `docs/topics/moe-recipe-suite.md`.

**Answer (confirmed with the collaborator who ran the sweep, 2026-08-21).** All **final**
checkpoints exist. One of the two will upload them to Hugging Face fairly soon. The
collaborator is also running a range of additional experiments; for some of those,
intermediate checkpoints / logs can likely be obtained.

**Consequence.** The sweep-reanalysis work is unblocked on artifacts (final checkpoints only
— no intermediate checkpoints from the original sweep, so training-dynamics analyses on the
sweep itself are not available; any "over training" analysis needs the new experiments or
runs of our own). Remaining open items: the HF upload landing, and which of the new
experiments will log intermediates.

---

## Open — not yet checked

- **Loss-curve coverage in the scaling-law table.** Does
  `processed/scaling-law/checkpoint-losses.parquet` cover every recipe × size × seed at
  reasonable resolution? Gates the multi-power-law fit (ANN-5, REC-7).
- **FLAME-MoE routing-log contents.** Which checkpoints, how many tokens, whether token
  identities are recoverable. Gates the routing follow-up (TRJ-moe-1/3) and TOK-obs-5
  entirely; decides T0 vs. T1 for them.


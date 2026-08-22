# Reset-response as a training-stage probe — a candidate fifth tomography probe

**Kind:** staging. Candidate exits: an additional probe in the checkpoint-tomography
battery (`checkpoint-tomography.md`), or an arm of the critical-period timing study. Gap
**G7**.

Source: the 2026-08-22 reinit/transfer literature pass (`reinit-and-transfer-literature.md`;
full report at `~/drotherm/data/.claude/datadec/2026-08-22/0031-reinit-transfer-litpass.md`).
Gap statements are quoted from that report; "closest work" citations were retrieved by the
subagent (arXiv IDs), but verdicts rest on abstracts and no forward-citation sweep was run.
---

## 2026-08-22 — the gap

**G7 — Reset-response as a measurement instrument for critical periods** *(medium
confidence)*. "Critical-period work (TACL, doi:10.1162/tacl_a_00725) intervenes on *data*;
reset work intervenes on *weights*. Nobody uses 'how fast does the model recover from an
interface reset at step t' as a scalar probe of where the model is in its training life.
If recovery cost is stage-dependent it would be a cheap, seed-robust critical-period
readout." Closest: TACL a_00725; PolyPythias (arXiv 2503.09543). Cost: small training runs
on PolyPythias checkpoints.

**Why it fits tomography.** The battery already has decay, hot, twin, and data-shifted
branches; a reset branch (reset an interface or a block, continue briefly, measure recovery
cost and barrier to the pre-reset model) is the same shape — a short continuation from a
checkpoint yielding a stage-dependent scalar — and shares the runner. It also gives the
critical-period timing study a weight-side intervention to set against its data-side
deficits.

**Waiting on:** whether tomography is promoted; whether to include this as its fifth probe.

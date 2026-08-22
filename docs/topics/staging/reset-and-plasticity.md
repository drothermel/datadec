# Resets and plasticity in LMs — does an interface reset restore plasticity, and which layers need resetting?

**Kind:** staging. Candidate exits: a project doc on the tiny-scale substrate; or absorption
into the plasticity reference topic and tiny-scale measurement. Gaps **G4, G9**.

Source: the 2026-08-22 reinit/transfer literature pass (`../reference/reinit-and-transfer-literature.md`;
full report at `~/drotherm/data/.claude/datadec/2026-08-22/0031-reinit-transfer-litpass.md`).
Gap statements are quoted from that report; "closest work" citations were retrieved by the
subagent (arXiv IDs), but verdicts rest on abstracts and no forward-citation sweep was run.
---

## 2026-08-22 — the gaps

**G4 — Does an embedding reset restore plasticity, or is it orthogonal to it?** *(high
confidence)*. "The plasticity literature resets *body* layers; the tokenizer literature
resets *interfaces* and never asks about plasticity. Given 2606.24752 now establishes LM
plasticity loss at 5M–314M, the question 'does the cheap interface reset buy any of what an
expensive body reset buys' is directly askable and unasked. Plasticity injection (arXiv
2305.15555) supplies a ready diagnostic." Cost: small training runs on existing checkpoints.

**G9 — Which layers actually need resetting in an LM?** *(medium confidence)*. "The RL
survey (arXiv 2411.04832) reports that plasticity loss is 'commonly believed' concentrated
in the last layers — stated as belief, not evidence — and no LM-side layer-wise reset
ablation exists." Closest: arXiv 2411.04832; calibrated partial resets (arXiv 2607.24996).
Cost: small training runs.

**Design sketch.** Layer-wise reset ablation (interface only; last block; middle blocks;
continual-backprop-style least-used units) on checkpoints with measured plasticity loss
(*Can Scale Save Us*, arXiv 2606.24752), each arm followed by the plasticity-injection
diagnostic and the Dohare/Lyle panel (curvature, feature rank, dead units, weight norm,
Fisher trace). Ties to the warm-starting intervention grid's "period-reopening
interventions" row.

**Waiting on:** a decision to promote.

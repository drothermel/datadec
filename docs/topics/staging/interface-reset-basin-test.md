# Is an interface reset basin-preserving?

**Kind:** staging. Candidate exits: a small project doc, or an optional direction inside
landscape geometry (it is a direct use of GEO's interpolation tool). Gap **G3** — "the
single best-shaped question for Danielle's program."

Source: the 2026-08-22 reinit/transfer literature pass (`../reference/reinit-and-transfer-literature.md`;
full report at `~/drotherm/data/.claude/datadec/2026-08-22/0031-reinit-transfer-litpass.md`).
Gap statements are quoted from that report; "closest work" citations were retrieved by the
subagent (arXiv IDs), but verdicts rest on abstracts and no forward-citation sweep was run.
---

## 2026-08-22 — the gap

"The layer-wise LMC result (arXiv 2307.06966) says middle layers own the barrier and
per-layer perturbations are near-barrier-free — which predicts an embedding reset stays in
the basin — but nobody has tested this by resetting an interface and measuring the barrier
to the pre-reset solution. I searched LMC + reinitialization, loss barrier + reset, and
CKA + tokenizer swap; found the instruments and found reset methods, never the two
combined." Closest: arXiv 2307.06966; LMC of MoEs (arXiv 2509.11348). Cost: forward passes
on existing checkpoints plus short recovery runs.

**Design sketch.** For checkpoints at several stages (PolyPythias, arXiv 2503.09543): reset
the input embeddings (and separately the head), run a short recovery, then measure (i) the
linear-interpolation barrier between the recovered model and the pre-reset model, raw and
permutation-aligned; (ii) CKA / layer-wise feature connectivity; (iii) the same for a body
reset of matched parameter count as the contrast. Prediction from the interpretive frame in
: interface resets are basin-preserving, body resets are
not. Many seeds (the Butterfly Effect, arXiv 2506.13234).

**Waiting on:** a decision to promote, or to add as GEO-opt-6.

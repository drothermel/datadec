# Embedding-reset dynamics — recovery curves, input/output asymmetry, basins, and plasticity

> **Draft scaffolding (2026-08-22).** Promoted from a staging topic. The quoted material in §4
> is external text; §1–§3 are synthesized scaffolding not yet reviewed by Danielle. Treat them as
> provisional until this note is removed.

**Program pillars served:** how (a cheap, seed-robust phenomenon on a released many-seed
substrate), mechanism (interface resets as basin-preserving perturbations; resets as
period-reopening interventions), apex (a training-history effect that recovers fast —
measured properly). (Program: `README.md` → Program.)

**One-line pitch.** Danielle's 2020–21 result — reset a pretrained LM's input/output
embeddings, continue on the original data, recover in a tiny fraction of the run — is now
corroborated in spirit at LLM scale (EEVE at ~2B tokens; 6× CPT reductions from good
initialization) but has never been studied as a phenomenon with controlled scale, stage,
and seeds. Measure the recovery-cost curve, explain the input-vs-output asymmetry, test
whether an interface reset stays in the basin, and ask whether it restores plasticity.

IDs: RESET-1–RESET-4 (core), RESET-opt-1–RESET-opt-5. Built from gaps G1, G2, G3, G4, G8,
G9, G10 of the reinit/transfer literature pass.

**Paper goal.** Workshop-sized from RESET-1 + RESET-2 (the recovery curves and the
asymmetry); main-conference with the basin test and the plasticity arm.

**Substrate.** PolyPythias (arXiv 2503.09543): 50 pretraining runs, 9 seeds × 5 sizes
(14M–410M), ~7,000 checkpoints. The Slicing-and-Dicing MoE repo for an MoE arm.

Compute tiers: **T0** = analysis of published tables only; **T1** = forward passes with
existing checkpoints; **T1+** = checkpoint merging plus re-running evals; **T2** = short
continued-training branches or fine-tunes; **T3** = new pretraining runs.

---

## 1. What the project involves

### Core experiment

1. **Recovery-cost curve (RESET-1, T2).** Reset the input embeddings at checkpoints across
   stages and sizes; continue training on the original data; measure tokens to recover
   X% of pre-reset loss (curves, not a single number), with seeds. Estimates in circulation
   span 500 steps to 2B tokens; no controlled curve exists.
2. **Input vs. output asymmetry (RESET-2).** Reset input-only, head-only, and both; tied vs.
   untied embeddings; the asymmetry is observed (distinct optimal inits) but unexplained.
3. **Seeds and regularization control (RESET-3).** Every arm with enough seeds for
   confidence intervals; re-run under tuned regularization — in vision, reinit's benefit
   disappears once regularization is tuned (Zaidi et al.); unchecked in LMs.
4. **Initialization in the limit (RESET-4).** Naive vs. FOCUS / OMP / ZeTT initialization:
   do they converge to the same place given enough tokens (init as a pure speed knob)?

### Optional directions

- **RESET-opt-1: Is an interface reset basin-preserving?** Barrier (raw and aligned) and
  CKA / stitching between the recovered model and the pre-reset model; body reset of
  matched parameter count as the contrast. Layer-wise LMC predicts interface resets are
  near-barrier-free. Cross-listed as GEO-opt-6.
- **RESET-opt-2: Does an interface reset restore plasticity?** Plasticity-injection
  diagnostic and the Dohare/Lyle panel after interface vs. body resets, on checkpoints
  with measured plasticity loss.
- **RESET-opt-3: Which layers need resetting in an LM?** Layer-wise reset ablation
  (interface; last block; middle blocks; least-used units) — the "last layers" belief is
  weakly evidenced.
- **RESET-opt-4: Reset-response as a stage probe.** Recovery cost from a reset at step t as
  a critical-period readout (cross-listed with checkpoint tomography's fifth probe and the
  intervention grid).
- **RESET-opt-5: Storage.** Save branch endpoints as quantized deltas from the reset point.

---

## 2. Doability and impact

### Overall doability: **high** — small runs on a released many-seed substrate

- The seeded "≥50B tokens to recover" figure was a misreading (Dagan et al. say >50B tokens
  lets specialization *pay off*); the actual recovery cost at small scale is unmeasured,
  which is the opportunity.
- Risks: confounds from weight tying (control explicitly); recovery-curve shape may depend
  on the continued-training LR schedule (re-warm vs. not — report both); trajectories are
  seed-sensitive (many seeds).
- Composes with the intervention grid (resets as period-reopening interventions) and with
  landscape geometry (the basin test uses its interpolation tool).

### Per-direction workshop-paper impact

| Direction | Impact | Rationale |
|-----------|--------|-----------|
| RESET-1 recovery curve | **High** | Reconciles estimates spanning three orders of magnitude; the phenomenon behind standard industrial practice, measured. |
| RESET-2 asymmetry | High | Observed as a tuning finding; mechanism unasked. |
| RESET-3 control | Medium (required) | Whether the effect survives tuned regularization and seeds. |
| RESET-4 init in the limit | Medium | May partly exist; cheap arm. |
| RESET-opt-1 basin test | **High** | "The single best-shaped question for Danielle's program." |
| RESET-opt-2 plasticity | High if positive | Does the cheap interface reset buy what an expensive body reset buys? |
| RESET-opt-3 layers | Medium-high | First LM-side layer-wise reset ablation. |

---

## 3. Infrastructure build sequence

1. **PolyPythias checkpoint loader** and continued-training harness (deterministic order;
   re-warm option).
2. **Reset operators**: input / head / both; tied/untied handling; initialization schemes
   (naive, FOCUS, OMP, ZeTT).
3. **Recovery-curve logging** (loss vs. tokens; held-out per-token loss on the shared set).
4. **Seed/regularization arms** (RESET-3).
5. *(Optional)* Interpolation + alignment + stitching tooling (RESET-opt-1); plasticity
   injection and panel (RESET-opt-2); layer-wise reset ablation (RESET-opt-3).


---

## 4. External assessments and origin notes

Dated notes from external conversations and the staging topic this doc was promoted from,
recorded for consolidation — not decisions. Related-work claims in quoted text are
unverified unless a citation is given.

### Origin notes — moved from `topics/staging/reset-recovery-dynamics.md`

## 2026-08-22 — the gaps

**G1 — Recovery-cost curve for an embedding reset** *(high confidence)*. "No paper measures
how many tokens are needed to recover from an input-embedding reset as a controlled
function of model size and how far into pretraining the reset happens. Estimates in
circulation span 500 steps (2608.03494), 2B tokens (EEVE), and '>50B' (Dagan, misread)…
every hit optimizes *initialization quality* at one scale on one model, never the recovery
*curve*." Cost: small training runs.

**G2 — Input-vs-output embedding reset asymmetry, explained** *(high confidence)*.
"2608.03494 establishes that input and output embeddings want *different* init strategies,
and reports it as a tuning finding with no mechanism. No paper isolates resetting the LM
head alone vs the input embedding alone vs both, and measures recovery separately. Given
weight tying is common, this is also a confound nobody has controlled." Cost: small runs.

**G10 — Does init quality matter once you continue training long enough?** *(lower
confidence — may exist)*. "The whole init literature optimizes a quantity (init loss/BPB)
that 2608.03494 shows is an unreliable predictor of convergence. The implied question — do
FOCUS/OMP/ZeTT/naive all converge to the same place given enough tokens, making init a pure
*speed* knob? — is asked implicitly but I found no explicit convergence-crossover study."

**Substrate and design notes.** PolyPythias (arXiv 2503.09543; 14M–410M, 9 seeds × 5 sizes,
~7k checkpoints) supplies checkpoints at many stages and seeds; the reset × stage × size
grid with recovery curves is the core; arms for input-only / output-only / both, tied vs.
untied, and naive vs. FOCUS/OMP/ZeTT init. Report recovery as curves (tokens to X% of
pre-reset loss), not a single number.

**Waiting on:** a decision to promote; whether to fold G10 in as an arm or drop it.

### Origin notes — moved from `topics/staging/interface-reset-basin-test.md`

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

### Origin notes — moved from `topics/staging/reset-and-plasticity.md`

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

### Origin notes — moved from `topics/staging/reset-effects-many-seed-lm.md`

## 2026-08-22 — the gap

**G8 — Many-seed replication of reset effects in LMs** *(medium confidence)*. "*When Does
Re-initialization Work?* (Zaidi et al., arXiv 2206.10011) did 15,000 vision models and found
the effect *disappears* under tuned regularization — a result nobody has checked in LMs.
Combined with the Butterfly Effect finding (arXiv 2506.13234) that trajectories are
seed-sensitive, most single-seed LM reset claims are underpowered. PolyPythias makes the
seed dimension free." Cost: small training runs.

**Design note.** Same shape as the warm-starting decomposition's factorial — reset
interventions × regularization/optimizer settings × seeds — with the outcome being whether
the reset effect's confidence interval excludes zero once regularization is tuned. This is
the "exhaust the boring explanations" discipline applied to resets.

**Waiting on:** a decision on whether this stands alone or becomes the seed/regularization
requirement in the other reset topics.

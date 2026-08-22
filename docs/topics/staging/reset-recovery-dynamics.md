# Embedding-reset recovery dynamics — cost curve, input/output asymmetry, and whether init matters in the limit

**Kind:** staging. Candidate exits: a project doc (small-scale, many-seed study of recovery
after resetting input and/or output embeddings, as a function of scale, training stage,
seed, and initialization); or absorption into tiny-scale measurement. Gaps **G1, G2, G10**.

Source: the 2026-08-22 reinit/transfer literature pass (`../reference/reinit-and-transfer-literature.md`;
full report at `~/drotherm/data/.claude/datadec/2026-08-22/0031-reinit-transfer-litpass.md`).
Gap statements are quoted from that report; "closest work" citations were retrieved by the
subagent (arXiv IDs), but verdicts rest on abstracts and no forward-citation sweep was run.
**Danielle origin.** Her 2020–21 private result (reset the embeddings, continue on the
original data, recover in a tiny fraction of the run) — now corroborated in spirit by EEVE
(arXiv 2402.14714, ~2B tokens) and *Beyond Initialization Loss* (arXiv 2608.03494, 6× CPT
reduction), but "never studied as a *phenomenon* with controlled scale and seeds."

---

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

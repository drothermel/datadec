# Parametrization, hyperparameter transfer, and low-precision training — reference topic

**Kind:** reference (accumulator for µP-family parametrizations, width/depth
hyperparameter transfer, and numerics that make small-proxy sweeps predictive of larger
runs). Entries are dated; claims are condensed from responses and unverified unless
marked.

Why it matters here: every many-small-runs design in this repo (GEO, TINY, EDP,
DataDecide-dense, the MSUITE recipe suite) assumes that hyperparameters tuned on a small
proxy say something about the target size, and DataDecide's own per-size hyperparameters
are part of the recipe confound (DCARD). The parametrization literature is where that
assumption is either justified or broken.

---

## Undated (intake 2026-08-22) — u-µP: the Unit-Scaled Maximal Update Parametrization (Blake et al., arXiv 2407.17465; two turns)

**Danielle's prompts.** (1) Key takeaways from the attached paper. (2) Describe the
contents and importance of the provided figure (the paper's Figure 1).

**Response 1 (condensed).** u-µP combines µP with Unit Scaling (all activations, weights,
and gradients at unit variance at init). Three stabilities targeted: feature learning,
hyperparameter, numerical. Problems with standard µP as claimed: fails to transfer for
Llama-style models in realistic training setups; no principled choice of which HPs to
sweep; "base shape" machinery adds complexity; low-precision training often fails despite
theory. Contributions: no base shapes and simpler scaling rules; a redesigned, smaller,
interpretable, less interdependent HP set; **independent search** (sweep LR alone, then
each multiplier separately, then combine) — 9 runs vs. 339 for a random search of
comparable quality; **out-of-the-box FP8** via a plain cast, no dynamic rescaling, ~70% of
matmuls in FP8, no significant loss degradation vs. BF16; a mathematically equivalent
unit-scaled residual scheme with an interpretable attention-vs-FFN branch control; the
**embedding LR rule changed from c_emb = 1 to c_emb = 1/√fan-out**, fixing a width-scaling
issue. Results: equal or lower loss than µP, better transfer, continued gains with width
where µP showed diminishing returns; validated up to 7B on 300B tokens; open-source
library and user guide.

**Response 2 — Figure 1 (condensed).** (a) Loss vs. number of runs: µP random search
(hundreds of runs) against u-µP independent search — LR sweep first (×), then multipliers
(○), then the combination (◇); the LR sweep alone is near-optimal. (b) LR sweeps across
widths 128→4096: u-µP's optimum found at width 256 transfers to 4096; µP's optimum shifts
with width (circled proxy→target points). (c) FP32 vs. FP8 loss curves (solid/dashed)
coincide for u-µP; the caption states standard µP fails under the same simple cast.

**Intake notes.**

- What to keep as method, not marketing: (i) independent HP search is a *sweep design*
  — a small proxy sweep with decoupled axes — usable in any many-small-runs project
  regardless of parametrization, if the decoupling holds; the paper's claim that it holds
  is specific to u-µP's HP set. (ii) The embedding-LR correction is a concrete, checkable
  deviation from the canonical µP rule. (iii) Panel (b) is the only panel that tests the
  transfer claim; (a) and (c) are about search cost and numerics.
- For DataDecide: the suite uses per-size hand-set hyperparameters (DCARD records which),
  i.e. it is *not* µP-parametrized, so its cross-size comparisons carry the usual
  "was the small model's LR optimal" confound. A DataDecide-dense retrain
  (`../staging/datadecide-dense.md`) could adopt u-µP to buy transfer and a 9-run sweep
  per recipe — a design decision to record there, not decided here.
- Cautions: the response's "4× speedup" and ">35× search reduction" are the paper's
  framing of FP8 peak throughput and 9-vs-339 runs, not measured end-to-end gains;
  "continued benefits from width unlike µP" is a paper claim about its own experiments;
  the first response attached a giant S3 pre-signed URL as the citation for every
  bullet, which is the uploaded PDF, not an independent source. Unverified throughout;
  canonical ID 2407.17465 (v3 read).
- Related and absent (Claude-added, unverified): the original µP/µTransfer (Yang et al.,
  2203.03466); Unit Scaling (Blake et al., 2303.11257); depth-µP / CompleteP for
  depth-wise transfer; these are the obvious next entries if this topic grows.

## 2026-08-24 — HP-scaling-law cluster from the NotebookLM pretraining notebook

From the 11-paper NotebookLM notebook (bundle:
`nblm-pretraining-dynamics-notebook.md`; main entry in
`schedules-and-annealing-literature.md`, same date; no IDs supplied,
agent-generated, unverified):

- **CompleteP detail** (previously a bare "related and absent" name here):
  extends µP with Query-Key-norm and AdamW-ε adjustments to unify width, depth,
  batch-size, and token-horizon transfer; per-module HP optimization
  (evolutionary search) at 50M transfers to a 420× larger compute scale with up
  to a 1.32× training speedup.
- **Power Lines** — WD/BS scaling laws from hundreds of µP models (111M–3.3B,
  SlimPajama): the optimal AdamW timescale decreases as a power law of
  tokens-per-parameter D/N (optimal weight decay pre-calculable); optimal and
  critical batch sizes scale with **D alone, independent of N**.
- **Step Law (Predictable Scale, Part I)** — 3,700 runs (dense + MoE, 60M–1B,
  ≤100B tokens): the loss landscape is **strictly convex in (LR, BS)**; optimal
  BS depends primarily on D, optimal LR jointly on N and D; transfers across
  topologies, MoE sparsity, and data recipes.

Convergent claim across both: batch size is a dataset-size story, not a
model-size story. Directly relevant to the small-scale sweep designs
(wsd-suite, tiny-scale-measurement): these laws say which HP axes must be
re-tuned per scale and which can be fixed by formula.

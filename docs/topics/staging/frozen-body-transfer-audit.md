# Frozen-body transfer, re-audited — LR-tuning asymmetry, the reservoir null, and how much of the body a frozen interface can reach

**Kind:** staging. Candidate exits: a project doc directly downstream of Rothermel et al.
2021 (arXiv 2107.12460), or an arm of the ICL-as-post-training / elicitation-ceiling work.
Gaps **G5, G6**. Both rest on keyword absence rather than a forward-citation sweep of
2107.12460 — run that sweep first.

Source: the 2026-08-22 reinit/transfer literature pass (`../reference/reinit-and-transfer-literature.md`;
full report at `~/drotherm/data/.claude/datadec/2026-08-22/0031-reinit-transfer-litpass.md`).
Gap statements are quoted from that report; "closest work" citations were retrieved by the
subagent (arXiv IDs), but verdicts rest on abstracts and no forward-citation sweep was run.
---

## 2026-08-22 — the gaps

**G5 — LR-tuning asymmetry as an unaudited confound in modern frozen-body transfer claims**
*(medium-high)*. "The 2021 correction showed frozen-vs-finetuned comparisons invert under
proper LR tuning. Modern frozen-body work (X-Fusion, arXiv 2504.20996; PDE adaptation,
arXiv 2510.05278; frozen time-series transformers, arXiv 2508.18130) makes structurally
identical claims. I searched for follow-up audits and found none; the rebuttal appears
cited but not operationalized. A re-audit with per-condition LR sweeps, plus the reservoir
null from arXiv 2508.18130 (does a *randomly initialized* frozen body do as well?), would be
cheap and high-value." Cost: small training runs.

**G6 — How much of what the body carries can a frozen interface reach?** *(medium-high)*.
"The 2021 result was that a frozen body underperforms full finetuning — i.e. the interface
could not *reach* the body's content with 2021 instruments. With modern elicitation (linear
probes across depth, CKA, sparse autoencoders, plasticity injection as diagnostic), the
question becomes quantitative: what fraction of the body's usable content is
interface-reachable, and does that fraction change with scale or training stage? No paper
reframes the frozen/finetuned gap as an elicitation-ceiling measurement." Closest: arXiv
2107.12460; arXiv 2410.06225. Cost: forward passes on existing checkpoints plus light probe
training.

**Relation to the research hypothesis.** G6 is the frozen-interface instance of the
capability-vs-accessibility decomposition in `../../research-hypothesis.md`; G5 is the 2021
paper's own method applied to its successors ("would the 2021 author be able to kill this
result by tuning the null harder?").

**Waiting on:** the 2107.12460 forward-citation sweep; a decision to promote.

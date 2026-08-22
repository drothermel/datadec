# Layer looping, recursive depth, and cross-layer weight tying — literature reference

**Kind:** reference (accumulator for architectures that apply one set of layer weights
repeatedly to the same token — Universal Transformers / ALBERT lineage — with emphasis on
LLM-scale work from the last five years). Entries are dated. Characterizations are a
SciSpace agent's; identifiers unverified. No project in `potential-projs/` currently
depends on this; it is an architecture-interest accumulator. Nearest neighbours:
`moe-literature.md` (MoEUT / Sparse UT sit at the intersection), `plasticity.md`.

**Artifacts:** none — the SciSpace download failed repeatedly and Danielle chose to keep
only the pasted report ("I don't think I care enough about this family").

---

## 2026-08-22 — SciSpace deep review (undated, ~2026)

**Danielle's prompt (verbatim):**

> Do a literature review on layer looping or repeating, or layers with tied weights, in
> machine learning. I'm referring specifically to the mechanism in the AlBERT and
> Universal Transformers papers, and others, where on set of layer weights is used
> multiple times on one token. I want you to start with papers that cite AlBERT and
> Universal Transformers and fit my description, but also include any other potentially
> relevant papers. Focus especially on papers about LLMs from the past 5 years.

**The map the report gives (condensed).**

- *Foundations:* Universal Transformers (Dehghani et al. 2018; shared layer applied
  recurrently with ACT halting; Turing-complete with enough memory) and ALBERT (Lan et
  al. 2019; cross-layer sharing of attention / FFN / both; factorized embeddings).
- *Modern recursive variants at LM scale:* **Relaxed Recursive Transformers** (Bae et al.
  2410.20672; K-layer block repeated, per-iteration LoRA deltas initialised by SVD of
  layer differences; recursive Gemma 1B uptrained from 2B ≈ full Gemma after 60B tokens +
  distillation; depth-wise batching for 2–3× throughput); **Sparse Universal
  Transformer** (Tan et al. 2310.07096; shared block with SMoE in FFN and attention,
  stick-breaking halting, ~50% compute cut on formal-language tasks); **MoEUT** (Csordás
  et al. 2405.16039; MoE FFN + SwitchHead attention inside a shared group, peri-layernorm;
  fixes UT's poor parameter-to-compute ratio); **Dynamic Layer Tying** (Hay et al.
  2401.12819; learn which layers share); **Mixture of LoRAs for recursive transformers**
  (2512.12880); **Head-wise Shareable Attention** (Cao et al. 2402.11819; share ~30% of
  heads at ~99.5% retained accuracy); **Looped Transformers for Length Generalization**
  (Fan et al. 2409.15647); **Retrofitted recurrence** (McLeish et al. 2511.07384; add
  loops to pretrained LMs); **recurrent-depth latent reasoning** (Geiping et al.
  2502.05171; scale test-time compute by iterating a block); SpiralFormer
  (multi-resolution recursion; no ID); Liger linearization (2503.01496); Embedding
  recycling (2207.04993; adjacent, not looping).
- *Adaptive depth:* Depth-Adaptive Transformer (Elbayad et al. ICLR 2020), Recurrent
  Transformers with Dynamic Halt (Chowdhury et al. 2402.00976), ELBERT early exit for
  ALBERT (ICASSP 2021).
- *Theory:* Looped Transformers as Programmable Computers (Giannou et al. 2301.13196);
  looped transformers learn multi-step gradient descent in context (Gatmiry et al.
  2410.08292; Chen et al. 2410.11268); log-depth expressivity (Merrill et al.
  2503.03961); residual scaling / stability of looped transformers (Wang et al.; no ID);
  depth and compositional generalization (Petty et al. 2310.19956, NAACL 2024).
- *Scaling:* an "iso-depth scaling law for looped LMs" (Schwethelm et al.; cited into
  the wrong reference slot) — first loops give most of the gain, 2–4 iterations a
  reasonable default; diminishing returns beyond.
- *Report's synthesis:* 2–18× parameter reduction at modest loss; recurrence is
  computation, not only compression; adaptive halting matters; hybrids (LoRA, MoE,
  hierarchical) dominate pure sharing; no frontier production LLM uses it (as of the
  report).

**Intake notes.**

- Citation numbering is broken: [1], [2], [6], [8], [26], [30] each point at two or
  three different papers; ALBERT itself is not in the reference list; Schwethelm's
  scaling law and ELBERT share Dehghani's slot. Resolve by title.
- Factual slip: "ALBERT-xxlarge … 144 layers vs. 12 … 18× reduction per layer" is
  garbled — ALBERT-xxlarge is 12 layers (235M) and the 18× figure is ALBERT-base vs.
  BERT-base parameter count; the "12× more layers" claim is not in the paper.
- **LLM-scale canon missing**, which is exactly what was asked for: *Mixture-of-
  Recursions* (Bae et al. 2507.10524; per-token recursion depth via routing, the 2025
  successor to Relaxed Recursive Transformers); *Reasoning with Latent Thoughts: On the
  Power of Looped Transformers* (Saunshi et al. ICLR 2025, 2502.17416 — the case that
  loops buy reasoning depth at fixed parameters, with a looped-model scaling analysis);
  *Ouro / Scaling Latent Reasoning via Looped Language Models* (2510.25741, 1.4B–2.6B
  looped LMs pretrained at scale); *Looped Transformers Are Better at Learning Learning
  Algorithms* (Yang et al. 2311.12424); CoTFormer (2310.10845); Inner Thinking
  Transformer (2502.13842); Subformer (Reid et al. 2021) and Takase & Kiyono's "Lessons
  on parameter sharing across layers" (2104.06022) for the ALBERT-lineage sharing
  patterns (cycle / sequence / cycle-rev); Tiny Recursive Models (2510.04871); Huginn is
  the Geiping model already cited. The report's "limited adoption in production" claim
  should be checked against these 2025 releases.
- Numbers as the agent reported; unverified.

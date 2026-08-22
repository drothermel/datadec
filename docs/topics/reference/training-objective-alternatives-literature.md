# Alternatives and modifications to cross-entropy for LLM training — literature reference

**Kind:** reference (accumulator for training-side objectives: reweighted / selected-token
CE, alternative scoring rules and probability families, and non-next-token objectives).
Entries are dated. Characterizations are a SciSpace agent's; identifiers unverified
unless from the parsed PDFs. Sibling: `loss-alternative-metrics-literature.md` (the
evaluation-side question), `token-level-literature.md`, `distillation-literature.md`
(reverse-KL / on-policy objectives), `synthetic-data-literature.md`.

Why it matters here: the token-reweighting family is the *training-side mirror* of TOK's
per-token movement question (which tokens does training actually move, and should it);
the "model-capability continuum" result says the right objective depends on where on the
size ladder you are — a DataDecide-shaped claim; and any retrain substrate
(DataDecide-dense) that wants an objective arm has its menu here.

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-alts-to-CE-loss-llm-pretraining-agent-artifacts-zip_e225323f-208b-41a7-abd2-1ce72571fd27_1787425213/` — the report (md + LaTeX/PDF), 10 papers parsed with figures
and page renders, 9 further full texts, 37 mostly off-topic arXiv downloads, and 15
search CSVs (three merged with extracted objective/mechanism columns). **`INDEX.md`
inside the folder is the file-level index** with the missing-canon list.

---

## 2026-08-22 — SciSpace deep review (undated, ~2026)

**Danielle's prompt (verbatim):**

> I want you to do a literature review of modifications to, or alternatives to Cross
> Entropy Loss for LLM pretraining or finetuning. Examples include completely differently
> training objectives, as well as modifications such as reweighting the loss from certain
> tokens to be more important. I do not want you to include any commonly studied RLHF
> methods, unless they are applied to pretraining.

**What survives, by family (RLHF-adjacent entries removed).**

- *Token reweighting and selection within CE.* By difficulty / entropy: **MiLe**
  (2024, Findings NAACL; scale loss by predictive entropy; 468M–6.7B on the Pile),
  **TALR** (2509.20758; w ∝ p(x)^(1/τ), a curriculum that downweights hard tokens),
  **RFT** (2412.14780; reasoning vs. boilerplate tokens by relative loss), **IR-DRO**
  (2402.14270; keep moderately-high-loss samples, drop the highest as noise). By
  frequency / information: **Power-Law Decay Loss** (2505.16900), **Rho-1 / selective LM**
  (2404.07965; reference-model excess-loss scoring, train on the top tokens; 15B
  OpenWebMath and 80B general tokens), **ESLM** (2505.19893; value-at-risk thresholding
  per batch, recovers CVaR minimization; GPT-2 pretraining FLOP savings). By gradient
  utility: **VCORE** (2510.27462; closed-form Gibbs weights from a one-backward probe,
  variance-controlled), **ScaleGrad** (2021; gradient edits to favour novel tokens).
  Domain-level: **Velocitune** (2411.14318; weight domains by learning velocity),
  tDRO (2408.10613), XDoGE (2512.10545), online sample reweighting (Zhao et al. 2024).
- *Alternative scoring rules and probability families.* **"Beyond Log Likelihood"**
  (Li et al. 2510.00526): the family f_α(p) = (1 − p^α)/α with NLL at α→0; a
  *model-capability continuum* — prior-leaning objectives (α = 1, i.e. −p) beat NLL by
  up to 16% at the model-strong end, NLL wins at the model-weak end. **MixCE**
  (2305.16958; forward + reverse CE). **Strictly proper scoring rules** (Shao et al.
  2405.18906; Brier / spherical at token level with smoothing; +3 BLEU / ROUGE on
  LLaMA-7B). CV-inspired: focal, Lovász, Dice (Cambrin et al. 2409.13641; +42% exact
  match on math fine-tuning, 3–7B models). Contrastive token learning for degeneration
  (2205.02517). Supervised contrastive fine-tuning (Gunel et al. ICLR 2021; Moukafih et
  al. 2022) — classification fine-tuning, not LM training.
- *Beyond next-token prediction.* **Multi-token prediction** (Gloeckle et al.
  2404.19737; implicitly upweights "choice-point" tokens; gains on code; up to 13B),
  MTP curricula (2505.22757), **patch-level training** (2407.12665; predict K-token
  patches, ~50% cost reduction at matched loss), "filling the mutual-information gap"
  (2511.00198). **Concept-level objectives** (Iyer et al. 2601.11791; surface forms of
  one concept count as correct). Denoising: UL2 mixture-of-denoisers, SpacTor-T5
  (2401.13160), continuous-paragraph-denoise diffusion LMs, RTS/SLM structural
  objectives (2309.08272). Embedding-space: **LLM-JEPA** (2509.14252), Focused
  Transformer contrastive KV training (2307.03170).

**Intake notes.**

- RLHF-adjacent methods the prompt excluded were included anyway — ASPO, λ-GRPO, UFT,
  sequence-level CPO, GRACE; dropped above.
- Missing canon: label smoothing / confidence penalty; unlikelihood (Welleck et al.
  2019); DeepSeek-V3's MTP at scale (2412.19437); fill-in-the-middle (2207.14255);
  instruction-loss masking vs. loss-over-instructions (2405.14394 — the
  fine-tuning-side "which tokens count" question); z-loss and auxiliary losses; latent
  reasoning objectives; Byte Latent Transformer. The 2510.00526 capability-continuum
  paper is the single most decision-relevant entry and deserves a read.
- For this repo: the token-selection family (Rho-1, MiLe, ESLM, VCORE) and TOK's
  per-token movement are the same question from opposite sides — see
  `../../potential-projs/token-movement.md` §4; the capability-continuum result is a
  size-ladder claim testable on DataDecide-dense if an objective arm is ever added.
- Numbers as the agent reported; unverified.

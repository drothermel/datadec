# Reinitialization, vocabulary swaps, and interface transfer — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

> **Danielle interest flag (2026-08-22).** Danielle is specifically interested in what happened
> with this literature after her early embedding-reset result, and in whether there are
> places she could contribute to the direction. This is the only reference topic carrying
> such a flag. Acting on it would mean: (1) a targeted literature pass on the current state
> — tokenizer/vocabulary transfer and its continued-pretraining costs, body-frozen interface
> transfer, reset-and-distill methods (ITER-style) in and beyond RL, and the
> basin-preserving-vs-determining reading; (2) a gap list; (3) if a gap is real, a staging
> topic or project doc. Until then, keep accumulating references here.

Why it matters here: resets are "ways to jump to different spots in the loss landscape"
(Danielle); the embedding-reset result is an early Danielle hypothesis that became standard
practice; and the distinction between basin-preserving interface resets and
basin-determining early deficits organizes the whole reset family (shrink-and-perturb,
continual backprop, embedding resets, ITER's distill-into-fresh-network).

---

## 2026-08-18 — the reset-the-embeddings lineage (from the Research Trajectory page)

Prompt context (Danielle): before this line of work, an earlier project asked whether a
pretrained LM could be transferred to non-language tasks, and specifically whether changing
tokenization/vocabulary — then handled by training from scratch or complex vocabulary
merging — could be done by resetting the input (and maybe output) layer and continuing
training on the original dataset. Very preliminary results said yes: a tiny fraction of the
full training length met or exceeded previous performance with the new tokenization,
though not strong enough to believe. What is the lineage, and how is it handled now?

"**You were right, and it's now the default.**"

- Artetxe et al. 2020, *On the Cross-lingual Transferability of Monolingual
  Representations* — "transferred a monolingual model to new languages by retraining *only*
  the embedding layer against a frozen body — your hypothesis in cross-lingual form."
- *How to Do a Vocab Swap? A Study of Embedding Replacement for Pre-trained Transformers* —
  "it's possible to re-learn embeddings for a swapped vocabulary even from naive
  initialization — even when the new vocabulary comes from a different source language —
  with smart initializations helping."
- Standard industrial practice: vocabulary extension plus continued pretraining — Chinese
  LLaMA-2 (*Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca*), Korean
  SOLAR/Phi-2 (*Efficient and Effective Vocabulary Expansion Towards Multilingual LLMs*),
  Arabic bilingual LLaMA (ALLaM) — "or full tokenizer replacement with complete embedding
  reinitialization."
- The "strange manipulation" school "mutated into initialization schemes for your recipe
  rather than alternatives to it": WECHSEL; FVT (*Fast Vocabulary Transfer for Language
  Model Compression*); FOCUS ("new-token embeddings as sparse combinations of semantically
  overlapping old tokens"); ZeTT (*Zero-Shot Tokenizer Transfer*, "a hypernetwork that
  predicts embedding matrices for arbitrary new tokenizers"). "All exist to lower the
  *starting* loss — but replacing or significantly altering the tokenizer still requires
  extensive continued pretraining to avoid degradation… full recovery after a tokenizer
  change demands on the order of ≥50B tokens of continued training" (*Getting the Most Out
  of Your Tokenizer for Pre-training and Domain Adaptation*).
- "So the field's settled answer is exactly your preliminary finding: reset-and-retrain does
  the real work at a small fraction of from-scratch cost; clever initialization just
  shortens the bill."
- LM → non-language transfer: Lu et al. 2021, *Frozen Pretrained Transformers as Universal
  Computation Engines* — "froze the body and swapped input/output layers for non-language
  tasks. You were independently converging on both."

**The landscape reading.** "Resetting the embedding layer is a large jump in parameter
space that *doesn't leave the basin*, because the body — where the basin identity lives —
is untouched. The embeddings just re-solve a matching problem into a stable representation,
which is also why this reset doesn't re-trigger critical-period damage: the period concerns
the formation of the body's representation, not the interface to it. Interface resets are
basin-preserving; early-training deficits are basin-*determining*. Same operation, opposite
regimes."

**ITER.** Igl, Farquhar, Luketina, Böhmer & Whiteson, *Transient Non-Stationarity and
Generalisation in Deep Reinforcement Learning* (ICLR 2021, arXiv 2006.05826) — "neural
networks exhibit a memory effect where transient non-stationarities permanently impact the
latent representation and adversely affect generalisation — so ITER augments RL training by
repeatedly distilling the current policy into a freshly initialised network, which thereby
experiences less non-stationarity, improving generalisation on ProcGen and Multiroom."
"ITER's fix is mechanistically the most interesting one in the whole reset family, because
it's the only one that separates *function* from *trajectory*… Distillation into a fresh
network transfers the *behavior* while discarding the *parameters entirely*… That ITER's
students generalize *better* than their teachers is direct evidence that the damage lives
in parameter-space history rather than in the function — you can launder the trajectory."

---

## 2026-08-18 — Danielle's own entry in the body-frozen transfer thread (verified: her paper)

- Rothermel, Li, Rocktäschel & Foerster, *Don't Sweep your Learning Rate under the Rug: A
  Closer Look at Cross-modal Transfer of Pretrained Transformers* (2021, arXiv 2107.12460).
  "Lu et al.'s claim that frozen pretrained transformers match or beat training from scratch
  across modalities was an artifact of not tuning learning rates; with proper tuning,
  pretrained transformers do outperform or match scratch on every task — but only when the
  entire model is fine-tuned, with frozen variants often greatly lagging, in direct
  contradiction to the original findings, and the genuine positive result that on
  CIFAR10-LRA, fine-tuning the full pretrained model beats training from scratch by a large
  margin. Reported, notably, with error bounds across 3 seeds."
- Its substantive stance, as read back: "*frozen interfaces lag; the body must be updated;
  but transfer through the body is real*… what a pretrained model carries is genuine, but
  accessing it without weight updates is limited, and apparent no-update successes deserve
  suspicion." The embedding-reset project is "the same decomposition, from the other side"
  (interfaces are cheap, the body is where the value lives).
- "The modern literature landed exactly on your 2021 dividing line — prompt-based adaptation
  excels few-shot but plateaus as data grows while fine-tuning keeps going — which means the
  Bornschein–Lyle paper (*Fine-Tuned In-Context Learners for Efficient Adaptation*) is the
  direct descendant of yours."

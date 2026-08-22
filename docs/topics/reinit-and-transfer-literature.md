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
> topic or project doc. **Step (1) and (2) done 2026-08-22** — see the literature-pass section at the end of this doc
> and the full report at
> `~/drotherm/data/.claude/datadec/2026-08-22/0031-reinit-transfer-litpass.md`. Step (3): the
> gaps were split into six staging topics on 2026-08-22 (`reset-recovery-dynamics`,
> `interface-reset-basin-test`, `reset-and-plasticity`, `frozen-body-transfer-audit`,
> `reset-response-stage-probe`, `reset-effects-many-seed-lm`); promotion decisions are
> deferred to the end-of-intake review.

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
  of Your Tokenizer for Pre-training and Domain Adaptation*). **Correction (lit pass, 2026-08-22):** Dagan,
  Synnaeve & Rozière (ICML 2024, arXiv 2402.01035) say that with *more than* 50B tokens of
  fine-tuning one *can profitably specialize* the tokenizer — a statement about when
  specialization pays off, not a recovery floor. The seeded framing inverts it.
- "So the field's settled answer is exactly your preliminary finding: reset-and-retrain does
  the real work at a small fraction of from-scratch cost; clever initialization just
  shortens the bill."
- LM → non-language transfer: Lu et al. 2021, *Pretrained Transformers as Universal
  Computation Engines* (arXiv 2103.05247; "Frozen Pretrained Transformer" is the model name) — "froze the body and swapped input/output layers for non-language
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

---

## 2026-08-22 — targeted literature pass (Opus subagent; full report: `~/drotherm/data/.claude/datadec/2026-08-22/0031-reinit-transfer-litpass.md`)

All citations below were retrieved by the subagent (arXiv IDs given); verdicts rest on
abstracts and paper pages, not full PDFs, and no forward-citation sweep was run.

**Verification of the seeded claims.** All verified except: Dagan et al. (arXiv 2402.01035)
— the ">=50B tokens to recover" reading is inverted (see correction above); ALLaM (arXiv
2407.15390) — mechanism not confirmed; FVT — method confirmed, exact citation not; the
"Vocab Swap" study is an OpenReview submission (MsjB2ohCJO1) with no confirmed venue; Lu et
al.'s title is *Pretrained Transformers as Universal Computation Engines*. Rothermel et al.
2021 (arXiv 2107.12460, ICML 2021 SSL workshop) confirmed as stated. The "interface resets
are basin-preserving / early deficits are basin-determining" reading is an interpretive
frame with no paper behind it — its absence is gap G3.

**(a) Tokenizer / vocabulary transfer — "converged on initialization quality; recovery
dynamics measured only as a byproduct."**
- ZeTT (Minixhofer, Ponti, Vulić; NeurIPS 2024; arXiv 2405.07883): prior init heuristics
  are near-chance in true zero-shot transfer.
- OMP tokenizer transplantation (Goddard & Fernandes Neto; arXiv 2506.06607): training-free,
  beats WECHSEL/FOCUS/ZeTT on zero-shot preservation.
- MATT (Haltiuk & Smywinski-Pohl; arXiv 2510.21954): distills source→target attention
  patterns as warm-up — the first method treating the body's dynamics, not just embedding
  geometry, as the target.
- *Beyond Initialization Loss* (arXiv 2608.03494, 2026): >20 init strategies on a 30B MoE;
  best init gives a 6× CPT reduction; **input and output embeddings have distinct optimal
  inits**; init loss/BPB are unreliable predictors of convergence, ~50 CPT steps are
  reliable.
- *Teaching Old Tokenizers New Words* (Purason et al.; EACL 2026 Findings; arXiv
  2512.03989); Dobler & de Melo academic-budget adaptation (arXiv 2408.15793);
  convex-hull initialization (arXiv 2407.05841); EEVE (arXiv 2402.14714 — proficiency
  within 2B tokens, "explicitly contra… trillions of training tokens"); Learned Embedding
  Propagation (arXiv 2412.21140 — CPT of embeddings "redistribute[s] existing language
  knowledge among new tokens").
- Reading: "continued-pretraining cost estimates range across three orders of magnitude
  (500 steps to >50B tokens) with no controlled study reconciling them."

**(b) Body-frozen transfer — "the 'why does it work' question is largely unanswered."**
- Lu et al. (arXiv 2103.05247); Rothermel et al. (arXiv 2107.12460) — "despite being the
  load-bearing rebuttal, this line of critique appears to have been under-absorbed — no
  2022–2026 paper systematically re-audits frozen-vs-finetuned claims for LR-tuning
  asymmetry."
- Frozen-LM multimodal (arXiv 2106.13884); X-Fusion (ICCV 2025; arXiv 2504.20996); Decoding
  PDEs (arXiv 2510.05278); *Frozen in Time* (arXiv 2508.18130) — frozen *random* dynamics as
  a reservoir: "a live confound that most frozen-body transfer papers do not rule out."

**(c) Reset-based methods — "large and healthy but almost entirely RL/vision."**
- Plasticity injection (Nikishin et al.; arXiv 2305.15555) — "the most directly borrowable
  *instrument*": if injection helps, plasticity was the binding constraint.
- Reset & Distill (Ahn et al.; arXiv 2403.05066); *When Does Re-initialization Work?*
  (Zaidi, Berariu, Kim, Bornschein, Clopath, Teh, Pascanu; arXiv 2206.10011) — >15,000
  vision models: reinit helps without other regularization, little once regularization is
  tuned, significantly under label noise.
- Plasticity-loss survey in RL (arXiv 2411.04832): last-layer resets are standard; the
  "concentrated in last layers" belief is weakly evidenced.
- *Can Scale Save Us From Plasticity Loss in LLMs?* (Hernandez-Garcia, Figliolia, Millidge;
  arXiv 2606.24752, June 2026): 5M–314M params; plasticity loss follows a sublinear scaling
  law; scale delays but does not prevent it, in continual *and stationary* settings.
- Spectral collapse (arXiv 2509.22335); activation design (arXiv 2509.22562); calibrated
  partial resets (arXiv 2607.24996).

**(d) Resets in landscape / connectivity terms — "thinnest sub-thread; the instruments
exist but have not been pointed at resets."**
- Layer-wise LMC (arXiv 2307.06966): per-layer barriers are insignificant relative to the
  full-model barrier; **middle layers create barriers** — predicts interface-only
  perturbations are near-barrier-free.
- LMC of MoEs (arXiv 2509.11348); *Landscaping LMC* (arXiv 2406.16300); *The Butterfly
  Effect* (arXiv 2506.13234 — trajectories highly sensitive to initial conditions, so reset
  studies need many seeds); Fisher-guided selective forgetting (arXiv 2502.00802);
  representation-plasticity timeline in LLMs (arXiv 2410.06225).

**(e) Small scale, many seeds.**
- **PolyPythias** (van der Wal, Lesci, Muller-Eberstein, Saphra, Schoelkopf, Zuidema,
  Biderman; ICLR 2025; arXiv 2503.09543): 50 pretraining runs, 9 new seeds × 5 sizes
  (14M–410M), ~7,000 checkpoints — "this is the substrate."
- Critical periods in LM finetuning (TACL, doi:10.1162/tacl_a_00725); *Smooth Scaling Laws
  Hide Stepwise Token Learning* (arXiv 2606.29858).

**Gap list (ranked by confidence that the gap is real).**
1. **G1 — Recovery-cost curve for an embedding reset** vs. scale, training stage, seed.
   Estimates span 500 steps to "2B tokens"; no controlled curve. Cost: small runs.
2. **G2 — Input-vs-output embedding reset asymmetry, explained.** 2608.03494 observes it as
   a tuning finding; nobody isolates head-only vs. input-only vs. both (weight tying is an
   uncontrolled confound). Cost: small runs.
3. **G3 — Is an interface reset basin-preserving?** Layer-wise LMC predicts yes; nobody has
   reset an interface and measured the barrier to the pre-reset solution. Cost: forward
   passes on existing checkpoints + short recovery runs. "The single best-shaped question
   for Danielle's program."
4. **G4 — Does an embedding reset restore plasticity, or is it orthogonal?** The plasticity
   literature resets body layers; the tokenizer literature never asks. Plasticity injection
   is the ready diagnostic. Cost: small runs on existing checkpoints.
5. **G5 — LR-tuning asymmetry as an unaudited confound in modern frozen-body claims**
   (downstream of 2107.12460), with the reservoir null (random frozen body). Cost: small
   runs. *Medium-high; rests on keyword absence, not a citation sweep.*
6. **G6 — How much of what the body carries can a frozen interface reach?** (downstream of
   2107.12460) — reframe the frozen/finetuned gap as an elicitation-ceiling measurement
   with modern probes. Cost: forward passes + light probe training. *Medium-high.*
7. **G7 — Reset-response as a critical-period instrument**: recovery cost from an interface
   reset at step t as a stage probe, on PolyPythias. *Medium.*
8. **G8 — Many-seed replication of reset effects in LMs** (2206.10011's "disappears under
   tuned regularization" is unchecked in LMs). *Medium.*
9. **G9 — Which layers actually need resetting in an LM?** *Medium.*
10. **G10 — Does init quality matter once you train long enough** (convergence crossover)?
    *Lower; may exist.*

**Caveats (the subagent's).** No full-PDF reads; no forward-citation sweep (so G5/G6 are
the weakest "nobody followed up" claims — run the 2107.12460 forward graph before building
on them); search skewed to recent arXiv HTML, so 2021–2022 workshop work is
under-represented; several 2026 IDs have had little scrutiny.

---

## 2026-08-18 — model stitching as the measurement form of the embedding-reset experiment

"Model stitching (Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021) is the behavioral
version [of functional identifiability]: if a trained adapter layer lets model A's bottom
half drive model B's top half at low penalty, they're functionally interchangeable at that
depth — and note this is *literally your embedding-reset experiment* as a measurement
rather than a method." Relevant to the interface-reset staging topics.

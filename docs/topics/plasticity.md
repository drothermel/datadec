# Plasticity — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts; not waiting to be
promoted or absorbed). Entries are dated and quoted close to verbatim; related-work claims
are unverified unless a citation is given.

Why it matters here: plasticity is the continual-learning name for a property every project
in this portfolio touches — whether a model can still learn from the next data, as a function
of where it is in training. It appears as a response-vector component in functional
featurization, as the "modular plasticity" question in the MoE projects, as an endogenous
non-stationarity in the accounting framing, and as the critical-period reading of data-order
effects.

---

## 2026-08-18 — the plasticity thread (from the Research Trajectory page)

Prompt context (Danielle): a previous project asked whether features of a loss curve could
estimate or predict properties relevant to "success," starting from Sutton's student and
Clare Lyle's plasticity work at CIFAR-10 scale and ending at LLM pretraining and the
multi-power law.

**Papers**

- Shibhansh Dohare et al., *Loss of plasticity in deep continual learning* (Nature 2024;
  earlier arXiv "Maintaining Plasticity…", 2306.13812). "Standard deep-learning methods
  gradually lose plasticity in continual-learning settings until they learn no better than a
  shallow network, demonstrated on ImageNet (repurposed as task sequences) and RL problems,
  and propose continual backpropagation (selectively reinitializing dormant/unuseful units
  during training). The incremental-CIFAR experiments… are in their codebase."
- Clare Lyle et al., *Understanding Plasticity in Neural Networks* (ICML 2023, arXiv
  2303.01486). "A systematic empirical analysis finding plasticity loss is deeply connected
  to changes in loss-landscape curvature, often occurring without saturated units."
- Lyle et al., *Disentangling the Causes of Plasticity Loss in Neural Networks* (arXiv
  2402.18762). Follow-up.
- J. Fernando Hernandez-Garcia et al., *Can Scale Save Us From Plasticity Loss in Large
  Language Models?* "Papers now ask whether plasticity loss appears in LLM-scale training…
  which matters for continual pretraining and mid-training regimes."

**Thoughts**

- "In the plasticity literature, the latent quantity is future trainability — can the network
  still reduce loss on the *next* task? Lyle's work explicitly hunts for cheap-to-compute
  training statistics (curvature, feature rank, dead units, weight norm) that correlate with
  or cause that ability." → Candidate components for a branch response vector or probe
  battery (see `../potential-projs/functional-featurization.md` FUNC-4,
  `checkpoint-tomography.md`).
- "The plasticity answer so far [to 'what low-dimensional summary of training dynamics
  forecasts a capability'] is 'no single statistic — curvature comes closest' (Lyle)."
- Methodological flavor: "plasticity is mechanistic/causal — intervening on curvature,
  resets, normalization to see what restores learning" — at "CIFAR/ImageNet/Atari with many
  cheap seeds." The same intervention-with-replicates style is what the tiny-scale program
  proposes for LM training.
- Regime: "plasticity work assumes an explicitly non-stationary data stream (task sequences,
  RL bootstrapping)."

---

## 2026-08-18 — the plasticity group's move into ICL-vs-fine-tuning

"*Fine-Tuned In-Context Learners for Efficient Adaptation* is by Bornschein, Clare Lyle,
Razvan Pascanu et al. — the plasticity crowd literally moved into 'ICL vs fine-tuning as
adaptation' territory." See `icl-as-posttraining.md`.

---

## 2026-08-18 — warm-starting, and whether plasticity mechanisms explain the stationary case

**Papers**
- Ash & Adams, *On Warm-Starting Neural Network Training* (NeurIPS 2020). Stationary
  incremental setting; warm-started models generalize worse than re-initialized ones at
  similar training loss; shrink-and-perturb as the fix; diagnosis was a gradient-norm
  imbalance between old and new samples — "a symptom, not a mechanism."
- *DASH: Warm-Starting Neural Network Training in Stationary Settings without Loss of
  Plasticity* (NeurIPS 2024). Theory for the stationary case: "the model has *memorized
  noise* from the small early dataset, and shrinking should be direction-aware… Notably they
  argue non-stationarity-motivated plasticity fixes are ineffective in the stationary
  setting — i.e., the Dohare/Lyle mechanisms may *not* be the explanation here."
- *What Can Grokking Teach Us About Learning Under Non-Stationarity* (2025). "Re-warming the
  effective learning rate closes the generalization gap, and a higher relative number of
  dead units does not predict a large warm-starting gap."

**Thoughts**
- Three live hypotheses for the warm-starting gap — noise memorization, effective learning
  rate, classic plasticity mechanisms — plus mundane candidates (optimizer state reset,
  weight decay, warmup, AdamW). "Nobody has run the factorial that adjudicates them on the
  original benchmark." Design in `warmstarting-decomposition.md`.
- The diagnostic panel to log at matched training loss: curvature, feature rank, dead
  units, weight norm, gradient-norm ratio.

---

## 2026-08-18 — the origin of the plasticity literature (from the critical-periods discussion)

"'Information Plasticity decreases and the network can no longer adapt' (Achille, Rovere &
Soatto, 2017–2019) *is* plasticity loss, five years before Dohare's Nature paper, with the
Fisher trace as the diagnostic — and since the Fisher matrix approximates the Hessian for
log-loss, Lyle's later 'plasticity loss is about loss-landscape curvature' finding is nearly
a rediscovery in different coordinates." Add the Fisher trace to the diagnostic panel. See
[critical-periods.md](critical-periods.md).

---

## 2026-08-22 — additions from the reinit literature pass (arXiv IDs retrieved)

- Hernandez-Garcia, Figliolia, Millidge, *Can Scale Save Us From Plasticity Loss in Large
  Language Models?* (arXiv 2606.24752, 2026): 5M–314M; sublinear scaling law; scale delays
  but does not prevent plasticity loss, in continual and stationary settings.
- Nikishin et al., *Deep RL with Plasticity Injection* (arXiv 2305.15555) — a diagnostic: if
  injection helps, plasticity was binding.
- Zaidi et al., *When Does Re-initialization Work?* (arXiv 2206.10011) — >15,000 vision
  models; reinit helps without regularization, little once tuned, significantly under label
  noise.
- Spectral collapse (arXiv 2509.22335); activation-function design (arXiv 2509.22562);
  calibrated partial resets (arXiv 2607.24996); plasticity-loss survey in RL (arXiv
  2411.04832); Fisher-guided selective forgetting (arXiv 2502.00802).

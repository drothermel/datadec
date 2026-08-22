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

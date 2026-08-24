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
  battery (see `../../potential-projs/functional-featurization.md` FUNC-4,
  `../staging/checkpoint-tomography.md`).
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
adaptation' territory." See `../../potential-projs/icl-elicitability.md`.

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
  original benchmark." Design in `../../potential-projs/intervention-grid.md`.
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

## 2026-08-24 — NotebookLM Continual Learning notebook (CoLLAs talks + her Roam notes; assembled ≥2024-10)

Danielle supplied the outputs of a NotebookLM notebook on continual learning
(bundle: `nblm-continual-learning-notebook.md` in the 2026-08-24 intake bundle;
data table + source list + two synthesis reports). Sources are mainly CoLLAs
2022/2023 talk recordings — Sutton (CBP), Lyle (plasticity mechanisms), Van Roy
(CCRL), Bing Liu (CL/OOD unification), Rish (CL at scale), Aljundi (CL with
pretrained models), Larochelle (knowledge mobilization) — plus a James Harrison
learned-optimization talk and **Danielle's own Roam daily-pages export
(2024-10-02)** as source [2], so table rows citing [2] partially reflect her own
2024 notes (e.g. "Resetting the network improves plasticity more than any other
intervention; Adam introduces 'magic' to optimization"). **Reliability caveat:**
NotebookLM's own inaccuracy warning applies, and the first synthesis report is
visibly transcript-garbled ("Clara Lyle", "Ashton Adams" = Ash & Adams, "Mammal"
= MAML, "itbid" = IDBD, "Socar ICML 2023" plausibly = Sokar et al. dormant
neurons, "Vastava & Danny Tarlow" plausibly = Shrivastava & Tarlow repo-level
prompt augmentation; identifications inferred, unverified). The second ("Deep
Research") report is cleaner. Talk-level detail beyond this record's existing
paper anchors (Dohare/CBP, Lyle's ICML/disentangling papers, Ash & Adams — all
already above):

- **Sutton talk (CoLLAs 2022):** CBP's utility measure (weight magnitude ×
  activity × plasticity), replacement rate ~1e-6, dead-unit percentage and
  effective rank as the maintained quantities; Slippery Ant — standard PPO
  collapses under changing friction, Continual PPO (CBP) keeps running.
- **Lyle talk (CoLLAs 2023):** probe-task loss as the plasticity definition;
  label-shuffle spikes in dead units becoming unrecoverable; Adam epsilon/moment
  tuning removes specific post-shuffle pathologies but only delays plasticity
  loss; GD trajectories vs random walks — ill-conditioning and gradient
  collinearity as the trained-in pathology; width + LayerNorm help but don't
  solve; the two-hot categorical trick (bounded gradients → less plasticity
  loss); CReLU; regress-features-to-initialization regularization; resetting
  dead units / last layer as the strongest single intervention.
- **Van Roy talk (CoLLAs 2023):** CL as computationally constrained RL —
  maximize average reward under information capacity C; ~C log C parameters
  needed to hold C bits trainable by SGD; prediction error decomposes into
  *informational* error (forgetting = relevant history lost) vs *inferential*
  error (implasticity = current observations underused); the idealized Bayesian
  gold standard P*_t with KL distance; IDBD step-size adaptation needs a
  capacity correction term; **L2-toward-initial-weights beat CBP across their
  benchmarks at lower complexity** — a directly actionable baseline claim for
  the reset staging topics; average reward keeps growing with capacity in
  complex CL environments (no plateau).
- **CL evaluation cluster (tutorial parts I–II, Aljundi):** prequential
  test-then-train evaluation; anytime accuracy; the **stability gap** (worst-case
  drop right after drift; EMA-of-weights evaluation model reduces it);
  compute-bounded CL where plain Experience Replay beats complex methods by
  processing more data per budget; realistic-stream benchmarks (CLEAR,
  Wild-Time, CLOC, CORe50) vs shuffled class-incremental setups;
  **representation forgetting is much milder than head forgetting** (linear
  probes; non-linear with a large initial drop) — convergent with the
  frozen-body/reset staging topics' premise; "no adaptation" + LDA head as a
  hard-to-beat baseline (+15% over NCM); update-only-3%-of-first-MLP-layer
  preserving CLIP holdout within 0.1%.
- **Scale cluster (Rish talk):** larger pretrained models forget less;
  self-supervised pretraining more CL-robust than supervised; continual
  pretraining (Pile→SlimPajama) with LR re-warming and soft gradient masking
  beats from-scratch; BNSL for non-monotonic scaling shapes; Chinchilla vs
  over-train-small ("Llama strategy") framed as data-optimal vs
  inference-optimal.
- **Learned-optimization block (Harrison talk; adjacent material from the same
  notebook):** VeLO (82 tasks, 4,000 TPU-months; ≥4× tuned Adam on >50% of
  tasks but lags on 8B+ LMs and fails on RL/PPO); optimization framed as a
  POMDP; ES/zeroth-order estimators for chaotic unrolled graphs; stability-at-
  initialization biases and magnitude normalization for meta-generalization.
  Related to the whetstone optimizer-stack world; no dedicated accumulator —
  lives here with the notebook.
- **Knowledge-mobilization block (Larochelle talk):** Head-to-Toe probing
  (all-layer linear probe closes most of the fine-tuning gap at <1% FLOPs/
  storage); URT (attention over multiple pretrained backbones); repo-level
  prompt augmentation for Codex (out-of-file context; garbled attribution).

Drift-taxonomy and evaluation vocabulary routed to
`nonstationarity-accounting.md` (same intake). All claims talk-transcript-level,
agent-distilled, unverified; no arXiv IDs supplied anywhere in the notebook.

## 2026-08-24 — the LM-pretraining plasticity pair (NotebookLM pretraining notebook)

Two sources from the 11-paper pretraining notebook (bundle:
`nblm-pretraining-dynamics-notebook.md`; main entry in
`schedules-and-annealing-literature.md`; no IDs supplied, unverified) answer
part of Danielle's standing where-did-plasticity-go-for-LMs interest — the
plasticity thread has arrived in LM pretraining hyperparameter space:

- **"Overtrained Language Models Are Harder to Fine-Tune"** — defines
  **catastrophic overtraining**: OLMo models (15M–90M, ≤128B tokens) pretrained
  on more tokens become progressively more sensitive to parameter
  transformations; under task misalignment, fine-tuning an overtrained model
  degrades both in-domain and OOD performance. Pretraining-token budget is not
  monotonically beneficial for adaptability.
- **"Weight Decay Improves Language Model Plasticity"** — Llama-2/OLMo-2 to 4B:
  higher pretraining weight decay improves fine-tuning accuracy and plasticity;
  mechanism reported as linearly-separated representations and reduced
  pseudo-rank of attention matrices. The headline for HP selection: choosing
  pretraining HPs on pretraining loss alone does not yield the best post-trained
  model — the same pretrain-vs-posttrain objective split the CL notebook's
  stability–plasticity ledger formalizes, now at the HP-selection level.

Related from the same notebook: the CPT paper's "loss potential" (initial loss
level dictating adaptability) as a plasticity-adjacent quantity, and the
CL-notebook convergence that these are the LM-scale descendants of the
warm-starting / Ash & Adams line already anchored above.

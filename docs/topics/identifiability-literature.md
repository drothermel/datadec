# Causal representation learning and identifiability — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: "the formal language your comparability problem has been missing."
Path-dependence (critical periods, warm-start scars, ITER's memory effect, strategy-distinct
basins) is possible exactly when the learning problem is non-identifiable; the basin
literature's permutation-alignment and neuron-identifiability work is an attempt to
quotient out known symmetries; interventional designs are identification strategies; and
the Platonic Representation Hypothesis is an empirical claim that identifiability improves
with scale.

---

## 2026-08-18 — identifiability as the frame for path-dependence (from the Research Trajectory page)

**Papers**
- Schölkopf et al., *Toward Causal Representation Learning*.
- Hyvärinen (and co-authors), *Nonlinear Independent Component Analysis: Existence and
  Uniqueness Results*; Khemakhem et al., iVAE (*Variational Autoencoders and Nonlinear ICA:
  A Unifying Framework*).
- *Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability* (2026)
  — "the consistent assignment of features to neurons across random seeds, with
  symmetry-breaking mechanisms characterizing which functions neurons can implement."
- Huh et al. 2024, *Position: The Platonic Representation Hypothesis* — "representations of
  independently trained models converge as scale and data grow."

**The core question, and the restatement**
- "Under what conditions does the data distribution determine the learned representation
  uniquely, up to a symmetry group?"
- "Path-dependence… is possible *exactly when the learning problem is non-identifiable*:
  multiple functionally-inequivalent solutions attain the same loss, so which one you get
  depends on trajectory. Every 'permanent damage' result in your grid is an empirical
  non-identifiability certificate; your 2021 LR paper and the ELR results are cases where
  apparent path-dependence dissolved, i.e., the solution was more determined than it looked
  once the optimizer was given fair access to it."

**Three consequences**
1. "It makes the basin literature's direction legible: permutation alignment, Git Re-Basin,
   and the 2026 neuron-identifiability framework are all attempts to quotient out the
   *known* symmetry group so that whatever variation remains is *real* underdetermination.
   Your 'metrics are only comparable within a basin' worry becomes: metrics are comparable
   after quotienting by the identified symmetries, and basin-distinctness measures the
   residual non-identifiability."
2. "It gives interventions their proper role: identifiability theory's central result-shape
   is that observational data leaves latents underdetermined while interventional data pins
   them down — and your entire methodology is interventional (deficits, decay branches,
   resets, distillation arms)… each is an intervention chosen to identify structure that
   observational loss curves provably cannot. That's not an analogy; it's the same epistemic
   logic, and citing it turns your experimental-design chapter from 'a collection of clever
   probes' into 'an identification strategy.'"
3. "The thesis's sharpest open conjecture: the Platonic Representation Hypothesis… is an
   empirical claim that *identifiability improves with scale*. If true, path-dependence — and
   with it critical periods, recipe effects on elicitability, all of it — should *wash out*
   as models grow, which would mean your small-scale effects have a scale ceiling; if false,
   the scars persist and pretraining-recipe choices matter at the frontier. Your ladder
   design (tiny transformers → DataDecide → sparse large confirmations) is, almost
   accidentally, the right instrument for measuring how fast the underdetermination closes
   with scale."

---

## 2026-08-18 — the CRL foundations, formalism to operational measurements (from the Research Trajectory page)

Prompt context (Danielle): lay out the causal-representation-learning related work and
concepts that form the foundation of tying the three-paper grid to identifiability.
"Organized from formalism to operational measurements — with honest notes on which links
are established and which are the analogical leaps your project would be making explicit."

**1. Identifiability as "unique up to a symmetry group."**
- Hyvärinen & Pajunen (1999), *Nonlinear ICA: Existence and Uniqueness Results* — latents
  are "*fundamentally non-identifiable* from observational i.i.d. data alone."
- Hyvärinen's time-contrastive learning and auxiliary-variable nonlinear ICA; Khemakhem et
  al., iVAE (2020) — identifiability "*restored* by additional structure: auxiliary
  variables, non-stationarity, or multiple environments, with recovery guaranteed only up
  to a residual group (permutations, elementwise transforms)."
- Grammar: "a learning problem's solution set is characterized by (data, objective,
  function class) up to a group G; anything not pinned down by those is available for the
  *trajectory* to select. 'Path dependence exists' = 'the residual underdetermination beyond
  G is nonempty and SGD's selection within it is history-sensitive.'"

**2. Interventions buy identifiability.**
- Schölkopf et al. 2021, *Toward Causal Representation Learning*; Brehmer et al. 2022,
  *Weakly Supervised Causal Representation Learning* (identifiability from paired
  pre/post-intervention observations); Ahuja et al., *Interventional Causal Representation
  Learning*; von Kügelgen et al. 2021, *Self-Supervised Learning with Data Augmentations
  Provably Isolates Content from Style*; the sparse-mechanism-shift principle.
- The honesty note: "CRL identifies latent factors *in data* from interventions *on data*;
  you're identifying latent structure *in training dynamics* from interventions *on
  training*. Same epistemic logic, different object — your contribution is
  operationalizing the logic at the new level, not applying an existing theorem."

**3. The known symmetry group of networks — what must be quotiented.**
- Sussmann 1992 (single-hidden-layer uniqueness up to permutation/sign); ReLU positive
  rescalings; attention-head permutations. Entezari et al. (LMC modulo permutation); Git
  Re-Basin (weight matching); REPAIR (activation renormalization); the 2026
  neuron-identifiability framework.
- "Barrier measurements come in two flavors, raw and permutation-aligned, and the
  *difference* between them is informative — raw-barrier-high/aligned-barrier-low means
  'same solution class, different parameterization' (benign), while aligned-barrier-high
  means genuine solution-class divergence (the real scar)."

**4. Functional identifiability tests — comparison without touching weights.**
- Roeder, Metz & Kingma 2021, *On Linear Identifiability of Learned Representations* —
  "fit the optimal linear map between two models' representations; the residual is the
  identifiability gap."
- Model stitching: Lenc & Vedaldi 2015; Bansal, Nakkiran & Barak 2021 — "if a trained
  adapter layer lets model A's bottom half drive model B's top half at low penalty, they're
  functionally interchangeable at that depth — and note this is *literally your
  embedding-reset experiment* as a measurement rather than a method."
- CKA (Kornblith et al.) — cheap and scalable, but "can be dominated by a few directions
  and disagree with stitching — so use stitching/linear-map residuals as ground truth and
  CKA as the scalable proxy."
- Platonic Representation Hypothesis (Huh et al. 2024) — "identifiability improves with
  scale" as the conjecture the ladder design can put error bars on.

**5. The selection principle and the timing question.**
- Implicit-bias / simplicity-bias literatures (SGD selects non-uniformly; shortcut learning
  as selection pathology); Juneja et al. as "the NLP existence proof that the classes are
  real and behaviorally distinct." "What's missing from all of it is *when* selection
  happens and what moves it — and that's precisely what your grid measures."
- Commitment-event precedents: Frankle et al. (sibling runs become linearly connected only
  after a critical number of steps); Fort et al., *Deep Learning vs. Kernel Learning* (NTK
  rapid early rotation then stabilization).
- Singular learning theory: the local learning coefficient (Watanabe; Lau, Murfet et al.'s
  estimator), developmental interpretability (*Differentiation and Specialization of
  Attention Heads via the Refined LLC*; *Loss Landscape Degeneracy and Stagewise
  Development in Transformers*) — "a per-checkpoint scalar measuring the *degeneracy* of the
  current solution neighborhood… Degeneracy is the local face of non-identifiability."

**Assembled core claim.** "The critical period is an *identifiability phase transition* —
before commitment, the solution class is underdetermined given data-so-far and
interventions select among classes (measured: sibling divergence under aligned barriers,
high LLC, failed stitching to controls, permanent damage); after commitment, interventions
perturb within a class (measured: alignment and stitching recover, damage is transient,
retunable away à la ELR). Each foundational paper then makes a specific sub-claim — Achille:
input-statistics deficits during the window select a class with permanently different
low-level features; Ash & Adams: data poverty during the window selects a class ill-suited
to the full distribution, and shrink-perturb works by partially re-opening selection; Igl:
non-stationarity drift accumulates class-selection scars, and distillation escapes them
because a fresh student re-runs selection under better data."

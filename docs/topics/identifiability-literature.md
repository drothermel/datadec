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

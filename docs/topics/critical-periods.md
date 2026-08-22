# Critical learning periods — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: "arguably the deepest root of your whole tree" — the critical-period /
Information-Plasticity line anchors the plasticity thread (it predates Dohare by five
years), the warm-starting gap (starting past the Fisher peak), the basin story (the critical
period as the window before basin commitment), the dataset-featurization tool (Task2Vec is
the same Fisher formalism pointed at data), and the LLM-scale data-placement results.

---

## 2026-08-18 — the Soatto–Achille line (from the Research Trajectory page)

Prompt context (Danielle): Soatto and Achille have extensive work on information bottlenecks
and "something about how if you blur a cat's vision as a kitten it will never be able to see
unblurred" — their NN-training equivalent had a name. How does it interact with the rest?

"The name you're forgetting is 'critical learning periods.'"

**Papers**
- Achille, Rovere & Soatto, *Critical Learning Periods in Deep Networks* (ICLR 2019). "Deep
  networks exhibit critical periods during which a temporary stimulus deficit — their
  cataract analog was blurring/downsampling images early in training — can permanently
  impair the development of a skill, with impairment depending on deficit onset and length,
  just as in animal models. The mechanism concept is **Information Plasticity**, measured by
  the Fisher Information of the weights: information rises rapidly early in training, then
  decreases, preventing redistribution of information resources — and deficit sensitivity
  during critical periods correlates strongly with this effective connectivity, with the
  memorization phase followed by a reorganization phase where forgetting is necessary for
  invariance but comes at the price of reduced adaptability later in training."
- *Critical Learning Periods for Multisensory Integration in Deep Networks* — "critical
  periods arise from complex, unstable early transient dynamics which are decisive of the
  final performance and learned representations."
- *Continual Backprop: Stochastic Gradient Descent with Persistent Randomness* — one of the
  interventions that, in this frame, "artificially reopen the period."
- Task2Vec (Achille et al.) — "the Fisher-embedding dataset representation behind the
  alignment coefficients… is Achille's work. Same formalism, pointed at data instead of
  dynamics."

**How it anchors the other threads**
- "**It's the origin of the plasticity literature.** 'Information Plasticity decreases and
  the network can no longer adapt' (2017–2019) *is* plasticity loss, five years before
  Dohare's Nature paper, with the Fisher trace as the diagnostic — and since the Fisher
  matrix approximates the Hessian for log-loss, Lyle's later 'plasticity loss is about
  loss-landscape curvature' finding is nearly a rediscovery in different coordinates."
- "**Warm-starting is a critical-period phenomenon.** Initializing from a converged model
  means starting *past* the Fisher peak, in the low-Information-Plasticity regime — Ash &
  Adams' gap restated. Shrink-and-perturb, ELR re-warming, and continual backprop are all,
  in this frame, interventions that artificially reopen the period."
- "**It grounds the basin story.** Combine with the linear-mode-connectivity fact that
  checkpoints become linearly connected only after early training stabilizes, and 'critical
  period' becomes: *the window before the run commits to a basin/valley*. The river is
  chosen early, while Fisher information is high; after commitment, you travel it. Deficits
  during the window change which river you're in — permanently, because post-commitment
  plasticity won't take you over the walls."
- "**The LLM-scale echo.** The 2025–2026 data-placement results — early exposure shaping
  models more durably than late data, the final-window effects, safety behaviors from
  pretraining resisting post-training removal — are critical-period phenomenology at
  scale, mostly published without the connection drawn."

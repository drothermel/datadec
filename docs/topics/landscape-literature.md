# Loss landscape, river-valley, and mode connectivity — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: every matched-loss comparison in the portfolio assumes the compared
checkpoints sit in comparable regions of the landscape; the river-valley picture is the
mechanism behind annealed readouts and token-level movement; basin tests are the core of
the landscape-geometry project; and the "same basin" condition is why task arithmetic and
model merging work at all.

---

## 2026-08-18 — valley/mountain position and basin identification (from the Research Trajectory page)

Prompt context (Danielle): metrics may only be comparable between models in the same basin.
Is there a way to identify whether a model is in the valley vs. climbing the mountains, and
whether two models are in the same basin?

"There's no settled scalar measure of 'same basin' or 'on the river.' What exists is a
toolkit of pairwise tests: interpolation barrier (raw and permutation-aligned), the
convex-vs-monotone interpolation signature, feature-space connectivity, curve-collapse —
each partial."

**Valley / mountain position**
- Wen et al., *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss
  Landscape View* (arXiv 2410.05192) — "the canonical statement: pretraining loss conjectured
  to resemble a deep valley with a river at its bottom; the sustained high-LR phase drives
  progress along the river and the decay phase drives progress in the mountain direction,
  both critical. Their proposed *measurement* is interpolation-based: the loss on the linear
  interpolation between two stable-phase checkpoints is convex and unimodal (a valley
  cross-section), while between two decay-phase checkpoints it decays smoothly and
  monotonically. That interpolation signature is currently the closest thing to a 'river
  test.' They also offer a mechanism for *why* language loss has this shape: highly
  deterministic tokens (facts, knowledge) contribute the river direction, while uncertain,
  ambiguous tokens create the steep hillsides — which… says the valley geometry is
  *data-property-dependent*, i.e., plausibly recipe-dependent."
- *Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning Rate Scheduler* —
  "plotting the landscape in coordinates defined by the global pre-cooldown→final direction
  vs. the local Adam-steps direction, explicitly noting that a clear visualization of the
  river valley had been lacking."
- *Scaling with Collapse: Efficient and Predictable Training of LLM Families* (arXiv
  2509.25087) — "well-tuned training runs' loss curves collapse onto a shared shape,
  suggesting families of runs traveling the same river — a possible cross-run comparability
  criterion built from curves alone, no weight access needed."
- The multi-power law paper (Luo et al., arXiv 2503.12811) "flags this… the 'decay-induced
  loss drop' term is, in river-valley language, descending from oscillating on the walls
  down to the river."

**Basin identification**
- "The operational definition remains linear mode connectivity — interpolate two weight
  sets; if performance along the path stays comparable to the endpoints, they probably lie
  in the same loss basin."
- Frankle et al., *Linear Mode Connectivity and the Lottery Ticket Hypothesis* —
  "same-run-early-split models are linearly connected."
- Entezari et al., *The Role of Permutation Invariance in Linear Mode Connectivity of Neural
  Networks* (conjecture); Ainsworth et al., *Git Re-Basin: Merging Models modulo Permutation
  Symmetries* — "independently trained models are connected *only after permutation
  alignment*." *Unveiling Linear Mode Connectivity of Re-Basin from Neuron Distribution
  Perspective* — "re-basin methods often reduce barriers only marginally and work poorly
  early in training, with no unified theory of when they succeed."
- *Going Beyond Linear Mode Connectivity: The Layerwise Linear Feature Connectivity* —
  connectivity in activation space, not just loss.
- *On the Emergence of Cross-Task Linearity in the Pretraining-Finetuning Paradigm* — "models
  fine-tuned from a common checkpoint stay in a shared linear regime, which is precisely
  why task arithmetic and model souping (*Model soups*, Wortsman et al.) work, and why they
  *only* work within a basin."
- *Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability* (2026)
  — "trying to explain basin structure via neuron identifiability — whether features get
  consistently assigned to neurons across random seeds — which would, if it matures, give a
  principled answer to 'when are two models' internal metrics comparable.'"

**Comparability precedent**
- Juneja et al., *Linear Connectivity Reveals Generalization Strategies* (ICLR 2023) —
  "fine-tuned models cluster into distinct linearly-connected basins, and models in
  *different* basins implement *different generalization strategies* (e.g., heuristic vs.
  syntax-sensitive on NLI) despite similar in-distribution accuracy. That's the strongest
  existing evidence… functionally, 'same metric value, different basin' can mean different
  mechanisms, so mechanism-level metrics (task vectors, GD-similarity scores) may not be
  comparable across basins at all."

**Thought.** "Nobody has connected either literature to *metric validity*: no paper says 'ICL
scores / task vectors / plasticity statistics are comparable iff models pass test X.'"

---

## 2026-08-18 — valley geometry as a data property (from the WSD/featurization discussion)

"The river-valley hypothesis **attributes the valley geometry itself to data properties** —
**deterministic tokens forming the river**, **uncertain tokens the walls** — meaning a
dataset's 'determinism profile' (cheap to estimate with any reference model) is a candidate
feature predicting not just performance but *landscape geometry*, i.e., annealing behavior."
This is the link between this topic and `data-featurization-literature.md` / REC-4.

---

## 2026-08-18 — Wen et al.'s own validation of the token mechanism

"They built a toy bigram language (cities with name distributions of varying determinism),
showed it reproduces the river-valley loss geometry, and demonstrated that the stable
learning-rate phase learns the deterministic tokens whereas the decay phase learns better
the stochastic tokens. On real data they validated the mapping correlationally: a
significant Spearman correlation (~0.39) between token-level uncertainty and local sharpness
of the landscape." See `token-level-literature.md`.

---

## 2026-08-18 — the critical period as the window before basin commitment

"Combine [critical periods] with the linear-mode-connectivity fact that checkpoints become
linearly connected only after early training stabilizes, and 'critical period' becomes:
*the window before the run commits to a basin/valley*. The river is chosen early, while
Fisher information is high; after commitment, you travel it. Deficits during the window
change which river you're in — permanently, because post-commitment plasticity won't take
you over the walls." See `critical-periods.md` and `critical-period-timing-study.md`.

---

## 2026-08-18 — basin-preserving vs. basin-determining resets

"Resetting the embedding layer is a large jump in parameter space that *doesn't leave the
basin*, because the body — where the basin identity lives — is untouched… Interface resets
are basin-preserving; early-training deficits are basin-*determining*. Same operation,
opposite regimes." Distillation into a fresh network (ITER) is the one reset that leaves the
basin by construction. See `reinit-and-transfer-literature.md`.

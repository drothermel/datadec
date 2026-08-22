# Token-level dynamics — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: token-level movement (TOK) measures which tokens move between
checkpoints and which respond to decay; the river/wall mapping, the epistemic/aleatoric
split, loss-trajectory taxonomies, and the RLVR "forking tokens" results are its prior art.

---

## 2026-08-22 — pointer: token-reweighting training objectives

The objectives that select or reweight tokens during training (Rho-1, MiLe, ESLM, TALR,
VCORE, multi-token prediction) are accumulated in
`training-objective-alternatives-literature.md`; this file stays with token-level
measurement and movement.

## 2026-08-18 — mapping tokens into river vs. wall, and how the mapping moves (from the Research Trajectory page)

Prompt context (Danielle): Wen et al. describe "highly deterministic tokens (facts, knowledge)
contribute the river direction, while uncertain, ambiguous tokens create the steep
hillsides." Have there been investigations mapping data tokens into the two buckets, or
watching the mapping change over training?

"The mapping has been made statically, and there are fragments of the dynamics, but the full
'watch the bucket assignment evolve over training' study doesn't exist."

**Wen et al. (arXiv 2410.05192) — the static version plus one dynamic observation.** "They
built a toy bigram language (cities with name distributions of varying determinism), showed
it reproduces the river-valley loss geometry, and demonstrated that the stable learning-rate
phase learns the deterministic tokens whereas the decay phase learns better the stochastic
tokens. On real data they validated the mapping correlationally: a significant Spearman
correlation (~0.39) between token-level uncertainty and local sharpness of the landscape…
What they didn't do is track individual tokens' bucket membership as a function of training
time — the mapping is treated as a fixed property of the data."

**Uncertainty decomposition.** *Token-Level Uncertainty-Aware Objective for Language Model
Post-Training* — "token-level uncertainty splits into epistemic (reducible; the model hasn't
learned it yet) and aleatoric (irreducible ambiguity in language), these vary enormously
across tokens, and epistemic uncertainty decreases faster for low-aleatoric examples as
training progresses." River-valley reading: "aleatoric uncertainty is the *true* hillside (a
data property, fixed), while epistemic uncertainty is distance-not-yet-traveled along the
river — and a token's *apparent* bucket (measured by current loss/entropy) migrates as
epistemic uncertainty drains away… that paper is the closest existing measurement of it,
though it never connects to landscape geometry."

**Loss-trajectory taxonomy.** Rho-1 (*Not All Tokens Are What You Need for Pretraining*) —
"classified tokens by their *loss trajectories* across pretraining checkpoints:
persistently-high, persistently-low, descending, and (notably) fluctuating/ascending
categories, finding that only a minority of tokens show meaningful loss descent late in
training. That's literally a token-bucket-over-time taxonomy, built for data-selection
purposes with no landscape interpretation attached."

**Post-training token regimes.** *Revisiting Entropy in Reinforcement Learning for Large
Reasoning Models* — "masking RLVR updates to different token regimes produces qualitatively
different training dynamics — with certain regimes driving stability and others driving
collapse." *Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
Reinforcement Learning for LLM Reasoning* — "a small high-entropy minority of tokens carries
most of RLVR's effect." "That's your hillside bucket resurfacing as the locus of
post-training, which is a suggestive bridge nobody has drawn explicitly: the tokens that
form the valley walls in pretraining look like the same tokens where RL does its work."

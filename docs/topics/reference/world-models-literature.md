# World models and agentic RL — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are
dated and quoted close to verbatim; related-work claims are unverified unless a citation
is given.

Why it matters here: LLMs-as-world-models is the model-based flank of the
LLM-in-classic-RL thread (see the LLM-as-optimizer entry in
`prompt-optimization-landscape.md` — DICL put an LLM in the transition-model slot) and
background for the whetstone-envs minigrid spec-out (staging placeholder). The
theory side (world models as a byproduct of competence) touches the program's
interest in what capabilities imply about internal structure.

---

## 2026-08-24 — NotebookLM world-models notebook (2 papers; founding entry)

Danielle supplied a NotebookLM notebook over two world-model papers (bundle:
`nblm-world-models-notebook.md` in the 2026-08-24 intake bundle; first authors
given, no IDs; agent-generated, unverified; NotebookLM inaccuracy caveat):

- **"From Word to World: Can LLMs be Implicit Text-based World Models?"**
  (Yixia Li; no ID). Three-level evaluation framework (fidelity/consistency,
  scalability, agent utility) for SFT'd LLM world models over five text
  environments (ALFWorld, SciWorld, TextWorld, WebShop, StableToolBench).
  Findings: fine-tuned LLMs are reliable next-state predictors; **structured
  environments saturate at ~20K trajectories while open-ended ones (WebShop,
  StableToolBench) are non-saturating at 160K** — capacity must scale with
  environment entropy. Utilities: pre-execution verification of irreversible
  actions (WebShop checkout gating with 2–10 verification budgets), synthetic
  trajectories competitive with real data, and world-model SFT as RL
  warm-starting. Notable methods detail: **mixed-agent (sub-optimal-inclusive)
  trajectories beat expert-only data** for robustness (weak-agent consistency
  ratio 0.49 → 0.81) — coverage of failure states matters.
- **"General agents contain world models"** (Jonathan Richens; no ID supplied —
  plausibly the 2025 DeepMind theory paper, arXiv 2506.01622, identification
  inferred). Formal result: any goal-conditioned policy meeting a regret bound
  across multi-step LTL goals must encode the environment's transition
  probabilities — **no model-free shortcut to general agency** — with an
  extraction algorithm recovering transition functions from policy behavior
  alone (goal-switching queries; no access to activations). Model accuracy
  scales with agent competence and goal depth (error O(δ/√n)); **myopic (n=1)
  agents provably need no world model**. Tested on a synthetic 20-state/5-action
  cMP.

The report's synthesis frames the pair as empirical-simulative vs
formal-reductionist and derives the "internalization thesis": world models as
inevitable byproducts of long-horizon competence, with the capability ceiling
bounded by internal-model fidelity. Boundary conditions worth keeping:
simulation drift in open-ended environments without real-observation anchoring,
and the "POMDP gap" (true state richer than its text description) as
irreducible uncertainty.

# Rewritten vs. selected anneal data — an intervention axis for the decay branch

**Kind:** staging. Candidate exit: a named sub-variant of `../../potential-projs/wsd-suite.md`
WSD-opt-4 (mixed-in decay data), or an arm of `datadecide-dense.md` once that has a design
doc. Gate: verify the four lead papers; decide whether a rewritten slice can be produced
at DataDecide scale cheaply enough to be an arm rather than a project.

Source: the third answer to Danielle's annealing-data question (record in
`../reference/schedules-and-annealing-literature.md`, third entry; intake 2026-08-22).
Danielle asked to pull the option out so it is not forgotten (post-processing, 2026-08-22).

---

## The idea

Every annealing-data result on file treats the anneal slice as *selected* data — a
higher-quality subset upsampled late (Llama 3, Databricks, OLMo 2, MiniCPM, FineWeb-Edu).
A separate 2024–25 line produces anneal-grade data by *rewriting* ordinary data instead:
SwallowCode / SwallowMath (arXiv 2505.02881; "transform-and-retain" refinement of Python
and math, quoted +17.0 HumanEval / +12.4 GSM8K), ProX (2409.17115; a 0.3B model emits
per-document refinement programs), FinerWeb-10BT (2501.07314; line-level LLM filtering),
Nemotron-CC's synthetic rephrasing of high-quality segments (2412.02595). All unverified
here.

The question this opens for the decay branch: at matched tokens and matched branch point,
does **upgrading** the slice (rewrite the recipe's own data) beat **selecting** it
(upsample a high-quality source), and does the answer depend on recipe? Selection changes
the mixture; rewriting keeps the mixture and changes the per-document quality, so the two
separate "what the anneal is made of" from "which distribution it comes from" — the
confound that WSD-opt-4 currently leaves entangled.

## Why it is an arm, not a project

- It lives entirely inside an existing decay-branch design: one more slice type at the
  same branch points. No new instrument.
- The readout is the one the WSD suite and annealed readouts already produce (branch
  endpoint loss and task accuracy vs. the parent's cosine twin).
- The expensive part is producing the rewritten slice; at DataDecide's smallest scales the
  slice is small, so a mid-size open model can rewrite it.

## Intake notes

- Figures quoted from the source answer are the respondent's; SwallowCode's numbers are
  for full pretraining on the rewritten corpus, not for an anneal slice — the anneal-slice
  version is untested as far as the record shows.
- Connects to `../../potential-projs/recipe-featurization.md`'s syntheticity feature (a
  rewritten slice is partly synthetic) and to the synthetic-data reference topic.

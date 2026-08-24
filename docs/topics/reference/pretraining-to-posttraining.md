# Pretraining → post-training — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: the portfolio's original hypothesis — pretraining choices shape
post-training success even at matched final pretraining performance — and the earlier
negative result ("post-training did nothing") that ended the previous project. The
post-training optional directions in annealed readouts (ANN-opt-3), the WSD suite
(WSD-opt-2), token-level movement (TOK-opt-4), and tiny-scale measurement (TINY options)
all descend from it.

---

## 2026-08-22 — pointer: distillation from pre- vs. post-trained teachers

Danielle's six-question distillation review (`distillation-literature.md`) found no
controlled comparison of distilling a student from a base teacher then post-training it
vs. distilling from the post-trained teacher directly — this file's question with a
teacher in the loop, and an open design.

## 2026-08-22 — pointer: midtraining toward a target suite

The forward-run version of this file's question — deliberately pretraining or
midtraining *for* a downstream suite, and how that trades against post-training — is
accumulated in `targeted-pretraining-midtraining-literature.md` (key entry: the
pre-/mid-training/RL interplay study 2512.07783).

## 2026-08-18 — DataDecide, and the two paper clusters (from the Research Trajectory page)

Prompt context (Danielle): DataDecide "seemed like an awesome source of variance which could
make trying to predict behavior interesting," and the search for proxy metrics at low scale
seemed related; this became "a project asking whether pretraining choices impact
post-training success, even when pretrained final perf is held constant. The direction hit
a wall when our post-training seemingly had no effect despite using standard procedures and
datasets."

**DataDecide.** Ian Magnusson et al., *DataDecide: How to Predict Best Pretraining Data with
Small Experiments* (Ai2, ICML 2025, arXiv 2504.11393). "Controlled pretraining experiments
across 25 corpora (varying sources, deduplication, filtering) up to 100B tokens, model sizes
up to 1B parameters, and 3 random seeds; they find that ranking models at a single small
size (150M) predicts the best dataset at the 1B target scale ~80% of the time, and no
scaling-law method among 8 baselines beats that simple baseline… using continuous
likelihood metrics as proxies in small experiments makes benchmarks including MMLU, ARC,
HellaSwag, MBPP, and HumanEval >80% predictable at the 1B scale with just 0.01% of the
compute. This is exactly the 'at low scale, accuracy is noise, so find a smoother
observable' move."

**Pretraining choices → post-training success, with pretrain performance held constant**
("a hot topic in 2025–2026"):

- *Similar Models Learn Differently: Final-Window Pretraining Shapes Post-Training Beyond
  SFT* (arXiv 2607.25063, 2026). "Closest to your exact experimental design: models that look
  similar after SFT diverge under identical post-training depending on late-pretraining data
  interventions. It explicitly cites Dohare's plasticity paper."
- Rosie Zhao, Alexandru Meterez et al., *Echo Chamber: RL Post-training Amplifies Behaviors
  Learned in Pretraining* (COLM 2025; arXiv 2504.07912). "Trains models from scratch on
  controlled pretraining mixtures, then compares PPO, GRPO, and Expert Iteration across
  scales. RL tends to amplify output patterns inherited from pretraining… also argues that
  controlled small-model proxies can yield real insight into RL behavior."
- Syeda Nahida Akter et al., *Front-Loading Reasoning: The Synergy between Pretraining and
  Post-Training Data* (NVIDIA, ICLR 2026; arXiv 2510.03264). "In controlled 8B experiments,
  broad and diverse reasoning data is most useful during pretraining, while SFT benefits
  more from a smaller curated set of high-quality long-CoT examples."
- Lawrence Feng et al., *Early Data Exposure Improves Robustness to Subsequent Fine-Tuning*
  (arXiv 2605.12705, 2026): "moving some target-domain data into pretraining improves
  retention after subsequent fine-tuning, even when immediate post-training performance is
  similar." Christina Baek et al., *The Finetuner's Fallacy: When to Pretrain with Your
  Finetuning Data* (arXiv 2603.16177, 2026): "early domain exposure can be more durable than
  introducing the same data only during fine-tuning, while the repetition schedule affects
  whether the model generalizes, overfits, or later forgets the domain."
- Jingyan Shen et al., *Understanding Reasoning from Pretraining to Post-Training* (arXiv
  2607.16097, 2026). "Lower pretraining loss strongly predicts higher post-RL pass@1 at fixed
  RL compute, while the local RL-improvement slope grows with log pretraining tokens… This is
  in interesting tension with your original hypothesis because final pretraining loss
  carries strong predictive signal here, though not all of it."

**"Post-training did nothing" — not alone:**

- Andreas Hochlehnert et al., *A Sober Look at Progress in Language Model Reasoning: Pitfalls
  and Paths to Reproducibility* (COLM 2025; arXiv 2504.07086). "RL applied to
  distillation-based models yields little to no statistically significant gain in the tested
  settings… Small benchmarks produce unstable estimates, making multiple seed runs
  essential."
- Rulin Shao et al., *Spurious Rewards: Rethinking Training Signals in RLVR* (ICML 2026;
  arXiv 2506.10947). "On Qwen2.5-Math-7B, GRPO with random rewards improves MATH-500 by 21.4
  points… while comparable spurious rewards generally fail outside Qwen families."
- Yang Yue et al., *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs
  Beyond the Base Model?* (NeurIPS 2025 oral; arXiv 2504.13837): RLVR "often improve[s]
  pass@k at small k but fail[s] to expand the base model's reasoning boundary at large k."
  Fang Wu and Yejin Choi, *On the Limits of RLVR: Support, Entropy, and the Illusion of
  Reasoning* (AI for Math Workshop, ICML 2025): RLVR "as predominantly support-preserving,
  entropy-reducing reweighting." Counterpoints: *The Invisible Leash* (arXiv 2507.14843)
  and *RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs* (arXiv 2506.14245; ICLR
  2026).
- Small models: Yutong Chen et al. (arXiv 2505.17988) — "small-scale SFT on Qwen2.5-1.5B can
  reduce MATH-500 accuracy from 23.8% to 18.4% even while eliciting reasoning-style
  behaviors"; Renjie Luo et al., *Through the Valley: Path to Effective Long CoT Training for
  Small Language Models* (EMNLP 2025; arXiv 2506.07712) — "Long CoT Degradation… attributed
  to error accumulation."

**Reading the earlier project in hindsight.** "You were testing 'pretraining shapes
post-training beyond final loss' at scales where (a) post-training gains are largely
elicitation of capabilities your models didn't yet have, (b) the Qwen confound was silently
inflating the literature's baseline expectations, and (c) benchmark noise swamps effect
sizes without multi-seed evaluation. All three of those are now papers. Your negative
result was, in hindsight, an early observation of a real phenomenon."

**Connective tissue.** "The field is converging on exactly your framing — final pretraining
loss is an insufficient statistic for downstream success, and the open question is what
*else* about the training trajectory (data order, late-window exposure, the 'state of the
learner' in plasticity terms) predicts post-training outcomes."

---

## 2026-08-18 — additional references from the experiment-design discussion

- *Provable Benefits of RLVR over SFT for Reasoning Models: Learning to Backtrack
  Efficiently* — graph-pathfinding-style synthetic testbed where "a seed costs minutes";
  causal rather than correlational.
- TinyZero — "RL visibly works at 0.5–3B on countdown and simple arithmetic."
- The design alternatives themselves are in `../../potential-projs/movement-microscope.md`.

---

## 2026-08-18 — ICL vs. fine-tuning as access routes to the same capabilities

- *Eliciting Fine-Tuned Transformer Capabilities via Inference-Time Techniques* — "a formal
  argument that capabilities acquired through SFT can be approximated by the base model via
  in-context learning without parameter updates."
- Bornschein, Lyle, Pascanu et al., *Fine-Tuned In-Context Learners for Efficient Adaptation*
  — "prompt-based methods excel in few-shot scenarios but their effectiveness plateaus as
  more data becomes available." Design consequence in `../../potential-projs/icl-elicitability.md`.

---

## 2026-08-18 — the data-placement results as critical-period phenomenology

"The 2025–2026 data-placement results — early exposure shaping models more durably than
late data (*Early Data Exposure…*), the final-window effects (*Similar Models Learn
Differently*), safety behaviors from pretraining resisting post-training removal — are
critical-period phenomenology at scale, mostly published without the connection drawn." See
`critical-periods.md`.

## Undated (~2026) — Danielle's first-hand account of the earlier project, and "the AI2 dataset by Kyle"

Danielle's statement (verbatim — the project fact of record for why the previous project
ended; see `../../potential-projs/movement-microscope.md` §4 for the follow-up design):

> So a while ago, I was working on a research project with a friend where we were
> considering the data decide model set, where you have models of a variety of sizes from 4
> million all the way up to 1 billion trained on a variety of different recipes, released
> with checkpoints and evaluations. And we were exploring the effect of post-training,
> specifically supervised fine tuning, but what we found was that using established
> post-training datasets like Tulu, Tulu 3, didn't make any difference in task performance
> on any of the tasks that we tried, ranging from like multiple choice tasks all the way up
> to things like human eval, just like no movement at all, which was really shocking, but
> then when we talked to more people, then they suggested that this actually might be more
> common than we expect and that they were working at AllenAI on creating a dataset that
> actually could be used in fine-tuning to move metrics on very specific tasks. I think
> one of the author's names was Kyle. And so I'm curious whether that dataset has come out
> and also what types of explorations exist for ways to fit pretrained models, whether
> that's supervised fine-tuning or other methods of post-training, but with a focus on
> specifically fairly low-budget approaches to fitting very small models on tasks where we
> can actually evaluate performance.

Facts fixed by this: the SFT data was Tulu / Tulu 3; evaluations spanned multiple-choice
through HumanEval; the result was no movement on any of them; the AI2 contact said this is
"more common than we expect" and that a dataset built to move metrics on specific tasks
via fine-tuning was in progress, with a co-author named Kyle. The open item is whether
that dataset has been released.

**Response (condensed).** Two diagnoses — (A) eval misalignment (SFT changes
style/format/alignment, the harness does not reward it, or penalizes format drift) and
(B) benchmark signal-to-noise at small deltas (small but real gains invisible without
tight train–eval matching; "a core theme in DataDecide-style work"). Identified the
dataset as **FollowIR** (arXiv 2403.15246; Kyle Lo co-author; instruction-following
reranking; FollowIR-train on HF; FollowIR-7B). Low-budget methods that "tend to move
metrics" for small models: task-targeted, eval-shaped SFT (hundreds–thousands of in-domain
examples in the harness's prompt style, LoRA/QLoRA); capability distillation from a
stronger teacher (verified solutions / traces) rather than chat-style instruction tuning;
preference optimization on verifiable tasks (DPO; RLVR-style loops with deterministic
rewards; cites an "Olmo-3.1-32B-Instruct" card for SFT → DPO → verifiable-reward RL);
data selection over data volume (arXiv 2503.01807). A concrete plan: one task with tight
feedback (HumanEval/MBPP, GSM-style exact match, or FollowIR pairs) → lock the harness and
run evals twice for variance → 500–5,000 eval-shaped examples, verifier-filtered if
synthetic → short PEFT with a small LR sweep and a held-out slice → DPO on verifier-ranked
pairs if SFT moves a little.

**Intake notes.**

- The FollowIR identification is a guess, not a finding: FollowIR is a JHU-led retrieval
  benchmark (Weller et al.) with Kyle Lo as one co-author; it has nothing to do with
  "fine-tuning to move metrics on multiple-choice / coding tasks for small pretrained
  models," which is what the contact described. Treat the "did it come out?" question as
  **open** — the resolution is to ask the contact, not to search for "Kyle + AI2 + dataset."
  The "Olmo-3.1-32B-Instruct" citation looks fabricated (OLMo 3 exists; a 3.1 32B Instruct
  card is unverified); the rest of the citations are plausible but unchecked.
- Diagnoses (A) and (B) are the two hypotheses the movement-microscope doc already
  operationalizes (noise floor / detection limit; within-reach tasks; eval-format
  sensitivity) — this entry adds nothing to the design but is the cleanest statement of the
  original observation in Danielle's own words, which the project docs had only
  paraphrased.
- The method list is standard; the one point worth keeping is the ordering *task-shaped
  SFT → distillation → DPO on verifiable pairs*, with LR sweep flagged as mattering more
  than expected for small models — consistent with MIC's "guaranteed-effect calibration"
  step (memorization / distillation first, to prove the instrument can see movement at all).

## 2026-08-24 — SFT-vs-RL skill profiles and procedural-knowledge influence (NBLM reasoning notebook)

The training-paradigm half of the four-paper NotebookLM reasoning notebook
(bundle: `nblm-reasoning-mechanisms-notebook.md`; main entry in
`generalization-and-ood-literature.md`, same date; no IDs supplied,
agent-generated, unverified):

- **"How and Why LLMs Generalize" (fine-grained SFT-vs-RL).** Decomposes
  reasoning into five atomic cognitive skills (calculation, enumeration,
  simulation, fact retrieval, diagnostic) across four domains and traces how
  post-training redistributes them in Qwen models: **RL preserves a balanced
  skill profile that transfers out-of-domain; SFT induces jagged
  over-specialization** — spikes in narrow skills with regressions in core ones
  (simulation), reading as surface-pattern overfit. A skill-resolved instance
  of the elicitation-vs-capability question: what post-training moves is the
  profile, not just the aggregate score.
- **"Procedural Knowledge in Pretraining Drives Reasoning"** (plausibly Ruis
  et al. 2411.12580; identification inferred from the title). EK-FAC influence
  functions over 5M pretraining documents, 7B/35B models: for factual queries
  the answer documents dominate influence; **for reasoning queries the answers
  rarely matter — influence concentrates on documents carrying similar
  procedural knowledge, especially code and mathematical text**. Directly
  relevant to recipe-composition questions (why code/math fractions move
  reasoning benchmarks) and the retrieval-vs-strategy reading of benchmark
  performance.

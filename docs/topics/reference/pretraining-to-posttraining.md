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

**Retrospective reading of the earlier project.** "You were testing 'pretraining shapes
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
- The design alternatives themselves are in `../staging/posttraining-experiment-design.md`.

---

## 2026-08-18 — ICL vs. fine-tuning as access routes to the same capabilities

- *Eliciting Fine-Tuned Transformer Capabilities via Inference-Time Techniques* — "a formal
  argument that capabilities acquired through SFT can be approximated by the base model via
  in-context learning without parameter updates."
- Bornschein, Lyle, Pascanu et al., *Fine-Tuned In-Context Learners for Efficient Adaptation*
  — "prompt-based methods excel in few-shot scenarios but their effectiveness plateaus as
  more data becomes available." Design consequence in `../staging/icl-as-posttraining.md`.

---

## 2026-08-18 — the data-placement results as critical-period phenomenology

"The 2025–2026 data-placement results — early exposure shaping models more durably than
late data (*Early Data Exposure…*), the final-window effects (*Similar Models Learn
Differently*), safety behaviors from pretraining resisting post-training removal — are
critical-period phenomenology at scale, mostly published without the connection drawn." See
`critical-periods.md`.

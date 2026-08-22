# In-context learning — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: ICL is the candidate gradient-free "post-training" stage
(`icl-as-posttraining.md`), and the emergence-of-ICL literature already contains the
portfolio's thesis in miniature — pretraining data properties change what a model can do
in context even at similar training loss.

---

## 2026-08-18 — emergence of ICL depends on pretraining data properties (from the Research Trajectory page)

- Chan et al. (2022), *Data distributional properties drive emergent in-context learning in
  transformers*. "Showed with small transformers on Omniglot-style image sequences that
  whether ICL emerges *at all* depends on pretraining data properties (burstiness, class
  distribution skew, within-class variation) — often with little difference in ordinary
  training loss."
- Raventós et al., *Pretraining task diversity and the emergence of non-Bayesian in-context
  learning for regression*. *(Citation gap on the Notion page: the author is not linked
  to a page mention, unlike the other citations; verify and link.)* "Did the analogous thing for linear-regression ICL, finding a
  task-diversity threshold."
- "That literature is your hypothesis, already demonstrated in miniature — but it's framed
  as 'when does ICL emerge,' not as 'ICL-ability as a measurable functional of pretraining
  recipe that predicts adaptation at larger scale.' That reframing is your gap."
- Mechanism assumption to state explicitly: "ViT-ICL and LLM-ICL plausibly share mechanism
  (induction-head-like circuits) but that's an assumption of the design."

---

## 2026-08-18 — analyzing ICL: gradient approximations, extractable objects, and measurement statistics (from the Research Trajectory page)

Prompt context (Danielle): how can what happens during in-context learning be analyzed —
are there gradient approximations, and what statistics measure the intermediate impact of
ICL?

**The "gradient approximation" literature — and its debunking arc**
- von Oswald et al., *Transformers learn in-context by gradient descent*: "you can construct
  key/query/value matrices such that one linear self-attention step on the tokens is
  *identical* to gradient-induced dynamics."
- Akyürek et al., *What learning algorithm is in-context learning? Investigations with
  linear models*: extended to ridge regression.
- Dai et al., *Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient
  Descent as Meta-Optimizers*: "a kernel perspective… ICL as implicit finetuning… they
  define concrete similarity statistics between attention-induced hidden-state updates and
  actual fine-tuning gradient updates."
- Ahn et al., *Transformers learn to implement preconditioned gradient descent for
  in-context learning*.
- **Contradiction:** *In-context Learning and Gradient Descent Revisited* — on realistic NLP
  tasks, "problematic metrics, insufficient baselines, and, damningly, even *untrained*
  models achieve comparable ICL–GD similarity scores despite not exhibiting ICL."
- **Contradiction:** *The Initialization Determines Whether In-Context Learning Is Gradient
  Descent* — "whether ICL corresponds to gradient descent at all depends on precise
  conditions (e.g., initialization)."
- Consequence: "use the GD-similarity statistics, but with the untrained-model control the
  original papers skipped."

**The extractable-object family: task / function / state vectors**
- Hendel et al., *In-Context Learning Creates Task Vectors*; Todd et al., *Function Vectors
  in Large Language Models* — "a demonstration set gets compressed into a transportable
  hidden-state vector."
- *In-Context Learning State Vector with Inner and Momentum Optimization* — "a 'state
  vector' capturing the ICL processing state stored in attention activations, explicitly
  analyzing its similarities with parameters trained via gradient descent," refined with
  optimizer-style tricks.
- *Learning Task Representations from In-Context Learning* — "represents ICL tasks as a
  learned weighted sum of all attention heads."
- "These give you parameter-like measurables without parameters: vector norm (how much
  'learning' happened), direction stability across demo orderings (cheap variance
  estimates), transferability across prompts (generalization), and — the recipe-comparison
  payoff — whether two base models pretrained differently produce differently-structured
  task vectors at matched loss."

**Measurement statistics**
- Olsson et al., *In-context Learning and Induction Heads* — the original "ICL score" "was
  literally the loss difference between an early and late token position"; induction-head
  strength (prefix-matching / copying scores on synthetic sequences) as "the mechanistic
  correlate of ICL emergence, measurable per-checkpoint."
- *What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task
  Learning* — "shuffled-label and format-only controls separate 'the demos told the model
  which task' from 'the model learned the mapping.'"

**Thought.** "The field's best ICL measurement is a loss curve over context position, fit
with power laws, complete with emergence thresholds and proxy-metric debates — your
pretraining-scale question recapitulated inside a single forward pass."

# Task vectors — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Citations below
were supplied by Danielle with arXiv identifiers (2026-08-22) and are not subject to the
"unverified related-work claims" caveat; summaries are the papers' own abstracts, lightly
trimmed.

**Two senses of "task vector."** (1) *Activation-space* task/function vectors: an in-context
demonstration set compressed into a transportable hidden-state vector (Hendel et al.; Todd
et al.; see `icl-literature.md`). (2) *Weight-space* task vectors: the difference between a
fine-tuned and a pre-trained model's weights (Ilharco et al.), composable by arithmetic. The
two are linked by the gradient view — a weight-space task vector from one epoch of
fine-tuning is the negative scaled gradient (Zhou et al.), and ICL is argued to implement
something gradient-like in activation space (see the ICL–GD arc in `icl-literature.md`).

Why it matters here: weight-space task vectors are the natural readout of a branch's
direction (functional featurization FUNC-4), the first-epoch-gradient result supports the
surrogate ladder's cheapest tier (FUNC-5), functional (activation-based) task identity is
the same move as cross-architecture expert matching (MoE partitions PART-4), and ICL task
vectors are statistic #4 of the ICL protocol (`icl-as-posttraining.md`).

---

## 2026-08-22 — papers supplied by Danielle

**Weight-space task vectors and model merging**

- Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi, *Editing Models
  with Task Arithmetic* (ICLR 2023; arXiv 2212.04089). "A task vector specifies a direction
  in the weight space of a pre-trained model, such that movement in that direction improves
  performance on the task. We build task vectors by subtracting the weights of a pre-trained
  model from the weights of the same model after fine-tuning on a task… these task vectors
  can be modified and combined together through arithmetic operations such as negation and
  addition… Negating a task vector decreases performance on the target task, with little
  change in model behavior on control tasks… adding task vectors together can improve
  performance on multiple tasks at once… when tasks are linked by an analogy relationship
  of the form 'A is to B as C is to D', combining task vectors from three of the tasks can
  improve performance on the fourth." Danielle: "a bit of a different take on task vectors."
- Zhou, Solombrino, Crisostomi, Bucarelli, D'Inverno, Silvestri, Rodolà, *On Task Vectors
  and Gradients* (arXiv 2508.16082, v6 Oct 2025). "Under standard gradient descent, a task
  vector generated from one epoch of finetuning is exactly equivalent to the negative
  gradient of the loss, scaled by the learning rate. For the practical multi-epoch setting…
  this equivalence holds approximately, with a second-order error term that we explicitly
  bound for feed-forward networks… the first-epoch gradient dominates the finetuning
  trajectory in both norm and direction… merging models finetuned for only a single epoch
  often yields performance comparable to merging fully converged models… reframe[s] task
  arithmetic as a form of approximate multitask learning… highlighting the critical role of
  early training dynamics in model merging."
- Rinaldi, Panariello, Salici, Porrello, Calderara, *Transporting Task Vectors across
  Different Architectures without Training* (ICML 2026; arXiv 2602.12952). "Theseus, a
  training-free method for transporting task updates across heterogeneous-width models.
  Rather than matching parameters, we characterize a task update by the functional effect
  it induces on intermediate representations. We formalize task-vector transport as a
  functional matching problem on observed activations and show that, after aligning
  representation spaces via orthogonal Procrustes analysis, it admits a stable closed-form
  solution that preserves the geometry of the update… task updates can be meaningfully
  transferred across architectures when task identity is defined functionally rather than
  parametrically."
- Kim, Lee, Jung, Ryu, Hong, *Task Vector Quantization for Memory-Efficient Model Merging*
  (arXiv 2503.06921, v2 Aug 2025). "Quantizing task vectors (i.e., the difference between
  pre-trained and fine-tuned checkpoints) instead of quantizing fine-tuned checkpoints… task
  vectors exhibit a narrow weight range, enabling low precision quantization (e.g., 4 bit)…
  Residual Task Vector Quantization… decomposes the task vector into a base vector and
  offset component." (Practical: branch endpoints stored as quantized deltas from the
  branch point would make saving endpoint weights for every branch cheap.)

**Activation-space (ICL) task vectors**

- Dong, Jiang, Zhu, Ning, *Understanding Task Vectors in In-Context Learning: Emergence,
  Functionality, and Limitations* (arXiv 2506.09048). "The Linear Combination Conjecture,
  positing that task vectors act as single in-context demonstrations formed through linear
  combinations of the original ones… task vectors naturally emerge in linear transformers
  trained on triplet-formatted prompts through loss landscape analysis… we predict the
  failure of task vectors on representing high-rank mappings and confirm this on practical
  LLMs… suggesting an enhancement of task vectors by injecting multiple ones into few-shot
  prompts."
- Yang, Cho, Ding, Inoue, *Task Vectors, Learned Not Extracted: Performance Gains and
  Mechanistic Insight* (ICLR 2026; arXiv 2509.24169). "Directly training Learned Task
  Vectors (LTVs), which surpass extracted TVs in accuracy and exhibit superior flexibility —
  acting effectively at arbitrary layers, positions, and even with ICL prompts… at the low
  level they steer predictions primarily through attention-head OV circuits, with a small
  subset of 'key heads' most decisive. At a higher level… TV propagation is largely linear:
  early TVs are rotated toward task-relevant subspaces to improve logits of relevant labels,
  while later TVs are predominantly scaled in magnitude."

---

## 2026-08-18 — why task arithmetic works only within a basin (from the Research Trajectory page)

"Cross-task linearity in the pretraining–finetuning paradigm (*On the Emergence of Cross-Task
Linearity in the Pretraining-Finetuning Paradigm*) — models fine-tuned from a common
checkpoint stay in a shared linear regime, which is precisely why task arithmetic and model
souping (*Model soups*) work, and why they *only* work within a basin." See
`landscape-literature.md`.

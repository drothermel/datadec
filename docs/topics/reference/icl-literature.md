# In-context learning — reference topic

**Kind:** reference (standing accumulator of paper references and thoughts). Entries are dated
and quoted close to verbatim; related-work claims are unverified unless a citation is given.

Why it matters here: ICL is the candidate gradient-free "post-training" stage
(`../../potential-projs/icl-elicitability.md`), and the emergence-of-ICL literature already contains the
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

---

## 2026-08-22 — ICL task vectors: emergence, limitations, learned variants (citations supplied by Danielle)

- Dong et al., *Understanding Task Vectors in In-Context Learning: Emergence, Functionality,
  and Limitations* (arXiv 2506.09048) — the Linear Combination Conjecture (task vectors as
  linear combinations of the demonstrations); emergence in linear transformers on
  triplet-formatted prompts; predicted and confirmed failure on high-rank mappings.
- Yang et al., *Task Vectors, Learned Not Extracted* (ICLR 2026; arXiv 2509.24169) — directly
  trained task vectors outperform extracted ones; mechanism via attention-head OV circuits
  and a few key heads; propagation largely linear (early rotation, later scaling).
- The weight-space sense of "task vector" (Ilharco et al.'s task arithmetic and its
  gradient interpretation) is collected in `task-vectors.md`.

## Undated (intake 2026-08-22) — Lightweight adaptation + in-context learning hybrids

**Danielle's request.** An extended search for three examples — arXiv papers, benchmarks,
or strong company blogs — of "very lightweight fine-tuning or adaptation along with
in-context learning" and what the learnings are.

**Response (condensed; weak — two paper pointers and one unsourced claim).**
1. *Context Tuning for In-Context Optimization* (arXiv 2507.04221, 2025): optimise the
   context/prompt given to the model rather than its parameters — a lightweight tuning of
   the input that improves few-shot adaptation with frozen weights.
2. *You Only Fine-tune Once: Many-Shot In-Context Fine-Tuning for Large Language Models*
   (title as given; no ID): one lightweight fine-tuning run on a representative example
   set, then rely on in-context learning for subsequent tasks.
3. "Industry blog posts" about adding a small trainable adapter (LoRA) and then using
   in-context examples on top for efficient personalisation — **no source named**; treat as
   unsupported.

Stated learning: a small adapter plus in-context examples gets much of full fine-tuning's
benefit at a fraction of the cost.

**Relevance.** These are the "weight-update vs. elicitation" middle ground that the ICL
elicitability project (`../../potential-projs/icl-elicitability.md`) and the research
hypothesis frame as the thing to measure: where does a tiny update beat tuned elicitation,
and does the balance shift with scale? Both papers need verification before use; the third
example should be replaced by a real source (e.g. vendor LoRA + few-shot guidance) if it is
ever cited.

**Follow-up — soft prompts revisited (same conversation).** Danielle's question: before ICL
"was really a thing," prompt tuning meant *learning embeddings* to prompt models that could
not yet follow natural language well; is Context Tuning "kind of returning to something like
that," and are there modern approaches that combine natural-language context with tuned
pieces — embeddings, or token combinations "that don't necessarily mean as much for humans,
but are more meaningful for the agents"? Response (content-free; condensed): yes, a revival
of soft prompts with a modern twist — hybrid prompting that mixes human-readable
instructions with learned tokens/embeddings as "subtle cues," "certain recent research
papers" — **no papers or systems named**. Treat as a prompt for a real literature pass, not
as findings.

*Danielle-flagged lead.* The question itself is a good one for the elicitation-ceiling
framing: soft-prompt / learned-token steering (Lester et al. prompt tuning; prefix tuning;
P-tuning lineages) vs. natural-language ICL is a continuum of "how many tuned parameters
does elicitation get," and where a frozen body's reachable content saturates along that
continuum is exactly the ICL-elicitability ceiling question. Candidate lit-pass: modern
hybrids of NL context + learned tokens (e.g. gist/compression tokens, learned
tool/format tokens, optimized non-readable prompt strings from discrete prompt
optimisation) — unverified, to be searched.

**Follow-up — ICL learning curves as a parallel to training curves (same conversation).**
Danielle's thought: the fine-tuning vs. in-context-learning comparison "clearly is a direct
comparison, and the idea that by consuming more tokens, a model is moving towards a,
quote-unquote, more trained state, kind of, is a parallel to fine-tuning." So: we study
scaling laws and performance curves over the course of training for LLMs — is the same
done in the prompting space, e.g. performance as a function of the number of in-context
examples ("what is in an n-shot"), and how different prompt choices change that
"learning curve"? Response (content-free; condensed): yes, people treat example count /
prompt complexity as a resource analogous to training steps; diminishing returns and
plateaus; prompt design shifts the curve; mixed NL + learned embeddings can improve sample
efficiency — **no papers named**.

*Danielle-flagged lead (and a direct restatement of the ICL-elicitability core).* This is
the "ICL training run" framing in `../../potential-projs/icl-elicitability.md`: the
per-token/per-demonstration curve is the learning curve, prompt format and ordering are the
hyper-parameters, and the question of whether ICL curves obey scaling-law-like regularities
across model size and pretraining recipe is the project's measurement target. Candidate
lit-pass leads (mine, unverified): many-shot ICL scaling studies (hundreds to thousands of
demonstrations in long contexts); n-shot curves in the original GPT-3 evaluations; work
fitting power laws to ICL accuracy vs. number of shots; ICL-vs-fine-tuning matched-budget
comparisons; demonstration-ordering and calibration effects on the curve.

**Follow-up — is there an ICL analogue of the "fundamental x-axes" of training?** Danielle's
framing: scaling laws across tasks or LR schedules are often better computed against
*cumulative learning rate* than raw tokens — "there's some concept of step size of
learning" — and other indicators such as weight-norm movement serve as fundamental axes
of the learning process. "Is there an equivalent set of values within the space of
in-context learning?" Response (content-free; condensed): number/complexity of examples
as a "learning rate" analogue; example-to-task distance; response consistency across
repeated prompts as a "convergence" proxy — **no references**.

Danielle's pushback and proposal: the number of examples "is more similar to either a
compute metric or a number-of-tokens metric" — she would *not* expect things to scale with
example count the way they scale with learning-rate changes. Candidate analogy: let n be
the number of *unique* examples (the data axis) and **how often each example is repeated
in the prompt** be the step size / learning rate — "you're taking a bigger step on each
example." Response (content-free; agrees, cites unnamed "early findings" that repetition
helps models "lock in" patterns) — **no references**.

*Danielle-flagged lead — and a concrete protocol element for ICL-elicitability.* The
(unique examples) × (repetitions per example) factorial is a cheap, well-defined way to
ask whether ICL has a separable "data" axis and "step-size" axis like weight-space
training does: if repeating demonstrations shifts the per-token curve the way a larger LR
shifts a training curve (faster initial movement, different plateau / instability), the
analogy has content; if repetition only adds tokens, it doesn't. Other candidate ICL
x-axes worth recording alongside cumulative LR / weight-norm movement: cumulative
demonstration tokens; per-token loss movement on the query as the "trajectory"; attention
mass on demonstrations vs. query; task-vector norm growth across demonstrations
(`task-vectors.md`); prompt-order entropy. Related empirical literature (mine, unverified):
ICL as implicit gradient descent / meta-optimisation analyses; demonstration repetition
and duplication studies; many-shot ICL with repeated vs. unique examples.

**Follow-up — researchers and papers on ICL scaling (same conversation).** Danielle asked
for a search of recent arXiv papers: ~four senior (last-author) researchers whose labs work
on ICL scaling, plus directly relevant papers. Response named: David Alvarez-Melis
("MIT"), Percy Liang (Stanford), Jacob Andreas (MIT), Tatsunori Hashimoto (Stanford); and
papers *Bayesian Scaling Laws for In-Context Learning* (arXiv 2410.16531, late 2024),
*Scaling Laws for Many-Shot In-Context Learning with Self-Generated Annotations* (March
2025), *MachineLearningLM: Scaling Many-Shot In-Context Learning via Continued Pretraining*
(September 2025), and *Prompt Design and Repetition Strategies in In-Context Learning*
("Hashimoto et al., 2025").

*Reliability note.* The **author/lab attributions look fabricated or wrong** and must not be
reused: the one linked paper (2410.16531) is not an Alvarez-Melis paper as far as the
intake can tell, and Alvarez-Melis is at Harvard, not MIT; the fourth "paper" has the
shape of an invented title matching Danielle's own repetition idea. Keep only the three
paper *titles* and the one arXiv ID as search leads; rebuild the researcher list from a
real citation graph if wanted. Useful seed for a lit pass: many-shot ICL scaling laws;
Bayesian framings of accuracy vs. shots; continued-pretraining-for-ICL.

**Closing note (same conversation).** Danielle tabled the related-work investigation to
test the approach practically at small scale first; the practical setup is staged in
`../staging/clean-code-preference-icl.md`.

**Follow-up — many-shot ICL vs. long-context degradation (same conversation, later).**
Danielle: production models are benchmarked on "how well a model can use its context"
(needle-in-a-haystack) which "doesn't necessarily correlate to how well an agent can use
its context"; she has not seen work directly on the trade-off between more in-context
examples and worse performance over the course of the context window — any papers in the
last year? Response (thin): *In-Context Learning with Long-Context Models* (NAACL 2025,
aclanthology 2025.naacl-long.605) — performance improves with more examples up to a limit,
then diminishing returns; *Efficient Prompting via Dynamic In-Context Learning* (no
venue/ID) — adapt the number of examples to balance performance and cost. Unverified; the
first is a real lead for the many-shot curve question, the second is adjacent at best.
Her underlying question — separating "more examples help" from "longer context hurts" —
is the matched-context factorial already noted above.

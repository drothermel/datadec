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
  learning for regression*. "Did the analogous thing for linear-regression ICL, finding a
  task-diversity threshold."
- "That literature is your hypothesis, already demonstrated in miniature — but it's framed
  as 'when does ICL emerge,' not as 'ICL-ability as a measurable functional of pretraining
  recipe that predicts adaptation at larger scale.' That reframing is your gap."
- Mechanism assumption to state explicitly: "ViT-ICL and LLM-ICL plausibly share mechanism
  (induction-head-like circuits) but that's an assumption of the design."

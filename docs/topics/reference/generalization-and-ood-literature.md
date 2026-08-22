# Generalization formalisms and out-of-domain performance prediction — reference topic

**Kind:** reference (standing accumulator). Entries are dated and quoted close to verbatim.
Danielle's own flag on the source conversation: "not necessarily the most useful answers /
I'm not sure I trust the citations, but the sequence is worth including." Treat every
citation below as **unverified**; the response's links all carry `utm_source=chatgpt.com`
and none were checked.

Why it matters here: the three-question sequence (is pretraining unsupervised → how is
generalization formalized across supervised and self-supervised settings → how is OOD
performance measured and *predicted conditioned on the method*) is an early (2025-01-04)
trace of the question that later became "predict downstream outcomes from recipe / early
dynamics" — see `../../potential-projs/early-dynamics-prediction.md` and
`../../potential-projs/recipe-featurization.md`. The last question's framing, "predicting
the performance conditioned on the method," is the part worth keeping.

---

## 2025-01-04 — Three questions (Danielle) and the responses (condensed)

**Q1. Is pretraining LLMs considered an unsupervised learning approach?**
Response: yes in the broad sense (no external labels), but "self-supervised" is the more
accurate and increasingly used term — pseudo-labels come from the data itself (causal LM,
masked LM, span / permutation objectives). Nothing further of note.

**Q2. How have ML researchers formalized generalization to compare supervised AND
self-supervised settings?** Response listed four strands:

1. *Generalization bounds for SSL* — bounds on downstream classification error for
   contrastive SSL. Cited: "Towards the Generalization of Contrastive Self-Supervised
   Learning" (arXiv 2111.00743; also given as an OpenReview link).
2. *Mutual-information frameworks* — information shared between learned representations
   and the data distribution as a proxy for transfer.
3. *Empirical generalization-gap comparisons* — "SSL models often exhibit smaller
   generalization gaps." Cited: "For self-supervised learning, Rationality implies
   generalization …" (OpenReview `Srmggo3b3X6`).
4. *Theoretical analyses of why pretraining helps downstream NLP* — via modelling the data
   generation process. Cited: a Stanford thesis, "Theoretical insights on generalization in
   supervised and self-[supervised] …" (purl.stanford.edu/yw554bd7619).

**Q3. What work has been done on measuring out-of-domain generalization and predicting the
performance conditioned on the method?** Response listed five strands:

1. *Domain-generalization techniques* — data manipulation, representation learning, learning
   strategies; cites the survey "Generalizing to Unseen Domains: A Survey on Domain
   Generalization" (arXiv 2103.03097).
2. *Evaluation protocols for OOD* — avoid leakage from test domains; use self-supervised or
   random initial weights during evaluation; assess across multiple test domains. Cited:
   "Rethinking the Evaluation Protocol of Domain Generalization" (CVPR 2024).
3. *Calibration ↔ OOD* — multi-domain calibration as an indicator of robustness to shift.
   Cited: "On Calibration and Out-of-domain Generalization" (NeurIPS 2021).
4. *Theoretical frameworks* — "Towards a Theoretical Framework of Out-of-Distribution
   Generalization" (NeurIPS 2021).
5. *Empirical assessments* — average accuracy approximates true OOD performance poorly;
   worst-case-across-domains proposed instead. Cited: "Towards a Better Evaluation of
   Out-of-Domain Generalization" (arXiv 2405.19703).

Intake note: the response never actually answered the "predicting performance conditioned on
the method" half of Q3 — every strand is about *measuring* OOD generalization or making
models more robust, none about forecasting a method's OOD performance from its training
procedure. That gap is the interesting part, and it is the gap the recipe-featurization and
early-dynamics docs now occupy (there: predict downstream scores from recipe features /
early-window curve shape; here: the same question posed at the level of "method"). The
citations look like plausible real papers (the titles match known work) but were not checked
and may be mis-attributed in detail; the "SSL models exhibit smaller generalization gaps"
claim in particular is an overstatement of whatever the cited paper shows.

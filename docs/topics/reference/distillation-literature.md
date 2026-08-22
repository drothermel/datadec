# Knowledge distillation in LLMs — literature reference

**Kind:** reference (accumulator for LLM distillation as it bears on the program: teacher/
student scaling, the distillation objective, logit vs. token repetition, KD vs. scratch,
and pre- vs. post-trained teachers). Entries are dated. Characterizations are a SciSpace
agent's; identifiers unverified unless they come from the downloaded PDFs. Siblings:
`pretraining-to-posttraining.md` (questions 5–6 are that file's question run through a
teacher), `synthetic-data-literature.md` (sequence-level KD = training on teacher
samples), `small-scale-evaluation-metrics-literature.md`.

Why it matters here: distillation is the post-training arm MIC already lists (KL-to-
teacher as a ground-truth movement readout; distillation quality as a confound), the
cheapest route to the small specialised models ELI and TINY reason about, and — in
questions 5–6 — the cleanest experimental handle on "what does post-training add that
pretraining data did not": distil the same student from a base teacher and from its
post-trained sibling and compare.

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-llm-distillation-agent-artifacts-zip_deb5ff94-1238-4180-81b2-62dc9cedad35_1787424886/` — the report (md + LaTeX/PDF with 10 cropped figures), 34
downloaded PDFs (about half off-topic), and 11 search CSVs, two merged with
question-aligned extracted columns. **`INDEX.md` inside the folder is the file-level
index** and lists the canonical papers the review missed.

---

## 2026-08-22 — SciSpace deep review (undated, ~2026)

**Danielle's prompt (verbatim):**

> I want you to do a literature review of distillation in LLMs, focusing on the following
> areas: 1) the size of the teacher and student, potentially including scaling laws, 2)
> the objective to use for the distillation loss, and how to combine it with CE loss or
> other losses, 3) repetition of logits in distillation compared to repetition of raw data
> tokens in pretraining, 4) what conditions lead to distillation performing better than
> training from scratch, 5) the difference between distilling from pre-trained vs.
> post-trained models and 6) the difference in performance between distilling from a
> pre-trained teacher and then doing post-training, versus only distilling from a
> post-trained teacher.

**What the review says, per question (condensed; evidence quality noted).**

1. *Sizes and scaling.* Reported ratios span 2:1 (Minitron 15B/30B→8B, 2407.14679) to
   ~26:1 (7–13B→500M for preference distillation, ADPA 2502.17927); Peng et al.'s
   pre-training-distillation design-space study (2410.16215) finds moderate 2–4:1 best
   in general; larger teachers help MiniLLM students monotonically across 120M–13B;
   domain-aligned mid-size teachers can beat bigger ones (DDK 2407.16154). **No
   distillation scaling law is cited** — Busbridge et al. 2502.08606 is the missing
   anchor.
2. *Objective.* Forward KL is mode-covering and over-weights the teacher's tails;
   reverse KL (MiniLLM, ICLR 2024) is mode-seeking and better for instruction
   following, optimised by policy gradient with teacher-mixed sampling (α = 0.2) and
   length normalisation; JSD / skew-KL / α-β / TV variants studied in Concrete Score
   Matching (Kim et al. 2509.25837); token-wise divergence control (ToDi 2505.16297),
   BiLD (2406.13555), cross-tokenizer losses (ULD 2402.12030; multi-level OT
   2412.14528). Combination: equal-weight CE + KD (DSKD 2406.17328, T = 2.0);
   MiniLLM adds a language-modelling loss; ADPA weights an advantage-guided KD against
   SFT. Missing: on-policy GKD (Agarwal et al. 2306.13649), DistiLLM (2402.03898).
3. *Logit vs. token repetition.* The only evidence is Bui et al. (2404.19319):
   BERT-scale students under fixed compute — with limited data, KD beats scratch by
   1.3–2.4 points; with unlimited data, vanilla KD ≈ scratch while TinyBERT/MiniLM-style
   KD keep +1.2–1.5; scaling one run from 2.6B unique to 15.4B repeated tokens helped
   KD. Extrapolated to LLMs by the review without LLM evidence. The token-repetition side
   (Muennighoff et al., ≤4 epochs ≈ free) is in `synthetic-data-literature.md`.
4. *KD vs. scratch.* Data-limited and compute-limited regimes favour KD; unlimited data
   narrows the gap; KD wins more on generation/reasoning than on short-answer tasks
   (MiniLLM: similar for <5-token outputs); one counterexample where a heavily resourced
   from-scratch 2B beat a 10B→2B distilled model (GLMD 2306.06625: 85.9 vs. 85.3 on
   SuperGLUE); "well-read students learn better" (Turc et al. 2019) — pretrain the
   student before distilling.
5. *Pre- vs. post-trained teacher.* Pre-training distillation (MiniPLM 2410.17215; Peng
   et al.) yields reusable bases; post-training / alignment distillation (ADPA, DCKD;
   "revealing the power of post-training for SLMs via KD" 2509.26497) transfers aligned
   behaviour directly and avoids the small-model alignment tax (ADPA: 62.7% AlpacaEval
   win rate for a 1.8B student).
6. *Sequential vs. direct workflow.* **No controlled comparison found**; the review
   restates 5 with preference-distillation results and argues flexibility (sequential)
   vs. efficiency (direct). Open.

**Intake notes.**

- Questions 1, 3, and 6 are under-evidenced: 1 lacks the distillation scaling-law
  paper, 3 rests on an encoder-scale study, 6 has no direct evidence. Questions 2, 4, 5
  are reasonably covered.
- The reference list has 16 duplicate entries ([49]–[60] repeat earlier ones); "Yeongmin
  et al." is a first name; half the downloaded PDFs are off-topic (image KD, CLIP,
  speech, memes, energy modelling).
- For this repo the actionable reading is question 5/6: distilling one student from a
  DataDecide-style base teacher and from its post-trained sibling is a clean design for
  the apex question, and the review confirms nobody has published that comparison.
  Recorded in `../../potential-projs/movement-microscope.md` §4.

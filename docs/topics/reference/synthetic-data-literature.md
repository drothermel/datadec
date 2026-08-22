# Synthetic data for LM pretraining and finetuning — literature reference

**Kind:** reference (accumulator for synthetic / rephrased training data: scaling
behaviour, rewriting methods, instruction synthesis, and model-collapse / overfitting
avoidance). Entries are dated. Characterizations are a SciSpace agent's; identifiers are
arXiv numbers read from the downloaded PDFs. Siblings: `data-featurization-literature.md`
(syntheticity as a feature), `targeted-pretraining-midtraining-literature.md`,
`pretraining-to-posttraining.md`.

Why it matters here: synthetic and rephrased corpora are a recipe axis DataDecide does
not span but every current pretraining stack uses (REC's feature families need a
syntheticity / rephrasing measure); model collapse is the overfitting failure mode for
recipes built on generated text; and instruction-synthesis methods (Self-Instruct, GLAN)
are the post-training data options for ELI-3 / MIC when no task-specific SFT set exists.

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-synthetic-lm-training-agent-artifacts-zip_888d8c4b-0b22-4403-8619-693d35468c3e_1787424684/` — the LaTeX/PDF review (the deliverable), 14 key PDFs with
per-paper summaries, 40 cropped figures, 7 peripheral PDFs, and 8 search CSVs.
**`INDEX.md` inside the folder is the file-level index** with the arXiv→title map.

---

## 2026-08-22 — SciSpace deep review (undated, ~2026)

**Danielle's prompt (verbatim):**

> I would like you to review the literature on Synthetic data for language model
> pretraining and finetuning. Specifically, please focus on works related to scaling of
> models and/or data, methods such as rewriting, and how to avoid overfitting. After you're
> done, compile all of the results into a LaTeX pdf, and include figures from the papers as
> appropriate.

What she pasted is the agent's `final_synthesis.md` — per-paper summaries of four papers
(Self-Instruct, GLAN, and two surveys) — not the LaTeX review, which is the real
deliverable and is organized by her three themes. This entry records the paper set from
the bundle and the four pasted summaries condensed.

**The paper set the review is built on (by theme, from the PDF's structure).**

- *Scaling with synthetic data:* BeyondWeb (2508.10975; trillion-scale rephrasing,
  "lessons"); scaling data-constrained LMs (2305.16264; repetition up to ~4 epochs ≈
  free); when scaling meets LLM finetuning (2402.17193); quality-aware scaling with a
  quality parameter Q (2510.03313); diversity of synthetic data and its effect on
  training (2410.15226); best practices and lessons on synthetic data (COLM 2024,
  2404.07503).
- *Rewriting / generation methods:* rephrasing the web (WRAP; subsection in the PDF);
  Self-Instruct (2212.10560); GLAN (2402.13064; taxonomy → subjects → syllabi →
  questions, 10M pairs, no seed data); MAmmoTH2 (web-mined instructions); Instruct-SkillMix
  and diversity-driven generation; the two surveys (2406.15126 — prompting strategies:
  task specification, conditional prompting, in-context demos, multi-step and
  dataset-wise decomposition; curation and evaluation taxonomy; 2410.12896 — augmentation
  vs. synthesis, labeling, reformation/rewriting incl. WRAP and BioR, co-annotation,
  distillation vs. self-improvement, Alpaca/WizardLM/Orca/Phi-1 lineage).
- *Model collapse and overfitting:* collapse as a change of scaling laws ("A Tale of
  Tails", 2402.07043); beyond collapse — scaling up with synthesized data requires
  verification (2406.07515); how to synthesize text without collapse (2412.14689);
  repeated-data effects; transferable results from synthetic-image scaling laws; the
  review's "practical strategies" section.

**The four pasted summaries, condensed.** *Self-Instruct:* 175 seed tasks → 52K
instructions / 82K instances via bootstrapped generation with ROUGE-L < 0.7 filtering;
+33% on SuperNI over vanilla GPT-3, near InstructGPT-001; human review finds 54% of all
fields valid; gains plateau after ~16K instructions; regenerating outputs with a stronger
model adds ~10%. *GLAN:* knowledge taxonomy (GPT-4 + human edit) → subjects → syllabi →
single-/two-session homework questions → GPT-3.5 answers; 10M pairs, decontaminated;
Mistral-7B + GLAN reaches GSM8K 80.8, MATH 32.7, HumanEval 48.8, ARC-C 81.1 with no
task-specific data; near-identical train/test loss taken as evidence of no in-domain
exposure; STEM +8.1 but humanities/social sciences slightly negative. *Survey 2406.15126:*
generation = task specification + conditions + demonstrations + multi-step decomposition;
curation by uncertainty/diversity selection; >300 HF datasets tagged synthetic by mid-2024.
*Survey 2410.12896:* lifecycle view (preparation → pretraining → finetuning →
instruction-tuning), the augmentation/synthesis/labeling/reformation split, challenges
(evaluation pollution from synthetic data; reliability; "protected internal benchmarks"
recommendation).

**Intake notes.**

- The pasted text is the wrong artifact for the question; the themed answer is the PDF.
  Nothing in the paste addresses scaling or collapse; the PDF sections do.
- Two key PDFs (2405.03548, 2510.01631) are unsummarized and unidentified in the
  bundle; see `INDEX.md`.
- Canon to check for in the PDF before relying on it: WRAP 2401.16380; Phi-1 /
  TinyStories (only via surveys); Nemotron-CC / Cosmopedia rephrasing corpora; Shumailov
  et al. curse of recursion (2305.17493); Gerstgrasser et al. accumulate-don't-replace
  (2404.01413); Magpie; STaR / ReST self-improvement. Unverified.
- Numbers as the agent extracted; unverified.

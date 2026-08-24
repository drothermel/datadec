# Prompt compression and discrete prompt optimization — reference topic

**Kind:** reference (accumulator for the prompt-compression / discrete-prompt-optimization
literature that TLC and ELI must position against). Entries are dated. Paper figures below
are as reported by a SciSpace agent summary, not re-read from the PDFs; arXiv IDs and the
DOI are as the summary gave them.

Why it matters here: `../../potential-projs/text-latent-code-autoencoder.md` (TLC) treats
prompts as the only learned object and compresses code through a text latent; its
prior-art gate already names LLMLingua and gist tokens. `../../potential-projs/elicitation-gain.md`
(ELI) runs an outer optimizer over the interface with a fixed budget. The three papers
below are direct prior art for both: discrete, gradient-free prompt compression (PCRL,
Nano-Capsulator) and budget-accounted evolutionary prompt optimization for code (EPiC).

**Artifacts on disk (Danielle moved them to a permanent location):**
`~/drotherm/data/convo-artifacts/2026/scispace-prompt-compression-method-papers-agent-artifacts-zip_180fe9cc-24fb-4ab2-8dc5-61a6611f64ce_1787422486/`
— the three PDFs (`paper1_discrete_rl.pdf`, `paper2_natural_language.pdf`,
`paper3_evolutionary.pdf`), per-paper analyses, three summary variants, SciSpace metadata
(`target_papers_info.json`), and ~2,000 rows of search archives across SciSpace full-text,
SciSpace, Scholar, and arXiv queries (`*_prompt_compression_{1,2,3}.csv`,
`combined_prompt_compression_papers.csv`). The CSVs are the useful part for a later
prior-art pass: a seeded candidate list already exists.

---

## 2026-08-22 — SciSpace summary of three papers (search undated, ~2026)

**Danielle's prompt (verbatim):**

> Summarize the key arguments in these 3 papers: Learning to Compress Prompt in Natural
> Language Formats, Automated Prompt Engineering for Cost-Effective Code Generation Using
> Evolutionary Algorithm, Discrete Prompt Compression with Reinforcement Learning

**Surviving content, condensed.**

- **PCRL — Jung & Kim 2024, IEEE Access, DOI 10.1109/ACCESS.2024.3403426.** Discrete
  token-level prompt compression as sequence labeling: a policy (binary MLP on frozen
  DistilBERT) marks tokens include/exclude; reward = 1 − |p''|/|p| if ROUGE-L ≥ τ else −λ;
  trained by policy gradient (SCST) — no gradient access to the target LM, no labels, so
  black-box-API compatible. ~24.6% average compression (22.7% GPT2-XL, 26.4% FLAN-T5-XL);
  beats Selective Context and stopword removal; policy trained on small LMs transfers to
  LLaMA-2-7B, Falcon-7B, FLAN-T5-XXL, GPT-3.5-Turbo. Removed tokens are mostly stopwords,
  punctuation, word endings. Limitations as stated: extractive only (no paraphrase);
  ROUGE is a weak semantic proxy; omission can induce hallucination.
- **Nano-Capsulator — Zhou et al. 2024, arXiv 2402.18700.** Compress long prompts into
  natural-language "Capsule Prompts" that transfer across LLMs including APIs. Generator
  initialized from Vicuna-7B with LoRA; semantic-preservation loss (MSE between embeddings
  of original and capsule) multiplied by a utility reward with a hard length cutoff,
  reward from downstream task score change (embedding MSE / accuracy / GPT4Eval).
  Reported: 81.4% compression on CSQA (831 → 154 tokens), 4.5× latency reduction, 80.1%
  budget saving on TriviaQA-Long for Claude 2 and PaLM; 19.7% on GSM8K vs. 3.79% for
  AutoCompressors. Limitations: length–information trade-off, ~8 h on A40s, scoped to
  few-shot CoT and reading comprehension.
- **EPiC — Saluja et al. 2024, arXiv 2408.11198.** Evolutionary prompt optimization for
  code generation with explicit cost accounting. Initial-evaluation phase (run tests; stop
  if passing), then an evolutionary phase: LLM builds an initial population of prompt
  variants; fitness = test pass ratio; fitness-weighted selection; mutation by
  LLM-as-mutator or by synonym substitution (WordNet/GloVe — the cheaper and more
  cost-effective one); population 5–8 optimal. New metric **ATSP** (additional tokens per
  solved problem) = (T_m − T_b) / ((P_m − P_b) × N). Reported pass@1 gains of 2–10% on
  HumanEval+, MBPP+, BigCodeBench-Hard with o3-mini / Claude 3.7 Sonnet / DeepSeek V3;
  ATSP 20k vs. Reflexion 38k / LDB 196k / LATS 275k on HumanEval+ ("13× more
  cost-effective than LATS"). Limitations: three LLMs, LLM-generated tests, no
  hyperparameter tuning.
- Cross-paper points worth keeping: all three are gradient-free and discrete, so they
  work against black-box models and transfer across models; each uses a task-specific
  reward; the useful axis for TLC/ELI is *extractive* (PCRL) vs. *abstractive / NL
  re-generation* (Nano-Capsulator) vs. *prompt search with budget accounting* (EPiC).

**Intake notes.**

- The summary's "Practical implications" / "prompt compression is mature" / "discrete
  optimization is the future" sections are agent editorializing, dropped.
- ATSP is the closest published analog of ELI's fixed-budget accounting and TLC's
  "cost of the optimizer loop"; worth adopting or explicitly contrasting rather than
  inventing a parallel metric.
- Nano-Capsulator's "utility reward × semantic loss" is structurally the TLC objective
  with a different latent target (a shorter prompt rather than a text latent for code);
  TLC's prior-art gate should state the difference (TLC reconstructs; Capsule preserves
  downstream utility).
- EPiC's synonym-substitution mutator beating the LLM mutator on cost is a data point
  for ELI's outer-optimizer design (cheap mutators first).
- Figures and IDs unverified; PDFs are on disk for verification.

## 2026-08-24 — the 2026-02-04 Undermind novelty check: a 21-paper prompt-compression map (historical; unverified)

Danielle's sixth Feb-2026 novelty check (Undermind platform; verbatim multi-file
packet at
`~/drotherm/data/convo-artifacts/2026/2026-08-24-notion-lit-reviews/undermind-prompt-compression-review/`).
It maps the prompt-compression field along four dimensions (bottleneck/decoder
role; optimization regime; extractive-vs-generative bottleneck text; evaluation
target) and four timeline eras (2023 heuristic Selective Context → 2023–24
extractive-pruning-to-learned-encoders+RL → 2024–25 generative/meta-prompting →
2024–25 attention/model-internal). Core set of 21 papers; the RL/gradient-free
subset it emphasizes: PCRL (the closest analogue — RL-edited prompts scored by
output-distribution divergence vs. the original prompt's behavior), TACO-RL,
PIS, GPT-C, Cmprsr (GRPO), LLM-DCP, LanguaShrink; training-free search:
Style-Compress, DSPC, SCOPE, PartPrompt, EHPC, AttnComp. Foundational/adjacent
sub-pages add Selective Context 2310.06201, LongLLMLingua 2310.06839, Fei et al.
context-window semantic compression 2312.09571, LLMLingua-2, R2C, CPC,
"Fundamental Limits of Prompt Compression: A Rate-Distortion Framework for
Black-Box LLMs" (name matches plan row C's Girish rate-distortion item — confirm
same paper at verification), CompressionAttack (prompt compression as an attack
surface), and others (most name-only in the source; per-paper Notion pages exist
if IDs are needed later). Gap statement, verbatim: "No paper in this set encodes
programs, policies, or other structured semantics into short text codes and then
decodes them via a frozen LLM to reconstruct the original behavior in a formal
equivalence sense."

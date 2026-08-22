# HumanEvalExplain — published results and what they can and cannot support

**Kind:** reference (accumulator for results on the explain→regenerate subtask of
HumanEvalPack, OctoPack, Muennighoff et al. 2023, arXiv 2308.07124). Numbers are as
transcribed by a SciSpace agent from three papers, not re-read from the PDFs; the PDFs
and the full search archive are on disk.

Why it matters here: HumanEvalExplain *is* a text-latent autoencoder protocol for code —
the model writes a natural-language explanation of a function, then must regenerate the
function from its own explanation, scored by pass@1. That is TLC's reconstruction loop
(`../../potential-projs/text-latent-code-autoencoder.md`) with the explanation as the
latent and no length pressure. Danielle's stated goal for this search was a
correctness-vs-explanation-length plot across models and prompt formats — i.e. the TLC
rate–distortion curve, read off published numbers if possible.

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-humanevalexplain-results-agent-artifacts-zip_6e33d176-efc7-49db-8322-a5b1604bfd20_1787422834/` — per-paper answer files, the code-summarization PDF, arXiv
PDFs, the HumanEvalExplain/OctoPack search CSVs, and the four plots
(`plot1_avg_pass1_by_model.png` … `plot4_prompt_strategy.png`).

---

## 2026-08-22 — SciSpace deep search (undated, ~2026)

**Danielle's prompt (verbatim):**

> I wish to find all papers that report performance results on the HumanEvalExplain
> subtask of the HumanEvalPack as introduced by "OctoPack: Instruction Tuning Code Large
> Language Models" 2023 from Muennighoff et al.

Clarification she gave the agent (verbatim):

> I'm interested in all models, all language covered, and any papers since the release of
> OctoPack. I want to build a comparison plot of correctness versus explanation length
> based on different forms of applying models (eg different prompt formats, etc) and
> different models used.

**What the search found.** Of 304 deduplicated papers (SciSpace ×3, Scholar, arXiv; 35
PDFs read), only **three** report HumanEvalExplain numbers: OctoPack itself (2308.07124),
WaveCoder (Yu et al. 2023, 2312.14187), and "Large Language Models for Code
Summarization" (Szalontai et al. 2024, 2405.19032). Eight other candidates use
HumanEvalPack for Fix/Synthesize only or other summarization datasets (Summary-Mediated
Repair 2511.18782, CodeMind, XCoder, Selective Shot Learning 2412.12852, LeDex
2405.18649, SelfCodeAlign 2410.24198, Crystal 2411.04156, InstructCoder 2310.20329).

**Protocol (as stated by OctoPack).** 164 problems × 6 languages (Python, JavaScript,
Java, Go, C++, Rust); explanation generated, then code regenerated from the explanation
alone; pass@1 with n=20, T=0.2, top-p 0.95 (GPT-4 n=1). Zero-shot; OctoCoder/OctoGeeX
use `Question: … Answer: {function_start}`; WaveCoder's CodeAlpaca variants use the
Alpaca format.

**Average pass@1 (6-language mean unless noted; source in brackets):**

| Model | Size | Avg | Python | Notes |
|---|---|---|---|---|
| GPT-4 | – | 52.1 | 64.6 | [1]; n=1 |
| DeepSeek-instruct | 6.7B | 51.0 | 62.2 | [4]; greedy |
| WaveCoder-Ultra | 6.7B | 47.3 | 56.7 | [4] |
| MagiCoder-S-DS | 6.7B | 46.1 | 60.3 | [4] |
| WaveCoder-DS / -Pro | 6.7B | 41.3 / 41.3 | 48.2 / 53.0 | [4] |
| MagiCoder-DS | 6.7B | 40.7 | 55.5 | [2],[4] |
| WaveCoder-CL | 13B | 37.9 | 45.7 | [4] |
| DeepSeekCoder (base) | 6.7B | 34.6 | 43.9 | [4] |
| DeepSeek-CodeAlpaca | 6.7B | 34.0 | 40.8 | [4] |
| WaveCoder-CL | 7B | 32.4 | 41.4 | [4] |
| WaveCoder-SC | 15B | 30.8 | 37.1 | [4] |
| CodeLlama-instruct | 13B / 7B | 28.2 / 27.3 | 40.2 / 33.5 | [4] |
| CodeLlama-CodeAlpaca | 7B / 13B | 28.0 / 27.3 | 34.7 / 32.3 | [4]; Alpaca format |
| WizardCoder | 15B | 27.5 | 32.5 | [1] |
| OctoCoder | 16B | 24.5 | 35.1 ([1]) / 26.7 ([4]) | Q/A format |
| OctoGeeX | 6B | 22.9 | 30.4 | Q/A format |
| StarChat-β | 16B | 20.1 | 25.4 | [1] |
| BLOOMZ | 176B | 7.5 | 14.7 | [1] |
| InstructCodeT5++ | 16B | 3.5 | 20.8 | [1]; Python only nonzero |
| StarCoder, CodeGeeX2 (base) | 15B / 6B | 0.0 | 0.0 | [1],[4]: base models score zero |
| Llama3-8B-instruct | 8B | – | 42.7 | [2]; Python only |

Per-language pattern across models: Python > JS ≈ Java > Go ≈ C++ > Rust. Base code
models without instruction tuning score 0 everywhere. Alpaca-format vs. instruct-model
differences within CodeLlama are ±1 point (no real prompt-format effect measurable).

**Intake notes.**

- **The search cannot deliver Danielle's plot.** No paper reports explanation length, and
  none varies prompt format systematically (the agent's own §7.5 and §9.1 say so). A
  correctness-vs-length curve needs the generated explanations, which means running the
  HumanEvalPack harness (bigcode-evaluation-harness) with logged explanations — which is
  TLC-1's census with HumanEvalExplain as a ready-made task set. That is the actionable
  outcome: HumanEvalExplain is a free, standardized, six-language TLC-1 instance, and the
  published numbers are the "no length pressure" endpoint of the rate–distortion curve.
- **Transcription error in the GPT-4 and WizardCoder rows attributed to WaveCoder [4].**
  The [4] GPT-4 row (57.3, 51.2, 58.5, 38.4, 42.7, 52.1) is OctoPack's row shifted one
  column left (JS→Py … Avg→Rust); likewise WizardCoder's [4] row is a permutation of [1]'s.
  Consequences: "GPT-4 Rust 52.1 [4]" and "DeepSeek-instruct beats GPT-4 on Java/C++" in
  the report are artifacts of the shift, not results. Whether the shift is in the
  WaveCoder paper or the agent's extraction is checkable from the PDF on disk.
- The "cross-language consistency" section is internally contradictory (the "least
  consistent" σ values are lower than the "most consistent" ones). Dropped.
- The report's §8–§9 (why few papers report, future directions, recommendations) is
  agent editorializing; nothing retained beyond the factual gap statements.
- Coverage is as of the search; models the agent notes as unevaluated on this task:
  Claude, Gemini, Qwen-Coder, StarCoder2. Unverified.

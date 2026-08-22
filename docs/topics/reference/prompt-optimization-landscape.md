# Prompt optimization — landscape reference (LLM-as-optimizer and system-level)

**Kind:** reference (accumulator for the automatic-prompt-optimization subfield as it bears
on TLC's optimizer loop and ELI's outer optimizer of the interface). Entries are dated.
Characterizations are the SciSpace agent's; identifiers unverified. Sibling:
`prompt-compression-and-optimization-literature.md` (PCRL, Nano-Capsulator, EPiC in
detail) and `nl-bottleneck-prior-art.md` (the TLC prior-art verdicts).

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-prompt-optimization-agent-artifacts-zip_97f96bdd-5f03-456a-8061-f2e8204d51c2_1787423566/` — final report, an earlier draft, the insight-extraction
pass, and ~25 search CSVs, two merged with per-paper extracted method-category /
contribution / domain / limitations columns and one 648-row deep-search table.
**`INDEX.md` inside the folder is the file-level index**, including the canonical works
the review omits.

---

## 2026-08-22 — SciSpace deep review (undated, ~early 2026)

**Danielle's prompt (verbatim):**

> What is the current state of the Prompt Optimization subfield? What are the related
> works that are foundational, and what are the strongest recent themes? Please focus
> especially on approaches that use LLMs to do the prompt optimization and on approaches
> that optimize not just the prompt but also other aspects of the system (for example
> sampling hpms, tool use, etc). My goal is to get a general understanding of the subfield
> before diving deeper into specifics, so prioritize breadth and qualitative comparisons.

Scope answers (verbatim):

> Please include as broad a set of methods as possible, I'd say focus mostly on things like
> prompt optimization, so RAG systems are a bit further afield for this deep research. I'm
> interested across broad domains, but especially in the realm of verifiable tasks like
> code generation. And focus on recent developments but provide a brief intro on the
> historical perspective.
>
> For context, I'm currently writing a paper on compressing functions using an frozen
> pretrained black box LLMs as encoders and decoders, with instructions to compress the
> input function and then reconstruct it in a way that preserves functionality not surface
> form. We're using an external LLM to optimize the compression prompt. So I'm interested
> in things related to this but more interested in getting a broad understanding of the
> landscape.

**The map the report gives (condensed).**

- *History:* manual heuristics (pre-2023) → Promptbreeder's self-referential evolution
  (2023; evolves the mutation prompts too) → diversification (meta-prompting,
  gradient-inspired, system-level, code-specific; 2024–25) → compound-system
  optimization (2025–26).
- *Method families and representatives:* evolutionary / population (Promptbreeder
  2309.16797, PhaseEvo 2402.11347, AMPO 2410.08696); meta-prompting /
  generation-refinement with a critic (PromptWizard 2405.18369; Tang et al. "LLMs as
  prompt optimizers ≈ gradient optimizers", update direction + update method, 2402.17564;
  Learning from Contrastive Prompts 2409.15199); gradient-inspired textual updates
  (LLM-AutoDiff 2501.16673 — textual gradients through multi-component workflows;
  Dual-Phase accelerated 2406.13443); query-dependent learned generators (QPO
  2408.10504, offline RL); hybrids (Davari et al. reinforcement + diversification +
  migration for black-box LLMs, 2507.09839).
- *Joint instruction × exemplar optimization* as a settled finding (Wan et al. "Teach
  better or show smarter", 2406.15708).
- *System-level:* SPRIG (2410.14826) edit-based genetic search over system-prompt
  components — one optimized system prompt ≈ task-specific prompts across 47 tasks;
  Trace / OptoPrime (2406.16218) — workflows as graphs, an LLM optimizer over
  heterogeneous parameters incl. prompts, hyperparameters, and code; LLM-AutoDiff;
  RePrompt (2406.11132) for agent planning instructions from interaction histories; Lin
  et al. survey of LLM-based optimization of compound AI systems (2410.16392) listing
  targets: prompt components, sampling parameters, retrieval configs, tool specs,
  orchestration logic.
- *Verifiable / code:* Prochemy (2503.11085, execution-driven refinement), EPiC
  (2408.11198, cost-aware evolutionary), PromSec (security + functionality loop), Midolo
  et al. empirical prompt guidelines, Shashikala et al. accuracy + diversity objectives.
- *Feedback taxonomy:* validation accuracy; LLM-as-critic; execution feedback;
  contrastive (success vs. failure prompts).
- *Open problems the report lists:* evaluation protocol inconsistency; cross-model
  transfer of optimized prompts; sample efficiency; overfitting to validation sets;
  prompt brittleness; prompt length vs. performance ("prompt compression" as an open
  problem — the link to TLC); integration with fine-tuning.

**Intake notes.**

- For a "foundational works" question the review's omissions are severe: no APE, OPRO,
  ProTeGi/APO, TextGrad, DSPy/MIPROv2, GEPA, EvoPrompt, or RLPrompt. Its foundations are
  Promptbreeder plus surveys. The 648-row deep-search table on disk is the place to
  recover the missing anchors; the sibling prior-art bundle already has RLPrompt and
  SPELL.
- The most useful content for this repo is the system-level cluster — Trace/OptoPrime,
  LLM-AutoDiff, SPRIG, and the compound-AI survey's target list — because "optimize the
  harness, not just the prompt" is exactly the ELI outer loop and TLC's θ (prompts,
  templates, latent format, stage decomposition, tool use, sampling). These are the
  positioning references.
- The "implications for function compression" section is the agent restating
  Danielle's own context; nothing new, dropped except the prompt-length-vs-performance
  framing.
- Citation hygiene: [6] appears under two names; [5] in two roles; the comparison
  table's header is malformed. All identifiers as produced.

# text latent code autoencoder — high-recall related-work corpus

**Kind:** per-project recall corpus (two-tier convention, 2026-08-24). This file is
the *highest-recall* enumeration: every paper, method, or named prior-art item on
record in this repository that is possibly relevant to
[`text-latent-code-autoencoder.md`](../text-latent-code-autoencoder.md) — err-toward-inclusion by design. The curated
precision cut and Danielle's positioning live in that doc's §5; this file is the
working corpus behind it. Agent-maintained: intake adds new items here (and to §5
only when anchor-tier). **Nothing here is verified**; sources include
agent-generated records (SciSpace reviews, novelty checks, NBLM tables) marked
in-line. Each item cites where it sits on record.

*Recall corpus for TLC (`text-latent-code-autoencoder.md`). Highest-recall inventory of
every paper, method, tool, benchmark, or named prior-art item on record anywhere in this
repository that could bear on the project. Grouped by the concerns of the litreview plan's
rows A–F (`../../litreview/tlc-litreview-plan.md`). Err toward inclusion; one line each.
Nothing here is verified — most identifiers are agent-supplied (SciSpace, six Feb-2026
novelty checks, NotebookLM summary tables) and are flagged in-line as **unverified**; see
`../../litreview/citation-verification-ledger.md` for per-ID provenance. No positioning or
novelty claims are made here; roles and attributions only.*

**Row A — mechanism prior art: NL/text bottlenecks with frozen decoders, program
autoencoders, language-as-latent**

- **Language Bottleneck Models (LBM)** (2506.16982; Berthon & van der Schaar) — encoder LLM
  emits a short NL summary, frozen LLM decodes from it; encoder trained by GRPO against
  frozen-decoder performance; headline match of two independent Feb-2026 checks (Gemini,
  Perplexity); Danielle's read: prior work, nearest named framework, but non-verifiable
  grading target and ~3 prompts, no optimization loop — unread gate-1 item (source:
  docs/topics/reference/nl-bottleneck-prior-art.md; docs/litreview/tlc-litreview-plan.md).
- **GRPO / DeepSeekMath** (2402.03300; Shao et al.) — LBM's optimizer; group-relative
  scoring of candidate summaries against a frozen decoder, no value network; unverified
  (source: docs/topics/reference/nl-bottleneck-prior-art.md, 2026-08-24 Gemini entry).
- **GenDLN** (ACL SRW 2025, DOI 10.18653/v1/2025.acl-srw.92; no arXiv ID) — evolutionary
  *joint* prompt optimization over *stacked* frozen LLMs; the only three-keyword
  "high priority" hit the Dec-2025 SciSpace agent dismissed in one line; structurally the
  closest thing to the encoder+decoder harness search; open gate-1 must-read (source:
  docs/topics/reference/nl-bottleneck-prior-art.md; docs/litreview/tlc-litreview-plan.md).
- **Miao & Blunsom 2016, "Language as a Latent Variable"** (1609.07317; EMNLP 2016) —
  auto-encoding sentence compression: latent summary sentence drawn from a background LM
  prior, observed sentence reconstructed from it, discrete VAE via REINFORCE; recorded as
  the closest formal ancestor; advisor-recommended, PDF in hand, metadata verified against
  the PDF (source: docs/topics/reference/nl-bottleneck-prior-art.md, 2026-08-23 entry;
  text-latent-code-autoencoder.md §1 and §4 submission-record entry).
- **Nano-Capsulator** (2402.18700; Zhou et al. per SciSpace, first author Chuang per the
  prior-art bundle — attribution conflict on record) — generator rewrites long prompts into
  NL "capsule prompts" under a semantic-preservation loss × downstream-utility reward with a
  hard length cutoff; top partial match (4/6) on the Dec-2025 rubric; structurally closest
  objective, differing in target (source:
  docs/topics/reference/prompt-compression-and-optimization-literature.md;
  docs/topics/reference/nl-bottleneck-prior-art.md).
- **PCRL** (2308.08758; Jung & Kim, IEEE Access, DOI 10.1109/ACCESS.2024.3403426) — discrete
  token-level prompt compression as sequence labeling; policy gradient (SCST), black-box-API
  compatible, ~24.6% compression; the Undermind check calls its objective (divergence between
  the frozen model's output distributions under original vs. compressed prompt) the nearest
  formal cousin of behavioral reconstruction (source:
  docs/topics/reference/prompt-compression-and-optimization-literature.md).
- **Latent Programmer** (2012.00377) — discrete trained token latents for program synthesis;
  medium partial on the Dec-2025 rubric (source: docs/litreview/tlc-litreview-plan.md row A;
  docs/topics/reference/nl-bottleneck-prior-art.md).
- **Sentence Bottleneck Autoencoders** (2109.00055; Montero et al., ACL 2021.emnlp-main.137)
  — frozen transformers with a soft-vector bottleneck; the vector-latent counterpart
  (source: docs/litreview/tlc-litreview-plan.md row A; nl-bottleneck-prior-art.md).
- **ICAE** (2307.06945; Ge et al.) — in-context autoencoder, embedding memory slots as the
  latent; medium partial (source: docs/litreview/tlc-litreview-plan.md row A;
  docs/topics/reference/code-compression-literature.md).
- **RLPrompt** (2205.12548; Deng et al.) — RL over discrete text prompts; 3/6 partial on the
  Dec-2025 rubric; ID recovered from the Feb-2026 Gemini check (source:
  docs/topics/reference/nl-bottleneck-prior-art.md; litreview plan row A).
- **SPAE** (2306.17842; Yu et al.) — frozen-LLM autoencoding of images via *lexical tokens*;
  the existing non-prose rung on the NL-likeness axis ("SPAE for code" analogy from the
  Gemini check); unverified (source: docs/topics/reference/nl-bottleneck-prior-art.md).
- **SPELL** (2310.01260) — medium partial on the Dec-2025 rubric (source: litreview plan
  row A).
- **Proto-tokens** (2502.13063; Kuratov et al., ACL 2025) — frozen-LLM reconstruction of
  ~1.5k tokens from 1–2 trained embeddings; evidence reconstruction capacity does not require
  an NL latent — directly bears on the NL-necessity falsifiers (source:
  docs/topics/reference/nl-bottleneck-prior-art.md, Claude-check entry).
- **de Bruin et al., "Autoencoders as Tools for Program Synthesis"** (2108.07129) — program
  VAE with gradient-free evolutionary search over a *neural* latent; the non-NL-latent
  counterpart of the setup (source: docs/topics/reference/nl-bottleneck-prior-art.md).
- **Concept Bottleneck LLMs (CB-LLM)** (2412.07992) — text concepts as a bottleneck for
  classification; interpretability lineage, not reconstruction (source:
  docs/topics/reference/nl-bottleneck-prior-art.md).
- **Text Bottleneck Models (TBM)** (2310.19660; Ludan et al.) — text bottleneck for
  interpretable classification; ID recovered from the Perplexity check's Sources section
  (source: docs/topics/reference/nl-bottleneck-prior-art.md, Perplexity entry).
- **OverLang, "Teaching LLMs to Speak in Pseudocode for Efficient Compression"** (no ID;
  agents4agents.ai PDF, labeled "(blog)" in the Perplexity summary table) — RL-trained
  pseudocode shorthand decodable by other models; provenance weak, lead only; litreview plan
  says locate or discard (source: docs/topics/reference/nl-bottleneck-prior-art.md).
- **APRIL** (2509.25196) and **Proof2Silicon** (2509.06239) — RL prompt optimization for
  frozen code/hardware generators; classed by the source check as "the decoder half only"
  (source: docs/topics/reference/nl-bottleneck-prior-art.md).
- **CyclePrompt** (2402.08756; Diesendruck et al. 2024) — cycle-consistency refines prompts
  for code generation; in the submitted LLA bibliography; the Gemini check's third
  objectives-contrast axis (generation quality) (source: docs/litreview/citation-
  verification-ledger.md; docs/topics/reference/nl-bottleneck-prior-art.md).
- **CodeCloak** (2404.09066) — DRL prompt manipulation to *prevent* code reconstruction; the
  adversarial dual of the leakage thread (source: docs/topics/reference/nl-bottleneck-prior-
  art.md, Gemini entry).
- **SelfCP** (2405.17052; Gao et al.) — compressing over-limit prompts via the frozen LLM
  itself; a Consensus-check closest match (source: docs/topics/reference/nl-bottleneck-prior-
  art.md, Consensus entry).
- **NL in the Middle** (2507.08627; "Wong et al." per the ledger vs. "Tai, Nie, Golab &
  Wong, CASCON 2025" per the Consensus check — authorship discrepancy flagged) — code
  translation with NL intermediate representations; scored 5.5\* in the aggregate table
  (single-source estimate) (source: docs/topics/reference/nl-bottleneck-prior-art.md;
  citation-verification-ledger.md).
- **NL-Debugging** (2505.15356; Zhang et al. 2025) — translate buggy code to NL, refine in
  NL, regenerate; TLC's loop applied to repair; one of the two independent mechanism
  neighbors named by the ChatGPT check (source: docs/topics/reference/nl-bottleneck-prior-
  art.md; text-latent-code-autoencoder.md §5 anchor set).
- **PlanSearch** (2409.03733; Wang et al. 2024) — planning in natural language improves LLM
  search over code; the second independent NL-intermediate + frozen-decoder + search
  convergence (source: docs/topics/reference/nl-bottleneck-prior-art.md, ChatGPT entry).
- **Gilbert et al., "Semantic Compression with LLMs"** (2304.12512; ID conflict flagged —
  the Perplexity check links the same page to 2406.01989) — LLMs compress text/code into
  short representations preserving functional equivalence via manual prompting, no joint
  optimization; recorded in the §5 anchor set as the closest prior art (pipeline
  feasibility, three fixed prompts, no loop) (source: docs/topics/reference/nl-bottleneck-
  prior-art.md; code-compression-literature.md; text-latent-code-autoencoder.md §5).
- **GPT-C (Generative PrompT Compression)** (no arXiv ID on record) — from the Undermind
  review's RL subset; scored **4.5/6 with applied-to-programs ✅** in the SciSpace NBLM
  summary table, the highest match score in any scored table; agent-assigned and unverified
  (source: docs/topics/reference/nl-bottleneck-prior-art.md, table-layer entry;
  text-latent-code-autoencoder.md §5).
- **TACO-RL, PIS, Cmprsr (GRPO), LLM-DCP, LanguaShrink** — the Undermind review's RL /
  gradient-free prompt-compression subset; name-only in source (source:
  docs/topics/reference/prompt-compression-and-optimization-literature.md, 2026-08-24 entry).
- **Style-Compress, DSPC, SCOPE, PartPrompt, EHPC, AttnComp** — the Undermind review's
  training-free-search prompt-compression subset; name-only (source: same file).
- **Selective Context** (2310.06201), **LongLLMLingua** (2310.06839), **LLMLingua**
  (2310.05736), **LLMLingua-2**, **R2C**, **CPC**, **Fei et al. context-window semantic
  compression** (2312.09571), **CompressionAttack** — the Undermind review's
  foundational/adjacent prompt-compression tier; LLMLingua is also on TLC's original
  prior-art gate stub list (source: docs/topics/reference/prompt-compression-and-
  optimization-literature.md; nl-bottleneck-prior-art.md, Consensus entry).
- **Hidden CoT** (2409.08561; Liu et al.) — compressed chain-of-thought decoding; scored in
  the 4.5 band of the aggregate table (source: docs/topics/reference/nl-bottleneck-prior-
  art.md, Consensus entry).
- **CompLLM** (2509.19228), **Zip2Zip** (2506.01084; inference-time adaptive tokenization),
  **ReFIne** (2510.09062) — Gemini-check additions on the compression/tokenization flank
  (source: docs/topics/reference/nl-bottleneck-prior-art.md, Gemini entry).
- **FAST** (2501.09747; Pertsch et al.) — VLA action tokenization; the
  frozen-LLM-interprets-latent example in the Perplexity check's component table (source:
  docs/topics/reference/nl-bottleneck-prior-art.md, Perplexity entry).
- **LINT — Assessing the Interpretability of Programmatic Policies with LLMs** (no ID) —
  explain→regenerate formalized as an interpretability *score*; "LINT provides the metric"
  (source: docs/topics/reference/nl-bottleneck-prior-art.md, Gemini entry).
- **MIPS** (program synthesis via mechanistic interpretability), **APICoder**
  (private-library code generation), **SAMMO** (symbolic prompt program search) — category
  C/D items new from the Dec-2025 verdict's Prior Work Supplemental; no IDs in source
  (source: docs/topics/reference/nl-bottleneck-prior-art.md, 2026-08-24 addendum).
- **SAPS** (1810.09717; "Ain't Nobody Got Time For Coding") — tree-to-tree structure-aware
  program synthesis; 1.5/6 in the SciSpace summary table (source: same file, table-layer
  entry).
- **BART** (1910.13461) — denoising autoencoder for text, offered as the many-surface-forms →
  regular-form analogy for behavior-preserving normalization (source:
  text-latent-code-autoencoder.md §4 chunk 12; citation-verification-ledger.md).
- **VQ-VAE** (no ID on record) — discrete codes indexing a learned codebook matching
  continuous latents; the lineage §1 names for "text tokens as compositional discrete codes,
  frozen LLM as an enormous pretrained codebook"; unverified pointer (source:
  text-latent-code-autoencoder.md §1 and §4 2026-07-11 Point 3).
- **MUNIT / cycle-consistency style transfer** (no ID) — unverified lineage pointer for the
  factored function/style cross-decoding objective (source: text-latent-code-autoencoder.md
  §4 2026-07-11 Point 2).
- **"Conversion of Neural Networks into Logic Flows"** and **"Exploring Reasoning Reward
  Model for Agents"** — best new matches (2/6) in the Jan-2026 SciSpace update (source:
  docs/topics/reference/nl-bottleneck-prior-art.md).
- **Wei et al. 2019, code generation as the dual task of summarization** (ledgered as
  1910.05923 via the submitted bibliography; red-flagged as unresolved in the ChatGPT check's
  Notion source) (source: docs/topics/reference/nl-bottleneck-prior-art.md; ledger).
- **"RL4Prompt"** — name unresolved in the ChatGPT check; possibly RLPrompt (source:
  docs/topics/reference/nl-bottleneck-prior-art.md, ChatGPT entry).
- **Makharev & Ivanov 2025, code summarization beyond function level** and **Poudel et al.
  2024, DocuMint: docstring generation with small LMs** (no IDs) — Danielle-supplied rows
  from the submitted LLA bibliography (source: docs/litreview/citation-verification-ledger.md).
- **"Equivalent representations of code"** (OpenReview RMaB6cn07S) — comments, pseudocode,
  flowcharts as equivalent code representations (source: docs/topics/reference/
  nl-bottleneck-prior-art.md, measurement entry).
- **Generating Equivalent Representations of Code by Self-Reflection** (2410.03351; Li et
  al.) — Consensus-check survey-toggle standout (source: docs/topics/reference/
  nl-bottleneck-prior-art.md).
- **The novelty-check corpus itself** — six Feb-2026 checks in one ~7-hour session with six
  distinct verdicts (ChatGPT "published in parts"; Claude "novel 85–90% by combination";
  Consensus "novel 90–95%"; Gemini "partially novel"; Perplexity "equivalent method already
  published"; Undermind "poised for direct extensions"), plus the Dec-2025 SciSpace verdict
  ("appears novel," 378 papers, six-component rubric) and its Jan-2026 update — all
  agent-generated and unverified; Danielle's meta-note is that verdict tracks each review's
  prompt interpretation, not the literature (source: docs/topics/reference/
  nl-bottleneck-prior-art.md; text-latent-code-autoencoder.md §4 and §5).
- **The seven NotebookLM summary tables + the Aggregate Summary** — per-review distillations
  in a six-column schema (Paper | Year | Intermediate Representation Type | Optimization
  Method | Decoder Type | Key Innovation); the Claude table adds Critical Gaps and a tiered
  Overlap Percentage column; the SciSpace table (18 rows, 12 columns) adds "Applied to
  Programs?" and X/6 scores; the merged **nl-latents-aggregate-summary.md** is "the closest
  existing thing to a completed §5 comparison table" (top tiers LBM 5.5\*, NL in the Middle
  5.5\*, NL-Debugging 5.0\*; 4.5 band Gilbert, PlanSearch, RTC, DreamCoder/AutoDoc,
  FunSearch, GPT-C, Hidden CoT\*, SPAE\*). All scores agent-generated, dedup incomplete
  (Nano-Capsulator and EPiC appear twice) — **unverified** (source: docs/topics/reference/
  nl-bottleneck-prior-art.md, table-layer and aggregate entries).

**Row B — optimizer: LLM-as-optimizer, prompt search, system-level optimization**

- **GEPA** (2507.19457) — "Reflective Prompt Evolution Can Outperform Reinforcement
  Learning"; samples trajectories, reflects on failures in text, proposes prompt edits,
  combines lessons on a Pareto frontier; the named TLC-2 incumbent and the omission LLA
  Reviewer 3 dinged (source: docs/topics/reference/prompt-optimization-landscape.md;
  text-latent-code-autoencoder.md §3 and §4 submission record).
- **DSPy / COPRO / MIPROv2** (no IDs on record) — the propose–evaluate–select loop TLC-2
  starts from; COPRO's known failure mode is proposal mode collapse into paraphrases of the
  incumbent; MIPROv2 is on the missing-anchor list (source: text-latent-code-autoencoder.md
  §3 and §4; docs/litreview/tlc-litreview-plan.md gate 3).
- **OPRO** (2309.03409; Yang et al.) — LLMs as optimizers; trajectory-in-context conditioning
  on prompt–score history; the TLC-3 second arm (source: docs/topics/reference/
  prompt-optimization-landscape.md).
- **"Revisiting OPRO" / The Limitations of Small-Scale LLMs as Optimizers** (2405.10276; ACL
  Findings 2024) — small open models are weak optimizers; the template for Danielle's
  cheap-model re-evaluation angle (source: docs/topics/reference/prompt-optimization-
  landscape.md, 2026-08-23 entry).
- **APE, ProTeGi/APO, TextGrad** (no IDs on record) — the remaining missing prompt-
  optimization anchors; ProTeGi/TextGrad flavor the diagnosis-driven TLC-3 arm (source:
  docs/litreview/tlc-litreview-plan.md gate 3; text-latent-code-autoencoder.md §3).
- **EPiC** (2408.11198; Saluja et al. / "Taherkhani et al. 2025" in the pitch bibliography;
  slip: also given as 2410.14321) — evolutionary prompt optimization for code generation with
  the **ATSP** (additional tokens per solved problem) cost metric; the published analog of
  the optimizer-cost accounting; 3.5/6 partial (source: docs/topics/reference/
  prompt-compression-and-optimization-literature.md; prompt-optimization-landscape.md).
- **Promptbreeder** (2309.16797) — self-referential evolution, evolves the mutation prompts
  too (source: docs/topics/reference/prompt-optimization-landscape.md).
- **EvoPrompt** (2309.08532; Guo et al.) — evolutionary prompt optimization; distinct from
  EvoPrompting (source: docs/topics/reference/nl-bottleneck-prior-art.md, Claude entry).
- **EvoPrompting** (2302.14838; Chen et al.) — neural-architecture prompts (source:
  docs/litreview/tlc-litreview-plan.md row A).
- **PromptWizard** (2405.18369) — meta-prompting / generation-refinement with a critic
  (source: docs/topics/reference/prompt-optimization-landscape.md).
- **Trace / OptoPrime** (2406.16218) — workflows as graphs; an LLM optimizer over
  heterogeneous parameters incl. prompts, hyperparameters, and code; a nearest published
  relative of harness-level optimization (source: same file).
- **LLM-AutoDiff** (2501.16673) — textual gradients through multi-component workflows
  (source: same file).
- **SPRIG** (2410.14826) — edit-based genetic search over system-prompt components; one
  optimized system prompt ≈ task-specific prompts across 47 tasks (source: same file).
- **Lin et al., survey of LLM-based optimization of compound AI systems** (2410.16392) — the
  vocabulary for "optimize the whole pipeline"; target list includes prompt components,
  sampling parameters, retrieval configs, tool specs, orchestration logic (source: same file).
- **PhaseEvo** (2402.11347), **AMPO** (2410.08696) — evolutionary / population family
  (source: same file).
- **Tang et al., "LLMs as prompt optimizers ≈ gradient optimizers"** (2402.17564) — update
  direction + update method framing (source: same file).
- **Learning from Contrastive Prompts** (2409.15199) (source: same file).
- **Dual-Phase accelerated prompt optimization** (2406.13443) (source: same file).
- **QPO** (2408.10504) — query-dependent learned generator, offline RL (source: same file).
- **Davari et al.** (2507.09839) — reinforcement + diversification + migration hybrid for
  black-box LLMs (source: same file).
- **Wan et al., "Teach better or show smarter"** (2406.15708) — joint instruction × exemplar
  optimization as a settled finding (source: same file).
- **RePrompt** (2406.11132) — agent planning instructions learned from interaction histories
  (source: same file).
- **Prochemy** (2503.11085) — execution-driven prompt refinement for code; a nearest
  published relative on the code side (source: docs/topics/reference/prompt-optimization-
  landscape.md; nl-bottleneck-prior-art.md ICBINB grounding).
- **MCTS-OPS** (2508.05995) and **RL4QE** (no ID) — LLM-as-optimizer citations from the
  ICBINB grounding pass (source: docs/topics/reference/nl-bottleneck-prior-art.md).
- **SelfEvolve** (2306.02907) — reconstruction-stage citation from the ICBINB grounding pass
  (source: same file).
- **PromSec** (no ID) — security + functionality prompt loop; **Midolo et al.** empirical
  prompt guidelines ("2024" but carrying a 2601 arXiv number — slip flagged); **Shashikala
  et al.** accuracy + diversity objectives (source: docs/topics/reference/
  prompt-optimization-landscape.md; nl-bottleneck-prior-art.md).
- **LEO** (2403.02054) — LLM-based evolutionary optimizer with elitism; population-based
  numerical optimizer with hallucination guardrails (source: docs/topics/reference/
  prompt-optimization-landscape.md, 2026-08-23 entry).
- **APO systematic survey** (2502.16923) and **automatic prompt engineering survey**
  (2502.11560) — scaffolding for a fair comparison protocol (source: same file).
- **PromptBridge** (2512.01420) — cross-model prompt transfer (source: same file).
- **Cost-aware evolutionary prompt optimization** (PMLR v293 zehle25a; no arXiv ID) — the
  adjacent work for the curriculum-in-prompt-optimization novelty check (source:
  text-latent-code-autoencoder.md §4 chunk 3; docs/litreview/tlc-litreview-plan.md addendum).
- **FunSearch** (Romera-Paredes et al., Nature 2023, DOI 10.1038/s41586-…; no arXiv ID) —
  LLM as generator/mutator in a verified search loop, no bottleneck; 4.5 band in the
  aggregate table (source: docs/topics/reference/nl-bottleneck-prior-art.md, Consensus and
  Claude entries).
- **LLaMEA** (2405.20132; Stein & Bäck) — LLM evolutionary algorithm for metaheuristics
  (source: docs/topics/reference/nl-bottleneck-prior-art.md, Consensus entry).
- **UCB1** (Auer, Cesa-Bianchi & Fischer 2002) and **Thompson sampling** (Thompson 1933) —
  the bandit baselines in the submitted LLA paper (encoder-prompt selection as a
  multi-armed bandit over a structured template; 27 arms → 8,064-arm expanded grid); LLA
  Reviewer 1 asked for links to best-arm identification (source: text-latent-code-
  autoencoder.md §4 submission record and writing-sprint chunk 6).
- **Self-Debug** (2304.05128; Chen et al.) and **Reflexion** (Shinn et al., NeurIPS 2023; no
  ID) — single-model NL-reasoning loops with no separated encoder/decoder and no external
  search over the representation; ChatGPT-check secondary strands (source:
  docs/topics/reference/nl-bottleneck-prior-art.md).
- **Yuan et al. 2025** (Sci. Rep. 15:37300; no ID) — RL-trained query rephraser for a fixed
  code generator (source: same file).
- **CodePlan** (2309.12499; Bairi et al., FSE 2024) — fine-tuned pseudocode planning; blurs
  encoder/decoder roles (source: same file).
- **Tree of Thoughts** (2305.10601), **Self-Planning** (2303.06689), **SCoT / semantic
  chain-of-thought** (2310.10698), **Structured CoT** — search/planning-over-text
  neighbors from the Claude and Consensus checks and the ICBINB grounding pass (source:
  docs/topics/reference/nl-bottleneck-prior-art.md).

**Row C — compression, rate–distortion, and the lossless-baseline suite**

- **Tishby, Pereira & Bialek, information bottleneck** (physics/0004057) — min I(X;Z) −
  β I(Z;Y); the objective statement TLC adopts with Y_T the test-suite signature (source:
  text-latent-code-autoencoder.md §4 2026-08-22 IB entry).
- **Deep VIB** (1612.00410; Alemi et al.) — the "you optimize variational bounds, not MI"
  caution (source: same entry; nl-bottleneck-prior-art.md measurement entry).
- **Girish et al., "Fundamental Limits of Prompt Compression: A Rate-Distortion Framework
  for Black-Box LLMs"** (2407.15504) — distortion–rate as a linear program, large gap between
  current methods and the optimum, query-aware variable-rate adaptation; the formal axis for
  the length-vs-reconstruction curve; litreview gate 5 (source: docs/topics/reference/
  code-compression-literature.md; prompt-compression-and-optimization-literature.md).
- **Delétang et al., "Language Modeling Is Compression"** (2309.10668) — LM + arithmetic
  coding; the principle behind the strong LLM-arithmetic-coding baseline (source:
  docs/topics/reference/code-compression-literature.md; text-latent-code-autoencoder.md §4).
- **LLMZip** (no ID) — named alongside Delétang as the LLM-as-compressor baseline (source:
  text-latent-code-autoencoder.md §4 2026-07-11 two-interest-categories entry).
- **KoLMogorov Test** (2503.13992) — shortest program that outputs a sequence; compression as
  code generation (source: docs/topics/reference/code-compression-literature.md).
- **Katajainen, Penttonen & Teuhola 1986, "Syntax-directed Compression of Program Files"**
  (no ID) — parse tree + symbol table, 50–60% gain (source: same file, Pro-mode search entry).
- **Evans, "Compression via Guided Parsing"** (no ID) — parser-action stream under CFG
  predictions, functionally equivalent reconstruction (source: same file).
- **JSZap** (Burtscher, Livshits, Sinha, Zorn; MSR; no ID) — JavaScript AST as three streams
  (productions, identifiers, literals), ~10% smaller than gzip; the nearest modern
  source-language precedent and what "AST compact" should become at Layer 4 of the baseline
  suite (source: same file; text-latent-code-autoencoder.md §4 classical-precedents entry).
- **Stork, Haldar & Franz, "Generic Adaptive Syntax-Directed Compression for Mobile Code"**
  (no ID) — grammar-parameterized AST compression with PPM-style modeling + arithmetic
  coding, 5–50% smaller than the best Java scheme (source: same file).
- **Franz & Kistler, Slim Binaries / adaptive syntax-tree compression** (no ID) — >2× denser
  than Java bytecode (source: same file).
- **Ernst, Evans, Fraser, Lucco & Proebsting 1997, "Code Compression"** (PLDI; no ID) — wire
  representation ~21% of SPARC code for gcc (source: same file).
- **Evans & Fraser, "Bytecode Compression via Profiled Grammar Rewriting"** (no ID) — lcc
  bytecode 199 KB → 58 KB (source: same file).
- **Pugh, "Compressing Java Class Files"** / **Pack200** (no ID) — 17–41% of gzipped class
  files; drops debug attributes (source: same file).
- **Boffa et al. 2025, "On the compressibility of large-scale source code datasets"** (JSS;
  no ID) — Software Heritage C/C++/Java/JS/**Python**, 78 TiB → ~3 TiB; the citation for
  "exact source compression is not gzip" (source: same file).
- **C-Reduce** (Regehr et al., PLDI 2012), **Hierarchical Delta Debugging** (Misherghi & Su),
  **Picireny** (Python HDD over ANTLR), **Perses** (ICSE 2018), **C-Vise**, **J-Reduce** —
  program reducers, "lossy program minimizers whose correctness is an oracle"; the citation
  family for the test-relative guarantee (source: same file; text-latent-code-
  autoencoder.md §4 correctness-ladder entry).
- **Massalin 1987 superoptimizer** (no ID), **STOKE** (1211.0557), **Souper** (1711.04422),
  **egg / equality saturation** (no ID; recommended three times in the Feb-2026
  conversation) — semantics-preserving smaller code at IR/assembly level; e-graphs named as
  the formalism between rewriting and search (source: docs/topics/reference/
  code-compression-literature.md; text-latent-code-autoencoder.md §4 chunks 10–12;
  docs/litreview/tlc-litreview-plan.md addendum).
- **DreamCoder** (2006.08381), **Stitch** (Bowers et al.; compressivity metric, 806 → 604),
  **BABBLE** (2212.04596; e-graphs + anti-unification), **LILO** (2310.19791), **Leroy**
  (2410.06438; imperative/Python subset, ~1.04×) — library learning / abstraction invention;
  Leroy is the named contrast case for "compressing functions" (source:
  docs/topics/reference/code-compression-literature.md).
- **CuBERT** (Kanade et al. 2020; no ID), **Saletta et al. 2021 Java autoencoders**, **Zhang
  et al. EMSE 2025 hierarchy-compressed transformers**, **Rabin et al. 2020**, **Ding et al.
  2022** — the learned-embeddings sense of "code compression" (source: same file).
- **Gist tokens** (2304.08467; Mu et al.), **500xCompressor** (2408.03094), **query-guided
  compressor** (Cao et al., 2406.02376), **CodePromptZip** (2502.14925; He et al.),
  **LongCodeZip** (Shi et al.; no ID), **docstring compression** (Yang et al., 2410.22793),
  **Ostby "Stingy Context"** (18:1 hierarchical compression for auto-coding), **Johnson
  "Perplexity Paradox" / TAAC** — prompt/context compression for code LLMs (source: same
  file).
- **Compressor** (Shi et al., ASE 2022, "3 MB", 160×), **LORD** (2309.14021), **structural
  pruning** (2412.15921) — model compression of code LLMs (source: same file).
- **Tsai, "Revisiting data compression with language modeling"** (no ID) — LLMs as entropy
  coders (source: same file).
- **Cummins et al., "Don't transform the code, code the transforms"** (2410.08806) — precise
  rewriting via LLMs (source: same file).
- **Ong et al., layered contextual pruning** (no ID) — semantic-compression family (source:
  same file).
- **Hinton & Zemel MDL/autoencoders**, **VQ regularization** (no IDs) — theory pointers in
  the SciSpace map (source: same file).
- **ShortenDoc** (no ID) — docstring compression on HumanEval/EvoEval; ~30% compression often
  keeps or improves pass@1; recorded as the closest existing work to the compression
  project's NL-side question and added to gate 1 (source: docs/topics/reference/
  code-benchmarks-landscape.md; docs/litreview/tlc-litreview-plan.md gate 1).
- **Maveli, Vergari & Cohen, "Can LLMs Compress (and Decompress)?"** (no ID) — LLMs forward-
  and reverse-predict four lossless compressors as a code-understanding probe; **closed false
  alarm, Danielle read it, do not reopen** (source: docs/topics/reference/
  code-compression-literature.md; docs/litreview/tlc-litreview-plan.md gate 6).
- **The lossless-baseline tool suite** (tools, not papers; all unverified): raw bytes;
  zlib/DEFLATE 1/6/9; **Zopfli**; **zstd** 3/9/19/22 (RFC 8878) incl. trained dictionaries
  via COVER/fastCover, sweep 256 B–128 KB; **Brotli** q9/q11 (RFC 7932) + shared dictionary
  (RFC 9841); raw **LZMA2**/xz 9e; zlib `deflateSetDictionary`; bzip2; **libbsc**; **PPMd**
  (7-Zip / `pyppmd` Variant H, order 2–64); **paq8px**, **ZPAQ**, **cmix**; **NNCP**;
  Python-aware transforms — `python-minifier`, `tokenize` token codecs, alpha-renaming,
  compact preorder AST stream, grammar-based arithmetic coder over AST symbols, CPython
  bytecode (version-pinned), nearest-reference byte/token/AST diff (`bsdiff4`, `xdelta3`);
  **LibCST** as the exact-source (lossless CST) serialization; `pyminifier`, `compileall`,
  `zipapp`, Python 3.14 stdlib `compression.zstd` (source: docs/topics/reference/
  code-compression-literature.md, baseline-suite entry; text-latent-code-autoencoder.md §4).
- **General compression taxonomy references**: **ANS** (1311.2540; Duda), DEFLATE (RFC 1951),
  LZ4/Snappy, LZMA, PNG filters, BWT, PPM, PAQ/cmix, VCDIFF (RFC 3284), rsync, Git
  packfiles, content-defined chunking, JPEG/AAC/AV1/HEVC/VVC, DPCM, LPC/CELP, JPEG AI,
  **SoundStream** (2107.03312), EnCodec, zfp, SZ — the ten-assumption taxonomy whose last
  entry ("learned prior") is what TLC's decoder supplies (source: docs/topics/reference/
  code-compression-literature.md, turn-1 taxonomy).
- **The five-level correctness ladder** (byte-exact → source-equivalent → runtime-equivalent
  → test-equivalent → intent-equivalent), with Python reflection (`inspect`, `getattr`,
  `-O`/`-OO`) as why runtime-equivalent is not safely reachable; TLC sits at test-equivalent
  and every baseline should be labeled by level, with separate leaderboards per level
  (source: docs/topics/reference/code-compression-literature.md, turn-2 entry;
  text-latent-code-autoencoder.md §1 and §4).

**Row D/F — measurement of the bottleneck: usable information, probing, contrastive code
semantics, representation similarity**

- **𝒱-information / "A Theory of Usable Information Under Computational Constraints"**
  (2002.10689; Xu et al.) — the formalism TLC-0 is built on: I_𝒱(Z→B) relative to a declared
  extractor family (source: text-latent-code-autoencoder.md §1, §3, §4 extractable-
  information entry; nl-bottleneck-prior-art.md measurement entry).
- **Decodable Information Bottleneck** (Dubois et al., NeurIPS 2020; no arXiv ID on record) —
  IB relative to a predictive family; "the theory behind 'the decoder already knows
  programming'"; named as one of the two citations that state the project's analysis stance
  (source: docs/topics/reference/nl-bottleneck-prior-art.md, measurement entry).
- **Saxe et al. (ICLR 2018)** (no ID) — against the IB-theory-of-deep-learning claims; the
  caution against claiming compression ⇒ generalization (source: same file).
- **CPC / InfoNCE** (1807.03748; van den Oord et al.) — contrastive lower-bound-like proxy
  for I(R;Y_T) (source: text-latent-code-autoencoder.md §4 IB entry).
- **Conditional probing** (Hewitt et al., EMNLP 2021; no ID) — information beyond a baseline
  C = signature / imports / type hints / "Python knowledge" (source: nl-bottleneck-prior-
  art.md measurement entry).
- **Control tasks** (Hewitt & Liang; no ID) — shuffled behaviour labels, random
  implementation identities; in the probe suite from day one (source: same file).
- **MDL probing** (Voita & Titov, EMNLP 2020, ACL 2020.emnlp-main.14) — report online
  codelength, not accuracy (source: same file; text-latent-code-autoencoder.md §4).
- **Probing-as-MI** (Pimentel et al., ACL 2020; no ID) (source: nl-bottleneck-prior-art.md).
- **Rationale extraction** (Lei, Barzilay & Jaakkola, 1606.04155) — the text-input,
  label-output cousin of the bottleneck (source: same file).
- **ContraCode** (2007.04973; Jain et al.) — contrastive invariance to semantics-preserving
  transforms; "the published form of §1's invariance objective"; directly attacks the
  minification/formatting problem (source: same file; text-latent-code-autoencoder.md §4).
- **Corder** (2009.02731), **CoCoSoDa** (2204.03293), **CodeRetriever** (no ID) — contrastive
  code / NL–code search (source: nl-bottleneck-prior-art.md measurement entry).
- **CodeBERT** (2002.08155), **GraphCodeBERT** (2009.08366; data flow as a middle ground),
  **CodeT5** (2109.00859), **UniXcoder** (no ID), **CodeSearchNet** (1909.09436) — NL–code
  alignment models; instruments, not behaviour scores (source: same file).
- **Troshin & Chirkova** (2202.08975) — probing pretrained source-code models: syntax,
  identifiers, namespaces yes; semantic equivalence poorly (source: same file).
- **Naik et al.** (2207.07706) — RSA on CodeBERT/CodeNet: form-based patterns unless
  fine-tuned on semantic tasks (source: same file).
- **SVCCA** (1706.05806), **CKA** (no ID), **RSA** — representation-similarity tools for the
  optional "does Z collapse equivalents and separate near-misses" figure (source: same file).
- **LEEP** (2002.12462), **LogME** (no ID), **Task2Vec** (no ID) — transferability /
  probe-ease scores (source: same file).
- **Dynamic neural program embeddings from execution traces** (1711.07163) and
  **FuzzPretrain** (no ID) — execution-trace representations (source: same file).
- **CodeBLEU** (2009.10297) and **CodeBERTScore** (no ID) — side metrics only (source: same
  file).
- **LLM-as-a-judge survey** (2411.15594) and **CodeJudge** (no ID) — members of 𝒱, never
  ground truth (source: same file).
- **Intermediate-language study** (2407.05411) — NL is often the most effective intermediate;
  intermediate correctness only weakly correlates with final generation (source: same file).
- **Turpin et al., CoT faithfulness** (2305.04388) — evaluate what can be extracted from Z,
  not whether Z reads as correct (source: same file).
- **Codex / HumanEval pass@k unbiased estimator** (2107.03374) — the decoder-pass probe's
  estimator; "use 1 − C(n−c,k)/C(n,k), not the plug-in" (source: docs/topics/reference/
  estimation-and-calibration-methods.md; text-latent-code-autoencoder.md §4).
- **Bisimulation metrics** (no IDs; RL/POMDP line) — reward difference + discounted
  Wasserstein over next-state class distributions; a definition upgrade for behavioral
  equivalence, with the recorded caveat that in single-shot settings it degenerates to
  observational/contextual equivalence — skim-level, do not adopt as paper framing without
  PL review (source: text-latent-code-autoencoder.md §4 chunk 14; litreview plan addendum).
- **Conformal prediction** (Lei et al., 1604.04173) and **conformal risk control**
  (Angelopoulos et al., 2208.02814) — calibrating small-n pass-rate estimates and
  accept/route thresholds; flagged as cross-project tools (source: docs/topics/reference/
  estimation-and-calibration-methods.md; potential-projs/README.md).
- **Signal and Noise** (Heineman et al., NeurIPS 2025; no ID) — signal vs. noise in LM
  evaluation; continuous metrics beat accuracy; noisy-subtask filtering (source:
  docs/topics/reference/evaluation-methodology-literature.md).
- **OLMES** ("A Standard for Language Model Evaluations"; no ID) — evaluation-condition
  standardization precedent (source: same file).
- **Melis, Dyer & Blunsom 2018** — conclusions invert under equalized tuning budgets; the
  "headline phenomenon is a tuning artifact" ancestor (source: same file).

**Row D — evaluation protocol, equivalence, and reconstruct-from-description**

- **OctoPack / HumanEvalPack / HumanEvalExplain** (2308.07124; Muennighoff et al. 2023) —
  explain a function in NL, regenerate from the explanation alone, score pass@1; 164 problems
  × 6 languages, standardized harness; *is* TLC's reconstruction loop without length
  pressure; the encoder baseline prompt comes from here; the benchmark precedent for
  reconstruct-from-text evaluation (source: docs/topics/reference/humanevalexplain-results.md;
  text-latent-code-autoencoder.md §1, §4; litreview plan row D, gate 4).
- **RTC — Round-Trip Correctness** (2402.08699; Allamanis et al., ICML 2024) — code → NL
  summary → code, checked for semantic equivalence, as *unsupervised evaluation*, explicitly
  treating the NL description as a compression of behavior; TLC-0-relevant; the
  eval-not-learning contrast case (source: docs/topics/reference/nl-bottleneck-prior-art.md,
  ChatGPT entry; litreview plan row A).
- **WaveCoder** (2312.14187; Yu et al.) and **Szalontai et al., "Large Language Models for
  Code Summarization"** (2405.19032) — the only two other papers reporting HumanEvalExplain
  numbers; the published pass@1 table (GPT-4 52.1 avg / 64.6 Python; best open 6.7B 46–51;
  base models 0) is the no-length-pressure endpoint; **transcription-shift error flagged** in
  the WaveCoder-attributed GPT-4 and WizardCoder rows (source: docs/topics/reference/
  humanevalexplain-results.md).
- **HumanEval** (2107.03374) and **MBPP** (2108.07732) — the sandbox datasets; HumanEval's
  `check()` split into independent per-test outcomes; MBPP's prompt leaks test cases by
  design, which matters for leakage accounting (source: text-latent-code-autoencoder.md §1,
  §3, §4; nl-bottleneck-prior-art.md measurement entry; code-benchmarks-landscape.md).
- **HumanEval+ / MBPP+ / EvalPlus** (2305.01210) — ~80× tests; the current project dataset
  (HumanEvalPlus), with Danielle's first-hand defects on record: dramatic difficulty variance,
  some items solved by every model, some broken in the official release, items "SUPER short"
  so description-length compression is nearly meaningless (source: docs/topics/reference/
  code-benchmarks-landscape.md, 2026-08-23 note; text-latent-code-autoencoder.md §4 dataset-
  strategy entry).
- **CodeNet** (2105.12655) — millions of accepted solutions with I/O tests; the source of
  many-implementation / same-problem clusters, i.e. the behaviour-equivalent positives the
  contrastive game needs (source: docs/topics/reference/nl-bottleneck-prior-art.md;
  text-latent-code-autoencoder.md §3).
- **CETBench** (2506.04019; Oza et al.) — code-equivalence checking benchmark via program
  transformations; relevant to property-indexed equivalence and behavior-preserving
  normalization; Consensus-check standout (source: docs/topics/reference/
  nl-bottleneck-prior-art.md).
- **LiveCodeBench** (2403.07974) — dated, contamination-resistant problems; also its execution
  and output-prediction tracks; **CRUXEval** (no ID) — the "code reasoning / execution
  prediction" family named as the nearest neighbour to reconstruct-from-description
  evaluation (source: docs/topics/reference/code-benchmarks-landscape.md;
  text-latent-code-autoencoder.md §4 task-set entry).
- **ReCode** (2212.10264) — 30+ semantics-preserving transformations over docstrings, names,
  syntax, format on HumanEval+MBPP; the perturbation-robustness baseline TLC-0 control tasks
  should be compared against (source: docs/topics/reference/code-benchmarks-landscape.md).
- **NLPerturbator / HumanEval-R** (2406.19783) — NL perturbations; same role (source: same
  file).
- **CL-HumanEval** (no ID) — strips function names, variable names, execution examples; a
  ready-made "signature without hints" condition for the TLC-0 condition matrix (source:
  same file).
- **EvoEval** (2403.19114) — LLM-evolved into 7 benchmarks / 828 problems (difficult,
  creative, subtle, combine, tool-use, verbose, concise); harder variants if HumanEval
  saturates (source: same file).
- **HumanEval Pro / MBPP Pro** (2412.21199), **HumanEval_T / DyCodeEval / HumanEvalNext**
  (all cited together as 2412.01526 — at most one is right), **InstructHumanEval**,
  **HumanEval Infilling**, **HumanEvalComm** (2406.00215) — HumanEval's derivative ecosystem
  (source: same file).
- **HumanEval-X** (2303.17568), **MultiPL-E** (2208.08227), **HumanEval-XL** (2402.16694),
  **MBXP / Multilingual HumanEval**, **mHumanEval**, **HumanEval.jl**, **HumanEval-V**
  (2410.12381), **Qiskit HumanEval** (2406.14712) — multilingual and new-modality variants
  (source: same file).
- **DS-1000** (2211.11501) — execution-tested data-science code generation with surface-form
  constraints; a required read before building the data-manipulation task family; the
  precedent a generator must visibly improve on (source: docs/topics/reference/
  code-benchmarks-landscape.md; litreview plan addendum).
- **BigO(Bench)** (Facebook Research; no ID) — ~1.19M complexity-labeled solutions plus
  dynamic complexity-inference tooling; required before any complexity-property oracle;
  friendly last author; the tool-assisted route to graded property oracles P = {tests} →
  {tests, complexity} → {tests, complexity, purity/effects} (source: docs/topics/reference/
  code-benchmarks-landscape.md; text-latent-code-autoencoder.md §1; litreview plan addendum).
- **ENAMEL** (eff@k), **EvalPerf / DPE**, **EffiBench** (no IDs) — efficiency-beyond-
  correctness benchmarks (source: docs/topics/reference/code-benchmarks-landscape.md;
  evaluation-methodology-literature.md).
- **AlphaCodium** (2401.08500; slip: also given as 2401.19489) — the wrapper-only code
  precedent; direct-generation citation from the ICBINB grounding pass (source:
  docs/topics/reference/nl-bottleneck-prior-art.md; text-latent-code-autoencoder.md §4).
- **AceCoder** (DOI 10.1145/3675395) — direct-generation citation from the ICBINB grounding
  pass (source: docs/topics/reference/nl-bottleneck-prior-art.md).
- **Misu et al. 2024, Dafny prompt styles** (DOI 10.1145/3643763) — three prompt styles for
  reconstruction; ICBINB grounding (source: same file).
- **CodeT** (2207.10397; Chen et al.) — code generation with generated tests (source: same
  file, Claude entry).
- **TransCoder** (2006.03511) and **"Leveraging Automated Unit Tests for Unsupervised Code
  Translation"** (2110.06773) — required before the cross-language equivalence direction; the
  latter's generate-then-test-filter loop matches the library's spec+verify shape (source:
  text-latent-code-autoencoder.md §4 chunk 3; litreview plan addendum).
- **SymC** (PMLR v235 pei24b; no ID) and **ProgramTransformer** (ScienceDirect; no ID) —
  semantics-preserving-transformation prior art for the canonicalization/invariance thread
  (source: text-latent-code-autoencoder.md §4 chunk 9; litreview plan addendum).
- **CompilerGym** (no ID) — compiler pass sequencing as sequential decision-making, "rewrites
  as an action space"; **LLM + compiler feedback loops** (2403.14714, title not stated in
  source) (source: text-latent-code-autoencoder.md §4 chunk 9).
- **How2Bench** (2501.10711) — audit of 274 code benchmarks; 62% did not deduplicate,
  81.8% of 2023–24 benchmarks did not address contamination (source:
  docs/topics/reference/code-benchmarks-landscape.md).
- **LessLeak-Bench** (2502.06215), **CodeSearchNet inter-dataset duplication** (2401.07930),
  **HumanEval/MBPP contamination in the Pile / The Stack** (2403.04811), **ContextBench**
  (2602.05892), **StarCoder2 / The Stack v2** (2402.19173), **DéjàVu** (DOI
  10.1145/3133908), **Allamanis on duplicated corpora** — the contamination/dedup literature;
  its three-matrix recommendation (prompt↔prompt, code↔code, prompt↔code) is the same
  bookkeeping as TLC-0's leakage accounting (source: same file).
- **Summary-Mediated Repair** (2511.18782), **CodeMind**, **XCoder**, **Selective Shot
  Learning** (2412.12852), **LeDex** (2405.18649), **SelfCodeAlign** (2410.24198), **Crystal**
  (2411.04156), **InstructCoder** (2310.20329) — HumanEvalPack users that report Fix/Synthesize
  only, i.e. the near-miss set for the HumanEvalExplain search (source:
  docs/topics/reference/humanevalexplain-results.md).
- **xCodeEval, CodeXGLUE, FullStack Bench, DOMAINEVAL, VerilogEval/RTLLM, HumanEvo,
  MRG-Bench, DependEval, REPOEXEC, SWE-MERA, SWE-PolyBench, AutoCodeBench, LONGCODEU,
  BEHELM, SWE-bench (Lite/Verified/Multilingual/Multimodal), BigCodeBench, NaturalCodeBench,
  CoderEval, APPS, CodeContests** — the broader benchmark landscape as surveyed (source:
  docs/topics/reference/code-benchmarks-landscape.md).
- **Prompt-condition facts** (harness contract, not papers): HumanEval original is a raw stub
  completion prompt; OpenAI `simple-evals` prepends an instruction; lm-evaluation-harness
  separates `humaneval` from `humaneval_instruct`; BigCode has InstructHumanEval; MBPP's
  original is already few-shot instruction + visible asserts (`[BEGIN]`/`[DONE]` delimiters).
  Pass rates are not comparable across the raw/instruct choice — record the condition per run
  and pin the harness commit (source: docs/topics/reference/code-benchmarks-landscape.md;
  text-latent-code-autoencoder.md §4 prompt-condition entry).

**Row E and adjacent — structured output, elicitation, and format-vs-content**

- **SLOT** (EMNLP Industry 2025; no ID) — a lightweight fine-tuned post-processing
  *structurer* for other models' outputs; the strongest published version of the
  "formatting is a separate skill" intuition; relevant to TLC's frozen deterministic
  extraction step and its refusal/extraction failure taxonomy (source:
  docs/topics/reference/structured-output-literature.md).
- **"The Hidden Cost of Structure"** (RANLP 2025; no ID) — base models often benefit from
  constraints, instruction-tuned models degrade on generation (source: same file).
- **"Quantifying the Impact of Structured Output Format on LLMs' Reasoning Performance"**
  (EACL Findings 2026; no ID) — effect is positive/negative/neutral per (model, task, schema,
  prompt) (source: same file).
- **Structured Output Benchmark** (2604.25359), **ExtractBench** (2602.12247),
  **JSONSchemaBench** (2501.10868), **"When Correct Isn't Usable"** (2605.02363),
  **LLMStructBench** (2602.14743), **clinical SLM extraction** (2507.01810) — the "valid
  format ≠ correct content" evidence base, i.e. the format-vs-content split TLC's failure
  taxonomy makes ({refused, extraction/format failure, does not run, tests fail, pass});
  several are 2602–2605 IDs and the title–ID pairing is the thing to verify (source: same
  file).
- **GLiNER2** (2507.18546), **NuExtract / NuExtract 2.0**, **ScrapeGraphAI-100k**
  (2602.15189), **VAREX** (2603.15118), **Schema Reinforcement Learning** (2502.18878),
  **RL-Struct** (2512.00319), **schema-key-wording as instruction channel** (2604.14862),
  **PA-Tool** (2510.07248); constrained-decoding tooling Outlines / Guidance / XGrammar /
  llama.cpp grammars (source: same file).
- **Elicitation-gain (ELI) shared row E** — AlphaCodium; prompt tuning at small scale; ELI's
  own gate list; the same 𝒱-probe harness serves both projects with the roles swapped (source:
  docs/litreview/tlc-litreview-plan.md row E; potential-projs/elicitation-gain.md;
  text-latent-code-autoencoder.md §4 extractable-information entry).
- **Code feature extraction tooling** (`tokenize`, `ast`, `symtable`, `pyclbr`, and the
  eleven-family landscape survey) — the fixed-schema, no-training Python feature extractor
  for latent-length and structure measures (source: docs/topics/reference/
  code-feature-extraction-tooling.md; litreview plan row D).

**Project-internal artifacts, records, and data on file (not external literature)**

- **The submitted LLA workshop paper** — "Prompt Optimization for Behavioral Code
  Compression: Bandits vs LLM-in-the-Loop Search" (ICLR 2026 Workshop on Lifelong Agents;
  Rothermel, Li, Cho): frozen black-box encoder–decoder; encoder-prompt selection as a MAB
  over a 3-slot × 3-phrasing template (27 arms → 8,064-arm grid); UCB1 vs. Thompson vs.
  LLM-in-the-loop; headline metric the compression–correctness frontier S_m(c;N) summarized
  by frontier AUC; OOD on HumanEval++, a bit-operations family, Java/Rust, and
  Gemini-Flash-Lite decoder transfer. Reviews: R1=7, R2=5, R3=5 (**R3's objection was that
  GEPA and the prompt-optimization SOTA were undiscussed**); AC recommended Accept (Poster),
  Program Chairs rejected (source: text-latent-code-autoencoder.md §4 submission record).
- **The ICBINB/ICLR-2026 draft PDF** ("ICBINB: Code Synthesis and Reconstruction") — Eq. 4–5,
  7; the COMP-NL vs. COMP-SHORT latent-format contrast (a two-point sample of the
  NL-likeness ladder); "portable semantic contracts" (source:
  docs/topics/reference/nl-bottleneck-prior-art.md; text-latent-code-autoencoder.md §4).
- **The Cho pitch** — "LLM-as-Optimizer of Natural Language Bottleneck Model" (Recursive
  Self-Improvement Workshop @ ICLR 2026), with the harness formalism θ = (θ_E, θ_D), the
  Feas/Succ waterfall, J(θ), π_opt, and the Table 5 OpenRouter pricing menu (source:
  text-latent-code-autoencoder.md §4 Cho-pitch entry).
- **The synthetic function-generation library** (String Rules / Stateful / Simple Algorithms
  families at difficulty 3–5) and its four recorded defect classes — sampled-setting
  collisions, maintenance burden, identity-degenerate samples, uneven design-time difficulty;
  superseded by the HumanEvalPlus cutover (source: text-latent-code-autoencoder.md §4
  dataset-strategy entry).
- **The archived, never-run lossless-baseline harness skeleton** at
  `~/drotherm/data/convo-artifacts/2026/2026-08-22-lossless-baseline-harness/` — the
  `code_to_test` vs. scored `payload` split, per-representation dictionaries on a disjoint
  split, selector-byte accounting, `side_info` naming every uncounted prior; API details
  unverified (source: docs/topics/reference/code-compression-literature.md;
  text-latent-code-autoencoder.md §4 baseline-suite entry).
- **The five 2026-08-22 SciSpace bundles** (`scispace-nl-latents-rw-…` 100 files incl. the
  ICBINB draft; `scispace-code-compression-…` 173 files; `scispace-prompt-optimization-…`
  with a 648-row deep-search table; `scispace-prompt-compression-method-papers-…` with
  ~2,000 rows of search archives; `scispace-humanevalexplain-results-…`;
  `scispace-coding-datasets-and-benchmarks-…`) — each with an `INDEX.md`; the seeded
  candidate lists for the full prior-art pass (source: docs/litreview/tlc-litreview-plan.md;
  the four reference accumulators).
- **The 2026-08-24 Notion lit-review bundle** — six Feb-2026 platform checks, the Dec-2025
  SciSpace verdict's Notion copy plus its Prior Work Supplemental, seven NBLM summary tables,
  and the Aggregate Summary (source: docs/topics/reference/nl-bottleneck-prior-art.md).
- **Model-behavioral-divergence** (`docs/topics/staging/model-behavioral-divergence.md`) — the
  spun-out sibling using the TLC harness as testbed; its measurement menu (per-item success
  correlation, failure-mode distance via JS/Wasserstein, output diversity via AST/CFG
  fingerprints, prompt transfer, the time axis) overlaps TLC-0's representation-quality axis
  (source: docs/topics/reference/evaluation-methodology-literature.md; the staging doc).
- **Danielle's decision-quality / constraint-consistency thread** — choose-among-sketches-
  then-implement, its efficiency prior art (ENAMEL/EvalPerf/EffiBench/DS-1000/BigO(Bench)),
  and the four named measurement axes (variance, consistency, divergence, decision quality)
  (source: docs/topics/reference/evaluation-methodology-literature.md, chunks 5–9).
- **The property-indexed-equivalence vocabulary** — f ~_P g iff ∀p∈P, p(f)=p(g); abstract
  interpretation and cost semantics as the named frameworks; "canonicalization" explicitly
  rejected (it requires a unique, deterministic, total, idempotent representative) in favor
  of "abstracting programs into equivalence classes" and "behavior-preserving
  normalization"; the metric vocabulary class consistency / intra-class variance /
  representative quality / abstraction error (source: text-latent-code-autoencoder.md §1 and
  §4 chunks 11–12).
- **TLC-0 candidate metrics** — IR-distance (bytecode n-grams, CFG fingerprints, def-use
  chains, normalized opcode sequences, runtime traces); Q(d) = pass_rate − λ·variance −
  µ·cost; S_multi(d) (mean pass probability across M decoders); per-test outcome-vector
  similarity (cosine/Jaccard/KL) as the soft equivalence label; deterministic-canonicalizer
  conditions as the style-only information floor (source: text-latent-code-autoencoder.md §1
  and §4 chunks 10–14).
- **The E0–E5 closing experiment suite** — E1 distribution-compatibility ladder (raw / Black /
  +alpha-renaming / +desugaring / ugly-consistent; predicted inverted-U); E2 latent-format
  comparison at matched budget (freeform NL vs. structured-NL JSON vs. invented pseudo-DSL —
  the NL-necessity test); E3 cross-decoder transfer matrix; E4 equivalence-class variance
  compression (source: text-latent-code-autoencoder.md §4 chunk 14).
- **Gate items still open** (`docs/litreview/tlc-litreview-plan.md`): read GenDLN; locate
  ShortenDoc; fix the identifier slips (EPiC 2408.11198 not 2410.14321; AlphaCodium 2401.08500
  not 2401.19489; Midolo's 2601 number; Nano-Capsulator first author); supply APE / OPRO /
  ProTeGi / TextGrad / DSPy-MIPROv2 / GEPA anchors; decide the HumanEvalExplain benchmark-
  precedent sentence; decide whether to report a distortion–rate curve or a fixed-budget
  point; verify the curriculum-in-prompt-optimization novelty claim; resolve the
  Gilbert 2304.12512 vs. 2406.01989 ID conflict and the 2507.08627 authorship discrepancy.

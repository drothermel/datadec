# Citation verification ledger — SciSpace intake batch of 2026-08-22

Status: ledger only (decision 2026-08-22: consolidate now, verify when a gate runs).

Every arXiv identifier that entered the reference topics during the 2026-08-22 SciSpace
intake, with its origin. **Nothing here is verified.** Two origins:

- *agent-supplied* — the identifier appeared in a SciSpace report or bundle and was
  carried into the topic as reported (the same sessions produced swapped bibliography
  entries and a fabricated author list, so these are suspect).
- *Claude-added* — the identifier was supplied from the intake agent's memory in an
  "Intake notes" / missing-canon paragraph, not from any source in the bundle. These
  are hallucination-prone in the last digits and in title–ID pairing.

The verification pass (parked) resolves each row against the arXiv record: confirm the
ID exists, the title matches the claimed paper, and the year/author are as stated;
write `verified: yes/no/wrong-paper` and the correct ID. Run it as one Opus worker per
~40 rows with web access, read-only, writing back only this file. Rows feeding a gate
that is about to run go first.

Context column = the ~80 characters preceding the ID in the topic, as a memory aid only.

| ID | Context (as written in the topic) | Topic | Origin | Feeds | Verified |
|---|---|---|---|---|---|
| (no ID) | Maveli, Vergari & Cohen, "Can LLMs Compress (and Decompress)?" — checked by Danielle: unrelated | `code-compression-literature` | agent-supplied | — | |
| 2304.08467 | pt/context* compression for code LLMs — ICAE (Ge et al.), gist tokens (Mu et al. | `code-compression-literature` | agent-supplied | TLC | |
| 2304.12512 | 0.06438, 1.04×); (f) *semantic compression* — "Semantic Compression with LLMs" ( | `code-compression-literature` | agent-supplied | TLC | |
| 2309.14021 | on of code LLMs — Compressor "3 MB" (Shi et al. ASE 2022, 160×), LORD low-rank ( | `code-compression-literature` | agent-supplied | TLC | |
| 2406.02376 | l. 2304.08467), 500xCompressor (2408.03094), query-guided compressor (Cao et al. | `code-compression-literature` | agent-supplied | TLC | |
| 2407.15504 | ory:* Girish et al. rate–distortion framework for black-box prompt compression ( | `code-compression-literature` | agent-supplied | TLC | |
| 2408.03094 | de LLMs — ICAE (Ge et al.), gist tokens (Mu et al. 2304.08467), 500xCompressor ( | `code-compression-literature` | agent-supplied | TLC | |
| 2410.06438 |  *abstraction* — library learning, Leroy for imperative languages (Bellur et al. | `code-compression-literature` | agent-supplied | TLC | |
| 2410.08806 | riting via LLMs (Cummins et al. "Don't transform the code, code the transforms", | `code-compression-literature` | agent-supplied | TLC | |
| 2410.22793 | et al. 2502.14925), LongCodeZip (Shi et al.), docstring compression (Yang et al. | `code-compression-literature` | agent-supplied | TLC | |
| 2412.15921 | B" (Shi et al. ASE 2022, 160×), LORD low-rank (2309.14021), structural pruning ( | `code-compression-literature` | agent-supplied | TLC | |
| 2502.14925 | 3094), query-guided compressor (Cao et al. 2406.02376), CodePromptZip (He et al. | `code-compression-literature` | agent-supplied | TLC | |
| 2306.06625 | le where a heavily resourced from-scratch 2B beat a 10B→2B distilled model (GLMD | `distillation-literature` | agent-supplied | MIC | |
| 2306.13649 | ights an advantage-guided KD against SFT. Missing: on-policy GKD (Agarwal et al. | `distillation-literature` | agent-supplied | MIC | |
| 2402.03898 | d KD against SFT. Missing: on-policy GKD (Agarwal et al. 2306.13649), DistiLLM ( | `distillation-literature` | agent-supplied | MIC | |
| 2402.12030 | rgence control (ToDi 2505.16297), BiLD (2406.13555), cross-tokenizer losses (ULD | `distillation-literature` | agent-supplied | MIC | |
| 2404.19319 | (2402.03898). 3. *Logit vs. token repetition.* The only evidence is Bui et al. ( | `distillation-literature` | agent-supplied | MIC | |
| 2406.13555 | (Kim et al. 2509.25837); token-wise divergence control (ToDi 2505.16297), BiLD ( | `distillation-literature` | agent-supplied | MIC | |
| 2406.17328 |  2402.12030; multi-level OT 2412.14528). Combination: equal-weight CE + KD (DSKD | `distillation-literature` | agent-supplied | MIC | |
| 2407.14679 | noted).** 1. *Sizes and scaling.* Reported ratios span 2:1 (Minitron 15B/30B→8B, | `distillation-literature` | agent-supplied | MIC | |
| 2407.16154 | ally across 120M–13B; domain-aligned mid-size teachers can beat bigger ones (DDK | `distillation-literature` | agent-supplied | MIC | |
| 2410.16215 | , ADPA 2502.17927); Peng et al.'s pre-training-distillation design-space study ( | `distillation-literature` | agent-supplied | MIC | |
| 2410.17215 | stilling. 5. *Pre- vs. post-trained teacher.* Pre-training distillation (MiniPLM | `distillation-literature` | agent-supplied | MIC | |
| 2412.14528 | 6297), BiLD (2406.13555), cross-tokenizer losses (ULD 2402.12030; multi-level OT | `distillation-literature` | agent-supplied | MIC | |
| 2502.08606 | es (DDK 2407.16154). **No distillation scaling law is cited** — Busbridge et al. | `distillation-literature` | agent-supplied | MIC | |
| 2502.17927 | n 15B/30B→8B, 2407.14679) to ~26:1 (7–13B→500M for preference distillation, ADPA | `distillation-literature` | agent-supplied | MIC | |
| 2505.16297 | rete Score Matching (Kim et al. 2509.25837); token-wise divergence control (ToDi | `distillation-literature` | agent-supplied | MIC | |
| 2509.25837 | JSD / skew-KL / α-β / TV variants studied in Concrete Score Matching (Kim et al. | `distillation-literature` | agent-supplied | MIC | |
| 2509.26497 | distillation (ADPA, DCKD; "revealing the power of post-training for SLMs via KD" | `distillation-literature` | agent-supplied | MIC | |
| 2308.07124 | in→regenerate subtask of HumanEvalPack, OctoPack, Muennighoff et al. 2023, arXiv | `humanevalexplain-results` | agent-supplied | TLC | |
| 2310.20329 | 2, LeDex 2405.18649, SelfCodeAlign 2410.24198, Crystal 2411.04156, InstructCoder | `humanevalexplain-results` | agent-supplied | TLC | |
| 2312.14187 | manEvalExplain numbers: OctoPack itself (2308.07124), WaveCoder (Yu et al. 2023, | `humanevalexplain-results` | agent-supplied | TLC | |
| 2405.18649 | d Repair 2511.18782, CodeMind, XCoder, Selective Shot Learning 2412.12852, LeDex | `humanevalexplain-results` | agent-supplied | TLC | |
| 2405.19032 | 187), and "Large Language Models for Code Summarization" (Szalontai et al. 2024, | `humanevalexplain-results` | agent-supplied | TLC | |
| 2410.24198 | ind, XCoder, Selective Shot Learning 2412.12852, LeDex 2405.18649, SelfCodeAlign | `humanevalexplain-results` | agent-supplied | TLC | |
| 2411.04156 | ve Shot Learning 2412.12852, LeDex 2405.18649, SelfCodeAlign 2410.24198, Crystal | `humanevalexplain-results` | agent-supplied | TLC | |
| 2412.12852 | s (Summary-Mediated Repair 2511.18782, CodeMind, XCoder, Selective Shot Learning | `humanevalexplain-results` | agent-supplied | TLC | |
| 2511.18782 | for Fix/Synthesize only or other summarization datasets (Summary-Mediated Repair | `humanevalexplain-results` | agent-supplied | TLC | |
| 2104.06022 | t al. 2021) and Takase & Kiyono's "Lessons on parameter sharing across layers" ( | `layer-looping-literature` | Claude-added | — | |
| 2310.10845 | re Better at Learning Learning Algorithms* (Yang et al. 2311.12424); CoTFormer ( | `layer-looping-literature` | Claude-added | — | |
| 2311.12424 | ); *Looped Transformers Are Better at Learning Learning Algorithms* (Yang et al. | `layer-looping-literature` | Claude-added | — | |
| 2502.13842 | * (Yang et al. 2311.12424); CoTFormer (2310.10845); Inner Thinking Transformer ( | `layer-looping-literature` | Claude-added | — | |
| 2502.17416 | Latent Thoughts: On the Power of Looped Transformers* (Saunshi et al. ICLR 2025, | `layer-looping-literature` | Claude-added | — | |
| 2507.10524 | ing**, which is exactly what was asked for: *Mixture-of- Recursions* (Bae et al. | `layer-looping-literature` | Claude-added | — | |
| 2510.04871 | lineage sharing patterns (cycle / sequence / cycle-rev); Tiny Recursive Models ( | `layer-looping-literature` | Claude-added | — | |
| 2510.25741 | caling analysis); *Ouro / Scaling Latent Reasoning via Looped Language Models* ( | `layer-looping-literature` | Claude-added | — | |
| (no ID) | Schwethelm et al., iso-depth scaling law for looped LMs | `layer-looping-literature` | agent-supplied | — | |
| 2207.04993 | ution recursion; no ID); Liger linearization (2503.01496); Embedding recycling ( | `layer-looping-literature` | agent-supplied | — | |
| 2301.13196 | 2021). - *Theory:* Looped Transformers as Programmable Computers (Giannou et al. | `layer-looping-literature` | agent-supplied | — | |
| 2310.07096 | wise batching for 2–3× throughput); **Sparse Universal Transformer** (Tan et al. | `layer-looping-literature` | agent-supplied | — | |
| 2310.19956 | rmers (Wang et al.; no ID); depth and compositional generalization (Petty et al. | `layer-looping-literature` | agent-supplied | — | |
| 2401.12819 | fixes UT's poor parameter-to-compute ratio); **Dynamic Layer Tying** (Hay et al. | `layer-looping-literature` | agent-supplied | — | |
| 2402.00976 | ad et al. ICLR 2020), Recurrent Transformers with Dynamic Halt (Chowdhury et al. | `layer-looping-literature` | agent-supplied | — | |
| 2402.11819 | rsive transformers** (2512.12880); **Head-wise Shareable Attention** (Cao et al. | `layer-looping-literature` | agent-supplied | — | |
| 2405.16039 | g halting, ~50% compute cut on formal-language tasks); **MoEUT** (Csordás et al. | `layer-looping-literature` | agent-supplied | — | |
| 2409.15647 | etained accuracy); **Looped Transformers for Length Generalization** (Fan et al. | `layer-looping-literature` | agent-supplied | — | |
| 2410.08292 | looped transformers learn multi-step gradient descent in context (Gatmiry et al. | `layer-looping-literature` | agent-supplied | — | |
| 2410.11268 | n multi-step gradient descent in context (Gatmiry et al. 2410.08292; Chen et al. | `layer-looping-literature` | agent-supplied | — | |
| 2410.20672 |  recursive variants at LM scale:* **Relaxed Recursive Transformers** (Bae et al. | `layer-looping-literature` | agent-supplied | — | |
| 2502.05171 | d loops to pretrained LMs); **recurrent-depth latent reasoning** (Geiping et al. | `layer-looping-literature` | agent-supplied | — | |
| 2503.01496 |  block); SpiralFormer (multi-resolution recursion; no ID); Liger linearization ( | `layer-looping-literature` | agent-supplied | — | |
| 2503.03961 |  al. 2410.08292; Chen et al. 2410.11268); log-depth expressivity (Merrill et al. | `layer-looping-literature` | agent-supplied | — | |
| 2511.07384 | ralization** (Fan et al. 2409.15647); **Retrofitted recurrence** (McLeish et al. | `layer-looping-literature` | agent-supplied | — | |
| 2512.12880 | 19; learn which layers share); **Mixture of LoRAs for recursive transformers** ( | `layer-looping-literature` | agent-supplied | — | |
| 2312.10523 | tions. - Missing canon for the actual question: Paloma (per-domain BPB protocol, | `loss-alternative-metrics-literature` | Claude-added | TINY, DCARD, IRT | |
| 2508.13144 | 12.10523); DataDecide's per-character correct-probability and Signal-and-Noise ( | `loss-alternative-metrics-literature` | Claude-added | TINY, DCARD, IRT | |
| 2309.10668 | 11.10618); the review omits Delétang et al. "Language Modeling Is Compression" ( | `loss-alternative-metrics-literature` | agent-supplied | TINY, DCARD, IRT | |
| 2401.17139 |  (prediction-independent, not architecture-independent):* Diff-eRank (Wei et al. | `loss-alternative-metrics-literature` | agent-supplied | TINY, DCARD, IRT | |
| 2404.07965 | ench. Reference-model token scoring (Rho-1 / "Not all tokens are what you need", | `loss-alternative-metrics-literature` | agent-supplied | TINY, DCARD, IRT | |
| 2405.14782 | okens) — the standard fix, used by Biderman et al. "Lessons from the trenches" ( | `loss-alternative-metrics-literature` | agent-supplied | TINY, DCARD, IRT | |
| 2410.10672 | ined model, tracks loss and accuracy with scale), Matrix Nuclear-Norm (Li et al. | `loss-alternative-metrics-literature` | agent-supplied | TINY, DCARD, IRT | |
| 2410.14480 | Nuclear-Norm (Li et al. 2410.10672; O(n²) surrogate, 8–24× faster), a hybrid (Vo | `loss-alternative-metrics-literature` | agent-supplied | TINY, DCARD, IRT | |
| 2410.23771 | ly).** - *Token-selected / reweighted NLL as a metric:* **LongPPL** (Fang et al. | `loss-alternative-metrics-literature` | agent-supplied | TINY, DCARD, IRT | |
| 2411.15320 | he closest published instance of Danielle's example. **PPLqa** (Friedland et al. | `loss-alternative-metrics-literature` | agent-supplied | TINY, DCARD, IRT | |
| 2412.03719 |  larger gains out-of-domain and where tokenizer entropy is high; Vieira et al. ( | `loss-alternative-metrics-literature` | agent-supplied | TINY, DCARD, IRT | |
| 2511.08066 | s replacement" unclear. - *Compression-based:* information capacity (Yuan et al. | `loss-alternative-metrics-literature` | agent-supplied | TINY, DCARD, IRT | |
| 2511.10618 | ncy including tokenizer efficiency), entropy-estimation modelling (Badger et al. | `loss-alternative-metrics-literature` | agent-supplied | TINY, DCARD, IRT | |
| 2401.19489 | phy (EPiC given as both 2408.11198 and 2410.14321; AlphaCodium as 2401.08500 and | `nl-bottleneck-prior-art` | Claude-added | TLC | |
| 2408.11198 | eyword sample. - Identifier slips in the ICBINB bibliography (EPiC given as both | `nl-bottleneck-prior-art` | Claude-added | TLC | |
| 2410.14321 | - Identifier slips in the ICBINB bibliography (EPiC given as both 2408.11198 and | `nl-bottleneck-prior-art` | Claude-added | TLC | |
| (no ID) | GenDLN — Evolutionary stacked-LLM joint prompt optimization, ACL SRW 2025, DOI 10.18653/v1/2025.acl-srw.92 | `nl-bottleneck-prior-art` | agent-supplied | TLC | |
| 2306.02907 | uction (Misu et al. 2024 Dafny, 10.1145/3643763, three prompt styles; SelfEvolve | `nl-bottleneck-prior-art` | agent-supplied | TLC | |
| 2401.08500 | ons and baseline prompts for the workshop paper — direct generation (AlphaCodium | `nl-bottleneck-prior-art` | agent-supplied | TLC | |
| 2503.11085 | 3643763, three prompt styles; SelfEvolve 2306.02907), LLM-as-optimizer (Prochemy | `nl-bottleneck-prior-art` | agent-supplied | TLC | |
| 2508.05995 | Evolve 2306.02907), LLM-as-optimizer (Prochemy 2503.11085; EPiC; RL4QE; MCTS-OPS | `nl-bottleneck-prior-art` | agent-supplied | TLC | |
| 2402.18700 | omission can induce hallucination. - **Nano-Capsulator — Zhou et al. 2024, arXiv | `prompt-compression-and-optimization-literature` | agent-supplied | TLC, ELI | |
| 2408.11198 |  to few-shot CoT and reading comprehension. - **EPiC — Saluja et al. 2024, arXiv | `prompt-compression-and-optimization-literature` | agent-supplied | TLC, ELI | |
| 2309.16797 |  *Method families and representatives:* evolutionary / population (Promptbreeder | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2402.11347 |  representatives:* evolutionary / population (Promptbreeder 2309.16797, PhaseEvo | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2402.17564 | s as prompt optimizers ≈ gradient optimizers", update direction + update method, | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2405.18369 |  2410.08696); meta-prompting / generation-refinement with a critic (PromptWizard | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2406.11132 | us parameters incl. prompts, hyperparameters, and code; LLM-AutoDiff; RePrompt ( | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2406.13443 | 73 — textual gradients through multi-component workflows; Dual-Phase accelerated | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2406.15708 | r optimization* as a settled finding (Wan et al. "Teach better or show smarter", | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2406.16218 | mized system prompt ≈ task-specific prompts across 47 tasks; Trace / OptoPrime ( | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2408.10504 | ows; Dual-Phase accelerated 2406.13443); query-dependent learned generators (QPO | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2408.11198 |  *Verifiable / code:* Prochemy (2503.11085, execution-driven refinement), EPiC ( | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2409.15199 |  update direction + update method, 2402.17564; Learning from Contrastive Prompts | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2410.08696 | * evolutionary / population (Promptbreeder 2309.16797, PhaseEvo 2402.11347, AMPO | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2410.14826 | an et al. "Teach better or show smarter", 2406.15708). - *System-level:* SPRIG ( | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2410.16392 |  histories; Lin et al. survey of LLM-based optimization of compound AI systems ( | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2501.16673 | Contrastive Prompts 2409.15199); gradient-inspired textual updates (LLM-AutoDiff | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2503.11085 | eval configs, tool specs, orchestration logic. - *Verifiable / code:* Prochemy ( | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2507.09839 | s (Davari et al. reinforcement + diversification + migration for black-box LLMs, | `prompt-optimization-landscape` | agent-supplied | TLC, ELI | |
| 2205.10487 | etition up to ~4 epochs is nearly free, then diminishing) and Hernandez et al. ( | `regularization-literature` | Claude-added | MSUITE, dense | |
| 2205.14336 | nding that sparse models overfit in fine-tuning and its z-loss, Gating Dropout ( | `regularization-literature` | Claude-added | MSUITE, dense | |
| 2305.13230 | 23, "To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis" ( | `regularization-literature` | Claude-added | MSUITE, dense | |
| 2305.16264 | els overfit repeated data *more* than dense ones. Pair with Muennighoff et al. ( | `regularization-literature` | Claude-added | MSUITE, dense | |
| 2207.09094 | l. 2017; Switch, Fedus et al. 2022]; expert / cluster-level expert dropout (MoEC | `regularization-literature` | agent-supplied | MSUITE, dense | |
| 2209.09735 | ; layer norm and label smoothing (cited to Liu et al. 2020); relaxed attention ( | `regularization-literature` | agent-supplied | MSUITE, dense | |
| 2402.14526 | , SoftDedup commonness reweighting (2407.06654), ClusterClip balanced sampling ( | `regularization-literature` | agent-supplied | MSUITE, dense | |
| 2407.06654 | CL 2022; Kandpal et al. ICML 2022 — privacy), SoftDedup commonness reweighting ( | `regularization-literature` | agent-supplied | MSUITE, dense | |
| 2510.01185 |  multilingual MT); Dirichlet-prior shaping of router outputs for upcycled MoEs ( | `regularization-literature` | agent-supplied | MSUITE, dense | |
| 2402.04177 | ` cited as FinPythia DACP but the entry is PIXIU; `chang2024effective` points to | `small-scale-evaluation-metrics-literature` | Claude-added | TINY, IRT, EDP, DCARD | |
| (no ID) | "Koh & Liang 2026, rBridge" — unverifiable; likely fabricated | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY | |
| (no ID) | Xie et al. 2023 FinPythia-6.9B DACP (v2 bib points at PIXIU) | `small-scale-evaluation-metrics-literature` | agent-supplied | FUNC | |
| 2205.10487 | knowledge capacity 2 bits/parameter (2404.05405); repeated-data double descent ( | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2206.07682 | 404.05405); repeated-data double descent (2205.10487). - *Emergence:* Wei et al. | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2207.05221 | s a benchmark axis (2401.12794); self-evaluation EQT 2501.11721; Kadavath et al. | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2304.15004 | t (2205.10487). - *Emergence:* Wei et al. 2206.07682 vs. Schaeffer et al. mirage | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2305.12415 | 5% R², "small-bench" 3× smaller than BBH equally informative); Schellaert et al. | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2305.14947 | logistic fits; zero-shot to unseen families); Ye et al. BIG-bench predictability | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2305.16264 | ); finetuning scaling laws need R² ≥ 0.95 (Ivgi et al. 2022); data-constrained ( | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2305.17266 | iction:* Kaplan; Chinchilla; small-scale break below ~2.2e15 FLOPs (Pechi et al. | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2308.08493 | asks for emergent abilities 2412.07111. - *Contamination:* time-travel detection | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2310.08754 | soft-match accuracy); tokenizer metrics uncorrelated with downstream (Ali et al. | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2310.13800 | er accuracy/F1 (2401.03831); reference-based metrics failing for modern models ( | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2401.03831 | rmulation for 35× cheaper evaluation 2506.03592; Informedness over accuracy/F1 ( | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2401.12794 | ); conformal probes (Ashok & May 2025); conformal set size as a benchmark axis ( | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2403.08540 | raged over the training distribution, blind to which tokens matter; Gadre et al. | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2403.16952 | xy models often fail to predict larger ones); BiMix 2405.14908; data mixing laws | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2404.05405 |  tokenizer, not architecture (2502.12120); knowledge capacity 2 bits/parameter ( | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2405.10938 | Ps in overtrained regimes); context-aware 2510.14919; observational scaling laws | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2405.14908 | 75; ADO 2410.11820 (small proxy models often fail to predict larger ones); BiMix | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2406.01375 | e / selection laws:* AutoScale 2407.20177; UtiliMax / MEDU 2501.11747; D-CPT law | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2406.09334 | t et al. 2305.12415 (DeBERTa assessors predicting per-instance success); ProxyLM | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2406.11243 | regularized matrix factorization 2504.19811 (model ancestry as a prior); FamiCom | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2407.20177 |  hyperparameter scaling 2505.13738. - *Data mixture / selection laws:* AutoScale | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2409.15790 | s:* SLM-Bench 2508.15478 (15 SLMs, 9 tasks, 11 metrics incl. energy); SLM survey | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2410.03083 | 64); quality-aware Q (2510.03313); effective tokens = diversity × syntheticity ( | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2410.08527 |  (2410.03083, r = 0.83 over 200 models 25M–1.5B); FLP two-stage loss→performance | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2410.11820 | ws:* AutoScale 2407.20177; UtiliMax / MEDU 2501.11747; D-CPT law 2406.01375; ADO | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2412.04403 | FLP two-stage loss→performance 2410.08527 (5–10% error at 7B/13B); model ladders | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2412.04947 | bilities 2412.07111. - *Contamination:* time-travel detection 2308.08493; C2LEVA | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2412.07111 | 07682 vs. Schaeffer et al. mirage 2304.15004; proxy tasks for emergent abilities | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2501.11721 |  2025); conformal set size as a benchmark axis (2401.12794); self-evaluation EQT | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2501.11747 | .13738. - *Data mixture / selection laws:* AutoScale 2407.20177; UtiliMax / MEDU | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2502.12120 | 09404; loss-to-loss scaling determined by data and tokenizer, not architecture ( | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2504.19811 | multilingual performance, 37× speedup); lineage-regularized matrix factorization | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2505.13738 | low-dimensional capability space, emergence as sigmoids); hyperparameter scaling | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2506.03592 | d traces still correct); generative→NLU reformulation for 35× cheaper evaluation | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2507.03160 | (15 SLMs, 9 tasks, 11 metrics incl. energy); SLM survey 2409.15790; SLMs on code | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2507.09404 | arger ones); BiMix 2405.14908; data mixing laws 2403.16952; optimal-mixture laws | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2508.15478 | 1; Kadavath et al. 2207.05221 calibration. - *Small-model benchmarks:* SLM-Bench | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2510.03313 | d R² ≥ 0.95 (Ivgi et al. 2022); data-constrained (2305.16264); quality-aware Q ( | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2510.09351 | metrics incl. energy); SLM survey 2409.15790; SLMs on code 2507.03160; ReTraceQA | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2510.14919 |  points on some tasks; N and D beat FLOPs in overtrained regimes); context-aware | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2512.08894 | r metrics uncorrelated with downstream (Ali et al. 2310.08754); Krajewski et al. | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2601.19831 | acy route). - *Learned / neural predictors:* NeuNeu "Neural Neural Scaling Laws" | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2605.18607 | wnstream tasks. One example of a related paper > would be https://arxiv.org/abs/ | `small-scale-evaluation-metrics-literature` | agent-supplied | TINY, IRT, EDP, DCARD | |
| 2305.17493 | motron-CC / Cosmopedia rephrasing corpora; Shumailov et al. curse of recursion ( | `synthetic-data-literature` | Claude-added | REC | |
| 2401.16380 | ndle; see `INDEX.md`. - Canon to check for in the PDF before relying on it: WRAP | `synthetic-data-literature` | Claude-added | REC | |
| 2404.01413 |  curse of recursion (2305.17493); Gerstgrasser et al. accumulate-don't-replace ( | `synthetic-data-literature` | Claude-added | REC | |
| 2405.03548 | n the paste addresses scaling or collapse; the PDF sections do. - Two key PDFs ( | `synthetic-data-literature` | Claude-added | REC | |
| 2510.01631 |  addresses scaling or collapse; the PDF sections do. - Two key PDFs (2405.03548, | `synthetic-data-literature` | Claude-added | REC | |
| 2212.10560 | tion methods:* rephrasing the web (WRAP; subsection in the PDF); Self-Instruct ( | `synthetic-data-literature` | agent-supplied | REC | |
| 2305.16264 | 508.10975; trillion-scale rephrasing, "lessons"); scaling data-constrained LMs ( | `synthetic-data-literature` | agent-supplied | REC | |
| 2402.07043 | lapse and overfitting:* collapse as a change of scaling laws ("A Tale of Tails", | `synthetic-data-literature` | agent-supplied | REC | |
| 2402.13064 | rasing the web (WRAP; subsection in the PDF); Self-Instruct (2212.10560); GLAN ( | `synthetic-data-literature` | agent-supplied | REC | |
| 2402.17193 | 5.16264; repetition up to ~4 epochs ≈ free); when scaling meets LLM finetuning ( | `synthetic-data-literature` | agent-supplied | REC | |
| 2404.07503 |  training (2410.15226); best practices and lessons on synthetic data (COLM 2024, | `synthetic-data-literature` | agent-supplied | REC | |
| 2406.07515 | 043); beyond collapse — scaling up with synthesized data requires verification ( | `synthetic-data-literature` | agent-supplied | REC | |
| 2406.15126 | tructions); Instruct-SkillMix and diversity-driven generation; the two surveys ( | `synthetic-data-literature` | agent-supplied | REC | |
| 2410.12896 | os, multi-step and dataset-wise decomposition; curation and evaluation taxonomy; | `synthetic-data-literature` | agent-supplied | REC | |
| 2410.15226 | rameter Q (2510.03313); diversity of synthetic data and its effect on training ( | `synthetic-data-literature` | agent-supplied | REC | |
| 2412.14689 | ta requires verification (2406.07515); how to synthesize text without collapse ( | `synthetic-data-literature` | agent-supplied | REC | |
| 2508.10975 | theme, from the PDF's structure).** - *Scaling with synthetic data:* BeyondWeb ( | `synthetic-data-literature` | agent-supplied | REC | |
| 2510.03313 |  LLM finetuning (2402.17193); quality-aware scaling with a quality parameter Q ( | `synthetic-data-literature` | agent-supplied | REC | |
| 2506.20512 | -data line (MiniCPM, Llama 3, OLMo 2 / Dolmino, SmolLM, Nemotron), OctoThinker ( | `targeted-pretraining-midtraining-literature` | Claude-added | FUNC, ANN | |
| 2306.12070 | objective to optimize for fast adaptation. - **Task-robust minimax pretraining ( | `targeted-pretraining-midtraining-literature` | agent-supplied | FUNC, ANN | |
| 2512.07783 |  Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models" ( | `targeted-pretraining-midtraining-literature` | agent-supplied | FUNC, ANN | |
| 2207.14255 | leck et al. 2019); DeepSeek-V3's MTP at scale (2412.19437); fill-in-the-middle ( | `training-objective-alternatives-literature` | Claude-added | TOK, dense | |
| 2405.14394 | n-the-middle (2207.14255); instruction-loss masking vs. loss-over-instructions ( | `training-objective-alternatives-literature` | Claude-added | TOK, dense | |
| 2412.19437 | idence penalty; unlikelihood (Welleck et al. 2019); DeepSeek-V3's MTP at scale ( | `training-objective-alternatives-literature` | Claude-added | TOK, dense | |
| 2205.02517 | on math fine-tuning, 3–7B models). Contrastive token learning for degeneration ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2305.16958 | y up to 16% at the model-strong end, NLL wins at the model-weak end. **MixCE** ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2307.03170 | -space: **LLM-JEPA** (2509.14252), Focused Transformer contrastive KV training ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2309.08272 | 60), continuous-paragraph-denoise diffusion LMs, RTS/SLM structural objectives ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2401.13160 | one concept count as correct). Denoising: UL2 mixture-of-denoisers, SpacTor-T5 ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2402.14270 | ** (2412.14780; reasoning vs. boilerplate tokens by relative loss), **IR-DRO** ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2404.07965 | / information: **Power-Law Decay Loss** (2505.16900), **Rho-1 / selective LM** ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2404.19737 | g. - *Beyond next-token prediction.* **Multi-token prediction** (Gloeckle et al. | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2405.18906 | 305.16958; forward + reverse CE). **Strictly proper scoring rules** (Shao et al. | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2407.12665 | ains on code; up to 13B), MTP curricula (2505.22757), **patch-level training** ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2408.10613 | -level: **Velocitune** (2411.14318; weight domains by learning velocity), tDRO ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2409.13641 | ; +3 BLEU / ROUGE on LLaMA-7B). CV-inspired: focal, Lovász, Dice (Cambrin et al. | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2411.14318 | ** (2021; gradient edits to favour novel tokens). Domain-level: **Velocitune** ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2412.14780 | 509.20758; w ∝ p(x)^(1/τ), a curriculum that downweights hard tokens), **RFT** ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2505.16900 | op the highest as noise). By frequency / information: **Power-Law Decay Loss** ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2505.19893 | ng, train on the top tokens; 15B OpenWebMath and 80B general tokens), **ESLM** ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2505.22757 | itly upweights "choice-point" tokens; gains on code; up to 13B), MTP curricula ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2509.14252 | LMs, RTS/SLM structural objectives (2309.08272). Embedding-space: **LLM-JEPA** ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2509.20758 | ings NAACL; scale loss by predictive entropy; 468M–6.7B on the Pile), **TALR** ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2510.00526 |  scoring rules and probability families.* **"Beyond Log Likelihood"** (Li et al. | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2510.27462 |  minimization; GPT-2 pretraining FLOP savings). By gradient utility: **VCORE** ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2511.00198 | es, ~50% cost reduction at matched loss), "filling the mutual-information gap" ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2512.10545 | ** (2411.14318; weight domains by learning velocity), tDRO (2408.10613), XDoGE ( | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2601.11791 |  mutual-information gap" (2511.00198). **Concept-level objectives** (Iyer et al. | `training-objective-alternatives-literature` | agent-supplied | TOK, dense | |
| 2506.16982 | Language Bottleneck Models — Berthon & van der Schaar (second novelty check headline) | `nl-bottleneck-prior-art` | agent-supplied | TLC | check identified 2026-08-24: Perplexity, 2026-02-04; also Gemini check's headline |
| 2509.25196 | APRIL — RL prompt optimization for frozen code generators ("decoder half") | `nl-bottleneck-prior-art` | agent-supplied | TLC | |
| 2509.06239 | Proof2Silicon — RL prompt optimization for frozen LLM synthesis ("decoder half") | `nl-bottleneck-prior-art` | agent-supplied | TLC | |
| 2412.07992 | Concept Bottleneck LLMs — text concepts as bottleneck for classification | `nl-bottleneck-prior-art` | agent-supplied | TLC | |
| 2405.18392 | Hägele et al. — WSD/cooldown scaling laws, (1-sqrt) decay (annealing report 2) | `schedules-and-annealing-literature` | agent-supplied | ANN, WSD | |
| 2404.06395 | MiniCPM — decay-phase gradient dynamics, WSD (annealing report 2) | `schedules-and-annealing-literature` | agent-supplied | ANN, WSD | |
| 2406.17557 | FineWeb / FineWeb-Edu (annealing report 2) | `schedules-and-annealing-literature` | agent-supplied | ANN, WSD | |
| 2412.08905 | Phi-4 technical report — synthetic data, decontamination (annealing report 2) | `schedules-and-annealing-literature` | agent-supplied | ANN, WSD | |
| 2412.02595 | Nemotron-CC — ensemble classifiers, synthetic rephrasing (annealing report 2) | `schedules-and-annealing-literature` | agent-supplied | ANN, WSD | |
| 2412.17743 | YuLan-Mini — context extension during annealing (annealing report 2) | `schedules-and-annealing-literature` | agent-supplied | ANN, WSD | |
| 2503.17793 | benchmark contamination survey, 1–45% rates (annealing report 2) | `schedules-and-annealing-literature` | agent-supplied | ANN, WSD | |
| 2508.01483 | second LR-annealing scaling-law citation paired with Tissue 2408.11029 — unknown, possibly mis-ID | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2505.02881 | SwallowCode / SwallowMath — rewriting pretraining code/math data | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2409.17115 | ProX — programming every example, small-model data refinement | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2501.07314 | FinerWeb-10BT — LLM line-level filtering | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2410.04579 | temperature sampling vs. scalarization; mixture cooldown | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2407.08699 | Branch-and-Merge — merge subset-finetuned models, less forgetting | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2411.11266 | VersaTune — dynamic domain weighting in fine-tuning (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2405.07490 | curriculum learning, easy-to-hard (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2406.19853 | curriculum learning (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2411.02337 | curriculum learning (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2505.08364 | ADCL adaptive difficulty curriculum, "difficulty shift" (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2004.13833 | "data annealing" for informal language, 2020 — unrelated prior use of the term | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2501.00237 | domain shifts can reduce forgetting (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2405.17830 | catastrophic forgetting in LLM fine-tuning (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2308.08747 | forgetting scales with model size up to 7B (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2403.01244 | Self-Synthesized Rehearsal (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2310.05492 | instruction data mixing (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2406.08811 | Mixture-of-Skills (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2312.10793 | instruction data composition (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2410.10210 | quality over diversity in final-stage data (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2312.11508 | LIFT — instruction quality over quantity (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2508.03571 | KILO continual adaptation (drift) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2505.05427 | quality-over-quantity claim support (unexamined) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2412.06724 | LLM-as-judge quality assessment (unexamined) | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 2509.23629 | Annealed-RLVR — SFT heating inside RLVR; term collision | `schedules-and-annealing-literature` | agent-supplied | ANN | |
| 1604.04173 | Lei et al. — distribution-free predictive inference for regression (split conformal) | `estimation-and-calibration-methods` | agent-supplied | TLC, ELI, EDP | |
| 2208.02814 | Angelopoulos et al. — Conformal Risk Control | `estimation-and-calibration-methods` | agent-supplied | TLC, ELI, EDP | |
| 2107.03374 | Codex / HumanEval — unbiased pass@k estimator | `estimation-and-calibration-methods` | agent-supplied | TLC, ELI, EDP | |
| 2507.19457 | GEPA — Reflective Prompt Evolution Can Outperform Reinforcement Learning | `prompt-optimization-landscape` | agent-supplied | TLC | |
| 2604.25359 | The Structured Output Benchmark — schema compliance vs. value accuracy | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2602.12247 | ExtractBench — enterprise-document extraction; schema breadth failures | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2501.10868 | JSONSchemaBench — constrained decoding frameworks | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2605.02363 | "When Correct Isn't Usable" — 7–9B models solve but can't emit usable JSON | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2602.14743 | LLMStructBench — 22 models; validity vs. wrong values | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2507.01810 | clinical SLM extraction; JSON/YAML/XML parseability | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2507.18546 | GLiNER2 — unified NER/classification/structured extraction | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2602.15189 | ScrapeGraphAI-100k — schema-constrained web extraction dataset; 1.7B fine-tune | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2603.15118 | VAREX — document-extraction benchmark; sub-4B compliance vs. extraction | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2502.18878 | Schema Reinforcement Learning | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2512.00319 | RL-Struct — dense schema-derived rewards | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2604.14862 | Schema key wording as an instruction channel | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2510.07248 | PA-Tool — adapting tool/schema names to small models | `structured-output-literature` | agent-supplied | ELI, IRT | |
| 2504.11393 | DataDecide (Magnusson et al. 2025) — metric definitions cited from the HTML version | `datadecide-data-pipeline` | agent-supplied | DCARD, TINY | |
| 2407.21072 | length-normalization / multiple-choice scoring paper cited for acc_raw vs. per-length accuracies — title unknown | `datadecide-data-pipeline` | agent-supplied | DCARD | |
| 1311.2540 | Duda — Asymmetric Numeral Systems | `code-compression-literature` | agent-supplied | TLC | |
| 2107.03312 | SoundStream — neural audio codec | `code-compression-literature` | agent-supplied | TLC | |
| 1211.0557 | STOKE — stochastic superoptimization | `code-compression-literature` | agent-supplied | TLC | |
| 1711.04422 | Souper — synthesizing superoptimizer for LLVM IR | `code-compression-literature` | agent-supplied | TLC | |
| 2006.08381 | DreamCoder | `code-compression-literature` | agent-supplied | TLC | |
| 2310.19791 | LILO — learning interpretable libraries by compressing and documenting code | `code-compression-literature` | agent-supplied | TLC | |
| 2212.04596 | babble — library learning with e-graphs and anti-unification | `code-compression-literature` | agent-supplied | TLC | |
| 2503.13992 | The KoLMogorov Test — compression by code generation | `code-compression-literature` | agent-supplied | TLC | |
| physics/0004057 | Tishby, Pereira & Bialek — The information bottleneck method | `text-latent-code-autoencoder` §4 | agent-supplied | TLC | |
| 1612.00410 | Alemi et al. — Deep Variational Information Bottleneck (OpenReview HyxQzBceg; ID Claude-supplied) | `text-latent-code-autoencoder` §4 | Claude-added | TLC | |
| 1807.03748 | van den Oord et al. — Contrastive Predictive Coding / InfoNCE | `text-latent-code-autoencoder` §4 | agent-supplied | TLC | |
| 2002.10689 | Xu et al. — A Theory of Usable Information Under Computational Constraints (𝒱-information) | `text-latent-code-autoencoder` §4 | agent-supplied | TLC, ELI | |
| (ACL 2020.emnlp-main.14) | Voita & Titov — Information-Theoretic Probing with Minimum Description Length | `text-latent-code-autoencoder` §4 | agent-supplied | TLC | |
| 1606.04155 | Lei, Barzilay & Jaakkola — Rationalizing neural predictions | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 1909.09436 | CodeSearchNet | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2002.08155 | CodeBERT | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2009.08366 | GraphCodeBERT | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2109.00859 | CodeT5 | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2007.04973 | ContraCode — contrastive code representation learning | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2009.02731 | Corder | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2204.03293 | CoCoSoDa | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2202.08975 | Troshin & Chirkova — probing pretrained source-code models | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2207.07706 | Naik et al. — RSA probing of semantic grounding in CodeBERT | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 1706.05806 | SVCCA | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2002.12462 | LEEP — transferability estimate | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2108.07732 | MBPP (Austin et al.) | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2105.12655 | Project CodeNet | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2009.10297 | CodeBLEU | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 1711.07163 | dynamic neural program embeddings from execution traces | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2411.15594 | LLM-as-a-judge survey | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2407.05411 | intermediate languages for code generation (NL vs pseudocode vs PL) | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2505.15356 | NL-Debugging — NL as intermediate representation for debugging | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |
| 2305.04388 | Turpin et al. — language models don't always say what they think (CoT faithfulness) | `nl-bottleneck-prior-art` (measurement) | agent-supplied | TLC | |

Rows: 208 (181 agent-supplied, 27 Claude-added, 5 named without ID).

Known-bad or already-resolved items (do not re-verify): Patel et al. 2605.18607 authors
are Patel, Reddy, Mosbach, Bahdanau (PDF in the eval-of-llms bundle); EPiC is 2408.11198
not 2410.14321; AlphaCodium is 2401.08500 not 2401.19489; `gao2021framework` in the
small-scale-metrics v2 bib is The Pile, not lm-eval-harness; `luo2025scaling` there is
WizardCoder; `chang2024effective` there is 2402.04177 (different paper) and
`bhagia2024scaling` there is 2410.08527 (the FLP paper; model ladders is 2412.04403);
"The Shannon Paradox … 0.36 BPC" (Zenodo) is not a credible source.
| 2203.05482 | Model soups (Wortsman … Schmidt) — weight-space ensemble bridge | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2406.18665 | RouteLLM (Ong … Stoica) — query routing between deployed LLMs | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2312.02829 | MIMONets (Menet … Rahimi) — packed ensemble via superposition | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2310.07707 | MatFormer (Devvrit … Jain) — nested Transformer, elastic inference | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2402.00433 | Weight-Ensembling MoE for multi-task merging (Tang … Tao) | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2402.08562 | Higher Layers Need More LoRA Experts (Gao … Subrahmanian) | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2404.15159 | MixLoRA (Li … Tang) — LoRA experts with sparse routing in FFN | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2312.07987 | SwitchHead (Csordás … Schmidhuber) — MoE attention | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2404.02258 | Mixture-of-Depths (Raposo … Santoro) | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2308.00951 | From Sparse to Soft Mixtures of Experts (Puigcerver … Houlsby) | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2401.06066 | DeepSeekMoE (Dai … Liang) — shared + routed experts | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2401.04088 | Mixtral of Experts (Jiang … El Sayed) | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2402.07871 | Scaling Laws for Fine-Grained MoE (Krajewski … Jaszczur) | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2002.06715 | BatchEnsemble (Wen et al.) — response 1 citation | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2005.00247 | AdapterFusion (Pfeiffer et al.) — response 1 citation | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2404.13628 | Mixture of LoRA Experts — response 1 citation | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2210.05144 | Mixture of Attention Heads — response 1 citation | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2208.03306 | Branch-Train-Merge (Li et al.) — response 1 citation | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 1701.06538 | Sparsely-Gated MoE (Shazeer et al.) — response 1 citation | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2209.14375 | "hydra"/shared-trunk citation in response 1 — paper identity unchecked | `moe-literature` (design space) | agent-supplied | MOVE/PART/MSUITE | |
| 2403.07816 | Branch-Train-MiX — named in response 1, dropped from the list; closer fit for the gated-whole-model cell | `moe-literature` (design space) | Claude-added | MOVE/PART/MSUITE | |
| 2305.01210 | HumanEval+ / EvalPlus — ~80× tests | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2403.19114 | EvoEval — LLM-evolved HumanEval, 7 benchmarks / 828 problems | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2412.21199 | HumanEval Pro / MBPP Pro — self-invoking code generation | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2412.01526 | cited for HumanEval_T / DyCodeEval / HumanEvalNext together — at most one is right | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | April Perplexity convo pins this ID to the combinatorial-test-design paper (HumanEval_T's description) and gives HumanEvalNext 2503.05860 — partially untangles the triple |
| 2303.17568 | CodeGeeX / HumanEval-X — multilingual hand-written | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2208.08227 | MultiPL-E — HumanEval/MBPP transpiled to 18 languages | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2402.16694 | HumanEval-XL — 23 NLs × 12 PLs | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2212.10264 | ReCode — robustness transformations over docstrings/names/syntax/format | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2406.19783 | NLPerturbator / HumanEval-R — NL perturbations | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2406.00215 | HumanEvalComm — ambiguous descriptions, clarifying-question metrics | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2404.03114 | unit-test generation under comment/name/docstring manipulation — identity unchecked | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2410.12381 | HumanEval-V — visual context | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2406.14712 | Qiskit HumanEval | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2403.07974 | LiveCodeBench | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2501.10711 | How2Bench — audit of 274 code benchmarks | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | title-pairing check: paper-DB page for this ID is titled "Code Benchmarks Should Prioritize Rigor, Reliability, and Reproducibility" (2026-08-24) |
| 2502.06215 | LessLeak-Bench — 83 SE benchmarks vs pretraining corpora | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2401.07930 | CodeSearchNet inter-dataset duplication (SourcererCC) | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2602.05892 | ContextBench — pooled + deduplicated issue-resolution tasks | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2403.04811 | HumanEval/MBPP contamination in the Pile / The Stack (Levenshtein + Dolos) | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| 2402.19173 | StarCoder2 / The Stack v2 — near-dedup pipeline | `code-benchmarks-landscape` (HumanEval afterlife / overlap) | agent-supplied | TLC/ELI/IRT | |
| (no ID) | ShortenDoc — docstring compression on HumanEval/EvoEval; no identifier given in the response | `code-benchmarks-landscape` (HumanEval afterlife) | agent-supplied | TLC (gate 1) | |
| 2407.17465 | u-µP: Unit-Scaled Maximal Update Parametrization (Blake et al.) — Danielle-supplied PDF/URL | `parametrization-and-hp-transfer` | Danielle-supplied | GEO/TINY/EDP/DataDecide-dense | |
| 2203.03466 | µTransfer (Yang et al.) — named in intake note only | `parametrization-and-hp-transfer` | Claude-added | — | |
| 2303.11257 | Unit Scaling (Blake et al.) — named in intake note only | `parametrization-and-hp-transfer` | Claude-added | — | |
| 1404.3325 | Vigna — weighted Kendall τ (hyperbolic weights) — named in intake note only | `estimation-and-calibration-methods` (ranking metrics) | Claude-added | EDP | |
| (no ID) | Webber, Moffat & Zobel 2010, rank-biased overlap (ACM TOIS) — named in intake note only | `estimation-and-calibration-methods` (ranking metrics) | Claude-added | EDP | |
| 2309.03409 | OPRO — "Large Language Models as Optimizers" | `prompt-optimization-landscape` (2026-08-23 re-eval conversation) | agent-supplied | prompt-opt re-eval (TBD) | |
| 2405.10276 | Revisiting OPRO — small-scale LLMs as optimizers; ACL Findings 2024 per respondent (Danielle-named title) | `prompt-optimization-landscape` (2026-08-23 re-eval conversation) | agent-supplied | prompt-opt re-eval (TBD) | |
| 2403.02054 | LEO — evolutionary optimizer, reasoning with elitism (Danielle-named title) | `prompt-optimization-landscape` (2026-08-23 re-eval conversation) | agent-supplied | prompt-opt re-eval (TBD) | |
| 2502.16923 | Systematic survey of Automatic Prompt Optimization (2025) | `prompt-optimization-landscape` (2026-08-23 re-eval conversation) | agent-supplied | prompt-opt re-eval (TBD) | |
| 2502.11560 | Survey of Automatic Prompt Engineering (2025) | `prompt-optimization-landscape` (2026-08-23 re-eval conversation) | agent-supplied | prompt-opt re-eval (TBD) | |
| 2512.01420 | PromptBridge — cross-model prompt transfer (2025) | `prompt-optimization-landscape` (2026-08-23 re-eval conversation) | agent-supplied | prompt-opt re-eval (TBD) | |
| 2402.18700 | Pitch bibliography: Chuang et al. 2024, "Learning to compress prompt in natural language formats" — CONFLICTS with earlier agent-supplied row calling this "Nano-Capsulator — Zhou et al." | `text-latent-code-autoencoder` §4 (2026-08-23 pitch) | Danielle-supplied | TLC | |
| 2408.11198 | Pitch bibliography: Taherkhani et al. 2025, "Automated prompt engineering for cost-effective code generation using evolutionary algorithm" — CONFLICTS with earlier agent-supplied row calling this "EPiC — Saluja et al." | `text-latent-code-autoencoder` §4 (2026-08-23 pitch) | Danielle-supplied | TLC | |
| 2006.03511 | TransCoder — "Unsupervised Translation of Programming Languages" (2020) | `text-latent-code-autoencoder` §4 (chunk 3, cross-language) | agent-supplied | TLC (library leverage) | |
| 2110.06773 | "Leveraging Automated Unit Tests for Unsupervised Code Translation" (2021) | `text-latent-code-autoencoder` §4 (chunk 3, cross-language) | agent-supplied | TLC (library leverage) | |
| (no ID) | Cost-aware evolutionary prompt optimization, PMLR v293 zehle25a — only a proceedings link given | `text-latent-code-autoencoder` §4 (chunk 3, curriculum) | agent-supplied | TLC (optimizer loop) | |
| 2211.11501 | DS-1000 — data-science code generation benchmark (pandas/numpy/matplotlib/sklearn/scipy/torch) | `code-benchmarks-landscape` (2026-08-23 efficiency cluster) | agent-supplied | TLC (task families), divergence probe | |
| 2406.06647 | ENAMEL — eff@k efficiency metric, expert references; catalyzex link originally, ID from the April Perplexity convo | `code-benchmarks-landscape` (2026-08-23 efficiency cluster) | agent-supplied | divergence probe | |
| (no ID) | EvalPerf / DPE — efficiency-tiered reference profiling; only a catalyzex link given | `code-benchmarks-landscape` (2026-08-23 efficiency cluster) | agent-supplied | divergence probe | |
| 2402.02037 | EffiBench — NeurIPS D&B 2024, efficiency vs human canonical solutions; proceedings PDF link originally, ID from the April Perplexity convo | `code-benchmarks-landscape` (2026-08-23 efficiency cluster) | agent-supplied | divergence probe | |
| (no ID) | BigO(Bench) — complexity prediction + complexity-controlled generation (Facebook Research; only a github.io link given) | `code-benchmarks-landscape` (chunk-6 addition) | agent-supplied | divergence probe | |
| (no ID) | SymC — code symmetries, invariance to semantics-preserving rewrites (PMLR v235 pei24b; only proceedings link given) | `text-latent-code-autoencoder` §4 (chunk 9, canonicalization) | agent-supplied | TLC (canonicalization) | |
| (no ID) | ProgramTransformer — semantics-preserving transformation toolchain (ScienceDirect; no ID given) | `text-latent-code-autoencoder` §4 (chunk 9, canonicalization) | agent-supplied | TLC (canonicalization) | |
| 2403.14714 | LLM + compiler feedback loops for optimization (title not stated in response) | `text-latent-code-autoencoder` §4 (chunk 9, canonicalization) | agent-supplied | TLC (canonicalization) | |
| 1910.13461 | BART — denoising autoencoder for text (analogy for behavior-preserving normalization) | `text-latent-code-autoencoder` §4 (chunk 12) | agent-supplied | TLC (framing) | |
| 1609.07317 | Miao & Blunsom 2016 — Language as a Latent Variable (ASC); advisor-recommended, PDF in bundle | `nl-bottleneck-prior-art` + `text-latent-code-autoencoder` §4 | Danielle-supplied | TLC (gate 1 candidate) | yes (title/authors/ID checked against PDF) |
| 2304.12512 | Gilbert et al. 2023 — Semantic compression with LLMs (submitted paper's bibliography) | `text-latent-code-autoencoder` §4 (submission record) | Danielle-supplied | TLC | April Perplexity convo adds the ERE/SRE metric pair and a ~5× compression claim |
| 2402.08756 | Diesendruck et al. 2024 — cycle-consistency refines prompts (submitted paper's bibliography) | `text-latent-code-autoencoder` §4 (submission record) | Danielle-supplied | TLC | |
| 2505.15356 | Zhang et al. 2025 — NL-debugging: NL as intermediate representation for code debugging (submitted paper's bibliography) | `text-latent-code-autoencoder` §4 (submission record) | Danielle-supplied | TLC | |
| 1910.05923 | Wei et al. 2019 — code generation as dual task of summarization (submitted paper's bibliography) | `text-latent-code-autoencoder` §4 (submission record) | Danielle-supplied | TLC | |
| 2507.08627 | Wong et al. 2025 — NL in the middle: code translation with IRs (submitted paper's bibliography) | `text-latent-code-autoencoder` §4 (submission record) | Danielle-supplied | TLC | authorship discrepancy: 2026-02-04 Consensus review cites Tai, Nie, Golab & Wong (CASCON 2025) — resolve first author |
| 2409.03733 | Wang et al. 2024 — planning in natural language improves LLM search for code generation (submitted paper's bibliography) | `text-latent-code-autoencoder` §4 (submission record) | Danielle-supplied | TLC | |
| (no ID) | Makharev & Ivanov 2025 — code summarization beyond function level (prompt-slot source in submitted paper) | `text-latent-code-autoencoder` §4 (submission record) | Danielle-supplied | TLC (prompt space) | |
| (no ID) | Poudel et al. 2024 — DocuMint: docstring generation with small LMs (prompt-slot source in submitted paper) | `text-latent-code-autoencoder` §4 (submission record) | Danielle-supplied | TLC (prompt space) | |
| 2402.08699 | Allamanis et al. 2024 — RTC: round-trip correctness as unsupervised eval of code LLMs (2026-02-03 Notion novelty check) | `nl-bottleneck-prior-art` (2026-08-24 entry); litreview plan row A | agent-supplied (ChatGPT) | TLC | |
| 2304.05128 | Chen et al. 2023 — Self-Debug: teaching LLMs to self-debug (2026-02-03 Notion novelty check) | `nl-bottleneck-prior-art` (2026-08-24 entry) | agent-supplied (ChatGPT) | TLC | |
| 2309.12499 | Bairi et al. 2024 — CodePlan: repository-level coding with planning, FSE 2024 (2026-02-03 Notion novelty check) | `nl-bottleneck-prior-art` (2026-08-24 entry) | agent-supplied (ChatGPT) | TLC | |
| (no ID) | Shinn et al. 2023 — Reflexion: language agents with verbal RL, NeurIPS 2023 (2026-02-03 Notion novelty check; no arXiv ID given in source) | `nl-bottleneck-prior-art` (2026-08-24 entry) | agent-supplied (ChatGPT) | TLC | |
| (no ID) | Yuan et al. 2025 — enhancing queries for code generation with RL, Sci. Reports 15:37300 (2026-02-03 Notion novelty check; Nature DOI s41598-025-21271-4, no arXiv) | `nl-bottleneck-prior-art` (2026-08-24 entry) | agent-supplied (ChatGPT) | TLC | |
| 2405.17052 | Gao et al. 2024 — SelfCP: compressing over-limit prompts via the frozen LLM itself (2026-02-04 Consensus novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Consensus entry) | agent-supplied (Consensus) | TLC | |
| 2310.05736 | Jiang et al. 2023 — LLMLingua: prompt compression, EMNLP (2026-02-04 Consensus novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Consensus entry) | agent-supplied (Consensus) | TLC | |
| 2409.08561 | Liu et al. 2024 — Hidden CoT: compressed chain-of-thought decoding (2026-02-04 Consensus novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Consensus entry) | agent-supplied (Consensus) | TLC | |
| 2405.20132 | Stein & Bäck 2024 — LLaMEA: LLM evolutionary algorithm for metaheuristics, IEEE TEVC (2026-02-04 Consensus novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Consensus entry) | agent-supplied (Consensus) | TLC | |
| 2310.10698 | Ma et al. 2023 — SCoT: semantic chain-of-thought prompting for code generation (2026-02-04 Consensus novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Consensus entry) | agent-supplied (Consensus) | TLC | |
| (no ID) | Romera-Paredes et al. 2023 — FunSearch, Nature, DOI 10.1038/s41586-023-06924-6 (2026-02-04 Consensus novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Consensus entry) | agent-supplied (Consensus) | TLC | |
| (no ID) | Zhang et al. 2025 — transformer code model with compressed hierarchy representation, EMSE 30:60, DOI 10.1007/s10664-025-10612-6 (2026-02-04 Consensus novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Consensus entry) | agent-supplied (Consensus) | TLC | |
| 2506.04019 | Oza et al. 2025 — CETBench: code-equivalence checking benchmark via program transformations (Consensus review surveys toggle) | `nl-bottleneck-prior-art` (2026-08-24 Consensus entry, standout) | agent-supplied (Consensus) | TLC (equivalence) | |
| 2410.03351 | Li et al. 2024 — generating equivalent representations of code by self-reflection (Consensus review surveys toggle) | `nl-bottleneck-prior-art` (2026-08-24 Consensus entry, standout) | agent-supplied (Consensus) | TLC (equivalence) | |
| 2309.08532 | Guo et al. 2023 — EvoPrompt: evolutionary prompt optimization (2026-02-04 Claude novelty check; distinct from EvoPrompting 2302.14838) | `nl-bottleneck-prior-art` + `prompt-optimization-landscape` (2026-08-24 Claude entries) | agent-supplied (Claude) | TLC | |
| 2309.03409 | Yang et al. 2023 — OPRO: LLMs as optimizers (2026-02-04 Claude novelty check) | `prompt-optimization-landscape` (already named there; ID row for completeness) | agent-supplied (Claude) | TLC | |
| 2207.10397 | Chen et al. 2022 — CodeT: code generation with generated tests, ICLR 2023 (2026-02-04 Claude novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Claude entry) | agent-supplied (Claude) | TLC | |
| 2502.13063 | Kuratov et al. 2025 — Proto-tokens: cramming 1568 tokens into a single vector, ACL 2025 (2026-02-04 Claude novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Claude entry; NL-necessity relevance) | agent-supplied (Claude) | TLC (representation) | |
| 2108.07129 | de Bruin et al. 2021 — autoencoders as tools for program synthesis: VAE + gradient-free evolutionary search (2026-02-04 Claude novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Claude entry) | agent-supplied (Claude) | TLC | |
| 2303.06689 | Jiang et al. 2023 — Self-Planning code generation with LLMs, ASE 2024 (2026-02-04 Claude novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Claude entry) | agent-supplied (Claude) | TLC | |
| 2305.10601 | Yao et al. 2023 — Tree of Thoughts, NeurIPS 2023 (2026-02-04 Claude novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Claude entry) | agent-supplied (Claude) | TLC | |
| 2205.12548 | Deng et al. 2022 — RLPrompt: discrete text prompt optimization with RL (2026-02-04 Gemini novelty check; supplies the ID for row A's bare "RLPrompt") | `nl-bottleneck-prior-art` (2026-08-24 Gemini entry); litreview plan row A | agent-supplied (Gemini) | TLC | |
| 2402.03300 | Shao et al. 2024 — GRPO / DeepSeekMath: group relative policy optimization (2026-02-04 Gemini novelty check; LBM's optimizer) | `nl-bottleneck-prior-art` (2026-08-24 Gemini entry) | agent-supplied (Gemini) | TLC | |
| 2306.17842 | Yu et al. 2023 — SPAE: semantic pyramid autoencoder, frozen-LLM lexical-token latents for images (2026-02-04 Gemini novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Gemini entry; NL-likeness rung) | agent-supplied (Gemini) | TLC (representation) | |
| 2404.09066 | Noah et al. 2024 — CodeCloak: DRL prompt manipulation against code leakage (2026-02-04 Gemini novelty check; adversarial dual) | `nl-bottleneck-prior-art` (2026-08-24 Gemini entry; leakage thread) | agent-supplied (Gemini) | TLC | |
| (no ID) | LINT — assessing interpretability of programmatic policies with LLMs, explain→regenerate as a score (2026-02-04 Gemini novelty check; no arXiv ID in source) | `nl-bottleneck-prior-art` (2026-08-24 Gemini entry) | agent-supplied (Gemini) | TLC (TLC-0 metric) | |
| 2509.19228 | Berton et al. 2025 — CompLLM: compression for long-context QA (2026-02-04 Gemini novelty check) | `nl-bottleneck-prior-art` (2026-08-24 Gemini entry) | agent-supplied (Gemini) | TLC | |
| 2506.01084 | Geng et al. 2025 — Zip2Zip: inference-time adaptive tokenization via online compression (2026-02-04 Gemini novelty check sources) | `nl-bottleneck-prior-art` (2026-08-24 Gemini entry) | agent-supplied (Gemini) | TLC | |
| 2510.09062 | Sun et al. 2025 — ReFIne: trustworthy large reasoning models (2026-02-04 Gemini novelty check sources) | `nl-bottleneck-prior-art` (2026-08-24 Gemini entry) | agent-supplied (Gemini) | TLC | |
| 2310.19660 | Ludan et al. 2023 — TBM: text bottleneck models for interpretable classification (2026-02-04 Perplexity check sources, recovered 2026-08-24) | `nl-bottleneck-prior-art` (Perplexity entry) | agent-supplied (Perplexity) | TLC | |
| 2501.09747 | Pertsch et al. 2025 — FAST: efficient action tokenization for VLA models (2026-02-04 Perplexity check sources, recovered 2026-08-24) | `nl-bottleneck-prior-art` (Perplexity entry) | agent-supplied (Perplexity) | TLC | |
| 2109.00055 | Montero et al. 2021 — sentence bottleneck autoencoders from transformer LMs, EMNLP (row-A item; ID confirmed by 2026-02-04 Perplexity check sources) | `nl-bottleneck-prior-art`; litreview plan row A | agent-supplied | TLC | |
| 2406.01989 | UNRESOLVED PAIRING — the 2026-02-04 Perplexity check's sources link this ID to the Semantic-Compression page, conflicting with Gilbert = 2304.12512 (submitted bibliography); resolve which paper 2406.01989 actually is at verification | `nl-bottleneck-prior-art` (Perplexity entry, flag) | agent-supplied (Perplexity) | TLC | ID conflict |
| 2310.06201 | Li et al. 2023 — Selective Context: self-information-based content filtering (2026-02-04 Undermind check, timeline era 1) | `prompt-compression-and-optimization-literature` (2026-08-24 Undermind entry) | agent-supplied (Undermind) | TLC | |
| 2310.06839 | Jiang et al. 2023 — LongLLMLingua: long-context prompt compression (2026-02-04 Undermind check) | `prompt-compression-and-optimization-literature` (2026-08-24 Undermind entry) | agent-supplied (Undermind) | TLC | |
| 2312.09571 | Fei et al. 2024 — extending context window via semantic compression, ACL Findings (2026-02-04 Undermind check) | `prompt-compression-and-optimization-literature` (2026-08-24 Undermind entry) | agent-supplied (Undermind) | TLC | |
| 1810.09717 | SAPS 2018 — "Ain't Nobody Got Time For Coding": structure-aware program synthesis from NL, tree-to-tree autoencoder (SciSpace summary table, 1.5/6; ID recovered from the Dec-2025 supplemental 2026-08-24) | `nl-bottleneck-prior-art` (table-layer entry) | agent-supplied (NBLM/SciSpace) | TLC | |
| 2308.08758 | Jung & Kim 2023 — PCRL: discrete prompt compression with RL (ID recovered from the Dec-2025 SciSpace supplemental; fills row A's bare "PCRL") | `nl-bottleneck-prior-art`; litreview plan row A | agent-supplied (SciSpace) | TLC | |
| (no ID) | MIPS — opening the AI black box: program synthesis via mechanistic interpretability (Dec-2025 supplemental category C; no ID in source) | `nl-bottleneck-prior-art` (Dec-2025 addendum) | agent-supplied (SciSpace) | TLC | |
| (no ID) | APICoder — private-library-oriented code generation with LLMs (Dec-2025 supplemental; no ID in source) | `nl-bottleneck-prior-art` (Dec-2025 addendum) | agent-supplied (SciSpace) | TLC | |
| (no ID) | SAMMO — symbolic prompt program search, structure-aware compile-time prompt optimization (Dec-2025 supplemental; no ID in source) | `nl-bottleneck-prior-art` (Dec-2025 addendum) | agent-supplied (SciSpace) | TLC | |
| (no ID) | More Than a Score / PartialOrderEval — prompt-specificity ladder on HumanEval vs ParEval | TLC §4 + §5; `code-benchmarks-landscape` (2026-08-24) | Danielle-supplied (April-2026 notes; some characterizations quote an assistant) | TLC | |
| (no ID) | Exploring LM Code Generation with Auxiliary Functions / HumanExtension — oracle-context upper bound | `code-benchmarks-landscape` (2026-08-24) | Danielle-supplied (April-2026 notes; some characterizations quote an assistant) | TLC | |
| (no ID) | PromptBridge — cross-model prompt transfer, model drifting | TLC §4 + §5; `prompt-optimization-landscape` (2026-08-24) | Danielle-supplied (April-2026 notes; some characterizations quote an assistant) | TLC | |
| (no ID) | Discrete GA prompt optimization for secure Python code, JSS 232:112682 (2026) — non-transferability | `prompt-optimization-landscape` (2026-08-24) | Danielle-supplied (April-2026 notes; some characterizations quote an assistant) | TLC | |
| (no ID) | Tuning LLM-based Code Optimization via Meta-Prompting (industrial) — non-transferability | `prompt-optimization-landscape` (2026-08-24) | Danielle-supplied (April-2026 notes; some characterizations quote an assistant) | TLC | |
| (no ID) | Guidelines to Prompt LLMs for Code Generation — 10 empirical guidelines | `prompt-optimization-landscape` (2026-08-24) | Danielle-supplied (April-2026 notes; some characterizations quote an assistant) | TLC | |
| (no ID) | LM-CC / Rethinking Code Complexity Through the Lens of LLMs — rewrites lowering LM-CC improve pass@1 | TLC §4 + §5; `code-compression-literature` (2026-08-24) | Danielle-supplied (April-2026 notes; some characterizations quote an assistant) | TLC | |
| (no ID) | How do Humans and LLMs Process Confusing Code? — atoms of confusion, perplexity spikes | `code-compression-literature` (2026-08-24) | Danielle-supplied (April-2026 notes; some characterizations quote an assistant) | TLC | |
| (no ID) | LongCodeZip — code-RAG surface-form critique; compression beats uncompressed on RepoQA | TLC §4; `code-compression-literature` (2026-08-24) | Danielle-supplied (April-2026 notes; some characterizations quote an assistant) | TLC | |
| (no ID) | TRAAC / Think Right — adaptive attentive compression outer loop | `code-compression-literature` (2026-08-24) | Danielle-supplied (April-2026 notes; some characterizations quote an assistant) | TLC | |
| (no ID) | Fault in our Stars — benchmark prompt-quality audit (3,566 prompts / 9 benchmarks) | `code-benchmarks-landscape` (2026-08-24) | Danielle-supplied (April-2026 notes; some characterizations quote an assistant) | TLC | |
| (no ID) | Non-determinism of ChatGPT in Code Generation — temp 0 not deterministic (829 problems) | `code-benchmarks-landscape` (2026-08-24) | Danielle-supplied (April-2026 notes; some characterizations quote an assistant) | TLC | |
| 2503.05860 | HumanEvalNext / BenchFrame — corrected + hardened HumanEval (Koohestani et al.) | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC/ELI/IRT |  |
| 2210.14868 | MBXP / Multilingual HumanEval (Athiwaratkun et al., Amazon) | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC/ELI/IRT |  |
| 2410.09812 | PolyHumanEval — 14 languages, 110k+ translations | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC/ELI/IRT |  |
| 2410.15037 | mHumanEval — docstrings in 200+ NLs, 15 expert-validated | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC/ELI/IRT |  |
| 2502.19149 | PseudoEval — pseudocode replaces docstring; problem-solving vs language-coding split | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC |  |
| 2401.03065 | CRUXEval — input/output prediction on 800 short functions; CRUXEval-X to 19 PLs | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC (task families) | fills the no-ID CRUXEval mentions in plan row D and the TLC recall corpus |
| 2502.07316 | CodeI/O — I/O prediction as reasoning training signal | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC/ELI |  |
| 2306.03438 | Buggy-HumanEval — injected semantics-altering bugs in context | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC |  |
| 2411.07781 | RedCode — HumanEval-format prompts eliciting risky code | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | — |  |
| 2211.15395 | CodeExp — explanatory docstring generation, human-correlated metrics | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC |  |
| 2311.04850 | Yang et al. 2023 — Rethinking Benchmark and Contamination (8–18% HumanEval overlap in RedPajama/StarCoder-Data) | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC/ELI/IRT |  |
| 2308.01861 | ClassEval — 100 class-level Python tasks, ~500 person-hours of tests; named clearest fit for compression eval | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC (task suite) |  |
| 2306.11644 | phi-1 / Textbooks Are All You Need | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | — |  |
| 2503.02951 | KodCode — synthetic, unit-test-verified training set | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | — |  |
| 2105.09938 | APPS — 10k competitive-programming problems, judge tests | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC (task suite) |  |
| 2310.06770 | SWE-bench — 2,294 GitHub-issue tasks | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | — |  |
| 2507.12415 | SWE-Perf — performance-optimization SWE-bench variant | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | — |  |
| 2602.03712 | SWE-Refactor | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | — |  |
| 2406.15877 | BigCodeBench — 1,140 realistic library-use tasks (~5.6 tests/task per the convo) | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC (task suite) |  |
| 2102.04664 | CodeXGLUE — 10 tasks / 14 datasets SE suite | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | — |  |
| (no ID) | ContractEval (Lim et al. 2025) — contract-violation tests via SMT on HumanEval+; only a semanticscholar link | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC |  |
| (no ID) | SakuraEval — Japan-localized problems (IEEE 11396539) | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | — |  |
| (no ID) | REval — runtime-behavior reasoning wrapper, incremental consistency (IEEE 11029885) | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC |  |
| (no ID) | VRTG — variable-renaming adversarial attacks (ACM 10.1145/3723353) | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC |  |
| (no ID) | COFFE — CPU-instruction-count efficiency benchmark (ACM 10.1145/3715727) | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | divergence probe |  |
| (no ID) | DocStringEval — docstring generation for classes; BERTScore/Flesch/compression-ratio metrics (IEEE 11108633) | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC |  |
| (no ID) | Dainese et al. — docstring reformulation before solving; only a Semantic Scholar mention | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC |  |
| (no ID) | DiffMIA — contamination detection via rename-based pseudo-uncontaminated samples (IEEE 11396512) | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | — |  |
| (no ID) | ClassEval-T — ClassEval code-translation extension | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | — |  |
| (no ID) | OOP-Eval — 431 object-oriented patterns | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | — |  |
| (no ID) | DynaCode — MBPP+ atoms composed into nested call graphs (16 topologies, 189M problems); closest prior to the composition idea | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC (composition) |  |
| (no ID) | Fundamental Limits of Prompt Compression — black-box prompt compression as rate-distortion | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC (theory) |  |
| (no ID) | Extending Context Window via Semantic Compression — source-coding view | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC (theory) |  |
| (no ID) | CoderEval — 230 Py + 230 Java tasks from real projects; class-level subset suggested as ClassEval supplement | `code-benchmarks-landscape` (2026-08-24 Perplexity entry) | agent-supplied (Perplexity, April convo) | TLC (task suite) |  |

<!-- External report produced by a browsing deep-search assistant for Danielle on 2026-08-16 (MAQA Next Steps page); copied verbatim 2026-08-22. Citations unverified here. -->

# Exhaustive Multi-Answer Question Answering over Unstructured Corpora

## What followed QAMPARI, QUEST, and RoMQA—and what frontier language models can actually do

**Research report — August 2026**

## Executive summary

The research line is active, but it has split into three partially overlapping problems:

1. **Exhaustive list answering:** return all entities satisfying a question (QAMPARI, RoMQA, and now MoNaCo).
2. **Set-compositional retrieval:** retrieve every entity or document satisfying unions, intersections, negations, or filters (QUEST and recent controlled successors such as LIMIT+).
3. **Broad or fan-out research:** gather facts from many pages and synthesize a structured answer (FanOutQA, TANQ, MoNaCo, and WideSearch).

The central result is not that frontier models have solved the original problem. Closed-book models have improved substantially, but **recall remains the dominant failure mode**, especially as the answer set grows. Retrieval can help considerably, but only when it retrieves complementary evidence and the reader verifies it across passages. Naive retrieval can actually reduce performance by flooding the context with distractors. Oracle evidence produces a large gain across datasets, showing that both retrieval coverage and evidence-conditioned reasoning remain open problems.

A particularly clean recent comparison is [RI2VER (Findings of ACL 2025)](https://aclanthology.org/2025.findings-acl.354/). On QAMPARI, GPT-4o closed-book obtained macro answer F1 **24.59**, whereas a Llama-3.1-70B retrieval-and-verification system obtained **40.70**. On RoMQA the corresponding figures were **12.20** and **19.24**. These are not universal model rankings, but within one evaluation they demonstrate that explicit corpus access plus inter-passage verification materially outperforms parametric memory.

The newer [MoNaCo benchmark](https://arxiv.org/abs/2508.11133) supplies the clearest scale analysis. Its intermediate list questions show GPT-4o recall falling from roughly **61–66%** for lists of 2–20 items to **27.6%** for 101–500 items and **2.5%** above 500 items. On full MoNaCo questions, GPT-4o scored **48.98 F1** closed-book, **37.28** with naive top-20 BM25 retrieval, and **58.67** with oracle evidence. This captures the present state unusually well: models know a great deal, indiscriminate RAG is not enough, and finding the right evidence still leaves substantial reasoning work.

There is no single current leaderboard that cleanly answers “How good are frontier LMs at exhaustive multi-answer Wikipedia QA?” Results are fragmented across different tasks, answer caps, evidence settings, and metrics. A rigorous new comparison should therefore include a closed-book arm, fixed-corpus retrieval, oracle evidence, and preferably a structured-KB oracle, while measuring complete-set success rather than capped recall.

## 1. Defining the problem

The target task can be written as set prediction. Given a question \(q\) and a corpus \(C\), return the set

\[
A(q) = \{e : e\text{ satisfies all constraints in }q\}.
\]

This differs from ordinary open-domain QA in three important ways:

- **Coverage matters:** producing one correct entity is not enough.
- **Evidence is distributed:** different answers are often supported by different pages or passages.
- **Constraints interact:** union, intersection, exclusion, comparison, time, aggregation, and latent typing can determine membership in the answer set.

It is also useful to distinguish four evidence conditions:

| Condition | What the model receives | What it measures |
|---|---|---|
| Closed-book | Question only | Parametric memory plus reasoning; not “no external knowledge,” because Wikipedia may be encoded in training weights |
| Retrieved corpus | Question plus passages found from a fixed corpus | Retrieval coverage, context management, verification, and generation |
| Oracle evidence | Question plus gold supporting evidence | Reader/reasoner upper bound under corpus access |
| Structured-KB oracle | Queryable canonical entities and relations | Cost of deriving the answer from a knowledge base rather than raw text |

These conditions should not be conflated. In particular, an old Wikipedia benchmark is a weak test of pure closed-book generalization because both its source pages and possibly the benchmark itself may occur in model training data.

## 2. The three starting datasets

| Dataset | Core task | Scale and evidence | Original findings | Important limitation |
|---|---|---|---|---|
| [QAMPARI](https://aclanthology.org/2023.gem-1.9/) (GeM 2023) | Generate a list of entities, with evidence distributed across multiple Wikipedia paragraphs | At least 5 answers/question, about 13 on average; 2,000 dev/test examples plus more than 60,000 training examples; Aug. 2021 Wikipedia | Best original trained retrieval/generation system: 32.8 F1. Zero-shot text-davinci-003 closed-book: 13.8 F1. Reader over top-15 BM25 passages: 18.8 F1. Oracle passage selection still left a large gap | Gold answer sets can be incomplete; many later papers cap recall at five answers (“F1-5”), weakening the exhaustive requirement |
| [QUEST](https://aclanthology.org/2023.acl-long.784/) (ACL 2023) | Retrieve the entities/documents satisfying implicit set operations | 3,357 natural entity-seeking queries; union, intersection, and difference; Wikipedia document-level evidence; up to 20 target entities | T5-Large dual encoder substantially beat BM25, but mean complete recall at 100 remained only 0.142 | Primarily a retrieval/entity-set classification benchmark, not an end-to-end natural-language list-generation benchmark |
| [RoMQA](https://aclanthology.org/2023.findings-emnlp.470/) (Findings EMNLP 2023) | Answer related questions whose constraints vary; measure both average and worst-case cluster performance | Multi-answer, multi-evidence questions mined from Wikidata with Wikipedia/T-REx evidence | In the 100-candidate setting, supervised BART plus retrieval reached 63.8 F1 but only 37.9 robust F1; gold evidence reached 95.0/83.4. In open generation, GPT-3 few-shot reached 4.4 F1 and 0.4 robust F1 | Candidate and open-generation settings are different; open evaluation uses P@10 in part because some answer sets are very large |

Together, these datasets exposed three bottlenecks that still organize the field: retrieving *all* relevant evidence rather than redundant passages, applying set constraints reliably, and generating a complete set without hallucinated additions.

## 3. How the line continued

### 3.1 Retrieval diversified from relevance ranking to coverage and verification

Standard dense retrieval tends to return many passages supporting the same salient answer. Earlier work such as [Joint Passage Ranking](https://aclanthology.org/2021.emnlp-main.560/) explicitly reranked passages to cover new answers. Later work made the feedback loop more model-driven:

- [LLatrieval](https://aclanthology.org/2024.naacl-long.305/) uses an LLM to verify whether retrieved passages sufficiently support a response and iteratively retrieves more. On the citation-grounded ALCE version of QAMPARI, it improved answer correctness and citation quality over BM25 and RankGPT. This is important progress, but the ALCE protocol uses recall capped at five answers, so it should not be read as full-list coverage.
- [RI2VER](https://aclanthology.org/2025.findings-acl.354/) performs inter-passage verification, reasoning over accumulated evidence and resolving inconsistencies. It directly evaluates QAMPARI and RoMQA and shows clear gains over independent passage processing, simple concatenation, and closed-book baselines.
- [RVR: Retrieve-Verify-Retrieve for Comprehensive Question Answering](https://arxiv.org/abs/2602.18425) conditions each retrieval round on evidence already verified as relevant. On a 25.9-million-passage Aug. 2021 Wikipedia index, its best QAMPARI result reached mean complete recall at 100 of **33.70%** and ordinary Recall@100 of **68.70%**. On zero-shot QUEST it reached **6.02%** complete recall and **30.53%** ordinary recall. Thus, even a strong 2026 retrieval method commonly finds some answers while failing to retrieve evidence for the entire set.

The shift is conceptually important. The objective is no longer merely “rank passages relevant to the question”; it is “select a small set of passages whose *union* covers the answer set, while continually checking which constraints and answers remain unresolved.”

### 3.2 Closed-book prompting produced only modest gains on the original benchmarks

[Crafting in-context examples according to a model’s parametric knowledge](https://aclanthology.org/2024.findings-naacl.133/) repurposed QAMPARI and QUEST as closed-book multi-answer generation tasks. Selecting demonstrations based on what the model appeared to know gave small improvements, but absolute exact-match F1 remained low: approximately **15–16** on QAMPARI and **6** on QUEST for GPT-3.5 under the reported five-example setup. This is useful evidence that prompt selection alone does not overcome missing coverage.

These results are now technologically dated, but they establish a baseline pattern that continues in newer models: precision is often respectable while recall is poor, and prompting changes the margin more than the basic capability.

### 3.3 Long context does not simply subsume retrieval

[LOFT: Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?](https://aclanthology.org/2025.findings-naacl.374/) placed corpus subsets directly into long model contexts. At 128K context, multi-target retrieval mean recall was **0.61** on QAMPARI for Gemini 1.5 Pro, compared with **0.57** for the specialized retrieval baseline; however, on QUEST the corresponding figures were **0.30** versus **0.54**. In its answer-generation setting, specialized systems remained ahead on both QAMPARI and QUEST.

The study shows that a long-context model can search a modest provided corpus surprisingly well, but it does not establish that a model can absorb full Wikipedia or eliminate corpus indexing. The benchmark construction also caps the relevant documents at five for QAMPARI and three for QUEST, so it is easier than unbounded exhaustive retrieval.

### 3.4 QUEST became a test case for whether dense retrievers understand logic

Work such as [Does Dense Retrieval Understand Boolean Logic?](https://aclanthology.org/2024.findings-emnlp.156/) and [LOGICOL](https://aclanthology.org/2025.emnlp-main.608/) directly targeted conjunction, disjunction, and negation. LOGICOL’s logically informed contrastive training improved QUEST results.

The most informative recent check is [Reproducing Complex Set-Compositional Information Retrieval](https://arxiv.org/abs/2605.03824) (SIGIR 2026). It compares 12 retrieval models and four reasoning-oriented methods on QUEST, controlled QUEST variants, and the new LIMIT+ benchmark. Strong neural retrievers exceed roughly **0.41 Recall@100** on QUEST versus about **0.20** for BM25. Yet a leading logic-aware result of about **0.42** on QUEST falls below **0.02** on LIMIT+, where lexical retrieval is near **0.96** and dense single-vector methods remain below **0.10**. Performance also worsens with compositional depth.

The authors’ controlled tests support a sobering interpretation: some apparent logical competence on QUEST comes from semantic or lexical shortcuts rather than dependable satisfaction of set constraints. Generic agentic search methods are not uniformly superior either.

## 4. Direct evidence: closed-book versus corpus access

The following comparisons are internally valid within each row, but not across rows: datasets, metrics, model families, and evidence budgets differ.

| Study and task | Closed-book | Retrieved corpus | Oracle evidence | Interpretation |
|---|---:|---:|---:|---|
| QAMPARI original (2023), macro answer F1 | text-davinci-003: **13.8** | BM25 reader: **18.8**; best trained PIG-DPR: **32.8** | PIG oracle on dev: **62.4** | Retrieval helps; the oracle gap indicates severe retrieval/selection error plus remaining reader error |
| QAMPARI in RI2VER (2025), macro answer F1 | GPT-4o: **24.59** | RI2VER, Llama-3.1-70B: **40.70** | — | Inter-passage verification gives a large gain over a strong closed-book model |
| RoMQA in RI2VER (2025), macro answer F1 | GPT-4o: **12.20** | RI2VER, Llama-3.1-70B: **19.24** | — | Corpus access helps, but the absolute result remains low under robust multi-constraint answering |
| MoNaCo (2025), answer F1 | GPT-4o: **48.98** | BM25 top-20: **37.28** | All gold evidence: **58.67** | Naive RAG hurts; high-quality evidence helps by about ten F1 points over closed-book |

Two conclusions survive the protocol differences:

1. **The best evidence for corpus value is the oracle gap, not “RAG” as a label.** Gold evidence consistently improves results. A poor retriever or excessive context can obscure that benefit.
2. **Closed-book scores are not a clean lower bound on reasoning.** They combine memorized Wikipedia facts, possible benchmark contamination, implicit retrieval from parameters, and actual constraint reasoning.

## 5. Important successors and adjacent benchmarks

### MoNaCo: natural, highly distributed multi-step questions

[MoNaCo: More Natural and Complex Questions for Reasoning Across Dozens of Documents](https://arxiv.org/abs/2508.11133) is the closest current successor if the aim is natural QA over raw Wikipedia rather than templated list lookup. It contains 1,315 human-written, manually decomposed questions. A full question requires evidence from **43.3 unique Wikipedia pages on average**, compared with about 13 for QAMPARI and 10.5 for QUEST in the paper’s comparison. Its decompositions contain 8,549 intermediate list questions, averaging 16.2 answers, and include filtering, aggregation, arithmetic, and comparison.

On full questions, o3 reached **61.18 F1**, GPT-5 **60.11**, Gemini 2.5 Pro **59.11**, and Claude 4 Opus **55.03** under the paper’s closed-book evaluation. Despite those much stronger numbers than earlier model generations, the task is not solved: the best model produced a fully correct answer on only **38.7%** of questions. More importantly for list QA, recall collapses with set size, reaching 2.5% for intermediate lists above 500 answers.

MoNaCo is stronger evidence about current models than the old datasets because its questions were newly written and held back during evaluation. Its underlying facts still come from Wikipedia and may have been memorized, so it is not contamination-proof.

### FanOutQA: breadth across multiple pages

[FanOutQA](https://aclanthology.org/2024.acl-short.2/) contains 1,034 human-written questions, 7,305 decompositions, and evidence from 4,121 Wikipedia articles. Each question requires at least five articles. It explicitly compares closed-book, open-book, and evidence-provided settings. Contemporary models were below 50% under the best closed/open settings, while open-book humans reached about 85%. Several models became worse when supplied large evidence contexts, reinforcing the context-overload result.

FanOutQA is directly relevant to multi-document breadth, although its final answers are not always exhaustive entity sets.

### TANQ: lists become attributed tables

[TANQ](https://aclanthology.org/2025.tacl-1.23/) extends QAMPARI-style questions into tables: rows are answer entities, columns are requested attributes, and every cell should be traceable to evidence. Its 1,395 entries average 6.7 rows and four columns and include filtering, composition, and intersection. The final paper reports a best baseline F1 of **60.7**, still 12.3 points below human performance.

TANQ is valuable when the practical output is not merely a list of names but a research table with provenance.

### WideSearch: real web-scale exhaustive research

[WideSearch](https://arxiv.org/abs/2508.07999) is a 200-task English/Chinese benchmark of broad web research whose outputs are structured tables. It deliberately filters out tasks solvable from parametric memory alone and requires external tools. At ICLR 2026, most evaluated agents had success rates near zero and the best was about **5%**, while humans approached full completion with enough time.

WideSearch is not fixed-corpus Wikipedia QA, but it is perhaps the strongest present demonstration that frontier web agents do not yet reliably execute exhaustive, high-recall research.

### Other nearby work

- [ALCE](https://arxiv.org/abs/2305.14627) popularized citation-grounded generation and repurposed QAMPARI, but its five-answer recall cap changes the target.
- [AmbigQA](https://aclanthology.org/2020.emnlp-main.466/) produces multiple question–answer pairs for ambiguous questions; multiplicity comes from ambiguity rather than an exhaustive set constraint.
- [GRANOLA-QA](https://aclanthology.org/2024.acl-long.365/) studies valid answers at multiple granularities, not list completeness.
- [MulTiple](https://openreview.net/forum?id=qvxjSXiBlLF) contributes 17,580 time-sensitive multi-answer instances, but is more temporal and knowledge-base-oriented than the raw-Wikipedia setting.

## 6. What can be claimed about frontier models today?

### Supported claims

- **They are much stronger closed-book than the GPT-3-era baselines.** MoNaCo’s 2025–26 model comparison is the best broad evidence.
- **They often know correct items but stop too early.** Across MoNaCo’s set-size analysis and earlier QAMPARI work, recall degrades much faster than precision.
- **Corpus access can produce large gains.** RI2VER supplies a controlled direct demonstration on QAMPARI and RoMQA.
- **Retrieval quality is decisive.** MoNaCo’s BM25-versus-oracle result shows that “adding RAG” is not itself sufficient.
- **Logical/set generalization remains fragile.** LIMIT+ reveals collapses hidden by ordinary QUEST performance.
- **Real exhaustive research remains far from solved.** WideSearch’s low agent success rates and RVR’s low complete-recall numbers are strong evidence.

### Claims that the literature does not yet support

- That a current frontier model can reliably enumerate the full answer set from memory.
- That long context makes retrieval systems unnecessary at Wikipedia scale.
- That a good score on QAMPARI F1-5 demonstrates exhaustive list answering.
- That higher ordinary Recall@K implies evidence for every answer was retrieved.
- That closed-book success on old Wikipedia questions is uncontaminated generalization.
- That an overall F1 from one of these datasets can be directly ranked against an F1 from another.

## 7. Evaluation hazards

### Answer-set incompleteness

Wikipedia and Wikidata-derived gold sets are rarely perfect. A model may be penalized for a genuinely correct entity absent from the annotations. QAMPARI manually extended a subset of answer sets and found system rankings relatively stable, which supports benchmark usefulness, but precision values remain less secure than recall against a fixed annotated set.

### Metric drift

The phrase “QAMPARI result” can refer to at least two materially different metrics:

- original macro set precision/recall/F1 over the annotated list; or
- ALCE-style F1-5, where recall credit is capped after five correct answers.

Similarly, P@10 on RoMQA and capped relevant documents in LOFT do not test full-set recovery. Papers should be compared only after inspecting the precise evaluation script.

### Contamination and changing model knowledge

A question-only evaluation against 2021 Wikipedia measures parametric memory, but it cannot identify how the knowledge entered the model. The corpus may be in pretraining; benchmark examples may be indexed online; and proprietary training data are undisclosed. Fresh question writing, private tests, post-cutoff facts, and dated corpus snapshots reduce—without entirely eliminating—this ambiguity.

### Retrieval and generation are entangled

End-to-end F1 cannot say whether a missed entity was never retrieved, was retrieved but ignored, or was generated and rejected. Retrieval coverage, reader coverage conditional on gold evidence, and final set accuracy should be reported separately.

## 8. Recommended rigorous study design

For a modern comparison that would be persuasive to both retrieval and LLM researchers, use a hidden, freshly authored evaluation set over a frozen, dated Wikipedia corpus and evaluate the same frontier models in the following arms:

1. **Closed-book:** question only.
2. **Fixed-corpus RAG:** identical model plus a reproducible index and retrieval budget.
3. **Coverage-aware RAG:** iterative retrieval that conditions on already verified entities/evidence.
4. **Oracle evidence:** all gold passages, with a clearly stated context budget.
5. **Optional structured-KB oracle:** execute a verified symbolic query to establish the corpus-to-KB gap.

The primary measures should be:

- macro answer-set precision, recall, and F1;
- **complete-set accuracy** and complete-recall@K/MRecall@K;
- evidence recall and citation entailment for each answer;
- worst-case performance over related constraint variants, following RoMQA;
- abstention/calibration, latency, tokens, retrieval calls, and monetary cost.

Stratify results by answer-set size, number of evidence pages, operator type (union/intersection/difference), compositional depth, temporal freshness, and entity popularity. Publish both an exact canonical metric and an expanded evaluation that accepts aliases and adjudicated extra answers. Do not cap recall at five if exhaustiveness is the research question.

A useful benchmark suite would retain the original three datasets for historical continuity while adding:

- **MoNaCo** for natural, many-page, multi-step questions;
- **LIMIT+** for controlled logical generalization;
- **TANQ** if structured attributes and provenance matter;
- **WideSearch** as an external-validity stress test for real broad research.

## 9. Bottom line

The original intuition behind QAMPARI, QUEST, and RoMQA remains well founded: multi-answer QA is primarily a **coverage problem under constraints**, not merely ordinary QA repeated several times. Modern LMs have improved parametric knowledge and can synthesize many scattered facts, but no rigorous body of evidence shows reliable exhaustive performance without a corpus. With a corpus, specialized retrieval, iterative verification, and oracle evidence all help, yet current systems still miss substantial fractions of the answer set.

The most defensible contemporary conclusion is therefore:

> Frontier models are useful components for exhaustive multi-answer QA, but neither closed-book generation nor naive RAG is a solved system. High-recall evidence acquisition, constraint-faithful verification, and completeness-aware evaluation remain the critical research problems.

## Annotated primary-source bibliography

- Amouyal et al. [QAMPARI: An Open-domain Question Answering Benchmark for Questions with Many Answers from Multiple Paragraphs](https://aclanthology.org/2023.gem-1.9/), GeM 2023.
- Malaviya et al. [QUEST: A Retrieval Dataset of Entity-Seeking Queries with Implicit Set Operations](https://aclanthology.org/2023.acl-long.784/), ACL 2023.
- Zhong et al. [RoMQA: A Benchmark for Robust, Multi-evidence, Multi-answer Question Answering](https://aclanthology.org/2023.findings-emnlp.470/), Findings of EMNLP 2023.
- Gao et al. [Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627) (ALCE benchmark and QAMPARI adaptation), EMNLP 2023.
- Li et al. [LLatrieval: LLM-Verified Retrieval for Verifiable Generation](https://aclanthology.org/2024.naacl-long.305/), NAACL 2024.
- Mallen et al. [Crafting In-context Examples according to LMs’ Parametric Knowledge](https://aclanthology.org/2024.findings-naacl.133/), Findings of NAACL 2024.
- Zhu et al. [FanOutQA: A Multi-Hop, Multi-Document Question Answering Benchmark for Large Language Models](https://aclanthology.org/2024.acl-short.2/), ACL 2024.
- Pi et al. [Does Dense Retrieval Understand Boolean Logic?](https://aclanthology.org/2024.findings-emnlp.156/), Findings of EMNLP 2024.
- Lee et al. [Can Long-Context Language Models Subsume Retrieval, RAG, SQL, and More?](https://aclanthology.org/2025.findings-naacl.374/) (LOFT), Findings of NAACL 2025.
- Dhole et al. [Inter-Passage Verification for Open-Domain Question Answering](https://aclanthology.org/2025.findings-acl.354/) (RI2VER), Findings of ACL 2025.
- Zhao et al. [LOGICOL: Logically Informed Contrastive Learning for Set-Compositional Retrieval](https://aclanthology.org/2025.emnlp-main.608/), EMNLP 2025.
- Wang et al. [TANQ: An Open Domain Dataset of Table Answered Questions](https://aclanthology.org/2025.tacl-1.23/), TACL 2025.
- Wolfson et al. [MoNaCo: More Natural and Complex Questions for Reasoning Across Dozens of Documents](https://arxiv.org/abs/2508.11133), TACL 2025.
- Shao et al. [WideSearch: Benchmarking Agentic Broad Info-Seeking](https://arxiv.org/abs/2508.07999), ICLR 2026.
- [RVR: Retrieve-Verify-Retrieve for Comprehensive Question Answering](https://arxiv.org/abs/2602.18425), 2026.
- [Reproducing Complex Set-Compositional Information Retrieval](https://arxiv.org/abs/2605.03824), SIGIR 2026.

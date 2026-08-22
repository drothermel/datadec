# Project-approach principles — Danielle's takeaways for starting new research projects

**Kind:** reference — a standing accumulator for Danielle's own methodology principles,
drawn from looking back at past projects, plus feedback received on them. Cross-project;
applied concretely in `../../potential-projs/maqa-brute-force-baseline.md`.

Sources: Danielle's statement on the "MAQA Next Steps" page (2026-08-17; intake 2026-08-22)
and the response to it; a 2025 (spring, inferred) analysis of two weeks of Heptabase
research notes (original notes lost; response only; intake 2026-08-22).
---

## 2026-08-17 — Four principles (Danielle, lightly edited from speech)

1. **Start by being clear about the problem definition, the shape of the solution space
   you're considering, and what is likely to have a large impact on success within that
   shape.** (Choose your problem → choose your solution space → list what will impact
   outcomes most.)
2. **Get an intuition for the problem first — the datasets and the high-leverage axes of
   your solution shape.** "Especially for something so heuristic: how do super simple
   baselines perform and why is an essential question." Understand the distribution of the
   important object types along the axes that drive cost, performance, or both: "long tails
   can tank a bad design, and naive truncation can tank performance."
3. **If the dataset is noisy in a way that might affect your solution, spend a little time
   (a) scoping the damage and possibly (b) making a clean set to iterate against.**
4. **It is often easier to start against a single dataset**, provided (a) systems are
   reproducible and (b) each experiment tells you what does or doesn't work *and why* on that
   dataset — then redesigning/extending to similar datasets is fairly direct.

## 2026-08-17 — Feedback received (near-verbatim)

- **On 1.** The "what impacts outcomes" list "is a hypothesis list" — annotate each
  hypothesis with the experiment that confirms or kills it, "so the intuition phase and the
  measurement phase are the same artifact." Promote eval fidelity into the impact list: "a
  metric that eats 10 points of true performance dominates most design decisions you'll
  make." Frame scoping commitments as *measured* decisions ("the X assumption caps recall at
  Y%, we accept that") rather than implicit ones.
- **On 2.** Make it concrete: "read 50–100 questions by hand before writing any code," and
  produce the key distribution plots early. "Manual reading is underrated; it's where you
  notice things like 'a third of these questions are actually ambiguous about inclusion
  criteria,' which no aggregate metric surfaces."
- **On 3.** "Dataset cleaning is a project that expands to fill available time. Time-box it,
  version it, make the cleaning pipeline deterministic and publishable, and always report
  official and cleaned numbers side by side" — iterating only on the cleaned metric drifts
  toward decisions that don't show on the official set, and reviewers compare on official
  numbers. The audit ("how much of the headline difficulty is dataset noise vs. genuine task
  difficulty") can be the most citable single result.
- **On 4.** Single-dataset iteration risks overfitting design to construction artifacts;
  cheap mitigation: keep a small frozen slice of a sibling dataset as a transfer smoke test
  run occasionally — "not to optimize against, just to notice when a design choice is
  dataset-specific." Clause (b) "each experiment tells you why" is load-bearing, and
  per-question loss attribution (which stage first made the question unwinnable) is the
  mechanism that makes it real — elevate it to an explicit principle.
- **Two candidate additions.** (5) **Decide kill criteria up front** — what measured result
  would make you abandon the solution shape entirely rather than patch it. (6) **Put cost on
  every plot from day one** — for a "how far can simple brute force go" thesis, the
  performance-vs-compute curve is the contribution, and retrofitting cost tracking is
  miserable.

## 2026-08-17 — Implementation time, upfront work, and over-indexing

**Danielle's additional takeaway.** She often gets stuck on implementation time. Setting
up multiple datasets is "a dramatic use of time because nothing ever actually works" —
different baselines, setups, quirks; "it's often hard to even tell what the papers are
measuring, let alone understanding it from the released data, even if they do release a
repo." Some work is worth doing upfront even though it delays the gap between first
discussing a project with collaborators and delivering the results they care about: "if I
had done [these things] first, I would have been able to give those results more
consistently, in a way that was more meaningful, and we would have gotten final outcomes
much faster." But limit ceremony (e.g. keeping slices from other datasets) because it
backfires. Quick, bounded versions are what she wants: a small hand-annotated subset (take a
baseline's highly relevant evidence, personally inspect the passages that aren't gold
evidence); an answer matcher doing "what I think are reasonable string normalizations."
Over-indexing on a dataset or on one's own dev set is possible, but "ultimately what we
want is to produce solutions that work on the actual task" — and "maybe the problem is that
we just need a better dataset."

**Feedback received (near-verbatim).**
- *Integration tax.* "Every dataset you touch costs a fixed integration tax (loaders,
  corpus alignment, metric reimplementation, figuring out what the paper actually
  measured), and you should pay that tax at most once per project phase, not
  speculatively." The earlier transfer-slice suggestion is withdrawn for the iteration
  phase — it only matters at paper time, when everything is re-run anyway.
- *Front-load findings, defer scaffolding.* "The ceiling measurements aren't infrastructure
  that delays results — they are results. 'Closed-book gets X, answer-vocab caps us at Y,
  BM25@200 covers Z% of answers, the metric eats W points on perturbed gold' is a genuinely
  interesting first meeting, arguably more interesting than a mediocre end-to-end F1. The
  upfront work that actually burns you is the kind that produces no numbers: abstractions,
  configurability, multi-dataset harnesses, 'clean' pipelines for experiments you haven't
  run yet. So the principle isn't 'spend more time upfront,' it's 'front-load the work that
  emits findings, defer the work that emits scaffolding.'"
- *Error-driven annotation.* Using the system's disagreements with gold as the sampling
  frame is far more efficient than random sampling. Asymmetry: hand-checking
  retrieved-but-not-gold evidence and predicted-but-marked-wrong answers measures gold
  incompleteness as it punishes your system, not answers missing from gold that the baseline
  also misses — acceptable, because "the noise that matters is precisely the noise that
  distorts your measurements. You don't need the dataset to be clean, you need to know the
  error bars on your scores." Frame it as "estimate noise rates as scalars, correct or caveat
  the metrics accordingly," which keeps it time-boxed. A 50-line normalizer with a test file
  of known-equivalent pairs "is an afternoon and permanently improves every number you
  produce after it."
- *Over-indexing guardrails.* Touch the official test set rarely and only at milestones;
  when cleaned-dev and official-dev diverge in *trend* (not level), treat it as a finding —
  cleaning or dataset encodes something systematic. "Maybe the field needs a better dataset"
  is a live phase-3 pivot, not phase-1 creep. "The failure mode for someone with your
  instincts isn't over-indexing on QAMPARI; it's the cleaning-and-tooling rabbit hole
  wearing the costume of rigor."

## 2026-08-17 — Sequencing for delivery; unbiased error-driven annotation

**Danielle.** Agrees the ceiling numbers are findings; they have not historically been
treated that way. On annotation: surfacing candidates from existing systems is clearly
biased toward their errors; she proposed using *generic baselines* (not the proposed method)
to surface potential errors, to avoid biasing the fixed dataset toward the proposed
approach over the baselines it will be compared against. (Discussion of how to handle
collaborators' expectations was dropped from this record.)

**Feedback received (near-verbatim, method-neutral parts).**
- *Sequencing.* "The dumb floor system (string-link → BM25 → read → normalize) is maybe a
  week of work precisely because it's dumb. Ship that number first. Then the ceilings arrive
  in the same breath — 'baseline gets 18 F1; here's why, stage by stage; the metric alone
  eats 6 points; retrieval caps us at 61.'" Avoid sequencing as measurement-then-baseline
  with a quiet measurement phase: "Interleave, and lead with whatever emits a number
  soonest." A ceiling ladder that costs two extra days beyond the baseline needs no
  justification once the findings are on the table.
- *Baseline-surfaced annotation is valid* — it addresses "a dataset quietly tuned so that
  your method's characteristic outputs get credited." Three refinements: (1) use a **union of
  diverse baselines** (BM25, a dense retriever, closed-book LLM) — any single system's
  disagreements oversample its error style; (2) the real protection is a **system-blind
  annotation criterion** — each disputed answer or passage is judged against the question
  and Wikipedia alone, with no information about who produced it, and corrections apply in
  whichever direction they cut; "if the judgment procedure is system-agnostic, the sampling
  frame only determines coverage, not direction"; (3) **freeze the cleaned set (version,
  hash, date) before the proposed method ever runs against it** — "'the dataset was fixed
  before the method existed' ends the conversation."
- *Residual bias to state, not solve.* Baseline-surfaced candidates only cover errors some
  system makes; gold answers every system agrees with but which are wrong, and missing
  answers no system finds, stay invisible. Put it in a limitations sentence: the cleaned set
  is a lower bound on dataset noise. "You're estimating error bars, not achieving ground
  truth."

---

## 2025 (undated, ~spring) — Note-taking workflow feedback and the project tracks of that period

Danielle asked an assistant to analyze a two-week markdown export of her Heptabase research
notes: "(1) highlight the major projects I've been working on (either learning or research)
and (2) provide analysis, feedback and insight around both the note taking and the subjects
of the notes." The notes themselves are gone; only the response survives. Date inferred
from tool references (4o vs. o3, Heptabase Insight).

**Tracks it identified (historical record of what was active then).**

| Track | Summary from the response |
|---|---|
| Engineering Journey (`dr_exp` ↔ `deconCNN` ↔ `dr_analyze`) | Reproducible pipeline to generate, schedule, run, and analyse CNN experiments (the original requirement statement is quoted in `experiment-tooling.md`); Stage 1 planning → Stage 2 parallel impl → Stage 3 integration & testing (then current). Jobs DB, Manager/Worker, adapter layer, test refactor. |
| ExpMan + loss-slope study | Loss-curve linearity vs. accuracy on CIFAR-10 sweeps; early-epoch metrics as predictors; slope/R² diagnostics. → lineage of EDP, recorded in `../../potential-projs/early-dynamics-prediction.md` §4. |
| Learn-with-agents #1: empirical NTKs | Comparative learning sprint across Claude / Gemini / Perplexity for eNTK intuition. |
| Content-distribution mini-project | Platform comparison (Medium, Substack, Ghost, Hashnode) for dev tutorials. |
| Infrastructure cleanup & storage | `dr_results` repo split, `.gitignore` trim, Supabase run sync, git-LFS. |
| Tool / model landscape tracker | 4o vs. o3, Claude tiers, aggregators, dev tooling. |
| Personal productivity / knowledge management | Heptabase workflows, daily journals, whiteboard hierarchies, atomic-card method. |

**Workflow feedback that survived (near-verbatim, condensed).**

- What was working: stage-gated boards and daily journals that "mirror true project flow";
  an atomic-card habit ("one idea per card"); embedding external artefacts for provenance;
  systematic capture of agent conversations as "an audit trail for agent-generated code and
  decisions."
- Suggested refinements: a weekly roll-up card per active project (wins, blockers, next
  actions) linked to the stage board; a `YYYY-MM-DD-Topic` naming convention to avoid
  duplicate titles like "Stage 3"; a triaged-link marker; summarise long chat logs and
  back-link the summary to the log; a master index board with one section per track; tag
  hygiene (three top-level tags — `project`, `resource`, `prompt` — with specifics nested);
  status markers in card names (planned / in-flight / blocked / done); a prompt-snippet
  library; Friday pruning of done cards to a "Closed" board.
- Project-level advice: lock the `dr_exp` ↔ adapter API before adding sweep features, and
  keep a single-GPU ResNet-18 benchmark run green in CI before merging refactor PRs; finish
  the storage split before the next experiment wave "otherwise large artifacts will creep
  back into the code repos"; turn the tool tracker into a decision log by adding a
  one-line personal verdict per tool after trying it.

Intake note: most of this is generic productivity advice, kept here only because the
"audit trail for agent-generated decisions," "weekly roll-up per track," and "decision log,
not research list" points are the same instincts that now shape this repo's intake
conventions (`../../danielle-inputs.md`, dated §4 records, the potential-projs README
table). The seven-track table is the only surviving record of that period's projects.

# Project-approach principles — Danielle's takeaways for starting new research projects

**Kind:** reference — a standing accumulator for Danielle's own methodology principles,
drawn from looking back at past projects, plus feedback received on them. Cross-project;
applied concretely in `../staging/maqa-oracle-ladder.md`.

Source: Danielle's statement on the "MAQA Next Steps" page (2026-08-17; intake 2026-08-22)
and the response to it.
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

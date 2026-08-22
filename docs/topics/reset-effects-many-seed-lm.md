# Do reset effects in LMs survive tuned regularization and many seeds?

**Kind:** staging. Candidate exits: a replication-style project on PolyPythias, or a
cross-cutting requirement folded into every other reset staging topic. Gap **G8**.

Source: the 2026-08-22 reinit/transfer literature pass (`reinit-and-transfer-literature.md`; full report at `~/drotherm/data/.claude/datadec/2026-08-22/0031-reinit-transfer-litpass.md`). Gap statements are quoted from that report; "closest work" citations were retrieved by the subagent (arXiv IDs), but verdicts rest on abstracts and no forward-citation sweep was run.

---

## 2026-08-22 — the gap

**G8 — Many-seed replication of reset effects in LMs** *(medium confidence)*. "*When Does
Re-initialization Work?* (Zaidi et al., arXiv 2206.10011) did 15,000 vision models and found
the effect *disappears* under tuned regularization — a result nobody has checked in LMs.
Combined with the Butterfly Effect finding (arXiv 2506.13234) that trajectories are
seed-sensitive, most single-seed LM reset claims are underpowered. PolyPythias makes the
seed dimension free." Cost: small training runs.

**Design note.** Same shape as the warm-starting decomposition's factorial — reset
interventions × regularization/optimizer settings × seeds — with the outcome being whether
the reset effect's confidence interval excludes zero once regularization is tuned. This is
the "exhaust the boring explanations" discipline applied to resets.

**Waiting on:** a decision on whether this stands alone or becomes the seed/regularization
requirement in the other reset topics.

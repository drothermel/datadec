# Topics — staging and reference docs outside the project list

Two kinds of topic doc:

- **Staging** topics hold ideas without a project home and have the two exits described
  below (promoted to a project, or absorbed into project §4s; deleted either way).
- **Reference** topics are standing accumulators — paper references and thoughts on a theme
  (plasticity, loss-curve forecasting, …) that many projects draw on. They are never
  "resolved"; they just grow, dated entry by dated entry.

Scratch notes collected from external conversations that do not clearly belong to an existing
project in [../potential-projs/](../potential-projs/) or to the program-level notes in its
README. Entries use the same convention as the project docs' §4: dated, attributed, quoted
close to verbatim, no decisions.

Unlike project docs, topic docs may reference projects freely — they are staging, not plans.

**Caveat on quoted related-work claims.** Statements in quoted external text about what does or
does not exist in the literature ("nobody has…", "this is unclaimed", specific paper
attributions) are unverified: the responding agents did not run literature searches and have no
reliable knowledge of the current landscape. Treat them as leads to check, not facts. The ideas
stand on their own.

**Routing rule for a new excerpt:** clearly about one existing project → that doc's §4;
clearly program-level → the potential-projs README; neither → a topic doc here (create on first
need).

**Lifecycle:** a topic doc has two exits — it matures into a new `potential-projs/` doc (same
three-section template, prefixed IDs, §4 seeded from its entries), or its entries are absorbed
into existing projects' §4s. Either way the topic file is deleted at that point, so this
directory only holds live, unresolved material.

| Staging topic | What it is waiting on |
|---|---|
| [checkpoint-tomography.md](checkpoint-tomography.md) | A prior-art pass (devinterp / WSD follow-up literature) and a decision on whether the four-probe battery becomes the flagship instrument framing wrapping annealed readouts, landscape geometry, token-level movement, and functional featurization |
| [posttraining-experiment-design.md](posttraining-experiment-design.md) | Whether any of its designs (post-training power analysis / RL-ability proxy; "did SFT move the model in distribution space"; within-reach-task post-training; late-window cross-family intervention) becomes a project or is absorbed into TINY / ANN / TOK |
| [nonstationarity-accounting.md](nonstationarity-accounting.md) | Whether "non-stationarity accounting" becomes the thesis-level framing (see candidate program framings in the potential-projs README); its case study has moved to the MoE movement project |

Absorbed on 2026-08-21 into project docs: `moe-analysis-program` and
`moe-routing-as-data-instrument` → `moe-partitions.md` / `moe-movement.md`;
`moe-recipe-suite` → `moe-recipe-suite.md`; `small-scale-measurement-science` →
`tiny-scale-measurement.md`; `functional-featurization` → `functional-featurization.md`;
`beyond-datadecide-data-measurement` → `recipe-featurization.md` §4 and REC-11.

| Reference topic | Theme |
|---|---|
| [plasticity.md](plasticity.md) | Continual-learning plasticity: papers, cheap training statistics, links to response vectors, modular plasticity, critical periods |
| [loss-curve-forecasting.md](loss-curve-forecasting.md) | Multi-power law, loss→downstream-accuracy mappings, emergence-as-threshold caveat |
| [pretraining-to-posttraining.md](pretraining-to-posttraining.md) | DataDecide; pretraining choices → post-training outcomes at matched loss; the "post-training did nothing" literature; retrospective on the earlier project |

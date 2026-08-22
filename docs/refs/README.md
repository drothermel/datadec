# refs — local copies of external source material

Notion page mirrors and verbatim external reports. Re-pull or re-copy rather than edit.

Pulled with the Notion CLI, one call per page:

```bash
ntn pages get 3c1de135cd1f81ac8b01f62d63d403cb > research-trajectory-pre-to-post-training.md
```

`ntn pages get` leaves page mentions as `<mention-page url=".../p/<id>"/>` stubs; resolve them
to `[title](url)` by fetching each unique id's title with `ntn api /v1/pages/<id>` (the
`title` property) and substituting. Re-pull rather than edit these files.

| File | Source |
|---|---|
| `research-trajectory-pre-to-post-training.md` | https://app.notion.com/p/Research-Trajectory-Pre-to-Post-Training-3c1de135cd1f81ac8b01f62d63d403cb |
| `multi-answer-qa-state-of-research-2026.md` | Deep-search report (external assistant, 2026-08-16) linked from the "MAQA Next Steps" page; copied from Danielle's Desktop 2026-08-22 |
| `2025-07-early-dynamics-predict-model-performance.pdf` | Danielle's own July 2025 draft ("Early Dynamics Predict Model Performance", with K. Cho) — the DataDecide early-training → final-performance prediction proposal; copied from her Desktop 2026-08-22 |

## Citation gaps noticed during intake

Citations on the Notion page that lack a page mention / link, or otherwise need verifying.
Fix on the Notion page, then re-pull.

| Where | Citation | Noted |
|---|---|---|
| Toggle 5 ("How to Combine Vision and Language…") | "Raventós et al." — author not linked; paper title is linked | 2026-08-22 |
| Toggle 14 (embedding-reset lineage) | "≥50B tokens of continued training" attributed to Dagan et al. (arXiv 2402.01035) — inverted: the paper says >50B tokens lets you *profitably specialize* a tokenizer, not that recovery requires it | 2026-08-22 |
| Toggle 14 | Lu et al. 2021 title is *Pretrained Transformers as Universal Computation Engines*; "Frozen Pretrained Transformer" is the model name | 2026-08-22 |

## Conversation artifacts live outside the repo

Briefs, deep-research reports, PDFs, and search archives from external conversations are
stored under `~/drotherm/data/convo-artifacts/<year>/<conversation-folder>/`, not here.
Planning docs cite the absolute path; the intake note for each lives in the reference
topic that consumed it (e.g. `../topics/reference/code-feature-extraction-tooling.md`,
`../topics/reference/prompt-compression-and-optimization-literature.md`).

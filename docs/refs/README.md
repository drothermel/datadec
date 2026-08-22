# refs — local copies of Notion source pages

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

## Citation gaps noticed during intake

Citations on the Notion page that lack a page mention / link, or otherwise need verifying.
Fix on the Notion page, then re-pull.

| Where | Citation | Noted |
|---|---|---|
| Toggle 5 ("How to Combine Vision and Language…") | "Raventós et al." — author not linked; paper title is linked | 2026-08-22 |


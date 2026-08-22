# Literature review process and provider access

How the recipe-featurization literature review is actually being run (2026-08-21), and how to use the scholarly providers — OpenReview in particular now that an authenticated path exists. Companion to [recipe-featurization-litreview-plan.md](recipe-featurization-litreview-plan.md) (the approved design) and the working packet at
`~/drotherm/data/.claude/datadec/2026-08-21/1412-recipe-featurization-litreview/`.

## 1. Process overview

Three Claude Workflows run in sequence; the orchestrating session reviews between them and owns every live agent. All workers are Opus, fresh per task, with self-contained prompts; every worker is read-only except the single corpus writer. One file is the exchange contract throughout: `candidates.jsonl` (one row per work, schema in the plan §1.2). Dedup is done in the orchestrating script against a *seen* set (never against *accepted*), so refuted or merged papers do not resurface.

| Workflow | Agents | What it produced |
|---|---|---|
| **1 · Scope** | 7 | 75 seed rows from the existing registry and transcripts (63 exact, 12 title-only); five subdomain briefs (`briefs/A–E.md`); `scope-review.md` — partition confirmed, five ownership conflicts ruled, three cross-cutting gaps assigned (G1 small-n methodology, G2 QC-variant/filter-strength, G3 data-card precedent), per-track round budgets |
| **2 · Plan** | 5 | Three search plans (229 new queries, 48 citation-graph seeds), `retrieval-paths.md` with tested request templates for every provider, `plan.md` (741-line execution plan), `round0-ingest.jsonl` (61 brief-named IDs missing from the seeds) |
| **3 · Execute** | ≈180 | Round 0 → per-track sweeps → consolidation → tiering + quality → cards → corpus ingestion → synthesis → critic → one bounded follow-up round. Output: `recipe-featurization-litreview.md` in the packet, cards per paper, `paperpile-import.txt` |

### Workflow 3 stages in order

1. **Round 0 (mandatory).** Batch-verify every arXiv ID/DOI the briefs named that the seeds lacked; review-only title lookups for prose-only names; a serial pass applying the scope judge's merge/split rulings. Nothing is searched until the known papers are in.
2. **Sweeps, loop-until-dry per track.** Tracks A, B, D-eval, D-schedule, E start together; C and G1 start after D-eval's first round (C inherits D's verified annealing seeds). Each round: three *blind* finders per track — citation graph (OpenAlex `cites:`/`referenced_works`, 50 IDs per call), query (arXiv API with category/date windows; S2 bulk search with year filters, filtered locally), venue (authenticated OpenReview enumeration + ACL Anthology) — then script-side dedup against *seen*, then identity verification in slices of 15 (primary record retrieved; canonical key assigned; all OpenAlex manifestation IDs collected). A track is dry after two consecutive rounds with zero new normalized identifiers; hard stop at 5 rounds (3 for C, 2 for G1), logged as a critic finding if hit.
3. **Consolidate.** One writer merges every per-agent ledger into `candidates.jsonl` (previous versions kept) and partitions rows into ~20-paper slices.
4. **Tiering and quality.** Two independent adjudicators per slice assign core / supporting / peripheral and the project directions (D4 in scope); a tiebreak agent resolves disagreements. In parallel, quality scorers compute the external-reception axis (S2 batch counts as primary, OpenAlex summed across manifestations, cohort percentile via `group_by`, who-cites-it, venue and OpenReview reviewer scores, artifacts, quarantined author prior; `too-new-for-external` under 6 months).
5. **Cards.** Full deep-read cards (quoted setup, effect sizes with uncertainty, rigor rubric with quoted evidence, transfer-assumption register) for core and supporting; short cards for peripheral. Full text only — abstracts are not evidence.
6. **Corpus ingestion (serial).** `paper-corpus add` for works absent from the identifier index, then `paper-corpus enrich` for metadata and source-first full text. No Notion or Paperpile operations; the phd working-tree state is recorded in the log.
7. **Synthesis.** One agent per subdomain (lineage → frontier → contestation written back into each card → DataDecide-violated assumptions → gaps mapped to C-a/D1–D5), then an integrator producing the review with a per-direction known / contested / never-done table and citations keyed to corpus paper IDs.
8. **Completeness critic + one follow-up round.** Unswept modalities, claims resting on title-only papers, directions with <3 recent papers, cap hits, auth expiry — each becomes a bounded instruction; after one round the workflow stops.

### Invariants held fixed

- No paper enters synthesis without `identity_status: exact` and a retrieved primary record.
- Recency is a dedicated date-filtered sweep, not a hope.
- Corpus writes are append-only; the phd repository, Notion, and Paperpile are untouched.
- Every provider-touching agent reports calls, credits (from OpenAlex `x-ratelimit-*` headers), and 429s; the script enforces a $5 OpenAlex ceiling per workflow (estimate for the whole run ≈ $0.15).
- Secrets come from the environment via `mise` and are never printed, logged, or written.

## 2. Provider access

All credentials live in the mise secrets file (`dotfiles/home/.config/mise/secrets.env.json`) and are exposed to any process via `eval "$(mise env -s bash)"`. Currently available: `OPENALEX_API_KEY`, `OPENREVIEW_TOKEN`, `OPENREVIEW_USERNAME`. Not available: `SEMANTIC_SCHOLAR_API_KEY` (request pending; plan assumes none).

| Provider | Role in the review | Access | Key constraints |
|---|---|---|---|
| arXiv API | discovery by category/date/keyword; `id_list` batch lookups; primary records | none | ~3 s courtesy spacing |
| Semantic Scholar | headline citation counts (`citationCount`, `influentialCitationCount`) via `POST /graph/v1/paper/batch`; recall search via `/paper/search/bulk` | none | batch ≤20–50 IDs/call unauthenticated, 1.5 s spacing, backoff on 429; ranked `/paper/search` is closed without a key; never use per-paper `/citations` |
| OpenAlex | citation graph, manifestation IDs, venue/author metadata, cohort percentiles | `api_key` query param | metered per call; batch 50 per filter; `select=`; `search=` costs 10×; sum counts across manifestations; undercounts ML citations — never the headline count |
| OpenReview | venue enumeration (accepted, rejected, withdrawn), reviewer ratings and decisions | bearer token | see §3 |
| Crossref / ACL Anthology | DOI records; ACL venue lookups | none | polite pacing |

## 3. Using OpenReview with the token

OpenReview has no API keys. Authentication is account email + password exchanged for a JWT at `POST https://api2.openreview.net/login`; the token is then sent as a bearer header. The token in mise was generated by Danielle in her own terminal (agents never see the password and must not attempt to log in).

### 3.1 What the token changes

Without it, `GET /notes` in every query form returns HTTP 403 behind a Cloudflare Turnstile challenge (do not attempt to bypass it). With it, verified 2026-08-21:

- `GET /notes?content.venueid=ICLR.cc/2026/Conference&limit=2` → 200 with submissions
- `GET /notes?content.venueid=ICLR.cc/2026/Conference/Rejected_Submission` → 200
- the same calls unauthenticated → 403

So venue enumeration — accepted, rejected, withdrawn — is now possible, and reviewer ratings for rejected submissions are retrievable as contestation evidence.

### 3.2 Request pattern

```bash
eval "$(mise env -s bash)"
# discover the venue's field names first (unauthenticated works too)
curl -s "https://api2.openreview.net/groups?id=ICLR.cc/2026/Conference" \
  | python3 -c 'import json,sys; c=json.load(sys.stdin)["groups"][0]["content"]; print({k:c[k]["value"] for k in ("submission_venue_id","rejected_venue_id","withdrawn_venue_id","review_name","review_rating","decision_name") if k in c})'

# enumerate submissions (paginate with offset until an empty page; limit <= 1000)
curl -s -H "Authorization: Bearer $OPENREVIEW_TOKEN" \
  "https://api2.openreview.net/notes?content.venueid=ICLR.cc/2026/Conference&limit=1000&offset=0&select=id,content.title,content.abstract,content.venue,content.venueid,content.authors"

# reviews for one forum
curl -s -H "Authorization: Bearer $OPENREVIEW_TOKEN" \
  "https://api2.openreview.net/notes?forum=<forum-id>&invitation=ICLR.cc/2026/Conference/Submission<N>/-/Official_Review"
```

Rules for agents:

- Always `eval "$(mise env -s bash)"` first; reference `$OPENREVIEW_TOKEN` by name only. Never print, echo, log, or write it; redact it from any saved headers or URLs.
- Read the venue's `/groups` record before parsing: field names are venue-specific (ICML 2026 uses `overall_recommendation`, not `rating`; ICLR 2024 ratings are strings like `"3: reject, not good enough"`, ICLR 2025/2026 are integers).
- Decision is read off the submission note's `content.venueid` / `content.venue` (e.g. `ICLR.cc/2026/Conference` + `"ICLR 2026 Poster"` = accepted; `…/Rejected_Submission` = rejected). Decision notes are not needed.
- `count` may be absent when `select=` is used — paginate until an empty page. `venue=`, `group=`, and `forum=` do not filter server-side on `/notes/search`; bucket client-side on `content.venueid`.
- Pace at 1 request/s; back off on 429/503.
- Unauthenticated fallbacks still work: `GET /groups?id=…` and `GET /notes/search?term=…&source=forum` (free-text over title/abstract/review body, `offset` pagination, no server-side venue filter).

### 3.3 Token lifetime and expiry handling

The token is a JWT valid for **24 hours** from issue (current one: issued 2026-08-21 18:49 UTC, expires 2026-08-22 18:49 UTC; decode the `exp` claim from the middle base64 segment to check — no secret is needed to read it). If a request returns 401/403 mid-run, agents stop the OpenReview modality, record `openreview-auth-expired` in the run log, and continue with the unauthenticated `/groups` + `/notes/search` path. Agents never re-login.

To refresh (Danielle, in her own terminal — the password is never typed into an agent session):

```bash
read -s 'OR_PW?OpenReview password: '; echo; curl -s -o /tmp/or_login.json -w 'HTTP %{http_code}\n' -X POST https://api2.openreview.net/login -H 'Content-Type: application/json' -d "{\"id\":\"$OPENREVIEW_USERNAME\",\"password\":\"$OR_PW\"}"; unset OR_PW; python3 -c 'import json; print(json.load(open("/tmp/or_login.json"))["token"])'; rm /tmp/or_login.json
```

Paste the printed value into `secrets.env.json` as `OPENREVIEW_TOKEN`. (Run `eval "$(mise env -s bash)"` first so `$OPENREVIEW_USERNAME` is set, or substitute the email.)

### 3.4 Known limits

- The token is tied to a personal account; requests are attributable to it — keep usage to enumeration and reads, paced.
- `/notes/search` `count` caps at 10,000 on broad queries; narrow by term or paginate by venue enumeration instead.
- Workshop venue IDs vary by year (e.g. `NeurIPS.cc/2024/Workshop/ATTRIB` exists; the 2025 ID does not follow the same pattern) — confirm each via `/groups` or search before enumerating.

## 4. Where things are

| Artifact | Path |
|---|---|
| Approved plan and quality scheme | [recipe-featurization-litreview-plan.md](recipe-featurization-litreview-plan.md) |
| Working packet (briefs, plans, ledgers, cards, synthesis, final review) | `~/drotherm/data/.claude/datadec/2026-08-21/1412-recipe-featurization-litreview/` |
| Execution plan (Workflow 3 contract) | `<packet>/plan.md` |
| Tested provider request templates | `<packet>/retrieval-paths.md`, `<packet>/retrieval-addendum-openreview-auth.md` |
| Paper corpus (works, raw provider responses, full text) | `~/drotherm/data/papers/` via the `paper-corpus` CLI in `phd` |
| Final review (copied into this repo when complete) | `docs/potential-projs/recipe-featurization-litreview.md` |

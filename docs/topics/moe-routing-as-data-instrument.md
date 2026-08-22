# MoE routing as a data-measurement instrument

**Status:** topic (staging). Candidate exits: absorbed into the routing follow-up in
trajectory drift/diffusion and TOK-obs-5 in token-level movement as a reframing; or a new
"routing instrument" project in a data-measurement program.

**Question posed (Danielle, 2026-08-21).** What if MoE-based model releases are incorporated
for the "someone else already did the work" aspect? See
[../danielle-inputs.md](../danielle-inputs.md).

Related-work claims below are unverified (see [README.md](README.md)).

---

## 2026-08-21 — Response

"Partially agree — but MoE releases change *which side of the supervised problem* they help
with, and it's worth being precise about that, because 'someone else did the work' is true
for a different kind of work than DataDecide did."

**What MoE releases actually give you.** "Your featurization program needs two things:
**treatment variation** (many datasets/recipes) and **outcomes** (what the models learned).
DataDecide's gift was the treatment axis — 25 recipes, everything else controlled. MoE
releases give you almost none of that: FLAME-MoE is seven models from 38M to 1.7B active
parameters — a *scale* ladder, one data recipe. OLMoE is one recipe (a mix building on Dolma
and DataComp-Baseline, released with open data, code, logs, and intermediate checkpoints).
OpenMoE is one recipe. And the big 2025–26 MoE wave — Llama 4, DeepSeek V4, Qwen 3.6, Kimi
K2.6, gpt-oss, Command A+ — is open-*weights*, closed-data: no treatment axis and often no
checkpoints. So no public multi-recipe MoE suite exists. The recipe question — the core of
your thesis — cannot be asked with these artifacts."

"What they *do* give you is a much richer **outcome channel**. Dense models give you loss and
eval scores; MoE models additionally give you per-token, per-layer categorical routing
decisions and emergent expert specialization. That's a learned, discrete decomposition of the
data, produced for free as a training byproduct."

**The reframing: routing as a data fingerprint.** "The OpenMoE analysis found that routing
decisions are predominantly based on token IDs with minimal context relevance, and
token-to-expert assignments are determined early in pretraining and remain largely fixed. If
routing is mostly a token-identity/frequency phenomenon locked in early, then **expert
assignment is essentially a data fingerprint** — a clustering of the corpus that the model
committed to based on early data statistics. That inverts the direction of your program in a
productive way: instead of 'featurize the data, predict the model,' you get 'read the
model's routing, recover a featurization of the data.' Concretely:"

- "Does the expert decomposition recover your intrinsic features (domain composition,
  frequency bands, determinism profile)? If routing ≈ token-ID clustering, it should correlate
  strongly with cheap corpus statistics — and *deviations* from that (context-dependent
  routing, late reassignments) mark exactly the tokens your entropy-bucket hypothesis cares
  about."
- "Routing-commitment timing as a per-token version of your migration analysis, observable
  from existing checkpoints with *no decay branches needed* — the categorical channel makes
  'committed vs. still moving' directly legible instead of inferred from KL. There's already
  adjacent work treating this dynamically — e.g. a recent paper tracking a three-phase
  load-balance trajectory (surge, stabilization, relaxation) across OLMoE and OpenMoE
  checkpoints — but it's aggregate-level; the per-token, data-linked version is open."
- "Cross-suite: OLMoE (largely DCLM-based) vs. FLAME-MoE vs. OpenMoE all have checkpoints and
  known data. Three points isn't a recipe axis, but it's enough to ask whether
  expert-specialization structure tracks corpus composition across independent training
  setups."

**Where the framing is wrong.** "If the motivation is 'add MoE to get more free outcome data
for the featurization regression,' I'd say no — it adds architecture as a confound without
adding recipe variation, and n stays tiny. One model per data point, with architecture,
tokenizer, and scale all varying between releases, is a worse supervised problem than
DataDecide, not a better one."

"If the motivation is 'MoE routing is a new *instrument* for reading data structure out of
models' — yes, and I think that's genuinely underexplored. It slots into your program as a
measurement-layer contribution (like the drift/diffusion decomposition or IRT θ), not as an
outcome-table expansion. The honest version of the pitch: dense models force you to infer
data-driven structure from continuous proxies (KL, CKA, per-token loss); MoE models write
part of that structure down explicitly, and nobody has systematically joined routing
statistics to measured corpus features."

**Ordering implication.** "This strengthens the case for building your reference-model scorer
and corpus-feature extractors first (they're what routing gets joined *to*), and it makes
[the routing follow-up] less of an orphan — it becomes 'the routing instrument' chapter of
the same data-measurement program rather than a separate suite with a separate story. The
ingest-uncertainty gate still applies, though: whether FLAME-MoE's released routing logs
support per-token tracking across checkpoints determines whether this is a T0 join or a T1
recomputation, and that survey should still be step one."

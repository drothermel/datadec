# Alternatives to CE / NLL as an evaluation metric — literature reference

**Kind:** reference (accumulator for loss-replacement metrics: reweighted or
token-selected NLL, byte/character-normalized and tokenization-marginalized likelihoods,
representation-side readouts, compression-based scores). Entries are dated.
Characterizations are a SciSpace agent's; identifiers unverified unless from the
downloaded PDFs. Siblings: `small-scale-evaluation-metrics-literature.md` (proxies that
predict downstream accuracy), `evaluation-methodology-literature.md`,
`token-level-literature.md`.

Why it matters here: every DataDecide-facing project reads a "loss" somewhere — TINY's
method axis, IRT's continuous response variable, DCARD's PPL tables, TOK's per-token
movement, EDP's early-curve features. Which *variant* of NLL is read (per token, per
byte, on gold spans only, reweighted by token class) changes rankings at small scale, and
DataDecide's own finding that per-character-normalized correct-probability beats raw
likelihood is one instance of this literature.

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-alt-eval-metrics-for-llms-agent-artifacts-zip_69962c47-a151-4f10-b0bc-84aa7faabc11_1787425013/` — the report (md + LaTeX/PDF), 10 key PDFs with cropped
figures, 18 further full texts, 57 mostly off-topic arXiv downloads, and 15 search CSVs
(three merged with extracted method columns). **`INDEX.md` inside the folder is the
file-level index** with the missing-canon list.

---

## 2026-08-22 — SciSpace deep review (undated, ~2026)

**Danielle's prompt (verbatim):**

> I want you to do a literature review of alternatives to CE Loss or Negative Log
> Likelihood for evaluation. I specifically want you to look at metrics that are meant to
> be a replacement for loss. I want you to exclude task accuracy via generation or
> ranking, or any similar common metrics. One example of what I might want is a
> modification of NLL that excludes or reweights certain tokens. I want you to especially
> pay attention to methods which do NOT depend on vocabulary, tokenization, or model
> architecture.

**What survives, by family (evaluation-side only).**

- *Token-selected / reweighted NLL as a metric:* **LongPPL** (Fang et al. 2410.23771) —
  perplexity computed only on "key tokens" whose prediction gains from long context
  (selected by a long-context-influence score from a reference model); Pearson −0.96
  with long-context task accuracy where ordinary PPL is uninformative; the companion
  LongCE is the training loss. This is the closest published instance of Danielle's
  example. **PPLqa** (Friedland et al. 2411.15320) — |PPL(prompt+response) −
  PPL(response)| as an unsupervised response-quality score; reference-free; tracks human
  and LLM-judge rankings on MT-Bench. Reference-model token scoring (Rho-1 / "Not all
  tokens are what you need", 2404.07965) is a training-time selector but the same scorer
  defines an evaluation subset.
- *Tokenization- and vocabulary-independent likelihoods:* bits per byte / per character
  (loss normalized by bytes of the evaluated text, not tokens) — the standard fix,
  used by Biderman et al. "Lessons from the trenches" (2405.14782), ByteFlow, SuperBPE,
  MrT5, and the Script Tax study (BPC by script exposes tokenizer cost disparities);
  **marginal likelihood over tokenizations** (Cao & Rimell, EMNLP 2021) — sum over
  valid segmentations rather than the canonical one; larger gains out-of-domain and
  where tokenizer entropy is high; Vieira et al. (2412.03719) convert token-level LMs to
  character-level LMs exactly, giving character-string probabilities and
  tokenization-marginal perplexity; Takahashi et al. 2019 on why per-character and
  per-word perplexities are incommensurable; Brown et al. 1992 (1.75 bits/char bound)
  as the information-theoretic anchor.
- *Representation-side (prediction-independent, not architecture-independent):*
  Diff-eRank (Wei et al. 2401.17139; effective-rank drop of hidden-state covariance from
  untrained to trained model, tracks loss and accuracy with scale), Matrix Nuclear-Norm
  (Li et al. 2410.10672; O(n²) surrogate, 8–24× faster), a hybrid (Vo 2410.14480).
  Require hidden states; not applicable to black-box models; meaning as a "loss
  replacement" unclear.
- *Compression-based:* information capacity (Yuan et al. 2511.08066; compression
  efficiency including tokenizer efficiency), entropy-estimation modelling (Badger et al.
  2511.10618); the review omits Delétang et al. "Language Modeling Is Compression"
  (2309.10668), the canonical statement.
- *Semantic-distance scoring of predictions* (O'Neill et al. 2019): score wrong
  predictions by embedding distance to the target rather than 0/1 — the only entry that
  changes what "error" means rather than how tokens are counted.

**Intake notes.**

- About half of the review's "modified NLL" family is *training objectives* (FACE,
  selective LM, unlikelihood, TALR, AXE, focal/Lovász, strictly proper scoring rules,
  MixCE) — outside the ask; dropped above except where the same machinery yields a
  metric.
- One cited source is not credible ("The Shannon Paradox … 0.36 bits per character",
  Zenodo) and is presented as state of the art twice; ignore.
- "Architecture-independent" is the wrong label for the hidden-state metrics; the
  tokenization-independent family is the only one that meets Danielle's constraint, and
  BPB is its practical answer: normalize by bytes of a fixed evaluation string,
  optionally marginalize over tokenizations.
- Missing canon for the actual question: Paloma (per-domain BPB protocol, 2312.10523);
  DataDecide's per-character correct-probability and Signal-and-Noise (2508.13144);
  Patel et al. 2026 expert-trajectory reweighted token statistics
  (`small-scale-evaluation-metrics-literature.md`); gold-span BPB in OLMES; surprisal
  metrics. These are the related-work skeleton; the review is a seed.
- The arXiv download folder is search noise (photonics, astronomy, physics education);
  numbers as the agent reported, unverified.

# Regularization for transformer and MoE language models, especially on repeated data — literature reference

**Kind:** reference (accumulator for regularizers relevant to training small transformer
/ MoE LMs that see data more than once). Entries are dated. Characterizations are a
SciSpace agent's; identifiers unverified. Siblings: `moe-literature.md`,
`synthetic-data-literature.md` (repeated-data scaling sits there too),
`training-objective-alternatives-literature.md`.

Why it matters here: MSUITE and the DataDecide-dense substrate both train small models
on fixed corpora where multiple epochs are likely (MSUITE at 4–6 recipes; dense at the
smallest scales), and the Slicing-and-Dicing MoE sweep's 128× total/active ratio is
exactly the regime where MoE overfitting on repeats is reported. The retrain specs need
a stated regularization recipe.

**Artifacts on disk:** `~/drotherm/data/convo-artifacts/2026/scispace-regularization-methods-moe-agent-artifacts-zip_6c756278-eb52-4294-a000-1b2d39a29157_1787425549/` — the report, a companion open-access URL list (19 cited +
16 canonical papers), insight extraction, and ~50 search CSVs/JSONs; no PDFs.
**`INDEX.md` inside the folder is the file-level index** with the missing-canon list.

---

## 2026-08-22 — SciSpace review (undated, ~2026)

**Danielle's prompt (verbatim):**

> Write me a list of regularization methods for machine learning models. Specifically, I
> want to avoid overfitting Mixture of Experts Transformer Language Models on repeated
> data. I want you to include both well-supported ideas with extensive literature
> throughout general machine learning, such as dropout and weight normalization, as well
> as newer techniques that might be specific to MoE models or transformers. For each
> method, include citations.

**The list the report gives (condensed; citations as given, with the Part-2 canonical
anchors from the bundle's URL list added in brackets).**

- *General:* dropout [Srivastava et al. 2014]; weight decay / L2 [AdamW, Loshchilov &
  Hutter 2019]; L1; batch norm [Ioffe & Szegedy 2015]; early stopping; data augmentation
  (Hernández-García & König 2018 — augmentation can replace explicit regularizers;
  mixup); gradient clipping; **flooding** (Ishida et al. ICML 2020 — hold training loss
  above a floor); [weight normalization, Salimans & Kingma 2016; spectral norm; R-Drop
  consistency regularization; stochastic depth; DropBlock; label smoothing analysis,
  Müller et al. 2019].
- *Transformer-specific:* attention dropout; LayerDrop / structured dropout (Fan, Grave
  & Joulin ICLR 2020); UniDrop (feature + structure + data dropout, NAACL 2021); layer
  norm and label smoothing (cited to Liu et al. 2020); relaxed attention (2209.09735).
- *MoE-specific:* load-balancing auxiliary loss [Shazeer et al. 2017; Switch, Fedus et
  al. 2022]; expert / cluster-level expert dropout (MoEC 2207.09094; Elbayad et al.
  Findings ACL 2023 — gating dropout, conditional routing, and curriculum to fix MoE
  overfitting on low-resource languages in multilingual MT); Dirichlet-prior shaping of
  router outputs for upcycled MoEs (2510.01185); intra-/cross-layer expert-specialization
  regularizers (Hu et al., unpublished); [ST-MoE router z-loss, Zoph et al. 2022].
- *Repeated / duplicate data (the report's framing = remove the repeats):* exact
  deduplication (Lee et al. ACL 2022; Kandpal et al. ICML 2022 — privacy), SoftDedup
  commonness reweighting (2407.06654), ClusterClip balanced sampling (2402.14526),
  semantic deduplication, differential privacy and entropy filtering (SoK 2025);
  [Carlini et al. 2021 extraction].

**Intake notes.**

- **The report answers a different question.** Danielle asked how to regularize an MoE
  LM that *will* train on repeated data; §5 is about deduplicating so it doesn't. The
  paper written for her question is missing: **Xue et al. 2023, "To Repeat or Not To
  Repeat: Insights from Scaling LLM under Token-Crisis" (2305.13230)** — multi-epoch
  training of dense and MoE LMs; dropout is the regularizer that works (with a schedule
  that turns it on late), and MoE models overfit repeated data *more* than dense ones.
  Pair with Muennighoff et al. (2305.16264; repetition up to ~4 epochs is nearly free,
  then diminishing) and Hernandez et al. (2205.10487; repeated-data double descent
  that disproportionately damages induction heads). These three are the related-work
  skeleton for any DataDecide-dense or MSUITE regularization choice.
- Also missing on the MoE side: Switch Transformer's *expert dropout* (higher dropout
  inside experts than in the dense layers), ST-MoE's finding that sparse models overfit
  in fine-tuning and its z-loss, Gating Dropout (2205.14336), StableMoE, DeepSeek's
  auxiliary-loss-free balancing, OLMoE's stability recipe.
- Weight normalization, explicitly requested, appears only in the companion URL list.
- Several citations are off-target (load balancing cited to a 2025 forecasting paper;
  attention dropout to a speech-SSL paper; one "unpublished manuscript"; FedRW is
  federated-learning plumbing). Treat the report as a checklist, not a bibliography;
  the Part-2 URL list is the better citation source.

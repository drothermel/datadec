# Beyond DataDecide — dataset measurement for large pretrain / midtrain / post-train corpora

**Status:** topic (staging). Candidate exits: a program framing that wraps the existing
projects ("data measurement → training dynamics"), and/or new project docs for the specific
open sub-regions below (cross-suite transfer, midtraining data, measured data cards).

**Question posed (Danielle, 2026-08-21).** Is there space for dataset featurization/analysis on
the very large pretraining (or midtraining/post-training) datasets and their impacts, outside
the DataDecide-specific world? See [../danielle-inputs.md](../danielle-inputs.md).

---

## 2026-08-21 — Response

"Yes, with an important qualification: the space is real, but it's unevenly occupied, and the
crowded parts are exactly the ones that are easiest to do. So agreement, but the value depends
on picking the right sub-region."

**Where it's crowded.** "Descriptive corpus statistics as an end in themselves — WIMBD
basically planted that flag for the big public pretraining corpora (duplication,
contamination, toxicity, length distributions across C4, Pile, Dolma, etc.).
Quality-filtering ablations are also well-tilled: DCLM and FineWeb are essentially large
empirical programs on 'which filter produces which outcome.' And mixture optimization (DoReMi,
RegMix, data mixing laws) covers the 'predict the best weights' version. A paper whose
contribution is 'we computed features on big datasets' or 'we found that dedup and quality
filtering matter' lands in a saturated area."

**The structural bottleneck, and why DataDecide mattered.** "Featurization is cheap;
*outcomes* are expensive. The whole reason your featurizing proposal works is that someone
else paid for 25 recipes × 14 scales × 3 seeds. Outside DataDecide, your options for the
outcome side are: other controlled suites (DCLM's runs, FineWeb ablations, RegMix's 1000+
small models — these exist and are underused as supervised problems in exactly your sense),
observational model zoos (hundreds of open models, but confounded provenance — data,
architecture, and scale all vary at once), or training your own (compute-bound). This is the
real constraint on the generalized version of the idea, and it's worth being clear-eyed that
'featurize datasets and predict impacts' is really 'assemble (corpus, outcome) pairs,' which
is a data-collection problem more than an analysis problem."

**Where the genuinely open space is.**

- *Features that predict dynamics, not endpoints.* "Nearly all existing work maps features →
  final benchmark scores. Your determinism-profile → annealing-response idea generalizes:
  which data properties predict schedule sensitivity, emergence timing, noise levels,
  forgetting rates, susceptibility to churn. Nobody has a public map of that, and it's
  theory-motivated rather than fishing."
- *Cross-suite transfer of feature→outcome maps.* "Does a relationship fit on DataDecide hold
  on the DCLM pool or FineWeb ablations? A feature that predicts across suites is a finding;
  one that doesn't is an important negative about how contingent all these results are. This
  is cheap once the extractors exist and directly attacks the n=25 problem by pooling suites."
- *Midtraining/annealing data.* "This is the most practically hot and least publicly
  systematized area. Everyone post-MiniCPM knows you put 'high quality' data in the decay
  phase; what 'high quality' measurably *is*, and whether the right decay data is a function
  of the stable-phase data, is mostly folklore plus internal lab knowledge. A controlled
  featurization-plus-ablation study here would get read by practitioners immediately. The
  catch: fewer public controlled suites exist, so you'd likely need some training compute —
  though decay branches are cheap, which is exactly the [annealed-readouts / WSD-suite]
  machinery."
- *Post-training data.* "Also underdeveloped publicly, but different in character:
  instruction/preference/RLVR datasets are small enough that per-example influence methods
  (datamodels-style) become feasible, and the interesting features are things like difficulty
  distributions, response-length confounds, and diversity — not Zipf exponents. It's arguably
  a different field with the same slogan, and the confound structure (data interacts strongly
  with the base model) makes clean claims harder."
- *The measurement layer itself.* "Your [realized-composition] finding — labels ≠ realized
  token shares — almost certainly generalizes. Public datasets' stated compositions vs.
  measured compositions is unglamorous, highly citable, and nobody's job. A 'measured data
  cards' effort across the major public corpora is the WIMBD sequel that hasn't been written."

**One honest caution.** "The frontier labs do versions of this internally at scales and with
outcome data you can't match, and don't publish. That means the academic contribution has to
be either the *public artifact* (measured features + outcomes anyone can build on), the
*theory link* (determinism/geometry-style mechanistic features, which labs have less incentive
to care about), or the *dynamics angle*. Pure predictive performance — 'our features predict
benchmark scores with R²=X' — is a race you lose to people with a thousand internal
ablations."

**Framing.** "Agree, and I'd frame the generalized program as 'data measurement → training
dynamics,' not 'data features → benchmark scores.' The DataDecide work then becomes the first
instrument-validation study in a program rather than the whole program — which is also a much
better story for a thesis or grant than a single-suite reanalysis."

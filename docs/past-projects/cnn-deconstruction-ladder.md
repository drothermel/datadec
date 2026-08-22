# CNN deconstruction ladder (`deconCNN`, 2025) — past project record

**Status:** started 2025 (spring), not finished. Last known state: `dr_exp` / `deconCNN` /
`dr_analyze` pipeline at "Stage 3 integration & testing" (spring 2025); a first ablation
dataset of 15 configurations × 3–6 seeds on CIFAR-10 at a 25-epoch budget existed by
2025-06-12 (several versions were run); the slope/R² analyses on it are the separate
record `loss-slope-prediction.md`. Datasets not on file here — handles:
`regression_analysis_from_first_25_epochs.csv`, `epoch_metrics_long_format.csv`
(likely in `dr_results` / Supabase).

**What it was.** An ablation ladder over the building blocks of modern CNN vision recipes
(residual connections, normalization, activation, initialization, optimizer, schedule,
augmentation/regularization), run backwards from a full modern recipe toward LeNet-era
configurations, each rung trained briefly on CIFAR-10 (ImageNet-subset planned) with seeds
and a small HP sweep, tracking optimization-landscape metrics alongside accuracy.

**What was found.** With the full modern recipe as baseline (76.4% val at 25 epochs),
*every* removal improved validation accuracy (SGD 92.2%, no-warmup 93.7%, no-mixup
93.9%) — the short-budget underfit of heavy regularization, not evidence that the features
are useless. Budget is a hidden axis of the ablation. The later rungs were run at a
different LR (`lr-0.05`), confounding rung order with LR.

**What is unresolved.** Whether the ladder is revived: the design needs a per-rung budget
sweep (or convergence-matched rungs), a fixed modern recipe with one block varied per rung,
per-arm LR tuning, and pinned per-budget reference accuracies (original papers /
`pytorch-cifar` / airbench). The optimizer-arm choice (contrast set) was never settled.

**Where it points now.** `../potential-projs/landscape-geometry.md` and
`../potential-projs/tiny-scale-measurement.md` (TINY-opt-4 "design decisions tuned for larger
models" is this ladder's question at LM scale); the recipe-ablation literature it should be
positioned against — "Bag of Tricks" (arXiv 1812.01187), "Revisiting ResNets" (2103.07579),
"ResNet strikes back" (2110.00476); eNTK readouts per rung
(`../topics/reference/ntk-literature.md`). Tooling lineage: `../topics/reference/experiment-tooling.md`.

---

## Record (dated entries; Danielle's statements verbatim, responses condensed)

## Undated — Danielle's two framing statements (verbatim)

Requirement (from the logger question):

> I want to slowly deconstruct the improvements in architecture and training procedure that
> were introduced into CNN based vision models from today all the way back to the earliest
> "deep" models. For each step I want to train on CIFAR 10 and ImageNet for a minimal length
> of time to do fast iteration for comparison (not aiming for super competitive results),
> and I want to track key metrics about the optimization landscape as I go. This will
> involve making significant numbers of iterative changes and running a substantial number
> of runs (multiple seeds per configuration + simple hpm sweeps).

Scoping:

> I want to evaluate the impact of the fundamental building blocks of modern day CNN based
> vision architectures by choosing 5-10 of the largest research advances in architecture or
> training methodology (including things like residual connections, optimizer and
> initialization improvements, etc).
>
> Please present a definitive timeline of the major improvements to the CNN based vision
> architecture starting from LeNet in 1998 through modern day. The timeline should make it
> easy to visually understand the major change, and also to unfold more information about
> the change, why it was transformational at the time, and how it contributed to modern SOTA
> CNN based vision models.
>
> Focus on accuracy of your claims, and select the 5 largest impact changes to highlight as
> the first that I should run experiments on. Ask questions first if you need any additional
> information from me to refine the outcome to exactly what I'm looking for

The design, read from the two statements: a **ladder of ablations** over *building blocks*
(not whole architectures) — residual connections, normalization, activation, initialization,
optimizer, augmentation/regularization, schedule — each step trained briefly on CIFAR-10 and
a small ImageNet setting, multiple seeds + a small HP sweep per rung, with **optimization
landscape metrics** recorded alongside accuracy so the ladder yields a measurement story,
not just a leaderboard.

## Undated — The response: timeline and top-5 (condensed; figures unverified)

Timeline given (1998–2019): LeNet-5 (local receptive fields, shared weights, subsampling;
~60k params) → pre-AlexNet SVM era (ImageNet 2010 winner "52.9%") → AlexNet 2012 (ReLU,
dropout 0.5, GPU training, augmentation; "84.7% top-5") → VGG 2014 (3×3 only, depth; "74.4%
top-1") → GoogLeNet/Inception 2014 (parallel 1×1/3×3/5×5, 1×1 bottlenecks; "6.8M params vs.
VGG's 138M") → ResNet 2015 (H(x)=F(x)+x; "78.6% top-1, ResNet-152") → Batch Normalization
2015 → SENet 2017 (channel attention; "81.3%, surpassing human-level") → MobileNet 2017
(depthwise separable; "70.6%, 4.2M params") → DenseNet 2017 ("79.2%") → EfficientNet 2019
(compound scaling + NAS; "84.3% top-1, 8.4× smaller, 6.1× faster"). Training-method section:
Adam "became the standard optimizer"; He init for ReLU, Xavier for sigmoid/tanh; sigmoid/tanh
→ ReLU.

**Top-5 for experiments (the response's ranking, with its "experimental value" notes):**

| # | Innovation | Key idea | Experiments it suggested |
|---|---|---|---|
| 1 | AlexNet (2012) | ReLU, dropout, GPU training, augmentation | ReLU vs. other activations; dropout vs. overfitting |
| 2 | ResNet (2015) | residual connections | with/without skips; depth scaling with vs. without; gradient-flow measurements |
| 3 | Batch Norm (2015) | normalize layer inputs | with/without BN; learning-rate tolerance; alternative normalizers |
| 4 | LeNet-5 (1998) | conv fundamentals | conv vs. fully connected; weight sharing |
| 5 | EfficientNet (2019) | compound scaling, NAS | compound vs. single-axis scaling; efficiency–accuracy trade-off |

Suggested phasing: LeNet + AlexNet → ResNet + BN → EfficientNet, over ~6 weeks; metrics:
time to convergence, final val accuracy, "training loss smoothness," FLOPs/params, test
performance; datasets: CIFAR-10/100 for iteration, ImageNet subset for scale, MNIST for the
LeNet baseline.

## Undated — Recipe fact-finding for the ladder rungs (Perplexity Q&A; condensed)

A run of short factual questions about which training-recipe components each classic paper
actually used — the raw material for defining the ladder's rungs so that each ablation
matches a real historical configuration. The response's answers are tabulated below with a
verification column; the originals are blog/StackOverflow-sourced, and several are
incomplete or wrong (marked). **Verify against the papers before encoding a rung.**

| Paper | Component | Response's answer | Check |
|---|---|---|---|
| ResNet (He et al. 2015) | BN | yes — after each conv, before activation | correct |
| ResNet | dropout | no | correct |
| ResNet | weight decay | yes, 1e-4 | correct (momentum 0.9) |
| ResNet | optimizer | SGD + momentum, not AdamW | correct |
| ResNet | augmentation | random 224 crop from 256-resized image, flip, per-pixel mean subtraction | **incomplete** — the paper uses scale augmentation (shorter side sampled in [256, 480]), random crop, flip, per-pixel mean subtraction, *and* AlexNet-style PCA color augmentation; the "resize to 256" recipe is the evaluation/single-crop pipeline, not the training one |
| ResNet | filter sizes | 7×7 stem; 3×3 basic blocks (ResNet-18/34); 1×1–3×3–1×1 bottlenecks (50+) | correct |
| GoogLeNet (Inception v1) | dropout | yes, 0.4 before the final linear layer | correct (auxiliary heads used 0.7) |
| GoogLeNet | weight decay | yes, value unspecified | plausible; the paper does not state it — unverified |
| GoogLeNet | LRN | yes, early layers | correct (two LRN layers in the stem) |
| GoogLeNet | augmentation | "random cropping, horizontal flipping, PCA color augmentation" | **shaky** — the paper describes crops of 8–100% of image area with aspect ratio in [3/4, 4/3] plus photometric distortions (Howard 2013), and says the exact recipe varied across the ensemble; the response first said "color jitter," then reversed to PCA color aug; neither is quoted from the paper |
| LeNet-5 (1998) | dropout / augmentation | no dropout; shifting and simple geometric distortions | plausible; the 1998 paper's distortion experiments are on an augmented MNIST — unverified detail |
| VGG (2014) | weight decay / dropout / ReLU | 5e-4 WD; dropout 0.5 in first two FC layers; ReLU everywhere | correct |
| VGG | LRN | tested in config A-LRN, no gain, dropped | correct |
| VGG | pooling | non-overlapping 2×2 stride 2 | correct |
| AlexNet (2012) | contributions vs. baselines | depth (8 layers), ReLU vs. tanh/sigmoid, dropout in FC, overlapping 3×3/s2 pooling, LRN, two-GPU training, crop/flip/PCA color aug; 15.3% vs. 26.2% top-5 error | figures correct; "baselines were 1–2 conv layers" is a caricature (Ciresan et al. 2011–12 had deeper GPU CNNs); dropout was Hinton et al. 2012, popularized rather than introduced by AlexNet |
| AdamW | date vs. CutMix | AdamW late 2017 (Loshchilov & Hutter); CutMix 2019 | correct |
| AdamW | schedule convention | both cosine and step used; "no single default" | content-free; in practice cosine (often with warmup) dominates AdamW usage |
| CNN classification | BCE vs. CE | CE is standard; BCE for binary/multi-label | correct as stated — but note BCE for single-label ImageNet is a real modern recipe (e.g., ResNet strikes back, Wightman et al. 2021; unverified here), so "not standard" is dated |
| CIFAR-10 | color | RGB, 32×32 | correct |


## 2025-06-12 — First ablation dataset: what was run and how it was (mis)read

Danielle's statement (verbatim):

> I have a dataset from training CNN based vision models while removing features that
> have been added for the last many years. Note that the train acc and loss are calculated
> on the augmented data so it isn't comparable to the val.
>
> Please analyze this data

The dataset is not on file (2026-08-22); this section records what the response reveals
about it and how the response went wrong, so the analysis can be redone when the data is
found.

**What existed (reconstructed from the response).** 15 configurations, 3–6 seeds each,
named `step00_baseline` … `step1x_*`; the baseline is the **full modern recipe** and each
step *removes* one feature. Steps named: `step01_sgd` (optimizer swap — so the baseline
optimizer was adaptive, presumably AdamW), `step04_no_mixup`, `step05_no_warmup`,
`step07_no_residual`, `step10_no_dropout`, plus no-horizontal-flip, no-random-resized-crop,
no-color-jitter, no-RandAugment, no-CutMix, a ResNet-12 and an AlexNet architecture
variant, and an activation change. Grouped by the response into optimizer / augmentation /
schedule / architecture / regularization. Reported validation accuracies: baseline 76.37%;
`step01_sgd` 92.24%; `step05_no_warmup` 93.71%; `step04_no_mixup` 93.86%; every removal
improved validation accuracy by 1.6–17.5 points. Train−val gaps: baseline −0.319;
no-hflip +0.219; no-RRC +0.180; no-color-jitter +0.072; no-dropout +0.067; negative for
baseline, SGD, no-RandAugment, no-CutMix.

**The response's reading.** "The baseline configuration contains fundamental flaws";
the negative train−val gap is "a pattern typically associated with data leakage, inadequate
training, or over-regularization"; removing augmentation "challenges conventional wisdom";
configurations with positive gaps show "concerning overfitting"; recommendations: rebuild
the baseline on SGD without "excessive regularization," tune hyperparameters before
ablating, hunt for leakage.

**Intake note — the response ignored the stated design and the stated caveat.**

1. *The negative gap is the caveat, not a bug.* Danielle said train metrics are computed on
   augmented data. With mixup/CutMix/RandAugment/RRC on, train accuracy on augmented
   batches is *supposed* to sit far below clean validation accuracy; as augmentations are
   removed the gap shrinks and then flips positive. The response's "leakage" and
   "overfitting" labels are both just this artifact read in two directions. The gap is not
   interpretable at all under this design; only validation accuracy (and ideally a clean
   train-set eval) is.
2. *"Every removal improves" is the fixed-budget confound, not a flawed baseline.* The
   design trains each rung "for a minimal length of time." Heavy regularization (mixup +
   CutMix + RandAugment + dropout + warmup + AdamW) is the recipe that wins at *long*
   budgets (hundreds of epochs) and underfits badly at short ones. A full-modern baseline
   at 76% on CIFAR-10 with every single removal helping is exactly what a short budget
   predicts. So the dataset measures "which features hurt under a short budget," which is
   a legitimate and interesting question — but it is not "which features help," and the
   response's conclusion that modern practice is "fundamentally challenged" is the budget
   confound talking.
3. *Consequence for the ladder design.* The headline result is that **budget is a hidden
   axis** of the ablation: each rung's contribution is a function of training length, and a
   one-budget ladder cannot separate "this feature is useless" from "this feature needs
   more steps to pay off." The fix is either a small budget sweep per rung (e.g. 3 budgets)
   or matching rungs at a convergence criterion rather than a step count — the same
   matched-loss-vs-matched-steps issue the DataDecide-side docs wrestle with
   (`../research-hypothesis.md`). This is also where landscape / eNTK readouts
   (`../topics/reference/ntk-literature.md`) would add something the accuracy table cannot: whether
   a feature changes the *path* even when it does not change the short-budget endpoint.
4. *What to redo when the data is found.* Plot val accuracy vs. rung with seed error bars;
   drop the train−val gap entirely or recompute train accuracy on un-augmented data; group
   rungs by removal type but report per-rung; check whether the adaptive-optimizer baseline
   used a sensible LR (the +15.9 from switching to SGD at the same budget is large enough
   that an LR mismatch is the first suspect, before any claim about optimizers).

### Undated (~2025) — CIFAR-10 reference accuracies for the ladder's endpoints

Danielle's prompt was not pasted; the response is a "CIFAR-10 Baseline Accuracy Report for
Popular CNN Architectures" (ResNet, WRN, VGG, DenseNet, ConvNeXt) — presumably asked to
sanity-check where the ladder's top runs (~93.9% at 25 epochs) sit against published
baselines.

**What the response reported (unverified; mostly minor-venue papers):** ResNet-18
"82.94–89.56%"; ResNet-50 "76.04%" (attributed to overfitting); ResNet-20 ~91–92%
(91.43% as a pruning baseline); ResNet-32 94% under 50% pruning; a <5M-parameter ResNet at
96.04% (arXiv 2306.12100); WRN "92–95%"; VGG-16 "90–94%", VGG "93.43% with CIFAR-optimized
training"; DenseNet-121 "93–95%", often the top in its comparisons; ConvNeXt — nothing
found. Caveats it raised: Recht et al. 2018 (arXiv 1806.00451) — a fresh CIFAR-10.1 test
set drops accuracy 4–10%; Barz & Denzler (MDPI J. Imaging 2020) — 3.3% of test images have
near-duplicates in train, and "accuracy drops by 9–14%" when removed.

**Intake notes — use the original-paper numbers, not these.**

- Canonical CIFAR-10 figures (from memory; check the papers before quoting): He et al. 2015
  CIFAR-adapted ResNet-20 / 56 / 110 ≈ 91.3 / 93.0 / 93.4–93.6%; WRN-28-10 ≈ 96.1%
  (Zagoruyko & Komodakis 2016); DenseNet-BC (L=190, k=40) ≈ 96.5%; VGG-16 with BN and the
  standard recipe ≈ 93–94%; an ImageNet-style ResNet-18 adapted to 32×32 (3×3 stem, no
  maxpool) ≈ 95% with a 200-epoch SGD + cosine + crop/flip recipe (the widely used
  `pytorch-cifar` reference gives ResNet-18 ≈ 93.0%, DenseNet-121 ≈ 95.0%, VGG-16 ≈ 92.6%).
  The "ResNet-18 at 82.9%, ResNet-50 at 76.0%" numbers come from a paper that almost
  certainly used the unmodified ImageNet stem (7×7/s2 + maxpool) on 32×32 inputs, or
  trained briefly; they are not baselines, and "deeper overfits CIFAR" is the wrong lesson
  from them.
- The Barz & Denzler "9–14% drop" is misquoted: the drop they report is on the
  near-duplicate *subset*; on the full corrected test set (ciFAIR-10) the change is small.
  Recht et al.'s 4–10% CIFAR-10.1 drop is correct as stated.
- Source lists from Danielle's follow-up structured searches (all unvetted; kept so the
  report's provenance is recoverable). Base query ("baseline"/"vanilla" accuracy, the five
  model families, arXiv or GitHub, raw links only): arXiv 2306.07613, 2501.04700, 2501.01402,
  2406.14082, 2402.05521, 2306.12100, 2502.00663, 1908.07086, 2210.06583, 2309.07537,
  2308.16258, 2405.14669, 2305.00097, 2410.05871, 2403.17833, 2307.13078, 2408.10359,
  2310.04414; GitHub `KaimingHe/resnet-1k-layers` (the only original-author source in the
  list — the identity-mappings CIFAR code), `Microsoft/LQ-Nets`, `innaprop/innaprop`,
  `LINs-lab/ReLA`, `sreetamasarkar/rlnet`, `poloclub/robust-principles`, `FlaAI/DUCAT`,
  `Tongzhou0101/NNSplitter`, `jjzgeeks/GradCAM-AE`; plus ~10 IEEE/ACM documents and several
  minor-journal items (federated learning, quantization, analog hardware, a retinal-disease
  paper) that are off-topic.
- `+ title:"ResNet-18" + year:2020..2025`: arXiv 2306.12100, 2502.00663, 2205.12141,
  2209.01848, 2304.01910. A rerun of the same query returned a larger set ignoring the year
  filter: 1912.05831, 1905.11946 (EfficientNet), 1806.00451 (Recht), 1908.07086, 2110.09468,
  **2110.00476 ("ResNet strikes back", Wightman et al. 2021)**, **2103.07579 ("Revisiting
  ResNets", Bello et al. 2021)**, 2302.04638, **1812.01187 ("Bag of Tricks for Image
  Classification with CNNs", He et al. 2019)**, 2111.12273, 1905.00546, 2306.13092,
  1603.08029 (ResNet-in-ResNet), 2407.05440, 2312.10948, 2004.04989, 2411.12874,
  2009.08453, 1811.07270; GitHub `RICE-EIC/FracTrain`. The three bolded are exactly the
  *training-recipe* ablation papers the ladder is a small-scale cousin of — each isolates
  recipe components (schedule, augmentation, regularization, label smoothing, mixup, EMA…)
  at fixed architecture and reports recipe ≫ architecture at a given budget. They belong in
  the ladder's related work ahead of any CIFAR baseline survey.
- `+ ("under 100 epochs" OR "fast training")` (with the ResNet-18 title filter): arXiv
  2205.12141, 2306.12100, 2304.03486, 2304.01910, 2405.14669; `iduta/iresnet` (IEEE 9412193),
  `ksouvik52/DNR_ASP_DAC2021`, `jjzgeeks/GradCAM-AE`. A rerun added 2306.07613, 2405.18320,
  2406.14082, 2402.01114, 2410.12604, 2501.04700, 2305.00097, 2502.09822, 2405.14033,
  2501.01402, 2309.15328, **2404.00498 with `KellerJordan/cifar10-airbench`**, 2401.16732,
  2204.13650, 2307.13078. Without the title filter: 2205.02551, 2105.10879, 2110.00476
  (again), 2309.01694, 1806.00250, 2209.12839, 2111.09451, 2210.06583, 2003.12862,
  2011.14498, 2402.11857, 2210.04532, 2407.20020, 2406.04070, 1812.01187 (again), plus
  overlaps; no GitHub links.
- The short-budget references actually comparable to 25-epoch runs are the speed-run
  lineage — David Page's DAWNBench ResNet-9 (94% in ~24 epochs / 79 s), `tysam-code/
  hlb-CIFAR10`, and Keller Jordan's `airbench` (94% in seconds; 96% variants; the one item
  the searches did surface) — which give per-budget accuracy targets directly. From memory;
  verify.
- For the ladder: the pilot's ~93.9% at 25 epochs (SGD, no mixup) is consistent with a
  ResNet-18-class model on a short budget — roughly 1–1.5 points under the 200-epoch
  reference — and the baseline's 76% is far below any published number for that model
  class at any reasonable budget, which supports the short-budget-underfit reading of the
  full-modern-recipe rung. If the ladder is revived, pin each architecture rung's expected
  accuracy from the original paper, `pytorch-cifar`, or airbench at the *same* budget, not
  from surveys like this one.

### Undated (~2025) — Choosing optimizer arms for the ladder (Perplexity, four questions)

Danielle's prompts (verbatim):

> What optimizers besides Adam W and SGD M are frequently used when training CNN based
> vision models

> If I had to choose between RMSProp and AMSGrad to test alongside SGD-M and ADAM-W to get
> a sense of different optimization dynamics of training CNN based vision architectures
> based on optimizer selection, which should I choose and why?

> Is AMSGrad implemented in pytorch?

> Is RMSProp implemented in pytorch

So the ladder's optimizer rung was being designed as a small *arm set* — SGD-M and AdamW
plus one or two contrasting optimizers — chosen for **different optimization dynamics**,
not for best accuracy.

**Responses (condensed).** (1) A blog-sourced list: RMSprop, AdaGrad, AdaDelta, Adam,
AdaMax, AMSGrad, Nadam, RAdam, AdaBelief, diffGrad, AdaNorm, plus cyclic LR/momentum,
YellowFin, and RPROP variants, each with a one-line blurb. (2) Recommends **AMSGrad** over
RMSProp: "stronger convergence guarantees," "less sensitive to learning rate," "escaping
poor local minima," "one-line modification"; against RMSProp: no momentum by default, no
bias correction, "less effective with batch normalization." (3) `torch.optim.Adam(...,
amsgrad=True)` — correct. (4) `torch.optim.RMSprop` with `alpha`, `momentum`, `centered`
— correct.

**Intake notes.**

- The "frequently used" list is wrong about practice. For CNN image classification the
  optimizers actually in use are SGD-M (dominant in every recipe paper), AdamW (the
  ConvNeXt / modern-recipe default), RMSProp (the TensorFlow Inception/EfficientNet
  lineage), LARS/LAMB (large-batch), and more recently Lion, Adan, and SAM-as-wrapper;
  AdaGrad/AdaDelta/AdaMax/diffGrad/AdaNorm/RPROP are not "frequently used" for vision.
- **The AMSGrad recommendation is poor for the stated goal.** AMSGrad's max-of-second-moment
  fix addresses a convergence counterexample (Reddi et al. 2018); in practice its
  trajectories are nearly indistinguishable from Adam's on image classification, so it adds
  an arm with almost no dynamical contrast to AdamW — the opposite of what Danielle asked
  for. The "escapes local minima," "less LR-sensitive," and "RMSProp is less effective with
  BN" claims are unsupported. If the question is *different dynamics*, the contrasting arms
  are: RMSProp (a genuinely different preconditioner history — no bias correction, and the
  optimizer that trained EfficientNet), **Lion** (sign-based update — categorically different
  step geometry), **SAM** wrapped around SGD-M (different objective, not just different
  preconditioner), and optionally LARS for a layer-wise-scaled arm. A minimal contrast set
  is {SGD-M, AdamW, Lion, SAM-SGD}; {SGD-M, AdamW, RMSProp} is the historically faithful
  set for a ladder that also reproduces old recipes.
- Whatever arm set is chosen, each arm needs its own LR sweep at the ladder's budget
  (the mid-ladder `lr-0.05` confound above is what happens otherwise); compare at
  tuned-LR-per-arm, and record the eNTK / sharpness readouts (`../topics/reference/ntk-literature.md`)
  per arm — those are where "different dynamics" would actually show.

## Intake notes (scoping response)

- The response ignored "ask questions first" and answered at the level of **architectures**,
  whereas Danielle asked for **building blocks** "including things like residual connections,
  optimizer and initialization improvements." A building-block ladder would separate: ReLU;
  He/Xavier init; BN (and later LN/GN); residual connections; 3×3 stacking / bottlenecks;
  SGD+momentum vs. Adam (Adam is *not* the standard for CNN image classification — SGD with
  momentum + weight decay is, so the response's optimizer claim is wrong for this domain);
  LR schedules (step → cosine → one-cycle/warmup); augmentation (crop/flip → cutout/mixup/
  RandAugment); label smoothing; dropout vs. stochastic depth. EfficientNet's "compound
  scaling + NAS" is a search result, not a building block, and is a poor ablation rung.
- Figure hygiene: the response mixes top-1 and top-5 numbers in one progression (AlexNet
  top-5 "84.7%" vs. later top-1 numbers) and the "10.8% improvement" is the top-5 *error*
  drop (26.2% → 15.3%), not an accuracy gain; "SENet surpassing human-level" is a
  blog-grade claim; "ImageNet 2010 winner 52.9%" is top-1 while 2011's "74.2%" is top-5.
  Every citation is a blog/LinkedIn/Wikipedia page. Do not reuse any number from this
  section without checking the original paper.
- What is worth keeping from the response: the with/without-skip and with/without-BN
  contrasts with gradient-flow and LR-tolerance readouts — those are exactly the
  "landscape metrics" the requirement asked for, and they are the rungs where the landscape
  literature (`../topics/reference/landscape-literature.md`) already has measurements to compare
  against (sharpness, mode connectivity, loss-surface smoothing by skips/BN — Li et al.
  2018 visualizations; unverified here, check before citing). Candidate eNTK readouts per
  rung are listed in `../topics/reference/ntk-literature.md`.
- The rung definitions this Q&A was feeding: the classic recipes differ on *several* axes
  at once (optimizer, WD value, dropout placement, LRN vs. BN, pooling overlap, augmentation
  set, schedule). A clean ladder must hold the training recipe fixed at a modern baseline
  (SGD+momentum, WD, cosine, crop/flip) and vary one block per rung, rather than reproducing
  each paper's full historical recipe — otherwise rung differences are confounded exactly
  the way the research hypothesis (`../research-hypothesis.md`) says cross-era
  comparisons are.
- The loss-slope study (EDP lineage) was the first analysis run on this ladder's
  infrastructure; if the ladder is revived, the early-window-features → final-accuracy
  question can be asked per rung, which would make it a CNN-scale replication setting for
  EDP.

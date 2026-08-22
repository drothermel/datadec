# CNN deconstruction ladder — ablating the building blocks of modern CNNs backwards to LeNet

**Kind:** staging. Candidate exits: a standalone project doc (a learning-and-measurement
project outside DataDecide: program pillars served: none directly, but it is the CNN-scale
sibling of `../../potential-projs/landscape-geometry.md` and
`../../potential-projs/tiny-scale-measurement.md`, and the run infrastructure it drove
(`dr_exp`/`deconCNN`) is the lineage of `../../potential-projs/early-dynamics-prediction.md`);
or archival as a completed/past learning project. Gate: Danielle says whether this is still
live — the 2025 seven-track record (`../reference/project-approach-principles.md`) shows it
at "Stage 3 integration & testing" in ~spring 2025, and nothing later is on file.

Sources: two Perplexity conversations (undated, ~late 2024 / early 2025; intake 2026-08-22).
The requirement statement is in `../reference/experiment-tooling.md`; the scoping prompt and
answer are below. Danielle's prompts carry the content; the answer is a blog-sourced
timeline whose figures are unverified.
---

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

## Intake notes

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
  literature (`../reference/landscape-literature.md`) already has measurements to compare
  against (sharpness, mode connectivity, loss-surface smoothing by skips/BN — Li et al.
  2018 visualizations; unverified here, check before citing).
- The loss-slope study (EDP lineage) was the first analysis run on this ladder's
  infrastructure; if the ladder is revived, the early-window-features → final-accuracy
  question can be asked per rung, which would make it a CNN-scale replication setting for
  EDP.

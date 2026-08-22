# Loss-slope prediction (CIFAR-10, 2025) — past project record

**Status:** started 2025-06, not finished. Analyses were run on the CNN-ladder dataset
(`cnn-deconstruction-ladder.md`; ~34 runs of 25 epochs, 7 "controlled" + 27 "sequential")
through external-agent conversations and fed a presentation; the per-run regression table
and raw per-epoch metrics exist as `regression_analysis_from_first_25_epochs.csv` and
`epoch_metrics_long_format.csv` (not on file here). Superseded in ambition by the July-2025
DataDecide draft that became `../potential-projs/early-dynamics-prediction.md` (EDP), for
which this is the CNN-scale lineage.

**What it was.** Test the (advisor-supplied) hypothesis that more *linear* loss curves —
loss on a linear y-axis against log steps — indicate better training: fit a regression
line on three windows (full curve, first half, the very beginning) and relate the fit's
slope and R² to final validation accuracy and to how well tuned the configuration is.

**What was found.** (i) The 4-epoch validation-loss slope is the best single predictor of
final accuracy (|r| = 0.71), decaying to 0.36 for the full-curve slope — the *early window*
carries the signal. (ii) Direction, once the sign was resolved: faster early decline →
higher final accuracy (slope bins monotone: < −0.6 → 92.2%, > −0.25 → 81.7%); an agent's
"steeper predicts worse / overfitting" reading was a sign misread. (iii) Higher R²
(linearity) ↔ *lower* accuracy (r ≈ −0.4): at this budget linearity indexes slowness,
which cuts against the hypothesis. (iv) Neither literature search found any established
belief that linear curves mean better training; the documented shape priors are power-law.

**What is unresolved.** Statistics treat seeds as independent (34 runs from 15 configs);
clusters on the slope–accuracy plot are augmentation regime × a mid-ladder LR change, so
the slope's signal after conditioning on LR is unknown; n does not reconcile between the
two analyses; "tuning quality" as a target was never operationalized.

**Where it points now.** EDP §1 (three windows × fit types is this feature family) and
EDP §4 (lineage + the flag that the linearity premise is untested); the early-curve
related-work lines in `../topics/reference/loss-curve-forecasting.md` (Domhan 2015, Baker
2017, LC-PFN, Ding 2024; neural capacitance) and `../topics/reference/nas-literature.md`
(zero-cost proxies). If the data is found: redo with per-configuration means or a mixed
model, colour by (LR, augmentation regime, rung), and report slope signal net of LR.

---

## Record (dated entries; Danielle's statements verbatim, responses condensed)

The first-pass ablation reading that preceded these analyses is in
`cnn-deconstruction-ladder.md` ("2025-06-12 — First ablation dataset").

## 2025-06-12 — The linearity question, stated precisely

Danielle's follow-up (verbatim):

> These are the results from training CNN vision models on cifar data while removing major
> advancements from the last many years. The hpms are not well tuned, and the training
> loss / acc are evaluated on the augemented data.
>
> I'm interested in the linearity of the loss curves when you plot them in linear(y) vs
> log(x) space. Specifically, if you fit a regression line to the full curve, to just the
> first half too the very beginning, how does the linearity of the loss curve relate to the
> final validation accruacy? how does it relate to how well tuned the setting and hpms are?

This is the exact specification of the loss-slope study's feature (and the origin of EDP's
three-window × fit-type feature family in `../potential-projs/early-dynamics-prediction.md`
§1): loss on a linear y-axis against **log steps**, a linear regression fit on three
windows — full curve, first half, "the very beginning" — and the fit's linearity (R²) and
slope related to (a) final validation accuracy and (b) how well tuned the configuration is.
Note the second target: *tuning quality*, not just final accuracy — i.e. linearity as a
diagnostic of whether the recipe is in a good regime, which is the advisor's hypothesis
restated as a measurable claim (see `../topics/reference/loss-curve-forecasting.md`). She also
states up front that the hyperparameters are not well tuned.

**The response.** Admits in its first paragraph that the dataset it was given "contains only
summary statistics (final training and validation metrics) rather than epoch-by-epoch
training curves," then proceeds anyway: it invents a "convergence quality score" (from
final validation loss), a "stability score" (seed variance), reports r = 0.719 between the
former and validation accuracy and r = 0.508 for the latter, r = −0.016 for log final
train loss, and concludes that "loss curve linearity in log-linear space serves as a
reliable indicator of both final performance and hyperparameter tuning quality" and that
the top configurations "likely exhibit linear loss curves." Its useful content is one
paragraph: the methodology it recommends (per-epoch loss for every config × seed; fits on
full / first-half / first 10–20%; compare R² and slope across windows against final
accuracy; use early-window fit quality as a tuning/early-stopping signal) — which is a
restatement of Danielle's own question.

Intake note: the conclusion is unsupported — no curve was ever fit, and the proxies are
final-value statistics that say nothing about curve shape; "convergence quality ↔ accuracy"
is final val loss vs. final val accuracy, which is close to a tautology. Treat this
response as having zero evidential weight on the linearity hypothesis. What survives is
Danielle's specification, which should be the canonical definition of the feature when the
per-epoch data is located: fit $L = a + b\log t$ on each window, record $(b, R^2)$ per
config × seed, and regress final validation accuracy — and, separately, a tuning-quality
label — on them. The "tuning quality" target needs an operational definition (distance
from a tuned LR? rank within an LR sweep?); it is not in the data as described.

## 2025-06 — Second attempt, with per-epoch data: the slope / R² results

Danielle re-ran the same request ("the same intro as last time") with the per-epoch data
attached. The uploaded file was `regression_analysis_from_first_25_epochs.csv` (a
pre-computed per-run regression table over 25 training epochs — so the runs were 25 epochs
long and she had already fit the window regressions herself; this filename is the handle
for finding the dataset). The response reports, with n ≈ 34 runs (7 "controlled" + 27
"sequential steps"):

| Feature (validation loss, linear-y vs. log-x fit) | r with final val accuracy | p |
|---|---|---|
| slope, first 4 epochs | −0.710 | 3e-6 |
| slope, first 10 epochs | −0.599 | 2e-4 |
| slope, first 25 epochs (full) | −0.359 | 0.037 |
| R², validation curve | −0.418 | 0.014 |
| R², training curve | −0.338 | 0.051 |

Plus: "predictive power peaks in the first 3–5 epochs and then steadily declines";
controlled runs 0.918 ± 0.017 (n = 7) vs. sequential 0.875 ± 0.061 (n = 27).

**The response's reading.** "Steeper early validation loss decline predicts *worse* final
performance" — a "paradox"; fast early learners are "heading toward overfitting"; "higher
linearity … predicts worse final accuracy"; the controlled-vs-sequential gap shows "the
importance of consistent experimental conditions"; recommends monitoring the 4-epoch slope
as an early-warning system and preferring "moderate, steady progress."

**Intake notes.**

1. *The sign was misread* — settled by the cluster follow-up below: slopes are signed and
   the most-negative bin has the highest accuracy, so r(slope, accuracy) = −0.71 means
   **faster early decline predicts better final accuracy**, the intuitive direction, and
   the one consistent with the first-pass table (SGD / no-mixup / no-warmup both drop
   fastest and finish highest at this budget). The "paradox" and the overfitting narrative
   rest on reading the slope as a magnitude. The Heptabase-era summary ("validation-loss
   slope anticorrelates with final accuracy") is the raw signed r and is correct as a number.
2. *The window result is the real finding.* |r| falling from 0.71 (4 epochs) to 0.36 (25
   epochs) says the early-window slope carries more information about the endpoint than the
   full-curve slope — with 25-epoch runs, "the very beginning" window is the predictive one.
   That is the EDP premise at CNN scale, and it is the result worth carrying forward.
3. *R² result is direction-safe and interesting.* Higher linearity (in linear-y / log-x)
   ↔ lower final accuracy, r ≈ −0.4. Under this budget the good runs drop fast and then
   flatten (non-linear in this space); the poor runs decline slowly and steadily. So
   "linearity" here indexes *slowness*, which cuts against the hypothesis under test
   (`../topics/reference/loss-curve-forecasting.md`) — at least at short budgets and with untuned
   hyperparameters, as Danielle stated up front.
4. *Statistics are overstated.* The 34 "runs" are 15 configurations × seeds; seeds within a
   configuration are not independent draws for a configuration-level claim, so the p-values
   are optimistic. Report per-configuration means (n = 15) or a mixed model with a
   configuration random effect. The controlled-vs-sequential comparison is a comparison of
   *which configurations* are in each set, not of methodology; the response's "experimental
   consistency" reading is unsupported.
5. *No literature was cited for the "overfitting within 3–4 epochs" claim* ("research
   confirms …" with no source); drop it.

## 2025-06 (continued) — Clusters on the slope–accuracy plot; the sign question settled

Danielle's follow-up (verbatim):

> Within your findings about the correlation of smoothness with accuracy or the correlation
> of linearity / early loss fit line slope with linearity, are there different clusters of
> hpms that behave differently from each other? For example, on the smoothness vs accuracy
> plot it seems that there are clusters, do they correspond to specific hpms?

(She was looking at the plots herself and saw clusters; a second file,
`epoch_metrics_long_format.csv`, was attached this time — the raw per-epoch metrics in long
format — so both the raw curves and the regression table existed as files. The response
mentions "your presentation," so these analyses were feeding a talk.)

**What the response reports (by validation-slope bin, 4-epoch window):**

| Slope bin | n | mean acc | std | composition |
|---|---|---|---|---|
| very steep (< −0.6) | 9 | 0.922 | 0.017 | 7 of the 7 "controlled" runs + 2 |
| steep (−0.6 to −0.4) | 14 | 0.889 | 0.056 | all sequential |
| moderate (−0.4 to −0.25) | 7 | 0.859 | 0.066 | sequential; architectural changes |
| gentle (> −0.25) | 4 | 0.817 | 0.028 | sequential; all `lr-0.05` |

Sub-clusters named: "controlled" runs (fixed augmentation set; slopes −0.73 to −0.60; 0.918
± 0.017) vs. "sequential steps" (slopes −0.68 to −0.16; 0.875 ± 0.061); default-LR runs
(slopes −0.6 to −0.4, mean 0.874) vs. `lr-0.05` runs (the four gentlest slopes; mean 0.889,
high variance); and a low cluster of `step11_tanh_lr-0.05`, `step12_no_colorjitter_lr-0.05`,
`step13_no_rrc_lr-0.05` (0.817 ± 0.028). Within sequential runs alone r = −0.555. The
response then explains the clusters as "experimental consistency" vs. "confounding
variables" and keeps the "steeper predicts worse" headline, while writing "high performance
despite steep slopes" in its own table.

**Intake notes.**

1. *The sign question is settled by this table.* Slopes are signed (all negative); the most
   negative bin has the highest accuracy (0.922) and the least negative the lowest (0.817),
   monotonically. So r(slope, acc) = −0.71 means **faster early validation-loss decline →
   higher final accuracy.** The response's "paradox" was a sign misread, and this reply
   contradicts it without noticing. Record the direction as: *at a 25-epoch budget with
   untuned hyperparameters, early speed of validation-loss decline is the best single
   predictor of final accuracy, in the intuitive direction.*
2. *The clusters are the ladder's rungs, not "methodology."* The "controlled" set is the
   7 configurations run with a fixed augmentation set; "sequential" are the cumulative
   removal steps; and the later steps (`step11`–`step13`) were run at `lr-0.05` — i.e. the
   learning rate was changed partway down the ladder. The clusters on her plot are
   therefore (a) which augmentation regime and (b) which LR, confounded with rung order.
   The response's "controlled experiments eliminate confounding variables" reading is
   backwards: the LR change *is* the confound, and the low cluster is "tanh / no-colorjitter
   / no-RRC **at a different LR**," which cannot be attributed to the removed feature.
3. *Useful structure that survives.* Slope bins are monotone in accuracy across all 34
   runs, and the within-sequential r = −0.555 shows the relationship is not just the
   controlled-vs-sequential gap. For the predictor question this is encouraging: the
   early-window slope ranks configurations even inside one regime. For the ablation
   question it says the LR must be re-tuned (or at least held fixed) per rung before any
   rung's effect is read — which is the "tuning quality" target Danielle asked about
   earlier, now visible as a cluster variable.
4. *Redo list addition.* Colour the slope–accuracy scatter by (LR, augmentation regime,
   rung); fit the slope→accuracy regression with those as covariates; report whether the
   slope still carries signal after conditioning on LR. Resolve n: 34 runs vs. "15
   configurations × 3–6 seeds" from the first pass do not reconcile (15 × 3 = 45 minimum),
   so some configs or seeds were dropped between the two analyses — check against
   `epoch_metrics_long_format.csv`.


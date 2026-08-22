# Research hypothesis — Danielle's own statement, and its refinements

## The program's apex question (decided 2026-08-22)

Claimed training-history effects — critical periods, warm-start scars, non-stationarity
memory, recipe effects on elicitability — are confounded with regime-mismatched defaults and
uncontrolled elicitation. Build the measurement framework (calibrated elicitation,
replicated small-scale interventions, identifiability-aware comparisons) that separates
real path-dependence from measurement artifact, and use it to find where weight updates
beat tuned elicitation and how the balance shifts with scale.

This merges the hypothesis below (the evaluation-side half) with the thesis arc from
the Research Trajectory page (the training-history half). Three pillars — measurement
science at academic scale (*how*), non-stationarity accounting (*mechanism*), data
measurement → training dynamics (*independent variable*) — are laid out in
`potential-projs/README.md` → Program.

This is the organizing claim behind the project portfolio, in Danielle's words, followed by
the refinements agreed in discussion (dated, quoted close to verbatim, not decisions). The
potential-project docs are instruments for it; the candidate program framings in
`potential-projs/README.md` are alternative emphases over it.

---

## The hypothesis (Danielle, 2026-08-18)

> The way we're doing "evaluation" for these large multistage pipelines, where there are SOO
> many different knobs that clearly effect things but we cant individually test, is so broken
> that its very improbable we could separate out failed "transfer" due to using training
> approaches tuned aggressively to training from a random initialization vs due to an
> impossibility of transfer. And I think our way of evaluating means elicitation approaches
> mask a lot of the underlying signal about different interventions. So let's understand
> how to elicit effectively in different situations and then use *those* evals to uncover
> the training procedures that allow weight updates in phases that don't start from random
> initialization to succeed more than prompt optimization, ICL, etc.

**Restated (agreed).** "Multistage pipelines have accumulated defaults — LR schedules,
warmup, decay shapes, optimizer settings, data curricula, eval protocols — that were all
tuned in the from-random-init regime. When a warm-started phase (continued pretraining,
post-training, transfer) 'fails,' we therefore can't distinguish 'transfer is
impossible/limited here' from 'we ran a from-scratch recipe on a warm start.' And our evals
compound this: because elicitation is uncontrolled — every measured capability arrives
through some prompt/format/ICL pathway whose efficiency varies across models — intervention
effects get filtered through an unmeasured elicitation transfer function before we see
them. So the program is: first characterize and *control* elicitation per model-task (making
it a calibrated instrument and a tuned null model), then use elicitation-controlled evals to
search for warm-start-native training procedures, judging them by their delta over the
tuned elicitation ceiling."

"So I was wrong that you 'switched sides': you're not betting *on* gradient-free
elicitation, you're operationalizing your suspicion of it — promoting it from confound to
control condition."

**Lineage.** "The hypothesis is your 2021 paper twice over, applied to both halves of the
comparison. In 2021, the frozen-vs-finetuned comparison was broken because one arm got fair
tuning and the other didn't. Your current claim is that *both* arms of today's comparisons
are untuned: the weight-update arm inherits from-scratch hyperparameters (the
ELR/warm-starting results, DASH, and shrink-perturb are all existence proofs that re-tuning
the regime-mismatched knob dissolves apparent transfer failures — and the position paper
arguing 'lifetime tuning is incompatible with continual RL' makes the same complaint from
the RL side), and the measurement arm inherits untuned elicitation (Spurious Rewards and the
pass@k results are existence proofs that elicitation variance masquerades as training
effects). There's even a pre-2021 ancestor for the finding-shape: Melis et al. 2018 showed
LSTM-vs-transformer conclusions inverted under equalized tuning budgets. Your hypothesis
generalizes that to: *every* pipeline-stage comparison currently published is a Melis-2018
waiting to happen, and the elicitation layer is the least-tuned knob of all."

**What it changes in the thesis framing.** "The ICL/elicitation protocol isn't an outcome
measure anymore — it's *instrument calibration*. You're building the measurement theory
(how does elicitation efficiency vary across models, tasks, and training histories; how
many samples/orderings does a stable elicitation estimate need) so that the later
experiments can report 'capability change net of elicitation change,' which is the quantity
everyone currently conflates. And the warm-start factorial becomes the oldest, smallest
instance of the second half: re-tune the regime-mismatched knobs and watch how much
'impossibility' evaporates. The dissertation claim underneath: most published
transfer/continual/post-training conclusions are statements about *recipe-regime mismatch
plus elicitation variance*, not about model capability — and here is the evaluation
framework that separates the three."

**Two demands on the design.**
1. *Tuning-budget accounting* — "'tuned elicitation vs. tuned weight updates' is only
   meaningful under matched search budgets… or you've recreated the asymmetry you're
   diagnosing, just mirrored." (Revised below.)
2. *Anti-circularity* — "if elicitation efficiency is itself shaped by training history…
   then 'net of elicitation' risks subtracting out part of the effect you care about — so
   the framework needs both readouts, raw and elicitation-controlled, with the *difference*
   between them reported as its own measured quantity: how much of this intervention's
   effect is capability vs. accessibility. That decomposition — **capability delta vs.
   accessibility delta** — might actually be the cleanest one-line statement of what your
   whole program measures, and no existing evaluation framework provides it."

---

## Refinement 1 — matched budgets are impossible and misleading (Danielle, 2026-08-18)

> My real belief is basically that none of the training approaches that we currently have
> are theory based for the most part. All of them have been determined by guess and check
> trial and error across a huge community, who then double down on things that seem to
> work. Even if I was to throw a huge amount of compute at the elicitation budget, it still
> wouldn't be anything compared to the amount of compute that has gone into tuning the
> training-from-initialization approach. That's the issue we're trying to get at: we look
> at these individual experiments as if we need to control across these two different
> approaches, but you're not really controlling, because one has been aggressively tuned
> over more than a decade and the other hasn't really been substantially explored — and for
> the most part it's been explored from the foundation of the heavily tuned one. A
> controlled budget comparison is inherently impossible to make, and trying to make it would
> skew you towards doing the wrong thing.

**Agreed, with a named precedent.** "The closest named version is Sara Hooker's 'hardware
lottery' argument… Your version is sharper because you're pointing at the *software* stack:
init schemes, Adam's betas, warmup conventions, decay shapes, data curricula, even eval
formats are all crystallized search output from the from-scratch regime. A 'matched
budget' comparison then isn't paradigm-vs-paradigm; it's (incumbent + decade of communal
prior + small search) vs. (challenger + zero prior + small search)… Worse, the challenger's
search is usually *initialized at the incumbent's optimum*… Agreed: matched budgets as a
*verdict criterion* would systematically favor incumbents and steer you wrong."

**The pushback on the pushback: unfalsifiability.** "If elicitation or warm-start-native
training underperforms, 'it just hasn't had its decade yet' is always available — and that
move is exactly the one you'd (rightly) reject from someone defending a fading paradigm. Any
critique powerful enough to dismiss all unfavorable comparisons is too powerful to build a
thesis on. So the question isn't 'match budgets or don't'; it's what *replaces* the
matched-budget comparison as the falsifiable object."

**Proposed replacement: compare response curves, not points.** "Report performance as a
function of search budget for each paradigm and look at the shapes. A mature,
communally-exhausted paradigm should show a flat curve — marginal search buys nothing,
because the community already spent that gradient. An under-explored paradigm with real
headroom should show a steep, still-rising curve. That's the falsifiable version of your
claim: 'warm-start-native training is under-tuned' predicts steep tuning-response curves
where from-scratch recipes show flat ones, and if the challenger's curve is *also* flat at a
low ceiling, your hypothesis is wrong for that method and you've learned so cheaply… you're
no longer trying to equalize histories, you're measuring each paradigm's *sensitivity to
marginal effort*, which is comparable regardless of unequal pasts."

**Two supporting measurements.**
- *Optimum displacement*: "for each pipeline stage, measure how far the warm-start regime's
  tuned optimum sits from the from-scratch default in hyperparameter space… every large
  displacement found is simultaneously evidence for the thesis and a usable recipe
  improvement."
- *Historical base rates*: "a modest meta-analysis of 'how often does the incumbent's
  advantage survive serious re-tuning'" over the field's natural experiments (Rothermel
  2021; Melis 2018; ELR/warm-starting; the Qwen-RLVR corrections) vs. cases where communal
  investment still failed — "a legitimately publishable piece… on its own" (a workshop-sized paper).

"Budget *reporting* stays (so readers can interpret), while the decision-relevant
quantities become tuning-response slopes and optimum displacements rather than
head-to-head points."

---

## Refinement 2 — existence proofs, not comparisons (Danielle, 2026-08-18)

> I'm still not convinced: even if you tune the new method aggressively and see how it
> responds, fundamentally the knobs and leverage you'd use were identified as potentially
> useful from the incumbent. So we really can't make this comparable. In my mind it looks
> more like an existence proof or impossibility proof — you just have to find one
> counterexample. The best we can hope for as a clear indication is a demonstration that
> elicitation can move things strongly, and then, using the elicitation evaluation, that we
> are able to get better in some setting from a weight update as opposed to elicitation
> alone. When you have such a strong competitor, you're not going to be able to do a matched
> comparison and you need existence proofs — clear demonstrations. This isn't completely
> correct/rigorous but the hand-wavy direction is meaningful.

**Agreed; the rigorous skeleton.** "Switch from *estimation* ('which paradigm is better, all
else equal' — impossible, since all else was never equal) to *demonstration* ('what is
possible' — which a single instance settles). Existence claims are one-sided: 'weight
updates exceeded the tuned elicitation ceiling in setting S' is established by one clean
instance and *cannot be un-established by any amount of tuning of the loser*… The asymmetry
that poisons comparisons doesn't touch possibility claims; it only poisons superiority and
impossibility claims, which your program can simply decline to make." Historical
precedent: AlexNet, GPT-3's ICL, R1-zero "were existence proofs, not controlled
comparisons"; shrink-and-perturb and continual backprop were explicitly framed this way.

**Three steps.**
1. "Demonstrations that elicitation moves things strongly — this is nearly established
   already (Spurious Rewards is accidentally the field's loudest such demonstration), and
   your protocol's job is to make it *quantitative per setting*: a measured, tuned
   elicitation ceiling."
2. "Demonstrations that some weight-update procedure exceeds that ceiling somewhere — one
   clean case proves weight updates access something elicitation can't, which is the
   load-bearing possibility claim."
3. "*Many* such demonstrations across settings trace the boundary. A lattice of existence
   proofs is how you recover generality without ever running a matched comparison… like
   building a phase diagram from points." The tuned elicitation ceiling is "the *strong null
   model*, and every demonstration is a one-sided test against the strongest available
   null."

**The warning, and the discipline that replaces budget-matching.** "Existence proofs are
*the genre most vulnerable to the measurement failures you diagnosed*. The RLVR literature
circa 2024–2025 was precisely a pile of existence proofs… A demonstration regime shifts the
failure mode from unfair comparison to multiplicity: if you search fifty settings for one
success, a 'demonstration' is a lottery ticket, and the file drawer hides the forty-nine.
So the discipline… is demonstration *hygiene*: pre-specified settings, effect sizes with
confidence bounds across seeds, replication in at least a second model family (the Qwen
lesson), honest reporting of how many settings were searched, and — the thing that
separates a demonstration from an anecdote — a mechanism readout from your diagnostic panel
explaining *why* the ceiling was exceeded there, which turns one point into a predictive
claim about neighboring points, testable cheaply."

**The standard.** "Would the person who wrote 'Don't Sweep Your Learning Rate Under the Rug'
be able to kill this result by tuning the null harder? You spent your first publication
destroying an under-tuned existence proof; the program you're proposing is to produce
existence proofs that survive that author."

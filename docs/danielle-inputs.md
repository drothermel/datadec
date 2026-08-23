# Danielle's inputs — prompt log

Chronological, verbatim record of Danielle's own prompts and framing statements from the
external conversations whose responses are consolidated into this repo. One entry per prompt:
date, conversation, the prompt itself, and where the response was routed. The point is to make
the full set of her thoughts findable in one place; responses live wherever they were routed.

The prompts behind the "top-N" lists and the "general thoughts" sections were not captured
verbatim; add them here if recovered.

---

**How to read the archived conversations (Danielle, 2026-08-22):** she did not take the
agents' answers and use them; the conversations were a way to see how well different
agents worked, to document her own thoughts, and to pick up occasional pointers. Her
prompts are the content of record; the intake notes grading each response are part of the
agent comparison, not corrections to decisions she made.

## 2026-08-21

### Beyond DataDecide: generalizing dataset featurization

> ok, so I think if we move out of the datadecide specifically world, there's space for
> dataset featurization/analysis on these super large pretrain (or midtrain/posttrain, etc)
> dataset and their impacts. would you agree or not?

Response routed to: [potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md);
project-specific parts to [potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md) §4.

### Chunked datasets, contribution types, and stage-dependent effects

> Separate question. Going back to the investigation of datasets more broadly and their
> features. So I think that I was less interested in, like, these high level fairly ambiguous
> descriptors like quality or, I guess, deduplication is a very specific figure. Um, and
> instead, I'm interested in things that are more like and we, you know, chunk the datasets
> and then look at... and somehow identify different types of contributions to a dataset and
> look at how those contributions, um, are distributed and how they clearly move some types of
> metrics or the loss landscape when they're applied at different stages in training. Um,
> that's more what I was imagining. is that a thing?

Response routed to: [potential-projs/functional-featurization.md](potential-projs/functional-featurization.md)
(the origin entry); project-specific parts to
[potential-projs/wsd-suite.md](potential-projs/wsd-suite.md) §4 and
[potential-projs/landscape-geometry.md](potential-projs/landscape-geometry.md) §4.

### Short continuation from a checkpoint as a landscape probe

> perhaps even directly in relation to ideas like river valley or the loss landscape or same
> vs different basin models, something where we could take an intermediate checkpoint and
> continue pretraining for like 1/16th of the run length or less and then measure a statistic
> and point at movement. does that exist?

Response routed to: [topics/checkpoint-tomography.md](topics/staging/checkpoint-tomography.md);
project-specific parts to [potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md) §4
and [potential-projs/landscape-geometry.md](potential-projs/landscape-geometry.md) §4.

Standing note (same date): claims about related work in these responses should not
be taken as true — the agent did no extensive searches and has no innate knowledge of the
current landscape; the ideas are useful nonetheless.

### Combined prompt: beyond DataDecide + chunk types + short-branch probes

> Lets think more generally than DataDecide specifically: where is the space for dataset
> featurization/analysis on these super large pretrain/midtrain/posttrain datasets and their
> impacts?
>
> I'm less interested in super heuristic measures like duplication, or very ambiguous
> descriptors like "quality." Instead I'm interesting in something closer to chunking the
> datasets and identifying the types of the different chunks. Then potentially tying those
> different types to contributions like moving some metric or changing the loss landscape,
> etc, when applied at different stages. Specifically I'm imagining something like the
> river-valley hypotheses, same basin concepts, etc. But alternatively something like
> plasticity and other features that relate to continual learning and what is changing.
>
> I'm imagining that we could take an intermediate checkpoint and continue pretraining for
> like 1/16gh of the run length or less and then measure a statistic and analyze the
> direction movements or something. What do you think of this direction in relation or in
> contrast to some of the existing ideas?

This is the single-prompt refinement of the three prompts above.
Response routed to: [potential-projs/functional-featurization.md](potential-projs/functional-featurization.md)
— the combined-prompt entry there.

### MoE model releases as "someone else already did the work"

> what if we incorporate moe based model releases in this too for the "someone else already
> did the work" aspect?

Response routed to: [potential-projs/moe-partitions.md](potential-projs/moe-partitions.md);
project-specific parts to [potential-projs/trajectory-statistics.md](potential-projs/trajectory-statistics.md) §4,
[potential-projs/token-movement.md](potential-projs/token-movement.md) §4, and
[potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md) §4.

### Slicing-and-Dicing repo as apparatus for MoE experiments

> can you look at the slicing and dicing paper, regarding moes? im third author and we did
> that on our current compute and with a range of model sizes and a working repo. so im
> wondering if using that repo to run some additional exps in that range would be useful.
> for context its trending towards either weak accept or weak reject at neurips currently
> which id consider a win. but its lacking interesting analysis, im more interested in the
> analysis than the full empirical grid sweep for future projects.

Response routed to: [potential-projs/moe-recipe-suite.md](potential-projs/moe-recipe-suite.md) (only section 2
of the response was captured; it references an earlier "reweighting" of directions that is
also not captured); project-specific part to
[potential-projs/wsd-suite.md](potential-projs/wsd-suite.md) §4.

### MoE-style questions, free MoE artifacts, and the Slicing-and-Dicing codebase

> Then, what would it look like to ask similar styles of questions around MoEs? It seems that
> MoEs give much richer outcome channels and that their routing decisions could be used to
> categorize tokens. And the datasets and released models around MoEs provide an additional
> set of sources of "free" compute artifacts that could likely be combined or compared to
> some of these dense options?
>
> Additionally, if you look at Slicing and Dicing: https://arxiv.org/abs/2605.11689, this is
> a paper I'm the third author on so we have a robust codebase that enabled us to run the
> systematic study on our existing hardware + some guidance for hpm choice at the small model
> sizes. And, as context, the paper is trending towards weak accept/reject at NeurIPS, which
> is a win in my book and the level that I'd aim for with a full conference submission. But in
> the future I'd like to do something with more interesting analysis instead of just a large
> grid sweep.

Response routed to: [potential-projs/moe-partitions.md](potential-projs/moe-partitions.md);
project-specific parts to [potential-projs/trajectory-statistics.md](potential-projs/trajectory-statistics.md) §4
and [potential-projs/token-movement.md](potential-projs/token-movement.md) §4; the
sweep-reanalysis idea is also noted in [potential-projs/moe-recipe-suite.md](potential-projs/moe-recipe-suite.md).

### Slicing-and-Dicing checkpoints — availability (Danielle's own note)

> I checked with my collaborator and she does have all of the final checkpoints for the
> slicing and dicing paper and one of us will upload them to huggingface fairly soon so thats
> unblocked. there are also a range of additional exps she'll be doing so likely for some of
> them we can get intermediate logs, etc.

Recorded in: [open-questions-answered.md](open-questions-answered.md) (as a resolved gate),
[potential-projs/moe-partitions.md](potential-projs/moe-partitions.md), and
[potential-projs/moe-recipe-suite.md](potential-projs/moe-recipe-suite.md).

### IRT as a full-conference paper

> i feel like there has to be a path to turn something around IRT on datasets into a
> acceptance worthy full conference paper (which is what i personally meant by strong). if
> not in the nlp space then maybe in a continual learning /plasticity type of smaller scale
> space where the pitch is rigor and science more than large lab adoption?

Item 1 of a three-item response. Routed to:
[potential-projs/irt-reanalysis.md](potential-projs/irt-reanalysis.md) §4.

### Small-scale science, measurement signal, and sampling nonstationarity

> finally, very unsexy, but im interested in how far we can push models at really small
> scales, like 150M and below in datadecide for example. and how to even measure a training
> or hpm fitting signal is a big part of the problem there i think. also, theyre using such a
> small % of the data recipes that unless they use stratfied sampling throughout training
> they are likely getting real nonstationarity or not really hitting the percentages that
> they expect. so i feel like this is a space that would let academic labs, sciency
> questions that are about dynamics, local model runners, etc benefit even if the big labs
> wouldnt care. and at that scale i suspect you coukd do really cool things with
> elicitation, rl, multicomponent systems, exps with confidence intervals, etc because the
> models run so fast. thoughts?

Item 3 of the same three-item response (items: IRT full-conference path; Slicing-and-Dicing
repo; this). Routed to:
[potential-projs/tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md);
the sampling-nonstationarity point to
[potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md) §4 and an
open gate in [open-questions-answered.md](open-questions-answered.md); the synthesis to the
potential-projs README program-level notes.

### Small-scale direction, expanded (10M–150M; elicitation/post-training; dense vs. MoE)

> Finally, this next direction is less broadly interesting across industry, but im interested
> in how far we can push models at really small scales, like 10M-150M in the DataDecide suite
> for example. Both around elicitation of performance from the base models and around focused
> posttraining that is effective at this scale. A big question is how to even measure a
> training or hpm fitting signal, which I think is an interesting question to explore. And
> while big labs don't care, there is real value for the academic space, the sciency
> experiment space like the plasticity research, and for people who want to run local models,
> etc.
> Additionally, I think that training such small models is likely substantially impacted by
> design decisions being tuned for larger models. For example, they're using such a small % of
> the data recipes that unless they use stratfied sampling throughout training they are likely
> getting real non-stationarity or not really hitting the percentages that they expect. And I
> also think there's space for looking at dense vs moe at this scale because you start to see
> some very effective MoE models at very small (relative to frontier) active scales, so what if
> you go smaller?
> Finally, I suspect going down to that scale would let you do really cool things with
> elicitation, rl, multicomponent systems, exps with confidence intervals, etc because the
> models run so fast. That is, if you could get clear learning or reward signal and fit them to
> specific tasks.
> What do you think about this direction? How does it relate to the others or offer
> extensions/alternatives?

The expanded form of the previous prompt, from the other conversation. Routed to:
[potential-projs/tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md)
(second entry); project-specific parts to
[potential-projs/irt-reanalysis.md](potential-projs/irt-reanalysis.md) §4,
[potential-projs/trajectory-statistics.md](potential-projs/trajectory-statistics.md) §4,
[potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md) §4;
tiny-MoE part to [potential-projs/moe-partitions.md](potential-projs/moe-partitions.md); the
Drosophila composition point to the potential-projs README program-level notes.

### Non-stationarity in MoE routing, and as a cross-cutting subthread

> Interesting! on the non-stationarity piece, I've actually been thinking about how
> non-stationarity is handled in MoE models. specifically "The OpenMoE analysis found that
> routing decisions are predominantly based on token IDs with minimal context relevance, and
> token-to-expert assignments are determined early in pretraining and remain largely fixed."
> then, I would expect routing changes to lead to non-stationarity in training which would
> likely make things instable, and I wonder whether we've optimized the training setup to try
> to remove that instability which has had the effect of removing the ability or incentive for
> routing to update over training. I'm not confident enough in that very specific hypothesis
> to build a program around it, but it does seem like non-stationarity is a potentially large
> subthread that is playing out in many of these model training settings, not just the
> "continual learning" and "RL" spaces that often discuss them explicitly. thoughts?

Routed to: [topics/nonstationarity-accounting.md](topics/reference/nonstationarity-accounting.md);
the frozen-routing experiments also noted in
[potential-projs/moe-partitions.md](potential-projs/moe-partitions.md); the endogenous
self-curriculum point to [potential-projs/token-movement.md](potential-projs/token-movement.md) §4;
the accounting framing to the potential-projs README program-level notes.

### Three ranking prompts

> ok great, now, can you reweight the different project directions (including additional
> ones you might define) based on strongest project outcome from a 6-12 month long effort.
> main conference paper thats really strong vibes possible depending on how the results come
> out.

> Ok, given everything we've discussed, can you pull out the 10 strongest clearly separate
> workshop paper sized contributions, ordered with highest likelihood to come together
> quickly first + your justification for why? Put aside the question of shared infra for now.

> Then, can you give me a list of 10 weak accept tier or higher full conference submission
> projects (NeurIPS, ICLR, ACL, CoLLAs, etc) with as much distinction between them as
> possible (likely there will be some overlap because our idea space isn't quite large enough
> to support 10 non-overlapping directions, but do your best). Rank them by speed to produce
> (which includes the iteration time needed if its a less clear cut outcome or if there isn't
> an obvious pivot if the results don't go our way) but clearly label each by the "expected
> impact" and the "impact ceiling."

Responses routed to: [portfolio-rankings.md](portfolio-rankings.md) (all three lists, whole);
each project's or topic's own entries copied into its §4 / topic doc; portfolio-shape
observations into the potential-projs README program-level notes.

### Four main-conference projects from two workshop subs each

> ok, then, can you identify a set of four strongest/most plausible projects, aiming to be
> feasible and likely to be at least at the weak accept level+ projects for a main conf like
> NeurIPS, ICLR, ACL, etc, that have a core main paper project built off of two workshop
> submission sub-contributions (non archival of course). Again rank them by speed to produce,
> label them with the likelihood to be scooped soon, and with the expected impact and impact
> ceiling estimates. then recommend a place to start (this can be based on shared infra).

Response routed to: [portfolio-rankings.md](portfolio-rankings.md) (whole); P1/P2/P3 entries
to the IRT, recipe-featurization, and annealed-readouts §4s; P4 to
[potential-projs/moe-partitions.md](potential-projs/moe-partitions.md); the shared-foundation
starting recommendation to the potential-projs README program-level notes.

---

## 2026-08-18 — Research Trajectory Notion page (intake by dropdown, started 2026-08-21)

### Toggle 1 — Continual Learning: Plasticity + Multi-Power Laws

> During a previous research project i was interested in the continual learning space and
> specifically in whether features of a loss curve could be used to estimate or predict
> certain properties relevant to "success" however that was defined. I started out with the
> work by Sutton's student and Clare Lyle about plasticity (considering experiments on the
> scale of CIFAR-10) and somehow ended up at llm pretraining and papers like the Multi-Power
> law paper that predicts test loss from train loss and downstream accuracy from test loss.

Routed to: reference topics [topics/plasticity.md](topics/reference/plasticity.md) and
[topics/loss-curve-forecasting.md](topics/reference/loss-curve-forecasting.md);
[potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md) §4 and
[potential-projs/irt-reanalysis.md](potential-projs/irt-reanalysis.md) §4 (loss→accuracy
caveat); [topics/nonstationarity-accounting.md](topics/reference/nonstationarity-accounting.md); the
potential-projs README program-level notes (unifying question).

### Toggle 2 — DataDecide + Pretraining → Post-training

> Then I got sidetracked towards the DataDecide dataset, which seemed like an awesome source
> of variance which could make trying to predict behavior interesting. Also, the search for
> useful proxy metrics when trying to evaluate progress at such low scales seemed related.
> This became a project asking whether pretraining choices impact post-training success,
> even when pretrained final perf is held constant.
> The direction hit a wall when our post-training seemingly had no effect despite using
> standard procedures and datasets. Digging in further, and talking to others, we learned
> that this wasn't easy to figure out from the way people presented their results, but it
> also was seen by others. Ultimately we ended that project a bit demoralized.

Routed to: reference topic
[topics/pretraining-to-posttraining.md](topics/reference/pretraining-to-posttraining.md); §4 notes in
[potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md) (ANN-opt-3),
[potential-projs/wsd-suite.md](potential-projs/wsd-suite.md) (WSD-opt-2),
[potential-projs/token-movement.md](potential-projs/token-movement.md) (TOK-opt-4),
[potential-projs/tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md);
the potential-projs README program-level notes.

### Toggle 3 — Alternatives to Our Post-Training Experiment Design

> I was skeptical of returning to the post-training project because:
> - Iteration is so slow.
> - You really do need more seeds to see real differences.
> - Everything is so model specific that you'd need a fairly large sweep just to debug a
>   small scale issue.
> - Limiting yourself to existing "clean" pretraining sweeps makes it near impossible to
>   test for model family effects.
> - The data decide models are all tiny and have the "no movement during SFT" issue we
>   started with.
> At the same time, I don't want to fully abandon the whole direction.

Routed to: staging topic
[topics/posttraining-experiment-design.md](potential-projs/movement-microscope.md); papers
to [topics/pretraining-to-posttraining.md](topics/reference/pretraining-to-posttraining.md); §4 notes in
[potential-projs/tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md),
[potential-projs/token-movement.md](potential-projs/token-movement.md),
[potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md); the tension and
asymmetric design to the potential-projs README program-level notes.

### Toggle 4 — Use In-Context Learning as the Post-Training

> Alternatively could we treating "in context learning" as the "post-training" and explore
> any features that could be extracted from that or elicitation.

Danielle-flagged project seeds (the `→` notes on the toggle title):

> → "ICL-ability as a cheap predictor of finetunability across pretraining recipes."
> → "How compressible code is into natural language *for a given model pair* is a property
>   of their shared representations, so reconstruction fidelity could itself serve as a
>   capability probe, one that's graded rather than thresholded, unlike pass@1."

Routed to: staging topic [topics/icl-as-posttraining.md](potential-projs/icl-elicitability.md);
papers to [topics/pretraining-to-posttraining.md](topics/reference/pretraining-to-posttraining.md) and
[topics/plasticity.md](topics/reference/plasticity.md); cross-note in
[topics/posttraining-experiment-design.md](potential-projs/movement-microscope.md);
[potential-projs/tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md) §4;
the potential-projs README program-level notes.

### Toggle 5 — How to Combine Vision and Language for the ICL/Post-training Experiments

> I started out looking at plasticity in earlier CNN stuff, and then looking at what results
> I could get confidence bounds for and which I couldn't. Despite not finishing that
> direction, I think there's value to having both CNN experiments at a small scale in vision
> and LLM experiments in the larger scale for NLP. I'm wondering if there's a way to stay in
> this realm of experiments and combine these two spaces, potentially incorporating vision
> transformers instead of CNNs for clear comparisons if it would help.

Routed to: [topics/icl-as-posttraining.md](potential-projs/icl-elicitability.md) (second entry:
the two-tier design); new reference topic [topics/icl-literature.md](topics/reference/icl-literature.md)
(Chan et al. 2022; Raventós et al.); the matched-loss two-controls caution to
[potential-projs/trajectory-statistics.md](potential-projs/trajectory-statistics.md),
[potential-projs/landscape-geometry.md](potential-projs/landscape-geometry.md),
[potential-projs/irt-reanalysis.md](potential-projs/irt-reanalysis.md) §4 and the
potential-projs README program-level notes.

### Toggle 6 — Ash & Adams: Warmstarting and Shrink+Perturb

> I want to start the retrospective with the question that started it all: The Warmstarting
> + Shrink & Perturb paper by Ash and Adams that looked at the effect of training from
> scratch versus pretraining and then finetuning on the same task. They found that training
> from scratch is better than warmstarting. But, of course, was a long time ago and we've
> almost certainly fixed all of those problems now.
> Then, my question: Why didn't it work then? Why don't we hit these problems now?
> - I'm pretty confident that the plasticity research has provided the answers already.
> - However, I haven't really seen a clear reproduction of the original data with a
>   breakdown demonstrating the specific causes.

Routed to: staging topic
[topics/warmstarting-decomposition.md](potential-projs/intervention-grid.md); papers and
hypotheses to [topics/plasticity.md](topics/reference/plasticity.md); stabilizer reading to
[topics/nonstationarity-accounting.md](topics/reference/nonstationarity-accounting.md); bridge
experiment cross-note in [topics/icl-as-posttraining.md](potential-projs/icl-elicitability.md);
thesis chapter plan to the potential-projs README program-level notes.

### Toggle 7 — Analyzing ICL

> How you can analyze what's happening during in context learning? Are there gradient
> approximations? What types of statistics have people come up with to measure the
> intermediate impact of ICL?

Routed to: [topics/icl-literature.md](topics/reference/icl-literature.md) (papers, both the
gradient-approximation arc and the task/function/state-vector family); the ranked
measurement protocol to [topics/icl-as-posttraining.md](potential-projs/icl-elicitability.md); one
line to the potential-projs README program-level notes.

---

## 2026-08-22 — Danielle-supplied citations (during toggle intake)

> so this made me think of paper I saw previously that was quite interesting with a bit of a
> different take on task vectors: Editing Models with Task Arithmetic (Ilharco et al., ICLR
> 2023, arXiv 2212.04089). And then as I was looking for that paper I saw these that we
> should add to the task vector literature (the first citation about icl mainly I think) and
> anywhere else that they are relevant: On Task Vectors and Gradients (arXiv 2508.16082);
> Understanding Task Vectors in In-Context Learning (arXiv 2506.09048); Transporting Task
> Vectors across Different Architectures without Training (ICML 2026, arXiv 2602.12952);
> Task Vectors, Learned Not Extracted (ICLR 2026, arXiv 2509.24169); Task Vector
> Quantization for Memory-Efficient Model Merging (arXiv 2503.06921).

Routed to: new reference topic [topics/task-vectors.md](topics/reference/task-vectors.md); ICL-side
papers to [topics/icl-literature.md](topics/reference/icl-literature.md) and protocol refinements to
[topics/icl-as-posttraining.md](potential-projs/icl-elicitability.md);
[potential-projs/functional-featurization.md](potential-projs/functional-featurization.md)
§4 (task arithmetic as readout; first-epoch gradient as surrogate-ladder support;
quantized deltas for storage);
[potential-projs/moe-partitions.md](potential-projs/moe-partitions.md) §4 (Theseus's
functional task identity as the model for cross-granularity expert matching).

### Toggle 8 — Loss Basins and the River-Valley Explanation

> Another related idea is that of the loss basins and the river valley explanation of
> training. My understanding is that these metrics are sometimes only relevant/comparable
> between models if they're in the same basin. Is there currently a way to identify whether
> a model is currently in the valley versus climbing the mountains? And is there a way to
> identify whether two models are in the same basin?

Danielle-flagged project seed (the `→` note on the toggle title):

> → "Treat basin membership as a covariate of proxy-metric validity: Test whether 'Metrics
>   are comparable iff linear mode connectivity (two models are in the same basin)'. Report
>   your elicitability comparisons *conditional on* barrier height. If recipe effects on
>   ICL-ability only hold within low-barrier pairs, that's a finding. If they hold across
>   basins, that's a stronger one."

Routed to: [potential-projs/landscape-geometry.md](potential-projs/landscape-geometry.md) §4
(origin entry with the flagged seed); new reference topic
[topics/landscape-literature.md](topics/reference/landscape-literature.md);
[potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md) §4 (MPL ↔
river-valley); [topics/warmstarting-decomposition.md](potential-projs/intervention-grid.md),
[topics/icl-as-posttraining.md](potential-projs/icl-elicitability.md),
[topics/task-vectors.md](topics/reference/task-vectors.md); the matched-loss rule in the
potential-projs README.

### Toggle 9 — WSD and Annealing Effects + Dataset Metrics/Features

> I feel like the DataDecide models would be more useful if they were trained with WSD
> learning rate schedule because then we could actually do comparable annealing tests along
> the course of the trajectory. Unfortunately their evals are actually just showing the
> un-annealed or partially annealed performance directly after pretraining. This skews all
> results including post-training performance.
> Then, what are some of our options to work around this problem? Are there ways to do small
> extensions of the training from each checkpoint to gather more data? Or, would redo-ing
> some portion of the pretraining using WSD enable some interesting experiments? How does
> related work interact with this question.
> Separately, how does the field currently quantify differences between datasets? They
> become approximately black boxes because they are SO large, but we analyze trained models
> which actually are black boxes, so there must be things we can do for datasets as well.

Danielle-flagged project seeds (the `→` notes on the toggle title):

> → "Predict performance differences from dataset features."
> → "Does merging-as-annealing-proxy work on cosine mid-run checkpoints rather than just
>   stable-phase ones?"
> → "Does a dataset's 'determinism profile' predict landscape geometry?"

Routed to: origin entries (with the seeds) in
[potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md),
[potential-projs/wsd-suite.md](potential-projs/wsd-suite.md), and
[potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md) §4; new
reference topics
[topics/schedules-and-annealing-literature.md](topics/reference/schedules-and-annealing-literature.md)
and [topics/data-featurization-literature.md](topics/reference/data-featurization-literature.md);
[topics/landscape-literature.md](topics/reference/landscape-literature.md).

### Toggle 10 — Mapping Dataset Tokens into Deterministic vs Uncertain (river-valley test)

> When considering the Wen et al interpolation signal "river test" they describe tokens as
> having different effects, "highly deterministic tokens (facts, knowledge) contribute the
> river direction, while uncertain, ambiguous tokens create the steep hillsides." Have there
> been investigations into mapping data tokens into the two effect buckets and/or looking at
> how that mapping changes over training?

Danielle-flagged project seed (the `→` note on the toggle title):

> → "Validate river-valley theory implications on token loss behavior: Each annealing branch
>   tells you, token by token, how much loss drops under decay = an empirical 'hillside-ness'
>   score at that point in training (decay-responsive = wall, decay-inert = already at the
>   river). Branch repeatedly along the stable phase and you get the trajectory of the
>   mapping: which tokens migrate from decay-responsive to decay-inert, at what rate. And
>   also whether different pretraining corpora produce different migration dynamics for the
>   *same* held-out tokens. Cross that with the epistemic/aleatoric decomposition (aleatoric
>   estimated from an ensemble) and validate: 1. decay-responsiveness should track
>   epistemic-but-not-aleatoric uncertainty. 2. datasets should differ in their
>   epistemic-drainage schedules rather than their aleatoric floors."

Routed to: [potential-projs/token-movement.md](potential-projs/token-movement.md) §4 (origin
of Stage 2, with the seed); new reference topic
[topics/token-level-literature.md](topics/reference/token-level-literature.md);
[topics/landscape-literature.md](topics/reference/landscape-literature.md).

### Toggle 11 — 2026-08-18 17:39 (raw Q&A behind toggle 10)

> you said "Their proposed measurement is interpolation-based: … the valley geometry is
> data-property-dependent, i.e., plausibly recipe-dependent." have there been investigations
> into mapping data tokeks into the two effect buckets and looking at how that mapping
> changes over training?

This is the unedited question-and-answer that toggle 10 ("Mapping Dataset Tokens into
Deterministic vs Uncertain") was reorganized from; the response is the same content
(Wen et al.'s static mapping and toy validation; epistemic/aleatoric decomposition; Rho-1
taxonomy; RLVR token regimes; the causal per-token branch measurement). Already routed via
toggle 10 — nothing additional. From here on, toggles are raw timestamped Q&A; Danielle's
reorganization stopped at toggle 10.

### Toggle 12 — 2026-08-18 17:51 (determinism axis across datasets; Soatto & Achille)

> so then, for the deterministic axis, is this something you could measure across different
> data sets (eg percentage of deterministic tokens, etc)? has this already been done? And
> then one final piece that showed up early in my attempts to understand: soatto and achille
> have an extensive collection of work on information bottlnecks and something about how if
> you blur a cats vision as a kitten it will never be able to see unblurred (their proposed
> equivalent in NN training had a name but i forget). how does this line of work interact
> with these things we've been discussing?

Routed to: [potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md)
§4 (origin of REC-2's design) and
[topics/data-featurization-literature.md](topics/reference/data-featurization-literature.md); new
reference topic [topics/critical-periods.md](topics/reference/critical-periods.md); new staging topic
[topics/critical-period-timing-study.md](potential-projs/intervention-grid.md); notes in
[topics/warmstarting-decomposition.md](potential-projs/intervention-grid.md),
[topics/plasticity.md](topics/reference/plasticity.md),
[topics/landscape-literature.md](topics/reference/landscape-literature.md),
[potential-projs/landscape-geometry.md](potential-projs/landscape-geometry.md) §4,
[topics/pretraining-to-posttraining.md](topics/reference/pretraining-to-posttraining.md); the
potential-projs README program-level notes.

### Toggle 13 — 2026-08-18 18:04 (reproducing the critical-periods paper; reconciling the two core papers)

> Wow. It's really cool that it all does come together. Um, so my question is basically for
> the introduction to the retrospective, and then we discussed the experiments that I could
> do if my goal was basically to reproduce the Ash and Adams paper and then decompose. And I
> see how just adding fisher information, um, ties in the, um, critical learning periods
> idea as well. But I guess I'm curious if I wanted to do the same type of reproduction
> directly on the critical learning periods paper, like, whatever one we'd consider to be
> kind of the first, the twenty nineteen one, um, then what would that look like? because it
> seems like basically having experiments that kind of reproduce and then deconstruct with
> the reference or the perspective of current era. Um, these two core papers as the
> beginning seems like a really great way to start. And then as you've pointed out, each of
> the different continuation threads tie into one or both, and, really, the whole ultimate
> goal is to reconcile them in the modern era.

Routed to: [topics/critical-period-timing-study.md](potential-projs/intervention-grid.md)
(reproduction template, deconstruction axes, elicitability column);
[topics/warmstarting-decomposition.md](potential-projs/intervention-grid.md) (the unified
intervention grid and sequencing); [topics/icl-as-posttraining.md](potential-projs/icl-elicitability.md)
(the elicitability critical-period prediction); the potential-projs README program-level
notes (spine and one-sentence pitch).

### Toggle 14 — 2026-08-18 18:12 (embedding-reset hypothesis lineage; ITER)

> Amazing. And then if we go back even further in my research trajectory, then something I
> was looking at before getting into, like, before knowing about these papers and getting
> into this line of work that probably motivated it was the idea of whether it was possible
> to take a pretrained language model and then transfer it to nonlanguage tasks, um, but
> specifically looking at the fact that back then, this was before that was obviously, yes,
> possible, um, specifically looking at the fact that back then if you wanted to even, um,
> change tokenizations or change vocabulary or things like this, then you would generally
> just train from scratch, or you would do some, like, complex vocabulary merging
> techniques. And I guess our hypothesis was basically that you could just reset the input
> layer and maybe the output layer, um, and then just, like, continue training on the same
> dataset that it was originally trained on, um, to do that swap, um, just with a different
> structure. So, like, a different embedding layer, etcetera. And very preliminary results
> demonstrated this was true, that it took a tiny fraction of the full training length to
> meet or exceed previous perf with the new tokenization or divtionary. though they were
> definitely not strong enough to, like, believe it. But I think that a lot of the work
> around random reinitialization suggests that there is something to that and I guess I'm
> curious what the, like, lineage of that work is. I think it's obviously very tied into the
> rest of the things we've been talking about because random reinitialization or different
> types of reinitialization are clearly ways to, like, jump to different spots in the loss
> landscape. Um, but I'm curious how people handle that now and whether there are still all
> of these works around increasing vocabulary by doing strange manipulations as opposed to
> just, like, reinitialization of a subset and continued training.
>
> there was also a paper that introduced a method called ITER that handled nonstationarity
> by doing a re-init and distill phase periodically and the conclusion was basically that
> this worked great because the learned reps stayed really clean. this too seems quite
> relevant. can you find that paper and explain the link?

Routed to: new reference topic
[topics/reinit-and-transfer-literature.md](topics/reference/reinit-and-transfer-literature.md);
[topics/landscape-literature.md](topics/reference/landscape-literature.md) (basin-preserving vs.
basin-determining resets); [topics/nonstationarity-accounting.md](topics/reference/nonstationarity-accounting.md)
(ITER as the third founding statement);
[topics/warmstarting-decomposition.md](potential-projs/intervention-grid.md) (the
distill-into-fresh-network arm and its control);
[topics/icl-as-posttraining.md](potential-projs/icl-elicitability.md) (function-vs-trajectory test
for elicitability); the potential-projs README program-level notes.

### Interest flag — reinitialization and transfer (2026-08-22)

> I'm actually really interested in what happened with the reinit and transfer literature
> and looking at whether there are places that I could contribute to that direction.

Recorded as an interest flag at the top of
[topics/reinit-and-transfer-literature.md](topics/reference/reinit-and-transfer-literature.md) and in
the topics index; the flag spells out what acting on it would take (targeted literature
pass → gap list → staging topic or project doc if warranted).

### Toggle 15 — 2026-08-18 18:39 (Rothermel et al. 2021 and how it biased the current hypotheses)

> awesome, please look for a rothermel et al followup to the lu et al paper about learning
> rates. thats me! and describe how that work fits into this whole narrative and how that
> project's conclusions likely biased me towards my current interest and hypotheses around
> these directions

Routed to: [topics/reinit-and-transfer-literature.md](topics/reference/reinit-and-transfer-literature.md)
(verified entry for the paper and its stance);
[topics/warmstarting-decomposition.md](potential-projs/intervention-grid.md) (the fourth
founding cell — the control); [topics/icl-as-posttraining.md](potential-projs/icl-elicitability.md)
("quantifying the gap your 2021 paper discovered"); the potential-projs README
program-level notes (through-line, two priors, the arc framing).

### Toggle 16 — 2026-08-18 18:47 (grokking, double descent, causal representation learning / identifiability)

> amazing! ok, i think this i the last highly relevant research body i want to consider in
> relation to this arc: grokking and double descent. these feel very directly related? ahh,
> and to some extent a separate body of work is around causal representation learning and
> identifiability of a system. can you tie each of these research threads into our
> narrative?

Routed to: new reference topics
[topics/grokking-and-hidden-progress.md](topics/reference/grokking-and-hidden-progress.md) and
[topics/identifiability-literature.md](topics/reference/identifiability-literature.md);
[potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md) §4 (decay
branch as anti-grokking instrument);
[topics/loss-curve-forecasting.md](topics/reference/loss-curve-forecasting.md) (double descent as
boundary condition); [topics/landscape-literature.md](topics/reference/landscape-literature.md)
(symmetry quotienting);
[potential-projs/tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md) §4
(the ladder as an identifiability-vs-scale instrument); the potential-projs README
(fourth matched-loss control; the identification-strategy framing; the final map).

### Toggle 17, sub-toggles 1–3 — "Detour into my actual hypothesis and research paths" (18:57, 19:05, 19:12)

Danielle's three statements are reproduced in full (lightly cleaned of transcription
artifacts) in [research-hypothesis.md](research-hypothesis.md): the hypothesis itself
(broken multistage evaluation; elicitation masks intervention signal; learn to elicit, then
use those evals to find warm-start-native training procedures); the pushback that matched
tuning budgets are inherently impossible and misleading given a decade of communal tuning
of the from-scratch regime; and the position that the best achievable evidence is
existence proofs — demonstrations that elicitation moves things, then that a weight update
beats tuned elicitation in some setting.

Routed to: [research-hypothesis.md](research-hypothesis.md) (new, top-level); a fourth
candidate program framing in the potential-projs README; new reference topic
[topics/evaluation-methodology-literature.md](topics/reference/evaluation-methodology-literature.md);
notes in [topics/icl-as-posttraining.md](potential-projs/icl-elicitability.md) (instrument
calibration; strong null), [topics/warmstarting-decomposition.md](potential-projs/intervention-grid.md)
(optimum displacement), [topics/posttraining-experiment-design.md](potential-projs/movement-microscope.md)
(tuning-response curves; demonstration hygiene; meta-analysis).

### Literature pass result — reinit and transfer (2026-08-22)

The Opus subagent's report is at
`~/drotherm/data/.claude/datadec/2026-08-22/0031-reinit-transfer-litpass.md` and is
summarized in [topics/reinit-and-transfer-literature.md](topics/reference/reinit-and-transfer-literature.md)
(verification table, 2023–2026 state, ten ranked gaps). Open decision for Danielle: whether
G3 ("is an interface reset basin-preserving?") or another gap becomes a staging topic /
project.

### Staging topics from the reinit literature pass (2026-08-22)

> thats very cool from the lit pass! can we pull out as many separable staging topics from
> the gaps as possible and then we'll revisit at the end?

Six staging topics created from the ten gaps:
[topics/reset-recovery-dynamics.md](potential-projs/embedding-reset-dynamics.md) (G1, G2, G10),
[topics/interface-reset-basin-test.md](potential-projs/embedding-reset-dynamics.md) (G3),
[topics/reset-and-plasticity.md](potential-projs/embedding-reset-dynamics.md) (G4, G9),
[topics/frozen-body-transfer-audit.md](topics/staging/frozen-body-transfer-audit.md) (G5, G6),
[topics/reset-response-stage-probe.md](topics/staging/checkpoint-tomography.md) (G7),
[topics/reset-effects-many-seed-lm.md](potential-projs/embedding-reset-dynamics.md) (G8).
Promotion decisions deferred to the end-of-intake review.

### Toggle 17, sub-toggles 4–7 — skipped

Skipped at Danielle's direction (2026-08-22): not relevant to this consolidation.

### Toggle 18 — 2026-08-18 19:59 (practical plan for the three-paper replication + grid; CNN and LM in parallel; adviser context)

> Okay. So let's go back to practical considerations. Right now, I'm interested in taking
> the three foundational papers and doing a replication and then breakdown of each of them
> into the grid that we've already discussed to highlight the related aspects of each of
> them and form the shared vocab and conclusion space. And then I think the extension
> direction that I would wanna do from that is to tie them into the idea of identifiability
> (along the causal representation learning direction that you highlighted). And I think it
> would be ideal to do these experiments both with CNN vision models like the original
> papers used along with small LLMs on language tasks. this would let us do the parallel
> experiments and see how they diverge or parallel each other + provides flexibility for
> what direction we may want to go next as we scale. what do you think of this idea/starting
> point? how might it look under the goal of getting convincing results rapidly? my advisor
> doesn't particularly want papers, he wants evidence theres hope of making me an
> independent researcher, so just choosing small bundles i could submit has been clearly
> indicated as not sufficient (hence the less immediately publishable but more strongly
> defensible as a foundation plan)

Routed to: [topics/warmstarting-decomposition.md](potential-projs/intervention-grid.md)
(the one-harness plan, money figure, identifiability-as-measurement-layer, staggered
modalities, six-month checkpoints, two flags);
[topics/critical-period-timing-study.md](potential-projs/intervention-grid.md) (the
figure spec); the potential-projs README program-level notes (closing loops;
pre-registration; scope discipline).

### Caveat for toggles 18–20 (Danielle, 2026-08-22)

> the agent starts to push back based on its suggested approach to managing my phd advisor.
> please disregard everything there, it was misinformed but I think enough useful things
> came up later in the convo that we should still continue processing.

Applied: adviser-management content from these toggles is not recorded (the toggle-18
routing was revised accordingly); technical content is.

### Toggle 19 — 2026-08-18 20:02 (harness already implemented; ~2.5 weeks to initial replications)

> A key piece that I already have the harness implemented and with my agentic coding setup
> im confident that i can have an initial version of the replications + a few variations
> done with prelim analysis in ~2.5weeks if i focus. thats why this seems plausible to me.

Routed to: [topics/warmstarting-decomposition.md](potential-projs/intervention-grid.md)
(what compresses vs. not; known-answer replication as acceptance test; seeds and panel
over variations). Adviser remarks disregarded; the agentic-coding verification lecture
dropped at Danielle's direction (she knows it deeply).

### Toggle 20 — 2026-08-18 20:12 (CRL foundations for the identifiability tie-in)

> ok, can you lay out the Causal Rep Learning related work and concepts that form the
> foundation of your proposed approach to tying the 3 foundation paper grid into
> identifiability?

Routed to: [topics/identifiability-literature.md](topics/reference/identifiability-literature.md)
(five-part foundation and the assembled core claim);
[topics/critical-period-timing-study.md](potential-projs/intervention-grid.md) (the
claim and the four-instrument panel);
[potential-projs/landscape-geometry.md](potential-projs/landscape-geometry.md) §4 (raw vs.
aligned barriers; stitching / linear-map residuals; CKA caveat);
[topics/checkpoint-tomography.md](topics/staging/checkpoint-tomography.md) (LLC as panel member);
[topics/reinit-and-transfer-literature.md](topics/reference/reinit-and-transfer-literature.md)
(stitching as the embedding-reset experiment as measurement).

### Toggle 21 — 2026-08-18 22:32 (how an unpressured researcher would explore DataDecide movement)

> ok, now, for trying to identify movement of the pretrained data decide models at small
> scales (beyond the proxy metrics they've already identified), how would you suggest a
> different researcher with no external pressure might explore the space?

Routed to: new staging topic [topics/movement-microscope.md](potential-projs/movement-microscope.md)
(Stages 1–4); [potential-projs/trajectory-statistics.md](potential-projs/trajectory-statistics.md)
§4 (continued-pretraining control for the noise floor);
[potential-projs/token-movement.md](potential-projs/token-movement.md) §4 (post-training
twins of TOK-obs-4 / TOK-4);
[topics/posttraining-experiment-design.md](potential-projs/movement-microscope.md); the
potential-projs README program-level notes.

### Toggle 22 — 2026-08-18 22:34 (changes between pretraining checkpoints; Signal-vs-Noise)

> I'm actually interested in starting with looking for changes between the provided
> pretrain checkpoints. and please reference the signal vs noise paper when answering.

Routed to origin entries: [potential-projs/trajectory-statistics.md](potential-projs/trajectory-statistics.md)
§4 (Stage 0; TRJ-5's origin), [potential-projs/token-movement.md](potential-projs/token-movement.md)
§4 (Stage 1), [potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md)
§4 (Stage 2 → ANN-opt-7); the Signal-and-Noise citation to
[topics/evaluation-methodology-literature.md](topics/reference/evaluation-methodology-literature.md).
Standing rule from Danielle (2026-08-22): adviser-management content is always dropped; no
further reminders about the Notion citation fixes.

### Toggle 23 — 2026-08-18 22:37 (MoE down to 20–50M active)

> You can train moe models that do something at 20-50M active params, so I'm interested in
> looking all the way down to that scale.

Routed to: new reference topic [topics/moe-literature.md](topics/reference/moe-literature.md); origin
entries in [potential-projs/trajectory-statistics.md](potential-projs/trajectory-statistics.md)
(routing follow-up), [potential-projs/token-movement.md](potential-projs/token-movement.md)
(TOK-obs-5), [potential-projs/moe-movement.md](potential-projs/moe-movement.md),
[potential-projs/moe-partitions.md](potential-projs/moe-partitions.md),
[potential-projs/moe-recipe-suite.md](potential-projs/moe-recipe-suite.md),
[potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md) (merging needs
expert matching); [topics/critical-period-timing-study.md](potential-projs/intervention-grid.md)
(fourth commitment clock); [topics/identifiability-literature.md](topics/reference/identifiability-literature.md).

### Toggle 24 — 2026-08-18 22:45 (eval seeds at a fixed checkpoint; IRT on the DataDecide matrix)

> Do we expect evaling the same model checkpoint with different random seeds on the same
> dataset will give the same or different results? because 3 checkpoints is a pretty small
> n for averages. separately, it seems like the large set of eval models and tasks is the
> ideal setting for something like IRT and other approaches for analyzing benchmarks/eval
> datasets using the predictions from a set of models. DataDecide publishes the per task
> eval results and the perplexity eval results for a range of corpa too.

Danielle-flagged project seeds (the six `→` notes on the toggle title) are reproduced in
[potential-projs/irt-reanalysis.md](potential-projs/irt-reanalysis.md) §4 (origin entry).

Routed to: [potential-projs/irt-reanalysis.md](potential-projs/irt-reanalysis.md) §4;
[potential-projs/trajectory-statistics.md](potential-projs/trajectory-statistics.md) §4
(origin of TRJ-6); new reference topic [topics/irt-literature.md](topics/reference/irt-literature.md);
[topics/evaluation-methodology-literature.md](topics/reference/evaluation-methodology-literature.md)
and [potential-projs/tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md)
§4 (where eval variance lives).

### Promotions (2026-08-22)

> great, I completely agree! lets bundle the four reset topics into one doc, and lets remove
> any mention of retrospective and instead focus on the paper goal (workshop/main
> conference/thesis)

Applied: four new project docs — [potential-projs/icl-elicitability.md](potential-projs/icl-elicitability.md),
[potential-projs/intervention-grid.md](potential-projs/intervention-grid.md),
[potential-projs/movement-microscope.md](potential-projs/movement-microscope.md),
[potential-projs/embedding-reset-dynamics.md](potential-projs/embedding-reset-dynamics.md)
(the four reset topics bundled); eight staging topics absorbed and deleted; two kept staged
(`checkpoint-tomography`, `frozen-body-transfer-audit`) with gates. "Retrospective" framing
replaced throughout by paper/thesis goals.


### Text-latent code autoencoder — 2026-08-22

Danielle's prompt was not pasted; the respondent's playback of it: frozen frontier LLMs as
encoder and decoder behind APIs, a *text* latent, the harness (prompts) as the only learned
object, optimized by an LLM outer loop against round-trip reconstruction scored by test pass
rate, aiming at representation-space benefits (latent dynamics, style manipulation) without
weight updates.

Routed to: new staging topic
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md) (then a staging topic; promoted 2026-08-22)
(not project-specific; not program-level).

> ok, so we're actually collecting potential projects wide and far, they don't have to have
> anything to do with datadecide

> and even though we have a "thesis topic" proposal, if projects don't fall under it, then
> we'll just create alternative topic proposals for them

Standing rules recorded in the potential-projs and topics READMEs: the project collection is
wide; non-DataDecide ideas are first-class; the apex + pillars is one thesis-topic proposal,
with alternatives created for projects outside it.

### Text-latent code autoencoder, Point 1 (degenerate solution) — conversation 2026-07-11, intake 2026-08-22

Claude presented its pushback points one at a time so Danielle could respond to each.

> Yes. So I guess I have two thoughts on this. The first is that I'm not sure what it means in
> this case to be a degenerate solution or, like, clearly directly copying the text of the
> code sample as the output of the encoder and then directly copying it from the output of
> the encoder to the output of the decoder is a degenerate solution. But from initial
> experiments and from the way that I've seen LLMs work in general, I don't think that
> actually will happen. I do think that a lot of the code will end up in the intermediate,
> like, the output of the encoder, which I'm gonna call the representation. But I think that
> that's not necessarily a problem depending on what the goal is, um, because I think that,
> like, yes, you could decide to create a bottleneck by compressing, um, by, like, requiring a
> minimum, um, or, like, a maximum length for the representation. But, actually, the reason
> that we introduced bottlenecks originally was because There was nothing to bias the systems
> towards any solution other than just the degenerate solution. But if you're using LMS as
> the encoder and the decoder, they're actually incredibly strongly biased away from just
> copying the input as the output. And so I think that we might not need that and instead
> could optimize for pass rate, but then also, like, so for maintaining the core
> functionality information, but then also add other optimization targets, um, for other
> types of things that we would want. And, like, that could, for example, take… be taking two
> functions that… I don't know. Like, um, it could be really trying to set up an optimization
> problem that would make functionally equivalent, but surface form distinct code samples
> match to the same thing with some type of contrast or learning type of setup, etcetera.
> So… but I think that, like, in standard machine learning problems, then, yes, you have to
> worry about the degenerate solution. But I think that assuming that this would behave the
> same way as the previous version of encoders, auto encoders, is actually a really large
> assumption.

> Yeah. And I definitely think that, ultimately, it wouldn't make sense. Like, there are
> practical reasons why you would want to have a smaller representation than the initial
> code size. So I completely agree that the ultimate solution probably involves some type of
> actual bottleneck for… if for no other reason than it's, like, storing less data, using
> less resources, whatever. Um, but I just don't want to over optimize the plan or worry
> about like, designing around this type of problem if there's no reason to believe that it
> necessarily will actually be the biggest blocker because this domain, while structurally
> similar to the domain that we're used to, isn't actually the same. And I think that
> exploring how same or different it is is actually a big part of why this would be an
> interesting question. Does that make sense?

Routed to: [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md) (then a staging topic; promoted 2026-08-22)
("Point 1" section and revised open questions).

### Text-latent code autoencoder, Point 2 (objective vs. style goal) — conversation 2026-07-11, intake 2026-08-22

> Yeah. So I think that, um, that's partially because the objective was targeting kind of
> the general purpose case, and I think the style transfer goal, um, I think you would have
> to build on the objective. So the way that I would imagine doing something like a style
> transfer is instead of just having the basic objective where you're trying to learn an
> intermediate representation that could take any form, and you're optimizing for test pass
> rate, instead, um, I think that you could, for example, have a two part, um, intermediate
> representation where the loss then… or, like, the score, whatever, was based on
> regenerating with just the functional part versus regenerating with the functional and
> style part and then using, like, um, surface form reconstruction metrics on function plus
> style part and using functionality metrics only, the test passing, on the generated result
> from just the functional part. And if that… like, if it was possible to optimize that
> setup effectively, then you would end up with one representation for style and one
> representation for function. Uh, you might need You might need one more loss element,
> which would be keeping functionality out of the style, um, representation. But, like, at a
> very high level, in theory, then that would give you one representation for function, one
> representation for style, and then ideally, you would be able to swap out the style
> representations. And if it turns out to that actually, the optimal, um, the optimal form
> of these intermediate representations is human readable natural language text, then
> there's even a chance that you would be able to actually just write language into the
> style portion, or you could even try to optimize so that the style portion was human
> readable natural language text. So I think that there isn't any reason necessarily to
> believe that just purely optimizing the system would lead to human readable natural
> language text, but I think that the general structure makes space for different types of
> modifications of the optimization surface that could lead to different interesting
> outcomes, especially because our quote, unquote optimizer, um, as a smart LLM would not be
> doing your standard, like, not random, but, like, semi random search. Instead, it would
> have its own priors about what would work to make something… would work to make the
> encoder split out style from function, etcetera, um, and would have the ability to look at
> the outputs and adapt its prompts accordingly. And so I think that there's a lot of chance
> that you could play with the components in a way that was more effective. in this type of
> setup than trying to play with similar components in a, like, pre LLM standard auto
> encoder setup where it was notoriously very difficult to tune. Does that make sense?

> Yeah, I think you're getting into the weeds, I'm trying to describe the motivation + some
> illustrative examples for why I believe it's plausible and could be useful. So designing
> the loss function specifically isn't where we are in the discussion yet.

Routed to: [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md) (then a staging topic; promoted 2026-08-22)
("Point 2" section; loss-design detail kept as leads, marked as beyond the motivation stage).

### Text-latent code autoencoder, Point 3 (not quite an embedding) — conversation 2026-07-11, intake 2026-08-22

> Okay. I mean, I definitely am not trying to make the claim that I'm using the term
> embedding exactly accurately. This is definitely meant to be a, like, handwavy parallel.
> That being said, like, um, LLMs are trained aggressively over time in a way that means
> that their input and output space probably is actually a lot more structured than just,
> like, arbitrary whatever. Um, and so I'm not actually convinced, like, the inputs and
> outputs are, in some sense, numerical. and that we have the embeddings of the tokens
> themselves that are being passed into and coming out of the LLMs. So in symptoms, they are
> vectors, though the vector space is probably not the same shape as, like, a VAE
> constrained to a, like, unit ball or something like that. I see that point, um, but I'm
> not sure that, like, I sis… my understanding is that a lot of the design of the vector
> spaces that form the embeddings of the previous era were made to make it practical to
> actually learn a noncollapsed space that had different things that clustered near or far
> away from each other, whereas by using an LLM, like, we're kind of getting the space for
> free. And the question is, like, is this actually a useful space? But at least when used
> along with ALMs, then it's, like, clearly in some sense a useful space. Um, and we have,
> like, tools that are able to extract the usefulness from the space. And so I agree that
> what I'm talking about is not an intermediate representation that would behave exactly
> like the embeddings of old, but I'm not so convinced that it's, like, fundamentally worse
> necessarily, and I suspect that actually digging into, like, the math and theory of it
> all, there would be some things that were surprisingly similar that wouldn't seem so on
> their surface talking about just, like, natural language. Um, does that make sense? again,
> at a high level. Like, we're not talking about detailed details.

Routed to: [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md) (then a staging topic; promoted 2026-08-22)
("Point 3" section; new open question on borrowed geometry + the embedder check).

### Text-latent code autoencoder, Point 4 (prior art) — conversation 2026-07-11, intake 2026-08-22

> Yeah. I see your point there. Um, but I actually think that there's another nearby cluster
> of related work, which is, um, I guess, they're kind of two pieces of this. One piece is
> in the actual coding space, and that is around using natural language intermediates for
> tasks like translation between coding languages or to do, um, debugging. So, like,
> translating code into natural language, making hypotheses about what might be going
> wrong, and then doing rollouts in natural language before moving into… back into code. So
> I think an ld bug is the… is, like, in the title of the debugging paper that I'm talking
> about, and there were a few different natural language for translation papers that I saw
> as well. So that's one piece. And then there's another paper that was… I think it was
> called, like, natural language bottleneck or something like this. And the idea was
> basically that you… it wasn't for code, but it was for, like, student work, um, for,
> like, grading student work. And so the idea was that you did the bottleneck on the
> student work, and you use the intermediate representation for scoring, um, and they
> found that it was useful or something like that. But they didn't really optimize the
> bottleneck. They just tried a few different prompts, and that's also true for the code,
> um, example. So I think that, like, each of the individual pieces has definitely been
> done in terms of, like, having a bottleneck and using it, um, between code and natural
> language, and then separately optimizing a loop that has… optimizing a loop with a
> language model. of these pieces individually has been done, but I haven't really seen
> anything that does all of them together.

Routed to: [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md) (then a staging topic; promoted 2026-08-22)
("Point 4" section; prior-art map flagged as needing a literature pass).

### Text-latent code autoencoder — two interest categories — conversation 2026-07-11, intake 2026-08-22

> Perfect. Okay. So I guess then the next branching point is that I would say I'm more
> interested in kind of exploring what types of representations that you can make and,
> like, what you can do with them, basically, and how you can control them, um, stuff like
> that. And I think I would probably left on my own, choose to look at whether you could
> learn a representation that would be broadly, um, effective. Um, so probably looking at
> robustness for representation across different types of tasks that have code input and
> then also looking at robustness across different models. So, basically, like, could you
> actually optimize the encoder prompt such that you get a representation that you could
> pass it into any level of model and get high quality output from it, um, or even could
> you hyper optimize for a specific model? So let's say you wanted the smallest
> representation that would give a high pass rate on a very specific model, um, separately
> from the others. I think those are the types of things that I would investigate if I was
> just doing what I thought was interesting. But my adviser has decided that we are
> investigating whether we can beat lossless compression, um, by combining this approach to
> do lossy compression and then existing lossless compression algorithms on the
> intermediate representation, um, with the idea that because, like, because we're working
> with code, then we care about preserving functionality not surface form, and so that
> gives us the ability to tune the lossy compression portion, um, where we just need to
> reconstruct the functionality, not the surface form. And therefore, we should be able to
> do better than pure general case losses compression, um, which is fundamentally a
> different… like, I mean, it uses the same structure, and it does give us a very crisp
> optimization metric, um, but it also, I guess, feels kind of qualitatively different than
> the types of things that I was interested in. Um, but at the same time, I think that by
> pursuing his goal, I can learn things about my goal. Um, so the next thing that I'm gonna
> wanna talk about is kind of how to move his goal forward. But before I do that, I would
> like to pause and get your thoughts again at a high level, not at a, like, details
> implementation level about the two different interest categories and kind of the pros
> and cons of those different types of hypotheses, but also how they might be similar or
> different.

Routed to: [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md) (then a staging topic; promoted 2026-08-22)
("Two interest categories" section). The lab's compression goal is recorded as a project
framing; the response's "research agenda smuggled in as analysis" phrasing was rewritten as
a plain sequencing point.

### Text-latent code autoencoder — compression project state and experiment design — conversation 2026-07-11, intake 2026-08-22

> Great. What you just described aligns with what I have concluded, which is basically that,
> like, it makes sense. [adviser commentary omitted] […] though kind of wildly ambitious in
> some sense because beating the best compression algorithms is, like… I don't know. That's
> that's an interesting choice. Anyways, But, like, as you point out, like, in the
> theoretical sense, it should be possible. But, practically, I think that that's maybe not
> what I would assume LLM style optimization would be best at. Uh, yes. Okay. So then if we
> assume that we're moving on with his goal, then basically where I am now is I have a
> pretty solid implementation of the setup, and that I have, like, an inner loop, which is
> the encoding, decoding, test evaluation, and, um, compression evaluation of the
> intermediate representation. that I can run against different models. And he has declared
> that we will be using human eval exclusively, um, which is such an easy task that it's
> been a little bit hard to find models that perform poorly on it even with fairly low
> budgets. But what I've settled on basically is a scheme where we use the original human
> eval prompt, which is basically, like, like, for the for the decoder. We used the original
> human eval prompt, which is basically, like, generate a function matching this description
> or something like that for the decoder and for the encoder. Then there was a paper that
> did, like, human eval explain or something like that that had a prompt that was, like,
> describe this code. And so, obviously, that's a pretty bad prompt for our goal, but it was
> the baseline. So we're using that as a baseline. But I have a system that can swap those
> types of things out that works on human eval and that sweeps a budget. So the idea is to
> get some type of distribution over the space of length versus correctness, then we say do
> this in less than n characters. Um, and currently, we're not actually counting it as
> failure. If it goes over n characters, instead we're just treating that as, like, a way
> that a prompt can encourage different levels of trade off between correctness and
> compression. And so I have the ability to basically do evaluation. and I'm at the point
> where I'm ready to start looking at optimization of the system, uh, with the plan to be…
> to optimize just the encoder prompt, um, and to start with algorithms from DSPI, though I
> won't be using DSPI directly because it's not optimal for my setup. But now, basically,
> I'm thinking about how to structure this initial set of experiments because I think that
> there are kind of Like, keep in mind, there are a few key recipe pieces. So one key risky
> piece is that a lot of the human eval examples get a hundred percent pass rate, and some
> of them get zero percent pass rate. Um, uh, Well, although I think I might have fixed
> that. So most of them get a hundred percent pass rate, which then makes some very bad
> examples for any type of prompt optimization thing. So I think that one thing I need to do
> is to subset down the human eval examples, which I would do anyways because I shouldn't be
> training on the whole dataset we're evaluating. It's just too expensive. Um, so subset
> down to a smaller subset of examples that don't a hundred percent of the time pass. And
> then another thing is basically that as you pointed out, the variance is very high, um,
> even with temperature zero for the performance. And so I think trying to figure out what
> the right number of samples to use to give an evaluation metric for each optimization
> round is a key question, um, because I don't even know what the ballpark would be for,
> like, when it converges. So I'm trying to figure out kind of, like, what is the smallest
> number of things that I can explore to get, like, reasonable hyperparameters before
> trying, like, the simplest algorithms, which I think is copro, c o p r o, um, which is
> basically just sample some prompts, try the prompts, see how they do, sample some more
> prompts. Um, but I think probably that, like, running copro, totally straightforward. Um,
> but the thing that I'm more worried about is that I know for sure that frontier language
> models can optimize these prompts really well. I'm not sure that the structure that copro
> is using would produce that, and so I feel like the third like ex… like mini test before
> actually kicking off a round of experiments that I want to do is to just, like, work with
> a language model. Like, I don't know, something like Codex five point six or something
> like that. Something that's, like, really, really strong, but maybe not as, like,
> thoughtful as, like, Fable, for example. Um, and just basically be like, you… your task is
> to optimize something that's a little bit less complicated than test passing. So maybe
> something like your task is to optimize the number of samples that would let an
> optimization run converge to the true value or something like that. And so the way that
> it would optimize it is that it could give a number of samples and see what the pass rates
> were and then, like, tune that until it came to some type of conclusion. I don't know.
> Basically, like, I think it would be good to choose a more direct, less complex
> optimization problem to test out how the outer loop LLM would perform and what the quirks
> are of the outer loop LLM optimist before trying to actually use it for my problem because
> I feel like my problem is pretty complex. What do you think about all of this?

Routed to: [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md) (then a staging topic; promoted 2026-08-22)
("Compression project: implementation state and initial experiment design"). Lab decisions
(HumanEval only; compression goal) kept as project facts; adviser commentary omitted.

### Text-latent code autoencoder — sequencing the first measurements — conversation 2026-07-11, intake 2026-08-22

> Yeah. I completely agree with all of your points. Um, I think that my main thing is about,
> uh, sequencing. So I think that past that one is gonna be our metric mainly because I
> think that cost wise is the only one that makes sense. Um, and also, like, practically, I
> think, is the only one that really makes sense at this point because language models are
> so good that, like, at this point, what are we doing if we're not talking about past that
> one? Um, for, like, simple coding tasks, anyways. But I think then the question is
> sequencing in terms of you say that we wanna do, like, twenty samples for, um, all the
> budgetaries, but the question is kind of what should the budgetary even be? So, like,
> I've switched from using a fixed character budget to using a compression ratio against
> the length of the input code sample. And for the input code sample, I strip out the the
> doc string, um, because I see the doc string is basically the prompt for the original
> HumidieVel task. So I just pass in the code without the doc string or any comments. Um,
> and so then the target is a compression ratio compared to that, but then expressed as a
> number of tokens, but calculated per sample, basically. Um, and for my preliminary
> evaluation, like there's some kind of… there are open questions about how much extraction
> effort one puts in to try to normalize the LM output responses because we don't really
> care if the response format is not exactly correct because the goal is to, like, maximize
> compression. And so you could consider extraction to be part of the, um, decompression
> algorithm, for example. Um, but, like, basically, all… like, I'm using the cheapest models
> that exist on open router, and they're all at the, like, ninety five percent pass rate
> point. Say the worst one is maybe at, like, eighty percent pass rate, and that goes down
> to pretty low thresholds where basically, like, below a certain threshold is where you
> ask it to provide a description shorter than something, then a lot of the models will
> just say, I I don't know where. I can't do that. That's too small. Um, and so it's, uh,
> like, very, uh, step function jump between, like, feasible and zero percent. And feasible
> generally is, like, a very high pass rate. And so I'm trying to decide basically, like,
> how to choose the sequence of doing the, like, intensive sampling. Um, so taking a lot of
> samples to get the statistic measurement versus the budget sweep course and then probably
> fine grained. Um, and then also, as you say, ultimately, I will wanna use different
> optimizers. I'm really just trying to get a first thing off the ground, but I have to
> implement them right. And so, like, each additional optimizer is an additional period of
> time where I'm implementing instead of collecting results. And so I think it makes sense
> to start with something that's easy to implement and run the analysis on it first, um, so
> that I have something to base my intuition on. And I think that that's really… these are
> the questions I'm trying to answer.

Routed to: [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md) (then a staging topic; promoted 2026-08-22)
("Sequencing the first measurements; the cliff structure").

### Text-latent code autoencoder — baseline vs. seed — conversation 2026-07-11, intake 2026-08-22

> Okay. I think that makes a lot of sense. And then I think the last, like, um, decision
> point that I've been thinking about is basically, like, at the same point where my adviser
> dictated that we were gonna use him and eval, which I think we both agree is kind of
> suboptimal for this setting because the the original code is so short that… I mean, what
> are we even really measuring? Um, and also so simple. But at that same point, he stated
> that we had to use the, um, original human eval explained prompts as the baseline because
> we wanna get a good baseline. And so my interpretation of that is basically that then
> when we do this optimization, we should be optimizing from those prompts. But the problem
> is that, like, those prompts are basically just described this problem in natural
> language, and the task was to describe a natural language and then reconstruct. But
> that's not actually what we're doing, so it seems strange to start an optimization, like,
> to start an optimization for compression from a describe this function prompt instead of
> from a reconstruct, like, provide a representation that would let another model
> reconstruct prompt or from a compress this function as much as possible to reconstruct,
> um, something like that. And so I guess, like, I feel very conflicted between two
> possibilities, one which is to just continue with what my adviser instructed, which is to
> take our baseline and then optimize from it, which he didn't explicitly say, but I think
> he strongly implied, um, which then should give space for optimization performance if it
> seems like there's an obvious improvement on the prompt. versus starting with something
> that's already in the direction of optimization. Um, and, like, I guess, one of the things
> I'm worried about is just, like, it seems silly, but another thing that I'm worried about
> is that, like, by starting with a prompt that doesn't have anything to do with
> optimization, I'm worried that it'll bias the outer LLM towards exploring things in the
> space, the initial prompt, as opposed to towards exploring things in the space of the
> types of prompts that actually might optimize performance. Um, so what do you think about
> this trade off?

Routed to: [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md) (then a staging topic; promoted 2026-08-22)
("Baseline vs. seed"). The response's closing advice on what to say to the adviser was
dropped; the baseline constraint is recorded as a project fact.

### Text-latent code autoencoder — recap and large-n deliverables — conversation 2026-07-11, intake 2026-08-22

> Okay. This has been very helpful. So, basically, the conclusion, like, I'm gonna do my
> best to recap what I'm taking away from this conversation, and then I'm gonna ask you to
> make a document that actually captures, um, your improved, uh, writing and structured
> version of the motivation, the goal, my interests, my adviser's interests, and the plan
> for pursuing my adviser's interest first. Um, so my summary is basically that many
> different design decisions can be just considered arms in the experiment, but It is true
> that the first thing we really need is a statistical… well, we need a regime where there
> is the potential for optimization, and we need a statistical understanding of how many
> samples we need to get, um, any type of signal, um, on the behavior of a given prompt.
> And so we should start with a course sweep, um, and look at, for example, cliffs. And then
> we can choose those regions in order to do large sample collection to understand what the
> confidence bounds are. I guess, actually, that's a question that I have, which is
> basically, like, I do see, like, it's clear to me that we need a large sample collection,
> and we probably are interested in both variance and confidence estimation. Um, but I'm
> not sure exactly what the, like, conclusion that is actionable is that I wanna take out
> of that. Like, clearly, one actionable conclusion is basically given the expected
> difference that we wanna be able to capture, How many samples do we need to have the
> statistical power to do so? Um, if we're trying to optimize. But it seems like there are
> probably other key conclusions that we're trying to extract from this first pass of large
> n sweeps once we've identified the, like, key regimes. Um, yes. So that's an open
> question that I'm sure you can provide some insight into. And then I guess the last piece
> is basically that in addition to our, like, evaluation, then we can start with a simple
> optimizer That's just like a sampling sampling prompts and evaluating them in sequence.
> And we can run that with both the baseline and the task targeted prompt to see whether
> there is a difference and then to take those findings, plus maybe the experience of doing
> a human in the loop version where I am the optimizer, to design what the actual, like,
> first real optimization approach will be that trying to actually move the metrics more so
> than just validate the optimization loop. Would you say that this fairly accurately
> captures what we've discussed and concluded.

Routed to: [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md) (then a staging topic; promoted 2026-08-22)
("Recap of conclusions; what the large-n phase must output").

### Text-latent code autoencoder — fractional test pass rate — conversation 2026-07-11, intake 2026-08-22

> Interesting, for binary pass rate that makes a lot of sense. And the fact I didn't realize
> that points to the fact that I also have access to test pass percentage. I feel like
> binary pass rate is the relevant top line metric but it feels like test pass rate is
> useful signal too? And that had variance right?

Routed to: [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md) (then a staging topic; promoted 2026-08-22)
("Fractional test pass rate as a second signal").

### Text-latent code autoencoder — per-test split confirmed — conversation 2026-07-11, intake 2026-08-22

> Ah, yes. No. I split the human eval test, um, out into its subcomponent, so I have
> individual tests, um, from the check function.

> Yes, please. For the document.

Routed to: [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md) (then a staging topic; promoted 2026-08-22)
("Per-test outcomes confirmed; dividends and caveat"). The requested document follows as a
separate intake.

### Text-latent code autoencoder — promotion — 2026-08-22

Danielle pasted the structured write-up ("Optimizable Text Representations of Code with
Frozen Frontier LLMs — research framing and experimental plan, functional compression
first"). Applied: staging topic promoted to
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
(`TLC`); the write-up forms §1 and §3, the dated discussion moved to §4, §2 synthesized and
marked draft scaffolding. One sentence of adviser-communication advice was dropped.

## MAQA Next Steps (Notion page) — intake from 2026-08-22

Source page: "MAQA Next Steps" (app.notion.com/p/...3c1de135cd1f815ea18ad1c9776077ca). Excerpts
pasted one at a time.

### Turn wiki corpus into shards — conversation 2026-08-16

> I'm trying to think through how best to store a full Wikipedia corpus in a way that
> sharded, but where the shards co-locate things that are likely to be accessed together
> when doing question-answer tasks. And so some of the things that first come to mind are
> using the links between pages to form a graph, and then doing a, like, graph processing
> algorithm to find the, like, I don't know, is it min-cut algorithm? There are all kinds of
> different graph processing algorithms, so I'm sure there is some that exist to try to
> figure out how you can parse the graph into subsets with the goal of cutting the fewest
> branches, edges. But yes, so I guess I'm curious both about, for me, like, foundational
> idea, like graph algorithms and how people would approach this type of problem, what the
> options are, but then also how people have applied this type of thing to something like
> Wikipedia or some other large corpus for QA purposes in the literature.

Routed to: new staging topic [topics/wiki-qa-sharding.md](potential-projs/wiki-qa-sharding.md).

### Multi-answer QA state of the field — conversation 2026-08-16

> Okay, great. Now, about four years ago, I worked on multi-answer question answering, the
> idea that you have a single question that has a full list of answers that you assume is
> correct. So, what movies did Alfred Hitchcock write? I dunno, maybe he wrote books. But
> that's the idea. Like, and the questions could be simple, they could be complex. So, like,
> filtering with these types of questions, etc. And if you, the subset that I was interested
> in is assuming you have access to a knowledge base, but it isn't, like, parsed. So you
> have a corpora, but not a knowledge base, I guess. So you have access to the raw Wikipedia
> pages, but not a Wikipedia knowledge base. And there were a few different datasets that we
> looked at back then. And so I'm curious, basically, what existing research is there that
> has continued to either use these datasets or datasets like them, and what is the current
> state? And then also how effective, like, are there, like, rigorous research
> demonstrations of how effective large models, like frontier language models, for example,
> are at this, without a corpus or with a corpus, but especially without, because I think
> that would be the comparison point that people would be interested in. And so can you
> look, can you do kind of a deep search for this, looking through sources like Archive to
> try to find the related work on this and produce a report?
> the datasets were QAMPARI, QUEST, and RomQA

Routed to: new reference topic
[topics/multi-answer-qa-literature.md](topics/reference/multi-answer-qa-literature.md)
(Danielle interest flag); MAQA expanded in the wiki-sharding staging topic.

### Multi-answer QA — the full report — 2026-08-22

> the report

(File `~/Desktop/multi_answer_qa_state_of_research_2026.md` attached.) Copied verbatim to
[refs/multi-answer-qa-state-of-research-2026.md](refs/multi-answer-qa-state-of-research-2026.md);
distilled into [topics/multi-answer-qa-literature.md](topics/reference/multi-answer-qa-literature.md).

### Multi-answer QA — cleaner datasets — conversation 2026-08-16

> This is very helpful. Another question that I had back then, that I think we're in a much
> better place to answer now, is how to produce cleaner datasets. Because if we're looking
> at evaluating a recall measure over hundreds of documents, then it matters less. If we're
> looking at doing recall over eight, but only six are actually listed in the ground truth
> for the dataset, then that can dramatically swing the accuracy of the evaluation.
> Similarly, in the past, part of how you would create gold evidence passages were just to
> look for examples where a question entity and an answer entity both existed, or examples
> where you queried with BM25 for the question, and then it contained the answer, which
> leaves a lot of space for false positives that could definitely be filtered out fairly
> inexpensively now, given how strong our quote-unquote reader models are. And so, what
> additional work has been done on improving the quality of these QA datasets so that they
> are more accurate and provide more precise information about how to differentiate
> between different methods?

Routed to: [topics/multi-answer-qa-literature.md](topics/reference/multi-answer-qa-literature.md)
("Cleaner datasets: verification over enumeration").

### Wikipedia downloads under a storage budget — conversation 2026-08-16

> Interesting. Okay. So if right now I wanted to download Wikipedia-related data, but I had
> a limited amount of storage space, then what are my options in terms of kind of where I
> would download it, what level of granularity is there, like metadata and then the actual
> text, and what sizes are these? And then is it possible to download like just some
> percentage, some shards, or is it like you download the whole thing and then you can
> split it up yourself? Let's assume that I just want one date of Wikipedia, and it can be
> a current date or like very recent, or it can be historical.

Routed to: [topics/wiki-qa-sharding.md](potential-projs/wiki-qa-sharding.md) ("Downloading
Wikipedia under a storage budget").

### Link graph from Structured Wikipedia — conversation 2026-08-16

> is there something like the huggingface structed dataset for the wikipedia link graph too?

Routed to: [topics/wiki-qa-sharding.md](potential-projs/wiki-qa-sharding.md) ("Link graph from
Structured Wikipedia"); data-plan open question revised.

### Index stack at Wikipedia scale — conversation 2026-08-16

> great, next, the best approach to implement indices ranging from bm25 to vector indices
> used to be to use pyserini, but there must be newer better options now that still work at
> wikipedia scale?

Routed to: [topics/wiki-qa-sharding.md](potential-projs/wiki-qa-sharding.md) ("Index stack:
BM25 to vectors at Wikipedia scale").

### LanceDB alone — conversation 2026-08-16

> so what are the pros and cons of building on lancedb, and how far could i get using it
> alone for retrieval?

Routed to: [topics/wiki-qa-sharding.md](potential-projs/wiki-qa-sharding.md) ("LanceDB alone:
pros, cons, how far it goes"); index-plan open question revised.

### Qdrant solution — conversation 2026-08-16

> ok, so then, what would a solution with qdrant look like?

Routed to: [topics/wiki-qa-sharding.md](potential-projs/wiki-qa-sharding.md) ("A Qdrant-based
solution").

### DuckDB vs. LanceDB — conversation 2026-08-16

> how does duckdb compare to lancedb? for my use case but more generally is the interesting
> part of the question

Routed to: new reference topic
[topics/retrieval-storage-tooling.md](topics/reference/retrieval-storage-tooling.md) (the
general comparison, per Danielle's emphasis); project application in
[topics/wiki-qa-sharding.md](potential-projs/wiki-qa-sharding.md).

### Entity-ID graph candidate fetch and HDF5 — conversation 2026-08-16

> Interesting. So, one of the things that I'm trying to figure out is I previously had a
> research project where the goal was to, for every Wikipedia page, extract entity IDs, and
> then create a graph out of the entity IDs instead of out of the links. And then do
> candidate fetch from questions based on the question entities. And so this is a wildly
> heuristic approach. And unsurprisingly, I ran into issues with the candidate fetch itself
> in that, you know, an entity might show up tons and tons and tons of times. And so
> actually doing the joins and stuff with the database was rough. And so I ended up
> implementing my own set of indices and just storing the data in those indices using
> different types of IDs for different types of objects. And ultimately my conclusion was
> what would be best is to store everything in an HDF5 Py object so that I could read
> directly from disk. And if I implemented those reads in like a batched way, smartly, then
> that would make everything easier, like feasible in terms of time to do these queries.
> And I guess what I'm wondering is whether there is a better way to get around the types
> of problems that I was hitting where everything just doesn't fit into memory. Something
> has to be read, streamed from disk. But I can't assume that the sharding approach will
> work. And so then is implementing my own HDF5 Py data structure really the best
> alternative, or do solutions exist for this that I just didn't, I wasn't aware of?

Routed to: [topics/retrieval-storage-tooling.md](topics/reference/retrieval-storage-tooling.md)
(general analysis) and [topics/wiki-qa-sharding.md](potential-projs/wiki-qa-sharding.md)
(the proposed stack).

### Entity linking beyond string matching — conversation 2026-08-16

> ok, so then we were actually acting on text chunks from wikipedia and doing entity linking
> via string matching because we wanted every mention in a given page not just the first
> link. and we wanted an approach that could plausibly work with any arbitrary corpa without
> wikipedias linking. string matching was unsurprisingly very not fun, though
> computationally it seemed cheaper than running an expensive model over all if wikipedia,
> but i feel like there must now be a better option?

Routed to: new reference topic
[topics/entity-linking-at-scale.md](topics/reference/entity-linking-at-scale.md); pointer
added in [topics/wiki-qa-sharding.md](potential-projs/wiki-qa-sharding.md).

### MAQA decomposition goal — conversation 2026-08-16

> Okay, so one thing I definitely learned from the first time trying to attempt this problem
> is that there are so many pieces that interact, and there are kind of infinitely complex
> ways to try to address each of the pieces. And what I really want is to start with, like,
> the simplest of heuristic baselines and to decompose the different pieces of the problem
> as much as possible to try to understand, like, how well different pieces work,
> especially as you scale them up or down. And so my thought is, like, and of course it's
> challenging to decompose the piece of the problem because they interact with each other,
> but I feel like it should be possible to test question entity linking, to test retrieval,
> to test what the max possible retrieval is for some given assumption. So, like, if we
> assume that the entity set is given by the Wikipedia entity knowledge base, or if we
> assume that it's given by some form of normalization of Wikipedia page titles, that should
> give us, like, an upper bound because we can't predict entities that are not included in
> that set. Also, if we cap different pieces along the way, so if we cap the number of
> evidence passages that we consider, then in the best possible case where somehow we
> magically get the correct evidence passage and we get full, what's it called, full
> diversity in our evidence passage list, that still, for some of the questions, will
> prevent them from reaching 100%. And then basically using, sorry, I'm thinking about
> QAMPARI specifically, using things like for retrieval, taking all of Wikipedia and doing
> retrieval, and then looking for the presence of any evidence passage that's a ground truth
> evidence passage, looking for the presence of the correct answer helps us to understand
> whether we can handle complex questions with a single round of retrieval, etc. And for
> reading, verifying that if we pass in the full set of evidence passages, either all
> together or individually, depending on how we're actually doing the reading, seeing how
> many correct answers we get from that and whether our evaluation metric actually considers
> correct answers correct. I think each of those things are things that can be analyzed
> before trying to hook up the whole system together. Does that make sense as a goal? And if
> so, what are some ideas for kind of like the simplest possible heuristic approach for each
> of these things and the values that would actually be useful for understanding kind of
> like the very coarse-grain impact of different types of decisions.

Routed to: new staging topic [topics/maqa-oracle-ladder.md](potential-projs/maqa-brute-force-baseline.md)
(Danielle's goal recorded as the seed).

### MAQA goals restated (second model) — conversation 2026-08-17

> About four years ago I worked on a multi-answer question answering research project. You
> have a single question that has a full list of answers that you assume is correct. So,
> what movies did Joachim Trier direct? And the questions could be simple, they could be
> complex where you need to do set operations or multi-hop evidence retrieval to produce the
> final answer set. I was specifically interested in the setting where you have access to a
> knowledge corpus like raw wikipedia (not parsed or linked or annotated).
> There were a few different datasets that we looked at back then: QAMPARI, QUEST, RomQA
> and Monaco. I know existing research has continued, split into three core tracks:
> Exhaustive QA, Set based QA, and wide fan out QA. And of course LLMs have gotten much
> stronger, but my understanding is the F1 metrics on datasets like these are still
> surprisingly low.
> One thing that really bothered me before was that the dataset ground truth was not clean:
> you might be missing answers in the exhaustive test, the evidence passages were selected
> heuristically and were often incorrect, the scoring penalized formatting of answers more
> that correct vs incorrect, etc. And I believe many of these things have been investigated
> and improved, though maybe not for these datasets specifically. I would also assume more
> datasets in the space have been released.
> Another big issue for me before was the approach we attempted was very brute force:
> identify all enitiy mentions in the questions amd the wikipedia passages. Split wikipedia
> into 100 word chunks and then make a graph using the entity mentions as edges. Traverse
> the graph for candidate fetch. Cluster evidence by entity and rerank per entity for a
> subset to use for reading per-entity. I think ironically that building a knowledgebase
> like that from wikipedia and doing more exhaustive retrieval has actually held up in the
> era where LLMs are super strong but correctness is the issue. But the strung based
> matching, entity resolution across different surface forms, dealing with long tails, etc
> was frustrating and introduced infra constraints.
> Then I'm now interested in picking this project up to:
>    1. do a retrospective, taking the shape of the approach we considered then with the
>       datasets from then but applying what ive learned since then to make faster/better
>       progress.
>    2. then update to a variation that fits the current landscape, still sticking
>       moderately close to tye brute force design as a target first paper (workshop
>       likely) and sort of a heuristic baseline from someone who doesnt care about people
>       thinking the approach is cool and just wants to see how far we can get with simple
>       clean approaches.
>    3. then likely try to do a paper or two on more interesting approaches targetting
>       specific failure points
> does this overall intro to my goals make sense? please dont recommend approaches to me
> yet, but do perform some research to validate my assimptions about the current state of
> the field and provide me a foundation to work from + provide any feedback to my thoughts
> that you have.

Routed to: [topics/maqa-oracle-ladder.md](potential-projs/maqa-brute-force-baseline.md) (three-paper
arc as program framing + plan feedback) and
[topics/multi-answer-qa-literature.md](topics/reference/multi-answer-qa-literature.md)
(second validation pass).

### MAQA decomposition goal, second model — conversation 2026-08-17

Prompt identical to the 2026-08-16 "MAQA decomposition goal" entry above (with "Campari" for
QAMPARI); not repeated.

Routed to: [topics/maqa-oracle-ladder.md](potential-projs/maqa-brute-force-baseline.md) ("A second,
leaner version of the ladder", with a cross-reference to the first).

### Project-approach principles and MAQA problem definition — conversation 2026-08-17

> Okay. So I spent some time thinking about what some good general takeaways are in terms
> of how to approach new projects, um, from looking back at my approach to this project and
> also looking at my approach to other projects. And I guess I wanna list some out to get
> your opinion on them, um, and then show how I've been thinking about doing some of the
> steps for this project.
>    1. start by being clear about the problem def, shape of solution space youre
>       considering, and then what things are likely to have a large impact on success in
>       that solution shape
> Choose your problem
>       * exhaustive QA for questions like those in the QAMPARI dataset (entity centric, few
>         hop max, some simple set ops, 4-10 answers generally)
>       * using a knowledge corpa that is unannotated
>       * success measured as F1 on the cleanest possible dataset
> Choose your solution space
>       1. prepare corpa for use
>       2. get evidence from corpa
>       3. choose what evidence to use for answering
>       4. use evidence to answer
> → entity centric, form an entity set and answer from it.
> What will impact outcomes the most
>       * having a complete but previse entity set to select answers from
>       * being able to select small enough evidwnce chunks that all chunks needed for
>         multihop qs fit in the input usefully to the answering mechanism.
>       * diversity of evidence retrieved to allow for long tail answers
>       * normalization of entity mentiobs such that they are linkable and the result
>         matches the expected answer form
>       * an answer mechanism that is able to give accurate answers when given clear and
>         correct evidence
>       * some way to limit explosion of evidence according to time and infra constraints
>    2. its worthwhile to start by getting an intuition for the problem, especially for the
>       datasets + the high leverage axes from your solution shape
>       * especially for something so heuristic: how do super simple baselines perform and
>         why is an essential question
>       * you should understand the distribution of the important types of objects youre
>         working with along the major axes that impact solution cost, performance or both.
>         long tails can tank a bad design, and naive truncation can tank performance.
>    3. if you know your dataset is noisy in a way that might impact your solution, spend
>       just a little time (a) scoping the damage and potentially (b) making a clean set to
>       iterate against.
> -> i want to investigate the scale of the qampari issues in addition to the investigation
> they did (including entity string matching, evidence representativeness, etc), answer set
> completeness, amd then quickly use modern tools to improve the dev set + evaluation
> metric in a best effort type of way.
>    4. its often easirr to start working against a single dataset, and as long as (a) your
>       systems are reproducible and (b) each experiment tells you something about what
>       does or doesnt work and why on the one dataset, tou shoild have what you need to
>       redesign/extend your solution to more similar datasets fairly directly.
> -> start with qampari for historical reasons
> thoughts?

Routed to: new reference topic
[topics/project-approach-principles.md](topics/reference/project-approach-principles.md)
(the four general principles + feedback) and
[topics/maqa-oracle-ladder.md](potential-projs/maqa-brute-force-baseline.md) (problem definition,
solution shape, impact hypotheses, plan-specific feedback).

### Implementation time and bounded cleaning — conversation 2026-08-17

> Interesting. So one other major takeaway from the whole process is that I often get stuck
> on implementation time. So, basically, like, part of why I would say to start with one
> dataset instead of multiple is that trying to get multiple datasets set up is often really
> a dramatic use of time because nothing ever actually works. And I'll have their different
> baselines and their different, um, setups and quirks, and it's often hard to even tell
> what the papers are measuring, um, let alone understanding it from the release data, um,
> even if they do release a repo along with it. And so I think that I'm trying to also
> encode in this set of things. Basically, like, some stuff is worth the extra time upfront
> even though it delays the time between when I work with somebody, uh, collaborator and
> adviser, and they first talk about the project versus when I can give them the results
> they're actually interested in. And I think that, like, a main takeaway for me is that a
> lot of these things, if I had done first, I would have been able to give those results,
> um, more consistently, um, in a way that was more meaningful, and we would have gotten,
> like, final outcomes much faster. But I also think that I want to limit things like
> keeping slices from other datasets. or, like, too much ceremony over different things
> because that will then definitely backfire. And so I agree that, like, dataset cleaning
> and be its own whole thing. But I think also basically saying, okay, like a small subset
> of questions, and I'm going to look at them by hand and then maybe use, like, an LM to…
> like, for example, I'm gonna take a small subset of questions. I'm going to use a
> baseline method to get the high relevant evidence. And then for the ones that are not
> actually used as ground truth evidence, I'm personally gonna look at them and see whether
> they seem correct or not. So, basically, I'm gonna hand annotate small subset that I have
> a clear example or I'm gonna create a answer matcher that just does what I think are
> reasonable string normalizations before comparing things like this. Um, I think that
> those are things that can be done pretty quickly and that then I know what I'm working
> with. Um, as I go through the project and it is possible to overindex on a specific
> dataset, I think also… and it's possible to overindex on your own dev set. I think that
> ultimately what we want is to produce solutions work on the actual task. We're aiming to
> work on. Um, and so even if my solution works better on my dev set, then the actual dev
> set, then there are all kinds of tricky things you can do to fix that. also, like,
> ideally, that means it also translate vendor to other datasets. And maybe the problem is
> that we just need a better dataset. Um, so do these concerns make sense to your
> responses?

Routed to: [topics/project-approach-principles.md](topics/reference/project-approach-principles.md)
(integration tax; front-load findings; error-driven annotation; over-indexing guardrails)
and a bounded-cleaning open question in
[topics/maqa-oracle-ladder.md](potential-projs/maqa-brute-force-baseline.md).

### Baseline-surfaced annotation — conversation 2026-08-17

> Amazing. I agree with your point that these numbers are findings. [discussion of
> collaborator expectations omitted] […] I also agree about if we're correcting… or if we're
> doing annotation based on existing systems, then it's clearly a biased Getting a byproduct
> of errors that were king. I was hoping to offset using, um, January baselines to get the
> potential errors as opposed to my proposed… and I'm hoping that that would at least, um,
> avoid biasing the fixed dataset to match my proposed approach over the baselines we'd be
> comparing to. Does that seem valid?

Routed to: [topics/project-approach-principles.md](topics/reference/project-approach-principles.md)
(sequencing for delivery; unbiased error-driven annotation) and the bounded-cleaning entry
in [topics/maqa-oracle-ladder.md](potential-projs/maqa-brute-force-baseline.md). The response's
collaborator-management advice was dropped; its method-neutral sequencing point was kept.

### Post-intake decisions — 2026-08-22

> ok, lets start with (1) and I think we should call out two projects, because while the
> sharding isn't on my arc now, I think its an interesting and doable paper for a post-phd
> arc or a point where I want an "do some engineering" type of break
> (2) yes, defer
> (3) keep here
> (4) defer
> (5) defer

Applied: `maqa-oracle-ladder` → [potential-projs/maqa-brute-force-baseline.md](potential-projs/maqa-brute-force-baseline.md)
(`MAQA`); `wiki-qa-sharding` → [potential-projs/wiki-qa-sharding.md](potential-projs/wiki-qa-sharding.md)
(`SHARD`, flagged post-PhD / engineering break). Alternative topic proposals for non-pillar
projects deferred; `project-approach-principles` stays a reference topic; literature gates
and the draft-scaffolding review remain parked.

## Mixed topics — intake from 2026-08-22

### Annealing-data literature — undated conversation (response scoped to Oct 2025)

> I really want to better understand the recent research around data quality as it affects
> LLM annealing, especially changing data from pre-training to the annealing stage.
> Specifically I'm a 5th year phd student in LLM training, focusing on pretraining and the
> impact of design choices on post training outcomes. I'm interested in how the order and
> stage of introducing data impacts how well the model fits it, with an understanding that
> different labs use their highest quality data during the annealing phase because this
> somehow improves the overall results.
> Please do a really extensive and deep exploration of the recent work, eg the last year of
> research papers, related to the impact of annealing data and protocols on training
> outcomes. Please focus on academic sources like arxiv and be careful to clearly represent
> the contents of the papers precisely in your response. After covering the major recent
> work please provide a high level summary of the recent directions in the field and the
> direcions that you think will be really actively pursued in the near future.

Routed to: [topics/schedules-and-annealing-literature.md](topics/reference/schedules-and-annealing-literature.md)
(dated survey section with a relevance note for ANN / WSD / FUNC).

### Early-dynamics prediction draft review — 2025-07 (intake 2026-08-22)

> I'm a 3rd year PhD student in Machine Learning and I'm trying to design an investigation
> that is fast to execute on, that could serve as the core of a workshop paper at a top ML
> conference, and that could then be extended to a strong submission to a top ML
> conference. I created a latex document (attached as a pdf) with my initial thoughts and
> then aggregated notes (attached as a markdown file) from discussing my plans and refining
> design details with Gemini. Please consider the full plan and provide any feedback or
> suggestions for changes that would make this first round stronger or more likely to
> succeed without increasing the timeline or difficulty for me. Additionally, surface any
> concerns that you identify with recommendations for how I could mitigate them. Explain
> your reasoning for each suggestion.

(PDF attached; the Gemini notes markdown was not provided.) Routed to: new staging topic
[topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md); PDF
copied to [refs/](refs/README.md).

### Early-dynamics draft review, second response — 2025-07 (intake 2026-08-22)

Same prompt as the entry above (Danielle asked several similar questions); response
recorded in [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Second review: GBDT v0 design details").

### Early-dynamics draft review, third response (recipe families) — 2025-07 (intake 2026-08-22)

Follow-up in the same review sequence (prompt not provided; Danielle notes the Gemini notes
file cannot be recovered). Recorded in
[topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md) ("a
recipe-family scheme for leave-family-out CV"), with a correction to its family-size claim.

> this is related to the datadecide dataset, huh, we should note that link

Applied: DataDecide link and pillar relevance added to the early-dynamics staging header;
index row updated.

### Early-dynamics draft review, fourth and fifth responses — 2025-07 (intake 2026-08-22)

> Interesting, it seems that Dolma1.6++, C4, and the Falcon base fold only have a single
> model. Additionally, I would expect that the hold-out ablations of Dolma1.7 would look
> different than the base. Considering these would you still argue for the 8 fold approach?
> Why or why not? I've included the data table screenshot here to make your job searching
> for reference info easier.

> Perfect, then, considering the model sizes are "4M, 6M, 8M, 10M, 14M, 16M, 20M, 60M, 90M,
> 150M, 300M, 530M, 750M, 1B" but for all but 1B 2 seeds are only run until 25% whereas
> only 1 seed makes the full run, how would you recommend I do the expanding window of
> model sizes?

(Screenshot of the DataDecide recipe table attached.) Routed to:
[topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Fourth and fifth responses"), with an intake note that the seed-truncation premise
conflicts with the 2026-08-21 coverage check in open-questions-answered.md.

### Early-dynamics draft review, sixth and seventh responses — 2025-07 (intake 2026-08-22)

> Ok great, and then, this seems like it impacts your proposed validation split method. You
> said ""Add a stratified shuffle split (10 % held out) that (i) balances model-size buckets
> and (ii) holds out unseen seeds rather than unseen recipes." Based on this new seed info,
> assuming that I follow your recommendations for the recipe families and expanding model
> size windows, how would you recommend I handle validation splits?

> Excellent, and finally, propose splits for the proposed test of training across some data
> recipe families + smaller model sizes and testing them on larger models from held-out
> recipes?

Routed to: [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Sixth and seventh responses"), with a row-count correction.

### Early-dynamics draft review, eighth and ninth responses (featurization) — 2025-07 (intake 2026-08-22)

> Great, next lets talk more about featurization. You said "a) Log-transform and z-score
> metric within each model-size bucket before fitting. b) Pass model size as numeric feature
> so LightGBM can split on it." Can you explain (a) in more depth, what would this look like
> and why would I do it?
> Additionally, you said dataset features should be "(a) total tokens in recipe; (b) % code,
> % CC-derived, % social-media; (c) mean document length; (d) duplicate-rate estimate". Why
> choose these features? do you think these are sufficient?
> Please also provide any more thoughts you have on featurization.

> So, the two metrics I'm predicting are perplexity and correct_prob (the average of the
> probabilities for all correct continuations). I see why I would log transform the
> perplexity but should I also do this for correct_prob since its already on the 0 to 1
> scale?

Routed to: [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Eighth and ninth responses: featurization"), with a note linking the static features to
`REC`.

### Early-dynamics draft review, tenth response (target transforms) — 2025-07 (intake 2026-08-22)

> Interesting, so then would you recommend I apply the same transformation to the
> prediction target metrics?

Routed to: [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Tenth response: transform the targets too?").

### Early-dynamics draft review, eleventh response (training method) — 2025-07 (intake 2026-08-22)

> Excellent! Next lets talk about the training method, a few questions:
> "1. Spearman ρ / Kendall τ for ranking. 2. Calibration (ECE) for value prediction." - How?
> Why?
> "Pairwise-ranking objective (lambdarank) for ranking tasks instead of manual binary
> classification." - Is this different than using something like lambdamart for training
> the regression target version?
> "Fixed 16 equispaced points (log-time spacing) per metric. Works well with 512-leaf GBDTs
> and avoids variable-length feature vectors" - why?
> "GBDT with LightGBM, 512 leaves, learning-rate 0.05, early-stopping 50 rounds on val NDCG;
> lambda-rank objective for pairwise tasks; regression MSE for scalar targets." - are these
> what youd recommend I start with?

Routed to: [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Eleventh response: training method details"), with an intake note on the leaf-count
setting vs. dataset size.

### Early-dynamics draft review, responses 12–15 (metrics, ECE, lambdarank, two heads) — 2025-07 (intake 2026-08-22)

> So are you suggesting that I only need to report spearman rho and ECE? What about the
> fairly standard baseline measurements of relative and absolute accuracy + decision
> accuracy?

> when you say "Expected Calibration Error (ECE) on binned residuals", does "ece =
> expected_calibration_error(y_true, yhat)" handle binning already?

> for lambdarank win/loss, the X set should be different right because it should pass in
> pairs, or are you saying this is handled *by* the lambdarank algo?

> Ok, great, so can you expand your code example above to show what I'd do to have one
> ranking head and one regression head, including the suggested default params?

Routed to: [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Responses 12–15"), with intake notes on two code defects and on the query definition
(rank recipes within a size, not sizes within a recipe).

### Early-dynamics draft review, responses 16–18 (SHAP, rolling slopes, feature clarifications) — 2025-07 (intake 2026-08-22)

> Excellent! Next can you explain: "Use SHAP (TreeExplainer) once per CV fold; aggregate
> mean" - What is this? How? Why?

> Next, "Rolling-slope features (slope of a 5-point window slid across the early curve)" -
> Are you suggesting doing all the fits for this rolling window or just linear?

> Ok! And then my last 3 questions:
> "Also store "relative progress wrt LR schedule": % of warm-up completed, % of cosine decay
> completed." - this is wrt cumulative learning rate?
> "Noise scale estimate (Var(grad loss) ▭ ≈ E[(lossₜ − lossₜ₋₁)²])." - I only have access to
> the evaluation metrics from the different listed evaluations. Is there a way to reproduce
> this goal with these curves or should I table this suggestion?
> "Add "effective context length" = min(seq_len, tokens_seen/steps) for each checkpoint." -
> Batch size is number of sequences per batch and the sequence length is a uniform 2024 per
> model. Is this recommendation still relevant?

Routed to: [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Responses 16–18"), with an intake note on `%_decay_done` collinearity.

### Early-dynamics draft review, responses 19–20 (any-step targets; all-metrics → MMLU) — 2025-07 (intake 2026-08-22)

> Awesome! so, I have a great sense of how I'll do the pairwise ranking or regression based
> ranking for the "in domain" prediction where I use a single metric's early window to
> predict the final value.
> Two settings remain:
> 1. predicting intermediate values not just the final value
> 2. using many metrics early windows to predict a single other metric's final value (MMLU
> in this case) as somewhat of an upper bound

> Let me clarify a bit. For (1) I want to use one model to predict the target metric at
> *any* step so I need to featurize the target and it needs to be an input. For (2) the
> metrics I'm using are the evaluation values for a sequence of perplexity measurements and
> downstream task measurements at each checkpoint of the training run. Then, I want to use
> all those evaluation measures across the early window to predict MMLU at the end.
> Given these clarifications, please propose a way to accomplish these while staying as
> close to my existing setup as possible.

Routed to: [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Responses 19–20"), with intake notes on the feature-masking contradiction, per-τ metric
computation, and what the real upper bound is.

### Early-dynamics draft review, third distinct review — 2025-07 (intake 2026-08-22)

> ok, now, back to an alternate version of the chat about the feedback on the project
> report, let me know if this is verbatim the same as a previous answer

Not a duplicate — a third review with different calls (minimal feature slice; one CV axis;
one LambdaMART head; naive extrapolation baselines; clip before logit; average seeds after
splitting). Recorded in
[topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md) ("Third
distinct review"), with cross-review disagreements listed in the open questions.

### Early-dynamics, third-review thread responses 21–22 — 2025-07 (intake 2026-08-22)

> The narrow slice approach seems great, why target the pile perplexity instead of eg MMLU,
> a downstream metric? Perhaps I can very slightly expand the goal to predicting one
> perplexity value and one downstream task?
> I want to do the generalization across model scale if I have to choose one.
> And finally, can you provide more detail about the proposed super simple baselines (sanity
> checks)?

> Excellent! Given the new plan, please layout exactly the steps I should follow to
> implement from scratch to the meeting deliverables we just described. For each step,
> provide the code snippets that I will use.

Routed to: [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Responses 21–22"); her axis/target decisions recorded in the open questions; code kept
out of the doc (structure only) with intake notes on the stub split and placeholder
runtime figures.

### Early-dynamics, implementation state and three questions — 2025-07 (intake 2026-08-22)

> Awesome, so I've extracted all of the following features for each (model size, dataset)
> pair: [131-column feature list — schema recorded in the topic doc] […]
> I intend to use the following initial params for training GBDT: [LightGBM regressor
> params — recorded in the topic doc]
> And I intend to test generalization over an expanding window of model sizes. I've created
> a train vs eval set, and within train I've made it possible to select a 10% val set to do
> hpm tuning.
> Then, I need to decide: Which gbdt hpms to tune, I assume its fine to tune them once and
> then use them going forward? Which of the features should I be normalizing via "Z-score
> Per Model Size Bucket" after applying `log(ppl + 1e-8)` for all perplexity scale
> features? And is there anything else I need to do to prepare my features to use for
> training the gbdt?

Routed to: [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Implementation state: extracted features and training setup"); answer to follow.

(Answer to the three questions above — tuning set, normalisation map, `log1p`, pruning of
fit statistics — recorded under "Answers: what to tune, what to normalise, what to prune",
with intake notes on the ungrouped Optuna split, the perplexity/`log1p` slip, and the
unresolved per-size normalisation for held-out sizes.)

### Early-dynamics, R² vs. RMSE — 2025-07 (intake 2026-08-22)

> I was under the impression that residuals (r^2) and RMSE give you different information
> that you should interpret diffferently and can sometimes poinnt in different directions.
> Is this false?

Routed to: [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("R² vs. RMSE"), with an intake note that the response's within-one-fit argument does not
transfer to features compared across rows — Danielle's intuition holds there.

### Early-dynamics, pruned features and normalisation plan — 2025-07 (intake 2026-08-22)

> Ok, my new set of features is: [67-column pruned list — recorded in the topic doc]
> I've updated all normalization to use log1p instead of log (and I'm only pursuing pile
> perplexity for now). Then, my understanding is I need to:
> 1. Encode the following as 0/1: `is_mixed_dataset`
> 2. Not change: `lr_max`, `lr_final`, `batch_size`, `d_model`, `n_layers`, `n_heads`,
> `mlp_ratio`, `pct_code`, `pct_common_crawl`, `pct_social_media`, `duplicate_rate_pct`
> 3. Take log1p only: `total_steps`, `total_tokens`, `total_tokens_billions`,
> `mean_doc_length_tokens`, `warmup_steps`, `lr_decay_steps`, `full_early_num_steps`,
> `early_lr_decay_num_steps`, `warmup_num_steps`
> 4. For all the rest, if it is in the ppl scale take log1p and then apply the bucket
> normalization
> Is this correct?

Routed to: [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Pruned feature set and Danielle's normalisation plan"), with the answer (signed-log for
diffs; slopes/R² untransformed; optional log10 of learning rates) and intake notes.

### Early-dynamics, z-scoring implementation and unseen buckets — 2025-07 (intake 2026-08-22)

> Does this look like the correct way to calculate zscore aka normalize the features
> (assuming the correct log has already been applied prior to this function call:
> [`zscore_by_param` groupby-`params` implementation] Note "params" is the model size
> feature column.

> so since I'm going to be evaluating generalization across bucket sizes then my eval set
> will be all unseen buckets. so then I guess I'd want to use the next closest
> normalization? it seems this type of scaling will inherently make this type of
> generalization harder

Routed to: [topics/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
("Z-scoring implementation, and the unseen-bucket problem"); resolves the earlier open
question, with a note that global z-scoring is a no-op for LightGBM.

### Lightweight adaptation + ICL examples — undated (intake 2026-08-22)

> Can you do a little bit of an extended search, so like it's fine if it takes a while to
> get maybe like to look through the different examples and then choose three examples of
> this from either like research papers like from archive or benchmarks or strong maybe
> like blogs from companies that talk about how that version of very lightweight
> fine-tuning or adaptation along with in-context learning is being done and what the
> learnings are?

Routed to: [topics/icl-literature.md](topics/reference/icl-literature.md) (dated entry with
a quality caveat — the third example is unsourced).

> Interesting. So, I remember before in-context learning was really a thing, then there was
> this investigation in the idea of trying to learn embeddings, like prompt, I mean, I guess
> it's prompt tuning is what it was called then, but that has a meaning now, but trying to
> learn embeddings that you could use to prompt your models to get the results that you
> wanted before the models were strong enough to be able to understand human language,
> natural language so incredibly well. And so, I guess I wonder for something like the
> context tuning for in-context optimization, whether this is kind of returning to
> something kind of like that, and whether there are other examples of modern approaches
> that are kind of similar to that, where it's a combination of natural language in the
> in-context prompt, and then also like pieces that are tuned, whether that's like
> embeddings or like word combinations that don't necessarily mean as much for humans, but
> are more meaningful for the agents that they're being prompted with, would currently
> exist in that direction.

Routed to: [topics/icl-literature.md](topics/reference/icl-literature.md) (follow-up under
the same entry; response content-free, question recorded as a Danielle-flagged lead for a
lit pass).

> I previously found it very surprising, the comparison between fine-tuning and in-context
> learning, though I think that it's really held out, I guess. It clearly is a direct
> comparison, and the idea that by consuming more tokens, a model is moving towards a,
> quote-unquote, more trained state, kind of, is a parallel to fine-tuning, and so then
> that makes me wonder, we look at, for these large-language models, scaling laws and,
> like, performance curves over the course of training, and is that something that is also
> investigated in the prompt-tuning space in terms of, kind of, like, for example, how many
> examples, like, what is in an in-shot in terms of the performance on a task, or how
> different choices of prompts impact that, like, quote-unquote learning curve.

Routed to: [topics/icl-literature.md](topics/reference/icl-literature.md) (follow-up;
response content-free; Danielle's framing cross-referenced to the ICL-elicitability core).

> Is there an equivalent, like so I guess my understanding of the current research in
> scaling laws in like fine-tuning and post-training, I guess and pre-training, is that
> generally you can compute the scaling laws especially between different types of tasks
> or different types of learning rate schedules based on like the cumulative learning rate
> over time as opposed to just based on the number of tokens and that makes sense because
> there's some concept of like step size of learning and there are a few other metrics
> like the movement of the weight norm or things like this that are identified as like
> kind of x-axes or indicators of the learning process in some type of like fundamental
> way. Is there an equivalent set of values within the space of in-context learning?

> Interesting. So is there, I guess, I hear what you're saying that the number of examples
> could be thought of as like a step size, but I guess I see the number of examples in
> context learning as being more similar to either a compute metric or a number of tokens
> metric, like number of examples, and I'm not necessarily sure that I would expect things
> to scale with the number of examples in the same way that I would expect things to scale
> with like learning rate changes, and so then I'm trying to think about what would be a
> good, what would be some other potential analogies, and I guess one of them might be,
> for example, how often you repeat examples in your prompt, in that like if you
> considered n to be the number of unique examples, and then your step size or your
> learning rate or whatever was like how often each of the examples was repeated, then
> you're arguing maybe you're taking a bigger step on each example. What do you think
> about that idea, and how does that fall into the current like view of how to think about
> in context learning?

Routed to: [topics/icl-literature.md](topics/reference/icl-literature.md) (follow-up;
responses content-free; the unique-examples × repetitions factorial recorded as a
Danielle-flagged protocol lead for ICL-elicitability).

> Interesting. Can you again do a search, this time specifically in recent papers published
> on Archive, and give me an overview of maybe like four senior researchers, so like last
> authors who are working in this space, since presumably their labs are working in this
> space, and then maybe like a few papers that are directly relevant to this?

Routed to: [topics/icl-literature.md](topics/reference/icl-literature.md) (follow-up; paper
titles + one arXiv ID kept as leads; author/lab attributions flagged as unreliable).

### Clean-code preference test — undated (intake 2026-08-22)

> Wow. Huh. Interesting. Okay. So, I guess I didn't realize that I was so interested in
> this before I started asking questions, but I think probably I should table this
> investigation into the related work for now, since my goal is to test this approach out
> practically for myself in a small-scale way first. But it's cool to see that it actually
> does directly link to some very interesting and current research that could be helpful
> for my future research directions. Okay. Cool. So, now maybe we can think about how to
> set up kind of a very simple version that would be practically useful for me in this
> specific setting. And I guess some things that seem very, like, some high-priority
> aspects, I would say I think having some form of automated feedback. And I think that,
> like, probably trying to figure out agent feedback is, like, complex. And so, I was
> thinking something along the lines of having, like, a clear test case. So, basically,
> like, working with individual functions that have a very clear purpose. And then having,
> like, clear tests for each of the functions to verify that the, like, implemented
> version performs according to the expectations. Then using my reference implementations
> and comparing the length of the implementation from the model with the length of my
> implementation to get some sense of, like, is this a, like, was this efficiently
> implemented? Are kind of two of the initial forms of automated feedback that I imagined.
> What do you think about these ideas? Do you have feedback or other thoughts about how we
> can get some automated feedback on these functions?

> Okay, so then I guess the next piece of it that I'm trying to think through is I guess I
> want to be able to give the model as much signal about the piece, like the aspect of the
> example that I care the most about as possible. And I think that for me, it's basically
> like simplicity and what's it called, simplicity and like semantically meaningful naming
> and grouping of the code so that it's easy to read and tell what's happening in the
> code, and they're not like random additional things that like, I don't know, checks that
> are if else on things that are basically simplifiable to a Boolean or something like
> this. So I think they're just like a bunch of coding best practices that I would love to
> just enforce in the model's expectations from the beginning. And I understand that some
> of that is basically having a code base that follows all those best practices. But I
> also think that at the very least, I'd like to know how far I can expect models that
> have been in context prompted a bit to be able to go in terms of just like following
> these practices. So I guess I see this both as like an approach to set up a process for
> like preparing my models to be useful, but also an approach to kind of understanding
> like their limitations. And in that case, I would think that one approach to this would
> be to take some of my like, utterly clean examples, and then farm them out to a bunch of
> different models, where half of the models are trying to make them cleaner, and half of
> the models are trying to make them substantially less clean by inserting all of these
> different things that would actually violate the idea of clean code. So like
> intentionally replacing a succinct way of doing something with a less succinct way of
> doing something. And then basically, I can manually go through, like, I guess I could do
> a first pass to make sure that basically just to filter out things that don't actually
> run. But then I can manually go through to make sure I have like a quick labeling of the
> example implementations of like, bad to good. And then that can be like a little data
> set that can be used for like, giving what's it called? paired paired examples, which
> was a big thing. There's a name for that. What's the name for the idea of not giving
> rewards, but instead giving paired examples good and bad.

Routed to: new staging topic
[topics/clean-code-preference-icl.md](topics/staging/clean-code-preference-icl.md);
closing note in [topics/icl-literature.md](topics/reference/icl-literature.md).

> Interesting. So based on our previous conversation, I guess I'm now realizing that what
> I'm describing is something that I could see doing as a preparation for a workshop paper
> as well, which makes me wonder, specifically in the realm of creating small, high-quality
> coding datasets, whether this specific approach has been done recently and published on.

Routed to: [topics/clean-code-preference-icl.md](topics/staging/clean-code-preference-icl.md)
("Paper potential and prior-art check"); AgentPack / KODCODE kept as leads, with a note that
the style-preference prior art was not actually searched.

> Interesting. Okay. So, yeah, I think it makes sense to start with my functions mainly
> because the thing that I want the model to do is to, in fact, improve data in this
> specific domain, and so it's specifically relevant to me. I guess then the question, then
> this raises like practical questions. So, I guess the easiest types of functions that I
> think exist to start out with would be like parsing functions and data frame
> manipulation functions, because I have a lot of those, and they're also fairly well
> contained. And so then I guess probably the structure that I would imagine is that the
> prompt would involve some type of description of the task, and then it would involve
> some examples, and then there would be some held out examples that would be the like
> learning process where the agent would be asked to execute, and then there would need
> to be some way to get the code produced, test it, and then provide feedback. Does that
> sound? Maybe there would be so like there would be N in context examples in the
> original prompt, and then maybe like M additional interactive examples, and then I
> guess the performance on the additional interactive examples would provide an
> evaluation metric for the initial prompt, but then I guess there's not really a setup,
> like there's not really something built into that setup that would provide an
> evaluation of the like benefit of the active learning steps.

> Fascinating. Okay. And I guess there's an additional access that matters here, which is
> basically the percentage of the context window that I've been using. Because I guess if
> all of these are very short examples, since the percentage of the context window is
> very low, then I wouldn't necessarily expect there to be a large difference that was
> produced in terms of the agent ability between the first active learning and the last
> active learning example. But if instead I wanted to just basically throw all of the
> examples at an agent and just like not, it's not like preparing the agent to be able to
> be used by me interactively, but instead actually just testing how the active learning
> changes performance over time, then looking at that result, I would need to consider
> the impact of the trade-off between more examples versus like a longer context over
> which to remember.

Routed to: [topics/clean-code-preference-icl.md](topics/staging/clean-code-preference-icl.md)
("Experimental structure"); the two goals separated in the open questions.

> So my understanding of how people benchmark these, like as the industry-level production
> quality models, is that often, I guess I've heard people talk about tasks that measure
> how well, like in quotes, a model can use its context, like needle in a haystack type of
> tasks that don't necessarily correlate to like actually how well an agent can use its
> context. But I haven't actually seen things that directly look at this comparison
> between the number of, like the trade-off between more examples for in-context learning
> versus worst performance over the course of the context window, is that, can you look to
> see whether there are any papers in the last year that look at that?

> Is there tooling or best practices in terms of how to use? So I have access to Codex CLI,
> Cloud Code CLI, I guess Gemini. And so I guess I would imagine that the best way to get a
> bunch of examples, the best way to not use an API key, since I'm already paying for the
> subscriptions, to get a bunch of examples of better and worse versions of my code
> samples would be to use the CLI prompting mode, where you pass in a prompt, and then the
> agent gets a certain number of rounds of execution. Are there tools that support that or
> make this easier?

Routed to: [topics/icl-literature.md](topics/reference/icl-literature.md) (many-shot vs.
long-context leads) and [topics/clean-code-preference-icl.md](topics/staging/clean-code-preference-icl.md)
("Tooling for generating variants without API keys", with concrete CLI modes noted).

### Post-intake decisions — 2026-08-22 (second pass)

> great, yes for 1 promot to a project doc, yes for 2 keep staged, yes for 3 crosslisting.
> yes for defering 4/5

Applied: `early-dynamics-prediction` → [potential-projs/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
(`EDP`); `clean-code-preference-icl` stays staged; ICL-opt-7 (repetition-as-step-size
factorial) added to [potential-projs/icl-elicitability.md](potential-projs/icl-elicitability.md)
with the seed quoted in §4; literature gates and the draft-scaffolding review remain parked.

### 2026-08-22 — Heptabase notes analysis (2025, original notes lost)

> sadly I don't have the original doc anymore, but the answer seems useful.

Her prompt to the assistant (verbatim):

> I have started taking notes on my learning and research experiments in this new tool that
> lets me export the data as a markdown even though it was originally different cards on
> different boards, etc. It's been about two weeks that I've been using this tool for notes.
> Please analyze the contents of my notes and (1) highlight the major projects I've been
> working on (either learning or research) and (2) provide analysis, feedback and insight
> around both the note taking and the subjects of the notes.

Routed to: [topics/project-approach-principles.md](topics/reference/project-approach-principles.md)
(seven-track record + workflow feedback) and
[potential-projs/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md) §4
(the CIFAR-10 loss-slope study as EDP lineage, with the quoted "early findings" marked
unverified).

### 2026-08-22 — Workshop deadline calendar (two queries, early August 2026)

> ok, this is an entirely different type of information but its really useful: upcoming
> workshops!

Her prompts (verbatim):

> what are the next few sets of ml, rl or nlp conference workshop deadlines coming up in
> the next month or two?

> amazing, since conferences generally have a wave of deadlines for workshops at around the
> same time, maybe you can find as many major RL, ML and NLP conferences whose workshop
> paper waves happen in the next few months?

Routed to: new [topics/workshop-deadlines.md](topics/reference/workshop-deadlines.md)
(venue/timing accumulator; both answers merged into one dated table, with an intake note on
which dates had already passed and which waves the workshop-paper candidates could
realistically target).

### 2026-08-22 — Generalization / OOD question sequence (2025-01-04 conversation)

> ok, an even older convo, jan 4 2025 not necessarily the most useful answers/I'm not sure I
> trust the citations, but the sequence is worth including

Her prompts (verbatim):

> Is pretraining LLMs considered an unsupervised learning approach

> How have Machine Learning researchers formalized the idea of generalization to compare
> generalization performance in supervised AND self-supervised settings?

> What work has been done on measuring out of domain generalization performance and
> predicting the performance conditioned on the method?

Routed to: new [topics/generalization-and-ood-literature.md](topics/reference/generalization-and-ood-literature.md)
(responses condensed; all citations marked unverified; intake note that the
"predicting performance conditioned on the method" half was never answered and is the
thread that recipe-featurization / EDP pick up).

### 2026-08-22 — CNN-deconstruction logger question (Perplexity, undated, ~late 2024 / early 2025)

> ok another far back one, from perplexity this time.

Her prompt (verbatim):

> I want to slowly deconstruct the improvements in architecture and training procedure that
> were introduced into CNN based vision models from today all the way back to the earliest
> "deep" models. For each step I want to train on CIFAR 10 and ImageNet for a minimal
> length of time to do fast iteration for comparison (not aiming for super competitive
> results), and I want to track key metrics about the optimization landscape as I go. This
> will involve making significant numbers of iterative changes and running a substantial
> number of runs (multiple seeds per configuration + simple hpm sweeps).
>
> What are the top options for a logger setup (not tensorboard or wandb)

Routed to: new [topics/experiment-tooling.md](topics/reference/experiment-tooling.md)
(her requirement statement quoted as the `deconCNN` design statement and linked to the
seven-track record and to GEO / TINY; the logger comparison condensed and flagged as a
vendor-page summary that never evaluated against the stated requirements).

### 2026-08-22 — CNN-deconstruction scoping prompt (Perplexity, undated; with `top-5-cnn-innovations.md` from the Desktop)

Her prompt (verbatim):

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
> the first that I should run experiments on. Ask questions first if you need any
> additional information from me to refine the outcome to exactly what I'm looking for

Routed to: new staging topic [past-projects/cnn-deconstruction-ladder.md](past-projects/cnn-deconstruction-ladder.md) (then a staging topic; moved to past-projects 2026-08-22)
(both framing statements quoted; the timeline and the top-5 file condensed into one table;
intake notes: response answered at architecture level rather than building-block level,
ignored "ask questions first," mixes top-1/top-5 figures, and its Adam-as-standard claim is
wrong for CNN classification). The Desktop top-5 file was folded in rather than copied to
`refs/` (it is short and fully represented).

### 2026-08-22 — CNN recipe fact-finding (Perplexity, undated; 15 short questions)

Her prompts (verbatim, in order):

> did the original resnet paper use batch norm or dropout or weight decay or none of the above

> What data augmentations did the original resnet paper use?

> Does the resnet paper use multiple filter sizes or only 3x3 filters?

> what about resnet18

> Did Google LeNet use dropout? What data augmentation did it use?

> Did Google LeNet (Inception) use weight decay?

> Did VGG use weight decay, dropout, ReLU?

> What were the main contributions of AlexNet to training vision models? What were the
> baseline comparisons for each of these things?

> Was "Local Response Normalization (LRN)" used in VGG or Google LeNet (Inception)?

> Did the resnet paper use AdamW or SGD with momentum?

> did VGG use overlapping pooling?

> was adamW introduced before or after Cutmix

> Was AdamW generally used with a cosine annealing schedule or a step lr schedule?

> Is it standard to train vision cnn based models with BCE instead of CE loss?

> Did Inception use Color Jitter?

> are cifar-10 images black and white?

Routed to: [past-projects/cnn-deconstruction-ladder.md](past-projects/cnn-deconstruction-ladder.md) (then a staging topic; moved to past-projects 2026-08-22)
as a recipe fact table with a verification column (ResNet augmentation answer incomplete —
missing scale aug and PCA color; GoogLeNet augmentation shaky; BCE answer dated), plus an
intake note that the ladder must fix a modern recipe and vary one block per rung rather
than reproduce each paper's full historical recipe.

### 2026-08-22 — Empirical NTK educational overview (Perplexity, undated ~spring 2025)

> ok, lets store the link with this one because there's an interactive version too:
> https://www.perplexity.ai/search/15452705-722c-49aa-8fe3-ad348c1b781a

Her original prompt was not pasted; the response describes itself as an "interactive
educational overview of the empirical NTK" with a seven-section web app (the "Learn with
agents #1" track in the 2025 seven-track record).

Routed to: new [topics/ntk-literature.md](topics/reference/ntk-literature.md) (link stored
in the header; overview condensed; intake note on what the tutorial omits for measurement
use and candidate eNTK readouts for GEO and the CNN ladder); cross-linked from
`project-approach-principles.md` and `cnn-deconstruction-ladder.md`.

### 2026-08-22 — "More linear loss curves = better training?" (two answers; first received)

> ok, then two answers to the same question: "Is there a belief in machine learning research
> that "more linear loss curves indicate better training"? If so, what is it based on?"

Routed to: [topics/loss-curve-forecasting.md](topics/reference/loss-curve-forecasting.md)
(answer 1 condensed; intake note that it argues about smoothness, not linearity, and so
answers a different question; slot left for answer 2).

> note the hypothesis came from my advisor

Recorded in both the topic entry and the EDP lineage note.

> answer 2:

Answer 2 appended to the same topic entry; intake note that it manufactures support (conflates
loss-function with loss-curve smoothness; minor applied papers as evidence) and that neither
answer found a literature basis for *linearity* — the shape priors on record are power-law.

### 2026-08-22 — First CNN ablation dataset analysis (2025-06-12; dataset not on file)

> ok, so this would be much more helpful if I had the dataset I passed in, but we might
> find it later. And either way this is a solid historical note: Jun 12 2025. anyways, my
> statement:

> I have a dataset from training CNN based vision models while removing features that
> have been added for the last many years. Note that the train acc and loss are calculated
> on the augmented data so it isn't comparable to the val.
>
> Please analyze this data

Routed to: [past-projects/cnn-deconstruction-ladder.md](past-projects/cnn-deconstruction-ladder.md) (then a staging topic; moved to past-projects 2026-08-22)
as a dated section reconstructing the 15-config dataset from the response and recording why
the response's reading is wrong (negative train−val gap is the stated augmented-train
artifact, not leakage; "every removal improves" is the short-budget confound, not a flawed
baseline), with the design consequence that budget is a hidden axis of the ladder and a
redo list for when the dataset is found. Gate updated to include locating the dataset.

> it continues:

> These are the results from training CNN vision models on cifar data while removing major
> advancements from the last many years. The hpms are not well tuned, and the training
> loss / acc are evaluated on the augemented data.
>
> I'm interested in the linearity of the loss curves when you plot them in linear(y) vs
> log(x) space. Specifically, if you fit a regression line to the full curve, to just the
> first half too the very beginning, how does the linearity of the loss curve relate to the
> final validation accruacy? how does it relate to how well tuned the setting and hpms are?

Routed to: the same staging section (her spec recorded as the canonical feature definition
— linear-y vs. log-x fits on full / first-half / beginning windows, targets final val
accuracy *and* tuning quality; the response admitted it had no curves, fabricated
final-value proxies, and concluded anyway — zero evidential weight).

> (note that I didn't just take the answers from the agents and use them, but I was curious
> how well different agnets would work + found the convos a useful way to document my
> thoughts and maybe get some useful pointers)

Recorded as a standing reading note at the top of this log and in
`topics/reference/project-approach-principles.md`: the archived conversations are
Danielle's thinking-out-loud plus an informal comparison of agents, not decisions she
adopted; the intake notes that grade responses should be read as part of that comparison.

### 2026-08-22 — Second attempt with per-epoch data (2025-06)

> ok, I tried again with the same inro as last time. then:

(Prompt identical to the 2025-06-12 follow-up above; the attachment was
`regression_analysis_from_first_25_epochs.csv` — a pre-computed per-run regression table.)

Routed to: [past-projects/cnn-deconstruction-ladder.md](past-projects/cnn-deconstruction-ladder.md) (then a staging topic; moved to past-projects 2026-08-22)
(results table: 4-epoch slope |r| = 0.71 → full-curve 0.36; R² r ≈ −0.4; intake notes that
the slope sign is almost certainly misread by the agent, that the window result is the real
finding, that R² indexes slowness here, and that seeds were treated as independent); EDP
lineage note updated to point at it. The expiring S3 links were not stored.

> me: "Within your findings about the correlation of smoothness with accuracy or the
> correlation of linearity / early loss fit line slope with linearity, are there different
> clusters of hpms that behave differently from each other? For example, on the smoothness
> vs accuracy plot it seems that there are clusters, do they correspond to specific hpms?"

Routed to: the same staging section (slope-bin table recorded; it settles the sign —
fastest early decline ↔ highest accuracy, monotone across bins — and shows the clusters
are augmentation regime × a mid-ladder LR change, not "methodology"; redo list extended;
n mismatch between analyses flagged). Second attached file noted:
`epoch_metrics_long_format.csv`. EDP lineage note updated with the resolved direction.

### 2026-08-22 — NAS state of the field (Perplexity, undated ~2025)

> then, another switch up.

Her prompt (verbatim):

> What is the current state of neural architecture search subfield in machine learning
> research

Routed to: new [topics/nas-literature.md](topics/reference/nas-literature.md) (survey
condensed; intake note that the performance-estimation half — zero-cost proxies and
learning-curve extrapolation — is the closest prior art to EDP and the loss-slope study,
with the specific papers to check, and the top-k rank-collapse caution).

### 2026-08-22 — Related work for early-linearity prediction (Perplexity, two rounds, ~2025)

Her prompt (verbatim):

> I'm interested in using the linearity of the early epochs of a loss curve to predict
> final performance of a model. Specifically when training cnn based vision models of cifar
> 10. What related work exists for this?

Routed to: [topics/loss-curve-forecasting.md](topics/reference/loss-curve-forecasting.md)
(round 1 empty; round 2's reading list tabulated — Domhan 2015, LC-PFN, Ding 2024, neural
capacitance, zero-cost proxies — with Baker 2017 added from memory; intake note that
"linearity" is a curve-family choice inside the extrapolation literature, not a gap);
related-work pointer block added to the EDP doc; cross-linked from `nas-literature.md`.

### 2026-08-22 — CIFAR-10 baseline accuracy report (Perplexity, undated ~2025; prompt not pasted)

Her prompt was not pasted; the response titles itself "CIFAR-10 Baseline Accuracy Report for
Popular CNN Architectures" (ResNet, WRN, VGG, DenseNet, ConvNeXt).

Routed to: [past-projects/cnn-deconstruction-ladder.md](past-projects/cnn-deconstruction-ladder.md) (then a staging topic; moved to past-projects 2026-08-22)
(reported numbers condensed; intake note replacing them with canonical original-paper
figures from memory — to be checked — flagging the ResNet-18/50 "82.9 / 76.0%" numbers as
wrong-stem or short-budget artifacts and the near-duplicate "9–14% drop" as a misquote;
pilot's 93.9% @ 25 epochs placed against a ~95% 200-epoch reference).

Her query for the source-links list (supplied afterwards, verbatim):

> CIFAR-10 "baseline" OR "vanilla" accuracy
> model:(resnet OR wrn OR vgg OR denseNet OR convnext)
> report:(test accuracy OR error rate)
> site:arxiv.org OR site:github.com
>
> Show the raw list of source links only (no summary)

(The response titled itself "CIFAR-10 Baseline/Vanilla Model Accuracy Source Links".)

Routed to: the same staging section as a compact unvetted source list (arXiv IDs and
GitHub repos only); noted that the only original-author source is `KaimingHe/resnet-1k-layers`
and the rest are secondary or off-topic.

Her structured search query (verbatim):

> CIFAR-10 "baseline" OR "vanilla" accuracy
> model:(resnet OR wrn OR vgg OR denseNet OR convnext)
> report:(test accuracy OR error rate)
> site:arxiv.org OR site:github.com
> AND title:"ResNet-18"
> AND year:2020..2025
> AND ("under 100 epochs" OR "fast training")
>
> Show the raw list of source links only (no summary)

Routed to: the same staging section (links listed unvetted; intake note that the
short-budget references actually comparable to 25-epoch runs are the DAWNBench ResNet-9 /
hlb-CIFAR10 / airbench speed-run lineage, which the search missed).

> (same query without the `"under 100 epochs" OR "fast training"` clause; raw links only)

Routed to: same line in the staging section (two new IDs: arXiv 2209.01848, 2502.00663).

> (rerun of the `title:"ResNet-18" AND year:2020..2025` query; raw links only)

Routed to: same line in the staging section; the rerun returned a larger set ignoring the
year filter — flagged "ResNet strikes back" (2110.00476), "Revisiting ResNets" (2103.07579),
and "Bag of Tricks" (1812.01187) as the recipe-ablation papers the ladder should cite.

> (budgeted query without the ResNet-18 title filter; raw links only)

Routed to: the same source-list bullet in the staging section.

**Correction (2026-08-22):** an earlier edit had dropped the `## Intake notes` header of
`cnn-deconstruction-ladder.md`, so the sections for the 2025-06-12 analysis, the linearity
spec, the per-epoch slope/R² results, the cluster follow-up, the CIFAR-10 baselines, and the
source lists were silently not written by the commits that logged them. All were
reconstructed and written in this commit; the inputs-log entries above now point at real
sections.

### 2026-08-22 — Optimizer arms for the CNN ladder (Perplexity, undated ~2025; four prompts)

> What optimizers besides Adam W and SGD M are frequently used when training CNN based
> vision models

> If I had to choose between RMSProp and AMSGrad to test alongside SGD-M and ADAM-W to get
> a sense of different optimization dynamics of training CNN based vision architectures
> based on optimizer selection, which should I choose and why?

> Is AMSGrad implemented in pytorch?

> Is RMSProp implemented in pytorch

Routed to: [past-projects/cnn-deconstruction-ladder.md](past-projects/cnn-deconstruction-ladder.md) (then a staging topic; moved to past-projects 2026-08-22)
(arm-set design recorded; intake note that the "frequently used" list misstates practice
and the AMSGrad recommendation gives almost no dynamical contrast to AdamW — Lion / SAM /
RMSProp are the contrasting arms; per-arm LR sweeps required; PyTorch answers correct).

### 2026-08-22 — Earlier post-training project: first-hand account + "the AI2 dataset by Kyle" (last of tonight)

> ok great, a completely different topic, and the last one for tonight!

Her prompt (verbatim):

> So a while ago, I was working on a research project with a friend where we were
> considering the data decide model set, where you have models of a variety of sizes from 4
> million all the way up to 1 billion trained on a variety of different recipes, released
> with checkpoints and evaluations. And we were exploring the effect of post-training,
> specifically supervised fine tuning, but what we found was that using established
> post-training datasets like Tulu, Tulu 3, didn't make any difference in task performance
> on any of the tasks that we tried, ranging from like multiple choice tasks all the way up
> to things like human eval, just like no movement at all, which was really shocking, but
> then when we talked to more people, then they suggested that this actually might be more
> common than we expect and that they were working at AllenAI on creating a dataset that
> actually could be used in fine-tuning to move metrics on very specific tasks. I think
> one of the author's names was Kyle. And so I'm curious whether that dataset has come out
> and also what types of explorations exist for ways to fit pretrained models, whether
> that's supervised fine-tuning or other methods of post-training, but with a focus on
> specifically fairly low-budget approaches to fitting very small models on tasks where we
> can actually evaluate performance.

Routed to: [topics/pretraining-to-posttraining.md](topics/reference/pretraining-to-posttraining.md)
(statement verbatim as the project fact of record; response condensed; intake note that the
FollowIR identification is a guess and a mismatch, the "Olmo-3.1-32B-Instruct" citation
looks fabricated, and the "did the dataset come out?" question stays open — resolve by
asking the contact); origin pointer added to
[potential-projs/movement-microscope.md](potential-projs/movement-microscope.md) §4.

### 2026-08-22 — Tiny-model specialization via the outer layer (undated ~2026; PDF not on file)

> (I meant the convo was the last one, wee have a few more turns)

> ok, so again I don't have the input doc. but me:

Her prompt (verbatim, from speech) is quoted in full in the new staging topic.

Routed to: new staging topic [potential-projs/elicitation-gain.md](potential-projs/elicitation-gain.md) (then a staging topic; promoted 2026-08-22)
(hypothesis restated; response condensed into its A/B/C structure; intake notes: the response
skips the prompt-tuning-is-weak-at-small-scale result that bears directly on the hypothesis,
the honest form is "external large-model-fit interface makes a tiny model useful on a narrow
task"; prior-art starting list; links to TINY within-reach tasks, MIC calibration, and TLC
machinery; proposed first experiment — a capability existence test under an oracle DSL).
The PDF was the TLC draft, already represented by the TLC project doc.

> me: "I'm definitely most interested in the wrapper only. And I guess it feels like there
> are two different types of questions …" (quoted in full in the staging topic)

Routed to: the same staging topic (wrapper-only decision; Q1 cliff curve across DataDecide
sizes at fixed optimizer + budget; Q2 elicitation loop as a pre/post-training movement
detector; response's S_0 / S_opt / ΔS / iterations-to-threshold / stability metrics and two
controls recorded; intake notes link it to ICL-elicitability and MIC, reuse TLC's cliff
machinery, add swapped-executor and wrapper-transfer controls and an answer-leak audit, and
suggest testing Q2 on the earlier project's SFT checkpoints if they exist).

### 2026-08-22 — "Datasets are unknowable" objection and task-side querying (undated ~2026)

> me: "In a somewhat tangential direction, but related to the data-to-side dataset itself
> …" (quoted in full in the REC project doc §4)

Routed to: [potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md)
§4 (her statement verbatim as the origin of the task-conditioned feature family and the
rebuttal to the objection; response's four method families condensed, unsourced); pointer
entry in [topics/data-featurization-literature.md](topics/reference/data-featurization-literature.md)
noting that corpus-scale exact/near-duplicate infrastructure is the prerequisite and
overlaps with the retrieval-storage tooling topic.

> me: "Okay, this has been an incredibly helpful conversation. Can you please write up a
> summary of the topics that we discussed that would make it easy for me to know what type
> of design decisions we kind of agreed on, what the related works are that are linked to
> it, so that I can pull out the pieces that are most relevant to what I'm working on
> today?"

Routed to: [potential-projs/elicitation-gain.md](potential-projs/elicitation-gain.md) (then a staging topic; promoted 2026-08-22)
(the response's "agreed decisions" list recorded as the conversation's settled set, with
the note that wrapper-only is the only explicit decision she made; TLC-draft internals and
the related-work map condensed; AlphaCodium flagged) and a dated note in
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (draft internals; AlphaCodium for the prior-art gate; the wrapper-only sibling as a
candidate optional direction).

### 2026-08-22 — Repo read: datadec data layer against a three-paper plan (undated, ~2026-08)

> are you able to look at this github repo: https://github.com/danielle-rothermel/datadec
> it should be public?

Routed to: [topics/reference/datadecide-data-pipeline.md](topics/reference/datadecide-data-pipeline.md)
(new reference topic for this conversation; the response's claims checked against
`README.md` / `configs/` — all held; its open coverage question answered from
`sources.toml`: all 25 recipes have detail archives declared; the P1–P3 / A / B / C-a
labels belong to an earlier plan not on file, mapping to IRT / REC / TRJ-ANN recorded as
inferred).

### 2026-08-22 — Coverage and LR-derivation provenance (same repo-read conversation)

> the detailed parses exist for these and they've all been pulled, processed and uploaded
> to my private huggingface dataset. I don't remember exactly what I finally landed on for
> the lr derivations but I did my best to get the info and these estimates are the closest
> I could get. I think I scoured the repo, the issues, the drive docs, the paper and maybe
> even asked the paper authors (who generally didn't know a ton of the details because
> this was a huge sweep and they published what they did know...).

(with a screenshot of the private HF dataset listing all 25 recipe folders.)

Routed to: [topics/reference/datadecide-data-pipeline.md](topics/reference/datadecide-data-pipeline.md)
(second entry: coverage settled for all 25 recipes; LR provenance; MPL-fit-as-validation
and checkpoint-config spot-check ideas; the "24 in frame" miscount flagged) and a dated
note in [potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md)
§4 (the three-divergences framing for the data-card paper).

### 2026-08-22 — Training loss availability; sparse small-scale checkpoints; retrain idea (same conversation, two turns)

> ok, the one thing the authors said they could not provide was actual loss metrics across
> the runs. we only have the perplexity measures and accuracy measures for each checkpoint.
> does that become an issue?

> lets not treat "the authors could not provide training loss" as a fact yet, I have to
> double check to be sure I'm not misremembering. but reconstructing the missing quantity
> sounds like a good idea. also, especially the much smaller models have very few
> checkpoints (like 4-10) which was an issue back when I was trying to predict final values
> from early curve sections. I'm not sure if it will be an issue here but those are
> conveniently also the scales that it would be very doable to retrain some examples.
> thoughts?

Routed to: [topics/reference/datadecide-data-pipeline.md](topics/reference/datadecide-data-pipeline.md)
(third entry; the "no training loss" claim held as unconfirmed per Danielle, with the one
repo fact — `train_cross_entropy` exists in the scaling-law checkpoint-loss table but only
sparsely at 150M–1B; the "only unchecked gate" remark corrected against the 2026-08-21
spacing answer) and dated notes in
[potential-projs/trajectory-statistics.md](potential-projs/trajectory-statistics.md) §4
(small-scale density), [potential-projs/tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md)
§4 ("DataDecide-dense" substrate), and
[potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md) §4
(own-mixture CE reconstruction; fourth ledger entry, unconfirmed).

### 2026-08-22 — WSD arms in DataDecide-dense (same conversation)

> if I was going to do DataDecide-dense I'd want to also do WSD. because if we're investing
> in getting to the point that we can train and doing a grid then the value of having the
> smallscale wsd becomes much higher than the cost

Routed to: [topics/reference/datadecide-data-pipeline.md](topics/reference/datadecide-data-pipeline.md)
(fourth entry) and dated notes in [potential-projs/wsd-suite.md](potential-projs/wsd-suite.md)
§4 (small-scale pilot, WSD-opt-3 first; design cautions) and
[potential-projs/tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md) §4
(annealed-readout SNR hypothesis at small scale).

### 2026-08-22 — Paper-reproduction summary; BoolQ; four objections (same conversation, two turns)

> ok, then, I was having an agent try to reproduce the different claims from the datadecide
> paper with my postprocessed data to be sure that we can, and the summary of the parts we
> reproduced succesfully is interesting and reminded me of something: boolq is basically
> always sitting at random noise and has VERY high variance. and it makes me wonder whether
> its really so hard or whether something about the task formatting, etc is adversarial
> especially to small models. is that a question that fits somewhere in our 4 project
> design? and do any of these other verifications prompt additional thoughts?

(with the pasted reproduction summary of `docs/paper-validation-report.md`), then:

> questions:
>
> * so when I made a bump plot of the ordering across the model sizes, recipes, etc (tried
>   a few different things) as far as I could tell the ordering is super super super
>   consistent however you slice it. there might be crossovers but they're basically two
>   lines that are the same and are just jittery. so I'm a bit skeptical about the crossover
>   conclusion, but I might not have covered all the cases I thought I did (this was quite a
>   while ago)
> * would the "Broken as measurement" result also come from the task just being universally
>   too hard for this scale of models? because thats what reviewers have all concluded so
>   its unclear that IRT would distinguish this?
> * I'm a bit skeptical about "<= 1% compute" metrics because most of the model sizes don't
>   provide anywhere near that level of granularity if we're normalizing within size, and if
>   we're normalizing by 1B compute full training then that seems strange. thoughts?
> * also, there are definitely some dataset abnormalities, like 750M only has 1 seed that
>   trains fully I think, etc.

Routed to: [topics/reference/datadecide-data-pipeline.md](topics/reference/datadecide-data-pipeline.md)
(fifth entry: reproduced numbers as reported, both prompts verbatim, routing map);
[potential-projs/irt-reanalysis.md](potential-projs/irt-reanalysis.md) §4 (BoolQ
autopsy + variance-structure argument + format intervention; noise-aware crossings;
frontier design brief); [potential-projs/trajectory-statistics.md](potential-projs/trajectory-statistics.md)
§4 (spread-to-noise 0.798 as TRJ in embryo; drift-attributable crossings);
[potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md) §4
(validation report + coverage/abnormality ledger in the data card);
[potential-projs/README.md](potential-projs/README.md) program notes ("DataDecide with
error bars", candidate only).

### 2026-08-22 — Reproduction batches two and three; float-matching correction (same conversation, three turns)

> ok great, so then digging through the "directionally correct results" gave this

> Ok, then the last set of findings:

(each with a pasted summary from `docs/paper-validation-report.md`), then:

> "#1 is the most important finding in all three batches, and it lands squarely in P3's
> lap" is actually a methodological issue on our part I think. no human would try to
> compute match with floats, let alone integers, this would be a bucketed comparison

Routed to: [topics/reference/datadecide-data-pipeline.md](topics/reference/datadecide-data-pipeline.md)
(sixth entry: as-reported numbers, the correction as the record);
[potential-projs/irt-reanalysis.md](potential-projs/irt-reanalysis.md) §4 (margin
demoted to object of study; normalized correct probability as the response variable;
metric hierarchy; one-ability-two-difficulty-regimes null for the cluster claim; BoolQ
predictive-without-valid twist); [potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md)
§4 (compute-matched claim unassessed not refuted; shared matcher with TRJ-3; predicate
liveness guard); [potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md)
§4 (validation-section thesis; three-way classification; definition-matching pass).

### 2026-08-22 — Provenance statement for the reproduction pastes (end of conversation)

> ok, so that was the last set in the convo. the one thing I want to clearly indicate with
> all claims from the pastes is that I had an agent take a first stab at writing the
> verification code and then iterate based on findings I thought were suspect or bad
> methodology, but I will not consider these findings to be real until I personally
> read/debug/run/analyze the findings myself. But these are good flags for where I should
> look first.

Applied: a standing status statement at the head of the reproduction entries in
[topics/reference/datadecide-data-pipeline.md](topics/reference/datadecide-data-pipeline.md)
and a one-line provenance caveat on every §4 note that cites reproduction numbers
(irt-reanalysis ×2, trajectory-statistics, annealed-readouts, recipe-featurization ×2,
README program note).

### 2026-08-22 — Deep-research report: code feature extraction (brief dated 2026-07-03)

> great! ok, a completely different topic, I did a deep research prompt based on the
> attached doc. and got the answering report.

(attached: her own Report-2 brief and the resulting report, plus the two governing
guideline docs; stored at `~/drotherm/data/convo-artifacts/2026/2026-07-03-code-feature-extraction-deep-research/`;
the deep-research prompt itself was not pasted.)

> oh, please copy them, the paths will change

> lets actually move the docs from last round to that location too and make that the
> default destination for convo artifacts we want to refer to.

Convention adopted 2026-08-22: conversation artifacts the planning docs refer to
(briefs, reports, PDFs, search archives) live under `~/drotherm/data/convo-artifacts/<year>/`
in a per-conversation folder, not in the repository; planning docs cite the absolute path.

Routed to: [topics/reference/code-feature-extraction-tooling.md](topics/reference/code-feature-extraction-tooling.md)
(new reference topic: the brief's contract condensed; the report's tool inventory as a
seed list; intake note that the report broke the brief's form rules and omitted three of
four deliverables; links to TLC, the clean-code staging topic, and ELI's verifiers).

### 2026-08-22 — SciSpace summary: three prompt-compression / prompt-optimization papers

> Summarize the key arguments in these 3 papers: Learning to Compress Prompt in Natural
> Language Formats, Automated Prompt Engineering for Cost-Effective Code Generation Using
> Evolutionary Algorithm, Discrete Prompt Compression with Reinforcement Learning

(Danielle's note: the search "came with an answer but a bunch of really useful background
docs and search archives", moved to
`~/drotherm/data/convo-artifacts/2026/scispace-prompt-compression-method-papers-agent-artifacts-zip_180fe9cc-24fb-4ab2-8dc5-61a6611f64ce_1787422486/`.)

Routed to: [topics/reference/prompt-compression-and-optimization-literature.md](topics/reference/prompt-compression-and-optimization-literature.md)
(new reference topic: the three papers condensed, artifact pointer, agent editorializing
dropped) and dated notes in
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (prior-art gate additions; Nano-Capsulator as the nearest objective) and
[potential-projs/elicitation-gain.md](potential-projs/elicitation-gain.md) §4 (EPiC /
ATSP as the nearest budgeted outer optimizer; positioning sentence).

### 2026-08-22 — SciSpace search: HumanEvalExplain results

> I wish to find all papers that report performance results on the HumanEvalExplain
> subtask of the HumanEvalPack as introduced by "OctoPack: Instruction Tuning Code Large
> Language Models" 2023 from Muennighoff et al.

> I'm interested in all models, all language covered, and any papers since the release of
> OctoPack. I want to build a comparison plot of correctness versus explanation length
> based on different forms of applying models (eg different prompt formats, etc) and
> different models used.

(artifacts at `~/drotherm/data/convo-artifacts/2026/scispace-humanevalexplain-results-agent-artifacts-zip_6e33d176-efc7-49db-8322-a5b1604bfd20_1787422834/`)

Routed to: [topics/reference/humanevalexplain-results.md](topics/reference/humanevalexplain-results.md)
(new reference topic: the consolidated results table, protocol, the three source papers,
the transcription-shift error flagged, editorializing dropped) and a dated note in
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (HumanEvalExplain as a ready-made TLC-1 task set; the plot must be generated, not
found).

### 2026-08-22 — SciSpace prior-art search for the NL-bottleneck code autoencoder (three agent sessions)

> this artifact bundle has some of the most important citations, etc for my future work,
> can you look through the files there and index them clearly so that we can find what
> we're looking for in the future?

Her search brief (the "exhaustive conceptual and literature-level search" prompt with
the four-component match criteria and four output requirements) and her scope answers
("1. Prioritize the exact mechanism … 4. include all optimization methods even if they
aren't explicitly framed that way as long as they don't involve updating LLM weights")
are recorded in full in the reference topic. Pasted: the Dec-2025 FINAL VERDICT, the
Dec/Jan comparison, the Jan-2026 summary, and the ICBINB literature-grounded
recommendations.

Applied: `INDEX.md` written inside the bundle
(`~/drotherm/data/convo-artifacts/2026/scispace-nl-latents-rw-agent-artifacts-zip_cc4d31ce-8970-4a5a-9cd1-248b327a0b06_1787423020/`) — six sections: verdict docs in reading order, ICBINB grounding docs, her own
ICLR-2026 ICBINB draft PDF + the agent's read, deep dives on the three closest papers,
candidate JSONs, and the ~60 search CSVs grouped by query family; caveats on confidence
figures and identifier slips.

Routed to: [topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md)
(new reference topic) and a dated note in
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (verdicts; ICBINB grounding supersedes the prior-art stub list; GenDLN flagged for
her own read; identifier slips).

### 2026-08-22 — SciSpace deep review: coding datasets and benchmarks

> next artifact, lets index this one too

> What are the most frequently used coding datasets/benchmarks for LLMs as of late 2025
> and 2026? What tasks do they focus on and how are they used. What are the general
> strengths and weaknesses of these datasets/benchmarks? How do they relate to very recent
> coding datasets/benchmarks released mid 2025 to early 2026?

> I want to get a general understanding of the coding benchmark space, primarily from the
> perspective of NLP with LLMs use cases as an ML researcher, but I do want to include any
> highly relevant topics from more of the CS program synthesis side of things. Then, I
> want to understand what the possible focus areas are, aiming for qualitative insights and
> breadth more than depth.

Applied: `INDEX.md` inside the bundle (`~/drotherm/data/convo-artifacts/2026/scispace-coding-datasets-and-benchmarks-agent-artifacts-zip_bdc06926-5b5a-465b-b70b-9af7aa5a4fcd_1787423366/`) — the two report drafts, the 30-paper
source set keyed to the reference numbers, the merged CSVs with extracted columns, the
deep-search tables (two byte-identical duplicate pairs noted), raw query results, and a
list of well-known benchmarks the review does not cover.

Routed to: [topics/reference/code-benchmarks-landscape.md](topics/reference/code-benchmarks-landscape.md)
(new reference topic) and a dated note in
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (task-set choice: contamination critiques; code-reasoning benchmarks as the nearest
neighbour to reconstruct-from-description).

### 2026-08-22 — SciSpace deep review: prompt optimization

> artifact (lets just index all of these going forward).

> What is the current state of the Prompt Optimization subfield? What are the related
> works that are foundational, and what are the strongest recent themes? Please focus
> especially on approaches that use LLMs to do the prompt optimization and on approaches
> that optimize not just the prompt but also other aspects of the system (for example
> sampling hpms, tool use, etc). My goal is to get a general understanding of the subfield
> before diving deeper into specifics, so prioritize breadth and qualitative comparisons.

> Please include as broad a set of methods as possible, I'd say focus mostly on things like
> prompt optimization, so RAG systems are a bit further afield for this deep research. I'm
> interested across broad domains, but especially in the realm of verifiable tasks like
> code generation. And focus on recent developments but provide a brief intro on the
> historical perspective.
>
> For context, I'm currently writing a paper on compressing functions using an frozen
> pretrained black box LLMs as encoders and decoders, with instructions to compress the
> input function and then reconstruct it in a way that preserves functionality not surface
> form. We're using an external LLM to optimize the compression prompt. So I'm interested
> in things related to this but more interested in getting a broad understanding of the
> landscape.

Standing instruction adopted: every artifact bundle gets an `INDEX.md` at intake.
Applied here: `INDEX.md` in `~/drotherm/data/convo-artifacts/2026/scispace-prompt-optimization-agent-artifacts-zip_97f96bdd-5f03-456a-8061-f2e8204d51c2_1787423566/`.

Routed to: [topics/reference/prompt-optimization-landscape.md](topics/reference/prompt-optimization-landscape.md)
(new reference topic; omissions of canonical anchors flagged) and dated notes in
[potential-projs/elicitation-gain.md](potential-projs/elicitation-gain.md) §4 (the
system-level cluster as the outer loop's positioning set) and
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (where the optimizer loop sits; prompt-length-vs-performance as quotable positioning).

### 2026-08-22 — SciSpace deep review: code compression

> What is the related work on code compression? Both classic approaches that involve
> learning an embedding that can be reconstructed and more recent approaches that somehow
> use LLMs as components in the compression and reconstruction process?

> I'm currently writing a paper on compressing functions using an frozen pretrained black
> box LLMs as encoders and decoders, with instructions to compress the input function and
> then reconstruct it in a way that preserves functionality not surface form. Then, I'm
> interested in a broad overview of the landscape of code compression, both the
> foundational approaches and more recent approaches. Both training models and using LLMs.
> I want to understand the landscape, the types of comparisons that are done, the common
> methods, the standard positioning, the framing of why this would be useful, etc. And I
> want to understand that approaches that are similar to the one that we're proposing to
> understand such that they might be baselines or related work.

Applied: `INDEX.md` in `~/drotherm/data/convo-artifacts/2026/scispace-code-compression-agent-artifacts-zip_ae5ad3b7-95bd-4c9c-8caf-394ff76ac5fe_1787423736/` (173 files; merged corpora with extracted columns, ~100
deep-search tables by query stem, raw results, a papers-to-pull-first list, and the
broken-citation-numbering warning).

Routed to: [topics/reference/code-compression-literature.md](topics/reference/code-compression-literature.md)
(new reference topic); a dated note in
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (Maveli–Vergari–Cohen invertibility paper as the first prior-art gate item; landscape
for the related-work section; rate–distortion axis; comparisons to match); and an added
intake note in [topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md)
that the novelty verdict missed this paper.

### 2026-08-22 — correction: the Maveli et al. paper is unrelated

> so, "Can LLMs Compress (and Decompress)? Evaluating Code Understanding and Execution via
> Invertibility" seems like its code compression, but actually its using coding llms to try
> to forward predict and reverse predict the effect of four lossless compression models. it
> seems so promising and yet it really isn't very related to my work at all sadly

Applied: demoted in [topics/reference/code-compression-literature.md](topics/reference/code-compression-literature.md),
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4, [topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md),
the topics README row, and the bundle's `INDEX.md`; recorded as checked-and-unrelated so
it is not re-flagged.

### 2026-08-22 — SciSpace paper summary: Mixture of Parrots

> I want you to summarize this paper's key findings.
> https://proceedings.iclr.cc/paper_files/paper/2025/file/5bc3356e0fa1753fff7e8d6628e71b22-Paper-Conference.pdf

Applied: `INDEX.md` in `~/drotherm/data/convo-artifacts/2026/scispace-summarize-mixture-of-parrots/` (full-text markdown of the paper + the agent summary,
noted as truncated at its start).

Routed to: [topics/reference/moe-literature.md](topics/reference/moe-literature.md)
(dated entry: claims, why it matters for PART/MSUITE) and a dated note in
[potential-projs/moe-partitions.md](potential-projs/moe-partitions.md) §4 (the citation
behind the "mixture of parrots" line; mechanism for the reasoning null; the
knowledge-vs-reasoning decomposition as the comparison to reproduce on the sweep).

### 2026-08-22 — SciSpace literature review: evaluation metrics at small scale (two versions)

> Please do a literature review of papers related to metrics for evaluation of language
> models at small scale, especially for downstream tasks. One example of a related paper
> would be https://arxiv.org/abs/2605.18607 which you can start with.

> I want you to significantly expand sections 3.4, 3.5, 5.2, 5.3, 5.4, 6.2, 6.3, 7.2, 7.3,
> 7.4, 7.5, 7.6, which more details about the specific experiments and key takeaways. Also
> create a comparison table summarizing the most important papers with their methods,
> evaluation metrics, key results, followup questions, potential weaknesses, and any other
> crucial information.

(pasted: version 1 as markdown, version 2 as a LaTeX document with a 34-row comparison
table.)

> then the artifact bundle: `scispace-eval-of-llms-agent-artifacts-zip_e11a0b3d-…`

Applied: `INDEX.md` in `~/drotherm/data/convo-artifacts/2026/scispace-eval-of-llms-agent-artifacts-zip_e11a0b3d-f220-45af-9d38-6a50581427b3_1787424353/` (seed paper PDF + cropped figures, both review rounds,
insight extraction, 32 downloaded PDFs, search CSVs). The seed paper PDF settled the
author list: Patel, Reddy, Mosbach, Bahdanau — version 2's AI2 author list was
fabricated; corrected in the reference topic and the four §4 notes.

Routed to: [topics/reference/small-scale-evaluation-metrics-literature.md](topics/reference/small-scale-evaluation-metrics-literature.md)
(new reference topic: the seed paper condensed, the landscape by theme, intake notes on
the conflicting author lists, v2's swapped bibliography entries, and v2 restructuring
instead of expanding the requested sections) and dated notes in
[potential-projs/tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md)
§4 (incumbent proxy for small-scale decisions),
[potential-projs/irt-reanalysis.md](potential-projs/irt-reanalysis.md) §4 (frontier
baseline), [potential-projs/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
§4 (training-time forecasting prior art; NeuNeu),
[potential-projs/datadecide-data-card.md](potential-projs/datadecide-data-card.md) §4
(a published consumer to re-derive), and a pointer in
[topics/reference/loss-curve-forecasting.md](topics/reference/loss-curve-forecasting.md).

### 2026-08-22 — SciSpace literature review: pretraining and midtraining toward a target task suite

> Please do a literature review about pretraining and midtraining to target a specific
> task or suite of downstream tasks.

Applied: `INDEX.md` in `~/drotherm/data/convo-artifacts/2026/scispace-pretraining-task-analysis-agent-artifacts-zip_73fb24c0-fa4f-4ea3-b380-6dbe92b7e173_1787424452/` (report md/tex/pdf, insight extraction, six downloaded PDFs
with figures, search CSVs; caveat list of the LM midtraining canon the review missed).

Routed to: [topics/reference/targeted-pretraining-midtraining-literature.md](topics/reference/targeted-pretraining-midtraining-literature.md)
(new reference topic; LM-relevant entries only, drift flagged) and dated notes in
[potential-projs/functional-featurization.md](potential-projs/functional-featurization.md)
§4 (midtraining as the intervention FUNC measures; 2512.07783),
[potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md) §4 (pointer),
and [topics/reference/pretraining-to-posttraining.md](topics/reference/pretraining-to-posttraining.md)
(pointer).

### 2026-08-22 — SciSpace literature review: synthetic data for LM training

> I would like you to review the literature on Synthetic data for language model
> pretraining and finetuning. Specifically, please focus on works related to scaling of
> models and/or data, methods such as rewriting, and how to avoid overfitting. After you're
> done, compile all of the results into a LaTeX pdf, and include figures from the papers as
> appropriate.

(pasted: the agent's `final_synthesis.md` — four per-paper summaries — not the LaTeX
review, which is the deliverable in the bundle.)

Applied: `INDEX.md` in `~/drotherm/data/convo-artifacts/2026/scispace-synthetic-lm-training-agent-artifacts-zip_888d8c4b-0b22-4403-8619-693d35468c3e_1787424684/` (LaTeX/PDF review structure, arXiv→title map for the 14 key
PDFs and their per-paper summaries, figures, peripheral PDFs, CSVs; two unidentified PDFs
flagged).

Routed to: [topics/reference/synthetic-data-literature.md](topics/reference/synthetic-data-literature.md)
(new reference topic: paper set by theme from the PDF's structure; the four summaries
condensed) and a dated note in
[potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md) §4
(syntheticity/rephrasing feature; collapse references), with a pointer in
[topics/reference/data-featurization-literature.md](topics/reference/data-featurization-literature.md).

### 2026-08-22 — SciSpace literature review: distillation in LLMs (six questions)

> I want you to do a literature review of distillation in LLMs, focusing on the following
> areas: 1) the size of the teacher and student, potentially including scaling laws, 2)
> the objective to use for the distillation loss, and how to combine it with CE loss or
> other losses, 3) repetition of logits in distillation compared to repetition of raw data
> tokens in pretraining, 4) what conditions lead to distillation performing better than
> training from scratch, 5) the difference between distilling from pre-trained vs.
> post-trained models and 6) the difference in performance between distilling from a
> pre-trained teacher and then doing post-training, versus only distilling from a
> post-trained teacher.

Applied: `INDEX.md` in `~/drotherm/data/convo-artifacts/2026/scispace-llm-distillation-agent-artifacts-zip_deb5ff94-1238-4180-81b2-62dc9cedad35_1787424886/` (report md/tex/pdf with figure manifests; on-topic vs.
off-topic split of the 34 downloaded PDFs; merged CSVs with question-aligned columns;
missing canon and the duplicated reference block noted).

Routed to: [topics/reference/distillation-literature.md](topics/reference/distillation-literature.md)
(new reference topic: per-question condensation with evidence quality), a dated note in
[potential-projs/movement-microscope.md](potential-projs/movement-microscope.md) §4
(distillation arm defaults; the unclaimed sequential-vs-direct comparison; from-scratch
control), and a pointer in
[topics/reference/pretraining-to-posttraining.md](topics/reference/pretraining-to-posttraining.md).

### 2026-08-22 — SciSpace literature review: alternatives to CE / NLL for evaluation

> I want you to do a literature review of alternatives to CE Loss or Negative Log
> Likelihood for evaluation. I specifically want you to look at metrics that are meant to
> be a replacement for loss. I want you to exclude task accuracy via generation or
> ranking, or any similar common metrics. One example of what I might want is a
> modification of NLL that excludes or reweights certain tokens. I want you to especially
> pay attention to methods which do NOT depend on vocabulary, tokenization, or model
> architecture.

Applied: `INDEX.md` in `~/drotherm/data/convo-artifacts/2026/scispace-alt-eval-metrics-for-llms-agent-artifacts-zip_69962c47-a151-4f10-b0bc-84aa7faabc11_1787425013/` (report md/tex/pdf, 10 key PDFs mapped to papers, figure
manifest, 18 further full texts, the 57 off-topic arXiv downloads flagged, merged CSVs
with method columns; missing canon listed).

Routed to: [topics/reference/loss-alternative-metrics-literature.md](topics/reference/loss-alternative-metrics-literature.md)
(new reference topic: evaluation-side families only; training-objective drift, the crank
source, and the "architecture-independent" mislabel flagged), dated notes in
[potential-projs/tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md)
§4 (the which-loss axis; candidate TINY-opt) and
[potential-projs/datadecide-data-card.md](potential-projs/datadecide-data-card.md) §4
(bits-per-byte column for the PPL tables), and a pointer in
[topics/reference/small-scale-evaluation-metrics-literature.md](topics/reference/small-scale-evaluation-metrics-literature.md).

### 2026-08-22 — SciSpace literature review: alternatives to CE loss for LLM training

> I want you to do a literature review of modifications to, or alternatives to Cross
> Entropy Loss for LLM pretraining or finetuning. Examples include completely differently
> training objectives, as well as modifications such as reweighting the loss from certain
> tokens to be more important. I do not want you to include any commonly studied RLHF
> methods, unless they are applied to pretraining.

Applied: `INDEX.md` in `~/drotherm/data/convo-artifacts/2026/scispace-alts-to-CE-loss-llm-pretraining-agent-artifacts-zip_e225323f-208b-41a7-abd2-1ce72571fd27_1787425213/` (report md/tex/pdf; the 10 parsed papers mapped to stems;
further full texts; the mostly off-topic arXiv folder flagged; merged CSVs with
objective/mechanism columns; RLHF leakage and missing canon listed).

Routed to: [topics/reference/training-objective-alternatives-literature.md](topics/reference/training-objective-alternatives-literature.md)
(new reference topic, RLHF-adjacent entries removed), a dated note in
[potential-projs/token-movement.md](potential-projs/token-movement.md) §4 (token
selection objectives as TOK's mirror; a T0 coincidence test), and a pointer in
[topics/reference/token-level-literature.md](topics/reference/token-level-literature.md).

### 2026-08-22 — SciSpace literature review: layer looping / weight tying (report only)

> Do a literature review on layer looping or repeating, or layers with tied weights, in
> machine learning. I'm referring specifically to the mechanism in the AlBERT and
> Universal Transformers papers, and others, where on set of layer weights is used
> multiple times on one token. I want you to start with papers that cite AlBERT and
> Universal Transformers and fit my description, but also include any other potentially
> relevant papers. Focus especially on papers about LLMs from the past 5 years.

> artifact: download kept failing, I don't think I care enough about this family to care
> so lets stick to the resulting report.

Routed to: [topics/reference/layer-looping-literature.md](topics/reference/layer-looping-literature.md)
(new reference topic, no project link; broken citation numbering, an ALBERT factual slip,
and the missing 2025 LLM-scale canon flagged). No bundle, no INDEX.

### 2026-08-22 — SciSpace review: regularization for MoE transformer LMs on repeated data

> Write me a list of regularization methods for machine learning models. Specifically, I
> want to avoid overfitting Mixture of Experts Transformer Language Models on repeated
> data. I want you to include both well-supported ideas with extensive literature
> throughout general machine learning, such as dropout and weight normalization, as well
> as newer techniques that might be specific to MoE models or transformers. For each
> method, include citations.

Applied: `INDEX.md` in `~/drotherm/data/convo-artifacts/2026/scispace-regularization-methods-moe-agent-artifacts-zip_6c756278-eb52-4294-a000-1b2d39a29157_1787425549/` (report, the companion open-access URL list with 16 extra
canonical papers, insight extraction, ~50 search files; missing canon and off-target
citations listed).

Routed to: [topics/reference/regularization-literature.md](topics/reference/regularization-literature.md)
(new reference topic; the repeated-data mismatch and the missing token-crisis paper
flagged), a dated note in
[potential-projs/moe-recipe-suite.md](potential-projs/moe-recipe-suite.md) §4 (matched
regularization recipe as a stated design choice; read routing at matched epochs), and a
design-constraint line in
[topics/staging/datadecide-dense.md](topics/staging/datadecide-dense.md).

### Post-intake decisions — 2026-08-22 (SciSpace batch)

> amazing! ok, we've finished with scispace, which appears to be a source of many many
> citations but not particularly useful answers. then, what processing steps should we
> take?

> 1 and 2 yes. 3, (b) now (c) later sounds right. 4) yes, TLC is so lets write it.
> 5) agreed, promote all three 6) grouping sounds great, and last sounds good.

Applied (1, 2, 5 in this commit): `INDEX.md` written for the three un-indexed folders
(prompt-compression bundle, HumanEvalExplain bundle, the 2026-07-03 deep-research
folder); a dated SciSpace process entry in
[topics/reference/project-approach-principles.md](topics/reference/project-approach-principles.md)
with the five failure modes and their instances; promotions — TINY-opt-5 (which-loss
axis) in [tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md),
TOK-opt-6 (selection–movement coincidence) in [token-movement.md](potential-projs/token-movement.md),
MIC-opt-5 (sequential vs. direct distillation) in
[movement-microscope.md](potential-projs/movement-microscope.md). Steps 3, 4, 6 applied in the following commits:
[litreview/citation-verification-ledger.md](litreview/citation-verification-ledger.md)
(every arXiv ID that entered the reference topics today — 208 rows — tagged
agent-supplied vs. Claude-added, with the gate it feeds; verification run parked);
[litreview/tlc-litreview-plan.md](litreview/tlc-litreview-plan.md) (five subdomains with
seeds on file, six gate items incl. the GenDLN read, bounded orchestration by reference
to the REC plan); the topics README reference table regrouped into eight themed
sections.

### Post-intake decisions — 2026-08-22 (data-layer conversation)

> 1. I think data-card/ validation work should be its own project doc, so b as you say.
> 2. yes for folding BoolQ into IRT as a named direction
> 3. yes to data decide dense + wsd, I think this convo made it seem much more plausible
> 4. yes, lets add them

Applied: new [potential-projs/datadecide-data-card.md](potential-projs/datadecide-data-card.md)
(`DCARD`; the four data-card notes moved out of REC §4 with a pointer left behind; README
row); IRT-8 BoolQ autopsy, IRT-9 margin decomposition, IRT-10 format intervention added to
[potential-projs/irt-reanalysis.md](potential-projs/irt-reanalysis.md) (IRT-10 cross-listed
on ELI); new staging topic [topics/staging/datadecide-dense.md](topics/staging/datadecide-dense.md)
consolidating the design constraints from TINY/TRJ/WSD §4; two open items (training-loss
availability; 750M seed coverage) in [open-questions-answered.md](open-questions-answered.md).

### Post-intake decisions — 2026-08-22 (third pass)

> (1) promote (2) I did run a few versions of this, I think we need a new section that
> includes previous projects that were started but not finished. this and the loss slope
> prediction work go into that section.

Applied: `tiny-model-specialization` → [potential-projs/elicitation-gain.md](potential-projs/elicitation-gain.md)
(`ELI`), cross-listed on TLC, TINY, MIC, ICL; new [past-projects/](past-projects/README.md)
section with [cnn-deconstruction-ladder.md](past-projects/cnn-deconstruction-ladder.md) and
[loss-slope-prediction.md](past-projects/loss-slope-prediction.md) (split out of the ladder
staging topic; the SFT no-movement project noted as a candidate third record). Items 3
(EDP related-work gate) and 4 (open-questions entry for the AI2 dataset) were approved in a
follow-up ("yes to both 3 and 4") and applied.

### 2026-08-22 — second novelty check for the NL-bottleneck code autoencoder (answer only)

> ok, ready to go?  for this next one I'm skipping my prompt and just passing the answer:

Her prompt was not pasted; the response plays it back as compressing code into a
natural-language latent with an RL/search optimization loop and a frozen LLM decoder
(it calls the idea "CodeVLAE"). Verdict: "Equivalent Method Already Published", high
confidence — headline match Language Bottleneck Models (2506.16982), plus OverLang,
APRIL, Proof2Silicon, Concept Bottleneck LLMs, and two items already on file.

Routed: [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (the verdict reversal and what it rests on); full claim table and intake notes in
[topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md);
LBM added to [litreview/tlc-litreview-plan.md](litreview/tlc-litreview-plan.md) subdomain A
and gate item 1 (Danielle's read, with GenDLN); four IDs added to the citation ledger.

> yeah, LBM is very relevant, but its using the structure to grade student responses, very
> non-verifiable, and they only try 3 different prompts if I remember correctly so its
> definitely prior work, but doesn't quite give either a baseline, a method, or comparative
> results.

Applied: recorded as her read in the TLC §4 note and the prior-art reference entry; LBM
half of plan gate item 1 closed (GenDLN remains); positioning = nearest named framework in
§2, not a baseline.

### Undated (~2025; intake 2026-08-22) — annealing data quality, interactive-app request (second answer)

> I want to better understand the recent research around data quality as it affects LLM
> annealing, especially changing data from pre-training to the annealing stage. Please find
> one or two recent papers and visualize their results in an interactive educational
> app/document to guide me through understanding this content. I'm a phd student so target
> your level of depth to that.

The answer is a long browsing report (Llama 3, MiniCPM, Hägele et al., FineWeb-Edu, Phi-4,
Nemotron-CC, YuLan-Mini); the interactive app it mentions was not passed. Routed:
[topics/reference/schedules-and-annealing-literature.md](topics/reference/schedules-and-annealing-literature.md)
(new entry, condensed to what the Oct-2025 survey entry lacks; PDPC misattribution and
unsourced editorial numbers flagged); short ANN §4 note on decay-phase gradient statistics as
a decay-branch instrument; seven IDs added to the citation ledger.

### Undated (~2025; intake 2026-08-22) — annealing data quality, third answer (two versions)

> I want to better understand the recent research around data quality as it affects LLM
> annealing, especially changing data from pre-training to the annealing stage.

Version 1 drifts into instruction tuning / continual learning; version 2 is mostly term
collisions (Annealed-RLVR, simulated annealing for RLHF scheduling, newsletter posts).
Routed: [topics/reference/schedules-and-annealing-literature.md](topics/reference/schedules-and-annealing-literature.md)
(third entry; keeps the data-rewriting cluster, 2508.01483 as an ID to check, mixture-level
cooldown, Branch-and-Merge; drift and collisions listed); 25 IDs added to the citation
ledger, most tagged drift. No project-doc change.

### Post-intake decisions — 2026-08-22 (LBM check + annealing answers)

> hmm, I think it would be nice to pull it out so I don't forget the potential project
> option is there, where would you say we should put it with that goal in mind?

> then, yes for 1 and 2

Applied: new [topics/staging/rewritten-anneal-slice.md](topics/staging/rewritten-anneal-slice.md)
(README row; WSD §4 pointer as a variant of WSD-opt-4); Language Bottleneck Models named in
TLC §1's positioning paragraph with her read; a maintained working list (keepers by role,
known drift and term collisions) at the top of
[topics/reference/schedules-and-annealing-literature.md](topics/reference/schedules-and-annealing-literature.md).

### Undated (intake 2026-08-22) — estimating per-(model, docstring) performance; bootstrap; intervals; conformal prediction

> Ok, next topic. So even though this is in the context of a specific task, I think this
> whole line of questioning is super relevant to many of the projects that we've discussed.
> especially conformal prediction is something I want to make sure to flag as a potential
> tool/analysis method.

> ### Task
> I'm using LLMs to write functions based on a docstring and method signature.
> The model performance is defined as the percentage of test cases that its output passes
> on a predefined test set.
> I want to estimate a given model's performance at writing code for a single docstring.
> **Challenge**: Even with temperature 0 many models give diverse results which means I
> need multiple samples to estimate the expected performance.
> How can I improve the accuracy of my estimates at a fixed number of samples from a given
> (model, docstring) pair?

> What about bootstrapping?

> Ok, and what are the comparable other non bootstrap methods of estimating uncertainty.
> How do they compare to bootstrapping?

> How does conformal prediction relate to the problem and methods we've discussed so far?

> Ok, so I'm interested in how I could use conformal prediction along with cheap estimators
> given multiple docstrings and potentially multiple models.
> One thing I imagine is I could give a docstring to a model and ask whether it provides
> enough info to exactly reconstruct the original function, or to pass strict test sets or
> something like this
> How could something like that work to either improve my ability to predict model success,
> improve my estimates of model perf or improve a prompt use to have the model generate the
> code from the docstring?

Routed: new methods toolkit
[topics/reference/estimation-and-calibration-methods.md](topics/reference/estimation-and-calibration-methods.md)
(full condensed record: estimator discipline, interval-method table, conformal vs.
bootstrap targets, the conformal + cheap-estimator design, relevance map); TLC §4 note
(census measurement design, conformalized shrinkage, calibrate-after-selection); ELI §4
pointer; program-level note in the potential-projs README carrying her flag verbatim;
three IDs to the ledger.

### Undated (intake 2026-08-22) — embedding quality as a predictor of IRT task difficulty, training-free

> Let's say that I have a data set of tasks, and I have the evaluation results of a bunch of
> different methods on those tasks, and I have a measure of performance. Like I'm gonna take
> those evaluation results, and I'm gonna get some measure of difficulty per task, using the
> different methods to give some type of general predictive difficulty using something like
> IR2. Then I want to try a few different methods of embedding the tasks such that I can
> predict the difficulty, where, I guess, I specifically am interested in the right
> embedding approaches, and so that's why I'm not necessarily trying to just, like, train a
> predictor, but I guess I'm more curious about, like, correlation type metrics that don't
> require training. So if I produce a bunch of embeddings that I then use to cluster, then
> how can I take the cluster labels and the actual difficulty labels and produce a measure
> of embedding quality in terms of predicting difficulty?

(Spoken; "IR2" = IRT.) Routed: second entry in
[topics/reference/estimation-and-calibration-methods.md](topics/reference/estimation-and-calibration-methods.md)
(cluster R²/η², NMI/V-measure/ARI, kNN smoothness vs. shuffled baseline, pairwise distance
Spearman; two intake caveats on the null); IRT §4 note framing it as a generalization of
IRT-7 — candidate direction, no ID yet. No ledger rows (only a Columbia IRT web page was
cited).

### Undated (intake 2026-08-22) — first-pass difficulty estimation from 4 models × 8 prompts

> I want to evaluate the task samples in a data set to get a measure of difficulty. In order
> to do this I have taken the data samples and evaluated the performance of four different
> language models with eight different prompts each on the samples. It seems like this
> should be enough to do at least a first stab at IRT or some form of difficulty estimates
> on this data. How could I go about doing this?

Routed: third entry in
[topics/reference/estimation-and-calibration-methods.md](topics/reference/estimation-and-calibration-methods.md)
(staged recipe: smoothed pass rate → Rasch → many-facet mixed model → 2PL later; per-item
diagnostic split; leave-one-facet-out validation); IRT §4 note on what transfers to the
DataDecide fit (facets, the diagnostic table for IRT-8, validation for IRT-3); TLC §4
pointer since the 4 × 8 shape matches the census. Dataset not named.

> Okay, now say that the samples in the dataset are code generation instructions, and so for
> each generation, then I can evaluate an entire test set on that generation. And so I could
> either aggregate the test performance into a binary pass-fail, or I could produce an
> aggregate, like average pass-fail, or I could treat each sample test case as its own thing.
> How would that change the approach, and how should I reason about which of these to do?

(Continuation of the difficulty-estimation conversation.) Routed: continuation of the third
entry in [topics/reference/estimation-and-calibration-methods.md](topics/reference/estimation-and-calibration-methods.md)
(three response definitions as three constructs; full-pass primary / fractional secondary /
test-level diagnostic with a generation random effect; test-suite-density caveat;
per-instruction report); TLC §4 note recording the tension with TLC's fractional-score choice
and the near-miss score as an optimizer signal.

> Okay, so then, now, how does it change things if I'm actually using the evaluations as
> signal for an automated prompt optimization approach that uses something like DSPy and an
> algorithm like GEPA, where I want to optimize the prompt at two levels, or maybe even just
> at one level initially, but eventually at two levels. And so the kind of pair test feedback
> then might actually be very helpful. And so measuring the difficulty could allow me to
> cluster things to select the most useful or characterize, to select or characterize the
> most useful tasks and test cases to run, for example. How does that change the design and
> what would you recommend then?

> So you talk about grouping by requirement, but I don't actually have a way to group the
> tests by requirement, do I?

(Continuation.) Routed: TLC §4 note (utility over difficulty; three pools; score + feedback
metric; tiered tests; IRT as a curriculum tool; one level then two; empirical test
clustering by pass/fail vector when no requirement labels exist); GEPA entry in
[topics/reference/prompt-optimization-landscape.md](topics/reference/prompt-optimization-landscape.md)
(first missing anchor with an ID; plan gate 3 annotated); second continuation in the
estimation topic; ledger row.

### Undated (~mid-2026; intake 2026-08-22) — structured output as a separate skill; tiny structurer models

> I'm interested in using really small LLMs as tiny focused components of larger systems.
> And one core question in that type of system design is the question of structured output.
> Intuitively I suspect that the skill of adhering to a very specific output format is a
> different skill than solving many specific tasks and that it might be unnecessarily
> limiting to require that the same LLM does both task solving and output structuring.
> Then, I'm curious about the research findings around (1) how enforcing structured output
> impacts task performance especially for older/smaller models and (2) how small/optimized
> can we make a model whose sole goal is to structure information extracted from an
> unstructured text blob (assuming that the info is in the text blob in an easily
> extractable way, no reasoning required). What are the approaches to (2)?
>
> I'm looking for research in 2025 and 2026, with a strong preference for more recent papers
> since the landscape of LLM capabilities is evolving so quickly.

Routed: new [topics/reference/structured-output-literature.md](topics/reference/structured-output-literature.md)
(full condensed record; README row under the TLC/ELI group); ELI §4 note (her premise
restated; base-vs-instruction-tuned prediction for ELI-2; SLOT as a tiny-model wrapper
class); IRT §4 pointer for IRT-10's expected direction; thirteen IDs to the ledger. The
tiny-components systems interest has no project doc — left in the reference topic.

### Undated (intake 2026-08-22) — the OLMES metric columns in DataDecide (eight turns)

> I'm using the DataDecide dataset which evaluates on the OLMES evaluation tasks. There are
> a bunch of different metrics reported and I'm not sure how to interpret them. Can you
> provide a breakdown (cited) of what each of these metrics might be and different ways
> that people might calculate them (if there isn't one standard)? [column list]

> So would it be accurate to describe: predicted_index_raw / predicted_index_uncond /
> predicted_index_per_byte / per_char / per_token / correct_choice as being primarily values
> that were likely used on a per-question level to generate the more interesting metrics
> (the probs, logits, bits per byte, accuracy, etc)? Additionally, of all the metrics, which
> are possible to compute directly from other metrics provided?

> If we have the sum_logits_corr, can we exponentiate to get the prob_corr?

> so norm_correct_prob = correct_prob / total_prob ?

> Does it seem like the uncond_correct_prob_* and uncond_total_prob are building blocks (at
> the per-question level) to compute other metrics or are they likely actual metrics we're
> trying to extract?

> so is acc_uncond the only real metric computed with the uncond_* building blocks? why is
> there a _per_char and _per_token if there is only acc_uncond?

> Would it be possible to consider uncond_correct_prob as an additional possible proxy
> metric?

> Is it possible that "correct_prob" would have a different performance as a ranking metric
> than "sum_logits_correct"?

> ok, and then "uncond_total_prob" is again just the aggregate of a value that was only
> useful during per-question computations of other values, right? Can you also explain
> bits_per_byte_correct a bit more? How is it calculated, what does it mean intuitively,
> how should I think about it?

Routed: [topics/reference/datadecide-data-pipeline.md](topics/reference/datadecide-data-pipeline.md)
(new entry: what was settled — `correct_prob = exp(sum_logits_corr)` checked on data; what
is a guess — uncond as a difference, `correct_choice` binary vs. index, per-char as ÷ length,
bpb conventions; repo facts from `metrics.py` and `configs/olmes.toml`); DCARD §4 —
**DCARD-1(e) metric definitions pinned to the evaluation code**; TINY §4 — exp-related
proxies counted once, `uncond_correct_prob` as a candidate once defined; two ledger rows.

### Undated (intake 2026-08-22) — compute units and storing them (four turns)

> when discussing compute used to train LLMs you get really huge numbers, even looking at
> the values in trillions of flops is a huge number, and I've never heard someone say
> "trillion flop". What units do people actually use to discuss these values?

> If I want to store these values in a DB, then one thing I could do is just record
> pfs-days, but alternatively I could store the two pieces of scientific notation
> separately, what are they called, and how do people handle this in practice?

> yeah, storing it directly has led to pyarrow conversion issues due to values being too
> large hence my concern

> If I want the smallest type options that cna store (significand, exponent) in duckdb when
> the precision isn't really important (its a very loose approximation anyways) what are
> the best types for each part?

Routed: [topics/reference/datadecide-data-pipeline.md](topics/reference/datadecide-data-pipeline.md)
(units: pfs-days and FLOP in scientific notation; significand/exponent; DuckDB
`REAL`+`TINYINT`). Correction recorded there: the response's "float64 overflow at 10²⁵"
diagnosis is wrong (float64 reaches 10³⁰⁸; 2⁵³ bounds exact integers) — the likely cause is
an int64 overflow from an integer FLOP product before the Arrow cast, which the repo's
`model_utils.py` integer arithmetic would produce; fix by casting to float, not by
splitting columns. Unverified against the traceback. No project-doc change beyond a
data-card convention line.

### Undated (intake 2026-08-22) — lossless compression baselines on HumanEval samples (three turns)

> I want to test out standard lossless compression algorithms on fairly short code samples,
> like the HumanEval ground truth examples, one by one.
> I want to see what approach works best and then I'll try this on datasets with much
> longer samples.
> If I want to see the full range of fairly strong to really strong options on the code
> what algorithms or combinations of algorithms should I consider? I'm up for implementing
> the sequencing of algs in basically any language if there is a language that provides a
> particularly nice impl of something useful.

> so, I only care about (1) and (2) because I'm explicitly trying to plot "compression
> versus correctness" on a per sample level (where correctness is the result of doing other
> transforms before running unit tests). based on that statement, how would yo uupdate your
> recs?

> great, now, please focus on all methods, even quite slow ones, we don't have to worry
> about scaling, and consider how well methods perform on python not just on general text.
> and consider stacking multiple methods or preprocessing to really try to push the limit
> in my specific setting.
> then, give me an overview of my options!

Routed: TLC §4 note (two regimes; compressed-bytes x-axis; prior fairness and selector
cost; source-lossless vs. test-preserving; transform stacks; the six-layer suite replacing
"zstd and friends"; headline stated against the strongest fixed-prior baseline); tool
inventory in [topics/reference/code-compression-literature.md](topics/reference/code-compression-literature.md).
No new citations beyond Delétang 2309.10668 (already on file).

> amazing! so now, practically, how can I implement these different compression approaches?

(Fourth turn of the lossless-baseline conversation, passed separately.) Routed: TLC §4
addendum (harness shape: `code_to_test` vs. `payload`, result-row schema, per-representation
dictionaries, fixed filenames for external tools, oracle labels, implementation order);
pointer in the code-compression reference; the agent-written skeleton archived with an
`INDEX.md` at `~/drotherm/data/convo-artifacts/2026/2026-08-22-lossless-baseline-harness/`.

### Undated (intake 2026-08-22) — compression taxonomy → code correctness ladder → Pro-mode related-work search (four turns)

> Please give me an in depth overview of general lossless and lossy compression algorithms
> along with a clustering of the different types and purposes and assumptions they
> leverage. Any that are specially targeted at a specific application
> (vision/text/code/audio) should be included

> ok, so, lets say that I want to understand the available compression versus correctness
> tradeoffs for compressing code (eg python). I know that lossless compression should be
> defacto 100% correctness so then the question is just how far you can push the
> compression with a lossless algorithm. and then, I would assume lossy approaches that
> aren't specifically designed for code would have terrible performance if we were
> measuring something like test case pass rate?
> And I'm guessing that if I'm fine with any amount of time/compute spent on compression
> then I can get a much better compression ratio than if I want a fast lossless algorithm?

> Ok, please describe the categories of papers or available libraries you would look for to
> address this question:
> I need academic papers to cite related to methods which take in an initial code sample,
> produce a representation, and then generate code that either (1) exactly program flow of
> the original code, ignoring any variable/method naming, docstrings, etc or more
> permissively (2) will pass the same test cases as the original code (functional
> equivalence).
> I'm interested in the strongest lossless compressing method (even the very slow ones) and
> in any standard code compression libraries for something like python. but I'm also very
> interested in either lossless or lossy code compression approaches that have shown up in
> academic papers, either using machine learning or otherwise.
> I've done extensive searching and have mainly found papers around learned embeddings (so
> no reconstruction performance) or prompt compression (so no code-> prompt part) but I
> have found very little about code compression specifically that has a reconstruction
> component and a measure of the compressed intermediate representation.
> so describe to me how you would perform this search in Pro mode (before I kick you off in
> pro mode to do so)

> Amazing, please execute that plan, focusing ruthlessly around your identified question:
> "Does this method actually produce a smaller representation from code and reconstruct
> code/behavior with a measured correctness criterion?"
> However, there is no need to run a small empirical baseline suite.

Routed: [topics/reference/code-compression-literature.md](topics/reference/code-compression-literature.md)
(taxonomy summary; the five-level correctness ladder; the search plan's inclusion rules,
ten categories, evidence-table schema, per-level leaderboards; the results map — classical
syntax-directed / AST / bytecode compression, Boffa 2025, program reduction,
superoptimization, library learning, background-only clusters, Python tooling; the gap
statement); TLC §4 note (ladder, lossless competitors on the rate axis, TLC as a reducer
with an NL representation); litreview plan subdomain C seeds; eight ledger rows.

### Undated (intake 2026-08-22) — information bottleneck → proxies for the TLC objective (three turns)

> I'm thinking through the goal of representation learning from the perspective of an
> information bottleneck. And there I know one formulation is that you minimize the mutual
> information between the input and the representation while maximizing something else,
> which represents maximizing the information in the representation relevant to the target
> task, but I don't remember the actual equation in terms of information theory. What is it?

> And the theory is that if you learn a representation that optimizes this it will be the
> most useful for solving the task generally because you have removed all distractor info
> right?

> Ok great, so now, I'm interested in using an encoder-decoder setup to encode a Python
> program into a representation and then to decode the representation into a function that
> passes the same tests as the original function, aiming for a proxy for behavioral
> equivalence.
> I'm looking for a proxy for mutual information if the encoder and decoder are frozen
> pretrained language models and the intermediate representation is natural language. I'm
> not even sure how I'd measure the "mutual info" between the input function and the output
> function, let alone how I'd do that between language and function

Routed: TLC §4 note (IB objective; relevance variable = test-suite signature, not source;
proxies — description length −log p_LM(r) for I(X;R), smoothed K-sample pass probability
for I(R;Y_T), leakage incl. log p_D(x|r), InfoNCE lower bound; Behavioral Bottleneck
Score; two cautions on K and on which LM prices the rate); three ledger rows.

> Interesting, so the reason I'm interested in thinking in this direction is because in
> practice models often take a function and just strip out white space and flatten etc to
> shorten so while length of representation can be a proxy, there is a real concept of no
> behavioral information being lost even if the representation can't literally be run
> (because I guess our encoder already knows "programming" so that part doesn't have to be
> preserved in the representation). So I'm looking for alternative measures that get at the
> "extractable info" overlap between the x from X and z from Z that could be useful from an
> analysis perspective even if not directly optimizable. It could use a separate LM and
> prompting setup (or set of them) to aim to measure some type of baseline, etc?

(Continuation of the IB conversation.) Routed: TLC §4 note (abstraction-vs-minification
confound; predictive 𝒱-information; behavioral retention ratio; information profile with
the leakage target I_𝒱(Z→S | B); behavioral bottleneck index; four extractor
instantiations; the ∅/signature/Z/minified/X/oracle matrix); ELI §4 pointer (same
machinery, roles swapped); two ledger rows.

> Great, this is much more inline with what I was thinking. I also really like the
> contrastive directions in general. Please now do a deep dive into related work that looks
> at both trying to measure relatedness or shared info content of language vs code
> representations, on methods that try to estimate that relatedness for code or language or
> embeddings for a given task or model, and general analysis or bottleneck approaches that
> might be relevant to either the analysis or optimization of the bottleneck setup that I've
> described. Present the related work and then highlight the 5 most plausible/interesting
> directions that you'd suggest I start looking into.

(Continuation.) Routed: measurement entry in
[topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md)
(seven clusters; the five directions; intake notes); TLC §4 note (Decodable IB and
ContraCode as the two naming citations; where each direction lands; CodeNet as the source
of behaviour-equivalent positives); litreview plan subdomain F; twenty ledger rows.

### Post-intake decisions — 2026-08-22 (evening batch: estimation, difficulty, structured output, metrics, compression, IB)

> (1) I suspect that 1-3 and 4 actually describe different research programs for TLC, is
> that accurate?

> yeah, lets do the 3 layers. the thing is, I was most interested in the representation
> project […] so I started with representation but […] switched and am focusing on
> compression to satisfy my phd reqs. hence we need to keep both as clear separate projects

Applied: TLC restructured as three layers — TLC-0 measurement suite (𝒱-information
retention/leakage, condition matrix, contrastive game, controls) as the shared instrument;
the compression project (TLC-1–3) and the representation program (TLC-opt-*) as **separate
projects** sharing the harness and TLC-0; §1 objective restated in IB/𝒱 form with the
layered baseline suite; the sequencing paragraph rewritten (compression first for harness
and crispness, no longer because its metric is the representation instrument); TLC-2 gets
the DSPy/GEPA incumbent; standing instrumentation updated. Header records that Danielle's
primary interest is the representation program and the compression project is the thesis
track. Items 2–4 of the walkthrough (int64 check; calibrate-after-selection principle;
IRT-11 ID) still open.

> for 2, lets leave that as a flag in the doc, its not immediate an issue for me now so
> lets leave it to be checked when I pick up the project again.
> 3. yeah, lets add the paragraph
> 4. yes, definitely lets add an id, its a real thing and lets pull it out now.

Applied: int64 diagnosis marked "flag, not a task" in the pipeline reference;
calibrate-after-selection added to
[topics/reference/project-approach-principles.md](topics/reference/project-approach-principles.md)
as a standing principle; **IRT-11** (which item representation explains difficulty) added to
[potential-projs/irt-reanalysis.md](potential-projs/irt-reanalysis.md) with an impact row.

### Qualitative comparison tooling for pre-computed token probabilities (three turns)

> I'm a machine learning researcher who works on LLMs and I currently have a collection of
> models trained in different data recipes that have been evaluated in some standard QA
> tasks. I'm looking for good tools to speed up the process of (1) learning anything I can
> from the checkpoint itself (2) learning what I can about the data recipe used for
> training and (3) visualizing the question and predictions, individually and
> comparatively, looking at the actual output probabilities which I have per token/char,
> etc.
>
> Do accessible ecosystems of tools for these visualization purposes exist for a Python
> use case?

> Ok! Most of those tools seem to specialize in evaluation and visualization using the
> model itself along with the data. What if in general I only have the probabilities for
> a few tokens for each model-example pair and I want a way to visualize the examples and
> model responses in a way that focuses on being able to qualitatively compare models and
> learn from individual examples between models.  This might be more in the visualization
> space than the ML space.

> What if I'm committed to doing the analysis in a marimo notebook? Then what tooling is
> available?

Routed to [topics/reference/experiment-tooling.md](topics/reference/experiment-tooling.md)
(new entry; decision of record marimo + Altair) with a consumer pointer in
[topics/reference/datadecide-data-pipeline.md](topics/reference/datadecide-data-pipeline.md)
under the OLMES metric entry. No project ID — infrastructure for DCARD/IRT/TINY per-item
work.

### The design space between dense ensembles and sparse MoE (three turns)

> So on the one hand, there are mixture of experts models, which allow for really, really
> large numbers of combinations of paths through a model as compared to dense. On the
> other hand, you have the idea of ensembling the output of multiple completely different
> dense models as a way to try to combine information. And I'm assuming that there are a
> bunch of things that fall in between these two approaches of having a token go through
> all of a few different dense models versus being routed to a subset of the feedforward
> layers within all the other parts of a model. And so what are the things that lie in
> between these two settings?

> So for each of those you listed, can you describe where they belong on the 3 axes you
> described?

> great! now, can you give me a 10-15 paper reading list that covers at least one
> representative work from each of the cells in this space (since say 2022 at the
> earliest)?  Give me the paper title, first and last author, and then link to arxiv pdf

Routed to [topics/reference/moe-literature.md](topics/reference/moe-literature.md) (new
entry: three-axis frame, placement table, 13-paper list, cell-coverage gaps noted) with
20 agent-supplied + 1 Claude-added ledger rows. No project ID; vocabulary for
MOVE/PART/MSUITE and the pending "tiny LLMs as components" thread.

### HumanEval and MBPP prompting conventions (two turns)

> I'm trying to understand how people normally prompt a model to evaluate the human eval
> task.  from the original paper, "Evaluating Large Language Models Trained on Code" it
> seems that they just literally pass in the stub but I suspect that at this point there's
> some form of instruction somewhere in the prompt (system or otherwise)?

> Great, can you find me the same information for MBPP evaluation standard prompt
> practices?

Routed to [topics/reference/code-benchmarks-landscape.md](topics/reference/code-benchmarks-landscape.md)
(new entry) with a §4 pointer in
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
(prompt condition as part of the TLC-0 harness contract). No new ledger rows (HumanEval
2107.03374 and MBPP 2108.07732 already present; other sources are GitHub pages).

### HumanEval's derivative ecosystem; cross-benchmark overlap and dedup (two turns)

> HumanEval is such a commonly used coding dataset for language models that I'm sure
> there are many additional works that build on top of it in some way. I think an obvious
> way would be HumanEval++, which builds on top of it to augment the test sets to
> demonstrate that it could have more use if you do that. But it seems likely that there
> are things that, for example, extract the comments themselves, or just the stub, and see
> how well models can do at predicting. Or places where people use the code snippets to
> test things like docstring creation, etc. Or, for example, using it to generate
> additional data by doing artificial methods of augmenting either with the language model
> or using programmatic methods to manipulate code. I can think of a ton of different ways
> that you could build on something like HumanEval. And since it is so widely adopted, my
> thought is that probably some of this has been done. Please explore what has been done
> and give me an overview of the things, the different types of tasks, benchmarks,
> datasets, extensions that have been done on HumanEval. And if there are other datasets
> that are similar in that they are code datasets that can be used for generation with
> test sets or things like this, but that have been extensively manipulated and built upon
> in the way that I'm describing, then I'm interested in those as well.

> Very interesting. Are there attempts anywhere to catalog or deduplicate across major
> sets of coding datasets and or benchmarks? It seems like a substantial portion of the
> data samples are very very similar functions or compositions of similar functions and
> I'm curious about how this has been studied and what the conclusions have been/what type
> of impact this can have or even how you could go about categorizing/clustering etc on
> this info.

Routed to [topics/reference/code-benchmarks-landscape.md](topics/reference/code-benchmarks-landscape.md)
(new entry) with a §4 prior-art pointer in
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
and 21 ledger rows. Flagged: ShortenDoc belongs in TLC litreview gate 1; the
within-benchmark redundancy question Danielle actually asked is not answered by any cited
study.

### u-µP paper summary and Figure 1 (two turns)

> Can you describe the key takeaways from the attached paper? (also available online at
> https://arxiv.org/html/2407.17465v3

> Please describe the contents and importance of the figure that I've provided that is
> from the paper.

Routed to the new
[topics/reference/parametrization-and-hp-transfer.md](topics/reference/parametrization-and-hp-transfer.md)
with a design-option pointer in
[topics/staging/datadecide-dense.md](topics/staging/datadecide-dense.md) and three ledger
rows (one Danielle-supplied, two Claude-added). No project ID.

### Ranking metrics when decision accuracy saturates (one turn)

> I am predicting the ranking of a list of things (~25 items) and one metric that is
> relevant is "decision accuracy" which would be the average pairwise prediction accuracy.
> however, even bad baselines do great on this metric. so then I'd like to instead use a
> metric that captures "correct rank predicted" or somehting like this to capture that
> swapping element 1 and 2 is substantial even if 1 and 2 are both ranked higher than all
> the rest correctly so there's a high decision accruacy.
>
> My loose memory is that NDCG or something like that is a ranking metric that captures
> osmething like this, but it weights the values at the top of the list heavier or
> somethign like this? Waht woud be good metrics for me to consider

Routed to [topics/reference/estimation-and-calibration-methods.md](topics/reference/estimation-and-calibration-methods.md)
(new entry; the response's recsys menu re-read for a full-permutation target) with a §4
note in [potential-projs/early-dynamics-prediction.md](potential-projs/early-dynamics-prediction.md)
extending the 2025-07 metric suite. Two Claude-added ledger rows.


## Prompt-optimization re-evaluation conversation (ChatGPT) — intake from 2026-08-23

Danielle's framing at handoff: a very long conversation, "a real 'aha' moment for me
about the project I was working on"; delivered in chunks. Full verbatim transcript
(both sides) preserved at
`~/drotherm/data/convo-artifacts/2026/2026-08-23-prompt-opt-reeval-aha/transcript.md`;
source: https://chatgpt.com/c/6988915f-707c-8326-aa49-8b0aa7a18537

### Chunk 1 (four prompts)

> There are a bunch of papers from the last few years that explore different methods of
> prompt optimization for different tasks and with different models.  However, current
> cheap LLMs are often more powerful that the best LLM of 2 years ago.  Then, I'm curious
> which of the findings actually hold up in the modern era. I vaguely remember reading
> about ML conferences that have repro or blog post tracks, which this feels like the
> perfect kind of systematic investigation for (as it wouldn't provide a new approach or
> conclusion necessarily, it would be purely analysis which I'm unclear can really be
> submitted as a research paper even to a workshop).
>
> What are these options? And, have similar studies been done?  I'm specifically
> referring to things like the OPRO method with LLMs as optimizers, the later
> "revisiting" paper (Revisiting OPRO: The Limitations of Small-Scale LLMs as Optimizers
> ), Large Language Model-Based Evolutionary Optimizer: Reasoning with elitism, etc.

> "Prompt Optimization Re-eval Harness" - what would this even look like? as in if I
> just released the open source library that was well designed and usable for re-evaling
> prompt optimization approaches?

> so like, when you consider a phd students "success" in an ml phd do benchmark papers
> contribute or are they seen as "just engineering"? I feel like an analysis paper would
> be a (small) positive mark in the research bucket but im less sure about benchmarks?

> I'm technically mid but am really just hitting my stride in terms of defining a clear
> research trajectory, identifying my own directions and making them testable, writing
> the papers independently, etc.

Routed to [topics/reference/prompt-optimization-landscape.md](topics/reference/prompt-optimization-landscape.md)
(new dated entry, marked in progress) and six agent-supplied ledger rows (OPRO,
Revisiting OPRO, LEO, two APO surveys, PromptBridge — OPRO's missing-anchor slot now
filled). Project-level routing (staging vs. project doc, and which existing project the
"aha" lands on) deferred until the conversation completes.

### Chunk 2 (one prompt, with pitch PDF + Table 5 screenshot attached)

> (see attached doc and screenshot) "so, I just pitched my advisor on the pdf contents
> as a general thesis direction to move in (starting with a well scoped workshop paper
> which is a small subset of what's in that doc).  but one of the things that made me
> curious about this direction (beyond using OPRO+ as related work and wondering how
> accurate the conclusions are currently, when optimizing prompts, tasks, etc.  also,
> has posttraining in some cases robbed llms of previously existing general
> optimization skills because they've been partially collapsed to game metrics/fit
> benchmarks?) is the table that I screenshotted.  This is just the selection of models
> on open router right now that seem likely to be solid at coding, are very
> inexpensive, and are the most recent version released of their line.  But the number
> of wildly low price models on open router, including historical sequences of models,
> just seems like this amazing opportunity not to evaluate the models on your standard
> terminalbench one shot monstrosities, but actually to evaluate them on fairly well
> scoped tasks that we'd expect them to succeed at and then measure the variance within
> model and across models + extend to understanding how these "super cheap" modern
> options compare to the 1-3 affordable-SOTAish options that were generally published
> on in for previous methods.
>
> Does that make sense?"

The attached PDF is the pitch "LLM-as-Optimizer of Natural Language Bottleneck Model"
(Rothermel*, Li*, Cho) — the conversation's project is TLC. Routed to
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (new 2026-08-23 entry: pitch milestone, her three motivating curiosities including
the posttraining-collapse hypothesis, condensed response) with a chunk-2 note in
[topics/reference/prompt-optimization-landscape.md](topics/reference/prompt-optimization-landscape.md).
Pitch PDF preserved in the convo-artifacts bundle. Two Danielle-supplied ledger rows:
the pitch bibliography's author/title pairings for 2402.18700 (Chuang) and 2408.11198
(Taherkhani) conflict with earlier agent-supplied attributions (Zhou; Saluja).

### Timeline correction (2026-08-23, during intake)

> yeah, to be clear, this was around the Jan-Mar 2026 timeline so the time estimates
> etc are currently irrelevant, a paper was eventually submitted but rejected (which
> was fair, I wrote it very very very fast). [...] So, I'm only just getting back to
> thinking about the original version.

(One clause of adviser-related context elided per standing convention.) Corrections
applied to the TLC §4 entry and the prompt-optimization-landscape entry: conversation
is historical (~Jan–Mar 2026), the workshop paper was submitted and rejected, and the
current TLC docs are the return to the original direction.

### Chunk 3 (two prompts)

> this is very cool! one of the things you said that I'm curious about is the idea of a
> "more open ended coding task", what types of examples were you considering?
>
> Also, for the workshop we previously intended to submit to I ran a large number of
> runs to get performance and variance across models and I don't believe this type of
> analysis is going to go into the paper, making me wonder where else it could go.
>
> Then, I'm also wondering about the idea I've heard frequently from researchers that
> all the model provider's models have collapsed to almost identical solutions.  My
> results on simple fxn generation seems to suggest almost the opposite and makes me
> wonder whether for the specialized task of coding older models were actually *more*
> similar because their gains came mainly from general LLM improvements but now we're
> making great progess in the coding realm because of the ability to verify outputs
> for rewards or dataset generation, etc so different model families may actually have
> gotten *further* from each other in the coding task space over time.  I'm not quite
> sure how I'd design an experiment to test whether this is the case or not.
>
> And finally, I've built a pretty robust library for generating testable python
> functions.  And for our workshop paper I need to extend it to other langauges too.
> Then it makes me wonder what other types of ways I can leverage it either for easily
> generating specific useful data for specific ideas or as the base of a benchmark,
> etc.  Thoughts?

> 1) oooh, these more open-ended tasks would be really cool to add to my synthetic
> dataset library!!
>
> 2) variance experiments - do you think there's a way to try to frame the conclusions
> into its own focused workshop submission for a very different type of workshop at
> perhaps a more analysis or reliability kind of conference?
>
> 3) I love the idea of "distance between model behaviors",  each of the things you
> listed seems both informative and totally doable.  Do you think there are 1-3
> workshop sized hypothesis + exp designs in there where quick initial tests could
> narrow down which ones would be most pluasible?
>
> 4) "A. Cross-language semantic equivalence benchmark"!!!!! I hadn't even considered
> this as a direct result of using the same specs for different languages, thats such
> a cool concept!  Do you know if people have used the old Unsupervised MT techniques
> (handwavy description: aligning the tokens based on statistics to infer translations
> between language) on code translation?  I'm sure there are a ton of semi-supervised
> approaches but I see how this library would provide the ability to generate
> interesting supervised data.
>
> "C. Curriculum / controllable difficulty scaling" whoa, this would even be plausible
> as an approach to making the LLM-as-optimizer approach stronger by using curriculum
> learning.  which I'm sure exists in directly comparable related work but I haven't
> seen it immediately in my cursory pass so I suspect it isn't a standard component of
> all approaches?
>
> "F. Data generation for post-training / distillation experiments" -> this aligns
> with my interest of how far we can push small/weak models, I think that would be
> super cool.

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (chunk-3 entry: open-ended task templates, curriculum-for-optimizer candidate,
cross-language leverage, post-training data generation, the orphaned variance sweep and
its possible homes) and
[topics/reference/evaluation-methodology-literature.md](topics/reference/evaluation-methodology-literature.md)
(new entry: the model-divergence hypothesis verbatim + behavioral-distance designs
A–E and workshop slices H1–H3). Three ledger rows (TransCoder, unit-test-filtered code
translation, PMLR cost-aware prompt opt with no ID). Divergence-hypothesis
staging/promotion deferred to the post-conversation walkthrough.

### Current-state update during intake (2026-08-23, not part of the historical conversation)

> ahhh, something I learned after this convo: designing the families and their
> difficulty levels was very non-trivial and a variety of issues came up such as
> ending up with "different" functions by sampling different settings that ended up
> being identical, it being very difficult to maintain, some functions basically
> simplifying to identity, and the difficulty measures being very uneven across and
> within families.  so the synthetic functions would need some work to use, and a core
> reviewer pushback was why use synthetic functions at all, why not use standard
> datasets.  which my advisor firmly agreed with so we're currently using
> HumanEvalPlus, which I've discovered is much worse than my synthetic functions in
> that it has a dramatic variance of difficulty, some samples have been solved by
> every model I've tried on them, down to the cheapest, some are broken in the
> official release, they are SUPER short so compression is kinda meaningless, etc.
> So, I'm interested in combining synthetic and real world, reporting on humaneval
> because people assume it must be good, but also on better real world datasets, etc.

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (dataset-strategy entry: synthetic-library defect classes, the HumanEvalPlus
cutover, her first-hand critique, the combine-synthetic-and-real interest) and
[topics/reference/code-benchmarks-landscape.md](topics/reference/code-benchmarks-landscape.md)
(first-hand HumanEvalPlus defects note).

### Chunk 4 (one prompt)

> So as someone who writes a ton of research code, I've also been playing around with
> the idea of data manipulation problems where in the most general case the model can
> use any library from a whitelisted set (eg pandas, numpy, polars, no library, etc)
> and needs to accomplish a certain type of transformation on data which could be
> defined as filtering -> transforming columns -> aggregating -> writing, or could
> even involve creating a plot with a specific plotting library that was judged by
> pixel distance versus the plot that matched the spec.  all of those ideas have a ton
> of different ways you could implement them but you could describe the target
> succinctly and make it easily testable.
>
> I think you could even do things with say scikit learn vs scipy vs pytorch etc where
> there was some deterministic target (potentially made harder because it wasn't
> specified how to make it deterministic, but it would need to be possible) that
> involved using a functionality or even building a tiny model according to a spec,
> training it for a fixed number of steps and then evaluating it on some data and
> reporting the scores.  as long it was all tiny and stayed on the cpu I think this
> would be super doable (though it would be tricky to find settings where the same
> evaluation was identical across libraries for some classes of problems).
>
> Do you think this is a plausible direction?

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (chunk-4 entry: data-manipulation + tiny-ML task families, DSL design, canonical
evaluation, determinism strategy, roadmap). No new identifiers.

### Chunk 5 (three prompts)

> I have a docker sandboxed agent running built with pydantic ai, and as a user I'd
> REALLLY love to fit my coding agents to be pros at pydantic (which I realize is not
> in any of the categories we just discussed, but falls into another category of data
> structs and data classes). and then pandas, altair, matplotlib, torch.  And yes, I
> use uv and pytest but for now my synthetic tests are just input/output pairs.
>
> Separately, do you think there would be a version of this that could measure a
> different type of automatically verifiable correctness?  I'm thinking things like
> tasks with clear constraints that make one of the options clearly optimal given your
> constraint, but where an agent might not choose that option even if they technically
> solve the task.  So like, lets say you tell an agent that you care alot about the
> performance of your code and you need to write a function to sort a huge list of
> floats.  Ask it to choose between three implementation sketches that you saw online
> and then complete the implementation to provide you the function you need.  Then the
> sketches clearly output bubblesort, mergesort, radix sort or something like that.
> there's a clear right answer, if you instructed the agent to "choose one of the
> sketches" then a correct result should implement one of the algorithms, which you
> can verify.  You can verify the correctness of the implementation.  And you can
> verify that the agent chose the optimal one, probably directly via ast, but
> definitely via timing the sorting of a huge list.  I feel like this sounds like the
> kind of question that would be super annoying to generate, but actually, if you work
> backwards from "true statements" for software engineers, that are conveniently
> collected in books like "cracking the coding interview" in addition to a range of
> websites, you could make a small set of "clearly ranked choice options" and then
> generate a set of templated "task scenarios" and verification methods.  Thoughts?
> Does this already exist?

> Interesting, but it kind of sounds like it would be a pretty hard pitch for like,
> something you would expand to a full conference paper, for example, because there
> are so many things that do a very, very similar thing.

> Cool. I guess I'm interested in one, or, wait, oh, I missed this part of your
> answer, which is decision quality under constraints as a probe for model divergence
> and reliability. I think this would be very cool, but it also seems like it might
> seem very cool, and yet there probably are a bunch of types of outcomes that would
> make it very difficult to draw any conclusions, making it high risk.

Routed to [topics/reference/evaluation-methodology-literature.md](topics/reference/evaluation-methodology-literature.md)
(decision-quality-as-divergence-probe entry with A/B/C design, de-risking, and a
Claude-added option-position-bias caveat),
[topics/reference/code-benchmarks-landscape.md](topics/reference/code-benchmarks-landscape.md)
(efficiency-beyond-correctness cluster: ENAMEL, EvalPerf/DPE, EffiBench, DS-1000), and
a pointer entry in [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (DS-1000 as data-manipulation precedent; pydantic contract tasks; her agent stack).
Four ledger rows (DS-1000 with ID; three no-ID efficiency benchmarks).

### Chunk 6 (one prompt, voice-dictated)

> I'm actually really surprised that there isn't an evaluation task that's just a
> multiple choice setting with the family one example you're talking about where you
> have, like, well, you know, input size is huge. Your options are this O of n squared
> versus this O of n log n versus this O of n. Um, which one should you choose?
> Because that seems like a solid way to try to probe the agent's ability to
> explicitly retrieve and understanding about core foundational programming concepts.

Routed as an addendum to the decision-quality entry in
[topics/reference/evaluation-methodology-literature.md](topics/reference/evaluation-methodology-literature.md)
(bare-MCQ variant, commitment test, anti-memorization variants) with BigO(Bench) added
to [topics/reference/code-benchmarks-landscape.md](topics/reference/code-benchmarks-landscape.md)
and one no-ID ledger row.

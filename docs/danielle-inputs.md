# Danielle's inputs — prompt log

Chronological, verbatim record of Danielle's own prompts and framing statements from the
external conversations whose responses are consolidated into this repo. One entry per prompt:
date, conversation, the prompt itself, and where the response was routed. The point is to make
the full set of her thoughts findable in one place; responses live wherever they were routed.

The prompts behind the "top-N" lists and the "general thoughts" sections were not captured
verbatim; add them here if recovered.

---

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
[topics/early-dynamics-prediction.md](topics/staging/early-dynamics-prediction.md); PDF
copied to [refs/](refs/README.md).

### Early-dynamics draft review, second response — 2025-07 (intake 2026-08-22)

Same prompt as the entry above (Danielle asked several similar questions); response
recorded in [topics/early-dynamics-prediction.md](topics/staging/early-dynamics-prediction.md)
("Second review: GBDT v0 design details").

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


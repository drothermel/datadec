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

### Chunk 7 (two prompts, voice-dictated)

> Well, that's hilarious. I thought the event was the coolest concept, and then it
> turns out that the last author was one of my friends and favorite previous
> collaborators. Uh, so I guess it's not surprising. That's very nice.
>
> I guess I wonder then, um, so I'm not particularly interested in doing the multiple
> choice version. Um, so then I wonder whether you could repurpose the Big O benchmark
> as a source of like labeled data to produce the multiple sketches that the model has
> to choose one of and then implement that we're comparing or something like that. Do
> you know what I mean?

> Very cool. I mean, I guess I'm a little hesitant about the whole idea of publishing
> a new benchmark, just because it seems like there are a gazillion different
> benchmarks. And so I feel like I would prefer a setting where the paper relies on
> some other key thing, and the release of a benchmark is a byproduct. And it seems
> like there are interesting ideas in the analysis that you can do with this
> benchmark, which I understand then is basically we're releasing a benchmark, but
> instead of being like, whoa, I have a benchmark, then it's more like, I've done this
> interesting, very carefully designed analysis on this specific set of questions, and
> I created this benchmark as a way to do that. And then I introduced a, like, not
> super novel, but semi-novel approach that targeted one of the biggest failure modes
> that I discovered in my analysis, right? I think that that's pretty different than
> putting out a benchmark. What do you think?

Routed as a chunk-7 addendum to the decision-quality thread in
[topics/reference/evaluation-methodology-literature.md](topics/reference/evaluation-methodology-literature.md)
(BigO(Bench) mining designs, the collaborator connection, the analysis-first paper
shape with her framing near-verbatim flagged as a candidate standing principle, and a
Claude-added contamination caveat on mining real solutions). No new identifiers.

### Chunk 8 (two prompts, voice-dictated)

> So recently I've been trying to use the approach of taking a loosely plausible
> workshop subject and then trying to figure out how I could fit my ideas, like one of
> my ideas, or many, but like at least one idea to the workshop by changing the
> framing as a way to get better at thinking about the first versions of the ideas
> from different perspectives. And one of the workshops that I'm having a really hard
> time figuring out that has a name that sounds very related is the logical reasoning
> workshop at High Clear, because logical reasoning with LLMs is the full name, sounds
> like exactly what all these things are that I'm interested in, but it seems like a
> much more formal target setting or formal target methods that they're looking for,
> and I'm not really sure, like, I feel like these ideas, especially with like code
> verification and stuff like that, are, and then you get into like big O notation,
> whatever, these ideas are very close to the space of things where you have, for
> example, external logic tools that allow your LLMs to perform more reliably or stuff
> like that. But I just don't really still see how to take one of the ideas and make
> it a really good fit for this workshop.
>
> https://sites.google.com/view/iclr-2026-llmreasoning

> So it seems like, submitting, like this is a meta question, but like, it really
> seems like workshops are kind of the perfect testbed to dip a toe into areas that
> are slightly outside of your subject matter focus, and also to take a stab at
> getting better and faster at converting half-baked ideas into concrete tiny papers,
> where you could even contribute multiple to different workshops, and this seems like
> it's especially well-suited because of their two-stage submission process where you
> could start with a tight, tiny paper, perhaps even two tight, tiny papers in a very
> specific thing, and then based on the review feedback, you either improve it, then
> resubmit it, because it was rejected, or you can expand it into a fuller version for
> the second round, which seems like a really cool feature. Am I missing something?

Routed as a chunk-8 addendum to
[topics/reference/evaluation-methodology-literature.md](topics/reference/evaluation-methodology-literature.md)
(constraint-consistency as a fourth measurement axis; logic-workshop translation of
the portfolio; workshop-strategy gotchas; a second candidate standing principle —
workshop submissions must feed the big paper or kill a direction). The ICLR 2026
logical-reasoning workshop itself is historical — no workshop-deadlines update. No
new identifiers.

### Chunk 9 (one prompt, voice-dictated)

> OK, that makes sense. Excuse me. I guess a lot of your examples in the way that you could frame the things we've been discussing as good fits for this logical reasoning workshop have made me think about a category of approaches that I've been kind of wondering about for a while. And so, like, one way of thinking about them is around the, like, an idea that is very similar to taking a function that does what you want it to do, but is not in the format you want it to do, and encoding it into a representation that is not a function, and then decoding it with some type of instruction in order to make it into the format that you want. A very similar thing would be that instead of going from function to language to function, maybe you could go from function to, like, normalised function variant. So like in Python, like, I mean, I think this looks different in other languages, but I think in Python there are kinds, like, in every language, I think Python is one of the most limited, there are transformations that are functionality preserving, but that normalize the code in some way. And... linters make, like automatic fixing from linters makes great use of this. And so I think that linters especially are useful, like between autoformatters and linters, I think it is useful to enforce rules that impact performance or likelihood of bugs or things like that when you're linting and autofixing things and formatting it for human consumption. But it seems like it would be very plausible. I hypothesize that you can probably do a lot more forms of standardization that are not necessarily beneficial in some way for humans and for your ultimate code quality, but that are functionality preserving, like provably, that would take an existing code base to a much more standard version. And so then you could think of that as like, not necessarily like a language model decoder, but like some form of decoding or like, not necessarily a language model encoder, but just like some function that maps from, like, your raw Python code to the like normalized version, right? But then this raises the question of whether you can actually, like, compile the code into a language where you can actually do much more aggressive operations to normalize or optimize something like C, or like MIPS assembly code or something like that, and then convert it back in a standardized way to a much more standard variation of a Python function. And that seems like a way to introduce stability into these highly variable pipelines, basically. Like it shouldn't matter whether I put a comment above or below a certain line in terms of what the language model will produce if I asked it to do a round-trip reconstruction or if I asked it to change something on a different line, but I suspect it does make a difference. And so having some type of normalization, minification, et cetera, I suspect could make, at the very least, I would be shocked if it didn't lead to a measurable change in the performance of models in a way that was not uniform across the different model families and providers, and I think that alone would be interesting, but then I suspect there are some things that it would make better, and I also suspect that it would make it easier to optimize some types of tasks with less invasive optimizers like the wrapper system because there would be fewer variations that you're optimizing over. but I'm sure that these types of things have been explored in a variety of settings, but they feel very closely linked to the idea of using some type of constraint consistency logic to support a language model operation, and they also seem to be very closely linked to the idea of compressing code or modifying code stylistically but not functionally. And they also seem very related to things like compiler gym, which I think is a really interesting potential task setting for studying a bunch of the many, many topics that we've discussed today. What do you think?

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (chunk-9 entry with prior art — SymC, ProgramTransformer, LibCST, CompilerGym,
2403.14714 — the three-level decomposition, and Claude-added TLC ties: generalizes the
minified-X condition, output canonicalization for variance decomposition, the
wrapper-only sibling, code-side consistency perturbations) plus a consistency-axis
note in [topics/reference/evaluation-methodology-literature.md](topics/reference/evaluation-methodology-literature.md)
and three ledger rows.

### Chunk 10 (eleven prompts, voice-dictated; exchange ends mid-conversation — the last prompt is answered in the next chunk)

> But is the standard version of Python that's installed generally Cython? So like, I thought that what most people were using as Python was already at least C++ backed, if not C backed. Am I making that up?

> Okay, I think I need some terminology help. So what is an IR, and what would a canonical IR be, or IRVU? Also, I don't know, um, it's really surprising to me that you can't compile Python into other languages, into any other languages. It makes me wonder whether there's like a subset of Python that you can compile into other languages. Um, I don't know. Maybe I just really have a poor sense of how complex the main surface of a programming language that people use is.

> Wait, when you say it's not lossless, you're talking about in terms of the surface form string representation, but you're not talking about the functional representation, right? Like, when you say that you can't round trip with the Python AST, you can't produce the same function implementation, but you can produce the same functional Python function object, right?

> Okay, but if we're talking about the definition of preserving functionality that I'm alluding to when I have an autoencoder that takes code and produces a natural language description and then generates code from that description and then uses a test set to test behavior, then like, unless, it sounds like what you're saying is unless that behavior that we're testing is some type of inspection, so like, doc strings, file names, errors, debugging, etc., so unless it's some type of inspection, then like, any test set will pass. Any test set that passes on the input function and then round trips through AST will pass on the output function too, right? So like, in effect, it's lossless in terms of function preservation, as long as you're not like inspecting the function.

> Okay, but you still seem to be pivoting back to the idea that the tests are what allows me to make claims about AST-level normalization. And it seems like restricting the set of things that I am agreeing to do, like restricting the space of the Python language that I would use, in like, at least from a heuristic sense, a pretty clear heuristic of, like, I mean, there might be many, many, many different functions and gray areas, but you can list the major categories of things at a high level pretty clearly, right? Shortlist. And things that you're just really basically never testing, especially in the types of code that we're looking at, right? And so then, I think separately from saying, well, I define equivalence as the test passing in both cases, and so then AST-level normalization is lossless. I think that it's like, if the type of equivalence I care about is the type of equivalence that can be measured by having a fairly robust test set, but obviously it can't be exhaustive, that also is the type of equivalence, that means that AST-level normalization is lossless. This is the space that I'm interested in working in, but then it seems like if we assume or define that we are operating in that space, then we should be able to compile to lower levels than AST-level and make manipulations there too, right? Why wouldn't that be the case?  Because it seems like from the transient property, then like, if the trip from Python to AST to Python is lossless, then anything where the trip from AST to that thing to AST is lossless in the way that we define lossless, so lossless asterisk, then like, we should just be able to go from Python to that thing back to Python and have the behavioral equivalence at the level of strength that we care about, given the constraints that we are happy to accept.

> Okay, but I'm sorry, I'm gonna stop at the part where I'm confused. So what you say is, manifesting Python programs that are equivalent under your test oracle collapse to the same lower level of the representation, and there is no principled way to reconstruct any canonical Python program from that representation. And so, I understand that that is a many-to-like, it's a one-to-many mapping problem, right? We have one IR, and then there are many Python programs. But in the setting where we only care about producing a single correct sample from the many, given the one, then it seems like there is like a greedy decoding method of taking the one IR and decoding it into one Python program. Like, one approach to this would literally be to cache, like, for every IR chunk, then we have a mapping from that IR chunk to a canonical form in Python that it represents, where you could, in fact, write it in many, many, many, many different ways, but we define that this is the one that we are going to use. And so, like, obviously, this would be a suboptimal way to do it, though it might not be truly infeasible in the sense that, like, our ability to process data is actually really large, and Python is heavily compositional. And so, in... your standard, like, small to mid-sized library setting, then I suspect you actually could, if you chose the chunk sizes right, or even like intelligently, you probably could actually just cache the transformations. But all of this to say, in a one-to-many setting, as long as you just want one of the many, then, and especially in a setting where there's like, there's a heuristic, there are very strong heuristic ways to produce a verifiable output. Like I understand this isn't true in like an infinite space of whatever dimension and da-da-da-da-da. But like, this is something where it's a fairly small target set. And so I just, I don't see how the idea that you're collapsing many different Python functions into one IR is a problem. Yeah, I don't see how it's a problem in terms of using IR to canonicalize a Python function, let's say, under our lossless star constraint. What am I missing?

> Okay, sorry, I'm stopping halfway through your answer again. So, you're saying meaning is local, and chunk boundaries are not stable. I hear that. But, like, again, from the perspective of heuristics, then this seems, like, I'm not making an argument about programming language theory from the perspective of guarantees that can be applied uniformly across the full set of all possible programs. But if we focus on the, like, let's consider the distribution of programs that exist in the code bases of researchers and software engineers who started programming after, I don't know, 2010, and their first primary language was Python, right? That's a huge set of people, and that also, I think, happens to cover a substantial portion of the people who are in tech making these benchmarks, trying to make their tools work, etc., right? In those spaces, then maybe you have to drop some subset of libraries from the set of things that you can optimize, right? Like, maybe you have to do that because you can identify the places where you're doing inspection, eval, exec, things like this, right? But like, even that should be possible to identify with a high level of accuracy with some heuristic-based rules in this, like, distribution that we're talking about, right? And so then... Also in this distribution that we're talking about, you can do, like, honestly, you can do manipulations on the Python level, let alone on the AST level, to make the code more modular. And then you can compile this more modular code down to a lower IR in such a way that you don't have to preserve the boundaries when you come back up. You're not trying to locally reconstruct in a composable way. You did your decomposition in AST land. You moved down to IR land. You normalize and come back up to something that is no longer composable and pretty, but because you already separated your helper functions and stuff like this out in AST land before you started your round trip to IR, then I don't see why that wouldn't work as, like, a very strong heuristic that would likely substantially improve outputs, even if in some settings it would make the output look worse, but it would make it fairly canonical. Um And so when you say that lowering collapses distinctions that are semantically irrelevant for execution, but essential for reconstruction, like, you say once those are gone, any canonical reconstruction is arbitrary in ways that affect future transformations. But if you are regenerating the Python code, After you do all of your transformations by hopping from IR to AST to Python, then I don't see what the issue is. And more importantly, if we are not looking at the round trip, like, I still don't see why you couldn't, in the vast majority of cases that we hear about, do the round trip, but even without that, can't you compile the Python on one side of your autoencoder down and the Python on the other side of your autoencoder down to what's closest to machine language as possible, and then transform it according to the things that you're allowed to do in this low-level language, and then compare, and it seems like that would be a way to get a much easier comparison case than trying to compare in either the Python level or the AST.

> Okay, I guess I'm willing to accept that. Um, so then, my understanding of what you said prior to my extensive attempts to argue that it is possible to round trip um all the way to IR. I think you said basically that you have Python, then you have a CST, then you have an AST, then is the next level IR? Like the lowest level IR? Are there like more levels between like bytecode and AST? And I guess my understanding of AST is that like, most of the operations that we can do to normalize code in our existing dev tools are operating at the AST level. And so while you can do a lot, you can't really do like, it's not like you can basically say, normalize this given these constraints to make choices about each of the decisions that you'll have to make um to arrive at a function that is an equivalent AST form, even if functionally they were equivalent, functionally they are equivalent, like maybe by definition of generation or something like that. Um, is that true?

> Okay, but I'm still not satisfied, basically, because I hear what you're saying, right? That AST normalization does not equal global canonicalization, right? What I disagree with is that global canonicalization requires search, not just rewrites. And this is where I will go back to my caching example, where, okay, so we say there's too much composition, right? We have too much compositionality in Python. So then maybe we say that our target is to have canonicalization on the order of units that are some level of complexity, right? Like, I don't know, pipe, like compositional units of nest depth 3 or something like that, right? That's like high, that's still a large set of units. But if you first take all of the compositional units of nest depth 3, and then you convert them into AST, and then you do all of the different types of ways that you could normalize them, then you end up with, I think, a much smaller set of total units. And then I think there are some pretty standard things that can be implemented differently, but that do the same thing. Like, for example, your sort example. And I feel like you could implement a set of rules. That were rule-based mappings, so you don't have to search for everything. You have rule-based mappings that are used based on the definitions of sort, for example. And then you use that to canonicalize, and then you probably, yeah, I mean, sure, that will mess up occasionally, in which case having a verifier if you want it to be 100% would be necessary. I see that. But also, it seems likely that in a lot of these cases, while you can't just directly compile Python to another language, there are likely, in the same way that there have to be a fairly, like a bounded, not extreme subset of mappings that you can do between, like, semantic implementation categories that show up a lot, then I think there must be ways to map many of the concepts in Python to other languages, especially if you use certain types of AST transformations to try to unwind Python-specific things into a more explicit form that could then be mapped directly into C, for example, or C++ or whatever. And then, once you move into something that actually has the restrictions built into the language about typing and memory management and all of this, then you could do crazy optimization on it because you took your original Python code and you transformed it in a safe way into something that then you could transform in a safe way to C. And then in C, you can safely transform it to a different version of C code that is canonicalized in some way, and then that can be implemented back into Python. And so, like, again, this is a super long pipeline. I'm not saying this is the way that you should do this, but at the same time, like, linters and semantic fixing and things like black are basically just super long pipelines of rules. And so it's not that wild to be like, you know, we could use the sequence of steps to cover large swaths of the types of settings that would canonicalize the final text output of Python. Okay, explain to me why I'm wrong.

> Okay, I'm admitting that I am again stopping halfway through. Okay, so what I understand you're saying is, rule-based systems still embed search. It's just amortized, implicit, and front-loaded. But then, okay, so the distinction you're drawing is between a sequence of rules, like the rules used in linting, where you can prove that operating, like applying an operation, like lint fixing, applying an operation, provably preserves the functionality of the code because it preserves, like, it would be like having a math equation where you've just transformed the math equation as opposed to substituting or something. Okay, even in that scenario, I guess what I'm confused about is that, so if we can say that you can modify units, and then things are all basically composable, then the way that you solve most math, like most complex math problems, is that you do a collection of strange substitutions. So maybe you have some, like, limit assumptions or whatever, and then somehow everything just kind of magically falls into place. And I guess it's just really hard for me to believe that there's not a version of rule-based substitution where you could, in fact, unfold. Many, like, there are many surface text form Python functions that if you were to manually write out the path that decomposes the function into its independent pieces and decomposes the other form into its independent operation pieces, then you would see that they were equivalent. And therefore, that's not a form of search, that's a form of math equation solving in the same way that transforming an AST tree from one thing to another that preserves the identity is like math, not search plus verify. And so I guess I'm confused. I'm not saying literally nothing is search, but it seems like there's probably substantially more that is, very handwavily, more math than search and verify than the set of operations that the AST allows you to do in a clean way.

> """
> Once you translate Python semantics into C:
>
> you’ve chosen one memory model
>
> one evaluation strategy
>
> one control-flow encoding
>
> one exception strategy
>
> That choice is already arbitrary relative to Python semantics.
>
> When you translate back, you’re not “recovering Python” — you’re generating a Python re-encoding of a C program.
>
> That’s fine if your goal is:
>
> performance
>
> analysis
>
> comparison
>
> But it’s risky if your goal is:
>
> stable canonicalization for future edits
>
> invariance under small changes
> """
>
> I guess I think that you're assuming too many goals here, right? So like, I'm not making the hypothesis that taking a round trip by some convoluted way to C and then back to Python would allow language models to operate more effectively on a Python codebase across the board, right? That would be a hypothesis, but I think that would be a wild hypothesis, right? That's not like it's testable in some sense, but like, you test it, and then why? How? There's so many variables in that, right? I am making the hypothesis that while it does matter what the output format is, there is an impact to having a consistent output format for each type of operation in Python. And I think that it's possible the impact of that would be that you dramatically reduce the state space, because then it is easier to compress components. It may be easier to compress components because by having a canonical form, canonical maybe is the wrong word in the literal sense, but having a consistent form that certain types of operations are represented in then allows you to identify. Things that are in fact the same operation, and if you are a weak coding model, then it allows you to be fit to pay attention to a certain type of format because you always make mistakes when you see something that looks like this, and maybe this is kind of a strange output, but you can just track that whenever you see something like this, you've got to make sure to do this other thing. And so that's really, I mean, that's already too broad of a hypothesis, but that's the type of hypothesis that I'm interested in testing. And so I don't think it's really relevant whether I'm generating a Python re-encoding of a C program. If my goal is to generate a consistent version of different types of functionality, like behavioral functionality in Python, consistent ways of representing them in valid Python that you can actually run, I think that that type of thing can be done much more effectively in a language where you have strict typing and a lot of other constraints, because then you can actually identify which things are the same and which things are not the same, and you can merge, convert, etc. in that language before then bringing it back to the more expressive, less well-defined, less specifically defined language. Does that make sense?

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (chunk-10 entry: representation ladder, test-oracle-relative equivalence as
paper-ready formalization for TLC's C_s, the compile-down-and-back resolution,
rewrite-theory math-vs-search boundary, IR-distance as drift metric and
implementation-leakage operationalization, and her refined consistent-output-format
hypothesis). No new identifiers; e-graphs/equality-saturation named without citation
(Claude-added pointer flagged in the entry).

### Chunk 11 (five prompts, voice-dictated; the resolution of the canonicalization thread)

> okay, I swear I'm actually reading through everything that you're writing. I'm just doing it very slowly. And when I hit on a point that I'm confused about, then I am responding to it. Um, so like, I just read this section about how the reason I can say two sorting functions are equivalent is because I'm doing semantic matching on sorting, right? But, like, and I think that that's, like, um, okay. I see that, but I don't think that's the point I'm trying, like, that's definitely the point I made, right? But I don't think that's the point I'm trying to make. I think the point I'm trying to make is something closer to if we define the properties of a, I'm gonna call it a function, um a like chunk of code, a function that has inputs and outputs, um if we define the properties of a chunk of code that we care about as being, like, some version of big O notation, some version of like the scalar on the big O notation, um any type of side effects on the inputs, and then the set of outputs, then there are many There are many things that you can prove are equivalent on those properties through things like recursive proofs and things like that, right? Like, I feel like this is what we did in algorithms class. And so I do agree that you cannot prove that all of the steps are the same, and I think that that is fair. But you can prove that the runtime category is the same, the output will be the same given a, um, like a general form of an input for many cases, and that, like, at the very least, that there are no side effects. Um, there are definitely settings where you can prove that there are no side effects in two different functions, and I suspect you can prove a wider range of things. Um, and so I see your point, which is basically, I did not just say that these two functions are equivalent, but under the properties that I defined to be the key values that need to be equivalent for me to consider them equivalent star, then I can mathematically say that these functions are equivalent. Uh, is that, like, I mean, that's still not mathematically precise enough, but is that, like, closer to an accurate description of the levels of assumption I'm trying to make that are beyond just, well, I semantically think these are the same thing, um, that are more, like, principled than that, but are not claiming that these are the same programs.

> Okay, I got stuck again. So I'm at the point where you said, I think we're not disagreeing at this point, but this is more like an interesting question. So this is very similar in spirit to SSA forms and compilers, canonical IRs and theorem provers, desugared core languages and functional compilers, and byte pair encodings and NLP, lossy but regularizing. And that byte pair encoder encoding clarifies the exact parallel that I'm thinking about, which is basically what I, in my head and envisioning, but did not map to the like example where I was envisioning it from, is learning a tokenizer over the space of Python programs, but considering primitives that are much larger than byte pairs, and basically normalizing the text in such a way that perhaps destroys, like, for example, let's say that you remove the indentation and the impact of spacing by introducing some type of like colon character that doesn't actually exist in Python, but that allows you to reconstruct the text form Python and also in a more consistent way tokenize the text form with like a learned tokenizer based on the distribution of the data set. I suspect there are like some key details about how tokenizers work that would make this harder, but like, you can definitely do word tokenization, char tokenization, byte pair encoding tokenization. And so I don't see why you couldn't define like max chunk sizes and maybe min chunk sizes or something like that. Like I think the process of optimizing a tokenizer is pretty flexible, and that's basically exactly what I was imagining, is tokenizing. But like, maybe like a templated tokenizing or something like that. So it's like a mashup between templates, like composition and tokenization, which I realize is not actually what byte pair encoding does. Anyways, I'm curious what your thoughts are on that.

> A quick note, mainly to myself, but also to you. I really like the phrase behavior-preserving normalization. I think I also feel good about representation regularization, kind of, but not really, because I think that, like, basically, the word regularization feels really overloaded. The word semantic, I think in some cases is very precise in a way that I don't quite understand, and in other cases is very loose. And so I find it a little bit uncomfortable as a word to try to define a precise concept with. And then, like, idiom has very, well, I mean, my main association with the word idiom and idiom standardization is, like, a linguistic concept, which probably has a very precise definition, but then that feels like it doesn't necessarily map cleanly or obviously, at least to me, to what we're talking about. So behavioral-preserving normalization sounds great, and I will take the feedback that canonicalization is not what I'm doing. But if you could briefly explain what the definition of canonicalization is, because, like, I guess the settings that I've heard the phrase canonicalization are settings that are inherently, like, ill-defined. Like, basically, okay, so you're trying to canonicalize a set of strings that you extracted from a text document into a ground truth set of entities, right? And so, like, I guess in that sense, you could say your target is that you want to achieve canonicalization, but you really have no hope of actually achieving canonicalization, and because that feels very fuzzy, then I think of canonicalization as a somewhat fuzzy term to describe a, like, target that you are proposing a solution to try to obtain, but there is no chance that you or anybody else who is proposing a solution will ever actually obtain it.

> Okay, updated note, because I reached this part and your response. I like abstracting programs into equivalence classes even more, and I like whatever I said was my favorite so far in the last comment.

> """
> Now your earlier IR intuition finally clicks:
>
> A stricter language makes property extraction easier
>
> Lower IRs make Big-O, effects, and costs more explicit
>
> You’re not asking for reversibility
>
> You’re asking for faithful abstraction
>
> Then re-emitting Python is just:
>
> choosing a representative of the equivalence class
>
> Not “recovering semantics.”
> """ - I feel so seen lol, this is what I was trying to say at the beginning, but finally understand how to say now an hour later!
>
> “We define program equivalence relative to a set of observable properties (functional behavior, complexity class, and effects), and treat programs equivalent under these properties as interchangeable for downstream learning and analysis.” - recording this here for my future note taking

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (chunk-11 entry: the precise reduced-entropy hypothesis, property-indexed
equivalence and the abstraction framing, terminology decisions with her keeper
sentences verbatim, the structural-tokenization / program-vocabulary-induction thread,
and Claude-added TLC ties — C_s generalization, macro-token vocabulary as a
deterministic competitor baseline, the normalize-vs-measure-divergence tension, and
the candidate adoption of the new vocabulary into the doc's objective statement). No
new identifiers (code2vec/code2seq, grammar induction, e-graphs named without IDs).

### Chunk 12 (seven prompts, voice-dictated; the post-resolution debrief)

> Ok, I think you explain this in the next answer, but since im reading this now:
>
> "canonical behavioral abstraction" vs "canonical syntax" - is it possible that I internally map "canonical" to "canonical behavior abstraction" in most settings why the concept of "canonical syntax" didn't even occur to me?

> Okay, I made it to the part where we're talking about tokenizers, um, and specifically you're saying you need reversibility relative to your abstraction. And something that I think is becoming clear to me, both from my meeting with my advisor today and then in this conversation, is that, like, a pretty substantial constraint that is imposed by using black box LLMs is that they're trained over a distribution of text, and you really can't change that distribution, right? So, like, if I was gonna train my own LLM, then I could tokenize all of the Python code I could get my hands on and then take the best learned tokenizer ever and train my LLM based on those tokens. But the trade-off of having something so generally capable is also having to operate in the realm of the things of the format that it expects in order to get the best types of behavior and performance. And so in this case, what that means is that we need something that, like, we don't in the literal sense need, but to have, like, a high likelihood of, like, improving the system, then we almost certainly need something that will return whatever tokenization, encoding, abstraction that we produce into something that looks close enough to the distribution of Python that the language models are able to operate over it, um, at least well enough that it doesn't wipe out any gains from the effect of the abstraction, which is kind of an interesting trade-off that I hadn't, like, fully understood, like, I had not made it precise earlier, but I think it's much clearer to me now, that specific trade-off.And now that I've made that work precise, then I can also see how you could fairly trivially design an experiment that would actually show some version of that trade-off on a plot in a real-world setting, which probably would be an actually interesting plot, which is cool, and not probably so hard to make, at least in like a constrained setting, to illustrate a point. [adviser-related passage elided per standing convention]

> Wow, this is so clean!
> """
> . What this would actually look like (concretely)
>
> A plausible pipeline:
>
> Parse Python → AST or CST
>
> Normalize syntax (indentation, sugar, ordering)
>
> Identify frequent subtrees / patterns
>
> Replace them with typed macro tokens
>
> e.g., SORT_LIST(xs)
>
> MAP_FILTER(xs, f, p)
>
> Learn or define decoding rules back to Python
>
> Train / evaluate models on this representation
> """
>
> One day I will be able to go from my vaguely articulated but strongly held intuitions to a clear description and a crisp simple experimental setup that tests that one one thing quickly and in a convincing way!

> "Choosing a unique representative for each equivalence class, such that the mapping is deterministic, total, and idempotent." - ahhhh, I now understand where the confusion came from, this is an incredibly strong statement lol

> Again, for my notes:
> """
> 6. Clean terminology map (for your own notes)
>
> Here’s a quick cheat sheet you can keep:
>
> Canonicalization
> Unique, idempotent representative per equivalence class (strong, rare).
>
> Normalization
> Reduce variation, enforce conventions, preserve invariants (weaker, practical).
>
> Behavior-preserving normalization ✅
> Normalization where observable behavior is the invariant.
>
> Representation regularization
> Good concept, but “regularization” is ML-loaded and vague.
>
> Semantic normalization
> Too ambiguous unless you define semantics very carefully.
>
> Idiom standardization
> Linguistically overloaded; not precise enough.
>
> If you use behavior-preserving normalization and then explicitly define:
>
> “behavior is measured by X, Y, Z”
>
> you are on extremely solid ground.
> """
>
> Also, wait, my goal with the (f -> enc -> d -> dec -> f) setting is to produce "d" a "behavior-preserving normalization" of the code input "f" to "reduce representational variance while retraining functional equivalence under a defined observation model."  I just realized that my goal in both of these settings is literally identical, down to trying produce that single specific entity!  And thats why these ideas felt related even though one lived in the world of "Variational Autoencoders" and "ELBO bounds" and "RL-style optimization" and the other lived in the world of linters because they are two very very different approaches (on the surface) to accomplish the same thing for approx the same set of purposes in exactly the same domain! so cool!

> Wow. I don't think that I have had this level of like, whoa, it's beautiful type of joy since being an undergrad and learning foundational concepts in physics and math. It's so nice when things just fall out that way. It's really, really cool. Okay, um, and so now we're gonna move into the part that's like bubble bursting, which is, it's really cool, but it also seems, in some sense, obvious. Um, making me wonder, is this a, like, pretty standard comparison? And do people use this, um, I don't know, equivalence currently in different types of settings? I guess, well, I mean, in this form of formalization where we're talking about program normalization pipelines, then, like, that implies a coding setting. But, like, I think you could probably normalize other things too, less easily, probably, than code. But, like, so I don't think that has to be a coding setting. So, like, is this parallel used in any major work that is, like, well-known?  Or like, put another way, not my advisor, because I think my advisor is a terrible example. I would totally buy that I would say this and he'd be like, well, duh. But if you sample your average, fairly competent machine learning researcher or computer scientist or whatever, um, if I said this to them, would they be like, well, obviously. Do you think? Or would they be like, huh, interesting?

> """
> Everything you’ve been saying fits this model:
>
> Big-O equivalence → same class
>
> Same functional behavior → same class
>
> Same side-effect profile → same class
>
> Different surface forms → same class
>
> Different IR encodings → same class
>
> Then:
>
> normalization ≈ moving within a class
>
> compression ≈ collapsing class variance
>
> decoding ≈ choosing a representative
>
> Nothing about this requires:
>
> full semantic equality,
>
> syntactic sameness,
>
> or theoretical completeness.
> """
> - a very clean mapping

(One adviser-related passage elided from the second prompt per standing convention;
full verbatim in the transcript bundle.) Routed to
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (chunk-12 entry: the distribution-compatibility constraint and its inverted-U
experiment mapped onto the pitch's COMP-NL/COMP-SHORT knob; the d-unification merging
the deterministic-normalizer thread into TLC as the rule-based arm of the same
objective; the obviousness check — abstract interpretation, e-graphs, BART; the
equivalence-class metric vocabulary; the one-knob-one-plot template as a walkthrough
candidate). One ledger row (BART 1910.13461, agent-supplied).

### Chunk 13 (five prompts, voice-dictated except the drafted message)

> Okay, so two questions now that I'm no longer feeling defensive about the concept of round-trip faithfulness, compiling, etc. So, like, I think my very loose mental model that I like very much, I'm not saying is true, would be that, like, there is some code that only works, like, there is some code that you have to explicitly compile, and you have choice in an easy way over what compiler you use, and I would put, like, C++ in this category. And so then when I think about something like compiler optimization, then I'm thinking about a compiler that operates on whatever the, like, I guess it's a compiler IR that you generate from compiling something like C++ into that. Um, but I guess it's not compiling then, like, you called this something. Um, lowering the C++ code into the compiler IR in order to then compile the code where compiling Involves applying transformations that preserve the functionality while optimizing the functionality by some definition of functionality that honestly doesn't matter that much in this specific case for C++ to me, um, but while optimizing the... Like runtime, basically, uh, and probably compressing. So you're probably optimizing the runtime, you're probably optimizing the memory usage, if possible, and then you're probably optimizing the binary size, um, for the most part with a compiler. Um, and so that's why something like CompilerGym is an interesting setting, because, as you pointed out so clearly, this is the type of setting where the order of application of different transformations makes a difference, but if you can find a better policy to guide your compiling process, you can get huge gains in the, like, final qualities of your compiled code. Is that accurate? Um, so then the piece that I'm less clear about is I always kind of imagined that you could regenerate C++ code or C code from the compiled binary, but based on everything we've said, I'm not actually sure that, like, I think I might lean towards guessing that's not true, um, which I'm now curious. So that was my first question. I had a second question. That was why I started. Oh, okay, and then my second question was basically, I'm not necessarily saying this is the right path forward, but I do think that I'm now convinced. that this formulation shows up in a lot of different settings, and like, different settings, but like, also different, like, domains where you're crafting, like, not application domains, but like, um, kind of algorithmic domains, like, areas of study where you're crafting solutions to a given problem. One of the standard tools in the toolbox is to abstract, what was it? We've got this of attracting things, programs, into equivalence classes under behavioral properties in order to accomplish XYZ thing, right? And so if I've now identified two plausible ways that I think, like, there's definitely some version of both the programmatic manipulation and the auto-encoding setup where you could create a pretty cool representation that would give you different types of benefits. What those benefits are, not clear. How many things you'd have to try before you got that, not clear, but like, I'm pretty convinced that there's, like, at least one, probably many, many ways to get cool outcomes from these different approaches, and you probably would get pretty different types of cool outcomes. Um if you use like a standard approach to either of these things, and so that makes me think that if I was to really sit down and consider it, there are probably a lot more, like, classic techniques across different domains that target the same goal of the abstract representation, behavior preserving. Anyways, we're going to call it an abstract representation for right now, of equivalence classes, but that target that same outcome, but then get to it in somewhat different ways, which are shaped by the history of the domain, but also the different types of standard assumptions that you can make or like domain-specific settings where there probably is a way to take those different approaches and, like, pull out the pieces that you could apply and leave the pieces that are domain-specific, and see how this impacts the set of evaluation cases where you are looking at different ways, different types of abstract representations of the same equivalence classes, or possibly different equivalence classes, to see what types of benefits they might provide. And so it seems like one path forward. As a, like, structuring activity is to, like, collect, not collect them all. Like, that's a joke. That's a Pokemon reference, but, like, be on the lookout, collect them, talk to people about them, see which ones I can find. And just by having a collection, I think that opens up the possibility for interesting comparisons, compounding, um, or even an interesting suite of things to try whenever I find a new type of application for that kind of abstract representation. Okay, those are my two thoughts, both for my own reference, but also I'd like to know what your thoughts are on them.

> Okay, I just got to the part where we're talking about the trade-off that I can now plot, where the constraint is explicit that models expect a certain distribution, like pre-trained models expect a certain distribution of Python code and are optimized to operate on that distribute, like in that distribution. And as you're describing the plot, and I'm realizing that this is actually one of the arguments that I had with my advisor today, in like a positive way, but like, his strong pushback to my proposal was that if I'm suggesting that we want to, like, the loosest sense in autoencoder, with the natural language laden that we optimize on a reconstruction, but like a behavioral reconstruction loss, why does it have to be natural language, basically? It's like, it can be any form of compressed representation. There's no reason that it should be natural language. And I was basically like, that's fair, but I really think it has to be natural language, kind of. Like, it doesn't have to be perfect natural language, but like, it kind of has to be. And like, I can show you experiments where it does better when it's natural language, but they're not super precise yet. And like, I feel like there is a relationship between the explicit constraint that I did make, which is that I want these to be black box models that are trained by some, like, large model provider in a fairly general way for coding tasks. And so it just, like, I tried a few times to figure out how to express why that constraint implied that the version of this autoencoder that really seemed plausible was something That produced natural language. And I think it's fair to argue that maybe it doesn't produce natural language, maybe it produces a shorter functional implementation or something like that, but like, I don't think there's anything in our setup to guide the model into doing that. And so it seems like the easiest thing to guide the model into doing is something that kind of approximates natural language, and if we want the decoder to be able to decode from the intermediate representation, especially across different decoders, then, like, realistically, the thing that all of them are grounded in is natural language and then code. And so that is exactly what I was trying to express, and it is exactly what this plot would show, um, which is cool.

> "Because the encoder and decoder are black-box, pretrained LLMs, the intermediate representation must live in a region of representation space where those models already have strong priors." - what! thats so clear, so succinct, my advisor would have fallen out of his chair in shock lol.  yup, this would have done it

> Ok, help me proofread (for correctness not style) this message that I'll send to my advisor tomorrow morning?
>
> """
> <start message>
> So I spent most of yesterday evening trying to figure out why I feel so strongly that d_hat "should" be natural language.  (though I think your framing of the paper is definitely the one we should use)
>
> Can you tell me if there are clear holes in this argument:
>
> **Context.** 
> - Goal: "learn a behavior preserving compressed representation of code, d*." 
> - Method: Enc-Dec setup where Enc & Dec are black-box pretrained LLMs + parameterizable harnesses. We fit the harness param(s) by minimizing a behavior-preservation loss, defined by test cases, on the code reconstruction.
> - Sticking Point: "AND the compressed representations should be in natural language"
>
> **My Actual Reasoning**
> - Constrain the possible solution space s.t. all weight-optimized components are pretrained black-box LLMs + harnesses. 
>   - **Pros:** flexibility, generalization opportunity, rapid experimentation + low infra burden.
>   - **Cons:** the input distribution where the model performs well is already fixed according to the training data which primarily consisted of natural language and popular coding languages. 
> - **Hypothesis:** with high probability, useful intermediate representation live in the "region of representation space where these pretrained models already have strong priors" (as chatgpt would say, 5hrs into our discussion about this)
> - I'm not arguing that we need to constrain the intermediate representation to natural language. I'm predicting that the best performing intermediate representations will be some form of human interpretable text (due to the training data). 
> - Then, if we hand-tune initial conditions for the first experimental setups with the goal of maximizing the probability of strong performance, its reasonable to add inductive biases towards these forms of intermediate representations (eg with instructions to the encoder to write the representation in natural language)
>
> Do you disagree with that sequence, or just with the conclusions that we should draw from it?
> <end_message>
> """

> [adviser-communication passage elided per standing convention: her rationale for attributing agent-derived phrasing in the message — she gives attribution when adopting phrases from an agent and signals proactive reasoning rather than copying]

(The final prompt is elided as adviser-communication content; full verbatim in the
transcript bundle.) Routed to
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (chunk-13 entry: the NL-latent justification — keystone constraint sentence,
prediction-not-requirement refinement, cross-decoder robustness lever, the drafted
argument structure; compiler-model closure; the collect-them cross-domain abstraction
idea with its domain map, flagged as a possible reference-accumulator seed). No new
identifiers (Ghidra/Hex-Rays are tools).

### Chunk 14 (six prompts, voice-dictated)

> "We want the intermediate representation to be decodable not just by the same model, but robustly across similar pretrained LLMs." - So this is actually one of the things that I thought about the most as I was trying to actually write this up, because I think you can make an argument that that's not an explicit goal of standard autoencoder setups, right? Like, you never train an autoencoder with the assumption that you can swap out some other random autoencoder. I mean, some other random decoder portion, and get the same type of output. And so I think it would be fair to say that that's a very arbitrary additional constraint. But at the same time, like, in the same way that I didn't have a strong argument for why it would be useful for the representation to be natural language, and therefore it needed to come out as a theoretical implication from the system for it to be a constraint that we kept, I think in this case, I have a pretty strong argument for why it would be useful and like broadly considered useful to be able to swap out similarly pre-trained LLMs for either component in the system. And so while I don't think that that's necessary, I do think that we can build a strong, not case, but like, I think it can be a strong improvement to our proposed method and overall analysis to be able to do that type of swapping out. And therefore, it's worth designing the system in a way that lets this happen if we can easily. So I landed on not including that, but you are correct that it is kind of an implicit assumption.

> "You are not learning a latent from scratch; you’re discovering a representation that already aligns with existing priors." - your quote game is strong! well said!
>
> Ok, maybe we could end this conversation with your taking our main learnings (aka the things I learned from you patiently explaining things to me) and formulating them into a simple but strong potential set of experiments that I could run first to explore some key hypotheses?

> Wow, I just made it to the section where you're describing the different places that this structure shows up, and contrastive learning is exactly it. Collapsing nuisance factors, very clearly, also equivalent. That's really cool.Wait, I was gonna ask about bysimulation, because this was a, like, set of approaches that I thought was really cool in a like intuitive hand-wavy sense for trying to handle um like partially observable or like um procedurally generated, which presumably are different environments um that some of my friends that I worked with previously used a lot, and I always thought the concepts were cool, but I never quite understood, like, theoretically how they worked. But as we were going through the like sequence of trying to precisely define what I wanted, it really seemed very similar to the concept of using a bysimulation metrics, a metric, um or like whatever you, like, I'm sure there are a bunch of terms that are pretty close to that, that describe different things, um but to try to isolate some type of invariance in an environment um that allows you to operate in not like pixel space, but instead in some more meaningful representation space. And that is, in fact, on this list as well. That's pretty cool.

> Wow, that really is so cool. OK. But I think that also means that especially in our setting, where we have basically uh like clean mapping to a state action reward loop, then we could also directly, I mean, directly or, you know, kind of directly try to adapt successful approaches to by simulation in those environments to this type of setting. Not to say that it is the right type of approach, because those environments are quite different than this one, but there are obviously some similarities, and so that's a place to look, I think. But like, what is a much, like, I think the by simulation definition makes it clear that that is a drop-in definition that makes our statement, like, more precise and succinct at the same time. Um, that's awesome. But from like a functionality perspective, contrastive learning seems like something that could be smoothly introduced to the types of settings we're talking about in a way that would be really interesting. Um, and so that feels like a much more tractable direction for not quick wins, but like quick learnings, I guess. Um though, I would have to think more about what that would exactly look like in my mind.

> So I feel like we have a pretty solid way of evaluating equivalence between functions, right, with our test suite that's generated. But what is not clear to me, and I think hasn't been clear in the discussions in general, is a good way to evaluate the representations themselves. And I'm wondering if there's a way to use a contrastive learning-like approach to rank, I mean, like, to learn an ordering or like to learn a score or whatever, on the encoded intermediate representations that would let us have some, like, quantitative and somewhat grounded way to argue for some type of ordering.

> So when we say the pairwise is Y is one if both decoded programs that pass the same test, same equivalence buckets, then in practice would that be pass all or pass none? Or is this, this is an aggregate over all of the samples, and we're saying basically that the like, the subset of tasks that I and J passed is equivalent, then Y of I, J is one?

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (chunk-14 entry: cross-decoder decodability settled as emergent benefit not
assumption; the closing experiment suite E0–E5 consolidating the conversation's
hypotheses; the bisimulation frame with a Claude-added degeneracy caveat; the
representation-scoring menu — Q(d), cross-decoder agreement S_multi, contrastive
ranking, soft/in-expectation equivalence labels upgraded to per-test outcome vectors,
tied to the existing fractional-test-pass machinery). No new identifiers.

### Chunk 15 (four prompts, voice-dictated; includes a next-day resumption)

> Wow, this is really cool. So basically, I think my main remaining question is whether in the workshop version of our submission, I should update the way that we've been describing these concepts to use these new terms and explicitly call out that we're using these different approaches. So, for example, when Cho, my advisor, suggested that I do the simplest possible optimization loop where basically you just keep sampling actions until you get one that produces a reward higher than your current reward, and then if you can't within n steps, you stop your optimization loop. And if your optimization is producing values of reward that are slowly petering off, then you stop your optimization loop. And that's like a super valuable approach, but I think that like a really clean standard description of what we're trying to do is a multi-armed bandit, and there are a lot of standard, like the first thing you try is approaches and simple baselines that you would normally compare to. And so then the option, in that case, I think it's totally reasonable, basically, to do the his version because other people use his version, and so I think that that's a solid baseline, but like, Ultimately, I think we should, at the very least, try using multi-armed bandin formulations, and I think it can be in the appendix, but like, we should try it. And so then I think similarly with the concept of pi simulation or the idea of using a contrastive objective to rank the intermediate representations, it's not clear how many of these ideas should be just like included in a very simple, like, we did this by that, cite standard way of doing that, details in the appendix way. I think that's fair, but it also feels pretty substantial to shift our framing so like, so dramatically, I guess, to a much more formal version overnight. I don't know if that makes sense.

> That's really helpful. Um, just so that my mind is not spinning thinking about this, and instead I should be going to sleep, can you give me just like a brief description of what that, like, conference paper level next step might look like, given our discussion? So like, a paragraph or two, um just to kind of get a sense of what you were imagining so that I can insert it into my brain and then say, look, we figured this out for right now, and we can return to it later.

> And a paper like that could be submitted to, like, a standard conference and get, like, serious consideration for being accepted?

> Okay, it's the next day, and I have thought a bit about all the things we discussed, and I realized that it would be really useful if you could go through our conversations and pull out a summary of all of the design decisions that we've talked about for the different projects that are listed, because that will make it much easier for me to then go back through and pick up the ones that are most relevant to what I'm working on today.

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (chunk-15 entry: the greedy-loop-as-baseline / bandits-in-appendix decision with
its curriculum/budget-allocation convergence; the lens-not-commitment framing rule and
decision table; the conference-paper unification horizon and venue sketch; the
conversation's own design-decision digest, preserved verbatim in the transcript with
the repo entries as the durable cross-referenced version; one new detail — hand-
designed 10–30-construct macro set before any learned vocabulary). No new identifiers.

### Chunks 16–17 (six prompts, voice-dictated; session of 2026-02-08 — the conversation ends here)

> Today, what I'm interested in doing is thinking about how the experimental results that we already have could be packaged into some type of analysis paper about the behavior of LLMs and how they vary across, like within model-spec pairs, you get a huge variance. And then for a given model across specs, you get a huge variance, and then across providers, you get a huge variance. And I think that there just has to be, I am too interested in the outcomes of these experiments for there not to be some interesting narrative to be made from them. But it would need to be a narrative that could be submitted to an iClear workshop that has a submission date that is the 10th or later. So can you start by telling me what we've already covered in this direction? Because I feel like we talked about it. And then after you do that, I'm going to ask for you to look for a good fit for workshop if we haven't already covered that.

> Ok great, so before To create a candidate storyline, I think the most important first choice is to, or like first task, is to go and see what workshops are available with ICLR, because I think creating that narrative storyline will need to be directly targeted at one of the available workshops. And I don't think there are any that are like, basically just like an exact fit for this. And so I think we'll probably have to do a little bit of tuning of the narrative based on that. So can you go and search for all of the ICLR workshops that are, that have deadlines the 10th or later, and lay those out along with your, your view on what the most promising work that we could try to shape a narrative with this data around?

> Ok, so I think CAO is optimal but the deadline is super tight.  Then, I like the Re-Align Challenge track idea +  the Logical Reasoning of LLMs seems solid but I'm already planning a submission of work related to this around contradictions within (model, spec) pairs if we use a structured latent.  So I think it would be cool to aim for Re-align?

> interesting, I htink that C would be the coolest but I'm not sure what we'd need to make that claim.  what do you think we'd need for a workshop paper level claim like that?

> I like this a lot. I think one piece that I don't see in this, but that we have the data for, and I think would be really interesting, is to look at not just the failure modes, but also the behavioral diversity. So like, I have the analysis on basically like using AST, all of the different types of like if, while loop conditions and their presence, the level of complexity of the solution, variable naming length, length of lines, total length, like just a bunch of different things that are metrics on the code that are relevant things with like comments, doc strings, et cetera. And so it seems like the strongest argument for there being a lack of collapse is not just that there isn't collapse in the success versus failure reasons, but also that there clearly isn't collapse in like the output types for like fairly basic setups. Does that make sense? Do you think that that falls into the same category, or is that just like interesting but not really relevant to the main point of the paper?

> Okay, I completely agree. Then I'm gonna have a different agent look at the state of my code currently and then plan what we would need to change to be able to, or like implement to be able to collect the results for this workshop paper proposal and chunk it into, like, well-scoped tasks. So can you give me a final clear description of what the workshop paper is that we're trying to put together, what pieces we need for it, what the hypotheses are, and the main plots that we need to make to support those hypotheses, basically any information this agent would need to be able to put together a clear, like, actionable plan for what to do next?

Conversation dates now precise: chunks 1–15 on 2026-02-06, chunks 16–17 on 2026-02-08.
Routed to [topics/reference/evaluation-methodology-literature.md](topics/reference/evaluation-methodology-literature.md)
(chunks-16–17 addendum: the variance-analysis paper planning, the ICLR-workshop scan,
the Re-Align choice, the two-kinds-of-collapse distinction with success-conditioned
strategy diversity, and the full "Behavioral non-collapse in cheap coding LLMs" spec —
outcome an open intake question) and
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (instrumentation recap, the AST-metrics data asset, the planned structured-latent
contradictions submission, precise dates). No new identifiers.

### Post-conversation artifacts (2026-08-23, before the walkthrough): the submission, reviews, presentation, and the recommended paper

> ok, can we actually do a bit more before processing: talk about the workshop we did
> submit, look at the version we submitted, look at the updates I made in my
> presentation to my advisor + the one very useful looking paper he passed in earlier
> rounds of feedback (he was very non-convinced by anything, so then everything turned
> after these last artifacts so I just want to document them now.
>
> We ended up submitting to "ICLR 2026 Workshop on Lifelong Agents: Learning,
> Aligning, Evolving"  the paper is attached.

> Then, this is the presentation I made for my advisor to try to layout the actual
> work (because I ended up staying up two nights in a row and writing the whole thing
> myself in the extra 12 hours they delayed the deadline because my collaborator got
> sick and the results took that long to get, so the paper was VERY poorly written).
> and then, also attached it the paper he recommended that is quite relevant.

(The three OpenReview reviews, meta-review, and decision were pasted in full and are
preserved verbatim in the bundle's `openreview-reviews.md`; the submitted paper,
2026-02-25 presentation, and Miao & Blunsom PDF are beside it.) Routed to
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (submission-record entry: the LLA workshop paper, condensed reviews with the
AC-accept/PC-reject outcome, the presentation's franker readout and per-stage retro,
the DQE research statement, and Miao & Blunsom as the closest formal ancestor with the
LM-prior ↔ distribution-compatibility lineage observation), a new entry in
[topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md),
and nine ledger rows (Danielle-supplied; Miao & Blunsom verified against the PDF).

### Walkthrough round 1 (2026-08-23)

> ok, so there is no AST-metrics code but I do have some codebases around code
> processing that I can point you to later.  no other version was submitted, or even
> close to being submitted.
>
> agreed with adding statement as a dated program level note.  [one adviser-related
> clause elided per standing convention]  strong yes on miao + Blunsom and the rest of
> the must-reads.
>
> then, I also just found the discussion convo that I used while writing the paper,
> and I suspect it actually will make it clear the pieces I was confused about.
> basically as I was writing I came to the conclusion that I guess we should use the
> compression framing my advisor dictated but I didn't really understand it and really
> I understood my previous framing much better but I hadn't thought it through
> sufficiently to either do both or do mine in a way that wasn't a clear "I'm
> rejecting your instructions" way.  so I tried very hard to figure out how to do his
> with no sleep in very very little time.  and the conversation makes this  clear, so
> I think we should ingest it after you do this round of processing?

Applied: DQE statement as a dated program-level note in
[potential-projs/README.md](potential-projs/README.md) (marked not-adopted-at-the-time);
approved must-reads added to
[litreview/tlc-litreview-plan.md](litreview/tlc-litreview-plan.md) (Miao & Blunsom
gate-1 with PDF, DS-1000, BigO(Bench), e-graphs, TransCoder pair, SymC/
ProgramTransformer, curriculum novelty check, bisimulation skim). Facts corrected in
place: no Re-Align or Logical Reasoning submission was made or close; no AST-metrics
code exists (code-processing codebase pointers to follow). Framing-provenance note
added to the TLC §4 submission-record entry; the writing-period conversation is the
next intake. Still open from the walkthrough: TLC §1 adoption of the NL argument +
vocabulary; TLC-0 metric consolidation; divergence staging promotion; the four
candidate principles.

### Walkthrough round 2 (2026-08-23)

> 1(a) yes, and NL-as-prediction-not-requirement is really important, we should make
> that very clear.
>
> 2(a) yes, findable but not decided on yet
>
> 3-yes promote!
>
> 4 first two yes, lets add the second to with annotations about thos caveats

Applied: (1) a "Why the latent is natural language — a prediction, not a requirement"
block in [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§1 — the keystone constraint, the prediction-not-requirement emphasis with its two
falsifying experiments, the cross-decoder hierarchy, the adopted vocabulary, and the
Miao & Blunsom lineage; (2) a "TLC-0 candidate metrics" consolidation block at the end
of §1, explicitly findable-not-decided; (3)
[topics/staging/model-behavioral-divergence.md](topics/staging/model-behavioral-divergence.md)
created (hypothesis verbatim, the ready spec, distance designs, the corrected
no-AST-code fact, the time-sensitive OpenRouter snapshot note, gate), with the topics
README staging table and the evaluation-methodology pointer updated; (4) all four
principles added to
[topics/reference/project-approach-principles.md](topics/reference/project-approach-principles.md)
— benchmark-as-byproduct and module-or-kill as full principles, one-knob-one-plot and
lens-not-commitment as technique-level with the discussed caveats annotated.

### Writing-sprint conversation, chunk 1 (three prompts; historical, ~2026-02-09/10; intake 2026-08-23)

> Okay, I'm currently in the process of trying to write a paper about choosing the best prompt for an LLM that's trying to accomplish a task. The task will yield a reward. We want to maximize this reward, but I'm trying to decide how best to define "maximizing reward" in a meaningful way. We're viewing this as a multi-armed bandit problem, right? where different prompts are different bandits, and we have a finite search space. We're comparing it to UCB1 and Thompson sampling, and then we have "our method," which involves using an LLM in the loop to select the best prompt within the space, or rather, to select prompts within the space. One approach, as I understand bandit algorithms, is that they iteratively improve an estimate of the probability distribution over the arms for reward. Then, at the end of what I'll call the training set "epochs" or "pools," one approach would be to select the arm that had the best probability estimate (i.e., was estimated to have the highest probability) as the best arm. This would then be evaluated on a test set to determine the bandit algorithm's performance. If you were to do this with the LLM as the optimizer, then similarly, as the LLM chose different prompts to consider, you would examine the reward received for each prompt. You would then apply the same logic: from the prompts actually tried, which one yielded the highest reward? An alternative framing involves cumulative reward or regret, meaning you're not looking at a test set, but rather the training set itself. You would compare the total reward accrued over the course of "train time" (which is matched across all settings). In this scenario, the performance of an algorithm is defined as the best cumulative reward. However, it seems that in such a case, the algorithm needs to be designed with knowledge of the terminal state, which I believe UCB1 and Thompson sampling inherently possess to some extent. But it seems unfair for the LLM not to have information about what the terminal sampling will entail, or rather, that this is the goal. It seems that if we're giving the LLM this additional information on how to select actions to outperform basic baselines for bandit algorithms, then a key piece of information it should receive is how it's being evaluated. Currently, we're only providing historical performance based on its choices, along with the options it can choose for the next round. So, what do you think?

> So I think that this is actually part of what I find confusing, because the main pitch of the paper is that we can compress code into a compressed representation by using an encoder-decoder setup where the encoder and decoder are black-box LLMs. So, we pass in a function, and then it gets encoded into a description or an intermediate representation—whatever that is—and then it has to get decoded back into the function. However, we are evaluating based on behavior, not surface form. So, does the reconstructed function pass the same tests as the original function? This gives us an objective that we can then optimize against. And so, the idea is that the LLM as optimizer can optimize against that objective, right? And so then, the real thing we want is to show the actual compression performance of each of these approaches. Whereas, the reward we're talking about is: does the model hit a certain budget for compression? So, does the encoder actually encode into something that is within a budget, and then does the decoder decode into something that is able to pass the tests? The reward is based on that. And so then, what I initially imagined we would be plotting was the x-axis being the maximum budget we told the model—like we told the autoencoder it had to hit—and the y-axis being the success in terms of hitting that budget and passing the tests. But then, when I talked to my advisor, he suggested instead that the x-axis should be the actual cost. So, not the budget we told it to try to hit, but instead the actual length of the intermediate representations being the cost versus success, which is different. And I can see why that's useful because then you can get a Pareto front. But then, it seems like we're talking about a few different things because the thing we're optimizing is the success metric, but then the thing we care about evaluating is the compression performance. And honestly, even a bandit algorithm would be a plausible approach because it hasn't been used for this purpose in the literature. So, what I've been referring to as our method—which is the LLM as optimizer—doesn't necessarily have to win, though it clearly does win in the sense of being useful, because the bandit algorithm can really only work up to a certain size of search space. Whereas, the LLM as optimizer has a built-in prior, which allows it to work at a much, much, much larger search space, if not a semi-infinite search space. And so, I guess that's why I'm confused about how to make the comparisons. And just to be explicit, the way we're defining the arms is that the prompt for the encoder has, for example, three to five different slots, and each slot can take three to seven different values. And so then, we have to constrain that space to make it plausible for the bandit algorithms to have to pull all of the arms. But then, you could actually use the full space or even a bigger space if you were using the LLM as optimizer. And so, I'm at the point where our deadline is fairly soon, and I have a working implementation of everything and a lot of samples collected, but I'm just not quite sure what to plot. I have most of the paper written up in terms of the structure, and it's a workshop paper, so it doesn't have to be really solid. However, I do need to get the relevant plots and then put them in and update the claims a little bit, and I just don't even at this point know what I'm claiming. Because I thought I was claiming that the LLM as optimizer was better, but then as I think more about it, it's really just a question of how many pools and how many samples you use to get your estimate per pool. Basically, all of the hyperparameters directly impact which methods you'd expect to be better. In some sense, it's very unclear how you would actually do a fair comparison the way I've set it up, especially because I don't know what my metric is, whether it's the regret, online performance, or best arm identification. Does that make sense? I need some guidance. It doesn't even have to be right; I just need some sense of which direction to double down on for the next four hours.

> vOkay, so this makes sense, but the point where I'm getting confused is I don't understand how this collapses. I don't understand how to collapse our experimental design into this. So, what we're sweeping is this: I have a function family of synthetic functions, right? The function family is bucketed by difficulty D. We chose one difficulty bucket from this function family. Within this function family, we choose from 50 total specifications. Each specification corresponds to a function—specifically, a Python function that has a test suite, right? That is the function we use for encoding and decoding. The encoding is done with a specified maximum budget, right? Up until now, we've been saying that if the length is higher than the maximum budget, it immediately fails. However, in this scenario, we don't want to say that. We want to have a maximum budget and then measure the test success rate. So, specifying a budget is actually just adding a different arm in the arm space, but that blows up the arm space. Is it adding another arm? Is that something you're just sampling over? Are you actually looking for the Pareto frontier of the budgets we selected? It's like a hyperparameter. I don't know how to collapse that. Also, when sampling with the arms, we've differentiated between sampling a spec—which is a task in our dataset—versus sampling a solution from the spec, which is a reconstruction process that yields a performance evaluation. Two more hyperparameters we have to choose are how many specs per pool do we use, and how many samples per spec do we use when computing the reward for an arm? I'm also not sure how to collapse that into these descriptions. We're using characters for the cost. We're using characters for the budget description in the prompts. At this point, we don't have time to regenerate all the data. So we're using characters. And so I guess we're using characters for the cost as well. But I don't understand how to collapse these things into the Pareto frontier. You are describing the same thing as my advisor, but I also did not understand, when he explained it, how we would collapse what we're doing into that.

Documented verbatim in the bundle's companion transcript
(`writing-conversation-transcript.md`). Routed to
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-1 entry: BAI vs. cumulative regret, metric-disclosure
fairness, the three-conflated-axes diagnosis, Story A/Story B, the
budget-as-conditioning unlock, and the drop-in paragraph that became the paper's
Eq. 1). No new identifiers.

### Writing-sprint conversation, chunk 2 (four prompts; 2026-02-16 — date corrected from the earlier ~02-09/10 estimate; source link in the bundle transcript)

> Okay, so then I have a question about paper framing. So basically, I'm not sure. Initially, I had a very different framing for this project, and in that framing, I was pretty confident that the LLM as optimizer would perform better. Now, the paper's framing has changed so much that it's no longer clear to me that the LLM as optimizer will perform at the Pareto frontier, depending on what parameters we choose, right? And so I guess my question is this: This is a workshop paper, right? It doesn't have to be the most state-of-the-art thing. And I'm pretty sure that the idea—this combination of things—is novel. It does not exist. Having done a pretty substantial literature review, in terms of using this type of setup to compress code, especially using an LLM black-box-based encoder and decoder, with the code reconstruction being a behavioral equivalence. And then looking at the compression Pareto frontier, that's new. And so if that in itself is new, then my main reason for wanting to write this workshop paper so quickly is because I'm really interested in the idea of using LLMs as optimizers with systems that have LLM components, like this autoencoder, in order to fit them in the same way that you would fit a trained model with weights using normal machine learning approaches. The idea is that you can actually replicate many different types of algorithms and explore a lot of interesting concepts related to representation learning, identifiability, continual learning, etc. There are many things that you can do with this setup, and that's what I'm really interested in exploring, since I didn't see any paper that introduced all of these components together, despite doing a pretty comprehensive review. And so I wanted to put out a paper that included each of these pieces to stake a claim. But now the paper's framing has moved away from that, which is fine as long as we keep the LLM as optimizer in. But I think my question is this: Is there still a way to frame the paper, even if an LLM as an optimizer only shows up as best in settings that are clearly skewed towards it performing optimally? Such as settings where UCB couldn't possibly cover the whole sample space, or settings where, I don't know... Yeah, I mean, basically, settings where UCB couldn't cover the whole sample space, either because T is low or because the space is huge. Then, how do I frame that as the submission? Because it's no longer a statement that the method I'm proposing is superior, even though the idea of using bandit algorithms and treating it as a multi-arm bandit problem isn't something that has been done before. So, in some sense, all of these settings are new. Yeah, I just don't know how to think about framing this. And I don't want to hit the end of my three-hour window having not thought about it, because I'm going to have to update the abstract and other materials, regardless of what the outcomes of my results are.

> Okay, so do you think that AUC of the success versus cost up to max cost is a better sample efficiency plot? Or do you think expected cost at target success is better? It seems like expected cost at target success is easier. But how do you choose the target?

> so I was thinking to use the following for figure 1:
>
> - x = cost (IR length)
> - y = s_m(c) = best success achievable at cost ≤ c found by that method with N evaluations
>     - This is the method’s empirical pareto frontier/envelope
> - S_m(c) is the success of a method at cost c, which is binary based on whether the method found a compressed representation that was successful at this cost

> For the sake of time, if I have multiple stochastic decodes per spec—given that 't' is greater than the number of specs and you are sampling specs, which makes it very likely we'll sample the same spec multiple times—can I use the simple equation, 'Hanaut,' the optional enhancement, because it's simpler and I have a very limited amount of time? Or is that really aggressively aggressively bound? I'm not necessarily optimizing for this to get accepted. I just want to submit something plausible.

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-2 entry: the framing question with her stake-a-claim vision
as the DQE statement's ancestor; framings A/B/C and the A+B decision the paper
implemented; AUC-primary metric choice; the exact Eq. 1 / Figure-1 caption origin;
the coverage-vs-probability decision and its link to R2's later critique). No new
identifiers.

### Current-state additions during intake (2026-08-23): compression-thesis origin, decoupled sampling, infrastructure

> yessss, your points are very accurate.  and I think this was why I was confused that
> somehow I ended up doing prompt optimization when thats not really what I thought I
> was doing....
>
> in future convos with my advisor I was like "but if our goal is compression why
> would we ever setup a system like this one?? aren't there better ways?"  and he was
> like ... I don't understand, obviously this is a compression paper... which didn't
> seem to answer my question.  but ultimately I realized he was certain that by
> combining lossless and lossy compression we could beat all lossless compression at
> least on a pareto frontier level.  which I agree is true in theory, but  I feel like
> in practice claiming that all we have to do to write an easy straightforward paper
> is to **beat LOSSLESS COMPRESSION** seems kinda crazy.  but it is the task we now
> have, which means we actually do need a very well tuned complex prompt optimization
> algorithm because the baseline is a crazy hard bar that has been aggressively tuned
> for decades.
>
> also, one real benefit that my initial design had that I wasn't able to clearly
> articulate is that it took a very complicated pipeline and made it so you could do a
> substantial amount of sampling up front (the infrastructurally hard part) while
> still being able to do "optimization" because the search space is basically fixed.
> whereas I've spend more than a month trying to get my first optimization results
> with prompt optimization because while I could just run COPRO once or twice, if I
> want to be able to run it enough times to be able to make claims and then log things
> sufficiently to debug, etc, then I actually need a distributed system.  because the
> inner part is (prompt -> remote inference -> result -> remote inference -> result ->
> compute heavy parsing -> sandboxed evaluation).  and when you add a complex
> optimization loop around that its actually kinda  nightmare.  but I think I have it
> basically fully implemented for (COPRO, MIPROv2, GEPA, codex direct prompt proposer)
> using building block repos that work together.  now is maybe a good time to look at
> those.
>
> all should be public, all thats missing is the code-comp repo (not written yet) that
> combines whetstone-ai and dr-code to actually set up the exps.  this isn't what I
> was using for the submission of course, but its what this next round will use.  the
> general design principle is that agents will often mess with things they shouldn't
> so I tried to pull each level of primitives out so that we could freeze them during
> exp running without concern + update different pieces with clear versioning over
> time.

(Her ten-repo list with one-line descriptions follows in the original; recorded with
local versions in the TLC §4 entry.) Routed to
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (current-state entry: the compression track's beat-lossless goal with provenance
and her difficulty assessment; the decoupled-sampling architectural insight; the
optimizer-infrastructure inventory verified against local clones and the whetstone-ai
README — COPRO/GEPA platform-wired, MIPROv2/Codex-direct in-process; code-comp repo
pending; the whetstone-envs minigrid env flagged as a possible small-project spec-out).

### Writing-sprint conversation, chunk 3 (four prompts; 2026-02-16), with intake note

Her intake note (verbatim):

> note: my ultimate goal was that basically noting was copied verbatim from this convo
> into the paper, I honestly can't remember in the rush in the end whether I got to
> rewriting all the sections or not. but regardless, even if I was to submit to
> exactly the same venue with the same results I'd do a substantial rewrite...

> Okay, so then returning to the figure two question about AUC versus the other option, I guess I don't know how you calculate AUC because it it's um you don't have a curve. So how do I calculate it? DC, is this something or there's just like a standard SK learn? Function or something like this? Um how do I get at UC? Uh I'm totally sold on competing AUC. Just tell me how to get it

> Amazing. Now let's say I've written the paper based on this. Um, and then the last piece that I want to have is generalization experiments, which I was imagining doing according to the following notes. Um, does this make sense with what we've laid out? Or are there key changes you would suggest?
>
> ### Generalization Experiments
>
> - We ran each method on the train benchmark for a fixed budget T decode-vals.
> - Output a recommended prompt / arm which maximizes the training score under a fixed scoring rule: solve rate under a cost cap → fraction of train specs with ≥ 1 pass and cost ≤ c_0
> - Take the selected prompt, evaluate on
>     - diff langauge
>     - harder synthetic family
>     - HumanEval++
> - Evaluation
>     - run fixed prompt n=1-3 times, recording cost and pass rate
>     - if the range in cost is too low we can sweep budget but aim for one fixed value for eval (somewhere in the middle?)
>     - compute fraction of specs solved vs cost cap

> ok, can you give me how you would phrase the generalization experiment intro in the methods/exps section?

> Awesome. Can you also give me a caption / methods or exps section introduction for figure two?

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-3 entry: AUC mechanics, the two OOD-protocol saves — c_0 from
train only and post-hoc cap sweep from realized costs — the drafted templates, the
rewrite-provenance note with a §3.2 spot-check, and a Claude-added note that the
drafted AUC(N) Figure-2 plan partially changed before submission). No new identifiers.

### Writing-sprint conversation, chunk 4 (one prompt — her consolidated working document; 2026-02-16)

Reproduced with the same representation as the bundle transcript (unquoted because of
nested formatting; bracketed placeholders stand in for chunk-3 assistant material her
document embedded wholesale, preserved verbatim in the transcript's chunk 3):

<<<BEGIN DANIELLE CHUNK 4>>>
Ok, so this is the structure I'm working towards, any last changes before I start updating our writeup to match this & making the plots?

```
### Framing

1. **Task**: behavioral code compression with black-box LLM encoder-decoder system and unit test based behavioral equivalence criterion.
2. **Metric:** compression-correctness frontier (task coverage vs compression rate)
3. **Algorithms:** bandit baselines + LLM in the loop optimizer as a general search policy
4. **Findings:** regime dependent tradeoffs; LLM opt scales to larger spaces and remains competitive under tight eval budgets.

**Our primary metric is task coverage: whether a method discovers at least one passing reconstruction under a given cost cap within N evaluations. This differs from estimating the per-attempt success probability.**

### Goal

Answer the question: How good is the Pareto frontier we discover given N evaluations with each of the algorithms.

- Black-box multi-objective optimization
- Cost = length of intermediate representation (the thing we want to minimize)
- Quality = passes tests, 0/1 (the constraint)

### Figure 1: Given budget N, which method finds the best cost-success tradeoff?

*Per-Method Empirical Pareto Front*

→ **We evaluate methods in a best-of-N offline selection setting: after N evaluations over the encoder-prompt search space, the method selects the minimum-cost intermediate representation found which is decoded into a program that passes the original program’s test suite, our behavioral equivalence criterion.  We define cost as the intermediate representation length in characters.  If no passing candidate is found within  N evaluations, the run is counted as a failure.**

Let m be a method, s be a task (function) and each evaluation produces (c_i, y_i) with cost c_i in r and y in  0,1

- Per spec: S_{m,s}(c; N) = max over specs seen up to current step i, over cost_i less than c of y_i
    - → By cost c, did method m find at least one passing reconstruction for spec s with N evaluations?
    - → after spending N attempts, did we get at least one success
- Aggregate across specs (plot this)
    - S_m(c, N) = 1/|S| sum_s S_{m,s} (c, N) → [0, 1] → fraction of specs solvable under cost ≤ c after N evaluations.
- x = IR Length (chars)
- y = fraction of tasks solved (unit test pass) at IR length ≤ x → best observed success among attempts with cost ≤ c (per spec, then averaged)

→ coverage vs cost

<aside>
**Figure 1:** Compression–correctness frontier. For each method mmm, we plot Sm(c;N)S_m(c;N)Sm(c;N), the fraction of specs for which the method found at least one reconstruction that passes the unit tests with intermediate representation length ≤ ccc, after NNN total evaluations.
</aside>

### Figure 2: Pareto Frontier Quality vs Number of Evaluations

- x = number of evaluations spent (same for all methods)
- y = AUC of success-vs-cost up to a max cost C_max
- C_max → largest IR char length you consider meaningful
- How to compute it, the details

    [embedded verbatim copy of the chunk-3 AUC recipe, code snippet, and gotchas]

<aside>
[embedded verbatim copy of the chunk-3 Figure-2 caption and methods-intro drafts]
</aside>

### Generalization Experiments

<aside>
[embedded verbatim copy of the chunk-3 drafted generalization-intro paragraph]
</aside>

```python

### Recommend a prompt **for a target operating point**

[embedded verbatim copy of the chunk-3 points 1–6: Score(a; c_0) selection; c_0 from
train only; k decodes with coverage/pass@k-under-cap; post-hoc cap sweep not
instructed-budget sweep; per-method prompt-selection fairness; final train/test
structure]

```
```
<<<END DANIELLE CHUNK 4>>>

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-4 entry: the four final calibration edits and their landing
in the paper; the notes-layer provenance observation). No new identifiers.

### Writing-sprint conversation, chunk 5 (six prompts; 2026-02-16 — the section rewrites)

> Ok, I need help, looking at the current state of the draft there's no way I can rewrite from scratch fast enough. I'm going to ask you to take the section we have and give me a better one and thne I"m going to edit that down.  starting with the abstract:
>
> Code summarization is unlike natural text summarization. Code conforms to strict semantic and syntactic rules, and is primarily written to be executed, in contrast to natural text, in which style, sentiment, medium, function, and other aspects may all be crucial to its meaning. The meaning of code is in its function. Code summarization, then, can be judged on the basis of a simple metric: the success rate of \emph{reconstructing} code from the summary alone. We leverage this formulation of code summarization success for a novel autoencoder framework: we continually improve the prompt of a black box LLM for code-to-summary encoding, maximizing the code reconstruction success of a frozen black box LLM decoder. We experiment with several optimization algorithms: Upper Confidence Bound and Thompson sampling as baselines, and an LLM-as-an-optimizer approach.
> Viewing summarization as compression, we also find a clear cost-performance tradeoff; maximum summary length (cost) modulates summary completeness and correctness (performance), as measured via code reconstruction success. Thus, we provide users with a mechanism to traverse this cost-performance Pareto frontier. Our findings generalize across LLMs, coding languages, task types, and datasets.

> will this approx work?
>
> We study code compression with a behavioral equivalence constraint, where an encoder maps a program to a compact intermediate representation (IR) and a decoder reconstructs executable code whose behavioral equivalence is evaluated by unit tests for the inital program.  We use frozen weight pretrained LLMs for the encoder-decoder structure and treat encoder prompt selection as a black-box search, comparing UCB1 and Thompson sampling to an LLM-in-the-loop optimizer that proposes prompts using semantic priors.  Under a fixed budget of reconstruction-test evaluations, we evaluate methods by the compression–correctness frontier they discover, defined by task coverage as a function of maximum IR length.  Additionally we explore the sample efficiency of frontier discovery, finding a clear cost-performance tradeoff: bandits are competitive in small enumerable prompt sapces, while LLM-guided search reamins effective in larger spaces and low budget settings.  We further evaluate transfer by freezing the selected prompt and testing out-of-distribution, including HumanEval++.

> ok, lets skip the intro for now because well rewrite it based on the findings.  so then, our background and methods are super out of date (ignore the comments).  these need to be completely rewritten
>
> ```
> \section{Background \& Methods} \label{sec:definitions}
> % \subsection{Formal Definitions}
>
> \paragraph{Data Samples}
> Consider $\mathcal{D}$ to be the space of possible natural language descriptions for a function in the space of python functions $\mathcal{F}$.  
> % We define $\mathcal{D}_{B} \subset \mathcal{D}$ to be the subset of $D$ with maximum length $B$ characters.
> Our datasets in \S\ref{sec:expts} are sets of specs $s = \{f_s, T_s\}$, each consisting of a ground truth function $f_{s} \in \mathcal{F}$
> % a ground truth description $d_{s} \in \mathcal{D}$, 
> and a test suite $T_{s} = \{\tau_{s,1}, ..., \tau_{s,m_{s}}\}$. $f_{s}$ is one member of a class of semantically equivalent functions.
> % with the behavior identified in $d_s$. 
> $T_s$ tests for this functionality.
>
> \paragraph{Model}
> We construct an autoencoder composed of one encoder that maps from code to natural language and one decoder that maps from natural language to code. The encoder and decoder are each defined by a black box LLM and a configuration, $\theta_{E}$ or $\theta_{D}$, respectively, resulting in an autoencoder configuration $\theta := (\theta_{E}, \theta_{D})$. These configurations do not include model weights, which are encapsulated in the black box LLMs, but consist of system parameters, such as system prompt, description length budget $B$, sampling parameters, generation process stages, user prompt template, and input formatting functions $\phi_{E}$ and $\phi_{D}$. 
>
> From the encoder, we sample a text string $\hat{d}$, intended to be a function description. From the decoder, we sample a text string $\hat{f}^{\textrm{text}}$. We attempt to parse $\hat{f}^{\textrm{text}}$ into a python function $\hat{f} \in \mathcal{F}$.
> % \begin{equation}
> % \label{eq:enc-dec}
> % \begin{aligned}
> %   \hat{d} \sim \textrm{ENC}_{\theta_{E}}(f)
> % \end{aligned}
> % \qquad \qquad
> % \begin{aligned}
> %   \hat{f}^{\textrm{text}} \sim \textrm{DEC}_{\theta_{D}}(\hat{d})
> % \end{aligned}
> % \end{equation}
> For simplicity, we refer to both the raw text string $\hat{f}^{\textrm{text}}$ and the code we may extract from it as $\hat{f}$. For each spec $s$, we sample $n$ pairs of summaries and reconstructed code $(\hat{d}^{(i)}_{s}, \hat{f}^{(i)}_{s})$.
>
> \paragraph{Metric}
> % We consider a decoder generation $\hat{f}^{\textrm{text}}$ to be \emph{feasible} if it
> We define \emph{semantic correctness} $C_{s} : \mathcal{F} \rightarrow \{0,1\}$ to have value 1 if and only if $\hat{f}$ passes all tests: $C_{s}(\hat{f}) = \textbf{1}\{\hat{f} \textrm{ passes all tests in } T_{s}\}$.
>
> However, for any generation trajectory $(\hat{d}^{(i)}_{s}, \hat{f}^{(i)}_{s})$, there are multiple earlier points of failure: (1), $\hat{d}$ must remain under the length bottleneck $B$; (2), $\hat{f}^{(i)}_{s}$ must consist only of code; (3), the code in $\hat{f}^{(i)}_{s}$ must compile; (4), the code in $\hat{f}^{(i)}_{s}$ must run. Only at this point do we evaluate $C_s(\hat{f})$.
> We define \emph{feasibility} $\textrm{Feas}(s, i; \theta) = \textbf{1}\{|\hat{d}^{(i)}_{s}| \leq B\} \cdot \textbf{1}\{\textrm{code-only}(\hat{f}^{(i)}_{s})\} \cdot \textbf{1}\{\textrm{compiles}(\hat{f}^{(i)}_{s})\} \cdot \textbf{1}\{\textrm{runs}(\hat{f}^{(i)}_{s})\}$. For a decoder-only setup, we omit the description length bottleneck term $\textbf{1}\{|\hat{d}^{(i)}_{s}| \leq B\}$.
>
> This allows us to define success, which requires both feasibility and semantic correctness:
> % \begin{equation}
> % \label{eq:success}
> $\textrm{Succ}(s, i; \theta) = \textrm{Feas}(s, i; \theta) \cdot C_{s}(\hat{f}^{(i)}_{s})$
> % \end{equation}
>
> \paragraph{Problem Formulation} We view this problem through the lens of multi-armed bandits (MABs). In each of $t$ rounds, the agent chooses 1 of $k$ possible arms, based on rewards in prior rounds. Its objective is minimizing regret, defined as the expected difference between the optimal and actual rewards. In our framing, the $k$ arms are possible encoder prompts.
> % possible configurations $\theta$. The scope of our work focuses specifically on the effect of the encoder prompt. Since all other settings in the configurations are frozen, we can view the arms as possible encoder prompts.
>
> \paragraph{Optimization}
> We employ 3 methods for selecting prompts. For both the Upper Confidence Bound algorithm (UCB) and Thompson Sampling, in each round, the agent chooses a prompt, observes its reward, and then performs Bayesian updates on a prior estimation of the rewards. In UCB, the agent chooses the action with the highest confidence interval \emph{upper} bound. In Thompson, the agent samples actions weighted by its prior. Unlike these approaches, our LLM-as-an-optimizer algorithm does not maintain an explicit posterior. At training time, the LLM is given, in context, prior prompts and their associated rewards, and chooses the next encoder prompt.
>
> % Using Equation~\ref{eq:success} 
> We define a \emph{reward} or optimization objective $J(\theta) = \mathbb{E}_{s} \mathbb{E}_{i} \big[ \textrm{Succ}(s, i; \theta) \big]$. That is, we optimize for the expected number of successful trajectories. Since code reconstruction is verifiable, $\textrm{Succ}(s, i; \theta)$ is directly estimable, unlike prior work (\S\ref{sec:related_works}).
> ```

> I need another way to say this: For a given spec s and cost cap c,
> we define the per-spec envelope (not envelope)

> ok great, then, lets do the easier related works section, can we leave this as is?
>
> \section{Related Works} \label{sec:related_works}
>
> \paragraph{Code Autoencoders}
> \citet{semanticcompression} studies autoencoders for compression of code and natural language text, but do \emph{not} impose a constraint that the latent must be natural language. \citet{cycleprompt} uses multiple cycles through an autoencoder to refine code generations, and \citet{NLDebugging} uses a similar approach with a specific application to debugging.
>
> \paragraph{Code Summarization}
> \citet{wei2019codegenerationdualtask} do not use an autoencoder structure, but jointly train code summarization and generation models. \citet{nlinthemiddle} trains an encoder-decoder for code translation and find that the best objectives train the encoder to generate natural language summaries.
>
> \paragraph{Prompt Optimization}
> Several studies have investigated RL for optimizing prompts in code generation \citep{nanocapsulator,epic} or to write a natural language knowledge summary \citep{languagebottleneck}.
> Other optimization methods have also been applied: a combinatorial search over prompts for code generation \citep{PlanningInNL}, iterative code generation and math problem solving attempts from an LLM given prior best candidates \citep{OPRO},

> amazing, then, another section that needs to be completely rewritten:
>
> ```
> \section{Experiments} \label{sec:expts}
>
>
> We create a synthetically generated set of \emph{specs}, as defined in \S\ref{sec:definitions}. We define 2 families of functions using Stateful Algorithms for the training set and Bit Operations to evaluate generalization (\S\ref{sec:gen_results}). Each vary along multiple axes (Appendix Table~\ref{tab:task_families}), with allowed value ranges for each axis. For each family we select a medium-difficulty set of axis values and we sample 50 specs from this range, balancing for each axis. Additional details in Appendix~\S\ref{sec:data_app}.
>
> In addition  use HumanEval \citep{humaneval} for generalization and analysis, and for held-out test sets. \TODO{delete if untrue}
> % consisting of a base description, axes of variation, generated functions, tests and \emph{gold} prompts at 3 levels of brevity, each corresponding to one latent character budget setting.
>
> % \paragraph{Experimental Conditions.} Direct Generation, Autoencoder-like Setup \ml{possibly best to explain these in the subsections}
>
> \paragraph{Model Selection.} See Table~\ref{tab:models-by-experiment} for selected models for each experimental setting and see Appendix Table~\ref{tab:model-details-pricing} for details about cost comparison and model specifics.
>
> \paragraph{Prompts} We assemble candidate prompts modularly. We define 7 variable clauses to include in each prompt, which contain instructions about the LM's character budget, task, goal, additional constraints, role, and brevity. For each clause, we define a set of 4 to 8 possible phrasings, including the option to exclude that clause for some clauses. We generate all possible combinations that consist of exactly 1 chosen phrasing (or exclusion) for each clause, resulting in a set of \TODO{how many} possible prompts. These are the arms in our multi-armed bandit formulation. Exact prompts with all clauses and phrasings are in Appendix~\S\ref{sec:prompts_app}.
>
> \paragraph{Harness Settings} We use top-p sampling with $p=0.95$ and temperature 0.2 (Appendix~\S\ref{sec:hps_app}).
>
> \paragraph{Code Description Constraints} We impose a maximum length, or a character \emph{budget} $B$ on the descriptions. We vary $B$ in our experiments.
> % We experiment with \TODO{update with final numbers} 3 variations: 0.25, 0.5, and 1 time(s) the length, in, characters, of a reference description length, determined in preliminary investigations to differentiate between setups. 
> Additionally, descriptions may not contain code, as identified by common python signatures, indented blocks, or excessive code related characters. Details in Appendix~\S\ref{sec:bottleneck_app}.
>
> \paragraph{Metrics.} 
> Our primary metric is average validation $\textrm{Succ}(s, i; \theta)$, as defined in \S\ref{sec:formal_metric}. 
> % This is equivalent to the percentage of encoder, decoder generation pairs $(\hat{d}, \hat{f}^{\textrm{text}})$ which fulfill (1)  (2) description length budget (2) 
> % We also define a classification of failure cases (Appendix \S\ref{sec:failure_class_app}). \TODO{do we use this}
>
> \begin{table}[!h]
> \centering
> \small
> \caption{Models used in \S\ref{sec:expts}}
> \label{tab:models-by-experiment}
> \begin{tabular}{lcccc}
> \toprule
> \TODO{}
> \\
> % & \textbf{Provider} & \textbf{Cost Category} & \textbf{Variance Exps} & \textbf{Performance Exps}\\
> % \hline
> % \textbf{gpt-5-nano} & OpenAI & Low & Yes & Yes\\
> % \textbf{gpt-5.1-codex-mini} & OpenAI & Medium & Yes & No\\
> % \textbf{gemini-2.5-flash-lite} & Google & Low & Yes & Yes\\
> % \textbf{gemini-2.5-flash} & Google & Medium & Yes & No\\
> % \textbf{haiku-4.5} & Anthropic & Medium-High & Yes & No\\
> % \hline
> \bottomrule
> \end{tabular}
> \end{table}
>
> % \paragraph{Dataset.} We first use Fizzbuzz as a high sample variance test, sampling 100 generations on this spec from experimental setting. Additionally we sample 10 specs from each of the synthetic data families at difficulty level 3.  For each we sample 10 generations for each experimental setting.
>
> \subsection{Results} \label{sec:results}
>
> % \begin{figure}[!ht]
> % \begin{center}
> % %\framebox[4.0in]{$\;$}
> % \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{4cm}{0cm}}
> % \end{center}
> % \caption{varying encoder, decoder model pairing. Emphasize off-diagonal failures highlight that seemingly well formed compressions actually encode model-specific contracts instead of portable semantics.}
> % \end{figure}
> % To get a better estimate of the performance of each portion of the system for each of the Synthetic Data function families, we consider the three families at difficulty level 3, 4 and 5.  
> For each pair of (task family, difficulty level in $\{3,4,5\}$), we sample 50 specs with a balanced sampling strategy that adheres to quotas across buckets of axis values for a few primary axes of variation per family.  For each spec we then sample 3 generations through the full reconstruction pipeline.  We also evaluate Fizzbuzz through 50 samples of the full reconstruction pipeline. See Appendix~\S\ref{sec:expts_app} for more details. \TODO{not sure about any of this}
>
> Results in \TODO{} show that \TODO{}
>
> In \TODO{}, we observe that, by adjusting the character budget, we adjust reconstruction success rates. A higher description length maximum, up to \TODO{}, results in increased reconstruction success.
>
>
> % \subsection{Manual "Optimization"}\label{sec:expts_tuning}
> % We now aim to improve performance by applying 3 modifications to the prompts: being explicit, listing potential issues, and adding a structured checklist.  \ml{Q for D: which prompt does this correspond to?}
> % Stretch goal involves experimenting with multistep CoT or multi-shot experiments. \textbf{Even very problem specific manual tuning doesn't resolve the issue, as seen in XYZ remaining failure modes}.
>
> \subsection{Generalization} \label{sec:gen_results}
>
> \paragraph{Tasks} To study whether optimal prompts found in \S\ref{sec:results} generalize, without further tuning, to additional tasks not seen during optimization, we evaluate on \TODO{how many} additional function families in our synthetically generated dataset (Appendix~\S\ref{sec:data_app}). Results in \TODO{} demonstrate task generalization.
>
> \paragraph{Models} To study whether the optimal prompts found in \S\ref{sec:results} generalize, without further tuning, to additional black box encoder and decoder LLMs not seen during optimization, we evaluate with the models denoted in \TODO{}. Results in \TODO{} demonstrate model generalization.
>
> \paragraph{Coding Languages} To study whether the optimal prompts found in \S\ref{sec:results} generalize, without further tuning, to additional coding languages not seen during optimization, we evaluate with synthetic data, but with Java and Rust \TODO{make sure this is right} as the source code language. Results in \TODO{} demonstrate generalization to other coding languages.
> ```

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-5 entry: the per-section provenance analysis against the
submitted PDF — abstract mostly hers, Background & Methods and Experiments
assistant-skeleton with her edits, Related Works closest to assistant text with three
contrast sentences shipping near-verbatim; the pinpointed exit of the pitch formalism
from the paper; the two artifacts preserved in the old section's comments — the
orphaned variance sweep and the drafted-but-unrun encoder×decoder transfer-matrix
figure). No new identifiers.

### Writing-sprint conversation, chunk 6 (two prompts; 2026-02-16 — style and citations)

> is this "tons of little paragraphs" style standard or are you copying my collaborator's writing style?

> do I need to cite any of these methods:
>
> Optimization algorithms We compare three prompt-selection strategies: (i) UCB1, which selects
> prompts based on empirical mean reward plus an exploration bonus; (ii) Thompson sampling,
> which samples prompts in proportion to a Bayesian posterior over reward; and (iii) an LLM-in-
> the-loop optimizer, which proposes the next encoder prompt by conditioning on the prompt search
> history (previous prompts and observed outcomes) and leveraging the LLM’s semantic prior over
> prompt text. All methods share the same evaluation interface and are compared at the same total
> number of decode–test evaluations N .

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-6 entry: the paragraph-style taxonomy and hybrid conversion;
the Auer 2002 / Thompson 1933 citation additions — both in the submitted references;
the recurring spec-sampling truth check, resolved as uniform in the paper). No new
arXiv identifiers (both canonical citations are journal papers already implicit in the
submitted bibliography).

### Writing-sprint conversation, chunk 7 (four prompts; 2026-02-16 — figures, intro, citations, appendix)

> Can you give me really, really specific instructions for my floating language model for how to create figure one? Um, it created a single plot with budget fifty as opposed to using all the budgets and doing the bucketing and stuff. Um, I think that it's capable of doing it with more specific instructions, but I don't have time to write them?

> ok, can  you give me a general purpose intro I can use with like a placeholder for any findings we might have based on the pltos we're currently making? sorry, I mean an introduction seciton for the paper, has to be super short because we're near page limit

> so if i cite ucb1 and thompson sampling in the intro do I cite them again in the later sections or is once enough?

> ok, can you give me a plausible appendix? we use gpt 5 nano for enc/dec and haiku 4.5 for the llm as optim.  our generalization is flash lite for the model gen.   the prompt info in here is correct, basically everything else is outdated:
>
> [her old appendix, provided in full — reproduced here abridged only where noted: the synthetic-dataset section with empty task-families table; the empty description-length-bottleneck section; the Black Box LLMs section with the full 21-row model pricing table ("prices gathered on February 5, 2026" — gpt-oss-20b through claude-haiku-4.5); the Candidate Prompts section (correct per her note) with the 8-part clause template, prompt example, and the full Prompt Clause Options table (BUDGET ×8, TASK ×9, GOAL ×4, CONSTRAINT ×7, ROLE ×6, BREVITY ×6 phrasings); the sampling-parameters table (Direct Generation / Direct Compression / Bottleneck conditions, top-p 0.95, temperatures 0.2); the commented-out 8-category failure classification (latent missing/too long, latent contains code, decoder format, signature/output type, restrictions not followed, doesn't parse, doesn't run, tests don't pass); an Additional Results stub; the old formal-definitions appendix (θ = (θ_E, θ_D) configurations, ENC/DEC sampling equation, semantic correctness, the Feas waterfall Eq., Succ, J(θ), and the harness-fitting paragraph "We propose optimizing this objective by fitting a parametric program, specified as the system harness, H_θ, where θ ∈ Θ includes all design variables in the system such as prompts, templates, latent format, generation decomposition into stages, tool use, memory, and sampling hyperparameters"); and, in comments, the complete pitch-era prompt suite — DG-MINIMAL / DG-SIGNATURE / DG-STRICT with FB-SIGNATURE/FB-SPEC and synthetic contracts; ANTI-CHEATING-CONSTRAINTS; ENC-BASIC / ENC-FAITHFUL / ENC-STRUCTURED (full text); the two-step compression pipeline attributed to "(Gilbert 2023 Semantic Compression with LLMs)" with COMP-NL and COMP-SHORT full prompt text; the FizzBuzz two-outputs contract-variation figure with caption; the bottleneck-design experimental sketch including cross-model reconstruction; and an "Outer Loop LLM Based Optimization" stretch-goal stub]

(The fourth prompt's full old-appendix LaTeX is preserved verbatim as the bundle
artifact `lla-old-appendix-pre-rewrite.tex`; the transcript and this log carry it
abridged with pointers.) Routed to
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-7 entry: the figure-agent instruction block and the
instructed-vs-realized confusion recurring in the tooling; the submission's model
roles; the pre-rewrite appendix as primary source — pitch-era prompt suite including
COMP-NL/COMP-SHORT full text, ANTI-CHEATING-CONSTRAINTS, the 8-category failure
taxonomy, the 140/280/560 budget tiers, and the H_θ harness formalism surviving in
the appendix after its chunk-5 removal from the main text). No new identifiers
(Gilbert 2023 = 2304.12512, already in the ledger).

### Writing-sprint conversation, chunk 8 (three prompts; 2026-02-16/17 — title, keywords, 7:25am)

Her intake note on the third prompt (verbatim): "note, I think this is when I realized
the deadline was really extended to 6pm not 6am, it was 7:25am, I was very tired"

> great, suggestions for title better than "Multi-Armed Bandits can Optimize Code Summarization Prompts"

> then, keywords for open review?

> does it make sense for 3 slots with 3 discrete phrasings each to have 27 arms?

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-8 entry: the submitted title as the assistant's recommended
pick verbatim, with the shelved framing-C banner titles noted; keywords; the 7:25am
deadline realization and the 3³ = 27 sanity check). No new identifiers.

### Writing-sprint conversation, chunk 9 (three prompts; ends 07:33 on 2026-02-16), with intake corrections

Her intake note (verbatim): "Oh! I misremembered, the deadline was actually originally
supposed to be at 7 and this sequence of exchanges ends at exactly 7:33am on Feb 16."
(Two screenshots of the then-current near-overlapping AUC figures were attached;
described in the transcript, final versions in the submitted PDF.)

> so these are our current figures 1 & 2, basically the least interesting they could possibly be...

> but it is noise right, the band shows that its noise?

> ok, so if we're 30mins past the deadline and the portal is still open my thought is I put them in the figure slots, get the generalization results based on the data (running now) into the fig 3 slot, remove any claims of comparison and submit, then iterate after?

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-9 entry: the legibility triage and the regime-experiment
advice traced to the paper's 8,064-arm section; the noise-vs-uncertainty answer; the
verbatim submission decision; Claude-added notes — the OOD data still running at
decision time as the referent of the presentation's "not enough time to thoroughly
analyze," and the never-run paired difference-curve analysis flagged as the cheapest
outstanding item for any revival). No new identifiers.

### Writing-sprint conversation, chunk 10 (six prompts; 2026-02-16 post-07:33 — final polish), with intake note

Her intake note on the third prompt: "(I attached a draft that I no longer have)" —
the intermediate draft between the section rewrites and submission is lost; the
submitted PDF is the nearest surviving state.

> so, changes to the abstract?
>
> We study code compression with a behavioral equivalence constraint, where an encoder maps a program to a compact intermediate representation (IR) and a decoder reconstructs executable code whose behavioral equivalence is evaluated by unit tests for the inital program.  We instantiate this as a black-box LLM autoencoder with frozen pretrained encoder and decoder models, and treat encoder prompt selection as black-box search. We compare classic multi-armed bandit baselins (UCB1 and Thompson sampling) to an LLM-in-the-loop optimizer that proposes prompts using semantic priors.  Under a fixed budget of decode-test evaluations, we evaluate methods by the compression–correctness frontier they discover, defined by task coverage (fraction of tasks solved) as a function of an IR length cap.  We also measure the sample efficiency of frontier discovery, observing a clear cost-performance tradeoff: bandits are competitive in small enumerable prompt spaces, while LLM-guided search remains effective in larger spaces and lower budget settings.  Finally, we evaluate transfer by freezing the selected prompt and testing out-of-distribution, including HumanEval++.

> ok, and then the intro?
>
> Code compression differs from natural-language compression because correctness is primarily behavioral: a compressed representation is useful only if it supports reconstruction of code that behaves the same as the original. 
> We study behavioral code compression with a black-box LLM encoder–decoder that maps programs to compact intermediate representations (IRs) and reconstructs executable code evaluated by unit tests. 
> We treat encoder prompt selection as black-box search and compare bandit baselines (UCB1, Thompson sampling) to an LLM-in-the-loop optimizer. 
> We evaluate methods by the compression–correctness frontier they discover under a fixed evaluation budget and by the sample efficiency of frontier discovery. 
> Empirically, we observe \TODO{headline frontier result} and \TODO{headline sample-efficiency result}, and we test transfer by freezing the selected prompt and evaluating out-of-distribution, including HumanEval++.

> ok, it needs to be 5 pages, the figure 3 should be a 4panel grid but can be squashed however.  highest level of importance thoughts?

> so for section 2, remove the black box llm encoder-decoder section?

> any glaring issues now?

> I'm confused, so figure 1a and 1b are in the paper, figure 2 is going to be the generalization exps?  I left the placeholder to get a sense of the paper length

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-10 entry: softened abstract/intro traced to the submitted
versions; figure plan settled; §2 collapse; the glaring-issues checklist with a
cross-check of what got fixed vs. the caption typos that survived; item 8's
27-arms-only overreach flag confirming the 8,064-arm regime postdates 07:33). No new
identifiers.

### Writing-sprint conversation, chunk 11 (seven prompts; 09:09 on 2026-02-16 — the ownership pass)

> ok, I decided to assume that the portal will remain open until 6 and also to aim to make a version of this draft I feel good about, send it to my advisor for his thoughts, but make it clear that I'm not convinced we should submit and instead should probably aim for a different deadline with some more thought given to different aspects of the exploration/design/framing.
>
> Then, I'm working through the draft to write it in my own words based on what I actually want to say, and I'm hoping you can help me with the writing aspect this time (because my diction tanks when I"m exhausted).
>
> The first question is:
>
> "Code compression differs from natural-language compression because correctness is behavioral: the surface form representation can vary substantially while being behaviorally equivalent, and this equivalence can often be effecitvely evaluated (estimated, proxied??) through performance on a test suite. "
>
> there is a word that basically means "a solid estimator or reperesentative of" that I'm looking for in this sentence.  related to abstractions?

> ok so, basically, I want to say two surface forms are equivalent.  but you really can't say that in the general case.   but if instead I say that there are a few properties that define equivalence for me, then I can find a representation of the equivalence class.  and the equivalence class is defined by behavior but we can't measure "behavior" in the same way you can't take an expectation, so instead we have to estimate it like taking the empirical estimator of an expectation.  and thats what we're using the test suite as.

> "Passing all tests is an empirical estimator of agreement with the target behavior, and we use it as our correctness criterion throughout." is this a pretentious and pedantic sentance, or is it just specific?

> ok, new word problem:
>
> Code compression differs from natural-language compression because correctness is behavioral: the surface form representation can vary substantially while being behaviorally equivalent, and performance on a test suite can often serve as an effective and (computable? tractable? verifiable?) surragate for equivalence.

> "behavioral code compression" isn't really a thing is it?  as in "We study behavioral code compression with a black-box LLM encoder–decoder..."

> but plenty of people use test passing "verifiable surrogate" for code representations being equivalent, so it seems like this isn't really a term thats ours to coin?

> ok, so this is my new-and-probably-not-improved-but-more-me abstract:
>
> """
> We study code compression under a behavioral equivalence constraint: an encoder maps a program to a compact intermediate representation (IR) and a decoder reconstructs executable code whose equivalence is evaluated by unit tests for the original program. 
> We instantiate this as a black-box LLM encoder–decoder with frozen pretrained models, and treat encoder prompt selection as black-box search over a finite prompt space. 
> We evaluate classic multi-armed bandit strategies (UCB1 and Thompson sampling) and an LLM-in-the-loop optimizer that proposes prompts conditioned on the search history. 
> Under a fixed budget of reconstruction evaluations, we measure performance via the compression–correctness frontier discovered by each method, defined as fraction of tasks solved (reconstructed code passes all tests) as a function of an IR length cap. We summarize the sample efficiency of each method via the area under this curve as the number of evaluations increase. 
> Across settings we observe a consistent cost–performance tradeoff, and we further evaluate transfer by freezing the selected prompt and testing out-of-distribution, including across programming language, to a separate LLM encoder-decoder, and to HumanEval++.
> """
>
> And intro:
> """
> Code compression differs from natural-language compression because correctness is behavioral: the surface form representation can vary substantially while being behaviorally equivalent, and performance on a test suite can often serve as an effective and verifiable surragate for equivalence.
> We study code compression under a unit-test behavioral equivalence criterion, with a black-box LLM encoder–decoder that maps programs to compact intermediate representations (IRs) and reconstructs executable code. 
> We treat encoder prompt selection as black-box search and compare bandit baselines (UCB1, Thompson sampling) to an LLM-in-the-loop optimizer. 
> We evaluate methods by the compression–correctness frontier they discover under a fixed evaluation budget and by the sample efficiency of frontier discovery. 
> Across settings we observe a clear cost–performance tradeoff, and we test transfer by freezing the selected prompt and evaluating out-of-distribution, including HumanEval++."""
>
> When I get clearer results the last sentence will become a claim instead of an observation...
>
> But beyond that, glaring issues?

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-11 entry: the 09:09 reset — own words, not-convinced-should-
submit position; her equivalence-class/empirical-estimator articulation as continuity
with the 2026-02-06 property-indexed framing; the coining-ethics exchange producing
the submitted abstract's opening phrase; the abstract's final provenance revised to
her 9am rewrite plus minimal fixes; the open question of how the not-convinced
position became a same-day submission). No new identifiers.

### Writing-sprint conversation, chunk 12 (eleven prompts; 2026-02-16 — the ownership pass reaches Methods), with her retrospective verdict

Her intake message opening this chunk (verbatim):

> I actually am really glad I did submit, because up until this point I thought I was
> incapable of writing a workshop paper, and after submitting (and now, looking back)
> I think I did an absolutely spectacular job considering the dramatic time
> constraints and the complexity of the required infra + the confusion of trying to
> move from one framing to another on no sleep...  so I think future me would ideally
> come into the deadline push more prepared but I think submitting isn't a bad choice
> as long as you believe your results are true (if not good).  thoughts?

> next, at a high level I think I'm going to collapse these three into one paragraph that introduces the task, enc-dec and eval all together, with a simple mention of correctness + compression definitions. replacing:
>
> [her then-current Methods paragraphs, quoted in full in her message: Tasks; Black-Box LLM Encoder--Decoder; Evaluation Budget]
>
> Then I think we  talk about prompt search and optimization algorithms first, with a quick mention of the task coverage curve and a reference to the appendix for details.  Then we introduce the AUC in the experimental section when we discuss plotting it and our findings  which replaces:
>
> [her then-current Compression-Correctness Frontier / Sample Efficiency via AUC / Prompt Search / Optimization algorithms / Train-Time Objective paragraphs, quoted in full]
>
> And I think thats all we need in methods unless we want to pull related work to the front and merge it with background to provide some form of framing.  butI don't think its specific enough to our claims in the paper to do that yet, so that might not be my current goal.

> hmm, I think the intro should actually start with a "why is code compression potentially useful" statement

> ok, so I"ve been speaking in the encoder-decoder autoencoder space, but the whole cost/something tradeoff curve for compression has its own standard language doesn't it?

> ok, but then , do you talk about the lowest rate for an acceptable distortion?  how do you write the equivalent of "we aim to encode f_i into the most compressed  representation that can be succesfully decoded into a behaviorally equivalent function" in information theory speak?

> ok, the notation using "i" is a bit off, but what about this:
>
> "Given a coding dataset of functions and their associated test suites $\{f, T_{f}\}_{i}$, we aim to produce the shortest intermediate representation of $f_{i}$ that still decodes into a function $f'_{i}$ where $T_{f}(f') =1$ . "

> so what feel strange about this notation is that T is of f, so making it T_i feels like its burying the lead

> good? Given a coding dataset of functions and their associated test suites $\{(f_{i}, T_{f_{i}})\}_{i=1}^{n}$, we aim to produce the shortest intermediate representation of $f_{i}$ that still decodes into a function $f'_{i}$ which passes the test suite $T_{f_{i}}$ .

> so I think that using "dataset of functions" is confusing, because I think math functions before code funcitons

> ok, if I say programs, does that make the use of "f" strange to represent the ~functions~ I mean programs?

> sigh, probability makes that confusing, I'm happy with f, ok, let me continue

> if the first time I use the abbreviation LLM is here in the abstract I need to spell it out first right?  "We instantiate this as a black-box LLM encoder–decoder with frozen pretrained models, and treat encoder prompt selection as black-box search over a finite prompt space. "

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-12 entry: the Methods restructure and its landing as the
submitted §2; the why-useful intro instinct → the submitted opener; rate–distortion-
style language → the submitted discussion, with the constrained min-rate form; the
T_{f_i} notation negotiation producing the submitted Task sentence essentially
verbatim hers; the provenance ledger flip for the paper's front matter; her
retrospective glad-I-submitted verdict resolving the chunk-11 open question). No new
identifiers.

### Correction during intake (2026-08-23)

> thats actually not true, I only got the experimental setup running sometime in that
> 42 hour period, so I got all the results during that period...

(Correcting the intake agent's statement that the results pre-existed the writing
push.) Applied as a scope-correction note in the TLC §4 writing-sprint chunk-12
entry.

### Writing-sprint conversation, chunk 13 (fourteen prompts; 2026-02-16 — the ownership pass deepens)

> does this sentance soudn "overreachy":
>
> "Given a dataset of programs and their associated test suites $\{(f_{i}, T_{f_{i}})\}_{i=1}^{n}$, we aim to produce the shortest intermediate representation of $f_{i}$ that still decodes into a function $f'_{i}$ which passes the test suite $T_{f_{i}}$ . We use an enocder-decoder pipeline made up of black-box LLMs to compress the code, and we fit the pipeline to our task by searching over encoder prompts while holding the decoder prompt fixed."

> so from my perspective we are training/learning, thats why I think the whole "llm as optim" is cool.  but I realize we haven't really presented data to back this up yet.

> I feel loosely positive about the last sentence overall but its still a bit off:
>
> [her draft Task paragraph, quoted in full: the {(f_i, T_{f_i})} dataset sentence; the encoder-decoder pipeline of black-box LLMs with fixed decoder prompt; encoder prompt selection as outer-loop learning with bandits or LLM-in-the-loop; the fixed budget N in end-to-end reconstructions and test executions; the compression-correctness frontier S_m(c;N) shown in Figure 1 capturing at-least-one valid low-cost reconstruction per task under cap c]

> do you think the section title can go from "\section{Background \& Methods}" to jsut methods?

> and do you think its fair for the related work to be at the end in this case?  or is it really a cop-out

> would that be the second to last sentence in here?
>
> [her intro paragraph draft, quoted in full — the \TODO{Code Compression is Useful!} marker, the behavioral-correctness opener, the surrogate sentence, setup, methods, frontier evaluation, and the closing observe-tradeoff-and-transfer sentence]

> this feels like it should be parallel and its not "code autoencoders, code summarization, and prompt optimization"
>
> would it be better to break ti by putting prompt opt first, eg "to prompt optimization, code summarization and autoencoders"?

> I actually like option 1 because I feel like it makes it clear we're not claiming to be first for any of these thingsg inidividually, but does unify a collection of different topics which is more some of the novelty.  thoughts?

> so "We use a black-box large language model (LLM) encoder--decoder pipeline" sounds to me like the pipeline is a black box, but we're modifying one of the internal aspects, so I feel like its improtant to tie the black-box to the LLMs that are the encoder and decoders themselves

> what about htis:
>
> We use an encoder-decoder pipeline, built from black-box LLMs, to compress the code. 
> Holding the decoder prompt fixed, we treat enocder prompt selection as an outer-loop learning problem and use either bandit algorithms or an LLM-in-the-loop optimizer to propose prompts from feedback.

> "to propose prompts based on a feedback history trace. " ?

> so I like history of observed outcomes, but I feel like "trace" ties it to RL/Sequential decision making, which is my goal... thoughts?

> sigh, I'm actually feeling more positively towards this paper, probably because I feel more ownership again, the words were yours but alot of the thoughts really were mine.  but I also don't want to want to push for a 6pm deadline. sigh

> so I'd say my problem is the results.  like, I can't have two plots that look basically identical (despite supposedly plotting different things) that show that everything I tried (1) behaved the same as each other and (2) did exactly what you'd expect them to do.  but the idea of digging into the code and trying to figure out how I can salvage that is the daunting piece.  the writing is fun

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-13 entry: outer-loop-learning and interaction-trace phrases
traced verbatim to the submitted §2 with her RL-positioning rationale; the
combination-novelty stance landing in the submitted intro; the precision-instinct
catalog with the Background-&-Methods counter-datapoint; the February ownership
moment; the results diagnosis with the salvage menu — B and D executed before
submission, A and C outstanding and bridged to the divergence instrument). No new
identifiers.

### 2026-08-23 — writing-sprint conversation, chunk 14 (final two exchanges; sent with the compaction request)

> Note, these were the last two, and you're going to autocompact any moment so can you
> give me a compaction instruction instead of filing these away so we can be sure to
> preserve the important context during compaction?

> [12:54pm] Okay, I have a question. You say add early end frontiers, but I don't know what that means. Is it basically that we had figure one where we have the compression versus success and that's for a fixed N and we bucket? It by actual cost, and we plot different curves for each of the different methods, but then if we were to introduce shorter N, then we would get different curves. Overlaid. That would look different because they were for shorter portions of fitting. Is that what we're talking about?

> Okay, I have a new question. So if I were bucketing, how big does it make sense to make the buckets? Are we looking for a visible step function or something that looks more like a curve? Because it seems like we want a step function, and that's not quite what I have. But I feel like my plots look nice, but they also look boring.
> Also, I know that we aren't including information about the budget because we're bucketing on cost, but is it important that I keep the budget set that I use in the evaluations fixed across the methods because it seems like it would be, but maybe it's not.

> [2:32pm] Question. So if I'm going to add the n prime equals lower numbers, then do I actually need to resample for each of those lower numbers or can I just filter to those lower values for the length?

Routed to [potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 (writing-sprint chunk-14 entry: early-N' frontiers = prefix filtering by attempt
index, not resampling and not cost-filtering — the bias warning kept as a standing
analysis rule; bucketing rule of thumb ~30–60 x-points; evaluation budget N must be
matched across methods, instructed budget B fixed/scheduled/in-shared-arm-space; these
became the submitted appendix Figure 4 multi-T panels). Transcript preserved first at
her request (compaction-proofing), routing filed after. Conversation complete: 14
chunks, all 2026-02-16, ending 14:32. No new identifiers.

### 2026-08-24 — Notion lit-review intake begins

> ok great! so next lets pull in some lit reviews I did that are in Notion, can you remind me what our processing approach is?

> great! please pull this page first: https://app.notion.com/p/Lit-Review-2fcde135cd1f80358304da2a8f39ede3?v=2fcde135cd1f8087ae97000c181d4a2e

Routed: verbatim copy with resolved mentions in
`~/drotherm/data/convo-artifacts/2026/2026-08-24-notion-lit-reviews/` (batch INDEX
there); dated entries in
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 and [topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md);
five ledger rows (RTC 2402.08699, Self-Debug 2304.05128, CodePlan 2309.12499,
Reflexion and Yuan et al. without IDs); RTC added to litreview plan row A. No new
identifiers assigned.

### 2026-08-24 — Notion lit-review intake, page 2 (Consensus) + her read of the novelty-check corpus

> I would note that my reading of these different lit reviews is that each interpreted my prompt differently and therefore the "novelty" conclusion varied substantially based on what they thought I was proposing and therefore what related literature they considered.
>
> next: https://app.notion.com/p/Lit-Review-2fdde135cd1f8021a3f7d2f93d64bdbc?v=2fcde135cd1f8087ae97000c181d4a2e

Routed: verbatim copy in the batch bundle; dated entries in
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4 and [topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md)
(her note verbatim in both; yesterday's "rubric strictness" framing corrected to her
prompt-interpretation read); nine ledger rows + an authorship-discrepancy note on
2507.08627. No new identifiers.

### 2026-08-24 — Notion lit-review intake, page 3 (Claude)

> ok, next: https://app.notion.com/p/Lit-Review-2fdde135cd1f808d9386c7142796f3f7

Routed: verbatim copy in the batch bundle; dated entries in
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
§4, [topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md)
(plus a Miao & Blunsom provenance addendum — it was in this 2026-02-04 review before
the advisor recommended it), and
[topics/reference/prompt-optimization-landscape.md](topics/reference/prompt-optimization-landscape.md)
(the Feb record already held OPRO/EvoPrompt/PromptBreeder); seven ledger rows. No
new identifiers.

### 2026-08-24 — §5 "Related work and positioning" added to every project doc

> ok, do we have a clear section for each project that is "related work" already?

> yes, lets do the "Related work and positioning" because thats what I'll be working on next after we finish ingestion.  can you please make that change across all of the proposed projects?

All 22 docs in [potential-projs/](potential-projs/) gained a `## 5. Related work
and positioning` section at the end (scaffold: purpose note + raw-material
pointers; §4 stays the dated log, new entries insert before the §5 header).
[potential-projs/text-latent-code-autoencoder.md](potential-projs/text-latent-code-autoencoder.md)
additionally carries a starting inventory (novelty-check corpus + anchor set +
resubmission obligations). Convention documented in the README. This is Danielle's
next work surface after ingestion.

### 2026-08-24 — §5 filled from repository records via subagent fan-out

> can you use subagents to pull out existing related work from the rest of the pages, notes etc for the existing projects and fill in the section 5 as best as possible?

Executed as a Claude Workflow: eight read-only Opus extractors (Explore-type), one
per project cluster, under the contract at
`~/drotherm/data/.claude/datadec/2026-08-24/1321-section5-fill-contract.md`
(repo-sourced citations only; no positioning claims; agent-generated records marked
unverified in-line). All 21 scaffolded §5 sections in
[potential-projs/](potential-projs/) now carry a raw-material map and starting
inventory; positioning remains unwritten (Danielle's next work item). TLC's §5 was
already populated and untouched.

### 2026-08-24 — Notion lit-review intake, page 4 (Gemini) + the §5-update convention

> great! so from now on we should fill in the general locations + the RW sections.
>
> next: https://app.notion.com/p/Lit-Review-2fdde135cd1f8010b843eddc15898351

Convention adopted: intake now updates both the usual routing targets and the
relevant §5 "Related work and positioning" sections (raw-material maps +
inventories stay current). Routed: verbatim copy in the batch bundle; TLC §4 entry
(inserted above the §5 header) + §5 inventory updates (fourth check; LBM/GRPO/SPAE/
LINT/CodeCloak/CyclePrompt/RLPrompt added to the anchor set); nl-bottleneck entry +
LBM table-row update (two-check headline match); litreview plan row A (LBM
parenthetical, RLPrompt ID, SPAE); eight ledger rows. No new identifiers.

### 2026-08-24 — Notion lit-review intake, page 5 (Perplexity)

> great, next this one: https://app.notion.com/p/Lit-Review-2fdde135cd1f808eabd7fd9b74efdc11

Identity resolution: this page is the previously anonymous 2026-08-22 "second
novelty check" — Perplexity-generated, 2026-02-04. Feb-2026 set = five checks,
five verdicts. Routed: verbatim copy in the bundle; identity addendum on the
2026-08-22 entry + new dated entry in
[topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md);
OverLang provenance note; LBM attribution corrected (plan row A + table row); TLC
§4 entry + §5 corpus update (five checks; the dual-prompt dismissal to engage);
four ledger rows incl. the 2406.01989 ID-conflict flag. No new identifiers.

### 2026-08-24 — Notion lit-review intake, page 6 (Undermind, 14 sub-pages)

> ok, this one has a LOT of sub-pages, can you pull each of the subpages too please? https://app.notion.com/p/Lit-Review-2fdde135cd1f80d697d7de94ef4a8b53

All 14 sub-pages pulled; verbatim multi-file packet
`undermind-prompt-compression-review/` in the batch bundle (main + four dimension
pages + four timeline pages + trends/contributors/recommendations/summary-table/
foundational/adjacent). Routed: dated entries in
[topics/reference/prompt-compression-and-optimization-literature.md](topics/reference/prompt-compression-and-optimization-literature.md)
(the 21-paper map) and
[topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md)
(verdict + the Undermind-vs-Perplexity contrast pair); TLC §4 entry + §5 corpus
update (six checks); three ledger rows. The ~60 per-paper Notion pages (shared
paper database) were not pulled. No new identifiers.

### 2026-08-24 — summary-table intake begins (ChatGPT table)

> ok great! now lets start pulling in the tables.
>
> https://app.notion.com/p/ChatGPT-Lit-Review-Summary-Table-2fcde135cd1f8064bc7ce06ccae10988

NBLM (NotebookLM) table layer identified: per-review six-column distillations, each
with a Source Note back to its review and a shared downstream "NL Latents Lit
Review Aggregate Summary". ChatGPT table (8 rows, no new papers) preserved in the
bundle's `summary-tables/`; table layer introduced in
[topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md)
and TLC §4; §5 raw-material list gained the tables pointer. No ledger rows (no new
identifiers).

### 2026-08-24 — summary-table intake, Claude table

> next: https://app.notion.com/p/Claude-Lit-Review-Summary-Table-2fdde135cd1f80879b75e5490b71facf?v=2fcde135cd1f8087ae97000c181d4a2e

15 rows, no new papers; preserved as `summary-tables/claude-summary-table.md` in the
bundle. Second schema (gaps + tiered overlap %) noted in the table-layer entry of
[topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md)
as the better §5 template; PlanSearch "frozen CodeLlama" distillation-drift caveat
recorded. No ledger rows.

### 2026-08-24 — summary-table intake, Consensus table

> next: https://app.notion.com/p/Consensus-Lit-Review-Summary-Table-2fdde135cd1f8016b7d5ce46a8b2bec4?v=2fcde135cd1f8087ae97000c181d4a2e&pvs=25

8 rows, no new papers; preserved as `summary-tables/consensus-summary-table.md`.
Third schema (per-paper prior-art-justification gap sentences); covers only the
closest-matches tier. Table-layer entry extended in
[topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md).
No ledger rows.

### 2026-08-24 — summary-table intake, Gemini table

> next: https://app.notion.com/p/Gemini-Lit-Review-Summary-Table-2fdde135cd1f80868b03ed296a6e81c2?v=2fcde135cd1f8087ae97000c181d4a2e

7 rows, no new papers; preserved as `summary-tables/gemini-summary-table.md`.
Fourth schema (component-matrix style); drops LINT/Zip2Zip/ReFIne from source.
Table-layer entry extended. No ledger rows.

### 2026-08-24 — summary-table intake, Perplexity table

> next: https://app.notion.com/p/Perplexity-Lit-Review-Summary-Table-2fdde135cd1f80cf9cffff9a5d41c73b?v=2fcde135cd1f8087ae97000c181d4a2e

4 rows only, no new papers; preserved as
`summary-tables/perplexity-summary-table.md`. OverLang confirmed as an
Agents4Agents blog PDF (row note updated); Gilbert row repeats the 2406.01989 ID
conflict and conflates with TBM (evidence added to the existing flag). No ledger
rows.

### 2026-08-24 — summary-table intake, "Undermind" table (content mismatch)

> https://app.notion.com/p/Undermind-Lit-Review-Summary-Table-2fdde135cd1f809e8006d712c35e325e?v=2fcde135cd1f8087ae97000c181d4a2e

8 rows; preserved as `summary-tables/undermind-summary-table.md`. Flagged: content
is the Dec-2025 SciSpace six-component rubric's scored list (Nano-Capsulator 4/6,
EPiC 3.5/6, ...), not a distillation of the Undermind prompt-compression review its
Source Note points at. All papers already tracked; no ledger rows. Schema (score +
components + gaps) noted as the most §5-useful single table.

### 2026-08-24 — summary-table intake, SciSpace table

> next: https://app.notion.com/p/SciSpace-Lit-Review-Summary-Table-2fdde135cd1f802ba656c26d12197e35?v=2fcde135cd1f8087ae97000c181d4a2e

18 rows, 12-column scored schema; preserved as
`summary-tables/scispace-summary-table.md`. GPT-C = 4.5/6 top-scored closest match
(applied-to-programs ✅) — elevated in TLC §5's anchor set; score drift vs. the
Dec-2025-rubric table recorded; SAPS (2018, no ID) new → one ledger row. Source
Note reveals an unpulled SciSpace Lit Review page
(`2fdde135-cd1f-804a-b0d2-e05f44ed95c9`).

### 2026-08-24 — the SciSpace review page (+ Prior Work Supplemental sub-page)

> the scispace page: https://app.notion.com/p/Lit-Review-2fdde135cd1f804ab0d2e05f44ed95c9?v=2fcde135cd1f8087ae97000c181d4a2e&t=3c6de135cd1f8038ae8000a9369dcd7a

Identity: the Notion copy of the Dec-2025 SciSpace verdict (already intaken
2026-08-22 from disk); new = its Prior Work Supplemental sub-page (search strategy
+ per-paper checklists), both preserved in the bundle. Settles both scored tables'
derivation. PCRL = 2308.08758 and SAPS = 1810.09717 recovered (ledger + plan row A
updated); MIPS/APICoder/SAMMO added as no-ID rows. TLC §4/§5 updated.

### 2026-08-24 — the Aggregate Summary (chain endpoint)

> great! here's the top level merged table: https://app.notion.com/p/NL-Latents-Lit-Review-Aggregate-Summary-2fdde135cd1f80469c7ee9293dcbfedc

The Feb-3/4 session's final artifact: all seven tables merged, relevance-sorted,
source-attributed; no downstream. Preserved as
`nl-latents-aggregate-summary.md` in the bundle; capstone entries in
[topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md)
and TLC §4; §5 raw-material bullet upgraded (the aggregate = the closest existing
thing to a completed §5 comparison table, with its flaws listed). Chain fully
pulled: 7 reviews → 7 tables → aggregate. No new papers; no ledger rows.

### 2026-08-24 — two-tier related-work restructure (recall corpora + curated §5)

> so I'm in the process of cleaning the notion paper db data with the cheap fast notion agents, so lets hold off until then for pulling in more info or grabbing PDFs or verifying.  but I would love to do a pass now on cleaning all of the section 5 sets based on what we have so far.  I want the highest recall version of the related works section for each project to be clearly enumerated somewhere, but I'm open to the idea of having a separate document per proposed project that enumerates the possibly relevatn papers along with why they might be relevant and keeping the shorter list you propose in teh doc you described.  what do you think?

Adopted as the two-tier convention: per-project high-recall corpora in
[potential-projs/related-work/](potential-projs/related-work/) (22 files, one per
project; TLC's is the largest at ~100+ items) + §5 trimmed to curated cores (5–12
load-bearing items with roles, each ending in a pointer to its recall corpus).
Executed via a 9-worker read-only Opus fan-out (phase 2 of the contract at
`~/drotherm/data/.claude/datadec/2026-08-24/1321-section5-fill-contract.md`);
repo-sourced only, no positioning claims, unverified caveats carried. TLC's
hand-curated §5 kept, pointer appended. Notion pulls / PDF fetches / verification
all held pending Danielle's paper-DB cleanup.

### 2026-08-24 — submission-rule annotation + minigrid placeholder

> ok, I already flipped the ingest flags, update the submission rule annotations and just make a placeholder doc for the whetstone-envs minigrid spec out but not actually spec it out yet

Routed: dated annotation under module-or-kill in
[topics/reference/project-approach-principles.md](topics/reference/project-approach-principles.md)
(submit when results are true + claims calibrated to verification; preparation
target = framing settled before the push; permissible-vs-strategic split);
placeholder-only staging doc
[topics/staging/whetstone-minigrid-env.md](topics/staging/whetstone-minigrid-env.md)
(+ staging table row) with an explicit do-not-expand-until-she-initiates note.
Notion Ingested flags: Danielle flipped them herself.

### 2026-08-24 — prompt-interpretation principle adopted

> you can also update the prompt-interpretation principle in the project-approach-principles.md doc

Dated standing-principle entry added to
[topics/reference/project-approach-principles.md](topics/reference/project-approach-principles.md):
novelty verdicts are interpretation samples (her note verbatim; six-checks
evidence; how-to-apply = multi-platform commissioning, union-for-recall,
spread-maps-the-readings, never single-verdict-settled).

### 2026-08-24 — April-2026 reflection notes intaken

> great, now, can you  ingest this please: https://app.notion.com/p/Reflection-on-NBLM-of-Code-Comp-NL-Latents-3c6de135cd1f80678d98de65ef823bed?t=3c6de135cd1f80aaa7e900a9e85dc5b1

Her own April-2026 working notes (raw capture 2026-04-17; thematic re-sort created
tonight in the Notion cleanup) — first-hand, Danielle-supplied provenance. Verbatim
in the bundle. Routed: TLC §4 entry + §5 anchor-tier bullet (PartialOrderEval,
cross-model transfer trio, LM-CC, NL-in-the-Middle endorsement); TLC recall-corpus
dated section (all new items); dated entries in code-benchmarks-landscape
(task-family taxonomy + evaluation confounds), code-compression-literature
(black-box/abstractive self-labeling, LongCodeZip/TRAAC/ShortenDoc/LM-CC), and
prompt-optimization-landscape (transfer trio); twelve no-ID ledger rows
(Danielle-supplied; IDs to come from the paper DB). New timeline fact: the
direction stayed active through mid-April 2026.

### 2026-08-24 — code-datasets lineage/overlap note intaken

> ok, next: https://app.notion.com/p/Code-Datasets-Lineage-Overlap-and-Component-Analysis-33dde135cd1f804b93e2d120e8f0501f

Her curated note (ChatGPT-sourced). Verbatim in the bundle. Routed: dated entry in
[topics/reference/code-benchmarks-landscape.md](topics/reference/code-benchmarks-landscape.md)
(three lines of work; per-benchmark exposure numbers; clone-depth × granularity ×
representation backbone; her 6-step dedup pipeline); TLC recall-corpus line
(dataset-rehab machinery); 2501.10711 title-pairing flag added to its ledger row.
All five papers already ledgered — no new rows.

### 2026-08-24 — pooled+dedup benchmark placeholder

> yes, lets make a placeholder for the benchmark as a byproduct

Placeholder-only staging doc
[topics/staging/pooled-dedup-code-benchmark.md](topics/staging/pooled-dedup-code-benchmark.md)
(+ staging table row): line-of-work-3 seed from the lineage/overlap note, with her
6-step pipeline as the half-spec, ContextBench as precedent, and an explicit
benchmark-as-byproduct constraint for spec time. Do-not-expand-until-she-initiates
note included.

### 2026-08-24 — ingest link index added as a standing step

> ok great, lets add another step to this process, lets add an index of links ->
> ingest topics somewhere so we can look back if we want to in the future.

Created [ingest-link-index.md](ingest-link-index.md): one line per link-based ingest
(source link, what it is, routing targets), grouped by date. Backfilled with the full
2026-08-24 Notion batch (18 pages) and the 2026-08-22 MAQA Next Steps page. Standing
intake step from now on: every link-based ingest adds its row there in the same
commit.

### 2026-08-24 — source link for the HumanEval-ecosystem/overlap conversation

> link: https://chatgpt.com/c/69d807bb-7c84-8333-9a18-c182475985ca

(Followed by the full two-turn conversation text; her two prompts are already logged
verbatim in the 2026-08-22 entry "HumanEval's derivative ecosystem; cross-benchmark
overlap and dedup (two turns)".) Provenance completion, not a re-ingest: the content
was fully routed on 2026-08-22 (landscape entry + TLC §4 pointer + 21 ledger rows).
New today: verbatim capture of both responses in the 2026-08-24 bundle
(`chatgpt-humaneval-ecosystem-overlap-convo.md`); the conversation dated 2026-04-09
via its ID timestamp (inferred; eight days before the April reflection notes, so the
dedup thread was live by early April); identity established — turn 2 is the source of
her curated lineage/overlap Notion note (the 6-step pipeline refines its
multi-layer-graph proposal). Landscape section header dated; lineage-note entry
cross-referenced; ingest-link-index row added.

### 2026-08-24 — HumanEval-family tooling/leaderboards conversation ingested

> link: https://chatgpt.com/c/69d7f822-f760-832e-9bde-06213ac73301

> Is there a collection of human eval datasets and results somewhere, even better if
> they are provided along with codebase and comparisons across different recent
> models?

New ingest (unlike the previous link, this one had not been routed before). One-turn
ChatGPT conversation dated 2026-04-09 19:04 UTC via its ID timestamp (inferred) —
same evening as the ecosystem/overlap conversation, about an hour earlier. Verbatim
in the bundle (`chatgpt-humaneval-tooling-leaderboards-convo.md`). Routed: new dated
entry in
[topics/reference/code-benchmarks-landscape.md](topics/reference/code-benchmarks-landscape.md)
(tooling map: BigCode Evaluation Harness — the Feb-paper HumanEvalExplain harness —
EvalPlus leaderboard, Big Code Models Leaderboard with scriptable
`code_eval_board.csv`, Awesome-Code-Benchmark discovery index, BigCodeBench with
pre-generated model samples; REval = new name, link-list mention only). No ledger
rows — resources, not citations. Ingest-link-index row added.

### 2026-08-24 — April Perplexity thread ingested (task suites, composition, compression metrics)

> link: https://www.perplexity.ai/computer/tasks/e7748d8a-10b8-4ccf-bc34-f83c840ba770

Three turns, verbatim in the bundle
(`perplexity-humaneval-composition-compression-convo.md`). Turn 1's prompt is the
2026-04-09 ChatGPT ecosystem question verbatim (already logged in the 2026-08-22
entry) run on a second platform; undated (Perplexity task IDs carry no timestamp),
early April inferred. Her two new prompts:

> This is incredibly helpful. So I guess one of my questions is around the other
> data sets that you listed and how many of them actually have a test set that is
> considered sufficient to actually judge output quality. And then also, how many
> of them are long chunks of code? But perhaps not super long. I'm not looking for
> a repo. But class level or a bit higher than that. I am interested in testing
> out approaches to use language models to compress code. And while it is
> definitely possible, with human eval, the functions are also short that are
> pretty plausible way to compress the code is actually just to remove all of the
> indentation and you get what I think is a pretty short representation if we're
> staying in the space of tokens that language models are trained on. Because
> we're doing black box. API. Models. And so there's no additional training. So
> yes. What are some examples of code data sets that would match those criteria?

> Very interesting. So then, another question I would have is if there are a bunch
> of different data sets that have different lengths of code with different
> focuses, and they have test cases that are rigorous, then has there been work on
> composing these things because if you know the input and output structure in
> expectations, then composing would probably create a data distribution that is
> outside the norm. But it would also probably create much larger and more complex
> functions that you still would be able to test to some extent. And I think
> especially for setting around looking at code compression, then it's
> particularly interesting because I don't need to be able to describe what I want
> the agent to do. I just need to be able to pass in a chunk of code and see what
> else it provides that should be sufficient for reconstruction.

Routed: full three-turn entry in
[topics/reference/code-benchmarks-landscape.md](topics/reference/code-benchmarks-landscape.md)
(Perplexity's ecosystem map with new IDs; the test-rigor × length breakdown with
ClassEval as clearest compression testbed; the composition turn — "genuinely
unexplored," DynaCode closest, "the code is the specification"); metric/theory
entry in
[topics/reference/code-compression-literature.md](topics/reference/code-compression-literature.md)
(Gilbert ERE/SRE + ~5×, Fundamental Limits of Prompt Compression, Extending
Context Window via Semantic Compression, her trivial-baseline articulation); TLC
§4 dated entry + §5 anchor-set Gilbert line enriched with ERE/SRE + recall-corpus
section (ClassEval, CoderEval, APPS, DynaCode, PseudoEval, theory anchors); 34
new ledger rows + 4 updates (ENAMEL 2406.06647 and EffiBench 2402.02037 ID fills;
2412.01526 triple partially untangled via HumanEvalNext 2503.05860; Gilbert
ERE/SRE note). Ingest-link-index row added.

### 2026-08-24 — LLM-as-optimizer / toy-RL Perplexity task ingested

> link: https://www.perplexity.ai/computer/tasks/ddb5d88b-0e76-4664-bdd5-c61b94d23e17

> I'm really interested in the idea of using LLMs like API black box LLMs as
> optimizers for a range of problems. And I know that there are some papers that
> are well known for having titles like "LLM as an Optimizer," and that there are
> some works that, for example, use two LLMs to make an autoencoder, things like
> this. So I understand the idea of using LLMs as components like black box LLMs
> as components of systems that mirror previously considered standard machine
> learning approaches isn't new, but I also feel like I haven't really seen—like
> when I've looked into these works, I haven't necessarily seen things where it's
> like, "Ah, yes, I do think this is what I would call using an LLM as an
> optimizer," in a broad general sense. So I'm curious what some of the strongest
> works in that direction are, kind of regardless of the application area. Just
> things that make you say, "Ah, yes, this is in fact using an LLM as an
> optimizer," or using an LLM as a component in a classical machine learning
> system in kind of a novel way. And then I'm also curious about what exists out
> there on using LLMs to solve different types of classic optimization problems
> or even classic RL problems. I'm still thinking toy problems, but in settings
> where we're not designing an actual—in settings where we're doing minimal
> tuning, to translate the state and action space into language and then to
> translate back to the classic state and action space. I'm just curious what
> exists out there for this. So things like cart pole or, I don't know, minigrid,
> pushing a ball up a hill. I feel like there are a bunch of classical RL
> problems, at least, that I know of that are what you always start with. And so
> things like that.

One-turn Perplexity task, undated (her dual-LLM-autoencoder framing places it
after TLC ideation; the response cites 2025 work). Verbatim in the bundle
(`perplexity-llm-as-optimizer-toy-rl-convo.md`). Routed: full taxonomy entry in
[topics/reference/prompt-optimization-landscape.md](topics/reference/prompt-optimization-landscape.md)
(true optimizers OPRO/FunSearch+AlphaEvolve/Eureka/LLAMBO; component slots
ReEvo/LLM-SR/DICL; Matryoshka + ACING as the closest dual-LLM structures; toy-RL
survey with Atari-GPT's negative result and the outer-loop-optimizer vs
inner-loop-controller synthesis); TLC recall-corpus lines (Matryoshka, ACING,
Eureka's nesting, the FunSearch program-space principle); a prior-art pointer
line in the minigrid staging placeholder (pointer only — spec still deferred);
11 no-ID ledger rows (the response supplies no arXiv IDs). Ingest-link-index row
added.

### 2026-08-24 — NotebookLM Continual Learning notebook ingested

> ok, this ones a bit different, we're going to include the outputs from a
> Notebook LM notebook on a given topic (Continual Learning in this case).
>
> link: https://notebook.google.com/notebook/040273e8-020f-4926-b7ab-6af42dee3505

(Followed by three artifacts: the data table, the source list, the "Original
Synthesis Report," and the "New Deep Research Report" — all verbatim in the
bundle, `nblm-continual-learning-notebook.md`.) First continual-learning-side
intake of the day. Provenance: eleven sources, mainly CoLLAs 2022/23 talks
(Sutton, Lyle, Van Roy, Bing Liu, Rish, Aljundi, Larochelle) plus a Harrison
learned-optimization talk and **her own Roam daily-pages export (2024-10-02)**
as source [2]. Reliability flagged: NotebookLM's own caveat, plus the first
synthesis report is transcript-garbled (mis-heard names — "Ashton Adams" = Ash &
Adams, "Socar" plausibly Sokar et al., "itbid" = IDBD, etc.; identifications
inferred and marked). Routed: main cluster-by-cluster entry in
[topics/reference/plasticity.md](topics/reference/plasticity.md) (talk-level
detail beyond the existing paper anchors, incl. Van Roy's
L2-toward-init-beats-CBP baseline claim and the representation-vs-head
forgetting asymmetry convergent with the reset/frozen-body staging topics);
drift-taxonomy + stream-evaluation entry in
[topics/reference/nonstationarity-accounting.md](topics/reference/nonstationarity-accounting.md);
15 no-ID ledger rows (the notebook supplies no arXiv IDs anywhere).
Ingest-link-index row added under a new NotebookLM group.

### 2026-08-24 — NotebookLM LLM-evaluation notebook ingested

> link: https://notebook.google.com/notebook/2f70b7f0-4156-4e90-ab47-a34dac85ddc1

(Followed by the data table, source list, and two synthesis reports — verbatim in
the bundle, `nblm-llm-evaluation-notebook.md`.) Second NotebookLM notebook of the
day: LLM evaluation / meta-evaluation over 16 paper sources, three of which are
the program's own foundations (DataDecide, Signal-and-Noise, model ladders —
those rows restate canonical repo content). New material routed: the
2024–2026 meta-evaluation cluster in
[topics/reference/irt-literature.md](topics/reference/irt-literature.md) —
PSN-IRT 2505.15055 (neural 4PL benchmark diagnosis; nearest new neighbor to
IRT-as-instrument), Federiakin 2501.17200 (psychometric leaderboard reanalysis),
EffiEval 2508.09662, Benchmark² (CAD/DS/CBRC/BQS meta-metrics + the ARC
alignment-vs-discriminability paradox), ONEBench 2412.06745, ResampledBench
2504.09979, SparseEval 2602.07909, amortized calibration / difficulty-targeted
question generation (Truong), EvaLearn, ADeLe, gp-IRT detail; methodology-side
items in
[topics/reference/evaluation-methodology-literature.md](topics/reference/evaluation-methodology-literature.md)
(Zeng's fixed-scale perplexity paradox — PPL-CLM worse than chance for SFT
checkpoint selection, dual-title flag; Owen predictability; RUPBench; SciRIFF).
irt-reanalysis §5 got a new curated bullet (the compression line's current
frontier) + a recall-corpus section; 14 ledger rows (6 with notebook-supplied
IDs). Flagged: table-vs-report ARC discriminability inconsistency (0.11 vs
0.03). Ingest-link-index row added.

### 2026-08-24 — NotebookLM NL-latents/TLC notebook ingested

> link: https://notebook.google.com/notebook/8d1031d0-06f1-4437-af7e-0cd819fb9695

(Followed by the 24-source enumerated synthesis and the library-induction report
— verbatim in the bundle, `nblm-nl-latents-tlc-notebook.md`.) Third NotebookLM
notebook of the day, and the closest to home: the source set is her curated TLC /
NL-latents paper collection — all but three names already ledgered with IDs.
Routed: dated entry in
[topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md)
(the aggregate synthesis — discrete-NL-bottleneck + frozen-decoder +
gradient-free optimization as the shared design; idea-space search beats
code/token-space; frozen LLMs as "universal decompressors" — flagged as
§5-grade raw material but partly circular since it synthesizes her own corpus;
LILO/Stitch/AutoDoc depth incl. the code-obfuscation finding as an
NL-necessity-adjacent datum; concrete RL4QE detail); TLC §4 dated entry;
recall-corpus lines (LAPS, O'Connor & Andreas, Self-consistency, LILO/Stitch
enrichment); 3 no-ID ledger rows. Ingest-link-index row added.

### 2026-08-24 — NotebookLM pretraining-dynamics notebook ingested

> link: https://notebook.google.com/notebook/611a0dd1-628d-4b4b-8742-bc6d12767de0

(Followed by the 11-source synthesis and the MPL/PTQ report — verbatim in the
bundle, `nblm-pretraining-dynamics-notebook.md`.) Fourth NotebookLM notebook of
the day, on the pretraining side — home turf for the schedule program. MPL,
river valley, and CompleteP-as-name were already on record; the rest is new.
Routed: main entry in
[topics/reference/schedules-and-annealing-literature.md](topics/reference/schedules-and-annealing-literature.md)
(the PTQ finding — quantization error spikes exactly at LR decay, stable phase
PTQ-flat, souping/LAWA mitigate, a new consequence-of-anneal axis; MPL report
detail incl. the sqrt-cube decay η_max·(1−τ)^1.5 beating cosine and WSD; CPT
scaling law; mid-training taxonomy; IB-at-LLM-scale as a candidate trajectory
statistic; LoRA-LR-matters as a tuning-equity datum); HP-scaling cluster in
[topics/reference/parametrization-and-hp-transfer.md](topics/reference/parametrization-and-hp-transfer.md)
(CompleteP detail fill; Power Lines — AdamW timescale ~ D/N power law, batch
size scales with D alone; Step Law — strictly convex (LR,BS) landscape; the
convergent BS-is-a-D-story claim); the LM-plasticity pair in
[topics/reference/plasticity.md](topics/reference/plasticity.md) (catastrophic
overtraining + weight-decay-improves-plasticity — partial answer to her
standing where-did-plasticity-go interest, now at HP-selection level);
wsd-suite recall-corpus lines (sqrt-cube arm candidate, PTQ axis, Power
Lines/Step Law as fixes-by-formula for sweep design); 10 no-ID ledger rows (the
notebook supplies no IDs). Ingest-link-index row added.

### 2026-08-24 — NotebookLM reasoning-mechanisms notebook ingested

> link: https://notebook.google.com/notebook/19a583a2-ee35-40b7-bc03-cb0710e594bd

(Followed by the four-source synthesis and the architectures report — verbatim
in the bundle, `nblm-reasoning-mechanisms-notebook.md`.) Fifth NotebookLM
notebook of the day; no arXiv IDs supplied. Routed: main entry in
[topics/reference/generalization-and-ood-literature.md](topics/reference/generalization-and-ood-literature.md)
(encoder-vs-decoder FOL robustness with the NL-vs-NNL lexical ablation — a
design worth remembering; curvature-similarity probes; the OCR
generalization/hallucination duality via factorized matrices + nuclear-norm
implicit bias); training-paradigm half in
[topics/reference/pretraining-to-posttraining.md](topics/reference/pretraining-to-posttraining.md)
(RL preserves balanced atomic-skill profiles while SFT induces jagged
over-specialization — a skill-resolved instance of the elicitation question;
procedural-knowledge influence functions — code/math documents drive reasoning
queries, answer documents drive factual ones — recipe-composition relevance);
4 ledger rows, one with a Claude-added ID (2411.12580, inferred from the title,
plausibly Ruis et al.). Ingest-link-index row added.

### 2026-08-24 — NotebookLM world-models notebook ingested

> link: https://notebook.google.com/notebook/ecbd81fa-0e71-42bf-b86d-b1b0f3d32478

(Followed by the two-source synthesis and the convergence report — verbatim in
the bundle, `nblm-world-models-notebook.md`.) Sixth NotebookLM notebook of the
day. No existing accumulator covered world models, so this founded
[topics/reference/world-models-literature.md](topics/reference/world-models-literature.md)
(+ topics README row): "From Word to World" (SFT'd LLMs as reliable text-based
world models; structured environments saturate ~20K trajectories vs
non-saturating open-ended ones; irreversible-action gating, synthetic
trajectories, RL warm-starting; mixed-agent trajectories beat expert-only —
failure-state coverage matters) and "General agents contain world models"
(Richens — regret-bounded multi-step agents provably encode transition
probabilities, extractable from policy behavior alone; myopic agents don't; ID
2506.01622 Claude-added, inferred). Positioned as the model-based flank of the
LLM-in-classic-RL thread (DICL cross-ref) and minigrid spec-out background.
2 ledger rows. Ingest-link-index row added.

### 2026-08-24 — NotebookLM LLM-driven-optimization notebook ingested (link pending)

> link:

(Empty link line — the notebook link was not supplied; the ingest-link-index row
carries a fill-in note. Followed by the 25-source synthesis and the
hybrid-systems report — verbatim in the bundle,
`nblm-llm-optimization-notebook.md`.) Seventh NotebookLM notebook of the day,
extending the LLM-as-optimizer thread. Routed: main entry in
[topics/reference/prompt-optimization-landscape.md](topics/reference/prompt-optimization-landscape.md)
adding the standalone-vs-hybrid axis to the earlier outer-loop-vs-inner-loop
conclusion: pure LLM optimizers fail on state tracking / density estimation
(re-entering known-infeasible regions despite in-context history) and small
models lose to few-shot CoT (Revisiting OPRO, already on record); hybrids win —
Centaur shares CMA-ES state with the LLM and tops the autoresearch HPO
benchmark ("classical methods find the path; LLMs suggest where to look next");
evolutionary search beats RL updates for LLM idea search (2601.14525, mode
collapse); HAPO as a budget-frugality datum; MCTS hybrids; agent-scaffold
optimization; the RL-adjacent trio (Language Feedback Models,
sequential-decision-making reward-vs-action modeling, planning survey) tagged
for the minigrid context. AlphaEvolve's earlier no-ID ledger row filled with
2506.13131; 22 new ledger rows (7 with notebook-supplied IDs).

### 2026-08-24 — LLM-optimization notebook link supplied; tokenization notebook ingested

> ahh, the previous link was: https://notebook.google.com/notebook/5439c587-5feb-432d-a779-a23c454ebf25
>
> link: https://notebook.google.com/notebook/1d9de7a7-815a-4a65-a933-1abde72c1ff4

Link fill: the LLM-driven-optimization notebook's pending link recorded in the
ingest-link-index row, bundle header, and INDEX entry. Then the eighth
NotebookLM notebook ingested (verbatim in the bundle,
`nblm-tokenization-vocabulary-notebook.md`): tokenization / vocabulary scaling
/ cross-tokenizer transfer. Routed: Hayou's Large Vocabulary Parametrization +
√d-rule (µP's embedding rules break when vocabulary scales with width;
LR_emb/LR_hidden ≈ Θ(√d); flagged as directly relevant to
embedding-reset-dynamics and µP-based sweep design) in
[topics/reference/parametrization-and-hp-transfer.md](topics/reference/parametrization-and-hp-transfer.md);
the tokenizer/vocabulary flank in
[topics/reference/reinit-and-transfer-literature.md](topics/reference/reinit-and-transfer-literature.md)
(byte-level cross-tokenizer distillation with its IFEval collapse — the
contemporary answer to the tokenizer-change-cost question that record tracks;
BPE-dropout one-checkpoint segmentation unification; mT5-vs-ByT5 morphology
probing with the layer-depth and irregularity×data-share results; Llama 3
tokenizer economics; VocabTailor). Two off-topic notebook sources (TinyOS
history, a commercial market report) flagged in the bundle header, not routed.
8 ledger rows (Sennrich 1508.07909 and mT5 2010.11934 as Claude-added canonical
IDs). Ingest-link-index row added.

### 2026-08-24 — NotebookLM AI4SE notebook ingested

> link: https://notebook.google.com/notebook/04e09363-31e7-4bf5-acb3-a4fddbcc795e

(Followed by the eight-source synthesis and the AI4SE report — verbatim in the
bundle, `nblm-ai4se-code-notebook.md`.) Ninth NotebookLM notebook of the day;
no arXiv IDs supplied. Two known names got full detail: **ShortenDoc** (subword-
level entropy-sorted compression, 25–40% preserving/improving Pass@1, general
compressors fail past ~10% on code, and the **method-name dependency** —
identifier NL and docstring NL trade off, a direct NL-necessity/frontier datum)
routed to
[topics/reference/code-compression-literature.md](topics/reference/code-compression-literature.md)
along with the summarization-metric findings (BLEU/METEOR/BERTScore
anti-correlate with human ratings; GPT-4-as-judge closest — a caution for
d-quality measurement); **Prochemy** (mutation→execution-scored
evaluation→selection; works on o1-mini) routed to
[topics/reference/prompt-optimization-landscape.md](topics/reference/prompt-optimization-landscape.md)
with the new material: PromptCS (soft-prompt contrast case), the **Fluorine
feedback paradox** (raw counterexample feedback confuses; restart beats
conversational repair 7–21%; localized execution state helps — TransAgent/
LANTERN contrast), **LANTERN's cross-language repair hypothesis**
(representation-surface choice changes frozen-model capability —
cross-decoder/NL-likeness adjacent), and **RepE correctness probing** (test-free
ranking from an extractable internal correctness direction, ~3.75s vs RankEF's
172h). TLC recall corpus +6 lines; ShortenDoc ledger row annotated; 6 new
ledger rows. Ingest-link-index row added.

### 2026-08-24 — NotebookLM intermediate-representations notebook ingested

> link: https://notebook.google.com/notebook/beaf7015-f9ee-43e8-80a2-bd0089c5402d

(Followed by the five-source synthesis and the mechanistic-coherence report —
verbatim in the bundle, `nblm-intermediate-representations-notebook.md`.) Tenth
NotebookLM notebook of the day; all five papers TLC-core; no arXiv IDs
supplied. Headline: **"NL in the Middle" finally has content on the record**
(previously only its name, aggregate 5.5★ score, and her April endorsement):
CoT-with-NL-summary beats both zero-shot and two-step code→IR→code pipelines
(intermediate errors propagate), NL-before-AST ordering matters, and
code-specialized models benefit less. Also: NL-Debugging format detail (Sketch
> Pseudocode > Key Points, capacity-dependent — a capacity×abstraction-length
interaction for the NL-likeness ladder); **RTCE** bijective round-trip
execution benchmark (inversion collapses; name-collision flag vs RTC
2402.08699); **Proof2Silicon/PREFACE** (PPO prompt repair with a Dafny/SMT
verifier as reward over a frozen decoder); and **the Perplexity Paradox**
(code keywords carry ~79× perplexity and survive perplexity-based compressors
while math numbers get pruned; signature re-injection recovers +34pp — the
signature is the load-bearing NL, convergent with ShortenDoc's method-name
dependency; TAAC). Routed to
[topics/reference/nl-bottleneck-prior-art.md](topics/reference/nl-bottleneck-prior-art.md)
(main) and
[topics/reference/code-compression-literature.md](topics/reference/code-compression-literature.md)
(Perplexity Paradox); TLC §4 dated entry + §5 NL-in-the-Middle clause enriched
+ 5 recall-corpus lines; 4 ledger rows + NL-Debugging row note.
Ingest-link-index row added.

### 2026-08-24 — NotebookLM refactoring/self-improvement notebook ingested

> link: https://notebook.google.com/notebook/a31f1767-9de4-4842-9754-76fe6eb7d41c

(Followed by the 25-source enumeration and the agentic-refactoring report —
verbatim in the bundle, `nblm-refactoring-selfimprovement-notebook.md`.)
Eleventh NotebookLM notebook of the day, two clusters + one outlier. Routed:
the **agentic refactoring cluster** to
[topics/reference/code-benchmarks-landscape.md](topics/reference/code-benchmarks-landscape.md)
(SWE-Refactor detail filled for its bare-name ledger row; RefactorBench's
22%-vs-87% stateful-reasoning gap; CodeTaste's instructed-vs-open collapse;
MANTRA's Reviewer-agent result; the agents-do-janitorial-not-architectural
empirics incl. tangled commits and the readability-commits-reduce-
maintainability finding — background for agentic-development/defensive-design
interests); the **self-improvement/preference-optimization cluster** to
[topics/reference/pretraining-to-posttraining.md](topics/reference/pretraining-to-posttraining.md)
as a compact inventory (DNO, iterative-DPO's amplify-don't-instill finding —
convergent with today's SFT-vs-RL and mode-collapse threads; SCPO;
Self-Rewarding + its theory; data-autophagy warning); and **Scaling Laws for
Precision** to
[topics/reference/schedules-and-annealing-literature.md](topics/reference/schedules-and-annealing-literature.md)
as the other side of the PTQ dispute (degradation-increases-with-overtraining
vs the PTQ paper's LR-decay attribution — read together at verification; ID
2411.04330 Claude-added). SWE-Refactor row annotated; 21 new ledger rows
(2603.23443, 2508.06026, 2501.00663, 2411.04330 with IDs).
Ingest-link-index row added.

### 2026-08-24 — NotebookLM MoE notebook ingested

> link: https://notebook.google.com/notebook/4223a13e-3dd3-4f46-ab0e-97b464d2151e

(Followed by the 27-source comparative review and the evolution report —
verbatim in the bundle, `nblm-moe-notebook.md`.) Twelfth NotebookLM notebook of
the day, landing on the MoE program. Routed: main entry in
[topics/reference/moe-literature.md](topics/reference/moe-literature.md) —
canon detail (incl. ST-MoE's different-fine-tuning-protocols comparability
datum; OLMoE's from-scratch-vs-upcycled specialization contrast; DeepSeek-V3's
aux-loss-free balancing) plus the previously unrecorded **nine-paper MoE
scaling-law cluster** (Efficiency Leverage's activation-ratio power law;
G_opt≈6.78/S_opt≈0.31 matching DeepSeek-V3.1/Kimi-K2; dense-vs-MoE
transfer with MoE-as-implicit-regularizer — matched-loss-pillar relevant;
memory-optimality; the holistic law's shape-band-widens-with-scale caution for
small proxies; reading-comprehension-favors-density), heterogeneity (HMoE,
HDEE), capacity-aware inference (test-time 7× straggler despite training load
losses — routing-observable relevant), the A4 survey's >99%
expert-representation-collapse figure, and the headline **B2: MoE sparsity
helps memorization monotonically but hurts reasoning at scale, and the
degradation is not recoverable by post-training RL or test-time compute** —
convergent with Mixture of Parrots and load-bearing for the
pretraining→posttraining pillar. Shared recall sections added to all three MoE
project corpora; 24 ledger rows (8 canonical IDs Claude-added, marked).
Ingest-link-index row added.

### 2026-08-24 — post-run hygiene pass + convergent-findings note

> ok great, lets do the bundle index / hygeine pass + the cross cutting synthesis
> note. then can you give me a compaction prompt?
>
> I think the notion db is ready to be read and then updated by our new entries
> and notes + to be pulled into our local versions. And I think the perfect next
> step will be to discuss how to design that process (dynamic workflow, what are
> the target outcomes for notion, I'd love to actually promote the potential
> projects and topic pages there too with a timestamp so we can update as we go
> periodically) after the compaction.

Applied: bundle INDEX.md "Routed to" section now covers the full post-batch link
run (per-artifact routing map) alongside the original Notion-batch routing; new
reference accumulator
[topics/reference/convergent-intake-findings.md](topics/reference/convergent-intake-findings.md)
(+ topics README row) recording the three threads that recurred across
independent 2026-08-24 sources — amplify-don't-instill/mode collapse in cheap
improvement loops, the four downstream costs of over-training, and
representation-surface-changes-frozen-model-capability — each with witnesses,
accumulator pointers, a working statement, program relevance, and the
convergence-≠-verification caveat. STATE CHANGE noted for next session: the
Notion paper DB is ready — the holds are ending; next work item is a design
discussion (not yet execution) for the Notion↔repo sync process: reading the
cleaned DB, pushing our new entries/notes back, pulling into local versions, and
promoting potential-projs + topics pages to Notion with timestamps for periodic
updates.

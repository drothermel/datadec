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

Response routed to: [topics/checkpoint-tomography.md](topics/checkpoint-tomography.md);
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

Routed to: [topics/nonstationarity-accounting.md](topics/nonstationarity-accounting.md);
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

Routed to: reference topics [topics/plasticity.md](topics/plasticity.md) and
[topics/loss-curve-forecasting.md](topics/loss-curve-forecasting.md);
[potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md) §4 and
[potential-projs/irt-reanalysis.md](potential-projs/irt-reanalysis.md) §4 (loss→accuracy
caveat); [topics/nonstationarity-accounting.md](topics/nonstationarity-accounting.md); the
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
[topics/pretraining-to-posttraining.md](topics/pretraining-to-posttraining.md); §4 notes in
[potential-projs/annealed-readouts.md](potential-projs/annealed-readouts.md) (ANN-opt-3),
[potential-projs/wsd-suite.md](potential-projs/wsd-suite.md) (WSD-opt-2),
[potential-projs/token-movement.md](potential-projs/token-movement.md) (TOK-opt-4),
[potential-projs/tiny-scale-measurement.md](potential-projs/tiny-scale-measurement.md);
the potential-projs README program-level notes.


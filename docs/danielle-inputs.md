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

Response routed to: [topics/beyond-datadecide-data-measurement.md](topics/beyond-datadecide-data-measurement.md);
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

Response routed to: [topics/functional-featurization.md](topics/functional-featurization.md)
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
Response routed to: [topics/functional-featurization.md](topics/functional-featurization.md)
— the combined-prompt entry there.

### MoE model releases as "someone else already did the work"

> what if we incorporate moe based model releases in this too for the "someone else already
> did the work" aspect?

Response routed to: [topics/moe-routing-as-data-instrument.md](topics/moe-routing-as-data-instrument.md);
project-specific parts to [potential-projs/trajectory-statistics.md](potential-projs/trajectory-statistics.md) §4,
[potential-projs/token-movement.md](potential-projs/token-movement.md) §4, and
[potential-projs/recipe-featurization.md](potential-projs/recipe-featurization.md) §4.


# Deep Research Session Guidelines

## 1. Report Style

**Executive summary first.** Every report opens with a prose executive summary stating the subject, the scope boundaries (including what was excluded, in a sentence or two), and the report's shape. It stands alone: a reader who reads nothing else comes away with an accurate picture.

**Compact table of contents second.** A nested list mirroring the section and subsection structure. Because a text-to-speech reader will read it aloud, it is as compact as possible: one line per entry, no annotations, no formatting beyond simple indentation. This is the one place a list is required rather than permitted.

**Prose by default, lists when earned, tables never.** The report must read well end-to-end through text-to-speech. Prose reads naturally aloud; lists read acceptably and are permitted for genuinely enumerable content; tables do not work, because their meaning depends on two-dimensional structure that linear speech cannot convey. Content that would naturally be tabular is written as prose or a list. Inline symbols and code fences are avoided unless the content genuinely requires them (a name, a specific API). Subsection headers are used sparingly, each with enough prose beneath it to justify its existence.

## 2. Report Contents

**Every entry answers the same core questions.** What it is, what it takes as input, what it produces, what it is built on or depends on, and its maintenance status with evidence. This schema is the floor for every entry, including peripheral, legacy, or down-weighted ones: depth may shrink, but never below the schema. An entry that cannot be dismissed in two sentences under this rule was never adequately described by two sentences.

**Scope discipline.**

- The report answers the question posed and does not drift into adjacent questions — usage strategy, implementation planning, infrastructure, or "what to do next" — even when those would naturally arise.
- Coverage depth is proportional to an entry's centrality in the subject, subject to the schema floor above.
- Entries are actual instances drawn from the landscape, not hypothetical or self-constructed alternatives introduced to fill perceived gaps. Genuine gaps are stated explicitly rather than papered over.

**Coverage is complete and honest.**

- The full landscape is covered, including less-visible entries a casual survey would miss.
- Multiple instances of a category are each covered on their own terms; the landscape is never represented by its single most prominent member.
- Subsets, specializations, and components of a larger entry are described in their own right, not folded into the parent or omitted because the parent is covered. That one entry dominates another under some weighting of properties does not license skipping the description of either.
- Entries are treated on equal terms based on what they contain, not on packaging, publication venue, or prominence of source.

**Categorization reflects intrinsic properties.**

- Entries are grouped by what they are, what they operate on, and what they produce — properties visible to a reader inspecting the entry itself — not by intended use, assumed role, or the author's judgment of what an entry is "for."
- An entry belongs to a category because it shares the defining property, not because it is commonly discussed alongside other members.
- Cross-references are used when an entry legitimately belongs in multiple categories, rather than forcing it into one.

**Each entry is described in isolation.**

- An entry is described as if it were the only one the reader might use. Descriptions do not state what it feeds, gates, precedes, follows, complements, or is superseded by, and do not assign it a role, tier, or layer in any assumed composition or workflow.
- The only permitted references to other entries are factual identity claims ("X is a plugin for Y") and shared-technology claims ("both are built on Z").

**Description is not prescription.**

- Descriptions state what an entry is and does; they do not state what the reader should do with it. Evaluative language — "best," "recommended," "the right choice," "skip," "avoid" — does not appear.
- Status labels (unmaintained, deprecated, superseded, disputed, historical) are stated as factual properties with the specific evidence warranting them. A status label is never a substitute for description or a justification for a recommendation.

**No invented or unearned judgments.**

- The report introduces no numeric scores, rankings, or ratings of its own invention.
- Difficulty — of setup, use, learning, or adoption — is not assessed anywhere, in numbers or in adjectives. Facts about dependencies, install method, and runtime are reported; judgments layered on those facts ("heavy," "burden," "too slow for," "acceptable") are not.
- Engineering variables a competent user can change (throughput, out-of-the-box performance, reference-implementation quality) are never ranking or evaluation axes.
- Such properties may still be reported factually when they distinguish an entry substantively.
- Properties that are facts are reported. Properties that would require the author's judgment to assign are omitted rather than invented.

## 3. Research Transparency

**Negative space is named.** The report lists entries that were considered and excluded, each with the scope rule that excluded it. Absence of an entry must be distinguishable from ignorance of it; the considered-and-excluded list is what makes the scope boundary auditable and lets the reader catch omissions.

**The search frame is independent of the prompt's framing.** The search covers the landscape the subject actually names, not the landscape suggested by the prompt's incidental examples, the most convenient ecosystem, or the author's home territory. Any narrowing of the search space — by language, ecosystem, era, publication venue — is stated explicitly rather than left implicit.

**Facts are dated.** Landscape facts are volatile. Maintenance status, latest release, and similar claims are stated with their as-of date ("as of June 2026, the latest release is 3.2") rather than asserted as timeless truths.

**Comparative claims are verifiable.** Any claim that entry A covers, contains, or lacks something relative to entry B is accompanied by the underlying specifics — which items, which features, which versions — so the reader can check the claim rather than take it on trust. A comparison whose supporting facts are not stated is omitted.

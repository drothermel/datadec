# Data Repository Consolidation Plan

This document outlines the plan to reorganize data processing repositories for clarity and maintainability.

## Goals

1. Clear separation between general utilities and project-specific code
2. Separate external (AllenAI) data from internal experiment data
3. Parallel structure for related projects (ft-scaling / ft-pred)
4. Clean dependency flow from utilities → domain → projects → application

## Target Architecture

```
UTILITIES (general-purpose, reusable)
├── dr_wandb      → Wandb data downloading ✅ Published v0.2.0
├── dr_hf         → HuggingFace operations ✅ Published v0.1.0 (replaces hf_utils)
├── dr_duck       → DuckDB/MotherDuck utilities ✅ Published v0.1.0
├── dr_frames     → Pandas/DataFrame utilities ✅ Published v0.1.0
└── dr_render     → Table formatting ✅ Published v0.1.0 (renamed from dr_showntell)

EXTERNAL DATA (AllenAI published)
└── datadec       → DataDecide parsing/processing (CLEANUP - remove project-specific code)

YOUR PROJECTS (internal experiments)
├── ft-scaling    → Fine-tuning scaling analysis (NEW - consolidates dd_parsed + dr_ingest)
└── ft-pred       → Fine-tuning prediction (RENAME from ddpred)

APPLICATION (downstream consumer)
└── by-tomorrow-app → Serves visualizations
```

## Dependency Flow

```
         dr_wandb ←──────────────────────────┐
              ↓                              │
           dr_hf                             │
              ↓                              │
         dr_duck                             │
              ↓                              │
        dr_frames                            │
              ↓                              │
        dr_render                            │
              ↓                              │
          datadec ────────────────────────┐  │
              ↓                           ↓  ↓
        ft-scaling ←───────────────── ft-pred
              ↓                           ↓
        by-tomorrow-app ←─────────────────┘
```

## Current State

### Utility Packages - ALL COMPLETE ✅

| Repo | Purpose | Status |
|------|---------|--------|
| `dr_wandb` | Wandb data downloading CLI | ✅ Published v0.2.0 |
| `dr_hf` | HuggingFace utilities | ✅ Published v0.1.0 (replaces hf_utils) |
| `dr_duck` | DuckDB/MotherDuck utilities | ✅ Published v0.1.0 |
| `dr_frames` | Pandas/DataFrame utilities | ✅ Published v0.1.0 |
| `dr_render` | Table formatting | ✅ Published v0.1.0 (renamed from dr_showntell) |

### Dependent Repos - ALL UPDATED ✅

| Repo | New Dependencies | Status |
|------|------------------|--------|
| `dr_ingest` | dr-hf, dr-duck, dr-frames | ✅ Updated, merged |
| `dr_marimo` | dr-frames | ✅ Updated, merged |
| `ml-moe` | dr-frames | ✅ Updated, merged |
| `datadec` | dr-frames[formatting], dr-render | ✅ Updated, merged |

**Key Design Decision:** No re-exports. Callers import directly from utility packages.

### Repos Still Pending

| Repo | Current Purpose | Status |
|------|-----------------|--------|
| `dr_ingest` | Project-specific parsing (after utility extraction) | Needs cleanup or deprecation |
| `ddpred` | DataDecide prediction | Needs rename to ft-pred |
| `datadec` | DataDecide processing | Needs audit for project-specific code |
| `by-tomorrow-app` | Frontend + backend | Wait for other repos |

### HuggingFace Datasets

| Dataset | Current Contents | Target |
|---------|------------------|--------|
| `drotherm/dd_parsed` | Mixed (wandb runs, scaling law data, etc.) | Rename to `drotherm/ft-scaling`? |

## Phases

### Phase 1: Utility Packages ✅ COMPLETE

All utility packages published to PyPI:
- ✅ `dr-duck==0.1.0`
- ✅ `dr-frames==0.1.0`
- ✅ `dr-hf==0.1.0`
- ✅ `dr-wandb==0.2.0`
- ✅ `dr-render==0.1.0`

All dependent repos updated to import from utility packages:
- ✅ dr_ingest
- ✅ dr_marimo
- ✅ ml-moe
- ✅ datadec

### Phase 2: Create `ft-scaling`

New repo for fine-tuning scaling analysis:

- [ ] Create `ft-scaling` repo
- [ ] Migrate from `drotherm/dd_parsed` HuggingFace dataset
- [ ] Migrate project-specific code from `dr_ingest`
- [ ] Migrate any project-specific code from `datadec`
- [ ] Update HuggingFace dataset to `drotherm/ft-scaling`?

### Phase 3: Rename `ddpred` → `ft-pred`

Clean up prediction repo:

- [ ] Audit contents, move AllenAI parsing to `datadec`
- [ ] Rename repo to `ft-pred`
- [ ] Update to depend on `ft-scaling` for data

### Phase 4: Clean `datadec`

Ensure it only contains AllenAI DataDecide processing:

- [ ] Remove any project-specific code (moved to ft-scaling)
- [ ] Verify clean separation from ft-* repos

### Phase 5: Clean `dr_ingest`

After extraction to ft-scaling:

- [ ] Remove moved code
- [ ] Keep only what doesn't fit elsewhere (if anything)
- [ ] Consider deprecating if empty

### Phase 6: Update `by-tomorrow-app`

Decide how app consumes data:

- [ ] Option A: Keep serving from committed files
- [ ] Option B: Pull from `ft-scaling` HuggingFace dataset at runtime
- [ ] Option C: Use MotherDuck/dr_duck for queries
- [ ] Option D: Remove data viz, host elsewhere

## Open Questions

1. ~~**dr_ingest scope:** How much is generic DuckDB vs project-specific?~~ ✓ Audited
2. ~~**ddpred contents:** What AllenAI parsing exists there?~~ ✓ Audited
3. **Data serving:** What should by-tomorrow-app serve and from where?
4. **HuggingFace naming:** Rename `dd_parsed` → `ft-scaling`?

---

## Summary Table

| Current Repo | Destination | Content | Status |
|--------------|-------------|---------|--------|
| dr_ingest (DuckDB) | **dr_duck** | DuckDB/MotherDuck utilities | ✅ Published v0.1.0 |
| dr_ingest (pandas) | **dr_frames** | DataFrame manipulation utilities | ✅ Published v0.1.0 |
| hf_utils + dr_ingest/hf | **dr_hf** | HuggingFace operations | ✅ Published v0.1.0 |
| dr_showntell | **dr_render** | Table formatting | ✅ Published v0.1.0 |
| dr_ingest (project) | **ft-scaling** (new) | WandB parsing, pipelines | Pending |
| ddpred | **ft-pred** (rename) | Prediction models | Pending |
| ddpred (data loading) | **datadec** | DataDecide integration | Pending |
| datadec | **datadec** (cleanup) | AllenAI constants/loading | Pending |
| dd_parsed (HF) | **ft-scaling** (HF) | Rename dataset | Pending |

---

*Created: February 2025*
*Updated: February 2025*
*Status: Phase 1 complete (all utility packages published, all dependent repos updated). Ready for Phase 2 (ft-scaling).*

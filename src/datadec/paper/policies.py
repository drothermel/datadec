from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from datadec.config import DataDecideCatalog, OLMESContract
    from datadec.paper.models import PaperReproductionContract
    from datadec.paper.verifiers.olmes import NormalizedOlmesPolicy


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _require_unique(values: tuple[str, ...], *, label: str) -> None:
    duplicates = _duplicates(values)
    if duplicates:
        raise ValueError(f"duplicate {label} aliases: {duplicates!r}")


def _require_exact_aliases(
    configured: tuple[str, ...], canonical: tuple[str, ...], *, label: str
) -> None:
    _require_unique(configured, label=f"configured {label}")
    _require_unique(canonical, label=f"canonical {label}")
    missing = tuple(sorted(set(canonical).difference(configured)))
    unexpected = tuple(sorted(set(configured).difference(canonical)))
    if missing or unexpected:
        raise ValueError(
            f"{label} aliases drifted: missing={missing!r}, unexpected={unexpected!r}"
        )


def _require_known_aliases(
    configured: tuple[str, ...], canonical: tuple[str, ...], *, label: str
) -> None:
    _require_unique(configured, label=f"configured {label}")
    _require_unique(canonical, label=f"canonical {label}")
    missing = tuple(sorted(set(configured).difference(canonical)))
    if missing:
        raise ValueError(f"unknown {label} aliases: {missing!r}")


def resolve_olmes_policy(
    contract: PaperReproductionContract | None = None,
    *,
    catalog: DataDecideCatalog | None = None,
    olmes_contract: OLMESContract | None = None,
) -> NormalizedOlmesPolicy:
    """Resolve and cross-validate the paper's explicit OLMES analysis policy."""
    from datadec.config import (
        load_catalog,
        load_olmes_contract,
        load_paper_reproduction_contract,
    )
    from datadec.paper.analysis import (
        MMLU_SUBJECTS,
        OLMES_NON_MMLU_TASKS,
        TiePolicy,
    )
    from datadec.paper.verifiers.olmes import (
        FinalCheckpoint,
        MissingDataBehavior,
        NormalizedOlmesPolicy,
        OlmesTaskGrouping,
    )

    reproduction = contract or load_paper_reproduction_contract()
    repository_catalog = catalog or load_catalog()
    repository_olmes = olmes_contract or load_olmes_contract()
    analysis = reproduction.olmes_analysis

    catalog_recipes = tuple(
        recipe
        for family_recipes in repository_catalog.data_recipe_families.values()
        for recipe in family_recipes
    )
    _require_exact_aliases(analysis.recipes.aliases, catalog_recipes, label="recipe")
    _require_exact_aliases(
        analysis.recipes.aliases,
        tuple(repository_olmes.recipe_map.values()),
        label="OLMES recipe",
    )

    configured_seeds = tuple(
        dict.fromkeys(
            (*analysis.seeds.target_aliases, *analysis.seeds.prediction_aliases)
        )
    )
    _require_exact_aliases(
        configured_seeds,
        tuple(repository_catalog.seed_map),
        label="seed",
    )
    _require_exact_aliases(
        configured_seeds,
        tuple(repository_olmes.seed_map.values()),
        label="OLMES seed",
    )

    metric_columns = (
        analysis.metrics.target_column,
        *analysis.metrics.proxy_columns,
    )
    _require_known_aliases(
        metric_columns,
        tuple(repository_catalog.metric_names),
        label="catalog metric",
    )
    _require_known_aliases(
        metric_columns,
        repository_olmes.metrics.aggregate,
        label="OLMES aggregate metric",
    )
    aggregate_columns = tuple(
        column.name for column in repository_olmes.tables.aggregate.columns
    )
    _require_known_aliases(
        metric_columns,
        aggregate_columns,
        label="OLMES aggregate column",
    )

    primary_metric_tasks = tuple(
        repository_olmes.metrics.aggregate_primary_metric.model_dump()
    )
    if analysis.task_grouping.mmlu_task_name not in primary_metric_tasks:
        raise ValueError(
            "OLMES MMLU aggregate task alias is missing from the primary-metric policy"
        )
    configured_non_mmlu = analysis.task_grouping.non_mmlu_tasks
    contract_non_mmlu = tuple(
        task
        for task in primary_metric_tasks
        if task != analysis.task_grouping.mmlu_task_name
    )
    _require_exact_aliases(
        configured_non_mmlu, OLMES_NON_MMLU_TASKS, label="non-MMLU task"
    )
    _require_exact_aliases(
        configured_non_mmlu,
        contract_non_mmlu,
        label="OLMES primary-metric task",
    )
    _require_exact_aliases(
        analysis.task_grouping.mmlu_subjects,
        MMLU_SUBJECTS,
        label="MMLU subject",
    )

    models_by_name = {model.name: model for model in repository_catalog.models}
    model_names = tuple(model.name for model in repository_catalog.models)
    _require_unique(model_names, label="catalog model size")
    checkpoint_sizes = tuple(
        checkpoint.model_size for checkpoint in analysis.final_checkpoints.checkpoints
    )
    _require_exact_aliases(
        checkpoint_sizes, model_names, label="final-checkpoint model size"
    )
    for label, model_size in (
        ("target", analysis.target_model_size),
        ("noise", analysis.noise_model_size),
    ):
        if model_size not in models_by_name:
            raise ValueError(f"unknown {label} model-size alias: {model_size!r}")
        if model_size not in checkpoint_sizes:
            raise ValueError(
                f"{label} model size is missing a final checkpoint: {model_size!r}"
            )

    compute = analysis.compute
    if (
        compute.flops_per_token_per_parameter
        != repository_catalog.training.flops_per_token_per_parameter
    ):
        raise ValueError(
            "OLMES compute multiplier drifted from the catalog training contract"
        )
    _require_known_aliases(
        (compute.parameter_count_column, compute.token_count_column),
        aggregate_columns,
        label="OLMES compute column",
    )
    target_parameters = models_by_name[analysis.target_model_size].exact_parameter_count
    derived_target_compute = (
        compute.flops_per_token_per_parameter
        * target_parameters
        * compute.target_training_tokens
        * compute.target_run_count
    )
    if compute.target_compute_denominator != derived_target_compute:
        raise ValueError(
            "OLMES target compute denominator does not equal one "
            "6 * exact-parameter-count * target-token-count run"
        )

    return NormalizedOlmesPolicy(
        recipes=analysis.recipes.aliases,
        target_size=analysis.target_model_size,
        target_seeds=analysis.seeds.target_aliases,
        prediction_seeds=analysis.seeds.prediction_aliases,
        target_metric_column=analysis.metrics.target_column,
        proxy_metric_columns=analysis.metrics.proxy_columns,
        task_grouping=OlmesTaskGrouping(
            non_mmlu_tasks=configured_non_mmlu,
            mmlu_subjects=analysis.task_grouping.mmlu_subjects,
            mmlu_task_name=analysis.task_grouping.mmlu_task_name,
        ),
        final_checkpoints=tuple(
            FinalCheckpoint(
                model_size=checkpoint.model_size,
                step=checkpoint.step,
            )
            for checkpoint in analysis.final_checkpoints.checkpoints
        ),
        noise_size=analysis.noise_model_size,
        tie_policy=TiePolicy.COUNT_AS_INCORRECT,
        attempt_ddof=analysis.standard_deviation.attempt_ddof,
        within_recipe_ddof=(analysis.standard_deviation.within_recipe_noise_ddof),
        spread_ddof=analysis.standard_deviation.spread_ddof,
        missing_data_behavior=MissingDataBehavior.RECORD,
        parameter_count_column=compute.parameter_count_column,
        token_count_column=compute.token_count_column,
        target_compute_denominator=float(derived_target_compute),
    )


__all__ = ["resolve_olmes_policy"]

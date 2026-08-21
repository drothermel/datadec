from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from datadec.config import (
    load_catalog,
    load_olmes_contract,
    load_paper_reproduction_contract,
)
from datadec.paper import (
    OperationalizationBasis,
    PaperReproductionContract,
    PolicyStatus,
    resolve_olmes_policy,
)
from datadec.paper.analysis import MMLU_SUBJECTS, OLMES_NON_MMLU_TASKS, TiePolicy
from datadec.paper.verifiers.olmes import MissingDataBehavior

EXPECTED_RECIPES = (
    "Dolma1.7",
    "Dolma1.7 (no code)",
    "Dolma1.7 (no math, code)",
    "Dolma1.7 (no Reddit)",
    "Dolma1.7 (no Flan)",
    "Dolma1.6++",
    "C4",
    "FineWeb-Pro",
    "FineWeb-Edu",
    "Falcon",
    "Falcon+CC",
    "Falcon+CC (QC 10%)",
    "Falcon+CC (QC 20%)",
    "Falcon+CC (QC Orig 10%)",
    "Falcon+CC (QC Tulu 10%)",
    "DCLM-Baseline",
    "DCLM-Baseline (QC 10%)",
    "DCLM-Baseline (QC 20%)",
    "DCLM-Baseline (QC 7%, FW3)",
    "DCLM-Baseline (QC 7%, FW2)",
    "DCLM-Baseline (QC FW 3%)",
    "DCLM-Baseline (QC FW 10%)",
    "DCLM-Baseline 25% / Dolma 75%",
    "DCLM-Baseline 50% / Dolma 50%",
    "DCLM-Baseline 75% / Dolma 25%",
)
EXPECTED_PROXIES = (
    "acc_raw",
    "acc_per_token",
    "acc_per_char",
    "correct_prob",
    "correct_prob_per_token",
    "correct_prob_per_char",
    "margin",
    "margin_per_token",
    "margin_per_char",
    "norm_correct_prob",
    "norm_correct_prob_per_token",
    "norm_correct_prob_per_char",
    "total_prob",
    "total_prob_per_token",
    "total_prob_per_char",
)
EXPECTED_FINAL_STEPS = {
    "4M": 5725,
    "6M": 9182,
    "8M": 13039,
    "10M": 15117,
    "14M": 21953,
    "16M": 24432,
    "20M": 14584,
    "60M": 29042,
    "90M": 29901,
    "150M": 38157,
    "300M": 45787,
    "530M": 57786,
    "750M": 63589,
    "1B": 69369,
}


def test_resolved_olmes_policy_is_the_exact_first_full_data_policy() -> None:
    reproduction = load_paper_reproduction_contract()
    catalog = load_catalog()

    policy = resolve_olmes_policy(reproduction)

    assert policy.recipes == EXPECTED_RECIPES
    assert len(policy.recipes) == 25
    assert len(policy.recipes) * (len(policy.recipes) - 1) // 2 == 300
    assert policy.target_size == "1B"
    assert policy.target_seeds == ("default", "large aux 2", "large aux 3")
    assert policy.prediction_seeds == (
        "default",
        "small aux 2",
        "small aux 3",
    )
    assert policy.target_metric_column == "primary_metric"
    assert policy.proxy_metric_columns == EXPECTED_PROXIES
    assert policy.metric_columns == ("primary_metric", *EXPECTED_PROXIES)
    assert len(policy.metric_columns) == 16
    assert policy.task_grouping.non_mmlu_tasks == OLMES_NON_MMLU_TASKS
    assert policy.task_grouping.mmlu_subjects == MMLU_SUBJECTS
    assert len(policy.task_grouping.mmlu_subjects) == 57
    assert policy.task_grouping.mmlu_task_name == "mmlu"
    assert policy.final_step_by_size == EXPECTED_FINAL_STEPS
    assert policy.noise_size == "150M"
    assert policy.tie_policy is TiePolicy.COUNT_AS_INCORRECT
    assert policy.attempt_ddof == 1
    assert policy.within_recipe_ddof == 1
    assert policy.spread_ddof == 1
    assert policy.missing_data_behavior is MissingDataBehavior.RECORD
    assert policy.parameter_count_column == "exact_parameter_count"
    assert policy.token_count_column == "tokens"

    target_model = next(model for model in catalog.models if model.name == "1B")
    assert target_model.exact_parameter_count == 1_176_832_000
    expected_compute = 6 * target_model.exact_parameter_count * 100_000_000_000
    assert expected_compute == 706_099_200_000_000_000_000
    assert policy.target_compute_denominator == expected_compute

    analysis = reproduction.olmes_analysis
    assert analysis.standard_deviation.basis is (
        OperationalizationBasis.REPOSITORY_OPERATIONALIZED
    )
    assert "paper omits" in analysis.standard_deviation.description
    assert analysis.missing_data.allow_complete_case is False
    assert analysis.compute.denominator_scope == "single_target_run"
    assert analysis.compute.target_run_count == 1
    statuses = {entry.id: entry.status for entry in reproduction.policies}
    assert statuses["statistical_fit"] is PolicyStatus.UNRESOLVED


def test_resolver_rejects_recipe_alias_drift_and_catalog_duplicates() -> None:
    reproduction = load_paper_reproduction_contract()
    catalog = load_catalog()
    olmes = load_olmes_contract()

    drifted_families = dict(catalog.data_recipe_families)
    drifted_families["c4"] = ["C4-drifted"]
    drifted_catalog = catalog.model_copy(
        update={"data_recipe_families": drifted_families}
    )
    with pytest.raises(ValueError, match="recipe aliases drifted"):
        resolve_olmes_policy(
            reproduction, catalog=drifted_catalog, olmes_contract=olmes
        )

    duplicate_families = dict(catalog.data_recipe_families)
    duplicate_families["c4"] = ["Dolma1.7"]
    duplicate_catalog = catalog.model_copy(
        update={"data_recipe_families": duplicate_families}
    )
    with pytest.raises(ValueError, match="duplicate canonical recipe aliases"):
        resolve_olmes_policy(
            reproduction, catalog=duplicate_catalog, olmes_contract=olmes
        )


def test_resolver_rejects_task_metric_size_and_seed_alias_drift() -> None:
    reproduction = load_paper_reproduction_contract()
    catalog = load_catalog()
    olmes = load_olmes_contract()

    grouping = reproduction.olmes_analysis.task_grouping.model_copy(
        update={
            "non_mmlu_tasks": (
                *OLMES_NON_MMLU_TASKS[:-1],
                "winogrande-drifted",
            )
        }
    )
    task_analysis = reproduction.olmes_analysis.model_copy(
        update={"task_grouping": grouping}
    )
    task_contract = reproduction.model_copy(update={"olmes_analysis": task_analysis})
    with pytest.raises(ValueError, match="non-MMLU task aliases drifted"):
        resolve_olmes_policy(task_contract, catalog=catalog, olmes_contract=olmes)

    aggregate_metrics = tuple(
        metric for metric in olmes.metrics.aggregate if metric != "margin_per_char"
    )
    metric_contract = olmes.model_copy(
        update={
            "metrics": olmes.metrics.model_copy(update={"aggregate": aggregate_metrics})
        }
    )
    with pytest.raises(ValueError, match="unknown OLMES aggregate metric aliases"):
        resolve_olmes_policy(
            reproduction, catalog=catalog, olmes_contract=metric_contract
        )

    size_catalog = catalog.model_copy(update={"models": catalog.models[1:]})
    with pytest.raises(ValueError, match="final-checkpoint model size aliases drifted"):
        resolve_olmes_policy(reproduction, catalog=size_catalog, olmes_contract=olmes)

    seed_contract = olmes.model_copy(
        update={
            "seed_map": {
                key: value for key, value in olmes.seed_map.items() if key != 15
            }
        }
    )
    with pytest.raises(ValueError, match="OLMES seed aliases drifted"):
        resolve_olmes_policy(
            reproduction, catalog=catalog, olmes_contract=seed_contract
        )


def test_paper_contract_rejects_duplicate_analysis_aliases() -> None:
    raw = deepcopy(load_paper_reproduction_contract().model_dump())
    recipes = raw["olmes_analysis"]["recipes"]["aliases"]
    raw["olmes_analysis"]["recipes"]["aliases"] = (
        *recipes[:-1],
        recipes[0],
    )

    with pytest.raises(ValidationError, match="recipe aliases must be unique"):
        PaperReproductionContract.model_validate(raw)

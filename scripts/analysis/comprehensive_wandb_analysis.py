from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from rich.console import Console

from datadec.console_components import (
    InfoBlock,
    SectionRule,
    SectionTitlePanel,
    TitlePanel,
    create_counts_table,
)
from dr_frames import format_table
from dr_showntell import load_table_config
from datadec.wandb_eval import analysis_helpers
from datadec.wandb_eval import wandb_transforms as transforms
from datadec.wandb_eval.wandb_loader import WandBDataLoader

analysis_helpers.configure_pandas_display()

TABLE_CONFIG = load_table_config("wandb_analysis")
TABLE_STYLE = "zebra"
CRITICAL_FIELDS = ["model_size", "learning_rate", "train_loss", "state"]
ALL_RUN_TYPES = [
    "Training",
    "Training (minimal config)",
    "Training (config only)",
    "Eval-Only",
    "Unknown",
]


@dataclass
class DataQualityResults:
    total_runs: int
    run_classification: pd.Series
    field_completeness: List[Dict]
    metadata_validation: Dict[str, pd.Series]
    training_history_coverage: Dict[str, int]
    quality_score: float


@dataclass
class ExperimentalDesignResults:
    hyperparameter_extraction: Dict[str, int]
    experimental_grid: Dict[str, pd.DataFrame]
    sweep_completeness: List[Dict]
    method_comparison: pd.DataFrame
    special_experiments: List[Dict]
    coverage_score: float


@dataclass
class AnomalyDetectionResults:
    zero_lr_analysis: Dict
    failed_run_patterns: List[Dict]
    data_inconsistencies: List[Dict]
    outlier_summary: Dict
    reliability_score: float


@dataclass
class TrainingDynamicsResults:
    progression_summary: pd.DataFrame
    epoch_analysis: Dict
    convergence_patterns: List[Dict]
    performance_distribution: Dict
    success_score: float


@dataclass
class StrategyResults:
    plotting_viability: List[Dict]
    recommended_dimensions: List[Dict]
    data_gaps: List[str]
    next_experiments: List[str]
    analysis_limitations: List[str]
    confidence_scores: Dict[str, float]


@dataclass
class ComprehensiveAnalysisResults:
    data_quality: DataQualityResults
    experimental_design: ExperimentalDesignResults
    anomaly_detection: AnomalyDetectionResults
    training_dynamics: TrainingDynamicsResults
    recommendations: StrategyResults


def _extract_hyperparameters_unified(
    run_names: List[str],
) -> Tuple[List[Dict], Dict[str, int]]:
    extracted_params = []
    for name in run_names:
        params = transforms.extract_hyperparameters(name, ignore=[])
        converted_params = {"run_name": name}

        for key, value in params.items():
            if key.endswith("_rnp"):
                base_key = key[:-4]
                if base_key == "lr":
                    converted_params["learning_rate"] = value
                elif base_key == "params":
                    if isinstance(value, str) and value.endswith("M"):
                        converted_params["model_size_m"] = int(value[:-1])
                elif base_key == "total_tok":
                    converted_params["dataset_total_m"] = value
                elif base_key == "method":
                    converted_params["method"] = value
                elif base_key == "run_datetime":
                    converted_params["run_datetime"] = value
                else:
                    converted_params[base_key] = value
            else:
                converted_params[key] = value
        extracted_params.append(converted_params)

    extraction_stats = defaultdict(int)
    for params in extracted_params:
        for key in params:
            if key != "run_name" and params[key]:
                extraction_stats[key] += 1

    return extracted_params, dict(extraction_stats)


def _classify_run_type(row, has_history: bool) -> str:
    name = str(row["run_name"]).lower()

    has_training_config = any(
        [
            pd.notna(row.get("max_train_steps", None))
            and row.get("max_train_steps", 0) > 0,
            pd.notna(row.get("num_train_epochs", None))
            and row.get("num_train_epochs", 0) > 0,
            pd.notna(row.get("learning_rate", None))
            and row.get("learning_rate", 0) > 0,
        ]
    )

    if "eval" in name and not any(x in name for x in ["finetune", "dpo", "train"]):
        return "Eval-Only"
    elif has_history and has_training_config:
        return "Training"
    elif has_history and not has_training_config:
        return "Training (minimal config)"
    elif not has_history and has_training_config:
        return "Training (config only)"
    elif not has_history and not has_training_config:
        return "Eval-Only"
    else:
        return "Unknown"


def _calculate_data_quality(
    runs_df: pd.DataFrame, history_df: pd.DataFrame, extracted_params: List[Dict]
) -> DataQualityResults:
    runs_with_history = set(history_df["run_id"].unique())
    run_types = []
    for _, row in runs_df.iterrows():
        has_history = row["run_id"] in runs_with_history
        run_type = _classify_run_type(row, has_history)
        run_types.append(run_type)
    run_classification = pd.Series(run_types).value_counts()
    field_completeness = []
    for field in CRITICAL_FIELDS:
        if field in runs_df.columns:
            completeness = runs_df[field].notna().sum() / len(runs_df) * 100
            field_completeness.append(
                {
                    "field": field,
                    "complete_runs": runs_df[field].notna().sum(),
                    "total_runs": len(runs_df),
                    "completeness": f"{completeness:.1f}%",
                }
            )

    lr_comparison_data = []
    size_comparison_data = []
    for i, params in enumerate(extracted_params):
        run_data = runs_df.iloc[i]
        extracted_lr = params.get("learning_rate")
        metadata_lr = run_data.get("learning_rate", None)
        if extracted_lr and pd.notna(metadata_lr):
            try:
                extracted_float = float(extracted_lr)
                if abs(extracted_float - metadata_lr) < 1e-10:
                    lr_comparison_data.append("match")
                else:
                    lr_comparison_data.append("mismatch")
            except Exception:
                lr_comparison_data.append("parse_error")
        elif extracted_lr and pd.isna(metadata_lr):
            lr_comparison_data.append("name_only")
        elif not extracted_lr and pd.notna(metadata_lr):
            lr_comparison_data.append("metadata_only")
        else:
            lr_comparison_data.append("neither")
        extracted_size = params.get("model_size_m")
        metadata_size = run_data.get("model_size", None)
        if extracted_size and pd.notna(metadata_size):
            try:
                extracted_float = float(extracted_size) * 1e6
                if abs(extracted_float - metadata_size) / metadata_size < 0.01:
                    size_comparison_data.append("match")
                else:
                    size_comparison_data.append("mismatch")
            except Exception:
                size_comparison_data.append("parse_error")
        elif extracted_size:
            size_comparison_data.append("name_only")
        elif pd.notna(metadata_size):
            size_comparison_data.append("metadata_only")
        else:
            size_comparison_data.append("neither")

    metadata_validation = {
        "learning_rate": pd.Series(lr_comparison_data).value_counts(),
        "model_size": pd.Series(size_comparison_data).value_counts(),
    }
    training_history_coverage = {
        "runs_with_history": len(runs_with_history),
        "total_runs": len(runs_df),
        "coverage_percentage": len(runs_with_history) / len(runs_df) * 100,
    }
    finished_runs = (runs_df["state"] == "finished").sum()
    history_coverage = len(runs_with_history) / len(runs_df)
    field_avg_completeness = sum(
        float(fc["completeness"].rstrip("%"))
        for fc in field_completeness
        if fc["completeness"]
    ) / len(field_completeness)
    quality_score = (
        finished_runs / len(runs_df) * 40
        + history_coverage * 30
        + field_avg_completeness * 0.3
    )
    return DataQualityResults(
        total_runs=len(runs_df),
        run_classification=run_classification,
        field_completeness=field_completeness,
        metadata_validation=metadata_validation,
        training_history_coverage=training_history_coverage,
        quality_score=quality_score,
    )


def _calculate_experimental_design(
    runs_df: pd.DataFrame,
    extracted_params: List[Dict],
    extraction_stats: Dict[str, int],
) -> ExperimentalDesignResults:
    params_df = pd.DataFrame(extracted_params)
    if "model_size_m" in params_df.columns and "learning_rate" in params_df.columns:
        complete_params = params_df[
            params_df["model_size_m"].notna() & params_df["learning_rate"].notna()
        ]
        if len(complete_params) > 0:
            model_lr_crosstab = pd.crosstab(
                complete_params["model_size_m"],
                complete_params["learning_rate"],
                margins=True,
            )
            new_columns = []
            for col in model_lr_crosstab.columns:
                if col == "All":
                    new_columns.append("Present")
                elif isinstance(col, (int, float)):
                    new_columns.append(f"{float(col):.2e}")
                else:
                    new_columns.append(col)
            model_lr_crosstab.columns = new_columns
            if "All" in model_lr_crosstab.index:
                model_lr_crosstab = model_lr_crosstab.rename(index={"All": "Present"})
        else:
            model_lr_crosstab = pd.DataFrame()
    else:
        model_lr_crosstab = pd.DataFrame()
    if "model_size_m" in params_df.columns and "dataset_total_m" in params_df.columns:
        complete_params_data = params_df[
            params_df["model_size_m"].notna() & params_df["dataset_total_m"].notna()
        ]
        if len(complete_params_data) > 0:
            model_data_crosstab = pd.crosstab(
                complete_params_data["model_size_m"],
                complete_params_data["dataset_total_m"],
                margins=True,
            )
            if "All" in model_data_crosstab.columns:
                model_data_crosstab = model_data_crosstab.rename(
                    columns={"All": "Present"}
                )
        else:
            model_data_crosstab = pd.DataFrame()
    else:
        model_data_crosstab = pd.DataFrame()
    experimental_grid = {
        "model_lr": model_lr_crosstab,
        "model_data": model_data_crosstab,
    }
    sweep_completeness = []
    for param, count in extraction_stats.items():
        success_rate = count / len(extracted_params)
        sweep_completeness.append(
            {
                "parameter": param,
                "extracted_count": count,
                "total_runs": len(extracted_params),
                "success_rate": f"{success_rate * 100:.1f}%",
                "viability": "High"
                if success_rate > 0.7
                else "Medium"
                if success_rate > 0.4
                else "Low",
            }
        )
    if "method" in params_df.columns:
        method_comparison = params_df["method"].value_counts().to_frame("count")
        method_comparison["percentage"] = (
            method_comparison["count"] / len(params_df) * 100
        ).round(1)
    else:
        method_comparison = pd.DataFrame()
    special_experiments = []
    if "max_train_samples" in runs_df.columns:
        data_eff_count = runs_df["max_train_samples"].notna().sum()
        if data_eff_count > 0:
            special_experiments.append(
                {
                    "type": "Data Efficiency",
                    "count": data_eff_count,
                    "description": "Experiments with max_train_samples limits",
                }
            )
    if "reduce_loss_type" in runs_df.columns:
        loss_exp_count = runs_df["reduce_loss_type"].notna().sum()
        if loss_exp_count > 0:
            special_experiments.append(
                {
                    "type": "Loss Reduction",
                    "count": loss_exp_count,
                    "description": "Experiments with custom loss reduction",
                }
            )
    high_viability_params = sum(
        1 for sc in sweep_completeness if sc["viability"] == "High"
    )
    coverage_score = (
        (high_viability_params / len(sweep_completeness) * 100)
        if sweep_completeness
        else 0
    )
    return ExperimentalDesignResults(
        hyperparameter_extraction=extraction_stats,
        experimental_grid=experimental_grid,
        sweep_completeness=sweep_completeness,
        method_comparison=method_comparison,
        special_experiments=special_experiments,
        coverage_score=coverage_score,
    )


def _calculate_anomaly_detection(
    runs_df: pd.DataFrame, history_df: pd.DataFrame
) -> AnomalyDetectionResults:
    zero_lr_runs = runs_df[runs_df["learning_rate"] == 0.0]
    zero_lr_analysis = {
        "count": len(zero_lr_runs),
        "percentage": len(zero_lr_runs) / len(runs_df) * 100,
        "success_rate": (zero_lr_runs["state"] == "finished").sum()
        / len(zero_lr_runs)
        * 100
        if len(zero_lr_runs) > 0
        else 0,
    }
    state_counts = runs_df["state"].value_counts()
    failed_run_patterns = []
    for state, count in state_counts.items():
        if state != "finished":
            failed_run_patterns.append(
                {
                    "state": state,
                    "count": count,
                    "percentage": count / len(runs_df) * 100,
                }
            )
    data_inconsistencies = []
    if len(zero_lr_runs) > 0:
        data_inconsistencies.append(
            {
                "issue": "Zero Learning Rate",
                "count": len(zero_lr_runs),
                "impact": "High - affects training analysis",
            }
        )
    missing_history = len(runs_df) - history_df["run_id"].nunique()
    if missing_history > 0:
        data_inconsistencies.append(
            {
                "issue": "Missing Training History",
                "count": missing_history,
                "impact": "Medium - limits training analysis",
            }
        )
    outlier_summary = {
        "zero_lr_runs": len(zero_lr_runs),
        "failed_runs": len(runs_df) - (runs_df["state"] == "finished").sum(),
        "missing_history": missing_history,
    }
    total_issues = sum(outlier_summary.values())
    reliability_score = max(0, 100 - (total_issues / len(runs_df) * 100))
    return AnomalyDetectionResults(
        zero_lr_analysis=zero_lr_analysis,
        failed_run_patterns=failed_run_patterns,
        data_inconsistencies=data_inconsistencies,
        outlier_summary=outlier_summary,
        reliability_score=reliability_score,
    )


def _calculate_training_dynamics(
    runs_df: pd.DataFrame, history_df: pd.DataFrame
) -> TrainingDynamicsResults:
    if len(history_df) > 0:
        available_runs = list(history_df["run_id"].unique())[:10]
        if available_runs:
            progression_summary = analysis_helpers.analyze_training_progression(
                history_df, available_runs
            )
        else:
            progression_summary = pd.DataFrame()
    else:
        progression_summary = pd.DataFrame()
    epoch_analysis = {}
    if "num_train_epochs" in runs_df.columns:
        epochs_data = runs_df["num_train_epochs"].dropna()
        if len(epochs_data) > 0:
            epoch_analysis = {
                "runs_with_epochs": len(epochs_data),
                "mean_epochs": epochs_data.mean(),
                "epoch_range": f"{epochs_data.min():.1f} - {epochs_data.max():.1f}",
            }
    convergence_patterns = []
    if len(progression_summary) > 0:
        avg_steps = (
            progression_summary["total_steps"].mean()
            if "total_steps" in progression_summary.columns
            else 0
        )
        convergence_patterns.append(
            {
                "pattern": "Average Training Length",
                "value": f"{avg_steps:.0f} steps",
                "sample_size": len(progression_summary),
            }
        )
    performance_distribution = {}
    if "state" in runs_df.columns:
        success_rate = (runs_df["state"] == "finished").sum() / len(runs_df) * 100
        performance_distribution["overall_success_rate"] = success_rate
    success_score = performance_distribution.get("overall_success_rate", 0)
    return TrainingDynamicsResults(
        progression_summary=progression_summary,
        epoch_analysis=epoch_analysis,
        convergence_patterns=convergence_patterns,
        performance_distribution=performance_distribution,
        success_score=success_score,
    )


def _calculate_strategy_recommendations(
    data_quality: DataQualityResults,
    experimental_design: ExperimentalDesignResults,
    anomaly_detection: AnomalyDetectionResults,
    training_dynamics: TrainingDynamicsResults,
) -> StrategyResults:
    plotting_viability = []
    for sweep in experimental_design.sweep_completeness:
        if sweep["viability"] == "High":
            plotting_viability.append(
                {
                    "analysis_type": f"{sweep['parameter']} Analysis",
                    "viability": "High",
                    "data_points": sweep["extracted_count"],
                    "recommendation": "Suitable for primary analysis",
                }
            )
        elif sweep["viability"] == "Medium":
            plotting_viability.append(
                {
                    "analysis_type": f"{sweep['parameter']} Analysis",
                    "viability": "Medium",
                    "data_points": sweep["extracted_count"],
                    "recommendation": "Suitable for secondary analysis",
                }
            )
    recommended_dimensions = [
        {
            "dimension": "Model Size Scaling",
            "priority": "Primary",
            "rationale": "High extraction success rate",
        },
        {
            "dimension": "Learning Rate Sweeps",
            "priority": "Primary",
            "rationale": "Well-represented parameter space",
        },
        {
            "dimension": "Training Method Comparison",
            "priority": "Secondary",
            "rationale": "Good coverage for finetune vs DPO",
        },
    ]
    data_gaps = []
    if data_quality.quality_score < 70:
        data_gaps.append("Improve overall data completeness")
    if anomaly_detection.reliability_score < 80:
        data_gaps.append("Address anomalous runs and data inconsistencies")
    if experimental_design.coverage_score < 60:
        data_gaps.append("Expand hyperparameter sweep coverage")
    next_experiments = [
        "Focus on completing partial hyperparameter sweeps",
        "Address systematic data quality issues",
        "Prioritize experiments in well-covered parameter space",
    ]
    analysis_limitations = []
    if anomaly_detection.zero_lr_analysis["count"] > 0:
        analysis_limitations.append(
            "Zero learning rate runs may skew training analysis"
        )
    if data_quality.training_history_coverage["coverage_percentage"] < 80:
        analysis_limitations.append(
            "Limited training history reduces dynamics analysis reliability"
        )
    confidence_scores = {
        "hyperparameter_analysis": experimental_design.coverage_score,
        "training_dynamics": training_dynamics.success_score,
        "data_quality": data_quality.quality_score,
        "anomaly_detection": anomaly_detection.reliability_score,
    }
    return StrategyResults(
        plotting_viability=plotting_viability,
        recommended_dimensions=recommended_dimensions,
        data_gaps=data_gaps,
        next_experiments=next_experiments,
        analysis_limitations=analysis_limitations,
        confidence_scores=confidence_scores,
    )


def calculate_comprehensive_analysis() -> ComprehensiveAnalysisResults:
    loader = WandBDataLoader()
    runs_df, history_df = loader.load_runs_and_history()
    extracted_params, extraction_stats = _extract_hyperparameters_unified(
        runs_df["run_name"].tolist()
    )

    data_quality = _calculate_data_quality(runs_df, history_df, extracted_params)
    experimental_design = _calculate_experimental_design(
        runs_df, extracted_params, extraction_stats
    )
    anomaly_detection = _calculate_anomaly_detection(runs_df, history_df)
    training_dynamics = _calculate_training_dynamics(runs_df, history_df)
    recommendations = _calculate_strategy_recommendations(
        data_quality, experimental_design, anomaly_detection, training_dynamics
    )
    return ComprehensiveAnalysisResults(
        data_quality=data_quality,
        experimental_design=experimental_design,
        anomaly_detection=anomaly_detection,
        training_dynamics=training_dynamics,
        recommendations=recommendations,
    )


def _display_data_quality_section(console: Console, results: DataQualityResults):
    console.print(SectionRule("1. DATA QUALITY & COVERAGE ASSESSMENT"))
    console.print(InfoBlock(f"Total runs analyzed: {results.total_runs}"))
    console.print(InfoBlock(f"Overall quality score: {results.quality_score:.1f}/100"))
    console.print()
    console.print(SectionTitlePanel("Run Classification"))
    classification_data = []
    for run_type in ALL_RUN_TYPES:
        count = results.run_classification.get(run_type, 0)
        classification_data.append(
            {
                "run_type": run_type,
                "count": count,
                "percentage": f"{count / results.total_runs * 100:.1f}%",
            }
        )
    table = format_table(
        classification_data, title="Run Type Distribution", table_style=TABLE_STYLE
    )
    console.print(table)
    console.print()
    console.print(SectionTitlePanel("Field Completeness"))
    table = format_table(
        results.field_completeness,
        title="Critical Field Coverage",
        table_style=TABLE_STYLE,
    )
    console.print(table)
    console.print()
    console.print(SectionTitlePanel("Metadata Validation"))
    for param, validation_results in results.metadata_validation.items():
        validation_data = [
            {
                "category": category,
                "count": count,
                "description": _get_validation_description(category),
            }
            for category, count in validation_results.items()
        ]
        table = format_table(
            validation_data,
            title=f"{param.title()} Validation",
            table_style=TABLE_STYLE,
        )
        console.print(table)
    console.print()


def _display_experimental_design_section(
    console: Console, results: ExperimentalDesignResults
):
    console.print(SectionRule("2. EXPERIMENTAL DESIGN PATTERNS"))
    console.print(
        InfoBlock(
            f"Parameter extraction coverage score: {results.coverage_score:.1f}/100"
        )
    )
    console.print()
    console.print(SectionTitlePanel("Hyperparameter Extraction Success"))
    table = format_table(
        results.sweep_completeness,
        title="Parameter Extraction Rates",
        table_style=TABLE_STYLE,
    )
    console.print(table)
    console.print()
    if not results.experimental_grid["model_lr"].empty:
        console.print(
            SectionTitlePanel("Experimental Grid: Model Size × Learning Rate")
        )
        _display_experimental_grid_table(console, results.experimental_grid["model_lr"])
        console.print()
    if not results.method_comparison.empty:
        console.print(SectionTitlePanel("Training Method Distribution"))
        method_data = [
            {
                "method": method,
                "count": row["count"],
                "percentage": f"{row['percentage']:.1f}%",
            }
            for method, row in results.method_comparison.iterrows()
        ]
        table = format_table(
            method_data, title="Method Comparison", table_style=TABLE_STYLE
        )
        console.print(table)
        console.print()
    if results.special_experiments:
        console.print(SectionTitlePanel("Special Experiments"))
        table = format_table(
            results.special_experiments,
            title="Specialized Experiment Types",
            table_style=TABLE_STYLE,
        )
        console.print(table)
        console.print()


def _display_anomaly_detection_section(
    console: Console, results: AnomalyDetectionResults
):
    console.print(SectionRule("3. ANOMALY DETECTION & DATA ISSUES"))
    console.print(
        InfoBlock(f"Data reliability score: {results.reliability_score:.1f}/100")
    )
    console.print()
    console.print(SectionTitlePanel("Zero Learning Rate Analysis"))
    zero_lr_data = [
        {"metric": "Total zero LR runs", "value": results.zero_lr_analysis["count"]},
        {
            "metric": "Percentage of all runs",
            "value": f"{results.zero_lr_analysis['percentage']:.1f}%",
        },
        {
            "metric": "Success rate",
            "value": f"{results.zero_lr_analysis['success_rate']:.1f}%",
        },
    ]
    table = format_table(
        zero_lr_data, title="Zero Learning Rate Impact", table_style=TABLE_STYLE
    )
    console.print(table)
    console.print()
    if results.failed_run_patterns:
        console.print(SectionTitlePanel("Failed Run Patterns"))
        table = format_table(
            results.failed_run_patterns,
            title="Non-Finished Run States",
            table_style=TABLE_STYLE,
        )
        console.print(table)
        console.print()
    if results.data_inconsistencies:
        console.print(SectionTitlePanel("Data Inconsistencies"))
        table = format_table(
            results.data_inconsistencies,
            title="Identified Data Issues",
            table_style=TABLE_STYLE,
        )
        console.print(table)
        console.print()


def _display_training_dynamics_section(
    console: Console, results: TrainingDynamicsResults
):
    console.print(SectionRule("4. TRAINING DYNAMICS ANALYSIS"))
    console.print(InfoBlock(f"Training success score: {results.success_score:.1f}/100"))
    console.print()
    if not results.progression_summary.empty:
        console.print(SectionTitlePanel("Training Progression Sample"))
        console.print(
            InfoBlock(
                f"Sample of {len(results.progression_summary)} runs with training history"
            )
        )
        console.print(results.progression_summary.head(10))
        console.print()
    if results.epoch_analysis:
        console.print(SectionTitlePanel("Epoch Analysis"))
        epoch_data = [
            {"metric": key.replace("_", " ").title(), "value": str(value)}
            for key, value in results.epoch_analysis.items()
        ]
        table = format_table(
            epoch_data, title="Training Epoch Statistics", table_style=TABLE_STYLE
        )
        console.print(table)
        console.print()
    if results.convergence_patterns:
        console.print(SectionTitlePanel("Convergence Patterns"))
        table = format_table(
            results.convergence_patterns,
            title="Training Characteristics",
            table_style=TABLE_STYLE,
        )
        console.print(table)
        console.print()


def _display_strategy_recommendations_section(
    console: Console, results: StrategyResults
):
    console.print(SectionRule("5. STRATEGIC RECOMMENDATIONS"))
    console.print()
    console.print(SectionTitlePanel("Plotting Viability Assessment"))
    table = format_table(
        results.plotting_viability, title="Analysis Readiness", table_style=TABLE_STYLE
    )
    console.print(table)
    console.print()
    console.print(SectionTitlePanel("Recommended Analysis Dimensions"))
    table = format_table(
        results.recommended_dimensions,
        title="Strategic Priorities",
        table_style=TABLE_STYLE,
    )
    console.print(table)
    console.print()
    if results.data_gaps:
        console.print(SectionTitlePanel("Data Gaps & Improvements Needed"))
        for gap in results.data_gaps:
            console.print(InfoBlock(f"• {gap}", "yellow"))
        console.print()

    console.print(SectionTitlePanel("Next Experiment Recommendations"))
    for recommendation in results.next_experiments:
        console.print(InfoBlock(f"• {recommendation}", "green"))
    console.print()
    if results.analysis_limitations:
        console.print(SectionTitlePanel("Analysis Limitations"))
        for limitation in results.analysis_limitations:
            console.print(InfoBlock(f"⚠️  {limitation}", "red"))
        console.print()
    console.print(SectionTitlePanel("Confidence Scores"))
    confidence_data = [
        {
            "analysis_area": area.replace("_", " ").title(),
            "confidence": f"{score:.1f}/100",
        }
        for area, score in results.confidence_scores.items()
    ]
    table = format_table(
        confidence_data, title="Analysis Confidence Assessment", table_style=TABLE_STYLE
    )
    console.print(table)


def _display_experimental_grid_table(console: Console, crosstab_df: pd.DataFrame):
    display_crosstab = crosstab_df.copy()
    new_index = []
    for idx in display_crosstab.index:
        if idx == "Present":
            new_index.append(idx)
        else:
            new_index.append(f"{idx}M")
    display_crosstab.index = new_index
    table = create_counts_table(
        crosstab_df=display_crosstab,
        row_section_title="Model Size",
        col_section_title="Learning Rate",
        present_row_name="Present",
        present_col_name="Present",
        col_display_transform=None,
    )
    console.print(InfoBlock("Experimental coverage matrix (run counts):"))
    console.print(table)
    console.print(
        InfoBlock(
            f"Total analyzable runs: {int(crosstab_df.loc['Present', 'Present'])}"
        )
    )


def _get_validation_description(category: str) -> str:
    descriptions = {
        "match": "Values match exactly",
        "mismatch": "Values differ significantly",
        "name_only": "Only in run name",
        "metadata_only": "Only in metadata",
        "neither": "Missing from both",
        "parse_error": "Parsing failed",
    }
    return descriptions.get(category, category)


def display_comprehensive_analysis(
    results: ComprehensiveAnalysisResults, console: Console
):
    console.print(
        TitlePanel("Comprehensive WandB Analysis: Complete Experimental Assessment")
    )
    console.print()
    _display_data_quality_section(console, results.data_quality)
    _display_experimental_design_section(console, results.experimental_design)
    _display_anomaly_detection_section(console, results.anomaly_detection)
    _display_training_dynamics_section(console, results.training_dynamics)
    _display_strategy_recommendations_section(console, results.recommendations)


def main():
    console = Console()
    console.print(InfoBlock("Loading and analyzing WandB experimental data...", "blue"))
    console.print()
    results = calculate_comprehensive_analysis()
    display_comprehensive_analysis(results, console)


if __name__ == "__main__":
    main()

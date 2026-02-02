import sys
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from rich.console import Console

from datadec.console_components import (
    InfoBlock,
    SectionRule,
    SectionTitlePanel,
    TitlePanel,
)
from dr_frames import format_table
from datadec.wandb_eval import analysis_helpers
from datadec.wandb_eval import wandb_transforms as transforms
from datadec.wandb_eval.wandb_loader import WandBDataLoader

analysis_helpers.configure_pandas_display()

HEADER_MAP = {
    "training_method": "Method",
    "learning_rate": "LR",
    "model_size": "Size",
    "dataset_tokens_amount": "Tokens",
    "epochs": "Epochs",
    "data": "Data",
    "exp_name": "Exp",
    "checkpoint": "Ckpt",
    "max_train_samples": "Max Samples",
    "reduce_loss": "Reduce Loss",
    "run_datetime": "DateTime",
}

COLUMN_CONFIG = {
    "index": {"header": "#", "formatter": "integer"},
    "run_name": {"header": "Run Name", "formatter": "string"},
}


@dataclass
class ParsingAnalysisResults:
    total_runs: int
    run_names: list[str]
    name_lengths: list[int]
    short_names: list[str]
    medium_names: list[str]
    long_names: list[str]
    extracted_params: list[dict]
    extraction_stats: dict[str, int]
    extracted_lrs: list[str]
    unique_extracted_lrs: list[str]
    extracted_sizes: list[str]
    unique_sizes: list[str]
    token_amounts: list[str]
    unique_token_amounts: list[str]
    lr_comparison_counts: pd.Series
    size_comparison_counts: pd.Series
    reliable_params: list[tuple]
    unreliable_params: list[str]


def _load_and_process_runs() -> tuple[pd.DataFrame, list[str]]:
    loader = WandBDataLoader()
    runs_df, _ = loader.load_runs_and_history()
    run_names = runs_df["run_name"].tolist()
    return runs_df, run_names


def _analyze_name_patterns(run_names: list[str]) -> dict:
    name_lengths = [len(name) for name in run_names]
    short_names = [name for name in run_names if len(name) < 80]
    medium_names = [name for name in run_names if 80 <= len(name) < 120]
    long_names = [name for name in run_names if len(name) >= 120]

    return {
        "name_lengths": name_lengths,
        "short_names": short_names,
        "medium_names": medium_names,
        "long_names": long_names,
    }


def _extract_hyperparameters(run_names: list[str]) -> tuple[list[dict], dict[str, int]]:
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
                        converted_params["model_size"] = value[:-1]
                elif base_key == "total_tok":
                    converted_params["dataset_tokens_amount"] = str(value)
                elif base_key == "method":
                    converted_params["training_method"] = value
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


def _analyze_specific_parameters(extracted_params: list[dict]) -> dict:
    extracted_lrs = [
        params.get("learning_rate")
        for params in extracted_params
        if params.get("learning_rate")
    ]
    unique_extracted_lrs = list(set(extracted_lrs))[:15]
    extracted_sizes = [
        params.get("model_size")
        for params in extracted_params
        if params.get("model_size")
    ]
    unique_sizes = sorted(set(extracted_sizes), key=lambda x: float(x) if x else 0)
    token_amounts = [
        params.get("dataset_tokens_amount")
        for params in extracted_params
        if params.get("dataset_tokens_amount")
    ]
    unique_token_amounts = sorted(
        set(token_amounts), key=lambda x: float(x) if x else 0
    )
    return {
        "extracted_lrs": extracted_lrs,
        "unique_extracted_lrs": unique_extracted_lrs,
        "extracted_sizes": extracted_sizes,
        "unique_sizes": unique_sizes,
        "token_amounts": token_amounts,
        "unique_token_amounts": unique_token_amounts,
    }


def _compare_with_metadata(extracted_params: list[dict], runs_df: pd.DataFrame) -> dict:
    lr_comparison_data = []
    for i, params in enumerate(extracted_params):
        run_data = runs_df.iloc[i]
        extracted_lr = params.get("learning_rate")
        metadata_lr = run_data.get("learning_rate", None)
        if extracted_lr and pd.notna(metadata_lr):
            try:
                extracted_float = float(extracted_lr)
                if abs(extracted_float - metadata_lr) < 1e-10 or metadata_lr == 0.0:
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

    size_comparison_data = []
    for i, params in enumerate(extracted_params):
        run_data = runs_df.iloc[i]
        extracted_size = params.get("model_size")
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
    return {
        "lr_comparison_counts": pd.Series(lr_comparison_data).value_counts(),
        "size_comparison_counts": pd.Series(size_comparison_data).value_counts(),
    }


def _generate_recommendations(
    extraction_stats: dict[str, int], total_runs: int
) -> dict:
    reliable_params = []
    for param, count in extraction_stats.items():
        success_rate = count / total_runs
        if success_rate > 0.5:
            reliable_params.append((param, success_rate))
    reliable_params.sort(key=lambda x: x[1], reverse=True)
    unreliable_params = [
        param for param, count in extraction_stats.items() if count / total_runs < 0.3
    ]
    return {"reliable_params": reliable_params, "unreliable_params": unreliable_params}


def calculate_parsing_analysis() -> ParsingAnalysisResults:
    runs_df, run_names = _load_and_process_runs()
    name_analysis = _analyze_name_patterns(run_names)
    extracted_params, extraction_stats = _extract_hyperparameters(run_names)
    param_analysis = _analyze_specific_parameters(extracted_params)
    metadata_comparison = _compare_with_metadata(extracted_params, runs_df)
    recommendations = _generate_recommendations(extraction_stats, len(run_names))
    return ParsingAnalysisResults(
        total_runs=len(run_names),
        run_names=run_names,
        name_lengths=name_analysis["name_lengths"],
        short_names=name_analysis["short_names"],
        medium_names=name_analysis["medium_names"],
        long_names=name_analysis["long_names"],
        extracted_params=extracted_params,
        extraction_stats=extraction_stats,
        extracted_lrs=param_analysis["extracted_lrs"],
        unique_extracted_lrs=param_analysis["unique_extracted_lrs"],
        extracted_sizes=param_analysis["extracted_sizes"],
        unique_sizes=param_analysis["unique_sizes"],
        token_amounts=param_analysis["token_amounts"],
        unique_token_amounts=param_analysis["unique_token_amounts"],
        lr_comparison_counts=metadata_comparison["lr_comparison_counts"],
        size_comparison_counts=metadata_comparison["size_comparison_counts"],
        reliable_params=recommendations["reliable_params"],
        unreliable_params=recommendations["unreliable_params"],
    )


def _display_title_and_overview(console: Console, results: ParsingAnalysisResults):
    console.print(TitlePanel("Run Name Hyperparameter Parsing Analysis"))
    console.print(InfoBlock(f"Total runs analyzed: {results.total_runs}"))
    console.print()


def _display_name_pattern_analysis(console: Console, results: ParsingAnalysisResults):
    console.print(SectionRule("RUN NAME PATTERN ANALYSIS"))
    console.print(SectionTitlePanel("Sample Run Names (Evolution)"))
    for i, name in enumerate(results.run_names[:10]):
        console.print(f"  {i + 1:2}. {name[:80]}{'...' if len(name) > 80 else ''}")
    console.print()
    mean_length = sum(results.name_lengths) / len(results.name_lengths)
    length_data = [
        {"category": "Mean Length", "value": f"{mean_length:.0f} characters"},
        {
            "category": "Range",
            "value": f"{min(results.name_lengths)}-{max(results.name_lengths)} chars",
        },
        {"category": "Short (<80)", "value": f"{len(results.short_names)} runs"},
        {"category": "Medium (80-120)", "value": f"{len(results.medium_names)} runs"},
        {"category": "Long (≥120)", "value": f"{len(results.long_names)} runs"},
    ]
    table = format_table(
        length_data, title="Name Length Distribution", table_style="zebra"
    )
    console.print(table)
    console.print()


def _display_extraction_analysis(console: Console, results: ParsingAnalysisResults):
    console.print(SectionRule("HYPERPARAMETER EXTRACTION PATTERNS"))
    extraction_data = []
    for param, count in sorted(results.extraction_stats.items()):
        percentage = count / results.total_runs * 100
        extraction_data.append(
            {
                "parameter": param,
                "extracted": count,
                "total": results.total_runs,
                "success_rate": f"{percentage:.1f}%",
            }
        )
    table = format_table(
        extraction_data, title="Extraction Success Rates", table_style="zebra"
    )
    console.print(table)
    console.print()
    console.print(SectionTitlePanel("Parameter Patterns Found"))
    sorted_lrs = sorted(
        results.unique_extracted_lrs, key=lambda x: float(x) if x else 0
    )
    console.print(
        InfoBlock(
            f"Learning rates ({len(set(results.extracted_lrs))} unique): {', '.join(str(lr) for lr in sorted_lrs[:10])}",
            "dim",
        )
    )
    console.print(InfoBlock(f"Model sizes: {', '.join(results.unique_sizes)}", "dim"))
    token_info = [f"{amount}M" for amount in results.unique_token_amounts]
    console.print(
        InfoBlock(
            f"Token amounts ({len(results.unique_token_amounts)}): {', '.join(token_info)}",
            "dim",
        )
    )
    console.print()


def _display_metadata_comparison(console: Console, results: ParsingAnalysisResults):
    console.print(SectionRule("VALIDATION AGAINST METADATA"))
    lr_comparison_data = [
        {
            "category": category,
            "count": count,
            "description": _get_comparison_description(category),
        }
        for category, count in results.lr_comparison_counts.items()
    ]
    table = format_table(
        lr_comparison_data,
        title="Learning Rate: Name Extraction vs Metadata",
        table_style="zebra",
    )
    console.print(table)
    console.print()
    size_comparison_data = [
        {
            "category": category,
            "count": count,
            "description": _get_comparison_description(category),
        }
        for category, count in results.size_comparison_counts.items()
    ]
    table = format_table(
        size_comparison_data,
        title="Model Size: Name Extraction vs Metadata",
        table_style="zebra",
    )
    console.print(table)
    console.print()


def _display_recommendations(console: Console, results: ParsingAnalysisResults):
    console.print(SectionRule("PARSING RECOMMENDATIONS"))
    reliable_data = [
        {
            "parameter": param,
            "success_rate": f"{rate * 100:.1f}%",
            "recommendation": "Use name parsing"
            if rate > 0.7
            else "Consider hybrid approach",
        }
        for param, rate in results.reliable_params
    ]
    table = format_table(
        reliable_data, title="Reliability Assessment", table_style="zebra"
    )
    console.print(table)
    console.print()
    console.print(SectionTitlePanel("Strategic Parsing Approach"))
    strategy_items = [
        "Extract from names where reliability > 70%",
        "Use metadata fallback for unreliable parameters",
        "Cross-validate extracted vs metadata values",
        "Prioritize hybrid approach for robustness",
    ]
    for item in strategy_items:
        console.print(InfoBlock(f"• {item}", "green"))
    console.print()


def _get_comparison_description(category: str) -> str:
    descriptions = {
        "match": "Values match exactly",
        "mismatch": "Values differ significantly",
        "name_only": "Only in run name",
        "metadata_only": "Only in metadata",
        "neither": "Missing from both",
        "parse_error": "Parsing failed",
    }
    return descriptions.get(category, category)


def _display_detailed_examples(
    console: Console, results: ParsingAnalysisResults, num_examples: int = 50
):
    console.print(SectionRule("DETAILED RUN NAME EXAMPLES"))
    sorted_params = results.extracted_params.copy()
    params_with_dates = []
    params_without_dates = []
    for params in sorted_params:
        if params.get("run_datetime"):
            try:
                sort_key = params["run_datetime"]
                params_with_dates.append((sort_key, params))
            except Exception:
                params_without_dates.append(params)
        else:
            params_without_dates.append(params)
    params_with_dates.sort(key=lambda x: x[0], reverse=True)
    sorted_extracted_params = [p[1] for p in params_with_dates] + params_without_dates
    all_param_keys = set()
    for params in results.extracted_params:
        all_param_keys.update(params.keys())
    param_keys = sorted([key for key in all_param_keys if key != "run_name"])
    examples_data = []
    for i, params in enumerate(sorted_extracted_params[:num_examples]):
        example_row = {"index": i + 1, "run_name": params.get("run_name", "")}
        for key in param_keys:
            value = params.get(key, "")
            example_row[key] = "-" if (value == "" or value is None) else str(value)
        examples_data.append(example_row)
    for key in param_keys:
        header = HEADER_MAP.get(key, key.replace("_", " ").title())
        COLUMN_CONFIG[key] = {"header": header, "formatter": "string"}
    table = format_table(
        examples_data,
        title=f"Detailed Parameter Extraction Examples (First {len(examples_data)} runs, sorted by date)",
        table_style="zebra",
        column_config=COLUMN_CONFIG,
    )
    console.print(table)
    console.print()
    console.print(
        InfoBlock(
            f"Showing detailed extraction results for {len(examples_data)} run names with all discovered parameters.",
            "dim",
        )
    )
    console.print()


def display_parsing_analysis(
    results: ParsingAnalysisResults, console: Console, num_examples: int = 50
):
    _display_title_and_overview(console, results)
    _display_name_pattern_analysis(console, results)
    _display_extraction_analysis(console, results)
    _display_metadata_comparison(console, results)
    _display_recommendations(console, results)
    _display_detailed_examples(console, results, num_examples=num_examples)


def main():
    console = Console()
    if len(sys.argv) > 1:
        try:
            requested_examples = int(sys.argv[1])
        except ValueError:
            console.print(f"[red]Error: '{sys.argv[1]}' is not a valid number[/red]")
            console.print("Usage: python run_name_parsing.py [number_of_examples]")
            sys.exit(1)
    else:
        requested_examples = 50
    results = calculate_parsing_analysis()
    actual_examples = min(requested_examples, results.total_runs)
    console.print(
        InfoBlock(
            f"Requested {requested_examples} examples, showing {actual_examples} (total available: {results.total_runs})",
            "dim",
        )
    )
    console.print()
    display_parsing_analysis(results, console, num_examples=actual_examples)


if __name__ == "__main__":
    main()

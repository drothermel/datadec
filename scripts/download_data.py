"""Download and process DataDecide datasets with comprehensive demo."""

import argparse
import time
from typing import Optional

from datadec import DataDecide
from datadec import constants as consts


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def show_available_dataframes(dd: DataDecide) -> None:
    """Display all available DataFrames and their organization."""
    print_section_header("Available DataFrames")

    print("📊 Available DataFrames organized by pipeline stage:")
    print()

    # Group dataframes by category
    categories = {
        "Raw Data": ["ppl_raw", "dwn_raw"],
        "Intermediate Processing": [
            "dwn_metrics_expanded",
            "dwn_mmlu_averaged",
            "dwn_pivoted",
            "ppl_dwn_merged",
            "step_to_token_compute",
        ],
        "Parsed Data": ["ppl_parsed", "dwn_parsed"],
        "Final Datasets": ["full_eval", "mean_eval", "std_eval"],
        "Analysis-Ready": [
            "full_eval_with_details",
            "full_eval_with_lr",
            "mean_eval_with_details",
            "mean_eval_with_lr",
        ],
    }

    for category, dataframes in categories.items():
        print(f"🔹 {category}:")
        for df_name in dataframes:
            if df_name in dd.paths.available_dataframes:
                try:
                    df = dd.load_dataframe(df_name)
                    print(
                        f"   • {df_name:<25} {df.shape[0]:>8,} rows × {df.shape[1]:>3} cols"
                    )
                except Exception:
                    print(f"   • {df_name:<25} {'Not available'}")
        print()


def demo_basic_access(dd: DataDecide) -> None:
    """Demonstrate basic DataFrame access methods."""
    print_section_header("Basic DataFrame Access")

    print("🔍 Accessing main datasets:")
    print(
        f"   • Full evaluation data:  {dd.full_eval.shape[0]:>8,} rows × {dd.full_eval.shape[1]:>3} cols"
    )
    print(
        f"   • Mean evaluation data:  {dd.mean_eval.shape[0]:>8,} rows × {dd.mean_eval.shape[1]:>3} cols"
    )
    print(
        f"   • Model details:         {dd.model_details.shape[0]:>8,} rows × {dd.model_details.shape[1]:>3} cols"
    )
    print(
        f"   • Dataset details:       {dd.dataset_details.shape[0]:>8,} rows × {dd.dataset_details.shape[1]:>3} cols"
    )

    print("\n📋 Sample of full evaluation data:")
    print(dd.full_eval.head(3))


def demo_analysis_features(dd: DataDecide, min_params: str = "10M") -> None:
    """Demonstrate analysis-ready DataFrame generation."""
    print_section_header(f"Analysis Features (≥{min_params} parameters)")

    print("🧪 Testing get_analysis_df() with different configurations...")

    # Demo 1: Basic analysis DataFrame
    start_time = time.time()
    analysis_df = dd.get_analysis_df(min_params=min_params, verbose=True)
    elapsed = time.time() - start_time
    print(f"✅ Analysis DataFrame generated in {elapsed:.2f}s")
    print(
        f"   Final shape: {analysis_df.shape[0]:,} rows × {analysis_df.shape[1]} cols"
    )

    # Demo 2: Full data (no averaging)
    print("\n🔄 Comparing mean vs full data:")
    full_data = dd.get_analysis_df(min_params=min_params, return_means=False)
    print(f"   • With averaging across seeds: {analysis_df.shape[0]:,} rows")
    print(f"   • Without averaging (full):    {full_data.shape[0]:,} rows")

    # Demo 3: Show some key columns
    print("\n📊 Key columns in analysis DataFrame:")
    key_cols = ["params", "data", "step", "mmlu_average", "hellaswag", "arc_challenge"]
    lr_cols = [col for col in analysis_df.columns if "lr_" in col]

    print(
        "   • Core columns:",
        ", ".join(col for col in key_cols if col in analysis_df.columns),
    )
    if lr_cols:
        print(
            "   • Learning rate columns:",
            ", ".join(lr_cols[:3]) + ("..." if len(lr_cols) > 3 else ""),
        )

    print("\n📈 Sample analysis data:")
    display_cols = [col for col in key_cols if col in analysis_df.columns][:6]
    print(analysis_df[display_cols].head(3))


def demo_data_exploration(dd: DataDecide) -> None:
    """Demonstrate data exploration capabilities."""
    print_section_header("Data Exploration")

    # Show available model sizes and data recipes
    print("🏗️  Available model architectures:")
    model_sizes = sorted(
        consts.MODEL_SHAPES.keys(), key=lambda x: consts.HARDCODED_SIZE_MAPPING[x]
    )
    for size in model_sizes:
        params = consts.HARDCODED_SIZE_MAPPING[size]
        config = consts.MODEL_SHAPES[size]
        print(
            f"   • {size:<6} ({params:>11,} params) - {config['n_layers']} layers, {config['d_model']} dim"
        )

    print("\n📚 Available data recipe families:")
    for family, recipes in consts.DATA_RECIPE_FAMILIES.items():
        print(f"   • {family:<12} ({len(recipes)} recipes): {', '.join(recipes[:2])}")
        if len(recipes) > 2:
            print(f"     {'':<15} {'...' if len(recipes) > 3 else recipes[2]}")

    # Show actual data statistics
    full_eval = dd.full_eval
    print("\n📊 Dataset statistics:")
    print(f"   • Unique models:         {full_eval['params'].nunique()}")
    print(f"   • Unique data recipes:   {full_eval['data'].nunique()}")
    
    # Count task columns (tasks are columns, not rows)
    task_cols = [col for col in full_eval.columns if any(task in col for task in ['mmlu', 'arc', 'hellaswag', 'boolq', 'csqa', 'openbookqa', 'piqa', 'socialiqa', 'winogrande'])]
    unique_tasks = set()
    for col in task_cols:
        for task in ['mmlu', 'arc_challenge', 'arc_easy', 'hellaswag', 'boolq', 'csqa', 'openbookqa', 'piqa', 'socialiqa', 'winogrande']:
            if task in col:
                unique_tasks.add(task)
                break
    
    print(f"   • Evaluation tasks:      {len(unique_tasks)} tasks")
    print(f"   • Total training steps:  {full_eval['step'].max():,}")
    print(f"   • Seeds per experiment:  {full_eval['seed'].nunique()}")


def demo_filtering_options(
    dd: DataDecide, model_size: Optional[str] = None, data_recipe: Optional[str] = None
) -> None:
    """Demonstrate filtering capabilities."""
    if not model_size and not data_recipe:
        return

    print_section_header("Filtering Demo")

    # Start with analysis DataFrame
    base_df = dd.get_analysis_df(min_params="4M", return_means=False)
    print(f"🔧 Starting with {base_df.shape[0]:,} rows")

    if model_size:
        filtered_df = base_df[base_df["params"] == model_size]
        print(f"   • Filtered to {model_size}: {filtered_df.shape[0]:,} rows")
        base_df = filtered_df

    if data_recipe:
        filtered_df = base_df[base_df["data"] == data_recipe]
        print(f"   • Filtered to '{data_recipe}': {filtered_df.shape[0]:,} rows")
        base_df = filtered_df

    if base_df.shape[0] > 0:
        print("\n📋 Sample filtered data:")
        display_cols = ["params", "data", "step", "mmlu_average", "hellaswag"]
        available_cols = [col for col in display_cols if col in base_df.columns]
        print(base_df[available_cols].head(3))
    else:
        print("❌ No data matches the specified filters")


def demo_explore_mode(dd: DataDecide) -> None:
    """Interactive exploration mode."""
    print_section_header("Interactive Exploration Mode")

    print("🔍 Available exploration options:")
    print("   1. Try different model sizes")
    print("   2. Explore different data recipes")
    print("   3. Compare learning rate schedules")
    print("   4. Examine specific tasks")

    print("\n💡 Example queries you can run:")
    print("   • dd.get_analysis_df(min_params='300M')")
    print("   • dd.load_dataframe('ppl_raw')")
    print("   • dd.full_eval[dd.full_eval['params'] == '1B']")
    print("   • dd.model_details")

    print("\n📖 Access help: dd.paths.available_dataframes")


def main():
    """Download, process, and demonstrate DataDecide datasets."""
    parser = argparse.ArgumentParser(
        description="Download, process, and demonstrate DataDecide datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py                              # Basic download and demo
  python download_data.py --recompute_from parse       # Recompute from parsing stage
  python download_data.py --min_params 300M            # Demo with larger models only
  python download_data.py --explore                    # Interactive exploration mode
  python download_data.py --model_size 1B --data_recipe "Dolma1.7"  # Filtered demo
        """,
    )

    # Core arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory to store the downloaded and processed data.",
    )
    parser.add_argument(
        "--recompute_from",
        type=str,
        choices=["download", "metrics_expand", "parse", "merge", "aggregate", "all"],
        default=None,
        help="Stage to start recomputing from. Use 'all' to force complete reprocessing.",
    )

    # Demo configuration
    parser.add_argument(
        "--min_params",
        type=str,
        default="10M",
        help="Minimum model size for analysis demos (e.g., '10M', '300M', '1B')",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Enable interactive exploration mode with usage examples",
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Clear cache before running"
    )

    # Filtering options
    parser.add_argument(
        "--model_size",
        type=str,
        help="Filter demo to specific model size (e.g., '1B', '300M')",
    )
    parser.add_argument(
        "--data_recipe",
        type=str,
        help="Filter demo to specific data recipe (e.g., 'Dolma1.7', 'C4')",
    )

    args = parser.parse_args()

    print("🚀 DataDecide Setup and Demo")
    print(f"   Data directory: {args.data_dir}")
    if args.recompute_from:
        print(f"   Recomputing from: {args.recompute_from}")
    print()

    # Initialize DataDecide
    print("📦 Initializing DataDecide...")
    start_time = time.time()

    dd = DataDecide(
        data_dir=args.data_dir, recompute_from=args.recompute_from, verbose=True
    )

    if args.no_cache:
        dd.clear_cache()
        print("🗑️  Cache cleared")

    setup_time = time.time() - start_time
    print(f"✅ Setup completed in {setup_time:.2f}s")

    # Run demonstrations
    show_available_dataframes(dd)
    demo_basic_access(dd)
    demo_analysis_features(dd, min_params=args.min_params)
    demo_data_exploration(dd)

    if args.model_size or args.data_recipe:
        demo_filtering_options(dd, args.model_size, args.data_recipe)

    if args.explore:
        demo_explore_mode(dd)

    print_section_header("Setup Complete!")
    print("🎉 DataDecide is ready for analysis!")
    print(f"   • Access your data: dd = DataDecide('{args.data_dir}')")
    print("   • Quick analysis: dd.get_analysis_df()")
    print("   • Available DataFrames: dd.paths.available_dataframes")
    print()


if __name__ == "__main__":
    main()

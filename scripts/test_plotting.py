#!/usr/bin/env python3
"""
Test script for datadec plotting functionality.
Creates various configurations of scaling curve plots to validate the implementation.
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import datadec
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from datadec import DataDecide
from datadec.plotting import plot_scaling_curves, plot_model_comparison
import datadec.constants as consts

def main():
    # Load data
    print("Loading DataDecide data...")
    data_dir = repo_root / "outputs" / "example_data"
    
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist. Please run data pipeline first.")
        return
        
    dd = DataDecide(data_dir=str(data_dir), verbose=True)
    
    # Use mean_eval for averaged data across seeds
    df = dd.load_dataframe("mean_eval")
    print(f"Loaded mean_eval dataframe with shape: {df.shape}")
    print(f"Available columns: {list(df.columns)[:20]}...")  # Truncate long list
    print(f"Unique params: {sorted(df['params'].unique())}")
    print(f"Unique data: {sorted(df['data'].unique())[:10]}...")  # Truncate long list
    
    # Drop rows with NaN values in key plotting columns
    key_columns = ['tokens', 'pile-valppl', 'mmlu_average_correct_prob', 'params', 'data']
    available_key_columns = [col for col in key_columns if col in df.columns]
    print(f"Filtering NaN values in columns: {available_key_columns}")
    initial_shape = df.shape
    df = df.dropna(subset=available_key_columns)
    print(f"After NaN filtering: {df.shape} (removed {initial_shape[0] - df.shape[0]} rows)")
    
    # Create output directory for plots
    plots_dir = repo_root / "plots" / "test_plotting"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {plots_dir}")
    
    # Use the requested params and data values
    test_params = ["10M", "20M", "60M", "90M"]
    test_data = [
        "Dolma1.7",
        "DCLM-Baseline 25% / Dolma 75%",
        "DCLM-Baseline 50% / Dolma 50%",
        "DCLM-Baseline 75% / Dolma 25%", 
        "DCLM-Baseline"
    ]
    
    # Configuration 1: params as lines, data as subplots (your main use case)
    print("\n=== Configuration 1: params as lines, data as subplots ===")
    try:
        fig1, fm1 = plot_scaling_curves(
            df,
            x_col="tokens",
            y_col="pile-valppl", 
            line_col="params",
            subplot_col="data",
            params_filter=test_params,
            subplot_filter=test_data,
            title_prefix="Config 1"
        )
        fig1.savefig(plots_dir / "config1_params_lines_data_subplots.png", dpi=150, bbox_inches="tight")
        print("✓ Saved config1_params_lines_data_subplots.png")
    except Exception as e:
        print(f"✗ Error in config 1: {e}")
    
    # Configuration 2: data as lines, params as subplots (swapped)
    print("\n=== Configuration 2: data as lines, params as subplots ===")
    try:
        fig2, fm2 = plot_scaling_curves(
            df,
            x_col="tokens",
            y_col="pile-valppl",
            line_col="data", 
            subplot_col="params",
            params_filter=test_params,
            subplot_filter=test_data,
            title_prefix="Config 2"
        )
        fig2.savefig(plots_dir / "config2_data_lines_params_subplots.png", dpi=150, bbox_inches="tight")
        print("✓ Saved config2_data_lines_params_subplots.png")
    except Exception as e:
        print(f"✗ Error in config 2: {e}")
    
    # Configuration 3: Different metric (MMLU instead of perplexity)
    print("\n=== Configuration 3: MMLU metric ===")
    try:
        fig3, fm3 = plot_scaling_curves(
            df,
            x_col="tokens",
            y_col="mmlu_average_correct_prob",
            line_col="params",
            subplot_col="data", 
            params_filter=test_params,
            subplot_filter=test_data,
            title_prefix="Config 3"
        )
        fig3.savefig(plots_dir / "config3_mmlu_metric.png", dpi=150, bbox_inches="tight")
        print("✓ Saved config3_mmlu_metric.png")
    except Exception as e:
        print(f"✗ Error in config 3: {e}")
    
    # Configuration 4: Multi-metric comparison
    print("\n=== Configuration 4: Multi-metric comparison ===")
    try:
        test_metrics = ["pile-valppl", "mmlu_average_correct_prob"]
        # Check which metrics actually exist in the data
        available_metrics = [m for m in test_metrics if m in df.columns]
        print(f"Available metrics: {available_metrics}")
        
        if available_metrics:
            fig4, fm4 = plot_model_comparison(
                df,
                metrics=available_metrics,
                x_col="tokens",
                line_col="params",
                params_filter=test_params,
                data_filter=test_data
            )
            fig4.savefig(plots_dir / "config4_multi_metric.png", dpi=150, bbox_inches="tight")
            print("✓ Saved config4_multi_metric.png")
        else:
            print("✗ No valid metrics found for multi-metric plot")
    except Exception as e:
        print(f"✗ Error in config 4: {e}")
    
    # Configuration 5: Single data recipe, more params
    print("\n=== Configuration 5: Single data recipe, more params ===")
    try:
        all_params = sorted(df['params'].unique())[:5]  # Take first 5 params
        single_data = [test_data[0]] if test_data else [sorted(df['data'].unique())[0]]
        
        fig5, fm5 = plot_scaling_curves(
            df,
            x_col="tokens",
            y_col="pile-valppl",
            line_col="params",
            subplot_col="data",
            params_filter=all_params,
            subplot_filter=single_data,
            title_prefix="Config 5"
        )
        fig5.savefig(plots_dir / "config5_single_data_more_params.png", dpi=150, bbox_inches="tight")
        print("✓ Saved config5_single_data_more_params.png")
    except Exception as e:
        print(f"✗ Error in config 5: {e}")
    
    print(f"\nPlotting test complete! Check plots in: {plots_dir}")
    print("Use your image viewer to inspect the generated plots.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantization Impact Analysis Example

This script demonstrates how to use the enhanced quantization impact analysis
capabilities to evaluate different quantization strategies for a model.
"""

import os
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Ensure the project root is in the path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

# Import project components
from src.analysis.quantization import get_quantization_analyzer
from src.config.experiment_config import ExperimentConfiguration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_quantization_benchmark(model_name, strategies, priority="balanced"):
    """
    Run a quantization benchmark for a model.
    
    Args:
        model_name: Name of the model to benchmark
        strategies: List of quantization strategies to test
        priority: Priority for strategy selection
    
    Returns:
        Benchmark results
    """
    logger.info(f"Running quantization benchmark for model: {model_name}")
    
    # Get quantization analyzer
    analyzer = get_quantization_analyzer(
        model_name=model_name,
        results_dir=os.path.join(module_path, "results", "quantization")
    )
    
    # Configure sample size (smaller for demo purposes)
    analyzer.sample_size = 5
    
    # Prepare benchmark data
    logger.info("Preparing benchmark data...")
    if not analyzer.prepare_benchmark_data():
        logger.error("Failed to prepare benchmark data")
        return None
    
    # Run benchmark
    logger.info(f"Running benchmark with strategies: {strategies}")
    benchmark = analyzer.run_benchmark(
        strategies=strategies,
        priority=priority
    )
    
    if not benchmark:
        logger.error("Benchmark failed")
        return None
    
    logger.info(f"Benchmark completed. Best strategy: {benchmark.best_strategy}")
    return benchmark


def plot_memory_usage(benchmark):
    """
    Plot memory usage comparison.
    
    Args:
        benchmark: Benchmark results
    """
    # Extract data
    strategies = []
    memory_usages = []
    
    for result in benchmark.results:
        if result.error is None:
            strategies.append(result.strategy)
            memory_usages.append(result.memory_used_gb)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategies, memory_usages, color=sns.color_palette("viridis", len(strategies)))
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f} GB',
                ha='center', va='bottom', rotation=0)
    
    plt.title(f"Memory Usage by Quantization Strategy - {benchmark.model_name}")
    plt.xlabel("Quantization Strategy")
    plt.ylabel("Memory Usage (GB)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.join(module_path, "results", "quantization", "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "memory_usage_comparison.png"))
    
    logger.info(f"Memory usage plot saved to {output_dir}")


def plot_efficiency_comparison(benchmark):
    """
    Plot efficiency comparison (accuracy vs memory vs speed).
    
    Args:
        benchmark: Benchmark results
    """
    # Get visualization data
    if not hasattr(benchmark, "_visualization_data"):
        analyzer = get_quantization_analyzer(benchmark.model_name)
        viz_data = analyzer._generate_visualization_data(benchmark)
    else:
        viz_data = benchmark._visualization_data
    
    # Get normalized data for radar chart
    norm_data = viz_data.get("normalized_comparison", {})
    if not norm_data:
        logger.warning("No normalized data available for efficiency comparison")
        return
    
    # Extract data for radar chart
    strategies = norm_data.get("strategies", [])
    inv_metrics = norm_data.get("inverted_metrics", {})
    
    if not strategies or not inv_metrics:
        logger.warning("Insufficient data for radar chart")
        return
    
    # Create radar chart
    metrics = list(inv_metrics.keys())
    values = list(inv_metrics.values())
    
    # Number of variables
    N = len(metrics)
    
    # Create angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw one strategy per loop
    for i, strategy in enumerate(strategies):
        # Get values for this strategy
        vals = [values[j][i] for j in range(len(values))]
        vals += vals[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, vals, linewidth=2, linestyle='solid', label=strategy)
        ax.fill(angles, vals, alpha=0.1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Draw yticks
    ax.set_yticks([0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"])
    ax.set_rlim(0, 1)
    
    plt.title(f"Quantization Strategy Comparison - {benchmark.model_name}", size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save plot
    output_dir = os.path.join(module_path, "results", "quantization", "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "efficiency_comparison_radar.png"))
    
    logger.info(f"Efficiency comparison plot saved to {output_dir}")


def print_benchmark_results(benchmark):
    """
    Print benchmark results in a formatted table.
    
    Args:
        benchmark: Benchmark results
    """
    if not benchmark or not benchmark.results:
        logger.error("No benchmark results to print")
        return
    
    print("\n" + "="*80)
    print(f"QUANTIZATION BENCHMARK RESULTS: {benchmark.model_name}")
    print("="*80)
    
    # Create data for table
    data = []
    for result in benchmark.results:
        if result.error is None:
            data.append({
                "Strategy": result.strategy,
                "Memory (GB)": f"{result.memory_used_gb:.2f}",
                "Accuracy": f"{result.accuracy:.4f}",
                "Time (s)": f"{result.inference_time:.4f}",
                "Efficiency": f"{result.efficiency_score:.4f}",
                "Best": "✓" if result.strategy == benchmark.best_strategy else ""
            })
    
    # Print as DataFrame
    if data:
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
    
    # Print recommendations
    if benchmark.recommendations:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(benchmark.recommendations, 1):
            print(f"{i}. {rec}")
    
    print("="*80 + "\n")


def main():
    """Main function to run the example."""
    # Configure model and strategies to test
    model_name = "openai/clip-vit-base-patch32"  # Use a smaller model for the demo
    strategies = ["none", "int8", "int4", "gptq-int4"]  # Add/remove as needed
    
    try:
        # Run benchmark
        benchmark = run_quantization_benchmark(
            model_name=model_name,
            strategies=strategies,
            priority="balanced"
        )
        
        if benchmark:
            # Print results
            print_benchmark_results(benchmark)
            
            # Create visualizations
            plot_memory_usage(benchmark)
            plot_efficiency_comparison(benchmark)
            
            logger.info("Quantization analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error during quantization analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
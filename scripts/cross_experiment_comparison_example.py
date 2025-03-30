#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-Experiment Comparison Example

This script demonstrates how to use the enhanced cross-experiment comparison 
features to analyze and compare results from multiple experiments.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Ensure the project root is in the path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)

# Import project components
from src.analysis.comparison import ResultComparer
from src.results.schema import ExperimentResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_experiment_files(results_dir: str) -> List[str]:
    """
    Find experiment result files in the given directory.
    
    Args:
        results_dir: Directory to search for experiment files
        
    Returns:
        List of paths to experiment files
    """
    results_path = Path(results_dir)
    experiment_files = []
    
    # Look for JSON files that might contain experiment results
    for file_path in results_path.glob('**/*.json'):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Check if this looks like an experiment result file
            if 'experiment_name' in data and 'field_results' in data:
                experiment_files.append(str(file_path))
                logger.info(f"Found experiment file: {file_path}")
        except Exception as e:
            logger.warning(f"Error examining {file_path}: {e}")
    
    return experiment_files


def print_comparison_results(comparison_result, output_dir: str = None):
    """
    Print key information from the comparison results.
    
    Args:
        comparison_result: ExperimentComparisonResult instance
        output_dir: Optional directory to save comparison results
    """
    # Print basic information
    print("\n========== EXPERIMENT COMPARISON RESULTS ==========\n")
    print(f"Number of experiments compared: {len(comparison_result.experiment_metrics)}")
    print(f"Experiments: {', '.join(comparison_result.experiment_metrics.keys())}")
    
    # Print metrics for each experiment
    print("\n----- Experiment Metrics -----")
    for exp_name, metrics in comparison_result.experiment_metrics.items():
        print(f"\n{exp_name}:")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"  Avg Field Accuracy: {metrics['avg_field_accuracy']:.2%}")
        print(f"  Avg Field Error Rate: {metrics['avg_field_error_rate']:.4f}")
        print(f"  Avg Processing Time: {metrics['avg_field_processing_time']:.4f}s")
    
    # Print best experiment
    best_exp = comparison_result.find_best_experiment()
    if best_exp:
        print(f"\nBest performing experiment: {best_exp} with accuracy of "
              f"{comparison_result.experiment_metrics[best_exp]['overall_accuracy']:.2%}")
    
    # Print significant differences
    if comparison_result.significant_differences:
        print("\n----- Significant Differences -----")
        baseline = next(iter(comparison_result.experiment_metrics.keys()))
        for exp_name, diffs in comparison_result.significant_differences.items():
            print(f"\n{exp_name} vs {baseline}:")
            for metric, is_diff in diffs.items():
                if is_diff and not isinstance(is_diff, dict):
                    exp_value = comparison_result.experiment_metrics[exp_name][metric]
                    baseline_value = comparison_result.experiment_metrics[baseline][metric]
                    diff = exp_value - baseline_value
                    print(f"  {metric}: {exp_value:.4f} vs {baseline_value:.4f} (diff: {diff:+.4f})")
    
    # Print insights
    if comparison_result.insights:
        print("\n----- Insights -----")
        for insight in comparison_result.insights:
            print(f"[{insight['category']}] {insight['text']}")
    
    # Save comparison results if output directory is provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comparison result
        comparison_file = output_path / f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_result.to_dict(), f, indent=2)
        
        print(f"\nComparison results saved to {comparison_file}")


def main():
    """Main function to run the example."""
    # Configure paths
    results_dir = os.path.join(module_path, 'results')
    output_dir = os.path.join(module_path, 'analysis_results')
    
    # Find experiment result files
    logger.info(f"Searching for experiment files in {results_dir}")
    experiment_files = find_experiment_files(results_dir)
    
    if not experiment_files:
        logger.error(f"No experiment files found in {results_dir}")
        return
    
    # Load and compare experiments
    logger.info(f"Comparing {len(experiment_files)} experiments")
    comparer = ResultComparer()
    
    try:
        # Use the new method to load and compare experiments
        comparison_result = comparer.load_and_compare_experiments(
            experiment_files=experiment_files,
            metrics=['exact_match', 'character_error_rate', 'processing_time']
        )
        
        # Print and save comparison results
        print_comparison_results(comparison_result, output_dir)
        
    except Exception as e:
        logger.error(f"Error comparing experiments: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
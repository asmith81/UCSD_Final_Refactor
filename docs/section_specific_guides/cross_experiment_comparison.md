# Cross-Experiment Comparison Features

This document provides an overview of the cross-experiment comparison features
implemented in the analysis module.

## Overview

The cross-experiment comparison functionality allows users to compare results from
multiple experiments to identify performance trends, significant differences, and
generate insights to improve future experiments.

## Key Components

### 1. `ExperimentComparisonResult` Class

A comprehensive result class that stores and analyzes the comparison data:

- Calculates experiment-level metrics for direct comparison
- Identifies statistically significant differences between experiments
- Analyzes performance trends across experiments
- Finds the best-performing experiment based on various metrics
- Generates summaries and insights from the comparison data

### 2. `ExperimentComparisonStrategy` Class

Strategy class that implements the comparison logic:

- Extracts and compares metrics from multiple experiment results
- Analyzes field-specific performance across experiments
- Generates insights about performance, stability, and trends

### 3. Enhanced `ResultComparer` Class

The `ResultComparer` class has been extended with new methods:

- `compare_experiments()`: Directly compares loaded experiment results
- `load_and_compare_experiments()`: Loads experiment results from files and compares them

## Example Usage

```python
from src.analysis.comparison import ResultComparer
from src.results.schema import ExperimentResult

# Option 1: Load and compare experiments from files
comparer = ResultComparer()
comparison_result = comparer.load_and_compare_experiments(
    experiment_files=['path/to/experiment1.json', 'path/to/experiment2.json'],
    metrics=['exact_match', 'character_error_rate', 'processing_time']
)

# Option 2: Compare already loaded experiments
experiment1 = ExperimentResult.load_from_file('path/to/experiment1.json')
experiment2 = ExperimentResult.load_from_file('path/to/experiment2.json')

comparer = ResultComparer()
comparison_result = comparer.compare_experiments(
    experiments={
        'Experiment 1': experiment1,
        'Experiment 2': experiment2
    },
    metrics=['exact_match', 'character_error_rate', 'processing_time'],
    fields=['invoice_number', 'total_amount', 'date']  # Optional: limit to specific fields
)

# Access comparison results
best_experiment = comparison_result.find_best_experiment()
experiment_metrics = comparison_result.experiment_metrics
significant_differences = comparison_result.significant_differences
insights = comparison_result.insights

# Generate summary
summary = comparison_result.generate_summary()

# Save comparison results
comparison_result.save('path/to/comparison_results.json')
```

## Example Script

A complete example script is available at `scripts/cross_experiment_comparison_example.py`,
which demonstrates how to:

1. Find experiment result files in the results directory
2. Load and compare multiple experiments
3. Print and analyze the comparison results
4. Save the results to a JSON file

## Key Metrics Compared

The cross-experiment comparison analyzes several metrics:

- **Overall Accuracy**: Experiment-level accuracy across all fields
- **Field-specific Accuracies**: Performance for each individual field
- **Error Rates**: Character error rates and extraction failure rates
- **Processing Time**: Time required for extraction
- **Consistency**: Variance in performance across fields

## Generated Insights

The comparison automatically generates insights such as:

- Identification of the best-performing experiment
- Fields showing significant improvement or regression
- Performance stability across fields
- Performance trends over time
- Statistically significant differences between experiments

## Integration with Visualization

The comparison results are designed to be easily visualized using the
visualization module. The `to_visualization_data()` method converts the
comparison results into a format suitable for plotting and dashboard generation. 
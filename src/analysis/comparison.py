"""
Comparison Utilities for Extraction Results Analysis

This module provides comprehensive utilities for comparing extraction results across
different prompts, models, quantization strategies, and experiment runs.

Key features:
- Structured comparison of extraction performance
- Statistical significance testing
- Error pattern analysis
- Cross-field performance evaluation
"""

import os
import logging
import json
from enum import Enum, auto
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Generic, TypeVar
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy import stats

# Import project components
from src.results.schema import ExperimentResult, PromptPerformance, IndividualExtractionResult, ExtractionStatus

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic comparison results
T = TypeVar('T')


class ComparisonType(Enum):
    """Types of comparisons that can be performed."""
    PROMPT = auto()       # Compare different prompts for the same field
    MODEL = auto()        # Compare different models with similar configuration
    QUANTIZATION = auto()  # Compare different quantization strategies
    RUN = auto()          # Compare different experiment runs
    CROSS_FIELD = auto()  # Compare performance across different fields


class ComparisonMetric(Enum):
    """Standard metrics for comparisons."""
    EXACT_MATCH = "exact_match"                  # Exact match rate
    CHARACTER_ERROR_RATE = "character_error_rate"  # Character error rate
    PROCESSING_TIME = "processing_time"          # Processing time
    CONFIDENCE_SCORE = "confidence_score"        # Confidence score
    SUCCESS_RATE = "success_rate"                # Overall success rate
    MEMORY_USAGE = "memory_usage"                # Memory usage
    INFERENCE_TIME = "inference_time"            # Model inference time


@dataclass
class ComparisonContext:
    """
    Context information for a comparison.
    
    Stores configuration options and parameters for a specific comparison.
    """
    comparison_type: ComparisonType
    metrics: List[str] = field(default_factory=lambda: [
        ComparisonMetric.EXACT_MATCH.value,
        ComparisonMetric.CHARACTER_ERROR_RATE.value,
        ComparisonMetric.PROCESSING_TIME.value
    ])
    field: Optional[str] = None
    models: Optional[List[str]] = None
    prompts: Optional[List[str]] = None
    quantization_strategies: Optional[List[str]] = None
    run_ids: Optional[List[str]] = None
    fields: Optional[List[str]] = None
    significance_level: float = 0.05
    detailed_analysis: bool = True
    
    def validate(self) -> List[str]:
        """
        Validate the comparison context.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate based on comparison type
        if self.comparison_type == ComparisonType.PROMPT:
            if not self.field:
                errors.append("Field must be specified for prompt comparison")
            if not self.prompts or len(self.prompts) < 2:
                errors.append("At least two prompts must be specified for prompt comparison")
                
        elif self.comparison_type == ComparisonType.MODEL:
            if not self.models or len(self.models) < 2:
                errors.append("At least two models must be specified for model comparison")
                
        elif self.comparison_type == ComparisonType.QUANTIZATION:
            if not self.quantization_strategies or len(self.quantization_strategies) < 2:
                errors.append("At least two quantization strategies must be specified")
                
        elif self.comparison_type == ComparisonType.RUN:
            if not self.run_ids or len(self.run_ids) < 2:
                errors.append("At least two run IDs must be specified for run comparison")
                
        elif self.comparison_type == ComparisonType.CROSS_FIELD:
            if not self.fields or len(self.fields) < 2:
                errors.append("At least two fields must be specified for cross-field comparison")
        
        return errors


@dataclass
class ComparisonResult(Generic[T]):
    """
    Base class for all comparison results.
    
    Provides common functionality for storing and analyzing comparison results.
    """
    context: ComparisonContext
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    statistical_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    insights: List[Dict[str, str]] = field(default_factory=list)
    data: Optional[T] = None
    
    def add_metric(self, name: str, values: Dict[str, Any]) -> None:
        """
        Add a comparison metric.
        
        Args:
            name: Name of the metric
            values: Dictionary of values for the metric
        """
        self.metrics[name] = values
    
    def add_statistical_test(self, name: str, results: Dict[str, Any]) -> None:
        """
        Add statistical test results.
        
        Args:
            name: Name of the test
            results: Dictionary with test results
        """
        self.statistical_tests[name] = results
    
    def add_insight(self, insight: str, category: str = "general") -> None:
        """
        Add an insight derived from the comparison.
        
        Args:
            insight: Insight text
            category: Category of the insight
        """
        self.insights.append({
            "text": insight,
            "category": category
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert comparison result to a dictionary.
        
        Returns:
            Dictionary representation of the comparison result
        """
        return {
            "context": {
                "comparison_type": self.context.comparison_type.name,
                "metrics": self.context.metrics,
                "field": self.context.field,
                "models": self.context.models,
                "prompts": self.context.prompts,
                "quantization_strategies": self.context.quantization_strategies,
                "run_ids": self.context.run_ids,
                "fields": self.context.fields,
                "significance_level": self.context.significance_level,
                "detailed_analysis": self.context.detailed_analysis
            },
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "statistical_tests": self.statistical_tests,
            "insights": self.insights
        }
    
    def save(self, filepath: str) -> None:
        """
        Save comparison result to a JSON file.
        
        Args:
            filepath: Path to save the result
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_visualization_data(self) -> Dict[str, Any]:
        """
        Convert comparison results to a format suitable for visualization.
        
        Returns:
            Dictionary with data prepared for visualization
        """
        # Basic structure for visualization
        viz_data = {
            "type": self.context.comparison_type.name,
            "metrics": {},
            "entities": [],
            "insights": self.insights
        }
        
        # Extract entities based on comparison type
        if self.context.comparison_type == ComparisonType.PROMPT:
            viz_data["entities"] = self.context.prompts
            viz_data["field"] = self.context.field
            
        elif self.context.comparison_type == ComparisonType.MODEL:
            viz_data["entities"] = self.context.models
            
        elif self.context.comparison_type == ComparisonType.QUANTIZATION:
            viz_data["entities"] = self.context.quantization_strategies
            
        elif self.context.comparison_type == ComparisonType.RUN:
            viz_data["entities"] = self.context.run_ids
            
        elif self.context.comparison_type == ComparisonType.CROSS_FIELD:
            viz_data["entities"] = self.context.fields
        
        # Format metrics for visualization
        for metric_name, metric_data in self.metrics.items():
            viz_data["metrics"][metric_name] = metric_data
        
        return viz_data


@dataclass
class PromptComparisonResult(ComparisonResult[Dict[str, List[Dict[str, Any]]]]):
    """
    Comparison results for different prompts on the same field.
    """
    field: str = ""
    prompt_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_prompt: Optional[str] = None
    worst_prompt: Optional[str] = None
    
    def determine_best_prompt(self, metric: str = "exact_match") -> Optional[str]:
        """
        Determine the best performing prompt based on a metric.
        
        Args:
            metric: Metric to use for comparison (default: exact_match)
            
        Returns:
            Name of the best performing prompt or None if not available
        """
        if not self.prompt_metrics:
            return None
            
        # Find prompt with highest metric value
        best_value = -float('inf')
        best_prompt = None
        
        for prompt, metrics in self.prompt_metrics.items():
            if metric in metrics and metrics[metric] > best_value:
                best_value = metrics[metric]
                best_prompt = prompt
        
        self.best_prompt = best_prompt
        return best_prompt
    
    def determine_worst_prompt(self, metric: str = "exact_match") -> Optional[str]:
        """
        Determine the worst performing prompt based on a metric.
        
        Args:
            metric: Metric to use for comparison (default: exact_match)
            
        Returns:
            Name of the worst performing prompt or None if not available
        """
        if not self.prompt_metrics:
            return None
            
        # Find prompt with lowest metric value
        worst_value = float('inf')
        worst_prompt = None
        
        for prompt, metrics in self.prompt_metrics.items():
            if metric in metrics and metrics[metric] < worst_value:
                worst_value = metrics[metric]
                worst_prompt = prompt
        
        self.worst_prompt = worst_prompt
        return worst_prompt
    
    def calculate_performance_gap(self, metric: str = "exact_match") -> float:
        """
        Calculate the performance gap between best and worst prompt.
        
        Args:
            metric: Metric to use for comparison (default: exact_match)
            
        Returns:
            Performance gap as a ratio or 0 if not available
        """
        if not self.prompt_metrics or not self.best_prompt or not self.worst_prompt:
            return 0.0
            
        best_value = self.prompt_metrics[self.best_prompt].get(metric, 0)
        worst_value = self.prompt_metrics[self.worst_prompt].get(metric, 0)
        
        if worst_value == 0:
            return 0.0
            
        return best_value / worst_value


@dataclass
class ModelComparisonResult(ComparisonResult[Dict[str, Dict[str, Any]]]):
    """
    Comparison results for different models.
    """
    model_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_tradeoffs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def calculate_speed_accuracy_tradeoff(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate speed vs. accuracy tradeoff for each model.
        
        Returns:
            Dictionary with tradeoff metrics for each model
        """
        tradeoffs = {}
        
        for model, metrics in self.model_metrics.items():
            if "exact_match" in metrics and "processing_time" in metrics:
                accuracy = metrics["exact_match"]
                speed = 1.0 / metrics["processing_time"] if metrics["processing_time"] > 0 else 0
                
                # Calculate efficiency score (higher is better)
                efficiency = (accuracy * speed) if speed > 0 else 0
                
                tradeoffs[model] = {
                    "accuracy": accuracy,
                    "speed": speed,
                    "efficiency": efficiency
                }
        
        self.performance_tradeoffs = tradeoffs
        return tradeoffs
    
    def get_optimal_model(self, priority: str = "balanced") -> Optional[str]:
        """
        Get the optimal model based on a priority.
        
        Args:
            priority: Priority for selection (balanced, speed, accuracy)
            
        Returns:
            Name of the optimal model or None if not available
        """
        if not self.performance_tradeoffs:
            self.calculate_speed_accuracy_tradeoff()
            
        if not self.performance_tradeoffs:
            return None
        
        if priority == "balanced":
            # Use efficiency score (balanced consideration of speed and accuracy)
            metric = "efficiency"
        elif priority == "speed":
            metric = "speed"
        elif priority == "accuracy":
            metric = "accuracy"
        else:
            metric = "efficiency"
        
        # Find model with highest value for the selected metric
        best_value = -float('inf')
        optimal_model = None
        
        for model, tradeoffs in self.performance_tradeoffs.items():
            if metric in tradeoffs and tradeoffs[metric] > best_value:
                best_value = tradeoffs[metric]
                optimal_model = model
        
        return optimal_model


@dataclass
class QuantizationComparisonResult(ComparisonResult[Dict[str, Dict[str, Any]]]):
    """
    Comparison results for different quantization strategies.
    """
    strategy_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    memory_reduction: Dict[str, float] = field(default_factory=dict)
    accuracy_impact: Dict[str, float] = field(default_factory=dict)
    speed_impact: Dict[str, float] = field(default_factory=dict)
    
    def calculate_memory_impact(self, baseline: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate memory usage impact of each quantization strategy.
        
        Args:
            baseline: Optional baseline strategy for comparison
            
        Returns:
            Dictionary with memory reduction ratio for each strategy
        """
        if not self.strategy_metrics:
            return {}
            
        # Determine baseline (use non-quantized or highest memory usage as baseline)
        if baseline is None:
            # Look for "none" or "fp32" strategy
            if "none" in self.strategy_metrics:
                baseline = "none"
            elif "fp32" in self.strategy_metrics:
                baseline = "fp32"
            else:
                # Use strategy with highest memory usage
                baseline = max(
                    self.strategy_metrics.items(),
                    key=lambda x: x[1].get("memory_usage", 0)
                )[0]
        
        # Calculate reduction ratio for each strategy
        memory_reduction = {}
        baseline_memory = self.strategy_metrics[baseline].get("memory_usage", 0)
        
        if baseline_memory > 0:
            for strategy, metrics in self.strategy_metrics.items():
                memory_usage = metrics.get("memory_usage", 0)
                reduction_ratio = 1.0 - (memory_usage / baseline_memory)
                memory_reduction[strategy] = reduction_ratio
        
        self.memory_reduction = memory_reduction
        return memory_reduction
    
    def calculate_accuracy_impact(self, baseline: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate accuracy impact of each quantization strategy.
        
        Args:
            baseline: Optional baseline strategy for comparison
            
        Returns:
            Dictionary with accuracy impact ratio for each strategy
        """
        if not self.strategy_metrics:
            return {}
            
        # Determine baseline (use non-quantized or highest accuracy as baseline)
        if baseline is None:
            # Look for "none" or "fp32" strategy
            if "none" in self.strategy_metrics:
                baseline = "none"
            elif "fp32" in self.strategy_metrics:
                baseline = "fp32"
            else:
                # Use strategy with highest accuracy
                baseline = max(
                    self.strategy_metrics.items(),
                    key=lambda x: x[1].get("exact_match", 0)
                )[0]
        
        # Calculate accuracy impact for each strategy
        accuracy_impact = {}
        baseline_accuracy = self.strategy_metrics[baseline].get("exact_match", 0)
        
        if baseline_accuracy > 0:
            for strategy, metrics in self.strategy_metrics.items():
                accuracy = metrics.get("exact_match", 0)
                impact_ratio = (accuracy / baseline_accuracy) - 1.0
                accuracy_impact[strategy] = impact_ratio
        
        self.accuracy_impact = accuracy_impact
        return accuracy_impact
    
    def calculate_speed_impact(self, baseline: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate processing speed impact of each quantization strategy.
        
        Args:
            baseline: Optional baseline strategy for comparison
            
        Returns:
            Dictionary with speed impact ratio for each strategy
        """
        if not self.strategy_metrics:
            return {}
            
        # Determine baseline (use non-quantized or slowest as baseline)
        if baseline is None:
            # Look for "none" or "fp32" strategy
            if "none" in self.strategy_metrics:
                baseline = "none"
            elif "fp32" in self.strategy_metrics:
                baseline = "fp32"
            else:
                # Use strategy with highest processing time (slowest)
                baseline = max(
                    self.strategy_metrics.items(),
                    key=lambda x: x[1].get("processing_time", 0)
                )[0]
        
        # Calculate speed impact for each strategy
        speed_impact = {}
        baseline_time = self.strategy_metrics[baseline].get("processing_time", 0)
        
        if baseline_time > 0:
            for strategy, metrics in self.strategy_metrics.items():
                time = metrics.get("processing_time", 0)
                if time > 0:
                    # Calculate speedup factor (higher is better)
                    impact_ratio = (baseline_time / time) - 1.0
                    speed_impact[strategy] = impact_ratio
        
        self.speed_impact = speed_impact
        return speed_impact
    
    def get_optimal_strategy(self, priority: str = "balanced") -> Optional[str]:
        """
        Get the optimal quantization strategy based on a priority.
        
        Args:
            priority: Priority for selection (balanced, memory, speed, accuracy)
            
        Returns:
            Name of the optimal strategy or None if not available
        """
        if not self.strategy_metrics:
            return None
        
        # Calculate impacts if not already calculated
        if not self.memory_reduction:
            self.calculate_memory_impact()
        if not self.accuracy_impact:
            self.calculate_accuracy_impact()
        if not self.speed_impact:
            self.calculate_speed_impact()
        
        # Create scoring function based on priority
        if priority == "balanced":
            def score_func(strategy):
                memory_score = self.memory_reduction.get(strategy, 0) * 0.3
                speed_score = self.speed_impact.get(strategy, 0) * 0.3
                # Accuracy impact is negative if accuracy is reduced
                accuracy_score = max(-0.5, min(0.5, self.accuracy_impact.get(strategy, 0))) * 0.4
                return memory_score + speed_score + accuracy_score
        
        elif priority == "memory":
            def score_func(strategy):
                memory_score = self.memory_reduction.get(strategy, 0) * 0.7
                accuracy_score = max(-0.5, min(0.5, self.accuracy_impact.get(strategy, 0))) * 0.3
                return memory_score + accuracy_score
        
        elif priority == "speed":
            def score_func(strategy):
                speed_score = self.speed_impact.get(strategy, 0) * 0.7
                accuracy_score = max(-0.5, min(0.5, self.accuracy_impact.get(strategy, 0))) * 0.3
                return speed_score + accuracy_score
        
        elif priority == "accuracy":
            def score_func(strategy):
                accuracy_score = self.accuracy_impact.get(strategy, 0) * 0.7
                memory_score = self.memory_reduction.get(strategy, 0) * 0.15
                speed_score = self.speed_impact.get(strategy, 0) * 0.15
                return accuracy_score + memory_score + speed_score
        
        else:  # Default to balanced
            def score_func(strategy):
                memory_score = self.memory_reduction.get(strategy, 0) * 0.33
                speed_score = self.speed_impact.get(strategy, 0) * 0.33
                accuracy_score = max(-0.5, min(0.5, self.accuracy_impact.get(strategy, 0))) * 0.34
                return memory_score + speed_score + accuracy_score
        
        # Calculate scores and find optimal strategy
        scores = {strategy: score_func(strategy) for strategy in self.strategy_metrics.keys()}
        
        if not scores:
            return None
            
        return max(scores.items(), key=lambda x: x[1])[0]


@dataclass
class RunComparisonResult(ComparisonResult[Dict[str, Any]]):
    """
    Comparison results for different experiment runs.
    """
    run_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    differences: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def calculate_differences(self, baseline_run: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate differences between runs.
        
        Args:
            baseline_run: Optional baseline run for comparison
            
        Returns:
            Dictionary with difference metrics for each run
        """
        if not self.run_metrics:
            return {}
            
        # Determine baseline (use first run if not specified)
        if baseline_run is None or baseline_run not in self.run_metrics:
            baseline_run = next(iter(self.run_metrics))
        
        # Calculate differences for each run
        differences = {}
        baseline_metrics = self.run_metrics[baseline_run]
        
        for run, metrics in self.run_metrics.items():
            if run == baseline_run:
                continue
                
            run_diffs = {}
            for metric, value in metrics.items():
                if metric in baseline_metrics:
                    baseline_value = baseline_metrics[metric]
                    
                    # Calculate absolute and relative differences
                    absolute_diff = value - baseline_value
                    relative_diff = (value / baseline_value - 1.0) if baseline_value != 0 else float('inf')
                    
                    run_diffs[f"{metric}_abs_diff"] = absolute_diff
                    run_diffs[f"{metric}_rel_diff"] = relative_diff
            
            differences[run] = run_diffs
        
        self.differences = differences
        return differences
    
    def identify_significant_changes(self, threshold: float = 0.1) -> Dict[str, List[str]]:
        """
        Identify significant changes between runs.
        
        Args:
            threshold: Threshold for significant relative difference
            
        Returns:
            Dictionary mapping runs to lists of significantly changed metrics
        """
        if not self.differences:
            self.calculate_differences()
            
        significant_changes = {}
        
        for run, diffs in self.differences.items():
            significant = []
            
            for metric, value in diffs.items():
                if metric.endswith("_rel_diff") and abs(value) >= threshold:
                    # Extract base metric name
                    base_metric = metric.replace("_rel_diff", "")
                    significant.append(base_metric)
            
            if significant:
                significant_changes[run] = significant
        
        return significant_changes


@dataclass
class CrossFieldComparisonResult(ComparisonResult[Dict[str, Dict[str, Any]]]):
    """
    Comparison results for cross-field analysis.
    """
    field_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def calculate_field_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlations between field performance metrics.
        
        Returns:
            Dictionary with correlation coefficients between fields
        """
        if not self.field_metrics:
            return {}
            
        # Get list of fields and metrics
        fields = list(self.field_metrics.keys())
        if not fields:
            return {}
            
        all_metrics = set()
        for field_data in self.field_metrics.values():
            all_metrics.update(field_data.keys())
        
        # Calculate correlations for each metric
        correlations = {}
        
        for metric in all_metrics:
            metric_corr = {}
            
            for i, field1 in enumerate(fields):
                field1_metrics = self.field_metrics[field1]
                
                for field2 in fields[i+1:]:
                    field2_metrics = self.field_metrics[field2]
                    
                    # Skip if either field doesn't have this metric
                    if metric not in field1_metrics or metric not in field2_metrics:
                        continue
                    
                    # Calculate correlation
                    key = f"{field1}/{field2}"
                    metric_corr[key] = field1_metrics[metric] / field2_metrics[metric] if field2_metrics[metric] != 0 else float('inf')
            
            if metric_corr:
                correlations[metric] = metric_corr
        
        self.correlations = correlations
        return correlations
    
    def find_outlier_fields(self, metric: str = "exact_match", z_score_threshold: float = 2.0) -> List[str]:
        """
        Find outlier fields based on a specific metric.
        
        Args:
            metric: Metric to use for detection
            z_score_threshold: Z-score threshold for outlier detection
            
        Returns:
            List of outlier field names
        """
        if not self.field_metrics:
            return []
            
        # Get values for the specified metric
        values = []
        field_names = []
        
        for field, metrics in self.field_metrics.items():
            if metric in metrics:
                values.append(metrics[metric])
                field_names.append(field)
        
        if not values:
            return []
            
        # Calculate mean and standard deviation
        mean = np.mean(values)
        std = np.std(values)
        
        # Avoid division by zero
        if std == 0:
            return []
            
        # Find outliers
        outliers = []
        for i, value in enumerate(values):
            z_score = abs(value - mean) / std
            if z_score > z_score_threshold:
                outliers.append(field_names[i])
        
        return outliers


@dataclass
class ExperimentComparisonResult(ComparisonResult[Dict[str, ExperimentResult]]):
    """
    Comparison results for different experiments.
    
    This class provides comprehensive analysis of differences between
    multiple experiments, including model performance, prompt strategies,
    and field-specific analysis across experiments.
    """
    experiment_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    significant_differences: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    performance_trends: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    
    def calculate_experiment_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate and store key metrics for each experiment.
        
        Returns:
            Dictionary with metrics for each experiment
        """
        experiment_metrics = {}
        
        # Calculate metrics from the data
        if self.data:
            for exp_name, exp_result in self.data.items():
                metrics = {
                    'overall_accuracy': exp_result.overall_accuracy,
                    'total_fields': exp_result.total_fields,
                    'total_items': exp_result.total_items,
                }
                
                # Calculate field-specific metrics
                field_accuracies = {}
                field_error_rates = {}
                field_processing_times = {}
                
                for field, performances in exp_result.field_results.items():
                    field_accuracies[field] = sum(p.accuracy for p in performances) / len(performances) if performances else 0
                    field_error_rates[field] = sum(p.avg_character_error_rate for p in performances) / len(performances) if performances else 1.0
                    field_processing_times[field] = sum(p.avg_processing_time for p in performances) / len(performances) if performances else 0
                
                # Average metrics across fields
                metrics['avg_field_accuracy'] = sum(field_accuracies.values()) / len(field_accuracies) if field_accuracies else 0
                metrics['avg_field_error_rate'] = sum(field_error_rates.values()) / len(field_error_rates) if field_error_rates else 1.0
                metrics['avg_field_processing_time'] = sum(field_processing_times.values()) / len(field_processing_times) if field_processing_times else 0
                
                # Store field metrics
                metrics['field_accuracies'] = field_accuracies
                metrics['field_error_rates'] = field_error_rates
                metrics['field_processing_times'] = field_processing_times
                
                experiment_metrics[exp_name] = metrics
        
        self.experiment_metrics = experiment_metrics
        return experiment_metrics
    
    def identify_significant_differences(self, 
                                        baseline_experiment: Optional[str] = None, 
                                        threshold: float = 0.1) -> Dict[str, Dict[str, bool]]:
        """
        Identify significant differences between experiments.
        
        Args:
            baseline_experiment: Optional baseline experiment for comparison
            threshold: Threshold for significant relative difference
            
        Returns:
            Dictionary mapping experiments to metrics with significant differences
        """
        if not self.experiment_metrics:
            self.calculate_experiment_metrics()
            
        if not self.experiment_metrics:
            return {}
            
        # Determine baseline (use first experiment if not specified)
        if baseline_experiment is None or baseline_experiment not in self.experiment_metrics:
            baseline_experiment = next(iter(self.experiment_metrics))
        
        significant_differences = {}
        baseline_metrics = self.experiment_metrics[baseline_experiment]
        
        # Compare each experiment to the baseline
        for exp_name, metrics in self.experiment_metrics.items():
            if exp_name == baseline_experiment:
                continue
                
            exp_diffs = {}
            
            # Check each metric for significant differences
            for metric, value in metrics.items():
                if metric in baseline_metrics and not isinstance(value, dict):
                    baseline_value = baseline_metrics[metric]
                    
                    # Calculate relative difference
                    if baseline_value != 0:
                        rel_diff = abs(value / baseline_value - 1.0)
                        exp_diffs[metric] = rel_diff >= threshold
            
            # Check field-specific metrics
            if 'field_accuracies' in metrics and 'field_accuracies' in baseline_metrics:
                field_diff = {}
                
                # Find common fields
                common_fields = set(metrics['field_accuracies'].keys()) & set(baseline_metrics['field_accuracies'].keys())
                
                for field in common_fields:
                    field_acc = metrics['field_accuracies'][field]
                    baseline_field_acc = baseline_metrics['field_accuracies'][field]
                    
                    # Calculate relative difference
                    if baseline_field_acc != 0:
                        rel_diff = abs(field_acc / baseline_field_acc - 1.0)
                        field_diff[field] = rel_diff >= threshold
                
                exp_diffs['field_accuracies'] = field_diff
            
            significant_differences[exp_name] = exp_diffs
        
        self.significant_differences = significant_differences
        return significant_differences
    
    def analyze_performance_trends(self, 
                                  metric: str = 'overall_accuracy', 
                                  fields: Optional[List[str]] = None) -> Dict[str, List[float]]:
        """
        Analyze performance trends across experiments.
        
        Args:
            metric: Metric to analyze (default: overall_accuracy)
            fields: Optional list of fields to analyze
            
        Returns:
            Dictionary with performance trends
        """
        if not self.experiment_metrics:
            self.calculate_experiment_metrics()
            
        if not self.experiment_metrics:
            return {}
            
        trends = {}
        
        # Get experiment names in order (sorted by timestamp if available)
        if self.data:
            exp_names = sorted(
                self.data.keys(),
                key=lambda x: self.data[x].timestamp if hasattr(self.data[x], 'timestamp') else ''
            )
        else:
            exp_names = list(self.experiment_metrics.keys())
        
        # Extract trend for overall metric
        if metric != 'field_accuracies':
            overall_trend = [
                self.experiment_metrics[exp].get(metric, 0)
                for exp in exp_names
                if exp in self.experiment_metrics
            ]
            trends['overall'] = overall_trend
        
        # Extract field-specific trends if requested
        if fields and 'field_accuracies' in next(iter(self.experiment_metrics.values())):
            for field in fields:
                field_trend = []
                
                for exp in exp_names:
                    if exp in self.experiment_metrics and field in self.experiment_metrics[exp]['field_accuracies']:
                        field_trend.append(self.experiment_metrics[exp]['field_accuracies'][field])
                    else:
                        field_trend.append(0)
                
                if field_trend:
                    trends[field] = field_trend
        
        self.performance_trends = {'metric': metric, 'trends': trends}
        return trends
    
    def find_best_experiment(self, metric: str = 'overall_accuracy') -> Optional[str]:
        """
        Find the best performing experiment based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Name of the best performing experiment or None if not available
        """
        if not self.experiment_metrics:
            self.calculate_experiment_metrics()
            
        if not self.experiment_metrics:
            return None
            
        best_value = -float('inf')
        best_experiment = None
        
        for exp_name, metrics in self.experiment_metrics.items():
            if metric in metrics and not isinstance(metrics[metric], dict) and metrics[metric] > best_value:
                best_value = metrics[metric]
                best_experiment = exp_name
        
        return best_experiment
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of experiment comparison results.
        
        Returns:
            Dictionary with summary information
        """
        if not self.experiment_metrics:
            self.calculate_experiment_metrics()
            
        if not self.significant_differences:
            self.identify_significant_differences()
            
        # Find best experiment
        best_experiment = self.find_best_experiment()
        
        # Calculate overall improvement/regression
        summary = {
            'total_experiments': len(self.experiment_metrics),
            'best_experiment': best_experiment,
            'metrics_compared': list(next(iter(self.experiment_metrics.values())).keys()) 
                                if self.experiment_metrics else [],
            'significant_improvements': {},
            'significant_regressions': {}
        }
        
        # Extract significant improvements and regressions
        if self.significant_differences and self.experiment_metrics:
            baseline_exp = next(iter(self.significant_differences.keys()))
            baseline_metrics = self.experiment_metrics.get(baseline_exp, {})
            
            for exp_name, diffs in self.significant_differences.items():
                improvements = []
                regressions = []
                
                for metric, is_significant in diffs.items():
                    if not is_significant or isinstance(is_significant, dict):
                        continue
                        
                    exp_value = self.experiment_metrics.get(exp_name, {}).get(metric)
                    baseline_value = baseline_metrics.get(metric)
                    
                    if exp_value is not None and baseline_value is not None:
                        if metric == 'avg_field_error_rate':  # Lower is better
                            if exp_value < baseline_value:
                                improvements.append(metric)
                            else:
                                regressions.append(metric)
                        else:  # Higher is better for other metrics
                            if exp_value > baseline_value:
                                improvements.append(metric)
                            else:
                                regressions.append(metric)
                
                if improvements:
                    summary['significant_improvements'][exp_name] = improvements
                    
                if regressions:
                    summary['significant_regressions'][exp_name] = regressions
        
        return summary


class ComparisonStrategy(ABC):
    """
    Abstract base class for comparison strategies.
    
    Defines interface for different types of comparisons.
    """
    
    @abstractmethod
    def compare(self, data: Any, context: ComparisonContext) -> ComparisonResult:
        """
        Perform comparison on the provided data.
        
        Args:
            data: Data to compare
            context: Comparison context with configuration
            
        Returns:
            ComparisonResult with analysis results
        """
        pass
    
    @staticmethod
    def perform_statistical_test(
        dataset1: List[float],
        dataset2: List[float],
        test_type: str = "t_test",
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical significance testing between two datasets.
        
        Args:
            dataset1: First dataset
            dataset2: Second dataset
            test_type: Type of test to perform (t_test, mann_whitney)
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        results = {
            "test_type": test_type,
            "alpha": alpha,
            "significant": False,
            "p_value": None,
            "statistic": None
        }
        
        # Skip test if either dataset is too small
        if len(dataset1) < 2 or len(dataset2) < 2:
            results["error"] = "Datasets too small for statistical testing"
            return results
        
        try:
            if test_type == "t_test":
                # Perform two-sample t-test
                statistic, p_value = stats.ttest_ind(dataset1, dataset2, equal_var=False)
                
                results["statistic"] = float(statistic)
                results["p_value"] = float(p_value)
                results["significant"] = p_value < alpha
                
            elif test_type == "mann_whitney":
                # Perform Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(dataset1, dataset2)
                
                results["statistic"] = float(statistic)
                results["p_value"] = float(p_value)
                results["significant"] = p_value < alpha
                
            else:
                results["error"] = f"Unknown test type: {test_type}"
        
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    @staticmethod
    def calculate_effect_size(dataset1: List[float], dataset2: List[float]) -> Dict[str, float]:
        """
        Calculate effect size measures between two datasets.
        
        Args:
            dataset1: First dataset
            dataset2: Second dataset
            
        Returns:
            Dictionary with effect size measures
        """
        results = {}
        
        # Skip if either dataset is too small
        if len(dataset1) < 2 or len(dataset2) < 2:
            return {"error": "Datasets too small for effect size calculation"}
        
        try:
            # Cohen's d
            mean1 = np.mean(dataset1)
            mean2 = np.mean(dataset2)
            
            var1 = np.var(dataset1, ddof=1)
            var2 = np.var(dataset2, ddof=1)
            
            # Pooled standard deviation
            n1 = len(dataset1)
            n2 = len(dataset2)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            # Calculate Cohen's d
            if pooled_std > 0:
                cohen_d = (mean2 - mean1) / pooled_std
                results["cohen_d"] = float(cohen_d)
                
                # Interpret Cohen's d
                if abs(cohen_d) < 0.2:
                    results["effect_interpretation"] = "negligible"
                elif abs(cohen_d) < 0.5:
                    results["effect_interpretation"] = "small"
                elif abs(cohen_d) < 0.8:
                    results["effect_interpretation"] = "medium"
                else:
                    results["effect_interpretation"] = "large"
            
            # Percent difference
            if mean1 != 0:
                percent_diff = (mean2 - mean1) / abs(mean1) * 100
                results["percent_difference"] = float(percent_diff)
        
        except Exception as e:
            results["error"] = str(e)
        
        return results


class PromptComparisonStrategy(ComparisonStrategy):
    """
    Strategy for comparing different prompts for the same field.
    """
    
    def compare(self, data: Dict[str, List[Dict[str, Any]]], context: ComparisonContext) -> PromptComparisonResult:
        """
        Compare results from different prompts for the same field.
        
        Args:
            data: Dictionary mapping prompt names to lists of results
            context: Comparison context
            
        Returns:
            PromptComparisonResult with analysis results
        """
        # Initialize result object
        result = PromptComparisonResult(
            context=context,
            field=context.field or "",
            data=data
        )
        
        # Extract metrics for each prompt
        prompt_metrics = {}
        
        for prompt_name, prompt_results in data.items():
            metrics = {}
            
            # Calculate basic metrics
            metrics_to_calculate = set(context.metrics)
            
            # Extract values for each metric
            for metric in metrics_to_calculate:
                values = [r.get(metric, 0) for r in prompt_results if metric in r]
                
                if values:
                    metrics[metric] = float(np.mean(values))
                    metrics[f"{metric}_min"] = float(min(values))
                    metrics[f"{metric}_max"] = float(max(values))
                    metrics[f"{metric}_std"] = float(np.std(values))
            
            # Calculate success rate
            total_items = len(prompt_results)
            if total_items > 0:
                successful = sum(1 for r in prompt_results if r.get("exact_match", False))
                metrics["success_rate"] = successful / total_items
            
            prompt_metrics[prompt_name] = metrics
        
        # Store prompt metrics
        result.prompt_metrics = prompt_metrics
        
        # Find best and worst prompts
        result.determine_best_prompt("exact_match")
        result.determine_worst_prompt("exact_match")
        
        # Calculate performance gap
        performance_gap = result.calculate_performance_gap("exact_match")
        
        # Perform statistical testing
        if len(prompt_metrics) >= 2 and context.detailed_analysis:
            # Perform pairwise comparisons
            for metric in context.metrics:
                prompt_names = list(prompt_metrics.keys())
                
                for i, prompt1 in enumerate(prompt_names):
                    for prompt2 in prompt_names[i+1:]:
                        # Get values for each prompt
                        values1 = [r.get(metric, 0) for r in data[prompt1] if metric in r]
                        values2 = [r.get(metric, 0) for r in data[prompt2] if metric in r]
                        
                        # Skip if not enough data
                        if len(values1) < 2 or len(values2) < 2:
                            continue
                        
                        # Perform t-test
                        test_name = f"{prompt1}_vs_{prompt2}_{metric}"
                        test_result = self.perform_statistical_test(values1, values2)
                        
                        # Add effect size
                        effect_size = self.calculate_effect_size(values1, values2)
                        test_result.update(effect_size)
                        
                        # Add to results
                        result.add_statistical_test(test_name, test_result)
        
        # Add metrics to results
        for prompt_name, metrics in prompt_metrics.items():
            result.add_metric(prompt_name, metrics)
        
        # Generate insights
        self._generate_insights(result)
        
        return result
    
    def _generate_insights(self, result: PromptComparisonResult) -> None:
        """
        Generate insights from the comparison results.
        
        Args:
            result: Comparison result to analyze
        """
        # Add basic insights
        if result.best_prompt:
            best_metrics = result.prompt_metrics[result.best_prompt]
            if "exact_match" in best_metrics:
                result.add_insight(
                    f"Prompt '{result.best_prompt}' performs best with {best_metrics['exact_match']:.2%} accuracy",
                    category="performance"
                )
        
        if result.worst_prompt:
            worst_metrics = result.prompt_metrics[result.worst_prompt]
            if "exact_match" in worst_metrics and worst_metrics['exact_match'] < 0.5:
                result.add_insight(
                    f"Prompt '{result.worst_prompt}' performs poorly with only {worst_metrics['exact_match']:.2%} accuracy",
                    category="performance"
                )
        
        # Add insights about statistical significance
        for test_name, test_result in result.statistical_tests.items():
            if "error" in test_result:
                continue
                
            if test_result.get("significant", False):
                # Parse test name to get prompts and metric
                parts = test_name.split("_vs_")
                if len(parts) >= 2:
                    prompt1 = parts[0]
                    remaining = parts[1].rsplit("_", 1)
                    if len(remaining) == 2:
                        prompt2, metric = remaining
                        
                        result.add_insight(
                            f"Significant difference in {metric} between prompts '{prompt1}' and '{prompt2}' (p={test_result['p_value']:.4f})",
                            category="statistical"
                        )
        
        # Add insights about performance gap
        performance_gap = result.calculate_performance_gap("exact_match")
        if performance_gap > 1.5:
            result.add_insight(
                f"Large performance gap between best and worst prompts ({performance_gap:.1f}x difference)",
                category="performance"
            )
        
        # Add recommendations
        if result.best_prompt:
            result.add_insight(
                f"Consider using '{result.best_prompt}' as the primary prompt for {result.field}",
                category="recommendation"
            )


class ModelComparisonStrategy(ComparisonStrategy):
    """
    Strategy for comparing different models.
    """
    
    def compare(self, data: Dict[str, Dict[str, Any]], context: ComparisonContext) -> ModelComparisonResult:
        """
        Compare results from different models.
        
        Args:
            data: Dictionary mapping model names to results
            context: Comparison context
            
        Returns:
            ModelComparisonResult with analysis results
        """
        # Initialize result object
        result = ModelComparisonResult(
            context=context,
            data=data
        )
        
        # Extract metrics for each model
        model_metrics = {}
        
        for model_name, model_data in data.items():
            metrics = {}
            
            # Extract field results
            field_results = {}
            for field, field_data in model_data.items():
                if context.field and field != context.field:
                    continue
                    
                field_results[field] = field_data
            
            # Calculate metrics across fields
            for metric in context.metrics:
                values = []
                
                for field, field_data in field_results.items():
                    # Look for field-level metrics
                    if metric in field_data:
                        values.append(field_data[metric])
                    
                    # Look for prompt-level metrics
                    for prompt, prompt_data in field_data.get("prompt_results", {}).items():
                        if metric in prompt_data:
                            values.append(prompt_data[metric])
                
                if values:
                    metrics[metric] = float(np.mean(values))
                    metrics[f"{metric}_min"] = float(min(values))
                    metrics[f"{metric}_max"] = float(max(values))
                    metrics[f"{metric}_std"] = float(np.std(values))
            
            # Add overall metrics
            if "success_rate" in model_data:
                metrics["success_rate"] = model_data["success_rate"]
            
            if "memory_usage" in model_data:
                metrics["memory_usage"] = model_data["memory_usage"]
            
            model_metrics[model_name] = metrics
        
        # Store model metrics
        result.model_metrics = model_metrics
        
        # Calculate speed-accuracy tradeoff
        result.calculate_speed_accuracy_tradeoff()
        
        # Identify optimal model for different priorities
        optimal_balanced = result.get_optimal_model("balanced")
        optimal_speed = result.get_optimal_model("speed")
        optimal_accuracy = result.get_optimal_model("accuracy")
        
        # Add metrics to results
        for model_name, metrics in model_metrics.items():
            result.add_metric(model_name, metrics)
        
        # Generate insights
        self._generate_insights(result)
        
        return result
    
    def _generate_insights(self, result: ModelComparisonResult) -> None:
        """
        Generate insights from the comparison results.
        
        Args:
            result: Comparison result to analyze
        """
        # Add basic insights about optimal models
        optimal_balanced = result.get_optimal_model("balanced")
        if optimal_balanced:
            result.add_insight(
                f"Model '{optimal_balanced}' provides the best balance of speed and accuracy",
                category="performance"
            )
        
        optimal_speed = result.get_optimal_model("speed")
        if optimal_speed and optimal_speed != optimal_balanced:
            result.add_insight(
                f"Model '{optimal_speed}' provides the fastest processing speed",
                category="performance"
            )
        
        optimal_accuracy = result.get_optimal_model("accuracy")
        if optimal_accuracy and optimal_accuracy != optimal_balanced:
            result.add_insight(
                f"Model '{optimal_accuracy}' provides the highest accuracy",
                category="performance"
            )
        
        # Add insights about performance tradeoffs
        for model, tradeoffs in result.performance_tradeoffs.items():
            # Check for models with greatly imbalanced performance
            if tradeoffs["accuracy"] > 0.8 and tradeoffs["speed"] < 0.3:
                result.add_insight(
                    f"Model '{model}' has high accuracy but slow processing speed",
                    category="tradeoff"
                )
            elif tradeoffs["accuracy"] < 0.5 and tradeoffs["speed"] > 0.8:
                result.add_insight(
                    f"Model '{model}' has fast processing speed but low accuracy",
                    category="tradeoff"
                )
        
        # Add recommendations
        if optimal_balanced:
            result.add_insight(
                f"Consider using '{optimal_balanced}' as the primary model for general use cases",
                category="recommendation"
            )
        
        if optimal_speed and optimal_speed != optimal_balanced:
            result.add_insight(
                f"Consider using '{optimal_speed}' for time-sensitive applications where speed is critical",
                category="recommendation"
            )
        
        if optimal_accuracy and optimal_accuracy != optimal_balanced:
            result.add_insight(
                f"Consider using '{optimal_accuracy}' for high-precision tasks where accuracy is critical",
                category="recommendation"
            )


class QuantizationComparisonStrategy(ComparisonStrategy):
    """
    Strategy for comparing different quantization strategies.
    """
    
    def compare(self, data: Dict[str, Dict[str, Any]], context: ComparisonContext) -> QuantizationComparisonResult:
        """
        Compare results from different quantization strategies.
        
        Args:
            data: Dictionary mapping quantization strategy names to results
            context: Comparison context
            
        Returns:
            QuantizationComparisonResult with analysis results
        """
        # Initialize result object
        result = QuantizationComparisonResult(
            context=context,
            data=data
        )
        
        # Extract metrics for each strategy
        strategy_metrics = {}
        
        for strategy_name, strategy_data in data.items():
            metrics = {}
            
            # Extract metrics
            for metric in context.metrics:
                if metric in strategy_data:
                    metrics[metric] = strategy_data[metric]
            
            # Add memory usage if available
            if "memory_usage" in strategy_data:
                metrics["memory_usage"] = strategy_data["memory_usage"]
            
            # Add success rate if available
            if "success_rate" in strategy_data:
                metrics["success_rate"] = strategy_data["success_rate"]
            elif "exact_match" in strategy_data:
                metrics["success_rate"] = strategy_data["exact_match"]
            
            strategy_metrics[strategy_name] = metrics
        
        # Store strategy metrics
        result.strategy_metrics = strategy_metrics
        
        # Calculate impacts
        result.calculate_memory_impact()
        result.calculate_accuracy_impact()
        result.calculate_speed_impact()
        
        # Get optimal strategy for different priorities
        optimal_balanced = result.get_optimal_strategy("balanced")
        optimal_memory = result.get_optimal_strategy("memory")
        optimal_speed = result.get_optimal_strategy("speed")
        optimal_accuracy = result.get_optimal_strategy("accuracy")
        
        # Add metrics to results
        for strategy_name, metrics in strategy_metrics.items():
            result.add_metric(strategy_name, metrics)
        
        # Generate insights
        self._generate_insights(result)
        
        return result
    
    def _generate_insights(self, result: QuantizationComparisonResult) -> None:
        """
        Generate insights from the comparison results.
        
        Args:
            result: Comparison result to analyze
        """
        # Add basic insights about optimal strategies
        optimal_balanced = result.get_optimal_strategy("balanced")
        if optimal_balanced:
            result.add_insight(
                f"Strategy '{optimal_balanced}' provides the best balance of memory savings, speed, and accuracy",
                category="performance"
            )
        
        optimal_memory = result.get_optimal_strategy("memory")
        if optimal_memory and optimal_memory != optimal_balanced:
            result.add_insight(
                f"Strategy '{optimal_memory}' provides the highest memory savings",
                category="performance"
            )
        
        # Add insights about memory impact
        for strategy, impact in result.memory_reduction.items():
            if impact > 0.5:
                result.add_insight(
                    f"Strategy '{strategy}' reduces memory usage by {impact:.1%}",
                    category="memory"
                )
        
        # Add insights about accuracy impact
        for strategy, impact in result.accuracy_impact.items():
            if impact < -0.1:
                result.add_insight(
                    f"Strategy '{strategy}' reduces accuracy by {-impact:.1%}",
                    category="accuracy"
                )
            elif impact > 0.05:
                result.add_insight(
                    f"Strategy '{strategy}' surprisingly improves accuracy by {impact:.1%}",
                    category="accuracy"
                )
        
        # Add insights about speed impact
        for strategy, impact in result.speed_impact.items():
            if impact > 0.3:
                result.add_insight(
                    f"Strategy '{strategy}' improves processing speed by {impact:.1%}",
                    category="speed"
                )
        
        # Add recommendations
        if optimal_balanced:
            result.add_insight(
                f"Consider using '{optimal_balanced}' as the primary quantization strategy for general use cases",
                category="recommendation"
            )
        
        if optimal_memory and optimal_memory != optimal_balanced:
            result.add_insight(
                f"Consider using '{optimal_memory}' for memory-constrained environments",
                category="recommendation"
            )
        
        # Add hardware-specific recommendations
        for strategy, metrics in result.strategy_metrics.items():
            if "memory_usage" in metrics and metrics["memory_usage"] < 4.0:
                result.add_insight(
                    f"Strategy '{strategy}' enables deployment on devices with as little as {metrics['memory_usage']:.1f}GB GPU memory",
                    category="hardware"
                )


class RunComparisonStrategy(ComparisonStrategy):
    """
    Strategy for comparing different experiment runs.
    """
    
    def compare(self, data: List[Dict[str, Any]], context: ComparisonContext) -> RunComparisonResult:
        """
        Compare results from different experiment runs.
        
        Args:
            data: List of experiment result dictionaries
            context: Comparison context
            
        Returns:
            RunComparisonResult with analysis results
        """
        # Initialize result object
        result = RunComparisonResult(
            context=context,
            data={run.get("experiment_name", f"Run_{i}"): run for i, run in enumerate(data)}
        )
        
        # Extract metrics for each run
        run_metrics = {}
        
        for run_data in data:
            run_name = run_data.get("experiment_name", f"Run_{len(run_metrics)}")
            metrics = {}
            
            # Extract overall metrics
            if "overall_accuracy" in run_data:
                metrics["overall_accuracy"] = run_data["overall_accuracy"]
            
            if "total_execution_time" in run_data:
                metrics["total_execution_time"] = run_data["total_execution_time"]
            
            # Extract field-specific metrics if context.field is specified
            if context.field and "field_results" in run_data:
                field_results = run_data["field_results"].get(context.field, [])
                
                # Calculate average metrics across prompts
                for field_performance in field_results:
                    for metric in context.metrics:
                        if metric in field_performance:
                            metrics[f"{context.field}_{metric}"] = field_performance[metric]
            
            # Extract metrics for specific fields
            if context.fields and "field_results" in run_data:
                for field in context.fields:
                    field_results = run_data["field_results"].get(field, [])
                    
                    # Calculate average metrics across prompts
                    for field_performance in field_results:
                        for metric in context.metrics:
                            if metric in field_performance:
                                metrics[f"{field}_{metric}"] = field_performance[metric]
            
            run_metrics[run_name] = metrics
        
        # Store run metrics
        result.run_metrics = run_metrics
        
        # Calculate differences between runs
        result.calculate_differences()
        
        # Identify significant changes
        significant_changes = result.identify_significant_changes()
        
        # Add metrics to results
        for run_name, metrics in run_metrics.items():
            result.add_metric(run_name, metrics)
        
        # Generate insights
        self._generate_insights(result, significant_changes)
        
        return result
    
    def _generate_insights(self, result: RunComparisonResult, significant_changes: Dict[str, List[str]]) -> None:
        """
        Generate insights from the comparison results.
        
        Args:
            result: Comparison result to analyze
            significant_changes: Dictionary of significant changes by run
        """
        # Add insights about significant changes
        for run, changes in significant_changes.items():
            if changes:
                change_list = ", ".join(changes[:3])
                if len(changes) > 3:
                    change_list += f", and {len(changes) - 3} more"
                    
                result.add_insight(
                    f"Run '{run}' shows significant changes in {change_list}",
                    category="changes"
                )
        
        # Add insights about performance trends
        if len(result.run_metrics) >= 2:
            run_names = list(result.run_metrics.keys())
            
            # Look for consistent improvements
            improving_metrics = []
            for metric in result.context.metrics:
                values = []
                for run in run_names:
                    metrics = result.run_metrics[run]
                    if metric in metrics:
                        values.append(metrics[metric])
                
                if len(values) >= 2:
                    # Check if values are consistently increasing
                    if all(values[i] < values[i+1] for i in range(len(values)-1)):
                        improving_metrics.append(metric)
            
            if improving_metrics:
                metrics_list = ", ".join(improving_metrics)
                result.add_insight(
                    f"Consistent improvement observed in {metrics_list} across runs",
                    category="trend"
                )
        
        # Add recommendations
        if significant_changes:
            result.add_insight(
                "Consider investigating the factors contributing to the significant changes between runs",
                category="recommendation"
            )


class CrossFieldComparisonStrategy(ComparisonStrategy):
    """
    Strategy for comparing performance across different fields.
    """
    
    def compare(self, data: Dict[str, Dict[str, Any]], context: ComparisonContext) -> CrossFieldComparisonResult:
        """
        Compare performance across different fields.
        
        Args:
            data: Dictionary mapping field names to field results
            context: Comparison context
            
        Returns:
            CrossFieldComparisonResult with analysis results
        """
        # Initialize result object
        result = CrossFieldComparisonResult(
            context=context,
            data=data
        )
        
        # Extract metrics for each field
        field_metrics = {}
        
        for field, field_data in data.items():
            if context.fields and field not in context.fields:
                continue
                
            metrics = {}
            
            # Calculate average metrics across all prompts
            all_values = {}
            for prompt, prompt_data in field_data.get("prompt_results", {}).items():
                for metric in context.metrics:
                    if metric not in all_values:
                        all_values[metric] = []
                    
                    if metric in prompt_data:
                        all_values[metric].append(prompt_data[metric])
            
            # Calculate aggregate metrics
            for metric, values in all_values.items():
                if values:
                    metrics[metric] = float(np.mean(values))
                    metrics[f"{metric}_min"] = float(min(values))
                    metrics[f"{metric}_max"] = float(max(values))
                    metrics[f"{metric}_std"] = float(np.std(values))
            
            # Find best prompt for this field
            best_prompt = None
            best_value = -float('inf')
            
            for prompt, prompt_data in field_data.get("prompt_results", {}).items():
                if "exact_match" in prompt_data and prompt_data["exact_match"] > best_value:
                    best_value = prompt_data["exact_match"]
                    best_prompt = prompt
            
            if best_prompt:
                metrics["best_prompt"] = best_prompt
                metrics["best_exact_match"] = best_value
            
            field_metrics[field] = metrics
        
        # Store field metrics
        result.field_metrics = field_metrics
        
        # Calculate field correlations
        result.calculate_field_correlations()
        
        # Find outlier fields
        for metric in context.metrics:
            outliers = result.find_outlier_fields(metric)
            
            if outliers:
                outlier_key = f"outliers_{metric}"
                result.metrics[outlier_key] = outliers
        
        # Add metrics to results
        for field, metrics in field_metrics.items():
            result.add_metric(field, metrics)
        
        # Generate insights
        self._generate_insights(result)
        
        return result
    
    def _generate_insights(self, result: CrossFieldComparisonResult) -> None:
        """
        Generate insights from the comparison results.
        
        Args:
            result: Comparison result to analyze
        """
        # Add insights about field performance
        if result.field_metrics:
            # Find best and worst performing fields
            best_field = None
            best_value = -float('inf')
            worst_field = None
            worst_value = float('inf')
            
            for field, metrics in result.field_metrics.items():
                if "exact_match" in metrics:
                    if metrics["exact_match"] > best_value:
                        best_value = metrics["exact_match"]
                        best_field = field
                    
                    if metrics["exact_match"] < worst_value:
                        worst_value = metrics["exact_match"]
                        worst_field = field
            
            if best_field:
                result.add_insight(
                    f"Field '{best_field}' has the highest extraction accuracy at {best_value:.2%}",
                    category="performance"
                )
            
            if worst_field and worst_field != best_field:
                result.add_insight(
                    f"Field '{worst_field}' has the lowest extraction accuracy at {worst_value:.2%}",
                    category="performance"
                )
            
            # Add insights about performance gap
            if best_field and worst_field and best_field != worst_field:
                performance_gap = best_value / worst_value if worst_value > 0 else float('inf')
                
                if performance_gap > 1.5:
                    result.add_insight(
                        f"Large performance gap between fields ({performance_gap:.1f}x difference)",
                        category="performance"
                    )
        
        # Add insights about outliers
        for metric_key, outliers in result.metrics.items():
            if metric_key.startswith("outliers_") and outliers:
                base_metric = metric_key.replace("outliers_", "")
                outlier_list = ", ".join(outliers)
                
                result.add_insight(
                    f"Fields {outlier_list} are outliers in {base_metric} performance",
                    category="outliers"
                )
        
        # Add insights about correlations
        for metric, correlations in result.correlations.items():
            for pair, value in correlations.items():
                if abs(value - 1.0) < 0.1:
                    fields = pair.split("/")
                    
                    result.add_insight(
                        f"Fields {fields[0]} and {fields[1]} show very similar {metric} performance",
                        category="correlation"
                    )
                elif value > 2.0 or value < 0.5:
                    fields = pair.split("/")
                    
                    result.add_insight(
                        f"Fields {fields[0]} and {fields[1]} show significantly different {metric} performance",
                        category="correlation"
                    )
        
        # Add recommendations
        for field, metrics in result.field_metrics.items():
            if "best_prompt" in metrics and "exact_match" in metrics and metrics["exact_match"] < 0.5:
                result.add_insight(
                    f"Consider improving prompts for field '{field}' to increase accuracy",
                    category="recommendation"
                )
            
            if "exact_match_std" in metrics and metrics["exact_match_std"] > 0.2:
                result.add_insight(
                    f"Field '{field}' shows high variability across prompts, suggesting prompt engineering opportunities",
                    category="recommendation"
                )


class ExperimentComparisonStrategy(ComparisonStrategy):
    """
    Strategy for comparing multiple experiments.
    """
    
    def compare(self, data: Dict[str, ExperimentResult], context: ComparisonContext) -> ExperimentComparisonResult:
        """
        Compare multiple experiments.
        
        Args:
            data: Dictionary mapping experiment names to ExperimentResult instances
            context: Comparison context
            
        Returns:
            ExperimentComparisonResult instance
        """
        # Create comparison result
        result = ExperimentComparisonResult(context=context, data=data)
        
        # Calculate experiment metrics
        experiment_metrics = result.calculate_experiment_metrics()
        
        # Identify significant differences
        significant_differences = result.identify_significant_differences(
            baseline_experiment=context.run_ids[0] if context.run_ids and len(context.run_ids) > 0 else None,
            threshold=context.significance_level
        )
        
        # Analyze performance trends
        fields_to_analyze = context.fields or []
        if not fields_to_analyze and data:
            # Use fields from the first experiment
            first_exp = next(iter(data.values()))
            fields_to_analyze = list(first_exp.field_results.keys())
        
        result.analyze_performance_trends(
            metric='overall_accuracy',
            fields=fields_to_analyze
        )
        
        # Generate insights
        self._generate_insights(result)
        
        return result
    
    def _generate_insights(self, result: ExperimentComparisonResult) -> None:
        """
        Generate insights from experiment comparison results.
        
        Args:
            result: ExperimentComparisonResult instance
        """
        if not result.experiment_metrics:
            return
            
        # Find best experiment
        best_experiment = result.find_best_experiment()
        if best_experiment:
            result.add_insight(
                f"The experiment '{best_experiment}' achieved the highest overall accuracy "
                f"of {result.experiment_metrics[best_experiment]['overall_accuracy']:.2%}.",
                category="performance"
            )
        
        # Find most improved fields
        improved_fields = {}
        baseline_exp = next(iter(result.experiment_metrics.keys()))
        
        for exp_name, metrics in result.experiment_metrics.items():
            if exp_name == baseline_exp:
                continue
                
            if 'field_accuracies' in metrics and 'field_accuracies' in result.experiment_metrics[baseline_exp]:
                for field, accuracy in metrics['field_accuracies'].items():
                    if field in result.experiment_metrics[baseline_exp]['field_accuracies']:
                        baseline_accuracy = result.experiment_metrics[baseline_exp]['field_accuracies'][field]
                        if baseline_accuracy > 0:
                            improvement = (accuracy / baseline_accuracy) - 1.0
                            if improvement > 0:
                                if field not in improved_fields or improvement > improved_fields[field][1]:
                                    improved_fields[field] = (exp_name, improvement)
        
        # Add insights for top 3 most improved fields
        if improved_fields:
            top_improved = sorted(improved_fields.items(), key=lambda x: x[1][1], reverse=True)[:3]
            for field, (exp_name, improvement) in top_improved:
                result.add_insight(
                    f"The field '{field}' showed significant improvement of {improvement:.2%} "
                    f"in experiment '{exp_name}'.",
                    category="field_improvements"
                )
        
        # Analyze performance stability across fields
        for exp_name, metrics in result.experiment_metrics.items():
            if 'field_accuracies' in metrics and metrics['field_accuracies']:
                accuracies = list(metrics['field_accuracies'].values())
                if len(accuracies) >= 3:  # Need at least 3 fields for meaningful analysis
                    std_dev = np.std(accuracies)
                    cv = std_dev / np.mean(accuracies) if np.mean(accuracies) > 0 else 0
                    
                    if cv < 0.1:
                        result.add_insight(
                            f"Experiment '{exp_name}' shows consistent performance across all fields "
                            f"(coefficient of variation: {cv:.2f}).",
                            category="stability"
                        )
                    elif cv > 0.3:
                        result.add_insight(
                            f"Experiment '{exp_name}' shows high variability in performance across fields "
                            f"(coefficient of variation: {cv:.2f}). Consider investigating field-specific issues.",
                            category="stability"
                        )
        
        # Analyze processing time trends
        if result.performance_trends and 'metric' in result.performance_trends:
            if 'trends' in result.performance_trends and 'overall' in result.performance_trends['trends']:
                trend = result.performance_trends['trends']['overall']
                if len(trend) >= 3:  # Need at least 3 experiments for trend analysis
                    if trend[-1] > trend[0] * 1.2:
                        result.add_insight(
                            f"Overall performance shows a positive trend with {((trend[-1]/trend[0])-1):.2%} "
                            f"improvement from first to last experiment.",
                            category="trends"
                        )
                    elif trend[-1] < trend[0] * 0.8:
                        result.add_insight(
                            f"Overall performance shows a negative trend with {(1-(trend[-1]/trend[0])):.2%} "
                            f"decline from first to last experiment.",
                            category="trends"
                        )
        
        # Add metadata insights
        if result.data:
            # Count unique models
            models = set(exp.model_name for exp in result.data.values() if exp.model_name)
            if len(models) > 1:
                result.add_insight(
                    f"The comparison includes {len(models)} different models: {', '.join(models)}.",
                    category="metadata"
                )
            
            # Count unique prompt strategies
            strategies = set(exp.prompt_strategy for exp in result.data.values() if exp.prompt_strategy)
            if len(strategies) > 1:
                result.add_insight(
                    f"The comparison includes {len(strategies)} different prompt strategies: {', '.join(strategies)}.",
                    category="metadata"
                )


class ResultComparer:
    """
    Main class for comparing extraction results with various strategies.
    """
    
    def __init__(self):
        """Initialize the result comparer with available strategies."""
        self.strategies = {
            ComparisonType.PROMPT: PromptComparisonStrategy(),
            ComparisonType.MODEL: ModelComparisonStrategy(),
            ComparisonType.QUANTIZATION: QuantizationComparisonStrategy(),
            ComparisonType.RUN: RunComparisonStrategy(),
            ComparisonType.CROSS_FIELD: CrossFieldComparisonStrategy()
        }
    
    def compare_prompts(
        self, 
        field: str, 
        prompt_results: Dict[str, List[Dict[str, Any]]],
        metrics: Optional[List[str]] = None
    ) -> PromptComparisonResult:
        """
        Compare results from different prompts for the same field.
        
        Args:
            field: Field being compared
            prompt_results: Dictionary mapping prompt names to lists of results
            metrics: Optional list of metrics to compare
            
        Returns:
            PromptComparisonResult with analysis
        """
        # Create comparison context
        context = ComparisonContext(
            comparison_type=ComparisonType.PROMPT,
            field=field,
            prompts=list(prompt_results.keys()),
            metrics=metrics or [
                ComparisonMetric.EXACT_MATCH.value,
                ComparisonMetric.CHARACTER_ERROR_RATE.value,
                ComparisonMetric.PROCESSING_TIME.value
            ]
        )
        
        # Get appropriate strategy
        strategy = self.strategies[ComparisonType.PROMPT]
        
        # Perform comparison
        return strategy.compare(prompt_results, context)
    
    def compare_models(
        self, 
        model_results: Dict[str, Dict[str, Any]],
        field: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> ModelComparisonResult:
        """
        Compare results from different models.
        
        Args:
            model_results: Dictionary mapping model names to results
            field: Optional field to focus comparison on
            metrics: Optional list of metrics to compare
            
        Returns:
            ModelComparisonResult with analysis
        """
        # Create comparison context
        context = ComparisonContext(
            comparison_type=ComparisonType.MODEL,
            field=field,
            models=list(model_results.keys()),
            metrics=metrics or [
                ComparisonMetric.EXACT_MATCH.value,
                ComparisonMetric.PROCESSING_TIME.value,
                ComparisonMetric.MEMORY_USAGE.value
            ]
        )
        
        # Get appropriate strategy
        strategy = self.strategies[ComparisonType.MODEL]
        
        # Perform comparison
        return strategy.compare(model_results, context)
    
    def compare_quantization(
        self, 
        quantization_results: Dict[str, Dict[str, Any]],
        model_name: str,
        metrics: Optional[List[str]] = None
    ) -> QuantizationComparisonResult:
        """
        Compare results from different quantization strategies.
        
        Args:
            quantization_results: Dictionary mapping strategy names to results
            model_name: Name of the model being quantized
            metrics: Optional list of metrics to compare
            
        Returns:
            QuantizationComparisonResult with analysis
        """
        # Create comparison context
        context = ComparisonContext(
            comparison_type=ComparisonType.QUANTIZATION,
            models=[model_name],
            quantization_strategies=list(quantization_results.keys()),
            metrics=metrics or [
                ComparisonMetric.EXACT_MATCH.value,
                ComparisonMetric.PROCESSING_TIME.value,
                ComparisonMetric.MEMORY_USAGE.value
            ]
        )
        
        # Get appropriate strategy
        strategy = self.strategies[ComparisonType.QUANTIZATION]
        
        # Perform comparison
        return strategy.compare(quantization_results, context)
    
    def compare_cross_field(
        self, 
        field_results: Dict[str, Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> CrossFieldComparisonResult:
        """
        Compare performance across different fields.
        
        Args:
            field_results: Dictionary mapping field names to performance metrics
            metrics: Optional list of metrics to compare
            
        Returns:
            CrossFieldComparisonResult instance
        """
        # Create comparison context
        context = ComparisonContext(
            comparison_type=ComparisonType.CROSS_FIELD,
            fields=list(field_results.keys()),
            metrics=metrics or [
                ComparisonMetric.EXACT_MATCH.value,
                ComparisonMetric.CHARACTER_ERROR_RATE.value,
                ComparisonMetric.PROCESSING_TIME.value
            ]
        )
        
        # Create strategy and perform comparison
        strategy = CrossFieldComparisonStrategy()
        result = strategy.compare(field_results, context)
        
        return result
        
    def compare_experiments(
        self, 
        experiments: Dict[str, ExperimentResult],
        metrics: Optional[List[str]] = None,
        fields: Optional[List[str]] = None
    ) -> ExperimentComparisonResult:
        """
        Compare multiple experiments.
        
        Args:
            experiments: Dictionary mapping experiment names to ExperimentResult instances
            metrics: Optional list of metrics to compare
            fields: Optional list of fields to include in the comparison
            
        Returns:
            ExperimentComparisonResult instance
        """
        # Get experiment names
        experiment_names = list(experiments.keys())
        
        # Create comparison context
        context = ComparisonContext(
            comparison_type=ComparisonType.RUN,  # We use RUN type for experiments
            run_ids=experiment_names,
            fields=fields,
            metrics=metrics or [
                ComparisonMetric.EXACT_MATCH.value,
                ComparisonMetric.CHARACTER_ERROR_RATE.value,
                ComparisonMetric.PROCESSING_TIME.value
            ]
        )
        
        # Create strategy and perform comparison
        strategy = ExperimentComparisonStrategy()
        result = strategy.compare(experiments, context)
        
        return result
        
    def load_and_compare_experiments(
        self,
        experiment_files: List[str],
        metrics: Optional[List[str]] = None,
        fields: Optional[List[str]] = None
    ) -> ExperimentComparisonResult:
        """
        Load experiment results from files and compare them.
        
        Args:
            experiment_files: List of paths to experiment result files
            metrics: Optional list of metrics to compare
            fields: Optional list of fields to include in the comparison
            
        Returns:
            ExperimentComparisonResult instance
        """
        experiments = {}
        
        # Load experiment results
        for file_path in experiment_files:
            try:
                experiment = ExperimentResult.load_from_file(file_path)
                # Use experiment name from file or generate from path
                name = experiment.experiment_name or os.path.basename(file_path).split('.')[0]
                experiments[name] = experiment
            except Exception as e:
                logger.error(f"Error loading experiment from {file_path}: {e}")
        
        if not experiments:
            raise ValueError("No valid experiment results could be loaded.")
        
        # Compare experiments
        return self.compare_experiments(experiments, metrics, fields)
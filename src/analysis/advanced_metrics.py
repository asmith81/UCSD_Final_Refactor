"""
Advanced Metrics Calculation for Invoice Information Extraction

Provides comprehensive metrics analysis with statistical methods
to evaluate extraction performance across different prompts and models.
"""

import logging
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)


class AdvancedMetricsCalculator:
    """
    Comprehensive metrics calculator with advanced statistical analysis.
    """
    
    @staticmethod
    def calculate_basic_statistics(values: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistical measures for a list of values.
        
        Args:
            values: List of numeric values
        
        Returns:
            Dictionary with statistical metrics
        """
        if not values:
            return {
                "count": 0,
                "mean": float('nan'),
                "median": float('nan'),
                "std_dev": float('nan'),
                "min": float('nan'),
                "max": float('nan')
            }
        
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std_dev": float(np.std(values)),
            "min": float(min(values)),
            "max": float(max(values))
        }
    
    @staticmethod
    def calculate_confidence_interval(
        values: List[float], 
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate confidence interval for a list of values.
        
        Args:
            values: List of numeric values
            confidence_level: Confidence level for interval (default 95%)
        
        Returns:
            Dictionary with confidence interval details
        """
        if len(values) < 2:
            return {
                "confidence_level": confidence_level,
                "lower_bound": float('nan'),
                "upper_bound": float('nan'),
                "standard_error": float('nan')
            }
        
        # Calculate statistics
        mean = np.mean(values)
        std_dev = np.std(values, ddof=1)  # Sample standard deviation
        standard_error = std_dev / np.sqrt(len(values))
        
        # Calculate confidence interval
        degrees_of_freedom = len(values) - 1
        t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        
        margin_of_error = t_value * standard_error
        
        return {
            "confidence_level": confidence_level,
            "lower_bound": mean - margin_of_error,
            "upper_bound": mean + margin_of_error,
            "standard_error": standard_error
        }
    
    @staticmethod
    def compare_prompt_performance(
        prompt1_results: List[float], 
        prompt2_results: List[float], 
        metric: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Statistically compare performance between two prompts.
        
        Args:
            prompt1_results: List of results for first prompt
            prompt2_results: List of results for second prompt
            metric: Metric to compare (accuracy, character_error_rate)
        
        Returns:
            Dictionary with comparison results
        """
        # Statistical significance test (t-test)
        try:
            t_statistic, p_value = stats.ttest_ind(prompt1_results, prompt2_results)
            
            return {
                "metric": metric,
                "prompt1_stats": AdvancedMetricsCalculator.calculate_basic_statistics(prompt1_results),
                "prompt2_stats": AdvancedMetricsCalculator.calculate_basic_statistics(prompt2_results),
                "t_statistic": float(t_statistic),
                "p_value": float(p_value),
                "statistically_significant": p_value < 0.05,
                "interpretation": (
                    "Significant difference" if p_value < 0.05 else 
                    "No significant difference"
                )
            }
        except Exception as e:
            logger.error(f"Error in statistical comparison: {e}")
            return {
                "error": str(e)
            }
    
    @staticmethod
    def calculate_extraction_performance_metrics(
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for extraction results.
        
        Args:
            results: List of individual extraction results
        
        Returns:
            Dictionary with performance metrics
        """
        # Separate results by status and metric
        statuses = {}
        metrics = {
            "exact_match_rates": [],
            "character_error_rates": [],
            "processing_times": []
        }
        
        for result in results:
            # Aggregate statuses
            status = result.get('status', 'unknown')
            statuses[status] = statuses.get(status, 0) + 1
            
            # Collect metrics
            metrics['exact_match_rates'].append(result.get('exact_match', 0))
            metrics['character_error_rates'].append(result.get('character_error_rate', 1.0))
            metrics['processing_times'].append(result.get('processing_time', 0))
        
        # Calculate performance metrics
        performance_metrics = {
            "total_extractions": len(results),
            "status_distribution": statuses,
            "metrics": {
                metric_name: AdvancedMetricsCalculator.calculate_basic_statistics(metric_values)
                for metric_name, metric_values in metrics.items()
            },
            "confidence_intervals": {
                metric_name: AdvancedMetricsCalculator.calculate_confidence_interval(metric_values)
                for metric_name, metric_values in metrics.items()
            }
        }
        
        return performance_metrics
    
    @staticmethod
    def analyze_extraction_errors(
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze extraction errors to identify patterns and insights.
        
        Args:
            results: List of individual extraction results
        
        Returns:
            Dictionary with error analysis details
        """
        # Categorize errors
        error_categories = {}
        error_details = []
        
        for result in results:
            if 'error' in result or result.get('status', 'unknown') != 'success':
                error = result.get('error', 'Unknown error')
                error_categories[error] = error_categories.get(error, 0) + 1
                
                error_details.append({
                    'image_id': result.get('image_id', 'unknown'),
                    'error': error,
                    'ground_truth': result.get('ground_truth', ''),
                    'extracted_value': result.get('processed_extraction', '')
                })
        
        # Analyze error distribution
        error_analysis = {
            "total_errors": len(error_details),
            "error_categories": error_categories,
            "most_common_error": max(error_categories, key=error_categories.get) if error_categories else None,
            "error_details": error_details
        }
        
        return error_analysis
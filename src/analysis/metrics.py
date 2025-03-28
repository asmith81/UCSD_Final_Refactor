"""
Metrics calculation module for different field types.

This module provides:
- Base MetricsCalculator class for field-specific metrics
- Specialized implementations for work order and cost fields
- Factory function to create appropriate calculator instances
"""

import re
from typing import Dict, List, Any, Optional, Union
import logging

# Configure logger
logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Base class for field-specific metrics calculation.
    
    This abstract class defines the interface for calculating
    metrics for different field types, with common aggregation
    functionality.
    """
    
    def __init__(self, field_type: str, config: Dict[str, Any]):
        """Initialize with field type and metrics configuration.
        
        Args:
            field_type: Type of field (e.g., "work_order", "cost")
            config: Field-specific configuration dictionary
        """
        self.field_type = field_type
        self.config = config
        self.metrics_config = config.get("metrics", {})
    
    def calculate(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Calculate metrics for a prediction.
        
        Args:
            prediction: Extracted text from model
            ground_truth: Expected ground truth value
            
        Returns:
            Dictionary of metric names to values
            
        Note:
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement calculate()")
    
    def aggregate(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple predictions.
        
        Args:
            metrics_list: List of metric dictionaries from calculate()
            
        Returns:
            Dictionary of aggregated metrics with mean, min, max values
        """
        if not metrics_list:
            return {}
            
        result = {}
        for metric_name in metrics_list[0].keys():
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            if values:
                result[metric_name] = sum(values) / len(values)
                result[f"{metric_name}_min"] = min(values)
                result[f"{metric_name}_max"] = max(values)
        return result
    
    def preprocess(self, text: str) -> str:
        """Preprocess text for comparison (base implementation).
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if text is None:
            return ""
            
        # Convert to string and strip whitespace
        return str(text).strip()


class WorkOrderMetricsCalculator(MetricsCalculator):
    """Metrics calculator for work order field.
    
    Calculates metrics relevant for work order number extraction:
    - exact_match (boolean)
    - character_error_rate (float)
    - levenshtein_distance (integer)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with work order metrics configuration.
        
        Args:
            config: Work order field configuration dictionary
        """
        super().__init__("work_order", config)
    
    def preprocess(self, text: str) -> str:
        """Preprocess work order text for comparison.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text optimized for work order comparison
        """
        if text is None:
            return ""
            
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Apply field-specific preprocessing
        settings = self.metrics_config.get("settings", {})
        
        # Remove spaces if specified
        if settings.get("strip_whitespace", True):
            text = text.replace(" ", "")
        
        # Convert to lowercase if not case sensitive
        if not settings.get("case_sensitive", False):
            text = text.lower()
        
        # Extract only numeric portion if normalize_numbers is enabled
        if settings.get("normalize_numbers", True):
            numbers = re.findall(r'\d+', text)
            if numbers:
                text = numbers[-1]  # Use the last number found
        
        return text
    
    def calculate(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Calculate work order specific metrics.
        
        Args:
            prediction: Extracted work order number
            ground_truth: Expected work order number
            
        Returns:
            Dictionary with exact_match, character_error_rate, and levenshtein_distance
        """
        # Preprocess both texts
        proc_pred = self.preprocess(prediction)
        proc_gt = self.preprocess(ground_truth)
        
        # Initialize metrics
        metrics = {}
        
        # Calculate exact match (primary metric)
        exact_match = proc_pred == proc_gt
        metrics["exact_match"] = 1.0 if exact_match else 0.0
        
        # Calculate Levenshtein distance if ground truth is not empty
        if proc_gt:
            try:
                from Levenshtein import distance
                lev_distance = distance(proc_pred, proc_gt)
                metrics["levenshtein_distance"] = float(lev_distance)
                
                # Calculate character error rate (CER)
                metrics["character_error_rate"] = lev_distance / len(proc_gt)
            except ImportError:
                logger.warning("Levenshtein package not available - using fallback distance calculation")
                # Fallback to simplistic approach if Levenshtein not available
                metrics["levenshtein_distance"] = sum(1 for a, b in zip(proc_pred, proc_gt) if a != b)
                metrics["character_error_rate"] = metrics["levenshtein_distance"] / len(proc_gt) if proc_gt else 1.0
        else:
            metrics["levenshtein_distance"] = float(len(proc_pred)) if proc_pred else 0.0
            metrics["character_error_rate"] = 1.0 if proc_pred else 0.0
        
        return metrics


class CostMetricsCalculator(MetricsCalculator):
    """Metrics calculator for cost field.
    
    Calculates metrics relevant for cost extraction:
    - exact_match (boolean)
    - numeric_difference (float)
    - percentage_error (float)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with cost metrics configuration.
        
        Args:
            config: Cost field configuration dictionary
        """
        super().__init__("cost", config)
    
    def preprocess(self, text: str) -> str:
        """Preprocess cost value for comparison.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text optimized for cost comparison
        """
        if text is None:
            return ""
            
        # Convert to string and strip whitespace
        text = str(text).strip()
        
        # Apply field-specific preprocessing
        settings = self.metrics_config.get("settings", {})
        
        # Strip currency symbol if specified
        if settings.get("strip_currency", True):
            text = text.replace("$", "").replace("€", "").replace("£", "")
        
        # Remove commas if allowed
        if settings.get("allow_comma_separator", True):
            text = text.replace(",", "")
        
        # Convert to lowercase if not case sensitive
        if not settings.get("case_sensitive", False):
            text = text.lower()
        
        return text
    
    def extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract numeric value from text.
        
        Args:
            text: Preprocessed text to extract value from
            
        Returns:
            Float value or None if extraction fails
        """
        try:
            # Preprocess text
            processed_text = self.preprocess(text)
            
            # Try to extract a float value
            return float(processed_text)
        except (ValueError, TypeError):
            return None
    
    def calculate(self, prediction: str, ground_truth: str) -> Dict[str, float]:
        """Calculate cost specific metrics.
        
        Args:
            prediction: Extracted cost value
            ground_truth: Expected cost value
            
        Returns:
            Dictionary with exact_match, numeric_difference, and percentage_error
        """
        # Preprocess both texts for string comparison
        proc_pred = self.preprocess(prediction)
        proc_gt = self.preprocess(ground_truth)
        
        # Initialize metrics
        metrics = {}
        
        # Calculate exact match (primary metric)
        exact_match = proc_pred == proc_gt
        metrics["exact_match"] = 1.0 if exact_match else 0.0
        
        # Extract numeric values
        num_pred = self.extract_numeric_value(prediction)
        num_gt = self.extract_numeric_value(ground_truth)
        
        # Calculate numeric metrics if both values are available
        if num_pred is not None and num_gt is not None:
            # Numeric difference (absolute)
            metrics["numeric_difference"] = abs(num_pred - num_gt)
            
            # Percentage error
            if num_gt != 0:
                metrics["percentage_error"] = abs(num_pred - num_gt) / abs(num_gt) * 100
            else:
                metrics["percentage_error"] = 100.0 if num_pred != 0 else 0.0
        else:
            # Default values if numeric extraction failed
            metrics["numeric_difference"] = float('inf')
            metrics["percentage_error"] = 100.0
        
        return metrics


def create_metrics_calculator(field: str, config: Dict[str, Any]) -> MetricsCalculator:
    """Factory function to create field-specific metrics calculator.
    
    Args:
        field: Field type ("work_order", "cost", etc.)
        config: Field-specific configuration
        
    Returns:
        Appropriate MetricsCalculator instance
        
    Raises:
        ValueError: If field type is not supported
    """
    if field == "work_order":
        return WorkOrderMetricsCalculator(config)
    elif field == "cost":
        return CostMetricsCalculator(config)
    else:
        raise ValueError(f"Unknown field type: {field}")


# Additional utility functions for metrics calculation

def calculate_batch_metrics(results: List[Dict[str, Any]], field: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate metrics for a batch of results.
    
    Args:
        results: List of extraction results
        field: Field type
        config: Field configuration
        
    Returns:
        Dictionary of aggregated metrics
    """
    # Create calculator
    calculator = create_metrics_calculator(field, config)
    
    # Initialize metrics
    metrics = {}
    
    # Extract individual metrics
    metrics_list = []
    for result in results:
        if "error" not in result:
            prediction = result.get("processed_extraction", "")
            ground_truth = result.get("ground_truth", "")
            result_metrics = calculator.calculate(prediction, ground_truth)
            metrics_list.append(result_metrics)
    
    # Aggregate metrics
    if metrics_list:
        aggregated = calculator.aggregate(metrics_list)
        metrics.update(aggregated)
    
    # Add success rate
    success_count = sum(1 for r in results if r.get("exact_match", 0) > 0.5)
    total_count = len(results)
    metrics["success_rate"] = success_count / total_count if total_count > 0 else 0
    
    # Add counts
    metrics["total_count"] = total_count
    metrics["success_count"] = success_count
    metrics["error_count"] = sum(1 for r in results if "error" in r)
    
    return metrics


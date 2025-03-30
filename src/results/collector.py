"""
Enhanced Results Collection and Management System

This module provides a comprehensive system for collecting, storing,
analyzing, and comparing extraction experiment results. It offers:
- Structured storage for experiment results 
- Field-specific result management
- Comparative analysis across prompts, fields, and experiments
- Statistical significance testing
- Integration with visualization components
- Multiple export formats
"""

import os
import json
import logging
import statistics
import math
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Iterator

import numpy as np
import pandas as pd
from scipy import stats

# Import schema and metrics components
from src.results.schema import (
    ExperimentResult, 
    PromptPerformance, 
    IndividualExtractionResult,
    ExtractionStatus,
    FieldResult
)
from src.analysis.metrics import create_metrics_calculator
from src.config.experiment_config import ExperimentConfiguration, ExperimentType

# Configure logger
logger = logging.getLogger(__name__)


class ComparisonResult:
    """
    Structured representation of comparison results between multiple experiments,
    prompts, or quantization strategies.
    
    This class enables rich analysis and visualization of performance differences.
    """
    
    def __init__(
        self,
        name: str,
        primary_dimension: str,   # 'prompt', 'field', 'model', 'quantization', 'experiment'
        metrics: List[str],
        description: str = ""
    ):
        """
        Initialize comparison result with metadata.
        
        Args:
            name: Descriptive name for this comparison
            primary_dimension: Primary comparison dimension 
            metrics: List of metrics used for comparison
            description: Optional detailed description
        """
        self.name = name
        self.primary_dimension = primary_dimension
        self.metrics = metrics
        self.description = description
        self.timestamp = datetime.now().isoformat()
        
        # Storage for comparison values
        self.comparison_data: Dict[str, Dict[str, Any]] = {}
        self.statistical_tests: Dict[str, Dict[str, Any]] = {}
        self.summary: Dict[str, Any] = {}
        
        # Visualization suggestions based on comparison
        self.visualization_suggestions: List[Dict[str, Any]] = []
    
    def add_data_point(
        self,
        dimension_value: str,
        metric_values: Dict[str, float],
        sample_size: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a data point to the comparison.
        
        Args:
            dimension_value: Value for the primary dimension (e.g., prompt name)
            metric_values: Dictionary mapping metrics to their values
            sample_size: Number of samples used to calculate metrics
            metadata: Optional additional metadata for this data point
        """
        self.comparison_data[dimension_value] = {
            "metrics": metric_values,
            "sample_size": sample_size,
            "metadata": metadata or {}
        }
    
    def calculate_statistics(self) -> None:
        """
        Calculate statistical comparisons between data points.
        """
        if len(self.comparison_data) < 2:
            self.summary = {
                "status": "insufficient_data",
                "message": "At least two data points are needed for comparison"
            }
            return
        
        # For each metric, perform pairwise comparisons
        metric_summaries = {}
        for metric in self.metrics:
            # Create list of (dimension_value, metric_value, sample_size) tuples
            data_points = [
                (dim, data["metrics"].get(metric), data["sample_size"])
                for dim, data in self.comparison_data.items()
                if metric in data["metrics"]
            ]
            
            # Sort by metric value (descending)
            data_points.sort(key=lambda x: x[1] if x[1] is not None else float('-inf'), reverse=True)
            
            # Calculate statistics
            metric_summaries[metric] = {
                "best": data_points[0][0] if data_points else None,
                "best_value": data_points[0][1] if data_points else None,
                "worst": data_points[-1][0] if data_points else None,
                "worst_value": data_points[-1][1] if data_points else None,
                "avg_value": sum(p[1] for p in data_points if p[1] is not None) / sum(1 for p in data_points if p[1] is not None) 
                             if any(p[1] is not None for p in data_points) else None,
                "range": data_points[0][1] - data_points[-1][1] if data_points and data_points[0][1] is not None and data_points[-1][1] is not None else None,
                "pairwise_comparisons": {}
            }
            
            # Perform pairwise statistical tests where possible
            for i, (dim1, val1, n1) in enumerate(data_points):
                if val1 is None:
                    continue
                    
                for j, (dim2, val2, n2) in enumerate(data_points[i+1:], i+1):
                    if val2 is None:
                        continue
                        
                    # Perform statistical significance testing
                    # Note: For percentages, we use proportion test
                    # For continuous metrics, we'd need sample values for t-test
                    significant, p_value = self._test_significance(val1, val2, n1, n2)
                    
                    # Store comparison results
                    comparison_key = f"{dim1}_vs_{dim2}"
                    metric_summaries[metric]["pairwise_comparisons"][comparison_key] = {
                        "difference": val1 - val2,
                        "percent_difference": (val1 - val2) / val2 * 100 if val2 != 0 else float('inf'),
                        "statistically_significant": significant,
                        "p_value": p_value
                    }
        
        # Overall summary
        self.summary = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "metrics": metric_summaries,
            "dimension": self.primary_dimension,
            "has_significant_differences": any(
                any(comp["statistically_significant"] 
                    for comp in metric["pairwise_comparisons"].values())
                for metric in metric_summaries.values()
            )
        }
        
        # Generate visualization suggestions based on comparison
        self._generate_visualization_suggestions()
    
    def _test_significance(
        self, 
        val1: float, 
        val2: float, 
        n1: int, 
        n2: int
    ) -> Tuple[bool, float]:
        """
        Test statistical significance between two values.
        
        Args:
            val1: First value (proportion or mean)
            val2: Second value (proportion or mean)
            n1: Sample size for first value
            n2: Sample size for second value
            
        Returns:
            Tuple of (is_significant, p_value)
        """
        # For proportions (e.g., success rates)
        if 0 <= val1 <= 1 and 0 <= val2 <= 1:
            # Convert proportions to counts
            count1 = round(val1 * n1)
            count2 = round(val2 * n2)
            
            # Use chi-square test for proportions
            try:
                # Create contingency table
                contingency = np.array([
                    [count1, n1 - count1],
                    [count2, n2 - count2]
                ])
                
                # Chi-square test
                _, p_value, _, _ = stats.chi2_contingency(contingency)
                
                # Significant at 95% confidence
                significant = p_value < 0.05
                
                return significant, p_value
            except Exception as e:
                logger.warning(f"Error in statistical test: {e}")
                return False, 1.0
        
        # For other metrics, we'd ideally use t-test with raw values
        # But without raw values, use simplified approach
        difference = abs(val1 - val2)
        # A placeholder significance test - this should be replaced with
        # a proper statistical test when raw values are available
        significant = difference > 0.1  # Arbitrary threshold
        p_value = 0.5  # Placeholder
        
        return significant, p_value
    
    def _generate_visualization_suggestions(self) -> None:
        """
        Generate visualization suggestions based on comparison data.
        """
        suggestions = []
        
        # Bar chart for comparing metrics
        if len(self.comparison_data) >= 2:
            suggestions.append({
                "type": "bar_chart",
                "title": f"Comparison of {self.primary_dimension.title()} by {', '.join(self.metrics)}",
                "x_axis": self.primary_dimension,
                "y_axis": self.metrics[0] if self.metrics else None,
                "data": self.comparison_data
            })
        
        # Heatmap for multiple metrics
        if len(self.metrics) >= 2 and len(self.comparison_data) >= 2:
            suggestions.append({
                "type": "heatmap",
                "title": f"Heatmap of {self.primary_dimension.title()} Performance",
                "x_axis": self.primary_dimension,
                "y_axis": "metric",
                "data": self.comparison_data
            })
        
        # Statistical significance visualization
        if self.summary.get("has_significant_differences", False):
            suggestions.append({
                "type": "significance_plot",
                "title": "Statistical Significance of Differences",
                "data": self.summary["metrics"]
            })
        
        self.visualization_suggestions = suggestions
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert comparison result to a dictionary.
        
        Returns:
            Dictionary representation of the comparison
        """
        return {
            "name": self.name,
            "primary_dimension": self.primary_dimension,
            "metrics": self.metrics,
            "description": self.description,
            "timestamp": self.timestamp,
            "comparison_data": self.comparison_data,
            "summary": self.summary,
            "visualization_suggestions": self.visualization_suggestions
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ComparisonResult':
        """
        Create a ComparisonResult from a dictionary.
        
        Args:
            data: Dictionary representation of comparison result
            
        Returns:
            ComparisonResult instance
        """
        comparison = ComparisonResult(
            name=data.get("name", "Unnamed Comparison"),
            primary_dimension=data.get("primary_dimension", ""),
            metrics=data.get("metrics", []),
            description=data.get("description", "")
        )
        
        # Restore timestamp
        comparison.timestamp = data.get("timestamp", datetime.now().isoformat())
        
        # Restore comparison data
        comparison.comparison_data = data.get("comparison_data", {})
        
        # Restore summary
        comparison.summary = data.get("summary", {})
        
        # Restore visualization suggestions
        comparison.visualization_suggestions = data.get("visualization_suggestions", [])
        
        return comparison


class ResultsCollector:
    """
    System for collecting, storing, and analyzing extraction results.
    
    This class provides a comprehensive interface for:
    - Storing experiment results in a structured format
    - Calculating performance metrics
    - Comparing results across different dimensions
    - Exporting results for further analysis
    - Visualizing extraction performance
    """
    
    def __init__(
        self, 
        base_path: Union[str, Path], 
        experiment_name: Optional[str] = None
    ):
        """
        Initialize results collector.
        
        Args:
            base_path: Base path for storing results
            experiment_name: Optional experiment name
        """
        self.base_path = Path(base_path)
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory
        self.experiment_dir = self.base_path / self.experiment_name
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create directory structure
        self._create_directory_structure()
        
        # Initialize result data structures
        self.fields: List[str] = []
        self.prompts: List[str] = []
        self.quantization_strategies: List[str] = []
        self.model_name: Optional[str] = None
        
        # Results collections
        self.field_results: Dict[str, Dict[str, Any]] = {}
        self.field_metrics: Dict[str, Dict[str, Any]] = {}
        self.cross_field_metrics: Dict[str, Any] = {}
        
        # Experiment result
        self.experiment_result: Optional[ExperimentResult] = None
        
        # Set up experiment ID
        self.experiment_id = self.experiment_name
        
        logger.info(f"Initialized results collector for experiment: {self.experiment_name}")
    
    def _create_directory_structure(self) -> None:
        """
        Create the directory structure for storing results.
        """
        # Create main experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        subdirs = [
            "raw_results",
            "processed_results", 
            "metrics", 
            "visualizations", 
            "checkpoints",
            "comparisons"
        ]
        
        for subdir in subdirs:
            (self.experiment_dir / subdir).mkdir(exist_ok=True)
        
        # Create field-specific directories
        for field in ["work_order", "cost", "combined"]:
            field_dir = self.experiment_dir / field
            field_dir.mkdir(exist_ok=True)
            
            # Create field-specific subdirectories
            for subdir in ["raw", "processed", "visualizations"]:
                (field_dir / subdir).mkdir(exist_ok=True)
    
    def add_field_results(
        self, 
        field: str, 
        prompt_name: str, 
        results: List[Union[IndividualExtractionResult, Dict[str, Any]]],
        quantization: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> None:
        """
        Add results for a specific field and prompt.
        
        Args:
            field: Field type being extracted
            prompt_name: Name of the prompt used
            results: List of individual extraction results or dicts
            quantization: Optional quantization strategy used
            model_name: Optional model name used
        """
        # Convert dict results to IndividualExtractionResult if needed
        extraction_results = []
        for result in results:
            if isinstance(result, IndividualExtractionResult):
                extraction_results.append(result)
            else:
                # Convert dict to IndividualExtractionResult
                extraction_results.append(
                    self._convert_dict_to_extraction_result(result, field)
                )
        
        # Create or get performance tracker for this field and prompt
        performance = PromptPerformance(
            prompt_name=prompt_name,
            field=field
        )
        
        # Add results to performance tracker
        performance.results = extraction_results
        
        # Calculate metrics
        performance.calculate_metrics()
        
        # Store quantization info in metadata if provided
        if quantization:
            performance.metadata = performance.metadata or {}
            performance.metadata["quantization"] = quantization
            
            # Update experiment metadata
            if quantization not in self.metadata["quantization_strategies"]:
                self.metadata["quantization_strategies"].append(quantization)
        
        # Store model info in metadata if provided
        if model_name:
            performance.metadata = performance.metadata or {}
            performance.metadata["model_name"] = model_name
            
            # Update experiment metadata
            if model_name not in self.metadata["models"]:
                self.metadata["models"].append(model_name)
        
        # Add to experiment result
        self.experiment_result.add_field_results(field, performance)
        
        # Update metadata
        if field not in self.metadata["fields"]:
            self.metadata["fields"].append(field)
        if prompt_name not in self.metadata["prompts"]:
            self.metadata["prompts"].append(prompt_name)
        
        logger.info(f"Added {len(extraction_results)} results for field: {field}, prompt: {prompt_name}")
    
    def _convert_dict_to_extraction_result(
        self, 
        result_dict: Dict[str, Any],
        field: str
    ) -> IndividualExtractionResult:
        """
        Convert a dict to an IndividualExtractionResult object.
        
        Args:
            result_dict: Dictionary containing extraction result data
            field: Field type being extracted
            
        Returns:
            IndividualExtractionResult object
        """
        # Extract core fields
        image_id = result_dict.get('image_id', '')
        ground_truth = result_dict.get('ground_truth', '')
        
        # Try different field names for extracted value
        extracted_value = (
            result_dict.get('processed_extraction') or 
            result_dict.get('extraction') or 
            result_dict.get('extracted_value', '')
        )
        
        # Get performance metrics
        exact_match = result_dict.get('exact_match', False)
        character_error_rate = result_dict.get('character_error_rate', 1.0)
        confidence_score = result_dict.get('confidence', 0.0)
        
        # Get processing details
        processing_time = result_dict.get('processing_time', 0.0)
        
        # Determine status
        if 'error' in result_dict:
            status = ExtractionStatus.ERROR
            error_message = result_dict.get('error')
        elif exact_match:
            status = ExtractionStatus.SUCCESS
            error_message = None
        elif character_error_rate < 0.5:
            status = ExtractionStatus.PARTIAL_MATCH
            error_message = None
        else:
            status = ExtractionStatus.NO_MATCH
            error_message = None
        
        return IndividualExtractionResult(
            image_id=image_id,
            field=field,
            ground_truth=ground_truth,
            extracted_value=extracted_value,
            exact_match=exact_match,
            character_error_rate=character_error_rate,
            confidence_score=confidence_score,
            status=status,
            processing_time=processing_time,
            error_message=error_message
        )
    
    def save_raw_results(
        self, 
        field: str, 
        prompt_name: str, 
        results: List[Dict[str, Any]],
        quantization: Optional[str] = None
    ) -> Path:
        """
        Save raw extraction results to a JSON file.
        
        Args:
            field: Field type being extracted
            prompt_name: Name of the prompt used
            results: Raw results dictionary
            quantization: Optional quantization strategy used
            
        Returns:
            Path to the saved file
        """
        # Create filename with quantization info if provided
        if quantization:
            filename = f"{field}_{prompt_name}_{quantization}_raw_results.json"
        else:
            filename = f"{field}_{prompt_name}_raw_results.json"
        
        filepath = self.experiment_dir / "raw_results" / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved raw results to {filepath}")
        return filepath
    
    def get_field_path(self, field: str, file_type: str = "raw") -> Path:
        """
        Get path for field-specific storage.
        
        Args:
            field: Field type (work_order, cost, etc.)
            file_type: Type of file (raw, processed, visualization)
            
        Returns:
            Path object for the field directory
        """
        return self.experiment_dir / field / file_type
    
    def save_field_results(self, field: str, results: List[Dict[str, Any]]) -> Path:
        """
        Save extraction results for a specific field.
        
        Args:
            field: Field type (work_order, cost, etc.)
            results: List of extraction results
            
        Returns:
            Path to the saved file
        """
        # Validate input
        if not results:
            logger.warning(f"No results to save for {field}")
            return Path(self.experiment_dir)
        
        # Save complete results
        results_path = self.get_field_path(field, "raw") / "extraction_results.json"
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved {len(results)} {field} results to {results_path}")
        except Exception as e:
            logger.error(f"Error saving {field} results: {e}")
        
        # Organize results by prompt category
        by_category = {}
        for result in results:
            category = result.get("prompt_category", "unknown")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        # Save per-category results
        for category, category_results in by_category.items():
            category_path = self.get_field_path(field, "raw") / f"{category}_results.json"
            try:
                with open(category_path, 'w') as f:
                    json.dump(category_results, f, indent=2)
                logger.info(f"Saved {len(category_results)} {field} results for {category} to {category_path}")
            except Exception as e:
                logger.error(f"Error saving {field} results for {category}: {e}")
        
        return results_path
    
    def save_field_metrics(self, field: str, metrics: Dict[str, Any]) -> Path:
        """
        Save metrics for a specific field.
        
        Args:
            field: Field type (work_order, cost, etc.)
            metrics: Dictionary of metrics by prompt category
            
        Returns:
            Path to the saved file
        """
        # Validate input
        if not metrics:
            logger.warning(f"No metrics to save for {field}")
            return Path(self.experiment_dir)
        
        # Add timestamp
        metrics_with_meta = {
            "timestamp": datetime.now().isoformat(),
            "field": field,
            "metrics": metrics
        }
        
        # Save metrics
        metrics_path = self.get_field_path(field, "processed") / "metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics_with_meta, f, indent=2)
            logger.info(f"Saved {field} metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving {field} metrics: {e}")
        
        return metrics_path
    
    def calculate_cross_field_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics across different fields.
        
        Returns:
            Dictionary of cross-field metrics
        """
        cross_field_metrics = {
            "total_fields": len(self.experiment_result.field_results),
            "total_items": self.experiment_result.total_items,
            "overall_accuracy": self.experiment_result.overall_accuracy,
            "field_performance": {}
        }
        
        for field, performances in self.experiment_result.field_results.items():
            field_metrics = {
                "best_prompt": max(performances, key=lambda p: p.accuracy).prompt_name if performances else None,
                "avg_accuracy": np.mean([p.accuracy for p in performances]) if performances else 0,
                "avg_character_error_rate": np.mean([p.avg_character_error_rate for p in performances]) if performances else 1.0,
                "prompt_count": len(performances)
            }
            cross_field_metrics["field_performance"][field] = field_metrics
        
        return cross_field_metrics
    
    def save_cross_field_metrics(self, cross_field_metrics: Dict[str, Any]) -> Path:
        """
        Save metrics comparing performance across different fields.
        
        Args:
            cross_field_metrics: Dictionary of cross-field metrics
            
        Returns:
            Path to the saved file
        """
        # Validate input
        if not cross_field_metrics:
            logger.warning("No cross-field metrics to save")
            return Path(self.experiment_dir)
        
        # Add timestamp if not present
        if "timestamp" not in cross_field_metrics:
            cross_field_metrics["timestamp"] = datetime.now().isoformat()
        
        # Save cross-field metrics
        metrics_path = self.experiment_dir / "combined" / "processed" / "cross_field_metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(cross_field_metrics, f, indent=2)
            logger.info(f"Saved cross-field metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving cross-field metrics: {e}")
        
        return metrics_path
    
    def save_metrics(self, metrics: Dict[str, Any]) -> Path:
        """
        Save calculated metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            
        Returns:
            Path to the saved file
        """
        metrics_filename = "experiment_metrics.json"
        filepath = self.experiment_dir / "metrics" / metrics_filename
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved experiment metrics to {filepath}")
        return filepath
    
    def save_experiment_result(self) -> Path:
        """
        Save the complete experiment result to a JSON file.
        
        Returns:
            Path to the saved file
        """
        # Finalize experiment metadata
        self.metadata["completed_at"] = datetime.now().isoformat()
        self.metadata["total_items"] = self.experiment_result.total_items
        self.metadata["overall_accuracy"] = self.experiment_result.overall_accuracy
        
        # Save experiment metadata
        metadata_filepath = self.experiment_dir / "experiment_metadata.json"
        with open(metadata_filepath, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save experiment result
        result_filepath = self.experiment_dir / "experiment_result.json"
        self.experiment_result.save_to_file(str(result_filepath))
        
        logger.info(f"Saved experiment result to {result_filepath}")
        logger.info(f"Saved experiment metadata to {metadata_filepath}")
        return result_filepath
    
    def save_run_metadata(self, metadata: Dict[str, Any]) -> Path:
        """
        Save experiment run metadata.
        
        Args:
            metadata: Dictionary of experiment metadata
            
        Returns:
            Path to the saved file
        """
        # Validate input
        if not metadata:
            logger.warning("No metadata to save")
            return Path(self.experiment_dir)
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # Save metadata
        metadata_path = self.experiment_dir / "run_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved run metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving run metadata: {e}")
        
        return metadata_path
    
    def generate_dataframe(self, field: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a pandas DataFrame from extraction results.
        
        Args:
            field: Optional field to filter results
        
        Returns:
            DataFrame with extraction results
        """
        all_results = []
        
        for f, performances in self.experiment_result.field_results.items():
            # Skip if field specified and doesn't match
            if field and f != field:
                continue
            
            for perf in performances:
                for result in perf.results:
                    result_dict = result.to_dict()
                    result_dict.update({
                        "field": f,
                        "prompt_name": perf.prompt_name
                    })
                    
                    # Add quantization info if available
                    if perf.metadata and "quantization" in perf.metadata:
                        result_dict["quantization"] = perf.metadata["quantization"]
                    
                    # Add model info if available
                    if perf.metadata and "model_name" in perf.metadata:
                        result_dict["model_name"] = perf.metadata["model_name"]
                    
                    all_results.append(result_dict)
        
        return pd.DataFrame(all_results)
    
    def export_results(self, format: str = 'csv') -> Path:
        """
        Export results to different formats.
        
        Args:
            format: Output format (csv, excel, parquet)
        
        Returns:
            Path to exported file
        """
        # Generate DataFrame
        df = self.generate_dataframe()
        
        # Determine export path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"{self.experiment_name}_results.{format}"
        export_path = self.experiment_dir / "processed_results" / export_filename
        
        # Export based on format
        if format == 'csv':
            df.to_csv(export_path, index=False)
        elif format == 'excel':
            df.to_excel(export_path, index=False)
        elif format == 'parquet':
            df.to_parquet(export_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported results to {export_path}")
        return export_path
    
    def save_visualization(
        self, 
        field: str, 
        visualization_name: str, 
        file_path: Union[str, Path]
    ) -> Path:
        """
        Copy or save a visualization file to the appropriate location.
        
        Args:
            field: Field type or 'combined' for cross-field visualizations
            visualization_name: Name of the visualization file
            file_path: Path to the visualization file
            
        Returns:
            Path to the saved visualization
        """
        import shutil
        
        source_path = Path(file_path)
        
        # Verify source file exists
        if not source_path.exists():
            logger.error(f"Visualization file not found: {source_path}")
            return Path(self.experiment_dir)
        
        # Get destination path
        if field == "combined":
            dest_dir = self.experiment_dir / "combined" / "visualizations"
        else:
            dest_dir = self.get_field_path(field, "visualizations")
        
        # Ensure destination directory exists
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy or save file
        dest_path = dest_dir / visualization_name
        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Saved visualization to {dest_path}")
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
        
        return dest_path
    
    def list_results(self, field: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available result files.
        
        Args:
            field: Optional field type to filter results
            
        Returns:
            Dictionary of available result files by type
        """
        result_files = {
            "raw": [],
            "processed": [],
            "visualizations": []
        }
        
        # If field specified, only check that field's directory
        if field:
            for file_type in result_files.keys():
                path = self.get_field_path(field, file_type)
                if path.exists():
                    result_files[file_type] = [f.name for f in path.glob("*.json")]
        # Otherwise check all field directories
        else:
            for field_dir in self.experiment_dir.glob("*"):
                if not field_dir.is_dir() or field_dir.name in ["raw_results", "processed_results", "visualizations", "checkpoints", "comparisons", "metrics"]:
                    continue
                
                field_name = field_dir.name
                for file_type in result_files.keys():
                    path = self.get_field_path(field_name, file_type)
                    if path.exists():
                        for f in path.glob("*.json"):
                            result_files[file_type].append(f"{field_name}/{f.name}")
        
        return result_files
    
    def load_results(
        self, 
        field: str, 
        result_type: str, 
        filename: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load a specific result file.
        
        Args:
            field: Field type (work_order, cost, etc.)
            result_type: Type of result (raw, processed, visualization)
            filename: Name of the file to load
            
        Returns:
            Dictionary of loaded results or None if file not found/invalid
        """
        file_path = self.get_field_path(field, result_type) / filename
        
        if not file_path.exists():
            logger.error(f"Result file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading result file {file_path}: {e}")
            return None
    
    # ----- COMPARISON METHODS -----
    
    def compare_prompts(
        self, 
        field: str,
        prompts: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compare performance across different prompts for a specific field.
        
        Args:
            field: Field to compare prompts for
            prompts: List of prompt names to compare (if None, use all for the field)
            metrics: List of metrics to compare (if None, use defaults)
            name: Optional name for the comparison
            
        Returns:
            ComparisonResult with prompt comparison data
        """
        # Set default metrics if not specified
        if metrics is None:
            metrics = ["accuracy", "avg_character_error_rate", "avg_processing_time"]
        
        # Create comparison result
        comparison_name = name or f"Prompt Comparison - {field}"
        comparison = ComparisonResult(
            name=comparison_name,
            primary_dimension="prompt",
            metrics=metrics,
            description=f"Comparison of different prompts for field: {field}"
        )
        
        # Get field results
        field_results = self.experiment_result.field_results.get(field, [])
        
        if not field_results:
            logger.warning(f"No results found for field: {field}")
            return comparison
        
        # Filter prompts if specified
        if prompts:
            field_results = [r for r in field_results if r.prompt_name in prompts]
        
        # Add data points to comparison
        for performance in field_results:
            metric_values = {
                "accuracy": performance.accuracy,
                "avg_character_error_rate": performance.avg_character_rate,
                "avg_processing_time": performance.avg_processing_time
            }
            
            # Filter to requested metrics
            filtered_metrics = {k: v for k, v in metric_values.items() if k in metrics}
            
            # Add data point
            comparison.add_data_point(
                dimension_value=performance.prompt_name,
                metric_values=filtered_metrics,
                sample_size=performance.total_items,
                metadata=performance.metadata
            )
        
        # Calculate statistics
        comparison.calculate_statistics()
        
        # Save comparison
        self._save_comparison(comparison)
        
        # Store in memory
        self.comparisons[comparison.name] = comparison
        
        return comparison
    
    def compare_fields(
        self,
        prompts: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compare performance across different fields.
        
        Args:
            prompts: Optional list of prompts to consider (if None, use best prompt for each field)
            metrics: List of metrics to compare (if None, use defaults)
            name: Optional name for the comparison
            
        Returns:
            ComparisonResult with field comparison data
        """
        # Set default metrics if not specified
        if metrics is None:
            metrics = ["accuracy", "avg_character_error_rate"]
        
        # Create comparison result
        comparison_name = name or f"Field Comparison"
        comparison = ComparisonResult(
            name=comparison_name,
            primary_dimension="field",
            metrics=metrics,
            description=f"Comparison of extraction performance across different fields"
        )
        
        # Process each field
        for field, performances in self.experiment_result.field_results.items():
            if not performances:
                continue
                
            # Filter to specified prompts if applicable
            field_performances = performances
            if prompts:
                field_performances = [p for p in performances if p.prompt_name in prompts]
                
                if not field_performances:
                    logger.warning(f"No matching prompts for field {field}, skipping")
                    continue
            
            # Find best performance (based on accuracy) 
            best_performance = max(field_performances, key=lambda p: p.accuracy)
            
            # Get metric values
            metric_values = {
                "accuracy": best_performance.accuracy,
                "avg_character_error_rate": best_performance.avg_character_error_rate,
                "avg_processing_time": best_performance.avg_processing_time
            }
            
            # Filter to requested metrics
            filtered_metrics = {k: v for k, v in metric_values.items() if k in metrics}
            
            # Add field as data point
            comparison.add_data_point(
                dimension_value=field,
                metric_values=filtered_metrics,
                sample_size=best_performance.total_items,
                metadata={
                    "best_prompt": best_performance.prompt_name,
                    "prompt_count": len(field_performances)
                }
            )
        
        # Calculate statistics
        comparison.calculate_statistics()
        
        # Save comparison
        self._save_comparison(comparison)
        
        # Store in memory
        self.comparisons[comparison.name] = comparison
        
        return comparison
    
    def compare_quantization_strategies(
        self,
        field: str,
        prompt_name: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compare performance across different quantization strategies.
        
        Args:
            field: Field to compare quantization for
            prompt_name: Optional prompt to filter by
            metrics: List of metrics to compare (if None, use defaults)
            name: Optional name for the comparison
            
        Returns:
            ComparisonResult with quantization comparison data
        """
        # Set default metrics if not specified
        if metrics is None:
            metrics = ["accuracy", "avg_processing_time"]
        
        # Create comparison result
        comparison_name = name or f"Quantization Comparison - {field}"
        comparison = ComparisonResult(
            name=comparison_name,
            primary_dimension="quantization",
            metrics=metrics,
            description=f"Comparison of different quantization strategies for field: {field}"
        )
        
        # Get field results
        field_results = self.experiment_result.field_results.get(field, [])
        
        if not field_results:
            logger.warning(f"No results found for field: {field}")
            return comparison
        
        # Filter by prompt if specified
        if prompt_name:
            field_results = [r for r in field_results if r.prompt_name == prompt_name]
        
        # Group by quantization
        quantization_results = {}
        for performance in field_results:
            # Skip if no metadata or quantization info
            if not performance.metadata or "quantization" not in performance.metadata:
                continue
                
            quant = performance.metadata["quantization"]
            
            if quant not in quantization_results:
                quantization_results[quant] = []
                
            quantization_results[quant].append(performance)
        
        # Process each quantization strategy
        for quant, performances in quantization_results.items():
            # Find best performance for this quantization (if multiple prompts)
            best_performance = max(performances, key=lambda p: p.accuracy)
            
            # Get metric values
            metric_values = {
                "accuracy": best_performance.accuracy,
                "avg_character_error_rate": best_performance.avg_character_error_rate,
                "avg_processing_time": best_performance.avg_processing_time
            }
            
            # Filter to requested metrics
            filtered_metrics = {k: v for k, v in metric_values.items() if k in metrics}
            
            # Add quantization as data point
            comparison.add_data_point(
                dimension_value=quant,
                metric_values=filtered_metrics,
                sample_size=best_performance.total_items,
                metadata={
                    "best_prompt": best_performance.prompt_name,
                    "prompt_count": len(performances)
                }
            )
        
        # Calculate statistics
        comparison.calculate_statistics()
        
        # Save comparison
        self._save_comparison(comparison)
        
        # Store in memory
        self.comparisons[comparison.name] = comparison
        
        return comparison
    
    def compare_models(
        self,
        field: str,
        prompt_name: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compare performance across different models.
        
        Args:
            field: Field to compare models for
            prompt_name: Optional prompt to filter by
            metrics: List of metrics to compare (if None, use defaults)
            name: Optional name for the comparison
            
        Returns:
            ComparisonResult with model comparison data
        """
        # Set default metrics if not specified
        if metrics is None:
            metrics = ["accuracy", "avg_processing_time"]
        
        # Create comparison result
        comparison_name = name or f"Model Comparison - {field}"
        comparison = ComparisonResult(
            name=comparison_name,
            primary_dimension="model",
            metrics=metrics,
            description=f"Comparison of different models for field: {field}"
        )
        
        # Get field results
        field_results = self.experiment_result.field_results.get(field, [])
        
        if not field_results:
            logger.warning(f"No results found for field: {field}")
            return comparison
        
        # Filter by prompt if specified
        if prompt_name:
            field_results = [r for r in field_results if r.prompt_name == prompt_name]
        
        # Group by model
        model_results = {}
        for performance in field_results:
            # Skip if no metadata or model info
            if not performance.metadata or "model_name" not in performance.metadata:
                continue
                
            model = performance.metadata["model_name"]
            
            if model not in model_results:
                model_results[model] = []
                
            model_results[model].append(performance)
        
        # Process each model
        for model, performances in model_results.items():
            # Find best performance for this model (if multiple prompts)
            best_performance = max(performances, key=lambda p: p.accuracy)
            
            # Get metric values
            metric_values = {
                "accuracy": best_performance.accuracy,
                "avg_character_error_rate": best_performance.avg_character_error_rate,
                "avg_processing_time": best_performance.avg_processing_time
            }
            
            # Filter to requested metrics
            filtered_metrics = {k: v for k, v in metric_values.items() if k in metrics}
            
            # Add model as data point
            comparison.add_data_point(
                dimension_value=model,
                metric_values=filtered_metrics,
                sample_size=best_performance.total_items,
                metadata={
                    "best_prompt": best_performance.prompt_name,
                    "prompt_count": len(performances)
                }
            )
        
        # Calculate statistics
        comparison.calculate_statistics()
        
        # Save comparison
        self._save_comparison(comparison)
        
        # Store in memory
        self.comparisons[comparison.name] = comparison
        
        return comparison
    
    def compare_across_experiments(
        self,
        field: str,
        prompt_name: str,
        experiment_paths: List[Union[str, Path]],
        metrics: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compare performance across different experiments.
        
        Args:
            field: Field to compare
            prompt_name: Prompt name to compare
            experiment_paths: List of paths to experiment results
            metrics: List of metrics to compare (if None, use defaults)
            name: Optional name for the comparison
            
        Returns:
            ComparisonResult with experiment comparison data
        """
        # Set default metrics if not specified
        if metrics is None:
            metrics = ["accuracy", "avg_character_error_rate", "avg_processing_time"]
        
        # Create comparison result
        comparison_name = name or f"Cross-Experiment Comparison - {field}"
        comparison = ComparisonResult(
            name=comparison_name,
            primary_dimension="experiment",
            metrics=metrics,
            description=f"Comparison across different experiments for field: {field}, prompt: {prompt_name}"
        )
        
        # Add current experiment
        self._add_experiment_to_comparison(
            comparison=comparison,
            experiment_name=self.experiment_name,
            experiment_result=self.experiment_result,
            field=field,
            prompt_name=prompt_name,
            metrics=metrics
        )
        
        # Load and add other experiments
        for exp_path in experiment_paths:
            exp_path = Path(exp_path)
            exp_name = exp_path.name
            
            # Skip if same as current experiment
            if exp_name == self.experiment_name:
                continue
            
            # Load experiment result
            exp_result = self._load_experiment_result(exp_path)
            if not exp_result:
                logger.warning(f"Could not load experiment results from {exp_path}")
                continue
            
            # Add to comparison
            self._add_experiment_to_comparison(
                comparison=comparison,
                experiment_name=exp_name,
                experiment_result=exp_result,
                field=field,
                prompt_name=prompt_name,
                metrics=metrics
            )
        
        # Calculate statistics
        comparison.calculate_statistics()
        
        # Save comparison
        self._save_comparison(comparison)
        
        # Store in memory
        self.comparisons[comparison.name] = comparison
        
        return comparison
    
    def _add_experiment_to_comparison(
        self,
        comparison: ComparisonResult,
        experiment_name: str,
        experiment_result: ExperimentResult,
        field: str,
        prompt_name: str,
        metrics: List[str]
    ) -> None:
        """
        Add an experiment to a cross-experiment comparison.
        
        Args:
            comparison: ComparisonResult to update
            experiment_name: Name of the experiment
            experiment_result: ExperimentResult object
            field: Field to compare
            prompt_name: Prompt name to compare
            metrics: List of metrics to compare
        """
        # Find the matching performance
        field_results = experiment_result.field_results.get(field, [])
        matching_performances = [p for p in field_results if p.prompt_name == prompt_name]
        
        if not matching_performances:
            logger.warning(f"No matching performance found in experiment {experiment_name} for field {field}, prompt {prompt_name}")
            return
        
        # Use the first matching performance
        performance = matching_performances[0]
        
        # Get metric values
        metric_values = {
            "accuracy": performance.accuracy,
            "avg_character_error_rate": performance.avg_character_error_rate,
            "avg_processing_time": performance.avg_processing_time
        }
        
        # Filter to requested metrics
        filtered_metrics = {k: v for k, v in metric_values.items() if k in metrics}
        
        # Add experiment as data point
        comparison.add_data_point(
            dimension_value=experiment_name,
            metric_values=filtered_metrics,
            sample_size=performance.total_items,
            metadata={
                "field": field,
                "prompt": prompt_name
            }
        )
    
    def _load_experiment_result(self, experiment_path: Union[str, Path]) -> Optional[ExperimentResult]:
        """
        Load experiment result from a path.
        
        Args:
            experiment_path: Path to experiment directory or result file
            
        Returns:
            ExperimentResult or None if loading fails
        """
        experiment_path = Path(experiment_path)
        
        # Check cache first
        if str(experiment_path) in self._experiment_cache:
            return self._experiment_cache[str(experiment_path)]
        
        # Determine result file path
        if experiment_path.is_dir():
            result_path = experiment_path / "experiment_result.json"
        else:
            result_path = experiment_path
        
        # Check if file exists
        if not result_path.exists():
            logger.warning(f"Experiment result file not found: {result_path}")
            return None
        
        # Load result
        try:
            exp_result = ExperimentResult.load_from_file(str(result_path))
            
            # Cache result
            self._experiment_cache[str(experiment_path)] = exp_result
            
            return exp_result
        except Exception as e:
            logger.error(f"Error loading experiment result from {result_path}: {e}")
            return None
    
    def _save_comparison(self, comparison: ComparisonResult) -> Path:
        """
        Save a comparison result to disk.
        
        Args:
            comparison: ComparisonResult to save
            
        Returns:
            Path to the saved file
        """
        # Create comparisons directory if needed
        comparisons_dir = self.experiment_dir / "comparisons"
        comparisons_dir.mkdir(exist_ok=True)
        
        # Create normalized filename
        normalized_name = comparison.name.lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{normalized_name}_{timestamp}.json"
        
        filepath = comparisons_dir / filename
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(comparison.to_dict(), f, indent=2)
        
        logger.info(f"Saved comparison to {filepath}")
        return filepath
    
    def load_comparison(self, filepath: Union[str, Path]) -> Optional[ComparisonResult]:
        """
        Load a comparison result from disk.
        
        Args:
            filepath: Path to the comparison file
            
        Returns:
            ComparisonResult or None if loading fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"Comparison file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            comparison = ComparisonResult.from_dict(data)
            
            # Store in memory
            self.comparisons[comparison.name] = comparison
            
            return comparison
        except Exception as e:
            logger.error(f"Error loading comparison from {filepath}: {e}")
            return None
    
    def list_comparisons(self) -> List[str]:
        """
        List available comparisons.
        
        Returns:
            List of comparison file paths
        """
        comparisons_dir = self.experiment_dir / "comparisons"
        if not comparisons_dir.exists():
            return []
        
        return [str(f) for f in comparisons_dir.glob("*.json")]
    
    def generate_comparative_report(
        self,
        comparisons: Optional[List[ComparisonResult]] = None,
        format: str = "markdown"
    ) -> str:
        """
        Generate a comprehensive comparative report.
        
        Args:
            comparisons: List of comparisons to include (if None, use all available)
            format: Output format (markdown, html, json)
            
        Returns:
            Formatted report
        """
        # Use all comparisons if not specified
        if comparisons is None:
            comparisons = list(self.comparisons.values())
        
        if not comparisons:
            logger.warning("No comparisons available for report")
            return "No comparisons available for report"
        
        # Generate report based on format
        if format == "markdown":
            return self._generate_markdown_report(comparisons)
        elif format == "html":
            return self._generate_html_report(comparisons)
        elif format == "json":
            return json.dumps({
                "report_name": "Comparative Analysis Report",
                "timestamp": datetime.now().isoformat(),
                "comparisons": [c.to_dict() for c in comparisons]
            }, indent=2)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_markdown_report(self, comparisons: List[ComparisonResult]) -> str:
        """
        Generate a Markdown report from comparisons.
        
        Args:
            comparisons: List of comparisons to include
            
        Returns:
            Markdown formatted report
        """
        lines = []
        
        # Add header
        lines.append("# Comparative Analysis Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Experiment: {self.experiment_name}")
        lines.append("")
        
        # Add summary
        lines.append("## Summary")
        lines.append("")
        
        # Count comparison types
        comparison_types = {}
        for comp in comparisons:
            dim = comp.primary_dimension
            comparison_types[dim] = comparison_types.get(dim, 0) + 1
        
        for dim, count in comparison_types.items():
            lines.append(f"- {count} {dim} comparison{'s' if count > 1 else ''}")
        
        lines.append("")
        
        # Add each comparison
        for i, comp in enumerate(comparisons, 1):
            lines.append(f"## {i}. {comp.name}")
            lines.append(f"**Dimension**: {comp.primary_dimension}")
            if comp.description:
                lines.append(f"{comp.description}")
            lines.append("")
            
            # Add summary table
            lines.append("### Results")
            
            if not comp.summary or comp.summary.get("status") != "success":
                lines.append("*No analysis results available*")
                lines.append("")
                continue
            
            # Create tables for each metric
            for metric, metric_data in comp.summary.get("metrics", {}).items():
                lines.append(f"#### {metric.replace('_', ' ').title()}")
                lines.append("")
                
                # Create table header
                lines.append(f"| {comp.primary_dimension.title()} | Value | Difference from Best |")
                lines.append("| --- | --- | --- |")
                
                # Get data points sorted by value (descending)
                best_value = metric_data.get("best_value", 0)
                
                # Add data rows
                for dim_value, data in comp.comparison_data.items():
                    value = data["metrics"].get(metric)
                    if value is None:
                        continue
                        
                    # Calculate difference from best
                    diff = value - best_value
                    diff_str = f"{diff:.4f}" if diff != 0 else "Best"
                    
                    lines.append(f"| {dim_value} | {value:.4f} | {diff_str} |")
                
                lines.append("")
                
                # Add statistical significance info
                if metric_data.get("pairwise_comparisons"):
                    lines.append("**Statistical Significance:**")
                    lines.append("")
                    
                    significant_comparisons = [
                        (key, data) for key, data in metric_data["pairwise_comparisons"].items()
                        if data.get("statistically_significant", False)
                    ]
                    
                    if significant_comparisons:
                        for key, data in significant_comparisons:
                            html.append(f"<li class='significant'>{key.replace('_vs_', ' is significantly different from ')} (p={data['p_value']:.4f})</li>")
                    else:
                        html.append("<li>No statistically significant differences found</li>")
                    
                    html.append("</ul>")
            
            # Add visualization suggestions
            if comp.visualization_suggestions:
                html.append("<h3>Visualization Suggestions</h3>")
                html.append("<ul>")
                
                for viz in comp.visualization_suggestions:
                    html.append(f"<li>{viz['type']}: {viz['title']}</li>")
                
                html.append("</ul>")
        
        # Add footer
        html.append("<hr>")
        html.append(f"<p>Report generated by ResultsCollector on {datetime.now().strftime('%Y-%m-%d')}</p>")
        
        # Close HTML
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def analyze_quantization_impact(
        self,
        field: str,
        prompt_name: Optional[str] = None,
        model_name: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        include_memory_usage: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze the impact of different quantization strategies.
        
        Args:
            field: Field to analyze
            prompt_name: Optional prompt name to filter by
            model_name: Optional model name to filter by
            metrics: List of metrics to analyze (if None, use defaults)
            include_memory_usage: Whether to include memory usage analysis
            
        Returns:
            Dictionary with quantization impact analysis
        """
        # Set default metrics if not specified
        if metrics is None:
            metrics = ["accuracy", "avg_processing_time"]
        
        # Create comparison first
        comparison = self.compare_quantization_strategies(
            field=field,
            prompt_name=prompt_name,
            metrics=metrics
        )
        
        # Additional analysis beyond comparison
        impact_analysis = {
            "comparison": comparison.to_dict(),
            "tradeoffs": {},
            "recommendations": [],
            "memory_analysis": {} if include_memory_usage else None
        }
        
        # Add tradeoff analysis
        if "accuracy" in metrics and "avg_processing_time" in metrics:
            tradeoffs = self._analyze_accuracy_speed_tradeoff(comparison)
            impact_analysis["tradeoffs"]["accuracy_vs_speed"] = tradeoffs
            
            # Generate recommendations based on tradeoffs
            recommendations = self._generate_quantization_recommendations(tradeoffs)
            impact_analysis["recommendations"].extend(recommendations)
        
        # Add memory usage analysis if requested
        if include_memory_usage:
            memory_analysis = self._analyze_memory_usage(comparison)
            impact_analysis["memory_analysis"] = memory_analysis
            
            # Generate additional recommendations based on memory usage
            memory_recommendations = self._generate_memory_recommendations(memory_analysis)
            impact_analysis["recommendations"].extend(memory_recommendations)
        
        return impact_analysis
    
    def _analyze_accuracy_speed_tradeoff(
        self, 
        comparison: ComparisonResult
    ) -> Dict[str, Any]:
        """
        Analyze the tradeoff between accuracy and speed.
        
        Args:
            comparison: Comparison result
            
        Returns:
            Dictionary with tradeoff analysis
        """
        # Check if comparison contains necessary metrics
        has_accuracy = all("accuracy" in data["metrics"] for data in comparison.comparison_data.values())
        has_time = all("avg_processing_time" in data["metrics"] for data in comparison.comparison_data.values())
        
        if not (has_accuracy and has_time):
            return {
                "status": "insufficient_data",
                "message": "Not all strategies have both accuracy and speed metrics"
            }
        
        # Calculate tradeoff score for each strategy
        # Higher score is better (normalized to 0-1 scale)
        tradeoffs = {}
        
        # Get min/max values for normalization
        acc_values = [data["metrics"]["accuracy"] for data in comparison.comparison_data.values()]
        time_values = [data["metrics"]["avg_processing_time"] for data in comparison.comparison_data.values()]
        
        min_acc, max_acc = min(acc_values), max(acc_values)
        min_time, max_time = min(time_values), max(time_values)
        
        # Avoid division by zero
        acc_range = max_acc - min_acc
        time_range = max_time - min_time
        
        if acc_range == 0:
            acc_range = 1.0
        if time_range == 0:
            time_range = 1.0
        
        for strategy, data in comparison.comparison_data.items():
            # Normalize values to 0-1 scale
            norm_acc = (data["metrics"]["accuracy"] - min_acc) / acc_range
            # Invert time so lower is better
            norm_time = 1.0 - ((data["metrics"]["avg_processing_time"] - min_time) / time_range)
            
            # Calculate balanced score (equal weight to accuracy and speed)
            balanced_score = (norm_acc + norm_time) / 2
            
            # Calculate accuracy-prioritized score (75% accuracy, 25% speed)
            accuracy_score = (norm_acc * 0.75) + (norm_time * 0.25)
            
            # Calculate speed-prioritized score (25% accuracy, 75% speed)
            speed_score = (norm_acc * 0.25) + (norm_time * 0.75)
            
            tradeoffs[strategy] = {
                "normalized_accuracy": norm_acc,
                "normalized_speed": norm_time,
                "balanced_score": balanced_score,
                "accuracy_priority_score": accuracy_score,
                "speed_priority_score": speed_score,
                "raw_accuracy": data["metrics"]["accuracy"],
                "raw_processing_time": data["metrics"]["avg_processing_time"]
            }
        
        # Find best strategies for different priorities
        best_balanced = max(tradeoffs.items(), key=lambda x: x[1]["balanced_score"])
        best_accuracy = max(tradeoffs.items(), key=lambda x: x[1]["accuracy_priority_score"])
        best_speed = max(tradeoffs.items(), key=lambda x: x[1]["speed_priority_score"])
        
        return {
            "status": "success",
            "strategy_scores": tradeoffs,
            "best_balanced": best_balanced[0],
            "best_accuracy": best_accuracy[0],
            "best_speed": best_speed[0]
        }
    
    def _analyze_memory_usage(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """
        Analyze memory usage for different quantization strategies.
        
        Args:
            comparison: Comparison result
            
        Returns:
            Dictionary with memory usage analysis
        """
        # Extract memory usage from metadata
        memory_usage = {}
        
        for strategy, data in comparison.comparison_data.items():
            metadata = data.get("metadata", {})
            if not metadata:
                continue
                
            # Try to find memory usage in metadata
            memory_info = None
            
            # Look for memory usage in different potential locations
            if "memory_used_gb" in metadata:
                memory_info = metadata["memory_used_gb"]
            elif "memory_info" in metadata and "used_gb" in metadata["memory_info"]:
                memory_info = metadata["memory_info"]["used_gb"]
            
            if memory_info is not None:
                memory_usage[strategy] = memory_info
        
        if not memory_usage:
            return {
                "status": "insufficient_data",
                "message": "No memory usage information available"
            }
        
        # Calculate memory savings relative to full precision
        memory_savings = {}
        
        # Find highest memory usage (presumably the full precision model)
        max_memory = max(memory_usage.values())
        
        for strategy, memory in memory_usage.items():
            savings_percent = ((max_memory - memory) / max_memory) * 100 if max_memory > 0 else 0
            memory_savings[strategy] = {
                "memory_gb": memory,
                "savings_gb": max_memory - memory,
                "savings_percent": savings_percent
            }
        
        # Find strategy with best memory savings
        best_savings = min(memory_usage.items(), key=lambda x: x[1])
        
        return {
            "status": "success",
            "memory_usage": memory_usage,
            "memory_savings": memory_savings,
            "best_savings": best_savings[0]
        }
    
    def _generate_quantization_recommendations(
        self, 
        tradeoffs: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate recommendations based on tradeoff analysis.
        
        Args:
            tradeoffs: Tradeoff analysis dictionary
            
        Returns:
            List of recommendation dictionaries
        """
        if tradeoffs.get("status") != "success":
            return []
        
        recommendations = []
        
        # Add recommendation for balanced use case
        recommendations.append({
            "type": "balanced",
            "strategy": tradeoffs["best_balanced"],
            "description": f"For balanced performance considering both accuracy and speed, use {tradeoffs['best_balanced']} quantization."
        })
        
        # Add recommendation for accuracy-focused use case
        if tradeoffs["best_accuracy"] != tradeoffs["best_balanced"]:
            recommendations.append({
                "type": "accuracy",
                "strategy": tradeoffs["best_accuracy"],
                "description": f"For accuracy-critical tasks, use {tradeoffs['best_accuracy']} quantization."
            })
        
        # Add recommendation for speed-focused use case
        if tradeoffs["best_speed"] != tradeoffs["best_balanced"]:
            recommendations.append({
                "type": "speed",
                "strategy": tradeoffs["best_speed"],
                "description": f"For speed-critical tasks or real-time processing, use {tradeoffs['best_speed']} quantization."
            })
        
        return recommendations
    
    def _generate_memory_recommendations(
        self, 
        memory_analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate recommendations based on memory analysis.
        
        Args:
            memory_analysis: Memory analysis dictionary
            
        Returns:
            List of recommendation dictionaries
        """
        if memory_analysis.get("status") != "success":
            return []
        
        recommendations = []
        
        # Add recommendation for memory-constrained environments
        best_savings = memory_analysis.get("best_savings")
        if best_savings:
            recommendations.append({
                "type": "memory_optimization",
                "strategy": best_savings,
                "description": f"For memory-constrained environments, use {best_savings} quantization to minimize memory usage."
            })
        
        # Add general recommendation about memory vs. performance tradeoff
        recommendations.append({
            "type": "general",
            "strategy": None,
            "description": "Consider hardware constraints when selecting quantization strategy. Lower precision generally uses less memory but may impact accuracy."
        })
        
        return recommendations
    
    def cleanup_temporary_files(self) -> None:
        """
        Clean up temporary files created during the experiment.
        """
        # This would clean up any temporary files created during the experiment
        # For now, simply log the action
        logger.info("Cleaning up temporary files")
    
    # ----- UTILITY METHODS -----
    
    def calculate_statistical_significance(
        self,
        values1: List[float],
        values2: List[float],
        alpha: float = 0.05
    ) -> Tuple[bool, float]:
        """
        Calculate statistical significance between two sets of values.
        
        Args:
            values1: First set of values
            values2: Second set of values
            alpha: Significance level (default: 0.05)
            
        Returns:
            Tuple of (is_significant, p_value)
        """
        if not values1 or not values2:
            return False, 1.0
        
        try:
            # Use t-test for independent samples
            t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
            
            # Determine significance
            is_significant = p_value < alpha
            
            return is_significant, p_value
        except Exception as e:
            logger.warning(f"Error in statistical test: {e}")
            return False, 1.0
        
    def calculate_confidence_interval(
        self,
        values: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for a set of values.
        
        Args:
            values: List of values
            confidence: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        if not values:
            return 0.0, 0.0, 0.0
        
        try:
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            
            if len(values) <= 1 or stdev == 0.0:
                return mean, mean, mean
            
            # Calculate confidence interval
            degrees_freedom = len(values) - 1
            t_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
            
            margin_error = t_value * (stdev / math.sqrt(len(values)))
            
            lower_bound = mean - margin_error
            upper_bound = mean + margin_error
            
            return mean, lower_bound, upper_bound
        except Exception as e:
            logger.warning(f"Error calculating confidence interval: {e}")
            return statistics.mean(values), 0.0, 0.0
    
    def _generate_html_report(self, comparisons: List[ComparisonResult]) -> str:
        """
        Generate an HTML report from comparisons.
        
        Args:
            comparisons: List of comparisons to include
            
        Returns:
            HTML formatted report
        """
        # Create HTML header
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "  <title>Comparative Analysis Report</title>",
            "  <style>",
            "    body { font-family: Arial, sans-serif; margin: 20px; }",
            "    h1, h2, h3, h4 { color: #2c3e50; }",
            "    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "    th { background-color: #f2f2f2; }",
            "    .best { font-weight: bold; color: green; }",
            "    .worst { color: red; }",
            "    .significant { background-color: #ffffcc; }",
            "  </style>",
            "</head>",
            "<body>"
        ]
        
        # Add header
        html.append("<h1>Comparative Analysis Report</h1>")
        html.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html.append(f"<p>Experiment: {self.experiment_name}</p>")
        
        # Add summary
        html.append("<h2>Summary</h2>")
        html.append("<ul>")
        
        # Count comparison types
        comparison_types = {}
        for comp in comparisons:
            dim = comp.primary_dimension
            comparison_types[dim] = comparison_types.get(dim, 0) + 1
        
        for dim, count in comparison_types.items():
            html.append(f"<li>{count} {dim} comparison{'s' if count > 1 else ''}</li>")
        
        html.append("</ul>")
        
        # Add each comparison
        for i, comp in enumerate(comparisons, 1):
            html.append(f"<h2>{i}. {comp.name}</h2>")
            html.append(f"<p><strong>Dimension</strong>: {comp.primary_dimension}</p>")
            if comp.description:
                html.append(f"<p>{comp.description}</p>")
            
            # Add summary table
            html.append("<h3>Results</h3>")
            
            if not comp.summary or comp.summary.get("status") != "success":
                html.append("<p><em>No analysis results available</em></p>")
                continue
            
            # Create tables for each metric
            for metric, metric_data in comp.summary.get("metrics", {}).items():
                html.append(f"<h4>{metric.replace('_', ' ').title()}</h4>")
                
                # Create table
                html.append("<table>")
                html.append(f"<tr><th>{comp.primary_dimension.title()}</th><th>Value</th><th>Difference from Best</th></tr>")
                
                # Get data points sorted by value (descending)
                best_value = metric_data.get("best_value", 0)
                
                # Add data rows
                for dim_value, data in comp.comparison_data.items():
                    value = data["metrics"].get(metric)
                    if value is None:
                        continue
                        
                    # Calculate difference from best
                    diff = value - best_value
                    diff_str = f"{diff:.4f}" if diff != 0 else "Best"
                    
                    # Determine row class
                    row_class = ""
                    if dim_value == metric_data.get("best"):
                        row_class = "best"
                    elif dim_value == metric_data.get("worst"):
                        row_class = "worst"
                    
                    html.append(f"<tr class='{row_class}'>")
                    html.append(f"<td>{dim_value}</td>")
                    html.append(f"<td>{value:.4f}</td>")
                    html.append(f"<td>{diff_str}</td>")
                    html.append("</tr>")
                
                html.append("</table>")
                
                # Add statistical significance info
                if metric_data.get("pairwise_comparisons"):
                    html.append("<p><strong>Statistical Significance:</strong></p>")
                    html.append("<ul>")
                    
                    significant_comparisons = [
                        (key, data) for key, data in metric_data["pairwise_comparisons"].items()
                        if data.get("statistically_significant", False)
                    ]
                    
                    if significant_comparisons:
                        for key, data in significant_comparisons:
                            html.append(f"<li class='significant'>{key.replace('_vs_', ' is significantly different from ')} (p={data.get('p_value', 0):.4f})</li>")
                    else:
                        html.append("<li>No statistically significant differences found</li>")
                    
                    html.append("</ul>")
            
            # Add visualization suggestions
            if comp.visualization_suggestions:
                html.append("<h3>Visualization Suggestions</h3>")
                html.append("<ul>")
                
                for viz in comp.visualization_suggestions:
                    html.append(f"<li>{viz['type']}: {viz['title']}</li>")
                
                html.append("</ul>")
        
        # Add footer
        html.append("<hr>")
        html.append(f"<p>Report generated by ResultsCollector on {datetime.now().strftime('%Y-%m-%d')}</p>")
        
        # Close HTML
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)
    
    def calculate_effect_size(
        self,
        values1: List[float],
        values2: List[float]
    ) -> Tuple[float, str]:
        """
        Calculate Cohen's d effect size between two sets of values.
        
        Args:
            values1: First set of values
            values2: Second set of values
            
        Returns:
            Tuple of (effect_size, interpretation)
        """
        if not values1 or not values2:
            return 0.0, "unknown"
        
        try:
            # Calculate means
            mean1 = statistics.mean(values1)
            mean2 = statistics.mean(values2)
            
            # Calculate standard deviations
            if len(values1) <= 1 or len(values2) <= 1:
                return 0.0, "unknown"
                
            std1 = statistics.stdev(values1)
            std2 = statistics.stdev(values2)
            
            # Calculate pooled standard deviation
            n1, n2 = len(values1), len(values2)
            pooled_std = math.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            # Avoid division by zero
            if pooled_std == 0:
                return 0.0, "unknown"
                
            # Calculate Cohen's d
            d = abs(mean1 - mean2) / pooled_std
            
            # Interpret effect size
            if d < 0.2:
                interpretation = "negligible"
            elif d < 0.5:
                interpretation = "small"
            elif d < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"
                
            return d, interpretation
            
        except Exception as e:
            logger.warning(f"Error calculating effect size: {e}")
            return 0.0, "unknown"
    
    def generate_prompt_effectiveness_report(
        self,
        field: str,
        include_statistics: bool = True,
        max_prompts: int = 10
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report on prompt effectiveness for a field.
        
        Args:
            field: Field to analyze
            include_statistics: Whether to include detailed statistics
            max_prompts: Maximum number of prompts to include in report
            
        Returns:
            Dictionary with prompt effectiveness report
        """
        # Get field results
        field_results = self.experiment_result.field_results.get(field, [])
        
        if not field_results:
            return {
                "status": "no_data",
                "message": f"No results found for field: {field}"
            }
        
        # Sort prompts by effectiveness (accuracy)
        sorted_prompts = sorted(
            field_results, 
            key=lambda p: p.accuracy,
            reverse=True
        )
        
        # Limit number of prompts if needed
        if max_prompts > 0:
            sorted_prompts = sorted_prompts[:max_prompts]
        
        # Create basic report
        report = {
            "field": field,
            "prompts": [
                {
                    "name": p.prompt_name,
                    "accuracy": p.accuracy,
                    "character_error_rate": p.avg_character_error_rate,
                    "processing_time": p.avg_processing_time,
                    "total_items": p.total_items,
                    "successful_extractions": p.successful_extractions
                }
                for p in sorted_prompts
            ],
            "best_prompt": sorted_prompts[0].prompt_name if sorted_prompts else None,
            "total_prompts": len(sorted_prompts),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add detailed statistics if requested
        if include_statistics and len(sorted_prompts) >= 2:
            # Compare best prompt to others
            best_prompt = sorted_prompts[0]
            
            # Extract performance values for each metric
            statistics = {
                "exact_match_statistics": {},
                "error_rate_statistics": {},
                "processing_time_statistics": {}
            }
            
            # Extract exact match results for best prompt
            best_exact_matches = [
                1.0 if r.exact_match else 0.0
                for r in best_prompt.results
            ]
            
            # Compare with other prompts
            for i, prompt in enumerate(sorted_prompts[1:], 1):
                # Skip if too few results
                if not prompt.results or not best_prompt.results:
                    continue
                    
                # Extract exact match results for this prompt
                prompt_exact_matches = [
                    1.0 if r.exact_match else 0.0
                    for r in prompt.results
                ]
                
                # Calculate statistical significance
                significant, p_value = self.calculate_statistical_significance(
                    best_exact_matches,
                    prompt_exact_matches
                )
                
                # Calculate effect size
                effect_size, effect_interpretation = self.calculate_effect_size(
                    best_exact_matches,
                    prompt_exact_matches
                )
                
                # Add to statistics
                statistics["exact_match_statistics"][prompt.prompt_name] = {
                    "significantly_different": significant,
                    "p_value": p_value,
                    "effect_size": effect_size,
                    "effect_interpretation": effect_interpretation,
                    "difference": best_prompt.accuracy - prompt.accuracy
                }
                
                # Extract character error rates
                best_error_rates = [
                    r.character_error_rate
                    for r in best_prompt.results
                    if hasattr(r, 'character_error_rate') and r.character_error_rate is not None
                ]
                
                prompt_error_rates = [
                    r.character_error_rate
                    for r in prompt.results
                    if hasattr(r, 'character_error_rate') and r.character_error_rate is not None
                ]
                
                # Calculate statistical significance for error rates if enough data
                if best_error_rates and prompt_error_rates:
                    significant, p_value = self.calculate_statistical_significance(
                        best_error_rates,
                        prompt_error_rates
                    )
                    
                    # Calculate effect size
                    effect_size, effect_interpretation = self.calculate_effect_size(
                        best_error_rates,
                        prompt_error_rates
                    )
                    
                    # Add to statistics
                    statistics["error_rate_statistics"][prompt.prompt_name] = {
                        "significantly_different": significant,
                        "p_value": p_value,
                        "effect_size": effect_size,
                        "effect_interpretation": effect_interpretation,
                        "difference": best_prompt.avg_character_error_rate - prompt.avg_character_error_rate
                    }
                
                # Extract processing times
                best_times = [
                    r.processing_time
                    for r in best_prompt.results
                    if hasattr(r, 'processing_time') and r.processing_time is not None
                ]
                
                prompt_times = [
                    r.processing_time
                    for r in prompt.results
                    if hasattr(r, 'processing_time') and r.processing_time is not None
                ]
                
                # Calculate statistical significance for processing times if enough data
                if best_times and prompt_times:
                    significant, p_value = self.calculate_statistical_significance(
                        best_times,
                        prompt_times
                    )
                    
                    # Calculate effect size
                    effect_size, effect_interpretation = self.calculate_effect_size(
                        best_times,
                        prompt_times
                    )
                    
                    # Add to statistics
                    statistics["processing_time_statistics"][prompt.prompt_name] = {
                        "significantly_different": significant,
                        "p_value": p_value,
                        "effect_size": effect_size,
                        "effect_interpretation": effect_interpretation,
                        "difference": best_prompt.avg_processing_time - prompt.avg_processing_time
                    }
            
            # Add statistics to report
            report["statistics"] = statistics
            
            # Add summary of statistical findings
            significant_differences = sum(
                1 for s in statistics["exact_match_statistics"].values()
                if s.get("significantly_different", False)
            )
            
            report["statistical_summary"] = {
                "prompts_with_significant_differences": significant_differences,
                "total_comparisons": len(statistics["exact_match_statistics"])
            }
        
        return report
    
    def load_multiple_experiments(
        self,
        experiment_paths: List[Union[str, Path]]
    ) -> Dict[str, ExperimentResult]:
        """
        Load multiple experiments at once for comparison.
        
        Args:
            experiment_paths: List of paths to experiment directories
            
        Returns:
            Dictionary mapping experiment IDs to loaded ExperimentResult objects
        """
        experiment_loader = ExperimentLoader()
        results = {}
        
        for path in experiment_paths:
            exp_data = experiment_loader.load_experiment(path)
            if exp_data:
                exp_result, _ = exp_data
                experiment_id = exp_result.experiment_id or str(Path(path).name)
                results[experiment_id] = exp_result
        
        return results
    
    def find_similar_experiments(
        self,
        model_name: Optional[str] = None,
        field: Optional[str] = None,
        prompt_name: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find experiments similar to the current one.
        
        Args:
            model_name: Optional model name filter
            field: Optional field name filter
            prompt_name: Optional prompt name filter
            limit: Maximum number of experiments to return
            
        Returns:
            List of similar experiment metadata
        """
        # Get the current experiment metadata for comparison
        current_metadata = {
            'model': model_name or getattr(self, 'model_name', None),
            'fields': [field] if field else getattr(self, 'fields', []),
            'prompts': [prompt_name] if prompt_name else getattr(self, 'prompts', [])
        }
        
        # Find similar experiments
        experiment_loader = ExperimentLoader()
        all_experiments = experiment_loader.discover_experiments()
        
        # Filter and score experiments by similarity
        scored_experiments = []
        for exp_id, metadata in all_experiments.items():
            # Skip if this is the current experiment
            if exp_id == getattr(self, 'experiment_id', None):
                continue
                
            # Calculate similarity score (simple version)
            score = 0
            
            # Model match is important
            if current_metadata['model'] and metadata.get('model') == current_metadata['model']:
                score += 3
            
            # Field overlap
            if current_metadata['fields']:
                common_fields = set(current_metadata['fields']).intersection(metadata.get('fields', []))
                score += len(common_fields)
            
            # Prompt overlap
            if current_metadata['prompts']:
                common_prompts = set(current_metadata['prompts']).intersection(metadata.get('prompts', []))
                score += len(common_prompts)
            
            # Add if has some similarity
            if score > 0:
                scored_experiments.append((score, exp_id, metadata))
        
        # Sort by similarity score and limit results
        scored_experiments.sort(reverse=True)
        
        return [{'experiment_id': exp_id, **metadata} for _, exp_id, metadata in scored_experiments[:limit]]
    
    def compare_with_experiment(
        self,
        other_experiment_id: str,
        field: Optional[str] = None,
        prompt_name: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> ComparisonResult:
        """
        Compare current experiment with another specific experiment.
        
        Args:
            other_experiment_id: ID of experiment to compare with
            field: Optional field to focus comparison on
            prompt_name: Optional prompt to focus comparison on
            metrics: Optional list of metrics to include
            
        Returns:
            Comparison result between experiments
        """
        # Ensure we have the current experiment's result
        if not hasattr(self, 'experiment_result'):
            self.generate_experiment_result()
        
        # Create comparator and load the other experiment
        experiment_loader = ExperimentLoader()
        experiment_comparator = ExperimentComparator(experiment_loader)
        
        # Use current experiment ID and the other one
        current_exp_id = getattr(self, 'experiment_id', 'current_experiment')
        
        return experiment_comparator.compare_experiments(
            experiment_ids=[current_exp_id, other_experiment_id],
            field=field,
            prompt_name=prompt_name,
            metrics=metrics,
            name=f"Comparison: {current_exp_id} vs {other_experiment_id}"
        )
    
    def generate_experiment_dashboard(
        self,
        include_similar_experiments: bool = True,
        include_field_comparisons: bool = True,
        include_prompt_comparisons: bool = True,
        format: str = "html"
    ) -> str:
        """
        Generate a comprehensive dashboard for the current experiment.
        
        Args:
            include_similar_experiments: Whether to include similar experiments
            include_field_comparisons: Whether to include field comparisons
            include_prompt_comparisons: Whether to include prompt comparisons
            format: Output format ('html' or 'markdown')
            
        Returns:
            Dashboard content in the specified format
        """
        # Ensure we have the experiment result
        if not hasattr(self, 'experiment_result'):
            self.generate_experiment_result()
        
        # Collect dashboard components
        components = []
        
        # Basic experiment information
        experiment_info = {
            "name": getattr(self, 'experiment_name', 'Unnamed Experiment'),
            "id": getattr(self, 'experiment_id', ''),
            "timestamp": datetime.now().isoformat(),
            "model": getattr(self, 'model_name', ''),
            "fields": getattr(self, 'fields', []),
            "prompts": getattr(self, 'prompts', []),
            "quantization": getattr(self, 'quantization', None)
        }
        components.append(("experiment_info", experiment_info))
        
        # Add field metrics
        field_metrics = {}
        if hasattr(self, 'experiment_result') and hasattr(self.experiment_result, 'field_results'):
            for field, field_data in self.experiment_result.field_results.items():
                field_metrics[field] = {
                    "average_accuracy": field_data.average_accuracy if hasattr(field_data, 'average_accuracy') else None,
                    "sample_count": field_data.sample_count if hasattr(field_data, 'sample_count') else 0,
                    "prompt_count": len(field_data.prompt_performances) if hasattr(field_data, 'prompt_performances') else 0
                }
        components.append(("field_metrics", field_metrics))
        
        # Find similar experiments if requested
        if include_similar_experiments:
            similar_experiments = self.find_similar_experiments()
            components.append(("similar_experiments", similar_experiments))
        
        # Generate field comparisons if requested
        if include_field_comparisons and len(getattr(self, 'fields', [])) > 1:
            field_comparison = self.compare_fields()
            components.append(("field_comparison", field_comparison))
        
        # Generate prompt comparisons if requested
        if include_prompt_comparisons:
            prompt_comparisons = {}
            for field in getattr(self, 'fields', []):
                prompt_comparison = self.compare_prompts(field=field)
                if prompt_comparison:
                    prompt_comparisons[field] = prompt_comparison
            components.append(("prompt_comparisons", prompt_comparisons))
        
        # Generate HTML or Markdown
        if format.lower() == 'html':
            return self._generate_dashboard_html(components)
        else:
            return self._generate_dashboard_markdown(components)
    
    def _generate_dashboard_html(self, components: List[Tuple[str, Any]]) -> str:
        """
        Generate HTML version of the experiment dashboard.
        
        Args:
            components: List of dashboard components
            
        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; padding: 20px; }}
                .metric {{ display: inline-block; text-align: center; background: #f8f9fa; border-radius: 4px; padding: 10px; margin: 10px; min-width: 120px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .highlight {{ background-color: #e8f4f8; }}
                .chart-container {{ margin: 20px 0; height: 300px; }}
            </style>
        </head>
        <body>
            <div class="container">
        """
        
        # Process each component
        for component_type, component_data in components:
            if component_type == "experiment_info":
                html += self._render_experiment_info_html(component_data)
            elif component_type == "field_metrics":
                html += self._render_field_metrics_html(component_data)
            elif component_type == "similar_experiments":
                html += self._render_similar_experiments_html(component_data)
            elif component_type == "field_comparison":
                html += self._render_comparison_html(component_data, "Field Comparison")
            elif component_type == "prompt_comparisons":
                html += self._render_prompt_comparisons_html(component_data)
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _render_experiment_info_html(self, info: Dict[str, Any]) -> str:
        """Render experiment info as HTML."""
        return f"""
        <div class="card">
            <h1>{info.get('name', 'Unnamed Experiment')}</h1>
            <p><strong>ID:</strong> {info.get('id', '')}</p>
            <p><strong>Date:</strong> {info.get('timestamp', '').split('T')[0]}</p>
            <p><strong>Model:</strong> {info.get('model', '')}</p>
            <p><strong>Quantization:</strong> {info.get('quantization', 'None')}</p>
            <p><strong>Fields:</strong> {', '.join(info.get('fields', []))}</p>
            <p><strong>Prompts:</strong> {', '.join(info.get('prompts', []))}</p>
        </div>
        """
    
    def _render_field_metrics_html(self, metrics: Dict[str, Dict[str, Any]]) -> str:
        """Render field metrics as HTML."""
        html = """
        <div class="card">
            <h2>Field Metrics</h2>
            <table>
                <tr>
                    <th>Field</th>
                    <th>Average Accuracy</th>
                    <th>Sample Count</th>
                    <th>Prompt Count</th>
                </tr>
        """
        
        for field, field_metrics in metrics.items():
            accuracy = field_metrics.get('average_accuracy')
            accuracy_str = f"{accuracy:.2%}" if accuracy is not None else "N/A"
            
            html += f"""
            <tr>
                <td>{field}</td>
                <td>{accuracy_str}</td>
                <td>{field_metrics.get('sample_count', 0)}</td>
                <td>{field_metrics.get('prompt_count', 0)}</td>
            </tr>
            """
        
        html += """
            </table>
        </div>
        """
        return html
    
    def _render_similar_experiments_html(self, experiments: List[Dict[str, Any]]) -> str:
        """Render similar experiments as HTML."""
        if not experiments:
            return ""
            
        html = """
        <div class="card">
            <h2>Similar Experiments</h2>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Date</th>
                    <th>Model</th>
                    <th>Fields</th>
                </tr>
        """
        
        for exp in experiments:
            html += f"""
            <tr>
                <td>{exp.get('name', 'Unnamed')}</td>
                <td>{exp.get('timestamp', '').split('T')[0]}</td>
                <td>{exp.get('model', '')}</td>
                <td>{', '.join(exp.get('fields', []))}</td>
            </tr>
            """
        
        html += """
            </table>
        </div>
        """
        return html
    
    def _render_comparison_html(self, comparison: ComparisonResult, title: str) -> str:
        """Render comparison as HTML."""
        if not comparison.comparison_data:
            return ""
            
        html = f"""
        <div class="card">
            <h2>{title}</h2>
            <table>
                <tr>
                    <th>{comparison.primary_dimension.title()}</th>
        """
        
        # Add metric headers
        for metric in comparison.metrics:
            html += f"<th>{metric.replace('_', ' ').title()}</th>"
        
        html += "</tr>"
        
        # Add rows for each dimension value
        for dim_value, data in comparison.comparison_data.items():
            html += f"<tr><td>{dim_value}</td>"
            
            for metric in comparison.metrics:
                metric_value = data.get('metrics', {}).get(metric)
                if metric_value is not None:
                    # Format based on metric type
                    if metric in ['exact_match', 'accuracy', 'precision', 'recall', 'f1_score']:
                        html += f"<td>{metric_value:.2%}</td>"
                    elif metric == 'character_error_rate':
                        html += f"<td>{metric_value:.4f}</td>"
                    elif metric == 'processing_time':
                        html += f"<td>{metric_value:.2f}s</td>"
                    else:
                        html += f"<td>{metric_value}</td>"
                else:
                    html += "<td>N/A</td>"
            
            html += "</tr>"
        
        html += """
            </table>
        </div>
        """
        return html
    
    def _render_prompt_comparisons_html(self, comparisons: Dict[str, ComparisonResult]) -> str:
        """Render prompt comparisons as HTML."""
        if not comparisons:
            return ""
            
        html = """
        <div class="card">
            <h2>Prompt Comparisons</h2>
        """
        
        for field, comparison in comparisons.items():
            html += f"<h3>Field: {field}</h3>"
            html += self._render_comparison_html(comparison, "")
        
        html += "</div>"
        return html
    
    def _generate_dashboard_markdown(self, components: List[Tuple[str, Any]]) -> str:
        """
        Generate Markdown version of the experiment dashboard.
        
        Args:
            components: List of dashboard components
            
        Returns:
            Markdown string
        """
        markdown = "# Experiment Dashboard\n\n"
        
        # Process each component
        for component_type, component_data in components:
            if component_type == "experiment_info":
                markdown += self._render_experiment_info_markdown(component_data)
            elif component_type == "field_metrics":
                markdown += self._render_field_metrics_markdown(component_data)
            elif component_type == "similar_experiments":
                markdown += self._render_similar_experiments_markdown(component_data)
            elif component_type == "field_comparison":
                markdown += self._render_comparison_markdown(component_data, "Field Comparison")
            elif component_type == "prompt_comparisons":
                markdown += self._render_prompt_comparisons_markdown(component_data)
        
        return markdown
    
    def _render_experiment_info_markdown(self, info: Dict[str, Any]) -> str:
        """Render experiment info as Markdown."""
        return f"""
## {info.get('name', 'Unnamed Experiment')}

- **ID:** {info.get('id', '')}
- **Date:** {info.get('timestamp', '').split('T')[0]}
- **Model:** {info.get('model', '')}
- **Quantization:** {info.get('quantization', 'None')}
- **Fields:** {', '.join(info.get('fields', []))}
- **Prompts:** {', '.join(info.get('prompts', []))}

"""
    
    def _render_field_metrics_markdown(self, metrics: Dict[str, Dict[str, Any]]) -> str:
        """Render field metrics as Markdown."""
        markdown = "## Field Metrics\n\n"
        markdown += "| Field | Average Accuracy | Sample Count | Prompt Count |\n"
        markdown += "|-------|-----------------|--------------|-------------|\n"
        
        for field, field_metrics in metrics.items():
            accuracy = field_metrics.get('average_accuracy')
            accuracy_str = f"{accuracy:.2%}" if accuracy is not None else "N/A"
            
            markdown += f"| {field} | {accuracy_str} | {field_metrics.get('sample_count', 0)} | {field_metrics.get('prompt_count', 0)} |\n"
        
        markdown += "\n"
        return markdown
    
    def _render_similar_experiments_markdown(self, experiments: List[Dict[str, Any]]) -> str:
        """Render similar experiments as Markdown."""
        if not experiments:
            return ""
            
        markdown = "## Similar Experiments\n\n"
        markdown += "| Name | Date | Model | Fields |\n"
        markdown += "|------|------|-------|--------|\n"
        
        for exp in experiments:
            markdown += f"| {exp.get('name', 'Unnamed')} | {exp.get('timestamp', '').split('T')[0]} | {exp.get('model', '')} | {', '.join(exp.get('fields', []))} |\n"
        
        markdown += "\n"
        return markdown
    
    def _render_comparison_markdown(self, comparison: ComparisonResult, title: str) -> str:
        """Render comparison as Markdown."""
        if not comparison.comparison_data:
            return ""
            
        markdown = f"## {title}\n\n"
        markdown += f"| {comparison.primary_dimension.title()} |"
        
        # Add metric headers
        for metric in comparison.metrics:
            markdown += f" {metric.replace('_', ' ').title()} |"
        
        markdown += "\n|"
        
        # Add header separator
        for _ in range(len(comparison.metrics) + 1):
            markdown += "---|"
        
        markdown += "\n"
        
        # Add rows for each dimension value
        for dim_value, data in comparison.comparison_data.items():
            markdown += f"| {dim_value} |"
            
            for metric in comparison.metrics:
                metric_value = data.get('metrics', {}).get(metric)
                if metric_value is not None:
                    # Format based on metric type
                    if metric in ['exact_match', 'accuracy', 'precision', 'recall', 'f1_score']:
                        markdown += f" {metric_value:.2%} |"
                    elif metric == 'character_error_rate':
                        markdown += f" {metric_value:.4f} |"
                    elif metric == 'processing_time':
                        markdown += f" {metric_value:.2f}s |"
                    else:
                        markdown += f" {metric_value} |"
                else:
                    markdown += " N/A |"
            
            markdown += "\n"
        
        markdown += "\n"
        return markdown
    
    def _render_prompt_comparisons_markdown(self, comparisons: Dict[str, ComparisonResult]) -> str:
        """Render prompt comparisons as Markdown."""
        if not comparisons:
            return ""
            
        markdown = "## Prompt Comparisons\n\n"
        
        for field, comparison in comparisons.items():
            markdown += f"### Field: {field}\n\n"
            markdown += self._render_comparison_markdown(comparison, "")
        
        return markdown
    
    def load_experiment(self, experiment_path: Union[str, Path]) -> Optional[ExperimentResult]:
        """
        Load an experiment result from a specified path.
        
        Args:
            experiment_path: Path to the experiment directory
            
        Returns:
            Loaded experiment result or None if loading fails
        """
        try:
            path = Path(experiment_path)
            result_file = path / "experiment_result.json"
            
            if not result_file.exists():
                logger.warning(f"Experiment result file not found: {result_file}")
                return None
                
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            # Convert to ExperimentResult object
            result = ExperimentResult(
                experiment_id=data.get('experiment_id', str(path.name)),
                experiment_name=data.get('experiment_name', str(path.name)),
                model_name=data.get('model_name', ''),
                timestamp=data.get('timestamp', ''),
                average_accuracy=data.get('average_accuracy'),
                field_results={}
            )
            
            # Add field results
            for field, field_data in data.get('field_results', {}).items():
                result.field_results[field] = self._deserialize_field_result(field_data)
                
            return result
        except Exception as e:
            logger.error(f"Error loading experiment: {str(e)}")
            return None
            
    def _deserialize_field_result(self, field_data: Dict) -> FieldResult:
        """Helper to deserialize field result data"""
        field_result = FieldResult(
            field_name=field_data.get('field_name', ''),
            average_accuracy=field_data.get('average_accuracy'),
            sample_count=field_data.get('sample_count', 0),
            prompt_performances={}
        )
        
        # Add prompt performances
        for prompt, prompt_data in field_data.get('prompt_performances', {}).items():
            field_result.prompt_performances[prompt] = self._deserialize_prompt_performance(prompt_data)
            
        return field_result
        
    def _deserialize_prompt_performance(self, prompt_data: Dict) -> PromptPerformance:
        """Helper to deserialize prompt performance data"""
        return PromptPerformance(
            prompt_name=prompt_data.get('prompt_name', ''),
            accuracy=prompt_data.get('accuracy'),
            exact_match=prompt_data.get('exact_match'),
            character_error_rate=prompt_data.get('character_error_rate'),
            processing_time=prompt_data.get('processing_time'),
            sample_count=prompt_data.get('sample_count', 0)
        )
    
    def compare_experiments(
        self, 
        other_experiment_paths: List[Union[str, Path]],
        field: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compare this experiment with other experiments.
        
        Args:
            other_experiment_paths: List of paths to other experiments to compare with
            field: Optional field to focus comparison on
            metrics: Optional list of metrics to include in comparison
            name: Optional name for the comparison
            
        Returns:
            Comparison result between experiments
        """
        # Ensure we have the current experiment's result
        if not self.experiment_result:
            self.generate_experiment_result()
            
        # Default metrics if not specified
        if not metrics:
            metrics = ["accuracy", "exact_match", "character_error_rate", "processing_time"]
            
        # Create comparison result
        comparison = ComparisonResult(
            primary_dimension="experiment",
            metrics=metrics,
            name=name or f"Experiment Comparison ({len(other_experiment_paths) + 1} experiments)"
        )
        
        # Add current experiment data
        self._add_experiment_to_comparison(comparison, self.experiment_result, field)
        
        # Load and add other experiments
        for path in other_experiment_paths:
            other_result = self.load_experiment(path)
            if other_result:
                self._add_experiment_to_comparison(comparison, other_result, field)
                
        # Calculate statistics and generate visualization suggestions
        comparison.calculate_statistics()
                
        return comparison
        
    def _add_experiment_to_comparison(
        self, 
        comparison: ComparisonResult,
        experiment: ExperimentResult,
        field: Optional[str] = None
    ) -> None:
        """Helper to add experiment data to a comparison"""
        # Determine experiment name for the comparison
        exp_name = experiment.experiment_name or experiment.experiment_id or "Unnamed Experiment"
        
        # If field is specified, only compare that field
        if field and field in experiment.field_results:
            field_result = experiment.field_results[field]
            metrics_data = {
                "accuracy": field_result.average_accuracy,
                "sample_count": field_result.sample_count
            }
            
            # Add best prompt metrics
            best_prompt = self._find_best_prompt(field_result)
            if best_prompt:
                prompt_perf = field_result.prompt_performances[best_prompt]
                metrics_data.update({
                    "exact_match": prompt_perf.exact_match,
                    "character_error_rate": prompt_perf.character_error_rate,
                    "processing_time": prompt_perf.processing_time
                })
                
            comparison.add_data_point(
                dimension_value=exp_name,
                metrics=metrics_data,
                sample_size=field_result.sample_count
            )
        # Otherwise, use overall experiment metrics
        else:
            comparison.add_data_point(
                dimension_value=exp_name,
                metrics={
                    "accuracy": experiment.average_accuracy,
                    "sample_count": sum(fr.sample_count for fr in experiment.field_results.values())
                },
                sample_size=sum(fr.sample_count for fr in experiment.field_results.values())
            )
            
    def _find_best_prompt(self, field_result: FieldResult) -> Optional[str]:
        """Find the best performing prompt for a field result"""
        if not field_result.prompt_performances:
            return None
            
        return max(
            field_result.prompt_performances.items(),
            key=lambda x: x[1].accuracy or 0
        )[0]
        
    def find_similar_experiments(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        model_name: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find experiments similar to the current one based on model, fields, etc.
        
        Args:
            base_dir: Optional base directory to search for experiments
            model_name: Optional model name filter (defaults to current experiment's model)
            limit: Maximum number of experiments to return
            
        Returns:
            List of similar experiment metadata
        """
        # Use current model name if not specified
        model_filter = model_name or self.model_name
        
        # Determine base directory
        search_dir = Path(base_dir) if base_dir else Path(self.base_path).parent
        
        similar_experiments = []
        
        # Scan for experiment directories
        for exp_dir in search_dir.glob("*"):
            if not exp_dir.is_dir():
                continue
                
            # Skip current experiment
            if exp_dir.name == self.experiment_name:
                continue
                
            # Check for experiment result file
            result_file = exp_dir / "experiment_result.json"
            if not result_file.exists():
                continue
                
            try:
                # Load metadata
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    
                # Check if model matches
                if model_filter and data.get('model_name') != model_filter:
                    continue
                    
                # Calculate similarity score
                similarity_score = self._calculate_experiment_similarity(data)
                
                if similarity_score > 0:
                    similar_experiments.append({
                        "experiment_id": data.get('experiment_id', exp_dir.name),
                        "experiment_name": data.get('experiment_name', exp_dir.name),
                        "model_name": data.get('model_name', ''),
                        "timestamp": data.get('timestamp', ''),
                        "similarity_score": similarity_score,
                        "path": str(exp_dir)
                    })
            except Exception as e:
                logger.debug(f"Error processing experiment {exp_dir.name}: {str(e)}")
                continue
                
        # Sort by similarity score and limit results
        similar_experiments.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar_experiments[:limit]
        
    def _calculate_experiment_similarity(self, experiment_data: Dict) -> float:
        """Calculate similarity score between current experiment and another one"""
        score = 0.0
        
        # Compare fields
        current_fields = set(self.fields)
        other_fields = set(experiment_data.get('field_results', {}).keys())
        
        field_overlap = len(current_fields.intersection(other_fields))
        if field_overlap > 0:
            score += 0.5 * (field_overlap / max(len(current_fields), len(other_fields)))
            
        # Compare prompts (if we have that info)
        current_prompts = set(self.prompts)
        
        if current_prompts and 'prompts' in experiment_data:
            other_prompts = set(experiment_data.get('prompts', []))
            prompt_overlap = len(current_prompts.intersection(other_prompts))
            if prompt_overlap > 0:
                score += 0.3 * (prompt_overlap / max(len(current_prompts), len(other_prompts)))
                
        # Same model gets bonus points
        if experiment_data.get('model_name') == self.model_name:
            score += 0.2
            
        return score
        
    def generate_experiment_dashboard(
        self,
        output_path: Optional[Union[str, Path]] = None,
        include_similar: bool = True
    ) -> str:
        """
        Generate a comprehensive HTML dashboard for this experiment.
        
        Args:
            output_path: Optional path to save the dashboard HTML
            include_similar: Whether to include similar experiments
            
        Returns:
            Path to the generated dashboard file
        """
        # Ensure we have experiment result
        if not self.experiment_result:
            self.generate_experiment_result()
            
        # Create dashboard content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Dashboard: {self.experiment_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .card {{ background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
                       margin-bottom: 20px; padding: 20px; }}
                .metric {{ display: inline-block; text-align: center; background: #f8f9fa; 
                         border-radius: 4px; padding: 15px; margin: 10px; min-width: 140px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 5px 0; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .highlight {{ background-color: #e8f4f8; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="card">
                    <h1>{self.experiment_name}</h1>
                    <p><strong>Experiment ID:</strong> {self.experiment_id}</p>
                    <p><strong>Model:</strong> {self.model_name}</p>
                    <p><strong>Date:</strong> {self.experiment_result.timestamp or 'N/A'}</p>
                    
                    <div class="metrics-overview">
                        <div class="metric">
                            <div class="metric-label">Overall Accuracy</div>
                            <div class="metric-value">{self.experiment_result.average_accuracy*100:.1f}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Fields</div>
                            <div class="metric-value">{len(self.experiment_result.field_results)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Samples</div>
                            <div class="metric-value">{sum(fr.sample_count for fr in self.experiment_result.field_results.values())}</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Field Performance</h2>
                    <table>
                        <tr>
                            <th>Field</th>
                            <th>Accuracy</th>
                            <th>Samples</th>
                            <th>Best Prompt</th>
                        </tr>
        """
        
        # Add field results
        for field_name, field_result in self.experiment_result.field_results.items():
            best_prompt = self._find_best_prompt(field_result)
            best_prompt_name = best_prompt or "N/A"
            
            html += f"""
                        <tr>
                            <td>{field_name}</td>
                            <td>{field_result.average_accuracy*100:.1f}%</td>
                            <td>{field_result.sample_count}</td>
                            <td>{best_prompt_name}</td>
                        </tr>
            """
            
        html += """
                    </table>
                </div>
        """
        
        # Add prompt comparison for each field
        for field_name, field_result in self.experiment_result.field_results.items():
            if len(field_result.prompt_performances) > 1:
                html += f"""
                <div class="card">
                    <h2>Prompt Comparison: {field_name}</h2>
                    <table>
                        <tr>
                            <th>Prompt</th>
                            <th>Accuracy</th>
                            <th>Exact Match</th>
                            <th>Character Error Rate</th>
                            <th>Processing Time</th>
                        </tr>
                """
                
                for prompt_name, prompt_perf in field_result.prompt_performances.items():
                    html += f"""
                        <tr>
                            <td>{prompt_name}</td>
                            <td>{prompt_perf.accuracy*100:.1f}%</td>
                            <td>{prompt_perf.exact_match*100:.1f}%</td>
                            <td>{prompt_perf.character_error_rate:.4f}</td>
                            <td>{prompt_perf.processing_time:.2f}s</td>
                        </tr>
                    """
                    
                html += """
                    </table>
                </div>
                """
        
        # Add similar experiments if requested
        if include_similar:
            similar_exps = self.find_similar_experiments()
            if similar_exps:
                html += """
                <div class="card">
                    <h2>Similar Experiments</h2>
                    <table>
                        <tr>
                            <th>Name</th>
                            <th>Model</th>
                            <th>Date</th>
                            <th>Similarity</th>
                        </tr>
                """
                
                for exp in similar_exps:
                    html += f"""
                        <tr>
                            <td>{exp['experiment_name']}</td>
                            <td>{exp['model_name']}</td>
                            <td>{exp['timestamp'].split('T')[0] if 'T' in exp['timestamp'] else exp['timestamp']}</td>
                            <td>{exp['similarity_score']:.2f}</td>
                        </tr>
                    """
                    
                html += """
                    </table>
                </div>
                """
        
        # Close HTML
        html += """
            </div>
        </body>
        </html>
        """
        
        # Save to file if output path specified
        if output_path:
            out_path = Path(output_path)
        else:
            out_path = self.experiment_dir / "dashboard.html"
            
        with open(out_path, 'w') as f:
            f.write(html)
            
        logger.info(f"Generated experiment dashboard at: {out_path}")
        return str(out_path)

# Create a global instance function for convenience
def get_results_collector(
    base_path: Optional[Union[str, Path]] = None,
    experiment_name: Optional[str] = None,
    create_if_missing: bool = True
) -> 'ResultsCollector':
    """
    Get or create a ResultsCollector instance.
    
    Args:
        base_path: Base directory for storing results
        experiment_name: Name of the current experiment
        create_if_missing: Whether to create the collector if it doesn't exist
        
    Returns:
        ResultsCollector instance
    """
    # Use a module-level singleton
    global _results_collector
    
    if '_results_collector' not in globals() or _results_collector is None:
        if not base_path and not create_if_missing:
            raise ValueError("No ResultsCollector exists and create_if_missing is False")
        
        # Use default values if not specified
        if not base_path:
            import os
            # Try to find a suitable location
            base_path = os.environ.get('RESULTS_DIR', 'results')
        
        _results_collector = ResultsCollector(
            base_path=base_path, 
            experiment_name=experiment_name
        )
    
    return _results_collector


class ExperimentLoader:
    """
    System for managing, loading, and discovering experiment results from multiple sources.
    
    This class provides functionality for:
    - Discovering available experiment results in a results directory
    - Loading experiment results and configurations
    - Filtering experiments based on various criteria
    - Converting experiment metadata to pandas DataFrame for analysis
    """
    
    def __init__(self, base_directory: Optional[Union[str, Path]] = None):
        """
        Initialize the experiment loader.
        
        Args:
            base_directory: Base directory for experiment discovery, defaults to standard results directory
        """
        self.base_directory = Path(base_directory) if base_directory else Path("results")
        self._experiment_cache: Dict[str, Dict[str, Any]] = {}
        
    def discover_experiments(
        self, 
        force_refresh: bool = False,
        include_subdirectories: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Discover available experiment results in the base directory.
        
        Args:
            force_refresh: Force refresh of the experiment cache
            include_subdirectories: Whether to include subdirectories in the search
            
        Returns:
            Dictionary mapping experiment IDs to their metadata
        """
        # Return cached results if available and not forcing refresh
        if self._experiment_cache and not force_refresh:
            return self._experiment_cache
            
        logger.info(f"Discovering experiments in {self.base_directory}")
        
        # Initialize results
        discovered_experiments: Dict[str, Dict[str, Any]] = {}
        
        # Pattern for experiment results file
        results_pattern = "experiment_result.json"
        
        # Handle search pattern based on whether to include subdirectories
        if include_subdirectories:
            search_paths = list(self.base_directory.glob(f"**/{results_pattern}"))
        else:
            search_paths = list(self.base_directory.glob(f"{results_pattern}"))
            
        logger.info(f"Found {len(search_paths)} potential experiment result files")
        
        # Process each result file
        for result_path in search_paths:
            try:
                # Load the experiment result file
                with open(result_path, 'r') as f:
                    data = json.load(f)
                    
                # Check if it's a valid experiment result (has required fields)
                if not self._is_valid_experiment_result(data):
                    logger.debug(f"Skipping invalid experiment result: {result_path}")
                    continue
                    
                # Extract experiment ID
                experiment_id = data.get('experiment_id') or str(result_path.parent.name)
                
                # Extract metadata
                metadata = self._extract_experiment_metadata(data, result_path)
                
                # Add to discovered experiments
                discovered_experiments[experiment_id] = metadata
                
            except Exception as e:
                logger.debug(f"Error processing experiment result {result_path}: {str(e)}")
                continue
                
        logger.info(f"Discovered {len(discovered_experiments)} valid experiments")
        
        # Update cache
        self._experiment_cache = discovered_experiments
        
        return discovered_experiments
        
    def _is_valid_experiment_result(self, data: Dict[str, Any]) -> bool:
        """Check if the data appears to be a valid experiment result"""
        # Check for minimum required fields
        required_fields = ['experiment_name', 'field_results']
        return all(field in data for field in required_fields)
        
    def _extract_experiment_metadata(self, data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """Extract metadata from experiment result"""
        metadata = {
            'experiment_name': data.get('experiment_name', str(file_path.parent.name)),
            'model': data.get('model_name', ''),
            'timestamp': data.get('timestamp', ''),
            'path': str(file_path.parent),
            'fields': list(data.get('field_results', {}).keys()),
        }
        
        # Extract prompts from field results
        prompts = set()
        for field_data in data.get('field_results', {}).values():
            for prompt in field_data.get('prompt_performances', {}).keys():
                prompts.add(prompt)
                
        metadata['prompts'] = list(prompts)
        
        # Check if there's a matching config file
        config_path = file_path.parent / "experiment_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    metadata['experiment_type'] = config_data.get('experiment_type', '')
                    metadata['quantization'] = config_data.get('quantization', {}).get('strategy', '')
            except Exception:
                pass
                
        return metadata
        
    def load_experiment(
        self, 
        experiment_id_or_path: Union[str, Path]
    ) -> Optional[Tuple['ExperimentResult', 'ExperimentConfiguration']]:
        """
        Load a complete experiment by ID or path.
        
        Args:
            experiment_id_or_path: Experiment ID or path to experiment directory
            
        Returns:
            Tuple of (ExperimentResult, ExperimentConfiguration) or None if loading fails
        """
        # First check if this is a path
        path = Path(experiment_id_or_path)
        
        # If it's a directory, look for result file
        if path.is_dir():
            result_path = path / "experiment_result.json"
            config_path = path / "experiment_config.json"
        # If it's an experiment ID, check if it's in the cache
        elif experiment_id_or_path in self._experiment_cache:
            cache_entry = self._experiment_cache[experiment_id_or_path]
            result_path = Path(cache_entry['path']) / "experiment_result.json"
            config_path = Path(cache_entry['path']) / "experiment_config.json"
        # Otherwise, try to discover experiments
        else:
            self.discover_experiments()
            if experiment_id_or_path in self._experiment_cache:
                cache_entry = self._experiment_cache[experiment_id_or_path]
                result_path = Path(cache_entry['path']) / "experiment_result.json"
                config_path = Path(cache_entry['path']) / "experiment_config.json"
            else:
                logger.warning(f"Experiment not found: {experiment_id_or_path}")
                return None
        
        # Load result file
        experiment_result = None
        if result_path.exists():
            try:
                # Create a temporary collector to load the experiment
                collector = ResultsCollector(base_path=result_path.parent.parent, experiment_name=result_path.parent.name)
                experiment_result = collector.load_experiment(result_path.parent)
            except Exception as e:
                logger.error(f"Error loading experiment result: {str(e)}")
                return None
                
        # Load config file
        experiment_config = None
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    # Proper handling would create an ExperimentConfiguration object
                    # This is simplified for now
                    experiment_config = config_data
            except Exception as e:
                logger.debug(f"Error loading experiment config: {str(e)}")
                
        # If we couldn't load the result, return None
        if not experiment_result:
            return None
            
        return (experiment_result, experiment_config)
        
    def filter_experiments(
        self,
        type: Optional[str] = None,
        model: Optional[str] = None,
        fields: Optional[List[str]] = None,
        prompts: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        quantization: Optional[str] = None,
        name_contains: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Filter discovered experiments based on various criteria.
        
        Args:
            type: Filter by experiment type
            model: Filter by model name
            fields: Filter by fields (must contain all specified fields)
            prompts: Filter by prompts (must contain all specified prompts)
            start_date: Filter by start date (format: YYYY-MM-DD)
            end_date: Filter by end date (format: YYYY-MM-DD)
            quantization: Filter by quantization strategy
            name_contains: Filter by experiment name containing this string
            
        Returns:
            Filtered dictionary of experiments
        """
        # Make sure we have discovered experiments
        experiments = self.discover_experiments()
        
        # Apply filters
        filtered_experiments = {}
        
        for exp_id, metadata in experiments.items():
            # Filter by type
            if type and metadata.get('experiment_type', '') != type:
                continue
                
            # Filter by model
            if model and metadata.get('model', '') != model:
                continue
                
            # Filter by fields (must contain all specified fields)
            if fields:
                exp_fields = metadata.get('fields', [])
                if not all(field in exp_fields for field in fields):
                    continue
                    
            # Filter by prompts (must contain all specified prompts)
            if prompts:
                exp_prompts = metadata.get('prompts', [])
                if not all(prompt in exp_prompts for prompt in prompts):
                    continue
                    
            # Filter by date range
            if start_date or end_date:
                exp_date = metadata.get('timestamp', '').split('T')[0]
                if not exp_date:
                    continue
                    
                if start_date and exp_date < start_date:
                    continue
                    
                if end_date and exp_date > end_date:
                    continue
                    
            # Filter by quantization
            if quantization and metadata.get('quantization', '') != quantization:
                continue
                
            # Filter by name
            if name_contains and name_contains.lower() not in metadata.get('experiment_name', '').lower():
                continue
                
            # If passed all filters, add to result
            filtered_experiments[exp_id] = metadata
            
        return filtered_experiments
        
    def get_experiment_dataframe(self) -> pd.DataFrame:
        """
        Convert discovered experiments to a pandas DataFrame for easier analysis.
        
        Returns:
            DataFrame with experiment metadata
        """
        # Make sure we have discovered experiments
        experiments = self.discover_experiments()
        
        # Convert to list of records for pandas
        records = []
        for exp_id, metadata in experiments.items():
            record = {
                'experiment_id': exp_id,
                'experiment_name': metadata.get('experiment_name', ''),
                'model': metadata.get('model', ''),
                'date': metadata.get('timestamp', '').split('T')[0],
                'field_count': len(metadata.get('fields', [])),
                'prompt_count': len(metadata.get('prompts', [])),
                'experiment_type': metadata.get('experiment_type', ''),
                'quantization': metadata.get('quantization', ''),
                'path': metadata.get('path', '')
            }
            records.append(record)
            
        # Create DataFrame
        return pd.DataFrame(records)
        
    def batch_load_experiments(
        self,
        experiment_ids: List[str],
        batch_size: int = 10,
        results_only: bool = True
    ) -> Dict[str, Any]:
        """
        Load multiple experiments in batch mode.
        
        Args:
            experiment_ids: List of experiment IDs to load
            batch_size: Number of experiments to load in each batch
            results_only: Whether to return only experiment results (not configs)
            
        Returns:
            Dictionary mapping experiment IDs to loaded experiments
        """
        loaded_experiments = {}
        
        # Process in batches
        for i in range(0, len(experiment_ids), batch_size):
            batch = experiment_ids[i:i+batch_size]
            
            for exp_id in batch:
                exp_data = self.load_experiment(exp_id)
                if exp_data:
                    if results_only:
                        loaded_experiments[exp_id] = exp_data[0]  # Just the result
                    else:
                        loaded_experiments[exp_id] = exp_data  # (result, config) tuple
                        
        return loaded_experiments
        
    def iterate_experiments(
        self,
        filter_criteria: Optional[Dict[str, Any]] = None,
        batch_size: int = 10,
        filter_func = None
    ) -> Iterator[Tuple[str, 'ExperimentResult']]:
        """
        Iterator for processing large numbers of experiments efficiently.
        
        Args:
            filter_criteria: Dictionary of filter criteria to pass to filter_experiments
            batch_size: Number of experiments to load in each batch
            filter_func: Optional custom filter function that takes (exp_id, metadata) and returns boolean
            
        Yields:
            Tuples of (experiment_id, experiment_result)
        """
        # Discover and filter experiments
        experiments = self.discover_experiments()
        
        # Apply filter criteria if provided
        if filter_criteria:
            experiments = self.filter_experiments(**filter_criteria)
            
        # Apply custom filter if provided
        if filter_func:
            experiments = {
                exp_id: metadata for exp_id, metadata in experiments.items()
                if filter_func(exp_id, metadata)
            }
            
        # Get experiment IDs
        experiment_ids = list(experiments.keys())
        
        # Process in batches
        for i in range(0, len(experiment_ids), batch_size):
            batch = experiment_ids[i:i+batch_size]
            
            # Load batch
            batch_data = self.batch_load_experiments(batch, batch_size=batch_size)
            
            # Yield each result
            for exp_id, result in batch_data.items():
                yield (exp_id, result)


class ExperimentComparator:
    """
    Advanced experiment comparison system.
    
    This class provides functionality for comparing multiple experiments across
    various dimensions such as models, prompts, quantization strategies, and fields.
    It helps researchers identify performance patterns and make data-driven decisions.
    """
    
    def __init__(
        self, 
        experiment_loader: Optional['ExperimentLoader'] = None,
        metrics_calculator = None
    ):
        """
        Initialize experiment comparator.
        
        Args:
            experiment_loader: Optional ExperimentLoader instance to use
            metrics_calculator: Optional metrics calculator to use
        """
        self.experiment_loader = experiment_loader
        self.metrics_calculator = metrics_calculator
        
    def compare_experiments(
        self,
        experiment_ids: List[str],
        field: Optional[str] = None,
        prompt_name: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        name: Optional[str] = None,
        comparison_dimension: str = "experiment"
    ) -> ComparisonResult:
        """
        Compare multiple experiments based on specified parameters.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            field: Optional field to focus comparison on
            prompt_name: Optional prompt to focus comparison on
            metrics: Optional list of metrics to include in comparison
            name: Optional name for the comparison
            comparison_dimension: Primary dimension for comparison (experiment, field, prompt, etc.)
            
        Returns:
            Comparison result between experiments
        """
        # Default metrics if not specified
        if not metrics:
            metrics = ["accuracy", "exact_match", "character_error_rate", "processing_time"]
            
        # Create comparison result
        comparison = ComparisonResult(
            primary_dimension=comparison_dimension,
            metrics=metrics,
            name=name or f"Experiment Comparison ({len(experiment_ids)} experiments)"
        )
        
        # Get experiments data
        experiments_data = {}
        
        # If we have a loader, use it to load experiments
        if self.experiment_loader:
            for exp_id in experiment_ids:
                exp_data = self.experiment_loader.load_experiment(exp_id)
                if exp_data:
                    exp_result, _ = exp_data
                    experiments_data[exp_id] = exp_result
        # Otherwise, assume experiment_ids are paths and try to load directly
        else:
            for exp_id in experiment_ids:
                # Create a temporary collector to load the experiment
                collector = ResultsCollector(base_path=Path(exp_id).parent, experiment_name=Path(exp_id).name)
                exp_result = collector.load_experiment(exp_id)
                if exp_result:
                    experiments_data[exp_id] = exp_result
        
        # Compare based on field and prompt if specified
        for exp_id, exp_result in experiments_data.items():
            # Get a suitable name for the experiment
            exp_name = exp_result.experiment_name or exp_id
            
            # If field is specified, only compare that field
            if field and field in exp_result.field_results:
                field_result = exp_result.field_results[field]
                
                # If prompt is specified, only compare that prompt
                if prompt_name and prompt_name in field_result.prompt_performances:
                    prompt_perf = field_result.prompt_performances[prompt_name]
                    comparison.add_data_point(
                        dimension_value=exp_name,
                        metrics={
                            "accuracy": prompt_perf.accuracy,
                            "exact_match": prompt_perf.exact_match,
                            "character_error_rate": prompt_perf.character_error_rate,
                            "processing_time": prompt_perf.processing_time,
                            "sample_count": prompt_perf.sample_count
                        },
                        sample_size=prompt_perf.sample_count
                    )
                # Otherwise use field level metrics
                else:
                    # Find best prompt for other metrics
                    best_prompt = None
                    if field_result.prompt_performances:
                        best_prompt = max(
                            field_result.prompt_performances.items(),
                            key=lambda x: x[1].accuracy or 0
                        )[0]
                    
                    metrics_data = {
                        "accuracy": field_result.average_accuracy,
                        "sample_count": field_result.sample_count
                    }
                    
                    # Add best prompt metrics
                    if best_prompt:
                        prompt_perf = field_result.prompt_performances[best_prompt]
                        metrics_data.update({
                            "exact_match": prompt_perf.exact_match,
                            "character_error_rate": prompt_perf.character_error_rate,
                            "processing_time": prompt_perf.processing_time
                        })
                        
                    comparison.add_data_point(
                        dimension_value=exp_name,
                        metrics=metrics_data,
                        sample_size=field_result.sample_count
                    )
            # Otherwise use experiment level metrics
            else:
                comparison.add_data_point(
                    dimension_value=exp_name,
                    metrics={
                        "accuracy": exp_result.average_accuracy,
                        "sample_count": sum(fr.sample_count for fr in exp_result.field_results.values())
                    },
                    sample_size=sum(fr.sample_count for fr in exp_result.field_results.values())
                )
        
        # Calculate statistics and generate visualization suggestions
        comparison.calculate_statistics()
                
        return comparison
    
    def compare_models(
        self,
        model_names: List[str],
        field: Optional[str] = None,
        prompt_name: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        use_most_recent: bool = True
    ) -> ComparisonResult:
        """
        Compare performance across different models.
        
        Args:
            model_names: List of model names to compare
            field: Optional field to focus comparison on
            prompt_name: Optional prompt to focus comparison on
            metrics: Optional list of metrics to include
            use_most_recent: Whether to use the most recent experiment for each model
            
        Returns:
            Comparison result between models
        """
        if not self.experiment_loader:
            raise ValueError("ExperimentLoader is required for model comparison")
            
        # Find experiments for each model
        experiments_by_model = {}
        
        for model_name in model_names:
            # Get experiments for this model
            model_experiments = self.experiment_loader.filter_experiments(model=model_name)
            
            if not model_experiments:
                logger.warning(f"No experiments found for model: {model_name}")
                continue
                
            # If using most recent, sort by timestamp and get the latest
            if use_most_recent and len(model_experiments) > 1:
                # Sort by timestamp (descending)
                sorted_exps = sorted(
                    model_experiments.items(),
                    key=lambda x: x[1].get('timestamp', ''),
                    reverse=True
                )
                # Get the most recent experiment
                experiments_by_model[model_name] = sorted_exps[0][0]
            # Otherwise use all experiments for this model
            else:
                experiments_by_model[model_name] = list(model_experiments.keys())
        
        # Flatten the experiment IDs
        experiment_ids = []
        for model, exps in experiments_by_model.items():
            if isinstance(exps, list):
                experiment_ids.extend(exps)
            else:
                experiment_ids.append(exps)
        
        # Use the standard comparison method
        return self.compare_experiments(
            experiment_ids=experiment_ids,
            field=field,
            prompt_name=prompt_name,
            metrics=metrics,
            name=f"Model Comparison ({', '.join(model_names)})",
            comparison_dimension="model"
        )
    
    def compare_quantization_strategies(
        self,
        model_name: str,
        strategies: List[str],
        field: Optional[str] = None,
        prompt_name: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        use_most_recent: bool = True
    ) -> ComparisonResult:
        """
        Compare performance across different quantization strategies for a model.
        
        Args:
            model_name: Base model name
            strategies: List of quantization strategies to compare
            field: Optional field to focus comparison on
            prompt_name: Optional prompt to focus comparison on
            metrics: Optional list of metrics to include
            use_most_recent: Whether to use the most recent experiment for each strategy
            
        Returns:
            Comparison result between quantization strategies
        """
        if not self.experiment_loader:
            raise ValueError("ExperimentLoader is required for quantization comparison")
            
        # Find experiments for each quantization strategy
        experiments_by_strategy = {}
        
        for strategy in strategies:
            # Get experiments for this model with this quantization strategy
            strategy_experiments = self.experiment_loader.filter_experiments(
                model=model_name,
                quantization=strategy
            )
            
            if not strategy_experiments:
                logger.warning(f"No experiments found for model {model_name} with quantization: {strategy}")
                continue
                
            # If using most recent, sort by timestamp and get the latest
            if use_most_recent and len(strategy_experiments) > 1:
                # Sort by timestamp (descending)
                sorted_exps = sorted(
                    strategy_experiments.items(),
                    key=lambda x: x[1].get('timestamp', ''),
                    reverse=True
                )
                # Get the most recent experiment
                experiments_by_strategy[strategy] = sorted_exps[0][0]
            # Otherwise use all experiments for this strategy
            else:
                experiments_by_strategy[strategy] = list(strategy_experiments.keys())
        
        # Flatten the experiment IDs
        experiment_ids = []
        for strategy, exps in experiments_by_strategy.items():
            if isinstance(exps, list):
                experiment_ids.extend(exps)
            else:
                experiment_ids.append(exps)
        
        # Use the standard comparison method
        return self.compare_experiments(
            experiment_ids=experiment_ids,
            field=field,
            prompt_name=prompt_name,
            metrics=metrics,
            name=f"Quantization Comparison for {model_name}",
            comparison_dimension="quantization"
        )
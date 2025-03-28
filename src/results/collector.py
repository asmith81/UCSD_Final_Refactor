"""
Enhanced Results Collection and Management System

Provides comprehensive tracking, storage, and analysis of extraction experiment results.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd

# Import schema and metrics components
from src.results.schema import (
    ExperimentResult, 
    PromptPerformance, 
    IndividualExtractionResult,
    ExtractionStatus
)
from src.analysis.metrics import create_metrics_calculator

# Configure logger
logger = logging.getLogger(__name__)


class EnhancedResultsCollector:
    """
    Advanced results collection system with comprehensive tracking 
    and analysis capabilities.
    """
    
    def __init__(
        self, 
        base_path: Union[str, Path], 
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the results collector.
        
        Args:
            base_path: Base directory for storing results
            experiment_name: Name of the current experiment
        """
        self.base_path = Path(base_path)
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory structure
        self.experiment_dir = self.base_path / self.experiment_name
        self._create_directory_structure()
        
        # Initialize experiment result tracker
        self.experiment_result = ExperimentResult(experiment_name=self.experiment_name)
        
        # Initialize metadata
        self.metadata = {
            "experiment_name": self.experiment_name,
            "started_at": datetime.now().isoformat(),
            "fields": [],
            "models": [],
            "prompts": []
        }
    
    def _create_directory_structure(self):
        """
        Create directories for storing experiment results.
        """
        # Create main experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "raw_results",
            "processed_results", 
            "metrics", 
            "visualizations", 
            "checkpoints"
        ]
        
        for subdir in subdirs:
            (self.experiment_dir / subdir).mkdir(exist_ok=True)
    
    def add_field_results(
        self, 
        field: str, 
        prompt_name: str, 
        results: List[IndividualExtractionResult]
    ):
        """
        Add results for a specific field and prompt.
        
        Args:
            field: Field type being extracted
            prompt_name: Name of the prompt used
            results: List of individual extraction results
        """
        # Create or get performance tracker for this field and prompt
        performance = PromptPerformance(
            prompt_name=prompt_name,
            field=field,
            results=results
        )
        
        # Calculate metrics
        performance.calculate_metrics()
        
        # Add to experiment result
        self.experiment_result.add_field_results(field, performance)
        
        # Update metadata
        if field not in self.metadata["fields"]:
            self.metadata["fields"].append(field)
        if prompt_name not in self.metadata["prompts"]:
            self.metadata["prompts"].append(prompt_name)
    
    def save_raw_results(
        self, 
        field: str, 
        prompt_name: str, 
        results: List[Dict[str, Any]]
    ):
        """
        Save raw extraction results to a JSON file.
        
        Args:
            field: Field type being extracted
            prompt_name: Name of the prompt used
            results: Raw results dictionary
        """
        filename = f"{field}_{prompt_name}_raw_results.json"
        filepath = self.experiment_dir / "raw_results" / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved raw results to {filepath}")
    
    def calculate_cross_field_metrics(self):
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
                "best_prompt": max(performances, key=lambda p: p.accuracy).prompt_name,
                "avg_accuracy": np.mean([p.accuracy for p in performances]),
                "avg_character_error_rate": np.mean([p.avg_character_error_rate for p in performances])
            }
            cross_field_metrics["field_performance"][field] = field_metrics
        
        return cross_field_metrics
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """
        Save calculated metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
        """
        metrics_filename = "experiment_metrics.json"
        filepath = self.experiment_dir / "metrics" / metrics_filename
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved experiment metrics to {filepath}")
    
    def save_experiment_result(self):
        """
        Save the complete experiment result to a JSON file.
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
        self.experiment_result.save_to_file(result_filepath)
        
        logger.info(f"Saved experiment result to {result_filepath}")
        logger.info(f"Saved experiment metadata to {metadata_filepath}")
    
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
                    all_results.append(result_dict)
        
        return pd.DataFrame(all_results)
    
    def export_results(self, format: str = 'csv'):
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

    def generate_comparative_report(self):
        """
        Generate a comprehensive comparative report.
        
        Returns:
            Dictionary with comparative analysis
        """
        report = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "cross_field_metrics": self.calculate_cross_field_metrics(),
            "field_details": {}
        }
        
        # Detailed analysis for each field
        for field, performances in self.experiment_result.field_results.items():
            # Create metrics calculator for the field
            metrics_calculator = create_metrics_calculator(field, {})
            
            field_report = {
                "prompts": [],
                "best_prompt": None,
                "worst_prompt": None
            }
            
            # Analyze each prompt's performance
            prompt_metrics = []
            for perf in performances:
                prompt_metric = {
                    "prompt_name": perf.prompt_name,
                    "accuracy": perf.accuracy,
                    "character_error_rate": perf.avg_character_error_rate,
                    "processing_time": perf.avg_processing_time,
                    "total_items": perf.total_items
                }
                prompt_metrics.append(prompt_metric)
            
            # Sort prompts by performance
            prompt_metrics.sort(key=lambda x: x["accuracy"], reverse=True)
            
            field_report["prompts"] = prompt_metrics
            field_report["best_prompt"] = prompt_metrics[0] if prompt_metrics else None
            field_report["worst_prompt"] = prompt_metrics[-1] if prompt_metrics else None
            
            report["field_details"][field] = field_report
        
        return report
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logger
logger = logging.getLogger(__name__)

class ResultsCollector:
    """
    Handles storing and organizing results from extraction experiments.
    
    This class manages:
    - Field-specific result storage
    - Metrics collection
    - Comparative analysis data
    - Results organization by experiment
    """
    
    def __init__(self, base_path: Union[str, Path], experiment_name: str = None):
        """
        Initialize the results collector.
        
        Args:
            base_path: Base directory for storing results
            experiment_name: Name of the experiment (default: timestamped name)
        """
        self.base_path = Path(base_path)
        self.experiment_name = experiment_name or f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set up paths
        self.experiment_path = self.base_path / self.experiment_name
        self.create_directory_structure()
        
        logger.info(f"Initialized ResultsCollector for experiment '{self.experiment_name}'")
        logger.info(f"Results will be stored in {self.experiment_path}")
    
    def create_directory_structure(self) -> None:
        """Create the directory structure for storing results."""
        # Create main experiment directory
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        for subdir in ["raw", "processed", "visualizations", "checkpoints"]:
            (self.experiment_path / subdir).mkdir(exist_ok=True)
        
        # Create field-specific directories
        for field in ["work_order", "cost", "combined"]:
            field_dir = self.experiment_path / field
            field_dir.mkdir(exist_ok=True)
            
            # Create field-specific subdirectories
            for subdir in ["raw", "processed", "visualizations"]:
                (field_dir / subdir).mkdir(exist_ok=True)
    
    def get_field_path(self, field: str, file_type: str = "raw") -> Path:
        """
        Get path for field-specific storage.
        
        Args:
            field: Field type (work_order, cost, etc.)
            file_type: Type of file (raw, processed, visualization)
            
        Returns:
            Path object for the field directory
        """
        return self.experiment_path / field / file_type
    
    def save_field_results(self, field: str, results: List[Dict[str, Any]]) -> None:
        """
        Save extraction results for a specific field.
        
        Args:
            field: Field type (work_order, cost, etc.)
            results: List of extraction results
        """
        # Validate input
        if not results:
            logger.warning(f"No results to save for {field}")
            return
        
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
    
    def save_field_metrics(self, field: str, metrics: Dict[str, Any]) -> None:
        """
        Save metrics for a specific field.
        
        Args:
            field: Field type (work_order, cost, etc.)
            metrics: Dictionary of metrics by prompt category
        """
        # Validate input
        if not metrics:
            logger.warning(f"No metrics to save for {field}")
            return
        
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
    
    def save_comparative_metrics(self, field: str, comparison: Dict[str, Any]) -> None:
        """
        Save comparative metrics for prompt categories within a field.
        
        Args:
            field: Field type (work_order, cost, etc.)
            comparison: Dictionary of comparative metrics
        """
        # Validate input
        if not comparison:
            logger.warning(f"No comparison metrics to save for {field}")
            return
        
        # Add timestamp if not present
        if "timestamp" not in comparison:
            comparison["timestamp"] = datetime.now().isoformat()
        
        # Save comparison metrics
        comparison_path = self.get_field_path(field, "processed") / "prompt_comparison.json"
        try:
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"Saved {field} prompt comparison to {comparison_path}")
        except Exception as e:
            logger.error(f"Error saving {field} prompt comparison: {e}")
    
    def save_cross_field_metrics(self, cross_field_metrics: Dict[str, Any]) -> None:
        """
        Save metrics comparing performance across different fields.
        
        Args:
            cross_field_metrics: Dictionary of cross-field metrics
        """
        # Validate input
        if not cross_field_metrics:
            logger.warning("No cross-field metrics to save")
            return
        
        # Add timestamp if not present
        if "timestamp" not in cross_field_metrics:
            cross_field_metrics["timestamp"] = datetime.now().isoformat()
        
        # Save cross-field metrics
        metrics_path = self.experiment_path / "combined" / "processed" / "cross_field_metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(cross_field_metrics, f, indent=2)
            logger.info(f"Saved cross-field metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving cross-field metrics: {e}")
    
    def save_run_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Save experiment run metadata.
        
        Args:
            metadata: Dictionary of experiment metadata
        """
        # Validate input
        if not metadata:
            logger.warning("No metadata to save")
            return
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        # Save metadata
        metadata_path = self.experiment_path / "run_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved run metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving run metadata: {e}")
    
    def save_visualization(self, field: str, visualization_name: str, file_path: Union[str, Path]) -> None:
        """
        Copy or save a visualization file to the appropriate location.
        
        Args:
            field: Field type or 'combined' for cross-field visualizations
            visualization_name: Name of the visualization file
            file_path: Path to the visualization file
        """
        import shutil
        
        source_path = Path(file_path)
        
        # Verify source file exists
        if not source_path.exists():
            logger.error(f"Visualization file not found: {source_path}")
            return
        
        # Get destination path
        if field == "combined":
            dest_dir = self.experiment_path / "combined" / "visualizations"
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
            for field_dir in self.experiment_path.glob("*"):
                if not field_dir.is_dir() or field_dir.name in ["raw", "processed", "visualizations", "checkpoints"]:
                    continue
                
                field_name = field_dir.name
                for file_type in result_files.keys():
                    path = self.get_field_path(field_name, file_type)
                    if path.exists():
                        for f in path.glob("*.json"):
                            result_files[file_type].append(f"{field_name}/{f.name}")
        
        return result_files
    
    def load_results(self, field: str, result_type: str, filename: str) -> Optional[Dict[str, Any]]:
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
    
    def cleanup_temporary_files(self) -> None:
        """Clean up temporary files created during the experiment."""
        # This would clean up any temporary files created during the experiment
        # For now, simply log the action
        logger.info("Cleaning up temporary files")


"""
Result collection and storage utilities for multi-field extraction experiments.

This module provides:
- Structured result storage
- Metrics aggregation
- Experiment metadata management
- Checkpoint management
"""

import os
import json
import logging
from pathlib import Path
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


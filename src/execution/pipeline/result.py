"""
Pipeline Execution Result

Provides a comprehensive result object for pipeline execution,
capturing detailed information about the entire extraction process.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.config.experiment import ExperimentConfiguration
from src.config.paths import get_path_config


@dataclass
class PipelineExecutionResult:
    """
    Comprehensive result object capturing pipeline execution details.
    
    Tracks:
    - Experiment configuration
    - Stage-by-stage results
    - Overall pipeline metrics
    - Error information
    - Execution metadata
    """
    
    # Experiment configuration
    config: ExperimentConfiguration
    
    # Stage results
    stage_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Overall pipeline metrics
    overall_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution metadata
    timestamp: datetime = field(default_factory=datetime.now)
    total_execution_time: Optional[float] = None
    status: str = "pending"  # pending, success, partial_success, failed
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_stage_result(self, stage_name: str, stage_results: Dict[str, Any]) -> None:
        """
        Add results for a specific pipeline stage.
        
        Args:
            stage_name: Name of the stage
            stage_results: Results dictionary for the stage
        """
        self.stage_results[stage_name] = stage_results
    
    def add_error(self, error: Dict[str, Any]) -> None:
        """
        Add an error to the error list.
        
        Args:
            error: Error information dictionary
        """
        self.errors.append(error)
        
        # Update status if critical errors occur
        if self.status != "failed":
            self.status = "partial_success"
    
    def set_status(self, status: str) -> None:
        """
        Set the overall pipeline execution status.
        
        Args:
            status: Status of the pipeline execution
        """
        self.status = status
    
    def calculate_metrics(self) -> None:
        """
        Calculate overall metrics based on stage results.
        """
        # Aggregate metrics from different stages
        metrics = {}
        
        # Example: Aggregate extraction metrics
        if 'extraction' in self.stage_results:
            extraction_results = self.stage_results.get('extraction', {})
            
            # Calculate cross-field metrics
            field_metrics = {}
            for field, field_data in extraction_results.items():
                field_success_rates = []
                
                for prompt_name, prompt_results in field_data.get('prompt_results', {}).items():
                    # Calculate success rate for each prompt
                    total_results = len(prompt_results)
                    successful_results = sum(
                        1 for result in prompt_results 
                        if result.get('exact_match', False)
                    )
                    success_rate = successful_results / total_results if total_results > 0 else 0
                    
                    field_metrics[f"{field}_{prompt_name}_success_rate"] = success_rate
                    field_success_rates.append(success_rate)
                
                # Average success rate for the field
                if field_success_rates:
                    metrics[f"{field}_avg_success_rate"] = sum(field_success_rates) / len(field_success_rates)
        
        self.overall_metrics = metrics
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save the pipeline execution result to a file.
        
        Args:
            filename: Optional filename (defaults to timestamp-based name)
        
        Returns:
            Path to the saved result file
        """
        # Get paths configuration
        paths = get_path_config()
        
        # Generate filename if not provided
        if not filename:
            timestamp = self.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_result_{timestamp}.json"
        
        # Ensure results directory exists
        os.makedirs(paths.get('results_dir', 'results'), exist_ok=True)
        
        # Full path for saving
        full_path = os.path.join(paths.get('results_dir', 'results'), filename)
        
        # Convert result to dictionary
        result_dict = {
            "config": self.config.to_dict(),
            "stage_results": self.stage_results,
            "overall_metrics": self.overall_metrics,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
            "total_execution_time": self.total_execution_time,
            "status": self.status,
            "metadata": self.metadata
        }
        
        # Save to file
        with open(full_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        return full_path
    
    @classmethod
    def load(cls, filepath: str) -> 'PipelineExecutionResult':
        """
        Load a pipeline execution result from a file.
        
        Args:
            filepath: Path to the result file
        
        Returns:
            PipelineExecutionResult instance
        """
        # Read the JSON file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct the result
        config = ExperimentConfiguration.from_dict(data.get('config', {}))
        
        # Create result instance
        result = cls(config=config)
        
        # Populate other attributes
        result.stage_results = data.get('stage_results', {})
        result.overall_metrics = data.get('overall_metrics', {})
        result.errors = data.get('errors', [])
        result.timestamp = datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
        result.total_execution_time = data.get('total_execution_time')
        result.status = data.get('status', 'unknown')
        result.metadata = data.get('metadata', {})
        
        return result
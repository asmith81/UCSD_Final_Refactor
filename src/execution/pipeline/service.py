"""
Pipeline Execution Service

Provides the core orchestration mechanism for running 
extraction experiments through a comprehensive, 
stage-based pipeline.
"""

import time
import logging
from typing import Dict, Any, Optional, List

# Import pipeline components
from .base import BasePipelineStage
from .stages import (
    DataPreparationStage,
    ModelLoadingStage,
    PromptManagementStage,
    ExtractionStage,
    ResultsCollectionStage,
    AnalysisStage,
    VisualizationStage
)
from .result import PipelineExecutionResult
from .error_handling import ErrorHandler, ProgressTracker

# Import project configurations
from src.config.experiment import ExperimentConfiguration

# Configure logging
logger = logging.getLogger(__name__)


class ExtractionPipelineService:
    """
    Primary service for orchestrating the entire extraction pipeline.
    
    Manages:
    - Stage execution
    - Error handling
    - Progress tracking
    - Result generation
    """
    
    def __init__(
        self, 
        config: ExperimentConfiguration,
        stages: Optional[List[BasePipelineStage]] = None
    ):
        """
        Initialize the pipeline service.
        
        Args:
            config: Experiment configuration
            stages: Optional custom stages (defaults to standard pipeline)
        """
        self.config = config
        
        # Use provided stages or default pipeline stages
        self.stages = stages or [
            DataPreparationStage(),
            ModelLoadingStage(),
            PromptManagementStage(),
            ExtractionStage(),
            ResultsCollectionStage(),
            AnalysisStage(),
            VisualizationStage()
        ]
        
        # Initialize error handling and progress tracking
        self.error_handler = ErrorHandler(config)
        self.progress_tracker = ProgressTracker(
            config, 
            stages=[stage.__class__.__name__ for stage in self.stages]
        )
    
    def run(self) -> PipelineExecutionResult:
        """
        Execute the entire extraction pipeline.
        
        Returns:
            PipelineExecutionResult with comprehensive experiment details
        """
        # Create result object
        result = PipelineExecutionResult(config=self.config)
        
        # Track overall start time
        start_time = time.time()
        
        try:
            # Track overall pipeline progress
            self.progress_tracker.start_stage(self.stages[0].__class__.__name__)
            
            # Store results from previous stages
            previous_results = {}
            
            # Execute each stage sequentially
            for stage in self.stages:
                logger.info(f"Starting stage: {stage.__class__.__name__}")
                
                try:
                    # Start progress tracking for this stage
                    self.progress_tracker.start_stage(stage.__class__.__name__)
                    
                    # Execute stage
                    stage_results = stage.execute(
                        config=self.config, 
                        previous_results=previous_results
                    )
                    
                    # Add stage results to overall results
                    result.add_stage_result(
                        stage.__class__.__name__, 
                        stage_results
                    )
                    
                    # Update previous results for next stage
                    previous_results[stage.__class__.__name__.lower()] = stage_results
                    
                    # Mark stage as complete
                    self.progress_tracker.complete_stage(stage.__class__.__name__)
                
                except Exception as e:
                    # Handle stage-specific errors
                    error_entry = self.error_handler.add_error(
                        error=e, 
                        stage=stage.__class__.__name__,
                        context={
                            'previous_results': previous_results
                        }
                    )
                    
                    # Add error to result
                    result.add_error(error_entry.to_dict())
                    
                    # Update pipeline status
                    result.set_status('failed')
                    
                    # Optionally stop pipeline on first error
                    raise
            
            # Calculate overall metrics
            result.calculate_metrics()
            
            # Set final status
            result.set_status('success')
        
        except Exception as e:
            # Catch any unhandled errors
            logger.error(f"Pipeline execution failed: {e}")
            result.set_status('failed')
        
        finally:
            # Calculate total execution time
            result.total_execution_time = time.time() - start_time
            
            # Save error log if any errors occurred
            if result.errors:
                self.error_handler.save_error_log()
            
            # Save progress report
            self.progress_tracker.save_progress_report()
        
        return result
    
    def dry_run(self) -> Dict[str, Any]:
        """
        Perform a dry run of the pipeline without full execution.
        
        Validates configuration and provides a preview of stages.
        
        Returns:
            Dictionary with dry run information
        """
        dry_run_info = {
            "experiment_config": self.config.to_dict(),
            "stages": [stage.__class__.__name__ for stage in self.stages],
            "validation_errors": self.config.validate()
        }
        
        return dry_run_info


def run_extraction_pipeline(
    config: Optional[ExperimentConfiguration] = None,
    config_path: Optional[str] = None,
    **kwargs
) -> PipelineExecutionResult:
    """
    Convenience function to run an extraction pipeline.
    
    Args:
        config: Optional ExperimentConfiguration
        config_path: Optional path to configuration file
        **kwargs: Additional configuration parameters
    
    Returns:
        PipelineExecutionResult from the extraction experiment
    """
    # Load or create configuration
    if config_path:
        experiment_config = ExperimentConfiguration.load(config_path)
    elif config:
        experiment_config = config
    else:
        # Create default configuration
        experiment_config = ExperimentConfiguration(**kwargs)
    
    # Create pipeline service
    pipeline_service = ExtractionPipelineService(experiment_config)
    
    # Run the pipeline
    return pipeline_service.run()
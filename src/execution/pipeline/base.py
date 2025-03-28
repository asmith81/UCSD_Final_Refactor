"""
Base classes and abstract interfaces for the extraction pipeline stages.

This module defines the core abstractions for a modular, stage-based 
extraction pipeline with clear interfaces and error handling.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Import project configuration
from src.config.experiment import ExperimentConfiguration

# Configure logging
logger = logging.getLogger(__name__)


class PipelineStageError(Exception):
    """
    Base exception for pipeline stage-related errors.
    
    Provides a standardized way to handle and communicate 
    errors that occur during pipeline stage execution.
    """
    def __init__(
        self, 
        message: str, 
        stage_name: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize a pipeline stage error.
        
        Args:
            message: Descriptive error message
            stage_name: Name of the stage where error occurred
            original_error: Original exception that triggered this error
        """
        self.stage_name = stage_name
        self.original_error = original_error
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to a dictionary for logging and reporting.
        
        Returns:
            Dictionary representation of the error
        """
        return {
            "message": str(self),
            "stage": self.stage_name,
            "original_error": str(self.original_error) if self.original_error else None,
            "timestamp": datetime.now().isoformat()
        }


@dataclass
class StageMetadata:
    """
    Metadata for tracking stage execution details.
    
    Provides a standardized way to capture information 
    about each pipeline stage's execution.
    """
    stage_name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration: float = 0.0
    status: str = 'pending'  # pending, success, failed
    error: Optional[Dict[str, Any]] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    
    def mark_started(self) -> None:
        """Mark the stage as started."""
        self.start_time = datetime.now()
        self.status = 'running'
    
    def mark_completed(self) -> None:
        """Mark the stage as completed successfully."""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = 'success'
    
    def mark_failed(self, error: Optional[Exception] = None) -> None:
        """
        Mark the stage as failed.
        
        Args:
            error: Optional exception that caused the failure
        """
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status = 'failed'
        
        if error:
            self.error = {
                'type': type(error).__name__,
                'message': str(error),
                'details': str(error)
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to a dictionary for logging and reporting.
        
        Returns:
            Dictionary representation of stage metadata
        """
        return {
            "stage_name": self.stage_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "status": self.status,
            "error": self.error,
            "input_size": self.input_size,
            "output_size": self.output_size
        }


class BasePipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.
    
    Provides a consistent interface for stage execution,
    with built-in validation, logging, and error handling.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the pipeline stage.
        
        Args:
            name: Optional name for the stage (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.metadata = StageMetadata(stage_name=self.name)
        self.logger = logging.getLogger(f"pipeline.{self.name}")
    
    @abstractmethod
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate input configuration and previous stage results.
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
        
        Raises:
            PipelineStageError if input is invalid
        """
        pass
    
    @abstractmethod
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Core processing logic for the stage.
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
        
        Returns:
            Dictionary of stage-specific results
        """
        pass
    
    def execute(
        self, 
        config: ExperimentConfiguration,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the pipeline stage with comprehensive error handling.
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
        
        Returns:
            Dictionary of stage results
        
        Raises:
            PipelineStageError if stage execution fails
        """
        try:
            # Mark stage as started
            self.metadata.mark_started()
            self.metadata.input_size = len(previous_results)
            
            # Log stage start
            self.logger.info(f"Starting stage: {self.name}")
            
            # Validate inputs
            self._validate_input(config, previous_results)
            
            # Process stage
            results = self._process(config, previous_results)
            
            # Update metadata
            self.metadata.output_size = len(results)
            self.metadata.mark_completed()
            
            # Log stage completion
            self.logger.info(
                f"Completed stage: {self.name}. "
                f"Input size: {self.metadata.input_size}, "
                f"Output size: {self.metadata.output_size}, "
                f"Duration: {self.metadata.duration:.2f}s"
            )
            
            return results
        
        except Exception as e:
            # Log and handle any errors
            self.logger.error(
                f"Stage {self.name} failed: {e}", 
                exc_info=True
            )
            
            # Mark stage as failed
            self.metadata.mark_failed(e)
            
            # Raise as PipelineStageError for consistent error handling
            raise PipelineStageError(
                message=f"Error in stage {self.name}: {str(e)}",
                stage_name=self.name,
                original_error=e
            )


class PipelineConfiguration:
    """
    Configuration management for the entire pipeline.
    
    Provides a centralized way to manage pipeline-wide 
    configuration and settings.
    """
    
    def __init__(
        self, 
        experiment_config: ExperimentConfiguration,
        additional_settings: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pipeline configuration.
        
        Args:
            experiment_config: Base experiment configuration
            additional_settings: Optional additional pipeline settings
        """
        self.experiment_config = experiment_config
        self.additional_settings = additional_settings or {}
        
        # Validate and process configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """
        Validate the overall pipeline configuration.
        
        Performs basic sanity checks and sets up any 
        derived configuration parameters.
        
        Raises:
            ValueError if configuration is invalid
        """
        # Validate experiment configuration
        errors = self.experiment_config.validate()
        if errors:
            raise ValueError(f"Invalid experiment configuration: {errors}")
        
        # Additional validation logic can be added here
    
    def get(
        self, 
        key: str, 
        default: Optional[Any] = None
    ) -> Any:
        """
        Retrieve a configuration value.
        
        Checks experiment config first, then additional settings.
        
        Args:
            key: Configuration key to retrieve
            default: Default value if key is not found
        
        Returns:
            Configuration value or default
        """
        # Check experiment config first
        try:
            return self.experiment_config.get(key, default)
        except AttributeError:
            pass
        
        # Then check additional settings
        return self.additional_settings.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Comprehensive dictionary of configuration settings
        """
        config_dict = self.experiment_config.to_dict()
        config_dict.update(self.additional_settings)
        return config_dict
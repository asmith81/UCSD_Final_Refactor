"""
Comprehensive error handling and progress tracking for extraction pipeline.

This module provides robust mechanisms for:
- Tracking pipeline execution progress
- Managing and recovering from errors
- Generating detailed error reports
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Import project configuration and path utilities
from src.config.paths import get_path_config
from src.config.experiment import ExperimentConfiguration

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ErrorEntry:
    """
    Detailed error entry for tracking specific errors in the pipeline.
    
    Provides comprehensive information about each error occurrence.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    stage: Optional[str] = None
    error_type: Optional[str] = None
    error_message: str = ""
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error entry to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the error
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "stage": self.stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "context": self.context
        }


class ErrorHandler:
    """
    Central error management system for the extraction pipeline.
    
    Provides comprehensive error tracking, logging, 
    and potential recovery mechanisms.
    """
    
    def __init__(
        self, 
        config: Optional[ExperimentConfiguration] = None,
        max_error_entries: int = 100
    ):
        """
        Initialize the error handler.
        
        Args:
            config: Optional experiment configuration
            max_error_entries: Maximum number of errors to track
        """
        self.config = config
        self.max_error_entries = max_error_entries
        self.errors: List[ErrorEntry] = []
        
        # Get paths for error logging
        self.paths = get_path_config()
        
        # Setup error logging directory
        self.error_log_dir = self.paths.get_path('experiment_dir')
        if self.error_log_dir:
            os.makedirs(self.error_log_dir, exist_ok=True)
    
    def add_error(
        self, 
        error: Exception, 
        stage: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorEntry:
        """
        Add a new error to the error log.
        
        Args:
            error: The exception that occurred
            stage: Optional stage where the error occurred
            context: Optional additional context about the error
        
        Returns:
            Created ErrorEntry
        """
        # Limit number of tracked errors
        if len(self.errors) >= self.max_error_entries:
            self.errors.pop(0)
        
        # Import traceback for detailed error information
        import traceback
        
        # Create error entry
        error_entry = ErrorEntry(
            stage=stage,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            context=context or {}
        )
        
        # Add to error list
        self.errors.append(error_entry)
        
        # Log the error
        logger.error(
            f"Error in stage {stage}: {error_entry.error_message}",
            exc_info=True
        )
        
        return error_entry
    
    def save_error_log(self, filename: Optional[str] = None) -> str:
        """
        Save the current error log to a file.
        
        Args:
            filename: Optional custom filename
        
        Returns:
            Path to the saved error log
        """
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"error_log_{timestamp}.json"
        
        # Construct full path
        error_log_path = os.path.join(self.error_log_dir, filename)
        
        # Convert errors to dictionaries
        error_log_data = {
            "experiment_name": self.config.name if self.config else "unknown",
            "timestamp": datetime.now().isoformat(),
            "total_errors": len(self.errors),
            "errors": [error.to_dict() for error in self.errors]
        }
        
        # Write to file
        with open(error_log_path, 'w') as f:
            json.dump(error_log_data, f, indent=2)
        
        logger.info(f"Error log saved to {error_log_path}")
        return error_log_path
    
    def clear_errors(self) -> None:
        """
        Clear all tracked errors.
        """
        self.errors.clear()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of tracked errors.
        
        Returns:
            Dictionary with error summary information
        """
        # Group errors by stage and type
        error_summary = {
            "total_errors": len(self.errors),
            "errors_by_stage": {},
            "errors_by_type": {}
        }
        
        for error in self.errors:
            # Count by stage
            if error.stage:
                error_summary["errors_by_stage"][error.stage] = \
                    error_summary["errors_by_stage"].get(error.stage, 0) + 1
            
            # Count by error type
            if error.error_type:
                error_summary["errors_by_type"][error.error_type] = \
                    error_summary["errors_by_type"].get(error.error_type, 0) + 1
        
        return error_summary


class ProgressTracker:
    """
    Tracks and reports pipeline execution progress.
    
    Provides comprehensive progress monitoring 
    with detailed stage tracking and time management.
    """
    
    def __init__(
        self, 
        config: Optional[ExperimentConfiguration] = None,
        stages: Optional[List[str]] = None
    ):
        """
        Initialize progress tracker.
        
        Args:
            config: Optional experiment configuration
            stages: Optional list of stages to track
        """
        self.config = config
        self.stages = stages or []
        
        # Progress tracking
        self.current_stage_index = -1
        self.stage_progress: Dict[str, float] = {}
        
        # Timing
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.stage_start_times: Dict[str, datetime] = {}
        self.stage_durations: Dict[str, float] = {}
    
    def start_stage(self, stage_name: str) -> None:
        """
        Mark the start of a new stage.
        
        Args:
            stage_name: Name of the stage being started
        """
        # Update stage index
        if stage_name not in self.stages:
            self.stages.append(stage_name)
        
        self.current_stage_index = self.stages.index(stage_name)
        
        # Record start time
        self.stage_start_times[stage_name] = datetime.now()
        
        # Initialize progress
        self.stage_progress[stage_name] = 0.0
    
    def update_stage_progress(self, stage_name: str, progress: float) -> None:
        """
        Update progress for the current stage.
        
        Args:
            stage_name: Name of the stage
            progress: Progress percentage (0.0 - 1.0)
        """
        # Ensure progress is within valid range
        progress = max(0.0, min(1.0, progress))
        self.stage_progress[stage_name] = progress
    
    def complete_stage(self, stage_name: str) -> None:
        """
        Mark a stage as complete.
        
        Args:
            stage_name: Name of the stage completed
        """
        # Mark 100% progress
        self.stage_progress[stage_name] = 1.0
        
        # Calculate stage duration
        if stage_name in self.stage_start_times:
            duration = (datetime.now() - self.stage_start_times[stage_name]).total_seconds()
            self.stage_durations[stage_name] = duration
    
    def get_overall_progress(self) -> float:
        """
        Calculate overall pipeline progress.
        
        Returns:
            Overall progress percentage (0.0 - 1.0)
        """
        if not self.stages:
            return 0.0
        
        total_progress = sum(self.stage_progress.values())
        return total_progress / len(self.stages)
    
    def get_progress_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive progress report.
        
        Returns:
            Dictionary with detailed progress information
        """
        return {
            "experiment_name": self.config.name if self.config else "unknown",
            "start_time": self.start_time.isoformat(),
            "current_stage": self.stages[self.current_stage_index] if self.current_stage_index >= 0 else None,
            "overall_progress": self.get_overall_progress(),
            "stages": {
                stage: {
                    "progress": self.stage_progress.get(stage, 0.0),
                    "duration": self.stage_durations.get(stage, 0.0)
                }
                for stage in self.stages
            },
            "estimated_remaining_time": None  # Can be enhanced with more sophisticated estimation
        }
    
    def save_progress_report(self, filename: Optional[str] = None) -> str:
        """
        Save the current progress report to a file.
        
        Args:
            filename: Optional custom filename
        
        Returns:
            Path to the saved progress report
        """
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"progress_report_{timestamp}.json"
        
        # Construct full path
        progress_report_path = os.path.join(
            self.config.paths.experiment_dir if self.config else ".", 
            filename
        )
        
        # Get progress report
        report = self.get_progress_report()
        
        # Write to file
        with open(progress_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Progress report saved to {progress_report_path}")
        return progress_report_path
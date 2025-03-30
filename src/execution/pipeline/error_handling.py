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
from src.config.environment import get_environment_config
from src.config.path_config import get_path_config
from src.config.experiment import ExperimentConfiguration

# Import recovery mechanisms
from .recovery import RecoveryStrategy, RecoveryResult, ErrorRecoveryManager

# Try to import notebook-specific components
try:
    from IPython.display import display, HTML, clear_output
    from IPython import get_ipython
    NOTEBOOK_ENV = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
except (ImportError, NameError, AttributeError):
    NOTEBOOK_ENV = False

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
        max_error_entries: int = 100,
        recovery_enabled: bool = True
    ):
        """
        Initialize the error handler.
        
        Args:
            config: Optional experiment configuration
            max_error_entries: Maximum number of errors to track
            recovery_enabled: Whether to enable error recovery
        """
        self.config = config
        self.max_error_entries = max_error_entries
        self.errors: List[ErrorEntry] = []
        self.recovery_enabled = recovery_enabled
        
        # Get paths for error logging
        self.paths = get_path_config()
        
        # Setup error logging directory
        self.error_log_dir = self.paths.get_path('experiment_dir')
        if self.error_log_dir:
            os.makedirs(self.error_log_dir, exist_ok=True)
        
        # Initialize recovery manager if recovery is enabled
        self.recovery_manager = ErrorRecoveryManager(config=config) if recovery_enabled else None
    
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
    
    def attempt_recovery(
        self,
        error: Exception,
        stage: Any,  # BasePipelineStage
        config: ExperimentConfiguration,
        previous_results: Dict[str, Any]
    ) -> RecoveryResult:
        """
        Attempt to recover from an error.
        
        Args:
            error: The exception that occurred
            stage: The pipeline stage where the error occurred
            config: The experiment configuration
            previous_results: Results from previous stages
        
        Returns:
            Recovery result indicating success or failure
        """
        if not self.recovery_enabled or not self.recovery_manager:
            logger.warning("Recovery attempted but recovery is disabled")
            return RecoveryResult(success=False)
        
        # Attempt recovery
        return self.recovery_manager.attempt_recovery(
            error=error,
            stage=stage,
            config=config,
            previous_results=previous_results
        )
    
    def register_fallback_stage(self, stage_name: str, fallback_stage: Optional[Any]) -> None:
        """
        Register a fallback stage for a particular stage.
        
        Args:
            stage_name: Name of the stage that might fail
            fallback_stage: Fallback stage to use or None to skip
        """
        if self.recovery_enabled and self.recovery_manager:
            self.recovery_manager.register_fallback_stage(stage_name, fallback_stage)
    
    def create_checkpoint(self, stage: Any, results: Dict[str, Any]) -> Optional[str]:
        """
        Create a checkpoint for recovery.
        
        Args:
            stage: Stage that was completed
            results: Results from the stage
        
        Returns:
            Path to the created checkpoint file or None
        """
        if self.recovery_enabled and self.recovery_manager:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "experiment_name": self.config.name if self.config else "unknown"
            }
            return self.recovery_manager.create_checkpoint(stage, results, metadata)
        return None
    
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
        
        # Add recovery information if available
        if self.recovery_enabled and self.recovery_manager:
            error_log_data["recovery_summary"] = self.recovery_manager.get_recovery_summary()
        
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
        
        # Add recovery information if available
        if self.recovery_enabled and self.recovery_manager:
            error_summary["recovery"] = self.recovery_manager.get_recovery_summary()
        
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
        
        # Notebook display settings
        self.notebook_display_enabled = False
        self.notebook_progress_cell_id = None
        self.failed_stages = {}
    
    def configure_notebook_display(self, auto_clear: bool = True, update_interval: int = 1) -> None:
        """
        Configure progress tracking for notebook display.
        
        Args:
            auto_clear: Whether to clear output between updates
            update_interval: Interval between display refreshes in seconds
        """
        if not NOTEBOOK_ENV:
            logger.warning("Not running in a notebook environment, display configuration skipped")
            return
        
        self.notebook_display_enabled = True
        self.notebook_auto_clear = auto_clear
        self.notebook_update_interval = update_interval
        logger.info("Configured notebook display for progress tracking")
    
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
        
        # Update notebook display
        if self.notebook_display_enabled:
            self._update_notebook_display()
    
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
        
        # Update notebook display
        if self.notebook_display_enabled:
            self._update_notebook_display()
    
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
        
        # Update notebook display
        if self.notebook_display_enabled:
            self._update_notebook_display()
    
    def mark_stage_failed(self, stage_name: str) -> None:
        """
        Mark a stage as failed.
        
        Args:
            stage_name: Name of the stage that failed
        """
        # Record that stage failed
        if stage_name not in self.stage_progress:
            self.stage_progress[stage_name] = 0.0
        
        # Calculate stage duration
        if stage_name in self.stage_start_times:
            duration = (datetime.now() - self.stage_start_times[stage_name]).total_seconds()
            self.stage_durations[stage_name] = duration
        
        # Store failure information
        if not hasattr(self, 'failed_stages'):
            self.failed_stages = {}
        
        self.failed_stages[stage_name] = {
            "timestamp": datetime.now().isoformat(),
            "duration": self.stage_durations.get(stage_name, 0.0)
        }
        
        # Update notebook display
        if self.notebook_display_enabled:
            self._update_notebook_display(error=True)
    
    def _update_notebook_display(self, error: bool = False) -> None:
        """
        Update progress display in notebook environment.
        
        Args:
            error: Whether an error occurred in the current stage
        """
        if not NOTEBOOK_ENV:
            return
        
        # Clear previous output if configured
        if self.notebook_auto_clear:
            clear_output(wait=True)
        
        # Get current stage and progress
        current_stage = None
        if self.current_stage_index >= 0 and self.current_stage_index < len(self.stages):
            current_stage = self.stages[self.current_stage_index]
        
        current_progress = 0.0
        if current_stage and current_stage in self.stage_progress:
            current_progress = self.stage_progress[current_stage]
        
        # Calculate overall progress
        overall_progress = self.get_overall_progress()
        
        # Format time information
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        if elapsed_time < 60:
            elapsed_str = f"{elapsed_time:.1f} seconds"
        elif elapsed_time < 3600:
            elapsed_str = f"{elapsed_time / 60:.1f} minutes"
        else:
            elapsed_str = f"{elapsed_time / 3600:.1f} hours"
        
        # Format progress bars
        stage_progress_pct = int(current_progress * 100)
        overall_progress_pct = int(overall_progress * 100)
        
        # Create stage progress bar
        stage_progress_color = "#4CAF50"  # Green
        if error:
            stage_progress_color = "#F44336"  # Red
        elif stage_progress_pct < 30:
            stage_progress_color = "#FF9800"  # Orange
            
        stage_bar = f"""
        <div style="margin-top: 10px; margin-bottom: 10px;">
            <div style="background-color: #f1f1f1; border-radius: 5px; height: 24px; position: relative;">
                <div style="background-color: {stage_progress_color}; width: {stage_progress_pct}%; height: 100%; 
                           border-radius: 5px; position: absolute; transition: width 0.3s ease;">
                </div>
                <div style="position: absolute; width: 100%; height: 100%; display: flex; align-items: center; 
                           justify-content: center; color: #000000;">
                    {stage_progress_pct}%
                </div>
            </div>
        </div>
        """
        
        # Create overall progress bar
        overall_bar = f"""
        <div style="margin-bottom: 20px;">
            <div style="background-color: #f1f1f1; border-radius: 5px; height: 24px; position: relative;">
                <div style="background-color: #2196F3; width: {overall_progress_pct}%; height: 100%; 
                           border-radius: 5px; position: absolute; transition: width 0.3s ease;">
                </div>
                <div style="position: absolute; width: 100%; height: 100%; display: flex; align-items: center; 
                           justify-content: center; color: #000000;">
                    {overall_progress_pct}%
                </div>
            </div>
        </div>
        """
        
        # Create completed stages list
        completed_stages_html = ""
        for i, stage in enumerate(self.stages):
            if stage in self.stage_progress and self.stage_progress[stage] == 1.0:
                duration = self.stage_durations.get(stage, 0.0)
                duration_str = f"{duration:.1f}s" if duration < 60 else f"{duration / 60:.1f}m"
                
                completed_stages_html += f"""
                <div style="display: flex; justify-content: space-between; 
                            padding: 4px 10px; background-color: #E8F5E9; margin-bottom: 4px; 
                            border-left: 4px solid #4CAF50; border-radius: 2px;">
                    <span>{i+1}. {stage.replace('Stage', '')}</span>
                    <span>{duration_str}</span>
                </div>
                """
        
        # Create failed stages list
        failed_stages_html = ""
        for stage_name in self.failed_stages:
            failed_info = self.failed_stages[stage_name]
            duration = failed_info.get("duration", 0.0)
            duration_str = f"{duration:.1f}s" if duration < 60 else f"{duration / 60:.1f}m"
            
            failed_stages_html += f"""
            <div style="display: flex; justify-content: space-between; 
                        padding: 4px 10px; background-color: #FFEBEE; margin-bottom: 4px; 
                        border-left: 4px solid #F44336; border-radius: 2px;">
                <span>{stage_name.replace('Stage', '')}</span>
                <span>{duration_str}</span>
            </div>
            """
        
        # Create current stage indicator
        current_stage_html = ""
        if current_stage:
            stage_status = "Running"
            status_color = "#2196F3"  # Blue
            
            if error:
                stage_status = "Error"
                status_color = "#F44336"  # Red
            
            current_stage_html = f"""
            <div style="padding: 12px; background-color: #E3F2FD; margin-bottom: 10px; 
                       border-left: 5px solid {status_color}; border-radius: 2px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-weight: bold;">{current_stage.replace('Stage', '')}</span>
                    <span style="color: {status_color}; font-weight: bold;">{stage_status}</span>
                </div>
                {stage_bar}
            </div>
            """
        
        # Combine all elements
        html = f"""
        <div style="font-family: Arial, sans-serif; padding: 15px; border: 1px solid #ddd; 
                   border-radius: 8px; margin: 10px 0; max-width: 600px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                <h3 style="margin: 0;">Pipeline Progress</h3>
                <span style="color: #666;">Elapsed: {elapsed_str}</span>
            </div>
            
            <div style="margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>Overall Progress:</span>
                    <span>{overall_progress_pct}%</span>
                </div>
                {overall_bar}
            </div>
            
            {current_stage_html}
            
            <div style="display: flex;">
                <div style="flex: 1; margin-right: 10px;">
                    <h4 style="margin-top: 0;">Completed Stages</h4>
                    {completed_stages_html if completed_stages_html else "<p>No stages completed yet</p>"}
                </div>
                
                {f'''
                <div style="flex: 1;">
                    <h4 style="margin-top: 0; color: #D32F2F;">Failed Stages</h4>
                    {failed_stages_html}
                </div>
                ''' if failed_stages_html else ''}
            </div>
        </div>
        """
        
        # Display the HTML
        display(HTML(html))
    
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
        # Calculate elapsed time
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        # Generate report
        report = {
            "experiment_name": self.config.name if self.config else "unknown",
            "start_time": self.start_time.isoformat(),
            "elapsed_time": elapsed_time,
            "current_stage": self.stages[self.current_stage_index] if self.current_stage_index >= 0 else None,
            "overall_progress": self.get_overall_progress(),
            "stages": {
                stage: {
                    "progress": self.stage_progress.get(stage, 0.0),
                    "duration": self.stage_durations.get(stage, 0.0)
                }
                for stage in self.stages
            }
        }
        
        # Add failed stages if any
        if hasattr(self, 'failed_stages') and self.failed_stages:
            report["failed_stages"] = self.failed_stages
        
        return report
    
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
        
        # Get path configuration
        paths = get_path_config()
        
        # Construct full path
        progress_report_path = os.path.join(
            paths.get_results_path(filename)
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(progress_report_path), exist_ok=True)
        
        # Get progress report
        report = self.get_progress_report()
        
        # Write to file
        with open(progress_report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Progress report saved to {progress_report_path}")
        return progress_report_path
    
    def display_summary(self) -> None:
        """
        Display a summary of the pipeline execution.
        
        This is typically called after pipeline completion.
        """
        if not NOTEBOOK_ENV:
            logger.info("Not running in a notebook environment, skipping summary display")
            return
        
        # Format completion time
        total_time = (datetime.now() - self.start_time).total_seconds()
        if total_time < 60:
            total_time_str = f"{total_time:.1f} seconds"
        elif total_time < 3600:
            total_time_str = f"{total_time / 60:.1f} minutes"
        else:
            total_time_str = f"{total_time / 3600:.2f} hours"
        
        # Count completed and failed stages
        completed_stages = sum(1 for p in self.stage_progress.values() if p == 1.0)
        failed_stages = len(self.failed_stages) if hasattr(self, 'failed_stages') else 0
        total_stages = len(self.stages)
        
        # Create stage timing summary
        stage_timing_html = ""
        for stage in self.stages:
            if stage in self.stage_durations:
                duration = self.stage_durations[stage]
                
                # Format duration
                duration_str = f"{duration:.1f}s" if duration < 60 else f"{duration / 60:.1f}m"
                
                # Determine color based on relative duration
                is_failed = hasattr(self, 'failed_stages') and stage in self.failed_stages
                
                if is_failed:
                    color = "#FFEBEE"  # Light red
                    border_color = "#F44336"  # Red
                    status_text = "Failed"
                    status_color = "#D32F2F"  # Dark red
                else:
                    color = "#E8F5E9"  # Light green
                    border_color = "#4CAF50"  # Green
                    status_text = "Completed"
                    status_color = "#2E7D32"  # Dark green
                
                # Add to HTML
                stage_timing_html += f"""
                <div style="display: flex; justify-content: space-between; 
                            padding: 8px 12px; background-color: {color}; margin-bottom: 6px; 
                            border-left: 4px solid {border_color}; border-radius: 2px;">
                    <div>
                        <span style="font-weight: bold;">{stage.replace('Stage', '')}</span>
                        <span style="color: {status_color}; margin-left: 10px; font-size: 0.9em;">
                            {status_text}
                        </span>
                    </div>
                    <span>{duration_str}</span>
                </div>
                """
        
        # Create summary HTML
        summary_html = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px; border: 1px solid #ddd; 
                   border-radius: 8px; margin: 15px 0; max-width: 700px;">
            <h2 style="margin-top: 0; color: #2196F3;">Pipeline Execution Summary</h2>
            
            <div style="display: flex; flex-wrap: wrap; margin-bottom: 20px;">
                <div style="flex: 1; min-width: 200px; margin-right: 20px;">
                    <h4>Overview</h4>
                    <ul style="list-style-type: none; padding-left: 5px;">
                        <li><b>Total time:</b> {total_time_str}</li>
                        <li><b>Stages:</b> {completed_stages} completed / {total_stages} total</li>
                        <li>
                            <b>Status:</b> 
                            <span style="color: {'#D32F2F' if failed_stages > 0 else '#2E7D32'}; font-weight: bold;">
                                {'Failed' if failed_stages > 0 else 'Successful'}
                            </span>
                        </li>
                    </ul>
                </div>
                
                <div style="flex: 1; min-width: 200px;">
                    <h4>Statistics</h4>
                    <ul style="list-style-type: none; padding-left: 5px;">
                        <li><b>Completed stages:</b> {completed_stages}</li>
                        <li><b>Failed stages:</b> {failed_stages}</li>
                        <li><b>Success rate:</b> {(completed_stages / total_stages) * 100:.1f}%</li>
                    </ul>
                </div>
            </div>
            
            <h4>Stage Timing</h4>
            <div style="margin-bottom: 15px;">
                {stage_timing_html}
            </div>
        </div>
        """
        
        # Display the HTML
        display(HTML(summary_html))

# Export relevant classes
__all__ = ['ErrorEntry', 'ErrorHandler', 'ProgressTracker', 'RecoveryStrategy']
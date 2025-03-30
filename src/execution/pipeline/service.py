"""
Pipeline Execution Service

Provides the core orchestration mechanism for running 
extraction experiments through a comprehensive, 
stage-based pipeline with enhanced memory management,
robust error handling, and detailed progress tracking.
"""

import time
import logging
import gc
import traceback
from typing import Dict, Any, Optional, List, Type, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import pipeline components
from .base import BasePipelineStage, PipelineStageError, PipelineConfiguration
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
from .recovery import RecoveryStrategy, RecoveryResult, ErrorRecoveryManager

# Import project configurations
from src.config.experiment_config import ExperimentConfiguration
from src.models.model_service import get_model_service, optimize_memory

# Configure logging
logger = logging.getLogger(__name__)


class ExtractionPipelineService:
    """
    Primary service for orchestrating the entire extraction pipeline.
    
    Manages:
    - Stage execution with dependency injection
    - Memory optimization between stages
    - Error handling and recovery
    - Detailed progress tracking
    - Optional distributed processing
    """
    
    def __init__(
        self, 
        config: ExperimentConfiguration,
        stages: Optional[List[BasePipelineStage]] = None,
        error_handler: Optional[ErrorHandler] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        optimize_memory_between_stages: bool = True,
        enable_recovery: bool = True,
        distributed_execution: bool = False,
        worker_count: int = 1
    ):
        """
        Initialize the pipeline service with explicit dependencies.
        
        Args:
            config: Experiment configuration
            stages: Optional custom stages (defaults to standard pipeline)
            error_handler: Optional custom error handler
            progress_tracker: Optional custom progress tracker
            optimize_memory_between_stages: Whether to optimize memory between stages
            enable_recovery: Whether to enable error recovery
            distributed_execution: Whether to use distributed processing
            worker_count: Number of workers for distributed execution
        """
        self.config = config
        self.pipeline_config = PipelineConfiguration(config)
        
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
        
        # Initialize dependencies with dependency injection
        self.error_handler = error_handler or ErrorHandler(
            config=config,
            recovery_enabled=enable_recovery
        )
        
        self.progress_tracker = progress_tracker or ProgressTracker(
            config=config, 
            stages=[stage.__class__.__name__ for stage in self.stages]
        )
        
        # Memory management settings
        self.optimize_memory_between_stages = optimize_memory_between_stages
        
        # Error recovery settings
        self.enable_recovery = enable_recovery
        
        # Distributed execution settings
        self.distributed_execution = distributed_execution
        self.worker_count = max(1, worker_count)
        
        # Store stage dependencies for injection
        self.stage_dependencies = {}
        
        # Setup event hooks
        self.before_stage_hooks: List[Callable] = []
        self.after_stage_hooks: List[Callable] = []
        
        logger.info(f"Initialized pipeline with {len(self.stages)} stages")
        logger.info(f"Memory optimization: {optimize_memory_between_stages}")
        logger.info(f"Error recovery: {enable_recovery}")
        logger.info(f"Distributed execution: {distributed_execution} with {worker_count} workers")
    
    def add_before_stage_hook(self, hook: Callable) -> None:
        """
        Add a hook to run before each stage.
        
        Args:
            hook: Callable to execute before each stage
        """
        self.before_stage_hooks.append(hook)
    
    def add_after_stage_hook(self, hook: Callable) -> None:
        """
        Add a hook to run after each stage.
        
        Args:
            hook: Callable to execute after each stage
        """
        self.after_stage_hooks.append(hook)
    
    def register_dependency(self, stage_class: Type[BasePipelineStage], dependency: Any) -> None:
        """
        Register a dependency for a specific stage.
        
        Args:
            stage_class: Stage class to register dependency for
            dependency: Dependency to inject
        """
        stage_name = stage_class.__name__
        if stage_name not in self.stage_dependencies:
            self.stage_dependencies[stage_name] = []
        
        self.stage_dependencies[stage_name].append(dependency)
        logger.debug(f"Registered dependency for {stage_name}: {type(dependency).__name__}")
    
    def _inject_dependencies(self, stage: BasePipelineStage) -> None:
        """
        Inject dependencies for a stage.
        
        Args:
            stage: Stage to inject dependencies into
        """
        stage_name = stage.__class__.__name__
        if stage_name in self.stage_dependencies:
            for dependency in self.stage_dependencies[stage_name]:
                # Set dependency as attribute if stage has corresponding attribute
                attribute_name = self._get_dependency_attribute_name(dependency)
                if hasattr(stage, attribute_name):
                    setattr(stage, attribute_name, dependency)
                    logger.debug(f"Injected {attribute_name} into {stage_name}")
    
    def _get_dependency_attribute_name(self, dependency: Any) -> str:
        """
        Get attribute name for a dependency.
        
        Args:
            dependency: Dependency to get attribute name for
            
        Returns:
            Attribute name for the dependency
        """
        dependency_type = type(dependency).__name__
        return dependency_type[0].lower() + dependency_type[1:] 
    
    def _optimize_memory(self) -> Dict[str, Any]:
        """
        Optimize memory usage by cleaning up unused resources.
        
        Returns:
            Dictionary with memory optimization results
        """
        # Force garbage collection
        gc.collect()
        
        # Optimize GPU memory if available
        memory_info = optimize_memory()
        
        logger.info(f"Memory optimized. Freed: {memory_info.get('memory_freed_gb', 0):.2f} GB")
        return memory_info
    
    def _execute_stage(
        self, 
        stage: BasePipelineStage, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single stage with comprehensive error handling.
        
        Args:
            stage: Stage to execute
            previous_results: Results from previous stages
            
        Returns:
            Stage execution results
            
        Raises:
            PipelineStageError: If stage execution fails and recovery is disabled
        """
        stage_name = stage.__class__.__name__
        
        try:
            # Run before-stage hooks
            for hook in self.before_stage_hooks:
                hook(stage, self)
            
            # Update progress tracker
            self.progress_tracker.start_stage(stage_name)
            
            # Inject dependencies
            self._inject_dependencies(stage)
            
            # Execute stage
            logger.info(f"Executing stage: {stage_name}")
            stage_results = stage.execute(
                config=self.config, 
                previous_results=previous_results
            )
            
            # Mark stage complete
            self.progress_tracker.complete_stage(stage_name)
            
            # Create checkpoint if recovery is enabled
            if self.enable_recovery:
                checkpoint_path = self.error_handler.create_checkpoint(stage, stage_results)
                if checkpoint_path:
                    logger.info(f"Created checkpoint after stage {stage_name}: {checkpoint_path}")
            
            # Run after-stage hooks
            for hook in self.after_stage_hooks:
                hook(stage, self, stage_results)
            
            # Optimize memory if enabled
            if self.optimize_memory_between_stages:
                self._optimize_memory()
            
            return stage_results
            
        except Exception as e:
            # Handle stage execution error
            error_entry = self.error_handler.add_error(
                error=e, 
                stage=stage_name,
                context={
                    'previous_results': previous_results
                }
            )
            
            # Update progress tracker
            self.progress_tracker.mark_stage_failed(stage_name)
            
            # Attempt recovery if enabled
            if self.enable_recovery:
                recovery_result = self.error_handler.attempt_recovery(
                    error=e,
                    stage=stage,
                    config=self.config,
                    previous_results=previous_results
                )
                
                if recovery_result.success:
                    logger.info(f"Successfully recovered from error in stage {stage_name}")
                    return recovery_result.data
            
            # Re-raise the error if recovery failed or is disabled
            logger.error(f"Stage {stage_name} failed: {e}")
            raise PipelineStageError(
                message=f"Error in stage {stage_name}: {str(e)}",
                stage_name=stage_name,
                original_error=e
            )
    
    def _execute_distributed(self) -> PipelineExecutionResult:
        """
        Execute the pipeline using distributed processing.
        
        Returns:
            PipelineExecutionResult
        """
        # Create result object
        result = PipelineExecutionResult(config=self.config)
        
        # Track overall start time
        start_time = time.time()
        
        try:
            # Execute first stage (data preparation) normally
            first_stage = self.stages[0]
            previous_results = {
                first_stage.__class__.__name__.lower(): self._execute_stage(
                    first_stage, 
                    {}
                )
            }
            
            # Add stage results to overall results
            result.add_stage_result(
                first_stage.__class__.__name__, 
                previous_results[first_stage.__class__.__name__.lower()]
            )
            
            # Process remaining stages that can be distributed
            if len(self.stages) > 1:
                # Extract data that can be processed in parallel
                batch_items = self._extract_parallel_batch_items(previous_results)
                
                if batch_items:
                    # Process batch items in parallel
                    processed_results = self._process_batch_distributed(
                        batch_items, 
                        self.stages[1:], 
                        previous_results
                    )
                    
                    # Merge results back
                    for stage_idx, stage in enumerate(self.stages[1:], 1):
                        stage_name = stage.__class__.__name__
                        stage_results = processed_results.get(stage_name, {})
                        
                        # Add stage results to overall results
                        result.add_stage_result(stage_name, stage_results)
                        
                        # Update previous results for next stage
                        previous_results[stage_name.lower()] = stage_results
                else:
                    # Fall back to sequential execution if no batch items
                    logger.warning("No batch items found for distributed processing, falling back to sequential")
                    return self.run()
        
        except Exception as e:
            # Handle any errors in distributed execution
            logger.error(f"Distributed pipeline execution failed: {e}")
            logger.error(traceback.format_exc())
            result.set_status('failed')
            result.add_error({
                "message": str(e),
                "traceback": traceback.format_exc()
            })
        
        finally:
            # Calculate total execution time
            result.total_execution_time = time.time() - start_time
            
            # Save error log if any errors occurred
            if result.errors:
                self.error_handler.save_error_log()
            
            # Save progress report
            self.progress_tracker.save_progress_report()
        
        return result
    
    def _extract_parallel_batch_items(self, previous_results: Dict[str, Any]) -> List[Any]:
        """
        Extract batch items that can be processed in parallel.
        
        Args:
            previous_results: Results from previous stages
            
        Returns:
            List of batch items for parallel processing
        """
        # Try to find batch items in data preparation results
        data_prep_results = previous_results.get('datapreparationstage', {})
        
        batch_items = []
        for field, field_data in data_prep_results.items():
            if isinstance(field_data, dict) and 'batch_items' in field_data:
                batch_items.extend(field_data['batch_items'])
        
        return batch_items
    
    def _process_batch_distributed(
        self, 
        batch_items: List[Any],
        stages: List[BasePipelineStage],
        global_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process batch items using distributed execution.
        
        Args:
            batch_items: List of items to process
            stages: List of stages to execute for each item
            global_context: Shared context for all workers
            
        Returns:
            Dictionary with processed results
        """
        results = {}
        
        # Function to process a single batch item
        def process_item(item, item_idx):
            item_results = {}
            context = global_context.copy()
            
            try:
                # Execute each stage for this item
                for stage in stages:
                    stage_name = stage.__class__.__name__
                    
                    # Add the current item to context
                    context['current_item'] = item
                    context['item_index'] = item_idx
                    
                    # Execute stage for this item
                    stage_result = stage.execute(self.config, context)
                    
                    # Store result
                    item_results[stage_name] = stage_result
                    
                    # Update context for next stage
                    context[stage_name.lower()] = stage_result
                
                return item_idx, item_results
            except Exception as e:
                logger.error(f"Error processing batch item {item_idx}: {e}")
                return item_idx, {"error": str(e)}
        
        # Process items in parallel
        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            futures = [
                executor.submit(process_item, item, idx) 
                for idx, item in enumerate(batch_items)
            ]
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    item_idx, item_results = future.result()
                    
                    # Organize results by stage
                    for stage_name, stage_result in item_results.items():
                        if stage_name not in results:
                            results[stage_name] = {}
                        
                        # Store result for this item
                        results[stage_name][f"item_{item_idx}"] = stage_result
                        
                        # Update progress
                        self.progress_tracker.update_item_progress(
                            stage_name, 
                            len(futures), 
                            len([f for f in futures if f.done()])
                        )
                except Exception as e:
                    logger.error(f"Error retrieving batch processing result: {e}")
        
        return results
    
    def run(self) -> PipelineExecutionResult:
        """
        Execute the entire extraction pipeline with comprehensive tracking and error handling.
        
        Returns:
            PipelineExecutionResult with comprehensive experiment details
        """
        # Check if distributed execution is enabled
        if self.distributed_execution and self.worker_count > 1:
            logger.info(f"Running pipeline with distributed execution ({self.worker_count} workers)")
            return self._execute_distributed()
        
        # Create result object
        result = PipelineExecutionResult(config=self.config)
        
        # Track overall start time
        start_time = time.time()
        
        try:
            # Track overall pipeline progress
            self.progress_tracker.start_pipeline()
            
            # Store results from previous stages
            previous_results = {}
            
            # Execute each stage sequentially
            for stage in self.stages:
                stage_name = stage.__class__.__name__
                
                try:
                    # Execute the stage
                    stage_results = self._execute_stage(stage, previous_results)
                    
                    # Add stage results to overall results
                    result.add_stage_result(stage_name, stage_results)
                    
                    # Update previous results for next stage
                    previous_results[stage_name.lower()] = stage_results
                    
                except PipelineStageError as e:
                    # Handle stage-specific errors
                    error_entry = e.to_dict()
                    
                    # Add error to result
                    result.add_error(error_entry)
                    
                    # Update pipeline status
                    result.set_status('failed')
                    
                    # Check if we should continue despite error
                    if not self.config.get('continue_on_error', False):
                        logger.info("Stopping pipeline execution due to error")
                        break
                    
                    logger.info("Continuing pipeline execution despite error")
            
            # Calculate overall metrics if not already failed
            if result.status != 'failed':
                result.calculate_metrics()
                
                # Set final status
                result.set_status('success')
        
        except Exception as e:
            # Catch any unhandled errors
            logger.error(f"Pipeline execution failed: {e}")
            logger.error(traceback.format_exc())
            result.set_status('failed')
            result.add_error({
                "message": str(e),
                "traceback": traceback.format_exc()
            })
        
        finally:
            # Mark pipeline as complete
            self.progress_tracker.complete_pipeline()
            
            # Calculate total execution time
            result.total_execution_time = time.time() - start_time
            
            # Clean up resources
            if self.optimize_memory_between_stages:
                self._optimize_memory()
            
            # Save error log if any errors occurred
            if result.errors:
                self.error_handler.save_error_log()
            
            # Save progress report
            self.progress_tracker.save_progress_report()
        
        return result
    
    def dry_run(self) -> Dict[str, Any]:
        """
        Perform a dry run of the pipeline without full execution.
        
        Validates configuration and provides a preview of stages with dependency information.
        
        Returns:
            Dictionary with dry run information
        """
        dry_run_info = {
            "experiment_config": self.config.to_dict(),
            "stages": [
                {
                    "name": stage.__class__.__name__,
                    "dependencies": self.stage_dependencies.get(stage.__class__.__name__, []),
                    "expected_inputs": getattr(stage, "expected_inputs", []),
                    "expected_outputs": getattr(stage, "expected_outputs", [])
                }
                for stage in self.stages
            ],
            "validation_errors": self.config.validate(),
            "memory_optimization": self.optimize_memory_between_stages,
            "error_recovery": self.enable_recovery,
            "distributed_execution": {
                "enabled": self.distributed_execution,
                "worker_count": self.worker_count
            }
        }
        
        return dry_run_info


def run_extraction_pipeline(
    config: Optional[ExperimentConfiguration] = None,
    config_path: Optional[str] = None,
    stages: Optional[List[BasePipelineStage]] = None,
    optimize_memory: bool = True,
    enable_recovery: bool = True,
    distributed: bool = False,
    worker_count: int = 1,
    **kwargs
) -> PipelineExecutionResult:
    """
    Convenience function to run an extraction pipeline.
    
    Args:
        config: Optional ExperimentConfiguration
        config_path: Optional path to configuration file
        stages: Optional custom stages
        optimize_memory: Whether to optimize memory between stages
        enable_recovery: Whether to enable error recovery
        distributed: Whether to use distributed processing
        worker_count: Number of workers for distributed processing
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
    pipeline_service = ExtractionPipelineService(
        config=experiment_config,
        stages=stages,
        optimize_memory_between_stages=optimize_memory,
        enable_recovery=enable_recovery,
        distributed_execution=distributed,
        worker_count=worker_count
    )
    
    # Run the pipeline
    return pipeline_service.run()
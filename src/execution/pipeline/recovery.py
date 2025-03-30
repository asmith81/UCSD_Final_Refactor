"""
Error Recovery Mechanisms for Extraction Pipeline

This module provides robust error recovery strategies, checkpointing,
and state restoration capabilities for the extraction pipeline, enabling
resilient execution even in the face of errors or resource constraints.

Features include:
- Multiple recovery strategies for different error types
- Checkpoint creation and restoration
- Partial result recovery
- Automated retry with configurable policies
- Resource adaptation when memory constraints are encountered
"""

import os
import json
import logging
import time
import copy
import traceback
from typing import Dict, Any, List, Optional, Tuple, Callable, Type, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import tempfile
import pickle
import shutil

# Import pipeline components
from .base import BasePipelineStage, PipelineStageError, PipelineConfiguration
from src.config.experiment import ExperimentConfiguration
from src.config.paths import get_path_config

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    """
    Result of a recovery attempt.
    
    Contains information about whether recovery was successful,
    what data was recovered, and details about the recovery process.
    """
    success: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    strategy_used: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    recovery_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert recovery result to a dictionary.
        
        Returns:
            Dictionary representation of the recovery result
        """
        return {
            "success": self.success,
            "strategy_used": self.strategy_used,
            "recovery_time": self.recovery_time,
            "error_details": self.error_details
        }


class RecoveryStrategy:
    """
    Base class for recovery strategies.
    
    Defines the interface that all concrete recovery strategies must implement.
    """
    
    name: str = "base_strategy"
    description: str = "Base recovery strategy"
    
    def can_handle(self, error: Exception, stage: BasePipelineStage, context: Dict[str, Any]) -> bool:
        """
        Determine if this strategy can handle the given error.
        
        Args:
            error: The exception that occurred
            stage: The pipeline stage where the error occurred
            context: Additional context about the execution environment
            
        Returns:
            True if this strategy can handle the error, False otherwise
        """
        return False
    
    def recover(
        self, 
        error: Exception, 
        stage: BasePipelineStage,
        config: ExperimentConfiguration,
        previous_results: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RecoveryResult:
        """
        Attempt to recover from the given error.
        
        Args:
            error: The exception that occurred
            stage: The pipeline stage where the error occurred
            config: The experiment configuration
            previous_results: Results from previous stages
            context: Additional context about the execution environment
            
        Returns:
            Recovery result indicating success or failure
        """
        # Default implementation - no recovery
        return RecoveryResult(
            success=False,
            strategy_used=self.name,
            error_details={"message": str(error), "type": type(error).__name__}
        )


class RetryStrategy(RecoveryStrategy):
    """
    Simple retry strategy with configurable delays and attempt limits.
    
    Attempts to recover from transient errors by retrying the operation
    with increasing delays between attempts.
    """
    
    name: str = "retry"
    description: str = "Retry with exponential backoff"
    
    def __init__(
        self, 
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        transient_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """
        Initialize retry strategy.
        
        Args:
            max_attempts: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            backoff_factor: Factor to increase delay by with each attempt
            transient_exceptions: List of exception types that are considered transient
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        
        # Default transient exceptions if none provided
        self.transient_exceptions = transient_exceptions or [
            ConnectionError,
            TimeoutError,
            IOError,
            OSError
        ]
        
        # Track attempts per stage
        self.attempts: Dict[str, int] = {}
    
    def can_handle(self, error: Exception, stage: BasePipelineStage, context: Dict[str, Any]) -> bool:
        """
        Determine if this strategy can handle the given error.
        
        Args:
            error: The exception that occurred
            stage: The pipeline stage where the error occurred
            context: Additional context
            
        Returns:
            True if this is a transient error and max attempts not reached
        """
        stage_name = stage.__class__.__name__
        
        # Check if exception is one we consider transient
        is_transient = any(isinstance(error, ex_type) for ex_type in self.transient_exceptions)
        
        # Check if we've reached max attempts
        current_attempts = self.attempts.get(stage_name, 0)
        under_max_attempts = current_attempts < self.max_attempts
        
        return is_transient and under_max_attempts
    
    def recover(
        self, 
        error: Exception, 
        stage: BasePipelineStage,
        config: ExperimentConfiguration,
        previous_results: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RecoveryResult:
        """
        Attempt recovery by retrying the operation.
        
        Args:
            error: The exception that occurred
            stage: The pipeline stage where the error occurred
            config: The experiment configuration
            previous_results: Results from previous stages
            context: Additional context
            
        Returns:
            Recovery result
        """
        stage_name = stage.__class__.__name__
        
        # Increment attempt counter
        current_attempts = self.attempts.get(stage_name, 0)
        self.attempts[stage_name] = current_attempts + 1
        
        # Calculate delay
        delay = self.initial_delay * (self.backoff_factor ** current_attempts)
        
        logger.info(f"Retry strategy: Attempt {current_attempts + 1}/{self.max_attempts} "
                   f"for stage {stage_name}, waiting {delay:.2f} seconds")
        
        # Wait before retry
        time.sleep(delay)
        
        start_time = time.time()
        
        try:
            # Retry the stage execution
            stage_results = stage.execute(config=config, previous_results=previous_results)
            
            recovery_time = time.time() - start_time
            
            # Return successful recovery
            return RecoveryResult(
                success=True,
                data=stage_results,
                strategy_used=self.name,
                recovery_time=recovery_time
            )
            
        except Exception as retry_error:
            # Return failed recovery
            return RecoveryResult(
                success=False,
                strategy_used=self.name,
                error_details={
                    "message": str(retry_error),
                    "type": type(retry_error).__name__,
                    "attempt": current_attempts + 1,
                    "max_attempts": self.max_attempts
                }
            )


class MemoryOptimizationStrategy(RecoveryStrategy):
    """
    Recovery strategy for memory-related errors.
    
    Attempts to optimize memory usage when memory errors are encountered.
    """
    
    name: str = "memory_optimization"
    description: str = "Optimize memory and retry"
    
    def __init__(self, aggressive: bool = False):
        """
        Initialize memory optimization strategy.
        
        Args:
            aggressive: Whether to use aggressive memory optimization
        """
        self.aggressive = aggressive
        
        # Memory error types
        self.memory_error_types = [
            MemoryError,
            RuntimeError  # Some CUDA out of memory errors are RuntimeErrors
        ]
        
        # Memory error patterns in error messages
        self.memory_error_patterns = [
            "out of memory",
            "OOM",
            "memory allocation failed",
            "CUDA out of memory",
            "CUDA error: out of memory"
        ]
    
    def can_handle(self, error: Exception, stage: BasePipelineStage, context: Dict[str, Any]) -> bool:
        """
        Determine if this is a memory-related error.
        
        Args:
            error: The exception that occurred
            stage: The pipeline stage where the error occurred
            context: Additional context
            
        Returns:
            True if this is a memory-related error
        """
        # Check exception type
        is_memory_error_type = any(isinstance(error, error_type) for error_type in self.memory_error_types)
        
        # Check error message
        error_message = str(error).lower()
        has_memory_error_pattern = any(pattern.lower() in error_message for pattern in self.memory_error_patterns)
        
        return is_memory_error_type or has_memory_error_pattern
    
    def recover(
        self, 
        error: Exception, 
        stage: BasePipelineStage,
        config: ExperimentConfiguration,
        previous_results: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RecoveryResult:
        """
        Attempt recovery by optimizing memory usage and retrying.
        
        Args:
            error: The exception that occurred
            stage: The pipeline stage where the error occurred
            config: The experiment configuration
            previous_results: Results from previous stages
            context: Additional context
            
        Returns:
            Recovery result
        """
        import gc
        import torch
        
        logger.info(f"Memory optimization strategy: Attempting to recover from memory error in {stage.__class__.__name__}")
        
        start_time = time.time()
        
        # Optimize memory
        memory_freed = 0
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")
            
            if self.aggressive:
                # Get initial memory usage
                initial_memory = torch.cuda.memory_allocated()
                
                # Find and delete unused tensors
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.is_cuda:
                            obj.detach_()
                            del obj
                    except:
                        pass
                
                # Force garbage collection again
                gc.collect()
                torch.cuda.empty_cache()
                
                # Calculate freed memory
                final_memory = torch.cuda.memory_allocated()
                memory_freed = (initial_memory - final_memory) / (1024 ** 3)  # Convert to GB
                
                logger.info(f"Aggressive memory optimization freed {memory_freed:.2f} GB of CUDA memory")
        
        try:
            # Try to reduce batch size if applicable
            config_dict = config.to_dict()
            if 'batch_size' in config_dict:
                original_batch_size = config_dict['batch_size']
                # Reduce by 50%
                new_batch_size = max(1, original_batch_size // 2)
                config_dict['batch_size'] = new_batch_size
                
                # Create new config with reduced batch size
                optimized_config = ExperimentConfiguration.from_dict(config_dict)
                
                logger.info(f"Reduced batch size from {original_batch_size} to {new_batch_size}")
            else:
                optimized_config = config
            
            # Try to execute the stage with optimized memory
            stage_results = stage.execute(config=optimized_config, previous_results=previous_results)
            
            recovery_time = time.time() - start_time
            
            # Return successful recovery
            return RecoveryResult(
                success=True,
                data=stage_results,
                strategy_used=self.name,
                recovery_time=recovery_time
            )
            
        except Exception as retry_error:
            # Return failed recovery
            return RecoveryResult(
                success=False,
                strategy_used=self.name,
                error_details={
                    "message": str(retry_error),
                    "type": type(retry_error).__name__,
                    "memory_freed_gb": memory_freed
                }
            )


class CheckpointStrategy(RecoveryStrategy):
    """
    Recovery strategy using checkpoints.
    
    Creates checkpoints during pipeline execution and recovers from them when errors occur.
    """
    
    name: str = "checkpoint"
    description: str = "Recover from latest checkpoint"
    
    def __init__(
        self, 
        checkpoint_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        max_checkpoints: int = 5
    ):
        """
        Initialize checkpoint strategy.
        
        Args:
            checkpoint_interval: Number of stages between checkpoints
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        # Set up checkpoint directory
        self.paths = get_path_config()
        self.checkpoint_dir = checkpoint_dir or os.path.join(self.paths.get_path('experiment_dir'), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Track checkpoints
        self.checkpoints: List[str] = []
        self.last_checkpoint_stage = None
        self.stage_count = 0
    
    def can_handle(self, error: Exception, stage: BasePipelineStage, context: Dict[str, Any]) -> bool:
        """
        Determine if recovery from checkpoint is possible.
        
        Args:
            error: The exception that occurred
            stage: The pipeline stage where the error occurred
            context: Additional context
            
        Returns:
            True if there are available checkpoints
        """
        return len(self.checkpoints) > 0
    
    def create_checkpoint(
        self, 
        stage: BasePipelineStage, 
        results: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create a checkpoint after a stage.
        
        Args:
            stage: Stage that was completed
            results: Results from the stage
            metadata: Additional metadata
            
        Returns:
            Path to the created checkpoint file
        """
        stage_name = stage.__class__.__name__
        
        # Check if we should create a checkpoint
        self.stage_count += 1
        if self.stage_count % self.checkpoint_interval != 0:
            return None
        
        try:
            # Create checkpoint filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_filename = f"checkpoint_{stage_name}_{timestamp}.pkl"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            
            # Prepare checkpoint data
            checkpoint_data = {
                "stage_name": stage_name,
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "metadata": metadata
            }
            
            # Save checkpoint
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Update checkpoints list
            self.checkpoints.append(checkpoint_path)
            self.last_checkpoint_stage = stage_name
            
            # Remove old checkpoints if we exceed max
            if len(self.checkpoints) > self.max_checkpoints:
                oldest_checkpoint = self.checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    os.remove(oldest_checkpoint)
            
            logger.info(f"Created checkpoint after stage {stage_name}: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {str(e)}")
            return None
    
    def recover(
        self, 
        error: Exception, 
        stage: BasePipelineStage,
        config: ExperimentConfiguration,
        previous_results: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RecoveryResult:
        """
        Attempt recovery from the latest checkpoint.
        
        Args:
            error: The exception that occurred
            stage: The pipeline stage where the error occurred
            config: The experiment configuration
            previous_results: Results from previous stages
            context: Additional context
            
        Returns:
            Recovery result
        """
        if not self.checkpoints:
            return RecoveryResult(
                success=False,
                strategy_used=self.name,
                error_details={"message": "No checkpoints available"}
            )
        
        logger.info(f"Checkpoint strategy: Attempting to recover from latest checkpoint")
        
        start_time = time.time()
        
        try:
            # Load the latest checkpoint
            latest_checkpoint = self.checkpoints[-1]
            
            with open(latest_checkpoint, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            stage_name = checkpoint_data.get("stage_name")
            results = checkpoint_data.get("results", {})
            
            recovery_time = time.time() - start_time
            
            logger.info(f"Successfully recovered from checkpoint created after stage {stage_name}")
            
            # Return successful recovery
            return RecoveryResult(
                success=True,
                data=results,
                strategy_used=self.name,
                recovery_time=recovery_time,
                error_details={"recovered_from_stage": stage_name}
            )
            
        except Exception as recovery_error:
            # Return failed recovery
            return RecoveryResult(
                success=False,
                strategy_used=self.name,
                error_details={
                    "message": str(recovery_error),
                    "type": type(recovery_error).__name__
                }
            )


class AlternativePathStrategy(RecoveryStrategy):
    """
    Recovery strategy that attempts alternative execution paths.
    
    When a stage fails, this strategy attempts to complete the pipeline
    using an alternative execution path that bypasses the failed stage.
    """
    
    name: str = "alternative_path"
    description: str = "Use alternative execution path"
    
    def __init__(
        self, 
        fallback_stages: Dict[str, Optional[BasePipelineStage]] = None
    ):
        """
        Initialize alternative path strategy.
        
        Args:
            fallback_stages: Dictionary mapping stage names to fallback stages
        """
        self.fallback_stages = fallback_stages or {}
    
    def can_handle(self, error: Exception, stage: BasePipelineStage, context: Dict[str, Any]) -> bool:
        """
        Determine if we have an alternative path for this stage.
        
        Args:
            error: The exception that occurred
            stage: The pipeline stage where the error occurred
            context: Additional context
            
        Returns:
            True if we have a fallback for this stage
        """
        stage_name = stage.__class__.__name__
        
        # Check if we have a defined fallback
        has_fallback = stage_name in self.fallback_stages
        
        # If the fallback is None, it means we can skip this stage
        return has_fallback
    
    def recover(
        self, 
        error: Exception, 
        stage: BasePipelineStage,
        config: ExperimentConfiguration,
        previous_results: Dict[str, Any],
        context: Dict[str, Any]
    ) -> RecoveryResult:
        """
        Attempt recovery by using an alternative path.
        
        Args:
            error: The exception that occurred
            stage: The pipeline stage where the error occurred
            config: The experiment configuration
            previous_results: Results from previous stages
            context: Additional context
            
        Returns:
            Recovery result
        """
        stage_name = stage.__class__.__name__
        fallback_stage = self.fallback_stages.get(stage_name)
        
        logger.info(f"Alternative path strategy: "
                    f"{'Using fallback for' if fallback_stage else 'Skipping'} stage {stage_name}")
        
        start_time = time.time()
        
        try:
            if fallback_stage:
                # Execute fallback stage
                stage_results = fallback_stage.execute(config=config, previous_results=previous_results)
            else:
                # Skip the stage - create minimal results
                stage_results = {
                    "skipped": True,
                    "reason": "Skipped due to error",
                    "original_error": str(error)
                }
            
            recovery_time = time.time() - start_time
            
            # Return successful recovery
            return RecoveryResult(
                success=True,
                data=stage_results,
                strategy_used=self.name,
                recovery_time=recovery_time,
                error_details={
                    "action": "fallback" if fallback_stage else "skip",
                    "stage": stage_name
                }
            )
            
        except Exception as recovery_error:
            # Return failed recovery
            return RecoveryResult(
                success=False,
                strategy_used=self.name,
                error_details={
                    "message": str(recovery_error),
                    "type": type(recovery_error).__name__
                }
            )


class ErrorRecoveryManager:
    """
    Central manager for pipeline error recovery.
    
    Coordinates between different recovery strategies and maintains
    execution state for robust pipeline recovery.
    """
    
    def __init__(
        self, 
        config: Optional[ExperimentConfiguration] = None,
        strategies: Optional[List[RecoveryStrategy]] = None,
        create_checkpoints: bool = True,
        checkpoint_interval: int = 1,
        enable_aggressive_recovery: bool = False
    ):
        """
        Initialize error recovery manager.
        
        Args:
            config: Optional experiment configuration
            strategies: Optional list of recovery strategies
            create_checkpoints: Whether to create checkpoints
            checkpoint_interval: Interval between checkpoints
            enable_aggressive_recovery: Whether to use aggressive recovery techniques
        """
        self.config = config
        
        # Initialize default strategies if none provided
        if strategies is None:
            self.strategies = [
                RetryStrategy(),
                MemoryOptimizationStrategy(aggressive=enable_aggressive_recovery)
            ]
            
            # Add checkpoint strategy if enabled
            if create_checkpoints:
                self.strategies.append(
                    CheckpointStrategy(checkpoint_interval=checkpoint_interval)
                )
                
            # Add alternative path strategy
            self.strategies.append(AlternativePathStrategy())
        else:
            self.strategies = strategies
        
        # Initialize failed stage tracking
        self.failed_stages: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info(f"Initialized ErrorRecoveryManager with {len(self.strategies)} strategies")
        for strategy in self.strategies:
            logger.info(f"  - {strategy.name}: {strategy.description}")
    
    def create_checkpoint(
        self, 
        stage: BasePipelineStage, 
        results: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create a checkpoint after a stage.
        
        Args:
            stage: Stage that was completed
            results: Results from the stage
            metadata: Additional metadata
            
        Returns:
            Path to the created checkpoint file or None
        """
        # Find checkpoint strategy
        checkpoint_strategy = self._get_strategy_by_name('checkpoint')
        
        if checkpoint_strategy and isinstance(checkpoint_strategy, CheckpointStrategy):
            return checkpoint_strategy.create_checkpoint(stage, results, metadata)
        
        return None
    
    def register_fallback_stage(self, stage_name: str, fallback_stage: Optional[BasePipelineStage]) -> None:
        """
        Register a fallback stage for the alternative path strategy.
        
        Args:
            stage_name: Name of the stage that might fail
            fallback_stage: Fallback stage to use or None to skip
        """
        # Find alternative path strategy
        alternative_path_strategy = self._get_strategy_by_name('alternative_path')
        
        if alternative_path_strategy and isinstance(alternative_path_strategy, AlternativePathStrategy):
            alternative_path_strategy.fallback_stages[stage_name] = fallback_stage
            logger.info(f"Registered {'fallback' if fallback_stage else 'skip'} for stage {stage_name}")
    
    def _get_strategy_by_name(self, name: str) -> Optional[RecoveryStrategy]:
        """
        Get a strategy by name.
        
        Args:
            name: Name of the strategy to find
            
        Returns:
            Strategy or None if not found
        """
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        return None
    
    def attempt_recovery(
        self, 
        error: Exception, 
        stage: BasePipelineStage,
        config: ExperimentConfiguration,
        previous_results: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RecoveryResult:
        """
        Attempt to recover from an error using available strategies.
        
        Args:
            error: The exception that occurred
            stage: The pipeline stage where the error occurred
            config: The experiment configuration
            previous_results: Results from previous stages
            context: Additional context
            
        Returns:
            Recovery result
        """
        stage_name = stage.__class__.__name__
        context = context or {}
        
        logger.info(f"Attempting to recover from error in stage {stage_name}: {str(error)}")
        
        # Track stage failure
        if stage_name not in self.failed_stages:
            self.failed_stages[stage_name] = []
        
        self.failed_stages[stage_name].append({
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        })
        
        # Try each strategy in order
        for strategy in self.strategies:
            if strategy.can_handle(error, stage, context):
                logger.info(f"Trying recovery strategy: {strategy.name}")
                
                result = strategy.recover(error, stage, config, previous_results, context)
                
                if result.success:
                    logger.info(f"Successfully recovered using strategy: {strategy.name}")
                    return result
                else:
                    logger.info(f"Strategy {strategy.name} failed to recover")
        
        # All strategies failed
        logger.warning(f"All recovery strategies failed for error in stage {stage_name}")
        
        return RecoveryResult(
            success=False,
            error_details={
                "message": str(error),
                "type": type(error).__name__,
                "stage": stage_name,
                "strategies_attempted": [s.name for s in self.strategies]
            }
        )
    
    def get_recovery_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of recovery attempts.
        
        Returns:
            Dictionary with recovery summary information
        """
        return {
            "total_failed_stages": len(self.failed_stages),
            "failed_stages": {
                stage_name: {
                    "failure_count": len(failures),
                    "latest_failure": failures[-1] if failures else None,
                    "first_failure": failures[0] if failures else None
                }
                for stage_name, failures in self.failed_stages.items()
            },
            "available_strategies": [
                {
                    "name": strategy.name,
                    "description": strategy.description
                }
                for strategy in self.strategies
            ]
        }


# Export relevant classes
__all__ = [
    'RecoveryResult',
    'RecoveryStrategy',
    'RetryStrategy',
    'MemoryOptimizationStrategy',
    'CheckpointStrategy',
    'AlternativePathStrategy',
    'ErrorRecoveryManager'
] 
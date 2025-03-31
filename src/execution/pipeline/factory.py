"""
Pipeline Factory

This module provides factory methods for creating and configuring extraction pipelines
with appropriate stages, dependencies, and configuration. It streamlines the creation
of consistent pipeline instances across different experiment configurations.

The factory pattern centralizes pipeline creation logic and ensures proper dependency
injection, configuration validation, and resource management.
"""

import logging
from typing import Dict, Any, List, Optional, Type, Union, Callable, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

try:
    # Import IPython-specific components if available
    from IPython.display import display, HTML, JSON, clear_output
    from IPython import get_ipython
    NOTEBOOK_ENV = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
except (ImportError, NameError, AttributeError):
    NOTEBOOK_ENV = False

from src.config.experiment_config import ExperimentConfiguration, ExperimentType
from src.execution.pipeline.service import ExtractionPipelineService
from src.execution.pipeline.base import BasePipelineStage, PipelineConfiguration
from src.execution.pipeline.stages import (
    DataPreparationStage,
    ModelLoadingStage,
    PromptManagementStage,
    ExtractionStage,
    ResultsCollectionStage,
    AnalysisStage,
    VisualizationStage,
    ValidationStage,
    ResourceMonitoringStage,
    QuantizationSelectionStage,
    ExportStage
)
from src.execution.pipeline.error_handling import ErrorHandler, ProgressTracker
from src.models.model_service import get_model_service
from src.prompts.registry import get_prompt_registry
from src.results.collector import ResultsCollector
from src.analysis.visualization import get_visualization_service

# Import pipeline components
from .recovery import RecoveryStrategy, ErrorRecoveryManager, RetryStrategy, MemoryOptimizationStrategy, CheckpointStrategy

# Configure logging
logger = logging.getLogger(__name__)


class PipelineFactory:
    """
    Factory for creating and configuring extraction pipelines.
    
    This class centralizes the creation of extraction pipelines with appropriate
    stages, dependencies, and configuration based on experiment requirements.
    It ensures consistent pipeline setup and proper dependency injection.
    """
    
    @classmethod
    def create_pipeline(
        cls,
        config: Union[ExperimentConfiguration, Dict[str, Any]],
        stages: Optional[List[str]] = None,
        optimize_memory: bool = True,
        enable_recovery: bool = True,
        distributed: bool = False,
        worker_count: int = 1,
        custom_dependencies: Optional[Dict[str, Any]] = None
    ) -> ExtractionPipelineService:
        """
        Create a configured extraction pipeline.
        
        Args:
            config: Experiment configuration or configuration dictionary
            stages: Optional list of stages to include
            optimize_memory: Whether to optimize memory between stages
            enable_recovery: Whether to enable error recovery mechanisms
            distributed: Whether to use distributed processing
            worker_count: Number of workers for distributed execution
            custom_dependencies: Optional custom dependencies to inject
            
        Returns:
            Configured ExtractionPipelineService instance
        """
        # Convert dict to ExperimentConfiguration if needed
        if isinstance(config, dict):
            config = ExperimentConfiguration.from_dict(config)
        
        # Handle custom experiment type
        if config.type == ExperimentType.CUSTOM:
            # Create a pipeline with all stages
            stages = [
                DataPreparationStage(),
                ModelLoadingStage(),
                PromptManagementStage(),
                ExtractionStage(),
                ResultsCollectionStage(),
                ExportStage()
            ]
            
            # Add optional stages based on custom parameters
            if getattr(config, "include_analysis", True):
                stages.append(AnalysisStage())
                
            if getattr(config, "include_validation", True):
                stages.append(ValidationStage())
                
            if getattr(config, "include_visualization", True):
                stages.append(VisualizationStage())
                
            if getattr(config, "monitor_resources", False):
                stages.append(ResourceMonitoringStage())
            
            # Create and return pipeline
            return cls._create_pipeline_with_stages(config, stages)
        
        # Create pipeline stages
        pipeline_stages = cls._create_pipeline_stages(config, stages)
        
        # Initialize error recovery manager if recovery is enabled
        recovery_manager = None
        if enable_recovery:
            recovery_manager = ErrorRecoveryManager(
                config=config,
                create_checkpoints=True,
                checkpoint_interval=1,
                enable_aggressive_recovery=config.get('optimize_memory', False)
            )
            
            # Add strategy-specific configurations
            retry_strategy = RetryStrategy(
                max_attempts=3,
                initial_delay=1.0,
                backoff_factor=2.0
            )
            
            memory_strategy = MemoryOptimizationStrategy(
                aggressive=config.get('optimize_memory', False)
            )
            
            checkpoint_strategy = CheckpointStrategy(
                checkpoint_interval=1,
                max_checkpoints=5
            )
            
            # Set custom strategies if specified
            recovery_manager.strategies = [
                retry_strategy,
                memory_strategy,
                checkpoint_strategy
            ]
        
        # Create error handler with recovery manager
        error_handler = ErrorHandler(
            config=config, 
            recovery_enabled=enable_recovery
        )
        
        # Set recovery manager if we created one
        if recovery_manager and enable_recovery:
            error_handler.recovery_manager = recovery_manager
        
        # Create progress tracker
        progress_tracker = ProgressTracker(
            config=config,
            stages=[stage.__class__.__name__ for stage in pipeline_stages]
        )
        
        # Create pipeline service
        pipeline_service = ExtractionPipelineService(
            config=config,
            stages=pipeline_stages,
            error_handler=error_handler,
            progress_tracker=progress_tracker,
            optimize_memory_between_stages=optimize_memory,
            enable_recovery=enable_recovery,
            distributed_execution=distributed,
            worker_count=worker_count
        )
        
        # Inject common dependencies
        cls._inject_dependencies(pipeline_service, custom_dependencies)
        
        logger.info(f"Created pipeline with {len(pipeline_stages)} stages")
        return pipeline_service
    
    @classmethod
    def _create_pipeline_stages(
        cls,
        config: ExperimentConfiguration,
        stage_names: Optional[List[str]] = None
    ) -> List[BasePipelineStage]:
        """
        Create pipeline stages based on configuration and requested stages.
        
        Args:
            config: Experiment configuration
            stage_names: Optional list of stage names to include
            
        Returns:
            List of configured pipeline stages
        """
        # Define all available stages
        all_stages = {
            "data_preparation": DataPreparationStage,
            "quantization_selection": QuantizationSelectionStage,
            "model_loading": ModelLoadingStage,
            "prompt_management": PromptManagementStage,
            "extraction": ExtractionStage,
            "results_collection": ResultsCollectionStage,
            "analysis": AnalysisStage,
            "validation": ValidationStage,
            "visualization": VisualizationStage,
            "export": ExportStage,
            "resource_monitoring": ResourceMonitoringStage
        }
        
        # Determine which stages to include
        if stage_names is None:
            # Use default pipeline configuration
            default_pipeline = [
                "data_preparation",
                "model_loading",
                "prompt_management",
                "extraction",
                "results_collection",
                "analysis",
                "visualization"
            ]
            
            # Add quantization selection if experiment involves quantization
            if config.get('quantization') or config.quantization_strategies:
                default_pipeline.insert(1, "quantization_selection")
            
            # Add validation if explicitly enabled
            if config.get('validation', {}).get('enabled', False):
                default_pipeline.insert(-1, "validation")
            
            # Add export if explicitly enabled
            if config.get('export', {}).get('enabled', False):
                default_pipeline.append("export")
            
            # Add resource monitoring if explicitly enabled
            if config.get('monitoring', {}).get('enabled', False):
                default_pipeline.append("resource_monitoring")
            
            selected_stages = default_pipeline
        else:
            # Use explicitly specified stages
            selected_stages = stage_names
        
        # Create pipeline stages
        pipeline_stages = []
        
        for stage_name in selected_stages:
            if stage_name in all_stages:
                try:
                    # Create stage instance
                    stage_class = all_stages[stage_name]
                    stage = stage_class()
                    pipeline_stages.append(stage)
                    logger.info(f"Added stage: {stage.__class__.__name__}")
                except Exception as e:
                    logger.error(f"Error creating stage {stage_name}: {e}")
            else:
                logger.warning(f"Unknown stage: {stage_name}")
        
        return pipeline_stages
    
    @classmethod
    def _inject_dependencies(
        cls, 
        pipeline_service: ExtractionPipelineService,
        custom_dependencies: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Inject dependencies into pipeline stages.
        
        Args:
            pipeline_service: Pipeline service instance
            custom_dependencies: Optional custom dependencies to inject
        """
        config = pipeline_service.config
        
        # Create common dependencies
        model_service = get_model_service()
        prompt_registry = get_prompt_registry()
        results_collector = get_results_collector()
        visualization_service = get_visualization_service()
        
        # Register common dependencies
        pipeline_service.register_dependency(ModelLoadingStage, model_service)
        pipeline_service.register_dependency(PromptManagementStage, prompt_registry)
        pipeline_service.register_dependency(ResultsCollectionStage, results_collector)
        pipeline_service.register_dependency(VisualizationStage, visualization_service)
        pipeline_service.register_dependency(AnalysisStage, results_collector)
        pipeline_service.register_dependency(AnalysisStage, visualization_service)
        
        # Register custom dependencies if provided
        if custom_dependencies:
            for stage_class, dependency in custom_dependencies.items():
                pipeline_service.register_dependency(stage_class, dependency)
        
        # Configure resource monitoring if enabled
        if config.get('monitoring', {}).get('enabled', False):
            sampling_interval = config.get('monitoring', {}).get('sampling_interval', 5)
            
            # Get resource monitoring stage if it exists
            for stage in pipeline_service.stages:
                if isinstance(stage, ResourceMonitoringStage):
                    stage.start_resource_monitoring(interval=sampling_interval)
                    break

    @classmethod
    def _validate_extraction_fields(
        cls,
        config: ExperimentConfiguration,
        fields: List[str]
    ) -> List[str]:
        """
        Validate requested extraction fields against available fields.
        
        Args:
            config: Experiment configuration
            fields: List of fields to validate
            
        Returns:
            List of valid fields (subset of requested fields that exist)
        """
        # Get available extraction fields from configuration
        available_fields = config.get('extraction_fields', {}).keys()
        
        if not available_fields:
            # Fallback to default extraction fields if not specified in config
            available_fields = ["work_order", "cost", "date"]
        
        # Filter requested fields to only include valid ones
        valid_fields = [field for field in fields if field in available_fields]
        
        # Log warning for invalid fields
        invalid_fields = set(fields) - set(valid_fields)
        if invalid_fields:
            logger.warning(f"Ignoring invalid extraction fields: {', '.join(invalid_fields)}")
            logger.info(f"Available fields: {', '.join(available_fields)}")
        
        # If no valid fields, use the first available field
        if not valid_fields and available_fields:
            default_field = list(available_fields)[0]
            logger.warning(f"No valid fields specified, using default field: {default_field}")
            valid_fields = [default_field]
        
        return valid_fields

    @classmethod
    def create_quantization_pipeline(
        cls,
        config: ExperimentConfiguration,
        strategies: List[str],
        comparison_field: str = "work_order",
        comparison_set_size: int = 10
    ) -> ExtractionPipelineService:
        """
        Create a specialized pipeline for quantization comparison experiments.
        
        Args:
            config: Base experiment configuration
            strategies: List of quantization strategies to compare
            comparison_field: Field to use for comparison
            comparison_set_size: Size of the comparison set
            
        Returns:
            Configured pipeline for quantization experiments
        """
        # Validate field
        valid_fields = cls._validate_extraction_fields(config, [comparison_field])
        if not valid_fields:
            raise ValueError("Invalid comparison field specified")
        
        # Use the validated field
        comparison_field = valid_fields[0]
        
        # Create specialized configuration for quantization experiment
        quant_config = ExperimentConfiguration(
            name=f"{config.name}_quantization_experiment",
            description=f"Quantization comparison experiment using {len(strategies)} strategies",
            type="quantization_comparison",
            fields_to_extract=[comparison_field],
            model_name=config.model_name,
            quantization_strategies=strategies,
            dataset={
                "limit": comparison_set_size,
                "shuffle": True
            },
            visualization={
                "types": ["accuracy_bar", "memory_usage", "speed_comparison"]
            }
        )
        
        # Create pipeline with specialized stages
        pipeline_stages = [
            DataPreparationStage(),
            QuantizationSelectionStage(),
            ModelLoadingStage(),
            PromptManagementStage(),
            ExtractionStage(),
            ResultsCollectionStage(),
            AnalysisStage(),
            VisualizationStage(),
            ResourceMonitoringStage()
        ]
        
        # Create pipeline service
        pipeline_service = ExtractionPipelineService(
            config=quant_config,
            stages=pipeline_stages,
            optimize_memory_between_stages=True,
            enable_recovery=True
        )
        
        # Set up dependencies
        cls._inject_dependencies(pipeline_service, pipeline_stages)
        
        # Configure resource monitoring
        for stage in pipeline_stages:
            if isinstance(stage, ResourceMonitoringStage):
                stage.start_resource_monitoring(interval=1)
                break
        
        logger.info(f"Created quantization comparison pipeline with {len(strategies)} strategies")
        
        return pipeline_service

    @classmethod
    def create_prompt_comparison_pipeline(
        cls,
        config: ExperimentConfiguration,
        prompt_names: List[str] = None,
        prompt_category: str = None,
        distributed: bool = False
    ) -> ExtractionPipelineService:
        """
        Create a specialized pipeline for prompt comparison experiments.
        
        Args:
            config: Base experiment configuration
            prompt_names: Specific prompt names to compare (if provided)
            prompt_category: Prompt category to use (if prompt_names not provided)
            distributed: Whether to use distributed processing
            
        Returns:
            Configured pipeline for prompt comparison
        """
        # Validate fields
        valid_fields = cls._validate_extraction_fields(config, config.fields_to_extract)
        if not valid_fields:
            raise ValueError("No valid extraction fields specified")
        
        # Create a copy of the configuration
        prompt_config = ExperimentConfiguration(
            name=f"{config.name}_prompt_comparison",
            description="Prompt comparison experiment",
            type="prompt_comparison",
            fields_to_extract=valid_fields,
            model_name=config.model_name,
            prompt_names=prompt_names,
            prompt_category=prompt_category,
            visualization={
                "types": ["accuracy_comparison", "error_distribution", "prompt_radar"]
            }
        )
        
        # Create standard pipeline
        pipeline = cls.create_pipeline(
            config=prompt_config,
            optimize_memory=True,
            enable_recovery=True,
            distributed=distributed
        )
        
        logger.info(f"Created prompt comparison pipeline with {len(prompt_names) if prompt_names else 'all'} prompts")
        
        return pipeline

    @classmethod
    def create_field_extraction_pipeline(
        cls,
        config: ExperimentConfiguration,
        fields: Union[str, List[str]],
        optimize_for_speed: bool = False
    ) -> ExtractionPipelineService:
        """
        Create a specialized pipeline for field extraction.
        
        Args:
            config: Base experiment configuration
            fields: Specific field(s) to extract (string or list)
            optimize_for_speed: Whether to optimize pipeline for speed
            
        Returns:
            Configured pipeline for field extraction
        """
        # Handle single field as string
        if isinstance(fields, str):
            fields = [fields]
        
        # Validate fields
        valid_fields = cls._validate_extraction_fields(config, fields)
        
        if not valid_fields:
            raise ValueError("No valid extraction fields specified")
        
        # Create a field-specific configuration
        field_config = ExperimentConfiguration(
            name=f"{config.name}_{'_'.join(valid_fields)}_extraction",
            description=f"Extraction pipeline for fields: {', '.join(valid_fields)}",
            type="field_extraction",
            fields_to_extract=valid_fields,
            model_name=config.model_name
        )
        
        # Select stages based on optimization preference
        if optimize_for_speed:
            # Minimal pipeline for speed
            stages = [
                "data_preparation",
                "model_loading",
                "prompt_management",
                "extraction",
                "results_collection"
            ]
        else:
            # Standard pipeline
            stages = None
        
        # Create the pipeline
        pipeline = cls.create_pipeline(
            config=field_config,
            stages=stages,
            optimize_memory=True,
            enable_recovery=True,
            distributed=len(valid_fields) > 1  # Use distributed processing for multiple fields
        )
        
        logger.info(f"Created field extraction pipeline for: {', '.join(valid_fields)}")
        
        return pipeline

    @classmethod
    def create_model_comparison_pipeline(
        cls,
        model_names: List[str],
        fields: Union[str, List[str]],
        prompt_name: Optional[str] = None,
        sample_size: int = 20
    ) -> ExtractionPipelineService:
        """
        Create a specialized pipeline for comparing different models.
        
        Args:
            model_names: List of models to compare
            fields: Field(s) to extract for comparison (string or list)
            prompt_name: Specific prompt to use across all models (for fair comparison)
            sample_size: Number of samples to process
            
        Returns:
            Configured pipeline for model comparison
        """
        # Handle single field as string
        if isinstance(fields, str):
            fields = [fields]
        
        # Validate fields
        valid_fields = cls._validate_extraction_fields(
            ExperimentConfiguration(model_name=model_names[0]), 
            fields
        )
        
        if not valid_fields:
            raise ValueError("No valid extraction fields specified")
        
        # Create base configuration
        base_config = ExperimentConfiguration(
            name=f"model_comparison_{'_'.join(valid_fields)}",
            description=f"Comparison of {len(model_names)} models for extracting: {', '.join(valid_fields)}",
            type="model_comparison",
            fields_to_extract=valid_fields,
            model_name=model_names[0],  # Use first model as default
            dataset={
                "limit": sample_size,
                "shuffle": True
            },
            visualization={
                "types": ["model_accuracy_comparison", "model_speed_comparison", "model_memory_usage"]
            }
        )
        
        # Store all model names in metadata for the analysis stage
        base_config.parameters["comparison_models"] = model_names
        
        # Use specific prompt if provided
        if prompt_name:
            base_config.prompt_names = [prompt_name]
        
        # Create specialized stages for model comparison
        stages = [
            DataPreparationStage(),
            ModelLoadingStage(),  # Regular model loading stage is used
            PromptManagementStage(),
            ExtractionStage(),
            ResultsCollectionStage(),
            AnalysisStage(),  # Regular analysis stage is used
            VisualizationStage(),
            ResourceMonitoringStage()
        ]
        
        # Create pipeline service
        pipeline_service = ExtractionPipelineService(
            config=base_config,
            stages=stages,
            optimize_memory_between_stages=True,
            enable_recovery=True
        )
        
        # Set up dependencies
        model_service = get_model_service()
        prompt_registry = get_prompt_registry()
        results_collector = ResultsCollector(base_path="results")
        
        pipeline_service.register_dependency(ModelLoadingStage, model_service)
        pipeline_service.register_dependency(PromptManagementStage, prompt_registry)
        pipeline_service.register_dependency(ResultsCollectionStage, results_collector)
        
        logger.info(f"Created model comparison pipeline for {len(model_names)} models")
        
        return pipeline_service

    @classmethod
    def create_hybrid_experiment_pipeline(
        cls,
        models: List[str],
        prompts: List[str],
        fields: Union[str, List[str]],
        quantization_strategies: Optional[List[str]] = None,
        sample_size: int = 10
    ) -> ExtractionPipelineService:
        """
        Create a comprehensive hybrid experiment pipeline that can compare
        combinations of models, prompts, and quantization strategies.
        
        Args:
            models: List of models to compare
            prompts: List of prompts to compare
            fields: Field(s) to extract (string or list)
            quantization_strategies: Optional list of quantization strategies
            sample_size: Number of samples to process
            
        Returns:
            Configured pipeline for hybrid experiments
        """
        # Handle single field as string
        if isinstance(fields, str):
            fields = [fields]
        
        # Validate fields
        valid_fields = cls._validate_extraction_fields(
            ExperimentConfiguration(model_name=models[0]), 
            fields
        )
        
        if not valid_fields:
            raise ValueError("No valid extraction fields specified")
        
        # Create hybrid experiment configuration
        hybrid_config = ExperimentConfiguration(
            name=f"hybrid_experiment_{'_'.join(valid_fields)}",
            description=f"Hybrid experiment comparing {len(models)} models, "
                       f"{len(prompts)} prompts"
                       f"{f' and {len(quantization_strategies)} quantization strategies' if quantization_strategies else ''}"
                       f" for extracting: {', '.join(valid_fields)}",
            type="hybrid_experiment",
            fields_to_extract=valid_fields,
            model_name=models[0],  # Use first model as default
            prompt_names=prompts,
            dataset={
                "limit": sample_size,
                "shuffle": True
            },
            visualization={
                "types": ["hybrid_comparison_matrix", "performance_heatmap", "resource_usage"]
            }
        )
        
        # Store experiment parameters
        hybrid_config.parameters["comparison_models"] = models
        hybrid_config.parameters["comparison_prompts"] = prompts
        hybrid_config.parameters["comparison_quantization"] = quantization_strategies
        
        # Create standard stages for hybrid experiments (using existing stages)
        stages = [
            DataPreparationStage(),
            ModelLoadingStage(),
            PromptManagementStage(),
            ExtractionStage(),
            ResultsCollectionStage(),
            AnalysisStage(),
            VisualizationStage(),
            ExportStage(),
            ResourceMonitoringStage()
        ]
        
        # Create pipeline service with additional memory optimization
        pipeline_service = ExtractionPipelineService(
            config=hybrid_config,
            stages=stages,
            optimize_memory_between_stages=True,
            enable_recovery=True,
            distributed_execution=True,  # Use distributed execution for hybrid experiments
            worker_count=min(8, len(models) * (len(quantization_strategies or [1])))  # Scale workers based on combinations
        )
        
        # Set up dependencies
        model_service = get_model_service()
        prompt_registry = get_prompt_registry()
        results_collector = ResultsCollector(base_path="results")
        
        pipeline_service.register_dependency(ModelLoadingStage, model_service)
        pipeline_service.register_dependency(PromptManagementStage, prompt_registry)
        pipeline_service.register_dependency(ResultsCollectionStage, results_collector)
        
        # Start resource monitoring with higher frequency
        for stage in stages:
            if isinstance(stage, ResourceMonitoringStage):
                stage.start_resource_monitoring(interval=1)
                break
        
        logger.info(f"Created hybrid experiment pipeline with {len(models)*len(prompts)} combinations")
        
        return pipeline_service

    @classmethod
    def create_notebook_pipeline(
        cls,
        model_name: str = None,
        fields_to_extract: List[str] = None,
        template_name: str = "notebook_single_model",
        dataset_limit: int = 10,
        interactive: bool = True,
        optimize_memory: bool = True,
        **kwargs
    ) -> ExtractionPipelineService:
        """
        Create a notebook-friendly pipeline for interactive use.
        
        This method simplifies pipeline creation for notebook environments with
        sensible defaults and interactive progress tracking.
        
        Args:
            model_name: Model to use for extraction (optional, uses template default if not specified)
            fields_to_extract: Fields to extract (optional, uses template default if not specified)
            template_name: Template name to use as base configuration
            dataset_limit: Maximum number of samples to process
            interactive: Whether to enable interactive display updates
            optimize_memory: Whether to optimize memory usage
            **kwargs: Additional configuration parameters
            
        Returns:
            Notebook-friendly pipeline service
        """
        # Ensure we're in a notebook environment
        if not NOTEBOOK_ENV:
            logger.warning("Not running in a notebook environment, some features may not work")
        
        # Load template if provided
        try:
            template = get_template(template_name, category="notebook")
            if template is None:
                # Fallback to looking in other categories
                template = get_template(template_name)
            
            if template is None:
                logger.warning(f"Template '{template_name}' not found, using default configuration")
                config = ExperimentConfiguration(
                    name="notebook_extraction",
                    type=ExperimentType.NOTEBOOK_SINGLE_MODEL,
                    fields_to_extract=fields_to_extract or ["invoice_number", "date", "total_amount"],
                    model_name=model_name or "pixtral-12b",
                    dataset={"limit": dataset_limit, "shuffle": True},
                    notebook={"interactive": interactive, "display_progress": True}
                )
            else:
                # Create configuration from template
                config_params = {}
                if model_name:
                    config_params["model_name"] = model_name
                if fields_to_extract:
                    config_params["fields_to_extract"] = fields_to_extract
                if dataset_limit:
                    config_params["dataset"] = {"limit": dataset_limit, "shuffle": True}
                
                # Update with additional parameters
                config_params.update(kwargs)
                
                # Create configuration from template
                config = template.instantiate(**config_params)
        except Exception as e:
            logger.error(f"Error creating configuration from template: {e}")
            # Create default configuration
            config = ExperimentConfiguration(
                name="notebook_extraction",
                type=ExperimentType.NOTEBOOK_SINGLE_MODEL,
                fields_to_extract=fields_to_extract or ["invoice_number", "date", "total_amount"],
                model_name=model_name or "pixtral-12b",
                dataset={"limit": dataset_limit, "shuffle": True},
                notebook={"interactive": interactive, "display_progress": True}
            )
        
        # Set up interactive notebook settings
        notebook_settings = config.notebook or {}
        notebook_settings.update({
            "interactive": interactive,
            "display_progress": True,
            "visualization_inline": True
        })
        config.notebook = notebook_settings
        
        # Create pipeline with notebook-friendly stages
        stages = None  # Use default stages
        
        # Create pipeline
        pipeline = cls.create_pipeline(
            config=config,
            stages=stages,
            optimize_memory=optimize_memory,
            enable_recovery=True,
            distributed=False,
            worker_count=1
        )
        
        # Configure progress display for notebook environment
        if interactive and NOTEBOOK_ENV:
            pipeline.progress_tracker.configure_notebook_display()
        
        return pipeline

    @classmethod
    def create_model_comparison_notebook(
        cls,
        models: List[str],
        fields_to_extract: List[str] = None,
        prompt_name: str = None,
        dataset_limit: int = 20,
        **kwargs
    ) -> ExtractionPipelineService:
        """
        Create a notebook pipeline for comparing multiple models.
        
        Args:
            models: List of models to compare
            fields_to_extract: Fields to extract from invoices
            prompt_name: Prompt to use for all models
            dataset_limit: Maximum number of samples to process
            **kwargs: Additional configuration parameters
            
        Returns:
            Pipeline configured for model comparison
        """
        # Load appropriate template
        template = get_template("notebook_model_comparison", "notebook")
        
        # Create configuration parameters
        config_params = {
            "name": f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "fields_to_extract": fields_to_extract or ["invoice_number", "total_amount"],
            "model_comparison": {
                "models": models,
                "metrics": ["exact_match", "character_error_rate", "processing_time", "memory_usage"]
            },
            "dataset": {
                "limit": dataset_limit,
                "shuffle": True
            }
        }
        
        # Add prompt if specified
        if prompt_name:
            config_params["prompt_names"] = [prompt_name]
        
        # Update with additional parameters
        config_params.update(kwargs)
        
        # Create configuration
        if template:
            config = template.instantiate(**config_params)
        else:
            # Create default configuration
            config = ExperimentConfiguration(
                type=ExperimentType.NOTEBOOK_COMPARISON,
                **config_params
            )
        
        # Create specialized pipeline for model comparison
        pipeline = cls.create_pipeline(
            config=config,
            optimize_memory=True,
            enable_recovery=True
        )
        
        # Configure for notebook display
        if NOTEBOOK_ENV:
            pipeline.progress_tracker.configure_notebook_display()
        
        return pipeline

    @classmethod
    def create_quantization_notebook(
        cls,
        model_name: str,
        quantization_strategies: List[str],
        fields_to_extract: List[str] = None,
        dataset_limit: int = 20,
        **kwargs
    ) -> ExtractionPipelineService:
        """
        Create a notebook pipeline for analyzing quantization strategies.
        
        Args:
            model_name: Base model to quantize
            quantization_strategies: List of quantization strategies to compare
            fields_to_extract: Fields to extract from invoices
            dataset_limit: Maximum number of samples to process
            **kwargs: Additional configuration parameters
            
        Returns:
            Pipeline configured for quantization comparison
        """
        # Load appropriate template
        template = get_template("notebook_quantization_analysis", "notebook")
        
        # Create configuration parameters
        config_params = {
            "name": f"quantization_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_name": model_name,
            "fields_to_extract": fields_to_extract or ["invoice_number"],
            "quantization_strategies": quantization_strategies,
            "dataset": {
                "limit": dataset_limit,
                "shuffle": True
            },
            "visualization": {
                "types": ["accuracy_bar", "memory_usage", "speed_comparison", "quantization_heatmap"]
            }
        }
        
        # Update with additional parameters
        config_params.update(kwargs)
        
        # Create configuration
        if template:
            config = template.instantiate(**config_params)
        else:
            # Create default configuration
            config = ExperimentConfiguration(
                type=ExperimentType.QUANTIZATION_COMPARISON,
                **config_params
            )
        
        # Create specialized pipeline for quantization comparison
        pipeline = cls.create_quantization_pipeline(
            config=config,
            strategies=quantization_strategies,
            comparison_field=fields_to_extract[0] if fields_to_extract else "invoice_number",
            comparison_set_size=dataset_limit
        )
        
        # Configure for notebook display
        if NOTEBOOK_ENV:
            pipeline.progress_tracker.configure_notebook_display()
        
        return pipeline

    @classmethod
    def create_prompt_comparison_notebook(
        cls,
        prompt_names: List[str],
        model_name: str = None,
        fields_to_extract: List[str] = None,
        dataset_limit: int = 15,
        **kwargs
    ) -> ExtractionPipelineService:
        """
        Create a notebook pipeline for comparing multiple prompts.
        
        Args:
            prompt_names: List of prompts to compare
            model_name: Model to use for extraction
            fields_to_extract: Fields to extract from invoices
            dataset_limit: Maximum number of samples to process
            **kwargs: Additional configuration parameters
            
        Returns:
            Pipeline configured for prompt comparison
        """
        # Load appropriate template
        template = get_template("notebook_prompt_comparison", "notebook")
        
        # Create configuration parameters
        config_params = {
            "name": f"prompt_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "fields_to_extract": fields_to_extract or ["invoice_number", "date"],
            "prompt_names": prompt_names,
            "dataset": {
                "limit": dataset_limit,
                "shuffle": True
            },
            "visualization": {
                "types": ["accuracy_comparison", "error_distribution", "prompt_radar"]
            }
        }
        
        # Add model if specified
        if model_name:
            config_params["model_name"] = model_name
        
        # Update with additional parameters
        config_params.update(kwargs)
        
        # Create configuration
        if template:
            config = template.instantiate(**config_params)
        else:
            # Create default configuration
            config = ExperimentConfiguration(
                type=ExperimentType.PROMPT_COMPARISON,
                **config_params
            )
        
        # Create specialized pipeline for prompt comparison
        pipeline = cls.create_prompt_comparison_pipeline(
            config=config,
            prompt_names=prompt_names
        )
        
        # Configure for notebook display
        if NOTEBOOK_ENV:
            pipeline.progress_tracker.configure_notebook_display()
        
        return pipeline

    @classmethod
    def list_available_templates(cls) -> pd.DataFrame:
        """
        List available experiment templates in a notebook-friendly format.
        
        Returns:
            DataFrame with template information
        """
        if not NOTEBOOK_ENV:
            logger.warning("Not running in a notebook environment, returning plain list")
            return list_templates()
        
        templates = list_templates()
        
        # Convert to DataFrame for better notebook display
        df = pd.DataFrame(templates)
        
        # Display with styling
        if NOTEBOOK_ENV:
            styled_df = df.style.set_caption("Available Experiment Templates")
            display(styled_df)
        
        return df

    @classmethod
    def visualize_pipeline_stages(cls, pipeline: ExtractionPipelineService) -> None:
        """
        Visualize pipeline stages in a notebook environment.
        
        Args:
            pipeline: Pipeline to visualize
        """
        if not NOTEBOOK_ENV:
            logger.warning("Not running in a notebook environment, visualization not available")
            return
        
        # Create stage visualization
        stages = [stage.__class__.__name__.replace('Stage', '') for stage in pipeline.stages]
        stage_positions = list(range(len(stages)))
        
        # Create pipeline flow diagram
        fig, ax = plt.subplots(figsize=(14, 3))
        
        # Plot stage boxes
        for i, stage in enumerate(stages):
            rect = plt.Rectangle((i, 0.3), 0.8, 0.4, color='skyblue', alpha=0.8, ec='blue')
            ax.add_patch(rect)
            ax.text(i+0.4, 0.5, stage, ha='center', va='center', fontsize=9)
            
            # Add arrows
            if i < len(stages) - 1:
                ax.arrow(i+0.85, 0.5, 0.1, 0, head_width=0.05, head_length=0.05, 
                         fc='black', ec='black', length_includes_head=True)
        
        # Set plot parameters
        ax.set_xlim(-0.1, len(stages))
        ax.set_ylim(0, 1)
        ax.set_title('Pipeline Execution Flow')
        ax.axis('off')
        
        plt.tight_layout()
        display(fig)
        plt.close()


# Convenience functions for common pipeline creation scenarios

def create_extraction_pipeline(
    config: Union[ExperimentConfiguration, Dict[str, Any]],
    **kwargs
) -> ExtractionPipelineService:
    """
    Create a standard extraction pipeline.
    
    Args:
        config: Experiment configuration or configuration dictionary
        **kwargs: Additional arguments for pipeline creation
        
    Returns:
        Configured extraction pipeline
    """
    # Convert dictionary to ExperimentConfiguration if needed
    if isinstance(config, dict):
        config = ExperimentConfiguration.from_dict(config)
    
    return PipelineFactory.create_pipeline(config, **kwargs)


def create_quantization_comparison(
    model_name: str,
    strategies: List[str],
    field: str = "work_order",
    sample_size: int = 10
) -> ExtractionPipelineService:
    """
    Convenience function to create a quantization comparison pipeline.
    
    Args:
        model_name: Name of the model to test
        strategies: List of quantization strategies to compare
        field: Field to extract for comparison
        sample_size: Number of samples to use
        
    Returns:
        Configured quantization comparison pipeline
    """
    # Create simple configuration
    config = ExperimentConfiguration(
        name=f"{model_name}_quantization_comparison",
        model_name=model_name,
        fields_to_extract=[field]
    )
    
    return PipelineFactory.create_quantization_pipeline(
        config=config,
        strategies=strategies,
        comparison_field=field,
        comparison_set_size=sample_size
    )


def create_prompt_comparison(
    model_name: str,
    field: str,
    prompt_names: Optional[List[str]] = None,
    prompt_category: Optional[str] = None
) -> ExtractionPipelineService:
    """
    Convenience function to create a prompt comparison pipeline.
    
    Args:
        model_name: Name of the model to use
        field: Field to extract
        prompt_names: Specific prompt names to compare
        prompt_category: Prompt category to use
        
    Returns:
        Configured prompt comparison pipeline
    """
    config = ExperimentConfiguration(
        name=f"{field}_prompt_comparison",
        model_name=model_name,
        fields_to_extract=[field]
    )
    
    return PipelineFactory.create_prompt_comparison_pipeline(
        config=config,
        prompt_names=prompt_names,
        prompt_category=prompt_category
    )


def create_model_comparison(
    model_names: List[str],
    field: str = "work_order",
    prompt_name: Optional[str] = None,
    sample_size: int = 20
) -> ExtractionPipelineService:
    """
    Convenience function to create a model comparison pipeline.
    
    Args:
        model_names: List of models to compare
        field: Field to extract for comparison
        prompt_name: Specific prompt to use across all models
        sample_size: Number of samples to process
        
    Returns:
        Configured model comparison pipeline
    """
    return PipelineFactory.create_model_comparison_pipeline(
        model_names=model_names,
        fields=field,
        prompt_name=prompt_name,
        sample_size=sample_size
    )


def create_hybrid_experiment(
    models: List[str],
    prompts: List[str],
    fields: Union[str, List[str]] = "work_order",
    quantization_strategies: Optional[List[str]] = None,
    sample_size: int = 10
) -> ExtractionPipelineService:
    """
    Convenience function to create a hybrid experiment pipeline.
    
    Args:
        models: List of models to compare
        prompts: List of prompts to compare
        fields: Field(s) to extract (string or list)
        quantization_strategies: Optional list of quantization strategies
        sample_size: Number of samples to process
        
    Returns:
        Configured hybrid experiment pipeline
    """
    return PipelineFactory.create_hybrid_experiment_pipeline(
        models=models,
        prompts=prompts,
        fields=fields,
        quantization_strategies=quantization_strategies,
        sample_size=sample_size
    )

# Add these functions at the end of the file, after all existing functions

def create_notebook_pipeline(
    experiment_name: str,
    fields: Union[str, List[str]],
    model_name: str,
    prompt_name: Optional[str] = None,
    quantization: Optional[str] = None,
    sample_size: Optional[int] = None,
    ground_truth_path: Optional[str] = None,
    images_dir: Optional[str] = None,
    interactive: bool = True
) -> ExtractionPipelineService:
    """
    Create a notebook-friendly pipeline with simplified parameters.
    
    This function provides a simple interface for creating a pipeline in notebooks,
    with sensible defaults and rich output.
    
    Args:
        experiment_name: Name of the experiment
        fields: Field(s) to extract (string or list)
        model_name: Model to use
        prompt_name: Optional prompt name
        quantization: Optional quantization strategy
        sample_size: Optional sample size limitation
        ground_truth_path: Optional path to ground truth file
        images_dir: Optional path to images directory
        interactive: Whether to show interactive progress
        
    Returns:
        Configured notebook-friendly pipeline
    """
    # Convert single field to list
    if isinstance(fields, str):
        fields = [fields]
    
    # Create minimal configuration
    config = {
        'experiment_name': experiment_name,
        'fields_to_extract': fields,
        'model_name': model_name,
        'output': {
            'format': 'notebook',
            'visualize': True,
            'save_results': True
        },
        'dataset': {
            'limit': sample_size,
            'shuffle': True
        }
    }
    
    # Add prompt configuration if specified
    if prompt_name:
        config['prompt'] = {'name': prompt_name}
    
    # Add quantization if specified
    if quantization:
        config['quantization'] = quantization
    
    # Add ground truth and images paths if specified
    if ground_truth_path:
        config['ground_truth_path'] = ground_truth_path
    
    if images_dir:
        config['images_dir'] = images_dir
    
    # Create configuration object
    experiment_config = ExperimentConfiguration.from_dict(config)
    
    # Create pipeline using the factory method
    if interactive and NOTEBOOK_ENV:
        pipeline = PipelineFactory.create_notebook_pipeline(
            config=experiment_config,
            interactive_progress=True,
            memory_tracking=True,
            visualization_output="inline"
        )
    else:
        pipeline = PipelineFactory.create_pipeline(
            config=experiment_config
        )
    
    return pipeline

def run_notebook_extraction(
    experiment_name: str,
    fields: Union[str, List[str]],
    model_name: str,
    prompt_name: Optional[str] = None,
    quantization: Optional[str] = None,
    sample_size: Optional[int] = None,
    ground_truth_path: Optional[str] = None,
    images_dir: Optional[str] = None,
    visualize_results: bool = True,
    display_summary: bool = True
) -> Dict[str, Any]:
    """
    Run a complete extraction experiment in a notebook with a single function call.
    
    This function provides a simple interface for running a complete experiment
    in a Jupyter notebook with appropriate visualization and result display.
    
    Args:
        experiment_name: Name of the experiment
        fields: Field(s) to extract (string or list)
        model_name: Model to use
        prompt_name: Optional prompt name
        quantization: Optional quantization strategy
        sample_size: Optional sample size limitation
        ground_truth_path: Optional path to ground truth file
        images_dir: Optional path to images directory
        visualize_results: Whether to visualize results
        display_summary: Whether to display experiment summary
        
    Returns:
        Dictionary with experiment results
    """
    # Convert single field to list
    if isinstance(fields, str):
        fields = [fields]
    
    # Create minimal configuration
    config = {
        'experiment_name': experiment_name,
        'fields_to_extract': fields,
        'model_name': model_name,
        'output': {
            'format': 'notebook',
            'visualize': visualize_results,
            'save_results': True
        },
        'dataset': {
            'limit': sample_size,
            'shuffle': True
        }
    }
    
    # Add prompt configuration if specified
    if prompt_name:
        config['prompt'] = {'name': prompt_name}
    
    # Add quantization if specified
    if quantization:
        config['quantization'] = quantization
    
    # Add ground truth and images paths if specified
    if ground_truth_path:
        config['ground_truth_path'] = ground_truth_path
    
    if images_dir:
        config['images_dir'] = images_dir
    
    # Run experiment using the factory method
    return PipelineFactory.run_notebook_experiment(
        config=config,
        experiment_name=experiment_name,
        fields=fields,
        model_name=model_name,
        prompt_name=prompt_name,
        sample_size=sample_size,
        quantization=quantization,
        interactive=True,
        display_summary=display_summary
    )
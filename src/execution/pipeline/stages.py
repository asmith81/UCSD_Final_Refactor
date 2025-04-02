"""
Concrete implementations of pipeline stages for invoice extraction.

This module provides stage-specific processing logic for:
- Data preparation
- Model loading
- Prompt management
- Field extraction
- Results collection
- Analysis
- Visualization

Each stage follows consistent interfaces with robust validation,
enhanced error handling, detailed logging, and proper resource management.
"""

import os
import logging
import gc
import torch
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
import json

import pandas as pd

# Import base pipeline components
from .base import BasePipelineStage, PipelineStageError

# Import project modules
from src.config.experiment_config import ExperimentConfiguration
from src.data.loader import load_and_prepare_data
from src.models.model_service import get_model_service
from src.prompts.registry import get_prompt_registry
from src.execution.inference import process_image_with_metrics
from src.analysis.metrics import calculate_batch_metrics
from src.analysis.visualization import create_visualizations
from src.results.collector import ResultsCollector

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class StageRequirements:
    """
    Define input and output requirements for pipeline stages.
    
    This provides a standard for validating stage inputs and outputs.
    """
    required_inputs: Set[str] = field(default_factory=set)
    optional_inputs: Set[str] = field(default_factory=set)
    provided_outputs: Set[str] = field(default_factory=set)
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """
        Validate that all required inputs are present.
        
        Args:
            inputs: Dictionary of inputs to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check required inputs
        for required_input in self.required_inputs:
            if required_input not in inputs:
                errors.append(f"Missing required input: {required_input}")
        
        return errors


class DataPreparationStage(BasePipelineStage):
    """
    Prepare data for extraction by loading ground truth 
    and creating batches of images.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the data preparation stage."""
        super().__init__(name or "DataPreparation")
        
        # Define stage requirements
        self.requirements = StageRequirements(
            required_inputs=set(),
            optional_inputs=set(),
            provided_outputs={"ground_truth_df", "ground_truth_mapping", "batch_items"}
        )
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate input configuration for data preparation.
        
        Checks:
        - Ground truth path exists
        - Images directory exists
        - Fields to extract are specified
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Raises:
            PipelineStageError: If validation fails
        """
        # Validate ground truth path
        ground_truth_path = config.get('ground_truth_path')
        if not ground_truth_path:
            raise PipelineStageError(
                f"Ground truth path not specified in configuration",
                stage_name=self.name
            )
            
        if not os.path.exists(ground_truth_path):
            raise PipelineStageError(
                f"Ground truth file not found: {ground_truth_path}",
                stage_name=self.name
            )
        
        # Validate image directory
        images_dir = config.get('images_dir')
        if not images_dir:
            raise PipelineStageError(
                f"Images directory not specified in configuration",
                stage_name=self.name
            )
            
        if not os.path.exists(images_dir):
            raise PipelineStageError(
                f"Images directory not found: {images_dir}",
                stage_name=self.name
            )
        
        # Validate fields to extract
        fields = config.get('fields_to_extract', [])
        if not fields:
            raise PipelineStageError(
                "No fields specified for extraction",
                stage_name=self.name
            )
        
        logger.info(f"Configuration validated successfully for {self.name}")
        logger.info(f"Fields to extract: {', '.join(fields)}")
        logger.info(f"Using ground truth file: {ground_truth_path}")
        logger.info(f"Using images directory: {images_dir}")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare data for extraction by:
        - Loading ground truth
        - Creating image batches
        - Organizing data for processing
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Returns:
            Dictionary with prepared data for each field
            
        Raises:
            PipelineStageError: If data preparation fails
        """
        prepared_data = {}
        
        try:
            # Get configuration values
            ground_truth_path = config.get('ground_truth_path')
            images_dir = config.get('images_dir')
            fields_to_extract = config.get('fields_to_extract', [])
            
            logger.info(f"Starting data preparation for {len(fields_to_extract)} fields")
            
            # Apply data limits if specified
            data_limit = config.get('dataset', {}).get('limit')
            shuffle = config.get('dataset', {}).get('shuffle', False)
            
            if data_limit:
                logger.info(f"Applying data limit: {data_limit} items")
            
            if shuffle:
                logger.info("Shuffling data enabled")
                
            # Process each field
            for field in fields_to_extract:
                logger.info(f"Preparing data for field: {field}")
                
                # Determine the appropriate column name in the CSV
                csv_column_name = self._get_csv_column_name(config, field)
                logger.info(f"Using CSV column: {csv_column_name} for field {field}")
                
                try:
                    # Load and prepare data for each field
                    ground_truth_df, ground_truth_mapping, batch_items = load_and_prepare_data(
                        ground_truth_path=ground_truth_path,
                        image_dir=images_dir,
                        field_to_extract=field,
                        field_column_name=csv_column_name,
                        image_id_column='Invoice'
                    )
                    
                    # Apply limits and shuffling if specified
                    if shuffle and batch_items:
                        import random
                        random.shuffle(batch_items)
                        logger.info(f"Shuffled {len(batch_items)} items for field {field}")
                    
                    if data_limit and batch_items:
                        batch_items = batch_items[:int(data_limit)]
                        logger.info(f"Limited to {len(batch_items)} items for field {field}")
                    
                    prepared_data[field] = {
                        'ground_truth_df': ground_truth_df,
                        'ground_truth_mapping': ground_truth_mapping,
                        'batch_items': batch_items,
                        'total_items': len(batch_items)
                    }
                    
                    logger.info(f"Prepared {len(batch_items)} items for field {field}")
                    
                except Exception as e:
                    logger.error(f"Error preparing data for field {field}: {str(e)}")
                    raise PipelineStageError(
                        f"Error preparing data for field {field}: {str(e)}",
                        stage_name=self.name,
                        original_error=e
                    )
            
            logger.info(f"Data preparation completed successfully for {len(fields_to_extract)} fields")
            return prepared_data
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise PipelineStageError(
                f"Data preparation failed: {str(e)}",
                stage_name=self.name,
                original_error=e
            )
    
    def _get_csv_column_name(self, config: ExperimentConfiguration, field: str) -> str:
        """
        Get the appropriate CSV column name for a field.
        
        Args:
            config: Experiment configuration
            field: Field name
            
        Returns:
            CSV column name
        """
        # Check in extraction field configuration
        extraction_fields = config.get('extraction_fields', {})
        if field in extraction_fields and 'csv_column_name' in extraction_fields[field]:
            return extraction_fields[field]['csv_column_name']
        
        # Fallback to standard format
        return f"{field.replace('_', ' ').title()} Number"
    
    def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        logger.debug(f"Cleaning up resources for {self.name}")
        gc.collect()


class ModelLoadingStage(BasePipelineStage):
    """
    Load and prepare models for extraction.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the model loading stage."""
        super().__init__(name or "ModelLoading")
        
        # Define stage requirements
        self.requirements = StageRequirements(
            required_inputs=set(),
            optional_inputs=set(),
            provided_outputs={"model", "processor", "model_name", "quantization"}
        )
        
        # Model service dependency - will be injected
        self.model_service = None
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate model loading configuration.
        
        Checks:
        - Model name is specified
        - Required fields are present
        - Hardware requirements are met
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Raises:
            PipelineStageError: If validation fails
        """
        model_name = config.get('model_name')
        if not model_name:
            raise PipelineStageError(
                "No model name specified in configuration",
                stage_name=self.name
            )
        
        # Get model service
        model_service = self.model_service or get_model_service()
        
        # Check if model exists
        available_models = model_service.list_available_models()
        if model_name not in available_models:
            raise PipelineStageError(
                f"Model {model_name} not found in registry. Available models: {', '.join(available_models)}",
                stage_name=self.name
            )
        
        # Check hardware requirements
        can_run, issues = model_service.check_environment_for_model(model_name)
        if not can_run:
            raise PipelineStageError(
                f"Hardware requirements not met for model {model_name}: {', '.join(issues)}",
                stage_name=self.name
            )
        
        logger.info(f"Model {model_name} is available and hardware requirements are met")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Load models for each field to extract.
        
        Determines quantization strategy through:
        1. Explicitly specified quantization
        2. Optimal quantization based on hardware
        3. Fallback to default
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Returns:
            Dictionary with loaded model, processor, and metadata
            
        Raises:
            PipelineStageError: If model loading fails
        """
        # Load model with optimal quantization
        model_name = config.get('model_name')
        
        # Get model service
        model_service = self.model_service or get_model_service()
        
        try:
            # Determine quantization strategy
            quantization = None
            
            # Check for explicitly specified quantization
            if config.get('quantization'):
                quantization = config.get('quantization')
                logger.info(f"Using explicitly specified quantization: {quantization}")
            
            # Check for quantization strategies list
            elif config.quantization_strategies and len(config.quantization_strategies) > 0:
                quantization = config.quantization_strategies[0]
                logger.info(f"Using first quantization strategy from list: {quantization}")
            
            # If no explicit quantization, use model service to determine optimal
            if not quantization:
                try:
                    # Determine optimal quantization based on hardware and model requirements
                    quantization = model_service.get_optimal_quantization(
                        model_name=model_name,
                        prioritize=config.get('quantization_priority', 'balanced')
                    )
                    logger.info(f"Determined optimal quantization: {quantization}")
                except Exception as e:
                    logger.warning(f"Could not determine optimal quantization: {e}")
                    # Fallback to default
                    quantization = None
            
            # Log memory status before loading
            memory_before = self._get_memory_info()
            logger.info(f"Memory before model loading: {memory_before}")
            
            # Load model and processor
            logger.info(f"Loading model {model_name} with quantization {quantization}")
            
            loading_result = model_service.load_model(
                model_name=model_name,
                quantization=quantization,
                force_reload=config.get('force_model_reload', False)
            )
            
            if not loading_result.success:
                raise PipelineStageError(
                    f"Failed to load model {model_name}: {loading_result.error}",
                    stage_name=self.name,
                    original_error=loading_result.error
                )
            
            # Log memory status after loading
            memory_after = self._get_memory_info()
            logger.info(f"Memory after model loading: {memory_after}")
            logger.info(f"Memory used for model: {memory_after['allocated_memory_gb'] - memory_before['allocated_memory_gb']:.2f} GB")
            
            # Get quantization metadata and details
            quantization_metadata = {}
            try:
                model_config = model_service.get_model_config(model_name)
                if quantization:
                    quantization_metadata = model_config.get_quantization_metadata(quantization)
                    logger.info(f"Using quantization: {quantization_metadata['precision_name']} ({quantization_metadata['bits']} bits)")
            except Exception as e:
                logger.warning(f"Could not retrieve quantization metadata: {e}")
            
            result = {
                'model': loading_result.model,
                'processor': loading_result.processor,
                'model_name': model_name,
                'quantization': quantization,
                'quantization_metadata': quantization_metadata,
                'loading_time': loading_result.loading_time,
                'memory_info': loading_result.memory_used
            }
            
            logger.info(f"Model {model_name} loaded successfully in {loading_result.loading_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise PipelineStageError(
                f"Model loading failed: {str(e)}",
                stage_name=self.name,
                original_error=e
            )
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory information
        """
        memory_info = {
            "allocated_memory_gb": 0,
            "reserved_memory_gb": 0,
            "free_memory_gb": 0
        }
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            memory_info.update({
                "allocated_memory_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
                "reserved_memory_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
                "free_memory_gb": (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) - 
                                  (torch.cuda.memory_reserved(device) / (1024 ** 3))
            })
        
        return memory_info
    
    def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        logger.debug(f"Cleaning up resources for {self.name}")
        
        # Clean CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()

class QuantizationSelectionStage(BasePipelineStage):
    """
    Select optimal quantization strategy for model loading.
    
    This stage analyzes hardware capabilities and model requirements
    to determine the best quantization approach for the experiment.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the quantization selection stage."""
        super().__init__(name or "QuantizationSelection")
        
        # Define stage requirements
        self.requirements = StageRequirements(
            required_inputs=set(),
            optional_inputs=set(),
            provided_outputs={"selected_quantization", "quantization_metadata"}
        )
        
        # Model service dependency - will be injected
        self.model_service = None
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate configuration for quantization selection.
        
        Checks:
        - Model name is specified
        - Model is available in registry
        - Hardware requirements are accessible
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Raises:
            PipelineStageError: If validation fails
        """
        model_name = config.get('model_name')
        if not model_name:
            raise PipelineStageError(
                "No model name specified in configuration",
                stage_name=self.name
            )
        
        # Get model service
        model_service = self.model_service or get_model_service()
        
        # Check if model exists
        available_models = model_service.list_available_models()
        if model_name not in available_models:
            raise PipelineStageError(
                f"Model {model_name} not found in registry. Available models: {', '.join(available_models)}",
                stage_name=self.name
            )
        
        # If specific quantization strategies are provided, validate them
        if config.quantization_strategies:
            try:
                model_config = model_service.get_model_config(model_name)
                available_strategies = model_config.get_available_quantization_strategies()
                
                for strategy in config.quantization_strategies:
                    if strategy not in available_strategies:
                        raise PipelineStageError(
                            f"Invalid quantization strategy '{strategy}' for model {model_name}. "
                            f"Available strategies: {', '.join(available_strategies)}",
                            stage_name=self.name
                        )
            except Exception as e:
                raise PipelineStageError(
                    f"Error validating quantization strategies: {e}",
                    stage_name=self.name,
                    original_error=e
                )
        
        logger.info(f"Quantization selection configuration validated for model {model_name}")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine the optimal quantization strategy for model loading.
        
        The selection process follows this priority order:
        1. Explicitly specified quantization in configuration
        2. First strategy from quantization_strategies list
        3. Benchmark different strategies if benchmark_quantization is enabled
        4. Determine optimal strategy based on hardware capabilities
        5. Fallback to model's default quantization
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Returns:
            Dictionary with selected quantization strategy and metadata
            
        Raises:
            PipelineStageError: If quantization selection fails
        """
        try:
            model_name = config.get('model_name')
            logger.info(f"Selecting quantization strategy for model {model_name}")
            
            # Get model service
            model_service = self.model_service or get_model_service()
            model_config = model_service.get_model_config(model_name)
            
            # Initialize result
            result = {
                'selected_quantization': None,
                'quantization_metadata': {},
                'selection_method': None,
                'available_strategies': model_config.get_available_quantization_strategies(),
                'benchmarked': False,
                'benchmark_results': {}
            }
            
            # Step 1: Check for explicitly specified quantization
            if config.get('quantization'):
                quantization = config.get('quantization')
                logger.info(f"Using explicitly specified quantization: {quantization}")
                result['selected_quantization'] = quantization
                result['selection_method'] = 'explicit_config'
            
            # Step 2: Check for quantization strategies list
            elif config.quantization_strategies and len(config.quantization_strategies) > 0:
                quantization = config.quantization_strategies[0]
                logger.info(f"Using first quantization strategy from list: {quantization}")
                result['selected_quantization'] = quantization
                result['selection_method'] = 'strategy_list'
            
            # Step 3: Run benchmarks if enabled
            elif config.get('benchmark_quantization', False):
                logger.info("Benchmark mode enabled, comparing quantization strategies")
                benchmark_results = self._benchmark_quantization(model_name, model_service)
                
                # Select best strategy based on benchmarks
                if benchmark_results:
                    # Sort by metric (can customize the metric used)
                    sorted_results = sorted(
                        benchmark_results.items(),
                        key=lambda x: x[1].get('efficiency_score', float('inf'))
                    )
                    
                    # Use strategy with best score
                    best_strategy = sorted_results[0][0]
                    logger.info(f"Selected best strategy from benchmarks: {best_strategy}")
                    
                    result['selected_quantization'] = best_strategy
                    result['selection_method'] = 'benchmark'
                    result['benchmarked'] = True
                    result['benchmark_results'] = benchmark_results
                else:
                    logger.warning("Benchmark yielded no results, falling back to hardware-based selection")
            
            # Step 4: Use hardware-based optimal selection
            if not result['selected_quantization']:
                try:
                    # Determine optimal quantization based on hardware and model requirements
                    quantization = model_service.get_optimal_quantization(
                        model_name=model_name,
                        prioritize=config.get('quantization_priority', 'balanced')
                    )
                    logger.info(f"Determined optimal quantization based on hardware: {quantization}")
                    
                    result['selected_quantization'] = quantization
                    result['selection_method'] = 'hardware_optimal'
                except Exception as e:
                    logger.warning(f"Could not determine optimal quantization: {e}")
            
            # Step 5: Fallback to default
            if not result['selected_quantization']:
                # Get model's default quantization
                try:
                    default_strategy = model_config.get_available_quantization_strategies()[0]
                    logger.info(f"Using model's default quantization strategy: {default_strategy}")
                    
                    result['selected_quantization'] = default_strategy
                    result['selection_method'] = 'default'
                except Exception as e:
                    logger.error(f"Cannot determine default quantization: {e}")
                    raise PipelineStageError(
                        f"Failed to select quantization strategy: {e}",
                        stage_name=self.name,
                        original_error=e
                    )
            
            # Get metadata for selected strategy
            try:
                strategy = result['selected_quantization']
                if strategy:
                    metadata = model_config.get_quantization_metadata(strategy)
                    result['quantization_metadata'] = metadata
                    
                    logger.info(
                        f"Selected quantization: {strategy} "
                        f"({metadata.get('precision_name', 'unknown')}, "
                        f"{metadata.get('bits', 'unknown')} bits)"
                    )
            except Exception as e:
                logger.warning(f"Could not retrieve quantization metadata: {e}")
            
            # Generate and include hardware assessment
            result['hardware_assessment'] = self._assess_hardware_compatibility(
                model_name, 
                result['selected_quantization'],
                model_service
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Quantization selection failed: {str(e)}")
            raise PipelineStageError(
                f"Quantization selection failed: {str(e)}",
                stage_name=self.name,
                original_error=e
            )
    
    def _benchmark_quantization(
        self, 
        model_name: str,
        model_service: Any
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark different quantization strategies.
        
        Args:
            model_name: Name of the model
            model_service: Model service instance
            
        Returns:
            Dictionary with benchmark results for each strategy
        """
        logger.info(f"Benchmarking quantization strategies for {model_name}")
        
        try:
            # Get model config
            model_config = model_service.get_model_config(model_name)
            
            # Get available strategies
            strategies = model_config.get_available_quantization_strategies()
            
            if not strategies:
                logger.warning(f"No quantization strategies available for {model_name}")
                return {}
                
            logger.info(f"Benchmarking {len(strategies)} quantization strategies")
            
            # Run quick benchmark for each strategy
            results = {}
            
            for strategy in strategies:
                try:
                    # Start timing
                    start_time = time.time()
                    
                    # Try to load model with this strategy
                    logger.info(f"Testing strategy: {strategy}")
                    
                    # Clean memory before benchmark
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Record initial memory usage
                    start_memory = self._get_memory_info()
                    
                    # Load model
                    loading_result = model_service.load_model(
                        model_name=model_name,
                        quantization=strategy,
                        force_reload=True
                    )
                    
                    # Record memory after loading
                    end_memory = self._get_memory_info()
                    
                    # Calculate loading time
                    loading_time = time.time() - start_time
                    
                    # Run a small inference test if model loaded successfully
                    inference_time = None
                    if loading_result.success and loading_result.model is not None:
                        try:
                            # Create a small test input
                            device = next(loading_result.model.parameters()).device
                            
                            # Use random input tensors appropriate for the model
                            batch_size = 1
                            seq_length = 32
                            
                            if hasattr(loading_result.model, 'config'):
                                if hasattr(loading_result.model.config, 'hidden_size'):
                                    hidden_size = loading_result.model.config.hidden_size
                                else:
                                    hidden_size = 768  # Default for many models
                            else:
                                hidden_size = 768
                            
                            # Create test input tensor
                            test_input = torch.randint(0, 100, (batch_size, seq_length)).to(device)
                            
                            # Run inference
                            inference_start = time.time()
                            with torch.no_grad():
                                if hasattr(loading_result.model, 'generate'):
                                    _ = loading_result.model.generate(
                                        test_input, max_new_tokens=5
                                    )
                                else:
                                    _ = loading_result.model(test_input)
                            
                            inference_time = time.time() - inference_start
                            
                        except Exception as e:
                            logger.warning(f"Error during inference test for {strategy}: {e}")
                    
                    # Unload model
                    model_service.unload_model(model_name)
                    
                    # Clean up memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Store benchmark results
                    memory_used = end_memory["allocated_memory_gb"] - start_memory["allocated_memory_gb"]
                    
                    # Calculate efficiency score (can customize formula based on priorities)
                    # Lower is better - combines memory usage and speed
                    efficiency_score = (memory_used * 0.5) + (loading_time * 0.3)
                    if inference_time:
                        efficiency_score += (inference_time * 0.2)
                    
                    results[strategy] = {
                        "loading_success": loading_result.success,
                        "loading_time": loading_time,
                        "memory_used_gb": memory_used,
                        "inference_time": inference_time,
                        "efficiency_score": efficiency_score,
                        "error": str(loading_result.error) if loading_result.error else None
                    }
                    
                    # Log result
                    logger.info(
                        f"Benchmark result for {strategy}: "
                        f"Loading time={loading_time:.2f}s, "
                        f"Memory={memory_used:.2f}GB, "
                        f"Inference time={inference_time:.4f}s" if inference_time else "Inference time=N/A, "
                        f"Score={efficiency_score:.4f}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error benchmarking strategy {strategy}: {e}")
                    results[strategy] = {
                        "loading_success": False,
                        "error": str(e)
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during benchmarking: {e}")
            return {}
    
    def _assess_hardware_compatibility(
        self,
        model_name: str,
        quantization: str,
        model_service: Any
    ) -> Dict[str, Any]:
        """
        Assess hardware compatibility with selected quantization.
        
        Args:
            model_name: Name of the model
            quantization: Selected quantization strategy
            model_service: Model service instance
            
        Returns:
            Dictionary with hardware assessment
        """
        assessment = {
            "compatible": True,
            "warnings": [],
            "recommendations": []
        }
        
        try:
            # Get model config
            model_config = model_service.get_model_config(model_name)
            
            # Get quantization metadata
            metadata = model_config.get_quantization_metadata(quantization)
            
            # Get hardware info
            gpu_info = self._get_memory_info()
            
            # Check if GPU is required but not available
            if model_config.hardware_requirements.get("gpu_required", False) and not torch.cuda.is_available():
                assessment["compatible"] = False
                assessment["warnings"].append("Model requires GPU but none is available")
                assessment["recommendations"].append("Use a machine with GPU support")
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                if metadata.get("bits", 32) < 32:
                    assessment["warnings"].append(
                        f"Selected quantization ({quantization}) works best with GPU acceleration, but no GPU is available"
                    )
                assessment["recommendations"].append("Consider using a machine with GPU for better performance")
            
            # Check if memory is sufficient
            if torch.cuda.is_available():
                required_memory = model_config.get_memory_requirement_gb()
                bits = metadata.get("bits", 32)
                
                # Estimate memory requirement after quantization
                if bits < 32:
                    memory_factor = 32 / bits
                    required_memory = required_memory / memory_factor
                
                available_memory = gpu_info["free_memory_gb"]
                
                if required_memory > available_memory:
                    assessment["compatible"] = False
                    assessment["warnings"].append(
                        f"Insufficient GPU memory: {available_memory:.2f}GB available, {required_memory:.2f}GB required"
                    )
                    assessment["recommendations"].append("Try a more aggressive quantization strategy")
                    assessment["recommendations"].append("Close other applications to free up GPU memory")
                elif required_memory > (available_memory * 0.9):
                    assessment["warnings"].append(
                        f"GPU memory usage will be high: {required_memory:.2f}GB required, {available_memory:.2f}GB available"
                    )
                    assessment["recommendations"].append("Monitor memory usage during execution")
            
            # Add quantization-specific recommendations
            if metadata.get("precision_name") == "INT4":
                assessment["warnings"].append(
                    "INT4 quantization provides maximum memory savings but may reduce accuracy"
                )
            elif metadata.get("precision_name") == "INT8":
                assessment["warnings"].append(
                    "INT8 quantization balances memory usage and accuracy"
                )
            elif metadata.get("precision_name") in ["BF16", "FP16"]:
                if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major < 7:
                    assessment["warnings"].append(
                        f"{metadata.get('precision_name')} may not be fully accelerated on older GPUs"
                    )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error during hardware assessment: {e}")
            assessment["compatible"] = False
            assessment["warnings"].append(f"Error during hardware assessment: {str(e)}")
            return assessment
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory information
        """
        memory_info = {
            "allocated_memory_gb": 0,
            "reserved_memory_gb": 0,
            "free_memory_gb": 0
        }
        
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            memory_info.update({
                "allocated_memory_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
                "reserved_memory_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
                "free_memory_gb": (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) - 
                                  (torch.cuda.memory_reserved(device) / (1024 ** 3))
            })
        
        return memory_info
    
    def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        logger.debug(f"Cleaning up resources for {self.name}")
        
        # Clean CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()

class PromptManagementStage(BasePipelineStage):
    """
    Manage and prepare prompts for extraction.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the prompt management stage."""
        super().__init__(name or "PromptManagement")
        
        # Define stage requirements
        self.requirements = StageRequirements(
            required_inputs=set(),
            optional_inputs=set(),
            provided_outputs={"prompts_by_field"}
        )
        
        # Prompt registry dependency - will be injected
        self.prompt_registry = None
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate prompt configuration.
        
        Checks:
        - Fields for extraction are specified
        - Prompt category/names are valid if specified
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Raises:
            PipelineStageError: If validation fails
        """
        # Validate fields and prompt configuration
        fields = config.get('fields_to_extract', [])
        if not fields:
            raise PipelineStageError(
                "No fields specified for extraction",
                stage_name=self.name
            )
        
        # Get prompt registry
        prompt_registry = self.prompt_registry or get_prompt_registry()
        
        # Validate prompt category if specified
        if config.prompt_category:
            if config.prompt_category not in prompt_registry.list_categories():
                raise PipelineStageError(
                    f"Invalid prompt category: {config.prompt_category}. "
                    f"Available categories: {', '.join(prompt_registry.list_categories())}",
                    stage_name=self.name
                )
        
        # Validate prompt names if specified
        if config.prompt_names:
            available_prompts = prompt_registry.list_prompts()
            for prompt_name in config.prompt_names:
                if prompt_name not in available_prompts:
                    raise PipelineStageError(
                        f"Prompt not found: {prompt_name}. "
                        f"Available prompts: {', '.join(available_prompts)}",
                        stage_name=self.name
                    )
        
        logger.info(f"Prompt configuration validated for {len(fields)} fields")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare prompts for each field to extract.
        
        Handles:
        - Selection by category
        - Selection by explicit name
        - Fallback to default prompts
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Returns:
            Dictionary with prepared prompts by field
            
        Raises:
            PipelineStageError: If prompt preparation fails
        """
        # Get prompt registry
        prompt_registry = self.prompt_registry or get_prompt_registry()
        
        try:
            # Prepare prompts for each field
            field_prompts = {}
            fields = config.get('fields_to_extract', [])
            
            logger.info(f"Preparing prompts for {len(fields)} fields")
            
            for field in fields:
                prompts = []
                
                # Check for explicit prompt names first
                if config.prompt_names:
                    logger.info(f"Using explicitly specified prompts: {', '.join(config.prompt_names)}")
                    for prompt_name in config.prompt_names:
                        prompt = prompt_registry.get(prompt_name)
                        if prompt and prompt.field == field:
                            prompts.append(prompt.to_dict())
                
                # If no matching prompts found by name, check category
                if not prompts and config.prompt_category:
                    logger.info(f"Using prompts from category: {config.prompt_category}")
                    category_prompts = prompt_registry.get_by_field_and_category(
                        field, config.prompt_category
                    )
                    prompts = [p.to_dict() for p in category_prompts]
                
                # If still no prompts, get all for this field
                if not prompts:
                    logger.info(f"Using all available prompts for field: {field}")
                    field_specific_prompts = prompt_registry.get_by_field(field)
                    prompts = [p.to_dict() for p in field_specific_prompts]
                
                # If still no prompts, create a default prompt
                if not prompts:
                    logger.warning(f"No prompts found for field {field}. Creating default prompt.")
                    # Create a basic default prompt
                    default_prompt = {
                        'name': f'default_{field}',
                        'text': f'Extract the {field.replace("_", " ")} from this invoice image.',
                        'field': field,
                        'category': 'basic'
                    }
                    prompts = [default_prompt]
                
                logger.info(f"Prepared {len(prompts)} prompts for field {field}")
                field_prompts[field] = prompts
            
            # Log overall summary
            total_prompts = sum(len(prompts) for prompts in field_prompts.values())
            logger.info(f"Prepared {total_prompts} prompts across {len(fields)} fields")
            
            return {
                'prompts_by_field': field_prompts
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare prompts: {str(e)}")
            raise PipelineStageError(
                f"Prompt preparation failed: {str(e)}",
                stage_name=self.name,
                original_error=e
            )
    
    def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        logger.debug(f"Cleaning up resources for {self.name}")
        gc.collect()


class ExtractionStage(BasePipelineStage):
    """
    Perform field extraction across batches and prompts.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the extraction stage."""
        super().__init__(name or "Extraction")
        
        # Define stage requirements
        self.requirements = StageRequirements(
            required_inputs={"datapreparation", "modelloading", "promptmanagement"},
            optional_inputs=set(),
            provided_outputs={"field_extractions"}
        )
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate inputs for extraction stage.
        
        Checks:
        - Required data preparation results exist
        - Required model loading results exist
        - Required prompt results exist
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Raises:
            PipelineStageError: If validation fails
        """
        # Check required previous stage results
        required_keys = [
            'datapreparation', 
            'modelloading', 
            'promptmanagement'
        ]
        errors = self.requirements.validate_inputs(previous_results)
        if errors:
            raise PipelineStageError(
                f"Missing required inputs: {', '.join(errors)}",
                stage_name=self.name
            )
        
        # Validate data preparation results
        data_prep_results = previous_results['datapreparation']
        if not data_prep_results:
            raise PipelineStageError(
                "No data preparation results found",
                stage_name=self.name
            )
        
        # Validate model results
        model_results = previous_results['modelloading']
        if 'model' not in model_results or 'processor' not in model_results:
            raise PipelineStageError(
                "Missing model or processor in model loading results",
                stage_name=self.name
            )
        
        # Validate prompt results
        prompt_results = previous_results['promptmanagement']
        if 'prompts_by_field' not in prompt_results:
            raise PipelineStageError(
                "Missing prompts_by_field in prompt management results",
                stage_name=self.name
            )
        
        fields = config.get('fields_to_extract', [])
        for field in fields:
            if field not in data_prep_results:
                raise PipelineStageError(
                    f"Missing data preparation results for field: {field}",
                    stage_name=self.name
                )
                
            if field not in prompt_results['prompts_by_field']:
                raise PipelineStageError(
                    f"Missing prompts for field: {field}",
                    stage_name=self.name
                )
        
        logger.info(f"Extraction stage inputs validated successfully")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract fields using prepared models, prompts, and data.
        
        Performs:
        - Field extraction with each prompt
        - Result collection and error handling
        - Progress tracking
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Returns:
            Dictionary with extraction results by field and prompt
            
        Raises:
            PipelineStageError: If extraction fails
        """
        try:
            # Get models and processors
            model = previous_results['modelloading']['model']
            processor = previous_results['modelloading']['processor']
            model_name = previous_results['modelloading']['model_name']
            
            # Get data preparation results
            data_prep_results = previous_results['datapreparation']
            
            # Get prompt results
            prompt_results = previous_results['promptmanagement']['prompts_by_field']
            
            # Store extraction results
            all_field_results = {}
            
            # Get fields to extract
            fields = config.get('fields_to_extract', [])
            
            logger.info(f"Starting extraction for {len(fields)} fields")
            
            # Iterate through each field
            for field in fields:
                logger.info(f"Processing field: {field}")
                
                # Get field-specific data
                field_data = data_prep_results[field]
                batch_items = field_data['batch_items']
                
                # Get prompts for this field
                field_prompts = prompt_results[field]
                
                # Store results for this field
                field_results = {
                    'prompt_results': {},
                    'ground_truth_mapping': field_data['ground_truth_mapping'],
                    'total_items': field_data['total_items']
                }
                
                # Track total items processed
                total_items = len(batch_items)
                total_prompts = len(field_prompts)
                total_combinations = total_items * total_prompts
                
                logger.info(f"Processing {total_items} items with {total_prompts} prompts ({total_combinations} total extractions)")
                
                # Process with each prompt
                for prompt_index, prompt in enumerate(field_prompts):
                    prompt_name = prompt['name']
                    logger.info(f"Processing with prompt {prompt_index+1}/{total_prompts}: {prompt_name}")
                    
                    # Perform extraction for this prompt
                    extraction_results = []
                    successful_count = 0
                    error_count = 0
                    
                    # Process each batch item
                    for item_index, item in enumerate(batch_items):
                        # Log progress periodically
                        if (item_index + 1) % 10 == 0 or item_index == 0 or item_index == len(batch_items) - 1:
                            logger.info(f"Processing item {item_index+1}/{total_items} for prompt: {prompt_name}")
                        
                        try:
                            # Process the image
                            result = process_image_with_metrics(
                                image_path=item['image_path'],
                                ground_truth=item['ground_truth'],
                                prompt=prompt,
                                model_name=model_name,
                                field_type=field,
                                model=model,
                                processor=processor
                            )
                            
                            # Add to results
                            extraction_results.append(result)
                            
                            # Track success/failure
                            if result.get('exact_match', False):
                                successful_count += 1
                            
                            # Clean up memory periodically
                            if (item_index + 1) % 20 == 0:
                                self._clean_memory()
                                
                        except Exception as e:
                            logger.error(f"Error processing item {item_index+1} ({item['image_id']}): {e}")
                            
                            # Add error result
                            error_result = {
                                'error': str(e),
                                'image_path': item['image_path'],
                                'image_id': item['image_id'],
                                'ground_truth': item['ground_truth'],
                                'prompt': prompt_name,
                                'field': field
                            }
                            extraction_results.append(error_result)
                            error_count += 1
                    
                    # Store results for this prompt
                    field_results['prompt_results'][prompt_name] = extraction_results
                    
                    # Log prompt results summary
                    success_rate = successful_count / total_items if total_items > 0 else 0
                    logger.info(f"Prompt {prompt_name} results: "
                                f"{successful_count}/{total_items} successful extractions "
                                f"({success_rate:.2%}), {error_count} errors")
                
                # Store results for this field
                all_field_results[field] = field_results
                
                # Clean up memory after each field
                self._clean_memory()
            
            logger.info(f"Extraction completed for {len(fields)} fields")
            return {"field_extractions": all_field_results}
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise PipelineStageError(
                f"Extraction failed: {str(e)}",
                stage_name=self.name,
                original_error=e
            )
    
    def _clean_memory(self):
        """Clean up memory during extraction."""
        # Clean CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
    
    def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        logger.info(f"Cleaning up resources for {self.name}")
        self._clean_memory()


class ResultsCollectionStage(BasePipelineStage):
    """
    Collect and process extraction results.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the results collection stage."""
        super().__init__(name or "ResultsCollection")
        
        # Define stage requirements
        self.requirements = StageRequirements(
            required_inputs={"field_extractions"},
            optional_inputs=set(),
            provided_outputs={"processed_results", "results_collector"}
        )
        
        # Results collector dependency - will be injected
        self.results_collector = None
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate results before collection.
        
        Checks:
        - Field extraction results exist
        - Results structure is valid
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Raises:
            PipelineStageError: If validation fails
        """
        errors = self.requirements.validate_inputs(previous_results)
        if errors:
            raise PipelineStageError(
                f"Missing required inputs: {', '.join(errors)}",
                stage_name=self.name
            )
        
        # Validate extraction results
        if 'field_extractions' not in previous_results:
            raise PipelineStageError(
                "No field extractions found in previous results",
                stage_name=self.name
            )
        
        field_extractions = previous_results['field_extractions']
        if not field_extractions:
            raise PipelineStageError(
                "Empty field extractions results",
                stage_name=self.name
            )
        
        # Check that each field has prompt_results
        for field, field_data in field_extractions.items():
            if 'prompt_results' not in field_data:
                raise PipelineStageError(
                    f"Missing prompt_results for field: {field}",
                    stage_name=self.name
                )
        
        logger.info(f"Results collection inputs validated successfully")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process and collect extraction results.
        
        Performs:
        - Metrics calculation
        - Results storage
        - Summary generation
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Returns:
            Dictionary with processed results
            
        Raises:
            PipelineStageError: If results collection fails
        """
        try:
            # Create or get results collector
            results_collector = self.results_collector or ResultsCollector(
                base_path=config.get('results_dir', 'results'),
                experiment_name=config.get('name', 'extraction_experiment')
            )
            
            # Get field extractions
            field_extractions = previous_results['field_extractions']
            
            # Store processed results for each field
            processed_results = {}
            
            logger.info(f"Processing results for {len(field_extractions)} fields")
            
            # Iterate through field extractions
            for field, field_data in field_extractions.items():
                logger.info(f"Processing results for field: {field}")
                
                # Collect results for each prompt
                field_processed_results = {}
                
                for prompt_name, prompt_results in field_data['prompt_results'].items():
                    logger.info(f"Processing results for prompt: {prompt_name}")
                    
                    # Calculate metrics for this prompt's results
                    try:
                        metrics = calculate_batch_metrics(
                            prompt_results, 
                            field=field, 
                            config=config.to_dict()
                        )
                        
                        logger.info(f"Calculated metrics for {prompt_name}: "
                                    f"success_rate={metrics.get('success_rate', 0):.2%}, "
                                    f"total_count={metrics.get('total_count', 0)}")
                    except Exception as e:
                        logger.error(f"Error calculating metrics for prompt {prompt_name}: {e}")
                        metrics = {
                            "error": str(e),
                            "success_rate": 0,
                            "total_count": len(prompt_results)
                        }
                    
                    # Save field results
                    try:
                        results_collector.save_field_results(field, prompt_results)
                        logger.info(f"Saved {len(prompt_results)} results for field {field}, prompt {prompt_name}")
                    except Exception as e:
                        logger.error(f"Error saving results for field {field}, prompt {prompt_name}: {e}")
                    
                    try:
                        results_collector.save_field_metrics(field, metrics)
                        logger.info(f"Saved metrics for field {field}, prompt {prompt_name}")
                    except Exception as e:
                        logger.error(f"Error saving metrics for field {field}, prompt {prompt_name}: {e}")
                    
                    # Store processed results
                    field_processed_results[prompt_name] = {
                        'results': prompt_results,
                        'metrics': metrics
                    }
                
                processed_results[field] = field_processed_results
                
                # Generate and save field comparative metrics
                try:
                    field_comparison = self._generate_field_comparison(field, field_processed_results)
                    results_collector.save_comparative_metrics(field, field_comparison)
                    logger.info(f"Saved comparative metrics for field {field}")
                except Exception as e:
                    logger.error(f"Error generating comparison for field {field}: {e}")
            
            # Generate cross-field metrics
            try:
                cross_field_metrics = self._generate_cross_field_metrics(processed_results)
                results_collector.save_cross_field_metrics(cross_field_metrics)
                logger.info("Saved cross-field metrics")
            except Exception as e:
                logger.error(f"Error generating cross-field metrics: {e}")
            
            # Save experiment metadata
            try:
                metadata = {
                    "experiment_name": config.get('name'),
                    "model_name": config.get('model_name'),
                    "fields": list(field_extractions.keys()),
                    "timestamp": datetime.now().isoformat(),
                    "configuration": config.to_dict()
                }
                results_collector.save_run_metadata(metadata)
                logger.info("Saved experiment metadata")
            except Exception as e:
                logger.error(f"Error saving experiment metadata: {e}")
            
            logger.info("Results collection completed successfully")
            
            return {
                'processed_results': processed_results,
                'results_collector': results_collector
            }
            
        except Exception as e:
            logger.error(f"Results collection failed: {str(e)}")
            raise PipelineStageError(
                f"Results collection failed: {str(e)}",
                stage_name=self.name,
                original_error=e
            )
    
    def _generate_field_comparison(
        self,
        field: str,
        field_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate comparison metrics across prompts for a field.
        
        Args:
            field: Field name
            field_results: Results by prompt
            
        Returns:
            Dictionary with comparative metrics
        """
        comparison = {
            "field": field,
            "timestamp": datetime.now().isoformat(),
            "prompt_count": len(field_results),
            "prompt_metrics": {},
            "best_prompt": None,
            "worst_prompt": None
        }
        
        # Extract metrics for each prompt
        best_rate = -1
        worst_rate = 2  # Anything above 1 is fine for initialization
        best_prompt = None
        worst_prompt = None
        
        for prompt_name, prompt_data in field_results.items():
            metrics = prompt_data.get('metrics', {})
            success_rate = metrics.get('success_rate', 0)
            
            comparison["prompt_metrics"][prompt_name] = {
                "success_rate": success_rate,
                "total_count": metrics.get('total_count', 0),
                "error_count": metrics.get('error_count', 0)
            }
            
            # Track best and worst
            if success_rate > best_rate:
                best_rate = success_rate
                best_prompt = prompt_name
                
            if success_rate < worst_rate:
                worst_rate = success_rate
                worst_prompt = prompt_name
        
        comparison["best_prompt"] = {
            "name": best_prompt,
            "success_rate": best_rate
        } if best_prompt else None
        
        comparison["worst_prompt"] = {
            "name": worst_prompt,
            "success_rate": worst_rate
        } if worst_prompt else None
        
        return comparison
    
    def _generate_cross_field_metrics(
        self,
        processed_results: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Generate metrics comparing performance across different fields.
        
        Args:
            processed_results: Processed results by field and prompt
            
        Returns:
            Dictionary with cross-field metrics
        """
        cross_field = {
            "timestamp": datetime.now().isoformat(),
            "field_count": len(processed_results),
            "field_metrics": {},
            "best_field": None,
            "worst_field": None,
            "overall_metrics": {
                "total_extractions": 0,
                "successful_extractions": 0,
                "success_rate": 0
            }
        }
        
        # Calculate total extractions and success rates
        total_extractions = 0
        total_successes = 0
        
        # Extract metrics for each field
        best_rate = -1
        worst_rate = 2
        best_field = None
        worst_field = None
        
        for field, field_data in processed_results.items():
            field_success = 0
            field_total = 0
            
            # Aggregate metrics across prompts
            for prompt_name, prompt_data in field_data.items():
                metrics = prompt_data.get('metrics', {})
                success_count = metrics.get('success_count', 0)
                total_count = metrics.get('total_count', 0)
                
                field_success += success_count
                field_total += total_count
                
                total_successes += success_count
                total_extractions += total_count
            
            # Calculate field success rate
            field_rate = field_success / field_total if field_total > 0 else 0
            
            # Store field metrics
            cross_field["field_metrics"][field] = {
                "success_count": field_success,
                "total_count": field_total,
                "success_rate": field_rate
            }
            
            # Track best and worst
            if field_rate > best_rate:
                best_rate = field_rate
                best_field = field
                
            if field_rate < worst_rate:
                worst_rate = field_rate
                worst_field = field
        
        # Set overall metrics
        cross_field["overall_metrics"]["total_extractions"] = total_extractions
        cross_field["overall_metrics"]["successful_extractions"] = total_successes
        cross_field["overall_metrics"]["success_rate"] = (
            total_successes / total_extractions if total_extractions > 0 else 0
        )
        
        # Set best and worst fields
        cross_field["best_field"] = {
            "name": best_field,
            "success_rate": best_rate
        } if best_field else None
        
        cross_field["worst_field"] = {
            "name": worst_field,
            "success_rate": worst_rate
        } if worst_field else None
        
        return cross_field
    
    def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        logger.debug(f"Cleaning up resources for {self.name}")
        gc.collect()

class ExportStage(BasePipelineStage):
    """
    Export extraction results to various formats.
    
    This stage handles:
    - Exporting results to CSV, Excel, JSON, etc.
    - Generating formatted reports
    - Creating structured outputs for integration
    """
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate inputs for export stage.
        
        Checks:
        - Results from collection stage exist
        - Export formats are valid
        - Export directory is accessible
        """
        # Check results collection output exists
        if 'results_collection' not in previous_results:
            raise PipelineStageError("Missing results collection data for export")
        
        # Validate export formats if specified
        export_formats = config.get('export', {}).get('formats', ['json', 'csv'])
        valid_formats = ['json', 'csv', 'excel', 'parquet', 'html']
        
        for fmt in export_formats:
            if fmt.lower() not in valid_formats:
                raise PipelineStageError(f"Invalid export format: {fmt}. Valid formats: {', '.join(valid_formats)}")
        
        # Check if export directory is accessible
        export_dir = config.get('export', {}).get('directory')
        if export_dir and not os.access(export_dir, os.W_OK):
            raise PipelineStageError(f"Export directory not writable: {export_dir}")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Export results to specified formats.
        
        Supported formats:
        - JSON: Detailed structured data
        - CSV: Tabular extraction results
        - Excel: Multi-sheet workbook with results and metrics
        - Parquet: Columnar storage for analytics
        - HTML: Interactive report
        """
        # Get collected results
        collected_results = previous_results.get('results_collection', {})
        
        # Get export configuration
        export_config = config.get('export', {})
        export_formats = export_config.get('formats', ['json', 'csv'])
        export_dir = export_config.get('directory')
        
        # Use experiment directory if no specific export directory provided
        if not export_dir:
            paths = get_path_config()
            export_dir = paths.get('experiment_processed_dir')
            os.makedirs(export_dir, exist_ok=True)
        
        # Initialize export results
        export_results = {
            "exported_files": {},
            "formats": export_formats,
            "export_directory": export_dir
        }
        
        # Export in each requested format
        for fmt in export_formats:
            try:
                export_method = getattr(self, f"_export_to_{fmt.lower()}", None)
                
                if export_method:
                    export_path = export_method(collected_results, export_dir, config)
                    export_results["exported_files"][fmt] = export_path
                    self.logger.info(f"Exported results to {fmt.upper()}: {export_path}")
                else:
                    self.logger.warning(f"Export method for {fmt} not implemented")
            except Exception as e:
                self.logger.error(f"Error exporting to {fmt}: {e}")
                export_results["exported_files"][fmt] = {
                    "error": str(e),
                    "success": False
                }
        
        # Create export summary
        export_results["summary"] = {
            "successful_exports": sum(1 for v in export_results["exported_files"].values() 
                                    if isinstance(v, str)),
            "failed_exports": sum(1 for v in export_results["exported_files"].values() 
                                if isinstance(v, dict) and not v.get("success", False)),
            "timestamp": datetime.now().isoformat()
        }
        
        return export_results
    
    def _export_to_json(
        self, 
        collected_results: Dict[str, Any], 
        export_dir: str,
        config: ExperimentConfiguration
    ) -> str:
        """Export results to JSON format."""
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{config.name}_results_{timestamp}.json"
        export_path = os.path.join(export_dir, export_name)
        
        # Write JSON file
        with open(export_path, 'w') as f:
            json.dump(collected_results, f, indent=2)
        
        return export_path
    
    def _export_to_csv(
        self, 
        collected_results: Dict[str, Any], 
        export_dir: str,
        config: ExperimentConfiguration
    ) -> str:
        """Export results to CSV format."""
        # Create DataFrame from results
        data_rows = []
        
        # Process each field
        for field, field_results in collected_results.items():
            # Process each prompt's results
            for prompt_name, prompt_data in field_results.items():
                # Process individual extraction results
                for result in prompt_data.get('results', []):
                    row = {
                        'field': field,
                        'prompt_name': prompt_name,
                        'image_id': result.get('image_id', ''),
                        'ground_truth': result.get('ground_truth', ''),
                        'extraction': result.get('processed_extraction', ''),
                        'exact_match': result.get('exact_match', False),
                        'character_error_rate': result.get('character_error_rate', 1.0),
                        'processing_time': result.get('processing_time', 0.0),
                        'error': result.get('error', '')
                    }
                    data_rows.append(row)
        
        # Create DataFrame
        import pandas as pd
        df = pd.DataFrame(data_rows)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{config.name}_results_{timestamp}.csv"
        export_path = os.path.join(export_dir, export_name)
        
        # Write CSV file
        df.to_csv(export_path, index=False)
        
        return export_path
    
    def _export_to_excel(
        self, 
        collected_results: Dict[str, Any], 
        export_dir: str,
        config: ExperimentConfiguration
    ) -> str:
        """Export results to Excel format with multiple sheets."""
        import pandas as pd
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{config.name}_results_{timestamp}.xlsx"
        export_path = os.path.join(export_dir, export_name)
        
        # Create Excel writer
        with pd.ExcelWriter(export_path, engine='xlsxwriter') as writer:
            # Create summary sheet
            summary_data = {
                'field': [],
                'prompt': [],
                'total_samples': [],
                'successful_extractions': [],
                'accuracy': [],
                'avg_char_error_rate': [],
                'avg_processing_time': []
            }
            
            # Process each field
            for field, field_results in collected_results.items():
                # Process each prompt's results
                for prompt_name, prompt_data in field_results.items():
                    results = prompt_data.get('results', [])
                    if not results:
                        continue
                    
                    # Calculate metrics
                    total = len(results)
                    successful = sum(1 for r in results if r.get('exact_match', False))
                    accuracy = successful / total if total > 0 else 0
                    avg_cer = sum(r.get('character_error_rate', 1.0) for r in results) / total if total > 0 else 1.0
                    avg_time = sum(r.get('processing_time', 0.0) for r in results) / total if total > 0 else 0.0
                    
                    # Add to summary data
                    summary_data['field'].append(field)
                    summary_data['prompt'].append(prompt_name)
                    summary_data['total_samples'].append(total)
                    summary_data['successful_extractions'].append(successful)
                    summary_data['accuracy'].append(accuracy)
                    summary_data['avg_char_error_rate'].append(avg_cer)
                    summary_data['avg_processing_time'].append(avg_time)
                    
                    # Create detailed sheet for each prompt
                    detail_rows = []
                    for result in results:
                        row = {
                            'image_id': result.get('image_id', ''),
                            'ground_truth': result.get('ground_truth', ''),
                            'extraction': result.get('processed_extraction', ''),
                            'exact_match': result.get('exact_match', False),
                            'character_error_rate': result.get('character_error_rate', 1.0),
                            'processing_time': result.get('processing_time', 0.0),
                            'error': result.get('error', '')
                        }
                        detail_rows.append(row)
                    
                    # Create and write detail sheet
                    detail_df = pd.DataFrame(detail_rows)
                    sheet_name = f"{field}_{prompt_name}"[:31]  # Excel sheet name limit
                    detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Create and write summary sheet
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Add experiment metadata sheet
            metadata = {
                'key': ['experiment_name', 'model_name', 'fields_extracted', 'export_time'],
                'value': [
                    config.name,
                    config.get('model_name', 'unknown'),
                    ', '.join(collected_results.keys()),
                    timestamp
                ]
            }
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        return export_path
    
    def _export_to_parquet(
        self, 
        collected_results: Dict[str, Any], 
        export_dir: str,
        config: ExperimentConfiguration
    ) -> str:
        """Export results to Parquet format for analytics."""
        import pandas as pd
        
        # Similar approach to CSV but using Parquet format
        data_rows = []
        
        # Process each field
        for field, field_results in collected_results.items():
            # Process each prompt's results
            for prompt_name, prompt_data in field_results.items():
                # Process individual extraction results
                for result in prompt_data.get('results', []):
                    row = {
                        'field': field,
                        'prompt_name': prompt_name,
                        'image_id': result.get('image_id', ''),
                        'ground_truth': result.get('ground_truth', ''),
                        'extraction': result.get('processed_extraction', ''),
                        'exact_match': result.get('exact_match', False),
                        'character_error_rate': result.get('character_error_rate', 1.0),
                        'processing_time': result.get('processing_time', 0.0),
                        'error': result.get('error', '')
                    }
                    data_rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{config.name}_results_{timestamp}.parquet"
        export_path = os.path.join(export_dir, export_name)
        
        # Write Parquet file
        df.to_parquet(export_path, index=False)
        
        return export_path
    
    def _export_to_html(
        self, 
        collected_results: Dict[str, Any], 
        export_dir: str,
        config: ExperimentConfiguration
    ) -> str:
        """Export results to interactive HTML report."""
        import pandas as pd
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_name = f"{config.name}_report_{timestamp}.html"
        export_path = os.path.join(export_dir, export_name)
        
        # Prepare data for HTML report
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>Extraction Results: {config.name}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1, h2, h3 { color: #2c3e50; }",
            ".summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f2f2f2; }",
            "tr:nth-child(even) { background-color: #f9f9f9; }",
            ".success { color: green; }",
            ".failure { color: red; }",
            ".chart-container { height: 300px; margin-bottom: 30px; }",
            "</style>",
            # Include Chart.js for visualizations
            "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>",
            "</head>",
            "<body>",
            f"<h1>Extraction Results: {config.name}</h1>",
            f"<p>Model: {config.get('model_name', 'Unknown')}</p>",
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
            "<div class='summary'>",
            "<h2>Summary</h2>"
        ]
        
        # Process data for summary and detailed tables
        field_summaries = {}
        
        for field, field_results in collected_results.items():
            field_summary = {
                'total_prompts': len(field_results),
                'total_samples': 0,
                'successful_extractions': 0,
                'best_prompt': {'name': '', 'accuracy': 0},
                'prompt_data': []
            }
            
            for prompt_name, prompt_data in field_results.items():
                results = prompt_data.get('results', [])
                if not results:
                    continue
                
                # Calculate metrics
                total = len(results)
                successful = sum(1 for r in results if r.get('exact_match', False))
                accuracy = successful / total if total > 0 else 0
                
                # Update field summary
                field_summary['total_samples'] += total
                field_summary['successful_extractions'] += successful
                
                # Track best prompt
                if accuracy > field_summary['best_prompt']['accuracy']:
                    field_summary['best_prompt'] = {
                        'name': prompt_name,
                        'accuracy': accuracy
                    }
                
                # Store prompt data for detailed section
                field_summary['prompt_data'].append({
                    'name': prompt_name,
                    'total': total,
                    'successful': successful,
                    'accuracy': accuracy
                })
            
            field_summaries[field] = field_summary
        
        # Add summary tables to HTML
        html_content.append("<h3>Field Extraction Summary</h3>")
        html_content.append("<table>")
        html_content.append("<tr><th>Field</th><th>Total Samples</th><th>Successful</th><th>Accuracy</th><th>Best Prompt</th></tr>")
        
        for field, summary in field_summaries.items():
            overall_accuracy = summary['successful_extractions'] / summary['total_samples'] if summary['total_samples'] > 0 else 0
            html_content.append(
                f"<tr>"
                f"<td>{field}</td>"
                f"<td>{summary['total_samples']}</td>"
                f"<td>{summary['successful_extractions']}</td>"
                f"<td>{overall_accuracy:.2%}</td>"
                f"<td>{summary['best_prompt']['name']} ({summary['best_prompt']['accuracy']:.2%})</td>"
                f"</tr>"
            )
        
        html_content.append("</table>")
        html_content.append("</div>")  # Close summary div
        
        # Add charts
        html_content.append("<h2>Visualizations</h2>")
        
        # Accuracy chart
        html_content.append("<div class='chart-container'>")
        html_content.append("<canvas id='accuracyChart'></canvas>")
        html_content.append("</div>")
        
        # Create detailed sections for each field
        for field, summary in field_summaries.items():
            html_content.append(f"<h2>Field: {field}</h2>")
            
            # Prompt comparison table
            html_content.append("<h3>Prompt Performance</h3>")
            html_content.append("<table>")
            html_content.append("<tr><th>Prompt</th><th>Total Samples</th><th>Successful</th><th>Accuracy</th></tr>")
            
            for prompt_data in summary['prompt_data']:
                html_content.append(
                    f"<tr>"
                    f"<td>{prompt_data['name']}</td>"
                    f"<td>{prompt_data['total']}</td>"
                    f"<td>{prompt_data['successful']}</td>"
                    f"<td>{prompt_data['accuracy']:.2%}</td>"
                    f"</tr>"
                )
            
            html_content.append("</table>")
        
        # Add JavaScript for charts
        html_content.append("<script>")
        
        # Set up chart data
        html_content.append("document.addEventListener('DOMContentLoaded', function() {")
        
        # Accuracy chart data
        labels = []
        datasets = []
        
        for field, summary in field_summaries.items():
            prompt_names = [p['name'] for p in summary['prompt_data']]
            accuracies = [p['accuracy'] * 100 for p in summary['prompt_data']]
            
            labels = prompt_names if not labels else labels
            datasets.append({
                'label': field,
                'data': accuracies
            })
        
        # Format chart data as JavaScript
        html_content.append("  const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');")
        html_content.append("  const accuracyChart = new Chart(accuracyCtx, {")
        html_content.append("    type: 'bar',")
        html_content.append("    data: {")
        html_content.append(f"      labels: {json.dumps(labels)},")
        html_content.append("      datasets: [")
        
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
        for i, dataset in enumerate(datasets):
            color_idx = i % len(colors)
            html_content.append("        {")
            html_content.append(f"          label: '{dataset['label']}',")
            html_content.append(f"          data: {dataset['data']},")
            html_content.append(f"          backgroundColor: '{colors[color_idx]}',")
            html_content.append("          borderWidth: 1")
            html_content.append("        },")
        
        html_content.append("      ]")
        html_content.append("    },")
        html_content.append("    options: {")
        html_content.append("      scales: {")
        html_content.append("        y: {")
        html_content.append("          beginAtZero: true,")
        html_content.append("          title: {")
        html_content.append("            display: true,")
        html_content.append("            text: 'Accuracy (%)'")
        html_content.append("          }")
        html_content.append("        },")
        html_content.append("        x: {")
        html_content.append("          title: {")
        html_content.append("            display: true,")
        html_content.append("            text: 'Prompt'")
        html_content.append("          }")
        html_content.append("        }")
        html_content.append("      },")
        html_content.append("      plugins: {")
        html_content.append("        title: {")
        html_content.append("          display: true,")
        html_content.append("          text: 'Extraction Accuracy by Prompt and Field'")
        html_content.append("        }")
        html_content.append("      }")
        html_content.append("    }")
        html_content.append("  });")
        
        html_content.append("});")
        html_content.append("</script>")
        
        # Close HTML
        html_content.append("</body>")
        html_content.append("</html>")
        
        # Write HTML file
        with open(export_path, 'w') as f:
            f.write('\n'.join(html_content))
        
        return export_path

class AnalysisStage(BasePipelineStage):
    """
    Perform comprehensive results analysis.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the analysis stage."""
        super().__init__(name or "Analysis")
        
        # Define stage requirements
        self.requirements = StageRequirements(
            required_inputs={"processed_results"},
            optional_inputs={"results_collector"},
            provided_outputs={"analysis_results"}
        )
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate results before analysis.
        
        Checks:
        - Required processed results exist
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Raises:
            PipelineStageError: If validation fails
        """
        errors = self.requirements.validate_inputs(previous_results)
        if errors:
            raise PipelineStageError(
                f"Missing required inputs: {', '.join(errors)}",
                stage_name=self.name
            )
        
        # Validate processed results
        if 'processed_results' not in previous_results or 'resultscollection' not in previous_results:
            raise PipelineStageError(
                "No processed results found",
                stage_name=self.name
            )
        
        processed_results = previous_results['resultscollection']['processed_results']
        if not processed_results:
            raise PipelineStageError(
                "Empty processed results",
                stage_name=self.name
            )
        
        logger.info("Analysis stage inputs validated successfully")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze extraction results.
        
        Performs:
        - Trend identification
        - Correlation analysis
        - Error pattern detection
        - Performance insights generation
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Returns:
            Dictionary with analysis results
            
        Raises:
            PipelineStageError: If analysis fails
        """
        try:
            # Get processed results
            processed_results = previous_results['resultscollection']['processed_results']
            
            # Perform cross-field analysis
            cross_field_analysis = self._analyze_cross_field_performance(processed_results)
            
            # Perform prompt analysis
            prompt_analysis = self._analyze_prompt_effectiveness(processed_results)
            
            # Perform error analysis
            error_analysis = self._analyze_error_patterns(processed_results)
            
            # Generate insights
            insights = self._generate_performance_insights(
                cross_field_analysis,
                prompt_analysis,
                error_analysis
            )
            
            # Combine all analyses
            analysis_results = {
                "cross_field_analysis": cross_field_analysis,
                "prompt_analysis": prompt_analysis,
                "error_analysis": error_analysis,
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Analysis completed successfully")
            return {"analysis_results": analysis_results}
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise PipelineStageError(
                f"Analysis failed: {str(e)}",
                stage_name=self.name,
                original_error=e
            )
    
    def _analyze_cross_field_performance(
        self,
        processed_results: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze performance across different fields.
        
        Args:
            processed_results: Processed results by field and prompt
            
        Returns:
            Dictionary with cross-field analysis
        """
        analysis = {
            "field_performance": {},
            "best_field": None,
            "worst_field": None,
            "average_performance": 0
        }
        
        # Calculate metrics for each field
        field_rates = {}
        total_rate = 0
        
        for field, field_data in processed_results.items():
            # Find best prompt for this field
            best_prompt = None
            best_rate = -1
            
            for prompt_name, prompt_data in field_data.items():
                metrics = prompt_data.get('metrics', {})
                success_rate = metrics.get('success_rate', 0)
                
                if success_rate > best_rate:
                    best_rate = success_rate
                    best_prompt = prompt_name
            
            # Store field analysis
            field_rates[field] = best_rate
            total_rate += best_rate
            
            analysis["field_performance"][field] = {
                "best_prompt": best_prompt,
                "best_success_rate": best_rate
            }
        
        # Calculate average performance
        if field_rates:
            analysis["average_performance"] = total_rate / len(field_rates)
            
            # Find best and worst fields
            best_field = max(field_rates.items(), key=lambda x: x[1])
            worst_field = min(field_rates.items(), key=lambda x: x[1])
            
            analysis["best_field"] = {
                "name": best_field[0],
                "success_rate": best_field[1]
            }
            
            analysis["worst_field"] = {
                "name": worst_field[0],
                "success_rate": worst_field[1]
            }
        
        return analysis
    
    def _analyze_prompt_effectiveness(
        self,
        processed_results: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze effectiveness of different prompts.
        
        Args:
            processed_results: Processed results by field and prompt
            
        Returns:
            Dictionary with prompt effectiveness analysis
        """
        analysis = {
            "prompt_effectiveness": {},
            "best_prompts": {},
            "prompt_consistency": {}
        }
        
        # Track prompt performance across fields
        prompt_performance = {}
        
        for field, field_data in processed_results.items():
            for prompt_name, prompt_data in field_data.items():
                metrics = prompt_data.get('metrics', {})
                success_rate = metrics.get('success_rate', 0)
                
                if prompt_name not in prompt_performance:
                    prompt_performance[prompt_name] = []
                
                prompt_performance[prompt_name].append({
                    "field": field,
                    "success_rate": success_rate
                })
        
        # Calculate average effectiveness and consistency for each prompt
        for prompt_name, performances in prompt_performance.items():
            rates = [p["success_rate"] for p in performances]
            avg_rate = sum(rates) / len(rates) if rates else 0
            
            # Calculate consistency (standard deviation)
            import numpy as np
            consistency = 1.0 - float(np.std(rates)) if rates else 0
            
            analysis["prompt_effectiveness"][prompt_name] = {
                "average_success_rate": avg_rate,
                "field_count": len(performances),
                "consistency": consistency
            }
            
            # Track best prompt for each field
            for performance in performances:
                field = performance["field"]
                if field not in analysis["best_prompts"]:
                    analysis["best_prompts"][field] = {
                        "prompt": prompt_name,
                        "success_rate": performance["success_rate"]
                    }
                elif performance["success_rate"] > analysis["best_prompts"][field]["success_rate"]:
                    analysis["best_prompts"][field] = {
                        "prompt": prompt_name,
                        "success_rate": performance["success_rate"]
                    }
        
        # Calculate prompt consistency across fields
        prompt_consistency = {}
        for prompt_name, effectiveness in analysis["prompt_effectiveness"].items():
            prompt_consistency[prompt_name] = effectiveness["consistency"]
        
        analysis["prompt_consistency"] = prompt_consistency
        
        return analysis
    
    def _analyze_error_patterns(
        self,
        processed_results: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze error patterns in extraction results.
        
        Args:
            processed_results: Processed results by field and prompt
            
        Returns:
            Dictionary with error pattern analysis
        """
        analysis = {
            "common_errors": {},
            "field_specific_errors": {},
            "prompt_specific_errors": {},
            "error_rate": 0
        }
        
        # Track errors
        all_errors = []
        field_errors = {}
        prompt_errors = {}
        total_extractions = 0
        error_count = 0
        
        for field, field_data in processed_results.items():
            if field not in field_errors:
                field_errors[field] = []
                
            for prompt_name, prompt_data in field_data.items():
                if prompt_name not in prompt_errors:
                    prompt_errors[prompt_name] = []
                
                results = prompt_data.get('results', [])
                total_extractions += len(results)
                
                for result in results:
                    if 'error' in result:
                        error_count += 1
                        
                        # Extract error details
                        error_info = {
                            "field": field,
                            "prompt": prompt_name,
                            "error": result['error'],
                            "image_id": result.get('image_id', 'unknown')
                        }
                        
                        all_errors.append(error_info)
                        field_errors[field].append(error_info)
                        prompt_errors[prompt_name].append(error_info)
        
        # Calculate overall error rate
        analysis["error_rate"] = error_count / total_extractions if total_extractions > 0 else 0
        
        # Analyze common errors
        error_types = {}
        for error in all_errors:
            error_msg = error['error']
            if error_msg not in error_types:
                error_types[error_msg] = 0
            error_types[error_msg] += 1
        
        # Get top error types
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        for error_msg, count in sorted_errors[:5]:  # Top 5 errors
            analysis["common_errors"][error_msg] = count
        
        # Analyze field-specific errors
        for field, errors in field_errors.items():
            if not errors:
                continue
                
            field_error_rate = len(errors) / sum(
                len(prompt_data.get('results', [])) 
                for prompt_data in processed_results[field].values()
            )
            
            field_error_types = {}
            for error in errors:
                error_msg = error['error']
                if error_msg not in field_error_types:
                    field_error_types[error_msg] = 0
                field_error_types[error_msg] += 1
            
            # Get top error for this field
            if field_error_types:
                top_error = max(field_error_types.items(), key=lambda x: x[1])
                analysis["field_specific_errors"][field] = {
                    "error_rate": field_error_rate,
                    "top_error": top_error[0],
                    "top_error_count": top_error[1]
                }
        
        # Analyze prompt-specific errors
        for prompt, errors in prompt_errors.items():
            if not errors:
                continue
                
            # Count how many results use this prompt
            prompt_result_count = 0
            for field_data in processed_results.values():
                if prompt in field_data:
                    prompt_result_count += len(field_data[prompt].get('results', []))
            
            prompt_error_rate = len(errors) / prompt_result_count if prompt_result_count > 0 else 0
            
            prompt_error_types = {}
            for error in errors:
                error_msg = error['error']
                if error_msg not in prompt_error_types:
                    prompt_error_types[error_msg] = 0
                prompt_error_types[error_msg] += 1
            
            # Get top error for this prompt
            if prompt_error_types:
                top_error = max(prompt_error_types.items(), key=lambda x: x[1])
                analysis["prompt_specific_errors"][prompt] = {
                    "error_rate": prompt_error_rate,
                    "top_error": top_error[0],
                    "top_error_count": top_error[1]
                }
        
        return analysis
    
    def _generate_performance_insights(
        self,
        cross_field_analysis: Dict[str, Any],
        prompt_analysis: Dict[str, Any],
        error_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate insights based on analysis results.
        
        Args:
            cross_field_analysis: Cross-field performance analysis
            prompt_analysis: Prompt effectiveness analysis
            error_analysis: Error pattern analysis
            
        Returns:
            List of performance insights
        """
        insights = []
        
        # Generate field-based insights
        if "best_field" in cross_field_analysis and cross_field_analysis["best_field"]:
            best_field = cross_field_analysis["best_field"]["name"]
            best_rate = cross_field_analysis["best_field"]["success_rate"]
            
            insights.append({
                "type": "field_performance",
                "insight": f"Field '{best_field}' has the highest extraction accuracy at {best_rate:.2%}",
                "recommendation": "Consider using this field's prompts as templates for other fields",
                "priority": "medium"
            })
        
        if "worst_field" in cross_field_analysis and cross_field_analysis["worst_field"]:
            worst_field = cross_field_analysis["worst_field"]["name"]
            worst_rate = cross_field_analysis["worst_field"]["success_rate"]
            
            insights.append({
                "type": "field_performance",
                "insight": f"Field '{worst_field}' has the lowest extraction accuracy at {worst_rate:.2%}",
                "recommendation": "Consider revising prompts or using more detailed instructions",
                "priority": "high"
            })
        
        # Generate prompt-based insights
        best_prompt = None
        best_prompt_rate = -1
        
        for prompt, data in prompt_analysis.get("prompt_effectiveness", {}).items():
            if data["average_success_rate"] > best_prompt_rate:
                best_prompt_rate = data["average_success_rate"]
                best_prompt = prompt
        
        if best_prompt:
            insights.append({
                "type": "prompt_effectiveness",
                "insight": f"Prompt '{best_prompt}' has the highest average success rate at {best_prompt_rate:.2%}",
                "recommendation": "Consider using this prompt approach for other fields",
                "priority": "medium"
            })
        
        # Generate consistency insights
        consistency_insights = []
        for prompt, consistency in prompt_analysis.get("prompt_consistency", {}).items():
            if consistency > 0.9:  # High consistency threshold
                consistency_insights.append((prompt, consistency))
        
        if consistency_insights:
            top_consistent = max(consistency_insights, key=lambda x: x[1])
            insights.append({
                "type": "prompt_consistency",
                "insight": f"Prompt '{top_consistent[0]}' has the most consistent performance across fields",
                "recommendation": "This prompt works reliably across different contexts",
                "priority": "medium"
            })
        
        # Generate error-based insights
        if error_analysis.get("error_rate", 0) > 0.3:  # High error rate threshold
            insights.append({
                "type": "error_rate",
                "insight": f"Overall error rate is high at {error_analysis['error_rate']:.2%}",
                "recommendation": "Review error patterns and consider model or prompt adjustments",
                "priority": "high"
            })
        
        if error_analysis.get("common_errors"):
            top_error = next(iter(error_analysis["common_errors"].items()))
            insights.append({
                "type": "common_error",
                "insight": f"Most common error: '{top_error[0]}' occurred {top_error[1]} times",
                "recommendation": "Address this specific error pattern to improve overall performance",
                "priority": "high"
            })
        
        return insights
    
    def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        logger.debug(f"Cleaning up resources for {self.name}")
        gc.collect()

class ValidationStage(BasePipelineStage):
    """
    Validate extraction results for quality, consistency, and correctness.
    
    This stage performs:
    - Result format validation
    - Consistency checks across different prompts
    - Outlier detection
    - Pattern-based validation
    - Cross-field validation
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the validation stage."""
        super().__init__(name or "Validation")
        
        # Define stage requirements
        self.requirements = StageRequirements(
            required_inputs={"analysis_results", "processed_results"},
            optional_inputs={"field_extractions"},
            provided_outputs={"validation_results", "quality_metrics"}
        )
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate inputs for validation stage.
        
        Checks:
        - Required analysis results exist
        - Required processed results exist
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Raises:
            PipelineStageError: If validation fails
        """
        errors = self.requirements.validate_inputs(previous_results)
        if errors:
            raise PipelineStageError(
                f"Missing required inputs: {', '.join(errors)}",
                stage_name=self.name
            )
        
        # Check for analysis results
        if 'analysis' not in previous_results or 'analysis_results' not in previous_results['analysis']:
            raise PipelineStageError(
                "No analysis results found",
                stage_name=self.name
            )
        
        # Check for processed results
        if 'resultscollection' not in previous_results or 'processed_results' not in previous_results['resultscollection']:
            raise PipelineStageError(
                "No processed results found",
                stage_name=self.name
            )
        
        logger.info("Validation stage inputs verified successfully")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate extraction results through multiple validation approaches.
        
        Performs:
        - Format validation
        - Consistency validation
        - Pattern-based validation
        - Cross-field validation
        - Reference validation
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Returns:
            Dictionary with validation results and quality metrics
            
        Raises:
            PipelineStageError: If validation fails
        """
        try:
            # Get analysis results
            analysis_results = previous_results['analysis']['analysis_results']
            
            # Get processed results
            processed_results = previous_results['resultscollection']['processed_results']
            
            # Get validation configuration
            validation_config = config.get('validation', {})
            
            # Determine validation methods to apply
            validation_methods = validation_config.get('methods', [
                'format', 'consistency', 'pattern', 'cross_field', 'reference'
            ])
            
            logger.info(f"Applying validation methods: {', '.join(validation_methods)}")
            
            # Initialize validation results
            validation_results = {
                "field_validation": {},
                "prompt_validation": {},
                "overall_validation": {
                    "passed": True,
                    "issues_found": 0,
                    "warnings": []
                },
                "quality_metrics": {}
            }
            
            # Validate each field
            fields_to_validate = list(processed_results.keys())
            logger.info(f"Validating {len(fields_to_validate)} fields")
            
            for field in fields_to_validate:
                logger.info(f"Validating field: {field}")
                
                field_results = processed_results[field]
                field_validation = {}
                
                # Validate each prompt for this field
                for prompt_name, prompt_data in field_results.items():
                    logger.info(f"Validating results for prompt: {prompt_name}")
                    
                    prompt_results = prompt_data.get('results', [])
                    prompt_validation = {}
                    
                    # Apply selected validation methods
                    if 'format' in validation_methods:
                        format_validation = self._validate_format(
                            prompt_results, 
                            field, 
                            validation_config
                        )
                        prompt_validation["format"] = format_validation
                    
                    if 'pattern' in validation_methods:
                        pattern_validation = self._validate_patterns(
                            prompt_results, 
                            field, 
                            validation_config
                        )
                        prompt_validation["pattern"] = pattern_validation
                    
                    if 'reference' in validation_methods:
                        reference_validation = self._validate_against_reference(
                            prompt_results, 
                            field, 
                            validation_config
                        )
                        prompt_validation["reference"] = reference_validation
                    
                    # Store prompt validation results
                    field_validation[prompt_name] = prompt_validation
                    
                    # Add to prompt-level validation results
                    validation_results["prompt_validation"][f"{field}_{prompt_name}"] = {
                        "field": field,
                        "prompt": prompt_name,
                        "validation_summary": self._summarize_prompt_validation(prompt_validation)
                    }
                
                # Perform consistency validation across prompts
                if 'consistency' in validation_methods and len(field_results) > 1:
                    consistency_validation = self._validate_consistency(
                        field_results, 
                        field, 
                        validation_config
                    )
                    field_validation["consistency"] = consistency_validation
                
                # Store field validation results
                validation_results["field_validation"][field] = {
                    "prompt_validations": field_validation,
                    "validation_summary": self._summarize_field_validation(field_validation)
                }
            
            # Perform cross-field validation if applicable
            if 'cross_field' in validation_methods and len(fields_to_validate) > 1:
                cross_field_validation = self._validate_cross_fields(
                    processed_results, 
                    validation_config
                )
                validation_results["cross_field_validation"] = cross_field_validation
            
            # Calculate overall quality metrics
            quality_metrics = self._calculate_quality_metrics(
                validation_results, 
                processed_results,
                config
            )
            validation_results["quality_metrics"] = quality_metrics
            
            # Update overall validation summary
            validation_results["overall_validation"] = self._summarize_overall_validation(
                validation_results
            )
            
            # Log validation summary
            self._log_validation_summary(validation_results)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise PipelineStageError(
                f"Validation failed: {str(e)}",
                stage_name=self.name,
                original_error=e
            )
    
    def _validate_format(
        self, 
        results: List[Dict[str, Any]], 
        field: str,
        validation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate format of extraction results.
        
        Checks:
        - Required fields are present
        - Field values match expected formats
        - No unexpected data types
        
        Args:
            results: List of extraction results
            field: Field being validated
            validation_config: Validation configuration
            
        Returns:
            Dictionary with format validation results
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "issue_count": 0,
            "valid_count": 0,
            "invalid_count": 0
        }
        
        # Get field-specific format validation config
        field_format = validation_config.get('format', {}).get(field, {})
        
        # Default format validation if not specified
        if not field_format:
            # Use sensible defaults based on field type
            if field == 'work_order':
                field_format = {
                    "pattern": r"^\d+$",
                    "expected_type": "numeric",
                    "min_length": 3,
                    "max_length": 10
                }
            elif field == 'cost':
                field_format = {
                    "pattern": r"^\$?\s*[\d,]+(\.\d{2})?$",
                    "expected_type": "currency",
                    "allow_symbols": ["$", ",", "."]
                }
            elif field == 'date':
                field_format = {
                    "pattern": r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$",
                    "expected_type": "date"
                }
            else:
                # Generic defaults
                field_format = {
                    "min_length": 1,
                    "expected_type": "string"
                }
        
        for i, result in enumerate(results):
            # Skip results with errors
            if 'error' in result:
                continue
                
            extracted_value = result.get('processed_extraction') or result.get('extraction') or ''
            
            # Validate individual result
            issues = []
            valid = True
            
            # Check for empty values
            if not extracted_value and not field_format.get('allow_empty', False):
                issues.append("Empty extraction value")
                valid = False
            
            # Check length constraints
            if extracted_value:
                if 'min_length' in field_format and len(str(extracted_value)) < field_format['min_length']:
                    issues.append(
                        f"Value too short: {len(str(extracted_value))} chars, "
                        f"minimum {field_format['min_length']}"
                    )
                    valid = False
                    
                if 'max_length' in field_format and len(str(extracted_value)) > field_format['max_length']:
                    issues.append(
                        f"Value too long: {len(str(extracted_value))} chars, "
                        f"maximum {field_format['max_length']}"
                    )
                    valid = False
            
            # Check pattern if specified
            if extracted_value and 'pattern' in field_format:
                import re
                pattern = field_format['pattern']
                if not re.match(pattern, str(extracted_value)):
                    issues.append(f"Value doesn't match expected pattern: {pattern}")
                    valid = False
            
            # Check expected type
            if extracted_value and 'expected_type' in field_format:
                expected_type = field_format['expected_type']
                
                if expected_type == 'numeric':
                    # Check if value is numeric (after removing allowed symbols)
                    clean_value = str(extracted_value)
                    for symbol in field_format.get('allow_symbols', []):
                        clean_value = clean_value.replace(symbol, '')
                    
                    if not clean_value.isdigit():
                        issues.append(f"Expected numeric value, got: {extracted_value}")
                        valid = False
                
                elif expected_type == 'currency':
                    # Check if value looks like currency
                    import re
                    currency_pattern = r'^\$?\s*[\d,]+(\.\d{1,2})?$'
                    if not re.match(currency_pattern, str(extracted_value)):
                        issues.append(f"Expected currency value, got: {extracted_value}")
                        valid = False
                
                elif expected_type == 'date':
                    # Check if value looks like a date
                    import re
                    date_patterns = [
                        r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # MM/DD/YYYY, DD/MM/YYYY
                        r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$',    # YYYY/MM/DD
                        r'^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$'  # Month DD, YYYY
                    ]
                    if not any(re.match(pattern, str(extracted_value)) for pattern in date_patterns):
                        issues.append(f"Expected date value, got: {extracted_value}")
                        valid = False
            
            # Update result tracking
            if valid:
                validation_result["valid_count"] += 1
            else:
                validation_result["invalid_count"] += 1
                validation_result["issues"].append({
                    "index": i,
                    "image_id": result.get('image_id', f"result_{i}"),
                    "value": extracted_value,
                    "issues": issues
                })
                validation_result["issue_count"] += len(issues)
                validation_result["valid"] = False
                
        return validation_result
    
    def _validate_patterns(
        self, 
        results: List[Dict[str, Any]], 
        field: str,
        validation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate extracted values against expected patterns.
        
        Checks:
        - Values match field-specific patterns
        - Values follow expected conventions
        - Outlier detection for unusual values
        
        Args:
            results: List of extraction results
            field: Field being validated
            validation_config: Validation configuration
            
        Returns:
            Dictionary with pattern validation results
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "issue_count": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "patterns_detected": {}
        }
        
        # Get field-specific pattern validation config
        field_patterns = validation_config.get('patterns', {}).get(field, {})
        
        # Default pattern validation if not specified
        if not field_patterns:
            # Use sensible defaults based on field type
            if field == 'work_order':
                field_patterns = {
                    "expected_patterns": [r"^\d+$"],
                    "forbidden_patterns": [r"[A-Za-z]"],
                    "outlier_detection": True
                }
            elif field == 'cost':
                field_patterns = {
                    "expected_patterns": [r"^\$?\s*[\d,]+(\.\d{2})?$"],
                    "forbidden_patterns": [r"[A-Za-z]"],
                    "outlier_detection": True
                }
            elif field == 'date':
                field_patterns = {
                    "expected_patterns": [
                        r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$",
                        r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$"
                    ],
                    "outlier_detection": False
                }
        
        # Extract all values for analysis
        values = []
        for result in results:
            # Skip results with errors
            if 'error' in result:
                continue
                
            extracted_value = result.get('processed_extraction') or result.get('extraction') or ''
            if extracted_value:
                values.append(str(extracted_value))
        
        # Detect common patterns in the data
        pattern_counts = {}
        for value in values:
            # Try to categorize the pattern
            pattern = self._categorize_pattern(value, field)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Sort patterns by frequency
        sorted_patterns = sorted(
            pattern_counts.items(), 
            key=lambda x: x[1],
            reverse=True
        )
        
        # Store detected patterns
        validation_result["patterns_detected"] = {
            pattern: count for pattern, count in sorted_patterns
        }
        
        # Validate each result against patterns
        for i, result in enumerate(results):
            # Skip results with errors
            if 'error' in result:
                continue
                
            extracted_value = result.get('processed_extraction') or result.get('extraction') or ''
            if not extracted_value:
                continue
                
            # Validate against patterns
            issues = []
            valid = True
            
            # Check expected patterns
            if field_patterns.get('expected_patterns'):
                import re
                matches_any = False
                for pattern in field_patterns['expected_patterns']:
                    if re.match(pattern, str(extracted_value)):
                        matches_any = True
                        break
                
                if not matches_any:
                    issues.append(
                        f"Value doesn't match any expected pattern: {extracted_value}"
                    )
                    valid = False
            
            # Check forbidden patterns
            if field_patterns.get('forbidden_patterns'):
                import re
                for pattern in field_patterns['forbidden_patterns']:
                    if re.search(pattern, str(extracted_value)):
                        issues.append(
                            f"Value contains forbidden pattern '{pattern}': {extracted_value}"
                        )
                        valid = False
                        break
            
            # Update result tracking
            if valid:
                validation_result["valid_count"] += 1
            else:
                validation_result["invalid_count"] += 1
                validation_result["issues"].append({
                    "index": i,
                    "image_id": result.get('image_id', f"result_{i}"),
                    "value": extracted_value,
                    "issues": issues
                })
                validation_result["issue_count"] += len(issues)
                validation_result["valid"] = False
        
        # Perform outlier detection if enabled
        if field_patterns.get('outlier_detection', False) and len(values) >= 5:
            try:
                outliers = self._detect_outliers(values, field)
                
                for outlier in outliers:
                    # Find corresponding result
                    for i, result in enumerate(results):
                        extracted_value = result.get('processed_extraction') or result.get('extraction') or ''
                        if str(extracted_value) == outlier['value']:
                            # Add to issues
                            validation_result["issues"].append({
                                "index": i,
                                "image_id": result.get('image_id', f"result_{i}"),
                                "value": outlier['value'],
                                "issues": [f"Potential outlier: {outlier['reason']}"],
                                "is_outlier": True
                            })
                            validation_result["issue_count"] += 1
                            
                            # Don't mark as invalid since outliers may be correct
                            break
            except Exception as e:
                logger.warning(f"Error in outlier detection: {e}")
        
        return validation_result
    
    def _validate_consistency(
        self, 
        field_results: Dict[str, Dict[str, Any]], 
        field: str,
        validation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate consistency of results across different prompts.
        
        Checks:
        - Agreement between different prompts
        - Consensus detection
        - Confidence scoring based on agreement
        
        Args:
            field_results: Results for all prompts in a field
            field: Field being validated
            validation_config: Validation configuration
            
        Returns:
            Dictionary with consistency validation results
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "issue_count": 0,
            "agreement_rate": 0.0,
            "agreement_by_image": {},
            "consensus_values": {}
        }
        
        # Get field-specific consistency validation config
        consistency_config = validation_config.get('consistency', {})
        
        # Get minimum agreement threshold
        agreement_threshold = consistency_config.get('agreement_threshold', 0.5)
        
        # Extract results by image ID
        results_by_image = {}
        
        for prompt_name, prompt_data in field_results.items():
            prompt_results = prompt_data.get('results', [])
            
            for result in prompt_results:
                # Skip results with errors
                if 'error' in result:
                    continue
                    
                image_id = result.get('image_id')
                if not image_id:
                    continue
                    
                extracted_value = result.get('processed_extraction') or result.get('extraction') or ''
                ground_truth = result.get('ground_truth')
                
                if image_id not in results_by_image:
                    results_by_image[image_id] = {
                        'values': [],
                        'prompts': [],
                        'ground_truth': ground_truth
                    }
                
                results_by_image[image_id]['values'].append(extracted_value)
                results_by_image[image_id]['prompts'].append(prompt_name)
        
        # Calculate agreement for each image
        total_agreement = 0.0
        image_count = 0
        
        for image_id, image_data in results_by_image.items():
            values = image_data['values']
            
            # Skip images with fewer than 2 values
            if len(values) < 2:
                continue
                
            # Count occurrences of each value
            value_counts = {}
            for value in values:
                if value:  # Only count non-empty values
                    value_counts[value] = value_counts.get(value, 0) + 1
            
            # Find the most common value (consensus)
            if value_counts:
                consensus_value, consensus_count = max(
                    value_counts.items(), 
                    key=lambda x: x[1]
                )
                
                # Calculate agreement rate
                agreement_rate = consensus_count / len(values)
                
                # Store consensus information
                validation_result["consensus_values"][image_id] = {
                    "value": consensus_value,
                    "count": consensus_count,
                    "total": len(values),
                    "agreement_rate": agreement_rate
                }
                
                # Check if agreement is below threshold
                if agreement_rate < agreement_threshold:
                    # Get the disagreeing values
                    disagreeing_values = [v for v in values if v != consensus_value]
                    
                    validation_result["issues"].append({
                        "image_id": image_id,
                        "consensus_value": consensus_value,
                        "agreement_rate": agreement_rate,
                        "disagreeing_values": disagreeing_values,
                        "issue": f"Low agreement rate: {agreement_rate:.2%} < {agreement_threshold:.2%}"
                    })
                    validation_result["issue_count"] += 1
                    validation_result["valid"] = False
                
                # Store agreement rate for this image
                validation_result["agreement_by_image"][image_id] = agreement_rate
                
                # Add to total for average calculation
                total_agreement += agreement_rate
                image_count += 1
        
        # Calculate average agreement rate
        if image_count > 0:
            validation_result["agreement_rate"] = total_agreement / image_count
        
        return validation_result
    
    def _validate_against_reference(
        self, 
        results: List[Dict[str, Any]], 
        field: str,
        validation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate extraction results against reference data (ground truth).
        
        Checks:
        - Match with ground truth values
        - Distance measures for partial matches
        - Analysis of error patterns
        
        Args:
            results: List of extraction results
            field: Field being validated
            validation_config: Validation configuration
            
        Returns:
            Dictionary with reference validation results
        """
        validation_result = {
            "valid": True,
            "exact_match_rate": 0.0,
            "partial_match_rate": 0.0,
            "error_patterns": {},
            "issues": [],
            "issue_count": 0
        }
        
        # Get reference validation config
        reference_config = validation_config.get('reference', {})
        
        # Get matching thresholds
        partial_match_threshold = reference_config.get('partial_match_threshold', 0.8)
        
        # Track matches
        exact_matches = 0
        partial_matches = 0
        total_with_ground_truth = 0
        error_types = {}
        
        for i, result in enumerate(results):
            # Skip results with errors
            if 'error' in result:
                continue
                
            # Get values
            extracted_value = result.get('processed_extraction') or result.get('extraction') or ''
            ground_truth = result.get('ground_truth')
            
            # Skip items without ground truth
            if not ground_truth:
                continue
                
            total_with_ground_truth += 1
            
            # Check for exact match
            exact_match = (
                str(extracted_value).strip().lower() == 
                str(ground_truth).strip().lower()
            )
            
            if exact_match:
                exact_matches += 1
                continue
            
            # Calculate similarity for partial match
            similarity = self._calculate_similarity(
                str(extracted_value), 
                str(ground_truth)
            )
            
            if similarity >= partial_match_threshold:
                partial_matches += 1
            else:
                # Analyze error pattern
                error_pattern = self._analyze_error_pattern(
                    str(extracted_value), 
                    str(ground_truth),
                    field
                )
                
                error_types[error_pattern] = error_types.get(error_pattern, 0) + 1
                
                # Add to issues
                validation_result["issues"].append({
                    "index": i,
                    "image_id": result.get('image_id', f"result_{i}"),
                    "extracted": extracted_value,
                    "ground_truth": ground_truth,
                    "similarity": similarity,
                    "error_pattern": error_pattern
                })
                validation_result["issue_count"] += 1
        
        # Calculate match rates
        if total_with_ground_truth > 0:
            validation_result["exact_match_rate"] = exact_matches / total_with_ground_truth
            validation_result["partial_match_rate"] = (exact_matches + partial_matches) / total_with_ground_truth
        
        # Store error patterns
        validation_result["error_patterns"] = error_types
        
        # Determine overall validity
        validity_threshold = reference_config.get('validity_threshold', 0.7)
        validation_result["valid"] = (validation_result["exact_match_rate"] >= validity_threshold)
        
        return validation_result
    
    def _validate_cross_fields(
        self, 
        processed_results: Dict[str, Dict[str, Dict[str, Any]]],
        validation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate relationships between different fields.
        
        Checks:
        - Logical relationships between fields
        - Conditional dependencies
        - Consistency across related fields
        
        Args:
            processed_results: Results for all fields
            validation_config: Validation configuration
            
        Returns:
            Dictionary with cross-field validation results
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "issue_count": 0,
            "field_relationships": {}
        }
        
        # Get cross-field validation config
        cross_field_config = validation_config.get('cross_field', {})
        
        # Get field relationships
        field_relationships = cross_field_config.get('relationships', [])
        
        # Skip if no relationships defined
        if not field_relationships:
            validation_result["message"] = "No field relationships defined for validation"
            return validation_result
        
        # Validate each relationship
        for relationship in field_relationships:
            source_field = relationship.get('source_field')
            target_field = relationship.get('target_field')
            relationship_type = relationship.get('type')
            
            # Skip if fields are not in results
            if (source_field not in processed_results or 
                target_field not in processed_results):
                continue
            
            # Get source field results (use the best prompt's results)
            source_field_data = processed_results[source_field]
            source_prompt = self._get_best_prompt(source_field_data)
            
            if not source_prompt:
                continue
                
            source_results = source_field_data[source_prompt].get('results', [])
            
            # Get target field results
            target_field_data = processed_results[target_field]
            target_prompt = self._get_best_prompt(target_field_data)
            
            if not target_prompt:
                continue
                
            target_results = target_field_data[target_prompt].get('results', [])
            
            # Group results by image ID for comparison
            source_by_image = {
                r.get('image_id'): r for r in source_results 
                if 'error' not in r and r.get('image_id')
            }
            
            target_by_image = {
                r.get('image_id'): r for r in target_results 
                if 'error' not in r and r.get('image_id')
            }
            
            # Find common image IDs
            common_images = set(source_by_image.keys()) & set(target_by_image.keys())
            
            # Validate relationship for each common image
            relationship_issues = []
            
            for image_id in common_images:
                source_value = source_by_image[image_id].get('processed_extraction') or source_by_image[image_id].get('extraction')
                target_value = target_by_image[image_id].get('processed_extraction') or target_by_image[image_id].get('extraction')
                
                if not source_value or not target_value:
                    continue
                
                # Validate based on relationship type
                valid_relationship = True
                issue = None
                
                if relationship_type == 'greater_than':
                    # Numeric comparison
                    try:
                        source_num = float(str(source_value).replace('$', '').replace(',', ''))
                        target_num = float(str(target_value).replace('$', '').replace(',', ''))
                        
                        if not source_num > target_num:
                            valid_relationship = False
                            issue = f"{source_field} ({source_value}) should be greater than {target_field} ({target_value})"
                    except ValueError:
                        # Not numeric, skip
                        continue
                
                elif relationship_type == 'less_than':
                    # Numeric comparison
                    try:
                        source_num = float(str(source_value).replace('$', '').replace(',', ''))
                        target_num = float(str(target_value).replace('$', '').replace(',', ''))
                        
                        if not source_num < target_num:
                            valid_relationship = False
                            issue = f"{source_field} ({source_value}) should be less than {target_field} ({target_value})"
                    except ValueError:
                        # Not numeric, skip
                        continue
                
                elif relationship_type == 'contains':
                    # String containment
                    if str(target_value) not in str(source_value):
                        valid_relationship = False
                        issue = f"{source_field} ({source_value}) should contain {target_field} ({target_value})"
                
                elif relationship_type == 'date_before':
                    # Date comparison
                    try:
                        source_date = self._parse_date(source_value)
                        target_date = self._parse_date(target_value)
                        
                        if source_date and target_date and not source_date < target_date:
                            valid_relationship = False
                            issue = f"{source_field} ({source_value}) should be before {target_field} ({target_value})"
                    except Exception:
                        # Date parsing failed, skip
                        continue
                                
                elif relationship_type == 'date_after':
                    # Date comparison
                    try:
                        source_date = self._parse_date(source_value)
                        target_date = self._parse_date(target_value)
                        
                        if source_date and target_date and not source_date > target_date:
                            valid_relationship = False
                            issue = f"{source_field} ({source_value}) should be after {target_field} ({target_value})"
                    except Exception:
                        # Date parsing failed, skip
                        continue
                
                # Record issue if relationship is not valid
                if not valid_relationship and issue:
                    relationship_issues.append({
                        "image_id": image_id,
                        "source_field": source_field,
                        "source_value": source_value,
                        "target_field": target_field,
                        "target_value": target_value,
                        "issue": issue
                    })
            
            # Store relationship validation results
            relationship_key = f"{source_field}_{relationship_type}_{target_field}"
            validation_result["field_relationships"][relationship_key] = {
                "source_field": source_field,
                "target_field": target_field,
                "relationship_type": relationship_type,
                "issues": relationship_issues,
                "issue_count": len(relationship_issues),
                "valid": len(relationship_issues) == 0
            }
            
            # Update overall validation
            if relationship_issues:
                validation_result["issues"].extend(relationship_issues)
                validation_result["issue_count"] += len(relationship_issues)
                validation_result["valid"] = False
        
        return validation_result
    
    def _calculate_quality_metrics(
        self, 
        validation_results: Dict[str, Any],
        processed_results: Dict[str, Dict[str, Dict[str, Any]]],
        config: ExperimentConfiguration
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics based on validation results.
        
        Args:
            validation_results: Validation results
            processed_results: Processed extraction results
            config: Experiment configuration
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            "overall_quality_score": 0.0,
            "field_quality_scores": {},
            "prompt_quality_scores": {},
            "extraction_success_rate": 0.0,
            "validation_pass_rate": 0.0
        }
        
        # Calculate field quality scores
        field_validations = validation_results.get("field_validation", {})
        field_scores = {}
        
        for field, validation in field_validations.items():
            field_summary = validation.get("validation_summary", {})
            
            # Calculate quality score components
            format_score = field_summary.get("format_score", 0.0)
            pattern_score = field_summary.get("pattern_score", 0.0)
            consistency_score = field_summary.get("consistency_score", 0.0)
            reference_score = field_summary.get("reference_score", 0.0)
            
            # Calculate weighted quality score
            quality_score = (
                format_score * 0.2 +
                pattern_score * 0.2 +
                consistency_score * 0.2 +
                reference_score * 0.4  # Reference validation gets higher weight
            )
            
            field_scores[field] = quality_score
        
        # Calculate prompt quality scores
        prompt_validations = validation_results.get("prompt_validation", {})
        prompt_scores = {}
        
        for prompt_key, validation in prompt_validations.items():
            validation_summary = validation.get("validation_summary", {})
            prompt_scores[prompt_key] = validation_summary.get("quality_score", 0.0)
        
        # Calculate success rates
        total_extractions = 0
        successful_extractions = 0
        
        for field_data in processed_results.values():
            for prompt_data in field_data.values():
                results = prompt_data.get('results', [])
                
                total_extractions += len(results)
                successful_extractions += sum(
                    1 for r in results 
                    if 'error' not in r and r.get('exact_match', False)
                )
        
        if total_extractions > 0:
            quality_metrics["extraction_success_rate"] = successful_extractions / total_extractions
        
        # Calculate validation pass rate
        total_validations = len(field_validations) + len(prompt_validations)
        passed_validations = sum(
            1 for v in field_validations.values() 
            if v.get("validation_summary", {}).get("passed", False)
        )
        passed_validations += sum(
            1 for v in prompt_validations.values() 
            if v.get("validation_summary", {}).get("passed", False)
        )
        
        if total_validations > 0:
            quality_metrics["validation_pass_rate"] = passed_validations / total_validations
        
        # Store quality scores
        quality_metrics["field_quality_scores"] = field_scores
        quality_metrics["prompt_quality_scores"] = prompt_scores
        
        # Calculate overall quality score
        if field_scores:
            quality_metrics["overall_quality_score"] = sum(field_scores.values()) / len(field_scores)
        
        return quality_metrics
    
    def _summarize_prompt_validation(self, prompt_validation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize validation results for a prompt.
        
        Args:
            prompt_validation: Validation results for a prompt
            
        Returns:
            Dictionary with validation summary
        """
        summary = {
            "passed": True,
            "quality_score": 1.0,
            "issue_count": 0,
            "format_score": 1.0,
            "pattern_score": 1.0,
            "reference_score": 1.0
        }
        
        # Process format validation
        if format_validation := prompt_validation.get("format"):
            valid = format_validation.get("valid", True)
            issue_count = format_validation.get("issue_count", 0)
            
            if not valid:
                summary["passed"] = False
            
            summary["issue_count"] += issue_count
            
            # Calculate format score
            total_count = (
                format_validation.get("valid_count", 0) + 
                format_validation.get("invalid_count", 0)
            )
            
            if total_count > 0:
                format_score = format_validation.get("valid_count", 0) / total_count
                summary["format_score"] = format_score
        
        # Process pattern validation
        if pattern_validation := prompt_validation.get("pattern"):
            valid = pattern_validation.get("valid", True)
            issue_count = pattern_validation.get("issue_count", 0)
            
            if not valid:
                summary["passed"] = False
            
            summary["issue_count"] += issue_count
            
            # Calculate pattern score
            total_count = (
                pattern_validation.get("valid_count", 0) + 
                pattern_validation.get("invalid_count", 0)
            )
            
            if total_count > 0:
                pattern_score = pattern_validation.get("valid_count", 0) / total_count
                summary["pattern_score"] = pattern_score
        
        # Process reference validation
        if reference_validation := prompt_validation.get("reference"):
            valid = reference_validation.get("valid", True)
            issue_count = reference_validation.get("issue_count", 0)
            
            if not valid:
                summary["passed"] = False
            
            summary["issue_count"] += issue_count
            
            # Use exact match rate as reference score
            summary["reference_score"] = reference_validation.get("exact_match_rate", 0.0)
        
        # Calculate overall quality score
        summary["quality_score"] = (
            summary["format_score"] * 0.3 +
            summary["pattern_score"] * 0.3 +
            summary["reference_score"] * 0.4
        )
        
        return summary
    
    def _summarize_field_validation(self, field_validation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize validation results for a field.
        
        Args:
            field_validation: Validation results for a field
            
        Returns:
            Dictionary with validation summary
        """
        summary = {
            "passed": True,
            "quality_score": 1.0,
            "issue_count": 0,
            "format_score": 1.0,
            "pattern_score": 1.0,
            "consistency_score": 1.0,
            "reference_score": 1.0
        }
        
        # Process prompt validations
        prompt_validations = field_validation.get("prompt_validations", {})
        prompt_summaries = []
        
        for prompt_name, prompt_validation in prompt_validations.items():
            # Skip consistency validation (processed separately)
            if prompt_name == "consistency":
                continue
                
            # Get prompt summary
            prompt_summary = None
            
            if isinstance(prompt_validation, dict):
                prompt_summary = self._summarize_prompt_validation(prompt_validation)
            
            if prompt_summary:
                prompt_summaries.append(prompt_summary)
                
                # Update field summary
                if not prompt_summary.get("passed", True):
                    summary["passed"] = False
                
                summary["issue_count"] += prompt_summary.get("issue_count", 0)
        
        # Calculate average scores from prompt summaries
        if prompt_summaries:
            summary["format_score"] = sum(s.get("format_score", 1.0) for s in prompt_summaries) / len(prompt_summaries)
            summary["pattern_score"] = sum(s.get("pattern_score", 1.0) for s in prompt_summaries) / len(prompt_summaries)
            summary["reference_score"] = sum(s.get("reference_score", 1.0) for s in prompt_summaries) / len(prompt_summaries)
        
        # Process consistency validation
        if consistency_validation := prompt_validations.get("consistency"):
            valid = consistency_validation.get("valid", True)
            issue_count = consistency_validation.get("issue_count", 0)
            
            if not valid:
                summary["passed"] = False
            
            summary["issue_count"] += issue_count
            
            # Use agreement rate as consistency score
            summary["consistency_score"] = consistency_validation.get("agreement_rate", 0.0)
        
        # Calculate overall quality score
        summary["quality_score"] = (
            summary["format_score"] * 0.2 +
            summary["pattern_score"] * 0.2 +
            summary["consistency_score"] * 0.2 +
            summary["reference_score"] * 0.4
        )
        
        return summary
    
    def _summarize_overall_validation(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize overall validation results.
        
        Args:
            validation_results: All validation results
            
        Returns:
            Dictionary with overall validation summary
        """
        overall = {
            "passed": True,
            "issues_found": 0,
            "validation_count": 0,
            "passed_validations": 0,
            "warnings": [],
            "critical_issues": [],
            "quality_score": 0.0
        }
        
        # Count issues
        field_validations = validation_results.get("field_validation", {})
        for field, validation in field_validations.items():
            validation_summary = validation.get("validation_summary", {})
            
            overall["validation_count"] += 1
            overall["issues_found"] += validation_summary.get("issue_count", 0)
            
            if validation_summary.get("passed", True):
                overall["passed_validations"] += 1
            else:
                overall["passed"] = False
                
                # Add field-level warning
                overall["warnings"].append(
                    f"Field '{field}' failed validation with {validation_summary.get('issue_count', 0)} issues"
                )
        
        # Check cross-field validation
        if cross_field_validation := validation_results.get("cross_field_validation"):
            overall["validation_count"] += 1
            overall["issues_found"] += cross_field_validation.get("issue_count", 0)
            
            if not cross_field_validation.get("valid", True):
                overall["passed"] = False
                
                # Add cross-field warning
                overall["warnings"].append(
                    f"Cross-field validation failed with {cross_field_validation.get('issue_count', 0)} issues"
                )
                
                # Add relationship issues
                for rel_key, rel_data in cross_field_validation.get("field_relationships", {}).items():
                    if not rel_data.get("valid", True):
                        overall["warnings"].append(
                            f"Relationship '{rel_key}' failed with {rel_data.get('issue_count', 0)} issues"
                        )
        
        # Check quality metrics
        quality_metrics = validation_results.get("quality_metrics", {})
        overall["quality_score"] = quality_metrics.get("overall_quality_score", 0.0)
        
        # Add warning for low quality score
        if overall["quality_score"] < 0.7:
            overall["warnings"].append(
                f"Low overall quality score: {overall['quality_score']:.2%}"
            )
        
        # Calculate pass rate
        if overall["validation_count"] > 0:
            overall["pass_rate"] = overall["passed_validations"] / overall["validation_count"]
        else:
            overall["pass_rate"] = 0.0
        
        return overall
    
    def _log_validation_summary(self, validation_results: Dict[str, Any]) -> None:
        """
        Log summary of validation results.
        
        Args:
            validation_results: Validation results
        """
        overall = validation_results["overall_validation"]
        quality_metrics = validation_results.get("quality_metrics", {})
        
        logger.info("=" * 40)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 40)
        
        # Log overall status
        status = "PASSED" if overall["passed"] else "FAILED"
        logger.info(f"Overall validation: {status}")
        logger.info(f"Issues found: {overall['issues_found']}")
        
        if "pass_rate" in overall:
            logger.info(f"Validation pass rate: {overall['pass_rate']:.2%}")
        
        if "quality_score" in overall:
            logger.info(f"Overall quality score: {overall['quality_score']:.2%}")
        
        # Log warnings
        for warning in overall.get("warnings", []):
            logger.warning(f"Warning: {warning}")
        
        # Log field-level results
        field_validations = validation_results.get("field_validation", {})
        if field_validations:
            logger.info("\nField Validation Results:")
            
            for field, validation in field_validations.items():
                summary = validation.get("validation_summary", {})
                field_status = "PASSED" if summary.get("passed", True) else "FAILED"
                
                logger.info(f"  Field '{field}': {field_status} - "
                            f"Quality: {summary.get('quality_score', 0.0):.2%}, "
                            f"Issues: {summary.get('issue_count', 0)}")
        
        # Log extraction success rate
        if "extraction_success_rate" in quality_metrics:
            logger.info(f"\nExtraction success rate: {quality_metrics['extraction_success_rate']:.2%}")
        
        logger.info("=" * 40)
    
    def _categorize_pattern(self, value: str, field: str) -> str:
        """
        Categorize the pattern of a value.
        
        Args:
            value: Value to categorize
            field: Field type
            
        Returns:
            Pattern category
        """
        if not value:
            return "empty"
            
        # Convert to string
        value_str = str(value).strip()
        
        # Check for numeric values
        if value_str.isdigit():
            return "numeric_only"
        
        # Check for currency
        if value_str.startswith('$') or value_str.endswith('$'):
            return "currency"
        
        # Check for dates
        import re
        date_patterns = [
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # MM/DD/YYYY, DD/MM/YYYY
            r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$',    # YYYY/MM/DD
            r'^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$'  # Month DD, YYYY
        ]
        
        if any(re.match(pattern, value_str) for pattern in date_patterns):
            return "date"
        
        # Check for alphanumeric
        if value_str.isalnum():
            return "alphanumeric"
        
        # Check for text with spaces
        if all(c.isalnum() or c.isspace() for c in value_str):
            return "text_with_spaces"
        
        # Mixed format (includes special characters)
        return "mixed_format"
    
    def _detect_outliers(self, values: List[str], field: str) -> List[Dict[str, Any]]:
        """
        Detect outliers in extraction values.
        
        Args:
            values: List of extracted values
            field: Field type
            
        Returns:
            List of outliers with reasons
        """
        outliers = []
        
        # Skip if not enough values for detection
        if len(values) < 5:
            return outliers
        
        # For numeric fields
        if field in ['work_order', 'cost']:
            # Try to convert values to numbers
            numbers = []
            value_map = {}
            
            for value in values:
                try:
                    # Clean up value
                    clean_value = value.replace('$', '').replace(',', '')
                    num = float(clean_value)
                    numbers.append(num)
                    value_map[num] = value
                except ValueError:
                    # Skip non-numeric values
                    continue
            
            if len(numbers) >= 5:
                # Calculate quartiles
                q1 = np.percentile(numbers, 25)
                q3 = np.percentile(numbers, 75)
                iqr = q3 - q1
                
                # Define outlier bounds
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Identify outliers
                for num in numbers:
                    if num < lower_bound:
                        outliers.append({
                            "value": value_map[num],
                            "reason": f"Value is unusually low (below {lower_bound:.2f})"
                        })
                    elif num > upper_bound:
                        outliers.append({
                            "value": value_map[num],
                            "reason": f"Value is unusually high (above {upper_bound:.2f})"
                        })
        
        # For date fields
        elif field == 'date':
            # Try to parse dates
            dates = []
            value_map = {}
            
            for value in values:
                try:
                    date = self._parse_date(value)
                    if date:
                        dates.append(date)
                        value_map[date] = value
                except Exception:
                    # Skip invalid dates
                    continue
            
            if len(dates) >= 5:
                # Convert to numeric (days since epoch)
                date_nums = [d.toordinal() for d in dates]
                
                # Calculate quartiles
                q1 = np.percentile(date_nums, 25)
                q3 = np.percentile(date_nums, 75)
                iqr = q3 - q1
                
                # Define outlier bounds
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Identify outliers
                for date in dates:
                    date_num = date.toordinal()
                    if date_num < lower_bound:
                        outliers.append({
                            "value": value_map[date],
                            "reason": "Date is unusually early"
                        })
                    elif date_num > upper_bound:
                        outliers.append({
                            "value": value_map[date],
                            "reason": "Date is unusually late"
                        })
        
        # Length-based outliers (applicable to all fields)
        lengths = [len(str(value)) for value in values]
        
        if len(lengths) >= 5:
            # Calculate quartiles
            q1 = np.percentile(lengths, 25)
            q3 = np.percentile(lengths, 75)
            iqr = q3 - q1
            
            # Define outlier bounds
            lower_bound = max(1, q1 - 1.5 * iqr)  # Ensure minimum length of 1
            upper_bound = q3 + 1.5 * iqr
            
            # Identify outliers
            for value in values:
                length = len(str(value))
                if length < lower_bound:
                    outliers.append({
                        "value": value,
                        "reason": f"Value is unusually short ({length} chars)"
                    })
                elif length > upper_bound:
                    outliers.append({
                        "value": value,
                        "reason": f"Value is unusually long ({length} chars)"
                    })
        
        return outliers
    
    def _analyze_error_pattern(
        self, 
        extracted: str, 
        ground_truth: str,
        field: str
    ) -> str:
        """
        Analyze error pattern between extracted value and ground truth.
        
        Args:
            extracted: Extracted value
            ground_truth: Ground truth value
            field: Field type
            
        Returns:
            Error pattern description
        """
        # Convert to strings
        extracted = str(extracted).strip()
        ground_truth = str(ground_truth).strip()
        
        # Check for empty extraction
        if not extracted:
            return "empty_extraction"
        
        # Check for substring containment
        if extracted in ground_truth:
            return "partial_extraction"
        
        if ground_truth in extracted:
            return "over_extraction"
        
        # Check for digit mismatches
        if field == 'work_order' and extracted.isdigit() and ground_truth.isdigit():
            if len(extracted) == len(ground_truth):
                # Same length but different digits
                mismatched_positions = sum(1 for a, b in zip(extracted, ground_truth) if a != b)
                if mismatched_positions <= 2:
                    return "digit_error"
            
            # Check for transposition
            if sorted(extracted) == sorted(ground_truth):
                return "digit_transposition"
            
            # Check for off-by-one on length
            if abs(len(extracted) - len(ground_truth)) == 1:
                # Extra digit or missing digit
                if len(extracted) > len(ground_truth):
                    return "extra_digit"
                else:
                    return "missing_digit"
        
        # Check for currency formatting errors
        if field == 'cost':
            # Remove currency symbols and formatting
            extracted_clean = extracted.replace('$', '').replace(',', '').replace(' ', '')
            ground_truth_clean = ground_truth.replace('$', '').replace(',', '').replace(' ', '')
            
            try:
                # Compare numeric values
                extracted_num = float(extracted_clean)
                ground_truth_num = float(ground_truth_clean)
                
                if extracted_num == ground_truth_num:
                    return "formatting_difference"
                
                # Check for decimal point errors
                if extracted_num * 10 == ground_truth_num or extracted_num == ground_truth_num * 10:
                    return "decimal_point_error"
                
                # Check for order of magnitude errors
                if extracted_num * 100 == ground_truth_num or extracted_num == ground_truth_num * 100:
                    return "order_of_magnitude_error"
            except ValueError:
                pass
        
        # Check for date format issues
        if field == 'date':
            try:
                extracted_date = self._parse_date(extracted)
                ground_truth_date = self._parse_date(ground_truth)
                
                if extracted_date and ground_truth_date:
                    if extracted_date == ground_truth_date:
                        return "date_format_difference"
                    
                    # Check for month/day swapped (common in US/European formats)
                    swapped_extracted = self._swap_month_day(extracted)
                    swapped_date = self._parse_date(swapped_extracted)
                    
                    if swapped_date and swapped_date == ground_truth_date:
                        return "month_day_swap"
                    
                    # Check for year format difference
                    if extracted_date.day == ground_truth_date.day and extracted_date.month == ground_truth_date.month:
                        return "year_format_difference"
            except Exception:
                pass
        
        # Default: unclassified error
        return "unclassified_difference"
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings.
        
        Uses Levenshtein distance to compute similarity score.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Empty string handling
        if not str1 and not str2:
            return 1.0
        
        if not str1 or not str2:
            return 0.0
        
        # Try to use Levenshtein distance if available
        try:
            from Levenshtein import distance
            
            # Compute Levenshtein distance
            dist = distance(str1, str2)
            
            # Convert to similarity (0.0 to 1.0)
            max_len = max(len(str1), len(str2))
            similarity = 1.0 - (dist / max_len)
            
            return similarity
            
        except ImportError:
            # Fallback to simple character matching
            matched = sum(a == b for a, b in zip(str1, str2))
            similarity = matched / max(len(str1), len(str2))
            
            return similarity
    
    def _parse_date(self, date_str: str) -> Optional[datetime.date]:
        """
        Parse date string into date object.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            Parsed date or None if parsing fails
        """
        import re
        from datetime import datetime
        
        # Clean up the date string
        date_str = str(date_str).strip()
        
        # Try common date formats
        formats = [
            '%m/%d/%Y',
            '%m/%d/%y',
            '%d/%m/%Y',
            '%d/%m/%y',
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%Y/%m/%d',
            '%m-%d-%Y',
            '%m-%d-%y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y'
        ]
        
        for fmt in formats:
            try:
                date = datetime.strptime(date_str, fmt).date()
                return date
            except ValueError:
                continue
        
        # Try to parse month-day-year pattern with regex
        match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', date_str)
        if match:
            # Extract components
            first, second, year = match.groups()
            
            # Determine if it's mm/dd or dd/mm format
            try:
                if int(first) <= 12:
                    # Try as mm/dd/yyyy
                    month, day = int(first), int(second)
                else:
                    # Try as dd/mm/yyyy
                    day, month = int(first), int(second)
                
                # Handle 2-digit years
                if len(year) == 2:
                    year = 2000 + int(year) if int(year) < 50 else 1900 + int(year)
                else:
                    year = int(year)
                
                # Validate components
                if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                    return datetime(year, month, day).date()
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _swap_month_day(self, date_str: str) -> str:
        """
        Swap month and day in a date string.
        
        Args:
            date_str: Date string to transform
            
        Returns:
            Date string with month and day swapped
        """
        import re
        
        # Look for common date patterns
        match = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', date_str)
        if match:
            first, second, year = match.groups()
            return f"{second}/{first}/{year}"
        
        return date_str
    
    def _get_best_prompt(self, field_data: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Get the best performing prompt for a field.
        
        Args:
            field_data: Results for all prompts for a field
            
        Returns:
            Name of best performing prompt or None if no valid prompts
        """
        best_prompt = None
        best_score = -1
        
        for prompt_name, prompt_data in field_data.items():
            metrics = prompt_data.get('metrics', {})
            success_rate = metrics.get('success_rate', 0)
            
            if success_rate > best_score:
                best_score = success_rate
                best_prompt = prompt_name
        
        return best_prompt
    
    def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        logger.debug(f"Cleaning up resources for {self.name}")
        gc.collect()


class VisualizationStage(BasePipelineStage):
    """
    Generate visualizations for extraction results.
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the visualization stage."""
        super().__init__(name or "Visualization")
        
        # Define stage requirements
        self.requirements = StageRequirements(
            required_inputs={"analysis_results"},
            optional_inputs={"results_collector"},
            provided_outputs={"visualizations"}
        )
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate results before visualization.
        
        Checks:
        - Required analysis results exist
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Raises:
            PipelineStageError: If validation fails
        """
        errors = self.requirements.validate_inputs(previous_results)
        if errors:
            raise PipelineStageError(
                f"Missing required inputs: {', '.join(errors)}",
                stage_name=self.name
            )
        
        # Check for analysis results
        if 'analysis' not in previous_results or 'analysis_results' not in previous_results['analysis']:
            raise PipelineStageError(
                "No analysis results found",
                stage_name=self.name
            )
        
        logger.info("Visualization stage inputs validated successfully")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create visualizations for extraction results.
        
        Generates:
        - Accuracy comparison charts
        - Error distribution visualizations
        - Performance heatmaps
        - Trend analysis charts
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Returns:
            Dictionary with visualization results
            
        Raises:
            PipelineStageError: If visualization fails
        """
        try:
            # Get analysis results
            analysis_results = previous_results['analysis']['analysis_results']
            
            # Get results collector if available
            results_collector = previous_results.get('resultscollection', {}).get('results_collector')
            
            # Determine visualization types from configuration
            viz_types = config.get('visualization', {}).get(
                'types', 
                ['accuracy_bar', 'error_distribution']
            )
            
            logger.info(f"Generating visualizations: {', '.join(viz_types)}")
            
            # Determine output directory
            output_dir = config.get('results_dir')
            if output_dir:
                output_dir = Path(output_dir) / 'visualizations'
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Using output directory: {output_dir}")
            else:
                output_dir = None
            
            # Create visualizations
            visualizations = create_visualizations(
                results=analysis_results,
                output_dir=output_dir,
                show_plots=False  # We don't want to display, just save
            )
            
            # Save visualizations to results collector if available
            if results_collector:
                for viz_name, viz_path in visualizations.items():
                    try:
                        field = self._get_field_from_viz_name(viz_name)
                        results_collector.save_visualization(
                            field=field or 'combined',
                            visualization_name=os.path.basename(viz_path),
                            file_path=viz_path
                        )
                        logger.info(f"Saved visualization {viz_name} for field {field or 'combined'}")
                    except Exception as e:
                        logger.error(f"Error saving visualization {viz_name}: {e}")
            
            # Generate additional custom visualizations
            custom_visualizations = self._create_custom_visualizations(
                analysis_results, 
                config,
                output_dir
            )
            
            # Combine all visualizations
            all_visualizations = {**visualizations, **custom_visualizations}
            
            logger.info(f"Generated {len(all_visualizations)} visualizations")
            
            return {
                'visualizations': all_visualizations
            }
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
            raise PipelineStageError(
                f"Visualization failed: {str(e)}",
                stage_name=self.name,
                original_error=e
            )
    
    def _get_field_from_viz_name(self, viz_name: str) -> Optional[str]:
        """
        Extract field name from visualization name.
        
        Args:
            viz_name: Visualization name
            
        Returns:
            Field name or None if not field-specific
        """
        if '_work_order_' in viz_name:
            return 'work_order'
        elif '_cost_' in viz_name:
            return 'cost'
        elif '_date_' in viz_name:
            return 'date'
        elif '_field_' in viz_name:
            return None  # Cross-field visualization
        
        return None
    
    def _create_custom_visualizations(
        self,
        analysis_results: Dict[str, Any],
        config: ExperimentConfiguration,
        output_dir: Optional[Path]
    ) -> Dict[str, str]:
        """
        Create custom visualizations beyond the standard set.
        
        Args:
            analysis_results: Analysis results
            config: Experiment configuration
            output_dir: Output directory
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        custom_visualizations = {}
        
        try:
            # Only create if matplotlib is available
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 1. Create prompt consistency radar chart
            try:
                self._create_prompt_consistency_radar(
                    analysis_results, 
                    output_dir,
                    custom_visualizations
                )
            except Exception as e:
                logger.error(f"Error creating prompt consistency radar: {e}")
            
            # 2. Create error distribution chart
            try:
                self._create_error_distribution_chart(
                    analysis_results,
                    output_dir,
                    custom_visualizations
                )
            except Exception as e:
                logger.error(f"Error creating error distribution chart: {e}")
            
            # 3. Create performance insights chart
            try:
                self._create_insights_chart(
                    analysis_results,
                    output_dir,
                    custom_visualizations
                )
            except Exception as e:
                logger.error(f"Error creating insights chart: {e}")
            
            # Close all plots to free memory
            plt.close('all')
            
        except ImportError:
            logger.warning("Matplotlib not available, skipping custom visualizations")
        
        return custom_visualizations
    
    def _create_prompt_consistency_radar(
        self,
        analysis_results: Dict[str, Any],
        output_dir: Optional[Path],
        visualizations: Dict[str, str]
    ) -> None:
        """
        Create radar chart showing prompt consistency across fields.
        
        Args:
            analysis_results: Analysis results
            output_dir: Output directory
            visualizations: Dictionary to store visualization paths
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get prompt consistency data
        prompt_consistency = analysis_results.get('prompt_analysis', {}).get('prompt_consistency', {})
        if not prompt_consistency:
            return
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Set up radar chart
        labels = list(prompt_consistency.keys())
        values = [prompt_consistency[l] for l in labels]
        num_vars = len(labels)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Make the plot circular
        values += values[:1]
        angles += angles[:1]
        labels += labels[:1]
        
        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        
        # Set title
        plt.title('Prompt Consistency Across Fields', size=15)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save figure
        if output_dir:
            output_path = output_dir / 'prompt_consistency_radar.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            visualizations['prompt_consistency_radar'] = str(output_path)
    
    def _create_error_distribution_chart(
        self,
        analysis_results: Dict[str, Any],
        output_dir: Optional[Path],
        visualizations: Dict[str, str]
    ) -> None:
        """
        Create chart showing error distribution by field and prompt.
        
        Args:
            analysis_results: Analysis results
            output_dir: Output directory
            visualizations: Dictionary to store visualization paths
        """
        import matplotlib.pyplot as plt
        
        # Get error data
        field_errors = analysis_results.get('error_analysis', {}).get('field_specific_errors', {})
        prompt_errors = analysis_results.get('error_analysis', {}).get('prompt_specific_errors', {})
        
        if not field_errors and not prompt_errors:
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot field errors
        if field_errors:
            fields = list(field_errors.keys())
            error_rates = [field_errors[f]['error_rate'] for f in fields]
            
            ax1.bar(fields, error_rates, color='coral')
            ax1.set_title('Error Rates by Field')
            ax1.set_ylabel('Error Rate')
            ax1.set_ylim(0, max(error_rates) * 1.2)
            
            # Add value labels
            for i, v in enumerate(error_rates):
                ax1.text(i, v + 0.01, f'{v:.2%}', ha='center')
            
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Plot prompt errors
        if prompt_errors:
            prompts = list(prompt_errors.keys())
            error_rates = [prompt_errors[p]['error_rate'] for p in prompts]
            
            ax2.bar(prompts, error_rates, color='skyblue')
            ax2.set_title('Error Rates by Prompt')
            ax2.set_ylabel('Error Rate')
            ax2.set_ylim(0, max(error_rates) * 1.2)
            
            # Add value labels
            for i, v in enumerate(error_rates):
                ax2.text(i, v + 0.01, f'{v:.2%}', ha='center')
            
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure
        if output_dir:
            output_path = output_dir / 'error_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            visualizations['error_distribution'] = str(output_path)
    
    def _create_insights_chart(
        self,
        analysis_results: Dict[str, Any],
        output_dir: Optional[Path],
        visualizations: Dict[str, str]
    ) -> None:
        """
        Create chart visualizing performance insights.
        
        Args:
            analysis_results: Analysis results
            output_dir: Output directory
            visualizations: Dictionary to store visualization paths
        """
        import matplotlib.pyplot as plt
        
        # Get insights
        insights = analysis_results.get('insights', [])
        
        if not insights:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, len(insights) * 0.8 + 2))
        
        # Group insights by type
        insight_types = {}
        for insight in insights:
            insight_type = insight.get('type', 'other')
            if insight_type not in insight_types:
                insight_types[insight_type] = []
            insight_types[insight_type].append(insight)
        
        # Colors for each insight type
        colors = {
            'field_performance': 'lightgreen',
            'prompt_effectiveness': 'skyblue',
            'prompt_consistency': 'lightblue',
            'error_rate': 'coral',
            'common_error': 'salmon',
            'other': 'lightgray'
        }
        
        # Plot insights
        y_pos = 0
        labels = []
        colors_list = []
        
        for insight_type, type_insights in insight_types.items():
            for insight in type_insights:
                labels.append(insight.get('insight', ''))
                colors_list.append(colors.get(insight_type, 'lightgray'))
                y_pos += 1
        
        # Plot horizontal bars
        y_positions = range(len(labels))
        ax.barh(y_positions, [1] * len(labels), align='center', color=colors_list, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_title('Performance Insights')
        
        # Remove x-axis
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Add legend
        legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors.values()]
        legend_labels = list(colors.keys())
        ax.legend(legend_handles, legend_labels, loc='lower right')
        
        plt.tight_layout()
        
        # Save figure
        if output_dir:
            output_path = output_dir / 'performance_insights.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            visualizations['performance_insights'] = str(output_path)
    
    def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        logger.debug(f"Cleaning up resources for {self.name}")
        
        # Close any open figures
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass
        
        gc.collect()

class ResourceMonitoringStage(BasePipelineStage):
    """
    Monitor and report resource usage throughout pipeline execution.
    
    This stage tracks:
    - Memory usage (RAM and GPU)
    - CPU utilization
    - GPU utilization
    - Execution time statistics
    - Resource bottlenecks
    """
    
    def __init__(self, name: Optional[str] = None):
        """Initialize the resource monitoring stage."""
        super().__init__(name or "ResourceMonitoring")
        
        # Define stage requirements
        self.requirements = StageRequirements(
            required_inputs=set(),
            optional_inputs={"field_extractions", "processed_results"},
            provided_outputs={"resource_metrics", "performance_report"}
        )
        
        # Track resource usage throughout execution
        self.sampling_interval = 5  # seconds
        self.resource_samples = []
        self.stage_timings = {}
        self.monitoring_thread = None
        self.monitoring_active = False
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate resource monitoring configuration.
        
        Ensures optional dependencies are available if needed.
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
        """
        # Validate monitoring configuration
        monitoring_config = config.get('monitoring', {})
        
        # If advanced monitoring is enabled, check for psutil
        if monitoring_config.get('advanced_monitoring', False):
            try:
                import psutil
                logger.info("Advanced monitoring enabled with psutil")
            except ImportError:
                logger.warning("psutil is not available, advanced CPU monitoring will be limited")
        
        # If GPU monitoring is enabled, check for CUDA
        if monitoring_config.get('gpu_monitoring', False) and not torch.cuda.is_available():
            logger.warning("GPU monitoring enabled but no CUDA device is available")
        
        # Check if resource usage sampling is enabled
        if monitoring_config.get('resource_sampling', False):
            self.sampling_interval = monitoring_config.get('sampling_interval', 5)
            logger.info(f"Resource sampling enabled with {self.sampling_interval}s interval")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process resource metrics and generate performance report.
        
        Analyzes:
        - Resource usage patterns
        - Performance bottlenecks
        - Optimization opportunities
        
        Args:
            config: Experiment configuration
            previous_results: Results from previous stages
            
        Returns:
            Dictionary with resource metrics and performance report
        """
        try:
            # Stop ongoing monitoring if active
            if self.monitoring_active:
                self._stop_monitoring()
            
            # Gather stage timing information from previous stages
            self._collect_stage_timings(previous_results)
            
            # Gather system resource information
            system_resources = self._get_system_resources()
            
            # Analyze GPU memory usage
            gpu_memory_metrics = self._analyze_gpu_memory()
            
            # Calculate resource metrics
            resource_metrics = {
                "system_resources": system_resources,
                "gpu_metrics": gpu_memory_metrics,
                "stage_performance": self._analyze_stage_performance(),
                "resource_samples": self.resource_samples,
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate performance report with optimization recommendations
            performance_report = self._generate_performance_report(resource_metrics, config)
            
            # Check for potential memory leaks
            memory_leak_analysis = self._check_for_memory_leaks()
            if memory_leak_analysis["potential_leaks"]:
                performance_report["warnings"].extend(memory_leak_analysis["warnings"])
                performance_report["recommendations"].extend(memory_leak_analysis["recommendations"])
            
            # Generate result
            result = {
                "resource_metrics": resource_metrics,
                "performance_report": performance_report
            }
            
            # Log summary
            self._log_resource_summary(resource_metrics, performance_report)
            
            return result
            
        except Exception as e:
            logger.error(f"Resource monitoring failed: {str(e)}")
            raise PipelineStageError(
                f"Resource monitoring failed: {str(e)}",
                stage_name=self.name,
                original_error=e
            )
    
    def _collect_stage_timings(self, previous_results: Dict[str, Any]) -> None:
        """
        Collect timing information from all previous stages.
        
        Args:
            previous_results: Results from previous stages
        """
        for stage_name, stage_result in previous_results.items():
            if hasattr(stage_result, 'metadata') and hasattr(stage_result.metadata, 'duration'):
                self.stage_timings[stage_name] = stage_result.metadata.duration
            elif isinstance(stage_result, dict) and 'metadata' in stage_result:
                metadata = stage_result['metadata']
                if isinstance(metadata, dict) and 'duration' in metadata:
                    self.stage_timings[stage_name] = metadata['duration']
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """
        Get comprehensive system resource information.
        
        Returns:
            Dictionary with system resource metrics
        """
        resources = {
            "cpu": {},
            "memory": {},
            "gpu": {},
            "disk": {}
        }
        
        # Get CPU info
        try:
            import psutil
            resources["cpu"]["count"] = psutil.cpu_count(logical=True)
            resources["cpu"]["physical_count"] = psutil.cpu_count(logical=False)
            resources["cpu"]["percent"] = psutil.cpu_percent(interval=0.5)
            
            # Get per-CPU utilization
            per_cpu = psutil.cpu_percent(interval=0.5, percpu=True)
            resources["cpu"]["per_cpu_percent"] = per_cpu
            
            # Get CPU frequency if available
            if hasattr(psutil, 'cpu_freq'):
                freq = psutil.cpu_freq()
                if freq:
                    resources["cpu"]["frequency_mhz"] = freq.current
        except ImportError:
            resources["cpu"]["available"] = False
            logger.warning("psutil not available for CPU monitoring")
        except Exception as e:
            logger.warning(f"Error getting CPU info: {e}")
            resources["cpu"]["error"] = str(e)
        
        # Get memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            resources["memory"]["total_gb"] = mem.total / (1024 ** 3)
            resources["memory"]["available_gb"] = mem.available / (1024 ** 3)
            resources["memory"]["used_gb"] = mem.used / (1024 ** 3)
            resources["memory"]["percent"] = mem.percent
            
            # Get swap info
            swap = psutil.swap_memory()
            resources["memory"]["swap_total_gb"] = swap.total / (1024 ** 3)
            resources["memory"]["swap_used_gb"] = swap.used / (1024 ** 3)
            resources["memory"]["swap_percent"] = swap.percent
        except ImportError:
            # Fallback for systems without psutil
            resources["memory"]["available"] = False
        except Exception as e:
            logger.warning(f"Error getting memory info: {e}")
            resources["memory"]["error"] = str(e)
        
        # Get GPU info if available
        if torch.cuda.is_available():
            resources["gpu"]["available"] = True
            resources["gpu"]["count"] = torch.cuda.device_count()
            
            # Get info for each GPU
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                device_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024 ** 3),
                    "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024 ** 3)
                }
                
                # Get total memory
                try:
                    props = torch.cuda.get_device_properties(i)
                    device_info["total_memory_gb"] = props.total_memory / (1024 ** 3)
                    device_info["compute_capability"] = f"{props.major}.{props.minor}"
                except Exception as e:
                    logger.warning(f"Error getting GPU properties for device {i}: {e}")
                
                gpu_info.append(device_info)
            
            resources["gpu"]["devices"] = gpu_info
            
            # Try to get GPU utilization with nvidia-smi if available
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode == 0:
                    utilizations = [float(x.strip()) for x in result.stdout.strip().split('\n')]
                    for i, util in enumerate(utilizations):
                        if i < len(resources["gpu"]["devices"]):
                            resources["gpu"]["devices"][i]["utilization_percent"] = util
            except Exception as e:
                logger.warning(f"Error getting GPU utilization: {e}")
        else:
            resources["gpu"]["available"] = False
        
        # Get disk info
        try:
            import psutil
            disk_info = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "total_gb": usage.total / (1024 ** 3),
                        "used_gb": usage.used / (1024 ** 3),
                        "percent": usage.percent
                    })
                except Exception as e:
                    logger.warning(f"Error getting disk info for {partition.mountpoint}: {e}")
            
            resources["disk"]["partitions"] = disk_info
        except ImportError:
            resources["disk"]["available"] = False
        except Exception as e:
            logger.warning(f"Error getting disk info: {e}")
            resources["disk"]["error"] = str(e)
        
        return resources
    
    def _analyze_gpu_memory(self) -> Dict[str, Any]:
        """
        Analyze GPU memory usage in detail.
        
        Returns:
            Dictionary with GPU memory analysis
        """
        if not torch.cuda.is_available():
            return {"available": False}
        
        try:
            result = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "per_device": []
            }
            
            # Get memory info for each device
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": device_props.total_memory / (1024 ** 3),
                    "allocated_memory_gb": torch.cuda.memory_allocated(i) / (1024 ** 3),
                    "reserved_memory_gb": torch.cuda.memory_reserved(i) / (1024 ** 3),
                    "compute_capability": f"{device_props.major}.{device_props.minor}"
                }
                
                # Calculate utilization percentages
                device_info["allocated_percent"] = (
                    device_info["allocated_memory_gb"] / device_info["total_memory_gb"] * 100
                )
                device_info["reserved_percent"] = (
                    device_info["reserved_memory_gb"] / device_info["total_memory_gb"] * 100
                )
                
                result["per_device"].append(device_info)
            
            # Calculate overall metrics
            total_memory = sum(d["total_memory_gb"] for d in result["per_device"])
            allocated_memory = sum(d["allocated_memory_gb"] for d in result["per_device"])
            reserved_memory = sum(d["reserved_memory_gb"] for d in result["per_device"])
            
            result["total_memory_gb"] = total_memory
            result["allocated_memory_gb"] = allocated_memory
            result["reserved_memory_gb"] = reserved_memory
            result["allocated_percent"] = (allocated_memory / total_memory * 100) if total_memory > 0 else 0
            result["reserved_percent"] = (reserved_memory / total_memory * 100) if total_memory > 0 else 0
            
            # Memory fragmentation analysis
            # Higher fragmentation means the difference between reserved and allocated is large
            if reserved_memory > 0:
                result["fragmentation_percent"] = ((reserved_memory - allocated_memory) / reserved_memory * 100)
            else:
                result["fragmentation_percent"] = 0
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing GPU memory: {e}")
            return {"available": True, "error": str(e)}
    
    def _analyze_stage_performance(self) -> Dict[str, Any]:
        """
        Analyze performance of each pipeline stage.
        
        Returns:
            Dictionary with stage performance analysis
        """
        if not self.stage_timings:
            return {}
        
        analysis = {
            "total_time": sum(self.stage_timings.values()),
            "per_stage": {},
            "bottlenecks": []
        }
        
        # Analyze each stage
        for stage_name, duration in self.stage_timings.items():
            percentage = (duration / analysis["total_time"]) * 100 if analysis["total_time"] > 0 else 0
            
            analysis["per_stage"][stage_name] = {
                "duration": duration,
                "percentage": percentage
            }
            
            # Identify bottlenecks (stages taking >25% of total time)
            if percentage > 25:
                analysis["bottlenecks"].append({
                    "stage": stage_name,
                    "duration": duration,
                    "percentage": percentage
                })
        
        # Sort bottlenecks by duration
        analysis["bottlenecks"] = sorted(
            analysis["bottlenecks"], 
            key=lambda x: x["duration"],
            reverse=True
        )
        
        return analysis
    
    def _check_for_memory_leaks(self) -> Dict[str, Any]:
        """
        Check for potential memory leaks based on resource samples.
        
        Returns:
            Dictionary with memory leak analysis
        """
        result = {
            "potential_leaks": False,
            "warnings": [],
            "recommendations": []
        }
        
        # Need at least a few samples to detect a trend
        if len(self.resource_samples) < 3:
            return result
        
        # Check GPU memory trend
        try:
            gpu_allocated = [
                sample.get("gpu", {}).get("allocated_memory_gb", 0)
                for sample in self.resource_samples
            ]
            
            # Check if memory usage is consistently increasing
            if gpu_allocated and all(x < y for x, y in zip(gpu_allocated, gpu_allocated[1:])):
                increase_rate = (gpu_allocated[-1] - gpu_allocated[0]) / len(gpu_allocated)
                
                if increase_rate > 0.1:  # More than 100MB per sample
                    result["potential_leaks"] = True
                    result["warnings"].append(
                        f"Potential GPU memory leak detected: {increase_rate:.2f}GB per sample"
                    )
                    result["recommendations"].append(
                        "Consider adding explicit memory cleanup between processing stages"
                    )
                    result["recommendations"].append(
                        "Check for tensor references being held between iterations"
                    )
        except Exception as e:
            logger.warning(f"Error analyzing GPU memory trend: {e}")
        
        # Check system memory trend if samples include it
        try:
            if all("memory" in sample for sample in self.resource_samples):
                memory_used = [
                    sample.get("memory", {}).get("used_gb", 0)
                    for sample in self.resource_samples
                ]
                
                # Check if memory usage is consistently increasing
                if memory_used and all(x < y for x, y in zip(memory_used, memory_used[1:])):
                    increase_rate = (memory_used[-1] - memory_used[0]) / len(memory_used)
                    
                    if increase_rate > 0.2:  # More than 200MB per sample
                        result["potential_leaks"] = True
                        result["warnings"].append(
                            f"Potential system memory leak detected: {increase_rate:.2f}GB per sample"
                        )
                        result["recommendations"].append(
                            "Check for large objects being cached or not properly garbage collected"
                        )
        except Exception as e:
            logger.warning(f"Error analyzing system memory trend: {e}")
        
        return result
    
    def _generate_performance_report(
        self, 
        resource_metrics: Dict[str, Any],
        config: ExperimentConfiguration
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report with recommendations.
        
        Args:
            resource_metrics: Resource usage metrics
            config: Experiment configuration
            
        Returns:
            Dictionary with performance report
        """
        report = {
            "summary": "",
            "warnings": [],
            "recommendations": [],
            "optimization_opportunities": []
        }
        
        # GPU utilization analysis
        if resource_metrics.get("gpu_metrics", {}).get("available", False):
            gpu_metrics = resource_metrics["gpu_metrics"]
            
            # High memory utilization warning
            if gpu_metrics.get("allocated_percent", 0) > 90:
                report["warnings"].append(
                    f"GPU memory utilization is very high: {gpu_metrics['allocated_percent']:.1f}%"
                )
                report["recommendations"].append(
                    "Consider using a more aggressive quantization strategy"
                )
                report["recommendations"].append(
                    "Process smaller batches to reduce memory pressure"
                )
            
            # Memory fragmentation warning
            if gpu_metrics.get("fragmentation_percent", 0) > 30:
                report["warnings"].append(
                    f"High GPU memory fragmentation: {gpu_metrics['fragmentation_percent']:.1f}%"
                )
                report["recommendations"].append(
                    "More frequent memory cleanup between processing stages may help"
                )
        elif config.get('model_name') and 'gpu' in resource_metrics.get("system_resources", {}):
            # GPU is not being used but model might benefit from it
            report["optimization_opportunities"].append(
                "Model is running on CPU but could potentially benefit from GPU acceleration"
            )
        
        # Stage performance analysis
        stage_perf = resource_metrics.get("stage_performance", {})
        if stage_perf and "bottlenecks" in stage_perf and stage_perf["bottlenecks"]:
            bottlenecks = stage_perf["bottlenecks"]
            report["warnings"].append(
                f"Performance bottlenecks detected in {len(bottlenecks)} stages"
            )
            
            for bottleneck in bottlenecks[:2]:  # Report top 2 bottlenecks
                stage = bottleneck["stage"]
                percentage = bottleneck["percentage"]
                
                report["warnings"].append(
                    f"Stage '{stage}' is taking {percentage:.1f}% of total execution time"
                )
                
                # Add stage-specific recommendations
                if "extraction" in stage.lower():
                    report["recommendations"].append(
                        "Consider processing images in parallel to improve extraction throughput"
                    )
                elif "model" in stage.lower():
                    report["recommendations"].append(
                        "Try a more optimized model loading strategy or quantization method"
                    )
                elif "visualization" in stage.lower():
                    report["recommendations"].append(
                        "Generate visualizations selectively or in a separate process"
                    )
        
        # Generate summary
        total_time = stage_perf.get("total_time", 0)
        gpu_memory = resource_metrics.get("gpu_metrics", {}).get("allocated_memory_gb", 0)
        
        summary_parts = []
        
        if total_time > 0:
            summary_parts.append(f"Total execution time: {total_time:.2f}s")
        
        if gpu_metrics := resource_metrics.get("gpu_metrics", {}).get("available", False):
            summary_parts.append(
                f"GPU memory usage: {gpu_memory:.2f}GB "
                f"({gpu_metrics.get('allocated_percent', 0):.1f}%)"
            )
        
        if bottlenecks := stage_perf.get("bottlenecks", []):
            if bottlenecks:
                summary_parts.append(
                    f"Main bottleneck: {bottlenecks[0]['stage']} "
                    f"({bottlenecks[0]['percentage']:.1f}%)"
                )
        
        report["summary"] = " | ".join(summary_parts)
        
        # Add general recommendations
        if not report["recommendations"]:
            report["recommendations"].append(
                "No significant performance issues detected"
            )
        
        return report
    
    def _log_resource_summary(
        self, 
        resource_metrics: Dict[str, Any],
        performance_report: Dict[str, Any]
    ) -> None:
        """
        Log summary of resource usage and performance.
        
        Args:
            resource_metrics: Resource usage metrics
            performance_report: Performance report with recommendations
        """
        logger.info("=" * 40)
        logger.info("RESOURCE MONITORING SUMMARY")
        logger.info("=" * 40)
        
        if performance_report.get("summary"):
            logger.info(f"Summary: {performance_report['summary']}")
        
        # Log warnings
        for warning in performance_report.get("warnings", []):
            logger.warning(f"Warning: {warning}")
        
        # Log recommendations
        for recommendation in performance_report.get("recommendations", []):
            logger.info(f"Recommendation: {recommendation}")
        
        # Log GPU metrics if available
        if gpu_metrics := resource_metrics.get("gpu_metrics", {}).get("available", False):
            logger.info(f"GPU Memory: "
                        f"{gpu_metrics.get('allocated_memory_gb', 0):.2f}GB / "
                        f"{gpu_metrics.get('total_memory_gb', 0):.2f}GB "
                        f"({gpu_metrics.get('allocated_percent', 0):.1f}%)")
        
        # Log stage timings
        stage_perf = resource_metrics.get("stage_performance", {}).get("per_stage", {})
        if stage_perf:
            logger.info("Stage Timings:")
            for stage, metrics in stage_perf.items():
                logger.info(f"  {stage}: {metrics['duration']:.2f}s ({metrics['percentage']:.1f}%)")
        
        logger.info("=" * 40)
    
    def start_resource_monitoring(self, interval: float = 5) -> None:
        """
        Start background thread to monitor resource usage.
        
        Args:
            interval: Sampling interval in seconds
        """
        if self.monitoring_active:
            return
        
        self.sampling_interval = interval
        self.monitoring_active = True
        
        # Create and start monitoring thread
        import threading
        self.monitoring_thread = threading.Thread(
            target=self._monitor_resources_thread,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Started resource monitoring with {interval}s interval")
    
    def _monitor_resources_thread(self) -> None:
        """Background thread function to sample resource usage periodically."""
        import time
        
        while self.monitoring_active:
            try:
                # Get current resource usage
                sample = {
                    "timestamp": datetime.now().isoformat(),
                    "gpu": self._get_gpu_memory_status(),
                }
                
                # Try to get CPU and memory info if psutil is available
                try:
                    import psutil
                    sample["cpu"] = {
                        "percent": psutil.cpu_percent(interval=0.1)
                    }
                    mem = psutil.virtual_memory()
                    sample["memory"] = {
                        "used_gb": mem.used / (1024 ** 3),
                        "percent": mem.percent
                    }
                except ImportError:
                    pass
                
                # Add to samples
                self.resource_samples.append(sample)
                
                # Limit number of samples kept
                max_samples = 1000
                if len(self.resource_samples) > max_samples:
                    self.resource_samples = self.resource_samples[-max_samples:]
                
            except Exception as e:
                logger.warning(f"Error in resource monitoring thread: {e}")
            
            # Sleep until next sampling
            time.sleep(self.sampling_interval)
    
    def _stop_monitoring(self) -> None:
        """Stop background resource monitoring."""
        self.monitoring_active = False
        
        # Wait for thread to terminate
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
            self.monitoring_thread = None
        
        logger.info("Stopped resource monitoring")
    
    def _get_gpu_memory_status(self) -> Dict[str, Any]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dictionary with GPU memory information
        """
        if not torch.cuda.is_available():
            return {"available": False}
        
        try:
            result = {"available": True}
            
            # Get memory for current device
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            
            # Convert to GB
            result["allocated_memory_gb"] = allocated / (1024 ** 3)
            result["reserved_memory_gb"] = reserved / (1024 ** 3)
            
            # Get total memory if possible
            try:
                props = torch.cuda.get_device_properties(device)
                result["total_memory_gb"] = props.total_memory / (1024 ** 3)
                
                # Calculate percentages
                result["allocated_percent"] = (allocated / props.total_memory) * 100
                result["reserved_percent"] = (reserved / props.total_memory) * 100
            except Exception:
                pass
            
            return result
        except Exception as e:
            logger.warning(f"Error getting GPU memory status: {e}")
            return {"available": True, "error": str(e)}
    
    def cleanup(self) -> None:
        """Clean up resources used by this stage."""
        logger.debug(f"Cleaning up resources for {self.name}")
        
        # Stop monitoring if active
        if self.monitoring_active:
            self._stop_monitoring()
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
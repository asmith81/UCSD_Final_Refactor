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
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

import torch
import pandas as pd

# Import base pipeline components
from .base import BasePipelineStage, PipelineStageError

# Import project modules
from src.config.experiment import ExperimentConfiguration
from src.data.loader import load_and_prepare_data
from src.models.loader import load_model_and_processor
from src.prompts.registry import get_prompt_registry
from src.execution.inference import process_image_with_metrics
from src.analysis.metrics import calculate_batch_metrics
from src.analysis.visualization import create_visualizations
from src.results.collector import ResultsCollector
from src.models.model_service import get_model_service

# Configure logging
logger = logging.getLogger(__name__)


class DataPreparationStage(BasePipelineStage):
    """
    Prepare data for extraction by loading ground truth 
    and creating batches of images.
    """
    
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
        """
        # Validate ground truth path
        ground_truth_path = config.get('ground_truth_path')
        if not ground_truth_path or not os.path.exists(ground_truth_path):
            raise PipelineStageError(
                f"Ground truth file not found: {ground_truth_path}"
            )
        
        # Validate image directory
        images_dir = config.get('images_dir')
        if not images_dir or not os.path.exists(images_dir):
            raise PipelineStageError(
                f"Images directory not found: {images_dir}"
            )
        
        # Validate fields to extract
        fields = config.get('fields_to_extract', [])
        if not fields:
            raise PipelineStageError(
                "No fields specified for extraction"
            )
    
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
        """
        # Placeholder for data preparation logic
        prepared_data = {}
        
        for field in config.get('fields_to_extract', []):
            # Load and prepare data for each field
            ground_truth_df, ground_truth_mapping, batch_items = load_and_prepare_data(
                ground_truth_path=config.get('ground_truth_path'),
                image_dir=config.get('images_dir'),
                field_to_extract=field,
                field_column_name=f"{field.replace('_', ' ').title()} Number",
                image_id_column='Invoice'
            )
            
            prepared_data[field] = {
                'ground_truth_df': ground_truth_df,
                'ground_truth_mapping': ground_truth_mapping,
                'batch_items': batch_items
            }
        
        return prepared_data


class ModelLoadingStage(BasePipelineStage):
    """
    Load and prepare models for extraction.
    """
    
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
        """
        model_name = config.get('model_name')
        if not model_name:
            raise PipelineStageError("No model name specified")
    
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
        """
        # Load model with optimal quantization
        model_name = config.get('model_name')
        
        # Determine quantization strategy
        quantization = None
        
        # Get model service for quantization determination
        model_service = get_model_service()
        
        # Check for explicitly specified quantization
        if config.get('quantization'):
            quantization = config.get('quantization')
            logger.info(f"Using explicitly specified quantization: {quantization}")
        
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
        
        # Load model and processor
        try:
            model, processor = load_model_and_processor(
                model_name=model_name,
                quantization=quantization
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_name} with quantization {quantization}")
            raise PipelineStageError(
                f"Model loading failed: {str(e)}",
                stage_name=self.name
            )
        
        # Log quantization details
        quantization_metadata = {}
        try:
            model_config = model_service.get_model_config(model_name)
            if quantization:
                quantization_metadata = model_config.get_quantization_metadata(quantization)
        except Exception as e:
            logger.warning(f"Could not retrieve quantization metadata: {e}")
        
        return {
            'model': model,
            'processor': processor,
            'model_name': model_name,
            'quantization': quantization,
            'quantization_metadata': quantization_metadata
        }


class PromptManagementStage(BasePipelineStage):
    """
    Manage and prepare prompts for extraction.
    """
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate prompt configuration.
        """
        # Validate fields and prompt configuration
        fields = config.get('fields_to_extract', [])
        if not fields:
            raise PipelineStageError("No fields specified for extraction")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare prompts for each field to extract.
        """
        # Get prompt registry
        prompt_registry = get_prompt_registry()
        
        # Prepare prompts for each field
        field_prompts = {}
        for field in config.get('fields_to_extract', []):
            # Get prompts for this field
            prompts = prompt_registry.get_by_field(field)
            
            # If no prompts, create a default prompt
            if not prompts:
                logger.warning(f"No prompts found for field {field}. Creating default prompt.")
                # Create a basic default prompt
                default_prompt = {
                    'name': f'default_{field}',
                    'text': f'Extract the {field} from this invoice image.',
                    'field': field
                }
                prompts = [default_prompt]
            
            field_prompts[field] = prompts
        
        return field_prompts


class ExtractionStage(BasePipelineStage):
    """
    Perform field extraction across batches and prompts.
    """
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate inputs for extraction stage.
        """
        # Check required previous stage results
        required_keys = [
            'data_preparation', 
            'model_loading', 
            'prompt_management'
        ]
        for key in required_keys:
            if key not in previous_results:
                raise PipelineStageError(f"Missing required previous stage result: {key}")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract fields using prepared models, prompts, and data.
        """
        # Get models and processors
        model = previous_results['model_loading']['model']
        processor = previous_results['model_loading']['processor']
        model_name = previous_results['model_loading']['model_name']
        
        # Store extraction results
        all_field_results = {}
        
        # Iterate through each field
        for field, field_data in previous_results['data_preparation'].items():
            # Get prompts for this field
            field_prompts = previous_results['prompt_management'][field]
            batch_items = field_data['batch_items']
            
            # Store results for this field
            field_results = {
                'prompt_results': {},
                'ground_truth_mapping': field_data['ground_truth_mapping']
            }
            
            # Process with each prompt
            for prompt in field_prompts:
                # Perform extraction for this prompt
                extraction_results = []
                for item in batch_items:
                    try:
                        result = process_image_with_metrics(
                            image_path=item['image_path'],
                            ground_truth=item['ground_truth'],
                            prompt=prompt,
                            model_name=model_name,
                            field_type=field,
                            model=model,
                            processor=processor
                        )
                        extraction_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {item['image_path']}: {e}")
                        extraction_results.append({
                            'error': str(e),
                            'image_path': item['image_path']
                        })
                
                # Store results for this prompt
                field_results['prompt_results'][prompt['name']] = extraction_results
            
            # Store results for this field
            all_field_results[field] = field_results
        
        return all_field_results


class ResultsCollectionStage(BasePipelineStage):
    """
    Collect and process extraction results.
    """
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate results before collection.
        """
        if 'field_extractions' not in previous_results:
            raise PipelineStageError("No field extractions found")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process and collect extraction results.
        """
        # Create results collector
        results_collector = ResultsCollector(
            base_path=config.get('results_dir', 'results'),
            experiment_name=config.get('name', 'extraction_experiment')
        )
        
        # Store processed results for each field
        processed_results = {}
        
        # Iterate through field extractions
        field_extractions = previous_results['field_extractions']
        for field, field_data in field_extractions.items():
            # Collect results for each prompt
            field_processed_results = {}
            
            for prompt_name, prompt_results in field_data['prompt_results'].items():
                # Calculate metrics for this prompt's results
                metrics = calculate_batch_metrics(
                    prompt_results, 
                    field=field, 
                    config=config.to_dict()
                )
                
                # Save field results
                results_collector.save_field_results(field, prompt_results)
                results_collector.save_field_metrics(field, metrics)
                
                # Store processed results
                field_processed_results[prompt_name] = {
                    'results': prompt_results,
                    'metrics': metrics
                }
            
            processed_results[field] = field_processed_results
        
        return processed_results


class AnalysisStage(BasePipelineStage):
    """
    Perform comprehensive results analysis.
    """
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate results before analysis.
        """
        if 'results_collection' not in previous_results:
            raise PipelineStageError("No collected results found")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze extraction results.
        """
        # Perform cross-field analysis
        cross_field_analysis = {}
        
        # Iterate through collected results
        collected_results = previous_results['results_collection']
        for field, field_results in collected_results.items():
            # Aggregate metrics across prompts
            field_analysis = {
                'best_prompt': None,
                'worst_prompt': None,
                'average_performance': {}
            }
            
            # Compare prompt performance
            prompt_metrics = []
            for prompt_name, prompt_data in field_results.items():
                metrics = prompt_data['metrics']
                prompt_metrics.append({
                    'prompt_name': prompt_name,
                    'metrics': metrics
                })
            
            # Sort prompts by performance
            sorted_prompts = sorted(
                prompt_metrics, 
                key=lambda x: x['metrics'].get('success_rate', 0), 
                reverse=True
            )
            
            # Identify best and worst prompts
            if sorted_prompts:
                field_analysis['best_prompt'] = sorted_prompts[0]['prompt_name']
                field_analysis['worst_prompt'] = sorted_prompts[-1]['prompt_name']
            
            # Store field analysis
            cross_field_analysis[field] = field_analysis
        
        return {
            'cross_field_analysis': cross_field_analysis,
            'detailed_results': collected_results
        }


class VisualizationStage(BasePipelineStage):
    """
    Generate visualizations for extraction results.
    """
    
    def _validate_input(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> None:
        """
        Validate results before visualization.
        """
        if 'analysis' not in previous_results:
            raise PipelineStageError("No analysis results found")
    
    def _process(
        self, 
        config: ExperimentConfiguration, 
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create visualizations for extraction results.
        """
        # Get results from previous analysis stage
        analysis_results = previous_results['analysis']
        
        # Determine visualization types
        viz_types = config.get('visualization', {}).get(
            'types', 
            ['accuracy_bar', 'error_distribution']
        )
        
        # Create visualizations
        visualizations = create_visualizations(
            results=analysis_results,
            output_dir=config.get('results_dir', 'results/visualizations'),
            show_plots=False  # We don't want to display, just save
        )
        
        return {
            'visualizations': visualizations,
            'visualization_types': viz_types
        }
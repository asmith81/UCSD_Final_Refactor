"""
Pipeline orchestration for invoice field extraction experiments.

This module provides high-level pipeline functionality to:
1. Coordinate the entire extraction workflow
2. Manage experiments across models and prompts
3. Handle configuration, execution, and result collection
4. Support both interactive and programmatic usage
5. Implement error handling and resource management

The pipeline leverages the modular architecture of the project to
support extensible experimentation with different models, prompts,
and extraction strategies.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime

import torch
import pandas as pd
from tqdm.auto import tqdm

# Import project modules
from src.models.loader import load_model_and_processor, optimize_model_memory, get_gpu_memory_info
from src.models.registry import get_model_config, list_available_models
from src.prompts.registry import get_prompt, get_prompts_by_category, get_prompts_by_field, list_prompt_categories
from src.config.environment import get_environment_config
from src.config.paths import get_path_config
from src.execution.inference import process_image_with_metrics, postprocess_extraction
from src.execution.batch import process_batches, prepare_batch_items, estimate_optimal_batch_size

# Set up logging
logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    Main pipeline for orchestrating extraction experiments.
    
    This class coordinates the entire workflow from configuration
    to execution to result collection and analysis.
    """
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
        result_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the extraction pipeline.
        
        Args:
            experiment_name: Name for this experiment run
            config_path: Path to a JSON/YAML config file (optional)
            result_dir: Directory to store results (optional)
        """
        # Set up experiment name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"experiment_{self.timestamp}"
        
        # Initialize paths
        self.paths = get_path_config(experiment_name=self.experiment_name)
        if result_dir:
            self.paths.experiment_dir = result_dir
        
        # Set up experiment config
        self.config = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "environment": get_environment_config().environment
        }
        
        # Load config from file if provided
        if config_path:
            self._load_config(config_path)
        
        # Initialize state
        self.model = None
        self.processor = None
        self.results = []
        self.ground_truth_mapping = {}
        self.prompts = []
        self.results_by_prompt = {}
        self.current_prompt_index = 0
        self.current_batch = 0
        
        logger.info(f"Extraction pipeline initialized for experiment: {self.experiment_name}")
        logger.info(f"Results will be stored in: {self.paths.experiment_dir}")
        logger.info(f"Raw results: {self.paths.raw_dir}")
        logger.info(f"Processed results: {self.paths.processed_dir}")
        logger.info(f"Visualizations: {self.paths.visualizations_dir}")
    
    def _load_config(self, config_path: Union[str, Path]) -> None:
        """
        Load pipeline configuration from a file.
        
        Args:
            config_path: Path to the configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load configuration based on file extension
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                file_config = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Update configuration
        self.config.update(file_config)
        logger.info(f"Loaded configuration from {config_path}")
    
    def load_ground_truth(
        self,
        ground_truth_path: Optional[Union[str, Path]] = None,
        image_id_column: str = "Invoice",
        field_column: Optional[str] = None,
        ground_truth_mapping: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Load ground truth data for evaluation.
        
        Args:
            ground_truth_path: Path to the ground truth CSV file
            image_id_column: Column name containing image IDs
            field_column: Column name for the field to extract
                          (uses field_to_extract from config if None)
            ground_truth_mapping: Direct mapping to use instead of loading from CSV
        
        Returns:
            Dictionary mapping image IDs to ground truth values
        """
        # If ground truth mapping provided directly, use that
        if ground_truth_mapping is not None:
            logger.info(f"Using provided ground truth mapping with {len(ground_truth_mapping)} entries")
            self.ground_truth_mapping = ground_truth_mapping
            return ground_truth_mapping
        
        # Use provided path or default from paths
        if ground_truth_path is None:
            ground_truth_path = self.paths.ground_truth_path
        
        # Load CSV file
        ground_truth_df = pd.read_csv(ground_truth_path)
        logger.info(f"Loaded ground truth data: {len(ground_truth_df)} records")
        
        # Determine field column if not specified
        if field_column is None:
            field_type = self.config.get("field_to_extract", "work_order")
            # Map field types to column names based on known patterns
            field_mapping = {
                "work_order": "Work Order Number/Numero de Orden",
                "cost": "Total",
                # Add more mappings as needed
            }
            field_column = field_mapping.get(field_type, field_type)
            logger.info(f"Using field column: {field_column}")
        
        # Create mapping from image ID to ground truth
        mapping = {}
        for _, row in ground_truth_df.iterrows():
            # Convert image ID to string for consistent matching
            image_id = str(row[image_id_column])
            
            # Store the ground truth value
            if field_column in row:
                mapping[image_id] = str(row[field_column])
            else:
                logger.warning(f"Field column '{field_column}' not found in row")
        
        logger.info(f"Created ground truth mapping for {len(mapping)} images")
        self.ground_truth_mapping = mapping
        return mapping
    
    def set_prompts(self, prompts):
        """
        Set the collection of prompts to use for comparison experiments.
        
        Args:
            prompts: List of prompt objects or dictionaries with prompt information
            
        Returns:
            Self for method chaining
        """
        # Store prompts as an instance variable
        self.prompts = prompts
        
        # Initialize results dictionary if needed
        if not hasattr(self, 'results_by_prompt'):
            self.results_by_prompt = {}
        
        # Set up results storage for each prompt
        for prompt in prompts:
            # Get prompt identifier (name or ID)
            if isinstance(prompt, dict):
                prompt_key = prompt.get("name", str(id(prompt)))
            else:
                prompt_key = getattr(prompt, "name", str(id(prompt)))
            
            # Initialize empty results list if not already present
            if prompt_key not in self.results_by_prompt:
                self.results_by_prompt[prompt_key] = []
        
        logger.info(f"Set {len(prompts)} prompts for comparison")
        return self
    
    def get_current_prompt(self):
        """Get the current prompt being processed"""
        if 0 <= self.current_prompt_index < len(self.prompts):
            return self.prompts[self.current_prompt_index]
        return None
    
    def next_prompt(self):
        """Move to the next prompt for processing"""
        if self.current_prompt_index < len(self.prompts) - 1:
            self.current_prompt_index += 1
            return self.get_current_prompt()
        return None
    
    def reset_prompt_index(self):
        """Reset to the first prompt"""
        self.current_prompt_index = 0
        return self.get_current_prompt()
    
    def determine_optimal_batch_size(self, start_size=1, max_size=8):
        """Determine the optimal batch size for the GPU"""
        if not torch.cuda.is_available():
            logger.info("No GPU available, using batch size 1")
            return 1
            
        # Get a sample image to test
        sample_items = self.prepare_extraction_task(limit=1)
        if not sample_items:
            logger.warning("No images available to test batch size, using default")
            return self.config.get("batch_processing", {}).get("batch_size", 1)
            
        sample_image = sample_items[0]["image_path"]
        prompt = self.get_current_prompt() or self.get_experiment_prompt()
        
        try:
            # Use the optimization function from batch processing
            optimal_size = estimate_optimal_batch_size(
                model_name=self.config.get("model_name", "pixtral-12b"),
                image_path=sample_image,
                prompt=prompt,
                start_size=start_size,
                max_size=max_size
            )
            logger.info(f"Determined optimal batch size: {optimal_size}")
            return optimal_size
        except Exception as e:
            logger.warning(f"Error determining batch size: {e}")
            default_size = self.config.get("batch_processing", {}).get("batch_size", 1)
            logger.info(f"Using default batch size: {default_size}")
            return default_size
    
    def setup_model(
        self,
        model_name: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Set up model and processor for extraction.
        
        Args:
            model_name: Name of the model to use
            quantization: Quantization strategy
            **kwargs: Additional parameters for model loading
            
        Returns:
            Tuple of (model, processor)
        """
        # Use model name from config if not provided
        if model_name is None:
            model_name = self.config.get("model_name", "pixtral-12b")
        
        # Load model and processor
        logger.info(f"Loading model: {model_name}")
        model, processor = load_model_and_processor(
            model_name,
            quantization=quantization,
            **kwargs
        )
        
        # Store for later use
        self.model = model
        self.processor = processor
        self.config["model_name"] = model_name
        self.config["model_loaded_at"] = datetime.now().isoformat()
        
        # Log GPU info if available
        if torch.cuda.is_available():
            gpu_info = get_gpu_memory_info()
            logger.info(f"Model loaded on {gpu_info['device_name']}")
            logger.info(f"GPU memory: {gpu_info['allocated_memory_gb']:.2f} GB / {gpu_info['total_memory_gb']:.2f} GB")
        
        return model, processor
    
    def get_experiment_prompt(
        self,
        prompt_name: Optional[str] = None,
        prompt_category: Optional[str] = None,
        field_to_extract: Optional[str] = None
    ) -> Any:
        """
        Get prompt for the experiment.
        
        Args:
            prompt_name: Specific prompt name to use
            prompt_category: Category of prompts to use
            field_to_extract: Field to extract
            
        Returns:
            Prompt object or formatted prompt string
        """
        # Use config values if parameters not provided
        if prompt_name is None:
            prompt_name = self.config.get("prompt_name")
        
        if prompt_category is None:
            prompt_category = self.config.get("prompt_category")
        
        if field_to_extract is None:
            field_to_extract = self.config.get("field_to_extract", "work_order")
        
        # Get prompt by name if specified
        if prompt_name:
            prompt = get_prompt(prompt_name)
            if prompt:
                logger.info(f"Using prompt: {prompt_name}")
                return prompt
            else:
                logger.warning(f"Prompt '{prompt_name}' not found, falling back to category selection")
        
        # Get prompts by category and field if no specific prompt
        if prompt_category and field_to_extract:
            prompts = [p for p in get_prompts_by_category(prompt_category) 
                       if p.field_to_extract == field_to_extract]
            
            if prompts:
                prompt = prompts[0]  # Use the first one
                logger.info(f"Using prompt: {prompt.name} (from category {prompt_category})")
                return prompt
        
        # Fallback to a standard prompt for the field
        logger.warning("No matching prompt found, using default for field")
        standard_prompt = f"Extract the {field_to_extract} from this invoice image."
        return standard_prompt
    
    def prepare_extraction_task(
        self,
        image_paths: Optional[List[Union[str, Path]]] = None,
        limit: Optional[int] = None,
        shuffle: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Prepare the extraction task by gathering images and ground truth.
        
        Args:
            image_paths: List of image paths (if None, gets from path config)
            limit: Maximum number of images to process
            shuffle: Whether to shuffle the images
            
        Returns:
            List of dictionaries with image_path and ground_truth
        """
        # Get image paths if not provided
        if image_paths is None:
            image_paths = self.paths.get_image_paths(limit=limit)
        
        # Load ground truth if not already loaded
        if not self.ground_truth_mapping:
            self.load_ground_truth()
        
        # Prepare batch items
        items = prepare_batch_items(image_paths, self.ground_truth_mapping)
        
        # Apply limit if specified
        if limit is not None and limit > 0 and limit < len(items):
            items = items[:limit]
        
        # Shuffle if requested
        if shuffle:
            import random
            random.shuffle(items)
        
        logger.info(f"Prepared extraction task with {len(items)} images")
        return items
    
    def run_extraction(
        self,
        items: Optional[List[Dict[str, Any]]] = None,
        field_type: Optional[str] = None,
        prompt: Optional[Any] = None,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        metrics: List[str] = ["exact_match", "character_error_rate"],
        limit: Optional[int] = None,
        **kwargs
        ) -> List[Dict[str, Any]]:
        """
        Run the extraction pipeline.
        
        Args:
            items: List of items to process (if None, prepares from config)
            field_type: Type of field to extract (if None, uses config)
            prompt: Prompt to use (if None, gets from config)
            model_name: Model to use (if None, uses loaded model or config)
            batch_size: Size of batches (if None, estimates optimal)
            checkpoint_path: Path for checkpointing
            metrics: List of metrics to calculate
            limit: Maximum number of images to process
            **kwargs: Additional parameters for extraction
            
        Returns:
            List of extraction results
        """
        start_time = time.time()
        
        # Use config values for missing parameters
        if field_type is None:
            field_type = self.config.get("field_to_extract", "work_order")
        
        if model_name is None:
            model_name = self.config.get("model_name", "pixtral-12b")
        
        # Load model if not already loaded
        if self.model is None or self.processor is None:
            try:
                self.setup_model(model_name=model_name)
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise ValueError(f"Model loading failed: {str(e)}")
        
        # Get prompt if not provided
        if prompt is None:
            prompt = self.get_experiment_prompt()
        
        # Prepare items if not provided
        if items is None:
            items = self.prepare_extraction_task(limit=limit)
        
        # Set up checkpoint path if not provided
        if checkpoint_path is None and self.config.get("enable_checkpointing", True):
            checkpoint_path = self.paths.get_raw_path("checkpoint.json")
        
        # Estimate optimal batch size if not provided
        if batch_size is None and torch.cuda.is_available():
            sample_item = items[0] if items else None
            if sample_item:
                # Get a sample image to estimate batch size
                sample_image = sample_item["image_path"]
                try:
                    batch_size = estimate_optimal_batch_size(
                        model_name=model_name,
                        image_path=sample_image,
                        prompt=prompt
                    )
                except Exception as e:
                    logger.warning(f"Failed to estimate batch size: {str(e)}. Using default.")
                    batch_size = self.config.get("batch_processing", {}).get("default_batch_size", 1)
            else:
                batch_size = 1
        elif batch_size is None:
            # Default to 1 for CPU
            batch_size = 1
        
        logger.info(f"Starting extraction with batch size {batch_size}")
        logger.info(f"Field type: {field_type}, Model: {model_name}")
        logger.info(f"Processing {len(items)} images")
        
        # Configure error handling for batch processing
        error_handling = {
            "continue_on_error": self.config.get("error_handling", {}).get("continue_on_error", True),
            "max_failures": self.config.get("error_handling", {}).get("max_failures", len(items)),  # Default to allowing all to fail
            "error_callback": kwargs.get("error_callback", None)
        }
        
        # Add error handling to kwargs
        kwargs["error_handling"] = error_handling
        
        try:
            # Process batches with error handling
            results = process_batches(
                items=items,
                model_name=model_name,
                prompt=prompt,
                field_type=field_type,
                batch_size=batch_size,
                checkpoint_path=checkpoint_path,
                checkpoint_frequency=self.config.get("checkpoint_frequency", 5),
                resume_from_checkpoint=self.config.get("resume_from_checkpoint", True),
                metrics=metrics,
                show_progress=self.config.get("show_progress", True),
                model=self.model,
                processor=self.processor,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error during batch processing: {str(e)}")
            
            # Attempt to recover checkpoint if available
            if checkpoint_path and Path(checkpoint_path).exists():
                logger.info(f"Attempting to recover from checkpoint: {checkpoint_path}")
                try:
                    with open(checkpoint_path, 'r') as f:
                        results = json.load(f)
                    logger.info(f"Recovered {len(results)} results from checkpoint")
                except Exception as checkpoint_error:
                    logger.error(f"Failed to recover from checkpoint: {str(checkpoint_error)}")
                    # Return whatever we have, even if empty
                    results = []
            else:
                results = []
        
        # Count and log errors
        errors = [r for r in results if "error" in r]
        if errors:
            logger.warning(f"Extraction completed with {len(errors)} failed images out of {len(results)}")
            # Log details of first few errors
            for i, error in enumerate(errors[:3]):
                logger.warning(f"Error {i+1}: Image {error.get('image_id', 'unknown')} - {error.get('error', 'Unknown error')}")
        
        # Store results
        self.results = results
        
        # Log completion
        total_time = time.time() - start_time
        logger.info(f"Extraction completed in {total_time:.2f}s")
        logger.info(f"Processed {len(results)} images (Successful: {len(results) - len(errors)}, Failed: {len(errors)})")
        
        return results
    
    def analyze_results(self, results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Analyze extraction results and compute summary metrics.
        
        Args:
            results: List of extraction results (uses self.results if None)
            
        Returns:
            Dictionary with summary metrics
        """
        # Use stored results if none provided
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("No results to analyze")
            return {}
        
        # Extract metrics
        total_images = len(results)
        exact_matches = sum(1 for r in results if r.get("exact_match", False))
        avg_cer = sum(r.get("character_error_rate", 1.0) for r in results) / total_images
        avg_time = sum(r.get("processing_time", 0.0) for r in results) / total_images
        
        # Get model and field information
        model_name = results[0].get("model_name", self.config.get("model_name", "unknown"))
        field_type = results[0].get("field_type", self.config.get("field_to_extract", "unknown"))
        
        # Create summary
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "field_type": field_type,
            "total_images": total_images,
            "exact_match_count": exact_matches,
            "exact_match_accuracy": round(exact_matches / total_images * 100, 2) if total_images > 0 else 0,
            "average_character_error_rate": round(avg_cer, 4),
            "average_processing_time": round(avg_time, 2),
            "total_processing_time": round(sum(r.get("processing_time", 0.0) for r in results), 2)
        }
        
        # Log summary
        logger.info(f"Results analysis for experiment: {self.experiment_name}")
        logger.info(f"Model: {model_name}, Field: {field_type}")
        logger.info(f"Accuracy: {summary['exact_match_accuracy']}% ({exact_matches}/{total_images})")
        logger.info(f"Avg character error rate: {summary['average_character_error_rate']}")
        logger.info(f"Avg processing time: {summary['average_processing_time']}s per image")
        
        return summary
    
    def save_results(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        summary: Optional[Dict[str, Any]] = None,
        filename_prefix: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save results and summary to files.
        
        Args:
            results: List of extraction results (uses self.results if None)
            summary: Summary metrics (calculated if None)
            filename_prefix: Prefix for filename
            
        Returns:
            Dictionary with paths to saved files
        """
        # Use stored results if none provided
        if results is None:
            results = self.results
        
        # Calculate summary if not provided
        if summary is None:
            summary = self.analyze_results(results)
        
        # Generate prefix if not provided
        if filename_prefix is None:
            model_name = summary.get("model_name", "model")
            field_type = summary.get("field_type", "field")
            filename_prefix = f"{model_name}_{field_type}"
        
        # Ensure results directories exist
        os.makedirs(self.paths.raw_dir, exist_ok=True)
        os.makedirs(self.paths.processed_dir, exist_ok=True)
        os.makedirs(self.paths.experiment_dir, exist_ok=True)
        
        # Save detailed results to raw directory
        results_path = self.paths.get_raw_path(f"{filename_prefix}_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                "experiment_info": {
                    "name": self.experiment_name,
                    "timestamp": datetime.now().isoformat()
                },
                "results": results
            }, f, indent=2)
        
        # Save summary to processed directory
        summary_path = self.paths.get_processed_path(f"{filename_prefix}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save configuration to experiment directory (main level)
        config_path = self.paths.get_results_path(f"{filename_prefix}_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Raw results saved to {results_path}")
        logger.info(f"Processed summary saved to {summary_path}")
        logger.info(f"Configuration saved to {config_path}")
        
        return {
            "results": results_path,
            "summary": summary_path,
            "config": config_path
        }
    
    def run_prompt_comparison(
        self,
        field_type: Optional[str] = None,
        prompt_category: Optional[str] = None,
        num_images: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run a comparison of different prompts for the same field.
        
        Args:
            field_type: Type of field to extract
            prompt_category: Category of prompts to compare
            num_images: Number of images to use for comparison
            **kwargs: Additional parameters for extraction
            
        Returns:
            List of comparison results
        """
        # Use config values if not provided
        if field_type is None:
            field_type = self.config.get("field_to_extract", "work_order")
        
        # Get prompts to compare
        if prompt_category:
            # Get prompts from specific category for the field
            prompts = [p for p in get_prompts_by_category(prompt_category) 
                      if p.field_to_extract == field_type]
        else:
            # Get all prompts for the field
            prompts = get_prompts_by_field(field_type)
        
        if not prompts:
            logger.warning(f"No prompts found for field {field_type}")
            return []
        
        logger.info(f"Running prompt comparison for field {field_type}")
        logger.info(f"Testing {len(prompts)} different prompts on {num_images} images")
        
        # Prepare a subset of images
        items = self.prepare_extraction_task(limit=num_images)
        
        # Set up model if not already done
        if self.model is None or self.processor is None:
            self.setup_model()
        
        # Run extraction with each prompt
        comparison_results = []
        for prompt in prompts:
            logger.info(f"Testing prompt: {prompt.name}")
            
            # Run extraction
            results = self.run_extraction(
                items=items,
                field_type=field_type,
                prompt=prompt,
                **kwargs
            )
            
            # Analyze results
            summary = self.analyze_results(results)
            
            # Add to comparison
            comparison_results.append({
                "prompt_name": prompt.name,
                "prompt_category": prompt.category,
                "prompt_text": prompt.text,
                "results": summary
            })
        
        # Sort by accuracy
        comparison_results.sort(
            key=lambda x: x["results"].get("exact_match_accuracy", 0),
            reverse=True
        )
        
        # Save comparison results to processed directory
        comparison_path = self.paths.get_processed_path(f"prompt_comparison_{field_type}.json")
        with open(comparison_path, 'w') as f:
            json.dump({
                "experiment_info": {
                    "name": self.experiment_name,
                    "timestamp": datetime.now().isoformat(),
                    "field_type": field_type,
                    "num_images": len(items),
                    "num_prompts": len(prompts)
                },
                "comparison": comparison_results
            }, f, indent=2)
        
        logger.info(f"Prompt comparison results saved to {comparison_path}")
        
        return comparison_results
    
    def run_model_comparison(
        self,
        model_names: List[str],
        field_type: Optional[str] = None,
        prompt_name: Optional[str] = None,
        num_images: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run a comparison of different models with the same prompt.
        
        Args:
            model_names: List of models to compare
            field_type: Type of field to extract
            prompt_name: Name of prompt to use
            num_images: Number of images to use for comparison
            **kwargs: Additional parameters for extraction
            
        Returns:
            List of comparison results
        """
        # Use config values if not provided
        if field_type is None:
            field_type = self.config.get("field_to_extract", "work_order")
        
        # Get prompt
        prompt = self.get_experiment_prompt(prompt_name=prompt_name, field_to_extract=field_type)
        
        logger.info(f"Running model comparison for field {field_type}")
        logger.info(f"Testing {len(model_names)} models on {num_images} images")
        logger.info(f"Using prompt: {getattr(prompt, 'name', 'custom')}")
        
        # Prepare a subset of images
        items = self.prepare_extraction_task(limit=num_images)
        
        # Run extraction with each model
        comparison_results = []
        for model_name in model_names:
            logger.info(f"Testing model: {model_name}")
            
            # Clear previous model
            self.model = None
            self.processor = None
            optimize_memory(clear_cache=True)
            
            # Run extraction
            try:
                results = self.run_extraction(
                    items=items,
                    field_type=field_type,
                    prompt=prompt,
                    model_name=model_name,
                    **kwargs
                )
                
                # Analyze results
                summary = self.analyze_results(results)
                
                # Add to comparison
                comparison_results.append({
                    "model_name": model_name,
                    "results": summary
                })
            except Exception as e:
                logger.error(f"Error testing model {model_name}: {e}")
                comparison_results.append({
                    "model_name": model_name,
                    "error": str(e)
                })
        
        # Sort by accuracy (when available)
        comparison_results.sort(
            key=lambda x: x.get("results", {}).get("exact_match_accuracy", 0)
                         if "results" in x else -1,
            reverse=True
        )
        
        # Save comparison results to processed directory
        comparison_path = self.paths.get_processed_path(f"model_comparison_{field_type}.json")
        with open(comparison_path, 'w') as f:
            json.dump({
                "experiment_info": {
                    "name": self.experiment_name,
                    "timestamp": datetime.now().isoformat(),
                    "field_type": field_type,
                    "num_images": len(items),
                    "num_models": len(model_names),
                    "prompt": getattr(prompt, 'name', 'custom')
                },
                "comparison": comparison_results
            }, f, indent=2)
        
        logger.info(f"Model comparison results saved to {comparison_path}")
        
        return comparison_results
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the pipeline.
        """
        # Release model and processor
        self.model = None
        self.processor = None
        
        # Clear CUDA cache
        optimize_memory(clear_cache=True)
        
        logger.info("Pipeline resources cleaned up")


# Create a convenience function for running extraction
def run_extraction_pipeline(
    experiment_name: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> ExtractionPipeline:
    """
    Run a complete extraction pipeline with a single function call.
    
    Args:
        experiment_name: Name for the experiment
        config_path: Path to a config file
        **kwargs: Additional parameters for extraction
        
    Returns:
        ExtractionPipeline instance with results
    """
    # Create pipeline
    pipeline = ExtractionPipeline(
        experiment_name=experiment_name,
        config_path=config_path
    )
    
    # Run extraction
    pipeline.run_extraction(**kwargs)
    
    # Analyze and save results
    summary = pipeline.analyze_results()
    pipeline.save_results(summary=summary)
    
    # Clean up
    pipeline.cleanup()
    
    return pipeline
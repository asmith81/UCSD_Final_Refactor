"""
Notebook experiment utilities.

This module provides helper functions for experiment configuration,
template management, and simplified interfaces for running extraction
experiments in Jupyter notebooks.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("experiment_utils")


def list_available_models() -> List[Dict[str, Any]]:
    """
    List all available models in the models directory.
    
    Returns:
        List of dictionaries with model information
    """
    models_dir = os.environ.get("MODELS_DIR", "models")
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory {models_dir} does not exist.")
        return []
    
    models = []
    for model_dir in Path(models_dir).glob("*"):
        if model_dir.is_dir():
            model_info = {
                "name": model_dir.name,
                "path": str(model_dir),
                "size": sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024**3)  # Size in GB
            }
            
            # Check for config file
            config_file = model_dir / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    model_info.update({
                        "architecture": config.get("architectures", ["Unknown"])[0],
                        "vocab_size": config.get("vocab_size", "Unknown"),
                        "hidden_size": config.get("hidden_size", "Unknown")
                    })
                except:
                    pass
            
            models.append(model_info)
    
    return models


def list_available_prompts() -> Dict[str, List[str]]:
    """
    List all available prompts by field type.
    
    Returns:
        Dictionary mapping field types to lists of prompt names
    """
    try:
        # Import here to avoid dependency issues
        from src.prompts.prompt_registry import get_prompt_registry
        
        registry = get_prompt_registry()
        prompt_map = {}
        
        for prompt_name, prompt_info in registry.list_prompts().items():
            field_type = prompt_info.get("field_type", "general")
            if field_type not in prompt_map:
                prompt_map[field_type] = []
            prompt_map[field_type].append(prompt_name)
        
        return prompt_map
    except ImportError:
        logger.warning("Could not import prompt registry. Returning empty result.")
        return {}


def list_available_templates() -> List[Dict[str, Any]]:
    """
    List all available experiment templates.
    
    Returns:
        List of dictionaries with template information
    """
    try:
        # Import here to avoid dependency issues
        from src.config.experiment_config import get_available_templates
        
        templates = []
        for template_name, template in get_available_templates().items():
            templates.append({
                "name": template_name,
                "description": template.description,
                "type": template.template_type
            })
        return templates
    except ImportError:
        logger.warning("Could not import experiment templates. Returning empty result.")
        return []


def create_basic_experiment(
    model_name: str,
    fields: List[str],
    batch_size: int = 1,
    memory_optimization: bool = False,
    quantization: Optional[Dict[str, Any]] = None
) -> Any:  # Returns ExperimentConfiguration
    """
    Create a basic experiment configuration.
    
    Args:
        model_name: Name of the model to use
        fields: List of fields to extract
        batch_size: Batch size for processing
        memory_optimization: Whether to optimize memory usage
        quantization: Optional quantization configuration
        
    Returns:
        ExperimentConfiguration object
    """
    try:
        # Import here to avoid dependency issues
        from src.config.experiment_config import (
            create_experiment_config,
            ExperimentType
        )
        
        return create_experiment_config(
            experiment_type=ExperimentType.BASIC_EXTRACTION,
            model_name=model_name,
            fields=fields,
            batch_size=batch_size,
            memory_optimization=memory_optimization,
            quantization=quantization
        )
    except ImportError as e:
        logger.error(f"Could not import experiment configuration: {str(e)}")
        raise


def create_model_comparison_experiment(
    model_names: List[str],
    fields: List[str],
    batch_size: int = 1,
    memory_optimization: bool = False
) -> Any:  # Returns ExperimentConfiguration
    """
    Create a model comparison experiment configuration.
    
    Args:
        model_names: Names of models to compare
        fields: List of fields to extract
        batch_size: Batch size for processing
        memory_optimization: Whether to optimize memory usage
        
    Returns:
        ExperimentConfiguration object
    """
    try:
        # Import here to avoid dependency issues
        from src.config.experiment_config import (
            create_experiment_config,
            ExperimentType
        )
        
        return create_experiment_config(
            experiment_type=ExperimentType.MODEL_COMPARISON,
            model_name=",".join(model_names),  # Joined for the experiment name
            fields=fields,
            batch_size=batch_size,
            memory_optimization=memory_optimization,
            models_to_compare=model_names
        )
    except ImportError as e:
        logger.error(f"Could not import experiment configuration: {str(e)}")
        raise


def create_prompt_comparison_experiment(
    model_name: str,
    fields: List[str],
    prompt_variants: Dict[str, Dict[str, str]],
    batch_size: int = 1
) -> Any:  # Returns ExperimentConfiguration
    """
    Create a prompt comparison experiment configuration.
    
    Args:
        model_name: Name of the model to use
        fields: List of fields to extract
        prompt_variants: Dictionary mapping variant names to field->prompt mappings
        batch_size: Batch size for processing
        
    Returns:
        ExperimentConfiguration object
    """
    try:
        # Import here to avoid dependency issues
        from src.config.experiment_config import (
            create_experiment_config,
            ExperimentType
        )
        
        return create_experiment_config(
            experiment_type=ExperimentType.PROMPT_COMPARISON,
            model_name=model_name,
            fields=fields,
            batch_size=batch_size,
            prompt_variants=prompt_variants
        )
    except ImportError as e:
        logger.error(f"Could not import experiment configuration: {str(e)}")
        raise


def create_quantization_experiment(
    model_name: str,
    fields: List[str],
    quantization_strategies: List[Dict[str, Any]],
    batch_size: int = 1
) -> Any:  # Returns ExperimentConfiguration
    """
    Create a quantization comparison experiment configuration.
    
    Args:
        model_name: Name of the model to use
        fields: List of fields to extract
        quantization_strategies: List of quantization strategies to compare
        batch_size: Batch size for processing
        
    Returns:
        ExperimentConfiguration object
    """
    try:
        # Import here to avoid dependency issues
        from src.config.experiment_config import (
            create_experiment_config,
            ExperimentType
        )
        
        return create_experiment_config(
            experiment_type=ExperimentType.QUANTIZATION_COMPARISON,
            model_name=model_name,
            fields=fields,
            batch_size=batch_size,
            quantization_strategies=quantization_strategies
        )
    except ImportError as e:
        logger.error(f"Could not import experiment configuration: {str(e)}")
        raise


def load_experiment_template(
    template_name: str,
    **kwargs
) -> Any:  # Returns ExperimentConfiguration
    """
    Load a predefined experiment template.
    
    Args:
        template_name: Name of the template to load
        **kwargs: Override values for template parameters
        
    Returns:
        ExperimentConfiguration object
    """
    try:
        # Import here to avoid dependency issues
        from src.config.experiment_config import get_template
        
        template = get_template(template_name)
        if template is None:
            logger.error(f"Template '{template_name}' not found")
            available_templates = list_available_templates()
            template_names = [t["name"] for t in available_templates]
            logger.info(f"Available templates: {', '.join(template_names)}")
            raise ValueError(f"Template '{template_name}' not found")
        
        return template.create_configuration(**kwargs)
    except ImportError as e:
        logger.error(f"Could not import experiment templates: {str(e)}")
        raise


def run_extraction_experiment(
    config: Any,  # ExperimentConfiguration 
    data_path: Optional[Union[str, Path]] = None,
    show_progress: bool = True
) -> Any:  # Returns ExperimentResult
    """
    Run an extraction experiment with the given configuration.
    
    Args:
        config: Experiment configuration
        data_path: Path to data directory or file
        show_progress: Whether to show progress bars
        
    Returns:
        ExperimentResult object
    """
    try:
        # Import here to avoid dependency issues
        from src.execution.pipeline.factory import PipelineFactory
        
        # Set data path in config if provided
        if data_path:
            if not hasattr(config, "data_path") or not config.data_path:
                config.data_path = str(data_path)
        
        # Create and run pipeline
        logger.info(f"Creating pipeline for experiment: {config.experiment_name}")
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Fields: {', '.join(config.fields)}")
        
        _, pipeline_service = PipelineFactory.create_pipeline(config)
        
        # Configure progress display for notebooks
        if show_progress:
            pipeline_service.progress_tracker.use_notebook_display = True
        
        # Run pipeline
        logger.info("Starting pipeline execution")
        result = pipeline_service.run_pipeline(config)
        logger.info("Pipeline execution completed")
        
        return result
    except ImportError as e:
        logger.error(f"Could not import pipeline factory: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error running experiment: {str(e)}")
        raise


def visualize_experiment_results(
    result: Any,  # ExperimentResult
    output_format: str = "notebook"
) -> Any:
    """
    Visualize experiment results in the notebook.
    
    Args:
        result: Experiment result object
        output_format: Output format ('notebook', 'html', 'json')
        
    Returns:
        Visualization object or HTML string
    """
    try:
        # Import here to avoid dependency issues
        from src.analysis.visualization import create_results_visualization
        
        return create_results_visualization(
            result, 
            format=output_format
        )
    except ImportError as e:
        logger.error(f"Could not import visualization module: {str(e)}")
        # Fallback to simple dictionary display
        logger.info("Falling back to simple dictionary display")
        return asdict(result)


def search_results_by_criteria(
    model_name: Optional[str] = None,
    experiment_type: Optional[str] = None,
    fields: Optional[List[str]] = None,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for experiment results matching specified criteria.
    
    Args:
        model_name: Filter by model name
        experiment_type: Filter by experiment type
        fields: Filter by fields used
        min_date: Filter by minimum date (YYYY-MM-DD)
        max_date: Filter by maximum date (YYYY-MM-DD)
        
    Returns:
        List of matching experiment result metadata
    """
    try:
        # Import here to avoid dependency issues
        from src.results.collector import ExperimentLoader
        
        loader = ExperimentLoader()
        experiments = loader.filter_experiments(
            model=model_name,
            type=experiment_type,
            fields=fields,
            start_date=min_date,
            end_date=max_date
        )
        
        return [
            {
                "id": exp_id,
                "name": metadata.get("experiment_name", ""),
                "model": metadata.get("model", ""),
                "date": metadata.get("timestamp", ""),
                "fields": metadata.get("fields", []),
                "path": metadata.get("path", "")
            }
            for exp_id, metadata in experiments.items()
        ]
    except ImportError as e:
        logger.error(f"Could not import experiment loader: {str(e)}")
        return []


def load_experiment_results(experiment_id: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load experiment results by ID.
    
    Args:
        experiment_id: ID of the experiment to load
        
    Returns:
        Tuple of (ExperimentResult, metadata)
    """
    try:
        # Import here to avoid dependency issues
        from src.results.collector import ExperimentLoader
        
        loader = ExperimentLoader()
        result = loader.load_experiment(experiment_id)
        
        if result is None:
            logger.error(f"Experiment {experiment_id} not found")
            return None, {}
        
        experiment_result, experiment_config = result
        
        # Extract metadata from config
        if experiment_config:
            metadata = {
                "id": experiment_id,
                "name": getattr(experiment_config, "experiment_name", ""),
                "type": getattr(experiment_config, "experiment_type", ""),
                "model": getattr(experiment_config, "model_name", ""),
                "fields": getattr(experiment_config, "fields", []),
                "batch_size": getattr(experiment_config, "batch_size", 1),
                "quantization": getattr(experiment_config, "quantization", None)
            }
        else:
            metadata = {"id": experiment_id}
        
        return experiment_result, metadata
    except ImportError as e:
        logger.error(f"Could not import experiment loader: {str(e)}")
        raise


def compare_experiments(
    experiment_ids: List[str],
    field: Optional[str] = None,
    metrics: Optional[List[str]] = None
) -> Any:  # Returns ComparisonResult
    """
    Compare multiple experiments.
    
    Args:
        experiment_ids: List of experiment IDs to compare
        field: Optional field to focus comparison on
        metrics: Optional list of metrics to include
        
    Returns:
        ComparisonResult object
    """
    try:
        # Import here to avoid dependency issues
        from src.results.collector import ExperimentLoader, ExperimentComparator
        
        loader = ExperimentLoader()
        comparator = ExperimentComparator(loader)
        
        return comparator.compare_experiments(
            experiment_ids=experiment_ids,
            field=field,
            metrics=metrics
        )
    except ImportError as e:
        logger.error(f"Could not import experiment comparator: {str(e)}")
        raise


def get_default_fields() -> Dict[str, str]:
    """
    Get default field types with descriptions.
    
    Returns:
        Dictionary mapping field names to descriptions
    """
    return {
        "invoice_number": "The unique identifier for the invoice",
        "invoice_date": "The date when the invoice was issued",
        "due_date": "The date when payment is due",
        "total_amount": "The total amount to be paid",
        "subtotal": "The amount before taxes and discounts",
        "tax_amount": "The amount of tax charged",
        "vendor_name": "The name of the vendor or supplier",
        "vendor_address": "The address of the vendor or supplier",
        "customer_name": "The name of the customer",
        "customer_address": "The address of the customer",
        "line_items": "The individual items or services listed on the invoice"
    }


if __name__ == "__main__":
    # If executed as a script, print available models and templates
    print("Available Models:")
    for model in list_available_models():
        print(f"- {model['name']} ({model['size']:.2f} GB)")
    
    print("\nAvailable Templates:")
    for template in list_available_templates():
        print(f"- {template['name']}: {template['description']}") 
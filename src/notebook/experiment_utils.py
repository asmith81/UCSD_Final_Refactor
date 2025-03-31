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

# Import utilities from setup_utils
try:
    from src.notebook.setup_utils import get_system_info
except ImportError:
    logger.warning("Could not import get_system_info from setup_utils")
    
    # Define fallback get_system_info function
    def get_system_info() -> Dict[str, Any]:
        """
        Fallback function to get basic system information.
        
        Returns:
            Dictionary with basic system information
        """
        import platform
        import sys
        import os
        
        # Basic system information
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_path": sys.executable,
            "processor": platform.processor(),
            "packages": {},
            "env_vars": {},
            "disk_space": {"total_gb": 0, "free_gb": 0, "used_gb": 0},
            "in_runpod": os.path.exists("/workspace") and os.path.exists("/cache"),
            "python_implementation": platform.python_implementation() if hasattr(platform, "python_implementation") else "Unknown"
        }
        
        # Get package versions
        try:
            import pkg_resources
            important_packages = [
                "torch", "numpy", "pandas", "transformers"
            ]
            
            for package in important_packages:
                try:
                    version = pkg_resources.get_distribution(package).version
                    info["packages"][package] = version
                except:
                    info["packages"][package] = "not installed"
        except:
            pass
        
        # Check GPU availability
        info["gpu"] = {"available": False}
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = {
                    "available": True,
                    "name": torch.cuda.get_device_name(0),
                    "count": torch.cuda.device_count(),
                    "cuda_version": torch.version.cuda,
                }
        except:
            pass
        
        # Get environment variables that might be useful
        for env_var in ["CUDA_VISIBLE_DEVICES", "RUNPOD_POD_ID", "GPU_NAME"]:
            if env_var in os.environ:
                info["env_vars"][env_var] = os.environ[env_var]
        
        # Get disk space information
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            info["disk_space"] = {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3)
            }
        except:
            pass
            
        return info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("experiment_utils")


def list_available_models() -> List[Dict[str, Any]]:
    """
    List all available models from both model files and configuration files.
    
    Returns:
        List of dictionaries with model information
    """
    models = []
    
    # Check for model directories in the models folder
    # Try different possible locations for models
    models_dir = os.environ.get("MODELS_DIR", "models")
    
    # Check if it's an absolute path
    if not os.path.isabs(models_dir):
        # Try relative to current directory
        if not os.path.exists(models_dir):
            # Try relative to project root
            project_root = os.getcwd()
            possible_model_paths = [
                os.path.join(project_root, models_dir),
                os.path.join(project_root, "src", models_dir),
                os.path.join(project_root, "workspace", models_dir)
            ]
            
            for path in possible_model_paths:
                if os.path.exists(path):
                    models_dir = path
                    break
            else:
                # None of the paths exist, just use the original
                logger.debug(f"Models directory '{models_dir}' not found in any standard locations")
    
    # Now use the directory path, without logging warnings if it doesn't exist
    if os.path.exists(models_dir):
        # Check if directory has any model subdirectories
        model_dirs = list(Path(models_dir).glob("*"))
        if not model_dirs:
            logger.debug(f"Models directory {models_dir} exists but is empty.")
        else:
            for model_dir in model_dirs:
                if model_dir.is_dir():
                    model_info = {
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "source": "directory",
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
    
    # Also check for model configuration files in configs/models
    config_models_dir = "configs/models"
    if os.path.exists(config_models_dir):
        try:
            import yaml
        except ImportError:
            logger.debug("Could not import yaml module. Will not check for model configuration files.")
            return models
            
        # Check if directory has any YAML files
        yaml_files = list(Path(config_models_dir).glob("*.yaml"))
        if not yaml_files:
            logger.debug(f"Config models directory {config_models_dir} exists but contains no YAML files.")
        
        for model_file in yaml_files:
            try:
                with open(model_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                model_info = {
                    "name": os.path.splitext(model_file.name)[0],
                    "path": str(model_file),
                    "source": "config",
                    "config_file": str(model_file)
                }
                
                # Add any metadata from the config file
                if isinstance(config, dict):
                    if "model_id" in config:
                        model_info["model_id"] = config["model_id"]
                    if "description" in config:
                        model_info["description"] = config["description"]
                    if "architecture" in config:
                        model_info["architecture"] = config["architecture"]
                
                models.append(model_info)
            except Exception as e:
                logger.debug(f"Error loading model config from {model_file}: {e}")
    
    return models


def list_available_prompts() -> Dict[str, List[str]]:
    """
    List all available prompts by field type.
    
    Returns:
        Dictionary mapping field types to lists of prompt names
    """
    try:
        # Import here to avoid dependency issues
        from src.prompts.registry import get_registry
        
        registry = get_registry()
        prompt_map = {}
        
        for prompt_name in registry.list_all():
            prompt = registry.get(prompt_name)
            if prompt:
                field_type = prompt.field_to_extract
                if field_type not in prompt_map:
                    prompt_map[field_type] = []
                prompt_map[field_type].append(prompt_name)
        
        return prompt_map
    except ImportError as e:
        logger.warning(f"Could not import prompt registry: {e}")
        return {}


def list_available_templates() -> List[Dict[str, str]]:
    """
    List available experiment templates with metadata.
    
    Returns:
        List of dictionaries with template metadata
    """
    try:
        from src.config.experiment_config import get_available_templates
        
        templates = []
        templates_dict = get_available_templates()
        
        # templates_dict is a dictionary with template names as keys and ExperimentTemplate objects as values
        for template_name, template in templates_dict.items():
            templates.append({
                "name": template_name,
                "description": template.description,
                "type": getattr(template, "template_type", "unknown"),
                "category": getattr(template, "category", "general")
            })
        return templates
    except Exception as e:
        logging.warning(f"Error listing templates: {e}")
        return []


def load_experiment_template(template_name: str, category: Optional[str] = None) -> Optional[Any]:
    """
    Load an experiment template by name.
    
    Args:
        template_name: Name of the template to load
        category: Optional category to limit the search
        
    Returns:
        ExperimentTemplate object or None if not found
    """
    try:
        from src.config.experiment_config import get_template
        
        # Try to get the template
        template = get_template(template_name, category)
        
        if template:
            logger.info(f"Loaded template: {template.name} ({template.category})")
        else:
            logger.warning(f"Template not found: {template_name}")
            
        return template
    except Exception as e:
        logger.warning(f"Error loading template '{template_name}': {e}")
        return None


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
        prompt_variants: Dictionary mapping field names to prompt variant dictionaries
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


def load_from_template(
    template_name: str,
    **kwargs
) -> Any:  # Returns ExperimentConfiguration
    """
    Load experiment configuration from a template.
    
    Args:
        template_name: Name of the template to use
        **kwargs: Template-specific parameters
        
    Returns:
        ExperimentConfiguration object
    """
    try:
        # Import here to avoid dependency issues
        from src.config.experiment_config import get_template
        
        template = get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        return template.create_config(**kwargs)
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


def create_custom_experiment(
    model_name: str,
    fields: List[str],
    **custom_params
) -> Any:  # Returns ExperimentConfiguration
    """
    Create a fully customizable experiment configuration.
    
    Args:
        model_name: Name of the model to use
        fields: List of fields to extract
        **custom_params: Any additional custom parameters
        
    Returns:
        ExperimentConfiguration object
    """
    try:
        # Import here to avoid dependency issues
        from src.config.experiment_config import (
            create_experiment_config,
            ExperimentType
        )
        
        # Start with required parameters
        experiment_params = {
            "model_name": model_name,
            "fields_to_extract": fields,
        }
        
        # Add all custom parameters
        experiment_params.update(custom_params)
        
        return create_experiment_config(
            experiment_type=ExperimentType.CUSTOM,
            **experiment_params
        )
    except ImportError as e:
        logger.error(f"Could not import experiment configuration: {str(e)}")
        raise


if __name__ == "__main__":
    # If executed as a script, print available models and templates
    print("Available Models:")
    for model in list_available_models():
        print(f"- {model['name']} ({model['size']:.2f} GB)")
    
    print("\nAvailable Templates:")
    for template in list_available_templates():
        print(f"- {template['name']}: {template['description']}") 
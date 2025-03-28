"""
Experiment configuration management.

This module provides functionality to load and manage experiment-specific
configurations, separate from environment settings. It includes support for
prompt management integration.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Import environment configuration for path resolution
from src.config.environment import get_environment_config

# Set up logging
logger = logging.getLogger(__name__)

class ExperimentConfig:
    """
    Manages experiment-specific configuration settings.
    
    This class handles loading experiment configuration, providing
    access to experiment parameters, and validating experiment settings.
    """
    
    def __init__(self, 
                 experiment_type: str = None,
                 config_path: str = None,
                 overrides: Dict[str, Any] = None):
        """
        Initialize experiment configuration.
        
        Args:
            experiment_type: Type of experiment (prompt_comparison, model_comparison, etc.)
            config_path: Path to custom experiment configuration
            overrides: Dictionary of configuration overrides
        """
        # Get environment configuration for path resolution
        self.env_config = get_environment_config()
        self.project_root = self.env_config.get('paths.base_dir')
        
        # Load base experiment configuration
        self.config = self._load_base_config()
        
        # Set experiment type
        self.experiment_type = experiment_type
        if experiment_type and experiment_type in self.config.get('experiment_types', {}):
            self.experiment_type_config = self.config['experiment_types'][experiment_type]
            logger.info(f"Using experiment type: {experiment_type}")
        else:
            self.experiment_type_config = {}
            if experiment_type:
                logger.warning(f"Unknown experiment type: {experiment_type}")
        
        # Apply custom configuration if provided
        if config_path:
            self._apply_custom_config(config_path)
        
        # Apply overrides
        if overrides:
            self._apply_overrides(overrides)
            
        # Set up logging based on configuration
        self._configure_logging()
        
        logger.info("Experiment configuration initialized")
    
    def _load_base_config(self) -> Dict[str, Any]:
        """
        Load base experiment configuration from YAML file.
        
        Returns:
            Dictionary containing experiment configuration
        """
        # Default config path is in the configs directory
        config_path = os.path.join(self.project_root, 'configs', 'experiment.yaml')
        
        if not Path(config_path).exists():
            logger.warning(f"Experiment configuration file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Resolve path variables
        self._resolve_path_variables(config)
            
        return config
    
    def _resolve_path_variables(self, config: Dict[str, Any]) -> None:
        """
        Resolve path variables in the configuration.
        
        Args:
            config: Configuration dictionary to process
        """
        def _process_item(item):
            """Process a single item, replacing path variables if it's a string."""
            if isinstance(item, str) and "${PROJECT_ROOT}" in item:
                return item.replace("${PROJECT_ROOT}", self.project_root)
            return item
        
        def _process_dict(d):
            """Recursively process a dictionary, replacing path variables."""
            for key, value in d.items():
                if isinstance(value, dict):
                    _process_dict(value)
                elif isinstance(value, list):
                    d[key] = [_process_item(item) for item in value]
                else:
                    d[key] = _process_item(value)
        
        _process_dict(config)
    
    def _apply_custom_config(self, config_path: str) -> None:
        """
        Apply custom configuration from a specified file.
        
        Args:
            config_path: Path to custom configuration file
        """
        if not Path(config_path).exists():
            logger.warning(f"Custom configuration file not found: {config_path}")
            return
        
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
        
        # Resolve path variables
        self._resolve_path_variables(custom_config)
        
        # Recursively update configuration
        self._update_config(self.config, custom_config)
        
        logger.info(f"Applied custom configuration from {config_path}")
    
    def _update_config(self, base_config: Dict[str, Any], update_config: Dict[str, Any]) -> None:
        """
        Recursively update a configuration dictionary.
        
        Args:
            base_config: Base configuration to update
            update_config: Values to apply
        """
        for key, value in update_config.items():
            if (key in base_config and isinstance(base_config[key], dict) 
                    and isinstance(value, dict)):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Apply configuration overrides.
        
        Args:
            overrides: Dictionary of overrides using dot notation
        """
        for key, value in overrides.items():
            self.set(key, value)
        
        logger.info(f"Applied {len(overrides)} configuration overrides")
    
    def _configure_logging(self) -> None:
        """Configure logging based on experiment settings."""
        log_config = self.config.get('logging', {})
        
        if not log_config:
            return
        
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(levelname)s - %(message)s')
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format
        )
        
        # Configure file logging if enabled
        if log_config.get('log_to_file', False):
            log_dir = log_config.get('log_dir')
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                file_handler = logging.FileHandler(
                    os.path.join(log_dir, f"experiment_{self.experiment_type or 'default'}.log")
                )
                file_handler.setFormatter(logging.Formatter(log_format))
                logging.getLogger().addHandler(file_handler)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Dot-separated path to the configuration value
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        # First check experiment type specific config
        if self.experiment_type_config:
            type_value = self._get_from_dict(self.experiment_type_config, key)
            if type_value is not None:
                return type_value
        
        # Then check general config
        return self._get_from_dict(self.config, key, default)
    
    def _get_from_dict(self, source: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        Get a value from a dictionary using dot notation.
        
        Args:
            source: Source dictionary
            key: Dot-separated path to the value
            default: Default value if key is not found
            
        Returns:
            Value or default
        """
        keys = key.split('.')
        value = source
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.
        
        Args:
            key: Dot-separated path to the configuration value
            value: Value to set
        """
        keys = key.split('.')
        
        # Navigate to the parent dictionary
        current = self.config
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
    
    def get_metrics(self) -> List[str]:
        """
        Get metrics to calculate for the current experiment type.
        
        Returns:
            List of metric names
        """
        if self.experiment_type and 'metrics_focus' in self.experiment_type_config:
            return self.experiment_type_config['metrics_focus']
        
        return self.get('metrics.extract_metrics', ['exact_match'])
    
    def get_visualizations(self) -> List[str]:
        """
        Get visualizations to generate for the current experiment type.
        
        Returns:
            List of visualization types
        """
        if self.experiment_type and 'visualization_focus' in self.experiment_type_config:
            return self.experiment_type_config['visualization_focus']
        
        return self.get('visualization.default_plots', ['accuracy_bar'])
    
    def get_default_model(self) -> str:
        """
        Get default model for the current experiment type.
        
        Returns:
            Default model name
        """
        if self.experiment_type and 'default_model' in self.experiment_type_config:
            return self.experiment_type_config['default_model']
        
        return self.get('experiment.default_model', 'pixtral-12b')
    
    def get_default_prompt(self) -> str:
        """
        Get default prompt for the current experiment type.
        
        Returns:
            Default prompt type
        """
        if self.experiment_type and 'default_prompt' in self.experiment_type_config:
            return self.experiment_type_config['default_prompt']
        
        return self.get('experiment.default_prompt', 'basic')
    
    # ---- PROMPT MANAGEMENT INTEGRATION ----
    
    def get_prompts_for_experiment(self) -> List[Dict[str, Any]]:
        """
        Get the list of prompts to use for this experiment based on configuration.
        
        Returns:
            List of prompt information dictionaries
        """
        try:
            from src.prompts import get_registry, get_prompt
            
            registry = get_registry()
            result = []
            
            # Get experiment configuration
            prompt_category = self.get('experiment.prompt_category', 'all')
            field_to_extract = self.get('experiment.field_to_extract', 'work_order')
            
            # Handle "specific" prompt - using a specific prompt
            specific_prompt = self.get('experiment.specific_prompt')
            if prompt_category == "specific" and specific_prompt:
                prompt = get_prompt(specific_prompt)
                if prompt:
                    return [{
                        "name": prompt.name,
                        "text": prompt.text,
                        "category": prompt.category,
                    }]
                else:
                    logger.error(f"Specific prompt '{specific_prompt}' not found in registry")
                    return []
            
            # Handle normal categories
            for prompt in registry.iter_prompts(
                category=None if prompt_category == "all" else prompt_category, 
                field=field_to_extract
            ):
                result.append({
                    "name": prompt.name,
                    "text": prompt.text,
                    "category": prompt.category,
                })
            
            if not result:
                logger.warning(f"No prompts found for category: {prompt_category}, field: {field_to_extract}")
            
            return result
        except ImportError as e:
            logger.error(f"Error importing prompt modules: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting prompts for experiment: {e}")
            return []
    
    def validate_prompt_config(self) -> Dict[str, Any]:
        """
        Validate prompt-related configuration and return validation results.
        
        Returns:
            Dictionary with validation results
        """
        try:
            from src.prompts import list_prompt_categories, get_prompts_by_field, get_prompt
            
            valid = True
            messages = []
            
            prompt_category = self.get('experiment.prompt_category', 'all')
            field_to_extract = self.get('experiment.field_to_extract', 'work_order')
            
            # Check that the prompt category exists
            if prompt_category != "all" and prompt_category != "specific":
                categories = list_prompt_categories()
                if prompt_category not in categories:
                    valid = False
                    messages.append(f"Invalid prompt category: {prompt_category}. Valid categories: {', '.join(categories)}")
            
            # Check that there are prompts for the specified field
            prompts = get_prompts_by_field(field_to_extract)
            if not prompts:
                valid = False
                messages.append(f"No prompts found for field: {field_to_extract}")
            
            # For "specific" prompt category, check that the prompt exists
            if prompt_category == "specific":
                specific_prompt = self.get('experiment.specific_prompt')
                if not specific_prompt:
                    valid = False
                    messages.append("'experiment.specific_prompt' must be provided when prompt_category is 'specific'")
                elif not get_prompt(specific_prompt):
                    valid = False
                    messages.append(f"Specific prompt '{specific_prompt}' not found in registry")
            
            return {
                "valid": valid,
                "messages": messages,
            }
        except ImportError as e:
            return {
                "valid": False,
                "messages": [f"Error importing prompt modules: {e}"],
            }
        except Exception as e:
            return {
                "valid": False,
                "messages": [f"Error validating prompt configuration: {e}"],
            }
    
    def format_prompt_for_model(self, prompt_name: str, model_name: Optional[str] = None) -> Optional[str]:
        """
        Format a prompt for a specific model.
        
        Args:
            prompt_name: Name of the prompt to format
            model_name: Name of the model (defaults to experiment's default model)
            
        Returns:
            Formatted prompt text or None if prompt not found
        """
        try:
            from src.prompts import get_prompt, format_prompt
            
            # Get the model name from config if not provided
            if model_name is None:
                model_name = self.get_default_model()
            
            # Get the prompt
            prompt = get_prompt(prompt_name)
            if not prompt:
                logger.error(f"Prompt '{prompt_name}' not found in registry")
                return None
            
            # Format and return
            return format_prompt(prompt, model_name)
        except ImportError as e:
            logger.error(f"Error importing prompt modules: {e}")
            return None
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            return None
    
    def create_prompt_experiment(self, 
                                name: Optional[str] = None,
                                description: Optional[str] = None,
                                model_name: Optional[str] = None,
                                field: Optional[str] = None,
                                category: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a prompt comparison experiment configuration.
        
        Args:
            name: Experiment name (default: auto-generated)
            description: Experiment description (default: auto-generated)
            model_name: Model to use (default: from config)
            field: Field to extract (default: from config)
            category: Prompt category to test (default: from config)
            
        Returns:
            Experiment configuration dictionary
        """
        from datetime import datetime
        
        # Use defaults from config if not provided
        model_name = model_name or self.get_default_model()
        field_to_extract = field or self.get('experiment.field_to_extract', 'work_order')
        prompt_category = category or self.get('experiment.prompt_category', 'all')
        
        # Generate name and description if not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not name:
            name = f"prompt_comparison_{field_to_extract}_{timestamp}"
        
        if not description:
            description = f"Comparing {prompt_category} prompts for {field_to_extract} extraction using {model_name}"
        
        # Create experiment configuration
        experiment_config = {
            "experiment": {
                "name": name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "type": "prompt_comparison",
                "model_name": model_name,
                "field_to_extract": field_to_extract,
                "prompt_category": prompt_category,
                "metrics": self.get_metrics(),
                "visualizations": self.get_visualizations(),
            }
        }
        
        return experiment_config
    
    def print_summary(self) -> None:
        """Print a summary of the experiment configuration."""
        print(f"Experiment Type: {self.experiment_type or 'default'}")
        
        if self.experiment_type and self.experiment_type_config:
            print(f"\nExperiment Description: {self.experiment_type_config.get('description', 'N/A')}")
        
        print("\nMetrics:")
        for metric in self.get_metrics():
            print(f"  - {metric}")
        
        print("\nVisualizations:")
        for viz in self.get_visualizations():
            print(f"  - {viz}")
        
        print("\nKey Settings:")
        for key in ['experiment.default_field', 'dataset.default_limit', 'model_defaults.max_new_tokens']:
            value = self.get(key, 'N/A')
            print(f"  {key}: {value}")
        
        # Print prompt-related information if available
        try:
            prompts = self.get_prompts_for_experiment()
            print(f"\nPrompts for Experiment ({len(prompts)}):")
            for i, p in enumerate(prompts[:5], 1):  # Show first 5 prompts
                print(f"  {i}. {p['name']} ({p['category']})")
            
            if len(prompts) > 5:
                print(f"  ... and {len(prompts) - 5} more")
                
        except Exception as e:
            print(f"\nUnable to retrieve prompts: {e}")


# Singleton instance for global access
_experiment_config = None

def get_experiment_config(experiment_type: str = None, 
                         config_path: str = None,
                         overrides: Dict[str, Any] = None,
                         reload: bool = False) -> ExperimentConfig:
    """
    Get experiment configuration singleton.
    
    Args:
        experiment_type: Type of experiment
        config_path: Path to custom configuration
        overrides: Configuration overrides
        reload: Force configuration reload
        
    Returns:
        ExperimentConfig instance
    """
    global _experiment_config
    
    if _experiment_config is None or reload:
        _experiment_config = ExperimentConfig(
            experiment_type=experiment_type,
            config_path=config_path,
            overrides=overrides
        )
    elif (experiment_type and _experiment_config.experiment_type != experiment_type) or overrides:
        # Create a new instance if experiment type changed or overrides provided
        _experiment_config = ExperimentConfig(
            experiment_type=experiment_type,
            config_path=config_path,
            overrides=overrides
        )
        
    return _experiment_config


if __name__ == "__main__":
    # Test experiment configuration
    config = get_experiment_config("prompt_comparison")
    config.print_summary()
    
    # Test with overrides
    print("\nWith overrides:")
    config = get_experiment_config(
        "prompt_comparison",
        overrides={"experiment.prompt_category": "basic", "experiment.field_to_extract": "work_order"}
    )
    config.print_summary()
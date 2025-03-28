"""
Model registry for managing ML model configurations.

This module provides functionality to:
1. Discover available model configurations
2. Load and validate model configurations from YAML files
3. Provide a standardized interface for accessing model parameters
4. Support environment-specific overrides and variations
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import project utilities
from src.config.environment import get_environment_config
from src.config.paths import get_path_config

# Set up logging
logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Registry for managing model configurations.
    
    This class handles discovery, loading, and validation of model
    configurations, providing a consistent way to access model
    parameters across the application.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            models_dir: Optional override for the models configuration directory
        """
        # Get environment configuration
        self.env_config = get_environment_config()
        self.paths = get_path_config(create_dirs=False)
        
        # Set models configuration directory
        if models_dir is None:
            project_root = self.env_config.project_root
            self.models_dir = os.path.join(project_root, "configs", "models")
        else:
            self.models_dir = models_dir
            
        # Cache for loaded model configurations
        self.model_configs = {}
        
        logger.info(f"Model registry initialized with config dir: {self.models_dir}")
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model configurations.
        
        Returns:
            List of model names (without .yaml extension)
        """
        models = []
        
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith(".yaml"):
                    models.append(file.replace(".yaml", ""))
        
        logger.info(f"Found {len(models)} model configurations: {', '.join(models)}")
        return models
    
    def load_model_config(self, model_name: str, reload: bool = False) -> Dict[str, Any]:
        """
        Load model configuration from YAML file.
        
        Args:
            model_name: Name of the model (with or without organization prefix)
            reload: Force reloading even if already cached
                
        Returns:
            Dictionary containing model configuration
                
        Raises:
            FileNotFoundError: If model configuration file doesn't exist
        """
        # Return cached config if available and reload not requested
        if model_name in self.model_configs and not reload:
            return self.model_configs[model_name]
        
        # Extract just the model name if it includes organization prefix
        config_name = model_name.split("/")[-1] if "/" in model_name else model_name
        
        # Construct path to model configuration file - try without organization prefix first
        config_path = os.path.join(self.models_dir, f"{config_name}.yaml")
        
        # Check if file exists
        if not os.path.exists(config_path):
            # Try with organization prefix if applicable
            if "/" in model_name:
                org, model = model_name.split("/", 1)
                alt_path = os.path.join(self.models_dir, org, f"{model}.yaml")
                if os.path.exists(alt_path):
                    config_path = alt_path
                    logger.info(f"Using organization-prefixed config path: {alt_path}")
                else:
                    logger.error(f"Model configuration not found: {config_path} or {alt_path}")
                    raise FileNotFoundError(f"No configuration file found for model: {model_name}")
            else:
                logger.error(f"Model configuration not found: {config_path}")
                raise FileNotFoundError(f"No configuration file found for model: {model_name}")
        
        # Load YAML configuration
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validate required fields
            self._validate_model_config(config, model_name)
            
            # Add model name if not already present
            if "name" not in config:
                config["name"] = model_name
                
            # Apply environment-specific overrides
            config = self._apply_environment_overrides(config)
            
            # Cache the configuration
            self.model_configs[model_name] = config
            
            logger.info(f"Loaded configuration for model: {model_name}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading model configuration for {model_name}: {str(e)}")
            raise
    
    def _validate_model_config(self, config: Dict[str, Any], model_name: str) -> None:
        """
        Validate that a model configuration has required fields.
        
        Args:
            config: Model configuration dictionary
            model_name: Name of the model for logging
            
        Raises:
            ValueError: If configuration is missing required fields
        """
        required_fields = ["repo_id", "model_type"]
        
        for field in required_fields:
            if field not in config:
                error_msg = f"Model configuration for {model_name} missing required field: {field}"
                logger.error(error_msg)
                raise ValueError(error_msg)
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment-specific overrides to model configuration.
        
        Args:
            config: Base model configuration
            
        Returns:
            Model configuration with environment-specific overrides applied
        """
        environment = self.env_config.environment
        
        # Check if there are environment-specific overrides
        if "environments" in config and environment in config["environments"]:
            env_overrides = config["environments"][environment]
            
            # Apply overrides
            for key, value in env_overrides.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    # Merge dictionaries
                    config[key].update(value)
                else:
                    # Replace values
                    config[key] = value
            
            logger.info(f"Applied {environment}-specific overrides to model configuration")
        
        return config
    
    def get_loading_params(self, model_name: str, 
                          quantization: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameters for loading a specific model.
        
        Args:
            model_name: Name of the model
            quantization: Optional quantization strategy to use
            
        Returns:
            Dictionary of parameters to pass to model loading function
        """
        # Load model configuration
        config = self.load_model_config(model_name)
        
        # Start with default loading configuration
        loading_params = config.get("loading", {}).copy()
        
        # Apply quantization configuration if specified
        if quantization is not None and "quantization" in config:
            quant_config = config["quantization"]
            
            if quantization in quant_config.get("options", {}):
                # Get specific quantization options
                quant_options = quant_config["options"][quantization]
                
                # Update loading parameters with quantization options
                loading_params.update(quant_options)
                logger.info(f"Applied {quantization} quantization strategy for {model_name}")
            else:
                logger.warning(f"Requested quantization '{quantization}' not found for {model_name}")
        elif "quantization" in config:
            # Use default quantization if available
            default_quant = config["quantization"].get("default")
            
            if default_quant and default_quant in config["quantization"].get("options", {}):
                quant_options = config["quantization"]["options"][default_quant]
                loading_params.update(quant_options)
                logger.info(f"Applied default quantization ({default_quant}) for {model_name}")
        
        # Add cache directory
        loading_params["cache_dir"] = self.paths.model_cache_dir
        
        # Filter out parameters that aren't meant for the model constructor
        # These are parameters used by our application but not by the model itself
        params_to_remove = ['default_strategy']
        for param in params_to_remove:
            if param in loading_params:
                loading_params.pop(param)
                logger.info(f"Removed '{param}' from loading parameters (not for model constructor)")
        
        return loading_params
    
    def get_inference_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get default inference parameters for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of inference parameters
        """
        config = self.load_model_config(model_name)
        return config.get("inference", {}).copy()


# Create singleton instance for global access
_model_registry = None

def get_model_registry(reload: bool = False) -> ModelRegistry:
    """
    Get the model registry singleton.
    
    Args:
        reload: Force creating a new registry instance
        
    Returns:
        ModelRegistry instance
    """
    global _model_registry
    
    if _model_registry is None or reload:
        _model_registry = ModelRegistry()
        
    return _model_registry


def list_available_models() -> List[str]:
    """
    Get list of all available model configurations.
    
    Returns:
        List of model names
    """
    registry = get_model_registry()
    return registry.get_available_models()


def get_model_config(model_name: str, reload: bool = False) -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        reload: Force reloading the configuration from disk
        
    Returns:
        Model configuration dictionary
    """
    registry = get_model_registry()
    return registry.load_model_config(model_name, reload=reload)


def get_loading_params(model_name: str, 
                      quantization: Optional[str] = None) -> Dict[str, Any]:
    """
    Get parameters for loading a specific model.
    
    Args:
        model_name: Name of the model
        quantization: Optional quantization strategy to use
        
    Returns:
        Dictionary of parameters to pass to model loading function
    """
    registry = get_model_registry()
    return registry.get_loading_params(model_name, quantization=quantization)


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Simple test of the model registry
    print("Available models:")
    for model in list_available_models():
        print(f"- {model}")
        
    if list_available_models():
        test_model = list_available_models()[0]
        print(f"\nConfiguration for {test_model}:")
        config = get_model_config(test_model)
        print(f"Model type: {config.get('model_type')}")
        print(f"Repository: {config.get('repo_id')}")
        
        print(f"\nLoading parameters for {test_model}:")
        loading_params = get_loading_params(test_model)
        for key, value in loading_params.items():
            print(f"  {key}: {value}")
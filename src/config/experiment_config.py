"""
Experiment Configuration Management

This module provides a comprehensive system for configuring and managing
invoice information extraction experiments, with a focus on:
1. Systematic experiment design
2. Flexible configuration
3. Support for different experiment types
4. Reproducibility
5. Detailed tracking of experimental parameters
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union

# Import project configuration components
from src.config.base_config import BaseConfig, ConfigurationError
from src.config.environment import get_environment_config
from src.config.paths import get_path_config
from src.models.model_service import get_model_service
from src.prompts.registry import get_prompt_registry

# Set up logging
logger = logging.getLogger(__name__)


class ExperimentType:
    """
    Predefined experiment types with specific characteristics.
    
    Supports systematic exploration of:
    - Prompt variations
    - Model comparisons
    - Quantization strategies
    - Multi-field extraction
    """
    # Base experiment types
    PROMPT_COMPARISON = "prompt_comparison"
    MODEL_COMPARISON = "model_comparison"
    MULTI_FIELD = "multi_field"
    QUANTIZATION_COMPARISON = "quantization_comparison"
    FULL_GRID = "full_grid"
    
    # Advanced experiment types
    PROMPT_MODEL_GRID = "prompt_model_grid"
    QUANTIZATION_PROMPT_GRID = "quantization_prompt_grid"
    
    @classmethod
    def get_all_types(cls) -> List[str]:
        """
        Get all defined experiment types.
        
        Returns:
            List of experiment type names
        """
        return [
            getattr(cls, attr) for attr in dir(cls) 
            if not attr.startswith('_') and isinstance(getattr(cls, attr), str)
        ]


@dataclass
class ExperimentConfiguration(BaseConfig):
    """
    Comprehensive configuration for an invoice extraction experiment.
    
    Provides a flexible, type-safe way to define experiment parameters
    with support for different experiment types and systematic variations.
    """
    
    # Basic experiment metadata
    name: str = f"extraction_experiment_{os.getpid()}"
    description: str = ""
    type: str = ExperimentType.PROMPT_COMPARISON
    timestamp: Optional[str] = None
    
    # Core extraction parameters
    fields_to_extract: List[str] = field(default_factory=lambda: ["work_order"])
    model_name: str = "pixtral-12b"
    
    # Prompt configuration
    prompt_category: Optional[str] = None
    prompt_names: Optional[List[str]] = None
    
    # Quantization configuration
    quantization_strategies: Optional[List[str]] = None
    
    # Experiment-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Validation and tracking
    reproducibility_seed: Optional[int] = None
    
    # Experimental configuration sections
    dataset: Dict[str, Any] = field(default_factory=lambda: {
        "source": "default",
        "limit": None,
        "shuffle": False
    })
    
    metrics: List[str] = field(default_factory=lambda: [
        "exact_match", 
        "character_error_rate"
    ])
    
    visualization: Dict[str, Any] = field(default_factory=lambda: {
        "generate": True,
        "types": ["accuracy_bar", "error_distribution"]
    })
    
    def validate(self) -> List[str]:
        """
        Validate the experiment configuration.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate experiment type
        if self.type not in ExperimentType.get_all_types():
            errors.append(f"Invalid experiment type: {self.type}")
        
        # Validate model
        model_service = get_model_service()
        if self.model_name not in model_service.list_available_models():
            errors.append(f"Model not found in registry: {self.model_name}")
        
        # Validate prompt configuration
        prompt_registry = get_prompt_registry()
        if self.prompt_names:
            for prompt_name in self.prompt_names:
                if prompt_registry.get(prompt_name) is None:
                    errors.append(f"Prompt not found in registry: {prompt_name}")
        
        # Validate quantization strategies
        if self.quantization_strategies:
            model_config = model_service.get_model_config(self.model_name)
            available_strategies = model_config.get_available_quantization_strategies()
            for strategy in self.quantization_strategies:
                if strategy not in available_strategies:
                    errors.append(f"Invalid quantization strategy for {self.model_name}: {strategy}")
        
        # Validate fields to extract
        if not self.fields_to_extract:
            errors.append("At least one field must be specified for extraction")
        
        return errors
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfiguration':
        """
        Create an experiment configuration from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExperimentConfiguration instance
        """
        # Create instance with default values
        config = cls()
        
        # Update with provided configuration
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert experiment configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            key: getattr(self, key) 
            for key in [
                'name', 'description', 'type', 'timestamp',
                'fields_to_extract', 'model_name', 
                'prompt_category', 'prompt_names', 
                'quantization_strategies', 'parameters',
                'reproducibility_seed', 'dataset', 
                'metrics', 'visualization'
            ]
        }
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the experiment configuration to a file.
        
        Args:
            path: Optional path to save the configuration
                  (defaults to experiment results directory)
        
        Returns:
            Path to the saved configuration file
        """
        # Use provided path or generate one
        if path is None:
            paths = get_path_config()
            path = paths.get_results_path(f"{self.name}_config.yaml")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save as YAML
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
        
        logger.info(f"Saved experiment configuration to {path}")
        return path
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfiguration':
        """
        Load an experiment configuration from a file.
        
        Args:
            path: Path to the configuration file
        
        Returns:
            ExperimentConfiguration instance
        
        Raises:
            ConfigurationError: If file cannot be loaded
        """
        try:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Create configuration from dictionary
            config = cls.from_dict(config_dict)
            
            # Validate configuration
            errors = config.validate()
            if errors:
                raise ConfigurationError(f"Invalid configuration: {'; '.join(errors)}")
            
            return config
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration from {path}: {e}")
    
    def create_experiment_metadata(self) -> Dict[str, Any]:
        """
        Create comprehensive metadata for the experiment.
        
        Returns:
            Dictionary with experiment metadata
        """
        # Get environment configuration
        env_config = get_environment_config()
        
        return {
            "experiment_name": self.name,
            "description": self.description,
            "type": self.type,
            "timestamp": self.timestamp or os.getpid(),
            "environment": {
                "type": str(env_config.get_environment()),
                "gpu": env_config.hardware_info.to_dict() if env_config.hardware_info else {}
            },
            "configuration": self.to_dict()
        }


def create_experiment_config(
    experiment_type: str = ExperimentType.PROMPT_COMPARISON,
    **kwargs
) -> ExperimentConfiguration:
    """
    Convenience function to create an experiment configuration.
    
    Args:
        experiment_type: Type of experiment to create
        **kwargs: Additional configuration parameters
    
    Returns:
        ExperimentConfiguration instance
    """
    # Merge default configuration with provided parameters
    config_dict = {
        "type": experiment_type,
        **kwargs
    }
    
    return ExperimentConfiguration.from_dict(config_dict)


# Global experiment configuration management
def get_experiment_config(
    experiment_type: Optional[str] = None,
    config_path: Optional[str] = None,
    **kwargs
) -> ExperimentConfiguration:
    """
    Get an experiment configuration.
    
    Args:
        experiment_type: Type of experiment
        config_path: Path to a configuration file
        **kwargs: Additional configuration parameters
    
    Returns:
        ExperimentConfiguration instance
    """
    if config_path:
        # Load from file if path provided
        return ExperimentConfiguration.load(config_path)
    
    # Create a new configuration
    return create_experiment_config(
        experiment_type=experiment_type or ExperimentType.PROMPT_COMPARISON,
        **kwargs
    )
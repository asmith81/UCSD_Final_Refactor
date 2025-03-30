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
import json
import logging
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable

# Import project configuration components
from src.config.base_config import BaseConfig, ConfigurationError
from src.config.environment_config import get_environment_config
from src.config.path_config import get_path_config
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
    
    # Notebook-specific types
    NOTEBOOK_SINGLE_MODEL = "notebook_single_model"
    NOTEBOOK_COMPARISON = "notebook_comparison"
    NOTEBOOK_VISUALIZATION = "notebook_visualization"
    
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
    
    # Notebook settings
    notebook: Dict[str, Any] = field(default_factory=lambda: {
        "interactive": True,
        "display_progress": True,
        "memory_tracking": True,
        "visualization_inline": True
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
                'metrics', 'visualization', 'notebook'
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
    
    def copy(self, **kwargs) -> 'ExperimentConfiguration':
        """
        Create a copy of this configuration with optional modifications.
        
        Args:
            **kwargs: Fields to override in the new configuration
            
        Returns:
            New ExperimentConfiguration instance
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return ExperimentConfiguration.from_dict(config_dict)
    
    def save_as_template(self, template_name: str, template_category: str = "custom", description: Optional[str] = None) -> str:
        """
        Save the current configuration as a reusable template.
        
        Args:
            template_name: Name of the template
            template_category: Category for grouping related templates
            description: Optional description of the template
            
        Returns:
            Path to the saved template
        """
        template = ExperimentTemplate(
            name=template_name,
            category=template_category,
            description=description or self.description,
            configuration=self.to_dict()
        )
        
        return template.save()


@dataclass
class ExperimentTemplate:
    """
    Template for creating experiment configurations.
    
    Provides a way to define and reuse common experiment configurations,
    particularly useful for notebooks where users can select from templates.
    """
    name: str
    category: str = "custom"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate template after initialization."""
        # Ensure name is present
        if not self.name:
            raise ValueError("Template name cannot be empty")
            
        # Ensure configuration is present
        if not self.configuration:
            raise ValueError("Template must include a configuration")
            
        # Clean up name for filesystem usage
        self._fs_name = self.name.replace(" ", "_").lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert template to a dictionary.
        
        Returns:
            Dictionary representation of the template
        """
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "tags": self.tags,
            "configuration": self.configuration,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentTemplate':
        """
        Create a template from a dictionary.
        
        Args:
            data: Dictionary representation of the template
            
        Returns:
            ExperimentTemplate instance
        """
        return cls(
            name=data.get("name", "Unnamed Template"),
            category=data.get("category", "custom"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            configuration=data.get("configuration", {}),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentTemplate':
        """
        Load a template from a file.
        
        Args:
            path: Path to the template file
            
        Returns:
            ExperimentTemplate instance
        """
        try:
            with open(path, 'r') as f:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    data = yaml.safe_load(f)
                elif path.endswith('.json'):
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {path}")
                    
            return cls.from_dict(data)
        except Exception as e:
            raise ConfigurationError(f"Error loading template from {path}: {e}")
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the template to a file.
        
        Args:
            path: Optional path to save the template
                  (defaults to templates directory)
                  
        Returns:
            Path to the saved template file
        """
        # Use provided path or generate one
        if path is None:
            template_dir = _get_template_directory(self.category)
            path = os.path.join(template_dir, f"{self._fs_name}.yaml")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save as YAML
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
        
        logger.info(f"Saved experiment template to {path}")
        return path
    
    def instantiate(self, **kwargs) -> ExperimentConfiguration:
        """
        Create a new experiment configuration from this template.
        
        Args:
            **kwargs: Fields to override in the new configuration
            
        Returns:
            ExperimentConfiguration instance
        """
        # Create a copy of the template configuration
        config_dict = dict(self.configuration)
        
        # Update with provided parameters
        config_dict.update(kwargs)
        
        # Create configuration
        return ExperimentConfiguration.from_dict(config_dict)


def _get_template_directory(category: str = "") -> str:
    """
    Get directory for storing templates of a specific category.
    
    Args:
        category: Template category
        
    Returns:
        Path to the template directory
    """
    paths = get_path_config()
    base_dir = os.path.join(paths.get_path('config_dir'), 'templates')
    
    if category:
        base_dir = os.path.join(base_dir, category)
    
    # Ensure directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    return base_dir


def get_available_templates(category: Optional[str] = None, include_builtins: bool = True) -> Dict[str, List[ExperimentTemplate]]:
    """
    Get available experiment templates.
    
    Args:
        category: Optional category to filter templates
        include_builtins: Whether to include built-in templates
        
    Returns:
        Dictionary mapping categories to lists of templates
    """
    templates = {}
    paths = get_path_config()
    
    # Get base template directory
    template_dir = os.path.join(paths.get_path('config_dir'), 'templates')
    
    # Find all template files
    if category:
        search_dirs = [os.path.join(template_dir, category)]
    else:
        # List all subdirectories
        if os.path.exists(template_dir):
            search_dirs = [os.path.join(template_dir, d) for d in os.listdir(template_dir) 
                          if os.path.isdir(os.path.join(template_dir, d))]
        else:
            search_dirs = []
            
        # Add base directory
        search_dirs.append(template_dir)
    
    # Add built-in templates
    if include_builtins:
        search_dirs.append(os.path.join(paths.get_path('package_dir'), 'config', 'templates'))
    
    # Load templates from directories
    for directory in search_dirs:
        if not os.path.exists(directory):
            continue
            
        # Determine category from directory name
        dir_category = os.path.basename(directory)
        
        # Find all template files
        yaml_templates = glob.glob(os.path.join(directory, "*.yaml"))
        yml_templates = glob.glob(os.path.join(directory, "*.yml"))
        json_templates = glob.glob(os.path.join(directory, "*.json"))
        
        # Combine all templates
        template_files = yaml_templates + yml_templates + json_templates
        
        # Create category entry if needed
        if dir_category not in templates:
            templates[dir_category] = []
        
        # Load templates
        for template_path in template_files:
            try:
                template = ExperimentTemplate.load(template_path)
                templates[dir_category].append(template)
            except Exception as e:
                logger.warning(f"Failed to load template from {template_path}: {e}")
    
    return templates


def list_template_categories() -> List[str]:
    """
    List available template categories.
    
    Returns:
        List of template categories
    """
    templates = get_available_templates()
    return list(templates.keys())


def list_templates(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List available templates with their basic information.
    
    Args:
        category: Optional category to filter templates
        
    Returns:
        List of template metadata dictionaries
    """
    templates = get_available_templates(category)
    
    result = []
    for category, category_templates in templates.items():
        for template in category_templates:
            result.append({
                "name": template.name,
                "category": category,
                "description": template.description,
                "tags": template.tags
            })
    
    return result


def get_template(name: str, category: Optional[str] = None) -> Optional[ExperimentTemplate]:
    """
    Get a specific template by name.
    
    Args:
        name: Name of the template
        category: Optional category to narrow search
        
    Returns:
        ExperimentTemplate or None if not found
    """
    templates = get_available_templates(category)
    
    # Clean up name for matching
    clean_name = name.replace(" ", "_").lower()
    
    # Search for template
    for category_templates in templates.values():
        for template in category_templates:
            template_name = template.name.replace(" ", "_").lower()
            if template_name == clean_name:
                return template
    
    return None


def create_from_template(template_name: str, category: Optional[str] = None, **kwargs) -> ExperimentConfiguration:
    """
    Create experiment configuration from a template.
    
    Args:
        template_name: Name of the template
        category: Optional category of the template
        **kwargs: Parameters to override in the template configuration
        
    Returns:
        ExperimentConfiguration instance
        
    Raises:
        ConfigurationError: If template not found
    """
    template = get_template(template_name, category)
    
    if template is None:
        raise ConfigurationError(f"Template not found: {template_name}")
    
    return template.instantiate(**kwargs)


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
    template_name: Optional[str] = None,
    template_category: Optional[str] = None,
    **kwargs
) -> ExperimentConfiguration:
    """
    Get an experiment configuration.
    
    Args:
        experiment_type: Type of experiment
        config_path: Path to a configuration file
        template_name: Name of a template to use
        template_category: Category of the template
        **kwargs: Additional configuration parameters
    
    Returns:
        ExperimentConfiguration instance
    """
    if config_path:
        # Load from file if path provided
        return ExperimentConfiguration.load(config_path)
    
    if template_name:
        # Create from template if specified
        return create_from_template(template_name, template_category, **kwargs)
    
    # Create a new configuration
    return create_experiment_config(
        experiment_type=experiment_type or ExperimentType.PROMPT_COMPARISON,
        **kwargs
    )


# Create built-in templates
def _create_builtin_templates() -> None:
    """Create built-in experiment templates."""
    path_config = get_path_config()
    template_dir = os.path.join(path_config.get_path('package_dir'), 'config', 'templates')
    
    # Ensure directory exists
    os.makedirs(template_dir, exist_ok=True)
    
    # Define built-in templates
    templates = [
        # Basic notebook templates
        ExperimentTemplate(
            name="notebook_single_model",
            category="notebook",
            description="Simple notebook template for single model extraction",
            tags=["notebook", "basic", "beginner"],
            configuration={
                "name": "notebook_extraction",
                "type": ExperimentType.NOTEBOOK_SINGLE_MODEL,
                "fields_to_extract": ["invoice_number", "date", "total_amount"],
                "model_name": "pixtral-12b",
                "dataset": {
                    "limit": 10,
                    "shuffle": True
                },
                "notebook": {
                    "interactive": True,
                    "display_progress": True,
                    "memory_tracking": True,
                    "visualization_inline": True
                }
            }
        ),
        ExperimentTemplate(
            name="notebook_model_comparison",
            category="notebook",
            description="Compare different models on the same task",
            tags=["notebook", "comparison", "intermediate"],
            configuration={
                "name": "model_comparison",
                "type": ExperimentType.NOTEBOOK_COMPARISON,
                "fields_to_extract": ["total_amount"],
                "prompt_names": ["default_extraction"],
                "model_comparison": {
                    "models": ["gpt-4o-mini", "pixtral-12b", "claude-3-haiku"],
                    "metrics": ["exact_match", "character_error_rate", "processing_time"]
                },
                "dataset": {
                    "limit": 20,
                    "shuffle": True
                },
                "notebook": {
                    "interactive": True,
                    "display_progress": True,
                    "memory_tracking": True,
                    "visualization_inline": True
                }
            }
        ),
        ExperimentTemplate(
            name="notebook_prompt_comparison",
            category="notebook",
            description="Compare different prompts on the same model",
            tags=["notebook", "comparison", "prompts", "intermediate"],
            configuration={
                "name": "prompt_comparison",
                "type": ExperimentType.NOTEBOOK_COMPARISON,
                "fields_to_extract": ["invoice_number", "date"],
                "model_name": "pixtral-12b",
                "prompt_comparison": {
                    "prompts": ["default_extraction", "detailed_extraction", "concise_extraction"],
                    "metrics": ["exact_match", "character_error_rate"]
                },
                "dataset": {
                    "limit": 15,
                    "shuffle": True
                },
                "notebook": {
                    "interactive": True,
                    "display_progress": True,
                    "visualization_inline": True
                }
            }
        ),
        
        # Advanced notebook templates
        ExperimentTemplate(
            name="notebook_quantization_analysis",
            category="notebook",
            description="Compare quantization strategies for a model",
            tags=["notebook", "advanced", "quantization"],
            configuration={
                "name": "quantization_analysis",
                "type": ExperimentType.NOTEBOOK_COMPARISON,
                "fields_to_extract": ["invoice_number"],
                "model_name": "pixtral-12b",
                "quantization_strategies": ["none", "int8", "int4", "gptq-int4"],
                "dataset": {
                    "limit": 20,
                    "shuffle": True
                },
                "notebook": {
                    "interactive": True,
                    "display_progress": True,
                    "memory_tracking": True,
                    "visualization_inline": True
                }
            }
        ),
        
        # Visualization templates
        ExperimentTemplate(
            name="notebook_visualization_dashboard",
            category="visualization",
            description="Comprehensive experiment visualization dashboard",
            tags=["notebook", "visualization", "dashboard"],
            configuration={
                "name": "visualization_dashboard",
                "type": ExperimentType.NOTEBOOK_VISUALIZATION,
                "visualization": {
                    "generate": True,
                    "types": [
                        "accuracy_bar", 
                        "error_distribution", 
                        "confusion_matrix",
                        "timeline",
                        "memory_usage",
                        "comparative_radar"
                    ],
                    "interactive": True,
                    "export_formats": ["html", "png"]
                },
                "notebook": {
                    "interactive": True,
                    "display_progress": False,
                    "visualization_inline": True
                }
            }
        )
    ]
    
    # Save templates
    for template in templates:
        try:
            template_path = os.path.join(template_dir, f"{template.category}")
            os.makedirs(template_path, exist_ok=True)
            template.save(os.path.join(template_path, f"{template._fs_name}.yaml"))
        except Exception as e:
            logger.warning(f"Failed to create built-in template {template.name}: {e}")


# Create built-in templates when module is imported
try:
    _create_builtin_templates()
except Exception as e:
    logger.warning(f"Failed to create built-in templates: {e}")
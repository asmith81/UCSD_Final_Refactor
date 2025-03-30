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
from collections import defaultdict
from pathlib import Path

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
            config=self.to_dict(),
            category=template_category,
            description=description or self.description
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
    config: Dict[str, Any]
    category: str = "custom"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    template_type: str = "general"
    
    def __post_init__(self):
        """Validate template after initialization."""
        # Ensure name is present
        if not self.name:
            raise ValueError("Template name cannot be empty")
            
        # Ensure configuration is present
        if not self.config:
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
            "template_type": self.template_type,
            "config": self.config
        }
    
    @classmethod
    def from_file(cls, file_path: str) -> 'ExperimentTemplate':
        """
        Create template from a file.
        
        Args:
            file_path: Path to YAML/JSON template file
            
        Returns:
            ExperimentTemplate instance
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file has an invalid format
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Template file not found: {file_path}")
        
        # Determine file type from extension
        ext = os.path.splitext(file_path)[1].lower()
        
        # Load configuration from file
        try:
            if ext in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif ext == '.json':
                with open(file_path, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
            raise ValueError(f"Error parsing template file: {str(e)}")
        
        # Extract template metadata
        if not isinstance(config, dict):
            raise ValueError("Template must be a dictionary")
        
        # Get basic template properties
        template_name = config.get('name')
        if not template_name:
            # Use filename as fallback
            template_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Get category from config or parent directory
        template_category = config.get('category')
        if not template_category:
            # Use parent directory name as fallback
            parent_dir = os.path.basename(os.path.dirname(file_path))
            if parent_dir and parent_dir != 'templates':
                template_category = parent_dir
            else:
                template_category = 'general'
        
        # Get description
        template_description = config.get('description', f"Template for {template_name}")
        
        # Get tags
        template_tags = config.get('tags', [])
        if isinstance(template_tags, str):
            template_tags = [tag.strip() for tag in template_tags.split(',')]
        
        # Get template type
        template_type = config.get('template_type', 'general')
        
        # Remove metadata from config to avoid duplication
        experiment_config = config.copy()
        for key in ['name', 'category', 'description', 'tags', 'template_type']:
            if key in experiment_config:
                del experiment_config[key]
        
        return cls(
            name=template_name,
            category=template_category,
            description=template_description,
            config=experiment_config,
            tags=template_tags,
            template_type=template_type
        )
    
    def save(self, directory: Optional[str] = None) -> str:
        """
        Save template to a file.
        
        Args:
            directory: Optional directory to save the template in
                (defaults to appropriate category directory)
                
        Returns:
            Path to the saved template file
        """
        if not directory:
            # Get appropriate directory for this category
            directory = get_template_dir(self.category)
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate filename from template name
        filename = f"{self.name.lower().replace(' ', '_')}.yaml"
        filepath = os.path.join(directory, filename)
        
        # Convert to dictionary
        template_dict = self.to_dict()
        
        # Save to file
        with open(filepath, 'w') as f:
            yaml.dump(template_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved template to {filepath}")
        return filepath
    
    def instantiate(self, **kwargs) -> 'ExperimentConfiguration':
        """
        Create an experiment configuration from this template.
        
        Args:
            **kwargs: Override parameters for the configuration
            
        Returns:
            ExperimentConfiguration instance
        """
        # Create a copy of the template configuration
        config_dict = dict(self.config)
        
        # Update with provided parameters
        config_dict.update(kwargs)
        
        # Set name and description if not provided
        if 'name' not in config_dict:
            config_dict['name'] = f"{self.name}_experiment"
        
        if 'description' not in config_dict:
            config_dict['description'] = self.description
        
        # Create the experiment configuration
        return create_experiment_config(**config_dict)


def _get_template_directory(category: str = "") -> str:
    """
    Get directory for storing templates of a specific category.
    
    Args:
        category: Template category
        
    Returns:
        Path to the template directory
    """
    paths = get_path_config()
    
    # Get the config_dir using get() instead of get_path() to handle missing paths
    config_dir = paths.get('config_dir') or paths.get('configs_dir')
    
    # Fallback if config_dir is still None
    if not config_dir:
        project_root = paths.get('project_root', os.getcwd())
        config_dir = os.path.join(project_root, 'configs')
    
    base_dir = os.path.join(config_dir, 'templates')
    
    if category:
        base_dir = os.path.join(base_dir, category)
    
    # Ensure directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    return base_dir


def get_template_dir(category: Optional[str] = None) -> str:
    """
    Get the directory path for experiment templates.
    
    Args:
        category: Optional template category
        
    Returns:
        Path to the template directory or category subdirectory
    """
    # Get path configuration
    paths = get_path_config()
    
    # Try to get the templates directory using get() instead of get_path()
    template_dir = paths.get('templates_dir')
    
    # If not found, create it under configs_dir
    if not template_dir or not os.path.exists(template_dir):
        # Try 'config_dir' first, then 'configs_dir'
        config_dir = paths.get('config_dir') or paths.get('configs_dir')
        
        # Fallback if config_dir is still None
        if not config_dir or not os.path.exists(config_dir):
            project_root = paths.get('project_root', os.getcwd())
            config_dir = os.path.join(project_root, 'configs')
            os.makedirs(config_dir, exist_ok=True)
        
        template_dir = os.path.join(config_dir, 'templates')
        # Create directory if it doesn't exist
        os.makedirs(template_dir, exist_ok=True)
        
        # Add it to paths for future use
        if hasattr(paths, 'paths'):
            paths.paths['templates_dir'] = template_dir
    
    # If category is provided, include it in the path
    if category:
        category_dir = os.path.join(template_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        return category_dir
    
    return template_dir


def get_available_templates(
    category: Optional[str] = None,
    include_builtin: bool = True
) -> Dict[str, ExperimentTemplate]:
    """
    Get available experiment templates.
    
    Args:
        category: Optional category to filter by
        include_builtin: Whether to include built-in templates
        
    Returns:
        Dictionary mapping template names to template objects
    """
    result = {}
    
    # Get the template directory, ensuring it exists
    template_dir = get_template_dir()
    
    if os.path.exists(template_dir):
        # Get all YAML files from the template directory
        for file_path in Path(template_dir).glob('**/*.yaml'):
            # Skip files that don't have a valid parent directory (shouldn't happen)
            if file_path.parent == Path(template_dir):
                template_category = "general"
            else:
                template_category = file_path.parent.name
            
            # Skip if filtering by category and this doesn't match
            if category and template_category != category:
                continue
            
            try:
                # Load template from file
                template = ExperimentTemplate.from_file(str(file_path))
                if template:
                    template_name = template.name
                    result[template_name] = template
            except Exception as e:
                logger.warning(f"Error loading template from {file_path}: {str(e)}")
    
    # Include built-in templates if requested
    if include_builtin:
        builtin_templates = get_builtin_templates(category)
        for template_name, template in builtin_templates.items():
            if template_name not in result:  # Don't override user templates
                result[template_name] = template
    
    return result


def get_builtin_templates(
    category: Optional[str] = None
) -> Dict[str, ExperimentTemplate]:
    """
    Get built-in experiment templates.
    
    Args:
        category: Optional category to filter by
        
    Returns:
        Dictionary mapping template names to template objects
    """
    result = {}
    
    # Define notebook templates
    notebook_templates = {
        "notebook_single_model": ExperimentTemplate(
            name="notebook_single_model",
            category="notebook",
            description="Basic single-model extraction experiment for notebooks",
            template_type="notebook",
            config={
                "pipeline_type": "llm_extraction",
                "model": "phi-2",
                "prompts": {
                    "extraction": "default_extraction"
                },
                "fields": ["work_order", "cost", "date"],
                "image_limit": 10,
                "evaluation": {
                    "metrics": ["accuracy", "f1_score"],
                    "compare_to_ground_truth": True
                },
                "visualize_results": True
            }
        ),
        
        "notebook_model_comparison": ExperimentTemplate(
            name="notebook_model_comparison",
            category="notebook",
            description="Compare multiple models on the same extraction task",
            template_type="notebook",
            config={
                "pipeline_type": "llm_extraction",
                "models": ["phi-2", "llava-1.5-7b"],
                "prompts": {
                    "extraction": "default_extraction"
                },
                "fields": ["work_order", "cost", "date"],
                "image_limit": 5,
                "evaluation": {
                    "metrics": ["accuracy", "f1_score", "latency"],
                    "compare_to_ground_truth": True
                },
                "visualize_results": True,
                "comparison_metrics": ["accuracy", "latency"]
            }
        ),
        
        "notebook_prompt_comparison": ExperimentTemplate(
            name="notebook_prompt_comparison",
            category="notebook",
            description="Compare different prompt variants for extraction",
            template_type="notebook",
            config={
                "pipeline_type": "llm_extraction",
                "model": "phi-2",
                "prompts": {
                    "extraction": ["default_extraction", "detailed_extraction", "simple_extraction"]
                },
                "fields": ["work_order", "cost", "date"],
                "image_limit": 5,
                "evaluation": {
                    "metrics": ["accuracy", "f1_score"],
                    "compare_to_ground_truth": True
                },
                "visualize_results": True,
                "comparison_metrics": ["accuracy"]
            }
        ),
        
        "notebook_quantization_analysis": ExperimentTemplate(
            name="notebook_quantization_analysis",
            category="notebook",
            description="Compare model performance with different quantization levels",
            template_type="notebook",
            config={
                "pipeline_type": "llm_extraction",
                "model": "llava-1.5-7b",
                "quantization_config": [
                    {"bits": 4, "group_size": 128},
                    {"bits": 8, "group_size": 128},
                    None  # No quantization
                ],
                "prompts": {
                    "extraction": "default_extraction"
                },
                "fields": ["work_order", "cost", "date"],
                "image_limit": 3,
                "evaluation": {
                    "metrics": ["accuracy", "f1_score", "latency", "memory_usage"],
                    "compare_to_ground_truth": True
                },
                "visualize_results": True
            }
        )
    }
    
    # Add visualization template
    visualization_templates = {
        "notebook_visualization_dashboard": ExperimentTemplate(
            name="notebook_visualization_dashboard",
            category="visualization",
            description="Dashboard to visualize extraction results from multiple experiments",
            template_type="visualization",
            config={
                "experiments_to_load": 3,  # Number of experiment results to load
                "metrics_to_show": ["accuracy", "f1_score", "latency"],
                "visualization_types": ["bar_chart", "confusion_matrix", "sample_outputs"],
                "interactive": True
            }
        )
    }
    
    # Filter by category if requested, otherwise return all
    if category:
        if category == "notebook":
            result.update(notebook_templates)
        elif category == "visualization":
            result.update(visualization_templates)
    else:
        result.update(notebook_templates)
        result.update(visualization_templates)
    
    return result


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


def get_template(
    template_name: str, 
    category: Optional[str] = None
) -> Optional[ExperimentTemplate]:
    """
    Get a specific template by name and optional category.
    
    Args:
        template_name: Name of the template to retrieve
        category: Optional category to filter by
        
    Returns:
        ExperimentTemplate or None if not found
    """
    # Clean template name to normalize for matching
    template_name = template_name.lower().strip().replace(" ", "_")
    
    # First check if it's a built-in template
    builtin_templates = get_builtin_templates(category)
    for cat, templates in builtin_templates.items():
        for template in templates:
            if template.name.lower() == template_name:
                if category and cat != category:
                    continue
                return template
    
    # If not a built-in template, look in the file system
    # Get template directory based on category
    if category:
        template_dir = get_template_dir(category)
        # Look for template files with matching name in this directory
        file_patterns = [
            f"{template_name}.yaml",
            f"{template_name}.yml",
            f"{template_name}.json"
        ]
        
        for pattern in file_patterns:
            file_path = os.path.join(template_dir, pattern)
            if os.path.exists(file_path):
                try:
                    return ExperimentTemplate.from_file(file_path)
                except Exception as e:
                    logger.warning(f"Error loading template {file_path}: {e}")
                    continue
    else:
        # Search across all categories
        templates_dir = get_template_dir()
        # Look in the root templates directory first
        file_patterns = [
            os.path.join(templates_dir, f"{template_name}.yaml"),
            os.path.join(templates_dir, f"{template_name}.yml"),
            os.path.join(templates_dir, f"{template_name}.json")
        ]
        
        # Check each pattern
        for file_path in file_patterns:
            if os.path.exists(file_path):
                try:
                    return ExperimentTemplate.from_file(file_path)
                except Exception as e:
                    logger.warning(f"Error loading template {file_path}: {e}")
                    continue
        
        # If not found in root, check all subdirectories
        for category_dir in [d for d in os.listdir(templates_dir) 
                            if os.path.isdir(os.path.join(templates_dir, d))]:
            category_path = os.path.join(templates_dir, category_dir)
            
            for ext in ['.yaml', '.yml', '.json']:
                file_path = os.path.join(category_path, f"{template_name}{ext}")
                if os.path.exists(file_path):
                    try:
                        return ExperimentTemplate.from_file(file_path)
                    except Exception as e:
                        logger.warning(f"Error loading template {file_path}: {e}")
                        continue
    
    # Template not found
    logger.warning(f"Template '{template_name}' not found")
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
    
    # Try multiple approaches to determine the template directory
    template_dir = None
    
    # Approach 1: Use package_dir from path_config
    package_dir = path_config.get_path('package_dir')
    if package_dir is not None:
        template_dir = os.path.join(package_dir, 'config', 'templates')
        logger.info(f"Using package directory for templates: {template_dir}")
    
    # Approach 2: Try using src directory from project_root
    if template_dir is None or not os.path.exists(os.path.dirname(template_dir)):
        project_root = path_config.get('project_root')
        if project_root and os.path.exists(project_root):
            src_dir = os.path.join(project_root, 'src')
            if os.path.exists(src_dir):
                template_dir = os.path.join(src_dir, 'config', 'templates')
                logger.info(f"Using project root src directory for templates: {template_dir}")
    
    # Approach 3: Try common environment paths
    if template_dir is None or not os.path.exists(os.path.dirname(template_dir)):
        # Try RunPod standard location
        if os.path.exists('/workspace/src'):
            template_dir = os.path.join('/workspace', 'src', 'config', 'templates')
            logger.info(f"Using RunPod workspace directory for templates: {template_dir}")
    
    # Approach 4: Last resort - use current directory
    if template_dir is None or not os.path.exists(os.path.dirname(template_dir)):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(current_dir, 'templates')
        logger.info(f"Using current directory for templates: {template_dir}")
    
    # Ensure directory exists
    os.makedirs(template_dir, exist_ok=True)
    
    # Define built-in templates
    templates = [
        # Basic notebook templates
        ExperimentTemplate(
            name="notebook_single_model",
            config={
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
            },
            category="notebook",
            description="Simple notebook template for single model extraction",
            tags=["notebook", "basic", "beginner"]
        ),
        ExperimentTemplate(
            name="notebook_model_comparison",
            config={
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
            },
            category="notebook",
            description="Compare different models on the same task",
            tags=["notebook", "comparison", "intermediate"]
        ),
        ExperimentTemplate(
            name="notebook_prompt_comparison",
            config={
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
            },
            category="notebook",
            description="Compare different prompts on the same model",
            tags=["notebook", "comparison", "prompts", "intermediate"]
        ),
        
        # Advanced notebook templates
        ExperimentTemplate(
            name="notebook_quantization_analysis",
            config={
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
            },
            category="notebook",
            description="Compare quantization strategies for a model",
            tags=["notebook", "advanced", "quantization"]
        ),
        
        # Visualization templates
        ExperimentTemplate(
            name="notebook_visualization_dashboard",
            config={
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
            },
            category="visualization",
            description="Comprehensive experiment visualization dashboard",
            tags=["notebook", "visualization", "dashboard"]
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
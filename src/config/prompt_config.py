"""
Prompt Management System

This module provides a comprehensive system for managing prompts used in
invoice information extraction. It supports:

1. Creating and registering typed prompt configurations
2. Organizing prompts by category and field
3. Formatting prompts for different models
4. Comparing and evaluating prompt performance

The prompt system builds on the base configuration foundation but provides
specialized functionality for prompt experimentation and optimization.
"""

import os
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, Set, Callable

# Import base configuration 
from src.config.base_config import (
    BaseConfig, 
    ConfigurationManager,
    get_config_manager, 
    ExtractionFieldConfig,
    ModelConfig,
    ConfigurationError
)

# Set up logging
logger = logging.getLogger(__name__)


class PromptCategory(Enum):
    """Categories for organizing different types of prompts."""
    BASIC = auto()         # Simple, direct prompts
    DETAILED = auto()      # Prompts with additional context
    POSITIONED = auto()    # Prompts with spatial position information
    FORMATTED = auto()     # Prompts requesting specific output format
    CHAIN_OF_THOUGHT = auto()  # Prompts with step-by-step reasoning
    ZERO_SHOT = auto()     # No examples/context provided
    FEW_SHOT = auto()      # Includes examples
    CONTEXTUAL = auto()    # Prompts with document context
    FALLBACK = auto()      # Alternative approaches if primary fails
    EXPERIMENTAL = auto()  # New approaches being tested
    
    @classmethod
    def from_string(cls, value: str) -> 'PromptCategory':
        """Convert string representation to enum value."""
        try:
            return cls[value.upper()]
        except KeyError:
            # Allow lowercase as well
            for category in cls:
                if category.name.lower() == value.lower():
                    return category
            raise ValueError(f"Unknown prompt category: {value}")
    
    def __str__(self) -> str:
        """String representation of prompt category."""
        return self.name.lower()


@dataclass
class PromptTemplate:
    """
    A template for generating prompts with variable substitution.
    
    This class represents a parameterized prompt template that can be
    filled with specific values to generate a concrete prompt.
    """
    template: str
    variables: Set[str] = field(default_factory=set)
    description: str = ""
    
    def __post_init__(self):
        """Extract variables from the template after initialization."""
        if not self.variables:
            # Extract variables using regex
            self.variables = set(re.findall(r'\{([^{}]+)\}', self.template))
    
    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables.
        
        Args:
            **kwargs: Values for template variables
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValueError: If required variables are missing
        """
        # Check for missing variables
        missing = self.variables - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {', '.join(missing)}")
        
        # Format the template
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt template: {e}")


@dataclass
class PromptConfig(BaseConfig):
    """
    Configuration for a specific prompt.
    
    This class represents a complete prompt configuration including
    metadata, formatting options, and evaluation metrics.
    """
    name: str
    text: str
    field: str  # Which field this prompt is for (e.g., "work_order")
    category: PromptCategory
    description: str = ""
    version: str = "1.0"
    model_specific_formatting: Dict[str, str] = field(default_factory=dict)
    variables: Dict[str, str] = field(default_factory=dict)
    template: Optional[PromptTemplate] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set up the template if text contains variables."""
        if self.template is None and '{' in self.text and '}' in self.text:
            self.template = PromptTemplate(self.text)
    
    def validate(self) -> List[str]:
        """
        Validate the prompt configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.name:
            errors.append("Prompt name is required")
        
        if not self.text:
            errors.append("Prompt text is required")
        
        if not self.field:
            errors.append("Field name is required")
        
        if self.template and self.variables:
            # Check if all template variables have values
            missing_vars = self.template.variables - set(self.variables.keys())
            if missing_vars:
                errors.append(f"Missing values for template variables: {', '.join(missing_vars)}")
        
        return errors
    
    def format_for_model(self, model_name: str, image_placeholder: str = "[IMG]") -> str:
        """
        Format this prompt for a specific model.
        
        Args:
            model_name: Name of the model to format for
            image_placeholder: Placeholder text for image in the prompt
            
        Returns:
            Model-specific formatted prompt
        """
        # Get model configuration
        config_manager = get_config_manager()
        model_config = config_manager.get_model_config(model_name)
        
        if not model_config:
            logger.warning(f"No configuration found for model {model_name}, using default prompt format")
            return self.text
        
        # Check if we have model-specific formatting for this model
        if model_name in self.model_specific_formatting:
            return self.model_specific_formatting[model_name]
        
        # Check if we have model-specific formatting for this model's type
        if model_config.model_type in self.model_specific_formatting:
            return self.model_specific_formatting[model_config.model_type]
        
        # Use the model's default prompt format
        prompt_format = model_config.prompt_format
        
        # Replace format placeholders
        formatted_prompt = prompt_format.replace("{instruction}", self.text)
        formatted_prompt = formatted_prompt.replace("{image}", image_placeholder)
        
        return formatted_prompt
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert prompt configuration to a dictionary.
        
        Returns:
            Dictionary representation of the prompt configuration
        """
        return {
            "name": self.name,
            "text": self.text,
            "field": self.field,
            "category": str(self.category),
            "description": self.description,
            "version": self.version,
            "model_specific_formatting": self.model_specific_formatting,
            "variables": self.variables,
            "metrics": self.metrics,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptConfig':
        """
        Create a prompt configuration from a dictionary.
        
        Args:
            data: Dictionary representation of prompt configuration
            
        Returns:
            PromptConfig instance
        """
        # Convert category string to enum
        category_str = data.get("category", "basic")
        category = PromptCategory.from_string(category_str)
        
        # Create PromptConfig
        return cls(
            name=data.get("name", ""),
            text=data.get("text", ""),
            field=data.get("field", ""),
            category=category,
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            model_specific_formatting=data.get("model_specific_formatting", {}),
            variables=data.get("variables", {}),
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {})
        )


class PromptRegistry:
    """
    Registry for managing prompt configurations.
    
    This class provides a central repository for storing, retrieving,
    and managing prompt configurations across different fields and categories.
    """
    
    def __init__(self):
        """Initialize an empty prompt registry."""
        self._prompts: Dict[str, PromptConfig] = {}
        self._by_field: Dict[str, List[str]] = {}
        self._by_category: Dict[str, List[str]] = {}
    
    def register(self, prompt: PromptConfig) -> None:
        """
        Register a prompt in the registry.
        
        Args:
            prompt: PromptConfig to register
            
        Raises:
            ValueError: If prompt validation fails
        """
        # Validate prompt
        errors = prompt.validate()
        if errors:
            raise ValueError(f"Invalid prompt configuration: {', '.join(errors)}")
        
        # Add to main registry
        self._prompts[prompt.name] = prompt
        
        # Add to field index
        if prompt.field not in self._by_field:
            self._by_field[prompt.field] = []
        if prompt.name not in self._by_field[prompt.field]:
            self._by_field[prompt.field].append(prompt.name)
        
        # Add to category index
        category = str(prompt.category)
        if category not in self._by_category:
            self._by_category[category] = []
        if prompt.name not in self._by_category[category]:
            self._by_category[category].append(prompt.name)
    
    def get(self, name: str) -> Optional[PromptConfig]:
        """
        Get a prompt by name.
        
        Args:
            name: Name of the prompt to retrieve
            
        Returns:
            PromptConfig if found, None otherwise
        """
        return self._prompts.get(name)
    
    def list_prompts(self) -> List[str]:
        """
        Get list of all prompt names.
        
        Returns:
            List of prompt names
        """
        return list(self._prompts.keys())
    
    def list_fields(self) -> List[str]:
        """
        Get list of all fields with prompts.
        
        Returns:
            List of field names
        """
        return list(self._by_field.keys())
    
    def list_categories(self) -> List[str]:
        """
        Get list of all prompt categories.
        
        Returns:
            List of category names
        """
        return list(self._by_category.keys())
    
    def get_by_field(self, field: str) -> List[PromptConfig]:
        """
        Get all prompts for a specific field.
        
        Args:
            field: Field name
            
        Returns:
            List of prompts for the field
        """
        if field not in self._by_field:
            return []
        
        return [self._prompts[name] for name in self._by_field[field]]
    
    def get_by_category(self, category: Union[str, PromptCategory]) -> List[PromptConfig]:
        """
        Get all prompts in a specific category.
        
        Args:
            category: Category name or enum
            
        Returns:
            List of prompts in the category
        """
        if isinstance(category, PromptCategory):
            category = str(category)
            
        if category not in self._by_category:
            return []
        
        return [self._prompts[name] for name in self._by_category[category]]
    
    def get_by_field_and_category(
        self, 
        field: str, 
        category: Union[str, PromptCategory]
    ) -> List[PromptConfig]:
        """
        Get prompts for a specific field and category.
        
        Args:
            field: Field name
            category: Category name or enum
            
        Returns:
            List of matching prompts
        """
        if isinstance(category, PromptCategory):
            category = str(category)
            
        field_prompts = set(self._by_field.get(field, []))
        category_prompts = set(self._by_category.get(category, []))
        
        matching_names = field_prompts.intersection(category_prompts)
        return [self._prompts[name] for name in matching_names]
    
    def load_from_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Load prompts from a YAML file.
        
        Args:
            yaml_path: Path to YAML file
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If YAML is invalid
        """
        import yaml
        
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Prompt YAML file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Process prompts
            prompt_list = data.get("prompts", [])
            for prompt_data in prompt_list:
                # Make sure field is specified
                if "field_to_extract" in prompt_data and "field" not in prompt_data:
                    prompt_data["field"] = prompt_data["field_to_extract"]
                
                # Create and register prompt
                prompt = PromptConfig.from_dict(prompt_data)
                self.register(prompt)
            
            logger.info(f"Loaded {len(prompt_list)} prompts from {yaml_path}")
            
        except Exception as e:
            raise ValueError(f"Error loading prompts from {yaml_path}: {e}")
    
    def load_from_directory(self, directory: Union[str, Path]) -> None:
        """
        Load prompts from all YAML files in a directory.
        
        Args:
            directory: Directory containing prompt YAML files
            
        Raises:
            FileNotFoundError: If directory does not exist
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Prompt directory not found: {directory}")
        
        # Load all YAML files
        files_loaded = 0
        errors = []
        
        for yaml_file in directory.glob("*.yaml"):
            try:
                self.load_from_yaml(yaml_file)
                files_loaded += 1
            except Exception as e:
                errors.append(f"{yaml_file.name}: {str(e)}")
        
        if errors:
            logger.warning(f"Errors loading prompt files: {', '.join(errors)}")
        
        logger.info(f"Loaded prompts from {files_loaded} files in {directory}")


class PromptFormatter:
    """
    Utility for formatting prompts for different models.
    
    This class provides methods for formatting prompts according to
    model-specific requirements and conventions.
    """
    
    @staticmethod
    def format_for_model(
        prompt: Union[str, PromptConfig],
        model_name: str,
        image_placeholder: str = "[IMG]",
        variables: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Format a prompt for a specific model.
        
        Args:
            prompt: Prompt text or PromptConfig
            model_name: Name of the model to format for
            image_placeholder: Placeholder for image in the prompt
            variables: Optional variables to substitute in the prompt
            
        Returns:
            Formatted prompt string
        """
        # If prompt is already a PromptConfig, use its method
        if isinstance(prompt, PromptConfig):
            # Apply variables if needed
            if variables and prompt.template:
                # Combine default variables with provided ones
                all_vars = {**prompt.variables, **(variables or {})}
                prompt_text = prompt.template.format(**all_vars)
                
                # Create a temporary prompt with the formatted text
                temp_prompt = PromptConfig(
                    name=prompt.name,
                    text=prompt_text,
                    field=prompt.field,
                    category=prompt.category,
                    model_specific_formatting=prompt.model_specific_formatting
                )
                return temp_prompt.format_for_model(model_name, image_placeholder)
            
            # Just use the standard formatting
            return prompt.format_for_model(model_name, image_placeholder)
        
        # Get model configuration
        config_manager = get_config_manager()
        model_config = config_manager.get_model_config(model_name)
        
        if not model_config:
            logger.warning(f"No configuration found for model {model_name}, using raw prompt")
            return prompt
        
        # Apply model's prompt format
        prompt_format = model_config.prompt_format
        formatted = prompt_format.replace("{instruction}", prompt)
        formatted = formatted.replace("{image}", image_placeholder)
        
        return formatted
    
    @staticmethod
    def get_model_formatters() -> Dict[str, Callable[[str], str]]:
        """
        Get formatting functions for each model type.
        
        Returns:
            Dictionary mapping model types to formatter functions
        """
        formatters = {}
        
        # Get all model configs
        config_manager = get_config_manager()
        model_configs = config_manager.get_model_configs()
        
        # Create formatter for each model type
        for model_name, model_config in model_configs.items():
            model_type = model_config.model_type
            
            if model_type not in formatters:
                # Create a closure for this model type
                def make_formatter(config):
                    def formatter(prompt_text):
                        formatted = config.prompt_format.replace("{instruction}", prompt_text)
                        formatted = formatted.replace("{image}", config.image_format)
                        return formatted
                    return formatter
                
                formatters[model_type] = make_formatter(model_config)
        
        return formatters


# Global registry instance
_prompt_registry = None

def get_prompt_registry() -> PromptRegistry:
    """
    Get the global prompt registry instance.
    
    Returns:
        Global PromptRegistry instance
    """
    global _prompt_registry
    
    if _prompt_registry is None:
        _prompt_registry = PromptRegistry()
        
        # Try to load prompts from default location
        config_manager = get_config_manager()
        project_root = config_manager.get_project_root()
        prompt_dir = project_root / "configs" / "prompts"
        
        if prompt_dir.exists():
            try:
                _prompt_registry.load_from_directory(prompt_dir)
            except Exception as e:
                logger.warning(f"Error loading default prompts: {e}")
    
    return _prompt_registry


def register_prompt(prompt: PromptConfig) -> None:
    """
    Register a prompt in the global registry.
    
    Args:
        prompt: PromptConfig to register
    """
    registry = get_prompt_registry()
    registry.register(prompt)


def get_prompt(name: str) -> Optional[PromptConfig]:
    """
    Get a prompt by name from the global registry.
    
    Args:
        name: Name of the prompt to retrieve
        
    Returns:
        PromptConfig if found, None otherwise
    """
    registry = get_prompt_registry()
    return registry.get(name)


def get_prompts_by_field(field: str) -> List[PromptConfig]:
    """
    Get all prompts for a specific field from the global registry.
    
    Args:
        field: Field name
        
    Returns:
        List of prompts for the field
    """
    registry = get_prompt_registry()
    return registry.get_by_field(field)


def get_prompts_by_category(category: Union[str, PromptCategory]) -> List[PromptConfig]:
    """
    Get all prompts in a specific category from the global registry.
    
    Args:
        category: Category name or enum
        
    Returns:
        List of prompts in the category
    """
    registry = get_prompt_registry()
    return registry.get_by_category(category)


def format_prompt(
    prompt: Union[str, PromptConfig],
    model_name: str,
    image_placeholder: str = "[IMG]",
    variables: Optional[Dict[str, str]] = None
) -> str:
    """
    Format a prompt for a specific model.
    
    Args:
        prompt: Prompt text or PromptConfig
        model_name: Name of the model to format for
        image_placeholder: Placeholder for image in the prompt
        variables: Optional variables to substitute in the prompt
        
    Returns:
        Formatted prompt string
    """
    return PromptFormatter.format_for_model(prompt, model_name, image_placeholder, variables)


def create_prompt(
    text: str,
    field: str,
    name: Optional[str] = None,
    category: Union[str, PromptCategory] = PromptCategory.BASIC,
    description: str = "",
    register: bool = True
) -> PromptConfig:
    """
    Create a new prompt configuration.
    
    Args:
        text: Prompt text
        field: Field this prompt is for
        name: Optional prompt name (auto-generated if None)
        category: Prompt category
        description: Optional description
        register: Whether to register the prompt in the global registry
        
    Returns:
        Created PromptConfig
    """
    # Convert category string to enum if needed
    if isinstance(category, str):
        category = PromptCategory.from_string(category)
    
    # Generate name if not provided
    if name is None:
        # Create a name based on field and category
        field_part = field.lower().replace(" ", "_")
        category_part = str(category).lower()
        name = f"{category_part}_{field_part}"
        
        # Make sure name is unique
        registry = get_prompt_registry()
        base_name = name
        counter = 1
        
        while registry.get(name) is not None:
            name = f"{base_name}_{counter}"
            counter += 1
    
    # Create prompt
    prompt = PromptConfig(
        name=name,
        text=text,
        field=field,
        category=category,
        description=description
    )
    
    # Register if requested
    if register:
        register_prompt(prompt)
    
    return prompt


def load_prompts_from_yaml(yaml_path: Union[str, Path]) -> None:
    """
    Load prompts from a YAML file into the global registry.
    
    Args:
        yaml_path: Path to YAML file
    """
    registry = get_prompt_registry()
    registry.load_from_yaml(yaml_path)


def load_prompts_from_directory(directory: Union[str, Path]) -> None:
    """
    Load prompts from all YAML files in a directory into the global registry.
    
    Args:
        directory: Directory containing prompt YAML files
    """
    registry = get_prompt_registry()
    registry.load_from_directory(directory)
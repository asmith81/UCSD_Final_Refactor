"""
Prompt Registry

This module provides a central registry for managing, retrieving, and iterating
through different prompts used in the invoice processing system.

The PromptRegistry class stores prompts with their metadata and provides
methods to access them by name, type, or other attributes.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator, Any
from dataclasses import dataclass, field, asdict, fields

@dataclass
class Prompt:
    """
    A dataclass representing a prompt with its metadata.
    
    Attributes:
        name: Unique identifier for the prompt
        text: The actual prompt text
        category: Category of the prompt (basic, detailed, etc.)
        field_to_extract: Which field this prompt is designed to extract
        description: Human-readable description of the prompt
        version: Version of the prompt for tracking changes
        format_instructions: Special formatting instructions for this prompt
        metadata: Additional arbitrary metadata
    """
    name: str
    text: str
    category: str
    field_to_extract: str
    description: str = ""
    version: str = "1.0"
    format_instructions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def format_for_model(self, model_name: str) -> str:
        """
        Format the prompt text for a specific model.
        
        Args:
            model_name: Name of the model to format for
            
        Returns:
            Formatted prompt text
        """
        # Default formatting just returns the prompt text
        # Specific model formatters can be added here or in prompt_utils.py
        if model_name.lower().startswith("pixtral"):
            # Format specifically for Pixtral models with instruction tags
            return f"<s>[INST]{self.text}\n[IMG][/INST]"
        return self.text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prompt to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Prompt':
        """Create a Prompt instance from a dictionary."""
        # Filter out any keys not in Prompt's fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


class PromptRegistry:
    """
    A registry for storing, retrieving, and managing prompts.
    """
    
    def __init__(self):
        """Initialize an empty prompt registry."""
        self._prompts: Dict[str, Prompt] = {}
        self._categories: Dict[str, List[str]] = {}
        self._fields: Dict[str, List[str]] = {}
    
    def register(self, prompt: Prompt) -> None:
        """
        Register a prompt in the registry.
        
        Args:
            prompt: The Prompt object to register
        """
        self._prompts[prompt.name] = prompt
        
        # Update category index
        if prompt.category not in self._categories:
            self._categories[prompt.category] = []
        if prompt.name not in self._categories[prompt.category]:
            self._categories[prompt.category].append(prompt.name)
        
        # Update field index
        if prompt.field_to_extract not in self._fields:
            self._fields[prompt.field_to_extract] = []
        if prompt.name not in self._fields[prompt.field_to_extract]:
            self._fields[prompt.field_to_extract].append(prompt.name)
    
    def get(self, name: str) -> Optional[Prompt]:
        """
        Get a prompt by name.
        
        Args:
            name: Name of the prompt to retrieve
            
        Returns:
            The Prompt object if found, None otherwise
        """
        return self._prompts.get(name)
    
    def list_all(self) -> List[str]:
        """
        List all prompt names in the registry.
        
        Returns:
            List of prompt names
        """
        return list(self._prompts.keys())
    
    def list_categories(self) -> List[str]:
        """
        List all prompt categories in the registry.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def list_fields(self) -> List[str]:
        """
        List all extraction fields in the registry.
        
        Returns:
            List of field names
        """
        return list(self._fields.keys())
    
    def get_by_category(self, category: str) -> List[Prompt]:
        """
        Get all prompts in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of Prompt objects in the category
        """
        if category not in self._categories:
            return []
        
        return [self._prompts[name] for name in self._categories[category]]
    
    def get_by_field(self, field: str) -> List[Prompt]:
        """
        Get all prompts for a specific extraction field.
        
        Args:
            field: Field name
            
        Returns:
            List of Prompt objects for the field
        """
        if field not in self._fields:
            return []
        
        return [self._prompts[name] for name in self._fields[field]]
    
    def iter_prompts(self, 
                    category: Optional[str] = None, 
                    field: Optional[str] = None) -> Iterator[Prompt]:
        """
        Iterate through prompts, optionally filtered by category and/or field.
        
        Args:
            category: Optional category filter
            field: Optional field filter
            
        Yields:
            Prompt objects matching the criteria
        """
        # Get the set of prompts to iterate through
        if category and field:
            # Both filters
            prompts_to_check = set(self._categories.get(category, [])) & set(self._fields.get(field, []))
        elif category:
            # Just category filter
            prompts_to_check = set(self._categories.get(category, []))
        elif field:
            # Just field filter
            prompts_to_check = set(self._fields.get(field, []))
        else:
            # No filters
            prompts_to_check = set(self._prompts.keys())
        
        # Yield each matching prompt
        for name in prompts_to_check:
            yield self._prompts[name]
    
    def load_from_config(self, config_path: Union[str, Path]) -> None:
        """
        Load prompts from a YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Process the prompt configurations
        if 'prompts' in config:
            for prompt_data in config['prompts']:
                # Create and register the prompt
                prompt = Prompt.from_dict(prompt_data)
                self.register(prompt)
    
    def load_from_directory(self, dir_path: Union[str, Path]) -> None:
        """
        Load all YAML prompt configurations from a directory.
        
        Args:
            dir_path: Path to the directory containing YAML files
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        for file_path in dir_path.glob("*.yaml"):
            self.load_from_config(file_path)
    
    def save_to_config(self, config_path: Union[str, Path]) -> None:
        """
        Save all prompts to a YAML configuration file.
        
        Args:
            config_path: Path to save the YAML configuration
        """
        config_path = Path(config_path)
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert prompts to dictionaries for serialization
        config = {
            'prompts': [prompt.to_dict() for prompt in self._prompts.values()]
        }
        
        # Save to YAML
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def __len__(self) -> int:
        """Return the number of prompts in the registry."""
        return len(self._prompts)
    
    def __contains__(self, name: str) -> bool:
        """Check if a prompt name exists in the registry."""
        return name in self._prompts


# Create a singleton instance for global use
_registry = PromptRegistry()

# Public API functions
def get_registry() -> PromptRegistry:
    """Get the global prompt registry instance."""
    global _registry
    return _registry

def get_prompt_registry() -> PromptRegistry:
    """Get the global prompt registry instance (alias for get_registry)."""
    return get_registry()

def register_prompt(prompt: Prompt) -> None:
    """Register a prompt in the global registry."""
    _registry.register(prompt)

def get_prompt(name: str) -> Optional[Prompt]:
    """Get a prompt by name from the global registry."""
    return _registry.get(name)

def get_prompts_by_category(category: str) -> List[Prompt]:
    """Get prompts by category from the global registry."""
    return _registry.get_by_category(category)

def get_prompts_by_field(field: str) -> List[Prompt]:
    """Get prompts by field from the global registry."""
    return _registry.get_by_field(field)

def list_all_prompts() -> List[str]:
    """List all prompt names in the global registry."""
    return _registry.list_all()

def list_prompt_categories() -> List[str]:
    """List all prompt categories in the global registry."""
    return _registry.list_categories()

def load_prompts_from_config(config_path: Union[str, Path]) -> None:
    """Load prompts from a config file into the global registry."""
    _registry.load_from_config(config_path)

def load_all_prompt_configs(config_dir: Union[str, Path] = None) -> None:
    """
    Load all prompt configurations from the standard config directory
    or a specified directory.
    
    Args:
        config_dir: Optional custom config directory path
    """
    if config_dir is None:
        # Try to locate the default config directory
        project_root = os.environ.get('PROJECT_ROOT')
        if project_root:
            config_dir = Path(project_root) / 'configs' / 'prompts'
        else:
            # Attempt to find it relative to this file
            this_dir = Path(__file__).parent
            project_root = this_dir.parent.parent  # src/prompts -> src -> project_root
            config_dir = project_root / 'configs' / 'prompts'
    
    if not Path(config_dir).exists():
        raise FileNotFoundError(f"Prompt config directory not found: {config_dir}")
    
    _registry.load_from_directory(config_dir)


# Initialize with some default prompts based on the successful RunPod notebook
def _init_default_prompts():
    """Initialize the registry with default prompts from the successful RunPod experiment."""
    # Basic prompt that worked in RunPod
    register_prompt(Prompt(
        name="basic_work_order",
        text="Extract the work order number from this invoice image.",
        category="basic",
        field_to_extract="work_order",
        description="Simple direct prompt that worked well in initial RunPod tests."
    ))
    
    # More detailed prompt
    register_prompt(Prompt(
        name="detailed_work_order",
        text="This is an invoice image. Find and extract the work order number from this invoice. The work order number is typically labeled as 'Work Order Number' or 'Numero de Orden'.",
        category="detailed",
        field_to_extract="work_order",
        description="More detailed prompt with field label information."
    ))
    
    # Positioned prompt
    register_prompt(Prompt(
        name="positioned_work_order",
        text="Extract the work order number from this invoice image. The work order number is typically located in the upper portion of the invoice.",
        category="positioned",
        field_to_extract="work_order",
        description="Prompt that includes positional information about the field."
    ))
    
    # Formatted prompt
    register_prompt(Prompt(
        name="formatted_work_order",
        text="Extract the work order number from this invoice. Return only the numeric value without any additional text.",
        category="formatted",
        field_to_extract="work_order",
        description="Prompt that specifies the expected format of the output."
    ))


# Initialize default prompts on module import
_init_default_prompts()
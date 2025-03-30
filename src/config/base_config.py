"""
Configuration Management System

This module provides the core components for managing configuration across the
invoice processing system. It establishes a centralized, type-safe approach to
configuration with consistent validation and error handling.

The configuration system follows these principles:
1. Typed configuration objects for different domains
2. Explicit configuration validation
3. Clear precedence rules for configuration sources
4. Proper error handling with descriptive messages
5. Support for different environments (local, RunPod)
"""

import os
import sys
import json
import yaml
import shutil
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, TypeVar, Generic, Set

# Set up logging
logger = logging.getLogger(__name__)

# Type variable for configuration classes
T = TypeVar('T', bound='BaseConfig')


class ConfigurationError(Exception):
    """Base exception for configuration-related errors."""
    pass


class ConfigValidationError(ConfigurationError):
    """Exception raised when configuration validation fails."""
    def __init__(self, errors: List[str], config_type: str):
        self.errors = errors
        self.config_type = config_type
        error_message = f"Configuration validation failed for {config_type}:\n"
        error_message += "\n".join(f"  - {error}" for error in errors)
        super().__init__(error_message)


class ConfigurationNotFoundError(ConfigurationError):
    """Exception raised when a required configuration file or source is not found."""
    pass


class EnvironmentType(Enum):
    """Type of execution environment."""
    LOCAL = auto()
    RUNPOD = auto()
    CLOUD = auto()
    CUSTOM = auto()
    
    @classmethod
    def from_string(cls, value: str) -> 'EnvironmentType':
        """Convert string representation to enum value."""
        value_map = {
            "local": cls.LOCAL,
            "runpod": cls.RUNPOD,
            "cloud": cls.CLOUD,
            "custom": cls.CUSTOM
        }
        if value.lower() not in value_map:
            raise ValueError(f"Unknown environment: {value}")
        return value_map[value.lower()]
    
    def __str__(self) -> str:
        """String representation of environment type."""
        return self.name.lower()


@dataclass
class ExtractionFieldConfig:
    """Configuration for a field that can be extracted from invoices."""
    name: str  # Internal identifier for the field (e.g., "work_order")
    display_name: str  # Human-readable name (e.g., "Work Order Number")
    description: str = ""  # Description of what this field represents
    data_type: str = "string"  # Data type (string, number, date, etc.)
    is_required: bool = True  # Whether this field is required
    csv_column_name: str = ""  # Column name in the ground truth CSV
    validation_pattern: str = ""  # Regex pattern for validating extraction
    
    # Prompt templates for this field (can be model-specific)
    prompt_templates: Dict[str, str] = field(default_factory=dict)
    
    # Metrics to use for evaluating this field extraction
    metrics: List[str] = field(default_factory=list)
    
    # Additional field-specific settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate the field configuration."""
        errors = []
        
        if not self.name:
            errors.append("Field name is required")
        
        if not self.display_name:
            errors.append("Field display name is required")
        
        if not self.data_type:
            errors.append("Field data type is required")
        elif self.data_type not in ["string", "number", "date", "currency", "boolean"]:
            errors.append(f"Invalid data type: {self.data_type}")
        
        if self.validation_pattern:
            try:
                import re
                re.compile(self.validation_pattern)
            except re.error:
                errors.append(f"Invalid validation pattern: {self.validation_pattern}")
        
        return errors


@dataclass
class ModelConfig:
    """Configuration for a specific model that can be used for extraction."""
    name: str  # Internal identifier for the model
    display_name: str  # Human-readable name
    model_type: str  # Type of model (e.g., "pixtral", "llama", "doctr")
    repo_id: str = ""  # Repository ID for loading (if applicable)
    description: str = ""  # Description of the model
    
    # Hardware requirements
    gpu_required: bool = True
    min_gpu_memory_gb: float = 0.0
    cpu_fallback: bool = False
    
    # Loading configuration
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    torch_dtype: str = "float32"
    
    # Model-specific prompt formatting
    prompt_format: str = "{instruction}"
    image_format: str = "{image}"
    
    # Additional model-specific settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate the model configuration."""
        errors = []
        
        if not self.name:
            errors.append("Model name is required")
        
        if not self.display_name:
            errors.append("Model display name is required")
        
        if not self.model_type:
            errors.append("Model type is required")
        
        if self.gpu_required and self.min_gpu_memory_gb <= 0.0:
            errors.append("GPU memory requirement must be specified for GPU-required models")
        
        valid_dtypes = ["float32", "float16", "bfloat16", "int8", "int4"]
        if self.torch_dtype not in valid_dtypes:
            errors.append(f"Invalid torch dtype: {self.torch_dtype}. Must be one of {valid_dtypes}")
        
        # Check incompatible options
        if self.load_in_8bit and self.load_in_4bit:
            errors.append("Cannot enable both 8-bit and 4-bit quantization")
        
        return errors


class BaseConfig:
    """
    Base configuration class with core functionality.
    
    This class provides core configuration functionality including:
    - Environment detection
    - Path resolution
    - Configuration loading and merging
    - Basic validation
    """
    
    def __init__(
        self, 
        environment: Optional[EnvironmentType] = None,
        project_root: Optional[str] = None,
        config_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize base configuration.
        
        Args:
            environment: Optional environment type override
            project_root: Optional project root path override
            config_dir: Optional configuration directory override
            **kwargs: Additional configuration parameters
        """
        # Initialize core attributes
        self.environment = environment
        self.project_root = project_root
        self.config_dir = config_dir
        self.in_runpod = False
        self.version = "unknown"
        
        # Add any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize configuration
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize configuration with proper setup sequence."""
        # Detect environment and set up paths
        self.environment = self._detect_environment()
        
        # Load configurations in proper order
        self._load_default_config()
        self._load_environment_config()
        self._load_custom_config()
        
        # Validate the configuration
        self._validate()
    
    def _detect_environment(self) -> EnvironmentType:
        """
        Detect the execution environment (local or RunPod).
        
        This method uses various indicators to determine if the code is running
        in a RunPod environment or locally.
        
        Returns:
            EnvironmentType.RUNPOD or EnvironmentType.LOCAL
        """
        # Start by loading path configuration, which will detect project root
        from src.config.path_config import get_path_config
        paths = get_path_config(reset=True)
        
        # Use the path configuration's root
        project_root = paths.get('project_root')
        fallback = "/workspace" if self._is_runpod() else os.getcwd()
        
        # Use the detected project root if available, otherwise fallback
        if project_root and os.path.exists(project_root):
            self.project_root = project_root
        else:
            logger.warning(f"Could not determine project root, using fallback: {fallback}")
            self.project_root = fallback
        
        # Set common paths
        self.config_dir = paths.get('configs_dir')
        self.package_dir = paths.get('package_dir')
        self.data_dir = paths.get('data_dir')
        self.models_dir = paths.get('models_dir')
        self.results_dir = paths.get('results_dir')
        
        # Check for RunPod environment
        self.in_runpod = self._is_runpod()
        if self.in_runpod:
            logger.info("✅ Running in RunPod environment")
            return EnvironmentType.RUNPOD
        else:
            logger.info("💻 Running in local environment")
            return EnvironmentType.LOCAL
    
    def _is_runpod(self) -> bool:
        """
        Check if the code is running in a RunPod environment.
        
        Returns:
            True if running in RunPod, False otherwise
        """
        # Check for RunPod-specific environment variables
        runpod_indicators = ["RUNPOD_POD_ID", "RUNPOD_GPU_COUNT"]
        if any(indicator in os.environ for indicator in runpod_indicators):
            return True
        
        # Check for RunPod-specific directories
        runpod_paths = ["/cache", "/workspace"]
        if all(os.path.exists(path) for path in runpod_paths):
            return True
        
        # Check Docker cgroup (common in container environments)
        try:
            with open('/proc/1/cgroup', 'r') as f:
                if 'docker' in f.read():
                    # This is likely a container, possible RunPod
                    return True
        except (FileNotFoundError, IOError):
            # Not on Linux or not in a container
            pass
        
        return False
    
    def _load_default_config(self) -> None:
        """
        Load default configuration values.
        
        This provides a base set of configuration values that are logical defaults
        for the application.
        """
        # Load baseline configuration values
        base_config = {
            # Base paths - these should be overridden by _detect_environment
            "paths": {
                "base_dir": self.project_root,
                "config_dir": self.config_dir or os.path.join(self.project_root, "configs"),
                "data_dir": os.path.join(self.project_root, "data"),
                "models_dir": os.path.join(self.project_root, "models"),
                "results_dir": os.path.join(self.project_root, "results"),
                "logs_dir": os.path.join(self.project_root, "logs"),
            },
            
            # Logging configuration
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_level": "DEBUG",
                "console_level": "INFO",
            },
            
            # Default extraction fields
            "fields": ["work_order", "cost", "date"],
            
            # Default model configurations
            "models": {
                "default_llm": "gpt-4o",
                "default_ocr": "tesseract",
            },
        }
        
        # Set attributes on self from the base config
        for key, value in base_config.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
    def _load_environment_config(self) -> None:
        """
        Load environment-specific configuration.
        
        This loads configuration specific to the detected environment (LOCAL or RUNPOD).
        """
        # Determine environment config file name
        env_name = str(self.environment).lower()
        env_config_file = os.path.join(
            self.config_dir, 
            "environments", 
            f"{env_name}.yaml"
        )
        
        # Load environment-specific configuration if it exists
        if os.path.exists(env_config_file):
            try:
                with open(env_config_file, 'r') as f:
                    env_config = yaml.safe_load(f)
                
                # Apply environment configuration
                self._apply_config(env_config)
                logger.info(f"Loaded environment configuration from {env_config_file}")
            except Exception as e:
                logger.warning(f"Error loading environment config: {e}")
        else:
            logger.warning(f"Environment config file not found: {env_config_file}")
            
            # Add empty dict to maintain source order
            self.environment_config = {}
    
    def _load_custom_config(self) -> None:
        """
        Load custom configuration from a user-provided file.
        
        This allows for overriding specific settings without modifying the base or
        environment configurations.
        """
        # Check for custom config file
        custom_config_file = os.environ.get(
            "CUSTOM_CONFIG_FILE",
            os.path.join(self.config_dir, "custom.yaml")
        )
        
        # Load custom configuration if it exists
        if os.path.exists(custom_config_file):
            try:
                with open(custom_config_file, 'r') as f:
                    custom_config = yaml.safe_load(f)
                
                # Apply custom configuration
                self._apply_config(custom_config)
                logger.info(f"Loaded custom configuration from {custom_config_file}")
            except Exception as e:
                logger.warning(f"Error loading custom config: {e}")
    
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """
        Apply configuration values to this object.
        
        Args:
            config: Dictionary of configuration values to apply
        """
        if not isinstance(config, dict):
            logger.warning(f"Invalid configuration format: {type(config)}")
            return
        
        # Apply values to attributes
        for key, value in config.items():
            # If this is a dict and we already have a dict attribute with the same name,
            # merge instead of replace
            if isinstance(value, dict) and hasattr(self, key) and isinstance(getattr(self, key), dict):
                current = getattr(self, key)
                current.update(value)
            else:
                setattr(self, key, value)
    
    def _validate(self) -> List[str]:
        """
        Validate the configuration.
        
        Returns:
            List of validation errors or empty list if valid
        """
        errors = []
        
        # Check critical paths
        for path_name in ["project_root", "config_dir", "data_dir"]:
            path = getattr(self, path_name, None)
            if not path or not os.path.exists(path):
                errors.append(f"Critical path '{path_name}' is missing or invalid: {path}")
        
        # Log validation results
        if errors:
            for error in errors:
                logger.warning(f"Configuration error: {error}")
        else:
            logger.info("Configuration validation passed")
        
        return errors

    def validate(self) -> List[str]:
        """
        Validate the configuration and return a list of validation errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        return self._validate()
    
    def is_valid(self) -> bool:
        """
        Check if the configuration is valid.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        return len(self.validate()) == 0
    
    def validate_or_raise(self, config_type: Optional[str] = None) -> None:
        """
        Validate the configuration and raise an exception if invalid.
        
        Args:
            config_type: Optional type name for error messages
            
        Raises:
            ConfigurationError: If the configuration is invalid
        """
        errors = self.validate()
        if errors:
            raise ConfigurationError(
                f"Invalid {config_type or self.__class__.__name__} configuration: {'; '.join(errors)}"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key with dot notation support.
        
        Args:
            key: Configuration key (supports dot notation for nested access)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        # Handle dot notation
        if '.' in key:
            parts = key.split('.')
            curr = self
            
            for part in parts:
                if hasattr(curr, part):
                    curr = getattr(curr, part)
                elif isinstance(curr, dict) and part in curr:
                    curr = curr[part]
                else:
                    return default
            
            return curr
        
        # Direct attribute access
        if hasattr(self, key):
            return getattr(self, key)
        
        return default


class ConfigurationManager:
    """
    Central manager for all configuration in the system.
    
    This class is responsible for:
    1. Detecting the execution environment
    2. Loading configuration from multiple sources
    3. Applying precedence rules to merge configurations
    4. Creating typed configuration objects
    5. Validating configurations
    """
    
    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        config_dir: Optional[Union[str, Path]] = None,
        custom_config_path: Optional[Union[str, Path]] = None,
        environment_override: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the configuration manager.
        
        Args:
            project_root: Root directory of the project (discovered if None)
            config_dir: Directory containing configuration files (default: <project_root>/configs)
            custom_config_path: Path to a custom configuration file to load
            environment_override: Override the detected environment
            config_overrides: Dictionary of configuration overrides
        """
        # Set up basic properties
        self.project_root = self._find_project_root() if project_root is None else Path(project_root)
        self.config_dir = Path(config_dir) if config_dir else self.project_root / "configs"
        
        # Detect or override environment
        self.environment = self._detect_environment() if environment_override is None else EnvironmentType.from_string(environment_override)
        logger.info(f"Using environment: {self.environment}")
        
        # Store configuration sources with their priorities
        self.config_sources: List[ConfigSource] = []
        
        # Load configurations in priority order
        self._load_default_config()       # Priority 0
        self._load_environment_config()   # Priority 10
        
        # Load custom configuration if provided
        if custom_config_path:
            self._load_custom_config(custom_config_path)  # Priority 20
        
        # Apply explicit overrides if provided
        if config_overrides:
            self._apply_overrides(config_overrides)  # Priority 30
        
        # Merge configurations according to priority
        self.merged_config = self._merge_configurations()
        
        # Resolve path variables
        self.merged_config = self._resolve_path_variables(self.merged_config)
        
        logger.debug(f"Initialized ConfigurationManager with {len(self.config_sources)} sources")
        
        # Initialize field and model registries
        self._init_extraction_fields()
        self._init_model_configs()
    
    def _find_project_root(self) -> Path:
        """
        Find the project root directory.
        
        This method looks for standard project markers like .git, setup.py, or README.md
        to identify the project root directory.
        
        Returns:
            Path to the project root directory
        
        Raises:
            ConfigurationError: If project root cannot be determined
        """
        # Start with the current file's directory
        current_dir = Path(__file__).resolve().parent
        
        # Check if PROJECT_ROOT environment variable is set
        if "PROJECT_ROOT" in os.environ:
            project_root = Path(os.environ["PROJECT_ROOT"])
            if project_root.exists():
                logger.info(f"Using PROJECT_ROOT environment variable: {project_root}")
                return project_root
        
        # Check for RunPod environment
        if self._is_runpod():
            # In RunPod, the standard location is /workspace
            if Path("/workspace").exists():
                logger.info("RunPod environment detected, using /workspace as project root")
                # Set the PROJECT_ROOT environment variable for future use
                os.environ["PROJECT_ROOT"] = "/workspace"
                return Path("/workspace")
        
        # Look for common project markers
        markers = [".git", "setup.py", "requirements.txt", "README.md"]
        
        # Walk up the directory tree looking for markers
        search_dir = current_dir
        while search_dir != search_dir.parent:  # Stop at the filesystem root
            for marker in markers:
                if (search_dir / marker).exists():
                    logger.info(f"Project root marker '{marker}' found at: {search_dir}")
                    return search_dir
            search_dir = search_dir.parent
        
        # If we get here, we couldn't find the project root
        # Fallback to two directories up from this file
        fallback = current_dir.parent.parent
        logger.warning(f"Could not determine project root, using fallback: {fallback}")
        return fallback
    
    def _is_runpod(self) -> bool:
        """
        Check if the code is running in a RunPod environment.
        
        Returns:
            True if running in RunPod, False otherwise
        """
        # Check for RunPod-specific environment variables
        runpod_indicators = ["RUNPOD_POD_ID", "RUNPOD_GPU_COUNT"]
        if any(indicator in os.environ for indicator in runpod_indicators):
            return True
        
        # Check for RunPod-specific directories
        runpod_paths = ["/cache", "/workspace"]
        if all(os.path.exists(path) for path in runpod_paths):
            return True
        
        # Check Docker cgroup (common in container environments)
        try:
            with open('/proc/1/cgroup', 'r') as f:
                if 'docker' in f.read():
                    # This is likely a container, possible RunPod
                    return True
        except (FileNotFoundError, IOError):
            # Not on Linux or not in a container
            pass
        
        return False
    
    def _detect_environment(self) -> EnvironmentType:
        """
        Detect the execution environment (local or RunPod).
        
        This method uses various indicators to determine if the code is running
        in a RunPod environment or locally.
        
        Returns:
            EnvironmentType.RUNPOD or EnvironmentType.LOCAL
        """
        # Start by loading path configuration, which will detect project root
        from src.config.path_config import get_path_config
        paths = get_path_config(reset=True)
        
        # Use the path configuration's root
        project_root = paths.get('project_root')
        fallback = "/workspace" if self._is_runpod() else os.getcwd()
        
        # Use the detected project root if available, otherwise fallback
        if project_root and os.path.exists(project_root):
            self.project_root = project_root
        else:
            logger.warning(f"Could not determine project root, using fallback: {fallback}")
            self.project_root = fallback
        
        # Set common paths
        self.config_dir = paths.get('configs_dir')
        self.package_dir = paths.get('package_dir')
        self.data_dir = paths.get('data_dir')
        self.models_dir = paths.get('models_dir')
        self.results_dir = paths.get('results_dir')
        
        # Check for RunPod environment
        self.in_runpod = self._is_runpod()
        if self.in_runpod:
            logger.info("✅ Running in RunPod environment")
            return EnvironmentType.RUNPOD
        else:
            logger.info("💻 Running in local environment")
        return EnvironmentType.LOCAL
    
    def _load_default_config(self) -> None:
        """
        Load default configuration values.
        
        This provides a base set of configuration values that are logical defaults
        for the application.
        """
        # Load baseline configuration values
        base_config = {
            # Base paths - these should be overridden by _detect_environment
            "paths": {
                "base_dir": self.project_root,
                "config_dir": self.config_dir or os.path.join(self.project_root, "configs"),
                "data_dir": os.path.join(self.project_root, "data"),
                "models_dir": os.path.join(self.project_root, "models"),
                "results_dir": os.path.join(self.project_root, "results"),
                "logs_dir": os.path.join(self.project_root, "logs"),
            },
            
            # Logging configuration
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_level": "DEBUG",
                "console_level": "INFO",
            },
            
            # Default extraction fields
            "fields": ["work_order", "cost", "date"],
            
            # Default model configurations
            "models": {
                "default_llm": "gpt-4o",
                "default_ocr": "tesseract",
            },
        }
        
        # Set attributes on self from the base config
        for key, value in base_config.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
    def _load_environment_config(self) -> None:
        """
        Load environment-specific configuration.
        
        This loads configuration specific to the detected environment (LOCAL or RUNPOD).
        """
        # Determine environment config file name
        env_name = str(self.environment).lower()
        env_config_file = os.path.join(
            self.config_dir, 
            "environments", 
            f"{env_name}.yaml"
        )
        
        # Load environment-specific configuration if it exists
        if os.path.exists(env_config_file):
            try:
                with open(env_config_file, 'r') as f:
                    env_config = yaml.safe_load(f)
            
                # Apply environment configuration
                self._apply_config(env_config)
                logger.info(f"Loaded environment configuration from {env_config_file}")
            except Exception as e:
                logger.warning(f"Error loading environment config: {e}")
            else:
                logger.warning(f"Environment config file not found: {env_config_file}")
            
            # Add empty dict to maintain source order
            self.environment_config = {}
    
    def _load_custom_config(self, config_path: Union[str, Path]) -> None:
        """
        Load configuration from a custom YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Raises:
            ConfigurationNotFoundError: If the configuration file is not found
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
            
            # Add as custom configuration (higher priority than environment)
            self.config_sources.append(ConfigSource(
                name="custom_config",
                priority=20,
                data=custom_config or {}
            ))
            
            logger.info(f"Loaded custom configuration from {config_path}")
        
        except Exception as e:
            logger.error(f"Error loading custom configuration: {e}")
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}")
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Apply explicit configuration overrides.
        
        Args:
            overrides: Dictionary of configuration overrides
        """
        # Convert overrides to nested structure if dot notation is used
        structured_overrides = {}
        
        for key, value in overrides.items():
            if '.' in key:
                # Handle dot notation (e.g., "paths.data_dir")
                parts = key.split('.')
                current = structured_overrides
                
                # Navigate to the correct nested dictionary
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value
                current[parts[-1]] = value
            else:
                # Simple key
                structured_overrides[key] = value
        
        # Add as highest priority source
        self.config_sources.append(ConfigSource(
            name="overrides",
            priority=30,
            data=structured_overrides
        ))
        
        logger.debug(f"Applied {len(overrides)} configuration overrides")
    
    def _merge_configurations(self) -> Dict[str, Any]:
        """
        Merge configuration sources according to priority.
        
        This method combines all configuration sources into a single configuration
        dictionary, with higher priority sources overriding lower priority ones.
        
        Returns:
            Merged configuration dictionary
        """
        # Sort sources by priority
        sorted_sources = sorted(self.config_sources, key=lambda source: source.priority)
        
        # Start with an empty configuration
        merged = {}
        
        # Merge each source in priority order
        for source in sorted_sources:
            self._deep_update(merged, source.data)
        
        return merged
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a nested dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with updates
        """
        for key, value in source.items():
            # If both target and source have a dictionary for this key, recurse
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                # Otherwise replace or add the value
                target[key] = value
    
    def _resolve_path_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve path variables in the configuration.
        
        This method replaces ${PROJECT_ROOT} and other variables in path strings
        with their actual values.
        
        Args:
            config: Configuration dictionary to process
            
        Returns:
            Processed configuration with variables resolved
        """
        def _process_value(value):
            if isinstance(value, str):
                # Replace ${PROJECT_ROOT} with actual project root
                if "${PROJECT_ROOT}" in value:
                    return value.replace("${PROJECT_ROOT}", str(self.project_root))
                return value
            elif isinstance(value, dict):
                return {k: _process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_process_value(item) for item in value]
            return value
        
        return _process_value(config)
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "paths.data_dir")
            default: Default value to return if the key is not found
            
        Returns:
            Configuration value or default if not found
        """
        # Start from the merged configuration
        config = self.merged_config
        
        # Split the key path
        keys = key_path.split('.')
        
        # Navigate the nested dictionaries
        for key in keys:
            if not isinstance(config, dict) or key not in config:
                return default
            config = config[key]
        
        return config
    
    def set_config_value(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value
            value: Value to set
        """
        # Convert to override format
        overrides = {key_path: value}
        
        # Apply the override
        self._apply_overrides(overrides)
        
        # Update merged configuration
        self.merged_config = self._merge_configurations()
        self.merged_config = self._resolve_path_variables(self.merged_config)
        
        # Update field and model registries if needed
        if key_path.startswith("extraction_fields"):
            self._init_extraction_fields()
        elif key_path.startswith("models"):
            self._init_model_configs()
    
    def get_environment(self) -> EnvironmentType:
        """
        Get the current environment type.
        
        Returns:
            Current environment type (LOCAL or RUNPOD)
        """
        return self.environment
    
    def get_project_root(self) -> Path:
        """
        Get the project root directory.
        
        Returns:
            Path to the project root directory
        """
        return self.project_root
    
    def _init_extraction_fields(self) -> None:
        """Initialize extraction field configurations."""
        self.extraction_fields: Dict[str, ExtractionFieldConfig] = {}
        
        # Get extraction field configurations
        fields_config = self.merged_config.get("extraction_fields", {})
        
        # Create typed field configurations
        for field_name, field_data in fields_config.items():
            # Ensure name is included
            if "name" not in field_data:
                field_data["name"] = field_name
            
            # Create field configuration
            field_config = ExtractionFieldConfig(
                name=field_data.get("name", field_name),
                display_name=field_data.get("display_name", field_name.replace("_", " ").title()),
                description=field_data.get("description", ""),
                data_type=field_data.get("data_type", "string"),
                is_required=field_data.get("is_required", True),
                csv_column_name=field_data.get("csv_column_name", ""),
                validation_pattern=field_data.get("validation_pattern", ""),
                prompt_templates=field_data.get("prompt_templates", {}),
                metrics=field_data.get("metrics", []),
                settings=field_data.get("settings", {})
            )
            
            # Validate the field configuration
            errors = field_config.validate()
            if errors:
                logger.warning(f"Invalid extraction field configuration for {field_name}: {errors}")
            
            # Add to registry
            self.extraction_fields[field_name] = field_config
        
        logger.info(f"Initialized {len(self.extraction_fields)} extraction field configurations")
    
    def _init_model_configs(self) -> None:
        """Initialize model configurations."""
        self.model_configs: Dict[str, ModelConfig] = {}
        
        # Get model configurations
        models_config = self.merged_config.get("models", {})
        
        # Create typed model configurations
        for model_name, model_data in models_config.items():
            # Ensure name is included
            if "name" not in model_data:
                model_data["name"] = model_name
            
            # Create model configuration
            model_config = ModelConfig(
                name=model_data.get("name", model_name),
                display_name=model_data.get("display_name", model_name),
                model_type=model_data.get("model_type", ""),
                repo_id=model_data.get("repo_id", ""),
                description=model_data.get("description", ""),
                gpu_required=model_data.get("gpu_required", True),
                min_gpu_memory_gb=model_data.get("min_gpu_memory_gb", 0.0),
                cpu_fallback=model_data.get("cpu_fallback", False),
                load_in_8bit=model_data.get("load_in_8bit", False),
                load_in_4bit=model_data.get("load_in_4bit", False),
                torch_dtype=model_data.get("torch_dtype", "float32"),
                prompt_format=model_data.get("prompt_format", "{instruction}"),
                image_format=model_data.get("image_format", "{image}"),
                settings=model_data.get("settings", {})
            )
            
            # Validate the model configuration
            errors = model_config.validate()
            if errors:
                logger.warning(f"Invalid model configuration for {model_name}: {errors}")
            
            # Add to registry
            self.model_configs[model_name] = model_config
        
        logger.info(f"Initialized {len(self.model_configs)} model configurations")
    
    def get_extraction_field(self, field_name: str) -> Optional[ExtractionFieldConfig]:
        """
        Get configuration for a specific extraction field.
        
        Args:
            field_name: Name of the extraction field
            
        Returns:
            ExtractionFieldConfig if found, None otherwise
        """
        return self.extraction_fields.get(field_name)
    
    def get_extraction_fields(self) -> Dict[str, ExtractionFieldConfig]:
        """
        Get all extraction field configurations.
        
        Returns:
            Dictionary mapping field names to ExtractionFieldConfig objects
        """
        return self.extraction_fields
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfig if found, None otherwise
        """
        return self.model_configs.get(model_name)
    
    def get_model_configs(self) -> Dict[str, ModelConfig]:
        """
        Get all model configurations.
        
        Returns:
            Dictionary mapping model names to ModelConfig objects
        """
        return self.model_configs
    
    def get_available_extraction_fields(self) -> List[str]:
        """
        Get list of available extraction field names.
        
        Returns:
            List of extraction field names
        """
        return list(self.extraction_fields.keys())
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List of model names
        """
        return list(self.model_configs.keys())
    
    def validate_all_configs(self) -> Dict[str, List[str]]:
        """
        Validate all configuration objects.
        
        Returns:
            Dictionary mapping configuration types to lists of validation errors
        """
        validation_results = {}
        
        # Validate extraction field configurations
        field_errors = {}
        for field_name, field_config in self.extraction_fields.items():
            errors = field_config.validate()
            if errors:
                field_errors[field_name] = errors
        
        if field_errors:
            validation_results["extraction_fields"] = field_errors
        
        # Validate model configurations
        model_errors = {}
        for model_name, model_config in self.model_configs.items():
            errors = model_config.validate()
            if errors:
                model_errors[model_name] = errors
        
        if model_errors:
            validation_results["models"] = model_errors
        
        return validation_results
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return self.merged_config
    
    def get_notebook_config_summary(self) -> Dict[str, Any]:
        """
        Generate a configuration summary suitable for use in notebooks.
        
        Returns:
            Dictionary with configuration summary formatted for notebook display
        """
        import platform
        import sys
        import shutil
        import os
        
        # Start with basic environment info
        summary = {
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "in_notebook": True,
                "in_runpod": self.environment == EnvironmentType.RUNPOD
            },
            "paths": {
                "project_root": str(self.project_root),
                "data_dir": self.get_config_value("paths.data_dir", str(self.project_root / "data")),
                "models_dir": self.get_config_value("paths.models_dir", str(self.project_root / "models")),
                "results_dir": self.get_config_value("paths.results_dir", str(self.project_root / "results")),
                "configs_dir": self.get_config_value("paths.config_dir", str(self.project_root / "configs")),
                "working_dir": os.getcwd()
            },
            "gpu": {
                "available": False,
                "device_name": "None",
                "memory_gb": 0.0,
                "cuda_version": "None"
            },
            "packages": {},
            "disk_space": {"total_gb": 0, "free_gb": 0, "used_gb": 0}
        }
        
        # Check if GPU is available
        try:
            import torch
            if torch.cuda.is_available():
                summary["gpu"] = {
                    "available": True,
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    "cuda_version": torch.version.cuda
                }
        except ImportError:
            summary["gpu"]["available"] = "unknown (torch not installed)"
        
        # Get package versions
        try:
            import pkg_resources
            important_packages = [
                "torch", "numpy", "pandas", "matplotlib", 
                "transformers", "pillow", "tqdm", "pyyaml"
            ]
            
            for package in important_packages:
                try:
                    version = pkg_resources.get_distribution(package).version
                    summary["packages"][package] = version
                except:
                    summary["packages"][package] = "not installed"
        except ImportError:
            pass
        
        # Get disk space information
        try:
            total, used, free = shutil.disk_usage("/")
            summary["disk_space"] = {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3)
            }
        except:
            pass
        
        return summary
    
    def export_notebook_config(self, export_path: Optional[str] = None) -> str:
        """
        Export notebook configuration summary to a JSON file.
        
        Args:
            export_path: Path to export the summary (if None, a default path is used)
        
        Returns:
            Path to the exported file
        """
        import json
        import time
        import os
        from pathlib import Path
        
        # Get the config summary
        summary = self.get_notebook_config_summary()
        
        if export_path is None:
            # Use results directory if configured, otherwise use project root
            results_dir = self.get_config_value("paths.results_dir", str(self.project_root / "results"))
            os.makedirs(results_dir, exist_ok=True)
            
            # Create a timestamped filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(results_dir, f"config_summary_{timestamp}.json")
        
        # Export as JSON
        with open(export_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Configuration summary exported to: {export_path}")
        return export_path

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """
        Apply configuration values to this object.
        
        Args:
            config: Dictionary of configuration values to apply
        """
        if not isinstance(config, dict):
            logger.warning(f"Invalid configuration format: {type(config)}")
            return
        
        # Apply values to attributes
        for key, value in config.items():
            # If this is a dict and we already have a dict attribute with the same name,
            # merge instead of replace
            if isinstance(value, dict) and hasattr(self, key) and isinstance(getattr(self, key), dict):
                current = getattr(self, key)
                current.update(value)
            else:
                setattr(self, key, value)

    def _validate(self) -> List[str]:
        """
        Validate the configuration.
        
        Returns:
            List of validation errors or empty list if valid
        """
        errors = []
        
        # Check critical paths
        for path_name in ["project_root", "config_dir", "data_dir"]:
            path = getattr(self, path_name, None)
            if not path or not os.path.exists(path):
                errors.append(f"Critical path '{path_name}' is missing or invalid: {path}")
        
        # Log validation results
        if errors:
            for error in errors:
                logger.warning(f"Configuration error: {error}")
        else:
            logger.info("Configuration validation passed")
        
        return errors


# Create a singleton instance
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager() -> ConfigurationManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigurationManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    
    return _config_manager

def get_environment_config() -> Any:
    """
    Get the global environment configuration.
    
    This imports and returns the result of get_environment_config from the
    environment_config module, avoiding circular imports.
    
    Returns:
        EnvironmentConfig instance
    """
    try:
        from src.config.environment_config import get_environment_config as get_env_config
        return get_env_config()
    except ImportError as e:
        logger.warning(f"Failed to import environment config: {e}")
        # Return a BaseConfig instance as a fallback
        return BaseConfig()
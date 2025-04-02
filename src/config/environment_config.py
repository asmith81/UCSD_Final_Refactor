"""
Environment Configuration

This module provides configuration classes and utilities for managing
environment-specific settings and detecting the execution environment.
"""

import os
import sys
import logging
import platform
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import json
import psutil
import torch
import gc

# Import path utilities
from src.config.path_utils import detect_project_root, get_workspace_root, ensure_directory

# Set up logging
logger = logging.getLogger(__name__)

# Remove circular import
# from src.config.path_config import get_path, get_path_config

@dataclass
class EnvironmentConfig:
    """
    Configuration for the execution environment.
    
    Stores settings specific to the environment where the code is running,
    such as RunPod, local machine, etc.
    """
    
    # Environment type
    env_type: str = "local"
    
    # Environment name (for display)
    name: str = "Local Environment"
    
    # Environment description
    description: str = "Default local execution environment"
    
    # Path configuration
    paths: Dict[str, str] = field(default_factory=dict)
    
    # GPU settings
    gpu_settings: Dict[str, Any] = field(default_factory=dict)
    
    # API keys and credentials
    credentials: Dict[str, str] = field(default_factory=dict)
    
    # Memory limits
    memory_limits: Dict[str, int] = field(default_factory=dict)
    
    # Runtime settings
    runtime: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize paths if not provided."""
        if not self.paths:
            # Instead of using get_path_config(), detect project root directly
            project_root = self._detect_project_root()
            self.paths = self._create_default_paths(project_root)
            
        # Initialize default GPU settings if not provided
        if not self.gpu_settings:
            self.gpu_settings = {
                "use_gpu": True,
                "device": "cuda" if self._is_cuda_available() else "cpu",
                "precision": "float16",
                "memory_efficient": True,
            }
    
    def _detect_project_root(self) -> Path:
        """Detect project root directly without depending on path_config."""
        try:
            # Check for RunPod environment
            if self._is_runpod_env():
                logger.info("RunPod environment detected, using /workspace as project root")
                return Path("/workspace")
                
            # Check environment variable
            if "PROJECT_ROOT" in os.environ:
                root = Path(os.environ["PROJECT_ROOT"])
                if root.exists():
                    logger.info(f"Using PROJECT_ROOT environment variable: {root}")
                    return root
                    
            # Start with this file's location
            current_file = Path(__file__).resolve()
            current_dir = current_file.parent  # /src/config
            
            # Try to go up to find project root
            if current_dir.name == "config" and current_dir.parent.name == "src":
                project_root = current_dir.parent.parent  # Go up from /src/config to project root
                logger.info(f"Using detected project root from file location: {project_root}")
                return project_root
                
            # Fall back to current working directory
            cwd = Path(os.getcwd())
            logger.info(f"Could not determine project root, using current working directory: {cwd}")
            return cwd
        except Exception as e:
            logger.warning(f"Error detecting project root: {e}")
            logger.warning("Could not determine project root, using fallback: /workspace")
            return Path("/workspace")
    
    def _create_default_paths(self, project_root: Path) -> Dict[str, str]:
        """Create default paths dictionary based on project root."""
        return {
            'project_root': str(project_root),
            'src_dir': str(project_root / "src"),
            'data_dir': str(project_root / "data"),
            'models_dir': str(project_root / "models"),
            'config_dir': str(project_root / "configs"),
            'configs_dir': str(project_root / "configs"),  # Alias
            'results_dir': str(project_root / "results"),
            'logs_dir': str(project_root / "logs"),
            'cache_dir': str(project_root / "cache"),
        }
    
    def _is_runpod_env(self) -> bool:
        """Check if running in RunPod environment."""
        if any(var in os.environ for var in ["RUNPOD_POD_ID", "RUNPOD_API_KEY"]):
            return True
        if Path("/workspace").exists() and Path("/runpod").exists():
            return True
        return False
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
            
    def is_runpod(self) -> bool:
        """Check if running in a RunPod environment."""
        return self.env_type == "runpod"
    
    def is_local(self) -> bool:
        """Check if running in a local environment."""
        return self.env_type == "local"
    
    def is_colab(self) -> bool:
        """Check if running in Google Colab."""
        return self.env_type == "colab"
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key path.
        
        Args:
            key: Key path in dot notation (e.g., 'gpu_settings.device')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split('.')
        value = self
        
        try:
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value
        except Exception:
            return default
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "env_type": self.env_type,
            "name": self.name,
            "description": self.description,
            "paths": self.paths,
            "gpu_settings": self.gpu_settings,
            "credentials": self.credentials,
            "memory_limits": self.memory_limits,
            "runtime": self.runtime,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentConfig':
        """Create configuration from dictionary."""
        return cls(
            env_type=data.get("env_type", "local"),
            name=data.get("name", "Local Environment"),
            description=data.get("description", ""),
            paths=data.get("paths", {}),
            gpu_settings=data.get("gpu_settings", {}),
            credentials=data.get("credentials", {}),
            memory_limits=data.get("memory_limits", {}),
            runtime=data.get("runtime", {}),
        )
        
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'EnvironmentConfig':
        """Load configuration from YAML file."""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            if not isinstance(data, dict):
                logger.warning(f"Invalid config format in {filepath}, using defaults")
                return cls()
                
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"Error loading config from {filepath}: {e}")
            return cls()
            
    def save_to_file(self, filepath: Union[str, Path]) -> bool:
        """Save configuration to YAML file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            
            return True
        except Exception as e:
            logger.warning(f"Error saving config to {filepath}: {e}")
            return False

    def get_path(self, key: str, default: Any = None) -> Any:
        """
        Get a path value by key for API compatibility with PathConfig.
        
        Args:
            key: Path key to get value for
            default: Default value if key not found
            
        Returns:
            Path value or default if not found
        """
        if key in self.paths:
            return self.paths[key]
        return default

# Global instance
_env_config = None

def get_environment_config(env_type: str = "auto") -> EnvironmentConfig:
    """
    Get the environment configuration.
    
    Args:
        env_type: Environment type to use or 'auto' to detect
    
    Returns:
        EnvironmentConfig instance
    """
    global _env_config
    
    if _env_config is not None:
        return _env_config
    
    # Determine environment type if auto
    if env_type == "auto":
        env_type = _detect_environment_type()
    
    # Try to load config from file
    config_path = _get_environment_config_path()
    if config_path and os.path.exists(config_path):
        _env_config = EnvironmentConfig.load_from_file(config_path)
    else:
        # Create default config for the environment type
        _env_config = _create_default_config(env_type)
        
        # Attempt to save default config
        if config_path:
            _env_config.save_to_file(config_path)
    
    return _env_config

def _detect_environment_type() -> str:
    """
    Detect the current environment type.
    
    Returns:
        Environment type string ('runpod', 'colab', 'local', etc.)
    """
    # Check for RunPod
    if any(var in os.environ for var in ["RUNPOD_POD_ID", "RUNPOD_API_KEY"]):
        return "runpod"
    
    # Check for Google Colab
    try:
        import google.colab
        return "colab"
    except ImportError:
        pass
    
    # Check for Jupyter
    if 'ipykernel' in sys.modules:
        return "jupyter"
    
    # Default to local
    return "local"

def _get_environment_config_path() -> Path:
    """
    Get the path to the environment configuration file.
    
    Returns:
        Path to the environment configuration file
    """
    # First try to get from environment variable
    if "ENV_CONFIG_PATH" in os.environ:
        config_path = Path(os.environ["ENV_CONFIG_PATH"])
        if config_path.exists():
            return config_path
    
    # Then try to find in standard locations
    project_root = detect_project_root()
    config_paths = [
        project_root / "configs" / "environment.yaml",
        project_root / "configs" / "environment.yml",
        project_root / "configs" / "environment.json"
    ]
    
    for path in config_paths:
        if path.exists():
            return path
    
    # If no config found, create a default one
    default_config_path = project_root / "configs" / "environment.yaml"
    ensure_directory(default_config_path.parent)
    
    # Create default configuration
    default_config = {
        "environment": {
            "type": "local",
            "gpu": {
                "enabled": True,
                "memory_limit": 0.9,  # Use 90% of available GPU memory
                "precision": "float16"
            },
            "cpu": {
                "threads": psutil.cpu_count(),
                "memory_limit": 0.8  # Use 80% of available RAM
            }
        }
    }
    
    with open(default_config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    return default_config_path

def _create_default_config(env_type: str) -> EnvironmentConfig:
    """
    Create a default configuration for the given environment type.
    
    Args:
        env_type: Environment type
        
    Returns:
        Default EnvironmentConfig
    """
    # Detect project root
    project_root = _detect_project_root()
    
    if env_type == "runpod":
        return EnvironmentConfig(
            env_type="runpod",
            name="RunPod Environment",
            description="GPU cloud environment on RunPod",
            paths={
                'project_root': str(project_root),
                'src_dir': str(project_root / "src"),
                'data_dir': str(project_root / "data"),
                'models_dir': str(project_root / "models"),
                'config_dir': str(project_root / "configs"),
                'configs_dir': str(project_root / "configs"),  # Alias
                'results_dir': str(project_root / "results"),
                'logs_dir': str(project_root / "logs"),
                'cache_dir': "/cache"  # RunPod specific cache location
            },
            gpu_settings={
                "use_gpu": True,
                "device": "cuda", 
                "precision": "float16",
                "memory_efficient": True
            },
            memory_limits={
                "max_batch_size": 8,
                "max_parallel_processes": 2
            }
        )
    else:
        # Default local environment
        return EnvironmentConfig(
            env_type="local",
            name="Local Environment",
            description="Local development environment",
            # Empty paths - will be populated in __post_init__
            gpu_settings={
                "use_gpu": _is_cuda_available(),
                "device": "cuda" if _is_cuda_available() else "cpu",
                "precision": "float32",
                "memory_efficient": False
            }
        )

def _is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
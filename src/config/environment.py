"""
Environment configuration management for machine learning experiments.

This module provides functionality to:
1. Detect the current execution environment (local or RunPod)
2. Load appropriate configuration settings
3. Validate environment requirements
4. Provide a consistent interface for accessing configuration

The configuration system supports environment-specific settings while
maintaining a consistent API across different execution contexts.
"""

import os
import yaml
import logging
import platform
import torch
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

class EnvironmentConfig:
    """
    Manages environment-specific configuration settings.
    
    This class handles detection of the current environment, loading
    appropriate configuration files, and providing access to settings.
    """
    
    def __init__(self, config_dir: Optional[str] = None, environment: Optional[str] = None):
        """
        Initialize the environment configuration.
        
        Args:
            config_dir: Directory containing environment YAML files
            environment: Force a specific environment instead of auto-detection
        """
        self.project_root = self._find_project_root()
        
        # Set configuration directory
        if config_dir is None:
            self.config_dir = os.path.join(self.project_root, "configs", "environments")
        else:
            self.config_dir = config_dir
            
        # Detect or set environment
        if environment is not None:
            self.environment = environment
        else:
            self.environment = self._detect_environment()
            
        logger.info(f"Detected environment: {self.environment}")
        
        # Load configuration
        self.config = self._load_config()
        
        # Apply environment-specific settings
        self._setup_environment()
        
    def _find_project_root(self) -> str:
        """
        Find the project root directory based on current file location.
        
        Returns:
            Absolute path to the project root directory
        """
        # Start from the current file and go up until we find a marker
        current_path = Path(__file__).resolve().parent
        
        # Look for markers indicating the project root
        markers = [".git", "setup.py", "requirements.txt", "README.md"]
        
        while current_path != current_path.parent:
            for marker in markers:
                if (current_path / marker).exists():
                    return str(current_path)
            current_path = current_path.parent
            
        # If we can't find a marker, use the parent of the config directory
        return str(Path(__file__).resolve().parent.parent.parent)
    
    def _detect_environment(self) -> str:
        """
        Detect which environment we're running in.
        
        Returns:
            String identifier for the environment ("local" or "runpod")
        """
        # Method 1: Check for RunPod environment variables
        runpod_env_vars = ["RUNPOD_POD_ID", "RUNPOD_GPU_COUNT", "RUNPOD_API_KEY"]
        if any(os.environ.get(var) is not None for var in runpod_env_vars):
            logger.info("RunPod environment detected via environment variables")
            return "runpod"
        
        # Method 2: Check if we're in a Docker container (common on RunPod)
        try:
            with open('/proc/1/cgroup', 'r') as f:
                if 'docker' in f.read():
                    logger.info("Docker container detected, likely RunPod")
                    return "runpod"
        except (IOError, FileNotFoundError):
            pass  # Not in a Linux environment or not in a container
            
        # Method 3: Check for CUDA availability and GPU type
        if torch.cuda.is_available():
            try:
                # Get GPU name
                gpu_name = torch.cuda.get_device_name(0).lower()
                logger.info(f"Detected GPU: {gpu_name}")
                
                # Common RunPod GPU types
                runpod_gpus = ["a100", "a6000", "v100", "a40", "a10", "rtx", "tesla", "quadro"]
                if any(gpu_type in gpu_name for gpu_type in runpod_gpus):
                    logger.info(f"RunPod environment likely based on GPU type: {gpu_name}")
                    return "runpod"
                
                # Check CUDA version - RunPod often uses recent CUDA versions
                if hasattr(torch.version, 'cuda') and torch.version.cuda:
                    cuda_version = float(torch.version.cuda.split('.')[0])
                    if cuda_version >= 11.0:
                        logger.info(f"Modern CUDA version detected ({torch.version.cuda}), possibly RunPod")
                        # This alone isn't enough to determine RunPod, but it's a hint
            except Exception as e:
                logger.warning(f"Error checking GPU information: {e}")
        
        # Method 4: Check system memory profile (RunPod instances often have high RAM)
        try:
            import psutil
            total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB
            if total_memory > 30:  # RunPod instances often have >30GB RAM
                logger.info(f"High memory system detected ({total_memory:.1f}GB), possibly RunPod")
                # Again, this alone isn't definitive
        except (ImportError, Exception):
            pass
        
        # Check for any RunPod-specific paths
        if os.path.exists('/cache') or os.path.exists('/workspace'):
            logger.info("RunPod-specific directories detected")
            return "runpod"
            
        # Default to local environment
        logger.info("Local environment detected (no RunPod indicators found)")
        return "local"
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load environment configuration from YAML file.
        
        Returns:
            Dictionary containing configuration settings
        """
        config_path = Path(self.config_dir) / f"{self.environment}.yaml"
        
        # Check if the file exists
        if not Path(config_path).exists():
            logger.warning(f"Configuration file not found: {config_path}")
            logger.warning("Falling back to default configuration")
            
            # Use local.yaml as fallback
            fallback_path = Path(self.config_dir) / "local.yaml"
            if Path(fallback_path).exists():
                config_path = fallback_path
            else:
                raise FileNotFoundError(f"Cannot find configuration file for {self.environment}")
        
        # Load YAML configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Resolve path variables
        self._resolve_path_variables(config)
            
        return config
    
    def _resolve_path_variables(self, config: Dict[str, Any]) -> None:
        """
        Resolve path variables in the configuration.
        
        Args:
            config: Configuration dictionary with paths to resolve
        """
        if "paths" not in config:
            return
            
        for key, path in config["paths"].items():
            if isinstance(path, str) and "${PROJECT_ROOT}" in path:
                config["paths"][key] = path.replace("${PROJECT_ROOT}", self.project_root)
    
    def _setup_environment(self) -> None:
        """Apply environment-specific settings."""
        # Log environment info
        logger.info(f"Setting up {self.environment} environment")
        logger.info(f"Project root: {self.project_root}")
        
        # Verify hardware requirements if specified
        self._verify_hardware_requirements()
        
        # Apply environment-specific settings
        if self.environment == "runpod":
            self._setup_runpod_environment()
    
    def _setup_runpod_environment(self) -> None:
        """Apply RunPod-specific environment settings."""
        # Set cache directory if it exists
        if os.path.exists('/cache'):
            os.environ['TRANSFORMERS_CACHE'] = '/cache/huggingface'
            os.environ['TORCH_HOME'] = '/cache/torch'
            logger.info("Set cache directories to RunPod persistent storage")
        
        # Configure PyTorch for optimal performance
        if torch.cuda.is_available():
            # Use deterministic algorithms for reproducibility if specified
            if self.config.get("hardware", {}).get("deterministic", False):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                logger.info("Set PyTorch to use deterministic algorithms")
            else:
                # Use benchmark mode for performance
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark mode for performance")
    
    def _verify_hardware_requirements(self) -> None:
        """Verify that the current hardware meets the requirements."""
        if "hardware" not in self.config:
            return
            
        hardware = self.config["hardware"]
        
        # Check GPU requirements
        if hardware.get("gpu_required", False) and not torch.cuda.is_available():
            logger.warning("GPU is required but not available")
            
        # Check CUDA version if specified
        if torch.cuda.is_available() and "cuda_version" in hardware:
            required_version = hardware["cuda_version"]
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                current_version = torch.version.cuda
                logger.info(f"CUDA version: {current_version}")
                
                # Compare major versions (11.2 vs 11.4 should be compatible)
                required_major = required_version.split('.')[0]
                current_major = current_version.split('.')[0]
                
                if current_major != required_major:
                    logger.warning(
                        f"CUDA version mismatch: required {required_version}, "
                        f"current {current_version}"
                    )
            else:
                logger.warning("CUDA version information not available")
            
        # Check GPU memory if applicable
        if torch.cuda.is_available() and "gpu_memory_min" in hardware:
            required_memory = self._parse_memory_string(hardware["gpu_memory_min"])
            actual_memory = torch.cuda.get_device_properties(0).total_memory
            
            if actual_memory < required_memory:
                logger.warning(
                    f"GPU memory ({actual_memory/1e9:.1f}GB) is less than "
                    f"required ({required_memory/1e9:.1f}GB)"
                )
    
    def _parse_memory_string(self, memory_str: str) -> int:
        """
        Parse a memory string like "24GB" into bytes.
        
        Args:
            memory_str: String representation of memory
            
        Returns:
            Memory size in bytes
        """
        if isinstance(memory_str, (int, float)):
            return int(memory_str)
            
        memory_str = memory_str.upper()
        
        if memory_str.endswith("GB"):
            return int(float(memory_str[:-2]) * 1e9)
        elif memory_str.endswith("MB"):
            return int(float(memory_str[:-2]) * 1e6)
        elif memory_str.endswith("KB"):
            return int(float(memory_str[:-2]) * 1e3)
        else:
            try:
                return int(memory_str)
            except ValueError:
                logger.warning(f"Could not parse memory string: {memory_str}")
                return 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Dot-separated path to the configuration value
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self, model_name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get model configuration with optional overrides.
        
        Args:
            model_name: Name of the model
            overrides: Dictionary of configuration overrides
            
        Returns:
            Model configuration dictionary
        """
        # Start with default model settings
        model_config = self.config.get("model_defaults", {}).copy()
        
        # Add model-specific settings if available
        if "models" in self.config and model_name in self.config["models"]:
            model_specific = self.config["models"][model_name]
            model_config.update(model_specific)
        
        # Apply any runtime overrides
        if overrides:
            model_config.update(overrides)
            
        return model_config

    def print_summary(self) -> None:
        """Print a summary of the current environment configuration."""
        print(f"Environment: {self.environment}")
        print(f"Project root: {self.project_root}")
        
        # Print system information
        print("\nSystem Information:")
        print(f"  OS: {platform.system()} {platform.release()}")
        print(f"  Python: {platform.python_version()}")
        
        if torch.cuda.is_available():
            print("\nGPU Information:")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Memory: {memory_gb:.2f} GB")
        else:
            print("\nNo GPU detected")
        
        if "hardware" in self.config:
            hw = self.config["hardware"]
            print("\nHardware Configuration:")
            for key, value in hw.items():
                print(f"  {key}: {value}")
        
        if "model_defaults" in self.config:
            print("\nDefault Model Settings:")
            for key, value in self.config["model_defaults"].items():
                print(f"  {key}: {value}")
        
        print("\nPath Configuration:")
        for key, value in self.config.get("paths", {}).items():
            print(f"  {key}: {value}")
        
        if "dependencies" in self.config:
            print("\nDependency Requirements:")
            for package, version in self.config["dependencies"].items():
                print(f"  {package}: {version}")


# Singleton instance for global access
_env_config = None

def get_environment_config(reload: bool = False) -> EnvironmentConfig:
    """
    Get the environment configuration singleton.
    
    Args:
        reload: Force reloading the configuration
        
    Returns:
        EnvironmentConfig instance
    """
    global _env_config
    
    if _env_config is None or reload:
        _env_config = EnvironmentConfig()
        
    return _env_config


def setup_environment():
    """
    Set up the environment based on the detected configuration.
    This is a convenience function to be called at the beginning of scripts.
    """
    # Get environment configuration
    env_config = get_environment_config()
    
    # Apply PyTorch settings for RunPod
    if env_config.environment == "runpod" and torch.cuda.is_available():
        # Set default tensor type based on configured precision
        precision = env_config.get("model_defaults.precision", "float32")
        if precision == "bfloat16" and torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            logger.info("Set default tensor type to BFloat16")
        elif precision == "float16" and torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            logger.info("Set default tensor type to Float16")
    
    return env_config


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Simple test of environment detection and configuration loading
    env_config = get_environment_config()
    env_config.print_summary()
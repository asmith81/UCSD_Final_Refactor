"""
Environment Configuration System

This module provides functionality for:
1. Detecting and validating the execution environment
2. Managing environment-specific settings
3. Verifying hardware requirements
4. Setting up appropriate resources for each environment

The environment configuration builds on the base configuration but
adds specialized functionality for environment detection and setup.
"""

import os
import platform
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Try importing torch for GPU detection
try:
    import torch
except ImportError:
    torch = None

# Import base configuration components
from src.config.base_config import (
    BaseConfig,
    ConfigurationManager,
    get_config_manager,
    EnvironmentType,
    ConfigurationError
)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Information about available hardware resources."""
    
    # CPU information
    cpu_count: int = 0
    cpu_type: str = ""
    memory_gb: float = 0.0
    
    # GPU information
    has_gpu: bool = False
    gpu_count: int = 0
    gpu_type: str = ""
    gpu_memory_gb: float = 0.0
    cuda_version: str = ""
    cuda_available: bool = False
    
    # Platform information
    platform: str = ""
    python_version: str = ""
    
    @classmethod
    def detect(cls) -> 'HardwareInfo':
        """
        Detect available hardware resources.
        
        Returns:
            HardwareInfo object with detected hardware information
        """
        info = cls()
        
        # Platform information
        info.platform = platform.system()
        info.python_version = platform.python_version()
        
        # CPU information
        info.cpu_count = os.cpu_count() or 1
        
        # Get CPU type (platform-specific)
        if info.platform == "Linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            info.cpu_type = line.split(':', 1)[1].strip()
                            break
            except Exception:
                info.cpu_type = "Unknown CPU"
        elif info.platform == "Darwin":  # macOS
            try:
                output = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                info.cpu_type = output
            except Exception:
                info.cpu_type = "Unknown CPU"
        elif info.platform == "Windows":
            try:
                output = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().strip()
                info.cpu_type = output.split("\n")[1]
            except Exception:
                info.cpu_type = "Unknown CPU"
        
        # Get system memory
        try:
            if info.platform == "Linux":
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal'):
                            memory_kb = int(line.split()[1])
                            info.memory_gb = memory_kb / 1024 / 1024
                            break
            elif info.platform == "Darwin":  # macOS
                output = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
                info.memory_gb = int(output) / 1024 / 1024 / 1024
            elif info.platform == "Windows":
                output = subprocess.check_output(["wmic", "computersystem", "get", "TotalPhysicalMemory"]).decode()
                memory_bytes = int(output.split("\n")[1])
                info.memory_gb = memory_bytes / 1024 / 1024 / 1024
        except Exception:
            # Fallback to psutil if available
            try:
                import psutil
                info.memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
            except ImportError:
                info.memory_gb = 0.0
        
        # GPU information (requires torch)
        if torch is not None and torch.cuda.is_available():
            info.has_gpu = True
            info.cuda_available = True
            info.gpu_count = torch.cuda.device_count()
            
            if info.gpu_count > 0:
                info.gpu_type = torch.cuda.get_device_name(0)
                
                # Get GPU memory
                try:
                    props = torch.cuda.get_device_properties(0)
                    info.gpu_memory_gb = props.total_memory / 1024 / 1024 / 1024
                except Exception:
                    info.gpu_memory_gb = 0.0
            
            # Get CUDA version
            if hasattr(torch.version, 'cuda'):
                info.cuda_version = torch.version.cuda
        else:
            # Check if NVIDIA tools are available for GPU detection
            try:
                nvidia_output = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']).decode()
                
                # If we get here, NVIDIA tools are available
                info.has_gpu = True
                
                # Parse the output
                lines = nvidia_output.strip().split('\n')
                info.gpu_count = len(lines)
                
                if info.gpu_count > 0:
                    parts = lines[0].split(',')
                    info.gpu_type = parts[0].strip()
                    
                    # Parse memory string like "16384 MiB"
                    memory_str = parts[1].strip()
                    if 'MiB' in memory_str:
                        mem_mb = float(memory_str.split()[0])
                        info.gpu_memory_gb = mem_mb / 1024
            except (subprocess.SubprocessError, FileNotFoundError):
                # No NVIDIA tools available
                info.has_gpu = False
        
        return info


@dataclass
class EnvironmentRequirements:
    """Requirements for running the extraction system."""
    
    # CPU requirements
    min_cpu_count: int = 1
    min_memory_gb: float = 4.0
    
    # GPU requirements
    gpu_required: bool = False
    min_gpu_count: int = 1
    min_gpu_memory_gb: float = 0.0
    cuda_required: bool = False
    min_cuda_version: str = ""
    
    def validate(self, hardware_info: HardwareInfo) -> List[str]:
        """
        Validate hardware against requirements.
        
        Args:
            hardware_info: Detected hardware information
            
        Returns:
            List of validation errors (empty if requirements are met)
        """
        errors = []
        
        # Check CPU requirements
        if hardware_info.cpu_count < self.min_cpu_count:
            errors.append(f"Insufficient CPU cores: {hardware_info.cpu_count} available, {self.min_cpu_count} required")
        
        if hardware_info.memory_gb < self.min_memory_gb:
            errors.append(f"Insufficient memory: {hardware_info.memory_gb:.1f} GB available, {self.min_memory_gb:.1f} GB required")
        
        # Check GPU requirements if needed
        if self.gpu_required:
            if not hardware_info.has_gpu:
                errors.append("GPU required but not available")
            elif hardware_info.gpu_count < self.min_gpu_count:
                errors.append(f"Insufficient GPUs: {hardware_info.gpu_count} available, {self.min_gpu_count} required")
            elif hardware_info.gpu_memory_gb < self.min_gpu_memory_gb:
                errors.append(f"Insufficient GPU memory: {hardware_info.gpu_memory_gb:.1f} GB available, {self.min_gpu_memory_gb:.1f} GB required")
            
            # Check CUDA if required
            if self.cuda_required and not hardware_info.cuda_available:
                errors.append("CUDA required but not available")
            elif self.cuda_required and self.min_cuda_version:
                # Compare CUDA versions
                if not hardware_info.cuda_version:
                    errors.append(f"CUDA version {self.min_cuda_version} required but CUDA not detected")
                else:
                    # Simple version comparison (assumes standard version format)
                    current_parts = hardware_info.cuda_version.split('.')
                    required_parts = self.min_cuda_version.split('.')
                    
                    # Compare major version
                    if int(current_parts[0]) < int(required_parts[0]):
                        errors.append(f"Insufficient CUDA version: {hardware_info.cuda_version} available, {self.min_cuda_version} required")
                    # Compare minor version if major versions match
                    elif int(current_parts[0]) == int(required_parts[0]) and len(current_parts) > 1 and len(required_parts) > 1:
                        if int(current_parts[1]) < int(required_parts[1]):
                            errors.append(f"Insufficient CUDA version: {hardware_info.cuda_version} available, {self.min_cuda_version} required")
        
        return errors


@dataclass
class EnvironmentPaths:
    """Important paths for the current environment."""
    
    # Base directories
    project_root: Path
    config_dir: Path
    data_dir: Path
    results_dir: Path
    models_dir: Path
    cache_dir: Path
    
    # Environment-specific paths
    env_config_path: Path
    log_dir: Path
    
    @classmethod
    def from_config(cls, config_manager: ConfigurationManager) -> 'EnvironmentPaths':
        """
        Create environment paths from configuration.
        
        Args:
            config_manager: Configuration manager
            
        Returns:
            EnvironmentPaths object
        """
        # Get base paths from configuration
        project_root = config_manager.get_project_root()
        config_dir = Path(config_manager.get_config_value("paths.config_dir", str(project_root / "configs")))
        data_dir = Path(config_manager.get_config_value("paths.data_dir", str(project_root / "data")))
        results_dir = Path(config_manager.get_config_value("paths.results_dir", str(project_root / "results")))
        models_dir = Path(config_manager.get_config_value("paths.models_dir", str(project_root / "models")))
        
        # Set environment-specific paths
        environment = config_manager.get_environment()
        env_config_path = config_dir / "environments" / f"{environment}.yaml"
        log_dir = Path(config_manager.get_config_value("paths.log_dir", str(project_root / "logs")))
        
        # Special case for RunPod: use persistent storage for cache if available
        if environment == EnvironmentType.RUNPOD and Path("/cache").exists():
            cache_dir = Path("/cache/models")
        else:
            cache_dir = Path(config_manager.get_config_value("paths.cache_dir", str(models_dir / "cache")))
        
        return cls(
            project_root=project_root,
            config_dir=config_dir,
            data_dir=data_dir,
            results_dir=results_dir,
            models_dir=models_dir,
            cache_dir=cache_dir,
            env_config_path=env_config_path,
            log_dir=log_dir
        )
    
    def ensure_directories_exist(self) -> None:
        """Create all directories if they don't exist."""
        for attr_name in dir(self):
            # Skip non-path attributes and special methods
            if attr_name.startswith('_') or not attr_name.endswith('dir'):
                continue
            
            # Get the path
            path = getattr(self, attr_name)
            if isinstance(path, Path) and not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")


@dataclass
class DependencyInfo:
    """Information about Python dependencies."""
    
    # Core dependencies
    required_packages: Dict[str, str] = field(default_factory=dict)
    installed_packages: Dict[str, str] = field(default_factory=dict)
    missing_packages: List[str] = field(default_factory=list)
    
    @classmethod
    def check_dependencies(cls, required_packages: Dict[str, str]) -> 'DependencyInfo':
        """
        Check if required dependencies are installed.
        
        Args:
            required_packages: Dictionary mapping package names to version requirements
            
        Returns:
            DependencyInfo object with check results
        """
        info = cls(required_packages=required_packages)
        
        # Check each required package
        for package, version_req in required_packages.items():
            try:
                # Try to import the package
                module = __import__(package)
                
                # Get the installed version
                version = getattr(module, '__version__', 'unknown')
                info.installed_packages[package] = version
                
                # TODO: Add version compatibility checking if needed
                
            except ImportError:
                # Package is not installed
                info.missing_packages.append(package)
        
        return info


@dataclass
class EnvironmentConfig(BaseConfig):
    """Configuration for the execution environment."""
    
    # Basic environment information
    environment_type: EnvironmentType
    hardware_info: HardwareInfo
    paths: EnvironmentPaths
    
    # Environment requirements and capabilities
    requirements: EnvironmentRequirements = field(default_factory=EnvironmentRequirements)
    dependencies: DependencyInfo = field(default_factory=DependencyInfo)
    
    # Environment-specific settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """
        Validate the environment configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate hardware requirements
        hw_errors = self.requirements.validate(self.hardware_info)
        if hw_errors:
            errors.extend(hw_errors)
        
        # Check for missing dependencies
        if self.dependencies.missing_packages:
            packages = ", ".join(self.dependencies.missing_packages)
            errors.append(f"Missing required packages: {packages}")
        
        return errors
    
    @classmethod
    def create(cls, config_manager: Optional[ConfigurationManager] = None) -> 'EnvironmentConfig':
        """
        Create environment configuration from the current state.
        
        Args:
            config_manager: Optional configuration manager (created if None)
            
        Returns:
            EnvironmentConfig object
        """
        # Get or create configuration manager
        if config_manager is None:
            config_manager = get_config_manager()
        
        # Get environment type
        environment_type = config_manager.get_environment()
        
        # Detect hardware
        hardware_info = HardwareInfo.detect()
        
        # Get paths
        paths = EnvironmentPaths.from_config(config_manager)
        
        # Get requirements from configuration
        requirements = EnvironmentRequirements(
            min_cpu_count=config_manager.get_config_value("hardware.cpu_count", 1),
            min_memory_gb=config_manager.get_config_value("hardware.memory_min", 4.0),
            gpu_required=config_manager.get_config_value("hardware.gpu_required", False),
            min_gpu_memory_gb=config_manager.get_config_value("hardware.gpu_memory_min", 0.0),
            cuda_required=config_manager.get_config_value("hardware.cuda_required", False),
            min_cuda_version=config_manager.get_config_value("hardware.cuda_version", "")
        )
        
        # Check dependencies
        required_packages = config_manager.get_config_value("dependencies", {})
        dependencies = DependencyInfo.check_dependencies(required_packages)
        
        # Get environment settings
        settings = {}
        for section in ["model_defaults", "execution", "logging"]:
            section_config = config_manager.get_config_value(section, {})
            if section_config:
                settings[section] = section_config
        
        # Create environment configuration
        return cls(
            environment_type=environment_type,
            hardware_info=hardware_info,
            paths=paths,
            requirements=requirements,
            dependencies=dependencies,
            settings=settings
        )
    
    def setup_environment(self) -> None:
        """
        Set up the environment based on the configuration.
        
        This method prepares the environment for execution,
        including creating directories and setting environment
        variables as needed.
        """
        # Create necessary directories
        self.paths.ensure_directories_exist()
        
        # Set up environment-specific configuration
        if self.environment_type == EnvironmentType.RUNPOD:
            self._setup_runpod_environment()
        else:
            self._setup_local_environment()
        
        # Configure PyTorch if available
        if torch is not None and torch.cuda.is_available():
            self._configure_pytorch()
        
        logger.info(f"Environment setup complete for {self.environment_type}")
    
    def _setup_local_environment(self) -> None:
        """Set up local environment."""
        # Set up environment variables
        os.environ["PROJECT_ROOT"] = str(self.paths.project_root)
        
        # Set up Python path
        python_path = os.environ.get("PYTHONPATH", "")
        if str(self.paths.project_root) not in python_path:
            if python_path:
                os.environ["PYTHONPATH"] = f"{python_path}:{self.paths.project_root}"
            else:
                os.environ["PYTHONPATH"] = str(self.paths.project_root)
        
        logger.info("Local environment setup complete")
    
    def _setup_runpod_environment(self) -> None:
        """Set up RunPod environment."""
        # Set up environment variables
        os.environ["PROJECT_ROOT"] = str(self.paths.project_root)
        
        # Use RunPod persistent storage for cache if available
        if Path("/cache").exists():
            os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"
            os.environ["TORCH_HOME"] = "/cache/torch"
        
        # Set up Python path
        python_path = os.environ.get("PYTHONPATH", "")
        if str(self.paths.project_root) not in python_path:
            if python_path:
                os.environ["PYTHONPATH"] = f"{python_path}:{self.paths.project_root}"
            else:
                os.environ["PYTHONPATH"] = str(self.paths.project_root)
        
        logger.info("RunPod environment setup complete")
    
    def _configure_pytorch(self) -> None:
        """Configure PyTorch based on environment settings."""
        # Check for deterministic flag
        deterministic = self.settings.get("model_defaults", {}).get("deterministic", False)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("PyTorch configured for deterministic execution")
        else:
            # Use benchmark for better performance
            torch.backends.cudnn.benchmark = True
            logger.info("PyTorch configured for performance (benchmark enabled)")
        
        # Configure default tensor type based on precision
        precision = self.settings.get("model_defaults", {}).get("precision", "float32")
        if precision == "bfloat16" and torch.cuda.is_available() and hasattr(torch, 'bfloat16'):
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            logger.info("PyTorch default tensor type set to bfloat16")
        elif precision == "float16" and torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            logger.info("PyTorch default tensor type set to float16")
    
    def meets_model_requirements(self, model_name: str) -> Tuple[bool, List[str]]:
        """
        Check if the environment meets the requirements for a model.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            Tuple of (requirements_met, list_of_issues)
        """
        # Get config manager
        config_manager = get_config_manager()
        
        # Get model configuration
        model_config = config_manager.get_model_config(model_name)
        if model_config is None:
            return False, [f"Model configuration not found for {model_name}"]
        
        issues = []
        
        # Check GPU requirements
        if model_config.gpu_required and not self.hardware_info.has_gpu:
            issues.append(f"Model {model_name} requires a GPU, but none is available")
        
        # Check GPU memory
        if model_config.gpu_required and model_config.min_gpu_memory_gb > 0:
            if self.hardware_info.gpu_memory_gb < model_config.min_gpu_memory_gb:
                issues.append(
                    f"Model {model_name} requires {model_config.min_gpu_memory_gb:.1f}GB GPU memory, "
                    f"but only {self.hardware_info.gpu_memory_gb:.1f}GB is available"
                )
        
        # Check if CPU fallback is allowed when GPU is required but not available
        if model_config.gpu_required and not self.hardware_info.has_gpu and not model_config.cpu_fallback:
            issues.append(f"Model {model_name} requires GPU and does not support CPU fallback")
        
        return len(issues) == 0, issues
    
    def get_available_models(self) -> List[str]:
        """
        Get list of models that can run in the current environment.
        
        Returns:
            List of model names compatible with the current environment
        """
        # Get config manager
        config_manager = get_config_manager()
        
        # Get all models
        all_models = config_manager.get_available_models()
        
        # Filter to those that meet requirements
        available_models = []
        for model_name in all_models:
            meets_req, _ = self.meets_model_requirements(model_name)
            if meets_req:
                available_models.append(model_name)
        
        return available_models
    
    def print_summary(self) -> None:
        """Print a summary of the environment configuration."""
        print(f"\n{'=' * 40}")
        print(f"Environment: {self.environment_type}")
        print(f"{'=' * 40}")
        
        print("\nHardware Information:")
        print(f"  Platform: {self.hardware_info.platform}")
        print(f"  Python version: {self.hardware_info.python_version}")
        print(f"  CPU: {self.hardware_info.cpu_type} ({self.hardware_info.cpu_count} cores)")
        print(f"  Memory: {self.hardware_info.memory_gb:.1f}GB")
        
        if self.hardware_info.has_gpu:
            print("\nGPU Information:")
            print(f"  GPU: {self.hardware_info.gpu_type}")
            print(f"  GPU count: {self.hardware_info.gpu_count}")
            print(f"  GPU memory: {self.hardware_info.gpu_memory_gb:.1f}GB")
            print(f"  CUDA version: {self.hardware_info.cuda_version}")
        else:
            print("\nNo GPU detected")
        
        print("\nPath Information:")
        print(f"  Project root: {self.paths.project_root}")
        print(f"  Data directory: {self.paths.data_dir}")
        print(f"  Results directory: {self.paths.results_dir}")
        print(f"  Model cache: {self.paths.cache_dir}")
        
        # Print dependency information
        if self.dependencies.installed_packages:
            print("\nInstalled Dependencies:")
            for package, version in self.dependencies.installed_packages.items():
                req_version = self.dependencies.required_packages.get(package, "")
                if req_version:
                    print(f"  {package}: {version} (required: {req_version})")
                else:
                    print(f"  {package}: {version}")
        
        if self.dependencies.missing_packages:
            print("\nMissing Dependencies:")
            for package in self.dependencies.missing_packages:
                req_version = self.dependencies.required_packages.get(package, "")
                print(f"  {package} {req_version}")
        
        # Print available models
        available_models = self.get_available_models()
        if available_models:
            print("\nAvailable Models:")
            for model in available_models:
                print(f"  {model}")
        
        print("\nValidation:")
        errors = self.validate()
        if errors:
            print("  Issues found:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("  All requirements met")
        
        print("\n")


# Global environment configuration instance
_environment_config = None

def get_environment_config() -> EnvironmentConfig:
    """
    Get the global environment configuration instance.
    
    Returns:
        EnvironmentConfig instance
    """
    global _environment_config
    
    if _environment_config is None:
        _environment_config = EnvironmentConfig.create()
    
    return _environment_config


def setup_environment() -> EnvironmentConfig:
    """
    Set up the environment for execution.
    
    This function prepares the execution environment based on the
    detected environment type and configuration settings.
    
    Returns:
        Configured EnvironmentConfig instance
    """
    # Get environment configuration
    env_config = get_environment_config()
    
    # Set up the environment
    env_config.setup_environment()
    
    # Configure logging
    _configure_logging(env_config)
    
    return env_config


def _configure_logging(env_config: EnvironmentConfig) -> None:
    """
    Configure logging based on environment settings.
    
    Args:
        env_config: Environment configuration
    """
    # Get logging settings
    log_settings = env_config.settings.get("logging", {})
    log_level = log_settings.get("level", "INFO")
    log_format = log_settings.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_to_file = log_settings.get("log_to_file", True)
    
    # Convert log level string to constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format
    )
    
    # Add file handler if enabled
    if log_to_file:
        try:
            # Ensure log directory exists
            env_config.paths.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create log filename
            log_file = env_config.paths.log_dir / f"{env_config.environment_type}_{env_config.settings.get('experiment_name', 'extraction')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format))
            
            # Add to root logger
            logging.getLogger().addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.warning(f"Failed to set up file logging: {e}")


def check_environment_for_model(model_name: str) -> Tuple[bool, List[str]]:
    """
    Check if the current environment can run a specific model.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        Tuple of (can_run, list_of_issues)
    """
    env_config = get_environment_config()
    return env_config.meets_model_requirements(model_name)


def print_environment_info() -> None:
    """Print information about the current environment."""
    env_config = get_environment_config()
    env_config.print_summary()
"""
Model Service

This module provides a service-based approach to model management, including:
1. Clean interfaces for model loading and usage
2. Integration with configuration system
3. GPU memory optimization
4. Comprehensive error handling
5. Model tracking and lifecycle management

The design follows service-oriented principles with clear separation of concerns
and explicit dependency management.
"""

import os
import time
import torch
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Type, Union, Callable

# Import configuration components
from src.config.environment import get_environment_config
from src.config.paths import get_path_config

# Set up logging
logger = logging.getLogger(__name__)


class ModelServiceError(Exception):
    """Base exception class for model service errors."""
    pass


class ModelNotFoundError(ModelServiceError):
    """Raised when a requested model is not found in the registry."""
    pass


class ModelLoadingError(ModelServiceError):
    """Raised when a model cannot be loaded."""
    pass


class ModelMemoryError(ModelServiceError):
    """Raised when there is insufficient memory to load a model."""
    pass


class ModelExecutionError(ModelServiceError):
    """Raised when there is an error during model execution."""
    pass


class ModelConfigurationError(ModelServiceError):
    """Raised when there is an invalid model configuration."""
    pass


class ModelGPUInfo:
    """
    Tracks and provides information about GPU resources.
    """
    
    def __init__(self):
        """Initialize GPU info tracker."""
        self.available = torch.cuda.is_available()
        
        if self.available:
            self.device_count = torch.cuda.device_count()
            self.current_device = torch.cuda.current_device()
            self.device_name = torch.cuda.get_device_name(self.current_device)
            
            properties = torch.cuda.get_device_properties(self.current_device)
            self.total_memory = properties.total_memory
            self.total_memory_gb = self.total_memory / (1024 ** 3)
            self.compute_capability = (properties.major, properties.minor)
        else:
            self.device_count = 0
            self.current_device = None
            self.device_name = "No GPU"
            self.total_memory = 0
            self.total_memory_gb = 0
            self.compute_capability = (0, 0)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get current GPU memory usage information.
        
        Returns:
            Dictionary with GPU memory details
        """
        if not self.available:
            return {
                "available": False,
                "device_name": "No GPU",
                "total_memory_gb": 0,
                "allocated_memory_gb": 0,
                "reserved_memory_gb": 0,
                "free_memory_gb": 0,
                "utilization_percent": 0
            }
        
        try:
            # Get current memory usage
            allocated = torch.cuda.memory_allocated(self.current_device)
            reserved = torch.cuda.memory_reserved(self.current_device)
            free = self.total_memory - reserved
            
            # Convert to GB
            allocated_gb = allocated / (1024 ** 3)
            reserved_gb = reserved / (1024 ** 3)
            free_gb = free / (1024 ** 3)
            
            # Calculate utilization
            utilization = (allocated / self.total_memory) * 100
            
            return {
                "available": True,
                "device_name": self.device_name,
                "total_memory_gb": self.total_memory_gb,
                "allocated_memory_gb": allocated_gb,
                "reserved_memory_gb": reserved_gb,
                "free_memory_gb": free_gb,
                "utilization_percent": utilization,
                "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}"
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return {
                "available": True,
                "device_name": self.device_name,
                "error": str(e)
            }
    
    def clean_memory(self) -> Dict[str, Any]:
        """
        Clean up GPU memory by emptying cache and triggering garbage collection.
        
        Returns:
            Dictionary with before/after memory information
        """
        if not self.available:
            return {"available": False, "cleaned": False}
        
        before_info = self.get_memory_info()
        
        try:
            # Empty cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            after_info = self.get_memory_info()
            
            memory_freed = before_info["allocated_memory_gb"] - after_info["allocated_memory_gb"]
            logger.info(f"Cleaned GPU memory. Freed: {memory_freed:.2f} GB")
            
            return {
                "before": before_info,
                "after": after_info,
                "memory_freed_gb": memory_freed,
                "cleaned": True
            }
        except Exception as e:
            logger.error(f"Error cleaning GPU memory: {e}")
            return {
                "available": True,
                "cleaned": False,
                "error": str(e),
                "before": before_info
            }
    
    def check_memory_sufficient(self, required_gb: float) -> bool:
        """
        Check if there is sufficient GPU memory available.
        
        Args:
            required_gb: Required memory in GB
            
        Returns:
            True if sufficient memory is available, False otherwise
        """
        if not self.available:
            return False
        
        info = self.get_memory_info()
        available_gb = info["free_memory_gb"]
        
        return available_gb >= required_gb
    
    def __str__(self) -> str:
        """String representation of GPU info."""
        if not self.available:
            return "No GPU available"
        
        info = self.get_memory_info()
        return (
            f"GPU: {info['device_name']} | "
            f"Memory: {info['allocated_memory_gb']:.2f}/{info['total_memory_gb']:.2f} GB "
            f"({info['utilization_percent']:.1f}%) | "
            f"Compute: {info['compute_capability']}"
        )


class ModelConfig:
    """
    Model configuration with type checking and validation.
    """
    
    def __init__(
        self,
        name: str,
        repo_id: str,
        model_type: str,
        processor_type: str = "AutoProcessor",
        description: str = "",
        hardware_requirements: Optional[Dict[str, Any]] = None,
        loading_params: Optional[Dict[str, Any]] = None,
        quantization_options: Optional[Dict[str, Any]] = None,
        inference_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model configuration.
        
        Args:
            name: Model name (identifier)
            repo_id: Repository ID for loading
            model_type: Model class type
            processor_type: Processor class type
            description: Human-readable description
            hardware_requirements: Hardware requirements for the model
            loading_params: Parameters for model loading
            quantization_options: Quantization options by strategy
            inference_params: Default inference parameters
        """
        self.name = name
        self.repo_id = repo_id
        self.model_type = model_type
        self.processor_type = processor_type
        self.description = description
        
        # Set default values for optional parameters
        self.hardware_requirements = hardware_requirements or {
            "gpu_required": True,
            "gpu_memory_min": "0GB",
            "recommended_gpu": "",
            "cpu_fallback": False
        }
        
        self.loading_params = loading_params or {}
        self.quantization_options = quantization_options or {}
        self.inference_params = inference_params or {
            "max_new_tokens": 50,
            "do_sample": False,
            "temperature": 1.0
        }
        
        # Validate configuration
        self._validate()
    
    def _validate(self) -> None:
        """
        Validate model configuration.
        
        Raises:
            ModelConfigurationError: If configuration is invalid
        """
        # Check for required fields
        required_fields = {"name", "repo_id", "model_type"}
        for field in required_fields:
            if not getattr(self, field, None):
                raise ModelConfigurationError(f"Missing required field: {field}")
        
        # Validate hardware requirements
        if self.hardware_requirements.get("gpu_required", False):
            # Check for gpu_memory_min
            gpu_memory_min = self.hardware_requirements.get("gpu_memory_min", "0GB")
            if not isinstance(gpu_memory_min, (str, int, float)):
                raise ModelConfigurationError(
                    f"Invalid gpu_memory_min: {gpu_memory_min}. Expected string (e.g., '16GB') or number."
                )
    
    def get_loading_params(self, quantization: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameters for loading this model.
        
        Args:
            quantization: Optional quantization strategy name
            
        Returns:
            Dictionary of parameters for model loading
        """
        # Start with base loading parameters
        params = self.loading_params.copy()
        
        # Apply quantization parameters if specified
        if quantization and quantization in self.quantization_options:
            quant_params = self.quantization_options[quantization]
            params.update(quant_params)
            
        return params
    
    def get_inference_params(self) -> Dict[str, Any]:
        """
        Get default inference parameters for this model.
        
        Returns:
            Dictionary of inference parameters
        """
        return self.inference_params.copy()
    
    def get_memory_requirement_gb(self) -> float:
        """
        Get minimum memory requirement in GB.
        
        Returns:
            Memory requirement in GB
        """
        gpu_memory_min = self.hardware_requirements.get("gpu_memory_min", "0GB")
        
        if isinstance(gpu_memory_min, (int, float)):
            return float(gpu_memory_min)
        
        # Parse string like "16GB" or "24GB"
        try:
            if gpu_memory_min.endswith("GB"):
                return float(gpu_memory_min[:-2])
            elif gpu_memory_min.endswith("MB"):
                return float(gpu_memory_min[:-2]) / 1024
            else:
                return float(gpu_memory_min)
        except (ValueError, AttributeError):
            logger.warning(f"Could not parse memory requirement: {gpu_memory_min}, defaulting to 0GB")
            return 0.0
    
    def get_available_quantization_strategies(self) -> List[str]:
        """
        Get all available quantization strategies for this model.
        
        Returns:
            List of quantization strategy names
        """
        return list(self.quantization_options.keys())
    
    def get_quantization_metadata(self, strategy: str) -> Dict[str, Any]:
        """
        Get detailed metadata about a specific quantization strategy.
        
        Args:
            strategy: Name of the quantization strategy
            
        Returns:
            Dictionary with detailed quantization metadata
            
        Raises:
            ModelConfigurationError: If strategy not found
        """
        if strategy not in self.quantization_options:
            raise ModelConfigurationError(f"Quantization strategy '{strategy}' not found for model {self.name}")
        
        # Get the basic quantization options
        quant_params = self.quantization_options[strategy].copy()
        
        # Add metadata about precision, memory reductions, etc.
        metadata = {
            "strategy": strategy,
            "parameters": quant_params,
            "precision_name": self._get_precision_name(quant_params),
            "bits": self._get_quantization_bits(quant_params),
            "memory_reduction_factor": self._get_memory_reduction_factor(quant_params)
        }
        
        return metadata
    
    def _get_precision_name(self, quant_params: Dict[str, Any]) -> str:
        """Get human-readable precision name from quantization parameters."""
        if quant_params.get("load_in_8bit", False):
            return "INT8"
        elif quant_params.get("load_in_4bit", False):
            return "INT4"
        elif quant_params.get("torch_dtype") == "bfloat16" or "bfloat16" in str(quant_params.get("torch_dtype", "")):
            return "BF16"
        elif quant_params.get("torch_dtype") == "float16" or "float16" in str(quant_params.get("torch_dtype", "")):
            return "FP16"
        elif quant_params.get("torch_dtype") == "float32" or "float32" in str(quant_params.get("torch_dtype", "")):
            return "FP32"
        else:
            return "Unknown"
    
    def _get_quantization_bits(self, quant_params: Dict[str, Any]) -> int:
        """Get number of bits used for weights from quantization parameters."""
        if quant_params.get("load_in_4bit", False):
            return 4
        elif quant_params.get("load_in_8bit", False):
            return 8
        elif quant_params.get("torch_dtype") == "bfloat16" or "bfloat16" in str(quant_params.get("torch_dtype", "")):
            return 16
        elif quant_params.get("torch_dtype") == "float16" or "float16" in str(quant_params.get("torch_dtype", "")):
            return 16
        elif quant_params.get("torch_dtype") == "float32" or "float32" in str(quant_params.get("torch_dtype", "")):
            return 32
        else:
            return 0
    
    def _get_memory_reduction_factor(self, quant_params: Dict[str, Any]) -> float:
        """
        Calculate approximate memory reduction factor compared to FP32.
        
        Returns:
            Approximate factor by which memory is reduced (e.g., 2.0 means half the memory)
        """
        bits = self._get_quantization_bits(quant_params)
        if bits == 0:
            return 1.0
        return 32.0 / bits
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Create model configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            ModelConfig instance
        """
        # Process nested dictionaries
        hardware_requirements = config_dict.get("hardware", {})
        loading_params = config_dict.get("loading", {})
        
        # Process quantization options
        quantization_options = {}
        if "quantization" in config_dict and "options" in config_dict["quantization"]:
            quantization_options = config_dict["quantization"]["options"]
        
        # Process inference parameters
        inference_params = config_dict.get("inference", {})
        
        return cls(
            name=config_dict.get("name", ""),
            repo_id=config_dict.get("repo_id", ""),
            model_type=config_dict.get("model_type", ""),
            processor_type=config_dict.get("processor_type", "AutoProcessor"),
            description=config_dict.get("description", ""),
            hardware_requirements=hardware_requirements,
            loading_params=loading_params,
            quantization_options=quantization_options,
            inference_params=inference_params
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model configuration to dictionary.
        
        Returns:
            Dictionary representation of model configuration
        """
        return {
            "name": self.name,
            "repo_id": self.repo_id,
            "model_type": self.model_type,
            "processor_type": self.processor_type,
            "description": self.description,
            "hardware": self.hardware_requirements,
            "loading": self.loading_params,
            "quantization": {"options": self.quantization_options},
            "inference": self.inference_params
        }


class ModelLoadingResult:
    """
    Encapsulates the result of a model loading operation.
    """
    
    def __init__(
        self,
        model_name: str,
        success: bool,
        model: Any = None,
        processor: Any = None,
        error: Optional[Exception] = None,
        loading_time: float = 0.0,
        memory_used: Optional[Dict[str, Any]] = None,
        loading_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize model loading result.
        
        Args:
            model_name: Name of the model
            success: Whether loading was successful
            model: Loaded model (None if loading failed)
            processor: Loaded processor (None if loading failed)
            error: Exception if loading failed
            loading_time: Time taken to load the model in seconds
            memory_used: Memory usage information
            loading_params: Parameters used for loading
        """
        self.model_name = model_name
        self.success = success
        self.model = model
        self.processor = processor
        self.error = error
        self.loading_time = loading_time
        self.memory_used = memory_used or {}
        self.loading_params = loading_params or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert loading result to dictionary.
        
        Returns:
            Dictionary representation of loading result
        """
        return {
            "model_name": self.model_name,
            "success": self.success,
            "error": str(self.error) if self.error else None,
            "loading_time": self.loading_time,
            "memory_used": self.memory_used,
            "loading_params": self.loading_params,
            "timestamp": self.timestamp
        }
    
    def log_result(self) -> None:
        """Log the loading result."""
        if self.success:
            logger.info(
                f"Successfully loaded model {self.model_name} in {self.loading_time:.2f}s. "
                f"Memory: {self.memory_used.get('allocated_memory_gb', 0):.2f}GB"
            )
        else:
            logger.error(
                f"Failed to load model {self.model_name}: {self.error}"
            )
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """
        Extract and return quantization-specific information.
        
        Returns:
            Dictionary with quantization details
        """
        quant_info = {}
        
        # Check loading parameters for quantization clues
        params = self.loading_params
        
        # Identify quantization strategy
        if params.get("load_in_4bit", False):
            quant_info["type"] = "INT4"
            quant_info["bits"] = 4
            quant_info["quantization_method"] = params.get("bnb_4bit_quant_type", "unknown")
            quant_info["compute_dtype"] = str(params.get("bnb_4bit_compute_dtype", "unknown"))
        elif params.get("load_in_8bit", False):
            quant_info["type"] = "INT8"
            quant_info["bits"] = 8
            quant_info["quantization_method"] = params.get("bnb_8bit_quant_type", "unknown")
        elif "torch_dtype" in params:
            dtype = params["torch_dtype"]
            if "bfloat16" in str(dtype):
                quant_info["type"] = "BF16"
                quant_info["bits"] = 16
            elif "float16" in str(dtype):
                quant_info["type"] = "FP16"
                quant_info["bits"] = 16
            elif "float32" in str(dtype):
                quant_info["type"] = "FP32"
                quant_info["bits"] = 32
        
        # Add any memory details
        if self.memory_used:
            quant_info["memory_used_gb"] = self.memory_used.get("allocated_memory_gb", 0)
            quant_info["memory_total_gb"] = self.memory_used.get("total_memory_gb", 0)
            if "total_memory_gb" in self.memory_used and "allocated_memory_gb" in self.memory_used:
                quant_info["memory_efficiency"] = self.memory_used["allocated_memory_gb"] / self.memory_used["total_memory_gb"]
        
        return quant_info


class ModelService:
    """
    Service for model management, loading, and execution.
    
    This class provides a unified interface for working with machine learning models,
    handling configuration, loading, memory management, and execution.
    """
    
    def __init__(self):
        """Initialize the model service."""
        self.env_config = get_environment_config()
        self.paths = get_path_config()
        self.gpu_info = ModelGPUInfo()
        
        # Model registry
        self.model_configs = {}
        self.loaded_models = {}
        
        # Memory management settings
        self.auto_clean_memory = True
        self.memory_threshold = 0.9  # 90% GPU memory usage triggers cleanup
        
        # Load available model configurations
        self._load_model_configs()
        
        # Log initialization
        self._log_initialization()
    
    def _load_model_configs(self) -> None:
        """
        Load model configurations from config files.
        """
        model_configs_dir = self.paths.get("model_configs_dir")
        if not model_configs_dir or not os.path.exists(model_configs_dir):
            logger.warning(f"Model configs directory not found: {model_configs_dir}")
            return
        
        # Find all YAML model configs
        config_files = []
        for ext in [".yaml", ".yml"]:
            config_files.extend(list(Path(model_configs_dir).glob(f"*{ext}")))
        
        if not config_files:
            logger.warning(f"No model configuration files found in {model_configs_dir}")
            return
        
        # Load each configuration
        import yaml
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Create model configuration
                model_name = config_file.stem
                model_config = ModelConfig.from_dict(config_data)
                
                # Store in registry
                self.model_configs[model_name] = model_config
                logger.info(f"Loaded model configuration: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model configuration from {config_file}: {e}")
    
    def _log_initialization(self) -> None:
        """Log model service initialization."""
        logger.info(f"Model service initialized with {len(self.model_configs)} model configurations")
        
        if self.gpu_info.available:
            logger.info(f"GPU available: {self.gpu_info.device_name}")
            logger.info(f"GPU memory: {self.gpu_info.total_memory_gb:.2f}GB")
            logger.info(f"Compute capability: {self.gpu_info.compute_capability}")
        else:
            logger.info("No GPU available. Models will run on CPU.")
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfig instance
            
        Raises:
            ModelNotFoundError: If model is not found in registry
        """
        if model_name not in self.model_configs:
            raise ModelNotFoundError(f"Model not found in registry: {model_name}")
        
        return self.model_configs[model_name]
    
    def register_model_config(self, model_config: ModelConfig) -> None:
        """
        Register a model configuration.
        
        Args:
            model_config: ModelConfig instance to register
        """
        self.model_configs[model_config.name] = model_config
        logger.info(f"Registered model configuration: {model_config.name}")
    
    def list_available_models(self) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List of model names in the registry
        """
        return list(self.model_configs.keys())
    
    def get_model_class(self, model_type: str):
        """
        Dynamically import and return the appropriate model class.
        
        Args:
            model_type: String representing the model class to load
            
        Returns:
            Model class for loading
            
        Raises:
            ModelLoadingError: If model class cannot be imported
        """
        try:
            # Check for common model types
            if model_type == "LlavaForConditionalGeneration":
                from transformers import LlavaForConditionalGeneration
                return LlavaForConditionalGeneration
            elif model_type == "AutoModelForVision2Seq":
                from transformers import AutoModelForVision2Seq
                return AutoModelForVision2Seq
            elif model_type == "AutoModelForCausalLM":
                from transformers import AutoModelForCausalLM
                return AutoModelForCausalLM
            else:
                # Try dynamic import for custom models
                from importlib import import_module
                
                # Check if model type is a fully qualified name
                if '.' in model_type:
                    module_path, class_name = model_type.rsplit('.', 1)
                    module = import_module(module_path)
                    return getattr(module, class_name)
                else:
                    # Last resort: try to find in transformers
                    from transformers import AutoModel
                    return AutoModel
        except ImportError as e:
            error_msg = f"Could not import model class {model_type}: {e}"
            logger.error(error_msg)
            raise ModelLoadingError(error_msg) from e
        except Exception as e:
            error_msg = f"Error getting model class {model_type}: {e}"
            logger.error(error_msg)
            raise ModelLoadingError(error_msg) from e
    
    def load_model(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        force_reload: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> ModelLoadingResult:
        """
        Load a model by name.
        
        Args:
            model_name: Name of the model to load
            quantization: Quantization strategy to use
            force_reload: Whether to reload if already loaded
            cache_dir: Directory to cache model files
            **kwargs: Additional parameters for model loading
            
        Returns:
            ModelLoadingResult with loaded model and processor
        """
        # Check if model is already loaded
        if not force_reload and model_name in self.loaded_models:
            logger.info(f"Using already loaded model: {model_name}")
            loaded = self.loaded_models[model_name]
            return ModelLoadingResult(
                model_name=model_name,
                success=True,
                model=loaded["model"],
                processor=loaded["processor"],
                loading_time=0.0,
                memory_used=self.gpu_info.get_memory_info()
            )
        
        # Get model configuration
        try:
            model_config = self.get_model_config(model_name)
        except ModelNotFoundError as e:
            return ModelLoadingResult(
                model_name=model_name,
                success=False,
                error=e
            )
        
        # Check GPU requirements
        hardware_reqs = model_config.hardware_requirements
        if hardware_reqs.get("gpu_required", False) and not self.gpu_info.available:
            error = ModelMemoryError(
                f"Model {model_name} requires GPU but none is available"
            )
            return ModelLoadingResult(
                model_name=model_name,
                success=False,
                error=error
            )
        
        # Check memory requirements if GPU is available
        if self.gpu_info.available:
            memory_required = model_config.get_memory_requirement_gb()
            if memory_required > 0:
                if not self.gpu_info.check_memory_sufficient(memory_required):
                    # Try cleaning memory first
                    if self.auto_clean_memory:
                        self.gpu_info.clean_memory()
                        
                        # Check again after cleaning
                        if not self.gpu_info.check_memory_sufficient(memory_required):
                            error = ModelMemoryError(
                                f"Insufficient GPU memory for model {model_name}. "
                                f"Required: {memory_required:.2f}GB, "
                                f"Available: {self.gpu_info.get_memory_info()['free_memory_gb']:.2f}GB"
                            )
                            return ModelLoadingResult(
                                model_name=model_name,
                                success=False,
                                error=error,
                                memory_used=self.gpu_info.get_memory_info()
                            )
        
        # Get loading parameters
        loading_params = model_config.get_loading_params(quantization)
        
        # Add cache directory
        if cache_dir:
            loading_params["cache_dir"] = cache_dir
        elif self.paths.get("model_cache_dir"):
            loading_params["cache_dir"] = self.paths.get("model_cache_dir")
        
        # Update with kwargs
        loading_params.update(kwargs)
        
        # Get model and processor types
        model_type = model_config.model_type
        processor_type = model_config.processor_type
        
        # Record starting memory usage
        start_memory = self.gpu_info.get_memory_info()
        start_time = time.time()
        
        # Clean GPU memory before loading
        if self.auto_clean_memory and self.gpu_info.available:
            self.gpu_info.clean_memory()
        
        # Load model and processor
        try:
            # Get model class
            model_class = self.get_model_class(model_type)
            
            # Get processor class
            from transformers import AutoProcessor
            
            # Get repo ID
            repo_id = model_config.repo_id
            
            # Log loading parameters
            logger.info(f"Loading model {model_name} from {repo_id}")
            
            # Load model
            model = model_class.from_pretrained(repo_id, **loading_params)
            
            # Load processor
            processor = AutoProcessor.from_pretrained(
                repo_id,
                trust_remote_code=loading_params.get("trust_remote_code", True)
            )
            
            # Record memory and time
            end_memory = self.gpu_info.get_memory_info()
            loading_time = time.time() - start_time
            
            # Calculate memory used for loading
            memory_used = {}
            for key in start_memory:
                if key in end_memory and isinstance(start_memory[key], (int, float)):
                    memory_used[f"{key}_delta"] = end_memory[key] - start_memory[key]
            memory_used.update(end_memory)
            
            # Store in loaded models
            self.loaded_models[model_name] = {
                "model": model,
                "processor": processor,
                "loaded_at": datetime.now().isoformat(),
                "loading_time": loading_time,
                "memory_info": memory_used,
                "loading_params": loading_params
            }
            
            # Create and return result
            result = ModelLoadingResult(
                model_name=model_name,
                success=True,
                model=model,
                processor=processor,
                loading_time=loading_time,
                memory_used=memory_used,
                loading_params=loading_params
            )
            
            result.log_result()
            return result
            
        except Exception as e:
            error_msg = f"Error loading model {model_name}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Clean memory on error
            if self.auto_clean_memory and self.gpu_info.available:
                self.gpu_info.clean_memory()
            
            # Create and return error result
            return ModelLoadingResult(
                model_name=model_name,
                success=False,
                error=e,
                loading_time=time.time() - start_time,
                memory_used=self.gpu_info.get_memory_info(),
                loading_params=loading_params
            )
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if model was unloaded, False if not found
        """
        if model_name not in self.loaded_models:
            return False
        
        # Remove references to model and processor
        self.loaded_models.pop(model_name)
        
        # Clean up memory
        if self.gpu_info.available:
            self.gpu_info.clean_memory()
        
        logger.info(f"Unloaded model: {model_name}")
        return True
    
    def unload_all_models(self) -> None:
        """Unload all loaded models from memory."""
        model_names = list(self.loaded_models.keys())
        for model_name in model_names:
            self.unload_model(model_name)
        
        logger.info(f"Unloaded all models ({len(model_names)} total)")
    
    def get_loaded_model(
        self,
        model_name: str
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Get a loaded model and processor.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Tuple of (model, processor, loading_info)
            
        Raises:
            ModelNotFoundError: If model is not loaded
        """
        if model_name not in self.loaded_models:
            raise ModelNotFoundError(f"Model {model_name} is not loaded")
        
        loaded = self.loaded_models[model_name]
        return loaded["model"], loaded["processor"], loaded
    
    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a model is currently loaded.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model is loaded, False otherwise
        """
        return model_name in self.loaded_models
    
    def optimize_model_memory(self, model_name: str) -> Dict[str, Any]:
        """
        Apply memory optimization techniques to a loaded model.
        
        Args:
            model_name: Name of the model to optimize
            
        Returns:
            Dictionary with optimization results
            
        Raises:
            ModelNotFoundError: If model is not loaded
        """
        if model_name not in self.loaded_models:
            raise ModelNotFoundError(f"Model {model_name} is not loaded")
        
        model = self.loaded_models[model_name]["model"]
        
        try:
            # Record memory before optimization
            before_info = self.gpu_info.get_memory_info()
            
            # Enable gradient checkpointing if available
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info(f"Enabled gradient checkpointing for {model_name}")
            
            # Move model to appropriate device if not already there
            if torch.cuda.is_available() and not next(model.parameters()).is_cuda:
                model.to('cuda')
                logger.info(f"Moved model {model_name} to CUDA")
            
            # Apply any model-specific optimizations
            model_config = self.get_model_config(model_name)
            if "optimization" in model_config.to_dict():
                # Apply custom optimizations based on model type
                pass
            
            # Record memory after optimization
            after_info = self.gpu_info.get_memory_info()
            
            # Return results
            return {
                "model_name": model_name,
                "before": before_info,
                "after": after_info,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error optimizing model {model_name}: {e}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def save_model_metadata(
        self,
        model_name: str,
        metadata_dir: Optional[str] = None,
        include_quantization_details: bool = True
    ) -> Optional[str]:
        """
        Save metadata about a model with enhanced quantization information.
        
        Args:
            model_name: Name of the model
            metadata_dir: Optional directory to save metadata
            include_quantization_details: Whether to include detailed quantization info
            
        Returns:
            Path to saved metadata file or None if model not found
            
        Raises:
            ModelNotFoundError: If model is not in registry
        """
        # Check if model exists in registry
        if model_name not in self.model_configs:
            raise ModelNotFoundError(f"Model {model_name} not found in registry")
        
        # Get model configuration
        model_config = self.model_configs[model_name]
        
        # Create metadata dictionary
        metadata = {
            "model_name": model_name,
            "repo_id": model_config.repo_id,
            "model_type": model_config.model_type,
            "processor_type": model_config.processor_type,
            "description": model_config.description,
            "hardware_requirements": model_config.hardware_requirements,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add loading information if model is loaded
        if model_name in self.loaded_models:
            loaded_info = self.loaded_models[model_name]
            metadata.update({
                "loaded": True,
                "loaded_at": loaded_info.get("loaded_at"),
                "loading_time": loaded_info.get("loading_time"),
                "memory_info": loaded_info.get("memory_info")
            })
            
            # Add current quantization if model is loaded
            try:
                # Try to determine current quantization
                loading_params = loaded_info.get("loading_params", {})
                current_quantization = "unknown"
                
                if loading_params.get("load_in_4bit", False):
                    current_quantization = "int4"
                elif loading_params.get("load_in_8bit", False):
                    current_quantization = "int8"
                elif "torch_dtype" in loading_params:
                    dtype = str(loading_params["torch_dtype"])
                    if "bfloat16" in dtype:
                        current_quantization = "bfloat16"
                    elif "float16" in dtype:
                        current_quantization = "float16"
                    elif "float32" in dtype:
                        current_quantization = "float32"
                
                metadata["current_quantization"] = current_quantization
                
                # Add memory usage analysis
                if "memory_info" in loaded_info:
                    memory_info = loaded_info["memory_info"]
                    if "total_memory_gb" in memory_info and "allocated_memory_gb" in memory_info:
                        metadata["memory_usage_percent"] = (
                            memory_info["allocated_memory_gb"] / memory_info["total_memory_gb"] * 100
                        )
                    
                    # Add memory per parameter estimate if we can calculate it
                    if hasattr(loaded_info["model"], "num_parameters"):
                        num_params = loaded_info["model"].num_parameters()
                        if num_params > 0 and "allocated_memory_gb" in memory_info:
                            metadata["memory_per_billion_params_gb"] = (
                                memory_info["allocated_memory_gb"] / (num_params / 1e9)
                            )
            except Exception as e:
                logger.warning(f"Error analyzing current quantization: {e}")
        else:
            metadata["loaded"] = False
        
        # Add GPU information
        metadata["gpu_info"] = self.gpu_info.get_memory_info()
        
        # Add detailed quantization information if requested
        if include_quantization_details:
            try:
                # Get all available quantization strategies
                strategies = model_config.get_available_quantization_strategies()
                quantization_data = {}
                
                for strategy in strategies:
                    try:
                        strategy_metadata = model_config.get_quantization_metadata(strategy)
                        quantization_data[strategy] = strategy_metadata
                    except Exception as e:
                        logger.warning(f"Error getting metadata for strategy {strategy}: {e}")
                
                metadata["quantization"] = {
                    "available_strategies": strategies,
                    "strategy_details": quantization_data
                }
                
                # Add recommendation
                if self.gpu_info.available:
                    try:
                        recommended = self.get_optimal_quantization(model_name)
                        metadata["recommended_quantization"] = recommended
                    except Exception as e:
                        logger.warning(f"Error determining recommended quantization: {e}")
            except Exception as e:
                logger.warning(f"Error adding quantization details: {e}")
        
        # Determine save path
        if metadata_dir:
            save_dir = metadata_dir
        else:
            # Use processed directory in current experiment
            save_dir = self.paths.get("processed_dir", "results/processed")
        
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metadata
        filename = f"{model_name}_metadata.json"
        save_path = os.path.join(save_dir, filename)
        
        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved model metadata to {save_path}")
        return save_path
    
    def execute_model(
        self,
        model_name: str,
        inputs: Dict[str, Any],
        inference_params: Optional[Dict[str, Any]] = None,
        load_if_needed: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a model with given inputs.
        
        Args:
            model_name: Name of the model to execute
            inputs: Dictionary of inputs for the model
            inference_params: Optional inference parameters
            load_if_needed: Whether to load the model if not already loaded
            **kwargs: Additional parameters for model execution
            
        Returns:
            Dictionary with execution results
            
        Raises:
            ModelNotFoundError: If model is not found
            ModelLoadingError: If model loading fails
            ModelExecutionError: If model execution fails
        """
        # Check if model is loaded
        if not self.is_model_loaded(model_name):
            if not load_if_needed:
                raise ModelNotFoundError(f"Model {model_name} is not loaded")
            
            # Try to load the model
            loading_result = self.load_model(model_name)
            if not loading_result.success:
                raise ModelLoadingError(
                    f"Failed to load model {model_name}: {loading_result.error}"
                )
        
        # Get loaded model and processor
        model, processor, loading_info = self.get_loaded_model(model_name)
        
        # Get model configuration
        model_config = self.get_model_config(model_name)
        
        # Prepare inference parameters
        if inference_params is None:
            inference_params = model_config.get_inference_params()
        else:
            # Merge with defaults
            defaults = model_config.get_inference_params()
            defaults.update(inference_params)
            inference_params = defaults
        
        # Add kwargs to inference parameters
        inference_params.update(kwargs)
        
        # Record starting time and memory
        start_time = time.time()
        start_memory = self.gpu_info.get_memory_info()
        
        try:
            # Execute model
            with torch.no_grad():
                outputs = model.generate(**inputs, **inference_params)
            
            # Process outputs with processor if needed
            if hasattr(processor, 'batch_decode'):
                decoded_outputs = processor.batch_decode(
                    outputs, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            else:
                decoded_outputs = outputs
            
            # Calculate execution time and memory usage
            execution_time = time.time() - start_time
            end_memory = self.gpu_info.get_memory_info()
            
            # Create result dictionary
            result = {
                "success": True,
                "model_name": model_name,
                "outputs": outputs,
                "decoded_outputs": decoded_outputs,
                "execution_time": execution_time,
                "start_memory": start_memory,
                "end_memory": end_memory,
                "inference_params": inference_params
            }
            
            # Log successful execution
            logger.info(
                f"Model {model_name} executed successfully in {execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            # Log error
            error_msg = f"Error executing model {model_name}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Raise execution error
            raise ModelExecutionError(error_msg) from e
    
    def compare_quantization_strategies(
        self, 
        model_name: str, 
        strategies: Optional[List[str]] = None,
        test_inputs: Optional[Dict[str, Any]] = None,
        inference_params: Optional[Dict[str, Any]] = None,
        metrics: List[str] = ["memory_usage", "loading_time", "inference_time"],
        save_results: bool = True,
        results_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare different quantization strategies for a model.
        
        Args:
            model_name: Name of the model to test
            strategies: List of quantization strategies to compare (None = all available)
            test_inputs: Optional inputs to use for inference testing
            inference_params: Optional inference parameters
            metrics: List of metrics to include in comparison
            save_results: Whether to save results to file
            results_dir: Directory to save results (None = default processed dir)
            
        Returns:
            Dictionary with comparison results
            
        Raises:
            ModelNotFoundError: If model is not found in registry
            ModelConfigurationError: If any strategy is invalid
        """
        # Get model configuration
        model_config = self.get_model_config(model_name)
        
        # Get list of strategies to test
        if strategies is None:
            strategies = model_config.get_available_quantization_strategies()
        
        if not strategies:
            raise ModelConfigurationError(f"No quantization strategies found for model {model_name}")
        
        # Initialize results dictionary
        comparison_results = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "gpu_info": self.gpu_info.get_memory_info(),
            "strategies_compared": strategies,
            "results": {}
        }
        
        # Test each strategy
        for strategy in strategies:
            try:
                logger.info(f"Testing quantization strategy: {strategy}")
                strategy_results = {}
                
                # Unload any existing model
                if self.is_model_loaded(model_name):
                    self.unload_model(model_name)
                
                # Clean GPU memory
                if self.gpu_info.available:
                    self.gpu_info.clean_memory()
                
                # Record starting memory
                start_memory = self.gpu_info.get_memory_info()
                strategy_results["start_memory"] = start_memory
                
                # Load model with this strategy
                loading_result = self.load_model(model_name, quantization=strategy)
                
                # Record loading results
                strategy_results["loading_success"] = loading_result.success
                strategy_results["loading_time"] = loading_result.loading_time
                strategy_results["quantization_info"] = loading_result.get_quantization_info()
                
                # If loading failed, skip to next strategy
                if not loading_result.success:
                    strategy_results["error"] = str(loading_result.error)
                    comparison_results["results"][strategy] = strategy_results
                    continue
                
                # Record memory usage after loading
                post_load_memory = self.gpu_info.get_memory_info()
                strategy_results["post_load_memory"] = post_load_memory
                strategy_results["memory_usage_gb"] = post_load_memory["allocated_memory_gb"]
                
                # Calculate memory diff from start
                strategy_results["memory_increase_gb"] = (
                    post_load_memory["allocated_memory_gb"] - 
                    start_memory["allocated_memory_gb"]
                )
                
                # If test inputs provided, try inference
                if test_inputs is not None:
                    try:
                        # Get model and processor
                        model, processor, _ = self.get_loaded_model(model_name)
                        
                        # Run inference test
                        inference_start = time.time()
                        with torch.no_grad():
                            outputs = model.generate(**test_inputs, **(inference_params or {}))
                        inference_time = time.time() - inference_start
                        
                        # Record inference results
                        strategy_results["inference_success"] = True
                        strategy_results["inference_time"] = inference_time
                        
                        # Process outputs if needed
                        if hasattr(processor, 'batch_decode'):
                            decoded = processor.batch_decode(
                                outputs, 
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False
                            )
                            strategy_results["output_length"] = len(decoded[0]) if decoded else 0
                        
                        # Record memory after inference
                        post_inference_memory = self.gpu_info.get_memory_info()
                        strategy_results["post_inference_memory"] = post_inference_memory
                        strategy_results["inference_memory_increase_gb"] = (
                            post_inference_memory["allocated_memory_gb"] - 
                            post_load_memory["allocated_memory_gb"]
                        )
                    except Exception as e:
                        strategy_results["inference_success"] = False
                        strategy_results["inference_error"] = str(e)
                
                # Add metadata about the quantization strategy
                strategy_results["strategy_metadata"] = model_config.get_quantization_metadata(strategy)
                
                # Store results for this strategy
                comparison_results["results"][strategy] = strategy_results
                
                # Unload model
                self.unload_model(model_name)
                
            except Exception as e:
                logger.error(f"Error testing strategy {strategy}: {e}")
                comparison_results["results"][strategy] = {
                    "success": False,
                    "error": str(e)
                }
                
                # Make sure model is unloaded after an error
                if self.is_model_loaded(model_name):
                    self.unload_model(model_name)
        
        # Add summary metrics
        comparison_results["summary"] = self._summarize_quantization_comparison(comparison_results["results"])
        
        # Save results if requested
        if save_results:
            save_path = self._save_quantization_comparison(
                model_name=model_name,
                comparison_results=comparison_results,
                save_dir=results_dir
            )
            comparison_results["saved_to"] = save_path
        
        return comparison_results
    
    def _summarize_quantization_comparison(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create summary metrics from quantization comparison results.
        
        Args:
            results: Dictionary of results by strategy
            
        Returns:
            Dictionary with summary metrics
        """
        summary = {
            "strategies_count": len(results),
            "successful_strategies": 0,
            "fastest_loading": {"strategy": None, "time": float('inf')},
            "fastest_inference": {"strategy": None, "time": float('inf')},
            "lowest_memory": {"strategy": None, "memory_gb": float('inf')},
            "tradeoffs": {}
        }
        
        # Process each strategy's results
        for strategy, data in results.items():
            # Count successful strategies
            if data.get("loading_success", False):
                summary["successful_strategies"] += 1
                
                # Check if this is the fastest loading
                loading_time = data.get("loading_time", float('inf'))
                if loading_time < summary["fastest_loading"]["time"]:
                    summary["fastest_loading"] = {
                        "strategy": strategy,
                        "time": loading_time
                    }
                
                # Check if this has the lowest memory usage
                memory_usage = data.get("memory_usage_gb", float('inf'))
                if memory_usage < summary["lowest_memory"]["memory_gb"]:
                    summary["lowest_memory"] = {
                        "strategy": strategy,
                        "memory_gb": memory_usage
                    }
                
                # Check if this is the fastest inference
                if "inference_time" in data and data.get("inference_success", False):
                    inference_time = data.get("inference_time", float('inf'))
                    if inference_time < summary["fastest_inference"]["time"]:
                        summary["fastest_inference"] = {
                            "strategy": strategy,
                            "time": inference_time
                        }
        
        # Calculate trade-offs (memory vs. speed)
        viable_strategies = {}
        for strategy, data in results.items():
            if (data.get("loading_success", False) and 
                "memory_usage_gb" in data):
                # Only include strategies with both memory and speed metrics
                if "inference_time" in data and data.get("inference_success", False):
                    viable_strategies[strategy] = {
                        "memory_gb": data["memory_usage_gb"],
                        "inference_time": data["inference_time"],
                        "loading_time": data.get("loading_time", 0)
                    }
        
        # Calculate efficiency scores (lower is better)
        for strategy, metrics in viable_strategies.items():
            memory_score = metrics["memory_gb"] / 4  # Normalize to approx. 0-10 range
            inference_score = metrics["inference_time"] * 5  # Normalize to approx. 0-10 range
            
            # Combined score (equal weighting)
            efficiency_score = memory_score + inference_score
            
            summary["tradeoffs"][strategy] = {
                "memory_gb": metrics["memory_gb"],
                "inference_time": metrics["inference_time"],
                "efficiency_score": efficiency_score
            }
        
        # Find best overall strategy (lowest efficiency score)
        if summary["tradeoffs"]:
            best_strategy = min(
                summary["tradeoffs"].items(),
                key=lambda x: x[1]["efficiency_score"]
            )[0]
            
            summary["recommended_strategy"] = best_strategy
        
        return summary
    
    def _save_quantization_comparison(
        self,
        model_name: str,
        comparison_results: Dict[str, Any],
        save_dir: Optional[str] = None
    ) -> str:
        """
        Save quantization comparison results to file.
        
        Args:
            model_name: Name of the model
            comparison_results: Results dictionary
            save_dir: Directory to save results
            
        Returns:
            Path to saved file
        """
        # Determine save directory
        if save_dir:
            directory = save_dir
        else:
            # Use processed directory in experiment
            directory = self.paths.get("processed_dir", "results/processed")
        
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_quantization_comparison_{timestamp}.json"
        file_path = os.path.join(directory, filename)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        logger.info(f"Saved quantization comparison results to {file_path}")
        return file_path
    
    def get_optimal_quantization(
        self,
        model_name: str,
        available_memory_gb: Optional[float] = None,
        prioritize: str = "balanced"
    ) -> str:
        """
        Determine the optimal quantization strategy based on hardware.
        
        Args:
            model_name: Name of the model to load
            available_memory_gb: Available GPU memory in GB (auto-detect if None)
            prioritize: What to prioritize ('memory', 'speed', or 'balanced')
            
        Returns:
            Name of recommended quantization strategy
        """
        # Get model config
        model_config = self.get_model_config(model_name)
        
        # Get available strategies
        strategies = model_config.get_available_quantization_strategies()
        if not strategies:
            raise ModelConfigurationError(f"No quantization strategies found for model {model_name}")
        
        # Determine available memory
        if available_memory_gb is None:
            if self.gpu_info.available:
                memory_info = self.gpu_info.get_memory_info()
                available_memory_gb = memory_info["free_memory_gb"]
            else:
                # No GPU, use minimum possible
                return self._get_minimum_memory_strategy(model_config)
        
        # Calculate minimum memory needed with safety margin
        model_min_memory = model_config.get_memory_requirement_gb()
        required_memory = model_min_memory * 1.1  # 10% safety margin
        
        # If available memory is less than required with any strategy,
        # choose the most aggressive quantization
        if available_memory_gb < required_memory:
            return self._get_minimum_memory_strategy(model_config)
        
        # Filter strategies based on memory requirements
        viable_strategies = []
        for strategy in strategies:
            try:
                strategy_metadata = model_config.get_quantization_metadata(strategy)
                memory_factor = strategy_metadata.get("memory_reduction_factor", 1.0)
                strategy_min_memory = model_min_memory / memory_factor
                
                if strategy_min_memory <= available_memory_gb:
                    viable_strategies.append((strategy, strategy_metadata))
            except Exception as e:
                logger.warning(f"Error analyzing strategy {strategy}: {e}")
        
        if not viable_strategies:
            return self._get_minimum_memory_strategy(model_config)
        
        # Choose based on priority
        if prioritize == "memory":
            # Choose strategy with lowest memory usage
            return min(
                viable_strategies,
                key=lambda x: model_min_memory / x[1].get("memory_reduction_factor", 1.0)
            )[0]
        elif prioritize == "speed":
            # Choose strategy with highest precision (fastest)
            return max(
                viable_strategies,
                key=lambda x: x[1].get("bits", 0)
            )[0]
        else:  # balanced
            # Default to bfloat16 if available, otherwise most balanced
            for strategy, _ in viable_strategies:
                metadata = model_config.get_quantization_metadata(strategy)
                if metadata.get("precision_name") == "BF16":
                    return strategy
            
            # Otherwise, middle ground between int8 and float32
            return sorted(
                viable_strategies,
                key=lambda x: abs(x[1].get("bits", 0) - 16)
            )[0][0]
    
    def _get_minimum_memory_strategy(self, model_config: ModelConfig) -> str:
        """Get the strategy with minimum memory requirements."""
        strategies = model_config.get_available_quantization_strategies()
        
        # If no strategies, return default
        if not strategies:
            return "default"
        
        try:
            # Find strategy with highest memory reduction factor
            min_memory_strategy = max(
                [(s, model_config.get_quantization_metadata(s).get("memory_reduction_factor", 1.0)) 
                 for s in strategies],
                key=lambda x: x[1]
            )[0]
            return min_memory_strategy
        except Exception as e:
            logger.warning(f"Error finding minimum memory strategy: {e}")
            return strategies[0]  # First available strategy as fallback


# Singleton instance for global access
_model_service = None


def get_model_service() -> ModelService:
    """
    Get the ModelService singleton instance.
    
    Returns:
        ModelService instance
    """
    global _model_service
    
    if _model_service is None:
        _model_service = ModelService()
    
    return _model_service


def load_model(
    model_name: str,
    quantization: Optional[str] = None,
    force_reload: bool = False,
    **kwargs
) -> Tuple[Any, Any]:
    """
    Load a model using the ModelService.
    
    This is a convenience function for direct model loading.
    
    Args:
        model_name: Name of the model to load
        quantization: Optional quantization strategy
        force_reload: Whether to reload if already loaded
        **kwargs: Additional parameters for model loading
        
    Returns:
        Tuple of (model, processor)
        
    Raises:
        ModelLoadingError: If model loading fails
    """
    service = get_model_service()
    
    result = service.load_model(
        model_name=model_name,
        quantization=quantization,
        force_reload=force_reload,
        **kwargs
    )
    
    if not result.success:
        raise ModelLoadingError(f"Failed to load model {model_name}: {result.error}")
    
    return result.model, result.processor


def unload_all_models() -> None:
    """
    Unload all models and clean up memory.
    
    This is a convenience function for easy cleanup.
    """
    service = get_model_service()
    service.unload_all_models()


def optimize_memory() -> Dict[str, Any]:
    """
    Optimize GPU memory usage.
    
    Returns:
        Dictionary with memory information
    """
    service = get_model_service()
    return service.gpu_info.clean_memory()


def get_gpu_info() -> Dict[str, Any]:
    """
    Get current GPU information.
    
    Returns:
        Dictionary with GPU information
    """
    service = get_model_service()
    return service.gpu_info.get_memory_info()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model service
    service = get_model_service()
    
    # Print available models
    print(f"Available models: {service.list_available_models()}")
    
    # Print GPU information
    print(f"GPU info: {service.gpu_info}")
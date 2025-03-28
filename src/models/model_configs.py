"""
Model configuration data classes and validation utilities.

This module provides:
1. Structured classes for model configurations
2. Validation functions for configuration fields
3. Utilities for working with model configuration objects
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class HardwareRequirements:
    """Hardware requirements for models."""
    gpu_required: bool = False
    gpu_memory_min: str = "0GB"
    recommended_gpu: str = ""
    cpu_fallback: bool = False
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardwareRequirements':
        """Create a HardwareRequirements instance from a dictionary."""
        return cls(
            gpu_required=data.get('gpu_required', False),
            gpu_memory_min=data.get('gpu_memory_min', "0GB"),
            recommended_gpu=data.get('recommended_gpu', ""),
            cpu_fallback=data.get('cpu_fallback', False)
        )


@dataclass
class QuantizationOption:
    """Configuration for a specific quantization strategy."""
    torch_dtype: Optional[str] = None
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: Optional[str] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantizationOption':
        """Create a QuantizationOption instance from a dictionary."""
        # Extract known fields
        additional_params = {k: v for k, v in data.items() 
                            if k not in ['torch_dtype', 'device_map', 'load_in_8bit',
                                        'load_in_4bit', 'bnb_4bit_compute_dtype']}
        
        return cls(
            torch_dtype=data.get('torch_dtype'),
            device_map=data.get('device_map', "auto"),
            load_in_8bit=data.get('load_in_8bit', False),
            load_in_4bit=data.get('load_in_4bit', False),
            bnb_4bit_compute_dtype=data.get('bnb_4bit_compute_dtype'),
            additional_params=additional_params
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model loading."""
        result = {}
        
        # Add basic fields if they have values
        if self.torch_dtype:
            # Convert string dtype to actual torch dtype
            if self.torch_dtype == "bfloat16":
                import torch
                result["torch_dtype"] = torch.bfloat16
            elif self.torch_dtype == "float16":
                import torch
                result["torch_dtype"] = torch.float16
            elif self.torch_dtype == "float32":
                import torch
                result["torch_dtype"] = torch.float32
            else:
                result["torch_dtype"] = self.torch_dtype
                
        if self.device_map:
            result["device_map"] = self.device_map
            
        if self.load_in_8bit:
            result["load_in_8bit"] = True
            
        if self.load_in_4bit:
            result["load_in_4bit"] = True
            
        if self.bnb_4bit_compute_dtype:
            # Convert string dtype to actual torch dtype
            if self.bnb_4bit_compute_dtype == "bfloat16":
                import torch
                result["bnb_4bit_compute_dtype"] = torch.bfloat16
            elif self.bnb_4bit_compute_dtype == "float16":
                import torch
                result["bnb_4bit_compute_dtype"] = torch.float16
            else:
                result["bnb_4bit_compute_dtype"] = self.bnb_4bit_compute_dtype
        
        # Add any additional parameters
        result.update(self.additional_params)
        
        return result


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    default: str = "bfloat16"
    options: Dict[str, QuantizationOption] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantizationConfig':
        """Create a QuantizationConfig instance from a dictionary."""
        options = {}
        for key, option_data in data.get('options', {}).items():
            options[key] = QuantizationOption.from_dict(option_data)
            
        return cls(
            default=data.get('default', "bfloat16"),
            options=options
        )


@dataclass
class InferenceParams:
    """Parameters for model inference."""
    max_new_tokens: int = 50
    do_sample: bool = False
    temperature: float = 1.0
    batch_size: int = 1
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InferenceParams':
        """Create an InferenceParams instance from a dictionary."""
        # Extract known fields
        additional_params = {k: v for k, v in data.items() 
                            if k not in ['max_new_tokens', 'do_sample', 
                                         'temperature', 'batch_size']}
        
        return cls(
            max_new_tokens=data.get('max_new_tokens', 50),
            do_sample=data.get('do_sample', False),
            temperature=data.get('temperature', 1.0),
            batch_size=data.get('batch_size', 1),
            additional_params=additional_params
        )


@dataclass
class ModelConfig:
    """Structured representation of a model configuration."""
    name: str
    repo_id: str
    model_type: str
    processor_type: str = "AutoProcessor"
    description: str = ""
    hardware: HardwareRequirements = field(default_factory=HardwareRequirements)
    loading: Dict[str, Any] = field(default_factory=dict)
    quantization: Optional[QuantizationConfig] = None
    inference: InferenceParams = field(default_factory=InferenceParams)
    environments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], model_name: Optional[str] = None) -> 'ModelConfig':
        """Create a ModelConfig instance from a dictionary."""
        # Use provided name or from data
        name = model_name or data.get('name', "unknown")
        
        # Required fields
        if 'repo_id' not in data:
            raise ValueError(f"Model configuration for {name} missing required field: repo_id")
        
        if 'model_type' not in data:
            raise ValueError(f"Model configuration for {name} missing required field: model_type")
        
        # Parse nested configurations
        hardware = HardwareRequirements.from_dict(data.get('hardware', {}))
        
        quantization = None
        if 'quantization' in data:
            quantization = QuantizationConfig.from_dict(data['quantization'])
            
        inference = InferenceParams.from_dict(data.get('inference', {}))
        
        return cls(
            name=name,
            repo_id=data['repo_id'],
            model_type=data['model_type'],
            processor_type=data.get('processor_type', "AutoProcessor"),
            description=data.get('description', ""),
            hardware=hardware,
            loading=data.get('loading', {}),
            quantization=quantization,
            inference=inference,
            environments=data.get('environments', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    def get_loading_params(self, quantization: Optional[str] = None) -> Dict[str, Any]:
        """
        Get parameters for loading this model.
        
        Args:
            quantization: Optional quantization strategy to use
            
        Returns:
            Dictionary of parameters to pass to model loading function
        """
        # Start with default loading configuration
        loading_params = self.loading.copy()
        
        # Apply quantization configuration if specified
        if quantization is not None and self.quantization is not None:
            quant_options = self.quantization.options.get(quantization)
            
            if quant_options:
                # Convert quantization options to dict and update loading parameters
                loading_params.update(quant_options.to_dict())
            else:
                logger.warning(f"Requested quantization '{quantization}' not found for {self.name}")
                
        elif self.quantization is not None:
            # Use default quantization if available
            default_quant = self.quantization.default
            quant_options = self.quantization.options.get(default_quant)
            
            if quant_options:
                loading_params.update(quant_options.to_dict())
        
        return loading_params


def validate_model_config(config: Dict[str, Any], model_name: str) -> List[str]:
    """
    Validate a model configuration dictionary.
    
    Args:
        config: Model configuration dictionary
        model_name: Name of the model for error messages
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    required_fields = ["repo_id", "model_type"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate nested configurations
    if "hardware" in config and not isinstance(config["hardware"], dict):
        errors.append("Hardware configuration must be a dictionary")
        
    if "quantization" in config:
        quant = config["quantization"]
        if not isinstance(quant, dict):
            errors.append("Quantization configuration must be a dictionary")
        elif "options" in quant and not isinstance(quant["options"], dict):
            errors.append("Quantization options must be a dictionary")
        elif "default" in quant and "options" in quant and quant["default"] not in quant["options"]:
            errors.append(f"Default quantization '{quant['default']}' not found in options")
    
    # Validate inference parameters
    if "inference" in config:
        inf = config["inference"]
        if not isinstance(inf, dict):
            errors.append("Inference configuration must be a dictionary")
        else:
            if "max_new_tokens" in inf and not isinstance(inf["max_new_tokens"], int):
                errors.append("max_new_tokens must be an integer")
            if "temperature" in inf and not isinstance(inf["temperature"], (int, float)):
                errors.append("temperature must be a number")
    
    return errors


def create_model_config(config_dict: Dict[str, Any], 
                      model_name: Optional[str] = None) -> ModelConfig:
    """
    Create a ModelConfig object from a configuration dictionary.
    
    Args:
        config_dict: Dictionary containing model configuration
        model_name: Optional model name if not in config_dict
        
    Returns:
        ModelConfig instance
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate configuration
    errors = validate_model_config(config_dict, model_name or config_dict.get('name', 'unknown'))
    
    if errors:
        error_msg = f"Invalid model configuration: {', '.join(errors)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Create ModelConfig object
    return ModelConfig.from_dict(config_dict, model_name)


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample configuration
    sample_config = {
        "repo_id": "mistral-community/pixtral-12b",
        "model_type": "LlavaForConditionalGeneration",
        "processor_type": "AutoProcessor",
        "description": "Pixtral 12B vision-language model",
        "hardware": {
            "gpu_required": True,
            "gpu_memory_min": "24GB"
        },
        "loading": {
            "device_map": "cuda:0"
        },
        "quantization": {
            "default": "bfloat16",
            "options": {
                "bfloat16": {
                    "torch_dtype": "bfloat16",
                    "device_map": "cuda:0"
                }
            }
        },
        "inference": {
            "max_new_tokens": 50,
            "do_sample": False
        }
    }
    
    # Create a model config and test methods
    try:
        model_config = create_model_config(sample_config, "test-model")
        print(f"Created model config for {model_config.name}")
        print(f"Repository: {model_config.repo_id}")
        print(f"Model type: {model_config.model_type}")
        
        # Test getting loading parameters
        loading_params = model_config.get_loading_params()
        print("\nLoading parameters:")
        for key, value in loading_params.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {str(e)}")
```


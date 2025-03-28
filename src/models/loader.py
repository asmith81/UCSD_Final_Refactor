"""
Model loading utilities for invoice processing.

This module provides functionality to load and initialize various vision-language models
with appropriate configurations for invoice data extraction tasks.
"""

import os
import time
import logging
import torch
from typing import Dict, Any, Tuple, Optional, List

# Import project utilities
from src.config.environment import get_environment_config
from src.models.registry import get_model_config, get_loading_params

# Set up logging
logger = logging.getLogger(__name__)

def get_gpu_memory_info() -> Dict[str, Any]:
    """
    Get current GPU memory usage information.
    
    Returns:
        Dictionary with GPU memory information
    """
    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "memory_allocated": 0,
            "memory_reserved": 0,
            "memory_free": 0
        }
    
    try:
        # Get device properties
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Get current memory usage
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        free = total_memory - reserved
        
        return {
            "gpu_available": True,
            "device_name": device_name,
            "total_memory_gb": float(total_memory) / 1e9,
            "allocated_memory_gb": float(allocated) / 1e9,
            "reserved_memory_gb": float(reserved) / 1e9, 
            "free_memory_gb": float(free) / 1e9,
            "utilization_percent": float(allocated) / total_memory * 100
        }
    except Exception as e:
        logger.error(f"Error getting GPU memory info: {e}")
        return {
            "gpu_available": True,
            "error": str(e)
        }

def clean_gpu_memory() -> None:
    """
    Clean up GPU memory by emptying cache and triggering garbage collection.
    
    Returns:
        None
    """
    if torch.cuda.is_available():
        try:
            # Free unused memory
            torch.cuda.empty_cache()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("GPU memory cleaned and cache cleared")
        except Exception as e:
            logger.error(f"Error cleaning GPU memory: {e}")

def get_model_class(model_type: str):
    """
    Dynamically import and return the appropriate model class.
    
    Args:
        model_type: String representing the model class to load
        
    Returns:
        Model class for loading
    """
    try:
        if model_type == "LlavaForConditionalGeneration":
            from transformers import LlavaForConditionalGeneration
            return LlavaForConditionalGeneration
        elif model_type == "AutoModelForVision2Seq":
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq
        else:
            # Fallback to dynamic import
            from importlib import import_module
            
            # Split module and class name if fully qualified
            if '.' in model_type:
                module_path, class_name = model_type.rsplit('.', 1)
                module = import_module(module_path)
                return getattr(module, class_name)
            else:
                # Try importing from transformers as a last resort
                from transformers import AutoModel
                return AutoModel
    except ImportError as e:
        logger.error(f"Could not import model class {model_type}: {e}")
        raise

def load_model_and_processor(
    model_name: str, 
    quantization: Optional[str] = None, 
    cache_dir: Optional[str] = None, 
    **kwargs
) -> Tuple[Any, Any]:
    """
    Load model and processor with dynamic parameter filtering.
    
    Args:
        model_name: Name or repository ID of the model
        quantization: Optional quantization strategy
        cache_dir: Directory to cache model files
        **kwargs: Additional parameters for model loading
        
    Returns:
        Tuple of (model, processor)
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        # Record starting memory usage and time
        start_memory = get_gpu_memory_info()
        start_time = time.time()
        
        # Clean GPU memory before loading
        clean_gpu_memory()
        
        # Get model configuration
        model_config = get_model_config(model_name)
        
        # Get repository ID for model loading
        repo_id = model_config.get("repo_id", model_name)
        
        # Determine supported parameters
        supported_params = model_config.get("supported_parameters", [
            "device_map", 
            "torch_dtype", 
            "max_memory", 
            "low_cpu_mem_usage", 
            "trust_remote_code",
            "quantization_config",
            "attn_implementation"
        ])
        
        # Filter kwargs to only include supported parameters
        loading_params = {
            k: v for k, v in kwargs.items() 
            if k in supported_params
        }
        
        # Add cache directory if provided
        if cache_dir:
            loading_params["cache_dir"] = cache_dir
        
        # Get model and processor configuration
        model_type = model_config.get("model_type", "LlavaForConditionalGeneration")
        processor_type = model_config.get("processor_type", "AutoProcessor")
        
        # Get appropriate model class
        model_class = get_model_class(model_type)
        
        # Import processor
        from transformers import AutoProcessor
        
        # Log loading parameters for debugging
        logger.info(f"Loading model {model_name} from {repo_id}")
        logger.info(f"Loading parameters: {loading_params}")
        
        # Load the model
        model = model_class.from_pretrained(repo_id, **loading_params)
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            repo_id, 
            trust_remote_code=loading_params.get("trust_remote_code", True)
        )
        
        # Log memory and time usage
        end_memory = get_gpu_memory_info()
        loading_time = time.time() - start_time
        
        logger.info(f"Model loaded successfully in {loading_time:.2f} seconds")
        logger.info(f"Memory usage: {end_memory['allocated_memory_gb']:.2f} GB")
        
        return model, processor
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        
        # Clean up GPU memory on error
        clean_gpu_memory()
        
        # Raise a more informative error
        raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

def optimize_model_memory(model: Any) -> None:
    """
    Apply additional memory optimization techniques to the model.
    
    Args:
        model: Loaded model to optimize
    """
    try:
        # Move model to GPU if not already there
        if torch.cuda.is_available():
            model.to('cuda')
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
        
        logger.info("Model memory optimization completed")
    except Exception as e:
        logger.warning(f"Error during model memory optimization: {e}")

def get_model_size(model: Any) -> Dict[str, float]:
    """
    Calculate the size of the model in memory.
    
    Args:
        model: Loaded model to analyze
        
    Returns:
        Dictionary with model size information
    """
    try:
        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage
        param_size = 4  # Assuming 4 bytes per parameter (float32)
        total_size_mb = total_params * param_size / 1024 / 1024
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_size_mb": total_size_mb
        }
    except Exception as e:
        logger.warning(f"Error calculating model size: {e}")
        return {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "estimated_size_mb": 0
        }
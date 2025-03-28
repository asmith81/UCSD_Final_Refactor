"""
Core inference functions for extracting fields from invoice images.

This module provides the core functionality for:
1. Processing images with vision-language models
2. Extracting specific fields using prompts
3. Measuring performance and accuracy metrics
4. Handling errors and edge cases

This implementation is designed to work with the configuration-driven
architecture and registry patterns used throughout the project.
"""

import os
import time
import logging
import re
from typing import Dict, Any, Tuple, Optional, List, Union, Callable
from pathlib import Path

import torch
from PIL import Image
from Levenshtein import distance

# Import project modules
from src.models.loader import load_model_and_processor, optimize_model_memory, get_gpu_memory_info
from src.models.registry import get_model_config
from src.prompts.registry import get_prompt, Prompt
from src.config.environment import get_environment_config

# Set up logging
logger = logging.getLogger(__name__)

# GPU memory tracking

def track_gpu_memory(operation_name: str = "unknown") -> Dict[str, Any]:
    """
    Track GPU memory usage for a specific operation.
    
    Args:
        operation_name: Name of the operation being tracked
        
    Returns:
        Dictionary of memory metrics
    """
    if not torch.cuda.is_available():
        return {
            "operation": operation_name,
            "available": False,
            "timestamp": time.time()
        }
    
    try:
        # Get memory information
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        reserved_memory = torch.cuda.memory_reserved(0)
        free_memory = total_memory - allocated_memory
        
        # Convert to GB
        total_memory_gb = total_memory / 1e9
        allocated_memory_gb = allocated_memory / 1e9
        reserved_memory_gb = reserved_memory / 1e9
        free_memory_gb = free_memory / 1e9
        
        # Calculate utilization
        utilization_percent = (allocated_memory / total_memory) * 100
        
        return {
            "operation": operation_name,
            "available": True,
            "timestamp": time.time(),
            "total_memory_gb": total_memory_gb,
            "allocated_memory_gb": allocated_memory_gb,
            "reserved_memory_gb": reserved_memory_gb,
            "free_memory_gb": free_memory_gb,
            "utilization_percent": utilization_percent
        }
    except Exception as e:
        logger.error(f"Error getting GPU memory info: {e}")
        return {
            "operation": operation_name,
            "available": True,
            "error": str(e),
            "timestamp": time.time()
        }

def extract_field_from_image(
    image_path: Union[str, Path],
    prompt: Union[str, Prompt, Dict[str, Any]],
    model_name: str,
    field_type: str = "generic",  # Added parameter
    model: Any = None,
    processor: Any = None,
    quantization: Optional[str] = None,
    precision: Optional[str] = None,
    max_new_tokens: int = 50,
    do_sample: bool = False,
    **kwargs
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Extract a field from an invoice image using a vision-language model.
    
    This function implements the core inference logic with support for different
    models, prompts, and optimization strategies defined in configuration.
    
    Args:
        image_path: Path to the invoice image
        prompt: Prompt object, dictionary, or prompt text to use for extraction
        model_name: Name of the model to use (for loading or formatting)
        field_type: Type of field to extract (work_order, cost, etc.)
        model: Pre-loaded model (will be loaded if None)
        processor: Pre-loaded processor (will be loaded if None)
        quantization: Quantization strategy to use (from config if None)
        precision: Numeric precision to use (from config if None)
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling in generation
        **kwargs: Additional parameters to pass to the model
        
    Returns:
        Tuple of (extracted_text, processing_time_seconds, metadata)
        
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If model inference fails
    """
    start_time = time.time()
    
    # Get model configuration
    model_config = get_model_config(model_name)
    
    # Set up metadata to track processing information
    metadata = {
        "model_name": model_name,
        "quantization": quantization,
        "precision": precision,
        "model_config": model_config.get("description", ""),
        "field_type": field_type,  # Added line
        "gpu_info": get_gpu_memory_info() if torch.cuda.is_available() else {"gpu_available": False},
        "timestamp": time.time()
    }
    
    # Validate inputs
    if model is None or processor is None:
        logger.info(f"Model or processor not provided, loading {model_name}")
        # Use the provided quantization or fallback to config
        model, processor = load_model_and_processor(
            model_name,
            quantization=quantization,
            **kwargs
        )
        metadata["model_loaded"] = True
    else:
        metadata["model_loaded"] = False
    
    # Get the proper precision type
    precision_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    
    # Determine precision - from parameter, model config, or default
    if precision is None:
        # Get default precision from model config
        precision = model_config.get("default_precision", "bfloat16")
    
    target_dtype = precision_map.get(precision, torch.bfloat16)
    metadata["actual_precision"] = precision
    
    # Process prompt based on its type
    if isinstance(prompt, Prompt):
        # Handle Prompt class object
        formatted_prompt = prompt.format_for_model(model_name)
        metadata["prompt_name"] = prompt.name
        metadata["prompt_category"] = prompt.category
        metadata["prompt_field"] = prompt.field_to_extract
    elif isinstance(prompt, dict):
        # Handle dictionary prompt info
        if "formatted_text" in prompt:
            formatted_prompt = prompt["formatted_text"]
        elif "text" in prompt:
            formatted_prompt = prompt["text"]
        else:
            # Try to find any string value in the dictionary
            for key, value in prompt.items():
                if isinstance(value, str) and len(value) > 0:
                    formatted_prompt = value
                    logger.info(f"Using text from '{key}' field for prompt")
                    break
            else:
                # Fallback to default if no string found
                formatted_prompt = f"<s>[INST]Extract the {model_config.get('field_type', 'information')} from this invoice image.\n[IMG][/INST]"
                logger.warning(f"No suitable text found in prompt dictionary, using default: {formatted_prompt}")
        
        # Store metadata from the prompt info dictionary
        metadata["prompt_name"] = prompt.get("name", "unknown")
        metadata["prompt_category"] = prompt.get("category", "unknown")
        metadata["prompt_field"] = prompt.get("field_to_extract", "unknown")
    else:
        # Assume it's already a formatted string
        formatted_prompt = str(prompt)
        metadata["prompt_name"] = "custom_string"
    
    logger.info(f"Using formatted prompt: '{formatted_prompt}'")
    
    try:
        # Open and convert the image
        image = Image.open(image_path).convert("RGB")
        metadata["image_size"] = image.size
        
        # Process using processor - FIXED: pass image directly, not as a list
        try:
            # First try with direct image (not in a list)
            inputs = processor(
                text=formatted_prompt,
                images=image,  # Direct image, not wrapped in a list
                return_tensors="pt"
            )
        except Exception as e1:
            logger.warning(f"Error with direct image input: {e1}. Trying with list wrapping...")
            # Fall back to list-wrapped image if direct approach fails
            inputs = processor(
                text=formatted_prompt,
                images=[image],  # Wrapped in a list as fallback
                return_tensors="pt"
            )
        
        # Convert inputs to appropriate dtypes based on model architecture
        for key in inputs:
            if key == "pixel_values":
                inputs[key] = inputs[key].to(dtype=target_dtype, device=model.device)
            else:
                inputs[key] = inputs[key].to(device=model.device)
        
        # Record GPU memory before inference
        if torch.cuda.is_available():
            pre_inference_memory = torch.cuda.memory_allocated() / 1e9
            metadata["pre_inference_memory_gb"] = pre_inference_memory
        
        # Generate response
        inference_start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
        inference_time = time.time() - inference_start
        metadata["inference_time"] = inference_time
        
        # Record GPU memory after inference
        if torch.cuda.is_available():
            post_inference_memory = torch.cuda.memory_allocated() / 1e9
            metadata["post_inference_memory_gb"] = post_inference_memory
            metadata["inference_memory_delta_gb"] = post_inference_memory - pre_inference_memory
        
        # Decode the output - handling different model output formats
        extracted_text = processor.batch_decode(
            outputs, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # The model output should directly contain the field value
        processed_text = extracted_text.strip()
        
        # Add field-specific post-processing
        processed_text = postprocess_extraction(
            extracted_text=processed_text,
            field_type=field_type
        )
        
        processing_time = time.time() - start_time
        metadata["total_processing_time"] = processing_time
        
        return processed_text, processing_time, metadata
    
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        processing_time = time.time() - start_time
        metadata["error"] = str(e)
        metadata["total_processing_time"] = processing_time
        raise RuntimeError(f"Error during extraction: {str(e)}")


# Replace the existing postprocess_extraction function around line 219
def postprocess_extraction(
    extracted_text: str, 
    field_type: str,
    extraction_rules: Optional[Dict[str, Any]] = None
) -> str:
    """
    Clean up extracted text based on field type and specific rules.
    
    Args:
        extracted_text: Raw text from the model
        field_type: Type of field extracted (work_order, cost, date, etc.)
        extraction_rules: Optional custom rules for extraction
        
    Returns:
        Cleaned up extracted text
    """
    # Remove any leading/trailing whitespace
    cleaned_text = extracted_text.strip()
    
    # Use provided extraction rules if available
    if extraction_rules and field_type in extraction_rules:
        rule = extraction_rules[field_type]
        
        # Apply regex pattern if provided
        if "pattern" in rule:
            pattern = rule["pattern"]
            matches = re.findall(pattern, cleaned_text)
            if matches:
                return matches[-1] if rule.get("use_last_match", True) else matches[0]
        
        # Apply custom processing function if provided
        if "processor" in rule and callable(rule["processor"]):
            return rule["processor"](cleaned_text)
    
    # Default field-specific processing if no custom rules provided
    if field_type == "work_order":
        # Extract just the numeric value with regex
        numbers = re.findall(r'(\d+)', cleaned_text)
        if numbers:
            return numbers[-1]  # Return the last number found
    elif field_type == "cost" or field_type == "amount":
        # Extract currency amount - enhanced pattern
        currency_pattern = r'\$?\s*([\d,]+\.?\d*)'
        matches = re.findall(currency_pattern, cleaned_text)
        if matches:
            # Remove commas and return as float string
            amount = matches[0].replace(',', '')
            # Ensure proper decimal format (add .00 if needed)
            if '.' not in amount:
                amount = f"{amount}.00"
            elif amount.endswith('.'):
                amount = f"{amount}00"
            elif len(amount.split('.')[-1]) == 1:
                amount = f"{amount}0"
            return amount
    elif field_type == "date":
        # Enhanced date extraction
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # MM/DD/YYYY, DD/MM/YYYY
            r'(\d{2,4}[/-]\d{1,2}[/-]\d{1,2})',  # YYYY/MM/DD
            r'(\w{3,9}\s+\d{1,2},?\s+\d{2,4})',  # Month DD, YYYY
            r'(\d{1,2}\s+\w{3,9},?\s+\d{2,4})'   # DD Month, YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, cleaned_text)
            if matches:
                return matches[0]
    elif field_type == "vendor" or field_type == "company":
        # Try to extract company name - often at the beginning
        lines = cleaned_text.splitlines()
        if lines:
            # First non-empty line is often the company name
            for line in lines:
                if line.strip():
                    return line.strip()
    
    # For other field types, return the cleaned text
    return cleaned_text


def calculate_character_error_rate(prediction: str, ground_truth: str) -> float:
    """
    Calculate character error rate using Levenshtein distance.
    Lower is better, 0 is perfect match.
    
    Args:
        prediction: Extracted text from model
        ground_truth: Ground truth value
        
    Returns:
        Float representing the character error rate (0-1)
    """
    # Clean and standardize both strings
    pred_clean = str(prediction).strip()
    truth_clean = str(ground_truth).strip()
    
    if len(truth_clean) == 0:
        return 1.0 if len(pred_clean) > 0 else 0.0
    
    # Calculate Levenshtein distance
    lev_distance = distance(pred_clean, truth_clean)
    
    # Normalize by the length of the ground truth
    return lev_distance / len(truth_clean)


def calculate_metrics(
    prediction: str, 
    ground_truth: str,
    metrics: List[str] = ["exact_match", "character_error_rate"]
) -> Dict[str, Any]:
    """
    Calculate all specified metrics for a prediction.
    
    Args:
        prediction: Extracted text from model
        ground_truth: Ground truth value
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary of metric names and values
    """
    results = {}
    
    for metric in metrics:
        if metric == "exact_match":
            results[metric] = calculate_exact_match(prediction, ground_truth)
        elif metric == "character_error_rate":
            results[metric] = calculate_character_error_rate(prediction, ground_truth)
        # Add more metrics as needed
    
    return results


def process_image_with_metrics(
    image_path: Union[str, Path],
    ground_truth: str,
    prompt: Union[str, Prompt, Dict[str, Any]],
    model_name: str,
    field_type: str,
    model: Any = None,
    processor: Any = None,
    quantization: Optional[str] = None,
    precision: Optional[str] = None,
    extraction_rules: Optional[Dict[str, Any]] = None,
    metrics: List[str] = ["exact_match", "character_error_rate"],
    **kwargs
) -> Dict[str, Any]:
    """
    Process an image and calculate metrics against ground truth.
    
    This function combines extraction, post-processing, and evaluation in one step,
    with comprehensive support for different models, fields, and configurations.
    
    Args:
        image_path: Path to the invoice image
        ground_truth: Ground truth value for the field
        prompt: Prompt object, dictionary, or text to use
        model_name: Name of the model to use
        field_type: Type of field being extracted
        model: Pre-loaded model (will be loaded if None)
        processor: Pre-loaded processor (will be loaded if None)
        quantization: Quantization strategy to use
        precision: Numeric precision to use
        extraction_rules: Optional rules for field extraction
        metrics: List of metrics to calculate
        **kwargs: Additional parameters for model or extraction
        
    Returns:
        Dictionary with extraction results and metrics
    """
    try:
        # Extract the field
        extracted_text, processing_time, metadata = extract_field_from_image(
            image_path=image_path,
            prompt=prompt,
            model_name=model_name,
            model=model,
            processor=processor,
            quantization=quantization,
            precision=precision,
            **kwargs
        )
        
        # Post-process the extraction
        processed_text = postprocess_extraction(
            extracted_text=extracted_text, 
            field_type=field_type,
            extraction_rules=extraction_rules
        )
        
        # Calculate metrics
        metric_results = calculate_metrics(processed_text, ground_truth, metrics)
        
        # Create the result object
        result = {
            "image_id": Path(image_path).stem,
            "ground_truth": ground_truth,
            "raw_extraction": extracted_text,
            "processed_extraction": processed_text,
            "field_type": field_type,
            "processing_time": processing_time,
            "model_name": model_name,
            "metadata": metadata,
            **metric_results  # Include all calculated metrics
        }
        
        # Log result
        match_status = "✓" if metric_results.get("exact_match", False) else "✗"
        logger.info(f"Image {result['image_id']}: {match_status} | "
                   f"Extracted: '{processed_text}' | GT: '{ground_truth}' | "
                   f"Time: {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        
        # Return a failure result
        return {
            "image_id": Path(image_path).stem,
            "ground_truth": ground_truth,
            "raw_extraction": "ERROR",
            "processed_extraction": "ERROR",
            "field_type": field_type,
            "processing_time": 0.0,
            "model_name": model_name,
            "exact_match": False,
            "character_error_rate": 1.0,
            "error": str(e)
        }

# Add after the existing process_image_with_metrics function around line 332

def extract_work_order(
    image_path: Union[str, Path],
    prompt: Union[str, Prompt, Dict[str, Any]],
    model_name: str,
    model: Any = None,
    processor: Any = None,
    ground_truth: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract work order number from an invoice image.
    
    Args:
        image_path: Path to the invoice image
        prompt: Prompt object, dictionary, or text to use
        model_name: Name of the model to use
        model: Pre-loaded model (will be loaded if None)
        processor: Pre-loaded processor (will be loaded if None)
        ground_truth: Optional ground truth for metric calculation
        **kwargs: Additional parameters for extraction
        
    Returns:
        Dictionary with extraction results and metrics
    """
    # Define work order specific extraction rules
    extraction_rules = {
        "work_order": {
            "pattern": r'(\d{5,7})',  # Work orders are typically 5-7 digit numbers
            "use_last_match": True
        }
    }
    
    # Process the image
    return process_image_with_metrics(
        image_path=image_path,
        ground_truth=ground_truth or "",
        prompt=prompt,
        model_name=model_name,
        field_type="work_order",
        model=model,
        processor=processor,
        extraction_rules=extraction_rules,
        **kwargs
    )

def extract_cost(
    image_path: Union[str, Path],
    prompt: Union[str, Prompt, Dict[str, Any]],
    model_name: str,
    model: Any = None,
    processor: Any = None,
    ground_truth: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Extract cost/amount from an invoice image.
    
    Args:
        image_path: Path to the invoice image
        prompt: Prompt object, dictionary, or text to use
        model_name: Name of the model to use
        model: Pre-loaded model (will be loaded if None)
        processor: Pre-loaded processor (will be loaded if None)
        ground_truth: Optional ground truth for metric calculation
        **kwargs: Additional parameters for extraction
        
    Returns:
        Dictionary with extraction results and metrics
    """
    # Define cost specific extraction rules
    extraction_rules = {
        "cost": {
            "pattern": r'\$?\s*([\d,]+\.?\d*)',  # Match currency amounts
            "use_last_match": True
        }
    }
    
    # Add cost-specific metrics
    if "metrics" not in kwargs:
        kwargs["metrics"] = ["exact_match", "character_error_rate", "numeric_difference"]
    
    # Process the image
    return process_image_with_metrics(
        image_path=image_path,
        ground_truth=ground_truth or "",
        prompt=prompt,
        model_name=model_name,
        field_type="cost",
        model=model,
        processor=processor,
        extraction_rules=extraction_rules,
        **kwargs
    )

def extract_field(
    image_path: Union[str, Path],
    field_type: str,
    prompt: Union[str, Prompt, Dict[str, Any]],
    model_name: str,
    model: Any = None,
    processor: Any = None,
    ground_truth: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generic field extraction dispatcher that routes to specific extractors.
    
    Args:
        image_path: Path to the invoice image
        field_type: Type of field to extract (work_order, cost, etc.)
        prompt: Prompt object, dictionary, or text to use
        model_name: Name of the model to use
        model: Pre-loaded model (will be loaded if None)
        processor: Pre-loaded processor (will be loaded if None)
        ground_truth: Optional ground truth for metric calculation
        **kwargs: Additional parameters for extraction
        
    Returns:
        Dictionary with extraction results and metrics
    """
    # Route to field-specific extractors
    if field_type == "work_order":
        return extract_work_order(
            image_path=image_path,
            prompt=prompt,
            model_name=model_name,
            model=model,
            processor=processor,
            ground_truth=ground_truth,
            **kwargs
        )
    elif field_type == "cost":
        return extract_cost(
            image_path=image_path,
            prompt=prompt,
            model_name=model_name,
            model=model,
            processor=processor,
            ground_truth=ground_truth,
            **kwargs
        )
    else:
        # For fields without specialized extractors
        return process_image_with_metrics(
            image_path=image_path,
            ground_truth=ground_truth or "",
            prompt=prompt,
            model_name=model_name,
            field_type=field_type,
            model=model,
            processor=processor,
            **kwargs
        )
"""
Image Processing and Inference Module

This module provides functions for processing images and running inference
with vision-language models, including metric calculation.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
from PIL import Image
import numpy as np

from src.models.model_service import ModelService
from src.prompts.registry import PromptRegistry

logger = logging.getLogger(__name__)

def process_image_with_metrics(
    image: Image.Image,
    model_service: ModelService,
    prompt_registry: PromptRegistry,
    fields: List[str],
    batch_size: int = 1,
    show_progress: bool = False
) -> Dict[str, Any]:
    """
    Process a single image and extract specified fields using the provided model.
    
    Args:
        image: PIL Image to process
        model_service: Service for model operations
        prompt_registry: Registry for prompt templates
        fields: List of fields to extract
        batch_size: Batch size for processing (default: 1)
        show_progress: Whether to show progress information
        
    Returns:
        Dictionary containing:
        - extracted_fields: Dict of field names to extracted values
        - confidence_scores: Dict of field names to confidence scores
        - processing_time: Time taken for processing
        - memory_usage: Memory used during processing
    """
    try:
        # Start timing
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        # Get memory usage before processing
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        else:
            initial_memory = 0
            
        # Process each field
        extracted_fields = {}
        confidence_scores = {}
        
        for field in fields:
            # Get prompt template for field
            prompt_template = prompt_registry.get_prompt(field)
            if not prompt_template:
                logger.warning(f"No prompt template found for field: {field}")
                continue
                
            # Generate prompt
            prompt = prompt_template.format(field=field)
            
            # Run inference
            try:
                result = model_service.generate(
                    image=image,
                    prompt=prompt,
                    max_tokens=50,  # Adjust based on expected output length
                    temperature=0.7,
                    top_p=0.9
                )
                
                # Extract value and confidence
                extracted_fields[field] = result.text.strip()
                confidence_scores[field] = result.confidence if hasattr(result, 'confidence') else 1.0
                
            except Exception as e:
                logger.error(f"Error processing field {field}: {str(e)}")
                extracted_fields[field] = None
                confidence_scores[field] = 0.0
                
        # Record end time
        end_time.record()
        torch.cuda.synchronize()
        processing_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        
        # Get final memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
            memory_usage = final_memory - initial_memory
        else:
            memory_usage = 0
            
        # Log results if progress is enabled
        if show_progress:
            logger.info(f"Processed image in {processing_time:.2f} seconds")
            logger.info(f"Memory usage: {memory_usage:.2f} GB")
            logger.info("Extracted fields:")
            for field, value in extracted_fields.items():
                logger.info(f"  {field}: {value} (confidence: {confidence_scores[field]:.2f})")
                
        return {
            "extracted_fields": extracted_fields,
            "confidence_scores": confidence_scores,
            "processing_time": processing_time,
            "memory_usage": memory_usage
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise 
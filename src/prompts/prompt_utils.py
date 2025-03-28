"""
Prompt Utilities

This module provides utility functions for working with prompts,
including formatting for different models, templating, validation,
and manipulation operations.
"""

import re
from typing import Dict, Any, List, Optional, Union, Callable
import logging

from .registry import Prompt

# Configure logging
logger = logging.getLogger(__name__)

# Model-specific formatters
def format_for_pixtral(prompt_text: str, include_image: bool = True) -> str:
    """
    Format a prompt for Pixtral models using the instruction format.
    
    Args:
        prompt_text: The raw prompt text
        include_image: Whether to include the [IMG] tag (default: True)
        
    Returns:
        Formatted prompt text for Pixtral models
    """
    # Clean any existing instruction tags to avoid duplication
    text = remove_instruction_tags(prompt_text)
    
    # Format with instruction tags
    if include_image:
        return f"<s>[INST]{text}\n[IMG][/INST]"
    else:
        return f"<s>[INST]{text}[/INST]"


def format_for_llava(prompt_text: str, include_image: bool = True) -> str:
    """
    Format a prompt for LLaVA models.
    
    Args:
        prompt_text: The raw prompt text
        include_image: Whether this prompt includes an image (default: True)
        
    Returns:
        Formatted prompt text for LLaVA models
    """
    # LLaVA doesn't need special image tokens as they're handled by
    # the model's processor directly
    return prompt_text


# Map of model name patterns to formatter functions
MODEL_FORMATTERS = {
    "pixtral": format_for_pixtral,
    "llava": format_for_llava,
    # Add more model formatters as needed
}


def get_formatter_for_model(model_name: str) -> Callable:
    """
    Get the appropriate formatter function for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Formatter function for the specified model
    """
    model_name_lower = model_name.lower()
    
    for pattern, formatter in MODEL_FORMATTERS.items():
        if pattern in model_name_lower:
            return formatter
    
    # Default to returning the text as-is if no formatter found
    logger.warning(f"No specific formatter found for model: {model_name}. Using default.")
    return lambda text, include_image=True: text


def format_prompt(prompt: Union[Prompt, str], model_name: str, include_image: bool = True) -> str:
    """
    Format a prompt for a specific model.
    
    Args:
        prompt: Prompt object or prompt text
        model_name: Name of the target model
        include_image: Whether to include image tag if applicable
        
    Returns:
        Formatted prompt text
    """
    # Extract text if a Prompt object is provided
    if isinstance(prompt, Prompt):
        prompt_text = prompt.text
    else:
        prompt_text = prompt
    
    # Get and apply the appropriate formatter
    formatter = get_formatter_for_model(model_name)
    return formatter(prompt_text, include_image=include_image)


# Text manipulation utilities
def remove_instruction_tags(text: str) -> str:
    """
    Remove any existing instruction tags from the text.
    
    Args:
        text: The text to process
        
    Returns:
        Text with instruction tags removed
    """
    # Remove <s>, [INST], and [/INST] tags
    text = re.sub(r'<s>\s*', '', text)
    text = re.sub(r'\[INST\]\s*', '', text)
    text = re.sub(r'\s*\[/INST\]', '', text)
    
    return text


def remove_image_tags(text: str) -> str:
    """
    Remove image tags from the text.
    
    Args:
        text: The text to process
        
    Returns:
        Text with image tags removed
    """
    # Remove [IMG] tags
    return re.sub(r'\[IMG\]', '', text)


def clean_prompt_text(text: str) -> str:
    """
    Clean a prompt text by removing any model-specific formatting.
    
    Args:
        text: The text to clean
        
    Returns:
        Clean prompt text without special formatting tags
    """
    text = remove_instruction_tags(text)
    text = remove_image_tags(text)
    return text.strip()


# Templating utilities
def apply_template(template: str, variables: Dict[str, Any]) -> str:
    """
    Apply variables to a template string using {variable} syntax.
    
    Args:
        template: The template string with {variable} placeholders
        variables: Dictionary of variables to substitute
        
    Returns:
        Formatted string with variables applied
    """
    try:
        return template.format(**variables)
    except KeyError as e:
        logger.error(f"Missing variable in template: {e}")
        # Return the template with missing variables marked
        return re.sub(r'\{(\w+)\}', lambda m: variables.get(m.group(1), f"[MISSING:{m.group(1)}]"), template)


def create_prompt_variants(base_prompt: str, variations: Dict[str, List[str]]) -> List[str]:
    """
    Create multiple variants of a prompt by substituting different options.
    
    Args:
        base_prompt: Base prompt text with {variable} placeholders
        variations: Dictionary mapping variable names to lists of possible values
        
    Returns:
        List of all possible prompt variations
    """
    import itertools
    
    # Get all keys and possible values
    keys = list(variations.keys())
    value_combinations = list(itertools.product(*(variations[key] for key in keys)))
    
    # Generate all variants
    variants = []
    for values in value_combinations:
        variables = dict(zip(keys, values))
        variant = apply_template(base_prompt, variables)
        variants.append(variant)
    
    return variants


# Prompt analysis utilities
def get_word_count(text: str) -> int:
    """
    Count the number of words in a text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Number of words
    """
    # Clean the text first
    clean_text = clean_prompt_text(text)
    return len(clean_text.split())


def get_prompt_complexity_score(text: str) -> float:
    """
    Calculate a simple complexity score for a prompt (0-1).
    
    Args:
        text: The prompt text
        
    Returns:
        Complexity score between 0 and 1
    """
    # Clean the text first
    clean_text = clean_prompt_text(text)
    
    # Factors that contribute to complexity
    word_count = len(clean_text.split())
    avg_word_length = sum(len(word) for word in clean_text.split()) / max(1, word_count)
    sentence_count = len(re.split(r'[.!?]+', clean_text))
    
    # Calculate complexity factors (normalized)
    length_factor = min(1.0, word_count / 50.0)  # Normalize to 50 words
    word_complexity = min(1.0, avg_word_length / 8.0)  # Normalize to 8 chars
    structure_complexity = min(1.0, sentence_count / 5.0)  # Normalize to 5 sentences
    
    # Combined score
    return (0.5 * length_factor + 0.3 * word_complexity + 0.2 * structure_complexity)


def analyze_prompt(prompt: Union[Prompt, str]) -> Dict[str, Any]:
    """
    Analyze a prompt and return various metrics.
    
    Args:
        prompt: Prompt object or text to analyze
        
    Returns:
        Dictionary of prompt metrics
    """
    text = prompt.text if isinstance(prompt, Prompt) else prompt
    clean_text = clean_prompt_text(text)
    
    return {
        "word_count": get_word_count(clean_text),
        "character_count": len(clean_text),
        "complexity_score": get_prompt_complexity_score(clean_text),
        "has_instructions": "[INST]" in text,
        "has_image_tag": "[IMG]" in text,
        "sentence_count": len(re.split(r'[.!?]+', clean_text)),
    }


# Batch processing utilities
def batch_format_prompts(prompts: List[Prompt], model_name: str) -> List[str]:
    """
    Format multiple prompts for a specific model.
    
    Args:
        prompts: List of Prompt objects
        model_name: Target model name
        
    Returns:
        List of formatted prompt texts
    """
    return [format_prompt(p, model_name) for p in prompts]


def create_field_extraction_prompt(field_name: str, field_description: Optional[str] = None) -> str:
    """
    Create a basic extraction prompt for a specific field.
    
    Args:
        field_name: Name of the field to extract
        field_description: Optional description of the field
        
    Returns:
        Prompt text for field extraction
    """
    field_display = field_name.replace("_", " ")
    
    if field_description:
        return f"Extract the {field_display} from this invoice. {field_description}"
    else:
        return f"Extract the {field_display} from this invoice."

"""
Prompt Management System

This package provides a comprehensive system for managing, retrieving,
formatting, and evaluating prompts for invoice field extraction.

The system includes:
- A central registry for organizing prompts
- Specific prompt implementations for invoice processing
- Utility functions for prompt manipulation and formatting
- Integration with configuration files

Usage:
    from src.prompts import get_prompt, format_prompt, get_prompts_by_field
    
    # Get a specific prompt
    prompt = get_prompt("work_order_basic")
    
    # Format it for a specific model
    formatted = format_prompt(prompt, "pixtral-12b")
    
    # Get all prompts for a specific field
    work_order_prompts = get_prompts_by_field("work_order")
"""

# Import and expose core registry functionality
from .registry import (
    Prompt,
    PromptRegistry,
    get_registry,
    register_prompt,
    get_prompt,
    get_prompts_by_category,
    get_prompts_by_field,
    list_all_prompts,
    list_prompt_categories,
    load_prompts_from_config,
    load_all_prompt_configs,
)

# Import and expose utility functions
from .prompt_utils import (
    format_prompt,
    clean_prompt_text,
    analyze_prompt,
    batch_format_prompts,
    create_field_extraction_prompt,
    create_prompt_variants,
)

# Import and expose invoice-specific functions
from .invoice_prompts import (
    get_all_invoice_prompts,
    get_prompt_by_field_and_category,
    get_recommended_prompt,
)

# Initialize the prompt system
def initialize_prompt_system():
    """
    Initialize the prompt system by loading configurations and setting up
    the registry with default prompts.
    
    This should be called once at application startup.
    """
    try:
        # First, ensure the invoice prompts module is imported
        # This will register the default prompts defined in code
        import src.prompts.invoice_prompts
        
        # Then, try to load prompts from configuration files
        try:
            load_all_prompt_configs()
        except FileNotFoundError:
            # It's okay if config files don't exist yet
            pass
        
        # Log the initialization result
        from src.prompts.registry import get_registry
        registry = get_registry()
        prompt_count = len(registry)
        
        return {
            "status": "success",
            "prompt_count": prompt_count,
            "categories": list_prompt_categories(),
        }
    except Exception as e:
        # Log any initialization errors
        return {
            "status": "error",
            "error": str(e),
        }

# Version information
__version__ = "0.1.0"
"""
Invoice Prompts

This module provides specialized prompts for invoice processing tasks,
including extraction of work order numbers, costs, dates, and other
invoice-specific fields.

It builds on the core prompt registry and provides domain-specific
prompt implementations for invoice analysis.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from .registry import Prompt, register_prompt, get_registry, get_prompt, get_prompts_by_field


# Work Order Number Extraction Prompts
def register_work_order_prompts() -> None:
    """
    Register a comprehensive set of prompts for work order number extraction.
    These prompts explore different ways of asking the model to extract
    the work order number from invoices.
    """
    
    # Core prompt variations
    register_prompt(Prompt(
        name="work_order_basic",
        text="Extract the work order number from this invoice image.",
        category="basic",
        field_to_extract="work_order",
        description="Simple direct instruction that was effective in initial tests."
    ))
    
    register_prompt(Prompt(
        name="work_order_brief",
        text="What is the work order number?",
        category="basic",
        field_to_extract="work_order",
        description="Very brief query format to test if shorter prompts are effective."
    ))
    
    register_prompt(Prompt(
        name="work_order_detailed",
        text="This is an invoice image. Please find and extract the work order number. "
             "It may be labeled as 'Work Order Number', 'Numero de Orden', 'Order #', or similar.",
        category="detailed",
        field_to_extract="work_order",
        description="Detailed prompt providing multiple possible field labels."
    ))
    
    # Format instruction prompts
    register_prompt(Prompt(
        name="work_order_numeric_only",
        text="Extract the work order number from this invoice. Return only the numeric digits with no additional text.",
        category="formatted",
        field_to_extract="work_order",
        description="Explicitly requests numeric-only output format."
    ))
    
    register_prompt(Prompt(
        name="work_order_structured",
        text="Extract the work order number from this invoice and return it in the format: 'Work Order: {number}'",
        category="formatted",
        field_to_extract="work_order",
        description="Requests a specific structured output format."
    ))
    
    # Position-based prompts
    register_prompt(Prompt(
        name="work_order_upper_section",
        text="Look at the upper part of this invoice image and extract the work order number.",
        category="positioned",
        field_to_extract="work_order",
        description="Directs attention to the upper section of the invoice."
    ))
    
    register_prompt(Prompt(
        name="work_order_header",
        text="Extract the work order number from the header section of this invoice.",
        category="positioned",
        field_to_extract="work_order",
        description="Focuses on the header section specifically."
    ))
    
    # Context-based prompts
    register_prompt(Prompt(
        name="work_order_with_context",
        text="This invoice is for maintenance services. Extract the work order number that identifies this job.",
        category="contextual",
        field_to_extract="work_order",
        description="Provides business context to help model understand the document type."
    ))
    
    # Chain-of-thought prompts
    register_prompt(Prompt(
        name="work_order_cot",
        text="To extract the work order number from this invoice: "
             "1. Scan the document for fields labeled as work order, order number, or similar. "
             "2. Identify the associated numeric value. "
             "3. Return only the work order number.",
        category="chain_of_thought",
        field_to_extract="work_order",
        description="Uses step-by-step reasoning to guide the extraction process."
    ))
    
    # Multi-attempt prompts
    register_prompt(Prompt(
        name="work_order_multi_attempt",
        text="Find the work order number on this invoice. If you don't see a field explicitly "
             "labeled as 'work order', look for similar fields like 'order number', 'job number', "
             "or 'reference number'.",
        category="fallback",
        field_to_extract="work_order",
        description="Provides fallback strategies for when primary labels aren't found."
    ))


# Cost/Total Extraction Prompts
def register_cost_prompts() -> None:
    """
    Register prompts for extracting cost/total information from invoices.
    """
    register_prompt(Prompt(
        name="cost_basic",
        text="Extract the total cost from this invoice image.",
        category="basic",
        field_to_extract="cost",
        description="Simple instruction for total cost extraction."
    ))
    
    register_prompt(Prompt(
        name="cost_detailed",
        text="Find the total amount due on this invoice. This is typically the final amount "
             "after taxes and may be labeled as 'Total', 'Amount Due', 'Grand Total', or similar.",
        category="detailed",
        field_to_extract="cost",
        description="Detailed prompt specifying possible labels for the total cost."
    ))
    
    register_prompt(Prompt(
        name="cost_formatted",
        text="What is the total cost on this invoice? Return just the numeric amount with currency symbol.",
        category="formatted",
        field_to_extract="cost",
        description="Requests formatted output with currency symbol."
    ))
    
    register_prompt(Prompt(
        name="cost_bottom_section",
        text="Look at the bottom section of this invoice and extract the total amount due.",
        category="positioned",
        field_to_extract="cost",
        description="Directs attention to the typical location of total amounts."
    ))


# Date Extraction Prompts
def register_date_prompts() -> None:
    """
    Register prompts for extracting date information from invoices.
    """
    register_prompt(Prompt(
        name="date_basic",
        text="Extract the invoice date from this document.",
        category="basic",
        field_to_extract="date",
        description="Simple instruction for invoice date extraction."
    ))
    
    register_prompt(Prompt(
        name="date_detailed",
        text="Find the date when this invoice was issued. It may be labeled as 'Invoice Date', "
             "'Date', 'Issue Date', or similar.",
        category="detailed",
        field_to_extract="date",
        description="Detailed prompt with possible field labels for invoice date."
    ))
    
    register_prompt(Prompt(
        name="date_formatted",
        text="What is the invoice date? Return it in MM/DD/YYYY format.",
        category="formatted",
        field_to_extract="date",
        description="Requests a specific date format in the response."
    ))


# Utility functions for prompt management
def get_all_invoice_prompts() -> List[Prompt]:
    """
    Get all invoice-related prompts from the registry.
    
    Returns:
        List of all invoice prompts
    """
    registry = get_registry()
    fields = ["work_order", "cost", "date"]
    
    # Collect prompts for all invoice-related fields
    prompts = []
    for field in fields:
        prompts.extend(registry.get_by_field(field))
    
    return prompts


def get_prompt_by_field_and_category(field: str, category: str) -> List[Prompt]:
    """
    Get prompts for a specific field and category.
    
    Args:
        field: The field to extract (e.g., "work_order")
        category: The prompt category (e.g., "basic", "detailed")
        
    Returns:
        List of matching prompts
    """
    registry = get_registry()
    return [p for p in registry.iter_prompts(category=category, field=field)]


def get_recommended_prompt(field: str) -> Optional[Prompt]:
    """
    Get the currently recommended prompt for a specific field based on
    previous performance.
    
    Args:
        field: The field to extract (e.g., "work_order")
        
    Returns:
        The recommended Prompt object or None if not found
    """
    # This will be expanded as performance data is gathered
    # For now, return the prompt that worked well in initial tests
    if field == "work_order":
        return get_prompt("work_order_basic")
    elif field == "cost":
        return get_prompt("cost_basic")
    elif field == "date":
        return get_prompt("date_basic")
    return None


def load_prompt_performance_data(results_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load performance data for prompts from previous experiments.
    This can be used to automatically select the best-performing prompts.
    
    Args:
        results_dir: Optional path to results directory
        
    Returns:
        Dictionary mapping prompt names to performance metrics
    """
    # This is a placeholder for future implementation
    # In the future, this will read from experiment results
    return {}


# Initialize the registry with invoice prompts
def initialize_invoice_prompts():
    """Initialize all invoice-related prompts in the registry."""
    register_work_order_prompts()
    register_cost_prompts()
    register_date_prompts()


# Initialize prompts on module import
initialize_invoice_prompts()
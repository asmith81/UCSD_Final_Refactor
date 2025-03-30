# Prompt Management System

## Introduction

The Prompt Management System provides a comprehensive framework for organizing, creating, and experimenting with different prompting strategies for invoice information extraction. This document explains how to use the system effectively in your invoice processing workflows.

## Key Concepts

- **Prompt**: A text instruction sent to vision-language models to extract specific information
- **Prompt Registry**: Central storage for all prompt definitions with metadata
- **Prompt Category**: Classification of prompts by approach (basic, detailed, positioned, etc.)
- **Field to Extract**: The specific information being targeted (work order, cost, date)
- **Prompt Configuration**: YAML-based storage of prompt definitions

## System Architecture

The prompt management system consists of several components:

1. **Registry** (`src/prompts/registry.py`): Central storage and retrieval system
2. **Invoice Prompts** (`src/prompts/invoice_prompts.py`): Domain-specific prompt implementations
3. **Utilities** (`src/prompts/prompt_utils.py`): Helper functions for prompt manipulation
4. **Configurations** (`configs/prompts/*.yaml`): YAML-based prompt storage
5. **Experiment Integration** (`src/config/experiment.py`): Connection to experiment framework

## Getting Started

### Importing the Prompt System

```python
# Import the entire system
from src.prompts import (
    get_prompt,                # Get a single prompt by name
    get_prompts_by_field,      # Get all prompts for a specific field
    get_prompts_by_category,   # Get all prompts in a specific category
    format_prompt,             # Format a prompt for a specific model
    list_all_prompts           # List all available prompts
)

# Initialize the prompt system (already happens on import, but can be called explicitly)
from src.prompts import initialize_prompt_system
initialize_prompt_system()
```

### Basic Operations

#### Accessing Prompts

```python
# Get a specific prompt by name
prompt = get_prompt("basic_work_order")
print(f"Prompt text: {prompt.text}")
print(f"Category: {prompt.category}")
print(f"Field: {prompt.field_to_extract}")

# List all available prompts
all_prompts = list_all_prompts()
print(f"Available prompts: {all_prompts}")

# Get all prompts for work order extraction
work_order_prompts = get_prompts_by_field("work_order")
print(f"Found {len(work_order_prompts)} work order prompts")

# Get all 'detailed' category prompts
detailed_prompts = get_prompts_by_category("detailed")
print(f"Found {len(detailed_prompts)} detailed prompts")
```

#### Formatting Prompts for Models

```python
# Format a prompt for the Pixtral model
formatted_text = format_prompt(prompt, "pixtral-12b")
print(f"Formatted text: {formatted_text}")

# Format a prompt text directly
text = "Extract the work order number from this invoice."
formatted_text = format_prompt(text, "pixtral-12b")
```

### Working with Experiment Configuration

The prompt system integrates with the experiment configuration system:

```python
from src.config.experiment import get_experiment_config

# Create experiment configuration
config = get_experiment_config(
    experiment_type="prompt_comparison",
    overrides={
        "experiment.field_to_extract": "work_order",
        "experiment.prompt_category": "basic"
    }
)

# Get prompts for this experiment
experiment_prompts = config.get_prompts_for_experiment()
print(f"Experiment will use {len(experiment_prompts)} prompts")

# Format a prompt for the model specified in the experiment
model_name = config.get_default_model()
formatted = config.format_prompt_for_model("basic_work_order")
```

## Adding New Prompts

### Method 1: Adding to YAML Configuration

Create or modify YAML files in `configs/prompts/`:

```yaml
# configs/prompts/my_prompts.yaml
config_info:
  name: my_custom_prompts
  description: Custom prompts for invoice processing
  version: 1.0
  last_updated: "2025-03-13"

prompts:
  - name: my_custom_work_order
    text: "This invoice contains a work order number. Please identify and extract only the work order number value."
    category: custom
    field_to_extract: work_order
    description: "Custom prompt focusing on extraction precision."
    version: "1.0"
    metadata:
      source: "custom_development"
      rationale: "Testing if explicit focus on the value improves accuracy"
```

Then load the configuration:

```python
from src.prompts import load_prompts_from_config
load_prompts_from_config("configs/prompts/my_prompts.yaml")
```

### Method 2: Programmatic Registration

Register prompts directly in code:

```python
from src.prompts import Prompt, register_prompt

# Create and register a new prompt
new_prompt = Prompt(
    name="experimental_work_order",
    text="Please analyze this invoice and return only the work order number.",
    category="experimental",
    field_to_extract="work_order",
    description="Minimalist approach requesting only the target value."
)

register_prompt(new_prompt)
```

### Method 3: Using the API from Notebooks

Create and register prompts directly from experiment notebooks:

```python
from src.prompts import Prompt, register_prompt, get_registry

# Create a prompt with detailed metadata
notebook_prompt = Prompt(
    name="notebook_generated_prompt",
    text="Find the work order number in this invoice. It is a unique identifier for this job.",
    category="notebook_test",
    field_to_extract="work_order",
    description="Created during notebook experimentation.",
    metadata={
        "created_by": "data_scientist_name",
        "experiment_id": "exp_20250313",
        "hypothesis": "Adding context about the purpose improves extraction"
    }
)

# Register it with the global registry
register_prompt(notebook_prompt)

# Save all current prompts to a new config file
registry = get_registry()
registry.save_to_config("configs/prompts/notebook_generated.yaml")
```

## Creating Systematic Prompt Variations

The prompt utilities module provides tools for creating systematic prompt variations:

```python
from src.prompts.prompt_utils import create_prompt_variants, apply_template

# Create a template with variables
template = "Extract the {field} from this invoice. {additional_instruction}"

# Define variations for each variable
variations = {
    "field": ["work order number", "work order ID", "job number"],
    "additional_instruction": [
        "It is a numeric value.",
        "It may appear in the header section.",
        "Look for it near the top of the document."
    ]
}

# Generate all combinations
prompt_variants = create_prompt_variants(template, variations)
print(f"Generated {len(prompt_variants)} prompt variations")

# Apply a single template
variables = {
    "field": "work order number",
    "additional_instruction": "It should be a 5-6 digit number."
}
single_prompt = apply_template(template, variables)
```

## Analyzing Prompts

The system provides tools to analyze prompts:

```python
from src.prompts.prompt_utils import analyze_prompt, get_prompt_complexity_score

# Analyze a prompt
prompt = get_prompt("detailed_work_order")
analysis = analyze_prompt(prompt)
print(f"Word count: {analysis['word_count']}")
print(f"Complexity score: {analysis['complexity_score']}")
print(f"Has instructions: {analysis['has_instructions']}")

# Get complexity scores for all prompts of a certain type
prompts = get_prompts_by_field("work_order")
scores = [(p.name, get_prompt_complexity_score(p.text)) for p in prompts]
scores.sort(key=lambda x: x[1], reverse=True)

# Show prompts from most to least complex
for name, score in scores:
    print(f"{name}: {score:.2f}")
```

## Using Prompts in the Extraction Pipeline

Here's a sample workflow showing how to use prompts in your extraction pipeline:

```python
from src.prompts import get_prompts_by_field, format_prompt
from src.models.loader import load_model_and_processor
import torch
from PIL import Image

# Load model and processor
model, processor = load_model_and_processor("pixtral-12b")

# Get all work order prompts
work_order_prompts = get_prompts_by_field("work_order")

# Process an image with each prompt
results = []
image_path = "data/images/invoice_1234.jpg"
image = Image.open(image_path).convert("RGB")

for prompt in work_order_prompts:
    # Format the prompt for this model
    formatted_prompt = format_prompt(prompt, "pixtral-12b")
    
    # Process using the model's processor
    inputs = processor(
        text=formatted_prompt,
        images=[image],
        return_tensors="pt"
    )
    
    # Move inputs to GPU with appropriate dtype
    for key in inputs:
        if key == "pixel_values":
            inputs[key] = inputs[key].to(dtype=torch.bfloat16, device="cuda")
        else:
            inputs[key] = inputs[key].to(device="cuda")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
    # Decode the output
    extracted_text = processor.batch_decode(
        outputs, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    # Store result with prompt information
    results.append({
        "prompt_name": prompt.name,
        "prompt_category": prompt.category,
        "extracted_text": extracted_text.strip(),
        "prompt_text": prompt.text
    })

# Analyze which prompts performed best
for result in results:
    print(f"Prompt: {result['prompt_name']}")
    print(f"Category: {result['prompt_category']}")
    print(f"Extracted: {result['extracted_text']}")
    print("-" * 40)
```

## Running Prompt Comparison Experiments

To run a full prompt comparison experiment:

```python
from src.config.experiment import get_experiment_config
from src.models.loader import load_model_and_processor
from src.execution.inference import extract_field_from_image
from src.results.collector import save_extraction_results

# Create experiment configuration
experiment = get_experiment_config(
    experiment_type="prompt_comparison",
    overrides={
        "experiment.field_to_extract": "work_order",
        "experiment.prompt_category": "all"
    }
)

# Get prompts for this experiment
prompts = experiment.get_prompts_for_experiment()

# Load model
model_name = experiment.get_default_model()
model, processor = load_model_and_processor(model_name)

# Get list of images
image_paths = [...]  # Your image paths here

# Run extraction for each prompt
results = {}
for prompt_info in prompts:
    prompt_name = prompt_info["name"]
    formatted_prompt = experiment.format_prompt_for_model(prompt_name)
    
    prompt_results = []
    for image_path in image_paths:
        # Extract field using current prompt
        extracted_text, metrics = extract_field_from_image(
            image_path=image_path,
            prompt=formatted_prompt,
            model=model,
            processor=processor
        )
        
        prompt_results.append({
            "image_path": image_path,
            "extracted_text": extracted_text,
            **metrics  # Include processing time, confidence, etc.
        })
    
    # Store results for this prompt
    results[prompt_name] = prompt_results

# Save the results
experiment_name = experiment.get("experiment.name", "prompt_comparison")
save_extraction_results(results, experiment_name)
```

## Best Practices

1. **Start with Existing Categories**: Begin with the provided prompt categories before creating your own.

2. **Systematic Variation**: When creating new prompts, vary one aspect at a time to isolate what works.

3. **Version Control**: Use the version field to track evolution of prompts over time.

4. **Document Rationale**: Always include a description and rationale to understand the purpose of each prompt.

5. **Configuration-Driven**: Prefer YAML configurations for prompts you want to reuse across experiments.

6. **Performance Tracking**: Log which prompts perform best for which fields to build institutional knowledge.

7. **Model-Specific Optimization**: Different models may respond better to different prompt styles; test systematically.

## Troubleshooting

### Common Issues

1. **Prompts Not Found**
   - Check that the prompt name is correct
   - Ensure configurations are loaded with `load_all_prompt_configs()`
   - Verify that the registry is initialized

2. **Formatting Errors**
   - Ensure the model name matches expected patterns (e.g., "pixtral-12b")
   - Check for missing required fields in prompt definitions

3. **Performance Issues**
   - Different prompt categories can significantly impact extraction accuracy
   - Too complex prompts may exceed token limits
   - Too simple prompts may lack necessary guidance

### Debugging Tips

```python
# Check what's registered in the system
from src.prompts import get_registry, list_prompt_categories, list_all_prompts

registry = get_registry()
print(f"Total prompts registered: {len(registry)}")
print(f"Categories: {list_prompt_categories()}")
print(f"Fields: {registry.list_fields()}")

# Verify prompt loading from configs
import os
from pathlib import Path
from src.prompts import load_prompts_from_config

project_root = os.environ.get('PROJECT_ROOT', '')
config_dir = Path(project_root) / 'configs' / 'prompts'
print(f"Looking for prompt configs in: {config_dir}")
print(f"Files found: {list(config_dir.glob('*.yaml'))}")
```

## Advanced Topics

### Creating Custom Prompt Categories

You can extend the system with your own prompt categories:

```python
from src.prompts import Prompt, register_prompt

# Create prompts with a new category
register_prompt(Prompt(
    name="persona_work_order",
    text="You are an invoice processing expert. Please extract the work order number from this document.",
    category="persona",  # New category
    field_to_extract="work_order",
    description="Uses a persona framing for the instruction."
))

# Register more prompts in this category...
```

### Model-Specific Formatting

To add support for a new model's formatting requirements:

```python
# In src/prompts/prompt_utils.py
def format_for_new_model(prompt_text: str, include_image: bool = True) -> str:
    """Format a prompt for NewModel."""
    # Clean any existing formatting
    text = remove_instruction_tags(prompt_text)
    
    # Apply new model's formatting
    if include_image:
        return f"<image>\n{text}"
    else:
        return text

# Add to the MODEL_FORMATTERS dictionary
MODEL_FORMATTERS["new-model"] = format_for_new_model
```

### Creating a Custom Registry

For specialized experiments, you can create a separate registry instance:

```python
from src.prompts.registry import PromptRegistry, Prompt

# Create a custom registry
custom_registry = PromptRegistry()

# Add prompts to it
custom_registry.register(Prompt(
    name="specialized_prompt",
    text="Special prompt for custom experiment.",
    category="specialized",
    field_to_extract="work_order"
))

# Use the custom registry
prompts = list(custom_registry.iter_prompts())
```

## Conclusion

The Prompt Management System provides a flexible, powerful framework for organizing and experimenting with different prompt strategies. By systematically testing different approaches, you can identify the most effective prompts for each extraction task, improving the overall performance of your invoice processing system.
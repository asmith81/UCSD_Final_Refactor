# %% [markdown]
# # Invoice Extraction: Quick Experiment Execution
# 
# This notebook allows you to:
# 1. Select an experiment template or create a custom experiment
# 2. Configure model, fields, prompts, and quantization settings
# 3. Run the extraction pipeline
# 4. View and save results

# %%
# Import necessary modules
import os
import sys
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
from IPython.display import display, HTML

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("invoice_extraction")

# Add the project root to the path if not already there
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our utility modules
try:
    from src.notebook.setup_utils import get_system_info, check_gpu_availability
    from src.notebook.experiment_utils import (
        list_available_models, 
        list_available_prompts,
        list_available_templates,
        create_basic_experiment,
        create_model_comparison_experiment,
        create_prompt_comparison_experiment,
        create_quantization_experiment,
        load_experiment_template,
        run_extraction_experiment,
        get_default_fields,
        visualize_experiment_results
    )
    from src.notebook.error_utils import display_error, NotebookFriendlyError
    utils_available = True
except ImportError as e:
    print(f"⚠️ Error importing utilities: {str(e)}")
    print("⚠️ Make sure you've run the environment setup notebook first.")
    utils_available = False

# Check GPU availability
try:
    import torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU detected: {device_name} with {gpu_memory:.2f} GB memory")
    else:
        print("ℹ️ No GPU detected - will use CPU for extraction (slower)")
        gpu_memory = 0
except ImportError:
    print("⚠️ PyTorch not installed - cannot check GPU availability")
    gpu_available = False
    gpu_memory = 0

# %% [markdown]
# ## 1. Select Experiment Type
# 
# Choose from available templates or create a custom experiment.

# %%
if utils_available:
    # List available templates
    templates = list_available_templates()
    
    print("📋 Available experiment templates:")
    if templates:
        for i, template in enumerate(templates, 1):
            print(f"{i}. {template['name']}: {template['description']}")
    else:
        print("No pre-configured templates found. You can create a custom experiment.")
    
    # List available experiment types
    print("\n📋 Available experiment types:")
    print("1. Basic extraction - Test one model, prompt set, and field set")
    print("2. Model comparison - Compare multiple models on the same extraction task")
    print("3. Prompt comparison - Compare different prompt variants for extraction")
    print("4. Quantization comparison - Compare model optimization techniques")
    print("5. Custom experiment - Define your own experiment parameters")
    
    # Select experiment type (in a real notebook, this would be interactive)
    # For now, we'll default to basic extraction
    experiment_type = "basic"
    print(f"\n✅ Selected experiment type: {experiment_type}")
    
    # Usee ipywidgets for selection
    import ipywidgets as widgets
    from IPython.display import display
    
    # Template selection widget
    template_dropdown = widgets.Dropdown(
        options=[t['name'] for t in templates] + ['Custom'],
        description='Template:',
        disabled=False,
    )
    
    # Experiment type selection widget
    type_dropdown = widgets.Dropdown(
        options=['basic', 'model_comparison', 'prompt_comparison', 'quantization_comparison', 'custom'],
        description='Type:',
        disabled=False,
    )
    
    display(template_dropdown)
    display(type_dropdown)

else:
    print("Utilities not available, cannot list templates or experiment types.")

# %% [markdown]
# ## 2. Configure Variables
# 
# Set up the experiment parameters.

# %% [markdown]
# ### Available Configuration Options:
# 
# 1. **Models**: The vision-language model for extraction (e.g., `llava-1.5-7b`, `phi-2`)
# 2. **Fields**: What to extract (e.g., `invoice_number`, `total_amount`)
# 3. **Prompts**: Instructions for the model (can use templates or custom prompts)
# 4. **Quantization**: Memory optimization (e.g., `8bit`, `4bit`, or `none`)
# 5. **Batch Size**: Number of invoices to process at once

# %%
if utils_available:
    # Display available models
    print("📦 Available models:")
    models = list_available_models()
    available_models = []
    
    if models:
        for model in models:
            available_models.append(model['name'])
            print(f"• {model['name']} ({model['size']:.2f} GB)")
    else:
        print("⚠️ No models found in registry or config folder.")
        print("Please add model configuration files to configs/models/ directory or install models to your models directory.")
        available_models = []  # Empty list instead of hardcoded fallbacks
    
    # Display available fields
    print("\n📝 Available fields:")
    fields = get_default_fields()
    for field, description in fields.items():
        print(f"• {field}: {description}")
    
    # Display available prompts
    print("\n💬 Available prompts:")
    prompts = list_available_prompts()
    if prompts:
        for field_type, prompt_list in prompts.items():
            print(f"• {field_type}: {', '.join(prompt_list)}")
    else:
        print("Default prompts will be used")
    
    # Display quantization options
    print("\n⚙️ Quantization options:")
    print("• float32: Full precision (highest accuracy, highest memory usage)")
    print("• none: Default precision for the model (usually float16)")
    print("• bfloat16: BFloat16 precision (~50% memory saving, good accuracy)")
    print("• 8bit: 8-bit quantization (~50% memory saving)")
    print("• 4bit: 4-bit quantization (~75% memory saving)")
    
    # Recommend configuration based on hardware
    print("\n🔍 Recommended configuration for your hardware:")
    if gpu_available:
        # Check for bfloat16 support (Ampere or newer GPUs)
        has_bfloat16_support = False
        try:
            has_bfloat16_support = torch.cuda.get_device_capability()[0] >= 8
        except:
            pass
        
        if gpu_memory < 8:
            recommended_model = "phi-2"
            recommended_quant = "4bit"
        elif gpu_memory < 16:
            recommended_model = "llava-1.5-7b"
            if has_bfloat16_support:
                recommended_quant = "bfloat16"
            else:
                recommended_quant = "8bit"
        elif gpu_memory < 32:
            recommended_model = "llava-1.5-13b"
            if has_bfloat16_support:
                recommended_quant = "bfloat16"
            else:
                recommended_quant = "8bit"
        else:
            recommended_model = "bakllava-1"
            # For large GPUs, can consider full precision
            if gpu_memory > 48:
                recommended_quant = "float32"
            elif has_bfloat16_support:
                recommended_quant = "bfloat16"
            else:
                recommended_quant = "8bit"
    else:
        recommended_model = "phi-2"
        recommended_quant = "4bit"
    
    # If no models are available, don't recommend one that doesn't exist
    if not available_models:
        recommended_model = None
        print("• Model: No models available")
    else:
        # If the recommended model isn't in available_models, use the first available one
        if recommended_model not in available_models:
            recommended_model = available_models[0]
        print(f"• Model: {recommended_model}")
        
    print(f"• Quantization: {recommended_quant}")
    print(f"• Batch size: 1")
    
    # Initialize experiment configuration with recommended values
    experiment_config = {
        "model_name": recommended_model if recommended_model else "No models available",
        "fields": ["invoice_number", "invoice_date", "total_amount", "vendor_name"],
        "batch_size": 1,
        "memory_optimization": True,
        "quantization": recommended_quant
    }
    
    # Display initial configuration
    print("\n⚙️ Current experiment configuration:")
    for key, value in experiment_config.items():
        if isinstance(value, list):
            print(f"• {key}: {', '.join(value)}")
        else:
            print(f"• {key}: {value}")
    
    # Interactive UI for experiment configuration
    try:
        # Import widgets
        import ipywidgets as widgets
        from IPython.display import display, HTML
        
        print("\n💡 Use the controls below to customize your experiment:")
        
        # Experiment type selection
        experiment_type_dropdown = widgets.Dropdown(
            options=["basic", "model_comparison", "prompt_comparison", "quantization_comparison", "custom"],
            value="basic",
            description='Experiment type:',
        )
        
        # Model selection
        if available_models:
            model_dropdown = widgets.Dropdown(
                options=available_models,
                value=recommended_model if recommended_model in available_models else available_models[0],
                description='Model:',
            )
        else:
            model_dropdown = widgets.Dropdown(
                options=["No models available"],
                value="No models available",
                description='Model:',
                disabled=True
            )
            # Add a help message widget explaining the issue
            no_models_help = widgets.HTML(
                """<div style="background-color:#FFF3CD; padding:10px; border-radius:5px; margin:10px 0; border:1px solid #FFECB5">
                <b>⚠️ No models available!</b><br>
                Please add model configuration files to the <code>configs/models/</code> directory or install models to your models directory.
                </div>"""
            )
            display(no_models_help)
        
        # Field selection
        field_select = widgets.SelectMultiple(
            options=list(fields.keys()),
            value=["invoice_number", "invoice_date", "total_amount"],
            description='Fields:',
            layout=widgets.Layout(height='120px')
        )
        
        # Quantization selection
        quant_dropdown = widgets.Dropdown(
            options=["float32", "none", "bfloat16", "8bit", "4bit"],
            value=recommended_quant,
            description='Quantization:',
        )
        
        # Batch size
        batch_slider = widgets.IntSlider(
            value=1,
            min=1,
            max=10,
            step=1,
            description='Batch size:',
        )
        
        # Memory optimization
        memory_checkbox = widgets.Checkbox(
            value=True,
            description='Memory optimization',
        )
        
        # Create container for custom parameters
        custom_params_container = widgets.VBox([])
        custom_params = []
        
        def add_custom_param(b):
            """Add a custom parameter input group"""
            param_name = widgets.Text(description="Name:", placeholder="parameter_name")
            param_value = widgets.Text(description="Value:", placeholder="value")
            remove_btn = widgets.Button(description="❌", layout=widgets.Layout(width='40px'))
            
            param_box = widgets.HBox([param_name, param_value, remove_btn])
            custom_params.append((param_name, param_value, param_box))
            custom_params_container.children = list(custom_params_container.children) + [param_box]
            
            def remove_param(b):
                custom_params.remove((param_name, param_value, param_box))
                custom_params_container.children = [box for _, _, box in custom_params]
                update_config()  # Update config after removing a parameter
                
            remove_btn.on_click(remove_param)
        
        # Add custom parameter button
        add_param_button = widgets.Button(
            description="Add Custom Parameter",
            button_style='info',
            icon='plus'
        )
        add_param_button.on_click(add_custom_param)
        
        # Custom parameters section
        custom_section = widgets.VBox([
            widgets.HTML("<h4 style='margin-top:15px;'>Custom Parameters</h4>"),
            widgets.HTML("""
            <p style='font-size:0.9em;color:#666;'>
            Add any custom parameters needed for your experiment. Examples:
            <ul style='font-size:0.9em;color:#666;'>
                <li><b>temperature</b>: Controls randomness in model output (e.g., 0.7)</li>
                <li><b>max_tokens</b>: Maximum tokens to generate (e.g., 50)</li>
                <li><b>special_tokens</b>: Any special tokens for prompt formatting</li>
                <li><b>custom_config_path</b>: Path to custom configuration file</li>
            </ul>
            </p>
            """),
            add_param_button,
            custom_params_container
        ])
        
        # Initially hide custom section
        custom_section.layout.display = 'none'
        
        # Show/hide custom section based on experiment type
        def update_custom_section(change):
            if change['new'] == 'custom':
                custom_section.layout.display = 'block'
                # If no models available, show warning specifically for custom experiment
                if model_dropdown.value == "No models available":
                    with config_output:
                        config_output.clear_output()
                        print("\n⚠️ Warning: Custom experiment requires available models")
                        print("Please add model configuration files to configs/models/ directory or install models to your models directory.")
                        print("\nCustom parameters won't take effect until models are available.")
            else:
                custom_section.layout.display = 'none'
        
        experiment_type_dropdown.observe(update_custom_section, names='value')
        
        # Output area for configuration display
        config_output = widgets.Output()
        
        # Global variable for experiment type
        experiment_type = "basic"
        
        def update_config(change=None):
            global experiment_type
            experiment_type = experiment_type_dropdown.value
            
            # Update basic configuration
            experiment_config["model_name"] = model_dropdown.value
            experiment_config["fields"] = list(field_select.value)
            experiment_config["quantization"] = quant_dropdown.value
            experiment_config["batch_size"] = batch_slider.value
            experiment_config["memory_optimization"] = memory_checkbox.value
            
            # Handle custom parameters for custom experiment type
            if experiment_type == "custom":
                experiment_config["custom_parameters"] = {}
                for name_widget, value_widget, _ in custom_params:
                    param_name = name_widget.value.strip()
                    param_value = value_widget.value.strip()
                    
                    # Skip empty parameters
                    if not param_name:
                        continue
                        
                    # Try to convert to appropriate type
                    try:
                        # Try as number
                        if param_value.isdigit():
                            param_value = int(param_value)
                        elif param_value.replace('.', '', 1).isdigit():
                            param_value = float(param_value)
                        # Try as boolean
                        elif param_value.lower() in ('true', 'false'):
                            param_value = param_value.lower() == 'true'
                    except:
                        # Keep as string if parsing fails
                        pass
                        
                    experiment_config["custom_parameters"][param_name] = param_value
            elif "custom_parameters" in experiment_config:
                # Remove custom parameters if not a custom experiment
                del experiment_config["custom_parameters"]
            
            # Display updated configuration
            with config_output:
                config_output.clear_output()
                print("\n⚙️ Updated experiment configuration:")
                print(f"• Type: {experiment_type}")
                
                # Show model availability warning if needed
                if experiment_config["model_name"] == "No models available":
                    print("⚠️ Warning: No models available - experiment cannot be run")
                    print("• model_name: No models available ❌")
                else:
                    print(f"• model_name: {experiment_config['model_name']} ✅")
                
                # Display the rest of the configuration parameters
                for key, value in experiment_config.items():
                    if key == "model_name":
                        continue  # Already displayed above with status
                    elif key == "custom_parameters":
                        print(f"• {key}:")
                        if value:
                            for param_name, param_value in value.items():
                                print(f"    - {param_name}: {param_value}")
                        else:
                            print("    (No custom parameters defined)")
                    elif isinstance(value, list):
                        print(f"• {key}: {', '.join(map(str, value))}")
                    else:
                        print(f"• {key}: {value}")
                
                # Add tips based on experiment type
                if experiment_type == "custom" and experiment_config["model_name"] != "No models available":
                    print("\n💡 Tip: You can add more parameters using the Custom Parameters section")
                elif experiment_type == "model_comparison" and experiment_config["model_name"] == "No models available":
                    print("\n⚠️ Model comparison requires at least one available model")
        
        # Register observers
        experiment_type_dropdown.observe(update_config, names='value')
        model_dropdown.observe(update_config, names='value')
        field_select.observe(update_config, names='value')
        quant_dropdown.observe(update_config, names='value')
        batch_slider.observe(update_config, names='value')
        memory_checkbox.observe(update_config, names='value')
        
        # Create UI sections
        basic_section = widgets.VBox([
            widgets.HTML("<h4>Basic Parameters</h4>"),
            experiment_type_dropdown,
            model_dropdown,
            field_select,
            quant_dropdown,
            batch_slider,
            memory_checkbox
        ])
        
        # Display UI components
        display(widgets.VBox([
            widgets.HTML("<h3>Experiment Configuration</h3>"),
            basic_section,
            custom_section,
            widgets.HTML("<h4 style='margin-top:15px;'>Configuration Summary</h4>"),
            config_output
        ]))
        
        # Initialize configuration display
        update_config()
        
    except ImportError:
        print("\n⚠️ Interactive widgets not available. Using static configuration instead.")
        print("To enable interactive configuration, install ipywidgets: pip install ipywidgets")
        
else:
    print("Utilities not available, cannot display configuration options.")

# %% [markdown]
# ## 3. Create and Run the Experiment

# %%
if utils_available:
    try:
        # Verify that a model is available before proceeding
        if experiment_config["model_name"] == "No models available":
            print("❌ Cannot create experiment: No models are available.")
            print("Please add model configuration files to configs/models/ directory or install models to your models directory.")
            raise ValueError("No models available for experiment")
        
        # Create the experiment based on type
        print(f"🔬 Creating {experiment_type} extraction experiment...")
        
        # Prepare quantization configuration
        quant_config = None
        if experiment_config["quantization"] == "float32":
            quant_config = {"torch_dtype": "float32"}
        elif experiment_config["quantization"] == "bfloat16":
            quant_config = {"torch_dtype": "bfloat16"}
        elif experiment_config["quantization"] == "8bit":
            quant_config = {"bits": 8, "use_double_quant": True}
        elif experiment_config["quantization"] == "4bit":
            quant_config = {"bits": 4, "use_double_quant": True}
        
        # Create experiment object based on type
        if experiment_type == "basic":
            experiment = create_basic_experiment(
                model_name=experiment_config["model_name"],
                fields=experiment_config["fields"],
                batch_size=experiment_config["batch_size"],
                memory_optimization=experiment_config["memory_optimization"],
                quantization=quant_config
            )
            print(f"✅ Created basic experiment with model {experiment.model_name}")
            
        elif experiment_type == "model_comparison":
            # In the enhanced version, we get models from the UI or use defaults
            models_to_compare = experiment_config.get("models_to_compare", [
                "phi-2", 
                "llava-1.5-7b"
            ])
            
            experiment = create_model_comparison_experiment(
                model_names=models_to_compare,
                fields=experiment_config["fields"],
                batch_size=experiment_config["batch_size"],
                memory_optimization=experiment_config["memory_optimization"]
            )
            print(f"✅ Created model comparison experiment with models: {', '.join(experiment.models_to_compare)}")
            
        elif experiment_type == "prompt_comparison":
            # Get prompt variants from config or create defaults
            prompt_variants = experiment_config.get("prompt_variants", {
                "simple": {field: f"Extract the {field} from this invoice." for field in experiment_config["fields"]},
                "detailed": {field: f"Extract the {field} from this invoice. Look for text labeled '{field.replace('_', ' ')}' or similar." for field in experiment_config["fields"]}
            })
            
            experiment = create_prompt_comparison_experiment(
                model_name=experiment_config["model_name"],
                fields=experiment_config["fields"],
                prompt_variants=prompt_variants,
                batch_size=experiment_config["batch_size"]
            )
            print(f"✅ Created prompt comparison experiment with variants: {', '.join(prompt_variants.keys())}")
            
        elif experiment_type == "quantization_comparison":
            # Define quantization strategies
            strategies = experiment_config.get("quantization_strategies", [])
            
            # If no strategies defined in config, create based on hardware
            if not strategies:
                # Only include full precision if enough memory
                if gpu_memory > 40:
                    strategies.append({"torch_dtype": "float32"})
                    
                if gpu_memory > 16:  # Only include 'none' if enough memory
                    strategies.append(None)
                
                # Check for bfloat16 support
                has_bfloat16_support = False
                try:
                    has_bfloat16_support = torch.cuda.get_device_capability()[0] >= 8
                except:
                    pass
                    
                if has_bfloat16_support:
                    strategies.append({"torch_dtype": "bfloat16"})
                    
                strategies.append({"bits": 8, "use_double_quant": True})
                strategies.append({"bits": 4, "use_double_quant": True})
            
            experiment = create_quantization_experiment(
                model_name=experiment_config["model_name"],
                fields=experiment_config["fields"],
                quantization_strategies=strategies,
                batch_size=experiment_config["batch_size"]
            )
            print(f"✅ Created quantization comparison experiment with {len(strategies)} strategies")
            
        elif experiment_type == "custom":
            # Import the custom experiment creation function
            try:
                from src.notebook.experiment_utils import create_custom_experiment
            except ImportError:
                # If not yet implemented, we'll create a quick implementation here
                from src.config.experiment_config import create_experiment_config, ExperimentType
                
                def create_custom_experiment(model_name, fields, **custom_params):
                    """Create a custom experiment with flexible parameters."""
                    # Check if the model is available
                    if model_name == "No models available":
                        raise ValueError("Cannot create custom experiment: No models available")
                        
                    params = {
                        "model_name": model_name,
                        "fields_to_extract": fields,
                        **custom_params
                    }
                    
                    # Get experiment type enum if available or use string
                    exp_type = ExperimentType.CUSTOM if hasattr(ExperimentType, "CUSTOM") else "custom"
                    
                    return create_experiment_config(
                        experiment_type=exp_type,
                        **params
                    )
            
            # Extract custom parameters from config
            custom_params = experiment_config.get("custom_parameters", {})
            
            # Validate custom parameters
            if not custom_params:
                print("⚠️ No custom parameters specified. Using default experiment settings.")
                print("   Use the 'Add Custom Parameter' button to add experiment-specific parameters.")
            
            # Create the custom experiment
            experiment = create_custom_experiment(
                model_name=experiment_config["model_name"],
                fields=experiment_config["fields"],
                batch_size=experiment_config["batch_size"],
                memory_optimization=experiment_config["memory_optimization"],
                quantization=quant_config,
                **custom_params  # Add all custom parameters
            )
            
            print(f"✅ Created custom experiment with model {experiment.model_name}")
            if custom_params:
                print(f"   Custom parameters:")
                for key, value in custom_params.items():
                    print(f"   • {key}: {value}")
        
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        print(f"\n📋 Experiment details:")
        print(f"• Name: {experiment.name}")
        print(f"• Type: {experiment.experiment_type}")
        print(f"• Fields: {', '.join(experiment.fields if hasattr(experiment, 'fields') else experiment.fields_to_extract)}")
        
        # Create a run button instead of hardcoded True
        try:
            import ipywidgets as widgets
            from IPython.display import display
            
            run_button = widgets.Button(
                description="Run Experiment",
                button_style='success',
                icon='rocket'
            )
            
            output_area = widgets.Output()
            display(widgets.VBox([run_button, output_area]))
            
            def on_run_clicked(b):
                with output_area:
                    output_area.clear_output()
                    run_experiment(True)
            
            run_button.on_click(on_run_clicked)
            
            # Define function to run experiment
            def run_experiment(run_now=False):
                if run_now:
                    print(f"\n🚀 Running experiment: {experiment.name}")
                    print(f"This might take a few minutes, especially if models need to be downloaded.")
                    
                    # Start timer
                    start_time = time.time()
                    
                    # Run the experiment
                    data_path = experiment_config.get("data_path", os.environ.get("DATA_DIR", "data"))
                    result = run_extraction_experiment(
                        config=experiment,
                        data_path=data_path,
                        show_progress=True
                    )
                    
                    # Calculate runtime
                    runtime = time.time() - start_time
                    
                    print(f"✅ Experiment completed in {runtime:.2f} seconds!")
                    print(f"📊 Processed {len(result.extractions) if hasattr(result, 'extractions') else 'multiple'} invoices")
                    
                    # Save the experiment ID for later reference
                    experiment_id = result.experiment_id if hasattr(result, 'experiment_id') else f"{experiment.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    print(f"💾 Results saved with ID: {experiment_id}")
                    
                    # Display experiment variables for future reference
                    print("\n📋 Experiment variables (for replication):")
                    print(f"experiment_type = '{experiment_type}'")
                    print(f"experiment_config = {json.dumps(experiment_config, indent=2)}")
                    
                    return result
                else:
                    print("\n⏸️ Experiment ready but not running. Click the 'Run Experiment' button to execute.")
            
            # Initial message
            with output_area:
                print("\n⏸️ Experiment ready. Click the 'Run Experiment' button to execute.")
                
        except ImportError:
            # Fallback to non-interactive mode
            run_now = True  # In a real notebook, this would be a user input via a button
            
            if run_now:
                print(f"\n🚀 Running experiment: {experiment.name}")
                print(f"This might take a few minutes, especially if models need to be downloaded.")
                
                # Start timer
                start_time = time.time()
                
                # Run the experiment
                result = run_extraction_experiment(
                    config=experiment,
                    data_path=os.environ.get("DATA_DIR", "data"),
                    show_progress=True
                )
                
                # Calculate runtime
                runtime = time.time() - start_time
                
                print(f"✅ Experiment completed in {runtime:.2f} seconds!")
                print(f"📊 Processed {len(result.extractions) if hasattr(result, 'extractions') else 'multiple'} invoices")
                
                # Save the experiment ID for later reference
                experiment_id = result.experiment_id if hasattr(result, 'experiment_id') else f"{experiment.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"💾 Results saved with ID: {experiment_id}")
                
                # Display experiment variables for future reference
                print("\n📋 Experiment variables (for replication):")
                print(f"experiment_type = '{experiment_type}'")
                print(f"experiment_config = {json.dumps(experiment_config, indent=2)}")
            else:
                print("\n⏸️ Experiment ready but not running. Execute the cell again with run_now=True to execute.")
                
    except Exception as e:
        print(f"❌ Error creating/running experiment: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print("Utilities not available, cannot create or run experiments.")

# %% [markdown]
# ## 4. View Results
# 
# Explore the experiment results.

# %%
# This cell will display results when an experiment is run
if 'result' in locals():
    print("📊 Experiment Results")
    print("==================")
    
    # For basic experiment
    if hasattr(result, 'extractions'):
        print(f"Total extractions: {len(result.extractions)}")
        
        # Create a DataFrame for better viewing
        results_data = []
        for extraction in result.extractions[:10]:  # Show first 10
            row = {
                "document_id": extraction.document_id,
            }
            # Add extracted fields
            for field, value in extraction.extracted_fields.items():
                row[field] = value
            
            # Add accuracy if available
            if hasattr(extraction, 'accuracy'):
                row['accuracy'] = extraction.accuracy
            
            results_data.append(row)
        
        if results_data:
            df = pd.DataFrame(results_data)
            print("\n📋 Sample extraction results:")
            display(df)
        
        # Show overall metrics if available
        if hasattr(result, 'metrics'):
            print("\n📈 Overall metrics:")
            for metric, value in result.metrics.items():
                print(f"• {metric}: {value}")
    
    # For model comparison
    elif hasattr(result, 'model_results'):
        print("Model comparison results:")
        model_metrics = []
        for model_name, model_result in result.model_results.items():
            model_metrics.append({
                "model": model_name,
                "accuracy": getattr(model_result, 'accuracy', 'N/A'),
                "processing_time": getattr(model_result, 'processing_time', 'N/A'),
                "extractions": len(model_result.extractions) if hasattr(model_result, 'extractions') else 'N/A'
            })
        
        if model_metrics:
            df = pd.DataFrame(model_metrics)
            print("\n📊 Model comparison:")
            display(df)
    
    # For prompt comparison
    elif hasattr(result, 'prompt_results'):
        print("Prompt comparison results:")
        prompt_metrics = []
        for prompt_name, prompt_result in result.prompt_results.items():
            prompt_metrics.append({
                "prompt": prompt_name,
                "accuracy": getattr(prompt_result, 'accuracy', 'N/A'),
                "extractions": len(prompt_result.extractions) if hasattr(prompt_result, 'extractions') else 'N/A'
            })
        
        if prompt_metrics:
            df = pd.DataFrame(prompt_metrics)
            print("\n📊 Prompt comparison:")
            display(df)
    
    # For quantization comparison
    elif hasattr(result, 'quantization_results'):
        print("Quantization comparison results:")
        quant_metrics = []
        for quant_name, quant_result in result.quantization_results.items():
            quant_metrics.append({
                "quantization": quant_name,
                "accuracy": getattr(quant_result, 'accuracy', 'N/A'),
                "memory_usage": getattr(quant_result, 'memory_usage', 'N/A'),
                "processing_time": getattr(quant_result, 'processing_time', 'N/A')
            })
        
        if quant_metrics:
            df = pd.DataFrame(quant_metrics)
            print("\n📊 Quantization comparison:")
            display(df)
    
    # Create visualizations if available
    try:
        print("\n📈 Generating visualizations...")
        viz = visualize_experiment_results(result, output_format="notebook")
        display(viz)
    except Exception as e:
        print(f"⚠️ Could not generate visualizations: {str(e)}")
    
    # Provide code for loading these results later
    print("\n💾 Load these results later with:")
    print(f"""
    from src.notebook.experiment_utils import load_experiment_results
    result, metadata = load_experiment_results("{experiment_id}")
    """)
else:
    print("No experiment results available. Run an experiment first.")

# %% [markdown]
# ## 5. Next Steps
# 
# - Run different experiment types to compare models, prompts, or quantization strategies
# - Load previous experiment results for detailed analysis
# - Use the Results Analysis notebook to compare multiple experiments
# - Create custom prompt templates for better extraction accuracy

# %%
print("✅ Experiment complete!")
print("To run another experiment, modify the configuration above and re-run the cells.") 



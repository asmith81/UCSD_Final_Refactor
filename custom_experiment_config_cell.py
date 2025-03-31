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
        print("Models will be downloaded on first use. Common models:")
        available_models = ["phi-2", "llava-1.5-7b", "llava-1.5-13b", "bakllava-1"]
        for model in available_models:
            print(f"• {model}")
    
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
    
    print(f"• Model: {recommended_model}")
    print(f"• Quantization: {recommended_quant}")
    print(f"• Batch size: 1")
    
    # Initialize experiment configuration with recommended values
    experiment_config = {
        "model_name": recommended_model,
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
        model_dropdown = widgets.Dropdown(
            options=available_models,
            value=recommended_model,
            description='Model:',
        )
        
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
            widgets.HTML("<p style='font-size:0.9em;color:#666;'>Add any custom parameters needed for your experiment:</p>"),
            add_param_button,
            custom_params_container
        ])
        
        # Initially hide custom section
        custom_section.layout.display = 'none'
        
        # Show/hide custom section based on experiment type
        def update_custom_section(change):
            if change['new'] == 'custom':
                custom_section.layout.display = 'block'
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
                for key, value in experiment_config.items():
                    if key == "custom_parameters":
                        print(f"• {key}:")
                        for param_name, param_value in value.items():
                            print(f"    - {param_name}: {param_value}")
                    elif isinstance(value, list):
                        print(f"• {key}: {', '.join(map(str, value))}")
                    else:
                        print(f"• {key}: {value}")
        
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
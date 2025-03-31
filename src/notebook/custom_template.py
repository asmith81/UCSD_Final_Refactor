def create_custom_experiment_ui():
    """
    Create interactive UI for configuring a custom experiment.
    
    Returns:
        Widget interface for experiment configuration
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
        
        # Create basic parameter widgets
        model_dropdown = widgets.Dropdown(
            options=list_available_models(),
            description='Model:',
        )
        
        field_select = widgets.SelectMultiple(
            options=get_default_fields().keys(),
            description='Fields:',
        )
        
        # Create custom parameter section with add/remove functionality
        custom_params = []
        
        def add_param(b):
            """Add a new custom parameter to the UI."""
            param_name = widgets.Text(description="Name:")
            param_value = widgets.Text(description="Value:")
            param_box = widgets.HBox([param_name, param_value])
            custom_params.append((param_name, param_value))
            display(param_box)
        
        add_param_button = widgets.Button(description="Add Custom Parameter")
        add_param_button.on_click(add_param)
        
        # Create experiment button
        create_button = widgets.Button(description="Create Custom Experiment")
        output = widgets.Output()
        
        def on_create_clicked(b):
            """Create the experiment with custom parameters."""
            with output:
                # Collect all custom parameters
                custom_dict = {}
                for name_widget, value_widget in custom_params:
                    name = name_widget.value
                    value = value_widget.value
                    if name:
                        # Try to parse as different types
                        try:
                            # Try as number
                            if value.isdigit():
                                value = int(value)
                            elif "." in value:
                                value = float(value)
                            # Try as boolean
                            elif value.lower() in ("true", "false"):
                                value = value.lower() == "true"
                        except:
                            # Keep as string if parsing fails
                            pass
                        
                        custom_dict[name] = value
                
                # Create the experiment
                experiment = create_custom_experiment(
                    model_name=model_dropdown.value,
                    fields=list(field_select.value),
                    **custom_dict
                )
                
                print(f"Created custom experiment: {experiment.name}")
                print(f"Model: {experiment.model_name}")
                print(f"Fields: {', '.join(experiment.fields_to_extract)}")
                print(f"Custom parameters: {len(custom_dict)}")
                for name, value in custom_dict.items():
                    print(f"  - {name}: {value}")
        
        create_button.on_click(on_create_clicked)
        
        # Assemble the UI
        ui = widgets.VBox([
            widgets.HTML("<h3>Custom Experiment Configuration</h3>"),
            widgets.HTML("<p>Configure required parameters:</p>"),
            model_dropdown,
            field_select,
            widgets.HTML("<p>Add custom parameters (optional):</p>"),
            add_param_button,
            widgets.HBox([create_button, output])
        ])
        
        return ui
    
    except ImportError:
        print("ipywidgets not available. Please install with: pip install ipywidgets")
        return None 
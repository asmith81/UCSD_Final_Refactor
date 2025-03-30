# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Invoice Extraction: Results Analysis
#
# This notebook provides tools for analyzing and comparing extraction experiment results.

# %%
# Import necessary modules
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from IPython.display import display, HTML
from typing import List, Dict, Optional, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("results_analysis")

# Add the project root to the path if not already there
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import utility modules
try:
    from src.notebook.experiment_utils import load_experiment_results
    from src.results.collector import ResultsCollector, ComparisonResult, ExperimentLoader, ExperimentComparator
    from src.notebook.visualization_utils import NotebookVisualizationManager
    utils_available = True
except ImportError as e:
    print(f"⚠️ Error importing utilities: {str(e)}")
    utils_available = False

# Initialize visualization manager
viz_manager = NotebookVisualizationManager() if utils_available else None

# %% [markdown]
# ## 1. Load Experiment Results
#
# First, let's load the results from your experiments.

# %%
# Options for finding experiment results
if utils_available:
    # Create an experiment loader
    experiment_loader = ExperimentLoader()
    
    # Discover available experiments
    discovered_experiments = experiment_loader.discover_experiments(force_refresh=True)
    
    if discovered_experiments:
        # Convert to DataFrame for easier viewing
        exp_df = experiment_loader.get_experiment_dataframe()
        
        # Display experiment summary
        print(f"📊 Found {len(discovered_experiments)} experiments")
        print("\nExperiment Summary:")
        display(exp_df[['experiment_id', 'model', 'date', 'fields', 'type']].head(10))
        
        if len(discovered_experiments) > 10:
            print(f"...and {len(discovered_experiments) - 10} more experiments")
    else:
        print("No experiments found. Run an experiment first using the experiment configuration notebook.")
    
    # Provide code for manually specifying experiment ID
    print("\n💡 To load a specific experiment by ID:")
    print("""
    # Example:
    experiment_id = "your_experiment_id_here"
    result, metadata = load_experiment_results(experiment_id)
    """)
else:
    print("Utilities not available, cannot load experiments.")

# %% [markdown]
# ## 2. Select and Analyze an Experiment
#
# Choose an experiment to analyze from the list above.

# %%
# Replace with your experiment ID
if utils_available:
    # For demonstration - in a real notebook, users would select from the discovered experiments
    # Either use the most recent experiment or allow the user to specify one
    if discovered_experiments:
        # Sort by date and take the most recent
        most_recent_exp = sorted(discovered_experiments, key=lambda x: x.get('date', ''), reverse=True)[0]
        experiment_id = most_recent_exp.get('experiment_id')
        print(f"Using most recent experiment: {experiment_id}")
    else:
        # Example ID - users would replace this
        experiment_id = "example_experiment_id"
        print(f"No experiments found. Using example ID: {experiment_id}")
        print("In a real notebook, replace this with your actual experiment ID")
    
    try:
        # Load the experiment results
        result, metadata = load_experiment_results(experiment_id)
        print(f"✅ Loaded experiment: {experiment_id}")
        
        # Display experiment overview
        print("\n📋 Experiment Overview:")
        print(f"• Model: {metadata.get('model', 'N/A')}")
        print(f"• Date: {metadata.get('date', 'N/A')}")
        print(f"• Fields: {', '.join(metadata.get('fields', []))}")
        print(f"• Type: {metadata.get('type', 'N/A')}")
        
        # You'd use interactive widgets in a real notebook
        """
        import ipywidgets as widgets
        
        # Create experiment selector
        experiment_dropdown = widgets.Dropdown(
            options=[(f"{exp.get('experiment_id')} - {exp.get('model')} - {exp.get('date')}", 
                      exp.get('experiment_id')) for exp in discovered_experiments],
            description='Experiment:',
            disabled=False,
        )
        
        # Display the widget
        display(experiment_dropdown)
        
        # Function to load selected experiment
        def load_selected_experiment(change):
            global result, metadata
            experiment_id = change.new
            result, metadata = load_experiment_results(experiment_id)
            print(f"✅ Loaded experiment: {experiment_id}")
            
            # Display experiment overview
            print("\n📋 Experiment Overview:")
            print(f"• Model: {metadata.get('model', 'N/A')}")
            print(f"• Date: {metadata.get('date', 'N/A')}")
            print(f"• Fields: {', '.join(metadata.get('fields', []))}")
            print(f"• Type: {metadata.get('type', 'N/A')}")
        
        # Register callback
        experiment_dropdown.observe(load_selected_experiment, names='value')
        """
    except Exception as e:
        print(f"❌ Error loading experiment results: {str(e)}")
        print("In a real notebook, select a valid experiment ID")
else:
    print("Utilities not available, cannot analyze experiments.")

# %% [markdown]
# ## 3. Experiment Performance Analysis
#
# Analyze the extraction performance of the selected experiment.

# %%
if utils_available and 'result' in locals():
    try:
        # Create a dashboard of experiment results
        print("📊 Generating Experiment Dashboard...")
        dashboard = viz_manager.show_experiment_results(experiment_id=experiment_id)
        
        # Overall metrics
        print("\n📈 Overall Performance Metrics:")
        if hasattr(result, 'metrics'):
            for metric_name, metric_value in result.metrics.items():
                if isinstance(metric_value, (int, float)):
                    print(f"• {metric_name}: {metric_value:.4f}")
                else:
                    print(f"• {metric_name}: {metric_value}")
        else:
            print("No overall metrics available")
        
        # Field-specific performance
        print("\n📊 Field-Specific Performance:")
        
        # Create a performance summary DataFrame
        performance_data = []
        
        if hasattr(result, 'field_results'):
            for field, performances in result.field_results.items():
                for perf in performances:
                    performance_data.append({
                        'Field': field,
                        'Prompt': perf.prompt_name,
                        'Accuracy': perf.accuracy * 100 if hasattr(perf, 'accuracy') else 0,
                        'Success Count': perf.successful_extractions if hasattr(perf, 'successful_extractions') else 0,
                        'Total Items': perf.total_items if hasattr(perf, 'total_items') else 0
                    })
        
        if performance_data:
            performance_df = pd.DataFrame(performance_data)
            display(performance_df)
            
            # Create a bar chart of field accuracies
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Field', y='Accuracy', hue='Prompt', data=performance_df)
            plt.title('Extraction Accuracy by Field')
            plt.xlabel('Field')
            plt.ylabel('Accuracy (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("No field performance data available")
            
        # Sample extractions
        print("\n📝 Sample Extractions:")
        if hasattr(result, 'extractions') and result.extractions:
            # Create a DataFrame for sample extractions
            sample_data = []
            for i, extraction in enumerate(result.extractions[:5]):  # Show first 5
                row = {'Document ID': extraction.document_id}
                
                # Add extracted fields
                if hasattr(extraction, 'extracted_fields'):
                    for field, value in extraction.extracted_fields.items():
                        row[field] = value
                
                # Add accuracy if available
                if hasattr(extraction, 'accuracy'):
                    row['Accuracy'] = extraction.accuracy
                
                sample_data.append(row)
            
            if sample_data:
                sample_df = pd.DataFrame(sample_data)
                display(sample_df)
        else:
            print("No sample extractions available")
    except Exception as e:
        print(f"❌ Error analyzing experiment: {str(e)}")
else:
    print("Load an experiment first to analyze its performance")

# %% [markdown]
# ## 4. Compare Experiments
#
# Compare multiple experiments to identify the best performing configurations.

# %%
if utils_available:
    # Initialize experiment comparator
    comparator = ExperimentComparator(experiment_loader=experiment_loader)
    
    # Select experiments to compare
    # In a real notebook, users would select these interactively
    if len(discovered_experiments) >= 2:
        # Take two most recent experiments for demonstration
        sorted_experiments = sorted(discovered_experiments, key=lambda x: x.get('date', ''), reverse=True)
        experiment_ids = [exp.get('experiment_id') for exp in sorted_experiments[:2]]
        
        print(f"Comparing experiments: {', '.join(experiment_ids)}")
        
        # Compare experiments
        try:
            comparison_result = comparator.compare_experiments(
                experiment_ids=experiment_ids,
                name="Example Comparison"
            )
            
            print("\n📊 Experiment Comparison Results:")
            
            # Display comparison data
            if hasattr(comparison_result, 'data_points'):
                comparison_data = []
                
                for data_point in comparison_result.data_points:
                    comparison_data.append({
                        'Experiment': data_point.label,
                        'Accuracy': data_point.metrics.get('accuracy', 0) * 100,
                        'Processing Time': data_point.metrics.get('processing_time', 0),
                        'Sample Size': data_point.sample_size
                    })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    display(comparison_df)
                    
                    # Create comparison visualization
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='Experiment', y='Accuracy', data=comparison_df)
                    plt.title('Accuracy Comparison Between Experiments')
                    plt.xlabel('Experiment')
                    plt.ylabel('Accuracy (%)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()
            else:
                print("No comparison data available")
            
            # Show statistical significance
            if hasattr(comparison_result, 'statistics'):
                print("\n📈 Statistical Analysis:")
                
                for stat_name, stat_value in comparison_result.statistics.items():
                    print(f"• {stat_name}: {stat_value}")
                
                if 'significant_difference' in comparison_result.statistics:
                    sig_diff = comparison_result.statistics['significant_difference']
                    if sig_diff:
                        print("✅ The difference between experiments is statistically significant")
                    else:
                        print("❌ The difference between experiments is NOT statistically significant")
            
            # Show visualization suggestions
            if hasattr(comparison_result, 'visualization_suggestions') and comparison_result.visualization_suggestions:
                print("\n🎨 Visualization Suggestions:")
                for suggestion in comparison_result.visualization_suggestions:
                    print(f"• {suggestion}")
        except Exception as e:
            print(f"❌ Error comparing experiments: {str(e)}")
    else:
        print("At least two experiments are needed for comparison. Run more experiments first.")
        
    # Provide code template for model comparison
    print("\n💡 To compare different models:")
    print("""
    # Example:
    model_comparison = comparator.compare_models(
        model_names=["llava-1.5-7b", "phi-2"],
        field="invoice_number"
    )
    """)
    
    # Provide code template for prompt comparison
    print("\n💡 To compare different prompts:")
    print("""
    # Example:
    prompt_comparison = comparator.compare_experiments(
        experiment_ids=["exp_id_1", "exp_id_2"],
        field="invoice_number",
        comparison_dimension="prompt"
    )
    """)
else:
    print("Utilities not available, cannot compare experiments.")

# %% [markdown]
# ## 5. Advanced Analysis Templates
#
# Select from pre-configured analysis templates for common scenarios.

# %%
if utils_available:
    print("📊 Available Analysis Templates:")
    print("1. Model Performance Comparison - Compare accuracy across different models")
    print("2. Field Extraction Difficulty - Analyze which fields are hardest to extract")
    print("3. Quantization Impact - Measure the impact of model quantization on performance")
    print("4. Prompt Effectiveness - Compare different prompt strategies")
    
    # In a real notebook, users would select a template interactively
    selected_template = 1  # For demonstration
    print(f"\nSelected template: Model Performance Comparison")
    
    # Execute the selected template
    if selected_template == 1 and experiment_loader:
        # Model Performance Comparison template
        print("\n📊 Executing Model Performance Comparison...")
        
        # Find experiments with different models
        model_experiments = experiment_loader.filter_experiments(
            min_date="2023-01-01"  # Filter to recent experiments
        )
        
        # Group by model
        model_groups = {}
        for exp in model_experiments:
            model = exp.get('model', 'Unknown')
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(exp)
        
        # Use the most recent experiment for each model
        most_recent_by_model = {}
        for model, exps in model_groups.items():
            if exps:
                most_recent = sorted(exps, key=lambda x: x.get('date', ''), reverse=True)[0]
                most_recent_by_model[model] = most_recent
        
        if len(most_recent_by_model) >= 2:
            # Create model comparison
            model_names = list(most_recent_by_model.keys())
            print(f"Comparing models: {', '.join(model_names)}")
            
            try:
                model_comparison = comparator.compare_models(
                    model_names=model_names
                )
                
                # Display comparison results
                if hasattr(model_comparison, 'data_points'):
                    model_data = []
                    for data_point in model_comparison.data_points:
                        model_data.append({
                            'Model': data_point.label,
                            'Accuracy': data_point.metrics.get('accuracy', 0) * 100,
                            'Processing Time': data_point.metrics.get('processing_time', 0),
                            'Memory Usage': data_point.metrics.get('memory_usage', 0)
                        })
                    
                    if model_data:
                        model_df = pd.DataFrame(model_data)
                        display(model_df)
                        
                        # Create visualization
                        fig, ax1 = plt.subplots(figsize=(10, 6))
                        
                        # Accuracy bars
                        sns.barplot(x='Model', y='Accuracy', data=model_df, ax=ax1, color='blue', alpha=0.7)
                        ax1.set_ylabel('Accuracy (%)', color='blue')
                        
                        # Processing time line on secondary axis
                        ax2 = ax1.twinx()
                        sns.pointplot(x='Model', y='Processing Time', data=model_df, ax=ax2, color='red')
                        ax2.set_ylabel('Processing Time (s)', color='red')
                        
                        plt.title('Model Comparison: Accuracy vs. Processing Time')
                        plt.tight_layout()
                        plt.show()
                        
                        # Recommendations
                        print("\n🔍 Analysis Recommendations:")
                        best_accuracy_model = model_df.loc[model_df['Accuracy'].idxmax()]['Model']
                        fastest_model = model_df.loc[model_df['Processing Time'].idxmin()]['Model']
                        
                        print(f"• Best accuracy: {best_accuracy_model}")
                        print(f"• Fastest processing: {fastest_model}")
                        
                        if best_accuracy_model == fastest_model:
                            print(f"✅ {best_accuracy_model} offers the best overall performance")
                        else:
                            print(f"💡 Consider {best_accuracy_model} for accuracy-critical tasks")
                            print(f"💡 Consider {fastest_model} for speed-critical tasks")
                else:
                    print("No comparison data available")
            except Exception as e:
                print(f"❌ Error in model comparison: {str(e)}")
        else:
            print("Need at least two different models to compare. Run experiments with different models first.")
    
    # Provide code for other templates
    print("\n💡 To execute other analysis templates, select a different template number.")
else:
    print("Utilities not available, cannot run analysis templates.")

# %% [markdown]
# ## 6. Export Results
#
# Export your analysis for sharing or reporting.

# %%
if utils_available and 'result' in locals():
    # Export options
    print("📤 Export Options:")
    print("1. Export to HTML")
    print("2. Export to Markdown")
    print("3. Export to CSV")
    print("4. Export to JSON")
    
    # In a real notebook, users would select an option interactively
    export_format = 1  # For demonstration
    export_path = f"./export_{experiment_id}.html"  # Default path
    
    print(f"\nSelected export format: HTML")
    print(f"Export path: {export_path}")
    
    # Export based on selected format
    try:
        if export_format == 1:  # HTML
            # Generate HTML report using the visualization manager
            if viz_manager:
                html_report = viz_manager.export_experiment_report(
                    experiment_id=experiment_id,
                    output_format="html"
                )
                
                # Save to file
                with open(export_path, 'w') as f:
                    f.write(html_report)
                print(f"✅ Exported HTML report to: {export_path}")
        elif export_format == 2:  # Markdown
            # Generate Markdown report
            if viz_manager:
                md_report = viz_manager.export_experiment_report(
                    experiment_id=experiment_id,
                    output_format="markdown"
                )
                
                # Save to file
                md_path = export_path.replace('.html', '.md')
                with open(md_path, 'w') as f:
                    f.write(md_report)
                print(f"✅ Exported Markdown report to: {md_path}")
        elif export_format == 3:  # CSV
            # Export relevant data to CSV
            if hasattr(result, 'field_results'):
                # Create a flattened DataFrame
                export_data = []
                
                for field, performances in result.field_results.items():
                    for perf in performances:
                        row = {
                            'Field': field,
                            'Prompt': perf.prompt_name if hasattr(perf, 'prompt_name') else 'Default',
                            'Accuracy': perf.accuracy if hasattr(perf, 'accuracy') else 0,
                            'Success_Count': perf.successful_extractions if hasattr(perf, 'successful_extractions') else 0,
                            'Total_Items': perf.total_items if hasattr(perf, 'total_items') else 0
                        }
                        export_data.append(row)
                
                # Create DataFrame and export
                if export_data:
                    export_df = pd.DataFrame(export_data)
                    csv_path = export_path.replace('.html', '.csv')
                    export_df.to_csv(csv_path, index=False)
                    print(f"✅ Exported CSV data to: {csv_path}")
                else:
                    print("No data available to export")
            else:
                print("No field results available to export")
        elif export_format == 4:  # JSON
            # Export result as JSON
            json_path = export_path.replace('.html', '.json')
            
            # Convert result to JSON-serializable format
            if hasattr(result, 'to_dict'):
                json_data = result.to_dict()
            else:
                # Use __dict__ as fallback
                json_data = result.__dict__
            
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            print(f"✅ Exported JSON data to: {json_path}")
    except Exception as e:
        print(f"❌ Error exporting results: {str(e)}")
    
    # Provide code for programmatic export
    print("\n💡 To customize export:")
    print("""
    # Example for custom HTML export:
    custom_html = viz_manager.export_experiment_report(
        experiment_id=experiment_id,
        output_format="html",
        include_visualizations=True,
        include_raw_data=False
    )
    
    with open("custom_export.html", "w") as f:
        f.write(custom_html)
    """)
else:
    print("Load an experiment first to export its results.")

# %% [markdown]
# ## 7. Next Steps
#
# Based on your analysis, you can:
#
# 1. **Refine your experiments** - Adjust model parameters, prompts, or fields based on results
# 2. **Deploy your best model** - Use the Model Deployment notebook to deploy your best performing model
# 3. **Create custom visualizations** - Use the raw data to create custom visualizations for specific needs
# 4. **Share your results** - Export and share your analysis with team members

# %%
print("✅ Analysis complete!")
print("To analyze different experiments, return to Step 2 and select a different experiment.") 
"""
Notebook Visualization Utilities

This module provides specialized visualization utilities for Jupyter notebooks,
integrating with the core visualization system and enhancing it with notebook-specific
capabilities.
"""

import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Import the notebook visualizations from the core visualization module
# Move these imports inside functions to avoid circular imports
# from src.analysis.visualization import get_notebook_visualizations, NotebookVisualizations
from src.results.collector import ResultsCollector, get_results_collector

# Configure logging
logger = logging.getLogger(__name__)


class NotebookVisualizationManager:
    """
    Manager class for notebook-specific visualizations that integrates with the
    three-notebook system architecture.
    
    This class provides a higher-level interface for creating visualizations
    in Jupyter notebooks, with methods specifically designed for each of the
    three notebook roles (setup, experiment, analysis).
    """
    
    def __init__(self, 
                results_collector: Optional[ResultsCollector] = None,
                experiment_name: Optional[str] = None,
                output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the notebook visualization manager.
        
        Args:
            results_collector: Optional ResultsCollector instance
            experiment_name: Name of the experiment for visualization titles
            output_dir: Directory to save visualizations
        """
        # Initialize or get a results collector
        if results_collector is None and experiment_name is not None:
            self.results_collector = get_results_collector(experiment_name=experiment_name)
        else:
            self.results_collector = results_collector
        
        # Set output directory
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import visualization components here to avoid circular imports
        from src.analysis.visualization import get_notebook_visualizations
        
        # Get notebook visualizations instance
        self.notebook_viz = get_notebook_visualizations(
            results_collector=self.results_collector,
            style="notebook"
        )
        
        # Initialize experiment results cache for multiple experiments
        self.experiment_results_cache = {}
        
        # Check if we're in a notebook environment
        self._check_notebook_environment()
    
    def _check_notebook_environment(self):
        """Check if we're running in a Jupyter notebook and set up accordingly."""
        try:
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            self.is_notebook = (shell == 'ZMQInteractiveShell')
            
            if self.is_notebook:
                # Set up notebook-specific settings
                get_ipython().run_line_magic('matplotlib', 'inline')
                logger.info("Running in Jupyter notebook environment")
                
                # Try to load ipywidgets and plotly for interactive visualization
                try:
                    import ipywidgets
                    import plotly.express
                    self.has_interactive = True
                    logger.info("Interactive visualization components available")
                except ImportError:
                    self.has_interactive = False
                    logger.info("Interactive components not available. Install ipywidgets and plotly for full functionality.")
            else:
                self.is_notebook = False
                self.has_interactive = False
        except:
            self.is_notebook = False
            self.has_interactive = False
            logger.info("Not running in Jupyter notebook environment")

    # Setup Notebook Visualizations
    
    def show_environment_status(self, check_gpu: bool = True):
        """
        Create a visualization showing the environment setup status.
        
        Args:
            check_gpu: Whether to check and display GPU status
            
        Returns:
            Visualization of environment status
        """
        # Collect system information
        system_info = {
            'Python Version': sys.version,
            'Operating System': platform.system() + ' ' + platform.release()
        }
        
        # Check if IPython is available
        try:
            import IPython
            system_info['IPython Version'] = IPython.__version__
        except:
            system_info['IPython Version'] = 'Not available'
        
        # Check for required packages
        required_packages = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 
            'torch', 'transformers', 'PIL', 'ipywidgets', 'plotly'
        ]
        
        package_status = {}
        for package in required_packages:
            try:
                module = importlib.import_module(package)
                if hasattr(module, '__version__'):
                    package_status[package] = module.__version__
                else:
                    package_status[package] = 'Installed (version unknown)'
            except ImportError:
                package_status[package] = 'Not installed'
        
        # Check GPU if requested
        gpu_info = {}
        if check_gpu:
            try:
                import torch
                gpu_info['CUDA Available'] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    gpu_info['CUDA Version'] = torch.version.cuda
                    gpu_info['GPU Count'] = torch.cuda.device_count()
                    for i in range(torch.cuda.device_count()):
                        gpu_info[f'GPU {i}'] = torch.cuda.get_device_name(i)
                        gpu_info[f'GPU {i} Memory'] = f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
            except:
                gpu_info['Error'] = 'Could not retrieve GPU information'
        
        # Display the information
        if self.is_notebook:
            try:
                from IPython.display import display, HTML
                
                # System information
                display(HTML("<h2>System Information</h2>"))
                system_df = pd.DataFrame([system_info]).T.reset_index()
                system_df.columns = ['Component', 'Status']
                display(system_df)
                
                # Package information
                display(HTML("<h2>Package Information</h2>"))
                package_df = pd.DataFrame([package_status]).T.reset_index()
                package_df.columns = ['Package', 'Version']
                
                # Color code based on installation status
                def color_status(val):
                    if 'Not installed' in str(val):
                        return f'background-color: #FFCCCC'
                    else:
                        return ''
                
                display(package_df.style.applymap(color_status, subset=['Version']))
                
                # GPU information if available
                if gpu_info:
                    display(HTML("<h2>GPU Information</h2>"))
                    gpu_df = pd.DataFrame([gpu_info]).T.reset_index()
                    gpu_df.columns = ['Component', 'Status']
                    display(gpu_df)
                
                return True
            except:
                for section, data in [("System Information", system_info), 
                                     ("Package Information", package_status),
                                     ("GPU Information", gpu_info)]:
                    print(f"\n--- {section} ---")
                    for k, v in data.items():
                        print(f"{k}: {v}")
                return True
        else:
            # Text-based output for non-notebook environments
            for section, data in [("System Information", system_info), 
                                 ("Package Information", package_status),
                                 ("GPU Information", gpu_info)]:
                print(f"\n--- {section} ---")
                for k, v in data.items():
                    print(f"{k}: {v}")
            return True
    
    # Experiment Notebook Visualizations
    
    def show_experiment_progress(self, 
                               current_step: int, 
                               total_steps: int, 
                               step_name: str,
                               metrics: Optional[Dict[str, float]] = None):
        """
        Show a visualization of experiment progress.
        
        Args:
            current_step: Current step number
            total_steps: Total number of steps
            step_name: Name of the current step
            metrics: Optional metrics to display
            
        Returns:
            Progress visualization
        """
        if not self.is_notebook:
            print(f"Progress: {current_step}/{total_steps} - {step_name}")
            if metrics:
                for name, value in metrics.items():
                    print(f"  {name}: {value}")
            return
        
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
            
            # Create progress bar
            progress = widgets.FloatProgress(
                value=current_step,
                min=0,
                max=total_steps,
                description='Progress:',
                bar_style='info',
                style={'bar_color': '#0078D7'},
                orientation='horizontal'
            )
            
            # Create step label
            step_label = widgets.HTML(
                value=f"<h3>Step {current_step}/{total_steps}: {step_name}</h3>"
            )
            
            # Create metrics display if provided
            metrics_display = None
            if metrics:
                metrics_html = "<div style='margin-top: 10px; padding: 10px; background-color: #f5f5f5; border-radius: 5px;'>"
                metrics_html += "<h4>Metrics:</h4><ul>"
                for name, value in metrics.items():
                    if isinstance(value, float):
                        metrics_html += f"<li><b>{name}:</b> {value:.4f}</li>"
                    else:
                        metrics_html += f"<li><b>{name}:</b> {value}</li>"
                metrics_html += "</ul></div>"
                metrics_display = widgets.HTML(value=metrics_html)
            
            # Display the progress visualization
            clear_output(wait=True)
            if metrics_display:
                display(widgets.VBox([step_label, progress, metrics_display]))
            else:
                display(widgets.VBox([step_label, progress]))
                
            return progress
            
        except ImportError:
            # Fall back to text-based progress
            print(f"Progress: {current_step}/{total_steps} - {step_name}")
            if metrics:
                for name, value in metrics.items():
                    print(f"  {name}: {value}")
            return None
    
    def show_extraction_preview(self, 
                              sample_extractions: List[Dict],
                              field_name: str = None):
        """
        Show a preview of extraction results.
        
        Args:
            sample_extractions: List of extraction results
            field_name: Optional field name to filter by
            
        Returns:
            Extraction preview visualization
        """
        if not sample_extractions:
            logger.warning("No sample extractions provided")
            return None
        
        # Filter by field if specified
        if field_name:
            filtered_extractions = [
                ext for ext in sample_extractions 
                if ext.get('field_name') == field_name
            ]
        else:
            filtered_extractions = sample_extractions
        
        if not filtered_extractions:
            logger.warning(f"No extractions found for field '{field_name}'")
            return None
        
        # Create a DataFrame for display
        preview_data = []
        for ext in filtered_extractions:
            preview_data.append({
                'Field': ext.get('field_name', 'Unknown'),
                'Prompt': ext.get('prompt_name', 'Unknown'),
                'Document': ext.get('document_name', ext.get('document_id', 'Unknown')),
                'Extracted Value': ext.get('extracted_value', 'None'),
                'Ground Truth': ext.get('ground_truth', 'Unknown'),
                'Status': ext.get('status', 'Unknown'),
                'Confidence': ext.get('confidence', None)
            })
        
        preview_df = pd.DataFrame(preview_data)
        
        # Display the preview
        if self.is_notebook:
            try:
                from IPython.display import display, HTML
                
                # Style the DataFrame based on extraction status
                def style_status(val):
                    if val == 'Exact Match':
                        return 'background-color: #CCFFCC'
                    elif val == 'Partial Match':
                        return 'background-color: #FFFFCC'
                    elif val == 'Failed':
                        return 'background-color: #FFCCCC'
                    else:
                        return ''
                
                styled_df = preview_df.style.applymap(style_status, subset=['Status'])
                
                # Display with a title
                title = f"Extraction Preview for {field_name}" if field_name else "Extraction Preview"
                display(HTML(f"<h2>{title}</h2>"))
                display(styled_df)
                
                return styled_df
            except:
                print(f"\n--- Extraction Preview ---")
                print(preview_df)
                return preview_df
        else:
            print(f"\n--- Extraction Preview ---")
            print(preview_df)
            return preview_df
    
    # Analysis Notebook Visualizations
    
    def show_experiment_results(self, 
                              experiment_name: Optional[str] = None,
                              results_file: Optional[str] = None,
                              interactive: bool = True):
        """
        Show comprehensive results for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            results_file: Path to results file (if different from default)
            interactive: Whether to use interactive visualizations
            
        Returns:
            Dashboard of experiment results
        """
        # Get results either from collector or file
        results = None
        
        if results_file:
            # Try to load from file
            try:
                if not os.path.exists(results_file):
                    logger.error(f"Results file not found: {results_file}")
                    return None
                
                # Create a temporary collector to load the file
                temp_collector = get_results_collector()
                temp_collector.load_experiment(results_file)
                results = temp_collector.get_results()
                
                # Cache the results for future use
                if experiment_name:
                    self.experiment_results_cache[experiment_name] = results
                
            except Exception as e:
                logger.error(f"Error loading results file: {e}")
                return None
                
        elif experiment_name:
            # Check cache first
            if experiment_name in self.experiment_results_cache:
                results = self.experiment_results_cache[experiment_name]
            # Try to get from collector
            elif self.results_collector and self.results_collector.experiment_name == experiment_name:
                results = self.results_collector.get_results()
            # Try to create a new collector for this experiment
            else:
                try:
                    temp_collector = get_results_collector(experiment_name=experiment_name)
                    results = temp_collector.get_results()
                    # Cache the results
                    self.experiment_results_cache[experiment_name] = results
                except Exception as e:
                    logger.error(f"Error getting results for experiment {experiment_name}: {e}")
                    return None
        else:
            # Try to use the default collector
            if self.results_collector:
                results = self.results_collector.get_results()
            else:
                logger.error("No experiment name or results file provided, and no default collector available")
                return None
        
        if not results:
            logger.error("Could not retrieve experiment results")
            return None
        
        # Show the results
        if interactive and self.has_interactive:
            return self.notebook_viz.create_dashboard(results)
        else:
            return self.notebook_viz.accuracy_comparison(results, interactive=False)
    
    def compare_experiments(self, 
                          experiment_names: List[str],
                          metric: str = 'accuracy',
                          interactive: bool = True):
        """
        Compare results across multiple experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            metric: Metric to compare ('accuracy', 'processing_time', etc.)
            interactive: Whether to use interactive visualization
            
        Returns:
            Comparison visualization
        """
        if not experiment_names:
            logger.error("No experiment names provided for comparison")
            return None
        
        # Load results for each experiment
        experiment_results = {}
        for name in experiment_names:
            # Check cache first
            if name in self.experiment_results_cache:
                experiment_results[name] = self.experiment_results_cache[name]
                continue
                
            # Try to load from collector
            try:
                collector = get_results_collector(experiment_name=name)
                results = collector.get_results()
                if results:
                    experiment_results[name] = results
                    # Cache for future use
                    self.experiment_results_cache[name] = results
                else:
                    logger.warning(f"No results found for experiment {name}")
            except Exception as e:
                logger.error(f"Error loading results for experiment {name}: {e}")
        
        if not experiment_results:
            logger.error("Could not load any experiment results")
            return None
        
        # Create comparison visualization based on the metric
        if self.is_notebook and interactive and self.has_interactive:
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                from IPython.display import display
                
                # Create a comparison table
                comparison_data = []
                
                # Process each experiment
                for experiment_name, results in experiment_results.items():
                    # Extract field results
                    if 'field_results' not in results:
                        logger.warning(f"No field results found in experiment {experiment_name}")
                        continue
                        
                    # Process each field
                    for field_name, field_results in results['field_results'].items():
                        for prompt_result in field_results:
                            data_point = {
                                'Experiment': experiment_name,
                                'Field': field_name,
                                'Prompt': prompt_result.get('prompt_name', 'Unknown')
                            }
                            
                            # Add the requested metric
                            if metric == 'accuracy':
                                data_point['Value'] = prompt_result.get('accuracy', 0) * 100
                                data_point['Metric'] = 'Accuracy (%)'
                            elif metric == 'processing_time':
                                data_point['Value'] = prompt_result.get('processing_time', 0)
                                data_point['Metric'] = 'Processing Time (s)'
                            elif metric == 'success_count':
                                data_point['Value'] = prompt_result.get('successful_extractions', 0)
                                data_point['Metric'] = 'Successful Extractions'
                            elif metric == 'error_rate':
                                accuracy = prompt_result.get('accuracy', 0)
                                data_point['Value'] = (1 - accuracy) * 100
                                data_point['Metric'] = 'Error Rate (%)'
                            else:
                                # Try to get the metric directly from the result
                                data_point['Value'] = prompt_result.get(metric, 0)
                                data_point['Metric'] = metric
                            
                            comparison_data.append(data_point)
                
                if not comparison_data:
                    logger.error("No data available for comparison")
                    return None
                
                # Create DataFrame
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create visualization
                fig = px.bar(
                    comparison_df, 
                    x='Field', 
                    y='Value', 
                    color='Experiment',
                    barmode='group',
                    facet_col='Prompt',
                    title=f'Experiment Comparison: {metric.replace("_", " ").title()}',
                    labels={'Value': comparison_df['Metric'].iloc[0]},
                    height=500
                )
                
                # Update layout
                fig.update_layout(
                    legend_title='Experiment',
                    font=dict(size=12)
                )
                
                # Show the figure
                fig.show()
                
                return fig
                
            except Exception as e:
                logger.error(f"Error creating interactive comparison: {e}")
                # Fall back to static visualization
        
        # Static visualization
        try:
            # Create a multi-panel figure
            fig, axes = plt.subplots(
                1, len(experiment_results), 
                figsize=(5 * len(experiment_results), 6),
                sharey=True
            )
            
            if len(experiment_results) == 1:
                axes = [axes]
            
            # Plot each experiment
            for i, (name, results) in enumerate(experiment_results.items()):
                ax = axes[i]
                
                # Extract field results
                field_data = {}
                
                if 'field_results' in results:
                    for field, field_results in results['field_results'].items():
                        field_data[field] = {}
                        for prompt_result in field_results:
                            prompt_name = prompt_result.get('prompt_name', 'Unknown')
                            
                            # Get the requested metric
                            if metric == 'accuracy':
                                value = prompt_result.get('accuracy', 0) * 100
                            elif metric == 'processing_time':
                                value = prompt_result.get('processing_time', 0)
                            elif metric == 'success_count':
                                value = prompt_result.get('successful_extractions', 0)
                            elif metric == 'error_rate':
                                accuracy = prompt_result.get('accuracy', 0)
                                value = (1 - accuracy) * 100
                            else:
                                # Try to get the metric directly
                                value = prompt_result.get(metric, 0)
                                
                            field_data[field][prompt_name] = value
                
                # Convert to DataFrame for plotting
                if field_data:
                    df = pd.DataFrame(field_data).T
                    df.plot(kind='bar', ax=ax)
                    
                    # Set titles
                    ax.set_title(name)
                    if i == 0:
                        if metric == 'accuracy':
                            ax.set_ylabel('Accuracy (%)')
                        elif metric == 'processing_time':
                            ax.set_ylabel('Processing Time (s)')
                        elif metric == 'success_count':
                            ax.set_ylabel('Successful Extractions')
                        elif metric == 'error_rate':
                            ax.set_ylabel('Error Rate (%)')
                        else:
                            ax.set_ylabel(metric.replace('_', ' ').title())
                    
                    ax.set_xlabel('Field')
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set overall title
            fig.suptitle(f'Experiment Comparison: {metric.replace("_", " ").title()}', fontsize=16)
            plt.tight_layout()
            
            return fig
                
        except Exception as e:
            logger.error(f"Error creating comparison visualization: {e}")
            return None
    
    def export_visualization_pack(self, 
                                experiment_name: Optional[str] = None,
                                output_dir: Optional[Union[str, Path]] = None,
                                formats: List[str] = ['png', 'html', 'csv']):
        """
        Export a comprehensive pack of visualizations and data for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save the exports
            formats: List of formats to export
            
        Returns:
            Dictionary of exported files
        """
        # Set output directory
        if output_dir:
            output_dir = Path(output_dir)
        elif self.output_dir:
            output_dir = self.output_dir
        else:
            output_dir = Path(f"./visualization_exports/{experiment_name or 'experiment'}")
        
        # Create directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get results
        if experiment_name and experiment_name in self.experiment_results_cache:
            results = self.experiment_results_cache[experiment_name]
        elif self.results_collector:
            results = self.results_collector.get_results()
        else:
            logger.error("No results available for export")
            return {}
        
        if not results:
            logger.error("No results available for export")
            return {}
        
        # Export visualizations
        exported_files = {}
        
        # Export standard visualizations
        if 'png' in formats or 'pdf' in formats:
            # Use the notebook_viz to export plots
            for fmt in [f for f in formats if f in ['png', 'pdf', 'svg']]:
                viz_files = self.notebook_viz.export_all_plots(
                    results=results,
                    output_dir=output_dir / fmt,
                    format=fmt
                )
                
                # Add to exported files
                for name, path in viz_files.items():
                    exported_files[f"{name}_{fmt}"] = path
        
        # Export interactive dashboard
        if 'html' in formats:
            try:
                dashboard_path = output_dir / 'html' / 'dashboard.html'
                dashboard_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create dashboard
                self.notebook_viz.create_dashboard(
                    results=results,
                    output_filename=str(dashboard_path)
                )
                
                exported_files['dashboard_html'] = dashboard_path
            except Exception as e:
                logger.error(f"Error exporting dashboard: {e}")
        
        # Export data as CSV
        if 'csv' in formats:
            try:
                # Create data directory
                data_dir = output_dir / 'data'
                data_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract field results
                if 'field_results' in results:
                    # Create a flattened DataFrame of all results
                    flat_data = []
                    
                    for field, field_results in results['field_results'].items():
                        for result in field_results:
                            # Create a flat dictionary with all result data
                            flat_result = {
                                'field': field,
                                'prompt': result.get('prompt_name', 'Unknown'),
                                'accuracy': result.get('accuracy', 0),
                                'successful_extractions': result.get('successful_extractions', 0),
                                'total_items': result.get('total_items', 0)
                            }
                            
                            # Add any other fields from the result
                            for key, value in result.items():
                                if key not in flat_result and not isinstance(value, (list, dict)):
                                    flat_result[key] = value
                                    
                            flat_data.append(flat_result)
                    
                    # Create and save DataFrame
                    if flat_data:
                        results_df = pd.DataFrame(flat_data)
                        csv_path = data_dir / 'field_results.csv'
                        results_df.to_csv(csv_path, index=False)
                        exported_files['field_results_csv'] = csv_path
                
                # Export experiment metadata
                metadata = {
                    'experiment_name': experiment_name or results.get('experiment_name', 'Unknown'),
                    'timestamp': results.get('timestamp', 'Unknown'),
                    'model': results.get('model', 'Unknown'),
                    'total_documents': results.get('total_documents', 0),
                    'total_fields': len(results.get('field_results', {})),
                    'total_prompts': len(set(r.get('prompt_name', '') 
                                          for field_results in results.get('field_results', {}).values() 
                                          for r in field_results))
                }
                
                # Add any other top-level metadata
                for key, value in results.items():
                    if key not in ['field_results', 'document_results'] and not isinstance(value, (list, dict)):
                        metadata[key] = value
                
                # Save metadata
                metadata_df = pd.DataFrame([metadata])
                metadata_path = data_dir / 'experiment_metadata.csv'
                metadata_df.to_csv(metadata_path, index=False)
                exported_files['metadata_csv'] = metadata_path
                
            except Exception as e:
                logger.error(f"Error exporting data as CSV: {e}")
        
        # Report on exported files
        if exported_files:
            logger.info(f"Exported {len(exported_files)} files to {output_dir}")
            
            # Display summary in notebook
            if self.is_notebook:
                try:
                    from IPython.display import display, HTML
                    
                    html = f"<h2>Visualization Export Complete</h2>"
                    html += f"<p>Exported {len(exported_files)} files to <code>{output_dir}</code></p>"
                    
                    # Group by type
                    file_types = {}
                    for name, path in exported_files.items():
                        file_type = name.split('_')[-1]
                        if file_type not in file_types:
                            file_types[file_type] = []
                        file_types[file_type].append(path)
                    
                    # Create summary by type
                    html += "<ul>"
                    for file_type, paths in file_types.items():
                        html += f"<li><strong>{file_type.upper()}</strong>: {len(paths)} files</li>"
                    html += "</ul>"
                    
                    display(HTML(html))
                except:
                    pass
        
        return exported_files


# Global access function
def get_notebook_visualization_manager(
    results_collector: Optional[ResultsCollector] = None,
    experiment_name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None
) -> NotebookVisualizationManager:
    """
    Get a notebook visualization manager instance.
    
    Args:
        results_collector: Optional ResultsCollector instance
        experiment_name: Name of the experiment for visualization titles
        output_dir: Directory to save visualizations
        
    Returns:
        NotebookVisualizationManager instance
    """
    return NotebookVisualizationManager(
        results_collector=results_collector,
        experiment_name=experiment_name,
        output_dir=output_dir
    ) 
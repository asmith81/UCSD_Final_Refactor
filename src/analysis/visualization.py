"""
Results Visualization

This module provides comprehensive visualization capabilities for extraction results,
experiment performance metrics, and comparative analysis. It serves as the central
visualization service for the entire pipeline.
"""

import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Import the schema components to ensure type compatibility
from src.results.schema import ExperimentResult, PromptPerformance, IndividualExtractionResult, ExtractionStatus

# Configure logging
logger = logging.getLogger(__name__)


class VisualizationService:
    """
    Service for creating and managing visualizations for extraction results.
    
    This service centralizes all visualization functionality, making it easier
    to maintain and extend visualization capabilities while keeping pipeline
    stages focused on their core responsibilities.
    """
    
    def __init__(self, 
                output_dir: Optional[Union[str, Path]] = None,
                experiment_name: Optional[str] = None,
                style: str = "default"):
        """
        Initialize the visualization service.
        
        Args:
            output_dir: Base directory for storing visualizations
            experiment_name: Name of the current experiment
            style: Matplotlib style to use for visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.experiment_name = experiment_name or "experiment"
        
        # Set up directories if provided
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        self._set_plotting_style(style)
        
        # Register available visualization types
        self.visualization_types = {
            "accuracy_comparison": self._create_accuracy_comparison,
            "processing_time": self._create_processing_time_comparison,
            "error_analysis": self._create_error_analysis,
            "prompt_radar": self._create_prompt_radar,
            "field_performance": self._create_field_performance,
            "error_distribution": self._create_error_distribution,
            "performance_insights": self._create_performance_insights
        }
        
        logger.info(f"Visualization service initialized with {len(self.visualization_types)} visualization types")
    
    def _set_plotting_style(self, style: str) -> None:
        """
        Set the matplotlib plotting style.
        
        Args:
            style: Style name to use
        """
        try:
            if style == "default":
                plt.style.use('seaborn-v0_8-colorblind')
            else:
                plt.style.use(style)
        except Exception as e:
            logger.warning(f"Could not set plotting style {style}: {e}")
            # Fall back to default style
            pass
    
    def create_visualizations(self,
                            results: Union[Dict[str, Any], ExperimentResult],
                            viz_types: Optional[List[str]] = None,
                            show_plots: bool = False) -> Dict[str, Path]:
        """
        Create standard visualizations for extraction results.
        
        Args:
            results: Either an ExperimentResult object or a dictionary of results
            viz_types: Types of visualizations to create (None for all available)
            show_plots: Whether to display plots interactively
            
        Returns:
            Dictionary mapping visualization names to saved file paths
        """
        # Determine which visualization types to create
        if viz_types is None:
            viz_types = list(self.visualization_types.keys())
        
        logger.info(f"Creating {len(viz_types)} visualizations: {', '.join(viz_types)}")
        
        created_files = {}
        
        # Create each requested visualization
        for viz_type in viz_types:
            if viz_type in self.visualization_types:
                try:
                    viz_function = self.visualization_types[viz_type]
                    output_path = viz_function(results, show_plots)
                    
                    if output_path:
                        created_files[viz_type] = output_path
                        logger.info(f"Created visualization: {viz_type} -> {output_path}")
                    else:
                        logger.warning(f"Visualization {viz_type} did not produce an output file")
                
                except Exception as e:
                    logger.error(f"Error creating visualization {viz_type}: {e}")
            else:
                logger.warning(f"Unknown visualization type: {viz_type}")
        
        # Close all plots to free memory
        plt.close('all')
        
        return created_files
    
    def create_custom_visualization(self,
                                 data: Any,
                                 visualization_func: Callable,
                                 output_filename: str,
                                 show_plot: bool = False) -> Optional[Path]:
        """
        Create a custom visualization using a provided function.
        
        Args:
            data: Data to visualize
            visualization_func: Function that creates the visualization
            output_filename: Filename for the output file
            show_plot: Whether to display the plot interactively
            
        Returns:
            Path to the saved visualization or None if saving failed
        """
        try:
            # Call the visualization function
            fig = visualization_func(data)
            
            # Save the figure if output directory is set
            output_path = None
            if self.output_dir:
                output_path = self.output_dir / output_filename
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
            
            # Show or close the plot
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating custom visualization: {e}")
            return None
    
    def _create_accuracy_comparison(self,
                                  results: Union[Dict[str, Any], ExperimentResult],
                                  show_plots: bool = False) -> Optional[Path]:
        """
        Create accuracy comparison visualization across prompts and fields.
        
        Args:
            results: Extraction results
            show_plots: Whether to display plots interactively
            
        Returns:
            Path to the saved visualization or None if saving failed
        """
        # Prepare data for visualization
        chart_data = []
        
        # Extract data from results based on type
        if isinstance(results, ExperimentResult):
            for field, performances in results.field_results.items():
                for perf in performances:
                    chart_data.append({
                        'Field': field,
                        'Prompt': perf.prompt_name,
                        'Accuracy': perf.accuracy * 100,  # Convert to percentage
                        'Success Count': perf.successful_extractions,
                        'Total Items': perf.total_items
                    })
        elif isinstance(results, dict):
            # Handle dictionary results (from analysis stage)
            if 'cross_field_analysis' in results:
                for field, field_data in results.get('cross_field_analysis', {}).get('field_performance', {}).items():
                    best_prompt = field_data.get('best_prompt')
                    success_rate = field_data.get('best_success_rate', 0)
                    
                    if best_prompt:
                        chart_data.append({
                            'Field': field,
                            'Prompt': best_prompt,
                            'Accuracy': success_rate * 100,
                            'Success Count': 0,  # Not available in this format
                            'Total Items': 0     # Not available in this format
                        })
            
            # Also check detailed_results format
            elif 'field_results' in results:
                for field, field_data in results.get('field_results', {}).items():
                    for perf in field_data:
                        chart_data.append({
                            'Field': field,
                            'Prompt': perf.get('prompt_name', 'Unknown'),
                            'Accuracy': perf.get('accuracy', 0) * 100,
                            'Success Count': perf.get('successful_extractions', 0),
                            'Total Items': perf.get('total_items', 0)
                        })
        
        if not chart_data:
            logger.warning("No data available for accuracy comparison visualization")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(chart_data)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Prompt', y='Accuracy', hue='Field', data=df, ax=ax)
        
        # Enhance visualization
        ax.set_title('Extraction Accuracy by Prompt and Field', fontsize=16)
        ax.set_xlabel('Prompt', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')
        
        plt.tight_layout()
        
        # Save if output directory provided
        output_path = None
        if self.output_dir:
            output_path = self.output_dir / f"accuracy_comparison_{self.experiment_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return output_path
    
    def _create_processing_time_comparison(self,
                                         results: Union[Dict[str, Any], ExperimentResult],
                                         show_plots: bool = False) -> Optional[Path]:
        """
        Create processing time comparison visualization.
        
        Args:
            results: Extraction results
            show_plots: Whether to display plots interactively
            
        Returns:
            Path to the saved visualization or None if saving failed
        """
        # Prepare data for visualization
        chart_data = []
        
        # Extract data from results based on type
        if isinstance(results, ExperimentResult):
            for field, performances in results.field_results.items():
                for perf in performances:
                    chart_data.append({
                        'Field': field,
                        'Prompt': perf.prompt_name,
                        'Avg Processing Time (s)': perf.avg_processing_time,
                        'Total Items': perf.total_items
                    })
        elif isinstance(results, dict):
            # Handle dictionary results (from analysis stage)
            # This would need implementation based on analysis stage output format
            pass
        
        if not chart_data:
            logger.warning("No data available for processing time comparison visualization")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(chart_data)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Prompt', y='Avg Processing Time (s)', hue='Field', data=df, ax=ax)
        
        # Enhance visualization
        ax.set_title('Average Processing Time by Prompt and Field', fontsize=16)
        ax.set_xlabel('Prompt', fontsize=14)
        ax.set_ylabel('Processing Time (seconds)', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f s')
        
        plt.tight_layout()
        
        # Save if output directory provided
        output_path = None
        if self.output_dir:
            output_path = self.output_dir / f"processing_time_{self.experiment_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return output_path
    
    def _create_error_analysis(self,
                             results: Union[Dict[str, Any], ExperimentResult],
                             show_plots: bool = False) -> Optional[Path]:
        """
        Create error analysis visualization.
        
        Args:
            results: Extraction results
            show_plots: Whether to display plots interactively
            
        Returns:
            Path to the saved visualization or None if saving failed
        """
        # Initialize error data
        error_data = {}
        
        # Extract data from results based on type
        if isinstance(results, ExperimentResult):
            # Aggregate error data across all results
            for field, performances in results.field_results.items():
                for perf in performances:
                    for result in perf.results:
                        if result.status != ExtractionStatus.SUCCESS:
                            if result.status not in error_data:
                                error_data[result.status] = {
                                    'count': 0,
                                    'by_field': {},
                                    'by_prompt': {}
                                }
                            
                            # Increment counts
                            error_data[result.status]['count'] += 1
                            
                            # By field
                            if field not in error_data[result.status]['by_field']:
                                error_data[result.status]['by_field'][field] = 0
                            error_data[result.status]['by_field'][field] += 1
                            
                            # By prompt
                            if perf.prompt_name not in error_data[result.status]['by_prompt']:
                                error_data[result.status]['by_prompt'][perf.prompt_name] = 0
                            error_data[result.status]['by_prompt'][perf.prompt_name] += 1
        
        elif isinstance(results, dict):
            # Handle dictionary results (from error analysis)
            error_analysis = results.get('error_analysis', {})
            
            # Process field-specific errors
            field_errors = error_analysis.get('field_specific_errors', {})
            for field, field_data in field_errors.items():
                error_status = "ERROR"  # Default status name
                
                if error_status not in error_data:
                    error_data[error_status] = {
                        'count': 0,
                        'by_field': {},
                        'by_prompt': {}
                    }
                
                # Get error count
                error_count = field_data.get('top_error_count', 0)
                
                # Add to counts
                error_data[error_status]['count'] += error_count
                
                # By field
                if field not in error_data[error_status]['by_field']:
                    error_data[error_status]['by_field'][field] = 0
                error_data[error_status]['by_field'][field] += error_count
        
        if not error_data:
            logger.warning("No error data available for error analysis visualization")
            return None
        
        # Prepare data for visualization
        status_counts = [(str(status), data['count']) for status, data in error_data.items()]
        status_names, count_values = zip(*status_counts)
        
        # Create the plot
        fig = plt.figure(figsize=(12, 8))
        
        # Subplot 1: Error counts by status
        ax1 = fig.add_subplot(1, 2, 1)
        bars = ax1.bar(status_names, count_values, color='#ff7f0e')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        ax1.set_title('Error Counts by Status Type', fontsize=14)
        ax1.set_ylabel('Count', fontsize=12)
        plt.sca(ax1)
        plt.xticks(rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Subplot 2: Error distribution by field
        ax2 = fig.add_subplot(1, 2, 2)
        
        # Prepare data for field distribution
        field_data = {}
        for status, data in error_data.items():
            for field, count in data['by_field'].items():
                if field not in field_data:
                    field_data[field] = {}
                field_data[field][str(status)] = count
        
        # Check if we have field data
        if field_data:
            # Convert to DataFrame for visualization
            field_df = pd.DataFrame(field_data).fillna(0).T
            
            # Create stacked bar chart
            field_df.plot(kind='bar', stacked=True, ax=ax2)
            
            ax2.set_title('Error Distribution by Field', fontsize=14)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_xlabel('Field', fontsize=12)
            plt.sca(ax2)
            plt.xticks(rotation=45, ha='right')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            ax2.legend(title='Error Type')
        
        plt.tight_layout()
        
        # Save if output directory provided
        output_path = None
        if self.output_dir:
            output_path = self.output_dir / f"error_analysis_{self.experiment_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return output_path
    
    def _create_prompt_radar(self,
                           results: Dict[str, Any],
                           show_plots: bool = False) -> Optional[Path]:
        """
        Create radar chart showing prompt consistency across fields.
        
        Args:
            results: Extraction results (analysis results expected)
            show_plots: Whether to display plots interactively
            
        Returns:
            Path to the saved visualization or None if saving failed
        """
        # Get prompt consistency data
        prompt_consistency = results.get('prompt_analysis', {}).get('prompt_consistency', {})
        if not prompt_consistency:
            logger.warning("No prompt consistency data available for radar visualization")
            return None
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Set up radar chart
        labels = list(prompt_consistency.keys())
        values = [prompt_consistency[l] for l in labels]
        num_vars = len(labels)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Make the plot circular
        values += values[:1]
        angles += angles[:1]
        labels += labels[:1]
        
        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        
        # Set labels
        ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
        
        # Set title
        plt.title('Prompt Consistency Across Fields', size=15)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save figure
        output_path = None
        if self.output_dir:
            output_path = self.output_dir / f"prompt_consistency_radar_{self.experiment_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return output_path
    
    def _create_field_performance(self,
                                results: Dict[str, Any],
                                show_plots: bool = False) -> Optional[Path]:
        """
        Create visualization of field performance comparison.
        
        Args:
            results: Extraction results (analysis results expected)
            show_plots: Whether to display plots interactively
            
        Returns:
            Path to the saved visualization or None if saving failed
        """
        # Get field performance data
        field_performance = results.get('cross_field_analysis', {}).get('field_performance', {})
        if not field_performance:
            logger.warning("No field performance data available for visualization")
            return None
        
        # Prepare data for chart
        fields = []
        success_rates = []
        
        for field, field_data in field_performance.items():
            fields.append(field)
            success_rates.append(field_data.get('best_success_rate', 0) * 100)  # Convert to percentage
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(fields, success_rates, color='skyblue')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # Customize plot
        ax.set_title('Performance by Field (Best Prompt)', fontsize=16)
        ax.set_xlabel('Field', fontsize=14)
        ax.set_ylabel('Success Rate (%)', fontsize=14)
        ax.set_ylim(0, max(success_rates) * 1.1)  # Add some headroom
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure
        output_path = None
        if self.output_dir:
            output_path = self.output_dir / f"field_performance_{self.experiment_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return output_path
    
    def _create_error_distribution(self,
                                 results: Dict[str, Any],
                                 show_plots: bool = False) -> Optional[Path]:
        """
        Create visualization of error distribution by field and prompt.
        
        Args:
            results: Extraction results (analysis results expected)
            show_plots: Whether to display plots interactively
            
        Returns:
            Path to the saved visualization or None if saving failed
        """
        # Get error data
        field_errors = results.get('error_analysis', {}).get('field_specific_errors', {})
        prompt_errors = results.get('error_analysis', {}).get('prompt_specific_errors', {})
        
        if not field_errors and not prompt_errors:
            logger.warning("No error distribution data available for visualization")
            return None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot field errors
        if field_errors:
            fields = list(field_errors.keys())
            error_rates = [field_errors[f]['error_rate'] * 100 for f in fields]  # Convert to percentage
            
            ax1.bar(fields, error_rates, color='coral')
            ax1.set_title('Error Rates by Field')
            ax1.set_ylabel('Error Rate (%)')
            ax1.set_ylim(0, max(error_rates) * 1.2)
            
            # Add value labels
            for i, v in enumerate(error_rates):
                ax1.text(i, v + 1, f'{v:.1f}%', ha='center')
            
            plt.sca(ax1)
            plt.xticks(rotation=45, ha='right')
        
        # Plot prompt errors
        if prompt_errors:
            prompts = list(prompt_errors.keys())
            error_rates = [prompt_errors[p]['error_rate'] * 100 for p in prompts]  # Convert to percentage
            
            ax2.bar(prompts, error_rates, color='skyblue')
            ax2.set_title('Error Rates by Prompt')
            ax2.set_ylabel('Error Rate (%)')
            ax2.set_ylim(0, max(error_rates) * 1.2)
            
            # Add value labels
            for i, v in enumerate(error_rates):
                ax2.text(i, v + 1, f'{v:.1f}%', ha='center')
            
            plt.sca(ax2)
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure
        output_path = None
        if self.output_dir:
            output_path = self.output_dir / f"error_distribution_{self.experiment_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return output_path
    
    def _create_performance_insights(self,
                                   results: Dict[str, Any],
                                   show_plots: bool = False) -> Optional[Path]:
        """
        Create visualization of performance insights.
        
        Args:
            results: Extraction results (analysis results expected)
            show_plots: Whether to display plots interactively
            
        Returns:
            Path to the saved visualization or None if saving failed
        """
        # Get insights
        insights = results.get('insights', [])
        
        if not insights:
            logger.warning("No insights available for visualization")
            return None
        
        # Create figure with appropriate height for number of insights
        fig, ax = plt.subplots(figsize=(12, len(insights) * 0.8 + 2))
        
        # Group insights by type
        insight_types = {}
        for insight in insights:
            insight_type = insight.get('type', 'other')
            if insight_type not in insight_types:
                insight_types[insight_type] = []
            insight_types[insight_type].append(insight)
        
        # Colors for each insight type
        colors = {
            'field_performance': 'lightgreen',
            'prompt_effectiveness': 'skyblue',
            'prompt_consistency': 'lightblue',
            'error_rate': 'coral',
            'common_error': 'salmon',
            'other': 'lightgray'
        }
        
        # Plot insights
        y_pos = 0
        labels = []
        colors_list = []
        
        for insight_type, type_insights in insight_types.items():
            for insight in type_insights:
                labels.append(insight.get('insight', ''))
                colors_list.append(colors.get(insight_type, 'lightgray'))
                y_pos += 1
        
        # Plot horizontal bars
        y_positions = range(len(labels))
        ax.barh(y_positions, [1] * len(labels), align='center', color=colors_list, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_title('Performance Insights')
        
        # Remove x-axis
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Add legend
        legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors.values()]
        legend_labels = list(colors.keys())
        ax.legend(legend_handles, legend_labels, loc='lower right')
        
        plt.tight_layout()
        
        # Save figure
        output_path = None
        if self.output_dir:
            output_path = self.output_dir / f"performance_insights_{self.experiment_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return output_path


# Create singleton instance
_visualization_service = None

def get_visualization_service(output_dir: Optional[Union[str, Path]] = None,
                           experiment_name: Optional[str] = None,
                           style: str = "default") -> VisualizationService:
    """
    Get the visualization service singleton.
    
    Args:
        output_dir: Base directory for storing visualizations
        experiment_name: Name of the current experiment
        style: Matplotlib style to use for visualizations
        
    Returns:
        VisualizationService instance
    """
    global _visualization_service
    
    if _visualization_service is None:
        _visualization_service = VisualizationService(
            output_dir=output_dir,
            experiment_name=experiment_name,
            style=style
        )
    
    return _visualization_service


def create_visualizations(
    results: Union[Dict[str, Any], ExperimentResult],
    output_dir: Optional[Union[str, Path]] = None,
    show_plots: bool = False,
    viz_types: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Create standard visualizations for extraction results.
    
    This is a convenience function that uses the VisualizationService.
    
    Args:
        results: Either an ExperimentResult object or a dictionary of results
        output_dir: Directory to save visualizations (None to skip saving)
        show_plots: Whether to display plots interactively
        viz_types: Types of visualizations to create (None for default set)
        
    Returns:
        Dictionary mapping visualization names to saved file paths
    """
    # Extract experiment name if available
    experiment_name = None
    if isinstance(results, ExperimentResult):
        experiment_name = results.experiment_name
    elif isinstance(results, dict):
        experiment_name = results.get('experiment_name', 'experiment')
    
    # Get visualization service
    viz_service = get_visualization_service(
        output_dir=output_dir,
        experiment_name=experiment_name
    )
    
    # If not specified, use default visualization types
    if viz_types is None:
        viz_types = ["accuracy_comparison", "processing_time", "error_analysis"]
    
    # Create visualizations
    return viz_service.create_visualizations(
        results=results,
        viz_types=viz_types,
        show_plots=show_plots
    )


def create_model_comparison(
    result_files: List[str],
    output_dir: Optional[Union[str, Path]] = None,
    show_plots: bool = False
) -> Dict[str, Path]:
    """
    Create visualizations comparing results across different models.
    
    Args:
        result_files: List of result files to compare
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots interactively
        
    Returns:
        Dictionary mapping visualization names to saved file paths
    """
    visualizations = {}
    
    try:
        # Load results from files
        results = []
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    import json
                    result = json.load(f)
                results.append(result)
            except Exception as e:
                with open(file_path, 'r') as f:
                    import json
                    result = json.load(f)
                results.append(result)
            except Exception as e:
                logger.error(f"Error loading result file {file_path}: {e}")
        
        if not results:
            logger.warning("No valid result files found for comparison")
            return visualizations
        
        # Extract experiment names
        model_names = []
        for result in results:
            model_name = result.get('model_name', 'Unknown')
            model_names.append(model_name)
        
        # Create visualization service
        viz_service = get_visualization_service(
            output_dir=output_dir,
            experiment_name="model_comparison"
        )
        
        # Create accuracy comparison
        accuracy_viz = _create_accuracy_comparison_across_models(results, model_names)
        if accuracy_viz and output_dir:
            output_path = Path(output_dir) / "model_accuracy_comparison.png"
            accuracy_viz.savefig(output_path, dpi=300, bbox_inches='tight')
            visualizations['model_accuracy_comparison'] = output_path
            
            if show_plots:
                plt.figure(accuracy_viz.number)
                plt.show()
            else:
                plt.close(accuracy_viz)
        
        # Create processing time comparison
        time_viz = _create_processing_time_across_models(results, model_names)
        if time_viz and output_dir:
            output_path = Path(output_dir) / "model_processing_time_comparison.png"
            time_viz.savefig(output_path, dpi=300, bbox_inches='tight')
            visualizations['model_processing_time_comparison'] = output_path
            
            if show_plots:
                plt.figure(time_viz.number)
                plt.show()
            else:
                plt.close(time_viz)
        
        # Create memory usage comparison
        memory_viz = _create_memory_usage_across_models(results, model_names)
        if memory_viz and output_dir:
            output_path = Path(output_dir) / "model_memory_usage_comparison.png"
            memory_viz.savefig(output_path, dpi=300, bbox_inches='tight')
            visualizations['model_memory_usage_comparison'] = output_path
            
            if show_plots:
                plt.figure(memory_viz.number)
                plt.show()
            else:
                plt.close(memory_viz)
        
        # Close all remaining plots
        plt.close('all')
        
    except Exception as e:
        logger.error(f"Error creating model comparison: {e}")
    
    return visualizations


def _create_accuracy_comparison_across_models(
    results: List[Dict[str, Any]],
    model_names: List[str]
) -> Optional[plt.Figure]:
    """
    Create accuracy comparison across different models.
    
    Args:
        results: List of result dictionaries
        model_names: List of model names
        
    Returns:
        Matplotlib figure or None if creation failed
    """
    try:
        # Extract accuracy data
        accuracies = []
        for result in results:
            overall_accuracy = result.get('overall_accuracy', 0)
            accuracies.append(overall_accuracy * 100)  # Convert to percentage
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, accuracies, color='skyblue')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        # Customize plot
        ax.set_title('Accuracy Comparison Across Models', fontsize=16)
        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel('Overall Accuracy (%)', fontsize=14)
        ax.set_ylim(0, max(accuracies) * 1.1)  # Add some headroom
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating accuracy comparison across models: {e}")
        return None


def _create_processing_time_across_models(
    results: List[Dict[str, Any]],
    model_names: List[str]
) -> Optional[plt.Figure]:
    """
    Create processing time comparison across different models.
    
    Args:
        results: List of result dictionaries
        model_names: List of model names
        
    Returns:
        Matplotlib figure or None if creation failed
    """
    try:
        # Extract processing time data
        processing_times = []
        for result in results:
            # Look for processing time in different formats
            time = result.get('total_execution_time')
            if time is None:
                for field_results in result.get('field_results', {}).values():
                    for perf in field_results:
                        if 'avg_processing_time' in perf:
                            time = perf['avg_processing_time']
                            break
            
            processing_times.append(time or 0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, processing_times, color='lightgreen')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.2f}s', ha='center', va='bottom')
        
        # Customize plot
        ax.set_title('Processing Time Comparison Across Models', fontsize=16)
        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel('Processing Time (seconds)', fontsize=14)
        ax.set_ylim(0, max(processing_times) * 1.1)  # Add some headroom
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating processing time comparison across models: {e}")
        return None


def _create_memory_usage_across_models(
    results: List[Dict[str, Any]],
    model_names: List[str]
) -> Optional[plt.Figure]:
    """
    Create memory usage comparison across different models.
    
    Args:
        results: List of result dictionaries
        model_names: List of model names
        
    Returns:
        Matplotlib figure or None if creation failed
    """
    try:
        # Extract memory usage data
        memory_usages = []
        for result in results:
            # Look for memory info in different formats
            memory_usage = 0
            
            # Try to find in model_loading section
            model_loading = result.get('model_loading', {})
            memory_info = model_loading.get('memory_info', {})
            if 'allocated_memory_gb' in memory_info:
                memory_usage = memory_info['allocated_memory_gb']
            
            # Try to find in quantization_metadata
            if memory_usage == 0:
                quant_metadata = model_loading.get('quantization_metadata', {})
                if 'memory_used_gb' in quant_metadata:
                    memory_usage = quant_metadata['memory_used_gb']
            
            memory_usages.append(memory_usage)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, memory_usages, color='coral')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}GB', ha='center', va='bottom')
        
        # Customize plot
        ax.set_title('Memory Usage Comparison Across Models', fontsize=16)
        ax.set_xlabel('Model', fontsize=14)
        ax.set_ylabel('GPU Memory Usage (GB)', fontsize=14)
        ax.set_ylim(0, max(memory_usages) * 1.1)  # Add some headroom
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating memory usage comparison across models: {e}")
        return None


class EnhancedResultVisualizer:
    """
    Enhanced visualization class for creating interactive and dynamic visualizations.
    
    This class provides more advanced visualization capabilities that build on
    the VisualizationService, including interactive elements where appropriate.
    """
    
    def __init__(self, 
                output_dir: Optional[Union[str, Path]] = None,
                experiment_name: Optional[str] = None):
        """
        Initialize the enhanced visualizer.
        
        Args:
            output_dir: Base directory for storing visualizations
            experiment_name: Name of the current experiment
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.experiment_name = experiment_name or "experiment"
        
        # Set up visualization service
        self.viz_service = get_visualization_service(
            output_dir=output_dir,
            experiment_name=experiment_name
        )
    
    def create_interactive_dashboard(self, 
                                  results: Union[Dict[str, Any], ExperimentResult],
                                  output_filename: str = "dashboard.html") -> Optional[str]:
        """
        Create an interactive dashboard with visualization results.
        
        Args:
            results: Extraction results
            output_filename: Output HTML filename
            
        Returns:
            Path to the HTML file or None if creation failed
        """
        # This is a placeholder for creating interactive dashboards
        # Implementation would vary based on the visualization library chosen
        # (e.g., Plotly Dash, Bokeh, or similar)
        
        logger.warning("Interactive dashboard creation not yet implemented")
        return None
    
    def create_comparative_analysis(self,
                                 field_results: Dict[str, Dict[str, Any]],
                                 output_dirname: str = "comparative_analysis") -> Dict[str, str]:
        """
        Create a comprehensive comparative analysis across fields and prompts.
        
        Args:
            field_results: Results by field and prompt
            output_dirname: Output directory name
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        # Create output directory
        if self.output_dir:
            output_dir = self.output_dir / output_dirname
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            return {}
        
        visualizations = {}
        
        # This is a placeholder for creating comparative analysis
        # Implementation would vary based on requirements
        
        logger.warning("Comparative analysis creation not yet implemented")
        return visualizations


class NotebookVisualizations:
    """
    Specialized visualization capabilities designed specifically for Jupyter notebook environments.
    
    This class provides notebook-optimized visualizations with interactive elements, 
    simplified APIs, and enhanced display capabilities for Jupyter environments.
    """
    
    def __init__(self, 
                results_collector=None,
                style: str = "notebook",
                figsize_multiplier: float = 1.2):
        """
        Initialize notebook visualization tools.
        
        Args:
            results_collector: Optional ResultsCollector instance for data access
            style: Matplotlib style to use (defaults to notebook-optimized)
            figsize_multiplier: Multiplier for figure sizes to adapt to notebook display
        """
        self.results_collector = results_collector
        self.figsize_multiplier = figsize_multiplier
        self._set_notebook_style(style)
        self._check_notebook_environment()
        
    def _check_notebook_environment(self):
        """Check if running in a Jupyter notebook and configure accordingly."""
        try:
            # Try to get the IPython shell
            from IPython import get_ipython
            
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                # Jupyter notebook or qtconsole
                from IPython.display import display, HTML
                # Enable matplotlib inline mode
                get_ipython().run_line_magic('matplotlib', 'inline')
                # Try to enable widgets
                try:
                    get_ipython().run_line_magic('load_ext', 'ipywidgets')
                except:
                    pass
                self.is_notebook = True
                logger.info("Running in Jupyter notebook environment")
            else:
                self.is_notebook = False
        except:
            self.is_notebook = False
            logger.info("Not running in Jupyter notebook environment")
    
    def _set_notebook_style(self, style: str):
        """Set visualization style optimized for notebooks."""
        try:
            # Check if we have seaborn for better styling
            import seaborn as sns
            
            if style == "notebook":
                # Custom style optimized for notebooks
                plt.style.use('seaborn-v0_8-colorblind')
                sns.set_context("notebook", font_scale=1.2)
                
                # Make plots look better in both light and dark themes
                plt.rcParams['figure.facecolor'] = 'white'
                plt.rcParams['axes.facecolor'] = 'white'
                plt.rcParams['savefig.facecolor'] = 'white'
                plt.rcParams['figure.figsize'] = [
                    plt.rcParams['figure.figsize'][0] * self.figsize_multiplier,
                    plt.rcParams['figure.figsize'][1] * self.figsize_multiplier
                ]
            else:
                plt.style.use(style)
                
        except Exception as e:
            logger.warning(f"Could not set notebook style: {e}")
    
    def accuracy_comparison(self, 
                          results=None, 
                          interactive=True, 
                          show_confidence_intervals=True):
        """
        Create an interactive accuracy comparison visualization.
        
        Args:
            results: Results data (uses stored results_collector if None)
            interactive: Whether to add interactive elements
            show_confidence_intervals: Whether to show confidence intervals
            
        Returns:
            Visualization object (displayed inline in notebook)
        """
        # Get results from collector if not provided
        results_data = results if results is not None else (
            self.results_collector.get_results() if self.results_collector else None)
        
        if results_data is None:
            logger.error("No results data available for visualization")
            return None
            
        if interactive and self.is_notebook:
            return self._create_interactive_accuracy_comparison(results_data, show_confidence_intervals)
        else:
            return self._create_static_accuracy_comparison(results_data, show_confidence_intervals)
    
    def _create_interactive_accuracy_comparison(self, results, show_confidence_intervals=True):
        """Create an interactive accuracy comparison with ipywidgets."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
            
            # Prepare data
            chart_data = self._prepare_accuracy_data(results)
            
            if not chart_data:
                logger.warning("No data available for accuracy comparison")
                return None
                
            df = pd.DataFrame(chart_data)
            
            # Create widgets for interaction
            field_selector = widgets.SelectMultiple(
                options=sorted(df['Field'].unique()),
                value=sorted(df['Field'].unique()),
                description='Fields:',
                disabled=False
            )
            
            prompt_selector = widgets.SelectMultiple(
                options=sorted(df['Prompt'].unique()),
                value=sorted(df['Prompt'].unique()),
                description='Prompts:',
                disabled=False
            )
            
            sort_selector = widgets.Dropdown(
                options=['Field', 'Prompt', 'Accuracy'],
                value='Accuracy',
                description='Sort by:',
                disabled=False,
            )
            
            ascending_checkbox = widgets.Checkbox(
                value=False,
                description='Ascending',
                disabled=False
            )
            
            plot_type = widgets.RadioButtons(
                options=['bar', 'heatmap'],
                value='bar',
                description='Plot type:',
                disabled=False
            )
            
            # Create output widget for the plot
            plot_output = widgets.Output()
            
            # Define update function
            def update_plot(*args):
                with plot_output:
                    # Clear previous plot
                    plot_output.clear_output(wait=True)
                    
                    # Filter data based on widget values
                    filtered_df = df[
                        df['Field'].isin(field_selector.value) & 
                        df['Prompt'].isin(prompt_selector.value)
                    ]
                    
                    if filtered_df.empty:
                        print("No data available with current filter settings")
                        return
                    
                    # Sort data
                    filtered_df = filtered_df.sort_values(
                        by=sort_selector.value, 
                        ascending=ascending_checkbox.value
                    )
                    
                    # Create appropriate plot
                    fig, ax = plt.subplots(figsize=(10 * self.figsize_multiplier, 
                                                    6 * self.figsize_multiplier))
                    
                    if plot_type.value == 'bar':
                        # Bar chart
                        if sort_selector.value == 'Field':
                            sns.barplot(x='Field', y='Accuracy', hue='Prompt', data=filtered_df, ax=ax)
                            plt.xticks(rotation=45, ha='right')
                        else:
                            sns.barplot(x='Prompt', y='Accuracy', hue='Field', data=filtered_df, ax=ax)
                            plt.xticks(rotation=45, ha='right')
                            
                        # Add confidence intervals if requested
                        if show_confidence_intervals and 'Confidence_Low' in filtered_df.columns:
                            for i, row in filtered_df.iterrows():
                                ax.plot([i, i], 
                                        [row['Confidence_Low'], row['Confidence_High']], 
                                        color='black', alpha=0.6)
                    else:
                        # Heatmap
                        pivot_df = filtered_df.pivot_table(
                            index='Field', 
                            columns='Prompt', 
                            values='Accuracy'
                        )
                        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap="YlGnBu", ax=ax)
                    
                    # Enhance visualization
                    ax.set_title('Extraction Accuracy Comparison', fontsize=16)
                    ax.set_ylabel('Accuracy (%)', fontsize=14)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    plt.tight_layout()
                    plt.show()
            
            # Connect widgets to update function
            field_selector.observe(update_plot, names='value')
            prompt_selector.observe(update_plot, names='value')
            sort_selector.observe(update_plot, names='value')
            ascending_checkbox.observe(update_plot, names='value')
            plot_type.observe(update_plot, names='value')
            
            # Create layout
            controls = widgets.VBox([
                widgets.HBox([field_selector, prompt_selector]),
                widgets.HBox([sort_selector, ascending_checkbox, plot_type])
            ])
            
            # Display widgets and initial plot
            display(widgets.VBox([controls, plot_output]))
            update_plot()
            
            return plot_output
            
        except ImportError as e:
            logger.warning(f"Interactive visualization requires ipywidgets: {e}")
            # Fall back to static visualization
            return self._create_static_accuracy_comparison(results, show_confidence_intervals)
    
    def _create_static_accuracy_comparison(self, results, show_confidence_intervals=True):
        """Create a static accuracy comparison visualization."""
        # Prepare data
        chart_data = self._prepare_accuracy_data(results)
        
        if not chart_data:
            logger.warning("No data available for accuracy comparison")
            return None
            
        df = pd.DataFrame(chart_data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12 * self.figsize_multiplier, 8 * self.figsize_multiplier))
        sns.barplot(x='Prompt', y='Accuracy', hue='Field', data=df, ax=ax)
        
        # Add confidence intervals if available and requested
        if show_confidence_intervals and 'Confidence_Low' in df.columns:
            for i, row in df.iterrows():
                # Find the position of this bar
                bars = ax.patches
                bar_positions = np.linspace(0, len(df['Prompt'].unique()) - 1, len(bars))
                
                # Plot error bars
                ax.plot([bar_positions[i], bar_positions[i]], 
                        [row['Confidence_Low'], row['Confidence_High']], 
                        color='black', alpha=0.6)
        
        # Enhance visualization
        ax.set_title('Extraction Accuracy by Prompt and Field', fontsize=16)
        ax.set_xlabel('Prompt', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _prepare_accuracy_data(self, results):
        """Prepare data for accuracy visualizations from various result formats."""
        chart_data = []
        
        # Extract data from results based on type
        if isinstance(results, ExperimentResult):
            for field, performances in results.field_results.items():
                for perf in performances:
                    data_point = {
                        'Field': field,
                        'Prompt': perf.prompt_name,
                        'Accuracy': perf.accuracy * 100,  # Convert to percentage
                        'Success Count': perf.successful_extractions,
                        'Total Items': perf.total_items
                    }
                    
                    # Add confidence intervals if available
                    if hasattr(perf, 'confidence_interval'):
                        ci = perf.confidence_interval
                        if ci and len(ci) == 2:
                            data_point['Confidence_Low'] = ci[0] * 100
                            data_point['Confidence_High'] = ci[1] * 100
                    
                    chart_data.append(data_point)
                    
        elif isinstance(results, dict):
            # Handle dictionary results (from analysis stage)
            if 'cross_field_analysis' in results:
                for field, field_data in results.get('cross_field_analysis', {}).get('field_performance', {}).items():
                    best_prompt = field_data.get('best_prompt')
                    success_rate = field_data.get('best_success_rate', 0)
                    
                    data_point = {
                        'Field': field,
                        'Prompt': best_prompt,
                        'Accuracy': success_rate * 100,
                        'Success Count': 0,  # Not available in this format
                        'Total Items': 0     # Not available in this format
                    }
                    
                    # Add confidence intervals if available
                    if 'confidence_interval' in field_data:
                        ci = field_data.get('confidence_interval')
                        if ci and len(ci) == 2:
                            data_point['Confidence_Low'] = ci[0] * 100
                            data_point['Confidence_High'] = ci[1] * 100
                    
                    chart_data.append(data_point)
            
            # Also check detailed_results format
            elif 'field_results' in results:
                for field, field_data in results.get('field_results', {}).items():
                    for perf in field_data:
                        data_point = {
                            'Field': field,
                            'Prompt': perf.get('prompt_name', 'Unknown'),
                            'Accuracy': perf.get('accuracy', 0) * 100,
                            'Success Count': perf.get('successful_extractions', 0),
                            'Total Items': perf.get('total_items', 0)
                        }
                        
                        # Add confidence intervals if available
                        if 'confidence_interval' in perf:
                            ci = perf.get('confidence_interval')
                            if ci and len(ci) == 2:
                                data_point['Confidence_Low'] = ci[0] * 100
                                data_point['Confidence_High'] = ci[1] * 100
                        
                        chart_data.append(data_point)
        
        return chart_data
    
    def create_dashboard(self, results=None, output_filename=None):
        """
        Create an interactive dashboard with multiple visualizations.
        
        Args:
            results: Results data (uses stored results_collector if None)
            output_filename: Optional filename to save the dashboard HTML
            
        Returns:
            Dashboard object (displayed inline in notebook)
        """
        # Get results from collector if not provided
        results_data = results if results is not None else (
            self.results_collector.get_results() if self.results_collector else None)
        
        if results_data is None:
            logger.error("No results data available for dashboard")
            return None
        
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import ipywidgets as widgets
            from IPython.display import display, HTML
            
            # Prepare data
            chart_data = self._prepare_accuracy_data(results_data)
            
            if not chart_data:
                logger.warning("No data available for dashboard")
                return None
                
            df = pd.DataFrame(chart_data)
            
            # Create dashboard with tabs for different visualizations
            tab_layout = widgets.Tab()
            
            # Tab 1: Accuracy Comparison
            accuracy_tab = widgets.Output()
            with accuracy_tab:
                # Create plotly figure
                fig = px.bar(df, x='Prompt', y='Accuracy', color='Field', 
                           barmode='group', title='Extraction Accuracy',
                           labels={'Accuracy': 'Accuracy (%)'})
                fig.update_layout(
                    xaxis_title='Prompt',
                    yaxis_title='Accuracy (%)',
                    legend_title='Field',
                    font=dict(size=14)
                )
                fig.show()
            
            # Tab 2: Heatmap View
            heatmap_tab = widgets.Output()
            with heatmap_tab:
                # Create pivot table for heatmap
                pivot_df = df.pivot_table(index='Field', columns='Prompt', values='Accuracy')
                
                # Create heatmap
                fig = px.imshow(pivot_df, text_auto='.1f', aspect='auto',
                              labels=dict(x='Prompt', y='Field', color='Accuracy (%)'),
                              title='Accuracy Heatmap')
                
                fig.update_layout(
                    xaxis_title='Prompt',
                    yaxis_title='Field',
                    font=dict(size=14)
                )
                fig.show()
            
            # Tab 3: Success Counts
            counts_tab = widgets.Output()
            with counts_tab:
                # Filter out rows with no success count data
                counts_df = df[df['Success Count'] > 0].copy()
                
                if not counts_df.empty:
                    # Create figure with two subplots
                    fig = make_subplots(rows=1, cols=2, 
                                      subplot_titles=('Successful Extractions', 'Success Rate'),
                                      specs=[[{"type": "bar"}, {"type": "bar"}]])
                    
                    # Add Success Count bars
                    for field in counts_df['Field'].unique():
                        field_df = counts_df[counts_df['Field'] == field]
                        fig.add_trace(
                            go.Bar(
                                x=field_df['Prompt'],
                                y=field_df['Success Count'],
                                name=field
                            ),
                            row=1, col=1
                        )
                    
                    # Add success rate bars
                    for field in counts_df['Field'].unique():
                        field_df = counts_df[counts_df['Field'] == field]
                        fig.add_trace(
                            go.Bar(
                                x=field_df['Prompt'],
                                y=field_df['Accuracy'],
                                name=field
                            ),
                            row=1, col=2
                        )
                    
                    fig.update_layout(
                        title_text='Extraction Performance Metrics',
                        height=500,
                        showlegend=True
                    )
                    
                    # Update y-axis labels
                    fig.update_yaxes(title_text='Count', row=1, col=1)
                    fig.update_yaxes(title_text='Accuracy (%)', row=1, col=2)
                    
                    fig.show()
                else:
                    display(HTML("<p>No success count data available</p>"))
            
            # Assemble the tabs
            tab_layout.children = [accuracy_tab, heatmap_tab, counts_tab]
            
            # Set tab titles
            tab_layout.set_title(0, 'Accuracy Comparison')
            tab_layout.set_title(1, 'Heatmap View')
            tab_layout.set_title(2, 'Success Metrics')
            
            # Display the dashboard
            display(tab_layout)
            
            # Save if filename provided
            if output_filename and df is not None:
                try:
                    # Create a standalone HTML dashboard
                    from plotly.offline import plot
                    import plotly.io as pio
                    
                    # Create figures for standalone HTML
                    accuracy_fig = px.bar(df, x='Prompt', y='Accuracy', color='Field', 
                                        barmode='group', title='Extraction Accuracy')
                    
                    pivot_df = df.pivot_table(index='Field', columns='Prompt', values='Accuracy')
                    heatmap_fig = px.imshow(pivot_df, text_auto='.1f', aspect='auto',
                                          title='Accuracy Heatmap')
                    
                    # Combine into one HTML file
                    with open(output_filename, 'w') as f:
                        f.write("<html><head><title>Extraction Results Dashboard</title></head><body>")
                        f.write("<h1>Extraction Results Dashboard</h1>")
                        f.write("<div style='margin-bottom: 30px;'>")
                        f.write(plot(accuracy_fig, include_plotlyjs='cdn', output_type='div'))
                        f.write("</div><div style='margin-bottom: 30px;'>")
                        f.write(plot(heatmap_fig, include_plotlyjs='cdn', output_type='div'))
                        f.write("</div></body></html>")
                    
                    logger.info(f"Dashboard saved to {output_filename}")
                    
                except Exception as e:
                    logger.error(f"Error saving dashboard: {e}")
            
            return tab_layout
            
        except ImportError as e:
            logger.warning(f"Dashboard creation requires plotly and ipywidgets: {e}")
            # Fall back to regular matplotlib visualization
            return self.accuracy_comparison(results_data, interactive=False)
    
    def export_all_plots(self, results=None, output_dir=None, format='png'):
        """
        Export all visualizations to files.
        
        Args:
            results: Results data (uses stored results_collector if None)
            output_dir: Directory to save files (created if doesn't exist)
            format: File format ('png', 'pdf', 'svg')
            
        Returns:
            Dictionary mapping visualization names to saved file paths
        """
        # Get results from collector if not provided
        results_data = results if results is not None else (
            self.results_collector.get_results() if self.results_collector else None)
        
        if results_data is None:
            logger.error("No results data available for exporting visualizations")
            return {}
        
        # Create output directory if needed
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use current directory
            output_dir = Path('.')
        
        # Dictionary to track saved files
        saved_files = {}
        
        # Create and save standard visualizations
        # 1. Accuracy comparison
        fig = self._create_static_accuracy_comparison(results_data)
        if fig:
            filename = output_dir / f"accuracy_comparison.{format}"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            saved_files["accuracy_comparison"] = filename
            plt.close(fig)
        
        # 2. Create dashboard as HTML
        if format == 'html' or format == 'all':
            try:
                dashboard_filename = output_dir / "dashboard.html"
                self.create_dashboard(results_data, output_filename=str(dashboard_filename))
                saved_files["dashboard"] = dashboard_filename
            except Exception as e:
                logger.error(f"Error saving dashboard: {e}")
        
        # Report on saved files
        if saved_files:
            logger.info(f"Exported {len(saved_files)} visualizations to {output_dir}")
            
            # If in notebook, display a summary
            if self.is_notebook:
                try:
                    from IPython.display import display, HTML
                    html = "<h3>Exported Visualizations</h3><ul>"
                    for name, path in saved_files.items():
                        html += f"<li><strong>{name}</strong>: {path}</li>"
                    html += "</ul>"
                    display(HTML(html))
                except:
                    pass
        else:
            logger.warning("No visualizations were exported")
        
        return saved_files


def get_notebook_visualizations(results_collector=None, style="notebook"):
    """
    Get a NotebookVisualizations instance for use in Jupyter notebooks.
    
    Args:
        results_collector: Optional ResultsCollector for data access
        style: Visualization style to use
        
    Returns:
        NotebookVisualizations instance
    """
    return NotebookVisualizations(results_collector=results_collector, style=style)

# For backward compatibility
EnhancedResultVisualizer = NotebookVisualizations
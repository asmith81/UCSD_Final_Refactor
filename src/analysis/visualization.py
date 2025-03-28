"""
Results Visualization

This module provides functions for visualizing extraction results and experiment
performance metrics.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Import the schema components to ensure type compatibility
from src.results.schema import ExperimentResult, PromptPerformance, IndividualExtractionResult, ExtractionStatus


def create_visualizations(
    results: Union[Dict[str, Any], ExperimentResult],
    output_dir: Optional[Union[str, Path]] = None,
    show_plots: bool = False
) -> Dict[str, Path]:
    """
    Create standard visualizations for extraction results.
    
    Args:
        results: Either an ExperimentResult object or a dictionary of results 
                (typically loaded from a JSON file)
        output_dir: Directory to save visualizations (None to skip saving)
        show_plots: Whether to display plots interactively
        
    Returns:
        Dictionary mapping visualization names to saved file paths
    """
    # Convert dictionary to ExperimentResult if needed
    if isinstance(results, dict) and not isinstance(results, ExperimentResult):
        # This would require implementation if you're passing raw dictionaries
        # For now, we'll assume you're passing proper ExperimentResult objects
        pass
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = {}
    
    # Create accuracy comparison chart across prompts and fields
    accuracy_chart_path = _create_accuracy_comparison(results, output_dir, show_plots)
    if accuracy_chart_path:
        created_files['accuracy_comparison'] = accuracy_chart_path
    
    # Create processing time comparison
    time_chart_path = _create_processing_time_comparison(results, output_dir, show_plots)
    if time_chart_path:
        created_files['processing_time'] = time_chart_path
    
    # Create error analysis visualization
    error_chart_path = _create_error_analysis(results, output_dir, show_plots)
    if error_chart_path:
        created_files['error_analysis'] = error_chart_path
    
    # Close all plots to free memory
    plt.close('all')
    
    return created_files


def _create_accuracy_comparison(
    results: ExperimentResult,
    output_dir: Optional[Path] = None,
    show_plots: bool = False
) -> Optional[Path]:
    """Create accuracy comparison visualization across prompts and fields."""
    # Prepare data for visualization
    chart_data = []
    
    for field, performances in results.field_results.items():
        for perf in performances:
            chart_data.append({
                'Field': field,
                'Prompt': perf.prompt_name,
                'Accuracy': perf.accuracy * 100,  # Convert to percentage
                'Success Count': perf.successful_extractions,
                'Total Items': perf.total_items
            })
    
    if not chart_data:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(chart_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Prompt', y='Accuracy', hue='Field', data=df)
    
    # Enhance visualization
    plt.title('Extraction Accuracy by Prompt and Field', fontsize=16)
    plt.xlabel('Prompt', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%')
    
    plt.tight_layout()
    
    # Save if output directory provided
    output_path = None
    if output_dir:
        output_path = output_dir / f"accuracy_comparison_{results.experiment_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return output_path


def _create_processing_time_comparison(
    results: ExperimentResult,
    output_dir: Optional[Path] = None,
    show_plots: bool = False
) -> Optional[Path]:
    """Create processing time comparison visualization."""
    # Prepare data for visualization
    chart_data = []
    
    for field, performances in results.field_results.items():
        for perf in performances:
            chart_data.append({
                'Field': field,
                'Prompt': perf.prompt_name,
                'Avg Processing Time (s)': perf.avg_processing_time,
                'Total Items': perf.total_items
            })
    
    if not chart_data:
        return None
    
    # Create DataFrame
    df = pd.DataFrame(chart_data)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Prompt', y='Avg Processing Time (s)', hue='Field', data=df)
    
    # Enhance visualization
    plt.title('Average Processing Time by Prompt and Field', fontsize=16)
    plt.xlabel('Prompt', fontsize=14)
    plt.ylabel('Processing Time (seconds)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f s')
    
    plt.tight_layout()
    
    # Save if output directory provided
    output_path = None
    if output_dir:
        output_path = output_dir / f"processing_time_{results.experiment_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return output_path


def _create_error_analysis(
    results: ExperimentResult,
    output_dir: Optional[Path] = None,
    show_plots: bool = False
) -> Optional[Path]:
    """Create error analysis visualization."""
    # Aggregate error data across all results
    error_data = {}
    
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
    
    if not error_data:
        return None
    
    # Prepare data for visualization
    status_counts = [(status.name, data['count']) for status, data in error_data.items()]
    status_names, count_values = zip(*status_counts)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Error counts by status
    plt.subplot(1, 2, 1)
    bars = plt.bar(status_names, count_values, color='#ff7f0e')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')
    
    plt.title('Error Counts by Status Type', fontsize=14)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Subplot 2: Error distribution by field
    plt.subplot(1, 2, 2)
    
    # Prepare data for field distribution
    field_data = {}
    for status, data in error_data.items():
        for field, count in data['by_field'].items():
            if field not in field_data:
                field_data[field] = {}
            field_data[field][status.name] = count
    
    # Convert to DataFrame for visualization
    field_df = pd.DataFrame(field_data).fillna(0).T
    
    # Create stacked bar chart
    field_df.plot(kind='bar', stacked=True, ax=plt.gca())
    
    plt.title('Error Distribution by Field', fontsize=14)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Field', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Error Type')
    
    plt.tight_layout()
    
    # Save if output directory provided
    output_path = None
    if output_dir:
        output_path = output_dir / f"error_analysis_{results.experiment_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return output_path


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
    # Implementation to be added based on your specific needs
    # This would load results from multiple files and create comparison charts
    
    return {}


# Additional functions for specific visualization types can be added here
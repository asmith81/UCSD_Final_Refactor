"""
Quantization Analysis Visualization

This module provides visualization capabilities for analyzing quantization strategies,
including memory usage, inference speed, accuracy trade-offs, and hardware compatibility.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Configure logging
logger = logging.getLogger(__name__)

class QuantizationVisualizer:
    """
    Visualizer for quantization analysis results.
    
    This class provides methods for creating various visualizations
    to analyze quantization strategy performance and trade-offs.
    """
    
    def __init__(
        self,
        results_dir: Optional[Union[str, Path]] = None,
        style: str = 'seaborn',
        dpi: int = 300
    ):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory to save visualization outputs
            style: Matplotlib style to use
            dpi: Dots per inch for saved figures
        """
        self.results_dir = Path(results_dir) if results_dir else Path('results/visualizations')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use(style)
        self.dpi = dpi
        
        # Set up color palette
        self.colors = sns.color_palette("husl", 8)
        
        # Configure matplotlib defaults
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_memory_usage(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
        save: bool = True
    ) -> Figure:
        """
        Create memory usage visualization.
        
        Args:
            data: Memory usage data from benchmark results
            title: Optional plot title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots()
        
        # Extract data
        strategies = []
        memory_values = []
        for item in data['memory_usage']:
            strategies.append(item['strategy'])
            memory_values.append(item['memory_gb'])
        
        # Create bar plot
        bars = ax.bar(strategies, memory_values, color=self.colors)
        
        # Customize plot
        ax.set_title(title or 'Memory Usage by Quantization Strategy')
        ax.set_xlabel('Quantization Strategy')
        ax.set_ylabel('Memory Usage (GB)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom'
            )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save:
            self._save_figure(fig, 'memory_usage')
        
        return fig
    
    def plot_inference_speed(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
        save: bool = True
    ) -> Figure:
        """
        Create inference speed visualization.
        
        Args:
            data: Inference speed data from benchmark results
            title: Optional plot title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots()
        
        # Extract data
        strategies = []
        times = []
        for item in data['inference_speed']:
            strategies.append(item['strategy'])
            times.append(item['time_per_image'])
        
        # Create bar plot
        bars = ax.bar(strategies, times, color=self.colors)
        
        # Customize plot
        ax.set_title(title or 'Inference Speed by Quantization Strategy')
        ax.set_xlabel('Quantization Strategy')
        ax.set_ylabel('Time per Image (seconds)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom'
            )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save:
            self._save_figure(fig, 'inference_speed')
        
        return fig
    
    def plot_accuracy_comparison(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
        save: bool = True
    ) -> Figure:
        """
        Create accuracy comparison visualization.
        
        Args:
            data: Accuracy data from benchmark results
            title: Optional plot title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots()
        
        # Extract data
        strategies = []
        accuracies = []
        for item in data['accuracy']:
            strategies.append(item['strategy'])
            accuracies.append(item['accuracy'])
        
        # Create bar plot
        bars = ax.bar(strategies, accuracies, color=self.colors)
        
        # Customize plot
        ax.set_title(title or 'Accuracy by Quantization Strategy')
        ax.set_xlabel('Quantization Strategy')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=45)
        
        # Set y-axis to percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.1%}',
                ha='center',
                va='bottom'
            )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save:
            self._save_figure(fig, 'accuracy_comparison')
        
        return fig
    
    def plot_efficiency_scores(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
        save: bool = True
    ) -> Figure:
        """
        Create efficiency scores visualization.
        
        Args:
            data: Efficiency score data from benchmark results
            title: Optional plot title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots()
        
        # Extract data
        strategies = []
        scores = []
        for item in data['efficiency_scores']:
            strategies.append(item['strategy'])
            scores.append(item['score'])
        
        # Create bar plot
        bars = ax.bar(strategies, scores, color=self.colors)
        
        # Customize plot
        ax.set_title(title or 'Efficiency Scores by Quantization Strategy')
        ax.set_xlabel('Quantization Strategy')
        ax.set_ylabel('Efficiency Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom'
            )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save:
            self._save_figure(fig, 'efficiency_scores')
        
        return fig
    
    def plot_trade_off_analysis(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
        save: bool = True
    ) -> Figure:
        """
        Create speed-accuracy trade-off visualization.
        
        Args:
            data: Combined data from benchmark results
            title: Optional plot title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots()
        
        # Extract data
        strategies = []
        times = []
        accuracies = []
        for item in data['inference_speed']:
            strategy = item['strategy']
            strategies.append(strategy)
            times.append(item['time_per_image'])
            
            # Find corresponding accuracy
            accuracy_item = next(
                (a for a in data['accuracy'] if a['strategy'] == strategy),
                None
            )
            accuracies.append(accuracy_item['accuracy'] if accuracy_item else 0.0)
        
        # Create scatter plot
        scatter = ax.scatter(times, accuracies, c=range(len(strategies)), cmap='viridis')
        
        # Add labels for each point
        for i, strategy in enumerate(strategies):
            ax.annotate(
                strategy,
                (times[i], accuracies[i]),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        # Customize plot
        ax.set_title(title or 'Speed-Accuracy Trade-off Analysis')
        ax.set_xlabel('Inference Time (seconds)')
        ax.set_ylabel('Accuracy')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save:
            self._save_figure(fig, 'trade_off_analysis')
        
        return fig
    
    def plot_hardware_compatibility(
        self,
        data: Dict[str, Any],
        title: Optional[str] = None,
        save: bool = True
    ) -> Figure:
        """
        Create hardware compatibility visualization.
        
        Args:
            data: Hardware compatibility data from benchmark results
            title: Optional plot title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots()
        
        # Extract data
        strategies = []
        compatible = []
        exceeds_memory = []
        for item in data['hardware_compatibility']:
            strategies.append(item['strategy'])
            compatible.append(1 if item['compatible'] else 0)
            exceeds_memory.append(1 if item['exceeds_memory'] else 0)
        
        # Create stacked bar plot
        x = np.arange(len(strategies))
        width = 0.35
        
        ax.bar(x, compatible, width, label='Compatible', color='green')
        ax.bar(x, exceeds_memory, width, bottom=compatible, label='Exceeds Memory', color='red')
        
        # Customize plot
        ax.set_title(title or 'Hardware Compatibility Analysis')
        ax.set_xlabel('Quantization Strategy')
        ax.set_ylabel('Compatibility Status')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save:
            self._save_figure(fig, 'hardware_compatibility')
        
        return fig
    
    def create_comprehensive_report(
        self,
        data: Dict[str, Any],
        model_name: str,
        timestamp: Optional[str] = None
    ) -> None:
        """
        Create a comprehensive visualization report.
        
        Args:
            data: Complete benchmark results data
            model_name: Name of the model analyzed
            timestamp: Optional timestamp for the report
        """
        # Create timestamp if not provided
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create report directory
        report_dir = self.results_dir / f"report_{model_name}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create all visualizations
        self.plot_memory_usage(data, save=True)
        self.plot_inference_speed(data, save=True)
        self.plot_accuracy_comparison(data, save=True)
        self.plot_efficiency_scores(data, save=True)
        self.plot_trade_off_analysis(data, save=True)
        self.plot_hardware_compatibility(data, save=True)
        
        # Create summary figure with all plots
        self._create_summary_figure(data, report_dir)
        
        logger.info(f"Created comprehensive visualization report in {report_dir}")
    
    def _create_summary_figure(
        self,
        data: Dict[str, Any],
        report_dir: Path
    ) -> None:
        """
        Create a summary figure with all visualizations.
        
        Args:
            data: Complete benchmark results data
            report_dir: Directory to save the summary figure
        """
        # Create a large figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Memory usage
        ax1 = plt.subplot(231)
        self._plot_memory_usage_subplot(data, ax1)
        
        # Inference speed
        ax2 = plt.subplot(232)
        self._plot_inference_speed_subplot(data, ax2)
        
        # Accuracy comparison
        ax3 = plt.subplot(233)
        self._plot_accuracy_comparison_subplot(data, ax3)
        
        # Efficiency scores
        ax4 = plt.subplot(234)
        self._plot_efficiency_scores_subplot(data, ax4)
        
        # Trade-off analysis
        ax5 = plt.subplot(235)
        self._plot_trade_off_analysis_subplot(data, ax5)
        
        # Hardware compatibility
        ax6 = plt.subplot(236)
        self._plot_hardware_compatibility_subplot(data, ax6)
        
        # Add title
        fig.suptitle('Quantization Analysis Summary', fontsize=16, y=0.95)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the summary figure
        self._save_figure(fig, 'summary', report_dir)
    
    def _plot_memory_usage_subplot(self, data: Dict[str, Any], ax: Axes) -> None:
        """Helper method for memory usage subplot."""
        strategies = []
        memory_values = []
        for item in data['memory_usage']:
            strategies.append(item['strategy'])
            memory_values.append(item['memory_gb'])
        
        bars = ax.bar(strategies, memory_values, color=self.colors)
        ax.set_title('Memory Usage')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Memory (GB)')
        ax.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom'
            )
    
    def _plot_inference_speed_subplot(self, data: Dict[str, Any], ax: Axes) -> None:
        """Helper method for inference speed subplot."""
        strategies = []
        times = []
        for item in data['inference_speed']:
            strategies.append(item['strategy'])
            times.append(item['time_per_image'])
        
        bars = ax.bar(strategies, times, color=self.colors)
        ax.set_title('Inference Speed')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Time (s)')
        ax.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom'
            )
    
    def _plot_accuracy_comparison_subplot(self, data: Dict[str, Any], ax: Axes) -> None:
        """Helper method for accuracy comparison subplot."""
        strategies = []
        accuracies = []
        for item in data['accuracy']:
            strategies.append(item['strategy'])
            accuracies.append(item['accuracy'])
        
        bars = ax.bar(strategies, accuracies, color=self.colors)
        ax.set_title('Accuracy')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Accuracy')
        ax.tick_params(axis='x', rotation=45)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.1%}',
                ha='center',
                va='bottom'
            )
    
    def _plot_efficiency_scores_subplot(self, data: Dict[str, Any], ax: Axes) -> None:
        """Helper method for efficiency scores subplot."""
        strategies = []
        scores = []
        for item in data['efficiency_scores']:
            strategies.append(item['strategy'])
            scores.append(item['score'])
        
        bars = ax.bar(strategies, scores, color=self.colors)
        ax.set_title('Efficiency Scores')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom'
            )
    
    def _plot_trade_off_analysis_subplot(self, data: Dict[str, Any], ax: Axes) -> None:
        """Helper method for trade-off analysis subplot."""
        strategies = []
        times = []
        accuracies = []
        for item in data['inference_speed']:
            strategy = item['strategy']
            strategies.append(strategy)
            times.append(item['time_per_image'])
            
            accuracy_item = next(
                (a for a in data['accuracy'] if a['strategy'] == strategy),
                None
            )
            accuracies.append(accuracy_item['accuracy'] if accuracy_item else 0.0)
        
        scatter = ax.scatter(times, accuracies, c=range(len(strategies)), cmap='viridis')
        ax.set_title('Speed-Accuracy Trade-off')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Accuracy')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        ax.grid(True, linestyle='--', alpha=0.7)
        
        for i, strategy in enumerate(strategies):
            ax.annotate(
                strategy,
                (times[i], accuracies[i]),
                xytext=(5, 5),
                textcoords='offset points'
            )
    
    def _plot_hardware_compatibility_subplot(self, data: Dict[str, Any], ax: Axes) -> None:
        """Helper method for hardware compatibility subplot."""
        strategies = []
        compatible = []
        exceeds_memory = []
        for item in data['hardware_compatibility']:
            strategies.append(item['strategy'])
            compatible.append(1 if item['compatible'] else 0)
            exceeds_memory.append(1 if item['exceeds_memory'] else 0)
        
        x = np.arange(len(strategies))
        width = 0.35
        
        ax.bar(x, compatible, width, label='Compatible', color='green')
        ax.bar(x, exceeds_memory, width, bottom=compatible, label='Exceeds Memory', color='red')
        
        ax.set_title('Hardware Compatibility')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Status')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45)
        ax.legend()
    
    def _save_figure(
        self,
        fig: Figure,
        name: str,
        directory: Optional[Path] = None
    ) -> None:
        """
        Save a figure to file.
        
        Args:
            fig: Matplotlib figure to save
            name: Name for the file
            directory: Optional directory to save in
        """
        # Use provided directory or default results directory
        save_dir = directory or self.results_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = save_dir / filename
        
        # Save figure
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved figure to {filepath}")
        
        # Close figure to free memory
        plt.close(fig)

# Create a global instance function for convenience
def get_quantization_visualizer(
    results_dir: Optional[Union[str, Path]] = None,
    style: str = 'seaborn',
    dpi: int = 300
) -> 'QuantizationVisualizer':
    """
    Get or create a QuantizationVisualizer instance.
    
    Args:
        results_dir: Directory to save visualization outputs
        style: Matplotlib style to use
        dpi: Dots per inch for saved figures
        
    Returns:
        QuantizationVisualizer instance
    """
    # Use a module-level singleton
    global _quantization_visualizer
    
    if '_quantization_visualizer' not in globals() or _quantization_visualizer is None:
        _quantization_visualizer = QuantizationVisualizer(
            results_dir=results_dir,
            style=style,
            dpi=dpi
        )
    
    return _quantization_visualizer 
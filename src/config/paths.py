"""
Path management for ML experiments.

This module provides a centralized way to manage paths across different
environments, ensuring consistent access to data, models, and results
regardless of where the code is executed.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

# Import environment configuration
from src.config.environment import get_environment_config

# Set up logging
logger = logging.getLogger(__name__)

class PathConfig:
    """
    Manages paths for ML experiments across different environments.
    
    This class handles path resolution, creation, and validation,
    ensuring consistent access to files and directories regardless
    of the execution environment.
    """
    
    def __init__(self, experiment_name: str = None, create_dirs: bool = True):
        """
        Initialize path configuration.
        
        Args:
            experiment_name: Name for this experiment (used in results paths)
            create_dirs: Whether to create directories if they don't exist
        """
        # Get environment configuration
        self.env_config = get_environment_config()
        
        # Load paths from environment configuration
        self.base_dir = self.env_config.get('paths.base_dir')
        self.data_dir = self.env_config.get('paths.data_dir')
        self.images_dir = self.env_config.get('paths.images_dir')
        self.ground_truth_path = self.env_config.get('paths.ground_truth_path')
        self.results_dir = self.env_config.get('paths.results_dir')
        self.model_cache_dir = self.env_config.get('paths.model_cache_dir')
        
        # Set up experiment-specific paths
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = Path(self.results_dir) / self.experiment_name

        # Set up subdirectories within the experiment directory
        self.raw_dir = self.experiment_dir / "raw"
        self.processed_dir = self.experiment_dir / "processed"
        self.visualizations_dir = self.experiment_dir / "visualizations"
        
        # Create directories if needed
        if create_dirs:
            self._create_directories()
            
        logger.info(f"Path configuration initialized for experiment: {self.experiment_name}")
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.images_dir,
            self.results_dir,
            self.experiment_dir,
            self.raw_dir,
            self.processed_dir,
            self.visualizations_dir,
            self.model_cache_dir
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                logger.info(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
    
    def verify(self, verbose: bool = True) -> bool:
        """
        Verify that critical paths exist and are accessible.
        
        Args:
            verbose: Whether to log verification results
            
        Returns:
            True if all critical paths exist and are accessible
        """
        critical_paths = {
            "Base directory": self.base_dir,
            "Data directory": self.data_dir,
            "Images directory": self.images_dir,
            "Ground truth file": self.ground_truth_path,
            "Results directory": self.results_dir,
            "Experiment directory": self.experiment_dir
        }
        
        all_valid = True
        
        for name, path in critical_paths.items():
            exists = os.path.exists(path)
            if not exists:
                all_valid = False
                if verbose:
                    logger.warning(f"{name} not found: {path}")
            elif verbose:
                logger.info(f"{name} verified: {path}")
        
        return all_valid
    
    def get_image_paths(self, limit: Optional[int] = None) -> List[Path]:
        """
        Get paths to all image files in the images directory.
        
        Args:
            limit: Optional limit on the number of images to return
            
        Returns:
            List of Path objects for each image
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(Path(self.images_dir).glob(f"*{ext}")))
        
        # Sort paths for consistent order
        image_paths.sort()
        
        if limit is not None and limit > 0:
            image_paths = image_paths[:limit]
            
        logger.info(f"Found {len(image_paths)} images")
        return image_paths
    
    def get_results_path(self, filename: str) -> str:
        """
        Get path for a results file within the experiment directory.
        
        Args:
            filename: Name of the file
            
        Returns:
            Full path to the results file
        """
        return os.path.join(self.experiment_dir, filename)
    
    def get_results_path(self, filename: str) -> str:
        """
        Get path for a results file within the experiment directory.
        
        Args:
            filename: Name of the file
            
        Returns:
            Full path to the results file
        """
        return str(Path(self.experiment_dir) / filename)
    
    def get_processed_path(self, filename: str) -> str:
        """
        Get path for a processed results file within the experiment's processed directory.
        
        Args:
            filename: Name of the file
            
        Returns:
            Full path to the processed results file
        """
        return str(Path(self.processed_dir, filename))
    
    def get_visualization_path(self, filename: str) -> str:
        """
        Get path for a visualization file within the experiment's visualizations directory.
        
        Args:
            filename: Name of the file
            
        Returns:
            Full path to the visualization file
        """
        return str(Path(self.visualizations_dir, filename))
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str = "metadata.json") -> str:
        """
        Save experiment metadata to a JSON file.
        
        Args:
            metadata: Dictionary of metadata to save
            filename: Name of the file
            
        Returns:
            Path to the saved file
        """
        file_path = self.get_results_path(filename)
        
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Metadata saved to {file_path}")
        return file_path
    
    def to_dict(self) -> Dict[str, str]:
        """
        Convert path configuration to a dictionary.
        
        Returns:
            Dictionary representation of the path configuration
        """
        return {
            "base_dir": self.base_dir,
            "data_dir": self.data_dir,
            "images_dir": self.images_dir,
            "ground_truth_path": self.ground_truth_path,
            "results_dir": self.results_dir,
            "model_cache_dir": self.model_cache_dir,
            "experiment_name": self.experiment_name,
            "experiment_dir": self.experiment_dir,
            "raw_dir": self.raw_dir,
            "processed_dir": self.processed_dir,
            "visualizations_dir": self.visualizations_dir
        }
    
    def __str__(self) -> str:
        """String representation of the path configuration."""
        return "\n".join(f"{k}: {v}" for k, v in self.to_dict().items())


def get_path_config(experiment_name: Optional[str] = None, create_dirs: bool = True) -> PathConfig:
    """
    Get a PathConfig instance for the current environment.
    
    Args:
        experiment_name: Optional name for this experiment
        create_dirs: Whether to create directories if they don't exist
        
    Returns:
        PathConfig instance
    """
    return PathConfig(experiment_name=experiment_name, create_dirs=create_dirs)


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Simple test of path configuration
    paths = get_path_config("test_experiment")
    print(paths)
    paths.verify()
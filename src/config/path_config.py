"""
Path Configuration Management

This module provides a comprehensive system for managing path configurations
across different environments in the invoice extraction system. It builds on
the environment configuration to ensure consistent path resolution and access
throughout the application.

Key features:
- Centralized path resolution with environment awareness
- Experiment-specific path generation
- Path validation and creation utilities
- Support for both local and RunPod environments
- Integration with the configuration system
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set

# Import from our configuration system
from src.config.environment_config import get_environment_config

# Set up logging
logger = logging.getLogger(__name__)


class PathConfig:
    """
    Manages path configurations for the invoice extraction system.
    
    This class provides a centralized way to resolve, validate, and access
    paths throughout the application, with support for environment-specific
    path resolution and experiment organization.
    """
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        create_dirs: bool = True,
        custom_paths: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the path configuration.
        
        Args:
            experiment_name: Optional name for the current experiment
                (used for organizing results)
            create_dirs: Whether to create directories if they don't exist
            custom_paths: Optional dictionary of custom path overrides
        """
        # Get environment configuration
        self.env_config = get_environment_config()
        
        # Set up base paths from environment configuration
        self.project_root = self.env_config.get('paths.base_dir')
        if not self.project_root or not os.path.exists(self.project_root):
            self.project_root = self._find_project_root()
            logger.info(f"Project root determined as: {self.project_root}")
        
        # Set up standard paths with defaults
        self.paths = {
            'project_root': self.project_root,
            'configs_dir': os.path.join(self.project_root, 'configs'),
            'data_dir': self.env_config.get('paths.data_dir', os.path.join(self.project_root, 'data')),
            'models_dir': os.path.join(self.project_root, 'models'),
            'results_dir': self.env_config.get('paths.results_dir', os.path.join(self.project_root, 'results')),
            'logs_dir': os.path.join(self.project_root, 'logs'),
        }
        
        # Apply custom path overrides if provided
        if custom_paths:
            self.paths.update(custom_paths)
        
        # Set up derived paths
        self._setup_derived_paths()
        
        # Set up experiment-specific paths
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._setup_experiment_paths()
        
        # Create directories if requested
        if create_dirs:
            self._create_directories()
        
        logger.info(f"Path configuration initialized for experiment: {self.experiment_name}")
    
    def _find_project_root(self) -> str:
        """
        Find the project root directory from the current file location.
        
        Returns:
            Absolute path to the project root directory
        """
        # Start with the directory of this file
        current_file = Path(__file__).resolve()
        current_dir = current_file.parent
        
        # Look for common project root indicators
        root_indicators = [
            '.git',                # Git repository
            'requirements.txt',    # Python project
            'setup.py',            # Python package
            'README.md',           # Project documentation
            'src/config'           # Our specific project structure
        ]
        
        # Walk up the directory tree until we find a root indicator
        search_dir = current_dir
        while search_dir != search_dir.parent:  # Stop at filesystem root
            # Check if any indicators exist in this directory
            if any((search_dir / indicator).exists() for indicator in root_indicators):
                # If this is the config directory, go up to the project root
                if search_dir.name == 'config':
                    return str(search_dir.parent.parent)  # src/config -> src -> project_root
                return str(search_dir)
            
            # Move up one directory
            search_dir = search_dir.parent
        
        # If we couldn't find a proper indicator, default to two levels up from this file
        # (assuming src/config/paths.py structure)
        logger.warning("Could not find project root indicators, using default")
        return str(current_dir.parent.parent)
    
    def _setup_derived_paths(self) -> None:
        """Set up derived paths based on base paths."""
        # Data-related paths
        self.paths.update({
            'images_dir': self.env_config.get('paths.images_dir', os.path.join(self.paths['data_dir'], 'images')),
            'ground_truth_path': self.env_config.get('paths.ground_truth_path', 
                                                  os.path.join(self.paths['data_dir'], 'ground_truth.csv')),
        })
        
        # Model-related paths
        self.paths.update({
            'model_cache_dir': self.env_config.get('paths.model_cache_dir', 
                                                os.path.join(self.paths['models_dir'], 'cache')),
            'model_configs_dir': os.path.join(self.paths['configs_dir'], 'models'),
        })
        
        # Config-related paths
        self.paths.update({
            'environment_configs_dir': os.path.join(self.paths['configs_dir'], 'environments'),
            'prompt_configs_dir': os.path.join(self.paths['configs_dir'], 'prompts'),
            'experiment_configs_dir': os.path.join(self.paths['configs_dir'], 'experiments'),
            'pipeline_configs_dir': os.path.join(self.paths['configs_dir'], 'pipelines'),
        })
    
    def _setup_experiment_paths(self) -> None:
        """Set up experiment-specific paths."""
        # Create experiment directory structure
        self.paths.update({
            'experiment_dir': os.path.join(self.paths['results_dir'], self.experiment_name),
            'experiment_raw_dir': os.path.join(self.paths['results_dir'], self.experiment_name, 'raw'),
            'experiment_processed_dir': os.path.join(self.paths['results_dir'], self.experiment_name, 'processed'),
            'experiment_visualizations_dir': os.path.join(self.paths['results_dir'], self.experiment_name, 'visualizations'),
            'experiment_checkpoints_dir': os.path.join(self.paths['results_dir'], self.experiment_name, 'checkpoints'),
        })
        
        # Specific field directories within the experiment
        field_types = self.env_config.get('fields', ['work_order', 'cost', 'date'])
        for field in field_types:
            field_dir = os.path.join(self.paths['experiment_dir'], field)
            self.paths.update({
                f'field_{field}_dir': field_dir,
                f'field_{field}_raw_dir': os.path.join(field_dir, 'raw'),
                f'field_{field}_processed_dir': os.path.join(field_dir, 'processed'),
                f'field_{field}_visualizations_dir': os.path.join(field_dir, 'visualizations'),
            })
        
        # Experiment log path
        self.paths['experiment_log_path'] = os.path.join(
            self.paths['logs_dir'], 
            f"{self.experiment_name}.log"
        )
        
        # Experiment metadata path
        self.paths['experiment_metadata_path'] = os.path.join(
            self.paths['experiment_dir'],
            'experiment_metadata.json'
        )
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        # Identify which paths are directories (not files)
        directory_paths = {
            key: path for key, path in self.paths.items()
            if not key.endswith('_path') and not key.endswith('_file')
        }
        
        # Create each directory
        for name, path in directory_paths.items():
            if not os.path.exists(path):
                try:
                    os.makedirs(path, exist_ok=True)
                    logger.info(f"Created directory: {path}")
                except Exception as e:
                    logger.warning(f"Failed to create directory {path}: {e}")
    
    def get(self, path_key: str, default: Optional[str] = None) -> str:
        """
        Get a path by its key.
        
        Args:
            path_key: Key for the requested path
            default: Default value if the key is not found
            
        Returns:
            The requested path or the default value
        """
        return self.paths.get(path_key, default)
    
    def get_path(self, path_key: str) -> Optional[Path]:
        """
        Get a path as a Path object.
        
        Args:
            path_key: Key for the requested path
            
        Returns:
            Path object or None if not found
        """
        path_str = self.get(path_key)
        if path_str:
            return Path(path_str)
        return None
    
    def get_filepath(self, directory_key: str, filename: str) -> str:
        """
        Build a filepath by combining a directory key with a filename.
        
        Args:
            directory_key: Key for the directory
            filename: Name of the file
            
        Returns:
            Complete filepath as string
            
        Raises:
            ValueError: If the directory key is not found
        """
        directory = self.get(directory_key)
        if not directory:
            raise ValueError(f"Directory key not found: {directory_key}")
        
        return os.path.join(directory, filename)
    
    def get_raw_path(self, filename: str) -> str:
        """
        Get a path in the experiment's raw directory.
        
        Args:
            filename: Name of the file
            
        Returns:
            Path to the file in the raw directory
        """
        return self.get_filepath('experiment_raw_dir', filename)
    
    def get_processed_path(self, filename: str) -> str:
        """
        Get a path in the experiment's processed directory.
        
        Args:
            filename: Name of the file
            
        Returns:
            Path to the file in the processed directory
        """
        return self.get_filepath('experiment_processed_dir', filename)
    
    def get_visualization_path(self, filename: str) -> str:
        """
        Get a path in the experiment's visualizations directory.
        
        Args:
            filename: Name of the file
            
        Returns:
            Path to the file in the visualizations directory
        """
        return self.get_filepath('experiment_visualizations_dir', filename)
    
    def get_checkpoint_path(self, name: str) -> str:
        """
        Get a path for a checkpoint file.
        
        Args:
            name: Base name for the checkpoint
            
        Returns:
            Path to the checkpoint file
        """
        return self.get_filepath('experiment_checkpoints_dir', f"{name}_checkpoint.json")
    
    def get_field_path(self, field: str, subdir: str = None) -> str:
        """
        Get the path for a specific field's directory.
        
        Args:
            field: Field type (e.g., 'work_order', 'cost')
            subdir: Optional subdirectory within the field directory
            
        Returns:
            Path to the field directory or subdirectory
            
        Raises:
            ValueError: If the field is not recognized
        """
        field_key = f'field_{field}_dir'
        if field_key not in self.paths:
            raise ValueError(f"Unknown field type: {field}")
        
        field_dir = self.get(field_key)
        
        if subdir:
            subdir_key = f'field_{field}_{subdir}_dir'
            if subdir_key in self.paths:
                return self.get(subdir_key)
            return os.path.join(field_dir, subdir)
        
        return field_dir
    
    def get_results_path(self, filename: str) -> str:
        """
        Get a path in the experiment's root directory.
        
        Args:
            filename: Name of the file
            
        Returns:
            Path to the file in the experiment directory
        """
        return self.get_filepath('experiment_dir', filename)
    
    def get_config_path(self, config_type: str, name: str) -> str:
        """
        Get a path to a configuration file.
        
        Args:
            config_type: Type of configuration ('models', 'prompts', 'environments', etc.)
            name: Name of the configuration file (without extension)
            
        Returns:
            Path to the configuration file
            
        Raises:
            ValueError: If the configuration type is not recognized
        """
        config_dir_key = f'{config_type}_configs_dir'
        if config_dir_key not in self.paths:
            raise ValueError(f"Unknown configuration type: {config_type}")
        
        config_dir = self.get(config_dir_key)
        return os.path.join(config_dir, f"{name}.yaml")
    
    def get_image_paths(self, limit: Optional[int] = None) -> List[Path]:
        """
        Get paths to all image files in the images directory.
        
        Args:
            limit: Optional maximum number of images to return
            
        Returns:
            List of Path objects for the image files
        """
        images_dir = Path(self.get('images_dir'))
        
        # Image file extensions to look for
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        
        # Collect images with different extensions
        image_paths = []
        for ext in extensions:
            image_paths.extend(list(images_dir.glob(f"*{ext}")))
            image_paths.extend(list(images_dir.glob(f"*{ext.upper()}")))  # Case insensitive
        
        # Sort for consistent ordering
        image_paths.sort()
        
        # Apply limit if specified
        if limit is not None and limit > 0 and limit < len(image_paths):
            image_paths = image_paths[:limit]
            
        logger.info(f"Found {len(image_paths)} images in {images_dir}")
        return image_paths
    
    def verify_paths(self, required_paths: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Verify that required paths exist.
        
        Args:
            required_paths: List of path keys to verify
                (if None, checks a default set of critical paths)
            
        Returns:
            Dictionary mapping path keys to existence status
        """
        if required_paths is None:
            # Default set of critical paths to verify
            required_paths = [
                'project_root',
                'data_dir',
                'images_dir',
                'ground_truth_path',
                'results_dir',
                'experiment_dir',
                'model_cache_dir'
            ]
        
        verification_results = {}
        for key in required_paths:
            path = self.get(key)
            exists = path and os.path.exists(path)
            verification_results[key] = exists
            
            if not exists:
                logger.warning(f"Required path '{key}' not found: {path}")
            
        return verification_results
    
    def verify_critical_paths(self) -> bool:
        """
        Verify that all critical paths exist.
        
        Returns:
            True if all critical paths exist, False otherwise
        """
        verification_results = self.verify_paths()
        return all(verification_results.values())
    
    def save_experiment_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Save experiment metadata to a JSON file.
        
        Args:
            metadata: Dictionary of metadata to save
            
        Returns:
            Path to the saved metadata file
        """
        metadata_path = self.get('experiment_metadata_path')
        
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Add path information to metadata
        metadata.update({
            'paths': {
                'experiment_dir': self.get('experiment_dir'),
                'raw_dir': self.get('experiment_raw_dir'),
                'processed_dir': self.get('experiment_processed_dir'),
                'visualizations_dir': self.get('experiment_visualizations_dir'),
                'checkpoints_dir': self.get('experiment_checkpoints_dir')
            }
        })
        
        # Save to file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved experiment metadata to {metadata_path}")
        return metadata_path
    
    def to_dict(self) -> Dict[str, str]:
        """
        Convert the path configuration to a dictionary.
        
        Returns:
            Dictionary of all paths
        """
        return self.paths.copy()
    
    def __str__(self) -> str:
        """String representation of the path configuration."""
        path_list = [f"{key}: {value}" for key, value in self.paths.items()]
        return f"PathConfig for experiment '{self.experiment_name}':\n" + "\n".join(path_list)


# Singleton instance for global access
_path_config = None


def get_path_config(
    experiment_name: Optional[str] = None,
    create_dirs: bool = True,
    custom_paths: Optional[Dict[str, str]] = None,
    reload: bool = False
) -> PathConfig:
    """
    Get the path configuration singleton.
    
    Args:
        experiment_name: Optional name for the current experiment
        create_dirs: Whether to create directories if they don't exist
        custom_paths: Optional dictionary of custom path overrides
        reload: Force creation of a new instance
        
    Returns:
        PathConfig instance
    """
    global _path_config
    
    if _path_config is None or reload:
        _path_config = PathConfig(
            experiment_name=experiment_name,
            create_dirs=create_dirs,
            custom_paths=custom_paths
        )
    elif experiment_name and _path_config.experiment_name != experiment_name:
        # Create a new instance if experiment name has changed
        _path_config = PathConfig(
            experiment_name=experiment_name,
            create_dirs=create_dirs,
            custom_paths=custom_paths
        )
        
    return _path_config


# Helper functions for common path operations

def get_experiment_dir(experiment_name: Optional[str] = None) -> str:
    """
    Get the directory for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
            (if None, uses the current experiment in the singleton)
            
    Returns:
        Path to the experiment directory
    """
    paths = get_path_config(experiment_name=experiment_name)
    return paths.get('experiment_dir')


def get_experiment_result_path(filename: str, experiment_name: Optional[str] = None) -> str:
    """
    Get a path for a result file in a specific experiment.
    
    Args:
        filename: Name of the file
        experiment_name: Name of the experiment
            (if None, uses the current experiment in the singleton)
            
    Returns:
        Path to the result file
    """
    paths = get_path_config(experiment_name=experiment_name)
    return paths.get_results_path(filename)


def get_ground_truth_path() -> str:
    """
    Get the path to the ground truth CSV file.
    
    Returns:
        Path to the ground truth file
    """
    paths = get_path_config()
    return paths.get('ground_truth_path')


if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO)
    
    # Create and test path configuration
    paths = get_path_config("test_experiment")
    
    # Print path information
    print(f"Project root: {paths.get('project_root')}")
    print(f"Data directory: {paths.get('data_dir')}")
    print(f"Images directory: {paths.get('images_dir')}")
    print(f"Ground truth path: {paths.get('ground_truth_path')}")
    print(f"Experiment directory: {paths.get('experiment_dir')}")
    
    # Test path verification
    verification_results = paths.verify_paths()
    print("\nPath verification results:")
    for key, exists in verification_results.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {key}: {paths.get(key)}")
    
    # Test image path resolution
    image_paths = paths.get_image_paths(limit=5)
    print(f"\nFound {len(image_paths)} images:")
    for i, path in enumerate(image_paths[:5]):
        print(f"  {i+1}. {path}")
"""
Path Configuration Management

This module provides a comprehensive system for managing path configurations
across different environments in the invoice extraction system.

Key features:
- Centralized path resolution with environment awareness
- Experiment-specific path generation
- Path validation and creation utilities
- Support for both local and RunPod environments
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
import sys

# Set up logging
logger = logging.getLogger(__name__)

class PathConfig:
    """
    Simple path configuration system with no external dependencies.
    
    This class defines standard project paths relative to the project root
    without depending on any other configuration systems.
    """
    
    # Class-level cache for the singleton instance
    _instance = None
    
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
        # Detect project root
        self.project_root = self._detect_project_root()
        logger.info(f"Using project root: {self.project_root}")
        
        # Make project root accessible to other modules
        os.environ["PROJECT_ROOT"] = str(self.project_root)
        
        # Initialize all standard paths
        self.paths = {
            'project_root': str(self.project_root),
            'src_dir': str(self.project_root / "src"),
            'data_dir': str(self.project_root / "data"),
            'models_dir': str(self.project_root / "models"),
            'config_dir': str(self.project_root / "configs"),
            'configs_dir': str(self.project_root / "configs"),  # Alias for compatibility
            'results_dir': str(self.project_root / "results"),
            'logs_dir': str(self.project_root / "logs"),
            'cache_dir': str(self.project_root / "cache"),
            'notebooks_dir': str(self.project_root / "notebooks"),
            'tests_dir': str(self.project_root / "tests"),
            'package_dir': str(self.project_root / "src"),  # Alias for compatibility
        }
        
        # Add common subdirectories
        self._add_subdirectories()
        
        # Apply custom path overrides if provided
        if custom_paths:
            self.paths.update(custom_paths)
        
        # Set up experiment-specific paths
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._setup_experiment_paths()
        
        # Create directories if requested
        if create_dirs:
            self._create_directories()
    
    def _detect_project_root(self) -> Path:
        """
        Simple and reliable project root detection.
        
        Returns:
            Path object representing the project root
        """
        # Check for RunPod environment (explicit case)
        if self._is_runpod_env():
            return Path("/workspace")
        
        # Check environment variable (if set by user)
        if "PROJECT_ROOT" in os.environ:
            root = Path(os.environ["PROJECT_ROOT"])
            if root.exists():
                return root
        
        # Start with this file's location and search upward
        current_file = Path(__file__).resolve()
        search_dir = current_file.parent.parent.parent  # Start 3 levels up from config file
        
        # Look for common project markers
        markers = [".git", "setup.py", "requirements.txt", "README.md"]
        for marker in markers:
            if (search_dir / marker).exists():
                return search_dir
        
        # Fall back to current working directory
        return Path(os.getcwd())
    
    def _is_runpod_env(self) -> bool:
        """Check if running in RunPod environment."""
        if any(var in os.environ for var in ["RUNPOD_POD_ID", "RUNPOD_API_KEY"]):
            return True
        if Path("/workspace").exists() and Path("/runpod").exists():
            return True
        return False
    
    def _add_subdirectories(self) -> None:
        """Add common subdirectories to the paths dictionary."""
        # Data subdirectories
        self.paths.update({
            'images_dir': str(Path(self.paths['data_dir']) / "images"),
            'ground_truth_path': str(Path(self.paths['data_dir']) / "ground_truth.csv"),
        })
        
        # Config subdirectories
        self.paths.update({
            'environment_configs_dir': str(Path(self.paths['config_dir']) / "environments"),
            'prompt_configs_dir': str(Path(self.paths['config_dir']) / "prompts"),
            'experiment_configs_dir': str(Path(self.paths['config_dir']) / "experiments"),
            'pipeline_configs_dir': str(Path(self.paths['config_dir']) / "pipelines"),
            'templates_dir': str(Path(self.paths['config_dir']) / "templates"),
        })
        
        # Model subdirectories
        self.paths.update({
            'model_cache_dir': str(Path(self.paths['models_dir']) / "cache"),
        })
    
    def _setup_experiment_paths(self) -> None:
        """Set up experiment-specific paths."""
        self.paths.update({
            'experiment_dir': os.path.join(self.paths['results_dir'], self.experiment_name),
            'experiment_raw_dir': os.path.join(self.paths['results_dir'], self.experiment_name, 'raw'),
            'experiment_processed_dir': os.path.join(self.paths['results_dir'], self.experiment_name, 'processed'),
            'experiment_logs_dir': os.path.join(self.paths['logs_dir'], self.experiment_name),
        })
    
    def _create_directories(self) -> None:
        """Create directories if they don't exist."""
        for path_name, path_str in self.paths.items():
            if path_name.endswith('_dir'):  # Only create directories
                path = Path(path_str)
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        logger.warning(f"Could not create directory {path}: {e}")
    
    def get(self, path_name: str, default: Optional[str] = None) -> Optional[str]:
        """Get a path by name with default fallback."""
        return self.paths.get(path_name, default)
    
    def get_path(self, path_name: str) -> Optional[str]:
        """Get a path by name."""
        return self.paths.get(path_name)
    
    def update_paths(self, new_paths: Dict[str, str]) -> None:
        """Update multiple paths at once."""
        self.paths.update(new_paths)
    
    def get_results_path(self, filename: str) -> str:
        """Get path to a results file in the experiment directory."""
        return os.path.join(self.paths['experiment_dir'], filename)
    
    def verify_paths(self, required_paths: Optional[List[str]] = None) -> Dict[str, bool]:
        """Verify that required paths exist."""
        if required_paths is None:
            required_paths = ['project_root', 'config_dir', 'data_dir', 'models_dir', 'results_dir']
        
        results = {}
        for path_key in required_paths:
            path = self.get(path_key)
            results[path_key] = path is not None and os.path.exists(path)
        
        return results
    
    def get_image_paths(self, limit: Optional[int] = None) -> List[Path]:
        """
        Get paths to image files in the images directory.
        
        Args:
            limit: Maximum number of images to return
            
        Returns:
            List of Path objects for the image files
        """
        images_dir = Path(self.get('images_dir'))
        if not images_dir.exists():
            logger.warning(f"Images directory {images_dir} does not exist")
            return []
            
        # Common image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        
        # Collect all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(list(images_dir.glob(f"*{ext}")))
            image_paths.extend(list(images_dir.glob(f"*{ext.upper()}")))
        
        # Sort for consistent ordering
        image_paths.sort()
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            image_paths = image_paths[:limit]
            
        logger.info(f"Found {len(image_paths)} images in {images_dir}")
        return image_paths

# Global singleton instance
_path_config = None

def get_path_config(
    experiment_name: Optional[str] = None, 
    custom_paths: Optional[Dict[str, str]] = None,
    reset: bool = False
) -> PathConfig:
    """
    Get global PathConfig instance (singleton pattern).
    
    Args:
        experiment_name: Optional experiment name
        custom_paths: Optional custom path overrides
        reset: Whether to force creating a new instance
        
    Returns:
        PathConfig instance
    """
    global _path_config
    
    if _path_config is None or reset:
        _path_config = PathConfig(
            experiment_name=experiment_name,
            custom_paths=custom_paths
        )
    elif experiment_name and experiment_name != _path_config.experiment_name:
        _path_config.experiment_name = experiment_name
        _path_config._setup_experiment_paths()
        _path_config._create_directories()
    
    return _path_config

def get_path(path_name: str) -> Optional[str]:
    """Convenience function to get a path by name."""
    return get_path_config().get_path(path_name)

def get_paths() -> Dict[str, str]:
    """Get all paths as a dictionary."""
    return get_path_config().paths

def get_project_root() -> str:
    """Get the project root path."""
    return get_path_config().get('project_root')

def get_data_dir() -> str:
    """Get the data directory path."""
    return get_path_config().get('data_dir')

def get_models_dir() -> str:
    """Get the models directory path."""
    return get_path_config().get('models_dir')

def get_config_dir() -> str:
    """Get the config directory path."""
    return get_path_config().get('config_dir')

def get_results_dir() -> str:
    """Get the results directory path."""
    return get_path_config().get('results_dir')

def get_experiment_dir(experiment_name: Optional[str] = None) -> str:
    """Get experiment directory path."""
    if experiment_name:
        # Use the function to ensure consistent formatting
        return get_path_config(experiment_name=experiment_name).get('experiment_dir')
    return get_path_config().get('experiment_dir')

def get_experiment_result_path(filename: str, experiment_name: Optional[str] = None) -> str:
    """Get path to a result file in the experiment directory."""
    if experiment_name:
        return get_path_config(experiment_name=experiment_name).get_results_path(filename)
    return get_path_config().get_results_path(filename)

def get_ground_truth_path() -> str:
    """Get the ground truth CSV file path."""
    return get_path_config().get('ground_truth_path')

def add_project_root_to_path() -> None:
    """Add project root to Python path if not already there."""
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        logger.info(f"Added {project_root} to Python path")


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
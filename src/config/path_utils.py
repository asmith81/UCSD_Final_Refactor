"""
Path detection and management utilities.

This module provides functions for detecting project roots and managing paths
across different environments.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

# Set up logging
logger = logging.getLogger(__name__)

def detect_project_root(start_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Detect the project root directory by looking for standard project markers.
    
    Args:
        start_path: Optional starting path to begin search from (defaults to current directory)
        
    Returns:
        Path object pointing to the project root
    """
    # Convert start_path to Path object
    if start_path is None:
        start_path = os.getcwd()
    current_path = Path(start_path)
    
    # List of markers that indicate we've found the project root
    root_markers = [
        "src",  # Source code directory
        "configs",  # Configuration directory
        "data",  # Data directory
        "models",  # Models directory
        "notebooks",  # Notebooks directory
        "tests",  # Tests directory
        "requirements.txt",  # Python requirements file
        "setup.py",  # Python setup file
        "pyproject.toml",  # Modern Python project file
        ".git",  # Git repository
    ]
    
    # Start from current directory and walk up until we find project root
    while current_path != current_path.parent:
        # Check if current directory has enough markers to be considered the root
        markers_found = sum(1 for marker in root_markers if (current_path / marker).exists())
        
        if markers_found >= 2:  # Require at least 2 markers to be confident
            logger.info(f"Found project root at: {current_path}")
            return current_path
            
        # Move up one directory
        current_path = current_path.parent
    
    # If we get here, we couldn't find a clear project root
    # Fall back to the directory containing src/ if it exists
    src_path = Path(start_path)
    while src_path != src_path.parent:
        if (src_path / "src").exists():
            logger.info(f"Found project root (src-based) at: {src_path}")
            return src_path
        src_path = src_path.parent
    
    # If all else fails, use the current directory
    logger.warning("Could not detect project root, using current directory")
    return Path(os.getcwd())

def get_workspace_root() -> Path:
    """
    Get the workspace root directory, which may be different from the project root
    in certain environments (e.g., RunPod).
    
    Returns:
        Path object pointing to the workspace root
    """
    # Check for RunPod environment
    if os.path.exists("/workspace"):
        return Path("/workspace")
    
    # Check for environment variable
    workspace = os.environ.get("WORKSPACE_ROOT")
    if workspace and os.path.exists(workspace):
        return Path(workspace)
    
    # Fall back to project root
    return detect_project_root()

def ensure_directory(path: Union[str, Path], create: bool = True) -> Path:
    """
    Ensure a directory exists, optionally creating it if it doesn't.
    
    Args:
        path: Path to check/create
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    if create and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path 
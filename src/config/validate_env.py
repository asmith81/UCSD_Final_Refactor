"""
Environment validation script.

This script checks if the current environment meets the requirements
for running the invoice processing project. It validates dependencies,
hardware, and configuration settings.
"""

import os
import sys
import platform
from pathlib import Path
import importlib.util
import logging
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('environment_validation')

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
        
    try:
        if importlib.util.find_spec(module_name) is not None:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'unknown')
            logger.info(f"✅ {package_name} is installed (version: {version})")
            return True
        else:
            logger.warning(f"❌ {package_name} is not installed")
            return False
    except ImportError:
        logger.warning(f"❌ {package_name} is not installed")
        return False

def check_gpu():
    """Check for GPU availability and specifications."""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
            
            logger.info(f"✅ GPU is available: {device_name}")
            logger.info(f"   - Device count: {device_count}")
            logger.info(f"   - Memory: {memory:.2f} GB")
            
            # Check if memory is sufficient for Pixtral-12B
            if memory < 20:
                logger.warning(f"⚠️ GPU memory might be insufficient for Pixtral-12B (recommended: 24GB+)")
            
            return True
        else:
            logger.warning("❌ CUDA is not available")
            return False
    except ImportError:
        logger.warning("❌ PyTorch is not installed")
        return False

def find_project_root() -> Optional[Path]:
    """
    Attempt to find the project root directory.
    
    Returns:
        Path to the project root if found, None otherwise
    """
    # Check if PROJECT_ROOT environment variable is set
    if "PROJECT_ROOT" in os.environ:
        project_root = Path(os.environ["PROJECT_ROOT"])
        if project_root.exists():
            return project_root
    
    # Check for RunPod environment
    runpod_indicators = ["RUNPOD_POD_ID", "RUNPOD_GPU_COUNT"]
    if any(indicator in os.environ for indicator in runpod_indicators) or os.path.exists('/workspace'):
        # In RunPod, the standard project root is /workspace
        workspace_path = Path('/workspace')
        if workspace_path.exists():
            # Set environment variable for future use
            os.environ["PROJECT_ROOT"] = str(workspace_path)
            return workspace_path
    
    # Start with the current file's directory
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    
    # Look for common project markers
    markers = [
        ".git",
        "setup.py",
        "requirements.txt",
        "README.md",
        ".github"
    ]
    
    # Walk up the directory tree looking for markers
    search_dir = current_dir
    while search_dir != search_dir.parent:  # Stop at filesystem root
        for marker in markers:
            if (search_dir / marker).exists():
                return search_dir
        search_dir = search_dir.parent
    
    # If we reach here, we couldn't find the project root
    return None

def check_project_structure():
    """Check if the project structure is correctly set up."""
    project_root = find_project_root()
    
    if project_root is None:
        logger.warning("❌ Could not determine project root")
        
        # Check if in RunPod as fallback
        if os.path.exists('/workspace'):
            logger.info("✅ Running in RunPod environment, using /workspace as project root")
            project_root = Path('/workspace')
            os.environ['PROJECT_ROOT'] = str(project_root)
        else:
            return False
    
    logger.info(f"✅ Project root: {project_root}")
    
    # Check for core directories
    required_dirs = [
        "src",
        "configs",
        "data"
    ]
    
    # In RunPod, only check for critical directories
    if str(project_root) == '/workspace':
        required_dirs = ["src"]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            logger.warning(f"❌ Required directory not found: {dir_path}")
            # Create the directory if in RunPod to help setup
            if str(project_root) == '/workspace':
                try:
                    dir_path.mkdir(exist_ok=True)
                    logger.info(f"✅ Created directory: {dir_path}")
                except Exception as e:
                    logger.warning(f"Failed to create directory {dir_path}: {e}")
    
    # Add source directory to Python path if not already there
    src_dir = str(project_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        logger.info(f"✅ Added to Python path: {src_dir}")
    
    # Set PROJECT_ROOT environment variable if not already set
    if os.environ.get('PROJECT_ROOT') != str(project_root):
        os.environ['PROJECT_ROOT'] = str(project_root)
        logger.info(f"✅ Set PROJECT_ROOT environment variable: {project_root}")
    
    return True

def validate_environment():
    """Run all validation checks and return overall status."""
    logger.info(f"🔍 Validating environment on {platform.system()} {platform.release()}")
    
    # Check Python version
    python_version = platform.python_version()
    logger.info(f"🐍 Python version: {python_version}")
    if tuple(map(int, python_version.split('.'))) < (3, 8):
        logger.warning("⚠️ Python version 3.8+ is recommended")
    
    # Check critical packages
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'yaml': 'PyYAML',
        'PIL': 'Pillow',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib'
    }
    
    missing_packages = []
    for module, package in required_packages.items():
        if not check_import(module, package):
            missing_packages.append(package)
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Check project structure
    valid_structure = check_project_structure()
    
    # Overall validation result
    if missing_packages:
        logger.warning(f"❌ Missing required packages: {', '.join(missing_packages)}")
        logger.warning("Please install them using pip install <package-name>")
    
    if not has_gpu:
        logger.warning("⚠️ No GPU detected. The project will run slower on CPU.")
        logger.warning("Some models may be too large to run on CPU.")
    
    if not valid_structure:
        logger.warning("⚠️ Project structure has issues.")
        logger.warning("Please check the logs and fix any missing directories or files.")
    
    if not missing_packages and valid_structure:
        logger.info("✅ Environment validation passed!")
        return True
    else:
        logger.warning("⚠️ Environment validation completed with warnings.")
        return False

if __name__ == "__main__":
    # Run validation and set exit code
    success = validate_environment()
    sys.exit(0 if success else 1)
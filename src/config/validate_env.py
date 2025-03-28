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

def check_project_structure():
    """Check if the project structure is correctly set up."""
    # Try to determine project root
    current_path = Path(__file__).resolve().parent
    project_root = None
    
    # First check environment variable
    if 'PROJECT_ROOT' in os.environ:
        project_root = Path(os.environ['PROJECT_ROOT'])
        logger.info(f"📂 Project root from environment: {project_root}")
    else:
        # Try to determine programmatically
        markers = [".git", "setup.py", "requirements.txt", "README.md"]
        search_path = current_path
        
        while search_path != search_path.parent:
            if any((search_path / marker).exists() for marker in markers):
                project_root = search_path
                break
            search_path = search_path.parent
    
    if project_root is None:
        logger.warning("❌ Could not determine project root")
        return False
    
    # Check critical directories
    critical_dirs = [
        "configs/environments",
        "src/config",
        "data",
        "models/cache",
        "results"
    ]
    
    missing_dirs = []
    for dir_path in critical_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logger.warning(f"❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    
    # Check configuration files
    config_files = [
        "configs/environments/local.yaml",
        "configs/experiment.yaml"
    ]
    
    missing_files = []
    for file_path in config_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"❌ Missing configuration files: {', '.join(missing_files)}")
        return False
    
    logger.info("✅ Project structure is correctly set up")
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
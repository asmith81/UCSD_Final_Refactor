"""
Notebook environment setup and validation utilities.

This module provides functions for validating the runtime environment,
checking GPU availability, configuring paths, and ensuring all required
dependencies are installed.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import importlib
import pkg_resources
import json
import shutil
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("setup_utils")

# Suppress specific warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")
warnings.filterwarnings("ignore", message="torch._C._jit_set_profiling_executor")


def validate_environment() -> Dict[str, bool]:
    """
    Validate the runtime environment for compatibility with the extraction system.
    
    Returns:
        Dictionary with validation results for different components
    """
    results = {
        "python_version": False,
        "cuda_available": False,
        "required_packages": False,
        "directory_structure": False,
        "environment_variables": False,
        "disk_space": False
    }
    
    # Check Python version
    py_version = sys.version_info
    results["python_version"] = (py_version.major == 3 and py_version.minor >= 8)
    logger.info(f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro} - {'✓' if results['python_version'] else '✗'}")
    
    # Check CUDA availability
    try:
        import torch
        results["cuda_available"] = torch.cuda.is_available()
        
        if results["cuda_available"]:
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            logger.info(f"CUDA available: {device_name}, CUDA version: {cuda_version} - ✓")
        else:
            logger.warning("CUDA not available. GPU acceleration will not be used.")
    except ImportError:
        logger.warning("PyTorch not installed. Cannot check CUDA availability.")
    
    # Check required packages
    required_packages = [
        "torch", "numpy", "pandas", "matplotlib", 
        "tqdm", "pyyaml", "pillow", "transformers"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    results["required_packages"] = len(missing_packages) == 0
    if results["required_packages"]:
        logger.info("All required packages installed - ✓")
    else:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
    
    # Check directory structure
    base_dir = Path(os.getcwd())
    if not (base_dir / "src").exists():
        # Try to find project root
        if (base_dir.parent / "src").exists():
            base_dir = base_dir.parent
    
    required_dirs = ["src", "data", "models", "configs", "notebooks", "results"]
    missing_dirs = [d for d in required_dirs if not (base_dir / d).exists()]
    
    results["directory_structure"] = len(missing_dirs) == 0
    if results["directory_structure"]:
        logger.info("Directory structure valid - ✓")
    else:
        logger.warning(f"Missing directories: {', '.join(missing_dirs)}")
    
    # Check environment variables
    env_vars = ["DATA_DIR", "MODELS_DIR", "RESULTS_DIR", "CONFIGS_DIR"]
    missing_vars = [var for var in env_vars if not os.environ.get(var)]
    
    if not missing_vars:
        results["environment_variables"] = True
        logger.info("Environment variables configured - ✓")
    else:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Will try to set these from default paths.")
    
    # Check disk space
    try:
        free_space = shutil.disk_usage("/").free / (1024 ** 3)  # Free space in GB
        results["disk_space"] = free_space >= 10  # Require at least 10GB free
        logger.info(f"Free disk space: {free_space:.1f} GB - {'✓' if results['disk_space'] else '✗'}")
    except:
        logger.warning("Cannot check disk space")
    
    return results


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and provide detailed information.
    
    Returns:
        Dictionary with GPU information
    """
    gpu_info = {
        "available": False,
        "name": "Not available",
        "memory_total": 0,
        "memory_free": 0,
        "cuda_version": None,
        "device_count": 0
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["cuda_version"] = torch.version.cuda
            
            # Get memory information
            gpu_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
            
            # Reserved memory calculation (rough estimate)
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            gpu_info["memory_free"] = gpu_info["memory_total"] - reserved
            
            logger.info(f"GPU: {gpu_info['name']}")
            logger.info(f"CUDA Version: {gpu_info['cuda_version']}")
            logger.info(f"Total Memory: {gpu_info['memory_total']:.2f} GB")
            logger.info(f"Free Memory: {gpu_info['memory_free']:.2f} GB")
        else:
            logger.warning("No GPU available")
    except ImportError:
        logger.warning("PyTorch not installed. Cannot check GPU availability.")
    except Exception as e:
        logger.warning(f"Error checking GPU: {str(e)}")
    
    return gpu_info


def configure_paths(
    data_dir: Optional[str] = None,
    models_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    configs_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Configure system paths and set environment variables.
    
    Args:
        data_dir: Optional path to data directory
        models_dir: Optional path to models directory
        results_dir: Optional path to results directory
        configs_dir: Optional path to configs directory
        
    Returns:
        Dictionary with configured paths
    """
    # Determine base directory (project root)
    base_dir = Path(os.getcwd())
    if not (base_dir / "src").exists():
        if (base_dir.parent / "src").exists():
            base_dir = base_dir.parent
    
    # Set default paths if not provided
    if not data_dir:
        data_dir = str(base_dir / "data")
    if not models_dir:
        models_dir = str(base_dir / "models")
    if not results_dir:
        results_dir = str(base_dir / "results")
    if not configs_dir:
        configs_dir = str(base_dir / "configs")
    
    # Create directories if they don't exist
    Path(data_dir).mkdir(exist_ok=True)
    Path(models_dir).mkdir(exist_ok=True)
    Path(results_dir).mkdir(exist_ok=True)
    Path(configs_dir).mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ["DATA_DIR"] = data_dir
    os.environ["MODELS_DIR"] = models_dir
    os.environ["RESULTS_DIR"] = results_dir
    os.environ["CONFIGS_DIR"] = configs_dir
    
    # Set environment variable to indicate we're in a notebook environment
    os.environ["IN_NOTEBOOK"] = "1"
    
    paths = {
        "base_dir": str(base_dir),
        "data_dir": data_dir,
        "models_dir": models_dir,
        "results_dir": results_dir,
        "configs_dir": configs_dir
    }
    
    logger.info("Paths configured:")
    for key, path in paths.items():
        logger.info(f"  {key}: {path}")
    
    # Create .env file for future runs
    env_file = base_dir / ".env"
    with open(env_file, "w") as f:
        for key, path in paths.items():
            if key != "base_dir":
                f.write(f"{key.upper()}={path}\n")
        f.write("USE_GPU=True\n")
    
    logger.info(f"Environment configuration saved to {env_file}")
    
    return paths


def install_dependencies(
    requirements: Optional[List[str]] = None,
    upgrade: bool = False
) -> bool:
    """
    Install required dependencies.
    
    Args:
        requirements: Optional list of requirements to install
        upgrade: Whether to upgrade existing packages
        
    Returns:
        True if installation was successful
    """
    if requirements is None:
        # Default requirements
        requirements = [
            "torch>=2.0.0",
            "numpy>=1.20.0",
            "pandas>=1.3.0",
            "matplotlib>=3.4.0",
            "tqdm>=4.60.0",
            "pyyaml>=6.0",
            "pillow>=8.2.0",
            "transformers>=4.20.0",
            "ipywidgets>=7.7.0",
            "jupyterlab>=3.0.0"
        ]
    
    logger.info("Installing dependencies...")
    
    # First check which packages need installation
    to_install = []
    for req in requirements:
        package = req.split(">=")[0].split("==")[0].split("<")[0].strip()
        try:
            pkg_resources.get_distribution(package)
            if upgrade:
                to_install.append(req)
        except pkg_resources.DistributionNotFound:
            to_install.append(req)
    
    if not to_install:
        logger.info("All dependencies already installed.")
        return True
    
    # Install required packages
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(to_install)
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        logger.info("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        return False


def test_model_loading(model_name: str = "small") -> Dict[str, Any]:
    """
    Test loading a model to verify setup.
    
    Args:
        model_name: Name of the model to test load (small/medium/large)
        
    Returns:
        Dictionary with test results
    """
    results = {
        "success": False,
        "model_loaded": False,
        "memory_used": 0,
        "load_time": 0,
        "error": None
    }
    
    logger.info(f"Testing model loading: {model_name}")
    
    try:
        import torch
        import time
        from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
        
        start_time = time.time()
        
        # Based on model_name, choose a small/medium model for testing
        if model_name == "small":
            model_id = "google/flan-t5-small"  # ~300M parameters
        elif model_name == "medium":
            model_id = "google/flan-t5-base"   # ~800M parameters
        elif model_name == "large":
            model_id = "google/flan-t5-large"  # ~3B parameters
        else:
            model_id = model_name  # Use provided model name
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_id}")
        
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Track memory before loading
        if device == "cuda":
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / (1024**3)
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        model.to(device)
        
        # Calculate stats
        load_time = time.time() - start_time
        
        if device == "cuda":
            mem_after = torch.cuda.memory_allocated() / (1024**3)
            memory_used = mem_after - mem_before
        else:
            memory_used = 0
        
        # Run a simple prediction to ensure model works
        inputs = tokenizer("Hello, I am testing the extraction system.", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=20)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Update results
        results["success"] = True
        results["model_loaded"] = True
        results["memory_used"] = memory_used
        results["load_time"] = load_time
        results["device"] = device
        
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        logger.info(f"Memory used: {memory_used:.2f} GB")
        logger.info(f"Device: {device}")
        
        # Clean up
        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        results["error"] = str(e)
    
    return results


def setup_notebook_environment(
    check_gpu: bool = True,
    install_missing: bool = True,
    verify_model: bool = True,
    model_name: str = "small"
) -> Dict[str, Any]:
    """
    Complete setup function for notebook environment.
    
    Args:
        check_gpu: Whether to check GPU availability
        install_missing: Whether to install missing packages
        verify_model: Whether to verify model loading
        model_name: Model name to verify if verify_model is True
        
    Returns:
        Dictionary with setup results
    """
    results = {
        "environment_valid": False,
        "gpu_available": False,
        "paths_configured": False,
        "dependencies_installed": False,
        "model_verified": False
    }
    
    # Step 1: Validate environment
    logger.info("Step 1: Validating environment")
    env_validation = validate_environment()
    results["environment_valid"] = all(env_validation.values())
    
    # Step 2: Check GPU availability
    if check_gpu:
        logger.info("Step 2: Checking GPU availability")
        gpu_info = check_gpu_availability()
        results["gpu_available"] = gpu_info["available"]
        results["gpu_info"] = gpu_info
    
    # Step 3: Configure paths
    logger.info("Step 3: Configuring paths")
    paths = configure_paths()
    results["paths_configured"] = True
    results["paths"] = paths
    
    # Step 4: Install missing dependencies
    if install_missing and not env_validation["required_packages"]:
        logger.info("Step 4: Installing missing dependencies")
        install_success = install_dependencies()
        results["dependencies_installed"] = install_success
    else:
        results["dependencies_installed"] = env_validation["required_packages"]
    
    # Step 5: Verify model loading
    if verify_model and results["gpu_available"]:
        logger.info("Step 5: Verifying model loading")
        model_test = test_model_loading(model_name)
        results["model_verified"] = model_test["success"]
        results["model_test"] = model_test
    
    # Final status
    logger.info("Environment setup complete.")
    logger.info(f"Environment valid: {results['environment_valid']}")
    logger.info(f"GPU available: {results['gpu_available']}")
    logger.info(f"Paths configured: {results['paths_configured']}")
    logger.info(f"Dependencies installed: {results['dependencies_installed']}")
    if verify_model:
        logger.info(f"Model verified: {results['model_verified']}")
    
    return results


def get_system_info() -> Dict[str, Any]:
    """
    Get detailed information about the system.
    
    Returns:
        Dictionary with system information
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "memory": None,
        "gpu": None,
        "packages": {}
    }
    
    # Get RAM information
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["memory"] = {
            "total": vm.total / (1024**3),
            "available": vm.available / (1024**3),
            "percent_used": vm.percent
        }
    except ImportError:
        pass
    
    # Get GPU information
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
    except ImportError:
        pass
    
    # Get package versions
    important_packages = [
        "torch", "numpy", "pandas", "matplotlib", 
        "transformers", "pillow", "tqdm", "pyyaml"
    ]
    
    for package in important_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            info["packages"][package] = version
        except:
            info["packages"][package] = "not installed"
    
    return info


if __name__ == "__main__":
    # If executed as a script, run the complete setup
    results = setup_notebook_environment()
    print(json.dumps(results, indent=2)) 